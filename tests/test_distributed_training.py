"""DDP data coverage, global token losses, accumulation, and portable saves."""
from argparse import Namespace
from datetime import timedelta
import functools
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import build
import distributed_training as dt
import launch_run
import train
from common import ARCExampleDataset, collate_examples, capture_rng_state, restore_rng_state
from tinytransformer import TinyTransformer, TinyTransformerConfig
from test_family_data import corpus, inventory, fd
import variations


class _Batches:
    preserve_all_pairs = True
    drop_last = False

    def __init__(self, count=13, batch_size=4):
        self.count, self.batch_size, self._epoch = count, batch_size, 0

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = torch.randperm(self.count, generator=torch.Generator().manual_seed(self._epoch)).tolist()
        return iter([order[i:i+self.batch_size] for i in range(0, self.count, self.batch_size)])


def test_remote_single_node_nccl_uses_local_bootstrap_and_keeps_p2p():
    env = {"MDLARC_REMOTE": "1"}
    dt.configure_single_node_nccl(env)
    assert {key: env[key] for key in ("NCCL_NET", "NCCL_NET_PLUGIN", "NCCL_IB_DISABLE", "NCCL_SOCKET_IFNAME")} == {
        "NCCL_NET": "Socket", "NCCL_NET_PLUGIN": "none", "NCCL_IB_DISABLE": "1", "NCCL_SOCKET_IFNAME": "lo"}
    assert "NCCL_P2P_DISABLE" not in env
    assert env["NCCL_DEBUG"] == "INFO"


def test_single_node_nccl_preserves_explicit_transport_and_local_environment():
    explicit = {"MDLARC_REMOTE": "1", "NCCL_NET": "IB", "NCCL_DEBUG": "WARN"}
    dt.configure_single_node_nccl(explicit)
    assert explicit["NCCL_NET"] == "IB" and explicit["NCCL_DEBUG"] == "WARN"
    assert "NCCL_IB_DISABLE" not in explicit and "NCCL_NET_PLUGIN" not in explicit
    local = {"WORLD_SIZE": "8"}
    dt.configure_single_node_nccl(local)
    assert local == {"WORLD_SIZE": "8"}


@pytest.mark.parametrize("world_size", [2, 3, 4])
@pytest.mark.parametrize("count", [4, 5, 7, 13, 16, 17])
@pytest.mark.parametrize("per_gpu_batch", [1, 2])
def test_shards_preserve_every_pair_once_with_equal_nonempty_steps(world_size, count, per_gpu_batch):
    samplers = [dt.DistributedBatchSampler(_Batches(count, world_size * per_gpu_batch), rank=r, world_size=world_size)
                for r in range(world_size)]
    for epoch in [0, 1, 9, 0]:
        for sampler in samplers:
            sampler.set_epoch(epoch)
        planned_lengths = [len(sampler) for sampler in samplers]
        shards = [list(sampler) for sampler in samplers]
        assert planned_lengths == [len(shard) for shard in shards]
        assert len({len(shard) for shard in shards}) == 1
        assert all(batch for shard in shards for batch in shard)
        assert sorted(i for shard in shards for batch in shard for i in batch) == list(range(count))
        assert all(len(sampler) == len(shard) for sampler, shard in zip(samplers, shards))


def test_too_few_pairs_fails_instead_of_repeating_them():
    sampler = dt.DistributedBatchSampler(_Batches(1), rank=0, world_size=2)
    with pytest.raises(ValueError, match="one selected training pair per rank"):
        len(sampler)


@pytest.mark.parametrize("workers", [0, 1])
def test_family_two_demo_context_palette_shards_replay_original_visits(corpus, monkeypatch, workers):
    catalog, payload = inventory(corpus)
    identity = fd.package_catalog(catalog, payload, corpus.root / "assets_tmp/families.zip")
    identity["archive_path"] = "assets_tmp/families.zip"
    config = fd.normalize_config(corpus.raw, corpus.root)
    config["pretrain_data"]["catalog"] = identity
    output = corpus.root / "data"
    fd.build_family_dataset(config, output, root=corpus.root)
    monkeypatch.setattr(variations, "ROOT", corpus.root)
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    args = Namespace(seed=42, device="cpu", checkpoint_path=None,
        data_path=output / "challenges.json", variation_index_path=output / "variation_index.json",
        augmentation_sampling_mode="context_palette", attention_pattern="pair_group",
        training_demo_schedule="two", enable_aug=True, max_augments=80,
        batch_size=2, epochs=5, num_workers=workers, task_embeddings=False,
        d_model=24, n_heads=4, d_ff=48, n_layers=1, num_task_embedding_latents=2,
        dropout=0., embedding_dropout=0., attention_dropout=0., n_predict=1)
    monkeypatch.setattr(dt, "world_size", lambda: 2)
    loaders = []
    for current_rank in range(2):
        monkeypatch.setattr(dt, "rank", lambda r=current_rank: r)
        _, dataset, loader, *_ = build.build_model_and_data(args)
        if workers:
            assert loader.multiprocessing_context.get_start_method() == "spawn"
        loader.augmentor.prepare_training_schedule(loader.batch_sampler)
        loaders.append(loader)
    for epoch in [0, 1, 4, 0]:
        indices = []
        for loader in loaders:
            loader.batch_sampler.set_epoch(epoch)
            loader.augmentor.set_epoch(epoch)
            local = list(loader.batch_sampler)
            indices.extend(i for batch in local for i in batch)
            for batch, selected in zip(loader, local):
                expected = collate_examples([dataset[i] for i in selected], dataset=dataset,
                                             augment_selector=loader.augmentor.select_for_example)
                torch.testing.assert_close(batch["input_ids"], expected["input_ids"])
        source = loaders[0].batch_sampler.sampler
        assert sorted(indices) == sorted(source._indices_for_epoch(epoch))
        assert len(loaders[0]) == len(loaders[1])


@pytest.mark.parametrize("value", [0, -1, True, 2.0, "2"])
def test_invalid_ddp_config_fails_at_preview(value):
    with pytest.raises(ValueError, match="positive integer"):
        launch_run.build_plan({"distributed": {"nproc_per_node": value}}, Path("spec.json"))


def test_nccl_binds_local_gpu_before_init_and_cleans_up_on_failure(monkeypatch):
    calls = []
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "set_device", lambda rank: calls.append(("device", rank)))
    monkeypatch.setattr(dist, "init_process_group", lambda backend, **kwargs: calls.append(("init", backend)))
    control = object()
    monkeypatch.setattr(dist, "new_group", lambda **kwargs: calls.append(("control", kwargs["backend"])) or control)
    monkeypatch.setattr(dist, "destroy_process_group", lambda group=None: calls.append(("destroy_control",) if group is control else ("destroy",)))
    with pytest.raises(RuntimeError, match="training failed"):
        with dt.training_process(2):
            raise RuntimeError("training failed")
    assert calls == [("device", 1), ("init", "nccl"), ("control", "gloo"), ("destroy_control",), ("destroy",)]
    assert dt._checkpoint_group is None


def test_checkpoint_rng_and_barrier_use_cpu_control_group(monkeypatch):
    control = object()
    monkeypatch.setattr(dt, "_checkpoint_group", control)
    monkeypatch.setattr(dt, "active", lambda: True)
    monkeypatch.setattr(dt, "rank", lambda: 0)
    monkeypatch.setattr(dt, "world_size", lambda: 2)
    calls = []
    def gather(state, states, *, dst, group):
        assert state == {"cpu_rng": 42} and dst == 0 and group is control
        states[:] = [state, {"cpu_rng": 43}]
        calls.append("gather")
    monkeypatch.setattr(dist, "gather_object", gather)
    monkeypatch.setattr(dist, "barrier", lambda *, group: calls.append(group))
    assert dt.gather_checkpoint_rng_state({"cpu_rng": 42}) == [{"cpu_rng": 42}, {"cpu_rng": 43}]
    dt.checkpoint_barrier()
    assert calls == ["gather", control]


def test_checkpoint_control_rejects_implicit_nccl_object_transfer(monkeypatch):
    monkeypatch.setattr(dt, "_checkpoint_group", None)
    monkeypatch.setattr(dt, "active", lambda: True)
    monkeypatch.setattr(dt, "world_size", lambda: 2)
    monkeypatch.setattr(dt, "rank", lambda: 0)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    with pytest.raises(RuntimeError, match="CPU control group"):
        dt.gather_checkpoint_rng_state({})


def test_ddp_cuda_rng_restores_on_another_local_device_for_single_gpu_eval(monkeypatch):
    monkeypatch.setattr(dt, "world_size", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    saved = torch.tensor([1, 2, 3], dtype=torch.uint8)
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda device: saved)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: pytest.fail("Touched every GPU"))
    state = capture_rng_state(torch.device("cuda:1"))
    assert "cuda_local" in state and "cuda" not in state
    restored = []
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda rng, device: restored.append((rng, device)))
    restore_rng_state(state, torch.device("cuda:0"))
    assert restored[0][0] is saved and restored[0][1] == torch.device("cuda:0")


def test_ddp_training_uses_torchrun_but_evaluation_stays_single_process(monkeypatch):
    calls = []
    monkeypatch.setattr(launch_run.subprocess, "run", lambda cmd, **kwargs: calls.append(cmd))
    plan = {"distributed": {"nproc_per_node": 2}, "spec_path": Path("spec.json")}
    launch_run.run_training_stage_isolated(plan, "pretrain")
    launch_run.run_eval_stage_isolated(plan, "final")
    assert "torch.distributed.run" in calls[0]
    assert "--nproc-per-node=2" in calls[0] and "--max-restarts=0" in calls[0]
    assert calls[0][-2:] == ["--train-stage", "pretrain"]
    assert "torch.distributed.run" not in calls[1]


def _model(dataset):
    torch.manual_seed(42)
    return TinyTransformer(TinyTransformerConfig(
        d_model=24, n_heads=4, d_ff=48, n_layers=1, max_seq_len=dataset.max_seq_len,
        num_examples=dataset.num_examples, num_task_embedding_latents=2,
        task_embeddings=False, dihedral_embeddings=False,
        attention_pattern="pair_group", split_target_demo_attention=True,
        dropout=0., embedding_dropout=0., attention_dropout=0., n_predict=4, mtp_weight=.3,
    ))


def _dataset(path):
    return ARCExampleDataset(path, seed=42, training_demo_schedule="two", attention_pattern="pair_group")


def _loader(dataset, sampler):
    return DataLoader(dataset, batch_sampler=sampler,
                      collate_fn=functools.partial(collate_examples, dataset=dataset))


def _optimizer(model, name="normuon"):
    # The production hybrid is also replicated after DDP gradient averaging.
    if name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
    else:
        optimizer = train.SingleDeviceNorMuonWithAuxAdam([
            {"params": [p for p in model.parameters() if p.requires_grad and p.ndim >= 2],
             "use_muon": True, "lr": .003, "weight_decay": 0.},
            {"params": [p for p in model.parameters() if p.requires_grad and p.ndim < 2],
             "use_muon": False, "lr": .001, "weight_decay": 0.},
        ])
    optimizer.test_gradients = []
    def record_gradients(opt, args, kwargs):
        opt.test_gradients.append([p.grad.clone() for group in opt.param_groups
                                   for p in group["params"] if p.grad is not None])
    optimizer.register_step_pre_hook(record_gradients)
    return optimizer


def test_finetune_resume_keeps_historical_rows_but_only_trains_subset(tmp_path, monkeypatch):
    path = tmp_path / "subset.json"
    path.write_text(json.dumps({"present": {"train": [
        {"input": [[1]], "output": [[2]]}, {"input": [[3]], "output": [[4]]}]}}))
    dataset = ARCExampleDataset(path, task_whitelist=["historical", "present"],
        allow_missing_tasks=True, attention_pattern="pair_group", training_demo_schedule="all")
    model = _model(dataset)
    checkpoint = tmp_path / "checkpoint.pt"
    train.maybe_save_model(model, dataset, path, checkpoint, epoch=130, global_step=390,
                          rng_state=capture_rng_state(torch.device("cpu")))
    plan = launch_run.build_plan({"run_name": "resume-subset", "finetune_only": True,
        "finetune_resume": True, "checkpoint": str(checkpoint), "dataset": {"style": "finetune"},
        "defaults": {"device": "cpu", "num_workers": 0, "enable_aug": False, "FA": 2},
        "finetunes": {"validation": {"epochs": 330}}}, tmp_path / "spec.json")
    stage = plan["finetunes"]["validation"]
    stage["data_path"] = path
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    resumed, resumed_data, *_ = build.build_model_and_data(Namespace(**stage), build_dataloader=False)
    assert resumed_data.task_ids == dataset.task_ids
    assert resumed_data.embedding_task_ids == dataset.embedding_task_ids
    assert len(resumed_data) == 2
    assert all(example.task_id == "present" for example in resumed_data.examples)
    assert resumed._loaded_checkpoint["epoch"] == 130
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)


def _ddp_worker(rank, directory, optimizer_name):
    torch.set_num_threads(1)
    directory = Path(directory)
    dist.init_process_group("gloo", init_method=(directory / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        dataset = _dataset(directory / "tasks.json")
        model = _model(dataset)
        wrapper = DistributedDataParallel(model, find_unused_parameters=True, broadcast_buffers=False,
                                           gradient_as_bucket_view=True)
        torch.manual_seed(1729 + rank)
        sampler = dt.DistributedBatchSampler(_Batches(len(dataset)), rank=rank, world_size=2)
        optimizer = _optimizer(model, optimizer_name)
        scheduler = train._FractionalLambdaLR(optimizer, lambda epoch: 1.)
        for epoch in range(2):
            sampler.set_epoch(epoch)
            train.train_one_epoch(wrapper, _loader(dataset, sampler), optimizer, torch.device("cpu"),
                                  grad_clip=1., gradient_accumulation_steps=3, epoch=epoch,
                                  total_epochs=2, train_log_mode="step", log_location="file",
                                  log_handle=None)
        train._save_checkpoint(wrapper, dataset, directory / "tasks.json", directory / "checkpoint.pt",
                               optimizer=optimizer, global_step=8, epoch=2, scheduler=scheduler,
                               device=torch.device("cpu"))
        torch.save(model.state_dict(), directory / f"rank{rank}.pt")
        torch.save(optimizer.test_gradients[0], directory / f"grad{rank}.pt")
        expected_rng = torch.rand(4)
        build.resolve_device = lambda _: torch.device("cpu")
        resume_args = Namespace(seed=42, device="cpu", checkpoint_path=directory / "checkpoint.pt",
                                data_path=directory / "tasks.json", enable_aug=False,
                                training_demo_schedule="two")
        resumed, resumed_dataset, *_ = build.build_model_and_data(resume_args, build_dataloader=False)
        torch.testing.assert_close(torch.rand(4), expected_rng, rtol=0, atol=0)
        resumed_optimizer = _optimizer(resumed, optimizer_name)
        train._load_optimizer_state(resumed_optimizer, resumed._loaded_checkpoint["optimizer_state"],
                                    torch.device("cpu"))
        if optimizer_name == "normuon":
            resumed_optimizer.configure_muon_schedule(total_steps=None, start_step=optimizer._muon_step)
        resumed_wrapper = DistributedDataParallel(resumed, find_unused_parameters=True, broadcast_buffers=False)
        sampler.set_epoch(2)
        for current_model, current_optimizer, current_dataset in [
            (wrapper, optimizer, dataset), (resumed_wrapper, resumed_optimizer, resumed_dataset),
        ]:
            train.train_one_epoch(current_model, _loader(current_dataset, sampler), current_optimizer,
                                  torch.device("cpu"), grad_clip=1., gradient_accumulation_steps=3,
                                  epoch=2, total_epochs=3, train_log_mode="never", log_location="none")
        for actual, expected in zip(resumed.parameters(), model.parameters(), strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
@pytest.mark.parametrize("optimizer_name", ["adamw", "normuon"])
def test_two_process_training_matches_global_batch_with_mtp_uneven_tail_and_resume(tmp_path, monkeypatch, optimizer_name):
    # Thirteen pairs: tail of one is rebalanced; unequal output lengths give
    # different CE and MTP denominators on each rank, including empty heads.
    pairs = [{"input": [[i % 10]], "output": [[(i + 1) % 10] * (i % 4 + 1)]}
             for i in range(13)]
    (tmp_path / "tasks.json").write_text(json.dumps({"task": {"train": pairs}}))
    torch.set_num_threads(1)
    dataset = _dataset(tmp_path / "tasks.json")
    reference = _model(dataset)
    sampler = dt.DistributedBatchSampler(_Batches(len(dataset)), rank=0, world_size=2)
    # Reconstruct the exact global microbatches, before splitting by rank.
    optimizer = _optimizer(reference, optimizer_name)
    for epoch in range(2):
        sampler.set_epoch(epoch)
        class GlobalBatches:
            preserve_all_pairs = True
            def __iter__(self): return iter(sampler._batches())
            def __len__(self): return len(sampler)
        train.train_one_epoch(reference, _loader(dataset, GlobalBatches()), optimizer, torch.device("cpu"),
                              grad_clip=1., gradient_accumulation_steps=3, epoch=epoch,
                              total_epochs=2, train_log_mode="step", log_location="none")
    mp.spawn(_ddp_worker, args=(str(tmp_path), optimizer_name), nprocs=2, join=True)
    checkpoint = torch.load(tmp_path / "checkpoint.pt", weights_only=False)
    assert checkpoint["distributed_world_size"] == len(checkpoint["rng_state_by_rank"]) == 2
    assert not any(key.startswith("module.") for key in checkpoint["model_state"])
    rank0 = torch.load(tmp_path / "rank0.pt", weights_only=True)
    rank1 = torch.load(tmp_path / "rank1.pt", weights_only=True)
    for key, expected in reference.state_dict().items():
        torch.testing.assert_close(rank0[key], rank1[key], rtol=0, atol=0)
        # NorMuon's BF16 orthogonalization amplifies tiny summation-order
        # changes. Test its gradients directly and replica/resume equality;
        # AdamW also permits a tight final-weight reference comparison.
        if optimizer_name == "adamw":
            torch.testing.assert_close(rank0[key], expected, rtol=3e-4, atol=3e-5)
    gradients = torch.load(tmp_path / "grad0.pt", weights_only=True)
    for actual, expected in zip(gradients, optimizer.test_gradients[0], strict=True):
        torch.testing.assert_close(actual, expected, rtol=3e-4, atol=1e-6)
    # A DDP save must load through ordinary single-process evaluation.
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    args = Namespace(seed=42, device="cpu", checkpoint_path=tmp_path / "checkpoint.pt",
                     data_path=tmp_path / "tasks.json", enable_aug=False)
    evaluated, *_ = build.build_model_and_data(args, is_eval=True, eval_splits=["train"], build_dataloader=False)
    for key in rank0:
        torch.testing.assert_close(evaluated.state_dict()[key], rank0[key])
    # A changed topology cannot silently pretend to be an exact resume.
    with pytest.raises(ValueError, match="world size"):
        build.build_model_and_data(args, build_dataloader=False)
