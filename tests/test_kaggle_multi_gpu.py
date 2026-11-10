"""Kaggle worker lifecycle, real CPU DDP adaptation, and sharded decode/merge."""

from dataclasses import asdict
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import threading
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "kaggle")]
import build
import make_notebook
import multi_gpu as mg
import submission_utils as ku
from common import sequence_format_for_attention
from tinytransformer import TinyTransformer, TinyTransformerConfig


def test_worker_commands_keep_ddp_training_and_single_gpu_decode_separate(monkeypatch):
    command = mg.training_command("request.json", 4)
    assert "torch.distributed.run" in command
    assert "--nproc-per-node=4" in command and "--max-restarts=0" in command
    command = mg.inference_command("request.json", 3, 4)
    assert "torch.distributed.run" not in command
    assert command[-4:] == ["--rank", "3", "--world-size", "4"]
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("TORCHELASTIC_RUN_ID", "old-run")
    env = mg.worker_environment(gpu=2)
    assert env["CUDA_VISIBLE_DEVICES"] == "2"
    assert not {"RANK", "LOCAL_RANK", "WORLD_SIZE", "TORCHELASTIC_RUN_ID"} & set(env)


def test_largest_context_smoke_selection_never_uses_query_answers(tmp_path):
    tasks = {f"task{i}": {"train": [{"input": [[1] * (i + 1)], "output": [[2]]}],
                             "test": [{"input": [[3]], "output": [[9] * 30] * 30}]}
             for i in range(6)}
    source = tmp_path / "input.json"
    source.write_text(json.dumps(tasks))
    cleaned = ku.prepare_challenges(source, tmp_path / "clean.json", smoke_test=True,
                                    smoke_task_count=4, smoke_strategy="largest")
    assert list(cleaned) == ["task5", "task4", "task3", "task2"]
    assert all("output" not in pair for task in cleaned.values() for pair in task["test"])
    tasks["task0"]["test"][0]["output"] = [[0]]
    source.write_text(json.dumps(tasks))
    assert ku.prepare_challenges(source, tmp_path / "clean2.json", smoke_test=True,
                                 smoke_task_count=4, smoke_strategy="largest") == cleaned
    shards = mg.task_shards(cleaned, 4)
    assert all(len(shard) == 1 for shard in shards)
    assert sorted(key for shard in shards for key in shard) == sorted(cleaned)


def test_empty_decode_worker_does_not_load_a_model(tmp_path, monkeypatch):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"one": {"train": [], "test": []}}))
    monkeypatch.setattr(mg, "_runtime", lambda request: pytest.fail("Empty worker allocated a model"))
    mg.decode_worker({"data_path": str(path), "run_dir": str(tmp_path)}, 3, 4)
    assert json.loads((tmp_path / "shard-3.json").read_text()) == {}


def test_only_edge_blocks_enter_optimizer_and_middle_still_passes_gradients():
    import train

    class EdgeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Linear(4, 4)
            self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(6)])
            self.head = torch.nn.Linear(4, 1)

        def forward(self, x):
            x = self.embedding(x)
            for block in self.blocks:
                x = torch.tanh(block(x))
            return self.head(x).sum()

    torch.manual_seed(42)
    model = EdgeModel()
    report = mg.configure_trainable_blocks(model, first=2, last=2)
    assert report["trainable_blocks"] == [0, 1, 4, 5]
    expected = {id(p) for i in [0, 1, 4, 5] for p in model.blocks[i].parameters()}
    actual = {id(p) for p in model.parameters() if p.requires_grad}
    assert actual == expected
    muon, adam = train._build_param_groups(model, 0.1, 0.01, 0.01, 0.01)
    assert {id(p) for group in [*muon, *adam] for p in group["params"]} == expected
    model(torch.randn(2, 4)).backward()
    assert model.blocks[0].weight.grad is not None
    assert model.blocks[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for index in [2, 3] for p in model.blocks[index].parameters())
    assert model.embedding.weight.grad is None and model.head.weight.grad is None


@pytest.mark.parametrize("first,last", [(0, 0), (-1, 2), (7, 1), (True, 1), (1, None)])
def test_invalid_trainable_block_selection_fails(first, last):
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(6)])
    with pytest.raises(ValueError):
        mg.configure_trainable_blocks(model, first, last)


def test_parallel_failure_stops_siblings_and_keeps_error_log(tmp_path):
    # Exercise actual pipes/process termination without CUDA or network access.
    jobs = [
        ("broken", [sys.executable, "-u", "-c", "print('worker error', flush=True); raise SystemExit(7)"], dict(os.environ)),
        ("sibling", [sys.executable, "-u", "-c", "import time; time.sleep(30)"], dict(os.environ)),
    ]
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="broken failed"):
        mg.run_processes(jobs, tmp_path)
    assert time.monotonic() - start < 10
    assert "worker error" in (tmp_path / "broken.log").read_text()


def test_worker_logs_and_quiet_status_are_displayed_on_calling_thread(tmp_path, monkeypatch):
    # Notebook output must be emitted during the cell by its own execution thread.
    displayed = []
    def record_print(*values, **kwargs):
        displayed.append((threading.get_ident(), " ".join(map(str, values))))

    monkeypatch.setattr(mg, "print", record_print, raising=False)
    command = [sys.executable, "-u", "-c",
               "import time; print('begin', flush=True); time.sleep(0.3); print('end', flush=True)"]
    mg.run_processes([("quiet", command, dict(os.environ))], tmp_path, progress_interval=0.05)
    assert {thread for thread, _ in displayed} == {threading.get_ident()}
    messages = [message for _, message in displayed]
    assert any("Started PID" in message for message in messages)
    quiet = next(i for i, message in enumerate(messages) if "no new worker output" in message)
    assert "does not confirm optimizer progress" in messages[quiet]
    assert quiet < next(i for i, message in enumerate(messages) if message == "[quiet] end\n")
    assert (tmp_path / "quiet.log").read_text() == "begin\nend\n"


def test_training_launch_requests_first_completed_optimizer_update_marker(tmp_path, monkeypatch):
    request = {"num_gpus": 4, "run_dir": str(tmp_path), "save_path": str(tmp_path / "final.pt"),
               "progress_interval_seconds": 15}
    path = tmp_path / "request.json"
    mg.write_json(path, request)
    def run(jobs, directory, **kwargs):
        assert len(jobs) == 1
        assert jobs[0][2]["MDLARC_REPORT_FIRST_UPDATE"] == "1"
        assert jobs[0][2]["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
        assert kwargs["progress_interval"] == 15
        Path(request["save_path"]).touch()
    monkeypatch.setattr(mg, "run_processes", run)
    mg.launch_training(path)


def test_worker_logs_every_update_by_default_and_honors_explicit_override(tmp_path):
    request = {"defaults": make_notebook.DEFAULTS, "data_path": str(tmp_path / "data.json"),
               "save_path": str(tmp_path / "final.pt"),
               "finetune": dict(epochs=90, batch_size=1, accumulation=10, max_augments=80,
                                num_workers=0, normuon_lr=0.0016, adamw_lr=0.00028)}
    cfg = mg._config(request)
    assert cfg.train_log_mode == "step" and cfg.log_location == "both"
    assert cfg.epochs == 90 and cfg.gradient_accumulation_steps == 10
    request["train_log_mode"] = "epoch"
    assert mg._config(request).train_log_mode == "epoch"
    request["train_log_mode"] = "typo"
    with pytest.raises(ValueError, match="log mode"):
        mg._config(request)


def _cpu_training_rank(rank, folder):
    import train
    import distributed_training as dt
    folder = Path(folder)
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=(folder / "rendezvous").as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        build.resolve_device = lambda value: torch.device("cpu")
        train._build_optimizer = lambda args, model, *rest: torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=0.001)
        request = json.loads((folder / "request.json").read_text())
        mg.train_worker(request, device_override="cpu")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
def test_two_rank_adaptation_checkpoint_and_disjoint_decode_merge(tmp_path, monkeypatch):
    import evaluate

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(build, "resolve_device", lambda value: torch.device("cpu"))
    monkeypatch.setattr(torch, "compile", lambda function, **kwargs: function)
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    monkeypatch.setattr(evaluate, "_compiled_grid_update", evaluate._grid_update)
    monkeypatch.setattr(evaluate, "DEFAULT_MAX_NEW_TOKENS", 8)
    torch.manual_seed(42)
    model = TinyTransformer(TinyTransformerConfig(
        d_model=24, n_heads=4, d_ff=48, n_layers=6, num_examples=7,
        num_task_embedding_latents=5, task_embeddings=False,
        attention_pattern="pair_group", split_target_demo_attention=True,
        attention_dropout=0.0, dropout=0.0, embedding_dropout=0.0, n_predict=4,
    ))
    checkpoint = {"config": asdict(model.config), "model_state": model.state_dict(),
                  "sequence_format": sequence_format_for_attention("pair_group"),
                  "augmentation_sampling_mode": "context_palette", "epoch": 330,
                  "task_ids": [f"old/{i}" for i in range(7)],
                  "embedding_task_ids": [f"old/{i}" for i in range(7)]}
    source = tmp_path / "source.pt"
    torch.save(checkpoint, source)
    tasks = {f"hidden{i}": {"train": [
        {"input": [[i]], "output": [[i + 1]]}, {"input": [[i + 2]], "output": [[i + 3]]},
        {"input": [[i + 1, i]], "output": [[i, i + 1]]}], "test": [{"input": [[i + 4]]}]}
        for i in range(4)}
    data = tmp_path / "data.json"
    data.write_text(json.dumps(tasks))
    run_dir = tmp_path / "runs/cpu_run"
    request = {
        "working": str(tmp_path), "run_dir": str(run_dir), "num_gpus": 2,
        "checkpoint_path": str(source), "checkpoint_identity": {
            "epoch": 330, "augmentation_sampling_mode": "context_palette",
            "config": {"d_model": 24, "attention_pattern": "pair_group"}},
        "data_path": str(data), "save_path": str(run_dir / "finetuned.pt"),
        "inference_checkpoint": str(run_dir / "inference.pt"),
        "attention_backend": "native", "enable_compile": False, "smoke_test": True,
        "max_finetune_hours": 8.0, "eval_augments": 2, "eval_batch_size": 1,
        "trainable_blocks": {"first": 2, "last": 2},
        "defaults": dict(make_notebook.DEFAULTS, augmentation_sampling_mode="context_palette",
                         training_demo_schedule="all", inference_demo_policy="leave_one_out"),
        "finetune": dict(epochs=2, batch_size=1, accumulation=1, max_augments=2,
                         num_workers=0, normuon_lr=0.0016, adamw_lr=0.00028),
    }
    mg.write_json(tmp_path / "request.json", request)
    mp.spawn(_cpu_training_rank, args=(str(tmp_path),), nprocs=2, join=True)
    saved = torch.load(request["save_path"], map_location="cpu", weights_only=False)
    assert saved["epoch"] == 2
    assert saved["distributed_world_size"] == 2
    assert saved["embedding_task_ids"] == sorted(tasks)
    assert not torch.equal(saved["model_state"]["blocks.0.attention.qkv_proj.weight"],
                           checkpoint["model_state"]["blocks.0.attention.qkv_proj.weight"])
    changed = []
    for name, before in checkpoint["model_state"].items():
        if name == "task_factor_embedding.weight":
            continue  # Fresh task rows belong to runtime tasks and are frozen during updates.
        if any(name.startswith(f"blocks.{index}.") for index in (0, 1, 4, 5)):
            if not torch.equal(before, saved["model_state"][name]):
                changed.append(name)
        else:
            assert torch.equal(before, saved["model_state"][name]), name
    assert all(any(name.startswith(f"blocks.{index}.") for name in changed) for index in (0, 1, 4, 5))
    mg.make_inference_checkpoint(request["save_path"], request["inference_checkpoint"], 2)
    lean = torch.load(request["inference_checkpoint"], map_location="cpu", weights_only=False)
    assert not {"optimizer_state", "scheduler_state", "rng_state_by_rank"} & set(lean)
    for key, value in saved["model_state"].items():
        assert torch.equal(value, lean["model_state"][key]), key
    for rank in range(2):
        mg.decode_worker(request, rank, 2, device_override="cpu")
        info = json.loads((run_dir / f"train-{rank}.json").read_text())
        assert info["effective_batch_size"] == 2 and info["epochs"] == 2
        assert info["trainable_blocks"] == [0, 1, 4, 5]
    output = tmp_path / "submission.json"
    mg.merge_shards(request, tasks, output)
    ku.validate_submission(json.loads(output.read_text()), tasks)
    before = output.read_bytes()
    wrong = json.loads((run_dir / "shard-0.json").read_text())
    wrong.update(json.loads((run_dir / "shard-1.json").read_text()))
    mg.write_json(run_dir / "shard-0.json", wrong)
    with pytest.raises(ValueError, match="task ids"):
        mg.merge_shards(request, tasks, output)
    assert output.read_bytes() == before
