import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import launch_run  # noqa: E402
import build  # noqa: E402
import train  # noqa: E402
import variations  # noqa: E402
from scripts.run_on_colab import spec_requires_task_variations  # noqa: E402


def _variation_spec(style: str) -> dict:
    return {
        "dataset": {
            "dataset": "arc2",
            "style": style,
            "pretrain_data": {
                "kind": "task_variations",
                "snapshot": "task-variations-v1",
                "sources": ["arc2_train", "conceptarc", "arc1_filtered"],
                "original_sources": [],
                "sampling": {"variations_per_parent_per_epoch": 1},
            },
        },
        "enable_finetune": style == "finetune",
    }


def test_variation_specs_are_additive_and_keep_eval_support_stage() -> None:
    legacy = launch_run.build_plan({"enable_finetune": False}, Path("legacy.json"))["dataset"]
    assert "pretrain_data" not in legacy
    assert legacy["add_conceptarc"] and legacy["add_filtered"]

    finetune = launch_run.build_plan(_variation_spec("finetune"), Path("finetune.json"))
    joint = launch_run.build_plan(_variation_spec("pretrain"), Path("joint.json"))
    assert finetune["dataset"]["pretrain_data"]["eval_support_repeats_per_epoch"] == 0
    assert joint["dataset"]["pretrain_data"]["eval_support_repeats_per_epoch"] == 1
    assert not finetune["dataset"]["add_conceptarc"]
    assert finetune["assets"]["evaluation_data_path"].name == "priv_challenges.json"
    assert joint["assets"]["evaluation_data_path"].name == "priv_challenges.json"
    assert finetune["pretrain"]["variation_index_path"].name == "variation_index.json"
    assert "variation_index_path" not in finetune["finetunes"]["default"]


def test_variation_spec_rejects_eval_variations_and_legacy_addons() -> None:
    spec = _variation_spec("finetune")
    spec["dataset"]["pretrain_data"]["sources"] = ["arc2_eval"]
    with pytest.raises(ValueError, match="unsupported sources"):
        launch_run.build_plan(spec, Path("bad.json"))
    spec = _variation_spec("finetune")
    spec["dataset"]["add_rearc"] = 1000
    with pytest.raises(ValueError, match="Legacy dataset add-ons"):
        launch_run.build_plan(spec, Path("bad.json"))


@pytest.mark.parametrize("changes", [
    {"embedding_scope": "pair"},
    {"deduplicate_variation_inputs": 0},
    {"variations_per_parent_per_epoch": 43},
])
def test_variation_spec_rejects_invalid_sampling_policies(changes):
    spec = _variation_spec("finetune")
    spec["dataset"]["pretrain_data"]["sampling"].update(changes)
    with pytest.raises(ValueError):
        launch_run.build_plan(spec, Path("bad.json"))


def test_big_batch_mixed_spec_uses_all_variations_and_keeps_finetune_separate():
    path = ROOT / "specs/arc2_variations_originals_big_batch_finetune.json"
    plan = launch_run.build_plan(launch_run.load_spec(path), path)
    data = plan["dataset"]["pretrain_data"]
    assert data["original_sources"] == data["sources"] == ["arc2_train", "conceptarc", "arc1_filtered"]
    assert data["sampling"]["variations_per_parent_per_epoch"] == 42
    assert data["sampling"]["embedding_scope"] == "variation"
    assert data["sampling"]["deduplicate_variation_inputs"] is False
    assert data["eval_support_repeats_per_epoch"] == 0
    pretrain = plan["pretrain"]
    assert (pretrain["batch_size"], pretrain["epochs"]) == (96, 90)
    assert (pretrain["normuon_lr"], pretrain["adamw_lr"]) == (0.003, 0.00054)
    assert "qk_norm" not in pretrain
    assert pretrain["branch_norm"] is None and pretrain["residual_init"] is None
    assert pretrain["augmentation_sampling_mode"] == "single_grid_prefix"
    assert pretrain["mtp_decay"] == [0.8, 0.9, 1.0]
    finetune = plan["finetunes"]["arc2_eval"]
    assert (finetune["batch_size"], finetune["epochs"]) == (8, 330)
    assert (finetune["normuon_lr"], finetune["adamw_lr"]) == (0.0003, 0.003)
    assert plan["evals"]["final"]["inference_speculative"] is True
    assert plan["evals"]["final"]["inference_temperature"] in (None, 0.0)


@pytest.mark.parametrize('schedule', ['half_or_more', 'two'])
def test_legacy_variations_honor_demo_schedule_and_cap(tmp_path, monkeypatch, schedule):
    monkeypatch.setattr(variations, "ROOT", tmp_path)
    pair = {"input": [[1]], "output": [[2]]}
    (tmp_path / "variant.json").write_text(json.dumps({"train": [pair] * 9, "test": [pair]}))
    path = tmp_path / "parents.json"
    path.write_text(json.dumps({"parent": {"train": [pair] * 10}}))
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"snapshot": "task-variations-v1", "sampling": {},
        "eval_support_task_ids": [], "parents": {"parent": [
            {"seed": 0, "path": "variant.json", "pair_count": 10}]}}))
    dataset = variations.VariationTrainingDataset(json_path=path, index_path=index, seed=42,
        max_augments=80, attention_pattern="pair_group", training_demo_schedule=schedule)
    assert dataset.training_demo_schedule == schedule
    sampler = variations.VariationBatchSampler(dataset, 4, seed=42,
        variations_per_parent_per_epoch=1, eval_support_repeats_per_epoch=0)
    for epoch in range(8):
        totals = {source: Counter({i: 0 for i in range(10)}) for source in ("regular", "variation")}
        for index in sampler._indices_for_epoch(epoch):
            ex = dataset[index]
            visible = dataset.demonstration_indices(ex, ex.demo_visit_override)
            assert ex.pair_index not in visible
            assert len(visible) == 2 if schedule == 'two' else 5 <= len(visible) <= 6
            totals[index[0]].update(visible)
        assert all(len(set(counts.values())) == 1 for counts in totals.values())


def test_all_variations_have_independent_embeddings_and_family_initialization(tmp_path, monkeypatch):
    monkeypatch.setattr(variations, "ROOT", tmp_path)
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    pair = {"input": [[1]], "output": [[2]]}
    parents = {}
    for parent in ("arc2_train/first", "arc2_train/second"):
        variants = []
        for seed in range(42):
            path = f"{parent.rsplit('/', 1)[-1]}seed{seed}.json"
            (tmp_path / path).write_text(json.dumps({"train": [pair], "test": [pair]}))
            variants.append({"seed": seed, "path": path, "pair_count": 2})
        parents[parent] = variants
    data_path = tmp_path / "challenges.json"
    data_path.write_text(json.dumps({parent: {"train": [pair, pair]} for parent in parents}))
    index_path = tmp_path / "variation_index.json"
    index_path.write_text(json.dumps({
        "snapshot": "task-variations-v1", "archive_sha256": "verified",
        "sources": ["arc2_train"], "original_sources": ["arc2_train"],
        "sampling": {"variations_per_parent_per_epoch": 42, "embedding_scope": "variation",
                     "deduplicate_variation_inputs": False},
        "eval_support_repeats_per_epoch": 0, "eval_support_task_ids": [], "parents": parents,
    }))
    args = SimpleNamespace(
        seed=42, device="cpu", checkpoint_path=None, data_path=data_path,
        variation_index_path=index_path, batch_size=8, num_workers=0,
        enable_aug=False, max_augments=80, d_model=32, n_heads=4, d_ff=64,
        n_layers=1, num_task_embedding_latents=4, n_predict=1,
        dropout=0.0, embedding_dropout=0.0, attention_dropout=0.0, FA=2,
    )
    model, dataset, _, _, _ = build.build_model_and_data(args, build_dataloader=False)
    assert dataset.num_examples == 86
    sampler = variations.VariationBatchSampler(
        dataset, 8, seed=42, variations_per_parent_per_epoch=42, eval_support_repeats_per_epoch=0,
    )
    epoch0, epoch1 = (sampler._indices_for_epoch(epoch) for epoch in (0, 1))
    assert len(epoch0) == len(epoch1) == 172  # 84 two-pair variations + four original pairs.
    assert len(sampler) == len(list(sampler)) == 22
    sampler.drop_last = True
    sampler.minibatch_multiple = 2
    assert len(sampler) == len(list(sampler)) == 20
    sampler.drop_last = False
    identity = lambda item: item[:4] if item[0] == "variation" else item[:2]
    assert {identity(item) for item in epoch0} == {identity(item) for item in epoch1}
    assert [identity(item) for item in epoch0] != [identity(item) for item in epoch1]
    variant_rows = set()
    for index in (item for item in epoch0 if item[0] == "variation"):
        example = dataset[index]
        assert example.task_id == f"{index[1]}/variation/seed{index[2]}"
        variant_rows.add(example.task_embedding_ids[0])
    assert len(variant_rows) == 84
    parent_rows = [dataset.get_task_embedding_ids(parent)[0] for parent in parents]
    weight = model.task_factor_embedding.weight
    assert not torch.equal(weight[parent_rows[0]], weight[parent_rows[1]])
    for parent_row, rows in dataset.variation_embedding_families.items():
        torch.testing.assert_close(weight[rows], weight[parent_row].expand(42, -1))
        ids = torch.tensor([parent_row, *rows])
        embeds = model._lookup_example_embeds(ids)
        torch.testing.assert_close(embeds, embeds[0].expand_as(embeds))
        value_embeds = model._lookup_value_embedding(0, ids)
        torch.testing.assert_close(value_embeds, value_embeds[0].expand_as(value_embeds))

    # Updating one variant must not update its parent or sibling rows.
    before = weight.detach().clone()
    row = next(iter(variant_rows))
    optimizer = torch.optim.SGD([weight], lr=0.1)
    model._lookup_example_embeds(torch.tensor([row])).square().sum().backward()
    optimizer.step()
    assert not torch.equal(weight[row], before[row])
    unchanged_rows = [index for index in range(86) if index != row]
    torch.testing.assert_close(weight[unchanged_rows], before[unchanged_rows])

    checkpoint_path = tmp_path / "model.pt"
    train.maybe_save_model(model, dataset, data_path, checkpoint_path)
    args.checkpoint_path = checkpoint_path
    args.checkpoint_mode = "resume"
    resumed, resumed_dataset, _, _, _ = build.build_model_and_data(args, build_dataloader=False)
    assert resumed_dataset.embedding_task_ids == dataset.embedding_task_ids
    torch.testing.assert_close(resumed.task_factor_embedding.weight, weight)
    changed = json.loads(index_path.read_text())
    changed["sampling"]["deduplicate_variation_inputs"] = True
    index_path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="does not match"):
        build.build_model_and_data(args, build_dataloader=False)


def test_colab_auto_stages_only_training_specs_that_need_variations() -> None:
    spec = _variation_spec("finetune")
    assert spec_requires_task_variations(spec)
    assert not spec_requires_task_variations({"dataset": {"dataset": "arc2"}})
    spec.update(eval_only=True, skip_data_build=True)
    assert not spec_requires_task_variations(spec)


@pytest.mark.parametrize("with_split", [False, True])
def test_sampler_rotates_variations_and_only_uses_eval_train_pairs(tmp_path, monkeypatch, with_split) -> None:
    monkeypatch.setattr(variations, "ROOT", tmp_path)
    parent_id = "arc2_train/source"
    pair = {"input": [[1]], "output": [[2]]}
    for seed in range(2):
        (tmp_path / f"seed{seed}.json").write_text(json.dumps({
            "train": [pair] * (seed + 1), "test": [pair],
        }))
    (tmp_path / "challenges.json").write_text(json.dumps({
        parent_id: {"train": []},
        "eval": {"train": [pair], "test": [{"input": [[3]]}]},
    }))
    (tmp_path / "variation_index.json").write_text(json.dumps({
        "snapshot": "task-variations-v1",
        "archive_sha256": "x",
        "sources": ["arc2_train"],
        "original_sources": [],
        "sampling": {"variations_per_parent_per_epoch": 1},
        "eval_support_repeats_per_epoch": 2,
        "eval_support_task_ids": ["eval"],
        "parents": {parent_id: [
            {"seed": seed, "path": f"seed{seed}.json", "pair_count": seed + 2}
            for seed in range(2)
        ]},
        **({"parent_split": {"manifest": "split.json", "sha256": "fixed-split"}} if with_split else {}),
    }))
    dataset = variations.VariationTrainingDataset(
        json_path=tmp_path / "challenges.json",
        index_path=tmp_path / "variation_index.json",
        seed=42,
        max_augments=2,
    )
    sampler = variations.VariationBatchSampler(
        dataset, 2, seed=42,
        variations_per_parent_per_epoch=1,
        eval_support_repeats_per_epoch=2,
    )
    epoch0 = sampler._indices_for_epoch(0)
    epoch1 = sampler._indices_for_epoch(1)
    selected0 = {item[2] for item in epoch0 if item[0] == "variation"}
    selected1 = {item[2] for item in epoch1 if item[0] == "variation"}
    assert len(selected0) == len(selected1) == 1
    assert selected0 != selected1
    assert len([item for item in epoch0 if item[0] == "regular"]) == 2
    assert len([item for item in epoch0 if item[0] == "variation"]) in (2, 3)
    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        assert len(sampler) == len(list(sampler))
    example = dataset[next(item for item in epoch0 if item[0] == "variation")]
    assert example.task_embedding_ids == dataset.get_task_embedding_ids(parent_id)
    assert example.variation_path in {"seed0.json", "seed1.json"}
    tokens, sep, _ = dataset.build_training_sequence(example, 0, None)
    torch.testing.assert_close(tokens[sep + 1:], example.tokens[example.sep_index + 1:])
    assert len(dataset.demonstration_indices(example, 0)) == min(2, len(dataset.demonstration_pairs(example)) - 1)

    args = SimpleNamespace(
        seed=42, device="cpu", checkpoint_path=None,
        data_path=tmp_path / "challenges.json",
        variation_index_path=tmp_path / "variation_index.json",
        rearc_data_path=None, nvarc_training_path=None, nvarc_full_path=None,
        batch_size=2, num_workers=0, pin_memory=False,
        persistent_workers=False, prefetch_factor=None,
        enable_aug=True, max_augments=2, epochs=2,
        gradient_accumulation_steps=1,
        d_model=32, n_heads=4, d_ff=64, n_layers=1,
        num_task_embedding_latents=2,
        dropout=0.0, embedding_dropout=0.0, attention_dropout=0.0, FA=2,
    )
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    model, built_dataset, loader, device, _ = build.build_model_and_data(args)
    assert isinstance(built_dataset, variations.VariationTrainingDataset)
    assert device.type == "cpu"
    assert next(iter(loader))["example_ids"].shape[0] == 2

    checkpoint_path = tmp_path / "model.pt"
    train.maybe_save_model(model, built_dataset, args.data_path, checkpoint_path)
    args.checkpoint_path = checkpoint_path
    args.checkpoint_mode = "resume"
    build.build_model_and_data(args, build_dataloader=False)
    if with_split:
        original_index = (tmp_path / "variation_index.json").read_text()
        changed = json.loads(original_index)
        changed["parent_split"]["sha256"] = "different-split"
        (tmp_path / "variation_index.json").write_text(json.dumps(changed))
        with pytest.raises(ValueError, match="does not match"):
            build.build_model_and_data(args, build_dataloader=False)
        (tmp_path / "variation_index.json").write_text(original_index)
    args.seed = 99
    with pytest.raises(ValueError, match="does not match"):
        build.build_model_and_data(args, build_dataloader=False)
    args.seed = 42
    changed_index = json.loads((tmp_path / "variation_index.json").read_text())
    changed_index["archive_sha256"] = "changed"
    (tmp_path / "variation_index.json").write_text(json.dumps(changed_index))
    with pytest.raises(ValueError, match="does not match"):
        build.build_model_and_data(args, build_dataloader=False)
