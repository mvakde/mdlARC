import collections
import json
import re
import sys
import zipfile
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import launch_run
from dataset_building_scripts import build_variation_dataset as builder
from dataset_building_scripts.task_splits import content_sha256, load_parent_split


def test_fixed_conceptarc_split_has_the_requested_groups():
    manifest, _ = load_parent_split(ROOT, "dataset_building_scripts/dataset_splits/conceptarc_topbottom3d_v1.json")
    train = set(manifest["train"])
    familiar = set(manifest["eval_groups"]["familiar"])
    unseen = set(manifest["eval_groups"]["heldout_concept"])
    assert (len(train), len(familiar), len(unseen)) == (120, 30, 10)
    assert unseen == {f"TopBottom3D{i}" for i in range(1, 11)}
    assert len(train | familiar | unseen) == 160
    concepts = lambda ids: collections.Counter(re.sub(r"\d+$", "", task_id) for task_id in ids)
    assert set(concepts(train).values()) == {8}
    assert set(concepts(familiar).values()) == {2}
    assert concepts(train)["Count"] == 8
    assert ("MoveToBoundary9" in train) == ("MoveToBoundary10" in train)


@pytest.fixture
def split_corpus(tmp_path, monkeypatch):
    assets = tmp_path / "assets_tmp"
    originals = {}
    for index, name in enumerate(("Copy1", "Copy2", "Copy3", "TopBottom3D1"), 1):
        originals[name] = {
            "train": [{"input": [[index]], "output": [[index + 1]]}],
            "test": [{"input": [[index, index]], "output": [[index + 1, index + 1]]}],
        }
    original_path = assets / "ConceptARC/concept_all.json"
    original_path.parent.mkdir(parents=True)
    original_path.write_text(json.dumps(originals))
    manifest = {
        "schema_version": 1, "name": "tiny", "source": "conceptarc",
        "source_sha256": content_sha256(originals),
        "train": ["Copy1", "Copy2"],
        "eval_groups": {"familiar": ["Copy3"], "heldout_concept": ["TopBottom3D1"]},
    }
    split_path = tmp_path / "dataset_building_scripts/dataset_splits/tiny.json"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps(manifest))
    root = assets / "exported-datasets/task-variation"
    archive_path = assets / "task-variations-v1.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name in manifest["train"]:
            folder = root / "conceptarc" / name
            folder.mkdir(parents=True)
            for seed in range(42):
                path = folder / f"{name}seed{seed}.json"
                contents = json.dumps(originals[name])
                path.write_text(contents)
                archive.writestr(path.relative_to(assets).as_posix(), contents)
    # Even invalid held-out variation files must never be read or indexed.
    for name in ("Copy3", "TopBottom3D1"):
        folder = root / "conceptarc" / name
        folder.mkdir()
        (folder / "must-not-load.json").write_text("not JSON")
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "ASSETS_TMP", assets)
    monkeypatch.setattr(builder, "_archive_identity", lambda: (archive_path, "archive-digest"))
    monkeypatch.setattr(builder, "_ensure_extracted", lambda _: root)
    monkeypatch.setattr(launch_run, "REPO_ROOT", tmp_path)
    spec = {
        "dataset": {
            "style": "finetune",
            "evaluation": {"source": "conceptarc", "split_manifest": "dataset_building_scripts/dataset_splits/tiny.json"},
            "pretrain_data": {
                "kind": "task_variations", "snapshot": "task-variations-v1",
                "sources": ["conceptarc"], "original_sources": ["conceptarc"], "sampling": {},
            },
        },
    }
    return SimpleNamespace(root=tmp_path, originals=originals, original_path=original_path,
                           manifest=manifest, split_path=split_path, spec=spec)


def test_builder_filters_variations_and_originals_and_keeps_all_eval_answers_separate(split_corpus):
    data = split_corpus
    config = launch_run.build_dataset_config(data.spec["dataset"])
    output = data.root / "data"
    result = builder.build_variation_dataset(config, output)
    load = lambda name: json.loads((output / name).read_text())
    train = load("challenges.json")
    index = load("variation_index.json")
    heldout = load("priv_challenges.json")
    assert set(train) == set(index["parents"]) == {"conceptarc/Copy1", "conceptarc/Copy2"}
    assert result["variation_files"] == 84
    assert result["original_tasks"] == 2
    assert index["eval_support_task_ids"] == []
    assert set(heldout) == {"conceptarc/Copy3", "conceptarc/TopBottom3D1"}
    for group, ids in data.manifest["eval_groups"].items():
        challenges = load(f"eval_{group}_challenges.json")
        solutions = load(f"eval_{group}_solutions.json")
        assert set(challenges) == set(solutions) == {f"conceptarc/{task_id}" for task_id in ids}
        for task_id in ids:
            original = data.originals[task_id]
            key = f"conceptarc/{task_id}"
            assert challenges[key] == heldout[key] == {
                "train": original["train"],
                "test": [{"input": pair["input"]} for pair in original["test"]],
            }
            assert solutions[key] == [pair["output"] for pair in original["test"]]
    assert load("split_manifest.json") == data.manifest
    assert (output / "split_manifest.json").read_bytes() == data.split_path.read_bytes()
    assert index["parent_split"] == load_parent_split(data.root, "dataset_building_scripts/dataset_splits/tiny.json")[1]
    assert index["evaluation_source"] == "conceptarc"
    assert json.loads(data.original_path.read_text()) == data.originals


def _select_builder(data, kind):
    if kind == "original_tasks":
        data.spec["dataset"]["pretrain_data"] = {"kind": kind, "sources": ["conceptarc"]}
        return builder.build_original_dataset
    return builder.build_variation_dataset


@pytest.mark.parametrize("kind", ["task_variations", "original_tasks"])
@pytest.mark.parametrize("failure", ["source_changed", "missing_task", "overlap", "shared_pair"])
def test_builder_rejects_changed_or_leaking_splits(split_corpus, failure, kind):
    data = split_corpus
    build_data = _select_builder(data, kind)
    if failure == "source_changed":
        data.originals["Copy1"]["train"][0]["output"] = [[9]]
    elif failure == "missing_task":
        data.manifest["train"].remove("Copy2")
    elif failure == "overlap":
        data.manifest["eval_groups"]["familiar"].append("Copy1")
    else:
        data.originals["Copy3"]["train"] = data.originals["Copy1"]["train"]
        data.manifest["source_sha256"] = content_sha256(data.originals)
    data.original_path.write_text(json.dumps(data.originals))
    data.split_path.write_text(json.dumps(data.manifest))
    messages = {"source_changed": "source content", "missing_task": "cover every", "overlap": "disjoint", "shared_pair": "crosses"}
    with pytest.raises(ValueError, match=messages[failure]):
        config = launch_run.build_dataset_config(data.spec["dataset"])
        build_data(config, data.root / "data")


def test_split_runs_can_evaluate_before_finetune_without_training_on_eval_support(split_corpus):
    data = split_corpus
    data.spec["dataset"]["style"] = "pretrain"
    data.spec["enable_finetune"] = False
    plan = launch_run.build_plan(data.spec, data.root / "spec.json")
    assert set(plan["evals"]) == {"final_familiar", "final_heldout_concept"}
    assert {cfg["checkpoint"] for cfg in plan["evals"].values()} == {"pretrain.final"}
    assert plan["dataset"]["pretrain_data"]["eval_support_repeats_per_epoch"] == 0
    builder.build_variation_dataset(plan["dataset"], data.root / "data")
    assert set(json.loads((data.root / "data/challenges.json").read_text())) == {"conceptarc/Copy1", "conceptarc/Copy2"}
    data.spec["dataset"]["pretrain_data"]["eval_support_repeats_per_epoch"] = 1
    with pytest.raises(ValueError, match="keep evaluation support out"):
        launch_run.build_plan(data.spec, data.root / "spec.json")


@pytest.mark.parametrize("kind", ["task_variations", "original_tasks"])
@pytest.mark.parametrize("change", ["spec_manifest", "cached_manifest"])
def test_cached_data_cannot_silently_use_a_different_split(split_corpus, monkeypatch, change, kind):
    data = split_corpus
    build_data = _select_builder(data, kind)
    data.spec["skip_data_build"] = True
    plan = launch_run.build_plan(data.spec, data.root / "spec.json")
    build_data(plan["dataset"], plan["run_dir"] / "data")
    launch_run.validate_run_split(plan)
    if change == "spec_manifest":
        data.manifest["name"] = "changed"
        data.split_path.write_text(json.dumps(data.manifest))
    else:
        copied = plan["run_dir"] / "data/split_manifest.json"
        copied.write_text(copied.read_text() + "\n")
    monkeypatch.setattr(launch_run, "run_training_stage_isolated", lambda *args: pytest.fail("training started with a changed split"))
    with pytest.raises(ValueError, match="(does not match the current spec|does not match its recorded identity)"):
        launch_run.execute_plan(plan)


@pytest.mark.parametrize("kind", ["task_variations", "original_tasks"])
def test_eval_group_routes_decoding_and_scoring_but_finetunes_the_union(split_corpus, monkeypatch, kind):
    data = split_corpus
    _select_builder(data, kind)
    data.spec["evals"] = {
        group: {"checkpoint": "finetune.default.final", "group": group}
        for group in data.manifest["eval_groups"]
    }
    plan = launch_run.build_plan(data.spec, data.root / "spec.json")
    assert plan["finetunes"]["default"]["data_path"].name == "priv_challenges.json"
    decoded, scored = [], []
    def evaluate(cfg, **kwargs):
        assert cfg.data_path == kwargs["data_path"]
        decoded.append((cfg.group, cfg.data_path, kwargs["checkpoint_path"]))
        return ("final", {}, data.root / cfg.group / "submission.json")
    monkeypatch.setattr(launch_run, "load_runtime_modules", lambda: None)
    monkeypatch.setattr(launch_run, "evaluate", SimpleNamespace(run_evaluation=evaluate))
    monkeypatch.setattr(launch_run, "utils", SimpleNamespace(
        score_arc_submission=lambda solutions, _: scored.append(solutions),
    ))
    monkeypatch.setattr(launch_run, "write_score_summary", lambda *args, **kwargs: None)
    for name, cfg in plan["evals"].items():
        launch_run.run_eval_stage(plan, name, cfg)
    assert [path.name for _, path, _ in decoded] == ["eval_familiar_challenges.json", "eval_heldout_concept_challenges.json"]
    assert [path.name for path in scored] == ["eval_familiar_solutions.json", "eval_heldout_concept_solutions.json"]
    assert decoded[0][2] == decoded[1][2] == plan["finetunes"]["default"]["save_path"]
    data.spec["evals"]["familiar"]["group"] = "missing"
    with pytest.raises(ValueError, match="not a configured evaluation group"):
        launch_run.build_plan(data.spec, data.root / "spec.json")


def test_conceptarc_variation_spec_selects_topbottom3d_and_separate_final_evals():
    path = ROOT / "tests/fixtures/specs/conceptarc_variations_finetune.json"
    plan = launch_run.build_plan(launch_run.load_spec(path), path)
    assert plan["dataset"]["pretrain_data"]["sources"] == ["conceptarc"]
    assert plan["pretrain"]["epochs"] == 84
    assert set(plan["finetunes"]) == {"heldout"}
    assert {cfg["checkpoint"] for cfg in plan["evals"].values()} == {"finetune.heldout.final"}
    assert {cfg["group"] for cfg in plan["evals"].values()} == {"familiar", "heldout_concept"}


@pytest.mark.parametrize("kind,embedding_scope", [
    ("task_variations", "parent"), ("task_variations", "variation"), ("original_tasks", "parent"),
])
def test_finetune_extends_parent_embeddings_and_preserves_split_identity(split_corpus, monkeypatch, kind, embedding_scope):
    # Exercise actual checkpoint loading with new task IDs, without a training run.
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    import build
    import train

    data = split_corpus
    if kind == "task_variations":
        data.spec["dataset"]["pretrain_data"]["sampling"]["embedding_scope"] = embedding_scope
    build_data = _select_builder(data, kind)
    data.spec["defaults"] = {
        "device": "cpu", "d_model": 32, "n_heads": 4, "d_ff": 64, "n_layers": 1,
        "num_task_embedding_latents": 2, "num_workers": 0, "enable_aug": False,
        "max_augments": 1, "batch_size": 2, "FA": 2,
    }
    plan = launch_run.build_plan(data.spec, data.root / "spec.json")
    build_data(plan["dataset"], plan["run_dir"] / "data")
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    # The fixture variation paths are relative to its own repository root.
    import variations
    monkeypatch.setattr(variations, "ROOT", data.root)
    args = Namespace(**plan["pretrain"])
    model, dataset, _, _, _ = build.build_model_and_data(args)
    if embedding_scope == "variation":
        with torch.no_grad():
            model.task_factor_embedding.weight[dataset.variation_embedding_families[0][0]].add_(0.25)
    pretrained_factors = model.task_factor_embedding.weight.detach().clone()
    train.maybe_save_model(model, dataset, args.data_path, args.save_path)
    args = Namespace(**plan["finetunes"]["default"])
    model, dataset, _, _, _ = build.build_model_and_data(args)
    assert model.config.num_examples == (88 if embedding_scope == "variation" else 4)
    torch.testing.assert_close(model.task_factor_embedding.weight[:len(pretrained_factors)], pretrained_factors)
    assert {example.task_id for example in dataset.examples} == {"conceptarc/Copy3", "conceptarc/TopBottom3D1"}
    assert all(example.split == "train" for example in dataset.examples)
    train.maybe_save_model(model, dataset, args.data_path, args.save_path)
    for cfg in plan["evals"].values():
        eval_args = Namespace(**{**plan["shared"], **cfg, "checkpoint_path": args.save_path})
        eval_model, eval_dataset, _, _, _ = build.build_model_and_data(eval_args, is_eval=True, build_dataloader=False)
        metadata_key = "task_variations" if kind == "task_variations" else "task_data"
        assert eval_model._loaded_checkpoint[metadata_key]["parent_split"] == load_parent_split(data.root, "dataset_building_scripts/dataset_splits/tiny.json")[1]
        assert len(eval_dataset.test_pairs_by_task) == 1


@pytest.mark.parametrize("source", ["conceptarc", "arc2_train", "arc1_filtered"])
def test_original_builder_uses_only_training_parents_without_loading_variations(split_corpus, monkeypatch, source):
    data = split_corpus
    _select_builder(data, "original_tasks")
    data.spec["dataset"]["pretrain_data"]["sources"] = [source]
    data.spec["dataset"]["evaluation"]["source"] = source
    data.manifest["source"] = source
    data.split_path.write_text(json.dumps(data.manifest))
    original_path = data.root / "assets_tmp" / builder.ORIGINAL_FILES[source]
    original_path.parent.mkdir(parents=True, exist_ok=True)
    original_path.write_text(json.dumps(data.originals))
    monkeypatch.setattr(builder, "_archive_identity", lambda: pytest.fail("Original tasks requested the variation ZIP"))
    plan = launch_run.build_plan(data.spec, data.root / "spec.json")
    assert "variation_index_path" not in plan["pretrain"]
    launch_run.prepare_data(plan)
    output = plan["run_dir"] / "data"
    tasks = json.loads((output / "challenges.json").read_text())
    assert set(tasks) == {f"{source}/Copy1", f"{source}/Copy2"}
    for name in data.manifest["train"]:
        assert tasks[f"{source}/{name}"] == {"train": data.originals[name]["train"] + data.originals[name]["test"]}
    for group in data.manifest["eval_groups"]:
        heldout = json.loads((output / f"eval_{group}_challenges.json").read_text())
        assert all("output" not in pair for task in heldout.values() for pair in task["test"])
    launch_run.validate_run_split(plan)


@pytest.mark.parametrize("source,evaluation", [("arc2_train", "arc2_eval"), ("arc1_filtered", "arc1_eval")])
@pytest.mark.parametrize("style", ["finetune", "pretrain"])
def test_original_sources_support_official_evals_and_joint_training(tmp_path, monkeypatch, source, evaluation, style):
    monkeypatch.setattr(builder, "ASSETS_TMP", tmp_path)
    original = {"train": [{"input": [[1]], "output": [[2]]}], "test": [{"input": [[2]], "output": [[3]]}]}
    for name in (source, "conceptarc"):
        path = tmp_path / builder.ORIGINAL_FILES[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({f"parent_{name}": original}))
    challenge_file, solution_file = builder.EVALUATION_FILES[evaluation]
    challenges = {"heldout": {"train": original["train"], "test": [{"input": [[9]]}]}}
    (tmp_path / challenge_file).write_text(json.dumps(challenges))
    (tmp_path / solution_file).write_text(json.dumps({"heldout": [[[8]]]}))
    config = launch_run.build_dataset_config({
        "style": style, "evaluation": {"source": evaluation},
        "pretrain_data": {"kind": "original_tasks", "sources": [source, "conceptarc"]},
    })
    builder.build_original_dataset(config, tmp_path / "data")
    tasks = json.loads((tmp_path / "data/challenges.json").read_text())
    assert tasks[f"{source}/parent_{source}"]["train"] == original["train"] + original["test"]
    assert ("heldout" in tasks) == (style == "pretrain")
    if style == "pretrain":
        assert tasks["heldout"] == {"train": original["train"]}


@pytest.mark.parametrize("changes", [
    {"sources": []}, {"sources": ["arc2_eval"]}, {"sources": ["conceptarc", "conceptarc"]},
    {"sampling": {}}, {"snapshot": "task-variations-v1"}, {"original_sources": []},
    {"eval_support_repeats_per_epoch": 2},
])
def test_original_mode_rejects_invalid_or_variation_only_settings(changes):
    with pytest.raises(ValueError):
        launch_run.build_dataset_config({
            "style": "pretrain", "evaluation": {"source": "arc2_eval"},
            "pretrain_data": {"kind": "original_tasks", "sources": ["conceptarc"], **changes},
        })


def test_original_resume_rejects_changed_manifest_or_data(split_corpus, monkeypatch):
    sys.path.insert(0, str(ROOT / "src"))
    import torch
    import build
    import train
    data = split_corpus
    _select_builder(data, "original_tasks")
    data.spec["defaults"] = {
        "device": "cpu", "d_model": 32, "n_heads": 4, "d_ff": 64, "n_layers": 1,
        "num_task_embedding_latents": 2, "num_workers": 0, "enable_aug": False,
        "max_augments": 1, "batch_size": 2, "FA": 2,
    }
    plan = launch_run.build_plan(data.spec, data.root / "spec.json")
    builder.build_original_dataset(plan["dataset"], plan["run_dir"] / "data")
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    args = Namespace(**plan["pretrain"])
    model, dataset, _, _, _ = build.build_model_and_data(args)
    train.maybe_save_model(model, dataset, args.data_path, args.save_path)
    args.checkpoint_path = args.save_path
    build.build_model_and_data(args)
    manifest_path = args.dataset_manifest_path
    contents = manifest_path.read_text()
    metadata = json.loads(contents)
    metadata["parent_split"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Checkpoint original-task dataset"):
        build.build_model_and_data(args)
    manifest_path.write_text(contents)
    tasks = json.loads(args.data_path.read_text())
    tasks["conceptarc/Copy1"]["train"][0]["output"] = [[9]]
    args.data_path.write_text(json.dumps(tasks))
    with pytest.raises(ValueError, match="Training data does not match"):
        build.build_model_and_data(args)


def test_original_conceptarc_spec_matches_variation_finetune_and_eval():
    import copy
    path = ROOT / "tests/fixtures/specs/conceptarc_originals_finetune.json"
    spec = launch_run.load_spec(path)
    plan = launch_run.build_plan(spec, path)
    variation = launch_run.load_spec(ROOT / "tests/fixtures/specs/conceptarc_variations_finetune.json")
    assert spec["dataset"]["evaluation"] == variation["dataset"]["evaluation"]
    assert plan["pretrain"]["epochs"] == 330
    assert plan["pretrain"]["batch_size"] == 64
    assert plan["pretrain"]["enable_aug"] and plan["pretrain"]["max_augments"] == 80
    for key in ("finetune_defaults", "finetunes", "eval_defaults", "evals"):
        assert spec[key] == variation[key]
    # Pretrain-only overrides must never spill into finetuning or evaluation.
    alternative = copy.deepcopy(spec)
    alternative["pretrain"].update(batch_size=32, normuon_lr=0.004, adamw_lr=0.00072)
    alt = launch_run.build_plan(alternative, path)
    assert plan["finetunes"] == alt["finetunes"]
    assert plan["evals"] == alt["evals"]


def test_official_eval_source_is_selected_independently_of_variation_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, "ASSETS_TMP", tmp_path)
    for source, (challenge_file, solution_file) in builder.EVALUATION_FILES.items():
        challenge_path = tmp_path / challenge_file
        challenge_path.parent.mkdir()
        challenge_path.write_text(json.dumps({source: {"train": [], "test": [{"input": [[1]]}]}}))
        (tmp_path / solution_file).write_text(json.dumps({source: [[[2]]]}))
    for source in ("arc1_eval", "arc2_eval"):
        challenges, solutions, groups, manifest, identity = builder._evaluation_data({"evaluation": {"source": source}})
        assert set(challenges) == set(solutions) == {source}
        assert groups == {} and manifest is None and identity is None
    assert set(builder._evaluation_data({})[0]) == {"arc2_eval"}
