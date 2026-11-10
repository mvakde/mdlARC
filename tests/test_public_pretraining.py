"""Public-answer pretraining must be explicit and preserve frozen synthetic inputs."""

import hashlib
import json
from pathlib import Path
import sys
import tarfile
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import launch_run
import variations
from dataset_building_scripts import build_variation_dataset as builder
from scripts import package_compositions as packager
from workflows.vast import snapshot

PAIR = {"input": [[1]], "output": [[2]]}
TEST_PAIR = {"input": [[3]], "output": [[4]]}
TASK = {"train": [PAIR], "test": [TEST_PAIR]}
SOURCES = ["arc2_train", "arc2_eval", "conceptarc", "arc1_filtered"]


def public_spec():
    return {
        "dataset": {"dataset": "arc2", "style": "pretrain", "pretrain_data": {
            "kind": "task_variations", "snapshot": "task-variations-v1",
            "public_pretraining": True, "sources": SOURCES, "original_sources": SOURCES,
            "sampling": {"variations_per_parent_per_epoch": 42, "embedding_scope": "variation"},
        }},
        "enable_finetune": False, "skip_evals": True,
    }


def write_compositions(root, records=None):
    data = json.dumps(TASK).encode()
    records = records if records is not None else [
        {"path": f"task-compositions/v3/family/seed{seed}.json", "seed": seed,
         "composition_id": "composition/v3/arc2_eval/family", "source": "arc2_eval",
         "sha256": hashlib.sha256(data).hexdigest(), "stage": "author"}
        for seed in (0, 54321)
    ]
    path = root / "assets_tmp/compositions.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        for item in records:
            archive.writestr(item["path"], data)
        archive.writestr("composition_manifest.json", json.dumps({"schema_version": 1, "samples": records}))
    return {"archive_path": path.relative_to(root).as_posix(), "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.mark.parametrize("change, message", [
    ("implicit", "unsupported sources"), ("style", "requires pretrain style"),
    ("score", "requires skip_evals"), ("support", "rather than evaluation support"),
])
def test_public_answers_require_explicit_pretrain_without_public_scoring(change, message):
    spec = public_spec()
    if change == "implicit":
        spec["dataset"]["pretrain_data"].pop("public_pretraining")
    elif change == "style":
        spec["dataset"]["style"] = "finetune"
    elif change == "score":
        spec["skip_evals"] = False
    else:
        spec["dataset"]["pretrain_data"]["eval_support_repeats_per_epoch"] = 1
    with pytest.raises(ValueError, match=message):
        launch_run.build_plan(spec, Path("spec.json"))


@pytest.fixture
def public_corpus(tmp_path, monkeypatch):
    assets = tmp_path / "assets_tmp"
    archive_path = assets / "task-variations-v1.zip"
    assets.mkdir()
    with zipfile.ZipFile(archive_path, "w") as archive:
        for source in SOURCES:
            native_id = source + "_task"
            original = assets / builder.ORIGINAL_FILES[source]
            original.parent.mkdir(exist_ok=True)
            original.write_text(json.dumps({native_id: TASK}))
            for seed in range(42):
                archive.writestr(f"exported-datasets/task-variation/{builder.SOURCE_DIRS[source]}/"
                                 f"{native_id}/{native_id}seed{seed}.json", json.dumps(TASK))
    (assets / "ARC-2/arc2_eval_challenges.json").write_text(json.dumps({"arc2_eval_task": {
        "train": [PAIR], "test": [{"input": TEST_PAIR["input"]}],
    }}))
    (assets / "ARC-2/arc2_eval_solutions.json").write_text(json.dumps({"arc2_eval_task": [TEST_PAIR["output"]]}))
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "colab_drive.json").write_text(json.dumps({
        "dataset_size": archive_path.stat().st_size,
        "dataset_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }))
    for module in (builder, variations):
        monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "ASSETS_TMP", assets)
    monkeypatch.setattr(launch_run, "REPO_ROOT", tmp_path)
    spec = public_spec()
    spec["dataset"]["pretrain_data"]["compositions"] = write_compositions(tmp_path)
    return tmp_path, spec


def test_all_public_solved_pairs_and_every_composition_sample_are_in_each_epoch(public_corpus):
    root, spec = public_corpus
    plan = launch_run.build_plan(spec, root / "spec.json")
    assert not plan["finetunes"] and not plan["evals"]
    result = builder.build_variation_dataset(plan["dataset"], root / "data")
    assert (result["original_tasks"], result["variation_files"], result["composition_tasks"],
            result["composition_samples"]) == (4, 168, 1, 2)
    tasks = json.loads((root / "data/challenges.json").read_text())
    assert tasks["arc2_eval/arc2_eval_task"]["train"] == [PAIR, TEST_PAIR]
    dataset = variations.VariationTrainingDataset(json_path=root / "data/challenges.json",
        index_path=root / "data/variation_index.json", seed=42, max_augments=80)
    sampler = variations.VariationBatchSampler(dataset, 112, seed=42,
        variations_per_parent_per_epoch=42, eval_support_repeats_per_epoch=0)
    family = "composition/v3/arc2_eval/family"
    assert dataset.variation_task_ids[family] == [family + "/variation/seed0", family + "/variation/seed54321"]
    assert len({dataset.task_id_to_embedding_ids[task][0] for task in dataset.variation_task_ids[family]}) == 2
    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        indices = sampler._indices_for_epoch(epoch)
        assert len(indices) == 168 * 2 + 2 * 2 + 4 * 2
        composition_indices = [item for item in indices if item[0] == "variation" and item[1] == family]
        assert {(item[2], item[3]) for item in composition_indices} == {(0, 0), (0, 1), (1, 0), (1, 1)}
        assert len(composition_indices) == 4
        assert len(sampler) == len(list(sampler)) == 4
        test_target = next(item for item in composition_indices if item[3] == 1)
        assert dataset[test_target].pair_index == 1
    metadata = dataset.checkpoint_metadata(enable_aug=True)
    assert metadata["public_pretraining"] is True
    assert metadata["composition_archive_sha256"] == spec["dataset"]["pretrain_data"]["compositions"]["sha256"]


def test_changed_composition_archive_or_extracted_cache_is_rejected(public_corpus):
    root, spec = public_corpus
    identity = spec["dataset"]["pretrain_data"]["compositions"]
    tasks = builder._composition_tasks(identity, SOURCES)
    cached = root / next(iter(tasks.values()))[0]["path"]
    cached.write_text("{}")
    with pytest.raises(ValueError, match="Extracted composition sample changed"):
        builder._composition_tasks(identity, SOURCES)
    (root / identity["archive_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="approved experiment identity"):
        builder._composition_tasks(identity, SOURCES)


@pytest.mark.parametrize("change,message", [
    ("checksum", "checksum changed"), ("path", "Unsafe composition sample path"),
    ("source", "outside the selected"), ("duplicate", "Duplicate or invalid composition seed"),
])
def test_invalid_manifest_cannot_reach_training(public_corpus, change, message):
    root, spec = public_corpus
    with zipfile.ZipFile(root / "assets_tmp/compositions.zip") as archive:
        records = json.loads(archive.read("composition_manifest.json"))["samples"]
    if change == "checksum":
        records[0]["sha256"] = "0" * 64
    elif change == "path":
        records[0]["path"] = "task-compositions/../../outside.json"
    elif change == "source":
        records[0]["source"] = "private"
    else:
        records[1]["seed"] = records[0]["seed"]
    identity = write_compositions(root, records)
    with pytest.raises(ValueError, match=message):
        builder._composition_tasks(identity, SOURCES)


def test_composition_archive_is_in_vast_source_and_must_match_the_spec(public_corpus):
    root, spec = public_corpus
    for name in ("launch_run.py", "requirements.txt", "scripts/run_layout.py"):
        (root / name).write_text("")
    path = root / "spec.json"
    path.write_text(json.dumps(spec))
    snapshot.create(root, path, root / "source.tar.gz")
    with tarfile.open(root / "source.tar.gz") as archive:
        assert "scripts/colab_drive.json" in archive.getnames()
        assert archive.extractfile("assets_tmp/compositions.zip").read() == (root / "assets_tmp/compositions.zip").read_bytes()
    spec["dataset"]["pretrain_data"]["compositions"]["sha256"] = "0" * 64
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="selected experiment spec"):
        snapshot.source_files(root, path)


def test_packager_selects_only_v3_and_checks_author_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(packager, "ROOT", tmp_path)
    synth = tmp_path / "synth"
    (synth / "results/task-composition-v3-empty-placeholder").mkdir(parents=True)
    data = json.dumps(TASK).encode()
    for pipeline in ("task-composition-v3", "task-compsition-v2-test"):
        root = synth / "results" / pipeline
        job = root / "family"
        folder = job / "llm-inputs/author/checked-samples"
        folder.mkdir(parents=True)
        (root / "plan.json").write_text(json.dumps({"dataset": "arc2/evaluation"}))
        (job / "status.json").write_text(json.dumps({"state": "author_completed", "dataset": "arc2/evaluation", "parents": ["one", "two"]}))
        name = "llm-inputs/author/checked-samples/seed-0000000000.json"
        (job / name).write_bytes(data)
        (job / "checkpoints.json").write_text(json.dumps({"author": {name: hashlib.sha256(data).hexdigest()}}))
    output = tmp_path / "assets_tmp/compositions.zip"
    with pytest.raises(ValueError, match="No eligible"):
        packager.package(synth, output)
    result = packager.package(synth, output, include_author_completed=True, pipeline="v3")
    assert (result["tasks"], result["samples"], result["stages"]) == (1, 1, ["author"])
    with zipfile.ZipFile(output) as archive:
        assert all("v2" not in name for name in archive.namelist())
    with pytest.raises(ValueError, match="already exists"):
        packager.package(synth, output, include_author_completed=True)
    (synth / "results/task-composition-v3/family/llm-inputs/author/checked-samples/seed-0000000000.json").write_text("{}")
    with pytest.raises(ValueError, match="sample checksum changed"):
        packager.package(synth, tmp_path / "assets_tmp/changed.zip", include_author_completed=True)


def test_packager_requires_plan_for_a_root_with_saved_jobs(tmp_path):
    synth = tmp_path / "synth"
    job = synth / "results/task-composition-v3-incomplete/family"
    job.mkdir(parents=True)
    (job / "status.json").write_text(json.dumps({"state": "author_completed"}))
    with pytest.raises(FileNotFoundError, match="plan.json"):
        list(packager.selected_samples(synth, include_author_completed=True))
