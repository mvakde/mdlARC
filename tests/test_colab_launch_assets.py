from pathlib import Path
import hashlib
import json
import tarfile
import pytest

from scripts import run_on_colab


def test_adaptation_snapshot_pins_all_inputs(monkeypatch, tmp_path):
    for name in ("launch_run.py", "requirements.txt", "scripts/run_layout.py"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n")
    identities = {}
    for name in ("runtime", "challenges", "solutions"):
        path = tmp_path / "inputs" / (name + ".json")
        path.parent.mkdir(exist_ok=True)
        path.write_text("{}")
        identities[name] = {"path": path.relative_to(tmp_path).as_posix(),
                            "size": 2, "sha256": hashlib.sha256(b"{}").hexdigest()}
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"FA": 4, "checkpoint_adaptation": {
        "method": "lora", "inputs": identities}}))
    monkeypatch.setattr(run_on_colab, "ROOT", tmp_path)
    paths = run_on_colab.source_files(spec)
    assert all(tmp_path / identity["path"] in paths for identity in identities.values())
    (tmp_path / identities["challenges"]["path"]).write_text("[]")
    with pytest.raises(ValueError, match="Adaptation input changed"):
        run_on_colab.source_files(spec)


def test_colab_snapshot_contains_local_edits_without_credentials(monkeypatch, tmp_path: Path):
    for relative, contents in {
        "launch_run.py": "print('local launcher')",
        "requirements.txt": "torch",
        "scripts/run_layout.py": "# storage layout",
        "src/model.py": "print('uncommitted model edit')",
        "specs/example.json": json.dumps({"dataset": {"evaluation": {"split_manifest": "dataset_building_scripts/dataset_splits/fixed.json"}}}),
        "dataset_building_scripts/dataset_splits/fixed.json": "{\"fixed\": true}",
        "assets_tmp/ARC-2/arc2_train_both.json": "{}",
        "reviewed_arc2_training/old_branch_task.json": "{}",
        ".dataset-build.env": "SECRET=do-not-send",
        "src/__pycache__/model.pyc": "cached",
        "wheels/flash_attn-test.whl": "local wheel",
    }.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    (tmp_path / "wheels/environment.json").write_text(json.dumps({
        "wheel": "flash_attn-test.whl",
        "sha256": hashlib.sha256(b"local wheel").hexdigest(),
    }))
    monkeypatch.setattr(run_on_colab, "ROOT", tmp_path)

    snapshot = tmp_path / "source.tar.gz"
    run_on_colab.create_snapshot(tmp_path / "specs/example.json", snapshot)

    with tarfile.open(snapshot, "r:gz") as archive:
        names = archive.getnames()
        assert "src/model.py" in names
        assert archive.extractfile("src/model.py").read() == b"print('uncommitted model edit')"
        assert "assets_tmp/ARC-2/arc2_train_both.json" in names
        assert "dataset_building_scripts/dataset_splits/fixed.json" in names
        assert archive.extractfile("dataset_building_scripts/dataset_splits/fixed.json").read() == b'{"fixed": true}'
        assert "reviewed_arc2_training/old_branch_task.json" not in names
        assert ".dataset-build.env" not in names
        assert "src/__pycache__/model.pyc" not in names
        assert "wheels/environment.json" in names
        assert archive.extractfile("wheels/flash_attn-test.whl").read() == b"local wheel"
