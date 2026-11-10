import hashlib
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest

from scripts import run_layout, run_records


def test_branch_paths_are_stable_and_do_not_collide(monkeypatch, tmp_path):
    monkeypatch.setenv("MDLARC_RUN_BRANCH", "codex/newreason")
    assert run_layout.results_root(tmp_path) == tmp_path / "results/codex%2Fnewreason"
    assert run_layout.branch_folder("codex%2Fnewreason") == "codex%252Fnewreason"
    assert run_layout.runs_root(tmp_path) == tmp_path / "results/codex%2Fnewreason/runs"
    monkeypatch.setenv("MDLARC_REMOTE", "1")
    assert run_layout.runs_root(tmp_path) == tmp_path / "runs"


def test_bundles_preserve_notes_journals_specs_and_large_outputs_without_checkpoints(tmp_path):
    base = run_layout.results_root(tmp_path)
    for name, value in {
        "experiments/notes/session.md": b"research interpretation",
        "experiments/generated/session.md": b"observed facts",
        "experiments/INDEX.md": b"index",
        "colab_runs/_active/session.json": b'{"state":"completed"}',
        "colab_runs/_active/session.connection.json": b'{"kernel_id":"original"}',
        "colab_runs/_active/session.watch.lock": b"lock",
        "colab_runs/session/spec.json": b'{"run_name":"test"}',
        "colab_runs/session/runs/test/large.txt": b"x" * (21 * 1024 * 1024),
        "colab_runs/session/runs/test/tiny.pt": b"checkpoint",
        "colab_runs/other/spec.json": b"unrelated run",
    }.items():
        path = base / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)
    bundle = tmp_path / "records.tar.gz"
    receipt = run_records.pack_records(tmp_path, bundle, "session")
    first = bundle.read_bytes()
    assert receipt["sha256"] == hashlib.sha256(first).hexdigest()
    with tarfile.open(bundle) as archive:
        names = archive.getnames()
        assert "experiments/notes/session.md" in names
        assert "colab_runs/_active/session.connection.json" in names
        assert "colab_runs/session/runs/test/large.txt" in names
        assert not any(name.endswith((".pt", ".lock")) for name in names)
        assert not any("other" in name for name in names)
    run_records.pack_records(tmp_path, bundle, "session")
    assert bundle.read_bytes() == first


def test_identical_cloud_record_bundle_is_reused_and_corruption_fails(tmp_path):
    bundle = tmp_path / "records.tar.gz"
    bundle.write_bytes(b"records")
    item = {"id": "backup", "parents": ["run"], "size": "7",
            "md5Checksum": hashlib.md5(b"records").hexdigest()}
    queries = []
    def listing(**kwargs):
        queries.append(kwargs["q"])
        return SimpleNamespace(execute=lambda **kwargs: {"files": [item]})
    store = SimpleNamespace(
        accept_run_folder=lambda folder: None,
        service=SimpleNamespace(files=lambda: SimpleNamespace(list=listing)),
    )
    assert run_records.publish_bundle(store, "run", bundle)["id"] == "backup"
    assert "'run' in parents" in queries[0]
    item["md5Checksum"] = "wrong"
    with pytest.raises(ValueError, match="checksum verification"):
        run_records.publish_bundle(store, "run", bundle)
