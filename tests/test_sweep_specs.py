import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

from scripts import materialize_sweep_spec, run_on_colab


def trial_files(root: Path) -> tuple[Path, Path]:
    template = root / "specs" / "base.json"
    template.parent.mkdir(parents=True)
    template.write_text(json.dumps({
        "run_name": "sweep", "finetune_defaults": {"batch_size": 32, "normuon_lr": 0.001},
        "finetunes": {}, "evals": {},
    }))
    record = root / "results" / "test-branch" / "experiments" / "trials" / "trial.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({
        "trial_id": "b8_e90", "template": "specs/base.json",
        "overrides": {
            "finetune_defaults": {"batch_size": 8},
            "finetunes": {"b8_e90": {"epochs": 90}},
            "evals": {"b8_e90": {"checkpoint": "finetune.b8_e90.final"}},
        },
    }))
    return template, record


def test_materialized_trial_is_single_stage_and_preserves_base_settings(tmp_path):
    template, record = trial_files(tmp_path)
    staged = materialize_sweep_spec.stage(tmp_path, record)
    spec = json.loads(staged.read_text())
    assert staged == tmp_path / "specs" / "temp specs" / "b8_e90.json"
    assert spec["finetune_defaults"] == {"batch_size": 8, "normuon_lr": 0.001}
    assert list(spec["finetunes"]) == list(spec["evals"]) == ["b8_e90"]
    assert json.loads(template.read_text())["finetunes"] == {}

    staged.write_text("changed")
    with pytest.raises(FileExistsError, match="Staged spec differs"):
        materialize_sweep_spec.stage(tmp_path, record)
    materialize_sweep_spec.stage(tmp_path, record, replace=True)
    assert staged.read_bytes() == materialize_sweep_spec.materialize(tmp_path, record).spec_bytes


def test_trial_rejects_multiple_stages_and_template_escape(tmp_path):
    _, record = trial_files(tmp_path)
    data = json.loads(record.read_text())
    data["overrides"]["finetunes"]["other"] = {"epochs": 90}
    record.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="exactly trial_id"):
        materialize_sweep_spec.materialize(tmp_path, record)
    data["overrides"]["finetunes"].pop("other")
    data["template"] = "../outside.json"
    record.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="inside the repository"):
        materialize_sweep_spec.materialize(tmp_path, record)


def test_archive_contains_only_selected_spec_and_saves_its_exact_bytes(monkeypatch, tmp_path):
    template, record = trial_files(tmp_path)
    selected = materialize_sweep_spec.stage(tmp_path, record)
    (tmp_path / "specs" / "historical_stage2.json").write_text("{}")
    (tmp_path / "specs" / "temp specs" / "unrelated.json").write_text("{}")
    for name, contents in {"launch_run.py": "pass", "requirements.txt": "torch",
        "scripts/run_layout.py": "# storage layout",
                           "src/model.py": "local edit", "wheels/test.whl": "wheel"}.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)
    (tmp_path / "wheels" / "environment.json").write_text(json.dumps({
        "wheel": "test.whl", "sha256": hashlib.sha256(b"wheel").hexdigest(),
    }))
    monkeypatch.setattr(run_on_colab, "ROOT", tmp_path)
    archive = tmp_path / "source.tar.gz"
    run_on_colab.create_snapshot(selected, archive)
    with tarfile.open(archive) as source:
        names = source.getnames()
        assert "specs/temp specs/b8_e90.json" in names
        assert "specs/base.json" not in names
        assert "specs/historical_stage2.json" not in names
        assert "specs/temp specs/unrelated.json" not in names
    archived = tmp_path / "results" / "test-branch" / "colab_runs" / "older-session" / "specs" / "b8_e90.json"
    archived.parent.mkdir(parents=True)
    archived.write_bytes(selected.read_bytes())
    archived_archive = tmp_path / "archived-source.tar.gz"
    run_on_colab.create_snapshot(archived, archived_archive)
    with tarfile.open(archived_archive) as source:
        assert "results/test-branch/colab_runs/older-session/specs/b8_e90.json" in source.getnames()
        assert "specs/temp specs/b8_e90.json" not in source.getnames()
    resolved = materialize_sweep_spec.materialize(tmp_path, record)
    saved = run_on_colab.preserve_launch_spec(
        "session-1", archive, selected.relative_to(tmp_path),
        hashlib.sha256(resolved.spec_bytes).hexdigest(), resolved,
    )
    assert saved.read_bytes() == resolved.spec_bytes
    assert (saved.parent / "trial.json").read_bytes() == record.read_bytes()
    assert (saved.parent / "template.json").read_bytes() == template.read_bytes()
    provenance = json.loads((saved.parent / "spec_provenance.json").read_text())
    assert provenance["trial_id"] == "b8_e90"
    with pytest.raises(ValueError, match="changed between validation and packaging"):
        run_on_colab.preserve_launch_spec("session-2", archive, selected.relative_to(tmp_path), "0" * 64)


def test_staged_spec_requires_matching_trial_record_before_launch(monkeypatch, tmp_path):
    _, record = trial_files(tmp_path)
    selected = materialize_sweep_spec.stage(tmp_path, record)
    monkeypatch.setattr(run_on_colab, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_on_colab.py", "--config", str(selected)])
    with pytest.raises(SystemExit, match="2"):
        run_on_colab.main()
    monkeypatch.setattr(sys, "argv", ["run_on_colab.py", "--config", str(selected),
                                   "--trial-record", str(record), "--standalone-spec"])
    with pytest.raises(SystemExit, match="2"):
        run_on_colab.main()
    monkeypatch.setattr(sys, "argv", ["run_on_colab.py", "--config", str(selected),
                                   "--trial-record", str(record)])
    selected.write_text("{}")
    with pytest.raises(SystemExit, match="2"):
        run_on_colab.main()
    assert not (tmp_path / "results" / "test-branch" / "experiments" / "notes").exists()


def saved_files(destination):
    return {path.relative_to(destination).as_posix(): path.read_bytes()
            for path in destination.rglob("*") if path.is_file()}


def preserved_launch(monkeypatch, root, *, with_trial=True):
    template, record = trial_files(root)
    trial = materialize_sweep_spec.materialize(root, record) if with_trial else None
    selected = materialize_sweep_spec.stage(root, record) if trial else template
    spec_bytes = selected.read_bytes()
    archive = root / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as output:
        output.add(selected, arcname=selected.relative_to(root).as_posix(), recursive=False)
    monkeypatch.setattr(run_on_colab, "ROOT", root)
    saved_spec = run_on_colab.preserve_launch_spec(
        "session-1", archive, selected.relative_to(root), hashlib.sha256(spec_bytes).hexdigest(), trial,
    )
    destination = saved_spec.parent
    expected = {"source.tar.gz": archive.read_bytes(), "spec.json": spec_bytes,
                "spec_provenance.json": (destination / "spec_provenance.json").read_bytes()}
    if trial:
        expected.update({"trial.json": record.read_bytes(), "template.json": template.read_bytes()})
    assert saved_files(destination) == expected
    return destination, expected


def stub_result_download(monkeypatch, members):
    """Stub only remote I/O; checksums, staging and extraction run normally."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as output:
        for name, contents in members.items():
            member = tarfile.TarInfo(name)
            if contents is None:
                member.type = tarfile.SYMTYPE
                member.linkname = "../outside"
                output.addfile(member)
            else:
                member.size = len(contents)
                output.addfile(member, io.BytesIO(contents))
    payload = buffer.getvalue()
    manifest = {"sha256": hashlib.sha256(payload).hexdigest(), "parts": []}
    remote = {}
    middle = len(payload) // 2
    for index, chunk in enumerate((payload[:middle], payload[middle:])):
        name = f"mdlARC-results.part{index:04d}"
        manifest["parts"].append({"name": name, "size": len(chunk),
                                  "sha256": hashlib.sha256(chunk).hexdigest()})
        remote[f"/content/{name}"] = chunk

    def fake_download(command, *, timeout):
        assert command[:4] == ["colab", "download", "-s", "session-1"]
        assert timeout == 60
        data = (json.dumps(manifest).encode() if command[4] == run_on_colab.REMOTE_MANIFEST
                else remote[command[4]])
        Path(command[5]).write_bytes(data)

    monkeypatch.setattr(run_on_colab, "run_command", fake_download)
    return manifest


@pytest.mark.parametrize("with_trial", [False, True], ids=["standalone", "trial"])
@pytest.mark.parametrize("ok", [True, False], ids=["success", "failure"])
def test_result_recovery_keeps_saved_launch_records(monkeypatch, tmp_path, with_trial, ok):
    destination, preserved = preserved_launch(monkeypatch, tmp_path, with_trial=with_trial)
    status = {"ok": ok, "error": None if ok else "Training failed"}
    results = {"run_status.json": json.dumps(status).encode(),
               "runs/sweep/console.log": b"Training console\n"}
    stub_result_download(monkeypatch, results)
    assert run_on_colab.download_results("colab", "session-1", destination) == status
    assert saved_files(destination) == {**preserved, **results}


@pytest.mark.parametrize("member", [
    "source.tar.gz", "source.tar.gz/child", "spec.json", "trial.json", "template.json",
    "spec_provenance.json",
])
def test_result_recovery_rejects_collisions_with_saved_launch_records(monkeypatch, tmp_path, member):
    destination, preserved = preserved_launch(monkeypatch, tmp_path)
    stub_result_download(monkeypatch, {"run_status.json": b'{"ok":true}', member: b"conflict"})
    with pytest.raises(FileExistsError, match="Result archive conflicts"):
        run_on_colab.download_results("colab", "session-1", destination)
    assert saved_files(destination) == preserved


@pytest.mark.parametrize("existing_result", ["run_status.json", "runs/sweep/console.log"])
def test_result_recovery_rejects_existing_results(monkeypatch, tmp_path, existing_result):
    destination, _ = preserved_launch(monkeypatch, tmp_path)
    path = destination / existing_result
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"existing results")
    before = saved_files(destination)
    stub_result_download(monkeypatch, {"run_status.json": b'{"ok":true}'})
    with pytest.raises(FileExistsError, match="Result destination already contains results"):
        run_on_colab.download_results("colab", "session-1", destination)
    assert saved_files(destination) == before


@pytest.mark.parametrize("corruption", ["size", "chunk_hash", "archive_hash"])
def test_result_recovery_verifies_checksums_before_changing_saved_launch_records(
        monkeypatch, tmp_path, corruption):
    destination, preserved = preserved_launch(monkeypatch, tmp_path)
    manifest = stub_result_download(monkeypatch, {"run_status.json": b'{"ok":true}'})
    if corruption == "size":
        manifest["parts"][0]["size"] += 1
    elif corruption == "chunk_hash":
        manifest["parts"][0]["sha256"] = "0" * 64
    else:
        manifest["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="verification failed"):
        run_on_colab.download_results("colab", "session-1", destination)
    assert saved_files(destination) == preserved


@pytest.mark.parametrize("member,contents", [("../outside", b"escape"), ("link", None)])
def test_result_recovery_rejects_unsafe_archive_members(monkeypatch, tmp_path, member, contents):
    destination, preserved = preserved_launch(monkeypatch, tmp_path)
    stub_result_download(monkeypatch, {"run_status.json": b'{"ok":true}', member: contents})
    with pytest.raises(ValueError, match="Unsafe result archive member"):
        run_on_colab.download_results("colab", "session-1", destination)
    assert saved_files(destination) == preserved
    assert {path.name for path in destination.parent.iterdir()} == {"session-1"}
