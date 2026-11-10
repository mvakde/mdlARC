import hashlib
import io
import json
from contextlib import nullcontext
from pathlib import Path
import shutil
import sys
import tarfile
from types import SimpleNamespace

import pytest
import requests

from scripts import colab_connection, colab_run_docs, run_on_colab


@pytest.fixture
def unsubmitted(tmp_path, monkeypatch):
    spec = {"run_name": "original", "enable_finetune": False, "defaults": {"epochs": 330}}
    config = tmp_path / "specs/original.json"
    config.parent.mkdir()
    config.write_text(json.dumps(spec))
    colab_run_docs.start_journal(tmp_path, "session", config="specs/original.json", run_name="original",
                               gpu="A100", high_mem=True, min_gpu_memory_gb=75, spec_path=config)
    base = tmp_path / "results/test-branch/colab_runs"
    saved = base / "session"
    saved.mkdir()
    (saved / "spec.json").write_bytes(config.read_bytes())
    with tarfile.open(saved / "source.tar.gz", "w:gz") as archive:
        for name, data in [("specs/original.json", config.read_bytes()),
                           ("scripts/colab_remote.py", b"# exact uploaded remote code")]:
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    digest = hashlib.sha256((saved / "source.tar.gz").read_bytes()).hexdigest()
    colab_run_docs.update_journal(tmp_path, "session", drive_backed=True, stop_after_success=True,
                                state="needs_recovery", source_sha256=digest)
    monkeypatch.setattr(run_on_colab, "ROOT", tmp_path)
    monkeypatch.setattr(colab_connection, "prevent_idle_sleep", nullcontext)
    monkeypatch.setattr(colab_connection, "cli_python", lambda _: sys.executable)
    events = []

    def listing(command, **kwargs):
        events.append(command)
        assert command[1] == "ls" and kwargs["timeout"] == 30
        return SimpleNamespace(returncode=0, stdout="mdlARC-source.tar.gz\n", stderr="")

    def download(command, **kwargs):
        events.append(command)
        assert command[1] == "download" and kwargs["timeout"] == 60
        shutil.copyfile(saved / "source.tar.gz", command[-1])

    monkeypatch.setattr(run_on_colab.subprocess, "run", listing)
    monkeypatch.setattr(run_on_colab, "run_command", download)
    return tmp_path, saved, events


def resume():
    return run_on_colab.resume_unsubmitted_session("colab", "session",
        expected_endpoint="original-vm", expected_kernel="original-kernel")


def test_resume_uses_archived_code_spec_and_original_vm_without_upload_or_auth(unsubmitted, monkeypatch):
    root, saved, events = unsubmitted
    (root / "specs/original.json").write_text('{"run_name":"changed","defaults":{"epochs":1}}')

    def watch(colab, session, *, initial_command, stop_after_success):
        command = initial_command
        assert Path(command[command.index("--file") + 1]).read_bytes() == b"# exact uploaded remote code"
        assert command[command.index("--expected-endpoint") + 1] == "original-vm"
        assert command[command.index("--expected-kernel-id") + 1] == "original-kernel"
        assert "MDLARC_SPEC=specs/original.json" in command
        assert "MDLARC_MIN_GPU_MEMORY_GB=75" in command
        assert "MDLARC_DRIVE_AUTH=shared" in command
        assert "MDLARC_PRETRAIN_DRIVE_ID=" in command
        with pytest.raises(RuntimeError, match="Another watcher"):
            with colab_connection.session_lock(root / "results/test-branch/colab_runs/_active/session.watch.lock"):
                pass
        return 0

    monkeypatch.setattr(run_on_colab, "_watch_existing_session", watch)
    assert resume() == 0
    assert [call[1] for call in events] == ["ls", "download"]
    assert json.loads((saved / "spec.json").read_text())["defaults"]["epochs"] == 330


@pytest.mark.parametrize("submitted", [False, True])
def test_any_existing_execution_identity_prevents_resume_before_network(unsubmitted, monkeypatch, submitted):
    root, saved, events = unsubmitted
    run_on_colab.connection_identity("session").write_text(json.dumps({"submitted": submitted}))
    with pytest.raises(ValueError, match="Execution identity exists"):
        resume()
    assert events == []


@pytest.mark.parametrize("evidence", ["progress", "submission_status"])
def test_missing_identity_does_not_erase_other_submission_evidence(unsubmitted, evidence):
    root, saved, events = unsubmitted
    if evidence == "progress":
        colab_run_docs.update_journal(root, "session", stage="pretrain", epoch=1)
    else:
        run_on_colab.connection_identity("session").with_suffix(".status.json").write_text('{"submitted":true}')
    with pytest.raises(ValueError, match="Previous execution evidence"):
        resume()
    assert events == []


@pytest.mark.parametrize("changed", ["source", "spec", "checkpoint"])
def test_changed_provenance_or_continuation_is_rejected_before_network(unsubmitted, changed):
    root, saved, events = unsubmitted
    if changed == "source":
        (saved / "source.tar.gz").write_bytes(b"changed")
    else:
        spec = json.loads((saved / "spec.json").read_text())
        spec["checkpoint" if changed == "checkpoint" else "run_name"] = "other"
        (saved / "spec.json").write_text(json.dumps(spec))
        if changed == "checkpoint":
            colab_run_docs.update_journal(root, "session",
                spec_sha256=hashlib.sha256((saved / "spec.json").read_bytes()).hexdigest())
    with pytest.raises(ValueError):
        resume()
    assert events == []


@pytest.mark.parametrize("names", ["mdlARC-source.tar.gz\nmdlARC/\n",
                                   "mdlARC-source.tar.gz\nmdlARC-results-manifest.json\n",
                                   "mdlARC-source.tar.gz\nmdlARC-results.part0000\n", ""])
def test_remote_execution_evidence_or_missing_archive_prevents_submission(unsubmitted, monkeypatch, names):
    monkeypatch.setattr(run_on_colab.subprocess, "run", lambda *a, **k:
        SimpleNamespace(returncode=0, stdout=names))
    monkeypatch.setattr(run_on_colab, "_watch_existing_session", lambda *a, **k: pytest.fail("No submission"))
    with pytest.raises(ValueError, match="execution evidence"):
        resume()


def test_different_remote_archive_never_reaches_submission(unsubmitted, monkeypatch):
    monkeypatch.setattr(run_on_colab, "run_command", lambda command, **kwargs:
        Path(command[-1]).write_bytes(b"different uploaded source"))
    monkeypatch.setattr(run_on_colab, "_watch_existing_session", lambda *a, **k: pytest.fail("No submission"))
    with pytest.raises(ValueError, match="VM source archive differs"):
        resume()


def test_resume_checks_existing_kernel_and_never_posts_or_changes_alias(monkeypatch):
    backend = colab_connection.ColabBackend.__new__(colab_connection.ColabBackend)
    backend.session = "session"
    local = SimpleNamespace(endpoint="original-vm", kernel_id="original-kernel")
    backend.local_session = lambda: local
    backend.assignment = lambda endpoint: object()
    calls = []
    failures = [requests.ReadTimeout(), None]

    def request(proxy, method, path):
        calls.append((method, path))
        error = failures.pop(0)
        if error:
            raise error
        return {"id": "original-kernel", "execution_state": "idle"}

    backend.request = request
    monkeypatch.setattr(colab_connection.time, "sleep", lambda _: None)
    result = backend.create_identity("original", "hash", expected_endpoint="original-vm", expected_kernel="original-kernel")
    assert result["submitted"] is False and result["kernel_id"] == "original-kernel"
    assert calls == [("GET", "/api/kernels/original-kernel")] * 2
    assert local.kernel_id == "original-kernel"
    local.endpoint = "replacement-vm"
    with pytest.raises(ValueError, match="VM or kernel changed"):
        backend.create_identity("original", "hash", expected_endpoint="original-vm", expected_kernel="original-kernel")


def test_busy_or_changed_kernel_is_never_submitted():
    backend = colab_connection.ColabBackend.__new__(colab_connection.ColabBackend)
    backend.local_session = lambda: SimpleNamespace(endpoint="original-vm", kernel_id="original-kernel")
    backend.assignment = lambda endpoint: object()
    backend.existing_kernel = lambda *a: {"id": "original-kernel", "execution_state": "busy"}
    with pytest.raises(ValueError, match="not idle"):
        backend.create_identity("original", "hash", expected_endpoint="original-vm", expected_kernel="original-kernel")


@pytest.mark.parametrize("backup_fails", [False, True])
def test_recovery_records_must_verify_before_shutdown(unsubmitted, monkeypatch, backup_fails):
    root, saved, events = unsubmitted
    monkeypatch.setattr(run_on_colab, "watch_remote_run", lambda *a, **k: {"finalized": True})
    monkeypatch.setattr(run_on_colab, "recover_results", lambda *a:
        {"ok": True, "backups_complete": True, "drive_folder_id": "folder"})
    def backup(*args):
        events.append("record_backup")
        if backup_fails:
            raise RuntimeError("record verification failed")
    monkeypatch.setattr(run_on_colab, "sync_records_to_drive", backup)
    monkeypatch.setattr(run_on_colab, "run_command", lambda command, **kwargs: events.append(command[1]))
    if backup_fails:
        with pytest.raises(RuntimeError, match="record verification failed"):
            run_on_colab._watch_existing_session("colab", "session", initial_command=["initial-submission"])
        assert events == ["record_backup"]  # In particular, no stop command.
    else:
        assert run_on_colab._watch_existing_session("colab", "session", initial_command=["initial-submission"]) == 0
        assert events == ["record_backup", "stop"]
    journal = json.loads((root / "results/test-branch/colab_runs/_active/session.json").read_text())
    assert journal["state"] == ("needs_recovery" if backup_fails else "completed")


def test_cached_resume_wrapper_never_reads_or_installs_credentials(monkeypatch):
    from scripts.colab_oauth import launch
    arguments = ["--resume-unsubmitted", "session", "--expected-endpoint", "original-vm",
                 "--expected-kernel-id", "original-kernel"]
    delegated = []
    monkeypatch.setattr(launch, "delegate", lambda args: delegated.append(args) or 0)
    monkeypatch.setattr(launch, "authorize_session", lambda *args: pytest.fail("No reauthorization"))
    assert launch.main(arguments) == 0
    assert delegated == [arguments]
