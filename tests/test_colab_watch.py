import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from scripts import colab_run_docs, run_on_colab


@pytest.fixture
def recorded_run(monkeypatch, tmp_path):
    spec = tmp_path / "specs/test.json"
    spec.parent.mkdir()
    spec.write_text('{"run_name":"original"}')
    colab_run_docs.start_journal(tmp_path, "session", config="specs/test.json", run_name="original",
                               gpu="A100", high_mem=True, min_gpu_memory_gb=75, spec_path=spec)
    colab_run_docs.update_journal(tmp_path, "session", drive_backed=True, stop_after_success=True,
                                state="running", session_disposition="stop after success")
    saved = tmp_path / "results" / "test-branch" / "colab_runs/session/spec.json"
    saved.parent.mkdir()
    saved.write_bytes(spec.read_bytes())
    monkeypatch.setattr(run_on_colab, "ROOT", tmp_path)
    monkeypatch.setattr(run_on_colab, "session_assignment_available", lambda *args: None)
    return tmp_path


def test_watch_retries_network_loss_and_reads_new_epochs_without_executing(monkeypatch, recorded_run, capsys):
    clock = [0.0]
    monkeypatch.setattr(run_on_colab.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(run_on_colab.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    snapshots = iter([
        subprocess.TimeoutExpired("colab download", 30),
        ("Starting training stage 'pretrain' in an isolated process...\nEpoch 13/20\nEpoch 14/", None),
        ("Starting training stage 'pretrain' in an isolated process...\nEpoch 13/20\nEpoch 14/20\n", None),
        (None, {"finalized": True}),
    ])

    def snapshot(*args):
        value = next(snapshots)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(run_on_colab, "read_remote_snapshot", snapshot)
    monkeypatch.setattr(run_on_colab.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("No execution on reattach"))
    assert run_on_colab.watch_remote_run("colab", "session", "original", poll_interval=2) == {"finalized": True}
    journal = json.loads((recorded_run / "results" / "test-branch" / "colab_runs/_active/session.json").read_text())
    assert journal["epoch"] == 14
    assert journal["epoch_total"] == 20
    output = capsys.readouterr().out
    assert "retrying in 2s" in output
    assert output.count("Epoch 13/20") == 1
    assert "Epoch 14/20" in output
    assert "VM checked" in output
    report = (recorded_run / "results" / "test-branch" / "experiments/generated/session.md").read_text()
    assert "**Last observed epoch:** 14/20" in report
    assert "session.vm.log" in report


@pytest.mark.parametrize("exit_immediately", [False, True])
def test_watch_recovers_when_stream_process_hangs_or_exits(monkeypatch, recorded_run, exit_immediately, capsys):
    processes = []
    real_popen = subprocess.Popen

    def spawn(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        assert os.getpgid(process.pid) == process.pid
        processes.append(process)
        return process

    monkeypatch.setattr(run_on_colab.subprocess, "Popen", spawn)
    snapshots = iter([
        ("Epoch 13/20\n", None), ("Epoch 13/20\nEpoch 14/20\n", None),
        ("Epoch 13/20\nEpoch 14/20\nEpoch 15/20\n", None), (None, {"finalized": True}),
    ])

    def snapshot(*args):
        time.sleep(0.05)
        return next(snapshots)

    monkeypatch.setattr(run_on_colab, "read_remote_snapshot", snapshot)
    marker = recorded_run / "executions.txt"
    code = (f"from pathlib import Path; Path({str(marker)!r}).write_text('once'); "
            "print('terminal connected', flush=True); import time; "
            + ("raise SystemExit(1)" if exit_immediately else "time.sleep(30)"))
    started = time.monotonic()
    result = run_on_colab.watch_remote_run("colab", "session", "original",
                                          command=[sys.executable, "-u", "-c", code], poll_interval=0.01)
    assert result == {"finalized": True}
    assert time.monotonic() - started < 5
    assert marker.read_text() == "once"
    assert len(processes) == 1
    assert processes[0].poll() is not None
    assert "following the saved VM console" in capsys.readouterr().out
    journal = json.loads((recorded_run / "results" / "test-branch" / "colab_runs/_active/session.json").read_text())
    assert journal["epoch"] == 15


def test_snapshot_requests_are_bounded_and_reject_wrong_run(monkeypatch, recorded_run):
    calls = []

    def request(command, **kwargs):
        calls.append(command)
        assert kwargs["timeout"] == 30
        assert kwargs["stdin"] is subprocess.DEVNULL
        if command[1] == "ls":
            return SimpleNamespace(returncode=0, stdout="mdlARC-results-manifest.json\n", stderr="")
        Path(command[-1]).write_text(json.dumps({"run_name": "wrong", "session": "session"}))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_on_colab.subprocess, "run", request)
    with pytest.raises(ValueError, match="different run"):
        run_on_colab.read_remote_snapshot("colab", "session", "original")
    assert [command[1] for command in calls] == ["ls", "download"]


@pytest.mark.parametrize("quiet", [False, True])
def test_cli_error_redacts_runtime_url_and_bearer_before_output(quiet, capsys):
    secret = "fake-runtime-secret"
    code = ("import sys; sys.stderr.write('HTTPS error /api/contents?authuser=0&"
            f"colab-runtime-proxy-token={secret}&path=content Authorization: Bearer {secret}\\n'); "
            "raise SystemExit(1)")
    with pytest.raises(RuntimeError) as error:
        run_on_colab.run_command([sys.executable, "-c", code], quiet=quiet)
    output = capsys.readouterr()
    detail = str(error.value).split("\n", 1)[-1] if quiet else output.err
    assert secret not in str(error.value)
    assert secret not in detail
    assert "[REDACTED]" in detail
    assert "authuser=0" in detail and "path=content" in detail


def test_stream_log_and_callback_never_receive_runtime_credential(tmp_path, capsys):
    secret = "fake-runtime-secret"
    code = f"print('GET /content?token={secret}&path=log'); print('Epoch 13/20')"
    log = tmp_path / "stream.log"
    received = []
    run_on_colab.run_command([sys.executable, "-c", code], log_path=log, on_output=received.append)
    for text in [log.read_text(), "".join(received), capsys.readouterr().out]:
        assert secret not in text
        assert "token=[REDACTED]&path=log" in text
        assert "Epoch 13/20" in text


def test_file_poll_error_redacts_provider_url(monkeypatch, recorded_run):
    secret = "fake-runtime-secret"
    monkeypatch.setattr(run_on_colab.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stdout="", stderr=f"SSL error /content?colab-runtime-proxy-token={secret}&authuser=0"))
    with pytest.raises(RuntimeError) as error:
        run_on_colab.read_remote_snapshot("colab", "session", "original")
    assert secret not in str(error.value)
    assert "SSL error" in str(error.value)
    assert "authuser=0" in str(error.value)


@pytest.mark.parametrize("finalized,ok,backups,should_stop", [
    (True, True, True, True), (False, True, True, False),
    (True, False, True, False), (True, True, False, False),
])
def test_reattach_uses_saved_identity_and_stops_only_after_safe_recovery(
        monkeypatch, recorded_run, finalized, ok, backups, should_stop):
    # Subsequent working-spec edits do not affect the existing run.
    (recorded_run / "specs/test.json").write_text('{"run_name":"changed"}')
    events = []

    def watch(colab, session, run_name):
        assert (colab, session, run_name) == ("colab", "session", "original")
        events.append("watched")
        return {"finalized": finalized}

    def recover(*args):
        events.append("recovered")
        return {"ok": ok, "backups_complete": backups}

    monkeypatch.setattr(run_on_colab, "watch_remote_run", watch)
    monkeypatch.setattr(run_on_colab, "download_results", recover)
    monkeypatch.setattr(run_on_colab, "run_command", lambda command, **kwargs: events.append(command[1]))
    monkeypatch.setattr(run_on_colab.shutil, "which", lambda name: "colab")
    monkeypatch.setattr(sys, "argv", ["run_on_colab.py", "--watch-session", "session"])
    assert run_on_colab.main() == (0 if ok and backups else 1)
    assert events == ["watched", "recovered", *(["stop"] if should_stop else [])]
    journal = json.loads((recorded_run / "results" / "test-branch" / "colab_runs/_active/session.json").read_text())
    assert journal["session_disposition"] == ("stopped" if should_stop else "retained for recovery")


def test_reattach_retries_recovery_and_keeps_vm_on_persistent_failure(monkeypatch, recorded_run):
    calls = []
    monkeypatch.setattr(run_on_colab, "watch_remote_run", lambda *args: {"finalized": True})
    monkeypatch.setattr(run_on_colab.time, "sleep", lambda seconds: None)

    def recover(*args):
        calls.append("download")
        raise RuntimeError("offline")

    monkeypatch.setattr(run_on_colab, "download_results", recover)
    monkeypatch.setattr(run_on_colab, "run_command", lambda *args, **kwargs: pytest.fail("Must retain the VM"))
    with pytest.raises(RuntimeError, match="offline"):
        run_on_colab.watch_existing_session("colab", "session")
    assert calls == ["download"] * 3
    journal = json.loads((recorded_run / "results" / "test-branch" / "colab_runs/_active/session.json").read_text())
    assert journal["state"] == "needs_recovery"


def test_reattach_rejects_modified_saved_spec_before_network_access(monkeypatch, recorded_run):
    (recorded_run / "results" / "test-branch" / "colab_runs/session/spec.json").write_text('{"run_name":"wrong"}')
    monkeypatch.setattr(run_on_colab, "watch_remote_run", lambda *args: pytest.fail("No network access"))
    with pytest.raises(ValueError, match="does not match"):
        run_on_colab.watch_existing_session("colab", "session")


def test_gone_vm_requires_two_authenticated_absence_checks(monkeypatch, recorded_run):
    clock = [0.0]
    monkeypatch.setattr(run_on_colab.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(run_on_colab.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    checks = iter([None, False, None, False, False])
    calls = []
    def probe(*args):
        calls.append(clock[0])
        return next(checks)
    monkeypatch.setattr(run_on_colab, "session_assignment_available", probe)
    monkeypatch.setattr(run_on_colab, "read_remote_snapshot", lambda *args: (_ for _ in ()).throw(RuntimeError("file request failed")))
    with pytest.raises(run_on_colab.colab_connection.SessionUnavailable):
        run_on_colab.watch_remote_run("colab", "session", "original", poll_interval=2)
    assert calls == [0, 2, 4, 6, 8]
    journal = json.loads((recorded_run / "results" / "test-branch" / "colab_runs/_active/session.json").read_text())
    assert journal["state"] == journal["session_disposition"] == "unavailable"


def test_new_runs_reattach_the_live_kernel_without_a_job_file(monkeypatch, recorded_run):
    run_on_colab.connection_identity("session").write_text("{}")
    monkeypatch.setattr(run_on_colab.colab_connection, "cli_python", lambda _: sys.executable)
    def watch(colab, session, run_name, *, command):
        assert "colab_connection.py" in command[2]
        assert "--file" not in command
        assert "--identity" in command
        return {"finalized": True}
    monkeypatch.setattr(run_on_colab, "watch_remote_run", watch)
    monkeypatch.setattr(run_on_colab, "recover_results", lambda *args: {"ok": True, "backups_complete": True})
    monkeypatch.setattr(run_on_colab, "run_command", lambda *args, **kwargs: None)
    assert run_on_colab.watch_existing_session("colab", "session") == 0


def test_empty_file_response_cannot_hide_confirmed_missing_assignment(monkeypatch, recorded_run):
    clock = [0.0]
    monkeypatch.setattr(run_on_colab.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(run_on_colab.time, "sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(run_on_colab, "read_remote_snapshot", lambda *args: (None, None))
    monkeypatch.setattr(run_on_colab, "session_assignment_available", lambda *args: False)
    with pytest.raises(run_on_colab.colab_connection.SessionUnavailable):
        run_on_colab.watch_remote_run("colab", "session", "original", poll_interval=2)
    assert clock[0] >= 120
