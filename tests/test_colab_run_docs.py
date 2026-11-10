import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from scripts import colab_run_docs, run_on_colab


def test_checkpoint_adaptation_scores_and_memory_are_reported(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "validation_scores.json").write_text(json.dumps({"tasks": 40, "queries": 56,
        "top1": {"percent": 25.0}, "top2": {"percent": 32.5}}))
    (run / "training_stats.json").write_text(json.dumps({"seconds": 100,
        "trainable_parameters": 9437184, "peak_allocated_gib": 17.1}))
    lines = "\n".join(colab_run_docs._metric_lines(run, tmp_path))
    assert "top1=25%" in lines and "top2=32.5%" in lines
    assert "40 tasks, 56 queries" in lines and "peak_allocated_gib=17.1" in lines


def test_prompt_collects_two_answers_and_cancel_stops_launch(monkeypatch, tmp_path):
    questions = []
    replies = iter([
        SimpleNamespace(returncode=0, stdout="Check lower memory use\n", stderr=""),
        SimpleNamespace(returncode=0, stdout="Batch size 16\n", stderr=""),
    ])

    def fake_run(command, **kwargs):
        questions.append(command[-1])
        return next(replies)

    monkeypatch.setattr(colab_run_docs.sys, "platform", "darwin")
    monkeypatch.setattr(colab_run_docs.shutil, "which", lambda name: "/usr/bin/osascript")
    monkeypatch.setattr(colab_run_docs.subprocess, "run", fake_run)
    assert colab_run_docs.prompt_for_notes("arc2") == ("Check lower memory use", "Batch size 16")
    assert "intent" in questions[0]
    assert "changed" in questions[1]

    root = tmp_path / "repo"
    (root / "specs").mkdir(parents=True)
    (root / "specs" / "test.json").write_text('{"run_name":"test"}')
    monkeypatch.setattr(run_on_colab, "ROOT", root)
    monkeypatch.setattr(run_on_colab.colab_run_docs, "prompt_for_notes",
                        lambda name: (_ for _ in ()).throw(RuntimeError("cancelled")))
    calls = []
    monkeypatch.setattr(run_on_colab, "run_command", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr(sys, "argv", ["run_on_colab.py", "--config", "specs/test.json",
                                   "--session", "new-session"])
    with pytest.raises(RuntimeError, match="cancelled"):
        run_on_colab.main()
    assert calls == []
    assert not (root / "results" / "test-branch" / "experiments").exists()


def test_generated_report_uses_saved_facts_and_preserves_manual_note(tmp_path):
    root = tmp_path
    session = "mdlarc-test-123456"
    spec = root / "specs" / "test.json"
    spec.parent.mkdir()
    spec.write_text('{"run_name":"arc2","pretrain":{"epochs":84,"checkpoint_epochs":[42]}}')
    note = colab_run_docs.create_note(root, session, "arc2", "Check memory", "Batch 16")
    colab_run_docs.start_journal(root, session, config="specs/test.json", run_name="arc2",
                                 gpu="A100", high_mem=True, min_gpu_memory_gb=75, spec_path=spec)
    manual = note.read_text()
    archived_spec = root / "results" / "test-branch" / "colab_runs" / session / "specs" / "test.json"
    archived_spec.parent.mkdir(parents=True)
    archived_spec.write_bytes(spec.read_bytes())
    run_dir = root / "results" / "test-branch" / "colab_runs" / session / "runs" / "arc2"
    run_dir.mkdir(parents=True)
    (run_dir / "experiment.json").write_text(json.dumps({
        "spec": json.loads(spec.read_text()), "source_sha256": "a" * 64,
        "gpu": "A100", "gpu_memory_gib": 79.25,
    }))
    (run_dir / "console.log").write_text(
        "Starting training stage 'pretrain' in an isolated process...\n"
        "Epoch 1/84\n"
        "torch.OutOfMemoryError: CUDA out of memory\n"
        "subprocess.CalledProcessError: wrapper error\n"
    )
    (run_dir / "checkpoint_backups.json").write_text(json.dumps({"complete": True, "files": []}))
    (root / "results" / "test-branch" / "colab_runs" / session / "run_status.json").write_text(json.dumps({
        "ok": False, "backups_complete": True, "error": "training failed",
    }))
    colab_run_docs.render_index(root)
    report = (root / "results" / "test-branch" / "experiments" / "generated" / f"{session}.md").read_text()
    assert "**State:** failed" in report
    assert "**Last observed epoch:** 1/84" in report
    assert "torch.OutOfMemoryError" in report
    assert "Checkpoint uploads:** 0 recorded" in report
    assert "Uploaded source SHA-256:** " + "a" * 64 in report
    assert "Archived selected spec:** [test.json]" in report
    assert note.read_text() == manual
    assert session in (root / "results" / "test-branch" / "experiments" / "INDEX.md").read_text()


def test_live_progress_and_failed_refresh_remain_uncertain(monkeypatch, tmp_path):
    session = "mdlarc-test-123456"
    spec = tmp_path / "specs" / "test.json"
    spec.parent.mkdir()
    spec.write_text("{}")
    colab_run_docs.start_journal(tmp_path, session, config="specs/test.json", run_name="test",
                                 gpu="A100", high_mem=True, min_gpu_memory_gb=75, spec_path=spec)
    colab_run_docs.update_journal(tmp_path, session, state="running")
    assert colab_run_docs.observe_output(tmp_path, session, "Starting training stage 'pretrain' in an isolated process...")
    assert colab_run_docs.observe_output(tmp_path, session, "Epoch 3/84")
    assert not colab_run_docs.observe_output(tmp_path, session, "epoch=3 step=10 loss=1.0")
    monkeypatch.setattr(colab_run_docs.shutil, "which", lambda name: "/usr/local/bin/colab")
    monkeypatch.setattr(colab_run_docs.subprocess, "run",
                        lambda *args, **kwargs: SimpleNamespace(returncode=1))
    journal = colab_run_docs.refresh_session(tmp_path, session)
    assert journal["state"] == "running"
    assert journal["session_check"]["result"].startswith("unknown")
    report = (tmp_path / "results" / "test-branch" / "experiments" / "generated" / f"{session}.md").read_text()
    assert "**Last observed epoch:** 3/84" in report
    assert "may be stale" in report


def test_colab_command_streams_output_to_local_progress_log(tmp_path):
    log = tmp_path / "live.log"
    observed = []
    run_on_colab.run_command(
        [sys.executable, "-u", "-c", "print('Epoch 1/2')"],
        log_path=log, on_output=observed.append,
    )
    assert log.read_text() == "Epoch 1/2\n"
    assert observed == ["Epoch 1/2\n"]


def test_refresh_does_not_treat_zero_exit_missing_session_as_visible(monkeypatch, tmp_path):
    monkeypatch.setattr(colab_run_docs.shutil, "which", lambda name: "colab")
    monkeypatch.setattr(colab_run_docs.subprocess, "run", lambda *args, **kwargs:
                        SimpleNamespace(returncode=0, stdout="[colab] Session 'gone' not found.\n", stderr=""))
    journal = colab_run_docs.refresh_session(tmp_path, "gone")
    assert journal["session_check"]["result"] == "session not visible to CLI"


def test_connection_records_are_not_reported_as_extra_experiments(tmp_path):
    active = tmp_path / "results" / "test-branch" / "colab_runs/_active"
    active.mkdir(parents=True)
    (active / "real.json").write_text(json.dumps({"session": "real", "run_name": "run", "state": "running"}))
    (active / "real.connection.json").write_text(json.dumps({"session": "real", "run_name": "run", "kernel_id": "kernel"}))
    (active / "real.connection.status.json").write_text(json.dumps({"state": "connected"}))
    (active / "real.check-ready.json").write_text(json.dumps({"session": "real", "finalized": True}))
    colab_run_docs.render_index(tmp_path)
    assert colab_run_docs.known_sessions(tmp_path) == ["real"]
    report = (tmp_path / "results" / "test-branch" / "experiments/generated/real.md").read_text()
    assert "Saved kernel/execution identity" in report and "Local connection:** connected" in report
    assert len(list((tmp_path / "results" / "test-branch" / "experiments/generated").glob("*.md"))) == 1
