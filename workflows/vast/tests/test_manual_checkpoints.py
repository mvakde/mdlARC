"""Manual checkpoints stay on the VM and never enter automatic Drive uploads."""

import json
from types import SimpleNamespace

import pytest

from src.checkpoint_io import save_checkpoint_atomically
from workflows.vast import common, policy, worker


def test_manual_mode_preserves_local_retention_and_uploads_no_checkpoint(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    private = workspace / "private"
    (private / "queue").mkdir(parents=True)
    plan = {"branch": "scaling-ladder", "run_name": "tiny", "drive_folder_id": "folder",
            "policy": {"checkpoint_upload_mode": "manual"}}
    monkeypatch.setenv("MDLARC_CHECKPOINT_QUEUE", "inherited-queue-must-not-be-used")
    env = worker.clean_training_env(plan, private)
    assert "MDLARC_CHECKPOINT_QUEUE" not in env
    monkeypatch.delenv("MDLARC_CHECKPOINT_QUEUE")
    run_dir = workspace / "code/runs/tiny"
    base = run_dir / "pretrain/tiny.pt"
    for path in [base.with_name("tiny.epoch010.pt"), base.with_name("tiny.epoch020.pt"), base]:
        save_checkpoint_atomically(path, lambda temporary: temporary.write_bytes(b"model"),
                                   checkpoint_base_path=base, checkpoint_keep_last=1)
    assert list(base.parent.glob("*.pt")) == [base]
    # Even a leftover event or final recovery must not implicitly upload a model.
    common.atomic_json(private / "queue/stale.json", {"path": "pretrain/tiny.pt", "size": 5})
    (private / "uploader-stop").touch()
    uploads = []
    monkeypatch.setattr(worker, "store_for", lambda *args: SimpleNamespace(
        upload=lambda *values: uploads.append(values) or pytest.fail("No automatic checkpoint upload")))
    assert worker.uploader(workspace, plan) == 0
    ledger = json.loads((run_dir / "checkpoint_backups.json").read_text())
    assert ledger["upload_mode"] == "manual"
    assert ledger["files"] == uploads == []
    assert ledger["complete"] is False
    assert ledger["local_checkpoint_paths"] == ["pretrain/tiny.pt"]
    assert base.exists()
    assert worker.cleanup_action({"ok": True, "backups_complete": False},
                                 {"checkpoint_upload_mode": "manual", "success_action": "destroy"}) == "keep"


def test_checkpoint_upload_mode_is_explicit_and_validated():
    assert policy.validate({})["checkpoint_upload_mode"] == "automatic"
    assert policy.validate({"checkpoint_upload_mode": "manual"})["checkpoint_upload_mode"] == "manual"
    with pytest.raises(ValueError, match="checkpoint_upload_mode"):
        policy.validate({"checkpoint_upload_mode": "sometimes"})
