"""Checkpoint retention preserves the latest save and asynchronous backups."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import launch_run
from scripts import colab_drive
from src.checkpoint_io import save_checkpoint_atomically
from workflows.vast import common, worker


def save(path, base, keep_last):
    save_checkpoint_atomically(path, lambda temporary: temporary.write_bytes(path.name.encode()),
                               checkpoint_base_path=base, checkpoint_keep_last=keep_last)


@pytest.mark.parametrize("keep_last", [None, 1, 2])
def test_local_retention_and_final_save(tmp_path, monkeypatch, keep_last):
    monkeypatch.delenv("MDLARC_CHECKPOINT_QUEUE", raising=False)
    base = tmp_path / "tiny.pt"
    unrelated = tmp_path / "other.epoch001.pt"
    unrelated.write_bytes(b"keep")
    paths = [tmp_path / f"tiny.epoch{epoch:03}.pt" for epoch in (10, 20, 30)]
    for path in paths:
        save(path, base, keep_last)
        assert path.exists()
    save(base, base, keep_last)
    expected = paths + [base]
    if keep_last is not None:
        expected = expected[-keep_last:]
    assert set(tmp_path.glob("tiny*.pt")) == set(expected)
    assert unrelated.read_bytes() == b"keep"


def test_failed_replacement_preserves_previous_checkpoint(tmp_path, monkeypatch):
    monkeypatch.delenv("MDLARC_CHECKPOINT_QUEUE", raising=False)
    base, previous, replacement = (tmp_path / name for name in ("tiny.pt", "tiny.epoch010.pt", "tiny.epoch020.pt"))
    save(previous, base, 1)
    def fail(temporary):
        temporary.write_bytes(b"incomplete")
        raise OSError("disk full")
    with pytest.raises(OSError, match="disk full"):
        save_checkpoint_atomically(replacement, fail, checkpoint_base_path=base, checkpoint_keep_last=1)
    assert previous.exists() and not replacement.exists()
    assert list(tmp_path.iterdir()) == [previous]


@pytest.mark.parametrize("backend", ["colab", "vast"])
@pytest.mark.parametrize("upload_fails", [False, True])
def test_remote_retention_runs_in_uploader_and_preserves_drive_history(tmp_path, monkeypatch, backend, upload_fails):
    workspace = tmp_path / "workspace"
    run_dir, queue = workspace / "code/runs/tiny", workspace / "private/queue"
    queue.mkdir(parents=True)
    monkeypatch.setenv("MDLARC_RUN_DIR", str(run_dir))
    monkeypatch.setenv("MDLARC_CHECKPOINT_QUEUE", str(queue))
    base = run_dir / "pretrain/tiny.pt"
    paths = [base.with_name(f"tiny.epoch{epoch:03}.pt") for epoch in (10, 20)] + [base]
    for path in paths:
        save(path, base, 1)
    # Saving never waits for the uploader or deletes its queued inputs.
    assert all(path.exists() for path in paths)
    uploads = []
    def upload(path, folder, name):
        if upload_fails:
            raise OSError("Drive unavailable")
        uploads.append(name)
        return {"id": name, **common.hashes(path)}
    store = SimpleNamespace(upload=upload, upload_new=upload)
    if backend == "colab":
        stop = tmp_path / "stop"
        stop.touch()
        result = colab_drive.worker(store, "folder", run_dir, queue, stop)
    else:
        (workspace / "private/uploader-stop").touch()
        monkeypatch.setattr(worker, "store_for", lambda *args: store)
        result = worker.uploader(workspace, {"run_name": "tiny", "drive_folder_id": "folder"})
    ledger = json.loads((run_dir / "checkpoint_backups.json").read_text())
    assert result == int(upload_fails)
    if upload_fails:
        assert all(path.exists() for path in paths)
        assert not ledger["complete"]
    else:
        assert len(uploads) == len(paths) == len(ledger["files"])
        assert list(base.parent.glob("*.pt")) == [base]
        assert ledger["complete"] and worker.verify_checkpoint_ledger(run_dir, ledger)
        assert sum(bool(entry.get("local_deleted")) for entry in ledger["files"]) == 2
        base.unlink()
        assert not worker.verify_checkpoint_ledger(run_dir, ledger)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1"])
@pytest.mark.parametrize("location", ["defaults", "pretrain", "finetune_defaults", "finetunes"])
def test_specs_reject_invalid_retention(value, location):
    value_config = {"checkpoint_keep_last": value}
    spec = {location: {"validation": value_config} if location == "finetunes" else value_config}
    with pytest.raises(ValueError, match="checkpoint_keep_last"):
        launch_run.build_plan(spec, Path("spec.json"))


def test_retention_inheritance_and_development_recipe():
    spec_path = Path(__file__).resolve().parents[1] / "specs/arc2_family_development_large_default.json"
    plan = launch_run.build_plan(launch_run.load_spec(spec_path), spec_path)
    assert plan["pretrain"]["checkpoint_epochs"] == 10
    assert plan["pretrain"]["checkpoint_keep_last"] == 1
    assert launch_run.preview_plan(plan)["pretrain"]["checkpoint_keep_last"] == 1
    plan = launch_run.build_plan({"defaults": {"checkpoint_keep_last": 2},
                                 "finetunes": {"validation": {"checkpoint_keep_last": None}}}, Path("spec.json"))
    assert plan["pretrain"]["checkpoint_keep_last"] == 2
    assert plan["finetunes"]["validation"]["checkpoint_keep_last"] is None


def test_eval_rejects_a_pruned_checkpoint():
    with pytest.raises(ValueError, match="will be removed"):
        launch_run.build_plan({"defaults": {"checkpoint_epochs": 10, "checkpoint_keep_last": 1},
                               "evals": {"old": {"checkpoint": "pretrain.epoch.10"}}}, Path("spec.json"))


@pytest.mark.parametrize("save_final", [False, True])
def test_sparse_uploads_preserve_scheduled_and_latest_checkpoints(tmp_path, monkeypatch, save_final):
    workspace = tmp_path / "workspace"
    run_dir, queue = workspace / "code/runs/tiny", workspace / "private/queue"
    queue.mkdir(parents=True)
    monkeypatch.setenv("MDLARC_RUN_DIR", str(run_dir))
    monkeypatch.setenv("MDLARC_CHECKPOINT_QUEUE", str(queue))
    base = run_dir / "pretrain/tiny.pt"
    for epoch in (80, 90, 120, 130, 140):
        save(base.with_name(f"tiny.epoch{epoch:03}.pt"), base, 1)
    if save_final:
        save(base, base, 1)
    uploaded = []
    def upload(path, folder, name):
        uploaded.append(name)
        return {"id": name, **common.hashes(path)}
    monkeypatch.setattr(worker, "store_for", lambda *args: SimpleNamespace(upload=upload))
    (workspace / "private/uploader-stop").touch()
    result = worker.uploader(workspace, {"run_name": "tiny", "drive_folder_id": "folder",
                                        "policy": {"checkpoint_upload_every_epochs": 60}})
    assert result == 0
    assert set(uploaded) == {"pretrain__tiny.epoch120.pt",
                             "pretrain__tiny.pt" if save_final else "pretrain__tiny.epoch140.pt"}
    expected = {base} if save_final else {base.with_name("tiny.epoch120.pt"), base.with_name("tiny.epoch140.pt")}
    assert set(base.parent.glob("*.pt")) == expected
    ledger = json.loads((run_dir / "checkpoint_backups.json").read_text())
    assert ledger["complete"] and worker.verify_checkpoint_ledger(run_dir, ledger)


def test_training_cadence_prunes_only_after_success_and_saves_final_once(tmp_path, monkeypatch):
    from argparse import Namespace
    import torch
    import train

    monkeypatch.delenv("MDLARC_CHECKPOINT_QUEUE", raising=False)
    model = torch.nn.Linear(1, 1)
    monkeypatch.setattr(train, "_build_optimizer", lambda *args: torch.optim.SGD(model.parameters(), lr=1.0))
    seen = []
    def one_epoch(**kwargs):
        previous = tmp_path / f"tiny.epoch{kwargs['epoch'] // 10 * 10:02}.pt"
        if kwargs["epoch"] >= 10:
            assert previous.exists()
        return kwargs["start_step"] + 1
    def save_model(model, dataset, data_path, save_path, **kwargs):
        seen.append((save_path.name, kwargs["epoch"]))
        save(save_path, kwargs["checkpoint_base_path"], kwargs["checkpoint_keep_last"])
    monkeypatch.setattr(train, "train_one_epoch", one_epoch)
    monkeypatch.setattr(train, "maybe_save_model", save_model)
    args = Namespace(**{**launch_run.BASE_ARGS, "epochs": 30, "checkpoint_epochs": 10,
                         "checkpoint_keep_last": 1, "save_path": tmp_path / "tiny.pt",
                         # Retention alone: the 50/10 kept/rolling cadences are tested separately.
                         "kept_checkpoint_every": 0, "rolling_checkpoint_every": 0,
                         "log_location": "none"})
    train.train_model(args, model, [None], SimpleNamespace(), torch.device("cpu"), tmp_path / "data.json")
    assert seen == [("tiny.epoch10.pt", 10), ("tiny.epoch20.pt", 20), ("tiny.pt", 30)]
    assert list(tmp_path.glob("*.pt")) == [tmp_path / "tiny.pt"]
