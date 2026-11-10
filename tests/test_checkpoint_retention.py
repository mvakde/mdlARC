"""Checkpoint retention preserves the latest save and asynchronous backups."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import launch_run
from src.checkpoint_io import save_checkpoint_atomically


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


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "1"])
@pytest.mark.parametrize("location", ["defaults", "pretrain", "finetune_defaults", "finetunes"])
def test_specs_reject_invalid_retention(value, location):
    value_config = {"checkpoint_keep_last": value}
    spec = {location: {"validation": value_config} if location == "finetunes" else value_config}
    with pytest.raises(ValueError, match="checkpoint_keep_last"):
        launch_run.build_plan(spec, Path("spec.json"))


def test_retention_inheritance():
    plan = launch_run.build_plan({"defaults": {"checkpoint_epochs": 10, "checkpoint_keep_last": 1}}, Path("spec.json"))
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
