from pathlib import Path
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from torch import nn


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import _FractionalLambdaLR, train_one_epoch  # noqa: E402
import train  # noqa: E402


class _ScalarLossModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        input_ids: torch.Tensor,
        example_ids: torch.Tensor,
        **_: object,
    ) -> dict[str, torch.Tensor]:
        del input_ids, example_ids
        return {"loss": self.weight, "output_loss": self.weight}


def _single_batch() -> dict[str, torch.Tensor | bool]:
    return {
        "input_ids": torch.ones((1, 1), dtype=torch.long),
        "example_ids": torch.zeros(1, dtype=torch.int32),
        "sep_indices": torch.zeros(1, dtype=torch.int16),
        "positions_3d": torch.zeros((1, 1, 3), dtype=torch.int32),
        "has_padding": False,
    }


def test_fractional_scheduler_applies_warmup_at_construction() -> None:
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=2.0)

    scheduler = _FractionalLambdaLR(optimizer, lambda progress: progress / 4.0)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
    assert scheduler.get_last_lr() == pytest.approx([0.0])
    assert scheduler.last_epoch == -1.0


def test_first_optimizer_update_uses_first_warmup_lr() -> None:
    model = _ScalarLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = _FractionalLambdaLR(optimizer, lambda progress: progress)

    train_one_epoch(
        model=model,
        dataloader=[_single_batch()],
        optimizer=optimizer,
        device=torch.device("cpu"),
        grad_clip=0.0,
        scheduler=scheduler,
        epoch=0,
        steps_per_epoch=4,
        train_log_mode="never",
        log_location="none",
    )

    # The first update is at progress 1/4.  The broken ordering used the full
    # base LR (1.0), which would have changed the parameter from 1.0 to 0.0.
    assert model.weight.item() == pytest.approx(0.75)
    assert scheduler.get_last_lr() == pytest.approx([0.25])
    assert scheduler.last_epoch == pytest.approx(0.25)


def test_warmup_updates_follow_accumulation_boundaries() -> None:
    model = _ScalarLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = _FractionalLambdaLR(optimizer, lambda progress: progress)

    train_one_epoch(
        model, [_single_batch()] * 4, optimizer, torch.device("cpu"),
        grad_clip=0.0, gradient_accumulation_steps=2, scheduler=scheduler,
        epoch=0, train_log_mode="never", log_location="none",
    )

    # Two unit gradients are applied with LRs 0.5 and 1.0, not per minibatch.
    assert model.weight.item() == pytest.approx(-0.5)
    assert scheduler.last_epoch == pytest.approx(1.0)


def test_scheduler_resume_restores_group_lrs_and_continues_progress() -> None:
    parameters = [nn.Parameter(torch.tensor(1.0)) for _ in range(2)]
    optimizer = torch.optim.SGD([
        {"params": [parameters[0]], "lr": 2.0},
        {"params": [parameters[1]], "lr": 4.0},
    ])

    def schedule(progress):
        return [progress / 4.0, progress / 8.0]

    scheduler = _FractionalLambdaLR(optimizer, schedule)
    scheduler.step(1.5)
    saved = scheduler.state_dict()
    resumed = _FractionalLambdaLR(optimizer, schedule)
    assert resumed.get_last_lr() == [0.0, 0.0]
    resumed.load_state_dict(saved)
    assert [group["lr"] for group in optimizer.param_groups] == [0.75, 0.75]
    resumed.step(1.75)
    assert resumed.get_last_lr() == [0.875, 0.875]
    assert resumed.last_epoch == 1.75


def test_runs_overwrite_a_rolling_checkpoint_and_keep_every_fiftieth_epoch(monkeypatch, tmp_path):
    model = _ScalarLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    scheduler = _FractionalLambdaLR(optimizer, lambda progress: 1.0)
    scheduler.step(5.0)
    checkpoint = {
        "epoch": 5, "global_step": 5,
        "optimizer_name": train._optimizer_identity(optimizer),
        "optimizer_hparams": train._param_groups_snapshot(optimizer.param_groups),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    monkeypatch.setattr(train, "_build_optimizer", lambda *args: torch.optim.SGD(model.parameters(), lr=1.0))
    monkeypatch.setattr(train, "train_one_epoch", lambda **kwargs: kwargs["start_step"] + 1)
    saved = []

    def save(model, dataset, data_path, path, **kwargs):
        saved.append((path.name, kwargs["epoch"], kwargs.get("rolling", False)))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(kwargs["epoch"]))

    monkeypatch.setattr(train, "_save_checkpoint", save)
    save_path = tmp_path / "pretrain/tiny.pt"
    args = Namespace(
        epochs=120, checkpoint_mode="resume", weight_decay=0.0,
        grad_clip=0.0, warmup_pct=0.005, wsd_decay_start_pct=0.8,
        resume_warmup_pct=0.0, lr_floor=0.0001,
        train_log_mode="never", log_location="none",
        checkpoint_epochs=[], kept_checkpoint_every=50, rolling_checkpoint_every=10, save_path=save_path,
    )
    train.train_model(args, model, [_single_batch()], SimpleNamespace(), torch.device("cpu"),
                      Path("unused.json"), checkpoint=checkpoint)
    assert [entry for entry in saved if not entry[2]] == [
        ("tiny.epoch050.pt", 50, False), ("tiny.epoch100.pt", 100, False), ("tiny.pt", 120, False)]
    assert [entry[1] for entry in saved if entry[2]] == list(range(10, 120, 10))
    assert {entry[0] for entry in saved if entry[2]} == {"tiny.rolling.pt"}
    # Kept checkpoints stay; the rolling resume point is dropped once the final one exists.
    assert sorted(item.name for item in save_path.parent.iterdir()) == [
        "tiny.epoch050.pt", "tiny.epoch100.pt", "tiny.pt"]


def test_real_rolling_checkpoint_reloads_for_resume_and_skips_backup_queue(monkeypatch, tmp_path):
    import json
    import build
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    data_path = tmp_path / "train.json"
    data_path.write_text(json.dumps({"abc": {"train": [{"input": [[1]], "output": [[2]]}]}}))
    args = SimpleNamespace(
        seed=42, device="cpu", checkpoint_path=None, data_path=data_path,
        batch_size=1, num_workers=0, enable_aug=False, max_augments=1,
        d_model=32, n_heads=4, d_ff=64, n_layers=1, num_task_embedding_latents=4,
        n_predict=1, dropout=0.0, embedding_dropout=0.0, attention_dropout=0.0, FA=2,
    )
    model, dataset, _, _, _ = build.build_model_and_data(args, build_dataloader=False)
    run_dir, queue = tmp_path / "run", tmp_path / "queue"
    queue.mkdir()
    monkeypatch.setenv("MDLARC_CHECKPOINT_QUEUE", str(queue))
    monkeypatch.setenv("MDLARC_RUN_DIR", str(run_dir))
    rolling = train._rolling_checkpoint_path(run_dir / "pretrain/tiny.pt")
    for epoch in (10, 20):
        train.maybe_save_model(model, dataset, data_path, rolling, epoch=epoch, global_step=epoch, rolling=True)
    assert rolling.name == "tiny.rolling.pt"
    assert build.load_checkpoint(rolling)["epoch"] == 20
    assert list(queue.iterdir()) == []


@pytest.mark.parametrize("key", ["kept_checkpoint_every", "rolling_checkpoint_every"])
@pytest.mark.parametrize("value", [-1, True, 2.5])
def test_checkpoint_cadences_must_be_non_negative_integers(key, value):
    args = Namespace(epochs=20, checkpoint_mode="resume", save_path=Path("tiny.pt"),
                     kept_checkpoint_every=0, rolling_checkpoint_every=0)
    setattr(args, key, value)
    with pytest.raises(ValueError, match=key):
        train.train_model(args, _ScalarLossModel(), [_single_batch()], SimpleNamespace(), torch.device("cpu"),
                          Path("unused.json"), checkpoint={"epoch": 1, "global_step": 1})


@pytest.mark.parametrize("completed,total_epochs", [(67, 84), (67, 672), (12, 15), (8, 41)])
def test_resume_keeps_optimizer_state_and_uses_checkpoint_epoch_and_selected_total(monkeypatch, completed, total_epochs):
    model = _ScalarLossModel()
    original = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
    model.weight.grad = torch.tensor(1.0)
    original.step()
    scheduler = _FractionalLambdaLR(original, lambda progress: 1.0)
    scheduler.step(float(completed))
    checkpoint = {
        "epoch": completed, "global_step": completed,
        "optimizer_name": train._optimizer_identity(original),
        "optimizer_hparams": train._param_groups_snapshot(original.param_groups),
        "optimizer_state": original.state_dict(),
        "scheduler_state": scheduler.state_dict(),
    }
    restored = torch.optim.SGD(model.parameters(), lr=1.0, momentum=0.9)
    monkeypatch.setattr(train, "_build_optimizer", lambda *args: restored)
    updates = []

    def one_epoch(**kwargs):
        assert restored.state[model.weight]["momentum_buffer"].item() == 1.0
        assert kwargs["total_epochs"] == total_epochs
        assert kwargs["mtp_decay"] == [0.6, 0.85, 0.2]
        progress = float(kwargs["epoch"] + 1)
        kwargs["scheduler"].step(progress)
        updates.append((progress, restored.param_groups[0]["lr"]))
        return kwargs["start_step"] + 1

    saved = []
    monkeypatch.setattr(train, "train_one_epoch", one_epoch)
    monkeypatch.setattr(train, "_save_checkpoint", lambda *args, **kwargs: saved.append(kwargs))
    args = Namespace(
        epochs=total_epochs, checkpoint_mode="resume", weight_decay=0.0,
        grad_clip=0.0, warmup_pct=0.005, wsd_decay_start_pct=0.8,
        resume_warmup_pct=0.0, lr_floor=0.0001,
        train_log_mode="never", log_location="none",
        mtp_decay=[0.6, 0.85, 0.2],
    )
    train._train_single_model(
        args, model, [_single_batch()], SimpleNamespace(), torch.device("cpu"),
        Path("unused.json"), save_path=None, checkpoint_schedule=None, checkpoint=checkpoint,
    )
    assert len(updates) == total_epochs - completed
    assert updates[0][0] == completed + 1
    assert updates[-1][0] == total_epochs
    decay_start = total_epochs * 0.8
    first_factor = 1.0 if completed + 1 < decay_start else 0.0001 + 0.9999 * (total_epochs - completed - 1) / (total_epochs - decay_start)
    assert updates[0][1] == pytest.approx(first_factor)
    assert updates[-1][1] == pytest.approx(0.0001)
    assert saved[-1]["global_step"] == total_epochs
    assert saved[-1]["epoch"] == total_epochs


@pytest.mark.parametrize("completed,target", [(67, 67), (67, 42), (0, 0)])
def test_resume_rejects_target_at_or_below_checkpoint_epoch_before_updates(monkeypatch, completed, target):
    monkeypatch.setattr(train, "_build_optimizer", lambda *args: pytest.fail("Optimizer was constructed"))
    train_args = Namespace(epochs=target, checkpoint_mode="resume")
    with pytest.raises(ValueError, match="must be greater"):
        train._train_single_model(
            train_args, _ScalarLossModel(), [_single_batch()], SimpleNamespace(), torch.device("cpu"),
            Path("unused.json"), save_path=None, checkpoint_schedule=None,
            checkpoint={"epoch": completed, "global_step": completed},
        )


def test_optimizer_resume_compares_base_lr_and_still_rejects_changed_hparams():
    parameter = nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=2.0, momentum=0.9)
    scheduler = _FractionalLambdaLR(optimizer, lambda progress: 0.1)
    saved = {"optimizer_hparams": train._param_groups_snapshot(optimizer.param_groups)}
    fresh = torch.optim.SGD([parameter], lr=2.0, momentum=0.9)
    assert not train._optimizer_hparams_changed(fresh, saved)
    changed_lr = torch.optim.SGD([parameter], lr=3.0, momentum=0.9)
    assert train._optimizer_hparams_changed(changed_lr, saved)
    changed_momentum = torch.optim.SGD([parameter], lr=2.0, momentum=0.8)
    assert train._optimizer_hparams_changed(changed_momentum, saved)


def test_resumed_wsd_can_cool_existing_optimizer_groups_after_stable_phase():
    parameters = [nn.Parameter(torch.tensor(1.0)) for _ in range(2)]
    optimizer = torch.optim.SGD([
        {"params": [parameters[0]], "lr": 0.004},
        {"params": [parameters[1]], "lr": 0.00072},
    ])
    original_args = Namespace(epochs=330, warmup_pct=0.005,
                              wsd_decay_start_pct=0.8, lr_floor=0.0001)
    original = _FractionalLambdaLR(optimizer, train._base_lr_schedule(original_args, optimizer))
    original.step(30)
    saved = original.state_dict()
    continuation_args = Namespace(epochs=38, warmup_pct=0.005,
                                  wsd_decay_start_pct=30 / 38, lr_floor=0.0001)
    resumed = _FractionalLambdaLR(optimizer, train._base_lr_schedule(continuation_args, optimizer))
    resumed.load_state_dict(saved)
    assert resumed.get_last_lr() == pytest.approx([0.004, 0.00072])
    resumed.step(34)
    assert resumed.get_last_lr() == pytest.approx([0.0020002, 0.000360036])
    resumed.step(38)
    assert resumed.get_last_lr() == pytest.approx([0.0000004, 0.000000072])


def test_family_epoch_keeps_last_incomplete_accumulation_window():
    class FamilyLoader(list):
        batch_sampler = SimpleNamespace(preserve_all_pairs=True)
    model = _ScalarLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    step = train_one_epoch(
        model, FamilyLoader([_single_batch()] * 3), optimizer, torch.device('cpu'),
        grad_clip=0.0, gradient_accumulation_steps=2, train_log_mode='never', log_location='none',
    )
    assert step == 3
    assert model.weight.item() == pytest.approx(-1.0)  # A two-batch window, then a one-batch window.
