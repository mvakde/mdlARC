"""Input shortcuts, checkpoint behavior, and measured optimizer updates."""
from argparse import Namespace
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build
import launch_run
from tinytransformer import TinyTransformer, TinyTransformerConfig
from train import train_one_epoch
from test_mtp import _batch, _set_path


def _model(**options):
    torch.manual_seed(7)
    return TinyTransformer(TinyTransformerConfig(
        d_model=24, n_heads=4, d_ff=48, n_layers=16,
        num_examples=2, num_task_embedding_latents=2,
        dropout=0, embedding_dropout=0, attention_dropout=0,
        n_predict=4, **options,
    ))


def test_legacy_and_new_checkpoint_configs_and_loaded_weights():
    legacy = asdict(_model().config)
    del legacy["input_shortcuts"]
    config = build.build_checkpoint_config({"config": legacy})
    assert config.input_shortcuts == "learned"
    source = _model(input_shortcuts="fixed")
    config = build.build_checkpoint_config({"config": asdict(source.config)})
    restored = TinyTransformer(config)
    restored.load_state_dict(source.state_dict())
    for name, weight in source.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], weight)
    assert not restored.blocks[0].hidden_state_scale.requires_grad
    args = Namespace(input_shortcuts="learned", checkpoint_mode="finetune")
    with pytest.raises(ValueError, match="fresh models only"):
        build._configure_checkpoint_mtp(config, args, is_eval=False)


@pytest.mark.parametrize("path", ["padded", "packed"])
def test_16_layer_loss_gradients_and_diagnostics(monkeypatch, path):
    model = _model(input_shortcuts="fixed")
    _set_path(model, monkeypatch, path)
    batch = _batch(packed=path == "packed")
    optimizer = torch.optim.SGD(model.parameters(), lr=.001)
    log = io.StringIO()
    before = model.blocks[0].attention.out_proj.weight.detach().clone()
    train_one_epoch(model, [batch], optimizer, torch.device("cpu"),
                    grad_clip=1, log_location="file", log_handle=log,
                    depth_diagnostics_every=1, total_epochs=1, epoch=0)
    diagnostic = next(line for line in log.getvalue().splitlines() if line.startswith("depth_diagnostics"))
    stats = json.loads(diagnostic.split(" ", 2)[2])
    expected = (model.blocks[0].attention.out_proj.weight - before).norm() / before.norm()
    assert stats["0.attention.update_ratio"] == pytest.approx(expected.item(), rel=1e-5)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for block in model.blocks:
        assert block.hidden_state_scale.item() == 1
        assert block.hidden_state_scale.grad is None
        assert block.token_input_skip_scale.item() == 0
        assert block.example_input_skip_scale.item() == 0


@pytest.mark.parametrize("key,value", [("input_shortcuts", "bad"),
                                        ("depth_diagnostics_every", -1), ("depth_diagnostics_every", True)])
def test_invalid_spec_settings_rejected(key, value):
    with pytest.raises(ValueError):
        launch_run.training_overrides({key: value}, "defaults")


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, None])
def test_invalid_residual_output_lr_rejected(value):
    with pytest.raises(ValueError):
        launch_run.training_overrides({"normuon_residual_output_lr_multiplier": value}, "defaults")


def test_residual_output_lr_isolation_and_scheduler_resume(monkeypatch):
    import train
    monkeypatch.setattr(train, "_normuon_supported", lambda device: (True, ""))
    model = _model(input_shortcuts="fixed")
    args = Namespace(weight_decay=.1, normuon_lr=.0022, adamw_lr=.0034,
                     normuon_attention_lr_multiplier=.8,
                     normuon_fc_out_lr_multiplier=.6,
                     normuon_attention_lr_floor=.2)
    def optimizer():
        return train._build_optimizer(args, model, torch.device("cpu"), .01, .01, .01)
    legacy = optimizer()
    args.normuon_residual_output_lr_multiplier = 1.0
    explicit_default = optimizer()
    assert train._param_groups_snapshot(legacy.param_groups) == train._param_groups_snapshot(explicit_default.param_groups)
    assert [[id(p) for p in g["params"]] for g in legacy.param_groups] == [[id(p) for p in g["params"]] for g in explicit_default.param_groups]
    args.normuon_residual_output_lr_multiplier = .25
    reduced = optimizer()
    names = {id(p): n for n, p in model.named_parameters()}
    legacy_lrs = {id(p): g["lr"] for g in legacy.param_groups for p in g["params"]}
    reduced_params = [p for g in reduced.param_groups for p in g["params"]]
    assert len(reduced_params) == len({id(p) for p in reduced_params}) == len(legacy_lrs)
    affected = []
    for group in reduced.param_groups:
        for p in group["params"]:
            is_output = names[id(p)].endswith((".attention.out_proj.weight", ".ff.fc_out.weight"))
            assert group["lr"] == pytest.approx(legacy_lrs[id(p)] * (.25 if is_output else 1))
            if is_output:
                affected.append(names[id(p)])
        if group["group_name"] == "muon_attention_output":
            assert group["weight_decay"] == .01
            assert train._lr_floor_for_group(args, group) == .2
    assert len(affected) == 32
    # The scalar shortcut parameters are frozen and never enter the optimizer.
    assert all("skip_scale" not in names[id(p)] and "hidden_state_scale" not in names[id(p)] for p in reduced_params)
    original_lrs = [g["lr"] for g in reduced.param_groups]
    scheduler = train._FractionalLambdaLR(reduced, lambda progress: .5)
    scheduler.step(1.0)
    resumed = optimizer()
    resumed.load_state_dict(reduced.state_dict())
    resumed_scheduler = train._FractionalLambdaLR(resumed, lambda progress: .5)
    resumed_scheduler.load_state_dict(scheduler.state_dict())
    assert [g["lr"] for g in resumed.param_groups] == pytest.approx([lr * .5 for lr in original_lrs])
    checkpoint = {"optimizer_hparams": train._param_groups_snapshot(resumed.param_groups)}
    assert not train._optimizer_hparams_changed(optimizer(), checkpoint)
    checkpoint["optimizer_hparams"] = train._param_groups_snapshot(legacy.param_groups)
    assert train._optimizer_hparams_changed(optimizer(), checkpoint)
