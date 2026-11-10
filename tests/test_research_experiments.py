"""Experiment isolation, checkpoint contracts, launcher validation and optimizer groups."""
from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
import build
import launch_run
import train
from research_options import RESEARCH_DEFAULTS
from tinytransformer import TinyTransformer, TinyTransformerConfig
from test_looping import loop_model
from test_next_latent import latent_model
from test_mtp import _batch, _checkpoint


@pytest.mark.parametrize("factory", [loop_model, latent_model])
def test_checkpoint_roundtrip_preserves_saved_computation_and_optimizer_parameters(factory, tmp_path):
    original = factory()
    original(**_batch())["loss"].backward()
    optimizer = torch.optim.AdamW(original.parameters(), lr=.001)
    optimizer.step()
    checkpoint = _checkpoint(original)
    checkpoint["optimizer_state"] = optimizer.state_dict()
    target = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, target)
    checkpoint = torch.load(target, weights_only=False)
    restored = TinyTransformer(build.build_checkpoint_config(checkpoint))
    build._load_model_state(restored, checkpoint, ["a", "b"])
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=.001)
    restored_optimizer.load_state_dict(checkpoint["optimizer_state"])
    torch.testing.assert_close(original(**_batch())["loss"], restored(**_batch())["loss"], rtol=0, atol=0)
    assert restored.layer_schedule == original.layer_schedule
    for model, optim in ((original, optimizer), (restored, restored_optimizer)):
        optim.zero_grad()
        model(**_batch())["loss"].backward()
        optim.step()
    for name, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)


def test_legacy_checkpoint_defaults_have_no_new_tensors_and_inherit_on_resume():
    source = latent_model(next_latent_prediction=False)
    checkpoint = _checkpoint(source)
    for key in RESEARCH_DEFAULTS:
        checkpoint["config"].pop(key)
    config = build.build_checkpoint_config(checkpoint)
    assert config.loop_iterations == 1 and not config.next_latent_prediction
    restored = TinyTransformer(config)
    build._load_model_state(restored, checkpoint, ["a", "b"])
    assert restored.state_dict().keys() == source.state_dict().keys()


@pytest.mark.parametrize("factory,override", [
    (loop_model, dict(loop_iterations=3)),
    (latent_model, dict(next_latent_prediction=False)),
    (latent_model, dict(next_latent_hidden_dim=96)),
    (latent_model, dict(next_latent_kl_weight=.5)),
    (latent_model, dict(next_latent_horizon=4)),
])
def test_resume_rejects_research_changes_and_evaluation_keeps_saved_config(factory, override):
    config = factory().config
    args = Namespace(checkpoint_mode="resume", **override)
    with pytest.raises(ValueError, match="finetune"):
        build._configure_checkpoint_mtp(config, args, is_eval=False)
    assert build._configure_checkpoint_mtp(config, args, is_eval=True) is config
    args.checkpoint_mode = "finetune"
    updated = build._configure_checkpoint_mtp(config, args, is_eval=False)
    assert all(getattr(updated, key) == value for key, value in override.items())


@pytest.mark.parametrize("kind", ["loop", "latent"])
def test_explicit_finetune_can_add_and_remove_research_options_without_hiding_corruption(kind):
    source = loop_model(loop_iterations=1) if kind == "loop" else latent_model(next_latent_prediction=False, n_predict=4)
    checkpoint = _checkpoint(source)
    options = dict(loop_iterations=2) if kind == "loop" else dict(next_latent_prediction=True, n_predict=1, mtp_weight=0)
    updated = build._configure_checkpoint_mtp(source.config, Namespace(checkpoint_mode="finetune", **options), is_eval=False)
    model = TinyTransformer(updated)
    with pytest.raises(RuntimeError, match="key"):
        build._load_model_state(model, checkpoint, ["a", "b"], allow_mtp_resize=True)
    build._load_model_state(model, checkpoint, ["a", "b"], allow_mtp_resize=True, allow_research_changes=True)
    for key, value in source.state_dict().items():
        if not key.startswith("lm_head_mtp."):
            torch.testing.assert_close(model.state_dict()[key], value)
    changed = _checkpoint(model)
    build._load_model_state(source, changed, ["a", "b"], allow_mtp_resize=True, allow_research_changes=True)
    key = "loop_input_scale" if kind == "loop" else "next_latent_dynamics.mlp.0.weight"
    del changed["model_state"][key]
    with pytest.raises(ValueError, match="saved configuration"):
        build._load_model_state(source, changed, ["a", "b"], allow_mtp_resize=True, allow_research_changes=True)


@pytest.mark.parametrize("key,value", [
    ("loop_iterations", True), ("loop_iterations", 0), ("loop_iterations", 2.5),
    ("loop_start_layer", -1), ("loop_end_layer", "6"), ("loop_input_injection", 1),
    ("next_latent_prediction", "true"), ("next_latent_hidden_dim", 0),
    ("next_latent_hidden_dim", True), ("next_latent_regression_weight", -1),
    ("next_latent_regression_weight", float("nan")), ("next_latent_kl_weight", float("inf")),
    ("next_latent_kl_weight", True),
    ("next_latent_horizon", 0), ("next_latent_horizon", True),
    ("next_latent_horizon", 2.5),
])
def test_invalid_research_settings_fail_before_training(key, value):
    with pytest.raises(ValueError, match=key):
        TinyTransformerConfig(**{key: value})
    with pytest.raises(ValueError, match=key):
        launch_run.build_plan({"defaults": {key: value}}, ROOT / "tests/fixtures/specs/example.json")


@pytest.mark.parametrize("options", [
    dict(loop_iterations=2),
    dict(loop_iterations=2, loop_start_layer=2, loop_end_layer=2),
    dict(loop_iterations=2, loop_start_layer=2, loop_end_layer=9),
])
def test_invalid_loop_span_rejected_by_launcher(options):
    with pytest.raises(ValueError, match="loop_"):
        launch_run.build_plan({"defaults": options}, ROOT / "tests/fixtures/specs/example.json")


def test_merged_stage_loop_bounds_are_validated():
    with pytest.raises(ValueError, match="loop_"):
        launch_run.build_plan({"defaults": dict(loop_iterations=2, loop_start_layer=2, loop_end_layer=6),
                               "pretrain": dict(n_layers=4)}, ROOT / "tests/fixtures/specs/example.json")


@pytest.mark.parametrize("factory", [loop_model, latent_model])
def test_research_parameters_are_grouped_exactly_once_with_normuon(factory):
    model = factory()
    groups = train._collect_param_groups(model, include_muon=True)
    grouped = [id(parameter) for parameters in groups.values() for parameter in parameters]
    assert len(grouped) == len(set(grouped)) == len([p for p in model.parameters() if p.requires_grad])
    if model.next_latent_dynamics is not None:
        muon = {id(p) for p in groups["muon_other"]}
        assert all(id(model.next_latent_dynamics.mlp[i].weight) in muon for i in (0, 2, 4))


@pytest.mark.parametrize("name,changes", [
    ("context_palette_loop2.json", dict(loop_iterations=2, loop_start_layer=2, loop_end_layer=6,
                                      loop_input_injection=True, next_latent_prediction=False)),
    ("context_palette_next_latent.json", dict(loop_iterations=1, next_latent_prediction=True,
        next_latent_hidden_dim=1536, next_latent_regression_weight=1.0, next_latent_kl_weight=1.0,
        n_predict=1, mtp_weight=0.0)),
])
def test_experiment_specs_preserve_baseline_protocol_except_requested_treatment(name, changes):
    baseline = json.loads((ROOT / "tests/fixtures/specs/context_palette_schedule_two_lookup.json").read_text())
    variant = json.loads((ROOT / "tests/fixtures/specs" / name).read_text())
    expected = json.loads(json.dumps(baseline))
    expected["run_name"] = variant["run_name"]
    expected["defaults"].update(changes)
    assert variant == expected
    plan = launch_run.build_plan(variant, ROOT / "tests/fixtures/specs" / name)
    for key, value in changes.items():
        assert plan["pretrain"][key] == value
        assert launch_run.stage_summary(plan["pretrain"])[key] == value


@pytest.mark.parametrize("factory", [loop_model, latent_model])
@pytest.mark.parametrize("horizon", [1, 4, 8])
def test_compiled_training_matches_eager_loss_and_gradients(factory, horizon):
    # Dynamo tracing of shared modules and the eager transition selection is
    # exercised on CPU; native CUDA/BF16 kernels still require a GPU check.
    eager, traced = factory(), factory()
    eager.config.next_latent_horizon = traced.config.next_latent_horizon = horizon
    expected = eager(**_batch())
    compiled = torch.compile(traced, backend="aot_eager")
    actual = compiled(**_batch())
    torch.testing.assert_close(actual["loss"], expected["loss"])
    actual["loss"].backward()
    expected["loss"].backward()
    for (name, parameter), (_, reference) in zip(traced.named_parameters(), eager.named_parameters()):
        torch.testing.assert_close(parameter.grad, reference.grad, msg=name)
