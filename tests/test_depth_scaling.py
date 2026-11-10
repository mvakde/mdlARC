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
from test_target_demo_attention import BRANCH_OPTIONS


def _model(**options):
    torch.manual_seed(7)
    return TinyTransformer(TinyTransformerConfig(
        d_model=24, n_heads=4, d_ff=48, n_layers=options.pop("n_layers", 16),
        num_examples=2, num_task_embedding_latents=2,
        dropout=0, embedding_dropout=0, attention_dropout=0,
        n_predict=4, **options,
    ))


@pytest.mark.parametrize("options", BRANCH_OPTIONS)
def test_optional_initialization_changes_only_residual_weights_and_preserves_rng(options):
    baseline = _model()
    baseline_rng = torch.get_rng_state()
    candidate = _model(**options)
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    scale = (2 * candidate.config.n_layers) ** -.5 if options.get("residual_init") == "depth_scaled" else 1
    reference = baseline.state_dict()
    gains = []
    for name, value in candidate.state_dict().items():
        if name.endswith((".attention.output_norm.weight", ".ff.hidden_norm.weight")):
            gains.append(name)
            assert torch.equal(value, torch.ones_like(value))
        else:
            expected = reference[name]
            if name.endswith((".attention.out_proj.weight", ".ff.fc_out.weight")):
                expected = expected * scale
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
    assert len(gains) == (2 * candidate.config.n_layers if options.get("branch_norm") == "hidden_rms" else 0)


def test_omitted_explicit_and_legacy_defaults_have_exact_outputs_and_rng():
    baseline = _model()
    explicit = _model(residual_init="xavier", branch_norm="none")
    saved = asdict(baseline.config)
    saved.pop("residual_init"); saved.pop("branch_norm")
    legacy = TinyTransformer(build.build_checkpoint_config({"config": saved}))
    legacy.load_state_dict(baseline.state_dict(), strict=True)
    for block in baseline.blocks:
        assert block.attention.output_norm is None and block.ff.hidden_norm is None
    reference = baseline(**_batch())
    for model in (explicit, legacy):
        assert model.state_dict().keys() == baseline.state_dict().keys()
        result = model(**_batch())
        torch.testing.assert_close(result["logits"], reference["logits"], rtol=0, atol=0)
        torch.testing.assert_close(result["loss"], reference["loss"], rtol=0, atol=0)


def test_branch_norm_placement_matches_independent_rms_reference(monkeypatch):
    model = _model(branch_norm="hidden_rms", n_layers=1)
    attn, ff = model.blocks[0].attention, model.blocks[0].ff
    with torch.no_grad():
        attn.output_norm.weight.copy_(torch.linspace(.5, 1.5, 24))
        ff.hidden_norm.weight.copy_(torch.linspace(.5, 1.5, 48))
    def rms(x, weight):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * weight
    hidden = torch.randn(2, 5, 24)
    head_output = torch.randn(2, 4, 5, 6)
    monkeypatch.setattr(attn, "_apply_attention", lambda *args, **kwargs: head_output)
    concatenated = head_output.transpose(1, 2).reshape(2, 5, 24)
    torch.testing.assert_close(attn(hidden), attn.out_proj(rms(concatenated, attn.output_norm.weight)))
    values, gate = ff.fc_in(hidden).chunk(2, -1)
    expected = ff.fc_out(rms(values * torch.nn.functional.silu(gate), ff.hidden_norm.weight))
    torch.testing.assert_close(ff(hidden), expected)


@pytest.mark.parametrize("options", BRANCH_OPTIONS)
def test_compiled_ffn_uses_native_rms_only_when_enabled_without_extra_casts(options):
    ff = _model(n_layers=1, **options).blocks[0].ff
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward
    hidden = torch.randn(2, 5, 24)
    actual = torch.compile(ff, backend=backend, fullgraph=True)(hidden)
    torch.testing.assert_close(actual, ff(hidden))
    assert len(graphs) == 1
    nodes = list(graphs[0].graph.nodes)
    rms_nodes = [n for n in nodes if n.op == "call_function" and n.target in (torch.nn.functional.rms_norm, torch.rms_norm)]
    assert len(rms_nodes) == (1 if options.get("branch_norm") == "hidden_rms" else 0)
    assert not any(n.op == "call_method" and n.target in ("float", "to", "contiguous", "cpu", "item") for n in nodes)


@pytest.mark.parametrize("options", BRANCH_OPTIONS[1:])
@pytest.mark.parametrize("packed", [False, True])
def test_cpu_bfloat16_autocast_keeps_branch_activations_and_finite_gradients(monkeypatch, options, packed):
    model = _model(n_layers=2, **options)
    _set_path(model, monkeypatch, "packed" if packed else "padded")
    observed = []
    for block in model.blocks:
        for norm in (block.attention.output_norm, block.ff.hidden_norm):
            if norm is not None:
                norm.register_forward_hook(lambda module, inputs, output: observed.append((inputs[0].dtype, output.dtype)))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(**_batch(packed=packed))
    assert all(before == after == torch.bfloat16 for before, after in observed)
    if options.get("branch_norm") == "hidden_rms":
        assert len(observed) == 2 * model.config.n_layers
    for key in ("loss", "output_loss", "mtp_loss", "mtp_losses"):
        assert result[key].dtype == torch.float32 and torch.isfinite(result[key]).all()
    result["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize("options", BRANCH_OPTIONS)
def test_24_layer_packed_padded_parity_and_finite_gradients(monkeypatch, options):
    padded, packed = _model(n_layers=24, **options), _model(n_layers=24, **options)
    _set_path(padded, monkeypatch, "padded")
    _set_path(packed, monkeypatch, "packed")
    reference, actual = padded(**_batch()), packed(**_batch(packed=True))
    torch.testing.assert_close(actual["loss"], reference["loss"], rtol=2e-5, atol=1e-6)
    reference["loss"].backward(); actual["loss"].backward()
    for name, param in padded.named_parameters():
        other = dict(packed.named_parameters())[name]
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        assert other.grad is not None and torch.isfinite(other.grad).all(), name
        torch.testing.assert_close(param.grad, other.grad, rtol=1e-3, atol=2e-5)


@pytest.mark.parametrize("options", BRANCH_OPTIONS)
def test_fresh_builder_checkpoint_roundtrip_inherits_settings_and_does_not_rescale(tmp_path, monkeypatch, options):
    import train
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    path = tmp_path / "tasks.json"
    path.write_text(json.dumps({name: {"train": [{"input": [[1]], "output": [[2]]}], "test": []}
                                for name in ("a", "b")}))
    args = Namespace(**{**launch_run.BASE_ARGS, "device": "cpu", "d_model": 24,
                       "n_heads": 4, "d_ff": 48, "n_layers": 2, "dropout": 0,
                       "attention_dropout": 0, "embedding_dropout": 0, "data_path": path, "checkpoint_path": None,
                       "checkpoint_mode": "resume", **options})
    model, dataset, _, _, _ = build.build_model_and_data(args, build_dataloader=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    model(**_batch())["loss"].backward(); optimizer.step()
    save_path = tmp_path / "model.pt"
    train.maybe_save_model(model, dataset, path, save_path, optimizer=optimizer, epoch=1)
    checkpoint = build.load_checkpoint(save_path)
    for key, default in (("residual_init", "xavier"), ("branch_norm", "none")):
        assert checkpoint["config"][key] == options.get(key, default)
        setattr(args, key, None)
    # Old checkpoints without either field retain identical loading behavior.
    if not options:
        checkpoint["config"].pop("residual_init"); checkpoint["config"].pop("branch_norm")
        torch.save(checkpoint, save_path)
    args.checkpoint_path = save_path
    restored, _, _, _, _ = build.build_model_and_data(args, build_dataloader=False)
    for name, weight in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], weight, rtol=0, atol=0)
    torch.testing.assert_close(restored(**_batch())["logits"], model(**_batch())["logits"], rtol=0, atol=0)


@pytest.mark.parametrize("options", BRANCH_OPTIONS)
@pytest.mark.parametrize("mode,is_eval,eval_only", [("resume", False, False), ("finetune", False, False),
                                                   ("resume", True, False), ("resume", False, True)])
def test_checkpoint_options_inherit_or_match_and_reject_changes(options, mode, is_eval, eval_only):
    config = TinyTransformerConfig(**options)
    args = Namespace(checkpoint_mode=mode, eval_only=eval_only, residual_init=None, branch_norm=None)
    assert build._configure_checkpoint_mtp(config, args, is_eval=is_eval) == config
    for key, other in (("residual_init", "xavier" if config.residual_init == "depth_scaled" else "depth_scaled"),
                       ("branch_norm", "none" if config.branch_norm == "hidden_rms" else "hidden_rms")):
        setattr(args, key, getattr(config, key))
        assert build._configure_checkpoint_mtp(config, args, is_eval=is_eval) == config
        setattr(args, key, other)
        with pytest.raises(ValueError, match=key):
            build._configure_checkpoint_mtp(config, args, is_eval=is_eval)
        setattr(args, key, None)


@pytest.mark.parametrize("include_muon", [False, True])
def test_branch_rms_gains_are_trainable_no_decay_parameters(include_muon):
    import train
    model = _model(branch_norm="hidden_rms", n_layers=2)
    muon, adamw = train._build_param_groups(model, .1, .01, .01, .01, include_muon=include_muon)
    groups = [*muon, *adamw]
    grouped = [id(p) for group in groups for p in group["params"]]
    assert len(grouped) == len(set(grouped)) == sum(p.requires_grad for p in model.parameters())
    for name, p in model.named_parameters():
        if name.endswith((".attention.output_norm.weight", ".ff.hidden_norm.weight")):
            group, = [g for g in groups if any(q is p for q in g["params"])]
            assert p.requires_grad and group["group_name"] == "adamw_no_decay" and group["weight_decay"] == 0


@pytest.mark.parametrize("key,value", [("residual_init", "zero"), ("residual_init", "bad"),
                                        ("branch_norm", "rms"), ("branch_norm", True)])
def test_invalid_architecture_options_fail_validation(key, value):
    with pytest.raises(ValueError, match=key):
        TinyTransformerConfig(**{key: value})
    with pytest.raises(ValueError, match=key):
        launch_run.build_plan({"defaults": {key: value}}, Path("invalid.json"))


@pytest.mark.parametrize("options", BRANCH_OPTIONS)
def test_architecture_option_previews_show_fresh_values_or_checkpoint_inheritance(options):
    for checkpoint in (None, "saved.pt"):
        plan = launch_run.build_plan({"defaults": options, "checkpoint": checkpoint}, Path("spec.json"))
        summary = launch_run.stage_summary(plan["pretrain"])
        for key, default in (("residual_init", "xavier"), ("branch_norm", "none")):
            assert summary[key] == options.get(key, "checkpoint" if checkpoint else default)
            for stage in (plan["pretrain"], *plan["finetunes"].values(), *plan["evals"].values()):
                assert stage[key] == options.get(key)
            for preview in launch_run.preview_plan(plan)["evals"].values():
                assert preview[key] == options.get(key, "checkpoint")


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
