"""Learned-loop extensions: identity at init, decode consistency, contracts."""
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
from common import compute_positions_3d, encode_context
from research_options import loop_extension_shapes
from tinytransformer import TinyTransformer, TinyTransformerConfig
from test_group_attention import DEMOS, TARGET
from test_looping import loop_model
from test_mtp import _batch, _checkpoint, _sequences, _set_path


IDENTITY_OPTIONS = {
    "dense_k2": dict(loop_dense_mixing=True),
    "dense_k3": dict(loop_dense_mixing=True, loop_iterations=3),
    "iteration_embedding": dict(loop_iteration_embedding=True, loop_iterations=3),
    "gate_static": dict(loop_gate="static"),
    "gate_token": dict(loop_gate="token", loop_iterations=3),
    "norm_gains": dict(loop_iteration_norms=True, loop_iterations=3),
    "streams3_dynamic": dict(loop_streams=3),
    "streams4_static_k3": dict(loop_streams=4, loop_stream_gate="static", loop_iterations=3),
    "everything_k3": dict(loop_dense_mixing=True, loop_iteration_embedding=True, loop_gate="token",
                          loop_iteration_norms=True, loop_streams=4, loop_iterations=3),
}
DECODE_OPTIONS = {**IDENTITY_OPTIONS,
                  "ffn_first": dict(loop_ffn="first", loop_iterations=3),
                  "whole_stack_gates": dict(loop_gate="token", loop_start_layer=0, loop_end_layer=4)}


def _perturb(model, scale=0.3):
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith("loop_"):
                parameter.add_(scale * torch.randn(parameter.shape, generator=generator))
    return model


@pytest.mark.parametrize("name", sorted(IDENTITY_OPTIONS))
def test_extensions_start_equal_to_the_plain_loop(name):
    options = IDENTITY_OPTIONS[name]
    iterations = options.get("loop_iterations", 2)
    plain = loop_model(loop_iterations=iterations)
    plain.loop_input_scale.data.fill_(0.3)
    extended = loop_model(**options)
    extended.loop_input_scale.data.fill_(0.3)
    extra = set(extended.state_dict()) - set(plain.state_dict())
    assert extra == set(loop_extension_shapes(extended.config))
    for key, value in plain.state_dict().items():
        torch.testing.assert_close(extended.state_dict()[key], value, rtol=0, atol=0)
    for path in ("padded", "packed"):
        batch = _batch(path == "packed")
        torch.testing.assert_close(extended(**batch)["logits"], plain(**batch)["logits"], atol=2e-5, rtol=2e-5)
    result = extended(**_batch())
    result["loss"].backward()
    trainable = [p for n, p in extended.named_parameters() if n in extra]
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in trainable)


def test_closed_gates_recover_the_unlooped_model():
    looped = loop_model(loop_gate="static", loop_start_layer=0, loop_end_layer=4)
    unlooped = loop_model(loop_iterations=1, loop_start_layer=None, loop_end_layer=None)
    unlooped.load_state_dict({k: v for k, v in looped.state_dict().items() if not k.startswith("loop_")})
    with torch.no_grad():
        looped.loop_gate_bias[4:].fill_(-40.0)  # second pass over all four blocks closed
    torch.testing.assert_close(looped(**_batch())["logits"], unlooped(**_batch())["logits"], atol=1e-5, rtol=1e-5)
    with torch.no_grad():
        looped.loop_gate_bias[4:].fill_(0.0)
    assert not torch.allclose(looped(**_batch())["logits"], unlooped(**_batch())["logits"], atol=1e-3)


def test_attention_only_iterations_skip_the_shared_feed_forward():
    model = loop_model(loop_ffn="first", loop_iterations=3)
    plain = loop_model(loop_iterations=3)
    with torch.no_grad():
        for candidate in (model, plain):
            for block in candidate.blocks:
                block.ff.fc_out.weight.zero_()
                block.ff.fc_out.bias.zero_()
    torch.testing.assert_close(model(**_batch())["logits"], plain(**_batch())["logits"], rtol=0, atol=0)
    model, plain = loop_model(loop_ffn="first", loop_iterations=3), loop_model(loop_iterations=3)
    assert not torch.allclose(model(**_batch())["logits"], plain(**_batch())["logits"], atol=1e-4)


@pytest.mark.parametrize("name", sorted(DECODE_OPTIONS))
@pytest.mark.parametrize("pattern", ["dense", "pair_group"])
@pytest.mark.parametrize("block_size", [1, 3])
def test_cached_decoding_matches_teacher_forcing(name, pattern, block_size):
    model = _perturb(loop_model(attention_pattern=pattern, split_target_demo_attention=pattern == "pair_group",
                                **DECODE_OPTIONS[name])).eval()
    tokens, sep = encode_context(DEMOS, TARGET) if pattern == "pair_group" else _sequences()[1]
    tokens = tokens[None]
    positions = compute_positions_3d(tokens)
    full = model(tokens, torch.tensor([0]), sep_indices=torch.tensor([sep]), positions_3d=positions)
    prefix = model.forward_generate(tokens[:, :sep + 1], torch.tensor([0]), positions_3d=positions[:, :sep + 1])
    torch.testing.assert_close(prefix["logits"][:, 0], full["logits"][:, sep], atol=1e-5, rtol=1e-5)
    assert len(prefix["past_key_values"]) == model.cache_layer_count
    capacity = tokens.size(1) + 2
    demo_mask = torch.zeros(1, capacity, dtype=torch.bool)
    demo_mask[:, :sep + 1] = ((positions[:, :sep + 1, 2] > 0)
                              & (positions[:, :sep + 1, 2] < positions[:, sep:sep + 1, 2]))
    cache = tuple(tuple(torch.cat((tensor, tensor.new_zeros((*tensor.shape[:2], capacity - tensor.size(2),
                                                             tensor.size(-1)))), 2)
                        for tensor in pair) for pair in prefix["past_key_values"])
    for index in range(sep + 1, tokens.size(1), block_size):
        end = min(index + block_size, tokens.size(1))
        mask = torch.arange(capacity)[None] < end
        result = model.forward_verify(input_ids=tokens[:, index:end], example_ids=torch.tensor([0]),
            positions_3d=positions[:, index:end], past_key_values=cache,
            cache_position=torch.arange(index, end), attention_mask=mask, demo_attention_mask=demo_mask)
        torch.testing.assert_close(result["logits"], full["logits"][:, index:end], atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("name", ["everything_k3", "ffn_first", "whole_stack_gates"])
def test_padded_and_packed_paths_agree_and_later_tokens_cannot_leak(name, monkeypatch):
    model = _perturb(loop_model(**DECODE_OPTIONS[name])).eval()
    batch = _batch()
    original = model(**batch)["logits"]
    batch["input_ids"][1, -3] = (batch["input_ids"][1, -3] + 1) % 10
    changed = model(**batch)["logits"]
    torch.testing.assert_close(original[0], changed[0], rtol=0, atol=0)
    torch.testing.assert_close(original[1, :-3], changed[1, :-3], rtol=0, atol=0)
    results = {}
    for path in ("padded", "packed", "fallback"):
        _set_path(model, monkeypatch, path)
        results[path] = model(**_batch(path != "padded"))["loss"]
    torch.testing.assert_close(results["packed"], results["padded"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(results["fallback"], results["padded"], atol=1e-5, rtol=1e-5)


def test_extension_checkpoint_roundtrip_and_optimizer_groups(tmp_path):
    original = _perturb(loop_model(**IDENTITY_OPTIONS["everything_k3"]))
    checkpoint = _checkpoint(original)
    target = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, target)
    checkpoint = torch.load(target, weights_only=False)
    restored = TinyTransformer(build.build_checkpoint_config(checkpoint))
    build._load_model_state(restored, checkpoint, ["a", "b"])
    torch.testing.assert_close(original(**_batch())["loss"], restored(**_batch())["loss"], rtol=0, atol=0)
    groups = train._collect_param_groups(restored, include_muon=True)
    grouped = [id(p) for parameters in groups.values() for p in parameters]
    assert len(grouped) == len(set(grouped)) == len([p for p in restored.parameters() if p.requires_grad])
    no_decay = {id(p) for p in groups["no_decay"]}
    assert all(id(p) in no_decay for n, p in restored.named_parameters() if n.startswith("loop_"))
    damaged = dict(checkpoint)
    damaged["model_state"] = {k: v for k, v in checkpoint["model_state"].items() if k != "loop_stream_read"}
    with pytest.raises(ValueError, match="looping"):
        build._load_model_state(TinyTransformer(build.build_checkpoint_config(checkpoint)), damaged, ["a", "b"])


def test_compiled_extension_training_matches_eager():
    eager = _perturb(loop_model(**IDENTITY_OPTIONS["everything_k3"]))
    traced = _perturb(loop_model(**IDENTITY_OPTIONS["everything_k3"]))
    expected = eager(**_batch(True))
    actual = torch.compile(traced, backend="aot_eager")(**_batch(True))
    torch.testing.assert_close(actual["loss"], expected["loss"])
    actual["loss"].backward()
    expected["loss"].backward()
    for (name, parameter), (_, reference) in zip(traced.named_parameters(), eager.named_parameters()):
        if reference.grad is None:
            assert parameter.grad is None
            continue
        torch.testing.assert_close(parameter.grad, reference.grad, msg=name)


@pytest.mark.parametrize("key,value", [
    ("loop_gate", "learned"), ("loop_ffn", "none"), ("loop_stream_gate", "sinkhorn"),
    ("loop_gate_init", 0.0), ("loop_gate_init", 2.0), ("loop_gate_init", True),
    ("loop_streams", 1), ("loop_streams", -1), ("loop_streams", 2.0),
    ("loop_dense_mixing", 1), ("loop_iteration_norms", "yes"),
])
def test_invalid_extension_settings_fail(key, value):
    with pytest.raises(ValueError, match=key):
        TinyTransformerConfig(loop_iterations=2, loop_start_layer=1, loop_end_layer=3, **{key: value})
    with pytest.raises(ValueError, match=key):
        launch_run.build_plan({"defaults": {"loop_iterations": 2, "loop_start_layer": 2, "loop_end_layer": 6,
                                            key: value}}, ROOT / "tests/fixtures/specs/example.json")


def test_extensions_require_an_active_loop_and_reach_the_launcher_plan():
    with pytest.raises(ValueError, match="loop_streams"):
        TinyTransformerConfig(loop_streams=3)
    options = dict(loop_iterations=3, loop_start_layer=0, loop_end_layer=8, loop_gate="token",
                   loop_gate_init=0.5, loop_streams=4, loop_dense_mixing=True)
    plan = launch_run.build_plan({"defaults": options}, ROOT / "tests/fixtures/specs/example.json")
    for key, value in options.items():
        assert plan["pretrain"][key] == value
        assert launch_run.stage_summary(plan["pretrain"])[key] == value


@torch.inference_mode()
@pytest.mark.parametrize("name", ["everything_k3", "ffn_first", "whole_stack_gates"])
def test_extension_verification_compiles_fullgraph(name):
    model = _perturb(loop_model(**DECODE_OPTIONS[name])).eval()
    cache = tuple((torch.zeros(2, 4, 12, 6), torch.zeros(2, 4, 12, 6)) for _ in range(model.cache_layer_count))
    eager_cache = tuple(tuple(tensor.clone() for tensor in pair) for pair in cache)
    kwargs = dict(input_ids=torch.tensor([[1, 2], [3, 4]]), example_ids=torch.tensor([0, 1]),
                  positions_3d=torch.zeros(2, 2, 3, dtype=torch.int32),
                  attention_mask=torch.ones(2, 12, dtype=torch.bool),
                  cache_position=torch.tensor([[3, 4], [5, 6]]))
    expected = model.forward_verify(past_key_values=eager_cache, **kwargs)
    actual = torch.compile(model.forward_verify, backend="aot_eager", fullgraph=True)(past_key_values=cache, **kwargs)
    torch.testing.assert_close(actual["logits"], expected["logits"])


@pytest.mark.parametrize("path", ["padded", "packed", "fallback"])
def test_deep_supervision_exit_is_the_unlooped_network(path, monkeypatch):
    looped = loop_model(loop_exit_weight=0.5)
    unlooped = loop_model(loop_iterations=1, loop_start_layer=None, loop_end_layer=None)
    unlooped.load_state_dict({k: v for k, v in looped.state_dict().items() if not k.startswith("loop_")})
    for model in (looped, unlooped):
        _set_path(model, monkeypatch, path)
    batch = _batch(path != "padded")
    result = looped(**batch)
    torch.testing.assert_close(result["loop_exit_loss"], unlooped(**batch)["output_loss"], atol=1e-5, rtol=1e-5)
    expected = result["output_loss"] + looped.config.mtp_weight * result["mtp_loss"] + 0.5 * result["loop_exit_loss"]
    torch.testing.assert_close(result["loss"], expected)
    result["loss"].backward()
    assert looped.blocks[3].ff.fc_out.weight.grad.abs().sum() > 0
    looped.eval()
    assert looped(**batch)["loop_exit_loss"] is None


def test_deep_supervision_averages_exits_and_compiles():
    eager, traced = (loop_model(loop_exit_weight=0.3, loop_iterations=4, loop_streams=3) for _ in range(2))
    expected = eager(**_batch(True))
    actual = torch.compile(traced, backend="aot_eager")(**_batch(True))
    torch.testing.assert_close(actual["loss"], expected["loss"])
    torch.testing.assert_close(actual["loop_exit_loss"], expected["loop_exit_loss"])
    actual["loss"].backward()
    expected["loss"].backward()
    for (name, parameter), (_, reference) in zip(traced.named_parameters(), eager.named_parameters()):
        if reference.grad is not None:
            torch.testing.assert_close(parameter.grad, reference.grad, msg=name)


@pytest.mark.parametrize("exits, coda_calls", [(None, 4), ([2], 2), ([1, 3], 3), ([3, 2, 1], 4)])
def test_deep_supervision_exits_can_be_restricted_to_named_iterations(exits, coda_calls):
    """loop_model has one coda block (3): one main pass plus one pass per exit."""
    model = loop_model(loop_exit_weight=0.3, loop_iterations=4, loop_exit_iterations=exits)
    calls = []
    model.blocks[3].register_forward_hook(lambda *_: calls.append(1))
    result = model(**_batch(True))
    assert len(calls) == coda_calls
    assert result["loop_exit_loss"] is not None and torch.isfinite(result["loop_exit_loss"])
    if exits is not None and sorted(exits) == [1, 2, 3]:
        assert model.config.loop_exit_iterations == (1, 2, 3)
        reference = loop_model(loop_exit_weight=0.3, loop_iterations=4)(**_batch(True))
        torch.testing.assert_close(result["loop_exit_loss"], reference["loop_exit_loss"], rtol=0, atol=0)
    checkpoint = _checkpoint(model)
    assert build.build_checkpoint_config(checkpoint).loop_exit_iterations == model.config.loop_exit_iterations


@pytest.mark.parametrize("value", [[], [0], [4], [1, 1], "2", 2, [True], [1.0]])
def test_invalid_exit_iterations_fail(value):
    with pytest.raises(ValueError, match="loop_exit_iterations"):
        TinyTransformerConfig(loop_iterations=4, loop_start_layer=1, loop_end_layer=3,
                              loop_exit_weight=0.3, loop_exit_iterations=value)


def test_exit_iterations_need_an_active_loop_and_a_positive_weight():
    with pytest.raises(ValueError, match="loop_exit_iterations"):
        TinyTransformerConfig(loop_exit_iterations=[1])
    with pytest.raises(ValueError, match="loop_exit_weight"):
        TinyTransformerConfig(loop_iterations=4, loop_start_layer=1, loop_end_layer=3, loop_exit_iterations=[2])
    options = dict(loop_iterations=6, loop_start_layer=2, loop_end_layer=6, loop_exit_weight=0.3,
                   loop_exit_iterations=[3, 5])
    plan = launch_run.build_plan({"defaults": options}, ROOT / "tests/fixtures/specs/example.json")
    assert plan["pretrain"]["loop_exit_iterations"] == [3, 5]
    assert launch_run.stage_summary(plan["pretrain"])["loop_exit_iterations"] == [3, 5]


def test_loop_learning_rate_multiplier_isolates_only_loop_parameters():
    from argparse import Namespace
    model = loop_model(loop_streams=3, loop_dense_mixing=True)
    plain_muon, plain_adam = train._build_param_groups(model, 0.1, 0.01, 0.01, 0.01, include_muon=True)
    assert "adamw_loop" not in {g["group_name"] for g in plain_adam}
    muon, adam = train._build_param_groups(model, 0.1, 0.01, 0.01, 0.01, include_muon=True, separate_loop=True)
    groups = {g["group_name"]: g for g in train._with_adamw_group_lrs(Namespace(adamw_loop_lr_multiplier=10.0), adam, 1e-3)}
    loop_ids = {id(p) for n, p in model.named_parameters() if n.startswith("loop_") and p.requires_grad}
    assert {id(p) for p in groups["adamw_loop"]["params"]} == loop_ids
    assert groups["adamw_loop"]["lr"] == pytest.approx(1e-2) and groups["adamw_loop"]["weight_decay"] == 0.0
    assert groups["adamw_no_decay"]["lr"] == pytest.approx(1e-3)
    assert not loop_ids & {id(p) for p in groups["adamw_no_decay"]["params"]}
    assert train._loop_lr_multiplier(Namespace()) == 1.0
    with pytest.raises(ValueError, match="adamw_loop_lr_multiplier"):
        train._loop_lr_multiplier(Namespace(adamw_loop_lr_multiplier=-1.0))



def test_prefill_token_budget_reads_environment():
    import os
    import subprocess
    def budget(value):
        env = {k: v for k, v in os.environ.items() if k != "MDLARC_PREFILL_TOKEN_BUDGET"}
        if value is not None:
            env["MDLARC_PREFILL_TOKEN_BUDGET"] = value
        result = subprocess.run([sys.executable, "-c", "import evaluate; print(evaluate.PREFILL_TOKEN_BUDGET)"],
                                cwd=ROOT / "src", env=env, capture_output=True, text=True, check=True)
        return int(result.stdout.strip().splitlines()[-1])
    assert budget(None) == 524288
    assert budget("196608") == 196608
