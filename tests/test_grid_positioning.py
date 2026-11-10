"""R1 1k/1k positioning, spatial-base tuning and checkpoint compatibility."""
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import build
import evaluate
import launch_run
from common import compute_positions_3d, encode_pair
from tinytransformer import RotaryEmbedding3D, TinyTransformer, TinyTransformerConfig
from test_mtp import _batch, _set_path


TREATMENTS = [dict(rope_spatial_base=base) for base in (100.0, 1000.0, 10000.0)]
REMOVED_SETTINGS = dict(rope_axes="3d", rope_role_base=1000.0, grid_role_embeddings=False,
                        grid_boundary_positions="row_end", qk_norm="none", branch_norm="none",
                        residual_init="xavier")


def _model(**options):
    torch.manual_seed(7)
    return TinyTransformer(TinyTransformerConfig(
        d_model=48, n_heads=4, d_ff=96, n_layers=2, max_seq_len=144,
        num_examples=2, num_task_embedding_latents=2,
        dropout=0, embedding_dropout=0, attention_dropout=0, n_predict=4,
        **options,
    ))


def test_r1_rope_matches_independent_axis_rotation():
    rope = RotaryEmbedding3D(64)
    assert (rope.d_x, rope.d_y, rope.d_z) == (20, 20, 24)
    q, k = torch.randn(2, 3, 5, 64), torch.randn(2, 3, 5, 64)
    positions = torch.stack((torch.randint(0, 31, (2, 5)),
                             torch.randint(0, 30, (2, 5)), torch.randint(0, 3, (2, 5))), -1)
    expected = []
    for tensor in (q, k):
        parts = []
        for axis, part in enumerate(tensor.split((20, 20, 24), -1)):
            dim = part.size(-1)
            angles = positions[..., axis, None] / 1000 ** (torch.arange(0, dim, 2).float() / dim)
            real, imag = part[..., 0::2], part[..., 1::2]
            cosine, sine = angles.cos()[:, None], angles.sin()[:, None]
            parts.append(torch.stack((real * cosine - imag * sine, imag * cosine + real * sine), -1).flatten(-2))
        expected.append(torch.cat(parts, -1))
    for actual, reference in zip(rope.apply_rotary(q, k, positions), expected):
        torch.testing.assert_close(actual, reference)


def test_spatial_base_changes_only_spatial_frequencies():
    baseline, changed = RotaryEmbedding3D(64), RotaryEmbedding3D(64, spatial_base=100)
    assert not torch.equal(baseline.cos_x_cache, changed.cos_x_cache)
    assert not torch.equal(baseline.cos_y_cache, changed.cos_y_cache)
    torch.testing.assert_close(baseline.cos_z_cache, changed.cos_z_cache, rtol=0, atol=0)
    torch.testing.assert_close(baseline.sin_z_cache, changed.sin_z_cache, rtol=0, atol=0)


def test_saved_r1_checkpoint_preserves_weights_and_outputs():
    baseline = _model()
    legacy = asdict(baseline.config)
    del legacy["architecture_version"]
    legacy.update(REMOVED_SETTINGS)
    restored = TinyTransformer(build.build_checkpoint_config({"config": legacy}))
    restored.load_state_dict(baseline.state_dict(), strict=True)
    torch.testing.assert_close(baseline(**_batch())["logits"], restored(**_batch())["logits"], rtol=0, atol=0)


def test_canonical_positions_match_streamed_decoder_state():
    tokens, sep = encode_pair({'input': [[1, 2, 3], [4, 5, 6]], 'output': [[7, 8], [9, 0]]})
    raw = compute_positions_3d(tokens[None])
    state, _ = evaluate._derive_initial_state_from_prompt(tokens[None, :sep + 1], raw[:, :sep + 1])
    assert state.tolist() == [[0, 0, 2]]
    _, streamed = evaluate._speculative_grid_positions(state, tokens[None, sep + 1:])
    torch.testing.assert_close(streamed, raw[:, sep + 1:])


@pytest.mark.parametrize("treatment", TREATMENTS)
def test_packed_padded_gradients_and_saved_config_agree(monkeypatch, treatment):
    padded, packed = _model(**treatment), _model(**treatment)
    _set_path(padded, monkeypatch, "padded")
    _set_path(packed, monkeypatch, "packed")
    reference, actual = padded(**_batch()), packed(**_batch(packed=True))
    torch.testing.assert_close(actual["loss"], reference["loss"], rtol=2e-5, atol=1e-6)
    reference["loss"].backward()
    actual["loss"].backward()
    for name, param in padded.named_parameters():
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        torch.testing.assert_close(param.grad, dict(packed.named_parameters())[name].grad, rtol=2e-4, atol=2e-6)
    restored = TinyTransformer(build.build_checkpoint_config({"config": asdict(padded.config)}))
    restored.load_state_dict(padded.state_dict(), strict=True)
    torch.testing.assert_close(padded(**_batch())["logits"], restored(**_batch())["logits"])


@pytest.mark.parametrize("treatment", TREATMENTS)
@pytest.mark.parametrize("cached_conditioning", [False, True])
@torch.inference_mode()
def test_cached_decode_matches_teacher_forcing_through_row_boundaries(treatment, cached_conditioning):
    model = _model(**treatment).eval()
    tokens, sep = encode_pair({'input': [[1, 2, 3], [4, 5, 6]], 'output': [[7, 8], [9, 0]]})
    tokens = tokens[None]
    positions = compute_positions_3d(tokens)
    ids, dihedral = torch.tensor([0]), torch.tensor([0])
    full = model(input_ids=tokens, example_ids=ids, dihedral_ids=dihedral,
                 positions_3d=positions, sep_indices=torch.tensor([sep]))["logits"]
    shared = dict(example_ids=ids, dihedral_ids=dihedral)
    if cached_conditioning:
        shared.update(conditioning_embeds=model.compute_conditioning_embeds(ids, dihedral),
                      value_embeds=model.compute_value_embeds(ids))
    prefill = model.forward_generate(input_ids=tokens[:, :sep + 1], positions_3d=positions[:, :sep + 1], **shared)
    torch.testing.assert_close(prefill["logits"][:, 0], full[:, sep], rtol=1e-5, atol=1e-6)
    capacity = tokens.size(1)
    cache = tuple(tuple(torch.cat([t, t.new_zeros((*t.shape[:2], capacity - sep - 1, t.shape[-1]))], dim=2) for t in pair)
                  for pair in prefill["past_key_values"])
    for index in range(sep + 1, capacity):
        step = model.forward_generate(input_ids=tokens[:, index:index + 1], positions_3d=positions[:, index:index + 1],
                                      past_key_values=cache, cache_position=torch.tensor([index]),
                                      attention_mask=torch.arange(capacity)[None] <= index, **shared)
        torch.testing.assert_close(step["logits"][:, 0], full[:, index], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("treatment", TREATMENTS)
def test_checkpoint_positioning_is_inherited_and_cannot_change(treatment):
    config = _model(**treatment).config
    for mode in ("resume", "finetune"):
        inherited = build._configure_checkpoint_mtp(config, Namespace(checkpoint_mode=mode), is_eval=False)
        assert inherited.rope_spatial_base == config.rope_spatial_base
        different = 10000.0 if config.rope_spatial_base != 10000.0 else 100.0
        args = Namespace(checkpoint_mode=mode, rope_spatial_base=different)
        with pytest.raises(ValueError, match="fresh models only"):
            build._configure_checkpoint_mtp(config, args, is_eval=False)
        assert build._configure_checkpoint_mtp(config, args, is_eval=True) is config


@pytest.mark.parametrize("key,value", [
    ("rope_spatial_base", 1), ("rope_spatial_base", True), ("rope_spatial_base", float("nan")),
    ("rope_spatial_base", float("inf")), ("rope_spatial_base", "1000"),
])
def test_invalid_positioning_settings_rejected(key, value):
    with pytest.raises(ValueError, match=key):
        launch_run.training_overrides({key: value}, "defaults")
    with pytest.raises(ValueError, match=key):
        TinyTransformerConfig(**{key: value})


@pytest.mark.parametrize("spatial", [100., 1000., 10000.])
def test_configured_real_head_frequency_bases(spatial):
    model = TinyTransformer(TinyTransformerConfig(
        d_model=64, n_heads=1, d_ff=64, n_layers=1, num_examples=2,
        num_task_embedding_latents=2, rope_spatial_base=spatial,
    ))
    rope = model.blocks[0].attention.rope
    for axis, dim, base, length in (("x", 20, spatial, 32), ("y", 20, spatial, 32), ("z", 24, 1000., 7)):
        frequencies = base ** (-torch.arange(0, dim, 2).float() / dim)
        angles = torch.arange(length).float()[:, None] * frequencies
        torch.testing.assert_close(getattr(rope, f"cos_{axis}_cache"), angles.cos().repeat_interleave(2, -1))
        torch.testing.assert_close(getattr(rope, f"sin_{axis}_cache"), angles.sin().repeat_interleave(2, -1))
    assert rope.spatial_base == spatial


@pytest.mark.parametrize("key,value", list(REMOVED_SETTINGS.items()))
def test_removed_settings_are_no_longer_configurable(key, value):
    with pytest.raises(ValueError, match=key):
        launch_run.training_overrides({key: value}, "defaults")
    with pytest.raises(TypeError, match=key):
        TinyTransformerConfig(**{key: value})


@pytest.mark.parametrize("key,value", [
    ("rope_axes", "2d"), ("rope_role_base", 10000.), ("grid_role_embeddings", True),
    ("grid_boundary_positions", "row_start"), ("qk_norm", "rms"),
    ("branch_norm", "hidden_rms"), ("residual_init", "depth_scaled"),
])
def test_incompatible_ablation_checkpoints_are_rejected(key, value):
    config = {**asdict(_model().config), **REMOVED_SETTINGS, key: value}
    config.pop("architecture_version")
    with pytest.raises(ValueError, match=key):
        build.build_checkpoint_config({"config": config})


def test_unmarked_historical_role_base_is_not_silently_changed():
    config = asdict(_model().config)
    del config["architecture_version"]
    with pytest.raises(ValueError, match="rope_role_base"):
        build.build_checkpoint_config({"config": config})


def test_r1_model_defaults_and_serialized_checkpoint_identity():
    config = TinyTransformerConfig()
    assert config.rope_spatial_base == 1000.
    saved = asdict(config)
    assert not REMOVED_SETTINGS.keys() & saved.keys()
    assert build.build_checkpoint_config({"config": saved}).architecture_version == 2
    with pytest.raises(TypeError, match="architecture_version"):
        TinyTransformerConfig(architecture_version=1)


def test_fresh_run_defaults_keep_r1_architecture_and_objective():
    plan = launch_run.build_plan({"enable_finetune": False}, Path("spec.json"))
    defaults = plan["pretrain"]
    assert tuple(defaults[key] for key in ("d_model", "n_heads", "d_ff", "n_layers")) == (768, 12, 3072, 8)
    assert defaults["mtp_decay"] == [0.8, 0.9, 1.0]
    summary = launch_run.stage_summary(defaults)
    assert (summary["rope_spatial_base"], summary["n_predict"], summary["mtp_weight"]) == (1000., 4, .3)
    assert not REMOVED_SETTINGS.keys() & defaults.keys()
    assert not REMOVED_SETTINGS.keys() & summary.keys()


def test_shared_rope_tables_match_per_call_gathers_in_any_layout():
    """Packed training gathers cos/sin once and rotates [T, H, D] tensors with [T, 1, D] tables."""
    rope = RotaryEmbedding3D(64)
    torch.manual_seed(0)
    positions = torch.stack((torch.randint(0, 32, (2, 9)), torch.randint(0, 32, (2, 9)),
                             torch.randint(0, 7, (2, 9))), dim=-1)
    q, k = torch.randn(2, 4, 9, 64), torch.randn(2, 4, 9, 64)
    expected_q, expected_k = rope.apply_rotary(q, k, positions)
    cos, sin = rope.tables(positions.reshape(-1, 3))
    assert cos.shape == (18, 64) and sin.shape == (18, 64)
    tables = (cos.unsqueeze(1), sin.unsqueeze(1))
    packed_q, packed_k = rope.apply_rotary(q.transpose(1, 2).reshape(18, 4, 64),
                                           k.transpose(1, 2).reshape(18, 4, 64), tables=tables)
    torch.testing.assert_close(packed_q, expected_q.transpose(1, 2).reshape(18, 4, 64), rtol=0, atol=0)
    torch.testing.assert_close(packed_k, expected_k.transpose(1, 2).reshape(18, 4, 64), rtol=0, atol=0)
    with pytest.raises(ValueError, match="pos_xyz"):
        rope.apply_rotary(q, k)
