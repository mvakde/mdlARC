"""Depth sharing, causal isolation, and independent logical KV caches."""
from dataclasses import replace
import copy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import compute_positions_3d, encode_context
from tinytransformer import TinyTransformer, TinyTransformerConfig
from test_mtp import _batch, _sequences, _set_path
from test_group_attention import DEMOS, TARGET


def loop_model(**options):
    torch.manual_seed(7)
    config = dict(d_model=24, n_heads=4, d_ff=48, n_layers=4, max_seq_len=128,
                  num_examples=2, num_task_embedding_latents=2, n_predict=4,
                  dropout=0, embedding_dropout=0, attention_dropout=0,
                  loop_iterations=2, loop_start_layer=1, loop_end_layer=3)
    config.update(options)
    return TinyTransformer(TinyTransformerConfig(**config))


@pytest.mark.parametrize("path", ["padded", "packed", "fallback"])
def test_loop_matches_independently_unrolled_blocks_and_accumulated_gradients(path, monkeypatch):
    model = loop_model(loop_input_injection=False, task_embeddings=False)
    reference = copy.deepcopy(model)
    # Build an untied, six-block reference. Each recurrent call gets a distinct
    # module initialized to the corresponding physical block's exact weights.
    physical = list(reference.blocks)
    reference.blocks = torch.nn.ModuleList([copy.deepcopy(physical[i]) for i in (0, 1, 2, 1, 2, 3)])
    reference.config = replace(reference.config, n_layers=6, loop_iterations=1,
                               loop_start_layer=None, loop_end_layer=None)
    reference.layer_schedule = tuple(range(6))
    _set_path(model, monkeypatch, path)
    _set_path(reference, monkeypatch, path)
    batch = _batch(path != "padded")
    actual, expected = model(**batch), reference(**batch)
    torch.testing.assert_close(actual["logits"], expected["logits"])
    torch.testing.assert_close(actual["loss"], expected["loss"])
    actual["loss"].backward()
    expected["loss"].backward()
    for i, block in enumerate(model.blocks):
        uses = [j for j, owner in enumerate((0, 1, 2, 1, 2, 3)) if owner == i]
        for name, parameter in block.named_parameters():
            if parameter.requires_grad:
                summed = sum(dict(reference.blocks[j].named_parameters())[name].grad for j in uses)
                torch.testing.assert_close(parameter.grad, summed, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("pattern", ["dense", "pair_group"])
@pytest.mark.parametrize("injection", [False, True])
@pytest.mark.parametrize("block_size", [1, 3])
def test_loop_cached_decoding_matches_teacher_forcing_and_has_distinct_caches(pattern, injection, block_size):
    model = loop_model(attention_pattern=pattern, split_target_demo_attention=pattern == "pair_group",
                       loop_input_injection=injection).eval()
    if injection:
        model.loop_input_scale.data.fill_(0.3)
    tokens, sep = encode_context(DEMOS, TARGET) if pattern == "pair_group" else _sequences()[1]
    tokens = tokens[None]
    positions = compute_positions_3d(tokens)
    full = model(tokens, torch.tensor([0]), sep_indices=torch.tensor([sep]), positions_3d=positions)
    prefix = model.forward_generate(tokens[:, :sep + 1], torch.tensor([0]),
                                    positions_3d=positions[:, :sep + 1])
    torch.testing.assert_close(prefix["logits"][:, 0], full["logits"][:, sep], atol=2e-6, rtol=2e-5)
    assert len(prefix["past_key_values"]) == model.cache_layer_count == 6
    assert len({tensor.data_ptr() for pair in prefix["past_key_values"] for tensor in pair}) == 12
    capacity = tokens.size(1) + 2
    demo_mask = torch.zeros(1, capacity, dtype=torch.bool)
    demo_mask[:, :sep + 1] = (positions[:, :sep + 1, 2] > 0) & (positions[:, :sep + 1, 2] < positions[:, sep:sep + 1, 2])
    cache = tuple(tuple(torch.cat((tensor, tensor.new_zeros((*tensor.shape[:2], capacity - tensor.size(2), tensor.size(-1)))), 2)
                        for tensor in pair) for pair in prefix["past_key_values"])
    for index in range(sep + 1, tokens.size(1), block_size):
        end = min(index + block_size, tokens.size(1))
        mask = torch.arange(capacity)[None] < end
        result = model.forward_verify(input_ids=tokens[:, index:end], example_ids=torch.tensor([0]),
            positions_3d=positions[:, index:end], past_key_values=cache,
            cache_position=torch.arange(index, end), attention_mask=mask, demo_attention_mask=demo_mask)
        torch.testing.assert_close(result["logits"], full["logits"][:, index:end], atol=3e-6, rtol=3e-5)


def test_one_iteration_is_exact_baseline_and_zero_injection_is_noop():
    ordinary = loop_model(loop_iterations=1)
    no_options = TinyTransformer(replace(ordinary.config, loop_start_layer=None, loop_end_layer=None))
    no_options.load_state_dict(ordinary.state_dict())
    assert "loop_input_scale" not in ordinary.state_dict()
    torch.testing.assert_close(ordinary(**_batch())["logits"], no_options(**_batch())["logits"], rtol=0, atol=0)
    gated, plain = loop_model(), loop_model(loop_input_injection=False)
    for key, value in plain.state_dict().items():
        torch.testing.assert_close(gated.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(gated(**_batch())["logits"], plain(**_batch())["logits"], rtol=0, atol=0)
    result = gated(**_batch())
    result["loss"].backward()
    assert gated.loop_input_scale.grad.abs() > 0


@pytest.mark.parametrize("pattern", ["dense", "pair_group"])
def test_later_target_tokens_cannot_change_earlier_looped_predictions(pattern):
    model = loop_model(attention_pattern=pattern, split_target_demo_attention=pattern == "pair_group").eval()
    model.loop_input_scale.data.fill_(0.4)
    batch = _batch()
    original = model(**batch)["logits"]
    batch["input_ids"][1, -3] = (batch["input_ids"][1, -3] + 1) % 10
    changed = model(**batch)["logits"]
    torch.testing.assert_close(original[0], changed[0], rtol=0, atol=0)
    torch.testing.assert_close(original[1, :-3], changed[1, :-3], rtol=0, atol=0)


def test_physical_cache_count_is_rejected_for_looped_decode():
    model = loop_model().eval()
    with pytest.raises(ValueError, match="logical layer"):
        model.forward_generate(torch.tensor([[1]]), torch.tensor([0]), positions_3d=torch.zeros(1, 1, 3),
                               past_key_values=(None,) * 4)


@torch.inference_mode()
def test_looped_block_verification_compiles_fullgraph_with_separate_logical_caches():
    model = loop_model().eval()
    model.loop_input_scale.fill_(0.3)
    cache = tuple((torch.zeros(2, 4, 12, 6), torch.zeros(2, 4, 12, 6))
                  for _ in range(model.cache_layer_count))
    eager_cache = tuple(tuple(tensor.clone() for tensor in pair) for pair in cache)
    kwargs = dict(input_ids=torch.tensor([[1, 2], [3, 4]]), example_ids=torch.tensor([0, 1]),
                  positions_3d=torch.zeros(2, 2, 3, dtype=torch.int32),
                  attention_mask=torch.ones(2, 12, dtype=torch.bool),
                  cache_position=torch.tensor([[3, 4], [5, 6]]))
    expected = model.forward_verify(past_key_values=eager_cache, **kwargs)
    compiled = torch.compile(model.forward_verify, backend="aot_eager", fullgraph=True)
    actual = compiled(past_key_values=cache, **kwargs)
    torch.testing.assert_close(actual["logits"], expected["logits"])
    for pair, reference in zip(cache, eager_cache):
        for tensor, value in zip(pair, reference):
            torch.testing.assert_close(tensor, value)
