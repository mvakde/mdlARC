"""All-demo grouping, information flow, permutation symmetry, and decoding."""

from argparse import Namespace
import json
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounterWithBackend

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import build
import evaluate
import launch_run
import train
import variations
from common import (
    ARCExampleDataset, END_PAIR_TOKEN_ID, GROUP_SEQUENCE_FORMAT, START_TOKEN_ID,
    collate_examples, compute_positions_3d, encode_context, encode_pair,
)
from group_attention import role_positions
from tinytransformer import TinyTransformer, TinyTransformerConfig
from test_mtp import _batch, _set_path


def _model(**options):
    torch.manual_seed(7)
    return TinyTransformer(TinyTransformerConfig(
        d_model=24, n_heads=4, d_ff=48, n_layers=3, num_examples=2,
        num_task_embedding_latents=3, n_predict=4, dropout=0,
        attention_dropout=0, attention_pattern="pair_group", **options,
    ))


DEMOS = [
    {"input": [[1, 2], [3, 4]], "output": [[5], [6]]},
    {"input": [[7]], "output": [[8, 9]]},
    {"input": [[2, 3, 4]], "output": [[6, 5], [4, 3]]},
]
TARGET = {"input": [[9, 8], [7, 6]], "output": [[1, 0], [2, 3]]}


@pytest.mark.parametrize("demo_count", [0, 1, 3, 20])
def test_grid_roles_follow_pair_role_independently_of_demo_count(demo_count):
    demos = [DEMOS[index % len(DEMOS)] for index in range(demo_count)]
    tokens, sep = encode_context(demos, TARGET)
    raw = compute_positions_3d(tokens[None])
    mapped = role_positions(raw, raw[:, sep:sep + 1, 2])
    expected = [0]
    for index, pair in enumerate([*demos, TARGET]):
        single, single_sep = encode_pair(pair)
        input_role, output_role = (3, 4) if index == demo_count else (1, 2)
        expected.extend([input_role] * single_sep)
        expected.extend([output_role] * (len(single) - single_sep - 1))
    assert mapped[0, :, 2].tolist() == expected
    torch.testing.assert_close(mapped[..., :2], raw[..., :2], rtol=0, atol=0)


@pytest.mark.parametrize("path", ["padded", "packed"])
@torch.inference_mode()
def test_one_demo_retains_demo_and_target_input_identity(path, monkeypatch):
    model = _model(task_embeddings=False).eval()
    _set_path(model, monkeypatch, path)
    demo = {"input": [[1, 2], [3, 4]], "output": [[6, 5], [4, 3]]}
    target = {"input": [[7, 8], [9, 0]], "output": [[2, 1], [0, 9]]}
    original = encode_context([demo], target)
    swapped = encode_context([dict(demo, input=target["input"])],
                             dict(target, input=demo["input"]))
    def first_prediction(sequence):
        logits = model(**_batch(packed=path == "packed", sequences=[sequence]))["logits"]
        return logits[sequence[1]] if path == "packed" else logits[0, sequence[1]]
    # With shared input roles these assignments were indistinguishable: the
    # one-demo prefix is fully connected and has no other pair/target labels.
    assert (first_prediction(original) - first_prediction(swapped)).abs().max() > 1e-4


def test_mask_matches_independent_pair_and_prefix_rules():
    model = _model()
    tokens, sep = encode_context(DEMOS, TARGET)
    valid = torch.ones_like(tokens, dtype=torch.bool)[None]
    allowed = model._build_prefix_lm_sdpa_mask(valid, torch.tensor([sep + 1]),
                                              torch.float32, tokens[None])[0, 0].isfinite()
    # Derive pair spans from separately encoded pairs, not production metadata.
    pair_ids = [-1]
    for index, pair in enumerate([*DEMOS, TARGET]):
        single, _ = encode_pair(pair)
        pair_ids.extend([index] * (len(single) - 1))
    target_id = len(DEMOS)
    for q in range(len(tokens)):
        for k in range(len(tokens)):
            if q > sep:
                expected = k <= q
            else:
                q_hub = q == 0 or pair_ids[q] == target_id
                k_hub = k == 0 or pair_ids[k] == target_id
                expected = k <= sep and (q_hub or k_hub or pair_ids[q] == pair_ids[k])
            assert bool(allowed[q, k]) == expected, (q, k)


@pytest.mark.parametrize("path", ["padded", "fallback", "packed"])
def test_packed_attention_and_gradients_match_dense_mask_oracle(path, monkeypatch):
    model, reference = _model(), _model()
    sequences = [encode_context(DEMOS, TARGET), encode_context([], TARGET)]
    _set_path(model, monkeypatch, path)
    expected = reference(**_batch(sequences=sequences))
    actual = model(**_batch(packed=path != "padded", sequences=sequences))
    torch.testing.assert_close(actual["loss"], expected["loss"], rtol=2e-5, atol=1e-6)
    actual["loss"].backward()
    expected["loss"].backward()
    for name, param in model.named_parameters():
        other = dict(reference.named_parameters())[name]
        assert param.grad is not None and torch.isfinite(param.grad).all(), name
        torch.testing.assert_close(param.grad, other.grad, rtol=3e-4, atol=2e-6)


@pytest.mark.parametrize("path", ["padded", "packed"])
@torch.inference_mode()
def test_demo_permutations_preserve_target_logits(path, monkeypatch):
    model = _model().eval()
    _set_path(model, monkeypatch, path)
    def predictions(demos):
        tokens, sep = encode_context(demos, TARGET)
        result = model(**_batch(packed=path == "packed", sequences=[(tokens, sep)]))
        logits = result["logits"]
        return (logits[sep:] if path == "packed" else logits[0, sep:])
    expected = predictions(DEMOS)
    for order in ([2, 0, 1], [1, 0, 2], [2, 1, 0]):
        torch.testing.assert_close(predictions([DEMOS[i] for i in order]), expected,
                                   rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("path", ["padded", "packed"])
@torch.inference_mode()
def test_gold_output_cannot_leak_through_prefix_or_future_output(path, monkeypatch):
    model = _model().eval()
    _set_path(model, monkeypatch, path)
    tokens, sep = encode_context(DEMOS, TARGET)
    changed = tokens.clone()
    changed[sep + 4] = (changed[sep + 4] + 1) % 10
    def predict(ids):
        logits = model(**_batch(packed=path == "packed", sequences=[(ids, sep)]))["logits"]
        return logits if path == "packed" else logits[0]
    torch.testing.assert_close(predict(tokens)[:sep + 4], predict(changed)[:sep + 4], rtol=0, atol=0)


@pytest.mark.parametrize("path", ["padded", "packed"])
@pytest.mark.parametrize("demo_count", [0, 3])
@torch.inference_mode()
def test_other_batch_example_cannot_change_target_predictions(path, demo_count, monkeypatch):
    model = _model(task_embeddings=False).eval()
    _set_path(model, monkeypatch, path)
    first = encode_context(DEMOS[:demo_count], TARGET)
    second, sep = encode_context(DEMOS, TARGET)
    changed = second.clone()
    colors = changed < START_TOKEN_ID
    changed[colors] = (changed[colors] + 3) % 10
    before = model(**_batch(packed=path == "packed", sequences=[first, (second, sep)]))["logits"]
    after = model(**_batch(packed=path == "packed", sequences=[first, (changed, sep)]))["logits"]
    if path == "packed":
        before, after = before[:len(first[0])], after[:len(first[0])]
    else:
        before, after = before[0], after[0]
    torch.testing.assert_close(after, before, rtol=0, atol=0)


@torch.inference_mode()
def test_demos_communicate_through_hub_on_the_second_layer():
    model = _model().eval()
    tokens, sep = encode_context(DEMOS, TARGET)
    changed, _ = encode_context([DEMOS[0], dict(DEMOS[1], input=[[0]]), DEMOS[2]], TARGET)
    def trace(ids):
        states = []
        handles = [block.register_forward_hook(lambda module, args, output: states.append(output.clone()))
                   for block in model.blocks]
        model(**_batch(sequences=[(ids, sep)]))
        for handle in handles:
            handle.remove()
        return states
    original, perturbed = trace(tokens), trace(changed)
    demo_a_len = len(encode_pair(DEMOS[0])[0]) - 1
    torch.testing.assert_close(original[0][:, 1:demo_a_len + 1], perturbed[0][:, 1:demo_a_len + 1], rtol=0, atol=0)
    assert not torch.allclose(original[1][:, 1:demo_a_len + 1], perturbed[1][:, 1:demo_a_len + 1])


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("block_width", [1, 3])
@torch.inference_mode()
def test_left_padded_prefill_and_cached_decode_match_teacher_forcing(native, block_width, monkeypatch):
    model = _model().eval()
    _set_path(model, monkeypatch, "packed" if native else "padded")
    sequences = [encode_context(DEMOS, TARGET), encode_context([], TARGET)]
    full = model(**_batch(sequences=sequences))["logits"]
    prompts = [tokens[:sep + 1].tolist() for tokens, sep in sequences]
    ids, mask = evaluate._left_pad_sequences(prompts, END_PAIR_TOKEN_ID, torch.device("cpu"))
    positions = compute_positions_3d(ids, mask)
    out = model.forward_generate(ids, torch.arange(2), attention_mask=mask, positions_3d=positions)
    for row, (_, sep) in enumerate(sequences):
        torch.testing.assert_close(out["logits"][row, 0], full[row, sep], rtol=2e-5, atol=2e-6)
    output_tokens = sequences[0][0][sequences[0][1] + 1:]
    width = ids.size(1)
    capacity = width + len(output_tokens)
    cache = tuple(tuple(torch.cat((tensor, tensor.new_zeros((*tensor.shape[:2], capacity - width, tensor.size(-1)))), 2)
                        for tensor in layer) for layer in out["past_key_values"])
    cache_mask = torch.cat((mask, mask.new_zeros((2, capacity - width))), -1)
    state, _ = evaluate._derive_initial_state_from_prompt(ids, positions)
    for step in range(0, len(output_tokens), block_width):
        new = torch.stack([tokens[sep + 1 + step:sep + 1 + step + block_width]
                           for tokens, sep in sequences])
        states, pos = evaluate._speculative_grid_positions(state, new)
        state = states[:, -1]
        cache_mask[:, width + step:width + step + new.size(1)] = True
        forward = model.forward_generate if block_width == 1 else model.forward_verify
        result = forward(input_ids=new, example_ids=torch.arange(2), positions_3d=pos,
            past_key_values=cache, cache_position=torch.arange(width + step, width + step + new.size(1)),
            attention_mask=cache_mask)
        for row, (_, sep) in enumerate(sequences):
            expected = full[row, sep + 1 + step:sep + 1 + step + new.size(1)]
            torch.testing.assert_close(result["logits"][row], expected, rtol=2e-5, atol=2e-6)


def test_all_demos_training_evaluation_and_checkpoint_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(build, "resolve_device", lambda _: torch.device("cpu"))
    path = tmp_path / "tasks.json"
    pairs = DEMOS + [{"input": [[0, 1]], "output": [[1, 0]]}]
    path.write_text(json.dumps({"task": {"train": pairs, "test": [TARGET]}}))
    args = Namespace(seed=42, device="cpu", checkpoint_path=None, data_path=path,
        attention_pattern="pair_group", batch_size=2, num_workers=0, enable_aug=False,
        d_model=24, n_heads=4, d_ff=48, n_layers=2, dropout=0, attention_dropout=0)
    model, dataset, loader, _, _ = build.build_model_and_data(args)
    assert dataset.max_seq_len == 1 + 5 * 1860
    assert dataset.demonstration_indices(dataset.examples[0], 0) == (1, 2, 3)
    batch = next(iter(loader))
    assert batch["sep_indices"].dtype == torch.int32
    model(**{k: v for k, v in batch.items() if k not in ("has_padding", "task_ids", "splits", "has_output")})["loss"].backward()
    checkpoint_path = tmp_path / "group.pt"
    train.maybe_save_model(model, dataset, path, checkpoint_path)
    checkpoint = build.load_checkpoint(checkpoint_path)
    assert checkpoint["sequence_format"] == GROUP_SEQUENCE_FORMAT
    args.attention_pattern = "dense"  # Evaluation must inherit the checkpoint.
    restored, evaluation, _, _, _ = build.build_model_and_data(args, checkpoint=checkpoint,
        is_eval=True, build_dataloader=False)
    assert restored.config.attention_pattern == evaluation.attention_pattern == "pair_group"
    example = next(evaluation.iter_examples(split="test"))
    assert evaluation.demonstration_indices(example, 0) == (0, 1, 2, 3)
    attempts = evaluate._build_inference_attempts(evaluation, "test", None, None, 1, False)
    expected, sep = encode_context(pairs, {"input": TARGET["input"]})
    assert attempts[0].prompt == expected.tolist()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(ValueError, match="attention_pattern"):
        build._configure_checkpoint_mtp(model.config, Namespace(attention_pattern="dense"), is_eval=False)
    for old_format in ("two_demo_bidirectional_v1", "all_demo_pair_group_v1"):
        with pytest.raises(ValueError, match="fresh training is required"):
            build._validate_checkpoint_format(dict(checkpoint, sequence_format=old_format))


def test_large_context_has_wide_separator_and_distinct_target_roles(tmp_path, monkeypatch):
    pair = {"input": [[1] * 30 for _ in range(30)], "output": [[2] * 30 for _ in range(30)]}
    path = tmp_path / "large.json"
    path.write_text(json.dumps({"task": {"train": [pair] * 20, "test": [pair]}}))
    # Normal sampling caps contexts at six demos. Deliberately supply a larger
    # synthetic context to test the serializer's int32 separator boundary.
    dataset = ARCExampleDataset(path, splits=("test",), attention_pattern="pair_group",
                                max_seq_len=1 + 21 * 1860)
    monkeypatch.setattr(dataset, "demonstration_indices", lambda example, visit: tuple(range(20)))
    batch = collate_examples([dataset.examples[0]], dataset)
    assert batch["sep_indices"].item() > 32767
    assert batch["max_seqlen"] == 1 + 21 * 1860
    raw = batch["positions_3d"]
    mapped = role_positions(raw, raw[batch["sep_indices"][0].long(), 2])
    assert set(mapped[:, 2].tolist()) == {0, 1, 2, 3, 4}
    torch.testing.assert_close(mapped[:, :2], raw[:, :2])


def test_variation_context_uses_its_own_full_pool_and_capacity(tmp_path, monkeypatch):
    monkeypatch.setattr(variations, "ROOT", tmp_path)
    parent = "arc2_train/source"
    data_path = tmp_path / "tasks.json"
    data_path.write_text(json.dumps({parent: {"train": DEMOS[:1]}}))
    # Variations can have more pairs than their original parent task.
    pairs = DEMOS + [TARGET] * 3
    (tmp_path / "variant.json").write_text(json.dumps({"train": pairs[:3], "test": pairs[3:]}))
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps({"snapshot": "task-variations-v1", "eval_support_task_ids": [],
        "parents": {parent: [{"seed": 0, "path": "variant.json", "pair_count": len(pairs)}]}}))
    dataset = variations.VariationTrainingDataset(json_path=data_path, index_path=index_path,
        seed=42, max_augments=1, attention_pattern="pair_group")
    example = dataset[("variation", parent, 0, 2, 0)]
    assert dataset.demonstration_indices(example, 0) == (0, 1, 3, 4, 5)
    assert dataset.max_seq_len == 1 + 7 * 1860
    tokens, sep, _ = dataset.build_training_sequence(example, 0, None)
    expected, expected_sep = encode_context([pair for i, pair in enumerate(pairs) if i != 2], pairs[2])
    torch.testing.assert_close(tokens, expected)
    assert sep == expected_sep


def test_reusable_group_spec_selects_fresh_group_training():
    path = ROOT / "specs/group_attention_pretrain_eval.json"
    plan = launch_run.build_plan(launch_run.load_spec(path), path)
    assert plan["pretrain"]["attention_pattern"] == "pair_group"
    assert plan["pretrain"].get("checkpoint_path") is None


@pytest.fixture
def training_compile_settings(monkeypatch):
    # Match train_model: capturing ragged sizes/scalars exposed the live failure.
    # Keep these settings scoped so other compiler tests cannot mask regressions.
    for config in (getattr(torch.compiler, "config", None), torch._dynamo.config):
        if config is not None:
            for name in ("capture_scalar_outputs", "capture_dynamic_output_shape_ops"):
                if hasattr(config, name):
                    monkeypatch.setattr(config, name, True)
    monkeypatch.setattr(torch._dynamo.config, "suppress_errors", False)
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _set_compiled_packed_path(model, monkeypatch):
    _set_path(model, monkeypatch, "packed")
    # The CPU oracle uses Python loops/tensor slices, unlike native FlashAttention.
    # Keep that test substitute eager while compiling the real model around it.
    for block in model.blocks:
        monkeypatch.setattr(block.attention, "_apply_varlen_flash_attention",
                            torch.compiler.disable(block.attention._apply_varlen_flash_attention))


@pytest.mark.parametrize("task_embeddings", [False, True])
def test_compiled_training_accepts_group_layouts(monkeypatch, training_compile_settings, task_embeddings):
    model, reference = _model(task_embeddings=task_embeddings), _model(task_embeddings=task_embeddings)
    _set_compiled_packed_path(model, monkeypatch)
    # AOTAutograd checks the backward graph as well as Dynamo's forward trace.
    counter = CompileCounterWithBackend("aot_eager")
    compiled = torch.compile(model, backend=counter)
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
    for demo_sets in ((DEMOS, []), ([], []), (DEMOS[:1], DEMOS)):
        sequences = [encode_context(demos, TARGET) for demos in demo_sets]
        expected = reference(**_batch(sequences=sequences))
        result = compiled(**_batch(packed=True, sequences=sequences))
        torch.testing.assert_close(result["loss"], expected["loss"], rtol=2e-5, atol=1e-6)
        result["loss"].backward()
        expected["loss"].backward()
        for (name, param), other in zip(model.named_parameters(), reference.parameters()):
            if param.requires_grad:
                assert param.grad is not None and torch.isfinite(param.grad).all(), name
                torch.testing.assert_close(param.grad, other.grad, rtol=3e-4, atol=2e-6)
        optimizer.step()
        reference_optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        reference_optimizer.zero_grad(set_to_none=True)
    assert counter.frame_count > 0 and counter.op_count > 0
    for param, other in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(param, other, rtol=3e-4, atol=2e-6)


@pytest.mark.parametrize("output_rows", [(), (0,), (0, 1)])
def test_compiled_attention_handles_prefix_only_and_mixed_outputs(monkeypatch, training_compile_settings, output_rows):
    model = _model()
    _set_compiled_packed_path(model, monkeypatch)
    sequences = [encode_context(DEMOS, TARGET), encode_context([], TARGET)]
    sequences = [(tokens if row in output_rows else tokens[:sep + 1], sep)
                 for row, (tokens, sep) in enumerate(sequences)]
    batch = _batch(packed=True, sequences=sequences)
    padded = _batch(sequences=sequences)
    mask = model._build_prefix_lm_sdpa_mask(
        padded["attention_mask"], padded["sep_indices"].long() + 1,
        torch.float32, padded["input_ids"],
    )
    q, k, v = [torch.randn(batch["input_ids"].numel(), 4, 6, requires_grad=True) for _ in range(3)]

    def attend(q, k, v, input_ids, cu_seqlens, prefix_lengths):
        layout = model._build_packed_prefix_lm_layout(
            cu_seqlens, input_ids.numel(), prefix_lengths, input_ids,
        )
        return model.blocks[0].attention._apply_varlen_prefix_lm_attention(q, k, v, layout, 0.)

    counter = CompileCounterWithBackend("aot_eager")
    actual = torch.compile(attend, backend=counter)(
        q, k, v, batch["input_ids"], batch["cu_seqlens"], batch["sep_indices"].long() + 1,
    )
    expected, offset = [], 0
    for row, (tokens, _) in enumerate(sequences):
        end = offset + len(tokens)
        expected.append(F.scaled_dot_product_attention(
            q[offset:end].transpose(0, 1), k[offset:end].transpose(0, 1), v[offset:end].transpose(0, 1),
            attn_mask=mask[row, 0, :len(tokens), :len(tokens)],
        ).transpose(0, 1))
        offset = end
    expected = torch.cat(expected)
    assert counter.frame_count > 0 and counter.op_count > 0
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)
    for grad, reference in zip(torch.autograd.grad(actual.square().sum(), (q, k, v)),
                               torch.autograd.grad(expected.square().sum(), (q, k, v))):
        torch.testing.assert_close(grad, reference, rtol=3e-4, atol=2e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native FlashAttention requires CUDA")
@pytest.mark.parametrize("compiled", [False, True])
def test_cuda_native_group_attention_matches_padded_reference(monkeypatch, training_compile_settings, compiled):
    model, reference = _model(task_embeddings=False).cuda(), _model(task_embeddings=False).cuda()
    if not model._has_varlen_flash_attention():
        pytest.skip("FlashAttention is not installed")
    monkeypatch.setattr(reference, "_has_varlen_flash_attention", lambda: False)
    sequences = [encode_context(DEMOS, TARGET), encode_context([], TARGET)]
    # Use the real training backend when CUDA/FlashAttention are available.
    training_model = torch.compile(model) if compiled else model
    def on_cuda(batch):
        return {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        actual = training_model(**on_cuda(_batch(packed=True, sequences=sequences)))
        expected = reference(**on_cuda(_batch(sequences=sequences)))
    torch.testing.assert_close(actual["loss"], expected["loss"], rtol=.02, atol=.01)
    actual["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("value", ["window", True, 1])
def test_invalid_attention_pattern_is_rejected(value):
    with pytest.raises(ValueError, match="attention_pattern"):
        TinyTransformerConfig(attention_pattern=value)
    with pytest.raises(ValueError, match="attention_pattern"):
        launch_run.training_overrides({"attention_pattern": value}, "defaults")
