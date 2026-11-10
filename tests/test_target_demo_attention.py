"""Two-lookup attention: independent oracle, gradients, cache and compilation."""
import copy
from pathlib import Path
import sys
import pytest
import torch
import torch.nn.functional as F
from torch._dynamo.testing import CompileCounterWithBackend

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from common import encode_context, compute_positions_3d, END_PAIR_TOKEN_ID
from tinytransformer import TinyTransformer, TinyTransformerConfig
from target_demo_attention import build_split_layout, packed_attention, padded_attention
from group_attention import role_positions
from test_mtp import _batch, _set_path
from test_group_attention import DEMOS, TARGET, training_compile_settings


BRANCH_OPTIONS = [
    {},
    {"residual_init": "depth_scaled"},
    {"branch_norm": "hidden_rms"},
    {"residual_init": "depth_scaled", "branch_norm": "hidden_rms"},
]


def model(split=True, **options):
    torch.manual_seed(7)
    return TinyTransformer(TinyTransformerConfig(d_model=24, n_heads=4, d_ff=48,
        n_layers=3, num_examples=2, num_task_embedding_latents=3, task_embeddings=False,
        dropout=0, embedding_dropout=0, attention_dropout=0, n_predict=4,
        attention_pattern='pair_group', split_target_demo_attention=split, **options))


def test_shared_initialization_matches_reference_and_gates_start_half():
    baseline, split = model(False), model(True)
    for name, value in baseline.state_dict().items():
        torch.testing.assert_close(value, split.state_dict()[name], rtol=0, atol=0)
    for block in split.blocks:
        gate = block.attention.target_demo_gate(torch.randn(2, 5, 24)).sigmoid()
        assert gate.shape == (2, 5, 4)
        assert torch.equal(gate, torch.full_like(gate, .5))


def test_two_normalizations_and_per_head_gate_match_explicit_query_oracle():
    m = model().eval()
    tokens, sep = encode_context(DEMOS, TARGET)
    batch = _batch(sequences=[(tokens, sep)])
    bias = m._build_prefix_lm_sdpa_mask(batch['attention_mask'], torch.tensor([sep + 1]), torch.float32, tokens[None])
    roles = role_positions(batch['positions_3d'], batch['positions_3d'][:, sep:sep + 1, 2])[..., 2]
    q, k, v = [torch.randn(1, 4, len(tokens), 6) for _ in range(3)]
    gate_logits = torch.randn(1, len(tokens), 4)
    expected = torch.empty_like(q)
    for index in range(len(tokens)):
        allowed = bias[0, 0, index].isfinite()
        demo = (roles[0] == 1) | (roles[0] == 2)
        def attend(mask):
            return F.scaled_dot_product_attention(q[:, :, index:index+1], k[:, :, mask], v[:, :, mask])
        if demo[index]:
            value = attend(allowed)
        else:
            gate = gate_logits[:, index].sigmoid()[:, :, None, None]
            value = (1-gate) * attend(allowed & ~demo) + gate * attend(allowed & demo)
        expected[:, :, index:index+1] = value
    actual = padded_attention(q, k, v, bias, roles, gate_logits, 0.)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)


@pytest.mark.parametrize('compiled', [False, True])
@pytest.mark.parametrize('options', BRANCH_OPTIONS)
def test_packed_matches_padded_logits_loss_and_all_gradients(monkeypatch, compiled, training_compile_settings, options):
    actual, expected = model(**options), model(**options)
    sequences = [encode_context(DEMOS, TARGET), encode_context([], TARGET)]
    _set_path(actual, monkeypatch, 'packed')
    _set_path(expected, monkeypatch, 'padded')
    counter = CompileCounterWithBackend('aot_eager')
    training = torch.compile(actual, backend=counter) if compiled else actual
    reference = expected(**_batch(sequences=sequences), compute_training_metrics=True)
    result = training(**_batch(packed=True, sequences=sequences), compute_training_metrics=True)
    flat = torch.cat([reference['logits'][i, :len(t)] for i, (t, s) in enumerate(sequences)])
    torch.testing.assert_close(result['logits'], flat, rtol=3e-5, atol=2e-6)
    torch.testing.assert_close(result['loss'], reference['loss'])
    torch.testing.assert_close(result['training_metric_counts'], reference['training_metric_counts'])
    result['loss'].backward(); reference['loss'].backward()
    for name, param in actual.named_parameters():
        if param.requires_grad:
            other = dict(expected.named_parameters())[name]
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            torch.testing.assert_close(param.grad, other.grad, rtol=5e-4, atol=3e-6)
    assert any(p.grad.abs().sum() > 0 for n, p in actual.named_parameters() if 'target_demo_gate' in n)
    if compiled:
        assert counter.frame_count > 0


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('options', BRANCH_OPTIONS)
@torch.inference_mode()
def test_left_padded_prefill_cache_and_speculative_blocks_match_teacher_forcing(monkeypatch, native, options):
    m = model(**options).eval()
    _set_path(m, monkeypatch, 'packed' if native else 'padded')
    sequences = [encode_context(DEMOS, TARGET), encode_context([], TARGET)]
    full_batch = _batch(sequences=sequences)
    expected = m(**full_batch)['logits']
    prompts = [t[:s + 1] for t, s in sequences]
    width = max(map(len, prompts)) + 3
    ids = torch.full((2, width), END_PAIR_TOKEN_ID, dtype=torch.long)
    mask = torch.zeros_like(ids, dtype=torch.bool)
    for row, prompt in enumerate(prompts):
        ids[row, -len(prompt):] = prompt; mask[row, -len(prompt):] = True
    pos = compute_positions_3d(ids, mask)
    result = m.forward_generate(ids, torch.arange(2), positions_3d=pos, attention_mask=mask)
    for row, (_, sep) in enumerate(sequences):
        torch.testing.assert_close(result['logits'][row, 0], expected[row, sep], rtol=3e-5, atol=2e-6)
    out_len = len(sequences[0][0]) - sequences[0][1] - 1
    capacity = width + out_len + 4
    cache = tuple(tuple(F.pad(value, (0, 0, 0, capacity-width)) for value in layer) for layer in result['past_key_values'])
    valid = F.pad(mask, (0, capacity-width))
    demo_mask = F.pad(mask & (pos[..., 2] > 0) & (pos[..., 2] < pos[:, -1:, 2]), (0, capacity-width))
    for start in range(0, out_len, 2):
        count = min(2, out_len-start)
        new = torch.stack([t[s+1+start:s+1+start+count] for t,s in sequences])
        positions = torch.stack([full_batch['positions_3d'][row,s+1+start:s+1+start+count] for row,(t,s) in enumerate(sequences)])
        slots = (torch.arange(count) + width + start)[None].expand(2, -1)
        valid.scatter_(1, slots, True)
        result = m.forward_generate(new, torch.arange(2), positions_3d=positions,
            past_key_values=cache, cache_position=slots, attention_mask=valid, demo_attention_mask=demo_mask)
        for row, (_, sep) in enumerate(sequences):
            torch.testing.assert_close(result['logits'][row], expected[row,sep+1+start:sep+1+start+count], rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize('native', [False, True])
@torch.inference_mode()
def test_future_gold_is_inaccessible_and_demo_order_is_irrelevant(monkeypatch, native):
    m = model().eval(); _set_path(m, monkeypatch, 'packed' if native else 'padded')
    tokens, sep = encode_context(DEMOS, TARGET)
    changed = tokens.clone(); changed[sep+4] = (changed[sep+4]+1) % 10
    def predict(sequence):
        logits = m(**_batch(packed=native, sequences=[sequence]))['logits']
        return logits if native else logits[0]
    original = predict((tokens, sep))
    torch.testing.assert_close(original[:sep+4], predict((changed,sep))[:sep+4], rtol=0, atol=0)
    permuted = encode_context(list(reversed(DEMOS)), TARGET)
    torch.testing.assert_close(original[sep:], predict(permuted)[permuted[1]:], rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize('native', [False, True])
def test_real_generation_carries_demo_membership_into_decode(monkeypatch, native):
    import evaluate
    m = model().eval().to(torch.bfloat16)
    _set_path(m, monkeypatch, 'packed' if native else 'padded')
    monkeypatch.setattr(torch, 'compile', lambda fn, **kwargs: fn)
    # This helper was compiled at import time, before the compile monkeypatch.
    monkeypatch.setattr(evaluate, '_compiled_grid_update', evaluate._grid_update)
    original = m.forward_generate
    observed = []
    def checked(*args, **kwargs):
        if kwargs.get('past_key_values') is not None:
            membership = kwargs['demo_attention_mask']
            assert membership[0].any() and not membership[1].any()
            observed.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(m, 'forward_generate', checked)
    sequences = [encode_context(DEMOS, TARGET), encode_context([], TARGET)]
    prompts = [t[:s+1].tolist() for t,s in sequences]
    result = evaluate.batched_greedy_generate(m, prompts, [(0,-1),(1,-1)],
        torch.device('cpu'), max_new_tokens=3, use_prepared_cache=False)
    assert observed and len(result) == 2
    assert all(r[:len(p)] == p for r,p in zip(result,prompts))


@pytest.mark.parametrize('options', BRANCH_OPTIONS)
def test_compiled_single_token_cache_step_matches_eager(monkeypatch, options):
    m = model(**options).eval()
    tokens, sep = encode_context(DEMOS, TARGET)
    ids = tokens[:sep+1][None]
    pos = compute_positions_3d(ids)
    with torch.inference_mode():
        prefill = m.forward_generate(ids, torch.tensor([0]), positions_3d=pos)
        capacity = ids.size(1)+4
        cache = tuple(tuple(F.pad(v, (0,0,0,4)) for v in layer) for layer in prefill['past_key_values'])
        mask = F.pad(torch.ones_like(ids,dtype=torch.bool), (0,4))
        mask[:,ids.size(1)] = True
        demo = F.pad((pos[...,2]>0)&(pos[...,2]<pos[:,-1:,2]), (0,4))
        kwargs = dict(input_ids=tokens[sep+1:sep+2][None], example_ids=torch.tensor([0]),
            positions_3d=compute_positions_3d(tokens[None])[:,sep+1:sep+2], attention_mask=mask,
            demo_attention_mask=demo, cache_position=torch.tensor([ids.size(1)]))
        expected = m.forward_generate(past_key_values=tuple(tuple(v.clone() for v in layer) for layer in cache), **kwargs)
        counter = CompileCounterWithBackend('aot_eager')
        actual = torch.compile(m.forward_generate, backend=counter, fullgraph=True)(past_key_values=cache, **kwargs)
        torch.testing.assert_close(actual['logits'], expected['logits'])
        assert counter.frame_count > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Native FlashAttention needs CUDA')
@pytest.mark.parametrize('options', BRANCH_OPTIONS)
def test_native_cuda_training_matches_dense_reference(monkeypatch, options):
    actual, reference = model(**options).cuda(), model(**options).cuda()
    if not actual._has_varlen_flash_attention(): pytest.skip('FlashAttention not installed')
    monkeypatch.setattr(reference, '_has_varlen_flash_attention', lambda: False)
    sequences = [encode_context(DEMOS, TARGET), encode_context([], TARGET)]
    def cuda(batch): return {k:v.cuda() if torch.is_tensor(v) else v for k,v in batch.items()}
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = actual(**cuda(_batch(packed=True, sequences=sequences)))
        expected = reference(**cuda(_batch(sequences=sequences)))
    torch.testing.assert_close(output['loss'], expected['loss'], rtol=.02, atol=.01)
    output['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actual.parameters() if p.requires_grad)
