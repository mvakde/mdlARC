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


def model(split=True):
    torch.manual_seed(7)
    return TinyTransformer(TinyTransformerConfig(d_model=24, n_heads=4, d_ff=48,
        n_layers=3, num_examples=2, num_task_embedding_latents=3, task_embeddings=False,
        dropout=0, embedding_dropout=0, attention_dropout=0, n_predict=4,
        attention_pattern='pair_group', split_target_demo_attention=split))


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
def test_packed_matches_padded_logits_loss_and_all_gradients(monkeypatch, compiled, training_compile_settings):
    actual, expected = model(), model()
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
@torch.inference_mode()
def test_left_padded_prefill_cache_and_speculative_blocks_match_teacher_forcing(monkeypatch, native):
    m = model().eval()
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


@pytest.mark.parametrize('decode_cache', ['full', 'split'])
@pytest.mark.parametrize('native', [False, True])
def test_real_generation_carries_demo_membership_into_decode(monkeypatch, native, decode_cache):
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
        if kwargs.get('split_cache') is not None:
            # Demo keys live only in the first row's demo buffer.
            demo_lengths = kwargs['split_cache'][1]
            assert demo_lengths[0] > 0 and demo_lengths[1] == 0
            observed.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(m, 'forward_generate', checked)
    sequences = [encode_context(DEMOS, TARGET), encode_context([], TARGET)]
    prompts = [t[:s+1].tolist() for t,s in sequences]
    result = evaluate.batched_greedy_generate(m, prompts, [(0,-1),(1,-1)],
        torch.device('cpu'), max_new_tokens=3, decode_cache=decode_cache)
    assert observed and len(result) == 2
    assert all(r[:len(p)] == p for r,p in zip(result,prompts))


def test_compiled_single_token_cache_step_matches_eager(monkeypatch):
    m = model().eval()
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
def test_native_cuda_training_matches_dense_reference(monkeypatch):
    actual, reference = model().cuda(), model().cuda()
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



def test_gather_rows_modes_match_forward_and_agree_on_gradients(monkeypatch):
    import target_demo_attention as module

    torch.manual_seed(0)
    x = torch.randn(12, 2, 4, dtype=torch.bfloat16, requires_grad=True)
    index = torch.tensor([0, 3, 3, 7, 11, 1, 3])
    outputs, grads = [], []
    for mode in (True, False):
        monkeypatch.setattr(module, "FP32_GATHER_BACKWARD", mode)
        out = module._gather_rows(x, index)
        assert out.dtype == torch.bfloat16
        (out.float() * torch.arange(1, 8, dtype=torch.float32).view(-1, 1, 1)).sum().backward()
        outputs.append(out.detach().clone())
        grads.append(x.grad.detach().clone())
        x.grad = None
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
    torch.testing.assert_close(grads[0].float(), grads[1].float(), rtol=1e-2, atol=1e-2)
    monkeypatch.setenv("MDLARC_GATHER_ACCUMULATE", "native")
    assert module._fp32_gather_backward() is False
    monkeypatch.setenv("MDLARC_GATHER_ACCUMULATE", "fp32")
    assert module._fp32_gather_backward() is True
    monkeypatch.setenv("MDLARC_GATHER_ACCUMULATE", "maybe")
    with pytest.raises(ValueError):
        module._fp32_gather_backward()
    monkeypatch.setenv("MDLARC_GATHER_BACKWARD", "clone")
    with pytest.raises(ValueError):
        module._gather_backward_mode()
    monkeypatch.delenv("MDLARC_GATHER_BACKWARD")
    assert module._gather_backward_mode() == "custom"


def test_custom_gather_backward_matches_autograd_and_traces_under_compile(monkeypatch):
    """The in-place scatter-add backward equals index_select's own derivative in both
    accumulation modes, and the autograd.Function plus custom op trace under torch.compile."""
    import target_demo_attention as module

    torch.manual_seed(1)
    index = torch.tensor([0, 3, 3, 7, 11, 1, 3, 11])
    weights = torch.arange(1, 9, dtype=torch.float32).view(-1, 1, 1)
    for dtype, fp32 in ((torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)):
        x = torch.randn(12, 2, 4, dtype=dtype)
        results = {}
        for backward in ("autograd", "custom"):
            monkeypatch.setattr(module, "GATHER_BACKWARD", backward)
            monkeypatch.setattr(module, "FP32_GATHER_BACKWARD", fp32)
            leaf = x.clone().requires_grad_(True)
            out = module._gather_rows(leaf, index)
            assert out.dtype == dtype
            (out.float() * weights).sum().backward()
            results[backward] = (out.detach(), leaf.grad.detach())
        torch.testing.assert_close(results["custom"][0], results["autograd"][0], rtol=0, atol=0)
        torch.testing.assert_close(results["custom"][1], results["autograd"][1], rtol=0, atol=0)
    # Without a gradient requirement the custom path is not entered at all.
    monkeypatch.setattr(module, "GATHER_BACKWARD", "custom")
    plain = torch.randn(6, 3)
    torch.testing.assert_close(module._gather_rows(plain, index[:3]), plain.index_select(0, index[:3]))
    # The scatter-add op itself: duplicates sum, untouched rows stay zero, fp32 mode rounds once.
    grad = torch.randn(4, 2, dtype=torch.bfloat16)
    idx = torch.tensor([2, 0, 2, 5])
    expected = torch.zeros(6, 2, dtype=torch.float32).index_add_(0, idx, grad.float()).to(torch.bfloat16)
    torch.testing.assert_close(module.scatter_add_rows(grad, idx, 6, True), expected, rtol=0, atol=0)
    torch.testing.assert_close(module.scatter_add_rows(grad, idx, 6, False),
                               torch.zeros(6, 2, dtype=torch.bfloat16).index_add_(0, idx, grad), rtol=0, atol=0)
    # Dynamo + AOTAutograd trace the Function and the custom op with dynamic shapes.
    torch.compiler.reset()

    def gather_loss(source, rows):
        return (module._gather_rows(source, rows).float() ** 2).sum()

    compiled = torch.compile(gather_loss, backend="aot_eager", dynamic=True)
    for rows_count in (12, 20):
        source = torch.randn(rows_count, 2, 4, dtype=torch.bfloat16, requires_grad=True)
        reference = source.detach().clone().requires_grad_(True)
        rows = torch.randint(0, rows_count, (9,))
        compiled(source, rows).backward()
        gather_loss(reference, rows).backward()
        torch.testing.assert_close(source.grad, reference.grad, rtol=0, atol=0)


def test_multi_gather_shares_one_backward_buffer_and_matches_separate_gathers(monkeypatch):
    import target_demo_attention as module

    torch.manual_seed(2)
    indices = [torch.tensor([0, 5, 5, 2]), torch.tensor([7, 0]), torch.tensor([3, 3, 3, 9, 1])]
    for dtype, fp32 in ((torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)):
        monkeypatch.setattr(module, "FP32_GATHER_BACKWARD", fp32)
        x = torch.randn(10, 2, 3, dtype=dtype)
        results = {}
        for backward in ("autograd", "custom"):
            monkeypatch.setattr(module, "GATHER_BACKWARD", backward)
            leaf = x.clone().requires_grad_(True)
            outs = module._gather_rows_multi(leaf, indices)
            assert len(outs) == 3 and all(o.dtype == dtype for o in outs)
            loss = sum((o.float() * (i + 1)).sum() for i, o in enumerate(outs))
            loss.backward()
            results[backward] = ([o.detach() for o in outs], leaf.grad.detach())
        for a, b in zip(results["custom"][0], results["autograd"][0]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        # fp32 accumulation then one rounding matches the stock fp32 path exactly; in
        # native bf16 the three scatters round into one buffer in a different order,
        # so allow bf16 ulps there.
        tolerance = {"rtol": 0, "atol": 0} if (fp32 or dtype == torch.float32) else {"rtol": 1e-2, "atol": 1e-2}
        torch.testing.assert_close(results["custom"][1], results["autograd"][1], **tolerance)
    # An unused output contributes no gradient; a single index falls back to the plain gather.
    monkeypatch.setattr(module, "GATHER_BACKWARD", "custom")
    monkeypatch.setattr(module, "FP32_GATHER_BACKWARD", False)
    leaf = torch.randn(10, 2, 3, requires_grad=True)
    first, _unused = module._gather_rows_multi(leaf, indices[:2])
    first.sum().backward()
    expected = torch.zeros(10, 2, 3).index_add_(0, indices[0], torch.ones(4, 2, 3))
    torch.testing.assert_close(leaf.grad, expected, rtol=0, atol=0)
    assert len(module._gather_rows_multi(leaf, indices[:1])) == 1
    # Dynamo + AOTAutograd trace the variadic Function and the list-typed custom op.
    torch.compiler.reset()

    def loss_fn(source, a, b):
        outs = module._gather_rows_multi(source, [a, b])
        return (outs[0].float() ** 2).sum() + (outs[1].float() * 3).sum()

    compiled = torch.compile(loss_fn, backend="aot_eager", dynamic=True)
    for rows_count in (10, 16):
        source = torch.randn(rows_count, 2, 3, dtype=torch.bfloat16, requires_grad=True)
        reference = source.detach().clone().requires_grad_(True)
        a, b = torch.randint(0, rows_count, (6,)), torch.randint(0, rows_count, (4,))
        compiled(source, a, b).backward()
        loss_fn(reference, a, b).backward()
        torch.testing.assert_close(source.grad, reference.grad, rtol=0, atol=0)
