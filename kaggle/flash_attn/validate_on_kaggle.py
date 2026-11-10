"""Run in a fresh Kaggle Python process after offline wheel installation.

Exercises FlashAttention kernels without loading a checkpoint or training a model.
"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import importlib.metadata
import json
import pathlib
import platform
import sys

import torch
from flash_attn import flash_attn_varlen_func


def check_case(dtype, head_dim, q_lengths, k_lengths, causal, kv_heads):
    q_heads = 8
    q = torch.randn(sum(q_lengths), q_heads, head_dim, device='cuda', dtype=dtype, requires_grad=True)
    k = torch.randn(sum(k_lengths), kv_heads, head_dim, device='cuda', dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    qr, kr, vr = [x.detach().float().requires_grad_(True) for x in (q, k, v)]
    cu_q = torch.tensor([0, q_lengths[0], sum(q_lengths)], device='cuda', dtype=torch.int32)
    cu_k = torch.tensor([0, k_lengths[0], sum(k_lengths)], device='cuda', dtype=torch.int32)
    out = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max(q_lengths), max(k_lengths), dropout_p=0.0, causal=causal)

    expected = []
    qo = ko = 0
    for qn, kn in zip(q_lengths, k_lengths):
        qb = qr[qo:qo + qn].transpose(0, 1)
        kb = kr[ko:ko + kn].repeat_interleave(q_heads // kv_heads, dim=1).transpose(0, 1)
        vb = vr[ko:ko + kn].repeat_interleave(q_heads // kv_heads, dim=1).transpose(0, 1)
        scores = (qb @ kb.transpose(-2, -1)) * (head_dim ** -0.5)
        if causal:
            # FA2 aligns unequal-length causal masks at the bottom-right.
            rows = torch.arange(qn, device='cuda')[:, None]
            cols = torch.arange(kn, device='cuda')[None, :]
            scores = scores.masked_fill(cols > rows + kn - qn, float('-inf'))
        expected.append((scores.softmax(dim=-1) @ vb).transpose(0, 1))
        qo += qn
        ko += kn
    ref = torch.cat(expected)
    grad = torch.randn_like(out)
    out.backward(grad)
    ref.backward(grad.float())
    torch.cuda.synchronize()

    atol = rtol = 0.03 if dtype == torch.bfloat16 else 0.005
    pairs = [('output', out, ref)] + [
        (name + '_grad', a.grad, b.grad)
        for name, a, b in zip(('q', 'k', 'v'), (q, k, v), (qr, kr, vr))
    ]
    errors = {}
    for name, actual, reference in pairs:
        assert actual is not None and torch.isfinite(actual).all().item(), name
        torch.testing.assert_close(actual.float(), reference, atol=atol, rtol=rtol)
        errors[name + '_max_abs_error'] = (actual.float() - reference).abs().max().item()

    # Also exercise dropout forward/backward; its kernel RNG differs from reference.
    dropped = flash_attn_varlen_func(q, k, v, cu_q, cu_k, max(q_lengths), max(k_lengths), dropout_p=0.1, causal=causal)
    grads = torch.autograd.grad(dropped.float().square().mean(), (q, k, v))
    torch.cuda.synchronize()
    assert torch.isfinite(dropped).all().item()
    assert all(torch.isfinite(g).all().item() for g in grads)
    return {
        'dtype': str(dtype), 'head_dim': head_dim, 'q_lengths': q_lengths,
        'k_lengths': k_lengths, 'causal': causal, 'q_heads': q_heads,
        'kv_heads': kv_heads, 'atol': atol, 'rtol': rtol,
        'dropout_forward_backward_finite': True, **errors,
    }


def main():
    assert sys.version_info[:3] == (3, 13, 15), sys.version
    assert platform.system() == 'Linux' and platform.machine() == 'x86_64'
    assert torch.__version__ == '2.11.0+cu128', torch.__version__
    assert torch.version.cuda == '12.8', torch.version.cuda
    assert torch._C._GLIBCXX_USE_CXX11_ABI is True
    expected_version = '2.8.3.post1+cu128torch2.11cxx11abitrue.sm80'
    assert importlib.metadata.version('flash_attn') == expected_version
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability(0) == (8, 9)
    assert 'L4' in torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(1234)
    cases = []
    for dtype in (torch.float16, torch.bfloat16):
        for head_dim in (64, 128):
            for ql, kl, causal, kv_heads in (
                ([7, 17], [7, 17], False, 8),
                ([7, 17], [7, 17], True, 8),
                ([5, 11], [9, 23], True, 4),
            ):
                result = check_case(dtype, head_dim, ql, kl, causal, kv_heads)
                cases.append(result)
                print('PASS', json.dumps(result), flush=True)
    report = {
        'python': sys.version, 'torch': torch.__version__, 'torch_cuda': torch.version.cuda,
        'torch_cxx11_abi': torch._C._GLIBCXX_USE_CXX11_ABI,
        'flash_attn': expected_version, 'gpu': torch.cuda.get_device_name(0),
        'cases': cases, 'passed': True,
    }
    output = pathlib.Path('/kaggle/working/flash_attn_gpu_validation.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print('FLASHATTENTION L4 FORWARD/BACKWARD VALIDATION PASSED:', output)


if __name__ == '__main__':
    main()
