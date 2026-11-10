"""Selected attention dependencies and a bounded native FA4 readiness check."""
import importlib.metadata

FA4_VERSION = "4.0.0b36"
FA4_REQUIREMENT = f"flash-attn-4[cu13]=={FA4_VERSION}"


def selected_versions(spec):
    """Resolve explicit stage choices without importing Torch or building data."""
    base = {"FA": spec.get("FA", 2), **spec.get("defaults", {})}
    versions = set()
    if not spec.get("eval_only", False):
        versions.add(int({**base, **spec.get("pretrain", {})}["FA"]))
        if spec.get("enable_finetune", True):
            defaults = {**base, **spec.get("finetune_defaults", {})}
            for stage in (spec.get("finetunes") or {"default": {}}).values():
                versions.add(int({**defaults, **stage}["FA"]))
    if not spec.get("skip_evals", False):
        defaults = {**base, **spec.get("eval_defaults", {})}
        for stage in (spec.get("evals") or {"final": {}}).values():
            versions.add(int({**defaults, **stage}["FA"]))
    return versions


def fa4_installed():
    try:
        return importlib.metadata.version("flash-attn-4") == FA4_VERSION
    except importlib.metadata.PackageNotFoundError:
        return False


def check_fa4():
    """Native BF16 rectangular varlen forward/backward; no model or dataset."""
    import torch
    from flash_attn.cute import flash_attn_varlen_func
    if not torch.cuda.is_available():
        raise RuntimeError("FA4 validation requires CUDA.")
    # Separate processes in the normal launcher prevent compiler state from
    # leaking into training. Unequal Q/K lengths exercise output causal offset.
    for causal in (False, True):
        q = torch.randn(5, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(9, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn_like(k, requires_grad=True)
        cq = torch.tensor([0, 2, 5], device="cuda", dtype=torch.int32)
        ck = torch.tensor([0, 4, 9], device="cuda", dtype=torch.int32)
        out, _ = flash_attn_varlen_func(q, k, v, cu_seqlens_q=cq, cu_seqlens_k=ck,
            max_seqlen_q=3, max_seqlen_k=5, causal=causal, return_lse=False)
        reference = []
        for qa, qb, ka, kb in [(0, 2, 0, 4), (2, 5, 4, 9)]:
            mask = None
            if causal:
                mask = torch.arange(kb-ka, device="cuda")[None] <= (
                    torch.arange(qb-qa, device="cuda")[:, None] + (kb-ka)-(qb-qa))
            reference.append(torch.nn.functional.scaled_dot_product_attention(
                q[qa:qb].transpose(0, 1)[None].float(), k[ka:kb].transpose(0, 1)[None].float(),
                v[ka:kb].transpose(0, 1)[None].float(), attn_mask=mask)[0].transpose(0, 1))
        expected = torch.cat(reference).to(out.dtype)
        torch.testing.assert_close(out, expected, rtol=.03, atol=.03)
        gradients = torch.autograd.grad(out.float().square().sum(), (q, k, v))
        expected_gradients = torch.autograd.grad(expected.float().square().sum(), (q, k, v))
        for actual, target in zip(gradients, expected_gradients):
            if not torch.isfinite(actual).all():
                raise FloatingPointError("FA4 produced nonfinite gradients.")
            torch.testing.assert_close(actual, target, rtol=.08, atol=.08)
    torch.cuda.synchronize()
    return {"version": importlib.metadata.version("flash-attn-4"),
            "gpu": torch.cuda.get_device_name(), "rectangular_varlen_forward_backward": True}


if __name__ == "__main__":
    import json
    print(json.dumps(check_fa4()), flush=True)
