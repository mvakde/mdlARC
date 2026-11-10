"""Separately normalized target/demo attention for the isolated ablation.

Demo queries retain their pair-local spoke attention. START is a neutral member
of the target hub. Target input queries see the whole target input; output
queries see that input plus causal output history. Neither branch reads a
future target output. All calls reuse the ordinary Q/K/V projections.
"""
import os
from typing import List, Optional

import torch
import torch.nn.functional as F
from group_attention import packed_prefix_layout
from common import END_PAIR_TOKEN_ID, START_TOKEN_ID


def _cu(lengths):
    return torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).to(torch.int32)


def _ragged(q, k, q_lengths, k_lengths, causal=False):
    return dict(q=q, k=k, cq=_cu(q_lengths), ck=_cu(k_lengths),
                mq=int(q_lengths.max().item()), mk=int(k_lengths.max().item()), causal=causal)


def build_split_layout(input_ids, cu_seqlens, prefix_lengths):
    """Data-dependent metadata; caller keeps this outside torch.compile."""
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    rows = torch.repeat_interleave(torch.arange(len(lengths), device=input_ids.device), lengths)
    positions = torch.arange(input_ids.numel(), device=input_ids.device) - cu_seqlens[:-1].long()[rows]
    prefix = positions < prefix_lengths[rows]
    ends = (input_ids == END_PAIR_TOKEN_ID) & prefix
    pair_ids = (ends.long() + (positions == 0).long()).cumsum(0) - ends.long()
    target_ids = pair_ids[cu_seqlens[:-1].long() + prefix_lengths - 1]
    hub = prefix & ((pair_ids == target_ids[rows]) | (input_ids == START_TOKEN_ID))
    demos = prefix & ~hub
    local = ~demos
    outputs = ~prefix
    hub_lengths = torch.zeros_like(lengths).scatter_add_(0, rows, hub.long())
    demo_lengths = prefix_lengths - hub_lengths
    local_lengths = lengths - demo_lengths
    output_lengths = lengths - prefix_lengths
    result = {'hub': _ragged(hub.nonzero().flatten(), hub.nonzero().flatten(), hub_lengths, hub_lengths)}

    original = packed_prefix_layout(input_ids, cu_seqlens, prefix_lengths)
    # The original layout stores spoke groups followed by one hub per sequence.
    group_count = original['prefix_cu_seqlens'].numel() - 1 - lengths.numel()
    if group_count:
        cq = original['prefix_cu_seqlens'][:group_count + 1]
        ck = original['prefix_kv_cu_seqlens'][:group_count + 1]
        result['spokes'] = dict(q=original['prefix_token_indices'][:int(cq[-1].item())],
            k=original['prefix_kv_indices'][:int(ck[-1].item())], cq=cq, ck=ck,
            mq=int((cq[1:]-cq[:-1]).max().item()), mk=int((ck[1:]-ck[:-1]).max().item()), causal=False)
    active_output = output_lengths > 0
    if bool(active_output.any()):
        result['output'] = _ragged(outputs.nonzero().flatten(),
            (local & active_output[rows]).nonzero().flatten(),
            output_lengths[active_output], local_lengths[active_output], causal=True)
    active_demo = demo_lengths > 0
    if bool(active_demo.any()):
        result['cross'] = _ragged((local & active_demo[rows]).nonzero().flatten(),
            demos.nonzero().flatten(), local_lengths[active_demo], demo_lengths[active_demo])
    result.update(_assembly_rows(result, input_ids.numel()))
    if 'spokes' in result and 'cross' in result:
        result['spokes_cross'] = _merge_groups(result['spokes'], result['cross'])
    return result


def _merge_groups(first, second):
    """One varlen call for two non-causal groups; FlashAttention treats every
    cu_seqlens segment independently, so the outputs equal the separate calls."""
    return dict(
        q=torch.cat((first['q'], second['q'])), k=torch.cat((first['k'], second['k'])),
        cq=torch.cat((first['cq'], second['cq'][1:] + first['cq'][-1])),
        ck=torch.cat((first['ck'], second['ck'][1:] + first['ck'][-1])),
        mq=max(first['mq'], second['mq']), mk=max(first['mk'], second['mk']), causal=False,
        first_rows=int(first['q'].numel()),
    )


def _assembly_rows(layout, total_tokens):
    """Row map that places stacked group outputs with one gather.

    Spokes, hub and output queries partition the packed tokens, so every token
    has exactly one row in cat(spokes, hub, output); cross queries (the local
    tokens of sequences with demos) are redirected to the gated mix appended
    after those rows. `cross_rows` locates each cross query's unmixed row.
    """
    device = layout['hub']['q'].device
    rows = torch.empty(total_tokens, dtype=torch.long, device=device)
    offset = 0
    for name in ('spokes', 'hub', 'output'):
        if name in layout:
            queries = layout[name]['q']
            rows[queries] = torch.arange(offset, offset + queries.numel(), device=device)
            offset += queries.numel()
    select = rows.clone()
    extra = {}
    if 'cross' in layout:
        queries = layout['cross']['q']
        extra['cross_rows'] = rows[queries]
        select[queries] = torch.arange(offset, offset + queries.numel(), device=device)
    extra['select'] = select
    return extra


def _fp32_gather_backward() -> bool:
    """Whether row gathers should accumulate their backward in fp32.

    On PyTorch 2.11 (A100, 2026-11-08) index_select's bf16 backward ran through
    aten's indexFuncLargeIndex with compare-and-swap atomics at ~10% of HBM
    bandwidth, and gathering an fp32 view cut the step by 11%. On PyTorch 2.14
    (H100, same day) the bf16 path uses cutlass copy kernels at bandwidth and the
    fp32 scratch zeroing cost 7% instead. MDLARC_GATHER_ACCUMULATE=fp32|native
    overrides the version rule for experiments.
    """
    override = os.environ.get("MDLARC_GATHER_ACCUMULATE", "").strip().lower()
    if override in ("fp32", "native"):
        return override == "fp32"
    if override:
        raise ValueError("MDLARC_GATHER_ACCUMULATE must be fp32 or native.")
    major, minor = (int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
    return (major, minor) < (2, 13)


FP32_GATHER_BACKWARD = _fp32_gather_backward()


def _gather_backward_mode() -> str:
    """``custom`` (default) scatters the gather's gradient with an in-place
    index_add_ inside an opaque custom op; ``autograd`` keeps index_select's own
    derivative. MDLARC_GATHER_BACKWARD selects the mode for experiments.

    Autograd's formula for index_select is the out-of-place ``index_add``, which
    inductor cannot run in place: every backward gather first clones its zero
    buffer (a device-to-device memcpy of a full [tokens, heads, dim] tensor).
    On the H100 (2026-11-08) that was 365 copies and 5.4% of the step.
    """
    mode = os.environ.get("MDLARC_GATHER_BACKWARD", "custom").strip().lower()
    if mode not in ("custom", "autograd"):
        raise ValueError("MDLARC_GATHER_BACKWARD must be custom or autograd.")
    return mode


GATHER_BACKWARD = _gather_backward_mode()


def _scatter_add_rows_eager(grad, index, rows, fp32):
    """grad [n, ...] scattered back onto ``rows`` source rows, summing duplicates."""
    accumulate_dtype = torch.float32 if fp32 else grad.dtype
    out = grad.new_zeros((rows, *grad.shape[1:]), dtype=accumulate_dtype)
    out.index_add_(0, index, grad if accumulate_dtype == grad.dtype else grad.to(accumulate_dtype))
    return out if accumulate_dtype == grad.dtype else out.to(grad.dtype)


def _scatter_add_rows_multi_eager(grads, indices, rows, fp32):
    """Several gathers of one source share a single zero buffer and gradient."""
    first = grads[0]
    accumulate_dtype = torch.float32 if fp32 else first.dtype
    out = first.new_zeros((rows, *first.shape[1:]), dtype=accumulate_dtype)
    for grad, index in zip(grads, indices):
        out.index_add_(0, index, grad if accumulate_dtype == grad.dtype else grad.to(accumulate_dtype))
    return out if accumulate_dtype == first.dtype else out.to(first.dtype)


if hasattr(getattr(torch, "library", None), "custom_op"):
    @torch.library.custom_op("mdlarc::scatter_add_rows", mutates_args=())
    def scatter_add_rows(grad: torch.Tensor, index: torch.Tensor, rows: int, fp32: bool) -> torch.Tensor:
        return _scatter_add_rows_eager(grad, index, rows, fp32)

    @scatter_add_rows.register_fake
    def _scatter_add_rows_fake(grad, index, rows, fp32):
        return grad.new_empty((rows, *grad.shape[1:]))

    @torch.library.custom_op("mdlarc::scatter_add_rows_multi", mutates_args=())
    def scatter_add_rows_multi(grads: List[torch.Tensor], indices: List[torch.Tensor], rows: int, fp32: bool) -> torch.Tensor:
        return _scatter_add_rows_multi_eager(grads, indices, rows, fp32)

    @scatter_add_rows_multi.register_fake
    def _scatter_add_rows_multi_fake(grads, indices, rows, fp32):
        return grads[0].new_empty((rows, *grads[0].shape[1:]))
else:  # pragma: no cover - PyTorch < 2.4
    scatter_add_rows = _scatter_add_rows_eager
    scatter_add_rows_multi = _scatter_add_rows_multi_eager


class _GatherRowsMulti(torch.autograd.Function):
    """Several index_selects of one source whose backward fills one zero buffer and
    scatters every gradient into it: no per-gather buffer, no summing of partial grads."""

    @staticmethod
    def forward(ctx, x, fp32, *indices):
        ctx.save_for_backward(*indices)
        ctx.rows = x.shape[0]
        ctx.fp32 = fp32
        return tuple(x.index_select(0, index) for index in indices)

    @staticmethod
    def backward(ctx, *grads):
        indices = ctx.saved_tensors
        live = [(grad.contiguous(), index) for grad, index in zip(grads, indices) if grad is not None]
        if not live:
            return (None, None) + (None,) * len(indices)
        grad_x = scatter_add_rows_multi([g for g, _ in live], [i for _, i in live], ctx.rows, ctx.fp32)
        return (grad_x, None) + (None,) * len(indices)


def _gather_rows_multi(x, indices):
    """``[x.index_select(0, i) for i in indices]`` with one shared backward buffer."""
    indices = list(indices)
    if GATHER_BACKWARD == "custom" and torch.is_grad_enabled() and x.requires_grad and len(indices) > 1:
        return list(_GatherRowsMulti.apply(x, FP32_GATHER_BACKWARD, *indices))
    return [_gather_rows(x, index) for index in indices]


class _GatherRows(torch.autograd.Function):
    """index_select whose backward is one zero-fill plus one in-place index_add_."""

    @staticmethod
    def forward(ctx, x, index, fp32):
        ctx.save_for_backward(index)
        ctx.rows = x.shape[0]
        ctx.fp32 = fp32
        return x.index_select(0, index)

    @staticmethod
    def backward(ctx, grad):
        (index,) = ctx.saved_tensors
        return scatter_add_rows(grad.contiguous(), index, ctx.rows, ctx.fp32), None, None


def _gather_rows(x, index):
    """Row gather; see _fp32_gather_backward and _gather_backward_mode.

    The forward values are bit-identical in every mode (a gather is exact); the
    modes differ only in how the backward scatter-add accumulates and whether it
    clones its zero buffer first.
    """
    if GATHER_BACKWARD == "custom" and torch.is_grad_enabled() and x.requires_grad:
        return _GatherRows.apply(x, index, FP32_GATHER_BACKWARD)
    if FP32_GATHER_BACKWARD:
        return x.float().index_select(0, index).to(x.dtype)
    return x.index_select(0, index)


def packed_attention(attention, q, k, v, layout, gate_logits, dropout_p):
    # Group outputs are stacked in layout order and placed by the precomputed
    # row map with a single gather: no per-group index_copy or clone. The two
    # non-causal groups (spokes, cross) share one FlashAttention call, and the
    # groups' q/k/v row gathers share one backward buffer per source tensor.
    merged = layout.get('spokes_cross')
    order = ('spokes_cross', 'hub', 'output') if merged is not None else ('spokes', 'hub', 'output', 'cross')
    names = [name for name in order if name in layout]
    parts = [layout[name] for name in names]
    queries = _gather_rows_multi(q, [part['q'] for part in parts])
    keys = _gather_rows_multi(k, [part['k'] for part in parts])
    values = _gather_rows_multi(v, [part['k'] for part in parts])
    outputs = {}
    for name, part, part_q, part_k, part_v in zip(names, parts, queries, keys, values):
        outputs[name] = attention._apply_varlen_flash_attention(
            queries=part_q, keys=part_k, values=part_v, cu_seqlens_q=part['cq'], cu_seqlens_k=part['ck'],
            max_seqlen_q=part['mq'], max_seqlen_k=part['mk'],
            dropout_p=dropout_p, is_causal=part['causal'])
    if merged is not None:
        both = outputs['spokes_cross']
        rows = [both[:merged['first_rows']]]
        cross_out = both[merged['first_rows']:]
        rows.extend(outputs[name] for name in ('hub', 'output') if name in outputs)
    else:
        rows = [outputs[name] for name in ('spokes', 'hub', 'output') if name in outputs]
        cross_out = outputs.get('cross')
    stacked = torch.cat(rows) if len(rows) > 1 else rows[0]
    if cross_out is not None:
        part = layout['cross']
        gate = _gather_rows(gate_logits, part['q']).sigmoid().to(q.dtype).unsqueeze(-1)
        local = _gather_rows(stacked, layout['cross_rows'])
        stacked = torch.cat((stacked, (1 - gate) * local + gate * cross_out))
    # Rows with no demos retain their target branch exactly (effective gate=0).
    return _gather_rows(stacked, layout['select'])


def padded_attention(q, k, v, original_bias, roles, gate_logits, dropout_p):
    """Independent dense reference/fallback for training and prompt prefill."""
    demo_keys = (roles == 1) | (roles == 2)
    target_queries = ~demo_keys
    local_bias = original_bias.masked_fill(demo_keys[:, None, None, :], float('-inf'))
    demo_bias = original_bias.masked_fill(~demo_keys[:, None, None, :], float('-inf'))
    original = F.scaled_dot_product_attention(q, k, v, attn_mask=original_bias, dropout_p=dropout_p)
    local = F.scaled_dot_product_attention(q, k, v, attn_mask=local_bias, dropout_p=dropout_p)
    demo = F.scaled_dot_product_attention(q, k, v, attn_mask=demo_bias, dropout_p=dropout_p)
    has_demo = torch.isfinite(demo_bias).any(-1, keepdim=True)
    gate = gate_logits.sigmoid().to(q.dtype).transpose(1, 2).unsqueeze(-1)
    gate = torch.where(has_demo, gate, torch.zeros_like(gate))
    combined = (1 - gate) * local + gate * demo
    return torch.where(target_queries[:, None, :, None], combined, original)


def cached_attention(q, k, v, valid_keys, demo_keys, query_positions, gate_logits):
    """Two SDPA lookups over a fixed-capacity KV cache; compile-safe."""
    allowed = valid_keys[:, None, None, :k.size(2)]
    if query_positions is not None:
        causal = torch.arange(k.size(2), device=q.device)[None, None, None, :] <= query_positions[:, None, :, None]
        allowed = allowed & causal
    demo_keys = demo_keys[:, None, None, :k.size(2)]
    local_allowed = allowed & ~demo_keys
    demo_allowed = allowed & demo_keys
    local = F.scaled_dot_product_attention(q, k, v, attn_mask=local_allowed)
    demo = F.scaled_dot_product_attention(q, k, v, attn_mask=demo_allowed)
    gate = gate_logits.sigmoid().to(q.dtype).transpose(1, 2).unsqueeze(-1)
    gate = torch.where(demo_allowed.any(-1, keepdim=True), gate, torch.zeros_like(gate))
    return (1 - gate) * local + gate * demo


# ---------------------------------------------------------------------------
# Split decode cache: each row's demo keys and target keys live in separate
# buffers, so a generated token's two lookups read only their own keys, up to
# each row's filled length. Lengths stay on the device ([B] int32): one decode
# step has fixed shapes and no host reads, so it can be replayed as a CUDA
# graph. Layout is [batch, slot, head, dim], FlashAttention's cache layout.
# ---------------------------------------------------------------------------

try:  # FlashAttention 2 (sm80+)
    from flash_attn import flash_attn_with_kvcache as _fa2_with_kvcache
except ImportError:  # pragma: no cover - CUDA-only dependency
    _fa2_with_kvcache = None
try:  # FlashAttention 3 (Hopper)
    from flash_attn_interface import flash_attn_with_kvcache as _fa3_with_kvcache
except ImportError:  # pragma: no cover - CUDA-only dependency
    _fa3_with_kvcache = None

# 0 = masked SDPA over the split buffers (CPU, tests, fallback); 2/3 = FlashAttention.
KV_CACHE_SDPA = 0


def choose_kv_cache_kernel(device: torch.device, dtype: torch.dtype, head_dim: int, version: int) -> int:
    """FlashAttention cache kernel for a decode batch, or KV_CACHE_SDPA.

    FA2's split-KV decode kernel is preferred even on Hopper: on the d256 x6
    checkpoint (H200, head dim 64, Nov 8) it beat FA3 for single-token steps
    (12.2 s vs 14.4 s) and speculative blocks (4.1 s vs 4.5 s), and masked SDPA
    over the same buffers (11.7 s, not repeat-identical; 6.0 s speculative).
    FA3 is used when FA2 is missing. MDLARC_KV_CACHE_KERNEL=auto|flash|sdpa
    overrides for experiments; flash raises when no cache kernel can run here.
    """
    choice = os.environ.get("MDLARC_KV_CACHE_KERNEL", "auto").strip().lower()
    if choice not in ("auto", "flash", "sdpa"):
        raise ValueError("MDLARC_KV_CACHE_KERNEL must be auto, flash or sdpa.")
    usable = (choice != "sdpa" and torch.device(device).type == "cuda"
              and dtype in (torch.float16, torch.bfloat16) and head_dim % 8 == 0 and head_dim <= 256)
    kernel = KV_CACHE_SDPA
    if usable:
        hopper = torch.cuda.get_device_capability(device)[0] == 9
        if _fa2_with_kvcache is not None:
            kernel = 2
        elif hopper and _fa3_with_kvcache is not None:
            kernel = 3
    if choice == "flash" and kernel == KV_CACHE_SDPA:
        raise RuntimeError("MDLARC_KV_CACHE_KERNEL=flash, but no FlashAttention cache kernel can run here.")
    return kernel


def _flash_kv_cache_attention_eager(q, k_cache, v_cache, lengths, causal, kernel):
    function = _fa3_with_kvcache if kernel == 3 else _fa2_with_kvcache
    out = function(q, k_cache, v_cache, cache_seqlens=lengths, causal=causal)
    return out[0] if isinstance(out, tuple) else out


if hasattr(getattr(torch, "library", None), "custom_op"):
    # Opaque to torch.compile and safe inside CUDA graph capture: lengths are
    # device tensors and the kernels allocate only through PyTorch.
    @torch.library.custom_op("mdlarc::flash_kv_cache_attention", mutates_args=())
    def flash_kv_cache_attention(q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                                 lengths: torch.Tensor, causal: bool, kernel: int) -> torch.Tensor:
        return _flash_kv_cache_attention_eager(q, k_cache, v_cache, lengths, causal, kernel)

    @flash_kv_cache_attention.register_fake
    def _flash_kv_cache_attention_fake(q, k_cache, v_cache, lengths, causal, kernel):
        return torch.empty_like(q)
else:  # pragma: no cover - older PyTorch
    flash_kv_cache_attention = _flash_kv_cache_attention_eager


def reference_kv_cache_attention(q, k_cache, v_cache, lengths, causal):
    """Masked SDPA with FlashAttention's cache semantics.

    q [B, S, H, D]; caches [B, L, H, D]; row b reads slots < lengths[b]. With
    ``causal`` the S queries are the row's last S filled slots, so query i reads
    slots <= lengths[b] - S + i (FlashAttention's bottom-right alignment).
    """
    slots = torch.arange(k_cache.size(1), device=q.device)
    allowed = slots[None, None, :] < lengths.to(torch.long)[:, None, None]
    if causal:
        query_slots = lengths.to(torch.long)[:, None] - q.size(1) + torch.arange(q.size(1), device=q.device)[None, :]
        allowed = allowed & (slots[None, None, :] <= query_slots[:, :, None])
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k_cache.transpose(1, 2), v_cache.transpose(1, 2), attn_mask=allowed[:, None])
    return out.transpose(1, 2)


def _kv_cache_attention(q, k_cache, v_cache, lengths, causal, kernel):
    if kernel == KV_CACHE_SDPA:
        return reference_kv_cache_attention(q, k_cache, v_cache, lengths, causal)
    return flash_kv_cache_attention(q, k_cache, v_cache, lengths, causal, kernel)


def split_cache_attention(q, k, v, layer_cache, demo_lengths, target_lengths, gate_logits, kernel):
    """One cached step of the gated two-lookup attention over a split cache.

    q, k, v [B, S, H, D] for S new target-output tokens per row. Their keys are
    written at target slots target_lengths[b] + i (target_lengths counts the
    filled slots before this call and is advanced by the caller), then the
    target lookup reads them causally and the demo lookup reads all demos.
    gate_logits [B, S, H]. Rows without demos keep the target lookup alone.
    """
    demo_keys, demo_values, target_keys, target_values = layer_cache
    slots = target_lengths.to(torch.long)[:, None] + torch.arange(q.size(1), device=q.device)[None, :]
    rows = torch.arange(q.size(0), device=q.device)[:, None].expand_as(slots)
    # index_put_ stays in place under torch.compile; a scatter_ into these input
    # buffers was functionalized into a full copy of each buffer per layer and step.
    target_keys.index_put_((rows, slots), k)
    target_values.index_put_((rows, slots), v)
    q = q.contiguous()
    local = _kv_cache_attention(q, target_keys, target_values, target_lengths + q.size(1), True, kernel)
    # A demo-less row reads one finite slot; its gate is zero and where() keeps the target lookup.
    demo = _kv_cache_attention(q, demo_keys, demo_values, demo_lengths.clamp(min=1), False, kernel)
    gate = gate_logits.sigmoid().to(q.dtype).unsqueeze(-1)
    mixed = (1 - gate) * local + gate * demo
    return torch.where((demo_lengths > 0)[:, None, None, None], mixed, local)


def _round_up(value: int, multiple: int) -> int:
    return max(multiple, (int(value) + multiple - 1) // multiple * multiple)


def split_prompt_cache(prompt_key_values, valid, demo, decode_slots: int, slot_multiple: int = 128):
    """Compact a padded prompt cache into per-row demo and target buffers.

    prompt_key_values: a list of per-layer (keys, values) [B, H, L, D] from the
    prompt prefill; entries are released as they are converted. valid and demo
    are [B, L] masks: demo keys are valid demo-pair tokens, target keys every
    other valid token (START and the target input). Order within each part is
    preserved. Returns (layers, demo_lengths, target_lengths): per-layer
    (demo_k, demo_v, target_k, target_v) and int32 [B] filled lengths. The
    target buffers leave ``decode_slots`` free slots after the longest prompt
    target part (rounded up to slot_multiple) for generated tokens.
    """
    demo = valid & demo
    target = valid & ~demo
    demo_lengths = demo.sum(dim=1, dtype=torch.int32)
    target_lengths = target.sum(dim=1, dtype=torch.int32)
    longest_prompt_target = int(target_lengths.max())
    demo_slots = _round_up(int(demo_lengths.max()), slot_multiple)
    target_slots = _round_up(longest_prompt_target + max(0, int(decode_slots)), slot_multiple)
    length = valid.size(1)
    positions = torch.arange(length, device=valid.device)
    order = lambda mask: torch.sort(torch.where(mask, positions, positions + length), dim=1).indices
    demo_order = order(demo)[:, :min(demo_slots, length)]
    target_order = order(target)[:, :longest_prompt_target]
    layers = []
    for index in range(len(prompt_key_values)):
        keys, values = prompt_key_values[index]
        prompt_key_values[index] = None
        batch, heads, _, dim = keys.shape
        converted = []
        for tensor in (keys, values):
            rows = tensor.transpose(1, 2)
            demo_part = rows.new_zeros((batch, demo_slots, heads, dim))
            demo_part[:, :demo_order.size(1)] = rows.gather(
                1, demo_order[:, :, None, None].expand(-1, -1, heads, dim))
            converted.append(demo_part)
        for tensor in (keys, values):
            rows = tensor.transpose(1, 2)
            target_part = rows.new_zeros((batch, target_slots, heads, dim))
            target_part[:, :longest_prompt_target] = rows.gather(
                1, target_order[:, :, None, None].expand(-1, -1, heads, dim))
            converted.append(target_part)
        layers.append(tuple(converted))
    return tuple(layers), demo_lengths, target_lengths


# ---------------------------------------------------------------------------
# Packed prefix-LM attention layout, built once per batch on any device.
# The training collate builds these ragged index lists on the CPU inside the
# DataLoader worker, so the GPU never stalls on the nonzero()/item() round trips
# the layout needs. The model builds the same fields on the device for batches
# that arrive without them (evaluation, synthetic probes, tests).
# ---------------------------------------------------------------------------


def packed_prefix_lm_layout_fields(
    cu_seqlens: torch.Tensor,
    total_tokens: int,
    prefix_lengths: torch.Tensor,
    input_ids: Optional[torch.Tensor] = None,
    *,
    attention_pattern: str = "dense",
    split_target_demo: bool = False,
) -> dict:
    """Return the PackedPrefixLMLayout keyword fields for one packed batch.

    Pair counts, ragged indices, active output sequences, and FlashAttention
    maximum lengths depend on token values, so this runs eagerly and is shared
    by every transformer layer. No learned computation lives here.
    """
    device = cu_seqlens.device
    batch_size = int(cu_seqlens.numel()) - 1
    seq_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(
        device=device, dtype=torch.long
    )
    prefix_lengths = prefix_lengths.to(device=device)
    prefix_lengths = prefix_lengths.clamp(min=1)
    prefix_lengths = torch.minimum(prefix_lengths, seq_lengths)
    output_lengths = seq_lengths - prefix_lengths

    seq_ids = torch.repeat_interleave(
        torch.arange(batch_size, device=device, dtype=torch.long),
        seq_lengths,
    )
    token_offsets = torch.repeat_interleave(
        cu_seqlens[:-1].to(device=device, dtype=torch.long),
        seq_lengths,
    )
    local_positions = torch.arange(total_tokens, device=device, dtype=torch.long)
    local_positions = local_positions - token_offsets
    prefix_mask = local_positions < prefix_lengths[seq_ids]

    prefix_token_indices = prefix_mask.nonzero(as_tuple=False).squeeze(-1)
    prefix_cu_seqlens = torch.zeros(
        batch_size + 1, dtype=torch.int32, device=device
    )
    prefix_cu_seqlens[1:] = torch.cumsum(
        prefix_lengths.to(dtype=torch.int32), dim=0
    )

    prefix_fields = dict(prefix_token_indices=prefix_token_indices,
                         prefix_cu_seqlens=prefix_cu_seqlens,
                         max_prefix_len=int(prefix_lengths.max().item()))
    if attention_pattern == "pair_group":
        if input_ids is None:
            raise ValueError("Pair-group attention requires input_ids to identify pairs.")
        prefix_fields = packed_prefix_layout(input_ids, cu_seqlens, prefix_lengths)
        if split_target_demo:
            prefix_fields['split_target_demo_layout'] = build_split_layout(
                input_ids, cu_seqlens, prefix_lengths)

    active_output = output_lengths > 0
    if not torch.any(active_output):
        return prefix_fields

    output_token_indices = (~prefix_mask).nonzero(as_tuple=False).squeeze(-1)
    active_output_lengths = output_lengths[active_output]
    active_seq_lengths = seq_lengths[active_output]
    active_batch_size = int(active_output.sum().item())
    uses_full_kv = bool(torch.all(active_output).item())

    output_cu_seqlens = torch.zeros(
        active_batch_size + 1, dtype=torch.int32, device=device
    )
    output_cu_seqlens[1:] = torch.cumsum(
        active_output_lengths.to(dtype=torch.int32), dim=0
    )
    if uses_full_kv:
        output_kv_indices = None
        output_kv_cu_seqlens = cu_seqlens
    else:
        output_kv_mask = active_output[seq_ids]
        output_kv_indices = output_kv_mask.nonzero(as_tuple=False).squeeze(-1)
        output_kv_cu_seqlens = torch.zeros(
            active_batch_size + 1, dtype=torch.int32, device=device
        )
        output_kv_cu_seqlens[1:] = torch.cumsum(
            active_seq_lengths.to(dtype=torch.int32), dim=0
        )

    return dict(
        **prefix_fields,
        output_token_indices=output_token_indices,
        output_cu_seqlens=output_cu_seqlens,
        output_kv_indices=output_kv_indices,
        output_kv_cu_seqlens=output_kv_cu_seqlens,
        max_output_len=int(active_output_lengths.max().item()),
        max_output_kv_len=int(active_seq_lengths.max().item()),
        output_uses_full_kv=uses_full_kv,
    )

