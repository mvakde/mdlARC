"""Separately normalized target/demo attention for the isolated ablation.

Demo queries retain their pair-local spoke attention. START is a neutral member
of the target hub. Target input queries see the whole target input; output
queries see that input plus causal output history. Neither branch reads a
future target output. All calls reuse the ordinary Q/K/V projections.
"""
import torch
import torch.nn.functional as F
from group_attention import packed_prefix_layout
from common import END_PAIR_TOKEN_ID, START_TOKEN_ID


def _map_tensors(value, transform, memo):
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if torch.is_tensor(value):
        result = transform(value)
    elif isinstance(value, dict):
        result = {key: _map_tensors(item, transform, memo) for key, item in value.items()}
    else:
        return value
    memo[identity] = result
    return result


class PreparedLayout:
    """CPU collation metadata, pinnable by DataLoader before async transfer."""
    def __init__(self, data):
        self.data = data

    def pin_memory(self):
        return PreparedLayout(_map_tensors(self.data, lambda tensor: tensor.pin_memory(), {}))

    def to(self, device, non_blocking=False):
        return _map_tensors(self.data,
            lambda tensor: tensor.to(device=device, non_blocking=non_blocking), {})


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
    hub_indices = hub.nonzero().flatten()
    result = {'spokes_hub': _ragged(hub_indices, hub_indices, hub_lengths, hub_lengths)}

    original = packed_prefix_layout(input_ids, cu_seqlens, prefix_lengths)
    # The original layout stores spoke groups followed by one hub per sequence.
    group_count = original['prefix_cu_seqlens'].numel() - 1 - lengths.numel()
    if group_count:
        cq = original['prefix_cu_seqlens'][:group_count + 1]
        ck = original['prefix_kv_cu_seqlens'][:group_count + 1]
        # Batch independent non-causal segments; the hub still has ONLY its
        # target-input keys, not the original single-lookup hub's full prefix.
        result['spokes_hub'] = _ragged(
            torch.cat((original['prefix_token_indices'][:int(cq[-1].item())], hub_indices)),
            torch.cat((original['prefix_kv_indices'][:int(ck[-1].item())], hub_indices)),
            torch.cat((cq[1:] - cq[:-1], hub_lengths)),
            torch.cat((ck[1:] - ck[:-1], hub_lengths)))
    active_output = output_lengths > 0
    if bool(active_output.any()):
        result['output'] = _ragged(outputs.nonzero().flatten(),
            (local & active_output[rows]).nonzero().flatten(),
            output_lengths[active_output], local_lengths[active_output], causal=True)
    active_demo = demo_lengths > 0
    if bool(active_demo.any()):
        result['cross'] = _ragged((local & active_demo[rows]).nonzero().flatten(),
            demos.nonzero().flatten(), local_lengths[active_demo], demo_lengths[active_demo])
    return result


def packed_attention(attention, q, k, v, layout, gate_logits, dropout_p, hidden_states=None):
    def attend(part):
        return attention._apply_varlen_flash_attention(
            queries=q.index_select(0, part['q']), keys=k.index_select(0, part['k']),
            values=v.index_select(0, part['k']), cu_seqlens_q=part['cq'], cu_seqlens_k=part['ck'],
            max_seqlen_q=part['mq'], max_seqlen_k=part['mk'],
            dropout_p=dropout_p, is_causal=part['causal'])
    output = torch.empty_like(q)
    for name in ('spokes_hub', 'output'):
        if name in layout:
            part = layout[name]
            output = output.index_copy(0, part['q'], attend(part))
    if 'cross' in layout:
        part = layout['cross']
        # Demo queries never use this projection. Keep shared parameter names
        # and initialization while projecting only the target rows that do.
        logits = (attention.target_demo_gate(hidden_states.index_select(0, part['q']))
                  if hidden_states is not None else gate_logits.index_select(0, part['q']))
        gate = logits.sigmoid().to(q.dtype).unsqueeze(-1)
        local = output.index_select(0, part['q'])
        output = output.index_copy(0, part['q'], (1 - gate) * local + gate * attend(part))
    # Rows with no demos retain their target branch exactly (effective gate=0).
    return output


def padded_attention(q, k, v, original_bias, roles, gate_logits, dropout_p, *, demo_role_max=2):
    """Independent dense reference/fallback for training and prompt prefill."""
    demo_keys = (roles >= 1) & (roles <= demo_role_max)
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


def prepare_decode_cache(prompt_kvs, demo_keys, valid_keys, output_capacity, attentions):
    """Keep one frozen concatenated demo segment per row, plus a target cache.

    Unlike shifted attention, ALL demos in a row share ONE softmax. K already
    carries the layer's RoPE. Compaction is the only later repacking operation.
    """
    valid, demos = valid_keys.cpu(), demo_keys.cpu().bool()
    local_indices = [(v & ~d).nonzero().flatten() for v, d in zip(valid, demos)]
    demo_indices = [(v & d).nonzero().flatten() for v, d in zip(valid, demos)]
    local_width = max(map(len, local_indices))
    capacity = local_width + output_capacity
    mask = valid.new_zeros((valid.size(0), capacity))
    columns = torch.zeros((valid.size(0), local_width), dtype=torch.long)
    for row, indices in enumerate(local_indices):
        mask[row, local_width-len(indices):local_width] = True
        columns[row, local_width-len(indices):] = indices
    owners = tuple(row for row, indices in enumerate(demo_indices) if len(indices))
    lengths = tuple(len(demo_indices[row]) for row in owners)
    device = valid_keys.device
    rows = torch.arange(valid.size(0))[:, None].expand_as(columns).reshape(-1).to(device)
    columns = columns.flatten().to(device)
    if owners:
        demo_rows = torch.repeat_interleave(torch.tensor(owners), torch.tensor(lengths)).to(device)
        demo_columns = torch.cat([demo_indices[row] for row in owners]).to(device)
    local_kvs, caches = [], []
    for (keys, values), attention in zip(prompt_kvs, attentions):
        buffers = []
        for tensor in (keys, values):
            buffer = tensor.new_zeros((tensor.size(0), tensor.size(1), capacity, tensor.size(-1)))
            gathered = tensor[rows, :, columns].reshape(tensor.size(0), local_width, tensor.size(1), tensor.size(-1))
            buffer[:, :, :local_width] = gathered.transpose(1, 2)
            buffers.append(buffer)
        local_kvs.append(tuple(buffers))
        caches.append(_demo_cache(keys[demo_rows, :, demo_columns], values[demo_rows, :, demo_columns],
            owners, lengths, device.type == 'cuda' and attention._has_selected_varlen_flash_attention())
            if owners else None)
    return tuple(local_kvs), tuple(caches), mask.to(device), local_width


def _demo_cache(keys, values, owners, lengths, native):
    from itertools import accumulate
    device = keys.device
    maximum = max(lengths)
    cache = dict(keys=keys, values=values, owners=owners, lengths=lengths,
        rows=torch.tensor(owners, device=device),
        cq=torch.arange(len(owners)+1, device=device, dtype=torch.int32),
        ck=torch.tensor([0, *accumulate(lengths)], device=device, dtype=torch.int32),
        max_length=maximum, native=native)
    if not native:
        for name, tensor in [('padded_keys', keys), ('padded_values', values)]:
            cache[name] = torch.stack([F.pad(chunk, (0, 0, 0, 0, 0, maximum-length))
                for chunk, length in zip(tensor.split(lengths), lengths)]).transpose(1, 2)
        cache['mask'] = (torch.arange(maximum, device=device)[None] <
            torch.tensor(lengths, device=device)[:, None])[:, None, None, :]
    return cache


def select_decode_cache(cache, rows):
    """Repack immutable demo segments only when completed rows are removed."""
    from itertools import accumulate
    if cache is None:
        return None
    remap = {old: new for new, old in enumerate(rows)}
    keep = [i for i, owner in enumerate(cache['owners']) if owner in remap]
    if not keep:
        return None
    offsets = [0, *accumulate(cache['lengths'])]
    indices = torch.cat([torch.arange(offsets[i], offsets[i+1]) for i in keep]).to(cache['keys'].device)
    return _demo_cache(cache['keys'].index_select(0, indices), cache['values'].index_select(0, indices),
        tuple(remap[cache['owners'][i]] for i in keep), tuple(cache['lengths'][i] for i in keep), cache['native'])


def prepared_cached_attention(attention, q, k, v, hidden, valid_keys, query_positions, cache):
    """Target softmax and one combined-demo softmax with the original gate."""
    allowed = valid_keys[:, None, None, :k.size(2)]
    if query_positions is not None:
        allowed = allowed & (torch.arange(k.size(2), device=q.device)[None, None, None, :] <= query_positions[:, None, :, None])
    local = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
    if cache is None:
        return local
    rows, width = cache['rows'], q.size(2)
    queries = q.index_select(0, rows)
    if cache['native']:
        cross = attention._apply_varlen_flash_attention(
            queries.transpose(1, 2).reshape(-1, q.size(1), q.size(-1)),
            cache['keys'], cache['values'], cache['cq'] * width, cache['ck'],
            width, cache['max_length'], 0.0, False)
        cross = cross.reshape(-1, width, q.size(1), q.size(-1)).transpose(1, 2)
    else:
        cross = F.scaled_dot_product_attention(queries, cache['padded_keys'], cache['padded_values'], attn_mask=cache['mask'])
    gate = attention.target_demo_gate(hidden.index_select(0, rows)).sigmoid().to(q.dtype).transpose(1, 2).unsqueeze(-1)
    mixed = (1-gate) * local.index_select(0, rows) + gate * cross
    return local.index_copy(0, rows, mixed)
