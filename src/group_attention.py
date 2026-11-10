"""Pair-local prefix attention with the target input as a shared hub.

Layouts contain indices, not learned pair labels. The packed layout batches
each demo's queries against that demo plus the hub, and the hub's queries
against the complete prefix. No cross-demo attention matrix is constructed.
"""

import torch

from common import END_PAIR_TOKEN_ID, START_TOKEN_ID


def role_positions(positions: torch.Tensor, target_input_z: torch.Tensor) -> torch.Tensor:
    """Map grid counters to demo roles 1/2 and target roles 3/4.

    target_input_z is the serialized grid number at the target separator,
    broadcastable to positions[..., 2]. Keep the counters in the caller's
    data/streaming state; only the positions passed to RoPE use fixed roles.
    """
    z = positions[..., 2]
    roles = (z - 1).remainder(2) + 1
    roles = torch.where(z >= target_input_z, roles + 2, roles)
    roles = torch.where(z > 0, roles, 0)
    return torch.cat((positions[..., :2], roles.unsqueeze(-1)), dim=-1)


def padded_prefix_structure(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, prefix_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pair IDs and hub membership, ignoring padding and gold outputs."""
    positions = torch.arange(input_ids.size(1), device=input_ids.device)
    prefix = attention_mask & (positions[None] < prefix_lengths[:, None])
    ends = (input_ids == END_PAIR_TOKEN_ID) & prefix
    pair_ids = ends.long().cumsum(-1) - ends.long()
    target_pair = pair_ids.gather(1, (prefix_lengths.long() - 1)[:, None])
    hub = prefix & ((pair_ids == target_pair) | (input_ids == START_TOKEN_ID))
    return pair_ids, hub


def _cu(lengths: torch.Tensor) -> torch.Tensor:
    return torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).to(torch.int32)


def packed_prefix_layout(
    input_ids: torch.Tensor, cu_seqlens: torch.Tensor, prefix_lengths: torch.Tensor,
) -> dict:
    """Build linear-size ragged Q/KV lists once, shared by every model layer."""
    device = input_ids.device
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
    prefix_lengths = prefix_lengths.long()
    seq_ids = torch.repeat_interleave(torch.arange(lengths.numel(), device=device), lengths)
    offsets = torch.repeat_interleave(cu_seqlens[:-1].long(), lengths)
    positions = torch.arange(input_ids.numel(), device=device) - offsets
    prefix = positions < prefix_lengths[seq_ids]
    ends = (input_ids == END_PAIR_TOKEN_ID) & prefix
    # Sequence starts separate IDs even when the previous sequence has no demos.
    pair_ids = (ends.long() + (positions == 0).long()).cumsum(0) - ends.long()
    target_ids = pair_ids[cu_seqlens[:-1].long() + prefix_lengths - 1]
    hub = prefix & ((pair_ids == target_ids[seq_ids]) | (input_ids == START_TOKEN_ID))
    demos = prefix & ~hub
    demo_indices = demos.nonzero().flatten()
    hub_indices = hub.nonzero().flatten()
    prefix_indices = prefix.nonzero().flatten()
    hub_lengths = torch.zeros_like(lengths).scatter_add_(0, seq_ids, hub.long())
    hub_cu = _cu(hub_lengths)

    _, demo_lengths = torch.unique_consecutive(pair_ids[demo_indices], return_counts=True)
    demo_cu = _cu(demo_lengths)
    if demo_lengths.numel():
        demo_sequences = seq_ids[demo_indices[demo_cu[:-1].long()]]
        kv_lengths = demo_lengths + hub_lengths[demo_sequences]
        kv_cu = _cu(kv_lengths)
        kv_groups = torch.repeat_interleave(torch.arange(demo_lengths.numel(), device=device), kv_lengths)
        kv_positions = torch.arange(kv_groups.numel(), device=device) - kv_cu[:-1].long()[kv_groups]
        local_lengths = demo_lengths[kv_groups]
        # Each group's keys are [own pair, shared hub]. Clamp both gather
        # branches so even the branch discarded by torch.where stays in bounds.
        own = demo_indices[demo_cu[:-1].long()[kv_groups] + torch.minimum(kv_positions, local_lengths - 1)]
        shared = hub_indices[hub_cu[:-1].long()[demo_sequences[kv_groups]] + (kv_positions - local_lengths).clamp(min=0)]
        demo_keys = torch.where(kv_positions < local_lengths, own, shared)
    else:
        kv_lengths = lengths.new_empty(0)
        demo_keys = demo_indices

    query_lengths = torch.cat((demo_lengths, hub_lengths))
    key_lengths = torch.cat((kv_lengths, prefix_lengths))
    return dict(
        prefix_token_indices=torch.cat((demo_indices, hub_indices)),
        prefix_cu_seqlens=_cu(query_lengths),
        max_prefix_len=int(query_lengths.max().item()),
        prefix_kv_indices=torch.cat((demo_keys, prefix_indices)),
        prefix_kv_cu_seqlens=_cu(key_lengths),
        max_prefix_kv_len=int(key_lengths.max().item()),
    )
