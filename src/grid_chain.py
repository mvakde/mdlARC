"""Per-grid target weighting for zero, one or two supervised scratch grids."""
import torch
from common import IO_SEPARATOR_TOKEN_ID


def grid_stages(targets, valid, cu_seqlens=None):
    # A separator belongs to the grid it closes; the following color begins
    # the next grid. Prefix/demonstration separators never advance this counter.
    boundaries = ((targets == IO_SEPARATOR_TOKEN_ID) & valid).long()
    stages = boundaries.cumsum(-1) - boundaries
    if targets.dim() == 1:
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).long()
        ids = torch.repeat_interleave(torch.arange(len(lengths), device=targets.device),
                                      lengths, output_size=targets.numel())
        stages = stages - stages[cu_seqlens[:-1].long()][ids]
    else:
        ids = torch.arange(targets.size(0), device=targets.device)[:, None].expand_as(targets)
    return stages, ids


def balanced_grid_weights(targets, valid, cu_seqlens=None, stages=None):
    """Mean tokens within grids, mean grids within examples, mean examples.

    Returns weights with sum one and the number of participating examples for
    distributed normalization. No token-length or trace-length preference leaks
    into the loss, and an empty head remains finite with zero gradients.
    """
    derived, ids = grid_stages(targets, valid, cu_seqlens)
    stages = derived if stages is None else stages
    batch = targets.size(0) if targets.dim() == 2 else cu_seqlens.numel() - 1
    groups = (ids * 3 + stages.clamp(0, 2)).reshape(-1)
    counts = torch.zeros(batch * 3, device=targets.device).scatter_add(
        0, groups, valid.reshape(-1).float())
    grids_per_example = (counts.reshape(batch, 3) > 0).sum(-1)
    units = (grids_per_example > 0).sum()
    denominator = counts[groups].reshape_as(targets).clamp(min=1)
    denominator = denominator * grids_per_example[ids].clamp(min=1) * units.clamp(min=1)
    return valid.float() / denominator, units


def final_grid_mask(targets, valid, cu_seqlens=None):
    stages, ids = grid_stages(targets, valid, cu_seqlens)
    batch = targets.size(0) if targets.dim() == 2 else cu_seqlens.numel() - 1
    last = torch.zeros(batch, dtype=torch.long, device=targets.device).scatter_reduce(
        0, ids.reshape(-1), (stages * valid).reshape(-1), reduce="amax")
    return valid & (stages == last[ids])
