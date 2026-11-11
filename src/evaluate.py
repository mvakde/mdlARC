"""Evaluation pipeline for the ARC transformer model.

Contains: inference generation, teacher-forced accuracy, AAIVR voting, evaluation orchestration.
For model/data building, see build.py.
"""

import argparse
import contextlib
import gzip
import hashlib
import json
import math
import os
import random
from collections import Counter, deque
from dataclasses import dataclass, field, replace
from itertools import groupby
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F

from build import (
    build_model_and_data,
    load_checkpoint,
)
from common import (
    END_PAIR_TOKEN_ID,
    IO_SEPARATOR_TOKEN_ID,
    NEXT_LINE_TOKEN_ID,
    START_TOKEN_ID,
    VOCAB_SIZE,
    apply_color_permutation_to_grid,
    apply_dihedral_transform,
    apply_inverse_dihedral_transform,
    build_augmentation_schedule,
    build_unique_input_augments,
    color_augmentation_domain,
    _color_mapping_tensor,
    normalize_augmentation_sampling_mode,
    compute_positions_3d,
    grid_to_tokens,
    invert_color_mapping,
    is_rectangular_grid,
    tokens_to_grid,
    IDENTITY_SCALE,
    demo_bounded_scale_specs,
    downscale_grid_exact,
    inference_scale_schedule,
    is_identity_scale,
    oriented_scale_factor,
    symbolic_guard_rejection,
    symbolic_guards,
    upscale_grid,
)
from target_demo_attention import choose_kv_cache_kernel, split_prompt_cache
from tinytransformer import TinyTransformer, build_output_targets_and_mask


# =============================================================================
# Inference Generation
# =============================================================================

# 30*30 colors + 29 row separators + END_PAIR (replaces the last newline).
DEFAULT_MAX_NEW_TOKENS = 30 * (30 + 1)
# Tokens per prefill chunk; lower it via MDLARC_PREFILL_TOKEN_BUDGET when deep loops make KV caches large.
PREFILL_TOKEN_BUDGET = int(os.environ.get("MDLARC_PREFILL_TOKEN_BUDGET", "524288"))
DEFAULT_EVAL_BATCH_SIZE = 120
PROMPT_LENGTH_BUCKET_SIZE = 512
# Split target/demo models decode over a split demo/target KV cache ("split")
# unless MDLARC_DECODE_CACHE=full selects the older single padded buffer.
DECODE_CACHE = os.environ.get("MDLARC_DECODE_CACHE", "split").strip().lower()
# Greedy split-cache steps replay as CUDA graphs unless MDLARC_DECODE_CUDA_GRAPHS=0.
DECODE_CUDA_GRAPHS = os.environ.get("MDLARC_DECODE_CUDA_GRAPHS", "1").strip() != "0"
if DECODE_CACHE not in ("split", "full"):
    raise ValueError("MDLARC_DECODE_CACHE must be split or full.")


def _grid_update(state, token_ids, start_id, sep_id, end_id, nl_id):
    x, y, z = state.unbind(-1)
    pos_x = torch.clamp(x, min=0, max=30)
    pos_y = torch.clamp(y, min=0, max=29)
    pos_z = z

    is_start = token_ids == start_id
    is_sep = token_ids == sep_id
    is_end = token_ids == end_id
    is_newline = token_ids == nl_id

    zeros = torch.zeros_like(x)
    pos_z = torch.where(is_start, zeros, pos_z)

    next_x = x + 1
    next_y = y
    next_z = z
    next_x = torch.where(is_newline, zeros, next_x)
    next_y = torch.where(is_newline, y + 1, next_y)
    is_boundary = is_sep | is_end
    next_x = torch.where(is_boundary | is_start, zeros, next_x)
    next_y = torch.where(is_boundary | is_start, zeros, next_y)
    next_z = torch.where(is_boundary, z + 1, next_z)
    next_z = torch.where(is_start, torch.ones_like(next_z), next_z)

    return torch.stack([next_x, next_y, next_z], dim=-1), torch.stack([pos_x, pos_y, pos_z], dim=-1)


_compiled_grid_update = torch.compile(_grid_update, mode="default", fullgraph=True)


def _speculative_grid_positions(state, tokens):
    states, positions = [state], []
    for index in range(tokens.size(1)):
        state, position = _grid_update(state, tokens[:, index], START_TOKEN_ID,
                                       IO_SEPARATOR_TOKEN_ID, END_PAIR_TOKEN_ID, NEXT_LINE_TOKEN_ID)
        states.append(state)
        positions.append(position)
    return torch.stack(states, dim=1), torch.stack(positions, dim=1)


def _commit_speculative_tokens(draft, targets, hidden, grid_states, emitted, finished, generated, token_limit):
    """Accept the main-head-approved draft prefix plus its correction.

    Returns (anchor, grid_state, advance, emitted, finished, pending); advance is
    each row's number of cached block tokens to keep (the pending token and the
    accepted drafts). The correction stays uncached as the next pending token.
    """
    width = targets.size(1)
    accepted = (draft == targets[:, :-1]).long().cumprod(dim=1).sum(dim=1)
    correction = targets.gather(1, accepted[:, None]).squeeze(1)
    outgoing = torch.cat([draft, correction[:, None]], dim=1)
    outgoing = outgoing.scatter(1, accepted[:, None], correction[:, None])
    columns = torch.arange(width, device=targets.device)[None, :]
    active = ~finished
    is_end = outgoing == END_PAIR_TOKEN_ID
    before_end = is_end.long().cumsum(dim=1) - is_end.long() == 0
    valid = active[:, None] & (columns <= accepted[:, None]) & before_end & (columns < (token_limit - emitted)[:, None])
    # Extra scratch columns let inactive/over-budget rows write without
    # duplicate scatter indices or overwriting their last emitted token.
    slots = torch.where(active[:, None], emitted[:, None] + columns, token_limit + columns)
    generated.scatter_(1, slots, torch.where(valid, outgoing, END_PAIR_TOKEN_ID))
    emitted = emitted + valid.long().sum(dim=1)
    finished = finished | (valid & is_end).any(dim=1) | (emitted >= token_limit)
    advance = torch.where(active, accepted + 1, 0)
    anchor = hidden.gather(1, accepted[:, None, None].expand(-1, 1, hidden.size(-1))).squeeze(1)
    grid_state = grid_states.gather(1, (accepted + 1)[:, None, None].expand(-1, 1, 3)).squeeze(1)
    grid_state = torch.where(active[:, None], grid_state, grid_states[:, 0])
    pending = torch.where(finished, END_PAIR_TOKEN_ID, correction)
    return anchor, grid_state, advance, emitted, finished, pending


def _commit_speculative_block(draft, targets, hidden, grid_states, cache_lengths,
                              attention_mask, emitted, finished, generated, token_limit):
    """Commit a main-head-approved prefix and leave the correction uncached."""
    anchor, grid_state, advance, emitted, finished, pending = _commit_speculative_tokens(
        draft, targets, hidden, grid_states, emitted, finished, generated, token_limit)
    cache_lengths = cache_lengths + advance
    attention_mask &= torch.arange(attention_mask.size(1), device=targets.device)[None, :] < cache_lengths[:, None]
    return anchor, grid_state, cache_lengths, emitted, finished, pending


def _finish_generated_sequences(prompts, generated):
    results = []
    for prompt, row in zip(prompts, generated.tolist()):
        if END_PAIR_TOKEN_ID in row:
            row = row[:row.index(END_PAIR_TOKEN_ID) + 1]
        results.append(list(prompt) + row)
    return results


def _batched_speculative_decode(model, outputs, past_key_values, attention_mask,
                                initial_state, finished, current_len, token_limit, shared):
    """One pending main-head token plus K-1 auxiliary proposals per pass."""
    batch_size, capacity = attention_mask.shape
    latent_draft = getattr(model.config, "next_latent_prediction", False)
    width_limit = model.config.next_latent_horizon + 1 if latent_draft else model.config.n_predict
    if not getattr(model, "_speculative_mode_reported", False):
        print(f"Speculative decoding: {'NextLat' if latent_draft else 'MTP'}; "
              f"{width_limit - 1} draft tokens per verification; greedy main-head verification.", flush=True)
        model._speculative_mode_reported = True
    device = attention_mask.device
    generated = torch.full((batch_size, token_limit + width_limit), END_PAIR_TOKEN_ID, dtype=torch.long, device=device)
    if token_limit == 0:
        return generated[:, :0]
    pending = torch.where(finished, END_PAIR_TOKEN_ID, outputs["logits"][:, -1].argmax(dim=-1))
    generated[:, 0] = pending
    emitted = torch.ones(batch_size, dtype=torch.long, device=device)
    finished = finished | (pending == END_PAIR_TOKEN_ID) | (emitted >= token_limit)
    cache_lengths = torch.full_like(emitted, current_len)
    grid_state = initial_state
    anchor = outputs["hidden_states"][:, -1]
    if not hasattr(model, "_compiled_verify"):
        model._compiled_verify = torch.compile(model.forward_verify, mode="default", fullgraph=True)
    if latent_draft and not hasattr(model, "_compiled_next_latent"):
        model._compiled_next_latent = torch.compile(model.predict_next_latent, mode="default", fullgraph=True)
    if not latent_draft and not hasattr(model, "_compiled_mtp"):
        model._compiled_mtp = torch.compile(model.predict_mtp, mode="default", fullgraph=True)
    if not hasattr(model, "_compiled_spec_positions"):
        model._compiled_spec_positions = torch.compile(_speculative_grid_positions, mode="default", fullgraph=True)
        model._compiled_spec_commit = torch.compile(_commit_speculative_block, mode="default", fullgraph=True)

    while not bool(finished.all()):
        torch.compiler.cudagraph_mark_step_begin()
        # Shrink only at cache capacity; draft rejection never changes width.
        largest_start = int(torch.where(finished, 0, cache_lengths).max().item())
        width = min(width_limit, capacity - largest_start)
        if width > 1:
            proposals = model._compiled_next_latent(anchor, pending) if latent_draft else model._compiled_mtp(anchor)
            draft = proposals[:, :width - 1]
        else:
            draft = pending.new_empty((batch_size, 0))
        tokens = torch.cat([pending[:, None], draft], dim=1)
        starts = torch.where(finished, cache_lengths.clamp(max=capacity - width), cache_lengths)
        slots = starts[:, None] + torch.arange(width, device=device)[None, :]
        grid_states, positions = model._compiled_spec_positions(grid_state, tokens)
        key_valid = (~finished)[:, None] & (tokens != END_PAIR_TOKEN_ID).long().cumprod(dim=1).bool()
        key_valid = torch.where(finished[:, None], attention_mask.gather(1, slots), key_valid)
        attention_mask.scatter_(1, slots, key_valid)
        block_mask = model.build_decode_block_mask_for_step(attention_mask, slots, capacity, device) if device.type == "cuda" else None
        verified = model._compiled_verify(input_ids=tokens, positions_3d=positions,
                                          past_key_values=tuple(past_key_values), attention_mask=attention_mask,
                                          cache_position=slots, decode_block_mask=block_mask,
                                          **shared)
        active_rows = (~finished).long().sum()
        previous_lengths = cache_lengths
        anchor, grid_state, cache_lengths, emitted, finished, pending = model._compiled_spec_commit(
            draft, verified["logits"].argmax(dim=-1), verified["hidden_states"], grid_states,
            cache_lengths, attention_mask, emitted, finished, generated, token_limit)
        # Diagnostics only, kept on device: each active row caches its accepted
        # drafts plus the pending token, so the length change counts acceptances.
        _speculative_stats(model, device).add_(torch.stack([
            active_rows, (cache_lengths - previous_lengths).sum() - active_rows, active_rows * (width - 1)]))
    return generated[:, :token_limit]


def _speculative_stats(model, device):
    """[active row verifications, accepted drafts, proposed drafts] for this process."""
    stats = getattr(model, "_speculative_acceptance", None)
    if stats is None:
        stats = model._speculative_acceptance = torch.zeros(3, dtype=torch.long, device=device)
    return stats


def speculative_summary(model) -> Optional[str]:
    stats = getattr(model, "_speculative_acceptance", None)
    if stats is None:
        return None
    verifications, accepted, proposed = (int(value) for value in stats.tolist())
    rate = accepted / proposed if proposed else 0.0
    per_pass = (accepted + verifications) / verifications if verifications else 0.0
    return (f"Speculative acceptance: {accepted}/{proposed} drafts ({100 * rate:.2f}%), "
            f"{per_pass:.3f} tokens per verification over {verifications} row verifications")


class _DoneMonitor:
    """Sync-free stop check for a decode loop.

    Each step copies its all-rows-finished flag into pinned host memory and
    records an event. Before step i the loop reads the flag of step i - LAG,
    whose event has long completed, so the CPU never waits on the step it just
    queued and stays at most LAG steps ahead of the GPU. The cost is up to
    LAG - 1 extra steps after the last row finishes; finished rows only emit
    END, so outputs are unchanged. On the CPU the flag is read directly.
    """

    LAG = 4

    def __init__(self, flag: torch.Tensor) -> None:
        self.flag = flag
        self.cuda = flag.device.type == "cuda"
        if self.cuda:
            self.host = torch.zeros(self.LAG, dtype=torch.bool, pin_memory=True)
            self.events = [torch.cuda.Event() for _ in range(self.LAG)]

    def record(self, index: int) -> None:
        if self.cuda:
            slot = index % self.LAG
            self.host[slot:slot + 1].copy_(self.flag.view(1), non_blocking=True)
            self.events[slot].record()

    def done(self, index: int) -> bool:
        if not self.cuda:
            return bool(self.flag)
        if index < self.LAG:
            return False
        slot = index % self.LAG
        self.events[slot].synchronize()
        return bool(self.host[slot])


# Captured graphs and capture failures, for tests and logs.
DECODE_GRAPH_STATS = {"captured": 0, "failed": 0}


class _GraphedStep:
    """Runs one decode step function per call, replayed as a CUDA graph.

    The step reads and writes only fixed buffers. The first call runs a real
    step on a side stream, which compiles kernels and initializes libraries
    outside capture, then captures the same step without running it; every
    later call replays the graph. Each call therefore executes exactly one step.
    A failed capture is reported once per model and the step runs uncaptured.
    """

    def __init__(self, function, enabled: bool, owner) -> None:
        self.function = function
        self.enabled = enabled and not getattr(owner, "_decode_graph_failed", False)
        self.owner = owner
        self.graph = None

    def __call__(self) -> None:
        if self.graph is not None:
            self.graph.replay()
            return
        if not self.enabled:
            self.function()
            return
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            self.function()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                self.function()
        except Exception as error:  # noqa: BLE001 - keep the evaluation running uncaptured
            DECODE_GRAPH_STATS["failed"] += 1
            self.enabled = False
            self.owner._decode_graph_failed = True
            print(f"CUDA graph capture of the decode step failed ({type(error).__name__}: {error}); "
                  "decoding continues without graphs.", flush=True)
            return
        DECODE_GRAPH_STATS["captured"] += 1
        self.graph = graph


def _compiled_step(model, function):
    """One compiled entry per step function and model (own Dynamo cache per function)."""
    name = f"_compiled{function.__name__}"
    compiled = model.__dict__.get(name)
    if compiled is None:
        compiled = torch.compile(function, mode="default", fullgraph=True)
        setattr(model, name, compiled)
    return compiled


@contextlib.contextmanager
def _decode_compile_settings():
    """Compile a decode step once for all batch shapes.

    Without this, every batch's new sizes (rows, demo and target slots), and
    coincidences between them under duck shaping, recompiled the step; at the
    recompile limit Dynamo silently ran it eagerly. Sizes are marked dynamic
    up front, equal sizes stay independent symbols, and the limit is raised.
    """
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch._dynamo.config.patch(recompile_limit=64))
        shape_config = getattr(getattr(torch.fx, "experimental", None), "_config", None)
        if shape_config is not None and hasattr(shape_config, "use_duck_shape"):
            stack.enter_context(shape_config.patch(use_duck_shape=False))
        yield


def _mark_dynamic(*entries):
    """Trace the varying dims (rows, cache slots) as symbols: ``(tensor, dims)`` pairs."""
    mark = getattr(torch._dynamo, "maybe_mark_dynamic", None)
    if mark is None:
        return
    for tensor, dims in entries:
        if isinstance(tensor, torch.Tensor):
            for dim in dims:
                if dim < tensor.dim():
                    mark(tensor, dim)


def _mark_decode_dynamic(cache, *row_tensors):
    """Batch rows and split-cache slots vary per decode batch."""
    layers, demo_lengths, target_lengths, _ = cache
    _mark_dynamic(*((tensor, (0,)) for tensor in (demo_lengths, target_lengths, *row_tensors)),
                  *((tensor, (0, 1)) for layer in layers for tensor in layer))


def _split_decode_step(model, next_token, logits, generated, step, finished, grid_state, cache, shared, all_done):
    """One cached split-cache step over fixed buffers, with no host reads.

    Picks the greedy token from ``logits`` unless ``next_token`` is given,
    records it at column ``step``, advances the grid state, runs the cached
    forward (which writes the token's keys at target_lengths) and leaves the
    next logits in ``logits``. Finished rows keep emitting END.
    """
    if next_token is None:
        next_token = logits[:, -1].argmax(dim=-1)
    next_token = torch.where(finished, END_PAIR_TOKEN_ID, next_token)
    generated.scatter_(1, step.expand(next_token.size(0), 1), next_token[:, None])
    finished |= next_token == END_PAIR_TOKEN_ID
    state, positions = _grid_update(grid_state, next_token, START_TOKEN_ID, IO_SEPARATOR_TOKEN_ID,
                                    END_PAIR_TOKEN_ID, NEXT_LINE_TOKEN_ID)
    grid_state.copy_(state)
    output = model.forward_generate(input_ids=next_token[:, None], positions_3d=positions[:, None],
                                    split_cache=cache, **shared)
    logits.copy_(output["logits"])
    cache[2].add_(1)
    step.add_(1)
    all_done.copy_(finished.all())


def _split_greedy_decode(model, prefill_logits, cache, initial_state, finished, token_limit, shared,
                         temperature, top_k, use_graphs):
    """Token-at-a-time decode over a split cache; greedy steps replay as CUDA graphs."""
    batch_size, device = finished.size(0), finished.device
    generated = torch.full((batch_size, token_limit), END_PAIR_TOKEN_ID, dtype=torch.long, device=device)
    if token_limit == 0:
        return generated
    greedy = (temperature is not None and temperature <= 0) or (temperature is None and top_k is None) or top_k == 1
    logits = prefill_logits[:, -1:].clone()
    step = torch.zeros(1, dtype=torch.long, device=device)
    finished = finished.clone()
    grid_state = initial_state.to(dtype=torch.int32).clone()
    all_done = torch.zeros((), dtype=torch.bool, device=device)
    compiled = _compiled_step(model, _split_decode_step)
    buffers = (logits, generated, step, finished, grid_state, cache, shared, all_done)
    _mark_decode_dynamic(cache, logits, generated, finished, grid_state, *_shared_row_tensors(shared))

    def run(next_token=None):
        with _decode_compile_settings():
            compiled(model, next_token, *buffers)

    runner = _GraphedStep(run, use_graphs and greedy and device.type == "cuda", model)
    monitor = _DoneMonitor(all_done)
    for index in range(token_limit):
        if monitor.done(index):
            break
        if greedy:
            runner()
        else:
            run(_select_next_token(logits, temperature=temperature, top_k=top_k))
        monitor.record(index)
    return generated


def _shared_row_tensors(shared):
    """Per-row tensors among the shared forward inputs (value embeddings may be None)."""
    rows = [shared.get("example_ids"), shared.get("dihedral_ids"), shared.get("conditioning_embeds")]
    rows.extend(shared.get("value_embeds") or ())
    return [tensor for tensor in rows if isinstance(tensor, torch.Tensor)]


def _split_speculative_step(model, anchor, pending, grid_state, cache, emitted, finished, generated, shared,
                            all_done, stats, *, latent_draft, width, token_limit):
    """Draft, verify and commit one block over fixed buffers, with no host reads."""
    if latent_draft:
        proposals = model.predict_next_latent(anchor, pending)
    else:
        proposals = model.predict_mtp(anchor)
    draft = proposals[:, :width - 1]
    tokens = torch.cat([pending[:, None], draft], dim=1)
    grid_states, positions = _speculative_grid_positions(grid_state, tokens)
    verified = model.forward_generate(input_ids=tokens, positions_3d=positions, split_cache=cache,
                                      return_hidden_states=True, **shared)
    active_rows = (~finished).long().sum()
    new_anchor, new_state, advance, new_emitted, new_finished, new_pending = _commit_speculative_tokens(
        draft, verified["logits"].argmax(dim=-1), verified["hidden_states"], grid_states,
        emitted, finished, generated, token_limit)
    # Diagnostics only, kept on device: each active row caches its accepted drafts plus the
    # pending token, so advance counts acceptances (see speculative_summary).
    stats.add_(torch.stack([active_rows, advance.sum() - active_rows, active_rows * (width - 1)]))
    cache[2].add_(advance.to(cache[2].dtype))
    anchor.copy_(new_anchor)
    grid_state.copy_(new_state)
    emitted.copy_(new_emitted)
    finished.copy_(new_finished)
    pending.copy_(new_pending)
    all_done.copy_(finished.all())


def _split_speculative_decode(model, outputs, cache, initial_state, finished, token_limit, width, shared, use_graphs):
    """Self-speculative greedy decode over a split cache.

    The target buffers leave ``width`` spare slots, so every block has the full
    width (no capacity check); stale keys past a row's length are never read.
    """
    latent_draft = getattr(model.config, "next_latent_prediction", False)
    if not getattr(model, "_speculative_mode_reported", False):
        print(f"Speculative decoding: {'NextLat' if latent_draft else 'MTP'}; "
              f"{width - 1} draft tokens per verification; greedy main-head verification.", flush=True)
        model._speculative_mode_reported = True
    batch_size, device = finished.size(0), finished.device
    generated = torch.full((batch_size, token_limit + width), END_PAIR_TOKEN_ID, dtype=torch.long, device=device)
    if token_limit == 0:
        return generated[:, :0]
    pending = torch.where(finished, END_PAIR_TOKEN_ID, outputs["logits"][:, -1].argmax(dim=-1))
    generated[:, 0] = pending
    emitted = torch.ones(batch_size, dtype=torch.long, device=device)
    finished = finished | (pending == END_PAIR_TOKEN_ID) | (emitted >= token_limit)
    anchor = outputs["hidden_states"][:, -1].clone()
    grid_state = initial_state.clone()
    all_done = finished.all().clone()
    compiled = _compiled_step(model, _split_speculative_step)
    stats = _speculative_stats(model, device)
    buffers = (anchor, pending, grid_state, cache, emitted, finished, generated, shared, all_done, stats)
    _mark_decode_dynamic(cache, anchor, pending, grid_state, emitted, finished, generated, *_shared_row_tensors(shared))

    def run():
        with _decode_compile_settings():
            compiled(model, *buffers, latent_draft=latent_draft, width=width, token_limit=token_limit)

    runner = _GraphedStep(run, use_graphs and device.type == "cuda", model)
    monitor = _DoneMonitor(all_done)
    # Every pass emits at least one token for each unfinished row.
    for index in range(token_limit):
        if monitor.done(index):
            break
        runner()
        monitor.record(index)
    return generated[:, :token_limit]


def _full_decode_step(model, next_token, logits, generated, step, finished, grid_state, past_key_values,
                      attention_mask, cache_position, shared, all_done):
    """The ordinary step over the single padded cache, on fixed buffers (for graph replay)."""
    if next_token is None:
        next_token = logits[:, -1].argmax(dim=-1)
    next_token = torch.where(finished, END_PAIR_TOKEN_ID, next_token)
    generated.scatter_(1, step.expand(next_token.size(0), 1), next_token[:, None])
    finished |= next_token == END_PAIR_TOKEN_ID
    state, positions = _grid_update(grid_state, next_token, START_TOKEN_ID, IO_SEPARATOR_TOKEN_ID,
                                    END_PAIR_TOKEN_ID, NEXT_LINE_TOKEN_ID)
    grid_state.copy_(state)
    attention_mask.scatter_(1, cache_position.expand(next_token.size(0), 1), (~finished)[:, None])
    output = model.forward_generate(input_ids=next_token[:, None], positions_3d=positions[:, None],
                                    past_key_values=past_key_values, attention_mask=attention_mask,
                                    cache_position=cache_position, **shared)
    logits.copy_(output["logits"])
    cache_position.add_(1)
    step.add_(1)
    all_done.copy_(finished.all())


def _full_speculative_step(model, anchor, pending, grid_state, past_key_values, attention_mask, cache_lengths,
                           emitted, finished, generated, shared, all_done, stats, *, latent_draft, width, token_limit):
    """The speculative block over the single padded cache, on fixed buffers (for graph replay).

    The buffer keeps ``width`` spare slots, so blocks never shrink at capacity.
    """
    if latent_draft:
        proposals = model.predict_next_latent(anchor, pending)
    else:
        proposals = model.predict_mtp(anchor)
    draft = proposals[:, :width - 1]
    tokens = torch.cat([pending[:, None], draft], dim=1)
    capacity = attention_mask.size(1)
    starts = torch.where(finished, cache_lengths.clamp(max=capacity - width), cache_lengths)
    slots = starts[:, None] + torch.arange(width, device=tokens.device)[None, :]
    grid_states, positions = _speculative_grid_positions(grid_state, tokens)
    key_valid = (~finished)[:, None] & (tokens != END_PAIR_TOKEN_ID).long().cumprod(dim=1).bool()
    key_valid = torch.where(finished[:, None], attention_mask.gather(1, slots), key_valid)
    attention_mask.scatter_(1, slots, key_valid)
    verified = model.forward_generate(input_ids=tokens, positions_3d=positions, past_key_values=past_key_values,
                                      attention_mask=attention_mask, cache_position=slots,
                                      return_hidden_states=True, **shared)
    active_rows = (~finished).long().sum()
    new_anchor, new_state, new_lengths, new_emitted, new_finished, new_pending = _commit_speculative_block(
        draft, verified["logits"].argmax(dim=-1), verified["hidden_states"], grid_states,
        cache_lengths, attention_mask, emitted, finished, generated, token_limit)
    stats.add_(torch.stack([active_rows, (new_lengths - cache_lengths).sum() - active_rows,
                            active_rows * (width - 1)]))
    anchor.copy_(new_anchor)
    grid_state.copy_(new_state)
    cache_lengths.copy_(new_lengths)
    emitted.copy_(new_emitted)
    finished.copy_(new_finished)
    pending.copy_(new_pending)
    all_done.copy_(finished.all())


def _graphed_full_decode(model, outputs, prompt_kvs, attention_mask, demo_mask, initial_state, finished,
                         current_len, token_limit, width, shared, speculative):
    """Greedy decode over the single padded cache with sync-free, graph-replayed steps.

    For A/B comparison with the split cache (MDLARC_DECODE_CACHE=full on CUDA).
    """
    batch_size, device = finished.size(0), finished.device
    capacity = (current_len + token_limit + width + 127) // 128 * 128
    past_key_values = []
    for keys, values in prompt_kvs:
        buffers = []
        for tensor in (keys, values):
            buffer = tensor.new_zeros((*tensor.shape[:2], capacity, tensor.size(-1)))
            buffer[:, :, :current_len] = tensor
            buffers.append(buffer)
        past_key_values.append(tuple(buffers))
    past_key_values = tuple(past_key_values)
    mask = torch.zeros((batch_size, capacity), dtype=torch.bool, device=device)
    mask[:, :current_len] = attention_mask
    full_demo = torch.zeros_like(mask)
    full_demo[:, :current_len] = demo_mask
    shared = dict(shared, demo_attention_mask=full_demo)
    all_done = torch.zeros((), dtype=torch.bool, device=device)
    dims = [(tensor, (0, 2)) for layer in past_key_values for tensor in layer]
    dims += [(mask, (0, 1)), (full_demo, (0, 1))]
    dims += [(tensor, (0,)) for tensor in _shared_row_tensors(shared)]
    if speculative:
        latent_draft = getattr(model.config, "next_latent_prediction", False)
        generated = torch.full((batch_size, token_limit + width), END_PAIR_TOKEN_ID, dtype=torch.long, device=device)
        if token_limit == 0:
            return generated[:, :0]
        pending = torch.where(finished, END_PAIR_TOKEN_ID, outputs["logits"][:, -1].argmax(dim=-1))
        generated[:, 0] = pending
        emitted = torch.ones(batch_size, dtype=torch.long, device=device)
        finished = finished | (pending == END_PAIR_TOKEN_ID) | (emitted >= token_limit)
        cache_lengths = torch.full_like(emitted, current_len)
        anchor = outputs["hidden_states"][:, -1].clone()
        grid_state = initial_state.clone()
        all_done.copy_(finished.all())
        compiled = _compiled_step(model, _full_speculative_step)
        buffers = (anchor, pending, grid_state, past_key_values, mask, cache_lengths, emitted, finished,
                   generated, shared, all_done, _speculative_stats(model, device))
        _mark_dynamic(*dims, *((tensor, (0,)) for tensor in (anchor, pending, grid_state, cache_lengths,
                                                              emitted, finished, generated)))

        def run():
            with _decode_compile_settings():
                compiled(model, *buffers, latent_draft=latent_draft, width=width, token_limit=token_limit)
    else:
        generated = torch.full((batch_size, token_limit), END_PAIR_TOKEN_ID, dtype=torch.long, device=device)
        if token_limit == 0:
            return generated
        logits = outputs["logits"][:, -1:].clone()
        step = torch.zeros(1, dtype=torch.long, device=device)
        cache_position = torch.tensor([current_len], dtype=torch.long, device=device)
        finished = finished.clone()
        grid_state = initial_state.to(dtype=torch.int32).clone()
        compiled = _compiled_step(model, _full_decode_step)
        buffers = (logits, generated, step, finished, grid_state, past_key_values, mask, cache_position,
                   shared, all_done)
        _mark_dynamic(*dims, *((tensor, (0,)) for tensor in (logits, generated, finished, grid_state)))

        def run():
            with _decode_compile_settings():
                compiled(model, None, *buffers)
    runner = _GraphedStep(run, True, model)
    monitor = _DoneMonitor(all_done)
    for index in range(token_limit):
        if monitor.done(index):
            break
        runner()
        monitor.record(index)
    return generated[:, :token_limit]


class BatchGridState:
    def __init__(self, initial_state: torch.Tensor) -> None:
        self.state = initial_state.to(dtype=torch.int32)

    def update(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.view(-1).to(device=self.state.device)
        self.state, positions = _compiled_grid_update(
            self.state, token_ids, START_TOKEN_ID, IO_SEPARATOR_TOKEN_ID, END_PAIR_TOKEN_ID, NEXT_LINE_TOKEN_ID
        )
        return positions


def _select_next_token(logits: torch.Tensor, temperature: Optional[float] = None, top_k: Optional[int] = None) -> torch.Tensor:
    last_logits = logits[:, -1, :]
    if (temperature is None and top_k is None) or (temperature is not None and temperature <= 0):
        return torch.argmax(last_logits, dim=-1)
    if temperature is None:
        temperature = 1.0
    use_top_k = top_k is not None and top_k > 0
    if use_top_k:
        top_k = int(top_k)
        if top_k == 1:
            return torch.argmax(last_logits, dim=-1)
        top_k = min(top_k, last_logits.size(-1))
        top_values, top_indices = torch.topk(last_logits, top_k, dim=-1)
        scaled = (top_values / temperature).float()
        probs = torch.softmax(scaled, dim=-1)
        next_index = torch.multinomial(probs, num_samples=1)
        return top_indices.gather(-1, next_index).squeeze(-1)
    scaled = (last_logits / temperature).float()
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def _left_pad_sequences(
    sequences: Sequence[Sequence[int]],
    pad_token_id: int,
    device: torch.device,
    padded_length: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        raise ValueError("Expected at least one sequence.")
    batch_size = len(sequences)
    longest_sequence = max(len(seq) for seq in sequences)
    max_len = longest_sequence if padded_length is None else int(padded_length)
    if max_len < longest_sequence:
        raise ValueError(
            f"padded_length={max_len} is shorter than the longest sequence "
            f"({longest_sequence})."
        )
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    for idx, seq in enumerate(sequences):
        seq_len = len(seq)
        start = max_len - seq_len
        input_ids[idx, start:] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[idx, start:] = True
    return input_ids, attention_mask


def _pad_cached_positions(cached_positions: Sequence[torch.Tensor], max_len: int, device: torch.device) -> torch.Tensor:
    positions = torch.zeros((len(cached_positions), max_len, 3), dtype=torch.int32, device=device)
    for idx, pos in enumerate(cached_positions):
        seq_len = pos.size(0)
        start = max_len - seq_len
        positions[idx, start:] = pos.to(device=device, dtype=torch.int32)
    return positions


def _derive_initial_state_from_prompt(input_ids: torch.Tensor, positions_3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    last_tokens = input_ids[:, -1]
    last_positions = positions_3d[:, -1]
    x, y, z = last_positions.unbind(-1)

    next_x = x + 1
    next_y = y
    next_z = z
    next_x = torch.where(last_tokens == NEXT_LINE_TOKEN_ID, torch.zeros_like(next_x), next_x)
    next_y = torch.where(last_tokens == NEXT_LINE_TOKEN_ID, y + 1, next_y)
    is_boundary = (last_tokens == IO_SEPARATOR_TOKEN_ID) | (last_tokens == END_PAIR_TOKEN_ID)
    next_x = torch.where(is_boundary, torch.zeros_like(next_x), next_x)
    next_y = torch.where(is_boundary, torch.zeros_like(next_y), next_y)
    next_z = torch.where(is_boundary, z + 1, next_z)
    next_x = torch.where(last_tokens == START_TOKEN_ID, torch.zeros_like(next_x), next_x)
    next_y = torch.where(last_tokens == START_TOKEN_ID, torch.zeros_like(next_y), next_y)
    next_z = torch.where(last_tokens == START_TOKEN_ID, torch.ones_like(next_z), next_z)

    initial_state = torch.stack([next_x, next_y, next_z], dim=-1)
    finished = last_tokens == END_PAIR_TOKEN_ID
    return initial_state, finished


@dataclass
class PromptPrefill:
    outputs: Dict[str, object]
    # Prompt KVs copied into (batch, heads, max_model_len, dim) buffers; None when the
    # caller asked for the raw prompt cache only (``full_cache=False``).
    past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    attention_mask: torch.Tensor  # (batch, max_model_len) key validity, prompt part filled
    current_len: int
    max_model_len: int
    initial_state: torch.Tensor
    finished: torch.Tensor
    example_ids: torch.Tensor
    dihedral_ids: torch.Tensor
    conditioning_embeds: torch.Tensor
    value_embeds: object
    split_kwargs: Dict[str, torch.Tensor]
    prompt_attention_mask: Optional[torch.Tensor] = None  # (batch, current_len)
    prompt_demo_mask: Optional[torch.Tensor] = None  # (batch, current_len), split models only


def _full_kv_buffers(prompt_kvs, max_model_len: int, device) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    past_key_values = []
    for k, v in prompt_kvs:
        B, H, L, D = k.shape
        k_buf = torch.zeros((B, H, max_model_len, D), dtype=torch.bfloat16, device=device)
        v_buf = torch.zeros((B, H, max_model_len, D), dtype=torch.bfloat16, device=device)
        k_buf[:, :, :L, :] = k
        v_buf[:, :, :L, :] = v
        past_key_values.append((k_buf, v_buf))
    return past_key_values


@torch.inference_mode()
def _prefill_prompts(
    model, prompts, example_ids, device, *, max_new_tokens: int, cached_positions=None,
    padded_prompt_length: Optional[int] = None, dihedral_ids=None, pair_identity_ids=None,
    return_hidden_states: bool = False, full_cache: bool = True,
) -> PromptPrefill:
    """Left-pad and prefill prompts into KV buffers with room for ``max_new_tokens`` more tokens.

    With ``full_cache=False`` the prompt KVs stay in ``outputs["past_key_values"]`` and no
    padded decode buffers are allocated (the split-cache decode builds its own).
    """
    batch_size = len(prompts)
    example_ids_tensor = torch.tensor(example_ids, dtype=torch.int32, device=device)
    dihedral_ids_tensor = torch.tensor(
        [0] * batch_size if dihedral_ids is None else dihedral_ids,
        dtype=torch.int32, device=device,
    )
    input_ids, attention_mask = _left_pad_sequences(
        prompts,
        END_PAIR_TOKEN_ID,
        device,
        padded_length=padded_prompt_length,
    )
    current_len = input_ids.size(1)
    identity_kwargs = {}
    if getattr(model.config, "pair_identity_vertices", 0):
        if pair_identity_ids is None:
            raise ValueError("Simplex inference requires prompt pair identities")
        prompt_pair_ids = torch.zeros_like(input_ids)
        for row, ids in enumerate(pair_identity_ids):
            prompt_pair_ids[row, -len(ids):] = torch.as_tensor(ids, device=device)
        identity_kwargs["pair_identity_ids"] = prompt_pair_ids
    batch_max_needed = current_len + max_new_tokens
    batch_max_needed = (batch_max_needed + 127) // 128 * 128
    max_model_len = min(batch_max_needed, model.config.max_seq_len)

    if cached_positions and all(p is not None for p in cached_positions):
        prompt_positions = _pad_cached_positions([p for p in cached_positions if p is not None], input_ids.size(1), device)
    else:
        prompt_positions = compute_positions_3d(input_ids, attention_mask).to(device=device, dtype=torch.int32)

    initial_state, finished = _derive_initial_state_from_prompt(input_ids, prompt_positions)

    full_attention_mask = torch.zeros((batch_size, max_model_len), dtype=torch.bool, device=device)
    full_attention_mask[:, :current_len] = attention_mask
    split_kwargs = {}
    demo_mask = None
    if getattr(model.config, "split_target_demo_attention", False):
        # Keep prefix membership explicitly with the KV cache. Output slots
        # always belong to the target branch, including speculative blocks.
        z = prompt_positions[..., 2]
        demo_mask = attention_mask & (z > 0) & (z < z[:, -1:])
        full_demo_mask = torch.zeros_like(full_attention_mask)
        full_demo_mask[:, :current_len] = demo_mask
        split_kwargs["demo_attention_mask"] = full_demo_mask

    conditioning_embeds = model.compute_conditioning_embeds(
        example_ids_tensor,
        dihedral_ids_tensor,
    ).to(dtype=model.conditioning_dtype)
    value_embeds = model.compute_value_embeds(example_ids_tensor)
    outputs = model.forward_generate(
        input_ids=input_ids, example_ids=example_ids_tensor, past_key_values=None,
        dihedral_ids=dihedral_ids_tensor,
        positions_3d=prompt_positions, attention_mask=attention_mask, conditioning_embeds=conditioning_embeds,
        value_embeds=value_embeds,
        return_hidden_states=return_hidden_states,
        **identity_kwargs,
    )
    past_key_values = _full_kv_buffers(outputs["past_key_values"], max_model_len, device) if full_cache else None
    return PromptPrefill(outputs, past_key_values, full_attention_mask, current_len, max_model_len,
                         initial_state, finished, example_ids_tensor, dihedral_ids_tensor,
                         conditioning_embeds, value_embeds, split_kwargs,
                         prompt_attention_mask=attention_mask, prompt_demo_mask=demo_mask)


@torch.inference_mode()
def batched_greedy_generate(
    model, prompts, example_ids, device, max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    cached_positions=None, temperature: Optional[float] = None, top_k: Optional[int] = None,
    padded_prompt_length: Optional[int] = None,
    dihedral_ids=None,
    pair_identity_ids=None,
    speculative: bool = False,
    decode_cache: Optional[str] = None,
    cuda_graphs: Optional[bool] = None,
):
    """Batched cached decoding of left-padded prompts.

    Split target/demo models decode over a split demo/target cache read by
    FlashAttention's cache kernel when available; greedy steps (ordinary and
    speculative) are sync-free and replay as CUDA graphs. ``decode_cache``
    ("split" or "full") and ``cuda_graphs`` override MDLARC_DECODE_CACHE and
    MDLARC_DECODE_CUDA_GRAPHS for this call.
    """
    model.eval()
    decode_cache = (decode_cache or DECODE_CACHE).lower()
    if decode_cache not in ("split", "full"):
        raise ValueError("decode_cache must be split or full.")
    use_graphs = DECODE_CUDA_GRAPHS if cuda_graphs is None else bool(cuda_graphs)
    batch_size = len(prompts)
    greedy = (temperature is not None and temperature <= 0) or (temperature is None and top_k is None) or top_k == 1
    if not isinstance(speculative, bool) and speculative != "auto":
        raise ValueError("speculative must be a boolean or 'auto'.")
    if speculative is True and not greedy:
        raise ValueError("Self-speculative decoding requires greedy inference (temperature=0).")
    # "auto": speculative exactly when the decode is greedy.
    speculative = speculative is True or (speculative == "auto" and greedy)
    use_speculative = speculative and (getattr(model.config, "next_latent_prediction", False)
                                      or model.config.n_predict > 1)

    prefill = _prefill_prompts(
        model, prompts, example_ids, device, max_new_tokens=max_new_tokens, cached_positions=cached_positions,
        padded_prompt_length=padded_prompt_length, dihedral_ids=dihedral_ids,
        pair_identity_ids=pair_identity_ids, return_hidden_states=use_speculative, full_cache=False,
    )
    outputs = prefill.outputs
    full_attention_mask, current_len, max_model_len = prefill.attention_mask, prefill.current_len, prefill.max_model_len
    initial_state, finished = prefill.initial_state, prefill.finished
    example_ids_tensor, dihedral_ids_tensor = prefill.example_ids, prefill.dihedral_ids
    conditioning_embeds, value_embeds, split_kwargs = prefill.conditioning_embeds, prefill.value_embeds, prefill.split_kwargs
    attention_mask, demo_mask = prefill.prompt_attention_mask, prefill.prompt_demo_mask
    split_model = getattr(model.config, "split_target_demo_attention", False)

    if decode_cache == "split" and split_model:
        token_limit = min(max_new_tokens, max_model_len - current_len)
        latent_draft = getattr(model.config, "next_latent_prediction", False)
        width = (model.config.next_latent_horizon + 1 if latent_draft else model.config.n_predict) if use_speculative else 1
        prompt_kvs = list(outputs.pop("past_key_values"))
        head_dim = prompt_kvs[0][0].size(-1)
        layers, demo_lengths, target_lengths = split_prompt_cache(
            prompt_kvs, attention_mask, demo_mask, decode_slots=token_limit + width)
        del prompt_kvs
        kernel = choose_kv_cache_kernel(device, layers[0][0].dtype, head_dim, getattr(model.config, "FA", 2))
        cache = (layers, demo_lengths, target_lengths, kernel)
        shared = dict(example_ids=example_ids_tensor, dihedral_ids=dihedral_ids_tensor,
                      conditioning_embeds=conditioning_embeds, value_embeds=value_embeds)
        if use_speculative:
            generated = _split_speculative_decode(model, outputs, cache, initial_state, finished,
                                                  token_limit, width, shared, use_graphs)
        else:
            generated = _split_greedy_decode(model, outputs["logits"], cache, initial_state, finished,
                                             token_limit, shared, temperature, top_k, use_graphs)
        return _finish_generated_sequences(prompts, generated)

    prompt_kvs = outputs["past_key_values"]
    if use_graphs and greedy and device.type == "cuda" and split_model:
        token_limit = min(max_new_tokens, max_model_len - current_len)
        latent_draft = getattr(model.config, "next_latent_prediction", False)
        width = (model.config.next_latent_horizon + 1 if latent_draft else model.config.n_predict) if use_speculative else 1
        shared = dict(example_ids=example_ids_tensor, dihedral_ids=dihedral_ids_tensor,
                      conditioning_embeds=conditioning_embeds, value_embeds=value_embeds)
        generated = _graphed_full_decode(model, outputs, prompt_kvs, attention_mask, demo_mask, initial_state,
                                         finished, current_len, token_limit, width, shared, use_speculative)
        return _finish_generated_sequences(prompts, generated)

    past_key_values = _full_kv_buffers(prompt_kvs, max_model_len, device)
    grid_state = BatchGridState(initial_state)

    if use_speculative:
        generated = _batched_speculative_decode(
            model, outputs, past_key_values, full_attention_mask, initial_state, finished,
            current_len, min(max_new_tokens, max_model_len - current_len),
            dict(example_ids=example_ids_tensor, dihedral_ids=dihedral_ids_tensor,
                 conditioning_embeds=conditioning_embeds, value_embeds=value_embeds, **split_kwargs))
        return _finish_generated_sequences(prompts, generated)

    if not hasattr(model, "_compiled_decode"):
        print("Compiling model for decoding step...")
        model._compiled_decode = torch.compile(model.forward_generate, mode="default", fullgraph=True)

    logits = outputs["logits"]

    cache_position = torch.tensor([current_len], dtype=torch.long, device=device)
    steps_remaining = min(max_new_tokens, max_model_len - current_len)
    generated_tokens_buffer = torch.full((batch_size, steps_remaining), END_PAIR_TOKEN_ID, dtype=torch.long, device=device)

    all_done = torch.zeros((), dtype=torch.bool, device=device)
    monitor = _DoneMonitor(all_done)
    for step_i in range(steps_remaining):
        if monitor.done(step_i):
            break
        torch.compiler.cudagraph_mark_step_begin()
        next_token = _select_next_token(
            logits,
            temperature=temperature,
            top_k=top_k,
        )
        next_token = torch.where(finished, torch.tensor(END_PAIR_TOKEN_ID, device=device), next_token)
        generated_tokens_buffer[:, step_i] = next_token
        finished = finished | (next_token == END_PAIR_TOKEN_ID)
        token_positions = grid_state.update(next_token).unsqueeze(1)
        active_rows = ~finished
        full_attention_mask[:, cache_position] = active_rows.unsqueeze(1)
        decode_block_mask = None
        # The split two-lookup path masks with the cache masks and ignores this.
        if not split_model and hasattr(model, "build_decode_block_mask_for_step"):
            decode_block_mask = model.build_decode_block_mask_for_step(
                attention_mask=full_attention_mask,
                cache_position=cache_position,
                kv_len=max_model_len,
                device=device,
            )
        outputs = model._compiled_decode(
            input_ids=next_token.unsqueeze(1), example_ids=example_ids_tensor,
            dihedral_ids=dihedral_ids_tensor,
            past_key_values=tuple(past_key_values), positions_3d=token_positions,
            attention_mask=full_attention_mask, cache_position=cache_position,
            conditioning_embeds=conditioning_embeds,
            value_embeds=value_embeds,
            decode_block_mask=decode_block_mask,
            **split_kwargs,
        )
        logits = outputs["logits"]
        cache_position.add_(1)
        all_done.copy_(finished.all())
        monitor.record(step_i)

    return _finish_generated_sequences(prompts, generated_tokens_buffer)


@dataclass
class InferenceAttempt:
    prompt: List[int]
    positions: torch.Tensor
    task_id: str
    pair_index: int
    task_embedding_ids: Tuple[int, int]
    dihedral_index: int
    color_mapping: List[int]
    source_colors: List[int]
    original_input: List[List[int]]
    pair_identity_ids: Optional[List[int]] = None
    # Block-upscale spec in the original orientation; identity unless scale views are on.
    scale_factor: Tuple[int, int] = IDENTITY_SCALE


@dataclass
class InferenceBatch:
    items: List[InferenceAttempt]
    padded_prompt_length: int


@dataclass
class InferenceScheduleItem:
    """One coordinated color and geometry augmentation."""

    color_mapping: torch.Tensor
    dihedral_index: int


def _prompt_bucket_length(
    prompt_length: int,
    *,
    max_seq_len: int,
    max_new_tokens: int,
) -> int:
    """Round a prompt to a reusable shape without consuming decode capacity."""
    prompt_length = int(prompt_length)
    max_seq_len = int(max_seq_len)
    max_new_tokens = max(0, int(max_new_tokens))
    if prompt_length < 1:
        raise ValueError("Prompt length must be positive.")
    if prompt_length > max_seq_len:
        raise ValueError(
            f"Prompt length {prompt_length} exceeds model capacity {max_seq_len}."
        )

    generation_tokens = min(max_new_tokens, max_seq_len - prompt_length)
    largest_safe_prompt = max_seq_len - generation_tokens
    bucketed_length = (
        (prompt_length + PROMPT_LENGTH_BUCKET_SIZE - 1)
        // PROMPT_LENGTH_BUCKET_SIZE
        * PROMPT_LENGTH_BUCKET_SIZE
    )
    return min(bucketed_length, largest_safe_prompt)


def _build_inference_batches(
    work_items: Sequence[InferenceAttempt],
    *,
    batch_size: int,
    max_seq_len: int,
    max_new_tokens: int,
    token_budget: int = PREFILL_TOKEN_BUDGET,
) -> List[InferenceBatch]:
    """Length-sort attempts and batch within reusable padded prompt shapes."""
    batch_size = max(1, int(batch_size))
    token_budget = max(1, int(token_budget))
    ordered_items = sorted(
        work_items,
        key=lambda item: len(item.prompt),
        reverse=True,
    )
    batches: List[InferenceBatch] = []
    for padded_prompt_length, items in groupby(
        ordered_items,
        key=lambda item: _prompt_bucket_length(
            len(item.prompt),
            max_seq_len=max_seq_len,
            max_new_tokens=max_new_tokens,
        ),
    ):
        group = list(items)
        budget_batch_size = max(1, token_budget // padded_prompt_length)
        effective_batch_size = min(batch_size, budget_batch_size)
        batches.extend(
            InferenceBatch(
                items=group[start : start + effective_batch_size],
                padded_prompt_length=padded_prompt_length,
            )
            for start in range(0, len(group), effective_batch_size)
        )
    return batches


def _case_seed(seed: int, task_id: str, pair_index: int, purpose: str) -> int:
    payload = f"{seed}|{task_id}|{pair_index}|{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def _normalize_arc_colors(colors: Sequence[int]) -> List[int]:
    return sorted({int(color) for color in colors if 0 <= int(color) <= 9})


def _balanced_inference_color_block(
    colors: Sequence[int],
    permutation: Sequence[int],
    offsets: Sequence[int],
    rng: random.Random,
    *,
    identity_first: bool,
) -> List[Tuple[int, ...]]:
    """Use each available destination once per source color in a full block."""
    colors = list(colors)
    period = max(1, len(permutation))
    rows = [
        tuple(
            permutation[(position + offset) % period]
            for offset in offsets
        )
        for position in range(period)
    ]
    if identity_first:
        identity_position = rows.index(tuple(colors))
        identity = rows.pop(identity_position)
        rng.shuffle(rows)
        rows.insert(0, identity)
    else:
        rng.shuffle(rows)
    return rows


def _build_balanced_inference_color_schedule(
    colors: Sequence[int],
    count: int,
    seed: int,
    *, destination_colors: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    """Balance each source color over all destinations within one inference case."""
    colors = _normalize_arc_colors(colors)
    requested = max(1, int(count))
    rng = random.Random(int(seed))
    targets: List[Tuple[int, ...]] = []
    permutation = list(range(10)) if destination_colors is None else list(destination_colors)
    period = max(1, len(permutation))
    rng.shuffle(permutation)
    inverse_positions = {
        destination: position
        for position, destination in enumerate(permutation)
    }
    if colors:
        identity_base = inverse_positions[colors[0]]
        identity_offsets = tuple(
            (inverse_positions[color] - identity_base) % period
            for color in colors
        )
        distinct_offset_count = math.perm(period - 1, len(colors) - 1)
    else:
        identity_offsets = ()
        distinct_offset_count = 1

    block_sizes = [period] * (requested // period)
    if requested % period:
        block_sizes.append(requested % period)
    offsets_seen: Set[Tuple[int, ...]] = set()
    block_offsets: List[Tuple[int, ...]] = []
    for block_index in range(len(block_sizes)):
        cycle_position = block_index % distinct_offset_count
        if cycle_position == 0:
            offsets_seen.clear()
        if block_index == 0:
            offsets = identity_offsets
        else:
            while True:
                offsets = (
                    (0,)
                    + tuple(rng.sample(range(1, period), len(colors) - 1))
                    if colors
                    else ()
                )
                if offsets not in offsets_seen:
                    break
        offsets_seen.add(offsets)
        block_offsets.append(offsets)

    for block_index, (block_size, offsets) in enumerate(
        zip(block_sizes, block_offsets)
    ):
        block = _balanced_inference_color_block(
            colors,
            permutation,
            offsets,
            rng,
            identity_first=block_index == 0,
        )
        targets.extend(block[:block_size])

    return torch.stack([
        _color_mapping_tensor(colors, target, destination_colors) for target in targets
    ])


def _color_mapping_key(
    mapping: Sequence[int],
    source_colors: Sequence[int],
) -> Tuple[int, ...]:
    return tuple(int(mapping[color]) for color in source_colors)


def _build_color_dihedral_assignment(
    color_counts: Counter,
    dihedral_counts: Counter,
    existing_edges: Counter,
    *,
    seed: int,
) -> Dict[int, List[Tuple[int, ...]]]:
    """Find the minimum-multiplicity color/dihedral contingency table."""
    total = sum(color_counts.values())
    assignments: Dict[int, List[Tuple[int, ...]]] = {
        int(dihedral): []
        for dihedral in dihedral_counts
    }
    if total == 0:
        return assignments

    rng = random.Random(int(seed))
    color_keys = [key for key, count in color_counts.items() if count > 0]
    dihedral_keys = [
        int(dihedral)
        for dihedral, count in dihedral_counts.items()
        if count > 0
    ]
    rng.shuffle(color_keys)
    rng.shuffle(dihedral_keys)
    source_node = ("source",)
    sink_node = ("sink",)

    for edge_limit in range(1, total + 1):
        residual: Dict[Tuple[object, ...], Dict[Tuple[object, ...], int]] = {}
        original_edges: Dict[
            Tuple[Tuple[object, ...], Tuple[object, ...]],
            int,
        ] = {}

        def add_edge(
            start: Tuple[object, ...],
            end: Tuple[object, ...],
            capacity: int,
        ) -> None:
            residual.setdefault(start, {})
            residual.setdefault(end, {})
            residual[start][end] = residual[start].get(end, 0) + capacity
            residual[end].setdefault(start, 0)
            original_edges[(start, end)] = (
                original_edges.get((start, end), 0) + capacity
            )

        for color in color_keys:
            color_node = ("color", color)
            add_edge(source_node, color_node, int(color_counts[color]))
            for dihedral in dihedral_keys:
                capacity = max(
                    0,
                    edge_limit - int(existing_edges[(dihedral, color)]),
                )
                if capacity:
                    add_edge(
                        color_node,
                        ("dihedral", dihedral),
                        capacity,
                    )
        for dihedral in dihedral_keys:
            add_edge(
                ("dihedral", dihedral),
                sink_node,
                int(dihedral_counts[dihedral]),
            )

        flow = 0
        while flow < total:
            parents: Dict[
                Tuple[object, ...],
                Optional[Tuple[object, ...]],
            ] = {source_node: None}
            queue = deque([source_node])
            while queue and sink_node not in parents:
                node = queue.popleft()
                for neighbor, capacity in residual[node].items():
                    if capacity > 0 and neighbor not in parents:
                        parents[neighbor] = node
                        queue.append(neighbor)
            if sink_node not in parents:
                break
            increment = total - flow
            node = sink_node
            while parents[node] is not None:
                parent = parents[node]
                assert parent is not None
                increment = min(increment, residual[parent][node])
                node = parent
            node = sink_node
            while parents[node] is not None:
                parent = parents[node]
                assert parent is not None
                residual[parent][node] -= increment
                residual[node][parent] += increment
                node = parent
            flow += increment

        if flow != total:
            continue
        for color in color_keys:
            color_node = ("color", color)
            for dihedral in dihedral_keys:
                dihedral_node = ("dihedral", dihedral)
                capacity = original_edges.get(
                    (color_node, dihedral_node),
                    0,
                )
                used = capacity - residual[color_node].get(dihedral_node, 0)
                assignments[dihedral].extend([color] * used)
        return assignments

    raise RuntimeError("Failed to construct a color/dihedral assignment.")


def _pair_color_mappings_with_dihedrals(
    mappings: torch.Tensor,
    dihedrals: Sequence[int],
    source_colors: Sequence[int],
    *,
    anchor_index: int,
    seed: int,
) -> List[torch.Tensor]:
    """Assign a balanced mapping multiset to fixed dihedrals with few joint repeats."""
    if len(mappings) != len(dihedrals):
        raise ValueError("Color and dihedral schedules must have equal lengths.")
    count = len(dihedrals)
    if count == 0:
        return []

    colors = _normalize_arc_colors(source_colors)
    keys = [
        _color_mapping_key(mapping.tolist(), colors)
        for mapping in mappings
    ]
    mapping_by_key = {
        key: mapping
        for key, mapping in zip(keys, mappings)
    }
    remaining_keys = Counter(keys)
    remaining_dihedrals = Counter(int(dihedral) for dihedral in dihedrals)
    assigned: List[Optional[torch.Tensor]] = [None] * count
    edge_counts: Counter = Counter()

    identity_key = keys[0]
    anchor_dihedral = int(dihedrals[anchor_index])
    assigned[anchor_index] = mapping_by_key[identity_key]
    remaining_keys[identity_key] -= 1
    remaining_dihedrals[anchor_dihedral] -= 1
    edge_counts[(anchor_dihedral, identity_key)] += 1

    rng = random.Random(int(seed))
    remaining_keys = Counter(
        {
            key: copies
            for key, copies in remaining_keys.items()
            if copies > 0
        }
    )
    remaining_dihedrals = Counter(
        {
            dihedral: slots
            for dihedral, slots in remaining_dihedrals.items()
            if slots > 0
        }
    )
    assigned_by_dihedral = _build_color_dihedral_assignment(
        remaining_keys,
        remaining_dihedrals,
        edge_counts,
        seed=seed,
    )

    for dihedral, mapping_keys in assigned_by_dihedral.items():
        rng.shuffle(mapping_keys)
        positions = [
            index
            for index, value in enumerate(dihedrals)
            if int(value) == dihedral and assigned[index] is None
        ]
        if len(positions) != len(mapping_keys):
            raise RuntimeError("Color/dihedral assignment produced inconsistent slots.")
        for position, key in zip(positions, mapping_keys):
            assigned[position] = mapping_by_key[key]

    if any(mapping is None for mapping in assigned):
        raise RuntimeError("Color/dihedral assignment left an unfilled attempt.")
    return [mapping for mapping in assigned if mapping is not None]


def _build_inference_case_schedule(
    *,
    source_colors: Sequence[int],
    attempt_count: int,
    use_aug: bool,
    seed: int,
    task_id: str,
    pair_index: int,
    augmentation_sampling_mode: Optional[str] = None,
    protected_colors: Sequence[int] = (),
    input_grid: Optional[Sequence[Sequence[int]]] = None,
    original_inputs: Sequence[Sequence[Sequence[int]]] = (),
) -> List[InferenceScheduleItem]:
    """Coordinate color and dihedral augmentations for one input grid."""
    if use_aug and input_grid is not None and normalize_augmentation_sampling_mode(augmentation_sampling_mode) == "single_grid_prefix":
        key = (task_id, "inference", pair_index)
        augments = build_unique_input_augments(
            {key: input_grid}, source_colors, protected_colors=protected_colors,
            original_inputs=original_inputs, max_augments=attempt_count, seed=seed,
        )[key]
        return [InferenceScheduleItem(
            color_mapping=augments.color_maps[color_index], dihedral_index=dihedral,
        ) for color_index, dihedral in zip(augments.color_map_indices, augments.dihedral_indices)]
    if use_aug:
        _, dihedrals = build_augmentation_schedule(
            source_colors,
            attempt_count,
            _case_seed(seed, task_id, pair_index, "augmentation"),
        )
    else:
        dihedrals = [0]
    schedule_count = len(dihedrals)
    anchor_index = next(
        (
            index
            for index, dihedral in enumerate(dihedrals)
            if int(dihedral) == 0
        ),
        0,
    )

    if use_aug:
        movable_colors, destinations = color_augmentation_domain(
            source_colors, protected_colors, augmentation_sampling_mode,
        )
        mappings = _build_balanced_inference_color_schedule(
            movable_colors,
            schedule_count,
            _case_seed(seed, task_id, pair_index, "color"),
            destination_colors=destinations,
        )
    else:
        mappings = torch.arange(VOCAB_SIZE, dtype=torch.long).unsqueeze(0)
    paired_mappings = _pair_color_mappings_with_dihedrals(
        mappings,
        dihedrals,
        source_colors,
        anchor_index=anchor_index,
        seed=_case_seed(seed, task_id, pair_index, "color-pairing"),
    )

    return [
        InferenceScheduleItem(
            color_mapping=mapping,
            dihedral_index=int(dihedral),
        )
        for mapping, dihedral in zip(paired_mappings, dihedrals)
    ]


def _build_inference_attempts(
    dataset,
    split: str,
    task_ids: Optional[Sequence[str]],
    pair_index: Optional[int],
    attempt_count: int,
    use_aug: bool,
) -> List[InferenceAttempt]:
    attempts: List[InferenceAttempt] = []
    allowed_tasks = None if task_ids is None else set(task_ids)
    mode = normalize_augmentation_sampling_mode(getattr(dataset, "augmentation_sampling_mode", None))
    examples = [example for example in dataset.iter_examples(split=split)
                if allowed_tasks is None or example.task_id in allowed_tasks]
    unique_schedules = {}
    if use_aug and mode == "single_grid_prefix":
        for task_id, task_examples in groupby(sorted(examples, key=lambda item: item.task_id), key=lambda item: item.task_id):
            train_pairs = dataset.pairs_by_task[task_id]
            test_pairs = dataset.test_pairs_by_task.get(task_id, [])
            target_pairs = test_pairs if split == "test" else train_pairs
            unique_schedules.update(build_unique_input_augments(
                {(task_id, split, example.pair_index): target_pairs[example.pair_index]["input"]
                 for example in task_examples},
                dataset.task_colors[task_id],
                protected_colors=getattr(dataset, "task_output_only_colors", {}).get(task_id, ()),
                original_inputs=[pair["input"] for pair in train_pairs + test_pairs],
                max_augments=attempt_count, seed=dataset.seed,
            ))
    for example in examples:
        if pair_index is not None and example.pair_index != pair_index:
            continue
        train_pairs = dataset.pairs_by_task[example.task_id]
        target_pairs = (
            dataset.test_pairs_by_task[example.task_id]
            if split == "test"
            else train_pairs
        )
        target_input = target_pairs[example.pair_index]["input"]
        source_colors = dataset.task_colors[example.task_id]
        augments = unique_schedules.get((example.task_id, split, example.pair_index))
        if use_aug and mode == "context_palette":
            from context_palette import color_schedule
            # Retain the existing inference geometry schedule and exact attempt
            # budget. Unlike target-only uniqueness, symmetries never drop visits.
            _, dihedrals = build_augmentation_schedule(
                source_colors, attempt_count,
                _case_seed(dataset.seed, example.task_id, example.pair_index, "augmentation"))
            mappings = color_schedule(dataset, example, dihedrals,
                                      seed=dataset.seed, max_augments=attempt_count)
            schedule = [InferenceScheduleItem(color_mapping=mapping, dihedral_index=dihedral)
                        for mapping, dihedral in zip(mappings, dihedrals)]
        elif augments is not None:
            schedule = [InferenceScheduleItem(
                color_mapping=augments.color_maps[color_index], dihedral_index=dihedral,
            ) for color_index, dihedral in zip(augments.color_map_indices, augments.dihedral_indices)]
        else:
            schedule = _build_inference_case_schedule(
                source_colors=source_colors,
                attempt_count=attempt_count,
                use_aug=use_aug,
                seed=dataset.seed,
                task_id=example.task_id,
                pair_index=example.pair_index,
                augmentation_sampling_mode=mode,
                protected_colors=getattr(dataset, "task_output_only_colors", {}).get(example.task_id, ()),
            )
        scale_views = [IDENTITY_SCALE] * len(schedule)
        if use_aug and getattr(dataset, "inference_scale_views", False):
            # Specs every demonstration admits and that fit the projected test output
            # (transductARC's demo bound), spread evenly over the existing views.
            specs = demo_bounded_scale_specs(
                target_input, [pair for pair in train_pairs if pair.get("output") is not None],
                anisotropic=dataset.scale_anisotropic, max_specs=dataset.scale_max_specs)
            scale_views = inference_scale_schedule(specs, len(schedule), _case_seed(
                dataset.seed, example.task_id, example.pair_index, "scale-views"))
        for visit, schedule_item in enumerate(schedule):
            scale = scale_views[visit]
            prompt_tensor, _ = dataset.build_context_sequence(
                example,
                {"input": target_input},
                schedule_item.dihedral_index,
                schedule_item.color_mapping,
                visit=visit,
                scale=None if is_identity_scale(scale) else scale,
            )
            prompt = prompt_tensor.tolist()
            prompt_tensor = prompt_tensor.unsqueeze(0)
            positions = compute_positions_3d(
                prompt_tensor, torch.ones_like(prompt_tensor, dtype=torch.bool)
            ).squeeze(0)
            attempts.append(
                InferenceAttempt(
                    prompt=prompt,
                    positions=positions,
                    task_id=example.task_id,
                    pair_index=example.pair_index,
                    task_embedding_ids=tuple(example.task_embedding_ids),
                    dihedral_index=schedule_item.dihedral_index,
                    color_mapping=schedule_item.color_mapping.tolist(),
                    # The restricted policy completes a full permutation, so
                    # inverse every label, including unexpected predictions.
                    source_colors=list(range(10)) if mode in ("single_grid_prefix", "context_palette") else list(source_colors),
                    original_input=[list(row) for row in target_input],
                    pair_identity_ids=dataset.pair_identity_ids(example, positions, visit=visit).tolist()
                        if getattr(dataset, "pair_identity_vertices", 0) else None,
                    scale_factor=tuple(scale),
                )
            )
    return attempts


def _prepare_model_for_inference(model: TinyTransformer) -> TinyTransformer:
    model.eval()
    if next(model.parameters()).dtype != torch.bfloat16:
        model.to(dtype=torch.bfloat16)
    return model


def _generate_with_oom_splitting(items, generate):
    """Fit a planned batch without dropping attempts or changing their order."""
    pending = deque([list(items)]) if items else deque()
    outputs = []
    while pending:
        current = pending.popleft()
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        python_rng = random.getstate()
        try:
            generated = generate(current)
        except torch.cuda.OutOfMemoryError:
            if len(current) == 1:
                raise
        else:
            outputs.extend(generated)
            continue
        # Exit the exception handler before collecting, so its traceback no
        # longer retains the failed generation's CUDA tensors and KV caches.
        import gc
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        random.setstate(python_rng)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        middle = len(current) // 2
        pending.appendleft(current[middle:])
        pending.appendleft(current[:middle])
        print(f"CUDA OOM: splitting physical inference batch {len(current)} into "
              f"{middle}+{len(current) - middle}; candidate attempts are unchanged.", flush=True)
    return outputs


@torch.inference_mode()
def run_split_inference(
    model: TinyTransformer, dataset, split: str, device: torch.device,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    task_ids: Optional[Sequence[str]] = None, pair_index: Optional[int] = None,
    temperature: Optional[float] = None, top_k: Optional[int] = None,
    augmentation_attempts: int = 0,
    speculative: bool = False,
    token_budget: Optional[int] = None,
) -> List[Dict[str, object]]:
    use_aug = int(augmentation_attempts) > 0
    token_budget = PREFILL_TOKEN_BUDGET if token_budget is None else int(token_budget)
    attempt_count = int(augmentation_attempts) if use_aug else 1
    work_items = _build_inference_attempts(
        dataset, split, task_ids, pair_index, attempt_count, use_aug
    )
    model = _prepare_model_for_inference(model)
    results: List[Dict[str, object]] = []
    batches = _build_inference_batches(
        work_items,
        batch_size=batch_size,
        max_seq_len=model.config.max_seq_len,
        max_new_tokens=max_new_tokens,
        token_budget=token_budget,
    )

    for batch_index, inference_batch in enumerate(batches):
        batch = inference_batch.items

        def generate(current):
            return batched_greedy_generate(
                model=model,
                prompts=[item.prompt for item in current],
                example_ids=[item.task_embedding_ids for item in current],
                dihedral_ids=[item.dihedral_index for item in current],
                pair_identity_ids=[item.pair_identity_ids for item in current],
                device=device,
                max_new_tokens=max_new_tokens,
                cached_positions=[item.positions for item in current],
                temperature=temperature,
                top_k=top_k,
                padded_prompt_length=inference_batch.padded_prompt_length,
                speculative=speculative,
            )
        sequences = _generate_with_oom_splitting(batch, generate)
        for sequence, item in zip(sequences, batch):
            generated = list(sequence[len(item.prompt) :])
            if END_PAIR_TOKEN_ID in generated:
                generated = generated[: generated.index(END_PAIR_TOKEN_ID)]
            results.append(
                {
                    "task_id": item.task_id,
                    "pair_index": item.pair_index,
                    "prompt_tokens": item.prompt,
                    "sequence": list(sequence),
                    "output_tokens": generated,
                    "output_grid": tokens_to_grid(generated),
                    "dihedral_index": item.dihedral_index,
                    "color_mapping": item.color_mapping,
                    "source_colors": item.source_colors,
                    "original_input": item.original_input,
                    **({} if is_identity_scale(item.scale_factor) else {"scale_factor": list(item.scale_factor)}),
                }
            )
        print(
            f"[{split}] Finished batch {batch_index + 1} / {len(batches)} "
            f"(size={len(batch)}, prompt_bucket={inference_batch.padded_prompt_length})"
        )

    if torch.device(device).type == "cuda":
        print(f"[{split}] Decode peak allocated memory: {torch.cuda.max_memory_allocated(device) / 2**30:.1f} GiB "
              f"({len(batches)} batches, batch cap {batch_size}, prefill token budget {token_budget})",
              flush=True)
    return results


# =============================================================================
# Teacher-forced Accuracy
# =============================================================================

@dataclass
class AccuracyAttempt:
    inference: InferenceAttempt
    tokens: torch.Tensor
    positions: torch.Tensor


def _build_accuracy_attempts(
    dataset,
    split: str,
    work_items: Sequence[InferenceAttempt],
    solutions: Dict[str, object],
    max_seq_len: int,
) -> List[AccuracyAttempt]:
    """Attach transformed targets without changing the inference prompt schedule."""
    attempts = []
    for item in work_items:
        pairs = (
            dataset.test_pairs_by_task[item.task_id]
            if split == "test"
            else dataset.pairs_by_task[item.task_id]
        )
        output = pairs[item.pair_index].get("output")
        if split == "test" and item.task_id in solutions:
            task_solutions = solutions[item.task_id]
            if not isinstance(task_solutions, list) or item.pair_index >= len(task_solutions):
                raise ValueError(f"Missing solution for {item.task_id}:{item.pair_index}.")
            solution = task_solutions[item.pair_index]
            if output is not None and output != solution:
                raise ValueError(f"Conflicting solutions for {item.task_id}:{item.pair_index}.")
            output = solution
        if output is None:
            raise ValueError(
                f"accuracy-only requires a correct output for {split} "
                f"{item.task_id}:{item.pair_index}; provide solutions.json or inline outputs."
            )
        if not is_rectangular_grid(output):
            raise ValueError(f"Invalid output grid for {split} {item.task_id}:{item.pair_index}.")
        scaled_output = output if is_identity_scale(item.scale_factor) else upscale_grid(output, item.scale_factor)
        output_tokens = grid_to_tokens(
            apply_dihedral_transform(scaled_output, item.dihedral_index), END_PAIR_TOKEN_ID
        )
        tokens = torch.tensor(
            item.prompt + [item.color_mapping[token] for token in output_tokens],
            dtype=torch.long,
        )
        if tokens.numel() > max_seq_len:
            raise ValueError(
                f"accuracy-only sequence for {split} {item.task_id}:{item.pair_index} "
                f"has {tokens.numel()} tokens, exceeding model capacity {max_seq_len}."
            )
        positions = compute_positions_3d(tokens.unsqueeze(0)).squeeze(0)
        attempts.append(AccuracyAttempt(item, tokens, positions))
    return attempts


def _build_accuracy_batches(
    attempts: Sequence[AccuracyAttempt],
    batch_size: int,
    token_budget: int = PREFILL_TOKEN_BUDGET,
) -> List[List[AccuracyAttempt]]:
    """Bound full-sequence work, including the teacher-forced output, per batch."""
    batch_size = max(1, int(batch_size))
    token_budget = max(1, int(token_budget))
    ordered = sorted(attempts, key=lambda item: item.tokens.numel(), reverse=True)
    batches = []
    start = 0
    while start < len(ordered):
        longest = ordered[start].tokens.numel()
        count = min(batch_size, max(1, token_budget // longest))
        batches.append(ordered[start : start + count])
        start += count
    return batches


@torch.inference_mode()
def run_split_accuracy(
    model: TinyTransformer, dataset, split: str, device: torch.device,
    solutions: Dict[str, object], batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    task_ids: Optional[Sequence[str]] = None, pair_index: Optional[int] = None,
    augmentation_attempts: int = 0,
    token_budget: Optional[int] = None,
) -> Dict[str, object]:
    """Measure exact-output argmax accuracy with one forward per packed batch."""
    use_aug = int(augmentation_attempts) > 0
    work_items = _build_inference_attempts(
        dataset, split, task_ids, pair_index,
        int(augmentation_attempts) if use_aug else 1, use_aug,
    )
    if not work_items:
        raise ValueError(f"No examples selected for accuracy-only evaluation of {split}.")
    attempts = _build_accuracy_attempts(
        dataset, split, work_items, solutions, model.config.max_seq_len
    )
    batches = _build_accuracy_batches(
        attempts, batch_size, token_budget=PREFILL_TOKEN_BUDGET if token_budget is None else int(token_budget))
    model = _prepare_model_for_inference(model)
    metric_counts = torch.zeros(6, dtype=torch.long, device=device)
    for batch_index, batch in enumerate(batches):
        lengths = torch.tensor([item.tokens.numel() for item in batch], dtype=torch.int32)
        cu_seqlens = torch.zeros(len(batch) + 1, dtype=torch.int32)
        cu_seqlens[1:] = lengths.cumsum(0)
        # Keep the bidirectional boundary at the final input/output separator.
        # Output tokens attend causally; reference answers cannot leak into the prefix.
        identity_kwargs = {}
        if getattr(model.config, "pair_identity_vertices", 0):
            identity_kwargs["pair_identity_ids"] = torch.cat([
                F.pad(torch.tensor(item.inference.pair_identity_ids, dtype=torch.long),
                    (0, item.tokens.numel() - len(item.inference.prompt))) for item in batch
            ]).to(device)
        outputs = model(
            input_ids=torch.cat([item.tokens for item in batch]).to(device),
            example_ids=torch.tensor(
                [item.inference.task_embedding_ids for item in batch],
                dtype=torch.int32, device=device,
            ),
            sep_indices=torch.tensor(
                [len(item.inference.prompt) - 1 for item in batch],
                dtype=torch.int32, device=device,
            ),
            dihedral_ids=torch.tensor(
                [item.inference.dihedral_index for item in batch],
                dtype=torch.int32, device=device,
            ),
            positions_3d=torch.cat([item.positions for item in batch]).to(device),
            cu_seqlens=cu_seqlens.to(device),
            max_seqlen=int(lengths.max()),
            compute_training_metrics=True,
            compute_mtp_loss=False,
            compute_next_latent_loss=False,
            **identity_kwargs,
        )
        metric_counts.add_(outputs["training_metric_counts"])
        del outputs
        print(f"[{split}] Accuracy batch {batch_index + 1} / {len(batches)} (size={len(batch)})")

    counts = metric_counts.cpu().tolist()
    summary = {
        "total_sequences": len(attempts),
        "total_cases": len({(item.task_id, item.pair_index) for item in work_items}),
    }
    for name, correct, total in zip(
        ("color", "special_token", "exact_output"), counts[::2], counts[1::2]
    ):
        summary[f"{name}_correct"] = correct
        summary[f"{name}_total"] = total
        summary[f"{name}_accuracy"] = correct / total if total else None
    return summary


# =============================================================================
# Candidate re-scoring
# =============================================================================

# Top-2 selection rules over a decode's candidates: the vote count alone (as AAIVR
# ranks them), the summed teacher-forced log-likelihood over every augmentation,
# and votes with likelihood breaking ties.
RESCORE_RULES = ("votes", "likelihood", "votes_then_likelihood")
DEFAULT_RESCORE_TOP_K = 20


@dataclass
class RescoreAttempt:
    inference: InferenceAttempt
    candidate_index: int
    tokens: torch.Tensor
    positions: torch.Tensor


def _load_rescore_candidates(path, top_k: int) -> Dict[Tuple[str, int], List[dict]]:
    """Each test output's most-voted candidates from a decode's candidates.json.gz.

    Input copies are dropped, as the voting filter drops them; ties keep dump order."""
    with gzip.open(path, "rt") as handle:
        data = json.load(handle)
    candidates = {}
    for pair in data["pairs"]:
        kept = [entry for entry in pair["candidates"]
                if not entry.get("input_copy") and is_rectangular_grid(entry["grid"])]
        kept.sort(key=lambda entry: -int(entry["count"]))
        candidates[(str(pair["task_id"]), int(pair["pair_index"]))] = kept[:max(1, int(top_k))]
    return candidates


def _candidate_output_tokens(item: InferenceAttempt, grid) -> List[int]:
    """A canonical candidate grid as that attempt's output tokens (ending in the end token).

    A scale view's candidate is block-upscaled before the dihedral, as its prompt grids are."""
    if not is_identity_scale(item.scale_factor):
        grid = upscale_grid(grid, item.scale_factor)
    output_tokens = grid_to_tokens(apply_dihedral_transform(grid, item.dihedral_index), END_PAIR_TOKEN_ID)
    return [item.color_mapping[token] for token in output_tokens]


@torch.inference_mode()
def _score_continuations_cached(
    model, items: Sequence[InferenceAttempt], continuations: Sequence[Sequence[Sequence[int]]],
    device: torch.device, padded_prompt_length: Optional[int] = None,
) -> List[List[Optional[float]]]:
    """Summed log-likelihood of every continuation after its row's prompt, prefilling each prompt once.

    Each prompt goes through the decode's own prefill into a KV cache; every candidate is then
    teacher-forced as one block over that cache (as speculative verification does), so the shared
    prompt is computed once per view instead of once per candidate. None = does not fit."""
    max_block = max(len(cont) - 1 for conts in continuations for cont in conts)
    prefill = _prefill_prompts(
        model, [item.prompt for item in items], [item.task_embedding_ids for item in items], device,
        max_new_tokens=max(1, max_block), cached_positions=[item.positions for item in items],
        padded_prompt_length=padded_prompt_length, dihedral_ids=[item.dihedral_index for item in items],
        pair_identity_ids=[item.pair_identity_ids for item in items],
    )
    start, capacity = prefill.current_len, prefill.max_model_len
    first = torch.log_softmax(prefill.outputs["logits"][:, -1].float(), dim=-1)
    shared = dict(example_ids=prefill.example_ids, dihedral_ids=prefill.dihedral_ids,
                  conditioning_embeds=prefill.conditioning_embeds, value_embeds=prefill.value_embeds,
                  **prefill.split_kwargs)
    batch_size = len(items)
    scores: List[List[Optional[float]]] = [[None] * len(conts) for conts in continuations]
    for index in range(max(len(conts) for conts in continuations)):
        rows = [row for row in range(batch_size) if index < len(continuations[row])
                and start + len(continuations[row][index]) - 1 <= capacity
                and len(items[row].prompt) + len(continuations[row][index]) <= model.config.max_seq_len]
        if not rows:
            continue
        width = max(1, max(len(continuations[row][index]) - 1 for row in rows))
        tokens = torch.full((batch_size, width), END_PAIR_TOKEN_ID, dtype=torch.long)
        targets = torch.zeros((batch_size, width + 1), dtype=torch.long)
        lengths = torch.zeros(batch_size, dtype=torch.long)
        positions = torch.zeros((batch_size, width, 3), dtype=torch.int32)
        for row in rows:
            cont, prompt = continuations[row][index], items[row].prompt
            count = len(cont)
            tokens[row, :count - 1] = torch.tensor(cont[:-1], dtype=torch.long)
            targets[row, :count] = torch.tensor(cont, dtype=torch.long)
            lengths[row] = count
            # The same 3D positions the full prompt + candidate sequence gets.
            full_positions = compute_positions_3d(torch.tensor(prompt + cont, dtype=torch.long)[None]).squeeze(0)
            positions[row, :count - 1] = full_positions[len(prompt):len(prompt) + count - 1]
        block_valid = (torch.arange(width)[None, :] < (lengths - 1).clamp(min=0)[:, None]).to(device)
        attention_mask = prefill.attention_mask.clone()
        attention_mask[:, start:] = False
        attention_mask[:, start:start + width] = block_valid
        slots = (start + torch.arange(width, device=device))[None, :].expand(batch_size, -1).contiguous()
        block_mask = (model.build_decode_block_mask_for_step(attention_mask, slots, capacity, device)
                      if device.type == "cuda" else None)
        outputs = model.forward_generate(
            input_ids=tokens.to(device), positions_3d=positions.to(device),
            past_key_values=tuple(prefill.past_key_values), attention_mask=attention_mask,
            cache_position=slots, decode_block_mask=block_mask, **shared,
        )
        targets = targets.to(device)
        block = torch.log_softmax(outputs["logits"].float(), dim=-1).gather(
            2, targets[:, 1:width + 1, None]).squeeze(2)
        sums = first.gather(1, targets[:, :1]).squeeze(1) + (block * block_valid).sum(dim=1)
        values = sums.tolist()
        for row in rows:
            scores[row][index] = values[row]
        del outputs
    return scores


def _build_rescore_attempts(
    work_items: Sequence[InferenceAttempt],
    candidates: Dict[Tuple[str, int], List[dict]],
    max_seq_len: int,
) -> Tuple[List[RescoreAttempt], int]:
    """Append each candidate, transformed like that attempt's prompt, to the prompt."""
    attempts, too_long = [], 0
    for item in work_items:
        for index, entry in enumerate(candidates.get((str(item.task_id), int(item.pair_index)), ())):
            tokens = torch.tensor(item.prompt + _candidate_output_tokens(item, entry["grid"]), dtype=torch.long)
            if tokens.numel() > max_seq_len:
                too_long += 1
                continue
            positions = compute_positions_3d(tokens.unsqueeze(0)).squeeze(0)
            attempts.append(RescoreAttempt(item, index, tokens, positions))
    return attempts, too_long


def _rank_candidates(entries: Sequence[dict], rule: str) -> List[dict]:
    if rule == "votes":
        key = lambda entry: -entry["count"]
    elif rule == "likelihood":
        key = lambda entry: -entry["log_likelihood"]
    elif rule == "votes_then_likelihood":
        key = lambda entry: (-entry["count"], -entry["log_likelihood"])
    else:
        raise ValueError(f"Unknown rescore rule: {rule}")
    return sorted(entries, key=key)


@torch.inference_mode()
def run_split_rescore(
    model: TinyTransformer, dataset, split: str, device: torch.device,
    candidates: Dict[Tuple[str, int], List[dict]], batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    augmentation_attempts: int = 0, prefix_cache: bool = True,
    task_ids: Optional[Sequence[str]] = None,
) -> Dict[Tuple[str, int], List[dict]]:
    """Sum each candidate's teacher-forced output log-likelihood over the inference schedule.

    Every augmentation scores every candidate with the same prompt the decode used.
    A candidate missing from some attempt (too long) gets -inf for that test output.
    ``prefix_cache`` prefills each view's prompt once and scores all its candidates against the
    cache; otherwise each prompt + candidate is a full packed sequence (the reference path).
    """
    use_aug = int(augmentation_attempts) > 0
    work_items = _build_inference_attempts(
        dataset, split, task_ids, None, int(augmentation_attempts) if use_aug else 1, use_aug,
    )
    model = _prepare_model_for_inference(model)
    if prefix_cache:
        totals, counts, sequences, too_long = _rescore_totals_cached(
            model, work_items, candidates, device, batch_size, split)
    else:
        totals, counts, sequences, too_long = _rescore_totals_packed(
            model, work_items, candidates, device, batch_size, split)
    expected = {}
    for item in work_items:
        pair_key = (str(item.task_id), int(item.pair_index))
        expected[pair_key] = expected.get(pair_key, 0) + 1
    scored = {}
    for pair_key, entries in candidates.items():
        rows = []
        for index, entry in enumerate(entries):
            key = (*pair_key, index)
            complete = counts.get(key, 0) == expected.get(pair_key, 0) and pair_key in expected
            rows.append({"grid": entry["grid"], "count": int(entry["count"]),
                         "log_likelihood": totals[key] if complete else float("-inf"),
                         "scored_attempts": counts.get(key, 0)})
        scored[pair_key] = rows
    print(f"Rescored {sequences} sequences ({too_long} candidate attempts exceeded the model length; "
          f"{'prefix-cached' if prefix_cache else 'full sequences'}).")
    return scored


def _rescore_totals_cached(model, work_items, candidates, device, batch_size, split):
    totals: Dict[Tuple[str, int, int], float] = {}
    counts: Dict[Tuple[str, int, int], int] = {}
    sequences = too_long = 0
    batches = _build_inference_batches(work_items, batch_size=batch_size, max_seq_len=model.config.max_seq_len,
                                       max_new_tokens=DEFAULT_MAX_NEW_TOKENS)
    for batch_index, inference_batch in enumerate(batches):
        items = [item for item in inference_batch.items
                 if candidates.get((str(item.task_id), int(item.pair_index)))]
        if items:
            continuations = [[_candidate_output_tokens(item, entry["grid"])
                              for entry in candidates[(str(item.task_id), int(item.pair_index))]] for item in items]
            scores = _score_continuations_cached(model, items, continuations, device,
                                                 padded_prompt_length=inference_batch.padded_prompt_length)
            for item, row in zip(items, scores):
                for index, value in enumerate(row):
                    if value is None:
                        too_long += 1
                        continue
                    key = (str(item.task_id), int(item.pair_index), index)
                    totals[key] = totals.get(key, 0.0) + value
                    counts[key] = counts.get(key, 0) + 1
                    sequences += 1
        print(f"[{split}] Rescore batch {batch_index + 1} / {len(batches)} "
              f"(prompts={len(items)}, prompt_bucket={inference_batch.padded_prompt_length})")
    return totals, counts, sequences, too_long


def _rescore_totals_packed(model, work_items, candidates, device, batch_size, split):
    attempts, too_long = _build_rescore_attempts(work_items, candidates, model.config.max_seq_len)
    batches = _build_accuracy_batches(attempts, batch_size)
    totals: Dict[Tuple[str, int, int], float] = {}
    counts: Dict[Tuple[str, int, int], int] = {}
    for batch_index, batch in enumerate(batches):
        lengths = torch.tensor([item.tokens.numel() for item in batch], dtype=torch.int32)
        cu_seqlens = torch.zeros(len(batch) + 1, dtype=torch.int32)
        cu_seqlens[1:] = lengths.cumsum(0)
        input_ids = torch.cat([item.tokens for item in batch]).to(device)
        sep_indices = torch.tensor(
            [len(item.inference.prompt) - 1 for item in batch], dtype=torch.int32, device=device,
        )
        identity_kwargs = {}
        if getattr(model.config, "pair_identity_vertices", 0):
            identity_kwargs["pair_identity_ids"] = torch.cat([
                F.pad(torch.tensor(item.inference.pair_identity_ids, dtype=torch.long),
                      (0, item.tokens.numel() - len(item.inference.prompt))) for item in batch
            ]).to(device)
        outputs = model(
            input_ids=input_ids,
            example_ids=torch.tensor(
                [item.inference.task_embedding_ids for item in batch], dtype=torch.int32, device=device,
            ),
            sep_indices=sep_indices,
            dihedral_ids=torch.tensor(
                [item.inference.dihedral_index for item in batch], dtype=torch.int32, device=device,
            ),
            positions_3d=torch.cat([item.positions for item in batch]).to(device),
            cu_seqlens=cu_seqlens.to(device),
            max_seqlen=int(lengths.max()),
            compute_training_metrics=False,
            compute_mtp_loss=False,
            compute_next_latent_loss=False,
            **identity_kwargs,
        )
        targets, valid = build_output_targets_and_mask(
            input_ids, sep_indices, cu_seqlens=cu_seqlens.to(device),
        )
        token_log_probs = torch.log_softmax(outputs["logits"].float(), dim=-1).gather(
            1, targets.clamp_min(0).unsqueeze(1)).squeeze(1) * valid
        sequence_ids = torch.arange(len(batch), device=device).repeat_interleave(lengths.to(device))
        sums = torch.zeros(len(batch), device=device).index_add_(0, sequence_ids, token_log_probs)
        for item, value in zip(batch, sums.tolist()):
            key = (str(item.inference.task_id), int(item.inference.pair_index), item.candidate_index)
            totals[key] = totals.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1
        del outputs
        print(f"[{split}] Rescore batch {batch_index + 1} / {len(batches)} (size={len(batch)})")
    return totals, counts, len(attempts), too_long


def rescore_selections(
    model, dataset, selections: Sequence["AAIVRSelection"], device: torch.device, *,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE, augmentation_attempts: int = 0,
    top_k: int = DEFAULT_RESCORE_TOP_K, task_ids: Optional[Sequence[str]] = None, split: str = "test",
) -> List["AAIVRSelection"]:
    """Re-rank each output's ``top_k`` voted candidates by summed log-likelihood over the decode's views.

    The rescored candidates come first (highest likelihood first, vote order on ties), then the
    rest in vote order; the submitted top-2 are the two most likely."""
    candidates = {(str(s.task_id), int(s.original_pair_index)): list(s.ranked_candidates[:max(1, int(top_k))])
                  for s in selections if s.ranked_candidates}
    scored = run_split_rescore(model, dataset, split, device, candidates, batch_size=batch_size,
                               augmentation_attempts=augmentation_attempts, prefix_cache=True, task_ids=task_ids)
    rescored = []
    for selection in selections:
        rows = scored.get((str(selection.task_id), int(selection.original_pair_index)))
        if not rows:
            rescored.append(selection)
            continue
        order = sorted(range(len(rows)), key=lambda j: -rows[j]["log_likelihood"])
        ranked = [selection.ranked_candidates[j] for j in order] + list(selection.ranked_candidates[len(rows):])
        rescored.append(replace(selection, ranked_candidates=ranked,
                                selected_outputs=[entry["grid"] for entry in ranked[:2]],
                                log_likelihoods=[rows[j]["log_likelihood"] for j in order]))
    return rescored


def write_rescore_report(directory, scored, solutions_path, *, metadata: dict) -> Path:
    """One submission per selection rule, each scored like a decode, plus per-pair details."""
    directory = Path(directory)
    task_pairs: Dict[str, int] = {}
    for task_id, pair_index in scored:
        task_pairs[task_id] = max(task_pairs.get(task_id, 0), pair_index + 1)
    solutions = {}
    if solutions_path is not None and Path(solutions_path).is_file():
        with Path(solutions_path).open() as handle:
            solutions = json.load(handle)
    rules = {}
    for rule in RESCORE_RULES:
        submission = {}
        for task_id, num_pairs in task_pairs.items():
            attempts = []
            for pair_index in range(num_pairs):
                ranked = _rank_candidates(scored.get((task_id, pair_index), []), rule)
                grids = [entry["grid"] for entry in ranked[:2]] + [[[0]]] * 2
                attempts.append({"attempt_1": grids[0], "attempt_2": grids[1]})
            submission[task_id] = attempts
        path = directory / f"rescore_{rule}_submission.json"
        path.write_text(json.dumps(submission))
        rules[rule] = {"submission": path.name}
        if solutions:
            from utils import score_arc_submission
            result = score_arc_submission(Path(solutions_path), path)
            rules[rule].update(score=result["score"], percentage=result["percentage"])
    pairs = []
    oracle_hits = 0
    for (task_id, pair_index), entries in sorted(scored.items()):
        truth = solutions.get(task_id, [None] * (pair_index + 1))
        truth = truth[pair_index] if pair_index < len(truth) else None
        hit = any(entry["grid"] == truth for entry in entries)
        oracle_hits += hit
        pairs.append({"task_id": task_id, "pair_index": pair_index, "correct_in_candidates": hit,
                      "candidates": [{"count": entry["count"], "log_likelihood": entry["log_likelihood"],
                                      "scored_attempts": entry["scored_attempts"],
                                      "correct": entry["grid"] == truth} for entry in entries]})
    report = {**metadata, "rules": rules, "num_test_outputs": len(pairs),
              "test_outputs_with_correct_candidate": oracle_hits if solutions else None, "pairs": pairs}
    path = directory / "rescore.json"
    path.write_text(json.dumps(report, indent=1, allow_nan=True))
    for rule, values in rules.items():
        if "percentage" in values:
            print(f"Rescore {rule}: {values['score']:g} ({values['percentage']:.2f}%)", flush=True)
    return path


# =============================================================================
# AAIVR - Automated Augmentation Inverse Voting and Ranking
# =============================================================================

def _grid_to_tuple(grid: Sequence[Sequence[int]]) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(int(val) for val in row) for row in grid)


def _tuple_to_grid(grid_tuple: Tuple[Tuple[int, ...], ...]) -> List[List[int]]:
    return [list(row) for row in grid_tuple]


@dataclass
class AAIVRSelection:
    task_id: str
    original_pair_index: int
    selected_outputs: List[List[List[int]]]
    ranked_candidates: List[Dict[str, object]]
    num_generated: int
    num_valid: int
    discarded_non_rectangular: int
    discarded_input_copies: int
    # Scale views whose output is not an exact block upscale (wrong size or mixed blocks).
    discarded_scale_mismatch: int = 0
    # Votes on candidates a demo-derived symbolic guard rejected, by rule.
    guard_rejections: Dict[str, int] = field(default_factory=dict)
    # Summed teacher-forced log-likelihood per rescored candidate grid (likelihood rescoring).
    log_likelihoods: Optional[List[float]] = None


def _candidate_annotations(raw_selections: Sequence[AAIVRSelection], final_selections: Sequence[AAIVRSelection],
                           dataset, guards_on: bool) -> Dict[Tuple[str, int], Dict[tuple, Dict[str, object]]]:
    """Per candidate grid: the symbolic guard that rejects it and its rescored log-likelihood."""
    annotations: Dict[Tuple[str, int], Dict[tuple, Dict[str, object]]] = {}
    guards_by_task: Dict[str, Dict[str, object]] = {}
    for selection in raw_selections:
        key = (str(selection.task_id), int(selection.original_pair_index))
        entry = annotations.setdefault(key, {})
        if not guards_on:
            continue
        task_id, index = key
        if task_id not in guards_by_task:
            guards_by_task[task_id] = symbolic_guards(
                [pair for pair in dataset.pairs_by_task.get(task_id, []) if pair.get("output") is not None])
        test_pairs = dataset.test_pairs_by_task.get(task_id, [])
        test_input = test_pairs[index]["input"] if index < len(test_pairs) else None
        for candidate in selection.ranked_candidates:
            rule = symbolic_guard_rejection(guards_by_task[task_id], candidate["grid"], test_input)
            if rule is not None:
                entry.setdefault(_grid_to_tuple(candidate["grid"]), {})["guard"] = rule
    for selection in final_selections:
        if selection.log_likelihoods is None:
            continue
        entry = annotations.setdefault((str(selection.task_id), int(selection.original_pair_index)), {})
        for candidate, value in zip(selection.ranked_candidates, selection.log_likelihoods):
            entry.setdefault(_grid_to_tuple(candidate["grid"]), {})["log_likelihood"] = value
    return annotations


def apply_symbolic_guards(selections: Sequence[AAIVRSelection], dataset, top_k: int = 2) -> List[AAIVRSelection]:
    """Drop candidates that break an invariant every demonstration shares; keep vote order.

    A test output whose every candidate is rejected keeps its unfiltered ranking (a guess beats none)."""
    guards_by_task: Dict[str, Dict[str, object]] = {}
    guarded = []
    for selection in selections:
        task_id = str(selection.task_id)
        if task_id not in guards_by_task:
            demos = [pair for pair in dataset.pairs_by_task.get(task_id, []) if pair.get("output") is not None]
            guards_by_task[task_id] = symbolic_guards(demos)
        test_pairs = dataset.test_pairs_by_task.get(task_id, [])
        index = int(selection.original_pair_index)
        test_input = test_pairs[index]["input"] if index < len(test_pairs) else None
        kept, rejections = [], {}
        for entry in selection.ranked_candidates:
            rule = symbolic_guard_rejection(guards_by_task[task_id], entry["grid"], test_input)
            if rule is None:
                kept.append(entry)
            else:
                rejections[rule] = rejections.get(rule, 0) + int(entry["count"])
        if not kept:
            kept = list(selection.ranked_candidates)
        guarded.append(replace(selection, ranked_candidates=kept,
                               selected_outputs=[entry["grid"] for entry in kept[:top_k]],
                               guard_rejections=rejections))
    return guarded


def run_aaivr_on_results(
    results: Sequence[Dict[str, object]],
    top_k: int = 2,
    discard_input_copies: bool = True,
    rng: Optional[random.Random] = None,
) -> List[AAIVRSelection]:
    """Invert augmentations and rank predictions by vote count."""
    rng = rng if rng is not None else random
    case_map: Dict[Tuple[str, int], Dict[str, object]] = {}

    for res in results:
        task_id = res.get("task_id")
        pair_index = res.get("pair_index")
        if task_id is None or pair_index is None:
            continue
        key = (str(task_id), int(pair_index))
        stats = case_map.setdefault(
            key,
            {"counts": {}, "generated": 0, "valid": 0, "dropped_rect": 0, "dropped_input": 0, "dropped_scale": 0},
        )
        stats["generated"] += 1
        predicted_grid = res.get("output_grid", [])
        if not is_rectangular_grid(predicted_grid):
            stats["dropped_rect"] += 1
            continue

        try:
            mapping = list(res["color_mapping"])
            source_colors = res.get("source_colors")
            if source_colors is None:
                mapping_tensor = torch.tensor(mapping, dtype=torch.long)
                inverse_tensor = torch.empty_like(mapping_tensor)
                inverse_tensor[mapping_tensor] = torch.arange(mapping_tensor.numel())
                inverse = inverse_tensor.tolist()
            else:
                inverse = invert_color_mapping(mapping, source_colors)
            normalized_grid = apply_color_permutation_to_grid(predicted_grid, inverse)
            scale = res.get("scale_factor")
            if scale is not None and not is_identity_scale(scale):
                # The output was generated in the scaled, dihedral-transformed frame.
                normalized_grid = downscale_grid_exact(
                    normalized_grid, oriented_scale_factor(scale, int(res.get("dihedral_index", 0))))
                if normalized_grid is None:
                    stats["dropped_scale"] += 1
                    continue
            normalized_grid = apply_inverse_dihedral_transform(
                normalized_grid, int(res.get("dihedral_index", 0))
            )
        except (KeyError, TypeError, ValueError):
            stats["dropped_rect"] += 1
            continue

        if not is_rectangular_grid(normalized_grid):
            stats["dropped_rect"] += 1
            continue
        if discard_input_copies and normalized_grid == res.get("original_input"):
            stats["dropped_input"] += 1
            continue

        stats["valid"] += 1
        grid_key = _grid_to_tuple(normalized_grid)
        counts = stats["counts"]
        counts[grid_key] = counts.get(grid_key, 0) + 1

    selections: List[AAIVRSelection] = []
    for (task_id, base_idx), stats in sorted(case_map.items()):
        items = list(stats["counts"].items())
        rng.shuffle(items)
        items.sort(key=lambda pair: pair[1], reverse=True)
        ranked = [{"grid": _tuple_to_grid(grid), "count": count} for grid, count in items]
        selections.append(
            AAIVRSelection(
                task_id=task_id,
                original_pair_index=base_idx,
                selected_outputs=[entry["grid"] for entry in ranked[:top_k]],
                ranked_candidates=ranked,
                num_generated=stats["generated"],
                num_valid=stats["valid"],
                discarded_non_rectangular=stats["dropped_rect"],
                discarded_input_copies=stats["dropped_input"],
                discarded_scale_mismatch=stats["dropped_scale"],
            )
        )
    return selections


# =============================================================================
# Evaluation Pipeline
# =============================================================================

def summarize_split_results(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Return basic statistics about generated results."""
    return {"total_sequences": len(results)}


def _build_submission_from_aaivr(selections: Sequence[AAIVRSelection]) -> Dict[str, List[Dict[str, object]]]:
    submission_data: Dict[str, List[Dict[str, object]]] = {}
    temp_grouping: Dict[str, Dict[int, Dict[str, object]]] = {}

    for item in selections:
        task_id = item.task_id
        pair_idx = item.original_pair_index
        if task_id not in temp_grouping:
            temp_grouping[task_id] = {}
        top_grids = item.selected_outputs[:2]
        if not top_grids:
            top_grids = [[[0]]]
        pair_dict = {"attempt_1": top_grids[0], "attempt_2": top_grids[1] if len(top_grids) > 1 else top_grids[0]}
        temp_grouping[task_id][pair_idx] = pair_dict

    for task_id, pairs_map in temp_grouping.items():
        sorted_indices = sorted(pairs_map.keys())
        submission_data[task_id] = [pairs_map[idx] for idx in sorted_indices]

    return submission_data


def run_evaluation(
    cfg: argparse.Namespace,
    *,
    run_name: str,
    max_augments: int,
    data_path: Path,
    checkpoint_path: Path,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    splits: Sequence[str] = ("test",),
    task_ids: Optional[Sequence[str]] = None,
    timing_path: Path = Path("runs/timing.txt"),
) -> Tuple[str, Dict[str, Dict[str, object]], Path]:
    """Run full decoding or teacher-forced accuracy for one configuration.

    Args:
        cfg: Config namespace; mode defaults to full-decode. accuracy-only reads
            reference outputs from cfg.solutions_path (default: beside data_path)
            or inline challenge outputs.
        run_name: Name for this evaluation run (creates runs/<run_name>/ directory).
        max_augments: Number of augmented variants per example for test-time augmentation (TTA).
            Higher values = more diverse predictions for AAIVR voting, but slower inference.
        data_path: Path to challenges.json file.
        checkpoint_path: Path to model checkpoint (.pt file).
        batch_size: Batch size for inference.
        splits: Dataset splits to evaluate (default: ["test"]).
        task_ids: Optional list of specific task IDs to evaluate (None = all tasks).
        timing_path: Path to write timing information.

    Returns:
        Tuple of (run_name, evaluation_results, artifact_path), where the artifact
        is submission.json for full-decode or accuracy.json for accuracy-only.
    """
    cfg = argparse.Namespace(**vars(cfg))
    mode = getattr(cfg, "mode", "full-decode")
    if mode not in ("full-decode", "accuracy-only", "rescore"):
        raise ValueError('Evaluation mode must be "full-decode", "accuracy-only" or "rescore".')
    timing_path = Path(timing_path)
    timing_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(checkpoint_path)
    data_path = Path(data_path)
    t_start = perf_counter()

    print(f"\n{'=' * 60}")
    mode_label = "Augmented" if getattr(cfg, "enable_aug", False) else "No-aug"
    print(f"STARTING EVALUATION: {run_name} ({mode}; {mode_label} augs: {max_augments})")
    print(f"{'=' * 60}\n")

    base_run_dir = Path("runs") / run_name
    base_run_dir.mkdir(parents=True, exist_ok=True)
    submission_path = base_run_dir / "submission.json"

    cfg.checkpoint_path = checkpoint_path
    cfg.data_path = data_path
    use_aug = bool(getattr(cfg, "enable_aug", False))
    if use_aug:
        cfg.max_augments = int(max_augments)

    checkpoint = load_checkpoint(checkpoint_path)

    print("Building model and dataset...")
    model, dataset, _, device, _ = build_model_and_data(
        cfg,
        checkpoint=checkpoint,
        is_eval=True,
        eval_splits=splits,
        build_dataloader=False,
        eval_task_ids=task_ids,
    )
    evaluation: Dict[str, Dict[str, object]] = {}
    if mode == "rescore":
        candidates_path = Path(cfg.candidates_path)
        top_k = int(getattr(cfg, "rescore_top_k", None) or DEFAULT_RESCORE_TOP_K)
        solutions_path = Path(getattr(cfg, "solutions_path", None) or data_path.with_name("solutions.json"))
        scored = run_split_rescore(
            model=model, dataset=dataset, split="test", device=device,
            candidates=_load_rescore_candidates(candidates_path, top_k), batch_size=batch_size,
            augmentation_attempts=max_augments if use_aug else 0,
            prefix_cache=bool(getattr(cfg, "rescore_prefix_cache", True)),
        )
        t_duration = perf_counter() - t_start
        rescore_path = write_rescore_report(
            base_run_dir, scored, solutions_path if solutions_path.is_file() else None,
            metadata={"mode": mode, "run_name": run_name, "checkpoint_path": str(checkpoint_path),
                      "candidates_path": str(candidates_path), "top_k": top_k, "seed": int(cfg.seed),
                      "max_augments": int(max_augments) if use_aug else 0, "duration_seconds": t_duration},
        )
        evaluation["test"] = {"summary": json.loads(rescore_path.read_text())["rules"]}
        print(f"Finished {run_name}. Rescore report saved to {rescore_path}")
        with timing_path.open("a") as handle:
            handle.write(f"Evaluation {run_name} (rescore): {t_duration:.4f} s\n")
        return (run_name, evaluation, rescore_path)
    if mode == "accuracy-only":
        solutions_path = Path(getattr(cfg, "solutions_path", None) or data_path.with_name("solutions.json"))
        solutions = {}
        if solutions_path.is_file():
            with solutions_path.open() as handle:
                solutions = json.load(handle)
            if not isinstance(solutions, dict):
                raise ValueError(f"Solutions must be a JSON object: {solutions_path}")
        print("Computing teacher-forced argmax accuracy; sampling settings are ignored.")
        for split in splits:
            summary = run_split_accuracy(
                model=model, dataset=dataset, split=split, device=device,
                solutions=solutions, batch_size=batch_size, task_ids=task_ids,
                augmentation_attempts=max_augments if use_aug else 0,
                token_budget=getattr(cfg, "prefill_token_budget", None),
            )
            evaluation[split] = {"summary": summary}
            print(
                f"Split: {split} | exact_output_accuracy={100 * summary['exact_output_accuracy']:.2f}% "
                f"({summary['exact_output_correct']}/{summary['exact_output_total']} attempts)"
            )
        t_duration = perf_counter() - t_start
        accuracy_path = base_run_dir / "accuracy.json"
        with accuracy_path.open("w") as handle:
            json.dump({
                "mode": mode,
                "run_name": run_name,
                "checkpoint_path": str(checkpoint_path),
                "data_path": str(data_path),
                "solutions_path": str(solutions_path) if solutions_path.is_file() else None,
                "seed": int(cfg.seed),
                "enable_aug": use_aug,
                "max_augments": int(max_augments) if use_aug else 0,
                "duration_seconds": t_duration,
                "splits": evaluation,
            }, handle, indent=2)
            handle.write("\n")
        print(f"Finished {run_name}. Accuracy saved to {accuracy_path}")
        print(f"Evaluation took {t_duration:.2f}s")
        with timing_path.open("a") as handle:
            handle.write(f"Evaluation {run_name} (accuracy-only): {t_duration:.4f} s\n")
        return (run_name, evaluation, accuracy_path)

    for split in splits:
        split_results = run_split_inference(
            model=model,
            dataset=dataset,
            split=split,
            device=device,
            augmentation_attempts=max_augments if use_aug else 0,
            batch_size=batch_size,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            task_ids=task_ids,
            temperature=getattr(cfg, "inference_temperature", None),
            top_k=getattr(cfg, "inference_top_k", None),
            speculative=getattr(cfg, "inference_speculative", False),
            token_budget=getattr(cfg, "prefill_token_budget", None),
        )
        evaluation[split] = {
            "results": split_results,
            "summary": summarize_split_results(split_results),
        }

    for split in splits:
        summary = evaluation.get(split, {}).get("summary", {})
        total = summary.get("total_sequences", 0)
        print(f"Split: {split} | Generated {total} sequences")

    print(f"Running AAIVR for {run_name}...")
    test_results = evaluation.get("test", {}).get("results", [])
    aaivr_results: List[AAIVRSelection] = []
    if test_results:
        aaivr_results = run_aaivr_on_results(test_results)
        print(f"AAIVR aggregated {len(test_results)} predictions into {len(aaivr_results)} submission entries")
    else:
        print("No test results for AAIVR.")

    # Selection after voting: symbolic guards drop candidates that break an invariant every demo
    # shares; likelihood rescoring then re-ranks the surviving top candidates. The plain-vote
    # top-2 (and votes + guards) stay reported beside the submitted top-2.
    voted_results = aaivr_results
    extra_top2: Dict[str, List[AAIVRSelection]] = {}
    if aaivr_results and getattr(cfg, "inference_symbolic_guards", False):
        aaivr_results = apply_symbolic_guards(aaivr_results, dataset)
        extra_top2["top2_votes"] = voted_results
        rejected = Counter()
        for selection in aaivr_results:
            rejected.update(selection.guard_rejections)
        print(f"Symbolic guards rejected {sum(rejected.values())} votes: {dict(rejected.most_common())}")
    if aaivr_results and getattr(cfg, "inference_rescore", False):
        before_rescore = aaivr_results
        aaivr_results = rescore_selections(
            model, dataset, aaivr_results, device, batch_size=batch_size,
            augmentation_attempts=max_augments if use_aug else 0,
            top_k=int(getattr(cfg, "rescore_top_k", None) or DEFAULT_RESCORE_TOP_K), task_ids=task_ids,
        )
        extra_top2["top2_votes_guards" if "top2_votes" in extra_top2 else "top2_votes"] = before_rescore

    # Unfiltered candidate dump, plus oracle and top-2 scores whenever reference outputs exist. The
    # extra aggregation uses its own RNG, so the submission's voting and tie-breaks are unchanged.
    solutions_path = Path(getattr(cfg, "solutions_path", None) or data_path.with_name("solutions.json"))
    if test_results:
        try:  # diagnostics must never cost a finished decode
            from utils import write_candidate_dump, write_oracle_report
            raw_selections = run_aaivr_on_results(test_results, discard_input_copies=False, rng=random.Random(0))
            test_inputs = {(str(r["task_id"]), int(r["pair_index"])): r.get("original_input")
                           for r in test_results if r.get("task_id") is not None and r.get("pair_index") is not None}
            write_candidate_dump(base_run_dir, raw_selections, test_inputs, max_augments if use_aug else 1,
                                 annotations=_candidate_annotations(raw_selections, aaivr_results, dataset,
                                                                    getattr(cfg, "inference_symbolic_guards", False)))
            if solutions_path.is_file():
                write_oracle_report(base_run_dir, raw_selections, aaivr_results, solutions_path,
                                    max_augments if use_aug else 1, extra_top2=extra_top2)
        except Exception as exc:
            print(f"Candidate dump or oracle scoring failed: {exc!r}")

    print(f"Generating submission.json for {run_name}...")
    submission_data = _build_submission_from_aaivr(aaivr_results)
    with submission_path.open("w") as handle:
        json.dump(submission_data, handle)

    t_duration = perf_counter() - t_start
    print(f"Finished {run_name}. Submission saved to {submission_path}")
    print(f"Evaluation took {t_duration:.2f}s")
    acceptance = speculative_summary(model)
    if acceptance:
        print(acceptance)
    with timing_path.open("a") as handle:
        handle.write(f"Evaluation {run_name}: {t_duration:.4f} s\n")
        if acceptance:
            handle.write(f"Evaluation {run_name} {acceptance}\n")

    return (run_name, evaluation, submission_path)
