"""Evaluation pipeline for the ARC transformer model.

Contains: inference generation, teacher-forced accuracy, AAIVR voting, evaluation orchestration.
For model/data building, see build.py.
"""

import argparse
import hashlib
import json
import math
import os
import random
from collections import Counter, deque
from dataclasses import dataclass
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
)
from tinytransformer import TinyTransformer
import target_demo_attention


# =============================================================================
# Inference Generation
# =============================================================================

# 30*30 colors + 29 row separators + END_PAIR (replaces the last newline).
DEFAULT_MAX_NEW_TOKENS = 30 * (30 + 1)
# Tokens per prefill chunk; lower it via MDLARC_PREFILL_TOKEN_BUDGET when deep loops make KV caches large.
PREFILL_TOKEN_BUDGET = int(os.environ.get("MDLARC_PREFILL_TOKEN_BUDGET", "524288"))
DEFAULT_EVAL_BATCH_SIZE = 120
PROMPT_LENGTH_BUCKET_SIZE = 512


def _limit_target_grids(logits, next_z, target_input_z=None):
    """After two intermediates, the remaining grid must be the final output."""
    if target_input_z is None:
        return logits
    anchor = target_input_z.squeeze(-1)
    if next_z.dim() > 1:
        anchor = anchor[:, None]
    capped = logits.clone()
    capped[..., IO_SEPARATOR_TOKEN_ID] = torch.where(next_z - anchor >= 3,
        float("-inf"), capped[..., IO_SEPARATOR_TOKEN_ID])
    return capped


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


def _compile_inference(function, **kwargs):
    # Compaction, prompt buckets and final speculative widths change shapes.
    # Generalize tensor dimensions and retain Dynamo's bounded eager fallback:
    # fullgraph=True turns its normal recompile limit into a fatal run error.
    return torch.compile(function, mode="default", dynamic=True, fullgraph=False, **kwargs)


_compiled_grid_update = _compile_inference(_grid_update)


def _speculative_grid_positions(state, tokens):
    states, positions = [state], []
    for index in range(tokens.size(1)):
        state, position = _grid_update(state, tokens[:, index], START_TOKEN_ID,
                                       IO_SEPARATOR_TOKEN_ID, END_PAIR_TOKEN_ID, NEXT_LINE_TOKEN_ID)
        states.append(state)
        positions.append(position)
    return torch.stack(states, dim=1), torch.stack(positions, dim=1)


def _commit_speculative_block(draft, targets, hidden, grid_states, cache_lengths,
                              attention_mask, emitted, finished, generated, token_limit):
    """Commit a main-head-approved prefix and leave the correction uncached."""
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
    cache_lengths = cache_lengths + torch.where(active, accepted + 1, 0)
    attention_mask &= torch.arange(attention_mask.size(1), device=targets.device)[None, :] < cache_lengths[:, None]
    anchor = hidden.gather(1, accepted[:, None, None].expand(-1, 1, hidden.size(-1))).squeeze(1)
    grid_state = grid_states.gather(1, (accepted + 1)[:, None, None].expand(-1, 1, 3)).squeeze(1)
    grid_state = torch.where(active[:, None], grid_state, grid_states[:, 0])
    pending = torch.where(finished, END_PAIR_TOKEN_ID, correction)
    return anchor, grid_state, cache_lengths, emitted, finished, pending


def _finish_generated_sequences(prompts, generated):
    results = []
    for prompt, row in zip(prompts, generated.tolist()):
        if END_PAIR_TOKEN_ID in row:
            row = row[:row.index(END_PAIR_TOKEN_ID) + 1]
        results.append(list(prompt) + row)
    return results


def _decode_mask(model, attention_mask, cache_position, device):
    # Split target/demo attention implements its own masks. In particular, do not
    # synchronize cache_position to the CPU to build an unused Flex block mask.
    if (device.type != "cuda" or getattr(model.config, "split_target_demo_attention", False)
            or not hasattr(model, "build_decode_block_mask_for_step")):
        return None
    return model.build_decode_block_mask_for_step(
        attention_mask, cache_position, attention_mask.size(1), device)


def _compact_decode_batch(past_key_values, attention_mask, shared, keep):
    """Amortize cache copies by shrinking only after at least 25% finish."""
    batch_size = attention_mask.size(0)
    def select(value):
        if torch.is_tensor(value):
            return value.index_select(0, keep) if value.ndim and value.size(0) == batch_size else value
        if isinstance(value, tuple):
            return tuple(select(item) for item in value)
        return value
    selected = {name: select(value) for name, value in shared.items() if name != "target_decode_cache"}
    if "target_decode_cache" in shared:
        rows = keep.cpu().tolist()
        selected["target_decode_cache"] = tuple(
            {"demo": target_demo_attention.select_decode_cache(layer["demo"], rows)}
            for layer in shared["target_decode_cache"])
    return select(tuple(past_key_values)), attention_mask.index_select(0, keep), selected


def _batched_speculative_decode(model, outputs, past_key_values, attention_mask,
                                initial_state, finished, current_len, token_limit, shared,
                                compact_finished=True, compile_decode=True):
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
    original_rows = torch.arange(batch_size, device=device)
    completed = generated.new_full(generated.shape, END_PAIR_TOKEN_ID)
    if compile_decode and not hasattr(model, "_compiled_verify"):
        model._compiled_verify = _compile_inference(model.forward_verify)
    if compile_decode and latent_draft and not hasattr(model, "_compiled_next_latent"):
        model._compiled_next_latent = _compile_inference(model.predict_next_latent)
    if compile_decode and not hasattr(model, "_compiled_mtp"):
        model._compiled_mtp = _compile_inference(model.predict_mtp)
        model._compiled_spec_positions = _compile_inference(_speculative_grid_positions)
        model._compiled_spec_commit = _compile_inference(_commit_speculative_block)
    verify = model._compiled_verify if compile_decode else model.forward_verify
    if latent_draft:
        next_latent = model._compiled_next_latent if compile_decode else model.predict_next_latent
        draft_tokens = lambda anchor, pending: next_latent(anchor, pending)
    else:
        mtp = model._compiled_mtp if compile_decode else model.predict_mtp
        draft_tokens = lambda anchor, pending: mtp(anchor)
    grid_positions = model._compiled_spec_positions if compile_decode else _speculative_grid_positions
    commit = model._compiled_spec_commit if compile_decode else _commit_speculative_block

    while True:
        remaining = int((~finished).sum().item())
        if not remaining:
            break
        if compact_finished and remaining * 4 <= finished.numel() * 3:
            keep = (~finished).nonzero().flatten()
            completed.index_copy_(0, original_rows, generated)
            past_key_values, attention_mask, shared = _compact_decode_batch(
                past_key_values, attention_mask, shared, keep)
            generated, emitted, finished, pending, cache_lengths, grid_state, anchor, original_rows = (
                value.index_select(0, keep) for value in
                (generated, emitted, finished, pending, cache_lengths, grid_state, anchor, original_rows))
            batch_size = remaining
        torch.compiler.cudagraph_mark_step_begin()
        # Shrink only at cache capacity; draft rejection never changes width.
        largest_start = int(torch.where(finished, 0, cache_lengths).max().item())
        width = min(width_limit, capacity - largest_start)
        draft = draft_tokens(anchor, pending)[:, :width - 1] if width > 1 else pending.new_empty((batch_size, 0))
        tokens = torch.cat([pending[:, None], draft], dim=1)
        starts = torch.where(finished, cache_lengths.clamp(max=capacity - width), cache_lengths)
        slots = starts[:, None] + torch.arange(width, device=device)[None, :]
        grid_states, positions = grid_positions(grid_state, tokens)
        key_valid = (~finished)[:, None] & (tokens != END_PAIR_TOKEN_ID).long().cumprod(dim=1).bool()
        key_valid = torch.where(finished[:, None], attention_mask.gather(1, slots), key_valid)
        attention_mask.scatter_(1, slots, key_valid)
        block_mask = _decode_mask(model, attention_mask, slots, device)
        verified = verify(input_ids=tokens, positions_3d=positions,
                                          past_key_values=tuple(past_key_values), attention_mask=attention_mask,
                                          cache_position=slots, decode_block_mask=block_mask,
                                          **shared)
        verified_logits = _limit_target_grids(verified["logits"], grid_states[:, 1:, 2], shared.get("target_input_z"))
        anchor, grid_state, cache_lengths, emitted, finished, pending = commit(
            draft, verified_logits.argmax(dim=-1), verified["hidden_states"], grid_states,
            cache_lengths, attention_mask, emitted, finished, generated, token_limit)
    completed.index_copy_(0, original_rows, generated)
    return completed[:, :token_limit]


class BatchGridState:
    def __init__(self, initial_state: torch.Tensor, *, compiled=True) -> None:
        self.state = initial_state.to(dtype=torch.int32)
        self.compiled = compiled

    def update(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.view(-1).to(device=self.state.device)
        update = _compiled_grid_update if self.compiled else _grid_update
        self.state, positions = update(
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


@torch.inference_mode()
def batched_greedy_generate(
    model, prompts, example_ids, device, max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
    cached_positions=None, temperature: Optional[float] = None, top_k: Optional[int] = None,
    padded_prompt_length: Optional[int] = None,
    dihedral_ids=None,
    pair_identity_ids=None,
    speculative=False,
    compact_finished: bool = True,
    use_prepared_cache: bool = True,
    compile_decode: bool = True,
    forced_prefixes=None,
):
    model.eval()
    batch_size = len(prompts)
    greedy = (temperature is not None and temperature <= 0) or (temperature is None and top_k is None) or top_k == 1
    if speculative is True and not greedy:
        raise ValueError("Self-speculative decoding requires greedy inference (temperature=0).")
    if not isinstance(speculative, bool) and speculative != "auto":
        raise ValueError("speculative must be a boolean or 'auto'.")
    use_speculative = (speculative is True or (speculative == "auto" and greedy)) and (
        getattr(model.config, "next_latent_prediction", False) or model.config.n_predict > 1)
    if forced_prefixes is not None:
        use_speculative = False  # Oracle tokens must enter through causal cache updates.

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
    grid_state = BatchGridState(initial_state, compiled=compile_decode)
    current_len = input_ids.size(1)

    full_attention_mask = torch.zeros((batch_size, max_model_len), dtype=torch.bool, device=device)
    full_attention_mask[:, :current_len] = attention_mask
    split_kwargs = {}
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
    ).to(dtype=model.example_embedding.weight.dtype)
    value_embeds = model.compute_value_embeds(example_ids_tensor)
    outputs = model.forward_generate(
        input_ids=input_ids, example_ids=example_ids_tensor, past_key_values=None,
        dihedral_ids=dihedral_ids_tensor,
        positions_3d=prompt_positions, attention_mask=attention_mask, conditioning_embeds=conditioning_embeds,
        value_embeds=value_embeds,
        return_hidden_states=use_speculative,
        **identity_kwargs,
    )
    prompt_kvs = outputs["past_key_values"]

    token_limit = min(max_new_tokens, max_model_len-current_len)
    if getattr(model.config, "split_target_demo_attention", False) and use_prepared_cache:
        past_key_values, demos, full_attention_mask, current_len = target_demo_attention.prepare_decode_cache(
            prompt_kvs, full_demo_mask[:, :current_len], attention_mask, token_limit,
            # One prepared demo cache per logical layer, so looped blocks get one per pass.
            [model.blocks[layer_idx].attention for layer_idx in model.layer_schedule])
        split_kwargs = {"target_decode_cache": tuple({"demo": cache} for cache in demos)}
    else:
        past_key_values = []
        for pair in prompt_kvs:
            buffers = []
            for tensor in pair:
                buffer = tensor.new_zeros((*tensor.shape[:2], max_model_len, tensor.size(-1)))
                buffer[:, :, :tensor.size(2)] = tensor
                buffers.append(buffer)
            past_key_values.append(tuple(buffers))
        del pair, tensor
    # The full prompt caches otherwise remain live for the entire generation.
    outputs = {name: value for name, value in outputs.items() if name != "past_key_values"}
    del prompt_kvs
    shared = dict(example_ids=example_ids_tensor, dihedral_ids=dihedral_ids_tensor,
                  conditioning_embeds=conditioning_embeds, value_embeds=value_embeds, **split_kwargs)
    if getattr(model.config, "predict_intermediate_grids", False):
        shared["target_input_z"] = prompt_positions[:, -1, 2:3]

    if use_speculative:
        generated = _batched_speculative_decode(
            model, outputs, past_key_values, full_attention_mask, initial_state, finished,
            current_len, token_limit, shared, compact_finished=compact_finished, compile_decode=compile_decode)
        return _finish_generated_sequences(prompts, generated)

    if compile_decode and not hasattr(model, "_compiled_decode"):
        print("Compiling model for decoding step...")
        model._compiled_decode = _compile_inference(model.forward_generate)
    decode = model._compiled_decode if compile_decode else model.forward_generate

    logits = outputs["logits"]

    cache_position = torch.tensor([current_len], dtype=torch.long, device=device)
    steps_remaining = token_limit
    generated_tokens_buffer = torch.full((batch_size, steps_remaining), END_PAIR_TOKEN_ID, dtype=torch.long, device=device)
    original_rows = torch.arange(batch_size, device=device)
    forced = None
    if forced_prefixes is not None:
        if len(forced_prefixes) != batch_size or any(len(p) >= steps_remaining for p in forced_prefixes):
            raise ValueError("Oracle prefixes require one row per prompt and room for the final grid")
        forced = torch.full((batch_size, steps_remaining), -1, dtype=torch.long, device=device)
        for row, prefix in enumerate(forced_prefixes):
            forced[row, :len(prefix)] = torch.as_tensor(prefix, device=device)

    for step_i in range(steps_remaining):
        torch.compiler.cudagraph_mark_step_begin()
        next_token = _select_next_token(
            _limit_target_grids(logits, grid_state.state[:, 2:3], shared.get("target_input_z")),
            temperature=temperature,
            top_k=top_k,
        )
        if forced is not None:
            gold = forced[original_rows, step_i]
            next_token = torch.where(gold >= 0, gold, next_token)
        next_token = torch.where(finished, torch.tensor(END_PAIR_TOKEN_ID, device=device), next_token)
        generated_tokens_buffer[original_rows, step_i] = next_token
        finished = finished | (next_token == END_PAIR_TOKEN_ID)
        remaining = int((~finished).sum().item())
        if not remaining or step_i+1 == steps_remaining:
            break
        token_positions = grid_state.update(next_token).unsqueeze(1)
        # Sampling keeps its original row/RNG schedule. Greedy rows can be
        # compacted without changing which output belongs to which attempt.
        if compact_finished and greedy and remaining * 4 <= finished.numel() * 3:
            keep = (~finished).nonzero().flatten()
            past_key_values, full_attention_mask, shared = _compact_decode_batch(
                past_key_values, full_attention_mask, shared, keep)
            next_token, finished, token_positions, grid_state.state, original_rows = (
                value.index_select(0, keep) for value in
                (next_token, finished, token_positions, grid_state.state, original_rows))
        active_rows = ~finished
        full_attention_mask[:, cache_position] = active_rows.unsqueeze(1)
        decode_block_mask = _decode_mask(model, full_attention_mask, cache_position, device)
        outputs = decode(
            input_ids=next_token.unsqueeze(1),
            past_key_values=tuple(past_key_values), positions_3d=token_positions,
            attention_mask=full_attention_mask, cache_position=cache_position,
            decode_block_mask=decode_block_mask,
            **shared,
        )
        logits = outputs["logits"]
        cache_position.add_(1)

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
        for visit, schedule_item in enumerate(schedule):
            prompt_tensor, _ = dataset.build_context_sequence(
                example,
                {"input": target_input},
                schedule_item.dihedral_index,
                schedule_item.color_mapping,
                visit=visit,
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
    speculative=False,
    prefill_token_budget: int = PREFILL_TOKEN_BUDGET,
    oracle_intermediates: bool = False,
) -> List[Dict[str, object]]:
    chain = getattr(getattr(model, "config", None), "predict_intermediate_grids", False)
    if oracle_intermediates and (split != "train" or not chain):
        raise ValueError("Gold intermediates are restricted to labeled demo diagnostics, never test queries")
    use_aug = int(augmentation_attempts) > 0
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
        token_budget=prefill_token_budget,
    )

    for batch_index, inference_batch in enumerate(batches):
        batch = inference_batch.items
        def generate(current):
            forced = None
            if oracle_intermediates:
                forced = []
                for item in current:
                    example = next(ex for ex in dataset.iter_examples(split="train")
                                   if ex.task_id == item.task_id and ex.pair_index == item.pair_index)
                    raw = [token for grid in dataset.target_intermediates(example, 0)
                           for token in grid_to_tokens(apply_dihedral_transform(grid, item.dihedral_index), IO_SEPARATOR_TOKEN_ID)]
                    forced.append([item.color_mapping[token] for token in raw])
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
                **({"forced_prefixes": forced} if forced is not None else {}),
            )
        sequences = _generate_with_oom_splitting(batch, generate)
        for sequence, item in zip(sequences, batch):
            generated = list(sequence[len(item.prompt) :])
            if END_PAIR_TOKEN_ID in generated:
                generated = generated[: generated.index(END_PAIR_TOKEN_ID)]
            trace = generated
            grids = [[]]
            if chain:
                for token in generated:
                    if token == IO_SEPARATOR_TOKEN_ID:
                        grids.append([])
                    else:
                        grids[-1].append(token)
                generated = grids[-1]
            results.append(
                {
                    "task_id": item.task_id,
                    "pair_index": item.pair_index,
                    "prompt_tokens": item.prompt,
                    "sequence": list(sequence),
                    "output_tokens": generated,
                    "output_grid": tokens_to_grid(generated),
                    **({"intermediate_grids": [tokens_to_grid(g) for g in grids[:-1]],
                        "target_trace_tokens": trace,
                        "oracle_intermediates": oracle_intermediates} if chain else {}),
                    "dihedral_index": item.dihedral_index,
                    "color_mapping": item.color_mapping,
                    "source_colors": item.source_colors,
                    "original_input": item.original_input,
                }
            )
        print(
            f"[{split}] Finished batch {batch_index + 1} / {len(batches)} "
            f"(size={len(batch)}, prompt_bucket={inference_batch.padded_prompt_length})"
        )

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
        output_tokens = grid_to_tokens(
            apply_dihedral_transform(output, item.dihedral_index), END_PAIR_TOKEN_ID
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
    prefill_token_budget: int = PREFILL_TOKEN_BUDGET,
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
    batches = _build_accuracy_batches(attempts, batch_size, token_budget=prefill_token_budget)
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
            {"counts": {}, "generated": 0, "valid": 0, "dropped_rect": 0, "dropped_input": 0},
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
    if mode not in ("full-decode", "accuracy-only", "intermediate-diagnostic"):
        raise ValueError('Unknown evaluation mode.')
    if mode == "intermediate-diagnostic":
        splits = ("train",)
    prefill_token_budget = getattr(cfg, "prefill_token_budget", PREFILL_TOKEN_BUDGET)
    if type(prefill_token_budget) is not int or prefill_token_budget <= 0:
        raise ValueError("prefill_token_budget must be a positive integer.")
    timing_path = Path(timing_path)
    timing_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(checkpoint_path)
    data_path = Path(data_path)
    t_start = perf_counter()

    print(f"\n{'=' * 60}")
    mode_label = "Augmented" if getattr(cfg, "enable_aug", False) else "No-aug"
    print(f"STARTING EVALUATION: {run_name} ({mode}; {mode_label} augs: {max_augments})")
    print(f"Evaluation batch cap: {batch_size}; prefill token budget: {prefill_token_budget}")
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
    chain = getattr(getattr(model, "config", None), "predict_intermediate_grids", False)
    if mode == "intermediate-diagnostic":
        if not chain or use_aug:
            raise ValueError("Intermediate diagnostics require a target-chain checkpoint and enable_aug=false")
        manifest = json.loads(data_path.with_name("dataset_manifest.json").read_text())
        holds = manifest.get("diagnostic_holdouts", [])
        if not holds or any(h["pair_index"] != 0 for h in holds):
            raise ValueError("Intermediate diagnostics require a prepared, withheld labeled-demo split")
        import hashlib
        for hold in holds:
            pair = dataset.pairs_by_task[hold["task"]][0]
            raw = json.dumps([pair["input"], pair["output"]], separators=(",", ":"), sort_keys=True).encode()
            if hashlib.sha256(raw).hexdigest() != hold["io_sha256"]:
                raise ValueError("Diagnostic pair differs from its withheld I/O identity")
        selected = sorted(h["task"] for h in holds)[:int(getattr(cfg, "diagnostic_cases", 64))]
        if not selected:
            raise ValueError("No eligible labeled demo targets for the intermediate diagnostic")
        diagnostic = {"protocol": "labeled evaluation demo targets withheld globally from pretraining and excluded from their own contexts",
                      "held_out_from_pretraining": True,
                      "official_arc_score": False, "cases": len(selected), "results": {}}
        for oracle in (False, True):
            results = run_split_inference(model, dataset, "train", device, batch_size=batch_size,
                max_new_tokens=3 * DEFAULT_MAX_NEW_TOKENS, task_ids=selected, pair_index=0,
                augmentation_attempts=0, speculative=getattr(cfg, "inference_speculative", "auto"),
                prefill_token_budget=prefill_token_budget, oracle_intermediates=oracle)
            correct = sum(item["output_grid"] == dataset.pairs_by_task[item["task_id"]][0]["output"] for item in results)
            diagnostic["results"]["gold_intermediates" if oracle else "generated_intermediates"] = {
                "final_grid_correct": correct, "total": len(results),
                "final_grid_accuracy": correct / max(1, len(results)), "candidates": results}
        diagnostic["gold_intermediate_accuracy_gain"] = (diagnostic["results"]["gold_intermediates"]["final_grid_accuracy"]
            - diagnostic["results"]["generated_intermediates"]["final_grid_accuracy"])
        path = base_run_dir / "intermediate_diagnostic.json"
        path.write_text(json.dumps(diagnostic, indent=2) + chr(10))
        print(f"Intermediate diagnostic: {diagnostic['cases']} demo targets; gold-intermediate gain={diagnostic['gold_intermediate_accuracy_gain']:.4f}")
        return run_name, diagnostic, path
    if mode == "accuracy-only":
        if chain:
            raise ValueError("Target-chain checkpoints require generated evaluation or the intermediate diagnostic")
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
                prefill_token_budget=prefill_token_budget,
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
                "prefill_token_budget": prefill_token_budget,
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
            max_new_tokens=(3 if chain else 1) * DEFAULT_MAX_NEW_TOKENS,
            task_ids=task_ids,
            temperature=getattr(cfg, "inference_temperature", None),
            top_k=getattr(cfg, "inference_top_k", None),
            speculative=getattr(cfg, "inference_speculative", "auto"),
            prefill_token_budget=prefill_token_budget,
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

    # Unfiltered candidate dump, plus oracle and top-2 scores whenever reference outputs exist. The
    # extra aggregation uses its own RNG, so the submission's voting and tie-breaks are unchanged.
    solutions_path = Path(getattr(cfg, "solutions_path", None) or data_path.with_name("solutions.json"))
    if test_results:
        try:  # diagnostics must never cost a finished decode
            from utils import write_candidate_dump, write_oracle_report
            raw_selections = run_aaivr_on_results(test_results, discard_input_copies=False, rng=random.Random(0))
            test_inputs = {(str(r["task_id"]), int(r["pair_index"])): r.get("original_input")
                           for r in test_results if r.get("task_id") is not None and r.get("pair_index") is not None}
            write_candidate_dump(base_run_dir, raw_selections, test_inputs, max_augments if use_aug else 1)
            if solutions_path.is_file():
                write_oracle_report(base_run_dir, raw_selections, aaivr_results, solutions_path,
                                    max_augments if use_aug else 1)
        except Exception as exc:
            print(f"Candidate dump or oracle scoring failed: {exc!r}")

    print(f"Generating submission.json for {run_name}...")
    submission_data = _build_submission_from_aaivr(aaivr_results)
    with submission_path.open("w") as handle:
        json.dump(submission_data, handle)

    t_duration = perf_counter() - t_start
    print(f"Finished {run_name}. Submission saved to {submission_path}")
    print(f"Evaluation took {t_duration:.2f}s")
    with timing_path.open("a") as handle:
        handle.write(f"Evaluation {run_name}: {t_duration:.4f} s\n")

    return (run_name, evaluation, submission_path)
