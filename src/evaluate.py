"""Evaluation pipeline for the ARC transformer model.

Contains: inference generation, AAIVR voting, evaluation orchestration, submission building.
For model/data building, see build.py.
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from build import build_model_and_data
from common import (
    Augmentor,
    Augments,
    END_TOKEN_ID,
    IO_SEPARATOR_TOKEN_ID,
    NEXT_LINE_TOKEN_ID,
    START_TOKEN_ID,
    VOCAB_SIZE,
    apply_color_permutation_to_grid,
    apply_color_permutation_to_tokens,
    apply_dihedral_transform,
    apply_inverse_dihedral_transform,
    build_augmentor,
    compute_positions_3d,
    extract_output_tokens,
    grid_to_tokens,
    is_rectangular_grid,
    split_grids_from_tokens,
    tokens_to_grid,
)
from tinytransformer import TinyTransformer


# =============================================================================
# Inference Generation
# =============================================================================

DEFAULT_MAX_NEW_TOKENS = 931


@torch.compile(mode="default", fullgraph=True)
def _compiled_grid_update(state, token_ids, start_id, sep_id, end_id, nl_id):
    x, y, z = state.unbind(-1)
    pos_x = torch.clamp(x, min=0, max=30)
    pos_y = torch.clamp(y, min=0, max=29)
    pos_z = z

    is_start = token_ids == start_id
    is_sep = token_ids == sep_id
    is_end = token_ids == end_id
    is_newline = token_ids == nl_id

    zeros = torch.zeros_like(x)
    pos_x = torch.where(is_start | is_sep | is_end, zeros, pos_x)
    pos_y = torch.where(is_start | is_sep | is_end, zeros, pos_y)
    pos_z = torch.where(is_start, zeros, pos_z)
    pos_z = torch.where(is_sep, torch.full_like(pos_z, 2), pos_z)
    pos_z = torch.where(is_end, torch.full_like(pos_z, 4), pos_z)

    next_x = x + 1
    next_y = y
    next_z = z
    next_x = torch.where(is_newline, zeros, next_x)
    next_y = torch.where(is_newline, y + 1, next_y)
    next_x = torch.where(is_sep, zeros, next_x)
    next_y = torch.where(is_sep, zeros, next_y)
    next_z = torch.where(is_sep, torch.full_like(next_z, 3), next_z)
    next_x = torch.where(is_end | is_start, x, next_x)
    next_y = torch.where(is_end | is_start, y, next_y)
    next_z = torch.where(is_start, z, next_z)
    next_z = torch.where(is_end, z, next_z)

    return torch.stack([next_x, next_y, next_z], dim=-1), torch.stack([pos_x, pos_y, pos_z], dim=-1)


class BatchGridState:
    def __init__(self, initial_state: torch.Tensor) -> None:
        self.state = initial_state.clone().long()

    def update(self, token_ids: torch.Tensor) -> torch.Tensor:
        token_ids = token_ids.view(-1).to(device=self.state.device)
        self.state, positions = _compiled_grid_update(
            self.state, token_ids, START_TOKEN_ID, IO_SEPARATOR_TOKEN_ID, END_TOKEN_ID, NEXT_LINE_TOKEN_ID
        )
        self.state = self.state.clone()
        return positions.clone()


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


def _left_pad_sequences(sequences: Sequence[Sequence[int]], pad_token_id: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(sequences)
    max_len = max(len(seq) for seq in sequences)
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    for idx, seq in enumerate(sequences):
        seq_len = len(seq)
        start = max_len - seq_len
        input_ids[idx, start:] = torch.tensor(seq, dtype=torch.long, device=device)
        attention_mask[idx, start:] = True
    return input_ids, attention_mask


def _pad_cached_positions(cached_positions: Sequence[torch.Tensor], max_len: int, device: torch.device) -> torch.Tensor:
    positions = torch.zeros((len(cached_positions), max_len, 3), dtype=torch.long, device=device)
    for idx, pos in enumerate(cached_positions):
        seq_len = pos.size(0)
        start = max_len - seq_len
        positions[idx, start:] = pos.to(device=device, dtype=torch.long)
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
    next_x = torch.where(last_tokens == IO_SEPARATOR_TOKEN_ID, torch.zeros_like(next_x), next_x)
    next_y = torch.where(last_tokens == IO_SEPARATOR_TOKEN_ID, torch.zeros_like(next_y), next_y)
    next_z = torch.where(last_tokens == IO_SEPARATOR_TOKEN_ID, torch.full_like(next_z, 3), next_z)
    next_x = torch.where(last_tokens == END_TOKEN_ID, x, next_x)
    next_y = torch.where(last_tokens == END_TOKEN_ID, y, next_y)
    next_z = torch.where(last_tokens == END_TOKEN_ID, z, next_z)
    next_x = torch.where(last_tokens == START_TOKEN_ID, torch.zeros_like(next_x), next_x)
    next_y = torch.where(last_tokens == START_TOKEN_ID, torch.zeros_like(next_y), next_y)
    next_z = torch.where(last_tokens == START_TOKEN_ID, torch.ones_like(next_z), next_z)

    initial_state = torch.stack([next_x, next_y, next_z], dim=-1)
    finished = last_tokens == END_TOKEN_ID
    return initial_state, finished


@torch.inference_mode()
def batched_greedy_generate(
    model, prompts, example_ids, device, max_new_tokens=931,
    cached_positions=None, temperature: Optional[float] = None, top_k: Optional[int] = None,
):
    model.eval()
    model.to(dtype=torch.bfloat16)
    batch_size = len(prompts)

    example_ids_tensor = torch.tensor(example_ids, dtype=torch.long, device=device)
    input_ids, attention_mask = _left_pad_sequences(prompts, END_TOKEN_ID, device)
    current_len = input_ids.size(1)
    batch_max_needed = current_len + max_new_tokens
    batch_max_needed = (batch_max_needed + 127) // 128 * 128
    max_model_len = min(batch_max_needed, model.config.max_seq_len)

    if cached_positions and all(p is not None for p in cached_positions):
        prompt_positions = _pad_cached_positions([p for p in cached_positions if p is not None], input_ids.size(1), device)
    else:
        prompt_positions = compute_positions_3d(input_ids, attention_mask).to(device=device, dtype=torch.long)

    initial_state, finished = _derive_initial_state_from_prompt(input_ids, prompt_positions)
    grid_state = BatchGridState(initial_state)
    example_embeds = model.example_embedding(example_ids_tensor).to(dtype=torch.bfloat16)
    current_len = input_ids.size(1)

    full_attention_mask = torch.zeros((batch_size, max_model_len), dtype=torch.bool, device=device)
    full_attention_mask[:, :current_len] = attention_mask

    outputs = model.forward_generate(
        input_ids=input_ids, example_ids=example_ids_tensor, past_key_values=None,
        positions_3d=prompt_positions, attention_mask=attention_mask, example_embeds=example_embeds,
    )
    logits = outputs["logits"]
    prompt_kvs = outputs["past_key_values"]

    past_key_values = []
    for k, v in prompt_kvs:
        B, H, L, D = k.shape
        k_buf = torch.zeros((B, H, max_model_len, D), dtype=torch.bfloat16, device=device)
        v_buf = torch.zeros((B, H, max_model_len, D), dtype=torch.bfloat16, device=device)
        k_buf[:, :, :L, :] = k
        v_buf[:, :, :L, :] = v
        past_key_values.append((k_buf, v_buf))
    past_key_values = tuple(past_key_values)

    if not hasattr(model, "_compiled_decode"):
        print("Compiling model for decoding step...")
        model._compiled_decode = torch.compile(model.forward_generate, mode="default", fullgraph=True)

    cache_position = torch.tensor([current_len], dtype=torch.long, device=device)
    steps_remaining = min(max_new_tokens, max_model_len - current_len)
    generated_tokens_buffer = torch.full((batch_size, steps_remaining), END_TOKEN_ID, dtype=torch.long, device=device)

    for step_i in range(steps_remaining):
        if finished.all():
            break
        torch.compiler.cudagraph_mark_step_begin()
        next_token = _select_next_token(logits, temperature=temperature, top_k=top_k)
        next_token = torch.where(finished, torch.tensor(END_TOKEN_ID, device=device), next_token)
        generated_tokens_buffer[:, step_i] = next_token
        finished = finished | (next_token == END_TOKEN_ID)
        token_positions = grid_state.update(next_token).unsqueeze(1)
        full_attention_mask.index_fill_(1, cache_position, True)
        outputs = model._compiled_decode(
            input_ids=next_token.unsqueeze(1), example_ids=example_ids_tensor,
            past_key_values=past_key_values, positions_3d=token_positions,
            attention_mask=full_attention_mask, cache_position=cache_position, example_embeds=example_embeds,
        )
        logits = outputs["logits"]
        cache_position.add_(1)

    generated_cpu = generated_tokens_buffer.tolist()
    results = []
    for i, prompt in enumerate(prompts):
        gen_seq = []
        for token in generated_cpu[i]:
            gen_seq.append(token)
            if token == END_TOKEN_ID:
                break
        results.append(list(prompt) + gen_seq)
    return results


def _build_prompt_from_tokens(tokens: Sequence[int]) -> List[int]:
    if IO_SEPARATOR_TOKEN_ID not in tokens:
        raise ValueError("Prompt sequence is missing <input_output_separator>.")
    sep_idx = tokens.index(IO_SEPARATOR_TOKEN_ID)
    return list(tokens[: sep_idx + 1])


def _select_tokens_for_example(example: object, transform_index: Optional[int]) -> Tuple[List[int], Optional[torch.Tensor]]:
    tokens = getattr(example, "tokens")
    cached_positions = getattr(example, "cached_positions", None)
    if transform_index is not None:
        tokens_by_dihedral = getattr(example, "tokens_by_dihedral", None)
        if tokens_by_dihedral:
            if transform_index < 0 or transform_index >= len(tokens_by_dihedral):
                raise ValueError(f"Invalid dihedral transform index {transform_index}.")
            tokens = tokens_by_dihedral[transform_index]
            cached_by_dihedral = getattr(example, "cached_positions_by_dihedral", None)
            if cached_by_dihedral:
                cached_positions = cached_by_dihedral[transform_index]
    token_list = tokens.tolist() if isinstance(tokens, torch.Tensor) else list(tokens)
    return token_list, cached_positions


def _sequence_length_for_example(example: object, transform_index: Optional[int]) -> int:
    if transform_index is not None:
        tokens_by_dihedral = getattr(example, "tokens_by_dihedral", None)
        if tokens_by_dihedral:
            return int(tokens_by_dihedral[transform_index].size(0))
    seq_len = getattr(example, "seq_len", None)
    if seq_len is not None:
        return int(seq_len)
    tokens = getattr(example, "tokens")
    return int(tokens.size(0) if isinstance(tokens, torch.Tensor) else len(tokens))


def _prepare_examples_for_inference(
    examples: Sequence[object],
    include_targets: bool = False,
    solutions: Optional[Dict[Tuple[str, int], List[List[int]]]] = None,
    color_mappings: Optional[Sequence[Optional[Sequence[int]]]] = None,
    color_apply_fn: Optional[Callable[[str], bool]] = None,
    dihedral_transform_indices: Optional[Sequence[Optional[int]]] = None,
    pair_indices: Optional[Sequence[int]] = None,
) -> Tuple[List[List[int]], List[int], List[Dict[str, object]], List[Optional[torch.Tensor]], List[List[int]]]:
    prompts: List[List[int]] = []
    example_ids: List[int] = []
    metadata: List[Dict[str, object]] = []
    cached_positions: List[Optional[torch.Tensor]] = []
    target_tokens: List[List[int]] = []

    for idx, ex in enumerate(examples):
        if not hasattr(ex, "tokens"):
            raise ValueError("Examples must provide a 'tokens' attribute.")
        transform_index = dihedral_transform_indices[idx] if dihedral_transform_indices is not None else None
        raw_tokens, cached = _select_tokens_for_example(ex, transform_index)
        split = getattr(ex, "split", None)
        mapping = color_mappings[idx] if color_mappings is not None else None
        should_color = mapping is not None and (color_apply_fn is None or color_apply_fn(split))
        tokens = apply_color_permutation_to_tokens(raw_tokens, mapping) if should_color else raw_tokens
        prompt_tokens = _build_prompt_from_tokens(tokens)
        prompts.append(prompt_tokens)
        example_ids.append(int(getattr(ex, "example_id", 0)))
        if cached is not None:
            cached_positions.append(cached[: len(prompt_tokens)])
        else:
            cached_positions.append(None)

        targets: List[int] = []
        if include_targets and getattr(ex, "has_output", False):
            targets = extract_output_tokens(tokens)
        elif include_targets and solutions is not None:
            key = (getattr(ex, "task_id", None), getattr(ex, "pair_index", None))
            if key in solutions and solutions[key] is not None:
                target_grid = solutions[key]
                if transform_index is not None:
                    target_grid = apply_dihedral_transform(target_grid, transform_index)
                if should_color:
                    target_grid = apply_color_permutation_to_grid(target_grid, mapping)
                targets = grid_to_tokens(target_grid)
        target_tokens.append(targets)
        pair_index = pair_indices[idx] if pair_indices is not None else getattr(ex, "pair_index", None)
        metadata.append({
            "task_id": getattr(ex, "task_id", None),
            "pair_index": pair_index,
            "example_id": getattr(ex, "example_id", None),
            "split": getattr(ex, "split", None),
        })

    return prompts, example_ids, metadata, cached_positions, target_tokens


def _build_generation_results(
    sequences: Sequence[Sequence[int]],
    metadata: Sequence[Dict[str, object]],
    prompts: Sequence[Sequence[int]],
    target_output_tokens: Sequence[Sequence[int]],
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for seq, meta, prompt, target in zip(sequences, metadata, prompts, target_output_tokens):
        output_tokens = extract_output_tokens(seq)
        predicted_grid = tokens_to_grid(output_tokens)
        target_grid = tokens_to_grid(target) if target else []
        result = {
            "task_id": meta.get("task_id"),
            "pair_index": meta.get("pair_index"),
            "example_id": meta.get("example_id"),
            "split": meta.get("split"),
            "prompt_tokens": list(prompt),
            "sequence": list(seq),
            "output_tokens": output_tokens,
            "output_grid": predicted_grid,
            "target_output_tokens": list(target),
            "target_grid": target_grid,
        }
        results.append(result)
    return results


def _run_generation_batch(
    model: TinyTransformer,
    prompts: Sequence[Sequence[int]],
    example_ids: Sequence[int],
    metadata: Sequence[Dict[str, object]],
    cached_positions: Sequence[Optional[torch.Tensor]],
    device: torch.device,
    max_new_tokens: int,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
    target_output_tokens: Optional[Sequence[Sequence[int]]] = None,
) -> List[Dict[str, object]]:
    sequences = batched_greedy_generate(
        model=model, prompts=prompts, example_ids=example_ids, device=device,
        max_new_tokens=max_new_tokens, cached_positions=cached_positions,
        temperature=temperature, top_k=top_k,
    )
    return _build_generation_results(
        sequences=sequences, metadata=metadata, prompts=prompts,
        target_output_tokens=target_output_tokens if target_output_tokens is not None else [[] for _ in prompts],
    )


def _load_solutions_for_dataset(dataset) -> Dict[Tuple[str, int], List[List[int]]]:
    solutions_map: Dict[Tuple[str, int], List[List[int]]] = {}
    source_path = getattr(dataset, "source_path", None)
    if source_path is None:
        return solutions_map
    solutions_path = Path(source_path).with_name("solutions.json")
    if not solutions_path.exists():
        return solutions_map
    try:
        data = json.loads(solutions_path.read_text())
        for task_id, outputs in data.items():
            if not isinstance(outputs, list):
                continue
            for idx, grid in enumerate(outputs):
                if grid is not None:
                    solutions_map[(task_id, idx)] = grid
    except Exception:
        return solutions_map
    return solutions_map


def _gather_examples_for_split(
    dataset, split: str, task_ids: Optional[Sequence[str]] = None, pair_index: Optional[int] = None,
    require_outputs: bool = False, solutions: Optional[Dict[Tuple[str, int], List[List[int]]]] = None,
):
    examples = []
    for example in dataset.iter_examples(split=split):
        if task_ids is not None and example.task_id not in task_ids:
            continue
        if pair_index is not None and example.pair_index != pair_index:
            continue
        has_solution = solutions is not None and ((example.task_id, example.pair_index) in solutions)
        if require_outputs and not example.has_output and not has_solution:
            continue
        examples.append(example)
    return examples


def _identity_mapping() -> List[int]:
    return list(range(VOCAB_SIZE))


def _split_allows_color(augmentor: Augmentor, split: str) -> bool:
    if split == "test":
        return bool(augmentor.color_apply_to_test_split)
    return True


def _split_allows_dihedral(augmentor: Augmentor, split: str) -> bool:
    if split == "test":
        return bool(augmentor.dihedral_apply_to_test_split)
    return True


def _allowed_tuple_indices(augments: Augments, *, allow_color: bool, allow_dihedral: bool) -> List[int]:
    indices: List[int] = []
    for idx, (d_idx, c_idx) in enumerate(zip(augments.dihedral_indices, augments.color_map_indices)):
        if not allow_color and c_idx != 0:
            continue
        if not allow_dihedral and d_idx != 0:
            continue
        indices.append(idx)
    if not indices:
        indices = [augments.identity_tuple_index]
    return indices


def _build_color_mappings_by_task(
    examples: Sequence[object], augmentor: Augmentor, split: str,
) -> Tuple[Dict[str, List[List[int]]], Dict[str, Dict[Tuple[int, ...], int]]]:
    mappings_by_task: Dict[str, List[List[int]]] = {}
    mapping_index_by_task: Dict[str, Dict[Tuple[int, ...], int]] = {}
    identity = _identity_mapping()
    identity_key = tuple(identity)
    allow_color = _split_allows_color(augmentor, split)
    allow_dihedral = _split_allows_dihedral(augmentor, split)

    for ex in examples:
        task_id = getattr(ex, "task_id", None)
        if task_id is None:
            continue
        if task_id not in mappings_by_task:
            mappings_by_task[task_id] = [identity]
            mapping_index_by_task[task_id] = {identity_key: 0}

    for ex in examples:
        task_id = getattr(ex, "task_id", None)
        if task_id is None:
            continue
        key = (task_id, getattr(ex, "split", None), getattr(ex, "pair_index", None))
        augments = augmentor.augments_by_key.get(key)
        if augments is None:
            continue
        allowed = _allowed_tuple_indices(augments, allow_color=allow_color, allow_dihedral=allow_dihedral)
        for idx in allowed:
            map_idx = augments.color_map_indices[idx]
            mapping = augments.color_maps[map_idx].tolist()
            mapping_key = tuple(mapping)
            lookup = mapping_index_by_task[task_id]
            if mapping_key not in lookup:
                lookup[mapping_key] = len(mappings_by_task[task_id])
                mappings_by_task[task_id].append(mapping)
    return mappings_by_task, mapping_index_by_task


@torch.inference_mode()
def run_split_inference(
    model: TinyTransformer, dataset, split: str, device: torch.device,
    batch_size: int = 16, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    task_ids: Optional[Sequence[str]] = None, pair_index: Optional[int] = None,
    include_targets: bool = True, temperature: Optional[float] = None, top_k: Optional[int] = None,
    augmentor: Optional[Augmentor] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, List[List[int]]], bool]:
    solutions = _load_solutions_for_dataset(dataset) if include_targets else None
    examples = _gather_examples_for_split(
        dataset, split=split, task_ids=task_ids, pair_index=pair_index,
        require_outputs=include_targets, solutions=solutions,
    )
    if not examples:
        return [], {}, False

    work_items = []
    color_mappings_by_task: Dict[str, List[List[int]]] = {}
    dihedral_augmented = False
    pack_dihedral = False

    if augmentor is None:
        for ex in examples:
            work_items.append((ex, None, None, 0, None, ex.pair_index, _sequence_length_for_example(ex, None)))
    else:
        allow_color = _split_allows_color(augmentor, split)
        allow_dihedral = _split_allows_dihedral(augmentor, split)
        color_mappings_by_task, mapping_index_by_task = _build_color_mappings_by_task(examples, augmentor, split)
        for ex in examples:
            task_id = getattr(ex, "task_id", None)
            if task_id is None:
                continue
            key = (task_id, getattr(ex, "split", None), getattr(ex, "pair_index", None))
            augments = augmentor.augments_by_key.get(key)
            if augments is None:
                continue
            allowed = _allowed_tuple_indices(augments, allow_color=allow_color, allow_dihedral=allow_dihedral)
            for idx in allowed:
                dihedral_idx = int(augments.dihedral_indices[idx])
                color_map_idx = int(augments.color_map_indices[idx])
                mapping = augments.color_maps[color_map_idx].tolist()
                mapping_key = tuple(mapping)
                mapping_idx = mapping_index_by_task[task_id].get(mapping_key, 0)
                seq_len = _sequence_length_for_example(ex, dihedral_idx)
                base_pair_index = getattr(ex, "pair_index", None)
                work_items.append((ex, dihedral_idx, dihedral_idx, mapping_idx, mapping, base_pair_index, seq_len))
                if dihedral_idx != 0:
                    dihedral_augmented = True
        pack_dihedral = allow_dihedral and dihedral_augmented

    work_items.sort(key=lambda item: item[-1], reverse=True)
    all_results: List[Dict[str, object]] = []

    for start in range(0, len(work_items), batch_size):
        chunk = work_items[start : start + batch_size]
        batch_examples = [item[0] for item in chunk]
        batch_transform_indices = [item[2] for item in chunk]
        batch_c_indices = [item[3] for item in chunk]
        batch_mappings = [item[4] for item in chunk]
        batch_pair_indices = []
        for item in chunk:
            base_pair_index = item[5]
            if augmentor is not None and pack_dihedral and base_pair_index is not None:
                batch_pair_indices.append(int(base_pair_index) * 8 + item[1])
            else:
                batch_pair_indices.append(base_pair_index)

        if augmentor is None:
            prompts, example_ids, metadata, cached_positions, target_output_tokens = _prepare_examples_for_inference(
                batch_examples, include_targets=include_targets, solutions=solutions,
            )
        else:
            prompts, example_ids, metadata, cached_positions, target_output_tokens = _prepare_examples_for_inference(
                batch_examples, include_targets=include_targets, solutions=solutions,
                color_mappings=batch_mappings, color_apply_fn=None,
                dihedral_transform_indices=batch_transform_indices, pair_indices=batch_pair_indices,
            )

        batch_results = _run_generation_batch(
            model=model, prompts=prompts, example_ids=example_ids, metadata=metadata,
            cached_positions=cached_positions, device=device, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k,
            target_output_tokens=target_output_tokens if include_targets else None,
        )

        print(f"[{split}] Finished batch {start // batch_size + 1} / {(len(work_items) + batch_size - 1) // batch_size}")

        if augmentor is None:
            all_results.extend(batch_results)
        else:
            for res, c_idx, d_idx in zip(batch_results, batch_c_indices, batch_transform_indices):
                res["color_permutation_index"] = c_idx
                res["dihedral_index"] = d_idx
                all_results.append(res)

    return all_results, color_mappings_by_task, dihedral_augmented


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
    target_grid: Optional[List[List[int]]] = None
    pass_at_k: Optional[bool] = None


def run_aaivr_on_results(
    results: Sequence[Dict[str, object]],
    top_k: int = 2,
    discard_input_copies: bool = True,
    rng: Optional[random.Random] = None,
    is_dihedral_augmented: bool = False,
    color_mappings_by_task: Optional[Dict[str, Sequence[Sequence[int]]]] = None,
) -> List[AAIVRSelection]:
    """Aggregate augmented predictions via AAIVR voting."""
    rng = rng if rng is not None else random
    case_map: Dict[Tuple[str, int], Dict[str, object]] = {}

    inverse_color_mappings_by_task: Dict[str, List[List[int]]] = {}
    if color_mappings_by_task is not None:
        for task_id, mappings in color_mappings_by_task.items():
            inv_list: List[List[int]] = []
            for mapping in mappings:
                fwd = mapping if isinstance(mapping, torch.Tensor) else torch.tensor(mapping, dtype=torch.long)
                inv = torch.zeros_like(fwd)
                inv[fwd] = torch.arange(len(fwd), dtype=torch.long, device=fwd.device)
                inv_list.append(inv.tolist())
            inverse_color_mappings_by_task[task_id] = inv_list

    for res in results:
        task_id = res.get("task_id")
        pair_index = res.get("pair_index")
        if task_id is None or pair_index is None:
            continue

        if is_dihedral_augmented:
            base_pair_index = int(pair_index) // 8
            transform_index = int(pair_index) % 8
        else:
            base_pair_index = int(pair_index)
            transform_index = 0

        color_idx = res.get("color_permutation_index", 0)
        predicted_grid = res.get("output_grid", [])
        prompt_tokens = res.get("prompt_tokens", [])
        input_grids = split_grids_from_tokens(prompt_tokens)
        input_grid = input_grids[0] if input_grids else []

        key = (task_id, base_pair_index)
        if key not in case_map:
            case_map[key] = {"counts": {}, "generated": 0, "valid": 0, "dropped_rect": 0, "dropped_input": 0, "target_grid": None}
        stats = case_map[key]
        stats["generated"] += 1

        target_grid = res.get("target_grid", [])
        if stats["target_grid"] is None and is_rectangular_grid(target_grid):
            try:
                norm_target = apply_inverse_dihedral_transform(target_grid, transform_index)
                inv_list = inverse_color_mappings_by_task.get(task_id, [])
                if inv_list and color_idx > 0 and color_idx < len(inv_list):
                    norm_target = apply_color_permutation_to_grid(norm_target, inv_list[color_idx])
                if is_rectangular_grid(norm_target):
                    stats["target_grid"] = norm_target
            except Exception:
                pass

        if not is_rectangular_grid(predicted_grid):
            stats["dropped_rect"] += 1
            continue
        if discard_input_copies and input_grid and predicted_grid == input_grid:
            stats["dropped_input"] += 1
            continue

        try:
            normalized_grid = apply_inverse_dihedral_transform(predicted_grid, transform_index)
            inv_list = inverse_color_mappings_by_task.get(task_id, [])
            if inv_list and color_idx > 0 and color_idx < len(inv_list):
                normalized_grid = apply_color_permutation_to_grid(normalized_grid, inv_list[color_idx])
        except Exception:
            stats["dropped_rect"] += 1
            continue

        if not is_rectangular_grid(normalized_grid):
            stats["dropped_rect"] += 1
            continue

        stats["valid"] += 1
        grid_key = _grid_to_tuple(normalized_grid)
        counts: Dict[Tuple[Tuple[int, ...], ...], int] = stats["counts"]
        counts[grid_key] = counts.get(grid_key, 0) + 1

    selections: List[AAIVRSelection] = []
    for (task_id, base_idx), stats in sorted(case_map.items()):
        items = list(stats["counts"].items())
        if items:
            rng.shuffle(items)
            items.sort(key=lambda pair: pair[1], reverse=True)
        ranked_candidates = [{"grid": _tuple_to_grid(grid_key), "count": count} for grid_key, count in items]
        selected_outputs = [entry["grid"] for entry in ranked_candidates[:top_k]]

        target_grid = stats.get("target_grid")
        pass_at_k = None
        if target_grid is not None:
            pass_at_k = any(grid == target_grid for grid in selected_outputs)

        selections.append(AAIVRSelection(
            task_id=task_id, original_pair_index=base_idx, selected_outputs=selected_outputs,
            ranked_candidates=ranked_candidates, num_generated=stats["generated"], num_valid=stats["valid"],
            discarded_non_rectangular=stats["dropped_rect"], discarded_input_copies=stats["dropped_input"],
            target_grid=target_grid, pass_at_k=pass_at_k,
        ))

    return selections


def summarize_aaivr_pass_at_k(selections: Sequence[AAIVRSelection]) -> Dict[str, int]:
    """Return counts for how many tasks have ALL their pairs in top-k."""
    tasks: Dict[str, List[AAIVRSelection]] = {}
    for sel in selections:
        tasks.setdefault(sel.task_id, []).append(sel)

    total_tasks = len(tasks)
    solved_tasks = 0
    failures = []

    for task_id, pairs in tasks.items():
        is_solved = True
        pair_failures = []
        for p in pairs:
            if p.pass_at_k is None:
                is_solved = False
                pair_failures.append(f"Pair {p.original_pair_index}: Target missing/unknown")
            elif not p.pass_at_k:
                is_solved = False
                if p.num_valid == 0:
                    reason = f"No valid candidates generated (tried {p.num_generated})"
                else:
                    reason = "Top-k candidates incorrect"
                pair_failures.append(f"Pair {p.original_pair_index}: {reason}")

        if is_solved and len(pairs) > 0:
            solved_tasks += 1
        else:
            failures.append(f"Task {task_id}: {', '.join(pair_failures)}")

    if failures:
        print(f"\nAAIVR Failures ({len(failures)}/{total_tasks} tasks):")
        for f in failures:
            print(f"  - {f}")

    return {"evaluated": total_tasks, "hits": solved_tasks}


# =============================================================================
# Evaluation Pipeline
# =============================================================================

def _has_correct_shape(sequence: Sequence[int], predicted_tokens: Sequence[int], target_tokens: Sequence[int]) -> bool:
    if not target_tokens:
        return False
    if len(predicted_tokens) != len(target_tokens):
        return False
    target_newlines = [idx for idx, tok in enumerate(target_tokens) if tok == NEXT_LINE_TOKEN_ID]
    predicted_newlines = [idx for idx, tok in enumerate(predicted_tokens) if tok == NEXT_LINE_TOKEN_ID]
    if target_newlines != predicted_newlines:
        return False
    try:
        sep_idx = sequence.index(IO_SEPARATOR_TOKEN_ID)
        end_idx = sequence.index(END_TOKEN_ID, sep_idx + 1)
    except ValueError:
        return False
    return (end_idx - (sep_idx + 1)) == len(target_tokens)


def _pixel_accuracy(predicted_tokens: Sequence[int], target_tokens: Sequence[int]) -> Optional[float]:
    predicted_digits = [tok for tok in predicted_tokens if 0 <= tok <= 9]
    target_digits = [tok for tok in target_tokens if 0 <= tok <= 9]
    if not target_digits or len(predicted_digits) != len(target_digits):
        return None
    correct = sum(1 for p, t in zip(predicted_digits, target_digits) if p == t)
    return correct / len(target_digits)


def summarize_split_results(results: Sequence[Dict[str, object]]) -> Dict[str, object]:
    num_shape_correct = 0
    num_fully_correct = 0
    accuracies: List[float] = []
    fully_correct_results: List[Dict[str, object]] = []

    for res in results:
        predicted_tokens = res.get("output_tokens", [])
        target_tokens = res.get("target_output_tokens", [])
        sequence = res.get("sequence", [])
        shape_ok = _has_correct_shape(sequence, predicted_tokens, target_tokens)
        res["shape_correct"] = shape_ok
        if not shape_ok:
            continue
        num_shape_correct += 1
        acc = _pixel_accuracy(predicted_tokens, target_tokens)
        if acc is not None:
            res["pixel_accuracy"] = acc
            accuracies.append(acc)
        if predicted_tokens == target_tokens:
            num_fully_correct += 1
            fully_correct_results.append(res)

    avg_pixel_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    return {
        "total_sequences": len(results),
        "num_shape_correct": num_shape_correct,
        "avg_pixel_accuracy": avg_pixel_accuracy,
        "num_fully_correct": num_fully_correct,
        "fully_correct_results": fully_correct_results,
    }


class TeeLogger:
    def __init__(self, filepath: Path) -> None:
        self.terminal = sys.stdout
        self.log = Path(filepath).open("w")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.close()


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


def _compute_arc_style_score(selections: Sequence[AAIVRSelection]) -> Tuple[float, int, float]:
    tasks_map: Dict[str, List[AAIVRSelection]] = {}
    for res in selections:
        tasks_map.setdefault(res.task_id, []).append(res)

    arc_score = 0.0
    total_tasks = len(tasks_map)
    for pairs in tasks_map.values():
        n_pairs = len(pairs)
        if n_pairs > 0:
            n_solved = sum(1 for p in pairs if p.pass_at_k)
            arc_score += n_solved / n_pairs

    max_score = total_tasks
    pct = (arc_score / max_score * 100) if max_score > 0 else 0.0
    return arc_score, max_score, pct


def run_evaluation_configs(
    cfg: argparse.Namespace,
    eval_configs: Sequence[Tuple[str, int, Path]],
    *,
    eval_batch_size: int = 1300,
    splits: Sequence[str] = ("test",),
    checkpoint_path: Optional[Path] = None,
    include_targets: bool = False,
    task_ids: Optional[Sequence[str]] = None,
    log_correct_grids: bool = False,
    timing_path: Path = Path("runs/timing.txt"),
) -> List[Tuple[str, Dict[str, Dict[str, object]], Path]]:
    timing_path = Path(timing_path)
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    state: Dict[str, object] = {}
    results: List[Tuple[str, Dict[str, Dict[str, object]], Path]] = []

    prev_checkpoint = getattr(cfg, "checkpoint_path", None)
    prev_data_path = getattr(cfg, "data_path", None)
    prev_aug = cfg.max_augments

    resolved_checkpoint_path = checkpoint_path
    if resolved_checkpoint_path is None:
        resolved_checkpoint_path = getattr(cfg, "checkpoint_path", None)
    if resolved_checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided for evaluation.")

    for config in eval_configs:
        if len(config) != 3:
            raise ValueError("eval_configs entries must be (name, aug_count, data_path).")
        run_name, max_eval_augments, dataset_path = config
        dataset_path = Path(dataset_path)
        t_start = perf_counter()

        print(f"\n{'=' * 60}")
        mode_label = "Augmented" if getattr(cfg, "enable_aug", False) else "No-aug"
        print(f"STARTING PIPELINE: {run_name} ({mode_label} augs: {max_eval_augments})")
        print(f"{'=' * 60}\n")

        base_run_dir = Path("runs") / run_name
        base_run_dir.mkdir(parents=True, exist_ok=True)
        eval_log_path = base_run_dir / "eval_log.txt"
        aaivr_log_path = base_run_dir / "aaivr.txt"
        submission_path = base_run_dir / "submission.json"

        cfg.checkpoint_path = Path(resolved_checkpoint_path)
        cfg.data_path = dataset_path
        use_aug = bool(getattr(cfg, "enable_aug", False))
        if use_aug:
            cfg.max_augments = int(max_eval_augments)

        reuse_dataset = None
        prior_dataset = state.get("dataset")
        if prior_dataset is not None:
            prior_path = getattr(prior_dataset, "source_path", None)
            if prior_path is not None and Path(prior_path) == dataset_path:
                reuse_dataset = prior_dataset

        checkpoint = state.get("checkpoint")
        if checkpoint is None or state.get("checkpoint_path") != str(resolved_checkpoint_path):
            checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu", weights_only=False)
            state["checkpoint"] = checkpoint
            state["checkpoint_path"] = str(resolved_checkpoint_path)

        print("Building model and dataloader for config...")
        model, dataset, _, device, _ = build_model_and_data(cfg, checkpoint=checkpoint, reuse_dataset=reuse_dataset)
        state["dataset"] = dataset

        def log_eval(msg: str) -> None:
            print(msg)
            with eval_log_path.open("a") as handle:
                handle.write(msg + "\n")

        color_mappings_by_split: Dict[str, Dict[str, List[List[int]]]] = {}
        dihedral_by_split: Dict[str, bool] = {}

        if use_aug:
            max_augments = int(getattr(cfg, "max_augments", 0) or 0)
            augmentor = getattr(dataset, "augmentor", None)
            if augmentor is None or getattr(augmentor, "max_augments", None) != max_augments:
                enable_color = bool(getattr(cfg, "enable_color_aug", False))
                enable_dihedral = bool(getattr(cfg, "enable_dihedral_aug", False))
                color_apply_to_test = bool(getattr(cfg, "color_apply_to_test", False))
                dihedral_apply_to_test = bool(getattr(cfg, "dihedral_apply_to_test", False))

                augmentor = build_augmentor(
                    dataset.examples, dataset.task_input_colors, max_augments=max_augments,
                    enable_color=enable_color, enable_dihedral=enable_dihedral, seed=int(cfg.seed),
                    color_apply_to_test_split=color_apply_to_test, dihedral_apply_to_test_split=dihedral_apply_to_test,
                )
                dataset.augmentor = augmentor

            evaluation: Dict[str, Dict[str, object]] = {}
            for split in splits:
                split_results, color_maps, dihedral_augmented = run_split_inference(
                    model=model, dataset=dataset, split=split, device=device, augmentor=augmentor,
                    batch_size=eval_batch_size, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, task_ids=task_ids,
                    include_targets=include_targets, temperature=getattr(cfg, "inference_temperature", None),
                    top_k=getattr(cfg, "inference_top_k", None),
                )
                summary = summarize_split_results(split_results)
                evaluation[split] = {"results": split_results, "summary": summary}
                color_mappings_by_split[split] = color_maps
                dihedral_by_split[split] = dihedral_augmented
            evaluation["_aug"] = {"color_mappings_by_split": color_mappings_by_split, "dihedral_augmented_by_split": dihedral_by_split}
        else:
            evaluation: Dict[str, Dict[str, object]] = {}
            for split in splits:
                split_results, _, _ = run_split_inference(
                    model=model, dataset=dataset, split=split, device=device, batch_size=eval_batch_size,
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS, task_ids=task_ids, include_targets=include_targets,
                    temperature=getattr(cfg, "inference_temperature", None), top_k=getattr(cfg, "inference_top_k", None),
                )
                summary = summarize_split_results(split_results)
                evaluation[split] = {"results": split_results, "summary": summary}

        epochs = getattr(cfg, "epochs", None)
        epoch_label = f"{epochs}ep" if epochs is not None else "eval"
        label = "augments" if use_aug else "no-aug"
        log_eval(f"\n-- {epoch_label} {max_eval_augments}{label} --\n")
        dihedral_augmented = dihedral_by_split.get("test", False) if use_aug else False

        for split in splits:
            summary = evaluation.get(split, {}).get("summary", {})
            total = summary.get("total_sequences", 0)
            shape_ok = summary.get("num_shape_correct", 0)
            fully_correct = summary.get("num_fully_correct", 0)
            avg_pixel_acc = summary.get("avg_pixel_accuracy", 0.0)
            log_eval(f"Split: {split} | Seq: {total} | Shape OK: {shape_ok} | Fully Correct: {fully_correct} | Pixel Acc: {avg_pixel_acc:.4f}")

            if log_correct_grids and fully_correct > 0:
                log_eval(f"  [Correct Grids Details for {split}]")
                is_dihedral_split = dihedral_by_split.get(split, dihedral_augmented) if use_aug else False
                correct_results = summary.get("fully_correct_results", [])
                for res in correct_results:
                    raw_idx = res.get("pair_index", 0)
                    if is_dihedral_split:
                        pair_id = raw_idx // 8
                        dihedral_id = raw_idx % 8
                    else:
                        pair_id = raw_idx
                        dihedral_id = 0
                    color_id = res.get("color_permutation_index", 0)
                    grid = res.get("output_grid", [])
                    log_eval(f"    T:{res.get('task_id')} | Pair:{pair_id} | Dihedral:{dihedral_id} | Color:{color_id} -> {grid}")

        print(f"Running AAIVR for {run_name}...")
        if hasattr(sys.stdout, "log"):
            sys.stdout = sys.stdout.terminal
        sys.stdout = TeeLogger(aaivr_log_path)

        try:
            test_results = evaluation.get("test", {}).get("results", [])
            aaivr_results: List[AAIVRSelection] = []
            if test_results:
                aaivr_color_mappings = color_mappings_by_split.get("test") if use_aug else None
                aaivr_results = run_aaivr_on_results(
                    test_results, is_dihedral_augmented=dihedral_augmented, color_mappings_by_task=aaivr_color_mappings,
                )
            else:
                print("No test results for AAIVR.")

            summarize_aaivr_pass_at_k(aaivr_results)
            arc_score, max_score, pct = _compute_arc_style_score(aaivr_results)
            print(f"Official ARC style scoring: {arc_score:.2f}/{max_score} ({pct:.2f}%)")
        finally:
            if hasattr(sys.stdout, "terminal"):
                sys.stdout.close()
                sys.stdout = sys.stdout.terminal

        print(f"Generating submission.json for {run_name}...")
        submission_data = _build_submission_from_aaivr(aaivr_results)
        with submission_path.open("w") as handle:
            json.dump(submission_data, handle)

        t_duration = perf_counter() - t_start
        print(f"Finished {run_name}. Submission saved to {submission_path}")
        print(f"Run {run_name} took {t_duration:.2f}s")
        with timing_path.open("a") as handle:
            handle.write(f"Evaluation {run_name}: {t_duration:.4f} s\n")
        results.append((run_name, evaluation, submission_path))

    cfg.checkpoint_path = prev_checkpoint
    cfg.data_path = prev_data_path
    cfg.max_augments = prev_aug

    print("\nAll evaluation runs completed.")
    return results
