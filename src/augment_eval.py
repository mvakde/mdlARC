from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

import inference
from augment import Augmentor, Augments
from tinytransformer import TinyTransformer
from utils import VOCAB_SIZE


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


def _allowed_tuple_indices(
    augments: Augments, *, allow_color: bool, allow_dihedral: bool
) -> List[int]:
    indices: List[int] = []
    for idx, (d_idx, c_idx) in enumerate(
        zip(augments.dihedral_indices, augments.color_map_indices)
    ):
        if not allow_color and c_idx != 0:
            continue
        if not allow_dihedral and d_idx != 0:
            continue
        indices.append(idx)
    if not indices:
        indices = [augments.identity_tuple_index]
    return indices


def _build_color_mappings_by_task(
    examples: Sequence[object],
    augmentor: Augmentor,
    split: str,
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
        allowed = _allowed_tuple_indices(
            augments, allow_color=allow_color, allow_dihedral=allow_dihedral
        )
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
def run_split_inference_augmented(
    model: TinyTransformer,
    dataset,
    split: str,
    device: torch.device,
    *,
    augmentor: Augmentor,
    batch_size: int = 16,
    max_new_tokens: int = inference.DEFAULT_MAX_NEW_TOKENS,
    task_ids: Optional[Sequence[str]] = None,
    pair_index: Optional[int] = None,
    include_targets: bool = True,
    temperature: Optional[float] = None,
    top_k: Optional[int] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, List[List[int]]], bool]:
    solutions = inference._load_solutions_for_dataset(dataset) if include_targets else None
    examples = inference._gather_examples_for_split(
        dataset,
        split=split,
        task_ids=task_ids,
        pair_index=pair_index,
        require_outputs=include_targets,
        solutions=solutions,
    )
    if not examples:
        return [], {}, False

    allow_color = _split_allows_color(augmentor, split)
    allow_dihedral = _split_allows_dihedral(augmentor, split)

    color_mappings_by_task, mapping_index_by_task = _build_color_mappings_by_task(
        examples, augmentor, split
    )

    work_items = []
    dihedral_augmented = False
    for ex in examples:
        task_id = getattr(ex, "task_id", None)
        if task_id is None:
            continue
        key = (task_id, getattr(ex, "split", None), getattr(ex, "pair_index", None))
        augments = augmentor.augments_by_key.get(key)
        if augments is None:
            continue
        allowed = _allowed_tuple_indices(
            augments, allow_color=allow_color, allow_dihedral=allow_dihedral
        )
        for idx in allowed:
            dihedral_idx = int(augments.dihedral_indices[idx])
            color_map_idx = int(augments.color_map_indices[idx])
            mapping = augments.color_maps[color_map_idx].tolist()
            mapping_key = tuple(mapping)
            mapping_idx = mapping_index_by_task[task_id].get(mapping_key, 0)
            seq_len = inference._sequence_length_for_example(ex, dihedral_idx)
            base_pair_index = getattr(ex, "pair_index", None)
            work_items.append(
                (
                    ex,
                    dihedral_idx,
                    dihedral_idx,
                    mapping_idx,
                    mapping,
                    base_pair_index,
                    seq_len,
                )
            )
            if dihedral_idx != 0:
                dihedral_augmented = True

    work_items.sort(key=lambda item: item[-1], reverse=True)

    all_results: List[Dict[str, object]] = []

    pack_dihedral = allow_dihedral and dihedral_augmented

    for start in range(0, len(work_items), batch_size):
        chunk = work_items[start : start + batch_size]

        batch_examples = [item[0] for item in chunk]
        batch_dihedral_indices = [item[1] for item in chunk]
        batch_transform_indices = [item[2] for item in chunk]
        batch_c_indices = [item[3] for item in chunk]
        batch_mappings = [item[4] for item in chunk]
        batch_pair_indices = []
        for item in chunk:
            base_pair_index = item[5]
            if pack_dihedral and base_pair_index is not None:
                batch_pair_indices.append(int(base_pair_index) * 8 + item[1])
            else:
                batch_pair_indices.append(base_pair_index)

        prompts, example_ids, metadata, cached_positions, target_output_tokens = (
            inference._prepare_examples_for_inference(
                batch_examples,
                include_targets=include_targets,
                solutions=solutions,
                color_mappings=batch_mappings,
                color_apply_fn=None,
                dihedral_transform_indices=batch_transform_indices,
                pair_indices=batch_pair_indices,
            )
        )

        batch_results = inference._run_generation_batch(
            model=model,
            prompts=prompts,
            example_ids=example_ids,
            metadata=metadata,
            cached_positions=cached_positions,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            target_output_tokens=target_output_tokens if include_targets else None,
        )

        print(
            f"[{split}] Finished batch {start // batch_size + 1} / {(len(work_items) + batch_size - 1) // batch_size}"
        )

        for res, c_idx, d_idx in zip(
            batch_results, batch_c_indices, batch_dihedral_indices
        ):
            res["color_permutation_index"] = c_idx
            res["dihedral_index"] = d_idx
            all_results.append(res)

    return all_results, color_mappings_by_task, dihedral_augmented
