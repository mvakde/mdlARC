from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from utils import (
    apply_color_permutation_to_grid,
    apply_inverse_dihedral_transform,
    is_rectangular_grid,
    split_grids_from_tokens,
)

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
    """Aggregate augmented predictions via AAIVR voting. (automated augmentation inverse)

    The function assumes pair_index encodes augmentation order (mod 8) if is_dihedral_augmented is True.
    It now also handles inverting color permutations if color info is provided.
    """
    rng = rng if rng is not None else random
    case_map: Dict[Tuple[str, int], Dict[str, object]] = {}

    # 1. Pre-calculate Inverse Color Mappings
    inverse_color_mappings_by_task: Dict[str, List[List[int]]] = {}
    if color_mappings_by_task is not None:
        for task_id, mappings in color_mappings_by_task.items():
            inv_list: List[List[int]] = []
            for mapping in mappings:
                fwd = (
                    mapping
                    if isinstance(mapping, torch.Tensor)
                    else torch.tensor(mapping, dtype=torch.long)
                )
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
            # Dataset has 8 copies per pair encoded in index
            base_pair_index = int(pair_index) // 8
            transform_index = int(pair_index) % 8
        else:
            # Standard dataset: index is just the pair index
            base_pair_index = int(pair_index)
            transform_index = 0

        color_idx = res.get("color_permutation_index", 0)

        predicted_grid = res.get("output_grid", [])
        prompt_tokens = res.get("prompt_tokens", [])
        input_grids = split_grids_from_tokens(prompt_tokens)
        input_grid = input_grids[0] if input_grids else []

        key = (task_id, base_pair_index)
        if key not in case_map:
            case_map[key] = {
                "counts": {},
                "generated": 0,
                "valid": 0,
                "dropped_rect": 0,
                "dropped_input": 0,
                "target_grid": None,
            }
        stats = case_map[key]
        stats["generated"] += 1

        # 2. Normalize Target Grid (Geometric Inverse + Color Inverse)
        target_grid = res.get("target_grid", [])
        if stats["target_grid"] is None and is_rectangular_grid(target_grid):
            try:
                # Geometric Inverse
                norm_target = apply_inverse_dihedral_transform(
                    target_grid, transform_index
                )
                # Color Inverse
                inv_list = inverse_color_mappings_by_task.get(task_id, [])
                if inv_list and color_idx > 0 and color_idx < len(inv_list):
                    norm_target = apply_color_permutation_to_grid(
                        norm_target, inv_list[color_idx]
                    )

                if is_rectangular_grid(norm_target):
                    stats["target_grid"] = norm_target
            except Exception:
                pass

        # 3. Validation Checks
        if not is_rectangular_grid(predicted_grid):
            stats["dropped_rect"] += 1
            continue
        if discard_input_copies and input_grid and predicted_grid == input_grid:
            stats["dropped_input"] += 1
            continue

        # 4. Normalize Predicted Grid (Geometric Inverse + Color Inverse)
        try:
            # Geometric Inverse
            normalized_grid = apply_inverse_dihedral_transform(
                predicted_grid, transform_index
            )

            # Color Inverse
            inv_list = inverse_color_mappings_by_task.get(task_id, [])
            if inv_list and color_idx > 0 and color_idx < len(inv_list):
                normalized_grid = apply_color_permutation_to_grid(
                    normalized_grid, inv_list[color_idx]
                )
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
            rng.shuffle(items)  # tie-break randomly before sorting by count
            items.sort(key=lambda pair: pair[1], reverse=True)
        ranked_candidates = [
            {"grid": _tuple_to_grid(grid_key), "count": count}
            for grid_key, count in items
        ]
        selected_outputs = [entry["grid"] for entry in ranked_candidates[:top_k]]

        target_grid = stats.get("target_grid")
        pass_at_k = None
        if target_grid is not None:
            pass_at_k = any(grid == target_grid for grid in selected_outputs)

        selections.append(
            AAIVRSelection(
                task_id=task_id,
                original_pair_index=base_idx,
                selected_outputs=selected_outputs,
                ranked_candidates=ranked_candidates,
                num_generated=stats["generated"],
                num_valid=stats["valid"],
                discarded_non_rectangular=stats["dropped_rect"],
                discarded_input_copies=stats["dropped_input"],
                target_grid=target_grid,
                pass_at_k=pass_at_k,
            )
        )

    return selections


def summarize_aaivr_pass_at_k(selections: Sequence[AAIVRSelection]) -> Dict[str, int]:
    """Return counts for how many tasks have ALL their pairs in top-k."""
    # Group by task_id
    tasks: Dict[str, List[AAIVRSelection]] = {}
    for sel in selections:
        tasks.setdefault(sel.task_id, []).append(sel)

    total_tasks = len(tasks)
    solved_tasks = 0
    failures = []

    for task_id, pairs in tasks.items():
        # A task is solved if ALL its pairs are solved (pass_at_k is True)
        is_solved = True
        pair_failures = []

        for p in pairs:
            if p.pass_at_k is None:
                # Target missing or logic failed to find it
                is_solved = False
                pair_failures.append(
                    f"Pair {p.original_pair_index}: Target missing/unknown"
                )
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

    # Print details as requested
    if failures:
        print(f"\nAAIVR Failures ({len(failures)}/{total_tasks} tasks):")
        for f in failures:
            print(f"  - {f}")

    # Return structure compatible with 'hits'/'evaluated' expectations
    # Evaluated now refers to Tasks, Hits to Solved Tasks
    return {"evaluated": total_tasks, "hits": solved_tasks}
