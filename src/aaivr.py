from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from utils import (
    apply_color_permutation_to_grid,
    extract_task_input_colors,
    generate_color_mapping_tensors,
    generate_task_color_mappings,
    load_challenges,
    plot_grids,
    split_grids_from_tokens,
)

_DIHEDRAL_TRANSFORM_NAMES = [
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "flip_horizontal",
    "flip_vertical",
    "flip_main_diagonal",
    "flip_anti_diagonal",
]


def _dihedral_copy(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(row) for row in grid]


def _dihedral_rot90(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    if not grid:
        return []
    return [list(row) for row in zip(*grid[::-1])]


def _dihedral_rot180(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(reversed(row)) for row in reversed(grid)]


def _dihedral_rot270(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    if not grid:
        return []
    return [list(row) for row in zip(*grid)][::-1]


def _dihedral_flip_horizontal(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(reversed(row)) for row in grid]


def _dihedral_flip_vertical(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    return [list(row) for row in reversed(grid)]


def _dihedral_flip_main_diagonal(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    if not grid:
        return []
    return [list(row) for row in zip(*grid)]


def _dihedral_flip_anti_diagonal(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    return _dihedral_flip_vertical(_dihedral_rot90(grid))


_DIHEDRAL_TRANSFORMS = {
    "identity": _dihedral_copy,
    "rot90": _dihedral_rot90,
    "rot180": _dihedral_rot180,
    "rot270": _dihedral_rot270,
    "flip_horizontal": _dihedral_flip_horizontal,
    "flip_vertical": _dihedral_flip_vertical,
    "flip_main_diagonal": _dihedral_flip_main_diagonal,
    "flip_anti_diagonal": _dihedral_flip_anti_diagonal,
}

_DIHEDRAL_INVERSES = {
    "identity": "identity",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "flip_horizontal": "flip_horizontal",
    "flip_vertical": "flip_vertical",
    "flip_main_diagonal": "flip_main_diagonal",
    "flip_anti_diagonal": "flip_anti_diagonal",
}


def is_rectangular_grid(grid: Sequence[Sequence[int]]) -> bool:
    """Return True if all rows have the same non-zero length."""
    if not grid:
        return False
    first_row_len = len(grid[0])
    if first_row_len == 0:
        return False
    return all(len(row) == first_row_len for row in grid)


def apply_inverse_dihedral_transform(
    grid: Sequence[Sequence[int]], transform_index: int
) -> List[List[int]]:
    """Undo a dihedral transform using the known augmentation index (mod 8)."""
    if transform_index < 0:
        raise ValueError("transform_index must be non-negative.")
    transform_name = _DIHEDRAL_TRANSFORM_NAMES[transform_index % 8]
    inverse_name = _DIHEDRAL_INVERSES[transform_name]
    return _DIHEDRAL_TRANSFORMS[inverse_name](grid)


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
    color_aug_seed: Optional[int] = None,
    max_color_augments: int = 0,
) -> List[AAIVRSelection]:
    """Aggregate augmented predictions via AAIVR voting. (automated augmentation inverse)

    The function assumes pair_index encodes augmentation order (mod 8) if is_dihedral_augmented is True.
    It now also handles inverting color permutations if color info is provided.
    """
    rng = rng if rng is not None else random
    case_map: Dict[Tuple[str, int], Dict[str, object]] = {}

    # 1. Pre-calculate Inverse Color Mappings
    inverse_color_mappings_by_task: Dict[str, List[List[int]]] = {}
    inverse_color_mappings_global: List[List[int]] = []
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
    elif max_color_augments > 0:
        seed = color_aug_seed if color_aug_seed is not None else 42
        forward_tensors = generate_color_mapping_tensors(max_color_augments, seed)
        for fwd in forward_tensors:
            inv = torch.zeros_like(fwd)
            inv[fwd] = torch.arange(len(fwd), dtype=torch.long, device=fwd.device)
            inverse_color_mappings_global.append(inv.tolist())

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
                if color_mappings_by_task is not None:
                    inv_list = inverse_color_mappings_by_task.get(task_id, [])
                else:
                    inv_list = inverse_color_mappings_global
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
            if color_mappings_by_task is not None:
                inv_list = inverse_color_mappings_by_task.get(task_id, [])
            else:
                inv_list = inverse_color_mappings_global
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


def _invert_color_mappings(
    mappings: Sequence[Sequence[int]],
) -> List[List[int]]:
    inverse_list: List[List[int]] = []
    for mapping in mappings:
        fwd = (
            mapping
            if isinstance(mapping, torch.Tensor)
            else torch.tensor(mapping, dtype=torch.long)
        )
        inv = torch.zeros_like(fwd)
        inv[fwd] = torch.arange(len(fwd), dtype=torch.long, device=fwd.device)
        inverse_list.append(inv.tolist())
    return inverse_list


def _task_order_from_results(results: Sequence[Dict[str, object]]) -> List[str]:
    order: List[str] = []
    seen = set()
    for res in results:
        task_id = res.get("task_id")
        if task_id is None or task_id in seen:
            continue
        order.append(task_id)
        seen.add(task_id)
    return order


def _resolve_task_id(
    results: Sequence[Dict[str, object]],
    task_id: Optional[str],
    task_index: Optional[int],
) -> Optional[str]:
    if task_id:
        return task_id
    if task_index is None:
        return None
    task_order = _task_order_from_results(results)
    if task_index < 0 or task_index >= len(task_order):
        return None
    return task_order[task_index]


def _load_task_color_mappings(
    task_id: str, dataset_path: Path, max_color_augments: int, seed: int
) -> List[torch.Tensor]:
    challenges = load_challenges(dataset_path)
    if task_id not in challenges:
        raise ValueError(f"Task '{task_id}' not found in {dataset_path}")
    colors = extract_task_input_colors(challenges[task_id])
    mappings_by_task = generate_task_color_mappings(
        {task_id: colors}, max_color_augments, seed
    )
    return list(mappings_by_task.get(task_id, []))


def visualize_aaivr_flow(
    results: Sequence[Dict[str, object]],
    dataset_path: Optional[Path],
    input_index: int,
    *,
    task_id: Optional[str] = None,
    task_index: Optional[int] = None,
    is_dihedral_augmented: bool = False,
    max_color_augments: int = 0,
    color_aug_seed: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> None:
    """Visualize augmented input/output pairs grouped by AAIVR-normalized outputs."""
    if not results:
        print("No evaluation results provided.")
        return

    resolved_task_id = _resolve_task_id(results, task_id, task_index)
    if resolved_task_id is None:
        task_count = len(_task_order_from_results(results))
        print(
            "Could not resolve task. Provide task_id or a valid task_index "
            f"(0-{max(0, task_count - 1)})."
        )
        return

    inverse_color_mappings: List[List[int]] = []
    if max_color_augments > 0:
        if dataset_path is None:
            print("dataset_path is required when max_color_augments > 0.")
            return
        seed = color_aug_seed if color_aug_seed is not None else 42
        if color_aug_seed is None:
            print("Warning: color_aug_seed not provided; defaulting to 42.")
        mappings = _load_task_color_mappings(
            resolved_task_id, Path(dataset_path), max_color_augments, seed
        )
        inverse_color_mappings = _invert_color_mappings(mappings)

    entries: List[Dict[str, object]] = []
    for res in results:
        if res.get("task_id") != resolved_task_id:
            continue
        raw_pair_index = res.get("pair_index")
        if raw_pair_index is None:
            continue
        raw_pair_index = int(raw_pair_index)
        if is_dihedral_augmented:
            base_pair_index = raw_pair_index // 8
            dihedral_index = raw_pair_index % 8
        else:
            base_pair_index = raw_pair_index
            dihedral_index = 0
        if base_pair_index != int(input_index):
            continue

        color_index = int(res.get("color_permutation_index") or 0)
        prompt_tokens = res.get("prompt_tokens") or []
        input_grids = split_grids_from_tokens(prompt_tokens)
        input_grid = input_grids[0] if input_grids else []
        output_grid = res.get("output_grid") or []

        normalized_key = None
        if is_rectangular_grid(output_grid):
            try:
                normalized = apply_inverse_dihedral_transform(
                    output_grid, dihedral_index
                )
                if (
                    inverse_color_mappings
                    and color_index > 0
                    and color_index < len(inverse_color_mappings)
                ):
                    normalized = apply_color_permutation_to_grid(
                        normalized, inverse_color_mappings[color_index]
                    )
                if is_rectangular_grid(normalized):
                    normalized_key = _grid_to_tuple(normalized)
            except Exception:
                normalized_key = None

        entries.append(
            {
                "input_grid": input_grid,
                "output_grid": output_grid,
                "raw_pair_index": raw_pair_index,
                "base_pair_index": base_pair_index,
                "dihedral_index": dihedral_index,
                "color_index": color_index,
                "normalized_key": normalized_key,
            }
        )

    if not entries:
        print(
            f"No results found for task {resolved_task_id} input_index {input_index}."
        )
        return

    groups: Dict[Tuple[Tuple[int, ...], ...], List[Dict[str, object]]] = {}
    invalid_entries: List[Dict[str, object]] = []
    for entry in entries:
        key = entry.get("normalized_key")
        if key is None:
            invalid_entries.append(entry)
            continue
        groups.setdefault(key, []).append(entry)

    items = list(groups.items())
    if rng is None:
        rng = random
    if items:
        rng.shuffle(items)
        items.sort(key=lambda pair: len(pair[1]), reverse=True)

    print(
        f"AAIVR flow for task {resolved_task_id} | input_index {input_index} | total {len(entries)}"
    )
    print(f"Grouped outputs: {len(items)} normalized, {len(invalid_entries)} invalid")

    rank = 1
    for _, group_entries in items:
        print(f"\nPreference order: {rank} | Count: {len(group_entries)}")
        for entry in group_entries:
            dihedral_label = _DIHEDRAL_TRANSFORM_NAMES[
                entry["dihedral_index"] % 8
            ]
            title = (
                f"{resolved_task_id} | input {entry['base_pair_index']} | "
                f"pair {entry['raw_pair_index']} | dihedral {entry['dihedral_index']}:{dihedral_label} | "
                f"color {entry['color_index']}"
            )
            try:
                plot_grids([entry["input_grid"], entry["output_grid"]], title=title)
            except Exception as exc:
                print(f"Skipping plot for pair {entry['raw_pair_index']}: {exc}")
        rank += 1

    if invalid_entries:
        print(
            f"\nPreference order: {rank} | Count: {len(invalid_entries)} | Invalid outputs"
        )
        for entry in invalid_entries:
            dihedral_label = _DIHEDRAL_TRANSFORM_NAMES[
                entry["dihedral_index"] % 8
            ]
            title = (
                f"{resolved_task_id} | input {entry['base_pair_index']} | "
                f"pair {entry['raw_pair_index']} | dihedral {entry['dihedral_index']}:{dihedral_label} | "
                f"color {entry['color_index']}"
            )
            try:
                plot_grids([entry["input_grid"], entry["output_grid"]], title=title)
            except Exception as exc:
                print(f"Skipping plot for pair {entry['raw_pair_index']}: {exc}")
