#!/usr/bin/env python3
"""Augment challenges.json with dihedral (rotations + reflections) transforms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

Grid = List[List[int]]
Pair = Dict[str, Grid]
Transform = Callable[[Grid], Grid]


def _copy_grid(grid: Grid) -> Grid:
    return [list(row) for row in grid]


def _rotate90(grid: Grid) -> Grid:
    if not grid:
        return []
    return [list(row) for row in zip(*grid[::-1])]


def _rotate180(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in reversed(grid)]


def _rotate270(grid: Grid) -> Grid:
    if not grid:
        return []
    return [list(row) for row in zip(*grid)][::-1]


def _flip_horizontal(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in grid]


def _flip_vertical(grid: Grid) -> Grid:
    return [list(row) for row in reversed(grid)]


def _flip_main_diagonal(grid: Grid) -> Grid:
    if not grid:
        return []
    return [list(row) for row in zip(*grid)]


def _flip_anti_diagonal(grid: Grid) -> Grid:
    return _flip_vertical(_rotate90(grid))


# Order matters to keep deterministic augmentation ordering (identity first).
TRANSFORMS: List[Tuple[str, Transform]] = [
    ("identity", _copy_grid),
    ("rot90", _rotate90),
    ("rot180", _rotate180),
    ("rot270", _rotate270),
    ("flip_horizontal", _flip_horizontal),
    ("flip_vertical", _flip_vertical),
    ("flip_main_diagonal", _flip_main_diagonal),
    ("flip_anti_diagonal", _flip_anti_diagonal),
]


def _load_json(path: Path) -> Dict[str, dict]:
    return json.loads(path.read_text())


def _augment_pairs(pairs: List[Pair]) -> List[Pair]:
    augmented: List[Pair] = []
    for pair in pairs:
        input_grid = pair["input"]
        output_grid = pair.get("output")
        for _, transform in TRANSFORMS:
            new_pair: Pair = {"input": transform(input_grid)}
            if output_grid is not None:
                new_pair["output"] = transform(output_grid)
            augmented.append(new_pair)
    return augmented


def augment_dataset(challenges: Dict[str, dict]) -> Dict[str, dict]:
    augmented: Dict[str, dict] = {}
    for task_id, payload in challenges.items():
        new_payload = dict(payload)
        if "train" in payload:
            new_payload["train"] = _augment_pairs(list(payload.get("train", [])))
        if "test" in payload:
            new_payload["test"] = _augment_pairs(list(payload.get("test", [])))
        augmented[task_id] = new_payload
    return augmented


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Augment challenges.json with dihedral transforms."
    )
    parser.add_argument(
        "--input",
        default="assets/challenges.json",
        help="Path to the source challenges.json file.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path (defaults to challenges_dihedral_both.json next to input). For now, you need dihedral_both in the name because submission construction depends on it (I am stupid, will fix)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Challenges file not found: {input_path}")
    output_path = (
        Path(args.output)
        if args.output is not None
        else input_path.with_name("challenges_dihedral_both.json")
    )

    challenges = _load_json(input_path)
    augmented = augment_dataset(challenges)
    output_path.write_text(json.dumps(augmented, indent=2))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
