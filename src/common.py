"""Common utilities shared across build, train, and evaluate modules.

Contains: vocabulary definitions, tokenization, grid transforms, dataset classes,
dataloader creation, 3D position computation, and RNG/device handling.
"""

import functools
import hashlib
import json
import math
import os
import random
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from numba import njit
from torch.utils.data import DataLoader, Dataset, Sampler
import distributed_training as distributed
from demo_schedule import (
    MAX_CONTEXT_DEMOS, balanced_demonstrations, half_or_more_demonstrations, training_demonstrations,
)
from pair_identity import balanced_leave_one_out, balanced_pair_vertices, leave_one_out, schedule_seed


# =============================================================================
# Vocabulary Definitions
# =============================================================================

VOCAB_SIZE = 14

SPECIAL_TOKENS = ["<start>", "<next_line>", "<input_output_separator>", "<end_pair>"]
TOKEN_TO_ID: Dict[str, int] = {str(i): i for i in range(10)}
for offset, token in enumerate(SPECIAL_TOKENS, start=10):
    TOKEN_TO_ID[token] = offset

ID_TO_TOKEN = {idx: token for token, idx in TOKEN_TO_ID.items()}

START_TOKEN_ID = TOKEN_TO_ID["<start>"]
NEXT_LINE_TOKEN_ID = TOKEN_TO_ID["<next_line>"]
IO_SEPARATOR_TOKEN_ID = TOKEN_TO_ID["<input_output_separator>"]
END_PAIR_TOKEN_ID = TOKEN_TO_ID["<end_pair>"]

DEMO_PAIRS = 2
MAX_PAIR_LEN = 1861
# One start token followed by three input/output pairs, each at most 30x30.
MAX_SEQ_LEN = 1 + (DEMO_PAIRS + 1) * (MAX_PAIR_LEN - 1)
SEQUENCE_FORMAT = "two_demo_bidirectional_v1"
GROUP_SEQUENCE_FORMAT = "all_demo_pair_group_target_roles_two_lookup_v1"
IGNORE_INDEX = -100

AUGMENTATION_SAMPLING_MODES = ("full_task_prefix", "single_grid_prefix", "context_palette")


def normalize_attention_pattern(value: Optional[str]) -> str:
    pattern = "dense" if value is None else value
    if pattern not in ("dense", "pair_group"):
        raise ValueError("attention_pattern must be 'dense' or 'pair_group'.")
    return pattern


def sequence_format_for_attention(pattern: str) -> str:
    return GROUP_SEQUENCE_FORMAT if normalize_attention_pattern(pattern) == "pair_group" else SEQUENCE_FORMAT


def normalize_augmentation_sampling_mode(value: Optional[str]) -> str:
    mode = "full_task_prefix" if value is None else value
    if mode not in AUGMENTATION_SAMPLING_MODES:
        raise ValueError(
            "augmentation_sampling_mode must be 'full_task_prefix', 'single_grid_prefix' or 'context_palette'."
        )
    return mode


# =============================================================================
# Color Extraction
# =============================================================================

def _extract_pairs_colors(
    pairs: Sequence[Dict[str, object]], key: str
) -> Set[int]:
    colors: Set[int] = set()
    for pair in pairs:
        grid = pair.get(key) or []
        for row in grid:
            for val in row:
                val_i = int(val)
                if 0 <= val_i <= 9:
                    colors.add(val_i)
    return colors


def extract_task_colors(task: Dict[str, object]) -> List[int]:
    """Return colors available from demonstrations and test inputs."""
    train_pairs = task.get("train", [])
    test_pairs = task.get("test", [])
    return sorted(
        _extract_pairs_colors(train_pairs, "input")
        | _extract_pairs_colors(train_pairs, "output")
        | _extract_pairs_colors(test_pairs, "input")
    )


def extract_pairs_colors(pairs: Sequence[Dict[str, object]]) -> List[int]:
    """Return every color appearing in a list of input/output pairs."""
    return sorted(_extract_pairs_colors(pairs, "input") | _extract_pairs_colors(pairs, "output"))


def extract_task_output_only_colors(task: Dict[str, object]) -> List[int]:
    """Protect demonstration output colors absent from every known task input.

    Test outputs are deliberately excluded, even when a file contains them.
    """
    train = task.get("train", [])
    inputs = _extract_pairs_colors(train, "input") | _extract_pairs_colors(task.get("test", []), "input")
    return sorted(_extract_pairs_colors(train, "output") - inputs)


def color_augmentation_domain(
    colors: Sequence[int], protected_colors: Sequence[int], mode: Optional[str]
) -> Tuple[List[int], Optional[List[int]]]:
    if normalize_augmentation_sampling_mode(mode) == "context_palette":
        raise ValueError("context_palette requires the selected visible context")
    if normalize_augmentation_sampling_mode(mode) == "full_task_prefix":
        return list(colors), None
    protected = set(protected_colors)
    return [color for color in colors if color not in protected], [
        color for color in range(10) if color not in protected
    ]


# =============================================================================
# Dihedral Transforms (8 geometric operations)
# =============================================================================

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


def apply_dihedral_transform(
    grid: Sequence[Sequence[int]], transform_index: int
) -> List[List[int]]:
    if transform_index < 0:
        raise ValueError("transform_index must be non-negative.")
    transform_name = _DIHEDRAL_TRANSFORM_NAMES[transform_index % 8]
    return _DIHEDRAL_TRANSFORMS[transform_name](grid)


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


def apply_inverse_dihedral_transform(
    grid: Sequence[Sequence[int]], transform_index: int
) -> List[List[int]]:
    """Undo a dihedral transform using the known augmentation index (mod 8)."""
    if transform_index < 0:
        raise ValueError("transform_index must be non-negative.")
    transform_name = _DIHEDRAL_TRANSFORM_NAMES[transform_index % 8]
    inverse_name = _DIHEDRAL_INVERSES[transform_name]
    return _DIHEDRAL_TRANSFORMS[inverse_name](grid)


# =============================================================================
# Scale (Block-Upscale) Augmentation
# =============================================================================
# Integer block-upscale ("scale") augmentation for training and decoding.
#
# Follows transductARC (K. Schweighofer, 2026, MIT; ``src/augment.py`` and ``src/eval.py``):
#
# * A spec ``(fr, fc)`` turns every cell into an ``fr x fc`` block; ``(1, 1)`` is identity.
#   Specs live in the grid's original orientation and compose as "scale, then dihedral",
#   so a dihedral that swaps rows and columns sees the axes swapped
#   (:func:`oriented_scale_factor`).
# * A pair's spec pool lists the specs that keep its input and output within 30x30, identity
#   first and grouped by level ``k = max(fr, fc)``, the square ``(k, k)`` before the
#   anisotropic ``(k, m), (m, k)`` for ``m < k``, truncated to ``max_specs`` (12 there)::
#
#       (1,1) (2,2) (2,1) (1,2) (3,3) (3,1) (1,3) (3,2) (2,3) (4,4) (4,1) (1,4) ...
#
#   His length cap holds two 30x30 grids, so it never removes a spec that keeps the grids
#   within 30x30; the pool here therefore applies only the 30x30 bound.
# * Training draws one spec uniformly from the pool of every pair in the sequence. His
#   single demo "pins" the scale to a spec both pairs offer; with several demos the
#   sequence offers the target pair's specs that every visible demo also offers
#   (:func:`sequence_scale_specs`). One spec scales every grid of the sequence.
# * Decoding bounds specs by the demonstrations, since the unknown test output must still
#   fit 30x30 (:func:`demo_bounded_scale_specs`). Candidates are mapped back by an exact
#   downscale that rejects grids whose size is not divisible by the factor or whose blocks
#   are not uniform (:func:`downscale_grid_exact`); his reference strides instead.
#
# Every choice is a pure function of seeds and visit numbers, so no shared RNG is consumed
# and the stream with scale augmentation off is unchanged.

SCALE_MAX_GRID_DIM = 30
DEFAULT_SCALE_MAX_SPECS = 12
# Demos shown with a ReARC training target (ARCExampleDataset.rearc_demonstration_indices).
DEFAULT_REARC_MAX_DEMOS = 4
IDENTITY_SCALE = (1, 1)
# rot90, rot270, flip_main_diagonal, flip_anti_diagonal in _DIHEDRAL_TRANSFORM_NAMES
# (the same order as transductARC): these map an R x C grid to C x R.
SCALE_TRANSPOSING_DIHEDRALS = frozenset({1, 3, 6, 7})

ScaleSpec = Tuple[int, int]


def normalize_scale(factor) -> ScaleSpec:
    """An int (isotropic) or a (rows, cols) pair as a pair of integers >= 1."""
    if isinstance(factor, (tuple, list)):
        fr, fc = int(factor[0]), int(factor[1])
    else:
        fr = fc = int(factor)
    if fr < 1 or fc < 1:
        raise ValueError(f"Scale factors must be >= 1, got {factor!r}")
    return fr, fc


def is_identity_scale(factor) -> bool:
    return factor is None or normalize_scale(factor) == IDENTITY_SCALE


def oriented_scale_factor(factor, dihedral_index: int) -> ScaleSpec:
    """The factor in the dihedral-transformed frame, where a transposing dihedral swaps axes."""
    fr, fc = normalize_scale(factor)
    return (fc, fr) if int(dihedral_index) % 8 in SCALE_TRANSPOSING_DIHEDRALS else (fr, fc)


def grid_dims(grid: Optional[Sequence[Sequence[int]]]) -> Tuple[int, int]:
    if not grid:
        return 0, 0
    return len(grid), max((len(row) for row in grid), default=0)


def upscale_grid(grid: Sequence[Sequence[int]], factor) -> List[List[int]]:
    """Each cell becomes an ``fr x fc`` block of its value."""
    fr, fc = normalize_scale(factor)
    if (fr, fc) == IDENTITY_SCALE:
        return [list(row) for row in grid]
    scaled: List[List[int]] = []
    for row in grid:
        wide = [int(value) for value in row for _ in range(fc)]
        scaled.extend(list(wide) for _ in range(fr))
    return scaled


def downscale_grid_exact(grid: Sequence[Sequence[int]], factor) -> Optional[List[List[int]]]:
    """Exact inverse of :func:`upscale_grid`, or None when ``grid`` is not an upscale.

    Rejects ragged or empty grids, sizes not divisible by the factor and blocks that are
    not one uniform colour.
    """
    fr, fc = normalize_scale(factor)
    if not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid):
        return None
    rows, cols = len(grid), len(grid[0])
    if (fr, fc) == IDENTITY_SCALE:
        return [list(row) for row in grid]
    if rows % fr or cols % fc:
        return None
    result: List[List[int]] = []
    for block_row in range(0, rows, fr):
        reference = grid[block_row]
        values = [int(reference[c]) for c in range(0, cols, fc)]
        for r in range(block_row, block_row + fr):
            row = grid[r]
            for index, start in enumerate(range(0, cols, fc)):
                if any(int(row[c]) != values[index] for c in range(start, start + fc)):
                    return None
        result.append(values)
    return result


@functools.lru_cache(maxsize=None)
def scale_specs_for_dims(rows: int, cols: int, out_rows: int = 0, out_cols: int = 0, *,
                         anisotropic: bool = True, max_specs: Optional[int] = None,
                         max_dim: int = SCALE_MAX_GRID_DIM) -> Tuple[ScaleSpec, ...]:
    """transductARC's ``available_scale_specs_for_dims`` (draw order, identity first).

    Caps are per axis over the input and output (``max_dim // extent``); since a
    transposing dihedral swaps both the grid and the factor, the caps hold in every
    orientation.
    """
    max_rows, max_cols = max(rows, out_rows), max(cols, out_cols)
    row_cap = max(1, max_dim // max_rows) if max_rows else 1
    col_cap = max(1, max_dim // max_cols) if max_cols else 1
    specs: List[ScaleSpec] = []
    for k in range(1, max(row_cap, col_cap) + 1):
        candidates: List[ScaleSpec] = [(k, k)]
        if anisotropic:
            for m in range(1, k):
                candidates.extend(((k, m), (m, k)))
        for fr, fc in candidates:
            if fr > row_cap or fc > col_cap:
                continue
            specs.append((fr, fc))
            if max_specs is not None and len(specs) >= max_specs:
                return tuple(specs)
    return tuple(specs)


def pair_scale_specs(pair, *, anisotropic: bool = True,
                     max_specs: Optional[int] = DEFAULT_SCALE_MAX_SPECS) -> Tuple[ScaleSpec, ...]:
    """One pair's pool from its input and (when present) output dimensions."""
    rows, cols = grid_dims(pair["input"])
    out_rows, out_cols = grid_dims(pair.get("output"))
    return scale_specs_for_dims(rows, cols, out_rows, out_cols,
                                anisotropic=anisotropic, max_specs=max_specs)


def sequence_scale_specs(target, demos: Iterable, *, anisotropic: bool = True,
                         max_specs: Optional[int] = DEFAULT_SCALE_MAX_SPECS) -> Tuple[ScaleSpec, ...]:
    """Specs of the target pair's pool that every visible demo's pool also offers."""
    specs = pair_scale_specs(target, anisotropic=anisotropic, max_specs=max_specs)
    for demo in demos:
        offered = set(pair_scale_specs(demo, anisotropic=anisotropic, max_specs=max_specs))
        specs = tuple(spec for spec in specs if spec in offered)
    return specs


def training_scale_seed(seed: int, task_id: str, split: str, pair_index: int, visit: int) -> int:
    return schedule_seed(seed, task_id, split, int(pair_index), "scale-augmentation-v1", int(visit))


def draw_scale(specs: Sequence[ScaleSpec], seed: int) -> ScaleSpec:
    """Uniform over ``specs`` (as transductARC's ``randrange``), from a private RNG."""
    if not specs:
        return IDENTITY_SCALE
    if len(specs) == 1:
        return tuple(specs[0])
    return tuple(specs[random.Random(int(seed)).randrange(len(specs))])


def demo_bounded_scale_specs(input_grid, demos: Iterable, *, anisotropic: bool = True,
                             max_specs: Optional[int] = DEFAULT_SCALE_MAX_SPECS) -> List[ScaleSpec]:
    """transductARC's ``demo_bounded_scale_specs`` without its (never binding) length cap.

    1. Project the test output size per axis from each demo's in/out size ratio (only
       integral projections within 1..30 count; the largest wins, a missing axis falls
       back to the input's) and keep specs that fit the input and that projection.
    2. Keep only specs every demonstration pair itself admits.
    """
    if max_specs is not None and max_specs <= 1:
        return [IDENTITY_SCALE]
    in_rows, in_cols = grid_dims(input_grid)
    demos = [demo for demo in demos
             if demo.get("input") and demo["input"][0] and demo.get("output") and demo["output"][0]]
    projected = [0, 0]
    for demo in demos:
        (demo_in_rows, demo_in_cols), (demo_out_rows, demo_out_cols) = grid_dims(demo["input"]), grid_dims(demo["output"])
        for axis, (demo_in, demo_out, test_in) in enumerate(((demo_in_rows, demo_out_rows, in_rows),
                                                              (demo_in_cols, demo_out_cols, in_cols))):
            expected = test_in * Fraction(demo_out, demo_in)
            if expected.denominator != 1 or not 1 <= expected <= SCALE_MAX_GRID_DIM:
                continue
            projected[axis] = max(projected[axis], int(expected))
    if any(projected):
        projected = [p or d for p, d in zip(projected, (in_rows, in_cols))]
    specs = list(scale_specs_for_dims(in_rows, in_cols, projected[0], projected[1], anisotropic=anisotropic))
    for demo in demos:
        admitted = set(pair_scale_specs(demo, anisotropic=anisotropic, max_specs=None))
        specs = [spec for spec in specs if spec in admitted]
    return specs[:max_specs] if max_specs is not None else specs


def inference_scale_schedule(specs: Sequence[ScaleSpec], count: int, seed: int) -> List[ScaleSpec]:
    """Balanced decode views: each spec once per shuffled cycle of ``len(specs)`` views."""
    specs = [tuple(spec) for spec in specs] or [IDENTITY_SCALE]
    schedule: List[ScaleSpec] = []
    cycle = 0
    while len(schedule) < int(count):
        order = list(range(len(specs)))
        random.Random(schedule_seed(seed, "scale-views-v1", cycle)).shuffle(order)
        schedule.extend(specs[index] for index in order)
        cycle += 1
    return schedule[:int(count)]


# =============================================================================
# Symbolic Candidate Guards (demo-derived output checks)
# =============================================================================
# Ported in logic from transductARC src/guard.py (MIT). Every rule is derived from the task's
# demonstrations alone and only ever rejects a candidate that breaks an invariant all demos share,
# e.g. a fixed output/input size ratio, an unchanged colour set, or output = a crop of the input.


def _grid_colors(grid: Sequence[Sequence[int]]) -> Set[int]:
    return {int(value) for row in grid for value in row}


def _foreground_counts(grid: Sequence[Sequence[int]]) -> Dict[int, int]:
    counts: Dict[int, int] = {}
    for row in grid:
        for value in row:
            if value != 0:
                counts[int(value)] = counts.get(int(value), 0) + 1
    return counts


def _is_subgrid(sub: Sequence[Sequence[int]], grid: Sequence[Sequence[int]]) -> bool:
    outer = np.asarray(grid, dtype=np.int16)
    inner = np.asarray(sub, dtype=np.int16)
    if (outer.ndim != 2 or inner.ndim != 2 or inner.shape[0] > outer.shape[0]
            or inner.shape[1] > outer.shape[1]):
        return False
    windows = np.lib.stride_tricks.sliding_window_view(outer, inner.shape)
    return bool((windows == inner).all(axis=(2, 3)).any())


def symbolic_guards(demos: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Invariants every demonstration (input -> output) shares.

    The rules follow the demo-derived candidate guards of transductARC (MIT licence).
    """
    guards: Dict[str, Any] = dict(forbid_empty=True, forbid_copy=True, row_ratio=None, col_ratio=None,
                                  color_set=False, color_counts=False, crop=False, single_color=False,
                                  dropped_color=None)
    pairs = [(demo["input"], demo["output"]) for demo in demos]
    if not pairs or any(not i or not i[0] or not o or not o[0] for i, o in pairs):
        return guards
    guards["forbid_empty"] = not any(all(v == 0 for row in o for v in row) for _, o in pairs)
    guards["forbid_copy"] = not any(i == o for i, o in pairs)
    row_ratios = {Fraction(len(o), len(i)) for i, o in pairs}
    col_ratios = {Fraction(len(o[0]), len(i[0])) for i, o in pairs}
    if len(row_ratios) == 1:
        guards["row_ratio"] = next(iter(row_ratios))
    if len(col_ratios) == 1:
        guards["col_ratio"] = next(iter(col_ratios))
    guards["color_set"] = all(_grid_colors(i) == _grid_colors(o) for i, o in pairs)
    guards["color_counts"] = all(_foreground_counts(i) == _foreground_counts(o) for i, o in pairs)
    guards["crop"] = all(_is_subgrid(o, i) for i, o in pairs)
    guards["single_color"] = all(len(_grid_colors(o) - {0}) == 1 for _, o in pairs)
    dropped = []
    for i, o in pairs:
        input_colors, output_colors = _grid_colors(i), _grid_colors(o)
        removed = input_colors - output_colors
        if len(removed) != 1 or not output_colors <= input_colors:
            dropped = []
            break
        dropped.append(next(iter(removed)))
    if dropped and len(set(dropped)) == 1:
        guards["dropped_color"] = dropped[0]
    return guards


def symbolic_guard_rejection(
    guards: Dict[str, Any], grid: Sequence[Sequence[int]], test_input: Optional[Sequence[Sequence[int]]],
) -> Optional[str]:
    """Name of the first invariant ``grid`` breaks for this test input, or None if it passes."""
    if not grid or not grid[0]:
        return "empty_grid"
    if guards["forbid_empty"] and not any(v for row in grid for v in row):
        return "all_background"
    if guards["forbid_copy"] and test_input and grid == test_input:
        return "input_copy"
    for size, input_size, ratio, name in (
        (len(grid), len(test_input) if test_input else 0, guards["row_ratio"], "row_ratio"),
        (len(grid[0]), len(test_input[0]) if test_input and test_input[0] else 0, guards["col_ratio"], "col_ratio"),
    ):
        if ratio is None or input_size <= 0:
            continue
        expected = input_size * ratio
        if expected.denominator != 1 or not 1 <= expected <= SCALE_MAX_GRID_DIM:
            continue
        if size != expected:
            return name
    if guards["single_color"] and len(_grid_colors(grid) - {0}) > 1:
        return "single_color"
    if test_input and test_input[0]:
        if guards["crop"] and not _is_subgrid(grid, test_input):
            return "crop"
        if guards["color_set"] and _grid_colors(grid) != _grid_colors(test_input):
            return "color_set"
        if guards["color_counts"] and _foreground_counts(grid) != _foreground_counts(test_input):
            return "color_counts"
        if guards["dropped_color"] is not None:
            expected_colors = _grid_colors(test_input) - {guards["dropped_color"]}
            colors = _grid_colors(grid)
            if colors - expected_colors or expected_colors - colors - {0}:
                return "dropped_color"
    return None


# =============================================================================
# Color Permutation
# =============================================================================

def apply_color_permutation_to_tokens(
    tokens: Sequence[int], mapping: Sequence[int]
) -> List[int]:
    """Apply a color permutation mapping to a token list."""
    return [int(mapping[tok] if 0 <= tok < len(mapping) else tok) for tok in tokens]


def apply_color_permutation_to_grid(
    grid: Sequence[Sequence[int]], mapping: Sequence[int]
) -> List[List[int]]:
    return [
        [int(mapping[val] if 0 <= val < len(mapping) else val) for val in row]
        for row in grid
    ]


def invert_color_mapping(
    mapping: Sequence[int], source_colors: Sequence[int]
) -> List[int]:
    """Invert an injective color mapping, leaving colors outside its image unchanged."""
    inverse = list(range(len(mapping)))
    for source in source_colors:
        source_i = int(source)
        if 0 <= source_i < len(mapping):
            target = int(mapping[source_i])
            if 0 <= target < len(inverse):
                inverse[target] = source_i
    return inverse


# =============================================================================
# Grid/Token Conversion
# =============================================================================

def _value_to_token_id(value: int) -> int:
    if value not in range(10):
        raise ValueError(f"Grid values must be digits in [0, 9], received {value}")
    return value


def grid_to_tokens(
    grid: Iterable[Iterable[int]], final_token: int = NEXT_LINE_TOKEN_ID
) -> List[int]:
    """Flatten a grid, using ``final_token`` in place of its last newline."""
    tokens: List[int] = []
    for row in grid:
        for value in row:
            tokens.append(_value_to_token_id(int(value)))
        tokens.append(NEXT_LINE_TOKEN_ID)
    if not tokens:
        raise ValueError("ARC grids must contain at least one row.")
    tokens[-1] = int(final_token)
    return tokens


def encode_example(
    input_grid: Iterable[Iterable[int]],
    output_grid: Optional[Iterable[Iterable[int]]] = None,
    include_output: bool = True,
    append_end: bool = True,
) -> List[int]:
    """Serializes an ARC pair into a single token stream."""
    tokens = [START_TOKEN_ID]
    tokens.extend(grid_to_tokens(input_grid, IO_SEPARATOR_TOKEN_ID))
    if include_output and output_grid is not None:
        final_token = END_PAIR_TOKEN_ID if append_end else NEXT_LINE_TOKEN_ID
        tokens.extend(grid_to_tokens(output_grid, final_token))
    return tokens


def encode_pair(
    pair: Dict[str, object],
    transform_index: int = 0,
    mapping: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int]:
    """Serialize one input and optional output, applying the same augmentation."""
    output = pair.get("output")
    tokens = encode_example(
        apply_dihedral_transform(pair["input"], transform_index),
        apply_dihedral_transform(output, transform_index) if output is not None else None,
    )
    sep_index = tokens.index(IO_SEPARATOR_TOKEN_ID)
    tensor = torch.tensor(tokens, dtype=torch.long)
    return (mapping[tensor] if mapping is not None else tensor), sep_index


def encode_context(
    demonstrations: Sequence[Dict[str, object]],
    target: Dict[str, object],
    transform_index: int = 0,
    mapping: Optional[torch.Tensor] = None,
    scale: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, int]:
    """Serialize complete demos and a target using one shared augmentation.

    A non-identity ``scale`` (rows, cols) block-upscales every grid in its original
    orientation before the dihedral transform (``scale_augmentation``).
    """
    if scale is not None and is_identity_scale(scale):
        scale = None

    def view(grid):
        return apply_dihedral_transform(grid if scale is None else upscale_grid(grid, scale), transform_index)

    tokens = [START_TOKEN_ID]
    target_sep = -1
    for index, pair in enumerate([*demonstrations, target]):
        if index < len(demonstrations) and pair.get("output") is None:
            raise ValueError("Demonstrations must have outputs.")
        tokens.extend(grid_to_tokens(view(pair["input"]), IO_SEPARATOR_TOKEN_ID))
        if index == len(demonstrations):
            target_sep = len(tokens) - 1
        if pair.get("output") is not None:
            tokens.extend(grid_to_tokens(view(pair["output"]), END_PAIR_TOKEN_ID))
    tensor = torch.tensor(tokens, dtype=torch.long)
    return (mapping[tensor] if mapping is not None else tensor), target_sep


def select_balanced_demonstrations(
    pair_indices: Sequence[int], *, visit: int, seed: int,
) -> Tuple[int, ...]:
    """Two distinct demos; every prefix balances inclusion and each slot to ±1.

    A cycle has N rows. Odd N pairs adjacent entries in two copies of a
    shuffled bag. Even N first pairs the bag, then reverses each pair's slots.
    Every demo appears twice per cycle, once in each slot. No mutable worker
    state or training/augmentation RNG is consumed.
    """
    count = len(pair_indices)
    if count < 2:
        return tuple(pair_indices)
    cycle, row = divmod(max(0, int(visit)), count)
    order = _order_for_cycle(seed, pair_indices, 13, cycle)
    if count % 2:
        return order[(2 * row) % count], order[(2 * row + 1) % count]
    first, second = order[2 * (row % (count // 2)):2 * (row % (count // 2)) + 2]
    return (first, second) if row < count // 2 else (second, first)


def load_challenges(json_path: Path) -> Dict[str, dict]:
    with Path(json_path).open("r") as handle:
        return json.load(handle)


def load_nvarc_pairs(json_path: Path) -> List[Dict[str, object]]:
    with Path(json_path).open("r") as handle:
        pairs = json.load(handle)
    if not isinstance(pairs, list):
        raise TypeError(f"NVARC file {json_path} must contain a list of input/output pairs.")
    return pairs


def split_nvarc_task_id(task_id: str) -> Tuple[str, str]:
    first_task_id, separator, second_task_id = str(task_id).partition("_")
    if not separator or not first_task_id or not second_task_id:
        raise ValueError(
            f"Invalid NVARC task id '{task_id}': expected '<task1>_<task2>'."
        )
    return first_task_id, second_task_id


def build_task_embedding_index(
    task_ids: Sequence[str],
    *,
    regular_task_ids: Optional[Set[str]] = None,
    nvarc_task_ids: Optional[Set[str]] = None,
    embedding_task_whitelist: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Dict[str, int], Dict[str, Tuple[int, int]]]:
    regular_task_ids = set(regular_task_ids or ())
    nvarc_task_ids = set(nvarc_task_ids or ())
    embedding_task_ids: List[str] = []
    embedding_task_id_to_index: Dict[str, int] = {}
    task_id_to_embedding_ids: Dict[str, Tuple[int, int]] = {}

    def ensure_embedding_task_id(task_id: str) -> int:
        existing = embedding_task_id_to_index.get(task_id)
        if existing is not None:
            return existing
        new_index = len(embedding_task_ids)
        embedding_task_ids.append(task_id)
        embedding_task_id_to_index[task_id] = new_index
        return new_index

    if embedding_task_whitelist is not None:
        for embedding_task_id in embedding_task_whitelist:
            ensure_embedding_task_id(str(embedding_task_id))

    for task_id in task_ids:
        is_nvarc_task = task_id in nvarc_task_ids
        split_task_ids: Optional[Tuple[str, str]] = None
        if not is_nvarc_task and "_" in task_id:
            try:
                split_task_ids = split_nvarc_task_id(task_id)
            except ValueError:
                split_task_ids = None
            if split_task_ids is not None:
                is_nvarc_task = (
                    (split_task_ids[0] in regular_task_ids or split_task_ids[0] in embedding_task_id_to_index)
                    and (split_task_ids[1] in regular_task_ids or split_task_ids[1] in embedding_task_id_to_index)
                )
        if is_nvarc_task:
            if split_task_ids is None:
                split_task_ids = split_nvarc_task_id(task_id)
            first_task_id, second_task_id = split_task_ids
            task_id_to_embedding_ids[task_id] = (
                ensure_embedding_task_id(first_task_id),
                ensure_embedding_task_id(second_task_id),
            )
            continue
        task_id_to_embedding_ids[task_id] = (ensure_embedding_task_id(task_id), -1)

    return embedding_task_ids, embedding_task_id_to_index, task_id_to_embedding_ids


def scan_nvarc_dir(root: Optional[Path]) -> Dict[str, Path]:
    if root is None:
        return {}
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Missing expected NVARC directory: {root}")
    paths_by_task: Dict[str, Path] = {}
    for path in sorted(root.rglob("*.json")):
        if path.stem in paths_by_task:
            raise ValueError(f"Duplicate NVARC task id '{path.stem}' in {root}.")
        paths_by_task[path.stem] = path
    return paths_by_task


def tokens_to_grid(tokens: Sequence[int]) -> List[List[int]]:
    """Converts a flat sequence of tokens into a grid (list of rows)."""
    rows: List[List[int]] = []
    current_row: List[int] = []
    for token in tokens:
        if token == NEXT_LINE_TOKEN_ID:
            if current_row:
                rows.append(current_row)
                current_row = []
            continue
        if 0 <= token <= 9:
            current_row.append(token)
        else:
            break
    if current_row:
        rows.append(current_row)
    return rows


def extract_output_tokens(sequence: Sequence[int]) -> List[int]:
    """Return the target output following the sequence's last separator."""
    outputs: List[int] = []
    last_separator = max(
        (index for index, token in enumerate(sequence) if token == IO_SEPARATOR_TOKEN_ID),
        default=-1,
    )
    for token in sequence[last_separator + 1 :]:
        if token == END_PAIR_TOKEN_ID:
            break
        outputs.append(token)
    return outputs


def tokens_to_string(tokens: Sequence[int]) -> str:
    """Convert a token id sequence into a space-delimited string."""
    parts: List[str] = []
    for tok in tokens:
        parts.append(ID_TO_TOKEN.get(int(tok), str(int(tok))))
    return " ".join(parts)


def split_grids_from_tokens(tokens: Sequence[int]) -> List[List[List[int]]]:
    """Split a token sequence into multiple grids."""
    grids: List[List[List[int]]] = []
    current_grid: List[List[int]] = []
    current_row: List[int] = []

    for tok in tokens:
        if tok == START_TOKEN_ID:
            continue
        if tok == IO_SEPARATOR_TOKEN_ID or tok == END_PAIR_TOKEN_ID:
            if current_row:
                current_grid.append(current_row)
                current_row = []
            if current_grid:
                grids.append(current_grid)
            current_grid = []
            continue
        if tok == NEXT_LINE_TOKEN_ID:
            if current_row:
                current_grid.append(current_row)
                current_row = []
            continue
        if 0 <= tok <= 9:
            current_row.append(int(tok))
        else:
            break

    if current_row:
        current_grid.append(current_row)
    if current_grid:
        grids.append(current_grid)
    return grids


def is_rectangular_grid(grid: Sequence[Sequence[int]]) -> bool:
    """Return True if all rows have the same non-zero length."""
    if not grid:
        return False
    first_row_len = len(grid[0])
    if first_row_len == 0:
        return False
    return all(len(row) == first_row_len for row in grid)


# =============================================================================
# 3D Position Computation (Numba-accelerated)
# =============================================================================

@njit
def _fill_3d_positions_numba(ids, mask, out, start_id, sep_id, end_id, nl_id):
    B, S = ids.shape
    for b in range(B):
        x = 0
        y = 0
        z = 1
        for t in range(S):
            if not mask[b, t]:
                continue

            val = ids[b, t]

            if val == start_id:
                out[b, t, 0] = 0
                out[b, t, 1] = 0
                out[b, t, 2] = 0
                x = 0
                y = 0
                z = 1
                continue

            px = x
            if px < 0:
                px = 0
            if px > 30:
                px = 30

            py = y
            if py < 0:
                py = 0
            if py > 29:
                py = 29

            out[b, t, 0] = px
            out[b, t, 1] = py
            out[b, t, 2] = z

            if val == sep_id or val == end_id:
                x = 0
                y = 0
                z += 1
            elif val == nl_id:
                x = 0
                y += 1
            else:
                x += 1


def compute_positions_3d(
    input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Compute per-token 3D grid coordinates on CPU (using numba)."""
    if input_ids.dim() != 2:
        raise ValueError("input_ids must have shape [batch, seq_len].")

    ids_cpu = input_ids.detach().cpu()
    ids_np = ids_cpu.numpy()

    if attention_mask is None:
        mask_np = np.ones_like(ids_np, dtype=bool)
    else:
        mask_np = attention_mask.detach().cpu().numpy().astype(bool)

    B, S = ids_np.shape
    pos_np = np.zeros((B, S, 3), dtype=np.int32)

    _fill_3d_positions_numba(
        ids_np,
        mask_np,
        pos_np,
        START_TOKEN_ID,
        IO_SEPARATOR_TOKEN_ID,
        END_PAIR_TOKEN_ID,
        NEXT_LINE_TOKEN_ID,
    )

    return torch.from_numpy(pos_np).to(device=input_ids.device)


# =============================================================================
# Dataset Classes
# =============================================================================

@dataclass
class SequenceExample:
    tokens: torch.LongTensor
    sep_index: int
    example_id: int
    task_id: str
    split: str
    pair_index: int
    has_output: bool
    seq_len: int
    task_embedding_ids: Tuple[int, int]
    preselected_dihedral_index: Optional[int] = None
    is_rearc: bool = False
    is_nvarc: bool = False
    augment_epoch_override: Optional[int] = None
    dihedral_occurrence_override: Optional[int] = None
    demo_visit_override: Optional[int] = None
    variation_path: Optional[str] = None
    variation_colors: Tuple[int, ...] = ()
    variation_output_only_colors: Tuple[int, ...] = ()


class ARCExampleDataset(Dataset):
    """Dataset that turns ARC tasks into autoregressive token sequences."""

    def __init__(
        self,
        json_path: Path,
        rearc_json_path: Optional[Path] = None,
        nvarc_training_dir: Optional[Path] = None,
        nvarc_full_dir: Optional[Path] = None,
        rearc_pairs_per_epoch: int = 10,
        nvarc_pairs_per_epoch: int = 10,
        nvarc_puzzles_per_epoch: Optional[int] = None,
        splits: Sequence[str] = ("train", "test"),
        include_outputs: bool = True,
        max_seq_len: Optional[int] = None,
        drop_long_sequences: bool = False,
        task_whitelist: Optional[Sequence[str]] = None,
        embedding_task_whitelist: Optional[Sequence[str]] = None,
        allow_missing_tasks: bool = False,
        seed: int = 0,
        max_augments: int = 1,
        augmentation_sampling_mode: Optional[str] = None,
        index_present_tasks_only: bool = False,
        selected_task_ids: Optional[Sequence[str]] = None,
        attention_pattern: str = "dense",
        training_demo_schedule: str = "all",
        inference_demo_policy: str = "all",
        pair_identity_vertices: int = 0,
        max_training_demos: Optional[int] = None,
        inference_max_demos: Optional[int] = None,
        scale_augmentation: bool = False,
        scale_max_specs: int = DEFAULT_SCALE_MAX_SPECS,
        scale_anisotropic: bool = True,
        rearc_max_demos: int = DEFAULT_REARC_MAX_DEMOS,
    ) -> None:
        available_splits = {"train", "test"}
        for split in splits:
            if split not in available_splits:
                raise ValueError(
                    f"Unsupported split '{split}'. Expected values in {available_splits}."
                )

        self.source_path = Path(json_path)
        self.rearc_source_path = (
            Path(rearc_json_path) if rearc_json_path is not None else None
        )
        self.nvarc_training_dir = (
            Path(nvarc_training_dir) if nvarc_training_dir is not None else None
        )
        self.nvarc_full_dir = (
            Path(nvarc_full_dir) if nvarc_full_dir is not None else None
        )
        self.attention_pattern = normalize_attention_pattern(attention_pattern)
        if training_demo_schedule not in ("all", "half_or_more", "two"):
            raise ValueError("training_demo_schedule must be all, half_or_more or two")
        if inference_demo_policy not in ("all", "leave_one_out", "two"):
            raise ValueError("inference_demo_policy must be all, leave_one_out or two")
        self.training_demo_schedule = training_demo_schedule
        self.inference_demo_policy = inference_demo_policy
        # None keeps the uncapped schedules; an integer (1-6) selects the balanced capped
        # schedules (demo_schedule.training_demonstrations / balanced_demonstrations).
        for name, cap in (("max_training_demos", max_training_demos), ("inference_max_demos", inference_max_demos)):
            if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int)
                                    or not 1 <= cap <= MAX_CONTEXT_DEMOS):
                raise ValueError(f"{name} must be null or an integer from 1 to {MAX_CONTEXT_DEMOS}")
        self.max_training_demos = max_training_demos
        self.inference_max_demos = inference_max_demos
        self.pair_identity_vertices = int(pair_identity_vertices)
        # Training-only block upscale of every grid in a regular training sequence
        # (scale_augmentation.py); off leaves every sequence unchanged.
        self.scale_augmentation = bool(scale_augmentation)
        self.scale_max_specs = int(scale_max_specs)
        self.scale_anisotropic = bool(scale_anisotropic)
        if self.scale_max_specs < 1:
            raise ValueError("scale_max_specs must be >= 1.")
        # Most demos shown with a ReARC target (rearc_demonstration_indices).
        self.rearc_max_demos = int(rearc_max_demos)
        if self.rearc_max_demos < 1:
            raise ValueError("rearc_max_demos must be >= 1.")
        self.max_seq_len = max_seq_len if max_seq_len is not None else (
            MAX_SEQ_LEN if self.attention_pattern == "dense" else 2**31 - 1
        )
        self.drop_long_sequences = drop_long_sequences
        self.include_outputs = include_outputs
        self.seed = int(seed)
        self.max_augments = max(1, int(max_augments))
        self.augmentation_sampling_mode = normalize_augmentation_sampling_mode(augmentation_sampling_mode)
        challenges = load_challenges(self.source_path)
        rearc_challenges = (
            load_challenges(self.rearc_source_path)
            if self.rearc_source_path is not None
            else {}
        )
        rearc_task_ids = set(rearc_challenges.keys())
        nvarc_source_paths = scan_nvarc_dir(self.nvarc_training_dir)
        for task_id, source_path in scan_nvarc_dir(self.nvarc_full_dir).items():
            if task_id in nvarc_source_paths:
                raise ValueError(f"Duplicate NVARC task id '{task_id}' across NVARC directories.")
            nvarc_source_paths[task_id] = source_path
        nvarc_task_ids = set(nvarc_source_paths.keys())
        duplicate_task_ids = (
            set(challenges.keys()) | rearc_task_ids
        ) & nvarc_task_ids
        if duplicate_task_ids:
            raise ValueError(
                "NVARC task ids must not overlap existing dataset ids: "
                + ", ".join(sorted(duplicate_task_ids)[:10])
            )

        if task_whitelist is not None:
            task_ids = list(task_whitelist)
        else:
            task_ids = sorted(set(challenges.keys()) | rearc_task_ids)
            task_ids.extend(sorted(nvarc_task_ids))
        if index_present_tasks_only:
            # Keep the embedding whitelist intact: learned row numbers and
            # fresh-factor initialization must survive evaluation filtering.
            present_task_ids = set(challenges) | rearc_task_ids | nvarc_task_ids
            task_ids = [task_id for task_id in task_ids if task_id in present_task_ids]
        if selected_task_ids is not None:
            selected = set(selected_task_ids)
            task_ids = [task_id for task_id in task_ids if task_id in selected]

        self.examples: List[SequenceExample] = []
        self.task_id_to_example_id: Dict[str, int] = {}
        self.indices_by_split: Dict[str, List[int]] = {split: [] for split in splits}
        self.task_ids = task_ids
        self.sequence_lengths: List[int] = []
        self.task_colors: Dict[str, List[int]] = {}
        self.task_output_only_colors: Dict[str, List[int]] = {}
        self.pairs_by_task: Dict[str, List[Dict[str, object]]] = {}
        self.test_pairs_by_task: Dict[str, List[Dict[str, object]]] = {}
        self.regular_train_indices_by_example_id: Dict[int, List[int]] = {}
        self.rearc_pairs_by_task: Dict[str, List[Dict[str, object]]] = {}
        self.rearc_pair_counts_by_example_id: Dict[int, int] = {}
        self.rearc_example_cache: Dict[Tuple[int, int, int], SequenceExample] = {}
        self.nvarc_source_paths = nvarc_source_paths
        self.nvarc_pair_counts_by_example_id: Dict[int, int] = {}
        self.nvarc_raw_cache: OrderedDict[str, List[Dict[str, object]]] = OrderedDict()
        self.nvarc_raw_cache_size = 16
        self.rearc_pairs_per_epoch = max(0, int(rearc_pairs_per_epoch))
        self.nvarc_pairs_per_epoch = max(0, int(nvarc_pairs_per_epoch))
        self.nvarc_puzzles_per_epoch = (
            None
            if nvarc_puzzles_per_epoch is None
            else max(0, int(nvarc_puzzles_per_epoch))
        )
        self.embedding_task_ids, self.embedding_task_id_to_index, self.task_id_to_embedding_ids = (
            build_task_embedding_index(
                task_ids,
                regular_task_ids=set(challenges.keys()) | rearc_task_ids,
                nvarc_task_ids=nvarc_task_ids,
                embedding_task_whitelist=embedding_task_whitelist,
            )
        )

        for example_id, task_id in enumerate(task_ids):
            self.task_id_to_example_id[task_id] = example_id
            is_rearc_task = task_id in rearc_task_ids
            is_nvarc_task = task_id in nvarc_task_ids
            if is_nvarc_task:
                nvarc_pairs = self._get_nvarc_pairs(task_id)
                self.pairs_by_task[task_id] = nvarc_pairs
                self.task_colors[task_id] = extract_pairs_colors(nvarc_pairs)
                self.task_output_only_colors[task_id] = extract_task_output_only_colors({"train": nvarc_pairs})
                nvarc_pair_count = len(nvarc_pairs)
                if nvarc_pair_count > 0:
                    self.nvarc_pair_counts_by_example_id[example_id] = nvarc_pair_count
                continue
            task = rearc_challenges.get(task_id) if is_rearc_task else challenges.get(task_id)
            if task is None:
                if allow_missing_tasks:
                    continue
                raise ValueError(f"Task id '{task_id}' was not found in the loaded challenge files.")
            train_pairs = list(task.get("train", []))
            self.pairs_by_task[task_id] = train_pairs
            self.test_pairs_by_task[task_id] = list(task.get("test", []))
            self.task_colors[task_id] = extract_task_colors(task)
            self.task_output_only_colors[task_id] = extract_task_output_only_colors(task)
            if is_rearc_task:
                if "train" in splits:
                    if train_pairs:
                        self.rearc_pairs_by_task[task_id] = train_pairs
                        self.rearc_pair_counts_by_example_id[example_id] = len(train_pairs)
                continue
            for split in splits:
                pairs = task.get(split, [])
                for pair_index, pair in enumerate(pairs):
                    example = self._build_sequence_example(
                        example_id=example_id,
                        task_id=task_id,
                        task_embedding_ids=self.task_id_to_embedding_ids[task_id],
                        split=split,
                        pair_index=pair_index,
                        pair=pair,
                        is_rearc=False,
                        is_nvarc=False,
                    )
                    if example is None:
                        continue
                    example_index = len(self.examples)
                    self.indices_by_split.setdefault(split, []).append(example_index)
                    self.examples.append(example)
                    self.sequence_lengths.append(example.seq_len)
                    if split == "train":
                        self.regular_train_indices_by_example_id.setdefault(
                            example_id, []
                        ).append(example_index)

        self.num_examples = len(self.embedding_task_ids)
        self.has_regular_train_sampling = bool(self.regular_train_indices_by_example_id)
        self.has_rearc_train_sampling = bool(self.rearc_pairs_by_task)
        self.has_nvarc_train_sampling = bool(self.nvarc_pair_counts_by_example_id)
        self.has_lazy_train_sampling = (
            self.has_regular_train_sampling
            or self.has_rearc_train_sampling
            or self.has_nvarc_train_sampling
        )
        if max_seq_len is None:
            # Reserve full-size grids and one target in addition to the demos.
            # This is a capacity limit, not padding or a preallocated buffer.
            counts = [len(pairs) for pairs in self.pairs_by_task.values()]
            self.max_seq_len = self.context_capacity(max(counts, default=0))

    @property
    def capped_training_demos(self) -> bool:
        """Balanced capped training contexts (max_training_demos set, or the two-demo schedule)."""
        return getattr(self, "max_training_demos", None) is not None or self.training_demo_schedule == "two"

    @property
    def capped_inference_demos(self) -> bool:
        """Balanced capped decode contexts (inference_max_demos set, or the two-demo policy)."""
        return getattr(self, "inference_max_demos", None) is not None or self.inference_demo_policy == "two"

    def context_capacity(self, pair_count: int) -> int:
        if self.attention_pattern == "dense":
            return MAX_SEQ_LEN
        if self.capped_training_demos or self.capped_inference_demos:
            pair_count = min(MAX_CONTEXT_DEMOS, int(pair_count))
        return max(MAX_SEQ_LEN, 1 + (int(pair_count) + 1) * (MAX_PAIR_LEN - 1))

    def _get_nvarc_pairs(self, task_id: str) -> List[Dict[str, object]]:
        cached = self.nvarc_raw_cache.pop(task_id, None)
        if cached is None:
            cached = load_nvarc_pairs(self.nvarc_source_paths[task_id])
        self.nvarc_raw_cache[task_id] = cached
        while len(self.nvarc_raw_cache) > self.nvarc_raw_cache_size:
            self.nvarc_raw_cache.popitem(last=False)
        return cached

    def _build_sequence_example(
        self,
        *,
        example_id: int,
        task_id: str,
        task_embedding_ids: Optional[Tuple[int, int]] = None,
        split: str,
        pair_index: int,
        pair: Dict[str, object],
        is_rearc: bool,
        is_nvarc: bool,
        dihedral_indices: Optional[Sequence[int]] = None,
    ) -> Optional[SequenceExample]:
        input_grid = pair["input"]
        output_grid = pair.get("output")

        has_output = output_grid is not None
        include_output_tokens = self.include_outputs and has_output
        append_end = include_output_tokens
        if task_embedding_ids is None:
            task_embedding_ids = self.task_id_to_embedding_ids[task_id]

        transform_index = 0 if dihedral_indices is None else int(dihedral_indices[0])
        tokens = encode_example(
            apply_dihedral_transform(input_grid, transform_index),
            apply_dihedral_transform(output_grid, transform_index) if has_output else None,
            include_output=include_output_tokens,
            append_end=append_end,
        )
        if len(tokens) > self.max_seq_len:
            if self.drop_long_sequences:
                return None
            raise ValueError(
                f"Sequence length {len(tokens)} exceeds max_seq_len={self.max_seq_len} "
                f"for task {task_id} ({split} pair {pair_index})."
            )
        return SequenceExample(
            tokens=torch.tensor(tokens, dtype=torch.long),
            sep_index=tokens.index(IO_SEPARATOR_TOKEN_ID),
            example_id=example_id,
            task_embedding_ids=(
                int(task_embedding_ids[0]),
                int(task_embedding_ids[1]),
            ),
            task_id=task_id,
            split=split,
            pair_index=pair_index,
            has_output=has_output,
            seq_len=len(tokens),
            preselected_dihedral_index=(
                transform_index if dihedral_indices is not None else None
            ),
            is_rearc=is_rearc,
            is_nvarc=is_nvarc,
        )

    def build_training_sequence(
        self,
        example: SequenceExample,
        transform_index: Optional[int],
        mapping: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, int, torch.Tensor]:
        pairs = (
            self.test_pairs_by_task[example.task_id]
            if example.split == "test"
            else self.demonstration_pairs(example)
        )
        pair = dict(pairs[example.pair_index])
        if not self.include_outputs:
            pair.pop("output", None)
        visit = example.demo_visit_override
        if visit is None:
            visit = int(example.augment_epoch_override or 0)
        token_tensor, target_sep = self.build_context_sequence(
            example, pair, int(transform_index or 0), mapping, visit=visit,
            scale=self.training_scale_factor(example, pair, visit),
        )
        if token_tensor.numel() > self.max_seq_len:
            raise ValueError(
                f"Sequence length {token_tensor.numel()} exceeds max_seq_len={self.max_seq_len}."
            )
        mask = torch.ones_like(token_tensor, dtype=torch.bool).unsqueeze(0)
        positions = compute_positions_3d(token_tensor.unsqueeze(0), mask).squeeze(0)
        return token_tensor, target_sep, positions

    def demonstration_pairs(self, example: SequenceExample) -> Sequence[Dict[str, object]]:
        return self.pairs_by_task[example.task_id]

    def rearc_demonstration_indices(self, example: SequenceExample, visit: int) -> Tuple[int, ...]:
        """Demos for a ReARC target: a few sibling pairs of the same task, drawn per visit.

        A ReARC task holds its original ARC pairs plus up to 1,000 generated pairs, so
        the pair-group schedules cannot show them all; transductARC shows one random
        sibling. Here the count follows the regular half-or-more rule over
        ``rearc_max_demos`` (2 to 4 by default), so sequences look like ARC tasks decoded
        leave-one-out. The draw comes from a private seed of (task, pair, visit); no shared
        RNG is consumed. Dense contexts keep their balanced two-demo schedule.
        """
        pairs = self.demonstration_pairs(example)
        eligible = [i for i, pair in enumerate(pairs)
                    if pair.get("output") is not None and i != example.pair_index]
        if not eligible:
            return ()
        upper = max(1, min(int(getattr(self, "rearc_max_demos", DEFAULT_REARC_MAX_DEMOS)), len(eligible)))
        lower = min(upper, max(2, (upper + 1) // 2))
        rng = random.Random(schedule_seed(
            self.seed, example.task_id, "rearc-demos-v1", int(example.pair_index), int(visit)))
        count = rng.randint(lower, upper)
        return tuple(sorted(rng.sample(eligible, count)))

    def demonstration_indices(self, example: SequenceExample, visit: int) -> Tuple[int, ...]:
        if example.is_rearc and self.attention_pattern == "pair_group":
            # Dense contexts already take two balanced demos; the pair-group schedules
            # ("all", "half_or_more") would otherwise show the whole ReARC pool.
            return self.rearc_demonstration_indices(example, visit)
        pairs = self.demonstration_pairs(example)
        if self.attention_pattern == "pair_group":
            capped = self._capped_demonstration_indices(example, visit, pairs)
            if capped is not None:
                return capped
        eligible = [i for i, pair in enumerate(pairs)
                    if pair.get("output") is not None
                    and (example.split != "train" or i != example.pair_index)]
        if self.attention_pattern == "pair_group":
            if self.training_demo_schedule == "half_or_more" and example.split == "train":
                return half_or_more_demonstrations(
                    eligible, visit=visit,
                    seed=_derive_order_seed(self.seed, (
                        example.variation_path or example.task_id,
                        "half-or-more-demos", example.pair_index,
                    )),
                )
            if self.inference_demo_policy == "leave_one_out" and example.split == "test":
                return leave_one_out(eligible, visit=visit, seed=schedule_seed(
                    self.seed, example.task_id, example.split, example.pair_index))
            return tuple(eligible)
        return select_balanced_demonstrations(
            eligible, visit=visit,
            seed=_derive_order_seed(self.seed, (
                example.variation_path or example.task_id,
                f"demos:{example.split}", example.pair_index,
            )),
        )

    def _capped_demonstration_indices(
        self, example: SequenceExample, visit: int, pairs: Sequence[Dict[str, object]],
    ) -> Optional[Tuple[int, ...]]:
        """Balanced capped pair-group contexts, or None for the uncapped schedules."""
        solved = tuple(i for i, pair in enumerate(pairs) if pair.get("output") is not None)
        if example.split == "train":
            if not self.capped_training_demos:
                return None
            cap = MAX_CONTEXT_DEMOS if self.max_training_demos is None else self.max_training_demos
            return training_demonstrations(
                solved, target=example.pair_index, visit=visit,
                half_or_more=self.training_demo_schedule == "half_or_more",
                max_demos=min(cap, 2 if self.training_demo_schedule == "two" else MAX_CONTEXT_DEMOS),
                seed=schedule_seed(self.seed,
                    example.variation_path or example.task_id,
                    "balanced-capped-training-demos-v1",
                ),
            )
        if not self.capped_inference_demos:
            return None
        # Match the maximum fine-tuning context: one demo becomes the target
        # there. The cap applies after this subtraction.
        cap = MAX_CONTEXT_DEMOS if self.inference_max_demos is None else self.inference_max_demos
        seed = schedule_seed(self.seed, example.task_id, example.split, example.pair_index)
        if self.inference_demo_policy == "leave_one_out":
            if cap >= MAX_CONTEXT_DEMOS:
                return balanced_leave_one_out(solved, visit=visit, seed=seed)
            count = min(len(solved) - 1, cap)
            return balanced_demonstrations(solved, count=count, visit=visit, seed=seed)
        count = 2 if self.inference_demo_policy == "two" else len(solved)
        return balanced_demonstrations(solved, count=min(count, cap), visit=visit, seed=seed)

    def training_scale_factor(
        self, example: SequenceExample, target: Dict[str, object], visit: int,
    ) -> Optional[Tuple[int, int]]:
        """One spec for every grid of a regular training sequence, or None when off.

        Uniform over the target pair's specs that every visible demo also offers, drawn
        from a private seed of (task, pair, visit): no shared RNG is consumed.
        """
        if (not getattr(self, "scale_augmentation", False) or example.split != "train"
                or example.is_rearc or example.is_nvarc):
            return None
        pairs = self.demonstration_pairs(example)
        demos = [pairs[i] for i in self.demonstration_indices(example, visit)]
        specs = sequence_scale_specs(target, demos, anisotropic=self.scale_anisotropic,
                                     max_specs=self.scale_max_specs)
        return draw_scale(specs, training_scale_seed(
            self.seed, example.task_id, example.split, example.pair_index, visit))

    def build_context_sequence(
        self, example: SequenceExample, target: Dict[str, object],
        transform_index: int, mapping: Optional[torch.Tensor], *, visit: int,
        scale: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, int]:
        pairs = self.demonstration_pairs(example)
        demos = [pairs[i] for i in self.demonstration_indices(example, visit)]
        return encode_context(demos, target, transform_index, mapping, scale=scale)

    def pair_identity_ids(self, example, positions, *, visit):
        """IDs use physical pair identity before role RoPE erases pair numbers."""
        ids = torch.zeros(positions.shape[:-1], dtype=torch.long)
        if not self.pair_identity_vertices:
            return ids
        visible = self.demonstration_indices(example, visit)
        # Count visits to this context, so skipped/subsampled contexts do not
        # repeatedly land on the same phase of the Latin cycle.
        varying = (example.split == "train" and self.training_demo_schedule != "all") or (
            example.split == "test" and self.inference_demo_policy != "all")
        capped = self.capped_training_demos if example.split == "train" else self.capped_inference_demos
        if capped:
            cap = self.max_training_demos if example.split == "train" else self.inference_max_demos
            varying = varying or len(self.demonstration_pairs(example)) > (MAX_CONTEXT_DEMOS if cap is None else cap)
        occurrence = sum(self.demonstration_indices(example, v) == visible for v in range(visit)) if varying else visit
        assignment = balanced_pair_vertices(visible, visit=occurrence,
            seed=schedule_seed(self.seed, example.variation_path or example.task_id,
                               example.split, example.pair_index),
            vertices=self.pair_identity_vertices)
        for slot, pair in enumerate(visible):
            ids[(positions[..., 2] == 2 * slot + 1) | (positions[..., 2] == 2 * slot + 2)] = assignment[pair]
        return ids

    def __len__(self) -> int:
        return len(self.examples)

    def _with_augment_epoch_override(
        self,
        example: SequenceExample,
        augment_epoch_override: Optional[int],
        dihedral_index: Optional[int] = None,
        dihedral_occurrence: Optional[int] = None,
    ) -> SequenceExample:
        override = None if augment_epoch_override is None else int(augment_epoch_override)
        transform = (
            example.preselected_dihedral_index
            if dihedral_index is None
            else int(dihedral_index)
        )
        occurrence = (
            example.dihedral_occurrence_override
            if dihedral_occurrence is None
            else int(dihedral_occurrence)
        )
        if (
            example.augment_epoch_override == override
            and example.preselected_dihedral_index == transform
            and example.dihedral_occurrence_override == occurrence
        ):
            return example
        return replace(
            example,
            augment_epoch_override=override,
            preselected_dihedral_index=transform,
            dihedral_occurrence_override=occurrence,
        )

    def _resolve_lazy_dihedral_index(
        self,
        *,
        task_id: str,
        pair_index: int,
        augment_epoch_override: int,
        is_nvarc: bool,
    ) -> int:
        augmentor = getattr(self, "augmentor", None)
        if augmentor is None:
            return 0
        if hasattr(augmentor, "select_lazy_dihedral"):
            return int(
                augmentor.select_lazy_dihedral(
                    task_id,
                    pair_index,
                    int(augment_epoch_override),
                    is_nvarc=bool(is_nvarc),
                )
            )
        order_seed = _derive_order_seed(
            int(getattr(augmentor, "rearc_seed", 0)),
            (task_id, "train", pair_index),
        )
        return _index_for_epoch(
            order_seed,
            list(range(8)),
            int(augment_epoch_override),
            2,
        )

    def __getitem__(self, idx: object) -> SequenceExample:
        if isinstance(idx, tuple):
            if idx and isinstance(idx[0], str):
                source = str(idx[0])
                if source == "regular":
                    example_index = int(idx[1])
                    augment_epoch_override = int(idx[2]) if len(idx) > 2 else 0
                    return self._with_augment_epoch_override(
                        self.examples[example_index],
                        augment_epoch_override,
                        int(idx[3]) if len(idx) > 3 else None,
                        int(idx[4]) if len(idx) > 4 else None,
                    )
                example_id = int(idx[1])
                pair_index = int(idx[2])
                augment_epoch_override = int(idx[3]) if len(idx) > 3 else 0
                scheduled_dihedral = int(idx[4]) if len(idx) > 4 else None
                dihedral_occurrence = int(idx[5]) if len(idx) > 5 else None
            else:
                example_id, pair_index = int(idx[0]), int(idx[1])
                augment_epoch_override = None
                scheduled_dihedral = None
                dihedral_occurrence = None
            task_id = self.task_ids[example_id]
            if task_id in self.rearc_pairs_by_task:
                pair = self.rearc_pairs_by_task[task_id][pair_index]
                is_rearc = True
                is_nvarc = False
            else:
                pair = self._get_nvarc_pairs(task_id)[pair_index]
                is_rearc = False
                is_nvarc = True
            selected_dihedral_index = scheduled_dihedral
            if selected_dihedral_index is None and augment_epoch_override is not None:
                selected_dihedral_index = self._resolve_lazy_dihedral_index(
                    task_id=task_id,
                    pair_index=pair_index,
                    augment_epoch_override=augment_epoch_override,
                    is_nvarc=is_nvarc,
                )
            cache_dihedral_index = (
                -1 if selected_dihedral_index is None else int(selected_dihedral_index)
            )
            cache_key = (example_id, pair_index, cache_dihedral_index)
            cached = self.rearc_example_cache.get(cache_key)
            if cached is not None:
                return self._with_augment_epoch_override(
                    cached,
                    augment_epoch_override,
                    selected_dihedral_index,
                    dihedral_occurrence,
                )
            example = self._build_sequence_example(
                example_id=example_id,
                task_id=task_id,
                task_embedding_ids=self.task_id_to_embedding_ids[task_id],
                split="train",
                pair_index=pair_index,
                pair=pair,
                is_rearc=is_rearc,
                is_nvarc=is_nvarc,
                dihedral_indices=(
                    None
                    if selected_dihedral_index is None
                    else (selected_dihedral_index,)
                ),
            )
            if example is None:
                raise ValueError(f"Unexpected dropped lazy example for {task_id}:{pair_index}.")
            if is_rearc:
                self.rearc_example_cache[cache_key] = example
            return self._with_augment_epoch_override(
                example,
                augment_epoch_override,
                selected_dihedral_index,
                dihedral_occurrence,
            )
        return self._with_augment_epoch_override(self.examples[idx], None)

    def get_task_example_id(self, task_id: str) -> int:
        return self.task_id_to_example_id[task_id]

    def get_task_embedding_ids(self, task_id: str) -> Tuple[int, int]:
        return self.task_id_to_embedding_ids[task_id]

    def iter_examples(
        self, split: Optional[str] = None, has_output: Optional[bool] = None
    ) -> Iterable[SequenceExample]:
        for example in self.examples:
            if split is not None and example.split != split:
                continue
            if has_output is not None and example.has_output != has_output:
                continue
            yield example


# =============================================================================
# Batch Sampler and DataLoader
# =============================================================================

def build_shuffle_balanced_dihedral_schedule(
    example_keys: Sequence[object],
    seed: int,
) -> List[int]:
    """Edge-colour occurrences so every full stream/example block uses D4 once."""
    if not example_keys:
        return []

    appearances: Dict[object, int] = {}
    left_ids: Dict[Tuple[object, int], int] = {}
    edges: List[Tuple[int, int]] = []
    for position, key in enumerate(example_keys):
        appearance = appearances.get(key, 0)
        appearances[key] = appearance + 1
        left_key = (key, appearance // 8)
        left_id = left_ids.setdefault(left_key, len(left_ids))
        edges.append((left_id, position // 8))

    rng = random.Random(int(seed))
    edge_order = list(range(len(edges)))
    rng.shuffle(edge_order)
    edge_colors = [-1] * len(edges)
    incident: Dict[Tuple[int, int], Dict[int, int]] = {}

    def assign(edge_id: int, color: int) -> None:
        left_id, right_id = edges[edge_id]
        edge_colors[edge_id] = color
        incident.setdefault((0, left_id), {})[color] = edge_id
        incident.setdefault((1, right_id), {})[color] = edge_id

    for edge_id in edge_order:
        left_id, right_id = edges[edge_id]
        left = (0, left_id)
        right = (1, right_id)
        left_colors = incident.get(left, {})
        right_colors = incident.get(right, {})
        common = [
            color
            for color in range(8)
            if color not in left_colors and color not in right_colors
        ]
        if common:
            assign(edge_id, rng.choice(common))
            continue

        alpha = rng.choice([color for color in range(8) if color not in left_colors])
        beta = rng.choice([color for color in range(8) if color not in right_colors])
        stack = [left]
        component_vertices = {left}
        component_edges: List[int] = []
        seen_edges: Set[int] = set()
        while stack:
            vertex = stack.pop()
            for color in (alpha, beta):
                alternating_edge = incident.get(vertex, {}).get(color)
                if alternating_edge is None or alternating_edge in seen_edges:
                    continue
                seen_edges.add(alternating_edge)
                component_edges.append(alternating_edge)
                edge_left, edge_right = edges[alternating_edge]
                other = (1, edge_right) if vertex[0] == 0 else (0, edge_left)
                if other not in component_vertices:
                    component_vertices.add(other)
                    stack.append(other)
        if right in component_vertices:
            raise RuntimeError("Failed to construct bipartite dihedral edge colouring.")

        for alternating_edge in component_edges:
            old_color = edge_colors[alternating_edge]
            edge_left, edge_right = edges[alternating_edge]
            del incident[(0, edge_left)][old_color]
            del incident[(1, edge_right)][old_color]
        for alternating_edge in component_edges:
            old_color = edge_colors[alternating_edge]
            new_color = beta if old_color == alpha else alpha
            edge_left, edge_right = edges[alternating_edge]
            edge_colors[alternating_edge] = new_color
            incident[(0, edge_left)][new_color] = alternating_edge
            incident[(1, edge_right)][new_color] = alternating_edge
        assign(edge_id, beta)

    return edge_colors


class LengthBucketBatchSampler(Sampler[List[int]]):
    """Group indices with similar sequence lengths to limit padding within a batch."""

    def __init__(
        self,
        lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        bucket_size: Optional[int] = None,
        drop_last: bool = False,
        seed: int = 0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self._epoch = 0
        bucket_size = bucket_size or batch_size * 4
        self.bucket_size = max(bucket_size, batch_size)

    def set_epoch(self, epoch: int) -> None:
        self._epoch = max(0, int(epoch))

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if not self.lengths:
            return iter(())

        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self._epoch)
            indices = torch.randperm(len(self.lengths), generator=generator).tolist()
        else:
            indices = sorted(
                range(len(self.lengths)),
                key=lambda idx: self.lengths[idx],
                reverse=True,
            )

        batches: List[List[int]] = []
        if self.shuffle:
            for start in range(0, len(indices), self.bucket_size):
                bucket = indices[start : start + self.bucket_size]
                bucket.sort(key=lambda idx: self.lengths[idx], reverse=True)
                for bucket_start in range(0, len(bucket), self.batch_size):
                    batch = bucket[bucket_start : bucket_start + self.batch_size]
                    if len(batch) == self.batch_size or not self.drop_last:
                        batches.append(batch)
            if len(batches) > 1:
                order = torch.randperm(len(batches), generator=generator).tolist()
                batches = [batches[i] for i in order]
        else:
            for start in range(0, len(indices), self.batch_size):
                batch = indices[start : start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)

        return iter(batches)


class LazyTrainBatchSampler(Sampler[List[object]]):
    """Build pair-centric epochs across regular ARC, REARC, and NVARC."""

    def __init__(
        self,
        *,
        regular_train_indices_by_example_id: Dict[int, Sequence[int]],
        rearc_pair_counts_by_example_id: Dict[int, int],
        nvarc_pair_counts_by_example_id: Dict[int, int],
        task_ids: Sequence[str],
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
        drop_last: bool = False,
        rearc_pairs_per_task: int = 10,
        nvarc_pairs_per_task: int = 10,
        nvarc_puzzles_per_epoch: Optional[int] = None,
        balance_dihedrals: bool = False,
        num_epochs: int = 1,
        minibatch_multiple: int = 1,
    ) -> None:
        self.regular_train_indices = [
            int(index)
            for indices in regular_train_indices_by_example_id.values()
            for index in indices
        ]
        self.regular_train_indices_by_example_id = {
            int(example_id): [int(index) for index in indices]
            for example_id, indices in regular_train_indices_by_example_id.items()
            if indices
        }
        self.rearc_pair_counts_by_example_id = {
            int(example_id): int(pair_count)
            for example_id, pair_count in rearc_pair_counts_by_example_id.items()
            if int(pair_count) > 0
        }
        self.nvarc_pair_counts_by_example_id = {
            int(example_id): int(pair_count)
            for example_id, pair_count in nvarc_pair_counts_by_example_id.items()
            if int(pair_count) > 0
        }
        self.task_ids = list(task_ids)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.rearc_pairs_per_task = max(0, int(rearc_pairs_per_task))
        self.nvarc_pairs_per_task = max(0, int(nvarc_pairs_per_task))
        self.nvarc_puzzles_per_epoch = (
            None
            if nvarc_puzzles_per_epoch is None
            else max(0, int(nvarc_puzzles_per_epoch))
        )
        self.balance_dihedrals = bool(balance_dihedrals)
        self.num_epochs = max(1, int(num_epochs))
        self.minibatch_multiple = max(1, int(minibatch_multiple))
        self._epoch = 0
        self._balanced_cache: Optional[
            Tuple[Tuple[int, bool, int, int], List[List[object]]]
        ] = None

    def set_epoch(self, epoch: int) -> None:
        self._epoch = max(0, int(epoch))

    def _sample_indices_for_epoch(
        self,
        *,
        order_seed: int,
        item_count: int,
        items_per_epoch: int,
        epoch_index: int,
    ) -> List[Tuple[int, int]]:
        if item_count <= 0 or items_per_epoch <= 0:
            return []
        sample_count = min(int(items_per_epoch), int(item_count))
        offset = int(epoch_index) * sample_count
        cycle = offset // item_count
        index_in_cycle = offset % item_count
        remaining = sample_count
        sampled: List[Tuple[int, int]] = []
        indices = list(range(item_count))
        while remaining > 0:
            order = _order_for_cycle(order_seed, indices, 4, cycle)
            take = min(remaining, item_count - index_in_cycle)
            sampled.extend(
                (int(item_index), int(cycle))
                for item_index in order[index_in_cycle : index_in_cycle + take]
            )
            remaining -= take
            cycle += 1
            index_in_cycle = 0
        return sampled

    def _sample_pairs_for_epoch(
        self,
        *,
        source: str,
        example_id: int,
        pair_count: int,
        pairs_per_epoch: int,
        epoch_index: int,
    ) -> List[Tuple[int, int]]:
        order_seed = _derive_order_seed(
            self.seed,
            (f"{source}_pair", self.task_ids[example_id], 0),
        )
        return self._sample_indices_for_epoch(
            order_seed=order_seed,
            item_count=pair_count,
            items_per_epoch=pairs_per_epoch,
            epoch_index=epoch_index,
        )

    def _sample_nvarc_puzzles_for_epoch(
        self, epoch_index: Optional[int] = None
    ) -> List[Tuple[int, int]]:
        example_ids = sorted(self.nvarc_pair_counts_by_example_id.keys())
        if not example_ids:
            return []
        puzzles_per_epoch = (
            len(example_ids)
            if self.nvarc_puzzles_per_epoch is None
            else self.nvarc_puzzles_per_epoch
        )
        sampled_positions = self._sample_indices_for_epoch(
            order_seed=_derive_order_seed(self.seed, ("nvarc_puzzle", "all", 0)),
            item_count=len(example_ids),
            items_per_epoch=puzzles_per_epoch,
            epoch_index=self._epoch if epoch_index is None else int(epoch_index),
        )
        return [
            (example_ids[position], selection_count)
            for position, selection_count in sampled_positions
        ]

    def __len__(self) -> int:
        total = len(self._indices_for_epoch(self._epoch))
        return (total + self.batch_size - 1) // self.batch_size

    def _indices_for_epoch(self, epoch: int) -> List[object]:
        epoch_i = max(0, int(epoch))
        rng = random.Random(self.seed + epoch_i)
        indices: List[object] = [
            ("regular", index, epoch_i) for index in self.regular_train_indices
        ]
        for example_id in sorted(self.rearc_pair_counts_by_example_id.keys()):
            sampled_pairs = self._sample_pairs_for_epoch(
                source="rearc",
                example_id=example_id,
                pair_count=self.rearc_pair_counts_by_example_id[example_id],
                pairs_per_epoch=self.rearc_pairs_per_task,
                epoch_index=epoch_i,
            )
            indices.extend(
                ("rearc", example_id, pair_index, augment_epoch)
                for pair_index, augment_epoch in sampled_pairs
            )
        for example_id, selection_count in self._sample_nvarc_puzzles_for_epoch(epoch_i):
            sampled_pairs = self._sample_pairs_for_epoch(
                source="nvarc",
                example_id=example_id,
                pair_count=self.nvarc_pair_counts_by_example_id[example_id],
                pairs_per_epoch=self.nvarc_pairs_per_task,
                epoch_index=selection_count,
            )
            indices.extend(
                ("nvarc", example_id, pair_index, augment_epoch)
                for pair_index, augment_epoch in sampled_pairs
            )
        if self.shuffle:
            rng.shuffle(indices)
        if self.drop_last:
            batch_count = len(indices) // self.batch_size
            if self.balance_dihedrals:
                batch_count -= batch_count % self.minibatch_multiple
            indices = indices[: batch_count * self.batch_size]
        return indices

    @staticmethod
    def _augmentation_key(index: object) -> object:
        values = tuple(index)
        if values[0] == "regular":
            return values[0], int(values[1])
        return values[0], int(values[1]), int(values[2])

    def _balanced_indices(self) -> List[List[object]]:
        horizon = max(self.num_epochs, self._epoch + 1)
        signature = (
            horizon,
            bool(self.drop_last),
            self.batch_size,
            self.minibatch_multiple,
        )
        if self._balanced_cache is not None and self._balanced_cache[0] == signature:
            return self._balanced_cache[1]

        epochs = [self._indices_for_epoch(epoch) for epoch in range(horizon)]
        flat = [index for epoch_indices in epochs for index in epoch_indices]
        keys = [self._augmentation_key(index) for index in flat]
        dihedrals = build_shuffle_balanced_dihedral_schedule(
            keys,
            _derive_order_seed(self.seed, ("dihedral", "train", 0)),
        )
        dihedral_counts: Dict[Tuple[object, int], int] = {}
        decorated: List[object] = []
        for index, key, dihedral in zip(flat, keys, dihedrals):
            count_key = (key, dihedral)
            occurrence = dihedral_counts.get(count_key, 0)
            dihedral_counts[count_key] = occurrence + 1
            decorated.append((*tuple(index), dihedral, occurrence))

        balanced_epochs: List[List[object]] = []
        cursor = 0
        for epoch_indices in epochs:
            next_cursor = cursor + len(epoch_indices)
            balanced_epochs.append(decorated[cursor:next_cursor])
            cursor = next_cursor
        self._balanced_cache = signature, balanced_epochs
        return balanced_epochs

    def __iter__(self):
        indices = (
            self._balanced_indices()[self._epoch]
            if self.balance_dihedrals
            else self._indices_for_epoch(self._epoch)
        )

        batches: List[List[object]] = []
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                batches.append(batch)
        return iter(batches)


def packed_prefix_layout_for_batch(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    sep_indices: torch.Tensor,
    attention_pattern: str,
) -> Dict[str, object]:
    """Build the model's packed attention layout on the collating CPU.

    The model adopts it when the pattern matches and rebuilds it otherwise, so
    collation never has to know the model's exact configuration. Pair-group
    batches also carry the split target/demo layout, which that model variant
    uses and the plain variant ignores.
    """
    from target_demo_attention import packed_prefix_lm_layout_fields  # lazy: that module imports common

    pattern = normalize_attention_pattern(attention_pattern)
    fields = packed_prefix_lm_layout_fields(
        cu_seqlens, int(input_ids.numel()), sep_indices.clamp(min=0) + 1, input_ids,
        attention_pattern=pattern, split_target_demo=pattern == "pair_group",
    )
    return {"attention_pattern": pattern, **fields}


def collate_examples(
    batch: List[SequenceExample],
    dataset: ARCExampleDataset,
    pad_token_id: int = END_PAIR_TOKEN_ID,
    augment_selector: Optional[
        Callable[[SequenceExample], Tuple[Optional[torch.Tensor], Optional[int]]]
    ] = None,
) -> Dict[str, torch.Tensor]:
    if not batch:
        raise ValueError("Empty batch encountered during collation.")
    del pad_token_id

    selected: List[
        Tuple[
            SequenceExample,
            torch.Tensor,
            torch.Tensor,
            int,
            int,
            int,
        ]
    ] = []
    for example in batch:
        mapping: Optional[torch.Tensor] = None
        transform_index = getattr(example, "preselected_dihedral_index", None)
        if augment_selector is not None:
            mapping, selected_transform_index = augment_selector(example)
            if transform_index is None:
                transform_index = selected_transform_index
            elif (
                selected_transform_index is not None
                and int(selected_transform_index) != int(transform_index)
            ):
                if getattr(dataset, "augmentation_sampling_mode", None) == "single_grid_prefix":
                    # A symmetric/exhausted unique-input pool may have no entry
                    # for the requested geometry. Serialize the selected entry.
                    transform_index = selected_transform_index
                else:
                    raise ValueError(
                        "Augmentor returned a dihedral index that disagrees with the "
                        f"preselected transform for {example.task_id}:{example.pair_index}."
                    )
        if transform_index is not None:
            if transform_index < 0 or transform_index >= 8:
                raise ValueError(
                    f"Invalid dihedral index {transform_index} for example {example.task_id}."
                )
        tokens, sep_index, positions = dataset.build_training_sequence(
            example, transform_index, mapping
        )
        seq_len = int(tokens.size(0))
        dihedral_id = 0 if transform_index is None else int(transform_index)
        selected.append((example, tokens, positions, seq_len, sep_index, dihedral_id))

    batch_size = len(selected)
    seq_lengths = torch.tensor([item[3] for item in selected], dtype=torch.int32)
    max_len = int(seq_lengths.max().item())
    total_tokens = int(seq_lengths.sum().item())

    input_ids = torch.zeros(total_tokens, dtype=torch.long)
    example_ids = torch.full((batch_size, 2), -1, dtype=torch.int32)
    dihedral_ids = torch.zeros(batch_size, dtype=torch.int32)
    # All-demo contexts can exceed the signed 16-bit separator range.
    sep_indices = torch.zeros(batch_size, dtype=torch.int32)
    positions_3d = torch.zeros((total_tokens, 3), dtype=torch.int32)
    pair_identity_ids = torch.zeros(total_tokens, dtype=torch.long) if getattr(dataset, "pair_identity_vertices", 0) else None

    cursor = 0
    for idx, (
        example,
        tokens,
        positions,
        seq_len,
        sep_index,
        dihedral_id,
    ) in enumerate(selected):
        next_cursor = cursor + seq_len
        input_ids[cursor:next_cursor] = tokens
        task_embedding_ids = tuple(int(task_id) for task_id in example.task_embedding_ids)
        if len(task_embedding_ids) != 2:
            raise ValueError(
                f"Expected exactly 2 task embedding ids for {example.task_id}, got {task_embedding_ids}."
            )
        example_ids[idx, 0] = task_embedding_ids[0]
        example_ids[idx, 1] = task_embedding_ids[1]
        dihedral_ids[idx] = dihedral_id
        sep_indices[idx] = int(sep_index)
        positions_3d[cursor:next_cursor] = positions
        if pair_identity_ids is not None:
            visit = example.demo_visit_override
            if visit is None:
                visit = int(example.augment_epoch_override or 0)
            pair_identity_ids[cursor:next_cursor] = dataset.pair_identity_ids(example, positions, visit=visit)
        cursor = next_cursor

    cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32)
    cu_seqlens[1:] = torch.cumsum(seq_lengths, dim=0)
    prefix_layout = packed_prefix_layout_for_batch(
        input_ids, cu_seqlens, sep_indices, getattr(dataset, "attention_pattern", "dense"))

    return {**({"pair_identity_ids": pair_identity_ids} if pair_identity_ids is not None else {}),
        "input_ids": input_ids,
        "cu_seqlens": cu_seqlens,
        "prefix_layout": prefix_layout,
        "max_seqlen": max_len,
        "has_padding": False,
        "example_ids": example_ids,
        "dihedral_ids": dihedral_ids,
        "sep_indices": sep_indices,
        "positions_3d": positions_3d,
        "task_ids": [example.task_id for example in batch],
        "splits": [example.split for example in batch],
        "has_output": [example.has_output for example in batch],
    }


def create_dataloader(
    dataset: ARCExampleDataset,
    batch_size: int,
    shuffle: bool = True,
    bucket_size_multiplier: int = 4,
    augment_selector: Optional[
        Callable[[SequenceExample], Tuple[Optional[torch.Tensor], Optional[int]]]
    ] = None,
    use_length_bucketing: bool = True,
    seed: int = 0,
    augmentor: Optional[object] = None,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    num_epochs: int = 1,
    gradient_accumulation_steps: int = 1,
) -> DataLoader:
    collate_fn = functools.partial(
        collate_examples,
        dataset=dataset,
        augment_selector=augment_selector,
    )

    num_workers = max(0, int(num_workers))
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    loader_kwargs: Dict[str, object] = {
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": bool(pin_memory),
    }
    if num_workers > 0:
        if distributed.world_size() > 1:
            # NCCL is initialized before workers start and is not fork safe.
            loader_kwargs["multiprocessing_context"] = "spawn"
        if persistent_workers is None:
            persistent_workers = True
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = (
            2 if prefetch_factor is None else max(1, int(prefetch_factor))
        )
        if augmentor is not None and hasattr(augmentor, "set_shared_epoch_buffer"):
            shared_epoch = torch.zeros((), dtype=torch.int64)
            shared_epoch.share_memory_()
            augmentor.set_shared_epoch_buffer(shared_epoch)

    if not use_length_bucketing and getattr(dataset, "has_lazy_train_sampling", False):
        batch_sampler = LazyTrainBatchSampler(
            regular_train_indices_by_example_id=dataset.regular_train_indices_by_example_id,
            rearc_pair_counts_by_example_id=dataset.rearc_pair_counts_by_example_id,
            nvarc_pair_counts_by_example_id=dataset.nvarc_pair_counts_by_example_id,
            task_ids=dataset.task_ids,
            batch_size=batch_size,
            seed=seed,
            shuffle=shuffle,
            rearc_pairs_per_task=getattr(dataset, "rearc_pairs_per_epoch", 10),
            nvarc_pairs_per_task=getattr(dataset, "nvarc_pairs_per_epoch", 10),
            nvarc_puzzles_per_epoch=getattr(dataset, "nvarc_puzzles_per_epoch", None),
            balance_dihedrals=augmentor is not None,
            num_epochs=num_epochs,
            minibatch_multiple=gradient_accumulation_steps,
        )
        return DataLoader(
            dataset,
            batch_sampler=distributed.shard_batch_sampler(batch_sampler),
            **loader_kwargs,
        )

    if use_length_bucketing:
        lengths = getattr(dataset, "sequence_lengths", None)
        if lengths is None:
            lengths = [len(dataset[i].tokens) for i in range(len(dataset))]

        bucket_size = max(batch_size * max(1, bucket_size_multiplier), batch_size)
        batch_sampler = LengthBucketBatchSampler(
            lengths=lengths,
            batch_size=batch_size,
            shuffle=shuffle,
            bucket_size=bucket_size,
            seed=seed,
        )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            **loader_kwargs,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        **loader_kwargs,
    )


# =============================================================================
# RNG and Device Handling
# =============================================================================

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def recommended_num_workers(max_workers: int = 8) -> int:
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 1 or max_workers <= 0:
        return 0
    return max(0, min(int(max_workers), cpu_count - 1))


def resolve_device(device_str: str) -> torch.device:
    device_str = str(device_str or "cuda").lower()
    if not device_str.startswith("cuda"):
        raise ValueError("Only CUDA is supported. Set device='cuda'.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return torch.device(device_str)


def capture_rng_state(device: torch.device) -> Dict[str, Any]:
    """Capture Python/numpy/torch RNG states so training can resume deterministically."""
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    state["torch"] = torch.get_rng_state()
    if torch.cuda.is_available() and device.type == "cuda":
        try:
            if distributed.world_size() > 1:
                state["cuda_local"] = torch.cuda.get_rng_state(device)
            else:
                state["cuda"] = torch.cuda.get_rng_state_all()
        except Exception:
            pass
    return state


def restore_rng_state(state: Dict[str, Any], device: torch.device) -> None:
    """Restore RNG state saved in a checkpoint."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and device.type == "cuda":
        if "cuda_local" in state:
            # DDP rank RNGs must also load on a single-GPU evaluation host.
            torch.cuda.set_rng_state(state["cuda_local"], device)
        elif "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])


# =============================================================================
# Augmentation System
# =============================================================================

def _extract_input_tokens(tokens: Sequence[int]) -> List[int]:
    input_tokens: List[int] = []
    for tok in tokens:
        val = int(tok)
        if val == IO_SEPARATOR_TOKEN_ID:
            break
        input_tokens.append(val)
    return input_tokens


def _hash_tokens(tokens: Sequence[int]) -> int:
    digest = hashlib.sha256(bytes(tokens)).digest()
    return int.from_bytes(digest[:8], "little")


def _derive_order_seed(seed: int, key: Tuple[str, str, int]) -> int:
    payload = f"{int(seed)}|{key[0]}|{key[1]}|{int(key[2])}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _cycle_seed(order_seed: int, mode: int, cycle: int) -> int:
    payload = f"{int(order_seed)}|{int(mode)}|{int(cycle)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _shuffled_indices(indices: Sequence[int], seed: int) -> List[int]:
    order = list(indices)
    rng = random.Random(int(seed))
    rng.shuffle(order)
    return order


def _order_for_cycle(
    order_seed: int,
    indices: Sequence[int],
    mode: int,
    cycle: int,
) -> List[int]:
    cycle_i = max(0, int(cycle))
    order = _shuffled_indices(indices, _cycle_seed(order_seed, mode, cycle_i))
    if cycle_i > 0:
        prev_order = _shuffled_indices(
            indices,
            _cycle_seed(order_seed, mode, cycle_i - 1),
        )
        if order == prev_order:
            order = order[1:] + order[:1]
    return order


def _index_for_epoch(order_seed: int, indices: Sequence[int], epoch: int, mode: int) -> int:
    if len(indices) == 1:
        return int(indices[0])
    epoch_i = max(0, int(epoch))
    cycle_len = len(indices)
    cycle = epoch_i // cycle_len
    offset = epoch_i % cycle_len
    order = _order_for_cycle(order_seed, indices, mode, cycle)
    return int(order[offset])


def _color_mapping_tensor(
    colors: Sequence[int], targets: Sequence[int], destination_colors: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    mapping = torch.arange(VOCAB_SIZE, dtype=torch.long)
    for source, target in zip(colors, targets):
        mapping[source] = target
    if destination_colors is not None:
        # Extend the injection to a permutation of all ten labels. Reserved
        # output-only labels stay fixed; unused labels cannot collide either.
        fixed = set(range(10)) - set(destination_colors)
        assigned = set(colors) | fixed
        remaining_sources = set(range(10)) - assigned
        remaining_targets = set(range(10)) - {int(mapping[source]) for source in assigned}
        for color in sorted(remaining_sources & remaining_targets):
            remaining_sources.remove(color)
            remaining_targets.remove(color)
        for source, target in zip(sorted(remaining_sources), sorted(remaining_targets)):
            mapping[source] = target
    return mapping


def random_color_mapping(
    colors: Sequence[int], seed: int, *, destination_colors: Optional[Sequence[int]] = None,
) -> torch.Tensor:
    colors = [int(color) for color in colors if 0 <= int(color) <= 9]
    rng = random.Random(int(seed))
    if destination_colors is None:
        permutation = list(colors)
        rng.shuffle(permutation)
    else:
        permutation = rng.sample(list(destination_colors), len(colors))
    return _color_mapping_tensor(colors, permutation, destination_colors)


def _unrank_color_injection(
    colors: Sequence[int], rank: int, destination_colors: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """Return one lexicographically ranked injection from colors into ARC colors."""
    available = list(range(10)) if destination_colors is None else list(destination_colors)
    targets: List[int] = []
    rank_i = int(rank)
    color_count = len(colors)
    for index in range(color_count):
        remaining = color_count - index - 1
        block_size = math.factorial(len(available) - 1) // math.factorial(
            len(available) - 1 - remaining
        )
        choice, rank_i = divmod(rank_i, block_size)
        targets.append(available.pop(choice))
    return tuple(targets)


def build_augmentation_schedule(
    colors: Sequence[int], count: int, seed: int, *,
    destination_colors: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, List[int]]:
    colors = sorted({int(color) for color in colors if 0 <= int(color) <= 9})
    requested = max(1, int(count))
    destination_count = 10 if destination_colors is None else len(destination_colors)
    injection_count = math.perm(destination_count, len(colors))
    color_count = min(injection_count, requested)
    schedule_count = min(requested, injection_count * 8)

    rank_rng = random.Random(_cycle_seed(seed, 6, 0))
    ranks = rank_rng.sample(range(injection_count), color_count)
    targets = [_unrank_color_injection(colors, rank, destination_colors) for rank in ranks]

    color_rng = random.Random(_cycle_seed(seed, 7, 0))
    dihedral_rng = random.Random(_cycle_seed(seed, 8, 0))
    color_pool: List[int] = []
    dihedral_pool: List[int] = []
    seen: Set[Tuple[int, int]] = set()
    selected: List[Tuple[int, int]] = []
    while len(selected) < schedule_count:
        if not color_pool:
            color_pool = list(range(color_count))
            color_rng.shuffle(color_pool)
        if not dihedral_pool:
            dihedral_pool = list(range(8))
            dihedral_rng.shuffle(dihedral_pool)
        pair = (color_pool.pop(), dihedral_pool.pop())
        if pair in seen:
            continue
        seen.add(pair)
        selected.append(pair)

    mappings: List[torch.Tensor] = []
    dihedrals: List[int] = []
    for color_index, dihedral in selected:
        mapping = _color_mapping_tensor(colors, targets[color_index], destination_colors)
        mappings.append(mapping)
        dihedrals.append(dihedral)
    return torch.stack(mappings), dihedrals


def _balanced_color_block(
    colors: Sequence[int], rng: random.Random, destination_colors: Optional[Sequence[int]] = None,
) -> List[Tuple[int, ...]]:
    permutation = list(range(10)) if destination_colors is None else list(destination_colors)
    if not permutation:
        return [tuple()]
    rng.shuffle(permutation)
    period = len(permutation)
    offsets = rng.sample(range(period), len(colors))
    block = [
        tuple(permutation[(position + offset) % period] for offset in offsets)
        for position in range(period)
    ]
    rng.shuffle(block)
    return block


def build_balanced_color_schedules(
    colors: Sequence[int],
    pair_keys: Sequence[Tuple[str, str, int]],
    count: int,
    seed: int,
    *, destination_colors: Optional[Sequence[int]] = None,
) -> Dict[Tuple[str, str, int], torch.Tensor]:
    """Build task-coordinated balanced-block color schedules for regular ARC."""
    colors = sorted({int(color) for color in colors if 0 <= int(color) <= 9})
    requested = max(1, int(count))
    period = 10 if destination_colors is None else max(1, len(destination_colors))
    schedules: Dict[Tuple[str, str, int], torch.Tensor] = {}
    task_seen: Set[Tuple[int, ...]] = set()

    for key in sorted(set(pair_keys)):
        rng = random.Random(_cycle_seed(_derive_order_seed(seed, key), 9, 0))
        pair_seen: Set[Tuple[int, ...]] = set()
        targets: List[Tuple[int, ...]] = []
        block_sizes = [period] * (requested // period)
        if requested % period:
            block_sizes.append(requested % period)

        for block_size in block_sizes:
            best: Optional[List[Tuple[int, ...]]] = None
            best_score: Optional[Tuple[int, int]] = None
            for _ in range(32):
                candidate = _balanced_color_block(colors, rng, destination_colors)[:block_size]
                score = (
                    sum(target in pair_seen for target in candidate),
                    sum(target in task_seen for target in candidate),
                )
                if best_score is None or score < best_score:
                    best, best_score = candidate, score
                if score == (0, 0):
                    break
            assert best is not None
            targets.extend(best)
            task_seen.update(best)
            pair_seen.update(best)

        rng.shuffle(targets)
        mappings = torch.stack([
            _color_mapping_tensor(colors, target, destination_colors) for target in targets
        ])
        schedules[key] = mappings

    return schedules


@dataclass
class Augments:
    color_maps: torch.Tensor
    dihedral_indices: List[int]
    color_map_indices: List[int]
    order_seed: int = 0
    unique_inputs: bool = False
    _candidates: Dict[Optional[int], List[int]] = field(default_factory=dict, repr=False)
    _orders: Dict[Tuple[int, Optional[int]], Tuple[int, List[int]]] = field(default_factory=dict, repr=False)


def _input_grid_bytes(grid: Sequence[Sequence[int]]) -> bytes:
    """Compact, shape-preserving input key; row separators cannot be recolored."""
    array = np.asarray(grid, dtype=np.uint8)
    rows = np.empty((array.shape[0], array.shape[1] + 1), dtype=np.uint8)
    rows[:, :-1] = array
    rows[:, -1] = NEXT_LINE_TOKEN_ID
    return rows.tobytes()


def _input_geometry_bytes(grid: Sequence[Sequence[int]]) -> List[bytes]:
    array = np.asarray(grid, dtype=np.uint8)
    variants = (
        array, np.rot90(array, -1), array[::-1, ::-1], np.rot90(array, 1),
        array[:, ::-1], array[::-1], array.T, array.T[::-1, ::-1],
    )
    return [_input_grid_bytes(variant) for variant in variants]


def _unique_color_targets(colors, destinations, rng):
    """Balanced proposals, skipping mappings equivalent on this input."""
    total = math.perm(len(destinations), len(colors))
    limit = min(total, 5000)
    seen = set()
    stale_blocks = 0
    while len(seen) < limit and stale_blocks < 32:
        added = False
        for target in _balanced_color_block(colors, rng, destinations):
            if target not in seen:
                seen.add(target)
                added = True
                yield target
                if len(seen) >= limit:
                    return
        stale_blocks = 0 if added else stale_blocks + 1
    # Finish small domains exhaustively; bound work for large color domains.
    ranks = rng.sample(range(total), min(total, 5000))
    for rank in ranks:
        target = _unrank_color_injection(colors, rank, destinations)
        if target not in seen:
            seen.add(target)
            yield target
            if len(seen) >= limit:
                return


def build_unique_input_augments(
    inputs_by_key: Dict[Tuple[str, str, int], Sequence[Sequence[int]]],
    colors: Sequence[int], *, protected_colors: Sequence[int] = (),
    original_inputs: Iterable[Sequence[Sequence[int]]] = (),
    max_augments: int, seed: int,
    mapping_dtype: torch.dtype = torch.long,
) -> Dict[Tuple[str, str, int], Augments]:
    """March-style task-wide pools, using byte translation instead of cell loops.

    Preserve every original pair, even duplicate originals. All added inputs
    must be distinct from every known original and every addition in this task.
    Try new input-color signatures first, then remaining geometries of accepted
    maps. Requested pool size is a cap, not a reason to repeat an input.
    """
    movable, destinations = color_augmentation_domain(colors, protected_colors, "single_grid_prefix")
    seen_inputs = {_input_grid_bytes(grid) for grid in original_inputs}
    seen_inputs.update(_input_grid_bytes(grid) for grid in inputs_by_key.values())
    states = []
    for key, grid in sorted(inputs_by_key.items()):
        input_colors = sorted(set(np.asarray(grid).ravel().tolist()) & set(movable))
        order_seed = _derive_order_seed(seed, key)
        rng = random.Random(order_seed)
        geometries = list(range(8))
        rng.shuffle(geometries)
        states.append(dict(
            key=key, inputs=_input_geometry_bytes(grid), colors=input_colors,
            rng=rng, targets=iter(_unique_color_targets(input_colors, destinations, rng)),
            maps=[list(range(VOCAB_SIZE))], signatures=[tuple(input_colors)],
            dihedrals=[0], map_indices=[0], counts=[1, 0, 0, 0, 0, 0, 0, 0],
            tried={(tuple(input_colors), 0)}, geometry_order=geometries,
            limit=min(max(1, int(max_augments)), math.perm(len(destinations), len(input_colors)) * 8),
            order_seed=order_seed,
        ))

    def accept(state, signature, mapping, map_index):
        table = bytes.maketrans(bytes(state["colors"]), bytes(signature))
        for dihedral in sorted(state["geometry_order"], key=state["counts"].__getitem__):
            if state["counts"][dihedral] >= math.ceil(state["limit"] / 8):
                continue
            candidate = (signature, dihedral)
            if candidate in state["tried"]:
                continue
            state["tried"].add(candidate)
            transformed = state["inputs"][dihedral].translate(table)
            if transformed in seen_inputs:
                continue
            seen_inputs.add(transformed)
            if map_index is None:
                map_index = len(state["maps"])
                state["maps"].append(mapping)
                state["signatures"].append(signature)
            state["dihedrals"].append(dihedral)
            state["map_indices"].append(map_index)
            state["counts"][dihedral] += 1
            return True
        return False

    # Round-robin across pairs so one pair cannot consume the entire shared
    # set of possible recolorings before another identical pair gets a turn.
    active = list(states)
    while active:
        remaining = []
        for state in active:
            if len(state["dihedrals"]) >= state["limit"]:
                continue
            signature = next(state["targets"], None)
            if signature is None:
                continue
            mapping = list(range(VOCAB_SIZE))
            other_colors = [color for color in movable if color not in state["colors"]]
            available = [color for color in destinations if color not in signature]
            other_targets = state["rng"].sample(available, len(other_colors))
            sources = state["colors"] + other_colors
            targets = list(signature) + other_targets
            for source, target in zip(sources, targets):
                mapping[source] = target
            # Complete the ten-label bijection; protected labels stay fixed.
            unused_sources = set(destinations) - set(sources)
            unused_targets = set(destinations) - set(targets)
            for source, target in zip(sorted(unused_sources - unused_targets), sorted(unused_targets - unused_sources)):
                mapping[source] = target
            existing_index = 0 if signature == tuple(state["colors"]) else None
            accept(state, signature, state["maps"][0] if existing_index == 0 else mapping, existing_index)
            remaining.append(state)
        active = remaining

    active = [state for state in states if len(state["dihedrals"]) < state["limit"]]
    while active:
        remaining = []
        for state in active:
            for index, (signature, mapping) in enumerate(zip(state["signatures"], state["maps"])):
                if accept(state, signature, mapping, index):
                    if len(state["dihedrals"]) < state["limit"]:
                        remaining.append(state)
                    break
        active = remaining

    return {
        state["key"]: Augments(
            color_maps=torch.tensor(state["maps"], dtype=mapping_dtype),
            dihedral_indices=state["dihedrals"], color_map_indices=state["map_indices"],
            order_seed=state["order_seed"], unique_inputs=True,
        ) for state in states
    }


def _select_augment_index(
    augments: Augments,
    order_seed: int,
    epoch: int,
    preselected_dihedral: Optional[int] = None,
    dihedral_occurrence: Optional[int] = None,
) -> int:
    if augments.unique_inputs:
        if len(augments.dihedral_indices) == 1:
            return 0
        candidates = augments._candidates.get(preselected_dihedral)
        if candidates is None:
            candidates = [index for index, dihedral in enumerate(augments.dihedral_indices)
                          if preselected_dihedral is None or dihedral == preselected_dihedral]
            if not candidates:
                candidates = list(range(len(augments.dihedral_indices)))
            augments._candidates[preselected_dihedral] = candidates
        if len(candidates) == 1:
            return candidates[0]
        selection = max(0, int(epoch if dihedral_occurrence is None else dihedral_occurrence))
        cycle, offset = divmod(selection, len(candidates))
        cache_key = (order_seed, preselected_dihedral)
        cached = augments._orders.get(cache_key)
        if cached is None or cached[0] != cycle:
            cached = (cycle, _order_for_cycle(order_seed, candidates, 3, cycle))
            augments._orders[cache_key] = cached
        return cached[1][offset]
    candidates = list(range(len(augments.dihedral_indices)))
    if preselected_dihedral is not None:
        matching = [
            index
            for index in candidates
            if int(augments.dihedral_indices[index]) == int(preselected_dihedral)
        ]
        candidates = matching or candidates
    selection_index = epoch if dihedral_occurrence is None else dihedral_occurrence
    return _index_for_epoch(order_seed, candidates, selection_index, 3)


class Augmentor:
    """Manages augmentation selection per example during training/evaluation."""

    def __init__(
        self,
        augments_by_key: Dict[Tuple[str, str, int], Augments],
        *,
        max_augments: int = 0,
    ) -> None:
        self.augments_by_key = augments_by_key
        self.max_augments = int(max_augments)
        self._enabled = True
        self._epoch = 0
        self._shared_epoch: Optional[torch.Tensor] = None

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)

    def set_shared_epoch_buffer(self, shared_epoch: Optional[torch.Tensor]) -> None:
        self._shared_epoch = shared_epoch
        if shared_epoch is not None:
            shared_epoch.fill_(int(self._epoch))

    def _current_epoch(self) -> int:
        if self._shared_epoch is not None:
            return int(self._shared_epoch.item())
        return self._epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = max(0, int(epoch))
        if self._shared_epoch is not None:
            self._shared_epoch.fill_(int(self._epoch))

    def _select_index_for_epoch(self, augments: Augments, epoch: Optional[int] = None) -> int:
        count = len(augments.dihedral_indices)
        if count <= 1:
            return 0
        effective_epoch = self._current_epoch() if epoch is None else max(0, int(epoch))
        return _select_augment_index(
            augments,
            augments.order_seed,
            effective_epoch,
        )

    def select_for_example(self, example: SequenceExample) -> Tuple[Optional[torch.Tensor], Optional[int]]:
        if not self._enabled:
            return None, None
        key = (example.task_id, example.split, example.pair_index)
        augments = self.augments_by_key.get(key)
        if augments is None:
            return None, None
        effective_epoch = getattr(example, "augment_epoch_override", None)
        selection_epoch = (
            self._current_epoch() if effective_epoch is None else effective_epoch
        )
        preselected = getattr(example, "preselected_dihedral_index", None)
        idx = _select_augment_index(
            augments,
            augments.order_seed,
            selection_epoch,
            preselected,
            getattr(example, "dihedral_occurrence_override", None),
        )
        color_idx = augments.color_map_indices[idx]
        dihedral_idx = (
            augments.dihedral_indices[idx] if preselected is None or augments.unique_inputs else int(preselected)
        )
        mapping = augments.color_maps[color_idx]
        if mapping.dtype != torch.long:
            mapping = mapping.long()
        return mapping, dihedral_idx


class CombinedAugmentor:
    """Dispatch regular, REARC, and NVARC augmentation policies from one selector."""

    def __init__(
        self,
        regular_augmentor: Optional[Augmentor],
        *,
        rearc_seed: int = 0,
        task_colors: Optional[Dict[str, Sequence[int]]] = None,
        max_augments: int = 0,
        augmentation_sampling_mode: Optional[str] = None,
        task_output_only_colors: Optional[Dict[str, Sequence[int]]] = None,
        dataset: Optional[ARCExampleDataset] = None,
    ) -> None:
        self.regular_augmentor = regular_augmentor
        self.rearc_seed = int(rearc_seed)
        self.augmentation_sampling_mode = normalize_augmentation_sampling_mode(augmentation_sampling_mode)
        self.task_output_only_colors = task_output_only_colors or {}
        self.dataset = dataset if self.augmentation_sampling_mode == "single_grid_prefix" else None
        self.task_colors = {
            str(task_id): tuple(int(color) for color in colors if 0 <= int(color) <= 9)
            for task_id, colors in (task_colors or {}).items()
        }
        self._enabled = True
        self._epoch = 0
        self._shared_epoch: Optional[torch.Tensor] = None
        source = regular_augmentor
        self.max_augments = (
            int(source.max_augments) if source is not None else max(1, int(max_augments))
        )
        self._rearc_augments_by_task: Dict[str, Augments] = {}
        self._unique_augmentors: Dict[str, Augmentor] = {}

    def _unique_augmentor(self, task_id: str, example: Optional[SequenceExample] = None) -> Augmentor:
        augmentor = self._unique_augmentors.get(task_id)
        if augmentor is None:
            if self.dataset is not None:
                pairs = self.dataset.pairs_by_task[task_id]
                inputs = {(task_id, "train", index): pair["input"] for index, pair in enumerate(pairs)}
                originals = [pair["input"] for pair in self.dataset.test_pairs_by_task.get(task_id, ())]
            elif example is not None:
                # Standalone selectors can still construct a one-pair pool.
                inputs = {(task_id, example.split, example.pair_index):
                          tokens_to_grid(example.tokens[1:example.sep_index].tolist())}
                originals = []
            else:
                raise ValueError("Unique lazy augmentation requires the source dataset.")
            augmentor = Augmentor(build_unique_input_augments(
                inputs, self.task_colors.get(task_id, ()),
                protected_colors=self.task_output_only_colors.get(task_id, ()),
                original_inputs=originals, max_augments=self.max_augments, seed=self.rearc_seed,
                mapping_dtype=torch.uint8,
            ), max_augments=self.max_augments)
            self._unique_augmentors[task_id] = augmentor
        return augmentor

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if self.regular_augmentor is not None:
            self.regular_augmentor.set_enabled(enabled)

    def set_shared_epoch_buffer(self, shared_epoch: Optional[torch.Tensor]) -> None:
        self._shared_epoch = shared_epoch
        if shared_epoch is not None:
            shared_epoch.fill_(int(self._epoch))
        if self.regular_augmentor is not None:
            self.regular_augmentor.set_shared_epoch_buffer(shared_epoch)

    def _current_epoch(self) -> int:
        if self._shared_epoch is not None:
            return int(self._shared_epoch.item())
        return self._epoch

    def set_epoch(self, epoch: int) -> None:
        self._epoch = max(0, int(epoch))
        if self._shared_epoch is not None:
            self._shared_epoch.fill_(int(self._epoch))
        if self.regular_augmentor is not None:
            self.regular_augmentor.set_epoch(epoch)

    def _select_nvarc_mapping(
        self,
        example: SequenceExample,
        epoch: int,
    ) -> Optional[torch.Tensor]:
        colors, destinations = color_augmentation_domain(
            self.task_colors.get(example.task_id, ()),
            self.task_output_only_colors.get(example.task_id, ()),
            self.augmentation_sampling_mode,
        )
        if not colors:
            return None
        mapping_seed = _derive_order_seed(
            self.rearc_seed,
            (example.task_id, "nvarc_color", example.pair_index),
        )
        return random_color_mapping(colors, _cycle_seed(mapping_seed, 5, epoch), destination_colors=destinations)

    def _rearc_augments(self, task_id: str) -> Augments:
        augments = self._rearc_augments_by_task.get(task_id)
        if augments is None:
            schedule_seed = _derive_order_seed(
                self.rearc_seed, (task_id, "rearc_augmentation", 0)
            )
            colors, destinations = color_augmentation_domain(
                self.task_colors.get(task_id, ()), self.task_output_only_colors.get(task_id, ()),
                self.augmentation_sampling_mode,
            )
            mappings, dihedrals = build_augmentation_schedule(
                colors, self.max_augments, schedule_seed, destination_colors=destinations
            )
            augments = Augments(
                color_maps=mappings,
                dihedral_indices=dihedrals,
                color_map_indices=list(range(len(dihedrals))),
            )
            self._rearc_augments_by_task[task_id] = augments
        return augments

    def _select_rearc_index(self, task_id: str, pair_index: int, epoch: int) -> int:
        augments = self._rearc_augments(task_id)
        order_seed = _derive_order_seed(
            self.rearc_seed, (task_id, "train", int(pair_index))
        )
        return _index_for_epoch(
            order_seed, list(range(len(augments.dihedral_indices))), epoch, 3
        )

    def select_lazy_dihedral(
        self, task_id: str, pair_index: int, epoch: int, *, is_nvarc: bool
    ) -> int:
        if self.augmentation_sampling_mode == "single_grid_prefix":
            augments = self._unique_augmentor(task_id).augments_by_key[(task_id, "train", pair_index)]
            return augments.dihedral_indices[_select_augment_index(augments, augments.order_seed, epoch)]
        if is_nvarc:
            order_seed = _derive_order_seed(
                self.rearc_seed, (task_id, "train", int(pair_index))
            )
            return _index_for_epoch(order_seed, list(range(8)), epoch, 2)
        augments = self._rearc_augments(task_id)
        return augments.dihedral_indices[
            self._select_rearc_index(task_id, pair_index, epoch)
        ]

    def select_for_example(
        self, example: SequenceExample
    ) -> Tuple[Optional[torch.Tensor], Optional[int]]:
        if not self._enabled:
            return None, None
        effective_epoch = getattr(example, "augment_epoch_override", None)
        if effective_epoch is None:
            effective_epoch = self._current_epoch()
        preselected = getattr(example, "preselected_dihedral_index", None)
        if self.augmentation_sampling_mode == "single_grid_prefix" and (example.is_rearc or example.is_nvarc):
            augmentor = self._unique_augmentor(example.task_id, example)
            augmentor.set_epoch(effective_epoch)
            return augmentor.select_for_example(example)
        if example.is_rearc:
            augments = self._rearc_augments(example.task_id)
            order_seed = _derive_order_seed(
                self.rearc_seed,
                (example.task_id, "train", int(example.pair_index)),
            )
            idx = _select_augment_index(
                augments,
                order_seed,
                effective_epoch,
                preselected,
                getattr(example, "dihedral_occurrence_override", None),
            )
            return (
                augments.color_maps[idx],
                augments.dihedral_indices[idx]
                if preselected is None
                else int(preselected),
            )
        if example.is_nvarc:
            key = (example.task_id, example.split, example.pair_index)
            order_seed = _derive_order_seed(self.rearc_seed, key)
            dihedral_idx = (
                _index_for_epoch(
                    order_seed,
                    list(range(8)),
                    effective_epoch,
                    2,
                )
                if preselected is None
                else int(preselected)
            )
            return self._select_nvarc_mapping(example, effective_epoch), dihedral_idx
        if self.regular_augmentor is not None:
            return self.regular_augmentor.select_for_example(example)
        return None, None


def build_augmentor(
    dataset: Iterable[SequenceExample],
    task_colors: Dict[str, Sequence[int]],
    *,
    max_augments: int,
    seed: int,
    augmentation_sampling_mode: Optional[str] = None,
    task_output_only_colors: Optional[Dict[str, Sequence[int]]] = None,
    task_inputs: Optional[Dict[str, Sequence[Sequence[Sequence[int]]]]] = None,
) -> Augmentor:
    """Build exact, seed-reproducible color/dihedral attempt schedules."""
    attempt_count = max(1, int(max_augments))
    augments_by_key: Dict[Tuple[str, str, int], Augments] = {}
    keys_by_task: Dict[str, List[Tuple[str, str, int]]] = {}
    inputs_by_task: Dict[str, Dict[Tuple[str, str, int], Sequence[Sequence[int]]]] = {}
    unique_inputs = normalize_augmentation_sampling_mode(augmentation_sampling_mode) == "single_grid_prefix"
    for example in dataset:
        key = (example.task_id, example.split, example.pair_index)
        keys_by_task.setdefault(example.task_id, []).append(key)
        if unique_inputs:
            inputs_by_task.setdefault(example.task_id, {})[key] = tokens_to_grid(
                example.tokens[1:example.sep_index].tolist()
            )

    for task_id in sorted(keys_by_task):
        if unique_inputs:
            augments_by_key.update(build_unique_input_augments(
                inputs_by_task[task_id], task_colors.get(task_id, ()),
                protected_colors=(task_output_only_colors or {}).get(task_id, ()),
                original_inputs=(task_inputs or {}).get(task_id, ()),
                max_augments=attempt_count, seed=seed,
            ))
            continue
        pair_keys = sorted(set(keys_by_task[task_id]))
        colors, destinations = color_augmentation_domain(
            task_colors.get(task_id, ()), (task_output_only_colors or {}).get(task_id, ()),
            augmentation_sampling_mode,
        )
        color_schedules = build_balanced_color_schedules(
            colors, pair_keys, attempt_count, seed, destination_colors=destinations
        )
        for key in pair_keys:
            schedule_seed = _derive_order_seed(
                seed, (task_id, f"{key[1]}:{key[2]}", 0)
            )
            _, dihedrals = build_augmentation_schedule(
                task_colors.get(task_id, ()), attempt_count, schedule_seed
            )
            if len(dihedrals) < attempt_count:
                dihedrals = (dihedrals * math.ceil(attempt_count / len(dihedrals)))[
                    :attempt_count
                ]
            augments_by_key[key] = Augments(
                color_maps=color_schedules[key],
                dihedral_indices=dihedrals,
                color_map_indices=list(range(attempt_count)),
                order_seed=_derive_order_seed(seed, key),
            )

    return Augmentor(augments_by_key, max_augments=attempt_count)
