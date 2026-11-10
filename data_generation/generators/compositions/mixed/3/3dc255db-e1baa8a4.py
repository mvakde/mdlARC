"""Literal implementation of proposal.json's count/expand/collapse idea.

Important semantic limitation: exact adjacent-vector collapse cancels every
positive duplicate-vector expansion. A's count controls the intermediate grid,
but cannot be necessary for the final answer. See composition.md. The tests
measure this limitation instead of claiming that the A ablation changes labels.
No parent module, fixture, seed, or output is consulted by solve().
"""
from __future__ import annotations

import hashlib

import numpy as np


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not np.all((0 <= grid) & (grid <= 9))):
        raise ValueError("a grid must be a 1..30 by 1..30 integer array in 0..9")
    return grid.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    cell = (r + dr, c + dc)
                    if cell in remaining:
                        remaining.remove(cell)
                        found.add(cell)
                        stack.append(cell)
        yield found


def _upward_geometry(mask):
    """Recognize a rectangular ring plus an inward-aligned outward tab.

    Entire rows/columns may already be repeated with unequal multiplicities.
    Thus the tab, border, and hollow can each be several cells thick.
    """
    h, w = mask.shape
    if h < 4 or w < 3:
        return None
    full = np.all(mask, axis=1)
    indices = np.flatnonzero(full)
    if not len(indices) or indices[0] == 0:
        return None
    top = int(indices[0])
    middle = top
    while middle < h and full[middle]:
        middle += 1
    bottom = middle
    while bottom < h and not full[bottom]:
        bottom += 1
    if middle == h or bottom == h or not np.all(full[bottom:]):
        return None
    tab = np.flatnonzero(mask[0])
    if (not len(tab) or tab[0] == 0 or tab[-1] == w - 1
            or len(tab) != tab[-1] - tab[0] + 1
            or not np.all(mask[:top] == mask[0])):
        return None
    hole = np.flatnonzero(~mask[middle])
    if (not len(hole) or hole[0] == 0 or hole[-1] == w - 1
            or len(hole) != hole[-1] - hole[0] + 1
            or not np.all(mask[middle:bottom] == mask[middle])):
        return None
    return middle, bottom, int(hole[0]), int(hole[-1] + 1), int(tab[0])


def _read_frames(grid):
    background = int(grid[0, 0])
    if not (np.all(grid[0] == background) and np.all(grid[-1] == background)
            and np.all(grid[:, 0] == background)
            and np.all(grid[:, -1] == background)):
        raise ValueError("the outer perimeter must show the background")
    frames = []
    for color in (int(c) for c in np.unique(grid) if c != background):
        for cells in _components(grid == color):
            if len(cells) < 9:
                continue
            rs, cs = zip(*cells)
            r0, c0 = min(rs), min(cs)
            crop = grid[r0:max(rs) + 1, c0:max(cs) + 1]
            mask = np.zeros(crop.shape, dtype=bool)
            for r, c in cells:
                mask[r - r0, c - c0] = True
            identifiers = np.arange(crop.size).reshape(crop.shape)
            candidates = []
            for rotation in range(4):
                geometry = _upward_geometry(np.rot90(mask, rotation))
                if geometry is None:
                    continue
                a, b, left, right, tab = geometry
                oriented = np.rot90(crop, rotation)
                cavity = oriented[a:b, left:right]
                values = set(map(int, np.unique(cavity))) - {background}
                if len(values) != 1 or color in values:
                    continue
                payload = next(iter(values))
                quantity = int(np.count_nonzero(cavity == payload))
                if quantity == cavity.size:
                    continue  # At least one visible hollow cell is required.
                mapping = np.rot90(identifiers, rotation)
                tip_id = int(mapping[0, tab])
                tr, tc = divmod(tip_id, crop.shape[1])
                payload_ids = mapping[a:b, left:right][cavity == payload]
                payload_cells = tuple((r0 + int(i) // crop.shape[1],
                                       c0 + int(i) % crop.shape[1])
                                      for i in payload_ids)
                candidates.append({"tip": (r0 + tr, c0 + tc),
                                   "axis": rotation % 2,
                                   "direction": "URDL"[rotation],
                                   "quantity": quantity,
                                   "frame_color": color,
                                   "payload_color": payload,
                                   "payload_cells": payload_cells,
                                   "bbox": (r0, c0, crop.shape[0], crop.shape[1])})
            if len(candidates) > 1:
                raise ValueError("ambiguous frame direction")
            frames.extend(candidates)
    if not frames:
        raise ValueError("no hollow payload frame")
    frames.sort(key=lambda f: f["tip"])
    return frames


def _expansion(grid, frames):
    """N total copies of the tip's full vector, clipped to a 30-cell axis.

    Shared-line requests take their maximum. The finite canvas budget is
    allocated in original line order; no original line is removed. Copies
    on either side of an identical vector are indistinguishable.
    """
    repeats = []
    for axis in (0, 1):
        demands = np.ones(grid.shape[axis], dtype=np.int64)
        for frame in frames:
            if frame["axis"] == axis:
                index = frame["tip"][axis]
                demands[index] = max(demands[index], frame["quantity"])
        counts = np.ones_like(demands)
        room = 30 - grid.shape[axis]
        for i, wanted in enumerate(demands):
            extra = min(int(wanted) - 1, room)
            counts[i] += extra
            room -= extra
        repeats.append(counts)
    return np.repeat(np.repeat(grid, repeats[0], axis=0), repeats[1], axis=1)


def _collapse(grid):
    keep_rows = np.r_[True, np.any(grid[1:] != grid[:-1], axis=1)]
    keep_columns = np.r_[True, np.any(grid[:, 1:] != grid[:, :-1], axis=0)]
    return grid[keep_rows][:, keep_columns].copy()


def _reference_collapse(grid):
    """A simple list-based check of the equality-run operation."""
    rows = []
    for row in grid.tolist():
        if not rows or row != rows[-1]:
            rows.append(row)
    columns = []
    for column in zip(*rows):
        if not columns or column != columns[-1]:
            columns.append(column)
    return np.array(list(zip(*columns)), dtype=np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    frames = _read_frames(grid)
    return _collapse(_expansion(grid, frames))


def _frame_patch(rng, background, colors, rotation, quantity=None, anchor=False):
    inner_h = 3 if anchor else int(rng.integers(1, 5))
    inner_w = 3 if anchor else int(rng.integers(2, 6))
    stem = 1 if anchor else int(rng.integers(1, 3))
    width = inner_w + 2
    tab = 2 if anchor else int(rng.integers(1, width - 1))
    patch = np.full((inner_h + 2 + stem, width), background, dtype=np.int8)
    frame, payload = colors
    patch[stem, :] = frame
    patch[-1, :] = frame
    patch[stem:, 0] = frame
    patch[stem:, -1] = frame
    patch[:stem, tab] = frame
    capacity = inner_h * inner_w
    q = quantity if quantity is not None else int(rng.integers(1, min(8, capacity)))
    if not 1 <= q < capacity:
        raise ValueError("payload does not fit")
    # Both scattered and connected-looking arrangements occur; only cells count.
    candidates = [(stem + r, c) for r in range(1, inner_h + 1)
                  for c in range(1, inner_w + 1)]
    if rng.random() < .4:
        pivot = candidates[int(rng.integers(len(candidates)))]
        candidates.sort(key=lambda p: (abs(p[0] - pivot[0]) + abs(p[1] - pivot[1]), p))
        selected = candidates[:q]
    else:
        selected = [candidates[int(i)] for i in rng.choice(capacity, q, replace=False)]
    for p in selected:
        patch[p] = payload
    return np.rot90(patch, -rotation).copy()


def _pack(rng, patches):
    for _ in range(8):
        h, w = (int(x) for x in rng.integers(11, 25, size=2))
        positions, blocked = [], np.zeros((h, w), dtype=bool)
        for patch in patches:
            ph, pw = patch.shape
            choices = [(r, c) for r in range(1, h - ph)
                       for c in range(1, w - pw)
                       if not np.any(blocked[r:r + ph, c:c + pw])]
            if not choices:
                break
            r, c = choices[int(rng.integers(len(choices)))]
            positions.append((r, c))
            blocked[r - 1:r + ph + 1, c - 1:c + pw + 1] = True
        if len(positions) == len(patches):
            return (h, w), positions, blocked
    # Three patches of maximum width eight fit in a 28-column shelf.
    h = max(p.shape[0] for p in patches) + 2
    w = sum(p.shape[1] for p in patches) + len(patches) + 1
    positions, blocked, c = [], np.zeros((h, w), dtype=bool), 1
    for patch in patches:
        ph, pw = patch.shape
        positions.append((1, c))
        blocked[0:ph + 2, c - 1:c + pw + 1] = True
        c += pw + 1
    return (h, w), positions, blocked


def _repeats(rng, length, expand):
    repeats = np.ones(length, dtype=np.int64)
    if expand:
        budget = int(rng.integers(2, min(12, 30 - length) + 1))
        # Leave some logical lines thin; variable thickness is visible.
        indices = rng.choice(length, int(rng.integers(1, min(4, length - 1) + 1)),
                             replace=False)
        for _ in range(budget):
            repeats[int(rng.choice(indices))] += 1
    return repeats


def _make_pair(rng, count, anchor, force_both=False):
    excluded = set(anchor["colors"]) if anchor else set()
    background = int(rng.choice([c for c in range(10) if c not in excluded]))
    palette = [c for c in range(10) if c != background]
    patches = []
    for i in range(count):
        if i == 0 and anchor:
            colors, rotation = anchor["colors"], anchor["rotation"]
            patch = _frame_patch(rng, background, colors, rotation,
                                 anchor["quantity"], anchor=True)
        else:
            colors = tuple(int(x) for x in rng.choice(palette, 2, replace=False))
            if i and rng.random() < .4:
                first = _read_frames(np.pad(patches[0], 1, constant_values=background))[0]
                colors = (first["payload_color"], first["frame_color"])
            patch = _frame_patch(rng, background, colors, int(rng.integers(4)))
        patches.append(patch)
    shape, positions, blocked = _pack(rng, patches)
    base = np.full(shape, background, dtype=np.int8)
    for patch, (r, c) in zip(patches, positions):
        ph, pw = patch.shape
        base[r:r + ph, c:c + pw] = patch
    # Isolated exterior marks are part of the vectors and are preserved.
    choices = [(r, c) for r in range(1, shape[0] - 1)
               for c in range(1, shape[1] - 1) if not blocked[r, c]]
    if choices and rng.random() < .45:
        cell = choices[int(rng.integers(len(choices)))]
        base[cell] = int(rng.choice(palette))
    operation = 0 if force_both else int(rng.choice(3, p=[.6, .2, .2]))
    rows = _repeats(rng, shape[0], operation != 2)
    columns = _repeats(rng, shape[1], operation != 1)
    inp = np.repeat(np.repeat(base, rows, axis=0), columns, axis=1)
    # Construct labels from the base, separately from the input-only reader.
    out = _reference_collapse(base)
    assert len(_read_frames(inp)) == count, "a frame was lost during enlargement"
    return {"input": inp, "output": out}


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= value <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    colors = tuple(int(x) for x in rng.choice(10, 2, replace=False))
    direction = int(rng.integers(4))
    puzzle, seen, first_quantities = {"train": [], "test": []}, set(), set()
    for split, size in (("train", num_train), ("test", num_test)):
        for index in range(size):
            anchor = ({"colors": colors, "rotation": direction,
                       "quantity": 2 if index == 0 else 5}
                      if split == "train" and index < 2 else None)
            count = (1 if index == 0 else 2) if anchor else int(rng.integers(1, 3))
            if split == "test":
                count = 3 if index == 0 else int(rng.integers(1, 4))
            for _ in range(32):
                pair = _make_pair(rng, count, anchor,
                                  force_both=split == "train" and index == 0)
                quantities = {f["quantity"] for f in _read_frames(pair["input"])}
                if _key(pair["input"]) in seen:
                    continue
                if split == "train" and index == 1 and not quantities - first_quantities:
                    continue
                break
            else:
                raise RuntimeError("bounded distinct-example construction exhausted")
            if split == "train" and index == 0:
                first_quantities = quantities
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    n, m = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen, training_counts, object_counts = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert _key(inp) not in seen, "duplicate input"
            seen.add(_key(inp))
            frames = _read_frames(inp)
            assert 1 <= len(frames) <= 3
            object_counts[split].append(len(frames))
            for frame in frames:
                assert len(frame["payload_cells"]) == frame["quantity"] >= 1
                assert all(inp[p] == frame["payload_color"] for p in frame["payload_cells"])
            assert np.array_equal(solve(inp), out), "literal proposal mismatch"
            assert np.array_equal(_reference_collapse(inp), out), "duplicate-vector inverse failed"
            assert np.array_equal(np.unique(inp), np.unique(out)), "symbol was changed/lost"
            assert np.array_equal(_collapse(out), out), "output is not compact"
            assert inp.shape != out.shape, "no compression evidence"
            assert np.array_equal(_reference_collapse(_expansion(inp, frames)), out)
            if split == "train":
                training_counts.update(f["quantity"] for f in frames)
    assert len(training_counts) >= 2, "payload quantities do not vary"
    assert max(object_counts["test"]) > max(object_counts["train"]), "missing object-count shift"
    first = np.asarray(puzzle["train"][0]["input"])
    target = np.asarray(puzzle["train"][0]["output"])
    assert all(a > b for a, b in zip(first.shape, target.shape)), "both-axis demonstration absent"


def self_test():
    """42 puzzles, streamed; no files, traces, subprocesses, or retained runs."""
    stats = {"seeds": 42, "pairs": 0, "a_omission_changes": 0,
             "b_omission_changes": 0, "expansion_changes": 0}
    puzzles, shapes, palettes, structures, quantities, directions = set(), set(), set(), set(), set(), set()
    frame_shapes, row_schedules, column_schedules = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                digest.update(repr(_key(inp)).encode())
                frames = _read_frames(inp)
                expanded = _expansion(inp, frames)
                before = inp.copy()
                assert np.array_equal(solve(inp), out) and np.array_equal(inp, before)
                # This is intentionally the opposite of a successful A necessity test.
                stats["a_omission_changes"] += not np.array_equal(_collapse(inp), out)
                stats["b_omission_changes"] += not np.array_equal(expanded, out)
                stats["expansion_changes"] += expanded.shape != inp.shape
                stats["pairs"] += 1
                shapes.add((inp.shape, out.shape))
                palettes.add(tuple(map(int, np.unique(inp))))
                role_map = {}
                normalized = bytes(role_map.setdefault(int(c), len(role_map)) for c in inp.flat)
                structures.add((inp.shape, hashlib.sha256(normalized).digest()))
                for axis, collection in ((0, row_schedules), (1, column_schedules)):
                    vectors = inp if axis == 0 else inp.T
                    lengths = []
                    for i, vector in enumerate(vectors):
                        if i == 0 or not np.array_equal(vector, vectors[i - 1]):
                            lengths.append(1)
                        else:
                            lengths[-1] += 1
                    collection.add(tuple(lengths))
                quantities.update(f["quantity"] for f in frames)
                directions.update(f["direction"] for f in frames)
                frame_shapes.update(f["bbox"][2:] for f in frames)
                # All positive whole-vector enlargements obey the cancellation law.
                rng = np.random.default_rng(seed + stats["pairs"])
                small = rng.integers(0, 10, size=(4, 5), dtype=np.int8)
                enlarged = np.repeat(np.repeat(small, rng.integers(1, 5, 4), 0),
                                     rng.integers(1, 5, 5), 1)
                assert np.array_equal(_collapse(enlarged), _collapse(small))
        puzzles.add(digest.digest())
    assert len(puzzles) == 42 and len(structures) >= 190
    assert len(shapes) >= 80 and len(palettes) >= 40 and len(frame_shapes) >= 30
    assert len(row_schedules) >= 70 and len(column_schedules) >= 70
    assert directions == set("URDL") and len(quantities) >= 15
    assert stats["a_omission_changes"] == 0
    assert stats["b_omission_changes"] == stats["pairs"]
    assert stats["expansion_changes"] >= 100
    repeated = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k])
               for s in ("train", "test")
               for a, b in zip(repeated[s], generate_puzzle(0)[s])
               for k in ("input", "output"))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + a * 10 + b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    stats.update({"distinct_puzzles": len(puzzles), "shape_pairs": len(shapes),
                  "palettes": len(palettes), "structures": len(structures),
                  "frame_shapes": len(frame_shapes), "quantities": sorted(quantities),
                  "directions": sorted(directions), "row_schedules": len(row_schedules),
                  "column_schedules": len(column_schedules)})
    return stats
