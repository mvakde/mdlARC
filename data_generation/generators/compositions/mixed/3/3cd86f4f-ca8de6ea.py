"""Clockwise ring interleaving followed by a bottom-anchored row shear.

Only NumPy and the standard library are used. Importing writes nothing; the
bounded self-test also keeps all puzzles and diagnostics in memory.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json

import numpy as np


# Within each ring the order is NW, NE, SE, SW, then a separate center.
_PERIMETER = ((0, 0), (0, 1), (0, 2), (1, 2),
              (2, 2), (2, 1), (2, 0), (1, 0))
_FINAL_SLOTS = ((0, 2), (0, 4), (2, 2), (2, 0),
                (0, 3), (1, 3), (2, 1), (1, 1), (1, 2))


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is outside its permitted range")
    return value


def _check_counts(num_train, num_test):
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    return nt, nq


def _grid(value):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError("grid must be a rectangular integer array") from error
    if array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape):
        raise ValueError("grid dimensions must be 1..30")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("grid cells must be integers")
    if np.any((array < 0) | (array > 9)):
        raise ValueError("grid colors must be 0..9")
    return array


def _coordinates(geometry):
    oy, ox, iy, ix, top, bottom, left, right = geometry
    cy, cx = top + oy, left + ox
    return ((cy - oy, cx - ox), (cy - oy, cx + ox),
            (cy + oy, cx + ox), (cy + oy, cx - ox),
            (cy - iy, cx - ix), (cy - iy, cx + ix),
            (cy + iy, cx + ix), (cy + iy, cx - ix), (cy, cx))


def _decode(input_grid):
    """Recover all geometry and values from the visible input, without state."""
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel().astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    if int(counts[background]) * 2 <= inp.size:
        raise ValueError("the uniform background must be a strict majority")
    points = np.argwhere(inp != background)
    if points.shape != (9, 2):
        raise ValueError("exactly nine foreground points are required")
    ys = sorted(set(int(v) for v in points[:, 0]))
    xs = sorted(set(int(v) for v in points[:, 1]))
    if len(ys) != 5 or len(xs) != 5:
        raise ValueError("the rings must occupy five row and column levels")
    cy, cx = ys[2], xs[2]
    oy, ox, iy, ix = cy - ys[0], cx - xs[0], cy - ys[1], cx - xs[1]
    if not (0 < iy < oy and 0 < ix < ox):
        raise ValueError("ring radii must be positive and strictly nested")
    if (ys != [cy - oy, cy - iy, cy, cy + iy, cy + oy]
            or xs != [cx - ox, cx - ix, cx, cx + ix, cx + ox]):
        raise ValueError("both rings must have the same center")
    geometry = (oy, ox, iy, ix, ys[0], inp.shape[0] - 1 - ys[-1],
                xs[0], inp.shape[1] - 1 - xs[-1])
    coords = _coordinates(geometry)
    if set(map(tuple, points.tolist())) != set(coords):
        raise ValueError("foreground must be exactly two corner rings and a center")
    return geometry, background, tuple(int(inp[r, c]) for r, c in coords)


def _compact(points):
    compact = np.empty((3, 3), dtype=np.int8)
    for quadrant in range(4):
        compact[_PERIMETER[2 * quadrant]] = points[quadrant]
        compact[_PERIMETER[2 * quadrant + 1]] = points[4 + quadrant]
    compact[1, 1] = points[8]
    return compact


def _shear(compact, background):
    height, width = compact.shape
    out = np.full((height, width + height - 1), background, dtype=np.int8)
    for row in range(height):
        start = height - 1 - row
        out[row, start:start + width] = compact[row]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Interleave the observed rings, place center, then shear all three rows."""
    _, background, points = _decode(input_grid)
    return _shear(_compact(points), background)


def _geometry(rng, rectangular, low, high):
    oy = int(rng.integers(low, high + 1))
    iy = int(rng.integers(1, oy))
    if rectangular:
        # Construct a different horizontal (outer, inner) pair without retries.
        choices = [(o, i) for o in range(low, high + 1)
                   for i in range(1, o) if (o, i) != (oy, iy)]
        ox, ix = choices[int(rng.integers(len(choices)))]
    else:
        ox, ix = oy, iy
    # At most six cells per side, and never beyond a 30-cell canvas axis.
    def margins(outer):
        budget = 29 - 2 * outer
        before = int(rng.integers(0, min(6, budget) + 1))
        after = int(rng.integers(0, min(6, budget - before) + 1))
        return before, after
    top, bottom = margins(oy)
    left, right = margins(ox)
    return (oy, ox, iy, ix, top, bottom, left, right)


def _point_colors(rng, background, mode):
    palette = rng.permutation([c for c in range(10) if c != background])
    if mode == "distinct":
        return tuple(int(v) for v in palette)
    if mode == "diagonal":
        a, b, c, d, center = map(int, palette[:5])
        return a, b, a, b, c, d, c, d, center
    if mode == "ring":
        a, b, c = map(int, palette[:3])
        return a, a, a, a, b, b, b, b, c
    if mode == "uniform":
        return (int(palette[0]),) * 9
    if mode == "reuse":
        # Explicit color sharing across radii and center, preserving geometry.
        a, b, c, d, e = map(int, palette[:5])
        return a, b, c, d, a, e, b, d, b
    count = int(rng.integers(2, 10))
    values = list(map(int, palette[:count]))
    values += [int(rng.choice(palette[:count])) for _ in range(9 - count)]
    return tuple(int(v) for v in rng.permutation(values))


def _pair(geometry, background, points):
    oy, ox, iy, ix, top, bottom, left, right = geometry
    inp = np.full((2 * oy + 1 + top + bottom,
                   2 * ox + 1 + left + right), background, dtype=np.int8)
    for position, color in zip(_coordinates(geometry), points):
        inp[position] = color
    # Ground truth is built directly, separately from solve's two-stage path.
    out = np.full((3, 5), background, dtype=np.int8)
    for slot, color in zip(_FINAL_SLOTS, points):
        out[slot] = color
    return {"input": inp, "output": out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed", 0)
    nt, nq = _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    rectangles = bool(rng.integers(2))
    puzzle = {"train": [], "test": []}
    used_geometry = set()
    first_bg = int(rng.integers(10))
    modes = ("distinct", "diagonal", "ring", "uniform", "reuse", "partial")
    train_extent = 0
    first_radii = None
    for index in range(nt + nq):
        training = index < nt
        if index == 0:
            rectangular, mode, low, high = False, "distinct", 2, 3
        elif index == 1:
            rectangular = rectangles
            mode, low, high = "reuse", 3, 7
        else:
            rectangular = rectangles and bool(rng.integers(2))
            mode = modes[int(rng.integers(len(modes)))]
            low, high = (2, 7) if training else (2, 14)
        # First test extrapolates ring scale beyond every training ring.
        if index == nt:
            low, high = train_extent + 1, 14
            mode = ("distinct", "diagonal", "ring", "reuse")[int(rng.integers(4))]
        background = first_bg if index == 0 else int(rng.integers(10))
        if index == 1:
            background = (first_bg + int(rng.integers(1, 10))) % 10
        geometry = None
        for _ in range(64):
            candidate = _geometry(rng, rectangular, low, high)
            if index == 1 and candidate[:4] == first_radii:
                continue
            if candidate not in used_geometry:
                geometry = candidate
                break
        if geometry is None:
            raise RuntimeError("geometry sampling exhausted its bounded retry budget")
        used_geometry.add(geometry)
        if index == 0:
            first_radii = geometry[:4]
        points = _point_colors(rng, background, mode)
        pair = _pair(geometry, background, points)
        puzzle["train" if training else "test"].append(pair)
        if training:
            train_extent = max(train_extent, *geometry[:2])
    validate_generated_puzzle(puzzle)
    return puzzle


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _digest(grid):
    array = np.asarray(grid, dtype=np.int8)
    return hashlib.sha256(bytes(array.shape) + array.tobytes()).hexdigest()


def _without_interleave(inp, background):
    # A concrete competing rule: pack visible points in raster order, then shear.
    return _shear(inp[inp != background].reshape(3, 3), background)


def validate_generated_puzzle(puzzle) -> None:
    """Check spatial conservation, exact transfer, teaching cases and shift."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, features = set(), {"train": [], "test": []}
    witnesses = {"interleave": False, "shear": False}
    first_test_interleave = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pairs must have exactly input and output")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            geometry, bg, points = _decode(inp)
            key = _digest(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(out.shape == (3, 5), "output must be 3 by 5")
            mask = np.zeros((3, 5), dtype=bool)
            for slot, color in zip(_FINAL_SLOTS, points):
                mask[slot] = True
                _require(int(out[slot]) == color, "incorrect positional transfer")
            _require(np.all(out[~mask] == bg), "all six padding cells use input background")
            _require(np.array_equal(out != bg, mask), "nine points must all remain visible")
            _require(Counter(map(int, inp[inp != bg])) == Counter(map(int, out[mask])),
                     "foreground multiplicities must be conserved")
            compact = _compact(points)
            for row in range(3):
                start = 2 - row
                _require(np.array_equal(out[row, start:start + 3], compact[row]),
                         "whole compact rows must be preserved by shear")
            _require(int(out[1, 2]) == points[8], "center must move with middle row")
            if split == "train":
                witnesses["interleave"] |= not np.array_equal(
                    out, _without_interleave(inp, bg))
                unshifted = np.full((3, 5), bg, dtype=np.int8)
                unshifted[:, :3] = compact
                witnesses["shear"] |= not np.array_equal(out, unshifted)
            elif not features["test"]:
                first_test_interleave = not np.array_equal(out, _without_interleave(inp, bg))
            features[split].append((geometry, bg, points))
    train, test = features["train"], features["test"]
    _require(any(len(set(p)) == 9 for _, _, p in train),
             "training must expose every individual correspondence")
    _require(len({bg for _, bg, _ in train}) >= 2, "training must vary visible background")
    _require(len({g[:4] for g, _, _ in train}) >= 2, "training must vary ring spacing")
    has_rectangle = lambda g: g[0] != g[1] or g[2] != g[3]
    _require(not any(has_rectangle(g) for g, _, _ in test)
             or any(has_rectangle(g) for g, _, _ in train),
             "rectangular tests require a rectangular demonstration")
    train_geometries = {g for g, _, _ in train}
    _require(all(g not in train_geometries for g, _, _ in test),
             "test geometry combinations must be new")
    _require(max(test[0][0][:2]) > max(max(g[:2]) for g, _, _ in train),
             "first test must extrapolate radial scale")
    _require(all(witnesses.values()), "both inherited mechanisms need training witnesses")
    _require(first_test_interleave, "first test must require interleaving as well as shear")


def _puzzle_digest(puzzle):
    return tuple(_digest(pair[role]) for split in ("train", "test")
                 for pair in puzzle[split] for role in ("input", "output"))


def _normalized_digest(grid):
    roles = {}
    norm = np.array([roles.setdefault(int(v), len(roles)) for v in grid.flat],
                    dtype=np.int8).reshape(grid.shape)
    return _digest(norm)


def self_test():
    """42 seeds, all legal counts, spatial/color covariance; no files saved."""
    shapes, geometries, structures, backgrounds, palette_sizes = set(), set(), set(), set(), set()
    digests = set()
    counts = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_puzzle_digest(puzzle))
        # Learn the nine-slot transfer only from the distinct-color demonstration.
        first = puzzle["train"][0]
        _, first_bg, first_points = _decode(first["input"])
        source = {c: i for i, c in enumerate(first_points)}
        learned_slots = {source[int(first["output"][r, c])]: (r, c)
                         for r, c in np.argwhere(first["output"] != first_bg)}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                g, bg, points = _decode(inp)
                geometries.add(g)
                shapes.add(inp.shape)
                structures.add(_normalized_digest(inp))
                backgrounds.add(bg)
                palette_sizes.add(len(set(points)))
                counts["pairs"] += 1
                counts["rectangular"] += g[0] != g[1] or g[2] != g[3]
                counts["outer_gap"] += g[0] - g[2] > 1 or g[1] - g[3] > 1
                counts["inner_gap"] += max(g[2:4]) > 1
                counts["uneven_margins"] += len(set(g[4:])) > 1
                counts["color_reuse"] += len(set(points)) < 9
                counts["boundary30"] += 30 in inp.shape
                counts["uniform"] += len(set(points)) == 1
                counts["zero_foreground"] += bg != 0 and 0 in points
                prediction = np.full((3, 5), bg, dtype=np.int8)
                for i, slot in learned_slots.items():
                    prediction[slot] = points[i]
                _require(np.array_equal(prediction, out), "learned transfer mismatch")
                _require(np.array_equal(solve(inp.copy()), out), "solve mismatch")
        # Translation and arbitrary global recoloring must preserve the rule.
        small = first["input"]
        translated = np.pad(small, ((1, 2), (2, 1)), constant_values=first_bg)
        _require(np.array_equal(solve(translated), first["output"]),
                 "translation invariance failed")
        permutation = np.random.default_rng(seed + 10000).permutation(10)
        _require(np.array_equal(solve(permutation[small]), permutation[first["output"]]),
                 "color covariance failed")
    _require(len(digests) == 42, "seeds did not produce 42 distinct puzzles")
    _require(len(shapes) >= 80 and len(geometries) >= 150 and len(structures) >= 150,
             "insufficient structural variation")
    _require(len(backgrounds) == 10 and len(palette_sizes) >= 6,
             "insufficient palette-role variation")
    _require(min(counts[k] for k in ("rectangular", "outer_gap", "inner_gap",
                                    "uneven_margins", "color_reuse")) >= 20,
             "structural axes underrepresented")
    _require(counts["uniform"] >= 3 and counts["boundary30"] >= 3
             and counts["zero_foreground"] >= 20, "simple or boundary cases missing")
    legal_calls = 0
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1000 + nt * 10 + nq, nt, nq)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (nt, nq),
                     "requested counts were not honored")
            legal_calls += 1
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _require(_puzzle_digest(generate_puzzle(84839914, 10, 4))
             == _puzzle_digest(generate_puzzle(84839914, 10, 4)), "nondeterministic")
    # Reject lost points / incorrect outputs, so validation is not a no-op.
    bad = generate_puzzle(0, 2, 1)
    bad["train"][0]["output"] = bad["train"][0]["output"].copy()
    bad["train"][0]["output"][0, 2] = int(bad["train"][0]["output"][0, 0])
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a missing foreground point")
    return {"seeds": 42, "default_pairs": counts["pairs"],
            "distinct_puzzles": len(digests), "input_shapes": len(shapes),
            "geometries": len(geometries), "normalized_inputs": len(structures),
            "backgrounds": sorted(backgrounds), "foreground_palette_sizes": sorted(palette_sizes),
            "coverage": dict(counts), "legal_count_combinations": legal_calls,
            "independent_solver": False}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
