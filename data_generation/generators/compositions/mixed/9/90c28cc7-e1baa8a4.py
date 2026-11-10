"""Crop one filled mosaic, then compress consecutive identical whole lines.

The required proposal is intentionally preserved despite its historical novelty
rejection. No parent module, saved example, or generator state is used by solve.
All routines are bounded and perform no filesystem or network writes.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


# SHA-256(shape bytes, input uint8 bytes, shape bytes, output uint8 bytes).
# Only exclusion hashes of the 19 supplied original/selected parent pairs.
_PARENT_PAIRS = frozenset((
    "c67235a30ffc1b17b5ba3bfbce383d70255c509e133ec5b151f8c31432976abd",
    "8db74c77ca88d36d7843871e16bf12b2d85de07e537abc2c576031b72a64516d",
    "be4f21901417badbf7cab99c825bb2f88850055cbe1b766bdf570607db13767a",
    "c8d7e03bfa1200f183089fc4b03cc97f1ae6bea3ff764b550b3abd8f11dcb951",
    "7cb6a9e5ca5338279ea4b46dac6fe0944bdec9086769b0cf25884590e6f80780",
    "71bb59d530a1a301ede6ae1e456bdb7059c583f47dad9c82550d81580adff667",
    "b21aa7a7c837278a173003afc811902d44dc7a927ac8783553d348fc2dcbe4af",
    "36a5bec239ca794bdb7d1d78c4ed41363bcf84ee4a33e399f80aff67e907238a",
    "6c534c8dd6c34559e55a039fb1b87eb01891cdc53741de63e0b9c2cd7140631c",
    "429017261081060b386a9868a45fb825117205d862b1eaf991a9bf7db7503d05",
    "e95ec8829d68d4488b0114073d98c298044f2bff4aa7514820fb1446d83b1228",
    "6b587d3dd3c1e1f0a0fa32cb993b663c3044c59d6c2e7b0cb2026fb9b5949eb4",
    "bc1750a1c50fda83aed527d06b24a6d64e7d23fbe7894eb551180bfc70dff020",
    "2ff1211a591cdfea8ff4be4d6b04eb978e57d094d1cbaa8159eb3f4f30084e28",
    "dfec3010e25f312130b99442bc3a2cd2dafab0a4ec20587ecaacbe030c3b4b9f",
    "558dd423ff1a79494ab5b0c83e61291b7be259ef12a4dc205ea8d60e3ea3d92c",
    "83e36a5ee31343afbdb60569cf606a81605b4b4cf9f8057e6b12617ec8fc1831",
    "353e97556ad7a5a954a46e9f26f1e9731e1300d80100b0925ed0ef83e642f03f",
    "2bab617a961219938ad67406d74c9002773a59fca844065fc54193e73b779f90",
))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not all(1 <= n <= 30 for n in array.shape)
            or not np.issubdtype(array.dtype, np.integer)
            or not np.all((0 <= array) & (array <= 9))):
        raise ValueError("grid must be a 1..30 by 1..30 integer array with colors 0..9")
    return array


def _parts(input_grid):
    """Identify the background and tight crop using the input alone."""
    grid = _grid(input_grid)
    background = int(grid[0, 0])
    if not (grid[0, -1] == grid[-1, 0] == grid[-1, -1] == background):
        raise ValueError("all four corners must identify the same background")
    rows, columns = np.nonzero(grid != background)
    if rows.size == 0:
        raise ValueError("a non-background mosaic is required")
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    crop = grid[top:bottom, left:right]
    margins = (top, grid.shape[0] - bottom, left, grid.shape[1] - right)
    if np.any(crop == background) or sum(v == 0 for v in margins) > 1:
        raise ValueError("mosaic must be filled and may touch at most one canvas side")
    return crop, background, margins


def _starts(grid, axis):
    changed = (np.any(grid[1:, :] != grid[:-1, :], axis=1) if axis == 0
               else np.any(grid[:, 1:] != grid[:, :-1], axis=0))
    return np.r_[0, np.flatnonzero(changed) + 1]


def _compress(grid):
    return grid[np.ix_(_starts(grid, 0), _starts(grid, 1))].copy()


def _bands_from_grid(grid, axis):
    return tuple(int(n) for n in np.diff(np.r_[_starts(grid, axis), grid.shape[axis]]))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the tight mosaic with maximal identical full-line runs collapsed."""
    crop, _, _ = _parts(input_grid)
    return _compress(crop)


def _observable(grid):
    return all(len(_starts(grid, axis)) == grid.shape[axis] for axis in (0, 1))


def _repeat(grid, axis):
    lines = grid if axis == 0 else grid.T
    return any(np.array_equal(lines[i], lines[j])
               for i in range(len(lines)) for j in range(i + 2, len(lines)))


def _witness(grid, axis):
    """A boundary is visible in exactly one crossing logical line."""
    lines = grid if axis == 0 else grid.T
    return lines.shape[1] >= 2 and any(np.count_nonzero(a != b) == 1
                                      for a, b in zip(lines, lines[1:]))


def _latent(rng, shape, background, mode):
    # Witness-column mode is the transpose of the same structural construction.
    transpose = mode == "witness_columns"
    working_shape = shape[::-1] if transpose else shape
    h, w = working_shape
    available = np.array([c for c in range(10) if c != background], dtype=np.int8)
    max_colors = min(9, h * w)
    count = (2 if mode == "checker" and max_colors >= 2 else
             int(rng.integers(2, min(5, max_colors) + 1))
             if mode.startswith("witness") else
             int(rng.integers(1 if max_colors == 1 else 2, max_colors + 1)))
    colors = rng.choice(available, size=count, replace=False)
    for _ in range(96):
        grid = rng.choice(colors, size=(h, w)).astype(np.int8)
        if mode.startswith("witness"):
            a, c = int(rng.integers(h - 2)), int(rng.integers(w - 1))
            grid[a + 2] = grid[a]
            grid[:, c + 1] = grid[:, c]
            grid[a + 1, c + 1] = rng.choice(colors[colors != grid[a + 1, c]])
        elif mode == "checker":
            grid = colors[np.indices((h, w)).sum(axis=0) % len(colors)]
        elif mode == "sparse":
            grid.fill(colors[0])
            positions = rng.choice(h * w, size=max(h, w), replace=False)
            grid.flat[positions] = rng.choice(colors[1:], size=len(positions))
        elif mode == "patch" and min(h, w) >= 2:
            r, c = int(rng.integers(h - 1)), int(rng.integers(w - 1))
            grid[r:r + 2, c:c + 2] = colors[0]
        elif mode == "repeat_rows" and h >= 3:
            a = int(rng.integers(h - 2))
            grid[a + 2] = grid[a]
        elif mode == "repeat_columns" and w >= 3:
            a = int(rng.integers(w - 2))
            grid[:, a + 2] = grid[:, a]
        if _observable(grid):
            return (grid.T if transpose else grid).copy()
    # Deterministic bounded fallback, preserving the promised witness features.
    grid = colors[np.indices((h, w)).sum(axis=0) % min(2, len(colors))]
    if mode.startswith("witness"):
        grid[:, 1] = grid[:, 0]
        grid[1, 1] = colors[0] if grid[1, 0] != colors[0] else colors[1]
    return (grid.T if transpose else grid).copy()


def _sample_bands(rng, length, *, varied=False, wide=False, maximum=14):
    for _ in range(64):
        mode = "wide" if wide else str(rng.choice(("unit", "uniform", "random", "sorted", "wide"),
                                                   p=(.12, .16, .35, .17, .20)))
        if mode == "unit":
            values = np.ones(length, dtype=int)
        elif mode == "uniform":
            values = np.full(length, int(rng.integers(1, min(6, maximum, 28 // length) + 1)))
        elif mode == "wide" and maximum >= 9:
            values = rng.integers(1, 4, size=length)
            peak = int(rng.integers(length))
            values[peak] = int(rng.integers(9, maximum + 1))
            if length > 1:
                neighbors = [i for i in (peak - 1, peak + 1) if 0 <= i < length]
                values[int(rng.choice(neighbors))] = 1
        else:
            values = rng.integers(1, min(6, maximum) + 1, size=length)
            if mode == "sorted":
                values.sort()
                if rng.random() < .5:
                    values = values[::-1]
        if sum(values) <= 28 and (not varied or (max(values) > 1 and len(set(values)) > 1)):
            return tuple(int(x) for x in values)
    values = [1] * length
    values[0] = min(maximum, 28 - length + 1)
    return tuple(values)


def _expand(grid, rows, columns):
    return np.repeat(np.repeat(grid, rows, axis=0), columns, axis=1)


def _embed(rng, crop, background):
    margins = []
    for size in crop.shape:
        extra = int(rng.integers(2, min(10, 30 - size) + 1))
        first = int(rng.integers(1, extra))
        margins.extend((first, extra - first))
    if rng.random() < .35:
        side = int(rng.integers(4))
        opposite = side ^ 1
        margins[opposite] += margins[side]
        margins[side] = 0
    top, bottom, left, right = margins
    canvas = np.full((top + crop.shape[0] + bottom, left + crop.shape[1] + right),
                     background, dtype=np.int8)
    canvas[top:top + crop.shape[0], left:left + crop.shape[1]] = crop
    return canvas


def _signature(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.uint8).tobytes()


def _pair_digest(pair):
    a, b = (np.asarray(pair[key], dtype=np.uint8) for key in ("input", "output"))
    return hashlib.sha256(bytes(a.shape) + a.tobytes() + bytes(b.shape) + b.tobytes()).hexdigest()


def _puzzle_digest(puzzle):
    return hashlib.sha256("".join(_pair_digest(pair) for split in ("train", "test")
                                  for pair in puzzle[split]).encode()).hexdigest()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate a reproducible puzzle. Counts are explicit, with defaults 4/1."""
    train_count, test_count = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(("height", "width", "both", "thickness")))
    shared_background = int(rng.integers(10)) if rng.random() < .40 else None
    first = (int(rng.integers(3, 6)), int(rng.integers(2, 6)))
    choices = [(h, w) for h in range(2, 6) for w in range(3, 6) if (h, w) != first]
    second = choices[int(rng.integers(len(choices)))]
    shapes = [first, second]
    for _ in range(2, train_count):
        shape = (int(rng.integers(2, 7)), int(rng.integers(2, 7)))
        if rng.random() < .12:
            shape = (1, shape[1]) if rng.random() < .5 else (shape[0], 1)
        shapes.append(shape)
    max_h, max_w = (max(s[axis] for s in shapes) for axis in (0, 1))
    for _ in range(test_count):
        if shift == "thickness":
            shape = shapes[int(rng.integers(train_count))]
        else:
            shape = (int(rng.integers(max_h + 1, 9)) if shift in ("height", "both")
                     else int(rng.integers(2, max_h + 1)),
                     int(rng.integers(max_w + 1, 9)) if shift in ("width", "both")
                     else int(rng.integers(2, max_w + 1)))
        shapes.append(shape)
    puzzle = {"train": [], "test": []}
    used = set()
    first_background = None
    modes = ("random", "checker", "sparse", "patch", "repeat_rows", "repeat_columns")
    for index, shape in enumerate(shapes):
        split = "train" if index < train_count else "test"
        anchor = index < 2
        mode = ("witness_rows" if index == 0 else "witness_columns" if index == 1 else
                str(rng.choice(modes)))
        if min(shape) == 1 and mode in ("sparse", "patch"):
            mode = "random"
        for _ in range(32):
            background = (shared_background if shared_background is not None else
                          int(rng.choice([c for c in range(10) if c != first_background]))
                          if index == 1 else int(rng.integers(10)))
            latent = _latent(rng, shape, background, mode)
            maximum = 6 if shift == "thickness" and split == "train" else 14
            bands = [_sample_bands(rng, n, varied=anchor,
                                   wide=(shift == "thickness" and split == "test"),
                                   maximum=maximum) for n in shape]
            pair = {"input": _embed(rng, _expand(latent, *bands), background),
                    "output": latent.copy()}
            signature = _signature(pair["input"])
            if signature not in used and _pair_digest(pair) not in _PARENT_PAIRS:
                used.add(signature)
                puzzle[split].append(pair)
                if index == 0:
                    first_background = background
                break
        else:
            raise RuntimeError("bounded unique-pair attempts exhausted")
    # Demonstration roles do not become a visible fixed order.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    """Check schema, reconstruction, causal ablations and training evidence."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "train/test schema")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, testing = set(), [], []
    evidence = {k: False for k in ("rows", "columns", "row_witness", "column_witness", "uneven_both")}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            source, target = (_grid(pair[key]) for key in ("input", "output"))
            signature = _signature(source)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            _require(_pair_digest(pair) not in _PARENT_PAIRS, "exact parent pair")
            crop, background, _ = _parts(source)
            rows, columns = (_bands_from_grid(crop, axis) for axis in (0, 1))
            _require(np.array_equal(solve(source), target), "wrong output")
            _require(_observable(target), "adjacent equal output lines")
            _require(np.array_equal(_expand(target, rows, columns), crop), "not a lossless band expansion")
            _require(np.array_equal(np.unique(crop), np.unique(target)), "colors changed")
            _require(background not in target, "background leaked into output")
            _require(not np.array_equal(_compress(source), target), "crop ablation has no effect")
            info = (target.shape, max(rows + columns))
            if split == "train":
                training.append(info)
                evidence["rows"] |= _repeat(target, 0)
                evidence["columns"] |= _repeat(target, 1)
                evidence["row_witness"] |= _witness(target, 0)
                evidence["column_witness"] |= _witness(target, 1)
                varied = len(set(rows)) > 1 and len(set(columns)) > 1
                evidence["uneven_both"] |= varied and max(rows) > 1 and max(columns) > 1
                if varied:
                    _require(not np.array_equal(crop, target), "collapse ablation has no effect")
                    _require(not np.array_equal(crop[_starts(crop, 0)], target), "column ablation has no effect")
                    _require(not np.array_equal(crop[:, _starts(crop, 1)], target), "row ablation has no effect")
            else:
                testing.append(info)
    _require(all(evidence.values()), "training misses repetition, single-witness seams or uneven axes")
    _require(len({shape for shape, _ in training}) >= 2, "training needs shape variation")
    max_h, max_w = (max(shape[axis] for shape, _ in training) for axis in (0, 1))
    largest = max(size for _, size in training)
    _require(all(shape[0] > max_h or shape[1] > max_w or size > largest for shape, size in testing),
             "test needs a logical-size or thickness extrapolation")


def _normalized(grid):
    roles = {}
    return grid.shape, bytes(roles.setdefault(int(c), len(roles)) for c in grid.flat)


def self_test():
    """Bounded 42-seed variation checks; no files, caches or saved traces."""
    puzzles, shapes, canvases, structures, backgrounds, palettes, schedules = (set() for _ in range(7))
    counts = {k: 0 for k in ("pairs", "edge", "one_axis", "uniform_rows", "uniform_columns",
                              "rich", "two_colors", "wide_thin", "row_repeat", "column_repeat")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed)), "nondeterminism")
        puzzles.add(_puzzle_digest(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                crop, background, margins = _parts(pair["input"])
                target = pair["output"]
                rows, columns = (_bands_from_grid(crop, axis) for axis in (0, 1))
                shapes.add(target.shape)
                canvases.add(pair["input"].shape)
                structures.add(_normalized(target))
                backgrounds.add(background)
                palettes.add(tuple(int(c) for c in np.unique(target)))
                schedules.add((rows, columns))
                counts["pairs"] += 1
                counts["edge"] += 0 in margins
                counts["one_axis"] += (max(rows) == 1) != (max(columns) == 1)
                counts["uniform_rows"] += len(set(rows)) == 1
                counts["uniform_columns"] += len(set(columns)) == 1
                counts["rich"] += len(np.unique(target)) >= 6
                counts["two_colors"] += len(np.unique(target)) == 2
                counts["row_repeat"] += _repeat(target, 0)
                counts["column_repeat"] += _repeat(target, 1)
                counts["wide_thin"] += any(min(a, b) == 1 and max(a, b) >= 9
                                           for bands in (rows, columns) for a, b in zip(bands, bands[1:]))
                # Same semantics after transpose, palette permutation and a new outer frame.
                _require(np.array_equal(solve(pair["input"].T), target.T), "transpose failure")
                recoloring = np.roll(np.arange(10, dtype=np.int8), seed % 9 + 1)
                _require(np.array_equal(solve(recoloring[pair["input"]]), recoloring[target]), "palette failure")
                if max(pair["input"].shape) <= 28:
                    _require(np.array_equal(solve(np.pad(pair["input"], 1, constant_values=background)), target),
                             "padding failure")
    for values, minimum, label in ((puzzles, 42, "puzzles"), (shapes, 20, "shapes"),
                                    (canvases, 60, "canvases"), (structures, 80, "structures"),
                                    (backgrounds, 8, "backgrounds"), (palettes, 40, "palettes"),
                                    (schedules, 80, "band schedules")):
        _require(len(values) >= minimum, f"too few {label}: {len(values)}")
    _require(all(value > 0 for value in counts.values()), "missing family variation")
    # Hand-worked seam/repeat fixture, its transpose, degenerate axes and one cell.
    fixtures = ((np.array([[2, 2, 5], [3, 4, 5], [2, 2, 5]]), (2, 1, 3), (1, 4, 2), 0),
                (np.array([[2, 3, 2], [2, 4, 2], [5, 5, 5]]), (1, 4, 2), (2, 1, 3), 9),
                (np.array([[2, 4, 2, 5]]), (3,), (2, 1, 3, 1), 0),
                (np.array([[0]]), (3,), (4,), 9))
    for target, rows, columns, background in fixtures:
        crop = _expand(target, rows, columns)
        for margins in (((1, 2), (3, 1)), ((0, 2), (1, 1)), ((1, 1), (0, 2))):
            source = np.pad(crop, margins, constant_values=background)
            before = source.copy()
            _require(np.array_equal(solve(source), target), "hand-worked fixture")
            _require(np.array_equal(source, before), "solve mutated input")
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            generate_puzzle(42000 + train * 10 + test, train, test)
    illegal = ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
               (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None), ("4", 1))
    for train, test in illegal:
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "logical_shapes": len(shapes),
            "canvas_shapes": len(canvases), "normalized_structures": len(structures),
            "background_colors": len(backgrounds), "palettes": len(palettes),
            "band_schedules": len(schedules), "counts": counts}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
