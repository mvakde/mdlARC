"""Parity-selected corners of a mosaic recovered by full-vector run collapse.

The whole square is content. There is no background or marker color.
Generation constructs source mosaics; solve only reads the supplied pixels.
All checks are bounded and this module writes no files.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


# Original and selected-variation pairs of A and B, hashed as shape + uint8 cells
# for input, followed by shape + uint8 cells for output.
_PARENT_PAIR_HASHES = frozenset((
    "c67235a30ffc1b17b5ba3bfbce383d70255c509e133ec5b151f8c31432976abd",
    "8db74c77ca88d36d7843871e16bf12b2d85de07e537abc2c576031b72a64516d",
    "be4f21901417badbf7cab99c825bb2f88850055cbe1b766bdf570607db13767a",
    "c8d7e03bfa1200f183089fc4b03cc97f1ae6bea3ff764b550b3abd8f11dcb951",
    "7cb6a9e5ca5338279ea4b46dac6fe0944bdec9086769b0cf25884590e6f80780",
    "71bb59d530a1a301ede6ae1e456bdb7059c583f47dad9c82550d81580adff667",
    "b21aa7a7c837278a173003afc811902d44dc7a927ac8783553d348fc2dcbe4af",
    "36a5bec239ca794bdb7d1d78c4ed41363bcf84ee4a33e399f80aff67e907238a",
    "6c534c8dd6c34559e55a039fb1b87eb01891cdc53741de63e0b9c2cd7140631c",
    "48d4b7267fbbd7d96c9f50d8ed8c749b0b7dd0b2071db4d385f7c44f894490e1",
    "33e5c1eef15e1241c2ccd6116efce49e36b0889dfffa743f764753046515e5f5",
    "c622ba9bdf30b4aa9eb8374043d8cb563e2933b9d0897b93f16bb92237cf1a69",
    "02cf1a1c5b4f0841a3d2e4d3966734a1fa1d9875f4dab1f8966444778da9562b",
    "bd088196bfabef92168a22a1190d7dc699879c7ea1215175d3e7dc16a6af2a56",
    "75ad96019a9aa6a95284d7eca1ac997f5869f1d82af639205678108beddf82cc",
    "d47a81ecf3cdd142ae41ad61e52257afea11b509e5d25add32204f79d515a0aa",
    "632cbff5576fc4008e02bd0350b89635e3bf84ca758c2ad3fea6e03c834edb81",
    "4dd8187e059cfa01dc496f54e0de7a5c33182c0e66e83c0e8aa0aba83a4f8049",
))
_LEGAL_COUNTS = tuple((a, b) for a in range(2, 11)
                      for b in range(1, min(4, a - 1) + 1))


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or not bool(np.all((0 <= grid) & (grid <= 9)))):
        raise ValueError("expected a 2D integer ARC grid with dimensions 1..30 and colors 0..9")
    return grid


def _collapse(grid):
    """Both axes are compared as complete vectors of the ORIGINAL grid."""
    row_starts = np.r_[0, 1 + np.flatnonzero(np.any(grid[1:] != grid[:-1], axis=1))]
    col_starts = np.r_[0, 1 + np.flatnonzero(np.any(grid[:, 1:] != grid[:, :-1], axis=0))]
    compact = grid[np.ix_(row_starts, col_starts)].copy()
    rows = np.diff(np.r_[row_starts, grid.shape[0]])
    cols = np.diff(np.r_[col_starts, grid.shape[1]])
    return compact, rows, cols


def _corner(compact, parity):
    return compact[:2, :2].copy() if parity else compact[:2, -2:].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Collapse maximal identical runs, then crop by the original square's parity."""
    grid = _grid(input_grid)
    side = grid.shape[0]
    if grid.shape != (side, side) or side < 2:
        raise ValueError("input must be square with side at least two")
    compact, _, _ = _collapse(grid)
    if min(compact.shape) < 2:
        raise ValueError("compact mosaic must contain at least two rows and two columns")
    return _corner(compact, side % 2)


def _visible(grid):
    return (bool(np.all(np.any(grid[1:] != grid[:-1], axis=1)))
            and bool(np.all(np.any(grid[:, 1:] != grid[:, :-1], axis=0))))


def _signature(grid):
    return grid.shape, np.asarray(grid, dtype=np.uint8).tobytes()


def _pair_hash(pair):
    blob = b"".join(bytes(np.asarray(pair[k]).shape)
                    + np.asarray(pair[k], dtype=np.uint8).tobytes()
                    for k in ("input", "output"))
    return hashlib.sha256(blob).hexdigest()


def _normalized(grid):
    mapping = {}
    cells = bytes(mapping.setdefault(int(v), len(mapping)) for v in grid.flat)
    return grid.shape, cells


def _axis_informative(patch):
    return (not np.array_equal(patch[0], patch[1])
            and not np.array_equal(patch[:, 0], patch[:, 1]))


def _single_witness_seam(grid):
    for differences in (np.count_nonzero(grid[:, 1:] != grid[:, :-1], axis=0),
                        np.count_nonzero(grid[1:] != grid[:-1], axis=1)):
        if bool(np.any(differences == 1)):
            return True
    return False


def _dedup_columns(grid):
    """Deliberately WRONG alternative: remove separated repetitions as well."""
    indices, seen = [], set()
    for j in range(grid.shape[1]):
        key = grid[:, j].tobytes()
        if key not in seen:
            seen.add(key)
            indices.append(j)
    return grid[:, indices]


def _field(rng, shape, palette):
    """Six content structures; no color has a semantic role."""
    h, w = shape
    k = len(palette)
    r, c = np.indices(shape)
    mode = int(rng.integers(6))
    if mode == 0:
        labels = rng.integers(k, size=shape)
    elif mode == 1:
        th, tw = int(rng.integers(2, min(h, 4) + 1)), int(rng.integers(2, min(w, 4) + 1))
        tile = rng.integers(k, size=(th, tw))
        labels = tile[r % th, c % tw]
    elif mode == 2:
        labels = (r + int(rng.integers(1, k)) * c + int(rng.integers(k))) % k
    elif mode == 3:
        labels = np.zeros(shape, dtype=int)
        offset = int(rng.integers(max(h, w)))
        # A sparse scaffold makes row and column boundaries visible in distant cells.
        for i in range(h):
            labels[i, (i + offset) % w] = int(rng.integers(1, k))
        for j in range(w):
            labels[(j + offset) % h, j] = int(rng.integers(1, k))
    elif mode == 4:
        a, b = rng.integers(k, size=h), rng.integers(k, size=w)
        for values in (a, b):
            for i in range(1, len(values)):
                if values[i] == values[i - 1]:
                    values[i] = (values[i] + int(rng.integers(1, k))) % k
        labels = (a[:, None] + b[None, :]) % k
    else:
        labels = rng.integers(k, size=shape)
        # An odd reflected axis has no forced identical middle pair.
        if w % 2:
            labels[:, w // 2 + 1:] = labels[:, :w // 2][:, ::-1]
        elif h % 2:
            labels[h // 2 + 1:] = labels[:h // 2][::-1]
    return palette[labels].astype(np.int8, copy=True)


def _mosaic(rng, shape, *, witness=False, seam_repeat=False):
    for _ in range(256):
        size = int(rng.integers(2, min(10, shape[0] * shape[1]) + 1))
        palette = rng.choice(np.arange(10, dtype=np.int8), size=size, replace=False)
        grid = _field(rng, shape, palette)
        if seam_repeat:
            # Columns 0 and 1 differ ONLY in logical row 1; row 0 hides the seam.
            grid[:, 1] = grid[:, 0]
            alternatives = palette[palette != grid[1, 0]]
            grid[1, 1] = rng.choice(alternatives)
            # A separated recurrence must survive, including at the right crop.
            grid[:, -1] = grid[:, 0]
        elif not witness:
            profile = float(rng.random())
            if profile < .20 and min(shape) >= 3:
                start = 0 if rng.random() < .5 else shape[1] - 2
                grid[:2, start:start + 2] = rng.choice(palette)
            elif profile < .36 and shape[1] >= 4:
                grid[:2, -2:] = grid[:2, :2]
        if not _visible(grid):
            continue
        if witness:
            left, right = _corner(grid, 1), _corner(grid, 0)
            if (not _axis_informative(left) or not _axis_informative(right)
                    or np.array_equal(left, right)):
                continue
        if seam_repeat and np.array_equal(_corner(_dedup_columns(grid), 0), _corner(grid, 0)):
            continue
        return grid
    raise RuntimeError("bounded mosaic sampling exhausted")


def _bands(rng, total, length, minima=None):
    base = np.ones(length, dtype=int) if minima is None else np.array(minima, dtype=int)
    remaining = total - int(base.sum())
    if remaining < 0:
        raise ValueError("side too small for requested bands")
    mode = int(rng.integers(4))
    if mode == 0:
        # Balanced scales, including uniform scales when compatible.
        base += remaining // length
        base[rng.permutation(length)[:remaining % length]] += 1
    elif mode == 1:
        base += rng.multinomial(remaining, np.full(length, 1 / length))
    elif mode == 2:
        big = int(remaining * float(rng.uniform(.65, 1.0)))
        base[int(rng.integers(length))] += big
        base += rng.multinomial(remaining - big, np.full(length, 1 / length))
    else:
        base += rng.multinomial(remaining, rng.dirichlet(np.full(length, .4)))
    return base


def _make_pair(rng, compact, side, used, *, witness=False):
    h, w = compact.shape
    row_min, col_min = np.ones(h, dtype=int), np.ones(w, dtype=int)
    if witness:
        row_min[0] = 2
        col_min[0] = col_min[-1] = 2
    for _ in range(64):
        rows = _bands(rng, side, h, row_min)
        cols = _bands(rng, side, w, col_min)
        if witness and len(set(rows)) == len(set(cols)) == 1:
            continue
        grid = np.repeat(np.repeat(compact, rows, axis=0), cols, axis=1)
        pair = {"input": grid, "output": _corner(compact, side % 2)}
        key = _signature(grid)
        if key in used or _pair_hash(pair) in _PARENT_PAIR_HASHES:
            continue
        used.add(key)
        return pair
    raise RuntimeError("bounded unique expansion sampling exhausted")


def _witness_shape(rng, excluded=()):
    choices = [(h, w) for h in range(2, 7) for w in range(3, 8)
               if (h, w) not in excluded]
    return choices[int(rng.integers(len(choices)))]


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _check_counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(int(seed))
    puzzle, used = {"train": [], "test": []}, set()
    train_shapes, previous_sides = set(), set()
    first_odd_larger = bool(rng.integers(2))

    # One matched odd/even pair always; four demonstrations give two such pairs,
    # in opposite size orders, with different compact geometry and content.
    groups = 2 if num_train >= 4 else 1
    for group in range(groups):
        shape = _witness_shape(rng, train_shapes)
        train_shapes.add(shape)
        compact = _mosaic(rng, shape, witness=True, seam_repeat=group == 0)
        odd_larger = first_odd_larger if group == 0 else not first_odd_larger
        even_choices = (10, 14, 18, 22) if group == 0 else tuple(range(10, 25, 2))
        candidates = [(e, e + (1 if odd_larger else -1)) for e in even_choices]
        candidates = [(e, o) for e, o in candidates if e not in previous_sides and o not in previous_sides]
        even, odd = candidates[int(rng.integers(len(candidates)))]
        previous_sides.update((even, odd))
        for side in (odd, even):
            puzzle["train"].append(_make_pair(rng, compact, side, used, witness=True))
    if num_train == 3:
        # Three sizes can already defeat any single increasing/decreasing
        # threshold: the third outcome repeats the smaller side's branch.
        shape = _witness_shape(rng, train_shapes)
        train_shapes.add(shape)
        compact = _mosaic(rng, shape, witness=True)
        side = max(previous_sides) + 1
        puzzle["train"].append(_make_pair(rng, compact, side, used, witness=True))
    while len(puzzle["train"]) < num_train:
        shape = (int(rng.integers(2, 9)), int(rng.integers(2, 9)))
        compact = _mosaic(rng, shape)
        train_shapes.add(shape)
        side = int(rng.integers(max(6, *shape), 27))
        puzzle["train"].append(_make_pair(rng, compact, side, used))
    order = rng.permutation(num_train)
    puzzle["train"] = [puzzle["train"][int(i)] for i in order]

    parities = [int(rng.integers(2))] if num_test == 1 else [0, 1]
    while len(parities) < num_test:
        parities.append(int(rng.integers(2)))
    rng.shuffle(parities)
    for parity in parities:
        # New compact geometry, with no test-only palette or texture convention.
        shapes = [(h, w) for h in range(2, 11) for w in range(2, 11)
                  if (h, w) not in train_shapes]
        shape = shapes[int(rng.integers(len(shapes)))]
        compact = _mosaic(rng, shape)
        lo = max(6, max(shape) + 2)
        sides = [n for n in range(lo, 31) if n % 2 == parity]
        side = int(rng.choice(sides))
        puzzle["test"].append(_make_pair(rng, compact, side, used))
    validate_generated_puzzle(puzzle)
    return puzzle


def _shortcut_failures(grid, compact, rows, cols, output):
    parity = grid.shape[0] % 2
    row_starts = np.r_[0, np.cumsum(rows)[:-1]]
    col_starts = np.r_[0, np.cumsum(cols)[:-1]]
    alternatives = {
        "raw_crop": _corner(grid, parity),
        "rows_only": _corner(grid[row_starts], parity),
        "columns_only": _corner(grid[:, col_starts], parity),
        "compact_height_parity": _corner(compact, compact.shape[0] % 2),
        "compact_width_parity": _corner(compact, compact.shape[1] % 2),
        "global_column_dedup": _corner(_dedup_columns(compact), parity),
    }
    top_starts = np.r_[0, 1 + np.flatnonzero(grid[0, 1:] != grid[0, :-1])]
    alternatives["top_row_column_runs"] = _corner(grid[np.ix_(row_starts, top_starts)], parity)
    return {name: not np.array_equal(value, output) for name, value in alternatives.items()}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, groups, training_parities, train_shapes, test_shapes = set(), {}, set(), set(), set()
    failures = {name: 0 for name in (
        "raw_crop", "rows_only", "columns_only", "compact_height_parity",
        "compact_width_parity", "global_column_dedup", "top_row_column_runs")}
    seam_with_uneven = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            grid, output = _grid(pair["input"]), _grid(pair["output"])
            n = grid.shape[0]
            _require(grid.shape == (n, n) and n >= 2, "input is not a square")
            compact, rows, cols = _collapse(grid)
            _require(min(compact.shape) >= 2 and _visible(compact), "invalid compact mosaic")
            _require(output.shape == (2, 2), "output must be 2x2")
            _require(np.array_equal(output, _corner(compact, n % 2)), "wrong compact corner")
            rebuilt = np.repeat(np.repeat(compact, rows, axis=0), cols, axis=1)
            _require(np.array_equal(grid, rebuilt), "expansion does not reconstruct input")
            _require(int(rows.sum()) == int(cols.sum()) == n, "wrong expansion totals")
            key = _signature(grid)
            _require(key not in seen, "duplicate input")
            _require(_pair_hash(pair) not in _PARENT_PAIR_HASHES, "exact parent pair copied")
            seen.add(key)
            if split == "train":
                train_shapes.add(compact.shape)
                if not np.array_equal(_corner(compact, 0), _corner(compact, 1)):
                    training_parities.add(n % 2)
                    groups.setdefault(_signature(compact), set()).add(n % 2)
                for name, failed in _shortcut_failures(grid, compact, rows, cols, output).items():
                    failures[name] += failed
                uneven = len(set(rows)) > 1 or len(set(cols)) > 1
                seam_with_uneven |= uneven and _single_witness_seam(compact)
            else:
                test_shapes.add(compact.shape)
    _require(training_parities == {0, 1}, "training does not distinguish both corners")
    _require(any(parities == {0, 1} for parities in groups.values()), "missing matched parity examples")
    _require(all(v > 0 for v in failures.values()), "training admits a named shortcut: " + str(failures))
    _require(seam_with_uneven, "missing uneven expansion with a globally exposed seam")
    _require(not train_shapes & test_shapes, "test must use new compact geometry")
    if len(puzzle["train"]) >= 3:
        sides = {p: {np.asarray(x["input"]).shape[0] for x in puzzle["train"]
                     if np.asarray(x["input"]).shape[0] % 2 == p} for p in (0, 1)}
        _require(max(sides[0]) > min(sides[1]) and max(sides[1]) > min(sides[0]),
                 "size threshold separates the training branches")
        if len(puzzle["train"]) >= 4:
            _require(all(len(s) >= 2 for s in sides.values()), "need two sizes of each parity")
    if len(puzzle["test"]) >= 2:
        _require({np.asarray(x["input"]).shape[0] % 2 for x in puzzle["test"]} == {0, 1},
                 "multiple tests must exercise both parities")


def _puzzle_digest(puzzle):
    return hashlib.sha256("".join(_pair_hash(p) for s in ("train", "test")
                                  for p in puzzle[s]).encode()).hexdigest()


def self_test():
    """42 streamed default puzzles, 30 legal count calls, and bounded metamorphic checks."""
    puzzles, sides, shapes, palettes, structures, profiles = set(), set(), set(), set(), set(), set()
    palette_sizes, colors, test_parities = set(), set(), set()
    stats = {"pairs": 0, "uniform_outputs": 0, "equal_corners": 0, "rectangular_compacts": 0,
             "separated_columns": 0, "one_pixel_bands": 0, "large_bands": 0,
             "uniform_axis": 0, "mismatched_height_parity": 0, "mismatched_width_parity": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_digest(puzzle))
        _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed)), "nondeterminism")
        rng = np.random.default_rng(seed + 123456)
        permutation = rng.permutation(10)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                before = grid.copy()
                grid.setflags(write=False)
                answer = solve(grid)
                _require(np.array_equal(answer, output) and np.array_equal(grid, before), "solver failure")
                _require(not np.shares_memory(answer, grid), "solver output aliases input")
                _require(np.array_equal(solve(permutation[grid]), permutation[output]), "color dependence")
                compact, rows, cols = _collapse(grid)
                n = grid.shape[0]
                sides.add(n)
                shapes.add(compact.shape)
                palettes.add(tuple(map(int, np.unique(grid))))
                palette_sizes.add(len(np.unique(grid)))
                colors.update(map(int, np.unique(grid)))
                structures.add(_normalized(compact))
                profiles.add((tuple(map(int, rows)), tuple(map(int, cols))))
                stats["pairs"] += 1
                stats["uniform_outputs"] += len(np.unique(output)) == 1
                stats["equal_corners"] += np.array_equal(_corner(compact, 0), _corner(compact, 1))
                stats["rectangular_compacts"] += compact.shape[0] != compact.shape[1]
                stats["separated_columns"] += _dedup_columns(compact).shape[1] < compact.shape[1]
                stats["one_pixel_bands"] += 1 in rows or 1 in cols
                stats["large_bands"] += max(max(rows), max(cols)) >= 8
                stats["uniform_axis"] += len(set(rows)) == 1 or len(set(cols)) == 1
                stats["mismatched_height_parity"] += n % 2 != compact.shape[0] % 2
                stats["mismatched_width_parity"] += n % 2 != compact.shape[1] % 2
                if split == "test":
                    test_parities.add(n % 2)
                if n < 30:
                    # Duplicate one whole row and one whole column: same compact,
                    # opposite original parity, hence the other selected corner.
                    r, c = int(rng.integers(n)), int(rng.integers(n))
                    expanded = np.insert(grid, r, grid[r], axis=0)
                    expanded = np.insert(expanded, c, expanded[:, c], axis=1)
                    _require(np.array_equal(_collapse(expanded)[0], compact), "bad metamorphic expansion")
                    _require(np.array_equal(solve(expanded), _corner(compact, (n + 1) % 2)),
                             "parity did not follow original input")
    _require(len(puzzles) == 42 and len(sides) >= 15 and len(shapes) >= 25, "geometry diversity")
    _require(len(structures) >= 70 and len(profiles) >= 140, "structural diversity")
    _require(len(palettes) >= 35 and len(palette_sizes) >= 7 and colors == set(range(10)), "palette diversity")
    _require(test_parities == {0, 1} and all(v > 0 for v in stats.values()), "coverage gap: " + str(stats))
    for a, b in _LEGAL_COUNTS:
        puzzle = generate_puzzle(7000 + 10 * a + b, a, b)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (a, b), "wrong explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, 1.5, True):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    for bad in (np.ones((1, 1), dtype=int), np.ones((3, 3), dtype=int),
                np.ones((2, 3), dtype=int), np.full((2, 2), 10),
                np.array([[1., 2.], [3., 4.]])):
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input accepted")
    # Explicit boundary cases outside the usual sampler's shape/scale range.
    for compact, rows, cols in (
        (np.array([[0, 2], [8, 5]]), [1, 1], [1, 1]),
        (np.array([[1, 1, 2], [1, 3, 1]]), [1, 4], [2, 1, 2]),
        (np.array([[4, 5], [5, 4], [4, 5]]), [1, 1, 2], [3, 1]),
    ):
        grid = np.repeat(np.repeat(compact, rows, axis=0), cols, axis=1)
        _require(np.array_equal(solve(grid), _corner(compact, grid.shape[0] % 2)), "boundary crop")
    # Larger compacts than the sampler produces still follow the same rule.
    r, c = np.indices((17, 23))
    compact = (r + 2 * c) % 7
    for n in (29, 30):
        rows, cols = np.ones(17, dtype=int), np.ones(23, dtype=int)
        rows[0] += n - 17
        cols[-1] += n - 23
        grid = np.repeat(np.repeat(compact, rows, axis=0), cols, axis=1)
        _require(np.array_equal(solve(grid), _corner(compact, n % 2)), "large compact failure")
    r, c = np.indices((30, 30))
    compact = (r + c) % 2
    _require(np.array_equal(solve(compact), compact[:2, -2:]), "unit-band 30x30 failure")
    largest = generate_puzzle(999, 10, 4)
    _require(_puzzle_digest(largest) == _puzzle_digest(generate_puzzle(999, 10, 4)), "count determinism")
    damaged = generate_puzzle(1)
    damaged["train"][0]["output"][0, 0] = (int(damaged["train"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted answer")
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "input_sides": sorted(sides),
            "compact_shapes": len(shapes), "actual_palette_sizes": sorted(palette_sizes),
            "palettes": len(palettes), "normalized_compact_structures": len(structures),
            "expansion_profiles": len(profiles), "legal_count_combinations": len(_LEGAL_COUNTS),
            "stats": {k: int(v) for k, v in stats.items()}}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
