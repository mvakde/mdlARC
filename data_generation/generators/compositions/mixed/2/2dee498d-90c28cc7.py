"""Decode adjacent row/column runs, then stably deduplicate whole columns.

No palette roles, background, orientation selector, or external state are used.
Generation and validation are bounded and do not write files.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)):
        raise ValueError("expected an integer grid with dimensions 1..30")
    if np.any(grid < 0) or np.any(grid > 9):
        raise ValueError("colors must be in 0..9")
    return grid.astype(np.int8, copy=True)


def _run_starts(grid, axis):
    vectors = grid if axis == 0 else grid.T
    return np.r_[0, np.flatnonzero(np.any(vectors[1:] != vectors[:-1], axis=1)) + 1]


def _decode(grid):
    # Boundaries are compared over the entire original vector on each axis.
    rows, columns = _run_starts(grid, 0), _run_starts(grid, 1)
    return grid[np.ix_(rows, columns)].copy()


def _unique_columns(grid):
    seen, keep = set(), []
    for j in range(grid.shape[1]):
        key = grid[:, j].tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(j)
    return grid[:, keep].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Apply the fixed B-then-A rule, deriving everything from input pixels."""
    return _unique_columns(_decode(_grid(input_grid)))


def _signature(grid):
    return tuple(grid.shape), grid.tobytes()


def _visible_boundaries(grid):
    return (all(not np.array_equal(a, b) for a, b in zip(grid, grid[1:]))
            and all(not np.array_equal(a, b) for a, b in zip(grid.T, grid.T[1:])))


def _near_columns(grid):
    return any(np.count_nonzero(grid[:, i] != grid[:, j]) == 1
               for i in range(grid.shape[1]) for j in range(i + 1, grid.shape[1]))


def _separated_rows(grid):
    return len({row.tobytes() for row in grid}) < grid.shape[0]


def _canonical(rng, height, width, kind):
    # Capacity counts independent row coordinates when a row is copied.
    effective_height = height - (kind == "row_return")
    minimum = 2
    while minimum ** effective_height < width:
        minimum += 1
    if kind == "cyclic":
        palette_size = int(rng.integers(max(width, minimum), 11))
    else:
        maximum = min(10, height * width)
        palette_size = int(rng.integers(minimum, maximum + 1))
    palette = rng.choice(10, palette_size, replace=False).astype(np.int8)
    for _ in range(512):
        if kind == "cyclic":
            # An optional organized pattern; literal color magnitude is irrelevant.
            shifts = rng.integers(palette_size, size=height)
            indices = (shifts[:, None] + np.arange(width)) % palette_size
            grid = palette[indices]
        else:
            # Sample distinct base-p encodings directly. This stays reliable
            # even when all possible low-palette column vectors are needed.
            capacity = palette_size ** effective_height
            if kind == "near":
                first = int(rng.integers(capacity))
                position = int(rng.integers(effective_height))
                power = palette_size ** position
                old = first // power % palette_size
                new = int(rng.choice([v for v in range(palette_size) if v != old]))
                second = first + (new - old) * power
                excluded = sorted((first, second))
                other = rng.choice(capacity - 2, size=width - 2, replace=False)
                codes = [first, second]
                for value in other:
                    value = int(value)
                    for hole in excluded:
                        if value >= hole:
                            value += 1
                    codes.append(value)
            else:
                codes = rng.choice(capacity, size=width, replace=False)
            indices = np.array([[int(code) // palette_size ** i % palette_size
                                 for code in codes] for i in range(effective_height)])
            grid = palette[indices]
            if kind == "row_return":
                grid = np.insert(grid, 2, grid[0], axis=0)
        if (_visible_boundaries(grid)
                and len({col.tobytes() for col in grid.T}) == width):
            return grid.astype(np.int8, copy=True)
    raise RuntimeError("bounded canonical sampling exhausted")


def _without_adjacent_repeats(values):
    result = []
    for value in values:
        value = int(value)
        if not result or result[-1] != value:
            result.append(value)
    return result


def _column_order(rng, width, style):
    if style == "early":
        # A separated repeat BEFORE the final new type defeats a prefix crop.
        order = list(range(width))
        position = int(rng.integers(2, width))
        earlier = int(rng.integers(position - 1))
        order.insert(position, earlier)
        extras = int(rng.integers(1, min(width + 4, 25 - len(order))))
        order.extend(map(int, rng.integers(width, size=extras)))
        # End with the first type: keeping LAST occurrences gives another order.
        if order[-1] == 0:
            order.append(1)
        order.append(0)
    elif style == "sweeps":
        order = list(range(width))
        traversals = int(rng.integers(2, min(3, 24 // width) + 1))
        for _ in range(traversals - 1):
            wave = list(range(width))
            if rng.random() < 0.5:
                wave.reverse()
            order.extend(wave)
    else:
        counts = rng.integers(1, 4, size=width)
        counts[0] = max(2, counts[0])
        while counts.sum() > 24:
            candidates = np.flatnonzero(counts > 1)
            counts[int(rng.choice(candidates))] -= 1
        order = np.repeat(np.arange(width), counts).tolist()
        rng.shuffle(order)
    order = _without_adjacent_repeats(order)
    if len(order) == width:
        order.append(int(rng.choice([i for i in range(width) if i != order[-1]])))
    return order


def _bands(rng, count, mode):
    # Every axis expands, but factors of one are legal. The exact sum is <=30.
    if mode == "uniform" and count <= 15:
        return np.full(count, int(rng.integers(2, min(4, 30 // count) + 1)))
    upper = min(30, max(count + 2, 4 * count))
    total = int(rng.integers(count + 2, upper + 1))
    bands = np.ones(count, dtype=np.int64)
    extra = total - count
    if mode == "spike":
        peak = int(rng.integers(count))
        large = int(rng.integers(max(1, extra // 2), extra + 1))
        bands[peak] += large
        extra -= large
    if extra:
        bands += np.bincount(rng.integers(count, size=extra), minlength=count)
    if len(set(map(int, bands))) == 1:
        bands[0] -= 1
        bands[1] += 1
    return bands


def _make_pair(rng, shape, kind, style, uneven=False):
    canonical = _canonical(rng, *shape, kind)
    order = _column_order(rng, shape[1], style)
    latent = canonical[:, order]
    first_ids = list(dict.fromkeys(order))
    # Labels follow construction, rather than calling solve to label itself.
    output = canonical[:, first_ids].copy()
    modes = ("mixed", "spike") if uneven else ("uniform", "mixed", "spike")
    rows = _bands(rng, latent.shape[0], str(rng.choice(modes)))
    columns = _bands(rng, latent.shape[1], str(rng.choice(modes)))
    source = np.repeat(np.repeat(latent, rows, axis=0), columns, axis=1)
    return {"input": source.astype(np.int8, copy=False), "output": output}


def _new_pair(rng, shape, kind, style, uneven, seen):
    for _ in range(64):
        pair = _make_pair(rng, shape, kind, style, uneven)
        if uneven:
            vectors = [tuple(map(int, col)) for col in pair["output"].T]
            if vectors == sorted(vectors):
                continue
        key = _signature(pair["input"])
        if key not in seen:
            seen.add(key)
            return pair
    raise RuntimeError("bounded distinct-input sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError("require 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    height_cap, width_cap = map(int, rng.integers(3, 7, size=2))
    first_shape = (int(rng.integers(3, height_cap + 1)),
                   int(rng.integers(3, width_cap + 1)))
    choices = [(h, w) for h in range(3, height_cap + 1)
               for w in range(2, width_cap + 1) if (h, w) != first_shape]
    second_shape = choices[int(rng.integers(len(choices)))]
    # Two complementary witnesses remain present even at the minimum count.
    plans = [(first_shape, "near", "early", True),
             (second_shape, "row_return", "sweeps", False)]
    for _ in range(num_train - 2):
        shape = (int(rng.integers(2, height_cap + 1)),
                 int(rng.integers(2, width_cap + 1)))
        kinds = ("random", "near", "cyclic")
        if shape[0] >= 3:
            kinds += ("row_return",)
        kind = str(rng.choice(kinds))
        styles = ("sweeps", "scatter") + (("early",) if shape[1] >= 3 else ())
        plans.append((shape, kind, str(rng.choice(styles)), False))
    rng.shuffle(plans)
    seen = set()
    train = [_new_pair(rng, shape, kind, style, uneven, seen)
             for shape, kind, style, uneven in plans]
    max_height = max(pair["output"].shape[0] for pair in train)
    max_width = max(pair["output"].shape[1] for pair in train)
    shift_axis = int(rng.integers(2))
    test = []
    for _ in range(num_test):
        if shift_axis == 0:
            shape = (int(rng.integers(max_height + 1, min(9, max_height + 3) + 1)),
                     int(rng.integers(2, width_cap + 1)))
        else:
            shape = (int(rng.integers(2, height_cap + 1)),
                     int(rng.integers(max_width + 1, min(9, max_width + 3) + 1)))
        kinds = ("random", "near", "cyclic") + (("row_return",) if shape[0] >= 3 else ())
        styles = ("sweeps", "scatter") + (("early",) if shape[1] >= 3 else ())
        test.append(_new_pair(rng, shape, str(rng.choice(kinds)),
                              str(rng.choice(styles)), False, seen))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _analysis(source):
    """Literal tuple-based boundary/identity audit, independent of generation."""
    rows = [tuple(map(int, row)) for row in source]
    columns = list(zip(*rows))
    ri = [i for i, row in enumerate(rows) if i == 0 or row != rows[i - 1]]
    ci = [j for j, col in enumerate(columns) if j == 0 or col != columns[j - 1]]
    latent = source[np.ix_(ri, ci)]
    row_bands = tuple(b - a for a, b in zip(ri, ri[1:] + [len(rows)]))
    col_bands = tuple(b - a for a, b in zip(ci, ci[1:] + [len(columns)]))
    vectors = [tuple(map(int, col)) for col in latent.T]
    retained = list(dict.fromkeys(vectors))
    expected = np.asarray(retained, dtype=np.int8).T.copy()
    ids = [retained.index(vector) for vector in vectors]
    k = len(retained)
    early = ids[:k] != list(range(k))
    return latent, expected, row_bands, col_bands, ids, early


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "exactly train and test are required")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal pair counts")
    seen, train_shapes = set(), set()
    early_uneven_near, returned_rows = False, False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair needs exactly input and output")
            source, target = _grid(pair["input"]), _grid(pair["output"])
            signature = _signature(source)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            latent, expected, rows, columns, ids, early = _analysis(source)
            _require(np.array_equal(expected, target), "wrong stable decoded columns")
            _require(np.array_equal(solve(source), target), "solve disagrees with audit")
            _require(_visible_boundaries(latent), "invisible decoded boundary")
            _require(source.shape[0] > latent.shape[0] == target.shape[0],
                     "B must remove at least one row copy")
            _require(source.shape[1] > latent.shape[1] > target.shape[1],
                     "need column expansion AND a nonadjacent column recurrence")
            _require(np.array_equal(np.repeat(np.repeat(latent, rows, axis=0),
                                             columns, axis=1), source),
                     "bands do not reconstruct the entire input")
            _require(len(set(ids)) == target.shape[1], "lost or invented column")
            _require(set(source.flat) == set(target.flat), "colors were changed or lost")
            _require(_visible_boundaries(target), "adjacent output rows/columns coincide")
            _require(np.array_equal(solve(target), target), "output is not a normal form")
            # Explicit necessity checks: each parent alone gives a wrong answer.
            _require(not np.array_equal(latent, target), "A is unnecessary")
            _require(not np.array_equal(_unique_columns(source), target), "B is unnecessary")
            if split == "train":
                train_shapes.add(target.shape)
                uneven = len(set(rows)) > 1 and len(set(columns)) > 1
                vectors = [tuple(map(int, col)) for col in target.T]
                last_order = sorted(range(len(vectors)),
                                    key=lambda value: max(i for i, v in enumerate(ids) if v == value))
                early_uneven_near |= (early and uneven and _near_columns(target)
                                      and vectors != sorted(vectors)
                                      and last_order != list(range(len(vectors))))
                returned_rows |= _separated_rows(target)
    _require(len(train_shapes) >= 2, "training needs distinct output shapes")
    _require(early_uneven_near, "missing early-repeat/uneven/near-column witness")
    _require(returned_rows, "training must preserve separated identical rows")
    heights = [p["output"].shape[0] if isinstance(p["output"], np.ndarray)
               else len(p["output"]) for p in puzzle["train"]]
    widths = [np.asarray(p["output"]).shape[1] for p in puzzle["train"]]
    _require(all(np.asarray(p["output"]).shape[0] > max(heights) for p in puzzle["test"])
             or all(np.asarray(p["output"]).shape[1] > max(widths) for p in puzzle["test"]),
             "tests must extrapolate a common logical dimension")


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streaming seeds, all count combinations, and semantic edge cases."""
    fingerprints, shapes, target_shapes, palettes, profiles, orders = (set() for _ in range(6))
    structures, shifts = set(), set()
    near, row_returns, one_pixel, large_band, unequal_recurrences = 0, 0, 0, 0, 0
    pairs_checked = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        if seed in (0, 17, 41):
            _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)),
                     "seed did not reproduce puzzle")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                _, _, rows, columns, ids, _ = _analysis(source)
                shapes.add(source.shape)
                target_shapes.add(target.shape)
                palettes.add(tuple(map(int, np.unique(target))))
                profiles.add((rows, columns))
                orders.add(tuple(ids))
                roles = {}
                normalized = bytes(roles.setdefault(int(v), len(roles)) for v in target.flat)
                structures.add((target.shape, normalized))
                near += _near_columns(target)
                row_returns += _separated_rows(target)
                one_pixel += 1 in rows + columns
                large_band += max(rows + columns) >= 8
                counts = [ids.count(i) for i in range(target.shape[1])]
                unequal_recurrences += len(set(counts)) > 1
                pairs_checked += 1
        shifts.add("taller" if puzzle["test"][0]["output"].shape[0]
                   > max(p["output"].shape[0] for p in puzzle["train"]) else "wider")
    _require(len(fingerprints) == 42, "duplicate puzzles across seeds")
    _require(len(shapes) >= 40 and len(target_shapes) >= 20, "narrow geometry")
    _require(len(profiles) >= 100 and len(orders) >= 60, "narrow expansion or recurrence")
    _require(len(palettes) >= 50 and len(structures) >= 100, "narrow palette or structure")
    _require(shifts == {"taller", "wider"}, "missing one extrapolation axis")
    _require(min(near, row_returns, one_pixel, large_band, unequal_recurrences) >= 10,
             "missing important structural cases")
    legal_count_calls = 0
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(10000 + 10 * train + test, train, test)
            _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test),
                     "explicit counts ignored")
            legal_count_calls += 1
    invalid = ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
               (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None))
    for train, test in invalid:
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for seed in (-1, True, 0.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal seed accepted")
    # A concrete hand-derived witness: three column types, a returning row,
    # an early nonadjacent duplicate, and nonuniform expansion on both axes.
    canonical = np.array([[1, 1, 3], [2, 4, 5], [1, 1, 3]], dtype=np.int8)
    latent = canonical[:, [0, 1, 0, 2, 1]]
    example = np.repeat(np.repeat(latent, [2, 1, 3], axis=0), [1, 3, 2, 1, 2], axis=1)
    saved = example.copy()
    _require(np.array_equal(solve(example), canonical), "hand-derived witness failed")
    _require(np.array_equal(example, saved), "solver mutated input")
    for edge in (np.array([[7]]), np.full((30, 30), 0),
                 np.array([[1, 2, 1, 3]]), np.array([[1], [1], [2], [1]])):
        result = solve(edge)
        _require(np.array_equal(solve(result), result), "edge case not idempotent")
    # Algebraic limitation is tested rather than concealed: A and B commute.
    _require(np.array_equal(_decode(_unique_columns(example)), canonical),
             "unexpected noncommutative behavior")
    return {"seeds": 42, "pairs_checked": pairs_checked,
            "legal_count_calls": legal_count_calls, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "output_shapes": len(target_shapes),
            "palettes": len(palettes), "normalized_outputs": len(structures),
            "expansion_profiles": len(profiles), "column_orders": len(orders),
            "near_column_pairs": near, "separated_row_pairs": row_returns,
            "one_pixel_band_pairs": one_pixel, "large_band_pairs": large_band,
            "unequal_recurrence_pairs": unequal_recurrences, "shift_axes": sorted(shifts)}
