"""Compress equality runs, then label compressed bars with adjacency squares.

The interval cover is a constructor, not a solver convention: staggered bar
endpoints make the columns inside longer bars observably different. All bar
lengths and the reference cell are read only after literal run compression.
No files, parent modules, learned examples, or generation state are used by solve.
"""

import hashlib
import numpy as np


LENGTHS = (2, 3, 4, 5)
DIGITS = {length: (length - 1) ** 2 % 10 for length in LENGTHS}
BACKGROUNDS = (0, 2, 3, 5, 7, 8)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    array = np.asarray(value)
    _require(array.ndim == 2 and all(1 <= n <= 30 for n in array.shape),
             "grids must be two-dimensional with sides in 1..30")
    _require(np.issubdtype(array.dtype, np.integer), "cells must be integers")
    _require(bool(np.all((array >= 0) & (array <= 9))), "colors must be in 0..9")
    return array.astype(np.int8, copy=False)


def _runs(grid, axis):
    vectors = grid if axis == 0 else grid.T
    starts = [0] + [i for i in range(1, len(vectors))
                    if not np.array_equal(vectors[i], vectors[i - 1])]
    lengths = tuple(b - a for a, b in zip(starts, starts[1:] + [len(vectors)]))
    return starts, lengths


def _compress(grid):
    rows, _ = _runs(grid, 0)
    columns, _ = _runs(grid, 1)
    return grid[np.ix_(rows, columns)].copy()


def _read_canonical(grid):
    """Parse a compressed input; a bar tuple is (row, start, length, color)."""
    background = int(grid[0, 0])
    _require(background in BACKGROUNDS, "background would hide a label")
    _require(all(np.all(edge == background)
                 for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "a background border is required")
    counts = np.count_nonzero(grid != background, axis=1)
    singles = np.flatnonzero(counts == 1)
    _require(len(singles) == 1, "compressed input needs one singleton marker row")
    marker_row = int(singles[0])
    reference = int(np.flatnonzero(grid[marker_row] != background)[0])
    _require(np.count_nonzero(grid[:, reference] != background) == 1,
             "reference column intersects a bar")
    active = np.flatnonzero(counts)
    _require(bool(np.all(np.diff(active) >= 2)), "active rows must be separated")
    bars = []
    for row in active:
        row = int(row)
        if row == marker_row:
            continue
        columns = np.flatnonzero(grid[row] != background)
        length = len(columns)
        _require(length in LENGTHS, "compressed bar length must be 2..5")
        start, end = int(columns[0]), int(columns[-1])
        _require(end - start + 1 == length, "bar must be contiguous")
        color = int(grid[row, start])
        _require(bool(np.all(grid[row, start:end + 1] == color)),
                 "bar must be monochrome")
        _require(end <= reference - 2 or start >= reference + 2,
                 "a background cell must separate bar and reference column")
        bars.append((row, start, length, color))
    _require(bool(bars), "at least one bar is required")
    return {"background": background, "marker_row": marker_row,
            "reference": reference, "marker_color": int(grid[marker_row, reference]),
            "bars": tuple(bars)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from the given input, with fixed family rules."""
    compressed = _compress(_grid(input_grid))
    meta = _read_canonical(compressed)
    output = compressed.copy()
    for row, _, length, _ in meta["bars"]:
        output[row, meta["reference"]] = (length - 1) ** 2 % 10
    # Labeling can create equal neighbors. Do not compress the output again.
    return output


def _count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_))
             and isinstance(value, (int, np.integer)), f"{name} must be an integer")
    value = int(value)
    _require(low <= value <= high, f"{name} must be in {low}..{high}")
    return value


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        digest.update(split.encode())
        for pair in puzzle[split]:
            for role in ("input", "output"):
                array = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(array.shape))
                digest.update(array.tobytes())
    return digest.hexdigest()


def _intervals(rng, geometry, anchor):
    """Cover every internal cut of each side by a visible bar endpoint.

    Initial bars vary freely. Greedy helpers are ordinary bars with the same
    output rule. Each added interval covers at least one previously missing cut.
    An interval is (side, start, length), using coordinates local to that side.
    """
    if geometry == 2:
        sides = (-1, 1)
        primary = int(rng.choice(sides))
        capacities = {primary: int(rng.integers(5 if anchor else 3, 7)),
                      -primary: int(rng.integers(3, 7))}
    else:
        sides = (-1,) if geometry == 0 else (1,)
        capacities = {sides[0]: int(rng.integers(5 if anchor else 3, 10))}
    if anchor:
        lengths = [2, 3, 4, 5, 2]
    else:
        lengths = [int(rng.integers(2, 6)) for _ in range(int(rng.integers(1, 6)))]
    intervals = []
    for length in lengths:
        choices = [s for s in sides if capacities[s] >= length]
        if not choices:
            # Only non-anchor two-sided capacities can both be below length 5.
            chosen = int(rng.choice(sides))
            length = min(length, capacities[chosen])
        else:
            chosen = int(rng.choice(choices))
        start = int(rng.integers(0, capacities[chosen] - length + 1))
        intervals.append((chosen, start, length))
    for side in sides:
        capacity = capacities[side]
        if not any(s == side for s, _, _ in intervals):
            length = int(rng.integers(2, min(5, capacity) + 1))
            intervals.append((side, int(rng.integers(0, capacity - length + 1)), length))
        candidates = [(start, length) for length in LENGTHS
                      for start in range(capacity - length + 1)]
        witnessed = {x for s, start, length in intervals if s == side
                     for x in (start, start + length)}
        missing = set(range(1, capacity)) - witnessed
        while missing:
            gains = [len(missing & {start, start + length})
                     for start, length in candidates]
            best = max(gains)
            _require(best > 0, "interval cover is impossible")
            options = [i for i, gain in enumerate(gains) if gain == best]
            start, length = candidates[int(rng.choice(options))]
            intervals.append((side, start, length))
            missing -= {start, start + length}
    _require(len(intervals) <= 12, "too many bars for the bounded constructor")
    return intervals, capacities


def _stretch(rng, canonical, meta, anchor, test):
    height, width = canonical.shape
    rows = np.ones(height, dtype=int)
    columns = np.ones(width, dtype=int)
    protected_column = None
    # Both axes and content, not just empty margins, stretch in the anchor.
    if anchor:
        rows[meta["bars"][0][0]] += 1
        rows[0] += 1
        shortest = min(meta["bars"], key=lambda bar: bar[2])
        columns[shortest[1]] += 1
        # A length-two bar has unequal pixel widths: its raw length is 3 or 4,
        # whose square code differs from the compressed length-two code 1.
        protected_column = shortest[1] + 1
        columns[meta["reference"]] += 1
    if test:
        # Extrapolate equality-run size while using only taught length codes.
        longest = max(meta["bars"], key=lambda bar: bar[2])
        target = longest[1] + int(rng.integers(0, longest[2]))
        columns[target] = int(rng.integers(4, min(7, 31 - width)))
    row_budget = 30 - int(rows.sum())
    col_budget = 30 - int(columns.sum())
    for _ in range(int(rng.integers(0, min(8, row_budget) + 1))):
        rows[int(rng.integers(0, height))] += 1
    for _ in range(int(rng.integers(0, min(9, col_budget) + 1))):
        eligible = np.arange(width) if test else np.flatnonzero(columns < 3)
        if protected_column is not None:
            eligible = eligible[eligible != protected_column]
        if not len(eligible):
            break
        columns[int(rng.choice(eligible))] += 1
    # Every example requires compression; no-op axes remain possible off-anchor.
    if rows.sum() == height and columns.sum() == width:
        rows[meta["bars"][0][0]] += 1
    return np.repeat(np.repeat(canonical, rows, axis=0), columns, axis=1)


def _make_pair(rng, background, marker, geometry, anchor, test):
    intervals, capacities = _intervals(rng, geometry, anchor)
    palette = [color for color in range(10) if color != background]
    other = int(rng.choice([color for color in palette if color != marker]))
    colors = [int(rng.choice(palette)) for _ in intervals]
    if anchor:
        colors[:5] = [marker, marker, other, other, other]
    # Margins and bar-reference gaps are explicit background columns. Their
    # absolute widths will be normalized along with the colored columns.
    left = capacities.get(-1, 0)
    right = capacities.get(1, 0)
    reference = left + 2 if left else 1
    width = reference + right + 3 if right else reference + 2
    slots = list(int(x) for x in rng.permutation(len(intervals) + 1))
    marker_slot = slots[-1]
    canonical = np.full((2 * len(slots) + 1, width), background, dtype=np.int8)
    canonical[2 * marker_slot + 1, reference] = marker
    for slot, (side, start, length), color in zip(slots[:-1], intervals, colors):
        offset = 1 if side == -1 else reference + 2
        canonical[2 * slot + 1, offset + start:offset + start + length] = color
    # Empty outer strips may disappear, but no cut inside any bar disappears.
    canonical = _compress(canonical)
    meta = _read_canonical(canonical)
    _require(sorted(b[2] for b in meta["bars"]) == sorted(i[2] for i in intervals),
             "column collapse destroyed a protected bar length")
    output = canonical.copy()
    for row, _, length, _ in meta["bars"]:
        output[row, meta["reference"]] = DIGITS[length]
    inp = _stretch(rng, canonical, meta, anchor, test)
    _require(np.array_equal(_compress(inp), canonical), "stretch lost canonical structure")
    return {"input": inp, "output": output}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Return one reproducible puzzle; counts are explicit integer arguments."""
    _require(not isinstance(seed, (bool, np.bool_))
             and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    rng = np.random.default_rng(int(seed))
    background = int(rng.choice(BACKGROUNDS))
    palette = [color for color in range(10) if color != background]
    marker = int(rng.choice(palette))
    varying_marker = bool(rng.integers(0, 2))
    geometry = int(rng.integers(0, 3))
    puzzle = {"train": [], "test": []}
    used = set()
    anchor_count = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if varying_marker and split == "train" and index == 1:
                local_marker = int(rng.choice([c for c in palette if c != marker]))
            elif varying_marker and not (split == "train" and index == 0):
                local_marker = int(rng.choice(palette))
            else:
                local_marker = marker
            local_geometry = geometry
            if geometry == 2 and not (split == "train" and index == 0):
                local_geometry = int(rng.integers(0, 3))
            anchor = split == "train" and index == 0
            for _ in range(64):
                pair = _make_pair(rng, background, local_marker, local_geometry,
                                  anchor, split == "test")
                n_bars = len(_read_canonical(_compress(pair["input"]))["bars"])
                if split == "train" and index == 1 and n_bars == anchor_count:
                    continue
                key = _key(pair["input"])
                if key in used:
                    continue
                used.add(key)
                puzzle[split].append(pair)
                if anchor:
                    anchor_count = n_bars
                break
            else:
                raise RuntimeError("bounded unique-pair construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
             "pair needs exactly input and output")
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    canonical = _compress(inp)
    meta = _read_canonical(canonical)
    _require(out.shape == canonical.shape, "output must have compressed dimensions")
    _, row_runs = _runs(inp, 0)
    _, column_runs = _runs(inp, 1)
    _require(np.array_equal(np.repeat(np.repeat(canonical, row_runs, axis=0),
                                     column_runs, axis=1), inp),
             "compression does not losslessly reconstruct the input")
    expected = canonical.copy()
    for row, _, length, _ in meta["bars"]:
        expected[row, meta["reference"]] = DIGITS[length]
    _require(np.array_equal(expected, out), "wrong label or changed preserved cell")
    _require(np.count_nonzero(out != canonical) == len(meta["bars"]),
             "exactly one visible label is required per compressed bar")
    _require(inp.shape != canonical.shape, "each input must contain repeated vectors")
    _require(np.array_equal(solve(inp), out), "input-only solve disagrees")
    meta.update(canonical=canonical, row_runs=row_runs, column_runs=column_runs)
    return meta


def validate_generated_puzzle(puzzle):
    """Check returned cells, semantic invariants and the demonstration contract."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    a, b = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "illegal counts")
    seen, metas = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            meta = _inspect_pair(pair)
            key = _key(pair["input"])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            metas[split].append(meta)
    all_meta = metas["train"] + metas["test"]
    _require(len({m["background"] for m in all_meta}) == 1, "background changes")
    anchor = metas["train"][0]
    _require({bar[2] for bar in anchor["bars"]} == set(LENGTHS),
             "anchor must teach all four arithmetic cases")
    _require(max(anchor["row_runs"]) > 1 and max(anchor["column_runs"]) > 1,
             "anchor must teach both compression axes")
    color_lengths, length_colors, lookup = {}, {}, {}
    train_sides = set()
    raw_count_conflict = False
    for pair, m in zip(puzzle["train"], metas["train"]):
        for row, start, length, color in m["bars"]:
            color_lengths.setdefault(color, set()).add(length)
            length_colors.setdefault(length, set()).add(color)
            digit = int(np.asarray(pair["output"])[row, m["reference"]])
            _require(length not in lookup or lookup[length] == digit, "conflicting codes")
            lookup[length] = digit
            raw_length = sum(m["column_runs"][start:start + length])
            raw_count_conflict |= (raw_length - 1) ** 2 % 10 != digit
            train_sides.add(-1 if start < m["reference"] else 1)
    _require(any(len(v) >= 2 for v in color_lengths.values()),
             "one bar color must have multiple lengths")
    _require(any(len(v) >= 2 for v in length_colors.values()),
             "one length must have multiple bar colors")
    _require(any(bar[3] == m["marker_color"] for m in metas["train"] for bar in m["bars"]),
             "training must distinguish marker role from color")
    _require(raw_count_conflict, "training must refute squaring raw pixel lengths")
    _require(len({len(m["bars"]) for m in metas["train"]}) >= 2,
             "training needs different bar counts")
    for pair, m in zip(puzzle["test"], metas["test"]):
        predicted = m["canonical"].copy()
        for row, start, length, _ in m["bars"]:
            _require(length in lookup, "test length not taught")
            _require((-1 if start < m["reference"] else 1) in train_sides,
                     "test side not taught")
            predicted[row, m["reference"]] = lookup[length]
        _require(np.array_equal(predicted, pair["output"]),
                 "taught length lookup does not explain test")
        _require(max(m["column_runs"]) >= 4, "missing stretch-size test shift")
    _require(all(max(m["column_runs"]) <= 3 for m in metas["train"]),
             "training stretch exceeded its curriculum")


def self_test():
    """Bounded, in-memory 42-seed variation audit plus API and adversarial checks."""
    fingerprints, shapes, layouts, backgrounds, bar_counts, sides = set(), set(), set(), set(), set(), set()
    rows_stretched = columns_stretched = raw_conflicts = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                m = _inspect_pair(pair)
                shapes.add(np.asarray(pair["input"]).shape)
                backgrounds.add(m["background"])
                bar_counts.add(len(m["bars"]))
                layouts.add((m["canonical"].shape, m["reference"],
                             tuple(bar[:3] for bar in m["bars"])))
                rows_stretched += max(m["row_runs"]) > 1
                columns_stretched += max(m["column_runs"]) > 1
                for _, start, length, _ in m["bars"]:
                    sides.add(-1 if start < m["reference"] else 1)
                    raw = sum(m["column_runs"][start:start + length])
                    raw_conflicts += (raw - 1) ** 2 % 10 != DIGITS[length]
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)),
                 "seed is not reproducible")
    _require(len(fingerprints) == 42 and len(shapes) >= 50 and len(layouts) >= 150,
             "insufficient structural variation")
    _require(len(backgrounds) == 6 and len(bar_counts) >= 6 and sides == {-1, 1},
             "insufficient palette/count/side variation")
    _require(rows_stretched >= 100 and columns_stretched >= 150 and raw_conflicts >= 200,
             "compression is not meaningfully exercised")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(700 + 10 * a + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A hand-built fixed point with the exact 2/3/4/5 cases, and a length-2
    # witness shifted one column to preserve the long bars' first adjacency.
    rows = ["000000000", "000000070", "000000000", "022222000",
            "000000000", "033330000", "000000000", "044400000",
            "000000000", "055000000", "000000000", "006600000", "000000000"]
    canonical = np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
    _require(np.array_equal(_compress(canonical), canonical), "manual fixture not canonical")
    wanted = canonical.copy()
    wanted[[3, 5, 7, 9, 11], 7] = [6, 9, 4, 1, 1]
    expanded = np.repeat(np.repeat(canonical, [1, 2] + [1] * 11, axis=0),
                         [1, 2, 3, 1, 1, 2, 1, 2, 1], axis=1)
    before = expanded.copy()
    _require(np.array_equal(solve(expanded), wanted), "manual squared-label fixture failed")
    _require(np.array_equal(expanded, before), "solve mutated its input")
    _require(np.array_equal(solve(np.fliplr(expanded)), np.fliplr(wanted)),
             "horizontal reflection failed")
    _require(np.array_equal(solve(np.flipud(expanded)), np.flipud(wanted)),
             "vertical reflection failed")
    # Damage each parent's contribution; validation must reject both.
    good = generate_puzzle(73, 2, 1)
    for replacement in (_compress(good["train"][0]["input"]),
                        np.zeros_like(good["train"][0]["output"])):
        bad = {s: [dict(pair) for pair in good[s]] for s in ("train", "test")}
        bad["train"][0]["output"] = replacement
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt output passed validation")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints),
            "input_shapes": len(shapes), "bar_layouts": len(layouts),
            "backgrounds": sorted(backgrounds), "bar_counts": sorted(bar_counts),
            "raw_count_counterexamples": raw_conflicts,
            "independent_solver": False}
