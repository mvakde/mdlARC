"""Concentric nine-point rings compressed to a three-by-three square."""

import numpy as np


_SOURCE_SIGNATURES = frozenset({
    (0, 6, 7, 2, 4, 3), (0, 1, 9, 5, 8, 7),
    (0, 2, 1, 3, 6, 4), (0, 7, 5, 6, 4, 2),
})


def _ensure(condition, message):
    if not condition:
        raise AssertionError(message)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        minimum = max(2, test + 1)
        train = minimum + seed % (11 - minimum)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _geometry(rng, regime, rectangular, fixed_scale):
    if fixed_scale:
        oy, iy = 2, 1
    elif regime == "target_test" and rng.random() < 0.62:
        oy = int(rng.integers(4, 8))
        iy = int(rng.integers(1, oy))
    else:
        oy = int(rng.integers(2, 7))
        iy = int(rng.integers(1, oy))
    if rectangular:
        ox = int(rng.integers(2, 8 if regime == "target_test" else 7))
        ix = int(rng.integers(1, ox))
        if (ox, ix) == (oy, iy):
            ox, ix = (2 if oy != 2 else 3), 1
    else:
        ox, ix = oy, iy
    if not rectangular and rng.random() < (0.12 if not fixed_scale else 0.22):
        oy = ox = 2
        iy = ix = 1
        margins = (0, 0, 0, 0)
    else:
        margins = tuple(int(v) for v in rng.integers(
            0, 6 if regime == "target_test" else 5, size=4))
    return oy, ox, iy, ix, *margins


def _colors(rng, background, mode):
    palette = [int(v) for v in rng.permutation(
        [color for color in range(10) if color != background])]
    if mode == "distinct":
        return tuple(palette)
    if mode == "diagonal":
        a, b, c, d, center = palette[:5]
        return (a, b, b, a, c, d, d, c, center)
    if mode == "reuse":
        for _ in range(24):
            pool = palette[:int(rng.integers(4, 8))]
            points = [int(pool[int(rng.integers(len(pool)))]) for _ in range(8)]
            points[4] = points[0]
            points.append(points[int(rng.integers(8))])
            if (len(set(points)) >= 4 and len(set(points[:4])) >= 2
                    and len(set(points[4:8])) >= 2):
                return tuple(points)
        return (palette[0], palette[1], palette[2], palette[3],
                palette[0], palette[4], palette[5], palette[6], palette[1])
    pool = palette[:int(rng.integers(5, 9))]
    points = [int(v) for v in pool]
    points += [int(pool[int(rng.integers(len(pool)))]) for _ in range(9 - len(points))]
    return tuple(int(v) for v in rng.permutation(points))


def _positions(geometry):
    oy, ox, iy, ix, top, _, left, _ = geometry
    cy, cx = top + oy, left + ox
    return ((cy - oy, cx - ox), (cy - oy, cx + ox),
            (cy + oy, cx - ox), (cy + oy, cx + ox),
            (cy - iy, cx - ix), (cy - iy, cx + ix),
            (cy + iy, cx - ix), (cy + iy, cx + ix), (cy, cx))


def _make_pair(geometry, background, points):
    oy, ox, _, _, top, bottom, left, right = geometry
    inp = np.full((2 * oy + 1 + top + bottom,
                   2 * ox + 1 + left + right), background, dtype=np.int8)
    for (row, col), color in zip(_positions(geometry), points):
        inp[row, col] = color
    out = np.array(((points[0], points[4], points[1]),
                    (points[6], points[8], points[5]),
                    (points[2], points[7], points[3])), dtype=np.int8)
    return {"input": inp, "output": out}


def _key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _is_source(pair):
    inp = pair["input"]
    if inp.shape != (5, 5):
        return False
    signature = (int(inp[0, 1]), int(inp[0, 0]), int(inp[0, 4]),
                 int(inp[1, 1]), int(inp[1, 3]), int(inp[2, 2]))
    if signature not in _SOURCE_SIGNATURES:
        return False
    bg, a, b, c, d, e = signature
    original = _make_pair((2, 2, 1, 1, 0, 0, 0, 0), bg,
                          (a, b, b, a, c, d, d, c, e))
    return _key(inp) == _key(original["input"]) and _key(pair["output"]) == _key(original["output"])


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or
            not isinstance(seed, (int, np.integer)) or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # The mapping is fixed; this puzzle-level choice only determines whether
    # independently sized horizontal and vertical ring radii are demonstrated.
    fixed_scale = bool(rng.random() < 0.10)
    allow_rectangles = not fixed_scale and bool(rng.random() < 0.68)
    puzzle = {"train": [], "test": []}
    seen_inputs, all_geometries, train_geometries = set(), set(), []
    first_background = int(rng.integers(10))
    for index in range(train_count + test_count):
        training = index < train_count
        regime = "source_train" if training else "target_test"
        if index == 0:
            rectangular, mode = False, "distinct"
        elif index == 1:
            rectangular = allow_rectangles
            mode = "diagonal" if rng.random() < 0.42 else "reuse"
        else:
            rectangular = allow_rectangles and rng.random() < (0.7 if not training else 0.5)
            mode = ("diagonal", "distinct", "reuse", "partial")[int(rng.integers(4))]
        background = first_background if index == 0 else int(rng.integers(10))
        if index == 1:
            background = (first_background + 1 + int(rng.integers(9))) % 10
        for _ in range(64):
            geometry = _geometry(rng, regime, rectangular, fixed_scale)
            if fixed_scale and index == 0:
                geometry = (2, 2, 1, 1, 0, 0, 0, 0)
            if index == 1 and not fixed_scale and geometry[:4] == train_geometries[0][:4]:
                continue
            if geometry in all_geometries:
                continue
            points = _colors(rng, background, mode)
            pair = _make_pair(geometry, background, points)
            key = _key(pair["input"])
            if key not in seen_inputs and not _is_source(pair):
                break
        else:
            # A top margin outside the sampling range gives a unique, legal
            # canvas while preserving the geometry demonstrated in training.
            if rectangular:
                oy, ox = 3, 2
            elif index == 1 and not fixed_scale:
                oy = ox = 2 if train_geometries[0][0] != 2 else 3
            else:
                oy = ox = 2
            geometry = (oy, ox, 1, 1, 6 + index, 0, 0, 0)
            pair = _make_pair(geometry, background, _colors(rng, background, "distinct"))
            key = _key(pair["input"])
            _ensure(key not in seen_inputs and not _is_source(pair), "fallback collision")
        seen_inputs.add(key)
        all_geometries.add(geometry)
        if training:
            train_geometries.append(geometry)
            puzzle["train"].append(pair)
        else:
            puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    inp, out = pair["input"], pair["output"]
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[int(np.argmax(counts))])
    rows, cols = np.nonzero(inp != background)
    _ensure(len(rows) == 9, "input must have exactly nine visible points")
    yr, xr = sorted(set(int(v) for v in rows)), sorted(set(int(v) for v in cols))
    _ensure(len(yr) == 5 and len(xr) == 5, "nested rings need five rows and columns")
    cy, cx = yr[2], xr[2]
    oy, iy, ox, ix = cy - yr[0], cy - yr[1], cx - xr[0], cx - xr[1]
    _ensure(0 < iy < oy and 0 < ix < ox, "rings must be strictly nested")
    _ensure(yr == [cy - oy, cy - iy, cy, cy + iy, cy + oy] and
            xr == [cx - ox, cx - ix, cx, cx + ix, cx + ox],
            "rings must share one center")
    geometry = (oy, ox, iy, ix, yr[0], inp.shape[0] - 1 - yr[4],
                xr[0], inp.shape[1] - 1 - xr[4])
    positions = _positions(geometry)
    _ensure(set(zip(rows.tolist(), cols.tolist())) == set(positions),
            "foreground must be exactly two rings and center")
    points = tuple(int(inp[r, c]) for r, c in positions)
    expected = _make_pair(geometry, background, points)["output"]
    _ensure(out.shape == (3, 3) and np.array_equal(out, expected),
            "output must copy points in clockwise quadrant order")
    return geometry, background, points


def validate_generated_puzzle(puzzle: dict) -> None:
    _ensure(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
            "puzzle must contain exactly train and test")
    _ensure(all(isinstance(puzzle[s], list) for s in ("train", "test")),
            "train and test must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _ensure(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    features = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _ensure(isinstance(pair, dict) and set(pair) == {"input", "output"},
                    "each pair must have exactly input and output")
            for grid in pair.values():
                _ensure(isinstance(grid, np.ndarray) and grid.ndim == 2,
                        "grids must be 2D NumPy arrays")
                _ensure(all(1 <= size <= 30 for size in grid.shape), "invalid grid dimensions")
                _ensure(np.issubdtype(grid.dtype, np.integer), "grids must use integer dtype")
                _ensure(int(grid.min()) >= 0 and int(grid.max()) <= 9, "invalid colors")
            ik, ok = _key(pair["input"]), _key(pair["output"])
            _ensure(ik not in seen_inputs and (ik, ok) not in seen_pairs,
                    "duplicate pair or input")
            seen_inputs.add(ik)
            seen_pairs.add((ik, ok))
            _ensure(not _is_source(pair), "exact source pair copy")
            features[split].append(_inspect_pair(pair))
    train, tests = features["train"], features["test"]
    _ensure(len(set(train[0][2])) == 9,
            "training must demonstrate all nine positional correspondences")
    _ensure(train[0][0][0] == train[0][0][1] and
            train[0][0][2] == train[0][0][3],
            "first training pair must demonstrate square rings")
    _ensure(train[0][1] != train[1][1], "training backgrounds must vary")
    _ensure(train[0][0] != train[1][0], "training must vary placement, spacing or scale")
    train_has_rectangle = any(g[0] != g[1] or g[2] != g[3] for g, _, _ in train)
    _ensure(not any(g[0] != g[1] or g[2] != g[3] for g, _, _ in tests)
            or train_has_rectangle, "rectangular tests need training demonstration")
    train_geometry = {g for g, _, _ in train}
    _ensure(all(g not in train_geometry for g, _, _ in tests),
            "test geometry must be a new combination")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _self_test_observed_values(grid):
    colors, counts = np.unique(grid, return_counts=True)
    background = colors[int(np.argmax(counts))]
    rows, cols = np.nonzero(grid != background)
    row_levels = sorted(set(int(v) for v in rows))
    col_levels = sorted(set(int(v) for v in cols))
    _ensure(len(rows) == 9 and len(row_levels) == 5 and len(col_levels) == 5,
            "self-test could not recover nine points")
    r0, r1, rc, r3, r4 = row_levels
    c0, c1, cc, c3, c4 = col_levels
    return tuple(int(grid[r, c]) for r, c in (
        (r0, c0), (r0, c4), (r4, c0), (r4, c4),
        (r1, c1), (r1, c3), (r3, c1), (r3, c3), (rc, cc)))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _ensure(len({_puzzle_key(p) for p in puzzles}) >= 40, "too few distinct puzzles")
    _ensure(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack range")
    _ensure(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack range")
    _ensure(_puzzle_key(generate_puzzle(123456, 7, 3)) ==
            _puzzle_key(generate_puzzle(123456, 7, 3)), "not deterministic")
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                              (2001, None, 1), (2002, None, 4),
                              (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        _ensure(train is None or len(puzzle["train"]) == train, "train count ignored")
        _ensure(test is None or len(puzzle["test"]) == test, "test count ignored")
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (2, 3), (4, 4)):
        try:
            generate_puzzle(3000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    geometries, backgrounds, color_counts = set(), set(), set()
    rectangular = offset = large = reuse = asymmetric = 0
    gap_outer = gap_center = combined = repeated_centers = novel_centers = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        # Infer the transfer from training grids alone, then predict all
        # outputs from their inputs without using generation parameters.
        first = puzzle["train"][0]
        first_values = _self_test_observed_values(first["input"])
        _ensure(len(set(first_values)) == 9, "training mapping is ambiguous")
        source_index = {color: i for i, color in enumerate(first_values)}
        learned_order = tuple(source_index[int(value)] for value in first["output"].flat)
        for pair in puzzle["train"] + puzzle["test"]:
            values = _self_test_observed_values(pair["input"])
            prediction = np.array([values[i] for i in learned_order]).reshape(3, 3)
            _ensure(np.array_equal(prediction, pair["output"]),
                    "training-only prediction mismatch")
        train_centers = {_inspect_pair(p)[2][8] for p in puzzle["train"]}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                geometry, bg, points = _inspect_pair(pair)
                oy, ox, iy, ix, top, bottom, left, right = geometry
                geometries.add(geometry)
                backgrounds.add(bg)
                color_counts.add(len(set(points)))
                is_rect = oy != ox or iy != ix
                is_offset = len({top, bottom, left, right}) > 1
                is_asym = (points[0] != points[3] or points[1] != points[2]
                           or points[4] != points[7] or points[5] != points[6])
                is_reuse = bool(set(points[:4]) & set(points[4:8])) or points[8] in points[:8]
                rectangular += is_rect
                offset += is_offset
                large += max(oy, ox) >= 4
                gap_outer += oy - iy > 1 or ox - ix > 1
                gap_center += iy > 1 or ix > 1
                reuse += is_reuse
                asymmetric += is_asym
                combined += is_rect and is_offset and is_asym and is_reuse
                if split == "test":
                    repeated_centers += points[8] in train_centers
                    novel_centers += points[8] not in train_centers
    _ensure(len(geometries) >= 150 and len(backgrounds) == 10 and len(color_counts) >= 5,
            "insufficient geometry or palette diversity")
    _ensure(min(rectangular, offset, large, reuse, asymmetric,
                gap_outer, gap_center) >= 35 and combined >= 5,
            "review axes or combinations are underrepresented")
    _ensure(repeated_centers > 0 and novel_centers > 0,
            "center colors remain scheduled across train and test")
