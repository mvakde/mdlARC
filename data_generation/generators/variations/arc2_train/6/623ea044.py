"""Synthetic single-marker diagonal-cross ARC puzzles."""

import numpy as np


_AUTO_COUNTS = ((2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
                (7, 1), (8, 2), (9, 3), (10, 4))
_SOURCE = frozenset({
    (15, 15, 0, 7, 5, 11), (15, 15, 0, 2, 3, 3),
    (7, 7, 0, 8, 3, 2), (17, 17, 0, 6, 7, 12),
})


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, num_train, num_test):
    train = None if num_train is None else _integer(num_train, "num_train", 2, 10)
    test = None if num_test is None else _integer(num_test, "num_test", 1, 4)
    if train is None and test is None:
        train, test = _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    elif train is None:
        low = max(2, test + 1)
        train = low + seed % (11 - low)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _child_rng(rng):
    return np.random.default_rng(int(rng.integers(0, 2**63 - 1)))


def _shape(rng, kind, low=7, high=30):
    if kind == "square":
        n = int(rng.integers(low, high + 1))
        return n, n
    short = int(rng.integers(low, high - 2))
    long = int(rng.integers(short + 3, high + 1))
    return (short, long) if kind == "wide" else (long, short)


def _compatible_shape(rng, kind, motif, low=7, high=30):
    if motif in ("center", "main", "anti"):
        kind = "square"
    height, width = _shape(rng, kind, low, high)
    if motif == "center" and height % 2 == 0:
        height += 1 if height < high else -1
        width = height
    if motif == "middle_row" and height % 2 == 0:
        height += 1 if height < high else -1
        if kind == "square":
            width = height
    if motif == "middle_col" and width % 2 == 0:
        width += 1 if width < high else -1
        if kind == "square":
            height = width
    return height, width


def _arms(height, width, row, col):
    return (min(row, col), min(row, width - 1 - col),
            min(height - 1 - row, col),
            min(height - 1 - row, width - 1 - col))


def _positions(shape, motif):
    height, width = shape
    result = []
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            arms = _arms(height, width, row, col)
            shortest, longest = min(arms), max(arms)
            matches = {
                "balanced": shortest >= 2 and longest - shortest <= 2,
                "moderate": shortest >= 2 and longest - shortest >= 2,
                "near_edge": shortest == 1,
                "middle_row": height % 2 == 1 and row == height // 2,
                "middle_col": width % 2 == 1 and col == width // 2,
                "center": height % 2 == width % 2 == 1
                          and row == height // 2 and col == width // 2,
                "main": height == width and row == col,
                "anti": height == width and row + col == height - 1,
                "free": True,
            }
            if matches[motif]:
                result.append((row, col))
    return result


def _choose_position(rng, shape, motif, used_geometry, colors,
                     relation=None, anchor=None, forbidden_norms=()):
    height, width = shape
    candidates = [
        (row, col) for row, col in _positions(shape, motif)
        if (height, width, row, col) not in used_geometry
        and (height, width, colors[0], colors[1], row, col) not in _SOURCE
        and (row / (height - 1), col / (width - 1)) not in forbidden_norms
    ]
    if not candidates:
        candidates = [
            (row, col) for row, col in _positions(shape, "free")
            if (height, width, row, col) not in used_geometry
            and (height, width, colors[0], colors[1], row, col) not in _SOURCE
            and (row / (height - 1), col / (width - 1)) not in forbidden_norms
        ]
    if not candidates:
        raise RuntimeError("no unused marker position for shape")
    if relation is not None and anchor is not None:
        anchors = anchor if isinstance(anchor, list) else [anchor]
        candidates.sort(key=lambda point: (
            min((point[0] / (height - 1) - ar) ** 2
                + (point[1] / (width - 1) - ac) ** 2
                for ar, ac in anchors)
        ), reverse=relation == "shift")
        candidates = candidates[:max(1, len(candidates) // 5)]
    row, col = candidates[int(rng.integers(len(candidates)))]
    used_geometry.add((height, width, row, col))
    return row, col


def _colors(order, policy, split, index, train_count, repeat):
    if policy == "stable":
        background = order[0]
        offset = index if split == "train" else index + (0 if repeat else train_count)
        foreground = order[1 + offset % 9]
    elif policy == "changing":
        absolute = index if split == "train" else index + train_count
        background = order[(2 * absolute) % 10]
        foreground = order[(2 * absolute + 1) % 10]
        if split == "test" and repeat:
            foreground = order[1 + (index % min(2, train_count)) * 2]
            if foreground == background:
                background = order[(2 * absolute + 2) % 10]
    else:  # The first two training pairs demonstrate a literal role exchange.
        if split == "train" and index < 2:
            background, foreground = ((order[0], order[1]) if index == 0
                                      else (order[1], order[0]))
        else:
            absolute = index + (0 if split == "train" else train_count)
            background = order[(absolute + 1) % 10]
            foreground = order[absolute % 10]
            if split == "test" and repeat:
                foreground = order[1]
                if foreground == background:
                    background = order[2]
    return int(background), int(foreground)


def _make_pair(shape, background, foreground, row, col):
    inp = np.full(shape, background, dtype=np.int8)
    inp[row, col] = foreground
    rows, cols = np.indices(shape)
    out = np.full(shape, background, dtype=np.int8)
    out[np.abs(rows - row) == np.abs(cols - col)] = foreground
    return {"input": inp, "output": out}


def _motif(rng):
    choices = ("balanced", "moderate", "near_edge", "middle_row",
               "middle_col", "center", "main", "anti")
    return choices[int(rng.choice(8, p=(.20, .21, .13, .10, .10, .08, .09, .09)))]


def _test_shape(rng, regime, motif, train_shapes):
    if regime == "larger_square":
        low = max(17, max(max(shape) for shape in train_shapes) + 1)
        return _compatible_shape(rng, "square", motif, low, 30), motif
    if regime == "shared_size":
        shape = train_shapes[int(rng.integers(len(train_shapes)))]
        return shape, motif if _positions(shape, motif) else "free"
    if regime == "smaller":
        cap = max(9, min(19, max(max(shape) for shape in train_shapes) - 1))
        kind = ("square", "wide", "tall")[int(rng.integers(3))] if cap >= 10 else "square"
        return _compatible_shape(rng, kind, motif, 7, cap), motif
    if regime == "aspect_change":
        rectangles = [s for s in train_shapes if s[0] != s[1]]
        if rectangles:
            reference = rectangles[0]
            kind = "tall" if reference[0] < reference[1] else "wide"
        else:
            kind = ("wide", "tall")[int(rng.integers(2))]
        if motif in ("center", "main", "anti"):
            motif = "moderate"
        return _compatible_shape(rng, kind, motif), motif
    kind = ("square", "wide", "tall")[int(rng.integers(3))]
    return _compatible_shape(rng, kind, motif), motif


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)

    # Sampling regimes are not hidden rules. Every input exposes its own
    # background and marker; the diagonal transformation is universal.
    palette_policy = ("stable", "changing", "exchange")[int(rng.integers(3))]
    test_regime = ("larger_square", "shared_size", "smaller",
                   "aspect_change", "free")[int(rng.integers(5))]
    position_relation = ("similar", "shift", "free")[int(rng.integers(3))]
    repeat = bool(rng.integers(2))
    color_order = tuple(int(x) for x in rng.permutation(10))

    puzzle = {"train": [], "test": []}
    used_geometry = set()
    train_shapes, train_positions = [], []
    for index in range(train_count):
        pair_rng = _child_rng(rng)
        motif = "balanced" if index == 0 else _motif(pair_rng)
        cap = 16 if test_regime == "larger_square" else 30
        if index == 0:
            low = 9 if test_regime != "smaller" else 14
            high = min(cap, 22) if test_regime != "smaller" else 24
            shape = _compatible_shape(pair_rng, "square", motif, low, high)
        else:
            kind = ("wide", "tall")[int(pair_rng.integers(2))] if index == 1 else (
                "square", "wide", "tall")[int(pair_rng.integers(3))]
            shape = _compatible_shape(pair_rng, kind, motif, 7, cap)
            if index == 1 and shape == train_shapes[0]:
                shape = (7, 7) if motif == "center" else (8, 8)
        colors = _colors(color_order, palette_policy, "train", index, train_count, repeat)
        row, col = _choose_position(pair_rng, shape, motif, used_geometry, colors,
                                    forbidden_norms=train_positions)
        puzzle["train"].append(_make_pair(shape, *colors, row, col))
        train_shapes.append(shape)
        train_positions.append((row / (shape[0] - 1), col / (shape[1] - 1)))

    for index in range(test_count):
        pair_rng = _child_rng(rng)
        motif = _motif(pair_rng)
        shape, motif = _test_shape(pair_rng, test_regime, motif, train_shapes)
        colors = _colors(color_order, palette_policy, "test", index, train_count, repeat)
        row, col = _choose_position(pair_rng, shape, motif, used_geometry,
                                    colors, position_relation, train_positions)
        puzzle["test"].append(_make_pair(shape, *colors, row, col))
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _record(pair):
    inp, out = pair["input"], pair["output"]
    assert inp.shape == out.shape
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 2
    singles = values[counts == 1]
    background = values[counts == inp.size - 1]
    assert len(singles) == len(background) == 1
    foreground, background = int(singles[0]), int(background[0])
    row, col = (int(x) for x in np.argwhere(inp == foreground)[0])
    height, width = inp.shape
    assert 1 <= row < height - 1 and 1 <= col < width - 1
    rows, cols = np.indices(inp.shape)
    expected = np.full(inp.shape, background, dtype=inp.dtype)
    expected[np.abs(rows - row) == np.abs(cols - col)] = foreground
    assert np.array_equal(out, expected)
    assert not np.array_equal(inp, out)
    assert (height, width, background, foreground, row, col) not in _SOURCE
    return height, width, background, foreground, row, col


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs, seen_pairs, seen_geometry = set(), set(), set()
    records = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(7 <= side <= 30 for side in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            key = (_grid_key(pair["input"]), _grid_key(pair["output"]))
            assert key[0] not in seen_inputs and key not in seen_pairs
            seen_inputs.add(key[0])
            seen_pairs.add(key)
            record = _record(pair)
            geometry = (record[0], record[1], record[4], record[5])
            assert geometry not in seen_geometry
            seen_geometry.add(geometry)
            records[split].append(record)
    train = records["train"]
    assert len({(r[0], r[1]) for r in train}) >= 2
    assert len({r[3] for r in train}) >= 2
    assert any(min(_arms(r[0], r[1], r[4], r[5])) >= 2 for r in train)
    assert len({(r[4] / (r[0] - 1), r[5] / (r[1] - 1)) for r in train}) >= 2


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[split]) for split in ("train", "test"))


def _predict_from_demonstrations(train, test_input):
    # Independent ray construction using only training pairs and test input.
    for pair in train:
        inp, out = pair["input"], pair["output"]
        values, counts = np.unique(inp, return_counts=True)
        assert len(values) == 2 and 1 in counts
        marker = int(values[np.argmin(counts)])
        background = int(values[np.argmax(counts)])
        row, col = (int(x) for x in np.argwhere(inp == marker)[0])
        prediction = _rays(inp.shape, background, marker, row, col)
        assert np.array_equal(prediction, out)
    values, counts = np.unique(test_input, return_counts=True)
    assert len(values) == 2 and 1 in counts
    marker = int(values[np.argmin(counts)])
    background = int(values[np.argmax(counts)])
    row, col = (int(x) for x in np.argwhere(test_input == marker)[0])
    return _rays(test_input.shape, background, marker, row, col)


def _rays(shape, background, marker, row, col):
    result = np.full(shape, background, dtype=np.int8)
    result[row, col] = marker
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        r, c = row + dr, col + dc
        while 0 <= r < shape[0] and 0 <= c < shape[1]:
            result[r, c] = marker
            r += dr
            c += dc
    return result


def self_test() -> None:
    first = generate_puzzle(123456, 6, 3)
    assert _puzzle_key(first) == _puzzle_key(generate_puzzle(123456, 6, 3))
    assert _puzzle_key(first) != _puzzle_key(generate_puzzle(123457, 6, 3))
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(3000 + (train or 0) * 10 + (test or 0), train, test)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(100, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")

    signatures, shapes, layouts, backgrounds, foregrounds = set(), set(), set(), set(), set()
    train_counts, test_counts = set(), set()
    center = middle_row = middle_col = diagonal = even = wide = tall = 0
    near_edge = balanced = moderate = 0
    stable_palette = changing_palette = role_exchange = repeated_marker = 0
    test_left = test_right = test_center = shared_size = smaller = larger = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_records = [_record(p) for p in puzzle["train"]]
        test_records = [_record(p) for p in puzzle["test"]]
        records = train_records + test_records
        local_backgrounds = {r[2] for r in records}
        stable_palette += len(local_backgrounds) == 1
        changing_palette += len(local_backgrounds) > 1
        role_exchange += any(a[2] == b[3] and a[3] == b[2]
                             for a in train_records for b in train_records if a is not b)
        repeated_marker += bool({r[3] for r in train_records} & {r[3] for r in test_records})
        train_shapes = {(r[0], r[1]) for r in train_records}
        train_max = max(max(r[0], r[1]) for r in train_records)
        for pair, record in zip(puzzle["test"], test_records):
            demonstrations = [{"input": p["input"].copy(), "output": p["output"].copy()}
                              for p in puzzle["train"]]
            assert np.array_equal(_predict_from_demonstrations(
                demonstrations, pair["input"].copy()), pair["output"])
            h, w, _, _, r, c = record
            test_left += c < (w - 1) / 2
            test_right += c > (w - 1) / 2
            test_center += c == (w - 1) / 2
            shared_size += (h, w) in train_shapes
            smaller += max(h, w) < train_max
            larger += min(h, w) > train_max
        for h, w, bg, fg, r, c in records:
            shapes.add((h, w))
            layouts.add((h, w, r, c))
            backgrounds.add(bg)
            foregrounds.add(fg)
            center += h % 2 == w % 2 == 1 and r == h // 2 and c == w // 2
            middle_row += h % 2 == 1 and r == h // 2
            middle_col += w % 2 == 1 and c == w // 2
            diagonal += h == w and (r == c or r + c == h - 1)
            even += h % 2 == 0 or w % 2 == 0
            wide += w > h
            tall += h > w
            arms = _arms(h, w, r, c)
            near_edge += min(arms) == 1
            balanced += min(arms) >= 2 and max(arms) - min(arms) <= 2
            moderate += min(arms) >= 2 and max(arms) - min(arms) >= 2
    assert len(signatures) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 35 and len(layouts) >= 150
    assert len(backgrounds) == 10 and len(foregrounds) == 10
    assert min(center, middle_row, middle_col, diagonal, even, wide, tall) > 0
    assert min(near_edge, balanced, moderate) > 0
    assert min(stable_palette, changing_palette, role_exchange, repeated_marker) > 0
    assert min(test_left, test_right, test_center, shared_size, smaller, larger) > 0
