"""Synthetic instances of the top-aligned, two-cell staircase ARC task."""

import numpy as np


_SOURCE_CASES = {
    (11, 8, 4, 2),
    (14, 7, 5, 3),
    (14, 8, 7, 4),
    (15, 9, 7, 3),
}
_SOURCE_COLORS = (0, 5, 8, 6)


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _choose_counts(rng, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _construct_grids(height, width, length, column, colors):
    background, seed, left, right = colors
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    inp[:length, column] = seed
    out[:length, column] = seed
    for c in range(column):
        out[:min(height, length + 2 * (column - c)), c] = left
    for c in range(column + 1, width):
        extent = max(0, length - 2 * (c - column))
        out[:extent, c] = right
    return inp, out


def _sample_geometry(rng, anchor, regime):
    if anchor:
        # One demonstration must make several unsaturated steps on each side
        # visible, even in a puzzle with just two training examples.
        length = int(rng.integers(5, 11))
        column = int(rng.integers(3, 7))
        height = int(rng.integers(length + 2 * column + 1,
                                  min(30, length + 2 * column + 7) + 1))
        width = int(rng.integers(column + 3, min(23, column + 10) + 1))
        return height, width, length, column

    if regime == "source_train":
        width = int(rng.integers(5, 20))
        length = int(rng.integers(2, 15))
    else:
        width = int(rng.integers(4, 24))
        length = int(rng.integers(1, 17))

    location = int(rng.integers(0, 10))
    if location == 0:
        column = 0
    elif location == 1:
        column = width - 1
    else:
        column = int(rng.integers(0, width))
    if column == 0 and length <= 2:
        column = 1

    # Sometimes choose a canvas height independently of the staircase.
    # Otherwise choose an offset around its full left extent. Width, color,
    # and this height choice use separate draws.
    if int(rng.integers(0, 3)) == 0:
        height = int(rng.integers(length + 1, 31))
    else:
        offset = int(rng.integers(-8, 10))
        height = int(np.clip(length + 2 * column + offset, length + 1, 30))
    return height, width, length, column


def _sample_visible_colors(rng, left, right, fixed):
    if fixed is not None:
        return fixed
    choices = [color for color in range(10) if color not in (left, right)]
    selected = rng.choice(choices, size=2, replace=False)
    return int(selected[0]), int(selected[1])


def _facts(inp):
    assert inp.ndim == 2
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 2
    background = int(values[np.argmax(counts)])
    seed = int(values[np.argmin(counts)])
    rows, columns = np.nonzero(inp != background)
    assert len(rows) > 0 and np.all(columns == columns[0])
    length = len(rows)
    assert np.array_equal(rows, np.arange(length))
    assert np.all(inp[:length, int(columns[0])] == seed)
    return background, seed, length, int(columns[0])


def _pair_key(pair):
    return tuple(
        (grid.shape, grid.tobytes())
        for grid in (pair["input"], pair["output"])
    )


def _make_pair(geometry, colors):
    inp, out = _construct_grids(*geometry, colors)
    return {"input": inp, "output": out}


def _sample_unique_pair(rng, anchor, regime, colors, used_geometry):
    geometry = _sample_geometry(rng, anchor, regime)
    # Deterministic, bounded fallback for a rare duplicate or exact source
    # copy. Keep length and seed column while changing the canvas.
    candidates = [geometry]
    height, width, length, column = geometry
    for candidate_height in range(length + 1, 31):
        for candidate_width in range(max(4, column + 1), 31):
            candidates.append((candidate_height, candidate_width, length, column))
    for candidate in candidates:
        if candidate in used_geometry:
            continue
        if colors == _SOURCE_COLORS and candidate in _SOURCE_CASES:
            continue
        used_geometry.add(candidate)
        return _make_pair(candidate, colors)
    raise AssertionError("no unique geometry available")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    if int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    n_train, n_test = _choose_counts(rng, num_train, num_test)

    palette = tuple(int(x) for x in rng.choice(10, size=4, replace=False))
    base_background, base_seed, left, right = palette
    variable_visible_colors = bool(rng.integers(0, 2))
    fixed = None if variable_visible_colors else (base_background, base_seed)
    used_geometry = set()
    train = []
    test = []

    # Independently seeded pair streams keep pair choices from becoming a
    # sequence of correlated height/width/length/color templates.
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=n_train + n_test)
    first_visible = (base_background, base_seed)
    for index in range(n_train):
        pair_rng = np.random.default_rng(pair_seeds[index])
        if index == 0:
            visible = first_visible
        else:
            visible = _sample_visible_colors(pair_rng, left, right, fixed)
            if variable_visible_colors and index == 1 and visible == first_visible:
                allowed = [x for x in range(10) if x not in (left, right, visible[1])]
                visible = (allowed[(allowed.index(visible[0]) + 1) % len(allowed)],
                           visible[1])
        colors = (visible[0], visible[1], left, right)
        train.append(_sample_unique_pair(pair_rng, index == 0,
                                         "source_train", colors, used_geometry))

    # The rich demonstration is deliberately not tied to train pair 1.
    train = [train[int(i)] for i in rng.permutation(n_train)]
    for index in range(n_test):
        pair_rng = np.random.default_rng(pair_seeds[n_train + index])
        visible = _sample_visible_colors(pair_rng, left, right, fixed)
        colors = (visible[0], visible[1], left, right)
        test.append(_sample_unique_pair(pair_rng, False, "target_test",
                                        colors, used_geometry))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4
    assert len(train) > len(test)

    pair_keys, input_keys, geometry_keys = set(), set(), set()
    left_values, right_values = set(), set()
    facts = []
    for regime, pairs in (("source_train", train), ("target_test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            height, width = inp.shape
            background, seed, length, column = _facts(inp)
            assert length < height and width >= 4
            assert np.array_equal(out[:length, column], inp[:length, column])
            geometry = (height, width, length, column)
            assert geometry not in geometry_keys
            geometry_keys.add(geometry)
            inp_key = (inp.shape, inp.tobytes())
            assert inp_key not in input_keys
            input_keys.add(inp_key)
            pair_key = _pair_key(pair)
            assert pair_key not in pair_keys
            pair_keys.add(pair_key)
            if column > 0:
                left_values.add(int(out[0, column - 1]))
            if column + 1 < width and length > 2:
                right_values.add(int(out[0, column + 1]))
            facts.append((regime, geometry, background, seed, pair))

    # Recover output-only roles solely from training outputs. A rich
    # demonstration exposes both roles even when other pairs are one-sided.
    train_left = {int(pair["output"][0, c - 1])
                  for pair in train
                  for _, _, _, c in [_facts(pair["input"])]
                  if c > 0}
    train_right = {int(pair["output"][0, c + 1])
                   for pair in train
                   for _, _, length, c in [_facts(pair["input"])]
                   if c + 1 < pair["input"].shape[1] and length > 2}
    assert len(train_left) == len(train_right) == 1
    left, right = next(iter(train_left)), next(iter(train_right))
    assert left != right
    assert left_values == {left} and right_values == {right}

    rich = False
    for regime, geometry, background, seed, pair in facts:
        height, width, length, column = geometry
        assert len({background, seed, left, right}) == 4
        expected_in, expected_out = _construct_grids(
            height, width, length, column, (background, seed, left, right))
        assert np.array_equal(pair["input"], expected_in)
        assert np.array_equal(pair["output"], expected_out)
        assert not ((background, seed, left, right) == _SOURCE_COLORS
                    and geometry in _SOURCE_CASES)
        if regime == "source_train":
            assert 5 <= width <= 30 and 2 <= length <= 14
            # Two adjacent left bars must have distinct, unsaturated lengths;
            # two right bars must also be present and decline by two.
            if column >= 2 and width - column >= 3 and length >= 5:
                if height >= length + 2 * column:
                    rich = True
        else:
            assert 4 <= width <= 30 and 1 <= length <= 16
    assert rich


def _puzzles_equal(a, b):
    return all(
        np.array_equal(pa[key], pb[key])
        for split in ("train", "test")
        for pa, pb in zip(a[split], b[split])
        for key in ("input", "output")
    ) and all(len(a[split]) == len(b[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    for index, (nt, ne) in enumerate(((2, 1), (3, 1), (6, 2),
                                       (9, 3), (10, 1), (10, 4))):
        puzzle = generate_puzzle(500 + index, nt, ne)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (nt, ne)
        validate_generated_puzzle(puzzle)
    for index, (nt, ne) in enumerate(((None, 1), (None, 4),
                                       (2, None), (10, None))):
        puzzle = generate_puzzle(600 + index, nt, ne)
        if nt is not None:
            assert len(puzzle["train"]) == nt
        if ne is not None:
            assert len(puzzle["test"]) == ne
        validate_generated_puzzle(puzzle)
    for kwargs in ({"num_train": 1}, {"num_train": 11},
                   {"num_test": 0}, {"num_test": 5},
                   {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": 2.0}):
        try:
            generate_puzzle(1, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal counts: {kwargs}")
    assert _puzzles_equal(generate_puzzle(8123, 8, 3),
                          generate_puzzle(8123, 8, 3))
    assert len({tuple(_pair_key(pair) for pair in puzzle["train"] + puzzle["test"])
                for puzzle in puzzles}) == 42

    lengths, columns, shapes, layouts, palettes = set(), set(), set(), set(), set()
    clipped_tests = small_tests = novel_test_lengths = novel_test_columns = 0
    variable_color_puzzles = stable_color_puzzles = 0
    short_bars = long_bars = one_sided = spacious_bottom = tight_right = 0
    for puzzle in puzzles:
        train_facts = [_facts(p["input"]) for p in puzzle["train"]]
        backgrounds = {(f[0], f[1]) for f in train_facts}
        variable_color_puzzles += len(backgrounds) > 1
        stable_color_puzzles += len(backgrounds) == 1
        train_lengths = {f[2] for f in train_facts}
        train_columns = {f[3] for f in train_facts}
        max_height = max(p["input"].shape[0] for p in puzzle["train"])
        max_width = max(p["input"].shape[1] for p in puzzle["train"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                height, width = pair["input"].shape
                background, seed, length, column = _facts(pair["input"])
                left = int(next(p["output"][0, _facts(p["input"])[3] - 1]
                                for p in puzzle["train"] if _facts(p["input"])[3] > 0))
                right = int(next(p["output"][0, _facts(p["input"])[3] + 1]
                                 for p in puzzle["train"]
                                 if _facts(p["input"])[3] + 1 < p["input"].shape[1]
                                 and _facts(p["input"])[2] > 2))
                # Independent cell-by-cell construction of the staircase.
                for row in range(height):
                    for col in range(width):
                        distance = col - column
                        extent = length - 2 * distance
                        expected = (seed if distance == 0 else
                                    left if distance < 0 else right)
                        if row >= extent:
                            expected = background
                        assert int(pair["output"][row, col]) == expected
                lengths.add(length)
                columns.add(column)
                shapes.add((height, width))
                layouts.add((height, width, length, column))
                palettes.add((background, seed, left, right))
                short_bars += length <= 3
                long_bars += length >= 10
                one_sided += column in (0, width - 1)
                spacious_bottom += height - min(height, length + 2 * column) >= 5
                tight_right += width - 1 - column <= max(0, (length - 1) // 2)
                if split == "test":
                    clipped_tests += height <= length + 2 * column
                    small_tests += height <= max_height or width <= max_width
                    novel_test_lengths += length not in train_lengths
                    novel_test_columns += column not in train_columns
    assert len(lengths) >= 12 and len(columns) >= 10
    assert len(shapes) >= 50 and len(layouts) >= 100 and len(palettes) >= 30
    assert variable_color_puzzles >= 10 and stable_color_puzzles >= 10
    assert min(clipped_tests, small_tests, novel_test_lengths,
               novel_test_columns, short_bars, long_bars, one_sided,
               spacious_bottom, tight_right) > 0
