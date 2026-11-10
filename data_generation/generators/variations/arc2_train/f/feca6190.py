"""Generate variations of the one-row, upward-right ray puzzle."""

from itertools import combinations

import numpy as np


_SOURCE_ROWS = {
    (0, 9, 0, 8, 4), (0, 0, 2, 0, 0), (4, 0, 6, 0, 8),
    (0, 4, 0, 0, 0), (1, 0, 7, 0, 0), (0, 6, 7, 8, 9),
}


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _legal(width, count):
    return 3 <= width <= 10 and 1 <= count < width and width * count <= 30


def sample_puzzle_parameters(rng):
    # Background is a learned puzzle convention; width is a pair property.
    return {
        "background": int(rng.integers(0, 10)),
        "base_width": int(rng.integers(5, 9)),
        "vary_width": bool(rng.random() < 0.72),
    }


def _construct_output(input_grid, background):
    row = input_grid[0]
    side = int(np.count_nonzero(row != background)) * len(row)
    output = np.full((side, side), background, dtype=np.int8)
    for shift in range(side):
        length = min(len(row), side - shift)
        output[side - 1 - shift, shift : shift + length] = row[:length]
    return output


def _layout(rng, width, count):
    if count == 1:
        choice = rng.random()
        if choice < 0.18:
            return (width - 1,)
        if choice < 0.34:
            return (0,)
    if count > 1 and rng.random() < 0.22:
        edge = 0 if rng.random() < 0.5 else width - 1
        rest = rng.choice([p for p in range(width) if p != edge], size=count - 1, replace=False)
        return tuple(sorted([edge, *(int(p) for p in rest)]))
    return tuple(sorted(int(p) for p in rng.choice(width, size=count, replace=False)))


def _colors(rng, background, count, repeat):
    palette = np.array([c for c in range(10) if c != background], dtype=np.int8)
    if repeat and count >= 2:
        available = 1 if count == 2 else int(rng.integers(2, count))
        chosen = rng.choice(palette, size=available, replace=False)
        result = np.concatenate((chosen, rng.choice(chosen, size=count - available, replace=True)))
        rng.shuffle(result)
        return result
    return rng.choice(palette, size=count, replace=False)


def _input_key(grid):
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


def _make_pair(rng, width, count, background, repeat, used_inputs, used_test_layouts=None):
    # Prefer fresh test geometries; recolor only after a finite set is exhausted.
    for fresh_geometry in (True, False):
        for _ in range(160):
            positions = _layout(rng, width, count)
            geometry = (width, positions)
            if fresh_geometry and used_test_layouts is not None and geometry in used_test_layouts:
                continue
            row = np.full(width, background, dtype=np.int8)
            row[list(positions)] = _colors(rng, background, count, repeat)
            grid = row.reshape(1, width)
            key = _input_key(grid)
            if key in used_inputs or tuple(int(x) for x in row) in _SOURCE_ROWS:
                continue
            used_inputs.add(key)
            if used_test_layouts is not None:
                used_test_layouts.add(geometry)
            return {"input": grid, "output": _construct_output(grid, background)}

    # Bounded deterministic fallback for collisions with earlier inputs.
    palette = [c for c in range(10) if c != background]
    for positions in combinations(range(width), count):
        for offset in range(9):
            row = np.full(width, background, dtype=np.int8)
            colors = [palette[(offset + i) % 9] for i in range(count)]
            if repeat and count >= 2:
                colors[-1] = colors[0]
            row[list(positions)] = colors
            grid = row.reshape(1, width)
            key = _input_key(grid)
            if key not in used_inputs and tuple(int(x) for x in row) not in _SOURCE_ROWS:
                used_inputs.add(key)
                if used_test_layouts is not None:
                    used_test_layouts.add((width, positions))
                return {"input": grid, "output": _construct_output(grid, background)}
    raise ValueError("no unique input available")


def _random_width(rng, count):
    choices = [w for w in range(3, 11) if _legal(w, count)]
    # Give short rows and long sparse rows enough weight to be visible
    # across a modest seed batch, while still sampling interior widths.
    weights = np.array([2.5 if w == 3 else 2.0 if w >= 9 else 1.0
                        for w in choices], dtype=float)
    return int(rng.choice(choices, p=weights / weights.sum()))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    background = parameters["background"]
    base = parameters["base_width"]
    variable = parameters["vary_width"] and num_train >= 3

    # Same width/different k teaches the count factor. The third training
    # row holds k fixed and changes width in the variable-width regime.
    train_specs = [(base, 1), (base, 3)]
    if num_train >= 3:
        if variable:
            choices = [w for w in range(4, 11) if w != base and _legal(w, 3)]
            train_specs.append((int(rng.choice(choices)), 3))
        else:
            train_specs.append((base, 2))
    while len(train_specs) < num_train:
        count = int(rng.choice([1, 2, 3, 4], p=[0.17, 0.28, 0.32, 0.23]))
        if variable:
            width = _random_width(rng, count)
        else:
            width = base
            if not _legal(width, count):
                count = int(rng.choice([1, 2, 3]))
        train_specs.append((width, count))

    used_inputs = set()
    train = []
    for index, (width, count) in enumerate(train_specs):
        repeat = count >= 2 and (index == 1 or rng.random() < 0.38)
        train.append(_make_pair(rng, width, count, background, repeat, used_inputs))
    rng.shuffle(train)

    used_layouts = {
        (pair["input"].shape[1], tuple(int(p) for p in np.flatnonzero(pair["input"][0] != background)))
        for pair in train
    }
    test = []
    for _ in range(num_test):
        if variable:
            count = int(rng.choice([1, 2, 3, 4, 5], p=[0.13, 0.20, 0.25, 0.27, 0.15]))
            width = _random_width(rng, count)
        else:
            options = [k for k in range(1, 6) if _legal(base, k)]
            weights = np.array([0.13, 0.20, 0.25, 0.27, 0.15][:len(options)], dtype=float)
            count = int(rng.choice(options, p=weights / weights.sum()))
            width = base
        repeat = count >= 2 and rng.random() < 0.42
        test.append(_make_pair(rng, width, count, background, repeat, used_inputs, used_layouts))

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _output_background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[name], list) for name in ("train", "test"))
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    backgrounds = [_output_background(pair["output"]) for pair in train]
    assert len(set(backgrounds)) == 1
    background = backgrounds[0]
    seen_inputs = set()
    seen_pairs = set()
    train_counts = set()
    train_widths = set()
    repeated_in_train = False
    multicolor_in_train = False
    width_count_pairs = set()

    for split_name in ("train", "test"):
        for pair in puzzle[split_name]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= size <= 30 for size in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            source, output = pair["input"], pair["output"]
            assert source.shape[0] == 1
            width = source.shape[1]
            row = source[0]
            positions = np.flatnonzero(row != background)
            count = len(positions)
            assert _legal(width, count)
            assert _output_background(output) == background
            side = width * count
            assert output.shape == (side, side)
            assert tuple(int(x) for x in row) not in _SOURCE_ROWS
            assert not np.array_equal(source, output)
            key = _input_key(source)
            pair_key = (key, _input_key(output))
            assert key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(key)
            seen_pairs.add(pair_key)

            # Cell-by-cell ray check independent of the row-copy constructor.
            expected = np.full((side, side), background, dtype=np.int8)
            for column in positions:
                for shift in range(side - int(column)):
                    expected[side - 1 - shift, int(column) + shift] = row[column]
            assert np.array_equal(output, expected)
            assert output[-1, :width].tolist() == row.tolist()

            if split_name == "train":
                train_counts.add(count)
                train_widths.add(width)
                width_count_pairs.add((width, count))
                color_count = len(set(int(row[p]) for p in positions))
                repeated_in_train |= color_count < count
                multicolor_in_train |= color_count >= 2

    assert 1 in train_counts and 3 in train_counts
    assert repeated_in_train
    assert multicolor_in_train
    if len(train_widths) > 1:
        assert any(w1 != w2 and k1 == k2 for w1, k1 in width_count_pairs
                   for w2, k2 in width_count_pairs)


def _puzzle_key(puzzle):
    return tuple(
        tuple((_input_key(p["input"]), _input_key(p["output"])) for p in puzzle[split])
        for split in ("train", "test")
    )


def _must_reject(**kwargs):
    try:
        generate_puzzle(123, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError for {kwargs}")


def self_test() -> None:
    cases = [(None, None), (6, 3), (2, 1), (10, 1), (10, 4),
             (None, 1), (None, 4), (2, None), (10, None)]
    for index, (n_train, n_test) in enumerate(cases):
        args = {"num_train": n_train, "num_test": n_test}
        first = generate_puzzle(9000 + index, **args)
        second = generate_puzzle(9000 + index, **args)
        validate_generated_puzzle(first)
        assert _puzzle_key(first) == _puzzle_key(second)
        if n_train is not None:
            assert len(first["train"]) == n_train
        if n_test is not None:
            assert len(first["test"]) == n_test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4}):
        _must_reject(**kwargs)

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_output_background(p["train"][0]["output"]) for p in puzzles}) >= 7

    widths = set()
    counts = set()
    layouts = set()
    test_layouts = set()
    sides = set()
    variable_width_puzzles = 0
    repeated_tests = 0
    separated_tests = 0
    test_starts_at_zero = 0
    singleton_last = 0
    train_four = 0
    test_five = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        widths_here = {pair["input"].shape[1] for split in ("train", "test") for pair in puzzle[split]}
        variable_width_puzzles += len(widths_here) > 1
        bg = _output_background(puzzle["train"][0]["output"])
        for split in ("train", "test"):
            for pair in puzzle[split]:
                row = pair["input"][0]
                width = len(row)
                positions = tuple(int(p) for p in np.flatnonzero(row != bg))
                count = len(positions)
                widths.add(width)
                counts.add(count)
                layouts.add((width, positions))
                sides.add(pair["output"].shape[0])
                if split == "train":
                    train_four += count == 4
                else:
                    test_layouts.add((width, positions))
                    repeated_tests += len(set(int(row[p]) for p in positions)) < count
                    separated_tests += any(b - a > 1 for a, b in zip(positions, positions[1:]))
                    test_starts_at_zero += positions[0] == 0
                    singleton_last += count == 1 and positions[0] == width - 1
                    test_five += count == 5
    assert widths == set(range(3, 11))
    assert counts == {1, 2, 3, 4, 5}
    assert len(layouts) >= 100 and len(test_layouts) >= 45
    assert len(sides) >= 16
    assert variable_width_puzzles >= 15
    assert repeated_tests >= 10 and separated_tests >= 15
    assert test_starts_at_zero >= 8 and singleton_last >= 1
    assert train_four >= 8 and test_five >= 2
