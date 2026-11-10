"""Generate dotted, crossing-line ARC puzzles."""

import numpy as np


_SOURCE_INPUTS = {
    ("0000030", "0000000", "0000030", "2020202", "0000030", "0000000", "0000030"),
    ("000000", "000006", "404040", "000006", "000000", "000006"),
    ("000000", "900000", "050505", "900000", "000000", "900000"),
    ("00000000", "00020000", "00000000", "00020000", "00000000", "00020000", "80808080", "00020000"),
    ("000400", "000000", "000400", "909090", "000400", "000000"),
    ("0000000060", "0000000000", "6060606060", "0000000000", "0000000060", "0000000000", "0000000060", "0000000000", "0000000060", "0000000000"),
}

_AUTO_COUNTS = (
    (2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3),
    (5, 1), (5, 2), (5, 3), (5, 4), (6, 2), (6, 3),
    (6, 4), (7, 3), (7, 4), (8, 3), (8, 4), (9, 4), (10, 4),
)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return int(value)


def _counts(seed, rng, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if train is None:
        return int(rng.choice([n for n in range(2, 11) if n > test])), test
    if test is None:
        return train, int(rng.choice([n for n in range(1, 5) if n < train]))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _rows(grid):
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _position(rng, length):
    category = int(rng.choice(3, p=(0.24, 0.26, 0.50)))
    if category == 0:
        options = (0, length - 1)
    elif category == 1:
        options = (1, length - 2)
    else:
        options = tuple(range(2, length - 2))
    return int(rng.choice(options))


def _phase(rng, length, crossing, period):
    options = [
        phase for phase in range(period)
        if sum(index != crossing for index in range(phase, length, period)) >= 3
    ]
    assert options
    return int(rng.choice(options))


def _dimensions(rng, regime, long_spacing, small_to_large):
    if small_to_large:
        low, high = (6, 8) if regime == "source_train" else (9, 15)
    else:
        low, high = (6, 16) if regime == "source_train" else (6, 19)
        if regime == "target_test" and rng.random() < 0.16:
            high = 24
    if long_spacing:
        low = max(low, 10)
    height = int(rng.integers(low, high + 1))
    width = height if rng.random() < 0.52 else int(rng.integers(low, high + 1))
    return height, width


def _make_pair(rng, regime, mode, long_spacing, small_to_large, default_background):
    height, width = _dimensions(rng, regime, long_spacing, small_to_large)
    row = _position(rng, height)
    column = _position(rng, width)
    background = default_background
    if rng.random() < 0.34:
        background = int(rng.choice([color for color in range(10) if color != default_background]))

    colors = [color for color in range(1, 10) if color != background]
    horizontal_color = int(rng.choice(colors))
    vertical_color = horizontal_color if mode.startswith("equal") else int(
        rng.choice([color for color in colors if color != horizontal_color])
    )

    if long_spacing:
        periods = [(3, 2), (2, 3), (3, 3)]
        horizontal_period, vertical_period = periods[int(rng.integers(0, len(periods)))]
    else:
        horizontal_period = vertical_period = 2
    horizontal_phase = _phase(rng, width, column, horizontal_period)
    vertical_phase = _phase(rng, height, row, vertical_period)

    inp = np.full((height, width), background, dtype=np.int8)
    inp[row, np.arange(horizontal_phase, width, horizontal_period)] = horizontal_color
    inp[np.arange(vertical_phase, height, vertical_period), column] = vertical_color
    if mode == "distinct_empty" or mode == "equal_empty":
        inp[row, column] = background
    elif mode == "distinct_horizontal" or mode == "equal_filled":
        inp[row, column] = horizontal_color
    elif mode == "distinct_vertical":
        inp[row, column] = vertical_color
    else:
        raise ValueError("unknown crossing mode")

    out = np.full_like(inp, background)
    out[:, column] = vertical_color
    out[row, :] = horizontal_color
    return {"input": inp, "output": out}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)

    # Only the transformation and its horizontal intersection priority are
    # shared conventions. Colors, background, and geometry are local evidence.
    default_background = 0 if rng.random() < 0.63 else int(rng.integers(1, 10))
    long_enabled = bool(rng.random() < 0.42)
    small_to_large = bool(not long_enabled and rng.random() < 0.28)

    train_modes = ["distinct_empty", "distinct_vertical"]
    extra_modes = ("distinct_empty", "distinct_horizontal", "distinct_vertical", "equal_empty", "equal_filled")
    while len(train_modes) < train_count:
        train_modes.append(str(rng.choice(extra_modes, p=(0.25, 0.20, 0.17, 0.20, 0.18))))
    test_modes = [str(rng.choice(extra_modes, p=(0.23, 0.18, 0.19, 0.20, 0.20))) for _ in range(test_count)]
    pair_seeds = rng.integers(0, 2**63, size=train_count + test_count, dtype=np.int64)

    puzzle = {"train": [], "test": []}
    seen = set()
    for split, modes, offset in (("train", train_modes, 0), ("test", test_modes, train_count)):
        regime = "source_train" if split == "train" else "target_test"
        for index, mode in enumerate(modes):
            pair_rng = np.random.default_rng(int(pair_seeds[offset + index]))
            if split == "train":
                long_spacing = long_enabled and (index == 1 or (index > 1 and pair_rng.random() < 0.29))
            else:
                long_spacing = long_enabled and pair_rng.random() < 0.58
            for _ in range(40):
                pair = _make_pair(pair_rng, regime, mode, long_spacing, small_to_large, default_background)
                key = _rows(pair["input"])
                if key not in seen and key not in _SOURCE_INPUTS:
                    break
            else:
                raise RuntimeError("could not construct a distinct pair")
            seen.add(key)
            puzzle[split].append(pair)

    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect(inp, out):
    colors, counts = np.unique(inp, return_counts=True)
    background = int(colors[np.argmax(counts)])
    marks = inp != background
    row_counts = np.count_nonzero(marks, axis=1)
    column_counts = np.count_nonzero(marks, axis=0)
    candidate_rows = np.flatnonzero(row_counts >= 3)
    candidate_columns = np.flatnonzero(column_counts >= 3)
    assert len(candidate_rows) == len(candidate_columns) == 1
    row, column = int(candidate_rows[0]), int(candidate_columns[0])

    outside = marks.copy()
    outside[row, :] = False
    outside[:, column] = False
    assert not np.any(outside)
    horizontal_positions = [c for c in range(inp.shape[1]) if c != column and marks[row, c]]
    vertical_positions = [r for r in range(inp.shape[0]) if r != row and marks[r, column]]
    assert len(horizontal_positions) >= 3 and len(vertical_positions) >= 3
    horizontal_color = int(inp[row, horizontal_positions[0]])
    vertical_color = int(inp[vertical_positions[0], column])
    assert all(inp[row, c] == horizontal_color for c in horizontal_positions)
    assert all(inp[r, column] == vertical_color for r in vertical_positions)

    def period_for(positions, length, crossing):
        matches = [
            period for period in (2, 3)
            for phase in range(period)
            if positions == [i for i in range(phase, length, period) if i != crossing]
        ]
        assert len(matches) == 1
        return matches[0]

    horizontal_period = period_for(horizontal_positions, inp.shape[1], column)
    vertical_period = period_for(vertical_positions, inp.shape[0], row)
    crossing = int(inp[row, column])
    assert crossing in (background, horizontal_color, vertical_color)
    expected = np.full_like(inp, background)
    expected[:, column] = vertical_color
    expected[row, :] = horizontal_color
    assert np.array_equal(out, expected)
    assert not np.array_equal(inp, out)
    return (background, row, column, horizontal_color, vertical_color,
            horizontal_period, vertical_period, crossing)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    seen_inputs, seen_pairs = set(), set()
    observations = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape == out.shape
            input_key = _rows(inp)
            pair_key = (input_key, _rows(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert input_key not in _SOURCE_INPUTS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            observations[split].append(_inspect(inp, out))

    train = observations["train"]
    test = observations["test"]
    assert any(item[3] != item[4] and item[7] == item[0] for item in train)
    if any(item[3] != item[4] and item[7] == item[4] for item in test):
        assert any(item[3] != item[4] and item[7] == item[4] for item in train)
    if any(3 in item[5:7] for item in test):
        assert any(3 in item[5:7] for item in train)


def _same_puzzle(first, second):
    return all(
        len(first[split]) == len(second[split]) and all(
            np.array_equal(left[name], right[name])
            for left, right in zip(first[split], second[split])
            for name in ("input", "output")
        ) for split in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts, layouts, grids = set(), set(), set(), set()
    coverage = set()
    puzzle_color_diversity = 0
    differing_backgrounds = 0
    overlap = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert _same_puzzle(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        palettes, backgrounds, sizes = set(), set(), {"train": set(), "test": set()}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                feature = _inspect(inp, out)
                background, row, column, hc, vc, hp, vp, crossing = feature
                palettes.add((hc, vc))
                backgrounds.add(background)
                sizes[split].add(inp.shape)
                layouts.add((inp.shape, row, column, hp, vp, crossing == background))
                grids.add((inp.shape, inp.tobytes()))
                crossing_mode = "empty" if crossing == background else "horizontal" if crossing == hc else "vertical"
                coverage.add((split, hp, vp, hc == vc, crossing_mode))
                coverage.add((split, "rectangle", inp.shape[0] != inp.shape[1]))
                coverage.add((split, "border_row", row in (0, inp.shape[0] - 1)))
                coverage.add((split, "interior_column", 1 < column < inp.shape[1] - 2))
                # Reconstruct the target directly from visible off-crossing marks.
                predicted = np.full_like(inp, background)
                predicted[:, column] = inp[next(r for r in range(inp.shape[0]) if r != row and inp[r, column] != background), column]
                predicted[row, :] = inp[row, next(c for c in range(inp.shape[1]) if c != column and inp[row, c] != background)]
                assert np.array_equal(predicted, out)
        puzzle_color_diversity += len(palettes) >= 2
        differing_backgrounds += len(backgrounds) >= 2
        overlap += bool(sizes["train"] & sizes["test"])

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(layouts) >= 120 and len(grids) >= 180
    assert puzzle_color_diversity >= 35 and differing_backgrounds >= 12
    assert overlap >= 2
    for split in ("train", "test"):
        assert (split, "rectangle", True) in coverage
        assert (split, "border_row", True) in coverage
        assert (split, "interior_column", True) in coverage
        assert any(item[0] == split and item[1] == 3 for item in coverage if isinstance(item[1], int))
        assert any(item[0] == split and item[3] is True for item in coverage if isinstance(item[1], int))
        assert any(item[0] == split and item[4] == "empty" for item in coverage if isinstance(item[1], int))
    assert any(item[0] == "test" and item[4] == "vertical" for item in coverage if isinstance(item[1], int))

    for seed, train, test in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                              (104, None, 1), (105, None, 4),
                              (106, 2, None), (107, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert _same_puzzle(puzzle, generate_puzzle(seed, train, test))
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(200, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts were accepted")

    damaged = generate_puzzle(300, 2, 1)
    damaged["train"][0]["output"] = damaged["train"][0]["output"].copy()
    _, row, column, _, vertical_color, _, _, _ = _inspect(
        damaged["train"][0]["input"], damaged["train"][0]["output"]
    )
    damaged["train"][0]["output"][row, column] = vertical_color
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect intersection was accepted")
