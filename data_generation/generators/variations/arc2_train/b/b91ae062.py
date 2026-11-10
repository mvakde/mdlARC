import numpy as np


_AUTO_COUNTS = ((2, 1), (3, 1), (4, 1), (4, 2), (5, 2),
                (6, 3), (7, 1), (7, 2), (8, 3), (9, 4),
                (10, 1), (10, 4))
_SOURCE_ROWS = (("080", "066", "980"), ("104", "040", "010"),
                ("320", "073", "000"), ("403", "220", "008"),
                ("670", "066", "000"), ("010", "087", "990"))
_SOURCE_INPUTS = frozenset(
    ((len(rows), len(rows[0])), tuple(int(digit) for row in rows for digit in row))
    for rows in _SOURCE_ROWS
)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _key(grid):
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _scale(grid, background):
    return len(set(int(value) for value in np.unique(grid)) - {background})


def _expand(grid, background):
    factor = _scale(grid, background)
    return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)


def _count(name, value, minimum, maximum):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _counts(seed, rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _shape(rng, factor, minimum_area=0):
    maximum = min(12, 30 // factor)
    for _ in range(40):
        mode = int(rng.integers(5))
        if mode == 0:
            rows, columns = 3, 3
        elif mode == 1:
            rows = 1
            columns = int(rng.integers(max(5, factor + 1), maximum + 1))
        elif mode == 2:
            rows = int(rng.integers(2, min(5, maximum) + 1))
            columns = int(rng.integers(max(5, factor + 1), maximum + 1))
        else:
            rows = int(rng.integers(2, min(8, maximum) + 1))
            columns = int(rng.integers(2, min(8, maximum) + 1))
        if bool(rng.integers(2)):
            rows, columns = columns, rows
        if rows * columns >= max(factor + 1, minimum_area):
            return rows, columns
    return 6, 6


def _occupied_count(rng, factor, area):
    mode = int(rng.integers(6))
    if mode == 0:
        return factor
    if mode == 1:
        return area - 1
    if mode == 2:
        return min(area - 1, factor + int(rng.integers(1, 4)))
    if mode == 3:
        return int(rng.integers(factor, min(area - 1, max(factor, area // 3)) + 1))
    return int(rng.integers(factor, area))


def _positions(rng, shape, count):
    rows, columns = shape
    area = rows * columns
    mode = int(rng.integers(5))
    if mode == 0 and rows > 1 and count <= area - columns:
        empty_row = int(rng.integers(rows))
        eligible = [index for index in range(area) if index // columns != empty_row]
    elif mode == 1 and columns > 1 and count <= area - rows:
        empty_column = int(rng.integers(columns))
        eligible = [index for index in range(area) if index % columns != empty_column]
    else:
        eligible = list(range(area))
    if mode == 2 and count < len(eligible):
        start = int(rng.choice(eligible))
        chosen = {start}
        while len(chosen) < count:
            frontier = set()
            for index in chosen:
                row, column = divmod(index, columns)
                for next_row, next_column in ((row - 1, column), (row + 1, column),
                                              (row, column - 1), (row, column + 1)):
                    neighbor = next_row * columns + next_column
                    if (0 <= next_row < rows and 0 <= next_column < columns
                            and neighbor in eligible and neighbor not in chosen):
                        frontier.add(neighbor)
            chosen.add(int(rng.choice(sorted(frontier))))
        return list(chosen)
    return [int(value) for value in rng.choice(eligible, size=count, replace=False)]


def _fill(rng, shape, colors, background, count):
    factor = len(colors)
    area = shape[0] * shape[1]
    if not factor <= count < area:
        raise ValueError("invalid foreground count")
    positions = _positions(rng, shape, count)
    extras = count - factor
    mode = int(rng.integers(3))
    if mode == 0:
        extra_colors = [colors[int(rng.integers(factor))]] * extras
    elif mode == 1:
        order = [int(value) for value in rng.permutation(factor)]
        extra_colors = [colors[order[index % factor]] for index in range(extras)]
    else:
        extra_colors = [colors[int(rng.integers(factor))] for _ in range(extras)]
    values = list(colors) + extra_colors
    rng.shuffle(values)
    if extras and bool(rng.integers(2)):
        repeated = next(color for color in colors if values.count(color) > 1)
        metric = lambda first, second: (abs(first // shape[1] - second // shape[1])
                                        + abs(first % shape[1] - second % shape[1]))
        candidate_pairs = [(metric(first, second), first, second)
                           for first in positions for second in positions if first < second]
        chosen_pair = (max if bool(rng.integers(2)) else min)(candidate_pairs)
        values.remove(repeated)
        values.remove(repeated)
        selected = {chosen_pair[1], chosen_pair[2]}
        rng.shuffle(values)
        assignments = {index: repeated for index in selected}
        assignments.update(zip((index for index in positions if index not in selected), values))
    else:
        assignments = dict(zip(positions, values))
    grid = np.full(area, background, dtype=np.int8)
    for index, color in assignments.items():
        grid[index] = color
    return grid.reshape(shape)


def _new_input(rng, shape, colors, background, count, used):
    for _ in range(80):
        grid = _fill(rng, shape, colors, background, count)
        if _key(grid) not in used and _key(grid) not in _SOURCE_INPUTS:
            return grid
    for shift in range(shape[0] * shape[1]):
        grid = np.full(shape[0] * shape[1], background, dtype=np.int8)
        locations = [(shift + index) % grid.size for index in range(count)]
        values = list(colors) + [colors[0]] * (count - len(colors))
        for location, color in zip(locations, values):
            grid[location] = color
        grid = grid.reshape(shape)
        if _key(grid) not in used and _key(grid) not in _SOURCE_INPUTS:
            return grid
    raise RuntimeError("could not construct a unique input")


def _pair(grid, background):
    return {"input": grid, "output": _expand(grid, background).astype(np.int8)}


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
    background = int(rng.integers(10))
    available = [color for color in range(10) if color != background]
    rng.shuffle(available)
    unseen_color = available[-1]
    training_colors = available[:-1]
    first_colors = training_colors[:2]
    second_colors = training_colors[2:6]
    anchor_shape = _shape(rng, 4, minimum_area=6)
    anchor_count = int(rng.integers(4, min(anchor_shape[0] * anchor_shape[1] - 1, 8) + 1))
    train = []
    test = []
    used = set()

    def append_pair(split, pair_rng, shape, colors, count):
        grid = _new_input(pair_rng, shape, colors, background, count, used)
        used.add(_key(grid))
        split.append(_pair(grid, background))

    child_seeds = rng.integers(0, np.iinfo(np.int64).max,
                               size=train_count + test_count, dtype=np.int64)
    append_pair(train, np.random.default_rng(int(child_seeds[0])),
                anchor_shape, first_colors, anchor_count)
    append_pair(train, np.random.default_rng(int(child_seeds[1])),
                anchor_shape, second_colors, anchor_count)
    training_factors = [2, 4] + ([3] if train_count >= 3 else [])
    if train_count >= 4:
        training_factors.append(2)
    training_factors.extend(int(value) for value in rng.integers(
        2, 5, size=train_count - len(training_factors)))
    for index in range(2, train_count):
        pair_rng = np.random.default_rng(int(child_seeds[index]))
        factor = training_factors[index]
        colors = [int(value) for value in pair_rng.choice(training_colors, size=factor, replace=False)]
        shape = _shape(pair_rng, factor)
        count = _occupied_count(pair_rng, factor, shape[0] * shape[1])
        if index == 3 and count == anchor_count:
            count = factor if factor != anchor_count else factor + 1
        append_pair(train, pair_rng, shape, colors, count)
    rng.shuffle(train)

    factors = (2, 3, 4, 5) if train_count >= 3 else (2, 3, 4)
    source_like_test = int(rng.integers(5)) == 0
    test_factors = [4 if source_like_test else int(rng.choice(factors))]
    while len(test_factors) < test_count:
        test_factors.append(int(rng.choice([factor for factor in factors
                                            if factor != test_factors[-1]])))
    for index, factor in enumerate(test_factors):
        pair_rng = np.random.default_rng(int(child_seeds[train_count + index]))
        include_unseen = index == 0 or bool(pair_rng.integers(2))
        if include_unseen:
            colors = [unseen_color] + [int(value) for value in pair_rng.choice(
                training_colors, size=factor - 1, replace=False)]
        else:
            colors = [int(value) for value in pair_rng.choice(
                training_colors, size=factor, replace=False)]
        shape = (3, 3) if index == 0 and source_like_test else _shape(pair_rng, factor)
        count = 5 if index == 0 and source_like_test else _occupied_count(
            pair_rng, factor, shape[0] * shape[1])
        append_pair(test, pair_rng, shape, colors, count)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list), "splits must be lists")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "invalid pair counts")
    for pair in train + test:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 "pair must have exactly input and output")
        for grid in pair.values():
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                     "grids must be two-dimensional arrays")
            _require(all(1 <= extent <= 30 for extent in grid.shape), "grid size out of bounds")
            _require(np.issubdtype(grid.dtype, np.integer), "grid must have integer dtype")
            _require(np.all((grid >= 0) & (grid <= 9)), "illegal grid colors")
    train_color_sets = [set(int(color) for color in np.unique(pair["input"])) for pair in train]
    shared = set.intersection(*train_color_sets)
    _require(len(shared) == 1, "training must identify one shared background color")
    background = shared.pop()
    inputs = set()
    train_factors = []
    train_occupancies = []
    for split_name, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            grid, output = pair["input"], pair["output"]
            _require(bool(np.any(grid == background)), "background is missing")
            factor = _scale(grid, background)
            _require(2 <= factor <= (4 if split_name == "train" else 5),
                     "invalid train/test color count")
            _require(np.array_equal(output, _expand(grid, background)), "incorrect block enlargement")
            _require(grid.shape != output.shape, "input and output must differ in size")
            key = _key(grid)
            _require(key not in inputs, "duplicate input")
            _require(key not in _SOURCE_INPUTS, "source pair copied")
            inputs.add(key)
            if split_name == "train":
                train_factors.append(factor)
                train_occupancies.append(int(np.count_nonzero(grid != background)))
    _require(2 in train_factors and 4 in train_factors, "training needs contrasting scales")
    if len(train) >= 3:
        _require(3 in train_factors, "training must show factor three")
    contrast = any(
        train[first]["input"].shape == train[second]["input"].shape
        and train_occupancies[first] == train_occupancies[second]
        and train_factors[first] != train_factors[second]
        for first in range(len(train)) for second in range(first + 1, len(train))
    )
    _require(contrast, "training needs a same-size, same-occupancy color-count contrast")
    if len(train) >= 4:
        _require(any(train_factors[first] == train_factors[second]
                     and train_occupancies[first] != train_occupancies[second]
                     for first in range(len(train)) for second in range(first + 1, len(train))),
                 "training needs a fixed-factor occupancy contrast")
    test_factors = [_scale(pair["input"], background) for pair in test]
    _require(5 not in test_factors or 3 in train_factors,
             "factor-five tests need three demonstrated training scales")
    if len(test) > 1:
        _require(len(set(test_factors)) > 1, "multiple tests should use different factors")
    seen_train_colors = set.union(*train_color_sets)
    _require(any(set(int(color) for color in np.unique(pair["input"])) - seen_train_colors
                 for pair in test), "tests should exercise a new literal color")


def self_test() -> None:
    def signature(puzzle):
        return tuple(tuple((_key(pair["input"]), _key(pair["output"]))
                           for pair in puzzle[split]) for split in ("train", "test"))

    for seed, train_count, test_count in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                                          (104, None, 1), (105, None, 4),
                                          (106, 2, None), (107, 10, None),
                                          (314159, 7, 3)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert signature(puzzle) == signature(generate_puzzle(seed, train_count, test_count))
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for options in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 3, "num_test": 4}, {"num_train": True},
                    {"num_test": False}):
        try:
            generate_puzzle(200, **options)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal request accepted: {options}")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({signature(puzzle) for puzzle in puzzles}) == 42
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    backgrounds, factors, shapes, occupancies, frequencies = set(), set(), set(), set(), set()
    test_factors = set()
    singleton = dense = empty_line = thin = large = interior = 0
    multiple_repeats = adjacent = separated = novel_tests = source_like_tests = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        background = set.intersection(*(set(int(color) for color in np.unique(pair["input"]))
                                        for pair in puzzle["train"])).pop()
        backgrounds.add(background)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                palette = set(int(value) for value in np.unique(grid)) - {background}
                factor = len(palette)
                expected = np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)
                assert np.array_equal(output, expected)
                assert all(1 <= size <= 30 for size in grid.shape + output.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.issubdtype(output.dtype, np.integer)
                assert _key(grid) not in _SOURCE_INPUTS
                factors.add(factor)
                shapes.add(grid.shape)
                occupancy = int(np.count_nonzero(grid != background))
                occupancies.add(occupancy)
                frequencies.add(tuple(sorted(int(np.count_nonzero(grid == color)) for color in palette)))
                singleton += occupancy == factor
                dense += occupancy == grid.size - 1
                empty_line += bool(np.any(np.all(grid == background, axis=0))
                                   or np.any(np.all(grid == background, axis=1)))
                thin += 1 in grid.shape
                large += max(grid.shape) > 4
                interior += (grid.shape[0] > 2 and grid.shape[1] > 2
                             and bool(np.any(grid[1:-1, 1:-1] != background)))
                multiple_repeats += sum(np.count_nonzero(grid == color) > 1
                                        for color in palette) > 1
                for color in palette:
                    positions = np.argwhere(grid == color)
                    if len(positions) > 1:
                        distances = [int(np.abs(first - second).sum())
                                     for first in positions for second in positions
                                     if not np.array_equal(first, second)]
                        adjacent += min(distances) == 1
                        separated += max(distances) >= 3
                if split == "test":
                    test_factors.add(factor)
                    training_palette = set().union(*(
                        set(int(value) for value in np.unique(item["input"]))
                        for item in puzzle["train"]))
                    novel_tests += bool(palette - training_palette)
                    source_like_tests += grid.shape == (3, 3) and factor == 4 and occupancy == 5
    assert len(backgrounds) >= 6
    assert factors == {2, 3, 4, 5} and test_factors == {2, 3, 4, 5}
    assert len(shapes) >= 25 and len(occupancies) >= 15 and len(frequencies) >= 12
    assert singleton >= 15 and dense >= 10 and empty_line >= 20
    assert thin >= 10 and large >= 20 and interior >= 20
    assert multiple_repeats >= 10 and adjacent >= 20 and separated >= 20
    assert novel_tests >= 20 and source_like_tests >= 3
    damaged = generate_puzzle(9001, 5, 2)
    damaged["train"][0]["output"][0, 0] = (int(damaged["train"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a corrupted output")
