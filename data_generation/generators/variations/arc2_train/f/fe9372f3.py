"""Solved cross-and-rays ARC puzzles with learned axial cycles."""

import numpy as np


_DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_AXES = ((-1, 0), (1, 0), (0, -1), (0, 1))
_SOURCE_LAYOUTS = ((7, 16, 3, 3), (12, 10, 4, 4), (30, 17, 11, 7))


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _make_input(height, width, row, column, background, seed):
    grid = np.full((height, width), background, dtype=np.int8)
    for dr, dc in ((0, 0),) + _AXES:
        grid[row + dr, column + dc] = seed
    return grid


def _locate_cross(grid):
    height, width = grid.shape
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 2
    candidates = []
    for index in np.flatnonzero(counts == 5):
        seed = int(values[index])
        background = int(values[1 - index])
        cells = {(int(r), int(c)) for r, c in zip(*np.where(grid == seed))}
        for row, column in cells:
            if not (1 <= row < height - 1 and 1 <= column < width - 1):
                continue
            expected = {(row, column)} | {(row + dr, column + dc) for dr, dc in _AXES}
            if cells == expected:
                candidates.append((background, seed, row, column))
    assert len(candidates) == 1
    return candidates[0]


def _axial_runs(grid):
    _, _, row, column = _locate_cross(grid)
    height, width = grid.shape
    runs = []
    for dr, dc in _AXES:
        run = []
        step = 2
        while 0 <= row + step * dr < height and 0 <= column + step * dc < width:
            run.append((row + step * dr, column + step * dc))
            step += 1
        runs.append(run)
    return runs


def _render(input_grid, diagonal, word):
    background, seed, row, column = _locate_cross(input_grid)
    height, width = input_grid.shape
    output = input_grid.copy()
    for dr, dc in _DIAGONALS:
        step = 1
        while 0 <= row + step * dr < height and 0 <= column + step * dc < width:
            output[row + step * dr, column + step * dc] = diagonal
            step += 1
    for run in _axial_runs(input_grid):
        for index, (r, c) in enumerate(run):
            output[r, c] = word[index % len(word)]
    return output


def _source_keys():
    result = set()
    for height, width, row, column in _SOURCE_LAYOUTS:
        source_input = _make_input(height, width, row, column, 0, 2)
        source_output = _render(source_input, 1, (8, 8, 4))
        result.add((_key(source_input), _key(source_output)))
    return result


def _anchor_geometry(rng, mode):
    long_side = int(rng.integers(18, 31))
    short_side = int(rng.integers(7, 18))
    if mode == 0:
        height, width = short_side, long_side
    elif mode == 1:
        height, width = long_side, short_side
    else:
        height = long_side
        width = max(18, min(30, long_side + int(rng.integers(-2, 3))))
    return height, width, *_choose_center(rng, height, width)


def _short_geometry(rng):
    narrow = int(rng.integers(3, 5))
    other = int(rng.integers(5, 10))
    near = 1 if narrow == 3 else int(rng.choice((1, 2)))
    other_center = 2 if int(rng.integers(0, 2)) == 0 else other - 3
    if int(rng.integers(0, 2)) == 0:
        return narrow, other, near, other_center
    return other, narrow, other_center, near


def _choose_center(rng, height, width):
    if height % 2 and width % 2 and rng.random() < 0.18:
        return height // 2, width // 2
    if rng.random() < 0.32:
        row = int(rng.choice((1, height - 2)))
        column = int(rng.integers(1, width - 1))
        if rng.random() < 0.5:
            row = int(rng.integers(1, height - 1))
            column = int(rng.choice((1, width - 2)))
        return row, column
    return int(rng.integers(1, height - 1)), int(rng.integers(1, width - 1))


def _varied_geometry(rng, mode=None, force_smallest=False):
    if mode is None:
        mode = int(rng.integers(0, 5))
    if mode == 0:
        if force_smallest or rng.random() < 0.3:
            height = width = 3
        else:
            height = int(rng.integers(3, 9))
            width = int(rng.integers(3, 9))
    elif mode == 1:
        height = int(rng.integers(5, 23))
        width = int(rng.integers(26, 31))
    elif mode == 2:
        height = int(rng.integers(26, 31))
        width = int(rng.integers(5, 23))
    elif mode == 3:
        height = int(rng.integers(9, 31))
        width = max(3, min(30, height + int(rng.integers(-1, 2))))
    else:
        height = int(rng.integers(7, 25))
        width = int(rng.integers(7, 25))
    return height, width, *_choose_center(rng, height, width)


def _infer_conventions(train):
    diagonal = set()
    observed = {}
    longest = 0
    for pair in train:
        input_grid, output_grid = pair["input"], pair["output"]
        _, _, row, column = _locate_cross(input_grid)
        height, width = input_grid.shape
        for dr, dc in _DIAGONALS:
            step = 1
            while 0 <= row + step * dr < height and 0 <= column + step * dc < width:
                diagonal.add(int(output_grid[row + step * dr, column + step * dc]))
                step += 1
        for run in _axial_runs(input_grid):
            longest = max(longest, len(run))
            for index, (r, c) in enumerate(run):
                observed.setdefault(index, set()).add(int(output_grid[r, c]))
    assert len(diagonal) == 1
    assert all(len(colors) == 1 for colors in observed.values())
    assert all(index in observed for index in range(longest))
    sequence = tuple(next(iter(observed[index])) for index in range(longest))
    assert len(set(sequence)) == 2
    for period in (2, 3, 4):
        if longest < 2 * period:
            continue
        if all(sequence[index] == sequence[index % period] for index in range(longest)):
            word = sequence[:period]
            # One accent per period, allowing cyclic launch phases.
            if period >= 3 and sorted(word.count(color) for color in set(word)) != [1, period - 1]:
                continue
            return next(iter(diagonal)), word
    raise AssertionError("training does not identify a supported axial cycle")


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert 0 <= int(grid.min()) <= int(grid.max()) <= 9


def _assert_coordinate_rule(input_grid, output_grid, diagonal, word):
    background, seed, center_row, center_column = _locate_cross(input_grid)
    for row in range(input_grid.shape[0]):
        for column in range(input_grid.shape[1]):
            dr, dc = row - center_row, column - center_column
            ar, ac = abs(dr), abs(dc)
            if ar + ac <= 1:
                expected = seed
            elif ar == ac:
                expected = diagonal
            elif dr == 0 or dc == 0:
                expected = word[(ar + ac - 2) % len(word)]
            else:
                expected = background
            assert int(output_grid[row, column]) == expected


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    seen_layouts = set()
    source_pairs = _source_keys()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            _validate_grid(input_grid)
            _validate_grid(output_grid)
            assert input_grid.shape == output_grid.shape
            _, _, center_row, center_column = _locate_cross(input_grid)
            layout = (*input_grid.shape, center_row, center_column)
            assert layout not in seen_layouts
            seen_layouts.add(layout)
            input_key = _key(input_grid)
            pair_key = input_key, _key(output_grid)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in source_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

    diagonal, word = _infer_conventions(puzzle["train"])
    output_colors = {diagonal, *word}
    assert len(output_colors) == 3
    train_palettes = set()
    train_backgrounds = set()
    train_seeds = set()
    short_training = False
    long_training = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            input_grid, output_grid = pair["input"], pair["output"]
            background, seed, _, _ = _locate_cross(input_grid)
            assert len({background, seed} | output_colors) == 5
            _assert_coordinate_rule(input_grid, output_grid, diagonal, word)
            assert not np.array_equal(input_grid, output_grid)
            runs = _axial_runs(input_grid)
            if split == "train":
                train_palettes.add((background, seed))
                train_backgrounds.add(background)
                train_seeds.add(seed)
                long_training |= max(map(len, runs)) >= 2 * len(word)
                lengths = [len(run) for run in runs]
                short_training |= 0 in lengths and any(length in (1, 2) for length in lengths)
    assert len(train_palettes) >= 2
    assert len(train_backgrounds) >= 2 and len(train_seeds) >= 2
    assert long_training and short_training


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)

    # Output-only roles and the axial word are learned once per puzzle.
    colors = [int(color) for color in rng.permutation(10)]
    diagonal, ordinary, accent = colors[:3]
    available = colors[3:]
    period = int(rng.choice((2, 3, 4), p=(0.25, 0.5, 0.25)))
    base_word = (ordinary,) * (period - 1) + (accent,)
    if period == 3:
        phase = int(rng.choice((0, 1, 2), p=(0.5, 0.25, 0.25)))
    else:
        phase = int(rng.integers(0, period))
    word = base_word[phase:] + base_word[:phase]

    # The first two training pairs each receive different visible colors.
    visible = [tuple(available[:2]), tuple(available[2:4])]
    for _ in range(train_count + test_count - 2):
        chosen = rng.choice(available, size=2, replace=False)
        visible.append((int(chosen[0]), int(chosen[1])))

    geometries = [_anchor_geometry(rng, seed % 3), _short_geometry(rng)]
    geometries += [_varied_geometry(rng) for _ in range(train_count - 2)]
    train_pairs = []
    test_pairs = []
    used_inputs = set()
    used_layouts = set()
    source_pairs = _source_keys()
    for index in range(train_count + test_count):
        for attempt in range(80):
            if index < train_count and attempt == 0:
                height, width, row, column = geometries[index]
            elif index == train_count and attempt == 0:
                mode = seed % 5
                height, width, row, column = _varied_geometry(
                    rng, mode, force_smallest=mode == 0 and seed % 2 == 0
                )
            else:
                height, width, row, column = _varied_geometry(rng)
            background, seed_color = visible[index]
            input_grid = _make_input(height, width, row, column, background, seed_color)
            output_grid = _render(input_grid, diagonal, word)
            pair_key = _key(input_grid), _key(output_grid)
            layout = (height, width, row, column)
            if (
                _key(input_grid) not in used_inputs
                and layout not in used_layouts
                and pair_key not in source_pairs
            ):
                used_inputs.add(_key(input_grid))
                used_layouts.add(layout)
                pair = {"input": input_grid, "output": output_grid}
                (train_pairs if index < train_count else test_pairs).append(pair)
                break
        else:
            raise RuntimeError("could not construct a unique pair")
    rng.shuffle(train_pairs)
    puzzle = {"train": train_pairs, "test": test_pairs}
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    return tuple((split, tuple((_key(pair["input"]), _key(pair["output"])) for pair in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert _fingerprint(generate_puzzle(123)) == _fingerprint(generate_puzzle(123))
    assert len({_fingerprint(p) for p in puzzles}) == 42

    words = set()
    layouts = set()
    sizes = set()
    backgrounds = set()
    train_lengths = set()
    test_aspects = set()
    centered = 0
    small_tests = 0
    wide_large_tests = 0
    tall_train_wide_test = 0
    shorter_tests = 0
    phase_positions = {3: set(), 4: set()}
    partial_train = 0
    partial_test = 0
    missing_test = 0
    for puzzle in puzzles:
        _, word = _infer_conventions(puzzle["train"])
        words.add(word)
        if len(word) >= 3:
            accent = min(set(word), key=word.count)
            phase_positions[len(word)].add(word.index(accent))
        train_shapes = [pair["input"].shape for pair in puzzle["train"]]
        if any(h > w + 1 for h, w in train_shapes) and any(
            pair["input"].shape[1] > pair["input"].shape[0] + 1
            for pair in puzzle["test"]
        ):
            tall_train_wide_test += 1
        if any(
            pair["input"].shape[0] < min(h for h, _ in train_shapes)
            or pair["input"].shape[1] < min(w for _, w in train_shapes)
            for pair in puzzle["test"]
        ):
            shorter_tests += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                background, _, row, column = _locate_cross(grid)
                height, width = grid.shape
                layouts.add((height, width, row, column))
                sizes.add((height, width))
                backgrounds.add(background)
                centered += 2 * row == height - 1 and 2 * column == width - 1
                if split == "train":
                    train_lengths.update(map(len, _axial_runs(grid)))
                else:
                    test_aspects.add("wide" if width > height + 1 else "tall" if height > width + 1 else "square")
                    small_tests += max(height, width) <= 5
                    wide_large_tests += width >= 26 and width > height
                lengths = list(map(len, _axial_runs(grid)))
                if any(length and length % len(word) for length in lengths):
                    if split == "train":
                        partial_train += 1
                    else:
                        partial_test += 1
                if split == "test" and 0 in lengths:
                    missing_test += 1
    assert {2, 3, 4} <= {len(word) for word in words}
    assert len(words) >= 6
    assert len(layouts) >= 160 and len(sizes) >= 100
    assert len(backgrounds) >= 8
    assert {0, 1, 2} <= train_lengths
    assert test_aspects == {"wide", "tall", "square"}
    assert centered > 0 and small_tests > 0 and wide_large_tests > 0
    assert len(phase_positions[3]) >= 3 and len(phase_positions[4]) >= 3
    assert tall_train_wide_test > 0 and shorter_tests > 0
    assert partial_train > 0 and partial_test > 0 and missing_test > 0

    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4), (1004, 7, 3)):
        puzzle = generate_puzzle(seed, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
        validate_generated_puzzle(puzzle)
    for seed, train, test in ((1101, None, 1), (1102, None, 4), (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for train, test in ((1, 1), (11, 1), (2, 0), (2, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(17, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid pair counts were accepted")
