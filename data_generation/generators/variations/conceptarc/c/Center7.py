import numpy as np


_SOURCE_INPUT_ROWS = {
    ("10000", "11000", "11100", "11110", "11111"),
    ("22222", "22220", "22200", "22000", "20000"),
    ("444444444", "044444444", "004444444", "000444444",
     "000044444", "000004444", "000000444", "000000044", "000000004"),
    ("600", "660", "666"),
    ("55555", "00000", "55550", "00000", "55500", "00000",
     "55000", "00000", "50000"),
    ("5555555555", "5555500000", "5000000000"),
}


def _key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _rows(grid):
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _check_count(value, name, low, high):
    if value is None:
        return
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= value <= high):
        raise ValueError(f"{name} must be an integer from {low} to {high}")


def _stripes(grid, color):
    positions, lengths, starts = [], [], []
    for row_number, row in enumerate(grid):
        cells = np.flatnonzero(row == color)
        if len(cells):
            if int(cells[-1] - cells[0] + 1) != len(cells):
                return None
            positions.append(row_number)
            lengths.append(len(cells))
            starts.append(int(cells[0]))
    if len(positions) < 3 or len(positions) % 2 != 1:
        return None
    if not (all(a < b for a, b in zip(lengths, lengths[1:]))
            or all(a > b for a, b in zip(lengths, lengths[1:]))):
        return None
    return positions, lengths, starts


def _parse_input(grid):
    colors = [int(x) for x in np.unique(grid)]
    assert len(colors) == 2, "input must have two colors"
    candidates = []
    for color in colors:
        stripes = _stripes(grid, color)
        if stripes is not None:
            candidates.append((color, *stripes))
    assert len(candidates) == 1, "foreground must be identifiable from the input"
    foreground, positions, lengths, starts = candidates[0]
    background = next(color for color in colors if color != foreground)
    return foreground, background, positions, lengths, starts


def _middle_prediction(grid):
    foreground, _, _, lengths, _ = _parse_input(grid)
    return np.full((1, lengths[len(lengths) // 2]), foreground, dtype=np.int8)


def _assemble(lengths, width, blanks, foreground, background, alignment, rng):
    # blanks gives leading, internal, and trailing blank-row counts.
    count = len(lengths)
    grid = np.full((count + sum(blanks), width), background, dtype=np.int8)
    position = blanks[0]
    shift = int(rng.integers(-1, 2))
    for index, length in enumerate(lengths):
        room = width - length
        if alignment == "left":
            start = 0
        elif alignment == "right":
            start = room
        elif alignment == "center":
            start = room // 2
        elif alignment == "inset":
            start = min(room, max(0, int(rng.integers(1, 4))))
        else:
            start = min(room, max(0, room // 2 + shift + int(rng.integers(-2, 3))))
        grid[position, start:start + length] = foreground
        position += 1 + blanks[index + 1]
    output = np.full((1, lengths[count // 2]), foreground, dtype=np.int8)
    return {"input": grid, "output": output}


def _random_lengths(rng, width, count, half=False):
    if half:
        middle = (width + 1) // 2
        if middle < count // 2 + 1 or width - middle < count // 2:
            return None
        lower = rng.choice(np.arange(1, middle), size=count // 2, replace=False)
        upper = rng.choice(np.arange(middle + 1, width + 1),
                           size=count // 2, replace=False)
        return sorted([middle, *(int(x) for x in lower), *(int(x) for x in upper)])
    endpoint = int(rng.integers(4))
    low = 1 if endpoint in (0, 1) else 2
    high = width if endpoint in (0, 2) else width - 1
    if high - low + 1 < count:
        low, high = 1, width
    return sorted(int(x) for x in rng.choice(
        np.arange(low, high + 1), size=count, replace=False))


def _blank_pattern(rng, count, shape, local_colors, witness):
    if witness:
        blanks = [int(rng.integers(0, 3))]
        blanks.extend(int(x) for x in rng.integers(0, 4, size=count - 1))
        blanks.append(int(rng.integers(0, 3)))
        blanks[1] = max(2, blanks[1])
        blanks[count - 1] = 0
    elif shape == "symmetric":
        gap = int(rng.integers(1, 4))
        edge = int(rng.integers(0, 3))
        blanks = [edge] + [gap] * (count - 1) + [edge]
    else:
        cap = 5 if shape == "tall" else 4
        blanks = [int(rng.integers(0, 4))]
        blanks.extend(int(x) for x in rng.integers(0, cap, size=count - 1))
        blanks.append(int(rng.integers(0, 4)))
    if local_colors and sum(blanks) == 0:
        blanks[int(rng.choice([0, count]))] = 1
    return blanks


def _draw_pair(rng, foreground, background, style, local_colors=False,
               witness=False, classic_test=False):
    if style == "simple":
        count = int(rng.choice([3, 5, 7, 9, 11, 13, 15, 17]))
        width = count
        lengths = list(range(1, count + 1))
        blanks = [0] * (count + 1)
    else:
        if style == "wide":
            count = int(rng.choice([3, 5, 7]))
            width = int(rng.integers(max(count + 4, 10), 31))
        elif style == "tall":
            count = int(rng.choice([3, 5, 7]))
            width = int(rng.integers(count, min(count + 7, 16)))
        elif style == "compact":
            count = int(rng.choice([3, 5, 7, 9]))
            width = int(rng.integers(count, min(count + 7, 19)))
        else:
            count = int(rng.choice([3, 5, 7, 9]))
            width = int(rng.integers(count + 2, 27))
        lengths = _random_lengths(rng, width, count, half=classic_test)
        if lengths is None:
            return None
        blanks = _blank_pattern(rng, count, style, local_colors, witness)
        if classic_test and style == "wide":
            blanks = [0] * (count + 1)
    if count + sum(blanks) > 30:
        return None
    if witness:
        middle = lengths[count // 2]
        positions = np.cumsum([blanks[0], *(1 + x for x in blanks[1:-1])])
        if (middle == (width + 1) // 2
                or 2 * positions[count // 2] == count + sum(blanks) - 1):
            return None
    if style != "simple" and width not in lengths and sum(blanks) == 0:
        blanks[0] = 1
        if classic_test:
            blanks[-1] = 1
    if int(rng.integers(2)):
        lengths.reverse()
    alignment = str(rng.choice(["left", "right", "center", "inset", "shift"]))
    if style == "simple":
        alignment = str(rng.choice(["left", "right", "center"]))
    pair = _assemble(lengths, width, blanks, foreground, background,
                     alignment, rng)
    try:
        parsed = _parse_input(pair["input"])
    except AssertionError:
        return None
    return pair if parsed[0] == foreground else None


def _distinct_pair(rng, foreground, background, styles, seen,
                   local_colors=False, witness=False, classic_test=False):
    for _ in range(500):
        style = str(rng.choice(styles))
        pair = _draw_pair(rng, foreground, background, style,
                          local_colors, witness, classic_test)
        if pair is None:
            continue
        key = _key(pair["input"])
        if key not in seen and _rows(pair["input"]) not in _SOURCE_INPUT_ROWS:
            seen.add(key)
            return pair
    raise RuntimeError("could not construct a distinct, unambiguous pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    _check_count(num_train, "num_train", 2, 10)
    _check_count(num_test, "num_test", 1, 4)
    rng = np.random.default_rng(seed)
    if num_test is None:
        limit = min(4, num_train - 1) if num_train is not None else 4
        num_test = int(rng.integers(1, limit + 1))
    if num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")

    classic = bool(rng.integers(0, 5) == 0)
    local_colors = not classic and bool(rng.integers(0, 2))
    base_background, base_foreground = (int(x) for x in
                                        rng.choice(10, size=2, replace=False))
    shared_foregrounds = [base_foreground, int(rng.choice(
        [x for x in range(10) if x not in (base_background, base_foreground)]))]
    puzzle = {"train": [], "test": []}
    seen = set()

    def colors(split, index):
        if local_colors:
            if split == "train" and index == 0:
                return base_foreground, base_background
            if split == "train" and index == 1:
                return base_background, base_foreground
            background = int(rng.integers(0, 10))
            foreground = int(rng.choice([x for x in range(10) if x != background]))
            return foreground, background
        if split == "train" and index < 2:
            return shared_foregrounds[index], base_background
        foreground = int(rng.choice([x for x in range(10) if x != base_background]))
        return foreground, base_background

    for index in range(num_train):
        foreground, background = colors("train", index)
        if classic:
            styles = ["simple"]
        elif index == 0:
            styles = ["compact", "wide", "tall", "free"]
        else:
            styles = ["simple", "simple", "compact", "wide", "tall", "free"]
        pair = _distinct_pair(rng, foreground, background, styles, seen,
                              local_colors, witness=not classic and index == 0)
        puzzle["train"].append(pair)

    for index in range(num_test):
        foreground, background = colors("test", index)
        if classic:
            styles = ["wide", "symmetric"]
        elif index == 0:
            styles = ["wide", "tall", "free", "symmetric"]
        else:
            styles = ["wide", "tall", "compact", "free", "symmetric", "simple"]
        pair = _distinct_pair(rng, foreground, background, styles, seen,
                              local_colors, classic_test=classic)
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _off_half(pair):
    _, _, _, lengths, _ = _parse_input(pair["input"])
    return lengths[len(lengths) // 2] != (pair["input"].shape[1] + 1) // 2


def _off_center(pair):
    _, _, positions, _, _ = _parse_input(pair["input"])
    return 2 * positions[len(positions) // 2] != pair["input"].shape[0] - 1


def _is_source_style(pair):
    grid = pair["input"]
    _, _, positions, lengths, _ = _parse_input(grid)
    return (grid.shape[0] == grid.shape[1] == len(positions)
            and sorted(lengths) == list(range(1, len(lengths) + 1)))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    train_backgrounds, train_foregrounds = set(), set()
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            key = _key(pair["input"])
            assert key not in seen
            seen.add(key)
            assert _rows(pair["input"]) not in _SOURCE_INPUT_ROWS
            foreground, background, _, _, _ = _parse_input(pair["input"])
            assert np.array_equal(pair["output"], _middle_prediction(pair["input"]))
            assert not np.array_equal(pair["input"], pair["output"])
            if split == "train":
                train_backgrounds.add(background)
                train_foregrounds.add(foreground)
    test_backgrounds = {_parse_input(p["input"])[1] for p in test}
    assert len(train_foregrounds) >= 2
    if not test_backgrounds <= train_backgrounds:
        assert len(train_backgrounds) >= 2
        assert bool(train_backgrounds & train_foregrounds)
    if any(_off_half(pair) for pair in test):
        assert any(_off_half(pair) for pair in train)
    if any(_off_center(pair) for pair in test):
        assert any(_off_center(pair) for pair in train)
    assert any(not _is_source_style(pair) for pair in test)


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[name], b[name])
        for a, b in zip(first[split], second[split])
        for name in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts, shapes, length_patterns = set(), set(), set(), set()
    backgrounds, foregrounds, starts, blank_patterns = set(), set(), set(), set()
    fingerprints = set()
    off_half_train = off_center_train = partial_train = local_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(tuple(_key(p["input"]) for split in ("train", "test")
                               for p in puzzle[split]))
        train_bgs = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                fg, bg, pos, lengths, start = _parse_input(grid)
                assert np.array_equal(_middle_prediction(grid), pair["output"])
                assert _rows(grid) not in _SOURCE_INPUT_ROWS
                shapes.add(grid.shape)
                length_patterns.add(tuple(lengths))
                backgrounds.add(bg)
                foregrounds.add(fg)
                starts.add(tuple(start))
                blank_patterns.add(tuple(b - a - 1 for a, b in zip(pos, pos[1:])))
                if split == "train":
                    train_bgs.add(bg)
                    off_half_train += _off_half(pair)
                    off_center_train += _off_center(pair)
                    partial_train += min(lengths) > 1 or max(lengths) < grid.shape[1]
        local_puzzles += len(train_bgs) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42
    assert len(shapes) >= 30 and len(length_patterns) >= 70
    assert len(backgrounds) >= 6 and len(foregrounds) >= 6
    assert len(starts) >= 30 and len(blank_patterns) >= 12
    assert off_half_train >= 20 and off_center_train >= 10
    assert partial_train >= 10 and local_puzzles >= 5
    for train, test in ((2, 1), (10, 1), (10, 4), (None, 1),
                        (None, 4), (2, None), (10, None)):
        seed = 917 + (train or 0) * 10 + (test or 0)
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert _puzzles_equal(puzzle, generate_puzzle(seed, train, test))
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4), (2, 4), (True, 1), (2.5, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts must raise ValueError")
