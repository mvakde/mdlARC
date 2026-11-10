import numpy as np


_SOURCE_INPUT_ROWS = (
    ((3, 0, 3), (3, 0, 3), (0, 3, 3)),
    ((0, 4, 0), (0, 4, 4), (4, 0, 0)),
    ((2, 0, 2), (0, 2, 0), (0, 0, 0)),
    ((6, 6, 0), (0, 6, 6), (0, 0, 6)),
    ((0, 0, 8), (0, 8, 0), (0, 0, 0)),
)
_SOURCE_INPUT_KEYS = frozenset(
    (3, 3) + tuple(value for row in rows for value in row)
    for rows in _SOURCE_INPUT_ROWS
)
_AUTO_COUNT_PAIRS = (
    (2, 1), (3, 1), (4, 2), (5, 3), (6, 4),
    (7, 1), (8, 2), (9, 3), (10, 4),
)


def _grid_key(grid: np.ndarray) -> tuple:
    return tuple(int(x) for x in grid.shape) + tuple(int(x) for x in grid.flat)


def _normalise_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    value = int(seed)
    if value < 0:
        raise ValueError("seed must be a non-negative integer")
    return value


def _normalise_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train_count = _normalise_count(num_train, "num_train")
    test_count = _normalise_count(num_test, "num_test")
    if train_count is not None and not 2 <= train_count <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test_count is not None and not 1 <= test_count <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train_count is None and test_count is None:
        train_count, test_count = _AUTO_COUNT_PAIRS[seed % len(_AUTO_COUNT_PAIRS)]
    elif train_count is None:
        minimum = max(2, test_count + 1)
        train_count = minimum + seed % (11 - minimum)
    elif test_count is None:
        test_count = 1 + seed % min(4, train_count - 1)
    if train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    return train_count, test_count


def _mask(grid: np.ndarray, background: int) -> int:
    return sum(1 << index for index, value in enumerate(grid.flat) if int(value) != background)


def _construct_output(input_grid: np.ndarray, background: int) -> np.ndarray:
    foreground_count = int(np.count_nonzero(input_grid != background))
    background_count = 9 - foreground_count
    if not 1 <= foreground_count <= 6 or foreground_count > background_count**2:
        raise ValueError("invalid two-color seed tile")
    output = np.full((3 * background_count, 3 * background_count), background, dtype=np.int8)
    for slot in range(foreground_count):
        slot_row, slot_col = divmod(slot, background_count)
        output[3 * slot_row:3 * slot_row + 3, 3 * slot_col:3 * slot_col + 3] = input_grid
    return output


def _sample_input(
    rng: np.random.Generator,
    foreground_count: int,
    background: int,
    foreground: int,
    seen_inputs: set[tuple],
    preferred_mask: int | None = None,
) -> np.ndarray:
    candidates = []
    for mask in range(1, 1 << 9):
        if mask.bit_count() != foreground_count:
            continue
        flat = tuple(foreground if mask & (1 << index) else background for index in range(9))
        key = (3, 3) + flat
        if key not in seen_inputs and key not in _SOURCE_INPUT_KEYS:
            candidates.append((mask, flat, key))
    if not candidates:
        raise ValueError("no unused tile remains")
    chosen = next((item for item in candidates if item[0] == preferred_mask), None)
    if chosen is None:
        chosen = candidates[int(rng.integers(len(candidates)))]
    seen_inputs.add(chosen[2])
    return np.asarray(chosen[1], dtype=np.int8).reshape(3, 3)


def _train_counts(rng: np.random.Generator, count: int) -> list[int]:
    wrap = int(rng.choice(np.array([5, 6])))
    middle = 4 if int(rng.integers(2)) else 2
    result = [3, wrap, middle, 11 - wrap]
    result.append(result[int(rng.integers(2))])
    result.extend([6 - middle, 1])
    while len(result) < count:
        result.append(int(rng.integers(1, 7)))
    return result[:count]


def _test_counts(seed: int, rng: np.random.Generator, count: int) -> list[int]:
    # The first count traverses all six legal cases across consecutive seeds.
    first = 2 if seed % 5 == 0 else 1 + seed % 6
    remainder = [value for value in range(1, 7) if value != first]
    rng.shuffle(remainder)
    return [first] + remainder[:count - 1]


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed_value = _normalise_seed(seed)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    rng = np.random.default_rng(seed_value)

    # The background is a puzzle-wide convention, visible in every training output.
    background = 0 if seed_value % 5 == 0 else int(rng.integers(1, 10))
    palette = [int(x) for x in rng.permutation([x for x in range(10) if x != background])]
    if background != 0 and seed_value % 3 == 0:
        palette.remove(0)
        palette.insert(0, 0)
    palette = palette[:int(rng.integers(2, 6))]
    train_counts = _train_counts(rng, train_count)
    test_counts = _test_counts(seed_value, rng, test_count)

    train_colors = [palette[0]]
    train_colors.append(palette[0] if int(rng.integers(2)) else palette[1])
    for index in range(2, train_count):
        if index == 4:
            source_index = train_counts.index(train_counts[4])
            train_colors.append(next(color for color in palette if color != train_colors[source_index]))
        else:
            train_colors.append(palette[int(rng.integers(len(palette)))])

    test_colors = []
    for index in range(test_count):
        unused = [color for color in range(10) if color != background and color not in train_colors]
        if (seed_value + index) % 2 == 0 or not unused:
            test_colors.append(train_colors[int(rng.integers(len(train_colors)))])
        else:
            test_colors.append(unused[int(rng.integers(len(unused)))])

    child_seeds = rng.integers(0, np.iinfo(np.uint32).max, size=train_count + test_count, dtype=np.uint32)
    seen_inputs: set[tuple] = set()
    train = []
    test = []
    for section, counts, colors, offset in (
        (train, train_counts, train_colors, 0),
        (test, test_counts, test_colors, train_count),
    ):
        for index, (foreground_count, foreground) in enumerate(zip(counts, colors)):
            pair_rng = np.random.default_rng(int(child_seeds[offset + index]))
            preferred = None
            if section is train and index == 4 and train_count >= 7 and seed_value % 4 == 0:
                source_index = train_counts.index(foreground_count)
                preferred = _mask(train[source_index]["input"], background)
            inp = _sample_input(pair_rng, foreground_count, background, foreground, seen_inputs, preferred)
            section.append({"input": inp, "output": _construct_output(inp, background)})

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert all(1 <= side <= 30 for side in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert bool(np.all((0 <= grid) & (grid <= 9)))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs = set()
    seen_pairs = set()
    backgrounds = set()
    train_counts = []
    test_counts = []
    for section, counts in ((train, train_counts), (test, test_counts)):
        for pair in section:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _validate_grid(inp)
            _validate_grid(out)
            assert inp.shape == (3, 3)
            input_key = _grid_key(inp)
            pair_key = (input_key, _grid_key(out))
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert input_key not in _SOURCE_INPUT_KEYS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

            background = int(out[-1, -1])
            backgrounds.add(background)
            colors = set(int(x) for x in np.unique(inp))
            assert len(colors) == 2 and background in colors
            foreground_count = int(np.count_nonzero(inp != background))
            background_count = 9 - foreground_count
            assert 1 <= foreground_count <= 6 and foreground_count <= background_count**2
            counts.append(foreground_count)
            assert out.shape == (3 * background_count, 3 * background_count)
            assert set(int(x) for x in np.unique(out)) == colors
            for slot in range(background_count**2):
                row, col = divmod(slot, background_count)
                block = out[3 * row:3 * row + 3, 3 * col:3 * col + 3]
                if slot < foreground_count:
                    assert np.array_equal(block, inp)
                else:
                    assert bool(np.all(block == background))
            assert int(np.count_nonzero(out != background)) == foreground_count**2
    assert len(backgrounds) == 1
    assert len(set(train_counts)) >= 2
    assert any(f > 9 - f for f in train_counts)
    if len(test_counts) > 1:
        assert len(set(test_counts)) > 1


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for pair in puzzle[section])
        for section in ("train", "test")
    )


def _component_count(mask: np.ndarray) -> int:
    remaining = {(int(r), int(c)) for r, c in zip(*np.nonzero(mask))}
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            row, col = stack.pop()
            for neighbor in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(puzzle) for puzzle in puzzles}) >= 40
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    backgrounds = set()
    sides = set()
    masks = set()
    test_counts = set()
    same_color_different_count = 0
    different_color_same_count = 0
    reused_test_color = 0
    new_test_color = 0
    zero_foreground = 0
    nonzero_background_wrap = 0
    zero_foreground_wrap = 0
    repeated_mask_different_color = 0
    connected = scattered = symmetric = asymmetric = 0
    familiar = novel = interpolated = extrapolated = 0
    mixed_tests = wrap_tests = sparse_tests = source_like = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        bg = int(puzzle["train"][0]["output"][-1, -1])
        backgrounds.add(bg)
        training = []
        testing = []
        seen_mask_colors = set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                fg = next(int(color) for color in np.unique(inp) if int(color) != bg)
                f = int(np.sum(inp == fg))
                b = int(np.sum(inp == bg))
                mask = _mask(inp, bg)
                masks.add(mask)
                sides.add(out.shape[0])
                component_count = _component_count(inp != bg)
                connected += component_count == 1
                scattered += component_count > 1
                horizontal = np.array_equal(inp != bg, np.fliplr(inp != bg))
                vertical = np.array_equal(inp != bg, np.flipud(inp != bg))
                symmetric += horizontal or vertical
                asymmetric += not (horizontal or vertical)
                nonzero_background_wrap += bg != 0 and f in (5, 6)
                zero_foreground_wrap += fg == 0 and f in (5, 6)
                if fg == 0:
                    zero_foreground += 1
                if section == "train":
                    training.append((f, fg))
                    if any(old_mask == mask and old_fg != fg for old_mask, old_fg in seen_mask_colors):
                        repeated_mask_different_color += 1
                    seen_mask_colors.add((mask, fg))
                else:
                    test_counts.add(f)
                    testing.append((f, fg))
                    reused_test_color += fg in {color for _, color in training}
                    new_test_color += fg not in {color for _, color in training}
                # Rebuild blocks from visible inputs, independently of construction.
                for slot in range(b * b):
                    row, col = divmod(slot, b)
                    expected = inp if slot < f else np.full((3, 3), bg, dtype=np.int8)
                    assert np.array_equal(out[3*row:3*row+3, 3*col:3*col+3], expected)
        same_color_different_count += any(a != c and b == d for a, b in training for c, d in training)
        different_color_same_count += any(a == c and b != d for a, b in training for c, d in training)
        training_f = {f for f, _ in training}
        mixed_tests += len({f for f, _ in testing}) > 1
        wrap_tests += any(f in (5, 6) for f, _ in testing)
        sparse_tests += any(f == 1 for f, _ in testing)
        familiar += any(f in training_f for f, _ in testing)
        novel += any(f not in training_f for f, _ in testing)
        interpolated += any(min(training_f) < f < max(training_f) and f not in training_f for f, _ in testing)
        extrapolated += any(f < min(training_f) or f > max(training_f) for f, _ in testing)
        source_like += bg == 0 and testing[0][0] == 2
    assert len(backgrounds) >= 7
    assert sides == {9, 12, 15, 18, 21, 24}
    assert test_counts == set(range(1, 7))
    assert len(masks) >= 125
    assert same_color_different_count >= 8
    assert different_color_same_count >= 4
    assert reused_test_color >= 10
    assert new_test_color >= 10
    assert zero_foreground >= 5
    assert nonzero_background_wrap >= 20 and zero_foreground_wrap >= 3
    assert repeated_mask_different_color >= 1
    assert connected >= 40 and scattered >= 40
    assert symmetric >= 20 and asymmetric >= 40
    assert familiar >= 8 and novel >= 8
    assert interpolated >= 1 and extrapolated >= 5
    assert mixed_tests >= 10 and wrap_tests >= 10 and sparse_tests >= 5
    assert source_like >= 4
    for seed, train_count, test_count in ((700, 2, 1), (701, 10, 1), (702, 10, 4)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
    for seed, train_count, test_count in ((800, None, 1), (801, None, 4), (802, 2, None), (803, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    assert _puzzle_key(generate_puzzle(123456, 7, 3)) == _puzzle_key(generate_puzzle(123456, 7, 3))
    assert len({_puzzle_key(generate_puzzle(900 + i, 6, 2)) for i in range(12)}) >= 11
    for arguments in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(1000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
