import numpy as np


def _grid_key(grid):
    return grid.shape, tuple(int(value) for value in grid.flat)


def _stroke_grid(height, width, kind, length, background, foreground, row=0):
    grid = np.full((height, width), background, dtype=np.int8)
    if kind == 0:
        grid[row, :] = foreground
    elif kind == 1:
        columns = np.arange(length)
        grid[height - length + columns, columns] = foreground
    else:
        columns = np.arange(length)
        grid[length - 1 - columns, columns] = foreground
    return grid


def _source_input_keys():
    specifications = (
        (11, 14, 1, 9, 0, 4, 0),
        (11, 12, -1, 9, 0, 3, 0),
        (3, 13, -1, 3, 0, 2, 0),
        (10, 12, 1, 3, 0, 6, 0),
        (10, 4, 0, 4, 0, 8, 2),
    )
    return frozenset(_grid_key(_stroke_grid(*spec)) for spec in specifications)


_SOURCE_INPUT_KEYS = _source_input_keys()


def _fill_below_boundary(grid, kind, length, row, foreground):
    result = grid.copy()
    height, width = grid.shape
    if kind == 0:
        result[row:, :] = foreground
    elif kind == 1:
        for col in range(length):
            result[height - length + col:, col] = foreground
    else:
        for out_row in range(height):
            result[out_row, max(0, length - 1 - out_row):] = foreground
    return result


def _resolve_counts(rng, num_train, num_test):
    for name, value, low, high in (
        ("num_train", num_train, 2, 10),
        ("num_test", num_test, 1, 4),
    ):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
            if not low <= value <= high:
                raise ValueError(f"{name} must be in [{low}, {high}]")
    choices = [
        (train, test)
        for train in range(2, 11)
        for test in range(1, 5)
        if train > test
        and (num_train is None or train == num_train)
        and (num_test is None or test == num_test)
    ]
    if not choices:
        raise ValueError("num_train must exceed num_test")
    return choices[int(rng.integers(len(choices)))]


def _dimension(rng, minimum=3):
    group = int(rng.choice(3, p=[0.28, 0.48, 0.24]))
    low, high = ((minimum, 8), (9, 22), (23, 30))[group]
    return int(rng.integers(max(low, minimum), high + 1))


def _diagonal_length(rng, limit, demonstration, avoid):
    low = 3 if demonstration else 2
    high = limit - 1 if demonstration else limit
    if demonstration:
        length = int(rng.integers(low, high + 1))
    else:
        mode = int(rng.integers(4))
        if mode == 0:
            length = int(rng.integers(low, min(high, 6) + 1))
        elif mode == 1:
            length = int(rng.integers(max(low, high - 3), high + 1))
        else:
            length = int(rng.integers(low, high + 1))
    if avoid is not None and length == avoid and high > low:
        length = low + (length - low + 1) % (high - low + 1)
    return length


def _generate_pair(rng, kind, background, foreground, demonstration=False, avoid=None):
    minimum = 5 if demonstration else 3
    height = _dimension(rng, minimum)
    width = _dimension(rng, minimum)
    if kind == 0:
        length = width
        row = int(rng.integers(height - 1))
    else:
        length = _diagonal_length(rng, min(height, width), demonstration, avoid)
        row = 0
    inp = _stroke_grid(height, width, kind, length, background, foreground, row)
    out = _fill_below_boundary(inp, kind, length, row, foreground)
    return {"input": inp, "output": out}


def _choose_kinds(rng, train_count, test_count):
    train = [1, -1]
    train.extend(int(rng.choice([1, -1, 0], p=[0.35, 0.35, 0.30])) for _ in range(train_count - 2))
    rng.shuffle(train)
    test = [int(rng.choice([1, -1, 0], p=[0.35, 0.35, 0.30])) for _ in range(test_count)]
    return train, test


def _fresh_pair(rng, kind, background, foreground, used, demonstration, avoid):
    for _ in range(24):
        pair = _generate_pair(rng, kind, background, foreground, demonstration, avoid)
        key = _grid_key(pair["input"])
        if key not in used and key not in _SOURCE_INPUT_KEYS:
            return pair
    # A finite fallback keeps legal count requests independent of lucky draws.
    for height in range(5 if demonstration else 3, 31):
        for width in range(5 if demonstration else 3, 31):
            lengths = (width,) if kind == 0 else range(3 if demonstration else 2, min(height, width) + (0 if demonstration else 1))
            for length in lengths:
                if length == avoid:
                    continue
                rows = range(height - 1) if kind == 0 else (0,)
                for row in rows:
                    inp = _stroke_grid(height, width, kind, length, background, foreground, row)
                    key = _grid_key(inp)
                    if key not in used and key not in _SOURCE_INPUT_KEYS:
                        return {"input": inp, "output": _fill_below_boundary(inp, kind, length, row, foreground)}
    raise RuntimeError("no unique boundary stroke available")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # Canvas color is a visible convention shared by the puzzle. Stroke colors
    # are local to pairs; their numeric identities carry no rule.
    background = 0 if rng.random() < 0.55 else int(rng.integers(1, 10))
    train_kinds, test_kinds = _choose_kinds(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    used = set()
    demonstrated_lengths = {}
    for split, kinds in (("train", train_kinds), ("test", test_kinds)):
        for kind in kinds:
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            foreground = int(pair_rng.choice([color for color in range(10) if color != background]))
            demonstration = split == "train" and kind != 0 and kind not in demonstrated_lengths
            avoid = next(iter(demonstrated_lengths.values()), None) if demonstration and demonstrated_lengths else None
            pair = _fresh_pair(pair_rng, kind, background, foreground, used, demonstration, avoid)
            used.add(_grid_key(pair["input"]))
            puzzle[split].append(pair)
            if demonstration:
                demonstrated_lengths[kind] = int(np.count_nonzero(pair["input"] != background))
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((0 <= grid) & (grid <= 9))
    assert inp.shape == out.shape
    height, width = inp.shape
    assert height >= 3 and width >= 3
    colors, counts = np.unique(inp, return_counts=True)
    assert len(colors) == 2
    background = int(colors[np.argmax(counts)])
    foreground = int(colors[np.argmin(counts)])
    positions = np.argwhere(inp == foreground)
    positions = positions[np.argsort(positions[:, 1])]
    rows, cols = positions[:, 0], positions[:, 1]
    length = len(rows)
    assert length >= 2
    assert np.array_equal(cols, np.arange(length))
    differences = np.diff(rows)
    kind = int(differences[0])
    assert kind in (-1, 0, 1) and np.all(differences == kind)
    if kind == 0:
        assert length == width and rows[0] < height - 1
    elif kind == 1:
        assert rows[0] == height - length and rows[-1] == height - 1
    else:
        assert rows[0] == length - 1 and rows[-1] == 0

    # Reconstruct with column thresholds, independently of the fill routine.
    expected = np.full(inp.shape, background, dtype=np.int8)
    for col in range(width):
        if kind == 0:
            threshold = int(rows[0])
        elif kind == 1:
            threshold = height - length + col if col < length else height
        else:
            threshold = max(0, length - 1 - col)
        expected[threshold:, col] = foreground
    assert np.array_equal(out, expected)
    assert np.any(out != inp)
    assert _grid_key(inp) not in _SOURCE_INPUT_KEYS
    return {
        "kind": kind, "length": length, "height": height, "width": width,
        "row": int(rows[0]), "background": background, "foreground": foreground,
        "filled": int(np.count_nonzero(out == foreground)),
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used_inputs, used_pairs = set(), set()
    train_info, test_info = [], []
    for split, infos in (("train", train_info), ("test", test_info)):
        for pair in puzzle[split]:
            info = _inspect_pair(pair)
            inp_key = _grid_key(pair["input"])
            pair_key = (inp_key, _grid_key(pair["output"]))
            assert inp_key not in used_inputs and pair_key not in used_pairs
            used_inputs.add(inp_key)
            used_pairs.add(pair_key)
            infos.append(info)
    backgrounds = {info["background"] for info in train_info + test_info}
    assert len(backgrounds) == 1
    assert {info["kind"] for info in train_info}.issuperset({-1, 1})
    for kind in (-1, 1):
        assert any(
            info["kind"] == kind
            and info["length"] >= 3
            and info["height"] > info["length"]
            and info["width"] > info["length"]
            for info in train_info
        )


def _puzzle_signature(puzzle):
    return tuple(
        (_grid_key(pair["input"]), _grid_key(pair["output"]))
        for split in ("train", "test") for pair in puzzle[split]
    )


def self_test() -> None:
    source_like = _stroke_grid(3, 8, -1, 3, 0, 5)
    assert np.array_equal(
        _fill_below_boundary(source_like, -1, 3, 0, 5),
        np.array([[0, 0, 5, 5, 5, 5, 5, 5],
                  [0, 5, 5, 5, 5, 5, 5, 5],
                  [5, 5, 5, 5, 5, 5, 5, 5]], dtype=np.int8),
    )
    signatures, train_counts, test_counts = set(), set(), set()
    shapes, layouts, backgrounds, foregrounds = set(), set(), set(), set()
    train_horizontal = test_horizontal = short_test = long_test = 0
    repeated_colors = 0
    above_margins, right_margins, horizontal_rows = set(), set(), set()
    combinations, test_horizontal_positions = set(), set()
    wide = tall = compact = large = near_full = small_fill = 0
    near_top_horizontal = near_bottom_horizontal = 0
    full_height = full_width = simple_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = _puzzle_signature(puzzle)
        assert signature == _puzzle_signature(generate_puzzle(seed))
        signatures.add(signature)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_info = [_inspect_pair(pair) for pair in puzzle["train"]]
        test_info = [_inspect_pair(pair) for pair in puzzle["test"]]
        assert len({info["length"] for info in train_info if info["kind"] != 0}) >= 2
        train_horizontal += any(info["kind"] == 0 for info in train_info)
        test_horizontal += any(info["kind"] == 0 for info in test_info)
        simple_puzzles += all(info["kind"] != 0 for info in train_info + test_info)
        repeated_colors += bool({i["foreground"] for i in train_info} & {i["foreground"] for i in test_info})
        for split, infos in (("train", train_info), ("test", test_info)):
            for index, info in enumerate(infos):
                shapes.add((info["height"], info["width"]))
                layouts.add((info["height"], info["width"], info["kind"], info["length"], info["row"]))
                backgrounds.add(info["background"])
                foregrounds.add(info["foreground"])
                height, width, length = info["height"], info["width"], info["length"]
                fraction = info["filled"] / (height * width)
                wide += width >= 2 * height
                tall += height >= 2 * width
                compact += height <= 8 and width <= 8
                large += height >= 23 and width >= 23
                near_full += fraction >= 0.85
                small_fill += fraction <= 0.15
                if info["kind"] == 0:
                    horizontal_rows.add(info["row"])
                    near_top_horizontal += info["row"] <= 1
                    near_bottom_horizontal += info["row"] >= height - 3
                    if split == "test":
                        test_horizontal_positions.add(index)
                else:
                    above_margins.add(height - length)
                    right_margins.add(width - length)
                    combinations.add((
                        info["kind"], (height - length) // 4,
                        (width - length) // 4, info["background"],
                    ))
                    full_height += height == length
                    full_width += width == length
                if split == "test" and info["kind"] != 0:
                    short_test += info["length"] <= 5
                    long_test += info["length"] >= 9
    assert len(signatures) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 50 and len(layouts) >= 100
    assert len(backgrounds) >= 3 and len(foregrounds) >= 7
    assert train_horizontal >= 10 and test_horizontal >= 10
    assert short_test >= 8 and long_test >= 8
    assert repeated_colors >= 10
    assert simple_puzzles >= 1
    assert test_horizontal_positions == {0, 1, 2, 3}
    assert len(above_margins) >= 15 and len(right_margins) >= 15
    assert min(above_margins) == min(right_margins) == 0
    assert len(horizontal_rows) >= 10
    assert near_top_horizontal >= 5 and near_bottom_horizontal >= 5
    assert full_height >= 5 and full_width >= 5
    assert min(wide, tall, compact, large) >= 10
    assert near_full >= 10 and small_fill >= 10
    assert len(combinations) >= 80
    requests = ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))
    for index, (train_count, test_count) in enumerate(requests):
        puzzle = generate_puzzle(1000 + index, train_count, test_count)
        assert _puzzle_signature(puzzle) == _puzzle_signature(generate_puzzle(1000 + index, train_count, test_count))
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4), (True, 1), (2, False)):
        try:
            generate_puzzle(2000, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {(train_count, test_count)}")
