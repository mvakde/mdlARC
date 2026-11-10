import numpy as np


_SOURCE_PAIR_KEYS = (
    (
        (
            "000000000000000",
            "000000000000000",
            "088888888888880",
            "088888888888880",
            "088118883388880",
            "088118883388880",
            "088888888888880",
            "088888882288880",
            "088888882288880",
            "088888888888880",
            "088888888888880",
            "088888888888880",
            "000000000000000",
            "000000000000000",
            "000000000000000",
            "000000000000000",
        ),
        ((1, 3), (0, 2)),
    ),
    (
        (
            "000000000000000",
            "008888888800000",
            "008228888800000",
            "008228888800000",
            "008888888800000",
            "008888888800000",
            "008888833800000",
            "008888833800000",
            "008888888800000",
            "000000000000000",
            "000000000000000",
            "000000000000000",
            "000000000000000",
        ),
        ((2, 0), (0, 3)),
    ),
    (
        (
            "0000000000000000",
            "0888888888000000",
            "0888888888000000",
            "0855884488000000",
            "0855884488000000",
            "0888888888000000",
            "0888888888000000",
            "0888888888000000",
            "0833888888000000",
            "0833888888000000",
            "0888888888000000",
            "0888888888000000",
            "0888888888000000",
            "0000000000000000",
        ),
        ((5, 4), (3, 0)),
    ),
    (
        (
            "000000000000000",
            "000000000000000",
            "000000000000000",
            "008888888888880",
            "008888888888880",
            "008888888888880",
            "008822888866880",
            "008822888866880",
            "008888888888880",
            "008888888888880",
            "008811888833880",
            "008811888833880",
            "008888888888880",
            "000000000000000",
        ),
        ((2, 6), (1, 3)),
    ),
)


def _grid_key(grid: np.ndarray) -> tuple:
    normalized = np.ascontiguousarray(grid, dtype=np.int16)
    return tuple(grid.shape), normalized.tobytes()


def _input_rows_key(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _output_values_key(grid: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(value) for value in row)
        for row in grid
    )


def _matches_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    key = (
        _input_rows_key(input_grid),
        _output_values_key(output_grid),
    )
    return key in _SOURCE_PAIR_KEYS


def _checked_count(value: int | None, name: str, minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _resolve_counts(rng: np.random.Generator, num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must be greater than num_test")
        return train, test
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


_BLOCK_SIZES = ((2, 2), (1, 1), (3, 3), (3, 1), (2, 3))
_DIAGONALS = ((0, 3), (1, 2))


def _role_pair(rng: np.random.Generator) -> tuple[int, int]:
    colors = rng.choice(10, size=2, replace=False)
    return int(colors[0]), int(colors[1])


def _sample_extent(rng: np.random.Generator, block_extent: int) -> int:
    minimum = 2 * block_extent + 3
    if float(rng.random()) < 0.38:
        return int(rng.integers(max(18, minimum), 27))
    return int(rng.integers(minimum, 18))


def _sample_canvas_axis(rng: np.random.Generator, panel_extent: int) -> tuple[int, int, int]:
    draw = float(rng.random())
    if draw < 0.16:
        canvas = 30
    elif draw < 0.69:
        canvas = panel_extent + int(rng.integers(2, min(10, 30 - panel_extent) + 1))
    else:
        canvas = int(rng.integers(panel_extent + 2, 31))
    before = int(rng.integers(1, canvas - panel_extent))
    after = canvas - panel_extent - before
    return canvas, before, after


def _sample_levels(rng: np.random.Generator, extent: int,
                   block_extent: int) -> tuple[int, int]:
    positions = range(1, extent - block_extent)
    choices = [(a, b) for a in positions for b in positions
               if b >= a + block_extent + 1]
    assert choices
    return choices[int(rng.integers(len(choices)))]


def _generate_pair(rng: np.random.Generator, occupied: tuple[int, ...],
                   block_size: tuple[int, int], repeat: bool) -> tuple[np.ndarray, np.ndarray]:
    occupied = tuple(sorted(occupied))
    assert len(occupied) in (2, 3, 4)
    assert len(set(occupied)) == len(occupied)
    assert {slot // 2 for slot in occupied} == {0, 1}
    assert {slot % 2 for slot in occupied} == {0, 1}
    block_height, block_width = block_size
    background, panel_color = _role_pair(rng)
    panel_height = _sample_extent(rng, block_height)
    panel_width = _sample_extent(rng, block_width)
    height, top, _ = _sample_canvas_axis(rng, panel_height)
    width, left, _ = _sample_canvas_axis(rng, panel_width)
    row_levels = _sample_levels(rng, panel_height, block_height)
    column_levels = _sample_levels(rng, panel_width, block_width)

    inp = np.full((height, width), background, dtype=np.int8)
    inp[top:top + panel_height, left:left + panel_width] = panel_color
    out = np.full((2, 2), background, dtype=np.int8)
    palette = np.array([color for color in range(10)
                        if color not in (background, panel_color)], dtype=np.int8)
    colors = rng.choice(palette, size=len(occupied), replace=False)
    if repeat:
        positions = rng.choice(len(occupied), size=2, replace=False)
        colors[int(positions[1])] = colors[int(positions[0])]
    for slot, color in zip(occupied, colors):
        row = slot // 2
        col = slot % 2
        r = top + row_levels[row]
        c = left + column_levels[col]
        inp[r:r + block_height, c:c + block_width] = color
        out[row, col] = color
    return inp, out


def _random_pattern(rng: np.random.Generator, weights: tuple[float, float, float]) -> tuple[int, ...]:
    option = int(rng.choice(3, p=weights))
    if option == 0:
        return (0, 1, 2, 3)
    if option == 1:
        missing = int(rng.integers(4))
        return tuple(slot for slot in range(4) if slot != missing)
    return _DIAGONALS[int(rng.integers(2))]


def _training_patterns(rng: np.random.Generator, count: int) -> list[tuple[int, ...]]:
    missing = int(rng.integers(4))
    triple = tuple(slot for slot in range(4) if slot != missing)
    covering = _DIAGONALS[0] if missing in _DIAGONALS[0] else _DIAGONALS[1]
    patterns = [triple, covering]
    if count >= 3:
        patterns.append((0, 1, 2, 3))
    while len(patterns) < count:
        patterns.append(_random_pattern(rng, (0.19, 0.49, 0.32)))
    rng.shuffle(patterns)
    return patterns


def _test_patterns(rng: np.random.Generator, count: int) -> list[tuple[int, ...]]:
    if count == 1:
        return [_random_pattern(rng, (0.42, 0.34, 0.24))]
    patterns = [(0, 1, 2, 3), _random_pattern(rng, (0.0, 0.55, 0.45))]
    while len(patterns) < count:
        patterns.append(_random_pattern(rng, (0.36, 0.36, 0.28)))
    rng.shuffle(patterns)
    return patterns


def _size_plans(rng: np.random.Generator, train_count: int,
                test_count: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    # Each size used at test time has a training demonstration. A minority of
    # puzzles retain only the source's 2x2 size.
    sizes = [(2, 2)]
    if float(rng.random()) >= 0.14:
        alternatives = list(_BLOCK_SIZES[1:])
        rng.shuffle(alternatives)
        sizes.append(alternatives[0])
        if train_count >= 3 and float(rng.random()) < 0.27:
            sizes.append(alternatives[1])
    train = sizes.copy()
    while len(train) < train_count:
        train.append(sizes[int(rng.integers(len(sizes)))])
    rng.shuffle(train)
    test = [sizes[int(rng.integers(len(sizes)))] for _ in range(test_count)]
    return train, test


def _repeat_plans(rng: np.random.Generator, train_count: int,
                  test_count: int) -> tuple[list[bool], list[bool]]:
    train = [True, False]
    while len(train) < train_count:
        train.append(bool(rng.random() < 0.40))
    rng.shuffle(train)
    test = [bool(rng.random() < 0.40) for _ in range(test_count)]
    return train, test


def _components(region: np.ndarray, panel_color: int) -> list[tuple[int, int, int, int, int]]:
    height, width = region.shape
    seen = np.zeros(region.shape, dtype=bool)
    blocks = []
    for row in range(height):
        for col in range(width):
            if seen[row, col] or int(region[row, col]) == panel_color:
                continue
            color = int(region[row, col])
            stack = [(row, col)]
            seen[row, col] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (0 <= rr < height and 0 <= cc < width and not seen[rr, cc]
                            and int(region[rr, cc]) == color):
                        seen[rr, cc] = True
                        stack.append((rr, cc))
            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            assert len(cells) == (r1 - r0 + 1) * (c1 - c0 + 1)
            assert np.all(region[r0:r1 + 1, c0:c1 + 1] == color)
            blocks.append((r0, c0, r1 - r0 + 1, c1 - c0 + 1, color))
    return blocks


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray | None = None) -> dict:
    height, width = input_grid.shape
    background = int(input_grid[0, 0])
    assert np.all(input_grid[0] == background)
    assert np.all(input_grid[-1] == background)
    assert np.all(input_grid[:, 0] == background)
    assert np.all(input_grid[:, -1] == background)
    coords = np.argwhere(input_grid != background)
    assert len(coords) > 0
    top, left = (int(x) for x in coords.min(axis=0))
    bottom, right = (int(x) for x in coords.max(axis=0))
    assert 1 <= top <= bottom < height - 1
    assert 1 <= left <= right < width - 1
    region = input_grid[top:bottom + 1, left:right + 1]
    panel_color = int(region[0, 0])
    assert panel_color != background
    assert np.all(input_grid[:top] == background)
    assert np.all(input_grid[bottom + 1:] == background)
    assert np.all(input_grid[:, :left] == background)
    assert np.all(input_grid[:, right + 1:] == background)
    assert np.all(region != background)
    assert np.all(region[0] == panel_color)
    assert np.all(region[-1] == panel_color)
    assert np.all(region[:, 0] == panel_color)
    assert np.all(region[:, -1] == panel_color)
    blocks = _components(region, panel_color)
    assert 2 <= len(blocks) <= 4
    sizes = {(block[2], block[3]) for block in blocks}
    assert len(sizes) == 1
    block_height, block_width = next(iter(sizes))
    assert (block_height, block_width) in _BLOCK_SIZES
    row_levels = tuple(sorted({block[0] for block in blocks}))
    col_levels = tuple(sorted({block[1] for block in blocks}))
    assert len(row_levels) == len(col_levels) == 2
    assert row_levels[1] - row_levels[0] >= block_height + 1
    assert col_levels[1] - col_levels[0] >= block_width + 1
    assert all(1 <= r <= region.shape[0] - block_height - 1 for r in row_levels)
    assert all(1 <= c <= region.shape[1] - block_width - 1 for c in col_levels)
    expected = np.full((2, 2), background, dtype=np.int8)
    occupied = set()
    color_slots: dict[int, list[int]] = {}
    for r, c, bh, bw, color in blocks:
        slot_row = row_levels.index(r)
        slot_col = col_levels.index(c)
        slot = slot_row * 2 + slot_col
        assert slot not in occupied
        occupied.add(slot)
        expected[slot_row, slot_col] = color
        color_slots.setdefault(color, []).append(slot)
    if output_grid is not None:
        assert output_grid.shape == (2, 2)
        assert np.array_equal(output_grid, expected)
    return {
        "background": background,
        "panel": panel_color,
        "canvas_shape": input_grid.shape,
        "panel_shape": region.shape,
        "margins": (top, height - bottom - 1, left, width - right - 1),
        "block_size": (block_height, block_width),
        "row_levels": row_levels,
        "column_levels": col_levels,
        "occupied_slots": tuple(sorted(occupied)),
        "color_slots": color_slots,
        "repeated": any(len(slots) > 1 for slots in color_slots.values()),
        "expected": expected,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    inputs = set()
    pairs = set()
    info: dict[str, list[dict]] = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= d <= 30 for d in grid.shape)
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            key = _grid_key(inp)
            pair_key = (key, _grid_key(out))
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            assert not np.array_equal(inp, out)
            assert not _matches_source_pair(inp, out)
            info[split].append(_analyze_pair(inp, out))
    training = info["train"]
    tests = info["test"]
    train_roles = {(item["background"], item["panel"]) for item in training}
    assert len(train_roles) >= 2
    assert {item["block_size"] for item in tests}.issubset(
        {item["block_size"] for item in training})
    assert {item["repeated"] for item in training} == {False, True}
    training_patterns = {item["occupied_slots"] for item in training}
    assert any(len(pattern) == 3 for pattern in training_patterns)
    assert any(pattern in _DIAGONALS for pattern in training_patterns)
    demonstrated_slots = set()
    for item in training:
        demonstrated_slots.update(item["occupied_slots"])
        if len(item["occupied_slots"]) == 2:
            assert item["occupied_slots"] in _DIAGONALS
    assert demonstrated_slots == {0, 1, 2, 3}
    for item in tests:
        assert len(item["occupied_slots"]) in (2, 3, 4)
        if len(item["occupied_slots"]) == 2:
            assert item["occupied_slots"] in _DIAGONALS
    if test_count >= 2:
        assert any(len(item["occupied_slots"]) == 4 for item in tests)
        assert any(len(item["occupied_slots"]) < 4 for item in tests)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    train_patterns = _training_patterns(rng, train_count)
    test_patterns = _test_patterns(rng, test_count)
    train_sizes, test_sizes = _size_plans(rng, train_count, test_count)
    train_repeats, test_repeats = _repeat_plans(rng, train_count, test_count)
    used_inputs = set()

    def build(pattern: tuple[int, ...], size: tuple[int, int], repeat: bool) -> dict:
        child_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
        pair_rng = np.random.default_rng(child_seed)
        for _ in range(32):
            inp, out = _generate_pair(pair_rng, pattern, size, repeat)
            key = _grid_key(inp)
            if key not in used_inputs and not _matches_source_pair(inp, out):
                used_inputs.add(key)
                return {"input": inp, "output": out}
        raise RuntimeError("unable to construct a unique pair")

    train = [build(p, s, r) for p, s, r in zip(train_patterns, train_sizes, train_repeats)]
    # Roles are visible in every input. Ensure training actually demonstrates
    # that their literal colors can change between pairs.
    if len({(int(pair["input"][0, 0]), _analyze_pair(pair["input"])["panel"])
            for pair in train}) == 1:
        first = train[0]
        forbidden = (int(first["input"][0, 0]), _analyze_pair(first["input"])["panel"])
        for _ in range(32):
            replacement = build(train_patterns[1], train_sizes[1], train_repeats[1])
            new_roles = (int(replacement["input"][0, 0]),
                         _analyze_pair(replacement["input"])["panel"])
            if new_roles != forbidden:
                train[1] = replacement
                break
        else:
            raise RuntimeError("unable to vary training roles")
    test = [build(p, s, r) for p, s, r in zip(test_patterns, test_sizes, test_repeats)]
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a["input"], b["input"]) and
        np.array_equal(a["output"], b["output"])
        for a, b in zip(first[split], second[split]))
        for split in ("train", "test"))


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(73), generate_puzzle(73))
    assert _puzzles_equal(generate_puzzle(74, 7, 3), generate_puzzle(74, 7, 3))
    for index, (train, test) in enumerate(((2, 1), (10, 1), (10, 4),
                                           (None, 1), (None, 4),
                                           (2, None), (10, None))):
        puzzle = generate_puzzle(200 + index, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4},
                   {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(300, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid count accepted: {kwargs}")
    train_counts, test_counts = set(), set()
    roles, shapes, panels, layouts, sizes, test_patterns = (set() for _ in range(6))
    repeated_orientations = set()
    input_keys, puzzle_keys = set(), set()
    pair_count = 0
    large_panels = large_margins = max_canvases = 0
    low_rows = high_rows = low_cols = high_cols = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_key = tuple((split, tuple((_grid_key(pair["input"]),
                                         _grid_key(pair["output"]))
                                        for pair in puzzle[split]))
                           for split in ("train", "test"))
        puzzle_keys.add(puzzle_key)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                item = _analyze_pair(pair["input"], pair["output"])
                roles.add((item["background"], item["panel"]))
                shapes.add(item["canvas_shape"])
                panels.add(item["panel_shape"])
                layouts.add((item["panel_shape"], item["row_levels"],
                             item["column_levels"], item["occupied_slots"],
                             item["block_size"]))
                sizes.add(item["block_size"])
                if split == "test":
                    test_patterns.add(item["occupied_slots"])
                for slots in item["color_slots"].values():
                    if len(slots) > 1:
                        a, b = slots[:2]
                        repeated_orientations.add("row" if a // 2 == b // 2 else
                                                  "column" if a % 2 == b % 2 else
                                                  "diagonal")
                large_panels += int(max(item["panel_shape"]) >= 19)
                large_margins += int(max(item["margins"]) >= 6)
                max_canvases += int(30 in item["canvas_shape"])
                r0, r1 = item["row_levels"]
                c0, c1 = item["column_levels"]
                ph, pw = item["panel_shape"]
                bh, bw = item["block_size"]
                high_rows += int(2 * (r1 + bh / 2) < ph)
                low_rows += int(2 * (r0 + bh / 2) > ph)
                high_cols += int(2 * (c1 + bw / 2) < pw)
                low_cols += int(2 * (c0 + bw / 2) > pw)
                input_keys.add(_grid_key(pair["input"]))
                pair_count += 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(puzzle_keys) == 42 and len(input_keys) == pair_count
    assert len(roles) >= 25 and len(shapes) >= 40 and len(panels) >= 35
    assert len(layouts) >= 100
    assert sizes == set(_BLOCK_SIZES)
    assert {(0, 3), (1, 2), (0, 1, 2, 3)}.issubset(test_patterns)
    assert any(len(pattern) == 3 for pattern in test_patterns)
    assert repeated_orientations == {"row", "column", "diagonal"}
    assert large_panels >= 20 and large_margins >= 20 and max_canvases >= 10
    assert min(low_rows, high_rows, low_cols, high_cols) >= 5
