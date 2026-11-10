"""Generate ARC puzzles with horizontal bars moving one cell left."""

import numpy as np


def _rows_to_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[int(cell) for cell in row] for row in rows], dtype=np.int8)


_SOURCE_INPUTS = (
    _rows_to_grid(("00000000", "00333300", "00000000", "00000000", "03300000", "00000000", "00000000", "00000000")),
    _rows_to_grid(("444", "000", "000")),
    _rows_to_grid(("7777000", "0000000", "0077700", "0000000", "0777770", "0000000", "0000000")),
    _rows_to_grid(("0066666000", "0000000000", "0000006600", "0000000000", "0666000000", "0000000000", "0000000000", "0000666666", "0000000000", "0000000000")),
)


def _validated_count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng: np.random.Generator, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _validated_count(num_train, "num_train", 2, 10)
    test = _validated_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _array_key(grid: np.ndarray) -> tuple:
    return grid.shape, grid.astype(np.int16, copy=False).tobytes()


def _matches_source_input(grid: np.ndarray) -> bool:
    return any(grid.shape == source.shape and np.array_equal(grid, source) for source in _SOURCE_INPUTS)


def _background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    assert int(counts[index]) > grid.size // 2
    return int(colors[index])


def _bars(grid: np.ndarray, background: int) -> list[tuple[int, int, int, int]]:
    bars = []
    for row in range(grid.shape[0]):
        column = 0
        while column < grid.shape[1]:
            if int(grid[row, column]) == background:
                column += 1
                continue
            start = column
            color = int(grid[row, column])
            while column < grid.shape[1] and int(grid[row, column]) == color:
                column += 1
            end = column
            assert column == grid.shape[1] or int(grid[row, column]) == background
            bars.append((row, start, end, color))
    assert bars
    assert len({row for row, _, _, _ in bars}) < grid.shape[0]
    for left in bars:
        for right in bars:
            if right <= left or abs(left[0] - right[0]) != 1:
                continue
            assert left[2] < right[1] or right[2] < left[1]
    return bars


def _apply_rule(grid: np.ndarray, input_background: int, output_background: int) -> np.ndarray:
    result = np.full(grid.shape, output_background, dtype=np.int8)
    mask = grid[:, 1:] != input_background
    result[:, :-1][mask] = grid[:, 1:][mask]
    return result


def _sample_dimension(rng: np.random.Generator, regime: str) -> int:
    bands = ((3, 8), (9, 16), (17, 30))
    probabilities = (0.55, 0.27, 0.18) if regime == "source_train" else (0.23, 0.37, 0.40)
    low, high = bands[int(rng.choice(3, p=probabilities))]
    return int(rng.integers(low, high + 1))


def _can_place(bars: list[tuple[int, int, int]], height: int, row: int, start: int, end: int) -> bool:
    if row not in {bar[0] for bar in bars} and len({bar[0] for bar in bars}) >= height - 1:
        return False
    for old_row, old_start, old_end in bars:
        if abs(old_row - row) <= 1 and not (end < old_start or old_end < start):
            return False
    return True


def _sample_length(rng: np.random.Generator, width: int, maximum: int) -> int:
    choice = int(rng.integers(0, 5))
    if choice == 0:
        return 1
    if choice == 1:
        return int(rng.integers(1, min(maximum, 4) + 1))
    if choice == 2:
        return int(rng.integers(1, maximum + 1))
    if choice == 3:
        return int(rng.integers(max(1, maximum // 2), maximum + 1))
    return maximum if maximum == width else int(rng.integers(1, maximum + 1))


def _place_bar(rng: np.random.Generator, bars: list[tuple[int, int, int]], height: int, width: int, budget: int, mode: str) -> bool:
    maximum = min(width, budget)
    if mode == "vanish":
        lengths = [1]
    elif mode == "left":
        preferred = int(rng.integers(2, maximum + 1))
        lengths = [preferred] + [int(value) for value in rng.permutation([x for x in range(2, maximum + 1) if x != preferred])]
    elif mode == "nonleft":
        usable = min(maximum, width - 1)
        preferred = int(rng.integers(1, usable + 1))
        lengths = [preferred] + [int(value) for value in rng.permutation([x for x in range(1, usable + 1) if x != preferred])]
    else:
        preferred = _sample_length(rng, width, maximum)
        lengths = [preferred] + [int(value) for value in rng.permutation([x for x in range(1, maximum + 1) if x != preferred])]
    for length in lengths:
        positions = []
        for row in range(height):
            for start in range(width - length + 1):
                if mode in ("left", "vanish") and start != 0:
                    continue
                if mode == "nonleft" and start == 0:
                    continue
                if _can_place(bars, height, row, start, start + length):
                    positions.append((row, start))
        if positions:
            # Boundary positions receive extra weight, but every legal margin is possible.
            if mode == "random" and rng.random() < 0.35:
                boundary = [position for position in positions if position[1] in (0, width - length)]
                if boundary:
                    positions = boundary
            row, start = positions[int(rng.integers(len(positions)))]
            bars.append((row, start, start + length))
            return True
    return False


def _make_pair(rng: np.random.Generator, regime: str, input_background: int, output_background: int, familiar: tuple[int, ...], novel: tuple[int, ...], witness: bool = False) -> dict:
    if witness:
        height = int(rng.integers(6, 13))
        width = int(rng.integers(6, 13))
    else:
        height = _sample_dimension(rng, regime)
        width = _sample_dimension(rng, regime)
        if regime == "source_train" and rng.random() < 0.06:
            height, width = 3, 3
    capacity = min(12, max(1, height * width // 4), height * max(1, (width + 1) // 3))
    vanishing_case = not witness and rng.random() < 0.06
    if witness:
        desired = int(rng.integers(2, min(capacity, 5) + 1))
    elif vanishing_case:
        desired = 1
    elif rng.random() < 0.22:
        desired = int(rng.integers(1, min(capacity, 2) + 1))
    else:
        desired = int(rng.integers(1, capacity + 1))
    bars: list[tuple[int, int, int]] = []
    max_cells = max(1, int(height * width * 0.38))
    modes = ["left", "nonleft"] if witness else (["vanish"] if vanishing_case else [])
    modes.extend(["random"] * (desired - len(modes)))
    for mode in modes:
        used = sum(end - start for _, start, end in bars)
        if used >= max_cells:
            break
        _place_bar(rng, bars, height, width, max_cells - used, mode)
    if not bars:
        bars = [(0, 0, 1)]
    if witness:
        assert len(bars) >= 2 and bars[0][1] == 0 and bars[0][2] >= 2 and bars[1][1] > 0

    palette = familiar
    if regime == "target_test" and novel:
        selection = int(rng.integers(0, 3))
        if selection == 1:
            palette = novel
        elif selection == 2:
            palette = familiar + novel
    max_colors = min(3, len(bars), len(palette))
    color_count = 2 if witness else int(rng.integers(1, max_colors + 1))
    selected = [int(value) for value in rng.choice(palette, size=color_count, replace=False)]
    colors = [selected[index] for index in range(color_count)]
    colors += [selected[int(rng.integers(color_count))] for _ in range(len(bars) - color_count)]
    rng.shuffle(colors)
    grid = np.full((height, width), input_background, dtype=np.int8)
    for (row, start, end), color in zip(bars, colors):
        grid[row, start:end] = color
    return {"input": grid, "output": _apply_rule(grid, input_background, output_background)}


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    palette = [int(value) for value in rng.permutation(10)]
    output_background = palette[0]
    background_pool = tuple(palette[1:4])
    familiar = tuple(palette[4:7])
    novel = tuple(palette[7:])
    puzzle = {"train": [], "test": []}
    seen = set()
    for index in range(train_count + test_count):
        split = "train" if index < train_count else "test"
        regime = "source_train" if split == "train" else "target_test"
        if index == 0:
            background = background_pool[0]
        elif index == 1:
            background = background_pool[1]
        else:
            background = background_pool[int(rng.integers(len(background_pool)))]
        for _ in range(40):
            pair = _make_pair(rng, regime, background, output_background, familiar, novel, witness=index == 0)
            key = _array_key(pair["input"])
            if key not in seen and not _matches_source_input(pair["input"]):
                break
        else:
            raise RuntimeError("could not construct a unique pair")
        seen.add(key)
        puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    rng.shuffle(puzzle["test"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    output_background = None
    train_backgrounds = set()
    moved_colors = set()
    saw_clipping = False
    saw_ordinary = False
    saw_mixed = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            source, target = pair["input"], pair["output"]
            assert source.shape == target.shape and not np.array_equal(source, target)
            assert not _matches_source_input(source)
            key = _array_key(source)
            assert key not in seen
            seen.add(key)
            bg = _background(source)
            out_bg = _background(target)
            assert bg != out_bg and out_bg not in set(int(v) for v in np.unique(source))
            if output_background is None:
                output_background = out_bg
            assert out_bg == output_background
            bars = _bars(source, bg)
            assert np.array_equal(target, _apply_rule(source, bg, out_bg))
            assert sum(end - start for _, start, end, _ in bars) < source.size // 2 + 1
            if split == "train":
                train_backgrounds.add(bg)
                colors = {color for _, _, _, color in bars}
                saw_mixed |= len(colors) >= 2 and any(start > 0 for _, start, _, _ in bars)
                for _, start, end, color in bars:
                    if start == 0 and end >= 2:
                        saw_clipping = True
                        moved_colors.add(color)
                    elif start > 0:
                        saw_ordinary = True
                        moved_colors.add(color)
    assert len(train_backgrounds) >= 2
    assert saw_clipping and saw_ordinary and saw_mixed
    assert len(moved_colors) >= 2


def _puzzles_equal(left: dict, right: dict) -> bool:
    return all(len(left[split]) == len(right[split]) and all(np.array_equal(a[name], b[name]) for a, b in zip(left[split], right[split]) for name in ("input", "output")) for split in ("train", "test"))


def self_test() -> None:
    for seed, request in ((12345, {}), (67890, {"num_train": 7, "num_test": 3}), (70000, {"num_train": 2, "num_test": 1}), (70001, {"num_train": 10, "num_test": 1}), (70002, {"num_train": 10, "num_test": 4}), (71000, {"num_test": 1}), (71001, {"num_test": 4}), (71002, {"num_train": 2}), (71003, {"num_train": 10})):
        first = generate_puzzle(seed, **request)
        second = generate_puzzle(seed, **request)
        assert _puzzles_equal(first, second)
        validate_generated_puzzle(first)
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}, {"num_train": True}, {"num_train": 3.5}):
        try:
            generate_puzzle(72000, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid count request: {request}")
    measures = {name: set() for name in ("train_counts", "test_counts", "train_shapes", "test_shapes", "train_bar_counts", "test_bar_counts", "backgrounds", "layouts")}
    flags = {name: 0 for name in ("rect_train", "rect_test", "large_train", "large_test", "singletons", "vanishing", "shared_row", "adjacent_rows", "mixed_colors", "familiar_test", "novel_test", "full_width", "train_shape_variety", "train_count_variety", "train_length_variety", "background_variety", "mixed_test_colors")}
    for seed in range(5000, 5042):
        puzzle = generate_puzzle(seed)
        measures["train_counts"].add(len(puzzle["train"]))
        measures["test_counts"].add(len(puzzle["test"]))
        train_colors = set()
        train_backgrounds = set()
        train_shapes = set()
        train_bar_counts = set()
        train_lengths = set()
        for pair in puzzle["train"]:
            bg = _background(pair["input"])
            bars = _bars(pair["input"], bg)
            train_colors |= {color for _, _, _, color in bars}
            train_backgrounds.add(bg)
            train_shapes.add(pair["input"].shape)
            train_bar_counts.add(len(bars))
            train_lengths.update(end - start for _, start, end, _ in bars)
        flags["background_variety"] += len(train_backgrounds) >= 2
        flags["train_shape_variety"] += len(train_shapes) >= 2
        flags["train_count_variety"] += len(train_bar_counts) >= 2
        flags["train_length_variety"] += len(train_lengths) >= 2
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                bg = _background(grid)
                bars = _bars(grid, bg)
                shape = grid.shape
                measures[split + "_shapes"].add(shape)
                measures[split + "_bar_counts"].add(len(bars))
                measures["backgrounds"].add((bg, _background(pair["output"])))
                measures["layouts"].add((split, shape, tuple((r, s, e - s) for r, s, e, _ in bars)))
                flags["rect_" + split] += shape[0] != shape[1]
                flags["large_" + split] += max(shape) >= 17
                flags["singletons"] += any(e - s == 1 for _, s, e, _ in bars)
                flags["vanishing"] += len(np.unique(pair["output"])) == 1
                flags["shared_row"] += len({r for r, _, _, _ in bars}) < len(bars)
                rows = {r for r, _, _, _ in bars}
                flags["adjacent_rows"] += any(r + 1 in rows for r in rows)
                flags["mixed_colors"] += len({color for _, _, _, color in bars}) >= 2
                flags["full_width"] += any(e - s == shape[1] for _, s, e, _ in bars)
                if split == "test":
                    colors = {color for _, _, _, color in bars}
                    flags["familiar_test"] += bool(colors & train_colors)
                    flags["novel_test"] += bool(colors - train_colors)
                    flags["mixed_test_colors"] += bool(colors & train_colors) and bool(colors - train_colors)
                expected = np.full(shape, _background(pair["output"]), dtype=np.int8)
                for row, start, end, color in bars:
                    if end > 1:
                        expected[row, max(0, start - 1):end - 1] = color
                assert np.array_equal(expected, pair["output"])
    assert len(measures["train_counts"]) >= 4 and len(measures["test_counts"]) >= 3
    assert len(measures["train_shapes"]) >= 20 and len(measures["test_shapes"]) >= 20
    assert len(measures["layouts"]) >= 150 and len(measures["backgrounds"]) >= 18
    assert all(value > 0 for value in flags.values()), flags
    assert not _puzzles_equal(generate_puzzle(88001, 6, 2), generate_puzzle(88002, 6, 2))
