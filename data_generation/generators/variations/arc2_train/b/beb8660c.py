import numpy as np


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _pair_key(input_grid: np.ndarray, output_grid: np.ndarray) -> tuple:
    return _grid_key(input_grid), _grid_key(output_grid)


def _source_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.array([[int(c) for c in row] for row in rows], dtype=np.int8)


_SOURCE_ROWS = (
    (
        ("0001", "0000", "2220", "0000", "0330", "0000", "8888"),
        ("0000", "0000", "0000", "0001", "0033", "0222", "8888"),
    ),
    (
        ("0220000", "0000300", "1110000", "0000000", "0555555",
         "0000000", "6666600", "0004444", "0000000", "8888888"),
        ("0000000", "0000000", "0000000", "0000003", "0000022",
         "0000111", "0004444", "0066666", "0555555", "8888888"),
    ),
    (("220", "040", "888"), ("004", "022", "888")),
    (
        ("66600000", "00001111", "00000000", "04444400",
         "00000770", "00000000", "22222220", "00333333",
         "09000000", "00000000", "88888888"),
        ("00000000", "00000000", "00000000", "00000009",
         "00000077", "00000666", "00001111", "00044444",
         "00333333", "02222222", "88888888"),
    ),
)
_SOURCE_PAIR_KEYS = frozenset(
    _pair_key(_source_grid(a), _source_grid(b)) for a, b in _SOURCE_ROWS
)


def _valid_count(name: str, value: int | None, low: int, high: int) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")


def _counts(seed_code: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    _valid_count("num_train", num_train, 2, 10)
    _valid_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        test = 1 + seed_code % 4
        low = max(2, test + 1)
        train = low + (seed_code // 4) % (11 - low)
    elif num_train is None:
        test = int(num_test)
        low = max(2, test + 1)
        train = low + seed_code % (11 - low)
    elif num_test is None:
        train = int(num_train)
        test = 1 + seed_code % min(4, train - 1)
    else:
        train, test = int(num_train), int(num_test)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _width_schedule(rng: np.random.Generator, train_count: int,
                    test_count: int, mode: int) -> tuple[list[int], list[int]]:
    if mode == 0:  # Some source-like 3..7 demonstrations, then width 8.
        train = [3, 7]
        tests = [8] * test_count
        extra_range = (3, 8)
    elif mode == 1:  # A test width already demonstrated.
        train = [int(rng.integers(4, 9)), int(rng.integers(6, 10))]
        if train[0] == train[1]:
            train[0] = 4
        tests = [train[int(rng.integers(0, 2))] for _ in range(test_count)]
        extra_range = (3, 10)
    elif mode == 2:  # Smaller tests.
        train = [8, int(rng.integers(9, 11))]
        tests = [int(rng.integers(6, 8)) for _ in range(test_count)]
        extra_range = (8, 11)
    elif mode == 3:  # One or several new lengths.
        train = [int(rng.integers(3, 7)), int(rng.integers(6, 9))]
        if train[0] == train[1]:
            train[0] = 3
        tests = [int(rng.integers(max(train) + 1, 13))
                 for _ in range(test_count)]
        extra_range = (3, 9)
    else:  # Wide sparse canvases.
        train = [int(rng.integers(6, 9)), int(rng.integers(9, 13))]
        tests = [int(rng.integers(max(11, max(train) + 1), 19))
                 for _ in range(test_count)]
        extra_range = (4, 13)
    while len(train) < train_count:
        train.append(int(rng.integers(*extra_range)))
    return train, tests


def _bar_lengths(rng: np.random.Generator, width: int, complete: bool,
                 min_bars: int, shared_row: bool) -> list[int]:
    if complete or width == 3:
        return list(range(1, width))
    maximum = min(8, width - 2)
    count = int(rng.integers(min_bars, maximum + 1))
    if shared_row:
        remainder = rng.choice(np.arange(3, width), size=count - 2, replace=False)
        return sorted([1, 2, *(int(x) for x in remainder)])
    return sorted(int(x) for x in rng.choice(
        np.arange(1, width), size=count, replace=False
    ))


def _height(rng: np.random.Generator, count: int) -> int:
    minimum = count + 1
    band = int(rng.choice(4, p=[0.25, 0.36, 0.24, 0.15]))
    if band == 0:
        extra = 0
    elif band == 1:
        extra = int(rng.integers(1, 4))
    elif band == 2:
        extra = int(rng.integers(4, 9))
    else:
        extra = int(rng.integers(9, min(18, 31 - minimum)))
    return minimum + extra


def _output(height: int, width: int, background: int, baseline: int,
            colors: dict[int, int]) -> np.ndarray:
    result = np.full((height, width), background, dtype=np.int8)
    result[-1, :] = baseline
    for rank, length in enumerate(sorted(colors, reverse=True)):
        result[height - 2 - rank, width - length:] = colors[length]
    return result


def _input_rows(rng: np.random.Generator, lengths: list[int], height: int,
                shared_row: bool, pattern: str) -> dict[int, int]:
    ordered = sorted(lengths)
    if pattern in ("final_rows", "near_finished"):
        rows = {length: height - 2 - rank
                for rank, length in enumerate(sorted(lengths, reverse=True))}
        if shared_row:
            rows[ordered[1]] = rows[ordered[0]]
        return rows
    occupied = sorted(int(x) for x in rng.choice(
        height - 1, size=len(lengths) - int(shared_row), replace=False
    ))
    if pattern == "ascending":
        order = ordered
    elif pattern == "descending":
        order = ordered[::-1]
    else:
        order = [int(x) for x in rng.permutation(ordered)]
    if shared_row:
        # The shortest two bars are separate runs on the same row.
        order = [length for length in order if length != ordered[1]]
    rows = {length: row for length, row in zip(order, occupied)}
    if shared_row:
        rows[ordered[1]] = rows[ordered[0]]
    return rows


_PATTERNS = ("scattered", "ascending", "descending", "left_edge",
             "right_edge", "final_rows", "near_finished")


def _make_pair(rng: np.random.Generator, width: int, background: int,
               baseline: int, complete: bool, shared_row: bool,
               reused_color: bool, pattern: str) -> tuple[np.ndarray, np.ndarray]:
    min_bars = 4 if shared_row or reused_color else 2
    lengths = _bar_lengths(rng, width, complete, min_bars, shared_row)
    height = _height(rng, len(lengths))
    rows = _input_rows(rng, lengths, height, shared_row, pattern)
    starts: dict[int, int] = {}
    for length in lengths:
        last_start = width - length
        if pattern == "final_rows":
            starts[length] = int(rng.integers(0, last_start))
        elif pattern == "near_finished":
            starts[length] = last_start
        elif pattern == "left_edge":
            starts[length] = 0 if rng.random() < 0.85 else int(rng.integers(last_start + 1))
        elif pattern == "right_edge":
            starts[length] = last_start if rng.random() < 0.85 else int(rng.integers(last_start + 1))
        else:
            starts[length] = int(rng.integers(last_start + 1))
    if pattern == "near_finished":
        moved_length = lengths[int(rng.integers(len(lengths)))]
        starts[moved_length] = int(rng.integers(0, width - moved_length))
    if shared_row:
        first, second = lengths[:2]
        gap = int(rng.integers(1, min(3, width - first - second) + 1))
        left = int(rng.integers(0, width - first - second - gap + 1))
        if rng.random() < 0.5:
            starts[first], starts[second] = left, left + first + gap
        else:
            starts[second], starts[first] = left, left + second + gap

    available = [color for color in range(10) if color not in (background, baseline)]
    distinct = rng.choice(available, size=len(lengths), replace=False)
    colors = {length: int(color) for length, color in zip(lengths, distinct)}
    if reused_color:
        candidates = [(a, b) for a in lengths for b in lengths
                      if a < b and rows[a] != rows[b]
                      and abs(rows[a] - rows[b]) >= 2
                      and abs(lengths.index(a) - lengths.index(b)) >= 2]
        if not candidates:
            candidates = [(a, b) for a in lengths for b in lengths
                          if a < b and rows[a] != rows[b]]
        assert candidates
        a, b = candidates[int(rng.integers(len(candidates)))]
        colors[b] = colors[a]

    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[-1, :] = baseline
    for length in lengths:
        row, start = rows[length], starts[length]
        assert np.all(input_grid[row, start:start + length] == background)
        input_grid[row, start:start + length] = colors[length]
    result = _output(height, width, background, baseline, colors)
    if np.array_equal(input_grid, result):
        length = lengths[0]
        row, start = rows[length], starts[length]
        input_grid[row, start:start + length] = background
        input_grid[row, 0:length] = colors[length]
    assert not np.array_equal(input_grid, result)
    return input_grid, result


def _read_bars(input_grid: np.ndarray, background: int, baseline: int) -> dict:
    height, width = input_grid.shape
    assert 3 <= width <= 18 and height >= 3
    assert background != baseline
    assert np.all(input_grid[-1, :] == baseline)
    assert not np.any(input_grid[:-1] == baseline)
    bars: dict[int, tuple[int, int, int]] = {}
    repeated: dict[int, list[int]] = {}
    shared_rows = 0
    for row in range(height - 1):
        assert np.any(input_grid[row] == background)
        runs = []
        col = 0
        while col < width:
            color = int(input_grid[row, col])
            if color == background:
                col += 1
                continue
            start = col
            while col < width and int(input_grid[row, col]) == color:
                col += 1
            assert col == width or int(input_grid[row, col]) == background
            length = col - start
            assert 1 <= length < width and length not in bars
            bars[length] = color, row, start
            repeated.setdefault(color, []).append(row)
            runs.append((start, length, color))
        assert len(runs) <= 2
        if len(runs) == 2:
            shared_rows += 1
            assert runs[0][2] != runs[1][2]
            assert runs[1][0] >= runs[0][0] + runs[0][1] + 1
    assert 2 <= len(bars) <= 8
    assert height >= len(bars) + 1
    assert shared_rows <= 1
    repeated_colors = [color for color, rows in repeated.items() if len(rows) > 1]
    assert len(repeated_colors) <= 1
    if repeated_colors:
        rows = repeated[repeated_colors[0]]
        assert len(rows) == 2 and rows[0] != rows[1]
    return {"bars": bars, "shared": bool(shared_rows),
            "reuse": bool(repeated_colors), "sparse": len(bars) < width - 1}


def _predict_from_training(train_pairs: list[dict], input_grid: np.ndarray) -> np.ndarray:
    # Only demonstrations and this input supply the color roles and answer.
    background = int(train_pairs[0]["output"][0, 0])
    baseline = int(train_pairs[0]["output"][-1, 0])
    for pair in train_pairs:
        assert int(pair["output"][0, 0]) == background
        assert np.all(pair["output"][-1, :] == baseline)
    bars = _read_bars(input_grid, background, baseline)["bars"]
    return _output(input_grid.shape[0], input_grid.shape[1], background,
                   baseline, {length: item[0] for length, item in bars.items()})


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed_code = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed_code)
    train_count, test_count = _counts(seed_code, num_train, num_test)
    roles = rng.choice(np.arange(10), size=2, replace=False)
    background, baseline = (int(x) for x in roles)
    train_widths, test_widths = _width_schedule(
        rng, train_count, test_count, seed_code % 5
    )
    extension = (seed_code // 5) % 4
    train: list[dict] = []
    test: list[dict] = []
    seen_inputs: set[tuple] = set()
    for split, widths in (("train", train_widths), ("test", test_widths)):
        for index, width in enumerate(widths):
            if split == "train" and index == 0:
                complete, shared, reuse, pattern = True, False, False, "scattered"
            elif split == "train" and index == 1:
                complete = False
                shared, reuse = bool(extension & 1), bool(extension & 2)
                pattern = _PATTERNS[int(rng.integers(len(_PATTERNS)))]
            else:
                complete = width <= 9 and rng.random() < 0.43
                if split == "test" and index == 0:
                    shared, reuse = bool(extension & 1), bool(extension & 2)
                else:
                    shared = bool(extension & 1) and rng.random() < 0.38
                    reuse = bool(extension & 2) and rng.random() < 0.38
                pattern = _PATTERNS[int(rng.integers(len(_PATTERNS)))]
            if width == 3:
                complete, shared, reuse = True, False, False
            elif width < 6:
                shared, reuse = False, False
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            for attempt in range(128):
                input_grid, output_grid = _make_pair(
                    pair_rng, width, background, baseline, complete,
                    shared, reuse, pattern
                )
                key = _grid_key(input_grid)
                if key in seen_inputs or _pair_key(input_grid, output_grid) in _SOURCE_PAIR_KEYS:
                    continue
                seen_inputs.add(key)
                (train if split == "train" else test).append(
                    {"input": input_grid, "output": output_grid}
                )
                break
            else:
                raise RuntimeError(f"could not create unique {split} pair {index}")
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[part], list) for part in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    background = int(puzzle["train"][0]["output"][0, 0])
    baseline = int(puzzle["train"][0]["output"][-1, 0])
    assert background != baseline
    seen_inputs, seen_pairs = set(), set()
    train_features, test_features = [], []
    train_widths = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert int(out[0, 0]) == background
            assert np.all(out[-1, :] == baseline)
            input_key, pair_key = _grid_key(inp), _pair_key(inp, out)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            assert pair_key not in _SOURCE_PAIR_KEYS
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            features = _read_bars(inp, background, baseline)
            assert np.array_equal(out, _predict_from_training(puzzle["train"], inp))
            assert np.array_equal(np.bincount(inp.flat, minlength=10),
                                  np.bincount(out.flat, minlength=10))
            (train_features if split == "train" else test_features).append(features)
            if split == "train":
                train_widths.add(inp.shape[1])
    assert len(train_widths) >= 2
    assert not train_features[0]["sparse"]
    assert not train_features[0]["shared"] and not train_features[0]["reuse"]
    assert train_features[1]["sparse"]
    for property_name in ("sparse", "shared", "reuse"):
        if any(f[property_name] for f in test_features):
            assert any(f[property_name] for f in train_features)


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple(_pair_key(p["input"], p["output"]) for p in puzzle[split])
                 for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) >= 40
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    widths, heights, layouts, roles = set(), set(), set(), set()
    counts = {name: 0 for name in ("complete", "sparse", "shared", "reuse",
                                  "both", "tight", "tall", "one_move",
                                  "final_rows_shifted", "ascending",
                                  "descending", "mostly_left", "mostly_right",
                                  "sparse_shared_reuse", "source_like",
                                  "vary_bar_count")}
    shifts = {"seen": 0, "smaller": 0, "larger": 0}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        roles.add((int(puzzle["train"][0]["output"][0, 0]),
                   int(puzzle["train"][0]["output"][-1, 0])))
        train_widths = {p["input"].shape[1] for p in puzzle["train"]}
        counts["source_like"] += int(
            3 in train_widths and 7 in train_widths
            and all(p["input"].shape[1] == 8 for p in puzzle["test"])
        )
        bar_counts_in_puzzle = set()
        for pair in puzzle["test"]:
            width = pair["input"].shape[1]
            shifts["seen" if width in train_widths else
                   "smaller" if width < min(train_widths) else "larger"] += 1
            assert np.array_equal(pair["output"],
                                  _predict_from_training(puzzle["train"], pair["input"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                bg = int(puzzle["train"][0]["output"][0, 0])
                base = int(puzzle["train"][0]["output"][-1, 0])
                analysis = _read_bars(inp, bg, base)
                bars = analysis["bars"]
                bar_counts_in_puzzle.add(len(bars))
                widths.add(inp.shape[1])
                heights.add(inp.shape[0])
                layouts.add((inp.shape, tuple(sorted(
                    (length, row, start) for length, (_, row, start) in bars.items()
                ))))
                counts["sparse" if analysis["sparse"] else "complete"] += 1
                counts["shared"] += int(analysis["shared"])
                counts["reuse"] += int(analysis["reuse"])
                counts["both"] += int(analysis["shared"] and analysis["reuse"])
                counts["sparse_shared_reuse"] += int(
                    analysis["sparse"] and analysis["shared"] and analysis["reuse"]
                )
                if len(bars) >= 4:
                    by_row = sorted((row, length, start)
                                    for length, (_, row, start) in bars.items())
                    if not analysis["shared"]:
                        ordered_lengths = [length for _, length, _ in by_row]
                        counts["ascending"] += int(
                            ordered_lengths == sorted(ordered_lengths)
                        )
                        counts["descending"] += int(
                            ordered_lengths == sorted(ordered_lengths, reverse=True)
                        )
                    counts["mostly_left"] += int(
                        4 * sum(start == 0 for _, _, start in by_row) >= 3 * len(bars)
                    )
                    counts["mostly_right"] += int(
                        4 * sum(start == inp.shape[1] - length
                                for _, length, start in by_row) >= 3 * len(bars)
                    )
                counts["tight"] += int(inp.shape[0] == len(bars) + 1)
                counts["tall"] += int(inp.shape[0] >= len(bars) + 10)
                moved = sum(
                    row != inp.shape[0] - 2 - rank or start != inp.shape[1] - length
                    for rank, (length, (_, row, start)) in enumerate(
                        sorted(bars.items(), reverse=True)
                    )
                )
                counts["one_move"] += int(moved == 1)
                counts["final_rows_shifted"] += int(
                    all(row == inp.shape[0] - 2 - rank for rank, (_, (_, row, _))
                        in enumerate(sorted(bars.items(), reverse=True)))
                    and moved > 0
                )
                assert _pair_key(inp, pair["output"]) not in _SOURCE_PAIR_KEYS
        counts["vary_bar_count"] += int(len(bar_counts_in_puzzle) > 1)
    assert len(widths) >= 12 and len(heights) >= 18
    assert len(layouts) >= 150 and len(roles) >= 15
    assert all(value >= 4 for value in shifts.values())
    assert counts["complete"] >= 42 and counts["sparse"] >= 42
    assert counts["shared"] >= 5 and counts["reuse"] >= 5 and counts["both"] >= 2
    assert counts["tight"] >= 10 and counts["tall"] >= 5
    assert counts["one_move"] >= 3 and counts["final_rows_shifted"] >= 3
    assert counts["ascending"] >= 5 and counts["descending"] >= 5
    assert counts["mostly_left"] >= 5 and counts["mostly_right"] >= 5
    assert counts["sparse_shared_reuse"] >= 2
    assert counts["source_like"] >= 5 and counts["vary_bar_count"] >= 35
    for seed, train, test in ((8100, 2, 1), (8101, 10, 1),
                              (8102, 10, 4), (8103, 6, 3)):
        puzzle = generate_puzzle(seed, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for seed, train, test in ((8200, None, 1), (8201, None, 4),
                              (8202, 2, None), (8203, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    assert _puzzle_key(generate_puzzle(8300, 7, 3)) == _puzzle_key(
        generate_puzzle(8300, 7, 3))
    assert _puzzle_key(generate_puzzle(8301)) == _puzzle_key(generate_puzzle(8301))
    assert len({_puzzle_key(generate_puzzle(8400 + i, 5, 2))
                for i in range(10)}) >= 9
    for request in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                    {"num_test": 5}, {"num_train": 2, "num_test": 2},
                    {"num_train": 2, "num_test": 3}):
        try:
            generate_puzzle(8500, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted illegal count request: {request}")
