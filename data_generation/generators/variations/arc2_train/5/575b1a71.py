import numpy as np


_SOURCE_PAIR_ROWS = {
    (
        (
            "0555555055",
            "5555555555",
            "5555055555",
            "5555555055",
            "5555055555",
            "5555555555",
            "5555555555",
            "5555055555",
            "5555555505",
            "5555055555",
        ),
        (
            "1555555355",
            "5555555555",
            "5555255555",
            "5555555355",
            "5555255555",
            "5555555555",
            "5555555555",
            "5555255555",
            "5555555545",
            "5555255555",
        ),
    ),
    (
        (
            "5555505555",
            "5555555555",
            "5555555555",
            "5055505555",
            "5555005555",
            "5055055555",
            "5055055555",
            "5555555505",
            "5055005505",
            "5555555505",
        ),
        (
            "5555535555",
            "5555555555",
            "5555555555",
            "5155535555",
            "5555235555",
            "5155255555",
            "5155255555",
            "5555555545",
            "5155235545",
            "5555555545",
        ),
    ),
    (
        (
            "5555500555",
            "5555555555",
            "5555555555",
            "5555555550",
            "5555555550",
            "5555555500",
            "5555555555",
            "5555550555",
            "5555555550",
            "5555555555",
        ),
        (
            "5555512555",
            "5555555555",
            "5555555555",
            "5555555554",
            "5555555554",
            "5555555534",
            "5555555555",
            "5555552555",
            "5555555554",
            "5555555555",
        ),
    ),
    (
        (
            "5550555555",
            "5555555555",
            "5555555555",
            "0555555555",
            "5555505555",
            "5505555555",
            "5505555555",
            "0555555555",
            "5555555555",
            "0555505555",
        ),
        (
            "5553555555",
            "5555555555",
            "5555555555",
            "1555555555",
            "5555545555",
            "5525555555",
            "5525555555",
            "1555555555",
            "5555555555",
            "1555545555",
        ),
    ),
}


def _check(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    return tuple(grid.shape), tuple(int(x) for x in grid.ravel())


def _rows_key(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(x)) for x in row) for row in grid)


def _pair_rows_key(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple:
    return _rows_key(input_grid), _rows_key(output_grid)


def _optional_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _optional_count("num_train", num_train, 2, 10)
    test = _optional_count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _sample_shape(rng):
    band = int(rng.choice(4, p=[0.20, 0.22, 0.35, 0.23]))
    if band == 0:
        return 10, 10
    if band == 1:
        width = int(rng.integers(4, 10))
        height = int(rng.integers(8 if width <= 5 else 5, 17))
    elif band == 2:
        height = int(rng.integers(7, 21))
        width = int(rng.integers(8, 21))
    else:
        if rng.random() < 0.5:
            height = int(rng.integers(21, 31))
            width = int(rng.integers(7, 31))
        else:
            height = int(rng.integers(7, 31))
            width = int(rng.integers(21, 31))
    return height, width


def _column_candidates(width, region):
    if region == "left":
        return list(range(max(4, (2 * width + 2) // 3)))
    if region == "right":
        span = max(4, (2 * width + 2) // 3)
        return list(range(width - span, width))
    if region == "middle":
        margin = max(1, width // 6) if width >= 8 else 0
        candidates = list(range(margin, width - margin))
        return candidates if len(candidates) >= 4 else list(range(width))
    return list(range(width))


def _sample_columns(rng, width, region, contiguous):
    if region == "edge":
        if width == 4:
            return (0, 1, 2, 3)
        middle = rng.choice(np.arange(1, width - 1), size=2, replace=False)
        return (0, *sorted(int(x) for x in middle), width - 1)
    candidates = _column_candidates(width, region)
    if contiguous and len(candidates) >= 4:
        first = int(rng.integers(candidates[0], candidates[-1] - 2))
        return tuple(range(first, first + 4))
    chosen = rng.choice(candidates, size=4, replace=False)
    return tuple(sorted(int(x) for x in chosen))


def _sample_counts(rng, height, width, kind):
    cap = max(4, (height * width) // 4)
    maximum = min(height, max(2, (height + 1) // 2), cap - 3)
    if kind == "singletons":
        return [1, 1, 1, 1]
    if kind == "equal":
        value = int(rng.integers(1, min(maximum, cap // 4) + 1))
        return [value] * 4
    if kind == "repeated":
        if cap < 8:
            return [1, 1, 1, 1]
        counts = [2, 2, 2, 2]
    elif kind == "dominant":
        counts = [1, 1, 1, 1]
        rank = int(rng.integers(0, 4))
        counts[rank] = min(maximum, cap - 3, int(rng.integers(3, maximum + 1)))
    else:
        counts = [int(x) for x in rng.integers(1, maximum + 1, size=4)]
    while sum(counts) > cap:
        options = [i for i, n in enumerate(counts) if n > (2 if kind == "repeated" else 1)]
        if not options:
            break
        counts[int(rng.choice(options))] -= 1
    if kind == "mixed" and len(set(counts)) == 1:
        if sum(counts) < cap and counts[0] < maximum:
            counts[int(rng.integers(0, 4))] += 1
        elif counts[0] > 1:
            counts[int(rng.integers(0, 4))] -= 1
    return counts


def _sample_rows(rng, height, counts, arrangement):
    rows = [set() for _ in range(4)]
    if arrangement == "separated" and sum(counts) <= height:
        shuffled = [int(x) for x in rng.permutation(height)]
        cursor = 0
        for rank, count in enumerate(counts):
            rows[rank].update(shuffled[cursor:cursor + count])
            cursor += count
        return rows
    if arrangement == "vertical":
        for rank, count in enumerate(counts):
            start = int(rng.integers(0, height - count + 1))
            rows[rank].update(range(start, start + count))
        return rows
    if arrangement in ("horizontal", "block"):
        shared = int(rng.integers(0, height))
        for rank in range(4):
            rows[rank].add(shared)
        if arrangement == "block":
            rank = int(rng.integers(0, 4))
            run = min(counts[rank], max(2, height // 3))
            start = min(shared, height - run)
            rows[rank].update(range(start, start + run))
    elif arrangement == "staircase":
        first = int(rng.integers(0, height - 3))
        for rank in range(4):
            rows[rank].add(first + rank)
    for rank, count in enumerate(counts):
        remaining = [r for r in range(height) if r not in rows[rank]]
        extra = count - len(rows[rank])
        if extra:
            rows[rank].update(int(x) for x in rng.choice(remaining, size=extra, replace=False))
    return rows


def _make_pair(rng, shape, region, convention, count_kind=None):
    height, width = shape
    arrangement = str(rng.choice(
        ["scattered", "horizontal", "staircase", "vertical", "separated", "block"],
        p=[0.25, 0.18, 0.14, 0.17, 0.11, 0.15],
    ))
    if count_kind is None:
        count_kind = str(rng.choice(
            ["mixed", "equal", "singletons", "repeated", "dominant"],
            p=[0.35, 0.22, 0.10, 0.19, 0.14],
        ))
    contiguous = arrangement in ("horizontal", "staircase", "block") and rng.random() < 0.74
    columns = _sample_columns(rng, width, region, contiguous)
    counts = _sample_counts(rng, height, width, count_kind)
    row_sets = _sample_rows(rng, height, counts, arrangement)
    background, marker, rank_colors = convention
    inp = np.full(shape, background, dtype=np.int8)
    out = inp.copy()
    for rank, column in enumerate(columns):
        for row in row_sets[rank]:
            inp[row, column] = marker
            out[row, column] = rank_colors[rank]
    return inp, out


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    colors = [int(x) for x in rng.choice(10, size=6, replace=False)]
    convention = (colors[0], colors[1], tuple(colors[2:]))
    size_mode = str(rng.choice(["fixed", "variable", "mostly_fixed"], p=[0.38, 0.42, 0.20]))
    fixed_shape = _sample_shape(rng)
    shift = str(rng.choice(["right_left", "left_right", "middle", "edge", "mixed"]))
    paired_probe = rng.random() < 0.35
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            if size_mode == "fixed" or (size_mode == "mostly_fixed" and rng.random() < 0.75):
                shape = fixed_shape
            else:
                shape = _sample_shape(rng)
            if shift == "right_left":
                region = "right" if split == "train" else "left"
            elif shift == "left_right":
                region = "left" if split == "train" else "right"
            elif shift in ("middle", "edge"):
                region = shift
            else:
                region = str(rng.choice(["left", "right", "middle", "edge", "free"]))
            forced_count = None
            if split == "train" and index == 0:
                forced_count = "mixed"
            elif paired_probe and split == "test" and index == 0:
                forced_count = "equal"
            for _ in range(200):
                inp, out = _make_pair(rng, shape, region, convention, forced_count)
                key = _grid_key(inp)
                if key not in used_inputs and _pair_rows_key(inp, out) not in _SOURCE_PAIR_ROWS:
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
            used_inputs.add(key)
            puzzle[split].append({"input": inp, "output": out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(inp, out):
    _check(inp.shape == out.shape, "shape changed")
    values, counts = np.unique(inp, return_counts=True)
    _check(len(values) == 2, "input must have exactly two colors")
    background = int(values[int(np.argmax(counts))])
    marker = int(values[int(np.argmin(counts))])
    _check(int(np.max(counts)) > int(np.min(counts)), "background is not dominant")
    mask = inp == marker
    _check(int(np.count_nonzero(mask)) * 4 <= inp.size, "markers are not sparse")
    columns = tuple(int(c) for c in np.flatnonzero(np.any(mask, axis=0)))
    _check(len(columns) == 4, "input needs four occupied columns")
    _check(bool(np.all(out[~mask] == background)), "background changed")
    labels = []
    profile = []
    for column in columns:
        seen = np.unique(out[mask[:, column], column])
        _check(len(seen) == 1, "inconsistent label within column")
        labels.append(int(seen[0]))
        profile.append(int(np.count_nonzero(mask[:, column])))
    _check(len(set(labels)) == 4, "rank labels are not distinct")
    _check(not (set(labels) & {background, marker}), "semantic colors overlap")
    _check(set(int(x) for x in np.unique(out)) == {background, *labels}, "output palette changed")
    return background, marker, tuple(labels), columns, tuple(profile)


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _check(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "pair counts")
    seen_inputs = set()
    convention = None
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid type")
                _check(all(1 <= n <= 30 for n in grid.shape), "grid size")
                _check(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                _check(bool(np.all((0 <= grid) & (grid <= 9))), "grid colors")
            key = _grid_key(inp)
            _check(key not in seen_inputs, "duplicate input")
            seen_inputs.add(key)
            _check(_pair_rows_key(inp, out) not in _SOURCE_PAIR_ROWS, "source pair copied")
            analysis = _analyze_pair(inp, out)
            learned = analysis[:3]
            if convention is None:
                convention = learned
            _check(learned == convention, "convention changed within puzzle")
            if split == "train":
                _check(set(learned[2]).issubset(set(int(x) for x in np.unique(out))), "train hides a rank")
    background, marker, labels = convention
    for pair in puzzle["test"]:
        inp, out = pair["input"], pair["output"]
        mask = inp == marker
        columns = tuple(int(c) for c in np.flatnonzero(np.any(mask, axis=0)))
        predicted = np.full(inp.shape, background, dtype=out.dtype)
        for rank, column in enumerate(columns):
            predicted[mask[:, column], column] = labels[rank]
        _check(np.array_equal(predicted, out), "test not recoverable from training")


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ("input", "output")
    ) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    for offset, (train, test) in enumerate(((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(1000 + offset, train, test)
        validate_generated_puzzle(puzzle)
        _check(train is None or len(puzzle["train"]) == train, "explicit train count")
        _check(test is None or len(puzzle["test"]) == test, "explicit test count")
        _check(_puzzles_equal(puzzle, generate_puzzle(1000 + offset, train, test)), "not deterministic")
    for request in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (True, 1)):
        try:
            generate_puzzle(2000, *request)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _check(not _puzzles_equal(puzzles[0], puzzles[1]), "seeds failed to vary")
    _check(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack diversity")
    _check(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack diversity")
    palettes, shapes, layouts, profiles = set(), set(), set(), set()
    counts_seen = {"all_singleton": 0, "all_equal": 0, "no_singleton": 0,
                   "horizontal_four": 0, "vertical_three": 0, "nonmonotone": 0}
    fixed_puzzles = 0
    variable_puzzles = 0
    position_regimes = {"right_left": 0, "left_right": 0, "middle": 0, "edge": 0}
    equal_test_after_mixed_train = 0
    narrow_puzzles = 0
    large_puzzles = 0
    fixed_ten_puzzles = 0
    combined_equal_horizontal = 0
    for puzzle in puzzles:
        all_shapes = set()
        split_features = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                background, marker, labels, columns, profile = _analyze_pair(inp, out)
                mask = inp == marker
                expected = np.full(inp.shape, background, dtype=out.dtype)
                for rank, column in enumerate(columns):
                    expected[mask[:, column], column] = labels[rank]
                _check(np.array_equal(expected, out), "incorrect column-rank transform")
                palettes.add((background, marker, labels))
                shapes.add(inp.shape)
                all_shapes.add(inp.shape)
                layouts.add((split, inp.shape, columns))
                profiles.add(profile)
                counts_seen["all_singleton"] += max(profile) == 1
                counts_seen["all_equal"] += len(set(profile)) == 1
                counts_seen["no_singleton"] += min(profile) > 1
                counts_seen["nonmonotone"] += labels != tuple(sorted(labels))
                counts_seen["horizontal_four"] += any(bool(np.all(mask[r, c:c + 4])) for r in range(inp.shape[0]) for c in range(inp.shape[1] - 3))
                counts_seen["vertical_three"] += any(bool(np.all(mask[r:r + 3, c])) for r in range(inp.shape[0] - 2) for c in columns)
                width = inp.shape[1]
                left_span = max(4, (2 * width + 2) // 3)
                middle_margin = max(1, width // 6) if width >= 8 else 0
                split_features[split].append({
                    "width": width,
                    "center": sum(columns) / (4 * (width - 1)),
                    "left": columns[-1] < left_span,
                    "right": columns[0] >= width - left_span,
                    "middle": width >= 8 and columns[0] >= middle_margin and columns[-1] < width - middle_margin,
                    "edge": columns[0] == 0 and columns[-1] == width - 1,
                    "mixed_counts": len(set(profile)) > 1,
                    "equal_counts": len(set(profile)) == 1,
                    "equal_horizontal": len(set(profile)) == 1 and any(bool(np.all(mask[r, c:c + 4])) for r in range(inp.shape[0]) for c in range(width - 3)),
                })
        fixed_puzzles += len(all_shapes) == 1
        variable_puzzles += len(all_shapes) > 1
        fixed_ten_puzzles += all_shapes == {(10, 10)}
        narrow_puzzles += any(width <= 7 for _, width in all_shapes)
        large_puzzles += any(max(shape) >= 21 for shape in all_shapes)
        train = split_features["train"]
        test = split_features["test"]
        wide = all(x["width"] >= 8 for x in train + test)
        center_difference = sum(x["center"] for x in train) / len(train) - sum(x["center"] for x in test) / len(test)
        position_regimes["right_left"] += wide and all(x["right"] for x in train) and all(x["left"] for x in test) and center_difference > 0.12
        position_regimes["left_right"] += wide and all(x["left"] for x in train) and all(x["right"] for x in test) and center_difference < -0.12
        position_regimes["middle"] += all(x["middle"] for x in train + test)
        position_regimes["edge"] += wide and all(x["edge"] for x in train + test)
        equal_test_after_mixed_train += any(x["mixed_counts"] for x in train) and any(x["equal_counts"] for x in test)
        combined_equal_horizontal += any(x["equal_horizontal"] for x in train + test)
    _check(len(palettes) >= 30 and len(shapes) >= 30 and len(layouts) >= 100 and len(profiles) >= 20, "low structural diversity")
    _check(fixed_puzzles >= 4 and variable_puzzles >= 8, "size regimes not varied")
    _check(fixed_ten_puzzles >= 2 and narrow_puzzles >= 5 and large_puzzles >= 8, "size range missing")
    _check(all(value >= 3 for value in position_regimes.values()), "column-position regimes missing")
    _check(equal_test_after_mixed_train >= 8 and combined_equal_horizontal >= 3, "review combinations missing")
    _check(all(counts_seen[k] > threshold for k, threshold in {
        "all_singleton": 2, "all_equal": 10, "no_singleton": 10,
        "horizontal_four": 3, "vertical_three": 5, "nonmonotone": 30,
    }.items()), "review variation missing")
