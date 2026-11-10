"""Generate uniform-canvas ARC lattice puzzles."""

import numpy as np


_SOURCE_SIDES = frozenset((5, 7, 9, 11))
_BANDS = tuple((line, gap) for line in range(1, 4) for gap in range(1, 4))


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        minimum = max(2, test + 1)
        train = minimum + seed % (11 - minimum)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _mask(height: int, width: int, row: tuple[int, int], col: tuple[int, int]) -> np.ndarray:
    rows = (np.arange(height) % sum(row)) < row[0]
    cols = (np.arange(width) % sum(col)) < col[0]
    return rows[:, None] | cols[None, :]


def _anchor_visible(length: int, band: tuple[int, int]) -> bool:
    line_cells = (np.arange(length) % sum(band)) < band[0]
    return not bool(np.array_equal(line_cells, line_cells[::-1]))


def _build_pair(height: int, width: int, background: int, foreground: int,
                row: tuple[int, int], col: tuple[int, int]) -> dict:
    input_grid = np.full((height, width), background, dtype=np.int8)
    output_grid = input_grid.copy()
    output_grid[_mask(height, width, row, col)] = foreground
    return {"input": input_grid, "output": output_grid}


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, grid.tobytes())


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple((split, tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                               for p in puzzle[split])) for split in ("train", "test"))


def _source_copy(height: int, width: int, background: int, foreground: int,
                 row: tuple[int, int], col: tuple[int, int]) -> bool:
    return (height == width and height in _SOURCE_SIDES and background == 0
            and foreground == 1 and row == (1, 1) and col == (1, 1))


def _exact_source_pair(inp: np.ndarray, out: np.ndarray) -> bool:
    height, width = inp.shape
    if height != width or height not in _SOURCE_SIDES or not bool(np.all(inp == 0)):
        return False
    source_output = np.where(_mask(height, width, (1, 1), (1, 1)), 1, 0)
    return bool(np.array_equal(out, source_output))


def _shape(rng: np.random.Generator, minimum: int, maximum: int,
           style: int, used: set, background: int, foreground: int,
           row: tuple[int, int], col: tuple[int, int]) -> tuple[int, int]:
    # Bounded sampling with a deterministic fallback.
    for _ in range(80):
        if style == 1 and minimum <= 12:
            height = int(rng.integers(minimum, min(maximum, 12) + 1))
            width = int(rng.integers(max(16, minimum), maximum + 1))
        elif style == 2 and minimum <= 12:
            height = int(rng.integers(max(16, minimum), maximum + 1))
            width = int(rng.integers(minimum, min(maximum, 12) + 1))
        else:
            height = int(rng.integers(minimum, maximum + 1))
            width = int(rng.integers(minimum, maximum + 1))
        if ((height, width) not in used
                and not _source_copy(height, width, background, foreground, row, col)):
            return height, width
    for height in range(minimum, maximum + 1):
        for width in range(minimum, maximum + 1):
            if ((height, width) not in used
                    and not _source_copy(height, width, background, foreground, row, col)):
                return height, width
    raise AssertionError("shape pool exhausted")


def _candidate_conventions(train: list) -> tuple[int, list]:
    foreground = None
    for pair in train:
        background = int(pair["input"][0, 0])
        introduced = set(int(c) for c in np.unique(pair["output"])) - {background}
        _require(len(introduced) == 1, "training must reveal one foreground")
        local = introduced.pop()
        if foreground is None:
            foreground = local
        _require(local == foreground, "foreground changes between demonstrations")
    candidates = []
    for row in _BANDS:
        for col in _BANDS:
            if all(np.array_equal(
                    pair["output"],
                    np.where(_mask(*pair["input"].shape, row, col), foreground,
                             int(pair["input"][0, 0])))
                   for pair in train):
                candidates.append((row, col))
    return foreground, candidates


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must contain exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    backgrounds = set()
    even_height = even_width = False
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            location = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     f"{location} must contain input and output")
            inp, out = pair["input"], pair["output"]
            for name, grid in (("input", inp), ("output", out)):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         f"{location}.{name} is not a 2D NumPy array")
                height, width = grid.shape
                _require(1 <= height <= 30 and 1 <= width <= 30,
                         f"{location}.{name} size is invalid")
                _require(np.issubdtype(grid.dtype, np.integer),
                         f"{location}.{name} must contain integers")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         f"{location}.{name} contains an illegal color")
            height, width = inp.shape
            _require(out.shape == inp.shape and min(height, width) >= 3,
                     f"{location} must preserve dimensions and show the motif")
            _require(bool(np.all(inp == inp[0, 0])), f"{location} input is not uniform")
            background = int(inp[0, 0])
            _require(background in set(int(c) for c in np.unique(out)),
                     f"{location} loses its input background")
            _require(not np.array_equal(inp, out), f"{location} is unchanged")
            inp_key = _grid_key(inp)
            pair_key = (inp_key, _grid_key(out))
            _require(inp_key not in seen_inputs and pair_key not in seen_pairs,
                     "duplicate input or pair")
            seen_inputs.add(inp_key)
            seen_pairs.add(pair_key)
            if split == "train":
                backgrounds.add(background)
                even_height |= height % 2 == 0
                even_width |= width % 2 == 0
    _require(len(backgrounds) >= 2, "training must show changing backgrounds")
    _require(even_height and even_width,
             "training must expose top and left phase with even dimensions")
    foreground, candidates = _candidate_conventions(train)
    _require(len(candidates) == 1, "training does not uniquely reveal the lattice")
    row, col = candidates[0]
    _require(any(_anchor_visible(p["input"].shape[0], row) for p in train)
             and any(_anchor_visible(p["input"].shape[1], col) for p in train),
             "training does not distinguish the top and left anchors")
    _require(any(p["input"].shape[0] >= 2 * sum(row) + 1 and
                 p["input"].shape[1] >= 2 * sum(col) + 1 for p in train),
             "training does not show enough repeats")
    for split in ("train", "test"):
        for pair in puzzle[split]:
            inp, out = pair["input"], pair["output"]
            height, width = inp.shape
            background = int(inp[0, 0])
            _require(background != foreground, "input background equals foreground")
            expected = np.where(_mask(height, width, row, col), foreground, background)
            _require(np.array_equal(out, expected), "pair violates the learned lattice")
            _require(not _exact_source_pair(inp, out),
                     "pair exactly duplicates a source pair")
            _require(bool(np.any(out == background)), "pair has no lattice holes")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if int(seed) < 0:
        raise ValueError("seed must be non-negative")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(int(seed), num_train, num_test)
    foreground = int(rng.integers(0, 10))
    if int(rng.integers(0, 5)) < 3:
        row = col = (1, 1)
    else:
        row = _BANDS[int(rng.integers(0, len(_BANDS)))]
        col = _BANDS[int(rng.integers(0, len(_BANDS)))]
        if row == col == (1, 1):
            col = (1, 2)
    rich = row != (1, 1) or col != (1, 1)
    backgrounds = [color for color in range(10) if color != foreground]
    rng.shuffle(backgrounds)
    train_backgrounds = backgrounds[:2] + [int(rng.choice(backgrounds))
                                             for _ in range(train_count - 2)]
    test_backgrounds = [int(rng.choice(backgrounds)) for _ in range(test_count)]
    used = set()
    train_shapes = []
    anchor_low = 18 if rich else 8
    anchor_high = 22 if rich else 20
    heights = [n for n in range(anchor_low, anchor_high + 1, 2)
               if _anchor_visible(n, row)]
    widths = [n for n in range(anchor_low, anchor_high + 1, 2)
              if _anchor_visible(n, col)]
    height = int(rng.choice(heights))
    width = int(rng.choice(widths))
    train_shapes.append((height, width))
    used.add((height, width))
    if not rich and int(rng.integers(0, 5)) == 0:
        second = (3, 3)
    else:
        second = _shape(rng, 17 if rich else 6, 23, int(rng.integers(0, 3)),
                        used, train_backgrounds[1], foreground, row, col)
    train_shapes.append(second)
    used.add(second)
    for index in range(2, train_count):
        shape = _shape(rng, 10 if rich else 3, 23, int(rng.integers(0, 3)),
                       used, train_backgrounds[index], foreground, row, col)
        train_shapes.append(shape)
        used.add(shape)
    test_shapes = []
    for index in range(test_count):
        shape = _shape(rng, 6 if rich else 3, 30, int(rng.integers(0, 3)),
                       used, test_backgrounds[index], foreground, row, col)
        test_shapes.append(shape)
        used.add(shape)
    training = [_build_pair(*shape, background, foreground, row, col)
                for shape, background in zip(train_shapes, train_backgrounds)]
    rng.shuffle(training)
    puzzle = {"train": training,
              "test": [_build_pair(*shape, background, foreground, row, col)
                       for shape, background in zip(test_shapes, test_backgrounds)]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _expect_value_error(function) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    measures = {
        "exact": 0, "rich": 0, "small": 0, "reaches_30": 0,
        "wide": 0, "tall": 0, "mixed_parity": 0, "nonmonotone": 0,
        "smaller_test": 0, "interpolating_test": 0, "larger_test": 0,
        "large_jump": 0, "source_palette": 0, "rich_mixed": 0,
    }
    foregrounds, backgrounds, shapes = set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        foreground, conventions = _candidate_conventions(puzzle["train"])
        _require(len(conventions) == 1, "self-test could not infer the rule")
        row, col = conventions[0]
        foregrounds.add(foreground)
        measures["exact" if row == col == (1, 1) else "rich"] += 1
        train_shapes = [p["input"].shape for p in puzzle["train"]]
        test_shapes = [p["input"].shape for p in puzzle["test"]]
        all_shapes = train_shapes + test_shapes
        measures["small"] += (3, 3) in train_shapes
        measures["reaches_30"] += any(30 in shape for shape in test_shapes)
        measures["wide"] += any(w > 1.5 * h for h, w in all_shapes)
        measures["tall"] += any(h > 1.5 * w for h, w in all_shapes)
        measures["mixed_parity"] += any(h % 2 != w % 2 for h, w in all_shapes)
        measures["nonmonotone"] += any(
            train_shapes[i][0] > train_shapes[i + 1][0] or
            train_shapes[i][1] > train_shapes[i + 1][1]
            for i in range(len(train_shapes) - 1))
        train_areas = [h * w for h, w in train_shapes]
        test_areas = [h * w for h, w in test_shapes]
        measures["smaller_test"] += any(area < min(train_areas) for area in test_areas)
        measures["interpolating_test"] += any(
            min(train_areas) < area < max(train_areas) for area in test_areas)
        measures["larger_test"] += any(area > max(train_areas) for area in test_areas)
        measures["large_jump"] += any(
            h >= max(th for th, _ in train_shapes) + 5 or
            w >= max(tw for _, tw in train_shapes) + 5
            for h, w in test_shapes)
        measures["rich_mixed"] += (
            (row, col) != ((1, 1), (1, 1)) and
            any(h % 2 != w % 2 for h, w in all_shapes))
        for pair in puzzle["train"] + puzzle["test"]:
            shape = pair["input"].shape
            background = int(pair["input"][0, 0])
            shapes.add(shape)
            backgrounds.add(background)
            measures["source_palette"] += (background, foreground) == (0, 1)
        for pair in puzzle["test"]:
            inp = pair["input"].copy()
            predicted = np.full(inp.shape, int(inp[0, 0]), dtype=np.int8)
            predicted[_mask(*inp.shape, row, col)] = foreground
            _require(np.array_equal(predicted, pair["output"]),
                     "training-only prediction failed")
    for seed, train_count, test_count in ((71, 2, 1), (72, 10, 1), (73, 10, 4),
                                          (74, None, 1), (75, None, 4),
                                          (76, 2, None), (77, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            _require(len(puzzle["train"]) == train_count, "train count ignored")
        if test_count is not None:
            _require(len(puzzle["test"]) == test_count, "test count ignored")
    for seed, counts in ((80, (None, None)), (81, (6, 2)), (82, (10, 4))):
        _require(_puzzle_key(generate_puzzle(seed, *counts)) ==
                 _puzzle_key(generate_puzzle(seed, *counts)), "nondeterministic")
    for call in (lambda: generate_puzzle(1, 1), lambda: generate_puzzle(2, 11),
                 lambda: generate_puzzle(3, num_test=0),
                 lambda: generate_puzzle(4, num_test=5),
                 lambda: generate_puzzle(5, 2, 2),
                 lambda: generate_puzzle(6, 3, 4)):
        _expect_value_error(call)
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts too narrow")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts too narrow")
    _require(len({_puzzle_key(p) for p in puzzles}) == 42, "duplicate puzzles")
    _require(len({_candidate_conventions(p["train"])[1][0] for p in puzzles}) >= 8,
             "band patterns too narrow")
    _require(len(shapes) >= 100, "dimension combinations too narrow")
    _require(len(foregrounds) == 10 and len(backgrounds) == 10,
             "semantic color roles too narrow")
    for name, minimum in (("exact", 15), ("rich", 10), ("small", 3),
                          ("reaches_30", 3), ("wide", 20), ("tall", 20),
                          ("mixed_parity", 20), ("nonmonotone", 25),
                          ("smaller_test", 5), ("interpolating_test", 10),
                          ("larger_test", 5), ("large_jump", 5),
                          ("source_palette", 1), ("rich_mixed", 5)):
        _require(measures[name] >= minimum, f"insufficient {name} coverage")
