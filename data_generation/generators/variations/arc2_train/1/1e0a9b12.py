"""Synthetic instances of stable, downward, column-wise ARC gravity."""

import numpy as np


_AUTO_COUNTS = (
    (2, 1), (3, 2), (4, 3), (5, 4), (6, 1), (7, 2),
    (8, 3), (9, 4), (10, 1), (6, 2), (8, 4), (10, 3),
)


def _rows(*lines: str) -> np.ndarray:
    return np.array([[int(c) for c in line] for line in lines], dtype=np.int8)


_SOURCE_PAIRS = (
    (_rows("000009", "000800", "000000", "400000", "407800", "407000"),
     _rows("000000", "000000", "000000", "400000", "407800", "407809")),
    (_rows("0409", "0000", "0460", "1000"),
     _rows("0000", "0000", "0400", "1469")),
    (_rows("00010", "03000", "03012", "60000", "03000"),
     _rows("00000", "00000", "03000", "03010", "63012")),
    (_rows("02043", "50000", "00600", "52040", "50000"),
     _rows("00000", "00000", "50000", "52040", "52643")),
)


def _require(ok: bool, message: str) -> None:
    if not bool(ok):
        raise AssertionError(message)


def _count(value: int | None, name: str, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        return _AUTO_COUNTS[seed % len(_AUTO_COUNTS)]
    if train is None:
        train = max(2, test + 1) + seed % (11 - max(2, test + 1))
    if test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    largest = int(np.argmax(counts))
    _require(np.count_nonzero(counts == counts[largest]) == 1,
             "background is not unambiguous")
    _require(int(counts[largest]) > grid.size // 2,
             "background must occupy a majority of the grid")
    return int(colors[largest])


def _compact(grid: np.ndarray, background: int) -> np.ndarray:
    result = np.full_like(grid, background)
    for col in range(grid.shape[1]):
        sequence = grid[:, col][grid[:, col] != background]
        if sequence.size:
            result[-sequence.size:, col] = sequence
    return result


def _moving_columns(grid: np.ndarray, background: int) -> tuple[int, int]:
    moved = settled = 0
    height = grid.shape[0]
    for col in range(grid.shape[1]):
        positions = np.flatnonzero(grid[:, col] != background)
        if not positions.size:
            continue
        if np.array_equal(positions, np.arange(height - positions.size, height)):
            settled += 1
        else:
            moved += 1
    return moved, settled


def _internal_gap(grid: np.ndarray, background: int) -> bool:
    for col in range(grid.shape[1]):
        positions = np.flatnonzero(grid[:, col] != background)
        if positions.size > 1 and np.any(np.diff(positions) > 1):
            return True
    return False


def _order_example(grid: np.ndarray, background: int) -> bool:
    for col in range(grid.shape[1]):
        sequence = tuple(int(x) for x in grid[:, col] if x != background)
        if (len(sequence) >= 3 and len(set(sequence)) > 1
                and sequence != sequence[::-1]
                and sequence != tuple(sorted(sequence))):
            return True
    return False


def _shared_color_columns(grid: np.ndarray, background: int) -> bool:
    profiles = {}
    for col in range(grid.shape[1]):
        colors = set(int(x) for x in grid[:, col] if x != background)
        for color in colors:
            profiles.setdefault(color, []).append(int(np.count_nonzero(grid[:, col] == color)))
    return any(len(counts) > 1 and len(set(counts)) > 1 for counts in profiles.values())


def _broken_bar(grid: np.ndarray, background: int) -> bool:
    height, width = grid.shape
    for row in range(height):
        for left in range(width - 2):
            color = int(grid[row, left])
            if color == background or not np.all(grid[row, left:left + 3] == color):
                continue
            destinations = []
            for col in range(left, left + 3):
                positions = np.flatnonzero(grid[:, col] != background)
                rank = int(np.count_nonzero(positions < row))
                destinations.append(height - positions.size + rank)
            if len(set(destinations)) > 1:
                return True
    return False


def _choose_shape(rng: np.random.Generator, parameters: dict, regime: str,
                  index: int) -> tuple[int, int]:
    style = parameters["style"]
    if style == "similar":
        base_h, base_w = parameters["base_shape"]
        h = int(np.clip(base_h + rng.integers(-2, 3), 4, 18))
        w = int(np.clip(base_w + rng.integers(-2, 3), 4, 18))
    else:
        bands = ((4, 6), (7, 13), (14, 23), (24, 30))
        weights = ([0.38, 0.40, 0.17, 0.05] if regime == "source_train"
                   else [0.18, 0.37, 0.29, 0.16] if style == "larger_test"
                   else [0.32, 0.36, 0.23, 0.09])
        h_band = bands[int(rng.choice(4, p=weights))]
        w_band = bands[int(rng.choice(4, p=weights))]
        h = int(rng.integers(h_band[0], h_band[1] + 1))
        w = int(rng.integers(w_band[0], w_band[1] + 1))
    if regime == "source_train" and index == 0:
        h, w = max(h, 5), max(w, 5)
    if regime == "source_train" and index == 1:
        h, w = max(h, 6), max(w, 5)
    return h, w


def _palette(rng: np.random.Generator, background: int, count: int,
             required: int | None = None) -> np.ndarray:
    choices = [color for color in range(10) if color != background and color != required]
    if required is None:
        return np.asarray(rng.choice(choices, size=count, replace=False), dtype=np.int8)
    rest = rng.choice(choices, size=count - 1, replace=False)
    return np.asarray([required, *rest], dtype=np.int8)


def _initial_positions(rng: np.random.Generator, height: int, count: int) -> np.ndarray:
    if count == height:
        return np.arange(height)
    kind = int(rng.choice(3, p=[0.27, 0.28, 0.45]))
    if kind == 0:
        return np.arange(height - count, height)
    if kind == 1:
        start = int(rng.integers(0, height - count))
        return np.arange(start, start + count)
    return np.sort(rng.choice(height, size=count, replace=False))


def _base_grid(rng: np.random.Generator, height: int, width: int,
               background: int, palette: np.ndarray, minimum_columns: int) -> np.ndarray:
    grid = np.full((height, width), background, dtype=np.int8)
    fractions = ((0.18, 0.42), (0.42, 0.72), (0.72, 1.0))
    low, high = fractions[int(rng.choice(3, p=[0.30, 0.43, 0.27]))]
    number = int(np.clip(round(width * rng.uniform(low, high)), minimum_columns, width))
    columns = rng.choice(width, size=number, replace=False)
    density = int(rng.choice(3, p=[0.38, 0.43, 0.19]))
    counts = []
    for _ in columns:
        if density == 0:
            count = int(rng.integers(1, min(height, 3) + 1))
        elif density == 1:
            count = int(rng.integers(1, min(height, max(4, height // 2)) + 1))
        else:
            count = int(rng.integers(1, height + 1))
        counts.append(count)
    budget = max(number, int(height * width * 0.40))
    if rng.random() < 0.16 and height + number - 1 <= budget:
        counts[int(rng.integers(number))] = height
    while sum(counts) > budget:
        candidates = [i for i, count in enumerate(counts) if count > 1]
        if not candidates:
            break
        counts[int(rng.choice(candidates))] -= 1
    for col, count in zip(columns, counts):
        positions = _initial_positions(rng, height, count)
        if count > 1 and rng.random() < 0.46:
            colors = rng.choice(palette, size=count, replace=True)
        else:
            colors = np.repeat(rng.choice(palette), count)
        grid[positions, col] = colors
    return grid


def _anchor_simple(rng: np.random.Generator, grid: np.ndarray,
                   background: int, palette: np.ndarray) -> set[int]:
    height, width = grid.shape
    settled, moving = rng.choice(width, size=2, replace=False)
    grid[:, settled] = background
    grid[:, moving] = background
    color = int(rng.choice(palette))
    count = int(rng.choice([1, height]))
    grid[height - count:, settled] = color
    grid[[0, 2, height - 1], moving] = color
    return {int(settled), int(moving)}


def _anchor_rich(rng: np.random.Generator, grid: np.ndarray,
                 background: int, palette: np.ndarray) -> set[int]:
    height, width = grid.shape
    rich_col = int(rng.choice([0, width - 1]))
    grid[:, rich_col] = background
    first, second, third = (int(x) for x in palette[:3])
    grid[[0, 2, height - 3, height - 1], rich_col] = (first, second, first, third)
    starts = [col for col in range(width - 2) if rich_col not in range(col, col + 3)]
    left = int(rng.choice(starts))
    grid[:, left:left + 3] = background
    row = int(rng.integers(0, height - 2))
    grid[row, left:left + 3] = second
    grid[row + 2, left] = first
    return {rich_col, left, left + 1, left + 2}


def _trim_to_majority(rng: np.random.Generator, grid: np.ndarray,
                      background: int, protected: set[int]) -> None:
    limit = (grid.size - 1) // 2
    while int(np.count_nonzero(grid != background)) > limit:
        candidates = np.argwhere(grid != background)
        candidates = [tuple(int(x) for x in position) for position in candidates
                      if int(position[1]) not in protected]
        if not candidates:
            raise RuntimeError("protected motifs exceed background majority")
        row, col = candidates[int(rng.integers(len(candidates)))]
        grid[row, col] = background


def _make_pair(rng: np.random.Generator, parameters: dict, regime: str,
               index: int, train_colors: set[int]) -> dict:
    height, width = _choose_shape(rng, parameters, regime, index)
    background = parameters["background"]
    minimum_columns = 3 if regime == "source_train" and index < 2 else 1
    number_hint = int(rng.integers(minimum_columns, width + 1))
    minimum_colors = 3 if regime == "source_train" and index == 1 else 2 if regime == "source_train" and index == 0 else 1
    palette_size = int(rng.integers(minimum_colors, min(9, number_hint + 2) + 1))
    unseen = [c for c in range(10) if c != background and c not in train_colors]
    required = int(rng.choice(unseen)) if regime == "target_test" and unseen and rng.random() < 0.55 else None
    palette = _palette(rng, background, palette_size, required)
    grid = _base_grid(rng, height, width, background, palette, minimum_columns)
    protected: set[int] = set()
    if regime == "source_train" and index == 0:
        protected = _anchor_simple(rng, grid, background, palette)
    elif regime == "source_train" and index == 1:
        protected = _anchor_rich(rng, grid, background, palette)
    if required is not None and not np.any(grid == required):
        columns = np.flatnonzero(np.any(grid != background, axis=0))
        col = int(rng.choice(columns))
        row = int(np.flatnonzero(grid[:, col] != background)[0])
        grid[row, col] = required
    _trim_to_majority(rng, grid, background, protected)
    if _background(grid) != background:
        raise RuntimeError("construction exceeded background budget")
    output = _compact(grid, background)
    return {"input": grid, "output": output}


def _array_key(grid: np.ndarray) -> tuple:
    return (grid.shape, grid.tobytes())


def _pair_key(pair: dict) -> tuple:
    return (_array_key(pair["input"]), _array_key(pair["output"]))


def _source_copy(pair: dict) -> bool:
    return any(np.array_equal(pair["input"], source_input)
               and np.array_equal(pair["output"], source_output)
               for source_input, source_output in _SOURCE_PAIRS)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    background = int(rng.choice(10, p=[0.55] + [0.05] * 9))
    parameters = {
        "background": background,
        "style": str(rng.choice(["similar", "larger_test", "recombine"], p=[0.27, 0.35, 0.38])),
        "base_shape": (int(rng.integers(6, 15)), int(rng.integers(6, 15))),
    }
    child_seeds = rng.integers(0, np.iinfo(np.uint64).max, size=train_count + test_count,
                               dtype=np.uint64)
    puzzle = {"train": [], "test": []}
    seen = set()
    train_colors: set[int] = set()
    child_index = 0
    for regime, count in (("source_train", train_count), ("target_test", test_count)):
        split = "train" if regime == "source_train" else "test"
        for index in range(count):
            pair_rng = np.random.default_rng(int(child_seeds[child_index]))
            child_index += 1
            for _ in range(64):
                pair = _make_pair(pair_rng, parameters, regime, index, train_colors)
                key = _array_key(pair["input"])
                if key not in seen and not _source_copy(pair):
                    break
            else:
                raise RuntimeError("could not make a distinct pair")
            seen.add(key)
            puzzle[split].append(pair)
            if split == "train":
                train_colors.update(int(x) for x in np.unique(pair["input"]) if x != background)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count,
             "illegal pair counts")
    seen_inputs = set()
    seen_pairs = set()
    backgrounds = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input and output")
            for grid in (pair["input"], pair["output"]):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a two-dimensional NumPy array")
                _require(all(1 <= side <= 30 for side in grid.shape),
                         "grid dimension outside 1 through 30")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must be integer")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "invalid ARC color")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape == out.shape, "shape changed")
            _require(_array_key(inp) not in seen_inputs, "duplicate input")
            _require(_pair_key(pair) not in seen_pairs, "duplicate pair")
            seen_inputs.add(_array_key(inp))
            seen_pairs.add(_pair_key(pair))
            _require(not _source_copy(pair), "source pair copied")
            background = _background(inp)
            _require(_background(out) == background, "background changed")
            backgrounds.add(background)
            _require(np.array_equal(out, _compact(inp, background)),
                     "incorrect stable downward compaction")
            for col in range(inp.shape[1]):
                before = inp[:, col][inp[:, col] != background]
                after = out[:, col][out[:, col] != background]
                _require(np.array_equal(before, after), "column sequence changed")
                _require(np.all(out[:out.shape[0] - after.size, col] == background),
                         "output column has a gap")
    _require(len(backgrounds) == 1, "background convention differs between pairs")
    background = next(iter(backgrounds))
    training = [pair["input"] for pair in puzzle["train"]]
    _require(any(_moving_columns(grid, background)[0] > 0 for grid in training),
             "training lacks falling cells")
    _require(any(_internal_gap(grid, background) for grid in training),
             "training lacks gap removal")
    _require(any(all(x > 0 for x in _moving_columns(grid, background)) for grid in training),
             "training lacks moving and settled columns together")
    _require(any(_shared_color_columns(grid, background) for grid in training),
             "training lacks one color in differently sized columns")
    _require(any(_order_example(grid, background) for grid in training),
             "training lacks an informative multicolor ordering example")
    _require(any(_broken_bar(grid, background) for grid in training),
             "training lacks a breaking horizontal cluster")


def _fall_simulation(grid: np.ndarray, background: int) -> np.ndarray:
    falling = grid.copy()
    while True:
        changed = False
        for row in range(falling.shape[0] - 2, -1, -1):
            for col in range(falling.shape[1]):
                if falling[row, col] != background and falling[row + 1, col] == background:
                    falling[row + 1, col] = falling[row, col]
                    falling[row, col] = background
                    changed = True
        if not changed:
            return falling


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a["input"], b["input"]) and
                   np.array_equal(a["output"], b["output"])
                   for a, b in zip(first[split], second[split]))
               for split in ("train", "test"))


def self_test() -> None:
    for source_input, source_output in _SOURCE_PAIRS:
        _require(np.array_equal(_fall_simulation(source_input, 0), source_output),
                 "source task disagrees with falling simulation")
    train_counts, test_counts = set(), set()
    shapes, layouts, palettes = set(), set(), set()
    backgrounds, heights, occupied_counts = set(), set(), set()
    settled_mixtures = order_examples = broken_bars = shared_colors = 0
    new_test_colors = 0
    within_shape_changes = within_palette_changes = 0
    within_occupancy_changes = within_height_changes = 0
    full_columns = equal_heights = empty_columns = zero_payloads = 0
    mixed_color_columns = all_settled = combined_axes = large_tests = 0
    test_color_novelty_puzzles = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        background = _background(puzzle["train"][0]["input"])
        backgrounds.add(background)
        pairs = puzzle["train"] + puzzle["test"]
        local_shapes, local_palettes = set(), set()
        local_occupied, local_heights = set(), set()
        train_colors = set()
        test_has_novel_color = False
        for pair in puzzle["train"]:
            train_colors.update(int(x) for x in np.unique(pair["input"]) if x != background)
            settled_mixtures += int(all(x > 0 for x in _moving_columns(pair["input"], background)))
            order_examples += int(_order_example(pair["input"], background))
            broken_bars += int(_broken_bar(pair["input"], background))
            shared_colors += int(_shared_color_columns(pair["input"], background))
        for pair in puzzle["test"]:
            novel = bool({int(x) for x in np.unique(pair["input"])
                          if x != background} - train_colors)
            new_test_colors += int(novel)
            test_has_novel_color |= novel
            large_tests += int(pair["input"].size >
                               max(item["input"].size for item in puzzle["train"]))
        for pair in pairs:
            grid = pair["input"]
            _require(np.array_equal(_fall_simulation(grid, background), pair["output"]),
                     "independent falling simulation disagrees")
            shapes.add(grid.shape)
            local_shapes.add(grid.shape)
            columns = [c for c in range(grid.shape[1])
                       if np.any(grid[:, c] != background)]
            counts = [int(np.count_nonzero(grid[:, c] != background)) for c in columns]
            heights.update(counts)
            local_heights.update(counts)
            occupied_counts.add(len(columns))
            local_occupied.add(len(columns))
            full_columns += int(any(count == grid.shape[0] for count in counts))
            equal_heights += int(len(counts) > len(set(counts)))
            empty_columns += int(len(columns) < grid.shape[1])
            zero_payloads += int(background != 0 and np.any(grid == 0))
            mixed_color_columns += int(any(
                len(set(int(x) for x in grid[:, c] if x != background)) > 1
                for c in columns))
            all_settled += int(np.array_equal(grid, pair["output"]))
            combined_axes += int(
                _moving_columns(grid, background)[1] > 0
                and _internal_gap(grid, background)
                and _shared_color_columns(grid, background))
            palette = frozenset(int(x) for x in np.unique(grid) if x != background)
            palettes.add(palette)
            local_palettes.add(palette)
            layouts.add((grid.shape, (grid != background).tobytes()))
        within_shape_changes += int(len(local_shapes) > 1)
        within_palette_changes += int(len(local_palettes) > 1)
        within_occupancy_changes += int(len(local_occupied) > 1)
        within_height_changes += int(len(local_heights) > 1)
        test_color_novelty_puzzles += int(test_has_novel_color)
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity collapsed")
    _require(len(shapes) >= 40 and any(h >= 24 or w >= 24 for h, w in shapes),
             "dimension diversity collapsed")
    _require(any(h > w for h, w in shapes) and any(w > h for h, w in shapes),
             "aspect ratio diversity collapsed")
    _require(len(layouts) >= 180 and len(palettes) >= 25,
             "structure or palette diversity collapsed")
    _require(len(backgrounds) >= 4 and 0 in backgrounds,
             "background diversity collapsed")
    _require(max(heights) >= 8 and max(occupied_counts) >= 8,
             "stack or occupied-column diversity collapsed")
    _require(min(occupied_counts) <= 2 and new_test_colors >= 5,
             "sparse columns or new test colors missing")
    _require(min(set(range(1, 11)) & heights) == 1,
             "one-cell columns missing")
    _require(all(x >= 20 for x in (settled_mixtures, order_examples,
                                   broken_bars, shared_colors)),
             "training evidence collapsed")
    _require(within_shape_changes >= 30 and within_palette_changes >= 30,
             "within-puzzle diversity collapsed")
    _require(within_occupancy_changes >= 30 and within_height_changes >= 30,
             "within-puzzle occupancy or stack-height diversity collapsed")
    _require(full_columns >= 8 and equal_heights >= 30 and empty_columns >= 30,
             "full, equal-height, or empty-column cases collapsed")
    _require(zero_payloads >= 8 and mixed_color_columns >= 30,
             "zero payload or multicolor columns collapsed")
    _require(all_settled >= 1 and combined_axes >= 20,
             "settled queries or combined variation axes collapsed")
    _require(large_tests >= 5 and test_color_novelty_puzzles >= 5,
             "train/test shifts collapsed")
    for seed, train, test in ((201, 2, 1), (202, 10, 1), (203, 10, 4),
                              (301, None, 1), (302, None, 4),
                              (303, 2, None), (304, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            _require(len(puzzle["train"]) == train, "explicit train count ignored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "explicit test count ignored")
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4},
                      {"num_train": True}, {"num_test": False}):
        try:
            generate_puzzle(401, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid count accepted: {arguments}")
    _require(_puzzles_equal(generate_puzzle(123456), generate_puzzle(123456)),
             "automatic generation is not deterministic")
    _require(_puzzles_equal(generate_puzzle(654321, 7, 3),
                           generate_puzzle(654321, 7, 3)),
             "explicit generation is not deterministic")
    _require(not _puzzles_equal(generate_puzzle(1), generate_puzzle(2)),
             "different seeds produced identical puzzles")
