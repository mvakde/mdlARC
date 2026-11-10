"""Generate crosses decorated with alternating Chebyshev-distance shells."""

import numpy as np


_SOURCE_SIGNATURES = frozenset({
    (10, 10, 4, 6, 0, 3, 4),
    (20, 20, 5, 7, 0, 3, 4),
    (12, 12, 8, 6, 0, 3, 4),
})
_COUNT_SCHEDULE = (
    (2, 1), (3, 1), (4, 2), (5, 1),
    (3, 2), (6, 3), (2, 1), (7, 2),
    (8, 4), (4, 1), (9, 3), (10, 4),
)


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        return _COUNT_SCHEDULE[seed % len(_COUNT_SCHEDULE)]
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _choose_palette(rng, ring, previous=None, change=None):
    colors = [color for color in range(10) if color != ring]
    if previous is None:
        selected = rng.choice(colors, size=2, replace=False)
        return int(selected[0]), int(selected[1])
    bg, structural = previous
    if change is None:
        change = int(rng.integers(0, 4))
    if change == 0:
        return previous
    if change in (1, 3):
        bg = int(rng.choice([color for color in colors if color != bg and color != structural]))
    if change in (2, 3):
        structural = int(rng.choice([color for color in colors if color != bg and color != structural]))
    return bg, structural


def _sample_dimensions(rng, regime, anchor=False):
    if anchor:
        return int(rng.integers(10, 31)), int(rng.integers(10, 31))
    band = int(rng.choice(3, p=(0.22, 0.51, 0.27) if regime == "source_train"
                          else (0.32, 0.40, 0.28)))
    if band == 0:
        height, width = (int(rng.integers(5, 10)) for _ in range(2))
    elif band == 1:
        height, width = (int(rng.integers(10, 25)) for _ in range(2))
    else:
        height = int(rng.integers(25, 31))
        width = int(rng.integers(7, 31))
        if int(rng.integers(2)):
            height, width = width, height
    if rng.random() < 0.20:
        width = height
    return height, width


def _sample_axis(rng, length, regime):
    # Geometry choices are independent for rows and columns.
    probabilities = (0.19, 0.17, 0.24, 0.40) if regime == "source_train" else (0.30, 0.16, 0.20, 0.34)
    mode = int(rng.choice(4, p=probabilities))
    if mode == 0:
        choices = [0, 1, length - 2, length - 1]
    elif mode == 1:
        choices = [2, 3, length - 4, length - 3]
    elif mode == 2:
        center = (length - 1) // 2
        choices = [max(0, min(length - 1, center + step)) for step in (-2, -1, 0, 1, 2)]
    else:
        choices = list(range(length))
    return int(rng.choice(choices))


def _sample_geometry(rng, regime, anchor=False):
    height, width = _sample_dimensions(rng, regime, anchor)
    if anchor:
        rows = [r for r in range(4, height - 4) if 2 * r != height - 1]
        row = int(rng.choice(rows))
        column = int(rng.integers(4, width - 4))
    else:
        row = _sample_axis(rng, height, regime)
        column = _sample_axis(rng, width, regime)
    return height, width, row, column


def _make_pair(geometry, palette, ring):
    height, width, row, column = geometry
    background, structural = palette
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[row, :] = structural
    input_grid[:, column] = structural
    rr, cc = np.indices(input_grid.shape)
    distance = np.maximum(np.abs(rr - row), np.abs(cc - column))
    output_grid = input_grid.copy()
    output_grid[(input_grid == background) & (distance > 0) & (distance % 2 == 0)] = ring
    return {"input": input_grid, "output": output_grid}


def _unique_geometry(rng, regime, used, anchor=False):
    for _ in range(128):
        geometry = _sample_geometry(rng, regime, anchor)
        if geometry not in used:
            used.add(geometry)
            return geometry
    # Bounded deterministic fallback, normally unreachable.
    for height in range(10 if anchor else 5, 31):
        for width in range(10 if anchor else 5, 31):
            for row in range(4 if anchor else 0, height - 4 if anchor else height):
                if anchor and 2 * row == height - 1:
                    continue
                for column in range(4 if anchor else 0, width - 4 if anchor else width):
                    geometry = height, width, row, column
                    if geometry not in used:
                        used.add(geometry)
                        return geometry
    raise RuntimeError("No unused cross geometry")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    ring = int(rng.integers(10))
    constant_palette = rng.random() < 0.43
    palettes = [_choose_palette(rng, ring)]
    for index in range(1, train_count + test_count):
        if constant_palette:
            palettes.append(palettes[0])
        else:
            palettes.append(_choose_palette(rng, ring, palettes[-1],
                                            change=int(rng.integers(1, 4)) if index == 1 else None))

    # Separate reproducible streams keep each pair's geometry draw independent.
    child_seeds = set()
    pair_rngs = []
    while len(pair_rngs) < train_count + test_count:
        child_seed = int(rng.integers(0, 1 << 63))
        if child_seed not in child_seeds:
            child_seeds.add(child_seed)
            pair_rngs.append(np.random.default_rng(child_seed))

    anchor_index = int(rng.integers(train_count))
    geometries = [None] * (train_count + test_count)
    used = set()
    for index in range(train_count):
        geometries[index] = _unique_geometry(pair_rngs[index], "source_train", used,
                                             anchor=index == anchor_index)

    # Some same-size reflected pairs isolate vertical placement, as in the source shift.
    reflected = rng.random() < 0.27
    if reflected:
        h, w, row, column = geometries[anchor_index]
        reflection = (h, w, h - 1 - row, column)
        if reflection not in used:
            geometries[train_count] = reflection
            palettes[train_count] = palettes[anchor_index]
            used.add(reflection)
    for index in range(train_count, train_count + test_count):
        if geometries[index] is None:
            geometries[index] = _unique_geometry(pair_rngs[index], "target_test", used)

    puzzle = {
        "train": [_make_pair(geometries[i], palettes[i], ring) for i in range(train_count)],
        "test": [_make_pair(geometries[i], palettes[i], ring)
                 for i in range(train_count, train_count + test_count)],
    }
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_key(grid):
    # Normalize dtype so numerically identical grids cannot evade duplication.
    return grid.shape, np.asarray(grid, dtype=np.int8).tobytes()


def _validate_grid(grid, label):
    _check(isinstance(grid, np.ndarray), f"{label} must be an array")
    _check(grid.ndim == 2, f"{label} must have two dimensions")
    _check(all(1 <= size <= 30 for size in grid.shape), f"{label} has invalid size")
    _check(np.issubdtype(grid.dtype, np.integer), f"{label} must contain integers")
    _check(bool(np.all((0 <= grid) & (grid <= 9))), f"{label} has invalid colors")


def _inspect_pair(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    _check(input_grid.shape == output_grid.shape, "Input/output sizes differ")
    height, width = input_grid.shape
    _check(height >= 5 and width >= 5, "Cross is too small to show shell phase")
    values = [int(color) for color in np.unique(input_grid)]
    _check(len(values) == 2, "Input must have two colors")
    candidates = []
    for structural in values:
        full_rows = np.flatnonzero(np.all(input_grid == structural, axis=1))
        full_columns = np.flatnonzero(np.all(input_grid == structural, axis=0))
        if len(full_rows) != 1 or len(full_columns) != 1:
            continue
        background = next(color for color in values if color != structural)
        row, column = int(full_rows[0]), int(full_columns[0])
        expected = np.full(input_grid.shape, background, dtype=input_grid.dtype)
        expected[row, :] = structural
        expected[:, column] = structural
        if np.array_equal(input_grid, expected):
            candidates.append((background, structural, row, column))
    _check(len(candidates) == 1, "Input lacks one unambiguous complete cross")
    background, structural, row, column = candidates[0]
    introduced = set(int(color) for color in np.unique(output_grid)) - set(values)
    _check(len(introduced) == 1, "Output must introduce one shell color")
    ring = next(iter(introduced))
    rr, cc = np.indices(input_grid.shape)
    distance = np.maximum(np.abs(rr - row), np.abs(cc - column))
    expected = input_grid.copy()
    expected[(input_grid == background) & (distance > 0) & (distance % 2 == 0)] = ring
    _check(np.array_equal(output_grid, expected), "Incorrect shell parity or structural preservation")
    for d in (1, 2):
        _check(bool(np.any((input_grid == background) & (distance == d))),
               f"Distance {d} is not visible")
    return {
        "geometry": (height, width, row, column),
        "palette": (background, structural),
        "ring": ring,
        "signature": (height, width, row, column, background, structural, ring),
        "closed_four": min(row, height - 1 - row, column, width - 1 - column) >= 4,
        "min_margin": min(row, height - 1 - row, column, width - 1 - column),
    }


def _predict_from_training(train, input_grid):
    # The introduced color is inferred only from demonstrations. Input roles
    # are recovered from the test cross, regardless of its visible colors.
    rings = {_inspect_pair(pair)["ring"] for pair in train}
    _check(len(rings) == 1, "Training examples disagree about shell color")
    ring = next(iter(rings))
    values = [int(color) for color in np.unique(input_grid)]
    _check(ring not in values, "Test input uses the learned shell color")
    _check(len(values) == 2, "Test input has unexpected colors")
    for structural in values:
        rows = np.flatnonzero(np.all(input_grid == structural, axis=1))
        columns = np.flatnonzero(np.all(input_grid == structural, axis=0))
        if len(rows) == 1 and len(columns) == 1:
            row, column = int(rows[0]), int(columns[0])
            background = next(color for color in values if color != structural)
            rr, cc = np.indices(input_grid.shape)
            distance = np.maximum(np.abs(rr - row), np.abs(cc - column))
            predicted = input_grid.copy()
            predicted[(input_grid == background) & (distance > 0) & (distance % 2 == 0)] = ring
            return predicted
    raise AssertionError("Test cross cannot be identified")


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle keys")
    train, test = puzzle["train"], puzzle["test"]
    _check(isinstance(train, list) and isinstance(test, list), "Splits must be lists")
    _check(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
           "Invalid pair counts")
    seen_inputs, seen_pairs, analyses = set(), set(), {"train": [], "test": []}
    for split, pairs in (("train", train), ("test", test)):
        for index, pair in enumerate(pairs):
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair keys")
            _validate_grid(pair["input"], f"{split} {index} input")
            _validate_grid(pair["output"], f"{split} {index} output")
            input_key = _grid_key(pair["input"])
            pair_key = input_key, _grid_key(pair["output"])
            _check(input_key not in seen_inputs and pair_key not in seen_pairs, "Duplicate pair")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            analysis = _inspect_pair(pair)
            _check(analysis["signature"] not in _SOURCE_SIGNATURES, "Exact source pair copied")
            analyses[split].append(analysis)
    rings = {a["ring"] for a in analyses["train"] + analyses["test"]}
    _check(len(rings) == 1, "Shell color must be learned across the puzzle")
    _check(any(a["closed_four"] for a in analyses["train"]),
           "Training lacks a complete distance-four demonstration")
    train_geometries = {a["geometry"] for a in analyses["train"]}
    _check(all(a["geometry"] not in train_geometries for a in analyses["test"]),
           "Test repeats a training layout")
    for pair in test:
        _check(np.array_equal(_predict_from_training(train, pair["input"]), pair["output"]),
               "Test cannot be predicted from training and its input")


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    for seed, train_count, test_count in ((500, 2, 1), (501, 10, 1), (502, 10, 4),
                                          (601, None, 1), (602, None, 4),
                                          (603, 2, None), (604, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            _check(len(puzzle["train"]) == train_count, "Train count ignored")
        if test_count is not None:
            _check(len(puzzle["test"]) == test_count, "Test count ignored")
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(700, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Accepted illegal counts: {arguments}")
    _check(_puzzle_key(generate_puzzle(173)) == _puzzle_key(generate_puzzle(173)),
           "Automatic generation is not deterministic")
    _check(_puzzle_key(generate_puzzle(211, 8, 3)) == _puzzle_key(generate_puzzle(211, 8, 3)),
           "Explicit generation is not deterministic")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _check(len({_puzzle_key(p) for p in puzzles}) == 42, "Repeated complete puzzles")
    _check(len({len(p["train"]) for p in puzzles}) >= 4, "Too few train counts")
    _check(len({len(p["test"]) for p in puzzles}) >= 3, "Too few test counts")
    _check(sum(len(p["train"]) == 2 and len(p["test"]) == 1 for p in puzzles) >= 4,
           "Too few short puzzles")
    all_analyses = {split: [] for split in ("train", "test")}
    changing, constant, same_side, mixed_train, reflected = 0, 0, 0, 0, 0
    background_only, cross_only, both_colors = 0, 0, 0
    combined_variation, smaller_test, larger_test = 0, 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train = [_inspect_pair(pair) for pair in puzzle["train"]]
        test = [_inspect_pair(pair) for pair in puzzle["test"]]
        all_analyses["train"].extend(train)
        all_analyses["test"].extend(test)
        changing += len({a["palette"] for a in train + test}) > 1
        constant += len({a["palette"] for a in train + test}) == 1
        initial = train[0]["palette"]
        background_only += any(a["palette"][0] != initial[0] and
                               a["palette"][1] == initial[1] for a in train + test)
        cross_only += any(a["palette"][0] == initial[0] and
                          a["palette"][1] != initial[1] for a in train + test)
        both_colors += any(a["palette"][0] != initial[0] and
                           a["palette"][1] != initial[1] for a in train + test)
        combined_variation += len({a["palette"] for a in train + test}) > 1 and any(
            max(a["geometry"][:2]) <= 9 and a["min_margin"] < 2 for a in train + test)
        smaller_test += any(b["geometry"][0] * b["geometry"][1] <
                            a["geometry"][0] * a["geometry"][1]
                            for a in train for b in test)
        larger_test += any(b["geometry"][0] * b["geometry"][1] >
                           a["geometry"][0] * a["geometry"][1]
                           for a in train for b in test)
        side = lambda a: (2 * a["geometry"][2] > a["geometry"][0] - 1) - (2 * a["geometry"][2] < a["geometry"][0] - 1)
        same_side += any(side(a) == side(b) for a in train for b in test)
        mixed_train += len({side(a) for a in train}) > 1
        reflected += any(a["geometry"][:2] == b["geometry"][:2] and
                         a["geometry"][2] + b["geometry"][2] == a["geometry"][0] - 1 and
                         a["geometry"][3] == b["geometry"][3] and a["palette"] == b["palette"]
                         for a in train for b in test)
        for pair in puzzle["test"]:
            _check(np.array_equal(_predict_from_training(puzzle["train"], pair["input"]),
                                  pair["output"]), "Training-only prediction failed")
    combined = all_analyses["train"] + all_analyses["test"]
    geometries = {a["geometry"] for a in combined}
    dimensions = {a["geometry"][:2] for a in combined}
    _check(len(geometries) >= 150 and len(dimensions) >= 65, "Geometry diversity is low")
    _check(changing >= 12 and constant >= 8, "Palette regimes lack diversity")
    _check(min(background_only, cross_only, both_colors) >= 3,
           "Background and cross recoloring modes lack diversity")
    _check(combined_variation >= 4, "Palette and clipped compact shells never combine")
    _check(same_side >= 8 and mixed_train >= 8 and reflected >= 4,
           "Vertical placement regimes lack diversity")
    _check(smaller_test >= 8 and larger_test >= 8,
           "Test grid scales lack diversity")
    _check(sum(max(a["geometry"][:2]) <= 9 for a in combined) >= 20,
           "Too few compact examples")
    _check(sum(a["min_margin"] <= 1 for a in combined) >= 25,
           "Too few clipped shells")
    _check(sum(max(a["geometry"][:2]) >= 25 for a in all_analyses["train"]) >= 15,
           "Large training grids are missing")
    _check({"upper", "center", "lower"} ==
           {"upper" if 2 * a["geometry"][2] < a["geometry"][0] - 1 else
            "lower" if 2 * a["geometry"][2] > a["geometry"][0] - 1 else "center"
            for a in all_analyses["test"]}, "Test vertical zones lack diversity")
