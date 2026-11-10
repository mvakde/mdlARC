import numpy as np


_CROSS_OFFSETS = (
    (0, 0),
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)

# Raw grids from the supplied source task. These are used only to ensure that
# no generated input/output pair exactly duplicates a source pair.
_SOURCE_TASK_PAIRS = (
    (
        (
            "0001000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0001000000",
            "0000000000",
        ),
        (
            "0001000000",
            "0000000000",
            "0000000000",
            "0003000000",
            "0033300000",
            "0003000000",
            "0000000000",
            "0000000000",
            "0001000000",
            "0000000000",
        ),
    ),
    (
        (
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0100000100",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
        (
            "0000000000",
            "0000000000",
            "0000000000",
            "0000300000",
            "0103330100",
            "0000300000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
    ),
    (
        (
            "000100",
            "000000",
            "000000",
            "000000",
            "000000",
            "000000",
            "000100",
            "000000",
            "000000",
            "000000",
        ),
        (
            "000100",
            "000000",
            "000300",
            "003330",
            "000300",
            "000000",
            "000100",
            "000000",
            "000000",
            "000000",
        ),
    ),
    (
        (
            "000000000000",
            "000000000000",
            "000000000000",
            "010000000001",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
        ),
        (
            "000000000000",
            "000000000000",
            "000000300000",
            "010003330001",
            "000000300000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
            "000000000000",
        ),
    ),
)

_LEGAL_COUNT_PAIRS = tuple(
    (n_train, n_test)
    for n_test in range(1, 5)
    for n_train in range(max(2, n_test + 1), 11)
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _optional_count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    n_train = _optional_count("num_train", num_train, 2, 10)
    n_test = _optional_count("num_test", num_test, 1, 4)
    if n_train is None and n_test is None:
        return _LEGAL_COUNT_PAIRS[seed % len(_LEGAL_COUNT_PAIRS)]
    if n_train is None:
        choices = tuple(range(max(2, n_test + 1), 11))
        n_train = choices[seed % len(choices)]
    if n_test is None:
        choices = tuple(range(1, min(4, n_train - 1) + 1))
        n_test = choices[seed % len(choices)]
    if n_train <= n_test:
        raise ValueError("num_train must exceed num_test")
    return n_train, n_test


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, grid.tobytes())


def _pair_key(pair: dict) -> tuple:
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _matches_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    input_rows = tuple("".join(map(str, row)) for row in input_grid)
    output_rows = tuple("".join(map(str, row)) for row in output_grid)
    return any(input_rows == source_input and output_rows == source_output
               for source_input, source_output in _SOURCE_TASK_PAIRS)


def _pick_position(rng: np.random.Generator, low: int, high: int) -> int:
    if low == high:
        return low
    if rng.random() < 0.30:
        return low if rng.random() < 0.5 else high
    return int(rng.integers(low, high + 1))


def _choose_shape(rng: np.random.Generator, halfspan: int, orientation: str) -> tuple[int, int]:
    minimum = 2 * halfspan + 1
    if halfspan <= 4 and rng.random() < 0.25:
        aligned = minimum
        perpendicular = int(rng.integers(3, 7))
    else:
        ranges = ((minimum, min(30, minimum + 3)),
                  (min(30, minimum + 4), min(30, minimum + 10)),
                  (min(30, minimum + 11), 30))
        available = [(a, b) for a, b in ranges if a <= b and a >= minimum]
        a, b = available[int(rng.integers(len(available)))]
        aligned = int(rng.integers(a, b + 1))
        if rng.random() < 0.20:
            perpendicular = aligned
        else:
            a, b = ((3, 7), (8, 16), (17, 30))[int(rng.integers(3))]
            perpendicular = int(rng.integers(a, b + 1))
    if orientation == "horizontal":
        return perpendicular, aligned
    return aligned, perpendicular


def _draw_pair(
    rng: np.random.Generator,
    halfspan: int,
    orientation: str,
    colors: tuple[int, int, int],
    used_inputs: set,
    shape: tuple[int, int] | None = None,
    forbidden_shape: tuple[int, int] | None = None,
) -> dict:
    background, marker, construction = colors
    for _ in range(256):
        height, width = shape if shape is not None else _choose_shape(rng, halfspan, orientation)
        if (height, width) == forbidden_shape:
            continue
        if orientation == "horizontal":
            if height < 3 or width < 2 * halfspan + 1:
                continue
            center_row = _pick_position(rng, 1, height - 2)
            center_col = _pick_position(rng, halfspan, width - halfspan - 1)
            markers = ((center_row, center_col - halfspan),
                       (center_row, center_col + halfspan))
        else:
            if width < 3 or height < 2 * halfspan + 1:
                continue
            center_row = _pick_position(rng, halfspan, height - halfspan - 1)
            center_col = _pick_position(rng, 1, width - 2)
            markers = ((center_row - halfspan, center_col),
                       (center_row + halfspan, center_col))
        input_grid = np.full((height, width), background, dtype=np.int8)
        for row, col in markers:
            input_grid[row, col] = marker
        output_grid = input_grid.copy()
        for dr, dc in _CROSS_OFFSETS:
            output_grid[center_row + dr, center_col + dc] = construction
        key = _grid_key(input_grid)
        if key in used_inputs or _matches_source_pair(input_grid, output_grid):
            continue
        used_inputs.add(key)
        return {"input": input_grid, "output": output_grid}
    raise RuntimeError("could not construct a distinct pair")


def _choose_other_span(rng: np.random.Generator, current: int, maximum: int) -> int:
    shorter = tuple(range(2, min(current, maximum + 1)))
    longer = tuple(range(current + 1, maximum + 1))
    if shorter and longer:
        options = shorter if rng.random() < 0.5 else longer
    else:
        options = shorter or longer
    if not options:
        raise ValueError("no alternate halfspan fits")
    return int(options[int(rng.integers(len(options)))])


def _palette_for_pair(
    rng: np.random.Generator,
    base: tuple[int, int, int],
    mode: int,
    index: int,
) -> tuple[int, int, int]:
    background, marker, construction = base
    if index == 0 or mode == 0:
        return base
    if mode in (1, 3):
        choices = [color for color in range(10)
                   if color not in (background, marker, construction)]
        background = int(choices[int(rng.integers(len(choices)))])
    if mode in (2, 3):
        choices = [color for color in range(10)
                   if color not in (marker, background, construction)]
        marker = int(choices[int(rng.integers(len(choices)))])
    return background, marker, construction


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    _require(input_grid.shape == output_grid.shape, "input/output shape differs")
    values, counts = np.unique(input_grid, return_counts=True)
    _require(len(values) == 2, "input needs one background and one marker color")
    marker_candidates = values[counts == 2]
    _require(len(marker_candidates) == 1, "input needs exactly two equal markers")
    marker = int(marker_candidates[0])
    background = int(values[values != marker][0])
    _require(int(np.count_nonzero(input_grid == background)) == input_grid.size - 2,
             "all other input cells must be background")
    markers = np.argwhere(input_grid == marker)
    first, second = (tuple(int(x) for x in row) for row in markers)
    horizontal = first[0] == second[0]
    vertical = first[1] == second[1]
    _require(horizontal != vertical, "markers must share exactly one axis")
    separation = (abs(second[1] - first[1]) if horizontal
                  else abs(second[0] - first[0]))
    _require(separation >= 4 and separation % 2 == 0,
             "marker separation must be even and at least four")
    halfspan = separation // 2
    center = ((first[0] + second[0]) // 2, (first[1] + second[1]) // 2)
    cross = {(center[0] + dr, center[1] + dc) for dr, dc in _CROSS_OFFSETS}
    height, width = input_grid.shape
    _require(all(0 <= r < height and 0 <= c < width for r, c in cross),
             "complete cross must fit in grid")
    _require(not ({first, second} & cross), "cross overlaps a marker")
    changed = {tuple(int(x) for x in row) for row in np.argwhere(input_grid != output_grid)}
    _require(changed == cross, "only the five midpoint cross cells may change")
    construction_values = {int(output_grid[r, c]) for r, c in cross}
    _require(len(construction_values) == 1, "cross needs one construction color")
    construction = next(iter(construction_values))
    _require(construction not in (background, marker),
             "construction must differ from marker and background")
    return {"background": background, "marker": marker,
            "construction": construction,
            "orientation": "horizontal" if horizontal else "vertical",
            "halfspan": halfspan, "shape": (height, width),
            "center": center,
            "edge_arm": any(r in (0, height - 1) or c in (0, width - 1)
                            for r, c in cross)}


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
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    base = tuple(int(x) for x in rng.choice(10, size=3, replace=False))
    palette_mode = int(rng.choice(4, p=(0.35, 0.20, 0.20, 0.25)))
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max,
                              size=train_count + test_count, dtype=np.int64)
    used_inputs: set = set()
    train: list[dict] = []
    train_features: list[dict] = []
    first_span = int(rng.integers(2, 11))
    one_orientation = rng.random() < 0.25
    shared_orientation = "horizontal" if rng.random() < 0.5 else "vertical"
    for index in range(train_count):
        pair_rng = np.random.default_rng(int(pair_seeds[index]))
        span = (first_span if index == 0 else
                _choose_other_span(pair_rng, first_span, 10) if index == 1 else
                int(pair_rng.integers(2, 11)))
        orientation = (shared_orientation if one_orientation else
                       "horizontal" if pair_rng.random() < 0.5 else "vertical")
        colors = _palette_for_pair(pair_rng, base, palette_mode, index)
        pair = _draw_pair(pair_rng, span, orientation, colors, used_inputs,
                          forbidden_shape=train_features[0]["shape"] if index == 1 else None)
        train.append(pair)
        train_features.append(_analyze_pair(pair["input"], pair["output"]))

    demonstrated_orientations = tuple(sorted({x["orientation"] for x in train_features}))
    test: list[dict] = []
    for index in range(test_count):
        pair_rng = np.random.default_rng(int(pair_seeds[train_count + index]))
        colors = _palette_for_pair(pair_rng, base, palette_mode, train_count + index)
        for _ in range(128):
            reference = train_features[int(pair_rng.integers(train_count))]
            change_shape = bool(pair_rng.integers(2))
            change_span = bool(pair_rng.integers(2))
            fixed_shape = None if change_shape else reference["shape"]
            if fixed_shape is None:
                span = (_choose_other_span(pair_rng, reference["halfspan"], 12)
                        if change_span else reference["halfspan"])
                orientation = demonstrated_orientations[
                    int(pair_rng.integers(len(demonstrated_orientations)))]
            else:
                h, w = fixed_shape
                feasible = [orientation for orientation in demonstrated_orientations
                            if (w if orientation == "horizontal" else h) >= 5]
                if not feasible:
                    continue
                maximum = max(min(12, ((w if orientation == "horizontal" else h) - 1) // 2)
                              for orientation in feasible)
                if change_span:
                    if maximum == 2 and reference["halfspan"] == 2:
                        continue
                    span = _choose_other_span(pair_rng, reference["halfspan"], maximum)
                else:
                    span = reference["halfspan"]
                feasible = [orientation for orientation in feasible
                            if (w if orientation == "horizontal" else h) >= 2 * span + 1]
                if not feasible:
                    continue
                orientation = feasible[int(pair_rng.integers(len(feasible)))]
            try:
                pair = _draw_pair(pair_rng, span, orientation, colors, used_inputs,
                                  shape=fixed_shape,
                                  forbidden_shape=reference["shape"] if change_shape else None)
                break
            except RuntimeError:
                continue
        else:
            # Unique geometry is always available on a newly drawn canvas.
            span = int(pair_rng.integers(2, 13))
            orientation = demonstrated_orientations[
                int(pair_rng.integers(len(demonstrated_orientations)))]
            pair = _draw_pair(pair_rng, span, orientation, colors, used_inputs)
        test.append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle needs exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test,
             "illegal train/test pair counts")
    seen_inputs: set = set()
    seen_pairs: set = set()
    features: list[dict] = []
    for pair in puzzle["train"] + puzzle["test"]:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 "pair needs exactly input and output")
        for name in ("input", "output"):
            grid = pair[name]
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                     f"{name} must be a two-dimensional NumPy array")
            _require(all(1 <= d <= 30 for d in grid.shape), "grid dimensions out of bounds")
            _require(np.issubdtype(grid.dtype, np.integer), "grid needs integer dtype")
            _require(bool(np.all((0 <= grid) & (grid <= 9))), "grid color out of bounds")
        input_key = _grid_key(pair["input"])
        pair_key = _pair_key(pair)
        _require(input_key not in seen_inputs and pair_key not in seen_pairs,
                 "duplicate input or pair")
        seen_inputs.add(input_key)
        seen_pairs.add(pair_key)
        _require(not _matches_source_pair(pair["input"], pair["output"]),
                 "generated pair copies a source pair")
        features.append(_analyze_pair(pair["input"], pair["output"]))
    train_features = features[:n_train]
    test_features = features[n_train:]
    construction_colors = {x["construction"] for x in features}
    _require(len(construction_colors) == 1,
             "construction color must be learned once from training")
    _require(len({x["shape"] for x in train_features}) >= 2,
             "training must vary canvas dimensions")
    _require(len({x["halfspan"] for x in train_features}) >= 2,
             "training must vary marker distance")
    orientations = {x["orientation"] for x in train_features}
    _require(all(x["orientation"] in orientations for x in test_features),
             "test orientation must be demonstrated in training")


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple(_pair_key(pair) for pair in puzzle[split])
                 for split in ("train", "test"))


def _predict_using_training(train_pairs: list[dict], test_input: np.ndarray) -> np.ndarray:
    learned_colors = []
    for pair in train_pairs:
        changed_values = pair["output"][pair["input"] != pair["output"]]
        _require(len(changed_values) == 5 and len(np.unique(changed_values)) == 1,
                 "training does not reveal one five-cell construction color")
        learned_colors.append(int(changed_values[0]))
    _require(len(set(learned_colors)) == 1, "training construction colors disagree")
    values, counts = np.unique(test_input, return_counts=True)
    _require(len(values) == 2, "test input must reveal marker and background")
    marker = values[np.argmin(counts)]
    positions = np.argwhere(test_input == marker)
    _require(positions.shape == (2, 2), "test input needs two markers")
    center = (positions[0] + positions[1]) // 2
    predicted = test_input.copy()
    for dr, dc in _CROSS_OFFSETS:
        predicted[center[0] + dr, center[1] + dc] = learned_colors[0]
    return predicted


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        training_copies = [{"input": pair["input"].copy(),
                            "output": pair["output"].copy()}
                           for pair in puzzle["train"]]
        for pair in puzzle["test"]:
            predicted = _predict_using_training(training_copies,
                                                pair["input"].copy())
            _require(np.array_equal(predicted, pair["output"]),
                     "test output is not recoverable from training and test input")
    _require(len({_puzzle_key(p) for p in puzzles}) >= 40,
             "puzzles lack cross-seed diversity")
    _require(len({len(p["train"]) for p in puzzles}) >= 4 and
             len({len(p["test"]) for p in puzzles}) >= 3,
             "automatic pair counts lack diversity")
    for seed, train_count, test_count in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                                          (2001, None, 1), (2002, None, 4),
                                          (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train_count, test_count)
        if train_count is not None:
            _require(len(puzzle["train"]) == train_count, "train count not honored")
        if test_count is not None:
            _require(len(puzzle["test"]) == test_count, "test count not honored")
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}, {"num_train": 3.0},
                 {"num_test": True}):
        try:
            generate_puzzle(3001, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal count accepted: {args}")
    _require(_puzzle_key(generate_puzzle(812345, 7, 3)) ==
             _puzzle_key(generate_puzzle(812345, 7, 3)),
             "same seed and counts must be deterministic")
    _require(_puzzle_key(generate_puzzle(93421)) ==
             _puzzle_key(generate_puzzle(93421)), "automatic counts must be deterministic")
    train_shapes, test_shapes, train_spans, test_spans = set(), set(), set(), set()
    train_layouts, colors, edge_count = set(), set(), 0
    fixed_palettes = varied_palettes = both_orientations = repeated_orientation = 0
    shorter = repeated = longer = 0
    mode_puzzles = {(False, False): 0, (False, True): 0,
                    (True, False): 0, (True, True): 0}
    exact_compact_count = 0
    for puzzle in puzzles:
        train = [_analyze_pair(p["input"], p["output"]) for p in puzzle["train"]]
        test = [_analyze_pair(p["input"], p["output"]) for p in puzzle["test"]]
        exact_compact_count += sum(x["shape"] in ((5, 3), (3, 5))
                                   for x in train + test)
        train_shapes.update(x["shape"] for x in train)
        test_shapes.update(x["shape"] for x in test)
        train_spans.update(x["halfspan"] for x in train)
        test_spans.update(x["halfspan"] for x in test)
        train_layouts.update((x["shape"], x["orientation"], x["halfspan"], x["center"])
                             for x in train)
        colors.add(train[0]["construction"])
        edge_count += sum(x["edge_arm"] for x in train + test)
        palettes = {(x["background"], x["marker"]) for x in train + test}
        fixed_palettes += len(palettes) == 1
        varied_palettes += len(palettes) > 1
        both_orientations += len({x["orientation"] for x in train}) == 2
        repeated_orientation += len({x["orientation"] for x in train}) == 1
        observed_modes = set()
        for x in test:
            shorter += x["halfspan"] < max(y["halfspan"] for y in train)
            repeated += x["halfspan"] in {y["halfspan"] for y in train}
            longer += x["halfspan"] > min(y["halfspan"] for y in train)
            for y in train:
                mode = (x["shape"] != y["shape"],
                        x["halfspan"] != y["halfspan"])
                observed_modes.add(mode)
        for mode in observed_modes:
            mode_puzzles[mode] += 1
    _require(len(train_shapes) >= 18 and len(test_shapes) >= 12,
             "canvas variation is too narrow")
    _require(len(train_layouts) >= 35 and len(train_spans) >= 7 and len(test_spans) >= 7,
             "structural variation is too narrow")
    _require(len(colors) >= 8 and fixed_palettes >= 5 and varied_palettes >= 5,
             "palette variation is too narrow")
    _require(both_orientations >= 5 and repeated_orientation >= 2,
             "training orientation schedules are too narrow")
    _require(shorter >= 5 and repeated >= 5 and longer >= 5,
             "test gaps lack short, repeated, or longer cases")
    _require(all(count >= 5 for count in mode_puzzles.values()),
             "test canvas/gap combinations lack coverage")
    _require(edge_count >= 5 and exact_compact_count >= 1,
             "compact or edge-touching crosses lack coverage")
