import numpy as np


_DIRECTIONS = ("horizontal_negative", "horizontal_positive",
               "vertical_negative", "vertical_positive")

# Exact source inputs are excluded even if a generated connector color differs.
_SOURCE_INPUT_SPECS = {
    ((12, 14), ((0, 11), (7, 2))),
    ((13, 13), ((1, 2), (10, 8))),
    ((11, 14), ((6, 12), (8, 2))),
    ((13, 14), ((1, 1), (12, 8))),
}


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _normalize_count(
    value: int | None,
    name: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )
    return value


def _choose_counts(
    rng: np.random.Generator,
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _normalize_count(num_train, "num_train", 2, 10)
    test_count = _normalize_count(num_test, "num_test", 1, 4)

    if train_count is None and test_count is None:
        # This mapping guarantees broad count diversity over consecutive seeds.
        test_count = 1 + seed % 4
        minimum_train = max(2, test_count + 1)
        number_of_choices = 11 - minimum_train
        train_count = minimum_train + (seed // 4) % number_of_choices
    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal num_test exists for the requested num_train")
        test_count = int(rng.integers(1, maximum_test + 1))

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    return {
        "connector": int(rng.integers(0, 10)),
        "direction_mode": ("simple", "generalize", "mixed")[
            int(rng.integers(0, 3))
        ],
    }


def _validate_grid(grid: np.ndarray, label: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{label} must be a NumPy array")
    _require(grid.ndim == 2, f"{label} must be two-dimensional")

    height, width = grid.shape
    _require(1 <= height <= 30, f"{label} height is outside 1..30")
    _require(1 <= width <= 30, f"{label} width is outside 1..30")
    _require(
        np.issubdtype(grid.dtype, np.integer),
        f"{label} must use an integer dtype",
    )
    _require(
        np.all((grid >= 0) & (grid <= 9)),
        f"{label} contains a color outside 0..9",
    )


def _analyze_input(
    input_grid: np.ndarray,
) -> tuple[int, int, tuple[tuple[int, int], tuple[int, int]]]:
    values, counts = np.unique(input_grid, return_counts=True)
    _require(
        values.size == 2,
        "each input must contain exactly background and endpoint colors",
    )

    endpoint_indices = np.flatnonzero(counts == 2)
    _require(
        endpoint_indices.size == 1,
        "exactly one input color must occur twice",
    )

    endpoint_index = int(endpoint_indices[0])
    background_index = 1 - endpoint_index

    endpoint_color = int(values[endpoint_index])
    background_color = int(values[background_index])
    _require(
        int(counts[background_index]) > 2,
        "the background must occupy more than two cells",
    )

    locations_array = np.argwhere(input_grid == endpoint_color)
    locations = tuple(
        sorted((int(row), int(col)) for row, col in locations_array)
    )
    _require(len(locations) == 2, "the input must contain two endpoints")

    first, second = locations
    _require(first[0] != second[0], "endpoint rows must differ")
    _require(first[1] != second[1], "endpoint columns must differ")

    return background_color, endpoint_color, locations


def _geometry(
    endpoints: tuple[tuple[int, int], tuple[int, int]],
) -> dict:
    top, bottom = sorted(endpoints)
    row_span = bottom[0] - top[0]
    signed_col_span = bottom[1] - top[1]
    col_span = abs(signed_col_span)

    axis = "horizontal" if col_span > row_span else "vertical"
    sign = "positive" if signed_col_span > 0 else "negative"
    regime = f"{axis}_{sign}"

    return {
        "top": top,
        "bottom": bottom,
        "row_span": row_span,
        "signed_col_span": signed_col_span,
        "col_span": col_span,
        "diagonal_steps": min(row_span, col_span),
        "residual_steps": abs(row_span - col_span),
        "regime": regime,
    }


def _construct_output(
    input_grid: np.ndarray,
    connector_color: int,
) -> np.ndarray:
    background_color, endpoint_color, endpoints = _analyze_input(input_grid)
    _require(
        connector_color not in (background_color, endpoint_color),
        "connector color must differ from input colors",
    )

    start, destination = sorted(endpoints)
    row_delta = destination[0] - start[0]
    col_delta = destination[1] - start[1]
    abs_row_delta = abs(row_delta)
    abs_col_delta = abs(col_delta)

    _require(row_delta != 0 and col_delta != 0, "both offsets must be nonzero")
    _require(
        abs_row_delta != abs_col_delta,
        "row and column separations must be unequal",
    )

    row_sign = 1 if row_delta > 0 else -1
    col_sign = 1 if col_delta > 0 else -1
    diagonal_move = (row_sign, col_sign)
    diagonal_count = min(abs_row_delta, abs_col_delta)

    if abs_row_delta > abs_col_delta:
        straight_move = (row_sign, 0)
        straight_count = abs_row_delta - abs_col_delta
    else:
        straight_move = (0, col_sign)
        straight_count = abs_col_delta - abs_row_delta

    output_grid = input_grid.copy()

    def draw_route(straight_first: bool) -> None:
        row, col = start
        if straight_first:
            segments = (
                (straight_move, straight_count),
                (diagonal_move, diagonal_count),
            )
        else:
            segments = (
                (diagonal_move, diagonal_count),
                (straight_move, straight_count),
            )

        for (move_row, move_col), count in segments:
            for _ in range(count):
                row += move_row
                col += move_col
                _require(
                    0 <= row < input_grid.shape[0]
                    and 0 <= col < input_grid.shape[1],
                    "a constructed route left the grid",
                )
                if (row, col) != destination:
                    output_grid[row, col] = connector_color

        _require(
            (row, col) == destination,
            "a constructed route did not reach the destination",
        )

    draw_route(straight_first=True)
    draw_route(straight_first=False)

    return output_grid


def _expected_connector_cells(endpoints: tuple) -> set[tuple[int, int]]:
    top, bottom = sorted(endpoints)
    row_span = bottom[0] - top[0]
    col_span = abs(bottom[1] - top[1])
    col_sign = 1 if bottom[1] > top[1] else -1
    diagonal = min(row_span, col_span)
    residual = abs(row_span - col_span)
    cells: set[tuple[int, int]] = set()
    if row_span > col_span:
        cells.update((top[0] + j, top[1]) for j in range(1, residual + 1))
        cells.update((top[0] + residual + i, top[1] + col_sign * i)
                     for i in range(1, diagonal + 1))
        cells.update((top[0] + i, top[1] + col_sign * i)
                     for i in range(1, diagonal + 1))
        cells.update((top[0] + diagonal + j, bottom[1])
                     for j in range(1, residual + 1))
    else:
        cells.update((top[0], top[1] + col_sign * j)
                     for j in range(1, residual + 1))
        cells.update((top[0] + i, top[1] + col_sign * (residual + i))
                     for i in range(1, diagonal + 1))
        cells.update((top[0] + i, top[1] + col_sign * i)
                     for i in range(1, diagonal + 1))
        cells.update((bottom[0], top[1] + col_sign * (diagonal + j))
                     for j in range(1, residual + 1))
    cells.discard(top)
    cells.discard(bottom)
    return cells


def _analyze_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> dict:
    _require(
        input_grid.shape == output_grid.shape,
        "input and output dimensions must match",
    )

    background_color, endpoint_color, endpoints = _analyze_input(input_grid)
    geometry = _geometry(endpoints)

    _require(
        geometry["row_span"] != geometry["col_span"],
        "endpoint separations must be unequal",
    )
    _require(
        geometry["row_span"] > 0 and geometry["col_span"] > 0,
        "both endpoint separations must be positive",
    )

    output_colors = {int(value) for value in np.unique(output_grid)}
    extra_colors = output_colors - {background_color, endpoint_color}
    _require(
        len(extra_colors) == 1,
        "output must introduce exactly one connector color",
    )
    connector_color = next(iter(extra_colors))

    _require(
        output_colors
        == {background_color, endpoint_color, connector_color},
        "output has an invalid color set",
    )

    connector_cells = _expected_connector_cells(endpoints)
    actual_connector_cells = {tuple(map(int, point)) for point in
                              np.argwhere(output_grid == connector_color)}
    _require(actual_connector_cells == connector_cells,
             "output connector cells do not form the two-route perimeter")
    remaining = output_grid.copy()
    for point in connector_cells:
        remaining[point] = input_grid[point]
    _require(np.array_equal(remaining, input_grid),
             "output changes a background or endpoint outside the perimeter")
    _require(
        not np.array_equal(input_grid, output_grid),
        "a nondegenerate generated pair must change",
    )

    return {
        "roles": (background_color, endpoint_color, connector_color),
        "endpoints": endpoints,
        "geometry": geometry,
    }


def _matches_source_input(
    input_grid: np.ndarray,
    background_color: int,
    endpoint_color: int,
    endpoints: tuple[tuple[int, int], tuple[int, int]],
) -> bool:
    if background_color != 0 or endpoint_color != 8:
        return False
    specification = (tuple(input_grid.shape), tuple(sorted(endpoints)))
    return specification in _SOURCE_INPUT_SPECS


def _grid_key(grid: np.ndarray) -> tuple:
    canonical = np.ascontiguousarray(grid, dtype=np.int8)
    return tuple(grid.shape), canonical.tobytes()


def _pair_key(pair: dict) -> tuple:
    return _grid_key(pair["input"]), _grid_key(pair["output"])


def _sample_displacement(
    rng: np.random.Generator, direction: str, scale: str,
) -> tuple[int, int]:
    # Diagonal length and straight length are sampled separately. Their sum
    # is the dominant coordinate span, at most 29 on an ARC grid.
    ranges = {
        "strong": ((4, 9), (3, 7)),
        "general": ((1, 18), (1, 18)),
        "small": ((1, 5), (1, 5)),
        "familiar": ((4, 10), (2, 8)),
        "large": ((9, 18), (2, 11)),
        "broad": ((2, 10), (7, 18)),
        "near": ((5, 18), (1, 2)),
        "source": ((7, 10), (4, 7)),
    }
    diagonal_range, residual_range = ranges[scale]
    diagonal = int(rng.integers(diagonal_range[0], diagonal_range[1] + 1))
    upper_residual = min(residual_range[1], 29 - diagonal)
    residual = int(rng.integers(residual_range[0], upper_residual + 1))
    if direction.startswith("horizontal"):
        row_span, col_span = diagonal, diagonal + residual
    else:
        row_span, col_span = diagonal + residual, diagonal
    signed_col_span = col_span if direction.endswith("positive") else -col_span
    return row_span, signed_col_span


def _sample_axis_padding(
    rng: np.random.Generator, span: int, after_edge: bool = False,
) -> tuple[int, int]:
    capacity = 29 - span
    if after_edge:
        return int(rng.integers(0, capacity + 1)), 0
    if capacity == 0:
        return 0, 0
    style = int(rng.integers(0, 5))
    if style == 0:
        return 0, int(rng.integers(0, capacity + 1))
    if style == 1:
        return int(rng.integers(0, capacity + 1)), 0
    if style == 2 and capacity >= 2:
        before = int(rng.integers(1, capacity))
        after = int(rng.integers(1, capacity - before + 1))
        return before, after
    before = int(rng.integers(0, capacity + 1))
    after = int(rng.integers(0, capacity - before + 1))
    return before, after


def _sample_input_colors(
    rng: np.random.Generator, connector: int,
    avoid: tuple[int, int] | None,
) -> tuple[int, int]:
    backgrounds = [c for c in range(10) if c != connector]
    if avoid is not None:
        backgrounds.remove(avoid[0])
    background = int(rng.choice(backgrounds))
    endpoints = [c for c in range(10) if c not in (connector, background)]
    if avoid is not None and avoid[1] in endpoints:
        endpoints.remove(avoid[1])
    endpoint = int(rng.choice(endpoints))
    return background, endpoint


def _make_pair(
    rng: np.random.Generator,
    row_span: int,
    signed_col_span: int,
    connector: int,
    avoid_colors: tuple[int, int] | None = None,
    bottom_edge: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    col_span = abs(signed_col_span)
    top_padding, bottom_padding = _sample_axis_padding(
        rng, row_span, after_edge=bottom_edge,
    )
    left_padding, right_padding = _sample_axis_padding(rng, col_span)
    height = row_span + 1 + top_padding + bottom_padding
    width = col_span + 1 + left_padding + right_padding

    if signed_col_span > 0:
        top_endpoint = (top_padding, left_padding)
        bottom_endpoint = (top_padding + row_span, left_padding + col_span)
    else:
        top_endpoint = (top_padding, left_padding + col_span)
        bottom_endpoint = (top_padding + row_span, left_padding)

    background, endpoint = _sample_input_colors(
        rng, connector, avoid_colors,
    )
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[top_endpoint] = endpoint
    input_grid[bottom_endpoint] = endpoint
    return input_grid, _construct_output(input_grid, connector)


def _independent_pair_seeds(
    rng: np.random.Generator,
    count: int,
) -> list[int]:
    result: list[int] = []
    seen: set[int] = set()
    maximum = int(np.iinfo(np.int64).max)

    for _ in range(count):
        candidate = int(rng.integers(0, maximum))
        while candidate in seen:
            candidate = (candidate + 1) % maximum
        seen.add(candidate)
        result.append(candidate)

    return result


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    rng = np.random.default_rng(seed)
    train_count, test_count = _choose_counts(
        rng,
        seed,
        num_train,
        num_test,
    )
    puzzle_parameters = sample_puzzle_parameters(rng)

    mode = puzzle_parameters["direction_mode"]
    if mode == "simple":
        first = _DIRECTIONS[int(rng.integers(0, 4))]
        train_directions = [first] * train_count
    elif mode == "generalize":
        train_directions = ["horizontal_negative", "vertical_positive"]
        train_directions.extend(
            _DIRECTIONS[int(rng.integers(0, 4))]
            for _ in range(train_count - 2)
        )
    else:
        horizontal = _DIRECTIONS[int(rng.integers(0, 2))]
        vertical = ("vertical_positive" if horizontal.endswith("negative")
                    else "vertical_negative")
        train_directions = [horizontal, vertical]
        train_directions.extend(
            _DIRECTIONS[int(rng.integers(0, 4))]
            for _ in range(train_count - 2)
        )

    pair_seeds = _independent_pair_seeds(
        rng,
        train_count + test_count,
    )

    train_pairs: list[dict] = []
    test_pairs: list[dict] = []
    used_input_keys: set[tuple] = set()
    first_profile: tuple[int, int] | None = None
    first_colors: tuple[int, int] | None = None

    for pair_index, direction in enumerate(train_directions):
        pair_rng = np.random.default_rng(pair_seeds[pair_index])
        for _ in range(100):
            scale = "strong" if pair_index < 2 else (
                ("general", "small", "broad", "near", "familiar")
                [int(pair_rng.integers(0, 5))]
            )
            row_span, signed_col_span = _sample_displacement(
                pair_rng, direction, scale,
            )
            profile = (min(row_span, abs(signed_col_span)),
                       abs(row_span - abs(signed_col_span)))
            if pair_index == 1 and profile == first_profile:
                continue
            input_grid, output_grid = _make_pair(
                pair_rng,
                row_span,
                signed_col_span,
                puzzle_parameters["connector"],
                avoid_colors=first_colors if pair_index == 1 else None,
            )

            background, endpoint, endpoints = _analyze_input(input_grid)
            input_key = _grid_key(input_grid)
            if input_key in used_input_keys:
                continue
            if _matches_source_input(
                input_grid,
                background,
                endpoint,
                endpoints,
            ):
                continue

            pair = {"input": input_grid, "output": output_grid}
            train_pairs.append(pair)
            used_input_keys.add(input_key)

            if pair_index == 0:
                first_profile = profile
                first_colors = (background, endpoint)
            break
        else:
            raise RuntimeError("could not generate a unique training pair")

    for test_index in range(test_count):
        pair_rng = np.random.default_rng(
            pair_seeds[train_count + test_index]
        )
        for _ in range(100):
            if test_index == 0:
                scale = ("small", "familiar", "large", "broad", "near",
                         "source")[seed % 6]
            else:
                scale = ("general", "small", "familiar", "large", "broad",
                         "near", "source")[int(pair_rng.integers(0, 7))]
            if mode == "simple":
                direction = train_directions[0]
                if scale == "source" and direction != "vertical_positive":
                    scale = "large"
            elif scale == "source":
                direction = "vertical_positive"
            elif mode == "generalize" and test_index == 0:
                direction = ("horizontal_positive", "vertical_negative")[
                    int(pair_rng.integers(0, 2))]
            else:
                direction = _DIRECTIONS[int(pair_rng.integers(0, 4))]
            row_span, signed_col_span = _sample_displacement(
                pair_rng, direction, scale,
            )
            input_grid, output_grid = _make_pair(
                pair_rng,
                row_span,
                signed_col_span,
                puzzle_parameters["connector"],
                bottom_edge=(scale == "source"),
            )

            background, endpoint, endpoints = _analyze_input(input_grid)
            input_key = _grid_key(input_grid)
            if input_key in used_input_keys:
                continue
            if _matches_source_input(
                input_grid,
                background,
                endpoint,
                endpoints,
            ):
                continue

            pair = {"input": input_grid, "output": output_grid}
            test_pairs.append(pair)
            used_input_keys.add(input_key)
            break
        else:
            raise RuntimeError("could not generate a unique test pair")

    puzzle = {
        "train": train_pairs,
        "test": test_pairs,
    }
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict), "puzzle must be a dictionary")
    _require(
        set(puzzle.keys()) == {"train", "test"},
        "puzzle must contain exactly train and test",
    )
    _require(isinstance(puzzle["train"], list), "train must be a list")
    _require(isinstance(puzzle["test"], list), "test must be a list")

    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])
    _require(2 <= train_count <= 10, "train count is outside 2..10")
    _require(1 <= test_count <= 4, "test count is outside 1..4")
    _require(train_count > test_count, "train count must exceed test count")

    records = {"train": [], "test": []}
    input_keys: set[tuple] = set()
    pair_keys: set[tuple] = set()

    for split_name in ("train", "test"):
        for pair_index, pair in enumerate(puzzle[split_name]):
            label = f"{split_name}[{pair_index}]"
            _require(isinstance(pair, dict), f"{label} must be a dictionary")
            _require(
                set(pair.keys()) == {"input", "output"},
                f"{label} must contain exactly input and output",
            )

            input_grid = pair["input"]
            output_grid = pair["output"]
            _validate_grid(input_grid, f"{label}.input")
            _validate_grid(output_grid, f"{label}.output")

            record = _analyze_pair(input_grid, output_grid)
            records[split_name].append(record)

            input_key = _grid_key(input_grid)
            current_pair_key = _pair_key(pair)
            _require(
                input_key not in input_keys,
                "duplicate input grid found within the puzzle",
            )
            _require(
                current_pair_key not in pair_keys,
                "duplicate input/output pair found within the puzzle",
            )
            input_keys.add(input_key)
            pair_keys.add(current_pair_key)

            background, endpoint, _ = record["roles"]
            _require(
                not _matches_source_input(
                    input_grid,
                    background,
                    endpoint,
                    record["endpoints"],
                ),
                "a generated input exactly duplicates a source input",
            )

    connectors = {record["roles"][2]
                  for split in ("train", "test") for record in records[split]}
    _require(len(connectors) == 1, "connector color changes within a puzzle")
    connector = next(iter(connectors))
    backgrounds = {record["roles"][0] for record in records["train"]}
    endpoints = {record["roles"][1] for record in records["train"]}
    _require(len(backgrounds) >= 2,
             "training does not demonstrate changing backgrounds")
    _require(len(endpoints) >= 2,
             "training does not demonstrate changing endpoint colors")
    first_two = records["train"][:2]
    strong_profiles = set()
    for record in first_two:
        geometry = record["geometry"]
        _require(geometry["diagonal_steps"] >= 4
                 and geometry["residual_steps"] >= 3,
                 "the first two training outlines must be clearly formed")
        strong_profiles.add((geometry["diagonal_steps"],
                             geometry["residual_steps"]))
    _require(len(strong_profiles) == 2,
             "the clear training outlines must have different slopes")

    train_directions = {record["geometry"]["regime"]
                        for record in records["train"]}
    train_axes = {direction.split("_")[0] for direction in train_directions}
    train_signs = {direction.split("_")[1] for direction in train_directions}
    for record in records["test"]:
        direction = record["geometry"]["regime"]
        if direction not in train_directions:
            _require(len(train_axes) == 2 and len(train_signs) == 2,
                     "an unseen test direction lacks two-axis, two-sign training evidence")
    for split in ("train", "test"):
        for record in records[split]:
            _require(record["geometry"]["regime"] in _DIRECTIONS,
                     "invalid direction")
            _require(len(set(record["roles"])) == 3,
                     "a pair must have three distinct color roles")
            _require(record["roles"][2] == connector,
                     "a test connector cannot be inferred from training")


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first.keys()) != {"train", "test"}:
        return False
    if set(second.keys()) != {"train", "test"}:
        return False

    for split_name in ("train", "test"):
        if len(first[split_name]) != len(second[split_name]):
            return False
        for first_pair, second_pair in zip(
            first[split_name], second[split_name]
        ):
            for grid_name in ("input", "output"):
                first_grid = first_pair[grid_name]
                second_grid = second_pair[grid_name]
                if first_grid.dtype != second_grid.dtype:
                    return False
                if not np.array_equal(first_grid, second_grid):
                    return False
    return True


def _puzzle_fingerprint(puzzle: dict) -> tuple:
    return (
        tuple(_pair_key(pair) for pair in puzzle["train"]),
        tuple(_pair_key(pair) for pair in puzzle["test"]),
    )


def _check_direct_task_invariants(puzzle: dict) -> None:
    # Infer the only hidden convention from demonstrations, then predict each
    # test from a copy of its input. No generation parameter enters the solve.
    connectors = set()
    for pair in puzzle["train"]:
        changed = pair["output"] != pair["input"]
        connectors.update(int(value) for value in np.unique(pair["output"][changed]))
    _require(len(connectors) == 1, "training does not determine a connector")
    connector = next(iter(connectors))

    for pair in puzzle["test"]:
        input_grid = pair["input"].copy()
        _, _, endpoint_positions = _analyze_input(input_grid)
        prediction = input_grid.copy()
        for point in _expected_connector_cells(endpoint_positions):
            prediction[point] = connector
        _require(np.array_equal(prediction, pair["output"]),
                 "training-only test prediction failed")


def self_test() -> None:
    automatic_puzzles: list[dict] = []
    train_counts: set[int] = set()
    test_counts: set[int] = set()
    connector_colors: set[int] = set()
    grid_shapes: set[tuple[int, int]] = set()
    structural_layouts: set[tuple] = set()
    target_profiles: set[tuple[int, int]] = set()
    puzzle_fingerprints: set[tuple] = set()
    directions = {"train": set(), "test": set()}
    test_scale_relation = set()
    edge_sides = {"train": set(), "test": set()}
    test_residuals: set[int] = set()
    test_diagonals: set[int] = set()
    simple_direction_puzzles = 0
    unseen_direction_puzzles = 0
    combined_puzzles = 0

    # Forty-two fixed seeds exercise automatic counts and broad diversity.
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _check_direct_task_invariants(puzzle)
        automatic_puzzles.append(puzzle)

        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_fingerprints.add(_puzzle_fingerprint(puzzle))

        train_records = [_analyze_pair(pair["input"], pair["output"])
                         for pair in puzzle["train"]]
        test_records = [_analyze_pair(pair["input"], pair["output"])
                        for pair in puzzle["test"]]
        connector_colors.add(train_records[0]["roles"][2])
        train_regimes = {record["geometry"]["regime"]
                         for record in train_records}
        test_regimes = {record["geometry"]["regime"]
                        for record in test_records}
        if len(train_regimes | test_regimes) == 1:
            simple_direction_puzzles += 1
        if test_regimes - train_regimes:
            unseen_direction_puzzles += 1
        train_dominants = [max(record["geometry"]["row_span"],
                               record["geometry"]["col_span"])
                           for record in train_records]
        for record in test_records:
            dominant = max(record["geometry"]["row_span"],
                           record["geometry"]["col_span"])
            if dominant < min(train_dominants):
                test_scale_relation.add("smaller")
            elif dominant > max(train_dominants):
                test_scale_relation.add("larger")
            else:
                test_scale_relation.add("familiar")
        if len(train_regimes | test_regimes) > 1 and any(
            min(row for row, _ in record["endpoints"]) == 0
            for record in train_records
        ) and any(
            0 < min(row for row, _ in record["endpoints"])
            and max(row for row, _ in record["endpoints"])
            < pair["input"].shape[0] - 1
            for record, pair in zip(test_records, puzzle["test"])
        ):
            combined_puzzles += 1

        for split_name in ("train", "test"):
            for pair in puzzle[split_name]:
                input_grid = pair["input"]
                record = _analyze_pair(input_grid, pair["output"])
                geometry = record["geometry"]
                directions[split_name].add(geometry["regime"])
                top, bottom = record["endpoints"]
                if top[0] == 0:
                    edge_sides[split_name].add("top")
                if bottom[0] == input_grid.shape[0] - 1:
                    edge_sides[split_name].add("bottom")
                if min(top[1], bottom[1]) == 0:
                    edge_sides[split_name].add("left")
                if max(top[1], bottom[1]) == input_grid.shape[1] - 1:
                    edge_sides[split_name].add("right")

                grid_shapes.add(tuple(input_grid.shape))
                structural_layouts.add(
                    (
                        split_name,
                        tuple(input_grid.shape),
                        tuple(record["endpoints"]),
                        geometry["row_span"],
                        geometry["signed_col_span"],
                    )
                )
                if split_name == "test":
                    target_profiles.add(
                        (geometry["diagonal_steps"],
                         geometry["residual_steps"])
                    )
                    test_diagonals.add(geometry["diagonal_steps"])
                    test_residuals.add(geometry["residual_steps"])

    _require(
        len(train_counts) >= 4,
        "automatic generation did not vary train counts enough",
    )
    _require(
        len(test_counts) >= 3,
        "automatic generation did not vary test counts enough",
    )
    _require(
        len(puzzle_fingerprints) >= 40,
        "different seeds did not produce enough distinct puzzles",
    )
    _require(
        len(connector_colors) >= 8,
        "puzzle-level connector colors are insufficiently diverse",
    )
    _require(
        len(grid_shapes) >= 35,
        "grid shapes are insufficiently diverse",
    )
    _require(
        len(structural_layouts) >= 120,
        "endpoint layouts are insufficiently diverse",
    )
    _require(
        len(target_profiles) >= 15,
        "target scale and residual profiles are insufficiently diverse",
    )
    _require(directions["train"] == set(_DIRECTIONS)
             and directions["test"] == set(_DIRECTIONS),
             "all four directions must occur in both splits")
    _require(test_scale_relation == {"smaller", "familiar", "larger"},
             "tests must include smaller, familiar, and larger scales")
    _require(edge_sides["train"] == {"top", "bottom", "left", "right"}
             and edge_sides["test"] == {"top", "bottom", "left", "right"},
             "all edge contacts must occur in both splits")
    _require(min(test_residuals) <= 2 and max(test_residuals) >= 10
             and len(test_diagonals) >= 10,
             "test diagonal and residual ranges are too narrow")
    _require(simple_direction_puzzles >= 3 and unseen_direction_puzzles >= 3,
             "direction modes do not include simple and generalizing puzzles")
    _require(combined_puzzles >= 3,
             "orientation, color, and border changes do not combine")

    # Determinism with automatic and explicit counts.
    _require(
        _puzzles_equal(generate_puzzle(991), generate_puzzle(991)),
        "automatic-count generation is not deterministic",
    )
    _require(
        _puzzles_equal(
            generate_puzzle(992, num_train=7, num_test=3),
            generate_puzzle(992, num_train=7, num_test=3),
        ),
        "explicit-count generation is not deterministic",
    )
    _require(
        not _puzzles_equal(automatic_puzzles[0], automatic_puzzles[1]),
        "different seeds unexpectedly produced identical puzzles",
    )

    # Legal boundary count combinations.
    boundary_cases = (
        (2, 1),
        (10, 1),
        (10, 4),
    )
    for index, (train_count, test_count) in enumerate(boundary_cases):
        puzzle = generate_puzzle(
            2000 + index,
            num_train=train_count,
            num_test=test_count,
        )
        _require(len(puzzle["train"]) == train_count, "wrong train count")
        _require(len(puzzle["test"]) == test_count, "wrong test count")
        validate_generated_puzzle(puzzle)
        _check_direct_task_invariants(puzzle)

    # Partial count specification.
    partial_cases = (
        {"num_train": None, "num_test": 1},
        {"num_train": None, "num_test": 4},
        {"num_train": 2, "num_test": None},
        {"num_train": 10, "num_test": None},
    )
    for index, arguments in enumerate(partial_cases):
        puzzle = generate_puzzle(3000 + index, **arguments)
        validate_generated_puzzle(puzzle)
        _check_direct_task_invariants(puzzle)

        if arguments["num_train"] is not None:
            _require(
                len(puzzle["train"]) == arguments["num_train"],
                "partially specified train count was not preserved",
            )
        if arguments["num_test"] is not None:
            _require(
                len(puzzle["test"]) == arguments["num_test"],
                "partially specified test count was not preserved",
            )

    # Illegal individual values and illegal count relationships.
    illegal_cases = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
        {"num_train": 4.0, "num_test": 1},
        {"num_train": 3, "num_test": True},
    )
    for index, arguments in enumerate(illegal_cases):
        raised = False
        try:
            generate_puzzle(4000 + index, **arguments)
        except ValueError:
            raised = True
        _require(raised, f"illegal count case was accepted: {arguments}")
