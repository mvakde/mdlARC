import numpy as np


_DIRECTIONS = ((1, -1), (-1, 1), (1, 1), (-1, -1))
_DOMINANCES = ("vertical", "horizontal", "equal")

# (height, width, departure_position, destination_position), zero-indexed.
_SOURCE_LAYOUTS = frozenset(
    {
        (15, 16, (1, 6), (12, 3)),
        (15, 12, (1, 1), (13, 8)),
        (16, 16, (13, 3), (1, 11)),
        (13, 12, (3, 11), (10, 1)),
        (16, 16, (13, 12), (1, 4)),
    }
)


def _validated_count(
    value: int,
    name: str,
    low: int,
    high: int,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = (
        None
        if num_train is None
        else _validated_count(num_train, "num_train", 2, 10)
    )
    test_count = (
        None
        if num_test is None
        else _validated_count(num_test, "num_test", 1, 4)
    )

    if train_count is None and test_count is None:
        # Consecutive seeds cycle through all four test counts and many
        # different legal training counts.
        test_count = 1 + seed % 4
        lower = test_count + 1
        train_count = lower + (seed // 4) % (11 - lower)
    elif train_count is None:
        lower = max(2, test_count + 1)
        train_count = int(rng.integers(lower, 11))
    elif test_count is None:
        upper = min(4, train_count - 1)
        test_count = int(rng.integers(1, upper + 1))

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    colors = rng.permutation(10)
    background_count = int(rng.integers(2, 5))
    return {
        "departure": int(colors[0]),
        "destination": int(colors[1]),
        "connector": int(colors[2]),
        "backgrounds": tuple(int(x) for x in colors[3:3 + background_count]),
    }


def _sign(value: int) -> int:
    return 1 if value > 0 else -1


def _connector_cells(
    departure: tuple[int, int],
    destination: tuple[int, int],
) -> list[tuple[int, int]]:
    r, c = departure
    target_r, target_c = destination

    row_delta = target_r - r
    col_delta = target_c - c
    assert row_delta != 0 and col_delta != 0

    row_sign = _sign(row_delta)
    col_sign = _sign(col_delta)
    row_distance = abs(row_delta)
    col_distance = abs(col_delta)

    cells: list[tuple[int, int]] = []

    # Mandatory first diagonal step.
    r += row_sign
    c += col_sign
    if (r, c) != destination:
        cells.append((r, c))

    # Consume the surplus distance on the dominant axis.
    if row_distance > col_distance:
        for _ in range(row_distance - col_distance):
            r += row_sign
            if (r, c) != destination:
                cells.append((r, c))
    elif col_distance > row_distance:
        for _ in range(col_distance - row_distance):
            c += col_sign
            if (r, c) != destination:
                cells.append((r, c))

    # Finish diagonally once the remaining distances are equal.
    while (r, c) != destination:
        r += row_sign
        c += col_sign
        if (r, c) != destination:
            cells.append((r, c))

    return cells


def _sample_distances(
    rng: np.random.Generator,
    dominance: str,
) -> tuple[int, int]:
    if dominance == "equal":
        distance = int(rng.integers(3, 25))
        return distance, distance

    # The mixture covers almost balanced paths, long straight sections,
    # long diagonal sections, and the intermediate cases.  The two lengths
    # are still drawn afresh for each pair and are capped only by the grid.
    mode = int(rng.integers(0, 4))
    if mode == 0:
        minor = int(rng.integers(3, 19))
        excess = int(rng.integers(1, 3))
    elif mode == 1:
        minor = int(rng.integers(3, 8))
        excess = int(rng.integers(9, 21))
    elif mode == 2:
        minor = int(rng.integers(10, 24))
        excess = int(rng.integers(1, 6))
    else:
        minor = int(rng.integers(3, 17))
        excess = int(rng.integers(1, 13))
    excess = min(excess, 28 - minor)

    if dominance == "vertical":
        return minor + excess, minor
    if dominance == "horizontal":
        return minor, minor + excess

    raise ValueError("unknown dominance")


def _sample_dimensions(
    rng: np.random.Generator,
    row_distance: int,
    col_distance: int,
) -> tuple[int, int]:
    # Canvas aspect is chosen separately from path dominance.  A wide
    # canvas is possible even for a vertically dominant path and vice versa.
    mode = int(rng.integers(0, 4))
    scale = int(rng.integers(0, 4))

    def draw(minimum: int, maximum: int = 30) -> int:
        if scale == 0:
            upper = min(maximum, minimum + 4)
            lower = minimum
        elif scale == 1:
            upper = min(maximum, minimum + 10)
            lower = minimum
        elif scale == 2:
            upper = maximum
            lower = minimum
        else:
            upper = maximum
            lower = max(minimum, min(25, maximum))
        return int(rng.integers(lower, upper + 1))

    if mode == 0:  # square
        side = draw(max(row_distance, col_distance) + 1)
        return side, side
    if mode == 1:  # wide
        if row_distance + 1 <= 27:
            height = draw(row_distance + 1, 27)
            width = draw(max(col_distance + 1, height + 3))
            return height, width
    if mode == 2:  # tall
        if col_distance + 1 <= 27:
            width = draw(col_distance + 1, 27)
            height = draw(max(row_distance + 1, width + 3))
            return height, width
    return (
        draw(row_distance + 1),
        draw(col_distance + 1),
    )


def _place_axis(
    rng: np.random.Generator,
    length: int,
    distance: int,
    direction: int,
) -> tuple[int, int]:
    if direction == 1:
        departure = int(rng.integers(0, length - distance))
        return departure, departure + distance

    departure = int(rng.integers(distance, length))
    return departure, departure - distance


def generate_pair(
    rng: np.random.Generator,
    regime: str,
    puzzle_parameters: dict,
    direction: tuple[int, int] | None = None,
    dominance: str | None = None,
    background: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown regime")

    if direction is None:
        direction = _DIRECTIONS[int(rng.integers(0, len(_DIRECTIONS)))]

    direction = tuple(int(x) for x in direction)
    if (
        len(direction) != 2
        or direction[0] not in (-1, 1)
        or direction[1] not in (-1, 1)
    ):
        raise ValueError("direction must contain two signs")

    if dominance is None:
        dominance = _DOMINANCES[int(rng.integers(0, len(_DOMINANCES)))]
    if dominance not in _DOMINANCES:
        raise ValueError("unknown dominance")

    row_distance, col_distance = _sample_distances(rng, dominance)
    height, width = _sample_dimensions(rng, row_distance, col_distance)

    departure_r, destination_r = _place_axis(
        rng, height, row_distance, direction[0]
    )
    departure_c, destination_c = _place_axis(
        rng, width, col_distance, direction[1]
    )

    departure = (departure_r, departure_c)
    destination = (destination_r, destination_c)

    if background is None:
        background = int(rng.choice(puzzle_parameters["backgrounds"]))
    if background not in puzzle_parameters["backgrounds"]:
        raise ValueError("background is outside the puzzle palette")

    input_grid = np.full(
        (height, width),
        background,
        dtype=np.int8,
    )
    input_grid[departure] = puzzle_parameters["departure"]
    input_grid[destination] = puzzle_parameters["destination"]

    output_grid = input_grid.copy()
    for r, c in _connector_cells(departure, destination):
        output_grid[r, c] = puzzle_parameters["connector"]

    return input_grid, output_grid


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _pair_key(pair: dict) -> tuple:
    return (
        _grid_key(pair["input"]),
        _grid_key(pair["output"]),
    )


def _is_original_source_input(grid: np.ndarray) -> bool:
    values, counts = np.unique(grid, return_counts=True)

    if set(int(x) for x in values) != {0, 1, 2}:
        return False

    count_by_color = {
        int(value): int(count)
        for value, count in zip(values, counts)
    }
    if count_by_color != {0: grid.size - 2, 1: 1, 2: 1}:
        return False

    departure = tuple(int(x) for x in np.argwhere(grid == 2)[0])
    destination = tuple(int(x) for x in np.argwhere(grid == 1)[0])

    signature = (
        grid.shape[0],
        grid.shape[1],
        departure,
        destination,
    )
    return signature in _SOURCE_LAYOUTS


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
    train_count, test_count = _resolve_counts(
        seed,
        rng,
        num_train,
        num_test,
    )
    puzzle_parameters = sample_puzzle_parameters(rng)

    # Opposite training directions expose both signs on both axes, even for
    # a two-example task. Rotate the omitted quadrant across seeds.
    first = _DIRECTIONS[int(rng.integers(0, 4))]
    opposite = (-first[0], -first[1])
    remaining = [d for d in _DIRECTIONS if d not in (first, opposite)]
    held_out = remaining.pop(int(rng.integers(0, 2)))
    train_directions = [first, opposite] + remaining
    train_dominances = ["vertical", "horizontal"]
    if int(rng.integers(0, 2)):
        train_dominances.reverse()
    backgrounds = list(puzzle_parameters["backgrounds"])
    rng.shuffle(backgrounds)

    puzzle = {
        "train": [],
        "test": [],
    }
    seen_inputs: set[tuple] = set()

    for index in range(train_count):
        pair_seed = int(
            rng.integers(
                0,
                np.iinfo(np.int64).max,
                dtype=np.int64,
            )
        )
        pair_rng = np.random.default_rng(pair_seed)

        direction = (
            train_directions[index]
            if index < len(train_directions)
            else train_directions[int(pair_rng.integers(0, len(train_directions)))]
        )
        dominance = (
            train_dominances[index]
            if index < 2
            else _DOMINANCES[int(pair_rng.integers(0, 3))]
        )
        background = backgrounds[index] if index < 2 else None

        for _ in range(200):
            input_grid, output_grid = generate_pair(
                pair_rng,
                "source_train",
                puzzle_parameters,
                direction,
                dominance,
                background,
            )
            input_key = _grid_key(input_grid)
            if (
                input_key not in seen_inputs
                and not _is_original_source_input(input_grid)
            ):
                break
        else:
            raise RuntimeError(
                "could not construct a distinct training pair"
            )

        seen_inputs.add(input_key)
        puzzle["train"].append(
            {
                "input": input_grid,
                "output": output_grid,
            }
        )

    shown_directions = train_directions[:min(train_count, 3)]
    if test_count == 1:
        test_directions = [
            held_out if int(rng.integers(0, 10)) < 7
            else shown_directions[int(rng.integers(0, len(shown_directions)))]
        ]
    else:
        test_directions = [
            held_out,
            shown_directions[int(rng.integers(0, len(shown_directions)))],
        ]
        while len(test_directions) < test_count:
            test_directions.append(_DIRECTIONS[int(rng.integers(0, 4))])
        rng.shuffle(test_directions)

    test_dominances = list(_DOMINANCES)
    rng.shuffle(test_dominances)
    if test_count == 1:
        test_dominances = [test_dominances[0]]
    elif test_count == 2:
        test_dominances = test_dominances[:2]
    elif test_count == 4:
        test_dominances.append(_DOMINANCES[int(rng.integers(0, 3))])

    for test_index in range(test_count):
        pair_seed = int(
            rng.integers(
                0,
                np.iinfo(np.int64).max,
                dtype=np.int64,
            )
        )
        pair_rng = np.random.default_rng(pair_seed)

        direction = test_directions[test_index]
        dominance = test_dominances[test_index]
        # Occasionally reproduce the source's precise horizontal-reflection
        # relationship, using a generated training pair as the counterpart.
        reflected_pair = None
        if direction == held_out and int(pair_rng.integers(0, 5)) == 0:
            mirror_direction = (direction[0], -direction[1])
            for candidate in puzzle["train"]:
                start = _marker_position(candidate["input"], puzzle_parameters["departure"])
                end = _marker_position(candidate["input"], puzzle_parameters["destination"])
                if (_sign(end[0] - start[0]), _sign(end[1] - start[1])) == mirror_direction:
                    reflected_pair = candidate
                    break

        for attempt in range(200):
            if attempt == 0 and reflected_pair is not None:
                input_grid = np.fliplr(reflected_pair["input"]).copy()
                output_grid = np.fliplr(reflected_pair["output"]).copy()
            else:
                input_grid, output_grid = generate_pair(
                    pair_rng,
                    "target_test",
                    puzzle_parameters,
                    direction,
                    dominance,
                )
            input_key = _grid_key(input_grid)
            if (
                input_key not in seen_inputs
                and not _is_original_source_input(input_grid)
            ):
                break
        else:
            raise RuntimeError(
                "could not construct a distinct test pair"
            )

        seen_inputs.add(input_key)
        puzzle["test"].append(
            {
                "input": input_grid,
                "output": output_grid,
            }
        )

    validate_generated_puzzle(puzzle)
    return puzzle


def _input_roles(
    grid: np.ndarray,
) -> tuple[int, frozenset[int]]:
    values, counts = np.unique(grid, return_counts=True)
    assert len(values) == 3

    color_counts = {
        int(value): int(count)
        for value, count in zip(values, counts)
    }
    backgrounds = [
        color
        for color, count in color_counts.items()
        if count == grid.size - 2
    ]
    singletons = [
        color
        for color, count in color_counts.items()
        if count == 1
    ]

    assert len(backgrounds) == 1
    assert len(singletons) == 2
    assert len(set(singletons + backgrounds)) == 3

    return backgrounds[0], frozenset(singletons)


def _marker_position(
    grid: np.ndarray,
    color: int,
) -> tuple[int, int]:
    positions = np.argwhere(grid == color)
    assert positions.shape == (1, 2)
    return tuple(int(x) for x in positions[0])


def _expected_from_input(
    grid: np.ndarray,
    departure_color: int,
    destination_color: int,
    connector_color: int,
) -> np.ndarray:
    departure = _marker_position(grid, departure_color)
    destination = _marker_position(grid, destination_color)

    expected = grid.copy()
    for r, c in _connector_cells(departure, destination):
        expected[r, c] = connector_color
    return expected


def _formula_expected(
    grid: np.ndarray,
    departure_color: int,
    destination_color: int,
    connector_color: int,
) -> np.ndarray:
    """A coordinate formula used by the self-test, separate from drawing."""
    start = _marker_position(grid, departure_color)
    end = _marker_position(grid, destination_color)
    dr = abs(end[0] - start[0])
    dc = abs(end[1] - start[1])
    sr = _sign(end[0] - start[0])
    sc = _sign(end[1] - start[1])
    surplus = abs(dr - dc)
    expected = grid.copy()
    for step in range(1, max(dr, dc)):
        row_progress = step if dr >= dc else max(1, step - surplus)
        col_progress = step if dc >= dr else max(1, step - surplus)
        expected[
            start[0] + sr * row_progress,
            start[1] + sc * col_progress,
        ] = connector_color
    return expected


def _analyze_task(puzzle: dict) -> dict:
    all_pairs = puzzle["train"] + puzzle["test"]

    backgrounds: set[int] = set()
    common_markers = None
    common_connector = None
    train_matches: list[set[int]] = []

    for pair in all_pairs:
        input_grid = pair["input"]
        output_grid = pair["output"]

        background, markers = _input_roles(input_grid)
        input_colors = set(int(x) for x in np.unique(input_grid))
        output_colors = set(int(x) for x in np.unique(output_grid))
        added_colors = output_colors - input_colors

        assert len(added_colors) == 1
        connector = next(iter(added_colors))
        assert output_colors == input_colors | {connector}
        assert not np.array_equal(input_grid, output_grid)

        backgrounds.add(background)
        if common_markers is None:
            common_markers = markers
            common_connector = connector
        else:
            assert markers == common_markers
            assert connector == common_connector

        marker_list = sorted(markers)
        first = _marker_position(input_grid, marker_list[0])
        second = _marker_position(input_grid, marker_list[1])

        assert first[0] != second[0]
        assert first[1] != second[1]

        row_distance = abs(first[0] - second[0])
        col_distance = abs(first[1] - second[1])
        assert min(row_distance, col_distance) >= 3

    assert not (backgrounds & common_markers)
    assert common_connector not in backgrounds

    # Infer the departure color using training outputs only. Balanced paths
    # may match both directions, but the required asymmetric anchors make the
    # intersection unique.
    possible_departures = set(common_markers)

    for pair in puzzle["train"]:
        matches: set[int] = set()

        for candidate in common_markers:
            other = next(
                color
                for color in common_markers
                if color != candidate
            )
            expected = _expected_from_input(
                pair["input"],
                candidate,
                other,
                common_connector,
            )
            if np.array_equal(expected, pair["output"]):
                matches.add(candidate)

        assert matches
        train_matches.append(matches)
        possible_departures &= matches

    assert len(possible_departures) == 1
    departure_color = next(iter(possible_departures))
    destination_color = next(
        color
        for color in common_markers
        if color != departure_color
    )
    assert any(
        matches == {departure_color}
        for matches in train_matches
    )

    train_row_signs: set[int] = set()
    train_col_signs: set[int] = set()
    train_dominances: set[str] = set()

    for section in ("train", "test"):
        for pair in puzzle[section]:
            input_grid = pair["input"]
            departure = _marker_position(
                input_grid,
                departure_color,
            )
            destination = _marker_position(
                input_grid,
                destination_color,
            )

            row_delta = destination[0] - departure[0]
            col_delta = destination[1] - departure[1]
            direction = (
                _sign(row_delta),
                _sign(col_delta),
            )

            row_distance = abs(row_delta)
            col_distance = abs(col_delta)
            if row_distance > col_distance:
                dominance = "vertical"
            elif col_distance > row_distance:
                dominance = "horizontal"
            else:
                dominance = "equal"

            expected = _expected_from_input(
                input_grid,
                departure_color,
                destination_color,
                common_connector,
            )
            assert np.array_equal(expected, pair["output"])
            assert len(
                _connector_cells(departure, destination)
            ) >= 2
            assert not _is_original_source_input(input_grid)

            if section == "train":
                train_row_signs.add(direction[0])
                train_col_signs.add(direction[1])
                train_dominances.add(dominance)
    assert train_row_signs == {-1, 1}
    assert train_col_signs == {-1, 1}

    # The training set demonstrates both possible dominant-axis behaviors.
    assert "vertical" in train_dominances
    assert "horizontal" in train_dominances

    return {
        "backgrounds": frozenset(backgrounds),
        "departure": departure_color,
        "destination": destination_color,
        "connector": common_connector,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    input_keys: set[tuple] = set()
    pair_keys: set[tuple] = set()

    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict)
        assert set(pair.keys()) == {"input", "output"}

        for grid in (pair["input"], pair["output"]):
            assert isinstance(grid, np.ndarray)
            assert grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30
            assert 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((grid >= 0) & (grid <= 9))

        assert pair["input"].shape == pair["output"].shape
        assert pair["input"].dtype == pair["output"].dtype

        input_key = _grid_key(pair["input"])
        pair_key = _pair_key(pair)

        assert input_key not in input_keys
        assert pair_key not in pair_keys

        input_keys.add(input_key)
        pair_keys.add(pair_key)

    _analyze_task(puzzle)


def _puzzle_key(puzzle: dict) -> tuple:
    return (
        tuple(_pair_key(pair) for pair in puzzle["train"]),
        tuple(_pair_key(pair) for pair in puzzle["test"]),
    )


def _puzzles_equal(
    first: dict,
    second: dict,
) -> bool:
    if set(first.keys()) != {"train", "test"}:
        return False
    if set(second.keys()) != {"train", "test"}:
        return False

    for section in ("train", "test"):
        if len(first[section]) != len(second[section]):
            return False

        for left, right in zip(first[section], second[section]):
            if not np.array_equal(
                left["input"],
                right["input"],
            ):
                return False
            if not np.array_equal(
                left["output"],
                right["output"],
            ):
                return False

    return True


def self_test() -> None:
    automatic: list[dict] = []

    # Forty-two fixed automatic-count seeds.
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        automatic.append(puzzle)

    assert len(
        {len(puzzle["train"]) for puzzle in automatic}
    ) >= 4
    assert len(
        {len(puzzle["test"]) for puzzle in automatic}
    ) >= 3

    # Whole-puzzle determinism with explicit counts.
    deterministic_a = generate_puzzle(918273, 7, 3)
    deterministic_b = generate_puzzle(918273, 7, 3)
    assert _puzzles_equal(deterministic_a, deterministic_b)

    # Explicit counts, including all requested boundary cases.
    explicit_cases = (
        (2, 1),
        (10, 1),
        (10, 4),
        (6, 2),
        (7, 3),
    )
    for index, (train_count, test_count) in enumerate(
        explicit_cases
    ):
        puzzle = generate_puzzle(
            2000 + index,
            train_count,
            test_count,
        )
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)

    # Partial count specification.
    partial_cases = (
        (3000, None, 1),
        (3001, None, 4),
        (3002, 2, None),
        (3003, 10, None),
    )
    for seed, requested_train, requested_test in partial_cases:
        puzzle = generate_puzzle(
            seed,
            requested_train,
            requested_test,
        )
        if requested_train is not None:
            assert len(puzzle["train"]) == requested_train
        if requested_test is not None:
            assert len(puzzle["test"]) == requested_test
        validate_generated_puzzle(puzzle)

    # Illegal and impossible counts.
    illegal_cases = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
        {"num_train": True, "num_test": 1},
    )
    for arguments in illegal_cases:
        try:
            generate_puzzle(4000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "illegal count arguments were accepted"
            )

    palettes: set[tuple[int, int, int]] = set()
    shapes: set[tuple[int, int]] = set()
    layouts: set[tuple] = set()
    path_lengths: set[int] = set()
    directions: set[tuple[int, int]] = set()
    dominances: set[str] = set()
    puzzle_keys: set[tuple] = set()
    test_directions: set[tuple[int, int]] = set()
    test_dominances: set[str] = set()
    test_surpluses: set[int] = set()
    multi_background_puzzles = 0
    wide_tests = square_tests = tall_tests = 0
    held_out_test_puzzles = shown_test_puzzles = 0
    test_combinations: set[tuple[str, str]] = set()
    explicit_reflections = 0
    small_canvases = near_limit_canvases = 0
    long_test_diagonals = 0
    up_left_horizontal = up_left_equal = 0

    for puzzle in automatic:
        roles = _analyze_task(puzzle)
        palettes.add(
            (
                roles["departure"],
                roles["destination"],
                roles["connector"],
            )
        )
        if len(roles["backgrounds"]) > 1:
            multi_background_puzzles += 1
        puzzle_keys.add(_puzzle_key(puzzle))

        local_inputs: set[tuple] = set()
        local_pairs: set[tuple] = set()
        local_train_directions: set[tuple[int, int]] = set()
        local_test_directions: set[tuple[int, int]] = set()

        for section in ("train", "test"):
            for pair in puzzle[section]:
                input_grid = pair["input"]
                output_grid = pair["output"]

                departure = _marker_position(
                    input_grid,
                    roles["departure"],
                )
                destination = _marker_position(
                    input_grid,
                    roles["destination"],
                )

                row_delta = destination[0] - departure[0]
                col_delta = destination[1] - departure[1]
                direction = (
                    _sign(row_delta),
                    _sign(col_delta),
                )

                row_distance = abs(row_delta)
                col_distance = abs(col_delta)
                if row_distance > col_distance:
                    dominance = "vertical"
                elif col_distance > row_distance:
                    dominance = "horizontal"
                else:
                    dominance = "equal"

                # Independently reconstruct every ground-truth output.
                expected = _formula_expected(
                    input_grid,
                    roles["departure"],
                    roles["destination"],
                    roles["connector"],
                )
                assert np.array_equal(expected, output_grid)

                assert input_grid.shape == output_grid.shape
                assert np.issubdtype(
                    input_grid.dtype,
                    np.integer,
                )
                assert np.issubdtype(
                    output_grid.dtype,
                    np.integer,
                )
                assert np.all(
                    (input_grid >= 0) & (input_grid <= 9)
                )
                assert np.all(
                    (output_grid >= 0) & (output_grid <= 9)
                )
                assert not _is_original_source_input(input_grid)

                input_key = _grid_key(input_grid)
                pair_key = _pair_key(pair)
                assert input_key not in local_inputs
                assert pair_key not in local_pairs
                local_inputs.add(input_key)
                local_pairs.add(pair_key)

                shapes.add(
                    tuple(int(x) for x in input_grid.shape)
                )
                layouts.add(
                    (
                        input_grid.shape,
                        departure,
                        destination,
                    )
                )
                path_lengths.add(
                    len(
                        _connector_cells(
                            departure,
                            destination,
                        )
                    )
                )
                directions.add(direction)
                dominances.add(dominance)
                if max(input_grid.shape) <= 16:
                    small_canvases += 1
                if max(input_grid.shape) >= 29:
                    near_limit_canvases += 1
                if section == "train":
                    local_train_directions.add(direction)
                else:
                    local_test_directions.add(direction)
                    test_directions.add(direction)
                    test_dominances.add(dominance)
                    test_surpluses.add(abs(row_distance - col_distance))
                    height, width = input_grid.shape
                    if height < width:
                        wide_tests += 1
                        aspect = "wide"
                    elif height > width:
                        tall_tests += 1
                        aspect = "tall"
                    else:
                        square_tests += 1
                        aspect = "square"
                    test_combinations.add((aspect, dominance))
                    if min(row_distance, col_distance) >= 15:
                        long_test_diagonals += 1
                    if direction == (-1, -1) and dominance == "horizontal":
                        up_left_horizontal += 1
                    if direction == (-1, -1) and dominance == "equal":
                        up_left_equal += 1
                    if any(
                        np.array_equal(input_grid, np.fliplr(train["input"]))
                        for train in puzzle["train"]
                    ):
                        explicit_reflections += 1

        if local_test_directions - local_train_directions:
            held_out_test_puzzles += 1
        if local_test_directions & local_train_directions:
            shown_test_puzzles += 1

    # Different seeds normally produce different complete puzzles.
    assert len(puzzle_keys) >= 40

    # Substantial puzzle-level and structural diversity.
    assert len(palettes) >= 20
    assert len(shapes) >= 30
    assert len(layouts) >= 200
    assert len(path_lengths) >= 10
    assert multi_background_puzzles >= 30
    assert held_out_test_puzzles >= 20
    assert shown_test_puzzles >= 15
    assert min(wide_tests, square_tests, tall_tests) >= 10
    assert test_directions == set(_DIRECTIONS)
    assert test_dominances == set(_DOMINANCES)
    assert 1 in test_surpluses
    assert max(test_surpluses) >= 10
    assert len(test_combinations) == 9
    assert explicit_reflections >= 1
    assert small_canvases >= 20
    assert near_limit_canvases >= 40
    assert long_test_diagonals >= 10
    assert up_left_horizontal >= 1
    assert up_left_equal >= 1

    assert directions == set(_DIRECTIONS)

    # Balanced paths are permitted in additional training examples, while
    # both dominant-axis cases are always demonstrated.
    assert dominances == {
        "vertical",
        "horizontal",
        "equal",
    }
