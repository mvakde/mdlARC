import numpy as np


_DIRECTIONS = (
    (-1, 0),  # north
    (0, 1),   # east
    (1, 0),   # south
    (0, -1),  # west
)

_SOURCE_PAIR_TEXT = {
    (
        "000000/000000/300080/000000/000000/000000",
        "000300/000300/333380/000000/000000/000000",
    ),
    (
        "000000/000000/000000/300060/000000/000000/000800/000000",
        "000000/000000/000000/333360/000300/000333/000800/000000",
    ),
    (
        "00000000/00000000/30006000/00000000/00000006/00080000/"
        "00000000/00000000",
        "00000000/00000000/33336000/00030000/00033336/00080030/"
        "00000030/00000030",
    ),
    (
        "00600000/00000080/00000000/00000000/30080000/00000000/"
        "00000000/00000000/00000000",
        "00600300/00333380/00300000/00300000/33380000/00000000/"
        "00000000/00000000/00000000",
    ),
    (
        "000000/000000/300600/000000/000000/000000",
        "000000/000000/333600/003000/003000/003000",
    ),
    (
        "000000000600/000000600000/000000000080/006000000000/"
        "000000080000/000000000000/000000000000/300800000000/"
        "000000000000/000000000000",
        "000000000600/000000600333/000000333380/006000300000/"
        "003333380000/003000000000/003000000000/333800000000/"
        "000000000000/000000000000",
    ),
}


def _ensure(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _checked_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )
    return result


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _checked_count("num_train", num_train, 2, 10)
    test_count = _checked_count("num_test", num_test, 1, 4)

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = 1 + (seed % 4)
        raw_train_count = 2 + ((seed // 4) % 9)
        train_count = max(test_count + 1, raw_train_count)
        return train_count, test_count

    if train_count is None:
        assert test_count is not None
        train_count = int(rng.integers(test_count + 1, 11))
        return train_count, test_count

    maximum_test = min(4, train_count - 1)
    test_count = int(rng.integers(1, maximum_test + 1))
    return train_count, test_count


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    colors = [int(value) for value in rng.permutation(10)[:4]]
    return dict(zip(
        ("background", "path", "clockwise", "counterclockwise"),
        colors,
    ))


def _step(
    position: tuple[int, int],
    direction: tuple[int, int],
    distance: int = 1,
) -> tuple[int, int]:
    return (
        position[0] + direction[0] * distance,
        position[1] + direction[1] * distance,
    )


def _distance_to_edge(
    position: tuple[int, int],
    direction: tuple[int, int],
    height: int,
    width: int,
) -> int:
    row, column = position

    if direction == (-1, 0):
        return row
    if direction == (1, 0):
        return height - 1 - row
    if direction == (0, -1):
        return column
    if direction == (0, 1):
        return width - 1 - column
    raise AssertionError("invalid direction")


def _initial_direction(
    emitter: tuple[int, int],
    height: int,
    width: int,
) -> tuple[int, int]:
    row, column = emitter
    candidates = []

    if row == 0:
        candidates.append((0, 2))
    if column == width - 1:
        candidates.append((1, 3))
    if row == height - 1:
        candidates.append((2, 0))
    if column == 0:
        candidates.append((3, 1))

    _ensure(
        len(candidates) == 1,
        "the emitter must occupy exactly one non-corner boundary",
    )
    return candidates[0]


def _choose_emitter(
    rng: np.random.Generator,
    height: int,
    width: int,
    side: int,
) -> tuple[tuple[int, int], int]:
    if side == 0:
        return (0, int(rng.integers(1, width - 1))), 2
    if side == 1:
        return (int(rng.integers(1, height - 1)), width - 1), 3
    if side == 2:
        return (height - 1, int(rng.integers(1, width - 1))), 0
    if side == 3:
        return (int(rng.integers(1, height - 1)), 0), 1
    raise ValueError("side must be from 0 through 3")


def _sample_role_sequence(
    rng: np.random.Generator,
    marker_count: int,
    require_both: bool,
) -> list[str]:
    if require_both and marker_count < 2:
        raise ValueError("at least two markers are required to show both roles")

    first = "cw" if int(rng.integers(0, 2)) == 0 else "ccw"
    other = "ccw" if first == "cw" else "cw"
    mode = int(rng.integers(0, 4))
    if mode == 0:
        roles = [first] * marker_count
    elif mode == 1:
        roles = [first if index % 2 == 0 else other
                 for index in range(marker_count)]
    elif mode == 2:
        roles = [first if index < marker_count // 2 else other
                 for index in range(marker_count)]
    else:
        roles = ["cw" if int(rng.integers(0, 2)) == 0 else "ccw"
                 for _ in range(marker_count)]

    if require_both and len(set(roles)) == 1:
        index = int(rng.integers(1, marker_count))
        roles[index] = "ccw" if roles[0] == "cw" else "cw"

    return roles


def _trace_output(
    input_grid: np.ndarray,
    background_color: int,
    path_color: int,
    clockwise_color: int,
    counterclockwise_color: int,
) -> np.ndarray:
    height, width = input_grid.shape

    emitter_locations = np.argwhere(input_grid == path_color)
    _ensure(
        len(emitter_locations) == 1,
        "an input must have exactly one path-colored emitter",
    )
    emitter = tuple(int(value) for value in emitter_locations[0])
    _, direction_index = _initial_direction(emitter, height, width)

    marker_mask = (
        (input_grid == clockwise_color)
        | (input_grid == counterclockwise_color)
    )
    remaining = {
        tuple(int(value) for value in location):
        int(input_grid[tuple(location)])
        for location in np.argwhere(marker_mask)
    }
    _ensure(bool(remaining), "at least one turn marker is required")

    output_grid = input_grid.copy()
    current = emitter
    traversed = {emitter}

    while remaining:
        direction = _DIRECTIONS[direction_index]
        aligned = []

        for marker in remaining:
            if direction[0] != 0:
                if marker[1] != current[1]:
                    continue
                distance = (marker[0] - current[0]) * direction[0]
            else:
                if marker[0] != current[0]:
                    continue
                distance = (marker[1] - current[1]) * direction[1]

            if distance > 0:
                aligned.append((int(distance), marker))

        _ensure(
            bool(aligned),
            "every remaining marker must belong to the directed chain",
        )

        distance, marker = min(
            aligned,
            key=lambda item: (item[0], item[1]),
        )
        _ensure(
            distance >= 2,
            "a marker must have at least one background cell before it",
        )

        for offset in range(1, distance):
            cell = _step(current, direction, offset)
            _ensure(
                cell not in traversed,
                "the generated route may not self-intersect",
            )
            _ensure(
                int(input_grid[cell]) == background_color,
                "a route segment must traverse only background",
            )
            output_grid[cell] = path_color
            traversed.add(cell)

        current = _step(current, direction, distance - 1)
        marker_color = remaining.pop(marker)

        if marker_color == clockwise_color:
            direction_index = (direction_index + 1) % 4
        else:
            _ensure(
                marker_color == counterclockwise_color,
                "unknown marker color",
            )
            direction_index = (direction_index - 1) % 4

    direction = _DIRECTIONS[direction_index]
    exit_distance = _distance_to_edge(
        current,
        direction,
        height,
        width,
    )
    _ensure(
        exit_distance >= 1,
        "the final turn must leave a nonempty route to the boundary",
    )

    for offset in range(1, exit_distance + 1):
        cell = _step(current, direction, offset)
        _ensure(
            cell not in traversed,
            "the final ray may not self-intersect",
        )
        _ensure(
            int(input_grid[cell]) == background_color,
            "the final ray must traverse only background",
        )
        output_grid[cell] = path_color
        traversed.add(cell)

    return output_grid


def _try_layout(
    rng: np.random.Generator,
    height: int,
    width: int,
    roles: list[str],
    side: int,
    maximum_segment: int,
) -> dict | None:
    for _ in range(180):
        emitter, direction_index = _choose_emitter(
            rng,
            height,
            width,
            side,
        )
        current = emitter
        path_cells = {emitter}
        markers: list[tuple[tuple[int, int], str]] = []
        successful = True

        for role_index, role in enumerate(roles):
            direction = _DIRECTIONS[direction_index]
            available = _distance_to_edge(
                current,
                direction,
                height,
                width,
            )
            maximum_length = min(available - 1, maximum_segment)

            if maximum_length < 1:
                successful = False
                break

            candidate_lengths = np.arange(1, maximum_length + 1)
            rng.shuffle(candidate_lengths)
            chosen = None
            marker_positions = {position for position, _ in markers}

            for raw_length in candidate_lengths:
                length = int(raw_length)
                segment = [
                    _step(current, direction, offset)
                    for offset in range(1, length + 1)
                ]
                marker = _step(current, direction, length + 1)

                if any(
                    cell in path_cells or cell in marker_positions
                    for cell in segment
                ):
                    continue
                if marker in path_cells or marker in marker_positions:
                    continue
                if max(
                    abs(marker[0] - emitter[0]),
                    abs(marker[1] - emitter[1]),
                ) <= 1:
                    continue
                if any(
                    max(
                        abs(marker[0] - previous[0]),
                        abs(marker[1] - previous[1]),
                    ) <= 1
                    for previous, _ in markers
                ):
                    continue

                turn = 1 if role == "cw" else -1
                next_direction_index = (direction_index + turn) % 4
                next_direction = _DIRECTIONS[next_direction_index]
                needed_room = 2 if role_index + 1 < len(roles) else 1

                if _distance_to_edge(
                    segment[-1],
                    next_direction,
                    height,
                    width,
                ) < needed_room:
                    continue

                chosen = (
                    segment,
                    marker,
                    next_direction_index,
                )
                break

            if chosen is None:
                successful = False
                break

            segment, marker, direction_index = chosen
            path_cells.update(segment)
            markers.append((marker, role))
            current = segment[-1]

        if not successful:
            continue

        final_direction = _DIRECTIONS[direction_index]
        final_distance = _distance_to_edge(
            current,
            final_direction,
            height,
            width,
        )
        if final_distance < 1:
            continue

        final_segment = [
            _step(current, final_direction, offset)
            for offset in range(1, final_distance + 1)
        ]
        marker_positions = {position for position, _ in markers}

        if any(
            cell in path_cells or cell in marker_positions
            for cell in final_segment
        ):
            continue

        path_cells.update(final_segment)

        return {
            "emitter": emitter,
            "markers": markers,
            "path": path_cells,
        }

    return None


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    puzzle_parameters: dict,
    side: int,
    forced_roles: list[str] | None = None,
    marker_count: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in {"source_train", "target_test"}:
        raise ValueError("unknown generation regime")

    for _ in range(800):
        if regime == "source_train":
            require_both = False
            default_marker_count = int(rng.integers(1, 7))
        else:
            require_both = False
            default_marker_count = int(rng.integers(1, 9))

        if forced_roles is not None:
            roles = list(forced_roles)
        else:
            count = (
                default_marker_count
                if marker_count is None
                else marker_count
            )
            roles = _sample_role_sequence(
                rng,
                count,
                require_both=require_both,
            )

        upper = 19 if regime == "source_train" else 26
        lower = 6 if len(roles) <= 4 else 9
        height = int(rng.integers(lower, upper))
        width = int(rng.integers(lower, upper))
        maximum_segment = max(height, width)

        layout = _try_layout(
            rng,
            height,
            width,
            roles,
            side,
            maximum_segment,
        )
        if layout is None:
            continue

        background = puzzle_parameters["background"]
        path = puzzle_parameters["path"]
        clockwise = puzzle_parameters["clockwise"]
        counterclockwise = puzzle_parameters["counterclockwise"]

        input_grid = np.full(
            (height, width),
            background,
            dtype=np.int8,
        )
        input_grid[layout["emitter"]] = path

        for marker, role in layout["markers"]:
            input_grid[marker] = (
                clockwise if role == "cw" else counterclockwise
            )

        output_grid = input_grid.copy()
        for cell in layout["path"]:
            output_grid[cell] = path

        try:
            reconstructed = _trace_output(
                input_grid,
                background,
                path,
                clockwise,
                counterclockwise,
            )
        except AssertionError:
            continue

        if not np.array_equal(reconstructed, output_grid):
            continue
        if np.array_equal(input_grid, output_grid):
            continue

        return input_grid, output_grid

    # A simple directed turn is a deterministic emergency construction for
    # crowded random layouts. The side and learned marker meaning persist.
    height, width = 9, 10
    fallback_emitters = ((0, 3), (3, width - 1),
                         (height - 1, 3), (3, 0))
    emitter = fallback_emitters[side]
    direction = _DIRECTIONS[(side + 2) % 4]
    marker = _step(emitter, direction, 4)
    background = puzzle_parameters["background"]
    path = puzzle_parameters["path"]
    role = (forced_roles or roles)[0]
    marker_color = puzzle_parameters[
        "clockwise" if role == "cw" else "counterclockwise"
    ]
    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[emitter] = path
    input_grid[marker] = marker_color
    return input_grid, _trace_output(
        input_grid, background, path,
        puzzle_parameters["clockwise"],
        puzzle_parameters["counterclockwise"],
    )


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(value) for value in grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_key(pair: dict) -> tuple:
    return (
        _grid_key(pair["input"]),
        _grid_key(pair["output"]),
    )


def _grid_text(grid: np.ndarray) -> str:
    return "/".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        (
            split,
            _grid_key(pair["input"]),
            _grid_key(pair["output"]),
        )
        for split in ("train", "test")
        for pair in puzzle[split]
    )


def _infer_conventions(train_pairs: list[dict]) -> dict:
    pair_roles = [_infer_pair_colors(pair) for pair in train_pairs]
    marker_sets = [set(roles[2]) for roles in pair_roles]
    marker_colors = sorted(set().union(*marker_sets))
    _ensure(len(marker_colors) == 2,
            "training must demonstrate two marker colors")
    _ensure(all(not (set(roles[:2]) & set(marker_colors))
                for roles in pair_roles),
            "background and path must be distinct from markers")

    valid_assignments = []
    for clockwise, counterclockwise in (
        (marker_colors[0], marker_colors[1]),
        (marker_colors[1], marker_colors[0]),
    ):
        try:
            valid = all(
                np.array_equal(
                    _trace_output(pair["input"], background, path,
                                  clockwise, counterclockwise),
                    pair["output"],
                )
                for pair, (background, path, _) in zip(train_pairs, pair_roles)
            )
        except AssertionError:
            valid = False
        if valid:
            valid_assignments.append((clockwise, counterclockwise))
    _ensure(len(valid_assignments) == 1,
            "training must uniquely determine both turn roles")
    clockwise, counterclockwise = valid_assignments[0]
    return {"clockwise": clockwise, "counterclockwise": counterclockwise}


def _infer_pair_colors(pair: dict) -> tuple[int, int, tuple[int, ...]]:
    input_grid, output_grid = pair["input"], pair["output"]
    _ensure(input_grid.shape == output_grid.shape,
            "input and output dimensions must match")
    values, counts = np.unique(input_grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    _ensure(int(np.count_nonzero(input_grid == background))
            > input_grid.size // 2, "background must dominate")
    changed = input_grid != output_grid
    _ensure(bool(np.any(changed)), "path must grow")
    _ensure(bool(np.all(input_grid[changed] == background)),
            "only background may change")
    grown = np.unique(output_grid[changed])
    _ensure(len(grown) == 1, "exactly one path color must grow")
    path = int(grown[0])
    _ensure(path != background, "path and background must differ")
    _ensure(int(np.count_nonzero(input_grid == path)) == 1,
            "input must contain one emitter")
    markers = tuple(sorted(int(color) for color in values
                           if int(color) not in {background, path}))
    _ensure(1 <= len(markers) <= 2, "one or two marker colors required")
    return background, path, markers


def _validate_task_pair(
    pair: dict,
    regime: str,
    conventions: dict,
) -> None:
    input_grid = pair["input"]
    output_grid = pair["output"]
    height, width = input_grid.shape

    background, path, pair_markers = _infer_pair_colors(pair)
    clockwise = conventions["clockwise"]
    counterclockwise = conventions["counterclockwise"]
    allowed_colors = {
        background,
        path,
        clockwise,
        counterclockwise,
    }
    _ensure(len(allowed_colors) == 4,
            "each pair has four distinct semantic color roles")
    _ensure(set(pair_markers) <= {clockwise, counterclockwise},
            "marker colors must retain their puzzle-level meaning")

    _ensure(
        input_grid.shape == output_grid.shape,
        "input and output dimensions must be preserved",
    )
    _ensure(
        set(int(value) for value in np.unique(input_grid))
        <= allowed_colors,
        "input contains an unknown color",
    )
    _ensure(
        set(int(value) for value in np.unique(output_grid))
        <= allowed_colors,
        "output contains an unknown color",
    )
    _ensure(
        set(int(value) for value in np.unique(input_grid))
        == set(int(value) for value in np.unique(output_grid)),
        "the transformation must preserve the pair's palette",
    )
    _ensure(
        int(np.count_nonzero(input_grid == background))
        > input_grid.size // 2,
        "background must dominate each input",
    )
    _ensure(
        int(np.count_nonzero(input_grid == path)) == 1,
        "each input must have exactly one emitter",
    )

    emitter = tuple(
        int(value)
        for value in np.argwhere(input_grid == path)[0]
    )
    _initial_direction(emitter, height, width)

    marker_mask = (
        (input_grid == clockwise)
        | (input_grid == counterclockwise)
    )
    marker_positions = [
        tuple(int(value) for value in location)
        for location in np.argwhere(marker_mask)
    ]
    marker_total = len(marker_positions)

    for index, marker in enumerate(marker_positions):
        _ensure(
            max(
                abs(marker[0] - emitter[0]),
                abs(marker[1] - emitter[1]),
            ) > 1,
            "markers must be isolated from the emitter",
        )
        for other in marker_positions[index + 1:]:
            _ensure(
                max(
                    abs(marker[0] - other[0]),
                    abs(marker[1] - other[1]),
                ) > 1,
                "input markers must be isolated from each other",
            )

    _ensure(regime in {"source_train", "target_test"},
            "unknown validation regime")
    _ensure(6 <= height <= 25 and 6 <= width <= 25,
            "pair dimensions must fit the source-supported range")
    _ensure(1 <= marker_total <= 8,
            "pair must have one to eight markers")

    expected = _trace_output(
        input_grid,
        background,
        path,
        clockwise,
        counterclockwise,
    )
    _ensure(
        np.array_equal(expected, output_grid),
        "output is not the exact marker-directed path trace",
    )
    _ensure(
        not np.array_equal(input_grid, output_grid),
        "generated pairs must not be degenerate",
    )



def _validate_generated_puzzle_impl(puzzle: dict) -> None:
    _ensure(isinstance(puzzle, dict), "puzzle must be a dictionary")
    _ensure(
        set(puzzle.keys()) == {"train", "test"},
        "puzzle must contain exactly train and test",
    )
    _ensure(
        isinstance(puzzle["train"], list)
        and isinstance(puzzle["test"], list),
        "train and test must be lists",
    )

    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])
    _ensure(2 <= train_count <= 10, "illegal training pair count")
    _ensure(1 <= test_count <= 4, "illegal test pair count")
    _ensure(train_count > test_count, "train count must exceed test count")

    input_keys = set()
    pair_keys = set()

    for split in ("train", "test"):
        for pair in puzzle[split]:
            _ensure(isinstance(pair, dict), "each pair must be a dictionary")
            _ensure(
                set(pair.keys()) == {"input", "output"},
                "each pair must contain exactly input and output",
            )

            for grid_name in ("input", "output"):
                grid = pair[grid_name]
                _ensure(
                    isinstance(grid, np.ndarray),
                    f"{grid_name} must be a NumPy array",
                )
                _ensure(grid.ndim == 2, "every grid must be two-dimensional")
                height, width = grid.shape
                _ensure(
                    1 <= height <= 30 and 1 <= width <= 30,
                    "grid dimensions must be from 1 through 30",
                )
                _ensure(
                    np.issubdtype(grid.dtype, np.integer),
                    "grids must use an integer dtype",
                )
                _ensure(
                    bool(np.all((grid >= 0) & (grid <= 9))),
                    "grid colors must be from 0 through 9",
                )

            input_key = _grid_key(pair["input"])
            pair_key = _pair_key(pair)
            _ensure(
                input_key not in input_keys,
                "duplicate input grid found",
            )
            _ensure(
                pair_key not in pair_keys,
                "duplicate input/output pair found",
            )
            input_keys.add(input_key)
            pair_keys.add(pair_key)

            source_signature = (
                _grid_text(pair["input"]),
                _grid_text(pair["output"]),
            )
            _ensure(
                source_signature not in _SOURCE_PAIR_TEXT,
                "a generated pair exactly duplicates a source pair",
            )

    conventions = _infer_conventions(puzzle["train"])

    for pair in puzzle["train"]:
        _validate_task_pair(pair, "source_train", conventions)
    for pair in puzzle["test"]:
        _validate_task_pair(pair, "target_test", conventions)

    # Both marker roles must be demonstrated; their mapping is established
    # by the unique training-consistent trace in _infer_conventions.
    for marker_color in conventions.values():
        _ensure(any(np.any(pair["input"] == marker_color)
                    for pair in puzzle["train"]),
                "every test marker role must be taught in training")


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate_generated_puzzle_impl(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError(
            f"malformed or invalid generated puzzle: {exc}"
        ) from exc


def _new_pair_rng(rng: np.random.Generator) -> np.random.Generator:
    child_seed = int(
        rng.integers(0, np.iinfo(np.int64).max)
    )
    return np.random.default_rng(child_seed)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, (int, np.integer))
        or int(seed) < 0
    ):
        raise ValueError("seed must be a nonnegative integer")

    seed_value = int(seed)
    rng = np.random.default_rng(seed_value)

    train_count, test_count = _resolve_counts(
        seed_value,
        rng,
        num_train,
        num_test,
    )
    parameters = sample_puzzle_parameters(rng)
    side_mode = int(rng.integers(0, 3))
    base_side = int(rng.integers(0, 4))
    if side_mode == 0:
        train_sides = [base_side] * train_count
        test_sides = [base_side] * test_count
    elif side_mode == 1:
        train_sides = [base_side, (base_side + int(rng.integers(1, 4))) % 4]
        train_sides.extend(int(rng.integers(0, 4))
                           for _ in range(train_count - 2))
        test_sides = [int(rng.integers(0, 4)) for _ in range(test_count)]
    else:
        held_out = (base_side + int(rng.integers(1, 4))) % 4
        train_sides = [base_side] * train_count
        test_sides = [held_out] * test_count

    marker_colors = {parameters["clockwise"], parameters["counterclockwise"]}
    available_pair_colors = [color for color in range(10)
                             if color not in marker_colors]
    recolor = bool(rng.integers(0, 2))

    def pair_parameters(index: int) -> dict:
        local = parameters.copy()
        if recolor and index:
            background, path = (int(value) for value in
                                rng.choice(available_pair_colors, 2,
                                           replace=False))
            local["background"] = background
            local["path"] = path
        return local

    puzzle = {
        "train": [],
        "test": [],
    }
    used_inputs = set()

    for index in range(train_count):
        if index == 0:
            forced_roles = ["cw"]
            marker_count = None
        elif index == 1:
            forced_roles = ["ccw"]
            marker_count = None
        else:
            forced_roles = None
            marker_count = int(rng.integers(1, 7))

        side = train_sides[index]
        local_parameters = pair_parameters(index)

        for _ in range(60):
            pair_rng = _new_pair_rng(rng)
            input_grid, output_grid = _generate_pair(
                pair_rng,
                "source_train",
                local_parameters,
                side,
                forced_roles=forced_roles,
                marker_count=marker_count,
            )
            input_key = _grid_key(input_grid)

            if (input_key not in used_inputs
                    and (_grid_text(input_grid), _grid_text(output_grid))
                    not in _SOURCE_PAIR_TEXT):
                used_inputs.add(input_key)
                puzzle["train"].append(
                    {
                        "input": input_grid,
                        "output": output_grid,
                    }
                )
                break
        else:
            raise ValueError("unable to produce a unique training pair")

    for index in range(test_count):
        marker_count = int(rng.integers(1, 4)) if bool(rng.integers(0, 3) == 0) else int(rng.integers(4, 9))
        side = test_sides[index]
        local_parameters = pair_parameters(train_count + index)

        for _ in range(60):
            pair_rng = _new_pair_rng(rng)
            input_grid, output_grid = _generate_pair(
                pair_rng,
                "target_test",
                local_parameters,
                side,
                marker_count=marker_count,
            )
            input_key = _grid_key(input_grid)

            if (input_key not in used_inputs
                    and (_grid_text(input_grid), _grid_text(output_grid))
                    not in _SOURCE_PAIR_TEXT):
                used_inputs.add(input_key)
                puzzle["test"].append(
                    {
                        "input": input_grid,
                        "output": output_grid,
                    }
                )
                break
        else:
            raise ValueError("unable to produce a unique test pair")

    validate_generated_puzzle(puzzle)
    return puzzle


def _layout_signature(
    input_grid: np.ndarray,
    conventions: dict,
    regime: str,
) -> tuple:
    marker_colors = {conventions["clockwise"],
                     conventions["counterclockwise"]}
    values, counts = np.unique(input_grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    path = next(int(value) for value in values
                if int(value) not in marker_colors | {background})
    emitter = tuple(
        int(value)
        for value in np.argwhere(
            input_grid == path
        )[0]
    )
    side, _ = _initial_direction(
        emitter,
        input_grid.shape[0],
        input_grid.shape[1],
    )
    clockwise_positions = tuple(
        sorted(
            tuple(int(value) for value in location)
            for location in np.argwhere(
                input_grid == conventions["clockwise"]
            )
        )
    )
    counterclockwise_positions = tuple(
        sorted(
            tuple(int(value) for value in location)
            for location in np.argwhere(
                input_grid == conventions["counterclockwise"]
            )
        )
    )

    return (
        regime,
        input_grid.shape,
        side,
        emitter,
        clockwise_positions,
        counterclockwise_positions,
    )


def _review_features(pair: dict, conventions: dict) -> dict:
    grid, output = pair["input"], pair["output"]
    background, path, _ = _infer_pair_colors(pair)
    height, width = grid.shape
    emitter = tuple(int(v) for v in np.argwhere(grid == path)[0])
    side, direction_index = _initial_direction(emitter, height, width)
    markers = {
        tuple(int(v) for v in location): int(grid[tuple(location)])
        for location in np.argwhere(
            (grid == conventions["clockwise"])
            | (grid == conventions["counterclockwise"])
        )
    }
    remaining = markers.copy()
    current = emitter
    roles = []
    lengths = []
    while remaining:
        direction = _DIRECTIONS[direction_index]
        forward = []
        for marker in remaining:
            if direction[0] and marker[1] == current[1]:
                distance = (marker[0] - current[0]) * direction[0]
            elif direction[1] and marker[0] == current[0]:
                distance = (marker[1] - current[1]) * direction[1]
            else:
                continue
            if distance > 0:
                forward.append((distance, marker))
        _ensure(bool(forward), "review measurement found a broken chain")
        distance, marker = min(forward)
        lengths.append(distance - 1)
        current = _step(current, direction, distance - 1)
        role = "cw" if remaining.pop(marker) == conventions["clockwise"] else "ccw"
        roles.append(role)
        direction_index = (direction_index + (1 if role == "cw" else -1)) % 4

    near_pass = any(
        sum(
            0 <= marker[0] + direction[0] < height
            and 0 <= marker[1] + direction[1] < width
            and int(output[_step(marker, direction)]) == path
            for direction in _DIRECTIONS
        ) > 1
        for marker in markers
    )
    positions = np.array(list(markers) + [emitter])
    span = tuple(int(np.ptp(positions[:, axis])) + 1 for axis in (0, 1))
    return {
        "shape": (height, width),
        "side": side,
        "colors": (background, path),
        "roles": tuple(roles),
        "long_gap": max(lengths) >= 7,
        "near_pass": near_pass,
        "span": span,
    }


def self_test() -> None:
    automatic_puzzles = []
    train_counts = set()
    test_counts = set()
    puzzle_signatures = set()
    palettes = set()
    shapes = set()
    layouts = set()
    test_marker_counts = set()
    review_counts = {
        key: 0 for key in (
            "near_pass", "train_long", "test_short", "test_long",
            "long_gap", "test_same_role", "test_alternating",
            "small_test_axis", "mixed_test_axes", "wide_marker_span",
            "same_side_puzzle", "mixed_train_sides", "held_out_test_side",
            "pair_recolor", "within_count_variation", "within_shape_variation",
            "combined_near_and_mixed_axes",
        )
    }

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        automatic_puzzles.append(puzzle)

        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_signatures.add(_puzzle_key(puzzle))

        conventions = _infer_conventions(puzzle["train"])
        palettes.add((conventions["clockwise"],
                      conventions["counterclockwise"]))
        measured = {split: [_review_features(pair, conventions)
                            for pair in puzzle[split]]
                    for split in ("train", "test")}
        train_features, test_features = measured["train"], measured["test"]
        all_features = train_features + test_features
        train_sides = {feature["side"] for feature in train_features}
        review_counts["same_side_puzzle"] += len(
            {feature["side"] for feature in all_features}) == 1
        review_counts["mixed_train_sides"] += len(train_sides) > 1
        review_counts["held_out_test_side"] += any(
            feature["side"] not in train_sides for feature in test_features)
        review_counts["pair_recolor"] += len(
            {feature["colors"] for feature in all_features}) > 1
        review_counts["within_count_variation"] += len(
            {len(feature["roles"]) for feature in all_features}) > 1
        review_counts["within_shape_variation"] += len(
            {feature["shape"] for feature in all_features}) > 1

        for split, features in measured.items():
            for feature in features:
                review_counts["near_pass"] += feature["near_pass"]
                review_counts["long_gap"] += feature["long_gap"]
                review_counts["wide_marker_span"] += max(feature["span"]) >= 8
                if split == "train":
                    review_counts["train_long"] += len(feature["roles"]) >= 4
                else:
                    count = len(feature["roles"])
                    review_counts["test_short"] += count <= 3
                    review_counts["test_long"] += count >= 4
                    review_counts["test_same_role"] += len(set(feature["roles"])) == 1
                    review_counts["test_alternating"] += (
                        count >= 3 and all(
                            a != b for a, b in zip(feature["roles"],
                                                     feature["roles"][1:])
                        )
                    )
                    small = min(feature["shape"]) <= 10
                    mixed = small and max(feature["shape"]) >= 15
                    review_counts["small_test_axis"] += small
                    review_counts["mixed_test_axes"] += mixed
                    review_counts["combined_near_and_mixed_axes"] += (
                        mixed and feature["near_pass"])

        for split, regime in (
            ("train", "source_train"),
            ("test", "target_test"),
        ):
            for pair in puzzle[split]:
                input_grid = pair["input"]
                output_grid = pair["output"]

                shapes.add(input_grid.shape)
                layouts.add(
                    _layout_signature(
                        input_grid,
                        conventions,
                        regime,
                    )
                )
                _ensure(
                    np.issubdtype(
                        input_grid.dtype,
                        np.integer,
                    )
                    and np.issubdtype(
                        output_grid.dtype,
                        np.integer,
                    ),
                    "self-test found a non-integer grid",
                )
                _ensure(
                    input_grid.shape == output_grid.shape,
                    "self-test found a dimension change",
                )
                _ensure(
                    not np.array_equal(input_grid, output_grid),
                    "self-test found a degenerate pair",
                )
                _ensure(
                    (
                        _grid_text(input_grid),
                        _grid_text(output_grid),
                    )
                    not in _SOURCE_PAIR_TEXT,
                    "self-test found a copied source pair",
                )

                if split == "test":
                    marker_count = int(
                        np.count_nonzero(
                            (
                                input_grid
                                == conventions["clockwise"]
                            )
                            | (
                                input_grid
                                == conventions[
                                    "counterclockwise"
                                ]
                            )
                        )
                    )
                    test_marker_counts.add(marker_count)

    _ensure(
        len(train_counts) >= 4,
        "automatic generation must vary training counts",
    )
    _ensure(
        len(test_counts) >= 3,
        "automatic generation must vary test counts",
    )
    _ensure(
        len(puzzle_signatures) >= 40,
        "different seeds must usually produce different puzzles",
    )
    _ensure(
        len(palettes) >= 25,
        "42 seeds must produce substantial palette diversity",
    )
    _ensure(
        len(shapes) >= 60,
        "42 seeds must produce substantial dimensional diversity",
    )
    _ensure(
        len(layouts) >= 200,
        "42 seeds must produce substantial structural diversity",
    )
    _ensure(
        len(test_marker_counts) >= 5 and min(test_marker_counts) <= 3
        and max(test_marker_counts) >= 6,
        "test chains must cover short and long cases",
    )
    for name, threshold in (
        ("near_pass", 20), ("train_long", 20),
        ("test_short", 5), ("test_long", 20),
        ("long_gap", 30), ("test_same_role", 5),
        ("test_alternating", 10), ("small_test_axis", 8),
        ("mixed_test_axes", 5), ("wide_marker_span", 30),
        ("same_side_puzzle", 3), ("mixed_train_sides", 3),
        ("held_out_test_side", 3), ("pair_recolor", 8),
        ("within_count_variation", 40),
        ("within_shape_variation", 40),
        ("combined_near_and_mixed_axes", 2),
    ):
        _ensure(review_counts[name] >= threshold,
                f"insufficient review coverage for {name}")

    deterministic_a = generate_puzzle(9001, 7, 3)
    deterministic_b = generate_puzzle(9001, 7, 3)
    validate_generated_puzzle(deterministic_a)
    validate_generated_puzzle(deterministic_b)
    _ensure(
        _puzzle_key(deterministic_a)
        == _puzzle_key(deterministic_b),
        "fixed seeds and counts must be deterministic",
    )

    automatic_repeat = generate_puzzle(7)
    _ensure(
        _puzzle_key(automatic_repeat)
        == _puzzle_key(automatic_puzzles[7]),
        "automatic count generation must also be deterministic",
    )

    different_seed = generate_puzzle(9002, 7, 3)
    validate_generated_puzzle(different_seed)
    _ensure(
        _puzzle_key(deterministic_a)
        != _puzzle_key(different_seed),
        "different seeds should normally differ",
    )

    boundary_cases = (
        (2, 1),
        (10, 1),
        (10, 4),
    )
    for offset, (requested_train, requested_test) in enumerate(
        boundary_cases
    ):
        puzzle = generate_puzzle(
            10000 + offset,
            requested_train,
            requested_test,
        )
        validate_generated_puzzle(puzzle)
        _ensure(
            len(puzzle["train"]) == requested_train
            and len(puzzle["test"]) == requested_test,
            "explicit boundary counts were not honored",
        )

    partial_cases = (
        (None, 1),
        (None, 4),
        (2, None),
        (10, None),
    )
    for offset, (requested_train, requested_test) in enumerate(
        partial_cases
    ):
        puzzle = generate_puzzle(
            11000 + offset,
            requested_train,
            requested_test,
        )
        validate_generated_puzzle(puzzle)

        if requested_train is not None:
            _ensure(
                len(puzzle["train"]) == requested_train,
                "partial training count was not honored",
            )
        if requested_test is not None:
            _ensure(
                len(puzzle["test"]) == requested_test,
                "partial test count was not honored",
            )

    illegal_cases = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
    )
    for kwargs in illegal_cases:
        try:
            generate_puzzle(12000, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"illegal count request did not raise ValueError: {kwargs}"
            )
