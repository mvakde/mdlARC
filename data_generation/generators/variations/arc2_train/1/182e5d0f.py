import numpy as np
from itertools import permutations


_SIDE_NAMES = ("top", "bottom", "left", "right")
_NEIGHBORS = ((1, 0), (-1, 0), (0, 1), (0, -1))

_SOURCE_PAIR_ROWS = (
    (
        (
            "030777777777",
            "733333333333",
            "777777777775",
            "777777777777",
            "777777777777",
            "077777777777",
            "337777777777",
            "037777777770",
            "737777777733",
            "737777777730",
            "737777777737",
            "775777777537",
        ),
        (
            "030777777777",
            "757777777777",
            "777777777777",
            "777777777777",
            "777777777777",
            "077777777777",
            "337777777777",
            "037777777770",
            "737777777753",
            "737777777770",
            "737777777777",
            "775777777777",
        ),
    ),
    (
        (
            "777777777",
            "077777777",
            "333777777",
            "073777777",
            "773777777",
            "773777777",
            "773777777",
            "773777777",
            "773577777",
        ),
        (
            "777777777",
            "077777777",
            "357777777",
            "077777777",
            "777777777",
            "777777777",
            "777777777",
            "777777777",
            "777777777",
        ),
    ),
    (
        (
            "5370307777777",
            "7377333337777",
            "7377777777775",
            "7377777777777",
            "0377777777777",
            "3377777777777",
            "0777777777777",
            "7777777777777",
            "7777777777777",
            "7777777777777",
            "0777777777775",
            "3377773333377",
            "0357703077777",
        ),
        (
            "7770307777777",
            "7777333337777",
            "7777777777775",
            "7777777777777",
            "0777777777777",
            "3577777777777",
            "0777777777777",
            "7777777777777",
            "7777777777777",
            "7777777777777",
            "0777777777775",
            "3577773333377",
            "0777703077777",
        ),
    ),
    (
        (
            "73577777770307",
            "73777777777333",
            "73777777777775",
            "73777777777777",
            "73777777777777",
            "03777777777777",
            "33777777777777",
            "07777777777777",
            "77777777777777",
            "77777777777777",
            "07777777777777",
            "33777777777777",
            "03777777773377",
            "77577777703075",
        ),
        (
            "77777777770307",
            "77777777777577",
            "77777777777777",
            "77777777777777",
            "77777777777777",
            "07777777777777",
            "35777777777777",
            "07777777777777",
            "77777777777777",
            "77777777777777",
            "07777777777777",
            "33777777777777",
            "03777777773377",
            "77577777703075",
        ),
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _is_integer_argument(value: object) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
    )


def _validate_requested_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if not _is_integer_argument(value):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_counts(
    seed_value: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_requested_count(
        "num_train", num_train, 2, 10
    )
    test_count = _validate_requested_count(
        "num_test", num_test, 1, 4
    )

    if train_count is None and test_count is None:
        test_count = 1 + seed_value % 4
        minimum_train = max(2, test_count + 1)
        choices = list(range(minimum_train, 11))
        train_count = choices[(seed_value // 4) % len(choices)]

    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        span = 11 - minimum_train
        train_count = minimum_train + int(rng.integers(0, span))

    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count for num_train")
        test_count = 1 + int(rng.integers(0, maximum_test))

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    colors = [
        int(x)
        for x in rng.choice(
            np.arange(10, dtype=np.int8),
            size=4,
            replace=False,
        )
    ]
    return {
        "background": colors[0],
        "cap": colors[1],
        "conduit": colors[2],
        "marker": colors[3],
    }


def _add(
    first: tuple[int, int],
    second: tuple[int, int],
) -> tuple[int, int]:
    return first[0] + second[0], first[1] + second[1]


def _scale(
    direction: tuple[int, int],
    amount: int,
) -> tuple[int, int]:
    return direction[0] * amount, direction[1] * amount


def _inside(
    position: tuple[int, int],
    height: int,
    width: int,
) -> bool:
    return (
        0 <= position[0] < height
        and 0 <= position[1] < width
    )


def _manhattan(
    first: tuple[int, int],
    second: tuple[int, int],
) -> int:
    return (
        abs(first[0] - second[0])
        + abs(first[1] - second[1])
    )


def _propose_candidate(
    rng: np.random.Generator,
    height: int,
    width: int,
    side: int,
    status: str,
) -> dict | None:
    if side == 0:
        center = (0, int(rng.integers(1, width - 1)))
        inward = (1, 0)
        boundary_direction = (0, 1)
    elif side == 1:
        center = (height - 1, int(rng.integers(1, width - 1)))
        inward = (-1, 0)
        boundary_direction = (0, 1)
    elif side == 2:
        center = (int(rng.integers(1, height - 1)), 0)
        inward = (0, 1)
        boundary_direction = (1, 0)
    elif side == 3:
        center = (int(rng.integers(1, height - 1)), width - 1)
        inward = (0, -1)
        boundary_direction = (1, 0)
    else:
        raise ValueError("unknown side")

    turn_sign = -1 if int(rng.integers(0, 2)) == 0 else 1
    tangent = _scale(boundary_direction, turn_sign)

    # The two legs are sampled independently.  Short arms are common in the
    # source, but a uniform tail allows arms to span a large available canvas.
    first_limit = (height if side < 2 else width) - 2
    first_length = _sample_leg(rng, first_limit)
    turn = _add(center, _scale(inward, first_length))
    if tangent[0]:
        second_limit = height - 1 - turn[0] if tangent[0] > 0 else turn[0]
    else:
        second_limit = width - 1 - turn[1] if tangent[1] > 0 else turn[1]
    if second_limit < 1:
        return None
    second_length = _sample_leg(rng, second_limit)

    path: list[tuple[int, int]] = []
    current = center

    for _ in range(first_length):
        current = _add(current, inward)
        path.append(current)

    for _ in range(second_length):
        current = _add(current, tangent)
        path.append(current)

    cap_positions = (
        _add(center, boundary_direction),
        _add(center, _scale(boundary_direction, -1)),
    )

    structural_cells = set(path)
    structural_cells.add(center)
    structural_cells.update(cap_positions)

    if not all(
        _inside(position, height, width)
        for position in structural_cells
    ):
        return None

    endpoint = path[-1]

    if status == "bare":
        marker = None
        all_cells = structural_cells
        return {
            "side": side, "status": status, "center": center,
            "caps": cap_positions, "path": path, "endpoint": endpoint,
            "marker": marker, "cells": all_cells,
            "leg_lengths": (first_length, second_length),
        }
    if status == "complete":
        marker_offsets = list(_NEIGHBORS)
    elif status == "diagonal":
        marker_offsets = [(dr, dc) for dr in (-1, 1) for dc in (-1, 1)]
    elif status == "axial":
        marker_offsets = [
            _scale(direction, distance)
            for distance in range(2, min(8, max(height, width)))
            for direction in _NEIGHBORS
        ]
    elif status == "remote":
        marker_offsets = [
            (dr, dc)
            for dr in range(-8, 9)
            for dc in range(-8, 9)
            if 3 <= abs(dr) + abs(dc) <= 9
            and abs(dr) != abs(dc)
        ]
    else:
        raise ValueError("unknown candidate status")

    marker_choices = []
    for offset in marker_offsets:
        marker_position = _add(endpoint, offset)
        if (
            _inside(marker_position, height, width)
            and marker_position not in structural_cells
            and all(
                _manhattan(marker_position, position) > 1
                for position in path[:-1]
            )
        ):
            marker_choices.append(marker_position)

    if not marker_choices:
        return None

    marker = marker_choices[int(rng.integers(0, len(marker_choices)))]
    all_cells = set(structural_cells)
    all_cells.add(marker)

    return {
        "side": side,
        "status": status,
        "center": center,
        "caps": cap_positions,
        "path": path,
        "endpoint": endpoint,
        "marker": marker,
        "cells": all_cells,
        "leg_lengths": (first_length, second_length),
    }


def _sample_leg(rng: np.random.Generator, limit: int) -> int:
    if int(rng.integers(0, 4)) == 0:
        return int(rng.integers(1, limit + 1))
    return int(rng.integers(1, min(limit, 5) + 1))


def _compatible_candidate(
    candidate: dict,
    accepted: list[dict],
) -> bool:
    for previous in accepted:
        for new_cell in candidate["cells"]:
            for old_cell in previous["cells"]:
                if max(
                    abs(new_cell[0] - old_cell[0]),
                    abs(new_cell[1] - old_cell[1]),
                ) <= 1:
                    return False

    return True


def _sample_shape(
    rng: np.random.Generator,
    regime: str,
    object_count: int,
) -> tuple[int, int]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown generation regime")
    weights = (0.38, 0.42, 0.20) if regime == "source_train" else (0.24, 0.42, 0.34)
    ranges = ((9, 15), (12, 23), (18, 31))
    dimensions = []
    for _ in range(2):
        band = int(rng.choice(3, p=weights))
        low, high = ranges[band]
        dimensions.append(int(rng.integers(low, high)))
    # Physical space, rather than a train/test template, limits crowded grids.
    if object_count >= 4 and min(dimensions) < 12:
        dimensions[int(np.argmin(dimensions))] = int(rng.integers(12, 19))
    return dimensions[0], dimensions[1]


def generate_pair(
    rng: np.random.Generator,
    regime: str,
    puzzle_parameters: dict,
    sides: list[int],
    statuses: list[str],
    extra_marker_count: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if len(sides) != len(statuses) or not sides:
        raise ValueError("candidate specifications are inconsistent")

    object_count = len(sides)

    for _ in range(120):
        height, width = _sample_shape(rng, regime, object_count)

        accepted: list[dict] = []
        layout_succeeded = True

        for side, status in zip(sides, statuses):
            chosen = None
            for _ in range(220):
                proposal = _propose_candidate(
                    rng,
                    height,
                    width,
                    side,
                    status,
                )
                if (
                    proposal is not None
                    and _compatible_candidate(proposal, accepted)
                ):
                    chosen = proposal
                    break

            if chosen is None:
                layout_succeeded = False
                break

            accepted.append(chosen)

        if not layout_succeeded:
            continue

        background = puzzle_parameters["background"]
        cap = puzzle_parameters["cap"]
        conduit = puzzle_parameters["conduit"]
        marker = puzzle_parameters["marker"]

        input_grid = np.full(
            (height, width),
            background,
            dtype=np.int8,
        )

        for candidate in accepted:
            for position in candidate["caps"]:
                input_grid[position] = cap
            input_grid[candidate["center"]] = conduit
            for position in candidate["path"]:
                input_grid[position] = conduit
            if candidate["marker"] is not None:
                input_grid[candidate["marker"]] = marker

        if extra_marker_count:
            occupied = set().union(*(candidate["cells"] for candidate in accepted))
            extras_succeeded = True
            for _ in range(extra_marker_count):
                detached_choices = [
                    (row, column)
                    for row in range(height)
                    for column in range(width)
                    if all(
                        max(abs(row - p[0]), abs(column - p[1])) > 1
                        for p in occupied
                    )
                    and all(
                        _manhattan((row, column), candidate["endpoint"]) >= 3
                        for candidate in accepted
                    )
                ]
                if not detached_choices:
                    extras_succeeded = False
                    break
                extra_marker = detached_choices[int(rng.integers(0, len(detached_choices)))]
                input_grid[extra_marker] = marker
                occupied.add(extra_marker)
            if not extras_succeeded:
                continue

        output_grid = input_grid.copy()

        for candidate in accepted:
            if candidate["status"] != "complete":
                continue

            output_grid[candidate["path"][0]] = marker
            for position in candidate["path"][1:]:
                output_grid[position] = background
            output_grid[candidate["marker"]] = background

        return input_grid, output_grid

    raise ValueError("unable to construct a non-overlapping pair")


def _make_train_specs(
    rng: np.random.Generator,
    count: int,
) -> list[tuple[list[int], list[str]]]:
    orientation_order = [int(x) for x in rng.permutation(4)]
    gap = ("axial", "remote")[int(rng.integers(0, 2))]
    mode = int(rng.integers(0, 4))
    if mode == 0:
        patterns = [["complete", "diagonal"], ["complete", gap]]
    elif mode == 1:
        patterns = [["complete", "complete", gap], ["complete", "diagonal"]]
    elif mode == 2:
        patterns = [["complete", "diagonal", gap], ["complete", "complete"]]
    else:
        patterns = [["complete", gap], ["complete", "diagonal", "complete"]]

    specs = []
    for index, statuses in enumerate(patterns):
        sides = orientation_order[2 * index:2 * index + 2]
        if len(statuses) == 3:
            sides.append(int(rng.integers(0, 4)))
        order = [int(x) for x in rng.permutation(len(statuses))]
        specs.append(([sides[i] for i in order], [statuses[i] for i in order]))

    for _ in range(2, count):
        object_count = int(rng.choice([1, 2, 3, 4, 5], p=[.23, .25, .24, .20, .08]))
        sides = _sample_sides(rng, object_count)
        roll = float(rng.random())
        if roll < .13:
            statuses = ["diagonal" if int(rng.integers(0, 2)) else "remote" for _ in sides]
        elif roll < .35:
            statuses = ["complete"] * object_count
        else:
            statuses = _sample_statuses(rng, object_count)
        specs.append((sides, statuses))
    return [specs[int(i)] for i in rng.permutation(len(specs))]


def _make_test_specs(
    rng: np.random.Generator,
    count: int,
) -> list[tuple[list[int], list[str]]]:
    specs = []
    for _ in range(count):
        object_count = int(rng.choice([1, 2, 3, 4, 5], p=[.13, .16, .24, .32, .15]))
        sides = _sample_sides(rng, object_count)
        roll = float(rng.random())
        if roll < .13:
            statuses = ["complete"] * object_count
        elif roll < .23:
            statuses = ["diagonal" if int(rng.integers(0, 2)) else "remote" for _ in sides]
        else:
            statuses = _sample_statuses(rng, object_count)
        specs.append((sides, statuses))
    return specs


def _sample_sides(rng: np.random.Generator, count: int) -> list[int]:
    roll = float(rng.random())
    available = 1 if roll < .15 else (2 if roll < .43 else 4)
    chosen = [int(x) for x in rng.choice(4, size=available, replace=False)]
    return [chosen[int(rng.integers(0, available))] for _ in range(count)]


def _sample_statuses(rng: np.random.Generator, count: int) -> list[str]:
    choices = ("complete", "diagonal", "axial", "remote")
    probabilities = (.51, .18, .14, .17)
    return [str(rng.choice(choices, p=probabilities)) for _ in range(count)]


def _grid_key(grid: np.ndarray) -> tuple:
    normalized = np.asarray(grid, dtype=np.int8)
    return tuple(normalized.shape), normalized.tobytes()


def _pair_key(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple:
    return _grid_key(input_grid), _grid_key(output_grid)


def _rows_equal(
    grid: np.ndarray,
    rows: tuple[str, ...],
) -> bool:
    if grid.shape != (len(rows), len(rows[0])):
        return False

    actual = tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )
    return actual == rows


def _is_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    for source_input, source_output in _SOURCE_PAIR_ROWS:
        if (
            _rows_equal(input_grid, source_input)
            and _rows_equal(output_grid, source_output)
        ):
            return True
    return False


def _build_unique_pair(
    rng: np.random.Generator,
    regime: str,
    parameters: dict,
    sides: list[int],
    statuses: list[str],
    used_inputs: set,
    extra_marker_count: int = 0,
) -> dict:
    for _ in range(40):
        pair_seed = int(
            rng.integers(0, np.iinfo(np.int64).max)
        )
        pair_rng = np.random.default_rng(pair_seed)

        try:
            input_grid, output_grid = generate_pair(
                pair_rng, regime, parameters, sides, statuses,
                extra_marker_count,
            )
        except ValueError:
            continue

        input_key = _grid_key(input_grid)

        if input_key in used_inputs:
            continue

        if _is_source_pair(input_grid, output_grid):
            continue

        used_inputs.add(input_key)
        return {
            "input": input_grid,
            "output": output_grid,
        }

    raise ValueError("unable to generate a unique puzzle pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if not _is_integer_argument(seed):
        raise ValueError("seed must be an integer")

    seed_value = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed_value)

    train_count, test_count = _resolve_counts(
        seed_value,
        rng,
        num_train,
        num_test,
    )

    puzzle_parameters = sample_puzzle_parameters(rng)
    recolor_pairs = bool(rng.random() < .58)
    train_specs = _make_train_specs(rng, train_count)
    test_specs = _make_test_specs(rng, test_count)
    # This optional extension is taught explicitly before it appears at test.
    train_extra_index = None
    test_extra_index = None
    if train_count >= 3 and rng.random() < .42:
        train_index = next(i for i, spec in enumerate(train_specs) if len(spec[0]) < 5)
        train_sides, train_statuses = train_specs[train_index]
        train_specs[train_index] = (
            train_sides + [int(rng.integers(0, 4))],
            train_statuses + ["bare"],
        )
        train_extra_index = (train_index + 1) % train_count
        test_index = int(rng.integers(0, test_count))
        test_extra_index = (test_index + 1) % test_count if test_count > 1 else None
        sides, statuses = test_specs[test_index]
        if len(sides) == 1:
            sides = _sample_sides(rng, 2)
            statuses = ["complete", "bare"]
        else:
            statuses = list(statuses)
            statuses[int(rng.integers(0, len(statuses)))] = "bare"
        test_specs[test_index] = (sides, statuses)

    used_inputs: set = set()
    train_pairs = []
    test_pairs = []

    for index, (sides, statuses) in enumerate(train_specs):
        parameters = (
            sample_puzzle_parameters(rng)
            if recolor_pairs and index > 0
            else puzzle_parameters
        )
        train_pairs.append(
            _build_unique_pair(
                rng,
                "source_train",
                parameters,
                sides,
                statuses,
                used_inputs,
                int(index == train_extra_index),
            )
        )

    for index, (sides, statuses) in enumerate(test_specs):
        parameters = sample_puzzle_parameters(rng) if recolor_pairs else puzzle_parameters
        test_pairs.append(
            _build_unique_pair(
                rng,
                "target_test",
                parameters,
                sides,
                statuses,
                used_inputs,
                int(index == test_extra_index),
            )
        )

    puzzle = {
        "train": train_pairs,
        "test": test_pairs,
    }

    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_colors(grid: np.ndarray) -> set[int]:
    return {int(value) for value in np.unique(grid)}


def _analyze_pair_input(
    input_grid: np.ndarray,
    roles: tuple[int, int, int, int],
) -> list[dict]:
    background, cap, conduit, marker = roles
    role_set = {background, cap, conduit, marker}

    _require(
        _grid_colors(input_grid) == role_set,
        "input does not contain exactly the puzzle palette",
    )

    height, width = input_grid.shape
    _require(
        height >= 5 and width >= 5,
        "grid is too small for a valid socket",
    )

    sockets: list[dict] = []

    for column in range(1, width - 1):
        if (
            int(input_grid[0, column - 1]) == cap
            and int(input_grid[0, column]) == conduit
            and int(input_grid[0, column + 1]) == cap
            and int(input_grid[1, column]) == conduit
        ):
            sockets.append(
                {
                    "side": 0,
                    "center": (0, column),
                    "caps": (
                        (0, column - 1),
                        (0, column + 1),
                    ),
                    "inward": (1, 0),
                }
            )

        if (
            int(input_grid[height - 1, column - 1]) == cap
            and int(input_grid[height - 1, column]) == conduit
            and int(input_grid[height - 1, column + 1]) == cap
            and int(input_grid[height - 2, column]) == conduit
        ):
            sockets.append(
                {
                    "side": 1,
                    "center": (height - 1, column),
                    "caps": (
                        (height - 1, column - 1),
                        (height - 1, column + 1),
                    ),
                    "inward": (-1, 0),
                }
            )

    for row in range(1, height - 1):
        if (
            int(input_grid[row - 1, 0]) == cap
            and int(input_grid[row, 0]) == conduit
            and int(input_grid[row + 1, 0]) == cap
            and int(input_grid[row, 1]) == conduit
        ):
            sockets.append(
                {
                    "side": 2,
                    "center": (row, 0),
                    "caps": (
                        (row - 1, 0),
                        (row + 1, 0),
                    ),
                    "inward": (0, 1),
                }
            )

        if (
            int(input_grid[row - 1, width - 1]) == cap
            and int(input_grid[row, width - 1]) == conduit
            and int(input_grid[row + 1, width - 1]) == cap
            and int(input_grid[row, width - 2]) == conduit
        ):
            sockets.append(
                {
                    "side": 3,
                    "center": (row, width - 1),
                    "caps": (
                        (row - 1, width - 1),
                        (row + 1, width - 1),
                    ),
                    "inward": (0, -1),
                }
            )

    _require(sockets, "no boundary sockets were found")

    socket_centers = {socket["center"] for socket in sockets}
    _require(
        len(socket_centers) == len(sockets),
        "a socket was detected more than once",
    )

    actual_caps = {
        (int(row), int(column))
        for row, column in zip(
            *np.where(input_grid == cap)
        )
    }
    expected_caps = {
        position
        for socket in sockets
        for position in socket["caps"]
    }

    _require(
        len(expected_caps) == 2 * len(sockets),
        "socket caps overlap",
    )
    _require(
        actual_caps == expected_caps,
        "cap-colored cells are not exactly the socket caps",
    )

    conduit_cells = {
        (int(row), int(column))
        for row, column in zip(
            *np.where(input_grid == conduit)
        )
    }

    covered_conduit_cells: set[tuple[int, int]] = set()
    candidates: list[dict] = []

    for socket in sockets:
        center = socket["center"]
        stack = [center]
        component: set[tuple[int, int]] = set()

        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)

            for direction in _NEIGHBORS:
                neighbor = _add(current, direction)
                if (
                    neighbor in conduit_cells
                    and neighbor not in component
                ):
                    stack.append(neighbor)

        _require(
            component.isdisjoint(covered_conduit_cells),
            "two sockets share a conduit component",
        )
        _require(
            sum(
                other["center"] in component
                for other in sockets
            ) == 1,
            "a conduit component contains multiple sockets",
        )

        adjacency = {
            position: [
                neighbor
                for direction in _NEIGHBORS
                if (neighbor := _add(position, direction))
                in component
            ]
            for position in component
        }

        endpoints = [
            position
            for position, neighbors in adjacency.items()
            if len(neighbors) == 1
        ]

        _require(
            len(endpoints) == 2 and center in endpoints,
            "conduit component is not a simple socket-rooted path",
        )
        _require(
            all(
                len(neighbors) in (1, 2)
                for neighbors in adjacency.values()
            ),
            "conduit component branches",
        )

        previous = None
        current = center
        ordered_path: list[tuple[int, int]] = []

        while True:
            next_positions = [
                position
                for position in adjacency[current]
                if position != previous
            ]

            if not next_positions:
                break

            _require(
                len(next_positions) == 1,
                "conduit traversal is ambiguous",
            )

            next_position = next_positions[0]
            ordered_path.append(next_position)
            previous, current = current, next_position

        _require(
            len(ordered_path) == len(component) - 1,
            "conduit path did not cover its component",
        )

        directions: list[tuple[int, int]] = []
        previous_position = center

        for position in ordered_path:
            directions.append(
                (
                    position[0] - previous_position[0],
                    position[1] - previous_position[1],
                )
            )
            previous_position = position

        segments: list[list] = []
        for direction in directions:
            if not segments or segments[-1][0] != direction:
                segments.append([direction, 1])
            else:
                segments[-1][1] += 1

        _require(
            len(segments) == 2,
            "conduit must turn exactly once",
        )
        _require(
            segments[0][0] == socket["inward"],
            "conduit does not initially leave the boundary inward",
        )
        _require(
            (
                segments[0][0][0] * segments[1][0][0]
                + segments[0][0][1] * segments[1][0][1]
            ) == 0,
            "conduit turn is not perpendicular",
        )
        _require(
            segments[0][1] >= 1 and segments[1][1] >= 1,
            "conduit legs are too short",
        )

        covered_conduit_cells.update(component)
        candidates.append(
            {
                "side": socket["side"],
                "center": center,
                "caps": socket["caps"],
                "path": ordered_path,
                "endpoint": ordered_path[-1],
                "leg_lengths": (
                    int(segments[0][1]),
                    int(segments[1][1]),
                ),
            }
        )

    _require(
        covered_conduit_cells == conduit_cells,
        "unassigned conduit-colored cells exist",
    )

    marker_cells = {
        (int(row), int(column))
        for row, column in zip(
            *np.where(input_grid == marker)
        )
    }

    _require(marker_cells, "marker role is absent from the input")

    for marker_position in marker_cells:
        _require(
            sum(
                _manhattan(marker_position, candidate["endpoint"]) == 1
                for candidate in candidates
            ) <= 1,
            "marker touches multiple endpoints",
        )
        _require(
            all(
                _manhattan(marker_position, position) > 1
                for candidate in candidates
                for position in candidate["path"][:-1]
            ),
            "marker touches a nonterminal conduit cell",
        )

    for candidate in candidates:
        touching = [
            position for position in marker_cells
            if _manhattan(position, candidate["endpoint"]) == 1
        ]
        _require(len(touching) <= 1, "multiple markers touch one endpoint")
        candidate["marker"] = touching[0] if touching else None
        candidate["style"] = "complete" if touching else "detached"
        candidate["nearest_marker_distance"] = min(
            _manhattan(position, candidate["endpoint"])
            for position in marker_cells
        )

    return candidates


def _check_pair_transformation(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    roles: tuple[int, int, int, int],
) -> list[dict]:
    background, _, _, marker = roles
    role_set = set(roles)

    _require(
        input_grid.shape == output_grid.shape,
        "input and output dimensions differ",
    )
    _require(
        _grid_colors(output_grid) == role_set,
        "output does not preserve the puzzle palette",
    )

    candidates = _analyze_pair_input(input_grid, roles)
    expected = input_grid.copy()

    for candidate in candidates:
        if candidate["style"] != "complete":
            continue

        expected[candidate["path"][0]] = marker
        for position in candidate["path"][1:]:
            expected[position] = background
        expected[candidate["marker"]] = background

    _require(
        np.array_equal(expected, output_grid),
        "output does not implement exact conduit retraction",
    )

    return candidates


def _infer_roles_from_input(input_grid: np.ndarray) -> tuple[int, int, int, int]:
    palette = sorted(_grid_colors(input_grid))
    _require(
        len(palette) == 4,
        "the input must use exactly four colors",
    )
    valid_role_assignments = []
    for assignment in permutations(palette):
        try:
            _analyze_pair_input(input_grid, assignment)
        except AssertionError:
            continue
        valid_role_assignments.append(assignment)
    _require(
        len(valid_role_assignments) == 1,
        "input geometry does not uniquely reveal the color roles",
    )
    return tuple(int(value) for value in valid_role_assignments[0])


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(
        isinstance(puzzle, dict),
        "puzzle must be a dictionary",
    )
    _require(
        set(puzzle.keys()) == {"train", "test"},
        "puzzle must contain exactly train and test",
    )
    _require(
        isinstance(puzzle["train"], list)
        and isinstance(puzzle["test"], list),
        "train and test must be lists",
    )

    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])

    _require(
        2 <= train_count <= 10,
        "illegal training-pair count",
    )
    _require(
        1 <= test_count <= 4,
        "illegal test-pair count",
    )
    _require(
        train_count > test_count,
        "training-pair count must exceed test-pair count",
    )

    input_keys = set()
    pair_keys = set()

    for section in ("train", "test"):
        for pair in puzzle[section]:
            _require(
                isinstance(pair, dict),
                "each pair must be a dictionary",
            )
            _require(
                set(pair.keys()) == {"input", "output"},
                "each pair must contain exactly input and output",
            )

            for grid_name in ("input", "output"):
                grid = pair[grid_name]
                _require(
                    isinstance(grid, np.ndarray),
                    "every grid must be a NumPy array",
                )
                _require(
                    grid.ndim == 2,
                    "every grid must be two-dimensional",
                )
                _require(
                    1 <= grid.shape[0] <= 30
                    and 1 <= grid.shape[1] <= 30,
                    "grid dimensions are outside ARC limits",
                )
                _require(
                    np.issubdtype(grid.dtype, np.integer),
                    "grid dtype must be integer",
                )
                _require(
                    bool(np.all((grid >= 0) & (grid <= 9))),
                    "grid contains a color outside 0 through 9",
                )

            input_key = _grid_key(pair["input"])
            full_pair_key = _pair_key(
                pair["input"], pair["output"]
            )

            _require(
                input_key not in input_keys,
                "duplicate input grid",
            )
            _require(
                full_pair_key not in pair_keys,
                "duplicate input/output pair",
            )
            _require(
                not _is_source_pair(
                    pair["input"], pair["output"]
                ),
                "generated pair duplicates a source-task pair",
            )

            input_keys.add(input_key)
            pair_keys.add(full_pair_key)

    analyses = {"train": [], "test": []}
    for section in ("train", "test"):
        for pair in puzzle[section]:
            roles = _infer_roles_from_input(pair["input"])
            analyses[section].append(
                _check_pair_transformation(pair["input"], pair["output"], roles)
            )

    train_candidates = [candidate for candidates in analyses["train"] for candidate in candidates]
    _require(
        {candidate["side"] for candidate in train_candidates} == {0, 1, 2, 3},
        "training does not show all boundary orientations",
    )
    _require(
        sum(candidate["style"] == "complete" for candidate in train_candidates) >= 2,
        "training does not sufficiently show retraction",
    )
    _require(
        any(candidate["style"] == "detached" for candidate in train_candidates),
        "training does not show preservation of detached arms",
    )
    _require(
        any(candidate["nearest_marker_distance"] == 2 for candidate in train_candidates),
        "training lacks a close near miss",
    )
    _require(
        any(candidate["nearest_marker_distance"] >= 2 for candidate in train_candidates),
        "training lacks a detached marker",
    )


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first.keys()) != set(second.keys()):
        return False

    for section in ("train", "test"):
        if len(first[section]) != len(second[section]):
            return False

        for first_pair, second_pair in zip(
            first[section], second[section]
        ):
            if not np.array_equal(
                first_pair["input"], second_pair["input"]
            ):
                return False
            if not np.array_equal(
                first_pair["output"], second_pair["output"]
            ):
                return False

    return True


def _puzzle_fingerprint(puzzle: dict) -> tuple:
    return tuple(
        (
            section,
            tuple(
                _pair_key(pair["input"], pair["output"])
                for pair in puzzle[section]
            ),
        )
        for section in ("train", "test")
    )


def self_test() -> None:
    train_counts, test_counts = set(), set()
    shapes, object_counts, palettes = set(), set(), set()
    layout_signatures, puzzle_fingerprints = set(), set()
    short_legs = long_legs = near_corners = remote_misses = 0
    repeated_edge_train = repeated_edge_test = 0
    small_test = identity_test = single_test = recolored = 0
    tall = wide = full_size = 0
    marker_free_pairs = extra_marker_pairs = 0
    marker_free_tests = extra_marker_tests = 0

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_fingerprints.add(_puzzle_fingerprint(puzzle))
        train_palettes = {_infer_roles_from_input(p["input"]) for p in puzzle["train"]}
        recolored += len(train_palettes) > 1

        for section in ("train", "test"):
            for pair in puzzle[section]:
                roles = _infer_roles_from_input(pair["input"])
                palettes.add(roles)
                candidates = _check_pair_transformation(pair["input"], pair["output"], roles)
                marker_count = int(np.count_nonzero(pair["input"] == roles[3]))
                marker_free_pairs += marker_count < len(candidates)
                extra_marker_pairs += marker_count > len(candidates)
                if section == "test":
                    marker_free_tests += marker_count < len(candidates)
                    extra_marker_tests += marker_count > len(candidates)
                h, w = pair["input"].shape
                shapes.add((h, w))
                object_counts.add(len(candidates))
                tall += h >= 1.4 * w
                wide += w >= 1.4 * h
                full_size += h == 30 or w == 30
                sides = [candidate["side"] for candidate in candidates]
                if len(set(sides)) < len(sides):
                    if section == "train":
                        repeated_edge_train += 1
                    else:
                        repeated_edge_test += 1
                if section == "test":
                    small_test += h <= 14 and w <= 14
                    identity_test += np.array_equal(pair["input"], pair["output"])
                    single_test += len(candidates) == 1

                signature = tuple(sorted(
                    (c["side"], c["center"], c["leg_lengths"], c["style"])
                    for c in candidates
                ))
                layout_signatures.add((section, (h, w), signature))
                for candidate in candidates:
                    a, b = candidate["leg_lengths"]
                    short_legs += a == 1 or b == 1
                    long_legs += a >= 8 or b >= 8
                    near_corners += candidate["center"][1 if candidate["side"] < 2 else 0] in (1, (w if candidate["side"] < 2 else h) - 2)
                    remote_misses += candidate["style"] == "detached" and candidate["nearest_marker_distance"] >= 3

    for name, value, minimum in (
        ("train-count diversity", len(train_counts), 4),
        ("test-count diversity", len(test_counts), 3),
        ("palette diversity", len(palettes), 30),
        ("shape diversity", len(shapes), 50),
        ("layout diversity", len(layout_signatures), 120),
        ("object-count diversity", len(object_counts), 5),
        ("seed diversity", len(puzzle_fingerprints), 40),
        ("short legs", short_legs, 20),
        ("long legs", long_legs, 5),
        ("near-corner sockets", near_corners, 5),
        ("remote near misses", remote_misses, 5),
        ("repeated-edge training", repeated_edge_train, 5),
        ("repeated-edge tests", repeated_edge_test, 5),
        ("compact tests", small_test, 1),
        ("identity tests", identity_test, 1),
        ("single-object tests", single_test, 1),
        ("recolored puzzles", recolored, 8),
        ("tall grids", tall, 5),
        ("wide grids", wide, 5),
        ("30-cell dimensions", full_size, 1),
        ("marker-free-arm examples", marker_free_pairs, 1),
        ("extra-detached-marker examples", extra_marker_pairs, 1),
        ("marker-free-arm tests", marker_free_tests, 1),
        ("extra-detached-marker tests", extra_marker_tests, 1),
    ):
        _require(value >= minimum, f"insufficient {name}: {value} < {minimum}")

    deterministic_first = generate_puzzle(
        123456,
        num_train=6,
        num_test=2,
    )
    deterministic_second = generate_puzzle(
        123456,
        num_train=6,
        num_test=2,
    )
    _require(
        _puzzles_equal(
            deterministic_first,
            deterministic_second,
        ),
        "whole-puzzle generation is not deterministic",
    )

    different_seed_fingerprints = {
        _puzzle_fingerprint(
            generate_puzzle(
                2000 + offset,
                num_train=5,
                num_test=2,
            )
        )
        for offset in range(8)
    }
    _require(
        len(different_seed_fingerprints) >= 7,
        "different seeds usually produce identical puzzles",
    )

    boundary_cases = (
        (3000, 2, 1),
        (3001, 10, 1),
        (3002, 10, 4),
    )

    for seed, requested_train, requested_test in boundary_cases:
        puzzle = generate_puzzle(
            seed,
            num_train=requested_train,
            num_test=requested_test,
        )
        validate_generated_puzzle(puzzle)
        _require(
            len(puzzle["train"]) == requested_train,
            "explicit num_train was not honored",
        )
        _require(
            len(puzzle["test"]) == requested_test,
            "explicit num_test was not honored",
        )

    partial_cases = (
        (4000, None, 1),
        (4001, None, 4),
        (4002, 2, None),
        (4003, 10, None),
    )

    for seed, requested_train, requested_test in partial_cases:
        puzzle = generate_puzzle(
            seed,
            num_train=requested_train,
            num_test=requested_test,
        )
        validate_generated_puzzle(puzzle)

        if requested_train is not None:
            _require(
                len(puzzle["train"]) == requested_train,
                "partial num_train was not honored",
            )
        if requested_test is not None:
            _require(
                len(puzzle["test"]) == requested_test,
                "partial num_test was not honored",
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

    for arguments in illegal_cases:
        raised = False
        try:
            generate_puzzle(5000, **arguments)
        except ValueError:
            raised = True

        _require(
            raised,
            f"illegal count arguments did not raise: {arguments}",
        )
