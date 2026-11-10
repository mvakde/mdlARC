import numpy as np


_DIRECTIONS = ("N", "E", "S", "W")
_DIR_VECTOR = {
    "N": (-1, 0),
    "E": (0, 1),
    "S": (1, 0),
    "W": (0, -1),
}
_CORNER_OFFSETS = (
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)
_SOURCE_INPUTS = {
    (((4, 10), (11, 2)), ("E", "N")),
    (((3, 3), (11, 12)), ("S", "W")),
    (((2, 2), (13, 10)), ("N", "E")),
    (((2, 10), (11, 3)), ("E", "W")),
}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _validate_count_argument(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, rng, num_train, num_test):
    num_train = _validate_count_argument(
        "num_train", num_train, 2, 10
    )
    num_test = _validate_count_argument(
        "num_test", num_test, 1, 4
    )

    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        span = 10 - num_test
        num_train = (
            num_test
            + 1
            + ((seed // 4 + 3 * (seed % 4)) % span)
        )
    elif num_train is None:
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_test = int(
            rng.integers(1, min(4, num_train - 1) + 1)
        )

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    return {"connector": int(rng.integers(10))}


def _distance_map(grid, wall_color, start):
    height, width = grid.shape
    distances = np.full(
        (height, width), -1, dtype=np.int32
    )

    start_row, start_col = start
    distances[start_row, start_col] = 0

    queue = [(start_row, start_col)]
    head = 0

    while head < len(queue):
        row, col = queue[head]
        head += 1
        next_distance = int(distances[row, col]) + 1

        for dr, dc in _DIR_VECTOR.values():
            new_row = row + dr
            new_col = col + dc

            if (
                0 <= new_row < height
                and 0 <= new_col < width
                and distances[new_row, new_col] == -1
                and grid[new_row, new_col] != wall_color
            ):
                distances[new_row, new_col] = next_distance
                queue.append((new_row, new_col))

    return distances


def _construct_output(input_grid, parameters, centers):
    wall = parameters["wall"]
    background = parameters["background"]
    connector = parameters["connector"]

    first, second = centers
    from_first = _distance_map(
        input_grid, wall, first
    )
    from_second = _distance_map(
        input_grid, wall, second
    )

    shortest_length = int(from_first[second])
    if shortest_length < 0:
        raise RuntimeError(
            "generated terminals are disconnected"
        )

    shortest_union = (
        (input_grid == background)
        & (from_first >= 0)
        & (from_second >= 0)
        & (
            from_first + from_second
            == shortest_length
        )
    )

    output_grid = input_grid.copy()
    output_grid[shortest_union] = connector
    return output_grid


def _place_chamber(
    grid,
    center,
    opening,
    parameters,
):
    row, col = center
    wall = parameters["wall"]
    terminal = parameters["terminal"]

    for dr, dc in _CORNER_OFFSETS:
        grid[row + dr, col + dc] = wall

    for direction, (dr, dc) in _DIR_VECTOR.items():
        if direction != opening:
            grid[row + dr, col + dc] = wall

    grid[row, col] = terminal


def _sample_dimensions(rng):
    pick = float(rng.random())
    if pick < 0.08:
        return 15, 15
    if pick < 0.16:
        return 30, 30
    if pick < 0.30:
        return int(rng.integers(7, 12)), int(rng.integers(7, 12))
    return int(rng.integers(8, 31)), int(rng.integers(8, 31))


def generate_pair(
    rng,
    puzzle_parameters,
    mode,
):
    for attempt in range(300):
        height, width = _sample_dimensions(rng) if attempt < 240 else (19, 21)
        if mode in ("broad", "straight"):
            if mode == "broad" and min(height, width) < 10:
                continue
            horizontal = bool(rng.integers(2))
            if horizontal:
                margin = 2 if mode == "broad" else 1
                if width < 2 * margin + 4:
                    continue
                a = int(rng.integers(margin, width - margin - 4))
                b = int(rng.integers(a + 4, width - margin))
                row = int(rng.integers(2, height - 2))
                centers = ((row, a), (row, b))
                openings = ("W", "E") if mode == "broad" else ("E", "W")
            else:
                margin = 2 if mode == "broad" else 1
                if height < 2 * margin + 4:
                    continue
                a = int(rng.integers(margin, height - margin - 4))
                b = int(rng.integers(a + 4, height - margin))
                col = int(rng.integers(2, width - 2))
                centers = ((a, col), (b, col))
                openings = ("N", "S") if mode == "broad" else ("S", "N")
        elif mode == "edge":
            if min(height, width) < 10:
                continue
            c0 = int(rng.integers(2, width - 6))
            c1 = int(rng.integers(c0 + 4, width - 2))
            centers, openings = ((2, c0), (height - 2, c1)), ("N", "E")
            turns = int(rng.integers(4))
            for _ in range(turns):
                centers = tuple((c, height - 1 - r) for r, c in centers)
                openings = tuple(_DIRECTIONS[(_DIRECTIONS.index(d) + 1) % 4] for d in openings)
                height, width = width, height
        elif mode == "source":
            if min(height, width) < 10:
                continue
            r0 = int(rng.integers(2, height - 6))
            r1 = int(rng.integers(r0 + 4, height - 1))
            c1 = int(rng.integers(2, width - 6))
            c0 = int(rng.integers(c1 + 4, width - 2))
            centers, openings = ((r0, c0), (r1, c1)), ("E", "W")
        else:
            category = int(rng.integers(3))
            if category == 0:
                dr, dc = (0, int(rng.integers(4, width - 2))) if rng.integers(2) else (int(rng.integers(4, height - 2)), 0)
            elif category == 1:
                dr, dc = (int(rng.integers(1, 3)), int(rng.integers(4, width - 2))) if rng.integers(2) else (int(rng.integers(4, height - 2)), int(rng.integers(1, 3)))
            else:
                dr, dc = int(rng.integers(4, height - 2)), int(rng.integers(4, width - 2))
            dr *= 1 if rng.integers(2) else -1
            dc *= 1 if rng.integers(2) else -1
            rlo, rhi = max(1, 1 - dr), min(height - 2, height - 2 - dr)
            clo, chi = max(1, 1 - dc), min(width - 2, width - 2 - dc)
            if rlo > rhi or clo > chi:
                continue
            r = int(rng.integers(rlo, rhi + 1))
            c = int(rng.integers(clo, chi + 1))
            centers = ((r, c), (r + dr, c + dc))
            openings = tuple(_DIRECTIONS[int(rng.integers(4))] for _ in range(2))
        centers, openings = zip(*sorted(zip(centers, openings)))
        centers, openings = tuple(centers), tuple(openings)
        if abs(centers[0][0] - centers[1][0]) < 4 and abs(centers[0][1] - centers[1][1]) < 4:
            continue
        if any(not (0 <= r + 2 * _DIR_VECTOR[d][0] < height and 0 <= c + 2 * _DIR_VECTOR[d][1] < width)
               for (r, c), d in zip(centers, openings)):
            continue
        input_grid = np.full((height, width), puzzle_parameters["background"], dtype=np.int8)
        for center, opening in zip(centers, openings):
            _place_chamber(input_grid, center, opening, puzzle_parameters)
        try:
            output_grid = _construct_output(input_grid, puzzle_parameters, centers)
        except RuntimeError:
            continue
        size = int(np.count_nonzero(output_grid == puzzle_parameters["connector"]))
        length = int(_distance_map(input_grid, puzzle_parameters["wall"], centers[0])[centers[1]])
        if mode == "broad" and size <= length - 1:
            continue
        if mode == "straight" and size != length - 1:
            continue
        if mode == "edge" and not (np.any(output_grid[0] == puzzle_parameters["connector"])
                                   or np.any(output_grid[-1] == puzzle_parameters["connector"])
                                   or np.any(output_grid[:, 0] == puzzle_parameters["connector"])
                                   or np.any(output_grid[:, -1] == puzzle_parameters["connector"])):
            continue
        return input_grid, output_grid
    raise RuntimeError("could not construct a connected chamber layout")


def _grid_key(grid):
    return (
        grid.shape,
        grid.dtype.str,
        grid.tobytes(),
    )


def _pair_key(input_grid, output_grid):
    return (
        _grid_key(input_grid),
        _grid_key(output_grid),
    )


def _inspect_pair(input_grid, output_grid):
    _require(
        input_grid.shape == output_grid.shape,
        "input/output dimensions differ",
    )

    values, counts = np.unique(
        input_grid, return_counts=True
    )
    _require(
        len(values) == 3,
        "an input must contain exactly three colors",
    )

    terminal_candidates = values[counts == 2]
    wall_candidates = values[counts == 14]

    _require(
        len(terminal_candidates) == 1,
        "terminal color is not identifiable",
    )
    _require(
        len(wall_candidates) == 1,
        "wall color is not identifiable",
    )

    terminal = int(terminal_candidates[0])
    wall = int(wall_candidates[0])

    _require(
        terminal != wall,
        "terminal and wall colors coincide",
    )

    background_values = [
        int(value)
        for value in values
        if int(value) not in (terminal, wall)
    ]
    _require(
        len(background_values) == 1,
        "background color is not identifiable",
    )
    background = background_values[0]

    _require(
        int(
            np.count_nonzero(
                input_grid == background
            )
        )
        > 14,
        "background is not the dominant field",
    )

    centers = [
        tuple(int(x) for x in position)
        for position in np.argwhere(
            input_grid == terminal
        )
    ]
    centers.sort()

    _require(
        len(centers) == 2,
        "there must be two terminal centers",
    )

    expected_wall_cells = set()
    chamber_areas = []
    openings = []
    height, width = input_grid.shape

    for row, col in centers:
        _require(
            1 <= row < height - 1
            and 1 <= col < width - 1,
            "a chamber touches the grid boundary",
        )

        area = {
            (row + dr, col + dc)
            for dr in (-1, 0, 1)
            for dc in (-1, 0, 1)
        }
        chamber_areas.append(area)

        for dr, dc in _CORNER_OFFSETS:
            _require(
                int(
                    input_grid[
                        row + dr,
                        col + dc,
                    ]
                )
                == wall,
                "a chamber corner is not a wall",
            )
            expected_wall_cells.add(
                (row + dr, col + dc)
            )

        open_directions = []

        for direction, (dr, dc) in (
            _DIR_VECTOR.items()
        ):
            value = int(
                input_grid[
                    row + dr,
                    col + dc,
                ]
            )

            if value == background:
                open_directions.append(direction)
            else:
                _require(
                    value == wall,
                    "invalid chamber side-middle cell",
                )
                expected_wall_cells.add(
                    (row + dr, col + dc)
                )

        _require(
            len(open_directions) == 1,
            "chamber does not have one opening",
        )
        openings.append(open_directions[0])

    _require(
        chamber_areas[0].isdisjoint(
            chamber_areas[1]
        ),
        "chambers overlap",
    )
    _require(
        max(abs(centers[0][0] - centers[1][0]),
            abs(centers[0][1] - centers[1][1])) >= 4,
        "chambers need a background gap",
    )

    actual_wall_cells = {
        tuple(int(x) for x in position)
        for position in np.argwhere(
            input_grid == wall
        )
    }
    _require(
        actual_wall_cells
        == expected_wall_cells,
        "extra or missing wall cells",
    )

    for (row, col), direction in zip(centers, openings):
        dr, dc = _DIR_VECTOR[direction]
        _require(
            0 <= row + 2 * dr < height and 0 <= col + 2 * dc < width,
            "an opening is trapped at the boundary",
        )

    changed = input_grid != output_grid
    _require(
        np.any(changed),
        "the transformation must change background cells",
    )
    _require(
        np.all(
            input_grid[changed] == background
        ),
        "a non-background input cell was changed",
    )

    connector_values = np.unique(
        output_grid[changed]
    )
    _require(
        len(connector_values) == 1,
        "changed cells use multiple connector colors",
    )

    connector = int(connector_values[0])
    _require(
        connector
        not in {
            background,
            wall,
            terminal,
        },
        "connector is not new",
    )
    _require(
        set(
            int(value)
            for value in np.unique(output_grid)
        )
        == {
            background,
            wall,
            terminal,
            connector,
        },
        "output palette is inconsistent",
    )

    from_first = _distance_map(
        input_grid,
        wall,
        centers[0],
    )
    from_second = _distance_map(
        input_grid,
        wall,
        centers[1],
    )
    shortest_length = int(
        from_first[centers[1]]
    )

    _require(
        shortest_length >= 0,
        "terminals are disconnected",
    )

    shortest_union = (
        (input_grid == background)
        & (from_first >= 0)
        & (from_second >= 0)
        & (
            from_first + from_second
            == shortest_length
        )
    )

    independently_expected = input_grid.copy()
    independently_expected[
        shortest_union
    ] = connector

    _require(
        np.array_equal(
            output_grid,
            independently_expected,
        ),
        "output is not the complete shortest-path union",
    )

    _require(
        np.any(
            (input_grid == background)
            & ~shortest_union
        ),
        "degenerate instance recolors all background",
    )

    for (row, col), direction in zip(
        centers, openings
    ):
        dr, dc = _DIR_VECTOR[direction]
        _require(
            bool(
                shortest_union[
                    row + dr,
                    col + dc,
                ]
            ),
            "a mandatory chamber opening was not recolored",
        )

    _require(
        not (
            input_grid.shape == (15, 15)
            and (background, wall, terminal, connector) == (0, 2, 5, 4)
            and (tuple(centers), tuple(openings)) in _SOURCE_INPUTS
        ),
        "exact source pair copied",
    )

    return {
        "roles": (
            background,
            wall,
            terminal,
            connector,
        ),
        "centers": tuple(centers),
        "openings": tuple(openings),
        "shape": input_grid.shape,
        "connector_cells": int(
            np.count_nonzero(shortest_union)
        ),
        "shortest_length": shortest_length,
        "edge_connector": bool(
            np.any(shortest_union[0])
            or np.any(shortest_union[-1])
            or np.any(shortest_union[:, 0])
            or np.any(shortest_union[:, -1])
        ),
    }


def validate_generated_puzzle(
    puzzle: dict,
) -> None:
    _require(
        isinstance(puzzle, dict),
        "puzzle must be a dictionary",
    )
    _require(
        set(puzzle.keys())
        == {"train", "test"},
        "invalid puzzle keys",
    )
    _require(
        isinstance(puzzle["train"], list),
        "train must be a list",
    )
    _require(
        isinstance(puzzle["test"], list),
        "test must be a list",
    )

    num_train = len(puzzle["train"])
    num_test = len(puzzle["test"])

    _require(
        2 <= num_train <= 10,
        "invalid training pair count",
    )
    _require(
        1 <= num_test <= 4,
        "invalid test pair count",
    )
    _require(
        num_train > num_test,
        "training count must exceed test count",
    )

    input_keys = set()
    pair_keys = set()
    connector_color = None
    training_palettes = set()
    training_path_kinds = set()

    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(
                isinstance(pair, dict),
                "a pair must be a dictionary",
            )
            _require(
                set(pair.keys())
                == {"input", "output"},
                "invalid pair keys",
            )

            for name in ("input", "output"):
                grid = pair[name]

                _require(
                    isinstance(grid, np.ndarray),
                    "a grid is not an ndarray",
                )
                _require(
                    grid.ndim == 2,
                    "a grid is not two-dimensional",
                )
                _require(
                    1 <= grid.shape[0] <= 30
                    and 1 <= grid.shape[1] <= 30,
                    "grid dimensions are outside ARC limits",
                )
                _require(
                    np.issubdtype(
                        grid.dtype,
                        np.integer,
                    ),
                    "grid dtype is not integer",
                )
                _require(
                    int(np.min(grid)) >= 0
                    and int(np.max(grid)) <= 9,
                    "grid contains an invalid color",
                )

            input_grid = pair["input"]
            output_grid = pair["output"]

            input_key = _grid_key(input_grid)
            pair_key = _pair_key(
                input_grid,
                output_grid,
            )

            _require(
                input_key not in input_keys,
                "duplicate input grid",
            )
            _require(
                pair_key not in pair_keys,
                "duplicate input/output pair",
            )

            input_keys.add(input_key)
            pair_keys.add(pair_key)

            info = _inspect_pair(
                input_grid,
                output_grid,
            )

            if connector_color is None:
                connector_color = info["roles"][3]
            _require(info["roles"][3] == connector_color, "connector convention changed")
            if split == "train":
                training_palettes.add(info["roles"][:3])
                training_path_kinds.add(
                    "single" if info["connector_cells"] == info["shortest_length"] - 1
                    else "union"
                )

    _require(len(training_palettes) >= 2, "training must show pair-level recoloring")
    _require(training_path_kinds == {"single", "union"},
             "training must show both one and multiple shortest routes")


def _append_unique_pair(
    destination,
    pair_rng,
    parameters,
    mode,
    used_inputs,
    used_pairs,
):
    for _ in range(200):
        input_grid, output_grid = generate_pair(
            pair_rng,
            parameters,
            mode,
        )

        input_key = _grid_key(input_grid)
        pair_key = _pair_key(
            input_grid,
            output_grid,
        )

        try:
            _inspect_pair(input_grid, output_grid)
        except AssertionError:
            continue

        if (
            input_key not in used_inputs
            and pair_key not in used_pairs
        ):
            destination.append(
                {
                    "input": input_grid,
                    "output": output_grid,
                }
            )
            used_inputs.add(input_key)
            used_pairs.add(pair_key)
            return

    raise RuntimeError(
        "could not generate a unique pair"
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(
        seed, (bool, np.bool_)
    ) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError(
            "seed must be a non-negative integer"
        )

    seed = int(seed)
    if seed < 0:
        raise ValueError(
            "seed must be a non-negative integer"
        )

    rng = np.random.default_rng(seed)

    num_train, num_test = _resolve_counts(
        seed,
        rng,
        num_train,
        num_test,
    )

    parameters = sample_puzzle_parameters(rng)
    connector = parameters["connector"]

    total_pairs = num_train + num_test
    child_seeds = rng.integers(
        0,
        np.iinfo(np.uint64).max,
        size=total_pairs,
        dtype=np.uint64,
    )

    puzzle = {
        "train": [],
        "test": [],
    }
    used_inputs = set()
    used_pairs = set()
    child_index = 0

    previous_palette = None
    for index in range(num_train):
        pair_rng = np.random.default_rng(
            int(child_seeds[child_index])
        )
        child_index += 1
        choices = [color for color in range(10) if color != connector]
        palette = tuple(int(color) for color in pair_rng.choice(choices, size=3, replace=False))
        while palette == previous_palette:
            palette = tuple(int(color) for color in pair_rng.choice(choices, size=3, replace=False))
        previous_palette = palette
        pair_parameters = dict(zip(("background", "wall", "terminal", "connector"), palette + (connector,)))
        if index == 0:
            mode = "broad"
        elif index == 1:
            mode = "straight"
        else:
            mode = "edge" if pair_rng.random() < 0.19 else "random"

        _append_unique_pair(
            puzzle["train"],
            pair_rng,
            pair_parameters,
            mode,
            used_inputs,
            used_pairs,
        )

    for _ in range(num_test):
        pair_rng = np.random.default_rng(
            int(child_seeds[child_index])
        )
        child_index += 1
        choices = [color for color in range(10) if color != connector]
        palette = tuple(int(color) for color in pair_rng.choice(choices, size=3, replace=False))
        while palette == previous_palette:
            palette = tuple(int(color) for color in pair_rng.choice(choices, size=3, replace=False))
        previous_palette = palette
        pair_parameters = dict(zip(("background", "wall", "terminal", "connector"), palette + (connector,)))
        p = pair_rng.random()
        mode = "source" if p < 0.13 else "broad" if p < 0.31 else "edge" if p < 0.43 else "random"

        _append_unique_pair(
            puzzle["test"],
            pair_rng,
            pair_parameters,
            mode,
            used_inputs,
            used_pairs,
        )

    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    if (
        set(first.keys()) != {"train", "test"}
        or set(second.keys()) != {"train", "test"}
    ):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for pair_a, pair_b in zip(
            first[split],
            second[split],
        ):
            if not np.array_equal(
                pair_a["input"],
                pair_b["input"],
            ):
                return False
            if not np.array_equal(
                pair_a["output"],
                pair_b["output"],
            ):
                return False

    return True


def _puzzle_fingerprint(puzzle):
    pieces = []

    for split in ("train", "test"):
        pieces.append(split)

        for pair in puzzle[split]:
            pieces.append(
                _pair_key(
                    pair["input"],
                    pair["output"],
                )
            )

    return tuple(pieces)


def _predict_from_training(train, test_input):
    # This check receives only returned demonstrations and a fresh input.
    examples = []
    for pair in train:
        before, after = pair["input"], pair["output"]
        changed = before != after
        _require(np.any(changed), "empty demonstration")
        examples.append(int(np.unique(after[changed])[0]))
    _require(len(set(examples)) == 1, "ambiguous connector convention")
    connector = examples[0]
    values, counts = np.unique(test_input, return_counts=True)
    marker = int(values[counts == 2][0])
    wall = int(values[counts == 14][0])
    background = int(next(value for value in values if value not in (marker, wall)))
    centers = [tuple(map(int, position)) for position in np.argwhere(test_input == marker)]

    def flood(start):
        distances = {start: 0}
        frontier = [start]
        while frontier:
            next_frontier = []
            for row, col in frontier:
                for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                    candidate = (row + dr, col + dc)
                    r, c = candidate
                    if (0 <= r < test_input.shape[0] and 0 <= c < test_input.shape[1]
                            and test_input[r, c] != wall and candidate not in distances):
                        distances[candidate] = distances[(row, col)] + 1
                        next_frontier.append(candidate)
            frontier = next_frontier
        return distances

    left, right = flood(centers[0]), flood(centers[1])
    length = left[centers[1]]
    prediction = test_input.copy()
    for position, first_distance in left.items():
        if position in right and first_distance + right[position] == length:
            if test_input[position] == background:
                prediction[position] = connector
    return prediction


def self_test() -> None:
    cases = ((2000, 2, 1), (2001, 10, 1), (2002, 10, 4),
             (2003, None, 1), (2004, None, 4), (2005, 2, None),
             (2006, 10, None), (2007, 7, 3))
    for seed, num_train, num_test in cases:
        puzzle = generate_puzzle(seed, num_train, num_test)
        validate_generated_puzzle(puzzle)
        if num_train is not None:
            _require(len(puzzle["train"]) == num_train, "train count ignored")
        if num_test is not None:
            _require(len(puzzle["test"]) == num_test, "test count ignored")
    for num_train, num_test in ((1, None), (11, None), (None, 0),
                                (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(100, num_train, num_test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    _require(_puzzles_equal(generate_puzzle(932), generate_puzzle(932)), "nondeterminism")
    train_counts, test_counts, fingerprints = set(), set(), set()
    shapes, palettes, layouts, sizes, directions, categories = (set() for _ in range(6))
    edges = source_like = shared = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_puzzle_fingerprint(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_features = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair["input"], pair["output"])
                centers, openings = info["centers"], info["openings"]
                dr = abs(centers[0][0] - centers[1][0])
                dc = abs(centers[0][1] - centers[1][1])
                category = "aligned" if min(dr, dc) == 0 else "near" if min(dr, dc) <= 2 else "diagonal"
                feature = (category, openings)
                directions.add(openings)
                categories.add(category)
                shapes.add(info["shape"])
                palettes.add(info["roles"][:3])
                layouts.add((info["shape"], centers, openings))
                sizes.add(info["connector_cells"])
                edges += int(info["edge_connector"])
                source_like += int(split == "test" and centers[0][0] < centers[1][0]
                                   and centers[0][1] > centers[1][1] and openings == ("E", "W"))
                if split == "train":
                    train_features.add(feature)
                else:
                    shared += int(feature in train_features)
        for pair in puzzle["test"]:
            predicted = _predict_from_training(puzzle["train"], pair["input"].copy())
            _require(np.array_equal(predicted, pair["output"]), "test not recoverable from demonstrations")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3, "count diversity")
    _require(len(fingerprints) == 42 and len(shapes) >= 65 and len(layouts) >= 175,
             "structural diversity")
    _require(len(palettes) >= 80 and len(sizes) >= 20, "palette or route-size diversity")
    _require(len(directions) == 16 and len(categories) == 3, "direction or spacing diversity")
    _require(edges >= 8 and source_like >= 2 and shared >= 2, "edge or regime diversity")
