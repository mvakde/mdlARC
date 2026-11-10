import hashlib
from collections import Counter

import numpy as np


def _coerce_optional_count(
    value: int | None,
    name: str,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(
    seed_value: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _coerce_optional_count(num_train, "num_train")
    num_test = _coerce_optional_count(num_test, "num_test")

    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        num_test = 1 + seed_value % 4
        num_train = (
            num_test
            + 1
            + (seed_value // 4) % (10 - num_test)
        )
    elif num_train is None:
        num_train = (
            num_test
            + 1
            + seed_value % (10 - num_test)
        )
    else:
        upper = min(4, num_train - 1)
        num_test = 1 + seed_value % upper

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError(
            "the requested pair counts cannot form a legal puzzle"
        )

    return int(num_train), int(num_test)


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    # The route convention is shared by every pair.  Color identities are
    # not a convention in the source task: roles are read from each input.
    colors = tuple(int(x) for x in rng.choice(10, size=4, replace=False))
    return {
        "anchor_palette": colors,
        "route_convention": ("positive_axis", "incoming_face", "portal_middle"),
    }


def _sample_roles(
    rng: np.random.Generator,
    anchor: tuple[int, ...],
    previous: list[tuple[int, ...]],
) -> tuple[int, ...]:
    if not previous:
        return anchor
    mode = int(rng.choice(3, p=(0.40, 0.25, 0.35)))
    if mode == 0:
        return tuple(int(x) for x in rng.choice(10, size=4, replace=False))
    base = previous[int(rng.integers(0, len(previous)))]
    if mode == 1:
        return base
    if rng.random() < 0.40:
        return (base[0], base[2], base[1], base[3])
    permutation = tuple(int(x) for x in rng.permutation(4))
    if permutation == (0, 1, 2, 3):
        permutation = (1, 0, 3, 2)
    return tuple(base[index] for index in permutation)


def _sample_dimensions(
    rng: np.random.Generator,
    orientation: str,
    demonstrative: bool,
) -> tuple[int, int]:
    if rng.random() < 0.18:
        return 30, 30
    height = int(rng.integers(12, 31))
    width = int(rng.integers(12, 31))
    if demonstrative:
        if orientation == "vertical":
            width = max(width, 16)
        else:
            height = max(height, 16)
    return height, width


def _sample_barrier_positions(
    rng: np.random.Generator,
    progress_length: int,
    count: int,
) -> tuple[int, ...]:
    minimum_separation = 3
    first_allowed = 3
    last_allowed = progress_length - 4

    adjusted_last = (
        last_allowed
        - (count - 1) * (minimum_separation - 1)
    )
    choices = np.arange(
        first_allowed,
        adjusted_last + 1,
        dtype=np.int64,
    )

    if choices.size < count:
        raise ValueError(
            "grid is too small for the requested barriers"
        )

    compressed = np.sort(
        rng.choice(choices, size=count, replace=False)
    )
    positions = (
        compressed
        + np.arange(count) * (minimum_separation - 1)
    )
    return tuple(int(x) for x in positions)


def _sample_portal_centers(
    rng: np.random.Generator,
    parallel_length: int,
    count: int,
    mode: str,
) -> tuple[int, ...]:
    values = np.arange(
        1,
        parallel_length - 1,
        dtype=np.int64,
    )
    if values.size < 3:
        raise ValueError(
            "grid is too small for three-cell portals"
        )

    if mode == "monotone":
        centers = sorted(int(x) for x in rng.choice(values, size=count, replace=False))
        if rng.random() < 0.5:
            centers.reverse()
        return tuple(centers)
    if mode == "repeat":
        centers = [int(rng.choice(values)) for _ in range(count)]
        index = int(rng.integers(1, count))
        centers[index] = centers[index - 1]
        return tuple(centers)
    if mode == "mixed":
        return tuple(int(rng.choice(values)) for _ in range(count))
    if mode != "zigzag" or count < 3:
        raise ValueError("invalid portal sequence mode")

    centers = [int(rng.choice(values))]
    centers.append(int(rng.choice(values[values != centers[0]])))
    while len(centers) < count:
        previous_delta = centers[-1] - centers[-2]
        candidates = values[values < centers[-1]] if previous_delta > 0 else values[values > centers[-1]]
        centers.append(int(rng.choice(candidates)))
    return tuple(centers)


def _choose_parallel_coordinate(
    rng: np.random.Generator,
    length: int,
    forbidden: int,
    allow_alignment: bool,
) -> int:
    if allow_alignment and rng.random() < 0.27:
        return forbidden
    choices = np.arange(length, dtype=np.int64)
    choices = choices[choices != forbidden]
    return int(rng.choice(choices))


def _paint_segment(
    grid: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: int,
) -> None:
    r0, c0 = start
    r1, c1 = end

    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo:hi + 1] = color
    else:
        assert c0 == c1
        lo, hi = sorted((r0, r1))
        grid[lo:hi + 1, c0] = color


def _draw_route(
    input_grid: np.ndarray,
    orientation: str,
    barriers: tuple[int, ...],
    portal_centers: tuple[int, ...],
    leading_endpoint: tuple[int, int],
    trailing_endpoint: tuple[int, int],
    route_color: int,
) -> np.ndarray:
    output = input_grid.copy()
    current = tuple(leading_endpoint)

    if orientation == "vertical":
        for barrier, portal in zip(
            barriers, portal_centers
        ):
            waypoints = (
                (current[0], barrier - 1),
                (portal, barrier - 1),
                (portal, barrier),
            )
            for waypoint in waypoints:
                _paint_segment(
                    output,
                    current,
                    waypoint,
                    route_color,
                )
                current = waypoint

        waypoints = (
            (current[0], trailing_endpoint[1]),
            tuple(trailing_endpoint),
        )
    else:
        for barrier, portal in zip(
            barriers, portal_centers
        ):
            waypoints = (
                (barrier - 1, current[1]),
                (barrier - 1, portal),
                (barrier, portal),
            )
            for waypoint in waypoints:
                _paint_segment(
                    output,
                    current,
                    waypoint,
                    route_color,
                )
                current = waypoint

        waypoints = (
            (trailing_endpoint[0], current[1]),
            tuple(trailing_endpoint),
        )

    for waypoint in waypoints:
        _paint_segment(
            output,
            current,
            waypoint,
            route_color,
        )
        current = waypoint

    return output


def generate_pair(
    rng: np.random.Generator,
    regime: str,
    puzzle_parameters: dict,
    orientation: str,
    roles: tuple[int, int, int, int],
    demonstrative: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown generation regime")
    if orientation not in ("vertical", "horizontal"):
        raise ValueError("unknown barrier orientation")

    assert puzzle_parameters["route_convention"] == (
        "positive_axis", "incoming_face", "portal_middle"
    )
    height, width = _sample_dimensions(rng, orientation, demonstrative)

    (
        background,
        barrier_color,
        portal_color,
        route_color,
    ) = roles

    if orientation == "vertical":
        progress_length = width
        parallel_length = height
    else:
        progress_length = height
        parallel_length = width

    maximum_count = min(6, (progress_length - 7) // 3 + 1)
    minimum_count = 3 if demonstrative else 2
    barrier_count = int(rng.integers(minimum_count, maximum_count + 1))

    barriers = _sample_barrier_positions(
        rng,
        progress_length,
        barrier_count,
    )
    if demonstrative:
        portal_mode = "zigzag"
    else:
        portal_mode = str(rng.choice(
            ("zigzag", "monotone", "repeat", "mixed"),
            p=(0.38, 0.20, 0.20, 0.22),
        ))
        if barrier_count == 2 and portal_mode == "zigzag":
            portal_mode = "mixed"
    centers = _sample_portal_centers(
        rng,
        parallel_length,
        barrier_count,
        portal_mode,
    )

    leading_axis = int(rng.integers(0, barriers[0] - 1))
    trailing_axis = int(rng.integers(barriers[-1] + 2, progress_length))

    leading_parallel = _choose_parallel_coordinate(
        rng,
        parallel_length,
        centers[0],
        not demonstrative,
    )
    trailing_parallel = _choose_parallel_coordinate(
        rng,
        parallel_length,
        centers[-1],
        not demonstrative,
    )

    input_grid = np.full(
        (height, width),
        background,
        dtype=np.int8,
    )

    if orientation == "vertical":
        for barrier, center in zip(barriers, centers):
            input_grid[:, barrier] = barrier_color
            input_grid[
                center - 1:center + 2,
                barrier,
            ] = portal_color

        leading = (leading_parallel, leading_axis)
        trailing = (trailing_parallel, trailing_axis)
    else:
        for barrier, center in zip(barriers, centers):
            input_grid[barrier, :] = barrier_color
            input_grid[
                barrier,
                center - 1:center + 2,
            ] = portal_color

        leading = (leading_axis, leading_parallel)
        trailing = (trailing_axis, trailing_parallel)

    input_grid[leading] = route_color
    input_grid[trailing] = route_color

    output_grid = _draw_route(
        input_grid,
        orientation,
        barriers,
        centers,
        leading,
        trailing,
        route_color,
    )
    return input_grid, output_grid


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        grid.shape,
        tuple(int(x) for x in grid.ravel()),
    )


def _pair_key(pair: dict) -> tuple:
    return (
        _grid_key(pair["input"]),
        _grid_key(pair["output"]),
    )


_SOURCE_INPUT_DIGESTS = frozenset({
    "2baaaca74fc4584356ab76e42c6416fd80bd58bc6d90bb6ae35adadb196ecbfa",
    "5b0c7b71c09004ddf89b36e626bdf11d7ee86f4d00b09b39b13604bb1e337786",
    "ebb17fb0861606453931472a8cb250d732c47b4150471358e940dae05e425ac2",
    "9f0045ca1a93590f5c2a7f3426d841a22b3af3ab11be7990bd72cbdf021c5523",
})


def _input_digest(grid: np.ndarray) -> str:
    compact = np.asarray(grid, dtype=np.uint8)
    return hashlib.sha256(bytes(compact.shape) + compact.tobytes()).hexdigest()


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError(
            "seed must be a nonnegative integer"
        )

    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError(
            "seed must be a nonnegative integer"
        )

    train_count, test_count = _resolve_counts(
        seed_value,
        num_train,
        num_test,
    )

    rng = np.random.default_rng(seed_value)
    puzzle_parameters = sample_puzzle_parameters(rng)

    puzzle = {
        "train": [],
        "test": [],
    }
    used_inputs = set()
    previous_roles: list[tuple[int, ...]] = []
    first_orientation = "vertical" if rng.random() < 0.5 else "horizontal"

    for index in range(train_count):
        if index == 0:
            orientation = first_orientation
        elif index == 1:
            orientation = "horizontal" if first_orientation == "vertical" else "vertical"
        else:
            orientation = "vertical" if rng.random() < 0.5 else "horizontal"

        roles = _sample_roles(rng, puzzle_parameters["anchor_palette"], previous_roles)
        if index == 1 and roles == previous_roles[0]:
            roles = (roles[0], roles[2], roles[1], roles[3])
        previous_roles.append(roles)

        pair_seed = int(
            rng.bit_generator.random_raw()
        )
        pair_rng = np.random.default_rng(pair_seed)

        for _ in range(100):
            input_grid, output_grid = generate_pair(
                pair_rng,
                "source_train",
                puzzle_parameters,
                orientation,
                roles,
                demonstrative=index < 2,
            )
            key = _grid_key(input_grid)

            if key not in used_inputs:
                used_inputs.add(key)
                puzzle["train"].append(
                    {
                        "input": input_grid,
                        "output": output_grid,
                    }
                )
                break
        else:
            raise ValueError(
                "could not generate distinct training inputs"
            )

    for _index in range(test_count):
        orientation = "vertical" if rng.random() < 0.5 else "horizontal"
        roles = _sample_roles(rng, puzzle_parameters["anchor_palette"], previous_roles)
        previous_roles.append(roles)
        pair_seed = int(
            rng.bit_generator.random_raw()
        )
        pair_rng = np.random.default_rng(pair_seed)

        for _ in range(100):
            input_grid, output_grid = generate_pair(
                pair_rng,
                "target_test",
                puzzle_parameters,
                orientation,
                roles,
            )
            key = _grid_key(input_grid)

            if key not in used_inputs:
                used_inputs.add(key)
                puzzle["test"].append(
                    {
                        "input": input_grid,
                        "output": output_grid,
                    }
                )
                break
        else:
            raise ValueError(
                "could not generate distinct test inputs"
            )

    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_input(grid: np.ndarray) -> dict:
    colors, counts = np.unique(
        grid,
        return_counts=True,
    )
    assert colors.size == 4

    route_candidates = [
        int(color)
        for color, count in zip(colors, counts)
        if int(count) == 2
    ]
    assert len(route_candidates) == 1
    route_color = route_candidates[0]

    grouped = {}

    for orientation in ("vertical", "horizontal"):
        if orientation == "vertical":
            axis_length = grid.shape[1]
        else:
            axis_length = grid.shape[0]

        for axis_position in range(axis_length):
            if orientation == "vertical":
                line = grid[:, axis_position]
            else:
                line = grid[axis_position, :]

            line_colors = np.unique(line)
            if line_colors.size != 2:
                continue

            for portal_value in line_colors:
                portal_indices = np.flatnonzero(
                    line == portal_value
                )
                if portal_indices.size != 3:
                    continue

                contiguous = (
                    portal_indices[1]
                    == portal_indices[0] + 1
                    and portal_indices[2]
                    == portal_indices[1] + 1
                )
                if not contiguous:
                    continue

                structural_value = line_colors[
                    line_colors != portal_value
                ]
                assert structural_value.size == 1

                key = (
                    orientation,
                    int(structural_value[0]),
                    int(portal_value),
                )
                grouped.setdefault(key, []).append(
                    (
                        axis_position,
                        int(portal_indices[1]),
                    )
                )

    candidates = [
        (key, entries)
        for key, entries in grouped.items()
        if len(entries) >= 2
    ]
    assert len(candidates) == 1

    (
        orientation,
        barrier_color,
        portal_color,
    ), entries = candidates[0]

    entries = sorted(entries)
    barriers = tuple(
        position for position, _ in entries
    )
    centers = tuple(
        center for _, center in entries
    )

    assert len(set(barriers)) == len(barriers)
    assert route_color not in (
        barrier_color,
        portal_color,
    )

    remaining = (
        set(int(x) for x in colors)
        - {
            route_color,
            barrier_color,
            portal_color,
        }
    )
    assert len(remaining) == 1
    background = remaining.pop()

    endpoint_cells = [
        tuple(int(x) for x in cell)
        for cell in np.argwhere(
            grid == route_color
        )
    ]
    assert len(endpoint_cells) == 2

    if orientation == "vertical":
        endpoint_cells.sort(
            key=lambda cell: cell[1]
        )
        leading, trailing = endpoint_cells

        assert leading[1] <= barriers[0] - 2
        assert trailing[1] >= barriers[-1] + 2
        parallel_length = grid.shape[0]
    else:
        endpoint_cells.sort(
            key=lambda cell: cell[0]
        )
        leading, trailing = endpoint_cells

        assert leading[0] <= barriers[0] - 2
        assert trailing[0] >= barriers[-1] + 2
        parallel_length = grid.shape[1]

    assert all(
        1 <= center <= parallel_length - 2
        for center in centers
    )

    # Reconstructing the complete input proves that there are
    # no extra objects, incomplete barriers, or stray noise.
    reconstructed = np.full(
        grid.shape,
        background,
        dtype=grid.dtype,
    )

    if orientation == "vertical":
        for barrier, center in zip(
            barriers, centers
        ):
            reconstructed[:, barrier] = barrier_color
            reconstructed[
                center - 1:center + 2,
                barrier,
            ] = portal_color
    else:
        for barrier, center in zip(
            barriers, centers
        ):
            reconstructed[barrier, :] = barrier_color
            reconstructed[
                barrier,
                center - 1:center + 2,
            ] = portal_color

    reconstructed[leading] = route_color
    reconstructed[trailing] = route_color
    assert np.array_equal(reconstructed, grid)

    return {
        "orientation": orientation,
        "barriers": barriers,
        "portal_centers": centers,
        "background": background,
        "barrier_color": barrier_color,
        "portal_color": portal_color,
        "route_color": route_color,
        "leading": leading,
        "trailing": trailing,
    }


def _expected_output(
    input_grid: np.ndarray,
    info: dict,
) -> np.ndarray:
    return _draw_route(
        input_grid,
        info["orientation"],
        info["barriers"],
        info["portal_centers"],
        info["leading"],
        info["trailing"],
        info["route_color"],
    )


def _role_tuple(
    info: dict,
) -> tuple[int, int, int, int]:
    return (
        info["background"],
        info["barrier_color"],
        info["portal_color"],
        info["route_color"],
    )


def _parallel_coordinate(
    info: dict,
    endpoint_name: str,
) -> int:
    endpoint = info[endpoint_name]
    if info["orientation"] == "vertical":
        return endpoint[0]
    return endpoint[1]


def _has_reversal(centers: tuple[int, ...]) -> bool:
    deltas = [b - a for a, b in zip(centers, centers[1:])]
    return any(a * b < 0 for a, b in zip(deltas, deltas[1:]))


def _validate_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30
    assert 1 <= grid.shape[1] <= 30
    assert np.issubdtype(
        grid.dtype,
        np.integer,
    )
    assert int(grid.min()) >= 0
    assert int(grid.max()) <= 9


def _validate_pair(pair: dict) -> dict:
    assert isinstance(pair, dict)
    assert set(pair.keys()) == {
        "input",
        "output",
    }

    input_grid = pair["input"]
    output_grid = pair["output"]

    _validate_grid(input_grid)
    _validate_grid(output_grid)
    assert input_grid.shape == output_grid.shape

    info = _analyze_input(input_grid)
    assert 2 <= len(info["barriers"]) <= 6
    assert all(
        later - earlier >= 3
        for earlier, later in zip(info["barriers"], info["barriers"][1:])
    )
    assert _input_digest(input_grid) not in _SOURCE_INPUT_DIGESTS

    expected = _expected_output(
        input_grid,
        info,
    )
    assert np.array_equal(
        output_grid,
        expected,
    )
    assert not np.array_equal(
        input_grid,
        output_grid,
    )

    changed = input_grid != output_grid
    portal_to_route = (
        changed
        & (input_grid == info["portal_color"])
        & (output_grid == info["route_color"])
    )

    # Exactly one cell per portal is crossed.
    assert (
        int(portal_to_route.sum())
        == len(info["barriers"])
    )

    # Every other changed cell is background painted as route.
    assert np.all(
        output_grid[changed]
        == info["route_color"]
    )

    return info


def validate_generated_puzzle(
    puzzle: dict,
) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {
        "train",
        "test",
    }
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert (
        len(puzzle["train"])
        > len(puzzle["test"])
    )

    all_pairs = (
        puzzle["train"]
        + puzzle["test"]
    )
    input_keys = set()
    pair_keys = set()
    train_infos = []
    test_infos = []

    for index, pair in enumerate(all_pairs):
        info = _validate_pair(pair)
        input_key = _grid_key(pair["input"])
        pair_key = _pair_key(pair)

        assert input_key not in input_keys
        assert pair_key not in pair_keys

        input_keys.add(input_key)
        pair_keys.add(pair_key)

        if index < len(puzzle["train"]):
            train_infos.append(info)
        else:
            test_infos.append(info)

    # Each orientation used by a test has a rich demonstration.  This
    # teaches the same positive-axis, incoming-face route under rotation.
    assert {info["orientation"] for info in train_infos} == {
        "vertical", "horizontal"
    }
    assert _role_tuple(train_infos[0]) != _role_tuple(train_infos[1])
    for orientation in ("vertical", "horizontal"):
        demonstrations = [
            info for info in train_infos if info["orientation"] == orientation
        ]
        assert any(
            len(info["barriers"]) >= 3
            and _has_reversal(info["portal_centers"])
            and _parallel_coordinate(info, "leading") != info["portal_centers"][0]
            and _parallel_coordinate(info, "trailing") != info["portal_centers"][-1]
            for info in demonstrations
        )

    # Role colors are pair-level, but all four geometric roles stay distinct.
    assert all(len(set(_role_tuple(info))) == 4 for info in train_infos + test_infos)


def _puzzles_equal(
    left: dict,
    right: dict,
) -> bool:
    if set(left.keys()) != set(right.keys()):
        return False

    for split in ("train", "test"):
        if len(left[split]) != len(right[split]):
            return False

        for pair_left, pair_right in zip(
            left[split],
            right[split],
        ):
            if not np.array_equal(
                pair_left["input"],
                pair_right["input"],
            ):
                return False
            if not np.array_equal(
                pair_left["output"],
                pair_right["output"],
            ):
                return False

    return True


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        _pair_key(pair)
        for split in ("train", "test")
        for pair in puzzle[split]
    )


def _trace_route_independently(input_grid: np.ndarray) -> np.ndarray:
    """A cell-step reconstruction using only visible input geometry."""
    info = _analyze_input(input_grid)
    result = input_grid.copy()
    current = info["leading"]
    orientation = info["orientation"]
    route_color = info["route_color"]
    background = info["background"]
    portal_color = info["portal_color"]

    def walk_to(destination: tuple[int, int]) -> None:
        nonlocal current
        dr = destination[0] - current[0]
        dc = destination[1] - current[1]
        assert dr == 0 or dc == 0
        steps = abs(dr) + abs(dc)
        step_r = 0 if dr == 0 else dr // abs(dr)
        step_c = 0 if dc == 0 else dc // abs(dc)
        for _ in range(steps):
            current = (current[0] + step_r, current[1] + step_c)
            assert int(input_grid[current]) in (background, portal_color, route_color)
            result[current] = route_color

    for barrier, center in zip(info["barriers"], info["portal_centers"]):
        if orientation == "vertical":
            walk_to((current[0], barrier - 1))
            walk_to((center, barrier - 1))
            walk_to((center, barrier))
        else:
            walk_to((barrier - 1, current[1]))
            walk_to((barrier - 1, center))
            walk_to((barrier, center))
    trailing = info["trailing"]
    if orientation == "vertical":
        walk_to((current[0], trailing[1]))
    else:
        walk_to((trailing[0], current[1]))
    walk_to(trailing)
    return result


def _predict_from_training(train_pairs: list[dict], test_input: np.ndarray) -> np.ndarray:
    # The examples establish the incoming-face, center-crossing convention.
    assert { _analyze_input(pair["input"])["orientation"] for pair in train_pairs } == {
        "vertical", "horizontal"
    }
    for pair in train_pairs:
        assert np.array_equal(
            _trace_route_independently(pair["input"]), pair["output"]
        )
    return _trace_route_independently(test_input)


def self_test() -> None:
    automatic = [
        generate_puzzle(seed)
        for seed in range(42)
    ]

    for puzzle in automatic:
        validate_generated_puzzle(puzzle)

        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_input(
                    pair["input"]
                )
                expected = _expected_output(
                    pair["input"],
                    info,
                )
                assert np.array_equal(
                    pair["output"],
                    expected,
                )
                assert np.array_equal(
                    pair["output"],
                    _trace_route_independently(pair["input"]),
                )
        for pair in puzzle["test"]:
            prediction = _predict_from_training(puzzle["train"], pair["input"])
            assert np.array_equal(prediction, pair["output"])

    train_counts = {
        len(puzzle["train"])
        for puzzle in automatic
    }
    test_counts = {
        len(puzzle["test"])
        for puzzle in automatic
    }

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3

    puzzle_keys = {
        _puzzle_key(puzzle)
        for puzzle in automatic
    }

    palette_signatures = {
        (
            _role_tuple(
                _analyze_input(
                    puzzle["train"][0]["input"]
                )
            ),
            _role_tuple(
                _analyze_input(
                    puzzle["train"][1]["input"]
                )
            ),
        )
        for puzzle in automatic
    }

    layout_signatures = set()
    observed_shapes = set()

    for puzzle in automatic:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_input(
                    pair["input"]
                )
                observed_shapes.add(
                    pair["input"].shape
                )
                layout_signatures.add(
                    (
                        pair["input"].shape,
                        info["orientation"],
                        info["barriers"],
                        info["portal_centers"],
                        info["leading"],
                        info["trailing"],
                    )
                )

    assert len(puzzle_keys) >= 38
    assert len(palette_signatures) >= 24
    assert len(layout_signatures) >= 100
    assert len(observed_shapes) >= 25

    review_counts = {"train": Counter(), "test": Counter()}
    orientation_counts = {"train": {}, "test": {}}
    observed_gaps = set()
    observed_lateral_moves = set()
    observed_lead_depths = set()
    observed_trail_depths = set()
    horizontal_first_vertical_test = 0
    mixed_test_orientations = 0
    for puzzle in automatic:
        train_orientations = [
            _analyze_input(pair["input"])["orientation"]
            for pair in puzzle["train"]
        ]
        test_orientations = [
            _analyze_input(pair["input"])["orientation"]
            for pair in puzzle["test"]
        ]
        horizontal_first_vertical_test += (
            train_orientations[0] == "horizontal" and "vertical" in test_orientations
        )
        mixed_test_orientations += len(set(test_orientations)) == 2
        previous_roles = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                info = _analyze_input(grid)
                roles = _role_tuple(info)
                orientation = info["orientation"]
                orientation_counts[split].setdefault(orientation, set()).add(
                    len(info["barriers"])
                )
                progress_axis = 1 if orientation == "vertical" else 0
                progress_length = grid.shape[progress_axis]
                lead_depth = info["leading"][progress_axis]
                trail_depth = progress_length - 1 - info["trailing"][progress_axis]
                interior = lead_depth > 0 and trail_depth > 0
                centers = info["portal_centers"]
                deltas = [b - a for a, b in zip(centers, centers[1:])]
                observed_gaps.update(
                    b - a - 1 for a, b in zip(info["barriers"], info["barriers"][1:])
                )
                observed_lateral_moves.update(abs(delta) for delta in deltas)
                observed_lead_depths.add(lead_depth)
                observed_trail_depths.add(trail_depth)
                repeated = 0 in deltas
                monotone = len(deltas) >= 2 and (
                    all(delta > 0 for delta in deltas)
                    or all(delta < 0 for delta in deltas)
                )
                aligned = (
                    _parallel_coordinate(info, "leading") == centers[0]
                    or _parallel_coordinate(info, "trailing") == centers[-1]
                )
                rectangular = grid.shape[0] != grid.shape[1]
                swapped = any(
                    roles == (old[0], old[2], old[1], old[3])
                    for old in previous_roles
                )
                exact_reuse = roles in previous_roles
                new_palette = bool(previous_roles) and all(
                    set(roles) != set(old) for old in previous_roles
                )
                facts = {
                    "full": grid.shape == (30, 30),
                    "small": min(grid.shape) <= 16,
                    "rectangle": rectangular,
                    "interior_endpoints": interior,
                    "edge_endpoint": not interior,
                    "reversal": _has_reversal(centers),
                    "repeat": repeated,
                    "monotone": monotone,
                    "endpoint_alignment": aligned,
                    "exact_palette_reuse": exact_reuse,
                    "role_exchange": swapped,
                    "new_palette": new_palette,
                    "combined_rectangle_repeat_interior": rectangular and repeated and interior,
                    "combined_full_vertical_reversal": grid.shape == (30, 30) and orientation == "vertical" and _has_reversal(centers),
                }
                review_counts[split].update(key for key, value in facts.items() if value)
                previous_roles.append(roles)

    for split in ("train", "test"):
        assert set(orientation_counts[split]) == {"vertical", "horizontal"}
        assert all(
            observed == set(range(2, 7))
            for observed in orientation_counts[split].values()
        )
        for property_name in (
            "full", "small", "rectangle", "interior_endpoints",
            "edge_endpoint", "reversal", "repeat", "monotone",
            "endpoint_alignment", "exact_palette_reuse", "role_exchange",
            "new_palette", "combined_rectangle_repeat_interior",
            "combined_full_vertical_reversal",
        ):
            assert review_counts[split][property_name] > 0, (split, property_name)
    assert horizontal_first_vertical_test > 0
    assert mixed_test_orientations > 0
    assert min(observed_gaps) == 2 and max(observed_gaps) >= 12
    assert min(observed_lateral_moves) == 0 and max(observed_lateral_moves) >= 20
    assert min(observed_lead_depths) == 0 and max(observed_lead_depths) >= 10
    assert min(observed_trail_depths) == 0 and max(observed_trail_depths) >= 10

    explicit_cases = [
        (101, 3, 1),
        (102, 6, 3),
        (103, 2, 1),
        (104, 10, 1),
        (105, 10, 4),
    ]

    for seed, train_count, test_count in explicit_cases:
        puzzle = generate_puzzle(
            seed,
            train_count,
            test_count,
        )
        assert (
            len(puzzle["train"])
            == train_count
        )
        assert (
            len(puzzle["test"])
            == test_count
        )
        validate_generated_puzzle(puzzle)

    partial_cases = [
        (201, None, 1),
        (202, None, 4),
        (203, 2, None),
        (204, 10, None),
    ]

    for seed, train_count, test_count in partial_cases:
        puzzle = generate_puzzle(
            seed,
            train_count,
            test_count,
        )

        if train_count is not None:
            assert (
                len(puzzle["train"])
                == train_count
            )
        if test_count is not None:
            assert (
                len(puzzle["test"])
                == test_count
            )

        validate_generated_puzzle(puzzle)

    first = generate_puzzle(
        8675309,
        7,
        3,
    )
    second = generate_puzzle(
        8675309,
        7,
        3,
    )
    assert _puzzles_equal(first, second)

    different_a = generate_puzzle(9001)
    different_b = generate_puzzle(9002)
    assert not _puzzles_equal(
        different_a,
        different_b,
    )

    illegal_cases = [
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {
            "num_train": 2,
            "num_test": 2,
        },
        {
            "num_train": 3,
            "num_test": 4,
        },
    ]

    for kwargs in illegal_cases:
        try:
            generate_puzzle(300, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "illegal count request was accepted: "
                f"{kwargs}"
            )
