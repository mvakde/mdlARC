import numpy as np


# Exact source pairs, used only to ensure that no generated pair is copied.
_SOURCE_PAIRS_TEXT = (
    (
        (
            "0000070700", "0000070700", "0007770700", "0007000700",
            "0007077700", "0007070000", "0007070000", "0007970000",
            "0007070000", "0007070000",
        ),
        (
            "0000078700", "0000078700", "0007778700", "0007888700",
            "0007877700", "0007870000", "0007870000", "0007870000",
            "0007870000", "0007870000",
        ),
    ),
    (
        (
            "7070000000", "7070000000", "7077770000", "7000070000",
            "7777070000", "0007077770", "0007900070", "0007777070",
            "0000007070", "0000007070",
        ),
        (
            "7870000000", "7870000000", "7877770000", "7888870000",
            "7777870000", "0007877770", "0007900070", "0007777070",
            "0000007070", "0000007070",
        ),
    ),
    (
        (
            "0070700000", "0070700000", "0070777000", "0070007000",
            "0077797000", "0000707000", "0000707770", "0000700070",
            "0000777070", "0000007070",
        ),
        (
            "0078700000", "0078700000", "0078777000", "0078887000",
            "0077787000", "0000787000", "0000787770", "0000788870",
            "0000777870", "0000007870",
        ),
    ),
    (
        (
            "0007070000", "0007070000", "0007077700", "0007000700",
            "0007770700", "0000070700", "0000070700", "0000070777",
            "0000070000", "0000077777",
        ),
        (
            "0007870000", "0007870000", "0007877700", "0007888700",
            "0007778700", "0000078700", "0000078700", "0000078777",
            "0000078888", "0000077777",
        ),
    ),
    (
        (
            "0000000707", "0000000707", "0000777707", "0000700007",
            "0000707777", "7777707000", "7000097000", "7077777000",
            "7070000000", "7070000000",
        ),
        (
            "0000000787", "0000000787", "0000777787", "0000788887",
            "0000787777", "7777787000", "7000097000", "7077777000",
            "7070000000", "7070000000",
        ),
    ),
)

_SOURCE_PAIR_SIGNATURES = frozenset(
    ("".join(inp), "".join(out)) for inp, out in _SOURCE_PAIRS_TEXT
)

# Side identifiers.
_TOP = 0
_RIGHT = 1
_BOTTOM = 2
_LEFT = 3


def _check_optional_count(value, name, lower, upper):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    if not lower <= int(value) <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")


def _resolve_counts(rng, num_train, num_test):
    _check_optional_count(num_train, "num_train", 2, 10)
    _check_optional_count(num_test, "num_test", 1, 4)

    if num_train is not None:
        num_train = int(num_train)
    if num_test is not None:
        num_test = int(num_test)

    if (
        num_train is not None
        and num_test is not None
        and num_train <= num_test
    ):
        raise ValueError("num_train must be greater than num_test")

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))

    if not (2 <= num_train <= 10):
        raise ValueError("resolved num_train is illegal")
    if not (1 <= num_test <= 4):
        raise ValueError("resolved num_test is illegal")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _sample_puzzle_parameters(rng):
    palette = rng.choice(np.arange(10), size=4, replace=False)
    return {
        "background": int(palette[0]),
        "rail": int(palette[1]),
        "marker": int(palette[2]),
        "fill": int(palette[3]),
        "rotation": int(rng.integers(0, 4)),
    }


def _neighbors(cell, height, width):
    r, c = cell
    if r > 0:
        yield (r - 1, c)
    if r + 1 < height:
        yield (r + 1, c)
    if c > 0:
        yield (r, c - 1)
    if c + 1 < width:
        yield (r, c + 1)


def _append_segment(route, target):
    r, c = route[-1]
    tr, tc = target
    if r != tr and c != tc:
        raise AssertionError("segments must be orthogonal")
    if (r, c) == (tr, tc):
        return

    dr = 0 if r == tr else (1 if tr > r else -1)
    dc = 0 if c == tc else (1 if tc > c else -1)

    while (r, c) != (tr, tc):
        r += dr
        c += dc
        route.append((r, c))


def _sample_levels(rng, height, leg_count, side_exit):
    # Separate horizontal runs enough to keep the outlined channel simple.
    base_gaps = np.array(
        [2] + [3] * (leg_count - 1) + [1 if side_exit else 2],
        dtype=np.int64,
    )
    slack = (height - 1) - int(base_gaps.sum())
    if slack < 0:
        raise AssertionError("height is too small for the route")

    if slack:
        probabilities = np.full(
            leg_count + 1, 1.0 / (leg_count + 1), dtype=float
        )
        base_gaps += rng.multinomial(slack, probabilities)

    levels = []
    position = int(base_gaps[0])
    levels.append(position)
    for i in range(1, leg_count):
        position += int(base_gaps[i])
        levels.append(position)

    if (height - 1) - levels[-1] != int(base_gaps[-1]):
        raise AssertionError("invalid level allocation")

    return levels


def _classify_route_cell(route, index):
    if not 0 < index < len(route) - 1:
        raise AssertionError("a classified route cell must be internal")

    previous = route[index - 1]
    current = route[index]
    following = route[index + 1]

    v1 = (previous[0] - current[0], previous[1] - current[1])
    v2 = (following[0] - current[0], following[1] - current[1])

    if v1[0] + v2[0] == 0 and v1[1] + v2[1] == 0:
        return "straight"
    return "bend"


def _make_canonical_route(rng, regime):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown generation regime")
    if rng.random() < 0.29:
        height = width = 10
    else:
        height = int(rng.integers(10, 25))
        width = int(rng.integers(10, 25))

    side_exit = bool(rng.random() < (0.28 if regime == "source_train" else 0.43))
    maximum_legs = min(4, (height - 1) // 3 if side_exit else (height - 5) // 3 + 1)
    maximum_legs = max(1, maximum_legs)
    if maximum_legs == 1:
        leg_count = 1
        side_exit = False
    else:
        # The source has one- and two-run routes; longer routes are occasional.
        choices = list(range(1 if not side_exit else 2, maximum_legs + 1))
        weights = np.array([3 if n == 2 else (2 if n == 1 else 1) for n in choices], dtype=float)
        weights /= weights.sum()
        leg_count = int(rng.choice(choices, p=weights))
    levels = _sample_levels(rng, height, leg_count, side_exit)

    start_column = int(rng.integers(1, width - 1))
    first_sign = 1 if rng.random() < 0.5 else -1
    signs = [first_sign]
    for _ in range(1, leg_count):
        signs.append(signs[-1] if rng.random() < 0.70 else -signs[-1])

    targets = []
    current_column = start_column
    for i, sign in enumerate(signs):
        if side_exit and i == leg_count - 1:
            target = width - 1 if sign > 0 else 0
            if abs(target - current_column) < 2:
                raise ValueError("side exit would be too short")
        else:
            candidates = list(range(current_column + 2, width - 1)) if sign > 0 else list(range(1, current_column - 1))
            if not candidates:
                raise ValueError("horizontal run does not fit")
            # A triangular length distribution favors the short source runs,
            # while still admitting long traversals on larger canvases.
            distances = np.array([1.0 / (1 + abs(c - current_column) - 2) for c in candidates])
            distances /= distances.sum()
            target = int(rng.choice(candidates, p=distances))
        targets.append(target)
        current_column = target

    route = [(0, start_column)]
    current_column = start_column
    for level, target_column in zip(levels, targets):
        _append_segment(route, (level, current_column))
        _append_segment(route, (level, target_column))
        current_column = target_column
    if not side_exit:
        _append_segment(route, (height - 1, current_column))
    assert len(route) == len(set(route))
    return height, width, route


def _route_walls(route, height, width):
    route_set = set(route)
    walls = {
        neighbor
        for cell in route
        for neighbor in _neighbors(cell, height, width)
        if neighbor not in route_set
    }
    for i in range(1, len(route) - 1):
        if _classify_route_cell(route, i) == "bend":
            r, c = route[i]
            pr, pc = route[i - 1]
            nr, nc = route[i + 1]
            outer = (r + (r - pr) - (nr - r), c + (c - pc) - (nc - c))
            assert 0 <= outer[0] < height and 0 <= outer[1] < width
            assert outer not in route_set
            walls.add(outer)
    return walls


def _fallback_route(index):
    # A bounded, deterministic construction for the unlikely case that
    # random proposals repeatedly make touching rails.
    height = 12 + index % 15
    width = 12 + (7 * index) % 15
    route = [(0, 2)]
    for target in ((3, 2), (3, 5), (7, 5), (7, 9), (height - 1, 9)):
        _append_segment(route, target)
    return height, width, route


def _rail_components(walls, height, width):
    remaining = set(walls)
    components = []
    while remaining:
        start = remaining.pop()
        component = {start}
        stack = [start]
        while stack:
            for neighbor in _neighbors(stack.pop(), height, width):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    stack.append(neighbor)
        components.append(component)
    return components


def _generate_pair(rng, regime, marker_mode, puzzle_parameters, route_override=None):
    if marker_mode not in ("none", "straight", "bend"):
        raise ValueError("unknown marker mode")
    height, width, route = (
        _make_canonical_route(rng, regime)
        if route_override is None else route_override
    )
    walls = _route_walls(route, height, width)
    if len(_rail_components(walls, height, width)) != 2:
        raise ValueError("rails meet or separate")

    marker_position = None
    if marker_mode == "straight":
        candidates = [
            route[i]
            for i in range(1, len(route) - 1)
            if _classify_route_cell(route, i) == "straight"
        ]
        marker_position = candidates[int(rng.integers(0, len(candidates)))]
    elif marker_mode == "bend":
        candidates = [
            route[i]
            for i in range(1, len(route) - 1)
            if _classify_route_cell(route, i) == "bend"
        ]
        if len(candidates) < 3:
            raise ValueError("stopping example needs a later bend")
        weights = np.array([1.0 if i == 0 else (2.5 if i == 2 else 1.5) for i in range(len(candidates))])
        weights /= weights.sum()
        marker_position = candidates[int(rng.choice(len(candidates), p=weights))]

    background = puzzle_parameters["background"]
    rail = puzzle_parameters["rail"]
    marker = puzzle_parameters["marker"]
    fill = puzzle_parameters["fill"]

    input_grid = np.full(
        (height, width), background, dtype=np.int8
    )
    for r, c in walls:
        input_grid[r, c] = rail
    if marker_position is not None:
        input_grid[marker_position] = marker

    output_grid = input_grid.copy()

    if marker_mode == "bend":
        marker_index = route.index(marker_position)
        cells_to_fill = route[:marker_index]
    else:
        cells_to_fill = route

    for r, c in cells_to_fill:
        output_grid[r, c] = fill

    rotation = puzzle_parameters["rotation"]
    if rotation:
        input_grid = np.rot90(input_grid, rotation).copy()
        output_grid = np.rot90(output_grid, rotation).copy()

    return input_grid, output_grid


def _array_signature(array):
    return (
        tuple(int(v) for v in array.shape),
        tuple(int(v) for v in array.ravel()),
    )


def _pair_signature(pair):
    return (
        _array_signature(pair["input"]),
        _array_signature(pair["output"]),
    )


def _source_pair_signature(input_grid, output_grid):
    if input_grid.shape != (10, 10) or output_grid.shape != (10, 10):
        return None
    inp = "".join(str(int(v)) for v in input_grid.ravel())
    out = "".join(str(int(v)) for v in output_grid.ravel())
    return inp, out


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(
        rng, num_train, num_test
    )

    puzzle_parameters = _sample_puzzle_parameters(rng)

    # With two pairs, demonstrate both marker classifications. With three
    # or more, also demonstrate complete filling without a marker.
    base_modes = ["straight", "bend"]
    if train_count >= 3:
        base_modes.append("none")

    while len(base_modes) < train_count:
        base_modes.append(str(rng.choice(("straight", "none", "bend"))))

    permutation = rng.permutation(train_count)
    train_modes = [base_modes[int(i)] for i in permutation]

    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    seen_pairs = set()

    test_modes = list(rng.choice(("bend", "straight", "none"), size=test_count, p=(0.52, 0.28, 0.20)))
    generation_specs = [
        ("train", "source_train", mode) for mode in train_modes
    ] + [
        ("test", "target_test", mode) for mode in test_modes
    ]

    for split, regime, marker_mode in generation_specs:
        accepted = False

        for attempt in range(280):
            child_seed = int(
                rng.integers(0, np.iinfo(np.int64).max)
            )
            pair_rng = np.random.default_rng(child_seed)
            try:
                input_grid, output_grid = _generate_pair(
                    pair_rng, regime, marker_mode, puzzle_parameters,
                    _fallback_route(attempt - 250 + len(seen_inputs)) if attempt >= 250 else None,
                )
                _expected_output_for_side(
                    input_grid,
                    (_TOP - puzzle_parameters["rotation"]) % 4,
                    puzzle_parameters["background"],
                    puzzle_parameters["rail"],
                    puzzle_parameters["marker"],
                    puzzle_parameters["fill"],
                )
            except (AssertionError, ValueError):
                continue

            candidate = {
                "input": input_grid,
                "output": output_grid,
            }
            input_signature = _array_signature(input_grid)
            pair_signature = _pair_signature(candidate)

            if input_signature in seen_inputs:
                continue
            if pair_signature in seen_pairs:
                continue
            if (
                _source_pair_signature(input_grid, output_grid)
                in _SOURCE_PAIR_SIGNATURES
            ):
                continue

            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            puzzle[split].append(candidate)
            accepted = True
            break

        if not accepted:
            raise RuntimeError("could not generate a unique valid pair")

    validate_generated_puzzle(puzzle)
    return puzzle


def _portal_cells(grid, side, rail):
    height, width = grid.shape
    portals = []

    if side == _TOP:
        for c in range(1, width - 1):
            if (
                int(grid[0, c]) != rail
                and int(grid[0, c - 1]) == rail
                and int(grid[0, c + 1]) == rail
            ):
                portals.append((0, c))

    elif side == _BOTTOM:
        r = height - 1
        for c in range(1, width - 1):
            if (
                int(grid[r, c]) != rail
                and int(grid[r, c - 1]) == rail
                and int(grid[r, c + 1]) == rail
            ):
                portals.append((r, c))

    elif side == _LEFT:
        for r in range(1, height - 1):
            if (
                int(grid[r, 0]) != rail
                and int(grid[r - 1, 0]) == rail
                and int(grid[r + 1, 0]) == rail
            ):
                portals.append((r, 0))

    elif side == _RIGHT:
        c = width - 1
        for r in range(1, height - 1):
            if (
                int(grid[r, c]) != rail
                and int(grid[r - 1, c]) == rail
                and int(grid[r + 1, c]) == rail
            ):
                portals.append((r, c))

    else:
        raise AssertionError("unknown side")

    return portals


def _recover_ordered_route(input_grid, side, background, rail, marker):
    height, width = input_grid.shape
    portals = _portal_cells(input_grid, side, rail)
    assert len(portals) == 1
    start = portals[0]

    component = {start}
    stack = [start]

    while stack:
        cell = stack.pop()
        for neighbor in _neighbors(cell, height, width):
            if (
                neighbor not in component
                and int(input_grid[neighbor]) != rail
            ):
                component.add(neighbor)
                stack.append(neighbor)

    assert len(component) >= 5

    degrees = {}
    for cell in component:
        degrees[cell] = sum(
            neighbor in component
            for neighbor in _neighbors(cell, height, width)
        )
        assert degrees[cell] in (1, 2)

    endpoints = {cell for cell, degree in degrees.items() if degree == 1}
    assert len(endpoints) == 2
    assert start in endpoints

    other_endpoint = next(cell for cell in endpoints if cell != start)
    assert other_endpoint[0] in (0, height - 1) or other_endpoint[1] in (0, width - 1)

    boundary_cells = {
        (r, c)
        for r, c in component
        if r in (0, height - 1) or c in (0, width - 1)
    }
    assert boundary_cells == endpoints

    ordered = []
    previous = None
    current = start

    while True:
        ordered.append(current)
        choices = [
            neighbor
            for neighbor in _neighbors(current, height, width)
            if neighbor in component and neighbor != previous
        ]
        if not choices:
            break
        assert len(choices) == 1
        previous, current = current, choices[0]

    assert len(ordered) == len(component)
    assert ordered[-1] == other_endpoint

    expected_walls = _route_walls(ordered, height, width)

    observed_walls = {
        (int(r), int(c))
        for r, c in np.argwhere(input_grid == rail)
    }
    assert observed_walls == expected_walls
    assert len(_rail_components(observed_walls, height, width)) == 2

    for r in range(height):
        for c in range(width):
            cell = (r, c)
            value = int(input_grid[cell])
            if cell in component:
                assert value in (background, marker)
            elif cell in expected_walls:
                assert value == rail
            else:
                assert value == background

    marker_positions = [
        cell for cell in ordered if int(input_grid[cell]) == marker
    ]
    assert len(marker_positions) <= 1

    return ordered, marker_positions


def _expected_output_for_side(
    input_grid, side, background, rail, marker, fill
):
    route, marker_positions = _recover_ordered_route(
        input_grid, side, background, rail, marker
    )

    expected = input_grid.copy()
    marker_index = None

    if not marker_positions:
        mode = "none"
        cells_to_fill = route
    else:
        marker_position = marker_positions[0]
        marker_index = route.index(marker_position)
        assert 0 < marker_index < len(route) - 1
        mode = _classify_route_cell(route, marker_index)
        r, c = marker_position
        rail_offsets = {
            (nr - r, nc - c)
            for nr, nc in _neighbors(marker_position, *input_grid.shape)
            if int(input_grid[nr, nc]) == rail
        }
        assert len(rail_offsets) == 2
        first, second = tuple(rail_offsets)
        assert (first[0] + second[0] == 0 and first[1] + second[1] == 0) == (mode == "straight")

        if mode == "straight":
            cells_to_fill = route
        else:
            cells_to_fill = route[:marker_index]

    for cell in cells_to_fill:
        expected[cell] = fill

    return {
        "expected": expected,
        "route": route,
        "mode": mode,
        "marker_index": marker_index,
    }


def _infer_roles_from_training(training_pairs):
    input_colors = set()
    output_colors = set()
    changed_input_sets = []

    for pair in training_pairs:
        input_grid = pair["input"]
        output_grid = pair["output"]

        input_colors.update(int(v) for v in np.unique(input_grid))
        output_colors.update(int(v) for v in np.unique(output_grid))

        changed = input_grid != output_grid
        assert np.any(changed)
        changed_input_sets.append(
            {int(v) for v in np.unique(input_grid[changed])}
        )

    new_output_colors = output_colors - input_colors
    assert len(new_output_colors) == 1
    fill = next(iter(new_output_colors))

    background_candidates = set(changed_input_sets[0])
    for values in changed_input_sets[1:]:
        background_candidates.intersection_update(values)
    assert len(background_candidates) == 1
    background = next(iter(background_candidates))

    changed_nonbackground = set().union(*changed_input_sets) - {background}
    assert len(changed_nonbackground) == 1
    marker = next(iter(changed_nonbackground))

    rail_candidates = input_colors - {background, marker}
    assert len(rail_candidates) == 1
    rail = next(iter(rail_candidates))

    assert len({background, rail, marker, fill}) == 4

    return {
        "background": background,
        "rail": rail,
        "marker": marker,
        "fill": fill,
    }


def _analyze_pair_for_side(pair, side, roles):
    input_grid = pair["input"]
    output_grid = pair["output"]

    result = _expected_output_for_side(
        input_grid,
        side,
        roles["background"],
        roles["rail"],
        roles["marker"],
        roles["fill"],
    )
    assert np.array_equal(result["expected"], output_grid)
    return result


def _frame_coordinate(shape, side, cell):
    height, width = shape
    r, c = cell

    if side == _TOP:
        return r, c
    if side == _LEFT:
        return c, height - 1 - r
    if side == _BOTTOM:
        return height - 1 - r, width - 1 - c
    if side == _RIGHT:
        return width - 1 - c, r
    raise AssertionError("unknown side")


def _lateral_size(shape, side):
    height, width = shape
    return width if side in (_TOP, _BOTTOM) else height


def _route_frame_metrics(route, shape, side):
    coordinates = [
        _frame_coordinate(shape, side, cell) for cell in route
    ]

    first_lateral_edge = None
    first_lateral_sign = None
    lateral_runs = []
    current_run = []

    for edge_index in range(len(coordinates) - 1):
        d1, t1 = coordinates[edge_index]
        d2, t2 = coordinates[edge_index + 1]
        dd = d2 - d1
        dt = t2 - t1

        assert abs(dd) + abs(dt) == 1
        assert dd in (0, 1)

        if first_lateral_edge is None and dt != 0:
            first_lateral_edge = edge_index
            first_lateral_sign = 1 if dt > 0 else -1

        if dt != 0:
            current_run.append(dt)
        elif current_run:
            lateral_runs.append(current_run)
            current_run = []

    if current_run:
        lateral_runs.append(current_run)

    assert first_lateral_edge is not None

    return {
        "coordinates": coordinates,
        "first_lateral_edge": first_lateral_edge,
        "first_lateral_sign": first_lateral_sign,
        "lateral_signs": tuple(1 if run[0] > 0 else -1 for run in lateral_runs),
        "lateral_lengths": tuple(len(run) for run in lateral_runs),
        "bend_indices": tuple(i for i in range(1, len(route) - 1) if _classify_route_cell(route, i) == "bend"),
        "exit_side": next(s for s in (_TOP, _RIGHT, _BOTTOM, _LEFT) if route[-1] in _side_cells(shape, s)),
    }


def _side_cells(shape, side):
    height, width = shape
    if side == _TOP:
        return {(0, c) for c in range(width)}
    if side == _BOTTOM:
        return {(height - 1, c) for c in range(width)}
    if side == _LEFT:
        return {(r, 0) for r in range(height)}
    return {(r, width - 1) for r in range(height)}


def _check_pair_geometry(detail, shape, side):
    metrics = _route_frame_metrics(detail["route"], shape, side)
    coordinates = metrics["coordinates"]
    assert coordinates[0][0] == 0
    assert coordinates[0][1] in range(1, _lateral_size(shape, side) - 1)
    assert metrics["first_lateral_edge"] >= 2
    assert all(length >= 2 for length in metrics["lateral_lengths"])
    assert 1 <= len(metrics["lateral_lengths"]) <= 4
    assert len(metrics["bend_indices"]) in range(2, 9)
    if detail["mode"] == "bend":
        assert detail["marker_index"] in metrics["bend_indices"]
        assert len(metrics["bend_indices"]) >= 3
    return metrics


def _analyze_puzzle(puzzle):
    roles = _infer_roles_from_training(puzzle["train"])

    valid_training_sides = {}
    for side in (_TOP, _RIGHT, _BOTTOM, _LEFT):
        details = []
        try:
            for pair in puzzle["train"]:
                details.append(
                    _analyze_pair_for_side(pair, side, roles)
                )
        except AssertionError:
            continue
        valid_training_sides[side] = details

    # Training alone must reveal the global entrance direction.
    assert len(valid_training_sides) == 1
    entry_side = next(iter(valid_training_sides))
    train_details = valid_training_sides[entry_side]

    test_details = [
        _analyze_pair_for_side(pair, entry_side, roles)
        for pair in puzzle["test"]
    ]

    train_modes = [detail["mode"] for detail in train_details]
    test_modes = [detail["mode"] for detail in test_details]

    assert "straight" in train_modes
    assert "bend" in train_modes
    if len(train_modes) >= 3:
        assert "none" in train_modes

    for pair, detail in zip(puzzle["train"] + puzzle["test"], train_details + test_details):
        _check_pair_geometry(detail, pair["input"].shape, entry_side)

    return {
        "roles": roles,
        "entry_side": entry_side,
        "train_details": train_details,
        "test_details": test_details,
    }


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])

    assert 2 <= train_count <= 10
    assert 1 <= test_count <= 4
    assert train_count > test_count

    all_pairs = puzzle["train"] + puzzle["test"]
    seen_inputs = set()
    seen_pairs = set()

    for pair in all_pairs:
        assert isinstance(pair, dict)
        assert set(pair.keys()) == {"input", "output"}

        input_grid = pair["input"]
        output_grid = pair["output"]

        assert isinstance(input_grid, np.ndarray)
        assert isinstance(output_grid, np.ndarray)
        assert input_grid.ndim == 2
        assert output_grid.ndim == 2
        assert input_grid.shape == output_grid.shape

        for grid in (input_grid, output_grid):
            height, width = grid.shape
            assert 1 <= height <= 30
            assert 1 <= width <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert int(grid.min()) >= 0
            assert int(grid.max()) <= 9

        assert not np.array_equal(input_grid, output_grid)

        input_signature = _array_signature(input_grid)
        pair_signature = _pair_signature(pair)

        assert input_signature not in seen_inputs
        assert pair_signature not in seen_pairs
        seen_inputs.add(input_signature)
        seen_pairs.add(pair_signature)

        source_signature = _source_pair_signature(
            input_grid, output_grid
        )
        assert source_signature not in _SOURCE_PAIR_SIGNATURES

    analysis = _analyze_puzzle(puzzle)
    roles = analysis["roles"]

    # Enforce puzzle-wide role consistency, including absence of the fill
    # color from every input.
    allowed_input = {
        roles["background"], roles["rail"], roles["marker"]
    }
    allowed_output = allowed_input | {roles["fill"]}

    for pair in all_pairs:
        input_colors = {int(v) for v in np.unique(pair["input"])}
        output_colors = {int(v) for v in np.unique(pair["output"])}

        assert input_colors <= allowed_input
        assert output_colors <= allowed_output
        assert roles["fill"] not in input_colors
        assert roles["rail"] in input_colors
        assert roles["background"] in input_colors


def _puzzles_equal(first, second):
    if set(first.keys()) != set(second.keys()):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for pair_a, pair_b in zip(first[split], second[split]):
            if not np.array_equal(pair_a["input"], pair_b["input"]):
                return False
            if not np.array_equal(pair_a["output"], pair_b["output"]):
                return False

    return True


def _puzzle_signature(puzzle):
    return tuple(
        (
            split,
            tuple(_pair_signature(pair) for pair in puzzle[split]),
        )
        for split in ("train", "test")
    )


def _copy_puzzle(puzzle):
    return {
        split: [
            {"input": pair["input"].copy(), "output": pair["output"].copy()}
            for pair in puzzle[split]
        ]
        for split in ("train", "test")
    }


def self_test() -> None:
    automatic_puzzles = []

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        _analyze_puzzle(puzzle)
        automatic_puzzles.append(puzzle)

    train_counts = {
        len(puzzle["train"]) for puzzle in automatic_puzzles
    }
    test_counts = {
        len(puzzle["test"]) for puzzle in automatic_puzzles
    }
    assert len(train_counts) >= 4
    assert len(test_counts) >= 3

    # Deliberately damage each of the two task-specific invariants.
    first = automatic_puzzles[0]
    first_analysis = _analyze_puzzle(first)
    roles = first_analysis["roles"]
    route = first_analysis["train_details"][0]["route"]
    height, width = first["train"][0]["input"].shape
    route_set = set(route)
    neighbor_walls = {
        neighbor for cell in route
        for neighbor in _neighbors(cell, height, width)
        if neighbor not in route_set
    }
    observed_walls = {
        (int(r), int(c))
        for r, c in np.argwhere(first["train"][0]["input"] == roles["rail"])
    }
    missing_corner = _copy_puzzle(first)
    corner = next(iter(observed_walls - neighbor_walls))
    missing_corner["train"][0]["input"][corner] = roles["background"]
    missing_corner["train"][0]["output"][corner] = roles["background"]
    try:
        validate_generated_puzzle(missing_corner)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing outer corner was accepted")

    bend_pair_index = next(i for i, d in enumerate(first_analysis["train_details"]) if d["mode"] == "bend")
    bend_detail = first_analysis["train_details"][bend_pair_index]
    bad_stop = _copy_puzzle(first)
    downstream_cell = bend_detail["route"][bend_detail["marker_index"] + 1]
    bad_stop["train"][bend_pair_index]["output"][downstream_cell] = roles["fill"]
    try:
        validate_generated_puzzle(bad_stop)
    except AssertionError:
        pass
    else:
        raise AssertionError("fill beyond a stopping marker was accepted")

    deterministic_a = generate_puzzle(104729, 7, 3)
    deterministic_b = generate_puzzle(104729, 7, 3)
    assert _puzzles_equal(deterministic_a, deterministic_b)

    distinct_seed_signatures = {
        _puzzle_signature(generate_puzzle(seed, 5, 2))
        for seed in range(200, 208)
    }
    assert len(distinct_seed_signatures) >= 7

    boundary_cases = (
        (2, 1),
        (10, 1),
        (10, 4),
    )
    for index, (train_count, test_count) in enumerate(boundary_cases):
        puzzle = generate_puzzle(
            300 + index,
            num_train=train_count,
            num_test=test_count,
        )
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)

    partial_cases = (
        (None, 1),
        (None, 4),
        (2, None),
        (10, None),
    )
    for index, (train_count, test_count) in enumerate(partial_cases):
        puzzle = generate_puzzle(
            400 + index,
            num_train=train_count,
            num_test=test_count,
        )
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)

    illegal_cases = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    )
    for kwargs in illegal_cases:
        try:
            generate_puzzle(500, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"illegal counts did not raise ValueError: {kwargs}"
            )

    palettes = set()
    entry_sides = set()
    shapes = set()
    literal_input_signatures = set()
    normalized_layout_signatures = set()
    total_pairs = 0
    split_features = {"train": [], "test": []}
    train_direction_variety = 0
    test_direction_variety = 0
    exit_variety = 0
    shape_variety = 0

    for puzzle in automatic_puzzles:
        analysis = _analyze_puzzle(puzzle)
        roles = analysis["roles"]
        palettes.add(
            (
                roles["background"],
                roles["rail"],
                roles["marker"],
                roles["fill"],
            )
        )
        entry_sides.add(analysis["entry_side"])

        puzzle_features = []
        for split, details in (("train", analysis["train_details"]), ("test", analysis["test_details"])):
            for pair, detail in zip(puzzle[split], details):
                metric = _route_frame_metrics(
                    detail["route"], pair["input"].shape,
                    analysis["entry_side"],
                )
                feature = {
                    "shape": pair["input"].shape,
                    "mode": detail["mode"],
                    "marker_index": detail["marker_index"],
                    "route_length": len(detail["route"]),
                    "bend_indices": metric["bend_indices"],
                    "signs": metric["lateral_signs"],
                    "lengths": metric["lateral_lengths"],
                    "exit": metric["exit_side"],
                    "entry": analysis["entry_side"],
                }
                split_features[split].append(feature)
                puzzle_features.append((split, feature))

        train_features = [f for s, f in puzzle_features if s == "train"]
        test_features = [f for s, f in puzzle_features if s == "test"]
        train_direction_variety += len({f["signs"][0] for f in train_features}) == 2
        test_direction_variety += len({f["signs"][0] for f in test_features}) == 2
        exit_variety += len({f["exit"] for _, f in puzzle_features}) >= 2
        shape_variety += len({f["shape"] for _, f in puzzle_features}) >= 2

        for pair in puzzle["train"] + puzzle["test"]:
            total_pairs += 1
            input_grid = pair["input"]
            output_grid = pair["output"]

            shapes.add(input_grid.shape)
            literal_input_signatures.add(
                _array_signature(input_grid)
            )

            normalized = np.zeros(input_grid.shape, dtype=np.int8)
            normalized[input_grid == roles["rail"]] = 1
            normalized[input_grid == roles["marker"]] = 2
            normalized_layout_signatures.add(
                _array_signature(normalized)
            )

            assert input_grid.shape == output_grid.shape
            assert 1 <= input_grid.shape[0] <= 30
            assert 1 <= input_grid.shape[1] <= 30
            assert np.issubdtype(input_grid.dtype, np.integer)
            assert np.issubdtype(output_grid.dtype, np.integer)
            assert int(input_grid.min()) >= 0
            assert int(input_grid.max()) <= 9
            assert int(output_grid.min()) >= 0
            assert int(output_grid.max()) <= 9
            assert (
                _source_pair_signature(input_grid, output_grid)
                not in _SOURCE_PAIR_SIGNATURES
            )

    assert len(palettes) >= 20
    assert len(entry_sides) >= 3
    assert len(shapes) >= 30
    assert len(literal_input_signatures) >= int(0.90 * total_pairs)
    assert len(normalized_layout_signatures) >= int(0.80 * total_pairs)

    train_features = split_features["train"]
    test_features = split_features["test"]
    assert shape_variety >= 30
    assert train_direction_variety >= 12
    assert test_direction_variety >= 5
    assert exit_variety >= 15

    for features in (train_features, test_features):
        assert {f["signs"][0] for f in features} == {-1, 1}
        assert any(f["shape"] == (10, 10) for f in features)
        assert any(len(f["bend_indices"]) == 2 for f in features)
        assert any(len(f["bend_indices"]) == 3 for f in features)
        assert any(len(f["bend_indices"]) == 4 for f in features)
        assert any(len(f["bend_indices"]) >= 5 for f in features)
        assert any(len(f["signs"]) >= 2 and f["signs"][0] == f["signs"][1] for f in features)
        assert any(len(f["signs"]) >= 2 and f["signs"][0] != f["signs"][1] for f in features)
        assert any(f["exit"] not in (f["entry"], (f["entry"] + 2) % 4) for f in features)
        assert any(f["shape"] == (10, 10) and len(f["signs"]) >= 2 and f["signs"][0] == f["signs"][1] and f["exit"] not in (f["entry"], (f["entry"] + 2) % 4) for f in features)
        assert min(length for f in features for length in f["lengths"]) <= 2
        assert max(length for f in features for length in f["lengths"]) >= 8

    assert {f["mode"] for f in train_features} == {"straight", "bend", "none"}
    assert {f["mode"] for f in test_features} == {"straight", "bend", "none"}
    for features in (train_features, test_features):
        ordinals = {
            f["bend_indices"].index(f["marker_index"]) + 1
            for f in features if f["mode"] == "bend"
        }
        assert {1, 2, 3} <= ordinals
        straight_fraction_bins = {
            min(2, 3 * f["marker_index"] // f["route_length"])
            for f in features if f["mode"] == "straight"
        }
        assert len(straight_fraction_bins) >= 2
