import numpy as np


# Rejecting an exact source input is stronger than rejecting an exact
# source input/output pair, because identical pairs necessarily have
# identical inputs.
_SOURCE_INPUTS = (
    (
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0002222222220000",
        "0000000000000000",
        "0000000000000000",
        "0000000000000000",
        "0000003333333330",
        "0000000000000000",
        "0000000000000000",
        "0000000000010000",
        "0000000000001000",
        "0000000000000100",
        "0000000000000010",
    ),
    (
        "0000000000000000",
        "0000000030000001",
        "0000000030000010",
        "0000000030000100",
        "0000000030001000",
        "0000000030010000",
        "0000000030000000",
        "0000000030000000",
        "0000000030000000",
        "0000000030000000",
        "0000000030000000",
        "0000000030000000",
        "0000000030000000",
        "0000000030000000",
        "0000000030000000",
        "0000000000000000",
    ),
    (
        "1000000000000000",
        "0100000000000000",
        "0010000020000000",
        "0001000020000000",
        "0000100020000000",
        "0000010020000000",
        "0000000020000000",
        "0000000020000000",
        "0000000020000000",
        "0000000020000000",
        "0000000020000000",
        "0000000020000000",
        "0000000020000000",
        "0000000020000000",
        "0000000000000000",
        "0000000000000000",
    ),
    (
        "0100000000000000",
        "0010300000000000",
        "0001300000000300",
        "2000300000000300",
        "2000300000000300",
        "2000200030200300",
        "2000200030200300",
        "2000200030200300",
        "2000200030200000",
        "2000200200200000",
        "2000200200200000",
        "2000200200200000",
        "0000200200000000",
        "0000200200000000",
        "0000000200000000",
        "0000000000000000",
    ),
    (
        "0000000001000000",
        "0000200010000000",
        "0000200100000000",
        "0000200000000000",
        "0000200000000000",
        "0000200000000000",
        "0000200000300000",
        "0000200000300000",
        "0000200000300000",
        "0000200000300000",
        "0000200000300000",
        "0000200000300020",
        "0000200000300020",
        "0000200000000020",
        "0000200000000020",
        "0000000000000000",
    ),
)

_SOURCE_INPUT_SIGNATURES = {"".join(rows) for rows in _SOURCE_INPUTS}


def _normalise_seed(seed: int) -> int:
    if (
        isinstance(seed, (bool, np.bool_))
        or not isinstance(seed, (int, np.integer))
    ):
        raise ValueError("seed must be an integer")
    return int(seed) % (1 << 64)


def _count_value(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(
    seed_value: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _count_value("num_train", num_train, 2, 10)
    num_test = _count_value("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        num_test = 1 + seed_value % 4
        first_train = max(2, num_test + 1)
        span = 11 - first_train
        num_train = first_train + (
            (3 * seed_value + seed_value // 4) % span
        )
    elif num_train is None:
        first_train = max(2, num_test + 1)
        span = 11 - first_train
        num_train = first_train + ((5 * seed_value + 1) % span)
    elif num_test is None:
        last_test = min(4, num_train - 1)
        num_test = 1 + ((7 * seed_value + 2) % last_test)

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return int(num_train), int(num_test)


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        grid.shape,
        tuple(int(value) for value in grid.ravel()),
    )


def _is_source_input(grid: np.ndarray) -> bool:
    if grid.shape != (16, 16):
        return False
    signature = "".join(str(int(value)) for value in grid.ravel())
    return signature in _SOURCE_INPUT_SIGNATURES


def _components(
    grid: np.ndarray,
    color: int,
) -> list[list[tuple[int, int]]]:
    remaining = {
        (int(row), int(col))
        for row, col in np.argwhere(grid == color)
    }
    result = []

    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]

        while stack:
            row, col = stack.pop()
            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            ):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
                    component.append(neighbor)

        result.append(component)

    return result


def _component_orientation(
    component: list[tuple[int, int]],
) -> str:
    assert len(component) >= 3, (
        "every bar component must contain at least three cells"
    )

    rows = {point[0] for point in component}
    cols = {point[1] for point in component}

    if len(rows) == 1:
        values = sorted(point[1] for point in component)
        assert values == list(range(values[0], values[-1] + 1))
        return "horizontal"

    if len(cols) == 1:
        values = sorted(point[0] for point in component)
        assert values == list(range(values[0], values[-1] + 1))
        return "vertical"

    raise AssertionError("bar components must be straight and axis-aligned")


def _bar_orientation(
    grid: np.ndarray,
    row: int,
    col: int,
    color: int,
) -> str:
    for component in _components(grid, color):
        if (row, col) in component:
            return _component_orientation(component)
    raise AssertionError("bar cell is not part of a valid component")


def _all_bars_well_formed(
    grid: np.ndarray,
    roles: dict,
) -> None:
    for color in (roles["transmit"], roles["reflect"]):
        for component in _components(grid, color):
            _component_orientation(component)


def _seed_information(
    grid: np.ndarray,
    trail: int,
) -> dict:
    height, width = grid.shape
    points = {
        (int(row), int(col))
        for row, col in np.argwhere(grid == trail)
    }
    assert len(points) >= 3

    neighbors = {}
    for row, col in points:
        adjacent = []
        for dr, dc in (
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ):
            point = (row + dr, col + dc)
            if point in points:
                adjacent.append(point)
        neighbors[(row, col)] = adjacent

    endpoints = [
        point
        for point, adjacent in neighbors.items()
        if len(adjacent) == 1
    ]
    assert len(endpoints) == 2
    assert all(len(adjacent) in (1, 2) for adjacent in neighbors.values())

    boundary_endpoints = [
        point
        for point in endpoints
        if (
            point[0] in (0, height - 1)
            or point[1] in (0, width - 1)
        )
    ]
    assert len(boundary_endpoints) == 1

    order = [boundary_endpoints[0]]
    previous = None

    while len(order) < len(points):
        choices = [
            point
            for point in neighbors[order[-1]]
            if point != previous
        ]
        assert len(choices) == 1
        previous, next_point = order[-1], choices[0]
        order.append(next_point)

    assert len(set(order)) == len(points)

    dr = order[1][0] - order[0][0]
    dc = order[1][1] - order[0][1]
    assert abs(dr) == abs(dc) == 1

    for first, second in zip(order, order[1:]):
        assert (
            second[0] - first[0],
            second[1] - first[1],
        ) == (dr, dc)

    boundary_row, boundary_col = order[0]
    if boundary_row == 0:
        entry_side = "top"
    elif boundary_row == height - 1:
        entry_side = "bottom"
    elif boundary_col == 0:
        entry_side = "left"
    else:
        entry_side = "right"

    return {
        "length": len(points),
        "tip": order[-1],
        "direction": (dr, dc),
        "entry_side": entry_side,
    }


def _trace_ray(
    input_grid: np.ndarray,
    roles: dict,
) -> tuple[np.ndarray, dict]:
    background = roles["background"]
    trail = roles["trail"]
    transmit = roles["transmit"]
    reflect = roles["reflect"]

    allowed = {background, trail, transmit, reflect}
    assert set(int(value) for value in np.unique(input_grid)) <= allowed

    seed = _seed_information(input_grid, trail)
    output = input_grid.copy()

    height, width = input_grid.shape
    row, col = seed["tip"]
    dr, dc = seed["direction"]

    events = []
    path = []
    states = set()
    background_steps = 0

    for _ in range(4 * height * width + 1):
        state = (row, col, dr, dc)
        assert state not in states, "ray route must be finite and non-cyclic"
        states.add(state)

        next_row = row + dr
        next_col = col + dc

        if not (
            0 <= next_row < height
            and 0 <= next_col < width
        ):
            return output, {
                "seed": seed,
                "events": events,
                "path": path,
                "background_steps": background_steps,
                "exit": (next_row, next_col),
            }

        value = int(output[next_row, next_col])

        if value == background:
            output[next_row, next_col] = trail
            row, col = next_row, next_col
            path.append((row, col))
            background_steps += 1

        elif value == transmit:
            orientation = _bar_orientation(
                input_grid,
                next_row,
                next_col,
                transmit,
            )
            events.append(
                ("transmit", orientation, next_row, next_col)
            )
            output[next_row, next_col] = trail
            row, col = next_row, next_col
            path.append((row, col))

        elif value == reflect:
            orientation = _bar_orientation(
                input_grid,
                next_row,
                next_col,
                reflect,
            )
            events.append(
                ("reflect", orientation, next_row, next_col)
            )
            if orientation == "vertical":
                dc = -dc
            else:
                dr = -dr

        else:
            raise AssertionError(
                "ray may not collide with its own trail or an unknown cell"
            )

    raise AssertionError("ray did not terminate")




def _bar_cells(target, orientation, start, length):
    row, col = target
    if orientation == "vertical":
        return [(r, col) for r in range(start, start + length)]
    return [(row, c) for c in range(start, start + length)]


def _place_bar(grid, target, orientation, color, roles, rng, forbidden, long=False):
    height, width = grid.shape
    axis = height if orientation == "vertical" else width
    coordinate = target[0] if orientation == "vertical" else target[1]
    upper = min(axis, 18 if long else 14)
    if upper < 3:
        return False
    lengths = list(range(3, upper + 1))
    rng.shuffle(lengths)
    for length in lengths:
        starts = list(range(max(0, coordinate - length + 1),
                            min(coordinate, axis - length) + 1))
        rng.shuffle(starts)
        for start in starts:
            cells = _bar_cells(target, orientation, start, length)
            if any(cell in forbidden for cell in cells if cell != target):
                continue
            if any(grid[cell] != roles["background"] for cell in cells):
                continue
            # Keep separate physical segments visually distinct. A crossing of
            # different bar colors would obscure the struck orientation.
            if any(
                grid[nr, nc] in (roles["transmit"], roles["reflect"])
                for r, c in cells
                for nr in range(max(0, r - 1), min(height, r + 2))
                for nc in range(max(0, c - 1), min(width, c + 2))
            ):
                continue
            for cell in cells:
                grid[cell] = color
            return True
    return False


def _add_unhit_bar(grid, roles, rng, expected_events, orientation=None,
                   color=None, edge=False, long=False):
    height, width = grid.shape
    background = roles["background"]
    orientations = [orientation] if orientation else ["vertical", "horizontal"]
    for _ in range(180):
        direction = orientations[int(rng.integers(len(orientations)))]
        axis = height if direction == "vertical" else width
        other = width if direction == "vertical" else height
        low = min(10, axis) if long else 3
        high = min(axis, 18 if long else 12)
        if low > high:
            continue
        length = int(rng.integers(low, high + 1))
        if edge and int(rng.integers(2)) == 0:
            start = 0 if int(rng.integers(2)) == 0 else axis - length
        else:
            start = int(rng.integers(0, axis - length + 1))
        line = int(rng.integers(0, other))
        cells = ([(r, line) for r in range(start, start + length)]
                 if direction == "vertical" else
                 [(line, c) for c in range(start, start + length)])
        if any(grid[cell] != background for cell in cells):
            continue
        chosen_color = color if color is not None else (
            roles["transmit"] if int(rng.integers(2)) else roles["reflect"])
        candidate = grid.copy()
        if any(
            grid[nr, nc] in (roles["transmit"], roles["reflect"])
            for r, c in cells
            for nr in range(max(0, r - 1), min(height, r + 2))
            for nc in range(max(0, c - 1), min(width, c + 2))
        ):
            continue
        for cell in cells:
            candidate[cell] = chosen_color
        try:
            _all_bars_well_formed(candidate, roles)
            _, stats = _trace_ray(candidate, roles)
        except AssertionError:
            continue
        if stats["events"] == expected_events:
            grid[:, :] = candidate
            return True
    return False


def _transform_scene(grid, rng, allow_transpose=True):
    if allow_transpose and int(rng.integers(2)):
        grid = grid.T.copy()
    if int(rng.integers(2)):
        grid = np.flipud(grid).copy()
    if int(rng.integers(2)):
        grid = np.fliplr(grid).copy()
    return grid


def _planned_scene(rng, roles, seed_length, desired, size=None,
                   require_mixed=False):
    background, trail = roles["background"], roles["trail"]
    for _ in range(600):
        height = size[0] if size else int(rng.integers(16, 31))
        width = size[1] if size else int(rng.integers(16, 31))
        dc = 1 if int(rng.integers(2)) else -1
        col0 = int(rng.integers(3, width - 3))
        row, col = seed_length - 1, col0 + dc * (seed_length - 1)
        if not 1 <= col < width - 1:
            continue
        grid = np.full((height, width), background, dtype=np.int8)
        path = set()
        for r in range(seed_length):
            point = (r, col0 + dc * r)
            grid[point] = trail
            path.add(point)
        dr = 1
        planned = []
        feasible = True
        for kind, orientation in desired:
            gap = int(rng.integers(1, 5))
            for _step in range(gap):
                row += dr
                col += dc
                if not (1 <= row < height - 1 and 1 <= col < width - 1):
                    feasible = False
                    break
                if grid[row, col] != background or (row, col) in path:
                    feasible = False
                    break
                path.add((row, col))
            if not feasible:
                break
            target = (row + dr, col + dc)
            if not (1 <= target[0] < height - 1 and 1 <= target[1] < width - 1):
                feasible = False
                break
            if grid[target] != background or target in path:
                feasible = False
                break
            color = roles["transmit" if kind == "T" else "reflect"]
            if not _place_bar(grid, target, orientation, color, roles,
                              rng, path, long=int(rng.integers(4)) == 0):
                feasible = False
                break
            planned.append(("transmit" if kind == "T" else "reflect",
                            orientation))
            if kind == "T":
                row, col = target
                path.add(target)
            elif orientation == "vertical":
                dc = -dc
            else:
                dr = -dr
        if not feasible:
            continue
        try:
            _all_bars_well_formed(grid, roles)
            output, stats = _trace_ray(grid, roles)
        except AssertionError:
            continue
        if [(e[0], e[1]) for e in stats["events"]] != planned:
            continue
        if stats["background_steps"] < 2:
            continue
        if require_mixed and {e[1] for e in stats["events"]} != {"horizontal", "vertical"}:
            continue
        return grid
    raise ValueError("unable to construct planned scene")


def _corridor_scene(rng, roles, seed_length, size=None):
    background, trail, reflect = (roles[k] for k in
                                  ("background", "trail", "reflect"))
    for _ in range(200):
        height = size[0] if size else int(rng.integers(16, 31))
        width = size[1] if size else int(rng.integers(16, 31))
        first = int(rng.integers(seed_length + 2, min(height - 5, seed_length + 8)))
        right = int(rng.integers(max(first + 1, 6), width - 1))
        left = right - 3
        col0 = right - first
        if col0 < 0 or left < 1:
            continue
        bounces = int(rng.integers(4, min(9, height - first - 1)))
        grid = np.full((height, width), background, dtype=np.int8)
        for r in range(seed_length):
            grid[r, col0 + r] = trail
        for c in (left, right):
            grid[first - 2:first + bounces, c] = reflect
        try:
            output, stats = _trace_ray(grid, roles)
            _all_bars_well_formed(grid, roles)
        except AssertionError:
            continue
        if len(stats["events"]) >= 4 and stats["background_steps"] >= 2:
            if int(rng.integers(3)) != 0:
                _add_hit_transmission(grid, roles, rng, stats)
            return grid
    raise ValueError("unable to construct mirror corridor")


def _add_hit_transmission(grid, roles, rng, original_stats):
    path = list(original_stats["path"])
    rng.shuffle(path)
    seed_cells = set(
        (int(row), int(col))
        for row, col in np.argwhere(grid == roles["trail"])
    )
    forbidden = seed_cells | set(path)
    original_reflections = [event for event in original_stats["events"]
                            if event[0] == "reflect"]
    for target in path:
        orientations = ["vertical", "horizontal"]
        rng.shuffle(orientations)
        for orientation in orientations:
            candidate = grid.copy()
            if not _place_bar(candidate, target, orientation,
                              roles["transmit"], roles, rng, forbidden):
                continue
            try:
                _, stats = _trace_ray(candidate, roles)
                _all_bars_well_formed(candidate, roles)
            except AssertionError:
                continue
            reflections = [event for event in stats["events"]
                           if event[0] == "reflect"]
            transmissions = [event for event in stats["events"]
                             if event[0] == "transmit"]
            if (reflections == original_reflections
                    and len(transmissions) == 1
                    and stats["background_steps"] >= 2):
                grid[:, :] = candidate
                return True
    return False


def _decorate(grid, roles, rng, force_long=False):
    try:
        _, stats = _trace_ray(grid, roles)
    except AssertionError:
        return grid
    expected = stats["events"]
    if force_long:
        _add_unhit_bar(grid, roles, rng, expected, long=True,
                       edge=int(rng.integers(2)) == 0)
    for _ in range(int(rng.integers(0, 3))):
        _add_unhit_bar(grid, roles, rng, expected,
                       edge=int(rng.integers(3)) == 0,
                       long=int(rng.integers(5)) == 0)
    return grid


def _make_pair(rng, roles, regime, plan, size=None, allow_transpose=True):
    for _ in range(8):
        seed_length = int(rng.integers(3, 8))
        if plan == "corridor":
            grid = _corridor_scene(rng, roles, seed_length, size)
        else:
            desired = plan
            grid = _planned_scene(rng, roles, seed_length, desired,
                                  size, len({orientation for _, orientation in desired}) > 1)
        grid = _decorate(grid, roles, rng, force_long=int(rng.integers(6)) == 0)
        grid = _transform_scene(grid, rng, allow_transpose)
        try:
            output, stats = _trace_ray(grid, roles)
            if stats["background_steps"] >= 2 and len(stats["events"]) >= 1:
                return grid.astype(np.int8), output.astype(np.int8)
        except AssertionError:
            pass
    raise ValueError("unable to construct valid pair")


def _infer_role_candidates(pairs):
    if not pairs:
        return []
    colors = sorted({int(value) for pair in pairs
                     for value in np.unique(pair["input"])})
    if len(colors) != 4:
        return []
    candidates = []
    for background in colors:
        for trail in colors:
            if trail == background:
                continue
            for transmit in colors:
                if transmit in (background, trail):
                    continue
                reflect = next(color for color in colors
                               if color not in (background, trail, transmit))
                roles = dict(background=background, trail=trail,
                             transmit=transmit, reflect=reflect)
                valid = True
                for pair in pairs:
                    grid = pair["input"]
                    counts = [int(np.sum(grid == color)) for color in colors]
                    if counts[colors.index(background)] != max(counts) or counts.count(max(counts)) != 1:
                        valid = False
                        break
                    try:
                        _all_bars_well_formed(grid, roles)
                        expected, _ = _trace_ray(grid, roles)
                    except (AssertionError, ValueError):
                        valid = False
                        break
                    if not np.array_equal(expected, pair["output"]):
                        valid = False
                        break
                if valid:
                    candidates.append(roles)
    return candidates


def _assert_pair_rule(pair, roles):
    grid, output = pair["input"], pair["output"]
    assert grid.shape == output.shape
    assert set(int(v) for v in np.unique(grid)) <= set(roles.values())
    _all_bars_well_formed(grid, roles)
    expected, stats = _trace_ray(grid, roles)
    assert np.array_equal(output, expected)
    changed = grid != output
    assert np.any(changed)
    assert np.all(output[changed] == roles["trail"])
    assert stats["background_steps"] >= 2
    return stats


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for name in ("input", "output"):
                grid = pair[name]
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert pair["input"].shape == pair["output"].shape
            assert not _is_source_input(pair["input"])
            key = _grid_key(pair["input"])
            pairkey = (key, _grid_key(pair["output"]))
            assert key not in inputs and pairkey not in pairs
            inputs.add(key)
            pairs.add(pairkey)
    candidates = _infer_role_candidates(puzzle["train"])
    assert len(candidates) == 1, "training must identify all color roles"
    roles = candidates[0]
    train_stats = [_assert_pair_rule(pair, roles) for pair in puzzle["train"]]
    test_stats = [_assert_pair_rule(pair, roles) for pair in puzzle["test"]]
    train_effects = {(e[0], e[1]) for s in train_stats for e in s["events"]}
    test_effects = {(e[0], e[1]) for s in test_stats for e in s["events"]}
    assert {"transmit", "reflect"} <= {kind for kind, _ in train_effects}
    assert test_effects <= train_effects
    assert all(s["seed"]["length"] >= 3 for s in train_stats + test_stats)
    assert all(len(s["events"]) >= 1 for s in test_stats)


def _random_plan(rng, count, orientation=None, both=False):
    if both:
        kinds = ["T", "R"] + ["T" if int(rng.integers(2)) else "R"
                                 for _ in range(count - 2)]
    else:
        kinds = ["T" if int(rng.integers(2)) else "R" for _ in range(count)]
    rng.shuffle(kinds)
    if orientation is None:
        orientations = ["vertical", "horizontal"]
        choices = [orientations[int(rng.integers(2))] for _ in range(count)]
        if count >= 2 and len(set(choices)) == 1:
            choices[-1] = orientations[1 - orientations.index(choices[-1])]
    else:
        choices = [orientation] * count
    return list(zip(kinds, choices))


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed_value = _normalise_seed(seed)
    num_train, num_test = _resolve_counts(seed_value, num_train, num_test)
    rng = np.random.default_rng(seed_value)
    sampled = [int(v) for v in rng.choice(10, size=4, replace=False)]
    roles = dict(zip(("background", "trail", "transmit", "reflect"), sampled))
    puzzle = {"train": [], "test": []}
    seen = set()
    # Shuffle the coverage demonstrations: neither pair index nor seed length
    # determines the event kind. Both orientations of both materials are seen.
    coverage_orientations = ["horizontal", "vertical"]
    rng.shuffle(coverage_orientations)
    plans = []
    for orientation in coverage_orientations:
        plans.append((_random_plan(rng, 2, orientation, both=True), True))
    for _ in range(num_train - 2):
        event_count = int(rng.integers(1, 4))
        style = int(rng.integers(5))
        if style == 0 and event_count >= 2:
            plans.append((_random_plan(rng, 2, None), False))
        else:
            orientation = ("horizontal" if int(rng.integers(2))
                           else "vertical")
            plans.append((_random_plan(rng, event_count, orientation), False))
    rng.shuffle(plans)
    shared_size = None
    if int(rng.integers(3)) == 0:
        shared_size = (16, 16) if int(rng.integers(2)) else (
            int(rng.integers(17, 29)), int(rng.integers(17, 29)))
    for section, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            if section == "train":
                plan, locked_orientation = plans[index]
            else:
                locked_orientation = False
                style = int(rng.integers(5))
                if style == 0:
                    plan = "corridor"
                elif style == 1:
                    plan = _random_plan(rng, int(rng.integers(2, 4)), None, both=True)
                else:
                    plan = _random_plan(rng, int(rng.integers(1, 5)),
                                        "horizontal" if int(rng.integers(2)) else "vertical",
                                        both=style == 2)
            size = shared_size if shared_size and int(rng.integers(2)) else None
            if section == "test" and int(rng.integers(6)) == 0:
                size = (16, 16)
            for attempt in range(12):
                try:
                    input_grid, output_grid = _make_pair(
                        rng, roles, section, plan, size,
                        allow_transpose=not locked_orientation)
                except ValueError:
                    # A fresh composition is more useful than an exhausted
                    # geometric placement with the same constraints.
                    if section == "test" or not locked_orientation:
                        plan = _random_plan(
                            rng, 2,
                            "horizontal" if int(rng.integers(2)) else "vertical",
                            both=True)
                    continue
                key = _grid_key(input_grid)
                if key in seen or _is_source_input(input_grid):
                    continue
                seen.add(key)
                puzzle[section].append({"input": input_grid, "output": output_grid})
                break
            else:
                raise ValueError(f"unable to generate distinct {section} pair {index}: {plan}")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(
        np.array_equal(a[name], b[name])
        for section in ("train", "test")
        for a, b in zip(first[section], second[section])
        for name in ("input", "output")
    ) and all(len(first[s]) == len(second[s]) for s in ("train", "test"))


def self_test() -> None:
    train_counts, test_counts = set(), set()
    palettes, shapes, layouts, entry_sides = set(), set(), set(), set()
    orders, seed_lengths, material_counts = set(), set(), set()
    single_material = mixed_orientation = long_bar = corridor = 0
    compact_tests = side_tests = horizontal_tests = mixed_tests = 0
    simple_tests = long_mixed_tests = dense_tests = 0
    edge_unhit_bars = very_long_bars = single_material_tests = 0
    shared_canvas = overlap_lengths = overlap_events = 0
    within_length_variety = within_event_variety = 0
    total = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        roles = _infer_role_candidates(puzzle["train"])[0]
        palettes.add(tuple(roles.values()))
        by_section = {"train": [], "test": []}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                total += 1
                grid = pair["input"]
                stats = _assert_pair_rule(pair, roles)
                by_section[section].append((grid.shape, stats["seed"]["length"],
                                            len(stats["events"])))
                shapes.add(grid.shape)
                layouts.add(_canonical_layout_key(grid, roles))
                entry_sides.add(stats["seed"]["entry_side"])
                orders.add(tuple(e[0] for e in stats["events"]))
                seed_lengths.add(stats["seed"]["length"])
                material = {e[0] for e in stats["events"]}
                material_counts.add(len(material))
                single_material += len({roles["transmit"], roles["reflect"]} &
                                       set(int(v) for v in np.unique(grid))) == 1
                mixed_orientation += len({e[1] for e in stats["events"]}) == 2
                bars = [(color, component)
                        for color in (roles["transmit"], roles["reflect"])
                        for component in _components(grid, color)]
                hit_cells = {(e[2], e[3]) for e in stats["events"]}
                edge_unhit_bars += any(
                    not (set(component) & hit_cells)
                    and any(r in (0, grid.shape[0] - 1)
                            or c in (0, grid.shape[1] - 1)
                            for r, c in component)
                    for _, component in bars)
                long_bar += any(len(component) >= 12
                                for _, component in bars)
                very_long_bars += any(len(component) >= 14
                                      for _, component in bars)
                for component in _components(grid, roles["reflect"]):
                    hits = sum((e[2], e[3]) in component for e in stats["events"]
                               if e[0] == "reflect")
                    corridor += hits >= 2
                if section == "test":
                    compact_tests += grid.shape == (16, 16)
                    side_tests += stats["seed"]["entry_side"] in ("left", "right")
                    horizontal_tests += any(e[1] == "horizontal"
                                            for e in stats["events"])
                    mixed_tests += len({e[1] for e in stats["events"]}) == 2
                    simple_tests += len(stats["events"]) == 1
                    long_mixed_tests += (len(stats["events"]) >= 4
                                         and len({e[0] for e in stats["events"]}) == 2)
                    single_material_tests += len({roles["transmit"],
                                                  roles["reflect"]}
                                                 & set(int(v) for v in np.unique(grid))) == 1
                    reflections = [(e[2], e[3]) for e in stats["events"]
                                   if e[0] == "reflect"]
                    dense_tests += any(abs(a[0] - b[0]) == 1
                                       or abs(a[1] - b[1]) == 1
                                       for a, b in zip(reflections, reflections[1:]))
        train_features, test_features = by_section["train"], by_section["test"]
        shared_canvas += bool({x[0] for x in train_features}
                              & {x[0] for x in test_features})
        overlap_lengths += bool({x[1] for x in train_features}
                                & {x[1] for x in test_features})
        overlap_events += bool({x[2] for x in train_features}
                               & {x[2] for x in test_features})
        within_length_variety += len({x[1] for x in train_features + test_features}) > 1
        within_event_variety += len({x[2] for x in train_features + test_features}) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 30 and len(shapes) >= 20
    assert len(layouts) >= int(0.95 * total)
    assert entry_sides == {"top", "bottom", "left", "right"}
    assert len(orders) >= 5 and len(seed_lengths) >= 4
    assert material_counts == {1, 2}
    assert single_material >= 5 and mixed_orientation >= 2
    assert long_bar >= 5 and corridor >= 1
    assert compact_tests >= 2 and side_tests >= 10
    assert horizontal_tests >= 10 and mixed_tests >= 3
    assert simple_tests >= 3 and long_mixed_tests >= 3
    assert dense_tests >= 2 and single_material_tests >= 3
    assert edge_unhit_bars >= 5 and very_long_bars >= 5
    assert shared_canvas >= 2 and overlap_lengths >= 15
    assert overlap_events >= 15
    assert within_length_variety >= 35 and within_event_variety >= 35
    explicit = generate_puzzle(20260827, 7, 3)
    assert _puzzles_equal(explicit, generate_puzzle(20260827, 7, 3))
    assert not _puzzles_equal(generate_puzzle(101, 5, 2), generate_puzzle(102, 5, 2))
    for train, test in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(3000 + train * 10 + test, train, test)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train, test)
    for seed, train, test in ((4001, None, 1), (4002, None, 4),
                              (4003, 2, None), (4004, 10, None)):
        validate_generated_puzzle(generate_puzzle(seed, train, test))
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(9, **args)
        except ValueError:
            continue
        raise AssertionError("invalid count request accepted")


def _canonical_layout_key(grid, roles):
    canonical = np.empty(grid.shape, dtype=np.int8)
    for value, name in enumerate(("background", "trail", "transmit", "reflect")):
        canonical[grid == roles[name]] = value
    return _grid_key(canonical)
