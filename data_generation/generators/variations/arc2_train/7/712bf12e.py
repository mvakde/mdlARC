import numpy as np


_SOURCE_INPUT_ROWS = (
    (
        "00000050005000", "00000000500050", "00000000000500",
        "00550005500000", "00550000000000", "00000050005000",
        "05500000050000", "50500500000005", "50005050000000",
        "50000000005000", "00500050005500", "00000005055000",
        "50200205502000",
    ),
    (
        "0500000005000", "0500055000000", "5000000500050",
        "0000050000000", "0000500000005", "0000500000000",
        "5000000000055", "0050000000000", "0050050050000",
        "0500500500000", "0000005500500", "0020020002000",
    ),
    (
        "050000050000055", "050000000050050", "500000000000000",
        "000050000005000", "000000005050000", "055550000050050",
        "550000050550005", "050000000050000", "000550000500000",
        "050000050050500", "055000500050000", "005000000550500",
        "055200020200550",
    ),
    (
        "0005050000055", "0000050505550", "0000005000000",
        "5050005500005", "0055050000005", "0000000050500",
        "0055050500000", "5000055050000", "0500000005500",
        "0505000505050", "0500000005500", "5050505000000",
        "0002000202550",
    ),
)


def _resolve_counts(rng, num_train, num_test):
    def checked(value, low, high, name):
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

    num_train = checked(num_train, 2, 10, "num_train")
    num_test = checked(num_test, 1, 4, "num_test")

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        upper = min(4, num_train - 1)
        num_test = int(rng.integers(1, upper + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _apply_rule(input_grid, background, blocker, path):
    height, width = input_grid.shape
    source_columns = [
        int(column)
        for column in np.flatnonzero(
            input_grid[height - 1] == path
        )
    ]

    output = input_grid.copy()
    trajectories = []
    statuses = []

    for source_col in source_columns:
        row, col = height - 1, source_col
        trail = [(row, col)]

        for _ in range(height * width + 1):
            if row == 0:
                status = "top"
                break

            if input_grid[row - 1, col] == background:
                row -= 1
                trail.append((row, col))
                output[row, col] = path
                continue

            if (
                col + 1 < width
                and input_grid[row, col + 1] == background
            ):
                col += 1
                trail.append((row, col))
                output[row, col] = path
                continue

            status = (
                "right_boundary"
                if col == width - 1
                else "blocked"
            )
            break
        else:
            raise AssertionError("trajectory did not terminate")

        trajectories.append(trail)
        statuses.append(status)

    return output, statuses, trajectories


def _make_route(rng, height, width, start, limit, outcome, style="ordinary",
                terminal_max=None, pre_shift_cap=None):
    """Plant a route as local up/right decisions and the blockers that cause them."""
    if outcome == "top":
        terminal = 0
    elif style == "short":
        terminal = height - 2
    else:
        high = height - 2 if terminal_max is None else terminal_max
        terminal = int(rng.integers(1, high + 1))

    capacity = limit - start
    if outcome == "right_boundary":
        assert capacity >= 1
        if style == "complex":
            cap = min(3, capacity - 1, pre_shift_cap or capacity)
            pre_shift = min(cap, 2)
        else:
            cap = min(4, capacity - 1, pre_shift_cap or capacity)
            pre_shift = int(rng.integers(0, cap + 1))
        lengths = []
        if pre_shift:
            if pre_shift >= 2 and int(rng.integers(0, 2)):
                lengths = [1, pre_shift - 1]
            else:
                lengths = [pre_shift]
        lengths.append(capacity - pre_shift)
        available = list(range(terminal + 1, height))
        if len(lengths) - 1 > len(available):
            lengths = [pre_shift, capacity - pre_shift]
        chosen = sorted((int(x) for x in rng.choice(
            available, size=len(lengths) - 1, replace=False)), reverse=True)
        event_rows = chosen + [terminal]
    else:
        cap = min(6, capacity)
        if style == "complex":
            assert cap >= 3
            shift = int(rng.integers(3, cap + 1))
            lengths = [1, 1, shift - 2]
        elif style == "straight" or style == "short":
            lengths = []
        else:
            shift = int(rng.integers(0, cap + 1))
            if shift == 0:
                lengths = []
            else:
                groups = int(rng.integers(1, min(3, shift) + 1))
                lengths = [1] * groups
                for _ in range(shift - groups):
                    lengths[int(rng.integers(0, groups))] += 1
        available = list(range(max(1, terminal + 1), height))
        while len(lengths) > len(available):
            last = lengths.pop()
            lengths[-1] += last
        if style == "complex":
            chosen = [height - 1]
            chosen.extend(int(x) for x in rng.choice(
                available[:-1], size=len(lengths) - 1, replace=False))
            event_rows = sorted(chosen, reverse=True)
        elif lengths:
            event_rows = sorted((int(x) for x in rng.choice(
                available, size=len(lengths), replace=False)), reverse=True)
        else:
            event_rows = []

    row, col = height - 1, start
    route = [(row, col)]
    blockers = set()
    for turn_row, length in zip(event_rows, lengths):
        while row > turn_row:
            row -= 1
            route.append((row, col))
        for _ in range(length):
            blockers.add((row - 1, col))
            col += 1
            route.append((row, col))
    while row > terminal:
        row -= 1
        route.append((row, col))
    if outcome == "blocked":
        blockers.add((row - 1, col))
        blockers.add((row, col + 1))
    assert col <= limit and len(route) > 1
    assert not set(route) & blockers
    return {"route": route, "blockers": blockers}


def _source_columns(rng, width, foundation):
    if foundation:
        first = int(rng.integers(0, width - 7))
        second = int(rng.integers(first + 4, width - 3))
        third = int(rng.integers(second + 2, width - 1))
        return [first, second, third]
    valid = [
        (a, b, c)
        for a in range(width - 4)
        for b in range(a + 2, width - 2)
        for c in range(b + 2, width - 1)
    ]
    return list(valid[int(rng.integers(len(valid)))])


def _place_obstacles(rng, height, width, routes, required, density):
    forbidden = set().union(*(set(p["route"]) for p in routes))
    blockers = set(required)
    target = max(len(blockers), int(round(density * height * width)))
    # Rectangles, ledges, and staggered local clusters precede isolated cells.
    attempts = 0
    while len(blockers) < target and attempts < height * width:
        attempts += 1
        if int(rng.integers(0, 4)) == 0:
            break
        row = int(rng.integers(0, height))
        col = int(rng.integers(0, width))
        shape = int(rng.integers(0, 3))
        if shape == 0:
            cells = [(row + dr, col + dc)
                     for dr in range(int(rng.integers(1, 4)))
                     for dc in range(int(rng.integers(2, 5)))]
        elif shape == 1:
            cells = [(row, col + dc) for dc in range(int(rng.integers(2, 7)))]
        else:
            cells = [(row + dc // 2, col + dc)
                     for dc in range(int(rng.integers(2, 6)))]
        for cell in cells:
            if len(blockers) >= target:
                break
            if (0 <= cell[0] < height and 0 <= cell[1] < width
                    and cell not in forbidden):
                blockers.add(cell)
    candidates = [(r, c) for r in range(height) for c in range(width)
                  if (r, c) not in forbidden and (r, c) not in blockers]
    need = target - len(blockers)
    if need:
        chosen = rng.choice(len(candidates), size=need, replace=False)
        blockers.update(candidates[int(i)] for i in chosen)
    return blockers


def _fallback_pair(rng, mode, boundary_index, avoid_palette):
    height, width = 12, 18
    columns = (1, 7, 13)
    blockers = set()
    if mode == "foundation":
        blockers.update(((10, 1), (5, 2), (5, 3), (5, 13), (6, 14)))
    elif mode == "boundary":
        start = columns[boundary_index]
        blockers.update((4, c) for c in range(start, width))
        if boundary_index < 2:
            blockers.update(((9, 13), (10, 14)))
        if boundary_index == 0:
            blockers.update(((9, 7), (10, 8)))
        if boundary_index == 2:
            blockers.update(((5, 7), (6, 8)))
    elif mode == "mixed":
        blockers.update(((5, 7), (6, 8)))

    template = np.zeros((height, width), dtype=np.int8)
    for row, col in blockers:
        template[row, col] = 1
    for col in columns:
        template[-1, col] = 2
    _, _, trails = _apply_rule(template, 0, 1, 2)
    plans = [{"route": trail} for trail in trails]
    blockers = _place_obstacles(rng, height, width, plans, blockers, 0.20)
    colors = rng.choice(10, size=3, replace=False)
    background, blocker, path = (int(v) for v in colors)
    if (background, blocker, path) == avoid_palette:
        background, blocker = blocker, background
    input_grid = np.full((height, width), background, dtype=np.int8)
    for row, col in blockers:
        input_grid[row, col] = blocker
    for col in columns:
        input_grid[-1, col] = path
    output_grid, _, _ = _apply_rule(input_grid, background, blocker, path)
    return {"input": input_grid, "output": output_grid}


def _generate_pair(rng, regime, index, avoid_palette=None):
    foundation = regime == "source_train" and index == 0
    if foundation:
        mode = "foundation"
    elif regime == "source_train" and index == 1:
        mode = "boundary"
    else:
        mode = str(rng.choice(
            ["all_top", "mixed", "boundary"],
            p=[0.24, 0.38, 0.38] if regime == "source_train"
            else [0.08, 0.32, 0.60]))
    requested_boundary = int(rng.integers(0, 3)) if mode == "boundary" else None
    for attempt in range(300):
        height = int(rng.integers(8, 23))
        if int(rng.integers(0, 3)) == 0:
            width = max(10, min(24, height + int(rng.integers(-5, 6))))
        else:
            width = int(rng.integers(10, 25))
        columns = _source_columns(rng, width, foundation)
        if foundation:
            outcomes = ["top", "top", "blocked"]
            styles = ["complex", "straight", "ordinary"]
            boundary_index = None
        else:
            if mode == "all_top":
                outcomes = ["top"] * 3
            elif mode == "mixed":
                outcomes = ["top", "blocked", str(rng.choice(["top", "blocked"]))]
                rng.shuffle(outcomes)
            else:
                boundary_index = requested_boundary
                outcomes = ["top", "blocked", "top"]
                outcomes[boundary_index] = "right_boundary"
                if boundary_index == 0:
                    outcomes[1:] = ["blocked", "blocked"]
                elif boundary_index == 1:
                    outcomes[2] = "blocked"
            boundary_index = outcomes.index("right_boundary") if "right_boundary" in outcomes else None
            styles = ["ordinary"] * 3
            if boundary_index is not None and boundary_index < 2:
                for j in range(boundary_index + 1, 3):
                    styles[j] = "short"
            if regime == "target_test" and boundary_index is not None:
                styles[boundary_index] = "complex"

        plans = [None] * 3
        possible = True
        for j in (2, 1, 0):
            outcome = outcomes[j]
            if outcome == "right_boundary":
                limit = width - 1
                terminal_max = height - 5 if j < 2 else None
                pre_cap = columns[j + 1] - columns[j] - 1 if j < 2 else None
            else:
                limit = width - 2 if j == 2 else columns[j + 1] - 1
                if outcome == "blocked" and j < 2:
                    limit -= 1
                terminal_max = None
                pre_cap = None
            if limit < columns[j] or (styles[j] == "complex" and
                    outcome != "right_boundary" and limit - columns[j] < 3):
                possible = False
                break
            plans[j] = _make_route(rng, height, width, columns[j], limit,
                                   outcome, styles[j], terminal_max, pre_cap)
        if not possible:
            continue
        routes = [set(p["route"]) for p in plans]
        blocked = [p["blockers"] for p in plans]
        if any(routes[i] & routes[j] or routes[i] & blocked[j] or
               routes[j] & blocked[i]
               for i in range(3) for j in range(i + 1, 3)):
            continue
        density = float(rng.uniform(0.10, 0.30 if regime == "source_train" else 0.34))
        required = set().union(*blocked)
        blockers = _place_obstacles(rng, height, width, plans, required, density)
        colors = rng.choice(10, size=3, replace=False)
        background, blocker, path = (int(v) for v in colors)
        if (background, blocker, path) == avoid_palette:
            background, blocker = blocker, background
        input_grid = np.full((height, width), background, dtype=np.int8)
        for row, col in blockers:
            input_grid[row, col] = blocker
        for col in columns:
            input_grid[height - 1, col] = path
        output_grid, actual, actual_routes = _apply_rule(
            input_grid, background, blocker, path)
        if (actual == outcomes and all(actual_routes[j] == plans[j]["route"]
                                       for j in range(3))
                and np.count_nonzero(input_grid == background) > len(blockers)
                and not np.array_equal(input_grid, output_grid)):
            return {"input": input_grid, "output": output_grid}
    return _fallback_pair(rng, mode, requested_boundary, avoid_palette)


def _infer_roles(input_grid):
    height = input_grid.shape[0]
    values, counts = np.unique(input_grid, return_counts=True)
    assert len(values) == 3
    tally = {int(v): int(n) for v, n in zip(values, counts)}
    sources = [int(v) for v in values if tally[int(v)] == 3 and
               np.all(np.argwhere(input_grid == v)[:, 0] == height - 1)]
    assert len(sources) == 1
    path = sources[0]
    other = [int(v) for v in values if v != path]
    background = max(other, key=lambda v: tally[v])
    blocker = min(other, key=lambda v: tally[v])
    assert tally[background] > tally[blocker] > 3
    return background, blocker, path


def _array_key(grid):
    return grid.shape, grid.tobytes()


def _features(input_grid, trajectories, statuses, blocker):
    per_path = []
    background = _infer_roles(input_grid)[0]
    for trail, status in zip(trajectories, statuses):
        rights = [i for i in range(len(trail) - 1)
                  if trail[i + 1][1] == trail[i][1] + 1]
        runs = sum(i == 0 or i - 1 not in rights for i in rights)
        per_path.append({
            "status": status,
            "first_right": bool(rights and rights[0] == 0),
            "right_steps": len(rights),
            "right_runs": runs,
            "terminal_row": trail[-1][0],
            "start_col": trail[0][1],
            "up_priority": any(
                trail[i + 1][0] == trail[i][0] - 1 and
                trail[i][1] + 1 < input_grid.shape[1] and
                input_grid[trail[i][0], trail[i][1] + 1] == background
                for i in range(len(trail) - 1)),
        })
    return per_path


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    train_features = []
    palettes = set()
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = pair["input"], pair["output"]
            for grid in (a, b):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((0 <= grid) & (grid <= 9))
            assert a.shape == b.shape
            key = _array_key(a)
            assert key not in seen
            seen.add(key)
            rows = tuple("".join(str(int(v)) for v in row) for row in a)
            assert rows not in _SOURCE_INPUT_ROWS
            background, blocker, path = _infer_roles(a)
            palettes.add((background, blocker, path))
            assert np.count_nonzero(a[-1] == path) == 3
            assert np.array_equal(a == blocker, b == blocker)
            changed = a != b
            assert np.any(changed)
            assert np.all(a[changed] == background)
            assert np.all(b[changed] == path)
            expected, statuses, trails = _apply_rule(a, background, blocker, path)
            assert np.array_equal(expected, b)
            assert all(len(t) > 1 for t in trails)
            assert all(set(trails[i]).isdisjoint(trails[j])
                       for i in range(3) for j in range(i + 1, 3))
            features = _features(a, trails, statuses, blocker)
            if section == "train":
                train_features.extend(features)
    assert len(palettes) >= 2
    assert any(f["first_right"] for f in train_features)
    assert any(f["right_runs"] >= 2 for f in train_features)
    assert any(f["up_priority"] for f in train_features)
    assert {"top", "blocked", "right_boundary"} <= {
        f["status"] for f in train_features}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    child_seeds = rng.integers(0, np.iinfo(np.int64).max,
                               size=num_train + num_test, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    for i in range(num_train):
        avoid = _infer_roles(puzzle["train"][0]["input"]) if i == 1 else None
        puzzle["train"].append(_generate_pair(
            np.random.default_rng(int(child_seeds[i])), "source_train", i, avoid))
    for i in range(num_test):
        puzzle["test"].append(_generate_pair(
            np.random.default_rng(int(child_seeds[num_train + i])),
            "target_test", i))
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple((section, _array_key(pair["input"]), _array_key(pair["output"]))
                 for section in ("train", "test") for pair in puzzle[section])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    shapes, starts, palettes, densities, status_patterns = set(), set(), set(), set(), set()
    train_outcomes = set()
    close_sources = False
    same_third = False
    immediate = multi_run = long_shift = boundary_nonlast = all_top = False
    low_trap = high_trap = straight = tall = small = cluster = False
    train_boundary = test_boundary = False
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for section in ("train", "test"):
            for pair in puzzle[section]:
                a, b = pair["input"], pair["output"]
                bg, block, path = _infer_roles(a)
                expected = a.copy()
                trails = []
                for start in np.flatnonzero(a[-1] == path):
                    r, c = a.shape[0] - 1, int(start)
                    trail = [(r, c)]
                    while r:
                        if a[r - 1, c] == bg:
                            r -= 1
                        elif c + 1 < a.shape[1] and a[r, c + 1] == bg:
                            c += 1
                        else:
                            break
                        expected[r, c] = path
                        trail.append((r, c))
                    trails.append(trail)
                assert np.array_equal(expected, b)
                _, statuses, _ = _apply_rule(a, bg, block, path)
                fs = _features(a, trails, statuses, block)
                shapes.add(a.shape)
                starts.add(tuple(int(x) for x in np.flatnonzero(a[-1] == path)))
                palettes.add((bg, block, path))
                densities.add(round(float(np.mean(a == block)), 2))
                status_patterns.add(tuple(statuses))
                train_outcomes.update(statuses if section == "train" else ())
                cols = sorted(f["start_col"] for f in fs)
                close_sources |= min(cols[1] - cols[0], cols[2] - cols[1]) <= 2
                same_third |= any(sum(lo <= c < hi for c in cols) >= 2
                                  for lo, hi in ((0, a.shape[1] // 3),
                                                 (a.shape[1] // 3, 2 * a.shape[1] // 3),
                                                 (2 * a.shape[1] // 3, a.shape[1])))
                immediate |= any(f["first_right"] for f in fs)
                multi_run |= any(f["right_runs"] >= 2 for f in fs)
                long_shift |= any(f["right_steps"] >= 3 for f in fs)
                low_trap |= any(f["status"] == "blocked" and
                                f["terminal_row"] >= a.shape[0] - 4 for f in fs)
                high_trap |= any(f["status"] == "blocked" and
                                 f["terminal_row"] <= 2 for f in fs)
                straight |= any(f["right_steps"] == 0 for f in fs)
                tall |= a.shape[0] > a.shape[1]
                small |= max(a.shape) <= 12
                cluster |= any(a[r, c] == block and a[r + 1, c] == block and
                               a[r, c + 1] == block and a[r + 1, c + 1] == block
                               for r in range(a.shape[0] - 1)
                               for c in range(a.shape[1] - 1))
                if section == "train":
                    all_top |= all(s == "top" for s in statuses)
                    train_boundary |= "right_boundary" in statuses
                else:
                    test_boundary |= "right_boundary" in statuses
                boundary_nonlast |= any(f["status"] == "right_boundary" and
                                        i < 2 for i, f in enumerate(fs))
    assert len(shapes) >= 35 and len(starts) >= 35
    assert len(palettes) >= 70 and len(densities) >= 15
    assert len(status_patterns) >= 10
    assert {"top", "blocked", "right_boundary"} <= train_outcomes
    assert all((close_sources, same_third, immediate, multi_run, long_shift,
                boundary_nonlast, all_top, low_trap, high_trap, straight,
                tall, small, cluster, train_boundary, test_boundary))
    for i, (nt, ns) in enumerate(((2, 1), (10, 1), (10, 4), (6, 2),
                                   (None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(500 + i, nt, ns)
        assert nt is None or len(p["train"]) == nt
        assert ns is None or len(p["test"]) == ns
    assert _puzzle_key(generate_puzzle(777, 7, 3)) == _puzzle_key(generate_puzzle(777, 7, 3))
    assert _puzzle_key(generate_puzzle(777, 7, 3)) != _puzzle_key(generate_puzzle(778, 7, 3))
    for nt, ns in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 3), (3, 4)):
        try:
            generate_puzzle(900, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
