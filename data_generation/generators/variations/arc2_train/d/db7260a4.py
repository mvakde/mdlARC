import numpy as np


_SOURCE_INPUT_PATTERNS = {
    (
        "0000100000",
        "0000000000",
        "0000000000",
        "0020020200",
        "0020020200",
        "0020020200",
        "0022220200",
        "0000000000",
        "0000000000",
        "0000000000",
    ),
    (
        "0000000010",
        "0000000000",
        "0000000000",
        "0020000200",
        "0020000200",
        "0020000200",
        "0022222200",
        "0000000000",
        "0000000000",
        "0000000000",
    ),
    (
        "0000100000",
        "0000000000",
        "0000000000",
        "0020000200",
        "0020000200",
        "0020000200",
        "0022220200",
        "0000000000",
        "0000000000",
        "0000000000",
    ),
    (
        "0000001000",
        "0000000000",
        "0000000000",
        "0020020200",
        "0020020200",
        "0020020200",
        "0020222200",
        "0000000000",
        "0000000000",
        "0000000000",
    ),
    (
        "0000001000",
        "0000000000",
        "0000000000",
        "0020000200",
        "0020000200",
        "0020000200",
        "0022222200",
        "0000000000",
        "0000000000",
        "0000000000",
    ),
    (
        "0000000010",
        "0000000000",
        "0000020002",
        "0000020002",
        "0000020222",
        "0000000000",
        "0020000020",
        "0020000020",
        "0022222220",
        "0000000000",
    ),
}


def _checked_optional_count(
    name: str,
    value: int | None,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _choose_counts(
    seed_key: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _checked_optional_count("num_train", num_train, 2, 10)
    num_test = _checked_optional_count("num_test", num_test, 1, 4)

    selector = (
        seed_key
        ^ (seed_key >> 29)
        ^ 0x9E3779B97F4A7C15
    ) & ((1 << 64) - 1)

    if num_train is None and num_test is None:
        num_test = 1 + seed_key % 4
        low = num_test + 1
        num_train = low + (seed_key // 4) % (11 - low)
    elif num_train is None:
        low = num_test + 1
        num_train = low + selector % (11 - low)
    elif num_test is None:
        largest_test = min(4, num_train - 1)
        num_test = 1 + selector % largest_test

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return int(num_train), int(num_test)


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _rows_of(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _extract_vessels(
    grid: np.ndarray,
    background: int,
    wall: int,
) -> list[tuple[int, int, int, int, int]]:
    height, width = grid.shape
    sides: list[tuple[int, int, int]] = []

    for col in range(width):
        row = 0
        while row < height:
            if int(grid[row, col]) != wall:
                row += 1
                continue

            top = row
            while (
                row + 1 < height
                and int(grid[row + 1, col]) == wall
            ):
                row += 1
            floor = row

            if floor - top >= 1:
                sides.append((top, floor, col))
            row += 1

    vessels: set[tuple[int, int, int, int, int]] = set()

    for index, (top, floor, left) in enumerate(sides):
        for top2, floor2, right in sides[index + 1 :]:
            if (
                top2 != top
                or floor2 != floor
                or right <= left
            ):
                continue

            if (
                right - left - 1 < 1
                or top < 2
                or floor > height - 2
            ):
                continue

            if not np.all(
                grid[top:floor, left + 1 : right] == background
            ):
                continue

            if not np.all(
                grid[top - 1, left + 1 : right] == background
            ):
                continue

            floor_values = grid[floor, left + 1 : right]
            if not np.all(
                (floor_values == wall)
                | (floor_values == background)
            ):
                continue

            gap_offsets = np.flatnonzero(
                floor_values == background
            )
            if len(gap_offsets) > 1:
                continue

            gap = (
                -1
                if len(gap_offsets) == 0
                else left + 1 + int(gap_offsets[0])
            )
            vessels.add((top, floor, left, right, gap))

    vessel_list = sorted(vessels)
    assert vessel_list, "input must contain at least one vessel"

    covered = np.zeros(grid.shape, dtype=bool)
    for top, floor, left, right, gap in vessel_list:
        covered[top : floor + 1, left] = True
        covered[top : floor + 1, right] = True

        for col in range(left, right + 1):
            if col != gap:
                covered[floor, col] = True

    assert np.array_equal(
        covered,
        grid == wall,
    ), "all structural cells must form vessels"

    return vessel_list


def _solve_details(
    grid: np.ndarray,
    roles: tuple[int, int, int],
) -> tuple[
    np.ndarray,
    list[tuple[int, int, int, int, int]],
    list[tuple[tuple[int, int, int, int, int], int]],
    str,
]:
    background, wall, source = roles

    marker_positions = np.argwhere(grid == source)
    assert marker_positions.shape == (
        1,
        2,
    ), "there must be exactly one source marker"

    source_row, source_col = (
        int(value) for value in marker_positions[0]
    )

    vessels = _extract_vessels(
        grid,
        background,
        wall,
    )

    output = grid.copy()
    output[source_row, source_col] = background

    trace: list[
        tuple[tuple[int, int, int, int, int], int]
    ] = []

    current_row = source_row + 1
    current_col = source_col
    levels = sorted({vessel[0] for vessel in vessels})

    while True:
        selected = None

        for top in levels:
            if top < current_row:
                continue

            at_level = [
                vessel
                for vessel in vessels
                if vessel[0] == top
            ]
            touching = [
                vessel
                for vessel in at_level
                if vessel[2] <= current_col <= vessel[3]
            ]

            if not touching:
                continue

            entering = [
                vessel
                for vessel in touching
                if vessel[2] < current_col < vessel[3]
            ]

            assert (
                len(touching) == 1
                and len(entering) == 1
            ), "trajectory may not hit a wall"

            selected = entering[0]
            break

        if selected is None:
            output[-1, :] = source
            return output, vessels, trace, "spill"

        trace.append((selected, current_col))
        top, floor, left, right, gap = selected

        if gap == -1:
            output[
                top:floor,
                left + 1 : right,
            ] = source
            return output, vessels, trace, "fill"

        current_row = floor + 1
        current_col = gap


def _draw_input(
    height: int,
    width: int,
    specs: list[tuple[int, int, int, int, int]],
    source_col: int,
    roles: tuple[int, int, int],
) -> np.ndarray:
    background, wall, source = roles
    grid = np.full(
        (height, width),
        background,
        dtype=np.int8,
    )

    for top, floor, left, right, gap in specs:
        grid[top : floor + 1, left] = wall
        grid[top : floor + 1, right] = wall
        grid[floor, left : right + 1] = wall

        if gap != -1:
            grid[floor, gap] = background

    grid[0, source_col] = source
    return grid


def _interval_containing(rng, width, column, interior_width):
    low = max(1, column - interior_width)
    high = min(column - 1, width - interior_width - 3)
    assert low <= high
    left = int(rng.integers(low, high + 1))
    return left, left + interior_width + 1


def _all_intervals(width, minimum=1, maximum=8):
    return [
        (left, left + interior_width + 1)
        for interior_width in range(minimum, min(maximum, width - 4) + 1)
        for left in range(1, width - interior_width - 2)
    ]


def _compatible_interval(candidate, placed):
    left, right = candidate
    return all(
        right == other_left or right + 3 <= other_left
        or left == other_right or left >= other_right + 3
        for other_left, other_right in placed
    )


def _add_vessel(rng, specs, top, floor, interval, gap):
    left, right = interval
    specs.append((top, floor, left, right, gap))


def _add_distractors(rng, specs, top, floor, width, protected_col, goal):
    placed = [(spec[2], spec[3]) for spec in specs if spec[0] == top]
    for _ in range(goal):
        candidates = [
            interval for interval in _all_intervals(width, 1, 6)
            if _compatible_interval(interval, placed)
            and (protected_col is None or not interval[0] <= protected_col <= interval[1])
        ]
        if not candidates:
            break
        shared = [
            interval for interval in candidates
            if any(interval[1] == left or interval[0] == right for left, right in placed)
        ]
        if shared and rng.random() < 0.55:
            candidates = shared
        interval = candidates[int(rng.integers(len(candidates)))]
        left, right = interval
        gap = -1 if rng.random() < 0.52 else int(rng.integers(left + 1, right))
        _add_vessel(rng, specs, top, floor, interval, gap)
        placed.append(interval)


def _geometry(rng, level_count):
    for _ in range(60):
        top = int(rng.integers(2, 8))
        levels = []
        for index in range(level_count):
            interior_height = int(rng.integers(2, 7))
            if index == level_count - 1 and index > 0 and rng.random() < 0.28:
                interior_height = levels[0][1] - levels[0][0]
            floor = top + interior_height
            levels.append((top, floor))
            top = floor + int(rng.integers(2, 6))
        height = levels[-1][1] + int(rng.integers(2, 8))
        if height <= 30:
            return levels, height
    levels = []
    top = 2
    for _ in range(level_count):
        floor = top + 2
        levels.append((top, floor))
        top = floor + 2
    return levels, levels[-1][1] + 2


def _make_pair(rng, roles, actions, force_offaxis=False, distractor_policy=None):
    level_count = len(actions)
    assert 1 <= level_count <= 3
    if level_count == 1 and rng.random() < 0.20:
        width, height = 10, 10
        top = int(rng.integers(2, 5))
        levels = [(top, top + int(rng.integers(2, min(5, 9 - top))))]
    else:
        width = int(rng.integers(8, 14)) if rng.random() < 0.38 else int(rng.integers(14, 31))
        levels, height = _geometry(rng, level_count)
    source_col = int(rng.integers(2, width - 2))
    current_col = source_col
    specs = []
    selected = []
    stopped = False

    for index, (top, floor) in enumerate(levels):
        action = actions[index]
        if stopped:
            action = "after"
        if action in ("fill", "leak", "after"):
            minimum_width = 2 if action == "leak" and force_offaxis and index == 0 else 1
            if minimum_width == 1 and rng.random() < 0.27:
                interior_width = 1
            else:
                interior_width = int(rng.integers(max(2, minimum_width), min(8, width - 4) + 1))
            if action == "fill" and index > 0 and selected and rng.random() < 0.30:
                interior_width = selected[0][3] - selected[0][2] - 1
            interval = _interval_containing(rng, width, current_col, interior_width)
            left, right = interval
            if action == "leak":
                if force_offaxis and index == 0:
                    alternatives = [col for col in range(left + 1, right) if col != current_col]
                    gap = alternatives[int(rng.integers(len(alternatives)))]
                elif rng.random() < 0.28:
                    gap = current_col
                else:
                    gap = int(rng.integers(left + 1, right))
            elif action == "after" and rng.random() < 0.5:
                gap = int(rng.integers(left + 1, right))
            else:
                gap = -1
            _add_vessel(rng, specs, top, floor, interval, gap)
            selected.append((top, floor, left, right, gap))
            if action == "leak":
                current_col = gap
            elif action == "fill":
                stopped = True
        elif action == "miss":
            candidates = [
                interval for interval in _all_intervals(width, 1, 6)
                if not interval[0] <= current_col <= interval[1]
            ]
            interval = candidates[int(rng.integers(len(candidates)))]
            left, right = interval
            gap = -1 if rng.random() < 0.55 else int(rng.integers(left + 1, right))
            _add_vessel(rng, specs, top, floor, interval, gap)
        else:
            raise AssertionError("unknown route action")

        if distractor_policy == "none":
            goal = 0
        elif distractor_policy == "some" and index == 0:
            goal = int(rng.integers(1, 4))
        else:
            goal = int(rng.choice(4, p=[0.37, 0.34, 0.22, 0.07]))
        _add_distractors(rng, specs, top, floor, width, current_col if action == "miss" else None, goal)

    input_grid = _draw_input(height, width, specs, source_col, roles)
    output_grid, vessels, trace, outcome = _solve_details(input_grid, roles)
    expected = "fill" if "fill" in actions else "spill"
    assert outcome == expected
    assert len(vessels) >= level_count
    if distractor_policy == "some":
        assert sum(vessel[0] == levels[0][0] for vessel in vessels) >= 2
    return input_grid, output_grid


def _infer_roles_from_input(grid):
    background = int(grid[0, 0])
    colors = {int(value) for value in np.unique(grid)}
    assert len(colors) == 3
    markers = [int(value) for value in np.unique(grid[0]) if int(value) != background]
    assert len(markers) == 1
    source = markers[0]
    assert np.count_nonzero(grid == source) == 1
    wall = next(iter(colors - {background, source}))
    return background, wall, source


def _pair_roles(rng, base_roles, stable, first=False):
    if stable or (first and rng.random() < 0.35):
        return base_roles
    values = rng.choice(10, size=3, replace=False)
    return tuple(int(value) for value in values)


def _train_plan(index, rng):
    if index == 0:
        return ["fill"], False, "none" if rng.random() < 0.48 else None
    if index == 1:
        return ["leak"], True, "none" if rng.random() < 0.48 else None
    if index == 2:
        return ["miss"], False, None
    if index == 3:
        return ["fill"], False, "some"
    if index == 4:
        return ["leak", "fill"], False, None
    if index == 5:
        return ["fill", "after"], False, None
    plans = (["fill"], ["leak"], ["miss"], ["leak", "fill"],
             ["miss", "fill"], ["leak", "miss"], ["fill", "after"])
    plan = list(plans[int(rng.integers(len(plans)))])
    return plan, False, None


def _test_plan(index, train_count, rng):
    two_level = (["leak", "fill"], ["leak", "miss"])
    if train_count >= 3:
        two_level += (["miss", "fill"], ["miss", "leak"])
    if train_count >= 6:
        two_level += (["fill", "after"],)
    three_level = (["leak", "leak", "fill"], ["leak", "leak", "miss"],
                   ["miss", "leak", "fill"])
    if train_count >= 6:
        three_level += (["leak", "fill", "after"],)
    one_level = (["fill"], ["leak"], ["miss"])
    if index == 0:
        pool = two_level + (three_level if train_count >= 5 else ())
    else:
        pool = one_level + two_level + (three_level if train_count >= 5 else ())
    return list(pool[int(rng.integers(len(pool)))])


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed_key = int(seed) % (1 << 64)
    train_count, test_count = _choose_counts(seed_key, num_train, num_test)
    rng = np.random.default_rng(seed_key)
    base_roles = tuple(int(value) for value in rng.choice(10, size=3, replace=False))
    stable_palette = bool(rng.integers(0, 2))
    train_plans = [_train_plan(index, rng) for index in range(train_count)]
    rng.shuffle(train_plans)
    puzzle = {"train": [], "test": []}
    seen = set()

    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            for _ in range(120):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                roles = _pair_roles(pair_rng, base_roles, stable_palette, first=split == "train" and index == 0)
                if split == "train":
                    actions, offaxis, policy = train_plans[index]
                else:
                    actions, offaxis, policy = _test_plan(index, train_count, pair_rng), False, None
                try:
                    input_grid, output_grid = _make_pair(pair_rng, roles, actions, offaxis, policy)
                except AssertionError:
                    continue
                key = _grid_key(input_grid)
                if key not in seen and _rows_of(input_grid) not in _SOURCE_INPUT_PATTERNS:
                    break
            else:
                raise AssertionError("could not generate a unique pair")
            seen.add(key)
            puzzle[split].append({"input": input_grid, "output": output_grid})

    validate_generated_puzzle(puzzle)
    return puzzle


def _analysis(pair):
    grid = pair["input"]
    roles = _infer_roles_from_input(grid)
    expected, vessels, trace, outcome = _solve_details(grid, roles)
    assert np.array_equal(expected, pair["output"]), "output violates the vessel rule"
    levels = {vessel[0] for vessel in vessels}
    return roles, vessels, trace, outcome, levels


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    seen_pairs = set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((grid >= 0) & (grid <= 9)))
            inp, out = pair["input"], pair["output"]
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            input_key = _grid_key(inp)
            pair_key = input_key, _grid_key(out)
            assert input_key not in seen_inputs and pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            assert _rows_of(inp) not in _SOURCE_INPUT_PATTERNS
            roles = _infer_roles_from_input(inp)
            background, wall, source = roles
            assert int(inp[0, 0]) == background
            marker_cols = np.flatnonzero(inp[0] == source)
            assert len(marker_cols) == 1
            assert 1 <= int(marker_cols[0]) < inp.shape[1] - 1
            assert int(inp[0, -1]) == background
            assert np.all(inp[-1] == background)
            assert np.all(inp[1] == background)
            assert set(int(value) for value in np.unique(out)) == set(roles)
            analysis = _analysis(pair)
            analyses[split].append(analysis)
            if split == "train":
                assert len(analysis[4]) <= 2
            else:
                assert len(analysis[4]) <= 3
                if index == 0:
                    assert len(analysis[4]) >= 2

    training = analyses["train"]
    assert any(item[3] == "fill" and len(item[2]) == 1 and len(item[4]) == 1
               for item in training)
    assert any(item[3] == "spill" and len(item[2]) == 1
               and item[2][0][0][4] != item[2][0][1] for item in training)
    if len(training) >= 3:
        assert any(item[3] == "spill" and not item[2] for item in training)
    if len(training) >= 4:
        assert any(item[3] == "fill" and len(item[4]) == 1 and len(item[1]) >= 2
                   for item in training)
    if len(training) >= 5:
        assert any(item[3] == "fill" and len(item[4]) == 2 and len(item[2]) == 2
                   for item in training)
    if len(training) >= 6:
        assert any(item[3] == "fill" and item[2][-1][0][0] < max(item[4])
                   for item in training)
    if any(len(item[4]) == 3 for item in analyses["test"]):
        assert len(training) >= 5
        assert any(item[3] == "fill" and len(item[4]) == 2 and len(item[2]) == 2
                   for item in training)
    if any(item[3] == "fill" and item[2][-1][0][0] < max(item[4]) for item in analyses["test"]):
        assert any(item[3] == "fill" and item[2][-1][0][0] < max(item[4]) for item in training)
    for roles, vessels, trace, outcome, levels in analyses["test"]:
        if outcome == "fill":
            assert trace and trace[-1][0][4] == -1
        if len(trace) >= 2:
            for (vessel, entry), (next_vessel, next_entry) in zip(trace, trace[1:]):
                assert vessel[4] != -1 and vessel[4] == next_entry
                assert vessel[1] < next_vessel[0]


def _puzzle_key(puzzle):
    return tuple((split, tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                               for pair in puzzle[split])) for split in ("train", "test"))


def _source_example(input_rows, output_rows):
    inp = np.array([[int(cell) for cell in row] for row in input_rows], dtype=np.int8)
    out = np.array([[int(cell) for cell in row] for row in output_rows], dtype=np.int8)
    prediction, _, _, _ = _solve_details(inp, (0, 2, 1))
    assert np.array_equal(prediction, out)


def self_test() -> None:
    # Source training examples exercise both narrow-compartment orientations.
    _source_example(
        ("0000100000", "0000000000", "0000000000", "0020020200", "0020020200",
         "0020020200", "0022220200", "0000000000", "0000000000", "0000000000"),
        ("0000000000", "0000000000", "0000000000", "0021120200", "0021120200",
         "0021120200", "0022220200", "0000000000", "0000000000", "0000000000"),
    )
    _source_example(
        ("0000001000", "0000000000", "0000000000", "0020020200", "0020020200",
         "0020020200", "0020222200", "0000000000", "0000000000", "0000000000"),
        ("0000000000", "0000000000", "0000000000", "0020021200", "0020021200",
         "0020021200", "0020222200", "0000000000", "0000000000", "0000000000"),
    )
    _source_example(
        ("0000100000", "0000000000", "0000000000", "0020000200", "0020000200",
         "0020000200", "0022220200", "0000000000", "0000000000", "0000000000"),
        ("0000000000", "0000000000", "0000000000", "0020000200", "0020000200",
         "0020000200", "0022220200", "0000000000", "0000000000", "1111111111"),
    )
    _source_example(
        ("0000001000", "0000000000", "0000000000", "0020000200", "0020000200",
         "0020000200", "0022222200", "0000000000", "0000000000", "0000000000"),
        ("0000000000", "0000000000", "0000000000", "0021111200", "0021111200",
         "0021111200", "0022222200", "0000000000", "0000000000", "0000000000"),
    )
    # The source test input alone gives an upper gap at 6 and a lower intact vessel.
    source_test_input = np.array(
        [[int(cell) for cell in row] for row in
         ("0000000010", "0000000000", "0000020002", "0000020002", "0000020222",
          "0000000000", "0020000020", "0020000020", "0022222220", "0000000000")],
        dtype=np.int8,
    )
    inferred, source_vessels, source_trace, source_outcome = _solve_details(
        source_test_input, (0, 2, 1)
    )
    assert source_outcome == "fill" and len(source_trace) == 2
    assert source_trace[0][0][4] == 6 and source_trace[1][0][4] == -1
    assert source_trace[0][0][1] - source_trace[0][0][0] == source_trace[1][0][1] - source_trace[1][0][0]
    assert np.all(inferred[6:8, 3:8] == 1)
    assert np.array_equal(inferred == 2, source_test_input == 2)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(puzzle) for puzzle in puzzles}) == 42
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    shapes, normalized, roles_seen, vessel_counts = set(), set(), set(), set()
    outcomes = {"train": set(), "test": set()}
    widths, heights, test_levels, terminal_width_changes, terminal_height_changes = set(), set(), set(), set(), set()
    one_vessel = set()
    shared = False
    mixed_palette = False
    stable_palette = False
    gap_alignment = set()
    early_stop = False
    miss_upper = False
    leak_lower_spill = False
    narrow_types = set()
    narrow_reached = set()
    source_distances = set()
    level_spacing = set()
    ten_by_ten = 0
    train_stacked = False
    test_routes = set()
    off_route_types = set()
    first_train_regimes = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        first_training = _analysis(puzzle["train"][0])
        first_train_regimes.add((first_training[3], len(first_training[4])))
        palettes = set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                roles, vessels, trace, outcome, levels = _analysis(pair)
                palettes.add(roles)
                roles_seen.add(roles)
                shapes.add(inp.shape)
                widths.add(inp.shape[1])
                heights.add(inp.shape[0])
                normalized.add((inp.shape, tuple(roles.index(int(value)) for value in inp.ravel())))
                vessel_counts.add(len(vessels))
                outcomes[split].add(outcome)
                ten_by_ten += inp.shape == (10, 10)
                source_distances.add(min(levels))
                ordered_levels = sorted(levels)
                for upper, lower in zip(ordered_levels, ordered_levels[1:]):
                    upper_floor = max(vessel[1] for vessel in vessels if vessel[0] == upper)
                    level_spacing.add(lower - upper_floor - 1)
                selected = {item[0] for item in trace}
                for vessel in vessels:
                    kind = "intact" if vessel[4] == -1 else "broken"
                    if vessel[3] - vessel[2] - 1 == 1:
                        narrow_types.add(kind)
                        if vessel in selected:
                            narrow_reached.add(kind)
                    if vessel not in selected:
                        off_route_types.add(kind)
                if len(vessels) == 1 and len(levels) == 1:
                    one_vessel.add((split, outcome, len(trace)))
                if any(a[3] == b[2] for a in vessels for b in vessels if a != b and a[0] == b[0]):
                    shared = True
                for vessel, entry in trace:
                    if vessel[4] != -1:
                        gap_alignment.add(vessel[4] == entry)
                if split == "train" and len(levels) > 1:
                    train_stacked = True
                if split == "test":
                    test_levels.add(len(levels))
                    test_routes.add((outcome, len(trace) == 0, len(trace) >= 2))
                    if len(trace) >= 2 and outcome == "fill":
                        first, terminal = trace[0][0], trace[-1][0]
                        terminal_width_changes.add(np.sign((terminal[3] - terminal[2]) - (first[3] - first[2])))
                        terminal_height_changes.add(np.sign((terminal[1] - terminal[0]) - (first[1] - first[0])))
                    if outcome == "fill" and trace and trace[-1][0][0] < max(levels):
                        early_stop = True
                    if len(levels) > 1 and trace and trace[0][0][0] > min(levels):
                        miss_upper = True
                    if outcome == "spill" and trace and trace[-1][0][4] != -1 and len(levels) > 1:
                        leak_lower_spill = True
        mixed_palette |= len(palettes) > 1
        stable_palette |= len(palettes) == 1
    assert len(shapes) >= 35 and len(normalized) >= 100
    assert len(widths) >= 15 and len(heights) >= 15
    assert len(roles_seen) >= 30 and len(vessel_counts) >= 3
    assert outcomes["test"] == {"fill", "spill"}
    assert ("train", "fill", 1) in one_vessel and ("train", "spill", 1) in one_vessel
    assert {1, 2, 3}.issubset(test_levels)
    assert {-1, 0, 1}.issubset(terminal_width_changes)
    assert {-1, 0, 1}.issubset(terminal_height_changes)
    assert gap_alignment == {False, True}
    assert shared and early_stop and miss_upper and leak_lower_spill
    assert {("spill", 1), ("fill", 1), ("fill", 2)}.issubset(first_train_regimes)
    assert narrow_types == {"intact", "broken"}
    assert narrow_reached == {"intact", "broken"}
    assert source_distances == set(range(2, 8))
    assert level_spacing == {1, 2, 3, 4}
    assert ten_by_ten >= 15 and train_stacked
    assert off_route_types == {"intact", "broken"}
    assert {("fill", False, False), ("fill", False, True),
            ("spill", False, False), ("spill", True, False)}.issubset(test_routes)
    assert mixed_palette and stable_palette
    for index, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4), (7, 3))):
        puzzle = generate_puzzle(1000 + index, train_count, test_count)
        assert len(puzzle["train"]) == train_count and len(puzzle["test"]) == test_count
    for index, (train_count, test_count) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + index, train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    assert _puzzle_key(generate_puzzle(314159, 8, 3)) == _puzzle_key(generate_puzzle(314159, 8, 3))
    assert len({_puzzle_key(generate_puzzle(5000 + seed, 6, 2)) for seed in range(8)}) == 8
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
                      {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
                      {"num_train": True, "num_test": 1}):
        try:
            generate_puzzle(0, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {arguments}")
