import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        ("000002", "011111", "000000", "111110", "200000", "000000"),
        ("444442", "411111", "444444", "111114", "244444", "000000"),
    ),
    (
        ("00012", "00010", "01010", "01000", "21000"),
        ("00012", "44414", "41414", "41444", "21000"),
    ),
    (
        ("200000", "111100", "000000", "000111", "000002", "000000"),
        ("244440", "111140", "004440", "004111", "004442", "000000"),
    ),
    (
        (
            "0001100012",
            "0001100010",
            "0001100010",
            "0001100010",
            "0101101010",
            "0100001000",
            "0100001000",
            "0100001000",
            "0100001000",
            "2100001000",
        ),
        (
            "0001100012",
            "0001100014",
            "0001100014",
            "4441144414",
            "4141141414",
            "4144441444",
            "4100001000",
            "4100001000",
            "4100001000",
            "2100001000",
        ),
    ),
    (
        ("21112", "01010", "01010", "01010", "00000"),
        ("21112", "41014", "41014", "41014", "44444"),
    ),
    (
        (
            "2100010002",
            "0100010100",
            "0101010100",
            "0101010100",
            "0101010100",
            "0001000100",
            "0001010100",
            "0001010100",
            "0001010100",
            "0001010000",
        ),
        (
            "2100014442",
            "4144414100",
            "4141414100",
            "4141414100",
            "4141414100",
            "4441444100",
            "0001010100",
            "0001010100",
            "0001010100",
            "0001010000",
        ),
    ),
)


def _rows_signature(rows):
    return (
        len(rows),
        len(rows[0]),
        tuple(int(ch) for row in rows for ch in row),
    )


_SOURCE_PAIR_SIGNATURES = {
    (_rows_signature(input_rows), _rows_signature(output_rows))
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
}


def _grid_signature(grid):
    return (
        int(grid.shape[0]),
        int(grid.shape[1]),
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_signature(pair):
    return (
        _grid_signature(pair["input"]),
        _grid_signature(pair["output"]),
    )


def _check_count(name, value, low, high):
    if value is None:
        return

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _resolve_counts(rng, num_train, num_test):
    _check_count("num_train", num_train, 2, 10)
    _check_count("num_test", num_test, 1, 4)

    if num_train is None and num_test is None:
        code = int(rng.integers(0, 2**31))
        test_count = 1 + code % 4
        minimum_train = max(2, test_count + 1)
        train_count = (
            minimum_train
            + (code // 4) % (11 - minimum_train)
        )
    elif num_train is None:
        test_count = int(num_test)
        train_count = int(
            rng.integers(max(2, test_count + 1), 11)
        )
    elif num_test is None:
        train_count = int(num_train)
        maximum_test = min(4, train_count - 1)
        test_count = int(rng.integers(1, maximum_test + 1))
    else:
        train_count = int(num_train)
        test_count = int(num_test)

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(rng):
    colors = [
        int(value)
        for value in rng.choice(10, size=4, replace=False)
    ]
    return {
        "background": colors[0],
        "wall": colors[1],
        "terminal": colors[2],
        "route": colors[3],
    }


def _shortest_path_and_count(
    grid,
    wall_color,
    start,
    goal,
):
    height, width = grid.shape

    distances = np.full(
        (height, width),
        -1,
        dtype=np.int32,
    )
    ways = np.zeros(
        (height, width),
        dtype=np.int8,
    )
    parent_row = np.full(
        (height, width),
        -1,
        dtype=np.int16,
    )
    parent_col = np.full(
        (height, width),
        -1,
        dtype=np.int16,
    )

    queue_row = np.empty(
        height * width,
        dtype=np.int16,
    )
    queue_col = np.empty(
        height * width,
        dtype=np.int16,
    )

    start_row, start_col = start
    goal_row, goal_col = goal

    distances[start_row, start_col] = 0
    ways[start_row, start_col] = 1
    parent_row[start_row, start_col] = start_row
    parent_col[start_row, start_col] = start_col

    queue_row[0] = start_row
    queue_col[0] = start_col

    head = 0
    tail = 1

    while head < tail:
        row = int(queue_row[head])
        col = int(queue_col[head])
        head += 1

        next_distance = int(distances[row, col]) + 1

        for delta_row, delta_col in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            next_row = row + delta_row
            next_col = col + delta_col

            if not (
                0 <= next_row < height
                and 0 <= next_col < width
            ):
                continue

            if int(grid[next_row, next_col]) == wall_color:
                continue

            if distances[next_row, next_col] == -1:
                distances[next_row, next_col] = next_distance
                ways[next_row, next_col] = ways[row, col]
                parent_row[next_row, next_col] = row
                parent_col[next_row, next_col] = col

                queue_row[tail] = next_row
                queue_col[tail] = next_col
                tail += 1

            elif distances[next_row, next_col] == next_distance:
                ways[next_row, next_col] = min(
                    2,
                    int(ways[next_row, next_col])
                    + int(ways[row, col]),
                )

    if distances[goal_row, goal_col] == -1:
        return None, 0

    path = []
    row = goal_row
    col = goal_col

    while (row, col) != (start_row, start_col):
        path.append((row, col))
        row, col = (
            int(parent_row[row, col]),
            int(parent_col[row, col]),
        )

        if row < 0:
            return None, int(ways[goal_row, goal_col])

    path.append((start_row, start_col))
    path.reverse()

    return path, int(ways[goal_row, goal_col])


def _turns(path):
    steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(steps, steps[1:]))


def _source_side_cells(height, width, style, rng):
    if style == "same":
        row = int(rng.choice((0, height - 1)))
        return (row, 0), (row, width - 1)
    if style == "opposite":
        rows = (0, height - 1)
        if int(rng.integers(2)):
            rows = rows[::-1]
        return (rows[0], 0), (rows[1], width - 1)
    if style == "offset":
        return ((int(rng.integers(1, height - 1)), 0),
                (int(rng.integers(1, height - 1)), width - 1))
    if style == "mixed":
        corner_row = int(rng.choice((0, height - 1)))
        side_row = int(rng.integers(1, height - 1))
        if int(rng.integers(2)):
            return (corner_row, 0), (side_row, width - 1)
        return (side_row, 0), (corner_row, width - 1)
    raise ValueError("unknown endpoint style")


def _make_bars(rng, mode):
    if mode in ("split", "interior", "mixed"):
        height = int(rng.integers(8, 15))
        barrier_count = int(rng.integers(3 if mode == "mixed" else 2, 5))
    elif mode == "many_bars":
        height = int(rng.integers(9, 15))
        barrier_count = int(rng.integers(4, 6))
    else:
        height = int(rng.integers(5, 14))
        barrier_count = int(rng.integers(1, 5))

    thicknesses = [int(rng.choice((1, 1, 1, 2, 2, 3)))
                   for _ in range(barrier_count)]
    if barrier_count == 1:
        thicknesses[0] = int(rng.integers(3, 5))
    if mode in ("split", "interior", "mixed"):
        selected = int(rng.integers(barrier_count))
        thicknesses[selected] = 1
        other = int(rng.choice([index for index in range(barrier_count) if index != selected])) if mode == "mixed" else -1
        if other >= 0:
            thicknesses[other] = 1
    else:
        selected = -1
        other = -1
    gaps = [int(rng.choice((1, 1, 1, 1, 2, 2, 3)))
            for _ in range(barrier_count - 1)]
    # Large spaces are useful chiefly with short chains: long chains otherwise
    # have many equally short choices within the open corridor.
    if barrier_count >= 4 and int(rng.integers(3)):
        gaps = [min(gap, 2) for gap in gaps]
    width = 2 + sum(thicknesses) + sum(gaps)
    if width > 19:
        return None
    wall_mask = np.zeros((height, width), dtype=bool)
    spans = []
    col = 1
    top_first = bool(rng.integers(2))
    for index, thickness in enumerate(thicknesses):
        top = top_first == (index % 2 == 0)
        clearance = int(rng.integers(1, min(4, height - 2)))
        if top:
            first, last = 0, height - clearance - 1
        else:
            first, last = clearance, height - 1
        wall_mask[first:last + 1, col:col + thickness] = True
        spans.append((first, last, col, col + thickness - 1, top))
        col += thickness + (gaps[index] if index < len(gaps) else 0)

    if mode in ("split", "mixed"):
        first, last, left, right, _ = spans[selected]
        if last - first + 1 < 7:
            return None
        gap_row = int(rng.integers(first + 3, last - 2))
        wall_mask[gap_row, left:right + 1] = False
        opened = {(gap_row, c) for c in range(left, right + 1)}
    else:
        opened = set()
    if mode in ("interior", "mixed"):
        first, last, left, right, top = spans[other if mode == "mixed" else selected]
        if last - first + 1 < 5:
            return None
        gap_row = first if top else last
        wall_mask[gap_row, left:right + 1] = False
        opened.update((gap_row, c) for c in range(left, right + 1))

    style = str(rng.choice(("same", "same", "opposite", "offset", "mixed")))
    start, goal = _source_side_cells(height, width, style, rng)
    if wall_mask[start] or wall_mask[goal]:
        return None
    return wall_mask, start, goal, opened


def _make_u(rng):
    height = int(rng.integers(5, 14))
    width = int(rng.integers(5, 15))
    wall_mask = np.zeros((height, width), dtype=bool)
    left_width = int(rng.choice((1, 1, 2)))
    right_width = int(rng.choice((1, 1, 2)))
    left_margin = int(rng.choice((1, 1, 2)))
    right_margin = int(rng.choice((1, 1, 2)))
    if width - left_width - right_width - left_margin - right_margin < 1:
        return None
    left_col = left_margin
    right_col = width - right_width - right_margin
    first_depth = height - int(rng.choice((1, 1, 2)))
    second_depth = height - int(rng.choice((1, 1, 2)))
    wall_mask[0, left_col:right_col + right_width] = True
    wall_mask[:first_depth, left_col:left_col + left_width] = True
    wall_mask[:second_depth, right_col:right_col + right_width] = True
    if int(rng.integers(2)):
        start = (0, left_col - 1)
        goal = (0, right_col + right_width)
    else:
        first_row = int(rng.integers(0, min(first_depth, height - 1)))
        second_row = int(rng.integers(0, min(second_depth, height - 1)))
        start = (first_row, 0)
        goal = (second_row, width - 1)
    return wall_mask, start, goal, set()


def _add_branch(rng, wall_mask, start, goal):
    height, width = wall_mask.shape
    candidates = []
    for row in range(1, height - 1):
        for col in range(1, width - 1):
            if wall_mask[row, col] or (row, col) in (start, goal):
                continue
            vertical = int(wall_mask[row - 1, col]) + int(wall_mask[row + 1, col])
            horizontal = int(wall_mask[row, col - 1]) + int(wall_mask[row, col + 1])
            if vertical == 1 and horizontal == 0:
                candidates.append((row, col))
            elif horizontal == 1 and vertical == 0:
                candidates.append((row, col))
    if not candidates:
        return None
    row, col = candidates[int(rng.integers(len(candidates)))]
    result = wall_mask.copy()
    result[row, col] = True
    return result


def _orient(mask, start, goal, opened, rng):
    k = int(rng.integers(4))
    reflect = bool(rng.integers(2))
    height, width = mask.shape
    def transform(cell):
        row, col = cell
        h, w = height, width
        for _ in range(k):
            row, col = w - 1 - col, row
            h, w = w, h
        if reflect:
            col = w - 1 - col
        return row, col
    result = np.rot90(mask, k).copy()
    if reflect:
        result = result[:, ::-1].copy()
    return result, transform(start), transform(goal), {transform(cell) for cell in opened}


def _build_candidate(rng, mode):
    if mode == "u":
        built = _make_u(rng)
    else:
        built = _make_bars(rng, mode)
    if built is None:
        return None
    wall_mask, start, goal, opened = built
    if mode == "branch":
        wall_mask = _add_branch(rng, wall_mask, start, goal)
        if wall_mask is None:
            return None
    return _orient(wall_mask, start, goal, opened, rng)


def _path_from_mask(mask, start, goal):
    path, count = _shortest_path_and_count(mask.astype(np.int8), 1, start, goal)
    if path is None or count != 1 or len(path) <= 2:
        return None
    distance = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    if len(path) - 1 <= distance or _turns(path) < 2:
        return None
    if int(np.count_nonzero(~mask)) <= len(path) + 2:
        return None
    return path


def generate_pair(rng, regime, puzzle_parameters, mode=None):
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown pair regime")
    if mode is None:
        mode = str(rng.choice(("bars", "u", "split", "interior", "branch")))
    for _ in range(320):
        candidate = _build_candidate(rng, mode)
        if candidate is None:
            continue
        wall_mask, start, goal, opened = candidate
        if any(len(component) < 3 for component in _wall_components(wall_mask)):
            continue
        path = _path_from_mask(wall_mask, start, goal)
        if path is None:
            continue
        used_openings = opened.intersection(path)
        if mode in ("split", "interior") and not used_openings:
            continue
        if mode == "mixed" and len(used_openings) < 2:
            continue
        if mode == "many_bars" and len(_wall_components(wall_mask)) < 4:
            continue
        if mode in ("u", "branch") and all(
            _is_rectangle(component) for component in _wall_components(wall_mask)
        ):
            continue
        background = puzzle_parameters["background"]
        wall = puzzle_parameters["wall"]
        terminal = puzzle_parameters["terminal"]
        route = puzzle_parameters["route"]
        input_grid = np.full(wall_mask.shape, background, dtype=np.int8)
        input_grid[wall_mask] = wall
        input_grid[start] = terminal
        input_grid[goal] = terminal
        output_grid = input_grid.copy()
        for cell in path[1:-1]:
            output_grid[cell] = route
        return input_grid, output_grid
    raise RuntimeError("could not construct a unique route for mode " + mode)


def _infer_roles(pair):
    input_grid, output_grid = pair["input"], pair["output"]
    assert input_grid.shape == output_grid.shape
    changed = input_grid != output_grid
    assert np.any(changed)
    old_values = np.unique(input_grid[changed])
    new_values = np.unique(output_grid[changed])
    assert len(old_values) == len(new_values) == 1
    background, route = int(old_values[0]), int(new_values[0])
    assert background != route
    input_colors = {int(color) for color in np.unique(input_grid)}
    assert len(input_colors) == 3 and background in input_colors
    assert route not in input_colors
    candidates = [color for color in input_colors - {background}
                  if int(np.count_nonzero(input_grid == color)) == 2]
    assert len(candidates) == 1
    terminal = candidates[0]
    wall = next(iter(input_colors - {background, terminal}))
    assert int(np.count_nonzero(input_grid == wall)) >= 3
    assert {int(color) for color in np.unique(output_grid)} == input_colors | {route}
    return background, wall, terminal, route


def _wall_components(mask):
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or visited[row, col]:
                continue
            cells = []
            stack = [(row, col)]
            visited[row, col] = True
            while stack:
                current_row, current_col = stack.pop()
                cells.append((current_row, current_col))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = current_row + dr, current_col + dc
                    if 0 <= rr < height and 0 <= cc < width and mask[rr, cc] and not visited[rr, cc]:
                        visited[rr, cc] = True
                        stack.append((rr, cc))
            components.append(cells)
    return components


def _is_rectangle(component):
    rows = [cell[0] for cell in component]
    cols = [cell[1] for cell in component]
    return len(component) == (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def _terminals_on_boundary(grid, terminal):
    height, width = grid.shape
    cells = [tuple(int(x) for x in cell) for cell in np.argwhere(grid == terminal)]
    return len(cells) == 2 and all(
        row in (0, height - 1) or col in (0, width - 1) for row, col in cells
    )


def _validate_pair_rule(pair, roles):
    background, wall, terminal, route = roles
    input_grid, output_grid = pair["input"], pair["output"]
    assert _infer_roles(pair) == roles
    assert _terminals_on_boundary(input_grid, terminal)
    components = _wall_components(input_grid == wall)
    assert components and all(len(component) >= 3 for component in components)
    endpoints = [tuple(int(x) for x in cell) for cell in np.argwhere(input_grid == terminal)]
    path, count = _shortest_path_and_count(input_grid, wall, endpoints[0], endpoints[1])
    assert path is not None and count == 1 and len(path) > 2
    assert len(path) - 1 > abs(endpoints[0][0] - endpoints[1][0]) + abs(endpoints[0][1] - endpoints[1][1])
    assert _turns(path) >= 2
    assert int(np.count_nonzero(input_grid == background)) > len(path)
    expected = input_grid.copy()
    for cell in path[1:-1]:
        assert input_grid[cell] == background
        expected[cell] = route
    assert np.array_equal(output_grid, expected)
    return path


def _predict_from_training(train, test_input):
    roles = _infer_roles(train[0])
    assert all(_infer_roles(pair) == roles for pair in train)
    background, wall, terminal, route = roles
    assert {int(x) for x in np.unique(test_input)} == {background, wall, terminal}
    endpoints = [tuple(int(x) for x in cell) for cell in np.argwhere(test_input == terminal)]
    assert len(endpoints) == 2
    path, count = _shortest_path_and_count(test_input, wall, endpoints[0], endpoints[1])
    assert path is not None and count == 1
    prediction = test_input.copy()
    for cell in path[1:-1]:
        prediction[cell] = route
    return prediction


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    inputs, pairs = set(), set()
    roles = None
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            signature = _pair_signature(pair)
            assert signature not in pairs and signature not in _SOURCE_PAIR_SIGNATURES
            input_signature = _grid_signature(pair["input"])
            assert input_signature not in inputs
            pairs.add(signature)
            inputs.add(input_signature)
            inferred = _infer_roles(pair)
            if roles is None:
                roles = inferred
            assert inferred == roles
            _validate_pair_rule(pair, roles)
            if split == "test":
                assert np.array_equal(_predict_from_training(train, pair["input"]), pair["output"])
    assert roles is not None
    # Even the minimum training count includes a genuinely compound or split wall.
    assert any(any(not _is_rectangle(comp) for comp in _wall_components(
        pair["input"] == roles[1])) or len(_wall_components(pair["input"] == roles[1])) >= 3
        for pair in train)


def _mode_plan(rng, train_count, test_count):
    train_modes = ["u", "bars"]
    if train_count >= 3:
        train_modes.append("split")
    if train_count >= 4:
        train_modes.append("many_bars")
    if train_count >= 5:
        train_modes.append("interior")
    if train_count >= 6:
        train_modes.append("mixed")
    while len(train_modes) < train_count:
        train_modes.append(str(rng.choice(("bars", "bars", "u", "split", "interior", "branch", "many_bars", "mixed"))))
    rng.shuffle(train_modes)
    test_modes = [str(rng.choice(("bars", "bars", "u", "split", "split", "interior", "interior", "branch", "many_bars", "mixed", "mixed")))
                  for _ in range(test_count)]
    return train_modes, test_modes


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    parameters = sample_puzzle_parameters(rng)
    train_modes, test_modes = _mode_plan(rng, train_count, test_count)
    puzzle = {"train": [], "test": []}
    used_inputs, used_pairs = set(), set()
    for split, modes, regime in (("train", train_modes, "source_train"),
                                 ("test", test_modes, "target_test")):
        for mode in modes:
            for attempt in range(16):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                try:
                    input_grid, output_grid = generate_pair(pair_rng, regime, parameters, mode)
                except RuntimeError:
                    if attempt == 15:
                        raise
                    continue
                pair = {"input": input_grid, "output": output_grid}
                input_signature = _grid_signature(input_grid)
                pair_signature = _pair_signature(pair)
                if input_signature in used_inputs or pair_signature in used_pairs or pair_signature in _SOURCE_PAIR_SIGNATURES:
                    continue
                puzzle[split].append(pair)
                used_inputs.add(input_signature)
                used_pairs.add(pair_signature)
                break
            else:
                raise RuntimeError("could not construct distinct puzzle pairs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a["input"], b["input"]) and np.array_equal(a["output"], b["output"])
        for a, b in zip(first[split], second[split])) for split in ("train", "test"))


def _puzzle_signature(puzzle):
    return tuple((split, tuple(_pair_signature(pair) for pair in puzzle[split]))
                 for split in ("train", "test"))


def _split_component_pairs(components):
    pairs = set()
    for first_index, first in enumerate(components):
        if not _is_rectangle(first):
            continue
        for second_index in range(first_index + 1, len(components)):
            second = components[second_index]
            if not _is_rectangle(second):
                continue
            for transpose in (False, True):
                a = [(col, row) if transpose else (row, col) for row, col in first]
                b = [(col, row) if transpose else (row, col) for row, col in second]
                a_rows, b_rows = {row for row, _ in a}, {row for row, _ in b}
                a_cols, b_cols = {col for _, col in a}, {col for _, col in b}
                if len(a_cols) != 1 or a_cols != b_cols or a_rows & b_rows:
                    continue
                gap = max(min(a_rows), min(b_rows)) - min(max(a_rows), max(b_rows)) - 1
                if 1 <= gap <= 3:
                    pairs.add((first_index, second_index))
    return pairs


def _wide_passage(components, path):
    for first_index, first in enumerate(components):
        if not _is_rectangle(first):
            continue
        for second in components[first_index + 1:]:
            if not _is_rectangle(second):
                continue
            for transpose in (False, True):
                a = [(col, row) if transpose else (row, col) for row, col in first]
                b = [(col, row) if transpose else (row, col) for row, col in second]
                a_rows, b_rows = [row for row, _ in a], [row for row, _ in b]
                a_cols, b_cols = [col for _, col in a], [col for _, col in b]
                if (max(a_rows) - min(a_rows) < 3 or max(b_rows) - min(b_rows) < 3
                        or max(a_cols) - min(a_cols) > 2 or max(b_cols) - min(b_cols) > 2):
                    continue
                left, right = sorted(((min(a_cols), max(a_cols)), (min(b_cols), max(b_cols))))
                gap = right[0] - left[1] - 1
                if 2 <= gap <= 4 and any(
                    left[1] < (row if transpose else col) < right[0]
                    for row, col in path
                ):
                    return True
    return False


def _features(pair):
    roles = _infer_roles(pair)
    grid = pair["input"]
    components = _wall_components(grid == roles[1])
    terminals = [tuple(int(x) for x in cell) for cell in np.argwhere(grid == roles[2])]
    path, _ = _shortest_path_and_count(grid, roles[1], terminals[0], terminals[1])
    height, width = grid.shape
    sides = [set(name for name, condition in (("top", r == 0), ("bottom", r == height - 1),
             ("left", c == 0), ("right", c == width - 1)) if condition) for r, c in terminals]
    noncorner = any(len(side) == 1 for side in sides)
    same_side = bool(sides[0] & sides[1])
    interior_indices = {index for index, comp in enumerate(components)
                        if all(0 < r < height - 1 and 0 < c < width - 1 for r, c in comp)}
    split_pairs = _split_component_pairs(components)
    split_indices = {index for pair_indices in split_pairs for index in pair_indices}
    wall_mask = grid == roles[1]
    thick = bool(np.any(wall_mask[:-1, :-1] & wall_mask[1:, :-1] &
                        wall_mask[:-1, 1:] & wall_mask[1:, 1:]))
    far_wall = sum(all(abs(row - pr) + abs(col - pc) > 1 for pr, pc in path)
                   for row, col in np.argwhere(wall_mask)) >= 3
    path_cells = set(path)
    dead_end = False
    for row, col in np.argwhere(grid == roles[0]):
        row, col = int(row), int(col)
        if (row, col) in path_cells:
            continue
        degree = sum(0 <= rr < height and 0 <= cc < width and not wall_mask[rr, cc]
                     for rr, cc in ((row - 1, col), (row + 1, col),
                                    (row, col - 1), (row, col + 1)))
        if degree == 1:
            dead_end = True
            break
    alternative = False
    for cell in path[1:-1]:
        blocked = grid.copy()
        blocked[cell] = roles[1]
        other_path, _ = _shortest_path_and_count(blocked, roles[1], terminals[0], terminals[1])
        if other_path is not None:
            assert len(other_path) > len(path)
            alternative = True
            break
    return {
        "shape": grid.shape,
        "components": len(components),
        "nonrect": any(not _is_rectangle(comp) for comp in components),
        "noncorner": noncorner,
        "same_side": same_side,
        "interior": bool(interior_indices),
        "split": bool(split_pairs),
        "mixed": bool(split_pairs and interior_indices - split_indices),
        "wide": _wide_passage(components, path),
        "thick": thick,
        "far_wall": far_wall,
        "dead_end": dead_end,
        "alternative": alternative,
        "turns": _turns(path),
        "detour": len(path) - 1 - (abs(terminals[0][0] - terminals[1][0]) +
                                    abs(terminals[0][1] - terminals[1][1])),
    }


def self_test() -> None:
    for seed in range(6):
        validate_generated_puzzle(generate_puzzle(seed))
    first = generate_puzzle(314159, 6, 3)
    assert _puzzles_equal(first, generate_puzzle(314159, 6, 3))
    assert len({_puzzle_signature(generate_puzzle(900 + seed, 5, 2)) for seed in range(8)}) >= 7
    for index, (ntrain, ntest) in enumerate(((2, 1), (10, 1), (10, 4), (6, 2))):
        puzzle = generate_puzzle(2000 + index, ntrain, ntest)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (ntrain, ntest)
        validate_generated_puzzle(puzzle)
    for index, (ntrain, ntest) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(3000 + index, ntrain, ntest)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
        validate_generated_puzzle(puzzle)
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(0, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts, shapes, component_counts, palettes, signatures = (set() for _ in range(6))
    measured = {key: 0 for key in ("nonrect", "noncorner", "same_side", "interior",
                                   "split", "mixed", "wide", "thick", "far_wall",
                                   "dead_end", "alternative")}
    same_by_components = {}
    test_small = test_large = 0
    index_to_components = [set() for _ in range(10)]
    puzzle_varied = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        palettes.add(_infer_roles(puzzle["train"][0]))
        signatures.add(_puzzle_signature(puzzle))
        local_shapes, local_wall_patterns = set(), set()
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                feature = _features(pair)
                shape = feature["shape"]
                count = feature["components"]
                shapes.add(shape)
                component_counts.add(count)
                for key in measured:
                    measured[key] += int(feature[key])
                same_by_components.setdefault(count, set()).add(feature["same_side"])
                assert feature["turns"] >= 2 and feature["detour"] >= 2
                local_shapes.add(shape)
                local_wall_patterns.add((count, feature["nonrect"], feature["interior"], feature["split"]))
                if split == "train":
                    index_to_components[index].add(count)
                elif max(shape) <= 9:
                    test_small += 1
                elif max(shape) >= 11:
                    test_large += 1
        puzzle_varied += int(len(local_shapes) >= 2 and len(local_wall_patterns) >= 2)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(palettes) >= 20 and len(shapes) >= 20 and len(signatures) >= 40
    assert {1, 2, 3, 4}.issubset(component_counts)
    assert measured["nonrect"] >= 30 and measured["noncorner"] >= 30
    assert measured["same_side"] >= 20 and measured["same_side"] <= 360
    assert measured["interior"] >= 20 and measured["split"] >= 10
    assert measured["mixed"] >= 5 and measured["wide"] >= 10
    assert measured["thick"] >= 30 and measured["far_wall"] >= 30
    assert measured["dead_end"] >= 20
    assert measured["alternative"] >= 30
    assert all(same_by_components.get(count) == {False, True} for count in (1, 2, 3, 4))
    assert test_small >= 5 and test_large >= 10
    assert all(len(index_to_components[index]) >= 2 for index in range(2))
    assert puzzle_varied >= 35
