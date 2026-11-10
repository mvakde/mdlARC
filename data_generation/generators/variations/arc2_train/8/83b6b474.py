import numpy as np


_SOURCE_PAIRS = (
    (
        (
            "999999",
            "982222",
            "989999",
            "488999",
            "999919",
            "999119",
        ),
        (
            "2222",
            "8994",
            "8991",
            "8811",
        ),
    ),
    (
        (
            "2222222222222222",
            "2222222222222122",
            "2222222222222122",
            "2222222222222122",
            "2222228888222222",
            "2222222222222222",
            "2222222222222222",
            "2222222222222222",
            "2266222222222222",
            "2262222222222922",
            "2222222222999922",
            "2222422222222222",
            "2222422222222222",
            "2222422222222222",
            "2222442222222222",
            "2222222222222222",
        ),
        (
            "668888",
            "622221",
            "422221",
            "422221",
            "422229",
            "449999",
        ),
    ),
    (
        (
            "66888886",
            "69666666",
            "69666666",
            "69666666",
            "69616466",
            "69616466",
            "99611116",
            "66666666",
        ),
        (
            "888889",
            "466669",
            "466669",
            "166669",
            "166669",
            "111199",
        ),
    ),
)


def _validate_optional_count(value, name, lower, upper):
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


def _resolve_counts(seed, num_train, num_test):
    train_count = _validate_optional_count(
        num_train, "num_train", 2, 10
    )
    test_count = _validate_optional_count(
        num_test, "num_test", 1, 4
    )

    if (
        train_count is not None
        and test_count is not None
        and train_count <= test_count
    ):
        raise ValueError("num_train must be greater than num_test")

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        train_count = (
            test_count
            + 1
            + (seed // 4) % (10 - test_count)
        )
    elif train_count is None:
        minimum = max(2, test_count + 1)
        train_count = minimum + seed % (11 - minimum)
    elif test_count is None:
        maximum = min(4, train_count - 1)
        test_count = 1 + seed % maximum

    if not (
        2 <= train_count <= 10
        and 1 <= test_count <= 4
        and train_count > test_count
    ):
        raise ValueError("illegal train/test count combination")

    return int(train_count), int(test_count)


def sample_puzzle_parameters(rng):
    # The source has no hidden cross-pair convention. Colors are local roles;
    # this palette only supplies varied literal assignments.
    return tuple(int(value) for value in rng.permutation(10))


def _perimeter(side):
    cells = []

    for col in range(side):
        cells.append((0, col))
    for row in range(1, side):
        cells.append((row, side - 1))
    for col in range(side - 2, -1, -1):
        cells.append((side - 1, col))
    for row in range(side - 2, 0, -1):
        cells.append((row, 0))

    assert len(cells) == 4 * side - 4
    return tuple(cells)


def _edge_interiors(side):
    edges = [[], [], [], []]

    for index, (row, col) in enumerate(_perimeter(side)):
        if row == 0 and 0 < col < side - 1:
            edges[0].append(index)
        elif col == side - 1 and 0 < row < side - 1:
            edges[1].append(index)
        elif row == side - 1 and 0 < col < side - 1:
            edges[2].append(index)
        elif col == 0 and 0 < row < side - 1:
            edges[3].append(index)

    return edges


def _normalized_shape(coords):
    array = np.asarray(coords, dtype=np.int64)
    minimum_row = int(array[:, 0].min())
    minimum_col = int(array[:, 1].min())

    return tuple(
        sorted(
            (
                int(row) - minimum_row,
                int(col) - minimum_col,
            )
            for row, col in array
        )
    )


def _is_connected(coords):
    points = {
        (int(row), int(col))
        for row, col in np.asarray(coords).tolist()
    }
    if not points:
        return False

    stack = [next(iter(points))]
    visited = set()

    while stack:
        point = stack.pop()
        if point in visited:
            continue
        visited.add(point)
        row, col = point
        for neighbor in (
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ):
            if neighbor in points and neighbor not in visited:
                stack.append(neighbor)

    return len(visited) == len(points)


def _is_bent(shape):
    rows = {row for row, _ in shape}
    cols = {col for _, col in shape}
    return len(rows) > 1 and len(cols) > 1


def _solve_shape_assemblies(shapes, side, background, limit=2):
    perimeter_cells = _perimeter(side)
    target_mask = 0
    for row, col in perimeter_cells:
        target_mask |= 1 << (row * side + col)

    candidates = {}

    for color, shape in shapes.items():
        height = max(row for row, _ in shape) + 1
        width = max(col for _, col in shape) + 1
        color_candidates = []

        for anchor_row in range(side - height + 1):
            for anchor_col in range(side - width + 1):
                translated = tuple(
                    (
                        anchor_row + row,
                        anchor_col + col,
                    )
                    for row, col in shape
                )

                if not all(
                    row == 0
                    or row == side - 1
                    or col == 0
                    or col == side - 1
                    for row, col in translated
                ):
                    continue

                mask = 0
                for row, col in translated:
                    mask |= 1 << (row * side + col)
                color_candidates.append((mask, translated))

        if not color_candidates:
            return []

        candidates[int(color)] = color_candidates

    order = sorted(
        candidates,
        key=lambda color: (len(candidates[color]), color),
    )
    solutions = []
    placements = {}

    def search(position, occupied_mask):
        if len(solutions) >= limit:
            return

        if position == len(order):
            if occupied_mask != target_mask:
                return

            grid = np.full(
                (side, side),
                int(background),
                dtype=np.int8,
            )
            for placed_color, cells in placements.items():
                for row, col in cells:
                    grid[row, col] = int(placed_color)
            solutions.append(grid)
            return

        color = order[position]
        for mask, cells in candidates[color]:
            if occupied_mask & mask:
                continue
            placements[color] = cells
            search(position + 1, occupied_mask | mask)
            placements.pop(color, None)

            if len(solutions) >= limit:
                return

    search(0, 0)
    return solutions


def _segments_from_cuts(perimeter, cuts):
    cuts = sorted(int(cut) for cut in cuts)
    count = len(cuts)
    length = len(perimeter)
    segments = []

    for index, start in enumerate(cuts):
        if index + 1 < count:
            cell_indices = range(start, cuts[index + 1])
            segment = tuple(perimeter[i] for i in cell_indices)
        else:
            segment = tuple(
                [perimeter[i] for i in range(start, length)]
                + [perimeter[i] for i in range(0, cuts[0])]
            )
        segments.append(segment)

    return segments


def _random_cuts(rng, side, piece_count):
    length = 4 * side - 4
    style = int(rng.integers(0, 5))
    required = set()
    if style == 1:  # A color boundary falls on a corner.
        required.add((side - 1) * int(rng.integers(0, 4)))
    elif style == 2:  # A singleton or a very short arc.
        start = int(rng.integers(0, length))
        required.update((start, (start + int(rng.integers(1, 3))) % length))
    elif style == 3:  # A complete straight side is one fragment.
        edge = int(rng.integers(0, 4))
        required.update((edge * (side - 1), (edge * (side - 1) + side) % length))
    elif style == 4:  # A long arc is likely to turn twice.
        start = int(rng.integers(0, length))
        span = int(rng.integers(side + 1, min(length - 1, 2 * side + 2)))
        required.update((start, (start + span) % length))
    remaining = [index for index in range(length) if index not in required]
    extra = rng.choice(remaining, size=piece_count - len(required), replace=False)
    return sorted(required | {int(value) for value in extra})


def _construct_output(
    rng,
    side,
    piece_count,
    background,
    piece_colors,
    regime,
):
    perimeter = _perimeter(side)

    for attempt in range(400):
        cuts = _random_cuts(rng, side, piece_count)

        segments = _segments_from_cuts(perimeter, cuts)
        assigned_colors = [
            int(color)
            for color in rng.permutation(
                np.asarray(piece_colors, dtype=np.int8)
            )
        ]

        output = np.full(
            (side, side),
            int(background),
            dtype=np.int8,
        )
        shapes = {}

        for color, segment in zip(assigned_colors, segments):
            for row, col in segment:
                output[row, col] = color
            shapes[color] = _normalized_shape(segment)

        solutions = _solve_shape_assemblies(
            shapes,
            side,
            background,
            limit=2,
        )
        if (
            len(solutions) == 1
            and np.array_equal(solutions[0], output)
        ):
            return output, shapes

    raise RuntimeError("could not construct a unique frame assembly")


def _candidate_translations(shape, canvas_size):
    height = max(row for row, _ in shape) + 1
    width = max(col for _, col in shape) + 1
    candidates = []

    for anchor_row in range(canvas_size[0] - height + 1):
        for anchor_col in range(canvas_size[1] - width + 1):
            candidates.append(
                tuple(
                    (
                        anchor_row + row,
                        anchor_col + col,
                    )
                    for row, col in shape
                )
            )

    return candidates


def _touches_occupied(cells, occupied):
    for row, col in cells:
        if (
            (row - 1, col) in occupied
            or (row + 1, col) in occupied
            or (row, col - 1) in occupied
            or (row, col + 1) in occupied
        ):
            return True
    return False


def _has_cross_color_contact(grid, background):
    height, width = grid.shape

    for row in range(height):
        for col in range(width):
            color = int(grid[row, col])
            if color == background:
                continue

            if row + 1 < height:
                other = int(grid[row + 1, col])
                if other != background and other != color:
                    return True

            if col + 1 < width:
                other = int(grid[row, col + 1])
                if other != background and other != color:
                    return True

    return False


def _has_interleaved_boxes(grid, colors):
    boxes = []
    for color in colors:
        cells = np.argwhere(grid == color)
        boxes.append((int(cells[:, 0].min()), int(cells[:, 0].max()),
                      int(cells[:, 1].min()), int(cells[:, 1].max())))
    return any(
        max(a[0], b[0]) <= min(a[1], b[1])
        and max(a[2], b[2]) <= min(a[3], b[3])
        for index, a in enumerate(boxes) for b in boxes[index + 1:]
    )


def _input_is_already_assembled(
    input_grid, output_grid, background
):
    foreground = np.argwhere(input_grid != background)
    minimum_row = int(foreground[:, 0].min())
    maximum_row = int(foreground[:, 0].max())
    minimum_col = int(foreground[:, 1].min())
    maximum_col = int(foreground[:, 1].max())

    if (
        maximum_row - minimum_row + 1,
        maximum_col - minimum_col + 1,
    ) != output_grid.shape:
        return False

    crop = input_grid[
        minimum_row : maximum_row + 1,
        minimum_col : maximum_col + 1,
    ]
    return np.array_equal(crop, output_grid)


def _scatter_fragments(
    rng,
    shapes,
    background,
    output,
    canvas_size,
    prefer_contact,
):
    all_candidates = {
        color: _candidate_translations(shape, canvas_size)
        for color, shape in shapes.items()
    }
    colors_array = np.asarray(list(shapes), dtype=np.int8)

    for _ in range(200):
        occupied = set()
        placed = {}
        order = [
            int(color) for color in rng.permutation(colors_array)
        ]
        failed = False

        for color in order:
            valid = [
                cells
                for cells in all_candidates[color]
                if occupied.isdisjoint(cells)
            ]
            if not valid:
                failed = True
                break

            if occupied:
                touching = [
                    cells
                    for cells in valid
                    if _touches_occupied(cells, occupied)
                ]
                separated = [
                    cells
                    for cells in valid
                    if not _touches_occupied(cells, occupied)
                ]

                if prefer_contact and touching:
                    pool = touching
                elif not prefer_contact and separated:
                    pool = separated
                else:
                    pool = valid
            else:
                pool = valid

            chosen = pool[int(rng.integers(0, len(pool)))]
            placed[color] = chosen
            occupied.update(chosen)

        if failed or len(placed) != len(shapes):
            continue

        input_grid = np.full(
            canvas_size,
            int(background),
            dtype=np.int8,
        )
        for color, cells in placed.items():
            for row, col in cells:
                input_grid[row, col] = color

        if (
            prefer_contact
            and not _has_cross_color_contact(
                input_grid, background
            )
        ):
            continue

        if _input_is_already_assembled(
            input_grid, output, background
        ):
            continue

        return input_grid

    raise RuntimeError("could not scatter fragments without overlap")


def _choose_pair_colors(rng, piece_count, palette):
    selected = rng.choice(10, size=piece_count + 1, replace=False)
    colors = [palette[int(index)] for index in selected]
    return colors[0], colors[1:]


def _generate_pair(
    rng,
    regime,
    puzzle_parameters,
    archetype,
):
    side, piece_count = archetype
    background, piece_colors = _choose_pair_colors(
        rng, piece_count, puzzle_parameters
    )

    output, shapes = _construct_output(
        rng,
        side,
        piece_count,
        background,
        piece_colors,
        regime,
    )

    # Compact source-style canvases and spacious scatters share the same rule.
    minimum = side + 2
    maximum = min(30, side + 12)
    canvas_mode = int(rng.integers(0, 8))
    if canvas_mode == 0:
        first = second = minimum
    elif canvas_mode == 1:
        first = second = min(30, side + 10)
    elif canvas_mode == 2:
        first = second = int(rng.integers(minimum, maximum + 1))
    elif canvas_mode in (3, 4):
        first = int(rng.integers(minimum, min(minimum + 3, maximum + 1)))
        second = int(rng.integers(minimum, min(minimum + 5, maximum + 1)))
    else:
        first = int(rng.integers(minimum, maximum + 1))
        second = int(rng.integers(minimum, maximum + 1))
    canvas_size = (first, second)
    prefer_contact = bool(rng.integers(0, 2))

    input_grid = _scatter_fragments(
        rng,
        shapes,
        background,
        output,
        canvas_size,
        prefer_contact,
    )

    return {
        "input": input_grid.astype(np.int8, copy=False),
        "output": output.astype(np.int8, copy=False),
    }


def _grid_key(grid):
    return (
        int(grid.shape[0]),
        int(grid.shape[1]),
        tuple(int(value) for value in grid.reshape(-1)),
    )


def _pair_key(pair):
    return (
        _grid_key(pair["input"]),
        _grid_key(pair["output"]),
    )


def _matches_rows(grid, rows):
    if grid.shape != (len(rows), len(rows[0])):
        return False

    for row_index, text in enumerate(rows):
        for col_index, character in enumerate(text):
            if int(grid[row_index, col_index]) != int(character):
                return False
    return True


def _is_source_pair(input_grid, output_grid):
    return any(
        _matches_rows(input_grid, input_rows)
        and _matches_rows(output_grid, output_rows)
        for input_rows, output_rows in _SOURCE_PAIRS
    )


def _analyze_pair(pair):
    input_grid = pair["input"]
    output_grid = pair["output"]

    assert input_grid.shape != output_grid.shape

    colors, counts = np.unique(input_grid, return_counts=True)
    assert len(colors) >= 2

    maximum_count = int(counts.max())
    maximum_indices = np.flatnonzero(counts == maximum_count)
    assert len(maximum_indices) == 1

    background_index = int(maximum_indices[0])
    background = int(colors[background_index])
    foreground_colors = tuple(
        int(color) for color in colors if int(color) != background
    )

    foreground_total = int(
        sum(
            np.count_nonzero(input_grid == color)
            for color in foreground_colors
        )
    )
    assert maximum_count > foreground_total
    assert (foreground_total + 4) % 4 == 0

    side = (foreground_total + 4) // 4
    assert 4 <= side <= 8
    assert output_grid.shape == (side, side)

    input_color_set = {
        int(color) for color in np.unique(input_grid)
    }
    output_color_set = {
        int(color) for color in np.unique(output_grid)
    }
    assert input_color_set == output_color_set
    assert 3 <= len(foreground_colors) <= 6
    assert input_grid.shape[0] >= side + 2
    assert input_grid.shape[1] >= side + 2
    assert not _input_is_already_assembled(input_grid, output_grid, background)

    perimeter_mask = np.zeros((side, side), dtype=bool)
    for row, col in _perimeter(side):
        perimeter_mask[row, col] = True

    assert np.all(output_grid[perimeter_mask] != background)
    assert np.all(output_grid[~perimeter_mask] == background)

    shapes = {}
    sizes = {}
    bends = {}

    for color in foreground_colors:
        input_coords = np.argwhere(input_grid == color)
        output_coords = np.argwhere(output_grid == color)

        assert len(input_coords) > 0
        assert len(input_coords) == len(output_coords)
        assert _is_connected(input_coords)
        assert _is_connected(output_coords)

        input_shape = _normalized_shape(input_coords)
        output_shape = _normalized_shape(output_coords)
        assert input_shape == output_shape

        shapes[color] = input_shape
        sizes[color] = len(input_shape)
        bends[color] = _is_bent(input_shape)

    solutions = _solve_shape_assemblies(
        shapes,
        side,
        background,
        limit=2,
    )
    assert len(solutions) == 1
    assert np.array_equal(solutions[0], output_grid)

    return {
        "background": background,
        "foreground": foreground_colors,
        "side": side,
        "piece_count": len(foreground_colors),
        "canvas_shape": tuple(int(x) for x in input_grid.shape),
        "shapes": shapes,
        "sizes": sizes,
        "bends": bends,
    }


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    seen_inputs = set()
    seen_pairs = set()
    train_metadata = []
    test_metadata = []

    for section_name in ("train", "test"):
        for pair in puzzle[section_name]:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {"input", "output"}

            for grid_name in ("input", "output"):
                grid = pair[grid_name]
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0
                assert int(grid.max()) <= 9

            input_key = _grid_key(pair["input"])
            pair_key = _pair_key(pair)
            assert input_key not in seen_inputs
            assert pair_key not in seen_pairs
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

            assert not _is_source_pair(
                pair["input"], pair["output"]
            )

            metadata = _analyze_pair(pair)
            if section_name == "train":
                train_metadata.append(metadata)
            else:
                test_metadata.append(metadata)

    # No cross-pair color convention is needed. Two distinct training
    # geometries demonstrate the local reconstruction rule.
    assert len({(m["side"], m["piece_count"]) for m in train_metadata}) >= 2
    assert all(4 <= m["side"] <= 8 for m in test_metadata)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")

    seed_value = int(seed)
    if seed_value < 0:
        raise ValueError("seed must be a nonnegative integer")

    train_count, test_count = _resolve_counts(
        seed_value,
        num_train,
        num_test,
    )
    rng = np.random.default_rng(seed_value)
    puzzle_parameters = sample_puzzle_parameters(rng)

    train_types = [
        (int(rng.integers(4, 6)), int(rng.integers(3, 5))),
        (int(rng.integers(5, 8)), int(rng.integers(4, 6))),
    ]
    if train_types[1] == train_types[0]:
        train_types[1] = (train_types[1][0], 5)
    for _ in range(2, train_count):
        side = int(rng.integers(4, 9))
        train_types.append((side, int(rng.integers(3, min(6, side + 1) + 1))))

    test_types = []
    for _ in range(test_count):
        if int(rng.integers(0, 3)) == 0:
            test_types.append(train_types[int(rng.integers(0, len(train_types)))])
        else:
            side = int(rng.integers(5, 9))
            test_types.append((side, int(rng.integers(4, min(6, side + 1) + 1))))

    puzzle = {"train": [], "test": []}
    used_inputs = set()
    used_pairs = set()

    for section_name, types in (("train", train_types), ("test", test_types)):
        regime = "source_train" if section_name == "train" else "target_test"
        for archetype in types:
            generated = None
            for _ in range(100):
                pair_seed = int(rng.integers(0, np.iinfo(np.int64).max))
                pair_rng = np.random.default_rng(pair_seed)
                try:
                    candidate = _generate_pair(
                        pair_rng, regime, puzzle_parameters, archetype
                    )
                except RuntimeError:
                    continue

                input_key = _grid_key(candidate["input"])
                pair_key = _pair_key(candidate)
                if (
                    input_key in used_inputs
                    or pair_key in used_pairs
                    or _is_source_pair(candidate["input"], candidate["output"])
                ):
                    continue
                generated = candidate
                used_inputs.add(input_key)
                used_pairs.add(pair_key)
                break
            if generated is None:
                raise RuntimeError("could not generate a distinct pair")
            puzzle[section_name].append(generated)

    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzles_equal(first, second):
    if set(first.keys()) != {"train", "test"}:
        return False
    if set(second.keys()) != {"train", "test"}:
        return False

    for section_name in ("train", "test"):
        if len(first[section_name]) != len(second[section_name]):
            return False

        for first_pair, second_pair in zip(
            first[section_name], second[section_name]
        ):
            for grid_name in ("input", "output"):
                first_grid = first_pair[grid_name]
                second_grid = second_pair[grid_name]
                if first_grid.dtype != second_grid.dtype:
                    return False
                if not np.array_equal(first_grid, second_grid):
                    return False

    return True


def _puzzle_key(puzzle):
    return (
        tuple(_pair_key(pair) for pair in puzzle["train"]),
        tuple(_pair_key(pair) for pair in puzzle["test"]),
    )


def _binary_layout_key(grid, background):
    return (
        int(grid.shape[0]),
        int(grid.shape[1]),
        tuple(
            int(value != background)
            for value in grid.reshape(-1)
        ),
    )


def _expect_value_error(seed, num_train, num_test):
    try:
        generate_puzzle(
            seed,
            num_train=num_train,
            num_test=num_test,
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError was not raised")


def self_test():
    automatic_puzzles = []

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        automatic_puzzles.append(puzzle)

    train_counts = {
        len(puzzle["train"]) for puzzle in automatic_puzzles
    }
    test_counts = {
        len(puzzle["test"]) for puzzle in automatic_puzzles
    }
    assert len(train_counts) >= 4
    assert len(test_counts) >= 3

    deterministic_a = generate_puzzle(
        314159,
        num_train=7,
        num_test=3,
    )
    deterministic_b = generate_puzzle(
        314159,
        num_train=7,
        num_test=3,
    )
    assert _puzzles_equal(deterministic_a, deterministic_b)
    validate_generated_puzzle(deterministic_a)

    explicit = generate_puzzle(
        271828,
        num_train=6,
        num_test=2,
    )
    assert len(explicit["train"]) == 6
    assert len(explicit["test"]) == 2
    validate_generated_puzzle(explicit)

    boundary_cases = (
        (1001, 2, 1),
        (1002, 10, 1),
        (1003, 10, 4),
    )
    for seed, train_count, test_count in boundary_cases:
        puzzle = generate_puzzle(
            seed,
            num_train=train_count,
            num_test=test_count,
        )
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)

    partial_test_one = generate_puzzle(
        2001,
        num_train=None,
        num_test=1,
    )
    assert len(partial_test_one["test"]) == 1
    validate_generated_puzzle(partial_test_one)

    partial_test_four = generate_puzzle(
        2002,
        num_train=None,
        num_test=4,
    )
    assert len(partial_test_four["test"]) == 4
    validate_generated_puzzle(partial_test_four)

    partial_train_two = generate_puzzle(
        2003,
        num_train=2,
        num_test=None,
    )
    assert len(partial_train_two["train"]) == 2
    assert len(partial_train_two["test"]) == 1
    validate_generated_puzzle(partial_train_two)

    partial_train_ten = generate_puzzle(
        2004,
        num_train=10,
        num_test=None,
    )
    assert len(partial_train_ten["train"]) == 10
    validate_generated_puzzle(partial_train_ten)

    illegal_cases = (
        (1, None),
        (11, None),
        (None, 0),
        (None, 5),
        (2, 2),
        (3, 4),
        (4, 4),
    )
    for train_count, test_count in illegal_cases:
        _expect_value_error(
            3000,
            train_count,
            test_count,
        )

    puzzle_keys = set()
    binary_layouts = set()
    shape_profiles = set()
    canvas_shapes = set()
    backgrounds = set()
    sizes = set()
    counts = set()
    corner_cuts = full_sides = singletons = three_sides = 0
    zero_side_cuts = multiple_side_cuts = small_singletons = long_bends = 0
    straight_full_sides = exact_compact = compact_4 = compact_6 = spacious_6 = 0
    compact = rectangular = contacts = separated = interleaved = 0
    shared_type = new_test_type = changed_test_background = 0
    incomplete_color_chain = role_swaps = 0

    for puzzle in automatic_puzzles:
        puzzle_keys.add(_puzzle_key(puzzle))
        train_metadata = [_analyze_pair(pair) for pair in puzzle["train"]]
        test_metadata = [_analyze_pair(pair) for pair in puzzle["test"]]
        train_types = {(m["side"], m["piece_count"]) for m in train_metadata}
        test_types = {(m["side"], m["piece_count"]) for m in test_metadata}
        shared_type += bool(train_types & test_types)
        new_test_type += bool(test_types - train_types)
        changed_test_background += len({m["background"] for m in test_metadata}) > 1
        incomplete_color_chain += any(
            any(m["background"] not in t["foreground"] for t in train_metadata)
            for m in test_metadata
        )
        role_swaps += any(
            m["background"] in t["foreground"]
            for m in train_metadata for t in test_metadata
        )

        for section_name, metadata_list in (("train", train_metadata), ("test", test_metadata)):
            for pair, metadata in zip(puzzle[section_name], metadata_list):
                side = metadata["side"]
                background = metadata["background"]
                output = pair["output"]
                ring = [int(output[r, c]) for r, c in _perimeter(side)]
                cuts = [index for index in range(len(ring)) if ring[index] != ring[index - 1]]
                corner_cuts += any(index % (side - 1) == 0 for index in cuts)
                interior_counts = [
                    sum(edge * (side - 1) < cut < (edge + 1) * (side - 1)
                        for cut in cuts)
                    for edge in range(4)
                ]
                zero_side_cuts += 0 in interior_counts
                multiple_side_cuts += any(value >= 2 for value in interior_counts)
                full_sides += any(
                    np.all(edge == color)
                    for edge in (output[0, :], output[:, -1], output[-1, :], output[:, 0])
                    for color in metadata["foreground"]
                )
                straight_full_sides += any(
                    np.all(edge == color) and metadata["sizes"][color] == side
                    for edge in (output[0, :], output[:, -1], output[-1, :], output[:, 0])
                    for color in metadata["foreground"]
                )
                singletons += 1 in metadata["sizes"].values()
                small_singletons += side == 4 and 1 in metadata["sizes"].values()
                long_bends += any(
                    metadata["bends"][color] and metadata["sizes"][color] >= side + 1
                    for color in metadata["foreground"]
                )
                three_sides += any(
                    sum((np.any(output[0, :] == color), np.any(output[:, -1] == color),
                         np.any(output[-1, :] == color), np.any(output[:, 0] == color))) >= 3
                    for color in metadata["foreground"]
                )
                binary_layouts.add(_binary_layout_key(pair["input"], background))
                shape_profiles.add(tuple(sorted(metadata["shapes"].values())))
                canvas_shapes.add(pair["input"].shape)
                backgrounds.add(background)
                sizes.add(side)
                counts.add(metadata["piece_count"])
                h, w = pair["input"].shape
                compact += h <= side + 4 and w <= side + 4
                exact_compact += (h, w) == (side + 2, side + 2)
                compact_4 += side == 4 and (h, w) == (6, 6)
                compact_6 += side == 6 and (h, w) == (8, 8)
                spacious_6 += side == 6 and (h, w) == (16, 16)
                rectangular += h != w
                interleaved += _has_interleaved_boxes(pair["input"], metadata["foreground"])
                touches = _has_cross_color_contact(pair["input"], background)
                contacts += touches
                separated += not touches

    assert len(puzzle_keys) == 42
    assert len(binary_layouts) >= 100
    assert len(shape_profiles) >= 30
    assert len(canvas_shapes) >= 25
    assert len(backgrounds) == 10
    assert sizes == {4, 5, 6, 7, 8}
    assert counts == {3, 4, 5, 6}
    assert corner_cuts >= 10 and full_sides >= 10
    assert singletons >= 10 and three_sides >= 10
    assert zero_side_cuts >= 10 and multiple_side_cuts >= 10
    assert small_singletons >= 3 and long_bends >= 10
    assert straight_full_sides >= 5
    assert compact >= 10 and rectangular >= 30
    assert exact_compact >= 5 and compact_4 >= 1 and compact_6 >= 1 and spacious_6 >= 1
    assert contacts >= 10 and separated >= 10
    assert interleaved >= 10
    assert shared_type >= 3 and new_test_type >= 3
    assert changed_test_background >= 3
    assert incomplete_color_chain >= 3 and role_swaps >= 3
