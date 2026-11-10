import numpy as np


_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 1), (2, 2)),
)

_SOURCE_ROWS = (
    (
        (
            "4777477776",
            "4774447776",
            "4777777776",
            "4777767776",
            "4777466676",
            "4777477776",
            "4776677776",
            "4746777676",
            "4744776676",
            "4744477776",
        ),
        (
            "4747777776",
            "4444777776",
            "4777777776",
            "4777776776",
            "4477776666",
            "4477777776",
            "4777777666",
            "4477777666",
            "4447777666",
            "4444777776",
        ),
    ),
    (
        (
            "977778878",
            "979977878",
            "979797778",
            "977777998",
            "977777978",
            "977778778",
            "977788878",
            "978778778",
            "977777778",
        ),
        (
            "977777888",
            "999777788",
            "997977778",
            "999777778",
            "997777778",
            "977777878",
            "977778888",
            "977778878",
            "977777778",
        ),
    ),
    (
        (
            "877777771",
            "871177771",
            "871777771",
            "877771171",
            "878711171",
            "878877781",
            "878871171",
            "878711171",
            "878771171",
        ),
        (
            "877777771",
            "877777111",
            "877777171",
            "877777111",
            "887771111",
            "888877771",
            "888777111",
            "887771111",
            "887777111",
        ),
    ),
)


def _rows_to_grid(rows):
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int64,
    )


_SOURCE_PAIRS = tuple(
    (_rows_to_grid(input_rows), _rows_to_grid(output_rows))
    for input_rows, output_rows in _SOURCE_ROWS
)


def _normalize_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    num_train = _normalize_count(num_train, "num_train")
    num_test = _normalize_count(num_test, "num_test")

    if num_train is not None and not 2 <= num_train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if num_test is not None and not 1 <= num_test <= 4:
        raise ValueError("num_test must be between 1 and 4")

    marker = int(seed)

    if num_train is None and num_test is None:
        num_test = 1 + marker % 4
        minimum_train = max(2, num_test + 1)
        num_train = (
            minimum_train
            + (marker // 4) % (11 - minimum_train)
        )
    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = minimum_train + marker % (11 - minimum_train)
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = 1 + marker % maximum_test

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _array_key(array):
    return (
        tuple(array.shape),
        tuple(int(value) for value in array.ravel()),
    )


def _inspect_rails(grid):
    vertical = bool(
        np.all(grid[:, 0] == grid[0, 0])
        and np.all(grid[:, -1] == grid[0, -1])
        and int(grid[0, 0]) != int(grid[0, -1])
    )
    horizontal = bool(
        np.all(grid[0, :] == grid[0, 0])
        and np.all(grid[-1, :] == grid[-1, 0])
        and int(grid[0, 0]) != int(grid[-1, 0])
    )

    assert vertical != horizontal, (
        "there must be exactly one pair of opposite rails"
    )

    if vertical:
        canonical = grid
        orientation = "vertical"
    else:
        canonical = grid.T
        orientation = "horizontal"

    left_color = int(canonical[0, 0])
    right_color = int(canonical[0, -1])
    values = {int(value) for value in np.unique(canonical)}

    assert len(values) == 3, (
        "a pair must use exactly two rail colors and one background"
    )
    assert left_color in values
    assert right_color in values

    background_values = values - {left_color, right_color}
    assert len(background_values) == 1
    background = next(iter(background_values))

    interior_values = {
        int(value) for value in np.unique(canonical[:, 1:-1])
    }
    assert interior_values == values, (
        "both rail identities and background must occur inside the rails"
    )

    return (
        canonical,
        orientation,
        background,
        left_color,
        right_color,
    )


def _components(canonical, color):
    height, width = canonical.shape

    remaining = {
        (row, column)
        for row in range(height)
        for column in range(1, width - 1)
        if int(canonical[row, column]) == color
    }

    result = []

    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack = [start]
        component = {start}

        while stack:
            row, column = stack.pop()

            for row_delta in (-1, 0, 1):
                for column_delta in (-1, 0, 1):
                    if row_delta == 0 and column_delta == 0:
                        continue

                    neighbor = (
                        row + row_delta,
                        column + column_delta,
                    )

                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        component.add(neighbor)
                        stack.append(neighbor)

        result.append(frozenset(component))

    result.sort(key=lambda piece: min(piece))
    return result


def _settle(pieces, direction, width):
    if direction == -1:
        ordered = sorted(
            pieces,
            key=lambda piece: min(column for _, column in piece),
        )
    else:
        ordered = sorted(
            pieces,
            key=lambda piece: -max(column for _, column in piece),
        )

    occupied = set()
    records = []

    for piece in ordered:
        current = set(piece)
        shift = 0

        while True:
            candidate = {
                (row, column + direction)
                for row, column in current
            }

            hits_rail = any(
                column <= 0 or column >= width - 1
                for _, column in candidate
            )
            hits_piece = bool(candidate & occupied)

            if hits_rail or hits_piece:
                blocked = hits_piece
                break

            current = candidate
            shift += 1
            assert shift < width

        final = frozenset(current)
        occupied.update(final)

        records.append(
            {
                "piece": piece,
                "final": final,
                "shift": shift,
                "blocked": blocked,
                "direction": direction,
            }
        )

    return frozenset(occupied), records


def _swept_cells(records):
    swept = set()

    for record in records:
        direction = record["direction"]

        for amount in range(record["shift"] + 1):
            swept.update(
                (
                    row,
                    column + direction * amount,
                )
                for row, column in record["piece"]
            )

    return swept


def _analyze_input(input_grid):
    (
        canonical,
        orientation,
        background,
        left_color,
        right_color,
    ) = _inspect_rails(input_grid)

    _, width = canonical.shape
    assert width >= 5

    left_pieces = _components(canonical, left_color)
    right_pieces = _components(canonical, right_color)

    assert left_pieces and right_pieces, (
        "both rail colors need at least one interior piece"
    )

    for pieces, direction in ((left_pieces, -1), (right_pieces, 1)):
        for piece in pieces:
            assert all(
                (column >= 2 if direction == -1 else column <= width - 3)
                for _, column in piece
            ), "pieces must be separated from their matching rail"
        for index, first in enumerate(pieces):
            first_lead = (min if direction == -1 else max)(
                column for _, column in first
            )
            for second in pieces[index + 1:]:
                second_lead = (min if direction == -1 else max)(
                    column for _, column in second
                )
                if first_lead == second_lead:
                    assert {row for row, _ in first}.isdisjoint(
                        {row for row, _ in second}
                    ), "tied pieces must have independent lanes"

    left_final, left_records = _settle(
        left_pieces,
        direction=-1,
        width=width,
    )
    right_final, right_records = _settle(
        right_pieces,
        direction=1,
        width=width,
    )

    assert all(record["shift"] > 0 for record in left_records + right_records)
    assert not (left_final & right_final), (
        "opposite colors may not collide in the result"
    )
    assert not (
        _swept_cells(left_records)
        & _swept_cells(right_records)
    ), "oppositely moving pieces must have disjoint swept paths"

    output_canonical = np.full(
        canonical.shape,
        background,
        dtype=np.int64,
    )
    output_canonical[:, 0] = left_color
    output_canonical[:, -1] = right_color

    for row, column in left_final:
        output_canonical[row, column] = left_color

    for row, column in right_final:
        output_canonical[row, column] = right_color

    if orientation == "vertical":
        expected_output = output_canonical
    else:
        expected_output = output_canonical.T.copy()

    return {
        "orientation": orientation,
        "background": background,
        "left_color": left_color,
        "right_color": right_color,
        "left_records": left_records,
        "right_records": right_records,
        "expected_output": expected_output,
    }


def _normalized_shape(piece):
    piece = tuple(piece)
    top = min(row for row, _ in piece)
    left = min(column for _, column in piece)
    return tuple(sorted((row - top, column - left) for row, column in piece))


_SOURCE_SHAPES = tuple(sorted({
    _normalized_shape(piece)
    for grid, _ in _SOURCE_PAIRS
    for color in (int(grid[0, 0]), int(grid[0, -1]))
    for piece in _components(grid, color)
}))


def _sample_shape(rng, max_width, max_height, small=False):
    sizes = (1, 2, 2, 3, 3, 4, 4, 5) if small else (
        1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8
    )
    for _ in range(40):
        if rng.random() < 0.38:
            shape = _SOURCE_SHAPES[int(rng.integers(len(_SOURCE_SHAPES)))]
        else:
            target = int(sizes[int(rng.integers(len(sizes)))])
            cells = {(0, 0)}
            while len(cells) < target:
                row, column = tuple(cells)[int(rng.integers(len(cells)))]
                delta_row = int(rng.integers(-1, 2))
                delta_col = int(rng.integers(-1, 2))
                if delta_row or delta_col:
                    cells.add((row + delta_row, column + delta_col))
            shape = _normalized_shape(cells)
        if rng.random() < 0.5:
            shape = _normalized_shape((column, row) for row, column in shape)
        if rng.random() < 0.5:
            shape = _normalized_shape((row, -column) for row, column in shape)
        if rng.random() < 0.5:
            shape = _normalized_shape((-row, column) for row, column in shape)
        shape_height = 1 + max(row for row, _ in shape)
        shape_width = 1 + max(column for _, column in shape)
        if shape_width <= max_width and shape_height <= max_height:
            return shape
    return ((0, 0),)


def _make_piece(shape, top, distance, direction, width):
    if direction == -1:
        return frozenset((top + row, distance + column) for row, column in shape)
    return frozenset((top + row, width - 1 - distance - column) for row, column in shape)


def _separate(first, second):
    return all(
        max(abs(row - other_row), abs(column - other_column)) > 1
        for row, column in first
        for other_row, other_column in second
    )


def _can_add(piece, same_pieces, other_pieces, height, width, direction):
    if not all(0 <= row < height and 1 <= column < width - 1 for row, column in piece):
        return False
    if direction == -1 and min(column for _, column in piece) < 2:
        return False
    if direction == 1 and max(column for _, column in piece) > width - 3:
        return False
    if any(not _separate(piece, old) for old in same_pieces):
        return False
    if any(piece & old for old in other_pieces):
        return False
    return True


def _place_side(rng, height, width, direction, count, collision, other_pieces):
    pieces = []
    max_distance = (width - 3 if rng.random() < 0.35
                    else min(width - 3, width // 2 + 1))
    if collision:
        chain_count = 2 if count == 4 and height >= 11 and rng.random() < 0.55 else 1
        for _chain in range(chain_count):
            for _ in range(70):
                near = _sample_shape(rng, min(2, max_distance - 2), min(4, height), True)
                far = _sample_shape(rng, min(2, max_distance - 2), min(4, height), True)
                near_width = 1 + max(column for _, column in near)
                far_width = 1 + max(column for _, column in far)
                near_distance = int(rng.integers(2, min(4, max_distance) + 1))
                far_min = near_distance + near_width + 1
                far_max = max_distance - far_width + 1
                if far_min > far_max:
                    continue
                far_distance = int(rng.integers(far_min, far_max + 1))
                near_height = 1 + max(row for row, _ in near)
                far_height = 1 + max(row for row, _ in far)
                common_row = int(rng.integers(height))
                near_rows = sorted({row for row, _ in near})
                far_rows = sorted({row for row, _ in far})
                near_top = common_row - near_rows[int(rng.integers(len(near_rows)))]
                far_top = common_row - far_rows[int(rng.integers(len(far_rows)))]
                if not (0 <= near_top <= height - near_height and 0 <= far_top <= height - far_height):
                    continue
                near_piece = _make_piece(near, near_top, near_distance, direction, width)
                far_piece = _make_piece(far, far_top, far_distance, direction, width)
                if (_can_add(near_piece, pieces, other_pieces, height, width, direction)
                        and _can_add(far_piece, pieces + [near_piece], other_pieces,
                                     height, width, direction)):
                    pieces.extend((near_piece, far_piece))
                    break
            else:
                return None
    for _ in range(count - len(pieces)):
        for _attempt in range(90):
            max_shape_width = min(4, max_distance - 1)
            shape = _sample_shape(rng, max_shape_width, min(5, height))
            shape_height = 1 + max(row for row, _ in shape)
            shape_width = 1 + max(column for _, column in shape)
            distance_max = max_distance - shape_width + 1
            if distance_max < 2:
                continue
            top = int(rng.integers(height - shape_height + 1))
            distance = int(rng.integers(2, distance_max + 1))
            piece = _make_piece(shape, top, distance, direction, width)
            if _can_add(piece, pieces, other_pieces, height, width, direction):
                pieces.append(piece)
                break
        else:
            return None
    return pieces


def _dimensions(rng, regime):
    size_mode = int(rng.choice(3, p=(0.34, 0.48, 0.18)))
    if size_mode == 0:
        return int(rng.integers(9, 12)), int(rng.integers(9, 12))
    if size_mode == 1:
        return int(rng.integers(9, 19)), int(rng.integers(10, 21))
    return int(rng.integers(14, 27)), int(rng.integers(15, 28))


def _pair_roles(rng, palette, fixed):
    if fixed:
        return tuple(palette[:3])
    selected = [int(x) for x in rng.choice(palette, size=3, replace=False)]
    return tuple(selected)


def _generate_pair(rng, regime, roles, orientation, collision_sides):
    background, left_color, right_color = roles
    for attempt in range(240):
        height, width = _dimensions(rng, regime)
        if attempt >= 160:
            height = max(height, 12)
            width = max(width, 13)
        counts = [int(rng.choice((1, 2, 2, 3, 3, 4))) for _ in range(2)]
        for index in collision_sides:
            counts[index] = max(2, counts[index])
        order = [0, 1] if rng.random() < 0.5 else [1, 0]
        placed = [None, None]
        for index in order:
            direction = -1 if index == 0 else 1
            other = placed[1 - index] or ()
            placed[index] = _place_side(
                rng, height, width, direction, counts[index],
                index in collision_sides, other,
            )
            if placed[index] is None:
                break
        if placed[0] is None or placed[1] is None:
            continue
        canonical = np.full((height, width), background, dtype=np.int64)
        canonical[:, 0] = left_color
        canonical[:, -1] = right_color
        for index, color in ((0, left_color), (1, right_color)):
            for piece in placed[index]:
                for row, column in piece:
                    canonical[row, column] = color
        input_grid = canonical if orientation == "vertical" else canonical.T.copy()
        try:
            info = _analyze_input(input_grid)
        except AssertionError:
            continue
        if len(info["left_records"]) != counts[0] or len(info["right_records"]) != counts[1]:
            continue
        if any(not any(record["blocked"] for record in info[("left_records", "right_records")[index]])
               for index in collision_sides):
            continue
        if not ({row for piece in placed[0] for row, _ in piece}
                & {row for piece in placed[1] for row, _ in piece}):
            continue
        if np.array_equal(input_grid, info["expected_output"]):
            continue
        return input_grid, info["expected_output"]
    raise AssertionError("could not construct an unambiguous pair")


def _is_source_pair(input_grid, output_grid):
    return any(
        input_grid.shape == source_input.shape
        and np.array_equal(input_grid, source_input)
        and np.array_equal(output_grid, source_output)
        for source_input, source_output in _SOURCE_PAIRS
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = [int(x) for x in rng.choice(10, size=int(rng.integers(4, 7)), replace=False)]
    train_palette, novel_color = palette[:-1], palette[-1]
    fixed_palette = rng.random() < 0.20
    fixed_orientation = rng.random() < 0.25
    base_orientation = ("vertical" if rng.random() < 0.5 else "horizontal")
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            roles = _pair_roles(pair_rng, train_palette, fixed_palette)
            if split == "test" and not fixed_palette and pair_rng.random() < 0.48:
                roles = list(roles)
                roles[int(pair_rng.integers(1, 3))] = novel_color
                roles = tuple(roles)
            orientation = (base_orientation if fixed_orientation else
                           ("vertical" if pair_rng.random() < 0.5 else "horizontal"))
            if split == "train" and index < 2:
                collision_sides = {index}
            elif split == "test":
                mode = int(pair_rng.choice(3, p=(0.35, 0.35, 0.30)))
                collision_sides = ({0}, {1}, {0, 1})[mode]
            else:
                mode = int(pair_rng.choice(4, p=(0.18, 0.27, 0.27, 0.28)))
                collision_sides = (set(), {0}, {1}, {0, 1})[mode]
            for _ in range(12):
                input_grid, output_grid = _generate_pair(
                    pair_rng, "source_train" if split == "train" else "target_test",
                    roles, orientation, collision_sides,
                )
                if _array_key(input_grid) not in seen_inputs and not _is_source_pair(input_grid, output_grid):
                    break
            else:
                raise AssertionError("could not construct a distinct pair")
            seen_inputs.add(_array_key(input_grid))
            puzzle[split].append({"input": input_grid, "output": output_grid})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen_inputs = set()
    train_collisions = [False, False]
    all_infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in (pair["input"], pair["output"]):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            input_grid, output_grid = pair["input"], pair["output"]
            assert input_grid.shape == output_grid.shape
            key = _array_key(input_grid)
            assert key not in seen_inputs
            seen_inputs.add(key)
            assert not _is_source_pair(input_grid, output_grid)
            info = _analyze_input(input_grid)
            assert np.array_equal(output_grid, info["expected_output"])
            assert not np.array_equal(input_grid, output_grid)
            assert set(np.unique(input_grid)) == set(np.unique(output_grid))
            left_rows = {row for record in info["left_records"] for row, _ in record["piece"]}
            right_rows = {row for record in info["right_records"] for row, _ in record["piece"]}
            assert left_rows & right_rows
            for side, records in enumerate((info["left_records"], info["right_records"])):
                if split == "train" and any(record["blocked"] for record in records):
                    train_collisions[side] = True
                assert all(record["shift"] >= 1 for record in records)
            if split == "test":
                assert any(record["blocked"] for records in
                           (info["left_records"], info["right_records"])
                           for record in records)
            all_infos[split].append(info)
    assert all(train_collisions), "training must show packing toward both rail roles"
    assert any(len(record["piece"]) > 1 for info in all_infos["train"]
               for record in info["left_records"])
    assert any(len(record["piece"]) > 1 for info in all_infos["train"]
               for record in info["right_records"])


def _puzzle_key(puzzle):
    return tuple((split, tuple((_array_key(pair["input"]), _array_key(pair["output"]))
                               for pair in puzzle[split])) for split in ("train", "test"))


def _chain_count(records, direction):
    parents = list(range(len(records)))
    for index, record in enumerate(records):
        if not record["blocked"]:
            continue
        ahead = {(row, column + direction) for row, column in record["final"]}
        for earlier in range(index):
            if ahead & records[earlier]["final"]:
                ancestor = earlier
                while parents[ancestor] != ancestor:
                    ancestor = parents[ancestor]
                parents[index] = ancestor
                break
    groups = {}
    for index in range(len(records)):
        ancestor = index
        while parents[ancestor] != ancestor:
            ancestor = parents[ancestor]
        groups[ancestor] = groups.get(ancestor, 0) + 1
    return sum(size > 1 for size in groups.values())


def self_test() -> None:
    for source_input, source_output in _SOURCE_PAIRS:
        assert np.array_equal(_analyze_input(source_input)["expected_output"],
                              source_output)
    automatic = generate_puzzle(7)
    validate_generated_puzzle(automatic)
    explicit = generate_puzzle(123, 7, 3)
    assert _puzzle_key(explicit) == _puzzle_key(generate_puzzle(123, 7, 3))
    assert _puzzle_key(explicit) != _puzzle_key(generate_puzzle(124, 7, 3))
    for seed, train, test in ((200, 2, 1), (201, 10, 1), (202, 10, 4),
                              (300, None, 1), (301, None, 4),
                              (302, 2, None), (303, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
    for arguments in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                      {"num_test": 5}, {"num_train": 2, "num_test": 2},
                      {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(0, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzle_keys, grid_keys = set(), set()
    train_counts, test_counts = set(), set()
    dimensions, shapes, palettes, orientations = set(), set(), set(), set()
    count_pairs, collision_sides, short_moves, long_moves = set(), set(), 0, 0
    overlap_rows, tied_leads = 0, 0
    compact, large, large_pieces, two_chains = 0, 0, 0, 0
    palette_changes, orientation_changes, count_changes = 0, 0, 0
    novel_test_colors, reused_roles, swapped_roles = 0, 0, 0
    source_shapes, five_row_shapes, cross_midpoint = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzle_keys.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzle_palettes, puzzle_orientations, puzzle_counts = set(), set(), set()
        split_infos = {"train": [], "test": []}
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid_keys.add(_array_key(pair["input"]))
                dimensions.add(pair["input"].shape)
                info = _analyze_input(pair["input"])
                split_infos[split].append(info)
                palettes.add((info["background"], info["left_color"], info["right_color"]))
                orientations.add(info["orientation"])
                puzzle_palettes.add((info["background"], info["left_color"], info["right_color"]))
                puzzle_orientations.add(info["orientation"])
                records = info["left_records"] + info["right_records"]
                counts = (len(info["left_records"]), len(info["right_records"]))
                count_pairs.add(counts)
                puzzle_counts.add(counts)
                collision_sides.add(tuple(any(record["blocked"] for record in side)
                                          for side in (info["left_records"], info["right_records"])))
                short_moves += sum(record["shift"] == 1 for record in records)
                long_moves += sum(record["shift"] >= 4 for record in records)
                for record in records:
                    shape = _normalized_shape(record["piece"])
                    shapes.add(shape)
                    large_pieces += len(record["piece"]) >= 6
                    source_shapes += shape in _SOURCE_SHAPES
                    five_row_shapes += max(row for row, _ in shape) >= 4
                canonical = pair["input"] if info["orientation"] == "vertical" else pair["input"].T
                middle = canonical.shape[1] // 2
                cross_midpoint += sum(
                    any(column > middle for _, column in record["piece"])
                    for record in info["left_records"]
                )
                cross_midpoint += sum(
                    any(column < middle for _, column in record["piece"])
                    for record in info["right_records"]
                )
                compact += canonical.shape in ((9, 9), (10, 10))
                large += max(canonical.shape) >= 20
                two_chains += any(_chain_count(side, direction) >= 2
                                  for side, direction in ((info["left_records"], -1),
                                                          (info["right_records"], 1)))
                overlap_rows += sum(left in row[1:-1] and right in row[1:-1]
                                    for row in canonical for left, right in [(info["left_color"], info["right_color"])])
                for side, direction in ((info["left_records"], -1), (info["right_records"], 1)):
                    leads = [(min if direction == -1 else max)(column for _, column in rec["piece"])
                             for rec in side]
                    tied_leads += len(leads) - len(set(leads))
        palette_changes += len(puzzle_palettes) > 1
        orientation_changes += len(puzzle_orientations) > 1
        count_changes += len(puzzle_counts) > 1
        training_colors = {
            color for info in split_infos["train"]
            for color in (info["background"], info["left_color"], info["right_color"])
        }
        train_roles = {(info["left_color"], info["right_color"])
                       for info in split_infos["train"]}
        novel_test_colors += any(
            info["left_color"] not in training_colors
            or info["right_color"] not in training_colors
            for info in split_infos["test"]
        )
        reused_roles += any((info["left_color"], info["right_color"]) in train_roles
                            for info in split_infos["test"])
        swapped_roles += any((info["right_color"], info["left_color"]) in train_roles
                             for info in split_infos["test"])
    assert len(puzzle_keys) == 42 and len(grid_keys) >= 100
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(dimensions) >= 20 and len(shapes) >= 20 and len(palettes) >= 20
    assert orientations == {"vertical", "horizontal"}
    assert len(count_pairs) >= 10 and len(collision_sides) >= 3
    assert short_moves >= 10 and long_moves >= 10 and overlap_rows >= 40
    assert tied_leads >= 1
    assert compact >= 5 and large >= 10 and large_pieces >= 20
    assert two_chains >= 2
    assert palette_changes >= 5 and orientation_changes >= 5 and count_changes >= 20
    assert novel_test_colors >= 5 and reused_roles >= 5 and swapped_roles >= 5
    assert source_shapes >= 20 and five_row_shapes >= 1 and cross_midpoint >= 20
