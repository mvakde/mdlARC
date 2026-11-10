import numpy as np


_LABEL_POSITIONS = (
    (0, 0, 1),
    (0, 1, 7),
    (0, 2, 6),
    (1, 0, 4),
    (1, 2, 5),
    (2, 0, 2),
    (2, 1, 9),
    (2, 2, 3),
)
_LABEL_SET = frozenset(label for _, _, label in _LABEL_POSITIONS)


_SOURCE_INPUT_TEXTS = frozenset(
    "\n".join(rows)
    for rows in (
        (
            "888888888888",
            "800000888888",
            "808800888888",
            "800000888888",
            "808880888888",
            "800000888888",
            "888888888888",
            "888800000888",
            "888800000888",
            "888800800888",
            "888800000888",
            "888800000888",
        ),
        (
            "888888888",
            "888888888",
            "880000088",
            "880808088",
            "880000088",
            "880808088",
            "880000088",
            "888888888",
            "888888888",
        ),
        (
            "8888888800000",
            "8888888800880",
            "8888888800880",
            "8888888800880",
            "8888888800000",
            "8888888888888",
            "8888888000008",
            "0000088088808",
            "0888088000808",
            "0088088080808",
            "0008088000008",
            "0000088888888",
            "8888888888888",
        ),
        (
            "8888888888800000",
            "8888888888808000",
            "8888888888808800",
            "8888888888808000",
            "8888888888800000",
            "8888800000888888",
            "8888808880888888",
            "8888808080800000",
            "8888800000800080",
            "8888800000808080",
            "8888888888808000",
            "8000008888800000",
            "8008808888888888",
            "8080808888888888",
            "8088008888888888",
            "8000008888888888",
        ),
    )
)


def _is_integer(value) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _validate_count(name: str, value, low: int, high: int):
    if value is None:
        return None
    if not _is_integer(value):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_count("num_train", num_train, 2, 10)
    test_count = _validate_count("num_test", num_test, 1, 4)

    if train_count is None and test_count is None:
        test_count = 1 + (seed % 4)
        minimum_train = test_count + 1
        number_of_choices = 11 - minimum_train
        train_count = minimum_train + ((seed // 4) % number_of_choices)
    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        number_of_choices = 11 - minimum_train
        train_count = minimum_train + (seed % number_of_choices)
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal num_test exists for num_train")
        test_count = 1 + (seed % maximum_test)

    if not (2 <= train_count <= 10):
        raise ValueError("num_train must be between 2 and 10")
    if not (1 <= test_count <= 4):
        raise ValueError("num_test must be between 1 and 4")
    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    if int(rng.integers(0, 2)) == 0:
        structural_color, field_color = 8, 0
    else:
        structural_color, field_color = 0, 8

    return {
        "structural_color": structural_color,
        "field_color": field_color,
    }


def _grid_text(grid: np.ndarray) -> str:
    return "\n".join(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _array_key(grid: np.ndarray) -> tuple:
    normalized = grid.astype(np.int8, copy=False)
    return tuple(grid.shape), normalized.tobytes()


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        (
            split,
            tuple(
                (
                    _array_key(pair["input"]),
                    _array_key(pair["output"]),
                )
                for pair in puzzle[split]
            ),
        )
        for split in ("train", "test")
    )


def _is_field_border(block: np.ndarray, field_color: int) -> bool:
    return bool(
        np.all(block[0, :] == field_color)
        and np.all(block[4, :] == field_color)
        and np.all(block[1:4, 0] == field_color)
        and np.all(block[1:4, 4] == field_color)
    )


def _find_boards(
    grid: np.ndarray,
    structural_color: int,
    field_color: int,
) -> list[tuple[int, int]]:
    height, width = grid.shape
    boards = []

    for row in range(height - 4):
        for column in range(width - 4):
            block = grid[row:row + 5, column:column + 5]
            if not _is_field_border(block, field_color):
                continue

            core = block[1:4, 1:4]
            if np.all(
                (core == structural_color) | (core == field_color)
            ):
                boards.append((row, column))

    return boards


def _transform_grid(
    input_grid: np.ndarray,
    structural_color: int,
    field_color: int,
) -> np.ndarray:
    output_grid = input_grid.copy()
    boards = _find_boards(
        input_grid,
        structural_color,
        field_color,
    )

    for top, left in boards:
        for core_row, core_column, label in _LABEL_POSITIONS:
            row = top + 1 + core_row
            column = left + 1 + core_column
            if input_grid[row, column] == field_color:
                output_grid[row, column] = label

    return output_grid


def _boards_are_separated(first: tuple[int, int], second: tuple[int, int]) -> bool:
    row, column = first
    other_row, other_column = second
    return bool(
        row + 5 < other_row or other_row + 5 < row
        or column + 5 < other_column or other_column + 5 < column
    )


def _touches_edge(position: tuple[int, int], height: int, width: int) -> bool:
    row, column = position
    return row in (0, height - 5) or column in (0, width - 5)


def _sample_count(rng: np.random.Generator, regime: str) -> int:
    if regime == "source_train":
        weights = (25, 30, 25, 15, 5)
    else:
        weights = (10, 15, 20, 30, 15, 10)
    return int(rng.choice(len(weights), p=np.array(weights) / sum(weights))) + 1


def _sample_side(
    rng: np.random.Generator, minimum: int, regime: str
) -> int:
    weights = (0.30, 0.32, 0.38) if regime == "source_train" else (0.50, 0.27, 0.23)
    band = int(rng.choice(3, p=weights))
    if band == 0:
        low, high = minimum, min(30, minimum + (5 if regime == "source_train" else 3))
    elif band == 1:
        low, high = min(30, minimum + 4), min(30, minimum + 14)
    else:
        low, high = max(minimum, 21), 30
    return int(rng.integers(low, high + 1))


def _sample_positions(
    rng: np.random.Generator,
    height: int,
    width: int,
    board_count: int,
    style: str,
) -> list[tuple[int, int]]:
    candidates = [
        (row, column)
        for row in range(height - 4)
        for column in range(width - 4)
        if style != "interior" or (
            0 < row < height - 5 and 0 < column < width - 5
        )
    ]
    if style == "corner":
        first_choices = [
            position for position in candidates
            if position[0] in (0, height - 5)
            and position[1] in (0, width - 5)
        ]
    elif style == "edge":
        first_choices = [
            position for position in candidates
            if _touches_edge(position, height, width)
        ]
    else:
        first_choices = candidates

    if board_count == 1:
        return [first_choices[int(rng.integers(len(first_choices)))]]

    for _ in range(96):
        first = first_choices[int(rng.integers(len(first_choices)))]
        selected = [first]
        for index in rng.permutation(len(candidates)):
            position = candidates[int(index)]
            if position != first and all(
                _boards_are_separated(position, other)
                for other in selected
            ):
                selected.append(position)
                if len(selected) == board_count:
                    return sorted(selected)
    # A deterministic packing handles unusually unlucky compact proposals.
    first_coordinate = 1 if style == "interior" else 0
    row_slots = int(np.ceil(np.sqrt(board_count)))
    column_slots = int(np.ceil(board_count / row_slots))
    positions = [
        (first_coordinate + 6 * row, first_coordinate + 6 * column)
        for row in range(row_slots)
        for column in range(column_slots)
    ][:board_count]
    assert all(row + 5 <= height and column + 5 <= width for row, column in positions)
    return positions


def _sample_mask(rng: np.random.Generator) -> int:
    category = int(rng.choice(6, p=(0.47, 0.16, 0.12, 0.10, 0.08, 0.07)))
    if category == 0:
        count = int(rng.integers(0, 9))
        indices = rng.permutation(8)[:count]
    elif category == 1:
        groups = ((0, 1, 2), (5, 6, 7), (0, 3, 5),
                  (2, 4, 7), (1, 6), (3, 4))
        indices = groups[int(rng.integers(len(groups)))]
    elif category == 2:
        corners = (0, 2, 5, 7)
        count = int(rng.integers(1, 5))
        indices = rng.permutation(corners)[:count]
    elif category == 3:
        indices = (0, 2, 5, 7) if int(rng.integers(2)) else (1, 3, 4, 6)
    elif category == 4:
        ring = (0, 1, 2, 4, 7, 6, 5, 3)
        start = int(rng.integers(8))
        length = int(rng.integers(2, 7))
        indices = [ring[(start + offset) % 8] for offset in range(length)]
    else:
        return 255 if int(rng.integers(2)) else 0
    mask = 0
    for index in indices:
        mask |= 1 << int(index)
    return mask


def _make_plan(rng: np.random.Generator, regime: str, board_count: int) -> dict:
    style_names = ("free", "edge", "interior", "corner")
    probabilities = (0.43, 0.25, 0.25, 0.07) if regime == "source_train" else (0.40, 0.25, 0.26, 0.09)
    style = style_names[int(rng.choice(4, p=probabilities))]
    minimum = {1: 8, 2: 11, 3: 13, 4: 16, 5: 19, 6: 20}[board_count]
    if style == "interior" and board_count <= 3:
        minimum += 2
    height = _sample_side(rng, minimum, regime)
    width = _sample_side(rng, minimum, regime)
    positions = _sample_positions(rng, height, width, board_count, style)
    boards = [
        {
            "position": position,
            "mask": _sample_mask(rng),
            "center_field": bool(rng.integers(2)),
        }
        for position in positions
    ]
    return {"height": height, "width": width, "boards": boards, "style": style}


def _write_board(
    grid: np.ndarray,
    top: int,
    left: int,
    mask: int,
    center_field: bool,
    structural_color: int,
    field_color: int,
) -> None:
    grid[top:top + 5, left:left + 5] = field_color
    for index, (core_row, core_column, _) in enumerate(_LABEL_POSITIONS):
        if mask & (1 << index):
            grid[top + 1 + core_row, left + 1 + core_column] = structural_color
    grid[top + 2, left + 2] = field_color if center_field else structural_color


def _realize_plan(plan: dict, structural_color: int) -> tuple[np.ndarray, np.ndarray]:
    field_color = 8 - structural_color
    input_grid = np.full(
        (plan["height"], plan["width"]), structural_color, dtype=np.int8
    )
    for board in plan["boards"]:
        top, left = board["position"]
        _write_board(
            input_grid, top, left, board["mask"], board["center_field"],
            structural_color, field_color,
        )
    output_grid = _transform_grid(input_grid, structural_color, field_color)
    assert len(_find_boards(input_grid, structural_color, field_color)) == len(plan["boards"])
    return input_grid, output_grid


def _make_puzzle(rng: np.random.Generator, train_count: int, test_count: int) -> dict:
    parameters = sample_puzzle_parameters(rng)
    base_structural = parameters["structural_color"]
    mixed_roles = bool(rng.integers(0, 4) == 0)
    train_counts = [_sample_count(rng, "source_train") for _ in range(train_count)]
    if len(set(train_counts)) == 1:
        index = int(rng.integers(train_count))
        train_counts[index] = 2 if train_counts[index] == 1 else 1
    test_counts = [_sample_count(rng, "target_test") for _ in range(test_count)]
    train_plans = [_make_plan(rng, "source_train", count) for count in train_counts]
    test_plans = [_make_plan(rng, "target_test", count) for count in test_counts]

    # The order of the edge and interior demonstrations varies by puzzle.
    edge_index, interior_index = rng.permutation(train_count)[:2]
    for index, style in ((int(edge_index), "edge"), (int(interior_index), "interior")):
        if train_plans[index]["style"] != style:
            plan = train_plans[index]
            minimum = {1: 8, 2: 11, 3: 13, 4: 16, 5: 19}[train_counts[index]]
            if style == "interior" and train_counts[index] <= 3:
                minimum += 2
            plan["height"] = max(plan["height"], minimum)
            plan["width"] = max(plan["width"], minimum)
            plan["style"] = style
            plan["boards"] = [
                {"position": position, "mask": _sample_mask(rng),
                 "center_field": bool(rng.integers(2))}
                for position in _sample_positions(
                    rng, plan["height"], plan["width"], train_counts[index], style
                )
            ]

    references = [
        (pair_index, board_index)
        for pair_index, plan in enumerate(train_plans)
        for board_index in range(len(plan["boards"]))
    ]
    order = rng.permutation(len(references))
    first_ref = references[int(order[0])]
    second_ref = references[int(order[1])]
    train_plans[first_ref[0]]["boards"][first_ref[1]]["center_field"] = True
    train_plans[second_ref[0]]["boards"][second_ref[1]]["center_field"] = False

    teaching_mode = int(rng.choice(3, p=(0.28, 0.35, 0.37)))
    if teaching_mode < 2:
        pair_index = 0 if teaching_mode == 0 else int(rng.integers(1, train_count))
        board_index = int(rng.integers(len(train_plans[pair_index]["boards"])))
        train_plans[pair_index]["boards"][board_index]["mask"] = 0
    else:
        for pair_index, board_index in references:
            board = train_plans[pair_index]["boards"][board_index]
            if board["mask"] == 0:
                board["mask"] = 1 << int(rng.integers(8))
        first_mask_indices = rng.permutation(8)[:int(rng.integers(2, 6))]
        first_mask = sum(1 << int(index) for index in first_mask_indices)
        available = [index for index in range(8) if not first_mask & (1 << index)]
        second_mask_indices = rng.permutation(available)[:int(rng.integers(1, len(available) + 1))]
        second_mask = sum(1 << int(index) for index in second_mask_indices)
        train_plans[first_ref[0]]["boards"][first_ref[1]]["mask"] = first_mask
        train_plans[second_ref[0]]["boards"][second_ref[1]]["mask"] = second_mask

    train_boards = [board for plan in train_plans for board in plan["boards"]]
    if not any(0 < board["mask"] < 255 for board in train_boards):
        candidate = next(board for board in train_boards if board["mask"] != 0)
        candidate["mask"] = 1 << int(rng.integers(8))
    for plan in train_plans + test_plans:
        if all(board["mask"] == 255 for board in plan["boards"]):
            plan["boards"][0]["mask"] = 1 << int(rng.integers(8))

    train_roles = [base_structural] * train_count
    if mixed_roles:
        train_roles = [base_structural if bool(rng.integers(2)) else 8 - base_structural
                       for _ in range(train_count)]
        train_roles[int(rng.integers(train_count))] = base_structural
        train_roles[int(rng.integers(train_count))] = 8 - base_structural
        if len(set(train_roles)) < 2:
            train_roles[0], train_roles[1] = base_structural, 8 - base_structural
    test_roles = [
        base_structural if not mixed_roles or bool(rng.integers(2)) else 8 - base_structural
        for _ in range(test_count)
    ]

    puzzle = {"train": [], "test": []}
    for split, plans, roles in (
        ("train", train_plans, train_roles),
        ("test", test_plans, test_roles),
    ):
        for plan, structural_color in zip(plans, roles):
            input_grid, output_grid = _realize_plan(plan, structural_color)
            puzzle[split].append({"input": input_grid, "output": output_grid})
    return puzzle


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if not _is_integer(seed) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count, test_count = _resolve_counts(int(seed), num_train, num_test)
    rng = np.random.default_rng(int(seed))
    for _ in range(12):
        puzzle = _make_puzzle(rng, train_count, test_count)
        try:
            validate_generated_puzzle(puzzle)
        except AssertionError:
            continue
        return puzzle
    raise RuntimeError("could not construct a valid, unique puzzle")


def _inspect_pair(input_grid: np.ndarray, output_grid: np.ndarray):
    assignments = []
    for structural_color, field_color in ((8, 0), (0, 8)):
        if set(np.unique(input_grid).tolist()) != {structural_color, field_color}:
            continue
        positions = _find_boards(input_grid, structural_color, field_color)
        if not positions:
            continue
        if any(
            not _boards_are_separated(first, second)
            for index, first in enumerate(positions)
            for second in positions[index + 1:]
        ):
            continue
        coverage = np.zeros(input_grid.shape, dtype=np.int8)
        for top, left in positions:
            coverage[top:top + 5, left:left + 5] += 1
        if np.any(coverage > 1):
            continue
        if not np.all(coverage[input_grid == field_color] == 1):
            continue
        if not np.all(input_grid[coverage == 0] == structural_color):
            continue
        if not np.array_equal(
            _transform_grid(input_grid, structural_color, field_color), output_grid
        ):
            continue
        assignments.append((structural_color, field_color, positions))
    assert len(assignments) == 1
    return assignments[0]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    input_keys = set()
    pair_keys = set()
    training_labels = set()
    training_roles = set()
    test_roles = set()
    center_values = set()
    training_has_edge = False
    training_has_interior = False
    training_has_partial_mask = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            input_grid, output_grid = pair["input"], pair["output"]
            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert int(grid.min()) >= 0 and int(grid.max()) <= 9
            assert input_grid.shape == output_grid.shape
            assert _grid_text(input_grid) not in _SOURCE_INPUT_TEXTS
            input_key = _array_key(input_grid)
            pair_key = (input_key, _array_key(output_grid))
            assert input_key not in input_keys and pair_key not in pair_keys
            input_keys.add(input_key)
            pair_keys.add(pair_key)
            assert not np.array_equal(input_grid, output_grid)

            structural_color, field_color, positions = _inspect_pair(
                input_grid, output_grid
            )
            if split == "train":
                training_roles.add(structural_color)
            else:
                test_roles.add(structural_color)
            for top, left in positions:
                edge = _touches_edge((top, left), *input_grid.shape)
                if split == "train":
                    training_has_edge |= edge
                    training_has_interior |= not edge
                    center = int(input_grid[top + 2, left + 2])
                    center_values.add(center == field_color)
                masked = 0
                for core_row, core_column, label in _LABEL_POSITIONS:
                    row, column = top + 1 + core_row, left + 1 + core_column
                    if input_grid[row, column] == structural_color:
                        masked += 1
                        assert output_grid[row, column] == structural_color
                    else:
                        assert output_grid[row, column] == label
                        if split == "train":
                            training_labels.add(label)
                if split == "train" and 0 < masked < 8:
                    training_has_partial_mask = True
                assert input_grid[top + 2, left + 2] == output_grid[top + 2, left + 2]
    assert training_labels == _LABEL_SET
    assert test_roles <= training_roles
    assert center_values == {True, False}
    assert training_has_edge and training_has_interior and training_has_partial_mask


def _expect_value_error(**kwargs) -> None:
    try:
        generate_puzzle(991, **kwargs)
    except ValueError:
        return
    assert False, "expected ValueError"


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(puzzle) for puzzle in puzzles}) == 42
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3

    train_counts = set()
    test_counts = set()
    shapes = set()
    layouts = set()
    masks = set()
    mask_levels = set()
    edge_sets = set()
    roles = set()
    first_pair_full = set()
    compact_multi_tests = 0
    large_sparse_train = 0
    unchanged_sharing = 0
    familiar_or_smaller_tests = 0
    count_increase = 0
    staggered_tests = 0
    no_edge_tests = 0
    later_full_puzzles = 0
    collective_puzzles = 0
    mixed_role_puzzles = 0
    masked_interior_single_train = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_board_counts = []
        test_board_counts = []
        first_full = False
        later_full = False
        any_train_full = False
        puzzle_roles = set()
        for split in ("train", "test"):
            for pair_index, pair in enumerate(puzzle[split]):
                grid = pair["input"]
                structural, field, positions = _inspect_pair(grid, pair["output"])
                roles.add(structural)
                puzzle_roles.add(structural)
                count = len(positions)
                shapes.add(grid.shape)
                layouts.add((grid.shape, tuple(positions)))
                if split == "train":
                    train_counts.add(count)
                    train_board_counts.append(count)
                    if count == 1 and min(grid.shape) >= 22:
                        large_sparse_train += 1
                else:
                    test_counts.add(count)
                    test_board_counts.append(count)
                    if count >= 3 and max(grid.shape) <= 20:
                        compact_multi_tests += 1
                    if len({row for row, _ in positions}) > 3:
                        staggered_tests += 1
                edges = tuple(sorted({
                    side for top, left in positions
                    for side, reached in (
                        ("top", top == 0), ("bottom", top == grid.shape[0] - 5),
                        ("left", left == 0), ("right", left == grid.shape[1] - 5),
                    ) if reached
                }))
                edge_sets.add(edges)
                if split == "test" and not edges:
                    no_edge_tests += 1
                board_masks = []
                for top, left in positions:
                    mask = tuple(
                        int(grid[top + 1 + row, left + 1 + column] == structural)
                        for row, column, _ in _LABEL_POSITIONS
                    )
                    masks.add(mask)
                    mask_levels.add(sum(mask))
                    board_masks.append(sum(bit << index for index, bit in enumerate(mask)))
                if split == "train" and pair_index == 0:
                    first_full |= 0 in board_masks
                if split == "train":
                    any_train_full |= 0 in board_masks
                    later_full |= pair_index > 0 and 0 in board_masks
                    if (
                        pair_index < 3 and count == 1 and not edges
                        and 0 < board_masks[0] < 255
                    ):
                        masked_interior_single_train += 1
                if 255 in board_masks and any(mask != 255 for mask in board_masks):
                    unchanged_sharing += 1
        first_pair_full.add(first_full)
        later_full_puzzles += later_full
        collective_puzzles += not any_train_full
        mixed_role_puzzles += len(puzzle_roles) > 1
        assert len(set(train_board_counts)) > 1
        if min(test_board_counts) <= max(train_board_counts):
            familiar_or_smaller_tests += 1
        if max(test_board_counts) > max(train_board_counts):
            count_increase += 1
    assert roles == {0, 8}
    assert train_counts >= {1, 2, 3, 4}
    assert test_counts >= {1, 2, 3, 4, 5, 6}
    assert len(shapes) >= 80 and len(layouts) >= 120
    assert len(masks) >= 80
    assert mask_levels == set(range(9))
    mask_values = {
        sum(bit << index for index, bit in enumerate(pattern))
        for pattern in masks
    }
    assert mask_values & {7, 224, 41, 148, 66, 24}
    assert mask_values & {165, 90}
    assert any(value and value & ~165 == 0 for value in mask_values)
    assert len(edge_sets) >= 8
    assert first_pair_full == {True, False}
    assert compact_multi_tests > 0 and large_sparse_train > 0
    assert unchanged_sharing > 0 and familiar_or_smaller_tests > 0
    assert count_increase > 0 and staggered_tests > 0 and no_edge_tests > 0
    assert later_full_puzzles > 0 and collective_puzzles > 0
    assert mixed_role_puzzles > 0 and masked_interior_single_train > 0

    explicit_cases = ((600, 2, 1), (601, 10, 1), (602, 10, 4), (603, 7, 3))
    for seed, train_count, test_count in explicit_cases:
        puzzle = generate_puzzle(seed, num_train=train_count, num_test=test_count)
        validate_generated_puzzle(puzzle)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
    for seed, train_count, test_count in (
        (700, None, 1), (701, None, 4), (702, 2, None), (703, 10, None)
    ):
        puzzle = generate_puzzle(seed, num_train=train_count, num_test=test_count)
        validate_generated_puzzle(puzzle)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
    first = generate_puzzle(54321, num_train=6, num_test=2)
    second = generate_puzzle(54321, num_train=6, num_test=2)
    assert _puzzle_key(first) == _puzzle_key(second)
    for kwargs in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4},
    ):
        _expect_value_error(**kwargs)
