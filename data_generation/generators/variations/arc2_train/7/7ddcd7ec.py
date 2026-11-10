import numpy as np


# Each entry is:
# (starter row offset from core top-left,
#  starter column offset,
#  ray row step,
#  ray column step)
#
# Direction indices:
# 0 = northwest, 1 = northeast, 2 = southwest, 3 = southeast
_DIRECTION_SPECS = (
    (-1, -1, -1, -1),
    (-1, 2, -1, 1),
    (2, -1, 1, -1),
    (2, 2, 1, 1),
)


# Flattened source pairs, used to verify that no generated pair is an exact
# copy of any supplied source pair.
_SOURCE_PAIR_STRINGS = (
    (
        (
            "0000000000"
            "0000004000"
            "0000440000"
            "0000440000"
            "0000004000"
            "0000000000"
            "0000000000"
            "0000000000"
            "0000000000"
            "0000000000"
        ),
        (
            "0000000400"
            "0000004000"
            "0000440000"
            "0000440000"
            "0000004000"
            "0000000400"
            "0000000040"
            "0000000004"
            "0000000000"
            "0000000000"
        ),
    ),
    (
        (
            "0000000000"
            "0000000000"
            "0033000000"
            "0033000000"
            "0000300000"
            "0000000000"
            "0000000000"
            "0000000000"
            "0000000000"
            "0000000000"
        ),
        (
            "0000000000"
            "0000000000"
            "0033000000"
            "0033000000"
            "0000300000"
            "0000030000"
            "0000003000"
            "0000000300"
            "0000000030"
            "0000000003"
        ),
    ),
    (
        (
            "0000000000"
            "0000000000"
            "0000007000"
            "0000770000"
            "0000770000"
            "0007000000"
            "0000000000"
            "0000000000"
            "0000000000"
            "0000000000"
        ),
        (
            "0000000070"
            "0000000700"
            "0000007000"
            "0000770000"
            "0000770000"
            "0007000000"
            "0070000000"
            "0700000000"
            "7000000000"
            "0000000000"
        ),
    ),
    (
        (
            "0000000000"
            "0000000000"
            "0000000000"
            "0000080000"
            "0008800000"
            "0008800000"
            "0080080000"
            "0000000000"
            "0000000000"
            "0000000000"
        ),
        (
            "0000000080"
            "0000000800"
            "0000008000"
            "0000080000"
            "0008800000"
            "0008800000"
            "0080080000"
            "0800008000"
            "8000000800"
            "0000000080"
        ),
    ),
)


def _validate_optional_count(
    name: str,
    value: int | None,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return result


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_optional_count(
        "num_train", num_train, 2, 10
    )
    test_count = _validate_optional_count(
        "num_test", num_test, 1, 4
    )

    if train_count is None and test_count is None:
        # This stratified deterministic rule guarantees broad count diversity
        # for consecutive seeds while always satisfying train > test.
        test_count = 1 + seed % 4
        number_of_train_choices = 10 - test_count
        offset = (seed + seed // 4) % number_of_train_choices
        train_count = test_count + 1 + offset
    elif train_count is None:
        assert test_count is not None
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        test_count = int(rng.integers(1, maximum_test + 1))

    assert train_count is not None
    assert test_count is not None

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _sample_shape(rng: np.random.Generator) -> tuple[int, int]:
    # Draw the axes separately so both square and rectangular grids occur.
    bands = ((6, 10), (10, 20), (20, 31))
    dimensions = []
    for _ in range(2):
        band = int(rng.choice(3, p=(0.27, 0.43, 0.30)))
        low, high = bands[band]
        dimensions.append(int(rng.integers(low, high)))
    return dimensions[0], dimensions[1]


def _sample_subset(
    rng: np.random.Generator, size: int
) -> tuple[int, ...]:
    return tuple(sorted(int(value) for value in rng.choice(
        4, size=size, replace=False
    )))


def _direction_plans(
    rng: np.random.Generator,
    seed: int,
    train_count: int,
    test_count: int,
) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    # The first two demonstrations cover all four directions even when the
    # caller requests the minimum training count.
    # Use a different seed stratum from the automatic test count so the
    # number of test pairs does not determine their ray cardinality.
    first_test_size = 1 + (seed // 4) % 4
    first_size = int(rng.integers(1, 4 if first_test_size == 4 else 5))
    order = [int(value) for value in rng.permutation(4)]
    first = tuple(sorted(order[:first_size]))
    remainder = tuple(sorted(order[first_size:]))
    if not remainder:
        remainder = _sample_subset(rng, int(rng.integers(1, 4)))
    training = [first, remainder]
    while len(training) < train_count:
        training.append(_sample_subset(rng, int(rng.integers(1, 5))))
    training = [training[int(i)] for i in rng.permutation(len(training))]

    candidates = [
        tuple(direction for direction in range(4) if mask & (1 << direction))
        for mask in range(1, 16)
    ]
    lead_options = [
        subset for subset in candidates
        if len(subset) == first_test_size and subset not in training
    ]
    if not lead_options:
        lead_options = [
            subset for subset in candidates if len(subset) == first_test_size
        ]
    testing = [lead_options[int(rng.integers(len(lead_options)))]]
    while len(testing) < test_count:
        options = [subset for subset in candidates if subset not in testing]
        testing.append(options[int(rng.integers(len(options)))])
    return training, testing


def _color_plan(
    rng: np.random.Generator, count: int, train_count: int,
) -> list[tuple[int, int]]:
    mode = int(rng.integers(3))
    colors = [int(value) for value in rng.permutation(10)]
    if mode == 0:
        background = colors[0]
        return [
            (background, colors[1 + int(rng.integers(9))])
            for _ in range(count)
        ]

    pairs = []
    for _ in range(count):
        background = int(rng.integers(10))
        foreground = int(rng.integers(9))
        if foreground >= background:
            foreground += 1
        pairs.append((background, foreground))

    if mode == 1:
        # A test reuses a demonstrated foreground.
        test_index = train_count
        test_background = pairs[test_index][0]
        train_foreground = pairs[0][1]
        if test_background == train_foreground:
            test_background = (test_background + 1) % 10
        pairs[test_index] = (test_background, train_foreground)
    else:
        # A pair of training examples reverses the semantic color roles.
        a, b = colors[:2]
        pairs[0] = (a, b)
        pairs[1] = (b, a)
        pairs[train_count] = (b, a)
    return pairs


def _render_output(
    input_grid: np.ndarray,
    core_row: int,
    core_col: int,
    foreground: int,
    directions: tuple[int, ...] | frozenset[int],
) -> np.ndarray:
    output = input_grid.copy()
    height, width = input_grid.shape

    for direction in directions:
        row_offset, col_offset, row_step, col_step = (
            _DIRECTION_SPECS[direction]
        )
        row = core_row + row_offset
        col = core_col + col_offset

        while 0 <= row < height and 0 <= col < width:
            output[row, col] = foreground
            row += row_step
            col += col_step

    return output


def _grid_signature(grid: np.ndarray) -> tuple:
    normalized = grid.astype(np.int8, copy=False)
    return tuple(grid.shape), normalized.tobytes()


def _pair_signature(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple:
    return _grid_signature(input_grid), _grid_signature(output_grid)


def _matches_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    if input_grid.shape != (10, 10) or output_grid.shape != (10, 10):
        return False

    input_text = "".join(
        str(int(value)) for value in input_grid.reshape(-1)
    )
    output_text = "".join(
        str(int(value)) for value in output_grid.reshape(-1)
    )
    return (input_text, output_text) in _SOURCE_PAIR_STRINGS


def _generate_pair(
    rng: np.random.Generator,
    shape: tuple[int, int],
    directions: tuple[int, ...],
    background: int,
    foreground: int,
    used_inputs: set,
    row_preference: int,
) -> tuple[np.ndarray, np.ndarray]:
    height, width = shape
    row_low = 2 if any(direction < 2 for direction in directions) else 0
    row_high = height - 4 if any(direction >= 2 for direction in directions) else height - 2
    col_low = 2 if any(direction in (0, 2) for direction in directions) else 0
    col_high = width - 4 if any(direction in (1, 3) for direction in directions) else width - 2
    positions = [
        (row, col)
        for row in range(row_low, row_high + 1)
        for col in range(col_low, col_high + 1)
    ]
    assert positions
    if rng.random() < 0.40:
        edge_positions = [
            position for position in positions
            if position[0] in (row_low, row_high)
            or position[1] in (col_low, col_high)
        ]
        other_positions = [position for position in positions if position not in edge_positions]
        positions = (
            [edge_positions[int(i)] for i in rng.permutation(len(edge_positions))]
            + [other_positions[int(i)] for i in rng.permutation(len(other_positions))]
        )
    else:
        positions = [positions[int(i)] for i in rng.permutation(len(positions))]

    if row_preference:
        margin = max(1, (row_high - row_low) // 3)
        if row_preference < 0:
            preferred = [position for position in positions if position[0] <= row_low + margin]
        else:
            preferred = [position for position in positions if position[0] >= row_high - margin]
        positions = preferred + [position for position in positions if position not in preferred]

    for core_row, core_col in positions:

        input_grid = np.full(
            (height, width),
            background,
            dtype=np.int8,
        )
        input_grid[
            core_row : core_row + 2,
            core_col : core_col + 2,
        ] = foreground

        for direction in directions:
            row_offset, col_offset, _, _ = _DIRECTION_SPECS[direction]
            input_grid[
                core_row + row_offset,
                core_col + col_offset,
            ] = foreground

        input_signature = _grid_signature(input_grid)
        if input_signature in used_inputs:
            continue

        output_grid = _render_output(
            input_grid,
            core_row,
            core_col,
            foreground,
            directions,
        )

        if np.array_equal(input_grid, output_grid):
            continue
        if _matches_source_pair(input_grid, output_grid):
            continue

        used_inputs.add(input_signature)
        return input_grid, output_grid

    raise ValueError("unable to construct a unique pair at this shape")


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

    rng = np.random.default_rng(seed_value)
    train_count, test_count = _resolve_counts(
        seed_value,
        rng,
        num_train,
        num_test,
    )

    count = train_count + test_count
    training_plan, testing_plan = _direction_plans(
        rng, seed_value, train_count, test_count
    )
    color_plan = _color_plan(rng, count, train_count)
    shared_shape = seed_value % 3 == 0
    if shared_shape:
        sampled = (10, 10) if seed_value % 11 == 0 else _sample_shape(rng)
        common_shape = (max(9, sampled[0]), max(9, sampled[1]))
        shapes = [common_shape] * count
    else:
        shapes = [_sample_shape(rng) for _ in range(count)]
        if len(set(shapes)) == 1:
            shapes[-1] = (30, 6) if shapes[-1] != (30, 6) else (6, 30)

    # Some common-canvas puzzles retain the source's vertical shift; some
    # reverse it. All other puzzles let train and test positions mix.
    relation = seed_value % 9 if shared_shape else -1

    # Spawn independent deterministic streams. No pair recreates the same
    # unchanged RNG seed.
    child_sequences = np.random.SeedSequence(seed_value).spawn(count)

    used_inputs: set = set()
    training_pairs: list[dict] = []
    testing_pairs: list[dict] = []

    for index in range(count):
        pair_rng = np.random.default_rng(child_sequences[index])
        directions = (
            training_plan[index] if index < train_count
            else testing_plan[index - train_count]
        )
        background, foreground = color_plan[index]
        if relation == 0:
            row_preference = -1 if index < train_count else 1
        elif relation == 3:
            row_preference = 1 if index < train_count else -1
        else:
            row_preference = 0
        input_grid, output_grid = _generate_pair(
            pair_rng,
            shapes[index],
            directions,
            background,
            foreground,
            used_inputs,
            row_preference,
        )
        pair = {"input": input_grid, "output": output_grid}
        if index < train_count:
            training_pairs.append(pair)
        else:
            testing_pairs.append(pair)

    puzzle = {
        "train": training_pairs,
        "test": testing_pairs,
    }
    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2

    height, width = grid.shape
    assert 1 <= height <= 30
    assert 1 <= width <= 30
    assert grid.dtype.kind in "iu"

    assert np.all(grid >= 0)
    assert np.all(grid <= 9)


def _analyze_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> dict:
    assert input_grid.shape == output_grid.shape
    height, width = input_grid.shape

    input_values, input_counts = np.unique(
        input_grid,
        return_counts=True,
    )
    assert len(input_values) == 2

    background_index = int(np.argmax(input_counts))
    background = int(input_values[background_index])
    foreground = int(
        input_values[1 - background_index]
    )
    assert int(input_counts[background_index]) > int(
        input_counts[1 - background_index]
    )

    foreground_mask = input_grid == foreground
    core_candidates: list[tuple[int, int]] = []

    for row in range(height - 1):
        for col in range(width - 1):
            if np.all(
                foreground_mask[row : row + 2, col : col + 2]
            ):
                core_candidates.append((row, col))

    assert len(core_candidates) == 1
    core_row, core_col = core_candidates[0]

    expected_input_cells = {
        (core_row, core_col),
        (core_row, core_col + 1),
        (core_row + 1, core_col),
        (core_row + 1, core_col + 1),
    }
    directions: set[int] = set()
    ray_lengths: list[int] = []
    terminations: list[str] = []

    for direction, spec in enumerate(_DIRECTION_SPECS):
        row_offset, col_offset, row_step, col_step = spec
        starter = (
            core_row + row_offset,
            core_col + col_offset,
        )
        if (
            0 <= starter[0] < height
            and 0 <= starter[1] < width
            and int(input_grid[starter]) == foreground
        ):
            # Each visible cue must add at least one further cell.
            assert 0 <= starter[0] + row_step < height
            assert 0 <= starter[1] + col_step < width
            directions.add(direction)
            expected_input_cells.add(starter)
            row, col = starter
            length = 0
            while 0 <= row + row_step < height and 0 <= col + col_step < width:
                row += row_step
                col += col_step
                length += 1
            ray_lengths.append(length)
            row_exits = not (0 <= row + row_step < height)
            col_exits = not (0 <= col + col_step < width)
            terminations.append(
                "corner" if row_exits and col_exits else
                "row" if row_exits else "column"
            )

    actual_input_cells = {
        (int(row), int(col))
        for row, col in np.argwhere(foreground_mask)
    }
    assert actual_input_cells == expected_input_cells
    assert 1 <= len(directions) <= 4

    expected_output = _render_output(
        input_grid,
        core_row,
        core_col,
        foreground,
        frozenset(directions),
    )
    assert np.array_equal(output_grid, expected_output)
    assert not np.array_equal(input_grid, output_grid)

    output_values = {int(value) for value in np.unique(output_grid)}
    assert output_values == {background, foreground}

    return {
        "background": background,
        "foreground": foreground,
        "core": (core_row, core_col),
        "directions": frozenset(directions),
        "shape": input_grid.shape,
        "added_cells": int(np.count_nonzero(input_grid != output_grid)),
        "ray_lengths": tuple(ray_lengths),
        "terminations": tuple(terminations),
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

    seen_inputs: set = set()
    seen_pairs: set = set()
    train_info: list[dict] = []
    test_info: list[dict] = []

    for section_name in ("train", "test"):
        for pair in puzzle[section_name]:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {"input", "output"}

            input_grid = pair["input"]
            output_grid = pair["output"]
            _validate_grid(input_grid)
            _validate_grid(output_grid)
            assert input_grid.shape == output_grid.shape

            input_signature = _grid_signature(input_grid)
            pair_signature = _pair_signature(
                input_grid,
                output_grid,
            )
            assert input_signature not in seen_inputs
            assert pair_signature not in seen_pairs
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)

            assert not _matches_source_pair(
                input_grid,
                output_grid,
            )

            info = _analyze_pair(input_grid, output_grid)
            if section_name == "train":
                train_info.append(info)
            else:
                test_info.append(info)

    demonstrated_directions: set[int] = set()
    for info in train_info:
        demonstrated_directions.update(info["directions"])
    assert demonstrated_directions == set(range(4))
    assert any(len(info["directions"]) >= 2 for info in train_info)
    assert all(
        info["directions"].issubset(demonstrated_directions)
        for info in test_info
    )
    assert len({info["directions"] for info in test_info}) == test_count


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first.keys()) != {"train", "test"}:
        return False
    if set(second.keys()) != {"train", "test"}:
        return False

    for section_name in ("train", "test"):
        first_pairs = first[section_name]
        second_pairs = second[section_name]
        if len(first_pairs) != len(second_pairs):
            return False

        for first_pair, second_pair in zip(
            first_pairs,
            second_pairs,
        ):
            for grid_name in ("input", "output"):
                first_grid = first_pair[grid_name]
                second_grid = second_pair[grid_name]
                if first_grid.dtype != second_grid.dtype:
                    return False
                if not np.array_equal(first_grid, second_grid):
                    return False

    return True


def _puzzle_signature(puzzle: dict) -> tuple:
    sections = []
    for section_name in ("train", "test"):
        pairs = []
        for pair in puzzle[section_name]:
            pairs.append(
                _pair_signature(
                    pair["input"],
                    pair["output"],
                )
            )
        sections.append(tuple(pairs))
    return tuple(sections)


def _require_value_error(**kwargs) -> None:
    try:
        generate_puzzle(0, **kwargs)
    except ValueError:
        return
    raise AssertionError(
        f"expected ValueError for arguments {kwargs}"
    )


def _predict_from_training(train_pairs: list[dict], test_input: np.ndarray) -> np.ndarray:
    # Reconstruct the four demonstrated diagonal continuations using only
    # visible training pairs and the current test input.
    demonstrated = set()
    for pair in train_pairs:
        info = _analyze_pair(pair["input"], pair["output"])
        demonstrated.update(info["directions"])
    assert demonstrated == set(range(4))

    colors, counts = np.unique(test_input, return_counts=True)
    assert len(colors) == 2
    foreground = int(colors[int(np.argmin(counts))])
    height, width = test_input.shape
    cores = [
        (row, col)
        for row in range(height - 1)
        for col in range(width - 1)
        if np.all(test_input[row:row + 2, col:col + 2] == foreground)
    ]
    assert len(cores) == 1
    core_row, core_col = cores[0]
    predicted = test_input.copy()
    for direction, (row_offset, col_offset, row_step, col_step) in enumerate(_DIRECTION_SPECS):
        starter_row = core_row + row_offset
        starter_col = core_col + col_offset
        if (
            0 <= starter_row < height
            and 0 <= starter_col < width
            and test_input[starter_row, starter_col] == foreground
        ):
            assert direction in demonstrated
            row, col = starter_row + row_step, starter_col + col_step
            while 0 <= row < height and 0 <= col < width:
                predicted[row, col] = foreground
                row += row_step
                col += col_step
    return predicted


def self_test() -> None:
    automatic_puzzles = [generate_puzzle(seed) for seed in range(42)]
    train_counts, test_counts = set(), set()
    signatures, shapes, layouts, subsets = set(), set(), set(), set()
    train_cardinalities, test_cardinalities = set(), set()
    heights, widths, backgrounds, foregrounds = set(), set(), set(), set()
    terminations, ray_lengths = set(), set()
    mixed_shape = mixed_background = role_swaps = reused_foreground = 0
    test_above_all = test_below_all = interleaved_rows = 0
    edge_adjacent = short_and_long = four_ray_tests = 0
    first_train_cardinalities = set()

    for puzzle in automatic_puzzles:
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_puzzle_signature(puzzle))
        train_info = [
            _analyze_pair(pair["input"], pair["output"])
            for pair in puzzle["train"]
        ]
        test_info = [
            _analyze_pair(pair["input"], pair["output"])
            for pair in puzzle["test"]
        ]
        all_info = train_info + test_info
        first_train_cardinalities.add(len(train_info[0]["directions"]))
        mixed_shape += len({info["shape"] for info in all_info}) > 1
        mixed_background += len({info["background"] for info in all_info}) > 1
        train_foregrounds = {info["foreground"] for info in train_info}
        reused_foreground += any(info["foreground"] in train_foregrounds for info in test_info)
        role_swaps += any(
            a["background"] == b["foreground"] and a["foreground"] == b["background"]
            for a in all_info for b in all_info if a is not b
        )
        if len({info["shape"] for info in all_info}) == 1:
            train_rows = [info["core"][0] for info in train_info]
            test_rows = [info["core"][0] for info in test_info]
            above = max(test_rows) < min(train_rows)
            below = min(test_rows) > max(train_rows)
            test_above_all += above
            test_below_all += below
            interleaved_rows += not (above or below)

        for section_name, infos in (("train", train_info), ("test", test_info)):
            for pair, info in zip(puzzle[section_name], infos):
                shape = info["shape"]
                shapes.add(shape)
                heights.add(shape[0])
                widths.add(shape[1])
                backgrounds.add(info["background"])
                foregrounds.add(info["foreground"])
                subsets.add(info["directions"])
                cardinality = len(info["directions"])
                (train_cardinalities if section_name == "train" else test_cardinalities).add(cardinality)
                four_ray_tests += section_name == "test" and cardinality == 4
                layout = (shape, info["core"], info["directions"])
                layouts.add(layout)
                terminations.update(info["terminations"])
                ray_lengths.update(info["ray_lengths"])
                short_and_long += (
                    len(info["ray_lengths"]) >= 2
                    and min(info["ray_lengths"]) <= 2
                    and max(info["ray_lengths"]) >= 5
                )
                row, col = info["core"]
                edge_adjacent += (
                    row == 0 or col == 0 or row + 1 == shape[0] - 1
                    or col + 1 == shape[1] - 1
                )
                assert not _matches_source_pair(pair["input"], pair["output"])
                if section_name == "test":
                    assert np.array_equal(
                        _predict_from_training(puzzle["train"], pair["input"]),
                        pair["output"],
                    )

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42
    assert len(shapes) >= 60 and len(layouts) >= 120
    assert min(heights) <= 7 and max(heights) >= 28
    assert min(widths) <= 7 and max(widths) >= 28
    assert len(backgrounds) == 10 and len(foregrounds) == 10
    assert len(subsets) == 15
    assert train_cardinalities == {1, 2, 3, 4}
    assert test_cardinalities == {1, 2, 3, 4}
    assert first_train_cardinalities == {1, 2, 3, 4}
    assert mixed_shape >= 15 and mixed_background >= 8
    assert test_above_all >= 1 and test_below_all >= 1 and interleaved_rows >= 1
    assert role_swaps >= 5 and reused_foreground >= 8
    assert edge_adjacent >= 5 and short_and_long >= 3
    assert terminations == {"row", "column", "corner"}
    assert min(ray_lengths) == 1 and max(ray_lengths) >= 10
    assert four_ray_tests >= 10

    first = generate_puzzle(12345)
    assert _puzzles_equal(first, generate_puzzle(12345))
    explicit = generate_puzzle(54321, 8, 3)
    assert _puzzles_equal(explicit, generate_puzzle(54321, 8, 3))
    assert not _puzzles_equal(generate_puzzle(700), generate_puzzle(701))
    for seed, requested_train, requested_test in (
        (101, 2, 1), (102, 10, 1), (103, 10, 4),
    ):
        puzzle = generate_puzzle(seed, requested_train, requested_test)
        assert len(puzzle["train"]) == requested_train
        assert len(puzzle["test"]) == requested_test
        validate_generated_puzzle(puzzle)
    for seed, requested_train, requested_test in (
        (201, None, 1), (202, None, 4), (203, 2, None), (204, 10, None),
    ):
        puzzle = generate_puzzle(seed, requested_train, requested_test)
        if requested_train is not None:
            assert len(puzzle["train"]) == requested_train
        if requested_test is not None:
            assert len(puzzle["test"]) == requested_test
        validate_generated_puzzle(puzzle)
    for arguments in (
        {"num_train": 1}, {"num_train": 11},
        {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4},
    ):
        _require_value_error(**arguments)
