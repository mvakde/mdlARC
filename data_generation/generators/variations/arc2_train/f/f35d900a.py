import numpy as np


_SOURCE_INPUT_SPECS = (
    (
        17,
        16,
        ((3, 2, 2), (3, 13, 4), (13, 2, 4), (13, 13, 2)),
    ),
    (
        17,
        14,
        ((3, 2, 1), (3, 10, 8), (14, 2, 8), (14, 10, 1)),
    ),
    (
        17,
        16,
        ((3, 4, 3), (3, 12, 8), (8, 4, 8), (8, 12, 3)),
    ),
    (
        14,
        14,
        ((2, 1, 2), (2, 6, 3), (8, 1, 3), (8, 6, 2)),
    ),
    (
        17,
        18,
        ((4, 2, 4), (4, 12, 1), (15, 2, 1), (15, 12, 4)),
    ),
)

_SOURCE_LAYOUT_KEYS = frozenset(
    {
        (17, 16, (3, 13), (2, 13)),
        (17, 14, (3, 14), (2, 10)),
        (17, 16, (3, 8), (4, 12)),
        (14, 14, (2, 8), (1, 6)),
        (17, 18, (4, 15), (2, 12)),
    }
)


def _valid_int(value) -> bool:
    return (
        isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
    )


def _check_requested_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if not _valid_int(value) or not low <= int(value) <= high:
        raise ValueError(
            f"{name} must be an integer from {low} through {high}"
        )

    return int(value)


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _check_requested_count(
        "num_train", num_train, 2, 10
    )
    test_count = _check_requested_count(
        "num_test", num_test, 1, 4
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError(
                "num_train must be greater than num_test"
            )
        return train_count, test_count

    if train_count is None and test_count is None:
        test_count = int(rng.integers(1, 5))
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))

    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))

    else:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError(
                "no legal test count for the requested train count"
            )
        test_count = int(rng.integers(1, maximum_test + 1))

    if train_count <= test_count:
        raise ValueError(
            "num_train must be greater than num_test"
        )

    return train_count, test_count


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _analyze_input(grid: np.ndarray) -> dict:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2

    height, width = grid.shape
    values, counts = np.unique(grid, return_counts=True)

    assert len(values) == 3

    background_candidates = values[counts == height * width - 4]
    assert len(background_candidates) == 1
    background = int(background_candidates[0])

    marker_positions = np.argwhere(grid != background)
    assert marker_positions.shape == (4, 2)

    rows = sorted({int(position[0]) for position in marker_positions})
    columns = sorted(
        {int(position[1]) for position in marker_positions}
    )

    assert len(rows) == 2
    assert len(columns) == 2

    top_row, bottom_row = rows
    left_column, right_column = columns

    expected_positions = {
        (top_row, left_column),
        (top_row, right_column),
        (bottom_row, left_column),
        (bottom_row, right_column),
    }
    actual_positions = set(
        map(tuple, marker_positions.tolist())
    )
    assert actual_positions == expected_positions

    assert top_row >= 1
    assert bottom_row <= height - 2
    assert left_column >= 1
    assert right_column <= width - 2

    assert bottom_row - top_row >= 3
    assert right_column - left_column >= 3

    top_left = int(grid[top_row, left_column])
    top_right = int(grid[top_row, right_column])
    bottom_left = int(grid[bottom_row, left_column])
    bottom_right = int(grid[bottom_row, right_column])

    assert top_left == bottom_right
    assert top_right == bottom_left
    assert top_left != top_right

    for marker_color in (top_left, top_right):
        assert int(np.count_nonzero(grid == marker_color)) == 2

    return {
        "height": height,
        "width": width,
        "background": background,
        "rows": (top_row, bottom_row),
        "cols": (left_column, right_column),
        "marker_colors": frozenset((top_left, top_right)),
        "horizontal_gap": right_column - left_column - 3,
        "vertical_gap": bottom_row - top_row - 3,
        "margins": (
            top_row - 1,
            height - bottom_row - 2,
            left_column - 1,
            width - right_column - 2,
        ),
    }


def _transform(
    input_grid: np.ndarray,
    connector_color: int,
) -> np.ndarray:
    info = _analyze_input(input_grid)

    input_colors = set(
        int(value) for value in np.unique(input_grid)
    )
    assert connector_color not in input_colors

    output_grid = input_grid.copy()
    marker_colors = tuple(info["marker_colors"])
    top_row, bottom_row = info["rows"]
    left_column, right_column = info["cols"]

    for row in (top_row, bottom_row):
        for column in (left_column, right_column):
            center_color = int(input_grid[row, column])
            other_color = (
                marker_colors[1]
                if center_color == marker_colors[0]
                else marker_colors[0]
            )

            output_grid[
                row - 1:row + 2,
                column - 1:column + 2,
            ] = other_color
            output_grid[row, column] = center_color

    for row in (top_row, bottom_row):
        for column in range(left_column + 2, right_column - 1):
            distance_from_left_tile = column - (left_column + 1)
            distance_from_right_tile = (
                right_column - 1
            ) - column
            nearest_distance = min(
                distance_from_left_tile,
                distance_from_right_tile,
            )

            if nearest_distance % 2 == 1:
                output_grid[row, column] = connector_color

    for column in (left_column, right_column):
        for row in range(top_row + 2, bottom_row - 1):
            distance_from_top_tile = row - (top_row + 1)
            distance_from_bottom_tile = (
                bottom_row - 1
            ) - row
            nearest_distance = min(
                distance_from_top_tile,
                distance_from_bottom_tile,
            )

            if nearest_distance % 2 == 1:
                output_grid[row, column] = connector_color

    return output_grid


def _infer_connector(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> int:
    input_colors = set(
        int(value) for value in np.unique(input_grid)
    )
    output_colors = set(
        int(value) for value in np.unique(output_grid)
    )

    assert input_colors.issubset(output_colors)

    output_only_colors = output_colors - input_colors
    assert len(output_only_colors) == 1

    connector_color = next(iter(output_only_colors))

    expected_output = _transform(
        input_grid,
        connector_color,
    )
    assert np.array_equal(output_grid, expected_output)

    return connector_color


def _matches_source_input(grid: np.ndarray) -> bool:
    for height, width, markers in _SOURCE_INPUT_SPECS:
        if grid.shape != (height, width):
            continue

        source_input = np.zeros(
            (height, width),
            dtype=np.int8,
        )
        for row, column, color in markers:
            source_input[row, column] = color

        if np.array_equal(grid, source_input):
            return True

    return False


def _sample_puzzle_parameters(
    rng: np.random.Generator,
    train_count: int,
) -> dict:
    palette = [int(value) for value in rng.permutation(10)]
    marker_count = int(rng.integers(2, min(5, train_count * 2) + 1))
    background_count = int(rng.integers(1, min(3, 9 - marker_count, train_count) + 1))
    return {
        "connector": palette[0],
        "marker_palette": tuple(palette[1:1 + marker_count]),
        "background_palette": tuple(
            palette[1 + marker_count:1 + marker_count + background_count]
        ),
    }


def _make_color_plan(
    rng: np.random.Generator,
    marker_palette: tuple[int, ...],
    num_train: int,
    num_test: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    colors = list(marker_palette)
    rng.shuffle(colors)
    all_pairs = [
        (colors[i], colors[j])
        for i in range(len(colors))
        for j in range(i + 1, len(colors))
    ]

    train_pairs = [(colors[0], colors[1])]
    covered = set(train_pairs[0])
    unseen = colors[2:]
    while unseen:
        remaining_slots = num_train - len(train_pairs)
        if len(unseen) >= 2 and len(unseen) >= remaining_slots * 2 - 1:
            pair = (unseen.pop(0), unseen.pop(0))
        else:
            covered_colors = tuple(sorted(covered))
            pair = (
                unseen.pop(0),
                covered_colors[int(rng.integers(0, len(covered_colors)))],
            )
        train_pairs.append(pair)
        covered.update(pair)

    initial_pairs = {frozenset(pair) for pair in train_pairs}
    reservable = [pair for pair in all_pairs if frozenset(pair) not in initial_pairs]
    reserved = (
        reservable[int(rng.integers(0, len(reservable)))]
        if reservable else None
    )
    available_train_pairs = [
        pair for pair in all_pairs if pair != reserved
    ]
    while len(train_pairs) < num_train:
        unused = [pair for pair in available_train_pairs if frozenset(pair) not in {
            frozenset(existing) for existing in train_pairs
        }]
        choices = unused if unused and float(rng.random()) < 0.8 else available_train_pairs
        train_pairs.append(choices[int(rng.integers(0, len(choices)))])
    rng.shuffle(train_pairs)

    familiar = [pair for pair in all_pairs if frozenset(pair) in {
        frozenset(existing) for existing in train_pairs
    }]
    novel = [pair for pair in all_pairs if pair not in familiar]
    rng.shuffle(familiar)
    rng.shuffle(novel)
    test_pairs = []
    if num_test >= 2 and novel:
        test_pairs.extend((familiar.pop(), novel.pop()))
    while len(test_pairs) < num_test:
        prefer_novel = bool(novel) and float(rng.random()) < 0.5
        options = novel if prefer_novel else familiar
        if not options:
            options = novel if novel else familiar
        if options:
            test_pairs.append(options.pop())
        else:
            test_pairs.append(all_pairs[int(rng.integers(0, len(all_pairs)))])
    rng.shuffle(test_pairs)
    return train_pairs, test_pairs


def _sample_gap(rng: np.random.Generator, regime: str) -> int:
    bucket = float(rng.random())
    if bucket < (0.30 if regime == "target_test" else 0.35):
        return int(rng.integers(1, 6))
    if bucket < (0.60 if regime == "target_test" else 0.75):
        return int(rng.integers(6, 12))
    return int(rng.integers(12, 21))


def _sample_axis_margins(
    rng: np.random.Generator,
    gap: int,
    first_edge: bool,
    second_edge: bool,
) -> tuple[int, int]:
    capacity = min(16, 24 - gap)
    if first_edge and second_edge:
        return 0, 0
    if first_edge:
        return 0, int(rng.integers(0, capacity + 1))
    if second_edge:
        return int(rng.integers(0, capacity + 1)), 0
    total = int(rng.integers(0, capacity + 1))
    first = int(rng.integers(0, total + 1))
    return first, total - first


def _sample_geometry(
    rng: np.random.Generator,
    regime: str,
    slot: int,
    train_gaps: tuple[set[int], set[int]],
    forced_edge: str | None,
) -> tuple[int, int, int, int, int, int]:
    if regime == "source_train":
        horizontal_gap = (
            int(rng.choice((6, 8, 10, 12, 14)))
            if slot == 0 else _sample_gap(rng, regime)
        )
        vertical_gap = (
            int(rng.choice((6, 8, 10, 12, 14)))
            if slot == 1 else _sample_gap(rng, regime)
        )
    elif regime == "target_test":
        gaps = []
        for known in train_gaps:
            if known and float(rng.random()) < 0.40:
                options = tuple(sorted(known))
                gaps.append(int(options[int(rng.integers(0, len(options)))]))
            else:
                choices = [gap for gap in range(1, 21) if gap not in known]
                if choices:
                    gaps.append(choices[int(rng.integers(0, len(choices)))])
                else:
                    gaps.append(_sample_gap(rng, regime))
        horizontal_gap, vertical_gap = gaps
    else:
        raise ValueError("unknown pair regime")

    edge = forced_edge
    if edge is None and regime == "target_test" and float(rng.random()) < 0.55:
        edge = ("top", "bottom", "left", "right")[int(rng.integers(0, 4))]
    top_margin, bottom_margin = _sample_axis_margins(
        rng, vertical_gap, edge == "top", edge == "bottom"
    )
    left_margin, right_margin = _sample_axis_margins(
        rng, horizontal_gap, edge == "left", edge == "right"
    )

    height = vertical_gap + 6 + top_margin + bottom_margin
    width = horizontal_gap + 6 + left_margin + right_margin
    top_row = top_margin + 1
    left_column = left_margin + 1
    return (
        height, width, top_row, top_row + vertical_gap + 3,
        left_column, left_column + horizontal_gap + 3,
    )


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    slot: int,
    marker_pair: tuple[int, int],
    background: int,
    connector: int,
    train_gaps: tuple[set[int], set[int]],
    forced_edge: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    (
        height, width, top_row, bottom_row, left_column, right_column,
    ) = _sample_geometry(rng, regime, slot, train_gaps, forced_edge)
    first_color, second_color = marker_pair
    if int(rng.integers(0, 2)):
        first_color, second_color = second_color, first_color

    input_grid = np.full((height, width), background, dtype=np.int8)
    input_grid[top_row, left_column] = first_color
    input_grid[top_row, right_column] = second_color
    input_grid[bottom_row, left_column] = second_color
    input_grid[bottom_row, right_column] = first_color
    return input_grid, _transform(input_grid, connector)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if not _valid_int(seed) or int(seed) < 0:
        raise ValueError(
            "seed must be a nonnegative integer"
        )

    rng = np.random.default_rng(int(seed))

    train_count, test_count = _resolve_counts(
        rng,
        num_train,
        num_test,
    )

    puzzle_parameters = _sample_puzzle_parameters(rng, train_count)

    train_color_plan, test_color_plan = _make_color_plan(
        rng,
        puzzle_parameters["marker_palette"],
        train_count,
        test_count,
    )

    backgrounds = list(puzzle_parameters["background_palette"])
    rng.shuffle(backgrounds)
    train_backgrounds = [
        backgrounds[index % len(backgrounds)]
        for index in range(train_count)
    ]
    rng.shuffle(train_backgrounds)
    test_backgrounds = [
        backgrounds[int(rng.integers(0, len(backgrounds)))]
        for _ in range(test_count)
    ]
    bottom_edge_slot = int(rng.integers(0, train_count))

    pair_seeds = rng.integers(
        0,
        2**63,
        size=train_count + test_count,
        dtype=np.int64,
    )

    puzzle = {
        "train": [],
        "test": [],
    }

    used_inputs = set()
    used_layouts = set()
    train_gaps: tuple[set[int], set[int]] = (set(), set())
    all_gaps: tuple[set[int], set[int]] = (set(), set())
    pair_number = 0

    generation_groups = (
        (
            "source_train",
            puzzle["train"],
            train_color_plan,
            train_backgrounds,
        ),
        (
            "target_test",
            puzzle["test"],
            test_color_plan,
            test_backgrounds,
        ),
    )

    for regime, destination, color_plan, background_plan in generation_groups:
        for slot, (marker_pair, background) in enumerate(
            zip(color_plan, background_plan)
        ):
            pair_rng = np.random.default_rng(
                int(pair_seeds[pair_number])
            )
            pair_number += 1

            for _ in range(500):
                input_grid, output_grid = _generate_pair(
                    pair_rng,
                    regime,
                    slot,
                    marker_pair,
                    background,
                    puzzle_parameters["connector"],
                    train_gaps,
                    "bottom" if regime == "source_train" and slot == bottom_edge_slot else None,
                )

                info = _analyze_input(input_grid)
                input_key = _grid_key(input_grid)
                layout_key = (
                    info["height"],
                    info["width"],
                    info["rows"],
                    info["cols"],
                )

                if (
                    input_key not in used_inputs
                    and layout_key not in used_layouts
                    and not _matches_source_input(input_grid)
                    and (
                        pair_number < train_count + test_count
                        or (
                            len(all_gaps[0] | {info["horizontal_gap"]}) > 1
                            and len(all_gaps[1] | {info["vertical_gap"]}) > 1
                        )
                    )
                ):
                    break
            else:
                raise ValueError(
                    "could not construct enough distinct valid pairs"
                )

            used_inputs.add(input_key)
            used_layouts.add(layout_key)
            all_gaps[0].add(info["horizontal_gap"])
            all_gaps[1].add(info["vertical_gap"])
            if regime == "source_train":
                train_gaps[0].add(info["horizontal_gap"])
                train_gaps[1].add(info["vertical_gap"])

            destination.append(
                {
                    "input": input_grid,
                    "output": output_grid,
                }
            )

    validate_generated_puzzle(puzzle)
    return puzzle


def _assert_grid(grid: np.ndarray) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2

    height, width = grid.shape
    assert 1 <= height <= 30
    assert 1 <= width <= 30

    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((grid >= 0) & (grid <= 9))


def _assert_source_train_structure(info: dict) -> None:
    assert 1 <= info["horizontal_gap"] <= 20
    assert 1 <= info["vertical_gap"] <= 20


def _assert_target_test_structure(info: dict) -> None:
    assert 1 <= info["horizontal_gap"] <= 20
    assert 1 <= info["vertical_gap"] <= 20


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {"train", "test"}

    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    input_keys = set()
    pair_keys = set()
    layout_keys = set()

    train_backgrounds = set()
    test_backgrounds = set()
    connectors = []
    train_infos = []
    test_infos = []

    groups = (
        ("source_train", puzzle["train"]),
        ("target_test", puzzle["test"]),
    )

    for regime, pairs in groups:
        for pair in pairs:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {
                "input",
                "output",
            }

            input_grid = pair["input"]
            output_grid = pair["output"]

            _assert_grid(input_grid)
            _assert_grid(output_grid)

            assert input_grid.shape == output_grid.shape
            assert not np.array_equal(
                input_grid,
                output_grid,
            )
            assert not _matches_source_input(input_grid)

            input_key = _grid_key(input_grid)
            pair_key = (
                input_key,
                _grid_key(output_grid),
            )

            assert input_key not in input_keys
            assert pair_key not in pair_keys

            input_keys.add(input_key)
            pair_keys.add(pair_key)

            info = _analyze_input(input_grid)
            connector = _infer_connector(
                input_grid,
                output_grid,
            )

            connectors.append(connector)

            layout_key = (
                info["height"],
                info["width"],
                info["rows"],
                info["cols"],
            )

            assert layout_key not in layout_keys
            layout_keys.add(layout_key)

            if regime == "source_train":
                _assert_source_train_structure(info)
                train_infos.append(info)
                train_backgrounds.add(info["background"])
            else:
                _assert_target_test_structure(info)
                test_infos.append(info)
                test_backgrounds.add(info["background"])

    assert len(set(connectors)) == 1
    connector = connectors[0]
    assert test_backgrounds.issubset(train_backgrounds)

    for pair in puzzle["train"]:
        train_input_colors = set(
            int(value)
            for value in np.unique(pair["input"])
        )
        train_output_colors = set(
            int(value)
            for value in np.unique(pair["output"])
        )

        assert connector not in train_input_colors
        assert connector in train_output_colors

    assert any(info["margins"][1] == 0 for info in train_infos)
    horizontal_restart = False
    vertical_restart = False
    for pair, info in zip(puzzle["train"], train_infos):
        background = info["background"]
        top_row, bottom_row = info["rows"]
        left_column, right_column = info["cols"]
        horizontal_gap = info["horizontal_gap"]
        vertical_gap = info["vertical_gap"]
        if horizontal_gap >= 6 and horizontal_gap % 2 == 0:
            observed = [
                int(pair["output"][top_row, column])
                for column in range(left_column + 2, right_column - 1)
            ]
            one_ended = [
                connector if distance % 2 else background
                for distance in range(1, horizontal_gap + 1)
            ]
            horizontal_restart |= observed != one_ended
        if vertical_gap >= 6 and vertical_gap % 2 == 0:
            observed = [
                int(pair["output"][row, left_column])
                for row in range(top_row + 2, bottom_row - 1)
            ]
            one_ended = [
                connector if distance % 2 else background
                for distance in range(1, vertical_gap + 1)
            ]
            vertical_restart |= observed != one_ended
    assert horizontal_restart and vertical_restart

    all_infos = train_infos + test_infos
    assert len({info["horizontal_gap"] for info in all_infos}) >= 2
    assert len({info["vertical_gap"] for info in all_infos}) >= 2

    train_marker_union = set()
    for info in train_infos:
        train_marker_union.update(
            info["marker_colors"]
        )

    assert 2 <= len(train_marker_union) <= 5
    assert not (train_backgrounds & train_marker_union)
    assert connector not in train_marker_union
    assert connector not in train_backgrounds

    for info in test_infos:
        assert set(
            info["marker_colors"]
        ).issubset(train_marker_union)
    assert not (test_backgrounds & train_marker_union)
    assert connector not in test_backgrounds


def _puzzles_equal(
    first: dict,
    second: dict,
) -> bool:
    if set(first.keys()) != set(second.keys()):
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False

        for first_pair, second_pair in zip(
            first[split],
            second[split],
        ):
            if not np.array_equal(
                first_pair["input"],
                second_pair["input"],
            ):
                return False

            if not np.array_equal(
                first_pair["output"],
                second_pair["output"],
            ):
                return False

    return True


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        (
            _grid_key(pair["input"]),
            _grid_key(pair["output"]),
        )
        for split in ("train", "test")
        for pair in puzzle[split]
    )


def self_test() -> None:
    for height, width, markers in _SOURCE_INPUT_SPECS:
        source_input = np.zeros((height, width), dtype=np.int8)
        for row, column, color in markers:
            source_input[row, column] = color
        assert _matches_source_input(source_input)
        info = _analyze_input(source_input)
        assert (height, width, info["rows"], info["cols"]) in _SOURCE_LAYOUT_KEYS

    automatic_puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(puzzle) for puzzle in automatic_puzzles}) == 42
    assert _puzzles_equal(automatic_puzzles[7], generate_puzzle(7))
    assert len({len(puzzle["train"]) for puzzle in automatic_puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in automatic_puzzles}) >= 3

    backgrounds = set()
    connectors = set()
    marker_counts = set()
    background_counts = set()
    horizontal_gaps = set()
    vertical_gaps = set()
    shapes = set()
    layouts = set()
    test_edges = set()
    aspect_types = set()
    familiar_tests = 0
    novel_tests = 0
    smaller_tests = 0
    mixed_background_puzzles = 0

    for puzzle in automatic_puzzles:
        validate_generated_puzzle(puzzle)
        train_infos = [_analyze_input(pair["input"]) for pair in puzzle["train"]]
        test_infos = [_analyze_input(pair["input"]) for pair in puzzle["test"]]
        train_markers = set().union(*(info["marker_colors"] for info in train_infos))
        train_pairs = {info["marker_colors"] for info in train_infos}
        train_backgrounds = {info["background"] for info in train_infos}
        marker_counts.add(len(train_markers))
        background_counts.add(len(train_backgrounds))
        if len(train_backgrounds) > 1:
            mixed_background_puzzles += 1
        for info in test_infos:
            assert info["marker_colors"].issubset(train_markers)
            assert info["background"] in train_backgrounds
            if info["marker_colors"] in train_pairs:
                familiar_tests += 1
            else:
                novel_tests += 1
            if info["height"] <= max(i["height"] for i in train_infos) or info["width"] <= max(i["width"] for i in train_infos):
                smaller_tests += 1
            for side, margin in zip(("top", "bottom", "left", "right"), info["margins"]):
                if margin == 0:
                    test_edges.add(side)

        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_input(pair["input"])
                connector = _infer_connector(pair["input"], pair["output"])
                assert np.array_equal(pair["output"], _transform(pair["input"], connector))
                assert not _matches_source_input(pair["input"])
                backgrounds.add(info["background"])
                connectors.add(connector)
                horizontal_gaps.add(info["horizontal_gap"])
                vertical_gaps.add(info["vertical_gap"])
                shapes.add((info["height"], info["width"]))
                layouts.add((info["height"], info["width"], info["rows"], info["cols"]))
                if info["horizontal_gap"] >= 2 * info["vertical_gap"]:
                    aspect_types.add("wide")
                elif info["vertical_gap"] >= 2 * info["horizontal_gap"]:
                    aspect_types.add("tall")
                else:
                    aspect_types.add("balanced")

    assert len(backgrounds) >= 7
    assert len(connectors) >= 7
    assert marker_counts == {2, 3, 4, 5}
    assert background_counts == {1, 2, 3}
    assert set(range(1, 16)).issubset(horizontal_gaps | vertical_gaps)
    assert max(horizontal_gaps) >= 18 and max(vertical_gaps) >= 18
    assert len(shapes) >= 100
    assert len(layouts) >= 150
    assert test_edges == {"top", "bottom", "left", "right"}
    assert aspect_types == {"wide", "tall", "balanced"}
    assert familiar_tests > 0 and novel_tests > 0
    assert smaller_tests > 0
    assert mixed_background_puzzles >= 5

    explicit_first = generate_puzzle(
        1001,
        6,
        3,
    )
    explicit_second = generate_puzzle(
        1001,
        6,
        3,
    )

    assert _puzzles_equal(
        explicit_first,
        explicit_second,
    )
    validate_generated_puzzle(explicit_first)

    different_seed = generate_puzzle(
        1002,
        6,
        3,
    )
    assert not _puzzles_equal(
        explicit_first,
        different_seed,
    )

    legal_requests = (
        (2001, 2, 1),
        (2002, 10, 1),
        (2003, 10, 4),
        (2004, None, 1),
        (2005, None, 4),
        (2006, 2, None),
        (2007, 10, None),
    )

    for seed, train_count, test_count in legal_requests:
        puzzle = generate_puzzle(
            seed,
            train_count,
            test_count,
        )
        validate_generated_puzzle(puzzle)

        if train_count is not None:
            assert len(puzzle["train"]) == train_count

        if test_count is not None:
            assert len(puzzle["test"]) == test_count

    illegal_requests = (
        (0, 1, None),
        (0, 11, None),
        (0, None, 0),
        (0, None, 5),
        (0, 2, 2),
        (0, 3, 4),
    )

    for seed, train_count, test_count in illegal_requests:
        try:
            generate_puzzle(
                seed,
                train_count,
                test_count,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "illegal count request did not raise ValueError"
            )
