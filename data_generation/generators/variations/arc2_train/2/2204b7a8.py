import numpy as np


_SOURCE_INPUT_ROWS = {
    (
        "4444444444",
        "0000000000",
        "0300000300",
        "0003000000",
        "0000000000",
        "0000000000",
        "0000300000",
        "0000000030",
        "0000000000",
        "7777777777",
    ),
    (
        "1000000002",
        "1000003002",
        "1000000002",
        "1000000002",
        "1000300002",
        "1000000002",
        "1030000002",
        "1000000002",
        "1000000002",
        "1000000002",
    ),
    (
        "8888888888",
        "0000000000",
        "0000300000",
        "0300000300",
        "0000000000",
        "0000000000",
        "0300003000",
        "0003000000",
        "0000000000",
        "9999999999",
    ),
    (
        "5300000004",
        "5000030034",
        "5000000004",
        "5003000004",
        "5000003004",
        "5003000004",
        "5000000004",
        "5000300004",
        "5030003004",
        "5000000004",
    ),
}


def _checked_count(value, name, lower, upper):
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


def _resolve_counts(rng, num_train, num_test):
    num_train = _checked_count(num_train, "num_train", 2, 10)
    num_test = _checked_count(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        largest_test = min(4, num_train - 1)
        if largest_test < 1:
            raise ValueError("no legal num_test exists for this num_train")
        num_test = int(rng.integers(1, largest_test + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return num_train, num_test


def _sample_puzzle_parameters(rng):
    palette = tuple(int(x) for x in rng.permutation(10))
    return {
        "palette_mode": int(rng.integers(0, 3)),
        "base_colors": palette[:4],
    }


def _sample_dimensions(rng, regime):
    assert regime in ("source_train", "target_test")
    roll = float(rng.random())
    if roll < 0.16:
        return 10, 10
    if roll < 0.36:
        return int(rng.integers(4, 10)), int(rng.integers(4, 10))
    if roll < 0.66:
        return int(rng.integers(8, 22)), int(rng.integers(8, 22))
    if roll < (0.83 if regime == "source_train" else 0.79):
        return int(rng.integers(20, 31)), int(rng.integers(20, 31))
    short = int(rng.integers(4, 9))
    long = int(rng.integers(18, 31))
    return (short, long) if rng.random() < 0.5 else (long, short)


def _candidate_markers(height, width, orientation, edge_mode, side_mode):
    candidates = []
    for row in range(height):
        for col in range(width):
            if orientation == "horizontal":
                if row == 0 or row == height - 1:
                    continue
                first_distance = row
                second_distance = height - 1 - row
                edge = 0 if col == 0 else 1 if col == width - 1 else -1
            else:
                if col == 0 or col == width - 1:
                    continue
                first_distance = col
                second_distance = width - 1 - col
                edge = 0 if row == 0 else 1 if row == height - 1 else -1

            if first_distance == second_distance:
                continue

            side = 0 if first_distance < second_distance else 1
            if side_mode in (0, 1) and side != side_mode:
                continue
            if edge_mode == 0 and edge != -1:
                continue
            if edge_mode == 1 and edge == 1:
                continue
            if edge_mode == 2 and edge == 0:
                continue
            candidates.append((row, col, side, edge))

    return candidates


def _select_marker_positions(
    rng,
    height,
    width,
    orientation,
    marker_count,
    edge_mode,
    side_mode,
):
    candidates = _candidate_markers(
        height, width, orientation, edge_mode, side_mode,
    )
    requirements = []
    if edge_mode in (1, 3):
        requirements.append(("edge", 0))
    if edge_mode in (2, 3):
        requirements.append(("edge", 1))
    if side_mode == 2:
        requirements.extend((("side", 0), ("side", 1)))

    best = []
    for _ in range(48):
        order = [candidates[int(i)] for i in rng.permutation(len(candidates))]
        chosen = []

        def compatible(candidate):
            row, col = candidate[:2]
            return all(abs(row - old[0]) + abs(col - old[1]) > 1 for old in chosen)

        for kind, value in requirements:
            if any(item[2 if kind == "side" else 3] == value for item in chosen):
                continue
            choice = next(
                (item for item in order if item[2 if kind == "side" else 3] == value and compatible(item)),
                None,
            )
            if choice is None:
                break
            chosen.append(choice)
        else:
            for item in order:
                if len(chosen) >= marker_count:
                    break
                if item not in chosen and compatible(item):
                    chosen.append(item)
            if len(chosen) > len(best):
                best = chosen
            if len(best) >= marker_count:
                break

    if not best:
        chosen = []

        def search(index):
            if index == len(requirements):
                return True
            kind, value = requirements[index]
            field = 2 if kind == "side" else 3
            if any(item[field] == value for item in chosen):
                return search(index + 1)
            for item in candidates:
                if item[field] == value and compatible(item):
                    chosen.append(item)
                    if search(index + 1):
                        return True
                    chosen.pop()
            return False

        assert search(0), "no feasible marker layout"
        for item in candidates:
            if len(chosen) >= marker_count:
                break
            if item not in chosen and compatible(item):
                chosen.append(item)
        best = chosen

    assert best, "no valid marker layout for the sampled dimensions"
    return [(row, col) for row, col, _, _ in best]


def _find_boundary_lines(grid):
    candidates = []

    top_color = int(grid[0, 0])
    bottom_color = int(grid[-1, 0])
    if (
        bool(np.all(grid[0, :] == top_color))
        and bool(np.all(grid[-1, :] == bottom_color))
        and top_color != bottom_color
    ):
        candidates.append(("horizontal", top_color, bottom_color))

    left_color = int(grid[0, 0])
    right_color = int(grid[0, -1])
    if (
        bool(np.all(grid[:, 0] == left_color))
        and bool(np.all(grid[:, -1] == right_color))
        and left_color != right_color
    ):
        candidates.append(("vertical", left_color, right_color))

    assert len(candidates) == 1, (
        "there must be exactly one opposite-line orientation"
    )
    return candidates[0]


def _solve_visible_input(input_grid):
    assert isinstance(input_grid, np.ndarray)
    assert input_grid.ndim == 2

    height, width = input_grid.shape
    orientation, first_line_color, second_line_color = (
        _find_boundary_lines(input_grid)
    )

    field_mask = np.ones((height, width), dtype=bool)
    if orientation == "horizontal":
        field_mask[0, :] = False
        field_mask[-1, :] = False
    else:
        field_mask[:, 0] = False
        field_mask[:, -1] = False

    field_values, field_counts = np.unique(
        input_grid[field_mask],
        return_counts=True,
    )
    assert len(field_values) == 2, (
        "the field must contain background and markers"
    )
    assert int(field_counts[0]) != int(field_counts[1]), (
        "sparse marker and background roles must be unambiguous"
    )

    background_index = int(np.argmax(field_counts))
    marker_index = 1 - background_index
    background = int(field_values[background_index])
    marker = int(field_values[marker_index])
    background_count = int(field_counts[background_index])
    marker_count = int(field_counts[marker_index])

    assert background_count > marker_count >= 1
    assert len(
        {
            background,
            marker,
            first_line_color,
            second_line_color,
        }
    ) == 4, "all four semantic roles need distinct colors"

    marker_mask = field_mask & (input_grid == marker)
    marker_positions_array = np.argwhere(marker_mask)
    assert len(marker_positions_array) == marker_count

    expected = input_grid.copy()
    side_counts = [0, 0]
    marker_positions = []

    for raw_row, raw_col in marker_positions_array:
        row = int(raw_row)
        col = int(raw_col)
        marker_positions.append((row, col))

        for delta_row, delta_col in (
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
        ):
            other_row = row + delta_row
            other_col = col + delta_col
            if (
                0 <= other_row < height
                and 0 <= other_col < width
                and bool(marker_mask[other_row, other_col])
            ):
                raise AssertionError("marker cells must be isolated")

        if orientation == "horizontal":
            first_distance = row
            second_distance = height - 1 - row
        else:
            first_distance = col
            second_distance = width - 1 - col

        assert first_distance != second_distance, (
            "every marker must be strictly nearer to one boundary"
        )

        if first_distance < second_distance:
            expected[row, col] = first_line_color
            side_counts[0] += 1
        else:
            expected[row, col] = second_line_color
            side_counts[1] += 1

    expected_colors = {
        background,
        first_line_color,
        second_line_color,
    }
    assert {
        int(x) for x in np.unique(expected)
    } == expected_colors

    metadata = {
        "orientation": orientation,
        "background": background,
        "marker": marker,
        "line_pair": (
            first_line_color,
            second_line_color,
        ),
        "marker_count": marker_count,
        "marker_positions": tuple(marker_positions),
        "side_counts": tuple(side_counts),
        "top_edge_marker": any(
            row == 0 for row, _ in marker_positions
        ),
        "other_edge_marker": (
            any(
                col in (0, width - 1)
                for _, col in marker_positions
            )
            if orientation == "horizontal"
            else any(
                row in (0, height - 1)
                for row, _ in marker_positions
            )
        ),
        "shape": (height, width),
    }
    return expected, metadata


def _make_pair(
    rng,
    regime,
    puzzle_parameters,
    orientation,
    previous_colors,
    required_side,
):
    height, width = _sample_dimensions(rng, regime)
    field_area = (height - 2) * width if orientation == "horizontal" else height * (width - 2)
    max_count = min(60, max(1, field_area // 6))
    roll = float(rng.random())
    if roll < 0.12:
        marker_count = 1
    elif roll < 0.24:
        marker_count = 2
    elif roll < 0.52:
        marker_count = int(rng.integers(3, 7))
    elif roll < 0.76:
        marker_count = int(rng.integers(7, 13))
    else:
        marker_count = int(rng.integers(13, max(14, max_count + 1)))
    marker_count = min(marker_count, max_count)

    edge_mode = int(rng.integers(0, 4))
    side_mode = required_side if required_side is not None else int(rng.choice(3, p=(0.27, 0.27, 0.46)))
    if marker_count == 1 and side_mode == 2:
        side_mode = int(rng.integers(0, 2))
    if marker_count == 1 and edge_mode == 3:
        edge_mode = int(rng.integers(1, 3))

    marker_positions = _select_marker_positions(
        rng, height, width, orientation, marker_count,
        edge_mode, side_mode,
    )

    background, marker, first_line_color, second_line_color = _sample_colors(
        rng, puzzle_parameters, previous_colors,
    )

    input_grid = np.full(
        (height, width),
        background,
        dtype=np.int8,
    )

    if orientation == "horizontal":
        input_grid[0, :] = first_line_color
        input_grid[-1, :] = second_line_color
    else:
        input_grid[:, 0] = first_line_color
        input_grid[:, -1] = second_line_color

    for row, col in marker_positions:
        input_grid[row, col] = marker

    output_grid, _ = _solve_visible_input(input_grid)
    return input_grid, output_grid, (background, marker, first_line_color, second_line_color)


def _sample_colors(rng, puzzle_parameters, previous_colors):
    mode = puzzle_parameters["palette_mode"]
    base = puzzle_parameters["base_colors"]
    if mode == 0:
        background, marker = base[:2]
        line_pair = base[2:4] if rng.random() < 0.8 else base[3:1:-1]
    else:
        if mode == 1 and rng.random() < 0.5:
            background, marker = base[:2]
        else:
            background, marker = (int(x) for x in rng.choice(10, size=2, replace=False))
        available = [color for color in range(10) if color not in (background, marker)]
        if previous_colors is not None and rng.random() < (0.45 if mode == 1 else 0.28) and all(
            color in available for color in previous_colors[2:]
        ):
            line_pair = previous_colors[2:]
        elif mode == 1 and rng.random() < 0.35 and all(color in available for color in base[2:]):
            line_pair = base[2:]
        else:
            line_pair = tuple(int(x) for x in rng.choice(available, size=2, replace=False))
    return background, marker, int(line_pair[0]), int(line_pair[1])


def _grid_signature(grid):
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _is_source_input(grid):
    if grid.shape != (10, 10):
        return False

    rows = tuple(
        "".join(
            str(int(value))
            for value in grid[row, :]
        )
        for row in range(10)
    )
    return rows in _SOURCE_INPUT_ROWS


def _generate_fresh_pair(
    pair_rng,
    regime,
    puzzle_parameters,
    orientation,
    previous_colors,
    required_side,
    seen_inputs,
):
    for _ in range(32):
        input_grid, output_grid, colors = _make_pair(
            pair_rng,
            regime,
            puzzle_parameters,
            orientation,
            previous_colors,
            required_side,
        )
        signature = _grid_signature(input_grid)

        if (
            signature in seen_inputs
            or _is_source_input(input_grid)
        ):
            continue

        seen_inputs.add(signature)
        return input_grid, output_grid, colors

    raise RuntimeError("could not construct a fresh valid pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(
        rng,
        num_train,
        num_test,
    )
    puzzle_parameters = _sample_puzzle_parameters(rng)
    orientation_mode = int(rng.integers(0, 6))
    if orientation_mode in (0, 1):
        train_orientations = ["horizontal", "vertical"] + [
            "horizontal" if rng.random() < 0.5 else "vertical"
            for _ in range(num_train - 2)
        ]
        rng.shuffle(train_orientations)
        if orientation_mode == 0:
            test_orientations = ["vertical"] * num_test
        else:
            test_orientations = ["horizontal", "vertical"][:num_test]
            while len(test_orientations) < num_test:
                test_orientations.append("horizontal" if rng.random() < 0.5 else "vertical")
            rng.shuffle(test_orientations)
    elif orientation_mode == 2:
        train_orientations = ["horizontal"] * num_train
        test_orientations = ["horizontal"] * num_test
    elif orientation_mode == 3:
        train_orientations = ["vertical"] * num_train
        test_orientations = ["vertical"] * num_test
    elif orientation_mode == 4:
        train_orientations = ["horizontal"] * num_train
        test_orientations = ["vertical"] * num_test
    else:
        train_orientations = ["vertical"] * num_train
        test_orientations = ["horizontal"] * num_test

    total_pairs = num_train + num_test

    child_seeds = rng.integers(
        0,
        np.iinfo(np.int64).max,
        size=total_pairs,
        dtype=np.int64,
    )

    puzzle = {
        "train": [],
        "test": [],
    }
    seen_inputs = set()
    previous_colors = None

    for index, orientation in enumerate(train_orientations):
        pair_rng = np.random.default_rng(
            int(child_seeds[index])
        )
        input_grid, output_grid, previous_colors = _generate_fresh_pair(
            pair_rng,
            "source_train",
            puzzle_parameters,
            orientation,
            previous_colors,
            index if index < 2 else None,
            seen_inputs,
        )
        puzzle["train"].append(
            {
                "input": input_grid,
                "output": output_grid,
            }
        )

    for test_index, orientation in enumerate(
        test_orientations
    ):
        index = num_train + test_index
        pair_rng = np.random.default_rng(
            int(child_seeds[index])
        )
        input_grid, output_grid, previous_colors = _generate_fresh_pair(
            pair_rng,
            "target_test",
            puzzle_parameters,
            orientation,
            previous_colors,
            None,
            seen_inputs,
        )
        puzzle["test"].append(
            {
                "input": input_grid,
                "output": output_grid,
            }
        )

    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_grid(grid):
    assert isinstance(grid, np.ndarray), (
        "every grid must be a NumPy array"
    )
    assert grid.ndim == 2, (
        "every grid must be two-dimensional"
    )

    height, width = grid.shape
    assert 1 <= height <= 30
    assert 1 <= width <= 30
    assert np.issubdtype(grid.dtype, np.integer), (
        "grid dtype must be integer"
    )
    assert bool(np.all((grid >= 0) & (grid <= 9))), (
        "grid colors must be between 0 and 9"
    )


def _analyze_pair(pair):
    input_grid = pair["input"]
    output_grid = pair["output"]

    assert input_grid.shape == output_grid.shape, (
        "the transformation must preserve dimensions"
    )

    expected_output, metadata = _solve_visible_input(
        input_grid
    )
    assert np.array_equal(output_grid, expected_output), (
        "output does not implement nearest-boundary "
        "marker recoloring"
    )
    assert not np.array_equal(input_grid, output_grid), (
        "every generated pair must contain a visible recoloring"
    )
    assert int(
        np.count_nonzero(input_grid != output_grid)
    ) == metadata["marker_count"]

    return metadata


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    num_train = len(puzzle["train"])
    num_test = len(puzzle["test"])
    assert 2 <= num_train <= 10
    assert 1 <= num_test <= 4
    assert num_train > num_test

    input_signatures = set()
    pair_signatures = set()
    analyses = {
        "train": [],
        "test": [],
    }

    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {
                "input",
                "output",
            }

            _validate_grid(pair["input"])
            _validate_grid(pair["output"])

            input_signature = _grid_signature(
                pair["input"]
            )
            pair_signature = (
                input_signature,
                _grid_signature(pair["output"]),
            )

            assert input_signature not in input_signatures, (
                "duplicate input grid in puzzle"
            )
            assert pair_signature not in pair_signatures, (
                "duplicate input/output pair in puzzle"
            )

            input_signatures.add(input_signature)
            pair_signatures.add(pair_signature)

            assert not _is_source_input(pair["input"]), (
                "a generated pair duplicates a source-task input"
            )
            analyses[split].append(
                _analyze_pair(pair)
            )

    all_metadata = (
        analyses["train"] + analyses["test"]
    )
    train_metadata = analyses["train"]
    assert sum(item["side_counts"][0] for item in train_metadata) > 0, (
        "training must demonstrate recoloring toward the first line"
    )
    assert sum(item["side_counts"][1] for item in train_metadata) > 0, (
        "training must demonstrate recoloring toward the second line"
    )
    for item in all_metadata:
        assert 4 <= min(item["shape"]) <= 30
        assert 1 <= item["marker_count"] < (
            (item["shape"][0] - 2) * item["shape"][1]
            if item["orientation"] == "horizontal"
            else item["shape"][0] * (item["shape"][1] - 2)
        ) // 2


def _puzzles_equal(first, second):
    if set(first.keys()) != {"train", "test"}:
        return False
    if set(second.keys()) != {"train", "test"}:
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


def _puzzle_signature(puzzle):
    return tuple(
        (
            split,
            _grid_signature(pair["input"]),
            _grid_signature(pair["output"]),
        )
        for split in ("train", "test")
        for pair in puzzle[split]
    )


def self_test() -> None:
    puzzles = []
    shapes = set()
    marker_counts = set()
    input_signatures = set()
    orientation_profiles = set()
    edge_patterns = set()
    train_counts = set()
    test_counts = set()
    mixed_parity = 0
    single_side = 0
    local_palette_changes = 0
    repeated_line_assignments = 0
    largest_count = 0
    small_pairs = 0
    long_narrow_pairs = 0
    full_size_pairs = 0

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        train_orientations = set()
        test_orientations = set()
        backgrounds = set()
        markers = set()
        line_pairs = []
        training_sides = [0, 0]

        for split in ("train", "test"):
            for pair in puzzle[split]:
                input_grid, output_grid = pair["input"], pair["output"]
                _, info = _solve_visible_input(input_grid)
                height, width = input_grid.shape
                shapes.add((height, width))
                marker_counts.add(info["marker_count"])
                largest_count = max(largest_count, info["marker_count"])
                input_signatures.add(_grid_signature(input_grid))
                backgrounds.add(info["background"])
                markers.add(info["marker"])
                line_pairs.append(info["line_pair"])
                (train_orientations if split == "train" else test_orientations).add(info["orientation"])
                if split == "train":
                    training_sides[0] += info["side_counts"][0]
                    training_sides[1] += info["side_counts"][1]
                if min(info["side_counts"]) == 0:
                    single_side += 1
                if min(height, width) <= 8:
                    small_pairs += 1
                if max(height, width) >= 20 and min(height, width) <= 8:
                    long_narrow_pairs += 1
                if max(height, width) == 30:
                    full_size_pairs += 1

                positions = info["marker_positions"]
                if len({(row + col) % 2 for row, col in positions}) == 2:
                    mixed_parity += 1
                if info["orientation"] == "horizontal":
                    edges = (any(col == 0 for _, col in positions), any(col == width - 1 for _, col in positions))
                else:
                    edges = (any(row == 0 for row, _ in positions), any(row == height - 1 for row, _ in positions))
                edge_patterns.add((split, info["orientation"], edges))

                # Reconstruct the answer from visible colors and distances, without
                # relying on the generation-time layout or its output routine.
                predicted = input_grid.copy()
                line_colors = (int(input_grid[0, 0]), int(input_grid[-1, 0])) if info["orientation"] == "horizontal" else (int(input_grid[0, 0]), int(input_grid[0, -1]))
                for row, col in positions:
                    coordinate, extent = (row, height) if info["orientation"] == "horizontal" else (col, width)
                    predicted[row, col] = line_colors[0 if coordinate < extent - 1 - coordinate else 1]
                assert np.array_equal(predicted, output_grid)
        assert min(training_sides) > 0
        orientation_profiles.add((frozenset(train_orientations), frozenset(test_orientations)))
        local_palette_changes += len(backgrounds) > 1 and len(markers) > 1
        repeated_line_assignments += len(line_pairs) > len(set(line_pairs))

    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(input_signatures) >= 300 and len(shapes) >= 80
    assert len(marker_counts) >= 20 and largest_count >= 40
    assert mixed_parity >= 50 and single_side >= 40
    assert small_pairs >= 40 and long_narrow_pairs >= 20 and full_size_pairs >= 8
    assert local_palette_changes >= 10 and repeated_line_assignments >= 8
    assert len(orientation_profiles) >= 5
    assert all(any(pattern == wanted for _, _, pattern in edge_patterns) for wanted in ((False, False), (True, False), (False, True), (True, True)))
    assert _puzzles_equal(puzzles[17], generate_puzzle(17))
    assert not _puzzles_equal(puzzles[17], puzzles[18])

    for offset, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4), (6, 3))):
        first = generate_puzzle(1000 + offset, train_count, test_count)
        second = generate_puzzle(1000 + offset, train_count, test_count)
        validate_generated_puzzle(first)
        assert _puzzles_equal(first, second)
        assert (len(first["train"]), len(first["test"])) == (train_count, test_count)

    for offset, (train_count, test_count) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(2000 + offset, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count

    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 3), (2, 4), (4.0, 1), (True, 1)):
        try:
            generate_puzzle(3000, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count arguments must raise ValueError")
