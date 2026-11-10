import numpy as np


_SOURCE_SPECS = (
    # height, width, orientation, first coordinate, spacing,
    # first color, second color, first border, second border
    (22, 9, "horizontal", 5, 2, 2, 3, 0, 8),
    (7, 23, "vertical", 5, 3, 1, 3, 0, 6),
    (24, 8, "horizontal", 7, 4, 4, 1, 0, 0),
    (10, 25, "vertical", 5, 2, 2, 8, 0, 9),
    (11, 27, "vertical", 5, 5, 3, 4, 0, 10),
)


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _normalize_count(
    value: int | None,
    name: str,
    minimum: int,
    maximum: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}"
        )
    return value


def _select_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _normalize_count(num_train, "num_train", 2, 10)
    num_test = _normalize_count(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        # This deterministic cycle guarantees broad count diversity over
        # consecutive seeds while always respecting train > test.
        num_test = 1 + seed % 4
        minimum_train = max(2, num_test + 1)
        span = 11 - minimum_train
        offset = seed // 4 + 2 * (seed % 4)
        num_train = minimum_train + offset % span
    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
    elif num_test is None:
        maximum_test = min(4, num_train - 1)
        num_test = int(rng.integers(1, maximum_test + 1))

    if not (
        2 <= num_train <= 10
        and 1 <= num_test <= 4
        and num_train > num_test
    ):
        raise ValueError(
            "counts must satisfy 2 <= num_train <= 10, "
            "1 <= num_test <= 4, and num_train > num_test"
        )

    return num_train, num_test


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    background = int(rng.integers(0, 10))
    return {
        "background": background,
        "background_mode": "stable" if rng.random() < 0.25 else "varied",
        "palette_mode": rng.choice(("stable", "reuse", "free"), p=(0.25, 0.3, 0.45)),
        "phase_colors": tuple(int(value) for value in rng.choice(
            [color for color in range(10) if color != background], size=2, replace=False,
        )),
    }


def _paint_output(
    shape: tuple[int, int],
    background: int,
    orientation: str,
    first_coordinate: int,
    spacing: int,
    first_color: int,
    second_color: int,
    dtype: np.dtype = np.int8,
) -> np.ndarray:
    output = np.full(shape, background, dtype=dtype)
    axis_length = shape[1] if orientation == "vertical" else shape[0]

    for phase_index, coordinate in enumerate(
        range(first_coordinate, axis_length, spacing)
    ):
        color = first_color if phase_index % 2 == 0 else second_color
        if orientation == "vertical":
            output[:, coordinate] = color
        else:
            output[coordinate, :] = color

    return output


def _make_source_pair(
    spec: tuple,
) -> tuple[np.ndarray, np.ndarray]:
    (
        height,
        width,
        orientation,
        first_coordinate,
        spacing,
        first_color,
        second_color,
        first_border,
        second_border,
    ) = spec

    input_grid = np.zeros((height, width), dtype=np.int8)
    second_coordinate = first_coordinate + spacing

    if orientation == "vertical":
        input_grid[first_border, first_coordinate] = first_color
        input_grid[second_border, second_coordinate] = second_color
    else:
        input_grid[first_coordinate, first_border] = first_color
        input_grid[second_coordinate, second_border] = second_color

    output_grid = _paint_output(
        input_grid.shape,
        0,
        orientation,
        first_coordinate,
        spacing,
        first_color,
        second_color,
        dtype=np.int8,
    )
    return input_grid, output_grid


def _source_pairs() -> list[tuple[np.ndarray, np.ndarray]]:
    return [_make_source_pair(spec) for spec in _SOURCE_SPECS]


def _is_exact_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    for source_input, source_output in _source_pairs():
        if (
            np.array_equal(input_grid, source_input)
            and np.array_equal(output_grid, source_output)
        ):
            return True
    return False


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    orientation: str,
    spacing: int,
    border_relation: str,
    background: int,
    phase_colors: tuple[int, int],
    forward_witness: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown pair regime")
    if orientation not in ("horizontal", "vertical") or not 1 <= spacing <= 9:
        raise ValueError("invalid stripe geometry")
    if border_relation not in ("same", "opposite"):
        raise ValueError("invalid border relation")

    minimum_length = (4 if forward_witness else 2) * spacing + 1
    if minimum_length > 30:
        raise ValueError("spacing cannot fit the requested demonstration")
    repetition_length = (
        30 if rng.random() < 0.15
        else int(rng.integers(minimum_length, 31))
    )
    cross_length = (
        30 if rng.random() < 0.15
        else int(rng.integers(3, 31))
    )
    if orientation == "vertical":
        height, width = cross_length, repetition_length
    else:
        height, width = repetition_length, cross_length

    minimum_start = spacing if forward_witness else 0
    maximum_start = repetition_length - 1 - (3 if forward_witness else 2) * spacing
    if not forward_witness and rng.random() < 0.38:
        minimum_start = max(minimum_start, repetition_length - 3 * spacing)
    first_coordinate = int(rng.integers(minimum_start, maximum_start + 1))
    second_coordinate = first_coordinate + spacing

    first_color, second_color = phase_colors
    assert len({background, first_color, second_color}) == 3

    first_side_code = int(rng.integers(0, 2))
    if border_relation == "same":
        second_side_code = first_side_code
    else:
        second_side_code = 1 - first_side_code

    first_border = (
        0 if first_side_code == 0 else cross_length - 1
    )
    second_border = (
        0 if second_side_code == 0 else cross_length - 1
    )

    input_grid = np.full(
        (height, width), background, dtype=np.int8
    )

    if orientation == "vertical":
        input_grid[first_border, first_coordinate] = first_color
        input_grid[second_border, second_coordinate] = second_color
    else:
        input_grid[first_coordinate, first_border] = first_color
        input_grid[second_coordinate, second_border] = second_color

    output_grid = _paint_output(
        input_grid.shape,
        background,
        orientation,
        first_coordinate,
        spacing,
        first_color,
        second_color,
        dtype=np.int8,
    )
    return input_grid, output_grid


def _grid_value_signature(grid: np.ndarray) -> tuple:
    normalized = np.asarray(grid, dtype=np.int8)
    return tuple(grid.shape), normalized.tobytes(order="C")


def _pair_value_signature(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple:
    return (
        _grid_value_signature(input_grid),
        _grid_value_signature(output_grid),
    )


def _analyze_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> dict:
    assert input_grid.shape == output_grid.shape
    height, width = input_grid.shape

    values, counts = np.unique(input_grid, return_counts=True)
    assert values.size == 3

    background_indices = np.flatnonzero(
        counts == input_grid.size - 2
    )
    assert background_indices.size == 1
    background = int(values[int(background_indices[0])])

    seed_coordinates = np.argwhere(input_grid != background)
    assert seed_coordinates.shape == (2, 2)

    seed_values = [
        int(input_grid[tuple(coordinate)])
        for coordinate in seed_coordinates
    ]
    assert seed_values[0] != seed_values[1]
    assert seed_values[0] != background
    assert seed_values[1] != background

    rows = seed_coordinates[:, 0]
    columns = seed_coordinates[:, 1]

    vertical_candidate = bool(
        np.all((rows == 0) | (rows == height - 1))
        and columns[0] != columns[1]
    )
    horizontal_candidate = bool(
        np.all((columns == 0) | (columns == width - 1))
        and rows[0] != rows[1]
    )
    assert vertical_candidate != horizontal_candidate

    if vertical_candidate:
        orientation = "vertical"
        axis_coordinates = columns
        side_coordinates = rows
        axis_length = width
        cross_length = height
    else:
        orientation = "horizontal"
        axis_coordinates = rows
        side_coordinates = columns
        axis_length = height
        cross_length = width

    order = np.argsort(axis_coordinates)
    ordered_seeds = seed_coordinates[order]
    ordered_axis = axis_coordinates[order]
    ordered_sides = side_coordinates[order]

    first_coordinate = int(ordered_axis[0])
    second_coordinate = int(ordered_axis[1])
    spacing = second_coordinate - first_coordinate
    assert spacing > 0
    assert 3 <= (axis_length - 1 - first_coordinate) // spacing + 1

    first_color = int(input_grid[tuple(ordered_seeds[0])])
    second_color = int(input_grid[tuple(ordered_seeds[1])])
    assert first_color != second_color

    expected_output = _paint_output(
        input_grid.shape,
        background,
        orientation,
        first_coordinate,
        spacing,
        first_color,
        second_color,
        dtype=input_grid.dtype,
    )
    assert np.array_equal(output_grid, expected_output)
    assert not np.array_equal(input_grid, output_grid)

    expected_colors = {first_color, second_color}
    if first_coordinate > 0 or spacing > 1:
        expected_colors.add(background)
    actual_output_colors = {
        int(value) for value in np.unique(output_grid)
    }
    assert actual_output_colors == expected_colors

    stripe_count = (
        (axis_length - 1 - first_coordinate) // spacing + 1
    )
    border_relation = (
        "same"
        if int(ordered_sides[0]) == int(ordered_sides[1])
        else "opposite"
    )

    return {
        "background": background,
        "orientation": orientation,
        "height": height,
        "width": width,
        "axis_length": axis_length,
        "cross_length": cross_length,
        "first_coordinate": first_coordinate,
        "second_coordinate": second_coordinate,
        "spacing": spacing,
        "first_color": first_color,
        "second_color": second_color,
        "sides": (
            int(ordered_sides[0]),
            int(ordered_sides[1]),
        ),
        "border_relation": border_relation,
        "stripe_count": stripe_count,
    }


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _normalize_seed(seed)
    rng = np.random.default_rng(seed)

    num_train, num_test = _select_counts(
        seed, rng, num_train, num_test
    )
    puzzle_parameters = sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    seen_inputs = set()
    seen_pairs = set()

    first_orientation = str(rng.choice(("horizontal", "vertical")))
    other_orientation = (
        "vertical" if first_orientation == "horizontal" else "horizontal"
    )
    mixed_training = bool(rng.random() < 0.65)
    orientations = [first_orientation, other_orientation if mixed_training else first_orientation]
    orientations.extend(
        str(rng.choice(("horizontal", "vertical"))) if mixed_training else first_orientation
        for _ in range(num_train - 2)
    )

    anchor_spacings = tuple(int(value) for value in rng.choice((2, 3, 4), size=2, replace=False))
    train_plan = []
    for index, orientation in enumerate(orientations):
        spacing = anchor_spacings[index] if index < 2 else int(rng.integers(1, 8))
        train_plan.append((orientation, spacing, index < 2))

    base_background = puzzle_parameters["background"]
    base_phases = puzzle_parameters["phase_colors"]
    previous_phases = base_phases
    pair_index = 0

    def choose_colors() -> tuple[int, tuple[int, int]]:
        nonlocal pair_index, previous_phases
        if puzzle_parameters["palette_mode"] == "stable":
            phases = base_phases
        elif puzzle_parameters["palette_mode"] == "reuse" and pair_index and rng.random() < 0.7:
            phases = previous_phases
        else:
            phases = tuple(int(value) for value in rng.choice(10, size=2, replace=False))
        if rng.random() < 0.45:
            phases = phases[::-1]
        if puzzle_parameters["background_mode"] == "stable":
            background = base_background
            if background in phases:
                phases = tuple(
                    int(value) for value in rng.choice(
                        [color for color in range(10) if color != background],
                        size=2, replace=False,
                    )
                )
        else:
            choices = [color for color in range(10) if color not in phases]
            if pair_index == 0 and base_background in choices:
                background = base_background
            elif pair_index == 1:
                background = int(rng.choice([color for color in choices if color != base_background]))
            else:
                background = int(rng.choice(choices))
        previous_phases = phases
        pair_index += 1
        return background, phases

    def make_unique_pair(
        regime: str,
        orientation: str,
        spacing: int,
        border_relation: str,
        forward_witness: bool,
    ) -> dict:
        background, phase_colors = choose_colors()
        for _ in range(64):
            child_seed = int(
                rng.integers(
                    0,
                    np.iinfo(np.int64).max,
                    dtype=np.int64,
                )
            )
            pair_rng = np.random.default_rng(child_seed)
            input_grid, output_grid = _generate_pair(
                pair_rng,
                regime,
                orientation,
                spacing,
                border_relation,
                background,
                phase_colors,
                forward_witness,
            )

            input_signature = _grid_value_signature(input_grid)
            pair_signature = _pair_value_signature(
                input_grid, output_grid
            )
            if input_signature in seen_inputs:
                continue
            if pair_signature in seen_pairs:
                continue
            if _is_exact_source_pair(input_grid, output_grid):
                continue

            _analyze_pair(input_grid, output_grid)
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)
            return {
                "input": input_grid,
                "output": output_grid,
            }

        raise ValueError("unable to generate a unique valid pair")

    for orientation, spacing, forward_witness in train_plan:
        border_relation = str(rng.choice(("same", "opposite")))
        puzzle["train"].append(
            make_unique_pair(
                "source_train",
                orientation,
                spacing,
                border_relation,
                forward_witness,
            )
        )
    rng.shuffle(puzzle["train"])

    train_spacings = [
        _analyze_pair(pair["input"], pair["output"])["spacing"]
        for pair in puzzle["train"]
    ]
    train_orientations = set(orientations)

    for _ in range(num_test):
        regime_choice = rng.random()
        if regime_choice < 0.38 and max(train_spacings) < 9:
            spacing = int(rng.integers(max(train_spacings) + 1, 10))
        elif regime_choice < 0.62:
            spacing = int(rng.choice(train_spacings))
        elif regime_choice < 0.79 and min(train_spacings) > 1:
            spacing = int(rng.integers(1, min(train_spacings)))
        else:
            spacing = int(rng.integers(1, 10))
        if len(train_orientations) == 1 and rng.random() < 0.38:
            orientation = other_orientation
        else:
            orientation = str(rng.choice(("horizontal", "vertical")))
        puzzle["test"].append(
            make_unique_pair(
                "target_test",
                orientation,
                spacing,
                str(rng.choice(("same", "opposite"))),
                False,
            )
        )
    rng.shuffle(puzzle["test"])

    validate_generated_puzzle(puzzle)
    return puzzle


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

    seen_inputs = set()
    seen_pairs = set()
    metadata_by_split = {"train": [], "test": []}

    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {"input", "output"}

            input_grid = pair["input"]
            output_grid = pair["output"]

            for grid in (input_grid, output_grid):
                assert isinstance(grid, np.ndarray)
                assert grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30
                assert 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert int(grid.min()) >= 0
                assert int(grid.max()) <= 9

            input_signature = _grid_value_signature(input_grid)
            pair_signature = _pair_value_signature(
                input_grid, output_grid
            )
            assert input_signature not in seen_inputs
            assert pair_signature not in seen_pairs
            seen_inputs.add(input_signature)
            seen_pairs.add(pair_signature)

            assert not _is_exact_source_pair(
                input_grid, output_grid
            )
            metadata_by_split[split].append(
                _analyze_pair(input_grid, output_grid)
            )

    train_metadata = metadata_by_split["train"]
    test_metadata = metadata_by_split["test"]
    assert len({metadata["spacing"] for metadata in train_metadata}) >= 2
    witnesses = [
        metadata for metadata in train_metadata
        if metadata["first_coordinate"] >= metadata["spacing"]
        and metadata["stripe_count"] >= 4
    ]
    assert witnesses
    for metadata in train_metadata + test_metadata:
        assert 3 <= metadata["height"] <= 30
        assert 3 <= metadata["width"] <= 30
        assert 1 <= metadata["spacing"] <= 9
        assert metadata["stripe_count"] >= 3
        assert metadata["first_coordinate"] >= 0
        assert metadata["border_relation"] in ("same", "opposite")


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first.keys()) != {"train", "test"}:
        return False
    if set(second.keys()) != {"train", "test"}:
        return False

    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for first_pair, second_pair in zip(
            first[split], second[split]
        ):
            if not np.array_equal(
                first_pair["input"], second_pair["input"]
            ):
                return False
            if not np.array_equal(
                first_pair["output"], second_pair["output"]
            ):
                return False
    return True


def _puzzle_fingerprint(puzzle: dict) -> tuple:
    pieces = []
    for split in ("train", "test"):
        pieces.append((split, len(puzzle[split])))
        for pair in puzzle[split]:
            pieces.append(
                _pair_value_signature(
                    pair["input"], pair["output"]
                )
            )
    return tuple(pieces)


def _predict_from_training(train_pairs: list[dict], test_input: np.ndarray) -> np.ndarray:
    def decode(input_grid: np.ndarray) -> tuple:
        colors, counts = np.unique(input_grid, return_counts=True)
        background = int(colors[np.argmax(counts)])
        seeds = np.argwhere(input_grid != background)
        height, width = input_grid.shape
        on_rows = all(row in (0, height - 1) for row in seeds[:, 0])
        on_columns = all(column in (0, width - 1) for column in seeds[:, 1])
        vertical = on_rows and seeds[0, 1] != seeds[1, 1]
        horizontal = on_columns and seeds[0, 0] != seeds[1, 0]
        assert vertical != horizontal
        axis = 1 if vertical else 0
        seeds = sorted(seeds, key=lambda position: position[axis])
        start = int(seeds[0][axis])
        spacing = int(seeds[1][axis] - start)
        phase_colors = tuple(int(input_grid[tuple(position)]) for position in seeds)
        return background, axis, start, spacing, phase_colors

    witnesses = 0
    for pair in train_pairs:
        background, axis, start, spacing, phase_colors = decode(pair["input"])
        if start >= spacing:
            preceding = start - spacing
            line = pair["output"][:, preceding] if axis == 1 else pair["output"][preceding, :]
            assert np.all(line == background)
            witnesses += 1
    assert witnesses

    background, axis, start, spacing, phase_colors = decode(test_input)
    prediction = np.full_like(test_input, background)
    for index, coordinate in enumerate(range(start, test_input.shape[axis], spacing)):
        if axis == 1:
            prediction[:, coordinate] = phase_colors[index % 2]
        else:
            prediction[coordinate, :] = phase_colors[index % 2]
    return prediction


def self_test() -> None:
    source_four_input, source_four_output = _source_pairs()[3]
    assert [(int(row), int(column), int(source_four_input[row, column]))
            for row, column in np.argwhere(source_four_input != 0)] == [(0, 5, 2), (9, 7, 8)]
    assert np.array_equal(source_four_output[0], np.array(
        [int(value) for value in "0000020802080208020802080"], dtype=np.int8,
    ))

    counts = {"train": set(), "test": set()}
    fingerprints = set()
    backgrounds = set()
    shapes = set()
    layouts = set()
    palettes = set()
    spacings = set()
    aspect_types = set()
    stripe_counts = set()
    varying_backgrounds = stable_backgrounds = reused_palettes = 0
    same_orientation = mixed_orientation = horizontal_tests = 0
    challenging_tests = familiar_tests = smaller_tests = 0
    same_orientation_both_relations = 0
    forward_by_both = 0
    corner_seeds = 0
    both_first_sides = reversed_palettes = combined_variations = 0

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_puzzle_fingerprint(puzzle))
        counts["train"].add(len(puzzle["train"]))
        counts["test"].add(len(puzzle["test"]))
        train = [_analyze_pair(pair["input"], pair["output"]) for pair in puzzle["train"]]
        test = [_analyze_pair(pair["input"], pair["output"]) for pair in puzzle["test"]]
        all_pairs = train + test
        orientations = {item["orientation"] for item in train}
        same_orientation += len(orientations) == 1
        mixed_orientation += len(orientations) == 2
        horizontal_tests += any(item["orientation"] == "horizontal" for item in test)
        backgrounds_here = {item["background"] for item in all_pairs}
        varying_backgrounds += len(backgrounds_here) > 1
        stable_backgrounds += len(backgrounds_here) == 1
        phase_sets = [tuple(sorted((item["first_color"], item["second_color"]))) for item in all_pairs]
        reused_palettes += len(set(phase_sets)) < len(phase_sets)
        ordered_palettes = {(item["first_color"], item["second_color"]) for item in all_pairs}
        reversed_palettes += any((second, first) in ordered_palettes for first, second in ordered_palettes)
        both_first_sides += len({item["sides"][0] == 0 for item in all_pairs}) == 2
        combined_variations += (
            len(backgrounds_here) > 1
            and any(item["stripe_count"] == 3 and item["axis_length"] < item["cross_length"] for item in all_pairs)
            and len({item["orientation"] for item in all_pairs}) == 2
        )
        witness_orientations = {
            item["orientation"] for item in train
            if item["first_coordinate"] >= item["spacing"]
        }
        forward_by_both += len(witness_orientations) == 2
        same_orientation_both_relations += any(
            {item["border_relation"] for item in train if item["orientation"] == orientation}
            == {"same", "opposite"} for orientation in orientations
        )
        for pair in puzzle["train"] + puzzle["test"]:
            predicted = _predict_from_training(puzzle["train"], pair["input"].copy())
            assert np.array_equal(predicted, pair["output"])
        for item in all_pairs:
            backgrounds.add(item["background"])
            shapes.add((item["height"], item["width"]))
            layouts.add((item["orientation"], item["height"], item["width"],
                         item["first_coordinate"], item["spacing"], item["border_relation"], item["sides"]))
            palettes.add((item["background"], item["first_color"], item["second_color"]))
            spacings.add(item["spacing"])
            stripe_counts.add(item["stripe_count"])
            axis = item["axis_length"]
            cross = item["cross_length"]
            aspect_types.add("square" if axis == cross else "long_axis" if axis > cross else "short_axis")
            corner_seeds += item["first_coordinate"] == 0
        for item in test:
            challenging_tests += item["spacing"] > max(train_item["spacing"] for train_item in train)
            familiar_tests += item["spacing"] in {train_item["spacing"] for train_item in train}
            smaller_tests += item["spacing"] < min(train_item["spacing"] for train_item in train)

    assert len(fingerprints) >= 40
    assert len(counts["train"]) >= 4 and len(counts["test"]) >= 3
    assert len(backgrounds) >= 8 and len(shapes) >= 70 and len(layouts) >= 100
    assert len(palettes) >= 50 and spacings == set(range(1, 10))
    assert {3, 4}.issubset(stripe_counts) and max(stripe_counts) >= 8
    assert aspect_types == {"square", "long_axis", "short_axis"}
    assert (30, 30) in shapes and min(min(shape) for shape in shapes) <= 5
    assert varying_backgrounds >= 10 and stable_backgrounds >= 3 and reused_palettes >= 8
    assert same_orientation >= 3 and mixed_orientation >= 10 and horizontal_tests >= 10
    assert min(challenging_tests, familiar_tests, smaller_tests) >= 3
    assert same_orientation_both_relations >= 5 and forward_by_both >= 10
    assert corner_seeds >= 3
    assert both_first_sides >= 10 and reversed_palettes >= 5 and combined_variations >= 5

    for index, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(40000 + index, train_count, test_count)
        validate_generated_puzzle(puzzle)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)

    for index, arguments in enumerate((
        {"num_train": None, "num_test": 1}, {"num_train": None, "num_test": 4},
        {"num_train": 2, "num_test": None}, {"num_train": 10, "num_test": None},
        {"num_train": 7, "num_test": 3}, {},
    )):
        first = generate_puzzle(50000 + index, **arguments)
        second = generate_puzzle(50000 + index, **arguments)
        validate_generated_puzzle(first)
        assert _puzzles_equal(first, second)
        for name, split in (("num_train", "train"), ("num_test", "test")):
            assert arguments.get(name) is None or len(first[split]) == arguments[name]
    assert not _puzzles_equal(generate_puzzle(9001), generate_puzzle(9002))

    for arguments in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
        {"num_train": 4, "num_test": 4}, {"num_train": 2.5, "num_test": 1},
        {"num_train": 5, "num_test": True},
    ):
        try:
            generate_puzzle(60000, **arguments)
        except ValueError:
            continue
        raise AssertionError(f"accepted illegal counts: {arguments}")
