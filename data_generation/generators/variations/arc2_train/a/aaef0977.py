import numpy as np


def _grid(*rows: str) -> np.ndarray:
    return np.array(
        [[int(character) for character in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIRS = (
    (
        _grid(
            "77777737",
            "77777777",
            "77777777",
            "77777777",
            "77777777",
            "77777777",
            "77777777",
            "77777777",
        ),
        _grid(
            "98250434",
            "69825040",
            "16982505",
            "31698252",
            "43169828",
            "04316989",
            "50431696",
            "25043161",
        ),
    ),
    (
        _grid(
            "77777777",
            "77777777",
            "77777777",
            "77777777",
            "77777777",
            "77577777",
            "77777777",
            "77777777",
        ),
        _grid(
            "43134052",
            "31613405",
            "16961340",
            "69896134",
            "98289613",
            "82528961",
            "98289613",
            "69896134",
        ),
    ),
    (
        _grid(
            "77777777",
            "77777777",
            "70777777",
            "77777777",
            "77777777",
            "77777777",
            "77777777",
            "77777777",
        ),
        _grid(
            "82896134",
            "25289613",
            "50528961",
            "25289613",
            "82896134",
            "98961340",
            "69613405",
            "16134052",
        ),
    ),
)


def _checked_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")

    return result


def _resolve_counts(
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _checked_count("num_train", num_train, 2, 10)
    num_test = _checked_count("num_test", num_test, 1, 4)

    if num_train is not None and num_test is not None:
        if num_train <= num_test:
            raise ValueError("num_train must be greater than num_test")
        return num_train, num_test

    if num_train is None and num_test is None:
        legal_choices = (
            (2, 1), (3, 1), (4, 1), (2, 1),
            (3, 1), (4, 1), (4, 2), (5, 1),
            (5, 2), (6, 2), (6, 3), (7, 3),
            (8, 3), (8, 4), (9, 4), (10, 4),
        )
        return legal_choices[int(rng.integers(len(legal_choices)))]

    if num_train is None:
        assert num_test is not None
        num_train = int(rng.integers(num_test + 1, 11))
    else:
        upper_test_count = min(4, num_train - 1)
        num_test = int(rng.integers(1, upper_test_count + 1))

    assert num_test is not None
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _sample_puzzle_parameters(
    rng: np.random.Generator,
    num_train: int,
    num_test: int,
) -> dict:
    background = int(rng.integers(10))

    non_background_colors = np.array(
        [color for color in range(10) if color != background],
        dtype=np.int8,
    )
    cycle = tuple(
        int(color) for color in rng.permutation(non_background_colors)
    )

    phase_order = [int(phase) for phase in rng.permutation(9)]
    train_phases = phase_order[:2] + [
        int(rng.choice(phase_order[:8])) for _ in range(num_train - 2)
    ]
    rng.shuffle(train_phases)
    known = list(set(train_phases))
    held_out = [phase for phase in range(9) if phase not in known]
    phase_regime = int(rng.integers(3))
    if phase_regime == 0:
        test_phases = [
            int(rng.choice(held_out)) for _ in range(num_test)
        ]
    elif phase_regime == 1 or num_test == 1:
        test_phases = [
            int(rng.choice(known)) for _ in range(num_test)
        ]
    else:
        test_phases = [
            int(rng.choice(known)), int(rng.choice(held_out))
        ] + [int(rng.integers(9)) for _ in range(num_test - 2)]
        rng.shuffle(test_phases)

    return {
        "background": background,
        "cycle": cycle,
        "train_phases": tuple(train_phases),
        "test_phases": tuple(test_phases),
    }


def _is_target_position(
    shape: tuple[int, int],
    row: int,
    column: int,
) -> bool:
    height, width = shape
    return (
        0 < row < height // 2
        and 0 < column < width // 2
    )


def _is_edge(
    shape: tuple[int, int],
    row: int,
    column: int,
) -> bool:
    height, width = shape
    return (
        row in (0, height - 1)
        or column in (0, width - 1)
    )


def _sample_shape(rng: np.random.Generator) -> tuple[int, int]:
    profile = int(rng.choice(4, p=[0.16, 0.52, 0.15, 0.17]))
    if profile == 0:  # compact
        return int(rng.integers(4, 10)), int(rng.integers(4, 10))
    if profile == 1:  # ordinary, including the source size
        return int(rng.integers(6, 21)), int(rng.integers(6, 21))
    if profile == 2:  # narrow in either orientation
        sides = (int(rng.integers(3, 6)), int(rng.integers(9, 27)))
        return sides if rng.random() < 0.5 else sides[::-1]
    sides = (int(rng.integers(21, 31)), int(rng.integers(7, 31)))
    return sides if rng.random() < 0.5 else sides[::-1]


def _position_class(shape: tuple[int, int], row: int, column: int) -> str:
    height, width = shape
    if row in (0, height - 1) and column in (0, width - 1):
        return "corner"
    if _is_edge(shape, row, column):
        return "edge"
    if _is_target_position(shape, row, column):
        return "upper_left"
    return "interior"


def _sample_position(
    rng: np.random.Generator,
    shape: tuple[int, int],
    preferred: str,
    witness: bool,
) -> tuple[int, int]:
    height, width = shape
    candidates = [
        (row, column)
        for row in range(height)
        for column in range(width)
        if (preferred == "any" or _position_class(shape, row, column) == preferred)
        and (not witness or max(
            row + column,
            row + width - 1 - column,
            height - 1 - row + column,
            height + width - 2 - row - column,
        ) >= 9)
    ]
    if not candidates:
        candidates = [
            (row, column)
            for row in range(height)
            for column in range(width)
            if not witness or max(
                row + column,
                row + width - 1 - column,
                height - 1 - row + column,
                height + width - 2 - row - column,
            ) >= 9
        ]
    if not candidates:
        raise ValueError("shape cannot display a nine-step wrap")
    return candidates[int(rng.integers(len(candidates)))]


def _distances(
    shape: tuple[int, int],
    row: int,
    column: int,
) -> np.ndarray:
    height, width = shape

    row_coordinates = np.arange(
        height, dtype=np.int16
    )[:, None]
    column_coordinates = np.arange(
        width, dtype=np.int16
    )[None, :]

    return (
        np.abs(row_coordinates - row)
        + np.abs(column_coordinates - column)
    )


def _construct_output(
    shape: tuple[int, int],
    row: int,
    column: int,
    seed_color: int,
    cycle: tuple[int, ...],
) -> np.ndarray:
    seed_phase = cycle.index(seed_color)
    distances = _distances(shape, row, column)
    palette = np.asarray(cycle, dtype=np.int8)

    return palette[(seed_phase + distances) % 9]


def _make_pair(
    shape: tuple[int, int],
    position: tuple[int, int],
    seed_color: int,
    parameters: dict,
) -> dict:
    row, column = position
    input_grid = np.full(shape, parameters["background"], dtype=np.int8)
    input_grid[row, column] = seed_color
    return {
        "input": input_grid,
        "output": _construct_output(
            shape, row, column, seed_color, parameters["cycle"]
        ),
    }


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        grid.shape,
        tuple(int(value) for value in grid.ravel()),
    )


def _pair_key(pair: dict) -> tuple:
    return (
        _grid_key(pair["input"]),
        _grid_key(pair["output"]),
    )


def _matches_source_input(grid: np.ndarray) -> bool:
    return any(
        grid.shape == source_input.shape
        and np.array_equal(grid, source_input)
        for source_input, _ in _SOURCE_PAIRS
    )


def _input_info(
    input_grid: np.ndarray,
) -> tuple[int, int, int, int]:
    assert input_grid.shape[0] >= 3
    assert input_grid.shape[1] >= 3

    values, counts = np.unique(
        input_grid, return_counts=True
    )
    assert len(values) == 2

    seed_values = values[counts == 1]
    background_values = values[
        counts == input_grid.size - 1
    ]

    assert len(seed_values) == 1
    assert len(background_values) == 1

    seed_color = int(seed_values[0])
    background = int(background_values[0])

    locations = np.argwhere(input_grid == seed_color)
    assert locations.shape == (1, 2)

    row, column = (
        int(coordinate) for coordinate in locations[0]
    )

    return background, seed_color, row, column


def _recover_successor(
    training_pairs: list[dict],
    background: int,
) -> dict[int, int]:
    recovered: dict[int, int] | None = None

    for pair in training_pairs:
        input_grid = pair["input"]
        output_grid = pair["output"]

        (
            pair_background,
            seed_color,
            row,
            column,
        ) = _input_info(input_grid)

        assert pair_background == background

        distances = _distances(
            input_grid.shape, row, column
        )
        if int(distances.max()) < 9:
            continue

        sequence: list[int] = []
        for distance in range(9):
            colors = np.unique(
                output_grid[distances == distance]
            )
            assert len(colors) == 1
            sequence.append(int(colors[0]))

        assert sequence[0] == seed_color
        assert np.array_equal(
            np.unique(output_grid[distances == 9]),
            np.asarray([seed_color]),
        )
        assert len(set(sequence)) == 9
        assert set(sequence) == (
            set(range(10)) - {background}
        )

        successor = {
            sequence[index]: sequence[(index + 1) % 9]
            for index in range(9)
        }

        if recovered is None:
            recovered = successor
        else:
            assert successor == recovered

    assert recovered is not None
    assert set(recovered) == set(range(10)) - {background}
    assert set(recovered.values()) == set(recovered)

    return recovered


def _expected_from_successor(
    shape: tuple[int, int],
    row: int,
    column: int,
    seed_color: int,
    successor: dict[int, int],
) -> np.ndarray:
    sequence = [seed_color]

    for _ in range(8):
        sequence.append(successor[sequence[-1]])

    assert len(set(sequence)) == 9
    assert successor[sequence[-1]] == seed_color

    distances = _distances(shape, row, column)
    palette = np.asarray(sequence, dtype=np.int8)

    return palette[distances % 9]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {"train", "test"}

    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    all_pairs = puzzle["train"] + puzzle["test"]

    input_keys: set[tuple] = set()
    pair_keys: set[tuple] = set()
    input_infos: list[tuple[int, int, int, int]] = []

    for pair in all_pairs:
        assert isinstance(pair, dict)
        assert set(pair.keys()) == {"input", "output"}

        for grid in (pair["input"], pair["output"]):
            assert isinstance(grid, np.ndarray)
            assert grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30
            assert 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((0 <= grid) & (grid <= 9))

        assert pair["input"].shape == pair["output"].shape

        input_key = _grid_key(pair["input"])
        pair_key = _pair_key(pair)

        assert input_key not in input_keys
        assert pair_key not in pair_keys

        input_keys.add(input_key)
        pair_keys.add(pair_key)

        # Stronger than merely rejecting an exact source input/output pair.
        assert not _matches_source_input(pair["input"])

        input_infos.append(_input_info(pair["input"]))

    backgrounds = {
        info[0] for info in input_infos
    }
    assert len(backgrounds) == 1
    background = next(iter(backgrounds))

    # This is reconstructed entirely from the returned training grids.
    successor = _recover_successor(
        puzzle["train"], background
    )

    train_count = len(puzzle["train"])
    train_infos = input_infos[:train_count]
    test_infos = input_infos[train_count:]

    assert any(
        int(_distances(pair["input"].shape, info[2], info[3]).max()) >= 9
        for pair, info in zip(puzzle["train"], train_infos)
    )

    train_seed_colors = {
        info[1] for info in train_infos
    }
    test_seed_colors = {
        info[1] for info in test_infos
    }

    assert len(train_seed_colors) >= 2

    # The training outputs expose all nine states, including every
    # held-out test seed color.
    training_output_colors: set[int] = set()
    for pair in puzzle["train"]:
        training_output_colors.update(
            int(color)
            for color in np.unique(pair["output"])
        )

    assert training_output_colors == (
        set(range(10)) - {background}
    )
    assert test_seed_colors <= training_output_colors

    # Verify every ground-truth output using only information recovered
    # from the visible inputs and training outputs.
    for pair, (_, seed_color, row, column) in zip(
        all_pairs, input_infos
    ):
        assert seed_color in successor
        assert not np.any(pair["output"] == background)

        expected = _expected_from_successor(
            pair["input"].shape,
            row,
            column,
            seed_color,
            successor,
        )
        assert np.array_equal(pair["output"], expected)

    # Explicitly reject exact original source pairs.
    for pair in all_pairs:
        for source_input, source_output in _SOURCE_PAIRS:
            assert not (
                pair["input"].shape == source_input.shape
                and pair["output"].shape
                == source_output.shape
                and np.array_equal(
                    pair["input"], source_input
                )
                and np.array_equal(
                    pair["output"], source_output
                )
            )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    parameters = _sample_puzzle_parameters(rng, num_train, num_test)
    total = num_train + num_test

    # Size regimes keep some whole puzzles at the source scale, share a
    # canvas across some pairs, or deliberately change size at test time.
    size_regime = int(rng.choice(5, p=[0.15, 0.15, 0.20, 0.23, 0.27]))
    if size_regime == 0:
        shapes = [(8, 8)] * total
    else:
        base = _sample_shape(rng)
        while sum(base) < 11:
            base = _sample_shape(rng)
        if size_regime == 1:
            shapes = [base] * total
        elif size_regime == 2:
            shared_count = int(rng.integers(2, total))
            shared_slots = set(
                int(slot) for slot in rng.choice(total, size=shared_count, replace=False)
            )
            shapes = [
                base if index in shared_slots else _sample_shape(rng)
                for index in range(total)
            ]
        elif size_regime == 3:
            shapes = [base] * num_train
            for _ in range(num_test):
                candidate = _sample_shape(rng)
                while candidate == base:
                    candidate = _sample_shape(rng)
                shapes.append(candidate)
        else:
            shapes = [_sample_shape(rng) for _ in range(total)]

    # The source's edge-to-interior shift is one regime among several
    # faithful placements of the same single origin.
    placement_regime = int(rng.integers(5))
    if placement_regime == 0:
        train_classes = ["edge", "interior"]
        test_classes = ["upper_left"]
    elif placement_regime == 1:
        train_classes = ["upper_left", "interior"]
        test_classes = ["interior"]
    elif placement_regime == 2:
        train_classes = ["corner", "edge"]
        test_classes = ["corner", "edge"]
    elif placement_regime == 3:
        train_classes = ["interior", "upper_left"]
        test_classes = ["edge", "corner"]
    else:
        train_classes = ["edge", "upper_left"]
        test_classes = ["interior", "corner", "edge"]
    rng.shuffle(train_classes)
    extra_classes = (
        ("upper_left", "interior") if placement_regime == 1
        else ("corner", "edge", "upper_left", "interior")
    )
    train_classes += [
        extra_classes[int(rng.integers(len(extra_classes)))]
        for _ in range(num_train - 2)
    ]
    test_classes = [
        test_classes[index % len(test_classes)]
        for index in range(num_test)
    ]
    if placement_regime in (1, 2, 3, 4):
        rng.shuffle(test_classes)

    witness_slot = int(rng.integers(num_train))
    puzzle = {"train": [], "test": []}
    seen_inputs: set[tuple] = set()
    phases = parameters["train_phases"] + parameters["test_phases"]
    classes = train_classes + test_classes
    for index, (phase, preferred) in enumerate(zip(phases, classes)):
        shape = shapes[index]
        witness = index == witness_slot
        if witness and sum(shape) < 11:
            shape = base if size_regime != 0 else (8, 8)
            if sum(shape) < 11:
                shape = (8, 8)
        for _ in range(100):
            position = _sample_position(rng, shape, preferred, witness)
            pair = _make_pair(
                shape, position, parameters["cycle"][phase], parameters
            )
            key = _grid_key(pair["input"])
            if key not in seen_inputs and not _matches_source_input(pair["input"]):
                seen_inputs.add(key)
                puzzle["train" if index < num_train else "test"].append(pair)
                break
        else:
            # A new shape is a deterministic escape from crowded fixed
            # canvases when explicit requests ask for many pairs.
            shape = (10 + index, 11 + index)
            position = _sample_position(rng, shape, preferred, witness)
            pair = _make_pair(
                shape, position, parameters["cycle"][phase], parameters
            )
            assert _grid_key(pair["input"]) not in seen_inputs
            seen_inputs.add(_grid_key(pair["input"]))
            puzzle["train" if index < num_train else "test"].append(pair)

    validate_generated_puzzle(puzzle)
    return puzzle


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
            first[split], second[split]
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
        (split, _pair_key(pair))
        for split in ("train", "test")
        for pair in puzzle[split]
    )


def self_test() -> None:
    # Automatic counts across 42 fixed puzzle seeds.
    automatic_puzzles = [
        generate_puzzle(seed)
        for seed in range(42)
    ]

    for puzzle in automatic_puzzles:
        validate_generated_puzzle(puzzle)

    # Explicit requested counts.
    explicit_puzzles = (
        generate_puzzle(101, 5, 2),
        generate_puzzle(102, 9, 4),
        generate_puzzle(103, 7, 3),
    )
    for puzzle in explicit_puzzles:
        validate_generated_puzzle(puzzle)

    # Whole-puzzle determinism.
    deterministic_a = generate_puzzle(
        123456, 8, 3
    )
    deterministic_b = generate_puzzle(
        123456, 8, 3
    )
    assert _puzzles_equal(
        deterministic_a, deterministic_b
    )

    # Different seeds should produce different complete puzzles.
    assert not _puzzles_equal(
        generate_puzzle(123456, 8, 3),
        generate_puzzle(123457, 8, 3),
    )

    # Legal boundary cases.
    for num_train, num_test in (
        (2, 1),
        (10, 1),
        (10, 4),
    ):
        puzzle = generate_puzzle(
            2000 + 10 * num_train + num_test,
            num_train,
            num_test,
        )
        assert len(puzzle["train"]) == num_train
        assert len(puzzle["test"]) == num_test
        validate_generated_puzzle(puzzle)

    # Partial count specifications.
    partial_cases = (
        (3001, None, 1),
        (3002, None, 4),
        (3003, 2, None),
        (3004, 10, None),
    )

    for seed, requested_train, requested_test in (
        partial_cases
    ):
        puzzle = generate_puzzle(
            seed,
            requested_train,
            requested_test,
        )

        if requested_train is not None:
            assert (
                len(puzzle["train"])
                == requested_train
            )
        if requested_test is not None:
            assert (
                len(puzzle["test"])
                == requested_test
            )

        validate_generated_puzzle(puzzle)

    # Illegal and impossible count specifications.
    illegal_cases = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
        {"num_train": True, "num_test": 1},
        {"num_train": 4.0, "num_test": 1},
    )

    for arguments in illegal_cases:
        try:
            generate_puzzle(999, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"illegal counts accepted: {arguments}"
            )

    # Diversity and independently reconstructed task invariants.
    train_counts = {
        len(puzzle["train"])
        for puzzle in automatic_puzzles
    }
    test_counts = {
        len(puzzle["test"])
        for puzzle in automatic_puzzles
    }

    backgrounds: set[int] = set()
    cycle_signatures: set[tuple] = set()
    shapes: set[tuple[int, int]] = set()
    layouts: set[tuple] = set()
    grid_signatures: set[tuple] = set()
    aspect_classes: set[int] = set()

    for puzzle in automatic_puzzles:
        background = _input_info(
            puzzle["train"][0]["input"]
        )[0]
        successor = _recover_successor(
            puzzle["train"], background
        )

        backgrounds.add(background)
        cycle_signatures.add(
            (
                background,
                tuple(
                    successor.get(color, -1)
                    for color in range(10)
                ),
            )
        )

        for split in ("train", "test"):
            for pair in puzzle[split]:
                (
                    _,
                    seed_color,
                    row,
                    column,
                ) = _input_info(pair["input"])

                height, width = pair["input"].shape

                shapes.add((height, width))
                layouts.add(
                    (
                        split,
                        height,
                        width,
                        row,
                        column,
                    )
                )
                grid_signatures.add(
                    _grid_key(pair["input"])
                )
                aspect_classes.add(
                    int(np.sign(width - height))
                )

                expected = _expected_from_successor(
                    (height, width),
                    row,
                    column,
                    seed_color,
                    successor,
                )
                assert np.array_equal(
                    pair["output"], expected
                )

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3

    assert len(
        {
            _puzzle_key(puzzle)
            for puzzle in automatic_puzzles
        }
    ) >= 40

    assert len(backgrounds) >= 7
    assert len(cycle_signatures) >= 35
    assert len(shapes) >= 75
    assert len(layouts) >= 150
    assert len(grid_signatures) >= 200

    # Landscape, square, and portrait grids must all occur.
    assert aspect_classes == {-1, 0, 1}

    review_coverage = {
        name: 0 for name in (
            "witness", "familiar", "held_out", "mixed_phase",
            "repeated_test_phase", "fixed_8", "shared_shape",
            "changed_test_shape", "compact_test", "large",
            "concise", "interior_only", "boundary_both",
            "boundary_to_interior", "interior_to_boundary",
            "upper_left_train", "combined_shift",
        )
    }
    observed_positions = {"train": set(), "test": set()}
    for puzzle in automatic_puzzles:
        train, test = puzzle["train"], puzzle["test"]
        train_infos = [_input_info(pair["input"]) for pair in train]
        test_infos = [_input_info(pair["input"]) for pair in test]
        train_positions = [
            _position_class(pair["input"].shape, info[2], info[3])
            for pair, info in zip(train, train_infos)
        ]
        test_positions = [
            _position_class(pair["input"].shape, info[2], info[3])
            for pair, info in zip(test, test_infos)
        ]
        observed_positions["train"].update(train_positions)
        observed_positions["test"].update(test_positions)
        witness = any(
            int(_distances(pair["input"].shape, info[2], info[3]).max()) >= 9
            for pair, info in zip(train, train_infos)
        )
        review_coverage["witness"] += witness
        known = {info[1] for info in train_infos}
        familiar = [info[1] in known for info in test_infos]
        review_coverage["familiar"] += all(familiar)
        review_coverage["held_out"] += not any(familiar)
        review_coverage["mixed_phase"] += any(familiar) and not all(familiar)
        review_coverage["repeated_test_phase"] += (
            len({info[1] for info in test_infos}) < len(test_infos)
        )
        train_shapes = {pair["input"].shape for pair in train}
        all_shapes = [pair["input"].shape for pair in train + test]
        changed_shape = any(pair["input"].shape not in train_shapes for pair in test)
        review_coverage["fixed_8"] += all(shape == (8, 8) for shape in all_shapes)
        review_coverage["shared_shape"] += len(set(all_shapes)) < len(all_shapes)
        review_coverage["changed_test_shape"] += changed_shape
        review_coverage["compact_test"] += any(
            min(pair["input"].shape) <= 5 for pair in test
        )
        review_coverage["large"] += any(max(shape) >= 21 for shape in all_shapes)
        review_coverage["concise"] += len(train) <= 4 and len(test) == 1
        train_boundary = any(pos in ("corner", "edge") for pos in train_positions)
        test_boundary = any(pos in ("corner", "edge") for pos in test_positions)
        review_coverage["interior_only"] += not train_boundary and not test_boundary
        review_coverage["boundary_both"] += train_boundary and test_boundary
        review_coverage["boundary_to_interior"] += train_boundary and not test_boundary
        review_coverage["interior_to_boundary"] += not train_boundary and test_boundary
        review_coverage["upper_left_train"] += "upper_left" in train_positions
        review_coverage["combined_shift"] += (
            changed_shape and test_boundary and any(familiar) and not all(familiar)
        )
    assert review_coverage["witness"] == len(automatic_puzzles)
    assert observed_positions == {
        "train": {"corner", "edge", "upper_left", "interior"},
        "test": {"corner", "edge", "upper_left", "interior"},
    }
    assert all(value > 0 for value in review_coverage.values())
