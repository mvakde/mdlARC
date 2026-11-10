import numpy as np


_RIGHT3 = 0
_LEFT2 = 1
_DOWN2 = 2
_LEFT4 = 3
_ALL_COMMANDS = frozenset((_RIGHT3, _LEFT2, _DOWN2, _LEFT4))

_MASKS = (
    np.array(
        [
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ],
        dtype=np.int8,
    ),
    np.array(
        [
            [1, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=np.int8,
    ),
    np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=np.int8,
    ),
    np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [1, 0, 1],
        ],
        dtype=np.int8,
    ),
)

# These three roles occur in every source input. Requiring a different
# assignment guarantees that no generated pair exactly copies a source pair.
_SOURCE_ALWAYS_VISIBLE_ROLES = (0, 4, 5)


def _normalize_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise ValueError("seed must be a nonnegative integer")

    result = int(seed)
    if result < 0:
        raise ValueError("seed must be a nonnegative integer")
    return result


def _normalize_count(
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
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _normalize_count("num_train", num_train, 2, 10)
    test_count = _normalize_count("num_test", num_test, 1, 4)

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        minimum_train = max(2, test_count + 1)
        train_count = (
            minimum_train
            + (seed // 4) % (11 - minimum_train)
        )
    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        train_count = minimum_train + seed % (11 - minimum_train)
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        test_count = 1 + seed % maximum_test

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    while True:
        roles = tuple(
            int(value)
            for value in rng.choice(10, size=7, replace=False)
        )
        if roles[:3] != _SOURCE_ALWAYS_VISIBLE_ROLES:
            break

    return {
        "background": roles[0],
        "separator": roles[1],
        "seed": roles[2],
        "command_colors": roles[3:],
    }


def _program_is_valid(
    commands: tuple[int, ...],
    start_col: int,
    n: int,
    regime: str,
    require_all: bool = False,
) -> bool:
    if regime == "source_train":
        if n not in (2, 3, 4):
            return False
    elif regime == "target_test":
        if n != 5:
            return False
    else:
        return False

    if len(commands) != 2 * n:
        return False
    if not 0 <= start_col < 7:
        return False
    if any(command not in _ALL_COMMANDS for command in commands):
        return False

    used_commands = set(commands)
    if require_all and used_commands != _ALL_COMMANDS:
        return False

    second_lane = commands[n:]
    has_second_lane_double_right = any(
        second_lane[index] == _RIGHT3
        and second_lane[index + 1] == _RIGHT3
        for index in range(len(second_lane) - 1)
    )

    if regime == "source_train":
        if len(used_commands) < 3:
            return False
        if has_second_lane_double_right:
            return False
    else:
        if used_commands != _ALL_COMMANDS:
            return False
        if not has_second_lane_double_right:
            return False

    rows_consumed = len(commands) + commands.count(_DOWN2)
    if rows_consumed > 4 * n - 2:
        return False

    col = start_col
    for command in commands:
        if command == _RIGHT3:
            if col + 2 > 6:
                return False
            col += 2
        elif command == _LEFT2:
            if col - 1 < 0:
                return False
            col -= 1
        elif command == _DOWN2:
            pass
        elif command == _LEFT4:
            if col - 3 < 0:
                return False
            col -= 3

    return True


def _sample_program(
    rng: np.random.Generator,
    n: int,
    regime: str,
    require_all: bool,
    used_signatures: set,
) -> tuple[tuple[int, ...], int]:
    length = 2 * n

    for _ in range(20000):
        start_col = int(rng.integers(0, 7))
        col = start_col
        command_list = []

        double_start = (
            int(rng.integers(n, length - 1))
            if regime == "target_test"
            else -2
        )
        failed = False

        for position in range(length):
            if position in (double_start, double_start + 1):
                if col + 2 > 6:
                    failed = True
                    break
                command = _RIGHT3
            else:
                legal_commands = [_DOWN2]

                if col + 2 <= 6:
                    legal_commands.append(_RIGHT3)
                if col - 1 >= 0:
                    legal_commands.append(_LEFT2)
                if col - 3 >= 0:
                    legal_commands.append(_LEFT4)

                command = legal_commands[
                    int(rng.integers(0, len(legal_commands)))
                ]

            command_list.append(command)

            if command == _RIGHT3:
                col += 2
            elif command == _LEFT2:
                col -= 1
            elif command == _LEFT4:
                col -= 3

        if failed:
            continue

        commands = tuple(command_list)
        signature = (n, start_col, commands)

        if (
            signature not in used_signatures
            and _program_is_valid(
                commands,
                start_col,
                n,
                regime,
                require_all,
            )
        ):
            used_signatures.add(signature)
            return commands, start_col

    # Deterministic exhaustive fallback. This should only be needed if an
    # unusually long random rejection streak occurs.
    for start_col in range(7):
        for encoded in range(4**length):
            value = encoded
            command_list = []

            for _ in range(length):
                command_list.append(value % 4)
                value //= 4

            commands = tuple(command_list)
            signature = (n, start_col, commands)

            if (
                signature not in used_signatures
                and _program_is_valid(
                    commands,
                    start_col,
                    n,
                    regime,
                    require_all,
                )
            ):
                used_signatures.add(signature)
                return commands, start_col

    raise ValueError("could not construct enough distinct valid pairs")


def _render_output(
    height: int,
    background: int,
    seed_color: int,
    start_col: int,
    commands: tuple[int, ...],
    command_colors,
) -> np.ndarray:
    output = np.full(
        (height, 7),
        background,
        dtype=np.int8,
    )
    output[0, start_col] = seed_color

    row = 0
    col = start_col

    for command in commands:
        color = int(command_colors[command])

        if command == _RIGHT3:
            row += 1
            assert row < height
            assert col + 2 < 7
            assert np.all(
                output[row, col:col + 3] == background
            )
            output[row, col:col + 3] = color
            col += 2

        elif command == _LEFT2:
            row += 1
            assert row < height
            assert col - 1 >= 0
            assert np.all(
                output[row, col - 1:col + 1] == background
            )
            output[row, col - 1:col + 1] = color
            col -= 1

        elif command == _DOWN2:
            assert row + 2 < height
            assert np.all(
                output[row + 1:row + 3, col] == background
            )
            output[row + 1:row + 3, col] = color
            row += 2

        elif command == _LEFT4:
            row += 1
            assert row < height
            assert col - 3 >= 0
            assert np.all(
                output[row, col - 3:col + 1] == background
            )
            output[row, col - 3:col + 1] = color
            col -= 3

        else:
            raise AssertionError("unknown command")

    return output


def generate_pair(
    rng: np.random.Generator,
    regime: str,
    puzzle_parameters: dict,
    n: int,
    require_all: bool,
    used_signatures: set,
) -> tuple[np.ndarray, np.ndarray]:
    commands, start_col = _sample_program(
        rng,
        n,
        regime,
        require_all,
        used_signatures,
    )

    height = 4 * n - 1
    background = int(puzzle_parameters["background"])
    separator = int(puzzle_parameters["separator"])
    seed_color = int(puzzle_parameters["seed"])
    command_colors = tuple(
        int(value)
        for value in puzzle_parameters["command_colors"]
    )

    input_grid = np.full(
        (height, 15),
        background,
        dtype=np.int8,
    )
    input_grid[:, 7] = separator

    for instruction_index, command in enumerate(commands):
        if instruction_index < n:
            band = instruction_index
            left = 0
        else:
            band = instruction_index - n
            left = 4

        top = 4 * band
        block = input_grid[
            top:top + 3,
            left:left + 3,
        ]
        block[_MASKS[command].astype(bool)] = (
            command_colors[command]
        )

    input_grid[0, 8 + start_col] = seed_color

    output_grid = _render_output(
        height,
        background,
        seed_color,
        start_col,
        commands,
        command_colors,
    )

    return input_grid, output_grid


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed_value = _normalize_seed(seed)
    train_count, test_count = _resolve_counts(
        seed_value,
        num_train,
        num_test,
    )

    rng = np.random.default_rng(seed_value)
    puzzle_parameters = sample_puzzle_parameters(rng)

    extra_order = tuple(
        int(value)
        for value in rng.permutation(
            np.array([3, 2, 4], dtype=np.int8)
        )
    )

    # The first pair exposes all commands. The second gives the longest
    # training-side length, and the third supplies the intermediate length.
    train_sizes = [2, 4, 3]
    while len(train_sizes) < train_count:
        train_sizes.append(
            extra_order[(len(train_sizes) - 3) % 3]
        )

    pair_rngs = [
        np.random.default_rng(
            int(
                rng.integers(
                    0,
                    np.iinfo(np.int64).max,
                )
            )
        )
        for _ in range(train_count + test_count)
    ]

    used_signatures = set()
    train_pairs = []

    for index in range(train_count):
        input_grid, output_grid = generate_pair(
            pair_rngs[index],
            "source_train",
            puzzle_parameters,
            train_sizes[index],
            index == 0,
            used_signatures,
        )
        train_pairs.append(
            {
                "input": input_grid,
                "output": output_grid,
            }
        )

    test_pairs = []
    for index in range(test_count):
        input_grid, output_grid = generate_pair(
            pair_rngs[train_count + index],
            "target_test",
            puzzle_parameters,
            5,
            True,
            used_signatures,
        )
        test_pairs.append(
            {
                "input": input_grid,
                "output": output_grid,
            }
        )

    puzzle = {
        "train": train_pairs,
        "test": test_pairs,
    }
    validate_generated_puzzle(puzzle)
    return puzzle


def _assert_grid(grid) -> None:
    assert isinstance(grid, np.ndarray)
    assert grid.ndim == 2

    height, width = grid.shape
    assert 1 <= height <= 30
    assert 1 <= width <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert int(grid.min()) >= 0
    assert int(grid.max()) <= 9


def _grid_key(grid: np.ndarray) -> tuple:
    normalized = np.asarray(grid, dtype=np.int8)
    return normalized.shape, normalized.tobytes()


def _analyze_pair(
    pair: dict,
    regime: str,
) -> dict:
    input_grid = pair["input"]
    output_grid = pair["output"]
    height, width = input_grid.shape

    assert width == 15
    assert (height + 1) % 4 == 0

    n = (height + 1) // 4
    if regime == "source_train":
        assert n in (2, 3, 4)
    else:
        assert regime == "target_test"
        assert n == 5

    assert output_grid.shape == (height, 7)

    background = int(input_grid[0, 3])
    assert np.all(input_grid[:, 3] == background)

    separator = int(input_grid[0, 7])
    assert separator != background
    assert np.all(input_grid[:, 7] == separator)

    initial_pane = input_grid[:, 8:15]
    seed_positions = np.argwhere(
        initial_pane != background
    )
    assert seed_positions.shape == (1, 2)
    assert int(seed_positions[0, 0]) == 0

    start_col = int(seed_positions[0, 1])
    seed_color = int(initial_pane[0, start_col])
    assert seed_color not in (background, separator)

    expected_initial_pane = np.full(
        (height, 7),
        background,
        dtype=np.int8,
    )
    expected_initial_pane[0, start_col] = seed_color
    assert np.array_equal(
        initial_pane,
        expected_initial_pane,
    )

    commands = []
    local_command_colors = {}
    expected_panel = np.full(
        (height, 7),
        background,
        dtype=np.int8,
    )

    # This loop is intentionally lane-major: all left-lane glyphs, then
    # all right-lane glyphs.
    for lane_left in (0, 4):
        for band in range(n):
            top = 4 * band
            block = input_grid[
                top:top + 3,
                lane_left:lane_left + 3,
            ]

            foreground_values = np.unique(
                block[block != background]
            )
            assert len(foreground_values) == 1

            foreground = int(foreground_values[0])
            assert foreground not in (
                separator,
                seed_color,
            )

            occupancy = (
                block != background
            ).astype(np.int8)

            matches = [
                command
                for command, mask in enumerate(_MASKS)
                if np.array_equal(occupancy, mask)
            ]
            assert len(matches) == 1

            command = matches[0]
            commands.append(command)

            if command in local_command_colors:
                assert (
                    local_command_colors[command]
                    == foreground
                )
            else:
                local_command_colors[command] = foreground

            expected_block = expected_panel[
                top:top + 3,
                lane_left:lane_left + 3,
            ]
            expected_block[
                _MASKS[command].astype(bool)
            ] = foreground

    # This verifies all spacer columns, inter-band rows, glyph pixels,
    # and the absence of instruction-panel noise.
    assert np.array_equal(
        input_grid[:, :7],
        expected_panel,
    )

    assert (
        len(set(local_command_colors.values()))
        == len(local_command_colors)
    )

    command_tuple = tuple(commands)
    assert _program_is_valid(
        command_tuple,
        start_col,
        n,
        regime,
    )

    expected_output = _render_output(
        height,
        background,
        seed_color,
        start_col,
        command_tuple,
        local_command_colors,
    )
    assert np.array_equal(
        output_grid,
        expected_output,
    )

    return {
        "regime": regime,
        "n": n,
        "background": background,
        "separator": separator,
        "seed_color": seed_color,
        "seed_col": start_col,
        "commands": command_tuple,
        "command_colors": local_command_colors,
    }


def validate_generated_puzzle(
    puzzle: dict,
) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    train_count = len(puzzle["train"])
    test_count = len(puzzle["test"])

    assert 2 <= train_count <= 10
    assert 1 <= test_count <= 4
    assert train_count > test_count

    input_keys = set()
    pair_keys = set()
    analyses = {
        "train": [],
        "test": [],
    }

    for split_name, regime in (
        ("train", "source_train"),
        ("test", "target_test"),
    ):
        for pair in puzzle[split_name]:
            assert isinstance(pair, dict)
            assert set(pair.keys()) == {
                "input",
                "output",
            }

            _assert_grid(pair["input"])
            _assert_grid(pair["output"])

            input_key = _grid_key(pair["input"])
            pair_key = (
                input_key,
                _grid_key(pair["output"]),
            )

            assert input_key not in input_keys
            assert pair_key not in pair_keys

            input_keys.add(input_key)
            pair_keys.add(pair_key)

            analyses[split_name].append(
                _analyze_pair(pair, regime)
            )

    first_training = analyses["train"][0]
    common_roles = (
        first_training["background"],
        first_training["separator"],
        first_training["seed_color"],
    )

    # Because these roles occur in every pair, this also guarantees that
    # no generated input/output pair exactly duplicates a source pair.
    assert common_roles != _SOURCE_ALWAYS_VISIBLE_ROLES

    # Four distinct commands in a two-band pair make every command and
    # the column-major instruction order visible immediately.
    assert first_training["n"] == 2
    assert (
        set(first_training["commands"])
        == _ALL_COMMANDS
    )
    assert len(first_training["commands"]) == 4

    assert any(
        analysis["n"] == 4
        for analysis in analyses["train"]
    )
    if train_count >= 3:
        assert any(
            analysis["n"] == 3
            for analysis in analyses["train"]
        )

    training_command_colors = {}

    for analysis in analyses["train"]:
        assert (
            analysis["background"],
            analysis["separator"],
            analysis["seed_color"],
        ) == common_roles

        for command, color in (
            analysis["command_colors"].items()
        ):
            if command in training_command_colors:
                assert (
                    training_command_colors[command]
                    == color
                )
            else:
                training_command_colors[command] = color

    assert (
        set(training_command_colors)
        == _ALL_COMMANDS
    )

    puzzle_palette = common_roles + tuple(
        training_command_colors[command]
        for command in range(4)
    )
    assert len(set(puzzle_palette)) == 7

    for analysis in analyses["test"]:
        assert (
            analysis["background"],
            analysis["separator"],
            analysis["seed_color"],
        ) == common_roles

        for command, color in (
            analysis["command_colors"].items()
        ):
            assert command in training_command_colors
            assert (
                training_command_colors[command]
                == color
            )

        assert (
            set(analysis["commands"])
            == _ALL_COMMANDS
        )

    # With four distinct commands, these common alternative traversal
    # hypotheses cannot match the first training program.
    first = first_training["commands"]
    left_lane = first[:2]
    right_lane = first[2:]

    alternatives = (
        right_lane + left_lane,
        (
            left_lane[0],
            right_lane[0],
            left_lane[1],
            right_lane[1],
        ),
        (
            right_lane[0],
            left_lane[0],
            right_lane[1],
            left_lane[1],
        ),
        tuple(reversed(left_lane))
        + tuple(reversed(right_lane)),
    )

    assert all(
        tuple(first) != tuple(alternative)
        for alternative in alternatives
    )


def _puzzles_equal(
    first: dict,
    second: dict,
) -> bool:
    if set(first.keys()) != set(second.keys()):
        return False

    for split_name in ("train", "test"):
        if (
            len(first[split_name])
            != len(second[split_name])
        ):
            return False

        for pair_a, pair_b in zip(
            first[split_name],
            second[split_name],
        ):
            if not np.array_equal(
                pair_a["input"],
                pair_b["input"],
            ):
                return False
            if not np.array_equal(
                pair_a["output"],
                pair_b["output"],
            ):
                return False

    return True


def _puzzle_key(
    puzzle: dict,
) -> tuple:
    result = []

    for split_name in ("train", "test"):
        for pair in puzzle[split_name]:
            result.append(
                (
                    split_name,
                    _grid_key(pair["input"]),
                    _grid_key(pair["output"]),
                )
            )

    return tuple(result)


def self_test() -> None:
    deterministic_a = generate_puzzle(
        12345,
        7,
        3,
    )
    deterministic_b = generate_puzzle(
        12345,
        7,
        3,
    )
    assert _puzzles_equal(
        deterministic_a,
        deterministic_b,
    )

    explicit = generate_puzzle(
        90210,
        6,
        2,
    )
    assert len(explicit["train"]) == 6
    assert len(explicit["test"]) == 2
    validate_generated_puzzle(explicit)

    boundary_cases = (
        (2, 1),
        (10, 1),
        (10, 4),
    )
    for index, (
        train_count,
        test_count,
    ) in enumerate(boundary_cases):
        puzzle = generate_puzzle(
            20000 + index,
            train_count,
            test_count,
        )
        assert (
            len(puzzle["train"])
            == train_count
        )
        assert (
            len(puzzle["test"])
            == test_count
        )
        validate_generated_puzzle(puzzle)

    partial_cases = (
        (30001, None, 1),
        (30002, None, 4),
        (30003, 2, None),
        (30004, 10, None),
    )
    for (
        seed,
        requested_train,
        requested_test,
    ) in partial_cases:
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

        assert (
            len(puzzle["train"])
            > len(puzzle["test"])
        )
        validate_generated_puzzle(puzzle)

    illegal_cases = (
        {"num_train": 1},
        {"num_train": 11},
        {"num_test": 0},
        {"num_test": 5},
        {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4},
    )

    for kwargs in illegal_cases:
        try:
            generate_puzzle(17, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"illegal counts did not fail: {kwargs}"
            )

    different_seed_keys = {
        _puzzle_key(
            generate_puzzle(
                40000 + offset,
                6,
                2,
            )
        )
        for offset in range(6)
    }
    assert len(different_seed_keys) >= 5

    automatic_puzzles = [
        generate_puzzle(seed)
        for seed in range(42)
    ]

    train_counts = set()
    test_counts = set()
    palettes = set()
    backgrounds = set()
    seed_columns = set()
    structural_signatures = set()
    exact_pair_signatures = set()
    train_heights = set()
    test_heights = set()
    program_lengths = set()

    for puzzle in automatic_puzzles:
        validate_generated_puzzle(puzzle)

        train_counts.add(
            len(puzzle["train"])
        )
        test_counts.add(
            len(puzzle["test"])
        )

        split_analyses = {
            "train": [],
            "test": [],
        }

        for split_name, regime in (
            ("train", "source_train"),
            ("test", "target_test"),
        ):
            for pair in puzzle[split_name]:
                analysis = _analyze_pair(
                    pair,
                    regime,
                )
                split_analyses[split_name].append(
                    analysis
                )

                structural_signatures.add(
                    (
                        regime,
                        analysis["n"],
                        analysis["seed_col"],
                        analysis["commands"],
                    )
                )
                exact_pair_signatures.add(
                    (
                        split_name,
                        _grid_key(pair["input"]),
                        _grid_key(pair["output"]),
                    )
                )
                seed_columns.add(
                    analysis["seed_col"]
                )
                program_lengths.add(
                    len(analysis["commands"])
                )

                if split_name == "train":
                    train_heights.add(
                        pair["input"].shape[0]
                    )
                else:
                    test_heights.add(
                        pair["input"].shape[0]
                    )

        first = split_analyses["train"][0]
        training_map = dict(
            first["command_colors"]
        )
        assert (
            set(training_map)
            == _ALL_COMMANDS
        )

        roles = (
            first["background"],
            first["separator"],
            first["seed_color"],
        )
        assert (
            roles
            != _SOURCE_ALWAYS_VISIBLE_ROLES
        )

        palette = roles + tuple(
            training_map[command]
            for command in range(4)
        )
        palettes.add(palette)
        backgrounds.add(first["background"])

        for analysis in (
            split_analyses["train"][1:]
            + split_analyses["test"]
        ):
            assert (
                analysis["background"],
                analysis["separator"],
                analysis["seed_color"],
            ) == roles

            for command, color in (
                analysis["command_colors"].items()
            ):
                assert (
                    training_map[command]
                    == color
                )

        training_commands = set()
        for analysis in split_analyses["train"]:
            training_commands.update(
                analysis["commands"]
            )
        assert (
            training_commands
            == _ALL_COMMANDS
        )

        assert all(
            analysis["n"] in (2, 3, 4)
            for analysis
            in split_analyses["train"]
        )

        assert all(
            analysis["n"] == 5
            and any(
                analysis["commands"][index]
                == _RIGHT3
                and analysis["commands"][index + 1]
                == _RIGHT3
                for index in range(5, 9)
            )
            for analysis
            in split_analyses["test"]
        )

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3

    assert train_heights == {
        7,
        11,
        15,
    }
    assert test_heights == {19}
    assert program_lengths == {
        4,
        6,
        8,
        10,
    }

    assert len(palettes) >= 35
    assert len(backgrounds) >= 6
    assert len(seed_columns) >= 5
    assert len(structural_signatures) >= 200
    assert len(exact_pair_signatures) >= 300
