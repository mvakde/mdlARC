import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        (
            "000800000",
            "000000000",
            "000000200",
            "001000000",
            "000000000",
            "000000000",
            "000000100",
            "020000000",
            "000000000",
        ),
        (
            "000800000",
            "000004040",
            "007000200",
            "071704040",
            "007000000",
            "000000700",
            "404007170",
            "020000700",
            "404000000",
        ),
    ),
    (
        (
            "000000000",
            "000000000",
            "000000000",
            "002000000",
            "000000000",
            "000000000",
            "000000100",
            "000000000",
            "000000000",
        ),
        (
            "000000000",
            "000000000",
            "040400000",
            "002000000",
            "040400000",
            "000000700",
            "000007170",
            "000000700",
            "000000000",
        ),
    ),
    (
        (
            "000000000",
            "000000000",
            "002000000",
            "000000000",
            "000000000",
            "000000600",
            "000000000",
            "000100000",
            "000000000",
        ),
        (
            "000000000",
            "040400000",
            "002000000",
            "040400000",
            "000000000",
            "000000600",
            "000700000",
            "007170000",
            "000700000",
        ),
    ),
    (
        (
            "000000000",
            "000000000",
            "000000100",
            "002000000",
            "000000000",
            "000008000",
            "000000000",
            "060000020",
            "000000000",
        ),
        (
            "000000000",
            "000000700",
            "040407170",
            "002000700",
            "040400000",
            "000008000",
            "000000404",
            "060000020",
            "000000404",
        ),
    ),
)


_CARDINAL = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(grid.shape),
        grid.astype(np.uint8, copy=False).tobytes(order="C"),
    )


def _rows_to_grid(rows: tuple[str, ...]) -> np.ndarray:
    return np.array(
        [[ord(ch) - ord("0") for ch in row] for row in rows],
        dtype=np.int8,
    )


_SOURCE_PAIR_KEYS = {
    (
        _grid_key(_rows_to_grid(input_rows)),
        _grid_key(_rows_to_grid(output_rows)),
    )
    for input_rows, output_rows in _SOURCE_PAIR_ROWS
}


def _checked_count(
    name: str,
    value: int | None,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


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
        num_test = int(rng.integers(1, 5))
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
    elif num_train is None:
        minimum_train = max(2, num_test + 1)
        num_train = int(rng.integers(minimum_train, 11))
    else:
        largest_test = min(4, num_train - 1)
        if largest_test < 1:
            raise ValueError("no legal num_test exists for num_train")
        num_test = int(rng.integers(1, largest_test + 1))

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def sample_puzzle_parameters(
    rng: np.random.Generator,
) -> dict:
    colors = [int(value) for value in rng.permutation(10)]
    inert_count = int(rng.integers(0, 4))
    return {
        "cardinal_seed": colors[0],
        "diagonal_seed": colors[1],
        "cardinal_halo": colors[2],
        "diagonal_halo": colors[3],
        "inert_colors": tuple(colors[4 : 4 + inert_count]),
        "backgrounds": tuple(colors[4 + inert_count :]),
    }


def _sample_positions(
    rng: np.random.Generator,
    height: int,
    width: int,
    object_colors: list[int],
    parameters: dict,
    separated: bool,
) -> list[tuple[int, int]]:
    active_offsets = {
        parameters["cardinal_seed"]: _CARDINAL,
        parameters["diagonal_seed"]: _DIAGONAL,
    }
    candidates = [(row, column) for row in range(height) for column in range(width)]
    order = sorted(object_colors, key=lambda color: color not in active_offsets)
    for _ in range(24):
        centers: list[tuple[int, int]] = []
        written: set[tuple[int, int]] = set()
        for color in order:
            offsets = active_offsets.get(color, ())
            for index in rng.permutation(len(candidates)):
                row, column = candidates[int(index)]
                if offsets and not (0 < row < height - 1 and 0 < column < width - 1):
                    continue
                if any(max(abs(row - other_row), abs(column - other_column)) < (3 if separated else 2)
                       for other_row, other_column in centers):
                    continue
                halo = {(row + delta_row, column + delta_column)
                        for delta_row, delta_column in offsets}
                if (row, column) in written or halo & (written | set(centers)):
                    continue
                centers.append((row, column))
                written.update(halo)
                break
            else:
                break
        else:
            return centers
    raise ValueError("cannot place nonconflicting isolated objects")


def _construct_output(
    input_grid: np.ndarray,
    parameters: dict,
) -> np.ndarray:
    background = _dominant_background(input_grid)
    output_grid = input_grid.copy()

    roles = (
        (
            parameters["cardinal_seed"],
            parameters["cardinal_halo"],
            _CARDINAL,
        ),
        (
            parameters["diagonal_seed"],
            parameters["diagonal_halo"],
            _DIAGONAL,
        ),
    )

    for seed_color, halo_color, offsets in roles:
        for row, column in np.argwhere(input_grid == seed_color):
            for delta_row, delta_column in offsets:
                target_row = int(row + delta_row)
                target_column = int(column + delta_column)

                if not (0 <= target_row < input_grid.shape[0] and
                        0 <= target_column < input_grid.shape[1]):
                    raise ValueError("cropped active glyph")

                if input_grid[target_row, target_column] != background:
                    raise ValueError(
                        "an object blocks a required halo cell"
                    )
                if output_grid[target_row, target_column] != background:
                    raise ValueError("two generated halos overlap")

                output_grid[target_row, target_column] = halo_color

    return output_grid


def _sample_dimensions(rng: np.random.Generator, anchor: bool) -> tuple[int, int]:
    if anchor:
        return int(rng.integers(9, 16)), int(rng.integers(9, 16))
    mode = int(rng.integers(0, 5))
    if mode == 0:
        height, width = 3, int(rng.integers(7, 19))
    elif mode == 1:
        height, width = int(rng.integers(4, 9)), int(rng.integers(7, 17))
    elif mode == 2:
        height, width = int(rng.integers(9, 19)), int(rng.integers(9, 19))
    elif mode == 3:
        height, width = int(rng.integers(18, 31)), int(rng.integers(9, 20))
    else:
        height, width = int(rng.integers(19, 31)), int(rng.integers(19, 31))
    if rng.integers(0, 2):
        height, width = width, height
    return height, width


def _sample_composition(
    rng: np.random.Generator, height: int, width: int, parameters: dict,
    regime: str, train_index: int | None,
) -> list[int]:
    if train_index == 0:
        return [parameters["cardinal_seed"], parameters["diagonal_seed"]]
    capacity = max(2, min(12, height * width // 25))
    if height == 3 or width == 3:
        capacity = min(capacity, max(2, max(height, width) // 3))
    upper = min(8, capacity)
    if regime == "target_test" and rng.integers(0, 3) == 0:
        cardinal = int(rng.integers(0, upper + 1))
        diagonal = int(rng.integers(0, upper + 1))
    else:
        cardinal = int(rng.integers(0, min(4, upper) + 1))
        diagonal = int(rng.integers(0, min(4, upper) + 1))
    if cardinal + diagonal > capacity:
        return _sample_composition(rng, height, width, parameters, regime, train_index)
    if train_index == 1:
        inert = list(parameters["inert_colors"])
    else:
        inert = []
        for color in parameters["inert_colors"]:
            inert.extend([color] * int(rng.integers(0, 4)))
    if not cardinal + diagonal + len(inert):
        return _sample_composition(rng, height, width, parameters, regime, train_index)
    if len(inert) + cardinal + diagonal > max(3, height * width // 6):
        return _sample_composition(rng, height, width, parameters, regime, train_index)
    return ([parameters["cardinal_seed"]] * cardinal
            + [parameters["diagonal_seed"]] * diagonal + inert)


def generate_pair(
    rng: np.random.Generator,
    regime: str,
    puzzle_parameters: dict,
    train_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError("unknown pair regime")
    for _ in range(100):
        height, width = _sample_dimensions(rng, train_index == 0)
        object_colors = _sample_composition(
            rng, height, width, puzzle_parameters, regime, train_index,
        )
        try:
            positions = _sample_positions(
                rng, height, width, object_colors, puzzle_parameters,
                separated=train_index == 0 or (regime == "source_train" and rng.integers(0, 4) == 0),
            )
        except ValueError:
            continue
        backgrounds = puzzle_parameters["backgrounds"]
        background = backgrounds[0] if train_index == 0 or rng.integers(0, 2) else backgrounds[int(rng.integers(len(backgrounds)))]
        input_grid = np.full((height, width), background, dtype=np.int8)
        for (row, column), color in zip(positions, object_colors):
            input_grid[row, column] = color
        return input_grid, _construct_output(input_grid, puzzle_parameters)
    raise ValueError("could not sample a feasible pair")


def _dominant_background(grid: np.ndarray) -> int:
    counts = np.bincount(
        grid.ravel().astype(np.int64),
        minlength=10,
    )
    maximum = int(counts.max())

    assert int(np.count_nonzero(counts == maximum)) == 1
    return int(np.argmax(counts))


def _object_positions(
    input_grid: np.ndarray,
    background: int,
    allowed_colors: set[int] | None = None,
) -> list[tuple[int, int]]:
    positions = [
        (int(row), int(column))
        for row, column in np.argwhere(input_grid != background)
    ]

    assert positions
    for row, column in positions:
        if allowed_colors is not None:
            assert int(input_grid[row, column]) in allowed_colors

    for index, (row_a, column_a) in enumerate(positions):
        for row_b, column_b in positions[index + 1 :]:
            assert max(
                abs(row_a - row_b),
                abs(column_a - column_b),
            ) >= 2

    return positions


def _infer_parameters_from_training(
    train_pairs: list[dict],
) -> dict:
    anchor = train_pairs[0]
    input_grid, output_grid = anchor["input"], anchor["output"]
    background = _dominant_background(input_grid)
    positions = _object_positions(input_grid, background)
    assert len(positions) == 2
    assert max(abs(positions[0][axis] - positions[1][axis]) for axis in (0, 1)) >= 3
    roles: dict[str, tuple[int, int]] = {}
    for row, column in positions:
        assert 0 < row < input_grid.shape[0] - 1
        assert 0 < column < input_grid.shape[1] - 1
        assert output_grid[row, column] == input_grid[row, column]
        for kind, offsets, others in (("cardinal", _CARDINAL, _DIAGONAL),
                                       ("diagonal", _DIAGONAL, _CARDINAL)):
            values = [int(output_grid[row + dr, column + dc]) for dr, dc in offsets]
            if len(set(values)) == 1 and values[0] != background and all(
                output_grid[row + dr, column + dc] == background for dr, dc in others
            ):
                assert kind not in roles
                roles[kind] = int(input_grid[row, column]), values[0]
    assert set(roles) == {"cardinal", "diagonal"}
    backgrounds = {_dominant_background(pair["input"]) for pair in train_pairs}
    centers = {int(pair["input"][row, column])
               for pair in train_pairs
               for row, column in _object_positions(
                   pair["input"], _dominant_background(pair["input"])
               )}
    active = {roles["cardinal"][0], roles["diagonal"][0]}
    inert = centers - active
    assert len(inert) <= 3
    role_colors = active | {roles["cardinal"][1], roles["diagonal"][1]} | inert
    assert len(role_colors) == 4 + len(inert)
    assert not role_colors & backgrounds
    parameters = {
        "cardinal_seed": roles["cardinal"][0],
        "diagonal_seed": roles["diagonal"][0],
        "cardinal_halo": roles["cardinal"][1],
        "diagonal_halo": roles["diagonal"][1],
        "inert_colors": tuple(sorted(inert)),
    }
    assert np.array_equal(output_grid, _construct_output(input_grid, parameters))
    return parameters


def _validate_pair_against_parameters(
    pair: dict,
    parameters: dict,
) -> dict[int, int]:
    input_grid = pair["input"]
    output_grid = pair["output"]
    background = _dominant_background(input_grid)

    center_colors = {
        parameters["cardinal_seed"],
        parameters["diagonal_seed"],
        *parameters["inert_colors"],
    }

    positions = _object_positions(
        input_grid,
        background,
        center_colors,
    )

    for row, column in positions:
        if int(input_grid[row, column]) in (
            parameters["cardinal_seed"], parameters["diagonal_seed"]
        ):
            assert 0 < row < input_grid.shape[0] - 1
            assert 0 < column < input_grid.shape[1] - 1

    try:
        expected = _construct_output(input_grid, parameters)
    except ValueError as error:
        raise AssertionError(str(error)) from error
    assert np.array_equal(output_grid, expected)

    counts = {color: 0 for color in center_colors}
    for row, column in positions:
        counts[int(input_grid[row, column])] += 1

    return counts


def _validate_recoverability(
    puzzle: dict,
    parameters: dict,
) -> None:
    train_center_colors: set[int] = set()
    changed_output_colors: set[int] = set()

    for pair in puzzle["train"]:
        input_grid = pair["input"]
        output_grid = pair["output"]

        train_center_colors.update(
            int(value)
            for value in np.unique(input_grid)
            if int(value) != _dominant_background(input_grid)
        )
        changed_output_colors.update(
            int(value)
            for value in output_grid[input_grid != output_grid]
        )

    required_centers = {
        parameters["cardinal_seed"],
        parameters["diagonal_seed"],
        *parameters["inert_colors"],
    }

    assert required_centers <= train_center_colors
    assert parameters["cardinal_halo"] in changed_output_colors
    assert parameters["diagonal_halo"] in changed_output_colors

    for pair in puzzle["test"]:
        test_colors = {
            int(value)
            for value in np.unique(pair["input"])
            if int(value) != _dominant_background(pair["input"])
        }
        assert test_colors <= train_center_colors


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict)
    assert set(puzzle.keys()) == {"train", "test"}
    assert isinstance(puzzle["train"], list)
    assert isinstance(puzzle["test"], list)

    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()

    for split in ("train", "test"):
        for pair in puzzle[split]:
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
            pair_key = (
                input_key,
                _grid_key(pair["output"]),
            )

            assert input_key not in seen_inputs
            assert pair_key not in seen_pairs
            assert pair_key not in _SOURCE_PAIR_KEYS

            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)

    parameters = _infer_parameters_from_training(puzzle["train"])

    role_colors = {
        parameters["cardinal_seed"],
        parameters["diagonal_seed"],
        parameters["cardinal_halo"],
        parameters["diagonal_halo"],
        *parameters["inert_colors"],
    }
    assert len(role_colors) == 4 + len(parameters["inert_colors"])
    assert not role_colors & {
        _dominant_background(pair["input"])
        for split in ("train", "test") for pair in puzzle[split]
    }

    for split in ("train", "test"):
        for pair in puzzle[split]:
            _validate_pair_against_parameters(
                pair,
                parameters,
            )

    _validate_recoverability(puzzle, parameters)


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
    parameters = sample_puzzle_parameters(rng)
    puzzle = {"train": [], "test": []}
    seen_inputs: set[tuple] = set()
    for split, count in (("train", num_train), ("test", num_test)):
        regime = "source_train" if split == "train" else "target_test"
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            for _ in range(100):
                input_grid, output_grid = generate_pair(
                    pair_rng, regime, parameters,
                    train_index=index if split == "train" else None,
                )
                input_key = _grid_key(input_grid)
                pair_key = input_key, _grid_key(output_grid)
                if input_key not in seen_inputs and pair_key not in _SOURCE_PAIR_KEYS:
                    break
            else:
                raise ValueError("could not generate a unique pair")
            seen_inputs.add(input_key)
            puzzle[split].append({"input": input_grid, "output": output_grid})

    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        tuple(
            (
                _grid_key(pair["input"]),
                _grid_key(pair["output"]),
            )
            for pair in puzzle[split]
        )
        for split in ("train", "test")
    )


def _puzzles_equal(
    left: dict,
    right: dict,
) -> bool:
    if set(left) != {"train", "test"}:
        return False
    if set(right) != {"train", "test"}:
        return False

    for split in ("train", "test"):
        if len(left[split]) != len(right[split]):
            return False

        for left_pair, right_pair in zip(
            left[split],
            right[split],
        ):
            if not np.array_equal(
                left_pair["input"],
                right_pair["input"],
            ):
                return False
            if not np.array_equal(
                left_pair["output"],
                right_pair["output"],
            ):
                return False

    return True


def _role_neutral_layout(
    pair: dict,
    parameters: dict,
) -> tuple:
    labels = {
        parameters["cardinal_seed"]: 1,
        parameters["diagonal_seed"]: 2,
        **{color: index + 3 for index, color in enumerate(parameters["inert_colors"])},
    }
    background = _dominant_background(pair["input"])

    objects = tuple(
        sorted(
            (
                labels[int(pair["input"][row, column])],
                int(row),
                int(column),
            )
            for row, column in np.argwhere(
                pair["input"] != background
            )
        )
    )

    return tuple(pair["input"].shape), objects


def self_test() -> None:
    puzzle_keys: set[tuple] = set()
    grid_keys: set[tuple] = set()
    layout_keys: set[tuple] = set()
    shape_keys: set[tuple] = set()
    parameter_keys: set[tuple] = set()
    train_counts: set[int] = set()
    test_counts: set[int] = set()
    inert_vocabularies: set[int] = set()
    active_compositions: set[tuple[int, int]] = set()
    backgrounds_changed = 0
    border_inert = 0
    close_centers = 0
    repeated_inert = 0
    mixed_inert = 0
    one_active_role = 0
    inert_only = 0
    small_grid = 0
    large_grid = 0
    dense_grid = 0
    matched_regimes = 0
    test_more = 0
    test_less = 0
    source_count_test = 0
    border_halo = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_keys.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        parameters = _infer_parameters_from_training(puzzle["train"])
        inert_vocabularies.add(len(parameters["inert_colors"]))
        parameter_keys.add(
            (
                parameters["cardinal_seed"],
                parameters["diagonal_seed"],
                parameters["cardinal_halo"],
                parameters["diagonal_halo"],
                *parameters["inert_colors"],
            )
        )
        _validate_recoverability(puzzle, parameters)
        all_pairs = puzzle["train"] + puzzle["test"]
        backgrounds_changed += len({
            _dominant_background(pair["input"]) for pair in all_pairs
        }) > 1
        train_totals = []
        test_totals = []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                input_grid, output_grid = pair["input"], pair["output"]
                grid_keys.add(_grid_key(input_grid))
                shape_keys.add(tuple(input_grid.shape))
                layout_keys.add(_role_neutral_layout(pair, parameters))
                assert (_grid_key(input_grid), _grid_key(output_grid)) not in _SOURCE_PAIR_KEYS
                background = _dominant_background(input_grid)
                centers = [(int(row), int(column)) for row, column in np.argwhere(input_grid != background)]
                expected_writes = {}
                counts = {color: int(np.count_nonzero(input_grid == color))
                          for color in (parameters["cardinal_seed"], parameters["diagonal_seed"],
                                        *parameters["inert_colors"])}
                cardinal = counts[parameters["cardinal_seed"]]
                diagonal = counts[parameters["diagonal_seed"]]
                active_compositions.add((cardinal, diagonal))
                (train_totals if split == "train" else test_totals).append(cardinal + diagonal)
                source_count_test += split == "test" and (cardinal, diagonal) == (1, 2)
                one_active_role += (cardinal == 0) != (diagonal == 0)
                inert_only += cardinal + diagonal == 0
                repeated_inert += any(counts[color] > 1 for color in parameters["inert_colors"])
                mixed_inert += sum(counts[color] > 0 for color in parameters["inert_colors"]) > 1
                small_grid += min(input_grid.shape) <= 6
                large_grid += max(input_grid.shape) >= 25
                dense_grid += len(centers) >= 9
                for row, column in centers:
                    color = int(input_grid[row, column])
                    if color in parameters["inert_colors"]:
                        border_inert += row in (0, input_grid.shape[0] - 1) or column in (0, input_grid.shape[1] - 1)
                    for other_row, other_column in centers:
                        close_centers += (row, column) < (other_row, other_column) and max(
                            abs(row - other_row), abs(column - other_column)
                        ) == 2
                    if color == parameters["cardinal_seed"]:
                        offsets, halo = _CARDINAL, parameters["cardinal_halo"]
                    elif color == parameters["diagonal_seed"]:
                        offsets, halo = _DIAGONAL, parameters["diagonal_halo"]
                    else:
                        offsets, halo = (), None
                    assert output_grid[row, column] == color
                    for delta_row, delta_column in offsets:
                        target = row + delta_row, column + delta_column
                        assert target not in expected_writes
                        assert input_grid[target] == background
                        expected_writes[target] = halo
                changed = {(int(row), int(column)) for row, column in np.argwhere(input_grid != output_grid)}
                assert changed == set(expected_writes)
                assert all(output_grid[position] == halo for position, halo in expected_writes.items())
                border_halo += any(row in (0, input_grid.shape[0] - 1) or
                                   column in (0, input_grid.shape[1] - 1)
                                   for row, column in changed)
        matched_regimes += bool(set(train_totals) & set(test_totals))
        test_more += max(test_totals) > max(train_totals)
        test_less += min(test_totals) < min(train_totals)

    assert len(train_counts) >= 4
    assert len(test_counts) >= 3
    assert len(puzzle_keys) >= 40
    assert len(parameter_keys) >= 30
    assert len(grid_keys) >= 200
    assert len(shape_keys) >= 35
    assert len(layout_keys) >= 120
    assert inert_vocabularies == {0, 1, 2, 3}
    assert len(active_compositions) >= 18
    assert backgrounds_changed >= 10
    assert border_inert and close_centers and repeated_inert and mixed_inert
    assert one_active_role and inert_only and small_grid and large_grid and dense_grid
    assert matched_regimes and test_more and test_less
    assert source_count_test and border_halo

    explicit_a = generate_puzzle(314159, 7, 3)
    explicit_b = generate_puzzle(314159, 7, 3)

    validate_generated_puzzle(explicit_a)
    assert _puzzles_equal(explicit_a, explicit_b)

    different_seed = generate_puzzle(314160, 7, 3)
    assert not _puzzles_equal(explicit_a, different_seed)

    damaged = {
        split: [{role: grid.copy() for role, grid in pair.items()}
                for pair in explicit_a[split]]
        for split in ("train", "test")
    }
    damaged["test"][0]["output"][0, 0] = -1
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted an invalid output")

    # Legal boundary cases.
    boundary_cases = (
        (2, 1),
        (10, 1),
        (10, 4),
    )
    for train_count, test_count in boundary_cases:
        puzzle = generate_puzzle(
            1000 + 10 * train_count + test_count,
            train_count,
            test_count,
        )
        validate_generated_puzzle(puzzle)
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count

    # Partial count specification.
    partial_cases = (
        (None, 1),
        (None, 4),
        (2, None),
        (10, None),
    )
    for index, (train_count, test_count) in enumerate(
        partial_cases
    ):
        puzzle = generate_puzzle(
            2000 + index,
            train_count,
            test_count,
        )
        validate_generated_puzzle(puzzle)

        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count

    # Illegal counts.
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
        try:
            generate_puzzle(
                999,
                train_count,
                test_count,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(
                "illegal pair counts did not raise ValueError"
            )
