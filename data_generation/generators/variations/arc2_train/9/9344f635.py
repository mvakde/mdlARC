import numpy as np


_SOURCE_PAIRS = (
    (
        (
            "77777777",
            "79777557",
            "79755777",
            "77777777",
            "77777777",
            "77777777",
            "77777447",
            "77777777",
        ),
        (
            "79777777",
            "55555555",
            "55555555",
            "79777777",
            "79777777",
            "79777777",
            "44444444",
            "79777777",
        ),
    ),
    (
        (
            "7777777",
            "7767777",
            "7767277",
            "7777277",
            "7557777",
            "7777117",
            "7777777",
        ),
        (
            "7767277",
            "7767277",
            "7767277",
            "7767277",
            "5555555",
            "1111111",
            "7767277",
        ),
    ),
    (
        (
            "777777777",
            "077777777",
            "077717337",
            "777717777",
            "777777777",
            "777777007",
            "777777777",
            "766777777",
            "777777777",
        ),
        (
            "077717777",
            "077717777",
            "333333333",
            "077717777",
            "077717777",
            "000000000",
            "077717777",
            "666666666",
            "077717777",
        ),
    ),
    (
        (
            "7777777777",
            "7777773717",
            "7887773717",
            "7777777777",
            "7778877777",
            "7577777777",
            "7577777777",
            "7777278877",
            "7777277777",
            "7777777777",
        ),
        (
            "7577273717",
            "7577273717",
            "8888888888",
            "7577273717",
            "8888888888",
            "7577273717",
            "7577273717",
            "8888888888",
            "7577273717",
            "7577273717",
        ),
    ),
)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)):
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    result = int(seed)
    if result < 0:
        raise ValueError("seed must be a non-negative integer")
    return result


def _validate_optional_count(
    name: str,
    value: int | None,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer")
    if not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(
            f"{name} must be between {lower} and {upper}, inclusive"
        )
    return result


def _resolve_counts(
    seed: int,
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
        test_count = 1 + seed % 4
        quotient = seed // 4
        train_count = test_count + 1 + quotient % (10 - test_count)
    elif train_count is None:
        lower = max(2, test_count + 1)
        train_count = lower + seed % (11 - lower)
    elif test_count is None:
        upper = min(4, train_count - 1)
        test_count = 1 + seed % upper

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _rows_key(grid: np.ndarray) -> tuple[str, ...]:
    return tuple(
        "".join(str(int(value)) for value in row)
        for row in grid
    )


def _is_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    return (_rows_key(input_grid), _rows_key(output_grid)) in _SOURCE_PAIRS


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.ravel()),
    )


def _pair_key(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple:
    return _grid_key(input_grid), _grid_key(output_grid)


def _background_from_grid(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    maximum = int(np.max(counts))
    assert int(np.sum(counts == maximum)) == 1
    return int(values[int(np.argmax(counts))])


def _analyze_input(
    grid: np.ndarray,
    expected_background: int | None = None,
) -> dict:
    background = _background_from_grid(grid)
    if expected_background is not None:
        assert background == expected_background

    horizontal = []
    vertical = []

    for raw_color in np.unique(grid):
        color = int(raw_color)
        if color == background:
            continue

        remaining = {
            (int(row), int(column))
            for row, column in np.argwhere(grid == color)
        }

        while remaining:
            start = next(iter(remaining))
            remaining.remove(start)
            component = []
            stack = [start]

            while stack:
                row, column = stack.pop()
                component.append((row, column))
                for neighbor in (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                ):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)

            component.sort()
            assert len(component) == 2
            (r1, c1), (r2, c2) = component
            assert abs(r1 - r2) + abs(c1 - c2) == 1

            cells = tuple(component)
            if r1 == r2:
                horizontal.append((r1, color, cells))
            else:
                vertical.append((c1, color, cells))

    row_colors = {}
    for row, color, _ in horizontal:
        if row in row_colors:
            assert row_colors[row] == color
        row_colors[row] = color

    column_colors = {}
    for column, color, _ in vertical:
        if column in column_colors:
            assert column_colors[column] == color
        column_colors[column] = color

    horizontal.sort(key=lambda item: (item[0], item[1], item[2]))
    vertical.sort(key=lambda item: (item[0], item[1], item[2]))

    return {
        "background": background,
        "horizontal": horizontal,
        "vertical": vertical,
    }


def _construct_output(
    input_grid: np.ndarray,
    analysis: dict,
) -> np.ndarray:
    output = np.full(
        input_grid.shape,
        analysis["background"],
        dtype=np.int8,
    )

    for column, color, _ in analysis["vertical"]:
        output[:, column] = color

    for row, color, _ in analysis["horizontal"]:
        output[row, :] = color

    return output


def _layout_key_for_background(
    grid: np.ndarray,
    background: int,
) -> tuple:
    mask = (grid != background).astype(np.uint8)
    return tuple(int(x) for x in grid.shape), mask.tobytes()


def _layout_key(grid: np.ndarray) -> tuple:
    return _layout_key_for_background(
        grid,
        _background_from_grid(grid),
    )


def _sample_spaced_indices(
    rng: np.random.Generator,
    size: int,
    count: int,
) -> list[int]:
    assert count >= 1
    assert size >= 2 * count - 1

    for _ in range(256):
        values = np.sort(
            rng.choice(size, size=count, replace=False)
        )
        if count == 1 or np.all(np.diff(values) >= 2):
            return [int(value) for value in values]

    base = np.arange(count, dtype=int) * 2
    slack = size - (2 * count - 1)
    offset = int(rng.integers(0, slack + 1))
    return [int(value + offset) for value in base]


def _place_dominoes(
    rng: np.random.Generator,
    height: int,
    width: int,
    descriptors: list[tuple[str, int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    candidate_lists = []

    for orientation, fixed_index, _ in descriptors:
        candidates = []
        if orientation == "h":
            for column in range(width - 1):
                candidates.append(
                    (
                        (fixed_index, column),
                        (fixed_index, column + 1),
                    )
                )
        else:
            for row in range(height - 1):
                candidates.append(
                    (
                        (row, fixed_index),
                        (row + 1, fixed_index),
                    )
                )

        order = rng.permutation(len(candidates))
        candidate_lists.append(
            [candidates[int(index)] for index in order]
        )

    occupied = {}
    placement = [None] * len(descriptors)

    search_budget = 12000

    def search(index: int) -> bool:
        nonlocal search_budget
        search_budget -= 1
        if search_budget < 0:
            return False
        if index == len(descriptors):
            return True

        _, _, color = descriptors[index]

        for cells in candidate_lists[index]:
            if any(cell in occupied for cell in cells):
                continue

            invalid = False
            for row, column in cells:
                for neighbor in (
                    (row - 1, column),
                    (row + 1, column),
                    (row, column - 1),
                    (row, column + 1),
                ):
                    if occupied.get(neighbor) == color:
                        invalid = True
                        break
                if invalid:
                    break

            if invalid:
                continue

            for cell in cells:
                occupied[cell] = color
            placement[index] = cells

            if search(index + 1):
                return True

            for cell in cells:
                del occupied[cell]
            placement[index] = None

        return False

    if not search(0):
        return None

    return list(placement)


def _sample_line_counts(rng: np.random.Generator, regime: str, anchor: bool) -> tuple[int, int]:
    maximum = 4 if regime == "source_train" else 5
    for _ in range(30):
        if anchor:
            horizontal = int(rng.integers(1, 5))
            vertical = int(rng.integers(1, 5))
        else:
            mode = int(rng.integers(0, 6))
            horizontal = 0 if mode == 0 else int(rng.integers(1, maximum + 1))
            vertical = 0 if mode == 1 else int(rng.integers(1, maximum + 1))
        if 1 <= horizontal + vertical <= (7 if regime == "source_train" else 9):
            return horizontal, vertical
    return (2, 2) if anchor else (1, 2)


def _sample_shape(rng: np.random.Generator, regime: str) -> tuple[int, int]:
    high = 15 if regime == "source_train" else 18
    height = int(rng.integers(7, high))
    width = height if rng.random() < 0.2 else int(rng.integers(7, high))
    return height, width


def _choose_colors(
    rng: np.random.Generator,
    background: int,
    horizontal_count: int,
    vertical_count: int,
    anchor: bool,
) -> tuple[list[int], list[int]]:
    available = [color for color in range(10) if color != background]
    maximum = min(6, horizontal_count + vertical_count + 1)
    palette_size = int(rng.integers(2 if anchor else 1, maximum + 1))
    palette = [int(value) for value in rng.choice(available, size=palette_size, replace=False)]
    horizontal = [int(rng.choice(palette)) for _ in range(horizontal_count)]
    vertical = [int(rng.choice(palette)) for _ in range(vertical_count)]
    if anchor and not any(h != v for h in horizontal for v in vertical):
        vertical[0] = next(color for color in palette if color != horizontal[0])
    return horizontal, vertical


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    background: int,
    anchor: bool = False,
    allow_redundancy: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    if regime not in ("source_train", "target_test"):
        raise ValueError(f"unknown regime: {regime}")

    for _ in range(160):
        height, width = _sample_shape(rng, regime)
        horizontal_count, vertical_count = _sample_line_counts(rng, regime, anchor)
        horizontal_rows = [int(value) for value in rng.choice(height, size=horizontal_count, replace=False)]
        vertical_columns = [int(value) for value in rng.choice(width, size=vertical_count, replace=False)]
        horizontal_colors, vertical_colors = _choose_colors(
            rng, background, horizontal_count, vertical_count, anchor
        )
        descriptors = [
            ("h", row, color) for row, color in zip(horizontal_rows, horizontal_colors)
        ] + [
            ("v", column, color) for column, color in zip(vertical_columns, vertical_colors)
        ]
        if allow_redundancy and rng.random() < 0.23:
            descriptors.append(descriptors[int(rng.integers(0, len(descriptors)))])
        descriptors = [descriptors[int(index)] for index in rng.permutation(len(descriptors))]
        placement = _place_dominoes(rng, height, width, descriptors)
        if placement is None:
            continue
        input_grid = np.full((height, width), background, dtype=np.int8)
        for (_, _, color), cells in zip(descriptors, placement):
            for row, column in cells:
                input_grid[row, column] = color
        if int(np.count_nonzero(input_grid == background)) <= input_grid.size // 2:
            continue
        try:
            analysis = _analyze_input(input_grid, expected_background=background)
        except AssertionError:
            continue
        if len(analysis["horizontal"]) != sum(axis == "h" for axis, _, _ in descriptors):
            continue
        if len(analysis["vertical"]) != sum(axis == "v" for axis, _, _ in descriptors):
            continue
        output_grid = _construct_output(input_grid, analysis)
        if np.array_equal(input_grid, output_grid):
            continue
        return input_grid, output_grid
    raise ValueError("unable to construct an unambiguous pair")


def _generate_unique_pair(
    rng: np.random.Generator,
    regime: str,
    background: int,
    anchor: bool,
    allow_redundancy: bool,
    used_inputs: set,
    used_layouts: set,
) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(30):
        input_grid, output_grid = _generate_pair(
            rng, regime, background, anchor, allow_redundancy
        )
        input_key = _grid_key(input_grid)
        layout_key = _layout_key_for_background(input_grid, background)
        if input_key in used_inputs or layout_key in used_layouts:
            continue
        if _is_source_pair(input_grid, output_grid):
            continue
        used_inputs.add(input_key)
        used_layouts.add(layout_key)
        return input_grid, output_grid
    raise ValueError("unable to construct enough distinct pairs")


def _has_redundancy(analysis: dict) -> bool:
    return (
        len({row for row, _, _ in analysis["horizontal"]}) < len(analysis["horizontal"])
        or len({column for column, _, _ in analysis["vertical"]}) < len(analysis["vertical"])
    )


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed_value = _validate_seed(seed)
    train_count, test_count = _resolve_counts(seed_value, num_train, num_test)
    rng = np.random.default_rng(seed_value)
    child_seeds = rng.integers(0, 2**63 - 1, size=train_count + test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    used_inputs: set = set()
    used_layouts: set = set()
    taught_redundancy = False
    first_background = None

    for index in range(train_count + test_count):
        split = "train" if index < train_count else "test"
        regime = "source_train" if split == "train" else "target_test"
        pair_rng = np.random.default_rng(int(child_seeds[index]))
        background_options = list(range(10))
        if index == 1:
            background_options.remove(first_background)
        background = int(pair_rng.choice(background_options))
        if index == 0:
            first_background = background
        input_grid, output_grid = _generate_unique_pair(
            pair_rng,
            regime,
            background,
            anchor=index == 0,
            allow_redundancy=(split == "train" and index > 0) or (
                split == "test" and taught_redundancy
            ),
            used_inputs=used_inputs,
            used_layouts=used_layouts,
        )
        puzzle[split].append({"input": input_grid, "output": output_grid})
        if split == "train" and _has_redundancy(_analyze_input(input_grid)):
            taught_redundancy = True

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])

    input_keys, pair_keys, layout_keys = set(), set(), set()
    train_backgrounds = set()
    training_priority = False
    taught_redundancy = False
    test_redundancy = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= dimension <= 30 for dimension in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert bool(np.all((0 <= grid) & (grid <= 9)))
            assert inp.shape == out.shape
            input_key, pair_key = _grid_key(inp), _pair_key(inp, out)
            assert input_key not in input_keys and pair_key not in pair_keys
            input_keys.add(input_key)
            pair_keys.add(pair_key)
            assert not _is_source_pair(inp, out)
            analysis = _analyze_input(inp)
            background = analysis["background"]
            assert int(np.count_nonzero(inp == background)) > inp.size // 2
            layout_key = _layout_key_for_background(inp, background)
            assert layout_key not in layout_keys
            layout_keys.add(layout_key)
            horizontal, vertical = analysis["horizontal"], analysis["vertical"]
            assert horizontal or vertical
            assert len({row for row, _, _ in horizontal}) <= 5
            assert len({column for column, _, _ in vertical}) <= 5
            assert all(
                sum(other_row == row for other_row, _, _ in horizontal) <= 2
                for row, _, _ in horizontal
            )
            assert all(
                sum(other_column == column for other_column, _, _ in vertical) <= 2
                for column, _, _ in vertical
            )
            assert np.array_equal(out, _construct_output(inp, analysis))
            assert not np.array_equal(inp, out)
            if split == "train":
                train_backgrounds.add(background)
                assert 7 <= inp.shape[0] <= 14 and 7 <= inp.shape[1] <= 14
                assert len({row for row, _, _ in horizontal}) + len({column for column, _, _ in vertical}) <= 7
                taught_redundancy |= _has_redundancy(analysis)
                if any(hcolor != vcolor for _, hcolor, _ in horizontal for _, vcolor, _ in vertical):
                    training_priority = True
            else:
                assert 7 <= inp.shape[0] <= 17 and 7 <= inp.shape[1] <= 17
                assert len({row for row, _, _ in horizontal}) + len({column for column, _, _ in vertical}) <= 9
                test_redundancy |= _has_redundancy(analysis)
    assert len(train_backgrounds) >= 2
    assert training_priority
    assert not test_redundancy or taught_redundancy
    first = _analyze_input(puzzle["train"][0]["input"])
    assert first["horizontal"] and first["vertical"]
    assert any(hcolor != vcolor for _, hcolor, _ in first["horizontal"] for _, vcolor, _ in first["vertical"])


def _puzzles_equal(first: dict, second: dict) -> bool:
    if set(first) != set(second):
        return False
    for split in ("train", "test"):
        if len(first[split]) != len(second[split]):
            return False
        for left, right in zip(first[split], second[split]):
            if not all(np.array_equal(left[role], right[role]) for role in ("input", "output")):
                return False
    return True


def _predict_from_demonstrations(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    horizontal_first_possible = True
    vertical_first_possible = True
    for pair in train:
        analysis = _analyze_input(pair["input"])
        for row, horizontal_color, _ in analysis["horizontal"]:
            for column, vertical_color, _ in analysis["vertical"]:
                observed = int(pair["output"][row, column])
                horizontal_first_possible &= observed == vertical_color
                vertical_first_possible &= observed == horizontal_color
    assert horizontal_first_possible != vertical_first_possible
    analysis = _analyze_input(test_input)
    predicted = np.full(test_input.shape, analysis["background"], dtype=np.int8)
    if vertical_first_possible:
        for column, color, _ in analysis["vertical"]:
            predicted[:, column] = color
        for row, color, _ in analysis["horizontal"]:
            predicted[row, :] = color
    else:
        for row, color, _ in analysis["horizontal"]:
            predicted[row, :] = color
        for column, color, _ in analysis["vertical"]:
            predicted[:, column] = color
    return predicted


def self_test() -> None:
    cases = [
        (918273, None, None), (424242, 7, 3), (6021, 2, 1),
        (6101, 10, 1), (6104, 10, 4), (7000, None, 1),
        (7001, None, 4), (7002, 2, None), (7003, 10, None),
    ]
    for seed, train_count, test_count in cases:
        first = generate_puzzle(seed, train_count, test_count)
        second = generate_puzzle(seed, train_count, test_count)
        assert _puzzles_equal(first, second)
        if train_count is not None:
            assert len(first["train"]) == train_count
        if test_count is not None:
            assert len(first["test"]) == test_count
        validate_generated_puzzle(first)
        for pair in first["test"]:
            assert np.array_equal(_predict_from_demonstrations(first["train"], pair["input"]), pair["output"])
    for arguments in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0},
        {"num_test": 5}, {"num_train": 2, "num_test": 2},
        {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4},
        {"num_train": 2.0, "num_test": 1}, {"num_train": True, "num_test": 1},
    ):
        try:
            generate_puzzle(8080, **arguments)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")

    train_counts, test_counts, backgrounds, shapes, structures = set(), set(), set(), set(), set()
    foreground_sizes = set()
    adjacent = shared_axis_color = repeated_vertical = redundant = 0
    row_only = column_only = mixed_test_rows = new_test_color = 0
    smaller_tests = variable_backgrounds = 0
    background_role_exchange = spanning_adjacent_rows = combined_tests = 0
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        fingerprints.add(tuple(_pair_key(pair["input"], pair["output"]) for split in ("train", "test") for pair in puzzle[split]))
        train_colors, train_areas, pair_backgrounds, train_backgrounds = set(), [], set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                analysis = _analyze_input(inp)
                assert np.array_equal(_predict_from_demonstrations(puzzle["train"], inp), pair["output"])
                background = analysis["background"]
                backgrounds.add(background)
                pair_backgrounds.add(background)
                shapes.add(inp.shape)
                structures.add(_layout_key(inp))
                foreground = set(int(value) for value in np.unique(inp)) - {background}
                foreground_sizes.add(len(foreground))
                horizontal, vertical = analysis["horizontal"], analysis["vertical"]
                rows = sorted({row for row, _, _ in horizontal})
                spanning_adjacent_rows += any(
                    {cells[0][0], cells[1][0]} <= set(rows)
                    for _, _, cells in vertical
                )
                adjacent += any(b - a == 1 for a, b in zip(rows, rows[1:]))
                shared_axis_color += bool({color for _, color, _ in horizontal} & {color for _, color, _ in vertical})
                repeated_vertical += len({color for _, color, _ in vertical}) < len({column for column, _, _ in vertical})
                redundant += _has_redundancy(analysis)
                row_only += bool(horizontal and not vertical)
                column_only += bool(vertical and not horizontal)
                if split == "train":
                    background_role_exchange += bool(foreground & train_backgrounds)
                    train_backgrounds.add(background)
                    train_colors |= foreground
                    train_areas.append(inp.size)
                else:
                    mixed_test_rows += len({color for _, color, _ in horizontal}) >= 2
                    new_test_color += bool(foreground - train_colors)
                    smaller_tests += inp.size <= max(train_areas)
                    background_role_exchange += bool(foreground & train_backgrounds)
                    combined_tests += (
                        inp.shape[0] != inp.shape[1]
                        and bool(foreground - train_colors)
                        and len({color for _, color, _ in horizontal}) >= 2
                    )
        variable_backgrounds += len(pair_backgrounds) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42 and len(backgrounds) >= 8
    assert len(shapes) >= 30 and len(structures) >= 150
    assert len(foreground_sizes) >= 5
    assert all(value > 0 for value in (
        adjacent, shared_axis_color, repeated_vertical, redundant,
        row_only, column_only, mixed_test_rows, new_test_color, smaller_tests,
        background_role_exchange, spanning_adjacent_rows, combined_tests,
    ))
    assert variable_backgrounds == 42
