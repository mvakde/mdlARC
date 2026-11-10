import numpy as np


_SOURCE_PAIR_ROWS = (
    (
        (
            "000000",
            "080000",
            "000000",
            "000000",
            "000500",
            "000000",
        ),
        (
            "000000",
            "088888",
            "000008",
            "000008",
            "000555",
            "000005",
        ),
    ),
    (
        (
            "000",
            "060",
            "000",
        ),
        (
            "000",
            "066",
            "006",
        ),
    ),
    (
        (
            "00000",
            "00800",
            "00000",
            "07000",
            "00000",
            "00600",
            "00000",
        ),
        (
            "00000",
            "00888",
            "00008",
            "07777",
            "00007",
            "00666",
            "00006",
        ),
    ),
    (
        (
            "000000",
            "002000",
            "000000",
            "030000",
            "000000",
            "000000",
        ),
        (
            "000000",
            "002222",
            "000002",
            "033333",
            "000003",
            "000003",
        ),
    ),
    (
        (
            "00080000",
            "00000000",
            "00700000",
            "00000000",
            "00000200",
            "00000000",
            "00000000",
        ),
        (
            "00088888",
            "00000008",
            "00777777",
            "00000007",
            "00000222",
            "00000002",
            "00000002",
        ),
    ),
)


def _validate_optional_count(
    value: int | None,
    name: str,
    lower: int,
    upper: int,
) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
    ):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not lower <= result <= upper:
        raise ValueError(
            f"{name} must be between {lower} and {upper}, inclusive"
        )
    return result


def _resolve_pair_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _validate_optional_count(
        num_train, "num_train", 2, 10
    )
    test_count = _validate_optional_count(
        num_test, "num_test", 1, 4
    )

    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        number_of_train_choices = 10 - test_count
        train_count = (
            test_count
            + 1
            + (seed // 4) % number_of_train_choices
        )
    elif train_count is None:
        assert test_count is not None
        number_of_train_choices = 10 - test_count
        train_count = (
            test_count
            + 1
            + seed % number_of_train_choices
        )
    elif test_count is None:
        maximum_test_count = min(4, train_count - 1)
        test_count = 1 + seed % maximum_test_count

    assert train_count is not None
    assert test_count is not None

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _background(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _seeds(grid: np.ndarray) -> list[tuple[int, int, int]]:
    background = _background(grid)
    return sorted(
        (int(row), int(col), int(grid[row, col]))
        for row, col in np.argwhere(grid != background)
    )


def _paint(input_grid: np.ndarray) -> np.ndarray:
    output = input_grid.copy()
    seeds = _seeds(input_grid)
    for index, (row, col, color) in enumerate(seeds):
        output[row, col:] = color
        stop = seeds[index + 1][0] if index + 1 < len(seeds) else len(output)
        output[row:stop, -1] = color
    return output


def _candidate_output(
    input_grid: np.ndarray, horizontal: str, vertical: str
) -> np.ndarray:
    """A separate row scan used to check what training examples teach."""
    background = _background(input_grid)
    seeds = _seeds(input_grid)
    height, width = input_grid.shape
    result = np.full((height, width), background, dtype=input_grid.dtype)
    by_row = {row: (col, color) for row, col, color in seeds}
    edge = width - 1 if horizontal == "right" else 0
    rows = range(height) if vertical == "down" else range(height - 1, -1, -1)
    active = None
    for row in rows:
        if row in by_row:
            col, active = by_row[row]
            if horizontal == "right":
                result[row, col:] = active
            else:
                result[row, :col + 1] = active
        if active is not None:
            result[row, edge] = active
    return result


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _rows_as_digits(grid: np.ndarray) -> tuple[str, ...]:
    return tuple("".join(str(int(value)) for value in row) for row in grid)


def _is_source_pair(input_grid: np.ndarray, output_grid: np.ndarray) -> bool:
    return (_rows_as_digits(input_grid), _rows_as_digits(output_grid)) in _SOURCE_PAIR_ROWS


def _choose_rows(
    rng: np.random.Generator, height: int, count: int, kind: str
) -> tuple[int, ...]:
    if kind == "single":
        return (int(rng.integers(0, height - 1)),)

    if kind == "layered":
        # Reserve a bottom cell for a visible final edge leg. The wider
        # height range permits both uneven handoffs and variable tails.
        for _ in range(24):
            rows = tuple(sorted(int(x) for x in rng.choice(height - 1, count, replace=False)))
            if any(b - a >= 2 for a, b in zip(rows, rows[1:])):
                return rows
        return tuple(2 * index for index in range(count))

    placement = int(rng.integers(0, 4))
    if placement == 0:
        rows = [0] + [int(x) for x in rng.choice(np.arange(1, height), count - 1, replace=False)]
    elif placement == 1:
        rows = [height - 1] + [int(x) for x in rng.choice(height - 1, count - 1, replace=False)]
    else:
        rows = [int(x) for x in rng.choice(height, count, replace=False)]
    return tuple(sorted(rows))


def _choose_columns(
    rng: np.random.Generator,
    rows: tuple[int, ...],
    width: int,
    kind: str,
) -> tuple[int, ...]:
    columns: list[int] = []
    for index, row in enumerate(rows):
        upper = width - 1 if kind in ("single", "layered") else width
        candidates = list(range(upper))
        if index and row == rows[index - 1] + 1:
            candidates = [col for col in candidates if abs(col - columns[-1]) > 1]
        if index + 1 < len(rows) and rows[index + 1] == row + 1:
            endpoint_candidates = [col for col in candidates if col in (0, width - 1)]
            if endpoint_candidates:
                candidates = endpoint_candidates
        if not candidates:
            # This is reachable only for a narrow interior-only layered grid.
            # Layered grids are sampled at width >= 5, so it is defensive.
            raise AssertionError("No isolated seed column available")
        if kind == "free" and width - 1 in candidates and rng.random() < 0.18:
            column = width - 1
        else:
            column = int(rng.choice(candidates))
        columns.append(column)
    return tuple(columns)


def _make_pair(
    rng: np.random.Generator,
    kind: str,
    count: int,
    background: int,
    width: int | None = None,
    test: bool = False,
) -> dict:
    minimum_height = 2 * count + 1 if kind == "layered" else max(3, count + 1)
    maximum_height = 18 if test else 15
    height = int(rng.integers(minimum_height, maximum_height + 1))
    if width is None:
        width = int(rng.integers(5 if kind == "layered" else 3, 16))
    if kind == "layered":
        width = max(5, width)
    rows = _choose_rows(rng, height, count, kind)
    columns = _choose_columns(rng, rows, width, kind)
    available_colors = [color for color in range(10) if color != background]
    colors = [int(value) for value in rng.choice(available_colors, size=count, replace=False)]
    grid = np.full((height, width), background, dtype=np.int8)
    for row, column, color in zip(rows, columns, colors):
        grid[row, column] = color
    return {"input": grid, "output": _paint(grid)}


def _test_width(rng: np.random.Generator, train_widths: list[int]) -> int:
    largest = max(train_widths)
    mode = int(rng.integers(0, 3))
    if mode == 0 and largest < 18:
        return int(rng.integers(largest + 1, min(19, largest + 5)))
    if mode == 1:
        return int(np.clip(int(rng.choice(train_widths)) + int(rng.integers(-2, 3)), 3, 18))
    if largest > 3:
        return int(rng.integers(3, largest))
    return int(rng.integers(3, 19))


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    train_count, test_count = _resolve_pair_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)

    # Direction, turn edge, and replacement order are the source conventions.
    # Colors and dimensions are local to a pair and visible in that pair.
    train_kinds = ["single", "layered"] + ["free"] * (train_count - 2)
    rng.shuffle(train_kinds)
    first_background = int(rng.integers(0, 10))
    second_background = int((first_background + rng.integers(1, 10)) % 10)
    train_backgrounds = [first_background, second_background]
    train_backgrounds.extend(int(rng.integers(0, 10)) for _ in range(train_count - 2))
    rng.shuffle(train_backgrounds)

    puzzle = {"train": [], "test": []}
    seen_inputs: set[tuple] = set()

    def append_pair(section: str, kind: str, count: int, background: int, width: int | None = None) -> None:
        for _ in range(80):
            pair = _make_pair(rng, kind, count, background, width, section == "test")
            key = _grid_key(pair["input"])
            if key in seen_inputs or _is_source_pair(pair["input"], pair["output"]):
                continue
            if np.array_equal(pair["input"], pair["output"]):
                continue
            seen_inputs.add(key)
            puzzle[section].append(pair)
            return
        raise AssertionError("Could not construct a unique changed pair")

    for kind, background in zip(train_kinds, train_backgrounds):
        count = 1 if kind == "single" else int(rng.integers(3, 5)) if kind == "layered" else int(rng.integers(1, 5))
        append_pair("train", kind, count, background)

    maximum_demonstrated = max(len(_seeds(pair["input"])) for pair in puzzle["train"])
    train_widths = [pair["input"].shape[1] for pair in puzzle["train"]]
    for _ in range(test_count):
        count = int(rng.integers(1, maximum_demonstrated + 1))
        background = int(rng.integers(0, 10))
        width = _test_width(rng, train_widths)
        append_pair("test", "free", count, background, width)

    validate_generated_puzzle(puzzle)
    return puzzle


def _validate_pair(pair: dict) -> list[tuple[int, int, int]]:
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    input_grid, output_grid = pair["input"], pair["output"]
    for grid in (input_grid, output_grid):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert int(grid.min()) >= 0 and int(grid.max()) <= 9
    assert input_grid.shape == output_grid.shape
    background = _background(input_grid)
    seeds = _seeds(input_grid)
    assert 1 <= len(seeds) <= 4
    assert input_grid.size - len(seeds) > len(seeds)
    assert len({row for row, _, _ in seeds}) == len(seeds)
    assert len({color for _, _, color in seeds}) == len(seeds)
    assert all(color != background for _, _, color in seeds)
    assert all(
        b_row - a_row > 1 or abs(b_col - a_col) > 1
        for (a_row, a_col, _), (b_row, b_col, _) in zip(seeds, seeds[1:])
    )
    assert np.array_equal(output_grid, _candidate_output(input_grid, "right", "down"))
    assert not np.array_equal(input_grid, output_grid)
    assert not _is_source_pair(input_grid, output_grid)
    return seeds


def _matching_rules(train_pairs: list[dict]) -> list[tuple[str, str]]:
    candidates = [(h, v) for h in ("left", "right") for v in ("up", "down")]
    return [
        (horizontal, vertical)
        for horizontal, vertical in candidates
        if all(
            np.array_equal(pair["output"], _candidate_output(pair["input"], horizontal, vertical))
            for pair in train_pairs
        )
    ]


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    seed_lists = []
    for pair in puzzle["train"] + puzzle["test"]:
        seeds = _validate_pair(pair)
        seed_lists.append(seeds)
        input_key = _grid_key(pair["input"])
        pair_key = (input_key, _grid_key(pair["output"]))
        assert input_key not in seen_inputs and pair_key not in seen_pairs
        seen_inputs.add(input_key)
        seen_pairs.add(pair_key)

    train_seeds = seed_lists[:train_count]
    test_seeds = seed_lists[train_count:]
    assert any(len(seeds) == 1 and seeds[0][0] < pair["input"].shape[0] - 1 and seeds[0][1] < pair["input"].shape[1] - 1 for pair, seeds in zip(puzzle["train"], train_seeds))
    assert any(len(seeds) >= 3 and any(b[0] - a[0] >= 2 for a, b in zip(seeds, seeds[1:])) and all(col < pair["input"].shape[1] - 1 for _, col, _ in seeds) for pair, seeds in zip(puzzle["train"], train_seeds))
    assert len({_background(pair["input"]) for pair in puzzle["train"]}) >= 2
    assert max(map(len, test_seeds)) <= max(map(len, train_seeds))
    matching_rules = _matching_rules(puzzle["train"])
    assert matching_rules == [("right", "down")]
    inferred_horizontal, inferred_vertical = matching_rules[0]
    for pair in puzzle["test"]:
        prediction = _candidate_output(pair["input"], inferred_horizontal, inferred_vertical)
        assert np.array_equal(prediction, pair["output"])


def _puzzle_signature(puzzle: dict) -> tuple:
    return tuple(
        (section, tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for pair in puzzle[section]))
        for section in ("train", "test")
    )


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_signature(puzzle) for puzzle in puzzles}) == 42
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    shapes: set[tuple[int, int]] = set()
    layouts: set[tuple] = set()
    colors: set[int] = set()
    backgrounds: set[int] = set()
    train_top = test_lower = narrower_tests = wider_tests = new_test_colors = 0
    right_edge = bottom_edge = adjacent = uneven_gaps = 0
    seed_counts: set[int] = set()
    training_count_orders: set[tuple[int, ...]] = set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        assert _matching_rules(puzzle["train"]) == [("right", "down")]
        train_widths = [pair["input"].shape[1] for pair in puzzle["train"]]
        training_count_orders.add(tuple(len(_seeds(pair["input"])) for pair in puzzle["train"]))
        train_colors = {color for pair in puzzle["train"] for _, _, color in _seeds(pair["input"])}
        for section in ("train", "test"):
            for pair in puzzle[section]:
                grid = pair["input"]
                seeds = _seeds(grid)
                height, width = grid.shape
                shapes.add(grid.shape)
                layouts.add((grid.shape, tuple((row, col) for row, col, _ in seeds)))
                colors.update(color for _, _, color in seeds)
                backgrounds.add(_background(grid))
                seed_counts.add(len(seeds))
                assert np.array_equal(pair["output"], _candidate_output(grid, "right", "down"))
                assert not _is_source_pair(grid, pair["output"])
                right_edge += any(col == width - 1 for _, col, _ in seeds)
                bottom_edge += any(row == height - 1 for row, _, _ in seeds)
                adjacent += any(b[0] == a[0] + 1 for a, b in zip(seeds, seeds[1:]))
                uneven_gaps += len(seeds) >= 3 and len({b[0] - a[0] for a, b in zip(seeds, seeds[1:])}) > 1
                if section == "train":
                    train_top += seeds[0][0] == 0
                else:
                    test_lower += seeds[0][0] > 0
                    narrower_tests += width < max(train_widths)
                    wider_tests += width > max(train_widths)
                    new_test_colors += any(color not in train_colors for _, _, color in seeds)
    assert len(shapes) >= 45 and len(layouts) >= 200
    assert seed_counts == {1, 2, 3, 4}
    assert colors == set(range(10)) and backgrounds == set(range(10))
    assert len(training_count_orders) >= 15
    assert train_top >= 5 and test_lower >= 10
    assert narrower_tests >= 10 and wider_tests >= 10
    assert new_test_colors >= 10
    assert right_edge >= 10 and bottom_edge >= 10 and adjacent >= 5 and uneven_gaps >= 5
    assert _puzzle_signature(generate_puzzle(314159, 7, 3)) == _puzzle_signature(generate_puzzle(314159, 7, 3))
    assert _puzzle_signature(generate_puzzle(314159)) != _puzzle_signature(generate_puzzle(314160))
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (7, 3)):
        puzzle = generate_puzzle(500 + train_count + test_count, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        validate_generated_puzzle(puzzle)
    for train_count, test_count in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(600 + (train_count or 0) + (test_count or 0), train_count, test_count)
        if train_count is not None:
            assert len(puzzle["train"]) == train_count
        if test_count is not None:
            assert len(puzzle["test"]) == test_count
        validate_generated_puzzle(puzzle)
    for arguments in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
        {"num_train": True}, {"num_test": 1.0},
    ):
        try:
            generate_puzzle(700, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {arguments}")
