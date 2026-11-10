import numpy as np


_SOURCE_INPUT_ROWS = (
    (
        "00006000",
        "00006000",
        "00006000",
        "00006000",
        "88886888",
        "00006000",
        "00006000",
        "00006000",
    ),
    (
        "0300",
        "2222",
        "0300",
        "0300",
    ),
    (
        "001000",
        "001000",
        "991999",
        "001000",
        "001000",
        "001000",
    ),
    (
        "000000005000",
        "000000005000",
        "000000005000",
        "000000005000",
        "000000005000",
        "000000005000",
        "333333335333",
        "000000005000",
        "000000005000",
        "000000005000",
        "000000005000",
        "000000005000",
    ),
)


def _rows_key(rows) -> tuple:
    return (
        len(rows),
        len(rows[0]),
        tuple(int(character) for row in rows for character in row),
    )


_SOURCE_INPUT_KEYS = frozenset(_rows_key(rows) for rows in _SOURCE_INPUT_ROWS)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        int(grid.shape[0]),
        int(grid.shape[1]),
        tuple(int(value) for value in grid.ravel()),
    )


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(
        (
            split,
            tuple(
                (
                    _grid_key(pair["input"]),
                    _grid_key(pair["output"]),
                )
                for pair in puzzle[split]
            ),
        )
        for split in ("train", "test")
    )


def _validated_count(name: str, value, minimum: int, maximum: int):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(
            f"{name} must be between {minimum} and {maximum}, inclusive"
        )
    return result


def _resolve_counts(
    rng: np.random.Generator,
    num_train,
    num_test,
) -> tuple[int, int]:
    train_count = _validated_count("num_train", num_train, 2, 10)
    test_count = _validated_count("num_test", num_test, 1, 4)

    if train_count is None and test_count is None:
        test_count = int(rng.integers(1, 5))
        train_count = int(rng.integers(test_count + 1, 11))
    elif train_count is None:
        minimum_train = max(2, test_count + 1)
        train_count = int(rng.integers(minimum_train, 11))
    elif test_count is None:
        maximum_test = min(4, train_count - 1)
        if maximum_test < 1:
            raise ValueError("no legal test count exists for num_train")
        test_count = int(rng.integers(1, maximum_test + 1))

    if train_count <= test_count:
        raise ValueError("num_train must be greater than num_test")

    return train_count, test_count


def _analyze_input(grid: np.ndarray) -> dict:
    """Find the unique conceptual crossing without using generator metadata."""
    height, width = grid.shape
    candidates = []

    for row in range(1, height - 1):
        for column in range(1, width - 1):
            outside_mask = np.ones(grid.shape, dtype=bool)
            outside_mask[row, :] = False
            outside_mask[:, column] = False

            background_values = np.unique(grid[outside_mask])
            horizontal_values = np.unique(np.delete(grid[row, :], column))
            vertical_values = np.unique(np.delete(grid[:, column], row))

            if (
                background_values.size != 1
                or horizontal_values.size != 1
                or vertical_values.size != 1
            ):
                continue

            background = int(background_values[0])
            horizontal = int(horizontal_values[0])
            vertical = int(vertical_values[0])
            center = int(grid[row, column])

            if len({background, horizontal, vertical}) != 3:
                continue
            if center not in (horizontal, vertical):
                continue

            top_line = (
                "horizontal" if center == horizontal else "vertical"
            )
            candidates.append(
                {
                    "row": row,
                    "column": column,
                    "background": background,
                    "horizontal": horizontal,
                    "vertical": vertical,
                    "top_line": top_line,
                }
            )

    _require(
        len(candidates) == 1,
        "input must contain exactly one valid interior line crossing",
    )
    return candidates[0]


def _construct_output(
    input_grid: np.ndarray,
    crossing: dict,
    highlight: int,
) -> np.ndarray:
    output = input_grid.copy()
    row = crossing["row"]
    column = crossing["column"]
    center_value = int(input_grid[row, column])

    output[row - 1 : row + 2, column - 1 : column + 2] = highlight
    output[row, column] = center_value
    return output


def _inspect_transformation(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> tuple[int, dict]:
    _require(
        input_grid.shape == output_grid.shape,
        "input and output dimensions must be equal",
    )

    crossing = _analyze_input(input_grid)
    row = crossing["row"]
    column = crossing["column"]

    frame_mask = np.zeros(input_grid.shape, dtype=bool)
    frame_mask[row - 1 : row + 2, column - 1 : column + 2] = True
    frame_mask[row, column] = False

    changed_mask = input_grid != output_grid
    _require(
        np.array_equal(changed_mask, frame_mask),
        "exactly the eight noncentral cells of the 3x3 frame must change",
    )

    highlighted_values = np.unique(output_grid[frame_mask])
    _require(
        highlighted_values.size == 1,
        "all eight frame cells must use one highlight color",
    )
    highlight = int(highlighted_values[0])

    _require(
        highlight not in {int(value) for value in np.unique(input_grid)},
        "highlight color must be newly introduced",
    )
    _require(
        int(output_grid[row, column]) == int(input_grid[row, column]),
        "crossing center must be preserved",
    )

    expected = _construct_output(input_grid, crossing, highlight)
    _require(
        np.array_equal(expected, output_grid),
        "output does not match the intersection-frame rule",
    )
    _require(
        np.unique(input_grid).size == 3,
        "input must contain exactly three colors",
    )
    _require(
        np.unique(output_grid).size == 4,
        "output must contain the three input colors plus the highlight",
    )

    return highlight, crossing


def _sample_colors(
    rng: np.random.Generator,
    highlight: int,
) -> tuple[int, int, int]:
    available = np.array(
        [color for color in range(10) if color != highlight],
        dtype=np.int8,
    )
    chosen = rng.choice(available, size=3, replace=False)
    return int(chosen[0]), int(chosen[1]), int(chosen[2])


def _dimension(rng, regime, small_only=False, large_only=False):
    if small_only:
        return int(rng.integers(4, 10))
    if large_only:
        return int(rng.integers(16, 31))
    weights = (0.42, 0.36, 0.22) if regime == "source_train" else (0.30, 0.37, 0.33)
    band = int(rng.choice(3, p=weights))
    return int(rng.integers(*((4, 11), (11, 21), (21, 31))[band]))


def _position(rng, length):
    mode = float(rng.random())
    if mode < 0.38:
        return 1 if int(rng.integers(2)) == 0 else length - 2
    if mode < 0.57:
        middle = (length - 1) // 2
        return middle + int(rng.integers(2)) if length % 2 == 0 else middle
    return int(rng.integers(1, length - 1))


def _make_pair(rng, highlight, palette, regime, small_only=False,
               large_only=False, challenge_train=False, challenge_test=False):
    if challenge_train:
        height = int(rng.integers(16, 31))
        width = int(rng.integers(16, 31))
        if width == height:
            width = 16 if height == 30 else height + 1
        row = 1 if int(rng.integers(2)) == 0 else height - 2
        column = _position(rng, width)
    elif challenge_test:
        height = int(rng.integers(4, 11))
        width = int(rng.integers(4, 11))
        row, column = _position(rng, height), _position(rng, width)
    else:
        height = _dimension(rng, regime, small_only, large_only)
        width = height if rng.random() < 0.22 else _dimension(
            rng, regime, small_only, large_only)
        row, column = _position(rng, height), _position(rng, width)
    background, horizontal, vertical = palette
    top_line = ("horizontal" if challenge_train else "vertical" if challenge_test
                else "horizontal" if int(rng.integers(2)) == 0 else "vertical")
    grid = np.full((height, width), background, dtype=np.int8)
    if top_line == "horizontal":
        grid[:, column] = vertical
        grid[row, :] = horizontal
    else:
        grid[row, :] = horizontal
        grid[:, column] = vertical
    output = _construct_output(grid, {"row": row, "column": column}, highlight)
    return {"input": grid, "output": output}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    highlight = int(rng.integers(10))
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # An occasional source-like scale shift; otherwise dimensions overlap freely.
    source_scale_shift = rng.random() < 0.16
    combined_example = not source_scale_shift and rng.random() < 0.20
    challenge_train_index = int(rng.integers(train_count)) if combined_example else -1
    challenge_test_index = int(rng.integers(test_count)) if combined_example else -1
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    train_palettes = []

    for split, count in (("train", train_count), ("test", test_count)):
        for index in range(count):
            regime = "source_train" if split == "train" else "target_test"
            if split == "train":
                if train_palettes and rng.random() < 0.48:
                    palette = train_palettes[int(rng.integers(len(train_palettes)))]
                else:
                    palette = _sample_colors(rng, highlight)
                train_palettes.append(palette)
            else:
                palette = (train_palettes[int(rng.integers(len(train_palettes)))]
                           if rng.random() < 0.47 else _sample_colors(rng, highlight))
            accepted = None
            for _ in range(80):
                pair = _make_pair(
                    rng, highlight, palette, regime,
                    small_only=source_scale_shift and split == "train",
                    large_only=source_scale_shift and split == "test",
                    challenge_train=split == "train" and index == challenge_train_index,
                    challenge_test=split == "test" and index == challenge_test_index,
                )
                key = _grid_key(pair["input"])
                if key not in used_inputs and key not in _SOURCE_INPUT_KEYS:
                    accepted = pair
                    break
            if accepted is None:
                # Bounded deterministic fallback with the same visible roles.
                for row in range(1, 29):
                    if accepted is not None:
                        break
                    for column in range(1, 29):
                        background, horizontal, vertical = palette
                        grid = np.full((30, 30), background, dtype=np.int8)
                        grid[row, :] = horizontal
                        grid[:, column] = vertical
                        key = _grid_key(grid)
                        if key not in used_inputs:
                            accepted = {
                                "input": grid,
                                "output": _construct_output(
                                    grid, {"row": row, "column": column}, highlight),
                            }
                            break
            _require(accepted is not None, "could not construct a unique pair")
            used_inputs.add(_grid_key(accepted["input"]))
            puzzle[split].append(accepted)

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list),
             "train and test must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4
             and train_count > test_count, "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    highlights = {"train": set(), "test": set()}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input and output")
            for name in ("input", "output"):
                grid = pair[name]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a 2D NumPy array")
                _require(np.issubdtype(grid.dtype, np.integer),
                         "grid must have integer dtype")
                _require(all(1 <= side <= 30 for side in grid.shape),
                         "grid size out of range")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9,
                         "grid colors out of range")
            input_key, output_key = _grid_key(pair["input"]), _grid_key(pair["output"])
            _require(input_key not in seen_inputs
                     and (input_key, output_key) not in seen_pairs,
                     "duplicate input or pair")
            _require(input_key not in _SOURCE_INPUT_KEYS, "source pair copied")
            seen_inputs.add(input_key)
            seen_pairs.add((input_key, output_key))
            highlight, crossing = _inspect_transformation(
                pair["input"], pair["output"])
            highlights[split].add(highlight)
            _require(1 <= crossing["row"] < pair["input"].shape[0] - 1
                     and 1 <= crossing["column"] < pair["input"].shape[1] - 1,
                     "frame not fully in bounds")
    _require(len(highlights["train"]) == 1,
             "training must teach one highlight color")
    _require(highlights["test"] == highlights["train"],
             "test highlight must match training")
    learned_highlight = next(iter(highlights["train"]))
    for pair in puzzle["test"]:
        crossing = _analyze_input(pair["input"])
        _require(np.array_equal(_construct_output(
            pair["input"], crossing, learned_highlight), pair["output"]),
            "test output not recoverable from training")


def self_test() -> None:
    for seed, train, test in (
        (12345, None, None), (54321, 7, 3),
        (20000, 2, 1), (20001, 10, 1), (20002, 10, 4),
        (30000, None, 1), (30001, None, 4),
        (30002, 2, None), (30003, 10, None),
    ):
        first = generate_puzzle(seed, train, test)
        second = generate_puzzle(seed, train, test)
        _require(_puzzle_key(first) == _puzzle_key(second),
                 "generation is not deterministic")
        validate_generated_puzzle(first)
        if train is not None:
            _require(len(first["train"]) == train, "train count not honored")
        if test is not None:
            _require(len(first["test"]) == test, "test count not honored")
    for train, test in (
        (1, None), (11, None), (None, 0), (None, 5),
        (2, 2), (3, 4), (4, 4), (2, 4),
    ):
        try:
            generate_puzzle(40000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")

    train_counts, test_counts, signatures, highlights = set(), set(), set(), set()
    shapes, layouts, role_assignments = set(), set(), set()
    quadrants, edges = set(), set()
    train_rectangles = test_small_squares = reused_test_palette = 0
    reused_train_palette = both_orders = same_orders = scale_shifts = 0
    train_larger_than_test = reassigned_test_palette = combined_cases = 0
    split_quadrants = {"train": set(), "test": set()}
    split_central = {"train": 0, "test": 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_puzzle_key(puzzle))
        train_records, test_records = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                highlight, crossing = _inspect_transformation(source, target)
                # Independently reconstruct the ring cell by cell.
                expected = source.copy()
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if (dr, dc) != (0, 0):
                            expected[crossing["row"] + dr,
                                     crossing["column"] + dc] = highlight
                _require(np.array_equal(expected, target),
                         "cellwise rule disagrees")
                height, width = source.shape
                palette = (crossing["background"], crossing["horizontal"],
                           crossing["vertical"])
                record = (height, width, crossing["row"], crossing["column"],
                          crossing["top_line"], palette)
                (train_records if split == "train" else test_records).append(record)
                shapes.add((height, width))
                layouts.add(record[:5])
                role_assignments.add((highlight,) + palette)
                quadrants.add((crossing["row"] < height / 2,
                               crossing["column"] < width / 2))
                split_quadrants[split].add((crossing["row"] < height / 2,
                                           crossing["column"] < width / 2))
                if (abs(crossing["row"] - (height - 1) / 2) <= 0.5
                        and abs(crossing["column"] - (width - 1) / 2) <= 0.5):
                    split_central[split] += 1
                if crossing["row"] == 1:
                    edges.add("top")
                if crossing["row"] == height - 2:
                    edges.add("bottom")
                if crossing["column"] == 1:
                    edges.add("left")
                if crossing["column"] == width - 2:
                    edges.add("right")
                if split == "train" and height != width:
                    train_rectangles += 1
                if split == "test" and height == width and height <= 10:
                    test_small_squares += 1
                if split == "train":
                    highlights.add(highlight)
        train_palettes = [r[5] for r in train_records]
        if len(set(train_palettes)) < len(train_palettes):
            reused_train_palette += 1
        if any(r[5] in train_palettes for r in test_records):
            reused_test_palette += 1
        if any(r[5] not in train_palettes for r in test_records):
            reassigned_test_palette += 1
        if len({r[4] for r in train_records}) == 2:
            both_orders += 1
        else:
            same_orders += 1
        if (min(r[0] for r in test_records) > max(r[0] for r in train_records)
                and min(r[1] for r in test_records) > max(r[1] for r in train_records)):
            scale_shifts += 1
        if any(t[0] > s[0] and t[1] > s[1]
               for t in train_records for s in test_records):
            train_larger_than_test += 1
        if (any(r[0] != r[1] and r[0] >= 16 and r[1] >= 16
                and r[4] == "horizontal"
                and (r[2] <= 2 or r[2] >= r[0] - 3
                     or r[3] <= 2 or r[3] >= r[1] - 3)
                for r in train_records)
                and any(r[4] == "vertical" and any(
                    r[0] < t[0] and r[1] < t[1] for t in train_records)
                    for r in test_records)):
            combined_cases += 1
        learned = _inspect_transformation(
            puzzle["train"][0]["input"], puzzle["train"][0]["output"])[0]
        for pair in puzzle["test"]:
            crossing = _analyze_input(pair["input"])
            predicted = _construct_output(pair["input"], crossing, learned)
            _require(np.array_equal(predicted, pair["output"]),
                     "training-only prediction disagrees")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "counts lack diversity")
    _require(len(signatures) >= 40 and len(highlights) >= 6,
             "puzzles lack diversity")
    _require(len(shapes) >= 45 and len(layouts) >= 100
             and len(role_assignments) >= 100,
             "structural or palette diversity too narrow")
    _require(len(quadrants) == 4
             and edges == {"top", "bottom", "left", "right"},
             "crossing positions lack coverage")
    _require(all(len(split_quadrants[s]) == 4 and split_central[s] >= 1
                 for s in ("train", "test")),
             "each split must cover quadrants and central crossings")
    _require(train_rectangles >= 20 and test_small_squares >= 2,
             "train/test shape overlap is inadequate")
    _require(reused_train_palette >= 5 and reused_test_palette >= 5,
             "palette reuse is inadequate")
    _require(reassigned_test_palette >= 5,
             "test palette reassignment is inadequate")
    _require(both_orders >= 5 and same_orders >= 1,
             "draw-order variation is inadequate")
    _require(scale_shifts >= 1 and train_larger_than_test >= 5,
             "train/test scale ordering is too rigid")
    _require(combined_cases >= 1, "combined geometry and draw-order case absent")
