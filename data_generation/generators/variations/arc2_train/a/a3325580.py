import numpy as np


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(x) for x in grid.shape),
        tuple(int(x) for x in grid.reshape(-1)),
    )


def _pair_key(pair: dict) -> tuple:
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _rows_key(rows: tuple[str, ...]) -> tuple:
    return (
        (len(rows), len(rows[0])),
        tuple(int(ch) for row in rows for ch in row),
    )


_SOURCE_ROWS = (
    (
        (
            "0000000000",
            "0000000000",
            "0000000000",
            "0080000000",
            "0080006000",
            "0080006000",
            "0000000000",
            "0000000000",
            "0000440000",
            "0000000000",
        ),
        ("8", "8", "8"),
    ),
    (
        (
            "0000000444",
            "0000000404",
            "0000000004",
            "0900060044",
            "0990060004",
            "9900666000",
            "0900006000",
            "0990000000",
            "0900000000",
            "0000000000",
        ),
        (
            "94",
            "94",
            "94",
            "94",
            "94",
            "94",
            "94",
            "94",
            "94",
        ),
    ),
    (
        (
            "0000000001",
            "7770022001",
            "0070002201",
            "0000002001",
            "0000000000",
            "0003000000",
            "0003300000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
        ("2", "2", "2", "2", "2"),
    ),
    (
        (
            "0000000000",
            "0000000000",
            "0000033000",
            "0000003000",
            "0200000000",
            "0220000000",
            "0000000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
        ("23", "23", "23"),
    ),
    (
        (
            "0000000000",
            "0000000888",
            "0040000008",
            "0040066008",
            "0044006000",
            "0040066000",
            "0000000000",
            "0330000000",
            "3300000000",
            "0000000000",
        ),
        ("468", "468", "468", "468", "468"),
    ),
    (
        (
            "0000000000",
            "0000000000",
            "0100000000",
            "0100000888",
            "0100000000",
            "0004400000",
            "0004000000",
            "0000000000",
            "0000000000",
            "0000000000",
        ),
        ("148", "148", "148"),
    ),
    (
        (
            "0000000010",
            "0050000111",
            "0550000000",
            "0050000000",
            "0000000000",
            "0000099000",
            "8000000000",
            "8002200000",
            "8002000000",
            "0002000000",
        ),
        ("521", "521", "521", "521"),
    ),
)

_SOURCE_PAIR_KEYS = frozenset(
    (_rows_key(input_rows), _rows_key(output_rows))
    for input_rows, output_rows in _SOURCE_ROWS
)


def _normalize_optional_count(
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


def _resolve_counts(
    seed: int,
    rng: np.random.Generator,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    train_count = _normalize_optional_count(
        num_train, "num_train", 2, 10
    )
    test_count = _normalize_optional_count(
        num_test, "num_test", 1, 4
    )

    if train_count is not None and test_count is not None:
        if train_count <= test_count:
            raise ValueError("num_train must be greater than num_test")
        return train_count, test_count

    if train_count is None and test_count is None:
        # This schedule guarantees genuine count diversity across nearby seeds.
        test_count = 1 + seed % 4
        lowest_train = max(2, test_count + 1)
        span = 11 - lowest_train
        train_count = lowest_train + (seed // 4) % span
        return train_count, test_count

    if train_count is None:
        lowest_train = max(2, test_count + 1)
        train_count = int(rng.integers(lowest_train, 11))
        return train_count, test_count

    highest_test = min(4, train_count - 1)
    test_count = int(rng.integers(1, highest_test + 1))
    return train_count, test_count


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple:
    top = min(row for row, _ in cells)
    left = min(col for _, col in cells)
    return tuple(sorted((row - top, col - left) for row, col in cells))


def _make_shape(rng: np.random.Generator, area: int, style: str | None = None) -> tuple:
    _require(1 <= area <= 9, "Invalid object area")
    if area == 1:
        return ((0, 0),)
    if style is None:
        styles = ["vertical", "horizontal", "block", "elbow", "random"]
        if area >= 4:
            styles += ["tee", "zigzag"]
        style = styles[int(rng.integers(len(styles)))]
    if style == "vertical":
        cells = {(row, 0) for row in range(area)}
    elif style == "horizontal":
        cells = {(0, col) for col in range(area)}
    elif style == "block":
        width = int(np.ceil(np.sqrt(area)))
        cells = {(index // width, index % width) for index in range(area)}
    elif style == "elbow":
        stem = max(2, (area + 1) // 2)
        cells = {(row, 0) for row in range(stem)}
        cells.update((stem - 1, col) for col in range(1, area - stem + 1))
    elif style == "tee" and area >= 4:
        cells = {(0, 0), (0, 1), (0, 2), (1, 1)}
        cells.update((row, 1) for row in range(2, area - 2))
    elif style == "zigzag" and area >= 4:
        cells = {(0, 0), (0, 1), (1, 1), (1, 2)}
        for index in range(4, area):
            cells.add((1 + (index - 3) // 2, 2 + (index - 3) % 2))
    elif style == "random":
        side = min(5, max(3, int(np.ceil(np.sqrt(area))) + 1))
        cells = {(int(rng.integers(side)), int(rng.integers(side)))}
        while len(cells) < area:
            frontier = set()
            for row, col in cells:
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    new = (row + dr, col + dc)
                    if 0 <= new[0] < side and 0 <= new[1] < side and new not in cells:
                        frontier.add(new)
            options = sorted(frontier)
            cells.add(options[int(rng.integers(len(options)))])
    else:
        raise ValueError(f"Unknown shape style: {style}")
    cells = set(_normalize_cells(cells))
    if int(rng.integers(2)):
        cells = {(col, -row) for row, col in cells}
    if int(rng.integers(2)):
        cells = {(row, -col) for row, col in cells}
    if int(rng.integers(2)):
        cells = {(-row, col) for row, col in cells}
    return _normalize_cells(cells)


def _shape_dimensions(shape: tuple) -> tuple[int, int]:
    return (1 + max(row for row, _ in shape), 1 + max(col for _, col in shape))


def _canvas_size(rng: np.random.Generator, shapes: list[tuple], mode: str) -> tuple[int, int]:
    if mode == "source_compact":
        return (10, 10)
    area = sum(len(shape) for shape in shapes)
    side = max(8, int(np.ceil(np.sqrt(area * float(rng.uniform(2.8, 4.4))))))
    if mode == "compact":
        height = side + int(rng.integers(-1, 2))
        width = side + int(rng.integers(-1, 2))
    elif mode == "tall":
        height = side + int(rng.integers(4, 10))
        width = max(7, side - int(rng.integers(2, 5)))
    elif mode == "broad":
        height = max(7, side - int(rng.integers(2, 5)))
        width = side + int(rng.integers(4, 10))
    else:
        height = side + int(rng.integers(2, 7))
        width = side + int(rng.integers(2, 7))
    max_height = max(_shape_dimensions(shape)[0] for shape in shapes)
    max_width = max(_shape_dimensions(shape)[1] for shape in shapes)
    return (min(30, max(height, max_height + 2)), min(30, max(width, max_width + 2)))


def _layout_input(
    rng: np.random.Generator,
    shapes: list[tuple],
    colors: list[int],
    background: int,
    mode: str,
) -> np.ndarray:
    base_height, base_width = _canvas_size(rng, shapes, mode)
    order = sorted(range(len(shapes)), key=lambda i: -len(shapes[i]))
    winner_area = max(len(shape) for shape in shapes)
    for expansion in (0, 2, 5, 9, 15):
        height = min(30, base_height + expansion)
        width = min(30, base_width + expansion)
        for _ in range(35):
            grid = np.full((height, width), background, dtype=np.int8)
            winner_lefts = set()
            success = True
            for index in order:
                shape = shapes[index]
                shape_height, shape_width = _shape_dimensions(shape)
                if shape_height > height or shape_width > width:
                    success = False
                    break
                placed = False
                for _ in range(100):
                    top = int(rng.integers(height - shape_height + 1))
                    left = int(rng.integers(width - shape_width + 1))
                    if len(shape) == winner_area and left in winner_lefts:
                        continue
                    cells = [(top + r, left + c) for r, c in shape]
                    if any(int(grid[r, c]) != background for r, c in cells):
                        continue
                    if any(
                        0 <= r + dr < height and 0 <= c + dc < width
                        and int(grid[r + dr, c + dc]) not in (background, colors[index])
                        for r, c in cells
                        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                    ):
                        continue
                    for row, col in cells:
                        grid[row, col] = colors[index]
                    if len(shape) == winner_area:
                        winner_lefts.add(left)
                    placed = True
                    break
                if not placed:
                    success = False
                    break
            if success:
                return grid
    raise RuntimeError("Could not place disconnected objects")


def _analyze_input(grid: np.ndarray) -> dict:
    values, counts = np.unique(grid, return_counts=True)
    mode_indices = np.flatnonzero(counts == np.max(counts))
    _require(len(mode_indices) == 1, "Ambiguous background")
    background = int(values[int(mode_indices[0])])
    _require(int(np.max(counts)) > grid.size // 2, "Background must be locally obvious")
    height, width = grid.shape
    visited = np.zeros(grid.shape, dtype=bool)
    objects = []
    for start_row in range(height):
        for start_col in range(width):
            if visited[start_row, start_col] or int(grid[start_row, start_col]) == background:
                continue
            color = int(grid[start_row, start_col])
            stack = [(start_row, start_col)]
            visited[start_row, start_col] = True
            cells = []
            while stack:
                row, col = stack.pop()
                cells.append((row, col))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            rows = [row for row, _ in cells]
            cols = [col for _, col in cells]
            objects.append({
                "color": color, "area": len(cells), "min_row": min(rows),
                "max_row": max(rows), "min_col": min(cols), "max_col": max(cols),
                "cells": tuple(cells),
            })
    _require(len(objects) >= 2, "Scene needs at least two objects")
    _require(len({obj["color"] for obj in objects}) == len(objects), "Object colors must be distinct")
    for row in range(height):
        for col in range(width):
            color = int(grid[row, col])
            if color == background:
                continue
            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = row + dr, col + dc
                if nr < height and nc < width:
                    other = int(grid[nr, nc])
                    _require(other in (background, color), "Different objects touch orthogonally")
    maximum_area = max(obj["area"] for obj in objects)
    winners = sorted((obj for obj in objects if obj["area"] == maximum_area), key=lambda obj: obj["min_col"])
    _require(len({obj["min_col"] for obj in winners}) == len(winners), "Winner order is ambiguous")
    expected_output = np.tile(np.asarray([obj["color"] for obj in winners], dtype=np.int8), (maximum_area, 1))
    return {"background": background, "objects": objects, "maximum_area": maximum_area,
            "winners": winners, "expected_output": expected_output}


def _solve_input(grid: np.ndarray) -> np.ndarray:
    return _analyze_input(grid)["expected_output"].copy()


def _validate_grid(grid: np.ndarray, name: str) -> None:
    _require(isinstance(grid, np.ndarray), f"{name} must be a NumPy array")
    _require(grid.ndim == 2, f"{name} must be two-dimensional")
    _require(all(1 <= dimension <= 30 for dimension in grid.shape), f"{name} dimensions invalid")
    _require(grid.dtype.kind in {"i", "u"}, f"{name} must use an integer dtype")
    _require(0 <= int(np.min(grid)) and int(np.max(grid)) <= 9, f"{name} colors invalid")


def _validate_pair(pair: dict, regime: str) -> dict:
    _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "Invalid pair schema")
    input_grid, output_grid = pair["input"], pair["output"]
    _validate_grid(input_grid, "input")
    _validate_grid(output_grid, "output")
    _require(input_grid.shape != output_grid.shape or not np.array_equal(input_grid, output_grid), "Unchanged pair")
    analysis = _analyze_input(input_grid)
    _require(np.array_equal(output_grid, analysis["expected_output"]), "Incorrect counting-and-column output")
    count = len(analysis["objects"])
    area = analysis["maximum_area"]
    winners = len(analysis["winners"])
    if regime == "source_train":
        _require(2 <= count <= 4 and 3 <= area <= 9 and 1 <= winners <= 3, "Invalid training regime")
    elif regime == "target_test":
        _require(2 <= count <= 7 and 3 <= area <= 9 and 1 <= winners <= 3, "Invalid test regime")
    else:
        raise AssertionError("Unknown regime")
    return analysis


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "Invalid puzzle schema")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "Splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "Invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split, regime in (("train", "source_train"), ("test", "target_test")):
        for pair in puzzle[split]:
            analysis = _validate_pair(pair, regime)
            analyses[split].append(analysis)
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, "Duplicate pair")
            _require(pair_key not in _SOURCE_PAIR_KEYS, "Source pair copy")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
    train = analyses["train"]
    _require(any(len(a["winners"]) == 1 and len(a["objects"]) > 1 for a in train), "Training must show rejection")
    _require(any(len(a["winners"]) == 3 and len(a["objects"]) == 4 for a in train), "Training must show a three-way tie and rejection")
    all_backgrounds = {a["background"] for a in train + analyses["test"]}
    if len(all_backgrounds) > 1:
        _require(len({a["background"] for a in train}) > 1, "Training must show changing backgrounds")


def _sample_areas(rng: np.random.Generator, regime: str, index: int) -> list[int]:
    if regime == "source_train" and index == 0:
        maximum = int(rng.integers(5, 8))
        count = int(rng.integers(2, 5))
        areas = [maximum, maximum - 1]
        areas += [int(rng.integers(1, maximum)) for _ in range(count - 2)]
    elif regime == "source_train" and index == 1:
        maximum = int(rng.integers(3, 10))
        areas = [maximum] * 3 + [int(rng.integers(1, maximum))]
    else:
        if regime == "source_train":
            count = int(rng.integers(2, 5))
        else:
            count = int(rng.integers(2, 8))
            if int(rng.integers(4)) == 0:
                count = int(rng.integers(5, 8))
        maximum = int(rng.integers(3, 10))
        if regime == "target_test" and count >= 5 and int(rng.integers(7)) == 0:
            maximum, count, winner_count = 4, 5, 3
        else:
            winner_count = int(rng.integers(1, min(3, count) + 1))
        if int(rng.integers(5)) == 0 and count <= 3:
            winner_count = count
        areas = [maximum] * winner_count
        areas += [int(rng.integers(1, maximum)) for _ in range(count - winner_count)]
    rng.shuffle(areas)
    return areas


def _generate_pair(
    rng: np.random.Generator,
    regime: str,
    pair_index: int,
    background: int,
    forced_color: int | None,
) -> dict:
    areas = _sample_areas(rng, regime, pair_index)
    available = [color for color in range(10) if color != background and color != forced_color]
    rng.shuffle(available)
    colors = available[:len(areas)]
    if forced_color is not None and forced_color != background:
        colors[-1] = forced_color
    rng.shuffle(colors)
    shapes = []
    tall_distractor_used = False
    for index, area in enumerate(areas):
        style = None
        if regime == "source_train" and pair_index == 0:
            if area == max(areas):
                style = "block"
            elif area == max(areas) - 1 and not tall_distractor_used:
                style = "vertical"
                tall_distractor_used = True
        shapes.append(_make_shape(rng, area, style))
    mode = ("compact", "tall", "broad", "roomy")[int(rng.integers(4))]
    if regime == "target_test" and len(areas) == 5 and max(areas) == 4 and areas.count(4) == 3:
        mode = "source_compact"
    input_grid = _layout_input(rng, shapes, colors, background, mode)
    return {"input": input_grid, "output": _solve_input(input_grid)}


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, rng, num_train, num_test)
    total = train_count + test_count
    first_background = int(rng.integers(10))
    mixed = bool(int(rng.integers(2)))
    backgrounds = [first_background]
    for _ in range(1, total):
        if mixed:
            choices = [color for color in range(10) if color != backgrounds[-1]]
            backgrounds.append(choices[int(rng.integers(len(choices)))])
        else:
            backgrounds.append(first_background)
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=total, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen_inputs, seen_pairs = set(), set()
    for position, pair_seed in enumerate(pair_seeds):
        regime = "source_train" if position < train_count else "target_test"
        index = position if regime == "source_train" else position - train_count
        forced_color = None
        if mixed:
            forced_color = backgrounds[1] if position == 0 else backgrounds[position - 1]
        pair_rng = np.random.default_rng(int(pair_seed))
        for _ in range(20):
            pair = _generate_pair(pair_rng, regime, index, backgrounds[position], forced_color)
            input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
            if input_key not in seen_inputs and pair_key not in seen_pairs and pair_key not in _SOURCE_PAIR_KEYS:
                break
        else:
            raise RuntimeError("Could not generate unique pair")
        seen_inputs.add(input_key)
        seen_pairs.add(pair_key)
        puzzle["train" if regime == "source_train" else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _predict_by_color_counts(input_grid: np.ndarray) -> np.ndarray:
    values, counts = np.unique(input_grid, return_counts=True)
    background = int(values[int(np.argmax(counts))])
    candidates = []
    for color in values:
        color = int(color)
        if color == background:
            continue
        rows, cols = np.where(input_grid == color)
        candidates.append((len(rows), int(np.min(cols)), color))
    maximum = max(area for area, _, _ in candidates)
    winners = sorted((left, color) for area, left, color in candidates if area == maximum)
    return np.repeat(np.asarray([[color for _, color in winners]], dtype=np.int8), maximum, axis=0)


def _independent_prediction(train_pairs: list[dict], test_input: np.ndarray) -> np.ndarray:
    _require(len(train_pairs) >= 2, "Prediction needs training evidence")
    for pair in train_pairs:
        _require(
            np.array_equal(_predict_by_color_counts(pair["input"]), pair["output"]),
            "Training contradicts the area-counting rule",
        )
    return _predict_by_color_counts(test_input)


def _four_cell_shape_class(obj: dict) -> str:
    cells = set(obj["cells"])
    height = obj["max_row"] - obj["min_row"] + 1
    width = obj["max_col"] - obj["min_col"] + 1
    if height == 1 or width == 1:
        return "line"
    if height == width == 2:
        return "square"
    degrees = [
        sum((row + dr, col + dc) in cells for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)))
        for row, col in cells
    ]
    if 3 in degrees:
        return "tee"
    short_counts = [
        sum((row - obj["min_row"] if height == 2 else col - obj["min_col"]) == index for row, col in cells)
        for index in range(min(height, width))
    ]
    return "zigzag" if sorted(short_counts) == [2, 2] else "bent"


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple(_pair_key(pair) for pair in puzzle[split]) for split in ("train", "test"))


def _expect_value_error(function) -> None:
    try:
        function()
    except ValueError:
        return
    raise AssertionError("Expected ValueError")


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_copy = [
            {"input": pair["input"].copy(), "output": pair["output"].copy()}
            for pair in puzzle["train"]
        ]
        for pair in puzzle["test"]:
            prediction = _independent_prediction(train_copy, pair["input"].copy())
            _require(np.array_equal(prediction, pair["output"]), "Independent prediction failed")
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "Training counts lack variety")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "Test counts lack variety")
    _require(len({_puzzle_key(p) for p in puzzles}) == len(puzzles), "Seeds duplicate puzzles")
    _require(_puzzle_key(generate_puzzle(123)) == _puzzle_key(generate_puzzle(123)), "Nondeterminism")
    for train, test in ((2, 1), (10, 1), (10, 4), (3, 2), (5, 3)):
        puzzle = generate_puzzle(800 + train * 10 + test, train, test)
        _require((len(puzzle["train"]), len(puzzle["test"])) == (train, test), "Explicit counts ignored")
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(900 + (train or 0) + (test or 0), train, test))
    for train, test in ((1, 1), (11, 1), (2, 0), (2, 5), (2, 2), (3, 4)):
        _expect_value_error(lambda t=train, u=test: generate_puzzle(5, t, u))
    analyses = [_analyze_input(pair["input"]) for puzzle in puzzles for pair in puzzle["test"]]
    _require(len({len(a["objects"]) for a in analyses}) >= 5, "Test object counts lack variety")
    _require(len({a["maximum_area"] for a in analyses}) >= 5, "Test areas lack variety")
    _require({len(a["winners"]) for a in analyses} == {1, 2, 3}, "Test tie counts lack variety")
    _require(any(len(a["objects"]) == len(a["winners"]) for a in analyses), "No loser-free tests")
    _require(any(len(a["objects"]) - len(a["winners"]) >= 3 for a in analyses), "No multi-loser tests")
    _require(any(len({obj["area"] for obj in a["objects"] if obj["area"] < a["maximum_area"]}) < len(a["objects"]) - len(a["winners"]) for a in analyses if len(a["objects"]) > len(a["winners"])), "No repeated loser area")
    _require(any(a["maximum_area"] == 4 and len(a["objects"]) == 5 and len(a["winners"]) == 3 for a in analyses), "No source-like test")
    all_analyses = [_analyze_input(pair["input"]) for puzzle in puzzles for pair in puzzle["train"] + puzzle["test"]]
    _require(any(len({a["background"] for a in [_analyze_input(pair["input"]) for pair in puzzle["train"] + puzzle["test"]]}) > 1 for puzzle in puzzles), "No mixed-background puzzle")
    _require(any(len({a["background"] for a in [_analyze_input(pair["input"]) for pair in puzzle["train"] + puzzle["test"]]}) == 1 for puzzle in puzzles), "No fixed-background puzzle")
    _require(any(left["max_col"] >= right["min_col"] for a in all_analyses for left, right in zip(sorted(a["objects"], key=lambda o: o["min_col"]), sorted(a["objects"], key=lambda o: o["min_col"])[1:])), "No overlapping object spans")
    _require(any(left["max_col"] + 1 == right["min_col"] for a in all_analyses for left, right in zip(sorted(a["objects"], key=lambda o: o["min_col"]), sorted(a["objects"], key=lambda o: o["min_col"])[1:])), "No adjacent object spans")
    _require(np.array_equal(_solve_input(np.asarray([[int(ch) for ch in row] for row in _SOURCE_ROWS[4][0]], dtype=np.int8)), np.asarray([[4, 6, 8]] * 5, dtype=np.int8)), "Source train 5 regression")
    tests = [(puzzle, pair, _analyze_input(pair["input"])) for puzzle in puzzles for pair in puzzle["test"]]
    _require(any(pair["input"].shape == (10, 10) for _, pair, _ in tests), "No source-sized compact tests")
    _require(any(pair["input"].shape[0] >= pair["input"].shape[1] + 4 for _, pair, _ in tests), "No tall tests")
    _require(any(pair["input"].shape[1] >= pair["input"].shape[0] + 4 for _, pair, _ in tests), "No broad tests")
    four_shapes = {_four_cell_shape_class(obj) for _, _, a in tests for obj in a["winners"] if obj["area"] == 4}
    _require(four_shapes == {"line", "square", "tee", "zigzag", "bent"}, "Four-cell winner shapes lack variety")
    _require(any(len({obj["min_row"] for obj in a["winners"]}) < len(a["winners"]) for _, _, a in tests if len(a["winners"]) >= 2), "No shared winner top row")
    _require(any(max(obj["min_row"] for obj in a["winners"]) - min(obj["min_row"] for obj in a["winners"]) >= 4 for _, _, a in tests if len(a["winners"]) >= 2), "No vertically separated winners")
    _require(any(len({_normalize_cells(set(obj["cells"])) for obj in a["winners"]}) < len(a["winners"]) for _, _, a in tests if len(a["winners"]) >= 2), "No matching winner silhouettes")
    _require(any(len({_normalize_cells(set(obj["cells"])) for obj in a["winners"]}) > 1 for _, _, a in tests if len(a["winners"]) >= 2), "No contrasting winner silhouettes")
    _require(any(any(loser["area"] < winner["area"] and loser["max_row"] - loser["min_row"] > winner["max_row"] - winner["min_row"] for loser in a["objects"] for winner in a["winners"]) for _, _, a in tests), "No taller smaller-area distractor")
    _require(any(len({(len(_analyze_input(pair["input"])["objects"]), pair["output"].shape) for pair in puzzle["test"]}) > 1 for puzzle in puzzles if len(puzzle["test"]) >= 2), "Tests within a puzzle never change composition")
