import itertools

import numpy as np


_SOURCE_INPUTS = (
    ((5, 5, 9, 9), (9, 5, 5, 5), (5, 7, 5, 7)),
    ((6, 2, 4, 2), (2, 2, 6, 6), (6, 4, 2, 4)),
    ((3, 3, 5, 5), (5, 8, 5, 8), (8, 8, 5, 8)),
    ((8, 5, 7, 8), (7, 7, 8, 8), (5, 5, 8, 5)),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _count(value: int | None, name: str, minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return result


def _resolve_counts(
    rng: np.random.Generator, seed: int, num_train: int | None, num_test: int | None
) -> tuple[int, int]:
    train_count = _count(num_train, "num_train", 2, 10)
    test_count = _count(num_test, "num_test", 1, 4)
    if train_count is not None and test_count is not None and train_count <= test_count:
        raise ValueError("num_train must exceed num_test")
    if train_count is None and test_count is None:
        test_count = 1 + seed % 4
        train_count = test_count + 1 + (seed // 4) % (10 - test_count)
    elif train_count is None:
        train_count = int(rng.integers(test_count + 1, 11))
    elif test_count is None:
        test_count = int(rng.integers(1, min(4, train_count - 1) + 1))
    return train_count, test_count


def _transform(grid: np.ndarray) -> np.ndarray:
    upper = np.concatenate((grid[::-1, ::-1], grid[::-1, :]), axis=1)
    lower = np.concatenate((grid[:, ::-1], grid), axis=1)
    return np.concatenate((upper, lower), axis=0).astype(np.int8, copy=False)


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _layout_key(grid: np.ndarray) -> tuple:
    labels: dict[int, int] = {}
    normalized = []
    for raw in grid.flat:
        color = int(raw)
        if color not in labels:
            labels[color] = len(labels)
        normalized.append(labels[color])
    return (grid.shape, tuple(normalized))


def _fully_asymmetric(grid: np.ndarray) -> bool:
    return not (
        np.array_equal(grid, grid[::-1, :])
        or np.array_equal(grid, grid[:, ::-1])
        or np.array_equal(grid, grid[::-1, ::-1])
    )


def _shape(rng: np.random.Generator, strength: int, boundary: bool = False) -> tuple[int, int]:
    if boundary:
        branch = int(rng.integers(0, 5))
        if branch == 0:
            return (1, 1)
        if branch == 1:
            return (1, int(rng.integers(2, 16)))
        if branch == 2:
            return (int(rng.integers(2, 16)), 1)
        return (2, 2)
    branch = int(rng.integers(0, 5 if strength else 3))
    if branch == 0:
        return (int(rng.integers(2, 7)), int(rng.integers(2, 8)))
    if branch == 1:
        return (int(rng.integers(3, 11)), int(rng.integers(3, 11)))
    if branch == 2:
        return (int(rng.integers(2, 6)), int(rng.integers(8, 16)))
    if branch == 3:
        return (int(rng.integers(8, 16)), int(rng.integers(2, 6)))
    return (int(rng.integers(8, 16)), int(rng.integers(8, 16)))


def _palette_size(rng: np.random.Generator, area: int) -> int:
    sizes = np.arange(1, 11)
    weights = np.array((0.035, 0.20, 0.29, 0.15, 0.09, 0.065, 0.055, 0.045, 0.04, 0.03))
    return min(area, int(rng.choice(sizes, p=weights / weights.sum())))


def _palette(rng: np.random.Generator, pool: tuple[int, ...], size: int) -> tuple[int, ...]:
    amount = min(size, len(pool))
    return tuple(int(value) for value in rng.choice(pool, size=amount, replace=False))


def _palettes(
    rng: np.random.Generator, mode: int, shapes: list[tuple[int, int]], train_count: int
) -> list[tuple[int, ...]]:
    total = len(shapes)
    if mode == 0:
        pool = tuple(int(value) for value in rng.choice(10, size=6, replace=False))
        triples = list(itertools.combinations(pool, 3))
        rng.shuffle(triples)
        training = triples[:train_count]
        familiar = tuple(sorted(set(itertools.chain.from_iterable(training))))
        remaining = [triple for triple in triples[train_count:] if set(triple) <= set(familiar)]
        if len(remaining) < total - train_count:
            remaining = list(itertools.combinations(familiar, 3))
        return [tuple(item) for item in training + remaining[: total - train_count]]
    pool = tuple(range(10))
    if mode == 2:
        pool = tuple(int(value) for value in rng.choice(10, size=8, replace=False))
        novel = tuple(value for value in range(10) if value not in pool)
    palettes = []
    for index, (height, width) in enumerate(shapes):
        if index < 2:
            size = min(height * width, 10 if mode == 3 and index == 1 else int(rng.integers(2, 6)))
        else:
            size = _palette_size(rng, height * width)
        palettes.append(_palette(rng, pool, size))
    if mode == 1:
        palettes[train_count] = palettes[int(rng.integers(0, train_count))]
    elif mode == 2:
        size = max(2, min(shapes[train_count][0] * shapes[train_count][1], len(palettes[train_count])))
        familiar = _palette(rng, pool, size - 1)
        palettes[train_count] = familiar + (int(rng.choice(novel)),)
    return palettes


def _candidate(
    rng: np.random.Generator,
    shape: tuple[int, int],
    palette: tuple[int, ...],
    style: int,
    symmetry: int,
) -> np.ndarray:
    height, width = shape
    count = len(palette)
    rows, columns = np.indices(shape)
    area = height * width
    if count == 1:
        indices = np.zeros(shape, dtype=np.int64)
    elif style == 0:
        weights = rng.dirichlet(rng.uniform(0.7, 2.5, size=count))
        indices = rng.choice(count, size=shape, p=weights)
    elif style == 1:
        seeds = rng.choice(area, size=min(count, area), replace=False)
        seed_rows, seed_columns = seeds // width, seeds % width
        distances = abs(rows[None] - seed_rows[:, None, None]) + abs(
            columns[None] - seed_columns[:, None, None]
        )
        indices = np.argmin(distances + rng.random(distances.shape) * 0.3, axis=0)
    elif style == 2:
        indices = np.zeros(shape, dtype=np.int64)
        amount = int(rng.integers(1, max(2, area // 3 + 1)))
        cells = rng.choice(area, size=min(amount, area), replace=False)
        indices.flat[cells] = rng.integers(1, count, size=len(cells))
    elif style == 3:
        axis = rows if rng.integers(0, 2) else columns
        band = int(rng.integers(1, 4))
        indices = (axis // band + int(rng.integers(0, count))) % count
    elif style == 4:
        if rng.random() < 0.55:
            indices = (rows + columns) % 2
        else:
            row_band = int(rng.integers(1, 4))
            column_band = int(rng.integers(1, 4))
            indices = (rows // row_band + columns // column_band) % count
    elif style == 5:
        base = rng.integers(0, count, size=width)
        indices = np.broadcast_to(base, shape).copy()
    elif style == 6:
        indices = np.zeros(shape, dtype=np.int64)
        for color in range(1, count):
            top = int(rng.integers(0, height))
            left = int(rng.integers(0, width))
            bottom = int(rng.integers(top + 1, height + 1))
            right = int(rng.integers(left + 1, width + 1))
            indices[top:bottom, left:right] = color
    else:
        indices = np.zeros(shape, dtype=np.int64)
        if height >= 3 and width >= 3:
            top = int(rng.integers(1, max(2, height // 3 + 1)))
            left = int(rng.integers(1, max(2, width // 3 + 1)))
            bottom = int(rng.integers(max(top + 1, 2 * height // 3), height))
            right = int(rng.integers(max(left + 1, 2 * width // 3), width))
            indices[top:bottom, left:right] = rng.integers(1, count, size=(bottom - top, right - left))
        else:
            indices.flat[int(rng.integers(0, area))] = int(rng.integers(1, count))
    if style in (3, 4) and count > 1 and rng.random() < 0.25:
        changes = rng.random(shape) < rng.uniform(0.05, 0.18)
        indices[changes] = rng.integers(0, count, size=int(np.count_nonzero(changes)))
    if style in (0, 1) and symmetry == 0 and area >= count:
        cells = rng.choice(area, size=count, replace=False)
        indices.flat[cells] = rng.permutation(count)
    if symmetry in (1, 4):
        for column in range(width // 2):
            indices[:, width - 1 - column] = indices[:, column]
    if symmetry in (2, 4):
        for row in range(height // 2):
            indices[height - 1 - row, :] = indices[row, :]
    if symmetry == 3:
        for row in range(height):
            for column in range(width):
                partner = (height - 1 - row, width - 1 - column)
                if (row, column) < partner:
                    indices[partner] = indices[row, column]
    return np.asarray(palette, dtype=np.int8)[indices]


def _make_input(
    rng: np.random.Generator,
    shape: tuple[int, int],
    palette: tuple[int, ...],
    style: int,
    symmetry: int,
    diagnostic: bool,
    require_all_colors: bool,
    seen_inputs: set[tuple],
    seen_layouts: set[tuple],
) -> np.ndarray:
    for attempt in range(160):
        current_shape = shape if attempt < 64 else ((3, 4) if require_all_colors else _shape(rng, 1))
        current_palette = palette
        if len(current_palette) > current_shape[0] * current_shape[1]:
            current_palette = current_palette[: current_shape[0] * current_shape[1]]
        current_style = style if attempt < 32 else (0 if diagnostic or require_all_colors else style)
        candidate = _candidate(rng, current_shape, current_palette, current_style, symmetry)
        key = _grid_key(candidate)
        layout = _layout_key(candidate)
        if diagnostic and (
            candidate.shape[0] < 2 or candidate.shape[1] < 2
            or candidate.size < 6 or not _fully_asymmetric(candidate)
        ):
            continue
        if require_all_colors and len(np.unique(candidate)) != len(current_palette):
            continue
        if key in seen_inputs or layout in seen_layouts:
            continue
        if tuple(tuple(int(value) for value in row) for row in candidate) in _SOURCE_INPUTS:
            continue
        return candidate
    raise RuntimeError("could not construct a distinct motif")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, seed, num_train, num_test)
    total = train_count + test_count
    strength = seed % 3
    mode = seed % 4
    common_shape = None
    if seed % 5 == 0:
        common_shape = ((3, 4), (4, 3), (5, 5), (2, 6), (4, 7))[int(rng.integers(0, 5))]
    shapes = []
    for index in range(total):
        if common_shape is not None:
            shape = common_shape
        elif index < 2:
            shape = _shape(rng, 0)
            while shape[0] < 2 or shape[1] < 2 or shape[0] * shape[1] < 6 or (index == 1 and shape == shapes[0]):
                shape = _shape(rng, 0)
        else:
            use_boundary = strength > 0 and rng.random() < (0.19 if index < train_count else 0.32)
            shape = _shape(rng, strength, use_boundary)
            if index == train_count and strength == 2 and rng.random() < 0.5:
                shape = _shape(rng, 2, True)
        shapes.append(shape)
    if mode == 1 and common_shape is None and shapes[train_count][0] * shapes[train_count][1] < 10:
        shapes[train_count] = _shape(rng, 0)
    if mode == 2 and shapes[train_count][0] * shapes[train_count][1] < 10:
        shapes[train_count] = (3, 4)
    palettes = _palettes(rng, mode, shapes, train_count)
    if mode == 1 and shapes[train_count][0] * shapes[train_count][1] < len(palettes[train_count]):
        shapes[train_count] = (3, 4)
    if mode == 0:
        for index, shape in enumerate(shapes):
            if min(shape) < 3:
                shapes[index] = (3, 4)
    pairs = {"train": [], "test": []}
    seen_inputs: set[tuple] = set()
    seen_layouts: set[tuple] = set()
    for index in range(total):
        diagnostic = index < 2
        split = "train" if index < train_count else "test"
        style = (0, 1)[index] if diagnostic else int((index + seed // 3) % 8)
        symmetry = 0 if diagnostic else int(rng.choice(5, p=(0.42, 0.17, 0.17, 0.12, 0.12)))
        if mode in (1, 2) and index == train_count:
            style, symmetry = 0, 0
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        input_grid = _make_input(
            pair_rng, shapes[index], palettes[index], style, symmetry, diagnostic, mode == 0,
            seen_inputs, seen_layouts
        )
        seen_inputs.add(_grid_key(input_grid))
        seen_layouts.add(_layout_key(input_grid))
        pairs[split].append({"input": input_grid, "output": _transform(input_grid)})
    validate_generated_puzzle(pairs)
    return pairs


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "invalid counts")
    seen_inputs: set[tuple] = set()
    seen_pairs: set[tuple] = set()
    seen_layouts: set[tuple] = set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair")
            source, target = pair["input"], pair["output"]
            for grid in (source, target):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "invalid grid type")
                _require(np.issubdtype(grid.dtype, np.integer), "noninteger grid")
                _require(all(1 <= side <= 30 for side in grid.shape), "invalid grid dimensions")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "invalid color")
            height, width = source.shape
            _require(target.shape == (2 * height, 2 * width), "dimensions must double")
            _require(np.array_equal(target[:height, :width], source[::-1, ::-1]), "wrong upper left")
            _require(np.array_equal(target[:height, width:], source[::-1, :]), "wrong upper right")
            _require(np.array_equal(target[height:, :width], source[:, ::-1]), "wrong lower left")
            _require(np.array_equal(target[height:, width:], source), "wrong lower right")
            _require(np.array_equal(np.unique(source), np.unique(target)), "colors changed")
            if split == "train" and index < 2:
                _require(height >= 2 and width >= 2 and source.size >= 6 and _fully_asymmetric(source), "training convention is ambiguous")
            input_key = _grid_key(source)
            pair_key = (input_key, _grid_key(target))
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair")
            _require(_layout_key(source) not in seen_layouts, "recolored duplicate layout")
            _require(tuple(tuple(int(value) for value in row) for row in source) not in _SOURCE_INPUTS, "source pair copied")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
            seen_layouts.add(_layout_key(source))


def _signature(puzzle: dict) -> tuple:
    return tuple(
        tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for pair in puzzle[split])
        for split in ("train", "test")
    )


def _solid_block(grid: np.ndarray) -> bool:
    colors, counts = np.unique(grid, return_counts=True)
    dominant = colors[int(np.argmax(counts))]
    for color in colors:
        if color == dominant:
            continue
        coordinates = np.argwhere(grid == color)
        top, left = coordinates.min(axis=0)
        bottom, right = coordinates.max(axis=0)
        if bottom > top and right > left and len(coordinates) == (bottom - top + 1) * (right - left + 1):
            return True
    return False


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["train"] + puzzle["test"]:
            source, target = pair["input"], pair["output"]
            height, width = source.shape
            for row in range(2 * height):
                for column in range(2 * width):
                    reflected_row = height - 1 - row if row < height else row - height
                    reflected_column = width - 1 - column if column < width else column - width
                    assert target[row, column] == source[reflected_row, reflected_column]
    assert _signature(generate_puzzle(314159)) == _signature(generate_puzzle(314159))
    assert _signature(generate_puzzle(271828, 7, 3)) == _signature(generate_puzzle(271828, 7, 3))
    for index, (train_count, test_count) in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(10000 + index, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
    for index, (train_count, test_count) in enumerate(((None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(11000 + index, train_count, test_count)
        assert train_count is None or len(puzzle["train"]) == train_count
        assert test_count is None or len(puzzle["test"]) == test_count
    for arguments in (
        {"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5},
        {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4},
    ):
        try:
            generate_puzzle(12000, **arguments)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    assert len({_signature(puzzle) for puzzle in puzzles}) >= 40
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    shapes = set()
    palettes = set()
    layouts = set()
    symmetry = set()
    variable_shapes = 0
    variable_palettes = 0
    novel_colors = 0
    reused_palettes = 0
    familiar_shapes = 0
    novel_shapes = 0
    larger_tests = 0
    smaller_tests = 0
    source_like = 0
    symmetry_and_rich_color = 0
    boundary_and_multicolor = 0
    content = {"row_stripes": 0, "column_stripes": 0, "checker": 0, "repeated_rows": 0,
               "sparse": 0, "margin": 0, "solid_block": 0}
    for puzzle in puzzles:
        pairs = puzzle["train"] + puzzle["test"]
        pair_shapes = {pair["input"].shape for pair in pairs}
        pair_palettes = {tuple(int(value) for value in np.unique(pair["input"])) for pair in pairs}
        train_shapes = {pair["input"].shape for pair in puzzle["train"]}
        train_areas = [pair["input"].size for pair in puzzle["train"]]
        variable_shapes += len(pair_shapes) > 1
        variable_palettes += len({len(item) for item in pair_palettes}) > 1
        training_colors = set(int(value) for pair in puzzle["train"] for value in np.unique(pair["input"]))
        familiar_shapes += any(pair["input"].shape in train_shapes for pair in puzzle["test"])
        novel_shapes += any(pair["input"].shape not in train_shapes for pair in puzzle["test"])
        larger_tests += any(pair["input"].size > max(train_areas) for pair in puzzle["test"])
        smaller_tests += any(pair["input"].size < min(train_areas) for pair in puzzle["test"])
        source_like += all(len(np.unique(pair["input"])) == 3 for pair in pairs) and all(
            set(int(value) for value in np.unique(pair["input"])) <= training_colors
            for pair in puzzle["test"]
        )
        novel_colors += any(
            set(int(value) for value in np.unique(pair["input"])) - training_colors
            for pair in puzzle["test"]
        )
        reused_palettes += any(
            tuple(int(value) for value in np.unique(pair["input"]))
            in {tuple(int(value) for value in np.unique(train["input"])) for train in puzzle["train"]}
            for pair in puzzle["test"]
        )
        for pair in pairs:
            grid = pair["input"]
            height, width = grid.shape
            colors, counts = np.unique(grid, return_counts=True)
            shapes.add(grid.shape)
            palettes.add(len(colors))
            layouts.add(_layout_key(grid))
            flags = (
                bool(np.array_equal(grid, grid[:, ::-1])),
                bool(np.array_equal(grid, grid[::-1, :])),
                bool(np.array_equal(grid, grid[::-1, ::-1])),
            )
            symmetry.add(flags)
            symmetry_and_rich_color += any(flags) and len(colors) >= 4
            boundary_and_multicolor += (height == 1 or width == 1 or grid.shape == (2, 2)) and len(colors) >= 2
            content["row_stripes"] += height > 1 and len(colors) > 1 and bool(np.all(grid == grid[:, :1]))
            content["column_stripes"] += width > 1 and len(colors) > 1 and bool(np.all(grid == grid[:1, :]))
            parity = np.indices(grid.shape).sum(axis=0) % 2
            content["checker"] += len(colors) == 2 and all(
                np.all(grid[parity == value] == grid[parity == value][0]) for value in (0, 1)
            )
            content["repeated_rows"] += height > 1 and len({tuple(row) for row in grid}) < height
            content["sparse"] += len(colors) > 1 and max(counts) / grid.size >= 0.75
            border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
            content["margin"] += height >= 3 and width >= 3 and bool(
                np.all(border == colors[int(np.argmax(counts))])
            ) and len(colors) > 1
            content["solid_block"] += _solid_block(grid)
    assert len(shapes) >= 35 and len(layouts) >= 180
    assert variable_shapes >= 25 and variable_palettes >= 20
    assert novel_colors >= 4 and reused_palettes >= 4
    assert familiar_shapes >= 5 and novel_shapes >= 15
    assert larger_tests >= 5 and smaller_tests >= 5 and source_like >= 5
    assert symmetry_and_rich_color >= 5 and boundary_and_multicolor >= 5
    assert content["row_stripes"] >= 8 and content["column_stripes"] >= 8
    assert content["checker"] >= 4 and content["repeated_rows"] >= 50
    assert content["sparse"] >= 20 and content["margin"] >= 10 and content["solid_block"] >= 8
    assert {1, 2, 3, 4, 5, 6}.issubset(palettes) and max(palettes) >= 8
    assert 10 in palettes and (1, 1) in shapes
    assert any(height == 1 for height, width in shapes)
    assert any(width == 1 for height, width in shapes)
    assert (2, 2) in shapes and any(max(shape) >= 12 for shape in shapes)
    assert {(False, False, False), (True, False, False), (False, True, False), (False, False, True)}.issubset(symmetry)
