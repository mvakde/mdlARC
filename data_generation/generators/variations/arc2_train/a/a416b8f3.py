import numpy as np


_SOURCE_INPUTS = {
    ((3, 0, 0), (2, 3, 0), (2, 1, 8), (0, 1, 0)),
    ((0, 5, 0), (5, 5, 2), (0, 0, 0)),
    ((5, 2, 3, 0), (2, 5, 3, 0), (5, 2, 8, 8), (0, 0, 6, 0)),
    ((4, 0, 0, 0), (4, 5, 0, 0), (0, 5, 6, 0), (6, 6, 1, 0), (0, 0, 0, 1)),
}

_STYLES = (
    "solid", "mosaic", "stripes", "blocks", "lines", "border",
    "dots", "motif", "walk", "checker", "margin",
)
_STYLE_WEIGHTS = (0.08, 0.18, 0.11, 0.11, 0.10, 0.08, 0.10, 0.09, 0.06, 0.05, 0.04)


def _require(condition: bool, message: str) -> None:
    if not bool(condition):
        raise AssertionError(message)


def _validate_seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    value = int(seed)
    if value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return value


def _optional_count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _optional_count("num_train", num_train, 2, 10)
    test = _optional_count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        low = max(2, test + 1)
        train = low + seed % (11 - low)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must be greater than num_test")
    return train, test


def _grid_key(grid: np.ndarray) -> tuple:
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _source_copy(grid: np.ndarray) -> bool:
    return tuple(tuple(int(x) for x in row) for row in grid) in _SOURCE_INPUTS


def _sample_shape(rng: np.random.Generator) -> tuple[int, int]:
    # Height and width are sampled anew for every pair. The categories give
    # useful mass to extremes without restricting the possible rectangles.
    category = int(rng.integers(0, 100))
    if category < 27:
        return int(rng.integers(2, 10)), int(rng.integers(2, 9))
    if category < 43:
        return int(rng.integers(1, 9)), int(rng.integers(8, 16))
    if category < 55:
        return int(rng.integers(8, 31)), int(rng.integers(1, 5))
    if category < 67:
        side = int(rng.integers(2, 16))
        return side, side
    if category < 79:
        if rng.random() < 0.5:
            return 1, int(rng.integers(1, 16))
        return int(rng.integers(2, 31)), 1
    if category < 89:
        return int(rng.integers(12, 31)), int(rng.integers(7, 16))
    return int(rng.integers(1, 31)), int(rng.integers(1, 16))


def _test_shape(rng: np.random.Generator, train_shapes: list[tuple[int, int]]) -> tuple[int, int]:
    relation = int(rng.integers(0, 100))
    reference = train_shapes[int(rng.integers(len(train_shapes)))]
    maximum = max(h for h, _ in train_shapes)
    if relation < 18:
        return reference  # same shape is a valid, useful transfer case
    if relation < 38 and maximum > 1:
        return int(rng.integers(1, maximum)), _sample_shape(rng)[1]
    if relation < 58 and maximum < 30:
        return min(30, maximum + int(rng.integers(1, 5))), _sample_shape(rng)[1]
    if relation < 74:
        h, w = train_shapes[0]
        return min(30, h + int(rng.integers(1, 4))), min(15, max(1, w + int(rng.integers(-2, 3))))
    return _sample_shape(rng)


def _style_capacity(style: str, height: int, width: int) -> int:
    area = height * width
    if style == "solid":
        return 1
    if style == "stripes":
        return min(6, max(height, width))
    if style == "border":
        return 3 if min(height, width) >= 4 else 2 if min(height, width) >= 3 else 1
    if style == "motif":
        return min(6, height * (width // 2)) if width >= 4 else 1
    if style == "margin":
        return min(6, (height - 2) * (width - 2)) if min(height, width) >= 4 else 1
    if style == "checker":
        return min(6, height + width - 1)
    if style in ("blocks", "lines"):
        return min(5, max(height, width))
    return min(6, area)


def _choose_style(rng: np.random.Generator, height: int, width: int) -> tuple[str, int]:
    style = str(rng.choice(_STYLES, p=_STYLE_WEIGHTS))
    capacity = _style_capacity(style, height, width)
    if capacity < 2 and style != "solid":
        style = "mosaic" if height * width >= 2 else "solid"
        capacity = _style_capacity(style, height, width)
    if style == "solid" or capacity == 1:
        return "solid", 1
    if style == "walk":
        capacity = min(capacity, 3)
    if style == "lines":
        capacity = min(capacity, 4)
    return style, int(rng.integers(2, capacity + 1))


def _mosaic(rng: np.random.Generator, height: int, width: int, colors: tuple[int, ...]) -> np.ndarray:
    count = height * width
    labels = np.concatenate((
        np.arange(len(colors), dtype=np.int8),
        rng.integers(0, len(colors), size=count - len(colors), dtype=np.int8),
    ))
    rng.shuffle(labels)
    return np.asarray(colors, dtype=np.int8)[labels].reshape(height, width)


def _pattern(rng: np.random.Generator, height: int, width: int, colors: tuple[int, ...], style: str) -> np.ndarray:
    k = len(colors)
    grid = np.full((height, width), colors[0], dtype=np.int8)
    if style == "solid":
        return grid
    if style == "mosaic":
        return _mosaic(rng, height, width, colors)
    if style == "stripes":
        axes = [axis for axis, length in enumerate((height, width)) if length >= k]
        axis = int(rng.choice(axes))
        length = (height, width)[axis]
        band = int(rng.integers(1, min(4, length // k) + 1))
        indices = (np.arange(length) // band + int(rng.integers(k))) % k
        if axis == 0:
            grid[:] = np.asarray(colors, dtype=np.int8)[indices, None]
        else:
            grid[:] = np.asarray(colors, dtype=np.int8)[indices][None, :]
        return grid
    if style in ("blocks", "lines"):
        axis = 0 if height >= width else 1
        length = (height, width)[axis]
        segments = np.array_split(np.arange(length), k)
        for i in range(1, k):
            segment = segments[i - 1]
            lo, hi = int(segment[0]), int(segment[-1]) + 1
            start = int(rng.integers(lo, hi))
            end = int(rng.integers(start + 1, hi + 1))
            if style == "lines":
                if axis == 0:
                    col = int(rng.integers(width))
                    grid[start:end, col] = colors[i]
                else:
                    row = int(rng.integers(height))
                    grid[row, start:end] = colors[i]
            elif axis == 0:
                col_start = int(rng.integers(width))
                col_end = int(rng.integers(col_start + 1, width + 1))
                grid[start:end, col_start:col_end] = colors[i]
            else:
                row_start = int(rng.integers(height))
                row_end = int(rng.integers(row_start + 1, height + 1))
                grid[row_start:row_end, start:end] = colors[i]
        return grid
    if style == "border":
        grid[0, :] = colors[1]
        grid[-1, :] = colors[1]
        grid[:, 0] = colors[1]
        grid[:, -1] = colors[1]
        if k == 3:
            grid[1:-1, 1:-1] = colors[2]
            grid[height // 2, width // 2] = colors[0]
        return grid
    if style == "dots":
        positions = rng.permutation(height * width)
        for i in range(1, k):
            grid.flat[int(positions[i - 1])] = colors[i]
        extra = int(rng.integers(0, max(2, height * width // 5 + 1)))
        for pos in positions[k - 1:k - 1 + extra]:
            grid.flat[int(pos)] = colors[int(rng.integers(1, k))]
        return grid
    if style == "motif":
        minimum = max(2, (k + height - 1) // height)
        period = int(rng.integers(minimum, width // 2 + 1))
        motif = _mosaic(rng, height, period, colors)
        return np.tile(motif, (1, (width + period - 1) // period))[:, :width]
    if style == "walk":
        row, col = int(rng.integers(height)), int(rng.integers(width))
        steps = max(k - 1, int(rng.integers(2, max(3, height * width // 2 + 2))))
        for step in range(steps):
            grid[row, col] = colors[1 + step % (k - 1)]
            direction = int(rng.integers(4))
            row = min(height - 1, max(0, row + (-1 if direction == 0 else 1 if direction == 1 else 0)))
            col = min(width - 1, max(0, col + (-1 if direction == 2 else 1 if direction == 3 else 0)))
        return grid
    if style == "checker":
        phase = int(rng.integers(k))
        grid[:] = np.asarray(colors, dtype=np.int8)[(np.arange(height)[:, None] + np.arange(width)[None, :] + phase) % k]
        if rng.random() < 0.5:
            grid = np.fliplr(grid).copy()
        return grid
    if style == "margin":
        top = int(rng.integers(1, height - 1))
        left = int(rng.integers(1, width - 1))
        bottom = int(rng.integers(top + 1, height))
        right = int(rng.integers(left + 1, width))
        if (bottom - top) * (right - left) < k:
            top, left, bottom, right = 1, 1, height - 1, width - 1
        grid[top:bottom, left:right] = _mosaic(rng, bottom - top, right - left, colors)
        return grid
    raise AssertionError("unknown pattern style")


def _make_grid(rng: np.random.Generator, height: int, width: int, colors: tuple[int, ...], style: str, witness: bool = False) -> np.ndarray:
    grid = _pattern(rng, height, width, colors, style)
    repeated = style == "motif"
    if not witness and not repeated and width >= 4 and rng.random() < 0.11:
        period = int(rng.integers(2, width // 2 + 1))
        grid = np.tile(grid[:, :period], (1, (width + period - 1) // period))[:, :width].copy()
        repeated = True
    symmetric = False
    if not witness and not repeated and width >= 2 and rng.random() < 0.14:
        grid[:, width // 2 + width % 2:] = np.fliplr(grid[:, :width // 2])
        symmetric = True
    if not witness and not repeated and not symmetric and width >= 2 and len(colors) >= 2:
        mode = int(rng.integers(4))
        row = int(rng.integers(height))
        if mode == 0:
            choices = [color for color in colors if color != 0]
            color = int(rng.choice(choices if choices else colors))
            grid[row, 0] = grid[row, -1] = color
        elif mode == 1:
            alternatives = [color for color in colors if color != int(grid[row, 0])]
            grid[row, -1] = int(rng.choice(alternatives))
    return grid


def _palette_pool(rng: np.random.Generator) -> tuple[int, ...]:
    size = int(rng.integers(3, 8))
    if rng.random() < 0.65:
        rest = rng.choice(np.arange(1, 10), size=size - 1, replace=False)
        return (0,) + tuple(int(x) for x in rest)
    return tuple(int(x) for x in rng.choice(np.arange(1, 10), size=size, replace=False))


def _palette(rng: np.random.Generator, pool: tuple[int, ...], count: int) -> tuple[int, ...]:
    return tuple(int(x) for x in rng.choice(np.asarray(pool), size=count, replace=False))


def _test_palette(rng: np.random.Generator, known: set[int], count: int) -> tuple[tuple[int, ...], int]:
    unseen = tuple(color for color in range(10) if color not in known)
    mode = int(rng.integers(0, 100))
    desired = 0 if mode < 30 else 1 if mode < 60 else int(rng.integers(2, 5))
    novel = min(desired, len(unseen), count)
    count = min(count, novel + len(known))
    selected = list(_palette(rng, unseen, novel)) if novel else []
    selected.extend(_palette(rng, tuple(sorted(known)), count - novel))
    rng.shuffle(selected)
    return tuple(selected), novel


def _witness(rng: np.random.Generator, pool: tuple[int, ...], shape: tuple[int, int] | None = None) -> np.ndarray:
    if shape is None:
        height = int(rng.integers(3, 10))
        width = int(rng.integers(3, 10))
    else:
        height, width = shape
    style = "mosaic" if rng.random() < 0.65 else "stripes"
    count = int(rng.integers(3, min(5, len(pool), max(height, width) if style == "stripes" else height * width) + 1))
    colors = _palette(rng, pool, count)
    for _ in range(12):
        grid = _make_grid(rng, height, width, colors, style, witness=True)
        if len(np.unique(grid)) >= 3 and not np.array_equal(grid, np.fliplr(grid)) and not _horizontal_repeat(grid):
            return grid
    grid = _mosaic(rng, height, width, colors)
    grid[0, :] = colors[0]
    grid[0, 0] = colors[1]
    grid[0, -1] = colors[2]
    return grid


def _horizontal_repeat(grid: np.ndarray) -> bool:
    width = grid.shape[1]
    return any(np.array_equal(grid[:, period:], grid[:, :-period]) for period in range(1, width // 2 + 1))


def _fallback(pair_number: int, pool: tuple[int, ...]) -> np.ndarray:
    height = 6 + pair_number
    width = 4 + pair_number % 10
    grid = np.full((height, width), pool[0], dtype=np.int8)
    grid[0, 0] = pool[1]
    grid[-1, -1] = pool[2]
    return grid


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _validate_seed(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # No hidden semantic convention exists: every literal input cell is copied.
    # This pool only keeps some colors available for varied test palette shifts.
    train_pool = _palette_pool(rng)
    compact_progression = bool(rng.random() < 0.16)
    puzzle = {"train": [], "test": []}
    seen = set()
    train_shapes = []
    train_used = set()

    for index in range(train_count + test_count):
        split = "train" if index < train_count else "test"
        pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
        if index == 0:
            height = width = 0
        elif split == "train":
            if compact_progression:
                height = int(pair_rng.integers(3, 5))
                width = int(pair_rng.integers(3, 5))
            else:
                height, width = _sample_shape(pair_rng)
            if index == 1 and (height, width) == train_shapes[0]:
                width = 7 - width if compact_progression else width + 1 if width < 15 else width - 1
        else:
            if compact_progression and index == train_count:
                height, width = 5, int(pair_rng.integers(3, 5))
            else:
                height, width = _test_shape(pair_rng, train_shapes)

        for attempt in range(24):
            if index == 0:
                shape = (int(pair_rng.integers(3, 5)), int(pair_rng.integers(3, 5))) if compact_progression else None
                grid = _witness(pair_rng, train_pool, shape)
            else:
                style, count = _choose_style(pair_rng, height, width)
                if split == "train":
                    count = min(count, len(train_pool))
                    colors = _palette(pair_rng, train_pool, count)
                    intended_novel = 0
                else:
                    colors, intended_novel = _test_palette(pair_rng, train_used, count)
                grid = _make_grid(pair_rng, height, width, colors, style)
                if split == "test" and len(set(int(x) for x in np.unique(grid)) - train_used) < intended_novel:
                    continue
            key = _grid_key(grid)
            if index == 1 and set(int(x) for x in np.unique(grid)) == set(int(x) for x in np.unique(puzzle["train"][0]["input"])):
                continue
            if key not in seen and not _source_copy(grid):
                break
        else:
            grid = _fallback(index, train_pool)
            if index == 1:
                grid[-1, -1] = train_pool[0]
            key = _grid_key(grid)
            for marker in range(10):
                if key not in seen and not _source_copy(grid):
                    break
                grid[1, marker % grid.shape[1]] = (int(grid[1, marker % grid.shape[1]]) + 1) % 10
                key = _grid_key(grid)
            else:
                raise RuntimeError("could not construct a unique pair")

        seen.add(key)
        pair = {"input": grid, "output": np.concatenate((grid, grid), axis=1)}
        puzzle[split].append(pair)
        if split == "train":
            train_shapes.append(grid.shape)
            train_used.update(int(x) for x in np.unique(grid))

    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle must have exactly train and test")
    _require(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits must be lists")
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count, "illegal pair counts")
    seen_inputs = set()
    seen_pairs = set()
    train_shapes = set()
    train_palettes = set()
    witnesses = 0
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair must have exactly input and output")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a two-dimensional NumPy array")
                h, w = grid.shape
                _require(1 <= h <= 30 and 1 <= w <= 30, "grid shape out of bounds")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, "illegal ARC color")
            h, w = inp.shape
            _require(out.shape == (h, 2 * w), "output must double input width")
            _require(np.array_equal(out[:, :w], inp) and np.array_equal(out[:, w:], inp), "output must contain two exact copies")
            _require(not _source_copy(inp), "exact source pair copied")
            inp_key = _grid_key(inp)
            pair_key = (inp_key, _grid_key(out))
            _require(inp_key not in seen_inputs and pair_key not in seen_pairs, "duplicate pair or input")
            seen_inputs.add(inp_key)
            seen_pairs.add(pair_key)
            if split == "train":
                train_shapes.add(inp.shape)
                train_palettes.add(tuple(int(x) for x in np.unique(inp)))
                if h >= 2 and w >= 2 and len(np.unique(inp)) >= 3 and not np.array_equal(inp, np.fliplr(inp)) and not _horizontal_repeat(inp):
                    witnesses += 1
    _require(len(train_shapes) >= 2, "training shapes must vary within the puzzle")
    _require(len(train_palettes) >= 2, "training palettes must vary within the puzzle")
    _require(witnesses >= 1, "training must contain an asymmetric multicolor whole-grid witness")


def _fingerprint(puzzle: dict) -> tuple:
    return tuple((split, tuple((_grid_key(pair["input"]), _grid_key(pair["output"])) for pair in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4, "train counts did not vary")
    _require(len({len(p["test"]) for p in puzzles}) >= 3, "test counts did not vary")
    _require(len({_fingerprint(p) for p in puzzles}) >= 40, "cross-seed puzzles repeated")
    shapes, normalized, palettes = set(), set(), set()
    all_pairs = []
    relations, novelty = set(), set()
    source_like = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        if all(3 <= pair["input"].shape[0] <= 4 and 3 <= pair["input"].shape[1] <= 4 for pair in puzzle["train"]) and any(pair["input"].shape[0] == 5 and pair["input"].shape[1] in (3, 4) for pair in puzzle["test"]):
            source_like += 1
        train_palette = {int(x) for pair in puzzle["train"] for x in np.unique(pair["input"])}
        max_train_height = max(pair["input"].shape[0] for pair in puzzle["train"])
        train_shapes = {pair["input"].shape for pair in puzzle["train"]}
        for pair in puzzle["test"]:
            grid = pair["input"]
            relations.add("short" if grid.shape[0] < max_train_height else "same" if grid.shape[0] == max_train_height else "tall")
            if grid.shape in train_shapes:
                relations.add("same_shape")
            new_count = len(set(int(x) for x in np.unique(grid)) - train_palette)
            novelty.add(min(2, new_count))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"]
                _require(np.array_equal(pair["output"], np.tile(grid, (1, 2))), "wrong horizontal copy")
                all_pairs.append(grid)
                shapes.add(grid.shape)
                colors = tuple(int(x) for x in np.unique(grid))
                palettes.add(colors)
                mapping = {color: i for i, color in enumerate(colors)}
                normalized.add((grid.shape, tuple(mapping[int(x)] for x in grid.flat)))
    _require(len(shapes) >= 45 and len(normalized) >= 150 and len(palettes) >= 35, "structural or palette diversity is too low")
    _require(relations == {"short", "same", "tall", "same_shape"}, "train/test size relations are incomplete")
    _require(source_like >= 3, "modest source-like size shifts are absent")
    _require(novelty == {0, 1, 2}, "test palette shifts are incomplete")
    _require(any(h == 1 for h, w in shapes) and any(w == 1 for h, w in shapes), "one-dimensional strips are absent")
    _require(any(h >= 18 and w >= 10 for h, w in shapes) and any(w >= 12 and h <= 8 for h, w in shapes), "large or wide shapes are absent")
    _require(any(len(np.unique(g)) == 1 for g in all_pairs) and any(len(np.unique(g)) == 2 for g in all_pairs) and any(len(np.unique(g)) >= 5 for g in all_pairs), "palette sizes are too narrow")
    _require(any(0 not in g for g in all_pairs) and any(_horizontal_repeat(g) for g in all_pairs), "zero-free or repeating inputs are absent")
    _require(any(g.shape[1] > 1 and np.array_equal(g, np.fliplr(g)) for g in all_pairs), "symmetric inputs are absent")
    _require(any(g.shape[1] > 1 and np.any(g[:, 0] == g[:, -1]) for g in all_pairs), "continuous joins are absent")
    _require(any(g.shape[1] > 1 and np.any(g[:, 0] != g[:, -1]) for g in all_pairs), "contrasting joins are absent")
    for seed, train, test in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4), (1004, 7, 3), (1101, None, 1), (1102, None, 4), (1103, 2, None), (1104, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        if train is not None:
            _require(len(puzzle["train"]) == train, "explicit train count ignored")
        if test is not None:
            _require(len(puzzle["test"]) == test, "explicit test count ignored")
        validate_generated_puzzle(puzzle)
    _require(_fingerprint(generate_puzzle(20260827, 8, 3)) == _fingerprint(generate_puzzle(20260827, 8, 3)), "explicit generation is nondeterministic")
    _require(_fingerprint(generate_puzzle(314159)) == _fingerprint(generate_puzzle(314159)), "automatic generation is nondeterministic")
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0}, {"num_test": 5}, {"num_train": 2, "num_test": 2}, {"num_train": 3, "num_test": 4}, {"num_train": 4, "num_test": 4}):
        try:
            generate_puzzle(0, **args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")
