"""Marker bars and nearest-marker rails for the reviewed ARC family."""

import numpy as np


_SOURCE = {
    ((3, 6, 2), (7, 8, 7), (12, 3, 8)),
    ((1, 6, 8), (3, 10, 1), (7, 3, 2), (9, 5, 3)),
    ((2, 3, 3), (7, 7, 2), (9, 4, 9)),
    ((2, 10, 6), (4, 4, 2), (11, 7, 8)),
    ((2, 3, 8), (4, 6, 2), (9, 11, 1), (12, 9, 3)),
}


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, num_train, num_test):
    if num_train is not None:
        num_train = _integer(num_train, "num_train", 2, 10)
    if num_test is not None:
        num_test = _integer(num_test, "num_test", 1, 4)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _markers(grid, background):
    points = np.argwhere(grid != background)
    points = points[np.argsort(points[:, 0], kind="stable")]
    rows = tuple(int(r) for r in points[:, 0])
    columns = tuple(int(c) for c in points[:, 1])
    colors = tuple(int(grid[r, c]) for r, c in points)
    return rows, columns, colors


def _draw(grid, background):
    rows, _, colors = _markers(grid, background)
    result = np.full(grid.shape, background, dtype=np.int8)
    markers = np.asarray(rows)
    for row in range(grid.shape[0]):
        owner = int(np.argmin(np.abs(markers - row)))
        result[row, 0] = colors[owner]
        result[row, -1] = colors[owner]
    for row, color in zip(rows, colors):
        result[row, :] = color
    result[0, :] = colors[0]
    result[-1, :] = colors[-1]
    return result


def _expected(grid, background):
    """Use midpoint intervals independently of the drawing routine."""
    rows, _, colors = _markers(grid, background)
    result = np.full(grid.shape, background, dtype=np.int8)
    for index, (row, color) in enumerate(zip(rows, colors)):
        lo = 0 if index == 0 else (rows[index - 1] + row) // 2 + 1
        hi = grid.shape[0] - 1 if index == len(rows) - 1 else (row + rows[index + 1]) // 2
        result[lo:hi + 1, 0] = color
        result[lo:hi + 1, -1] = color
        result[row, :] = color
    result[0, :] = colors[0]
    result[-1, :] = colors[-1]
    return result


def _shape(rng, demo=False):
    style = int(rng.integers(5))
    if style == 0:
        side = int(rng.integers(5, 31))
        height = width = side
    elif style == 1:
        height = int(rng.integers(12, 31))
        width = int(rng.integers(5, min(19, height)))
    elif style == 2:
        width = int(rng.integers(12, 31))
        height = int(rng.integers(5, min(19, width)))
    elif style == 3:
        height = width = 15
    else:
        height = int(rng.integers(5, 31))
        width = int(rng.integers(5, 31))
    return (max(height, 10) if demo else height, width)


def _positions(rng, length, count, demo=False):
    if demo:
        first = int(rng.integers(length - 4))
        outside = [r for r in range(length) if r < first or r > first + 4]
        extras = rng.choice(outside, size=count - 2, replace=False)
        return tuple(sorted((first, first + 4, *(int(x) for x in extras))))
    style = int(rng.integers(4))
    if style == 0 and count >= 3:
        step = int(rng.integers(1, (length - 1) // (count - 1) + 1))
        start = int(rng.integers(length - step * (count - 1)))
        return tuple(start + step * i for i in range(count))
    if style == 1:
        edge = 0 if int(rng.integers(2)) == 0 else length - 1
        rest = rng.choice([r for r in range(length) if r != edge], size=count - 1, replace=False)
        return tuple(sorted((edge, *(int(x) for x in rest))))
    if style == 2 and count >= 2:
        start = int(rng.integers(length - 1))
        rest = rng.choice([r for r in range(length) if r not in (start, start + 1)], size=count - 2, replace=False)
        return tuple(sorted((start, start + 1, *(int(x) for x in rest))))
    return tuple(sorted(int(x) for x in rng.choice(length, size=count, replace=False)))


def _columns(rng, rows, width, different=None):
    mode = int(rng.integers(4))
    if mode == 0 and all(b - a > 1 for a, b in zip(rows, rows[1:])):
        aligned = (int(rng.integers(width)),) * len(rows)
        if aligned != different:
            return aligned
    fixed = {}
    if mode == 1:
        fixed[int(rng.integers(len(rows)))] = 0 if int(rng.integers(2)) == 0 else width - 1
    if mode == 2:
        pairs = [(i, j) for i in range(len(rows)) for j in range(i + 1, len(rows)) if rows[j] - rows[i] > 1]
        if pairs:
            i, j = pairs[int(rng.integers(len(pairs)))]
            fixed[i] = fixed[j] = int(rng.integers(width))
    for _ in range(80):
        columns = []
        for i, row in enumerate(rows):
            if i in fixed:
                col = fixed[i]
            else:
                choices = [c for c in range(width) if not (i and row - rows[i - 1] == 1 and abs(c - columns[-1]) <= 1)]
                col = choices[int(rng.integers(len(choices)))]
            columns.append(col)
        candidate = tuple(columns)
        if candidate != different and all(rows[i] - rows[i - 1] > 1 or abs(candidate[i] - candidate[i - 1]) > 1 for i in range(1, len(rows))):
            return candidate
    candidate = tuple((2 * i) % width for i in range(len(rows)))
    if candidate == different:
        candidate = tuple((2 * i + 1) % width for i in range(len(rows)))
    return candidate


def _colors(rng, count, background, demo=False):
    pool = [c for c in range(10) if c != background]
    if demo or int(rng.integers(3)) == 0:
        return tuple(int(x) for x in rng.choice(pool, size=count, replace=False))
    colors = tuple(int(x) for x in rng.choice(pool, size=count, replace=True))
    if count >= 3 and int(rng.integers(2)) == 0 and len(set(colors)) == count:
        colors = colors[:-1] + (colors[0],)
    return colors


def _pair(rng, orientation, shape, demo=False, clone=None, test=False, background_choice=None, black_marker=False):
    if clone is None:
        height, width = shape
        if demo:
            count = int(rng.integers(3, min(5, height) + 1))
        elif test:
            mode = int(rng.integers(4))
            if mode == 0:
                count = int(rng.integers(1, 3))
            elif mode == 1:
                count = int(rng.integers(3, 5))
            elif mode == 2:
                count = int(rng.integers(5, min(8, height) + 1))
            else:
                count = int(rng.integers(1, min(8, height) + 1))
        else:
            count = int(rng.integers(1, min(8, height) + 1))
        rows = _positions(rng, height, count, demo)
        background = (0 if int(rng.integers(3)) == 0 else int(rng.integers(1, 10))) if background_choice is None else background_choice
        colors = _colors(rng, count, background, demo)
        if black_marker and background != 0 and 0 not in colors:
            colors = colors[:-1] + (0,)
        columns = _columns(rng, rows, width)
    else:
        height, width, rows, colors, background, old_columns = clone
        columns = _columns(rng, rows, width, old_columns)
    grid = np.full((height, width), background, dtype=np.int8)
    for row, col, color in zip(rows, columns, colors):
        grid[row, col] = color
    output = _draw(grid, background)
    if orientation == "vertical":
        grid, output = grid.T.copy(), output.T.copy()
    return {"input": grid, "output": output}, (height, width, rows, colors, background, columns)


def _source_input(grid):
    if grid.shape != (15, 15) or _background(grid) != 0:
        return False
    rows, columns, colors = _markers(grid, 0)
    return tuple(zip(rows, columns, colors)) in _SOURCE


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _integer(seed, "seed", 0, 2**63 - 1)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    orientation = "vertical" if int(rng.integers(2)) else "horizontal"
    constant_shape = int(rng.integers(5)) == 0
    clone_demo = train_count >= 3 and int(rng.integers(4)) == 0
    base_shape = _shape(rng, demo=True)
    puzzle = {"train": [], "test": []}
    seen = set()
    demo_info = None
    for split, total in (("train", train_count), ("test", test_count)):
        for index in range(total):
            demo = split == "train" and index == 0
            clone = demo_info if split == "train" and index == 1 and clone_demo else None
            for _ in range(200):
                shape = base_shape if constant_shape or demo else _shape(rng)
                final_train = split == "train" and index == train_count - 1
                background_choice = (0 if demo_info[4] != 0 else int(rng.integers(1, 10))) if final_train else None
                pair, info = _pair(rng, orientation, shape, demo=demo, clone=clone,
                                   test=split == "test", background_choice=background_choice,
                                   black_marker=(demo or final_train) and (background_choice != 0))
                key = _key(pair["input"])
                if key not in seen and not _source_input(pair["input"]):
                    break
            else:
                raise ValueError("could not construct a unique pair")
            puzzle[split].append(pair)
            seen.add(key)
            if demo:
                demo_info = info
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair, orientation):
    inp = pair["input"] if orientation == "horizontal" else pair["input"].T
    out = pair["output"] if orientation == "horizontal" else pair["output"].T
    background = _background(inp)
    rows, columns, colors = _markers(inp, background)
    _need(1 <= len(rows) <= min(9, inp.shape[0]), "invalid marker count")
    _need(len(set(rows)) == len(rows), "multiple markers on one bar")
    _need(all(rows[i] - rows[i - 1] > 1 or abs(columns[i] - columns[i - 1]) > 1 for i in range(1, len(rows))), "touching markers")
    _need(np.array_equal(out, _expected(inp, background)), "incorrect bar or rail")
    tie = any(rows[i + 1] - rows[i] >= 4 and (rows[i + 1] - rows[i]) % 2 == 0 and colors[i] != colors[i + 1] for i in range(len(rows) - 1))
    return rows, columns, colors, background, tie


def validate_generated_puzzle(puzzle: dict) -> None:
    _need(type(puzzle) is dict and set(puzzle) == {"train", "test"}, "wrong puzzle schema")
    _need(type(puzzle["train"]) is list and type(puzzle["test"]) is list, "splits must be lists")
    n, m = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= n <= 10 and 1 <= m <= 4 and n > m, "invalid split counts")
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(type(pair) is dict and set(pair) == {"input", "output"}, "wrong pair schema")
            for grid in pair.values():
                _need(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D NumPy array")
                _need(all(1 <= dim <= 30 for dim in grid.shape), "grid dimensions outside ARC limits")
                _need(np.issubdtype(grid.dtype, np.integer), "grid dtype must be integer")
                _need(bool(np.all((grid >= 0) & (grid <= 9))), "illegal ARC color")
            inp, out = pair["input"], pair["output"]
            _need(inp.shape == out.shape and min(inp.shape) >= 5, "invalid pair shape")
            _need(not np.array_equal(inp, out), "unchanged pair")
            _need(not _source_input(inp), "source input copied")
            ikey, okey = _key(inp), _key(out)
            _need(ikey not in seen_inputs and (ikey, okey) not in seen_pairs, "duplicate pair")
            seen_inputs.add(ikey)
            seen_pairs.add((ikey, okey))
    possibilities = []
    for axis in ("horizontal", "vertical"):
        try:
            analyses = {split: [_analyze(pair, axis) for pair in puzzle[split]] for split in ("train", "test")}
        except AssertionError:
            continue
        possibilities.append(analyses)
    _need(len(possibilities) == 1, "axis must be consistent and identifiable")
    training = possibilities[0]["train"]
    _need(any(x[4] for x in training), "training lacks a contrasting midpoint tie")
    _need(any(len(x[0]) >= 3 for x in training), "training lacks multiple bars")
    _need(0 in {x[3] for x in training} and any(x[3] != 0 for x in training), "training lacks background variation")
    _need(any(x[3] != 0 and 0 in x[2] for x in training), "training lacks a black marker")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"])) for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _need(len({_puzzle_key(p) for p in puzzles}) >= 40, "too few distinct puzzles")
    _need(len({len(p["train"]) for p in puzzles}) >= 4, "train counts lack diversity")
    _need(len({len(p["test"]) for p in puzzles}) >= 3, "test counts lack diversity")
    axes, shapes, backgrounds, counts, colors = set(), set(), set(), set(), set()
    aligned = edge = adjacent = boundary = equal_gaps = repeat = invariant = False
    black_marker = small = thirty = long_gap = mixed_parity = test_equal = novel_test_color = False
    aspect, test_modes = set(), set()
    combined_vertical = combined_edge = False
    variable_shapes = 0
    for puzzle in puzzles:
        records = []
        axis = next(a for a in ("horizontal", "vertical") if all(
            np.array_equal(pair["output"] if a == "horizontal" else pair["output"].T,
                           _expected(pair["input"] if a == "horizontal" else pair["input"].T,
                                     _background(pair["input"])))
            for split in ("train", "test") for pair in puzzle[split]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid = pair["input"] if axis == "horizontal" else pair["input"].T
                bg = _background(grid)
                rows, columns, palette = _markers(grid, bg)
                gaps = tuple(b - a for a, b in zip(rows, rows[1:]))
                records.append((split, grid.shape, rows, columns, palette, bg, pair["output"]))
                shapes.add(grid.shape)
                backgrounds.add(bg)
                counts.add(len(rows))
                colors.update(palette)
                aligned |= len(columns) > 1 and len(set(columns)) == 1
                edge |= any(c in (0, grid.shape[1] - 1) for c in columns)
                adjacent |= any(b - a == 1 for a, b in zip(rows, rows[1:]))
                boundary |= 0 in rows or grid.shape[0] - 1 in rows
                equal_gaps |= len(rows) >= 3 and len(set(np.diff(rows))) == 1
                repeat |= len(set(palette)) < len(palette)
                black_marker |= bg != 0 and 0 in palette
                small |= min(grid.shape) <= 7
                thirty |= max(grid.shape) == 30
                long_gap |= bool(gaps) and max(gaps) >= 8
                mixed_parity |= any(g % 2 for g in gaps) and any(g % 2 == 0 for g in gaps)
                aspect.add("square" if grid.shape[0] == grid.shape[1] else ("tall" if grid.shape[0] > grid.shape[1] else "wide"))
                combined_vertical |= axis == "vertical" and bg != 0 and len(set(palette)) < len(palette)
                combined_edge |= bg != 0 and 0 in palette and 1 in gaps and any(c in (0, grid.shape[1] - 1) for c in columns)
                if split == "test":
                    test_modes.add("simple" if len(rows) <= 2 else ("familiar" if len(rows) <= 4 else "larger"))
                    test_equal |= len(gaps) >= 2 and len(set(gaps)) == 1
        axes.add(axis)
        variable_shapes += len({x[1] for x in records}) > 1
        train = [x for x in records if x[0] == "train"]
        invariant |= any(a[1:3] == b[1:3] and a[4:6] == b[4:6] and a[3] != b[3] and np.array_equal(a[6], b[6]) for i, a in enumerate(train) for b in train[i + 1:])
        train_colors = {color for item in train for color in item[4]}
        novel_test_color |= any(set(item[4]) - train_colors for item in records if item[0] == "test")
    _need(axes == {"horizontal", "vertical"}, "missing orientation")
    _need(len(shapes) >= 35 and len({h for h, _ in shapes}) >= 15 and len({w for _, w in shapes}) >= 15, "shape diversity too low")
    _need(variable_shapes >= 15, "too few puzzles vary shape internally")
    _need(backgrounds == set(range(10)) and len(colors) == 10, "color diversity too low")
    _need({1, 2, 3, 4, 5, 6}.issubset(counts), "marker-count diversity too low")
    _need(all((aligned, edge, adjacent, boundary, equal_gaps, repeat, invariant)), "a review variation is absent")
    _need(all((black_marker, small, thirty, long_gap, mixed_parity, test_equal, novel_test_color)), "a color, size, gap, or test variation is absent")
    _need(aspect == {"square", "tall", "wide"} and test_modes == {"simple", "familiar", "larger"}, "aspect or test-count regimes are missing")
    _need(combined_vertical and combined_edge, "compatible variation axes did not combine")
    a = generate_puzzle(731, 7, 3)
    _need(_puzzle_key(a) == _puzzle_key(generate_puzzle(731, 7, 3)), "nondeterministic generation")
    _need(_puzzle_key(a) != _puzzle_key(generate_puzzle(732, 7, 3)), "seeds have same puzzle")
    for n, m in ((2, 1), (10, 1), (10, 4), (6, 2)):
        p = generate_puzzle(9000 + n * 10 + m, n, m)
        _need((len(p["train"]), len(p["test"])) == (n, m), "explicit counts ignored")
    for seed, n, m in ((101, None, 1), (102, None, 4), (103, 2, None), (104, 10, None)):
        p = generate_puzzle(seed, n, m)
        _need((n is None or len(p["train"]) == n) and (m is None or len(p["test"]) == m), "partial count ignored")
    for n, m in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(17, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
