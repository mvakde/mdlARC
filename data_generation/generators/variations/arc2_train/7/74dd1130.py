import numpy as np


# Transposition has no learned color roles or hidden direction parameter.
_SOURCE_INPUTS = {
    ((9, 9, 5), (5, 5, 8), (5, 8, 9)),
    ((2, 2, 5), (6, 2, 2), (5, 5, 5)),
    ((2, 6, 6), (2, 1, 1), (2, 6, 2)),
    ((2, 2, 1), (1, 5, 1), (5, 2, 2)),
    ((9, 3, 4), (9, 4, 4), (9, 3, 4)),
}


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid_key(grid):
    return tuple(grid.shape), tuple(int(v) for v in grid.flat)


def _nested(grid):
    return tuple(tuple(int(v) for v in row) for row in grid)


def _canonical(grid):
    mapping = {}
    return tuple(grid.shape), tuple(mapping.setdefault(int(v), len(mapping)) for v in grid.flat)


def _uniform_columns(grid):
    return tuple(c for c in range(grid.shape[1]) if np.all(grid[:, c] == grid[0, c]))


def _uniform_rows(grid):
    return tuple(r for r in range(grid.shape[0]) if np.all(grid[r, :] == grid[r, 0]))


def _visible_swaps(grid):
    if grid.shape[0] != grid.shape[1]:
        return -1
    n = grid.shape[0]
    return sum(grid[r, c] != grid[c, r] for r in range(n) for c in range(r + 1, n))


def _diagnostic(grid):
    answer = grid.T
    alternatives = (grid, np.rot90(grid, 1), np.rot90(grid, 2),
                    np.rot90(grid, 3), np.flipud(grid), np.fliplr(grid),
                    np.flipud(np.fliplr(answer)))
    return all(not np.array_equal(answer, other) for other in alternatives)


def _normalize_count(value, name, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _normalize_count(num_train, "num_train", 2, 10)
    test = _normalize_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def sample_puzzle_parameters(rng):
    # These are visual biases only; none affects the output transformation.
    pool_size = int(rng.integers(5, 11))
    pool = tuple(int(v) for v in rng.choice(10, size=pool_size, replace=False))
    return {"palette_pool": pool,
            "structure_balance": int(rng.integers(0, 3)),
            "size_bias": int(rng.integers(0, 3))}


def _shape(rng, size_bias, test=False):
    p = rng.random()
    if p < (0.28 if not test else 0.22):
        return 3, 3
    if p < 0.43:
        n = int(rng.integers(4, 7))
        return n, n
    if p < (0.56 if size_bias != 2 else 0.48):
        sides = (2, int(rng.integers(4, 9)))
    elif p < (0.90 if not test else 0.84):
        sides = (int(rng.integers(3, 7)), int(rng.integers(4, 9)))
    else:
        sides = (int(rng.integers(7, 11)), int(rng.integers(8, 13)))
    if rng.random() < 0.5:
        sides = sides[::-1]
    return sides


_STYLES = ("random", "columns", "rows", "repeated_rows",
           "repeated_columns", "corner", "diagonal", "blocks", "near_symmetric")


def _style(rng, balance, split, shape):
    weights = np.array([19, 13, 13, 9, 9, 10, 9, 10, 8], dtype=float)
    if balance == 0:
        weights[1] *= 0.65 if split == "train" else 2.5
    elif balance == 2:
        weights[1] *= 2.5 if split == "train" else 0.65
    if shape != (3, 3):
        weights[-1] = 0
    return str(rng.choice(_STYLES, p=weights / weights.sum()))


def _palette(rng, pool, previous):
    if previous and rng.random() < 0.23:
        return np.asarray(previous[int(rng.integers(len(previous)))], dtype=np.int8)
    count = int(rng.choice((2, 3, 4, 5), p=(0.28, 0.36, 0.25, 0.11)))
    return np.asarray(rng.choice(pool, size=min(count, len(pool)), replace=False), dtype=np.int8)


def _construct(rng, shape, colors, style):
    h, w = shape
    dominant = int(colors[0])
    probs = rng.dirichlet(np.ones(len(colors)) * float(rng.choice((0.6, 1.5, 4.0))))
    grid = np.asarray(rng.choice(colors, size=shape, p=probs), dtype=np.int8)
    if style == "columns":
        for c in rng.choice(w, size=int(rng.integers(1, w)), replace=False):
            grid[:, c] = int(rng.choice(colors))
    elif style == "rows":
        for r in rng.choice(h, size=int(rng.integers(1, h)), replace=False):
            grid[r, :] = int(rng.choice(colors))
    elif style == "repeated_rows":
        first, second = rng.choice(h, size=2, replace=False)
        grid[second, :] = grid[first, :]
    elif style == "repeated_columns":
        first, second = rng.choice(w, size=2, replace=False)
        grid[:, second] = grid[:, first]
    elif style == "corner":
        grid[:] = dominant
        rh, rw = int(rng.integers(2, min(h, 4) + 1)), int(rng.integers(2, min(w, 4) + 1))
        r0 = 0 if rng.random() < 0.5 else h - rh
        c0 = 0 if rng.random() < 0.5 else w - rw
        grid[r0:r0 + rh, c0:c0 + rw] = rng.choice(colors[1:], size=(rh, rw))
        for _ in range(max(1, h * w // 8)):
            grid[int(rng.integers(h)), int(rng.integers(w))] = int(rng.choice(colors))
    elif style == "diagonal":
        grid[:] = dominant
        for r in range(h):
            c = min(w - 1, (r * w) // h)
            grid[r, c] = int(rng.choice(colors[1:]))
            if rng.random() < 0.5:
                grid[r, w - 1 - c] = int(rng.choice(colors[1:]))
        for _ in range(max(1, h * w // 7)):
            grid[int(rng.integers(h)), int(rng.integers(w))] = int(rng.choice(colors))
    elif style == "blocks":
        grid[:] = dominant
        for _ in range(int(rng.integers(2, 5))):
            rh = int(rng.integers(1, min(h, 4) + 1))
            rw = int(rng.integers(1, min(w, 4) + 1))
            r0 = int(rng.integers(h - rh + 1))
            c0 = int(rng.integers(w - rw + 1))
            grid[r0:r0 + rh, c0:c0 + rw] = int(rng.choice(colors[1:]))
    elif style == "near_symmetric":
        grid = np.empty((3, 3), dtype=np.int8)
        for r in range(3):
            for c in range(r, 3):
                grid[r, c] = grid[c, r] = int(rng.choice(colors))
        changes = int(rng.integers(0, 4))
        for index in rng.choice(3, size=changes, replace=False):
            r, c = ((0, 1), (0, 2), (1, 2))[int(index)]
            grid[c, r] = int(rng.choice(colors[colors != grid[r, c]]))
    return grid


def _new_pair(rng, shape, style, pool, previous_palettes, earlier, seen, diagnostic=False):
    for _ in range(300):
        matching = [grid for grid in earlier if grid.shape == shape]
        if matching and rng.random() < 0.12 and not diagnostic:
            exemplar = matching[int(rng.integers(len(matching)))]
            labels = tuple(dict.fromkeys(int(v) for v in exemplar.flat))
            colors = np.asarray(rng.choice(pool, size=len(labels), replace=False), dtype=np.int8)
            grid = np.asarray([[colors[labels.index(int(v))] for v in row] for row in exemplar], dtype=np.int8)
        else:
            colors = _palette(rng, pool, previous_palettes)
            grid = _construct(rng, shape, colors, style)
        if not 2 <= len(np.unique(grid)) <= 5 or (diagnostic and not _diagnostic(grid)):
            continue
        key = _grid_key(grid)
        if key in seen or _nested(grid) in _SOURCE_INPUTS:
            continue
        seen.add(key)
        previous_palettes.append(tuple(int(v) for v in np.unique(grid)))
        earlier.append(grid)
        return {"input": grid, "output": grid.T.copy()}
    # Bounded fallback for rare crowded small-grid cases.
    for _ in range(300):
        colors = np.asarray(rng.choice(pool, size=3, replace=False), dtype=np.int8)
        grid = np.asarray(rng.choice(colors, size=shape), dtype=np.int8)
        if len(np.unique(grid)) < 2 or (diagnostic and not _diagnostic(grid)):
            continue
        key = _grid_key(grid)
        if key not in seen and _nested(grid) not in _SOURCE_INPUTS:
            seen.add(key)
            previous_palettes.append(tuple(int(v) for v in np.unique(grid)))
            earlier.append(grid)
            return {"input": grid, "output": grid.T.copy()}
    raise RuntimeError("unable to construct a distinct transpose pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    parameters = sample_puzzle_parameters(rng)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    train_shapes = [_shape(rng, parameters["size_bias"]) for _ in range(train_count)]
    test_shapes = [_shape(rng, parameters["size_bias"], test=True) for _ in range(test_count)]
    if any(h != w for h, w in test_shapes) and not any(h != w for h, w in train_shapes):
        train_shapes[int(rng.integers(train_count))] = (3, int(rng.integers(4, 7)))
    seen, earlier, previous_palettes = set(), [], []
    puzzle = {"train": [], "test": []}
    for split, shapes in (("train", train_shapes), ("test", test_shapes)):
        for index, shape in enumerate(shapes):
            style = _style(rng, parameters["structure_balance"], split, shape)
            pair = _new_pair(rng, shape, style, parameters["palette_pool"],
                             previous_palettes, earlier, seen,
                             diagnostic=(split == "train" and index == 0))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "illegal pair counts")
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            for grid in pair.values():
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid must be a 2D array")
                _require(all(1 <= d <= 30 for d in grid.shape), "illegal dimensions")
                _require(np.issubdtype(grid.dtype, np.integer), "noninteger cells")
                _require(np.all((grid >= 0) & (grid <= 9)), "illegal colors")
            inp, out = pair["input"], pair["output"]
            _require(inp.shape[0] >= 2 and inp.shape[1] >= 2, "input too small to show transpose")
            _require(out.shape == (inp.shape[1], inp.shape[0]) and np.array_equal(out, inp.T), "not a transpose")
            _require(2 <= len(np.unique(inp)) <= 5, "unsupported palette size")
            key = _grid_key(inp)
            _require(key not in seen, "duplicate input/pair")
            _require(_nested(inp) not in _SOURCE_INPUTS, "source pair copied")
            seen.add(key)
    _require(any(_diagnostic(p["input"]) for p in train), "training does not distinguish transpose")
    if any(p["input"].shape[0] != p["input"].shape[1] for p in test):
        _require(any(p["input"].shape[0] != p["input"].shape[1] for p in train),
                 "rectangular test lacks rectangular training evidence")


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test() -> None:
    automatic = generate_puzzle(271828)
    _require(_puzzle_key(automatic) == _puzzle_key(generate_puzzle(271828)), "nondeterministic")
    for seed, train, test in ((314159, 7, 3), (1001, 2, 1), (1002, 10, 1),
                              (1003, 10, 4), (2001, None, 1), (2002, None, 4),
                              (2003, 2, None), (2004, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        _require(train is None or len(puzzle["train"]) == train, "train count ignored")
        _require(test is None or len(puzzle["test"]) == test, "test count ignored")
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed, train, test)), "nondeterministic")
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (2, 3), (4, 4), (True, 1)):
        try:
            generate_puzzle(3000, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {args}")

    train_counts, test_counts, shapes, palettes, layouts, puzzles = set(), set(), set(), set(), set(), set()
    colors, swaps, col_counts, row_counts = set(), set(), set(), set()
    varied_shapes = varied_colors = varied_columns = 0
    square3 = rectangles = unchanged = 0
    source_like = reverse = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(_puzzle_key(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        shape_set, color_set, column_set = set(), set(), set()
        train_cols, test_cols = [], []
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                independent = np.asarray([[inp[r, c] for r in range(inp.shape[0])]
                                          for c in range(inp.shape[1])])
                _require(np.array_equal(out, independent), "independent transpose check")
                shape = tuple(inp.shape)
                count = len(np.unique(inp))
                columns = len(_uniform_columns(inp))
                rows = len(_uniform_rows(inp))
                shapes.add(shape); shape_set.add(shape)
                colors.add(count); color_set.add(count)
                palettes.add(tuple(int(v) for v in np.unique(inp)))
                layouts.add(_canonical(inp))
                col_counts.add(columns); column_set.add(columns)
                row_counts.add(rows)
                swaps.add(_visible_swaps(inp))
                square3 += shape == (3, 3)
                rectangles += shape[0] != shape[1]
                unchanged += np.array_equal(inp, out)
                _require(columns == len(_uniform_rows(out)) and rows == len(_uniform_columns(out)),
                         "band transfer failed")
                (train_cols if split == "train" else test_cols).append(columns)
        varied_shapes += len(shape_set) > 1
        varied_colors += len(color_set) > 1
        varied_columns += len(column_set) > 1
        source_like += max(test_cols) > max(train_cols)
        reverse += max(train_cols) > max(test_cols)
    _require(len(train_counts) >= 4 and len(test_counts) >= 3 and len(puzzles) >= 40, "count/seed diversity")
    _require(len(shapes) >= 20 and square3 >= 20 and rectangles >= 70, "shape diversity")
    _require(varied_shapes >= 30 and varied_colors >= 20, "within-puzzle shape/color diversity")
    _require(colors == {2, 3, 4, 5} and len(palettes) >= 35, "palette diversity")
    _require(len(layouts) >= 120, "layout diversity")
    _require({0, 1, 2, 3}.issubset(swaps) and unchanged >= 1, "small/unchanged transposes absent")
    _require({0, 1, 2}.issubset(col_counts) and {0, 1}.issubset(row_counts), "band diversity")
    _require(varied_columns >= 15 and source_like >= 4 and reverse >= 4, "band balance diversity")
