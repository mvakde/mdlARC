"""Whole-grid, two-axis mirror puzzles."""

import numpy as np


_SOURCE_INPUTS = {
    (2, 5, 2, 2, 6, 4, 2, 2, 2),
    (7, 1, 5, 7, 7, 1, 5, 3, 1),
    (1, 2, 1, 2, 8, 1, 8, 1, 6),
    (5, 3, 4, 3, 4, 5, 3, 4, 4),
    (1, 6, 6, 5, 2, 2, 2, 2, 2),
}


def _key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _source_copy(grid):
    return grid.shape == (3, 3) and _key(grid)[1] in _SOURCE_INPUTS


def _expand(grid):
    top = np.concatenate((grid, grid[:, ::-1]), axis=1)
    return np.concatenate((top, top[::-1, :]), axis=0)


def _asymmetric(grid):
    return (not np.array_equal(grid, grid[:, ::-1])
            and not np.array_equal(grid, grid[::-1, :])
            and not np.array_equal(grid, grid[::-1, ::-1]))


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _counts(seed, rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        # All four test counts and many train counts occur in a short run.
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _side(rng, regime):
    bins = ((2, 5), (5, 9), (9, 13), (13, 16))
    probabilities = ((0.46, 0.36, 0.14, 0.04)
                     if regime == "source_train" else (0.34, 0.33, 0.21, 0.12))
    low, high = bins[int(rng.choice(4, p=probabilities))]
    return int(rng.integers(low, high))


def _shape(rng, regime, first_train):
    draw = float(rng.random())
    if first_train and draw < 0.42:
        return 3, 3
    if draw < (0.22 if regime == "source_train" else 0.12):
        return 3, 3
    if draw < (0.31 if regime == "source_train" else 0.21):
        return (2, 4) if rng.random() < 0.5 else (4, 2)
    if draw < (0.35 if regime == "source_train" else 0.33):
        return int(rng.integers(13, 16)), int(rng.integers(13, 16))
    return _side(rng, regime), _side(rng, regime)


def _palette_size(rng, cells, regime, witness):
    if witness:
        return int(rng.integers(3, min(7, cells) + 1))
    weights = ((3, 8, 16, 16, 12, 10, 9, 8, 8, 10)
               if regime == "source_train" else (9, 14, 12, 12, 9, 9, 9, 9, 9, 8))
    choices = np.arange(1, min(10, cells) + 1)
    weights = np.asarray(weights[:len(choices)], dtype=float)
    return int(rng.choice(choices, p=weights / weights.sum()))


def _make_pattern(rng, shape, palette, regime, witness):
    h, w = shape
    k = len(palette)
    mode = int(rng.choice(7, p=(0.28, 0.20, 0.13, 0.13, 0.07, 0.10, 0.09)))
    if mode == 1:
        # A frequent color remains part of the tile, not a background.
        frequent = float(rng.uniform(0.68, 0.94))
        probabilities = np.full(k, (1 - frequent) / max(1, k - 1))
        probabilities[0] = frequent if k > 1 else 1
        grid = rng.choice(palette, size=shape, p=probabilities)
    elif mode == 2:
        grid = np.broadcast_to(rng.choice(palette, size=(h, 1)), shape).copy()
    elif mode == 3:
        grid = np.broadcast_to(rng.choice(palette, size=(1, w)), shape).copy()
    elif mode == 4:
        parity = np.fromfunction(lambda r, c: (r + c) % 2, shape, dtype=int)
        grid = palette[np.asarray(parity, dtype=int) % k]
    else:
        grid = rng.choice(palette, size=shape)
        if mode == 6:
            if rng.random() < 0.5:
                src, dst = rng.choice(h, size=2, replace=False)
                grid[dst, :] = grid[src, :]
            else:
                src, dst = rng.choice(w, size=2, replace=False)
                grid[:, dst] = grid[:, src]
    grid = np.array(grid, dtype=np.int64, copy=True)

    # Overlays combine with the base pattern, palette, and symmetry.
    if rng.random() < 0.23:
        r0 = int(rng.integers(0, h - 1))
        c0 = int(rng.integers(0, w - 1))
        r1 = int(rng.integers(r0 + 1, h + 1))
        c1 = int(rng.integers(c0 + 1, w + 1))
        grid[r0:r1, c0:c1] = int(rng.choice(palette))
    if rng.random() < 0.19:
        edge = int(rng.integers(4))
        color = int(rng.choice(palette))
        if edge == 0:
            grid[0, :] = color
        elif edge == 1:
            grid[-1, :] = color
        elif edge == 2:
            grid[:, 0] = color
        else:
            grid[:, -1] = color
    if mode in (2, 3, 4) and rng.random() < 0.34:
        for _ in range(int(rng.integers(1, min(4, h * w) + 1))):
            grid[int(rng.integers(h)), int(rng.integers(w))] = int(rng.choice(palette))

    if witness:
        return grid
    probabilities = ((0.55, 0.12, 0.12, 0.09, 0.12)
                     if regime == "source_train" else (0.43, 0.16, 0.16, 0.10, 0.15))
    symmetry = int(rng.choice(5, p=probabilities))
    if symmetry in (1, 4):
        grid[:, w - w // 2:] = grid[:, :w // 2][:, ::-1]
    if symmetry in (2, 4):
        grid[h - h // 2:, :] = grid[:h // 2, :][::-1, :]
    if symmetry == 3:
        for r in range(h):
            for c in range(w):
                opposite = (h - r - 1, w - c - 1)
                if (r, c) > opposite:
                    grid[r, c] = grid[opposite]
    return grid


def _pair(rng, regime, witness, forbidden_shape, used):
    for _ in range(48):
        shape = _shape(rng, regime, witness)
        if shape == forbidden_shape:
            continue
        cells = shape[0] * shape[1]
        if cells <= 10 and rng.random() < 0.24:
            # A compact tile can show almost every cell in a different color.
            # The one repeated label is placed independently of shape/orientation.
            k = cells - int(rng.integers(0, 2))
            palette = rng.choice(10, size=k, replace=False)
            labels = np.concatenate((np.arange(k), rng.integers(0, k, size=cells-k)))
            rng.shuffle(labels)
            grid = np.asarray(palette[labels].reshape(shape), dtype=np.int64)
        else:
            k = _palette_size(rng, cells, regime, witness)
            palette = rng.choice(10, size=k, replace=False)
            grid = _make_pattern(rng, shape, palette, regime, witness)
        if (_key(grid) in used or _source_copy(grid)
                or (witness and (not _asymmetric(grid) or np.unique(grid).size < 3))):
            continue
        return {"input": grid, "output": _expand(grid)}
    # Bounded deterministic fallback for rare collisions.
    for h in range(2, 16):
        for w in range(2, 16):
            if (h, w) == forbidden_shape:
                continue
            grid = rng.integers(0, 10, size=(h, w), dtype=np.int64)
            if (_key(grid) in used or _source_copy(grid)
                    or (witness and (not _asymmetric(grid) or np.unique(grid).size < 3))):
                continue
            return {"input": grid, "output": _expand(grid)}
    raise RuntimeError("could not construct a distinct mirrored pair")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    puzzle = {"train": [], "test": []}
    used = set()
    # The only puzzle-level convention is full-grid reflection. Colors,
    # dimensions, and equality patterns are independent pair properties.
    for i in range(train_count):
        forbidden_shape = puzzle["train"][0]["input"].shape if i == 1 else None
        pair = _pair(rng, "source_train", i == 0, forbidden_shape, used)
        used.add(_key(pair["input"]))
        puzzle["train"].append(pair)
    for _ in range(test_count):
        pair = _pair(rng, "target_test", False, None, used)
        used.add(_key(pair["input"]))
        puzzle["test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid_ok(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= side <= 30 for side in grid.shape)
    assert bool(np.all((grid >= 0) & (grid <= 9)))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    used_inputs, used_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            _grid_ok(grid)
            _grid_ok(output)
            h, w = grid.shape
            assert 2 <= h <= 15 and 2 <= w <= 15
            assert output.shape == (2 * h, 2 * w)
            assert np.array_equal(output[:h, :w], grid)
            assert np.array_equal(output[:h, w:], grid[:, ::-1])
            assert np.array_equal(output[h:, :w], grid[::-1, :])
            assert np.array_equal(output[h:, w:], grid[::-1, ::-1])
            assert set(int(x) for x in np.unique(output)) == set(int(x) for x in np.unique(grid))
            input_key, output_key = _key(grid), _key(output)
            assert input_key not in used_inputs
            assert (input_key, output_key) not in used_pairs
            assert not _source_copy(grid)
            used_inputs.add(input_key)
            used_pairs.add((input_key, output_key))
    # One two-dimensional witness distinguishes both reflections from plain
    # repetition. Another shape demonstrates that scale is not fixed.
    assert any(_asymmetric(p["input"]) for p in puzzle["train"])
    assert len({p["input"].shape for p in puzzle["train"]}) >= 2


def _signature(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _invalid_counts(**kwargs):
    try:
        generate_puzzle(6121, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError: {kwargs}")


def self_test() -> None:
    for args in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4),
                 (2, None), (10, None), (7, 3)):
        puzzle = generate_puzzle(8187, *args)
        validate_generated_puzzle(puzzle)
        assert args[0] is None or len(puzzle["train"]) == args[0]
        assert args[1] is None or len(puzzle["test"]) == args[1]
        assert _signature(puzzle) == _signature(generate_puzzle(8187, *args))
    for bad in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                {"num_test": 5}, {"num_train": 2, "num_test": 2},
                {"num_train": 3, "num_test": 4}):
        _invalid_counts(**bad)
    damaged = generate_puzzle(214, 4, 2)
    old = int(damaged["test"][0]["output"][0, 0])
    damaged["test"][0]["output"][0, 0] = (old + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted an incorrect output")

    shapes, palettes, structures, signatures = set(), set(), set(), set()
    train_counts, test_counts = set(), set()
    train_redundant = test_asymmetric = whole_symmetric = 0
    large = dominant = near_unique = 0
    coverage = {}

    def mark(name, condition):
        coverage[name] = coverage.get(name, 0) + int(bool(condition))

    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        assert _signature(puzzle) == _signature(generate_puzzle(seed))
        train_shapes = {p["input"].shape for p in puzzle["train"]}
        train_areas = [p["input"].size for p in puzzle["train"]]
        train_color_counts = {int(np.unique(p["input"]).size) for p in puzzle["train"]}
        mark("within_puzzle_shapes", len(train_shapes) >= 2)
        mark("within_puzzle_palettes", len(train_color_counts) >= 2)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                grid, output = pair["input"], pair["output"]
                h, w = grid.shape
                # Index-based check is separate from the concatenation code.
                for r in range(2 * h):
                    for c in range(2 * w):
                        assert output[r, c] == grid[min(r, 2 * h - 1 - r),
                                                     min(c, 2 * w - 1 - c)]
                shapes.add(grid.shape)
                color_count = int(np.unique(grid).size)
                palettes.add(color_count)
                labels = {}
                structures.add((grid.shape, tuple(labels.setdefault(int(v), len(labels)) for v in grid.flat)))
                counts = np.unique(grid, return_counts=True)[1]
                dominant += int(counts.max() / grid.size >= 0.7 and len(counts) >= 2)
                near_unique += int(grid.size <= 10 and len(counts) >= grid.size - 1)
                large += int(h >= 13 and w >= 13)
                whole_symmetric += int(not _asymmetric(grid))
                rows = len({tuple(row) for row in grid.tolist()}) < h
                cols = len({tuple(col) for col in grid.T.tolist()}) < w
                palindrome_row = any(np.array_equal(row, row[::-1]) for row in grid)
                palindrome_col = any(np.array_equal(col, col[::-1]) for col in grid.T)
                mark(f"{split}_source_scale", grid.shape == (3, 3))
                mark(f"{split}_source_color_scale", grid.shape == (3, 3) and color_count in (3, 4))
                mark(f"{split}_rectangle", h != w)
                mark(f"{split}_odd_side", h % 2 == 1 or w % 2 == 1)
                mark(f"{split}_even_sides", h % 2 == 0 and w % 2 == 0)
                mark(f"{split}_large", h >= 13 and w >= 13)
                mark(f"{split}_max_side", h == 15 or w == 15)
                mark(f"{split}_one_color", color_count == 1)
                mark(f"{split}_two_colors", color_count == 2)
                mark(f"{split}_dense_palette", color_count >= 8)
                mark(f"{split}_repeated_rows", rows)
                mark(f"{split}_repeated_columns", cols)
                mark(f"{split}_palindrome_row", palindrome_row)
                mark(f"{split}_palindrome_col", palindrome_col)
                mark(f"{split}_clear_local_pattern",
                     not (rows or cols or palindrome_row or palindrome_col))
                mark(f"{split}_left_right_symmetric", np.array_equal(grid, grid[:, ::-1]))
                mark(f"{split}_top_bottom_symmetric", np.array_equal(grid, grid[::-1, :]))
                mark(f"{split}_rotation_symmetric", np.array_equal(grid, grid[::-1, ::-1]))
                mark(f"{split}_uniform_edge", any(np.unique(edge).size == 1 for edge in
                                                    (grid[0], grid[-1], grid[:, 0], grid[:, -1])))
                mark(f"{split}_row_stripes", color_count >= 2 and
                     all(np.unique(grid[r]).size == 1 for r in range(h)))
                mark(f"{split}_column_stripes", color_count >= 2 and
                     all(np.unique(grid[:, c]).size == 1 for c in range(w)))
                parity = np.fromfunction(lambda r, c: (r + c) % 2, grid.shape, dtype=int)
                mark(f"{split}_checkerboard", color_count == 2 and
                     np.array_equal(grid, np.where(parity == 0, grid[0, 0], grid[0, 1])))
                mark(f"{split}_internal_block", any(
                     np.unique(grid[r:r+2, c:c+2]).size == 1
                     for r in range(1, h - 2) for c in range(1, w - 2)))
                mark(f"{split}_dominant", color_count >= 2 and counts.max() / grid.size >= 0.7)
                mark(f"{split}_near_unique", grid.size <= 10 and color_count >= grid.size - 1)
                mark(f"{split}_mixed_counts", counts.min() == 1 and counts.max() >= 2)
                mark(f"{split}_rect_symmetric", h != w and not _asymmetric(grid))
                if split == "train":
                    train_redundant += int(rows or cols or palindrome_row or palindrome_col)
                    mark("train_source_redundant", grid.shape == (3, 3) and
                         (rows or cols or palindrome_row or palindrome_col))
                else:
                    test_asymmetric += int(_asymmetric(grid))
                    mark("test_new_shape", grid.shape not in train_shapes)
                    mark("test_larger", grid.size > max(train_areas))
                    mark("test_smaller", grid.size < min(train_areas))
    assert len(signatures) == 42
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert (3, 3) in shapes and (2, 4) in shapes and (4, 2) in shapes
    assert len(shapes) >= 30 and any(h != w for h, w in shapes)
    assert {1, 2, 3, 4}.issubset(palettes) and max(palettes) >= 8
    assert len(structures) >= 120
    assert min(train_redundant, test_asymmetric, whole_symmetric,
               large, dominant, near_unique) > 0
    assert coverage["within_puzzle_shapes"] == 42
    assert coverage["within_puzzle_palettes"] >= 30
    for split in ("train", "test"):
        for axis in ("source_scale", "source_color_scale", "rectangle", "odd_side",
                     "even_sides", "large",
                     "max_side", "one_color", "two_colors", "dense_palette",
                     "repeated_rows", "repeated_columns", "palindrome_row",
                     "palindrome_col", "clear_local_pattern", "left_right_symmetric",
                     "top_bottom_symmetric", "rotation_symmetric", "uniform_edge",
                     "row_stripes", "column_stripes", "checkerboard",
                     "internal_block", "dominant", "near_unique", "mixed_counts",
                     "rect_symmetric"):
            assert coverage.get(f"{split}_{axis}", 0) > 0, (split, axis, coverage)
    for axis in ("train_source_redundant", "test_new_shape", "test_larger", "test_smaller"):
        assert coverage.get(axis, 0) > 0, (axis, coverage)
