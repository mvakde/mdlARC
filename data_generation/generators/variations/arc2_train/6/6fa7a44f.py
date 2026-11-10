"""Whole-grid vertical reflection puzzles."""

import numpy as np


_SOURCE_INPUTS = (
    np.array([[7, 7, 7], [9, 5, 5], [5, 1, 7]], dtype=np.int8),
    np.array([[4, 8, 4], [7, 6, 7], [8, 7, 8]], dtype=np.int8),
    np.array([[2, 6, 9], [2, 6, 9], [2, 9, 2]], dtype=np.int8),
    np.array([[9, 1, 4], [9, 1, 4], [2, 1, 1]], dtype=np.int8),
    np.array([[2, 9, 2], [8, 5, 2], [2, 2, 8]], dtype=np.int8),
)


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _checked_count("num_train", num_train, 2, 10)
    test = _checked_count("num_test", num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
        return train, test
    if test is None and train is None:
        test = int(rng.choice([1, 2, 3, 4], p=[0.55, 0.20, 0.15, 0.10]))
    if train is None:
        choices = [n for n in range(2, 11) if n > test]
        weights = np.array([1.2, 4.0, 5.0, 2.3, 1.2, 0.9, 0.7, 0.5, 0.5])
        weights = weights[np.array(choices) - 2]
        train = int(rng.choice(choices, p=weights / weights.sum()))
    if test is None:
        choices = [n for n in range(1, min(4, train - 1) + 1)]
        weights = np.array([0.55, 0.20, 0.15, 0.10])[:len(choices)]
        test = int(rng.choice(choices, p=weights / weights.sum()))
    return train, test


def _draw_shape(rng, split, profile, first=False):
    if profile == "source_size":
        return (3, 3)
    if first:
        if profile == "compact":
            return (int(rng.integers(3, 7)), int(rng.integers(2, 8)))
        return (int(rng.integers(3, 10)), int(rng.integers(2, 13)))
    if profile == "compact":
        classes = ["compact", "line", "wide", "tall"]
        weights = [0.78, 0.10, 0.07, 0.05] if split == "train" else [0.62, 0.11, 0.14, 0.13]
    else:
        classes = ["compact", "line", "wide", "tall", "large"]
        weights = [0.49, 0.08, 0.18, 0.17, 0.08] if split == "train" else [0.31, 0.10, 0.22, 0.22, 0.15]
    category = str(rng.choice(classes, p=weights))
    if category == "compact":
        return (int(rng.integers(2, 7)), int(rng.integers(2, 9)))
    if category == "line":
        if rng.random() < 0.5:
            return (1, int(rng.integers(2, 13)))
        return (int(rng.integers(2, 13)), 1)
    if category == "wide":
        return (int(rng.integers(2, 7)), int(rng.integers(9, 21 if split == "train" else 31)))
    if category == "tall":
        return (int(rng.integers(7, 13 if split == "train" else 16)), int(rng.integers(2, 9)))
    return (int(rng.integers(7, 13 if split == "train" else 16)),
            int(rng.integers(9, 21 if split == "train" else 31)))


def _test_shape(rng, profile, train_shapes):
    if profile == "source_size":
        return (3, 3)
    choice = rng.random()
    if choice < 0.31:
        return train_shapes[int(rng.integers(len(train_shapes)))]
    if choice < 0.49:
        h, w = train_shapes[int(rng.integers(len(train_shapes)))]
        if h > 1 and w > 1:
            if rng.random() < 0.5:
                return (int(rng.integers(1, h)), w)
            return (h, int(rng.integers(1, w)))
        if h > 2:
            return (int(rng.integers(2, h)), 1)
        if w > 2:
            return (1, int(rng.integers(2, w)))
    return _draw_shape(rng, "test", profile)


def _choose_palette(rng, area, mode, clear):
    if mode == "solid":
        count = 1
    elif clear:
        count = int(rng.integers(3, min(6, area) + 1))
    else:
        choices = np.arange(2, min(8, area) + 1)
        weights = np.array([4, 4, 3, 2, 1.5, 1, 0.5])[:len(choices)]
        count = int(rng.choice(choices, p=weights / weights.sum()))
    return np.asarray(rng.choice(10, size=count, replace=False), dtype=np.int8)


def _draw_grid(rng, shape, mode, clear=False):
    h, w = shape
    area = h * w
    palette = _choose_palette(rng, area, mode, clear)
    k = len(palette)
    if mode == "solid":
        grid = np.full(shape, palette[0], dtype=np.int8)
    elif mode in ("sparse", "near_solid"):
        grid = np.full(shape, palette[0], dtype=np.int8)
        limit = max(1, area // (3 if mode == "sparse" else 10))
        n = int(rng.integers(1, limit + 1))
        positions = rng.choice(area, size=n, replace=False)
        grid.flat[positions] = rng.choice(palette[1:], size=n)
    elif mode == "horizontal":
        colors = rng.choice(palette, size=h)
        grid = np.repeat(colors[:, None], w, axis=1).astype(np.int8)
    elif mode == "vertical":
        colors = rng.choice(palette, size=w)
        grid = np.repeat(colors[None, :], h, axis=0).astype(np.int8)
    elif mode == "repeated_rows":
        n = int(rng.integers(1, min(h, 4) + 1))
        prototypes = rng.choice(palette, size=(n, w))
        indices = rng.integers(0, n, size=h)
        grid = np.asarray(prototypes[indices], dtype=np.int8)
    elif mode == "blocks":
        rh = int(rng.integers(1, min(4, h) + 1))
        cw = int(rng.integers(1, min(4, w) + 1))
        rows = np.arange(h)[:, None] // rh
        columns = np.arange(w)[None, :] // cw
        grid = palette[(rows + columns + int(rng.integers(k))) % k]
    elif mode == "palindrome":
        top = rng.choice(palette, size=((h + 1) // 2, w)).astype(np.int8)
        grid = np.concatenate((top, top[:h // 2][::-1]), axis=0)
    else:
        grid = rng.choice(palette, size=shape).astype(np.int8)
        if clear:
            positions = rng.choice(area, size=k, replace=False)
            grid.flat[positions] = palette
    return np.asarray(grid, dtype=np.int8)


def _is_clear(grid):
    rows = {tuple(int(v) for v in row) for row in grid}
    return grid.shape[0] >= 3 and len(rows) >= 3 and not np.array_equal(grid[0], grid[-1])


def _grid_key(grid):
    return (grid.shape, tuple(int(value) for value in grid.flat))


def _new_input(rng, shape, index, seen, clear=False):
    modes = ["irregular", "horizontal", "vertical", "repeated_rows", "sparse",
             "blocks", "palindrome", "solid", "near_solid"]
    weights = [0.28, 0.10, 0.09, 0.12, 0.11, 0.11, 0.10, 0.04, 0.05]
    for attempt in range(48):
        mode = "irregular" if clear or attempt >= 32 else str(rng.choice(modes, p=weights))
        grid = _draw_grid(rng, shape, mode, clear)
        key = _grid_key(grid)
        if (not clear or _is_clear(grid)) and key not in seen and not any(
            grid.shape == source.shape and np.array_equal(grid, source)
            for source in _SOURCE_INPUTS
        ):
            seen.add(key)
            return grid
    # Finite fallback for tiny grids; every sampled shape has at least two cells.
    for code in range(100):
        grid = np.full(shape, 0, dtype=np.int8)
        grid.flat[0], grid.flat[1] = code // 10, code % 10
        if clear:
            grid[0, :] = 1
            grid[grid.shape[0] // 2, :] = 2
            grid[-1, :] = 3
        key = _grid_key(grid)
        if key not in seen and not any(
            grid.shape == source.shape and np.array_equal(grid, source)
            for source in _SOURCE_INPUTS
        ):
            seen.add(key)
            return grid
    raise AssertionError(f"no unique input for pair {index}")


def _reflect(grid):
    return np.concatenate((grid, grid[::-1]), axis=0).astype(np.int8, copy=False)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # Size profile changes sampling frequencies, never the inferred operation.
    profile = str(rng.choice(["source_size", "compact", "mixed"], p=[0.22, 0.30, 0.48]))
    train_shapes = []
    for i in range(train_count):
        if i and profile != "source_size" and rng.random() < 0.28:
            shape = train_shapes[int(rng.integers(len(train_shapes)))]
        else:
            shape = _draw_shape(rng, "train", profile, first=(i == 0))
        train_shapes.append(shape)
    test_shapes = [_test_shape(rng, profile, train_shapes) for _ in range(test_count)]
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=train_count + test_count)
    seen = set()
    train, test = [], []
    for i, shape in enumerate(train_shapes + test_shapes):
        pair_rng = np.random.default_rng(int(pair_seeds[i]))
        inp = _new_input(pair_rng, shape, i, seen, clear=(i == 0))
        pair = {"input": inp, "output": _reflect(inp)}
        (train if i < train_count else test).append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _predict_from_demonstrations(train, test_input):
    assert any(_is_clear(pair["input"]) for pair in train)
    for pair in train:
        assert np.array_equal(pair["output"], _reflect(pair["input"]))
    return _reflect(test_input)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train, test = puzzle["train"], puzzle["test"]
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in (pair["input"], pair["output"]):
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= n <= 30 for n in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer)
            assert bool(np.all((0 <= grid) & (grid <= 9)))
        inp, out = pair["input"], pair["output"]
        assert inp.shape[0] <= 15 and out.shape == (inp.shape[0] * 2, inp.shape[1])
        assert out.dtype == inp.dtype and np.array_equal(out, _reflect(inp))
        assert not any(inp.shape == source.shape and np.array_equal(inp, source)
                       for source in _SOURCE_INPUTS)
        ikey, okey = _grid_key(inp), _grid_key(out)
        assert ikey not in seen_inputs and (ikey, okey) not in seen_pairs
        seen_inputs.add(ikey)
        seen_pairs.add((ikey, okey))
    assert any(_is_clear(pair["input"]) for pair in train)
    for pair in test:
        assert np.array_equal(_predict_from_demonstrations(train, pair["input"]), pair["output"])


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(pair["input"]), _grid_key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def self_test() -> None:
    source_train = [{"input": grid, "output": _reflect(grid)} for grid in _SOURCE_INPUTS[:4]]
    assert np.array_equal(_predict_from_demonstrations(source_train, _SOURCE_INPUTS[4]),
                          _reflect(_SOURCE_INPUTS[4]))
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) >= 40
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    all_inputs = []
    same_size = source_size = smaller = lines = palindromes = solids = two_colors = 0
    late_colors = test_only_colors = mixed_tests = compact_counts = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train, test = puzzle["train"], puzzle["test"]
        all_pairs = train + test
        all_inputs.extend(pair["input"] for pair in all_pairs)
        train_shapes = {p["input"].shape for p in train}
        same_size += any(p["input"].shape in train_shapes for p in test)
        source_size += all(p["input"].shape == (3, 3) for p in all_pairs)
        smaller += any(any(p["input"].shape[0] <= t[0] and p["input"].shape[1] <= t[1]
                           and p["input"].shape != t for t in train_shapes) for p in test)
        compact_counts += len(train) in (3, 4) and len(test) == 1
        first_colors = set(np.unique(train[0]["input"]).tolist())
        train_colors = set().union(*(set(np.unique(p["input"]).tolist()) for p in train))
        late_colors += any(not set(np.unique(p["input"]).tolist()) <= first_colors for p in train[1:])
        test_only_colors += any(not set(np.unique(p["input"]).tolist()) <= train_colors for p in test)
        mixed_tests += len(test) > 1 and len({p["input"].shape for p in test}) > 1
        for pair in all_pairs:
            inp, out = pair["input"], pair["output"]
            assert np.array_equal(out[:len(inp)], inp)
            assert np.array_equal(out[len(inp):], inp[::-1])
            assert np.array_equal(out[len(inp)], inp[-1])
            assert not any(inp.shape == s.shape and np.array_equal(inp, s) for s in _SOURCE_INPUTS)
            lines += 1 in inp.shape
            palindromes += np.array_equal(inp, inp[::-1])
            solids += len(np.unique(inp)) == 1
            two_colors += len(np.unique(inp)) == 2
    assert source_size >= 3 and same_size >= 8 and smaller >= 3
    assert compact_counts >= 5 and mixed_tests >= 3
    assert late_colors >= 15 and test_only_colors >= 5
    assert lines >= 3 and palindromes >= 10 and solids >= 2 and two_colors >= 15
    assert len({grid.shape for grid in all_inputs}) >= 25
    assert len({_grid_key(grid) for grid in all_inputs}) >= int(0.9 * len(all_inputs))
    for seed, counts in enumerate(((2, 1), (10, 1), (10, 4), (7, 3)), start=100):
        puzzle = generate_puzzle(seed, *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
        validate_generated_puzzle(puzzle)
    for seed, arguments in enumerate(((None, 1), (None, 4), (2, None), (10, None)), start=200):
        puzzle = generate_puzzle(seed, *arguments)
        validate_generated_puzzle(puzzle)
        assert all(want is None or got == want for want, got in
                   zip(arguments, (len(puzzle["train"]), len(puzzle["test"]))))
    for counts in ((None, None), (8, 3), (None, 4), (2, None)):
        assert _puzzle_key(generate_puzzle(987654321, *counts)) == _puzzle_key(generate_puzzle(987654321, *counts))
    assert _puzzle_key(generate_puzzle(2)) != _puzzle_key(generate_puzzle(3))
    for counts in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {counts}")
