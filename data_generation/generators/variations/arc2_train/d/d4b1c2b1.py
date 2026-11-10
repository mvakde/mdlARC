"""Palette-cardinality enlargement puzzles."""

import numpy as np


# These are the original inputs. The output is determined uniquely by input.
_SOURCE_INPUTS = {
    (4, 4, 7, 8, 7, 7, 8, 8, 4),
    (8, 8, 8, 8, 8, 8, 8, 8, 8),
    (3, 3, 3, 3, 3, 3, 3, 3, 3),
    (4, 2, 8, 2, 2, 5, 8, 5, 4),
    (2, 2, 4, 4, 4, 4, 2, 4, 2),
    (1, 1, 1, 6, 6, 6, 6, 1, 6),
    (3, 6, 6, 3, 6, 6, 3, 3, 3),
    (7, 1, 7, 3, 3, 6, 8, 8, 6),
}
_SHAPES = ((3, 3), (2, 3), (3, 2), (2, 4), (4, 2),
           (2, 5), (5, 2), (3, 4), (4, 3), (4, 4))


def _count(name, value, lower, upper):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}]")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _palette(grid):
    return tuple(sorted(int(x) for x in np.unique(grid)))


def _source_copy(grid):
    return grid.shape == (3, 3) and tuple(int(x) for x in grid.flat) in _SOURCE_INPUTS


def _enlarge(grid):
    factor = len(np.unique(grid))
    return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1).astype(np.int8)


def _valid_shapes(factor, generalized):
    candidates = _SHAPES if generalized else ((3, 3),)
    return [shape for shape in candidates
            if factor <= shape[0] * shape[1] and max(shape) * factor <= 30]


def _layout(rng, shape, colors):
    """Make different arrangements, always displaying every selected color."""
    height, width = shape
    size = height * width
    count = len(colors)
    motif = int(rng.integers(0, 5))
    order = rng.permutation(size)
    labels = np.empty(size, dtype=np.int64)
    if motif == 0:  # scattered, with variable color frequencies
        labels[order[:count]] = np.arange(count)
        if size > count:
            weights = rng.dirichlet(np.ones(count) * float(rng.choice([0.4, 1.0, 3.0])))
            labels[order[count:]] = rng.choice(count, size=size - count, p=weights)
    elif motif == 1:  # common color and placed accents
        labels.fill(0)
        labels[order[:count - 1]] = np.arange(1, count)
        if size > count + 1 and count > 1:
            extras = int(rng.integers(0, size - count))
            labels[order[count - 1:count - 1 + extras]] = rng.integers(1, count, size=extras)
    elif motif == 2:  # compact nearest-seed regions
        seeds = order[:count]
        rows, cols = np.indices(shape)
        distances = (abs(rows[..., None] - seeds[None, None, :] // width)
                     + abs(cols[..., None] - seeds[None, None, :] % width))
        labels = np.argmin(distances, axis=2).ravel()
    elif motif == 3:  # cyclic traversal in a randomly oriented grid
        labels = (np.arange(size) + int(rng.integers(0, count))) % count
        labels = labels.reshape(shape)
        if rng.integers(2):
            labels = np.flip(labels, axis=0)
        if rng.integers(2):
            labels = np.flip(labels, axis=1)
        if shape[0] == shape[1] and rng.integers(2):
            labels = labels.T
        return np.asarray(colors[labels], dtype=np.int8).copy()
    else:  # row or column bands, with missing colors inserted
        horizontal = bool(rng.integers(2))
        axis = np.indices(shape)[0 if horizontal else 1]
        labels = (axis + int(rng.integers(0, count))) % count
        labels = labels.ravel()
        missing = [i for i in range(count) if i not in labels]
        labels[order[:len(missing)]] = missing
    return np.asarray(colors[labels.reshape(shape)], dtype=np.int8).copy()


def _fresh_grid(rng, factor, shape, used, reuse=()):
    """Bounded sampling with a deterministic color fallback."""
    for attempt in range(256):
        if reuse and attempt < 128 and rng.integers(3) == 0:
            colors = np.asarray(reuse[int(rng.integers(len(reuse)))], dtype=np.int8)
        else:
            colors = rng.choice(10, size=factor, replace=False).astype(np.int8)
        grid = _layout(rng, shape, colors)
        if len(np.unique(grid)) == factor and _key(grid) not in used and not _source_copy(grid):
            return grid
    for shift in range(10):
        colors = np.asarray([(shift + i) % 10 for i in range(factor)], dtype=np.int8)
        for phase in range(shape[0] * shape[1]):
            labels = (np.arange(shape[0] * shape[1]) + phase) % factor
            grid = colors[labels.reshape(shape)]
            if _key(grid) not in used and not _source_copy(grid):
                return grid.copy()
    raise RuntimeError("cannot make a distinct input")


def _factor_plan(rng, train_count, test_count, generalized):
    # Source-like, practiced-factor, smaller-factor, and larger-factor cases.
    profile = int(rng.integers(0, 4))
    ceiling = 7 if generalized else 9
    if profile == 0:
        low = 3
    elif profile == 3:
        low = int(rng.integers(3, 6 if generalized else 8))
    elif profile == 2:
        low = int(rng.integers(2, 6 if generalized else 8))
    else:
        low = int(rng.integers(2 if train_count >= 3 else 1,
                               6 if generalized else 8))
    high = low + 1
    if profile == 0:
        primary = 5
    elif profile == 1:
        primary = int(rng.choice([low, high]))
    elif profile == 2:
        primary = low - 1
    else:
        primary = high + 1
    train = [low, high]
    if train_count >= 3:
        train.append(low)  # same palette, different arrangement
    while len(train) < train_count:
        upper = 4 if profile == 0 else high
        minimum = 2 if train.count(1) >= 7 else 1
        train.append(int(rng.integers(minimum, upper + 1)))
    test = [primary]
    pool = [f for f in range(1, (5 if profile == 0 else ceiling) + 1)
            if f < 8 or high >= 6]
    pool.remove(primary)
    rng.shuffle(pool)
    if profile != 0 and test_count >= 2:
        practiced = [f for f in (low, high) if f in pool]
        chosen = int(rng.choice(practiced))
        test.append(chosen)
        pool.remove(chosen)
    test.extend(pool[:test_count - len(test)])
    return train, test


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    generalized = bool(rng.integers(0, 2))
    train_factors, test_factors = _factor_plan(rng, train_count, test_count, generalized)
    high = train_factors[1]
    shape_options = [s for s in _valid_shapes(high, generalized) if s != (3, 3)]
    contrast_shape = shape_options[int(rng.integers(len(shape_options)))] if generalized else (3, 3)
    used = set()
    train = []
    test = []

    # A one-cell recoloring introduces one new distinct color.
    low = train_factors[0]
    for _ in range(256):
        first = _fresh_grid(rng, low, contrast_shape, used)
        values, counts = np.unique(first, return_counts=True)
        repeated = values[counts >= 2]
        positions = np.argwhere(np.isin(first, repeated))
        row, col = positions[int(rng.integers(len(positions)))]
        novel = int(rng.choice([c for c in range(10) if c not in values]))
        second = first.copy()
        second[row, col] = novel
        if not _source_copy(second) and _key(second) not in used:
            break
    else:
        raise RuntimeError("cannot construct the one-cell contrast")
    for grid in (first, second):
        used.add(_key(grid))
        train.append({"input": grid, "output": _enlarge(grid)})

    if train_count >= 3:
        palette = np.asarray(_palette(first), dtype=np.int8)
        for _ in range(256):
            third = _layout(rng, contrast_shape, palette)
            difference = int(np.count_nonzero(third != first))
            if (len(np.unique(third)) == low and difference >= 2
                    and _key(third) not in used and not _source_copy(third)):
                break
        else:
            third = None
            for a in range(first.size):
                for b in range(a + 1, first.size):
                    if first.flat[a] == first.flat[b]:
                        continue
                    candidate = first.copy()
                    candidate.flat[a], candidate.flat[b] = first.flat[b], first.flat[a]
                    if _key(candidate) not in used and not _source_copy(candidate):
                        third = candidate
                        break
                if third is not None:
                    break
            if third is None:
                raise RuntimeError("cannot rearrange the palette")
        used.add(_key(third))
        train.append({"input": third, "output": _enlarge(third)})

    for index in range(len(train), train_count):
        factor = train_factors[index]
        options = _valid_shapes(factor, generalized)
        if generalized and index == 3 and (3, 3) in options:
            shape = (3, 3)
        else:
            shape = options[int(rng.integers(len(options)))]
        palettes = [_palette(pair["input"]) for pair in train
                    if len(_palette(pair["input"])) == factor]
        grid = _fresh_grid(rng, factor, shape, used, palettes)
        used.add(_key(grid))
        train.append({"input": grid, "output": _enlarge(grid)})

    for index, factor in enumerate(test_factors):
        options = _valid_shapes(factor, generalized)
        if generalized and index == 0:
            changed = [s for s in options if s != (3, 3)]
            options = changed or options
        shape = options[int(rng.integers(len(options)))]
        palettes = [_palette(pair["input"]) for pair in train
                    if len(_palette(pair["input"])) == factor]
        grid = _fresh_grid(rng, factor, shape, used, palettes)
        used.add(_key(grid))
        test.append({"input": grid, "output": _enlarge(grid)})

    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert int(grid.min()) >= 0 and int(grid.max()) <= 9


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen = set()
    factors = {"train": [], "test": []}
    for name, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _check_grid(inp)
            _check_grid(out)
            factor = len(np.unique(inp))
            assert 1 <= factor <= 9 and factor <= inp.size
            assert max(inp.shape) * factor <= 30
            assert out.shape == tuple(side * factor for side in inp.shape)
            assert np.array_equal(out, _enlarge(inp))
            assert _palette(inp) == _palette(out)
            key = _key(inp)
            assert key not in seen and not _source_copy(inp)
            seen.add(key)
            factors[name].append(factor)

    # The causal contrast is observable without generation metadata.
    first, second = train[0]["input"], train[1]["input"]
    assert first.shape == second.shape
    assert np.count_nonzero(first != second) == 1
    assert len(np.unique(second)) == len(np.unique(first)) + 1
    assert len(set(factors["train"])) >= 2
    if len(train) >= 3:
        third = train[2]["input"]
        assert third.shape == first.shape and _palette(third) == _palette(first)
        assert np.count_nonzero(third != first) >= 2
    if len(test) >= 2:
        assert len(set(factors["test"])) == len(test)
    if any(f >= 8 for f in factors["test"]):
        assert max(factors["train"]) >= 6
    if any(pair["input"].shape != (3, 3) for pair in test):
        assert any(pair["input"].shape != (3, 3) for pair in train)


def _equal(first, second):
    return all(len(first[name]) == len(second[name]) and all(
        np.array_equal(a["input"], b["input"])
        and np.array_equal(a["output"], b["output"])
        for a, b in zip(first[name], second[name]))
        for name in ("train", "test"))


def _normalized(grid):
    labels = {}
    return (grid.shape, tuple(labels.setdefault(int(x), len(labels)) for x in grid.flat))


def self_test() -> None:
    for seed, counts in ((20260827, (None, None)), (314159, (7, 3)),
                         (10000, (2, 1)), (10001, (10, 1)), (10002, (10, 4)),
                         (11000, (None, 1)), (11001, (None, 4)),
                         (11002, (2, None)), (11003, (10, None))):
        a = generate_puzzle(seed, *counts)
        b = generate_puzzle(seed, *counts)
        assert _equal(a, b)
        validate_generated_puzzle(a)
        if counts[0] is not None:
            assert len(a["train"]) == counts[0]
        if counts[1] is not None:
            assert len(a["test"]) == counts[1]
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4), (2.5, 1), (True, 1)):
        try:
            generate_puzzle(12000, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")

    train_counts, test_counts = set(), set()
    signatures, layouts, shapes, factors = set(), set(), set(), set()
    test_factors, colors, joint = set(), set(), set()
    contrasts = repeated_palettes = varied_tests = shape_seeds = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(tuple(_key(p["input"]) for part in puzzle.values() for p in part))
        contrasts += int(np.count_nonzero(puzzle["train"][0]["input"] != puzzle["train"][1]["input"]) == 1)
        repeated_palettes += int(len(puzzle["train"]) >= 3)
        varied_tests += int(len({len(np.unique(p["input"])) for p in puzzle["test"]}) >= 2)
        shape_seeds += int(any(p["input"].shape != (3, 3) for p in puzzle["test"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                factor = len(np.unique(inp))
                assert np.array_equal(out, np.kron(inp, np.ones((factor, factor), dtype=np.int8)))
                layouts.add(_normalized(inp))
                shapes.add(inp.shape)
                factors.add(factor)
                colors.update(int(x) for x in np.unique(inp))
                joint.add((inp.shape, factor))
                if split == "test":
                    test_factors.add(factor)
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) >= 40 and len(layouts) >= 100
    assert len(shapes) >= 7 and factors == set(range(1, 10))
    assert set(range(1, 10)).issubset(test_factors)
    assert colors == set(range(10)) and len(joint) >= 20
    assert contrasts == 42 and repeated_palettes >= 20
    assert varied_tests >= 15 and 8 <= shape_seeds <= 34
