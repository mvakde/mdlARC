"""Concentric three-color ARC puzzles with a one-step radial color cycle."""

import numpy as np


_SOURCE_SPECS = (
    ((8, 8), (8, 0, 5)),
    ((6, 6), (0, 7, 6)),
    ((6, 6), (3, 2, 0)),
    ((8, 8), (3, 7, 6)),
    ((6, 6), (9, 0, 1)),
)


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _pair_key(pair):
    return (_grid_key(pair["input"]), _grid_key(pair["output"]))


def _depths(shape):
    height, width = shape
    rows = np.minimum(np.arange(height), height - 1 - np.arange(height))[:, None]
    columns = np.minimum(np.arange(width), width - 1 - np.arange(width))[None, :]
    return np.minimum(rows, columns)


def _make_grid(shape, roles):
    return np.take(np.asarray(roles, dtype=np.int8), _depths(shape) % 3)


def _make_pair(shape, roles):
    return {
        "input": _make_grid(shape, roles),
        "output": _make_grid(shape, (roles[2], roles[0], roles[1])),
    }


_SOURCE_KEYS = frozenset(_pair_key(_make_pair(shape, roles)) for shape, roles in _SOURCE_SPECS)


def _validate_count(name, value, low, high):
    if value is None:
        return
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")


def _counts(rng, num_train, num_test):
    _validate_count("num_train", num_train, 2, 10)
    _validate_count("num_test", num_test, 1, 4)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_test = int(num_test)
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_train = int(num_train)
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    else:
        num_train, num_test = int(num_train), int(num_test)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _choice(rng, items):
    return items[int(rng.integers(len(items)))]


def _random_shape(rng, *, min_side=5, max_side=22, rectangular=None):
    if rectangular is None:
        rectangular = bool(rng.integers(0, 3) == 0)
    if rectangular:
        short = int(rng.integers(min_side, min(max_side, 20) + 1))
        long = int(rng.integers(short + 2, min(30, short + 10) + 1))
        return (short, long) if bool(rng.integers(0, 2)) else (long, short)
    side = int(rng.integers(min_side, max_side + 1))
    return (side, side)


def _train_shapes(rng, count):
    # A fourth depth witnesses the repeating cycle whenever a later test is deeper.
    first = int(rng.integers(6, 17))
    shapes = [(first, first)]
    if first == 6:
        second = _random_shape(rng, min_side=7, max_side=20)
    elif rng.random() < 0.28:
        second = shapes[0]  # Repeated dimensions with different color roles.
    else:
        second = _random_shape(rng, min_side=5, max_side=20)
    shapes.append(second)
    for _ in range(2, count):
        if rng.random() < 0.20:
            shapes.append(_choice(rng, shapes))
        else:
            shapes.append(_random_shape(rng))
    return shapes


def _test_shapes(rng, train_shapes, count):
    train_set = set(train_shapes)
    depths = [(min(h, w) + 1) // 2 for h, w in train_shapes]
    has_odd_square = any(h == w and h % 2 for h, w in train_shapes)
    rect_min_parities = {min(h, w) % 2 for h, w in train_shapes if h != w}
    squares = [(s, s) for s in range(5, 31) if s % 2 == 0 or has_odd_square]
    rectangles = [
        (h, w) for h in range(5, 31) for w in range(5, 31)
        if 2 <= abs(h - w) <= 10 and min(h, w) % 2 in rect_min_parities
    ]
    candidates = squares + rectangles
    shapes = []
    for _ in range(count):
        mode = int(rng.choice(4, p=(0.27, 0.22, 0.32, 0.19)))
        if mode == 0:
            pool = list(train_set)
        elif mode == 1:
            pool = [s for s in candidates if (min(s) + 1) // 2 < min(depths)]
        elif mode == 2:
            pool = [s for s in candidates if (min(s) + 1) // 2 > max(depths)]
        else:
            pool = [s for s in candidates if s not in train_set]
        if not pool:
            pool = [s for s in candidates if s not in train_set]
        if not pool:
            pool = candidates
        # Repeated test shapes remain possible; unique inputs are enforced separately.
        shapes.append(_choice(rng, pool))
    if len(set(train_shapes + shapes)) == 1:
        alternatives = [s for s in candidates if s not in train_set]
        shapes[0] = _choice(rng, alternatives)
    return shapes


def _palette(rng, reference=None, mode=None):
    if reference is None:
        return tuple(sorted(int(x) for x in rng.choice(10, size=3, replace=False)))
    if mode is None:
        mode = int(rng.choice(4, p=(0.30, 0.20, 0.28, 0.22)))
    # mode 0: identical set; 1/2: that many shared colors; 3: disjoint.
    shared = (3, 1, 2, 0)[mode]
    kept = tuple(int(x) for x in rng.choice(reference, size=shared, replace=False)) if shared else ()
    outside = [x for x in range(10) if x not in reference]
    fresh = tuple(int(x) for x in rng.choice(outside, size=3 - shared, replace=False))
    return tuple(sorted(kept + fresh))


def _orders(palette):
    a, b, c = palette
    return ((a, b, c), (a, c, b), (b, a, c), (b, c, a), (c, a, b), (c, b, a))


def _unique_pair(rng, shape, palette, used_inputs, preferred=None):
    # Exhaust all six role orders, then other palettes. The fallback is finite.
    palettes = [palette]
    palettes.extend((a, b, c) for a in range(10) for b in range(a + 1, 10)
                    for c in range(b + 1, 10) if (a, b, c) != palette)
    for candidate_palette in palettes:
        orders = list(_orders(candidate_palette))
        rng.shuffle(orders)
        if candidate_palette == palette and preferred is not None:
            orders.remove(preferred)
            orders.insert(0, preferred)
        for roles in orders:
            pair = _make_pair(shape, roles)
            key = _grid_key(pair["input"])
            if key not in used_inputs and _pair_key(pair) not in _SOURCE_KEYS:
                used_inputs.add(key)
                return pair
    raise AssertionError("no unique pair available")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _counts(rng, num_train, num_test)
    train_shapes = _train_shapes(rng, train_count)
    test_shapes = _test_shapes(rng, train_shapes, test_count)
    used_inputs = set()
    train = []
    test = []
    palettes = []
    for i, shape in enumerate(train_shapes):
        if i == 0:
            palette = _palette(rng)
        else:
            reference = palettes[int(rng.integers(0, len(palettes)))]
            palette = _palette(rng, reference)
        preferred = None
        if i == 1 and palette == palettes[0]:
            old = tuple(int(train[0]["input"][d, d]) for d in range(3))
            preferred = (old[1], old[0], old[2])
        pair = _unique_pair(rng, shape, palette, used_inputs, preferred)
        train.append(pair)
        palettes.append(tuple(sorted(set(int(x) for x in pair["input"].flat))))
    for shape in test_shapes:
        reference = _choice(rng, palettes)
        palette = _palette(rng, reference)
        test.append(_unique_pair(rng, shape, palette, used_inputs))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= dimension <= 30 for dimension in grid.shape)
    assert int(grid.min()) >= 0 and int(grid.max()) <= 9


def _rings(grid):
    depth = _depths(grid.shape)
    result = []
    for level in range(int(depth.max()) + 1):
        values = grid[depth == level]
        assert values.size and np.all(values == values[0])
        result.append(int(values[0]))
    return tuple(result)


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    _check_grid(inp)
    _check_grid(out)
    assert inp.shape == out.shape and min(inp.shape) >= 5
    inside, outside = _rings(inp), _rings(out)
    assert len(inside) >= 3
    roles = inside[:3]
    assert len(set(roles)) == 3
    assert inside == tuple(roles[d % 3] for d in range(len(inside)))
    assert outside == tuple(roles[(d - 1) % 3] for d in range(len(outside)))
    assert set(int(x) for x in np.unique(inp)) == set(roles)
    assert set(int(x) for x in np.unique(out)) == set(roles)
    assert not np.array_equal(inp, out)
    return roles


def _predict_from_train(train, test_input):
    # Infer direction only from demonstrations, then read test roles from its input.
    directions = set()
    for pair in train:
        inp, out = _rings(pair["input"]), _rings(pair["output"])
        first = inp[:3]
        directions.add(tuple(shift for shift in range(3)
                             if all(out[d] == first[(d + shift) % 3]
                                    for d in range(len(inp)))))
    assert directions == {(2,)}
    roles = _rings(test_input)[:3]
    assert len(set(roles)) == 3
    return _make_grid(test_input.shape, (roles[2], roles[0], roles[1]))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    inputs, pairs = set(), set()
    for pair in train + test:
        _check_pair(pair)
        input_key, pair_key = _grid_key(pair["input"]), _pair_key(pair)
        assert input_key not in inputs and pair_key not in pairs
        assert pair_key not in _SOURCE_KEYS
        inputs.add(input_key)
        pairs.add(pair_key)
    train_shapes = [pair["input"].shape for pair in train]
    assert len(set(train_shapes + [pair["input"].shape for pair in test])) >= 2
    assert any((min(shape) + 1) // 2 >= 4 for shape in train_shapes)
    odd_square = any(h == w and h % 2 for h, w in train_shapes)
    rect_parities = {min(h, w) % 2 for h, w in train_shapes if h != w}
    for pair in test:
        h, w = pair["input"].shape
        if h == w and h % 2:
            assert odd_square
        if h != w:
            assert min(h, w) % 2 in rect_parities
        assert np.array_equal(_predict_from_train(train, pair["input"]), pair["output"])


def _puzzle_key(puzzle):
    return tuple(tuple(_pair_key(pair) for pair in puzzle[split]) for split in ("train", "test"))


def _expect_invalid(**kwargs):
    try:
        generate_puzzle(101, **kwargs)
    except ValueError:
        return
    raise AssertionError(f"expected ValueError: {kwargs}")


def self_test() -> None:
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (7, 3)):
        puzzle = generate_puzzle(100 + train_count + test_count, train_count, test_count)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (train_count, test_count)
        validate_generated_puzzle(puzzle)
    for requested_train, requested_test in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(203, requested_train, requested_test)
        if requested_train is not None:
            assert len(puzzle["train"]) == requested_train
        if requested_test is not None:
            assert len(puzzle["test"]) == requested_test
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": True}):
        _expect_invalid(**kwargs)
    assert _puzzle_key(generate_puzzle(902)) == _puzzle_key(generate_puzzle(902))
    assert _puzzle_key(generate_puzzle(902, 7, 3)) == _puzzle_key(generate_puzzle(902, 7, 3))
    assert _puzzle_key(generate_puzzle(902)) != _puzzle_key(generate_puzzle(903))
    corpus = [generate_puzzle(seed) for seed in range(42)]
    train_counts, test_counts, shapes, palettes, cycles = set(), set(), set(), set(), set()
    repeated_train_shape = odd_train_test = rectangle_train_test = 0
    familiar_test = deeper_test = shallower_test = 0
    same_palette_swap = partial_overlap = disjoint_palette = 0
    for puzzle in corpus:
        validate_generated_puzzle(puzzle)
        train, test = puzzle["train"], puzzle["test"]
        train_counts.add(len(train))
        test_counts.add(len(test))
        tr_shapes = [p["input"].shape for p in train]
        tr_depths = [(min(s) + 1) // 2 for s in tr_shapes]
        repeated_train_shape += len(set(tr_shapes)) < len(tr_shapes)
        odd_train_test += any(s[0] == s[1] and s[0] % 2 for s in tr_shapes) and any(
            p["input"].shape[0] == p["input"].shape[1] and p["input"].shape[0] % 2 for p in test)
        rectangle_train_test += any(h != w for h, w in tr_shapes) and any(
            p["input"].shape[0] != p["input"].shape[1] for p in test)
        for pair in train + test:
            _check_pair(pair)
            shape = pair["input"].shape
            shapes.add(shape)
            palette = frozenset(int(x) for x in np.unique(pair["input"]))
            palettes.add(palette)
            cycles.add(_rings(pair["input"])[:3])
        for pair in test:
            shape = pair["input"].shape
            depth = (min(shape) + 1) // 2
            familiar_test += shape in tr_shapes
            deeper_test += depth > max(tr_depths)
            shallower_test += depth < min(tr_depths)
            assert np.array_equal(_predict_from_train(train, pair["input"]), pair["output"])
        tr_palettes = [frozenset(int(x) for x in np.unique(p["input"])) for p in train]
        tr_roles = [_rings(p["input"])[:3] for p in train]
        for i in range(len(train)):
            for j in range(i + 1, len(train)):
                intersection = len(tr_palettes[i] & tr_palettes[j])
                partial_overlap += intersection in (1, 2)
                disjoint_palette += intersection == 0
                same_palette_swap += intersection == 3 and tr_roles[i] != tr_roles[j]
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 45 and len(palettes) >= 40 and len(cycles) >= 80
    assert repeated_train_shape >= 5 and odd_train_test >= 3 and rectangle_train_test >= 3
    assert familiar_test >= 3 and deeper_test >= 3 and shallower_test >= 3
    assert same_palette_swap >= 3 and partial_overlap >= 10 and disjoint_palette >= 3
