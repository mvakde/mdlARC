import numpy as np


_SOURCE_INPUTS = (
    ((0, 0, 8), (8, 8, 8), (8, 0, 0)),
    ((1, 0), (1, 1)),
    ((0, 3), (3, 0)),
    (
        (0, 0, 2, 0),
        (2, 0, 2, 0),
        (0, 2, 0, 2),
        (2, 2, 2, 0),
    ),
    ((2, 0, 2), (0, 2, 0), (2, 2, 2)),
    (
        (0, 7, 0, 0),
        (7, 7, 7, 7),
        (0, 7, 0, 0),
        (0, 7, 0, 7),
    ),
    (
        (0, 8, 8, 0, 0),
        (8, 8, 8, 8, 8),
        (0, 8, 8, 0, 0),
        (8, 8, 8, 8, 8),
        (0, 8, 8, 0, 8),
    ),
)


def _normalize_count(
    value: int | None,
    name: str,
    low: int,
    high: int,
) -> int | None:
    if value is None:
        return None

    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer or None")

    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")

    return value


def _resolve_counts(
    seed: int,
    num_train: int | None,
    num_test: int | None,
) -> tuple[int, int]:
    num_train = _normalize_count(num_train, "num_train", 2, 10)
    num_test = _normalize_count(num_test, "num_test", 1, 4)

    if num_train is None and num_test is None:
        legal_counts = [
            (train_count, test_count)
            for test_count in range(1, 5)
            for train_count in range(max(2, test_count + 1), 11)
        ]
        return legal_counts[seed % len(legal_counts)]

    if num_train is None:
        choices = list(range(max(2, num_test + 1), 11))
        if not choices:
            raise ValueError("no legal num_train exists for num_test")
        return choices[seed % len(choices)], num_test

    if num_test is None:
        choices = list(range(1, min(4, num_train - 1) + 1))
        if not choices:
            raise ValueError("no legal num_test exists for num_train")
        return num_train, choices[seed % len(choices)]

    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")

    return num_train, num_test


def _tile_grid(grid: np.ndarray) -> np.ndarray:
    n = grid.shape[0]
    return np.tile(grid, (n, n))


def _grid_key(grid: np.ndarray) -> tuple:
    return (
        tuple(int(dimension) for dimension in grid.shape),
        tuple(int(value) for value in grid.ravel()),
    )


def _is_exact_source_pair(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
) -> bool:
    for rows in _SOURCE_INPUTS:
        source_input = np.asarray(rows, dtype=np.int8)
        source_output = _tile_grid(source_input)

        if np.array_equal(input_grid, source_input) and np.array_equal(
            output_grid, source_output
        ):
            return True

    return False


def _palette(rng, n, classic):
    # Color has no role in the rule; only the tile geometry matters.
    if classic:
        return np.asarray([0, int(rng.integers(1, 10))], dtype=np.int8)
    if rng.random() < 0.04:
        count = 1
    else:
        counts = np.arange(2, min(9, n * n) + 1)
        weights = np.asarray([0.43, 0.21, 0.14, 0.09, 0.06, 0.04, 0.02, 0.01])[:len(counts)]
        count = int(rng.choice(counts, p=weights / weights.sum()))
    return rng.choice(np.arange(10, dtype=np.int8), size=count, replace=False)


def _pattern(rng, n, colors, style):
    k = len(colors)
    if style == 10 or k == 1:
        return np.full((n, n), colors[0], dtype=np.int8)
    rows, cols = np.indices((n, n))
    if style == 0:  # Scatter with varying density.
        weights = rng.random(k) ** float(rng.uniform(0.5, 3.0)) + 0.03
        grid = rng.choice(colors, size=(n, n), p=weights / weights.sum())
    elif style == 1:  # Clean bands, without inserted pixels.
        line = rng.choice(colors, size=n)
        grid = (np.repeat(line[:, None], n, axis=1) if rng.integers(2)
                else np.repeat(line[None, :], n, axis=0))
    elif style == 2:  # Checker or a longer modular rhythm.
        a = int(rng.integers(1, k))
        b = int(rng.integers(1, k))
        grid = colors[(a * rows + b * cols + int(rng.integers(k))) % k]
    elif style == 3:  # Clean reflection.
        grid = rng.choice(colors, size=(n, n))
        if rng.integers(2):
            grid[:, n - n // 2:] = grid[:, :n // 2][:, ::-1]
        else:
            grid[n - n // 2:, :] = grid[:n // 2, :][::-1, :]
    elif style == 4:  # Overlapping rectangles.
        grid = np.full((n, n), colors[0], dtype=np.int8)
        for _ in range(int(rng.integers(1, n + 2))):
            r0, r1 = sorted(int(x) for x in rng.choice(n + 1, 2, replace=False))
            c0, c1 = sorted(int(x) for x in rng.choice(n + 1, 2, replace=False))
            grid[r0:r1, c0:c1] = colors[int(rng.integers(1, k))]
    elif style == 5:  # Diagonal strokes.
        grid = np.full((n, n), colors[0], dtype=np.int8)
        for color in colors[1:]:
            slope = int(rng.choice(np.asarray([-2, -1, 1, 2])))
            offset = int(rng.integers(n))
            grid[(rows + slope * cols + offset) % n == 0] = color
    elif style == 6:  # An asymmetric motif that meets opposite tile edges.
        grid = np.full((n, n), colors[0], dtype=np.int8)
        r = int(rng.integers(n))
        grid[r, 0] = colors[1]
        grid[r, -1] = colors[1]
        if n >= 3:
            rr = (r + int(rng.integers(1, n))) % n
            cc = int(rng.integers(1, n - 1))
            grid[rr, cc] = colors[int(rng.integers(1, k))]
            if k > 2:
                grid[rr, (cc + 1) % n] = colors[2]
    elif style == 7:  # Sparse marks.
        grid = np.full((n, n), colors[0], dtype=np.int8)
        count = int(rng.integers(1, max(2, n * n // 2 + 1)))
        for pos in rng.choice(n * n, count, replace=False):
            grid.flat[int(pos)] = colors[int(rng.integers(1, k))]
    elif style == 8:  # Quiet boundary with an interior mark.
        grid = np.full((n, n), colors[0], dtype=np.int8)
        if n >= 3:
            grid[1:-1, 1:-1] = rng.choice(colors, size=(n - 2, n - 2))
            grid[1, 1] = colors[1]
        else:
            grid[-1, -1] = colors[1]
    elif style == 9:  # Deliberately irregular bands.
        line = rng.choice(colors, size=n)
        grid = (np.repeat(line[:, None], n, axis=1) if rng.integers(2)
                else np.repeat(line[None, :], n, axis=0))
        for pos in rng.choice(n * n, int(rng.integers(1, n + 1)), replace=False):
            grid.flat[int(pos)] = colors[int(rng.integers(k))]
    else:
        raise ValueError("unknown style")
    return np.asarray(grid, dtype=np.int8)


def _style_schedule(rng, count, allow_uniform):
    styles = []
    while len(styles) < count:
        styles.extend(int(x) for x in rng.permutation(10))
    styles = styles[:count]
    if allow_uniform and count >= 3 and rng.random() < 0.25:
        styles[int(rng.integers(count))] = 10
    return styles


def _training_sizes(rng, count, include_five):
    sizes = []
    allowed = np.arange(2, 6 if include_five else 5)
    while len(sizes) < count:
        sizes.extend(int(x) for x in rng.permutation(allowed))
    return sizes[:count]


def _test_sizes(rng, count, train_sizes):
    seen = sorted(set(train_sizes))
    sizes = [5] if 5 not in seen else []
    if not sizes and count >= 2 and rng.random() < 0.65:
        sizes = [5, int(rng.choice(seen))]
    while len(sizes) < count:
        pick = rng.random()
        if pick < 0.47:
            sizes.append(5)
        elif pick < 0.79:
            sizes.append(int(rng.choice(seen)))
        else:
            sizes.append(int(rng.integers(2, 6)))
    rng.shuffle(sizes)
    return sizes


def _make_pair(rng, n, style, classic, used_inputs):
    for _ in range(120):
        colors = _palette(rng, n, classic)
        grid = _pattern(rng, n, colors, style)
        if style != 10 and len(np.unique(grid)) == 1:
            continue
        output = _tile_grid(grid)
        if _grid_key(grid) not in used_inputs and not _is_exact_source_pair(grid, output):
            used_inputs.add(_grid_key(grid))
            return {"input": grid, "output": output}
    # Bounded fallback for crowded 2x2 configurations.
    for code in range(1, 10000):
        grid = np.full((n, n), code % 10, dtype=np.int8)
        grid.flat[0] = (code // 10 + 1) % 10
        grid.flat[-1] = (code // 100 + 2) % 10
        output = _tile_grid(grid)
        if _grid_key(grid) not in used_inputs and not _is_exact_source_pair(grid, output):
            used_inputs.add(_grid_key(grid))
            return {"input": grid, "output": output}
    raise RuntimeError("could not construct a unique pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    # This is a visual preference, not a hidden semantic convention.
    classic = bool(rng.random() < 0.18)
    train_sizes = _training_sizes(rng, train_count, include_five=bool(rng.random() < 0.65))
    test_sizes = _test_sizes(rng, test_count, train_sizes)
    train_styles = _style_schedule(rng, train_count, allow_uniform=train_count >= 3)
    test_styles = _style_schedule(rng, test_count, allow_uniform=test_count >= 3)
    pair_seeds = rng.integers(0, 2**63 - 1, size=train_count + test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    for section, sizes, styles in (("train", train_sizes, train_styles),
                                   ("test", test_sizes, test_styles)):
        for n, style in zip(sizes, styles):
            index = len(puzzle["train"]) + len(puzzle["test"])
            pair_rng = np.random.default_rng(int(pair_seeds[index]))
            puzzle[section].append(_make_pair(pair_rng, n, style, classic, used_inputs))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs, train_sizes = set(), set(), set()
    nonuniform_train = 0
    for section in ("train", "test"):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            x, y = pair["input"], pair["output"]
            for grid in (x, y):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            n = x.shape[0]
            assert x.shape == (n, n) and 2 <= n <= 5
            assert y.shape == (n * n, n * n)
            assert np.array_equal(y, _tile_grid(x))
            assert set(np.unique(x)) == set(np.unique(y))
            assert not _is_exact_source_pair(x, y)
            xkey, pkey = _grid_key(x), (_grid_key(x), _grid_key(y))
            assert xkey not in inputs and pkey not in pairs
            inputs.add(xkey)
            pairs.add(pkey)
            if section == "train":
                train_sizes.add(n)
                nonuniform_train += len(np.unique(x)) > 1
    # At least two side lengths reveal that the repetition count depends on n.
    assert len(train_sizes) >= 2 and nonuniform_train >= 1


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(p["input"]), _grid_key(p["output"]))
                       for p in puzzle[section]) for section in ("train", "test"))


def _layout_key(grid):
    remap, labels = {}, []
    for value in grid.flat:
        color = int(value)
        if color not in remap:
            remap[color] = len(remap)
        labels.append(remap[color])
    return (grid.shape, tuple(labels))


def _prediction_from_demonstrations(train, test_input):
    # Infer the size rule using training pairs only, then use the test input.
    sizes = set()
    for pair in train:
        x, y = pair["input"], pair["output"]
        n = x.shape[0]
        assert y.shape == (n * n, n * n)
        assert all(y[r, c] == x[r % n, c % n]
                   for r in range(n * n) for c in range(n * n))
        sizes.add(n)
    assert len(sizes) >= 2
    n = test_input.shape[0]
    return np.asarray([[test_input[r % n, c % n]
                        for c in range(n * n)] for r in range(n * n)], dtype=np.int8)


def _observed_features(grid):
    n = grid.shape[0]
    colors, counts = np.unique(grid, return_counts=True)
    nonuniform = len(colors) > 1
    dominant = colors[np.argmax(counts)]
    boundary = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    band = nonuniform and (all(len(set(row)) == 1 for row in grid)
                           or all(len(set(col)) == 1 for col in grid.T))
    mirror = nonuniform and (np.array_equal(grid, grid[::-1])
                             or np.array_equal(grid, grid[:, ::-1]))
    checker = nonuniform and len(colors) == 2 and grid[0, 0] != grid[0, 1]
    if checker:
        checker = all(grid[r, c] == grid[0, (r + c) % 2]
                      for r in range(n) for c in range(n))
    return {
        "band": band,
        "mirror": mirror,
        "checker": checker,
        "inner_rhythm": n == 4 and nonuniform and all(
            grid[r, c] == grid[r % 2, c % 2]
            for r in range(n) for c in range(n)),
        "edge_join": n >= 3 and nonuniform and not mirror and any(
            grid[r, 0] == grid[r, -1] != dominant for r in range(n)),
        "quiet": n >= 3 and nonuniform and len(set(boundary)) == 1,
        "conspicuous": nonuniform and (np.any(grid[:, 0] != grid[:, -1])
                                       or np.any(grid[0] != grid[-1])),
        "sparse": 1 - counts.max() / (n * n) < 0.2,
        "dense": 1 - counts.max() / (n * n) > 0.45,
    }


def self_test() -> None:
    train_counts, test_counts = set(), set()
    signatures, layouts, palettes, profiles = set(), set(), set(), set()
    train_sides, test_sides, first_layouts = set(), set(), set()
    zero_free = rich = binary = uniform = 0
    train_five = seen_test = smaller_test = mixed_test = 0
    review_counts = {name: 0 for name in (
        "original_extrapolation", "novel_test_color", "classic_binary_seed",
        "first_structured", "band", "mirror", "inner_rhythm",
        "odd_checker", "edge_join", "quiet", "conspicuous",
        "rich_zero_free", "sparse", "dense", "novel_color_seen_side")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        signatures.add(_puzzle_key(puzzle))
        ts = {p["input"].shape[0] for p in puzzle["train"]}
        us = {p["input"].shape[0] for p in puzzle["test"]}
        assert len(ts) >= 2
        train_five += 5 in ts
        seen_test += bool(ts & us)
        smaller_test += any(n < max(ts) for n in us)
        mixed_test += len(us) >= 2
        review_counts["original_extrapolation"] += 5 not in ts and 5 in us
        train_colors = {int(c) for pair in puzzle["train"]
                        for c in np.unique(pair["input"])}
        test_colors = {int(c) for pair in puzzle["test"]
                       for c in np.unique(pair["input"])}
        novel_test_color = bool(test_colors - train_colors)
        review_counts["novel_test_color"] += novel_test_color
        review_counts["novel_color_seen_side"] += novel_test_color and bool(ts & us)
        review_counts["classic_binary_seed"] += all(
            len(np.unique(pair["input"])) == 2 and 0 in pair["input"]
            for section in ("train", "test") for pair in puzzle[section])
        profiles.add((tuple(p["input"].shape[0] for p in puzzle["train"]),
                      tuple(p["input"].shape[0] for p in puzzle["test"])))
        first_layouts.add(_layout_key(puzzle["train"][0]["input"]))
        observed = []
        for section in ("train", "test"):
            for pair in puzzle[section]:
                x, y = pair["input"], pair["output"]
                n = x.shape[0]
                assert x.dtype == np.int8 and y.dtype == np.int8
                assert np.array_equal(y, np.tile(x, (n, n)))
                assert not _is_exact_source_pair(x, y)
                layouts.add(_layout_key(x))
                color_set = frozenset(int(c) for c in np.unique(x))
                palettes.add(color_set)
                zero_free += 0 not in color_set
                rich += len(color_set) >= 4
                binary += len(color_set) == 2
                uniform += len(color_set) == 1
                observed.append((n, _observed_features(x), len(color_set), 0 in color_set))
                (train_sides if section == "train" else test_sides).add(n)
        first = observed[0][1]
        review_counts["first_structured"] += any(first[k] for k in
                                                  ("band", "mirror", "checker", "quiet", "edge_join"))
        for name in ("band", "mirror", "inner_rhythm", "edge_join", "quiet",
                     "conspicuous", "sparse", "dense"):
            review_counts[name] += any(features[name] for _, features, _, _ in observed)
        review_counts["odd_checker"] += any(
            n % 2 == 1 and features["checker"] for n, features, _, _ in observed)
        review_counts["rich_zero_free"] += any(
            count >= 4 and not has_zero for _, _, count, has_zero in observed)
        for pair in puzzle["test"]:
            prediction = _prediction_from_demonstrations(puzzle["train"],
                                                         pair["input"].copy())
            assert np.array_equal(prediction, pair["output"])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42 and len(layouts) >= 100
    assert len(palettes) >= 30 and len(profiles) >= 20
    assert len(first_layouts) >= 30
    assert train_sides == test_sides == {2, 3, 4, 5}
    assert train_five >= 8 and seen_test >= 8 and smaller_test >= 8 and mixed_test >= 4
    assert zero_free >= 30 and rich >= 15 and binary >= 40 and uniform >= 1
    assert review_counts["original_extrapolation"] >= 6
    assert review_counts["novel_test_color"] >= 8
    assert review_counts["classic_binary_seed"] >= 2
    assert review_counts["first_structured"] >= 12
    assert review_counts["band"] >= 12 and review_counts["mirror"] >= 12
    assert review_counts["inner_rhythm"] >= 2 and review_counts["odd_checker"] >= 3
    assert review_counts["edge_join"] >= 10 and review_counts["quiet"] >= 10
    assert review_counts["conspicuous"] >= 20
    assert review_counts["rich_zero_free"] >= 8
    assert review_counts["sparse"] >= 12 and review_counts["dense"] >= 12
    assert review_counts["novel_color_seen_side"] >= 6
    assert _puzzle_key(generate_puzzle(777)) == _puzzle_key(generate_puzzle(777))
    assert _puzzle_key(generate_puzzle(778, 7, 3)) == _puzzle_key(generate_puzzle(778, 7, 3))
    assert len({_puzzle_key(generate_puzzle(900 + i, 6, 2)) for i in range(8)}) >= 7
    for seed, a, b in ((100, 2, 1), (101, 10, 1), (102, 10, 4)):
        puzzle = generate_puzzle(seed, a, b)
        assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for seed, a, b in ((200, None, 1), (201, None, 4),
                       (202, 2, None), (203, 10, None)):
        puzzle = generate_puzzle(seed, a, b)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(500, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
