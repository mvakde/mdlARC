"""Reverse the colors of concentric one-cell frames by depth."""

import numpy as np


SOURCE_ORDERS = frozenset({
    (8, 1, 2, 4), (2, 1, 6), (7, 2, 4, 1, 3),
    (4, 2, 1, 3, 5, 8), (8, 2, 4, 3, 7, 6, 5),
})


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, rng, num_train, num_test):
    train = _count("num_train", num_train, 2, 10)
    test = _count("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _kind(grid):
    h, w = grid.shape
    return "rect" if h != w else ("odd" if h % 2 else "even")


def _limit(kind, reuse):
    return (12 if reuse else 9) if kind == "rect" else (15 if reuse else 10)


def _shape(kind, layers, rng):
    if kind == "even":
        return 2 * layers, 2 * layers
    if kind == "odd":
        return 2 * layers - 1, 2 * layers - 1
    extra = int(rng.integers(1, min(5, 16 - layers)))
    short, long = 2 * layers, 2 * (layers + extra)
    return (short, long) if rng.integers(2) else (long, short)


def _grid(shape, colors):
    h, w = shape
    rows, cols = np.indices(shape)
    depth = np.minimum.reduce((rows, cols, h - 1 - rows, w - 1 - cols))
    return np.asarray(colors, dtype=np.int8)[depth]


def _read(grid):
    layers = (min(grid.shape) + 1) // 2
    colors = tuple(int(grid[d, d]) for d in range(layers))
    assert np.array_equal(grid, _grid(grid.shape, colors))
    assert all(a != b for a, b in zip(colors, colors[1:]))
    return colors


def _colors(rng, layers, repeat):
    if not repeat or layers < 4:
        return tuple(int(c) for c in rng.choice(10, size=layers, replace=False))
    palette_size = int(rng.integers(3, min(layers - 1, 9) + 1))
    palette = tuple(int(c) for c in rng.choice(10, size=palette_size, replace=False))
    for _ in range(128):
        order = tuple(int(c) for c in rng.choice(palette, size=layers))
        if (len(set(order)) < layers and order != order[::-1]
                and all(a != b for a, b in zip(order, order[1:]))):
            return order
    return tuple(palette[i % palette_size] for i in range(layers))


def _contrast(rng, colors):
    if rng.integers(2):
        return colors[::-1]
    for _ in range(64):
        order = tuple(int(c) for c in rng.permutation(colors))
        if order != colors and order != order[::-1] and all(
            a != b for a, b in zip(order, order[1:])
        ):
            return order
    return colors[::-1]


def _pair(rng, kind, layers, repeat, seen, anchor=None):
    for _ in range(128):
        shape = anchor["input"].shape if anchor is not None else _shape(kind, layers, rng)
        colors = _contrast(rng, _read(anchor["input"])) if anchor is not None else _colors(rng, layers, repeat)
        if shape[0] == shape[1] and shape[0] % 2 == 0 and colors in SOURCE_ORDERS:
            continue
        source, target = _grid(shape, colors), _grid(shape, colors[::-1])
        signature = (shape, source.tobytes())
        if signature not in seen and not np.array_equal(source, target):
            seen.add(signature)
            return {"input": source, "output": target}
    raise RuntimeError("Could not produce a unique nontrivial pair")


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)

    # Every test capability is demonstrated in training. Geometry and color
    # reuse are the only puzzle-level choices; colors themselves are pair-local.
    reuse = bool(rng.integers(2))
    extensions = ([], ["odd"], ["rect"], ["odd", "rect"])[int(rng.integers(4))]
    extensions = extensions[:train_count - 1]
    kinds = ["even"] + extensions
    seen = set()
    train = []
    first_layers = int(rng.choice((5, 7, 9) if reuse else (3, 5, 7, 9)))
    first = _pair(rng, "even", first_layers, reuse, seen)
    train.append(first)
    for kind in extensions:
        layers = int(rng.integers(3, min(9, _limit(kind, reuse)) + 1))
        train.append(_pair(rng, kind, layers, reuse and bool(rng.integers(2)), seen))
    if len(train) < train_count and rng.random() < 0.67:
        train.append(_pair(rng, "even", first_layers, reuse, seen, first))
    while len(train) < train_count:
        kind = str(rng.choice(kinds))
        layers = int(rng.integers(2, min(10, _limit(kind, reuse)) + 1))
        repeat = reuse and layers >= 4 and rng.random() < 0.65
        train.append(_pair(rng, kind, layers, repeat, seen))

    test = []
    for index in range(test_count):
        kind = str(rng.choice(extensions)) if extensions and (index == 0 or rng.random() < 0.4) else str(rng.choice(kinds))
        anchors = [pair for pair in train if _kind(pair["input"]) == kind]
        anchor = anchors[int(rng.integers(len(anchors)))]
        anchor_layers = len(_read(anchor["input"]))
        maximum = _limit(kind, reuse)
        regime = int(rng.choice(4, p=(0.28, 0.23, 0.31, 0.18)))
        if regime == 0:
            layers = anchor_layers
        elif regime == 1:
            layers = max(2, anchor_layers - int(rng.integers(1, 4)))
        elif regime == 2:
            layers = min(maximum, anchor_layers + 1)
        else:
            layers = int(rng.integers(2, maximum + 1))
        if reuse and kind != "rect" and rng.random() < 0.22:
            layers = int(rng.integers(11, 16))
        repeat = reuse and layers >= 4 and (index == 0 or rng.random() < 0.7 or layers > 10)
        test.append(_pair(rng, kind, layers, repeat, seen))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen = set()
    features = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= side <= 30 for side in grid.shape)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            source, target = pair["input"], pair["output"]
            assert source.shape == target.shape and min(source.shape) >= 3
            assert source.shape[0] == source.shape[1] or all(side % 2 == 0 for side in source.shape)
            colors = _read(source)
            assert _read(target) == colors[::-1]
            assert not np.array_equal(source, target)
            assert not (source.shape[0] == source.shape[1] and source.shape[0] % 2 == 0 and colors in SOURCE_ORDERS)
            signature = (source.shape, source.tobytes())
            assert signature not in seen
            seen.add(signature)
            features[split].append((_kind(source), len(colors), len(set(colors)) < len(colors)))
    train_features = features["train"]
    assert any(kind == "even" for kind, _, _ in train_features)
    assert any(layers % 2 for _, layers, _ in train_features)
    for kind, _, repeated in features["test"]:
        assert any(train_kind == kind for train_kind, _, _ in train_features)
        if repeated:
            assert any(train_repeat for _, _, train_repeat in train_features)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    signatures, dimensions, kinds = set(), set(), set()
    regimes = {
        "same": 0, "smaller": 0, "larger": 0, "repeat": 0,
        "contrast": 0, "large": 0, "repeat20": 0, "rectrepeat": 0,
        "oddrepeat": 0, "two_layers": 0, "same_new_palette": 0,
        "same_parity": 0, "repeated_first_four": 0,
    }
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_shapes = [p["input"].shape for p in puzzle["train"]]
        train_layers = [len(_read(p["input"])) for p in puzzle["train"]]
        regimes["same_parity"] += train_layers[0] % 2 == train_layers[1] % 2
        if len(train_layers) >= 4:
            regimes["repeated_first_four"] += len(set(train_shapes[:4])) < 4
        for i, left in enumerate(puzzle["train"]):
            for right in puzzle["train"][i + 1:]:
                a, b = _read(left["input"]), _read(right["input"])
                if left["input"].shape == right["input"].shape and sorted(a) == sorted(b) and a != b:
                    regimes["contrast"] += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                source, target = pair["input"], pair["output"]
                colors = _read(source)
                for row in range(source.shape[0]):
                    for col in range(source.shape[1]):
                        depth = min(row, col, source.shape[0] - row - 1, source.shape[1] - col - 1)
                        assert int(target[row, col]) == colors[len(colors) - 1 - depth]
                signatures.add((source.shape, source.tobytes()))
                dimensions.add(source.shape)
                kinds.add(_kind(source))
                regimes["large"] += len(colors) >= 11
                regimes["repeat20"] += source.shape == (20, 20) and len(set(colors)) < 10
                regimes["rectrepeat"] += _kind(source) == "rect" and len(set(colors)) < len(colors)
                regimes["oddrepeat"] += _kind(source) == "odd" and len(set(colors)) < len(colors)
                regimes["two_layers"] += len(colors) == 2
                if split == "test":
                    regimes["same"] += source.shape in train_shapes
                    regimes["smaller"] += len(colors) < max(train_layers)
                    regimes["larger"] += len(colors) > max(train_layers)
                    regimes["repeat"] += len(set(colors)) < len(colors)
                    regimes["same_new_palette"] += source.shape in train_shapes and all(
                        source.shape != train_pair["input"].shape
                        or set(colors) != set(_read(train_pair["input"]))
                        for train_pair in puzzle["train"]
                    )
    assert len(signatures) >= 140 and len(dimensions) >= 20
    assert kinds == {"even", "odd", "rect"}
    assert all(value >= 3 for value in regimes.values())
    for seed, train, test in ((1000, 2, 1), (1001, 10, 1), (1002, 10, 4),
                              (1003, None, 1), (1004, None, 4),
                              (1005, 2, None), (1006, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        if train is not None:
            assert len(puzzle["train"]) == train
        if test is not None:
            assert len(puzzle["test"]) == test
        other = generate_puzzle(seed, train, test)
        for split in ("train", "test"):
            for a, b in zip(puzzle[split], other[split]):
                assert np.array_equal(a["input"], b["input"])
                assert np.array_equal(a["output"], b["output"])
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 3), (3, 4), (2, 4)):
        try:
            generate_puzzle(2000, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
