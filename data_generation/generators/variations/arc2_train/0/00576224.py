"""Whole-tile repetition with alternating reflected bands."""

import numpy as np


_SOURCE_INPUTS = (((7, 9), (4, 3)), ((8, 6), (6, 4)), ((3, 2), (7, 8)))
_ORIENTATIONS = ("horizontal", "vertical")
_REPETITIONS = (2, 3, 4, 5)
_BANDS = (3, 5)
_POSITION_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    return seed


def _count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _resolve_counts(seed: int, num_train: int | None, num_test: int | None) -> tuple[int, int]:
    train = _count(num_train, "num_train")
    test = _count(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be 2..10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be 1..4")
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _output(tile: np.ndarray, orientation: str, repetitions: int, bands: int) -> np.ndarray:
    if orientation == "horizontal":
        strip = np.tile(tile, (1, repetitions))
        return np.vstack([strip if i % 2 == 0 else np.fliplr(strip) for i in range(bands)])
    strip = np.tile(tile, (repetitions, 1))
    return np.hstack([strip if i % 2 == 0 else np.flipud(strip) for i in range(bands)])


def _key(grid: np.ndarray) -> tuple:
    return tuple(grid.shape), tuple(int(v) for v in grid.flat)


def _source_copy(tile: np.ndarray, output: np.ndarray) -> bool:
    return any(
        np.array_equal(tile, np.asarray(values))
        and np.array_equal(output, _output(np.asarray(values), "horizontal", 3, 3))
        for values in _SOURCE_INPUTS
    )


def _axis(shape: tuple[int, int], orientation: str) -> int:
    return shape[1] if orientation == "horizontal" else shape[0]


def _choose_shape(
    rng: np.random.Generator,
    convention: tuple[str, int, int],
    regime: str,
    previous: tuple[int, int] | None = None,
    clear: bool = False,
) -> tuple[int, int]:
    orientation, repetitions, bands = convention
    max_h, max_w = ((30 // bands, 30 // repetitions) if orientation == "horizontal"
                    else (30 // repetitions, 30 // bands))
    shapes, weights = [], []
    for h in range(1, max_h + 1):
        for w in range(1, max_w + 1):
            shape = (h, w)
            if shape == (1, 1):
                continue
            if clear and (min(shape) < 2 or _axis(shape, orientation) < 3):
                continue
            if previous is not None and (h == previous[0] or w == previous[1]):
                continue
            weight = 1.0
            if shape == (2, 2):
                weight += 38
            if 2 <= h <= 4 and 2 <= w <= 4:
                weight += 5
            if h != w:
                weight += 2
            if max(shape) >= 6:
                weight += 3 if regime == "source_train" else 7
            if min(shape) == 1:
                weight += 0.8 if regime == "source_train" else 5
            shapes.append(shape)
            weights.append(weight)
    probabilities = np.asarray(weights, dtype=float)
    probabilities /= probabilities.sum()
    return shapes[int(rng.choice(len(shapes), p=probabilities))]


def _four_cells(
    rng: np.random.Generator, mode: str, repeated_positions: tuple[int, int] | None = None
) -> np.ndarray:
    colors = rng.choice(10, size=4, replace=False).astype(np.int8)
    if mode == "distinct":
        values = colors
    elif mode == "one_pair":
        if repeated_positions is None:
            values = colors[[0, 0, 1, 2]][rng.permutation(4)]
        else:
            values = np.empty(4, dtype=np.int8)
            values[list(repeated_positions)] = colors[0]
            remaining = [i for i in range(4) if i not in repeated_positions]
            values[remaining] = colors[1:3]
    elif mode == "two_pairs":
        values = colors[[0, 0, 1, 1]][rng.permutation(4)]
    elif mode == "three_one":
        values = colors[[0, 0, 0, 1]][rng.permutation(4)]
    elif mode == "checkerboard":
        values = colors[[0, 1, 1, 0]]
    elif mode == "identical_rows":
        values = colors[[0, 1, 0, 1]]
    else:
        values = np.repeat(colors[0], 4)
    return values.reshape(2, 2)


def _sample_tile(
    rng: np.random.Generator,
    shape: tuple[int, int],
    orientation: str,
    regime: str,
    clear: bool,
) -> np.ndarray:
    h, w = shape
    if clear:
        # Three distinct cells distinguish reflection from a cyclic shift.
        size = int(rng.integers(3, min(10, h * w) + 1))
        colors = rng.choice(10, size=size, replace=False).astype(np.int8)
        tile = rng.choice(colors, size=shape).astype(np.int8)
        distinguished = rng.permutation(colors)[:3]
        if orientation == "horizontal":
            tile[0, :3] = distinguished
        else:
            tile[:3, 0] = distinguished
        axis = 1 if orientation == "horizontal" else 0
        if any(np.array_equal(np.flip(tile, axis=axis), np.roll(tile, shift, axis=axis))
               for shift in (-1, 1)):
            if orientation == "horizontal":
                tile[0, :] = distinguished[0]
                tile[0, :3] = distinguished
            else:
                tile[:, 0] = distinguished[0]
                tile[:3, 0] = distinguished
        return tile
    if shape == (2, 2):
        modes = ("distinct", "one_pair", "two_pairs", "three_one",
                 "checkerboard", "identical_rows", "mono")
        weights = ((5, 3, 2, 2, 2, 2, 1) if regime == "source_train"
                   else (1, 2, 3, 3, 3, 3, 2))
        mode = str(rng.choice(modes, p=np.asarray(weights) / sum(weights)))
        return _four_cells(rng, mode)
    modes = ("random", "random", "random", "row", "column", "checker", "sparse", "mono")
    mode = str(rng.choice(modes))
    if regime == "source_train" and mode == "mono":
        mode = "random"
    if mode == "mono":
        return np.full(shape, int(rng.integers(0, 10)), dtype=np.int8)
    if mode == "random":
        size = int(rng.integers(2, min(10, h * w) + 1))
        colors = rng.choice(10, size=size, replace=False).astype(np.int8)
        return rng.choice(colors, size=shape).astype(np.int8)
    colors = rng.choice(10, size=2, replace=False).astype(np.int8)
    rows, columns = np.indices(shape)
    if mode == "row":
        indices = rows % 2
    elif mode == "column":
        indices = columns % 2
    elif mode == "checker":
        indices = (rows + columns) % 2
    else:
        indices = np.zeros(shape, dtype=int)
        marks = rng.choice(h * w, size=max(1, h * w // 5), replace=False)
        indices.flat[marks] = 1
    return colors[indices].astype(np.int8)


def _new_pair(
    rng: np.random.Generator,
    convention: tuple[str, int, int],
    regime: str,
    seen: set,
    previous_shape: tuple[int, int] | None = None,
    clear: bool = False,
    forced_four_cell_mode: str | None = None,
    repeated_positions: tuple[int, int] | None = None,
) -> dict:
    for _ in range(120):
        if forced_four_cell_mode is None:
            shape = _choose_shape(rng, convention, regime, previous_shape, clear)
            tile = _sample_tile(rng, shape, convention[0], regime, clear)
        else:
            tile = _four_cells(rng, forced_four_cell_mode, repeated_positions)
        output = _output(tile, *convention)
        key = _key(tile)
        if key in seen or _source_copy(tile, output):
            continue
        seen.add(key)
        return {"input": tile, "output": output}
    raise RuntimeError("could not construct a new pair")


def _compatible(pairs: list) -> list[tuple[str, int, int]]:
    return [
        (orientation, repetitions, bands)
        for orientation in _ORIENTATIONS
        for repetitions in _REPETITIONS
        for bands in _BANDS
        if all(np.array_equal(pair["output"], _output(pair["input"], orientation, repetitions, bands))
               for pair in pairs)
    ]


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _seed(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    orientation = str(rng.choice(_ORIENTATIONS, p=(0.65, 0.35)))
    repetitions = int(rng.choice(_REPETITIONS, p=(0.19, 0.42, 0.23, 0.16)))
    bands = int(rng.choice(_BANDS, p=(0.7, 0.3)))
    convention = (orientation, repetitions, bands)
    seen = set()
    train = [_new_pair(rng, convention, "source_train", seen, clear=True)]
    train.append(_new_pair(rng, convention, "source_train", seen,
                           previous_shape=train[0]["input"].shape))
    for index in range(train_count - 2):
        if index == 0 and seed % 2 == 0:
            train.append(_new_pair(
                rng, convention, "source_train", seen,
                forced_four_cell_mode="one_pair",
                repeated_positions=_POSITION_PAIRS[(seed // 2) % 6],
            ))
        else:
            train.append(_new_pair(rng, convention, "source_train", seen))
    test = []
    if seed % 3 == 0:
        modes = ("distinct", "one_pair", "two_pairs", "three_one",
                 "checkerboard", "identical_rows", "mono")
        test.append(_new_pair(rng, convention, "target_test", seen,
                              forced_four_cell_mode=modes[seed % len(modes)]))
    for _ in range(test_count - len(test)):
        test.append(_new_pair(rng, convention, "target_test", seen))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "invalid pair counts")
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for index, pair in enumerate(puzzle[split]):
            label = f"{split}[{index}]"
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, f"{label}: invalid pair")
            for role in ("input", "output"):
                grid = pair[role]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, f"{label}.{role}: invalid grid")
                _require(all(1 <= n <= 30 for n in grid.shape), f"{label}.{role}: invalid shape")
                _require(np.issubdtype(grid.dtype, np.integer), f"{label}.{role}: invalid dtype")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), f"{label}.{role}: invalid color")
            tile, output = pair["input"], pair["output"]
            _require(tile.dtype == output.dtype, f"{label}: dtype changed")
            _require(not np.array_equal(tile, output), f"{label}: unchanged output")
            _require(np.array_equal(np.unique(tile), np.unique(output)), f"{label}: colors changed")
            input_key = _key(tile)
            pair_key = input_key, _key(output)
            _require(input_key not in seen_inputs and pair_key not in seen_pairs, f"{label}: duplicate")
            _require(not _source_copy(tile, output), f"{label}: source copy")
            seen_inputs.add(input_key)
            seen_pairs.add(pair_key)
    conventions = _compatible(train)
    _require(len(conventions) == 1, "training does not uniquely reveal the convention")
    orientation, repetitions, bands = conventions[0]
    _require(train[0]["input"].shape[0] != train[1]["input"].shape[0]
             and train[0]["input"].shape[1] != train[1]["input"].shape[1],
             "training does not vary both tile dimensions")
    axis = 1 if orientation == "horizontal" else 0
    _require(any(
        _axis(pair["input"].shape, orientation) >= 3
        and not np.array_equal(np.flip(pair["input"], axis=axis),
                               np.roll(pair["input"], 1, axis=axis))
        and not np.array_equal(np.flip(pair["input"], axis=axis),
                               np.roll(pair["input"], -1, axis=axis))
        for pair in train
    ), "training does not distinguish reflection from a shift")
    for pair in train + test:
        _require(np.array_equal(pair["output"], _output(pair["input"], *conventions[0])),
                 "pair violates the shared rule")


def _puzzle_key(puzzle: dict) -> tuple:
    return tuple(tuple((_key(pair["input"]), _key(pair["output"])) for pair in puzzle[s])
                 for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        convention = _compatible(puzzle["train"])
        assert len(convention) == 1
        orientation, repetitions, bands = convention[0]
        for pair in puzzle["train"] + puzzle["test"]:
            tile, output = pair["input"], pair["output"]
            if orientation == "horizontal":
                strip = np.concatenate([tile] * repetitions, axis=1)
                expected = np.concatenate(
                    [strip[:, ::-1] if i % 2 else strip for i in range(bands)], axis=0)
            else:
                strip = np.concatenate([tile] * repetitions, axis=0)
                expected = np.concatenate(
                    [strip[::-1, :] if i % 2 else strip for i in range(bands)], axis=1)
            assert np.array_equal(output, expected)
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    assert _puzzle_key(generate_puzzle(123456, 8, 3)) == _puzzle_key(generate_puzzle(123456, 8, 3))
    assert _puzzle_key(generate_puzzle(123456, 8, 3)) != _puzzle_key(generate_puzzle(123457, 8, 3))
    for seed, a, b in ((5001, 2, 1), (5002, 10, 1), (5003, 10, 4),
                       (6001, None, 1), (6002, None, 4), (6003, 2, None), (6004, 10, None)):
        puzzle = generate_puzzle(seed, a, b)
        validate_generated_puzzle(puzzle)
        assert a is None or len(puzzle["train"]) == a
        assert b is None or len(puzzle["test"]) == b
    for request in ((1, None), (11, None), (None, 0), (None, 5),
                    (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(7000, *request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {request}")
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) >= 40
    conventions = {_compatible(p["train"])[0] for p in puzzles}
    assert {v[0] for v in conventions} == set(_ORIENTATIONS)
    assert {v[1] for v in conventions} == set(_REPETITIONS)
    assert {v[2] for v in conventions} == set(_BANDS)
    pairs = [pair for p in puzzles for s in ("train", "test") for pair in p[s]]
    shapes = {pair["input"].shape for pair in pairs}
    assert len(shapes) >= 20
    assert any(min(shape) == 1 for shape in shapes)
    assert any(max(shape) >= 8 for shape in shapes)
    assert (2, 2) in shapes
    assert len({int(np.unique(pair["input"]).size) for pair in pairs}) >= 7
    assert any(np.unique(pair["input"]).size == 1 for p in puzzles for pair in p["test"])
    assert len({_key(pair["input"]) for pair in pairs}) >= 180
    assert any(_compatible(p["train"])[0] == ("horizontal", 3, 3) for p in puzzles)
    assert all(len({pair["input"].shape[0] for pair in p["train"]}) > 1
               and len({pair["input"].shape[1] for pair in p["train"]}) > 1
               for p in puzzles)
    assert sum(len({tuple(int(v) for v in np.unique(pair["input"]))
                    for split in ("train", "test") for pair in p[split]}) > 1
               for p in puzzles) >= 35
    assert any(0 in pair["input"] for pair in pairs)
    assert any(np.unique(pair["input"]).size >= 7 for pair in pairs)
    tests = [pair["input"] for p in puzzles for pair in p["test"]]
    assert any(min(tile.shape) == 1 for tile in tests)
    assert any(max(tile.shape) >= 6 for tile in tests)
    two_by_two_tests = [tile for tile in tests if tile.shape == (2, 2)]
    multiplicities = {tuple(sorted(np.unique(tile, return_counts=True)[1]))
                      for tile in two_by_two_tests}
    assert {(4,), (1, 3), (2, 2), (1, 1, 2), (1, 1, 1, 1)} <= multiplicities
    assert any(tile[0, 0] == tile[1, 1] and tile[0, 1] == tile[1, 0]
               and tile[0, 0] != tile[0, 1] for tile in two_by_two_tests)
    assert any(np.array_equal(tile[0], tile[1]) and tile[0, 0] != tile[0, 1]
               for tile in two_by_two_tests)
    repeated_positions = set()
    for puzzle in puzzles:
        for pair in puzzle["train"]:
            tile = pair["input"]
            if tile.shape != (2, 2):
                continue
            values, counts = np.unique(tile, return_counts=True)
            if tuple(sorted(counts)) == (1, 1, 2):
                repeated = values[np.flatnonzero(counts == 2)[0]]
                repeated_positions.add(tuple(int(v) for v in np.flatnonzero(tile.flat == repeated)))
    assert repeated_positions == set(_POSITION_PAIRS)
