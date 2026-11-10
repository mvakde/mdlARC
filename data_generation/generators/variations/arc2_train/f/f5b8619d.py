"""Marker-selected column completion followed by exact 2-by-2 tiling."""

import numpy as np


_SOURCE_INPUTS = (
    np.array([[0, 5, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
              [5, 0, 0, 0, 0, 5], [0, 0, 0, 0, 0, 0]], dtype=np.int8),
    np.array([[2, 0, 0], [0, 0, 0], [0, 0, 2]], dtype=np.int8),
    np.array([[0, 4], [0, 0]], dtype=np.int8),
    np.array([[0, 0, 3, 0], [0, 0, 0, 0],
              [0, 0, 0, 3], [3, 0, 0, 0]], dtype=np.int8),
)


def _complete_and_tile(grid: np.ndarray, background: int, fill: int) -> np.ndarray:
    completed = grid.copy()
    active = np.any(grid != background, axis=0)
    completed[:, active] = np.where(grid[:, active] == background,
                                    fill, grid[:, active])
    return np.tile(completed, (2, 2)).astype(np.int8, copy=False)


def _source_copy(grid: np.ndarray, output: np.ndarray) -> bool:
    return any(grid.shape == source.shape and np.array_equal(grid, source)
               and np.array_equal(output, _complete_and_tile(source, 0, 8))
               for source in _SOURCE_INPUTS)


def _normalize_count(name: str, value: int | None, low: int, high: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    result = int(value)
    if not low <= result <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return result


def _resolve_counts(rng: np.random.Generator, num_train: int | None,
                    num_test: int | None) -> tuple[int, int]:
    train = _normalize_count("num_train", num_train, 2, 10)
    test = _normalize_count("num_test", num_test, 1, 4)
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


def _training_shape(rng: np.random.Generator, anchor: bool,
                    square_only: bool) -> tuple[int, int]:
    low = 3 if anchor else 2
    height = int(rng.integers(low, 7))
    width = (height if square_only or rng.random() < 0.44
             else int(rng.integers(low, 7)))
    return height, width


def _test_shape(rng: np.random.Generator,
                train_shapes: list[tuple[int, int]]) -> tuple[int, int]:
    mode = rng.random()
    if mode < 0.14:
        return 4, 4
    if mode < 0.42:
        return train_shapes[int(rng.integers(len(train_shapes)))]
    if mode < 0.65:
        return int(rng.integers(2, 8)), int(rng.integers(2, 8))
    return int(rng.integers(7, 16)), int(rng.integers(7, 16))


def _active_columns(rng: np.random.Generator, width: int,
                    anchor: bool, test: bool) -> np.ndarray:
    if anchor:
        count = int(rng.integers(2, width))
    else:
        u = rng.random()
        if u < (0.16 if test else 0.22):
            count = 1
        elif u < (0.36 if test else 0.34):
            count = width
        else:
            count = int(rng.integers(1, width + 1))
    if count == width:
        return np.arange(width)
    if rng.random() < 0.35:
        start = int(rng.integers(0, width - count + 1))
        return np.arange(start, start + count)
    return np.sort(rng.choice(width, size=count, replace=False))


def _build_input(rng: np.random.Generator, shape: tuple[int, int],
                 background: int, marker: int, anchor: bool,
                 test: bool) -> np.ndarray:
    height, width = shape
    for _ in range(160):
        columns = _active_columns(rng, width, anchor, test)
        grid = np.full(shape, background, dtype=np.int8)
        for column in columns:
            grid[int(rng.integers(height)), int(column)] = marker
        if anchor:
            # Move this evidence among training pairs; do not fix its geometry.
            column = int(rng.choice(columns))
            vacant = np.flatnonzero(grid[:, column] == background)
            grid[int(rng.choice(vacant)), column] = marker
        else:
            if (len(columns) >= 2 and width >= 3
                    and rng.random() < (0.16 if test else 0.09)):
                grid[:, int(rng.choice(columns))] = marker
            for raw_column in columns:
                column = int(raw_column)
                if rng.random() < (0.54 if test else 0.44):
                    vacant = np.flatnonzero(grid[:, column] == background)
                    if len(vacant) > 1:
                        extra = int(rng.integers(1, min(3, len(vacant) - 1) + 1))
                        grid[rng.choice(vacant, size=extra, replace=False), column] = marker
        markers = int(np.count_nonzero(grid == marker))
        if markers * 2 >= grid.size:
            continue
        active = np.any(grid == marker, axis=0)
        if not np.any((grid == background) & active[None, :]):
            continue
        return grid
    raise RuntimeError("could not construct an input with identifiable color roles")


def _grid_key(grid: np.ndarray) -> tuple:
    return grid.shape, grid.tobytes()


def _roles(grid: np.ndarray) -> tuple[int, int]:
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 2 and counts[0] != counts[1]
    return int(colors[np.argmax(counts)]), int(colors[np.argmin(counts)])


def _make_pair(rng: np.random.Generator, shape: tuple[int, int],
               background: int, marker: int, fill: int, anchor: bool,
               test: bool, occupied: set) -> dict:
    if test:
        fallback = [(11, 12), (12, 13), (13, 14), (14, 15)]
    elif shape[0] == shape[1]:
        fallback = [(3, 3), (4, 4), (5, 5), (6, 6)]
    else:
        fallback = [(3, 4), (4, 3), (5, 6), (6, 5), (4, 6), (6, 4)]
    for candidate in dict.fromkeys([shape] + fallback):
        for _ in range(120):
            grid = _build_input(rng, candidate, background, marker, anchor, test)
            output = _complete_and_tile(grid, background, fill)
            key = _grid_key(grid)
            mask_key = (grid.shape, (grid == marker).tobytes())
            if (("grid", key) not in occupied and ("mask", mask_key) not in occupied
                    and not _source_copy(grid, output)):
                occupied.add(("grid", key))
                occupied.add(("mask", mask_key))
                return {"input": grid, "output": output}
    raise RuntimeError("could not construct a distinct pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    # Fill is the sole puzzle-wide color convention, revealed in every train output.
    fill = int(rng.integers(10))
    candidates = [color for color in range(10) if color != fill]
    rng.shuffle(candidates)
    palette_size = min(train_count, int(rng.integers(2, 4)))
    train_palette = candidates[:palette_size]
    background_pool = candidates[palette_size:]
    variable_background = bool(rng.random() < 0.62)
    if variable_background:
        backgrounds = [background_pool[0], background_pool[1]]
        backgrounds += [int(rng.choice(background_pool)) for _ in range(train_count - 2)]
        rng.shuffle(backgrounds)
    else:
        backgrounds = [int(rng.choice(background_pool))] * train_count
    markers = [int(train_palette[i % palette_size]) for i in range(train_count)]
    rng.shuffle(markers)
    anchor_index = int(rng.integers(train_count))
    square_only = bool(rng.random() < 0.13)
    shapes = [_training_shape(rng, i == anchor_index, square_only)
              for i in range(train_count)]
    occupied: set = set()
    train = [_make_pair(rng, shapes[i], backgrounds[i], markers[i], fill,
                        i == anchor_index, False, occupied)
             for i in range(train_count)]
    shapes = [pair["input"].shape for pair in train]
    # Both established and new markers appear when there are multiple tests.
    reuse = [True, False] if test_count >= 2 else [bool(rng.integers(2))]
    reuse += [bool(rng.integers(2)) for _ in range(test_count - len(reuse))]
    rng.shuffle(reuse)
    test = []
    for old_marker in reuse:
        background = (int(rng.choice(background_pool)) if variable_background
                      else backgrounds[0])
        if old_marker:
            marker = int(rng.choice(train_palette))
        else:
            fresh = [c for c in range(10)
                     if c != fill and c != background and c not in train_palette]
            marker = int(rng.choice(fresh))
        test.append(_make_pair(rng, _test_shape(rng, shapes), background,
                               marker, fill, False, True, occupied))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _predict_from_training(train: list[dict], test_input: np.ndarray) -> np.ndarray:
    """Use visible demonstrations and the test input, without generator state."""
    fills = []
    for pair in train:
        added = set(np.unique(pair["output"]).tolist()) - set(np.unique(pair["input"]).tolist())
        assert len(added) == 1
        fills.append(int(added.pop()))
    assert len(set(fills)) == 1
    background, marker = _roles(test_input)
    height, width = test_input.shape
    block = test_input.copy()
    for column in range(width):
        if marker in test_input[:, column]:
            for row in range(height):
                if test_input[row, column] == background:
                    block[row, column] = fills[0]
    result = np.empty((2 * height, 2 * width), dtype=np.int8)
    for row_offset in (0, height):
        for col_offset in (0, width):
            result[row_offset:row_offset + height,
                   col_offset:col_offset + width] = block
    return result


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    inputs, pairs, fills, train_markers, train_masks = set(), set(), set(), set(), set()
    test_markers = []
    anchor_evidence = False
    for regime, split in (("train", train), ("test", test)):
        for pair in split:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            grid, output = pair["input"], pair["output"]
            for item in (grid, output):
                assert isinstance(item, np.ndarray) and item.ndim == 2
                assert np.issubdtype(item.dtype, np.integer)
                assert all(1 <= side <= 30 for side in item.shape)
                assert np.all((0 <= item) & (item <= 9))
            key = _grid_key(grid)
            pair_key = key, _grid_key(output)
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            assert not _source_copy(grid, output)
            background, marker = _roles(grid)
            assert np.count_nonzero(grid == background) > np.count_nonzero(grid == marker)
            added = set(np.unique(output).tolist()) - {background, marker}
            assert len(added) == 1
            fill = int(added.pop())
            fills.add(fill)
            assert output.shape == (2 * grid.shape[0], 2 * grid.shape[1])
            assert np.array_equal(output, _complete_and_tile(grid, background, fill))
            active = np.any(grid == marker, axis=0)
            assert active.any()
            if regime == "train":
                assert all(2 <= side <= 6 for side in grid.shape)
                train_markers.add(marker)
                train_masks.add((grid.shape, (grid == marker).tobytes()))
                if (active.sum() >= 2 and active.sum() < grid.shape[1]
                        and np.any(np.sum(grid == marker, axis=0) >= 2)):
                    anchor_evidence = True
            else:
                assert all(2 <= side <= 15 for side in grid.shape)
                assert (grid.shape, (grid == marker).tobytes()) not in train_masks
                test_markers.append(marker)
                assert np.array_equal(output, _predict_from_training(train, grid))
    assert len(fills) == 1 and len(train_markers) >= 2 and anchor_evidence
    if len(test) >= 2:
        assert any(color in train_markers for color in test_markers)
        assert any(color not in train_markers for color in test_markers)


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[role], b[role])
        for a, b in zip(first[split], second[split]) for role in ("input", "output"))
        for split in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert all(_puzzles_equal(p, generate_puzzle(seed)) for seed, p in enumerate(puzzles))
    assert all(not _puzzles_equal(puzzles[0], p) for p in puzzles[1:])
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    trains = [pair for p in puzzles for pair in p["train"]]
    tests = [pair for p in puzzles for pair in p["test"]]
    all_pairs = trains + tests
    assert len({pair["input"].shape for pair in tests}) >= 20
    assert len({pair["input"].shape for pair in trains}) >= 18
    assert sum(pair["input"].shape != (4, 4) for pair in tests) >= len(tests) // 2
    assert any(pair["input"].shape == (4, 4) for pair in trains)
    assert any(len({pair["input"].shape for pair in p["train"]}) < len(p["train"])
               for p in puzzles)
    assert any(set(pair["input"].shape for pair in p["train"])
               & set(pair["input"].shape for pair in p["test"]) for p in puzzles)
    assert any(len({_roles(pair["input"])[0] for pair in p["train"]}) > 1 for p in puzzles)
    assert len({_roles(pair["input"])[0] for pair in all_pairs}) >= 7
    assert len({next(iter(set(np.unique(pair["output"])) - set(np.unique(pair["input"]))))
                for pair in trains}) >= 7
    observations = []
    for regime, split in (("train", trains), ("test", tests)):
        for pair in split:
            grid = pair["input"]
            background, marker = _roles(grid)
            column_counts = np.sum(grid == marker, axis=0)
            active = column_counts > 0
            observations.append((regime, grid.shape, background, int(active.sum()),
                                 bool(np.any(column_counts > 1)),
                                 bool(np.any(column_counts == grid.shape[0])),
                                 bool(active.all()),
                                 bool(np.any(active[:-1] & active[1:])),
                                 bool(np.any(np.diff(np.flatnonzero(active)) > 1)),
                                 (grid.shape, (grid == marker).tobytes())))
    train_obs = [item for item in observations if item[0] == "train"]
    test_obs = [item for item in observations if item[0] == "test"]
    assert len({item[-1] for item in observations}) >= 160
    assert sum(item[6] for item in train_obs) >= 3
    assert sum(item[6] for item in test_obs) >= 2
    assert sum(item[5] for item in observations) >= 2
    assert sum(item[4] for item in test_obs) >= 5
    assert sum(item[3] == 1 for item in test_obs) >= 5
    assert sum(item[3] >= 2 for item in test_obs) >= 10
    assert sum(item[7] for item in test_obs) >= 5
    assert sum(item[8] for item in test_obs) >= 5
    assert sum(item[1][0] >= 7 or item[1][1] >= 7 for item in test_obs) >= 10
    assert sum(item[1][0] != item[1][1] and item[4] and item[2] != 0
               for item in test_obs) >= 5
    assert sum(len({_roles(pair["input"])[0] for pair in p["train"]}) > 1
               for p in puzzles) >= 5
    assert sum(len({pair["input"].shape for pair in p["train"]}) < len(p["train"])
               for p in puzzles) >= 5
    assert sum(bool({pair["input"].shape for pair in p["train"]}
                    & {pair["input"].shape for pair in p["test"]})
               for p in puzzles) >= 5
    assert sum(len({int(np.any(pair["input"] == _roles(pair["input"])[1], axis=0).sum())
                    for pair in p["train"]}) >= 2 for p in puzzles) >= 20
    assert sum(all(pair["input"].shape[0] == pair["input"].shape[1]
                   for pair in p["train"]) for p in puzzles) >= 1
    assert sum(pair["input"].shape == (4, 4) for pair in tests) >= 2
    assert sum(pair["input"].shape == (4, 4) for pair in tests) <= len(tests) // 2
    reused = sum(_roles(pair["input"])[1] in
                 {_roles(train_pair["input"])[1] for train_pair in p["train"]}
                 for p in puzzles for pair in p["test"])
    assert 10 <= reused <= len(tests) - 10
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (6, 2), (7, 3)):
        p = generate_puzzle(1000 + train_count * 10 + test_count, train_count, test_count)
        assert (len(p["train"]), len(p["test"])) == (train_count, test_count)
    for index, (train_count, test_count) in enumerate(
            ((None, 1), (None, 4), (2, None), (10, None))):
        p = generate_puzzle(2000 + index, train_count, test_count)
        assert train_count is None or len(p["train"]) == train_count
        assert test_count is None or len(p["test"]) == test_count
    for train_count, test_count in ((1, None), (11, None), (None, 0), (None, 5),
                                    (2, 2), (3, 4), (4, 4), (2.0, 1),
                                    (2, True), ("2", 1)):
        try:
            generate_puzzle(9, train_count, test_count)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count request accepted")
    assert _puzzles_equal(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3))
