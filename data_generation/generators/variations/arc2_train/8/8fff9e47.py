"""Generate ARC puzzles in which 2-by-2 instructions build nested rings."""

import numpy as np


def _grid_key(grid: np.ndarray) -> tuple:
    return (tuple(grid.shape), tuple(int(x) for x in grid.flat))


_SOURCE_INPUTS = (
    ((1, 3, 9, 4), (5, 5, 2, 8), (9, 8, 3, 1),
     (4, 0, 1, 4), (2, 3, 6, 5), (3, 9, 8, 0)),
    ((9, 1, 1, 7, 7, 9), (2, 0, 7, 7, 0, 3),
     (2, 8, 7, 7, 2, 1), (5, 3, 9, 7, 7, 8)),
    ((6, 9, 7, 5), (5, 8, 8, 7), (0, 1, 2, 6), (8, 7, 4, 3)),
)
_SOURCE_INPUT_KEYS = frozenset(
    _grid_key(np.array(rows, dtype=np.int8)) for rows in _SOURCE_INPUTS
)


def _extract_tiles(grid: np.ndarray) -> list[np.ndarray]:
    return [grid[r:r + 2, c:c + 2]
            for r in range(0, grid.shape[0], 2)
            for c in range(0, grid.shape[1], 2)]


def _construct_output(grid: np.ndarray) -> np.ndarray:
    current = _extract_tiles(grid)[0].copy()
    for size, tile in enumerate(_extract_tiles(grid)[1:], start=2):
        nw, ne, sw, se = (int(x) for x in tile.flat)
        expanded = np.empty((2 * size, 2 * size), dtype=np.int8)
        expanded[0, :size] = expanded[:size, 0] = nw
        expanded[0, size:] = expanded[:size, -1] = ne
        expanded[-1, :size] = expanded[size:, 0] = sw
        expanded[-1, size:] = expanded[size:, -1] = se
        expanded[1:-1, 1:-1] = current
        current = expanded
    return current


def _coordinate_output(grid: np.ndarray) -> np.ndarray:
    """Independent coordinate description of the task, used for validation."""
    tiles = _extract_tiles(grid)
    count = len(tiles)
    result = np.empty((2 * count, 2 * count), dtype=np.int8)
    for row in range(2 * count):
        for col in range(2 * count):
            depth = max(abs(row - (count - .5)), abs(col - (count - .5)))
            ring = int(depth + .5)
            tile = tiles[ring - 1]
            result[row, col] = tile[int(row >= count), int(col >= count)]
    return result


def _validate_count(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    lo, hi = (2, 10) if name == "num_train" else (1, 4)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be between {lo} and {hi}")
    return value


def _pair_counts(rng: np.random.Generator, num_train: int | None,
                 num_test: int | None) -> tuple[int, int]:
    train = _validate_count(num_train, "num_train")
    test = _validate_count(num_test, "num_test")
    if train is not None and test is not None:
        if train <= test:
            raise ValueError("num_train must exceed num_test")
    elif train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(max(2, test + 1), 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


# Canonical equality patterns. Tile orientation is never changed after sampling.
_PATTERNS = (
    (0, 1, 2, 3),
    (0, 0, 1, 2), (0, 1, 0, 2), (0, 1, 2, 0),
    (0, 1, 1, 2), (0, 1, 2, 1), (0, 1, 2, 2),
    (0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0),
    (0, 0, 0, 1), (0, 0, 1, 0),
    (0, 1, 0, 0), (0, 1, 1, 1),
    (0, 0, 0, 0),
)


def _shape_for_count(rng: np.random.Generator, count: int) -> tuple[int, int]:
    factors = [(rows, count // rows) for rows in range(1, count + 1)
               if count % rows == 0]
    rows, cols = factors[int(rng.integers(len(factors)))]
    return 2 * rows, 2 * cols


def _sample_shape(rng: np.random.Generator, regime: str,
                  train_max: int) -> tuple[int, int]:
    if regime == "target_test" and rng.random() < .15:
        return (4, 4)
    if regime == "target_test":
        choice = rng.random()
        if choice < .5:
            count = int(rng.integers(train_max, 16))
        elif choice < .75 and train_max > 2:
            count = int(rng.integers(2, train_max))
        else:
            count = int(rng.integers(2, 16))
    else:
        count = int(rng.integers(2, 13))
    return _shape_for_count(rng, count)


def _make_instruction_grid(rng: np.random.Generator, shape: tuple[int, int],
                           palette_size: int, anchor: bool) -> np.ndarray:
    count = shape[0] * shape[1] // 4
    if anchor:
        patterns = [(0, 1, 2, 3) for _ in range(count)]
    else:
        weights = np.array([.27] + [.035] * 6 + [.07] * 3
                           + [.045] * 4 + [.13], dtype=float)
        if palette_size >= 8:
            weights[:7] *= 1.8
        allowed = [i for i, pattern in enumerate(_PATTERNS)
                   if len(set(pattern)) <= palette_size]
        probabilities = weights[allowed] / weights[allowed].sum()
        patterns = [_PATTERNS[int(rng.choice(allowed, p=probabilities))]
                    for _ in range(count)]

    uniform_pair = None
    if not anchor and count >= 4 and rng.random() < .18:
        indices = rng.choice(count, size=2, replace=False)
        uniform_pair = (int(indices[0]), int(indices[1]))
        for index in uniform_pair:
            patterns[index] = (0, 0, 0, 0)

    duplicate = None
    if not anchor and count >= 3 and rng.random() < .34:
        candidates = [i for i in range(count)
                      if uniform_pair is None or i not in uniform_pair]
        if len(candidates) >= 2:
            indices = rng.choice(candidates, size=2, replace=False)
            duplicate = (int(indices[0]), int(indices[1]))

    active = [i for i in range(count)
              if duplicate is None or i != duplicate[1]]
    slots = [(i, label) for i in active for label in set(patterns[i])]
    while len(slots) < palette_size:
        eligible = [i for i in active if len(set(patterns[i])) < 4
                    and (uniform_pair is None or i not in uniform_pair)]
        if not eligible:
            break
        index = eligible[int(rng.integers(len(eligible)))]
        patterns[index] = (0, 1, 2, 3) if palette_size >= 4 else (0, 1, 2, 2)
        slots = [(i, label) for i in active for label in set(patterns[i])]
    palette_size = min(palette_size, len(slots))
    palette = [int(x) for x in rng.choice(10, size=palette_size, replace=False)]

    assigned = {}
    if uniform_pair is not None:
        for index, color in zip(uniform_pair, palette[:2]):
            assigned[index, 0] = color
    remaining_colors = [c for c in palette if c not in assigned.values()]
    remaining_slots = [slot for slot in slots if slot not in assigned]
    rng.shuffle(remaining_slots)
    rng.shuffle(remaining_colors)
    for slot, color in zip(remaining_slots, remaining_colors):
        assigned[slot] = color
    for index, label in slots:
        if (index, label) not in assigned:
            used = {color for (i, _), color in assigned.items() if i == index}
            choices = [color for color in palette if color not in used]
            assigned[index, label] = int(rng.choice(choices))

    tiles = []
    for index, pattern in enumerate(patterns):
        if duplicate is not None and index == duplicate[1]:
            tiles.append(None)
        else:
            tiles.append(np.array([assigned[index, label] for label in pattern],
                                  dtype=np.int8).reshape(2, 2))
    if duplicate is not None:
        tiles[duplicate[1]] = tiles[duplicate[0]].copy()

    tile_cols = shape[1] // 2
    return np.block([tiles[row:row + tile_cols]
                     for row in range(0, count, tile_cols)]).astype(np.int8)


def _make_pair(rng: np.random.Generator, shape: tuple[int, int],
               anchor: bool, seen: set, avoid_palette: int | None = None) -> dict:
    count = shape[0] * shape[1] // 4
    for _ in range(40):
        if anchor:
            palette_size = int(rng.integers(6, min(10, 4 * count) + 1))
        else:
            palette_size = int(rng.integers(2, min(10, 4 * count) + 1))
        inp = _make_instruction_grid(rng, shape, palette_size, anchor)
        if avoid_palette is not None and len(np.unique(inp)) == avoid_palette:
            continue
        key = _grid_key(inp)
        if key in seen or key in _SOURCE_INPUT_KEYS:
            continue
        if anchor:
            signatures = [tuple(int(x) for x in tile.flat)
                          for tile in _extract_tiles(inp)]
            if len(set(signatures)) != len(signatures):
                continue
        seen.add(key)
        return {"input": inp, "output": _construct_output(inp)}
    # Fixed enumeration is a deterministic escape hatch if random collisions
    # ever exhaust the small retry budget for a legal requested pair count.
    for serial in range(1000):
        if anchor:
            # Anchor tiles must have unique signatures, including at nine rings.
            values = [((tile + corner + serial) % 10)
                      for tile in range(count) for corner in range(4)]
            tiles = [np.array(values[4 * i:4 * i + 4], dtype=np.int8).reshape(2, 2)
                     for i in range(count)]
            columns = shape[1] // 2
            inp = np.block([tiles[row:row + columns]
                            for row in range(0, count, columns)])
        else:
            values = np.arange(shape[0] * shape[1], dtype=np.int8) % 10
            values[:3] = (serial % 10, (serial // 10) % 10, (serial // 100) % 10)
            inp = values.reshape(shape)
        key = _grid_key(inp)
        if (key in seen or key in _SOURCE_INPUT_KEYS or
                (avoid_palette is not None and len(np.unique(inp)) == avoid_palette)):
            continue
        seen.add(key)
        return {"input": inp, "output": _construct_output(inp)}
    raise RuntimeError("could not make a unique instruction grid")


def _training_reveals_conventions(train: list[dict]) -> bool:
    # A two-dimensional, four-color-per-tile example fixes the scan order,
    # corner orientation, and one-cell-per-tile growth even if other examples
    # contain merged, repeated, or uniform rings.
    for pair in train:
        inp = pair["input"]
        if inp.shape[0] < 4 or inp.shape[1] < 4:
            continue
        tiles = _extract_tiles(inp)
        signatures = [tuple(int(x) for x in tile.flat) for tile in tiles]
        if (len(signatures) >= 4 and len(set(signatures)) == len(signatures)
                and all(len(set(signature)) == 4 for signature in signatures)):
            return True
    return False


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a nonnegative integer")
    if int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    train_count, test_count = _pair_counts(rng, num_train, num_test)
    # The source family has no latent convention: all colors are pair-local
    # payloads and tile order/orientation are fixed by the demonstrated rule.
    pair_seeds = rng.integers(0, 2**63, size=train_count + test_count,
                              dtype=np.int64)
    pair_rngs = [np.random.default_rng(int(value)) for value in pair_seeds]
    anchor_shapes = ((4, 4), (4, 6), (6, 4), (6, 6))
    anchor_shape = anchor_shapes[int(rng.integers(len(anchor_shapes)))]
    seen = set()
    train = [_make_pair(pair_rngs[0], anchor_shape, True, seen)]
    anchor_count = anchor_shape[0] * anchor_shape[1] // 4
    for index in range(1, train_count):
        local = pair_rngs[index]
        shape = _sample_shape(local, "source_train", anchor_count)
        if index == 1:
            # Even a two-demonstration puzzle changes ring count and aspect.
            if (shape[0] * shape[1] // 4 == anchor_count or
                    shape[0] * anchor_shape[1] == shape[1] * anchor_shape[0]):
                alternatives = [(2 * rows, 2 * cols)
                                for rows in range(1, 13)
                                for cols in range(1, 13)
                                if 2 <= rows * cols <= 12
                                and rows * cols != anchor_count
                                and rows * anchor_shape[1] != cols * anchor_shape[0]]
                shape = alternatives[int(local.integers(len(alternatives)))]
        avoid_palette = len(np.unique(train[0]["input"])) if index == 1 else None
        train.append(_make_pair(local, shape, False, seen, avoid_palette))
    train_max = max(pair["input"].size // 4 for pair in train)
    test = []
    for index in range(test_count):
        local = pair_rngs[train_count + index]
        shape = _sample_shape(local, "target_test", train_max)
        test.append(_make_pair(local, shape, False, seen))
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
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            assert inp.shape[0] % 2 == inp.shape[1] % 2 == 0
            count = inp.size // 4
            assert 2 <= count <= 15
            assert out.shape == (2 * count, 2 * count)
            key = _grid_key(inp)
            assert key not in seen and key not in _SOURCE_INPUT_KEYS
            seen.add(key)
            assert np.array_equal(out, _coordinate_output(inp))
            assert set(int(x) for x in np.unique(inp)) == set(int(x) for x in np.unique(out))
            for ring, tile in enumerate(_extract_tiles(inp), start=1):
                lo = count - ring
                hi = count + ring - 1
                assert np.array_equal(tile, out[np.ix_([lo, hi], [lo, hi])])
    assert _training_reveals_conventions(puzzle["train"])
    assert len({pair["input"].size // 4 for pair in puzzle["train"]}) >= 2
    assert len({len(np.unique(pair["input"])) for pair in puzzle["train"]}) >= 2
    assert len({pair["input"].shape[0] / pair["input"].shape[1]
                for pair in puzzle["train"]}) >= 2


def _puzzles_equal(first: dict, second: dict) -> bool:
    return all(len(first[split]) == len(second[split]) and
               all(np.array_equal(a[name], b[name])
                   for a, b in zip(first[split], second[split])
                   for name in ("input", "output"))
               for split in ("train", "test"))


def _expect_value_error(call) -> None:
    try:
        call()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def self_test() -> None:
    for train_count, test_count in ((2, 1), (10, 1), (10, 4), (7, 3)):
        puzzle = generate_puzzle(1000 + train_count * 10 + test_count,
                                 train_count, test_count)
        assert len(puzzle["train"]) == train_count
        assert len(puzzle["test"]) == test_count
        assert _puzzles_equal(puzzle, generate_puzzle(1000 + train_count * 10 + test_count,
                                                     train_count, test_count))
    for call in (lambda: generate_puzzle(-1),
                 lambda: generate_puzzle(1, num_train=1),
                 lambda: generate_puzzle(1, num_train=11),
                 lambda: generate_puzzle(1, num_test=0),
                 lambda: generate_puzzle(1, num_test=5),
                 lambda: generate_puzzle(1, num_train=2, num_test=2),
                 lambda: generate_puzzle(1, num_train=3, num_test=4)):
        _expect_value_error(call)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(puzzle["train"]) for puzzle in puzzles}) >= 4
    assert len({len(puzzle["test"]) for puzzle in puzzles}) >= 3
    shapes, palettes, patterns, counts = set(), set(), set(), set()
    repeated, uniform, sparse, large = 0, 0, 0, 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                shapes.add(inp.shape)
                palettes.add(len(np.unique(inp)))
                counts.add(inp.size // 4)
                signatures = [tuple(int(x) for x in tile.flat)
                              for tile in _extract_tiles(inp)]
                repeated += len(signatures) != len(set(signatures))
                uniform += any(len(set(signature)) == 1 for signature in signatures)
                sparse += len(np.unique(inp)) <= 4
                large += inp.size // 4 >= 10
                for signature in signatures:
                    labels = {}
                    patterns.add(tuple(labels.setdefault(value, len(labels))
                                       for value in signature))
    assert len(shapes) >= 15 and len(counts) >= 12
    assert len(palettes) >= 8 and len(patterns) >= 10
    assert repeated >= 5 and uniform >= 10 and sparse >= 10 and large >= 10
