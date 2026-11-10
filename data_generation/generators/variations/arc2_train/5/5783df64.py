"""Generate singleton-marker sector-compression ARC puzzles."""

import numpy as np


SIZES = ((2, 2), (3, 3), (4, 4), (2, 3), (3, 2), (3, 4),
         (4, 3), (5, 5), (2, 4), (4, 2), (5, 3), (3, 5))
WEIGHTS = np.array((18, 18, 10, 10, 10, 7, 7, 3, 5, 5, 4, 3), dtype=float)
WEIGHTS /= WEIGHTS.sum()
MODES = ("free", "aligned", "center", "row", "column", "edge", "boundary")
SOURCE = (
    (("000700800", "040000000", "000000000", "000002000", "000000001",
      "500000000", "003000600", "000090000", "000000000"), ("478", "521", "396")),
    (("060007", "000400", "200090", "003000", "000500", "100080"),
     ("647", "239", "158")),
    (("200100", "000007", "030090", "005000", "000600", "400080"),
     ("217", "359", "468")),
    (("000000900", "500006000", "000000000", "000000080", "000001000",
      "004000000", "000000000", "030200000", "000000007"), ("569", "418", "327")),
)


def _grid(rows):
    return np.array([[int(c) for c in row] for row in rows], dtype=np.int8)


SOURCE_PAIRS = tuple((_grid(a), _grid(b)) for a, b in SOURCE)


def _key(array):
    return (array.shape, array.tobytes())


def _source_copy(inp, out):
    return any(np.array_equal(inp, a) and np.array_equal(out, b)
               for a, b in SOURCE_PAIRS)


def _count(value, name, lo, hi):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}]")
    return value


def _counts(rng, seed, num_train, num_test):
    train = _count(num_train, "num_train", 2, 10)
    test = _count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _size(rng, excluded=()):
    indices = [i for i, size in enumerate(SIZES) if size not in excluded]
    weights = WEIGHTS[indices]
    return SIZES[int(rng.choice(indices, p=weights / weights.sum()))]


def _offsets(rng, lattice, size, mode):
    rows, cols = lattice
    height, width = size
    offsets = np.empty((rows, cols, 2), dtype=np.int16)
    if mode in ("free", "boundary"):
        offsets[:, :, 0] = rng.integers(0, height, size=(rows, cols))
        offsets[:, :, 1] = rng.integers(0, width, size=(rows, cols))
    elif mode in ("aligned", "center"):
        point = (int(rng.integers(height)), int(rng.integers(width))) if mode == "aligned" else (height // 2, width // 2)
        offsets[:] = point
    elif mode == "row":
        for row in range(rows):
            offsets[row, :] = (int(rng.integers(height)), int(rng.integers(width)))
    elif mode == "column":
        for col in range(cols):
            offsets[:, col] = (int(rng.integers(height)), int(rng.integers(width)))
    elif mode == "edge":
        edge = [(r, c) for r in range(height) for c in range(width)
                if r in (0, height - 1) or c in (0, width - 1)]
        for row in range(rows):
            for col in range(cols):
                offsets[row, col] = edge[int(rng.integers(len(edge)))]
    else:
        raise ValueError("unknown offset mode")
    if mode == "boundary":
        row = int(rng.integers(rows))
        col = int(rng.integers(cols - 1))
        upper_first = bool(rng.integers(2))
        offsets[row, col] = (0 if upper_first else height - 1, width - 1)
        offsets[row, col + 1] = (height - 1 if upper_first else 0, 0)
    return offsets


def _colors(rng, background, count, repeat):
    available = [color for color in range(10) if color != background]
    if not repeat and count <= 9:
        return rng.choice(available, size=count, replace=False).astype(np.int8)
    distinct = int(rng.integers(2, min(7, count - 1, 9) + 1))
    palette = rng.choice(available, size=distinct, replace=False)
    colors = np.concatenate((palette, rng.choice(palette, size=count - distinct)))
    rng.shuffle(colors)
    return colors.astype(np.int8)


def _build_pair(rng, lattice, size, background, repeat, mode):
    rows, cols = lattice
    height, width = size
    colors = _colors(rng, background, rows * cols, repeat).reshape(lattice)
    offsets = _offsets(rng, lattice, size, mode)
    inp = np.full((rows * height, cols * width), background, dtype=np.int8)
    for row in range(rows):
        for col in range(cols):
            local_row, local_col = offsets[row, col]
            inp[row * height + local_row, col * width + local_col] = colors[row, col]
    return {"input": inp, "output": colors.copy()}


def _background(inp):
    values, counts = np.unique(inp, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _scan_differs(pair):
    inp, out = pair["input"], pair["output"]
    return not np.array_equal(inp[inp != _background(inp)], out.reshape(-1))


def _new_pair(rng, lattice, size, background, repeat, mode, seen, scan_difference=False):
    for _ in range(120):
        pair = _build_pair(rng, lattice, size, background, repeat, mode)
        inp = pair["input"]
        mask = inp != background
        if np.any(mask[1:] & mask[:-1]) or np.any(mask[:, 1:] & mask[:, :-1]):
            continue
        key = _key(inp)
        if key in seen or _source_copy(pair["input"], pair["output"]):
            continue
        if scan_difference and not _scan_differs(pair):
            continue
        seen.add(key)
        return pair
    raise RuntimeError("could not construct a fresh sector pair")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    lattice = ((3, 3), (2, 3), (3, 4))[int(rng.choice(3, p=(0.68, 0.16, 0.16)))]
    varying_background = bool(rng.integers(2))
    repeated_palette = lattice[0] * lattice[1] > 9 or bool(rng.integers(2))
    first_background = int(rng.integers(10))
    backgrounds = [first_background]
    if varying_background:
        backgrounds.append(int(rng.choice([c for c in range(10) if c != first_background])))
    else:
        backgrounds.append(first_background)
    for _ in range(train_count + test_count - 2):
        backgrounds.append(int(rng.integers(10)) if varying_background else first_background)

    train_sizes = [_size(rng)]
    train_sizes.append(_size(rng, train_sizes))
    for _ in range(train_count - 2):
        train_sizes.append(_size(rng))
    puzzle = {"train": [], "test": []}
    seen = set()
    for index, size in enumerate(train_sizes):
        mode = "free" if index == 0 else MODES[int(rng.integers(len(MODES)))]
        repeat = repeated_palette and (index == 0 or bool(rng.integers(4) != 0))
        puzzle["train"].append(_new_pair(rng, lattice, size, backgrounds[index],
                                         repeat, mode, seen, index == 0))

    test_modes = ("free", "free", "aligned", "center", "row", "column",
                  "edge", "edge", "boundary", "boundary")
    for index in range(test_count):
        relation = int(rng.integers(4))
        if relation == 0:
            size = train_sizes[int(rng.integers(train_count))]
        elif relation == 1:
            size = min(train_sizes, key=lambda s: s[0] * s[1])
        else:
            size = _size(rng, train_sizes)
        mode = test_modes[int(rng.integers(len(test_modes)))]
        repeat = repeated_palette and bool(rng.integers(5) != 0)
        puzzle["test"].append(_new_pair(rng, lattice, size,
                                        backgrounds[train_count + index],
                                        repeat, mode, seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def _assert_grid(grid, label):
    assert isinstance(grid, np.ndarray), f"{label} must be a NumPy array"
    assert grid.ndim == 2 and all(1 <= side <= 30 for side in grid.shape), f"{label} has invalid shape"
    assert np.issubdtype(grid.dtype, np.integer), f"{label} must have integer cells"
    assert np.min(grid) >= 0 and np.max(grid) <= 9, f"{label} has invalid colors"


def _inspect_pair(pair):
    inp, out = pair["input"], pair["output"]
    rows, cols = out.shape
    assert (rows, cols) in ((3, 3), (2, 3), (3, 4)), "invalid output lattice"
    assert inp.shape[0] % rows == 0 and inp.shape[1] % cols == 0, "unequal sectors"
    size = (inp.shape[0] // rows, inp.shape[1] // cols)
    assert size in SIZES, "invalid sector dimensions"
    bg = _background(inp)
    assert int(np.count_nonzero(inp != bg)) == rows * cols, "wrong marker count"
    mask = inp != bg
    assert not np.any(mask[1:] & mask[:-1]) and not np.any(mask[:, 1:] & mask[:, :-1]), "markers must remain single cells"
    for row in range(rows):
        for col in range(cols):
            sector = inp[row * size[0]:(row + 1) * size[0],
                         col * size[1]:(col + 1) * size[1]]
            markers = sector[sector != bg]
            assert markers.size == 1, "each sector needs one marker"
            assert markers[0] == out[row, col], "sector color was not copied"
    return bg, size, len(np.unique(out)) < out.size


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys"
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    train_count, test_count = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= train_count <= 10 and 1 <= test_count <= 4 and train_count > test_count
    seen = set()
    train_info, test_info = [], []
    lattice = None
    for split, info in (("train", train_info), ("test", test_info)):
        for index, pair in enumerate(puzzle[split]):
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            _assert_grid(inp, f"{split}[{index}].input")
            _assert_grid(out, f"{split}[{index}].output")
            assert lattice is None or out.shape == lattice, "output lattice changed"
            lattice = out.shape
            key = _key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            assert not _source_copy(inp, out), "source pair copied"
            info.append(_inspect_pair(pair))
    assert len({item[1] for item in train_info}) >= 2, "training needs two scales"
    assert any(_scan_differs(pair) for pair in puzzle["train"]), "training must distinguish sector order from global scan order"
    train_backgrounds = {item[0] for item in train_info}
    if len(train_backgrounds) == 1:
        assert all(item[0] in train_backgrounds for item in test_info), "unseen background convention"
    if any(item[2] for item in test_info):
        assert any(item[2] for item in train_info), "repeated payloads need training evidence"


def _signature(puzzle):
    return tuple((split, tuple((_key(p["input"]), _key(p["output"]))
                               for p in puzzle[split])) for split in ("train", "test"))


def self_test() -> None:
    def predict_visible(train, test_input):
        rows, cols = train[0]["output"].shape
        counts = np.bincount(test_input.reshape(-1), minlength=10)
        background = int(np.argmax(counts))
        height, width = test_input.shape[0] // rows, test_input.shape[1] // cols
        prediction = np.empty((rows, cols), dtype=np.int8)
        for row in range(rows):
            for col in range(cols):
                sector = test_input[row * height:(row + 1) * height,
                                    col * width:(col + 1) * width]
                colors = sector[sector != background]
                assert colors.size == 1
                prediction[row, col] = colors[0]
        return prediction

    def expect_bad(**kwargs):
        try:
            generate_puzzle(17, **kwargs)
        except ValueError:
            return
        raise AssertionError(f"expected ValueError for {kwargs}")

    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": True},
                   {"num_test": False}):
        expect_bad(**kwargs)
    for seed, train, test in ((100, 2, 1), (101, 10, 1), (102, 10, 4),
                              (103, None, 1), (104, None, 4),
                              (105, 2, None), (106, 10, None)):
        puzzle = generate_puzzle(seed, train, test)
        validate_generated_puzzle(puzzle)
        assert train is None or len(puzzle["train"]) == train
        assert test is None or len(puzzle["test"]) == test
        assert _signature(puzzle) == _signature(generate_puzzle(seed, train, test))

    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    source_train = [{"input": inp, "output": out} for inp, out in SOURCE_PAIRS[:3]]
    assert np.array_equal(predict_visible(source_train, SOURCE_PAIRS[3][0]), SOURCE_PAIRS[3][1])
    shapes, sizes, backgrounds, repeat_count = set(), set(), set(), 0
    rectangular = aligned = boundary = familiar = novel = 0
    variable_background = zero_marker = row_shared = column_shared = edge_large = 0
    combined = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        variable_background += len({_background(p["input"]) for p in puzzle["train"]}) > 1
        train_sizes = set()
        for pair in puzzle["train"]:
            out = pair["output"]
            train_sizes.add((pair["input"].shape[0] // out.shape[0],
                             pair["input"].shape[1] // out.shape[1]))
        for pair in puzzle["train"] + puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            bg, size, repeated = _inspect_pair(pair)
            shapes.add(out.shape)
            sizes.add(size)
            backgrounds.add(bg)
            repeat_count += repeated
            rectangular += inp.shape[0] != inp.shape[1]
            zero_marker += bg != 0 and 0 in out
            offsets = [tuple(np.argwhere(inp[r * size[0]:(r + 1) * size[0],
                                              c * size[1]:(c + 1) * size[1]] != bg)[0])
                       for r in range(out.shape[0]) for c in range(out.shape[1])]
            aligned += len(set(offsets)) == 1
            row_shared += len(set(offsets)) > 1 and all(
                len(set(offsets[r * out.shape[1]:(r + 1) * out.shape[1]])) == 1
                for r in range(out.shape[0]))
            column_shared += len(set(offsets)) > 1 and all(
                len({offsets[r * out.shape[1] + c] for r in range(out.shape[0])}) == 1
                for c in range(out.shape[1]))
            edge_large += size[0] > 2 and size[1] > 2 and all(
                r in (0, size[0] - 1) or c in (0, size[1] - 1) for r, c in offsets)
            boundary += any(offsets[r * out.shape[1] + c][1] == size[1] - 1
                            and offsets[r * out.shape[1] + c + 1][1] == 0
                            for r in range(out.shape[0]) for c in range(out.shape[1] - 1))
        for pair in puzzle["test"]:
            size = (pair["input"].shape[0] // pair["output"].shape[0],
                    pair["input"].shape[1] // pair["output"].shape[1])
            familiar += size in train_sizes
            novel += size not in train_sizes
            combined += size not in train_sizes and repeated and bg != 0 and pair["input"].shape[0] != pair["input"].shape[1]
            assert np.array_equal(predict_visible(puzzle["train"], pair["input"].copy()), pair["output"])
    assert shapes == {(3, 3), (2, 3), (3, 4)}
    assert len(sizes) >= 9 and len(backgrounds) >= 8
    assert repeat_count >= 30 and rectangular >= 35
    assert aligned >= 5 and boundary >= 8
    assert familiar >= 10 and novel >= 10
    assert variable_background >= 5 and zero_marker >= 15
    assert row_shared >= 5 and column_shared >= 5 and edge_large >= 5
    assert combined >= 5
