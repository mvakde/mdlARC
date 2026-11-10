"""Generate whole-grid vertical-reflection ARC puzzles."""

import numpy as np


_SOURCE_INPUTS = (
    ((0, 0, 0), (1, 0, 1), (1, 1, 0)),
    ((1, 1, 0), (1, 1, 1), (0, 0, 0)),
    ((0, 0, 0), (0, 0, 1), (0, 0, 1)),
    ((0, 0, 0), (0, 0, 1), (1, 0, 0)),
)
_SOURCE_KEYS = {(3, 3, tuple(v for row in grid for v in row)) for grid in _SOURCE_INPUTS}
_STYLES = ("scattered", "blocks", "lines", "stripes", "repeated", "bands",
           "single", "near_solid", "symmetric", "uniform")


def _key(grid):
    return (int(grid.shape[0]), int(grid.shape[1]), tuple(int(v) for v in grid.flat))


def _reflect_rows(grid):
    return np.concatenate((grid, grid[::-1]), axis=0)


def _integer(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _seed(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError("seed must be a non-negative integer")
    return int(value)


def _counts(seed, num_train, num_test):
    train = _integer(num_train, "num_train", 2, 10)
    test = _integer(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + (seed // 4) % (10 - test)
    elif test is None:
        test = 1 + (seed + train) % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _mode(seed):
    # Offset the demonstration schedule from the count schedule so that
    # source-sized puzzles can also have the source's 3/1 pair counts.
    return (3 * seed + seed // 6) % 6


def _shape(rng, informative=False):
    # The input height can reach 15 because its output has twice as many rows.
    kinds = ("source", "small", "tall", "wide", "medium", "large")
    kind = kinds[int(rng.integers(len(kinds)))]
    if kind == "source":
        shape = (3, 3)
    elif kind == "small":
        shape = (int(rng.integers(2, 6)), int(rng.integers(2, 8)))
    elif kind == "tall":
        shape = (int(rng.integers(7, 16)), int(rng.integers(1, 7)))
    elif kind == "wide":
        shape = (int(rng.integers(1, 6)), int(rng.integers(9, 31)))
    elif kind == "medium":
        shape = (int(rng.integers(4, 11)), int(rng.integers(4, 16)))
    else:
        shape = (int(rng.integers(9, 16)), int(rng.integers(15, 31)))
    if informative and (shape[0] < 3 or shape[1] < 3):
        return (3, max(3, shape[1]))
    return shape


def _palette(rng, pool=tuple(range(10)), minimum=1):
    maximum = min(5, len(pool))
    count = int(rng.integers(minimum, maximum + 1))
    return tuple(int(v) for v in rng.choice(pool, size=count, replace=False))


def _pattern(rng, shape, palette, style):
    height, width = shape
    count = len(palette)
    index = np.zeros(shape, dtype=np.int8)
    if style == "scattered":
        weights = rng.dirichlet(np.ones(count))
        index = rng.choice(count, size=shape, p=weights).astype(np.int8)
    elif style == "blocks":
        for _ in range(int(rng.integers(2, 6))):
            r0 = int(rng.integers(height))
            c0 = int(rng.integers(width))
            r1 = int(rng.integers(r0 + 1, height + 1))
            c1 = int(rng.integers(c0 + 1, width + 1))
            index[r0:r1, c0:c1] = int(rng.integers(1, count)) if count > 1 else 0
    elif style == "lines":
        for _ in range(int(rng.integers(1, 5))):
            color = int(rng.integers(1, count)) if count > 1 else 0
            if rng.integers(2):
                index[int(rng.integers(height)), :] = color
            else:
                index[:, int(rng.integers(width))] = color
    elif style == "stripes":
        axis = int(rng.integers(2))
        period = int(rng.integers(2, min(6, max(shape) + 1))) if max(shape) >= 2 else 1
        values = rng.integers(count, size=period)
        if count > 1:
            values[0], values[-1] = 0, count - 1
        if axis == 0:
            index = np.broadcast_to(values[np.arange(height) % period, None], shape).copy()
        else:
            index = np.broadcast_to(values[None, np.arange(width) % period], shape).copy()
    elif style == "repeated":
        row = rng.integers(count, size=width)
        if count > 1 and width > 1:
            row[0], row[-1] = 0, count - 1
        index = np.broadcast_to(row, shape).copy()
        if height > 1 and rng.integers(3):
            index[int(rng.integers(height)), int(rng.integers(width))] = int(rng.integers(count))
    elif style == "bands":
        offset = int(rng.integers(count))
        index = np.broadcast_to(((np.arange(height) * count // height + offset) % count)[:, None], shape).copy()
    elif style == "single":
        index[int(rng.integers(height)), int(rng.integers(width))] = count - 1
    elif style == "near_solid":
        for position in rng.choice(height * width, size=min(height * width, int(rng.integers(1, 5))), replace=False):
            index.flat[int(position)] = int(rng.integers(1, count)) if count > 1 else 0
    elif style == "symmetric":
        symmetry = int(rng.integers(1, 4))
        if symmetry & 1:
            top = rng.integers(count, size=((height + 1) // 2, width))
            index = np.concatenate((top, top[:height // 2][::-1]), axis=0)
        else:
            index = rng.integers(count, size=shape)
        if symmetry & 2:
            index[:, (width + 1) // 2:] = index[:, :width // 2][:, ::-1]
    elif style != "uniform":
        raise ValueError("unknown style")
    # A rare hue can occur exactly once even when other colors form regions.
    if count >= 3 and style in ("scattered", "blocks", "lines", "near_solid") and height * width >= count:
        if rng.integers(3) == 0:
            index[index == count - 1] = 0
            index.flat[int(rng.integers(height * width))] = count - 1
    colors = np.asarray(palette, dtype=np.int8)
    return colors[index]


def _informative(grid):
    return (grid.shape[0] >= 3 and grid.shape[1] >= 3
            and len(np.unique(grid)) >= 2
            and not np.array_equal(grid, grid[::-1])
            and not np.array_equal(grid, grid[:, ::-1]))


def _make_input(rng, shape, palette, style, used, informative=False):
    for _ in range(96):
        grid = _pattern(rng, shape, palette, style)
        key = _key(grid)
        if key in used or key in _SOURCE_KEYS or (informative and not _informative(grid)):
            continue
        return grid
    # Deterministic fallback for highly constrained small shapes and palettes.
    for _ in range(512):
        grid = rng.choice(palette, size=shape).astype(np.int8)
        key = _key(grid)
        if key not in used and key not in _SOURCE_KEYS and (not informative or _informative(grid)):
            return grid
    raise RuntimeError("unable to create a unique input")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    seed = _seed(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    mode = _mode(seed)
    # There is no learned color or size convention. These policies only organize
    # demonstrations: each output always depends solely on its own input.
    shared_shape = mode in (0, 1) or (mode == 5 and bool(rng.integers(2)))
    shared_palette = mode in (0, 2) or (mode == 5 and bool(rng.integers(2)))
    fixed_shape = (3, 3) if mode == 0 else _shape(rng, informative=True) if shared_shape else None
    fixed_palette = (0, 1) if mode == 0 else _palette(rng, minimum=2) if shared_palette else None
    used = set()
    train, test = [], []
    train_shapes, train_palettes = [], []
    for split, count in (("source_train", train_count), ("target_test", test_count)):
        for index in range(count):
            informative = split == "source_train" and index == 0
            if fixed_shape is not None:
                shape = fixed_shape
            else:
                shape = _shape(rng, informative=informative)
                if (split == "source_train" and index == 1) or (split == "target_test" and mode == 4):
                    for _ in range(24):
                        if shape not in train_shapes:
                            break
                        shape = _shape(rng, informative=False)
                if split == "source_train" and index == 1 and shape in train_shapes:
                    h, w = train_shapes[0]
                    shape = (h + 1 if h < 15 else h - 1, w)
                if split == "target_test" and mode == 4 and shape in train_shapes:
                    shape = next((h, w) for h in range(3, 16) for w in range(3, 31) if (h, w) not in train_shapes)
            if fixed_palette is not None:
                palette = fixed_palette
            elif mode == 4:
                palette = _palette(rng, pool=tuple(range(6)) if split == "source_train" else tuple(range(6, 10)), minimum=2)
            else:
                palette = _palette(rng, minimum=2)
            if split == "source_train" and index == 1 and not shared_palette:
                for _ in range(24):
                    if set(palette) != set(train_palettes[0]):
                        break
                    palette = _palette(rng, pool=tuple(range(6)) if mode == 4 else tuple(range(10)), minimum=2)
                if set(palette) == set(train_palettes[0]):
                    palette = tuple(v for v in range(10) if v not in train_palettes[0])[:2]
            if informative:
                style = ("scattered", "blocks", "lines", "bands")[int(rng.integers(4))]
            elif split == "target_test" and mode == 4:
                style = _STYLES[int(rng.integers(4, len(_STYLES)))]
            else:
                style = _STYLES[int(rng.integers(len(_STYLES)))]
            grid = _make_input(rng, shape, palette, style, used, informative)
            used.add(_key(grid))
            pair = {"input": grid, "output": _reflect_rows(grid)}
            if split == "source_train":
                train.append(pair)
                train_shapes.append(shape)
                train_palettes.append(tuple(int(x) for x in np.unique(grid)))
            else:
                test.append(pair)
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
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            h, w = inp.shape
            assert h <= 15 and out.shape == (2 * h, w)
            assert np.array_equal(out[:h], inp)
            assert np.array_equal(out[h:], inp[::-1])
            assert np.array_equal(out, out[::-1])
            assert set(int(v) for v in np.unique(inp)) == set(int(v) for v in np.unique(out))
            key = _key(inp)
            assert key not in seen and key not in _SOURCE_KEYS
            seen.add(key)
    assert _informative(puzzle["train"][0]["input"])
    training_shapes = {pair["input"].shape for pair in puzzle["train"]}
    training_palettes = {tuple(int(x) for x in np.unique(pair["input"])) for pair in puzzle["train"]}
    training_colors = {int(x) for pair in puzzle["train"] for x in np.unique(pair["input"])}
    for pair in puzzle["test"]:
        grid = pair["input"]
        if grid.shape not in training_shapes and any(int(x) not in training_colors for x in np.unique(grid)):
            assert len(training_shapes) >= 2 and len(training_palettes) >= 2


def _equal(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[k], b[k]) and a[k].dtype == b[k].dtype
        for a, b in zip(first[s], second[s]) for k in ("input", "output"))
        for s in ("train", "test"))


def self_test() -> None:
    assert _equal(generate_puzzle(123), generate_puzzle(123))
    assert _equal(generate_puzzle(123, 7, 3), generate_puzzle(123, 7, 3))
    assert not _equal(generate_puzzle(100), generate_puzzle(101))
    for n, m in ((2, 1), (10, 1), (10, 4)):
        puzzle = generate_puzzle(5000 + n + m, n, m)
        assert (len(puzzle["train"]), len(puzzle["test"])) == (n, m)
    for n, m in ((None, 1), (None, 4), (2, None), (10, None)):
        puzzle = generate_puzzle(6000 + (n or 0) + (m or 0), n, m)
        assert n is None or len(puzzle["train"]) == n
        assert m is None or len(puzzle["test"]) == m
    for n, m in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (3.0, 1)):
        try:
            generate_puzzle(1, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    train_counts, test_counts, shapes, signatures = set(), set(), set(), set()
    multicolor = shared_shape = variable_shape = variable_palette = genuine_palette_shift = 0
    symmetric = single_mark = uniform = rare_hue = 0
    source_like = full_row = repeated_row = unusual_corner = 0
    wide = tall = test_new_shape = test_new_color = combined_shift = 0
    easy_test = one_row_test = 0
    corners = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        ts = {p["input"].shape for p in puzzle["train"]}
        all_shapes = {p["input"].shape for s in ("train", "test") for p in puzzle[s]}
        train_grids = [p["input"] for p in puzzle["train"]]
        test_grids = [p["input"] for p in puzzle["test"]]
        train_colors = {int(v) for x in train_grids for v in np.unique(x)}
        new_shape = any(x.shape not in ts for x in test_grids)
        new_color = any(any(int(v) not in train_colors for v in np.unique(x)) for x in test_grids)
        test_new_shape += new_shape
        test_new_color += new_color
        combined_shift += new_shape and new_color
        source_like += all(x.shape == (3, 3) and set(np.unique(x)) <= {0, 1}
                           for x in train_grids + test_grids)
        easy_test += any((x.size - max(np.unique(x, return_counts=True)[1])) / x.size
                         < min((t.size - max(np.unique(t, return_counts=True)[1])) / t.size
                               for t in train_grids) for x in test_grids)
        one_row_test += any(x.shape[0] == 1 for x in test_grids)
        shared_shape += len(all_shapes) == 1
        variable_shape += len(all_shapes) > 1
        palettes = {tuple(int(v) for v in np.unique(p["input"])) for s in ("train", "test") for p in puzzle[s]}
        variable_palette += len(palettes) > 1
        genuine_palette_shift += len({tuple(int(v) for v in np.unique(p["input"]))
                                      for s in ("train", "test") for p in puzzle[s]
                                      if len(np.unique(p["input"])) >= 2}) > 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                h, w = inp.shape
                assert out.shape == (2 * h, w)
                assert np.array_equal(out[:h], inp)
                assert np.array_equal(out[h:], np.flipud(inp))
                assert _key(inp) not in _SOURCE_KEYS
                shapes.add(inp.shape)
                signatures.add(_key(inp))
                multicolor += len(np.unique(inp)) >= 3
                symmetric += np.array_equal(inp, inp[::-1]) or np.array_equal(inp, inp[:, ::-1])
                uniform += len(np.unique(inp)) == 1
                values, counts = np.unique(inp, return_counts=True)
                single_mark += len(values) == 2 and int(min(counts)) == 1
                rare_hue += len(values) >= 3 and int(min(counts)) == 1
                unusual_corner += int(inp[0, 0]) != int(values[np.argmax(counts)])
                full_row += any(len(set(int(v) for v in row)) == 1 for row in inp)
                repeated_row += len({tuple(int(v) for v in row) for row in inp}) < h
                wide += w >= 2 * h
                tall += h >= 2 * w
                corners.add(int(inp[0, 0]))
        if _mode(seed) == 4:
            assert len(ts) >= 2
            assert len({tuple(int(v) for v in np.unique(p["input"])) for p in puzzle["train"]}) >= 2
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(shapes) >= 40 and len(signatures) >= 250
    assert shared_shape >= 5 and variable_shape >= 20 and variable_palette >= 20
    assert genuine_palette_shift >= 20
    assert multicolor >= 20 and symmetric >= 10 and uniform >= 3 and single_mark >= 3
    assert rare_hue >= 10 and unusual_corner >= 20
    assert source_like >= 5 and full_row >= 50 and repeated_row >= 50
    assert wide >= 20 and tall >= 10
    assert test_new_shape >= 10 and test_new_color >= 8 and combined_shift >= 5
    assert easy_test >= 5 and one_row_test >= 1
    assert len(corners) >= 8
