"""Whole-grid two-axis reflection puzzles."""

import numpy as np


_SOURCE = frozenset((
    ((3, 3), (3, 3, 1, 1, 3, 0, 0, 2, 2)),
    ((3, 3), (3, 3, 3, 0, 2, 2, 1, 1, 0)),
    ((3, 3), (2, 1, 0, 0, 2, 3, 0, 3, 0)),
    ((3, 3), (1, 1, 0, 0, 3, 2, 3, 3, 0)),
))
_MODES = ("asymmetric", "left_right", "top_bottom", "half_turn", "both")
_STYLES = ("irregular", "runs", "blocks", "bands", "repeated", "sparse", "checker")


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _key(grid):
    return (tuple(grid.shape), tuple(int(v) for v in grid.flat))


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} is out of range")
    return value


def _counts(seed, num_train, num_test):
    train = None if num_train is None else _integer(num_train, "num_train", 2, 10)
    test = None if num_test is None else _integer(num_test, "num_test", 1, 4)
    if test is None and train is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        minimum = max(2, test + 1)
        train = minimum + seed % (11 - minimum)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _expand(grid):
    top = np.concatenate((grid, grid[:, ::-1]), axis=1)
    return np.concatenate((top, top[::-1, :]), axis=0).astype(np.int8, copy=False)


def _manual_expand(grid):
    height, width = grid.shape
    result = np.empty((2 * height, 2 * width), dtype=grid.dtype)
    for row in range(2 * height):
        for col in range(2 * width):
            result[row, col] = grid[min(row, 2 * height - 1 - row),
                                    min(col, 2 * width - 1 - col)]
    return result


def _mode(grid):
    lr = np.array_equal(grid, grid[:, ::-1])
    tb = np.array_equal(grid, grid[::-1, :])
    rt = np.array_equal(grid, grid[::-1, ::-1])
    if lr and tb:
        return "both"
    if lr:
        return "left_right"
    if tb:
        return "top_bottom"
    if rt:
        return "half_turn"
    return "asymmetric"


def _shape(rng, profile, anchor, regime, first):
    if profile == "source":
        return (3, 3)
    if first:
        if rng.random() < 0.27:
            return (3, 3)
        return (int(rng.integers(3, 9)), int(rng.integers(3, 9)))
    roll = rng.random()
    if roll < 0.13:
        return anchor
    if roll < 0.37:
        return (int(np.clip(anchor[0] + rng.integers(-2, 3), 2, 15)),
                int(np.clip(anchor[1] + rng.integers(-2, 3), 2, 15)))
    if regime == "target_test" and rng.random() < 0.10:
        if rng.random() < 0.5:
            return (1, int(rng.integers(3, 16)))
        return (int(rng.integers(3, 16)), 1)
    if rng.random() < 0.22:
        return (3, 3)
    if rng.random() < 0.43:
        return (int(rng.integers(2, 7)), int(rng.integers(2, 9)))
    return (int(rng.integers(2, 16)), int(rng.integers(2, 16)))


def _desired_mode(rng, profile, regime, first):
    if first or profile == "source":
        return "asymmetric"
    if rng.random() < (0.63 if regime == "source_train" else 0.53):
        return "asymmetric"
    return str(rng.choice(_MODES[1:], p=(0.30, 0.30, 0.25, 0.15)))


def _representative(row, col, height, width, mode):
    if mode == "left_right":
        return (row, min(col, width - 1 - col))
    if mode == "top_bottom":
        return (min(row, height - 1 - row), col)
    if mode == "half_turn":
        return min((row, col), (height - 1 - row, width - 1 - col))
    if mode == "both":
        return (min(row, height - 1 - row), min(col, width - 1 - col))
    return (row, col)


def _texture(rng, height, width, colors, style):
    if style == "irregular":
        concentration = float(rng.choice((0.45, 1.0, 3.0)))
        return rng.choice(colors, size=(height, width),
                          p=rng.dirichlet(np.full(colors, concentration))).astype(np.int8)
    labels = np.full((height, width), int(rng.integers(colors)), dtype=np.int8)
    if style == "runs":
        for row in range(height):
            col = 0
            while col < width:
                span = int(rng.integers(1, min(5, width - col) + 1))
                labels[row, col:col + span] = rng.integers(colors)
                col += span
    elif style == "bands":
        if rng.random() < 0.5:
            labels[:] = rng.integers(colors, size=(height, 1))
        else:
            labels[:] = rng.integers(colors, size=(1, width))
    elif style == "repeated":
        if rng.random() < 0.5:
            period = int(rng.integers(1, min(height, 4) + 1))
            unit = rng.integers(colors, size=(period, width))
            labels[:] = unit[np.arange(height) % period, :]
        else:
            period = int(rng.integers(1, min(width, 4) + 1))
            unit = rng.integers(colors, size=(height, period))
            labels[:] = unit[:, np.arange(width) % period]
    elif style == "checker":
        labels[np.indices((height, width)).sum(axis=0) % 2 == 0] = rng.integers(colors)
    elif style == "sparse":
        for _ in range(max(1, int(height * width * rng.uniform(0.04, 0.19)))):
            labels[int(rng.integers(height)), int(rng.integers(width))] = rng.integers(colors)
    if style in ("blocks", "bands", "checker", "repeated"):
        for _ in range(int(rng.integers(1, max(3, height * width // 6 + 2)))):
            row, col = int(rng.integers(height)), int(rng.integers(width))
            end_row = min(height, row + int(rng.integers(1, min(5, height - row) + 1)))
            end_col = min(width, col + int(rng.integers(1, min(5, width - col) + 1)))
            labels[row:end_row, col:end_col] = rng.integers(colors)
    return labels


def _input(rng, shape, palette, mode, style, used):
    height, width = shape
    representatives = sorted({_representative(row, col, height, width, mode)
                              for row in range(height) for col in range(width)})
    colors = min(len(palette), len(representatives))
    palette = palette[:colors]
    for _ in range(128):
        labels = _texture(rng, height, width, colors, style)
        slots = rng.permutation(len(representatives))
        for label, slot in enumerate(slots[:colors]):
            labels[representatives[int(slot)]] = label
        symmetric = np.empty_like(labels)
        for row in range(height):
            for col in range(width):
                symmetric[row, col] = labels[_representative(row, col, height, width, mode)]
        grid = np.array(palette[symmetric], dtype=np.int8, copy=True)
        if _key(grid) in used or _key(grid) in _SOURCE:
            continue
        if mode == "asymmetric" and _mode(grid) != mode:
            continue
        return grid
    raise RuntimeError("could not construct a distinct grid")


def _fallback_input(index, split, palette, used):
    width = 2 + index + (10 if split == "test" else 0)
    shape = (3, width)
    for variant in range(len(used) + 5):
        labels = np.empty(shape, dtype=np.int8)
        labels[0, 0] = 0
        number = variant
        for row in range(3):
            for col in range(width):
                if row == 0 and col == 0:
                    continue
                labels[row, col] = 1 + ((row + col + number % 3) % 3)
                number //= 3
        grid = np.array(palette[labels], dtype=np.int8, copy=True)
        if _key(grid) not in used and _key(grid) not in _SOURCE:
            return grid
    raise RuntimeError("deterministic fallback exhausted")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    seed = _integer(seed, "seed", 0, 2**63 - 1)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    profile = "source" if rng.random() < 0.19 else "varied"
    shared_palette = np.sort(rng.choice(10, size=4, replace=False)).astype(np.int8)
    anchor = _shape(rng, profile, (3, 3), "source_train", True)
    puzzle = {"train": [], "test": []}
    used = set()
    for split, count in (("train", train_count), ("test", test_count)):
        regime = "source_train" if split == "train" else "target_test"
        for index in range(count):
            first = split == "train" and index == 0
            for attempt in range(24):
                shape = anchor if first else _shape(rng, profile, anchor, regime, False)
                mode = _desired_mode(rng, profile, regime, first)
                if profile == "source" or rng.random() < 0.30:
                    palette = shared_palette
                else:
                    colors = int(rng.integers(2, min(8, shape[0] * shape[1]) + 1))
                    palette = np.sort(rng.choice(10, size=colors, replace=False)).astype(np.int8)
                try:
                    grid = _input(rng, shape, palette, mode, str(rng.choice(_STYLES)), used)
                    break
                except RuntimeError:
                    pass
            else:
                grid = _fallback_input(index, split, shared_palette, used)
            used.add(_key(grid))
            puzzle[split].append({"input": grid, "output": _expand(grid)})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "splits")
    train, test = puzzle["train"], puzzle["test"]
    _check(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "counts")
    seen = set()
    for split, pairs in (("train", train), ("test", test)):
        for index, pair in enumerate(pairs):
            where = f"{split}[{index}]"
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, where + " keys")
            source, result = pair["input"], pair["output"]
            for grid in (source, result):
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2, where + " grid")
                _check(np.issubdtype(grid.dtype, np.integer), where + " dtype")
                _check(all(1 <= n <= 30 for n in grid.shape), where + " shape")
                _check(int(grid.min()) >= 0 and int(grid.max()) <= 9, where + " color")
            height, width = source.shape
            _check(result.shape == (2 * height, 2 * width), where + " doubling")
            _check(np.array_equal(result, _manual_expand(source)), where + " reflection")
            for color in np.unique(source):
                _check(np.count_nonzero(result == color) == 4 * np.count_nonzero(source == color),
                       where + " preservation")
            key = _key(source)
            _check(key not in seen and key not in _SOURCE, where + " duplicate/source")
            seen.add(key)
    first = train[0]["input"]
    _check(min(first.shape) >= 2 and len(np.unique(first)) >= 2, "first training evidence")
    _check(_mode(first) == "asymmetric", "first training symmetry")
    height, width = first.shape
    output = train[0]["output"]
    quadrants = (output[:height, :width], output[:height, width:],
                 output[height:, :width], output[height:, width:])
    _check(len({_key(part) for part in quadrants}) == 4, "four distinct quadrants")


def _same(a, b):
    return all(len(a[split]) == len(b[split]) and all(
        pa[name].dtype == pb[name].dtype and np.array_equal(pa[name], pb[name])
        for pa, pb in zip(a[split], b[split]) for name in ("input", "output"))
        for split in ("train", "test"))


def _normalized(grid):
    labels = {}
    return (grid.shape, tuple(labels.setdefault(int(v), len(labels)) for v in grid.flat))


def self_test() -> None:
    for seed, train, test in ((12345, None, None), (54321, 7, 3),
                              (10000, 2, 1), (10001, 10, 1), (10002, 10, 4),
                              (20000, None, 1), (20001, None, 4),
                              (20002, 2, None), (20003, 10, None)):
        a, b = generate_puzzle(seed, train, test), generate_puzzle(seed, train, test)
        _check(_same(a, b), "determinism")
        validate_generated_puzzle(a)
        if train is not None:
            _check(len(a["train"]) == train, "explicit train count")
        if test is not None:
            _check(len(a["test"]) == test, "explicit test count")
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(30000, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _check(not _same(puzzles[0], puzzles[1]), "different seeds")
    shapes, palettes, sizes, modes, layouts = set(), set(), set(), set(), set()
    shape_modes = {}
    mixed_shape = mixed_palette = asym_test = sym_train = 0
    source_like = one_dimensional_test = novel_test_color = mixed_test_mode = 0
    both_train = both_test = combined = large = repeated_row = dominant_half = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_grids = [pair["input"] for pair in puzzle["train"]]
        test_grids = [pair["input"] for pair in puzzle["test"]]
        grids = train_grids + test_grids
        mixed_shape += len({g.shape for g in grids}) > 1
        mixed_palette += len({tuple(np.unique(g)) for g in grids}) > 1
        asym_test += any(_mode(g) == "asymmetric" for g in test_grids)
        sym_train += any(_mode(g) != "asymmetric" for g in train_grids[1:])
        source_like += all(g.shape == (3, 3) and len(np.unique(g)) == 4
                           and _mode(g) == "asymmetric" for g in grids)
        one_dimensional_test += any(1 in g.shape for g in test_grids)
        train_colors = set().union(*(set(np.unique(g)) for g in train_grids))
        novel_test_color += any(set(np.unique(g)) - train_colors for g in test_grids)
        mixed_test_mode += len({_mode(g) for g in test_grids}) > 1
        both_train += any(_mode(g) == "both" for g in train_grids)
        both_test += any(_mode(g) == "both" for g in test_grids)
        combined += (len({g.shape for g in grids}) > 1
                     and len({tuple(np.unique(g)) for g in grids}) > 1
                     and len({_mode(g) for g in grids}) > 1)
        large += any(max(g.shape) > 8 for g in grids)
        for grid in grids:
            shapes.add(grid.shape)
            palette = tuple(int(v) for v in np.unique(grid))
            palettes.add(palette)
            sizes.add(len(palette))
            modes.add(_mode(grid))
            shape_modes.setdefault(grid.shape, set()).add(_mode(grid))
            layouts.add(_normalized(grid))
            repeated_row += len({tuple(row) for row in grid}) < grid.shape[0]
            _, color_counts = np.unique(grid, return_counts=True)
            dominant_half += color_counts.max() / grid.size >= 0.5
    _check(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _check(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    _check(len(shapes) >= 35 and any(max(s) > 8 for s in shapes), "shape diversity")
    _check(mixed_shape >= 20 and mixed_palette >= 20, "within-puzzle diversity")
    _check(len(palettes) >= 50 and {2, 3, 4}.issubset(sizes) and max(sizes) >= 6,
           "palette diversity")
    _check(set(_MODES).issubset(modes) and asym_test >= 12 and sym_train >= 5,
           "symmetry diversity")
    _check(len(layouts) >= 120, "layout diversity")
    _check(source_like >= 3 and one_dimensional_test >= 1 and novel_test_color >= 3,
           "source, one-dimensional, and color-shift coverage")
    _check(mixed_test_mode >= 5 and both_train >= 2 and both_test >= 2,
           "mixed and both-axis symmetry coverage")
    _check(combined >= 15 and large >= 10 and
           sum(len(value) > 1 for value in shape_modes.values()) >= 5,
           "combined shape and symmetry coverage")
    _check(repeated_row >= 30 and dominant_half >= 30,
           "repetition and unequal-area coverage")
