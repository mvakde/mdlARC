"""ARC puzzles with uniform cellwise 2x enlargement."""
import numpy as np


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _key(a):
    return tuple(a.shape), tuple(int(v) for v in a.flat)


def _scale2(a):
    return np.repeat(np.repeat(a, 2, axis=0), 2, axis=1).astype(np.int8)


_SOURCE_ROWS = (
    ((2, 1), (3, 1)),
    ((0, 5, 1), (5, 5, 5), (2, 5, 0)),
    ((2, 0, 3, 0), (2, 1, 3, 0), (0, 0, 3, 3), (0, 0, 3, 5)),
    ((2, 0, 0, 7, 8), (2, 1, 1, 0, 0), (0, 5, 6, 6, 0),
     (3, 5, 6, 0, 0), (0, 5, 0, 0, 0)),
)
_SOURCE_PAIRS = {(_key(a), _key(_scale2(a))) for a in
                 (np.asarray(rows, dtype=np.int8) for rows in _SOURCE_ROWS)}


def _optional_count(value, name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    return int(value)


def _counts(rng, seed, num_train, num_test):
    train = _optional_count(num_train, "num_train")
    test = _optional_count(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be between 2 and 10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be between 1 and 4")
    if train is None and test is None:
        train, test = ((3, 1), (2, 1), (4, 2), (5, 1), (6, 3), (3, 1),
                       (7, 2), (8, 4), (9, 3), (3, 1), (10, 1), (10, 4))[seed % 12]
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _transitions(a):
    return (int(np.count_nonzero(a[:, 1:] != a[:, :-1])),
            int(np.count_nonzero(a[1:, :] != a[:-1, :])))


def _side(rng):
    roll = int(rng.integers(100))
    if roll < 8:
        return 1
    if roll < 16:
        return 15
    return int(rng.integers(2, 15))


def _shapes(rng, seed, nt, nv):
    train = [(_side(rng), _side(rng)) for _ in range(nt)]
    test = [(_side(rng), _side(rng)) for _ in range(nv)]
    train[0] = tuple(max(2, side) for side in train[0])
    regime = seed % 7
    if regime == 0:
        for i in range(min(3, nt)):
            train[i] = (i + 2, i + 2)
        test[0] = (min(5, nt + 2),) * 2
    elif regime == 1:
        test[0] = train[int(rng.integers(nt))]
    elif regime == 2:
        test[0] = (int(rng.integers(1, max(h for h, _ in train))),
                   int(rng.integers(1, max(w for _, w in train))))
    elif regime == 3:
        train = [(min(h, 14), w) for h, w in train]
        test[0] = (int(rng.integers(max(h for h, _ in train) + 1, 16)),
                   int(rng.integers(1, max(w for _, w in train) + 1)))
    elif regime == 4:
        train = [(h, min(w, 14)) for h, w in train]
        test[0] = (int(rng.integers(1, max(h for h, _ in train) + 1)),
                   int(rng.integers(max(w for _, w in train) + 1, 16)))
    elif regime == 5:
        test[0] = (1, 1)
    return train, test


def _capacity(mode, h, w):
    if mode in (0, 9):
        return h * w
    if mode == 1:
        return h
    if mode == 2:
        return w
    if mode == 3:
        return h + w - 1
    if mode == 4:
        return 2 if h * w >= 2 else 0
    if mode == 5:
        return min(6, h * w) if h * w >= 2 else 0
    if mode == 6:
        return 2 if min(h, w) >= 3 else 0
    if mode == 7:
        return ((h + 1) // 2) * ((w + 1) // 2)
    if mode == 8:
        return 4 if min(h, w) >= 2 else 0
    raise ValueError("unknown layout")


def _pattern(rng, h, w, palette, mode):
    k = len(palette)
    if k == 1:
        return np.full((h, w), palette[0], dtype=np.int8)
    _require(k <= _capacity(mode, h, w), "palette exceeds layout capacity")
    rows, cols = np.indices((h, w))
    if mode == 0:  # Dense, irregular, and naturally uses the full palette.
        indices = np.concatenate((np.arange(k), rng.integers(k, size=h * w - k)))
        rng.shuffle(indices)
        indices = indices.reshape(h, w)
    elif mode in (1, 2):  # Clean horizontal or vertical bands.
        length = h if mode == 1 else w
        band = int(rng.integers(1, length // k + 1))
        coordinate = rows if mode == 1 else cols
        indices = (coordinate // band + int(rng.integers(k))) % k
    elif mode == 3:  # Reflected cyclic diagonals.
        rr = rows if rng.integers(2) else h - 1 - rows
        cc = cols if rng.integers(2) else w - 1 - cols
        indices = (rr + cc + int(rng.integers(k))) % k
    elif mode == 4:  # Asymmetric four-connected region.
        indices = np.zeros((h, w), dtype=np.int64)
        region = {(int(rng.integers(h)), int(rng.integers(w)))}
        target = int(rng.integers(1, max(2, min(h * w, 1 + 3 * h * w // 5))))
        while len(region) < target:
            frontier = set()
            for r, c in region:
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region:
                        frontier.add((nr, nc))
            if not frontier:
                break
            region.add(sorted(frontier)[int(rng.integers(len(frontier)))])
        for r, c in region:
            indices[r, c] = 1
    elif mode == 5:  # Isolated or separated marks on a plain field.
        indices = np.zeros((h, w), dtype=np.int64)
        count = int(rng.integers(k - 1, min(h * w - 1, max(k, h * w // 4)) + 1))
        locations = rng.choice(h * w, size=count, replace=False)
        colors = np.concatenate((np.arange(1, k),
                                 rng.integers(1, k, size=count - (k - 1))))
        rng.shuffle(colors)
        indices.flat[locations] = colors
    elif mode == 6:  # Contour with an intact interior hole.
        indices = np.zeros((h, w), dtype=np.int64)
        top, left = int(rng.integers(h - 2)), int(rng.integers(w - 2))
        bottom = int(rng.integers(top + 3, h + 1))
        right = int(rng.integers(left + 3, w + 1))
        indices[top:bottom, left:right] = 1
        indices[top + 1:bottom - 1, left + 1:right - 1] = 0
    elif mode == 7:  # Input already has same-color 2x2 tiles.
        ch, cw = (h + 1) // 2, (w + 1) // 2
        coarse = np.concatenate((np.arange(k), rng.integers(k, size=ch * cw - k)))
        rng.shuffle(coarse)
        indices = np.repeat(np.repeat(coarse.reshape(ch, cw), 2, axis=0),
                            2, axis=1)[:h, :w]
    elif mode == 8:  # Clean rectangular block partition.
        split_r, split_c = int(rng.integers(1, h)), int(rng.integers(1, w))
        labels = np.arange(4) % k
        rng.shuffle(labels)
        indices = labels[(rows >= split_r) * 2 + (cols >= split_c)]
    elif mode == 9:  # Uneven nearest-center territories.
        centers = rng.choice(h * w, size=k, replace=False)
        cr, cc = centers // w, centers % w
        indices = np.argmin(abs(rows[:, :, None] - cr) +
                            abs(cols[:, :, None] - cc), axis=2)
    else:
        raise ValueError("unknown layout")
    return np.asarray(palette, dtype=np.int8)[indices].astype(np.int8, copy=False)


def _palette(rng, colors, count):
    return tuple(int(x) for x in rng.choice(colors, size=count, replace=False))


def _test_palette(rng, relation, area, train_colors, largest_train):
    old = tuple(sorted(train_colors))
    new = tuple(c for c in range(10) if c not in train_colors)
    if relation == 0:
        return _palette(rng, old, min(area, largest_train - 1))
    if relation == 1:
        return _palette(rng, old, min(area, largest_train))
    if relation == 2:
        return _palette(rng, new, min(area, len(new), int(rng.integers(1, 5))))
    if relation == 3:
        if area == 1:
            return _palette(rng, new, 1)
        old_count = min(len(old), int(rng.integers(1, min(3, area - 1) + 1)))
        new_count = min(len(new), area - old_count, int(rng.integers(1, 4)))
        return _palette(rng, old, old_count) + _palette(rng, new, new_count)
    if area > len(old):
        extra = min(len(new), area - len(old), int(rng.integers(1, 4)))
        return old + _palette(rng, new, extra)
    return _palette(rng, old, max(1, min(area, len(old))))


def _make_input(rng, shape, palette, anchor):
    h, w = shape
    modes = [m for m in range(10) if len(palette) <= _capacity(m, h, w)]
    if anchor:
        modes = [m for m in modes if m in (0, 3, 7, 9)]
    for _ in range(12):
        mode = modes[int(rng.integers(len(modes)))]
        grid = _pattern(rng, h, w, palette, mode)
        if not anchor or all(x > 0 for x in _transitions(grid)):
            return grid
    return _pattern(rng, h, w, palette, 0)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    nt, nv = _counts(rng, seed, num_train, num_test)
    train_shapes, test_shapes = _shapes(rng, seed, nt, nv)
    # No color has a learned role. This pool leaves room for new test colors.
    train_pool = _palette(rng, tuple(range(10)), int(rng.integers(3, 8)))
    puzzle = {"train": [], "test": []}
    used_inputs = set()
    for split, shapes in (("train", train_shapes), ("test", test_shapes)):
        for index, initial_shape in enumerate(shapes):
            shape = initial_shape
            for attempt in range(48):
                h, w = shape
                area = h * w
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
                if split == "train":
                    if index == 0:
                        count = min(area, len(train_pool), int(pair_rng.integers(3, 5)))
                    else:
                        count = int(pair_rng.integers(1, min(6, len(train_pool), area) + 1))
                    palette = _palette(pair_rng, train_pool, count)
                elif index == 0:
                    colors = frozenset(int(c) for pair in puzzle["train"]
                                       for c in np.unique(pair["input"]))
                    largest = max(len(np.unique(pair["input"])) for pair in puzzle["train"])
                    palette = _test_palette(pair_rng, seed % 5, area, colors, largest)
                else:
                    count = int(pair_rng.integers(1, min(10, area) + 1))
                    palette = _palette(pair_rng, tuple(range(10)), count)
                grid = _make_input(pair_rng, shape, palette,
                                   anchor=split == "train" and index == 0)
                ik = _key(grid)
                if ik not in used_inputs and (ik, _key(_scale2(grid))) not in _SOURCE_PAIRS:
                    break
                if attempt == 23:
                    shape = (max(2, h), max(2, w))
            else:
                raise RuntimeError("could not construct a distinct input")
            used_inputs.add(ik)
            puzzle[split].append({"input": grid, "output": _scale2(grid)})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "puzzle must have exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "splits must be lists")
    nt, nv = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv, "illegal pair counts")
    seen_inputs, seen_pairs = set(), set()
    demonstrated = False
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "pair must have exactly input and output")
            a, b = pair["input"], pair["output"]
            for grid in (a, b):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2,
                         "grid must be a 2D NumPy array")
                _require(all(1 <= side <= 30 for side in grid.shape),
                         "grid dimensions must be 1..30")
                _require(np.issubdtype(grid.dtype, np.integer), "grid must be integer")
                _require(0 <= int(grid.min()) and int(grid.max()) <= 9,
                         "grid colors must be 0..9")
            _require(a.shape[0] <= 15 and a.shape[1] <= 15,
                     "input cannot be doubled within 30 cells")
            _require(b.shape == (2 * a.shape[0], 2 * a.shape[1]),
                     "output dimensions must double on both axes")
            _require(np.array_equal(b, np.repeat(np.repeat(a, 2, axis=0), 2, axis=1)),
                     "each cell must become its own 2x2 block")
            ik, pk = _key(a), (_key(a), _key(b))
            _require(ik not in seen_inputs and pk not in seen_pairs,
                     "duplicate input or pair")
            _require(pk not in _SOURCE_PAIRS, "exact source pair copied")
            seen_inputs.add(ik)
            seen_pairs.add(pk)
            if split == "train":
                horizontal, vertical = _transitions(a)
                demonstrated |= (a.shape[0] >= 2 and a.shape[1] >= 2
                                 and len(np.unique(a)) >= 3
                                 and horizontal > 0 and vertical > 0)
    _require(demonstrated, "training lacks multicolor two-axis evidence")


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[s])
                 for s in ("train", "test"))


def _layout_key(grid):
    names, normalized = {}, []
    for color in (int(v) for v in grid.flat):
        if color not in names:
            names[color] = len(names)
        normalized.append(names[color])
    return tuple(grid.shape), tuple(normalized)


def _predict_from_training(train_pairs, test_input):
    """Infer the only block size and cell mapping visible in demonstrations."""
    first = train_pairs[0]
    a, b = first["input"], first["output"]
    _require(b.shape[0] % a.shape[0] == 0 and b.shape[1] % a.shape[1] == 0,
             "training dimensions do not give integer scale factors")
    fh, fw = b.shape[0] // a.shape[0], b.shape[1] // a.shape[1]
    for pair in train_pairs:
        a, b = pair["input"], pair["output"]
        _require(b.shape == (fh * a.shape[0], fw * a.shape[1]),
                 "training scale factors disagree")
        for dr in range(fh):
            for dc in range(fw):
                _require(np.array_equal(b[dr::fh, dc::fw], a),
                         "training does not show cellwise replication")
    return np.kron(test_input, np.ones((fh, fw), dtype=np.int8))


def self_test() -> None:
    explicit = generate_puzzle(12345, 7, 3)
    _require(_puzzle_key(explicit) == _puzzle_key(generate_puzzle(12345, 7, 3)),
             "fixed seed is nondeterministic")
    _require(_puzzle_key(explicit) != _puzzle_key(generate_puzzle(12346, 7, 3)),
             "different seeds produced the same puzzle")
    for i, counts in enumerate(((2, 1), (10, 1), (10, 4),
                                (None, 1), (None, 4), (2, None), (10, None))):
        puzzle = generate_puzzle(20000 + i, *counts)
        validate_generated_puzzle(puzzle)
        for actual, requested in zip((len(puzzle["train"]), len(puzzle["test"])), counts):
            _require(requested is None or actual == requested, "requested count ignored")
    for counts in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(22000, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    corrupted = generate_puzzle(75)
    corrupted["test"][0]["output"] = corrupted["test"][0]["output"].copy()
    cell = corrupted["test"][0]["output"]
    cell[0, 0] = (int(cell[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted a wrong output block")
    train_counts, test_counts, puzzles = set(), set(), set()
    shapes, layouts, palettes, aspect = set(), set(), set(), set()
    size_cases, color_cases = set(), set()
    one_dimensional = unit = max_side = zero = 0
    within_shapes = within_palettes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        training_copy = [{"input": p["input"].copy(), "output": p["output"].copy()}
                         for p in puzzle["train"]]
        for test_pair in puzzle["test"]:
            predicted = _predict_from_training(training_copy, test_pair["input"].copy())
            _require(np.array_equal(predicted, test_pair["output"]),
                     "training-only prediction differs from the test output")
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        puzzles.add(_puzzle_key(puzzle))
        pairs = puzzle["train"] + puzzle["test"]
        within_shapes += len({p["input"].shape for p in pairs}) > 1
        within_palettes += len({frozenset(int(c) for c in np.unique(p["input"]))
                                for p in pairs}) > 1
        for pair in pairs:
            a, b = pair["input"], pair["output"]
            _require(all(np.array_equal(b[dr::2, dc::2], a)
                         for dr in (0, 1) for dc in (0, 1)),
                     "independent quadrant replication check failed")
            shapes.add(a.shape)
            layouts.add(_layout_key(a))
            palettes.add(frozenset(int(c) for c in np.unique(a)))
            aspect.add("square" if a.shape[0] == a.shape[1] else
                       "tall" if a.shape[0] > a.shape[1] else "wide")
            one_dimensional += min(a.shape) == 1
            unit += a.shape == (1, 1)
            max_side += 15 in a.shape
            zero += bool(np.any(a == 0))
        train_shapes = [p["input"].shape for p in puzzle["train"]]
        test_shape = puzzle["test"][0]["input"].shape
        mh, mw = max(h for h, _ in train_shapes), max(w for _, w in train_shapes)
        if test_shape in train_shapes:
            size_cases.add("same")
        if test_shape[0] < mh and test_shape[1] < mw:
            size_cases.add("smaller")
        if test_shape[0] > mh and test_shape[1] <= mw:
            size_cases.add("height_only")
        if test_shape[1] > mw and test_shape[0] <= mh:
            size_cases.add("width_only")
        if seed % 7 == 0 and len(train_shapes) >= 3:
            _require(train_shapes[:3] == [(2, 2), (3, 3), (4, 4)]
                     and test_shape == (5, 5), "source-like progression lost")
        train_colors = set(int(c) for p in puzzle["train"] for c in np.unique(p["input"]))
        test_colors = set(int(c) for c in np.unique(puzzle["test"][0]["input"]))
        largest = max(len(np.unique(p["input"])) for p in puzzle["train"])
        if len(test_colors) < largest:
            color_cases.add("fewer")
        if len(test_colors) == largest:
            color_cases.add("same_count")
        if not test_colors & train_colors:
            color_cases.add("disjoint")
        if test_colors & train_colors and test_colors - train_colors:
            color_cases.add("partial_new")
        if train_colors < test_colors:
            color_cases.add("strict_extension")
    _require(len(train_counts) >= 4 and len(test_counts) >= 3,
             "automatic pair counts lack diversity")
    _require(len(puzzles) == 42 and len(shapes) >= 35 and len(layouts) >= 100,
             "insufficient structural diversity")
    _require(len(palettes) >= 25 and aspect == {"square", "tall", "wide"},
             "insufficient palette or aspect diversity")
    _require(size_cases == {"same", "smaller", "height_only", "width_only"},
             "missing train/test size relationship")
    _require(color_cases == {"fewer", "same_count", "disjoint",
                             "partial_new", "strict_extension"},
             "missing train/test palette relationship")
    _require(one_dimensional >= 5 and unit >= 2 and max_side >= 5 and zero >= 10,
             "missing boundary geometries or color zero")
    _require(within_shapes >= 35 and within_palettes >= 35,
             "insufficient within-puzzle variation")
