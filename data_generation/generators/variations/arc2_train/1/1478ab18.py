"""Generators for the missing-square-corner triangle task."""

from itertools import combinations

import numpy as np


ORIENTATIONS = ("UL", "UR", "LL", "LR")
SOURCE_INPUTS = (
    ("57757777", "77777777", "75777777", "77757777", "77777777", "77777777", "77777777", "77777777"),
    ("77777777", "75777757", "77777777", "77777777", "77777777", "77777577", "75777777", "77777777"),
    ("77777775", "75777777", "77777777", "77777777", "77777777", "77777777", "77777777", "57777775"),
    ("75777777", "77777757", "77777777", "77777777", "77777777", "77777777", "75777775", "77777777"),
)
AUTO_COUNTS = ((2, 1), (3, 2), (4, 3), (5, 4), (6, 1),
               (7, 2), (8, 3), (9, 4), (10, 1))


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _count(value, name, minimum, maximum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _counts(rng, seed, num_train, num_test):
    if num_train is None and num_test is None:
        return AUTO_COUNTS[seed % len(AUTO_COUNTS)]
    train = None if num_train is None else _count(num_train, "num_train", 2, 10)
    test = None if num_test is None else _count(num_test, "num_test", 1, 4)
    if train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    if test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _signature(grid):
    return grid.shape, grid.tobytes()


def _quartet(points):
    """Infer one square and its inward cue from four positions."""
    rows = [p[0] for p in points]
    cols = [p[1] for p in points]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    side = r1 - r0
    if side < 3 or side != c1 - c0:
        return None
    corners = {"UL": (r0, c0), "UR": (r0, c1),
               "LL": (r1, c0), "LR": (r1, c1)}
    present = [name for name, point in corners.items() if point in points]
    if len(present) != 3:
        return None
    missing = next(name for name in ORIENTATIONS if name not in present)
    dr = 1 if missing[0] == "U" else -1
    dc = 1 if missing[1] == "L" else -1
    mr, mc = corners[missing]
    if (mr + dr, mc + dc) not in points:
        return None
    return {"orientation": missing, "bounds": (r0, r1, c0, c1),
            "side": side, "corners": corners}


def _groups(grid):
    colors, counts = np.unique(grid, return_counts=True)
    _check(len(colors) == 2, "input needs one background and one marker color")
    marker_indices = [i for i, n in enumerate(counts) if n in (4, 8) and counts[1-i] > n]
    _check(len(marker_indices) == 1, "marker role or count is ambiguous")
    marker = int(colors[marker_indices[0]])
    background = int(colors[1-marker_indices[0]])
    points = frozenset(tuple(map(int, p)) for p in np.argwhere(grid == marker))
    if len(points) == 4:
        info = _quartet(points)
        _check(info is not None, "invalid square quartet")
        return background, marker, [info]
    candidates = []
    ordered = sorted(points)
    for subset in combinations(ordered, 4):
        group = frozenset(subset)
        info = _quartet(group)
        if info is not None:
            candidates.append((group, info))
    partitions = []
    for i, (first_points, first) in enumerate(candidates):
        for second_points, second in candidates[i+1:]:
            if first_points | second_points != points or first_points & second_points:
                continue
            a0, a1, b0, b1 = first["bounds"]
            c0, c1, d0, d1 = second["bounds"]
            if not (a1 < c0 or c1 < a0 or b1 < d0 or d1 < b0):
                continue
            partitions.append((first, second))
    _check(len(partitions) == 1, "two marker quartets must have a unique partition")
    return background, marker, list(partitions[0])


def _draw(grid, groups, background, drawing):
    result = grid.copy()
    for info in groups:
        r0, r1, c0, c1 = info["bounds"]
        mr, mc = info["corners"][info["orientation"]]
        cells = {(mr, c) for c in range(c0, c1 + 1)}
        cells.update((r, mc) for r in range(r0, r1 + 1))
        adjacent = [p for name, p in info["corners"].items()
                    if name != info["orientation"] and (p[0] == mr or p[1] == mc)]
        start, end = adjacent
        dr = 1 if end[0] > start[0] else -1
        dc = 1 if end[1] > start[1] else -1
        cells.update((start[0] + k*dr, start[1] + k*dc)
                     for k in range(info["side"] + 1))
        for point in cells:
            if grid[point] == background:
                result[point] = drawing
    return result


def _choose_side(rng, regime):
    bands = ((3, 5), (6, 12), (13, 20), (21, 29))
    weights = (25, 30, 25, 20) if regime == "source_train" else (15, 25, 25, 35)
    band = int(rng.choice(4, p=np.array(weights) / 100))
    low, high = bands[band]
    return int(rng.integers(low, high + 1))


def _margin_pair(rng, limit, kind):
    if kind == "full":
        return 0, 0
    if kind == "interior" and limit >= 2:
        total = int(rng.integers(2, limit + 1))
        first = int(rng.integers(1, total))
        return first, total - first
    if kind == "edge" and limit >= 1:
        return (0, int(rng.integers(1, limit + 1))) if rng.integers(2) else (int(rng.integers(1, limit + 1)), 0)
    if kind == "unequal" and limit >= 4:
        large = int(rng.integers(max(3, limit//2), limit + 1))
        small = int(rng.integers(0, min(3, limit-large)+1))
        return (large, small) if rng.integers(2) else (small, large)
    total = int(rng.integers(0, limit + 1))
    first = int(rng.integers(0, total + 1))
    return first, total - first


def _single_layout(rng, regime):
    side = _choose_side(rng, regime)
    limit = 29 - side
    style = str(rng.choice(("interior", "edge", "corner", "full", "unequal", "free")))
    if style == "corner":
        top, bottom = _margin_pair(rng, limit, "edge")
        left, right = _margin_pair(rng, limit, "edge")
    elif style == "edge":
        top, bottom = _margin_pair(rng, limit, "edge")
        left, right = _margin_pair(rng, limit, "interior")
        if rng.integers(2):
            top, bottom, left, right = left, right, top, bottom
    else:
        top, bottom = _margin_pair(rng, limit, style)
        left, right = _margin_pair(rng, limit, style)
    return side + 1 + top + bottom, side + 1 + left + right, [(top, left, side)]


def _multi_layout(rng, regime):
    height = int(rng.integers(11, 31))
    width = int(rng.integers(11, 31))
    horizontal = bool(rng.integers(2))
    extent = height if horizontal else width
    split = int(rng.integers(5, extent - 5))
    if horizontal:
        regions = ((0, split, 0, width), (split+1, height, 0, width))
    else:
        regions = ((0, height, 0, split), (0, height, split+1, width))
    boxes = []
    for r0, r1, c0, c1 in regions:
        max_side = min(r1-r0, c1-c0) - 1
        side = int(rng.integers(3, max_side + 1))
        top = int(rng.integers(r0, r1 - side))
        left = int(rng.integers(c0, c1 - side))
        boxes.append((top, left, side))
    return height, width, boxes


def _place(grid, top, left, side, orientation, marker):
    r0, r1, c0, c1 = top, top+side, left, left+side
    corners = {"UL": (r0, c0), "UR": (r0, c1),
               "LL": (r1, c0), "LR": (r1, c1)}
    for name, point in corners.items():
        if name != orientation:
            grid[point] = marker
    mr, mc = corners[orientation]
    grid[mr + (1 if orientation[0] == "U" else -1),
         mc + (1 if orientation[1] == "L" else -1)] = marker


def _source_copy(grid):
    return grid.shape == (8, 8) and tuple("".join(str(int(v)) for v in row) for row in grid) in SOURCE_INPUTS


def _make_pair(rng, regime, orientations, multi, colors, seen):
    background, marker, drawing = colors
    for _ in range(160):
        height, width, boxes = _multi_layout(rng, regime) if multi else _single_layout(rng, regime)
        grid = np.full((height, width), background, dtype=np.int8)
        for i, (top, left, side) in enumerate(boxes):
            _place(grid, top, left, side, orientations[i], marker)
        if _source_copy(grid) or _signature(grid) in seen:
            continue
        try:
            _, _, groups = _groups(grid)
        except AssertionError:
            continue
        if len(groups) != len(boxes):
            continue
        output = _draw(grid, groups, background, drawing)
        seen.add(_signature(grid))
        return {"input": grid, "output": output}
    raise RuntimeError("unable to construct an unambiguous unique pair")


def _orientation_schedule(rng, train_count, test_count):
    mode = int(rng.integers(4))
    if mode == 0:
        common = str(rng.choice(ORIENTATIONS))
        return mode, [common]*train_count, [common]*test_count
    if mode == 1 and train_count >= 4:
        training = list(rng.permutation(ORIENTATIONS))
        training += [str(rng.choice(ORIENTATIONS)) for _ in range(train_count-4)]
        testing = [str(rng.choice(ORIENTATIONS)) for _ in range(test_count)]
    elif mode == 2:
        heldout = str(rng.choice(ORIENTATIONS))
        available = [o for o in ORIENTATIONS if o != heldout]
        if train_count == 2:
            positive = [o for o in available if o in ("UR", "LL")]
            negative = [o for o in available if o in ("UL", "LR")]
            training = [str(rng.choice(positive)), str(rng.choice(negative))]
        else:
            training = list(rng.permutation(available))
            training += [str(rng.choice(available)) for _ in range(train_count-3)]
        testing = [heldout] + [str(rng.choice(ORIENTATIONS)) for _ in range(test_count-1)]
    else:
        training = [str(rng.choice(("UR", "LL"))), str(rng.choice(("UL", "LR")))]
        training += [str(rng.choice(ORIENTATIONS)) for _ in range(train_count-2)]
        testing = [str(rng.choice(ORIENTATIONS)) for _ in range(test_count)]
    rng.shuffle(training)
    rng.shuffle(testing)
    return mode, training, testing


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    drawing = int(rng.integers(0, 10))
    mode, train_orientations, test_orientations = _orientation_schedule(rng, train_count, test_count)
    train_multi = set()
    if rng.random() < 0.75:
        train_multi = set(map(int, rng.choice(train_count, size=int(rng.integers(1, max(2, train_count//2+1))), replace=False)))
    test_multi = {i for i in range(test_count) if train_multi and rng.random() < 0.45}
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, size=train_count+test_count, dtype=np.int64)
    palette_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
    palettes = []
    for i in range(train_count+test_count):
        available = [c for c in range(10) if c != drawing]
        if i == 1:
            available.remove(palettes[0][0])
        background = int(palette_rng.choice(available))
        available = [c for c in range(10) if c not in (drawing, background)]
        if i == 1 and palettes[0][1] in available:
            available.remove(palettes[0][1])
        marker = int(palette_rng.choice(available))
        palettes.append((background, marker, drawing))
    seen = set()
    train, test = [], []
    for i in range(train_count+test_count):
        pair_rng = np.random.default_rng(int(pair_seeds[i]))
        is_train = i < train_count
        index = i if is_train else i-train_count
        primary = train_orientations[index] if is_train else test_orientations[index]
        multi = index in (train_multi if is_train else test_multi)
        secondary = primary if mode == 0 else str(pair_rng.choice([o for o in ORIENTATIONS if o != primary]))
        pair = _make_pair(pair_rng, "source_train" if is_train else "target_test",
                          (primary, secondary), multi, palettes[i], seen)
        (train if is_train else test).append(pair)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _check(isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list), "pair sections")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _check(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "pair counts")
    seen = set()
    drawing_colors = set()
    train_backgrounds, train_markers = set(), set()
    train_orientations, test_orientations = set(), set()
    train_multi = False
    test_multi = False
    for section in ("train", "test"):
        for pair in puzzle[section]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid array")
                _check(all(1 <= n <= 30 for n in grid.shape), "grid dimensions")
                _check(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                _check(bool(np.all((grid >= 0) & (grid <= 9))), "grid colors")
            _check(inp.shape == out.shape, "output shape")
            sig = _signature(inp)
            _check(sig not in seen and not _source_copy(inp), "duplicate or source input")
            seen.add(sig)
            background, marker, groups = _groups(inp)
            _check(len(groups) in (1, 2), "group count")
            (train_orientations if section == "train" else test_orientations).update(
                group["orientation"] for group in groups)
            added = set(map(int, np.unique(out))) - set(map(int, np.unique(inp)))
            _check(len(added) == 1, "drawing color must be visible")
            drawing = next(iter(added))
            drawing_colors.add(drawing)
            _check(np.array_equal(out, _draw(inp, groups, background, drawing)), "triangle rule")
            _check(not np.array_equal(inp, out), "transformation must change grid")
            _check(np.array_equal(inp == marker, out == marker), "marker preservation")
            if section == "train":
                train_backgrounds.add(background)
                train_markers.add(marker)
                train_multi |= len(groups) == 2
            else:
                test_multi |= len(groups) == 2
    _check(len(drawing_colors) == 1, "drawing color must be shared")
    _check(len(train_backgrounds) >= 2 and len(train_markers) >= 2,
           "training must demonstrate variable input colors")
    _check(not test_multi or train_multi, "test multiplicity needs a training demonstration")
    if test_orientations - train_orientations:
        _check(len(train_orientations) >= 2 and
               bool(train_orientations & {"UR", "LL"}) and
               bool(train_orientations & {"UL", "LR"}),
               "a held-out test rotation needs training corners on both diagonal slopes")


def _equal_puzzles(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output"))
               for s in ("train", "test"))


def _predict_from_training(train, test_input):
    drawing = None
    for pair in train:
        added = set(map(int, np.unique(pair["output"]))) - set(map(int, np.unique(pair["input"])))
        _check(len(added) == 1, "training drawing color")
        value = next(iter(added))
        _check(drawing is None or drawing == value, "inconsistent training drawing color")
        drawing = value
    background, _, groups = _groups(test_input)
    return _draw(test_input, groups, background, drawing)


def _independent_expected(inp, groups, background, drawing):
    result = inp.copy()
    for info in groups:
        r0, r1, c0, c1 = info["bounds"]
        missing = info["orientation"]
        edge_row = r0 if missing[0] == "U" else r1
        edge_col = c0 if missing[1] == "L" else c1
        for row in range(r0, r1 + 1):
            for col in range(c0, c1 + 1):
                on_diagonal = (row + col == r0 + c1) if missing in ("UL", "LR") else (row - col == r0 - c0)
                if inp[row, col] == background and (row == edge_row or col == edge_col or on_diagonal):
                    result[row, col] = drawing
    return result


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _check(len({len(p["train"]) for p in puzzles}) >= 4, "train count diversity")
    _check(len({len(p["test"]) for p in puzzles}) >= 3, "test count diversity")
    shapes, sides, orientations, palettes, layouts = set(), set(), set(), set(), set()
    multi_count = 0
    full_count = 0
    span3_tests = 0
    large_count = 0
    all_orientations_train = 0
    mixed_tests = 0
    common_puzzles = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        train_o, test_o = set(), set()
        for section in ("train", "test"):
            for pair in puzzle[section]:
                inp, out = pair["input"], pair["output"]
                bg, marker, groups = _groups(inp)
                drawing = next(iter(set(map(int, np.unique(out))) - {bg, marker}))
                _check(np.array_equal(_independent_expected(inp, groups, bg, drawing), out),
                       "independent outline construction failed")
                palettes.add((bg, marker, drawing))
                shapes.add(inp.shape)
                multi_count += len(groups) == 2
                if section == "test":
                    _check(np.array_equal(_predict_from_training(puzzle["train"], inp), out),
                           "training-based prediction failed")
                for info in groups:
                    orientation = info["orientation"]
                    (train_o if section == "train" else test_o).add(orientation)
                    orientations.add(orientation)
                    sides.add(info["side"])
                    layouts.add((inp.shape, info["bounds"], orientation))
                    r0, r1, c0, c1 = info["bounds"]
                    full_count += r0 == 0 and c0 == 0 and r1 == inp.shape[0]-1 and c1 == inp.shape[1]-1
                    large_count += info["side"] >= 21
                    if section == "test":
                        span3_tests += info["side"] == 3
        all_orientations_train += train_o == set(ORIENTATIONS)
        mixed_tests += len(test_o) >= 2
        common_puzzles += len(train_o | test_o) == 1
    _check(len(shapes) >= 45 and len(sides) >= 20 and len(layouts) >= 130, "structural diversity")
    _check(len(palettes) >= 50 and orientations == set(ORIENTATIONS), "role and orientation diversity")
    _check(multi_count >= 12 and full_count >= 2 and large_count >= 10 and span3_tests >= 1,
           "count, boundary, and scale diversity")
    _check(all_orientations_train >= 1 and mixed_tests >= 1 and common_puzzles >= 1,
           "within-puzzle orientation regimes")
    _check(_equal_puzzles(generate_puzzle(12345), generate_puzzle(12345)), "nondeterministic puzzle")
    _check(not _equal_puzzles(generate_puzzle(12345), generate_puzzle(12346)), "seeds collapsed")
    for train, test in ((2,1), (10,1), (10,4), (None,1), (None,4), (2,None), (10,None)):
        p = generate_puzzle(773, train, test)
        validate_generated_puzzle(p)
        if train is not None:
            _check(len(p["train"]) == train, "explicit training count")
        if test is not None:
            _check(len(p["test"]) == test, "explicit test count")
    for train, test in ((1,None), (11,None), (None,0), (None,5), (2,2), (4,4)):
        try:
            generate_puzzle(773, train, test)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
