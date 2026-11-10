"""Solved puzzles that serialize the number of foreground cells as a bar."""

import numpy as np


_SOURCE = (
    (((0, 7, 0), (0, 0, 0), (0, 0, 0)), ((7,),)),
    (((0, 2, 0), (2, 0, 0), (0, 2, 0)), ((2, 2, 2),)),
    (((0, 8, 0), (8, 8, 0), (8, 0, 0)), ((8, 8, 8, 8),)),
    (((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((1, 1),)),
    (((4, 4, 0), (4, 0, 4), (0, 0, 4)), ((4, 4, 4, 4, 4),)),
)


def _key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


_SOURCE_PAIRS = frozenset(
    (_key(np.asarray(a)), _key(np.asarray(b))) for a, b in _SOURCE
)
_SOURCE_INPUTS = frozenset(a for a, _ in _SOURCE_PAIRS)


def _legal_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return value


def _pair_counts(seed, num_train, num_test):
    train = _legal_count(num_train, "num_train", 2, 10)
    test = _legal_count(num_test, "num_test", 1, 4)
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


def _colors(rng, train_total, test_total, fixed_foreground, source_like):
    fixed = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != fixed]
    rng.shuffle(others)
    palette_size = min(train_total, int(rng.integers(2, min(6, train_total) + 1)))
    train_varying = (others[:palette_size] * ((train_total + palette_size - 1) // palette_size))[:train_total]
    rng.shuffle(train_varying)
    test_varying = []
    unseen = others[palette_size:]
    for i in range(test_total):
        if source_like and unseen:
            test_varying.append(unseen.pop(0))
        elif i == 0 or not unseen or bool(rng.integers(0, 2)):
            test_varying.append(int(rng.choice(train_varying)))
        else:
            test_varying.append(unseen.pop(0))
    if fixed_foreground:
        return ([(b, fixed) for b in train_varying],
                [(b, fixed) for b in test_varying])
    return ([(fixed, f) for f in train_varying],
            [(fixed, f) for f in test_varying])


def _counts(rng, train_total, test_total, source_like, bbox_contrast):
    if source_like:
        train = [1, 4] + [int(rng.integers(1, 5)) for _ in range(train_total - 2)]
        rng.shuffle(train)
        test = [int(x) for x in rng.choice(np.arange(5, 16), size=test_total, replace=False)]
    else:
        train = [int(rng.integers(1, 5)), int(rng.integers(6, 31))]
        for _ in range(train_total - 2):
            band = int(rng.integers(0, 4))
            bounds = ((1, 6), (4, 13), (8, 22), (15, 31))[band]
            train.append(int(rng.integers(*bounds)))
        if bbox_contrast:
            train[:2] = [6, 8]
        test = [int(rng.choice(train))]
        for _ in range(test_total - 1):
            if bool(rng.integers(0, 2)):
                test.append(int(rng.choice(train)))
            else:
                test.append(int(rng.integers(1, 31)))
    return train, test


_SIZE_RANGES = (
    (3, 5, 3, 5), (2, 6, 2, 6), (4, 10, 4, 10),
    (2, 7, 7, 18), (7, 18, 2, 7), (7, 18, 7, 18),
)
_DENSITIES = (0.12, 0.32, 0.58, 0.81)


def _dimensions(rng, count, forbidden_areas):
    size_kind = int(rng.integers(0, len(_SIZE_RANGES)))
    density = _DENSITIES[int(rng.integers(0, len(_DENSITIES)))]
    h0, h1, w0, w1 = _SIZE_RANGES[size_kind]
    candidates = [
        (h, w) for h in range(h0, h1 + 1) for w in range(w0, w1 + 1)
        if h * w > count and h * w not in forbidden_areas
    ]
    if not candidates:
        candidates = [
            (h, w) for h in range(2, 19) for w in range(2, 19)
            if h * w > count and h * w not in forbidden_areas
        ]
    # Sample near the chosen occupancy, with jitter to avoid rigid bins.
    desired = min(0.94, max(0.04, density + float(rng.uniform(-0.09, 0.09))))
    scores = np.asarray([abs(count / (h * w) - desired) for h, w in candidates])
    nearest = np.flatnonzero(scores <= float(scores.min()) + 0.08)
    return candidates[int(rng.choice(nearest))]


def _neighbors(index, height, width):
    r, c = divmod(index, width)
    return [nr * width + nc for nr, nc in ((r - 1, c), (r + 1, c),
            (r, c - 1), (r, c + 1)) if 0 <= nr < height and 0 <= nc < width]


def _positions(rng, height, width, count, mode):
    area = height * width
    if mode == "scatter":
        return [int(x) for x in rng.choice(area, count, replace=False)]
    if mode == "connected":
        chosen = {int(rng.integers(0, area))}
        while len(chosen) < count:
            frontier = sorted({j for i in chosen for j in _neighbors(i, height, width)} - chosen)
            chosen.add(int(rng.choice(frontier)))
        return sorted(chosen)
    if mode == "horizontal" and count <= width:
        r = int(rng.integers(0, height))
        c = int(rng.integers(0, width - count + 1))
        return [r * width + c + k for k in range(count)]
    if mode == "vertical" and count <= height:
        r = int(rng.integers(0, height - count + 1))
        c = int(rng.integers(0, width))
        return [(r + k) * width + c for k in range(count)]
    if mode == "block":
        frames = [(h, w) for h in range(2, height + 1) for w in range(2, width + 1)
                  if count <= h * w <= count + max(3, count // 3)]
        if frames:
            fh, fw = frames[int(rng.integers(0, len(frames)))]
            top = int(rng.integers(0, height - fh + 1))
            left = int(rng.integers(0, width - fw + 1))
            cells = [(top + r) * width + left + c for r in range(fh) for c in range(fw)]
            return [int(x) for x in rng.choice(cells, count, replace=False)]
    if mode == "corner":
        corner_r = int(rng.choice([0, height - 1]))
        corner_c = int(rng.choice([0, width - 1]))
        ordered = sorted(range(area), key=lambda i: (
            abs(i // width - corner_r) + abs(i % width - corner_c)
            + float(rng.uniform(-0.45, 0.45))))
        return ordered[:count]
    if mode == "separated":
        chosen = [int(rng.integers(0, area))]
        while len(chosen) < count:
            remaining = [i for i in range(area) if i not in chosen]
            distances = [min(abs(i // width - j // width) + abs(i % width - j % width)
                             for j in chosen) for i in remaining]
            farthest = [i for i, d in zip(remaining, distances) if d == max(distances)]
            chosen.append(int(rng.choice(farthest)))
        return chosen
    if mode == "diagonal":
        main = bool(rng.integers(0, 2))
        diagonal = [r * width + (r if main else width - 1 - r)
                    for r in range(min(height, width, count))]
        remaining = [i for i in range(area) if i not in diagonal]
        extra = rng.choice(remaining, count - len(diagonal), replace=False)
        return diagonal + [int(x) for x in extra]
    return [int(x) for x in rng.choice(area, count, replace=False)]


_MODES = ("scatter", "connected", "horizontal", "vertical", "block",
          "corner", "separated", "diagonal")


def _make_pair(height, width, positions, background, foreground):
    grid = np.full((height, width), background, dtype=np.int8)
    grid.flat[np.asarray(positions, dtype=np.int64)] = foreground
    bar = np.full((1, len(positions)), foreground, dtype=np.int8)
    return {"input": grid, "output": bar}


def _sample_pair(rng, background, foreground, count, used, forbidden_areas=()):
    for _ in range(96):
        height, width = _dimensions(rng, count, forbidden_areas)
        mode = _MODES[int(rng.integers(0, len(_MODES)))]
        pair = _make_pair(height, width, _positions(rng, height, width, count, mode),
                          background, foreground)
        key = _key(pair["input"])
        if key not in used:
            used.add(key)
            return pair
    raise RuntimeError("No distinct input found")


def _contrast_pairs(rng, roles, used):
    height = int(rng.integers(4, 10))
    width = int(rng.integers(4, 10))
    top = int(rng.integers(0, height - 2))
    left = int(rng.integers(0, width - 2))
    frame = [(top + r) * width + left + c for r in range(3) for c in range(3)]
    corners = {frame[0], frame[2], frame[6], frame[8]}
    result = []
    for count, (background, foreground) in zip((6, 8), roles):
        extra = [i for i in frame if i not in corners]
        positions = sorted(corners | set(int(x) for x in rng.choice(extra, count - 4, replace=False)))
        pair = _make_pair(height, width, positions, background, foreground)
        key = _key(pair["input"])
        assert key not in used
        used.add(key)
        result.append(pair)
    return result


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_total, test_total = _pair_counts(seed, num_train, num_test)
    source_like = seed % 5 == 0
    fixed_foreground = seed % 2 == 1
    bbox_contrast = not source_like and seed % 3 == 0
    train_roles, test_roles = _colors(rng, train_total, test_total,
                                     fixed_foreground, source_like)
    train_counts, test_counts = _counts(rng, train_total, test_total,
                                        source_like, bbox_contrast)
    pair_seeds = rng.integers(0, 2**63 - 1, size=train_total + test_total)
    used = set(_SOURCE_INPUTS)
    train = []
    if bbox_contrast:
        train.extend(_contrast_pairs(np.random.default_rng(int(pair_seeds[0])),
                                     train_roles[:2], used))
    for i in range(len(train), train_total):
        b, f = train_roles[i]
        train.append(_sample_pair(np.random.default_rng(int(pair_seeds[i])),
                                  b, f, train_counts[i], used))
    train_areas = {}
    for pair in train:
        train_areas.setdefault(pair["output"].shape[1], set()).add(pair["input"].size)
    test = []
    for i, count in enumerate(test_counts):
        b, f = test_roles[i]
        test.append(_sample_pair(np.random.default_rng(int(pair_seeds[train_total + i])),
                                 b, f, count, used, train_areas.get(count, ())))
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_grid(grid):
    assert isinstance(grid, np.ndarray) and grid.ndim == 2
    assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
    assert np.issubdtype(grid.dtype, np.integer)
    assert 0 <= int(grid.min()) and int(grid.max()) <= 9


def _training_convention(train):
    shared = set(int(x) for x in np.unique(train[0]["input"]))
    for pair in train[1:]:
        shared &= set(int(x) for x in np.unique(pair["input"]))
    assert len(shared) == 1
    fixed = next(iter(shared))
    bar_colors = {int(pair["output"][0, 0]) for pair in train}
    if bar_colors == {fixed}:
        return "foreground", fixed
    assert fixed not in bar_colors and len(bar_colors) >= 2
    return "background", fixed


def _solve_from_training(train, input_grid):
    role, fixed = _training_convention(train)
    colors = set(int(x) for x in np.unique(input_grid))
    assert len(colors) == 2 and fixed in colors
    foreground = fixed if role == "foreground" else next(iter(colors - {fixed}))
    count = int(np.count_nonzero(input_grid == foreground))
    return np.full((1, count), foreground, dtype=np.int8)


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    seen_inputs, seen_pairs = set(), set()
    for pair in train + test:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        a, b = pair["input"], pair["output"]
        _check_grid(a)
        _check_grid(b)
        colors = set(int(x) for x in np.unique(a))
        assert len(colors) == 2
        assert b.shape[0] == 1 and np.all(b == b[0, 0])
        foreground = int(b[0, 0])
        assert foreground in colors and b.shape[1] == int(np.count_nonzero(a == foreground))
        assert not (a.shape == b.shape and np.array_equal(a, b))
        key = _key(a)
        pair_key = (key, _key(b))
        assert key not in seen_inputs and pair_key not in seen_pairs
        assert key not in _SOURCE_INPUTS and pair_key not in _SOURCE_PAIRS
        seen_inputs.add(key)
        seen_pairs.add(pair_key)
    role, fixed = _training_convention(train)
    for pair in train + test:
        assert fixed in set(int(x) for x in np.unique(pair["input"]))
        assert np.array_equal(_solve_from_training(train, pair["input"]), pair["output"])
    train_counts = [p["output"].shape[1] for p in train]
    test_counts = [p["output"].shape[1] for p in test]
    if max(train_counts) <= 4:
        assert min(train_counts) == 1 and max(train_counts) == 4
        assert all(c > 4 for c in test_counts)
    else:
        assert any(c <= max(train_counts) for c in test_counts)
    train_areas = {}
    for pair in train:
        train_areas.setdefault(pair["output"].shape[1], set()).add(pair["input"].size)
    for pair in test:
        assert pair["input"].size not in train_areas.get(pair["output"].shape[1], ())


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p["input"]), _key(p["output"])) for p in puzzle[k])
                 for k in ("train", "test"))


def _component_sizes(mask):
    remaining = set(zip(*np.where(mask)))
    sizes = []
    while remaining:
        stack = [remaining.pop()]
        size = 0
        while stack:
            r, c = stack.pop()
            size += 1
            for neighbor in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return sizes


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(12345)) == _puzzle_key(generate_puzzle(12345))
    assert _puzzle_key(generate_puzzle(12345, 8, 3)) == _puzzle_key(generate_puzzle(12345, 8, 3))
    checked = list(puzzles)
    for seed, ntrain, ntest in ((1000, 2, 1), (1001, 10, 1), (1002, 10, 4),
                                (2000, None, 1), (2001, None, 4),
                                (2002, 2, None), (2003, 10, None)):
        puzzle = generate_puzzle(seed, ntrain, ntest)
        if ntrain is not None:
            assert len(puzzle["train"]) == ntrain
        if ntest is not None:
            assert len(puzzle["test"]) == ntest
        checked.append(puzzle)
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(999, *args)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {args}")
    dimensions, counts, masks, colors = set(), set(), set(), set()
    majority_train = majority_test = overlap = no_diagonal_test = large_sparse = 0
    source_like = same_count_new_area = same_bbox_new_count = 0
    fixed_foreground = fixed_background = reused_color = new_color = 0
    train_above_14 = test_above_14 = small_dense = 0
    horizontal = vertical = compact = disconnected = separated = 0
    for puzzle in checked:
        validate_generated_puzzle(puzzle)
        training = puzzle["train"]
        train_counts = {p["output"].shape[1] for p in training}
        role, fixed = _training_convention(training)
        colors.add((role, fixed))
        fixed_foreground += role == "foreground"
        fixed_background += role == "background"
        source_like += max(train_counts) <= 4
        train_areas = {}
        train_bboxes = []
        for pair in training:
            train_areas.setdefault(pair["output"].shape[1], set()).add(pair["input"].size)
            mask = pair["input"] == pair["output"][0, 0]
            rr, cc = np.where(mask)
            train_bboxes.append(((int(rr.max() - rr.min() + 1),
                                  int(cc.max() - cc.min() + 1)), pair["output"].shape[1]))
        same_bbox_new_count += any(
            a[0] == b[0] and a[1] != b[1]
            for i, a in enumerate(train_bboxes) for b in train_bboxes[i + 1:])
        same_count_new_area += any(
            pair["output"].shape[1] in train_areas
            and pair["input"].size not in train_areas[pair["output"].shape[1]]
            for pair in puzzle["test"])
        varying_train_colors = {
            int(color) for pair in training for color in np.unique(pair["input"])
            if int(color) != fixed
        }
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = pair["input"]
                solved = _solve_from_training(puzzle["train"], a)
                assert np.array_equal(solved, pair["output"])
                foreground = int(solved[0, 0])
                mask = a == foreground
                count = int(np.count_nonzero(mask))
                dimensions.add(a.shape)
                counts.add(count)
                masks.add((a.shape, tuple(int(x) for x in mask.flat)))
                rr, cc = np.where(mask)
                bbox_area = int((rr.max() - rr.min() + 1) * (cc.max() - cc.min() + 1))
                components = _component_sizes(mask)
                if split == "test":
                    horizontal += len(set(int(x) for x in rr)) == 1 and count >= 2
                    vertical += len(set(int(x) for x in cc)) == 1 and count >= 2
                    compact += bbox_area <= count + 2 and count >= 3
                    disconnected += len(components) > 1
                    separated += len(components) >= 3 and max(components) <= 2
                if count > a.size / 2:
                    if split == "train":
                        majority_train += 1
                    else:
                        majority_test += 1
                if a.size >= 80 and count / a.size <= 0.2:
                    large_sparse += 1
                if a.size <= 25 and count > a.size / 2:
                    small_dense += 1
                if split == "test":
                    test_above_14 += count > 14
                    overlap += count in train_counts
                    varying_color = next(int(x) for x in np.unique(a) if int(x) != fixed)
                    reused_color += varying_color in varying_train_colors
                    new_color += varying_color not in varying_train_colors
                    diagonal_length = min(a.shape)
                    main = all(a[i, i] == foreground for i in range(diagonal_length))
                    anti = all(a[i, a.shape[1] - 1 - i] == foreground for i in range(diagonal_length))
                    no_diagonal_test += not (main or anti)
                else:
                    train_above_14 += count > 14
    assert len(dimensions) >= 25 and len(counts) >= 20 and len(masks) >= 130
    assert len(colors) >= 8
    assert majority_train >= 5 and majority_test >= 5
    assert overlap >= 10 and no_diagonal_test >= 20 and large_sparse >= 5
    assert source_like >= 7 and same_count_new_area >= 25 and same_bbox_new_count >= 9
    assert fixed_foreground >= 15 and fixed_background >= 15
    assert reused_color >= 15 and new_color >= 10
    assert train_above_14 >= 10 and test_above_14 >= 5 and small_dense >= 10
    assert horizontal >= 1 and vertical >= 1 and compact >= 5
    assert disconnected >= 10 and separated >= 2
