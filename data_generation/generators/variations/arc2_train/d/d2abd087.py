import numpy as np


_NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


_SOURCE_PAIRS = {
    (
        (
            "0000000050",
            "0550005550",
            "5555000050",
            "0000000000",
            "0055550050",
            "0000000050",
            "0000055000",
            "0550055000",
            "0550055000",
            "0000000000",
        ),
        (
            "0000000010",
            "0220001110",
            "2222000010",
            "0000000000",
            "0011110010",
            "0000000010",
            "0000022000",
            "0110022000",
            "0110022000",
            "0000000000",
        ),
    ),
    (
        (
            "0000000000",
            "0000000000",
            "0055500000",
            "0055500000",
            "0000000000",
            "0000005500",
            "0000055500",
            "0550005000",
            "0555000000",
            "0000000000",
        ),
        (
            "0000000000",
            "0000000000",
            "0022200000",
            "0022200000",
            "0000000000",
            "0000002200",
            "0000022200",
            "0110002000",
            "0111000000",
            "0000000000",
        ),
    ),
    (
        (
            "5550000555",
            "0550550500",
            "0050550500",
            "0050000500",
            "0000550005",
            "0550055005",
            "0000055005",
            "0050000000",
            "0555500000",
            "0055000000",
        ),
        (
            "1110000222",
            "0110110200",
            "0010110200",
            "0010000200",
            "0000220001",
            "0110022001",
            "0000022001",
            "0010000000",
            "0111100000",
            "0011000000",
        ),
    ),
    (
        (
            "0550005500",
            "0550005500",
            "5555055000",
            "0000000000",
            "0550000500",
            "0555500500",
            "0000000500",
            "0000000500",
            "0555550000",
            "0000000000",
        ),
        (
            "0110002200",
            "0110002200",
            "1111022000",
            "0000000000",
            "0220000100",
            "0222200100",
            "0000000100",
            "0000000100",
            "0111110000",
            "0000000000",
        ),
    ),
}


def _checked_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(rng, num_train, num_test):
    train = _checked_count(num_train, "num_train", 2, 10)
    test = _checked_count(num_test, "num_test", 1, 4)
    if train is None and test is None:
        test = int(rng.integers(1, 5))
        train = int(rng.integers(test + 1, 11))
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _components(grid, color):
    height, width = grid.shape
    seen = set()
    result = []
    for row in range(height):
        for col in range(width):
            if int(grid[row, col]) != color or (row, col) in seen:
                continue
            stack = [(row, col)]
            seen.add((row, col))
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nxt = (r + dr, c + dc)
                    if (0 <= nxt[0] < height and 0 <= nxt[1] < width
                            and nxt not in seen and int(grid[nxt]) == color):
                        seen.add(nxt)
                        stack.append(nxt)
            result.append(tuple(cells))
    return result


def _normalize(cells):
    top = min(r for r, _ in cells)
    left = min(c for _, c in cells)
    return tuple(sorted((r - top, c - left) for r, c in cells))


def _orient(cells, rng):
    cells = list(cells)
    if rng.integers(2):
        cells = [(c, r) for r, c in cells]
    if rng.integers(2):
        cells = [(-r, c) for r, c in cells]
    if rng.integers(2):
        cells = [(r, -c) for r, c in cells]
    return _normalize(cells)


def _growth(area, rng, compact=False):
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (r + dr, c + dc)
                if nxt not in cells:
                    frontier.add(nxt)
        frontier = sorted(frontier)
        if compact:
            scores = np.array([sum(abs(r - rr) + abs(c - cc) == 1
                                   for rr, cc in cells) for r, c in frontier])
            best = [frontier[i] for i in range(len(frontier)) if scores[i] == max(scores)]
            chosen = best[int(rng.integers(len(best)))]
        else:
            chosen = frontier[int(rng.integers(len(frontier)))]
        cells.add(chosen)
    return cells


def _shape(area, rng):
    style = int(rng.integers(6))
    if style == 0:
        cells = {(0, c) for c in range(area)}
    elif style == 1 and area == 6:
        cells = {(r, c) for r in range(2) for c in range(3)}
    elif style == 1 and area >= 4:
        width = max(2, min(area - 1, int(rng.integers(2, 5))))
        cells = {(0, c) for c in range(width)} | {(r, 0) for r in range(area - width + 1)}
    elif style == 2 and area >= 4:
        length = (area + 1) // 2
        cells = {(0, c) for c in range(length)} | {(1, c) for c in range(area - length)}
    elif style == 3:
        cells = {(0, 0)}
        r = c = 0
        for step in range(1, area):
            if step % 2:
                c += 1
            else:
                r += 1
            cells.add((r, c))
    else:
        cells = _growth(area, rng, compact=(style == 5))
    return _orient(cells, rng)


def _dims(shape):
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _diagonal(first, second):
    other = set(second)
    return any((r + dr, c + dc) in other for r, c in first
               for dr in (-1, 1) for dc in (-1, 1))


def _place(shapes, height, width, rng, force_corner=False):
    occupied = set()
    placed = []
    for index, shape in enumerate(shapes):
        sh, sw = _dims(shape)
        candidates = []
        for top in range(height - sh + 1):
            for left in range(width - sw + 1):
                cells = tuple((top + r, left + c) for r, c in shape)
                if any((r, c) in occupied or (r + 1, c) in occupied
                       or (r - 1, c) in occupied or (r, c + 1) in occupied
                       or (r, c - 1) in occupied for r, c in cells):
                    continue
                if force_corner and index == 1 and not _diagonal(placed[0], cells):
                    continue
                candidates.append(cells)
        if not candidates:
            return None
        choice = candidates[int(rng.integers(len(candidates)))]
        placed.append(choice)
        occupied.update(choice)
    return placed


def _local_colors(rng, labels, fixed, first=None):
    choices = [x for x in range(10) if x not in labels]
    if fixed is not None:
        return fixed
    for _ in range(30):
        pair = tuple(int(x) for x in rng.choice(choices, size=2, replace=False))
        if pair != first:
            return pair
    raise AssertionError("unreachable color selection")


def _dimensions(rng):
    mode = int(rng.choice(4, p=[0.30, 0.20, 0.30, 0.20]))
    if mode == 0:
        return 10, 10
    if mode == 1:
        side = int(rng.integers(11, 18))
        return side, side
    if mode == 2:
        return int(rng.integers(10, 17)), int(rng.integers(11, 19))
    return int(rng.integers(16, 23)), int(rng.integers(16, 24))


def _area_plan(rng, regime, pair_index, compact, familiar,
               train_min_area, train_max_area):
    count = int(rng.integers(3, 7 if compact else 9))
    if regime == "source_train" and pair_index == 0:
        areas = [6, 5, 6]
    elif regime == "source_train" and pair_index == 1:
        targets = int(rng.integers(0, min(3, count - 1) + 1))
        areas = [7] + [6] * targets
    else:
        targets = int(rng.integers(0, min(3, count - 1) + 1))
        areas = [6] * targets
        if not areas:
            areas = [int(rng.choice((1, 2, 4, 5, 7, 8)))]
    max_area = 8 if compact else (10 if regime == "source_train" else 13)
    if regime == "source_train":
        max_area = min(max_area, train_max_area)
    pool = [a for a in range(train_min_area if regime == "source_train" else 1,
                            max_area + 1) if a != 6]
    if regime == "target_test" and len(areas) < count:
        strategy = int(rng.integers(4))
        if strategy == 0 and familiar:
            areas.append(int(rng.choice(sorted(familiar))))
        elif strategy == 1:
            areas.append(8)
        elif strategy == 2:
            novel = [a for a in pool if a not in familiar]
            if novel:
                smaller = [a for a in novel if a < min(familiar)] if familiar else []
                selection = smaller if smaller and rng.integers(2) else novel
                areas.append(int(rng.choice(selection)))
        elif strategy == 3 and not compact:
            areas.append(int(rng.integers(11, 14)))
    while len(areas) < count:
        areas.append(int(rng.choice(pool)))
    if len(areas) > count:
        areas = areas[:count]
    if not (regime == "source_train" and pair_index == 0):
        rng.shuffle(areas)
    return areas


def _render_pair(rng, dims, areas, colors, tutorial=False, matched=False):
    height, width = dims
    for _ in range(100):
        shapes = []
        if tutorial and matched:
            full = {(r, c) for r in range(2) for c in range(3)}
            missing = full - {(int(rng.integers(2)), int(rng.integers(3)))}
            # Both silhouettes receive exactly the same orientation.
            transpose = bool(rng.integers(2))
            flip_r = bool(rng.integers(2))
            flip_c = bool(rng.integers(2))
            def change(cells):
                result = [(c, r) if transpose else (r, c) for r, c in cells]
                result = [(-r if flip_r else r, -c if flip_c else c) for r, c in result]
                return _normalize(result)
            shapes = [change(full), change(missing), _shape(6, rng)]
            if shapes[2] == shapes[0]:
                shapes[2] = _orient({(0, c) for c in range(6)}, rng)
            for area in areas[3:]:
                shapes.append(_shape(area, rng))
        elif tutorial:
            shapes = [_shape(6, rng), _shape(5, rng), _shape(6, rng)]
            if shapes[2] == shapes[0]:
                shapes[2] = _orient({(0, c) for c in range(6)}, rng)
                if shapes[2] == shapes[0]:
                    shapes[2] = _normalize({(r, c) for r in range(2) for c in range(3)})
            for area in areas[3:]:
                shapes.append(_shape(area, rng))
        else:
            shapes = [_shape(area, rng) for area in areas]
        placed = _place(shapes, height, width, rng, force_corner=tutorial)
        if placed is None:
            continue
        background, foreground = colors[:2]
        target, other = colors[2:]
        inp = np.full(dims, background, dtype=np.int8)
        out = inp.copy()
        for cells in placed:
            label = target if len(cells) == 6 else other
            for r, c in cells:
                inp[r, c] = foreground
                out[r, c] = label
        if sorted(len(comp) for comp in _components(inp, foreground)) == sorted(areas):
            return {"input": inp, "output": out}
    return None


def _key(array):
    return (array.shape, array.tobytes())


def _rows(array):
    return tuple("".join(str(int(x)) for x in row) for row in array)


def _basic(pair):
    assert isinstance(pair, dict) and set(pair) == {"input", "output"}
    inp, out = pair["input"], pair["output"]
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.issubdtype(grid.dtype, np.integer)
        assert np.all((0 <= grid) & (grid <= 9))
    assert inp.shape == out.shape


def _inspect(pair):
    _basic(pair)
    inp, out = pair["input"], pair["output"]
    values, counts = np.unique(inp, return_counts=True)
    assert len(values) == 2 and counts[0] != counts[1]
    back = int(values[int(np.argmax(counts))])
    fore = int(values[int(np.argmin(counts))])
    assert np.all(out[inp == back] == back)
    comps = _components(inp, fore)
    assert 3 <= len(comps) <= 8
    labels = {6: set(), 0: set()}
    for comp in comps:
        colors = {int(out[r, c]) for r, c in comp}
        assert len(colors) == 1
        labels[6 if len(comp) == 6 else 0].update(colors)
    assert len(labels[0]) == 1
    if labels[6]:
        assert len(labels[6]) == 1
    assert fore not in {back, *labels[0], *labels[6]}
    return back, fore, labels, comps


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    train, test = puzzle["train"], puzzle["test"]
    assert isinstance(train, list) and isinstance(test, list)
    assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
    first = _inspect(train[0])
    assert len(first[2][6]) == len(first[2][0]) == 1
    target = next(iter(first[2][6]))
    other = next(iter(first[2][0]))
    assert target != other and target != first[0] and other != first[0]
    first_shapes = {_normalize(comp) for comp in first[3] if len(comp) == 6}
    assert len(first_shapes) >= 2
    assert any(len(comp) == 5 for comp in first[3])
    assert any(_diagonal(a, b) for a in first[3] for b in first[3]
               if len(a) == 6 and len(b) == 5)
    keys = set()
    source = _SOURCE_PAIRS
    train_sizes = set()
    for section in (train, test):
        for pair in section:
            back, fore, labels, comps = _inspect(pair)
            assert back not in {target, other} and fore not in {target, other}
            assert labels[0] == {other}
            assert not labels[6] or labels[6] == {target}
            key = _key(pair["input"])
            assert key not in keys
            keys.add(key)
            assert (_rows(pair["input"]), _rows(pair["output"])) not in source
            if section is train:
                train_sizes.update(len(comp) for comp in comps)
    assert 7 in train_sizes
    # The first demonstration anchors both labels and diagonal four-connectivity.
    # The second supplies an object larger than the target.


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    ntrain, ntest = _resolve_counts(rng, num_train, num_test)
    target, other = (int(x) for x in rng.choice(10, size=2, replace=False))
    variable_input = bool(rng.integers(2))
    matched_tutorial = bool(rng.integers(2))
    train_min_area = int(rng.choice((1, 2)))
    train_max_area = int(rng.choice((7, 8, 10)))
    fixed_colors = None if variable_input else _local_colors(rng, (target, other), None)
    puzzle = {"train": [], "test": []}
    used = set()
    familiar = set()
    first_colors = None
    for regime, n, section in (("source_train", ntrain, "train"),
                               ("target_test", ntest, "test")):
        for pair_index in range(n):
            accepted = False
            for retry in range(90):
                dims = _dimensions(rng)
                compact = max(dims) <= 11
                areas = _area_plan(rng, regime, pair_index, compact, familiar,
                                   train_min_area, train_max_area)
                local = _local_colors(rng, (target, other), fixed_colors,
                                      first_colors if variable_input and section == "train" and pair_index == 1 else None)
                pair = _render_pair(rng, dims, areas,
                                    (local[0], local[1], target, other),
                                    tutorial=regime == "source_train" and pair_index == 0,
                                    matched=matched_tutorial)
                if pair is None or _key(pair["input"]) in used:
                    continue
                if (_rows(pair["input"]), _rows(pair["output"])) in _SOURCE_PAIRS:
                    continue
                if first_colors is None:
                    first_colors = local
                used.add(_key(pair["input"]))
                puzzle[section].append(pair)
                if section == "train":
                    familiar.update(a for a in areas if a != 6)
                accepted = True
                break
            if not accepted:
                raise ValueError(f"could not place {regime} pair {pair_index}")
    validate_generated_puzzle(puzzle)
    return puzzle


def _same(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    samples = [generate_puzzle(seed) for seed in range(42)]
    for sample in samples:
        validate_generated_puzzle(sample)
    assert _same(generate_puzzle(12345), generate_puzzle(12345))
    assert not _same(generate_puzzle(12345), generate_puzzle(12346))
    for n, m in ((2, 1), (10, 1), (10, 4), (None, 1),
                 (None, 4), (2, None), (10, None), (6, 2)):
        puzzle = generate_puzzle(100 + (n or 0) + (m or 0), n, m)
        assert n is None or len(puzzle["train"]) == n
        assert m is None or len(puzzle["test"]) == m
    for n, m in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (4, 4), (3, 4)):
        try:
            generate_puzzle(17, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError((n, m))
    assert len({len(p["train"]) for p in samples}) >= 4
    assert len({len(p["test"]) for p in samples}) >= 3
    dims = {pair["input"].shape for p in samples for s in ("train", "test") for pair in p[s]}
    assert (10, 10) in dims and len(dims) >= 25
    all_sizes = set()
    counts = set()
    backgrounds = set()
    targets = set()
    compact = spacious = singleton = large = zero_target = 0
    variable_puzzles = 0
    for p in samples:
        local_colors = set()
        for s in ("train", "test"):
            for pair in p[s]:
                back, fore, labels, comps = _inspect(pair)
                local_colors.add((back, fore))
                sizes = [len(c) for c in comps]
                all_sizes.update(sizes)
                counts.add(len(comps))
                backgrounds.add(back)
                if labels[6]:
                    targets.update(labels[6])
                compact += pair["input"].shape == (10, 10)
                spacious += min(pair["input"].shape) >= 16
                singleton += 1 in sizes
                large += any(a > 9 for a in sizes)
                zero_target += 6 not in sizes
        variable_puzzles += len(local_colors) > 1
    assert counts.issuperset({3, 4, 5, 6})
    assert all_sizes.issuperset({1, 5, 6, 7, 8, 10})
    assert len(backgrounds) >= 6 and len(targets) >= 6
    assert compact >= 12 and spacious >= 12
    assert singleton >= 5 and large >= 5 and zero_target >= 5
    assert variable_puzzles >= 8
    review = {
        "crowded": 0, "diagonal_test": 0, "matched_first": 0,
        "unmatched_first": 0, "familiar_test": 0, "novel_small": 0,
        "novel_large": 0, "novel_eight": 0, "zero_target_test": 0,
        "one_target_test": 0, "three_objects": 0,
    }
    target_shapes = set()
    for puzzle in samples:
        training = [_inspect(pair) for pair in puzzle["train"]]
        familiar = {len(comp) for _, _, _, comps in training
                    for comp in comps if len(comp) != 6}
        first_components = training[0][3]
        def bounding_box(comp):
            return (max(r for r, _ in comp) - min(r for r, _ in comp) + 1,
                    max(c for _, c in comp) - min(c for _, c in comp) + 1)
        matched = any(len(a) == 6 and len(b) == 5 and
                      bounding_box(a) == bounding_box(b)
                      for a in first_components for b in first_components)
        review["matched_first" if matched else "unmatched_first"] += 1
        for section in ("train", "test"):
            for pair in puzzle[section]:
                back, fore, _, comps = _inspect(pair)
                sizes = [len(comp) for comp in comps]
                occupancy = sum(sizes) / pair["input"].size
                if pair["input"].shape == (10, 10) and 0.17 <= occupancy <= 0.35:
                    review["crowded"] += 1
                if len(comps) == 3:
                    review["three_objects"] += 1
                for comp in comps:
                    if len(comp) == 6:
                        target_shapes.add(_normalize(comp))
                if section == "test":
                    negatives = {size for size in sizes if size != 6}
                    review["familiar_test"] += negatives.issubset(familiar)
                    review["novel_small"] += any(size not in familiar and
                                                  size < min(familiar)
                                                  for size in negatives)
                    review["novel_large"] += any(size not in familiar and
                                                  size > max(familiar)
                                                  for size in negatives)
                    review["novel_eight"] += 8 in negatives and 8 not in familiar
                    review["zero_target_test"] += 6 not in sizes
                    review["one_target_test"] += sizes.count(6) == 1
                    review["diagonal_test"] += any(
                        len(a) == 6 and len(b) != 6 and _diagonal(a, b)
                        for a in comps for b in comps)
    assert review["crowded"] >= 8 and review["three_objects"] >= 8
    assert review["matched_first"] >= 5 and review["unmatched_first"] >= 5
    assert review["familiar_test"] >= 3
    assert review["novel_small"] >= 2 and review["novel_large"] >= 3
    assert review["novel_eight"] >= 2
    assert review["zero_target_test"] >= 2 and review["one_target_test"] >= 2
    assert review["diagonal_test"] >= 2 and len(target_shapes) >= 15
