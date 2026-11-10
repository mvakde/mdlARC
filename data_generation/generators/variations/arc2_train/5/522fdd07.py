"""Centered solid squares follow the cycle 1 -> 9 -> 7 -> 5 -> 3 -> 1."""

import numpy as np

_NEXT = {1: 9, 3: 1, 5: 3, 7: 5, 9: 7}
_SOURCE = {
    (7, ((1, 5, 13, 5), (3, 7, 4, 4), (4, 3, 1, 14), (6, 3, 6, 12), (9, 3, 10, 14))),
    (7, ((0, 3, 10, 11), (6, 9, 9, 4), (9, 5, 2, 13))),
    (7, ((0, 3, 4, 1), (2, 3, 11, 2), (5, 1, 5, 9), (8, 5, 13, 12))),
    (7, ((1, 1, 4, 11), (2, 3, 13, 13), (5, 3, 4, 4), (8, 5, 12, 6))),
    (7, ((0, 1, 11, 4), (1, 3, 3, 5), (3, 3, 1, 1), (4, 5, 13, 13), (9, 7, 3, 12))),
}


def _box(r, c, side):
    d = side // 2
    return r - d, r + d, c - d, c + d


def _separate(a, b):
    return a[1] + 1 < b[0] or b[1] + 1 < a[0] or a[3] + 1 < b[2] or b[3] + 1 < a[2]


def _inside(box, shape):
    return 0 <= box[0] <= box[1] < shape[0] and 0 <= box[2] <= box[3] < shape[1]


def _pack(rng, sizes, shape):
    order = sorted(range(len(sizes)), key=lambda i: max(sizes[i], _NEXT[sizes[i]]), reverse=True)
    candidates = {}
    for i, side in enumerate(sizes):
        margin = max(side, _NEXT[side]) // 2
        centers = [(r, c) for r in range(margin, shape[0] - margin)
                   for c in range(margin, shape[1] - margin)]
        rng.shuffle(centers)
        if side in (1, 9):
            target = int(rng.choice(3, p=[0.34, 0.42, 0.24]))
            centers.sort(key=lambda rc: abs(
                int(rc[0] in (margin, shape[0] - margin - 1)) +
                int(rc[1] in (margin, shape[1] - margin - 1)) - target))
        candidates[i] = [(r, c, _box(r, c, side), _box(r, c, _NEXT[side]))
                         for r, c in centers]
    placed = {}
    visits = 0

    def search(depth):
        nonlocal visits
        if depth == len(order):
            return True
        i = order[depth]
        for candidate in candidates[i]:
            visits += 1
            if visits > 80000:
                return False
            if all(_separate(candidate[2], old[2]) and _separate(candidate[3], old[3])
                   for old in placed.values()):
                placed[i] = candidate
                if search(depth + 1):
                    return True
                del placed[i]
        return False
    if not search(0):
        raise ValueError("No square packing")
    return [(placed[i][0], placed[i][1]) for i in range(len(sizes))]


def _draw(shape, background, sizes, colors, centers):
    inp = np.full(shape, background, dtype=np.int8)
    out = np.full(shape, background, dtype=np.int8)
    for side, color, (r, c) in zip(sizes, colors, centers):
        for grid, width in ((inp, side), (out, _NEXT[side])):
            r0, r1, c0, c1 = _box(r, c, width)
            grid[r0:r1 + 1, c0:c1 + 1] = color
    return {"input": inp, "output": out}


def _components(grid, background):
    h, w = grid.shape
    seen = np.zeros(grid.shape, dtype=bool)
    objects = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == background:
                continue
            color = int(grid[r, c])
            seen[r, c] = True
            cells = [(r, c)]
            for rr, cc in cells:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            cells.append((nr, nc))
            r0, r1 = min(x for x, _ in cells), max(x for x, _ in cells)
            c0, c1 = min(x for _, x in cells), max(x for _, x in cells)
            side = r1 - r0 + 1
            assert side in _NEXT and c1 - c0 + 1 == side and len(cells) == side * side
            assert np.all(grid[r0:r1 + 1, c0:c1 + 1] == color)
            objects.append((color, side, (r0 + r1) // 2, (c0 + c1) // 2))
    assert objects
    boxes = [_box(r, c, side) for _, side, r, c in objects]
    assert all(_separate(boxes[i], boxes[j]) for i in range(len(boxes))
               for j in range(i + 1, len(boxes)))
    return objects


def _check_pair(pair, background):
    inp, out = pair["input"], pair["output"]
    objects = _components(inp, background)
    expected = np.full(inp.shape, background, dtype=np.int8)
    boxes = []
    for color, side, r, c in objects:
        box = _box(r, c, _NEXT[side])
        assert _inside(box, inp.shape) and all(_separate(box, old) for old in boxes)
        boxes.append(box)
        r0, r1, c0, c1 = box
        expected[r0:r1 + 1, c0:c1 + 1] = color
    assert np.array_equal(expected, out)
    assert sorted(_components(out, background)) == sorted(
        (color, _NEXT[side], r, c) for color, side, r, c in objects)
    return objects


def _analyze(pair):
    answers = []
    for background in set(int(v) for v in np.unique(pair["input"])):
        try:
            objects = _components(pair["input"], background)
        except AssertionError:
            continue
        answers.append((background, objects))
    assert len(answers) == 1
    background, _ = answers[0]
    return background, _check_pair(pair, background)


def _signature(background, objects):
    return background, tuple(sorted(objects))


def _validate_counts(num_train, num_test):
    for count, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if count is not None and (
            isinstance(count, (bool, np.bool_)) or
            not isinstance(count, (int, np.integer)) or not low <= count <= high
        ):
            raise ValueError("Invalid pair count")
    if num_train is not None and num_test is not None and num_train <= num_test:
        raise ValueError("Training count must exceed test count")


def _sizes(rng, split):
    count = int(rng.choice((1, 2, 3, 4, 5), p=(0.13, 0.20, 0.25, 0.26, 0.16)))
    if split == "test" and rng.random() < 0.12:
        return [1, 1, 9]
    return [int(x) for x in rng.choice((1, 3, 5, 7, 9), size=count, replace=True)]


def _shape(rng, sizes, attempt):
    big = sum(side in (1, 9) for side in sizes)
    floor = max(11, 10 + 2 * len(sizes), 19 if big >= 2 else 11)
    if len(sizes) >= 4 and big:
        floor = max(floor, 20)
    if attempt >= 8:
        floor = max(floor, 26)
    if attempt >= 15:
        return 30, 30
    if attempt < 8 and floor <= 16 and rng.random() < 0.22:
        return 16, 16
    return int(rng.integers(floor, 31)), int(rng.integers(floor, 31))


def _colors(rng, sizes, background, pool, split):
    available = [c for c in pool if c != background]
    if split == "test":
        novel = [c for c in range(10) if c != background and c not in pool]
        if novel and rng.random() < 0.55:
            available.append(int(rng.choice(novel)))
    if not available:
        available = [c for c in range(10) if c != background]
    if len(sizes) >= 2 and rng.random() < 0.26:
        colors = [int(x) for x in rng.choice(available, size=len(sizes), replace=True)]
        i, j = rng.choice(len(sizes), size=2, replace=False)
        colors[int(j)] = colors[int(i)]
        return colors
    return [int(x) for x in rng.choice(
        available, size=len(sizes), replace=len(sizes) > len(available))]


def _make_pair(rng, sizes, background, pool, split, seen):
    for attempt in range(20):
        shape = _shape(rng, sizes, attempt)
        colors = _colors(rng, sizes, background, pool, split)
        try:
            centers = _pack(rng, sizes, shape)
        except ValueError:
            continue
        pair = _draw(shape, background, sizes, colors, centers)
        try:
            parsed_background, _ = _analyze(pair)
        except AssertionError:
            continue
        if parsed_background != background:
            continue
        key = (shape, pair["input"].tobytes())
        objects = [(color, side, r, c) for side, color, (r, c)
                   in zip(sizes, colors, centers)]
        if key in seen or _signature(background, objects) in _SOURCE:
            continue
        seen.add(key)
        return pair
    raise RuntimeError("Could not generate a distinct square scene")


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    _validate_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    if num_train is None:
        num_train = int(rng.integers(max(2, (num_test or 0) + 1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    _validate_counts(num_train, num_test)

    # Color has no puzzle-level meaning. The pool keeps training palettes readable.
    pool = [int(x) for x in rng.choice(10, size=int(rng.integers(4, 7)), replace=False)]
    fixed_background = rng.random() < 0.38
    backgrounds = [int(x) for x in rng.choice(10, size=num_train + num_test)]
    if fixed_background:
        backgrounds = [backgrounds[0]] * len(backgrounds)
    else:
        while backgrounds[1] == backgrounds[0]:
            backgrounds[1] = int(rng.integers(0, 10))
    states = [int(x) for x in rng.permutation((1, 3, 5, 7, 9))]
    cut = int(rng.choice((1, 2, 3, 4), p=(0.17, 0.33, 0.33, 0.17)))
    core = [states[:cut], states[cut:]]
    rng.shuffle(core)
    train_sizes = core + [_sizes(rng, "train") for _ in range(num_train - 2)]
    rng.shuffle(train_sizes)
    test_sizes = [_sizes(rng, "test") for _ in range(num_test)]
    puzzle = {"train": [], "test": []}
    seen = set()
    choices = [("train", s) for s in train_sizes] + [("test", s) for s in test_sizes]
    for index, (split, sizes) in enumerate(choices):
        pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
        pair = _make_pair(pair_rng, sizes, backgrounds[index], pool, split, seen)
        puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    n_train, n_test = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= n_train <= 10 and 1 <= n_test <= 4 and n_train > n_test
    seen = set()
    train_sizes = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2 and grid.dtype.kind in "iu"
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair["input"].shape == pair["output"].shape
            key = (pair["input"].shape, pair["input"].tobytes())
            assert key not in seen
            seen.add(key)
            assert not np.array_equal(pair["input"], pair["output"])
            background, objects = _analyze(pair)
            assert _signature(background, objects) not in _SOURCE
            if split == "train":
                train_sizes.update(side for _, side, _, _ in objects)
    assert train_sizes == set(_NEXT)


def _same_puzzle(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
        for k in ("input", "output")
    ) for s in ("train", "test"))


def self_test() -> None:
    counts_train, counts_test = set(), set()
    shapes, compositions, centers, backgrounds = set(), set(), set(), set()
    repeated, small, rectangular, exact_16, multiple_large = 0, 0, 0, 0, 0
    test_nine, test_without_one, test_small, test_novel = 0, 0, 0, 0
    varied_counts, varied_shapes, varied_backgrounds, early_nine = 0, 0, 0, 0
    large_centers = {1: set(), 9: set()}
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        counts_train.add(len(puzzle["train"]))
        counts_test.add(len(puzzle["test"]))
        train_colors = {color for pair in puzzle["train"]
                        for color, _, _, _ in _analyze(pair)[1]}
        local_counts, local_shapes, local_backgrounds = set(), set(), set()
        first_nine, first_one = None, None
        for split in ("train", "test"):
            for index, pair in enumerate(puzzle[split]):
                bg, objects = _analyze(pair)
                backgrounds.add(bg)
                local_backgrounds.add(bg)
                shapes.add(pair["input"].shape)
                local_shapes.add(pair["input"].shape)
                local_counts.add(len(objects))
                sizes = tuple(sorted(side for _, side, _, _ in objects))
                compositions.add((split, sizes))
                centers.update((side, r, c) for _, side, r, c in objects)
                repeated += len({color for color, _, _, _ in objects}) < len(objects)
                small += len(objects) <= 2
                rectangular += pair["input"].shape[0] != pair["input"].shape[1]
                exact_16 += pair["input"].shape == (16, 16)
                multiple_large += sum(side in (1, 9) for side in sizes) >= 2
                for _, side, r, c in objects:
                    if side in (1, 9):
                        shape = pair["input"].shape
                        edges = int(r in (4, shape[0] - 5)) + int(c in (4, shape[1] - 5))
                        large_centers[side].add(edges)
                if split == "train":
                    if 9 in sizes and first_nine is None:
                        first_nine = index
                    if 1 in sizes and first_one is None:
                        first_one = index
                else:
                    test_nine += 9 in sizes
                    test_without_one += 1 not in sizes
                    test_small += len(objects) <= 2
                    test_novel += any(color not in train_colors for color, _, _, _ in objects)
        varied_counts += len(local_counts) > 1
        varied_shapes += len(local_shapes) > 1
        varied_backgrounds += len(local_backgrounds) > 1
        early_nine += first_nine < first_one
    assert len(counts_train) >= 4 and len(counts_test) >= 3
    assert len(shapes) >= 25 and len(compositions) >= 30 and len(centers) >= 100
    assert len(backgrounds) >= 8 and repeated >= 10 and small >= 10
    assert rectangular >= 20 and exact_16 >= 5 and multiple_large >= 5
    assert test_nine >= 5 and test_without_one >= 5 and test_small >= 5 and test_novel >= 5
    assert varied_counts >= 20 and varied_shapes >= 20 and varied_backgrounds >= 5
    assert 0 < early_nine < 42
    assert large_centers[1] == large_centers[9] == {0, 1, 2}
    assert not _same_puzzle(puzzles[0], puzzles[1])
    assert _same_puzzle(generate_puzzle(17), generate_puzzle(17))
    for seed, train, test in ((101, 2, 1), (102, 10, 1), (103, 10, 4),
                              (104, None, 1), (105, None, 4),
                              (106, 2, None), (107, 10, None)):
        a = generate_puzzle(seed, train, test)
        assert _same_puzzle(a, generate_puzzle(seed, train, test))
        validate_generated_puzzle(a)
    for train, test in ((1, None), (11, None), (None, 0), (None, 5),
                        (2, 2), (3, 4)):
        try:
            generate_puzzle(500, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
