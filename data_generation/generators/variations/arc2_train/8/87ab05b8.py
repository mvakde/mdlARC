"""Noisy marker selects the midpoint quadrant to fill."""

import numpy as np


_SOURCE_PAIRS = (
    (((6,6,6,6),(6,9,6,1),(4,6,6,2),(6,6,5,6)),
     ((6,6,6,6),(6,6,6,6),(6,6,2,2),(6,6,2,2))),
    (((5,6,0,6),(6,6,6,6),(6,2,6,6),(6,6,6,4)),
     ((6,6,6,6),(6,6,6,6),(2,2,6,6),(2,2,6,6))),
    (((6,9,0,0),(9,6,1,6),(6,6,6,1),(8,6,6,2)),
     ((6,6,6,6),(6,6,6,6),(6,6,2,2),(6,6,2,2))),
    (((2,6,8,1),(6,6,6,6),(4,6,9,9),(0,5,6,6)),
     ((2,2,6,6),(2,2,6,6),(6,6,6,6),(6,6,6,6))),
)
_SHAPES = ((4, 6), (6, 4), (6, 6))


def _count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return value


def _counts(rng, num_train, num_test):
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train is None:
        if num_test is None:
            num_train = int(rng.choice(
                [2, 3, 4, 5, 6, 7, 8, 9, 10],
                p=[.05, .32, .30, .12, .08, .05, .04, .025, .015]))
        else:
            choices = np.arange(max(2, num_test + 1), 11)
            weights = np.array([1 / (1 + abs(int(x) - 4)) for x in choices], dtype=float)
            num_train = int(rng.choice(choices, p=weights / weights.sum()))
    if num_test is None:
        choices = np.arange(1, min(4, num_train - 1) + 1)
        weights = np.array([.45, .30, .18, .07][:len(choices)], dtype=float)
        num_test = int(rng.choice(choices, p=weights / weights.sum()))
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return num_train, num_test


def _quadrant(position, shape):
    return (int(position[0] >= shape[0] // 2), int(position[1] >= shape[1] // 2))


def _cells_in_quadrant(shape, q):
    h, w = shape
    return [(r, c) for r in range(q[0] * (h // 2), (q[0] + 1) * (h // 2))
            for c in range(q[1] * (w // 2), (q[1] + 1) * (w // 2))]


def _output(shape, background, signal, position):
    out = np.full(shape, background, dtype=np.int64)
    q = _quadrant(position, shape)
    out[q[0] * (shape[0] // 2):(q[0] + 1) * (shape[0] // 2),
        q[1] * (shape[1] // 2):(q[1] + 1) * (shape[1] // 2)] = signal
    return out


def _position(rng, shape, q, offset=None):
    if offset is None:
        offset = (int(rng.integers(shape[0] // 2)), int(rng.integers(shape[1] // 2)))
    return (q[0] * (shape[0] // 2) + offset[0],
            q[1] * (shape[1] // 2) + offset[1])


def _pair(rng, shape, q, background, signal, palette, mode=None,
          position=None, required_hue=None):
    if position is None:
        position = _position(rng, shape, q)
    if mode is None:
        mode = str(rng.choice(
            ["scatter", "run", "patch", "decoy", "clear", "filled"],
            p=[.27, .16, .14, .16, .13, .14]))
    area = shape[0] * shape[1]
    if area == 16:
        band = int(rng.choice(3, p=[.27, .43, .30]))
        lo, hi = [(1, 2), (3, 5), (6, 7)][band]
    else:
        band = int(rng.choice(3, p=[.24, .40, .36]))
        lo, hi = [(2, 4), (5, 8), (9, min(14, (area - 2) // 2))][band]
    n = int(rng.integers(lo, hi + 1))
    if mode == "decoy":
        n = max(4, n)
    if mode in ("run", "patch"):
        n = max(3, n)
    eligible = [(r, c) for r in range(shape[0]) for c in range(shape[1])
                if (r, c) != position]
    rng.shuffle(eligible)
    chosen = []
    locked = 0
    if mode == "decoy":
        other = [x for x in ((0, 0), (0, 1), (1, 0), (1, 1)) if x != q]
        dq = other[int(rng.integers(len(other)))]
        qr, qc = _position(rng, shape, dq, (0, 0))
        chosen = [(qr + dr, qc + dc) for dr in range(2) for dc in range(2)]
        locked = 4
    elif mode == "run":
        horizontal = bool(rng.integers(2))
        for anchor in eligible:
            run = [(anchor[0] + (0 if horizontal else k),
                    anchor[1] + (k if horizontal else 0)) for k in range(3)]
            if all(x in eligible for x in run):
                chosen = run
                locked = 3
                break
    elif mode == "patch":
        for anchor in eligible:
            block = [(anchor[0] + dr, anchor[1] + dc)
                     for dr in range(2) for dc in range(2)]
            patch = [x for x in block if x in eligible]
            if len(patch) >= 3:
                chosen = patch[:min(n, len(patch))]
                locked = len(chosen)
                break
    if mode == "clear":
        eligible = [x for x in eligible if _quadrant(x, shape) != q]
    elif mode == "filled":
        inside = [x for x in eligible if _quadrant(x, shape) == q]
        outside = [x for x in eligible if _quadrant(x, shape) != q]
        eligible = inside + outside
    for x in eligible:
        if len(chosen) >= n:
            break
        if x not in chosen:
            chosen.append(x)
    n = len(chosen)
    colors = list(palette)
    if required_hue is not None:
        assert required_hue in colors
    rng.shuffle(colors)
    if required_hue is not None:
        colors.remove(required_hue)
        colors.insert(0, required_hue)
    first = colors[0]
    labels = [first] * locked
    remainder = n - locked
    if remainder:
        # Both repeated hues and many singletons are possible in either split.
        max_distinct = min(remainder, len(colors))
        lower = 3 if max_distinct >= 4 and rng.random() < .65 else 1
        distinct = int(rng.integers(lower, max_distinct + 1))
        selected = colors[:distinct]
        labels += selected
        labels += [int(x) for x in rng.choice(selected, size=remainder - distinct)]
        tail = labels[locked:]
        rng.shuffle(tail)
        labels[locked:] = tail
    if required_hue is not None and required_hue not in labels:
        labels[-1] = required_hue
    inp = np.full(shape, background, dtype=np.int64)
    for cell, color in zip(chosen, labels):
        inp[cell] = int(color)
    inp[position] = signal
    return {"input": inp, "output": _output(shape, background, signal, position)}


def _reposition(pair, old, new, background, signal):
    inp = pair["input"].copy()
    inp[old] = background
    inp[new] = signal
    return {"input": inp, "output": _output(inp.shape, background, signal, new)}


def _grid_key(grid):
    return (grid.shape, tuple(int(x) for x in grid.flat))


def _source_copy(pair):
    inp = tuple(tuple(int(x) for x in row) for row in pair["input"])
    out = tuple(tuple(int(x) for x in row) for row in pair["output"])
    return (inp, out) in _SOURCE_PAIRS


def _has_decoy(inp, background, signal, selected):
    for q in ((0, 0), (0, 1), (1, 0), (1, 1)):
        if q == selected:
            continue
        cells = _cells_in_quadrant(inp.shape, q)
        rows = sorted({r for r, _ in cells})
        cols = sorted({c for _, c in cells})
        for r in rows[:-1]:
            for c in cols[:-1]:
                patch = {int(inp[r + dr, c + dc])
                         for dr in range(2) for dc in range(2)}
                if len(patch) == 1 and next(iter(patch)) not in (background, signal):
                    return True
    return False


def _roles(puzzle):
    signals = []
    for pair in puzzle["train"]:
        colors, counts = np.unique(pair["output"], return_counts=True)
        assert len(colors) == 2
        signals.append(int(colors[np.argmin(counts)]))
    assert len(set(signals)) == 1
    return signals[0]


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    signal = int(rng.integers(10))
    other_colors = [x for x in range(10) if x != signal]
    rng.shuffle(other_colors)
    variable_background = num_train >= 3 and bool(rng.random() < .20)
    backgrounds = tuple(other_colors[:2 if variable_background else 1])
    available = other_colors[len(backgrounds):]
    pool_choices = np.arange(2, len(available) + 1)
    pool_weights = np.array([min(int(x) - 1, 5) for x in pool_choices], dtype=float)
    pool_size = int(rng.choice(pool_choices, p=pool_weights / pool_weights.sum()))
    palette = tuple(available[:pool_size])
    reserved = palette[-1] if pool_size >= 3 and rng.random() < .48 else None
    train_palette = tuple(x for x in palette if x != reserved)
    extension = _SHAPES[int(rng.integers(3))] if rng.random() < .28 else None
    source_like = num_train <= 4 and bool(rng.random() < .22)
    quadrants = ((0, 0), (0, 1), (1, 0), (1, 1))
    if source_like:
        initial = (1, int(rng.integers(2)))
        different = (1, 1 - initial[1])
    else:
        initial = quadrants[int(rng.integers(4))]
        options = [q for q in quadrants
                   if (q[0] != initial[0]) + (q[1] != initial[1]) == 1]
        different = options[int(rng.integers(2))]
    train_q = [initial]
    if num_train == 2 or (extension is not None and num_train == 3):
        train_q.append(different)
    else:
        train_q += [initial, different]
    while len(train_q) < num_train:
        options = [q for q in quadrants if not source_like or q[0] == 1]
        if not source_like:
            fresh = [q for q in options if q not in train_q]
            if fresh:
                options = fresh
        train_q.append(options[int(rng.integers(len(options)))])
    train_shapes = [(4, 4)] * num_train
    if extension is not None:
        train_shapes[-1] = extension
    train_bgs = [backgrounds[0]] * num_train
    if variable_background:
        train_bgs[-1] = backgrounds[1]
        for i in range(3, num_train - 1):
            train_bgs[i] = backgrounds[int(rng.integers(2))]
    train = []
    seen = set()
    for i in range(num_train):
        pair_rng = np.random.default_rng(int(rng.bit_generator.random_raw()))
        if i in (1, 2) and len(train_q) >= 3 and train_q[0] == train_q[1] \
                and train_shapes[i] == (4, 4) and train_bgs[i] == train_bgs[0]:
            old = tuple(int(v) for v in np.argwhere(train[0]["input"] == signal)[0])
            if i == 1:
                choices = [x for x in _cells_in_quadrant((4, 4), train_q[0]) if x != old]
                new = choices[int(pair_rng.integers(len(choices)))]
            else:
                offset = (old[0] % 2, old[1] % 2)
                new = _position(pair_rng, (4, 4), train_q[i], offset)
            pair = _reposition(train[0], old, new, train_bgs[0], signal)
        else:
            for _ in range(32):
                pair = _pair(pair_rng, train_shapes[i], train_q[i], train_bgs[i], signal,
                             train_palette)
                if i == 0 and len(train_q) >= 3 and train_q[0] == train_q[1] \
                        and train_shapes[1] == (4, 4) and train_bgs[1] == train_bgs[0]:
                    clutter = (pair["input"] != train_bgs[0]) & (pair["input"] != signal)
                    if int(clutter.sum()) < 3:
                        continue
                if _grid_key(pair["input"]) not in seen and not _source_copy(pair):
                    break
        assert _grid_key(pair["input"]) not in seen and not _source_copy(pair)
        seen.add(_grid_key(pair["input"]))
        train.append(pair)
    train_quads = {_quadrant(tuple(np.argwhere(p["input"] == signal)[0]), p["input"].shape)
                   for p in train}
    test_q = []
    if source_like:
        test_q.append((0, int(rng.integers(2))))
    else:
        unseen = [q for q in quadrants if q not in train_quads]
        if unseen and rng.random() < .55:
            test_q.append(unseen[int(rng.integers(len(unseen)))])
        else:
            test_q.append(quadrants[int(rng.integers(4))])
    while len(test_q) < num_test:
        options = [q for q in quadrants if q != test_q[-1]]
        test_q.append(options[int(rng.integers(3))])
    test = []
    for i in range(num_test):
        pair_rng = np.random.default_rng(int(rng.bit_generator.random_raw()))
        shape = extension if extension is not None and (i == 0 or rng.random() < .55) else (4, 4)
        bg = backgrounds[int(rng.integers(len(backgrounds)))]
        required = reserved if reserved is not None and i == 0 and rng.random() < .75 else None
        for _ in range(64):
            pair = _pair(pair_rng, shape, test_q[i], bg, signal, palette,
                         required_hue=required)
            if _grid_key(pair["input"]) not in seen and not _source_copy(pair):
                break
        assert _grid_key(pair["input"]) not in seen and not _source_copy(pair)
        seen.add(_grid_key(pair["input"]))
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
    signal = _roles(puzzle)
    seen = set()
    train_quads = set()
    train_shapes = set()
    backgrounds = set()
    test_quads_seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            assert inp.shape in ((4, 4),) + _SHAPES
            key = _grid_key(inp)
            assert key not in seen
            seen.add(key)
            assert not _source_copy(pair)
            locations = np.argwhere(inp == signal)
            assert locations.shape == (1, 2)
            position = tuple(int(x) for x in locations[0])
            values, counts = np.unique(inp, return_counts=True)
            bg = int(values[np.argmax(counts)])
            assert bg != signal and np.count_nonzero(counts == counts.max()) == 1
            assert np.array_equal(out, _output(inp.shape, bg, signal, position))
            assert not np.array_equal(inp, out)
            assert 1 <= len(set(int(x) for x in values) - {bg, signal}) <= 8
            if split == "train":
                train_quads.add(_quadrant(position, inp.shape))
                train_shapes.add(inp.shape)
                backgrounds.add(bg)
            else:
                assert bg in backgrounds
                assert inp.shape in train_shapes
                test_quads_seen.add(_quadrant(position, inp.shape))
    assert len(train_quads) >= 2
    assert (4, 4) in train_shapes
    if len(puzzle["test"]) > 1:
        assert len(test_quads_seen) >= 2


def _puzzle_key(puzzle):
    return tuple((split, _grid_key(pair["input"]), _grid_key(pair["output"]))
                 for split in ("train", "test") for pair in puzzle[split])


def _observable_prediction(train, test_input):
    signals = []
    for pair in train:
        values, counts = np.unique(pair["output"], return_counts=True)
        signals.append(int(values[np.argmin(counts)]))
    assert len(set(signals)) == 1
    signal = signals[0]
    values, counts = np.unique(test_input, return_counts=True)
    background = int(values[np.argmax(counts)])
    locations = np.argwhere(test_input == signal)
    assert locations.shape == (1, 2)
    return _output(test_input.shape, background, signal, tuple(locations[0]))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in puzzles}) >= 4
    assert len({len(p["test"]) for p in puzzles}) >= 3
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(generate_puzzle(123)) == _puzzle_key(generate_puzzle(123))
    train_quads = set()
    test_quads = set()
    test_offsets = set()
    shapes = set()
    clutter_counts = {"train": set(), "test": set()}
    clutter_colors = {"train": set(), "test": set()}
    modes = {split: {"clean": 0, "full": 0, "decoy": 0}
             for split in ("train", "test")}
    short = 0
    mixed_tests = 0
    novel_test_hue = 0
    varied_background = 0
    familiar_tests = 0
    unseen_tests = 0
    upper_test_offsets = set()
    source_like = 0
    within_contrasts = 0
    across_contrasts = 0
    rich_tests = 0
    sparse_tests = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        signal = _roles(puzzle)
        train_bg = set()
        train_clutter = set()
        test_q_this = set()
        train_q_this = set()
        train_examples = []
        if len(puzzle["train"]) in (3, 4):
            short += 1
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                values, counts = np.unique(inp, return_counts=True)
                bg = int(values[np.argmax(counts)])
                pos = tuple(int(x) for x in np.argwhere(inp == signal)[0])
                q = _quadrant(pos, inp.shape)
                expected = np.full(inp.shape, bg, dtype=np.int64)
                h, w = inp.shape
                expected[q[0]*h//2:(q[0]+1)*h//2,
                         q[1]*w//2:(q[1]+1)*w//2] = signal
                assert np.array_equal(out, expected)
                clutter = (inp != bg) & (inp != signal)
                hues = set(int(x) for x in np.unique(inp[clutter]))
                clutter_counts[split].add(int(clutter.sum()))
                clutter_colors[split].add(len(hues))
                shapes.add(inp.shape)
                chosen = _cells_in_quadrant(inp.shape, q)
                inside = sum(bool(clutter[x]) for x in chosen)
                modes[split]["clean"] += inside == 0
                modes[split]["full"] += inside == len(chosen) - 1
                modes[split]["decoy"] += _has_decoy(inp, bg, signal, q)
                if split == "train":
                    train_quads.add(q)
                    train_q_this.add(q)
                    train_bg.add(bg)
                    train_clutter |= hues
                    train_examples.append((inp, out, q))
                else:
                    test_quads.add(q)
                    test_offsets.add((pos[0] % (h//2), pos[1] % (w//2)))
                    test_q_this.add(q)
                    novel_test_hue += bool(hues - train_clutter)
                    familiar_tests += q in train_q_this
                    unseen_tests += q not in train_q_this
                    rich_tests += len(hues) >= 5
                    sparse_tests += len(hues) <= 2
                    if q[0] == 0 and inp.shape == (4, 4):
                        upper_test_offsets.add((pos[0], pos[1] % 2))
                    assert np.array_equal(
                        _observable_prediction(puzzle["train"], inp.copy()), out)
        for a in range(len(train_examples)):
            for b in range(a + 1, len(train_examples)):
                ia, oa, qa = train_examples[a]
                ib, ob, qb = train_examples[b]
                if ia.shape != ib.shape:
                    continue
                changed = int(np.count_nonzero(ia != ib))
                within_contrasts += qa == qb and np.array_equal(oa, ob) and 0 < changed <= 3
                across_contrasts += (qa[0] != qb[0]) + (qa[1] != qb[1]) == 1 \
                    and not np.array_equal(oa, ob) and 0 < changed <= 3
        varied_background += len(train_bg) > 1
        mixed_tests += len(test_q_this) > 1
        source_like += train_q_this == {(1, 0), (1, 1)} and any(q[0] == 0 for q in test_q_this)
    role_assignments = set()
    for p in puzzles:
        values, counts = np.unique(p["train"][0]["output"], return_counts=True)
        role_assignments.add((int(values[np.argmax(counts)]), _roles(p)))
    assert len(role_assignments) >= 15
    assert len(train_quads) == len(test_quads) == 4
    assert len(test_offsets) >= 4
    assert len(upper_test_offsets) >= 3
    assert len(shapes) == 4
    assert len(clutter_counts["train"]) >= 6 and len(clutter_counts["test"]) >= 6
    assert max(clutter_colors["train"]) >= 4 and max(clutter_colors["test"]) >= 4
    assert min(clutter_colors["train"]) == min(clutter_colors["test"]) == 1
    assert all(all(k > 0 for k in modes[split].values()) for split in modes)
    assert mixed_tests >= 2 and novel_test_hue >= 1
    assert familiar_tests >= 1 and unseen_tests >= 1
    assert rich_tests >= 1 and sparse_tests >= 1
    assert source_like >= 1 and within_contrasts >= 5 and across_contrasts >= 5
    assert varied_background >= 1 and short >= 8
    for seed, a, b in ((501,2,1),(502,10,1),(503,10,4),
                       (601,None,1),(602,None,4),(603,2,None),(604,10,None)):
        p = generate_puzzle(seed, a, b)
        validate_generated_puzzle(p)
        assert (a is None or len(p["train"]) == a) and (b is None or len(p["test"]) == b)
        assert _puzzle_key(p) == _puzzle_key(generate_puzzle(seed, a, b))
    for request in ({"num_train":1},{"num_train":11},{"num_test":0},
                    {"num_test":5},{"num_train":2,"num_test":2},
                    {"num_train":3,"num_test":4}):
        try:
            generate_puzzle(700, **request)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError: {request}")
