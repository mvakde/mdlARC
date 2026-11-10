"""Synthetic ARC puzzles whose only shared rule is cellwise 2x enlargement."""

import numpy as np


_SOURCE_INPUT_KEYS = {
    (4, 4, (0, 0, 8, 0, 0, 8, 5, 5, 0, 0, 0, 5, 0, 0, 0, 0)),
    (3, 3, (0, 3, 0, 0, 7, 7, 0, 0, 0)),
    (5, 5, (0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 6, 6, 6, 0,
            0, 0, 1, 6, 0, 0, 0, 0, 0, 0)),
}


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(seed, num_train, num_test):
    train = None if num_train is None else _integer(num_train, "num_train")
    test = None if num_test is None else _integer(num_test, "num_test")
    if train is not None and not 2 <= train <= 10:
        raise ValueError("num_train must be 2..10")
    if test is not None and not 1 <= test <= 4:
        raise ValueError("num_test must be 1..4")
    if train is None and test is None:
        test = 1 + seed % 4
        train = max(2, test + 1) + (seed // 4) % (11 - max(2, test + 1))
    elif train is None:
        low = max(2, test + 1)
        train = low + seed % (11 - low)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _key(grid):
    return grid.shape + (tuple(int(x) for x in grid.flat),)


def _scale(grid):
    return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)


def _shape(rng, regime, train_shapes=()):
    """Sample dimensions independently except for deliberate test-size regimes."""
    if regime == "source_train":
        mode = int(rng.choice(4, p=[0.13, 0.34, 0.31, 0.22]))
    else:
        mode = int(rng.choice(5, p=[0.18, 0.24, 0.23, 0.20, 0.15]))
        if mode == 0:  # Simple extrapolation when there is space to grow.
            tallest = max(h for h, _ in train_shapes)
            widest = max(w for _, w in train_shapes)
            if tallest < 15 and widest < 15:
                return (int(rng.integers(tallest + 1, min(16, tallest + 5))),
                        int(rng.integers(widest + 1, min(16, widest + 5))))
            mode = 3
        elif mode == 4:  # Same-sized or smaller test; novelty lies elsewhere.
            h, w = train_shapes[int(rng.integers(len(train_shapes)))]
            if int(rng.integers(2)):
                return h, w
            h = max(1, h - int(rng.integers(0, 4)))
            w = max(1, w - int(rng.integers(0, 4)))
            if h == 1:
                w = max(3, w)
            if w == 1:
                h = max(3, h)
            return h, w
    if mode == 0:  # One-dimensional strip.
        length = int(rng.integers(3, 16))
        return (1, length) if int(rng.integers(2)) else (length, 1)
    if mode == 1:
        return int(rng.integers(2, 7)), int(rng.integers(2, 7))
    if mode == 2:
        return int(rng.integers(3, 11)), int(rng.integers(3, 11))
    return int(rng.integers(7, 16)), int(rng.integers(7, 16))


def _ranked_mask(rng, shape, count, motif, placement):
    h, w = shape
    rows, cols = np.indices(shape)
    eligible = np.ones(shape, dtype=bool)
    if placement == "center":
        eligible[0, :] = eligible[-1, :] = False
        eligible[:, 0] = eligible[:, -1] = False
    center_r, center_c = int(rng.integers(h)), int(rng.integers(w))
    noise = rng.random(shape)
    if motif == "scatter":
        score = noise
    elif motif == "cluster":
        centers = [(center_r, center_c)]
        if int(rng.integers(2)):
            centers.append((int(rng.integers(h)), int(rng.integers(w))))
        score = np.full(shape, h + w, dtype=float)
        for cr, cc in centers:
            score = np.minimum(score, abs(rows - cr) + abs(cols - cc))
        score += noise * 0.7
    elif motif == "band":
        direction = int(rng.integers(4))
        bases = (rows, cols, rows + cols, rows - cols)
        base = bases[direction]
        target = int(rng.integers(int(base.min()), int(base.max()) + 1))
        score = abs(base - target) + noise * 0.6
    elif motif == "checker":
        score = ((rows + cols + int(rng.integers(2))) % 2) * 2.0 + noise
    elif motif == "ring":
        radius = int(rng.integers(1, max(2, (h + w) // 3 + 1)))
        score = abs(abs(rows - center_r) + abs(cols - center_c) - radius) + noise * 0.6
    else:  # An elongated block with an independently sampled center.
        score = abs(rows - center_r) * float(rng.uniform(0.2, 1.8))
        score += abs(cols - center_c) * float(rng.uniform(0.2, 1.8))
        score += noise * 0.7
    forced_on = []
    forced_off = []
    if placement == "hole" and h >= 3 and w >= 3:
        cr, cc = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
        forced_off.append(cr * w + cc)
        forced_on.extend(((cr - 1) * w + cc, (cr + 1) * w + cc,
                          cr * w + cc - 1, cr * w + cc + 1))
    elif placement == "edge":
        edge = np.flatnonzero((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1))
        forced_on.append(int(rng.choice(edge)))
    flat_eligible = eligible.ravel().copy()
    flat_eligible[forced_off] = False
    flat_eligible[forced_on] = False
    available = np.flatnonzero(flat_eligible)
    count = max(len(forced_on), min(count, len(forced_on) + len(available)))
    chosen = available[np.argsort(score.ravel()[available], kind="stable")[:count - len(forced_on)]]
    mask = np.zeros(h * w, dtype=bool)
    mask[forced_on] = True
    mask[chosen] = True
    return mask.reshape(shape)


def _foreground_count(rng, shape, band):
    cells = shape[0] * shape[1]
    intervals = {
        "isolated": (1, min(3, max(1, cells // 7))),
        "sparse": (0.08, 0.25),
        "middle": (0.27, 0.65),
        "dense": (0.66, 0.88),
        "near_full": (0.89, 0.99),
    }
    low, high = intervals[band]
    if band == "isolated":
        return int(rng.integers(low, high + 1))
    return max(1, min(cells - 1, int(round(cells * float(rng.uniform(low, high))))))


def _colorize(rng, mask, background, color_count, style):
    count = int(mask.sum())
    color_count = min(color_count, count)
    available = [c for c in range(10) if c != background]
    colors = []
    if background != 0 and rng.random() < 0.62:
        colors.append(0)
        available.remove(0)
    if color_count > len(colors):
        colors.extend(int(v) for v in rng.choice(available, size=color_count - len(colors), replace=False))
    positions = np.argwhere(mask)
    values = np.empty(count, dtype=np.int8)
    if color_count == 1:
        values[:] = colors[0]
    elif style == "accent":
        values[:] = colors[0]
        picks = rng.choice(count, size=color_count - 1, replace=False)
        for i, pick in enumerate(picks):
            values[pick] = colors[i + 1]
    else:
        if style == "spatial":
            direction = int(rng.integers(4))
            r, c = positions[:, 0], positions[:, 1]
            scores = (r, c, r + c, r - c)[direction] + rng.random(count) * 0.15
            order = np.argsort(scores, kind="stable")
        else:
            order = rng.permutation(count)
        for i, chunk in enumerate(np.array_split(order, color_count)):
            values[chunk] = colors[i]
    grid = np.full(mask.shape, background, dtype=np.int8)
    grid[positions[:, 0], positions[:, 1]] = values
    return grid


def _evidence(grid):
    if min(grid.shape) < 3 or len(np.unique(grid)) < 3:
        return False
    if len({tuple(row) for row in grid}) <= 1 or len({tuple(col) for col in grid.T}) <= 1:
        return False
    # A middle cell of a different color makes copied spacing visible too.
    return bool(np.any((grid[:, :-2] == grid[:, 2:]) &
                       (grid[:, 1:-1] != grid[:, :-2])) or
                np.any((grid[:-2, :] == grid[2:, :]) &
                       (grid[1:-1, :] != grid[:-2, :])))


def _make_input(rng, shape, *, anchor=False, background=None, color_count=None):
    h, w = shape
    bands = ("isolated", "sparse", "middle", "dense", "near_full")
    motifs = ("scatter", "cluster", "band", "checker", "ring", "block")
    if anchor:
        band = "middle"
        placement = "free" if min(shape) < 5 else rng.choice(("free", "center", "edge"))
        color_count = int(rng.integers(2, 5))
    else:
        band = str(rng.choice(bands, p=[0.13, 0.20, 0.32, 0.20, 0.15]))
        placements = ("free", "center", "edge", "hole") if min(shape) >= 5 else ("free", "edge", "hole") if min(shape) >= 3 else ("free", "edge")
        placement = str(rng.choice(placements))
    if background is None:
        background = 0 if rng.random() < 0.58 else int(rng.integers(1, 10))
    if color_count is None:
        color_count = int(rng.choice((1, 2, 3, 4, 5), p=[0.25, 0.27, 0.23, 0.16, 0.09]))
    count = _foreground_count(rng, shape, band)
    mask = _ranked_mask(rng, shape, count, str(rng.choice(motifs)), placement)
    style = str(rng.choice(("balanced", "spatial", "accent"), p=[0.49, 0.36, 0.15]))
    return _colorize(rng, mask, background, color_count, style)


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    # Factor two is the only puzzle-level convention. Colors and geometry are cell values.
    classic_shapes = rng.random() < 0.15
    anchor_shape = (4, 4) if classic_shapes else (
        int(rng.integers(3, 9)), int(rng.integers(3, 9)))
    shapes = [anchor_shape] + [_shape(rng, "source_train") for _ in range(train_count - 1)]
    test_shapes = [_shape(rng, "target_test", shapes) for _ in range(test_count)]
    if classic_shapes:
        shapes[1] = (3, 3)
        test_shapes[0] = (5, 5)
    salts = rng.integers(0, 2**63, size=train_count + test_count, dtype=np.int64)
    puzzle = {"train": [], "test": []}
    seen = set()
    anchor_background = 0 if rng.random() < 0.58 else int(rng.integers(1, 10))
    for ordinal, (split, shape) in enumerate(
        [("train", s) for s in shapes] + [("test", s) for s in test_shapes]
    ):
        pair_rng = np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(ordinal, int(salts[ordinal]))))
        background = anchor_background if ordinal == 0 else None
        if ordinal == 1:
            background = (int(pair_rng.integers(1, 10)) if anchor_background == 0 else 0)
        for _ in range(32):
            grid = _make_input(pair_rng, shape, anchor=ordinal == 0,
                               background=background,
                               color_count=1 if ordinal == 1 else None)
            key = _key(grid)
            if key not in seen and key not in _SOURCE_INPUT_KEYS and (ordinal != 0 or _evidence(grid)):
                break
        else:
            # Distinct fallback patterns are guaranteed: fewer than 14 inputs can
            # precede this pair, and the 15x15 canvas offers 169 marker sites.
            shape = (15, 15)
            for site in range(13 * 13):
                grid = np.zeros(shape, dtype=np.int8)
                grid[0, 0] = grid[0, 2] = 1
                grid[2 + site // 13, 2 + site % 13] = 2
                key = _key(grid)
                if key not in seen:
                    break
        seen.add(key)
        puzzle[split].append({"input": grid, "output": _scale(grid)})
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "split lists")
    train, test = puzzle["train"], puzzle["test"]
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test), "pair counts")
    seen_inputs, seen_pairs = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            for role in ("input", "output"):
                grid = pair[role]
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, "grid type")
                _require(all(1 <= side <= 30 for side in grid.shape), "grid bounds")
                _require(np.issubdtype(grid.dtype, np.integer), "grid dtype")
                _require(bool(np.all((grid >= 0) & (grid <= 9))), "grid colors")
            inp, out = pair["input"], pair["output"]
            h, w = inp.shape
            _require(h <= 15 and w <= 15 and out.shape == (2 * h, 2 * w), "2x dimensions")
            _require(np.array_equal(out, _scale(inp)), "cellwise 2x copy")
            _require(len(np.unique(inp)) >= 2, "input must contain a visible pattern")
            inp_key, out_key = _key(inp), _key(out)
            _require(inp_key not in seen_inputs and (inp_key, out_key) not in seen_pairs, "duplicate")
            _require(inp_key not in _SOURCE_INPUT_KEYS, "source pair copied")
            seen_inputs.add(inp_key)
            seen_pairs.add((inp_key, out_key))
    _require(any(_evidence(pair["input"]) for pair in train), "training lacks two-axis evidence")


def _same_puzzle(first, second):
    return all(len(first[s]) == len(second[s]) and all(
        np.array_equal(a[role], b[role]) for a, b in zip(first[s], second[s])
        for role in ("input", "output")) for s in ("train", "test"))


def _components(mask):
    visited = np.zeros(mask.shape, dtype=bool)
    result = 0
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if visited[r, c]:
            continue
        result += 1
        stack = [(r, c)]
        visited[r, c] = True
        while stack:
            y, x = stack.pop()
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not visited[yy, xx]:
                    visited[yy, xx] = True
                    stack.append((yy, xx))
    return result


def _structure(grid):
    colors, counts = np.unique(grid, return_counts=True)
    background = int(colors[np.argmax(counts)])
    rows, cols = np.nonzero(grid != background)
    cropped = grid[rows.min():rows.max()+1, cols.min():cols.max()+1] if len(rows) else grid
    mapping = {background: 0}
    values = []
    for value in cropped.flat:
        value = int(value)
        if value not in mapping:
            mapping[value] = len(mapping)
        values.append(mapping[value])
    return cropped.shape + (tuple(values),)


def self_test() -> None:
    suite = [generate_puzzle(seed) for seed in range(42)]
    puzzle_keys = {tuple(tuple((_key(pair["input"]), _key(pair["output"]))
                               for pair in puzzle[split]) for split in ("train", "test"))
                   for puzzle in suite}
    _require(len(puzzle_keys) >= 40, "cross-seed puzzle diversity")
    for puzzle in suite:
        validate_generated_puzzle(puzzle)
        for pair in puzzle["train"] + puzzle["test"]:
            inp, out = pair["input"], pair["output"]
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    _require(bool(np.all(out[2*r:2*r+2, 2*c:2*c+2] == inp[r, c])), "independent block check")
            for color in np.unique(inp):
                _require(int(np.sum(out == color)) == 4 * int(np.sum(inp == color)), "color multiplicity")
    _require(len({len(p["train"]) for p in suite}) >= 4, "train count diversity")
    _require(len({len(p["test"]) for p in suite}) >= 3, "test count diversity")
    for seed, train, test in ((100, 2, 1), (101, 10, 1), (102, 10, 4),
                              (103, None, 1), (104, None, 4), (105, 2, None), (106, 10, None)):
        p = generate_puzzle(seed, train, test)
        validate_generated_puzzle(p)
        if train is not None:
            _require(len(p["train"]) == train, "explicit train count")
        if test is not None:
            _require(len(p["test"]) == test, "explicit test count")
        _require(_same_puzzle(p, generate_puzzle(seed, train, test)), "determinism")
    for kwargs in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                   {"num_test": 5}, {"num_train": 2, "num_test": 2},
                   {"num_train": 3, "num_test": 4}, {"num_train": 2.0},
                   {"num_test": True}):
        try:
            generate_puzzle(500, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"illegal counts accepted: {kwargs}")
    _require(_same_puzzle(generate_puzzle(77), generate_puzzle(77)), "automatic determinism")
    grids = [pair["input"] for p in suite for pair in p["train"] + p["test"]]
    tests = [(p, pair["input"]) for p in suite for pair in p["test"]]
    shapes = {g.shape for g in grids}
    _require(len({_key(g) for g in grids}) >= 0.95 * len(grids), "input diversity")
    _require(len(shapes) >= 35 and any(h == 1 for h, w in shapes) and any(w == 1 for h, w in shapes), "shape diversity")
    _require(any(h > w for h, w in shapes) and any(w > h for h, w in shapes), "aspect diversity")
    _require(any(max(g.shape) >= 12 for g in grids), "large canvas")
    _require(any(max(g.shape) <= 4 for g in grids), "small canvas")
    _require({2, 3, 4}.issubset({len(np.unique(g)) for g in grids}), "palette-size diversity")
    _require(any(0 not in g for g in grids), "zero must not be mandatory")
    _require(len({_structure(g) for g in grids}) >= 250, "structural diversity after recoloring/cropping")
    _require(any(_components(g != 0) >= 2 for g in grids), "disconnected motifs")
    _require(any(all(np.all(side == side.flat[0]) for side in (g[0], g[-1], g[:, 0], g[:, -1]))
                 and g[0, 0] == g[-1, -1] and np.any(g != g[0, 0]) for g in grids if min(g.shape) >= 3), "centered motifs")
    _require(any(len(np.unique(g)) >= 4 and int(np.min(np.unique(g, return_counts=True)[1])) >= 2
                 for g in grids), "balanced multicolor patterns")
    _require(any(len(np.unique(g)) >= 3 and int(np.min(np.unique(g, return_counts=True)[1])) == 1
                 for g in grids), "single-cell color accents")
    _require(any(np.max(np.unique(g, return_counts=True)[1]) / g.size >= .9 for g in grids), "near-full or near-empty grids")
    _require(any(np.any(g[0] != g[0, 0]) for g in grids), "edge-touching patterns")
    _require(any(any(g[r, c] != g[r-1, c] and g[r-1, c] == g[r+1, c] == g[r, c-1] == g[r, c+1]
                     for r in range(1, g.shape[0]-1) for c in range(1, g.shape[1]-1)) for g in grids), "interior holes")
    _require(any(np.all(np.concatenate((g[0], g[-1], g[:, 0], g[:, -1])) == g[0, 0])
                 and g[0, 0] != 0 and np.any(g == 0) for g in grids if min(g.shape) >= 3), "zero as interior foreground")
    _require(any(g.shape[0] <= max(t["input"].shape[0] for t in p["train"])
                 and g.shape[1] <= max(t["input"].shape[1] for t in p["train"])
                 for p, g in tests), "non-extrapolating tests")
    _require(sum(any(g.shape[0] > max(t["input"].shape[0] for t in p["train"])
                     and g.shape[1] > max(t["input"].shape[1] for t in p["train"])
                     for g in (x["input"] for x in p["test"])) for p in suite) >= 6,
             "size extrapolation diversity")
    _require(sum(len({len(np.unique(x["input"])) for x in p["train"] + p["test"]}) >= 2
                 for p in suite) >= 35, "within-puzzle color-count diversity")
    _require(sum(max(np.max(np.unique(x["input"], return_counts=True)[1]) / x["input"].size
                     for x in p["train"] + p["test"])
                 - min(np.max(np.unique(x["input"], return_counts=True)[1]) / x["input"].size
                       for x in p["train"] + p["test"]) >= .2 for p in suite) >= 30,
             "within-puzzle coverage diversity")
    _require(sum(len({tuple(np.unique(x["input"])) for x in p["train"] + p["test"]})
                 < len(p["train"]) + len(p["test"]) for p in suite) >= 3,
             "palette reuse must sometimes occur")
    _require(sum(any(any(q["input"].shape[0] <= t["input"].shape[0]
                         and q["input"].shape[1] <= t["input"].shape[1]
                         for t in p["train"]) for q in p["test"]) for p in suite) >= 15,
             "some tests must be same-sized or smaller")
    _require(sum(any(q["input"].shape in {t["input"].shape for t in p["train"]}
                     for q in p["test"]) for p in suite) >= 3, "equal-sized tests")
    _require(sum(any(np.mean(q["input"] != 0) < min(np.mean(t["input"] != 0) for t in p["train"])
                     for q in p["test"]) for p in suite) >= 5, "sparser tests")
    _require(sum((3, 3) in {t["input"].shape for t in p["train"]}
                 and (4, 4) in {t["input"].shape for t in p["train"]}
                 and (5, 5) in {q["input"].shape for q in p["test"]}
                 for p in suite) >= 2, "occasional source-like progression")
    _require(len({_key(g) for g in grids}) >= 100, "cross-seed difference")
