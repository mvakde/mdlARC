"""Generate sparse-lattice rectangle puzzles."""
import numpy as np

# Exact original inputs after removing the alternating scaffold.
_SOURCE = {
    (4, 2, (4, 4, 6, 0, 8, 8, 8, 4)),
    (2, 3, (8, 6, 3, 8, 4, 4)),
    (4, 4, (4, 8, 8, 8, 4, 4, 6, 8, 8, 3, 8, 8, 0, 8, 0, 8)),
    (2, 1, (6, 4)),
    (3, 3, (6, 3, 4, 8, 1, 7, 8, 7, 7)),
}
# Canvas height, width, and common outer-border/rectangle-gap thickness.
_CANVASES = ((26, 26, 2), (30, 26, 2), (26, 30, 2),
             (29, 26, 2), (26, 29, 2), (25, 25, 1),
             (29, 25, 1), (27, 27, 3), (28, 28, 4))


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _resolve_counts(seed, num_train, num_test):
    nt = _count("num_train", num_train, 2, 10)
    ns = _count("num_test", num_test, 1, 4)
    if nt is None and ns is None:
        ns = 1 + seed % 4
        choices = list(range(ns + 1, 11))
        nt = choices[(seed // 4) % len(choices)]
    elif nt is None:
        choices = list(range(ns + 1, 11))
        nt = choices[(seed // 4) % len(choices)]
    elif ns is None:
        ns = 1 + seed % min(4, nt - 1)
    if nt <= ns:
        raise ValueError("num_train must exceed num_test")
    return nt, ns


def _axes(interior, gap):
    return tuple(n for n in range(1, 15) if interior % n == 0 and interior // n > gap)


def _signature(coarse):
    return (coarse.shape[0], coarse.shape[1], tuple(int(x) for x in coarse.flat))


def _key(grid):
    return (grid.shape, np.asarray(grid, dtype=np.int16).tobytes())


def _runs(row, bg):
    found = []
    c = 0
    while c < len(row):
        color = int(row[c])
        if color == bg:
            c += 1
            continue
        end = c + 1
        while end < len(row) and int(row[end]) == color:
            end += 1
        found.append((color, c, end))
        c = end
    return found


def _render(coarse, bg, canvas, gap):
    height, width = canvas
    rows, cols = coarse.shape
    assert (height - gap) % rows == 0 and (width - gap) % cols == 0
    py, px = (height - gap) // rows, (width - gap) // cols
    assert py > gap and px > gap
    out = np.full(canvas, bg, dtype=np.int8)
    all_runs = [_runs(row, bg) for row in coarse]
    for r, runs in enumerate(all_runs):
        before = set(all_runs[r - 1]) if r else set()
        for color, left, right in runs:
            if (color, left, right) in before:
                continue
            depth = 1
            while r + depth < rows and (color, left, right) in all_runs[r + depth]:
                depth += 1
            y, x = gap + r * py, gap + left * px
            out[y:y + depth * py - gap, x:x + (right - left) * px - gap] = color
    return out


def _bridge_render(coarse, bg, canvas, gap):
    """Check the same rule by painting cores, then horizontal/vertical bridges."""
    height, width = canvas
    rows, cols = coarse.shape
    py, px = (height - gap) // rows, (width - gap) // cols
    out = np.full(canvas, bg, dtype=np.int8)
    for r in range(rows):
        for c in range(cols):
            color = int(coarse[r, c])
            if color == bg:
                continue
            y, x = gap + r * py, gap + c * px
            out[y:y + py - gap, x:x + px - gap] = color
            if c + 1 < cols and int(coarse[r, c + 1]) == color:
                out[y:y + py - gap, x + px - gap:x + px] = color
    for r in range(rows - 1):
        lower = set(_runs(coarse[r + 1], bg))
        for color, left, right in _runs(coarse[r], bg):
            if (color, left, right) in lower:
                y, x = gap + r * py, gap + left * px
                out[y + py - gap:y + py, x:x + (right - left) * px - gap] = color
    return out


def _encode(coarse, bg):
    rows, cols = coarse.shape
    grid = np.full((2 * rows + 1, 2 * cols + 1), bg, dtype=np.int8)
    grid[1::2, 1::2] = coarse
    return grid


def _decode(grid):
    height, width = grid.shape
    assert height >= 3 and width >= 3 and height % 2 and width % 2
    bg = int(grid[0, 0])
    assert np.all(grid[::2, :] == bg) and np.all(grid[:, ::2] == bg)
    coarse = grid[1::2, 1::2]
    assert np.any(coarse != bg)
    return coarse, bg


def _features(coarse, bg):
    runs = [_runs(row, bg) for row in coarse]
    horizontal = any(right - left > 1 for row in runs for _, left, right in row)
    exact = any(bool(set(a) & set(b)) for a, b in zip(runs, runs[1:]))
    unequal = any(ca == cb and (la, ra) != (lb, rb) and max(la, lb) < min(ra, rb)
                  for upper, lower in zip(runs, runs[1:])
                  for ca, la, ra in upper for cb, lb, rb in lower)
    return horizontal, exact, unequal


def _paint(coarse, row, left, right, color, bg):
    coarse[row, left:right] = color
    if left and int(coarse[row, left - 1]) == color:
        coarse[row, left - 1] = bg
    if right < coarse.shape[1] and int(coarse[row, right]) == color:
        coarse[row, right] = bg


def _inject(coarse, bg, colors, kind, rng):
    rows, cols = coarse.shape
    color = int(rng.choice(colors))
    if kind == "horizontal" and cols >= 2:
        row = int(rng.integers(rows))
        left = int(rng.integers(cols - 1))
        right = int(rng.integers(left + 2, cols + 1))
        _paint(coarse, row, left, right, color, bg)
    elif kind == "exact" and rows >= 2:
        depth = int(rng.integers(2, rows + 1))
        top = int(rng.integers(rows - depth + 1))
        left = int(rng.integers(cols))
        right = int(rng.integers(left + 1, cols + 1))
        for row in range(top, top + depth):
            _paint(coarse, row, left, right, color, bg)
    elif kind == "unequal" and rows >= 2 and cols >= 2:
        row = int(rng.integers(rows - 1))
        spans = [(a, b) for a in range(cols) for b in range(a + 1, cols + 1)]
        choices = [(a, b) for a in spans for b in spans
                   if a != b and max(a[0], b[0]) < min(a[1], b[1])]
        first, second = choices[int(rng.integers(len(choices)))]
        _paint(coarse, row, *first, color, bg)
        _paint(coarse, row + 1, *second, color, bg)
    elif kind == "return" and rows >= 3:
        first = int(rng.integers(rows - 2))
        last = int(rng.integers(first + 2, rows))
        left = int(rng.integers(cols))
        right = int(rng.integers(left + 1, cols + 1))
        for row in (first, last):
            _paint(coarse, row, left, right, color, bg)
        coarse[first + 1, left:right] = bg


def _sample_coarse(rng, shape, bg, role, corners):
    rows, cols = shape
    available = [x for x in range(10) if x != bg]
    k = int(rng.integers(1, min(8, rows * cols, 9) + 1))
    colors = [int(x) for x in rng.choice(available, size=k, replace=False)]
    if bg and 0 not in colors and rng.random() < 0.4:
        colors[0] = 0
    density = float(rng.choice((0.25, 0.4, 0.6, 0.8, 1.0)))
    coarse = np.full(shape, bg, dtype=np.int8)
    mask = rng.random(shape) < density
    choices = rng.choice(colors, size=shape)
    coarse[mask] = choices[mask]
    if role in ("exact", "unequal"):
        _inject(coarse, bg, colors, role, rng)
    else:
        kinds = ("horizontal", "exact", "unequal", "return")
        for _ in range(int(rng.integers(4))):
            _inject(coarse, bg, colors, kinds[int(rng.integers(4))], rng)
    if corners:
        coarse[0, 0] = int(rng.choice(colors))
        coarse[-1, -1] = int(rng.choice(colors))
    if not np.any(coarse != bg):
        coarse[int(rng.integers(rows)), int(rng.integers(cols))] = int(rng.choice(colors))
    return coarse


def _train_shapes(rng, row_axes, col_axes, classic, count):
    candidates = [(r, c) for r in row_axes if r >= 2 for c in col_axes if c >= 3]
    if classic:
        first = (2, int(rng.choice([c for c in col_axes if c >= 3])))
        alternatives = [s for s in candidates if s[0] == 4 and s[1] != first[1]]
    else:
        first = candidates[int(rng.integers(len(candidates)))]
        alternatives = [s for s in candidates if s[0] != first[0] and s[1] != first[1]]
    second = alternatives[int(rng.integers(len(alternatives)))]
    shapes = [first, second]
    all_shapes = ([(r, c) for r in (2, 4) for c in (1, 2, 3, 4)] if classic else
                  [(r, c) for r in row_axes for c in col_axes])
    while len(shapes) < count:
        shapes.append(all_shapes[int(rng.integers(len(all_shapes)))])
    return shapes


def _test_shape(rng, row_axes, col_axes, train_shapes, classic):
    if classic:
        return 3, 3
    train_rows = {r for r, _ in train_shapes}
    train_cols = {c for _, c in train_shapes}
    all_shapes = [(r, c) for r in row_axes for c in col_axes if r * c > 1]
    held = [s for s in all_shapes if s[0] not in train_rows or s[1] not in train_cols]
    shared = [s for s in all_shapes if s[0] in train_rows and s[1] in train_cols]
    recombined = [s for s in shared if s not in train_shapes]
    mode = int(rng.integers(3))
    pool = (held if mode == 0 and held else
            recombined if mode == 1 and recombined else
            shared if mode == 2 and shared else all_shapes)
    return pool[int(rng.integers(len(pool)))]


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    nt, ns = _resolve_counts(seed, num_train, num_test)
    forced_classic = seed % 9 == 0
    height, width, gap = ((26, 26, 2) if forced_classic else
                          _CANVASES[int(rng.integers(len(_CANVASES)))])
    row_axes, col_axes = _axes(height - gap, gap), _axes(width - gap, gap)
    classic = forced_classic or ((height, width, gap) == (26, 26, 2) and rng.random() < 0.45)
    shapes = _train_shapes(rng, row_axes, col_axes, classic, nt)
    roles = ["exact", "unequal"] + ["random"] * (nt - 2)
    order = rng.permutation(nt)
    train_specs = [(shapes[int(i)], roles[int(i)]) for i in order]
    test_shapes = [_test_shape(rng, row_axes, col_axes, shapes, classic) for _ in range(ns)]
    children = np.random.SeedSequence([seed, 834715]).spawn(nt + ns)
    seen = set()
    puzzle = {"train": [], "test": []}
    first_bg = None
    specs = [("train", *spec) for spec in train_specs] + [("test", shape, "random") for shape in test_shapes]
    for i, (split, shape, role) in enumerate(specs):
        pair_rng = np.random.default_rng(children[i])
        if split == "train" and first_bg is not None:
            bg = int(pair_rng.choice([x for x in range(10) if x != first_bg]))
        else:
            bg = int(pair_rng.integers(10))
        if first_bg is None:
            first_bg = bg
        for _ in range(300):
            coarse = _sample_coarse(pair_rng, shape, bg, role,
                                    split == "train" and len(puzzle["train"]) == 0)
            features = _features(coarse, bg)
            if role == "exact" and not (features[0] and features[1]):
                continue
            if role == "unequal" and not features[2]:
                continue
            if bg == 0 and _signature(coarse) in _SOURCE:
                continue
            inp = _encode(coarse, bg)
            key = _key(inp)
            if key in seen:
                continue
            seen.add(key)
            puzzle[split].append({"input": inp, "output": _render(coarse, bg, (height, width), gap)})
            break
        else:
            raise RuntimeError(f"could not construct {split} pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    pairs = puzzle["train"] + puzzle["test"]
    for pair in pairs:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
            assert np.issubdtype(grid.dtype, np.integer)
            assert int(grid.min()) >= 0 and int(grid.max()) <= 9
    first_in, first_out = puzzle["train"][0]["input"], puzzle["train"][0]["output"]
    first_coarse, first_bg = _decode(first_in)
    assert int(first_coarse[0, 0]) != first_bg and int(first_coarse[-1, -1]) != first_bg
    height, width = first_out.shape
    ys, xs = np.nonzero(first_out != first_bg)
    assert len(ys) > 0
    gap = int(ys.min())
    assert gap >= 1 and int(xs.min()) == gap
    assert height - 1 - int(ys.max()) == gap and width - 1 - int(xs.max()) == gap
    assert (height, width, gap) in _CANVASES
    seen_inputs, seen_pairs = set(), set()
    shapes, train_features, train_bgs = [], [False] * 3, set()
    for i, pair in enumerate(pairs):
        inp, out = pair["input"], pair["output"]
        coarse, bg = _decode(inp)
        rows, cols = coarse.shape
        assert rows in _axes(height - gap, gap) and cols in _axes(width - gap, gap)
        assert out.shape == (height, width)
        assert np.array_equal(out, _render(coarse, bg, (height, width), gap))
        assert set(int(x) for x in np.unique(inp)) == set(int(x) for x in np.unique(out))
        ik, pk = _key(inp), (_key(inp), _key(out))
        assert ik not in seen_inputs and pk not in seen_pairs
        seen_inputs.add(ik)
        seen_pairs.add(pk)
        assert not (bg == 0 and _signature(coarse) in _SOURCE)
        if i < len(puzzle["train"]):
            shapes.append((rows, cols))
            train_bgs.add(bg)
            train_features = [a or b for a, b in zip(train_features, _features(coarse, bg))]
    assert all(train_features)
    assert len({r for r, _ in shapes}) >= 2 and len({c for _, c in shapes}) >= 2
    assert len(train_bgs) >= 2


def _equal(a, b):
    return all(np.array_equal(x[k], y[k]) for split in ("train", "test")
               for x, y in zip(a[split], b[split]) for k in ("input", "output"))


def self_test() -> None:
    assert _equal(generate_puzzle(12345, 7, 3), generate_puzzle(12345, 7, 3))
    for i, counts in enumerate(((2, 1), (10, 1), (10, 4))):
        puzzle = generate_puzzle(20000 + i, *counts)
        assert (len(puzzle["train"]), len(puzzle["test"])) == counts
    for i, args in enumerate(({"num_train": None, "num_test": 1},
                              {"num_train": None, "num_test": 4},
                              {"num_train": 2, "num_test": None},
                              {"num_train": 10, "num_test": None})):
        p = generate_puzzle(21000 + i, **args)
        for name, value in args.items():
            if value is not None:
                assert len(p["train" if name == "num_train" else "test"]) == value
    for i, args in enumerate(({"num_train": 1}, {"num_train": 11},
                              {"num_test": 0}, {"num_test": 5},
                              {"num_train": 2, "num_test": 2},
                              {"num_train": 2, "num_test": 4},
                              {"num_train": 2.0, "num_test": 1})):
        try:
            generate_puzzle(22000 + i, **args)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    train_counts, test_counts = set(), set()
    canvases, shapes, bgs, layouts, keys = set(), set(), set(), set(), set()
    held, shared, holes, one_axis, large_axis = 0, 0, 0, 0, 0
    classic_shift = empty_test_rows = empty_test_cols = 0
    test_full = test_exact = test_unequal = deep_test = 0
    zero_payload = test_new_colors = test_familiar_colors = 0
    train_blank_edges = test_returns = 0
    test_palette_sizes = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        train_counts.add(len(puzzle["train"]))
        test_counts.add(len(puzzle["test"]))
        canvas = puzzle["train"][0]["output"].shape
        bg = int(puzzle["train"][0]["input"][0, 0])
        gap = int(np.nonzero(puzzle["train"][0]["output"] != bg)[0].min())
        canvases.add((*canvas, gap))
        train_shapes = {((p["input"].shape[0] - 1) // 2,
                         (p["input"].shape[1] - 1) // 2) for p in puzzle["train"]}
        train_rows, train_cols = {r for r, _ in train_shapes}, {c for _, c in train_shapes}
        classic_shift += (canvas, gap) == ((26, 26), 2) and train_rows == {2, 4} and all(
            (p["input"].shape[0] - 1) // 2 == 3 for p in puzzle["test"])
        train_colors = set()
        for pair in puzzle["train"]:
            coarse, background = _decode(pair["input"])
            train_colors.update(int(x) for x in np.unique(coarse) if int(x) != background)
        keys.add(tuple(_key(p["input"]) for p in puzzle["train"] + puzzle["test"]))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                coarse, background = _decode(pair["input"])
                assert np.array_equal(pair["output"], _bridge_render(coarse, background, canvas, gap))
                shapes.add(coarse.shape)
                bgs.add(background)
                mapping, labels = {}, []
                for value in coarse.flat:
                    value = int(value)
                    if value not in mapping:
                        mapping[value] = len(mapping)
                    labels.append(mapping[value])
                layouts.add((coarse.shape, tuple(labels)))
                holes += bool(np.any(coarse == background))
                one_axis += (1 in coarse.shape)
                large_axis += (max(coarse.shape) >= 6)
                zero_payload += background != 0 and bool(np.any(coarse == 0))
                if split == "test":
                    held += (coarse.shape[0] not in train_rows or coarse.shape[1] not in train_cols)
                    shared += (coarse.shape[0] in train_rows and coarse.shape[1] in train_cols)
                    empty_test_rows += any(np.all(row == background) for row in coarse)
                    empty_test_cols += any(np.all(col == background) for col in coarse.T)
                    test_full += bool(np.all(coarse != background))
                    test_exact += _features(coarse, background)[1]
                    test_unequal += _features(coarse, background)[2]
                    test_palette = {int(x) for x in np.unique(coarse) if int(x) != background}
                    test_palette_sizes.add(len(test_palette))
                    test_new_colors += bool(test_palette - train_colors)
                    test_familiar_colors += not bool(test_palette - train_colors)
                    runs = [_runs(row, background) for row in coarse]
                    test_returns += any(run in runs[r + 2] and run not in runs[r + 1]
                                        for r in range(len(runs) - 2) for run in runs[r])
                    deep_test += any(run in runs[r + 1] and run in runs[r + 2]
                                     for r in range(len(runs) - 2) for run in runs[r])
                else:
                    train_blank_edges += (np.all(coarse[0] == background) !=
                                          np.all(coarse[-1] == background)) or (
                                          np.all(coarse[:, 0] == background) !=
                                          np.all(coarse[:, -1] == background))
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(keys) == 42 and len(canvases) >= 6
    assert len(shapes) >= 25 and len(layouts) >= 150 and len(bgs) == 10
    assert held >= 8 and shared >= 8 and holes >= 25
    assert one_axis >= 5 and large_axis >= 15
    assert classic_shift >= 3 and train_blank_edges >= 10
    assert empty_test_rows >= 3 and empty_test_cols >= 3
    assert test_full >= 10 and test_exact >= 10 and test_unequal >= 10
    assert deep_test >= 3 and test_returns >= 3 and zero_payload >= 20
    assert test_new_colors >= 10 and test_familiar_colors >= 10
    assert len(test_palette_sizes) >= 6
    one_test_layouts = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed, 2, 1)
        coarse, background = _decode(puzzle["test"][0]["input"])
        mapping = {background: 0}
        labels = []
        for value in coarse.flat:
            value = int(value)
            if value not in mapping:
                mapping[value] = len(mapping)
            labels.append(mapping[value])
        one_test_layouts.add((coarse.shape, tuple(labels)))
    assert len(one_test_layouts) >= 35
