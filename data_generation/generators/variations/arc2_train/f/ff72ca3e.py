import numpy as np


_SOURCE_INPUTS = (
    ((9, 8), ((3, 3),), ((1, 2),)),
    ((9, 10), ((3, 4),), ((3, 1),)),
    ((9, 11), ((5, 4),), ((1, 5),)),
    ((12, 10), ((2, 3), (8, 6)),
     ((0, 6), (3, 8), (4, 1), (5, 4), (7, 0), (10, 1), (11, 9))),
    ((30, 20), ((5, 6), (15, 15), (25, 5)),
     ((2, 15), (7, 14), (11, 3), (15, 10), (16, 18),
      (19, 1), (19, 8), (19, 14), (27, 1), (27, 15))),
)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _positions(grid, color):
    return [tuple(map(int, p)) for p in np.argwhere(grid == color)]


def _distance(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _edge(p, h, w):
    return min(p[0], p[1], h - 1 - p[0], w - 1 - p[1])


def _kind(edge, obstacle):
    return "edge" if edge < obstacle else "obstacle" if obstacle < edge else "tie"


def _key(grid):
    return (grid.shape, grid.tobytes())


def _source_input(grid):
    for shape, centers, stops in _SOURCE_INPUTS:
        if grid.shape != shape:
            continue
        old = np.zeros(shape, dtype=np.int8)
        for p in centers:
            old[p] = 4
        for p in stops:
            old[p] = 5
        if np.array_equal(grid, old):
            return True
    return False


def _analyze(grid, center_color, stop_color):
    h, w = grid.shape
    centers = _positions(grid, center_color)
    stops = _positions(grid, stop_color)
    _require(centers and stops, "Every input needs both marker roles.")
    info = []
    mask = np.zeros((h, w), dtype=bool)
    for center in centers:
        edge = _edge(center, h, w)
        nearest = min(_distance(center, p) for p in stops)
        radius = min(edge, nearest - 1)
        _require(radius >= 1, "A generated center needs a visible square.")
        r, c = center
        mask[r-radius:r+radius+1, c-radius:c+radius+1] = True
        info.append((center, radius, _kind(edge, nearest - 1), edge, nearest))
    return mask, info


def _transform(grid, background, center, stop, fill):
    mask, info = _analyze(grid, center, stop)
    answer = grid.copy()
    answer[mask & (grid == background)] = fill
    return answer, info


def _counts(seed, num_train, num_test):
    for value, low, high, name in ((num_train, 2, 10, "num_train"),
                                   (num_test, 1, 4, "num_test")):
        if value is not None and (isinstance(value, (bool, np.bool_))
                                  or not isinstance(value, (int, np.integer))
                                  or not low <= int(value) <= high):
            raise ValueError(name + " is out of range.")
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = int(num_test) + 1 + seed % (10 - int(num_test))
    elif num_test is None:
        num_test = 1 + seed % min(4, int(num_train) - 1)
    if int(num_train) <= int(num_test):
        raise ValueError("num_train must exceed num_test.")
    return int(num_train), int(num_test)


def _shape(rng, regime, radius, count):
    # Aspect and scale are sampled separately from the radii and center count.
    mode = int(rng.integers(0, 4))
    if regime == "train":
        bands = (((8, 16), (8, 16)), ((12, 24), (8, 16)),
                 ((8, 16), (12, 24)), ((13, 24), (13, 24)))
    else:
        bands = (((10, 20), (10, 20)), ((20, 30), (11, 24)),
                 ((11, 24), (20, 30)), ((19, 30), (19, 30)))
    (h0, h1), (w0, w1) = bands[mode]
    h0 = max(h0, 2 * radius + 1)
    w0 = max(w0, 2 * radius + 1)
    if count >= 3:
        h0 = max(h0, 15)
        w0 = max(w0, 15)
    if h0 > h1 or w0 > w1:
        raise RuntimeError("The chosen aspect cannot hold this square.")
    return int(rng.integers(h0, h1 + 1)), int(rng.integers(w0, w1 + 1))


def _center_candidates(h, w, radius, kind):
    return [(r, c) for r in range(radius, h-radius)
            for c in range(radius, w-radius)
            if (_edge((r, c), h, w) == radius if kind != "obstacle"
                else _edge((r, c), h, w) >= radius + 1)]


def _relation(a, ra, b, rb):
    dr, dc = abs(a[0]-b[0]), abs(a[1]-b[1])
    if dr <= ra + rb and dc <= ra + rb:
        return "overlap"
    if dr <= ra + rb + 1 and dc <= ra + rb + 1:
        return "touch"
    return "separate"


def _build(rng, shape, specs, extras, relation, palette):
    h, w = shape
    candidates = [_center_candidates(h, w, r, kind) for r, kind, _ in specs]
    if any(not x for x in candidates):
        raise RuntimeError("No center placement.")
    order = sorted(range(len(specs)), key=lambda i: -specs[i][0])
    for _ in range(90):
        centers = [None] * len(specs)
        for i in order:
            pool = candidates[i]
            accepted = None
            for j in rng.permutation(len(pool))[:min(130, len(pool))]:
                p = pool[int(j)]
                if any(q is not None and _distance(p, q) <= 1 for q in centers):
                    continue
                if len(specs) >= 2 and i in (0, 1) and centers[1-i] is not None:
                    actual = _relation(p, specs[i][0], centers[1-i], specs[1-i][0])
                    if relation == "merge" and actual == "separate":
                        continue
                    if relation == "separate" and actual != "separate":
                        continue
                    if relation == "merge" and _distance(p, centers[1-i]) <= max(specs[i][0], specs[1-i][0]):
                        continue
                if relation == "separate" and any(
                    q is not None and _relation(p, specs[i][0], q, specs[k][0]) != "separate"
                    for k, q in enumerate(centers)
                ):
                    continue
                accepted = p
                break
            if accepted is None:
                break
            centers[i] = accepted
        if any(p is None for p in centers):
            continue
        stops = []

        def allowed(p, nonlimiting=False):
            if p in centers or p in stops:
                return False
            if any(_distance(p, q) <= 1 for q in centers + stops):
                return False
            return all(_distance(p, c) >= r + (2 if kind == "edge" or nonlimiting else 1)
                       for c, (r, kind, _) in zip(centers, specs))

        failed = False
        for i, (radius, kind, style) in enumerate(specs):
            if kind == "edge" or any(_distance(centers[i], p) == radius + 1 for p in stops):
                continue
            d = radius + 1
            cr, cc = centers[i]
            ring = []
            for row in range(max(0, cr-d), min(h, cr+d+1)):
                for col in range(max(0, cc-d), min(w, cc+d+1)):
                    dr, dc = abs(row-cr), abs(col-cc)
                    if max(dr, dc) != d:
                        continue
                    if style == "axial" and min(dr, dc) != 0:
                        continue
                    if style == "diagonal" and min(dr, dc) != d:
                        continue
                    if style == "intermediate" and not 0 < min(dr, dc) < d:
                        continue
                    p = (row, col)
                    if allowed(p):
                        ring.append(p)
            if not ring:
                failed = True
                break
            # A shared nearest blocker is possible whenever two rings intersect.
            shared = [p for p in ring if any(j != i and other_kind != "edge"
                      and _distance(p, centers[j]) == other_radius + 1
                      for j, (other_radius, other_kind, _) in enumerate(specs))]
            chosen = shared if shared and rng.random() < 0.3 else ring
            stops.append(chosen[int(rng.integers(len(chosen)))])
        if failed:
            continue
        if len(specs) > 0 and rng.random() < 0.35:
            eligible = [i for i, (_, kind, _) in enumerate(specs) if kind != "edge"]
            if eligible:
                i = eligible[int(rng.integers(len(eligible)))]
                r = specs[i][0]
                ring = [(row, col) for row in range(h) for col in range(w)
                        if _distance((row, col), centers[i]) == r + 1
                        and allowed((row, col))]
                if ring:
                    stops.append(ring[int(rng.integers(len(ring)))])
        for _ in range(max(extras, 1 if not stops else 0)):
            pool = [(row, col) for row in range(h) for col in range(w)
                    if allowed((row, col), nonlimiting=True)]
            if not pool:
                break
            stops.append(pool[int(rng.integers(len(pool)))])
        if not stops:
            continue
        if any(min(_distance(c, p) for p in stops) - 1 != r
               if kind != "edge" else
               min(_distance(c, p) for p in stops) - 1 <= r
               for c, (r, kind, _) in zip(centers, specs)):
            continue
        grid = np.full(shape, palette[0], dtype=np.int8)
        for p in centers:
            grid[p] = palette[1]
        for p in stops:
            grid[p] = palette[2]
        answer, info = _transform(grid, *palette)
        wanted = sorted((r, kind) for r, kind, _ in specs)
        got = sorted((item[1], item[2]) for item in info)
        if wanted == got and not _source_input(grid):
            return {"input": grid, "output": answer}
    raise RuntimeError("Could not place a valid marker layout.")


def _random_spec(rng, regime):
    radii = (1, 2, 3, 4, 5) if regime == "train" else (1, 2, 3, 4, 5, 6)
    weights = (0.23, 0.28, 0.27, 0.18, 0.04) if regime == "train" else (
        0.08, 0.20, 0.24, 0.20, 0.22, 0.06)
    r = int(rng.choice(radii, p=weights))
    kind = ("edge", "obstacle", "tie")[int(rng.integers(3))]
    styles = ("axial", "diagonal", "intermediate", "any") if r >= 2 else (
        "axial", "diagonal", "any")
    return (r, kind, styles[int(rng.integers(len(styles)))])


def _pair(rng, regime, specs, relation, palette):
    for _ in range(110):
        try:
            h, w = _shape(rng, regime, max(x[0] for x in specs), len(specs))
            # The number of unrelated stoppers is sampled independently of centers.
            extra = int(rng.integers(0, 4 if regime == "train" else 8))
            if rng.random() < 0.27:
                extra += int(rng.integers(3, 9))
            return _build(rng, (h, w), specs, extra, relation, palette)
        except RuntimeError:
            continue
    raise RuntimeError("Could not generate a pair.")


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer.")
    seed = int(seed) % (1 << 64)
    ntrain, ntest = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(int(v) for v in rng.choice(10, size=4, replace=False))
    teach_merge = bool(rng.random() < 0.42)
    puzzle = {"train": [], "test": []}
    seen = set()
    anchor_single = [(int(rng.integers(1, 5)), "obstacle", "axial")]
    anchor_multi = [
        (int(rng.integers(1, 4)), "edge", "any"),
        (int(rng.integers(2, 5)), "tie", "diagonal"),
    ]
    train_plans = [(anchor_single, "separate"),
                   (anchor_multi, "merge" if teach_merge else "separate")]
    for _ in range(ntrain - 2):
        count = int(rng.choice((1, 2, 3), p=(0.43, 0.43, 0.14)))
        specs = [_random_spec(rng, "train") for _ in range(count)]
        relation = ("merge" if count > 1 and rng.random() < 0.35 else "separate")
        train_plans.append((specs, relation))
    rng.shuffle(train_plans)
    for split in ("train", "test"):
        plans = train_plans if split == "train" else []
        if split == "test":
            for _ in range(ntest):
                count = int(rng.choice((1, 2, 3, 4), p=(0.13, 0.31, 0.42, 0.14)))
                specs = [_random_spec(rng, "test") for _ in range(count)]
                relation = ("merge" if teach_merge and count > 1
                            and rng.random() < 0.55 else "separate")
                plans.append((specs, relation))
        for specs, relation in plans:
            for _ in range(25):
                try:
                    pair = _pair(rng, split, specs, relation, palette)
                except RuntimeError:
                    # Independently sampled plans occasionally cannot fit.
                    if split == "test" or specs not in (anchor_single, anchor_multi):
                        specs = [_random_spec(rng, split) for _ in range(len(specs))]
                    continue
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Could not generate a distinct pair.")
    validate_generated_puzzle(puzzle)
    return puzzle


def _learn(train):
    colors, counts = np.unique(train[0]["input"], return_counts=True)
    _require(len(colors) == 3, "Inputs need one background and two markers.")
    background = int(colors[int(np.argmax(counts))])
    _require(int(np.count_nonzero(counts == max(counts))) == 1,
             "Background must be unambiguous.")
    markers = [int(c) for c in colors if c != background]
    input_colors = set(map(int, colors))
    fills = []
    for pair in train:
        _require(set(map(int, np.unique(pair["input"]))) == input_colors,
                 "Input palette changed within puzzle.")
        new = set(map(int, np.unique(pair["output"]))) - input_colors
        _require(len(new) == 1, "Each training output must reveal the fill color.")
        fills.append(next(iter(new)))
    _require(len(set(fills)) == 1, "Fill color changed within puzzle.")
    fill = fills[0]
    possibilities = []
    for center, stop in (markers, markers[::-1]):
        try:
            explains = all(np.array_equal(
                _transform(p["input"], background, center, stop, fill)[0],
                p["output"]) for p in train)
        except AssertionError:
            explains = False
        if explains:
            possibilities.append((center, stop))
    _require(len(possibilities) == 1, "Training does not identify marker roles.")
    center, stop = possibilities[0]
    return background, center, stop, fill


def _limiter_styles(grid, info, stop):
    stops = _positions(grid, stop)
    styles = set()
    for center, radius, kind, _, nearest in info:
        if kind == "edge":
            continue
        for p in stops:
            if _distance(center, p) != nearest:
                continue
            smaller = min(abs(center[0]-p[0]), abs(center[1]-p[1]))
            styles.add("axial" if smaller == 0 else
                       "diagonal" if smaller == nearest else "intermediate")
    return styles


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "Puzzle needs exactly train and test.")
    train, test = puzzle["train"], puzzle["test"]
    _require(isinstance(train, list) and isinstance(test, list),
             "Splits must be lists.")
    _require(2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test),
             "Illegal pair counts.")
    seen = set()
    for pair in train + test:
        _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                 "Malformed pair.")
        for grid in pair.values():
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2
                     and np.issubdtype(grid.dtype, np.integer),
                     "Grids must be integer NumPy matrices.")
            _require(all(1 <= n <= 30 for n in grid.shape)
                     and 0 <= int(grid.min()) <= int(grid.max()) <= 9,
                     "Illegal grid size or color.")
        x, y = pair["input"], pair["output"]
        _require(x.shape == y.shape and not np.array_equal(x, y),
                 "The rule must visibly preserve dimensions.")
        _require(not _source_input(x), "A source input was copied.")
        key = _key(x)
        _require(key not in seen, "Duplicate input.")
        seen.add(key)
    background, center, stop, fill = _learn(train)
    palette = {background, center, stop}
    train_counts, train_kinds, train_styles, train_relations = set(), set(), set(), set()
    test_relations = set()
    for split, pairs in (("train", train), ("test", test)):
        for pair in pairs:
            x, y = pair["input"], pair["output"]
            _require(set(map(int, np.unique(x))) == palette
                     and set(map(int, np.unique(y))) == palette | {fill},
                     "Palette changed or a role disappeared.")
            values, counts = np.unique(x, return_counts=True)
            _require(int(values[int(np.argmax(counts))]) == background,
                     "Background is not modal.")
            markers = _positions(x, center) + _positions(x, stop)
            _require(all(_distance(a, b) > 1 for i, a in enumerate(markers)
                         for b in markers[i+1:]), "Markers must be isolated.")
            expected, info = _transform(x, background, center, stop, fill)
            _require(np.array_equal(expected, y), "Incorrect centered squares.")
            radii = [v[1] for v in info]
            _require(all(1 <= r <= 7 for r in radii), "Unsupported square size.")
            relations = {_relation(a[0], a[1], b[0], b[1])
                         for i, a in enumerate(info) for b in info[i+1:]}
            if split == "train":
                train_counts.add(len(info))
                train_kinds.update(v[2] for v in info)
                train_styles.update(_limiter_styles(x, info, stop))
                train_relations.update(relations)
            else:
                test_relations.update(relations)
    _require(1 in train_counts and any(n >= 2 for n in train_counts),
             "Training must show single and multiple centers.")
    _require({"edge", "obstacle", "tie"} <= train_kinds,
             "Training must show edge, obstacle, and tied limits.")
    _require({"axial", "diagonal"} <= train_styles,
             "Training must show axial and diagonal Chebyshev blockers.")
    _require(not ({"touch", "overlap"} & test_relations)
             or bool({"touch", "overlap"} & train_relations),
             "A merged test needs a merged training demonstration.")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(p[k], q[k]) for p, q in zip(a[s], b[s])
        for k in ("input", "output")) for s in ("train", "test"))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({len(p["train"]) for p in puzzles}) >= 4
             and len({len(p["test"]) for p in puzzles}) >= 3,
             "Automatic pair counts lack diversity.")
    shapes, palettes, layouts, center_counts, radii, kinds = (
        set(), set(), set(), set(), set(), set())
    axial = diagonal = intermediate = busy = merged = wide = tall = 0
    radius_causes = set()
    stop_counts = {"train": set(), "test": set()}
    relations = {"train": set(), "test": set()}
    multiple_nearest = shared_limiter = extra_rich = 0
    mixed_scale_puzzles = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        palette = _learn(puzzle["train"])
        palettes.add(palette)
        train_features = set()
        train_sizes = [max(pair["input"].shape) for pair in puzzle["train"]]
        test_sizes = [max(pair["input"].shape) for pair in puzzle["test"]]
        mixed_scale_puzzles += min(test_sizes) <= max(train_sizes)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                x = pair["input"]
                h, w = x.shape
                shapes.add((h, w))
                wide += w >= h + 6
                tall += h >= w + 6
                _, info = _transform(x, *palette)
                centers = _positions(x, palette[1])
                stops = _positions(x, palette[2])
                stop_counts[split].add(len(stops))
                layouts.add((x.shape, tuple(centers), tuple(stops)))
                center_counts.add((split, len(centers)))
                radii.update(v[1] for v in info)
                kinds.update(v[2] for v in info)
                radius_causes.update((split, v[1], v[2]) for v in info)
                if split == "train":
                    train_features.add((len(centers), len(stops),
                                        tuple(sorted(v[1] for v in info)),
                                        tuple(sorted(v[2] for v in info))))
                styles = _limiter_styles(x, info, palette[2])
                axial += "axial" in styles
                diagonal += "diagonal" in styles
                intermediate += "intermediate" in styles
                busy += len(stops) >= 7
                pair_relations = {_relation(a[0], a[1], b[0], b[1])
                                  for i, a in enumerate(info) for b in info[i+1:]}
                relations[split].update(pair_relations)
                merged += bool({"touch", "overlap"} & pair_relations)
                multiple_nearest += any(sum(_distance(s, c) == r + 1 for s in stops) >= 2
                                        for c, r, kind, _, _ in info if kind != "edge")
                shared_limiter += any(sum(_distance(s, c) == r + 1
                                          for c, r, kind, _, _ in info
                                          if kind != "edge") >= 2 for s in stops)
                extra_rich += sum(not any(kind != "edge" and _distance(s, c) == r + 1
                                         for c, r, kind, _, _ in info)
                                  for s in stops) >= 3
        _require(len(train_features) >= 2,
                 "Training pairs do not vary their visible geometry.")
    _require(len(shapes) >= 65 and len(palettes) >= 25 and len(layouts) >= 175,
             "Shapes, palettes, or marker layouts lack diversity.")
    _require({1, 2, 3, 4} <= radii and {"edge", "obstacle", "tie"} <= kinds,
             "Square sizes or limiting causes lack diversity.")
    _require(all(v > 0 for v in (axial, diagonal, intermediate, busy, merged, wide, tall)),
             "A reviewed geometry or density regime is absent.")
    _require(all(( "test", 5, kind) in radius_causes
                 for kind in ("edge", "obstacle", "tie"))
             and any(( "test", 6, kind) in radius_causes
                     for kind in ("edge", "obstacle", "tie")),
             "Large square limits lack breadth.")
    _require({1, 2, 3} <= {n for split, n in center_counts if split == "test"}
             and min(stop_counts["train"]) <= 2 and max(stop_counts["train"]) >= 7
             and min(stop_counts["test"]) <= 2 and max(stop_counts["test"]) >= 10,
             "Object counts or marker density lack breadth.")
    _require({"separate", "touch", "overlap"} <= relations["train"]
             and {"separate", "touch", "overlap"} <= relations["test"]
             and all(v > 0 for v in (multiple_nearest, shared_limiter,
                                      extra_rich, mixed_scale_puzzles)),
             "Shared blockers, interaction, or mixed scale is absent.")
    for seed, nt, ns in ((1001, 2, 1), (1002, 10, 1), (1003, 10, 4),
                         (2001, None, 1), (2002, None, 4),
                         (2003, 2, None), (2004, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        _require((nt is None or len(p["train"]) == nt)
                 and (ns is None or len(p["test"]) == ns), "Count request ignored.")
    a = generate_puzzle(7319, 7, 3)
    _require(_equal(a, generate_puzzle(7319, 7, 3))
             and not _equal(a, generate_puzzle(7320, 7, 3)),
             "Seed behavior is not deterministic and distinct.")
    for args in ({"num_train": 1}, {"num_train": 11}, {"num_test": 0},
                 {"num_test": 5}, {"num_train": 2, "num_test": 2},
                 {"num_train": 3, "num_test": 4}):
        try:
            generate_puzzle(9001, **args)
        except ValueError:
            continue
        raise AssertionError("Illegal count accepted.")
