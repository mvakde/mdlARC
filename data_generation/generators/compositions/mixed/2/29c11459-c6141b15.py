"""Exchange glyph/segment positions, rendering a midpoint-split connection.

All semantic parameters are visible in the input. No files, cached examples,
parent modules, or generation state are consulted by solve().
"""

import numpy as np


HOST_SEEDS = (
    0, 647807201, 36625480, 82563545, 673678200, 145318638, 683215257,
    713830843, 900201438, 507177183, 151646926, 961064810, 840808932,
    996674640, 863830876, 60658829, 802562490, 394139163, 274404030,
    594458240, 92129732, 348187469, 684454764, 531154976, 207422829,
    991737982, 37985142, 241108183, 715506257, 778178151, 806049896,
    657299219, 424093071, 377523748, 123498788, 366403610, 685724285,
    354100046, 133703128, 112117213, 348282111, 100180341,
)


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    _require(_integer(num_train) and 2 <= num_train <= 10,
             "num_train must be an integer in 2..10")
    _require(_integer(num_test) and 1 <= num_test <= 4,
             "num_test must be an integer in 1..4")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    grid = np.asarray(value)
    _require(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape),
             "Grid dimensions must be 1..30")
    _require(np.issubdtype(grid.dtype, np.integer), "Grid cells must be integers")
    _require(np.all((0 <= grid) & (grid <= 9)), "Grid colors must be 0..9")
    return grid.astype(np.int8, copy=False)


def _components(cells):
    unseen = set(cells)
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        stack, found = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    p = (r + dr, c + dc)
                    if p in unseen:
                        unseen.remove(p)
                        found.add(p)
                        stack.append(p)
        result.append(frozenset(found))
    return sorted(result, key=min)


def _stamps(centers, motif):
    return frozenset((r + dr, c + dc) for r, c in centers for dr, dc in motif)


def _apart(first, second):
    # At least one background cell separates different foreground structures.
    return not any((r + dr, c + dc) in second for r, c in first
                   for dr in (-1, 0, 1) for dc in (-1, 0, 1))


def _analyze(value):
    grid = _grid(value)
    colors, counts = np.unique(grid, return_counts=True)
    _require(len(colors) == 5, "Expected background, glyph, two ends, and line core")
    largest = int(np.argmax(counts))
    _require(np.count_nonzero(counts == counts[largest]) == 1,
             "Background must be uniquely most frequent")
    bg = int(colors[largest])
    points = {int(color): frozenset(map(tuple, np.argwhere(grid == color).tolist()))
              for color in colors if color != bg}
    candidates = []
    for separator, core in points.items():
        # The visibly encoded separator is the solid interior of the source bar.
        if len(core) < 3 or len(core) % 2 == 0:
            continue
        rows = {p[0] for p in core}
        if len(rows) != 1:
            continue
        row = next(iter(rows))
        lo, hi = min(p[1] for p in core), max(p[1] for p in core)
        if hi - lo + 1 != len(core) or lo == 0 or hi == grid.shape[1] - 1:
            continue
        left, right = int(grid[row, lo - 1]), int(grid[row, hi + 1])
        if len({bg, separator, left, right}) != 4:
            continue
        if len(points.get(left, ())) != 1 or len(points.get(right, ())) != 1:
            continue
        remaining = set(points) - {separator, left, right}
        if len(remaining) != 1:
            continue
        glyph = remaining.pop()
        components = _components(points[glyph])
        if len(components) != 2:
            continue
        centers, motifs = [], []
        for comp in components:
            rs, cs = zip(*comp)
            if (min(rs) + max(rs)) % 2 or (min(cs) + max(cs)) % 2:
                break
            center = ((min(rs) + max(rs)) // 2, (min(cs) + max(cs)) // 2)
            centers.append(center)
            motifs.append(frozenset((r - center[0], c - center[1]) for r, c in comp))
        if len(motifs) != 2 or motifs[0] != motifs[1] or (0, 0) not in motifs[0]:
            continue
        centers = tuple(sorted(centers))
        if centers[0][0] != centers[1][0]:
            continue
        distance = centers[1][1] - centers[0][1]
        if distance < 4 or distance % 2:
            continue
        motif = motifs[0]
        endpoints = ((row, lo - 1), (row, hi + 1))
        segment = frozenset((row, c) for c in range(lo - 1, hi + 2))
        figure = frozenset((centers[0][0], c)
                           for c in range(centers[0][1], centers[1][1] + 1))
        copies = _stamps(endpoints, motif)
        if not all(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1] for r, c in copies):
            continue
        if len(_components(copies)) != 2:
            continue
        if not _apart(segment, points[glyph]) or not _apart(figure, copies):
            continue
        candidates.append({"background": bg, "glyph": glyph, "left": left,
                           "right": right, "separator": separator,
                           "endpoints": endpoints, "centers": centers,
                           "motif": motif, "segment": segment,
                           "figure": figure, "copies": copies,
                           "old_glyphs": points[glyph],
                           "source_half": (hi - lo + 2) // 2,
                           "target_half": distance // 2})
    _require(len(candidates) == 1, "No unique valid glyph/segment interpretation")
    return grid, candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, anchor, color and output location from input only."""
    grid, info = _analyze(input_grid)
    out = np.full(grid.shape, info["background"], dtype=np.int8)
    (row, left), (_, right) = info["centers"]
    middle = (left + right) // 2
    out[row, left:middle] = info["left"]
    out[row, middle] = info["separator"]
    out[row, middle + 1:right + 1] = info["right"]
    for r, c in info["copies"]:
        out[r, c] = info["glyph"]
    return out


def _featured_motif(rng):
    # Asymmetric, non-centroid-anchored T teaches the exact bounding-box anchor.
    motif = {(-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)}
    for _ in range(int(rng.integers(4))):
        motif = {(c, -r) for r, c in motif}
    if rng.integers(2):
        motif = {(r, -c) for r, c in motif}
    return frozenset(motif)


def _random_motif(rng):
    rr, rc = ((0, 0), (1, 1), (1, 2), (2, 1), (2, 2))[int(rng.integers(5))]
    if rr == 0:
        return frozenset({(0, 0)})
    allowed = {(r, c) for r in range(-rr, rr + 1) for c in range(-rc, rc + 1)}
    cells = {(0, 0)}
    target = int(rng.integers(4, len(allowed) + 1))
    while (len(cells) < target or min(r for r, c in cells) != -rr
           or max(r for r, c in cells) != rr or min(c for r, c in cells) != -rc
           or max(c for r, c in cells) != rc):
        frontier = sorted({(r + dr, c + dc) for r, c in cells
                           for dr in (-1, 0, 1) for dc in (-1, 0, 1)} & allowed - cells)
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return frozenset(cells)


def _dimension(rng, minimum):
    if rng.random() < 0.16:
        return 30
    return int(rng.integers(minimum, 31))


def _layout(rng, motif, source_half, target_half):
    rr, rc = max(abs(r) for r, c in motif), max(abs(c) for r, c in motif)
    source_width, target_width = 2 * source_half + 1, 2 * target_half + 1
    combined = source_width + target_width + 4 * rc + 1
    if combined <= 30 and rng.random() < 0.22:
        # The exchange is also defined when all four anchors share one row.
        width = _dimension(rng, combined)
        height = _dimension(rng, max(3, 2 * rr + 1))
        gap = int(rng.integers(1, width - combined + 2))
        start = int(rng.integers(width - (combined - 1 + gap) + 1))
        source_first = bool(rng.integers(2))
        if source_first:
            source_left = start + rc
            target_left = start + source_width + 2 * rc + gap + rc
        else:
            target_left = start + rc
            source_left = start + target_width + 2 * rc + gap + rc
        source_row = target_row = int(rng.integers(rr, height - rr))
    else:
        width = _dimension(rng, max(source_width, target_width) + 2 * rc)
        height = _dimension(rng, 3 * rr + 3)
        separation = int(rng.integers(rr + 2, height - 2 * rr))
        top = int(rng.integers(rr, height - rr - separation))
        source_row, target_row = top, top + separation
        if rng.integers(2):
            source_row, target_row = target_row, source_row
        source_left = int(rng.integers(rc, width - rc - source_width + 1))
        target_left = int(rng.integers(rc, width - rc - target_width + 1))
    endpoints = ((source_row, source_left), (source_row, source_left + source_width - 1))
    centers = ((target_row, target_left), (target_row, target_left + target_width - 1))
    return (height, width), endpoints, centers


def _make_pair(rng, motif, target_half, palette, relation=None):
    bg, glyph, left_color, right_color, separator = palette
    rc = max(abs(c) for r, c in motif)
    minimum, maximum = max(2, rc + 1), (29 - 2 * rc) // 2
    choices = [n for n in range(minimum, maximum + 1)
               if (relation != "contract" or n > target_half)
               and (relation != "expand" or n < target_half)]
    source_half = int(rng.choice(choices))
    shape, endpoints, centers = _layout(rng, motif, source_half, target_half)
    inp = np.full(shape, bg, dtype=np.int8)
    out = np.full(shape, bg, dtype=np.int8)
    (sr, sl), (_, se) = endpoints
    inp[sr, sl + 1:se] = separator
    inp[sr, sl], inp[sr, se] = left_color, right_color
    for r, c in _stamps(centers, motif):
        inp[r, c] = glyph
    (tr, tl), (_, te) = centers
    # Generation renders directly from its sampled geometry, not through solve.
    for c in range(tl, te + 1):
        out[tr, c] = left_color if c < (tl + te) // 2 else (
            separator if c == (tl + te) // 2 else right_color)
    for r, c in _stamps(endpoints, motif):
        out[r, c] = glyph
    return {"input": inp, "output": out}


def _key(grid):
    return grid.shape, np.ascontiguousarray(grid, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one coherent family instance, with longer test-center spacing."""
    num_train, num_test = _counts(num_train, num_test)
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    featured = _featured_motif(rng)
    base = tuple(int(c) for c in rng.choice(10, 5, replace=False))
    bg, glyph, left, right, sep = base
    # Change the visible separator and background as well as reversing end colors.
    second_palette = (glyph, sep, right, left, bg)
    base_half = int(rng.integers(3, 6))
    training_halves = [base_half, base_half + 1]
    training_halves += [base_half + int(rng.integers(3)) for _ in range(num_train - 2)]
    test_halves = [max(training_halves) + int(rng.integers(1, 4)) for _ in range(num_test)]
    puzzle, seen = {"train": [], "test": []}, set()
    for index, half in enumerate(training_halves + test_halves):
        split = "train" if index < num_train else "test"
        motif = featured if index == 0 or (split == "test" and rng.random() < 0.40) else _random_motif(rng)
        if index == 1 and motif == featured:
            motif = frozenset({(0, 0)})
        if index == 0:
            palette = base
        elif index == 1:
            palette = second_palette
        elif rng.random() < 0.30:
            palette = base if rng.integers(2) else second_palette
        else:
            palette = tuple(int(c) for c in rng.choice(10, 5, replace=False))
        relation = "contract" if index == 0 else "expand" if index == 1 else None
        for _ in range(32):
            pair = _make_pair(rng, motif, half, palette, relation)
            key = _key(pair["input"])
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("Bounded distinct-input placement exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Inspect returned grids and semantic invariants, without sampler metadata."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"},
             "Puzzle must contain exactly train and test")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")),
             "Splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"},
                     "Pair must contain exactly input and output")
            inp, info = _analyze(pair["input"])
            out = _grid(pair["output"])
            _require(inp.shape == out.shape, "Canvas dimensions changed")
            _require(_key(inp) not in seen, "Duplicate input")
            seen.add(_key(inp))
            infos[split].append(info)
            (row, left), (_, right) = info["centers"]
            middle = (left + right) // 2
            half = info["target_half"]
            # Counts catch partial stamping, wrong boundaries and wrong partitions.
            _require(np.count_nonzero(out == info["glyph"]) == 2 * len(info["motif"]),
                     "Glyph pixels not preserved exactly twice")
            _require(np.count_nonzero(out == info["separator"]) == 1,
                     "Output must have exactly one separator")
            for role in ("left", "right"):
                _require(np.count_nonzero(out == info[role]) == half,
                         "Endpoint-colored halves must have equal inclusive lengths")
            # A per-cell check also verifies erasure and all background outside supports.
            for r in range(inp.shape[0]):
                for c in range(inp.shape[1]):
                    expected = info["background"]
                    if (r, c) in info["copies"]:
                        expected = info["glyph"]
                    elif r == row and left <= c <= right:
                        expected = info["separator"] if c == middle else info[
                            "left" if c < middle else "right"]
                    _require(int(out[r, c]) == expected,
                             "Wrong row rendering, translation, erasure or outer background")
            _require(np.array_equal(solve(inp), out), "Input-only solve disagrees")
            # Explicit parent ablations: monochrome B connection; A in the old location.
            no_a = out.copy()
            no_a[row, left:right + 1] = info["separator"]
            no_b = inp.copy()
            (sr, sl), (_, se) = info["endpoints"]
            for c in range(sl, se + 1):
                no_b[sr, c] = info["separator"] if c == (sl + se) // 2 else info[
                    "left" if c < (sl + se) // 2 else "right"]
            _require(not np.array_equal(no_a, out), "Omitting A does not change the answer")
            _require(not np.array_equal(no_b, out), "Omitting B does not change the answer")
    training = infos["train"]
    _require(any(i["source_half"] > i["target_half"] for i in training),
             "Training must demonstrate contraction")
    _require(any(i["source_half"] < i["target_half"] for i in training),
             "Training must demonstrate expansion")
    _require(len({i["target_half"] for i in training}) >= 2,
             "Training must show different destination lengths")
    _require(len({i["separator"] for i in training}) >= 2,
             "Training must show that the core supplies a local separator color")
    _require(any(i["motif"] != frozenset((-r, -c) for r, c in i["motif"])
                 for i in training), "Training must demonstrate asymmetric copying")
    _require(len({i["motif"] for i in training}) >= 2, "Training must show multiple motifs")
    _require(all(i["target_half"] > max(j["target_half"] for j in training)
                 for i in infos["test"]), "Test must extend the demonstrated center spacing")


def self_test():
    """Bounded, streaming 42-seed checks; returns descriptive metrics, writes nothing."""
    shapes, motifs, structures, signatures = set(), set(), set(), set()
    roles = {name: set() for name in ("background", "glyph", "left", "right", "separator")}
    source_lengths, target_lengths, motif_sizes = set(), set(), set()
    observations = {name: 0 for name in ("above", "below", "same_row", "edge_touch",
                    "asymmetric", "same_length", "new_test_motif", "black_foreground")}
    total = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        repeated = generate_puzzle(seed)
        _require(all(np.array_equal(a[role], b[role])
                     for split in puzzle for a, b in zip(puzzle[split], repeated[split])
                     for role in ("input", "output")), "Seed is not reproducible")
        signatures.add(tuple(_key(p["input"]) for s in puzzle for p in puzzle[s]))
        train_motifs = {_analyze(p["input"])[1]["motif"] for p in puzzle["train"]}
        for split in puzzle:
            for pair in puzzle[split]:
                inp, info = _analyze(pair["input"])
                out = pair["output"]
                total += 1
                shapes.add(inp.shape)
                motifs.add(info["motif"])
                motif_sizes.add(len(info["motif"]))
                source_lengths.add(2 * info["source_half"] + 1)
                target_lengths.add(2 * info["target_half"] + 1)
                structures.add((inp.shape, info["endpoints"], info["centers"], info["motif"]))
                for name in roles:
                    roles[name].add(info[name])
                sr, tr = info["endpoints"][0][0], info["centers"][0][0]
                observations["above" if sr < tr else "below" if sr > tr else "same_row"] += 1
                occupied = info["segment"] | info["old_glyphs"] | info["figure"] | info["copies"]
                observations["edge_touch"] += any(r in (0, inp.shape[0] - 1) or
                    c in (0, inp.shape[1] - 1) for r, c in occupied)
                observations["asymmetric"] += info["motif"] != frozenset(
                    (-r, -c) for r, c in info["motif"])
                observations["same_length"] += info["source_half"] == info["target_half"]
                observations["new_test_motif"] += split == "test" and info["motif"] not in train_motifs
                observations["black_foreground"] += 0 in (info["glyph"], info["left"],
                                                           info["right"], info["separator"])
                # Role assignment and exact copying commute with palette permutations
                # and whole-canvas reflections; no arbitrary glyph turn is introduced.
                permutation = np.random.default_rng(seed + total).permutation(10)
                _require(np.array_equal(solve(permutation[inp]), permutation[out]),
                         "Color permutation changed the geometric rule")
                for flip in (np.fliplr, np.flipud):
                    _require(np.array_equal(solve(flip(inp)), flip(out)),
                             "Reflection broke anchor geometry or motif preservation")
    _require(len(signatures) == 42, "Seeds did not yield 42 distinct puzzles")
    _require(len(shapes) >= 50 and len(structures) >= 180 and len(motifs) >= 40,
             "Insufficient structural variation")
    _require(all(len(values) == 10 for values in roles.values()),
             "Every role should cover all ten colors across 42 seeds")
    _require(len(source_lengths) >= 8 and len(target_lengths) >= 7 and len(motif_sizes) >= 10,
             "Insufficient scale or pixel-count variation")
    _require(all(n > 0 for n in observations.values()), "A promised family regime is absent")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(20261023, a, b)
            _require(len(p["train"]) == a and len(p["test"]) == b, "Explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    # A damaged saved answer must be rejected by the semantic validator.
    damaged = generate_puzzle(0, 2, 1)
    _, info = _analyze(damaged["train"][0]["input"])
    (r, a), (_, b) = info["centers"]
    damaged["train"][0]["output"][r, (a + b) // 2] = info["left"]
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("Validator accepted an incorrect separator")
    return {"seeds": 42, "pairs": total, "distinct_puzzles": len(signatures),
            "shapes": len(shapes), "motifs": len(motifs), "structures": len(structures),
            "source_lengths": sorted(source_lengths), "target_lengths": sorted(target_lengths),
            "motif_pixel_counts": sorted(motif_sizes),
            "role_color_counts": {k: len(v) for k, v in roles.items()},
            "observations": observations, "legal_count_combinations": 30}
