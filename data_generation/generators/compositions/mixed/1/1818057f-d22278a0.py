"""Complete monochrome pluses seed clipped, alternating square contours.

All source detection uses the unmodified input. Only original background cells
can change. Sampling preferences never enter solve(). No files are read/written.
"""

from __future__ import annotations

import hashlib

import numpy as np


_ARMS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= d <= 30 for d in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("grids must be integer rectangles, dimensions 1..30, colors 0..9")
    return grid


def _background(grid):
    colors, counts = np.unique(grid, return_counts=True)
    index = int(np.argmax(counts))
    if int(counts[index]) * 2 <= grid.size:
        raise ValueError("the input background must occupy more than half the cells")
    return int(colors[index])


def _detect(grid, background):
    """Every monochrome five-cell cross, even inside a larger component."""
    mask = np.zeros(grid.shape, dtype=bool)
    if min(grid.shape) >= 3:
        middle = grid[1:-1, 1:-1]
        mask[1:-1, 1:-1] = (
            (middle != background)
            & (middle == grid[:-2, 1:-1])
            & (middle == grid[2:, 1:-1])
            & (middle == grid[1:-1, :-2])
            & (middle == grid[1:-1, 2:])
        )
    sources = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(mask)]
    union = mask.copy()
    union[1:] |= mask[:-1]
    union[:-1] |= mask[1:]
    union[:, 1:] |= mask[:, :-1]
    union[:, :-1] |= mask[:, 1:]
    return sources, union


def _distance(rows, cols, r, c, metric):
    dr, dc = abs(rows - r), abs(cols - c)
    if metric == "linf":
        return np.maximum(dr, dc)
    if metric == "l2":
        return dr * dr + dc * dc
    return dr + dc


def _nearest(shape, sources, metric="l1"):
    """O(height*width) workspace, including for densely overlapping inputs."""
    rows, cols = np.indices(shape)
    best = np.full(shape, 10000, dtype=np.int32)
    winner = np.full(shape, -1, dtype=np.int16)
    unique = np.zeros(shape, dtype=bool)
    for index, (r, c, _) in enumerate(sources):
        distance = _distance(rows, cols, r, c, metric)
        closer, equal = distance < best, distance == best
        unique[equal] = False
        unique[closer] = True
        winner[closer] = index
        best[closer] = distance[closer]
    return winner, unique, best


def _paint(grid, background, sources, *, metric="l1", contour="linf",
           ties="blank", preserve=True):
    answer = grid.copy() if preserve else np.full_like(grid, background)
    if not sources:
        return answer
    rows, cols = np.indices(grid.shape)
    winner, unique, best = _nearest(grid.shape, sources, metric)
    allowed = unique.copy()
    if ties == "first":
        allowed[:] = True
    elif ties == "same_color":
        colors = np.array([color for _, _, color in sources])
        winning_color = colors[winner]
        conflict = np.zeros(grid.shape, dtype=bool)
        for r, c, color in sources:
            conflict |= ((_distance(rows, cols, r, c, metric) == best)
                         & (winning_color != color))
        allowed = ~conflict
    eligible = grid == background if preserve else np.ones(grid.shape, dtype=bool)
    for index, (r, c, color) in enumerate(sources):
        radius = _distance(rows, cols, r, c, contour)
        even = np.ones(grid.shape, dtype=bool) if contour == "filled" else radius % 2 == 0
        answer[eligible & allowed & (winner == index) & even] = color
    return answer


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve any family input without seed, examples, or construction state.

    A uniform/no-plus input is unchanged. Non-background fragments are copied,
    including fragments on bisectors. Ties between SAME-colored sources are
    still ties: sources are center locations, not color classes.
    """
    grid = _grid(input_grid)
    background = _background(grid)
    sources, _ = _detect(grid, background)
    return _paint(grid, background, sources)


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.uint8).tobytes()


def _geometry_key(grid):
    background = _background(grid)
    mapping = {background: 0}
    flat = bytes(mapping.setdefault(int(v), len(mapping)) for v in grid.flat)
    return grid.shape, flat


def _greedy_sources(grid, background):
    """A deliberately wrong destructive detector, only for witness checks."""
    remaining = grid.copy()
    result = []
    for r in range(1, grid.shape[0] - 1):
        for c in range(1, grid.shape[1] - 1):
            color = int(remaining[r, c])
            if color != background and all(remaining[r + a, c + b] == color for a, b in _ARMS):
                result.append((r, c, color))
                for a, b in _ARMS:
                    remaining[r + a, c + b] = background
    return result


def _witnesses(grid):
    """Bounded counterfactuals, not an independent or blind solver."""
    background = _background(grid)
    sources, _ = _detect(grid, background)
    correct = _paint(grid, background, sources)
    colored = grid != background
    any_color_crosses = [
        (r, c, int(grid[r, c]))
        for r in range(1, grid.shape[0] - 1)
        for c in range(1, grid.shape[1] - 1)
        if all(colored[r + a, c + b] for a, b in _ARMS)
    ]
    variants = {
        "linf_ownership": _paint(grid, background, sources, metric="linf"),
        "l2_ownership": _paint(grid, background, sources, metric="l2"),
        "l1_contours": _paint(grid, background, sources, contour="l1"),
        "filled_territory": _paint(grid, background, sources, contour="filled"),
        "first_wins_ties": _paint(grid, background, sources, ties="first"),
        "merge_same_color": _paint(grid, background, sources, ties="same_color"),
        "erase_plus_arms": _paint(grid, background, sources, preserve=False),
        "destructive_detection": _paint(grid, background, _greedy_sources(grid, background)),
        "mixed_color_crosses": _paint(grid, background, any_color_crosses),
        "every_mark_is_source": _paint(grid, background, [
            (int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(grid != background)
        ]),
    }
    return {name: not np.array_equal(correct, value) for name, value in variants.items()}


def _features(grid):
    background = _background(grid)
    sources, union = _detect(grid, background)
    _, unique, _ = _nearest(grid.shape, sources)
    foreground = grid != background
    return {
        "shape": tuple(grid.shape), "area": int(grid.size), "background": background,
        "sources": len(sources), "colors": len({v for _, _, v in sources}),
        "overlap": 5 * len(sources) > int(union.sum()),
        "decoy": bool(np.any(foreground & ~union)),
        "edge": any(r in (1, grid.shape[0] - 2) or c in (1, grid.shape[1] - 2)
                    for r, c, _ in sources),
        "tie": bool(sources) and bool(np.any(~unique & ~foreground)),
    }


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _shape(rng, band):
    low, high = band
    h, w = map(int, rng.integers(low, high + 1, size=2))
    if rng.random() < .20:
        w = h
    return h, w


def _stamp(grid, background, r, c, color, offsets):
    points = [(r + a, c + b) for a, b in offsets]
    if not all(0 <= a < grid.shape[0] and 0 <= b < grid.shape[1] for a, b in points):
        return False
    if any(grid[a, b] not in (background, color) for a, b in points):
        return False
    extra = sum(grid[a, b] == background for a, b in set(points))
    if np.count_nonzero(grid != background) + extra > int(grid.size * .44):
        return False
    for a, b in points:
        grid[a, b] = color
    return True


def _add_sources(rng, grid, background, palette, count, centers, layout):
    h, w = grid.shape
    coordinates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    order = [coordinates[int(i)] for i in rng.permutation(len(coordinates))]
    if layout == "line":
        if w >= h:
            row = int(rng.integers(1, h - 1))
            preferred = [(row, c) for c in range(1, w - 1, int(rng.integers(3, 6)))]
        else:
            col = int(rng.integers(1, w - 1))
            preferred = [(r, col) for r in range(1, h - 1, int(rng.integers(3, 6)))]
        order = preferred + order
    elif layout == "array":
        row_gap, col_gap = map(int, rng.integers(3, 7, size=2))
        preferred = [(r, c) for r in range(1, h - 1, row_gap)
                     for c in range(1, w - 1, col_gap)]
        rng.shuffle(preferred)
        order = preferred + order
    for r, c in order:
        if len(centers) >= count:
            break
        # Outside the deliberately overlapping group, sources remain readable.
        if any(abs(r - a) + abs(c - b) < 3 for a, b in centers):
            continue
        color = int(palette[len(centers) % len(palette)])
        if _stamp(grid, background, r, c, color, _ARMS):
            centers.append((r, c))


def _clutter(rng, grid, background, palette, centers, mandatory):
    """T/X/ring/lines and extensions; all actual pluses are detected afterward."""
    kinds = ("T", "X", "ring", "line", "bend", "clipped", "extension", "diagonal")
    if mandatory:
        # A visibly broken monochrome cross distinguishes the color convention.
        candidates = [(r, c) for r in range(1, grid.shape[0] - 1)
                      for c in range(1, grid.shape[1] - 1)
                      if np.all(grid[r - 1:r + 2, c - 1:c + 2] == background)]
        if candidates:
            r, c = candidates[int(rng.integers(len(candidates)))]
            if _stamp(grid, background, r, c, int(palette[-1]), _ARMS):
                grid[r, c] = int(palette[0])
    count = int(rng.integers(2, 7)) if mandatory else int(rng.integers(0, 5))
    for index in range(count):
        kind = "T" if mandatory and index == 0 else str(rng.choice(kinds))
        color = int(rng.choice(palette))
        if kind in ("extension", "diagonal") and centers:
            r, c = centers[int(rng.integers(len(centers)))]
            color = int(grid[r, c])
            if kind == "diagonal":
                points = [(int(rng.choice((-1, 1))), int(rng.choice((-1, 1))))]
            else:
                a, b = _ARMS[1 + int(rng.integers(4))]
                points = [(a * k, b * k) for k in range(2, int(rng.integers(3, 7)))]
        else:
            r, c = int(rng.integers(1, grid.shape[0] - 1)), int(rng.integers(1, grid.shape[1] - 1))
            if kind == "T":
                points = [(0, -1), (0, 0), (0, 1), (1, 0)]
            elif kind == "X":
                points = [(0, 0), (-1, -1), (1, 1), (-1, 1), (1, -1)]
            elif kind == "ring":
                points = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
            elif kind == "clipped":
                r = 0
                points = [(0, -1), (0, 0), (0, 1), (1, 0)]
            elif kind == "bend":
                points = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
            else:
                points = [(0, k) for k in range(int(rng.integers(2, 7)))]
            turns = int(rng.integers(4)) if kind != "clipped" else 0
            for _ in range(turns):
                points = [(-b, a) for a, b in points]
        _stamp(grid, background, r, c, color, points)


def _scene(rng, shape, background, palette, kind, count):
    grid = np.full(shape, background, dtype=np.int8)
    h, w = shape
    centers = []
    if kind == "clean":
        horizontal = bool(rng.integers(2))
        length = w if horizontal else h
        gap = int(rng.choice(list(range(4, min(12, length - 3) + 1, 2))))
        start = int(rng.integers(1, length - 1 - gap))
        cross = int(rng.integers(1, (h if horizontal else w) - 1))
        centers = [(cross, start), (cross, start + gap)] if horizontal else [(start, cross), (start + gap, cross)]
        for index, (r, c) in enumerate(centers):
            _stamp(grid, background, r, c, int(palette[index]), _ARMS)
    else:
        if kind == "overlap":
            r, c = int(rng.integers(2, h - 3)), int(rng.integers(2, w - 3))
            a, b = ((0, 2), (2, 0), (1, 1), (1, -1))[int(rng.integers(4))]
            centers = [(r, c), (r + a, c + b)]
            for row, col in centers:
                _stamp(grid, background, row, col, int(palette[0]), _ARMS)
            # Some scenes contain adjacent centers as well as shared-arm ones.
            if count >= 5 and rng.random() < .5:
                if _stamp(grid, background, r, c + 1, int(palette[0]), _ARMS):
                    centers.append((r, c + 1))
        _add_sources(rng, grid, background, palette, count, centers,
                     str(rng.choice(("scatter", "line", "array"))))
        _clutter(rng, grid, background, palette, centers, kind == "overlap")
    # All four rotations and reflections retain the fixed rule.
    grid = np.rot90(grid, int(rng.integers(4)))
    if rng.random() < .5:
        grid = np.fliplr(grid)
    return grid.copy()


def _make_pair(rng, band, kind, count, background, palette, used, *, crowd_min=0):
    for _ in range(100):
        inp = _scene(rng, _shape(rng, band), background, palette, kind, count)
        info = _features(inp)
        if not 1 <= info["sources"] <= 14 or info["sources"] < crowd_min:
            continue
        if kind == "overlap":
            if not (info["overlap"] and info["decoy"] and 3 <= info["sources"] <= 8):
                continue
            evidence = _witnesses(inp)
            if not all(evidence[name] for name in ("l2_ownership", "merge_same_color", "destructive_detection", "mixed_color_crosses")):
                continue
        elif not crowd_min and info["sources"] > 8:
            continue
        key = _geometry_key(inp)
        if key in used:
            continue
        out = solve(inp)
        if np.array_equal(inp, out):
            continue
        used.add(key)
        return {"input": inp, "output": out}
    raise RuntimeError("bounded scene search exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one rule with clean and competing/overlap training witnesses."""
    n_train = _count(num_train, "num_train", 2, 10)
    n_test = _count(num_test, "num_test", 1, 4)
    if n_train <= n_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift = int(rng.integers(3))  # grow, shrink, or increase source competition
    train_band, test_band = (((9, 18), (20, 30)), ((19, 28), (9, 18)),
                             ((11, 23), (15, 28)))[shift]
    background_mode = int(rng.integers(3))
    fixed_background = 0 if background_mode == 0 else int(rng.integers(10))
    stable_palette = bool(rng.integers(2))
    core = list(map(int, rng.permutation(10)))

    def roles():
        background = int(rng.integers(10)) if background_mode == 2 else fixed_background
        choices = [color for color in core if color != background]
        if not stable_palette:
            rng.shuffle(choices)
        return background, choices[:int(rng.integers(3, 6))]

    puzzle = {"train": [], "test": []}
    used = set()
    for kind, count in [("clean", 2), ("overlap", int(rng.integers(3, 7)))]:
        background, palette = roles()
        puzzle["train"].append(_make_pair(rng, train_band, kind, count, background, palette, used))
    for _ in range(n_train - 2):
        kind = str(rng.choice(("random", "overlap"), p=(.7, .3)))
        count = int(rng.integers(3, 7)) if kind == "overlap" else int(rng.integers(1, 8))
        background, palette = roles()
        puzzle["train"].append(_make_pair(rng, train_band, kind, count, background, palette, used))
    rng.shuffle(puzzle["train"])
    largest_count = max(_features(pair["input"])["sources"] for pair in puzzle["train"])
    for _ in range(n_test):
        crowd_min = largest_count + 1 if shift == 2 else 0
        kind = "random" if shift == 2 else str(rng.choice(("random", "overlap"), p=(.7, .3)))
        count = (int(rng.integers(crowd_min, crowd_min + 3)) if crowd_min else
                 int(rng.integers(3, 7)) if kind == "overlap" else int(rng.integers(1, 8)))
        background, palette = roles()
        puzzle["test"].append(_make_pair(rng, test_band, kind, count, background, palette, used, crowd_min=crowd_min))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, input-derived geometry, preserved ink, and teaching evidence."""
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[split], list) for split in ("train", "test"))
    a, b = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    seen, structures = set(), set()
    records = {"train": [], "test": []}
    evidence = {}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            background = _background(inp)
            sources, union = _detect(inp, background)
            assert sources and not np.array_equal(inp, out)
            assert _key(inp) not in seen and _geometry_key(inp) not in structures
            seen.add(_key(inp))
            structures.add(_geometry_key(inp))
            assert np.array_equal(out[inp != background], inp[inp != background])
            assert np.array_equal(out[union], inp[union])
            assert set(map(int, np.unique(inp))) == set(map(int, np.unique(out)))
            winner, unique, _ = _nearest(inp.shape, sources)
            rows, cols = np.indices(inp.shape)
            background_cells = inp == background
            assert np.all(out[background_cells & ~unique] == background), "ties must remain blank"
            for index, (r, c, color) in enumerate(sources):
                owned = background_cells & unique & (winner == index)
                even = np.maximum(abs(rows - r), abs(cols - c)) % 2 == 0
                assert np.all(out[owned & even] == color)
                assert np.all(out[owned & ~even] == background)
            assert np.array_equal(solve(inp.copy()), out)
            records[split].append(_features(inp))
            if split == "train":
                for name, witnessed in _witnesses(inp).items():
                    evidence[name] = evidence.get(name, False) or witnessed
    assert all(evidence.values()), f"missing training distinctions: {evidence}"
    train, test = records["train"], records["test"]
    assert any(r["sources"] == 2 and not r["decoy"] and r["colors"] == 2 for r in train)
    assert any(r["overlap"] and r["decoy"] and r["sources"] >= 3 for r in train)
    assert (min(r["area"] for r in test) > max(r["area"] for r in train)
            or max(r["area"] for r in test) < min(r["area"] for r in train)
            or min(r["sources"] for r in test) > max(r["sources"] for r in train)), "no intended shift"


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, small counterexamples and API boundaries; no artifact storage."""
    fingerprints, shapes, geometries, backgrounds, counts = set(), set(), set(), set(), set()
    shifts, flags = set(), {name: 0 for name in ("overlap", "decoy", "edge", "tie")}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        groups = {s: [_features(p["input"]) for p in puzzle[s]] for s in ("train", "test")}
        train, test = groups["train"], groups["test"]
        if min(r["area"] for r in test) > max(r["area"] for r in train):
            shifts.add("growing")
        if max(r["area"] for r in test) < min(r["area"] for r in train):
            shifts.add("shrinking")
        if min(r["sources"] for r in test) > max(r["sources"] for r in train):
            shifts.add("crowding")
        for split in ("train", "test"):
            for pair, info in zip(puzzle[split], groups[split]):
                shapes.add(info["shape"])
                backgrounds.add(info["background"])
                counts.add(info["sources"])
                geometries.add(_geometry_key(pair["input"]))
                for name in flags:
                    flags[name] += bool(info[name])
    assert len(fingerprints) == 42 and len(geometries) >= 150
    assert len(shapes) >= 60 and any(30 in shape for shape in shapes)
    assert len(backgrounds) >= 8 and len(counts) >= 8 and 1 in counts
    assert shifts == {"growing", "shrinking", "crowding"}
    assert min(flags.values()) >= 10
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0))
    # All 512 binary 3x3 neighborhoods, translated to a nonzero background.
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)]).reshape(3, 3)
        inp = np.where(mask, 0, 7).astype(np.int8)
        sources, union = _detect(inp, 7)
        expected = all(mask[p] for p in ((1, 1), (0, 1), (2, 1), (1, 0), (1, 2)))
        assert bool(sources) == expected
        assert int(union.sum()) == (5 if expected else 0)
    # One emitter: original arms survive odd radius 1; radius 2 fills a square.
    inp = np.zeros((9, 9), dtype=np.int8)
    _stamp(inp, 0, 4, 4, 3, _ARMS)
    expected = inp.copy()
    rows, cols = np.indices(inp.shape)
    expected[np.maximum(abs(rows - 4), abs(cols - 4)) % 2 == 0] = 3
    assert np.array_equal(solve(inp), expected)
    # Same-colored sources still compete. The bisector at (2,4) is blank.
    inp = np.zeros((9, 9), dtype=np.int8)
    for c in (3, 5):
        _stamp(inp, 0, 4, c, 6, _ARMS)
    out = solve(inp)
    assert out[2, 4] == 0 and inp[4, 4] == out[4, 4] == 6
    assert len(_detect(inp, 0)[0]) == 2
    # No wrapping, no diagonal substitute, no source created by a mixed cross.
    inp = np.zeros((7, 9), dtype=np.int8)
    inp[0, 2:5] = 2
    inp[1, 3] = inp[6, 3] = 2
    _stamp(inp, 0, 3, 6, 4, _ARMS)
    inp[2, 6] = 5
    assert np.array_equal(solve(inp), inp)
    for shape in ((1, 1), (1, 30), (30, 1), (3, 3)):
        uniform = np.full(shape, 8, dtype=int)
        assert np.array_equal(solve(uniform), uniform)
    # Solving commutes with reflections/rotations and arbitrary palette swaps.
    puzzle = generate_puzzle(901, 2, 1)
    palette = np.array([7, 2, 9, 0, 1, 8, 3, 5, 6, 4])
    for pair in puzzle["train"] + puzzle["test"]:
        inp, out = pair["input"], pair["output"]
        assert np.array_equal(solve(np.rot90(inp)), np.rot90(out))
        assert np.array_equal(solve(np.fliplr(inp)), np.fliplr(out))
        assert np.array_equal(solve(palette[inp]), palette[out])
        assert np.array_equal(solve(inp.tolist()), out)
    # Reject changed source ink, not just malformed data.
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzle[s]] for s in puzzle}
    first = bad["train"][0]
    bg = _background(first["input"])
    r, c = np.argwhere(first["input"] != bg)[0]
    first["output"][r, c] = bg
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("validator accepted altered source ink")
    for a, b in ((2, 1), (5, 4), (10, 4)):
        puzzle = generate_puzzle(810 + a, a, b)
        assert len(puzzle["train"]) == a and len(puzzle["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == "__main__":
    self_test()
