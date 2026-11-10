import numpy as np


# Exact source inputs, used only to reject complete source copies. In
# particular, a new 15 by 15 puzzle is permitted.
_SOURCE_INPUTS = {
    "".join(rows) for rows in (
        (
            "000000000000000", "000000000000000", "000000000000000",
            "000220000220000", "000200000020000", "000205005020000",
            "000200505020000", "000200505020000", "000205005020000",
            "000200000020000", "000220000220000", "000000000000000",
            "000000000000000", "000000000000000", "000000000000000",
        ),
        (
            "000000000000000", "000000000000000", "000000000000000",
            "000000000000000", "000022222220000", "000020000020000",
            "000000555000000", "000000505000000", "000000000000000",
            "000000050000000", "000020000020000", "000022222220000",
            "000000000000000", "000000000000000", "000000000000000",
        ),
        (
            "000000000000000", "000000000000000", "000000000000000",
            "000220000220000", "000205005020000", "000205505020000",
            "000200505020000", "000220000220000", "000000000000000",
            "000000000000000", "000000000000000", "000000000000000",
            "000000000000000", "000000000000000", "000000000000000",
        ),
        (
            "000000000000000", "000000000000000", "000000000000000",
            "000000000000000", "000022222220000", "000020000020000",
            "000000555500000", "000005550000000", "000000000000000",
            "000000555000000", "000020000020000", "000022222220000",
            "000000000000000", "000000000000000", "000000000000000",
        ),
    )
}


def _integer(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


def _counts(seed, rng, num_train, num_test):
    train = _integer("num_train", num_train, 2, 10)
    test = _integer("num_test", num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4) % (11 - low)
    elif train is None:
        train = int(rng.integers(max(2, test + 1), 11))
    elif test is None:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    if train <= test:
        raise ValueError("num_train must exceed num_test")
    return train, test


def _palette(rng):
    return tuple(int(v) for v in rng.choice(10, size=3, replace=False))


def _orientations(rng, train_count, test_count):
    mode = int(rng.integers(0, 5))
    if mode == 0:
        return ["vertical"] * train_count, ["vertical"] * test_count
    if mode == 1:
        return ["horizontal"] * train_count, ["horizontal"] * test_count
    if mode == 2:
        return ["vertical"] * train_count, ["horizontal"] * test_count
    if mode == 3:
        return ["horizontal"] * train_count, ["vertical"] * test_count
    train = ["vertical", "horizontal"]
    train += [("vertical" if rng.random() < .5 else "horizontal")
              for _ in range(train_count - 2)]
    rng.shuffle(train)
    test = [("vertical" if rng.random() < .5 else "horizontal")
            for _ in range(test_count)]
    return train, test


def _motif(rng, span, depth, style):
    if style == "bar" or depth == 1:
        return {(t, 0) for t in range(span)}
    if style == "hollow" and span >= 3 and depth >= 3:
        return {(t, d) for t in range(span) for d in range(depth)
                if t in (0, span - 1) or d in (0, depth - 1)}
    if style == "diagonal":
        return {(t, t % 2) for t in range(span)}
    if style == "overhang":
        return {(t, 1 if t in (0, span - 1) else 0)
                for t in range(span)}
    if style == "elbow":
        return {(t, 0) for t in range(span)} | {(span // 2, 1)}

    # A one-cell-wide wandering path guarantees an eight-connected
    # silhouette and permits distal cells to protrude past nearer ones.
    d = int(rng.integers(depth))
    cells = set()
    for t in range(span):
        cells.add((t, d))
        d = max(0, min(depth - 1, d + int(rng.integers(-1, 2))))
    if style == "dense":
        target = int(rng.integers(max(span, (2 * span * depth) // 3),
                                  span * depth + 1))
    else:
        target = int(rng.integers(span, min(span * depth, span + 3) + 1))
    while len(cells) < target:
        frontier = {
            (t + dt, d + dd)
            for t, d in cells
            for dt in (-1, 0, 1) for dd in (-1, 0, 1)
            if 0 <= t + dt < span and 0 <= d + dd < depth
            and (t + dt, d + dd) not in cells
        }
        choices = sorted(frontier)
        cells.add(choices[int(rng.integers(len(choices)))])
    minimum = min(d for _, d in cells)
    return {(t, d - minimum) for t, d in cells}


def _motif_specs(rng, regime, first_train):
    if regime == "source_train":
        spans = [int(rng.integers(2, 6)), int(rng.integers(1, 6))]
    else:
        case = int(rng.integers(0, 5))
        if case == 0:  # One expanded motif, one small one.
            spans = [int(rng.integers(5, 9)), int(rng.integers(1, 4))]
        elif case == 1:  # Two modest motifs.
            spans = [int(rng.integers(2, 5)) for _ in range(2)]
        elif case == 2:  # Two broad motifs.
            spans = [int(rng.integers(5, 9)) for _ in range(2)]
        elif case == 3:  # A thin wide bar and an independent companion.
            spans = [int(rng.integers(5, 9)), int(rng.integers(2, 6))]
        else:
            spans = [int(rng.integers(1, 9)) for _ in range(2)]
        if rng.random() < .5:
            spans.reverse()
    specs = []
    for side, span in enumerate(spans):
        depth = int(rng.integers(1, 4 if regime == "source_train" else 5))
        style = ("bar", "diagonal", "sparse", "dense", "hollow",
                 "overhang", "elbow")[int(rng.integers(0, 7))]
        if first_train and side == 0:
            span, depth = max(3, span), 2
            style = ("diagonal", "overhang", "elbow")[int(rng.integers(0, 3))]
        if regime == "target_test" and 'case' in locals() and case == 3 and span >= 5:
            depth, style = 1, "bar"
        shape = _motif(rng, span, depth, style)
        gap = int(rng.integers(1, 4))
        specs.append((shape, gap))
    return specs


def _layout(rng, specs, arm, shared):
    spans = [max(t for t, _ in shape) + 1 for shape, _ in specs]
    distances = [gap + max(d for _, d in shape) for shape, gap in specs]
    if shared is not None:
        layout = dict(shared)
    else:
        long_length = int(rng.integers(max(7, max(spans) + 2), 18))
        separation_min = max(2 * max(distances) + 1,
                             sum(distances) + 3, 2 * arm + 3)
        separation_max = min(separation_min + 3,
                             29 - sum(distances))
        separation = int(rng.integers(separation_min, separation_max + 1))
        remaining = 29 - separation - sum(distances)
        extra0 = int(rng.integers(0, min(3, remaining) + 1))
        extra1 = int(rng.integers(0, min(3, remaining - extra0) + 1))
        exterior0 = distances[0] + extra0
        exterior1 = distances[1] + extra1
        tangent0 = int(rng.integers(1, 6))
        tangent1 = int(rng.integers(1, 6))
        layout = dict(long_length=long_length, separation=separation,
                      exterior0=exterior0, exterior1=exterior1,
                      tangent0=tangent0, tangent1=tangent1)
    assert layout["long_length"] >= max(spans) + 2
    assert layout["separation"] > 2 * max(distances)
    assert layout["separation"] > sum(distances) + 1
    assert layout["exterior0"] >= distances[0]
    assert layout["exterior1"] >= distances[1]
    return layout


def _draw_structure(grid, orientation, axis0, axis1, t0, t1, arm, color):
    if orientation == "vertical":
        grid[t0:t1 + 1, axis0] = color
        grid[t0:t1 + 1, axis1] = color
        for d in range(1, arm + 1):
            grid[t0, axis0 + d] = grid[t1, axis0 + d] = color
            grid[t0, axis1 - d] = grid[t1, axis1 - d] = color
    else:
        grid[axis0, t0:t1 + 1] = color
        grid[axis1, t0:t1 + 1] = color
        for d in range(1, arm + 1):
            grid[axis0 + d, t0] = grid[axis0 + d, t1] = color
            grid[axis1 - d, t0] = grid[axis1 - d, t1] = color


def _pair(rng, regime, orientation, palette, shared, first_train):
    specs = _motif_specs(rng, regime, first_train)
    arm = int(rng.integers(1, 4))
    layout = _layout(rng, specs, arm, shared)
    axis0 = layout["exterior0"]
    axis1 = axis0 + layout["separation"]
    t0 = layout["tangent0"]
    t1 = t0 + layout["long_length"] - 1
    axis_size = axis1 + layout["exterior1"] + 1
    tangent_size = t1 + layout["tangent1"] + 1
    shape = ((tangent_size, axis_size) if orientation == "vertical"
             else (axis_size, tangent_size))
    assert max(shape) <= 30
    background, structure, payload = palette
    inp = np.full(shape, background, dtype=np.int8)
    _draw_structure(inp, orientation, axis0, axis1, t0, t1, arm, structure)
    original, reflected = set(), set()
    for side, (motif, gap) in enumerate(specs):
        span = max(t for t, _ in motif) + 1
        offset = int(rng.integers(1, layout["long_length"] - span))
        axis = axis0 if side == 0 else axis1
        for t, d in motif:
            distance = gap + d
            tangent = t0 + offset + t
            interior_axis = axis + distance if side == 0 else axis - distance
            exterior_axis = axis - distance if side == 0 else axis + distance
            if orientation == "vertical":
                original.add((tangent, interior_axis))
                reflected.add((tangent, exterior_axis))
            else:
                original.add((interior_axis, tangent))
                reflected.add((exterior_axis, tangent))
    assert all(inp[p] == background for p in original | reflected)
    for point in original:
        inp[point] = payload
    out = inp.copy()
    for point in original:
        out[point] = background
    for point in reflected:
        out[point] = payload
    return {"input": inp, "output": out}


def _key(array):
    return (array.shape, array.astype(np.int16, copy=False).tobytes())


def _brackets(cells):
    if not cells:
        return []
    rows, cols = zip(*cells)
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    found = []
    for arm in (1, 2, 3):
        if r1 - r0 + 1 >= 5 and c1 - c0 >= 2 * arm + 3:
            expected = {(r, c) for r in range(r0, r1 + 1) for c in (c0, c1)}
            expected |= {(r, c) for r in (r0, r1)
                         for d in range(1, arm + 1)
                         for c in (c0 + d, c1 - d)}
            if expected == cells:
                found.append(dict(orientation="vertical", arm=arm,
                                  axis0=c0, axis1=c1, t0=r0, t1=r1))
        if c1 - c0 + 1 >= 5 and r1 - r0 >= 2 * arm + 3:
            expected = {(r, c) for c in range(c0, c1 + 1) for r in (r0, r1)}
            expected |= {(r, c) for c in (c0, c1)
                         for d in range(1, arm + 1)
                         for r in (r0 + d, r1 - d)}
            if expected == cells:
                found.append(dict(orientation="horizontal", arm=arm,
                                  axis0=r0, axis1=r1, t0=c0, t1=c1))
    return found


def _connected8(cells):
    if not cells:
        return False
    remaining = set(cells)
    stack = [remaining.pop()]
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
    return not remaining


def _analysis(inp, out):
    assert inp.shape == out.shape
    input_colors = {int(v) for v in np.unique(inp)}
    assert len(input_colors) == 3
    assert input_colors == {int(v) for v in np.unique(out)}
    changed = inp != out
    assert changed.any()
    transitions = {(int(a), int(b)) for a, b in zip(inp[changed], out[changed])}
    changed_colors = {v for pair in transitions for v in pair}
    assert len(changed_colors) == 2
    x, y = tuple(changed_colors)
    assert transitions == {(x, y), (y, x)}
    structure = next(iter(input_colors - changed_colors))
    structures = set(map(tuple, np.argwhere(inp == structure)))
    assert structures == set(map(tuple, np.argwhere(out == structure)))
    results = []
    for bracket in _brackets(structures):
        for background, payload in ((x, y), (y, x)):
            original = set(map(tuple, np.argwhere(inp == payload)))
            drawn = set(map(tuple, np.argwhere(out == payload)))
            if not original or not drawn or original & structures or drawn & structures:
                continue
            groups = [set(), set()]
            reflected = set()
            distances = [[], []]
            valid = True
            for r, c in original:
                coordinate = c if bracket["orientation"] == "vertical" else r
                a0, a1 = bracket["axis0"], bracket["axis1"]
                if not a0 < coordinate < a1:
                    valid = False
                    break
                d0, d1 = coordinate - a0, a1 - coordinate
                if d0 == d1:
                    valid = False
                    break
                side = 0 if d0 < d1 else 1
                distance = min(d0, d1)
                target_axis = a0 - distance if side == 0 else a1 + distance
                target = ((r, target_axis) if bracket["orientation"] == "vertical"
                          else (target_axis, c))
                if not (0 <= target[0] < inp.shape[0]
                        and 0 <= target[1] < inp.shape[1]):
                    valid = False
                    break
                groups[side].add((r, c))
                distances[side].append(distance)
                reflected.add(target)
            if not valid or reflected != drawn or not all(_connected8(g) for g in groups):
                continue
            if any(abs(r - s) <= 1 and abs(c - t) <= 1
                   for r, c in groups[0] for s, t in groups[1]):
                continue
            expected_in = np.full(inp.shape, background, dtype=inp.dtype)
            expected_out = expected_in.copy()
            for point in structures:
                expected_in[point] = expected_out[point] = structure
            for point in original:
                expected_in[point] = payload
            for point in reflected:
                expected_out[point] = payload
            if not np.array_equal(inp, expected_in) or not np.array_equal(out, expected_out):
                continue
            spans, depths, counts, densities, gaps = [], [], [], [], []
            for side, group in enumerate(groups):
                tangents = [r if bracket["orientation"] == "vertical" else c
                            for r, c in group]
                span = max(tangents) - min(tangents) + 1
                depth = max(distances[side]) - min(distances[side]) + 1
                spans.append(span)
                depths.append(depth)
                counts.append(len(group))
                densities.append(len(group) / (span * depth))
                gaps.append(min(distances[side]))
            results.append(dict(bracket, background=background,
                                structure=structure, payload=payload,
                                shape=inp.shape, spans=tuple(spans),
                                depths=tuple(depths), counts=tuple(counts),
                                densities=tuple(densities), gaps=tuple(gaps),
                                long_length=bracket["t1"] - bracket["t0"] + 1,
                                separation=bracket["axis1"] - bracket["axis0"],
                                four_connected=tuple(_connected4(g) for g in groups)))
    assert len(results) == 1
    return results[0]


def _connected4(cells):
    remaining = set(cells)
    stack = [remaining.pop()]
    while stack:
        r, c = stack.pop()
        for point in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if point in remaining:
                remaining.remove(point)
                stack.append(point)
    return not remaining


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    assert 2 <= len(puzzle["train"]) <= 10
    assert 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    inputs, pairs = set(), set()
    analyses = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = pair["input"], pair["output"]
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            if inp.shape == (15, 15):
                assert "".join(str(int(v)) for v in inp.flat) not in _SOURCE_INPUTS
            key, pair_key = _key(inp), (_key(inp), _key(out))
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            analyses[split].append(_analysis(inp, out))
    # Every training example independently demonstrates both directions of
    # the same near-axis reflection. Orientation, palette and hooks are local
    # visible properties, so they need no hidden puzzle convention.
    assert any(max(a["counts"]) >= 3 and max(a["depths"]) >= 2
               for a in analyses["train"])
    for a in analyses["train"] + analyses["test"]:
        assert len(a["spans"]) == 2 and all(n >= 1 for n in a["counts"])


def generate_puzzle(
    seed: int,
    num_train: int | None = None,
    num_test: int | None = None,
) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, rng, num_train, num_test)
    base_palette = _palette(rng)
    color_mode = int(rng.integers(0, 3))
    train_orientations, test_orientations = _orientations(rng, train_count, test_count)
    shared = None
    if rng.random() < .16:
        shared = dict(long_length=int(rng.integers(12, 16)),
                      separation=15,
                      exterior0=6, exterior1=6,
                      tangent0=int(rng.integers(2, 5)),
                      tangent1=int(rng.integers(2, 5)))
    puzzle = {"train": [], "test": []}
    used = set()
    for split, orientations in (("train", train_orientations),
                                ("test", test_orientations)):
        for index, orientation in enumerate(orientations):
            for _ in range(12):
                pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
                palette = base_palette
                if (split != "train" or index != 0) and (
                    color_mode == 2 or (color_mode == 1 and rng.random() < .6)
                ):
                    palette = _palette(pair_rng)
                pair = _pair(pair_rng, "source_train" if split == "train"
                             else "target_test", orientation, palette, shared,
                             split == "train" and index == 0)
                key = _key(pair["input"])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not construct a unique pair")
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_key(puzzle):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"]))
                       for pair in puzzle[split]) for split in ("train", "test"))


def _silhouette_features(inp, analysis):
    groups = [set(), set()]
    for r, c in np.argwhere(inp == analysis["payload"]):
        axis_coordinate = c if analysis["orientation"] == "vertical" else r
        side = (0 if axis_coordinate - analysis["axis0"]
                < analysis["axis1"] - axis_coordinate else 1)
        tangent = r if analysis["orientation"] == "vertical" else c
        axis = analysis["axis0"] if side == 0 else analysis["axis1"]
        groups[side].add((int(tangent), abs(int(axis_coordinate) - axis)))
    hollow, overhang = False, False
    for group in groups:
        ts, ds = zip(*group)
        t0, t1, d0, d1 = min(ts), max(ts), min(ds), max(ds)
        border = {(t, d) for t in range(t0, t1 + 1)
                  for d in range(d0, d1 + 1)
                  if t in (t0, t1) or d in (d0, d1)}
        hollow |= (t1 - t0 >= 2 and d1 - d0 >= 2
                   and border <= group
                   and len(group) < (t1 - t0 + 1) * (d1 - d0 + 1))
        nearest = [t for t, d in group if d == d0]
        overhang |= any((t < min(nearest) or t > max(nearest)) and d > d0
                        for t, d in group)
    return hollow, overhang


def self_test() -> None:
    automatic = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p["train"]) for p in automatic}) >= 4
    assert len({len(p["test"]) for p in automatic}) >= 3
    assert len({_puzzle_key(p) for p in automatic}) == 42
    assert _puzzle_key(automatic[17]) == _puzzle_key(generate_puzzle(17))
    assert _puzzle_key(generate_puzzle(701, 7, 3)) == _puzzle_key(generate_puzzle(701, 7, 3))
    assert _puzzle_key(generate_puzzle(701, 7, 3)) != _puzzle_key(generate_puzzle(702, 7, 3))
    for train, test in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(900 + train + test, train, test)
        assert (len(p["train"]), len(p["test"])) == (train, test)
    for i, (train, test) in enumerate(((None, 1), (None, 4),
                                       (2, None), (10, None))):
        p = generate_puzzle(1000 + i, train, test)
        assert train is None or len(p["train"]) == train
        assert test is None or len(p["test"]) == test
    for train, test in ((1, None), (11, None), (None, 0),
                        (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError")
    analyses = [[_analysis(pair["input"], pair["output"])
                 for split in ("train", "test") for pair in p[split]]
                for p in automatic]
    all_a = [a for group in analyses for a in group]
    assert {a["orientation"] for a in all_a} == {"vertical", "horizontal"}
    assert {a["arm"] for a in all_a} == {1, 2, 3}
    assert len({a["shape"] for a in all_a}) >= 30
    assert len({(a["orientation"], a["long_length"], a["separation"],
                 a["arm"]) for a in all_a}) >= 70
    assert len({(a["background"], a["structure"], a["payload"])
                for a in all_a}) >= 40
    assert any(not all(a["four_connected"]) for a in all_a)
    assert any(a["shape"] == (15, 15) for a in all_a)
    assert any(len({(a["shape"], a["axis0"], a["axis1"], a["t0"], a["t1"])
                    for a in group}) > 1 for group in analyses)
    assert any(len({(a["background"], a["structure"], a["payload"])
                    for a in group}) > 1 for group in analyses)
    schedules = set()
    varying_palette = varying_arm = varying_layout = 0
    varying_length = varying_separation = varying_gap = 0
    repeated_layout = fixed_palette = 0
    diagonal = hollow = overhang = compact = source_scale = 0
    mixed_gap = recolored_diagonal = 0
    test_mixed = test_modest = test_broad = test_complex = test_thin = 0
    test_density_mix = test_nonadjacent = 0
    for puzzle, group in zip(automatic, analyses):
        train = group[:len(puzzle["train"])]
        test = group[len(puzzle["train"]):]
        schedules.add((frozenset(a["orientation"] for a in train),
                       frozenset(a["orientation"] for a in test)))
        palettes = {(a["background"], a["structure"], a["payload"])
                    for a in group}
        layouts = {(a["orientation"], a["shape"], a["axis0"], a["axis1"],
                    a["t0"], a["t1"]) for a in group}
        varying_palette += len(palettes) > 1
        fixed_palette += len(palettes) == 1
        varying_arm += len({a["arm"] for a in group}) > 1
        varying_layout += len(layouts) > 1
        repeated_layout += len(layouts) == 1
        varying_length += len({a["long_length"] for a in group}) > 1
        varying_separation += len({a["separation"] for a in group}) > 1
        varying_gap += len({gap for a in group for gap in a["gaps"]}) > 1
        mixed_gap += any(a["gaps"][0] != a["gaps"][1] for a in group)
        for split in ("train", "test"):
            for pair, a in zip(puzzle[split],
                               train if split == "train" else test):
                is_diagonal = not all(a["four_connected"])
                is_hollow, has_overhang = _silhouette_features(pair["input"], a)
                diagonal += is_diagonal
                hollow += is_hollow
                overhang += has_overhang
                recolored_diagonal += is_diagonal and len(palettes) > 1
                compact += a["shape"] == (15, 15)
                source_scale += a["long_length"] <= 8 and a["separation"] <= 7
                if split == "test":
                    test_mixed += min(a["spans"]) <= 3 and max(a["spans"]) >= 5
                    test_modest += max(a["spans"]) <= 4
                    test_broad += min(a["spans"]) >= 5
                    test_complex += min(a["spans"]) >= 4 and min(a["counts"]) >= 6
                    test_thin += min(a["depths"]) == 1
                    test_density_mix += abs(a["densities"][0] -
                                            a["densities"][1]) >= .25
                    test_nonadjacent += min(a["gaps"]) > 1
    assert (frozenset({"vertical"}), frozenset({"vertical"})) in schedules
    assert (frozenset({"horizontal"}), frozenset({"horizontal"})) in schedules
    assert (frozenset({"vertical"}), frozenset({"horizontal"})) in schedules
    assert (frozenset({"horizontal"}), frozenset({"vertical"})) in schedules
    assert any(len(train_or) == 2 for train_or, _ in schedules)
    assert varying_palette >= 10 and fixed_palette >= 5
    assert varying_arm >= 20 and varying_layout >= 25 and repeated_layout >= 1
    assert varying_length >= 15 and varying_separation >= 15
    assert varying_gap >= 25 and mixed_gap >= 25
    assert diagonal >= 30 and hollow >= 5 and overhang >= 30
    assert recolored_diagonal >= 15
    assert compact >= 1 and source_scale >= 1
    assert test_mixed >= 10 and test_modest >= 5 and test_broad >= 5
    assert test_complex >= 5
    assert test_thin >= 10 and test_density_mix >= 10 and test_nonadjacent >= 10
