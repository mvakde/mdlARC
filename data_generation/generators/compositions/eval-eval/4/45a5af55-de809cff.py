"""Concentric depth partners: repair damage and compose depth-ranked stamps.

All inference is input-only. Generation uses no parent code or stored examples.
"""
from __future__ import annotations

import numpy as np


def _arc_grid(value):
    grid = np.asarray(value)
    if (grid.ndim != 2 or not all(1 <= n <= 30 for n in grid.shape)
            or not np.issubdtype(grid.dtype, np.integer)
            or np.any((grid < 0) | (grid > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return grid.astype(np.int8, copy=False)


def _geometry(shape, box):
    r0, c0, r1, c1 = box
    rr, cc = np.indices(shape)
    inside = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    depth = np.minimum.reduce((rr - r0, cc - c0, r1 - 1 - rr, c1 - 1 - cc))
    depth[~inside] = -1
    return depth, (min(r1 - r0, c1 - c0) - 1) // 2


def _analyze(value):
    """Infer the occupied rectangle, majority layer code, core and damage."""
    grid = _arc_grid(value)
    support = np.argwhere(grid != 0)
    if not len(support):
        raise ValueError("No occupied rectangle")
    lo, hi = support.min(axis=0), support.max(axis=0) + 1
    box = (*map(int, lo), *map(int, hi))
    depth, layers = _geometry(grid.shape, box)
    if layers < 2:
        raise ValueError("At least two ordinary layers are required")
    core = grid[depth == layers]
    if len(np.unique(core)) != 1 or core[0] == 0:
        raise ValueError("The central core must be intact and monochromatic")
    highlight = int(core[0])
    code = []
    for d in range(layers):
        shell = grid[depth == d]
        counts = np.bincount(shell, minlength=10)
        color = int(counts.argmax())
        if color in (0, highlight) or 2 * counts[color] <= len(shell):
            raise ValueError("Every layer needs a strict majority of its true color")
        code.append(color)
    if not set(map(int, np.unique(grid))) <= {0, highlight, *code}:
        raise ValueError("Noise must use a visible layer color or the core color")
    clean = np.zeros(grid.shape, dtype=np.int8)
    for d, color in enumerate(code):
        clean[depth == d] = color
    clean[depth == layers] = highlight
    holes = [tuple(map(int, p)) for p in np.argwhere((depth >= 0) & (depth < layers) & (grid == 0))]
    if not holes:
        raise ValueError("At least one erased layer cell is required")
    wrong = [tuple(map(int, p)) for p in np.argwhere((grid != clean) & (grid != 0))]
    return {"grid": grid, "box": box, "depth": depth, "layers": layers,
            "code": tuple(code), "highlight": highlight, "clean": clean,
            "holes": holes, "wrong": wrong}


def _neighbors(shape, point):
    r, c = point
    return [(y, x) for y in range(max(0, r - 1), min(shape[0], r + 2))
            for x in range(max(0, c - 1), min(shape[1], c + 2)) if (y, x) != point]


def _compose(a, *, partner="mirror", priority="deep", mark_wrong=False,
             retain_noise=False, centers_only=False, reverse_holes=False):
    """Stamp composer; optional alternatives are used only by evidence checks."""
    output = (a["grid"] if retain_noise else a["clean"]).copy()
    holes = a["holes"] + (a["wrong"] if mark_wrong else [])
    if reverse_holes:
        holes = list(reversed(holes))
    rank = np.full(output.shape, -1000, dtype=int)
    code, length = a["code"], a["layers"]
    for point in holes:
        d = int(a["depth"][point])
        if partner == "mirror":
            color = code[length - 1 - d]
        elif partner == "next":
            color = code[(d + 1) % length]
        elif partner == "previous":
            color = code[(d - 1) % length]
        elif partner == "outer":
            color = code[0]
        elif partner == "own":
            color = code[d]
        elif partner == "color_only":
            # Equal original colors are forced to have equal partners.
            color = code[length - 1 - code.index(code[d])]
        else:
            raise ValueError("Unknown partner policy")
        if centers_only:
            continue
        score = d if priority == "deep" else -d
        for q in _neighbors(output.shape, point):
            accept = (score >= rank[q] if priority in ("deep", "shallow")
                      else rank[q] == -1000 if priority == "first" else True)
            if accept:
                output[q] = color
                rank[q] = score
    for point in holes:
        output[point] = a["highlight"]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the scene; each hole borrows from its reversed layer index."""
    return _compose(_analyze(input_grid))


def _reference_output(a):
    """An authored cross-check: gather neighboring holes at each destination.

Unlike the scatter-and-rank composer, this computes each destination once.
It is not an externally supplied independent solver.
"""
    output = a["clean"].copy()
    holes = set(a["holes"])
    for r in range(output.shape[0]):
        for c in range(output.shape[1]):
            point = (r, c)
            if point in holes:
                output[point] = a["highlight"]
                continue
            nearby = holes.intersection(_neighbors(output.shape, point))
            if nearby:
                d = max(int(a["depth"][p]) for p in nearby)
                output[point] = a["code"][a["layers"] - 1 - d]
    return output


def _signature(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.tobytes()


def _normalized(values):
    labels = {}
    return tuple(labels.setdefault(int(c), len(labels)) for c in values)


def _evidence(a):
    holes = a["holes"]
    center_set = set(holes)
    demands = {}
    for p in holes:
        d = int(a["depth"][p])
        color = a["code"][a["layers"] - 1 - d]
        for q in _neighbors(a["grid"].shape, p):
            if q not in center_set:
                demands.setdefault(q, []).append((d, color))
    conflicts = sum(len({color for _, color in ds}) > 1 for ds in demands.values())
    fills_to_partners = {}
    for p in holes:
        d = int(a["depth"][p])
        fills_to_partners.setdefault(a["code"][d], set()).add(a["code"][a["layers"] - 1 - d])
    out = _compose(a)
    return {"layers": a["layers"], "shape": a["grid"].shape,
            "box_shape": (a["box"][2] - a["box"][0], a["box"][3] - a["box"][1]),
            "code": _normalized(a["code"]), "palette": len(set(a["code"])),
            "holes": len(holes), "wrong": len(a["wrong"]), "conflicts": conflicts,
            "repeated": any(x == y for x, y in zip(a["code"], a["code"][1:])),
            "same_color_distinct_partners": any(len(v) > 1 for v in fills_to_partners.values()),
            "all_depths": {int(a["depth"][p]) for p in holes} == set(range(a["layers"])),
            "adjacent": any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1
                            for i, p in enumerate(holes) for q in holes[:i]),
            "clipped": any(len(_neighbors(out.shape, p)) < 8 for p in holes),
            "outside": any(a["depth"][q] < 0 for p in holes for q in _neighbors(out.shape, p)),
            "core_crossing": any(a["depth"][q] == a["layers"] for p in holes for q in _neighbors(out.shape, p)),
            "visible_repair": any(out[p] == a["clean"][p] for p in a["wrong"])}


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[k], list) for k in ("train", "test"))
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    analyses, seen, highlights = [], set(), set()
    for pair in puzzle["train"] + puzzle["test"]:
        assert isinstance(pair, dict) and set(pair) == {"input", "output"}
        inp, out = _arc_grid(pair["input"]), _arc_grid(pair["output"])
        assert inp.shape == out.shape
        key = _signature(inp)
        assert key not in seen, "Duplicate input"
        seen.add(key)
        a = _analyze(inp)
        expected = _compose(a)
        assert np.array_equal(out, expected), "Incorrect input-derived output"
        assert np.all(out[tuple(np.array(a["holes"]).T)] == a["highlight"])
        untouched = np.ones(inp.shape, dtype=bool)
        for p in a["holes"]:
            untouched[p] = False
            for q in _neighbors(inp.shape, p):
                untouched[q] = False
        assert np.array_equal(out[untouched], a["clean"][untouched])
        assert np.array_equal(expected, _compose(a, reverse_holes=True)), "Order-dependent overlap"
        highlights.add(a["highlight"])
        analyses.append(a)
    assert len(highlights) == 1, "The visible core color is shared within a puzzle"
    train = analyses[:nt]
    features = [_evidence(a) for a in train]
    assert len({a["layers"] for a in train}) >= 2
    assert any(f["all_depths"] for f in features)
    assert any(f["repeated"] and f["same_color_distinct_partners"] and f["conflicts"] for f in features)
    assert any(f["clipped"] for f in features) and any(f["outside"] for f in features)
    assert any(f["visible_repair"] for f in features)
    # Collective evidence must actually rule out simpler answers, using inputs.
    alternatives = [{"partner": p} for p in ("next", "previous", "outer", "own", "color_only")]
    alternatives += [{"priority": p} for p in ("shallow", "first", "last")]
    alternatives += [{"mark_wrong": True}, {"retain_noise": True}, {"centers_only": True}]
    for options in alternatives:
        assert any(not np.array_equal(_compose(a, **options), _compose(a)) for a in train), (
            "Training fails to distinguish alternative", options)
    assert all(not np.array_equal(a["clean"], _compose(a)) for a in train), "Parent B has no effect"
    assert any(a["layers"] not in {b["layers"] for b in train} for a in analyses[nt:]), "Missing depth-count shift"
    assert not np.array_equal(_compose(analyses[nt]), _compose(analyses[nt], partner="own")), (
        "First test does not expose depth partnership")


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _sample_geometry(rng, layers, profile):
    short = 2 * layers + int(rng.integers(1, 3))
    elongation = int(rng.integers(0, min(7, 30 - short) + 1))
    h, w = short, short + elongation
    if rng.integers(2):
        h, w = w, h
    minimum_margin = 1 if profile == "collision" else 0
    margins = []
    for side in (h, w):
        room = 30 - side
        if room < 2 * minimum_margin:
            return None
        before = int(rng.integers(minimum_margin, min(3, room - minimum_margin) + 1))
        after = int(rng.integers(minimum_margin, min(3, room - before) + 1))
        margins.append([before, after])
    if profile == "isolated":
        # The outer-depth demonstration includes actual canvas clipping.
        axis, end = int(rng.integers(2)), int(rng.integers(2))
        margins[axis][end] = 0
    (top, bottom), (left, right) = margins
    return (h + top + bottom, w + left + right), (top, left, top + h, left + w)


def _sample_code(rng, layers, highlight, profile, palette_policy):
    available = [c for c in range(1, 10) if c != highlight]
    if palette_policy is not None:
        available = list(palette_policy)
    lower = 3 if profile == "collision" else 2 if profile == "isolated" else 1
    upper = min(len(available), layers - 1 if profile == "collision" else layers)
    size = int(rng.integers(lower, upper + 1))
    palette = list(map(int, rng.choice(available, size, replace=False)))
    if profile == "collision":
        # Equal intended fills, unequal depth partners: literal color is not an address.
        code = list(map(int, rng.choice(palette, layers)))
        code[0] = code[1] = palette[0]
        code[-2], code[-1] = palette[1], palette[2]
    elif profile == "isolated":
        code = list(map(int, rng.choice(palette, layers)))
        code[0], code[-1] = palette[0], palette[1]
    else:
        style = int(rng.integers(3))
        if style == 0:
            cuts = sorted(map(int, rng.choice(np.arange(1, layers), size - 1, replace=False)))
            code = [palette[sum(d >= cut for cut in cuts)] for d in range(layers)]
        elif style == 1:
            code = [palette[d % size] for d in range(layers)]
        else:
            code = list(map(int, rng.choice(palette, layers)))
    return tuple(code)


def _damage_ok(depth, layers, points, point):
    d = int(depth[point])
    if d < 0 or d >= layers or point in points:
        return False
    size = int(np.count_nonzero(depth == d))
    used = sum(depth[p] == d for p in points)
    return 2 * (used + 1) < size


def _select_holes(rng, depth, layers, box, profile):
    shape = depth.shape
    pools = {d: [tuple(map(int, p)) for p in np.argwhere(depth == d)] for d in range(layers)}
    if profile in ("isolated", "collision"):
        for _ in range(80):
            if profile == "isolated":
                outer = [p for p in pools[0] if len(_neighbors(shape, p)) < 8]
                holes = [outer[int(rng.integers(len(outer)))]]
                remaining = range(layers - 1, 0, -1)
            else:
                r0, c0, r1, c1 = box
                top_col = int(rng.integers(c0 + 2, c1 - 2))
                bottom_col = int(rng.integers(c0 + 2, c1 - 2))
                # Top and bottom witnesses defeat both first-writer and last-writer order.
                holes = [(r0, top_col), (r0 + 1, top_col),
                         (r1 - 1, bottom_col), (r1 - 2, bottom_col)]
                remaining = range(layers - 1, 1, -1)
            for d in remaining:
                candidates = pools[d].copy()
                rng.shuffle(candidates)
                for p in candidates:
                    if all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= 3 for q in holes):
                        holes.append(p)
                        break
                else:
                    break
            else:
                return holes
        return None
    candidates = [p for d in pools for p in pools[d]]
    rng.shuffle(candidates)
    style = int(rng.integers(3))
    target = int(rng.integers(1, min(32, max(4, len(candidates) // 12)) + 1))
    holes = []
    for p in candidates:
        if len(holes) >= target:
            break
        if not _damage_ok(depth, layers, holes, p):
            continue
        if style == 0 and any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) < 3 for q in holes):
            continue
        holes.append(p)
        if style > 0 and rng.random() < .5:
            offsets = [(0, 1), (1, 0), (1, 1), (1, -1), (0, 2), (2, 0)]
            rng.shuffle(offsets)
            for dr, dc in offsets[:int(rng.integers(1, 4))]:
                q = p[0] + dr, p[1] + dc
                if (len(holes) < target and 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]
                        and _damage_ok(depth, layers, holes, q)):
                    holes.append(q)
    return holes


def _make_pair(rng, layers, profile, highlight, palette_policy, informative=False):
    for _ in range(100):
        geometry = _sample_geometry(rng, layers, profile)
        if geometry is None:
            continue
        shape, box = geometry
        depth, _ = _geometry(shape, box)
        code = _sample_code(rng, layers, highlight, profile, palette_policy)
        clean = np.zeros(shape, dtype=np.int8)
        for d, color in enumerate(code):
            clean[depth == d] = color
        clean[depth == layers] = highlight
        holes = _select_holes(rng, depth, layers, box, profile)
        if not holes:
            continue
        inp = clean.copy()
        for p in holes:
            inp[p] = 0
        # Preserve nonzero support on every side so the occupied bounds remain visible.
        support = np.argwhere(inp != 0)
        if tuple(support.min(axis=0)) != box[:2] or tuple(support.max(axis=0) + 1) != box[2:]:
            continue
        candidates = [tuple(map(int, p)) for p in np.argwhere((depth >= 0) & (depth < layers))]
        rng.shuffle(candidates)
        # Training contains an error outside all stamps, so repair without marking is visible.
        target = int(rng.integers(1, 5)) if profile != "free" else int(rng.integers(0, 9))
        used, wrong = list(holes), []
        for p in candidates:
            if len(wrong) >= target:
                break
            if not _damage_ok(depth, layers, used, p):
                continue
            if not wrong and any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1 for q in holes):
                continue
            # The first ordinary error is the core color; it is still not a hole.
            colors = [c for c in set(code) | {highlight} if c != clean[p]]
            colors.sort()
            inp[p] = highlight if not wrong else colors[int(rng.integers(len(colors)))]
            used.append(p)
            wrong.append(p)
        if profile != "free" and not wrong:
            continue
        a = _analyze(inp)
        assert a["box"] == box and a["code"] == code and np.array_equal(a["clean"], clean)
        if informative and np.array_equal(_compose(a), _compose(a, partner="own")):
            continue
        return {"input": inp, "output": _compose(a)}
    raise RuntimeError("Bounded scene construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    highlight = int(rng.integers(1, 10))
    # A recurring-palette policy is a distribution choice, not hidden solver state.
    palette_policy = None
    if rng.random() < .4:
        available = [c for c in range(1, 10) if c != highlight]
        palette_policy = tuple(map(int, rng.choice(available, int(rng.integers(3, 9)), replace=False)))
    first_depth = int(rng.integers(3, 7))
    second_depth = int(rng.choice([d for d in range(4, 8) if d != first_depth]))
    train_depths = [first_depth, second_depth] + list(map(int, rng.integers(2, 13, max(0, nt - 2))))
    train, test, seen = [], [], set()
    for i, layers in enumerate(train_depths):
        profile = "isolated" if i == 0 else "collision" if i == 1 else "free"
        for _ in range(20):
            pair = _make_pair(rng, layers, profile, highlight, palette_policy)
            key = _signature(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("Distinct-input construction exhausted")
        seen.add(key)
        train.append(pair)
    # A genuinely unseen number of depths is required, with upward and downward shifts.
    unseen = [d for d in range(2, 15) if d not in train_depths]
    direction = int(rng.integers(3))
    preferred = ([d for d in unseen if d > max(train_depths)] if direction == 0 else
                 [d for d in unseen if d < min(train_depths)] if direction == 1 else unseen)
    first_test_depth = int(rng.choice(preferred or unseen))
    for i in range(ne):
        layers = first_test_depth if i == 0 else int(rng.integers(2, 15))
        for _ in range(20):
            pair = _make_pair(rng, layers, "free", highlight, palette_policy, informative=i == 0)
            key = _signature(pair["input"])
            if key not in seen:
                break
        else:
            raise RuntimeError("Distinct-input construction exhausted")
        seen.add(key)
        test.append(pair)
    rng.shuffle(train)
    puzzle = {"train": train, "test": test}
    validate_generated_puzzle(puzzle)
    return puzzle


def _fixture(shape, box, code, highlight, holes, wrong=()):
    depth, length = _geometry(shape, box)
    assert length == len(code)
    clean = np.zeros(shape, dtype=np.int8)
    for d, color in enumerate(code):
        clean[depth == d] = color
    clean[depth == length] = highlight
    inp = clean.copy()
    for p in holes:
        inp[p] = 0
    for p, color in wrong:
        inp[p] = color
    return inp


def self_test():
    """Bounded 42-seed diversity audit, semantic fixtures and rejection probes."""
    features, signatures, colors, highlights = [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_signature(p["input"]) for p in puzzle["train"] + puzzle["test"]))
        for pair in puzzle["train"] + puzzle["test"]:
            a = _analyze(pair["input"])
            assert np.array_equal(_reference_output(a), pair["output"])
            features.append(_evidence(a))
            colors.update(map(int, np.unique(pair["input"])))
            highlights.add(a["highlight"])
    assert len(signatures) == 42 and colors == set(range(10)) and len(highlights) >= 8
    assert len({f["shape"] for f in features}) >= 70
    assert len({(f["box_shape"], f["code"]) for f in features}) >= 120
    assert len({f["layers"] for f in features}) >= 10
    assert len({f["palette"] for f in features}) >= 6
    assert max(f["holes"] for f in features) >= 20 and min(f["holes"] for f in features) == 1
    assert any(f["wrong"] == 0 for f in features)
    for name in ("conflicts", "same_color_distinct_partners", "adjacent", "outside", "clipped", "core_crossing"):
        assert sum(bool(f[name]) for f in features) >= 20, name
    # Manually specified positions expose depth partnership and both collision orders.
    inp = _fixture((11, 11), (1, 1, 10, 10), (2, 2, 3, 4), 8,
                   [(1, 5), (2, 5), (9, 5), (8, 5)], [((3, 3), 8)])
    out = solve(inp)
    assert out[0, 5] == out[10, 5] == 4
    assert out[1, 4] == out[9, 4] == 3
    assert out[3, 3] == 3 and all(out[p] == 8 for p in [(1, 5), (2, 5), (9, 5), (8, 5)])
    assert np.array_equal(out, _reference_output(_analyze(inp)))
    corner = _fixture((5, 7), (0, 0, 5, 7), (5, 6), 8, [(0, 0), (1, 4)], [((0, 3), 8)])
    out = solve(corner)
    assert out[0, 0] == 8 and out[0, 1] == out[1, 0] == out[1, 1] == 6
    assert out[2, 3] == 5  # The halo crosses the protected input core in the output.
    same = _fixture((7, 7), (0, 0, 7, 7), (1, 2, 3), 8, [(0, 2), (0, 3)])
    out = solve(same)
    assert out[0, 2] == out[0, 3] == 8 and out[1, 2] == 3
    largest = _fixture((30, 30), (0, 0, 30, 30), tuple(1 + d % 7 for d in range(14)), 9, [(13, 13)])
    assert solve(largest).shape == (30, 30)
    assert np.array_equal(solve(largest), _reference_output(_analyze(largest)))
    long_core = _fixture((7, 30), (1, 0, 6, 30), (3, 4), 8,
                         [(1, 29), (2, 14), (5, 0)], [((1, 4), 8)])
    assert solve(long_core)[3, 14] == 3
    assert np.array_equal(solve(long_core), _reference_output(_analyze(long_core)))
    tied = _fixture((7, 7), (0, 0, 7, 7), (1, 2, 3), 8, [(1, 1)])
    tied_depth, _ = _geometry(tied.shape, (0, 0, 7, 7))
    shell = np.argwhere(tied_depth == 0)
    tied[tuple(shell[:len(shell) // 2].T)] = 2
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError("Ambiguous majority tie accepted")
    # All explicit legal counts and type/range failures are bounded too.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(1700 + 10 * nt + ne, nt, ne)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == ne
    for nt, ne in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (4, None)]:
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    a, b = generate_puzzle(87654321, 7, 3), generate_puzzle(87654321, 7, 3)
    assert all(np.array_equal(p[k], q[k]) for split in a for p, q in zip(a[split], b[split]) for k in p)
    bad = {split: [{k: v.copy() for k, v in p.items()} for p in a[split]] for split in a}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Validator accepted an incorrect output")
    return {"seeds": 42, "pairs": len(features), "input_shapes": len({f["shape"] for f in features}),
            "geometry_codes": len({(f["box_shape"], f["code"]) for f in features}),
            "depth_counts": sorted({f["layers"] for f in features}),
            "layer_palette_sizes": sorted({f["palette"] for f in features}),
            "hole_range": [min(f["holes"] for f in features), max(f["holes"] for f in features)],
            "conflict_pairs": sum(f["conflicts"] > 0 for f in features)}
