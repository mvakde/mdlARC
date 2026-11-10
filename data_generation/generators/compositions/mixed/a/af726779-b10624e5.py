"""Anchored seed strips followed by a pixel-scale gap recurrence.

Only NumPy and the standard library are used. Generation has no filesystem
effects. The solver reads every convention, including both ink colors, from
the current input. See task_spec.md for the raster and clipping conventions.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_counts(num_train, num_test):
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError("num_train must be an integer in 2..10")
    if not _integer(num_test) or not 1 <= num_test <= 4:
        raise ValueError("num_test must be an integer in 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    value = np.asarray(value)
    assert value.ndim == 2 and all(1 <= n <= 30 for n in value.shape)
    assert np.issubdtype(value.dtype, np.integer)
    assert np.all((value >= 0) & (value <= 9))
    return value.astype(np.int8, copy=False)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _next(mask, blank_centers=True):
    result = np.zeros(len(mask), dtype=bool)
    result[1:-1] = mask[:-2] & mask[2:]
    if blank_centers:
        result[1:-1] &= ~mask[1:-1]
    return result


def _history(mask, blank_centers=True):
    """In-memory, bounded masks only; no execution traces are saved."""
    result = []
    current = mask.copy()
    for _ in range(len(mask)):
        current = _next(current, blank_centers)
        if not current.any():
            return result
        result.append(current)
    raise AssertionError("gap recurrence failed to contract")


def _bounds(h, w, r, c):
    return ((0, r, 0, c), (0, r, c + 1, w),
            (r + 1, h, 0, c), (r + 1, h, c + 1, w))


def _signs(reference, destination):
    return (1 if reference // 2 == destination // 2 else -1,
            1 if reference % 2 == destination % 2 else -1)


def _square(panel, color):
    points = np.argwhere(panel == color)
    assert len(points)
    r, c = points.min(0).tolist()
    bottom, right = (points.max(0) + 1).tolist()
    side = bottom - r
    assert side == right - c and len(points) == side * side
    return r, c, side


def _decode(input_grid):
    grid = _grid(input_grid)
    rows = [r for r in range(1, len(grid) - 1) if np.all(grid[r] == grid[r, 0])]
    cols = [c for c in range(1, grid.shape[1] - 1) if np.all(grid[:, c] == grid[0, c])]
    crosses = [(r, c, int(grid[r, c])) for r in rows for c in cols
               if grid[r, 0] == grid[0, c]]
    assert len(crosses) == 1, "one full, one-cell-wide cross is required"
    r, c, divider = crosses[0]
    bounds = _bounds(*grid.shape, r, c)
    panels = [grid[r0:r1, c0:c1] for r0, r1, c0, c1 in bounds]
    counts = np.bincount(np.concatenate([p.ravel() for p in panels]), minlength=10)
    bg = int(np.argmax(counts))
    assert np.count_nonzero(counts == counts[bg]) == 1 and bg != divider
    sets = [set(map(int, np.unique(p))) for p in panels]
    shared = set.intersection(*sets) - {bg}
    assert len(shared) == 1
    anchor = shared.pop()
    assert anchor != divider
    refs = [p for p in range(4) if sets[p] - {bg, anchor}]
    assert len(refs) == 1
    reference = refs[0]
    seed_colors = sets[reference] - {bg, anchor}
    assert len(seed_colors) == 1
    ink = seed_colors.pop()
    assert len({bg, divider, anchor, ink}) == 4
    anchors = [_square(p, anchor) for p in panels]
    for p in range(4):
        if p != reference:
            assert sets[p] == {bg, anchor}
    points = np.argwhere(panels[reference] == ink)
    top, left = points.min(0).tolist()
    bottom, right = (points.max(0) + 1).tolist()
    unit = bottom - top
    ar, ac, side = anchors[reference]
    assert side % unit == 0 and (right - left) % unit == 0
    assert (top - ar) % unit == 0 and (left - ac) % unit == 0
    a = side // unit
    length = (right - left) // unit
    raw = panels[reference][top:bottom, left:right]
    mask = raw[0, ::unit] == ink
    assert mask[0] and mask[-1] and np.any(_next(mask))
    assert np.array_equal(raw, np.tile(np.repeat(np.where(mask, ink, bg), unit), (unit, 1)))
    x, y = (left - ac) // unit, (top - ar) // unit
    assert x >= a + 1 or x + length <= -1, "strip and anchor need a horizontal gap"
    assert all(s % a == 0 for _, _, s in anchors)
    return {"grid": grid, "bounds": bounds, "panels": panels, "background": bg,
            "divider": divider, "anchor": anchor, "ink": ink,
            "reference": reference, "anchors": anchors, "unit": unit,
            "a": a, "x": x, "y": y, "mask": mask}


def _geometry(data, p, scales=True, reflections=True):
    ar, ac, side = data["anchors"][p]
    v = side // data["a"] if scales else data["unit"]
    sy, sx = _signs(data["reference"], p) if reflections else (1, 1)
    top = ar + (data["y"] * v if sy == 1 else side - (data["y"] + 1) * v)
    left = ac + (data["x"] * v if sx == 1 else side - (data["x"] + len(data["mask"])) * v)
    mask = np.repeat(data["mask"][::sx], v)
    return top, left, v, mask


def _render(data, grow=True, scales=True, reflections=True, clip_first=True, blank_centers=True):
    """The optional switches implement explicit necessity-test alternatives."""
    output = data["grid"].copy()
    for p, (r0, r1, c0, c1) in enumerate(data["bounds"]):
        if p == data["reference"]:
            continue
        panel = output[r0:r1, c0:c1]
        h, w = panel.shape
        top, left, v, full_mask = _geometry(data, p, scales, reflections)
        # Rasterize the complete uniformly scaled strip, then clip it.
        active = np.flatnonzero(full_mask) + left
        visible = active[(active >= 0) & (active < w)]
        for row in range(max(0, top), min(h, top + v)):
            panel[row, visible] = data["ink"]
        frontier = top + v - 1
        if not grow or not 0 <= frontier < h:
            continue
        if clip_first:
            current = np.zeros(w, dtype=bool)
            current[visible] = True
            offset = 0
        else:
            current, offset = full_mask.copy(), left
        for generation, mask in enumerate(_history(current, blank_centers), 1):
            row = frontier + 2 * generation
            if row >= h:
                break
            cols = np.flatnonzero(mask) + offset
            cols = cols[(cols >= 0) & (cols < w)]
            panel[row, cols] = data["anchor"] if generation % 2 else data["ink"]
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Place/reflect/scale the seed first, then apply the local pixel rule."""
    return _render(_decode(input_grid))


def _mask(rng, kind, depth):
    chain = [1 if i % 2 == 0 else 0 for i in range(2 * depth + 1)]
    if kind == "pure":
        bits = chain
    elif kind == "mixed":
        bits = chain + [0, 0] + [1] * int(rng.integers(2, 4))
    elif kind == "multi":
        other = int(rng.integers(1, 3))
        bits = chain + [0, 0] + [1 if i % 2 == 0 else 0 for i in range(2 * other + 1)]
    elif kind == "dense":
        bits = [1] * int(rng.integers(5, 11))
        holes = list(range(1, len(bits) - 1))
        rng.shuffle(holes)
        for pos in holes[:int(rng.integers(1, 4))]:
            bits[pos] = 0
        if not any(bits[i - 1] and not bits[i] and bits[i + 1] for i in range(1, len(bits) - 1)):
            bits = [1, 1, 0, 1, 1]
    elif kind == "one":
        bits = [1] * int(rng.integers(1, 4)) + [0] + [1] * int(rng.integers(1, 4))
    else:
        raise ValueError("unknown seed construction")
    result = np.array(bits, dtype=bool)
    return result[::-1].copy() if rng.random() < .5 else result


def _axis(rng, minima):
    minimum = sum(minima) + 1
    if minimum > 30:
        return None
    total = 30 if rng.random() < .15 else int(rng.integers(minimum, 31))
    spare = total - minimum
    extra = int(rng.integers(spare + 1))
    return minima[0] + extra, minima[1] + spare - extra


def _local_pattern(mask, a, x, y, v, sy, sx):
    yy = y if sy == 1 else a - y - 1
    xx = x if sx == 1 else a - x - len(mask)
    return yy * v, xx * v, np.repeat(mask[::sx], v)


def _placements(rng, shape, a, v, dy, dx, mask, style, depth):
    """Enumerate at most 900 positions; retain only coordinates, never grids."""
    h, w = shape
    side = a * v
    candidates = []
    first = _next(mask)
    for r in range(h - side + 1):
        top, frontier = r + dy, r + dy + v - 1
        if top < 0 or top + v > h:
            continue
        if style == "full" and v == 1 and frontier + 2 * (depth + 1) >= h:
            continue
        if style == "short" and (v != 1 or not frontier + 2 < h <= frontier + 2 * depth):
            continue
        if style != "source" and v == 1 and frontier + 2 >= h:
            continue
        for c in range(w - side + 1):
            left = c + dx
            active = np.flatnonzero(mask) + left
            visible = active[(active >= 0) & (active < w)]
            if not len(visible):
                continue
            if style in ("source", "full", "short") and not (0 <= left and left + len(mask) <= w):
                continue
            local = np.zeros(w, dtype=bool)
            local[visible] = True
            if v == 1 and not _next(local).any():
                continue
            if style == "edge":
                unbounded = np.flatnonzero(first) + left
                unbounded = unbounded[(unbounded >= 0) & (unbounded < w)]
                if np.array_equal(unbounded, np.flatnonzero(_next(local))):
                    continue
            candidates.append((r, c))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _build(rng, mode, reference):
    # Two teaching examples are required even when num_train is only two.
    if mode in ("teach", "reduce"):
        mask = np.array([1, 0, 1, 0, 1, 1, 1], dtype=bool)
        if rng.random() < .5:
            mask = mask[::-1].copy()
        a, u, gap = 1, (1 if mode == "teach" else 2), 1
    elif mode == "shift":
        depth = int(rng.integers(4, 6))
        mask = _mask(rng, "pure", depth)
        if rng.random() < .55:
            mask = np.concatenate((mask, np.ones(int(rng.integers(1, 3)), dtype=bool)))
            if rng.random() < .5:
                mask = mask[::-1].copy()
        a, u, gap = int(rng.choice([1, 1, 2])), 1, 1
    else:
        kind = str(rng.choice(["pure", "mixed", "multi", "dense", "one"]))
        depth = int(rng.integers(1, 5 if mode == "test" else 4))
        mask = _mask(rng, kind, depth)
        a = int(rng.choice([1, 1, 2, 3]))
        u = int(rng.choice([1, 1, 1, 2, 3]))
        gap = int(rng.integers(1, 3))
    depth = len(_history(mask))
    length = len(mask)
    if (length + a + gap) * u > 24:
        return None
    x = a + gap if rng.random() < .5 else -(length + gap)
    # Off-anchor row offsets expose vertical reflection, rather than encoding it
    # as a hidden growth direction. Growth is ALWAYS globally downward.
    y = int(rng.choice([-2, -1, a, a + 1])) if mode in ("teach", "reduce") else int(rng.integers(-2, a + 3))
    destinations = [p for p in range(4) if p != reference]
    rng.shuffle(destinations)
    units = {reference: u}
    styles = {reference: "source"}
    if mode == "teach":
        for p, v, style in zip(destinations, [1, 1, 2], ["full", "edge", "free"]):
            units[p], styles[p] = v, style
    elif mode == "reduce":
        for p in destinations:
            units[p], styles[p] = 1, "full"
    elif mode == "shift":
        for p, v, style in zip(destinations, [1, 1, 2], ["short", "full", "free"]):
            units[p], styles[p] = v, style
    else:
        for p in destinations:
            units[p] = int(rng.choice([1, 1, 1, 2, 3]))
            styles[p] = "full" if rng.random() < .45 else "free"
        units[destinations[0]] = 1
        if mode == "test" and depth >= 2:
            styles[destinations[0]] = "short"
    patterns = {}
    min_rows, min_cols = [3, 3], [3, 3]
    for p in range(4):
        v = units[p]
        sy, sx = _signs(reference, p)
        dy, dx, bits = _local_pattern(mask, a, x, y, v, sy, sx)
        patterns[p] = dy, dx, bits
        side = a * v
        tail = 2 * (depth + 1) if styles[p] == "full" and v == 1 else 2 if p != reference and v == 1 else 0
        low, high = min(0, dy), max(side, dy + v + tail)
        need_h = high - low
        full_w = (a + gap + length) * v
        need_w = full_w if styles[p] in ("source", "full", "short") else max(side + 3, min(full_w, 8))
        min_rows[p // 2] = max(min_rows[p // 2], need_h)
        min_cols[p % 2] = max(min_cols[p % 2], need_w)
    heights, widths = _axis(rng, min_rows), _axis(rng, min_cols)
    if heights is None or widths is None:
        return None
    h, w = sum(heights) + 1, sum(widths) + 1
    bounds = _bounds(h, w, heights[0], widths[0])
    bg, divider, anchor, ink = map(int, rng.choice(10, 4, replace=False))
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[heights[0], :] = divider
    grid[:, widths[0]] = divider
    for p, (r0, r1, c0, c1) in enumerate(bounds):
        dy, dx, bits = patterns[p]
        v = units[p]
        pos = _placements(rng, (r1 - r0, c1 - c0), a, v, dy, dx, bits, styles[p], depth)
        if pos is None:
            return None
        r, c = pos
        panel = grid[r0:r1, c0:c1]
        side = a * v
        panel[r:r + side, c:c + side] = anchor
        if p == reference:
            columns = c + dx + np.flatnonzero(bits)
            panel[r + dy:r + dy + v, columns] = ink
    data = _decode(grid)
    return {"input": grid, "output": _render(data)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Reproducible bounded generation, with one coherent input-visible rule."""
    num_train, num_test = _check_counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    refs = [int(p) for p in rng.permutation(4)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            mode = (("teach" if index == 0 else "reduce" if index == 1 else "free")
                    if split == "train" else "shift" if index == 0 else "test")
            reference = refs[index] if split == "train" and index < 2 else int(rng.integers(4))
            for _ in range(192):
                pair = _build(rng, mode, reference)
                if pair is not None and _key(pair["input"]) not in seen:
                    break
            else:
                raise RuntimeError("bounded construction exhausted")
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(data):
    records = []
    for p, panel in enumerate(data["panels"]):
        if p == data["reference"]:
            continue
        top, left, v, bits = _geometry(data, p)
        h, w = panel.shape
        active = np.flatnonzero(bits) + left
        visible = active[(active >= 0) & (active < w)]
        local = np.zeros(w, dtype=bool)
        local[visible] = True
        generations = _history(local)
        frontier = top + v - 1
        assert 0 <= top and top + v <= h and len(visible)
        ar, ac, side = data["anchors"][p]
        assert np.all((visible < ac) | (visible >= ac + side)), "seed/cascade cannot overwrite anchor"
        records.append({"unit": v, "scale": (v, data["unit"]),
                        "depth": len(generations),
                        "visible_depth": min(len(generations), (h - 1 - frontier) // 2),
                        "natural_stop_visible": bool(generations) and frontier + 2 * (len(generations) + 1) < h,
                        "seed_clipped": len(visible) < len(active),
                        "bottom_clipped": bool(generations) and frontier + 2 * len(generations) >= h})
    return records


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, training, testing = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            data = _decode(inp)
            records = _features(data)
            assert any(r["visible_depth"] > 0 for r in records), "every pair must exercise the stencil"
            assert np.array_equal(out, _render(data))
            assert np.array_equal(out[inp != data["background"]], inp[inp != data["background"]])
            assert set(map(int, np.unique(out))) == set(map(int, np.unique(inp)))
            for p, (r0, r1, c0, c1) in enumerate(data["bounds"]):
                if p == data["reference"]:
                    assert np.array_equal(out[r0:r1, c0:c1], inp[r0:r1, c0:c1])
            if split == "train":
                training.append((data, records, out))
            else:
                testing.append((data, records, out))
    assert len({d["reference"] for d, _, _ in training}) >= 2
    assert any(any(r["visible_depth"] >= 2 and r["natural_stop_visible"] for r in records)
               for _, records, _ in training), "parity and natural termination need visible evidence"
    assert any(any(n < d for n, d in [r["scale"] for r in records]) for _, records, _ in training)
    assert any(any(n > d for n, d in [r["scale"] for r in records]) for _, records, _ in training)
    assert any(not np.array_equal(out, _render(d, clip_first=False)) for d, _, out in training)
    for alternative in ({"grow": False}, {"scales": False}, {"reflections": False}, {"blank_centers": False}):
        assert any(not np.array_equal(out, _render(d, **alternative)) for d, _, out in training)
    assert max(r["visible_depth"] for _, records, _ in testing for r in records) > max(
        r["visible_depth"] for _, records, _ in training for r in records)
    assert any(r["bottom_clipped"] for _, records, _ in testing for r in records)


def _hand_regressions():
    # Literal coordinates, not values computed by the renderer under test.
    # Reference TL: anchor at (1,1), strip at row 3, columns 3/5/7/8.
    inp = np.zeros((19, 23), dtype=np.int8)
    inp[9] = 9
    inp[:, 11] = 9
    inp[1, 1] = inp[1, 21] = inp[15, 1] = inp[15, 21] = 2
    inp[3, [3, 5, 7, 8]] = 7
    expected = inp.copy()
    expected[3, [14, 15, 17, 19]] = 7
    expected[5, [16, 18]] = 2
    expected[7, 17] = 7
    expected[13, [3, 5, 7, 8, 14, 15, 17, 19]] = 7
    expected[15, [4, 6, 16, 18]] = 2
    expected[17, [5, 17]] = 7
    assert np.array_equal(solve(inp), expected)
    # Enlarging only TR turns every gap into a two-pixel gap. Its strip is
    # two rows thick and clipped on the left; no derived row is permitted.
    enlarged = inp.copy()
    enlarged[:9, 12:] = 0
    enlarged[1:3, 20:22] = 2
    expected[:9, 12:] = 0
    expected[1:3, 20:22] = 2
    expected[5:7, [12, 13, 16, 17]] = 7
    assert np.array_equal(solve(enlarged), expected)


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.uint8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds, streaming statistics, hand regressions, counts, mutations.

    Keeps only one puzzle and small hash/feature sets at a time; writes nothing.
    Returned statistics are descriptive, not an independent semantic approval.
    """
    assert _next(np.array([1, 0, 1], dtype=bool)).tolist() == [False, True, False]
    assert not _next(np.ones(3, dtype=bool)).any()
    assert len(_history(np.array([1, 0, 1, 0, 1, 1], dtype=bool))) == 2
    assert not _next(np.repeat(np.array([1, 0, 1], dtype=bool), 2)).any()
    _hand_regressions()
    shapes, masks, refs, roles, source_units, anchor_sizes, ratios = (set() for _ in range(7))
    signatures, counts = set(), Counter()
    normalized = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        if seed < 3:
            assert _digest(puzzle) == _digest(generate_puzzle(seed))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                data = _decode(pair["input"])
                assert np.array_equal(solve(pair["input"]), pair["output"])
                shapes.add(data["grid"].shape)
                masks.add(tuple(map(int, data["mask"])))
                refs.add(data["reference"])
                source_units.add(data["unit"])
                anchor_sizes.update(s for _, _, s in data["anchors"])
                roles.add(tuple(data[k] for k in ("background", "divider", "anchor", "ink")))
                table = {data[k]: i for i, k in enumerate(("background", "divider", "anchor", "ink"))}
                normalized.add((data["grid"].shape, bytes(table[int(c)] for c in data["grid"].ravel())))
                for record in _features(data):
                    v, u = record["scale"]
                    g = int(np.gcd(v, u))
                    ratios.add((v // g, u // g))
                    counts["seed_clipped"] += record["seed_clipped"]
                    counts["bottom_clipped"] += record["bottom_clipped"]
                    counts["inert_scaled_strip"] += record["depth"] == 0
                    counts["multi_generation"] += record["visible_depth"] >= 2
                    counts["depth_" + str(record["visible_depth"])] += 1
                counts["pairs"] += 1
                counts["solid_run"] += bool(np.any(data["mask"][:-1] & data["mask"][1:]))
    assert len(signatures) == 42 and len(shapes) >= 35 and len(masks) >= 20
    assert refs == {0, 1, 2, 3} and source_units >= {1, 2}
    assert len(roles) >= 100 and len(normalized) >= 180
    assert {(1, 2), (2, 1), (1, 1)} <= ratios
    assert counts["depth_4"] + counts["depth_5"] >= 42
    assert all(counts[k] >= 10 for k in ("seed_clipped", "bottom_clipped", "inert_scaled_strip", "multi_generation", "solid_run"))
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            generate_puzzle(700 + train * 10 + test, train, test)
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Corrupt a blank spacer: validation must inspect output pixels.
    broken = generate_puzzle(91, 2, 1)
    data = _decode(broken["test"][0]["input"])
    point = tuple(np.argwhere(broken["test"][0]["output"] == data["background"])[0])
    broken["test"][0]["output"][point] = data["ink"]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("output corruption was accepted")
    return {"seeds": 42, "distinct_puzzles": len(signatures), "shapes": len(shapes),
            "seed_masks": len(masks), "color_normalized_inputs": len(normalized),
            "palette_roles": len(roles), "source_units": sorted(source_units),
            "anchor_sides": sorted(anchor_sizes), "scale_ratios": sorted(ratios),
            "counts": dict(counts), "independent_solver": False}
