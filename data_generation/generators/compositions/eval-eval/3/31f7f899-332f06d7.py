"""Folded track sorting, partitioned by full-square reachability.

Only NumPy and the standard library are used. Generation never reads evidence,
and solve reads only its argument. No execution traces or sample files are saved.
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid shape")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a.astype(np.int8, copy=True)


def _neighbors(p, shape):
    r, c = p
    for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]:
            yield q


def _components(mask):
    labels = np.zeros(mask.shape, dtype=np.int16)
    pieces = []
    for raw in np.argwhere(mask):
        p = tuple(map(int, raw))
        if labels[p]:
            continue
        label = len(pieces) + 1
        queue = [p]
        labels[p] = label
        for current in queue:
            for q in _neighbors(current, mask.shape):
                if mask[q] and not labels[q]:
                    labels[q] = label
                    queue.append(q)
        pieces.append(queue)
    return labels, pieces


def _box(points):
    a = np.asarray(points, dtype=int)
    low, high = a.min(axis=0), a.max(axis=0)
    r, c = map(int, low)
    h, w = map(int, high - low + 1)
    return r, c, h, w, len(points) == h * w


def _bar_box(points):
    r, c, h, w, solid = _box(points)
    _need(solid and 1 <= min(h, w) <= 2 and 3 <= max(h, w) <= 6,
          "accent component is not a nonsquare crossbar")
    return r, c, h, w


def _placement_labels(allowed, k):
    valid = np.lib.stride_tricks.sliding_window_view(allowed, (k, k)).all(axis=(-1, -2))
    return _components(valid)[0]


def _station_component(labels, point, k):
    r, c = point
    window = labels[max(0, r - k + 1):min(r + 1, labels.shape[0]),
                    max(0, c - k + 1):min(c + 1, labels.shape[1])]
    choices = set(map(int, np.unique(window))) - {0}
    _need(len(choices) == 1, "station has ambiguous square-reachable membership")
    return choices.pop()


def _analyze(value):
    """Recover roles, the directed guide, templates and placement components."""
    g = _grid(value)
    wall = int(g[0, 0])
    _need(all(int(g[p]) == wall for p in ((0, -1), (-1, 0), (-1, -1))),
          "wall corners disagree")
    _need(np.all(g[0] == wall) and np.all(g[-1] == wall)
          and np.all(g[:, 0] == wall) and np.all(g[:, -1] == wall), "wall frame absent")
    _need(len(_components(g != wall)[1]) == 1, "disconnected floor or unrelated object")
    _need(len(_components(g == wall)[1]) == 1, "closed alternate corridor route")
    counts = np.bincount(g.ravel(), minlength=10)
    counts[wall] = 0
    floor = int(np.argmax(counts))
    _need(counts[floor] > 0 and np.count_nonzero(counts == counts[floor]) == 1,
          "floor color is not geometrically dominant")
    other = set(map(int, np.unique(g))) - {wall, floor}
    squares = []
    for color in other:
        points = list(map(tuple, np.argwhere(g == color)))
        r, c, h, w, solid = _box(points)
        if solid and h == w and 2 <= h <= 4:
            squares.append((color, (r, c), h))
    _need(len(squares) == 1, "gauge must be the unique solid square color")
    gauge, gauge_at, k = squares[0]
    others = other - {gauge}
    components = {color: _components(g == color)[1] for color in sorted(others)}
    guide_candidates = []
    for color, pieces in components.items():
        try:
            for points in pieces:
                _bar_box(points)
        except AssertionError:
            guide_candidates.append(color)
    _need(len(guide_candidates) == 1, "guide role is ambiguous")
    guide = guide_candidates[0]
    path_mask = g == guide
    bars = []
    for color in sorted(others - {guide}):
        for points in components[color]:
            r, c, h, w = _bar_box(points)
            if h > w:
                crossings = [(x, c) for x in range(r, r + h)
                             if c > 0 and c + w < g.shape[1]
                             and g[x, c - 1] == guide and g[x, c + w] == guide]
                _need(len(crossings) == 1, "bar lacks one horizontal crossing")
                rr, _ = crossings[0]
                crossing_cells = [(rr, y) for y in range(c, c + w)]
            else:
                crossings = [(r, y) for y in range(c, c + w)
                             if r > 0 and r + h < g.shape[0]
                             and g[r - 1, y] == guide and g[r + h, y] == guide]
                _need(len(crossings) == 1, "bar lacks one vertical crossing")
                _, cc = crossings[0]
                crossing_cells = [(x, cc) for x in range(r, r + h)]
            for p in crossing_cells:
                path_mask[p] = True
            bars.append({"color": color, "cells": tuple(points), "cross": crossing_cells})
    _need(2 <= len(bars) <= 8, "station count outside supported family")
    _need(len({b["color"] for b in bars}) >= 2, "need distinct slot identities")
    path_cells = set(map(tuple, np.argwhere(path_mask)))
    adjacency = {p: [q for q in _neighbors(p, g.shape) if q in path_cells] for p in path_cells}
    _need(all(len(qs) in (1, 2) for qs in adjacency.values()), "guide branches or has an isolated cell")
    ends = [p for p, qs in adjacency.items() if len(qs) == 1]
    _need(len(ends) == 2, "guide is not an open chain")
    starts = [p for p in ends if any(g[q] == gauge for q in _neighbors(p, g.shape))]
    _need(len(starts) == 1, "guide start is not marked uniquely by gauge")
    path = [starts[0]]
    previous = None
    while True:
        choices = [q for q in adjacency[path[-1]] if q != previous]
        if not choices:
            break
        previous, q = path[-1], choices[0]
        _need(q not in path, "guide cycle")
        path.append(q)
    _need(len(path) == len(path_cells), "disconnected guide fragments")
    order = {p: i for i, p in enumerate(path)}
    labels = _placement_labels(g != wall, k)
    templates = {}
    for bar in bars:
        indices = sorted(order[p] for p in bar["cross"])
        _need(indices == list(range(indices[0], indices[-1] + 1)), "bent crossing")
        i, j = indices[0], indices[-1]
        _need(i > 0 and j + 1 < len(path), "station at unexposed guide end")
        anchor = path[i]
        d = tuple(path[i][axis] - path[i - 1][axis] for axis in (0, 1))
        _need(all(tuple(path[t + 1][axis] - path[t][axis] for axis in (0, 1)) == d
                  for t in range(i - 1, j + 1)), "guide bends inside a bar")
        n = (-d[1], d[0])
        offsets = [(p[0] - anchor[0]) * n[0] + (p[1] - anchor[1]) * n[1] for p in bar["cells"]]
        lo, hi = min(offsets), max(offsets)
        _need(lo <= 0 <= hi, "bar misses guide")
        length = hi - lo + 1
        _need(templates.setdefault(length, (lo, hi)) == (lo, hi), "different-offset length tie")
        bar.update(anchor=anchor, d=d, span=(lo, hi), width=len(indices), index=i,
                   region=_station_component(labels, anchor, k))
        _need(all(_station_component(labels, p, k) == bar["region"] for p in bar["cross"]),
              "one station crosses a square-reachability boundary")
    bars.sort(key=lambda b: b["index"])
    region_order = [b["region"] for b in bars]
    collapsed = [r for i, r in enumerate(region_order) if not i or r != region_order[i - 1]]
    _need(len(collapsed) == len(set(collapsed)), "guide revisits a sorting region")
    return {"grid": g, "wall": wall, "floor": floor, "gauge": gauge, "guide": guide,
            "k": k, "gauge_at": gauge_at, "path": tuple(path), "bars": bars,
            "labels": labels}


def _assigned(analysis, mode="regional"):
    bars = analysis["bars"]
    result = [b["span"] for b in bars]
    groups = ([list(range(len(bars)))] if mode == "global" else
              [[i for i, b in enumerate(bars) if b["region"] == region]
               for region in dict.fromkeys(b["region"] for b in bars)])
    for indices in groups:
        if mode == "reverse":
            templates = [bars[i]["span"] for i in indices[::-1]]
        else:
            templates = [bars[i]["span"] for i in sorted(indices, key=lambda i:
                         (bars[i]["span"][1] - bars[i]["span"][0] + 1)
                         * (bars[i]["width"] if mode == "area" else 1))]
        for i, span in zip(indices, templates):
            result[i] = span
    return result


def _paint_bar(grid, bar, span):
    r, c = bar["anchor"]
    dr, dc = bar["d"]
    nr, nc = -dc, dr
    for j in range(bar["width"]):
        for s in range(span[0], span[1] + 1):
            x, y = r + j * dr + s * nr, c + j * dc + s * nc
            _need(0 <= x < grid.shape[0] and 0 <= y < grid.shape[1], "transferred span leaves canvas")
            grid[x, y] = bar["color"]


def _render_analysis(a, spans):
    out = a["grid"].copy()
    for bar in a["bars"]:
        for p in bar["cells"]:
            out[p] = a["floor"]
    for p in a["path"]:
        out[p] = a["guide"]
    for bar, span in zip(a["bars"], spans):
        _paint_bar(out, bar, span)
    _need(np.array_equal(out != a["wall"], a["grid"] != a["wall"]), "template does not fit corridor")
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _analyze(input_grid)
    return _render_analysis(a, _assigned(a))


def _fixed(a):
    return [(b["anchor"], b["d"], b["width"], b["color"], b["region"]) for b in a["bars"]]


def _check_pair(pair):
    """Relational conservation check; does not use solve or its sorter/painter."""
    a, b = _analyze(pair["input"]), _analyze(pair["output"])
    _need(a["grid"].shape == b["grid"].shape, "canvas changed")
    for key in ("wall", "floor", "gauge", "guide", "gauge_at", "k", "path"):
        _need(a[key] == b[key], "scene or role changed: " + key)
    _need(_fixed(a) == _fixed(b), "fixed station changed")
    _need(np.array_equal(a["grid"] != a["wall"], b["grid"] != b["wall"]), "floor geometry changed")
    changed_zone = np.zeros(a["grid"].shape, dtype=bool)
    for bar in a["bars"] + b["bars"]:
        for p in bar["cells"]:
            changed_zone[p] = True
    _need(np.array_equal(a["grid"][~changed_zone], b["grid"][~changed_zone]), "unrelated cells changed")
    for region in dict.fromkeys(bar["region"] for bar in a["bars"]):
        before = [bar["span"] for bar in a["bars"] if bar["region"] == region]
        after = [bar["span"] for bar in b["bars"] if bar["region"] == region]
        _need(Counter(before) == Counter(after), "regional interval multiset changed")
        lengths = [hi - lo + 1 for lo, hi in after]
        _need(all(x <= y for x, y in zip(lengths, lengths[1:])), "regional lengths unsorted")
    return a, b


def _features(a):
    bars = a["bars"]
    initial = [b["span"] for b in bars]
    target = _assigned(a)
    moved = [i for i, (x, y) in enumerate(zip(initial, target)) if x != y]
    transported = False
    for i in moved:
        span = target[i]
        if span[0] + span[1] == 0:
            continue
        if any(b["span"] == span and b["region"] == bars[i]["region"]
               and b["d"] != bars[i]["d"] for b in bars):
            transported = True
    screen_indices = sorted(range(len(bars)), key=lambda i: bars[i]["anchor"])
    screen_target = initial.copy()
    for region in dict.fromkeys(b["region"] for b in bars):
        ids = [i for i in screen_indices if bars[i]["region"] == region]
        spans = sorted((bars[i]["span"] for i in ids), key=lambda s: s[1] - s[0])
        for i, span in zip(ids, spans):
            screen_target[i] = span
    lengths = tuple(s[1] - s[0] + 1 for s in initial)
    headings = [(q[0] - p[0], q[1] - p[1]) for p, q in zip(a["path"], a["path"][1:])]
    return {"stations": len(bars), "k": a["k"], "shape": a["grid"].shape,
            "regions": len(set(b["region"] for b in bars)),
            "group_sizes": tuple(sum(b["region"] == r for b in bars)
                                 for r in dict.fromkeys(b["region"] for b in bars)),
            "changed": bool(moved), "omit_b_differs": target != _assigned(a, "global"),
            "area_differs": target != _assigned(a, "area"),
            "reverse_differs": target != _assigned(a, "reverse"),
            "screen_differs": target != screen_target,
            "offcenter_moved": any(abs(target[i][0] + target[i][1]) > 1 for i in moved),
            "even_bias_moved": any((target[i][1] - target[i][0] + 1) % 2 == 0
                                   and abs(sum(target[i])) == 1 for i in moved),
            "turned_transfer": transported,
            "changed_colors": len({bars[i]["color"] for i in moved}),
            "mixed_widths": len({b["width"] for b in bars}) > 1,
            "lengths": lengths, "ties": len(set(lengths)) < len(lengths),
            "turns": sum(x != y for x, y in zip(headings, headings[1:])),
            "roles": tuple(a[key] for key in ("wall", "floor", "guide", "gauge"))}


def _count(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= value <= high:
        raise ValueError(name + " is outside its legal range")
    return int(value)


def _walk(rng, count):
    """Randomized finite depth-first search on nine nodes, no template bank."""
    start = tuple(map(int, rng.integers(0, 3, 2)))
    path, used = [start], {start}

    def extend():
        if len(path) == count:
            return True
        options = [q for q in _neighbors(path[-1], (3, 3)) if q not in used]
        rng.shuffle(options)
        for q in options:
            path.append(q)
            used.add(q)
            if extend():
                return True
            used.remove(q)
            path.pop()
        return False

    if not extend():
        # A nine-node path needs an endpoint on the majority checkerboard color.
        # Retry each start in randomized order, still an explicitly finite search.
        starts = [(r, c) for r in range(3) for c in range(3)]
        rng.shuffle(starts)
        for start in starts:
            path[:], used = [start], {start}
            if extend():
                break
        else:
            raise AssertionError("finite chamber walk search failed")
    return path


def _span_table(rng, lengths, witness):
    result = {}
    for length in sorted(set(lengths)):
        low, high = max(-3, 1 - length), min(0, 4 - length)
        lo = int(rng.integers(low, high + 1))
        result[length] = (lo, lo + length - 1)
    if witness:
        sign = 1 if rng.integers(2) else -1
        result[3] = (0, 2) if sign == 1 else (-2, 0)
        result[4] = (-2, 1) if sign == 1 else (-1, 2)
    return result


def _construct(rng, n, k, palette, accent_pool, witness):
    q = int(rng.integers(7, 9))
    m = q // 2
    walk = _walk(rng, n + 1)
    row_gaps = list(map(int, rng.integers(1, 3, 2)))
    col_gaps = list(map(int, rng.integers(1, 3, 2)))
    row_starts = [0, q + row_gaps[0], 2 * q + sum(row_gaps)]
    col_starts = [0, q + col_gaps[0], 2 * q + sum(col_gaps)]
    origins = [(row_starts[r], col_starts[c]) for r, c in walk]
    centers = [(r + m, c + m) for r, c in origins]
    allowed = np.zeros((28, 28), dtype=bool)
    guide_mask = np.zeros_like(allowed)
    for r, c in origins:
        allowed[r:r + q, c:c + q] = True
    if witness == 1:
        lengths, groups = [5, 3, 4, 3, 5], [0, 0, 0, 1, 1]
    elif witness == 2:
        lengths, groups = [4, 3, 5, 3, 4, 5], [0, 0, 1, 1, 1, 1]
    else:
        lengths = list(map(int, rng.integers(3, q - 1, n)))
        groups, region = [0], 0
        for _ in range(n - 1):
            region += int(rng.random() < 0.32)
            groups.append(region)
    for i, (a, b) in enumerate(zip(centers, centers[1:])):
        blocked = i > 0 and groups[i] != groups[i - 1]
        width = int(rng.integers(1, k)) if blocked else int(rng.integers(k, q + 1))
        if witness:
            # Show the actual comparison threshold, plus passable reductions.
            # Different gauge sizes therefore cannot share a fixed threshold.
            width = k - 1 if blocked else q if i == 0 else k + (i % 2)
        offset = int(rng.integers(max(0, m - width + 1), min(m, q - width) + 1))
        if a[0] == b[0]:
            left, right = sorted((a[1] - m, b[1] - m))
            r = a[0] - m + offset
            allowed[r:r + width, left + q:right] = True
        else:
            top, bottom = sorted((a[0] - m, b[0] - m))
            c = a[1] - m + offset
            allowed[top + q:bottom, c:c + width] = True
        dr = int(np.sign(b[0] - a[0]))
        dc = int(np.sign(b[1] - a[1]))
        p = a
        guide_mask[p] = True
        while p != b:
            p = p[0] + dr, p[1] + dc
            guide_mask[p] = True
    wall, floor, guide, gauge = palette
    base = np.full(allowed.shape, wall, dtype=np.int8)
    base[allowed] = floor
    base[guide_mask] = guide
    r, c = centers[0]
    r, c = r - k // 2, c - k // 2
    base[r:r + k, c:c + k] = gauge
    table = _span_table(rng, lengths, witness)
    widths = list(map(int, rng.integers(1, 3, n)))
    if witness == 1:
        widths[:3] = [1, 2, 2]
    colors = list(map(int, rng.choice(accent_pool, n)))
    colors[0], colors[1] = int(accent_pool[0]), int(accent_pool[1])
    bars = []
    for i, (a, b) in enumerate(zip(centers, centers[1:])):
        d = (int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1])))
        # The incoming arm is straight, and its center cell stays exposed.
        backward = (-d[0], -d[1])
        arm = m if backward[0] < 0 or backward[1] < 0 else q - 1 - m
        distance = int(rng.integers(widths[i], arm + 1))
        if i == n - 1:
            # A 1..2-cell terminal guide tip cannot be mistaken for an accent
            # bar, whose long dimension is at least three. It marks no value.
            distance = int(rng.integers(widths[i], min(arm, widths[i] + 1) + 1))
        anchor = (b[0] - distance * d[0], b[1] - distance * d[1])
        bars.append({"anchor": anchor, "d": d, "width": widths[i], "color": colors[i]})
    inp, out = base.copy(), base.copy()
    spans = [table[length] for length in lengths]
    final = spans.copy()
    for group in dict.fromkeys(groups):
        indices = [i for i, value in enumerate(groups) if value == group]
        # Construction uses explicit throat widths; verification reconstructs
        # square placements and checks relational conservation from cells.
        sorted_spans = [table[length] for length in sorted(lengths[i] for i in indices)]
        for i, span in zip(indices, sorted_spans):
            final[i] = span
    for bar, before, after in zip(bars, spans, final):
        _paint_bar(inp, bar, before)
        _paint_bar(out, bar, after)
    points = np.argwhere(allowed)
    low, high = points.min(axis=0), points.max(axis=0) + 1
    height, width = map(int, high - low)
    pads = []
    for dim in (height, width):
        extra = int(rng.integers(2, min(6, 30 - dim) + 1))
        before = int(rng.integers(1, extra))
        pads.append((before, extra - before))
    turns, reflect = int(rng.integers(4)), bool(rng.integers(2))

    def transform(grid):
        grid = grid[low[0]:high[0], low[1]:high[1]]
        grid = np.pad(grid, pads, constant_values=wall)
        grid = np.rot90(grid, turns)
        return np.array(np.fliplr(grid) if reflect else grid, dtype=np.int8, copy=True)

    return {"input": transform(inp), "output": transform(out)}


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count("num_train", num_train, 2, 10)
    nq = _count("num_test", num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(10, 4, replace=False)))
    remaining = [c for c in range(10) if c not in palette]
    accent_pool = list(map(int, rng.choice(remaining, int(rng.integers(2, 7)), replace=False)))
    sides = list(map(int, rng.choice([2, 3, 4], 2, replace=False)))
    specs = [(5, sides[0], 1), (6, sides[1], 2)]
    specs += [(int(rng.integers(4, 7)), int(rng.choice(sides)), 0) for _ in range(nt - 2)]
    test_specs = [(int(rng.integers(7, 9)), int(rng.choice(sides)), 0) for _ in range(nq)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, requests in (("train", specs), ("test", test_specs)):
        for n, k, witness in requests:
            for _ in range(96):
                pair = _construct(rng, n, k, palette, accent_pool, witness)
                a, _ = _check_pair(pair)
                f = _features(a)
                if _key(pair["input"]) in seen:
                    continue
                if witness and not (f["changed"] and f["omit_b_differs"] and f["turned_transfer"]
                                    and f["offcenter_moved"] and f["even_bias_moved"]
                                    and f["changed_colors"] >= 2 and f["reverse_differs"]
                                    and f["screen_differs"]):
                    continue
                if witness == 1 and not f["area_differs"]:
                    continue
                if split == "test" and not (f["changed"] and f["omit_b_differs"]):
                    continue
                puzzle[split].append(pair)
                seen.add(_key(pair["input"]))
                break
            else:
                raise AssertionError("bounded pair construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _need(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid pair counts")
    seen, measured, palette = set(), {"train": [], "test": []}, None
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            a, _ = _check_pair(pair)
            key = _key(a["grid"])
            _need(key not in seen, "duplicate input")
            seen.add(key)
            f = _features(a)
            if palette is None:
                palette = f["roles"]
            _need(f["roles"] == palette, "structural role convention changed")
            measured[split].append(f)
    train, test = measured["train"], measured["test"]
    _need(all(4 <= f["stations"] <= 6 for f in train), "training station range")
    _need({5, 6} <= {f["stations"] for f in train}, "missing count witnesses")
    _need(all(f["stations"] in (7, 8) for f in test), "test count shift missing")
    _need({f["k"] for f in test} <= {f["k"] for f in train}, "unseen test gauge size")
    _need(sum(f["changed"] and f["omit_b_differs"] for f in train) >= 2,
          "both parent concepts need two demonstrations")
    for feature in ("area_differs", "reverse_differs", "offcenter_moved", "even_bias_moved",
                    "turned_transfer", "screen_differs", "mixed_widths"):
        _need(any(f[feature] for f in train), "missing training evidence: " + feature)
    _need(any(f["changed_colors"] >= 2 for f in train), "slot color retention is not demonstrated")
    _need(all(f["changed"] and f["omit_b_differs"] for f in test), "test omits one parent")
    return None


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _gauge_intervention(grid, new_k):
    """Change only the visible gauge footprint, restoring its hidden guide arm."""
    a = _analyze(grid)
    out = a["grid"].copy()
    r, c = a["gauge_at"]
    out[out == a["gauge"]] = a["floor"]
    p = a["path"][0]
    next_p = a["path"][1]
    dr, dc = next_p[0] - p[0], next_p[1] - p[1]
    # Even squares have no single central cell after a reflection. Keep the
    # visible guide arm's row/column, rather than inventing a diagonal root.
    center = (r + a["k"] // 2, p[1]) if dr else (p[0], c + a["k"] // 2)
    current = center
    while current != p:
        out[current] = a["guide"]
        current = current[0] + dr, current[1] + dc
    rr, cc = center[0] - new_k // 2, center[1] - new_k // 2
    out[rr:rr + new_k, cc:cc + new_k] = a["gauge"]
    return out


def self_test():
    """Bounded 42-seed streaming audit; returns compact aggregate measurements."""
    # Direct-coordinate regressions: the same U-shaped floor, guide and bars
    # have spans [5,3,4]. A width-two throat blocks a three-square, giving
    # [3,5] | [4], but admits a two-square, giving [3,4,5]. Expected cells are
    # specified directly, without the constructor, sorter or bar painter.
    for k in (2, 3):
        base = np.zeros((18, 18), dtype=np.int8)
        for r, c in ((1, 1), (1, 10), (10, 10), (10, 1)):
            base[r:r + 7, c:c + 7] = 1
        base[3:6, 8:10] = 1
        base[8:10, 12:15] = 1
        base[12:14, 8:10] = 1
        base[4, 4:14] = 2
        base[4:14, 13] = 2
        base[13, 4:14] = 2
        start = 4 - k // 2
        base[start:start + k, start:start + k] = 3
        inp, expected = base.copy(), base.copy()
        inp[2:7, 11] = 4
        inp[11:13, 13:16] = 5
        inp[11:15, 6] = 6
        expected[2:5, 11] = 4
        expected[11:13, 11:15 if k == 2 else 16] = 5
        expected[11:16 if k == 2 else 15, 6] = 6
        _need(np.array_equal(solve(inp), expected), "explicit gauge/throat regression failed")
        _check_pair({"input": inp, "output": expected})
    signatures, shapes, structures, profiles, role_sets = set(), set(), set(), set(), [set() for _ in range(4)]
    hist = {key: Counter() for key in ("stations", "k", "regions", "turns", "group_sizes")}
    flags = Counter()
    pairs = 0
    sample = None
    intervention_found = False
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = _digest(puzzle)
        _need(signature == _digest(generate_puzzle(seed)), "nondeterministic puzzle")
        signatures.add(signature)
        if seed == 0:
            sample = puzzle
        for split in ("train", "test"):
            for pair in puzzle[split]:
                a = _analyze(pair["input"])
                _need(np.array_equal(solve(pair["input"]), pair["output"]), "solver disagreement")
                f = _features(a)
                pairs += 1
                shapes.add(f["shape"])
                for key in hist:
                    hist[key][f[key]] += 1
                profiles.add(f["group_sizes"])
                for i, color in enumerate(f["roles"]):
                    role_sets[i].add(color)
                for key, value in f.items():
                    if isinstance(value, bool) and value:
                        flags[key] += 1
                mapping = {}
                normalized = bytes(mapping.setdefault(int(x), len(mapping)) for x in a["grid"].ravel())
                structures.add(hashlib.sha256(bytes(a["grid"].shape) + normalized).hexdigest())
                permutation = np.array([9, 2, 7, 0, 5, 1, 8, 4, 6, 3], dtype=np.int8)
                _need(np.array_equal(solve(permutation[pair["input"]]), permutation[pair["output"]]),
                      "recoloring changed rule")
                if not intervention_found:
                    for new_k in (2, 3, 4):
                        if new_k == a["k"]:
                            continue
                        changed = _gauge_intervention(pair["input"], new_k)
                        other = _analyze(changed)
                        if _assigned(a) != _assigned(other):
                            _need(_features(a)["group_sizes"] != _features(other)["group_sizes"],
                                  "gauge intervention did not alter regions")
                            _need(np.array_equal(solve(changed), _render_analysis(other, _assigned(other))),
                                  "gauge intervention solve failed")
                            intervention_found = True
                            break
    _need(len(signatures) == 42, "cross-seed puzzle repetition")
    _need(len(shapes) >= 20 and len(structures) >= 180, "structural diversity collapsed")
    _need(set(hist["stations"]) == {4, 5, 6, 7, 8}, "count diversity missing")
    _need(set(hist["k"]) == {2, 3, 4}, "size diversity missing")
    _need(len(profiles) >= 15 and max(hist["regions"]) >= 4, "regional diversity missing")
    _need(max(hist["turns"]) >= 5, "folded paths missing")
    _need(all(colors == set(range(10)) for colors in role_sets), "role palette diversity missing")
    _need(flags["ties"] > 0 and flags["screen_differs"] >= 100, "semantic variation missing")
    _need(intervention_found, "no demonstrated gauge-size intervention")
    # Check every legal explicit request, plus type/boundary errors.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(4242 + 10 * nt + nq, nt, nq)
            _need((len(puzzle["train"]), len(puzzle["test"])) == (nt, nq), "count request ignored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, True), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    # Deliberately wrong outputs must fail the supplied-output checker.
    first = next(pair for pair in sample["train"] if _features(_analyze(pair["input"]))["area_differs"])
    a = _analyze(first["input"])
    mutations = [first["input"].copy(), _render_analysis(a, _assigned(a, "global")),
                 _render_analysis(a, _assigned(a, "area"))]
    broken = first["output"].copy()
    broken[0, 0] = a["floor"]
    mutations.append(broken)
    for wrong in mutations:
        try:
            _check_pair({"input": first["input"], "output": wrong})
        except AssertionError:
            pass
        else:
            raise AssertionError("accepted a bad-output mutation")
    return {"seeds": 42, "pairs": pairs, "direct_coordinate_regressions": 2,
            "distinct_puzzles": len(signatures),
            "shapes": len(shapes), "normalized_structures": len(structures),
            "region_profiles": len(profiles), "gauge_intervention": intervention_found,
            "feature_counts": dict(sorted(flags.items())),
            "histograms": {key: [[list(value) if isinstance(value, tuple) else value, count]
                                  for value, count in sorted(counter.items())]
                           for key, counter in hist.items()},
            "independent_semantic_verification": False}
