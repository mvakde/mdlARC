"""Alternating divider regions gate scaffold-registered motif exchange.

Only NumPy and the standard library are used. No file access or saved examples
are needed by generation, validation, or the input-only solver.
"""

from collections import deque
from math import gcd
import hashlib
import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)


def _require(ok, message):
    if not bool(ok):
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_), "noninteger grid")
    _require(np.all((a >= 0) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=np.int16)
    groups = []
    for sy, sx in zip(*np.nonzero(mask)):
        sy, sx = int(sy), int(sx)
        if labels[sy, sx] >= 0:
            continue
        number = len(groups)
        labels[sy, sx] = number
        todo, group = [(sy, sx)], []
        while todo:
            y, x = todo.pop()
            group.append((y, x))
            for dy, dx in N8 if diagonal else N4:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and labels[yy, xx] < 0:
                    labels[yy, xx] = number
                    todo.append((yy, xx))
        groups.append(group)
    return labels, groups


def _bounds(points):
    return min(y for y, x in points), max(y for y, x in points), min(x for y, x in points), max(x for y, x in points)


def _sig(mapping):
    return tuple(sorted((y, x, int(c)) for (y, x), c in mapping.items()))


def _geometry(wall):
    """Segment the underlying field, treating motifs and color seeds as field."""
    h, w = wall.shape
    _require(not wall[-1, 0], "lower-left anchor is blocked")
    wl, paths = _components(wall)
    _require(paths, "no divider")
    labels, regions = _components(~wall)
    _require(len(regions) == len(paths) + 1, "ineffective or closed divider")
    adjacency = [set() for _ in regions]
    turns, endpoint_sides = [], []
    for i, path in enumerate(paths):
        ends, neighbors = [], set()
        pathset = set(path)
        for y, x in path:
            degree = 0
            for dy, dx in N4:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w:
                    degree += int(wl[yy, xx] == i)
                    if labels[yy, xx] >= 0:
                        neighbors.add(int(labels[yy, xx]))
            _require(degree in (1, 2), "divider is branched or self-touching")
            if degree == 1:
                ends.append((y, x))
            for dy, dx in N8:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w:
                    _require(wl[yy, xx] in (-1, i), "different dividers touch")
        _require(len(ends) == 2 and all(y in (0, h - 1) or x in (0, w - 1) for y, x in ends), "divider needs two boundary endpoints")
        _require(len(neighbors) == 2, "divider must border exactly two regions")
        a, b = sorted(neighbors)
        adjacency[a].add(b)
        adjacency[b].add(a)
        previous, current = None, ends[0]
        directions = []
        while current != ends[1]:
            choices = [(current[0] + dy, current[1] + dx) for dy, dx in N4
                       if (current[0] + dy, current[1] + dx) in pathset and (current[0] + dy, current[1] + dx) != previous]
            _require(len(choices) == 1, "ambiguous path")
            nxt = choices[0]
            directions.append((nxt[0] - current[0], nxt[1] - current[1]))
            previous, current = current, nxt
        turns.append(sum(a != b for a, b in zip(directions, directions[1:])))
        def sides(p):
            y, x = p
            return tuple(name for ok, name in ((y == 0, "T"), (y == h - 1, "B"), (x == 0, "L"), (x == w - 1, "R")) if ok)
        endpoint_sides.append((sides(ends[0]), sides(ends[1])))
    root = int(labels[-1, 0])
    depth = [-1] * len(regions)
    depth[root] = 0
    todo = deque([root])
    while todo:
        a = todo.popleft()
        for b in sorted(adjacency[a]):
            if depth[b] < 0:
                depth[b] = depth[a] + 1
                todo.append(b)
            else:
                _require((depth[b] - depth[a]) % 2 == 1, "inconsistent parity")
    _require(all(d >= 0 for d in depth), "disconnected region graph")
    _require(sum(map(len, adjacency)) == 2 * len(paths), "region graph is not a tree")
    return {"labels": labels, "depth": depth, "root": root, "turns": turns,
            "degree": max(map(len, adjacency)), "endpoints": endpoint_sides, "paths": len(paths)}


def _primitive(crop):
    scale = 0
    for a in (crop, crop.T):
        for row in a:
            cuts = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, len(row)]
            for length in np.diff(cuts):
                scale = gcd(scale, int(length))
    small = crop[::scale, ::scale]
    _require(np.array_equal(np.repeat(np.repeat(small, scale, axis=0), scale, axis=1), crop), "nonuniform scale")
    return scale, small


def _separate(a, b):
    y, x, h, w = a
    yy, xx, hh, ww = b
    return y + h + 1 <= yy or yy + hh + 1 <= y or x + w + 1 <= xx or xx + ww + 1 <= x


def _paint(grid, template, anchor, scale):
    for (y, x), color in template.items():
        yy, xx = anchor[0] + scale * y, anchor[1] + scale * x
        grid[yy:yy + scale, xx:xx + scale] = color


def _read(input_grid):
    g = _grid(input_grid)
    h, w = g.shape
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] * 2 > g.size, "field must be a strict majority")
    border = np.zeros(g.shape, dtype=bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    bc = np.bincount(g[border], minlength=10)
    candidates = [c for c in range(10) if c != bg and bc[c] >= 2]
    _require(len(candidates) == 1, "divider color is not identifiable")
    wall_color = candidates[0]
    even = int(g[-1, 0])
    seeds = [c for c in range(10) if bc[c] == 1 and c not in (bg, wall_color)]
    _require(len(seeds) == 2 and even in seeds, "two boundary fill seeds required")
    odd = next(c for c in seeds if c != even)
    _require(counts[even] == counts[odd] == 1, "fill seed colors are reserved")
    info = _geometry(g == wall_color)
    labels, depth = info["labels"], info["depth"]
    oy, ox = map(int, np.argwhere(g == odd)[0])
    _require(depth[int(labels[oy, ox])] % 2 == 1, "alternate fill seed is not in an odd region")
    motif_mask = ~np.isin(g, [bg, wall_color, even, odd])
    _require(not np.any(motif_mask & border), "motifs must be interior")
    _, groups = _components(motif_mask, diagonal=True)
    objects = []
    for group in groups:
        y0, y1, x0, x1 = _bounds(group)
        crop = g[y0:y1 + 1, x0:x1 + 1]
        _require(np.count_nonzero(motif_mask[y0:y1 + 1, x0:x1 + 1]) == len(group), "interleaved motifs")
        _require(np.all(np.isin(crop, [bg] + [int(g[y, x]) for y, x in group])), "nonmotif material inside object box")
        s, small = _primitive(crop)
        mapping = {(int(y), int(x)): int(small[y, x]) for y, x in zip(*np.nonzero(small != bg))}
        objects.append({"box": (y0, x0), "scale": s, "signature": _sig(mapping)})
    signatures = sorted({o["signature"] for o in objects})
    _require(len(signatures) == 2, "exactly two normalized motif states required")
    minimum = min(o["scale"] for o in objects)
    _require({o["signature"] for o in objects if o["scale"] == minimum} == set(signatures), "both states need a common reference scale")
    maps = [{(y, x): c for y, x, c in sig} for sig in signatures]
    shifts = set()
    for color in sorted(set(maps[0].values()) & set(maps[1].values())):
        a = {p for p, c in maps[0].items() if c == color}
        b = {p for p, c in maps[1].items() if c == color}
        ay, _, ax, _ = _bounds(a)
        by, _, bx, _ = _bounds(b)
        shift = ay - by, ax - bx
        if a == {(y + shift[0], x + shift[1]) for y, x in b}:
            shifts.add(shift)
    registrations = []
    for dy, dx in sorted(shifts):
        b = {(y + dy, x + dx): c for (y, x), c in maps[1].items()}
        common = {p: c for p, c in maps[0].items() if b.get(p) == c}
        if len(common) >= 3:
            registrations.append((len(common), (dy, dx), b, common))
    _require(registrations, "no shared scaffold")
    best = max(r[0] for r in registrations)
    registrations = [r for r in registrations if r[0] == best]
    _require(len(registrations) == 1, "tied scaffold registration")
    _, shift, aligned, common = registrations[0]
    cy, _, cx, _ = _bounds(common)
    templates = [{(y - cy, x - cx): c for (y, x), c in t.items()} for t in (maps[0], aligned)]
    scaffold = {(y - cy, x - cx): c for (y, x), c in common.items()}
    union = set(templates[0]) | set(templates[1])
    y0, y1, x0, x1 = _bounds(union)
    rects, occurrences = [], []
    for o in objects:
        state = signatures.index(o["signature"])
        dy, dx = shift if state else (0, 0)
        s = o["scale"]
        anchor = o["box"][0] + (cy - dy) * s, o["box"][1] + (cx - dx) * s
        rect = anchor[0] + y0 * s, anchor[1] + x0 * s, (y1 - y0 + 1) * s, (x1 - x0 + 1) * s
        y, x, rh, rw = rect
        _require(1 <= y and y + rh < h and 1 <= x and x + rw < w, "replacement reaches grid boundary")
        local = labels[y:y + rh, x:x + rw]
        region = int(local[0, 0])
        _require(region >= 0 and np.all(local == region), "replacement spans divider regions")
        _require(all(_separate(rect, r) for r in rects), "replacement rectangles collide")
        rects.append(rect)
        occurrences.append({"anchor": anchor, "scale": s, "state": state, "region": region, "parity": depth[region] % 2, "rect": rect})
    _require({o["parity"] for o in occurrences} == {0, 1}, "both parity classes need motifs")
    info.update({"grid": g, "background": bg, "wall": wall_color, "fills": (even, odd),
                 "templates": templates, "scaffold": scaffold, "occurrences": occurrences})
    return info


def _render(info, treatment="parity"):
    labels, depth = info["labels"], info["depth"]
    out = info["grid"].copy()
    for rid, d in enumerate(depth):
        out[labels == rid] = info["fills"][d % 2]
    for o in info["occurrences"]:
        toggle = o["parity"] if treatment == "parity" else int(treatment == "all")
        _paint(out, info["templates"][o["state"] ^ toggle], o["anchor"], o["scale"])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read all geometry, templates, scales and fill colors from this input only."""
    return _render(_read(input_grid))


def _pick(rng, values):
    values = sorted(values)
    return values[int(rng.integers(len(values)))]


def _design(rng, change):
    h, w = int(rng.integers(3, 5)), int(rng.integers(3, 5))
    cy, cx = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
    style = int(rng.integers(3))
    if style == 0:
        core = {(cy, x) for x in range(w)} | {(y, cx) for y in range(h)}
    elif style == 1:
        core = {(y, 0) for y in range(h)} | {(h - 1, x) for x in range(w)} | {(cy, x) for x in range(cx + 1)}
    else:
        core = {(cy, cx)}
        while len(core) < int((h * w + 5) // 2):
            frontier = {(y + dy, x + dx) for y, x in core for dy, dx in N4
                        if 0 <= y + dy < h and 0 <= x + dx < w} - core
            core.add(_pick(rng, frontier))
    for _ in range(int(rng.integers(4))):
        frontier = {(y + dy, x + dx) for y, x in core for dy, dx in N4
                    if 0 <= y + dy < h and 0 <= x + dx < w} - core
        if frontier:
            core.add(_pick(rng, frontier))
    moving = {(cy, cx)} if change in ("recolor", "combined") else set()
    common = {p: 4 for p in core - moving}
    if rng.random() < .55:
        common[_pick(rng, common)] = 5
    def decoration(side):
        y0, y1, x0, x1 = _bounds(core)
        edge = {p for p in core if (p[0] == (y0 if side == 0 else y1))} if side < 2 else {p for p in core if p[1] == (x0 if side == 2 else x1)}
        y, x = _pick(rng, edge)
        dy, dx = ((-1, 0), (1, 0), (0, -1), (0, 1))[side]
        ly, lx = (0, 1) if side < 2 else (1, 0)
        length = int(rng.integers(1, 4))
        offset = int(rng.integers(length))
        patch = {(y + dy + ly * t, x + dx + lx * t) for t in range(-offset, length - offset)
                 if y0 <= y + ly * t <= y1 and x0 <= x + lx * t <= x1}
        if rng.random() < .25:
            patch.add((y + 2 * dy, x + 2 * dx))
        return patch
    side = int(rng.integers(4))
    a = decoration(side)
    b = a if change == "recolor" else decoration((side + int(rng.integers(1, 4))) % 4)
    maps = []
    for state, patch in enumerate((a, b)):
        mapping = dict(common)
        points = sorted(patch | moving)
        use_two = len(points) >= 2 and rng.random() < .6
        mapping.update({p: 6 + 2 * state + (i % 2 if use_two else 0) for i, p in enumerate(points)})
        maps.append(mapping)
    rotation, reflection = int(rng.integers(4)), bool(rng.integers(2))
    transformed = []
    for mapping in maps:
        t = {}
        for (y, x), c in mapping.items():
            for _ in range(rotation):
                y, x = x, -y
            t[y, -x if reflection else x] = c
        transformed.append(t)
    y0, y1, x0, x1 = _bounds(set(transformed[0]) | set(transformed[1]))
    _require(y1 - y0 < 6 and x1 - x0 < 6, "design too large")
    result = [{(y - y0, x - x0): c for (y, x), c in t.items()} for t in transformed]
    for t in result:
        ay, by, ax, bx = _bounds(t)
        crop = np.zeros((by - ay + 1, bx - ax + 1), dtype=np.int8)
        _paint(crop, t, (-ay, -ax), 1)
        _require(_primitive(crop)[0] == 1, "nonprimitive motif design")
    return result, (y1 - y0 + 1, x1 - x0 + 1)


def _crossing(rng, length, low, high, winding):
    level = int(rng.integers(low, high + 1))
    path = [(level, 0)]
    positions = list(range(2, length - 2, 3))
    rng.shuffle(positions)
    selected = sorted(positions[:int(rng.integers(1, min(4, len(positions)) + 1))]) if winding and high > low else []
    col = 0
    for end in selected + [length - 1]:
        path.extend((level, x) for x in range(col + 1, end + 1))
        col = end
        if end != length - 1:
            nxt = _pick(rng, set(range(low, high + 1)) - {level})
            sign = 1 if nxt > level else -1
            path.extend((y, col) for y in range(level + sign, nxt + sign, sign))
            level = nxt
    return path


def _make_geometry(rng, shape, kind, scale, fallback=False):
    bh, bw = shape
    if kind in ("chain", "through"):
        vertical = bool(rng.integers(2))
        td, ld = (bw, bh) if vertical else (bh, bw)
        max_jitter = min(2, (30 - (4 * td + (4 if kind == "chain" else 3))) // 2)
        jitter = 0 if fallback else int(rng.integers(0, max(0, max_jitter) + 1))
        if kind == "chain":
            minimum = 4 * td + 4 + 2 * jitter
            thick = int(rng.integers(minimum, 31))
            length = int(rng.integers(max(3 * ld + 3, 18), 31))
            first = td + 1 + int(rng.integers(thick - minimum + 1))
            second = first + jitter + 2 * td + 1
            paths = [_crossing(rng, length, first, first + jitter, True),
                     _crossing(rng, length, second, second + jitter, True)]
        else:
            minimum = (scale + 1) * td + 3 + jitter
            thick = int(rng.integers(minimum, 31))
            length = int(rng.integers(max(scale * ld + 2, 2 * ld + 3, 18), 31))
            base = scale * td + 1 + int(rng.integers(thick - minimum + 1))
            paths = [_crossing(rng, length, base, base + jitter, not fallback)]
        wall = np.zeros((thick, length), dtype=bool)
        for path in paths:
            for y, x in path:
                wall[y, x] = True
        if vertical:
            wall = wall.T.copy()
            if kind == "through":
                # Put the spacious side to the right of the lower-left root,
                # so an enlarged test occurrence actually has to exchange.
                wall = np.fliplr(wall).copy()
    else:
        h = int(rng.integers(max(22, 2 * bh + 5), 31))
        w = int(rng.integers(max(22, 2 * bw + 5), 31))
        wall = np.zeros((h, w), dtype=bool)
        if kind in ("star", "corners"):
            corners = [0, 1, 2] if kind == "star" else list(rng.choice(4, int(rng.integers(1, 5)), replace=False))
            for corner in corners:
                ry = int(rng.integers(bh + 1, min((h - 3) // 2, bh + 4) + 1))
                rx = int(rng.integers(bw + 1, min((w - 3) // 2, bw + 4) + 1))
                path = [(ry, x) for x in range(rx + 1)] + [(y, rx) for y in range(ry - 1, -1, -1)]
                for y, x in path:
                    if corner in (1, 2):
                        x = w - 1 - x
                    if corner in (2, 3):
                        y = h - 1 - y
                    wall[y, x] = True
        elif kind == "nested":
            for ry, rx in ((bh + 1, bw + 1), (2 * bh + 3, 2 * bw + 3)):
                for x in range(rx + 1):
                    wall[ry, w - 1 - x] = True
                for y in range(ry):
                    wall[y, w - 1 - rx] = True
        else:
            # A same-side U cut, with no wall junction or closed wall loop.
            a = int(rng.integers(1, 4))
            b = min(h - 3, a + scale * bh + 2)
            d = min(w - bw - 3, scale * bw + 2)
            wall[a, :d + 1] = True
            wall[b, :d + 1] = True
            wall[a:b + 1, d] = True
            if rng.random() < .5:
                wall = np.fliplr(wall).copy()
    return wall, _geometry(wall)


def _slots(labels, depth, rh, rw, parity=None, region=None):
    h, w = labels.shape
    result = []
    if rh + 2 > h or rw + 2 > w:
        return result
    for rid, d in enumerate(depth):
        if (parity is not None and d % 2 != parity) or (region is not None and rid != region):
            continue
        bad = (labels != rid).astype(np.int16)
        integral = np.pad(bad, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        sums = integral[rh:, rw:] - integral[:-rh, rw:] - integral[rh:, :-rw] + integral[:-rh, :-rw]
        valid = sums == 0
        valid[0] = valid[-1] = False
        valid[:, 0] = valid[:, -1] = False
        result.extend((int(y), int(x), rh, rw, rid) for y, x in zip(*np.nonzero(valid)))
    return result


def _pack(rng, info, shape, kind, max_scale):
    labels, depth = info["labels"], info["depth"]
    bh, bw = shape
    flip = int(rng.integers(2))
    if kind == "chain":
        deep = depth.index(2)
        specs = [(1, flip, 0, info["root"]), (1, 1 - flip, 0, deep),
                 (2, flip, 1, None), (1, 1 - flip, 1, None)]
    elif kind == "star":
        odds = [rid for rid, d in enumerate(depth) if d % 2]
        specs = [(1, 0, 0, None), (1, 1, 0, None)]
        specs += [(1, (i + flip) % 2, 1, rid) for i, rid in enumerate(odds)]
    else:
        large_parities = [p for p in (1, 0) if _slots(labels, depth, bh * max_scale, bw * max_scale, p)]
        if max_scale == 3:
            large_parities = [p for p in large_parities if p == 1]
        _require(large_parities, "no enlarged placement")
        p = large_parities[0]
        specs = [(1, 0, 1 - p, None), (1, 1, 1 - p, None), (max_scale, flip, p, None)]
    choices = []
    for s, state, parity, region in specs:
        options = _slots(labels, depth, bh * s, bw * s, parity, region)
        _require(options, "empty motif compartment")
        order = rng.permutation(len(options))[:80]
        choices.append([options[int(i)] for i in order])
    order = sorted(range(len(specs)), key=lambda i: (-specs[i][0] ** 2, len(choices[i])))
    placed = {}
    budget = [2500]
    def search(k):
        if k == len(order):
            return True
        i = order[k]
        for option in choices[i]:
            budget[0] -= 1
            if budget[0] < 0:
                return False
            if all(_separate(option[:4], other[:4]) for other in placed.values()):
                placed[i] = option
                if search(k + 1):
                    return True
                del placed[i]
        return False
    _require(search(0), "bounded motif packing exhausted")
    occurrences = [(placed[i], specs[i][0], specs[i][1]) for i in range(len(specs))]
    for _ in range(int(rng.integers(0, 5))):
        s = int(rng.choice([1, 1, 1, 2]))
        options = _slots(labels, depth, bh * s, bw * s)
        options = [o for o in options if all(_separate(o[:4], r[:4]) for r, _, _ in occurrences)]
        if options:
            occurrences.append((options[int(rng.integers(len(options)))], s, int(rng.integers(2))))
    return occurrences


def _pair(rng, kind, scale, change, palette, fallback=False):
    templates, shape = _design(rng, change)
    wall, info = _make_geometry(rng, shape, kind, scale, fallback)
    occurrences = _pack(rng, info, shape, kind, scale)
    inp = np.zeros(wall.shape, dtype=np.int8)
    inp[wall] = 1
    depth, labels = info["depth"], info["labels"]
    for rect, s, state in occurrences:
        _paint(inp, templates[state], rect[:2], s)
    inp[-1, 0] = 2
    border = np.zeros(wall.shape, dtype=bool)
    border[0] = border[-1] = True
    border[:, 0] = border[:, -1] = True
    odd_boundary = [(int(y), int(x)) for y, x in zip(*np.nonzero(border & ~wall))
                    if depth[int(labels[y, x])] % 2 == 1]
    inp[_pick(rng, odd_boundary)] = 3
    _require(np.count_nonzero(inp == 0) * 2 > inp.size, "insufficient open field")
    out = np.ones(wall.shape, dtype=np.int8)
    for rid, d in enumerate(depth):
        out[labels == rid] = 2 + d % 2
    for rect, s, state in occurrences:
        region = rect[4]
        _paint(out, templates[state ^ (depth[region] % 2)], rect[:2], s)
    # Literal color names are never semantic. Seeds and geometry expose roles.
    inp, out = palette[inp], palette[out]
    parsed = _read(inp)
    _require(np.array_equal(_render(parsed), out), "construction/reader disagreement")
    return {"input": inp.copy(), "output": out.copy()}


def _count(value, name, lo, hi):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)), name + " must be an integer")
    value = int(value)
    _require(lo <= value <= hi, "illegal " + name)
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)) and seed >= 0,
             "seed must be a nonnegative integer")
    seed = int(seed)
    rng = np.random.default_rng(seed)
    # None is an optional convenience for the supplied inspection utility.
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_train is None:
        num_test = _count(num_test, "num_test", 1, 4)
        num_train = int(rng.integers(num_test + 1, 11))
    elif num_test is None:
        num_train = _count(num_train, "num_train", 2, 10)
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    num_train, num_test = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    palette = rng.permutation(10).astype(np.int8)
    fixed_palette = bool(rng.random() < .2)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            for attempt in range(48):
                if split == "train" and index == 0:
                    kind, scale, change = "chain", 2, "combined"
                elif split == "train" and index == 1:
                    kind, scale, change = "star", 1, _pick(rng, ["combined", "recolor", "relocate"])
                else:
                    kind = _pick(rng, ["through", "corners", "nested", "u"])
                    scale = 3 if split == "test" else int(rng.integers(1, 3))
                    change = _pick(rng, ["combined", "recolor", "relocate"])
                fallback = attempt >= 40
                if fallback and not (split == "train" and index < 2):
                    kind = "through"
                try:
                    local_palette = palette if fixed_palette else rng.permutation(10).astype(np.int8)
                    pair = _pair(rng, kind, scale, change, local_palette, fallback)
                    key = pair["input"].shape, pair["input"].tobytes()
                    if key not in seen:
                        seen.add(key)
                        puzzle[split].append(pair)
                        break
                except ValueError:
                    continue
            else:
                raise RuntimeError("bounded pair construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ("train", "test")), "splits must be lists")
    nt, ns = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns, "invalid pair counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            key = inp.shape, inp.tobytes()
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _require(inp.shape == out.shape, "dimensions changed")
            info = _read(inp)
            expected = _render(info)
            _require(np.array_equal(out, expected), "incorrect parity-gated exchange")
            _require(np.array_equal(inp == info["wall"], out == info["wall"]), "divider changed")
            _require(not np.array_equal(_render(info, "none"), out), "B exchange is answer-irrelevant")
            _require(not np.array_equal(_render(info, "all"), out), "A gating is answer-irrelevant")
            for o in info["occurrences"]:
                y, x, h, w = o["rect"]
                if o["parity"] == 0:
                    before = inp[y:y + h, x:x + w]
                    after = out[y:y + h, x:x + w]
                    foreground = ~np.isin(before, [info["background"]])
                    _require(np.array_equal(before[foreground], after[foreground]), "even motif changed")
                for (dy, dx), color in info["scaffold"].items():
                    yy, xx = o["anchor"][0] + dy * o["scale"], o["anchor"][1] + dx * o["scale"]
                    _require(np.all(inp[yy:yy + o["scale"], xx:xx + o["scale"]] == color)
                             and np.all(out[yy:yy + o["scale"], xx:xx + o["scale"]] == color), "scaffold moved or recolored")
            infos[split].append(info)
    first, second = infos["train"][:2]
    _require(any(first["depth"][o["region"]] >= 2 and o["parity"] == 0 for o in first["occurrences"]), "missing depth-two demonstration")
    _require({(o["state"], o["parity"]) for o in first["occurrences"]} == {(0, 0), (1, 0), (0, 1), (1, 1)}, "training omits a conditional swap direction")
    _require(any(o["scale"] >= 2 and o["parity"] for o in first["occurrences"]), "missing enlarged swap demonstration")
    a, b = first["templates"]
    _require(set(a) != set(b) and any(a[p] != b[p] for p in set(a) & set(b)) and _bounds(a) != _bounds(b), "first demonstration must combine recoloring and shifted support")
    labels, depth = second["labels"], second["depth"]
    _require(all(labels[p] >= 0 and depth[int(labels[p])] % 2 for p in ((0, 0), (0, -1), (-1, -1))), "other corner anchors not distinguished")
    _require(second["degree"] >= 3, "missing branching adjacency demonstration")
    _require(all(any(o["scale"] >= 3 and o["parity"] for o in i["occurrences"]) for i in infos["test"]), "missing scale-three exchange test shift")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded, streaming diagnostics: no execution traces or grid files saved."""
    # Hand-constructed regression: neither geometry generation nor template
    # inference is used to build these expected pixels. Top/middle/bottom have
    # depths 2/1/0; both directions swap only in the middle band.
    carrier = {(0, 1): 4, (1, 0): 4, (1, 2): 4, (2, 1): 4}
    a = dict(carrier)
    a.update({(1, 1): 5, (-1, 1): 6})
    b = dict(carrier)
    b.update({(1, 1): 9, (3, 0): 8, (3, 1): 8})
    fixture = np.zeros((27, 25), dtype=np.int8)
    fixture[[6, 19], :] = 1
    for t, anchor, s in ((a, (21, 2), 1), (b, (2, 2), 1), (a, (9, 2), 2), (b, (10, 17), 1)):
        _paint(fixture, t, anchor, s)
    fixture[-1, 0], fixture[7, -1] = 2, 3
    expected = np.full(fixture.shape, 2, dtype=np.int8)
    expected[7:19] = 3
    expected[[6, 19], :] = 1
    for t, anchor, s in ((a, (21, 2), 1), (b, (2, 2), 1), (b, (9, 2), 2), (a, (10, 17), 1)):
        _paint(expected, t, anchor, s)
    _require(np.array_equal(solve(fixture), expected), "hand-constructed regression failed")
    _require(np.all(expected[7:9, 4:6] == 3) and np.all(expected[15:17, 2:6] == 8), "erasure/creation regression failed")
    damaged_seed = fixture.copy()
    damaged_seed[7, -1], damaged_seed[-1, -1] = 0, 3
    missing_state = fixture.copy()
    fi = _read(fixture)
    for o in fi["occurrences"]:
        y, x, h, w = o["rect"]
        missing_state[y:y + h, x:x + w] = 0
        _paint(missing_state, fi["templates"][0], o["anchor"], o["scale"])
    for malformed in (damaged_seed, missing_state):
        try:
            solve(malformed)
        except ValueError:
            continue
        raise AssertionError("invalid input accepted")
    shapes, walls, scaffolds, palettes, counts, scales, changes, hashes = set(), set(), set(), set(), set(), set(), set(), set()
    turns, degrees, endpoint_types, pair_total = set(), set(), set(), 0
    for seed in range(42):
        nt = 2 + seed % 9
        ns = 1 + (seed // 9) % min(4, nt - 1)
        puzzle = generate_puzzle(seed, nt, ns)
        hashes.add(_digest(puzzle))
        if seed < 3:
            _require(_digest(puzzle) == _digest(generate_puzzle(seed, nt, ns)), "nondeterminism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _read(pair["input"])
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "input-only solve failed")
                pair_total += 1
                shapes.add(pair["input"].shape)
                walls.add((pair["input"].shape, (pair["input"] == info["wall"]).tobytes()))
                scaffolds.add(tuple(sorted(info["scaffold"])))
                palettes.add((info["background"], info["wall"], info["fills"]))
                counts.add(len(info["occurrences"]))
                scales.update(o["scale"] for o in info["occurrences"])
                turns.update(info["turns"])
                degrees.add(info["degree"])
                endpoint_types.update(info["endpoints"])
                a, b = info["templates"]
                changes.add((set(a) != set(b), any(a[p] != b[p] for p in set(a) & set(b))))
    _require(len(hashes) == 42 and len(shapes) >= 35 and len(walls) >= 100, "structural diversity failed")
    _require(len(scaffolds) >= 30 and len(palettes) >= 35, "design or role diversity failed")
    _require(len(counts) >= 4 and scales >= {1, 2, 3}, "motif count/scale diversity failed")
    _require(max(turns) >= 4 and max(degrees) >= 3 and len(endpoint_types) >= 4, "topology diversity failed")
    _require(changes >= {(True, True), (True, False), (False, True)}, "state-change diversity failed")
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(6000 + nt * 10 + ns, nt, ns)
            _require((len(p["train"]), len(p["test"])) == (nt, ns), "explicit counts ignored")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, False), (4, "1")):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    sample = generate_puzzle(0)
    for target in ("wall", "scaffold", "odd", "even", "fill"):
        p = {s: [{k: v.copy() for k, v in pair.items()} for pair in sample[s]] for s in sample}
        pair = p["test"][0]
        info = _read(pair["input"])
        if target == "wall":
            point = tuple(np.argwhere(pair["input"] == info["wall"])[0])
        elif target == "fill":
            point = (-1, 0)
        else:
            o = next(o for o in info["occurrences"] if target == "scaffold" or o["parity"] == (target == "odd"))
            if target == "scaffold":
                y, x = min(info["scaffold"])
            else:
                a, b = info["templates"]
                y, x = next(p for p in sorted(set(a) | set(b)) if a.get(p) != b.get(p))
            point = o["anchor"][0] + y * o["scale"], o["anchor"][1] + x * o["scale"]
        pair["output"][point] = (int(pair["output"][point]) + 1) % 10
        try:
            validate_generated_puzzle(p)
        except ValueError:
            continue
        raise AssertionError("corrupt output accepted: " + target)
    return {"seeds": 42, "pairs": pair_total, "distinct_puzzles": len(hashes), "shapes": len(shapes),
            "divider_layouts": len(walls), "scaffold_shapes": len(scaffolds), "role_palettes": len(palettes),
            "motif_counts": sorted(counts), "scales": sorted(scales), "turn_counts": sorted(turns),
            "max_region_degree": max(degrees), "change_modes": sorted(changes)}
