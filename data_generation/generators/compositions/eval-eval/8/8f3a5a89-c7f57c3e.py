"""Shoreline switches: local binary templates are also the physical coastline.

Only NumPy and the standard library are used. No examples, files, seeds or
generation metadata are consulted by solve(). No execution traces are saved.
"""
from __future__ import annotations

import hashlib
from collections import deque
import numpy as np

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer rectangles, size 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _flood(allowed, start):
    reached = np.zeros(allowed.shape, bool)
    if not allowed[start]:
        return reached
    queue = deque([start])
    reached[start] = True
    h, w = allowed.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in N4:
            y, x = r + dr, c + dc
            if 0 <= y < h and 0 <= x < w and allowed[y, x] and not reached[y, x]:
                reached[y, x] = True
                queue.append((y, x))
    return reached


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = N8 if diagonal else N4
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _dilate(mask, diagonal=True):
    h, w = mask.shape
    p = np.pad(mask, 1)
    result = mask.copy()
    for dr, dc in N8 if diagonal else N4:
        result |= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return result


def _spread(allowed, initial):
    """Construction-side flood by synchronous dilation, without the solver BFS."""
    reached = initial & allowed
    for _ in range(allowed.size):
        newer = _dilate(reached, False) & allowed
        if np.array_equal(newer, reached):
            return reached
        reached = newer
    raise AssertionError("finite flood did not converge")


def _regions(land):
    result = []
    for group in _components(~land):
        m = np.zeros(land.shape, bool)
        for p in group:
            m[p] = True
        result.append(m)
    return sorted(result, key=lambda m: (-int(m.sum()), tuple(np.argwhere(m)[0])))


def _outside(selected, bfs=True):
    allowed = np.pad(~selected, 1, constant_values=True)
    if bfs:
        return _flood(allowed, (0, 0))
    start = np.zeros(allowed.shape, bool)
    start[0, 0] = True
    return _spread(allowed, start)


def _bounds(points):
    return min(r for r, c in points), max(r for r, c in points), min(c for r, c in points), max(c for r, c in points)


def _sig(mapping):
    return tuple(sorted((int(r), int(c), int(v)) for (r, c), v in mapping.items()))


def _primitive(crop):
    lengths = []
    for a in (crop, crop.T):
        for row in a:
            cuts = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, len(row)]
            lengths.extend(int(n) for n in np.diff(cuts))
    scale = int(np.gcd.reduce(lengths))
    small = crop[::scale, ::scale]
    assert np.array_equal(np.repeat(np.repeat(small, scale, 0), scale, 1), crop)
    return scale, small


def _separated(a, b):
    r, c, h, w = a
    y, x, hh, ww = b
    return r + h + 1 <= y or y + hh + 1 <= r or c + w + 1 <= x or x + ww + 1 <= c


def _render(grid, mapping, anchor, scale):
    for (r, c), color in mapping.items():
        y, x = anchor[0] + r * scale, anchor[1] + c * scale
        grid[y:y + scale, x:x + scale] = color


def _footprint(shape, points, anchor, scale):
    result = np.zeros(shape, bool)
    for r, c in points:
        y, x = anchor[0] + r * scale, anchor[1] + c * scale
        result[y:y + scale, x:x + scale] = True
    return result


def _read(input_grid):
    """Recover every convention from pixels before constructing any output."""
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    assert np.count_nonzero(grid == bg) * 2 > grid.size, "background must be a strict majority"
    singles = [int(c) for c, n in zip(colors, counts) if n == 1]
    assert len(singles) == 1, "the seed must be the sole singleton color"
    marker = singles[0]
    walls = []
    for c in colors:
        if c in (bg, marker):
            continue
        m = grid == c
        if (m[0].any() and m[-1].any()) or (m[:, 0].any() and m[:, -1].any()):
            walls.append(int(c))
    assert len(walls) == 1, "terrain is the unique color reaching opposite edges"
    wall = walls[0]
    mask = ~np.isin(grid, [bg, marker, wall])
    objects = []
    for group in _components(mask, True):
        r0, r1, c0, c1 = _bounds(group)
        crop = grid[r0:r1 + 1, c0:c1 + 1].copy()
        assert np.count_nonzero(mask[r0:r1 + 1, c0:c1 + 1]) == len(group), "interleaved motifs"
        crop[~mask[r0:r1 + 1, c0:c1 + 1]] = bg
        scale, small = _primitive(crop)
        mapping = {(int(r), int(c)): int(small[r, c]) for r, c in np.argwhere(small != bg)}
        objects.append({"box": (r0, c0), "scale": scale, "raw": _sig(mapping)})
    states = sorted({o["raw"] for o in objects})
    assert len(states) == 2 and len(objects) >= 2, "exactly two visible motif states are needed"
    minimum = min(o["scale"] for o in objects)
    assert {o["raw"] for o in objects if o["scale"] == minimum} == set(states), "both states need a common minimum scale"
    maps = [{(r, c): v for r, c, v in state} for state in states]
    # Whole-color patterns propose registrations; maximal unchanged material
    # chooses the unique shared scaffold rather than an occupied-box corner.
    shifts = set()
    for color in set(maps[0].values()) & set(maps[1].values()):
        a = {p for p, v in maps[0].items() if v == color}
        b = {p for p, v in maps[1].items() if v == color}
        ay, _, ax, _ = _bounds(a)
        by, _, bx, _ = _bounds(b)
        shift = ay - by, ax - bx
        if a == {(r + shift[0], c + shift[1]) for r, c in b}:
            shifts.add(shift)
    candidates = []
    for dy, dx in sorted(shifts):
        aligned = {(r + dy, c + dx): v for (r, c), v in maps[1].items()}
        common = {p for p, v in maps[0].items() if aligned.get(p) == v}
        if len(common) >= 3:
            candidates.append((len(common), (dy, dx), aligned, common))
    assert candidates, "no shared scaffold"
    best = max(v[0] for v in candidates)
    candidates = [v for v in candidates if v[0] == best]
    assert len(candidates) == 1, "ambiguous scaffold alignment"
    _, shift, aligned, common = candidates[0]
    ry, _, cx, _ = _bounds(common)
    templates = [{(r - ry, c - cx): v for (r, c), v in m.items()} for m in (maps[0], aligned)]
    core = {p for p, v in templates[0].items() if templates[1].get(p) == v}
    union = set(templates[0]) | set(templates[1])
    r0, r1, c0, c1 = _bounds(union)
    land = grid == wall
    rects, occs = [], []
    for obj in objects:
        state, s = states.index(obj["raw"]), obj["scale"]
        dy, dx = shift if state else (0, 0)
        anchor = obj["box"][0] + (ry - dy) * s, obj["box"][1] + (cx - dx) * s
        rect = anchor[0] + r0 * s, anchor[1] + c0 * s, (r1 - r0 + 1) * s, (c1 - c0 + 1) * s
        y, x, hh, ww = rect
        assert 0 <= y and 0 <= x and y + hh <= grid.shape[0] and x + ww <= grid.shape[1], "replacement would clip"
        assert all(_separated(rect, other) for other in rects), "motif union rectangles collide"
        support = _footprint(grid.shape, union, anchor, s)
        assert not np.any(support & np.isin(grid, [wall, marker])), "replacement overwrites terrain or seed"
        core_mask = _footprint(grid.shape, core, anchor, s)
        land |= core_mask
        rects.append(rect)
        occs.append({"anchor": anchor, "scale": s, "state": state, "core_mask": core_mask, "rect": rect})
    marker_pos = tuple(int(v) for v in np.argwhere(grid == marker)[0])
    selected = _flood(~land, marker_pos)
    external = _outside(selected)
    near = _dilate(selected)
    for obj in occs:
        obj["active"] = bool(np.any(obj["core_mask"] & near & external[1:-1, 1:-1]))
        obj["hole"] = not np.any(obj["core_mask"] & external[1:-1, 1:-1])
    return {"grid": grid, "bg": bg, "wall": wall, "marker": marker, "marker_pos": marker_pos,
            "templates": templates, "core": core, "union": union, "land": land,
            "selected": selected, "external": external, "objects": occs}


def _exchange(info, flags=None):
    result = info["grid"].copy()
    for i, obj in enumerate(info["objects"]):
        if obj["active"] if flags is None else flags[i]:
            blank = {p: info["bg"] for p in info["union"]}
            _render(result, blank, obj["anchor"], obj["scale"])
            _render(result, info["templates"][1 - obj["state"]], obj["anchor"], obj["scale"])
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Simultaneous, input-only shoreline exchange; never mutates the input."""
    return _exchange(_read(input_grid))


def _design(rng, change, fallback=False, corner=False):
    h = 3 if fallback else int(rng.integers(3, 5))
    core = {(r, 0) for r in range(h)}
    for r in range(1, h):
        core.add((r, 1))
    if not fallback:
        for r in range(h):
            if rng.random() < .45:
                core.add((r, 1))
        if rng.random() < .45:
            r = int(rng.integers(1, h))
            core.add((r, 2))
    if change == "combined":
        # The spine stays continuous; a visible non-land interior accent
        # guarantees that the first demonstration teaches recoloring too.
        core.discard((0, 1))
    if corner:
        # A state annotation occupies the mouth of a one-cell diagonal notch.
        # Treating annotations as solid would hide the socket's only contact.
        core.add((1, 2))
        core.discard((0, 2))
        maps = [{p: 1 for p in core} for _ in range(2)]
        maps[0].update({(h - 2, -1): 2, (h - 1, -1): 2})
        maps[1][0, 3] = 2
        return core, maps
    maxc = max(c for r, c in core)
    # Side patches keep the full height of the physical spine visible.
    left_start = 1 if corner else 0 if fallback else int(rng.integers(h - 1))
    left = {(left_start, -1), (left_start + 1, -1)}
    if not fallback and rng.random() < .45:
        left.add((left_start + 1, -2))
    right_r = max(r for r, c in core if c == maxc)
    right = {(right_r, maxc + 1), (max(0, right_r - 1), maxc + 1)}
    if not fallback and rng.random() < .35:
        right.add((right_r, maxc + 2))
    a = left if fallback or rng.random() < .5 else right
    b = a if change == "recolor" else right if a == left else left
    maps = [{p: 1 for p in core} for _ in range(2)]
    if change == "relocate":
        for state, patch in enumerate((a, b)):
            maps[state].update({p: 2 for p in patch})
    else:
        for state, patch in enumerate((a, b)):
            for j, p in enumerate(sorted(patch)):
                maps[state][p] = 2 + state * 2 + (j % 2 if not fallback and rng.random() < .3 else 0)
        gaps = sorted({(r, c) for r in range(h) for c in range(maxc + 1)} - core)
        if change == "combined" and gaps:
            p = gaps[int(rng.integers(len(gaps)))]
            maps[0][p], maps[1][p] = 2, 4
    return core, maps


def _positions(mask, hh, ww):
    """All all-true rectangles, using a bounded summed-area table."""
    h, w = mask.shape
    if hh > h or ww > w:
        return []
    p = np.pad(mask.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = p[hh:, ww:] - p[:-hh, ww:] - p[hh:, :-ww] + p[:-hh, :-ww]
    return [tuple(int(v) for v in q) for q in np.argwhere(sums == hh * ww)]


def _make_scene(rng, core, templates, scale, branch, selection, fallback=False, corner=False):
    h = 28 if fallback else int(rng.integers(24, 31))
    w = 30 if fallback else int(rng.integers(26, 31))
    union = set(templates[0]) | set(templates[1])
    y0, y1, x0, x1 = _bounds(union)
    lh, lw = y1 - y0 + 1, x1 - x0 + 1
    c = 11 if fallback else int(rng.integers(10, min(13, w // 2)))
    terrain = np.zeros((h, w), bool)
    terrain[:, c] = True
    r = 10 if fallback else int(rng.integers(3, h - lh * scale - 2))
    anchor = (r, c)
    rect = (r + y0 * scale, c + x0 * scale, lh * scale, lw * scale)
    if rect[1] < 1 or rect[1] + rect[3] >= w:
        return None
    terrain[r:r + lh * scale, c] = False
    occs = [{"anchor": anchor, "scale": scale, "state": 0, "rect": rect}]
    land = terrain | _footprint((h, w), core, anchor, scale)
    if branch:
        choices = [y for y in range(7, h - 7) if y < r - 1 or y > r + lh * scale]
        if not choices:
            return None
        row = choices[len(choices) // 2] if fallback else int(rng.choice(choices))
        terrain[row, c:] = True
        land[row, c:] = True
    # Small staircase bulges change the shoreline without a new rule.
    if not fallback:
        for _ in range(int(rng.integers(0, 4))):
            yy = int(rng.integers(h))
            if rect[0] - 1 <= yy <= rect[0] + rect[2]:
                continue
            side = -1 if rng.random() < .5 else 1
            extent = int(rng.integers(1, 4))
            xx = sorted((c, c + side * extent))
            terrain[yy: min(h, yy + int(rng.integers(1, 3))), xx[0]:xx[1] + 1] = True
        land |= terrain
    regions = _regions(land)
    if len(regions) != (3 if branch else 2):
        return None
    if selection == "small":
        target = min(regions, key=lambda m: int(m.sum()))
    elif selection == "large":
        target = regions[0]
    else:
        target = regions[int(rng.integers(len(regions)))]
    start = tuple(int(v) for v in np.argwhere(target)[0])

    def add(kind, region_start, state, s=1):
        nonlocal land, terrain
        region = _flood(~land, region_start)
        hh, ww = lh * s, lw * s
        free = region.copy()
        # A clear union rectangle reserves both states, including empty cells.
        for obj in occs:
            yy, xx, oh, ow = obj["rect"]
            free[max(0, yy - 1):min(h, yy + oh + 1), max(0, xx - 1):min(w, xx + ow + 1)] = False
        if kind in ("island", "corner"):
            free &= ~_dilate(land)
            free[[0, -1], :] = False
            free[:, [0, -1]] = False
        positions = _positions(free, hh, ww)
        rng.shuffle(positions)
        for yy, xx in positions:
            if yy < 1 or yy + hh > h - 1:
                continue
            aa = yy - y0 * s, xx - x0 * s
            connector = np.zeros((h, w), bool)
            if kind in ("peninsula", "corner"):
                sides = [0, 1]
                rng.shuffle(sides)
                found = False
                for side in sides:
                    connector[:] = False
                    if side == 0 and 1 <= yy <= 5:
                        connector[:yy, aa[1]] = True
                    elif side == 1 and 1 <= h - yy - hh <= 5:
                        connector[yy + hh:, aa[1]] = True
                    else:
                        continue
                    if not np.all(region[connector]):
                        continue
                    if any(np.any(connector[max(0, o["rect"][0] - 1):min(h, o["rect"][0] + o["rect"][2] + 1),
                                             max(0, o["rect"][1] - 1):min(w, o["rect"][1] + o["rect"][3] + 1)]) for o in occs):
                        continue
                    found = True
                    break
                if not found:
                    continue
            if kind == "corner":
                # A land mass enclosing the mutable decorations has a notch
                # ending diagonally against the scaffold's side tooth.
                # Its sole selected-region contact is a corner, not an edge.
                if s != 1 or (0, 3) not in union or (1, 2) not in core or yy > 5 or xx < 2 or xx + ww >= w - 1:
                    continue
                surround = np.zeros((h, w), bool)
                surround[yy - 1:yy + hh + 1, xx - 1:xx + ww + 1] = True
                if not np.all(region[surround]) or any(
                    not _separated((yy - 1, xx - 1, hh + 2, ww + 2), o["rect"]) for o in occs
                ):
                    continue
                surround[_footprint((h, w), union, aa, s)] = False
                surround[yy, aa[1] + 3:xx + ww + 1] = False
                # Force the connection above the right part of the land mass;
                # a bottom stem could pass through its enclosed decorations.
                connector[:] = False
                connector[:yy - 1, xx + ww] = True
                if not np.all(region[connector]):
                    continue
                connector |= surround
            trial = land | connector | _footprint((h, w), core, aa, s)
            new_region = _flood(~trial, region_start)
            # Corner sockets deliberately contain small, unselected annotation
            # pockets. Only the external free compartments are counted here.
            ext_regions = [m for m in _regions(trial) if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()]
            if not new_region.any() or len(ext_regions) != len(regions):
                continue
            if kind == "corner":
                footprint = _footprint((h, w), core, aa, s)
                if np.any(footprint & _dilate(new_region, False)) or not np.any(footprint & _dilate(new_region)):
                    continue
            terrain |= connector
            land = trial
            occs.append({"anchor": aa, "scale": s, "state": state, "rect": (yy, xx, hh, ww)})
            return True
        return False

    # These witnesses are physical placements, never hidden solver labels.
    if not add("corner" if corner else "peninsula", start, 1):
        return None
    other_regions = [m for m in _regions(land) if not m[start] and (m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any())]
    other = other_regions[int(rng.integers(len(other_regions)))]
    other_start = tuple(int(v) for v in np.argwhere(other)[0])
    if not add("peninsula", other_start, 1):
        return None
    if not add("island", start, 0):
        return None
    if not fallback:
        for _ in range(int(rng.integers(0, 4))):
            choices = [m for m in _regions(land) if m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()]
            m = choices[int(rng.integers(len(choices)))]
            p = tuple(int(v) for v in np.argwhere(m)[0])
            add(str(rng.choice(["island", "island", "peninsula"])), p, int(rng.integers(2)), 1 if rng.random() < .85 else 2)
        # Detached uncolored land can itself be an island; avoid union boxes.
        for _ in range(int(rng.integers(0, 3))):
            available = ~_dilate(land)
            for obj in occs:
                yy, xx, hh, ww = obj["rect"]
                available[max(0, yy - 1):min(h, yy + hh + 1), max(0, xx - 1):min(w, xx + ww + 1)] = False
            available[[0, -1], :] = False
            available[:, [0, -1]] = False
            pp = np.argwhere(available)
            if len(pp):
                p = tuple(int(v) for v in pp[int(rng.integers(len(pp)))])
                terrain[p] = land[p] = True
    selected = _flood(~land, start)
    reserved = np.zeros((h, w), bool)
    for obj in occs:
        yy, xx, hh, ww = obj["rect"]
        reserved[yy:yy + hh, xx:xx + ww] = True
    free = selected & ~reserved & ~terrain
    rim = _dilate(_outside(selected))[1:-1, 1:-1] & selected
    mode = int(rng.integers(3))
    candidates = np.argwhere(free & (rim if mode == 0 else ~rim if mode == 1 else free))
    if not len(candidates):
        candidates = np.argwhere(free)
    if not len(candidates):
        return None
    marker = tuple(int(v) for v in candidates[int(rng.integers(len(candidates)))])
    return terrain, occs, marker


def _materialize(scene, core, designs, palette, rotation, reflect):
    terrain, occs, marker = scene
    bg, wall, seed_color = palette[:3]
    labels = sorted(set(designs[0].values()) | set(designs[1].values()))
    colors = dict(zip(labels, palette[3:]))
    templates = [{p: colors[v] for p, v in m.items()} for m in designs]
    inp = np.full(terrain.shape, bg, np.int8)
    inp[terrain] = wall
    land = terrain.copy()
    for obj in occs:
        _render(inp, templates[obj["state"]], obj["anchor"], obj["scale"])
        land |= _footprint(inp.shape, core, obj["anchor"], obj["scale"])
    inp[marker] = seed_color
    start = np.zeros(inp.shape, bool)
    start[marker] = True
    selected = _spread(~land, start)
    exterior = _outside(selected, False)[1:-1, 1:-1]
    near = _dilate(selected)
    out = inp.copy()
    union = set(templates[0]) | set(templates[1])
    for obj in occs:
        footprint = _footprint(inp.shape, core, obj["anchor"], obj["scale"])
        if np.any(footprint & exterior & near):
            _render(out, {p: bg for p in union}, obj["anchor"], obj["scale"])
            _render(out, templates[1 - obj["state"]], obj["anchor"], obj["scale"])
    inp, out = np.rot90(inp, rotation), np.rot90(out, rotation)
    if reflect:
        inp, out = np.fliplr(inp), np.fliplr(out)
    return {"input": inp.copy(), "output": out.copy()}


def _fallback_scene(core, designs, branch, selection, corner, ordinal):
    """A constructive four-copy scene, including the teaching notch when needed."""
    shape, c = (28, 30), 11
    terrain = np.zeros(shape, bool)
    terrain[:, c] = True
    terrain[10:16, c] = False
    if branch:
        terrain[18, c:] = True
    union = set(designs[0]) | set(designs[1])
    y0, y1, x0, x1 = _bounds(union)
    lh, lw = y1 - y0 + 1, x1 - x0 + 1
    assert lh == 3
    occs = []

    def occurrence(anchor, scale, state):
        rect = anchor[0] + y0 * scale, anchor[1] + x0 * scale, lh * scale, lw * scale
        occs.append({"anchor": anchor, "scale": scale, "state": state, "rect": rect})

    occurrence((10, c), 2, 0)
    right = selection == "large"
    selected_anchor = (2, 22) if right else (2, 3)
    other_anchor = (2, 3) if right else (2, 23)
    if corner:
        yy, xx = selected_anchor[0] + y0, selected_anchor[1] + x0
        surround = np.zeros(shape, bool)
        surround[yy - 1:yy + lh + 1, xx - 1:xx + lw + 1] = True
        surround[_footprint(shape, union, selected_anchor, 1)] = False
        surround[yy, selected_anchor[1] + 3:xx + lw + 1] = False
        terrain |= surround
        terrain[:yy - 1, xx + lw] = True
    else:
        terrain[:2, selected_anchor[1]] = True
    terrain[:2, other_anchor[1]] = True
    occurrence(selected_anchor, 1, 1)
    occurrence(other_anchor, 1, 1)
    occurrence((19, 22) if right else (19, 4), 1, 0)
    land = terrain.copy()
    reserved = np.zeros(shape, bool)
    for obj in occs:
        land |= _footprint(shape, core, obj["anchor"], obj["scale"])
        r, cc, h, w = obj["rect"]
        reserved[r:r + h, cc:cc + w] = True
    selected = _flood(~land, (16, 29) if right else (16, 0))
    positions = np.argwhere(selected & ~reserved)
    # At least fourteen choices; a distinct ordinal handles repeated fallback
    # designs without an unbounded uniqueness retry or stored examples.
    assert len(positions) > 14
    marker = tuple(int(v) for v in positions[ordinal % len(positions)])
    return terrain, occs, marker


def _count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _key(grid):
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _count("num_train", num_train, 2, 10), _count("num_test", num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    rotation, reflection = int(rng.integers(4)), bool(rng.integers(2))
    stable_palette = [int(v) for v in rng.permutation(10)]
    stable = bool(rng.random() < .2)
    puzzle, seen = {"train": [], "test": []}, set()
    first_corners = None
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            witness = index < 2 and split == "train"
            change = "combined" if index == 0 or split == "test" else "relocate" if index == 1 else str(rng.choice(["combined", "relocate", "recolor"]))
            selection = ("small" if index == 0 else "large") if witness else "any"
            for attempt in range(48):
                fallback = attempt >= 40
                corner = witness and index == 1
                core, design = _design(rng, change, fallback, corner)
                scale = 2 if witness and index == 0 else 1 if witness else int(rng.choice([1, 2, 2, 3]))
                if fallback:
                    scale = 2
                if attempt == 47:
                    scene = _fallback_scene(core, design, split == "test", selection, corner, index if split == "train" else num_train + index)
                else:
                    scene = _make_scene(rng, core, design, scale, split == "test", selection, fallback, corner)
                if scene is None:
                    continue
                palette = stable_palette if stable else [int(v) for v in rng.permutation(10)]
                rot = rotation if witness else int(rng.integers(4))
                pair = _materialize(scene, core, design, palette, rot, reflection)
                try:
                    info = _read(pair["input"])
                    fs = _features(info)
                    if not np.array_equal(_exchange(info), pair["output"]):
                        continue
                    if not (fs["active"] >= 2 and fs["holes"] >= 1 and fs["far"] >= 1 and fs["both_directions"]):
                        continue
                    if any(np.array_equal(pair["output"], alt) for alt in _ablations(info).values()):
                        continue
                    if witness and index == 0 and fs["selected_rank"] != "small":
                        continue
                    if witness and index == 1 and (fs["selected_rank"] != "large" or set(fs["corners"]) & set(first_corners)):
                        continue
                    if _key(pair["input"]) in seen:
                        continue
                except (AssertionError, ValueError):
                    continue
                puzzle[split].append(pair)
                seen.add(_key(pair["input"]))
                if witness and index == 0:
                    first_corners = fs["corners"]
                break
            else:
                raise RuntimeError("bounded shoreline construction exhausted")
    return puzzle


def _features(info):
    os = info["objects"]
    a, b = info["templates"]
    sizes = [int(m.sum()) for m in _regions(info["land"])]
    area = int(info["selected"].sum())
    selected = info["selected"]
    both = {o["state"] for o in os if o["active"]} == {0, 1}
    hole = sum(o["hole"] for o in os)
    far = sum(not o["active"] and not o["hole"] for o in os)
    corners = tuple(i for i, v in enumerate((selected[0, 0], selected[0, -1], selected[-1, 0], selected[-1, -1])) if v)
    external_compartments = sum(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any() for m in _regions(info["land"]))
    return {"shape": tuple(info["grid"].shape), "count": len(os), "active": sum(o["active"] for o in os),
            "holes": hole, "far": far, "both_directions": both, "corners": corners,
            "compartments": len(sizes), "external_compartments": int(external_compartments),
            "corner_only": sum(o["active"] and not np.any(o["core_mask"] & _dilate(selected, False)) for o in os),
            "selected_rank": "small" if area < max(sizes) else "large",
            "scales": tuple(sorted(o["scale"] for o in os)), "roles": (info["bg"], info["wall"], info["marker"]),
            "core": tuple(sorted(info["core"])), "templates": (_sig(a), _sig(b)),
            "recolor": any(a[p] != b[p] for p in set(a) & set(b)), "relocate": set(a) != set(b),
            "bbox_shift": _bounds(a) != _bounds(b)}


def _ablations(info):
    """Actual output counterfactuals, not feature-name checks."""
    os = info["objects"]
    # Ignore A: exchange all copies. Ignore external-boundary/hole distinction:
    # exchange every scaffold touching the selected free region.
    all_flags = [True] * len(os)
    region_flags = [bool(np.any(o["core_mask"] & _dilate(info["selected"]))) for o in os]
    # Ignore B's scaffold as physical land: holes in wall sockets now leak.
    selected = _flood(info["grid"] != info["wall"], info["marker_pos"])
    # Keep scaffold locations as candidate land when evaluating boundary
    # membership, but deliberately ignore their blocking effect on reachability.
    selected &= ~(info["land"] & (info["grid"] != info["wall"]))
    external = _outside(selected)[1:-1, 1:-1]
    no_core = [bool(np.any(o["core_mask"] & external & _dilate(selected))) for o in os]
    return {"toggle_all": _exchange(info, all_flags), "toggle_selected_islands": _exchange(info, region_flags),
            "omit_scaffold_land": _exchange(info, no_core), "omit_exchange": info["grid"].copy()}


def _foreground_ablation(info):
    """Incorrectly treat the mutable annotation pixels as additional solid land."""
    grid = info["grid"]
    selected = _flood(np.isin(grid, [info["bg"], info["marker"]]), info["marker_pos"])
    external = _outside(selected)[1:-1, 1:-1]
    flags = [bool(np.any(o["core_mask"] & external & _dilate(selected))) for o in info["objects"]]
    return _exchange(info, flags)


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, train_infos = set(), []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _read(inp)
            assert np.array_equal(out, _exchange(info)), "incorrect shoreline exchange"
            fs = _features(info)
            assert fs["active"] and fs["holes"] and fs["far"]
            assert np.array_equal(inp[info["land"]], out[info["land"]]), "physical land moved"
            assert out[info["marker_pos"]] == info["marker"]
            # Check unchanged inactive copies and exact enlarged target support.
            for obj in info["objects"]:
                r, c, h, w = obj["rect"]
                if not obj["active"]:
                    assert np.array_equal(inp[r:r + h, c:c + w], out[r:r + h, c:c + w])
            for name, alternative in _ablations(info).items():
                assert not np.array_equal(out, alternative), f"vacuous parent contribution: {name}"
            if split == "train":
                train_infos.append(info)
    fs = [_features(i) for i in train_infos]
    assert any(f["selected_rank"] == "small" for f in fs)
    assert any(f["selected_rank"] == "large" for f in fs)
    assert not set.intersection(*(set(f["corners"]) for f in fs)), "fixed-corner shortcut"
    assert any(f["both_directions"] and max(f["scales"]) > min(f["scales"]) for f in fs)
    assert any(f["bbox_shift"] for f in fs), "registration by box corner was not distinguished"
    assert any(f["recolor"] for f in fs) and any(f["relocate"] for f in fs)
    assert any(f["corner_only"] for f in fs), "corner-inclusive shoreline contact was not demonstrated"
    assert any(not np.array_equal(_exchange(i), _foreground_ablation(i)) for i in train_infos), "annotation transparency was not demonstrated"


def self_test():
    """Bounded, streaming audit: 42 seeds, counts, mutations and equivariance."""
    shapes, cores, designs, schedules, counts, roles, signatures = set(), set(), set(), set(), set(), set(), set()
    pair_count = 0
    fallback = {"train": [], "test": []}
    rng = np.random.default_rng(990)
    for index, (split, change, selection, corner) in enumerate((
            ("train", "combined", "small", False), ("train", "relocate", "large", True),
            ("test", "combined", "any", False))):
        core, design = _design(rng, change, True, corner)
        scene = _fallback_scene(core, design, split == "test", selection, corner, index)
        fallback[split].append(_materialize(scene, core, design, list(range(10)), 0, False))
    validate_generated_puzzle(fallback)
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        local_shapes, local_roles = set(), set()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                copy = inp.copy()
                assert np.array_equal(solve(inp), out) and np.array_equal(inp, copy)
                info = _read(inp)
                f = _features(info)
                shapes.add(f["shape"])
                local_shapes.add(f["shape"])
                cores.add(f["core"])
                # Canonical role renaming avoids counting palette alone as design.
                color_ids, normalized = {}, []
                for template in f["templates"]:
                    normalized.append(tuple((r, c, color_ids.setdefault(v, len(color_ids))) for r, c, v in template))
                designs.add(tuple(normalized))
                schedules.add(f["scales"])
                counts.add(f["count"])
                roles.add(f["roles"])
                local_roles.add(f["roles"])
                digest.update(repr(inp.shape).encode())
                digest.update(inp.tobytes())
                pair_count += 1
        signatures.add(digest.hexdigest())
        assert len(local_shapes) >= 2
    assert len(signatures) == 42
    assert len(shapes) >= 30 and len(cores) >= 20 and len(designs) >= 60
    assert len(schedules) >= 12 and counts >= {4, 5, 6} and len(roles) >= 30
    # Every legal count combination is small and exercised once, not per seed.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(7300 + nt * 10 + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
            validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(123)
    q = generate_puzzle(123)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    pair = p["test"][0]
    for k in range(4):
        for mirror in (False, True):
            a, b = np.rot90(pair["input"], k), np.rot90(pair["output"], k)
            if mirror:
                a, b = np.fliplr(a), np.fliplr(b)
            assert np.array_equal(solve(a), b)
    # Within-compartment seed movement must not change switch decisions.
    info = _read(pair["input"])
    candidates = np.argwhere(info["selected"] & (info["grid"] == info["bg"]))
    for r, c in candidates[:3]:
        moved = info["grid"].copy()
        moved[info["marker_pos"]] = info["bg"]
        moved[r, c] = info["marker"]
        expected = pair["output"].copy()
        expected[info["marker_pos"]], expected[r, c] = info["bg"], info["marker"]
        assert np.array_equal(solve(moved), expected)
    bad = {s: [{k: v.copy() for k, v in a.items()} for a in p[s]] for s in p}
    g = bad["test"][0]["output"]
    g[0, 0] = (int(g[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("damaged output accepted")
    return {"seeds": 42, "pairs": pair_count, "shapes": len(shapes), "scaffolds": len(cores),
            "normalized_designs": len(designs), "scale_schedules": len(schedules), "motif_counts": sorted(counts),
            "role_palettes": len(roles), "independent_semantic_verification": False}
