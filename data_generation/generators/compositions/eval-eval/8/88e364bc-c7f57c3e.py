"""Alternating, scaffold-registered gait inside a direction-keyed enclosure.

The two exterior motifs are stationary state references. An interior motif tries
to exchange state and advance one logical pixel (its scale in canvas pixels).
A stride commits only if the other state's entire swept silhouette fits. A
rejected stride changes neither the anchor nor the state. No generation metadata
is needed by the reader or solver.
"""
from __future__ import annotations

from collections import deque
from math import gcd
import numpy as np


DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
CARDINAL = tuple(d for d in DIRECTIONS if 0 in d)
DIAGONAL = tuple(d for d in DIRECTIONS if 0 not in d)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger grid")
    _require(np.all((0 <= a) & (a <= 9)), "invalid colors")
    return a.astype(int, copy=False)


def _bounds(points):
    rows, cols = zip(*points)
    return min(rows), max(rows), min(cols), max(cols)


def _sig(mapping):
    return tuple(sorted((int(r), int(c), int(v)) for (r, c), v in mapping.items()))


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    components = []
    steps = DIRECTIONS if diagonal else CARDINAL
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, queue = {start}, deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    group.add(p)
                    queue.append(p)
        components.append(group)
    return components


def _primitive(crop):
    block = 0
    for a in (crop, crop.T):
        for row in a:
            starts = [0] + (np.flatnonzero(row[1:] != row[:-1]) + 1).tolist() + [len(row)]
            for x, y in zip(starts, starts[1:]):
                block = gcd(block, y - x)
    small = crop[::block, ::block]
    _require(np.array_equal(np.repeat(np.repeat(small, block, 0), block, 1), crop),
             "nonuniform pixel enlargement")
    return block, small


def _key_candidates(grid, bg):
    keys = []
    for group in _components(grid != bg, diagonal=True):
        r0, r1, c0, c1 = _bounds(group)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 5 or w < 5 or len(group) != h * w:
            continue
        crop = grid[r0:r1 + 1, c0:c1 + 1]
        wall = int(crop[0, 0])
        border = np.r_[crop[0], crop[-1], crop[:, 0], crop[:, -1]]
        others = set(map(int, np.unique(crop))) - {wall}
        if len(others) != 2 or np.any(border != wall):
            continue
        interpretations = []
        for tip in sorted(others):
            trail = next(iter(others - {tip}))
            tips = set(map(tuple, np.argwhere(crop == tip)))
            trails = set(map(tuple, np.argwhere(crop == trail)))
            if not 2 <= len(tips) <= 3:
                continue
            for dr, dc in DIRECTIONS:
                used, lengths = set(), []
                for r, c in sorted(tips):
                    stroke = []
                    p = r - dr, c - dc
                    while p in trails:
                        stroke.append(p)
                        p = p[0] - dr, p[1] - dc
                    if len(stroke) < 2 or used.intersection(stroke):
                        break
                    used.update(stroke)
                    lengths.append(len(stroke))
                if len(lengths) == len(tips) and used == trails:
                    interpretations.append((tip, trail, (dr, dc)))
        if len(interpretations) == 1:
            tip, trail, direction = interpretations[0]
            keys.append(dict(box=(r0, r1, c0, c1), wall=wall, trail=trail,
                             tip=tip, direction=direction, shape=(h, w)))
    _require(len(keys) == 1, "one uniquely readable direction key required")
    return keys[0]


def _align_references(refs):
    """Register two cropped state patterns by unchanged whole-color masks."""
    _require(len(refs) == 2 and all(o["scale"] == 1 for o in refs),
             "exactly two unit-scale exterior reference motifs required")
    signatures = sorted({o["raw"] for o in refs})
    _require(len(signatures) == 2, "references must show different states")
    maps = [{(r, c): v for r, c, v in sig} for sig in signatures]
    shifts = set()
    for color in set(maps[0].values()) & set(maps[1].values()):
        a = {p for p, v in maps[0].items() if v == color}
        b = {p for p, v in maps[1].items() if v == color}
        ar, _, ac, _ = _bounds(a)
        br, _, bc, _ = _bounds(b)
        shift = ar - br, ac - bc
        if a == {(r + shift[0], c + shift[1]) for r, c in b}:
            shifts.add(shift)
    candidates = []
    for dr, dc in sorted(shifts):
        shifted = {(r + dr, c + dc): v for (r, c), v in maps[1].items()}
        same = {p for p, v in maps[0].items() if shifted.get(p) == v}
        if len(same) >= 3:
            candidates.append((len(same), (dr, dc), shifted, same))
    _require(candidates, "no invariant scaffold")
    best = max(c[0] for c in candidates)
    winners = [c for c in candidates if c[0] == best]
    _require(len(winners) == 1, "ambiguous scaffold registration")
    _, shift, aligned, same = winners[0]
    ar, _, ac, _ = _bounds(same)
    templates = [{(r - ar, c - ac): v for (r, c), v in m.items()}
                 for m in (maps[0], aligned)]
    _require(set(templates[0]) != set(templates[1]), "gait states must have different silhouettes")
    return signatures, templates, shift, (ar, ac)


def _read_input(value):
    grid = _grid(value)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg] * 2 > grid.size, "background must occupy a strict majority")
    key = _key_candidates(grid, bg)
    key_mask = np.zeros(grid.shape, dtype=bool)
    r0, r1, c0, c1 = key["box"]
    key_mask[r0:r1 + 1, c0:c1 + 1] = True
    wall = (grid == key["wall"]) & ~key_mask
    _require(np.any(wall), "missing enclosure")
    # Objects do not obstruct region recovery: their cells belong to free space.
    labels = np.full(grid.shape, -1, dtype=int)
    regions = {}
    for group in _components(~wall):
        if any(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1) for r, c in group):
            continue
        label = len(regions)
        regions[label] = group
        for p in group:
            labels[p] = label
    _require(regions, "no enclosed free-space region")
    object_mask = (grid != bg) & ~key_mask & ~wall
    _require(not np.any(object_mask & np.isin(grid, [key["trail"], key["tip"]])),
             "key colors are reserved")
    refs, movers = [], []
    for group in _components(object_mask, diagonal=True):
        r0, r1, c0, c1 = _bounds(group)
        crop = np.where(object_mask[r0:r1 + 1, c0:c1 + 1],
                        grid[r0:r1 + 1, c0:c1 + 1], bg)
        _require(np.count_nonzero(object_mask[r0:r1 + 1, c0:c1 + 1]) == len(group),
                 "interleaved objects")
        scale, small = _primitive(crop)
        raw = _sig({(int(r), int(c)): int(small[r, c])
                    for r, c in np.argwhere(small != bg)})
        object_labels = {int(labels[p]) for p in group}
        _require(len(object_labels) == 1, "motif straddles a wall or enclosure")
        label = object_labels.pop()
        obj = dict(box=(r0, c0), scale=scale, raw=raw, cells=group, region=label)
        (refs if label == -1 else movers).append(obj)
    _require(1 <= len(movers) <= 3, "one to three interior movers required")
    signatures, templates, shift, origin = _align_references(refs)
    for obj in refs + movers:
        _require(obj["raw"] in signatures, "motif is not one of the two reference states")
        state = signatures.index(obj["raw"])
        dr, dc = shift if state else (0, 0)
        s = obj["scale"]
        obj.update(state=state, anchor=(obj["box"][0] + (origin[0] - dr) * s,
                                       obj["box"][1] + (origin[1] - dc) * s))
        _require(1 <= s <= 3, "unsupported scale")
    return dict(grid=grid, background=bg, key=key, wall=wall, regions=regions,
                refs=refs, movers=movers, templates=templates)


def _pixels(template, anchor, scale):
    ar, ac = anchor
    return {(ar + r * scale + dr, ac + c * scale + dc)
            for r, c in template for dr in range(scale) for dc in range(scale)}


def _fit_anchors(template, scale, free):
    offsets = sorted(_pixels(template, (0, 0), scale))
    r0, r1, c0, c1 = _bounds(free)
    dr0, dr1, dc0, dc1 = _bounds(offsets)
    ar0, ar1, ac0, ac1 = r0 - dr0, r1 - dr1, c0 - dc0, c1 - dc1
    if ar0 > ar1 or ac0 > ac1:
        return set()
    mask = np.zeros((r1 + 1, c1 + 1), dtype=bool)
    for p in free:
        mask[p] = True
    fits = np.ones((ar1 - ar0 + 1, ac1 - ac0 + 1), dtype=bool)
    for dr, dc in offsets:
        fits &= mask[ar0 + dr:ar1 + dr + 1, ac0 + dc:ac1 + dc + 1]
        if not np.any(fits):
            return set()
    return {(int(r) + ar0, int(c) + ac0) for r, c in np.argwhere(fits)}


def _walk(templates, anchor, state, scale, direction, free, *, toggle=True, corner=True,
          fits=None, collect=True, stride=None):
    """Return committed poses and the exact proposed silhouette that was refused.

    A tentative state exchange at the origin and every pixel of its translation
    must fit. Diagonal motion also checks both orthogonal intermediate placements
    at each microstep, generalizing A's no-corner-cutting rule to whole objects.
    """
    dr, dc = direction
    stride = scale if stride is None else stride
    if fits is None:
        fits = [_fit_anchors(t, scale, free) for t in templates]
    poses = [(anchor, state)]
    swept = _pixels(templates[state], anchor, scale) if collect else set()
    corner_witness = False
    for _ in range(31):
        next_state = 1 - state if toggle else state
        tentative = templates[next_state]
        if collect:
            swept.update(_pixels(tentative, anchor, scale))
        legal = anchor in fits[next_state]
        for t in range(1, stride + 1):
            p = anchor[0] + t * dr, anchor[1] + t * dc
            if collect:
                swept.update(_pixels(tentative, p, scale))
            if not legal or p not in fits[next_state]:
                legal = False
                break
            if dr and dc and corner:
                p1 = anchor[0] + t * dr, anchor[1] + (t - 1) * dc
                p2 = anchor[0] + (t - 1) * dr, anchor[1] + t * dc
                if collect:
                    swept.update(_pixels(tentative, p1, scale) | _pixels(tentative, p2, scale))
                if p1 not in fits[next_state] or p2 not in fits[next_state]:
                    legal = False
                    corner_witness = True
                    break
        if not legal:
            return dict(poses=poses, anchor=anchor, state=state, steps=len(poses) - 1,
                        swept=swept, corner=corner_witness)
        anchor = anchor[0] + stride * dr, anchor[1] + stride * dc
        state = next_state
        poses.append((anchor, state))
    raise AssertionError("enclosed gait failed to terminate")


def _paint(grid, template, anchor, scale):
    for (r, c), color in template.items():
        rr, cc = anchor[0] + r * scale, anchor[1] + c * scale
        _require(0 <= rr and rr + scale <= grid.shape[0] and
                 0 <= cc and cc + scale <= grid.shape[1], "paint clipped")
        grid[rr:rr + scale, cc:cc + scale] = color


def _analyze(input_grid):
    info = _read_input(input_grid)
    expected = info["grid"].copy()
    for obj in info["movers"]:
        for p in obj["cells"]:
            expected[p] = info["background"]
    used, walks = set(), []
    for obj in info["movers"]:
        free = info["regions"][obj["region"]]
        args = (info["templates"], obj["anchor"], obj["state"], obj["scale"],
                info["key"]["direction"], free)
        walk = _walk(*args)
        # No interaction/order rule is needed: complete proposal sweeps have a
        # background gap from one another, including the last rejected attempt.
        near = {(r + dr, c + dc) for r, c in walk["swept"]
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        _require(used.isdisjoint(near), "interacting mover sweeps")
        used.update(walk["swept"])
        plain = _walk(*args, toggle=False)
        walk["geometry_witness"] = plain["anchor"] != walk["anchor"]
        loose = _walk(*args, corner=False)
        walk["corner_witness"] = loose["anchor"] != walk["anchor"]
        pixel = _walk(*args, stride=1) if obj["scale"] > 1 else walk
        walk["stride_witness"] = (pixel["anchor"], pixel["state"]) != (walk["anchor"], walk["state"])
        _paint(expected, info["templates"][walk["state"]], walk["anchor"], obj["scale"])
        walks.append(walk)
    info.update(expected=expected, walks=walks)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive a complete answer solely from the current input's pixels."""
    return _analyze(input_grid)["expected"].copy()


def _design(rng, colors):
    """Small procedural scaffold and two connected, differently shaped poses."""
    carrier, stable, accent0, accent1, extra0, extra1 = colors
    h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    root = int(rng.integers(h)), int(rng.integers(w))
    core = {root}
    target = int(rng.integers(4, h * w + 1))
    while len(core) < target:
        frontier = sorted({(r + dr, c + dc) for r, c in core for dr, dc in CARDINAL
                           if 0 <= r + dr < h and 0 <= c + dc < w} - core)
        core.add(frontier[int(rng.integers(len(frontier)))])
    shared = {p: carrier for p in core}
    if len(core) >= 5 and rng.random() < .55:
        p = sorted(core)[int(rng.integers(len(core)))]
        shared[p] = stable
    interior = None
    if len(core) >= 6 and rng.random() < .6:
        interior = sorted(core)[int(rng.integers(len(core)))]
        if shared[interior] == stable:
            interior = None
    maps = []
    sides = rng.choice(4, 2, replace=bool(rng.random() < .25))
    for state in (0, 1):
        mapping = dict(shared)
        accent, extra = (accent0, extra0) if state == 0 else (accent1, extra1)
        if interior is not None:
            mapping[interior] = accent
        side = int(sides[state])
        r0, r1, c0, c1 = _bounds(core)
        boundary = sorted(p for p in core if (p[0] == r0 if side == 0 else p[0] == r1
                                             if side == 1 else p[1] == c0 if side == 2 else p[1] == c1))
        r, c = boundary[int(rng.integers(len(boundary)))]
        dr, dc = ((-1, 0), (1, 0), (0, -1), (0, 1))[side]
        depth = int(rng.integers(1, 3))
        for t in range(1, depth + 1):
            mapping[r + t * dr, c + t * dc] = accent if t == 1 or rng.random() < .6 else extra
        if rng.random() < .4:
            lateral = (dc, dr)
            sign = int(rng.choice([-1, 1]))
            mapping[r + dr + sign * lateral[0], c + dc + sign * lateral[1]] = extra
        maps.append(mapping)
    return maps


def _make_key(rng, direction, carrier, trail, tip):
    dr, dc = direction
    count = int(rng.choice([2, 2, 3]))
    length = int(rng.choice([2, 2, 3]))
    points = []
    for i in range(count):
        # The offset is perpendicular for cardinals, horizontal for diagonals.
        base = (i * 2, 0) if dr == 0 else (0, i * 2)
        stagger = int(rng.integers(2)) if dr and dc else 0
        for j in range(length + 1):
            points.append((base[0] + (j + stagger) * dr,
                           base[1] + (j + stagger) * dc, tip if j == length else trail))
    r0, r1, c0, c1 = _bounds([(r, c) for r, c, _ in points])
    key = np.full((r1 - r0 + 3, c1 - c0 + 3), carrier, dtype=int)
    for r, c, v in points:
        key[r - r0 + 1, c - c0 + 1] = v
    return key


def _wall_shape(rng, h, w, complexity):
    footprint = np.ones((h, w), dtype=bool)
    kind = int(rng.integers(3))
    if kind == 1:
        # Stepped bays cut away one to three corners of a filled footprint.
        for side in rng.choice(4, int(rng.integers(1, 4)), replace=False):
            dh, dw = int(rng.integers(1, h // 3)), int(rng.integers(1, w // 3))
            rs = slice(0, dh) if side < 2 else slice(h - dh, h)
            cs = slice(0, dw) if side % 2 == 0 else slice(w - dw, w)
            footprint[rs, cs] = False
    elif kind == 2:
        # Different stepped shores, including diagonal contacts in the wall.
        left, right = 0, 0
        for r in range(h):
            left = int(np.clip(left + rng.integers(-1, 2), 0, w // 4))
            right = int(np.clip(right + rng.integers(-1, 2), 0, w // 4))
            footprint[r, :left] = False
            if right:
                footprint[r, w - right:] = False
    padded = np.pad(footprint, 1)
    inside = footprint.copy()
    for dr, dc in CARDINAL:
        inside &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    wall = footprint & ~inside
    for _ in range(int(rng.integers(complexity + 1))):
        if rng.random() < .5:
            r, stop = int(rng.integers(3, h - 3)), int(rng.integers(3, w - 2))
            side = slice(0, stop) if rng.random() < .5 else slice(stop, w)
            wall[r, side] |= footprint[r, side]
        else:
            c, stop = int(rng.integers(3, w - 3)), int(rng.integers(3, h - 2))
            side = slice(0, stop) if rng.random() < .5 else slice(stop, h)
            wall[side, c] |= footprint[side, c]
    regions = []
    for group in _components(~wall):
        if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in group):
            regions.append(group)
    return wall, regions


def _select_walkers(rng, templates, regions, direction, scale, count, state,
                    parity=None, need_corner=False, need_geometry=True):
    candidates = []
    fit_cache = {(label, s): [_fit_anchors(t, s, free) for t in templates]
                 for label, free in enumerate(regions) for s in sorted({1, scale})}
    for label, free in enumerate(regions):
        fits = fit_cache[label, scale]
        for ar, ac in sorted(fits[state]):
            walk = _walk(templates, (ar, ac), state, scale, direction, free, fits=fits, collect=False)
            if walk["steps"] == 0 or (parity is not None and walk["steps"] % 2 != parity):
                continue
            if need_geometry:
                plain = _walk(templates, (ar, ac), state, scale, direction, free,
                              toggle=False, fits=fits, collect=False)
                if plain["anchor"] == walk["anchor"]:
                    continue
            if need_corner:
                loose = _walk(templates, (ar, ac), state, scale, direction, free,
                              corner=False, fits=fits, collect=False)
                if loose["anchor"] == walk["anchor"]:
                    continue
            if scale > 1:
                pixel = _walk(templates, (ar, ac), state, scale, direction, free,
                              stride=1, fits=fits, collect=False)
                if (pixel["anchor"], pixel["state"]) == (walk["anchor"], walk["state"]):
                    continue
            candidates.append((label, (ar, ac), state, scale, walk))
    if not candidates:
        return None
    first = candidates[int(rng.integers(len(candidates)))]
    label, anchor, st, s, _ = first
    first = (label, anchor, st, s, _walk(templates, anchor, st, s, direction,
                                       regions[label], fits=fit_cache[label, s]))
    selected = [first]
    used = set(first[4]["swept"])
    if count == 1:
        return selected
    # Extra walkers may have a different pose and scale; they share the rule.
    extras = []
    for label, free in enumerate(regions):
        for s in sorted({1, scale}):
            fits = fit_cache[label, s]
            for st in (0, 1):
                for anchor in sorted(fits[st]):
                    walk = _walk(templates, anchor, st, s, direction, free,
                                 fits=fits, collect=False)
                    if walk["steps"]:
                        extras.append((label, anchor, st, s, walk))
    for index in rng.permutation(len(extras)):
        obj = extras[int(index)]
        label, anchor, st, s, walk = obj
        if need_corner and len(selected) == 1 and (st == state or walk["steps"] % 2 != 1):
            continue
        walk = _walk(templates, anchor, st, s, direction, regions[label], fits=fit_cache[label, s])
        obj = (label, anchor, st, s, walk)
        near = {(r + dr, c + dc) for r, c in obj[4]["swept"]
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        if used.isdisjoint(near):
            selected.append(obj)
            used.update(obj[4]["swept"])
        if len(selected) == count:
            break
    if need_corner and len(selected) < 2:
        return None
    return selected


def _make_pair(rng, direction, scale, state, parity, need_corner, test):
    for attempt in range(160):
        palette = [int(c) for c in rng.permutation(10)]
        bg, wall_color, trail, tip = palette[:4]
        templates = _design(rng, palette[4:])
        # A 5-cell union permits scale three with ample room for real strides.
        union = set(templates[0]) | set(templates[1])
        ur0, ur1, uc0, uc1 = _bounds(union)
        if ur1 - ur0 > 4 or uc1 - uc0 > 4:
            continue
        key = _make_key(rng, direction, wall_color, trail, tip)
        ref_sizes = []
        for m in templates:
            r0, r1, c0, c1 = _bounds(m)
            ref_sizes.append((r1 - r0 + 1, c1 - c0 + 1))
        band = max(key.shape[0], *(h for h, w in ref_sizes)) + int(rng.integers(0, 2))
        min_width = key.shape[1] + sum(w for h, w in ref_sizes) + 4
        width = int(rng.integers(max(19, min_width), 31))
        wall_h = int(rng.integers(max(13, 5 * scale + 3), 30 - band))
        wall_w = int(rng.integers(max(15, 5 * scale + 3), width + 1))
        wall, regions = _wall_shape(rng, wall_h, wall_w, 3 if test or need_corner else 2)
        wanted = int(rng.choice([1, 1, 2, 2, 3]))
        if need_corner:
            wanted = max(2, wanted)
        walkers = _select_walkers(rng, templates, regions, direction, scale, wanted,
                                  state, parity, need_corner)
        if not walkers:
            continue
        height = band + 1 + wall_h
        a = np.full((height, width), bg, dtype=int)
        wall_top = band + 1
        wall_left = int(rng.integers(width - wall_w + 1))
        a[wall_top:wall_top + wall_h, wall_left:wall_left + wall_w][wall] = wall_color
        # Permuted, independently offset exterior key and templates. Rotation
        # below supplies every side; exterior placement has no semantic order.
        pieces = [("key", key.shape, None)] + [(str(i), ref_sizes[i], templates[i]) for i in (0, 1)]
        pieces = [pieces[int(i)] for i in rng.permutation(3)]
        widths = sum(size[1] for _, size, _ in pieces)
        slack = width - widths - 4
        gaps = [1, 1, 1, 1]
        for _ in range(slack):
            gaps[int(rng.integers(4))] += 1
        col = gaps[0]
        for index, (name, size, template) in enumerate(pieces):
            top = int(rng.integers(band - size[0] + 1))
            if name == "key":
                a[top:top + size[0], col:col + size[1]] = key
            else:
                r0, _, c0, _ = _bounds(template)
                _paint(a, template, (top - r0, col - c0), 1)
            col += size[1] + gaps[index + 1]
        b = a.copy()
        for _, anchor, st, s, walk in walkers:
            start = anchor[0] + wall_top, anchor[1] + wall_left
            finish = walk["anchor"][0] + wall_top, walk["anchor"][1] + wall_left
            _paint(a, templates[st], start, s)
            _paint(b, templates[walk["state"]], finish, s)
        rotation = int(rng.integers(4))
        a, b = np.rot90(a, rotation), np.rot90(b, rotation)
        if rng.random() < .5:
            a, b = np.fliplr(a), np.fliplr(b)
        pair = dict(input=a.copy(), output=b.copy())
        try:
            info = _analyze(pair["input"])
        except AssertionError:
            continue
        _require(np.array_equal(info["expected"], pair["output"]), "construction/reader mismatch")
        if not all(w["steps"] > 0 for w in info["walks"]):
            continue
        return pair
    raise AssertionError("bounded gait construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    # Two complementary demonstrations suffice even at the minimum train count:
    # an odd enlarged stride sequence and an even diagonal corner witness.
    plans = [(CARDINAL[int(rng.integers(4))], 2, 0, 1, False, False),
             (DIAGONAL[int(rng.integers(4))], 1, 1, 0, True, False)]
    if rng.random() < .5:
        plans.reverse()
    for _ in range(num_train - 2):
        plans.append((DIRECTIONS[int(rng.integers(8))], int(rng.choice([1, 1, 2])),
                      int(rng.integers(2)), None, False, False))
    for i in range(num_test):
        # First query composes a new design and enclosure with scale three.
        plans.append((DIRECTIONS[int(rng.integers(8))], 3 if i == 0 else int(rng.integers(1, 4)),
                      int(rng.integers(2)), None, False, True))
    for index, plan in enumerate(plans):
        for _ in range(8):
            pair = _make_pair(rng, *plan)
            fp = _fingerprint(pair["input"])
            if fp not in seen:
                seen.add(fp)
                break
        else:
            raise AssertionError("duplicate input construction exhausted")
        puzzle["train" if index < num_train else "test"].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "illegal counts")
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape, "dimensions changed")
            fp = _fingerprint(a)
            _require(fp not in seen, "duplicate input")
            seen.add(fp)
            info = _analyze(a)
            _require(np.array_equal(info["expected"], b), "incorrect gait output")
            _require(all(w["steps"] > 0 for w in info["walks"]), "generated movers must advance")
            _require(any(w["geometry_witness"] for w in info["walks"]), "state shape must affect stopping")
            _require(np.array_equal(a[info["wall"]], b[info["wall"]]), "walls changed")
            for ref in info["refs"]:
                _require(all(a[p] == b[p] for p in ref["cells"]), "reference changed")
            kr0, kr1, kc0, kc1 = info["key"]["box"]
            _require(np.array_equal(a[kr0:kr1 + 1, kc0:kc1 + 1], b[kr0:kr1 + 1, kc0:kc1 + 1]),
                     "direction legend changed")
            # Carrier pixel count is conserved while accent counts may change.
            common = {p: v for p, v in info["templates"][0].items()
                      if info["templates"][1].get(p) == v}
            shared_colors = set(common.values()) - (
                {v for p, v in info["templates"][0].items() if p not in common} |
                {v for p, v in info["templates"][1].items() if p not in common})
            for color in shared_colors:
                _require(np.count_nonzero(a == color) == np.count_nonzero(b == color), "scaffold lost")
            infos[split].append(info)
    training = infos["train"]
    _require({w["steps"] % 2 for i in training for w in i["walks"]} == {0, 1},
             "demonstrations must expose rollback and stride parity")
    _require(any({o["state"] for o in i["movers"]} == {0, 1} for i in training),
             "demonstrations must expose both initial states")
    _require(any(o["scale"] == 2 for i in training for o in i["movers"]), "missing scaled gait demonstration")
    _require(any(w["stride_witness"] for i in training for w in i["walks"]), "missing scale-relative stride witness")
    _require(any(w["corner_witness"] for i in training for w in i["walks"]), "missing diagonal corner evidence")
    _require(any(0 in i["key"]["direction"] for i in training), "missing cardinal evidence")
    return None


def self_test():
    """Bounded structural, semantic-ablation, corruption and 42-seed checks."""
    # Hand-derived endpoints, separate from random construction, pin down the
    # prospective silhouette, rollback, parity, scaled stride and corner rule.
    common = {(0, 0): 4, (1, 0): 4, (1, 1): 4}
    templates = [dict(common, **{}), dict(common, **{})]
    templates[0][0, 1] = 6
    templates[1][0, 2] = 7
    free = {(r, c) for r in range(1, 6) for c in range(1, 11)}
    even_walk = _walk(templates, (2, 2), 0, 1, (0, 1), free)
    _require((even_walk["steps"], even_walk["anchor"], even_walk["state"]) == (6, (2, 8), 0),
             "even-gait fixture failed")
    odd_walk = _walk(templates, (2, 2), 1, 1, (0, 1), free)
    _require((odd_walk["steps"], odd_walk["anchor"], odd_walk["state"]) == (7, (2, 9), 0),
             "odd-gait/rollback fixture failed")
    held = _walk(templates, (2, 9), 0, 1, (0, -1), free)
    _require((held["steps"], held["anchor"], held["state"]) == (0, (2, 9), 0),
             "tentative-origin clearance fixture failed")
    for scale, edge, endpoint in ((2, 17, 12), (3, 25, 17)):
        free_scaled = {(r, c) for r in range(1, 10) for c in range(1, edge + 1)}
        walk = _walk(templates, (2, 2), 1, scale, (0, 1), free_scaled)
        _require((walk["steps"], walk["anchor"], walk["state"]) == (5, (2, endpoint), 0),
                 "scale-relative gait fixture failed")
    free_corner = {(r, c) for r in range(1, 9) for c in range(1, 11)} - {(4, 2)}
    blocked = _walk(templates, (2, 2), 0, 1, (1, 1), free_corner)
    loose = _walk(templates, (2, 2), 0, 1, (1, 1), free_corner, corner=False)
    _require(blocked["steps"] == 0 and blocked["corner"] and loose["steps"] > 0,
             "whole-silhouette diagonal corner fixture failed")
    puzzles, shapes, designs, walls, directions, scales, counts = set(), set(), set(), set(), set(), set(), set()
    odd = even = corner = geometry = stride = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        key = tuple(_fingerprint(pair[k]) for split in ("train", "test")
                    for pair in p[split] for k in ("input", "output"))
        puzzles.add(key)
        if seed in (0, 21, 41):
            q = generate_puzzle(seed)
            _require(key == tuple(_fingerprint(pair[k]) for split in ("train", "test")
                                 for pair in q[split] for k in ("input", "output")), "nondeterminism")
        for split in ("train", "test"):
            for pair in p[split]:
                info = _analyze(pair["input"])
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "input-only solver mismatch")
                shapes.add(pair["input"].shape)
                directions.add(info["key"]["direction"])
                counts.add(len(info["movers"]))
                scales.update(o["scale"] for o in info["movers"])
                walls.add(_fingerprint(info["wall"].astype(int)))
                # Color-normalized, scaffold-registered designs: geometry and
                # attachment variation cannot be passed by palette changes.
                roles = {}
                designs.add(tuple(tuple((r, c, roles.setdefault(v, len(roles))) for r, c, v in _sig(t))
                                  for t in info["templates"]))
                for walk in info["walks"]:
                    odd += walk["steps"] % 2 == 1
                    even += walk["steps"] % 2 == 0
                    corner += walk["corner_witness"]
                    geometry += walk["geometry_witness"]
                    stride += walk["stride_witness"]
    _require(len(puzzles) == 42 and len(shapes) >= 35, "insufficient puzzle/canvas diversity")
    _require(len(designs) >= 65 and len(walls) >= 100, "insufficient geometric variation")
    _require(directions == set(DIRECTIONS) and scales == {1, 2, 3}, "direction/scale coverage")
    _require(counts >= {1, 2} and min(odd, even, corner, geometry, stride) >= 42, "missing gait witnesses")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(17, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "explicit counts failed")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p = generate_puzzle(0)
    for mutation in ("identity", "wall", "reference"):
        bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad["test"][0]
        if mutation == "identity":
            pair["output"] = pair["input"].copy()
        else:
            info = _read_input(pair["input"])
            point = tuple(np.argwhere(info["wall"])[0]) if mutation == "wall" else min(info["refs"][0]["cells"])
            pair["output"][point] = (int(pair["output"][point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted output accepted")
    return {"seeds": 42, "distinct_puzzles": len(puzzles), "canvas_shapes": len(shapes),
            "normalized_designs": len(designs), "wall_layouts": len(walls),
            "directions": len(directions), "scales": sorted(scales), "mover_counts": sorted(counts),
            "odd_walks": odd, "even_walks": even, "corner_witnesses": corner,
            "state_dependent_endpoints": geometry, "scale_relative_stride_witnesses": stride}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
