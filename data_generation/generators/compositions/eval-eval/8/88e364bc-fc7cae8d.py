"""Docking stamps: side-calibrated silhouettes stop a direction-driven motion.

All interpretation is reconstructed from the input. Generation uses separate
binary bay masks; solve uses enclosed components and visibly matched guide bars.
"""
from __future__ import annotations

import numpy as np

SIDES = ("T", "R", "B", "L")
NORMAL = {"T": (-1, 0), "R": (0, 1), "B": (1, 0), "L": (0, -1)}
CARDINAL = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIRECTIONS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
MATRICES = tuple(np.array(m, dtype=int) for m in (
    ((1, 0), (0, 1)), ((0, -1), (1, 0)), ((-1, 0), (0, -1)),
    ((0, 1), (-1, 0)), ((1, 0), (0, -1)), ((-1, 0), (0, 1)),
    ((0, 1), (1, 0)), ((0, -1), (-1, 0))))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "noninteger cells")
    _require(np.all((0 <= a) & (a <= 9)), "bad colors")
    return a.astype(int, copy=False)


def _fingerprint(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _components(mask):
    seen = ~mask.copy()
    result = []
    h, w = mask.shape
    for rr, cc in np.argwhere(mask):
        start = int(rr), int(cc)
        if seen[start]:
            continue
        seen[start] = True
        cells, stack = [], [start]
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in CARDINAL:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        result.append(cells)
    return result


def _bbox(cells):
    rows, cols = zip(*cells)
    return tuple(map(int, (min(rows), min(cols), max(rows) + 1, max(cols) + 1)))


def _side(a, s):
    return {"T": a[0], "B": a[-1], "L": a[:, 0], "R": a[:, -1]}[s]


def _adjacent(sides):
    return len(sides) == 2 and sum(abs(x) for x in
        (np.array(NORMAL[sides[0]]) + NORMAL[sides[1]])) == 2


def _key_decode(a, bg):
    interior = a != bg
    interior = interior.copy()
    interior[[0, -1], :] = False
    interior[:, [0, -1]] = False
    keys = []
    for cells in _components(interior):
        r, c, r1, c1 = _bbox(cells)
        sub = a[r:r1, c:c1]
        if min(sub.shape) < 4 or len(cells) != sub.size:
            continue
        carrier = int(sub[0, 0])
        if any(np.any(_side(sub, s) != carrier) for s in SIDES):
            continue
        others = set(map(int, np.unique(sub))) - {carrier}
        if len(others) != 2:
            continue
        readings = []
        for tip in sorted(others):
            trail = next(iter(others - {tip}))
            tips = set(map(tuple, np.argwhere(sub == tip)))
            trails = set(map(tuple, np.argwhere(sub == trail)))
            if len(tips) < 2:
                continue
            for d in DIRECTIONS:
                used, lengths = set(), []
                for tr, tc in sorted(tips):
                    stroke = []
                    tr, tc = tr - d[0], tc - d[1]
                    while (tr, tc) in trails:
                        stroke.append((tr, tc))
                        tr, tc = tr - d[0], tc - d[1]
                    if len(stroke) < 2 or used.intersection(stroke):
                        break
                    used.update(stroke)
                    lengths.append(len(stroke))
                if len(lengths) == len(tips) and used == trails:
                    readings.append((d, trail, tip, tuple(lengths)))
        if len(readings) == 1:
            d, trail, tip, lengths = readings[0]
            keys.append(dict(bounds=(r, c, r1, c1), carrier=carrier,
                             direction=d, trail=trail, tip=tip, lengths=lengths))
    _require(len(keys) == 1, "exactly one uniquely readable direction key required")
    return keys[0]


def _guides_decode(a, controls):
    bars = []
    for color in sorted(controls.values()):
        mask = a == color
        mask = mask.copy()
        mask[[0, -1], :] = False
        mask[:, [0, -1]] = False
        components = _components(mask)
        _require(len(components) == 1, "one interior guide per identity required")
        cells = components[0]
        r, c, r1, c1 = _bbox(cells)
        _require(min(r1 - r, c1 - c) == 1 and len(cells) >= 3,
                 "guides must be straight complete bars")
        bars.append((r, c, r1, c1, color))
    horizontal = [b for b in bars if b[2] - b[0] == 1]
    vertical = [b for b in bars if b[3] - b[1] == 1]
    _require(len(horizontal) == len(vertical) == 1, "guides must be perpendicular")
    hr, hc, _, hc1, hcolor = horizontal[0]
    vr, vc, vr1, _, vcolor = vertical[0]
    _require(hr in (vr - 1, vr1) and vc in (hc - 1, hc1), "misaligned guides")
    guides = {"T" if hr == vr - 1 else "B": hcolor,
              "L" if vc == hc - 1 else "R": vcolor}
    bounds = (vr, hc, vr1, hc1)
    _require(vr >= 3 and hc >= 3 and vr1 <= a.shape[0] - 3
             and hc1 <= a.shape[1] - 3, "card needs a moat from the frame")
    matrix = np.zeros((2, 2), dtype=int)
    for s, color in guides.items():
        target = next(t for t, value in controls.items() if value == color)
        matrix += np.outer(NORMAL[target], NORMAL[s])
    _require(any(np.array_equal(matrix, m) for m in MATRICES), "invalid correspondence")
    return bounds, guides, matrix


def _offsets(card, bg, anchor, matrix):
    pivot = np.argwhere(card == anchor)
    _require(len(pivot) == 1, "stencil must contain one anchor")
    p = pivot[0]
    return [(tuple(map(int, matrix @ (cell - p))), int(card[tuple(cell)]))
            for cell in np.argwhere(card != bg)]


def _rectangle_offsets(card, anchor, matrix):
    pivot = np.argwhere(card == anchor)[0]
    return [(tuple(map(int, matrix @ (np.array(p) - pivot))), 0)
            for p in np.ndindex(card.shape)]


def _rectangle_witness(info, paths):
    solid = _valid_anchors(info["free"], _rectangle_offsets(
        info["card"], info["anchor"], info["matrix"]))
    d = info["key"]["direction"]
    return any(not solid[path[-1]] or (solid[p] and path != _trace(solid, p, d))
               for p, path in zip(info["starts"], paths))


def _valid_anchors(free, offsets):
    """Erode free space by the exact occupied silhouette, including its anchor."""
    h, w = free.shape
    valid = np.ones(free.shape, dtype=bool)
    for (dr, dc), _ in offsets:
        shifted = np.zeros(free.shape, dtype=bool)
        r0, r1 = max(0, -dr), min(h, h - dr)
        c0, c1 = max(0, -dc), min(w, w - dc)
        if r0 < r1 and c0 < c1:
            shifted[r0:r1, c0:c1] = free[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        valid &= shifted
    return valid


def _trace(valid, start, d, corner=True):
    r, c = start
    dr, dc = d
    path = [(r, c)]
    h, w = valid.shape
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w) or not valid[nr, nc]:
            break
        if corner and dr and dc and (not valid[nr, c] or not valid[r, nc]):
            break
        r, c = nr, nc
        path.append((r, c))
    return path


def _swept(path, offsets, d):
    centers = set(path)
    if d[0] and d[1]:
        for r, c in path[:-1]:
            centers.add((r + d[0], c))
            centers.add((r, c + d[1]))
    return {(r + dr, c + dc) for r, c in centers for (dr, dc), _ in offsets}


def _decode(a):
    a = _grid(a)
    frequencies = np.bincount(a.ravel(), minlength=10)
    _require(np.count_nonzero(frequencies == frequencies.max()) == 1, "background frequency tie")
    bg = int(frequencies.argmax())
    controls = {}
    for s in SIDES:
        core = _side(a, s)[1:-1]
        _require(len(core) >= 3 and np.all(core == core[0]), "unclear outer reference side")
        if int(core[0]) != bg:
            controls[s] = int(core[0])
    _require(_adjacent(list(controls)) and len(set(controls.values())) == 2,
             "two adjacent distinct controls required")
    _require(all(a[p] == bg for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))),
             "frame corners must be background")
    key = _key_decode(a, bg)
    bounds, guides, matrix = _guides_decode(a, controls)
    r, c, r1, c1 = bounds
    card = a[r:r1, c:c1].copy()
    excluded = np.zeros(a.shape, dtype=bool)
    excluded[[0, -1], :] = True
    excluded[:, [0, -1]] = True
    excluded[r - 1:r1 + 1, c - 1:c1 + 1] = True
    kr, kc, kr1, kc1 = key["bounds"]
    excluded[kr:kr1, kc:kc1] = True
    wall = key["carrier"]
    remaining = set(map(int, np.unique(a[~excluded]))) - {bg, wall}
    _require(len(remaining) == 1, "launch anchor role must be visible outside the card")
    anchor = next(iter(remaining))
    reserved = {bg, wall, anchor, key["trail"], key["tip"], *controls.values()}
    _require(len(reserved) == 7, "local semantic roles must be distinct")
    ink = set(map(int, np.unique(card))) - {bg, anchor}
    _require(1 <= len(ink) <= 2 and not ink.intersection(reserved), "invalid stencil ink")
    _require(len(_components(card != bg)) == 1 and np.count_nonzero(card != bg) >= 5,
             "stencil must be a connected silhouette")
    offsets = _offsets(card, bg, anchor, matrix)
    # No guide/card/key cell may intrude into the bay. Closed free-space regions
    # are associated by their boundary color, not by a rectangular bounding box.
    free = np.zeros(a.shape, dtype=bool)
    regions, starts = [], []
    for cells in _components(((a == bg) | (a == anchor)) & ~excluded):
        edge, boundary = False, set()
        for rr, cc in cells:
            for dr, dc in CARDINAL:
                nr, nc = rr + dr, cc + dc
                if not (0 <= nr < a.shape[0] and 0 <= nc < a.shape[1]):
                    edge = True
                elif excluded[nr, nc]:
                    edge = True
                elif a[nr, nc] not in (bg, anchor):
                    boundary.add(int(a[nr, nc]))
        local_starts = [p for p in cells if a[p] == anchor]
        if local_starts:
            _require(not edge and boundary == {wall}, "launch point lacks a closed matching bay")
        if not edge and boundary == {wall}:
            regions.append(cells)
            for p in cells:
                free[p] = True
            starts.extend(local_starts)
    _require(1 <= len(starts) <= 4, "one to four launch points required")
    _require(len(starts) == np.count_nonzero((a == anchor) & ~excluded), "unassociated launch point")
    # Explicitly check all visible ink/key colors outside their allowed objects.
    allowed = {bg, wall, anchor}
    _require(set(map(int, np.unique(a[~excluded]))) <= allowed, "unexplained bay cells")
    for s in SIDES:
        segment = {"T": a[r - 1, c:c1], "B": a[r1, c:c1],
                   "L": a[r:r1, c - 1], "R": a[r:r1, c1]}[s]
        _require(np.all(segment == guides.get(s, bg)), "card guides incomplete")
    valid = _valid_anchors(free, offsets)
    _require(all(valid[p] for p in starts), "initial oriented footprint does not fit")
    return dict(background=bg, anchor=anchor, wall=wall, key=key, controls=controls,
                guides=guides, matrix=matrix, card=card, bounds=bounds,
                offsets=offsets, starts=sorted(starts), free=free, valid=valid,
                regions=regions, ink=ink)


def _prediction(a, info, *, point_only=False, corner=True, matrix=None, reverse=False):
    offsets = info["offsets"] if matrix is None else _offsets(
        info["card"], info["background"], info["anchor"], matrix)
    valid = info["free"] if point_only else _valid_anchors(info["free"], offsets)
    d = info["key"]["direction"]
    if reverse:
        d = (-d[0], -d[1])
    out = a.copy()
    for p in info["starts"]:
        out[p] = info["background"]
    paths = []
    for p in info["starts"]:
        path = _trace(valid, p, d, corner)
        paths.append(path)
        er, ec = path[-1]
        for (dr, dc), color in offsets:
            rr, cc = er + dr, ec + dc
            # Ablations can deliberately produce an illegal protrusion. Keeping
            # only in-grid cells makes their differing prediction inspectable.
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = color
    return out, paths


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return docking copies using only visible input geometry and identities."""
    a = _grid(input_grid)
    return _prediction(a, _decode(a))[0]


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _make_key(rng, d, palette, compact=False):
    wall, trail, tip = palette[1:4]
    length = 2 if compact else int(rng.integers(2, 4))
    points = []
    for k in range(2):
        base = (0, 2 * k) if d[0] else (2 * k, 0)
        stagger = 0 if compact else int(rng.integers(0, 2))
        for j in range(length + 1):
            points.append((base[0] + (j + stagger) * d[0],
                           base[1] + (j + stagger) * d[1], tip if j == length else trail))
    r0, c0, r1, c1 = _bbox([(r, c) for r, c, _ in points])
    pad = [1] * 4 if compact else list(map(int, rng.integers(1, 3, size=4)))
    a = np.full((r1 - r0 + pad[0] + pad[1], c1 - c0 + pad[2] + pad[3]), wall, dtype=int)
    for r, c, color in points:
        a[r - r0 + pad[0], c - c0 + pad[2]] = color
    return a


def _make_card(rng, palette):
    bg, anchor = palette[0], palette[4]
    for _ in range(60):
        h, w = map(int, rng.integers(3, 6, size=2))
        mask = np.zeros((h, w), dtype=bool)
        mask[int(rng.integers(h)), int(rng.integers(w))] = True
        desired = int(rng.integers(5, min(h * w, 16) + 1))
        for _ in range(100):
            if np.count_nonzero(mask) >= desired:
                break
            positions = list(map(tuple, np.argwhere(mask)))
            r, c = _pick(rng, positions)
            dr, dc = _pick(rng, CARDINAL)
            if 0 <= r + dr < h and 0 <= c + dc < w:
                mask[r + dr, c + dc] = True
        if np.count_nonzero(mask) < desired:
            continue
        a = np.full((h, w), bg, dtype=int)
        a[mask] = palette[7]
        cells = list(map(tuple, np.argwhere(mask)))
        if rng.random() < .5:
            for i in rng.permutation(len(cells))[:int(rng.integers(1, min(5, len(cells) - 1)))]:
                a[cells[int(i)]] = palette[8]
        a[_pick(rng, cells)] = anchor
        signatures = {tuple(sorted((p, color) for p, color in _offsets(a, bg, anchor, m)))
                      for m in MATRICES}
        if len(signatures) == 8:
            return a
    raise AssertionError("could not make an asymmetric stencil")


def _make_bay(rng):
    h, w = map(int, rng.integers(12, 19, size=2))
    footprint = np.ones((h, w), dtype=bool)
    kind = int(rng.integers(4))
    if kind == 1:
        for corner in rng.choice(4, int(rng.integers(1, 4)), replace=False):
            dh, dw = int(rng.integers(1, h // 3 + 1)), int(rng.integers(1, w // 3 + 1))
            rs = slice(0, dh) if corner < 2 else slice(h - dh, h)
            cs = slice(0, dw) if corner % 2 == 0 else slice(w - dw, w)
            footprint[rs, cs] = False
    elif kind == 2:
        yy, xx = np.indices((h, w))
        footprint = (abs((yy - (h - 1) / 2) / (h / 2)) ** 1.5
                     + abs((xx - (w - 1) / 2) / (w / 2)) ** 1.5 < 1.35)
    elif kind == 3:
        left, right = 0, 0
        for r in range(h):
            left = int(np.clip(left + rng.integers(-1, 2), 0, w // 4))
            right = int(np.clip(right + rng.integers(-1, 2), 0, w // 4))
            footprint[r, :left] = False
            if right:
                footprint[r, -right:] = False
    pad = np.pad(footprint, 1)
    free = footprint.copy()
    for dr, dc in CARDINAL:
        free &= pad[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    wall = footprint & ~free
    # A connected finger is useful geometry: its exposed tip can block a swept
    # diagonal silhouette despite a clear next diagonal footprint.
    for _ in range(int(rng.integers(0, 4))):
        if rng.random() < .5:
            c = int(rng.integers(3, w - 3))
            end = int(rng.integers(3, h - 3))
            sl = slice(0, end) if rng.random() < .5 else slice(end, h)
            wall[sl, c] |= footprint[sl, c]
        else:
            r = int(rng.integers(3, h - 3))
            end = int(rng.integers(3, w - 3))
            sl = slice(0, end) if rng.random() < .5 else slice(end, w)
            wall[r, sl] |= footprint[r, sl]
    return wall, footprint & ~wall


def _pack(rng, shapes):
    for attempt in range(24):
        H, W = (30, 30) if attempt >= 18 else tuple(map(int, rng.integers(22, 31, size=2)))
        occupied = np.zeros((H, W), dtype=bool)
        positions = []
        for h, w in shapes:
            candidates = []
            for r in range(2, H - h - 1):
                for c in range(2, W - w - 1):
                    if not occupied[r - 1:r + h + 1, c - 1:c + w + 1].any():
                        candidates.append((r, c))
            if not candidates:
                break
            r, c = _pick(rng, candidates)
            occupied[r:r + h, c:c + w] = True
            positions.append((r, c))
        if len(positions) == len(shapes):
            return (H, W), positions
    return None


def _paint_scene(rng, wall, card, key, starts, palette, inner, outer, packing):
    shape, positions = packing
    (br, bc), (cr, cc), (kr, kc) = positions
    a = np.full(shape, palette[0], dtype=int)
    h, w = wall.shape
    a[br:br + h, bc:bc + w][wall] = palette[1]
    for r, c in starts:
        a[br + r, bc + c] = palette[4]
    ch, cw = card.shape
    a[cr + 1:cr + ch + 1, cc + 1:cc + cw + 1] = card
    for s, color in zip(inner, palette[5:7]):
        if s == "T": a[cr, cc + 1:cc + cw + 1] = color
        elif s == "B": a[cr + ch + 1, cc + 1:cc + cw + 1] = color
        elif s == "L": a[cr + 1:cr + ch + 1, cc] = color
        else: a[cr + 1:cr + ch + 1, cc + cw + 1] = color
    kh, kw = key.shape
    a[kr:kr + kh, kc:kc + kw] = key
    for s, color in outer.items():
        _side(a, s)[1:-1] = color
    return a


def _matrix(inner, outer):
    m = np.zeros((2, 2), dtype=int)
    for s, target in zip(inner, outer):
        m += np.outer(NORMAL[target], NORMAL[s])
    return m


def _arrangement_matrix(inner_index, outer_index, reverse):
    inner = (SIDES[inner_index], SIDES[(inner_index + 1) % 4])
    outer = (SIDES[outer_index], SIDES[(outer_index + 1) % 4])
    return _matrix(inner, outer[::-1] if reverse else outer)


def _build(rng, palette, inner_index, outer_index, reverse, d, *, teaching=False, second_d=None):
    inner = (SIDES[inner_index], SIDES[(inner_index + 1) % 4])
    outer_sides = (SIDES[outer_index], SIDES[(outer_index + 1) % 4])
    if reverse:
        outer_sides = outer_sides[::-1]
    m1 = _matrix(inner, outer_sides)
    m2 = _matrix(inner, outer_sides[::-1])
    d2 = d if second_d is None else second_d
    for attempt in range(160):
        card = _make_card(rng, palette)
        wall, free = _make_bay(rng)
        offsets = [_offsets(card, palette[0], palette[4], m1)]
        dirs = [d]
        if teaching:
            offsets.append(_offsets(card, palette[0], palette[4], m2))
            dirs.append(d2)
        valid = [_valid_anchors(free, o) for o in offsets]
        if teaching:
            solid = [_valid_anchors(free, _rectangle_offsets(card, palette[4], m))
                     for m in (m1, m2)]
        candidates, strong = [], []
        for rr, cc in np.argwhere(np.logical_and.reduce(valid)):
            p = int(rr), int(cc)
            paths = [_trace(v, p, direction) for v, direction in zip(valid, dirs)]
            if any(len(path) < 2 for path in paths):
                continue
            sweeps = [_swept(path, o, direction) for path, o, direction in zip(paths, offsets, dirs)]
            item = (p, paths, set.union(*sweeps))
            candidates.append(item)
            if teaching:
                corner = any(path != _trace(v, p, direction, False)
                             for path, v, direction in zip(paths, valid, dirs))
                point_differs = any(path != _trace(free, p, direction)
                                    for path, direction in zip(paths, dirs))
                rectangle_differs = any(not v[path[-1]] or (v[p] and path != _trace(v, p, direction))
                                        for path, v, direction in zip(paths, solid, dirs))
                if corner and point_differs and rectangle_differs and paths[0][-1] != paths[1][-1]:
                    strong.append(item)
        if not candidates or (teaching and not strong):
            continue
        desired = int(rng.integers(1, 5))
        first = _pick(rng, strong if teaching else candidates)
        selected, used = [first], set(first[2])
        for i in rng.permutation(len(candidates)):
            if len(selected) >= desired:
                break
            item = candidates[int(i)]
            isolated = all(abs(item[0][0] - old[0][0]) + abs(item[0][1] - old[0][1]) > 1
                           for old in selected)
            if isolated and used.isdisjoint(item[2]):
                selected.append(item)
                used.update(item[2])
            if len(selected) >= desired:
                break
        key = _make_key(rng, d, palette)
        shapes = [wall.shape, (card.shape[0] + 2, card.shape[1] + 2), key.shape]
        packing = _pack(rng, shapes)
        if packing is None:
            key = _make_key(rng, d, palette, True)
            packing = ((30, 30), [(2, 2), (2, 21), (21, 2)])
        starts = [item[0] for item in selected]
        outer = dict(zip(outer_sides, palette[5:7]))
        a = _paint_scene(rng, wall, card, key, starts, palette, inner, outer, packing)
        info = _decode(a)
        pair = dict(input=a, output=_prediction(a, info)[0])
        if not teaching:
            return [pair]
        b = a.copy()
        for s in outer:
            _side(b, s)[1:-1] = palette[6] if outer[s] == palette[5] else palette[5]
        if d2 != d:
            # Compact keys have direction-independent square bounds only for
            # cardinal versus diagonal in general, so repaint and repack both
            # scenes with compact keys in equal padded carrier rectangles.
            key2 = _make_key(rng, d2, palette, True)
            side = max(*key.shape, *key2.shape)
            k1 = np.full((side, side), palette[1], dtype=int)
            k2 = k1.copy()
            k1[:key.shape[0], :key.shape[1]] = key
            k2[:key2.shape[0], :key2.shape[1]] = key2
            packing = _pack(rng, [wall.shape, shapes[1], (side, side)])
            if packing is None:
                # Guaranteed compact tail: both glyphs fit a 7-by-7 carrier.
                key = _make_key(rng, d, palette, True)
                k1 = np.full((7, 7), palette[1], dtype=int)
                k2 = k1.copy()
                k1[:key.shape[0], :key.shape[1]] = key
                k2[:key2.shape[0], :key2.shape[1]] = key2
                packing = ((30, 30), [(2, 2), (2, 21), (21, 2)])
            a = _paint_scene(rng, wall, card, k1, starts, palette, inner, outer, packing)
            b = _paint_scene(rng, wall, card, k2, starts, palette, inner,
                             dict(zip(outer_sides[::-1], palette[5:7])), packing)
            pair = dict(input=a, output=solve(a))
        return [pair, dict(input=b, output=solve(b))]
    raise AssertionError("bounded docking construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    nt = _integer(num_train, "num_train", 2, 10)
    nq = _integer(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    # Teaching contrasts share the complete bay, card, guides and launch points.
    arrangements = [(i, o, rev) for i in range(4) for o in range(4) for rev in (False, True)]
    first = _pick(rng, [x for x in arrangements if any(
        np.array_equal(_arrangement_matrix(*x), MATRICES[j]) for j in (1, 3))])
    held = _pick(rng, [(i, o) for i in range(4) for o in range(4) if (i, o) != first[:2]])
    palette = list(map(int, rng.permutation(10)))
    diagonal = _pick(rng, [d for d in DIRECTIONS if d[0] and d[1]])
    cardinal = _pick(rng, CARDINAL)
    contrast = _build(rng, palette, *first, diagonal, teaching=True,
                      second_d=cardinal if nt == 2 else None)
    puzzle = {"train": contrast, "test": []}
    seen = {_fingerprint(p["input"]) for p in contrast}
    for split, count in (("train", nt - 2), ("test", nq)):
        for index in range(count):
            for _ in range(20):
                if split == "test":
                    i, o = held
                    reverse = bool(rng.integers(2))
                else:
                    i, o, reverse = _pick(rng, [x for x in arrangements if x[:2] != held])
                d = cardinal if split == "train" and index == 0 else _pick(rng, DIRECTIONS)
                fresh_palette = list(map(int, rng.permutation(10)))
                pair = _build(rng, fresh_palette, i, o, reverse, d)[0]
                fp = _fingerprint(pair["input"])
                if fp not in seen:
                    break
            _require(fp not in seen, "duplicate input construction")
            seen.add(fp)
            puzzle[split].append(pair)
    # Teaching order is not a directional codebook.
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, information = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            a, b = _grid(pair["input"]), _grid(pair["output"])
            _require(a.shape == b.shape and not np.array_equal(a, b), "each pair must stamp on its canvas")
            fp = _fingerprint(a)
            _require(fp not in seen, "duplicate inputs")
            seen.add(fp)
            info = _decode(a)
            signatures = {tuple(sorted(_offsets(info["card"], info["background"], info["anchor"], m)))
                          for m in MATRICES}
            _require(len(signatures) == 8, "stencil orientation must be distinguishable")
            expected, paths = _prediction(a, info)
            _require(np.array_equal(expected, b), "wrong docking answer")
            _require(all(len(p) >= 2 for p in paths), "each launch must move")
            used = set()
            for path in paths:
                cells = _swept(path, info["offsets"], info["key"]["direction"])
                _require(used.isdisjoint(cells), "interacting stamp sweeps")
                _require(all(info["free"][p] for p in cells), "stamp crosses a wall")
                used.update(cells)
            fixed = (a != info["background"]) & (a != info["anchor"])
            _require(np.array_equal(a[fixed], b[fixed]), "wall, key, guide or template altered")
            _require(np.count_nonzero(a == info["anchor"]) == np.count_nonzero(b == info["anchor"]),
                     "anchor conservation")
            for color in info["ink"]:
                _require(np.count_nonzero(b == color) == (1 + len(paths)) * np.count_nonzero(a == color),
                         "stencil ink was lost, recolored or overlapped")
            info["point_witness"] = any(paths[j] != _trace(info["free"], p, info["key"]["direction"])
                                        for j, p in enumerate(info["starts"]))
            info["rectangle_witness"] = _rectangle_witness(info, paths)
            _require(all(abs(p[0] - q[0]) + abs(p[1] - q[1]) > 1
                         for j, p in enumerate(info["starts"]) for q in info["starts"][j + 1:]),
                     "launch points must be isolated")
            info["paths"] = paths
            info["corner_witness"] = any(path != _trace(info["valid"], p, info["key"]["direction"], False)
                                         for path, p in zip(paths, info["starts"]))
            information[split].append(info)
    train = information["train"]
    _require(len({a["key"]["direction"] for a in train}) >= 2, "missing direction variation in demonstrations")
    _require(any(a["corner_witness"] for a in train), "missing swept-corner demonstration")
    _require(any(a["point_witness"] for a in train), "missing silhouette-dependent stopping demonstration")
    _require(any(a["rectangle_witness"] for a in train), "missing transparent-silhouette demonstration")
    _require(any(not np.array_equal(a["matrix"], b["matrix"])
                 and a["guides"] == b["guides"] and set(a["controls"]) == set(b["controls"])
                 and np.array_equal(a["card"], b["card"])
                 and a["starts"] == b["starts"] and np.array_equal(a["free"], b["free"])
                 for j, a in enumerate(train) for b in train[j + 1:]), "missing identity correspondence contrast")
    train_arrangements = {(tuple(sorted(a["guides"])), tuple(sorted(a["controls"]))) for a in train}
    _require(all((tuple(sorted(a["guides"])), tuple(sorted(a["controls"]))) not in train_arrangements
                 for a in information["test"]), "test must recombine unseen occupied-side arrangements")
    # These ablations are family-specific witnesses, not independent solvers.
    for omit in ("point", "orientation", "polarity"):
        witnesses = []
        for pair, a in zip(puzzle["train"], train):
            grid = _grid(pair["input"])
            kwargs = {"point_only": True} if omit == "point" else (
                {"matrix": MATRICES[0]} if omit == "orientation" else {"reverse": True})
            witnesses.append(not np.array_equal(_prediction(grid, a, **kwargs)[0], _grid(pair["output"])))
        _require(any(witnesses), f"training does not defeat {omit} shortcut")


def measure_puzzle(puzzle):
    records = []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            a = _grid(pair["input"])
            info = _decode(a)
            _, paths = _prediction(a, info)
            kr, kc, kr1, kc1 = info["key"]["bounds"]
            mask = a == info["wall"]
            mask[kr:kr1, kc:kc1] = False
            r, c, r1, c1 = _bbox(np.argwhere(mask))
            sub = mask[r:r1, c:c1]
            records.append(dict(split=split, shape=a.shape, stencil=info["card"].shape,
                occupied=len(info["offsets"]), ink_colors=len(info["ink"]),
                matrix=next(j for j, m in enumerate(MATRICES) if np.array_equal(info["matrix"], m)),
                direction=info["key"]["direction"], launches=len(paths),
                distances=[len(p) - 1 for p in paths], background=info["background"],
                bay=(r1 - r, c1 - c), regions=len(info["regions"]),
                irregular=not all(np.all(_side(sub, s)) for s in SIDES),
                key=(kr1 - kr, kc1 - kc), lengths=info["key"]["lengths"],
                corner=any(path != _trace(info["valid"], start, info["key"]["direction"], False)
                           for path, start in zip(paths, info["starts"]))))
    return records


def self_test() -> dict:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [row for p in puzzles for row in measure_puzzle(p)]
    _require(len({tuple(_fingerprint(x["input"]) for x in p["train"] + p["test"])
                  for p in puzzles}) == 42, "seed diversity")
    for split in ("train", "test"):
        rows = [r for r in records if r["split"] == split]
        _require({r["matrix"] for r in rows} == set(range(8)), "symmetry coverage")
        _require({r["direction"] for r in rows} == set(DIRECTIONS), "compass coverage")
        _require(len({r["launches"] for r in rows}) >= 3, "launch-count variation")
        _require(any(r["irregular"] for r in rows) and any(not r["irregular"] for r in rows), "bay outline variation")
    _require(len({r["shape"] for r in records}) >= 25, "canvas variation")
    _require(len({r["bay"] for r in records}) >= 25, "bay scale/aspect variation")
    _require(len({r["stencil"] for r in records}) >= 6, "stencil extent variation")
    _require(len({r["occupied"] for r in records}) >= 8, "silhouette area variation")
    _require({r["ink_colors"] for r in records} == {1, 2}, "ink palette variation")
    _require(len({r["key"] for r in records}) >= 8, "glyph geometry variation")
    _require({d for r in records for d in r["distances"]} >= {1, 2, 3, 4, 5}, "travel distance variation")
    _require(any(r["regions"] > 1 for r in records), "chamber variation")
    # Every legal count combination, including the two-example teaching regime.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 11 * nt + nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "count contract")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (5, 5), (6, 5), (2, 2),
                   (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(0), puzzles[0]
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ("input", "output")), "nondeterminism")
    # An equivariance check exercises unseen rotated/reflected layouts without
    # relying on hidden source coordinates or the generator's sampling plans.
    for m_index in (1, 4, 6):
        transform = (lambda x: np.rot90(x)) if m_index == 1 else (
            (lambda x: np.fliplr(x)) if m_index == 4 else (lambda x: x.T))
        for pair in puzzles[0]["train"] + puzzles[0]["test"]:
            _require(np.array_equal(solve(transform(pair["input"])), transform(pair["output"])),
                     "whole-scene symmetry failure")
    for mutation in ("cell", "point_stop", "guide"):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
        pair = bad["test"][0]
        info = _decode(pair["input"])
        if mutation == "cell":
            pair["output"][info["starts"][0]] = info["wall"]
        elif mutation == "point_stop":
            pair["output"] = _prediction(pair["input"], info, point_only=True)[0]
        else:
            color = next(iter(info["guides"].values()))
            cells = np.argwhere(pair["input"] == color)
            r, c = next(tuple(map(int, p)) for p in cells if 0 < p[0] < pair["input"].shape[0] - 1
                        and 0 < p[1] < pair["input"].shape[1] - 1)
            pair["input"][r, c] = info["background"]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"accepted {mutation} corruption")
    return dict(seeds=42, pairs=len(records), input_shapes=len({r["shape"] for r in records}),
                bay_shapes=len({r["bay"] for r in records}),
                stencil_shapes=len({r["stencil"] for r in records}),
                launch_counts=sorted({r["launches"] for r in records}),
                corner_pairs=sum(r["corner"] for r in records))
