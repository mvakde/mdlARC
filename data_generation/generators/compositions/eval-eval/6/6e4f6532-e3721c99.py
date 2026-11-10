"""Perforated compass devices: topology determines a rigid docking pose.

Only standard-library modules and NumPy are used. No files are read or written
by generation, validation, solving, or self_test. All topology is 4-connected.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_THICKNESS = 3
_HOST_SEEDS = (0, 787254873, 270800998, 192403203, 166480753, 827670444,
               838008293, 885064567, 727094491, 461236466, 515821988,
               85574658, 871599232, 678202103, 200207540, 366431712,
               814678972, 123156661, 792987091, 331020184, 58021016,
               409391929, 60199007, 841654447, 939662042, 696461573,
               639244558, 745424228, 703789964, 239778740, 480350532,
               334021562, 466336072, 258283839, 705848466, 523321904,
               481067096, 494593837, 12996376, 109140310, 954585169,
               419020888)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grids must have two dimensions in 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and a.dtype.kind != 'b',
             "grid cells must be integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "grid colors must be in 0..9")
    return a


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _DIRS:
                p = (r + dr, c + dc)
                if p in todo:
                    todo.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _bbox(points):
    return (min(r for r, _ in points), max(r for r, _ in points),
            min(c for _, c in points), max(c for _, c in points))


def _mask(points):
    r0, r1, c0, c1 = _bbox(points)
    a = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        a[r - r0, c - c0] = True
    return a


def _cavities(mask):
    padded = np.pad(np.asarray(mask, dtype=bool), 1)
    return [g for g in _components(~padded)
            if all(0 < r < padded.shape[0] - 1 and
                   0 < c < padded.shape[1] - 1 for r, c in g)]


def _hole_count(points):
    return len(_cavities(_mask(points)))


def _pose(cells, transform):
    result = []
    for (r, c), color in cells.items():
        if transform >= 4:
            c = -c
        for _ in range(transform % 4):
            r, c = c, -r
        result.append((r, c, color))
    r0 = min(r for r, _, _ in result)
    c0 = min(c for _, c, _ in result)
    return {(r - r0, c - c0): int(v) for r, c, v in result}


def _shift(cells, offset):
    return {(r + offset[0], c + offset[1]): v for (r, c), v in cells.items()}


def _shape(points):
    return tuple(sorted(_pose({p: 1 for p in points}, 0)))


def _canonical(points):
    return min(tuple(sorted(_pose({p: 1 for p in points}, t))) for t in range(8))


def _key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.uint8).tobytes()


def _halo(points):
    return {(r + dr, c + dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _band_slices(h, w):
    return ((slice(0, 3), slice(3, w - 3)),
            (slice(3, h - 3), slice(w - 3, w)),
            (slice(h - 3, h), slice(3, w - 3)),
            (slice(3, h - 3), slice(0, 3)))


def _direction(badge, hub):
    r0, r1, c0, c1 = _bbox(hub)
    a, b, c, d = _bbox(badge)
    transverse_c = not (d < c0 or c > c1)
    transverse_r = not (b < r0 or a > r1)
    candidates = [i for i, good in enumerate((b < r0 and transverse_c,
                  c > c1 and transverse_r, a > r1 and transverse_c,
                  d < c0 and transverse_r)) if good]
    _require(len(candidates) == 1, "each badge must lie on one hub axis")
    return candidates[0]


def _parse(input_grid):
    a = _grid(input_grid)
    h, w = a.shape
    _require(min(h, w) >= 15, "a compass needs room for its four bars")
    bg = int(a[0, 0])
    for rows, cols in ((slice(0, 3), slice(0, 3)),
                       (slice(0, 3), slice(w - 3, w)),
                       (slice(h - 3, h), slice(0, 3)),
                       (slice(h - 3, h), slice(w - 3, w))):
        _require(bool(np.all(a[rows, cols] == bg)), "corners must be background")
    mapping, bars = {}, []
    for side, sl in enumerate(_band_slices(h, w)):
        patch = a[sl]
        colors = set(map(int, np.unique(patch))) - {bg}
        _require(len(colors) == 1, "a compass bar has one color")
        color = next(iter(colors))
        material = patch == color
        _require(len(_components(material)) == 1, "a bar must be connected")
        # A bar's entire outer contour is present; only its interior is punched.
        _require(bool(material[0].all() and material[-1].all() and
                      material[:, 0].all() and material[:, -1].all()),
                 "bar holes must be enclosed")
        k = len(_cavities(material))
        _require(k not in mapping, "compass classes must be distinct")
        mapping[k] = (side, color)
        bars.append(color)
    _require(len(set(bars)) == 4, "compass bar colors must be distinct")
    interior = np.zeros_like(a, dtype=bool)
    interior[3:-3, 3:-3] = True
    objects = _components(interior & (a != bg))
    mono, devices = [], []
    for obj in objects:
        values = {int(a[p]) for p in obj}
        if len(values) == 1:
            mono.append(obj)
        else:
            devices.append({p: int(a[p]) for p in obj})
    _require(bool(mono and devices), "need detached hubs and loose devices")
    reg_colors = {int(a[min(obj)]) for obj in mono}
    _require(len(reg_colors) == 1, "detached hub stencils share one color")
    reg = next(iter(reg_colors))
    other = set().union(*(set(d.values()) for d in devices)) - {reg}
    _require(len(other) == 2 and not (other | {reg}) & (set(bars) | {bg}),
             "hub, bridge, badge, background and bar roles must be distinct")
    # Badges are 2D glyphs. Bridge components are straight one-cell-wide lines.
    query_options = []
    for color in sorted(other):
        groups = _components(interior & (a == color))
        if all(len(g) >= 4 and min(_mask(g).shape) >= 2 for g in groups):
            query_options.append(color)
    _require(len(query_options) == 1, "badge color must be geometrically identifiable")
    query = query_options[0]
    bridge = next(iter(other - {query}))
    targets = {}
    for obj in mono:
        code = _canonical(obj)
        _require(code not in targets, "each hub code has exactly one target")
        targets[code] = obj
    parsed_devices = []
    for cells in devices:
        _require(set(cells.values()) == {reg, query, bridge},
                 "a loose device has all three material roles")
        hub = {p for p, v in cells.items() if v == reg}
        _require(len(_components(_mask(hub))) == 1, "hub must be connected")
        glyph_mask = np.zeros_like(a, dtype=bool)
        for p, v in cells.items():
            glyph_mask[p] = v == query
        badges = _components(glyph_mask)
        _require(2 <= len(badges) <= 4, "devices have two to four badges")
        dirs = [_direction(b, hub) for b in badges]
        _require(len(set(dirs)) == len(dirs), "one badge per occupied arm")
        _require(any((d + 1) % 4 in dirs for d in dirs),
                 "a device must have an adjacent pair of arms")
        for b in badges:
            _require(len(b) >= 4 and min(_mask(b).shape) >= 2,
                     "badges must be 2D glyphs")
            for cavity in _cavities(_mask(b)):
                r0, _, c0, _ = _bbox(b)
                _require(all(int(a[r0 + r - 1, c0 + c - 1]) == bg
                             for r, c in cavity), "badge cavities contain only background")
        lines_mask = np.zeros_like(a, dtype=bool)
        for p, v in cells.items():
            lines_mask[p] = v == bridge
        lines = _components(lines_mask)
        _require(len(lines) == len(badges) and
                 all(min(_mask(g).shape) == 1 for g in lines),
                 "bridges must be separate straight radial stems")
        # Each stem connects exactly the hub and one badge, with no extra pixels.
        for line in lines:
            adjacent = {(r + dr, c + dc) for r, c in line for dr, dc in _DIRS}
            _require(bool(adjacent & hub) and
                     sum(bool(adjacent & b) for b in badges) == 1,
                     "each bridge joins the hub to one badge")
        _require(_canonical(hub) in targets, "all device hub codes are visibly targeted")
        parsed_devices.append(dict(cells=cells, hub=hub, badges=badges,
                                   counts=tuple(_hole_count(b) for b in badges),
                                   directions=tuple(dirs)))
    return dict(grid=a, background=bg, query=query, bridge=bridge,
                registration=reg, mapping=mapping, bars=tuple(bars),
                targets=targets, devices=parsed_devices, interior=interior)


def _placements(device, info, use_topology=True, use_compass=True):
    """Enumerate distinct final maps, never break ties by D4 index."""
    reg = info['registration']
    target = info['targets'][_canonical(device['hub'])]
    maps = {}
    for t in range(8):
        cells = _pose(device['cells'], t)
        hub = {p for p, v in cells.items() if v == reg}
        delta = (min(target)[0] - min(hub)[0], min(target)[1] - min(hub)[1])
        if set(_shift({p: 1 for p in hub}, delta)) != target:
            continue
        badge_groups = []
        for badge, k in zip(device['badges'], device['counts']):
            # Apply exactly the SAME affine transformation to every subset.
            tagged = {p: int(p in badge) for p in device['cells']}
            posed = _pose(tagged, t)
            badge_groups.append(({p for p, v in posed.items() if v == 1}, k))
        valid = True
        for b, k in badge_groups:
            direction = _direction(b, hub)
            if use_topology and k not in info['mapping']:
                valid = False
                break
            if use_topology:
                side, color = info['mapping'][k]
            else:
                # Ablation: accept every badge and simply color by its final side.
                side, color = direction, info['bars'][direction]
            if use_compass and direction != side:
                valid = False
                break
            for p in b:
                cells[p] = color
        if not valid:
            continue
        placed = _shift(cells, delta)
        if any(not (3 <= r < info['grid'].shape[0] - 3 and
                    3 <= c < info['grid'].shape[1] - 3) for r, c in placed):
            continue
        if any(p in placed and placed[p] != reg
               for other in info['targets'].values() for p in other):
            continue
        maps[tuple(sorted(placed.items()))] = placed
    return list(maps.values())


def _resolve(info):
    accepted, erased, seen_codes = [], [], set()
    for d in info['devices']:
        candidates = _placements(d, info)
        _require(len(candidates) <= 1, "a device has ambiguous final pixels")
        if not candidates:
            erased.append(d)
            continue
        code = _canonical(d['hub'])
        _require(code not in seen_codes, "two valid devices compete for one stencil")
        seen_codes.add(code)
        accepted.append((d, candidates[0]))
    _require(seen_codes == set(info['targets']), "every target must have one valid device")
    occupied = set()
    for _, final in accepted:
        _require(not occupied & final.keys(), "final devices must be disjoint")
        occupied.update(final)
    return accepted, erased


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _parse(input_grid)
    accepted, _ = _resolve(info)
    result = info['grid'].copy()
    # Erase all loose devices before pasting; a final may occupy an old footprint.
    for device in info['devices']:
        for p in device['cells']:
            result[p] = info['background']
    for _, final in accepted:
        for p, color in final.items():
            result[p] = color
    return result


def _glyph(k, rng, rich=False, simple=False, wide_cavity=False):
    if wide_cavity:
        _require(k == 2, 'the area witness is a two-cavity glyph')
        a = np.ones((5, 5), dtype=bool)
        a[1, 1:3] = False
        a[3, 3] = False
    elif k == 0:
        h, w = (3, 3) if simple else tuple(map(int, rng.integers(2, 5, 2)))
        a = np.ones((h, w), dtype=bool)
    else:
        h, w = (3, 2 * k + 1) if k <= 2 else (5, 5 if k <= 4 else 7)
        a = np.ones((h, w), dtype=bool)
        positions = [(r, c) for r in range(1, h - 1, 2)
                     for c in range(1, w - 1, 2)]
        rng.shuffle(positions)
        for p in positions[:k]:
            a[p] = False
        if rich and not simple:
            axis = int(rng.integers(2))
            if a.shape[axis] < 6:
                index = int(rng.integers(1, a.shape[axis] - 1))
                a = np.insert(a, index, np.take(a, index, axis=axis), axis=axis)
    if not simple:
        h, w = a.shape
        protected = {(0, (w - 1) // 2), (0, w // 2),
                     (h - 1, (w - 1) // 2), (h - 1, w // 2),
                     ((h - 1) // 2, 0), (h // 2, 0),
                     ((h - 1) // 2, w - 1), (h // 2, w - 1)}
        for _ in range(10 if rich else 5):
            p = (int(rng.integers(h)), int(rng.integers(w)))
            if p in protected:
                continue
            b = a.copy()
            b[p] = ~b[p]
            if (b.any() and len(_components(b)) == 1 and
                    len(_cavities(b)) == k and
                    (not wide_cavity or max(map(len, _cavities(b)), default=0) > 1) and
                    min(_mask(set(map(tuple, np.argwhere(b)))).shape) >= 2):
                a = b
    a = np.rot90(a, int(rng.integers(4))).copy()
    if rng.random() < .5:
        a = a[:, ::-1].copy()
    return a


def _hub(code):
    if code == 0:
        return {(0, 0)}
    if code == 1:
        return {(0, 0), (0, 1), (1, 0), (1, 1)}
    return {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}


def _device(rng, code, badges, palette, rich=False, simple=False, witness=False):
    """badges is a visible-direction -> cavity-count mapping."""
    _, query, bridge, reg = palette[:4]
    hub = _hub(code)
    h, w = _mask(hub).shape
    center_r, center_c = (h - 1) // 2, (w - 1) // 2
    glyphs = {d: _glyph(k, rng, rich, simple, witness and k == 2)
              for d, k in badges.items()}
    gap = max(max(g.shape) // 2 for g in glyphs.values())
    cells = {p: reg for p in hub}
    for d, g in glyphs.items():
        gh, gw = g.shape
        if d == 0:
            origin = (-gap - gh, center_c - (gw - 1) // 2)
            line = [(r, center_c) for r in range(-gap, 0)]
        elif d == 1:
            origin = (center_r - (gh - 1) // 2, w + gap)
            line = [(center_r, c) for c in range(w, w + gap)]
        elif d == 2:
            origin = (h + gap, center_c - (gw - 1) // 2)
            line = [(r, center_c) for r in range(h, h + gap)]
        else:
            origin = (center_r - (gh - 1) // 2, -gap - gw)
            line = [(center_r, c) for c in range(-gap, 0)]
        patch = {(int(r) + origin[0], int(c) + origin[1]): query
                 for r, c in np.argwhere(g)}
        _require(not patch.keys() & cells.keys(), "internal badge collision")
        cells.update({p: bridge for p in line})
        cells.update(patch)
    return _pose(cells, 0)


def _frame(rng, h, w, palette, counts):
    a = np.full((h, w), palette[0], dtype=np.int8)
    for d, sl in enumerate(_band_slices(h, w)):
        patch = a[sl]
        patch[:] = palette[4 + d]
        work = patch if d in (0, 2) else patch.T
        k = counts[d]
        # Separated slots, with random position and occasional wider cavities.
        slots = list(range(1, work.shape[1] - 1, 2))
        rng.shuffle(slots)
        for c in slots[:k]:
            work[1, c] = palette[0]
    return a


def _positions(free, h, w):
    if h > free.shape[0] or w > free.shape[1]:
        return []
    summed = np.pad((~free).astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    bad = summed[h:, w:] - summed[:-h, w:] - summed[h:, :-w] + summed[:-h, :-w]
    return [tuple(map(int, p)) for p in np.argwhere(bad == 0)]


def _put(a, cells):
    for p, color in cells.items():
        a[p] = color


def _sample_pair(rng, palette, index, test, regime, attempt):
    keys = [0, 1, 2, 3]
    if test and regime == 'classes':
        keys = [0, 1, int(rng.choice([2, 3])), 4]
    elif index >= 2 and regime != 'classes' and rng.random() < .5:
        keys = [0, 1] + sorted(map(int, rng.choice([2, 3, 4], 2, replace=False)))
    counts = list(map(int, rng.permutation(keys)))
    if not test and index == 0:
        # First demonstration: the low two classes label adjacent sides.
        while (counts.index(0) - counts.index(1)) % 2 == 0:
            counts = list(map(int, rng.permutation(keys)))
    nvalid = 2 if test and regime == 'population' else 1
    if index >= 2 and regime != 'population' and rng.random() < .35:
        nvalid = 2
    codes = list(map(int, rng.choice(3, nvalid, replace=False)))
    if not test and index == 1:
        # Two equal-arm-count competitors must both fit in the teaching scene.
        codes[0] = 0
    narm = (4 if test and regime == 'arms' else
            3 if test else 2 if index == 0 else 3 if index == 1 else
            int(rng.integers(2, 4 if regime == 'arms' else 5)))
    final_devices = []
    for j, code in enumerate(codes):
        if j:
            start = int(rng.integers(4))
            dirs = [start, (start + 1) % 4]
        elif not test and index == 0:
            dirs = [counts.index(0), counts.index(1)]
        elif not test and index == 1:
            # Include both previously unshown high classes in the second lesson.
            dirs = sorted(set([counts.index(2), counts.index(3)] +
                              [int(rng.choice([d for d in range(4) if counts[d] not in (2, 3)]))]))
        elif test and regime == 'classes':
            omitted = int(rng.choice([d for d in range(4) if counts[d] != 4]))
            dirs = [d for d in range(4) if d != omitted]
        else:
            start = int(rng.integers(4))
            dirs = [(start + i) % 4 for i in range(narm)]
        final_devices.append(_device(rng, code, {d: counts[d] for d in dirs},
                                     palette, rich=attempt < 12 and index != 0 and j == 0,
                                     witness=not test and index == 1))
    # The decoy duplicates a real target's hub. It cannot be discarded by code.
    if not test and index == 0:
        absent = next(k for k in range(5) if k not in counts)
        decoy_badges = {0: 0, 1: absent}
    elif not test and index == 1:
        side = counts.index(0)
        opposite = counts[(side + 2) % 4]
        third = int(rng.choice([k for k in counts if k not in (0, opposite)]))
        decoy_badges = {0: 0, 1: opposite, 2: third}
    elif rng.random() < .5:
        k = int(rng.choice(counts))
        decoy_badges = {0: k, 1: k}
        if nvalid == 1 and rng.random() < .35:
            decoy_badges[2] = int(rng.choice(counts))
    else:
        absent = next(k for k in range(5) if k not in counts)
        decoy_badges = {0: int(rng.choice(counts[:2])), 1: absent}
    decoy = _device(rng, codes[0], decoy_badges, palette)
    transforms = [int(rng.integers(8)) for _ in final_devices]
    if not test and index < 2:
        transforms[0] = int(rng.integers(1, 4) if index == 0 else rng.integers(4, 8))
    sources = [_pose(d, t) for d, t in zip(final_devices, transforms)]
    sources.append(_pose(decoy, int(rng.integers(8))))
    if not test and index == 1 and attempt >= 12:
        # Preserve the reflected lesson while orienting the two bounding boxes
        # along the same long axis, where a 24-cell interior always has room.
        options = [_pose(final_devices[0], t) for t in range(4, 8)]
        sources[0] = next(d for d in options if _mask(d).shape[0] >= _mask(d).shape[1])
        if _mask(sources[-1]).shape[0] < _mask(sources[-1]).shape[1]:
            sources[-1] = _pose(sources[-1], 1)
    maxh = max(_bbox(d)[1] + 1 for d in sources)
    maxw = max(_bbox(d)[3] + 1 for d in sources)
    lowh, loww = max(15, maxh + 2), max(15, maxw + 2)
    if max(lowh, loww) > 24:
        return None
    ih = 24 if attempt >= 12 else int(rng.integers(lowh, 25))
    iw = 24 if attempt >= 12 else int(rng.integers(loww, 25))
    a = _frame(rng, ih + 6, iw + 6, palette, counts)
    base = a.copy()
    free = np.zeros_like(a, dtype=bool)
    free[3:-3, 3:-3] = True
    source_maps = []
    for device in sorted(sources, key=lambda d: -np.prod(_mask(d).shape)):
        dh, dw = _mask(device).shape
        options = _positions(free, dh, dw)
        if not options:
            return None
        r, c = options[int(rng.integers(len(options)))]
        placed = _shift(device, (r, c))
        source_maps.append(placed)
        free[max(3, r - 1):min(a.shape[0] - 3, r + dh + 1),
             max(3, c - 1):min(a.shape[1] - 3, c + dw + 1)] = False
    old = set().union(*(set(d) for d in source_maps))
    blocked_hubs = _halo(old)
    final_maps, final_blocked, target_maps = [], set(), []
    for device in final_devices:
        dh, dw = _mask(device).shape
        positions = [(r, c) for r in range(3, a.shape[0] - 3 - dh + 1)
                     for c in range(3, a.shape[1] - 3 - dw + 1)]
        rng.shuffle(positions)
        found = False
        for origin in positions:
            placed = _shift(device, origin)
            hub = {p: v for p, v in placed.items() if v == palette[3]}
            if placed.keys() & final_blocked or hub.keys() & blocked_hubs:
                continue
            final_maps.append(placed)
            target_maps.append(hub)
            final_blocked.update(_halo(placed))
            blocked_hubs.update(_halo(hub))
            found = True
            break
        if not found:
            return None
    for d in source_maps + target_maps:
        _put(a, d)
    # Construct the answer from the intended arrangement, independently of solve.
    out = base.copy()
    for device in final_maps:
        hub = {p for p, v in device.items() if v == palette[3]}
        for badge in _components(_paint_mask(device, palette[1], a.shape)):
            side = _direction(badge, hub)
            for p in badge:
                device[p] = palette[4 + side]
        _put(out, device)
    try:
        info = _parse(a)
        accepted, erased = _resolve(info)
        if len(accepted) != nvalid or len(erased) != 1 or not np.array_equal(solve(a), out):
            return None
    except ValueError:
        return None
    return {'input': a, 'output': out}


def _paint_mask(cells, color, shape):
    mask = np.zeros(shape, dtype=bool)
    for p, v in cells.items():
        mask[p] = v == color
    return mask


def _fallback(rng, palette, ordinal, arms):
    """Explicit finite fitting scene; no random rejection or alternate rule."""
    counts = [0, 1, 2, 3]
    a = _frame(rng, 30, 30, palette, counts)
    out = a.copy()
    dirs = [0, 2, 3] if arms == 3 else list(range(arms))
    valid = _device(rng, 0, {d: counts[d] for d in dirs}, palette,
                    simple=True, witness=arms >= 3)
    decoy_badges = {0: 0, 1: 2, 2: 1} if arms == 3 else {0: 0, 1: 4 if arms == 2 else 0}
    decoy = _device(rng, 0, decoy_badges, palette, simple=True)
    transform = 1 if arms == 2 else 4 + (ordinal // 3) % 4
    source = _shift(_pose(valid, transform), (3 + ordinal % 3, 3))
    bad_origin = (17, 17) if arms == 2 else (16, 18) if arms == 3 else (21, 21)
    bad = _shift(decoy, bad_origin)
    hub = min(p for p, v in valid.items() if v == palette[3])
    final = _shift(valid, (10 - hub[0], 19 - hub[1]))
    _put(a, source)
    _put(a, bad)
    a[10, 19] = palette[3]
    final_hub = {(10, 19)}
    for badge in _components(_paint_mask(final, palette[1], a.shape)):
        side = _direction(badge, final_hub)
        for p in badge:
            final[p] = palette[4 + side]
    _put(out, final)
    return {'input': a, 'output': out}


def _counts(seed, num_train, num_test):
    for value, name, lo, hi in ((num_train, 'num_train', 2, 10),
                                (num_test, 'num_test', 1, 4)):
        if value is not None:
            _require(not isinstance(value, (bool, np.bool_)) and
                     isinstance(value, (int, np.integer)) and lo <= value <= hi,
                     f'{name} must be an integer in {lo}..{hi}')
    if num_test is None:
        num_test = 1 + seed % min(4, (num_train or 10) - 1)
    if num_train is None:
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    _require(num_train > num_test, 'num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, 'seed must be a nonnegative integer')
    seed = int(seed)
    num_train, num_test = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(['arms', 'population', 'classes']))
    fixed_roles = bool(rng.random() < .4)
    puzzle_palette = list(map(int, rng.permutation(10)[:8]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            palette = puzzle_palette if fixed_roles else list(map(int, rng.permutation(10)[:8]))
            pair = None
            for attempt in range(40):
                candidate = _sample_pair(rng, palette, i, split == 'test', regime, attempt)
                if candidate is not None and _key(candidate['input']) not in seen:
                    pair = candidate
                    break
            if pair is None:
                arms = 2 if split == 'train' and i == 0 else 3 if split == 'train' else 4
                pair = _fallback(rng, palette, len(seen), arms)
            _require(_key(pair['input']) not in seen, 'duplicate generated input')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    info = _parse(np.asarray(pair['input']))
    accepted, erased = _resolve(info)
    old = set().union(*(set(d['cells']) for d in info['devices']))
    new = set().union(*(set(m) for _, m in accepted))
    return dict(shape=tuple(info['grid'].shape), roles=(info['background'], info['query'],
                info['bridge'], info['registration']) + info['bars'],
                targets=len(accepted), erased=len(erased),
                arms=tuple(sorted(len(d['badges']) for d, _ in accepted)),
                classes=tuple(sorted(info['mapping'])),
                bar_order=tuple(next(k for k, (s, _) in info['mapping'].items() if s == d)
                                for d in range(4)),
                badge_shapes=tuple(sorted(_mask(b).shape for d in info['devices'] for b in d['badges'])),
                cavity_areas=tuple(sorted(len(g) for d in info['devices'] for b in d['badges']
                                          for g in _cavities(_mask(b)))),
                irregular=sum(not _mask(g).all() for d in info['devices'] for b in d['badges']
                              for g in _cavities(_mask(b))),
                overlap=len(old & new),
                motions=tuple(_motion_type(d, m, info) for d, m in accepted),
                compass_needed=any(len(_placements(d, info, use_compass=False)) > 1
                                   for d, _ in accepted),
                topology_needed=any(_placements(d, info, use_topology=False) for d in erased))


def _motion_type(device, final, info):
    neutral = {p: info['query'] if v in info['bars'] else v for p, v in final.items()}
    signature = tuple(sorted(_pose(neutral, 0).items()))
    matches = [t for t in range(8)
               if tuple(sorted(_pose(device['cells'], t).items())) == signature]
    _require(bool(matches), 'a final assembly must be a rigid image of its source')
    return 'translation' if 0 in matches else 'rotation' if any(t < 4 for t in matches) else 'reflection'


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'},
             'puzzle must have exactly train/test')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(0, len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'},
                     'pairs must have exactly input/output')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'output shape must be preserved')
            _require(_key(inp) not in seen, 'inputs must be distinct')
            seen.add(_key(inp))
            info = _parse(inp)
            accepted, erased = _resolve(info)
            predicted = solve(inp)
            _require(np.array_equal(predicted, out), 'answer violates topological compass docking')
            _require(not np.array_equal(inp, out), 'a scene must change')
            _require(np.array_equal(inp[~info['interior']], out[~info['interior']]),
                     'all bars, holes and corners must be preserved')
            # Verify topology and whole-badge color after the actual rigid motion.
            for d, final in accepted:
                for k in d['counts']:
                    color = info['mapping'][k][1]
                    group = {p for p, v in final.items() if v == color}
                    _require(_hole_count(group) == k, 'recoloring must preserve each badge topology')
            if split == 'train':
                training.append((info, accepted, erased))
    _require(any(erased for _, _, erased in training), 'training must demonstrate device erasure')
    _require(any(len(_placements(d, info, use_compass=False)) > 1
                 for info, accepted, _ in training for d, _ in accepted),
             'training must distinguish compass orientation from class recoloring alone')
    _require(any(_placements(d, info, use_topology=False)
                 for info, _, erased in training for d in erased),
             'training must distinguish topological badges from arbitrary arm glyphs')
    # At least one impossible device uses exclusively listed classes: erasure
    # cannot be explained solely by filtering unlisted cavity counts.
    _require(any(all(k in info['mapping'] for k in d['counts']) and
                 len(set(d['counts'])) == len(d['counts'])
                 for info, _, erased in training for d in erased),
             'training must expose angular inconsistency with distinct listed classes')
    motions = {_motion_type(d, m, info) for info, accepted, _ in training for d, m in accepted}
    _require({'rotation', 'reflection'} <= motions, 'training must show a rotation and a reflection')
    _require(any(len(cavity) > 1 for _, accepted, _ in training for d, _ in accepted
                 for b in d['badges'] for cavity in _cavities(_mask(b))),
             'training must distinguish cavity count from cavity area')


def self_test():
    """42-seed bounded audit, interface edges, topology fixtures and mutations."""
    diagonal = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool)
    assert len(_cavities(diagonal)) == 1
    diagonal[0, 1] = False
    assert len(_cavities(diagonal)) == 0
    notch = np.array([[1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]], dtype=bool)
    assert len(_cavities(notch)) == 0
    records, signatures, digests = [], set(), []
    for seed in _HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                inp = pair['input'].copy()
                assert np.array_equal(solve(inp), pair['output'])
                assert np.array_equal(inp, pair['input']), 'solve mutated its argument'
                for grid in pair.values():
                    digest.update(bytes(grid.shape) + grid.tobytes())
                records.append(_features(pair))
                signatures.add(_key(inp))
        digests.append(digest.hexdigest())
    assert len(set(digests)) == 42 and len(signatures) == len(records)
    assert len({r['shape'] for r in records}) >= 20
    assert len({r['badge_shapes'] for r in records}) >= 50
    assert len({r['bar_order'] for r in records}) >= 20
    assert {n for r in records for n in r['arms']} == {2, 3, 4}
    assert {r['targets'] for r in records} == {1, 2}
    assert len({r['roles'] for r in records}) >= 30
    assert any(r['irregular'] for r in records)
    assert any(max(r['cavity_areas'], default=0) >= 3 for r in records)
    assert any(r['overlap'] for r in records)
    assert any(4 in r['classes'] for r in records)
    p, again = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in p
               for a, b in zip(p[s], again[s]) for k in a)
    # Force the finite fallback; every ordinal must fit and remain distinct.
    rng = np.random.default_rng(555)
    fallback = {'train': [_fallback(rng, list(range(8)), i, 2 if i == 0 else 3)
                           for i in range(10)],
                'test': [_fallback(rng, list(range(8)), i, 4) for i in range(10, 14)]}
    validate_generated_puzzle(fallback)
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            q = generate_puzzle(100 + n * 10 + m, n, m)
            assert len(q['train']) == n and len(q['test']) == m
    for n, m in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for cell in ((0, 3), (10, 10)):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        grid = bad['test'][0]['output']
        grid[cell] = (int(grid[cell]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('corrupted output accepted')
    return {'seeds': len(_HOST_SEEDS), 'pairs': len(records),
            'shapes': len({r['shape'] for r in records}),
            'role_palettes': len({r['roles'] for r in records}),
            'badge_shape_sequences': len({r['badge_shapes'] for r in records}),
            'bar_orders': len({r['bar_order'] for r in records}),
            'overlap_pairs': sum(r['overlap'] > 0 for r in records),
            'irregular_cavity_pairs': sum(r['irregular'] > 0 for r in records),
            'max_cavity_area': max(max(r['cavity_areas'], default=0) for r in records),
            'two_target_pairs': sum(r['targets'] == 2 for r in records),
            'digest_sha256': hashlib.sha256(''.join(digests).encode()).hexdigest()}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
