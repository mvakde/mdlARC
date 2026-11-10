"""Mirror relay rooms: geometric routing controls cumulative anchored reflection.

Only NumPy and the standard library are used. Ground truth paints a generated
integer lattice; solve independently decodes pixels and rescales exact rasters.
"""
from __future__ import annotations

from itertools import combinations, permutations
import json
import numpy as np

# N, E, S, W. Opposite ports have the same transverse coordinate convention:
# top/bottom read left to right; left/right read top to bottom.
OPPOSITE = (2, 3, 0, 1)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), "grid size")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((a >= 0) & (a <= 9)), "ARC colors required")
    return a.astype(np.int8, copy=True)


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _neighbors(point):
    r, c = point
    return ((r-1, c), (r+1, c), (r, c-1), (r, c+1))


def _connected(points):
    remaining = set(points)
    if not remaining:
        return False
    stack = [remaining.pop()]
    while stack:
        for q in _neighbors(stack.pop()):
            if q in remaining:
                remaining.remove(q)
                stack.append(q)
    return not remaining


def _edge(panel, side):
    return (panel[0, :], panel[:, -1], panel[-1, :], panel[:, 0])[side]


def _border_point(p, side, position):
    return ((0, position), (position, p-1), (p-1, position), (position, 0))[side]


def _partition(grid):
    candidates = []
    for divider in np.unique(grid):
        rows = list(map(int, np.flatnonzero(np.all(grid == divider, axis=1))))
        cols = list(map(int, np.flatnonzero(np.all(grid == divider, axis=0))))
        if not rows and not cols:
            continue
        rb, cb = [-1] + rows + [grid.shape[0]], [-1] + cols + [grid.shape[1]]
        hs = [b-a-1 for a, b in zip(rb, rb[1:])]
        ws = [b-a-1 for a, b in zip(cb, cb[1:])]
        if min(hs+ws) < 5 or len(set(hs+ws)) != 1:
            continue
        bounds = [(r+1, rb[i+1], c+1, cb[j+1])
                  for i, r in enumerate(rb[:-1]) for j, c in enumerate(cb[:-1])]
        if not 2 <= len(bounds) <= 9:
            continue
        panels = [grid[r0:r1, c0:c1].copy() for r0, r1, c0, c1 in bounds]
        corners = {int(p[r, c]) for p in panels
                   for r, c in ((0, 0), (0, -1), (-1, 0), (-1, -1))}
        if len(corners) != 1 or any(np.any(p == divider) for p in panels):
            continue
        bg = next(iter(corners))
        if bg == divider:
            continue
        candidates.append(dict(panels=panels, bounds=bounds, background=bg,
                               divider=int(divider), layout=(len(hs), len(ws))))
    _require(len(candidates) == 1, "one unambiguous room partition required")
    return candidates[0]


def _square(core, color):
    coords = np.argwhere(core == color)
    _require(len(coords) > 0, "missing anchor")
    r, c = map(int, coords.min(0))
    b, d = map(int, coords.max(0)+1)
    _require(b-r == d-c and len(coords) == (b-r)**2, "one solid square anchor")
    return r, c, b-r


def _route(ports, source):
    """Whole-chain search; local duplicate matches need not imply ambiguity."""
    solutions = []

    def visit(order, entered, sides):
        if len(solutions) >= 2:
            return
        current = order[-1]
        exits = [s for s in ports[current] if s != entered]
        if len(order) == len(ports):
            if not exits:
                solutions.append((tuple(order), tuple(sides)))
            return
        if len(exits) != 1:
            return
        side = exits[0]
        opposite = OPPOSITE[side]
        for nxt in range(len(ports)):
            if (nxt not in order and ports[nxt].get(opposite) == ports[current][side]):
                visit(order+[nxt], opposite, sides+[side])

    visit([source], None, [])
    _require(len(solutions) == 1, "route must use every room exactly once, uniquely")
    return solutions[0]


def _decode(grid):
    data = _partition(grid)
    panels, bg = data['panels'], data['background']
    cores = [p[1:-1, 1:-1] for p in panels]
    shared = set.intersection(*(set(map(int, np.unique(p))) for p in cores)) - {bg}
    _require(len(shared) == 1, "anchor identity is the common interior color")
    anchor = shared.pop()
    anchors = [_square(p, anchor) for p in cores]
    sources = [i for i, p in enumerate(cores) if np.any((p != bg) & (p != anchor))]
    _require(len(sources) == 1, "one complete source pattern required")
    source = sources[0]
    ports = []
    for panel in panels:
        port = {}
        for side in range(4):
            edge = _edge(panel, side)
            _require(not np.any(edge == anchor), "anchor cannot mark ports")
            sig = tuple(map(int, np.flatnonzero(edge != bg)))
            if sig:
                port[side] = sig
        _require(len(port) in (1, 2), "one or two ported sides per room")
        ports.append(port)
    _require(sum(len(p) == 1 for p in ports) == 2 and len(ports[source]) == 1,
             "source and last room are the two terminals")
    r, c, side = anchors[source]
    motif = {(int(y)-r, int(x)-c): int(cores[source][y, x])
             for y, x in np.argwhere(cores[source] != bg)}
    _require(_connected(motif), "source payload must be connected")
    order, crossings = _route(ports, source)
    flips, fy, fx = {}, False, False
    for j, room in enumerate(order):
        if j:
            if crossings[j-1] in (1, 3):
                fx = not fx
            else:
                fy = not fy
        flips[room] = (fy, fx)
    data.update(cores=cores, anchor=anchor, anchors=anchors, source=source,
                motif=motif, reference_side=side, ports=ports, order=order,
                crossings=crossings, flips=flips)
    return data


def _exact_raster(data, destination_side, fy, fx):
    """Refine source pixels, then require uniform destination-pixel areas.

    This supports sub-anchor detail and exact rational scales; no rounding,
    interpolation, or generator lattice information is used.
    """
    side, motif, bg = data['reference_side'], data['motif'], data['background']
    rs, cs = zip(*motif)
    top, left = min(rs)//side*side, min(cs)//side*side
    bottom = ((max(rs)+1+side-1)//side)*side
    right = ((max(cs)+1+side-1)//side)*side
    raw = np.full((bottom-top, right-left), bg, dtype=np.int8)
    for (r, c), color in motif.items():
        raw[r-top, c-left] = color
    refined = np.repeat(np.repeat(raw, destination_side, axis=0), destination_side, axis=1)
    h, w = refined.shape[0]//side, refined.shape[1]//side
    blocks = refined.reshape(h, side, w, side)
    low, high = blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))
    _require(np.array_equal(low, high), "scale cuts a color boundary")
    ar, ac = -top*destination_side//side, -left*destination_side//side
    if fy:
        low, ar = low[::-1, :], h-ar-destination_side
    if fx:
        low, ac = low[:, ::-1], w-ac-destination_side
    return low, (ar, ac)


def _paste(core, raster, top, left, bg, skip_color=None):
    r0, c0 = max(0, top), max(0, left)
    r1, c1 = min(core.shape[0], top+raster.shape[0]), min(core.shape[1], left+raster.shape[1])
    if r0 < r1 and c0 < c1:
        tile = raster[r0-top:r1-top, c0-left:c1-left]
        mask = tile != bg
        if skip_color is not None:
            mask &= tile != skip_color
        core[r0:r1, c0:c1][mask] = tile[mask]


def _render(grid, data, alternative=None):
    result, records = grid.copy(), []
    source = data['source']
    nc = data['layout'][1]
    for j, room in enumerate(data['order']):
        if room == source:
            continue
        fy, fx = data['flips'][room]
        if alternative == 'spatial':
            fy, fx = room//nc != source//nc, room%nc != source%nc
        elif alternative == 'spatial_parity':
            fy, fx = (room//nc-source//nc) % 2 == 1, (room%nc-source%nc) % 2 == 1
        elif alternative == 'no_reflection':
            fy = fx = False
        elif alternative == 'hop_parity':
            fy = fx = j % 2 == 1
        r, c, side = data['anchors'][room]
        requested = data['reference_side'] if alternative == 'no_scaling' else side
        raster, offset = _exact_raster(data, requested, fy, fx)
        r0, r1, c0, c1 = data['bounds'][room]
        core = result[r0+1:r1-1, c0+1:c1-1]
        anchor_mask = core == data['anchor'] if alternative == 'no_scaling' else None
        _paste(core, raster, r-offset[0], c-offset[1], data['background'],
               data['anchor'] if alternative == 'no_scaling' else None)
        if anchor_mask is not None:
            core[anchor_mask] = data['anchor']
        total = int(np.count_nonzero(raster != data['background']))
        visible = int(np.count_nonzero(core != data['background']))
        records.append(dict(room=room, lost=total-visible, total=total, visible=visible,
                            side=side, flips=(bool(fy), bool(fx))))
    return result, records


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from this input alone. Never mutates the input."""
    grid = _grid(input_grid)
    return _render(grid, _decode(grid))[0]


def _asymmetric(motif, side):
    return all({(side-1-r if fy else r, side-1-c if fx else c): color
                for (r, c), color in motif.items()} != motif
               for fy, fx in ((False, True), (True, False), (True, True)))


def _fine_geometry(motif, side, bg):
    for (r, c), color in motif.items():
        for dr, dc, offset in ((-1, 0, r), (1, 0, r+1), (0, -1, c), (0, 1, c+1)):
            if motif.get((r+dr, c+dc), bg) != color and offset % side:
                return True
    return False


def _choose(rng, options):
    return options[int(rng.integers(len(options)))]


def _make_motif(rng, a, span, anchor, details):
    motif = {(r, c): anchor for r in range(a) for c in range(a)}
    target = int(rng.integers(a*a+2, min(span*span, a*a+9)+1))
    for k in range(target-a*a):
        choices = []
        for q in sorted({q for p in motif for q in _neighbors(p)} - set(motif)):
            coords = list(motif) + [q]
            rs, cs = zip(*coords)
            if max(rs)-min(rs) < span and max(cs)-min(cs) < span:
                choices.append(q)
        if not choices:
            break
        motif[_choose(rng, choices)] = details[k % len(details)]
    return motif


def _lattice_raster(motif, a, unit, bg, fy, fx):
    """Construction renderer: paint each generated lattice rectangle directly."""
    transformed = {(a-1-r if fy else r, a-1-c if fx else c): color
                   for (r, c), color in motif.items()}
    rs, cs = zip(*transformed)
    top, left = min(rs), min(cs)
    raster = np.full(((max(rs)-top+1)*unit, (max(cs)-left+1)*unit), bg, dtype=np.int8)
    for (r, c), color in transformed.items():
        raster[(r-top)*unit:(r-top+1)*unit, (c-left)*unit:(c-left+1)*unit] = color
    return raster, (-top*unit, -left*unit)


def _placements(d, raster, offset, side, bg, clipped):
    candidates = []
    points = np.argwhere(raster != bg)
    for r in range(d-side+1):
        for c in range(d-side+1):
            positioned = points + [r-offset[0], c-offset[1]]
            visible = int(np.count_nonzero(np.all((positioned >= 0) & (positioned < d), axis=1)))
            if (visible < len(points)) == clipped and visible > side*side:
                candidates.append((r, c))
    return candidates


def _draw_links(rng, n, p, teaching):
    if teaching == 0:
        axes = [1, 0, 1]  # horizontal, vertical, horizontal
        if rng.random() < .5:
            axes = [1-x for x in axes]
    elif teaching == 1:
        axes = [1, 0, 0, 1, 0]
        if rng.random() < .5:
            axes = [1-x for x in axes]
    else:
        axes = list(map(int, rng.integers(0, 2, n-1)))
        if n >= 3:
            axes[0:2] = [0, 1] if rng.random() < .5 else [1, 0]
    sides = []
    for i, axis in enumerate(axes):
        sides.append(sides[-1] if i and axis == axes[i-1]
                     else _choose(rng, (1, 3) if axis else (0, 2)))
    singleton = [(i,) for i in range(1, p-1)]
    thick = [(i, i+1) for i in range(1, p-2)]
    separated = [s for s in combinations(range(1, p-1), 2) if s[1] > s[0]+1]
    signatures, used = [], set()
    repeat = teaching == 1 or (teaching is None and rng.random() < .35)
    repeat_at = 2 if teaching == 1 else next((i for i in range(1, n-1)
                                            if axes[i] == axes[i-1]), -1)
    for i, axis in enumerate(axes):
        if repeat and i == repeat_at:
            signatures.append(signatures[-1])
            continue
        pool = singleton if teaching == 0 else (
            thick if teaching == 1 and i == 1 else
            separated if teaching == 1 and i == 4 else
            singleton if teaching == 1 and i == 0 else
            _choose(rng, (singleton, thick, separated)))
        choices = [s for s in pool if (axis, s) not in used]
        if not choices:
            choices = [s for s in singleton+thick+separated if (axis, s) not in used]
        sig = _choose(rng, choices)
        signatures.append(sig)
        used.add((axis, sig))
    return sides, signatures


def _pack(panels, placement, layout, divider):
    nr, nc = layout
    p = panels[0].shape[0]
    grid = np.full((nr*(p+1)-1, nc*(p+1)-1), divider, dtype=np.int8)
    for slot, room in enumerate(placement):
        r, c = divmod(slot, nc)
        grid[r*(p+1):r*(p+1)+p, c*(p+1):c*(p+1)+p] = panels[room]
    return grid


def _build_pair(rng, teaching=None, test=False):
    if teaching == 0:
        n, layout, p = 4, (2, 2), int(rng.integers(8, 13))
    elif teaching == 1:
        n, layout, p = 6, _choose(rng, ((2, 3), (3, 2))), int(rng.integers(8, 10))
    elif test:
        n = _choose(rng, (6, 9, 9))
        layout = (3, 3) if n == 9 else _choose(rng, ((2, 3), (3, 2)))
        p = int(rng.integers(7, 10))
    else:
        n = _choose(rng, (2, 3, 4, 6))
        layouts = [(r, n//r) for r in range(1, n+1) if n % r == 0
                   and max(r, n//r)*8-1 <= 30]
        layout = _choose(rng, layouts)
        p = int(rng.integers(7, min(14, 31//max(layout)-1)+1))
    palette = list(map(int, rng.permutation(10)))
    bg, divider, anchor = palette[:3]
    detail_count = int(rng.integers(1, 4))
    details, port_colors = palette[3:3+detail_count], palette[3+detail_count:]
    a = 2 if teaching == 1 or rng.random() < .35 else 1
    d = p-2
    span = max(3, min(5, d//2))
    motif = _make_motif(rng, a, span, anchor, details)
    if teaching is not None and not _asymmetric(motif, a):
        return None
    sides, signatures = _draw_links(rng, n, p, teaching)
    panels = [np.full((p, p), bg, dtype=np.int8) for _ in range(n)]
    shared_ports = teaching == 1 or (teaching is None and rng.random() < .5)
    for i, (side, sig) in enumerate(zip(sides, signatures)):
        endpoint_colors = [port_colors[0], port_colors[0] if shared_ports else port_colors[1]]
        if rng.random() < .5:
            endpoint_colors.reverse()
        for room, edge_side, color in ((i, side, endpoint_colors[0]),
                                       (i+1, OPPOSITE[side], endpoint_colors[1])):
            for position in sig:
                panels[room][_border_point(p, edge_side, position)] = color
    units = [int(rng.integers(1, 3)) for _ in range(n)]
    if teaching is not None:
        units[1] = units[0]
        units[2] = 3-units[0]
    elif p >= 11 and a == 1 and rng.random() < .6:
        units[int(rng.integers(n))] = 3
    flags = [False]*n
    if teaching == 1:
        flags[2] = True  # visible clipping followed by a complete successor
    elif teaching is None:
        probability = float(rng.uniform(0, .65 if test else .4))
        flags = [False]+[bool(rng.random() < probability) for _ in range(n-1)]
    placements, full_rasters, fy, fx = [], [], False, False
    for i in range(n):
        if i:
            if sides[i-1] in (1, 3):
                fx = not fx
            else:
                fy = not fy
        raster, offset = _lattice_raster(motif, a, units[i], bg, fy, fx)
        side = a*units[i]
        options = _placements(d, raster, offset, side, bg, flags[i])
        if not options:
            return None
        r, c = _choose(rng, options)
        panels[i][1+r:1+r+side, 1+c:1+c+side] = anchor
        if i == 0:
            _paste(panels[i][1:-1, 1:-1], raster, r-offset[0], c-offset[1], bg)
        placements.append((r, c))
        full_rasters.append((raster, offset))
    output_panels = [q.copy() for q in panels]
    for i in range(1, n):
        raster, offset = full_rasters[i]
        r, c = placements[i]
        _paste(output_panels[i][1:-1, 1:-1], raster, r-offset[0], c-offset[1], bg)
    placement = list(map(int, rng.permutation(n)))
    return {'input': _pack(panels, placement, layout, divider),
            'output': _pack(output_panels, placement, layout, divider)}


def _signature_kind(sig):
    return 'single' if len(sig) == 1 else 'thick' if sig[-1]-sig[0]+1 == len(sig) else 'separated'


def _analyze(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    data = _decode(inp)
    expected, records = _render(inp, data)
    _require(np.array_equal(out, expected), "incorrect transported completion")
    _require(not np.array_equal(inp, out), "completion must be observable")
    # Protected routing layer, anchors, and the complete source stay intact.
    unchanged = inp != data['background']
    _require(np.array_equal(out[unchanged], inp[unchanged]), "existing cells changed")
    for record in records:
        _require(record['visible'] > record['side']**2, "each copy needs visible detail")
    alternatives = {name: not np.array_equal(_render(inp, data, name)[0], out)
                    for name in ('spatial', 'spatial_parity', 'no_reflection', 'hop_parity', 'no_scaling')}
    sigs = [data['ports'][room][side] for room, side in zip(data['order'], data['crossings'])]
    colored_mismatch = any(not np.array_equal(_edge(data['panels'][a], s)[list(sig)],
                                            _edge(data['panels'][b], OPPOSITE[s])[list(sig)])
                           for a, b, s, sig in zip(data['order'], data['order'][1:], data['crossings'], sigs))
    regrowth = any(records[i]['lost'] > 0 and records[i+1]['lost'] == 0
                   for i in range(len(records)-1))
    count_ports = [{side: (len(sig),) for side, sig in port.items()} for port in data['ports']]
    try:
        count_order, count_crossings = _route(count_ports, data['source'])
        count_shortcut_fails = (count_order, count_crossings) != (data['order'], data['crossings'])
    except AssertionError:
        count_shortcut_fails = True
    data.update(records=records, alternatives=alternatives, sigs=sigs,
                signature_kinds={_signature_kind(s) for s in sigs},
                colored_mismatch=colored_mismatch, regrowth=regrowth,
                count_shortcut_fails=count_shortcut_fails,
                fine_geometry=_fine_geometry(data['motif'], data['reference_side'], data['background']),
                asymmetric=_asymmetric(data['motif'], data['reference_side']))
    return data


def _teaching_ok(data, which):
    if not data['asymmetric'] or not all(data['alternatives'].values()):
        return False
    if which == 0:
        return (len(set(data['flips'].values())) == 4
                and all(r['lost'] == 0 for r in data['records'])
                and data['colored_mismatch']
                and len({s for _, _, s in data['anchors']}) > 1)
    return (data['regrowth'] and data['count_shortcut_fails'] and data['fine_geometry']
            and {'single', 'thick', 'separated'} <= data['signature_kinds'])


def _fallback_pair(rng, teaching, seen):
    """Small constructive relay, with a finite guaranteed fresh palette.

    This is a rare capacity fallback, not a stored puzzle or a solver lookup.
    Ninety background/divider assignments exceed the at-most-13 prior inputs.
    """
    which = 0 if teaching == 0 else 1
    n, layout, p = (4, (2, 2), 8) if which == 0 else (6, (2, 3), 8)
    palette = list(map(int, rng.permutation(10)))
    bg, divider, anchor, red, blue, port1, port2 = palette[:7]
    a = 1 if which == 0 else 2
    motif = {(r, c): anchor for r in range(a) for c in range(a)}
    motif.update({(-1, 0): red, (0, a): blue})
    sides = [1, 2, 1] if which == 0 else [1, 2, 2, 1, 2]
    sigs = [(1,), (2,), (3,)] if which == 0 else [(1,), (2, 3), (2, 3), (3,), (2, 4)]
    units = [1, 2, 1, 2] if which == 0 else [1, 2, 1, 2, 1, 2]
    placement = [2, 0, 3, 1] if which == 0 else [3, 0, 5, 2, 1, 4]
    panels = [np.full((p, p), bg, dtype=np.int8) for _ in range(n)]
    for i, (side, sig) in enumerate(zip(sides, sigs)):
        for room, edge_side, color in ((i, side, port1), (i+1, OPPOSITE[side], port2)):
            for position in sig:
                panels[room][_border_point(p, edge_side, position)] = color
    output_panels, fy, fx = [], False, False
    for i in range(n):
        if i:
            if sides[i-1] in (1, 3):
                fx = not fx
            else:
                fy = not fy
        raster, offset = _lattice_raster(motif, a, units[i], bg, fy, fx)
        side = a*units[i]
        r, c = _placements(p-2, raster, offset, side, bg, which == 1 and i == 2)[0]
        panels[i][1+r:1+r+side, 1+c:1+c+side] = anchor
        if i == 0:
            _paste(panels[i][1:-1, 1:-1], raster, r-offset[0], c-offset[1], bg)
        completed = panels[i].copy()
        _paste(completed[1:-1, 1:-1], raster, r-offset[0], c-offset[1], bg)
        output_panels.append(completed)
    base = {'input': _pack(panels, placement, layout, divider),
            'output': _pack(output_panels, placement, layout, divider)}
    _require(_teaching_ok(_analyze(base), which), 'constructive fallback witness')
    for new_bg, new_divider in permutations(range(10), 2):
        old = [bg, divider]+[c for c in range(10) if c not in (bg, divider)]
        new = [new_bg, new_divider]+[c for c in range(10) if c not in (new_bg, new_divider)]
        mapping = np.zeros(10, dtype=np.int8)
        mapping[old] = new
        candidate = {k: mapping[v] for k, v in base.items()}
        if _key(candidate['input']) not in seen:
            return candidate
    raise AssertionError('finite fresh-palette space exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    num_train, num_test = _integer(num_train, 'num_train'), _integer(num_test, 'num_test')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('require 2..10 train, 1..4 test, and num_train > num_test')
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            teaching = i if split == 'train' and i < 2 else None
            for _ in range(300):
                pair = _build_pair(rng, teaching, split == 'test')
                if pair is None or _key(pair['input']) in seen:
                    continue
                try:
                    data = _analyze(pair)
                except AssertionError:
                    continue
                if teaching is not None and not _teaching_ok(data, teaching):
                    continue
                break
            else:
                pair = _fallback_pair(rng, teaching, seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
        if split == 'train':
            rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'pair lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'pair counts')
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            inp = _grid(pair['input'])
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            analyses[split].append(_analyze(pair))
    training = analyses['train']
    _require(any(_teaching_ok(d, 0) for d in training), 'complete scaled four-orientation witness')
    _require(any(_teaching_ok(d, 1) for d in training), 'port and clipping/regrowth witness')
    demonstrated = set.union(*(d['signature_kinds'] for d in training))
    _require(all(d['signature_kinds'] <= demonstrated for d in analyses['test']), 'port types not demonstrated')
    _require(all(len(d['panels']) >= 6 for d in analyses['test']), 'test route length shift')


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def self_test() -> None:
    """Bounded 42-seed semantic/variation checks, plus hostile mutations."""
    puzzles, shapes, motifs, layouts, roles, sides, ratios = set(), set(), set(), set(), set(), set(), set()
    counts, flips, fine, clipped, repeats, source_slots = set(), set(), 0, 0, 0, set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.add(_puzzle_key(puzzle))
        if seed < 3:
            _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), 'determinism')
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                data = _analyze(pair)
                _require(np.array_equal(solve(pair['input']), pair['output']), 'solve agreement')
                shapes.add(pair['input'].shape)
                counts.add(len(data['panels']))
                layouts.add(data['layout'])
                roles.add((data['background'], data['divider'], data['anchor']))
                source_slots.add((data['layout'], data['source']))
                sides.update(s for _, _, s in data['anchors'])
                flips.update(data['flips'].values())
                ratios.update((s//int(np.gcd(s, data['reference_side'])),
                               data['reference_side']//int(np.gcd(s, data['reference_side'])))
                              for _, _, s in data['anchors'])
                rs, cs = zip(*data['motif'])
                normalized = np.full((max(rs)-min(rs)+1, max(cs)-min(cs)+1), 0, dtype=np.int8)
                colors = {data['anchor']: 1}
                for (r, c), color in sorted(data['motif'].items()):
                    normalized[r-min(rs), c-min(cs)] = colors.setdefault(color, len(colors)+1)
                motifs.add(_key(normalized))
                fine += data['fine_geometry']
                clipped += any(r['lost'] > 0 for r in data['records'])
                keys = [(s % 2, sig) for s, sig in zip(data['crossings'], data['sigs'])]
                repeats += len(set(keys)) < len(keys)
    _require(len(puzzles) == 42 and len(shapes) >= 15 and len(motifs) >= 100, 'structural diversity')
    _require({2, 3, 4, 6, 9} <= counts and {(2, 2), (2, 3), (3, 2), (3, 3)} <= layouts, 'route/layout diversity')
    _require(len(roles) >= 80 and len(sides) >= 4 and len(ratios) >= 4, 'palette/scale diversity')
    _require(len(source_slots) >= 20 and len(flips) == 4 and fine >= 42, 'source/geometry diversity')
    _require(clipped >= 42 and repeats >= 42, 'clipping and global routing')
    for train in range(2, 11):
        for test in range(1, min(4, train-1)+1):
            p = generate_puzzle(123, train, test)
            _require(len(p['train']) == train and len(p['test']) == test, 'explicit counts')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    original = generate_puzzle(42, 2, 1)
    fallback_seen, fallback_puzzle = set(), {'train': [], 'test': []}
    for split, which in (('train', 0), ('train', 1), ('test', None)):
        pair = _fallback_pair(np.random.default_rng(7), which, fallback_seen)
        fallback_seen.add(_key(pair['input']))
        fallback_puzzle[split].append(pair)
    validate_generated_puzzle(fallback_puzzle)
    bad = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in original[s]]
           for s in original}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('changed border accepted')
    # Ambiguous route: two interchangeable same-signature straight-throughs.
    ports = [{1: (2,)}, {3: (2,), 1: (2,)}, {3: (2,), 1: (2,)}, {3: (2,)}]
    try:
        _route(ports, 0)
    except AssertionError:
        pass
    else:
        raise AssertionError('interchangeable relays accepted')
    # Exact fine-detail source supports scale 2, but not an ambiguous 3/2.
    example = dict(reference_side=2, background=0,
                   motif={(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1, (-1, 0): 2})
    _exact_raster(example, 4, True, True)
    try:
        _exact_raster(example, 3, False, False)
    except AssertionError:
        pass
    else:
        raise AssertionError('mixed-color destination pixel accepted')


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(',', ':')))
