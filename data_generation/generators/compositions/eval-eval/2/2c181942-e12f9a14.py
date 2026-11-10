"""Rendezvous assembly: surveying-ray encounters materialize docking sockets.

Only NumPy and the standard library are used. solve() reads the input alone.
See task_spec.md for the geometric grammar, scope and evidence limitations.
"""
from __future__ import annotations

import json
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N, E, S, W
NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
PORTS = {0: ((-1, 0), (-1, 1)), 1: ((0, 2), (1, 2)),
         2: ((2, 0), (2, 1)), 3: ((0, -1), (1, -1))}
RING = tuple((r, c) for r in range(-1, 3) for c in range(-1, 3)
             if r in (-1, 2) or c in (-1, 2))
HOST_SEEDS = (0, 422430292, 134106014, 122058, 978446202, 60112649,
              354487085, 143201285, 141466718, 601690690, 475769483,
              245957234, 881348132, 43481311, 509767163, 104808343,
              17408759, 24059210, 63162033, 746952934, 492761814,
              641679494, 311554368, 159850297, 18001289, 208952942,
              661675995, 723649437, 738857619, 980878457, 469001316,
              645747329, 542585568, 881315182, 234918834, 957352428,
              930593760, 803661699, 815023630, 513140669, 740160650,
              893406504)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'integer cells required')
    _require(bool(np.all((a >= 0) & (a <= 9))), 'colors must be 0..9')
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _connected(mask, diagonal=True):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    stack = [cells.pop()]
    neighbors = NEIGHBORS if diagonal else DIRS
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            q = r + dr, c + dc
            if q in cells:
                cells.remove(q)
                stack.append(q)
    return not cells


def _attachments(mask):
    """All centered two-cell terminal edges, modulo equivalent final masks."""
    options = {}
    for t in range(4):
        m = np.rot90(mask, t).copy()
        h, w = m.shape
        if h < 2 or w < 2 or w % 2:
            continue
        neck = np.zeros(w, dtype=bool)
        neck[w // 2 - 1:w // 2 + 1] = True
        if np.array_equal(m[-1], neck):
            k = _key(m)
            if k not in options:
                options[k] = [m, []]
            options[k][1].append(t)
    return list(options.values())


def _canonical(mask):
    options = _attachments(mask)
    _require(_connected(mask) and len(options) == 1, 'ambiguous or missing attachment')
    return options[0]


def _expand(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _footprint(core):
    r, c = core
    return {(r + dr, c + dc) for dr in range(-1, 3) for dc in range(-1, 3)}


def _parse(a, allow_empty=False):
    h, w = a.shape
    bg = int(np.bincount(a.ravel(), minlength=10).argmax())
    _require(np.count_nonzero(a == bg) > a.size // 2, 'background must occupy a majority')
    candidate = (a[:-1, :-1] != bg) & (a[:-1, :-1] == a[1:, :-1])
    candidate &= (a[:-1, :-1] == a[:-1, 1:]) & (a[:-1, :-1] == a[1:, 1:])
    sources = []
    for rr, cc in np.argwhere(candidate):
        r, c = int(rr), int(cc)
        if not (1 <= r <= h - 3 and 1 <= c <= w - 3):
            continue
        color = int(a[r, c])
        values = [int(a[r + dr, c + dc]) for dr, dc in RING]
        foreground = [v for v in values if v != bg]
        if len(foreground) not in (10, 11) or len(set(foreground)) != 1:
            continue
        structure = foreground[0]
        if structure == color:
            continue
        gaps = [o for o, v in zip(RING, values) if v == bg]
        reverse = {o: side for side in PORTS for o in PORTS[side]}
        if any(o not in reverse for o in gaps):
            continue
        sources.append(dict(core=(r, c), color=color, structure=structure,
                            ports=[(o, reverse[o]) for o in gaps]))
    _require(2 <= len(sources) <= 4, 'need two to four fully visible launchers')
    _require(len({s['structure'] for s in sources}) == 1, 'common ring role required')
    _require(len({s['color'] for s in sources}) == len(sources), 'source colors are unique')
    sources.sort(key=lambda s: s['core'])
    structure = sources[0]['structure']
    source_cells, fixed, rays = set(), np.full_like(a, bg), []
    for sid, s in enumerate(sources):
        footprint = _footprint(s['core'])
        _require(not (source_cells & footprint), 'launcher footprints overlap')
        source_cells |= footprint
        r, c = s['core']
        fixed[r - 1:r + 3, c - 1:c + 3] = a[r - 1:r + 3, c - 1:c + 3]
        for offset, side in s['ports']:
            rays.append(dict(source=sid, color=s['color'], direction=side,
                             start=(r + offset[0], c + offset[1])))
    _require(not any(int(a[q]) == structure and q not in source_cells
                     for q in map(tuple, np.argwhere(a == structure))), 'unaccounted structural cells')
    remaining = {(int(r), int(c)) for r, c in np.argwhere(a != bg)} - source_cells
    prototypes = {}
    groups = []
    while remaining:
        start = min(remaining)
        color = int(a[start])
        remaining.remove(start)
        todo, cells = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr, dc in NEIGHBORS:
                q = r + dr, c + dc
                if q in remaining and int(a[q]) == color:
                    remaining.remove(q)
                    cells.add(q)
                    todo.append(q)
        _require(color not in prototypes, 'only one loose prototype per color')
        _require(not (_expand(cells) & source_cells), 'prototype touches a launcher')
        _require(not any(_expand(cells) & old for old in groups), 'loose prototypes touch')
        groups.append(cells)
        coords = np.array(sorted(cells), dtype=int)
        low, high = coords.min(axis=0), coords.max(axis=0)
        mask = np.zeros(tuple(high - low + 1), dtype=bool)
        mask[tuple((coords - low).T)] = True
        prototypes[color] = dict(mask=mask, cells=cells)
    _require((0 if allow_empty else 1) <= len(prototypes) <= 6,
             'need one to six separated prototypes')
    return dict(background=bg, structure=structure, sources=sources,
                source_cells=source_cells, fixed=fixed, rays=rays, prototypes=prototypes)


def _cross(a, b, shape):
    """Exact cardinal spatial intersection, measured from each port separately."""
    u, v = DIRS[a['direction']], DIRS[b['direction']]
    p, q = a['start'], b['start']
    det = u[0] * v[1] - u[1] * v[0]
    if not det:
        # Parallel separated lines are fine. Collinear forward overlap is not.
        same_line = (p[1] == q[1]) if u[0] else (p[0] == q[0])
        if same_line:
            delta = (q[0] - p[0]) * u[0] + (q[1] - p[1]) * u[1]
            if u == v or delta >= 0:
                raise ValueError('collinear rays or head-on cancellation')
        return None
    dr, dc = q[0] - p[0], q[1] - p[1]
    t = (dr * v[1] - dc * v[0]) // det
    s = (dr * u[1] - dc * u[0]) // det
    point = (p[0] + t * u[0], p[1] + t * u[1])
    if t <= 0 or s <= 0 or not (0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]):
        return None
    return point, t, s


def _encounters(info, shape, equal_clock=False):
    rays = info['rays']
    live, events = set(range(len(rays))), []
    while live:
        candidates = []
        for i in sorted(live):
            for j in sorted(live):
                if j <= i:
                    continue
                hit = _cross(rays[i], rays[j], shape)
                if hit is not None and (not equal_clock or hit[1] == hit[2]):
                    candidates.append((i, j, *hit))
        if not candidates:
            break
        nearest = {}
        for i, j, p, t, s in candidates:
            nearest[i] = min(nearest.get(i, 100), t)
            nearest[j] = min(nearest.get(j, 100), s)
        ready = [e for e in candidates if e[3] == nearest[e[0]] and e[4] == nearest[e[1]]]
        _require(bool(ready), 'cyclic spatial encounter order')
        for ray in live:
            hits = sum((i == ray and t == nearest.get(ray)) or
                       (j == ray and s == nearest.get(ray)) for i, j, p, t, s in candidates)
            _require(hits <= 1, 'tied or multiway first encounter')
        i, j, point, t, s = min(ready, key=lambda e: (max(e[3:]), sum(e[3:]), e[0], e[1]))
        _require(rays[i]['source'] != rays[j]['source'], 'one launcher cannot meet itself')
        u, v = DIRS[rays[i]['direction']], DIRS[rays[j]['direction']]
        resultant = (u[0] + v[0], u[1] + v[1])
        r, c = point
        hub = (r if resultant[0] > 0 else r - 3,
               c if resultant[1] > 0 else c - 3)
        events.append(dict(rays=(i, j), point=point, distances=(t, s), hub=hub,
                           resultant=resultant,
                           sides=((rays[i]['direction'] + 2) % 4,
                                  (rays[j]['direction'] + 2) % 4),
                           colors=(rays[i]['color'], rays[j]['color'])))
        live -= {i, j}
    # Surveying routes cannot run through fixed launcher cells. Loose pieces
    # are prototypes, not barriers; their locations have no role in tracing.
    end = {i: e['point'] for e in events for i in e['rays']}
    for i, ray in enumerate(rays):
        r, c = ray['start']
        dr, dc = DIRS[ray['direction']]
        while 0 <= r < shape[0] and 0 <= c < shape[1]:
            _require(int(info['fixed'][r, c]) == info['background'], 'ray crosses a fixed launcher')
            if (r, c) == end.get(i):
                break
            r, c = r + dr, c + dc
    return events


def _bar(side, hub):
    r, c = hub
    return (((r, c + 1), (r, c + 2)), ((r + 1, c + 3), (r + 2, c + 3)),
            ((r + 3, c + 1), (r + 3, c + 2)), ((r + 1, c), (r + 2, c)))[side]


def _docked(mask, side, hub):
    marker = np.zeros_like(mask)
    marker[-1, mask.shape[1] // 2 - 1:mask.shape[1] // 2 + 1] = True
    oriented, neck = np.rot90(mask, -side), np.rot90(marker, -side)
    dr, dc = DIRS[side]
    target = np.array(_bar(side, hub), dtype=int) + (dr, dc)
    shift = target.min(axis=0) - np.argwhere(neck).min(axis=0)
    return {tuple(map(int, q)) for q in np.argwhere(oriented) + shift}


def _render(info, events, shape, omit_shapes=False):
    out = info['fixed'].copy()
    bg, prototypes = info['background'], info['prototypes']
    reserved = set(info['source_cells'])
    hub_cells = []
    for event in events:
        r, c = event['hub']
        cells = {(r + dr, c + dc) for dr in range(4) for dc in range(4)}
        _require(all(0 <= rr < shape[0] and 0 <= cc < shape[1] for rr, cc in cells), 'socket clipping')
        _require(not (reserved & cells), 'socket overlaps another socket or launcher')
        reserved |= cells
        hub_cells.append(cells)
    used = []
    for event in events:
        cluster = set()
        for color, side in zip(event['colors'], event['sides']):
            bar = set(_bar(side, event['hub']))
            for q in bar:
                out[q] = color
            cluster |= bar
            if color not in prototypes or omit_shapes:
                continue
            mask, turns = _canonical(prototypes[color]['mask'])
            cells = _docked(mask, side, event['hub'])
            _require(all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells), 'cargo clipping')
            _require(not (reserved & cells), 'cargo overlaps socket or launcher')
            _require(all(int(out[q]) == bg for q in cells), 'assembled cargo overlap')
            for q in cells:
                out[q] = color
            cluster |= cells
            used.append(dict(color=color, side=side, canonical=mask, cells=cells,
                             rotations={(t - side) % 4 for t in turns}))
        event['cluster'] = cluster
    # Separate nodes by a background cell, while allowing adjacent arms of
    # one socket. This preserves readable repeated silhouettes.
    clusters = [e['cluster'] | hc for e, hc in zip(events, hub_cells)]
    for i, cluster in enumerate(clusters):
        for other in clusters[:i]:
            _require(not (_expand(cluster) & other), 'assembly sites touch')
    return out, used


def _inspect(a):
    info = _parse(a)
    events = _encounters(info, a.shape)
    _require(1 <= len(events) <= 2, 'one or two rendezvous required')
    out, used = _render(info, events, a.shape)
    _require(bool(used), 'each example must assemble at least one silhouette')
    info.update(events=events, used=used)
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive colors, routes, socket positions and attachments from this grid."""
    return _inspect(_grid(input_grid))[0]


def _mask(*rows):
    return np.array([[x == '#' for x in row] for row in rows], dtype=bool)


def _make_shape(rng, directional=False):
    for _ in range(60):
        width = int(rng.choice((2, 4, 6), p=(.18, .5, .32)))
        depth = int(rng.integers(2, 6))
        middle = width // 2
        m = np.zeros((depth, width), dtype=bool)
        m[:, middle - 1:middle + 1] = True
        stem = int(rng.integers(1, depth))
        kind = int(rng.integers(4))
        if width > 2 and kind == 0:
            m[:depth - stem] = True
        elif width > 2 and kind == 1:
            reach = min(width // 2 - 1, depth - 1)
            m[:] = False
            m[reach:, middle - 1:middle + 1] = True
            for r in range(reach):
                k = reach - r
                m[r, middle - 1 - k] = m[r, middle + k] = True
                if rng.random() < .35:
                    m[r, middle - 1 - k:middle + k + 1] = True
        elif width > 2:
            for r in range(depth - stem):
                left = int(rng.integers(0, middle))
                right = int(rng.integers(middle + 1, width + 1))
                m[r, left:right] = True
            m[0, 0] = m[0, -1] = True
            for r, c in np.argwhere(m[:depth - stem]):
                if c not in (middle - 1, middle) and rng.random() < .18:
                    m[r, c] = False
        rr, cc = np.nonzero(m)
        m = m[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
        options = _attachments(m)
        if _connected(m) and len(options) == 1 and (not directional or len(options[0][1]) == 1):
            return options[0][0]
    return _mask('####', '.##.')


def _build_sources(shape, specs):
    a = np.zeros(shape, dtype=np.int8)
    occupied = set()
    for spec in specs:
        r, c = spec['core']
        footprint = _footprint((r, c))
        _require(all(0 <= rr < shape[0] and 0 <= cc < shape[1] for rr, cc in footprint), 'source boundary')
        _require(not (occupied & footprint), 'source overlap')
        occupied |= footprint
        a[r - 1:r + 3, c - 1:c + 3] = 8
        a[r:r + 2, c:c + 2] = spec['color']
        for dr, dc in spec['ports']:
            a[r + dr, c + dc] = 0
    return a


def _spec_at_port(point, side, lane, color):
    offset = PORTS[side][lane]
    return dict(core=(point[0] - offset[0], point[1] - offset[1]), color=color, ports=[offset])


def _source_geometry(rng, mode, orientation, fork_style=None):
    # Construct relative geometry first. Its actual occupied bounds determine
    # the returned canvas; this avoids impossible edge-conditioned proposals.
    shared = (30, 30)
    if mode == 'single':
        shared_sides = [orientation]
        partner_sides = [(orientation + int(rng.choice((1, 3)))) % 4]
    else:
        style = fork_style if fork_style is not None else str(rng.choice(('adjacent', 'opposite', 'parallel')))
        if style == 'opposite':
            shared_sides = [orientation, (orientation + 2) % 4]
            partner_sides = [(orientation + 1) % 4, (orientation + 3) % 4]
        elif style == 'adjacent':
            shared_sides = [orientation, (orientation + 1) % 4]
            partner_sides = [(orientation + 1) % 4, (orientation + 2) % 4]
        else:
            shared_sides = [orientation, orientation]
            partner_sides = [(orientation + 1) % 4, (orientation + 3) % 4]
            # Nearer partner approaches the first lane from its outside.
            if orientation in (0, 3):
                partner_sides.reverse()
    ports = []
    for k, side in enumerate(shared_sides):
        ports.append(PORTS[side][k if len(set(shared_sides)) == 1 and len(shared_sides) == 2
                                else int(rng.integers(2))])
    specs = [dict(core=shared, color=1, ports=ports)]
    if len(shared_sides) == 2 and len(set(shared_sides)) == 1:
        near = int(rng.integers(5, 9))
        distances = [near, near + int(rng.integers(7, 13))]
        rng.shuffle(distances)
    else:
        distances = list(map(int, rng.integers(5, 12, len(ports))))
    for k, (offset, side, partner) in enumerate(zip(ports, shared_sides, partner_sides)):
        distance = distances[k]
        travel = int(rng.integers(5, 12))
        u, v = DIRS[side], DIRS[partner]
        start = (shared[0] + offset[0], shared[1] + offset[1])
        point = (start[0] + distance * u[0], start[1] + distance * u[1])
        partner_port = (point[0] - travel * v[0], point[1] - travel * v[1])
        specs.append(_spec_at_port(partner_port, partner, int(rng.integers(2)), k + 2))
    return specs


def _place_prototypes(rng, a, prototypes, specs):
    blocked = np.zeros(a.shape, dtype=bool)
    for spec in specs:
        for r, c in _expand(_footprint(spec['core'])):
            if 0 <= r < a.shape[0] and 0 <= c < a.shape[1]:
                blocked[r, c] = True
    order = list(prototypes)
    rng.shuffle(order)
    order.sort(key=lambda color: -prototypes[color].size)
    for color in order:
        m = prototypes[color]
        cells = np.argwhere(m)
        _require(all(x <= y for x, y in zip(m.shape, a.shape)), 'prototype too large')
        tops, lefts = np.indices((a.shape[0] - m.shape[0] + 1,
                                 a.shape[1] - m.shape[1] + 1))
        rr = tops.ravel()[:, None] + cells[:, 0]
        cc = lefts.ravel()[:, None] + cells[:, 1]
        candidates = np.flatnonzero(~np.any(blocked[rr, cc], axis=1))
        _require(len(candidates) > 0, 'prototype packing')
        index = int(rng.choice(candidates))
        a[rr[index], cc[index]] = color
        for r, c in zip(rr[index], cc[index]):
            blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    return a


def _try_pair(rng, mode, orientation, palette, forced_turns=None,
              idle=False, missing=False, fork_style=None, test=False):
    specs = _source_geometry(rng, mode, orientation, fork_style)
    # Teaching slots use directional masks, so rotations are visible.
    forced_turns = forced_turns or {}
    prototypes = {}
    for spec in specs:
        color = spec['color']
        canonical = _make_shape(rng, directional=color in forced_turns)
        if color in forced_turns:
            # Target side for the first port is visibly opposite its heading.
            side = next(s for s in PORTS if spec['ports'][0] in PORTS[s])
            destination = (side + 2) % 4
            t = (-destination - forced_turns[color]) % 4
        else:
            t = int(rng.integers(4))
        prototypes[color] = np.rot90(canonical, t).copy()
    if missing:
        del prototypes[int(rng.choice((2, 3) if mode == 'fork' else (1, 2)))]
    # Forecast the actual assembly before choosing a <=30 canvas. These
    # temporary construction arrays are never returned as ARC examples.
    staging = _build_sources((64, 64), specs)
    staged = _parse(staging, allow_empty=True)
    staged['prototypes'] = {c: dict(mask=m) for c, m in prototypes.items()}
    events = _encounters(staged, staging.shape)
    _require(len(events) == (2 if mode == 'fork' else 1), 'unplanned encounter count')
    _render(staged, events, staging.shape)
    bounds = set(staged['source_cells'])
    for event in events:
        r, c = event['hub']
        bounds |= {(r + dr, c + dc) for dr in range(4) for dc in range(4)}
        bounds |= event['cluster']
    coords = np.array(sorted(bounds), dtype=int)
    low, high = coords.min(axis=0), coords.max(axis=0)
    extent = high - low + 1
    _require(bool(np.all(extent <= 30)), 'assembly extent exceeds canvas')
    shape = tuple(max(int(n), int(rng.integers(16, 31))) for n in extent)
    shift = np.array([int(rng.integers(n - e + 1)) for n, e in zip(shape, extent)]) - low
    specs = [dict(s, core=tuple(map(int, np.array(s['core']) + shift))) for s in specs]
    if idle:
        # A boundary-facing extra emitter never reaches another ray. Try
        # locations against the real assembly rather than an imaginary box.
        for _ in range(60):
            side = int(rng.integers(4))
            r, c = int(rng.integers(1, shape[0] - 2)), int(rng.integers(1, shape[1] - 2))
            if side == 0:
                r = 1
            elif side == 1:
                c = shape[1] - 3
            elif side == 2:
                r = shape[0] - 3
            else:
                c = 1
            extra = dict(core=(r, c), color=4, ports=[PORTS[side][int(rng.integers(2))]])
            try:
                trial = _parse(_build_sources(shape, specs + [extra]), allow_empty=True)
                trial['prototypes'] = {col: dict(mask=m) for col, m in prototypes.items()}
                _render(trial, _encounters(trial, shape), shape)
            except ValueError:
                continue
            specs.append(extra)
            prototypes[4] = np.rot90(_make_shape(rng), int(rng.integers(4))).copy()
            break
        else:
            raise ValueError('idle launcher placement exhausted')
    distractor_count = int(rng.integers(1, 3)) if not idle else 1
    for color in range(5, 5 + distractor_count):
        canonical = (_mask('.##', '###', '##.') if rng.random() < .35
                     else _make_shape(rng))
        prototypes[color] = np.rot90(canonical, int(rng.integers(4))).copy()
    a = _place_prototypes(rng, _build_sources(shape, specs), prototypes, specs)
    out, info = _inspect(a)
    expected_count = 2 if mode == 'fork' else 1
    _require(len(info['events']) == expected_count, 'unplanned encounter count')
    if mode == 'fork':
        reused = {color for color in info['prototypes']
                  if sum(color in e['colors'] for e in info['events']) == 2}
        _require(bool(reused), 'fork must reuse a shared prototype')
    _require(any(e['distances'][0] != e['distances'][1] for e in info['events']), 'unequal distances required')
    if idle:
        _require(all(4 not in e['colors'] for e in info['events']), 'idle source interacted')
    if test:
        # Shift is structural recombination; sizes and literal palettes overlap.
        _require(any(p['canonical'].shape[0] >= 3 for p in info['used']), 'test needs a nontrivial silhouette')
    lookup = np.array(palette, dtype=np.int8)
    a, out = lookup[a], lookup[out]
    _require(np.array_equal(solve(a), out), 'recoloring changed semantics')
    return dict(input=a, output=out)


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' outside allowed range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, 2, 10, 'num_train'), _count(num_test, 1, 4, 'num_test')
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    base_palette = list(map(int, rng.permutation(10)))
    orientation = int(rng.integers(4))
    configs = []
    # Two forks expose all four socket quadrants. The topology itself varies
    # across puzzles; translation and color are not the only changing factors.
    teaching_style = str(rng.choice(('adjacent', 'opposite', 'parallel')))
    offset = 1 if teaching_style == 'opposite' else 2
    configs.append(dict(mode='fork', orientation=orientation, fork_style=teaching_style,
                        forced_turns={1: 0, 2: 1, 3: 2}, idle=True))
    configs.append(dict(mode='fork', orientation=(orientation + offset) % 4, fork_style=teaching_style,
                        forced_turns={1: 3, 2: 0, 3: 1}, idle=False))
    for _ in range(nt - 2):
        configs.append(dict(mode=str(rng.choice(('single', 'fork'))),
                            orientation=int(rng.integers(4)), idle=rng.random() < .3,
                            missing=rng.random() < .2, fork_style=teaching_style))
    rng.shuffle(configs)
    # Hold out a fork arrangement, not an undemonstrated collision law.
    held_out = [s for s in ('adjacent', 'opposite', 'parallel') if s != teaching_style]
    missing_taught = any(c.get('missing', False) for c in configs)
    tests = [dict(mode='fork', orientation=int(rng.integers(4)),
                  fork_style=str(rng.choice(held_out)),
                  idle=rng.random() < .25, missing=missing_taught and rng.random() < .3, test=True)
             for _ in range(ne)]
    puzzle, seen = dict(train=[], test=[]), set()
    for split, plan in (('train', configs), ('test', tests)):
        for index, config in enumerate(plan):
            palette = base_palette if rng.random() < .4 else list(map(int, rng.permutation(10)))
            # Training visibly changes palettes even under the smallest count.
            if split == 'train' and index == 1:
                palette = list(map(int, rng.permutation(10)))
                previous = puzzle['train'][0]['input']
                previous_bg = int(np.bincount(previous.ravel(), minlength=10).argmax())
                while palette[0] == previous_bg:
                    palette = list(map(int, rng.permutation(10)))
            last_error = None
            for attempt in range(600):
                try:
                    pair = _try_pair(rng, palette=palette, **config)
                    k = _key(pair['input'])
                    if k in seen:
                        continue
                    seen.add(k)
                    puzzle[split].append(pair)
                    break
                except ValueError as error:
                    last_error = error
            else:
                raise RuntimeError(f'bounded scene construction exhausted: {config}: {last_error}')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    out, info = _inspect(_grid(pair['input']))
    _require(np.array_equal(out, pair['output']), 'incorrect assembly output')
    active = {c for e in info['events'] for c in e['colors']}
    idle = {s['color'] for s in info['sources']} - active
    sockets_only, _ = _render(info, info['events'], out.shape, omit_shapes=True)
    try:
        clock_events = _encounters(info, out.shape, equal_clock=True)
        clock_out, _ = _render(info, clock_events, out.shape)
        clock_changes = not np.array_equal(clock_out, out)
    except ValueError:
        # A wrong clock law can also make a formerly consumed ray hit a
        # launcher or generate an overlapping socket; that law is invalid.
        clock_changes = True
    fork_types = set()
    for source in info['sources']:
        sides = [side for _, side in source['ports']]
        if len(sides) == 2:
            delta = (sides[0] - sides[1]) % 4
            fork_types.add('parallel' if delta == 0 else 'opposite' if delta == 2 else 'adjacent')
    return dict(shape=out.shape, background=info['background'], structure=info['structure'],
                source_count=len(info['sources']), event_count=len(info['events']),
                prototype_count=len(info['prototypes']), copy_count=len(info['used']),
                fork_types=fork_types,
                quadrants={e['resultant'] for e in info['events']},
                turns=set().union(*(p['rotations'] for p in info['used'] if len(p['rotations']) == 1)),
                unequal=any(e['distances'][0] != e['distances'][1] for e in info['events']),
                idle_witness=any(c in info['prototypes'] and len(_attachments(info['prototypes'][c]['mask'])) == 1
                                 for c in idle),
                distractors=len(set(info['prototypes']) - active),
                missing=len(active - set(info['prototypes'])),
                reused=any(sum(c in e['colors'] for e in info['events']) == 2 for c in info['prototypes']),
                shapes=[(p['canonical'].shape, p['canonical'].tobytes()) for p in info['used']],
                diagonal=any(not _connected(p['canonical'], False) for p in info['used']),
                symmetric=any(len(_attachments(info['prototypes'][p['color']]['mask'])[0][1]) > 1
                              for p in info['used']),
                asymmetric=any(not np.array_equal(p['canonical'], np.fliplr(p['canonical'])) for p in info['used']),
                source_port_sides=tuple(sorted(tuple(sorted(side for _, side in s['ports'])) for s in info['sources'])),
                encounter_points=tuple(e['point'] for e in info['events']),
                encounter_distances=tuple(e['distances'] for e in info['events']),
                clock_changes=clock_changes,
                omit_a_changes=not np.array_equal(sockets_only, out),
                omit_b_changes=not np.array_equal(info['fixed'], out))


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists required')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'illegal example counts')
    features, seen = dict(train=[], test=[]), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'canvas must be preserved')
            _require(_key(a) not in seen, 'duplicate input')
            seen.add(_key(a))
            out, info = _inspect(a)
            _require(np.array_equal(out, b), 'incorrect solved label')
            _require(np.array_equal(a[tuple(np.array(sorted(info['source_cells'])).T)],
                                   b[tuple(np.array(sorted(info['source_cells'])).T)]), 'launcher changed')
            # Each copy retains the complete cell count and exact rotated mask.
            for p in info['used']:
                _require(len(p['cells']) == int(p['canonical'].sum()), 'shape cell count changed')
                _require(all(int(b[q]) == p['color'] for q in p['cells']), 'payload color changed')
            f = _features(pair)
            _require(f['omit_a_changes'] and f['omit_b_changes'], 'both inherited concepts must affect each answer')
            features[split].append(f)
    train = features['train']
    _require(set().union(*(f['quadrants'] for f in train)) == {(-1, -1), (-1, 1), (1, -1), (1, 1)},
             'training must expose each resultant quadrant')
    _require(set().union(*(f['turns'] for f in train)) == set(range(4)), 'training rotation coverage')
    _require(any(f['idle_witness'] for f in train), 'teach collision-conditioned selection')
    _require(any(f['reused'] for f in train), 'teach prototype reuse')
    _require(all(f['unequal'] and f['clock_changes'] for f in train), 'teach spatial encounters')
    _require(len({(f['background'], f['structure']) for f in train}) >= 2, 'teach local color roles')
    _require(not any(f['missing'] for f in features['test']) or any(f['missing'] for f in train),
             'teach missing-prototype behavior before testing it')


def _puzzle_key(p):
    return tuple((_key(pair['input']), _key(pair['output'])) for s in ('train', 'test') for pair in p[s])


def self_test() -> dict:
    # Compact, hand-calculated geometry regressions, not sampled metadata.
    rays = [dict(start=(7, 2), direction=1, color=1, source=0),
            dict(start=(13, 10), direction=0, color=2, source=1)]
    assert _cross(rays[0], rays[1], (20, 20)) == ((7, 10), 8, 6)
    square = _mask('##', '##')
    bar = _mask('##', '##', '##')
    assert len(_canonical(square)[1]) == 4 and len(_canonical(bar)[1]) == 2
    assert _connected(_mask('#..#', '.##.')) and not _connected(_mask('#..#', '.##.'), False)
    t = _mask('####', '.##.')
    assert _docked(t, 3, (4, 10)) == {(4, 8), (5, 8), (6, 8), (7, 8), (5, 9), (6, 9)}
    # A complete hand-calculated unequal-distance scene, with one quarter-turn,
    # a symmetric attachment and an irregular unmatched silhouette.
    fixture = _build_sources((18, 20), [
        dict(core=(7, 1), color=1, ports=[(0, 2)]),
        dict(core=(14, 11), color=2, ports=[(-1, 0)])])
    expected = fixture.copy()
    fixture[1, 2:6], fixture[2, 3:5] = 1, 1
    fixture[11:13, 2:4] = 2
    fixture[1, 16:18], fixture[2, 15:18], fixture[3, 15:17] = 3, 3, 3
    for r, c in ((4, 9), (5, 9), (6, 9), (7, 9), (5, 10), (6, 10), (5, 11), (6, 11)):
        expected[r, c] = 1
    expected[7:10, 12:14] = 2
    assert np.array_equal(solve(fixture), expected)
    parsed = _parse(fixture)
    event = _encounters(parsed, fixture.shape)[0]
    assert (event['point'], event['hub'], event['distances']) == ((7, 11), (4, 11), (8, 6))
    assert not _encounters(parsed, fixture.shape, equal_clock=True)
    for turns in range(4):
        assert np.array_equal(solve(np.rot90(fixture, turns)), np.rot90(expected, turns))
    assert np.array_equal(solve(np.fliplr(fixture)), np.fliplr(expected))
    palette = np.array((8, 4, 0, 7, 2, 6, 5, 1, 9, 3), dtype=np.int8)
    assert np.array_equal(solve(palette[fixture]), palette[expected])
    # The solver's geometric domain extends beyond sampled width/depth ranges.
    # A width-8, depth-6 flange becomes an eight-cell vertical shoulder and a
    # five-cell horizontal stem, plus its two-cell socket bar.
    large = _build_sources((30, 30), [
        dict(core=(13, 1), color=1, ports=[(0, 2)]),
        dict(core=(24, 18), color=2, ports=[(-1, 0)])])
    large_expected = large.copy()
    large[0, :8], large[1:6, 3:5], large[0:2, 12:14] = 1, 1, 2
    large_expected[8:16, 12], large_expected[11:13, 13:19] = 1, 1
    large_expected[13:16, 19:21] = 2
    assert np.array_equal(solve(large), large_expected)
    ambiguous = _mask('.###', '####', '#.##', '.##.')
    try:
        _canonical(ambiguous)
    except ValueError:
        pass
    else:
        raise AssertionError('competing attachment edges accepted')
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    assert _puzzle_key(generate_puzzle(19, 10, 4)) == _puzzle_key(generate_puzzle(19, 10, 4))
    fs = [_features(pair) for p in puzzles for split in p.values() for pair in split]
    signatures = {s for f in fs for s in f['shapes']}
    assert len(signatures) >= 35, 'silhouette variation collapsed'
    assert len({f['shape'] for f in fs}) >= 30, 'canvas geometry variation collapsed'
    assert {2, 3, 4} <= {f['source_count'] for f in fs}
    assert {1, 2} == {f['event_count'] for f in fs}
    assert set().union(*(f['fork_types'] for f in fs)) == {'parallel', 'adjacent', 'opposite'}
    assert all(f['clock_changes'] and f['omit_a_changes'] and f['omit_b_changes'] for f in fs)
    assert len({f['source_port_sides'] for f in fs}) >= 10
    assert any(f['diagonal'] for f in fs) and any(f['symmetric'] for f in fs) and any(f['asymmetric'] for f in fs)
    assert any(f['missing'] for f in fs)
    assert {h for f in fs for (h, w), _ in f['shapes']} >= {2, 3, 4, 5}
    assert {w for f in fs for (h, w), _ in f['shapes']} >= {2, 4, 6}
    assert {f['background'] for f in fs} == set(range(10))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(71 + nt * 7 + ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    corrupt = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupted answer accepted')
    return dict(seeds=42, pairs=len(fs), silhouettes=len(signatures),
                grid_shapes=len({f['shape'] for f in fs}),
                port_configurations=len({f['source_port_sides'] for f in fs}),
                fork_types=sorted(set().union(*(f['fork_types'] for f in fs))),
                semantic_verification='generator-owned invariants and hand-calculated fixtures only')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
