"""ARC route printers: registered masks transported by repaired periodic walks.

Only standard-library modules and NumPy are used. No parent implementation,
stored examples, RNG state, or puzzle metadata is used by solve().
"""
from __future__ import annotations

import json
from numbers import Integral

import numpy as np

BG, MASK, REPAIR, REDIRECT, BEAD = 0, 8, 5, 4, 6
PAYLOAD = (1, 2, 3, 7, 9)
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((a >= 0) & (a <= 9))), "bad grid colors")
    return a.astype(np.int8, copy=True)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _neighbors(p, cells):
    return [_add(p, d) for d in DIRS if _add(p, d) in cells]


def _components(cells, diagonal=False):
    remaining, result = set(cells), []
    steps = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c) if diagonal else DIRS
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            p = stack.pop()
            for d in steps:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _positions(a, color):
    return {tuple(map(int, p)) for p in np.argwhere(a == color)}


def _parse_instructions(a):
    records = []
    masks = _components(_positions(a, MASK), diagonal=True)
    _require(bool(masks), "no stencil")
    for cells in masks:
        near = {_add(p, (r, c)) for p in cells for r in (-1, 0, 1) for c in (-1, 0, 1)}
        options = []
        for q in sorted(near):
            if not _inside(q, a.shape) or int(a[q]) not in (BG, MASK):
                continue
            for d in DIRS:
                target = (q[0] - d[0], q[1] - d[1])
                source = (q[0] - 2*d[0], q[1] - 2*d[1])
                if not _inside(source, a.shape) or not _inside(target, a.shape):
                    continue
                s, t = int(a[source]), int(a[target])
                if s in PAYLOAD and t in PAYLOAD and s != t:
                    options.append({"source": s, "target": t, "q": q, "direction": d,
                                    "mask": cells, "labels": {source, target},
                                    "offsets": tuple(sorted((r-q[0], c-q[1]) for r, c in cells))})
        _require(len(options) == 1, "stencil labels are ambiguous or missing")
        records.append(options[0])
    _require(len({r['source'] for r in records}) == len(records), "repeated source instruction")
    labels = [p for r in records for p in r['labels']]
    _require(len(labels) == len(set(labels)), "shared instruction labels")
    return records


def _repair_layers(a, records):
    work = a.copy()
    for rec in records:
        for p in rec['mask'] | rec['labels']:
            work[p] = BG
    sources = {r['source'] for r in records}
    _require(set(map(int, np.unique(work))) <= sources | {BG, REPAIR, REDIRECT, BEAD},
             "unassigned route color")
    pending = _positions(work, REPAIR) | _positions(work, BEAD)
    while pending:
        assigned = []
        for p in sorted(pending):
            colors = {int(work[q]) for d in DIRS if _inside(q := _add(p, d), work.shape)
                      and int(work[q]) in sources}
            if len(colors) == 1:
                work[p] = next(iter(colors))
                assigned.append(p)
        _require(bool(assigned), "annotation lacks a unique source-color neighbor")
        pending.difference_update(assigned)
    layers = {s: _positions(work, s) for s in sources}
    for s, cells in layers.items():
        original = set(cells)
        for p in sorted(original):
            for d in ((0, 1), (1, 0)):
                side = (d[1], -d[0])
                for distance in (2, 3):
                    q = (p[0] + distance*d[0], p[1] + distance*d[1])
                    if q not in original:
                        continue
                    middle = [(p[0] + k*d[0], p[1] + k*d[1]) for k in range(1, distance)]
                    if not all(int(work[x]) == BG for x in middle):
                        continue
                    # Nearby turning ends are not a witness for a new segment.
                    turns = [any(_add(x, t) in original for t in (side, (-side[0], -side[1])))
                             for x in (p, q)]
                    if not all(turns):
                        cells.update(middle)
    return layers


def _ordered_path(cells, a):
    ends = [p for p in cells if len(_neighbors(p, cells)) == 1]
    _require(len(ends) == 2 and all(1 <= len(_neighbors(p, cells)) <= 2 for p in cells),
             "route must be a simple open path")
    starts = [p for p in ends if int(a[p]) == BEAD]
    _require(len(starts) == 1, "one endpoint bead is required")
    path, previous, current = [], None, starts[0]
    while True:
        path.append(current)
        following = [p for p in _neighbors(current, cells) if p != previous]
        if not following:
            break
        _require(len(following) == 1 and following[0] not in path, "ambiguous path walk")
        previous, current = current, following[0]
    _require(len(path) == len(cells), "path did not cover its component")
    beads = [i for i, p in enumerate(path) if int(a[p]) == BEAD]
    _require(len(beads) == 2 and beads[0] == 0 and 2 <= beads[1] <= 6
             and beads[1] < len(path)-1, "two beads must encode an interval of 2..6")
    return path, beads[1]


def _ray(p, d, shape):
    cells = []
    p = _add(p, d)
    while _inside(p, shape):
        cells.append(p)
        p = _add(p, d)
    return cells


def _valid_walk(path):
    cells = set(path)
    return (len(cells) == len(path) and len(path) >= 2
            and all(abs(b[0]-a[0]) + abs(b[1]-a[1]) == 1 for a, b in zip(path, path[1:]))
            and all(len(_neighbors(p, cells)) == (1 if i in (0, len(path)-1) else 2)
                    for i, p in enumerate(path)))


def _decode(a):
    records = _parse_instructions(a)
    layers = _repair_layers(a, records)
    colors = sorted(layers)
    for i, color in enumerate(colors):
        for other in colors[i+1:]:
            _require(not layers[color] & layers[other] and
                     not any(_neighbors(p, layers[other]) for p in layers[color]),
                     "original routes must be separated")
    routes = []
    for color in sorted(layers):
        _require(bool(layers[color]), "instruction has no route")
        for component in _components(layers[color]):
            walk, stride = _ordered_path(component, a)
            routes.append({"source": color, "original": walk, "walk": walk,
                           "stride": stride, "pivot": None, "old": [], "new": []})
    for q in sorted(_positions(a, REDIRECT)):
        options = []
        for i, route in enumerate(routes):
            path = route['original']
            for j in range(route['stride']+1, len(path)-1):
                pivot = path[j]
                d = q[0]-pivot[0], q[1]-pivot[1]
                if d not in DIRS or path[j+1] != _add(pivot, (-d[0], -d[1])):
                    continue
                incoming = (pivot[0]-path[j-1][0], pivot[1]-path[j-1][1])
                if incoming[0]*d[0] + incoming[1]*d[1] != 0:
                    continue
                end = path[-1]
                if end[0] not in (0, a.shape[0]-1) and end[1] not in (0, a.shape[1]-1):
                    continue
                new = _ray(pivot, d, a.shape)
                corrected = path[:j+1] + new
                if new and _valid_walk(corrected):
                    options.append((i, j, d, new, corrected))
        _require(len(options) == 1, "direction marker needs a unique terminal branch")
        i, j, d, new, corrected = options[0]
        _require(routes[i]['pivot'] is None, "at most one redirect per route")
        routes[i].update(walk=corrected, pivot=j, direction=d,
                         old=routes[i]['original'][j+1:], new=new)
    return {"shape": a.shape, "records": records, "routes": routes, "input": a}


def _transport(offset, reference, heading):
    r, c = offset
    forward = r*reference[0] + c*reference[1]
    right = r*reference[1] - c*reference[0]
    return forward*heading[0] + right*heading[1], forward*heading[1] - right*heading[0]


def _heading(path, i, incoming=False):
    if i == len(path)-1 or (incoming and i > 0):
        return path[i][0]-path[i-1][0], path[i][1]-path[i-1][1]
    return path[i+1][0]-path[i][0], path[i+1][1]-path[i][1]


def _paint(data, *, original=False, points=False, fixed=False, incoming=False,
           dense=False, skip_gaps=False, centered=False, force_endpoint=False,
           reset_clock=False, manhattan=False):
    """Render footprints; optional incorrect readings are validation ablations."""
    out = np.zeros(data['shape'], dtype=np.int8)
    visits = {}
    records = {r['source']: r for r in data['records']}
    for route in data['routes']:
        rec = records[route['source']]
        path = route['original'] if original else route['walk']
        indices = list(range(len(path)))
        if skip_gaps:
            old = set(route['original'])
            indices = [i for i in indices if path[i] not in old or int(data['input'][path[i]]) != BG]
        indices = indices[::1 if dense else route['stride']]
        if reset_clock and route['pivot'] is not None and not original:
            j = route['pivot']
            indices = [i for i in indices if i < j] + list(range(j, len(path), route['stride']))
        if manhattan:
            start = path[0]
            indices = [i for i, p in enumerate(path)
                       if (abs(p[0]-start[0])+abs(p[1]-start[1])) % route['stride'] == 0]
        if force_endpoint and len(path)-1 not in indices:
            indices.append(len(path)-1)
        offsets = ((0, 0),) if points else rec['offsets']
        if centered:
            center = tuple((min(p[k] for p in offsets)+max(p[k] for p in offsets))//2 for k in (0, 1))
            offsets = tuple((r-center[0], c-center[1]) for r, c in offsets)
        for i in indices:
            p = path[i]
            heading = rec['direction'] if fixed else _heading(path, i, incoming)
            for offset in offsets:
                d = _transport(offset, rec['direction'], heading)
                q = _add(p, d)
                if _inside(q, out.shape):
                    _require(q not in visits or visits[q] == rec['target'], "conflicting ink footprints")
                    visits[q] = rec['target']
    for p, color in visits.items():
        out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the instruction geometry, route clock, and edits from this input."""
    return _paint(_decode(_grid(input_grid)))


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _line(a, b):
    length = abs(a[0]-b[0]) + abs(a[1]-b[1])
    d = (int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1])))
    return [_add(a, (k*d[0], k*d[1])) for k in range(length+1)]


def _corners(path):
    return [i for i in range(1, len(path)-1) if _heading(path, i) != _heading(path, i, True)]


def _random_walk(rng, shape, boundary=True, segments=None, margin=1):
    side = int(rng.integers(4))
    if boundary:
        start = ((0 if side == 0 else shape[0]-1, int(rng.integers(2, shape[1]-2)))
                 if side < 2 else (int(rng.integers(2, shape[0]-2)), 0 if side == 2 else shape[1]-1))
        axis = 0 if side < 2 else 1
    else:
        start = tuple(int(rng.integers(2, n-2)) for n in shape)
        axis = int(rng.integers(2))
    path = [start]
    desired = int(rng.integers(2, 6)) if segments is None else segments
    for _ in range(desired):
        values = [v for v in range(margin, shape[axis]-margin) if abs(v-path[-1][axis]) >= 3]
        rng.shuffle(values)
        for v in values:
            endpoint = list(path[-1])
            endpoint[axis] = v
            candidate = path + _line(path[-1], tuple(endpoint))[1:]
            if _valid_walk(candidate):
                path = candidate
                break
        else:
            break
        axis = 1-axis
    return list(reversed(path))


def _make_route(rng, shape, source, redirect, teaching, directions):
    for _ in range(180):
        straight = teaching == 0 and not redirect and rng.random() < .25
        path = _random_walk(rng, shape, boundary=redirect or teaching == 1 or rng.random() < .6,
                            segments=1 if straight else None, margin=2 if teaching == 1 else 1)
        if len(path) < 10:
            continue
        corners = _corners(path)
        if redirect:
            options = []
            for j in corners:
                d0 = _heading(path, j)
                d = -d0[0], -d0[1]
                if d not in directions or j < 6:
                    continue
                if teaching == 2 and not any(k > j for k in corners):
                    continue
                new = _ray(path[j], d, shape)
                corrected = path[:j+1]+new
                if not new or not _valid_walk(corrected):
                    continue
                strides = [s for s in range(2, 7) if j >= 2*s and (teaching != 2 or j % s != 0)]
                if strides:
                    options.append((j, d, new, corrected, strides))
            if not options:
                continue
            j, d, new, corrected, strides = _choose(rng, options)
            stride = _choose(rng, strides)
            return {"source": source, "original": path, "walk": corrected, "stride": stride,
                    "pivot": j, "direction": d, "old": path[j+1:], "new": new}
        strides = [s for s in range(2, 7) if s < len(path)-1 and (teaching != 1 or
                   ((len(path)-1) % s == 0 and any(k % s == 0 for k in corners)))]
        if strides:
            return {"source": source, "original": path, "walk": path, "stride": _choose(rng, strides),
                    "pivot": None, "old": [], "new": []}
    raise ValueError("route sampling exhausted")


def _sample_mask(rng, kind=None):
    kind = kind or _choose(rng, ('flag', 'elbow', 'comb', 'hole', 'hole', 'growth', 'line', 'chevron', 'single'))
    length = int(rng.integers(2, 5))
    sign = _choose(rng, (-1, 1))
    if kind == 'single':
        cells = {(0, 0)}
    elif kind in ('flag', 'elbow'):
        cells = {(0, c) for c in range(length+1)}
        cells |= {(sign*r, length) for r in range(1, int(rng.integers(2, 4)))}
    elif kind == 'comb':
        cells = {(0, c) for c in range(length+1)}
        cells |= {(sign, c) for c in range(0, length+1, 2)}
    elif kind == 'hole':
        radius = int(rng.integers(1, 3))
        cells = {(r, length) for r in range(-radius, radius+1)}
        cells |= {(r, c) for r in (-radius, radius) for c in range(length+1)}
        # A near registration witness remains present for radius two.
        cells.add((sign, 0))
        # The same hollow geometry can register at its open edge, one step
        # inside it, or one step outside its box. The label ray still defines
        # the exact point; none of these choices uses a geometric center.
        registration = _choose(rng, (-1, 0, 1))
        cells = {(r, c-registration) for r, c in cells}
    elif kind == 'line':
        cells = {(0, c) for c in range(length+1)}
    elif kind == 'chevron':
        cells = {(r, 2-abs(r)) for r in range(-2, 3)}
    else:
        cells = {(0, 0), (0, 1)}
        desired = int(rng.integers(4, 13))
        for _ in range(desired*8):
            p = _choose(rng, sorted(cells))
            q = _add(p, _choose(rng, DIRS))
            if -2 <= q[0] <= 2 and 0 <= q[1] <= 4:
                cells.add(q)
            if len(cells) >= desired:
                break
    return tuple(sorted(cells))


def _canonical_offsets(rec):
    return tuple(sorted(_transport(p, rec['direction'], (0, 1)) for p in rec['offsets']))


def _registration_kind(rec):
    cells = rec['offsets']
    if (0, 0) in cells:
        return 'occupied'
    lo = [min(p[k] for p in cells) for k in (0, 1)]
    hi = [max(p[k] for p in cells) for k in (0, 1)]
    if any(lo[k] > 0 or hi[k] < 0 for k in (0, 1)):
        return 'adjacent_hole'
    if all(lo[k] < 0 < hi[k] for k in (0, 1)):
        return 'interior_hole'
    return 'edge_hole'


def _place_instructions(rng, a, specs):
    records = []
    forbidden = {q for p in _positions(a, BEAD) | _positions(a, REDIRECT) | set().union(
                  *(_positions(a, c) for c in PAYLOAD)) for d in ((0, 0),)+DIRS
                 if _inside(q := _add(p, d), a.shape)}
    for source, target, offsets, direction in specs:
        cells = set(offsets) | {(0, 0), (-direction[0], -direction[1]),
                               (-2*direction[0], -2*direction[1])}
        lo = tuple(min(p[k] for p in cells) for k in (0, 1))
        hi = tuple(max(p[k] for p in cells) for k in (0, 1))
        locations = [(r, c) for r in range(-lo[0], a.shape[0]-hi[0])
                     for c in range(-lo[1], a.shape[1]-hi[1])]
        rng.shuffle(locations)
        for q in locations:
            absolute = {_add(q, p) for p in cells}
            if absolute & forbidden or any(int(a[p]) != BG for p in absolute):
                continue
            candidate = a.copy()
            mask = {_add(q, p) for p in offsets}
            for p in mask:
                candidate[p] = MASK
            labels = {(_add(q, (-2*direction[0], -2*direction[1]))): source,
                      (_add(q, (-direction[0], -direction[1]))): target}
            for p, color in labels.items():
                candidate[p] = color
            try:
                parsed = _parse_instructions(candidate)
            except ValueError:
                continue
            if len(parsed) != len(records)+1:
                continue
            rec = next((r for r in parsed if r['source'] == source), None)
            if rec is None or rec['q'] != q or rec['offsets'] != offsets:
                continue
            a = candidate
            records.append(rec)
            forbidden |= {_add(p, (r, c)) for p in absolute for r in (-1, 0, 1) for c in (-1, 0, 1)}
            break
        else:
            raise ValueError("instructions do not fit")
    return a, records


def _route_key(routes):
    return sorted((r['source'], tuple(r['original']), tuple(r['walk']), r['stride']) for r in routes)


def _different(data, expected, **wrong):
    try:
        return not np.array_equal(_paint(data, **wrong), expected)
    except ValueError:
        # A conflicting footprint is also incompatible with the valid output.
        return True


def _visible_hole(data, out):
    """A full hole-registered footprint must leave its own anchor empty."""
    records = {r['source']: r for r in data['records']}
    for route in data['routes']:
        rec = records[route['source']]
        if (0, 0) in rec['offsets']:
            continue
        path = route['walk']
        for i in range(0, len(path), route['stride']):
            footprint = [_add(path[i], _transport(o, rec['direction'], _heading(path, i)))
                         for o in rec['offsets']]
            if (int(out[path[i]]) == BG and all(_inside(p, out.shape) and int(out[p]) == rec['target']
                                              for p in footprint)):
                return True
    return False


def _make_pair(rng, palette, index, test, train_geometries, directions):
    for _ in range(220):
        if index == 0 and not test:
            sources, targets, kinds = [palette[0]], [palette[1]], ['flag']
        elif index == 1 and not test:
            sources, targets, kinds = palette[:2], [palette[2], palette[2]], ['hole', None]
        else:
            n = int(rng.integers(1, 4))
            sources = list(map(int, rng.choice(palette, n, replace=False)))
            if n > 1 and rng.random() < .45:
                target = _choose(rng, [c for c in palette if c not in sources])
                targets = [target]*n
            else:
                targets = [_choose(rng, [c for c in palette if c != s]) for s in sources]
            kinds = [None]*n
        n = len(sources)
        minimum = 12 if n == 1 else 19 if n == 2 else 24
        shape = tuple(int(rng.integers(minimum, 31)) for _ in (0, 1))
        routes, used = [], set()
        try:
            for i, source in enumerate(sources):
                teaching = 1 if index == 0 and not test else 2 if index == 1 and not test and i == 0 else 0
                redirect = teaching == 2 or (teaching != 1 and rng.random() < (.85 if test else .5))
                for _ in range(35):
                    route = _make_route(rng, shape, source, redirect, teaching, directions)
                    cells = set(route['original'])
                    q = route['walk'][route['pivot']+1] if redirect else None
                    guard = cells | ({q} if q is not None else set())
                    if guard & used or any(_neighbors(p, used) for p in guard):
                        continue
                    # Direction cells must start empty, including on this path.
                    if q is not None and q in cells:
                        continue
                    routes.append(route)
                    used |= guard
                    break
                else:
                    raise ValueError("separated routes do not fit")
            a = np.zeros(shape, dtype=np.int8)
            for route in routes:
                for p in route['original']:
                    a[p] = route['source']
                a[route['original'][0]] = a[route['original'][route['stride']]] = BEAD
                if route['pivot'] is not None:
                    a[route['walk'][route['pivot']+1]] = REDIRECT
            specs = []
            for s, t, kind in zip(sources, targets, kinds):
                canonical = _sample_mask(rng, kind)
                d = _choose(rng, DIRS)
                offsets = tuple(sorted(_transport(p, (0, 1), d) for p in canonical))
                specs.append((s, t, offsets, d))
            if test and not any(tuple(sorted(_transport(p, d, (0, 1)) for p in offsets))
                                not in train_geometries for s, t, offsets, d in specs):
                continue
            a, records = _place_instructions(rng, a, specs)
            construction = {"shape": shape, "records": records, "routes": routes, "input": a}
            # Ground truth uses sampled walks; the input decoder must recover them.
            out = _paint(construction)
            data = _decode(a)
            if _route_key(data['routes']) != _route_key(routes):
                continue
            # Annotate bends, pivots, endpoints, and ordinary straight cells.
            candidates = [p for route in routes for p in route['original'] if int(a[p]) in sources]
            rng.shuffle(candidates)
            forced = []
            if index == 1 and not test:
                r = routes[0]
                forced = [r['original'][j] for j in _corners(r['original']) if j > r['pivot']][:1]
                forced += [r['original'][r['pivot']]]
            desired = int(rng.integers(1, 7))
            count = 0
            for p in list(dict.fromkeys(forced+candidates)):
                if count >= desired and p not in forced:
                    break
                previous = int(a[p])
                a[p] = REPAIR
                try:
                    recovered = _decode(a)
                    _require(_route_key(recovered['routes']) == _route_key(routes), "marker changed route")
                except ValueError:
                    a[p] = previous
                else:
                    count += 1
            gap_candidates = [p for route in routes for j, p in enumerate(route['original'])
                              if 0 < j < len(route['original'])-1 and j not in _corners(route['original'])
                              and int(a[p]) == route['source']]
            rng.shuffle(gap_candidates)
            desired = int(rng.integers(1 if index == 0 and not test else 0, 5))
            gaps = 0
            for p in gap_candidates:
                if gaps >= desired:
                    break
                previous = int(a[p])
                a[p] = BG
                try:
                    recovered = _decode(a)
                    _require(_route_key(recovered['routes']) == _route_key(routes), "gap changed route")
                except ValueError:
                    a[p] = previous
                else:
                    gaps += 1
            data = _decode(a)
            _require(np.array_equal(_paint(data), out), "visible input differs from constructed answer")
            if index == 0 and not test:
                if not (gaps and all(_different(data, out, **wrong) for wrong in
                        ({'points': True}, {'fixed': True}, {'incoming': True}, {'dense': True},
                         {'skip_gaps': True}, {'centered': True}, {'manhattan': True}))):
                    continue
                # Endpoint faces out of the canvas; registration remains occupied.
                route, rec = data['routes'][0], data['records'][0]
                p = route['walk'][-1]
                d = _heading(route['walk'], len(route['walk'])-1)
                if not any(not _inside(_add(p, _transport(o, rec['direction'], d)), shape) for o in rec['offsets']):
                    continue
            if index == 1 and not test:
                if not (_different(data, out, original=True) and _different(data, out, force_endpoint=True) and
                        _different(data, out, reset_clock=True) and
                        any(int(a[p]) == REPAIR for r in data['routes'] for p in r['old']) and
                        _visible_hole(data, out)):
                    continue
            return {'input': a, 'output': out}
        except ValueError:
            continue
    raise ValueError("bounded scene construction exhausted")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    _require(num_train > num_test, "num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(PAYLOAD)))
    held_out = _choose(rng, DIRS) if rng.random() < .5 else None
    train_dirs = [d for d in DIRS if d != held_out]
    puzzle, seen, geometries = {'train': [], 'test': []}, set(), set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for _ in range(12):
                pair = _make_pair(rng, palette, index, split == 'test', geometries,
                                  [held_out] if split == 'test' and held_out is not None else train_dirs)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == 'train':
                        geometries.update(_canonical_offsets(r) for r in _parse_instructions(pair['input']))
                    break
            else:
                raise ValueError("distinct scene construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "bad puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), "bad pair counts")
    seen, train_data, train_geometries = set(), [], set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "bad pair keys")
            a, out = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == out.shape, "shape changed")
            key = a.shape, a.tobytes()
            _require(key not in seen, "duplicate input")
            seen.add(key)
            data = _decode(a)
            _require(np.array_equal(_paint(data), out), "incorrect route printing")
            _require(not np.array_equal(a, out), "identity example")
            _require(not any(np.any(out == c) for c in (MASK, REPAIR, REDIRECT, BEAD)), "annotation survived")
            # No unknown color, ink, or unaffected pixel may be inserted.
            _require(set(map(int, np.unique(out))) <= {BG} | {r['target'] for r in data['records']},
                     "output contains an unassigned ink")
            if split == 'train':
                train_data.append((data, out))
                train_geometries.update(_canonical_offsets(r) for r in data['records'])
            else:
                _require(any(_canonical_offsets(r) not in train_geometries for r in data['records']),
                         "query must expose a new registered stencil")
    for wrong in ({'points': True}, {'fixed': True}, {'incoming': True}, {'dense': True},
                  {'skip_gaps': True}, {'centered': True}, {'original': True}, {'force_endpoint': True},
                  {'reset_clock': True}, {'manhattan': True}):
        _require(any(_different(data, out, **wrong) for data, out in train_data),
                 "demonstrations fail to distinguish " + str(wrong))
    _require(any((0, 0) not in r['offsets'] for data, out in train_data for r in data['records']),
             "no hole-registration demonstration")
    _require(any(_visible_hole(data, out) for data, out in train_data), "hole registration is hidden by overlap")
    _require(any(int(data['input'][p]) == REPAIR for data, out in train_data
                 for route in data['routes'] for p in route['old']), "no repaired deleted branch")
    return True


def _puzzle_key(puzzle):
    return tuple((np.asarray(p[r]).shape, np.asarray(p[r], dtype=np.int8).tobytes())
                 for split in ('train', 'test') for p in puzzle[split] for r in ('input', 'output'))


def measure_puzzle(puzzle):
    """Author-owned measurements reconstructed from returned input grids."""
    result = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            data = _decode(_grid(pair['input']))
            routes, records = data['routes'], data['records']
            result.append({'split': split, 'shape': list(data['shape']), 'printers': len(routes),
                           'masks': [_canonical_offsets(r) for r in records],
                           'mask_sizes': [len(r['offsets']) for r in records],
                           'holes': sum((0, 0) not in r['offsets'] for r in records),
                           'registrations': [_registration_kind(r) for r in records],
                           'strides': [r['stride'] for r in routes],
                           'turns': [len(_corners(r['original'])) for r in routes],
                           'old_bends': [len(_corners(r['original'])) - len([i for i in _corners(r['original'])
                                         if i <= r['pivot']]) for r in routes if r['pivot'] is not None],
                           'directions': [r['direction'] for r in routes if r['pivot'] is not None],
                           'ray_lengths': [len(r['new']) for r in routes if r['pivot'] is not None],
                           'print_counts': [(len(r['walk'])-1)//r['stride']+1 for r in routes],
                           'gaps': sum(int(data['input'][p]) == BG for r in routes for p in r['original']),
                           'repair_count': int(np.count_nonzero(data['input'] == REPAIR)),
                           'deleted_repairs': sum(int(data['input'][p]) == REPAIR for r in routes for p in r['old'])})
    return result


def _manual_regressions():
    # The expected cells below were worked out by hand. This small printer
    # starts at (6,5), has interval two, and turns at its repair-marked pivot.
    a = np.zeros((10, 13), dtype=np.int8)
    a[2, 0:2] = (1, 2)
    for p in ((2, 2), (2, 3), (3, 3)):
        a[p] = MASK
    a[6, 5:10] = 1
    a[3:7, 9] = 1
    a[3, 9:13] = 1
    a[6, 5] = a[6, 7] = BEAD
    a[6, 9] = a[3, 9] = REPAIR
    a[4, 9] = BG
    a[7, 9] = REDIRECT
    redirected = np.zeros_like(a)
    for p in ((6, 5), (6, 6), (7, 6), (6, 7), (6, 8), (7, 8),
              (6, 9), (7, 9), (8, 9), (9, 9), (9, 8)):
        redirected[p] = 2
    assert np.array_equal(solve(a), redirected)
    # Without the arrow, the repaired empty cell (4,9) is itself a print anchor;
    # the corner turns the mask upward, and the final rightward print clips.
    b = a.copy()
    b[7, 9] = BG
    retained = np.zeros_like(b)
    for p in ((6, 5), (6, 6), (7, 6), (6, 7), (6, 8), (7, 8),
              (6, 9), (5, 9), (5, 10), (4, 9), (3, 9), (3, 10),
              (3, 11), (4, 11), (3, 12)):
        retained[p] = 2
    assert np.array_equal(solve(b), retained)
    # An adjacent empty registration, not the bounding-box center, is moved
    # with the mask. Several printing locations remain empty in the result.
    c = a.copy()
    c[c == MASK] = BG
    for p in ((2, 3), (3, 3), (3, 4)):
        c[p] = MASK
    hollow = np.zeros_like(c)
    for p in ((6, 6), (7, 6), (7, 7), (6, 8), (7, 8), (7, 9),
              (8, 8), (9, 9), (9, 8)):
        hollow[p] = 2
    assert np.array_equal(solve(c), hollow)
    # Reflection transports the handedness of the visible mask as well.
    for inp, expected in ((a, redirected), (b, retained), (c, hollow)):
        for k in range(4):
            for reflect in (False, True):
                x = np.fliplr(inp) if reflect else inp
                y = np.fliplr(expected) if reflect else expected
                assert np.array_equal(solve(np.rot90(x, k)), np.rot90(y, k))
    # Uniquely assigned source and ink roles may swap without changing geometry.
    recolored = a.copy()
    recolored[a == 1], recolored[a == 2] = 9, 3
    assert np.array_equal(solve(recolored), np.where(redirected == 2, 3, BG))


def self_test():
    _manual_regressions()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    features = [f for p in puzzles for f in measure_puzzle(p)]
    assert len({tuple(f['shape']) for f in features}) >= 45
    assert {f['printers'] for f in features} >= {1, 2, 3}
    assert {s for f in features for s in f['strides']} == set(range(2, 7))
    assert {tuple(d) for f in features for d in f['directions']} == set(DIRS)
    assert len({tuple(tuple(p) for p in m) for f in features for m in f['masks']}) >= 30
    assert max(n for f in features for n in f['mask_sizes']) >= 10
    assert {'occupied', 'edge_hole', 'interior_hole', 'adjacent_hole'} <= {
        k for f in features for k in f['registrations']}
    assert {0, 1, 2} <= {n for f in features for n in f['old_bends']}
    assert min(n for f in features for n in f['ray_lengths']) == 1
    assert any(0 in f['turns'] for f in features)
    assert any(f['gaps'] >= 2 and f['deleted_repairs'] and f['holes'] for f in features)
    # Input-only, palette-independent geometric regression under rotation.
    for puzzle in puzzles[:3]:
        for pair in puzzle['train']+puzzle['test']:
            a = pair['input'].copy()
            before = a.copy()
            assert np.array_equal(solve(a), pair['output']) and np.array_equal(a, before)
            for k in (1, 2, 3):
                assert np.array_equal(solve(np.rot90(a, k)), np.rot90(pair['output'], k))
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(puzzles[0])
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(700+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Full-grid checking must catch an inserted pixel and wrong ink at a print.
    p = puzzles[0]
    for site in ((0, 0), tuple(map(int, np.argwhere(p['test'][0]['output'] != BG)[0]))):
        broken = {s: [dict(q) for q in p[s]] for s in p}
        broken['test'][0]['output'] = broken['test'][0]['output'].copy()
        broken['test'][0]['output'][site] = (int(broken['test'][0]['output'][site])+1) % 10
        try:
            validate_generated_puzzle(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt output accepted")
    return {'seeds_checked': 42, 'pairs_checked': len(features),
            'shapes': len({tuple(f['shape']) for f in features}),
            'registered_masks': len({tuple(tuple(p) for p in m) for f in features for m in f['masks']})}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
