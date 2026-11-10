"""Census routers: object counts become the cross-section of a moving tally.

Only standard Python and NumPy are used.  The builder paints labels from its
construction geometry; solve reparses the input without construction metadata.
"""
from __future__ import annotations

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions")
    _check(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _check(bool(np.all((a >= 0) & (a <= 9))), "color range")
    return a.astype(np.int8, copy=True)


def _paint(a, cells, color):
    for p in cells:
        a[p] = color


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs), min(cs), max(cs)


def _expand(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _components(a, bg):
    todo = {(int(r), int(c)) for r, c in np.argwhere(a != bg)}
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        color = int(a[start])
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in todo and int(a[p]) == color:
                    todo.remove(p)
                    stack.append(p)
                    cells.add(p)
        result.append(dict(color=color, cells=cells, box=_box(cells)))
    return result


def _frame(part):
    r0, r1, c0, c1 = part['box']
    side = r1 - r0 + 1
    if side not in (5, 7, 9) or c1 - c0 + 1 != side:
        return False
    perimeter = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                 if r in (r0, r1) or c in (c0, c1)}
    return part['cells'] == perimeter


def _fork(part):
    """Recognize a base and three separated uniform, positive prong runs."""
    if len(part['cells']) <= 8:
        return None
    parses = []
    for out in DIRS:
        perp = out[1], -out[0]
        ks = [r * out[0] + c * out[1] for r, c in part['cells']]
        qs = [r * perp[0] + c * perp[1] for r, c in part['cells']]
        k0, q0 = min(ks), min(qs)
        local = {(r * out[0] + c * out[1] - k0,
                  r * perp[0] + c * perp[1] - q0) for r, c in part['cells']}
        width = max(q for k, q in local) + 1
        if width % 2 == 0 or {q for k, q in local if k == 0} != set(range(width)):
            continue
        lens = [sum((k, q) in local for k in range(1, 31)) for q in range(width)]
        expected = {(0, q) for q in range(width)}
        expected |= {(k, q) for q, n in enumerate(lens) for k in range(1, n + 1)}
        if expected != local:
            continue
        runs, q = [], 0
        while q < width:
            if lens[q] == 0:
                q += 1
                continue
            begin, n = q, lens[q]
            while q < width and lens[q] == n:
                q += 1
            runs.append((begin, q, n))
        if len(runs) != 3 or any(runs[i][1] >= runs[i + 1][0] for i in (0, 1)):
            continue
        ca, cb, length = runs[1]
        if ca + cb != width or cb - ca not in (1, 3, 5, 7):
            continue
        parses.append(dict(part, out=out, perp=perp, k0=k0, q0=q0,
                           width=width, span=cb-ca, capacity=(cb-ca+1)//2,
                           stop=length+1, runs=tuple(runs), lengths=tuple(x[2] for x in runs)))
    _check(len(parses) <= 1, "ambiguous fork")
    return parses[0] if parses else None


def _cell(f, k, q):
    kk, qq = f['k0'] + k, f['q0'] + q
    return kk * f['out'][0] + qq * f['perp'][0], kk * f['out'][1] + qq * f['perp'][1]


def _route(f, launch_k):
    # Pitch two across the direction of travel, including both span endpoints.
    first = (f['width'] - f['span']) // 2
    lanes = range(first, first + f['span'], 2)
    final = {_cell(f, f['stop'], q) for q in lanes}
    trail = {_cell(f, k, q) for k in range(f['stop'] + 1, launch_k + 1) for q in lanes}
    return final, trail


def _analyze(a):
    values, counts = np.unique(a, return_counts=True)
    _check(np.count_nonzero(counts == counts.max()) == 1, "background must be unique mode")
    bg = int(values[np.argmax(counts)])
    _check(bg != 0 and not np.any(a == 0), "zero is the reserved, input-absent trail")
    parts = _components(a, bg)
    stations, forks, objects = [], [], []
    for p in parts:
        if _frame(p):
            r0, r1, c0, c1 = p['box']
            stations.append(dict(p, center=((r0+r1)//2, (c0+c1)//2),
                                 side=r1-r0+1, forks=[]))
        else:
            f = _fork(p)
            if f is not None:
                forks.append(f)
            else:
                _check(1 <= len(p['cells']) <= 8, "unrecognized large object")
                objects.append(p)
    _check(1 <= len(stations) <= 3 and bool(forks), "missing stations or forks")
    colors = [s['color'] for s in stations]
    _check(len(set(colors)) == len(colors), "station colors must be distinct")
    inks = {f['color'] for f in forks}
    _check(len(inks) == 1 and not (inks & set(colors)), "one separate fork ink required")
    ink = next(iter(inks))
    _check(all(o['color'] != ink for o in objects), "fork ink used by loose object")
    for s in stations:
        r0, r1, c0, c1 = s['box']
        _check(np.all(a[r0+1:r1, c0+1:c1] == bg), "station must be empty")
        s['count'] = sum(o['color'] == s['color'] for o in objects)
        s['area'] = sum(len(o['cells']) for o in objects if o['color'] == s['color'])
        _check(s['count'] <= (s['side']-1)//2, "count exceeds station cross-section")
    for f in forks:
        candidates = []
        for si, s in enumerate(stations):
            r, c = s['center']
            k = r*f['out'][0] + c*f['out'][1] - f['k0']
            q = r*f['perp'][0] + c*f['perp'][1] - f['q0']
            if q == f['width']//2 and k > max(f['lengths']) + s['side']//2:
                candidates.append((si, k))
        _check(len(candidates) == 1, "fork needs one axially aligned station on its open side")
        si, k = candidates[0]
        f['launch_k'] = k
        stations[si]['forks'].append(f)
    fork_cells = set().union(*(f['cells'] for f in forks))
    all_routes = set()
    for s in stations:
        _check(1 <= len(s['forks']) <= 4, "station needs one through four forks")
        _check(len({f['out'] for f in s['forks']}) == len(s['forks']), "repeated receiver ray")
        capacities = [f['capacity'] for f in s['forks']]
        _check(len(set(capacities)) == len(capacities), "receiver capacities tie")
        _check(max(capacities) <= (s['side']-1)//2, "receiver span outside station")
        s['selected'] = next((f for f in s['forks'] if f['capacity'] == s['count']), None)
        s['final'], s['trail'] = set(), set()
        for f in s['forks']:
            final, trail = _route(f, f['launch_k'])
            route = final | trail
            _check(all(0 <= r < a.shape[0] and 0 <= c < a.shape[1] for r, c in route), "route off canvas")
            _check(route.isdisjoint(fork_cells), "route hits a fork")
            for other in stations:
                if other is not s:
                    r0, r1, c0, c1 = other['box']
                    _check(not any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in route),
                           'route crosses another station')
            # These are potential routes, so changing a census never needs
            # collision precedence rules or a different hidden geometry policy.
            for o in objects:
                _check(route.isdisjoint(o['cells']), "object in launch corridor")
            if f is s['selected']:
                s['final'], s['trail'] = final, trail
        selected_route = s['final'] | s['trail']
        _check(all_routes.isdisjoint(selected_route), "selected routes intersect")
        all_routes |= selected_route
    # Keep every visible component separated, including diagonal contacts.
    occupied = set()
    for p in parts:
        _check(occupied.isdisjoint(_expand(p['cells'])), "input components touch")
        occupied |= p['cells']
    return dict(background=bg, ink=ink, stations=stations, forks=forks, objects=objects)


def _answer(a, detail):
    result = np.full_like(a, detail['background'])
    for f in detail['forks']:
        _paint(result, f['cells'], f['color'])
    for s in detail['stations']:
        _paint(result, s['trail'], 0)
        _paint(result, s['final'], s['color'])
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only visible geometry, color matching and component counts."""
    a = _grid(input_grid)
    return _answer(a, _analyze(a))


def _shape(rng, area=None):
    """Count-independent connected shapes, independently turned and reflected."""
    area = int(rng.integers(1, 9)) if area is None else int(area)
    kind = int(rng.integers(4))
    if kind == 0:
        cells = {(0, i) for i in range(area)}
    elif kind == 1:
        width = int(rng.integers(1, min(4, area)+1))
        cells = {(i//width, i%width) for i in range(area)}
    elif kind == 2:
        arm = int(rng.integers(1, area+1))
        cells = {(0, i) for i in range(arm)} | {(i, 0) for i in range(area-arm+1)}
    else:
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in DIRS} - cells)
            cells.add(frontier[int(rng.integers(len(frontier)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.integers(2):
        cells = {(r, -c) for r, c in cells}
    r0, _, c0, _ = _box(cells)
    return {(r-r0, c-c0) for r, c in cells}


def _module(rng, capacities, rays, side=None, teaching=False):
    """Build a station with cardinal receivers in local coordinates."""
    minimum = 2 * max(capacities) + 1
    side = int(rng.choice([x for x in (5, 7, 9) if x >= minimum])) if side is None else side
    half = side//2
    frame = {(r, c) for r in range(-half, half+1) for c in range(-half, half+1)
             if abs(r) == half or abs(c) == half}
    specs = []
    for cap, d in zip(capacities, rays):
        span = 2*cap-1
        guard, gap = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        tail = int(rng.integers(2))
        width = span + 2*(guard+gap+tail)
        lengths = tuple(map(int, rng.integers(1, 5, size=3)))
        if teaching and cap == 1:
            outer = int(rng.integers(3, 5))
            lengths = (outer, int(rng.integers(1, 3)), outer)
        elif teaching and cap == 2:
            lengths = (int(rng.integers(1, 3)), int(rng.integers(3, 5)), int(rng.integers(1, 3)))
        # Prevent confusion with any loose object (which has area at most 8).
        if width + guard*(lengths[0]+lengths[2]) + span*lengths[1] <= 8:
            lengths = (2, 2, 2)
        specs.append(dict(capacity=cap, d=d, guard=guard, tail=tail,
                          width=width, span=span, lengths=lengths))
    cross_half = max([half] + [f['width']//2 for f in specs])
    for f in specs:
        # For adjacent receivers, put all structure beyond every other fork's
        # transverse reach. Opposed receivers can use shorter distances.
        radial = (cross_half if len(rays) >= 3 or
                  (len(rays) == 2 and rays[0] != tuple(-x for x in rays[1])) else half)
        radius = min(14, radial + 2 + max(f['lengths']) + int(rng.integers(0, 2)))
        f['radius'] = radius
        dr, dc = f['d']
        out, perp = (-dr, -dc), (dc, -dr)
        wh = f['width']//2
        local = {(0, q) for q in range(-wh, wh+1)}
        runs = ((-wh+f['tail'], -wh+f['tail']+f['guard'], f['lengths'][0]),
                (-(f['span']//2), f['span']//2+1, f['lengths'][1]),
                (wh-f['tail']-f['guard']+1, wh-f['tail']+1, f['lengths'][2]))
        local |= {(k, q) for qa, qb, n in runs for k in range(1, n+1) for q in range(qa, qb)}
        f['cells'] = {(radius*dr+k*out[0]+q*perp[0],
                       radius*dc+k*out[1]+q*perp[1]) for k, q in local}
    all_cells = frame | set().union(*(f['cells'] for f in specs))
    r0, r1, c0, c1 = _box(all_cells)
    def shift(cells):
        return {(r-r0, c-c0) for r, c in cells}
    for f in specs:
        f['cells'] = shift(f['cells'])
    return dict(frame=shift(frame), center=(-r0, -c0), side=side,
                forks=specs, h=r1-r0+1, w=c1-c0+1)


def _layout(rng, modules):
    gap = int(rng.integers(1, 4)) if len(modules) > 1 else 0
    h = sum(m['h'] for m in modules) + gap*(len(modules)-1)
    w = max(m['w'] for m in modules)
    if h > 30 or w > 30:
        return None
    placed, row = [], 0
    for m in modules:
        col = int(rng.integers(w-m['w']+1))
        def shift(cells):
            return {(r+row, c+col) for r, c in cells}
        center = m['center'][0]+row, m['center'][1]+col
        placed.append(dict(frame=shift(m['frame']), center=center, side=m['side'],
                           forks=[dict(f, cells=shift(f['cells'])) for f in m['forks']]))
        row += m['h'] + gap
    return dict(stations=placed, h=h, w=w)


def _teaching_plan(rng):
    for _ in range(100):
        modules = [_module(rng, list(map(int, rng.permutation((1, 2)))), ((0, -1), (0, 1)),
                           side=int(rng.choice((7, 9))), teaching=(j == 0)) for j in range(2)]
        plan = _layout(rng, modules)
        unequal_distances = any(len({f['radius']-f['lengths'][1]-1 for f in m['forks']}) > 1
                               for m in modules)
        if plan is not None and unequal_distances:
            return plan
    raise AssertionError('teaching apparatus packing failed')


def _ordinary_plan(rng, count, branching):
    if count == 4:
        capacities = [1, 2, 3, 4]
    else:
        pool = [x for x in (1, 2, 3, 4) if x != count]
        capacities = ([count] if count else []) + list(map(int, rng.choice(
            pool, branching-(1 if count else 0), replace=False)))
    capacities = list(map(int, rng.permutation(capacities)))
    if len(capacities) == 2 and rng.random() < .7:
        rays = ((0, -1), (0, 1))
    else:
        rays = [DIRS[int(x)] for x in rng.choice(4, len(capacities), replace=False)]
    return _layout(rng, [_module(rng, capacities, rays)])


def _place_pair(rng, plan, colors, bg, ink, object_specs, counts, rotation):
    """Output painting here uses radii and known counts, not the solver parser."""
    h0, w0 = plan['h'], plan['w']
    _check(h0 <= 30 and w0 <= 30, 'apparatus too large')
    # Canvas growth is a response to actual shape packing, without changing
    # the census or substituting easier objects.
    h = min(30, max(12, h0 + int(rng.integers(0, 6))))
    w = min(30, max(12, w0 + int(rng.integers(0, 6))))
    for attempt in range(12):
        if attempt >= 4:
            h, w = 30, 30
        top, left = int(rng.integers(h-h0+1)), int(rng.integers(w-w0+1))
        a = np.full((h, w), bg, dtype=np.int8)
        b = a.copy()
        blocked = np.zeros((h, w), dtype=bool)
        reserved = set()
        for si, s in enumerate(plan['stations']):
            frame = {(r+top, c+left) for r, c in s['frame']}
            _paint(a, frame, colors[si])
            rr0, rr1, cc0, cc1 = _box(frame)
            reserved |= {(r, c) for r in range(rr0, rr1+1) for c in range(cc0, cc1+1)}
            p = s['center'][0]+top, s['center'][1]+left
            for f in s['forks']:
                cells = {(r+top, c+left) for r, c in f['cells']}
                _paint(a, cells, ink)
                _paint(b, cells, ink)
                reserved |= cells
                cap, (dr, dc) = f['capacity'], f['d']
                distance = f['radius']-f['lengths'][1]-1
                perp = dc, -dr
                offsets = range(-(cap-1), cap, 2)
                route = {(p[0]+t*dr+q*perp[0], p[1]+t*dc+q*perp[1])
                         for t in range(distance+1) for q in offsets}
                reserved |= route
                if counts[si] == cap:
                    _paint(b, route, 0)
                    final = {(p[0]+distance*dr+q*perp[0], p[1]+distance*dc+q*perp[1])
                             for q in offsets}
                    _paint(b, final, colors[si])
        for r, c in _expand(reserved):
            if 0 <= r < h and 0 <= c < w:
                blocked[r, c] = True
        success = True
        # Shape geometry is retained on retry; no singleton packing fallback.
        order = sorted(range(len(object_specs)), key=lambda i: (-len(object_specs[i][1]), i))
        for i in order:
            color, shape = object_specs[i]
            sh, sw = max(r for r, c in shape)+1, max(c for r, c in shape)+1
            candidates = np.ones((h-sh+1, w-sw+1), dtype=bool)
            for dr, dc in shape:
                candidates &= ~blocked[dr:dr+h-sh+1, dc:dc+w-sw+1]
            rs, cs = np.nonzero(candidates)
            if not len(rs):
                success = False
                break
            j = int(rng.integers(len(rs)))
            cells = {(int(rs[j])+dr, int(cs[j])+dc) for dr, dc in shape}
            _paint(a, cells, color)
            for r, c in _expand(cells):
                if 0 <= r < h and 0 <= c < w:
                    blocked[r, c] = True
        if success:
            return dict(input=np.rot90(a, rotation).copy(), output=np.rot90(b, rotation).copy())
        h, w = min(30, h+3), min(30, w+3)
    return None


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed):
        raise ValueError('seed must be an integer')
    if not _integer(num_train) or not 2 <= num_train <= 10:
        raise ValueError('num_train must be an integer in 2..10')
    if not _integer(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError('num_test must be in 1..4 and smaller than num_train')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    nt, nq = int(num_train), int(num_test)
    puzzle, seen = {'train': [], 'test': []}, set()
    color_order = list(map(int, rng.permutation(np.arange(1, 10))))
    anchor_colors, witness_ink, novel = color_order[:2], color_order[2], color_order[3]
    base_rotation = int(rng.integers(4))
    teaching = _teaching_plan(rng)
    # Equal literal-color area, unequal counts, plus equal counts at unequal
    # areas, all guaranteed even when only two demonstrations are requested.
    census_area = int(rng.integers(4, 9))
    other_area = int(rng.choice([x for x in range(3, 9) if x != census_area]))
    observed = {1, 2}
    previous_bg = None
    for index in range(nt+nq):
        train = index < nt
        local = index if train else index-nt
        split = 'train' if train else 'test'
        if index < 2:
            counts = [1, 2] if index == 0 else [2, 1]
            plan = teaching
            colors, ink = anchor_colors[:], witness_ink
            a1 = int(rng.integers(1, census_area))
            a2 = int(rng.integers(1, other_area))
            forced = [[census_area], [a2, other_area-a2]] if index == 0 else [[a1, census_area-a1], [other_area]]
            rotation = (base_rotation + index) % 4
        else:
            if train:
                count = 3 if local == 2 else 0 if local == 3 else 4 if local == 4 else int(rng.integers(0, 5))
                branching = 3 if count == 3 else 4 if count == 4 else int(rng.integers(1, 5))
            else:
                options = sorted(observed - {0}) if local == 0 else sorted(observed)
                count = int(rng.choice(options))
                branching = int(rng.integers(3, 5))
            counts = [count]
            plan = _ordinary_plan(rng, count, branching)
            category_pool = [c for c in range(1, 10) if not train or c != novel]
            colors = [novel if not train and local == 0 else int(rng.choice(category_pool))]
            ink = int(rng.choice([c for c in range(1, 10) if c not in colors]))
            forced = [None]
            rotation = int(rng.integers(4))
        if train:
            observed.update(counts)
        bg = int(rng.choice([c for c in range(1, 10) if c not in colors and c != ink and c != previous_bg]))
        extras = [c for c in range(1, 10) if c not in colors and c not in (bg, ink)]
        nd = int(rng.integers(0, 5))
        if not any(counts):
            nd = max(1, nd)
        pair = None
        for attempt in range(30):
            specs = []
            for ci, (color, count) in enumerate(zip(colors, counts)):
                for j in range(count):
                    specs.append((color, _shape(rng, None if forced[ci] is None else forced[ci][j])))
            specs += [(int(rng.choice(extras)), _shape(rng)) for _ in range(nd)]
            pair = _place_pair(rng, plan, colors, bg, ink, specs, counts, rotation)
            if pair is None:
                continue
            key = (pair['input'].shape, pair['input'].tobytes())
            if key not in seen:
                break
        else:
            raise AssertionError('could not pack a distinct scene')
        seen.add(key)
        puzzle[split].append(pair)
        previous_bg = bg
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check semantics and teaching evidence reconstructed from returned grids."""
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _check(all(isinstance(puzzle[s], list) for s in puzzle), 'split schema')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _check(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'pair counts')
    seen, evidence, directions = set(), [], set()
    nearer_is_wrong, short_middle, long_middle = False, False, False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _check(a.shape == b.shape, 'output dimensions')
            key = a.shape, a.tobytes()
            _check(key not in seen, 'duplicate input')
            seen.add(key)
            d = _analyze(a)
            _check(np.array_equal(b, _answer(a, d)), 'wrong routing, tally, trail, or erasure')
            _check(not np.array_equal(a, b), 'identity example')
            for f in d['forks']:
                _check(all(b[p] == d['ink'] for p in f['cells']), 'fork modified')
            for o in d['objects']:
                _check(all(b[p] == d['background'] for p in o['cells']), 'object not erased')
            for s in d['stations']:
                f, n = s['selected'], s['count']
                _check(np.count_nonzero(b == s['color']) == (n if f is not None else 0), 'not one dot per component')
                if f is not None:
                    distance = f['launch_k'] - f['stop']
                    _check(len(s['final']) == n and len(s['trail']) == n*distance, 'wrong swept lane area')
                    _check(all(b[p] == 0 for p in s['trail']), 'broken trail')
                    _check(all(b[p] == s['color'] for p in s['final']), 'lost packet color')
                    _check(len(s['final']) > 0, 'empty positive packet')
                if split == 'train':
                    evidence.append((s['color'], n, s['area']))
                    if f is not None:
                        directions.add(tuple(-x for x in f['out']))
                        nearer_is_wrong |= any(g['launch_k']-g['stop'] < distance for g in s['forks'])
                        short_middle |= f['lengths'][1] < min(f['lengths'][0], f['lengths'][2])
                        long_middle |= f['lengths'][1] > max(f['lengths'][0], f['lengths'][2])
            expected_trail = set().union(*(s['trail'] for s in d['stations']))
            _check(np.count_nonzero(b == 0) == len(expected_trail), 'extra trail pixels')
    collision = any(c == cc and area == aa and n != nn for c, n, area in evidence for cc, nn, aa in evidence)
    independence = any(n == nn and area != aa for c, n, area in evidence for cc, nn, aa in evidence)
    _check(collision and independence, 'training does not rule out area shortcuts')
    _check({1, 2} <= {n for c, n, area in evidence}, 'missing unary progression')
    _check(len(directions) >= 2, 'training does not show geometric routing')
    _check(nearer_is_wrong, 'training permits always selecting the nearest fork')
    _check(short_middle and long_middle, 'training does not distinguish central-tip stopping from guard stopping')
    return None


def _signature(puzzle):
    return tuple((s, np.asarray(p['input']).shape, np.asarray(p['input']).tobytes(),
                  np.asarray(p['output']).tobytes()) for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """A bounded 42-seed semantic/diversity suite, plus targeted interventions."""
    sizes, geometries, masks, counts, sides, branches, inks, bgs = (set() for _ in range(8))
    signatures, records, query_counts = set(), [], set()
    for seed in range(42):
        nt = 2 + seed % 9
        nq = 1 + seed % min(4, nt-1)
        puzzle = generate_puzzle(seed, nt, nq)
        signatures.add(_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                d = _analyze(a)
                _check(np.array_equal(solve(a), b), 'solver disagreement')
                sizes.add(a.shape)
                inks.add(d['ink'])
                bgs.add(d['background'])
                for f in d['forks']:
                    geometries.add((f['width'], f['runs']))
                for s in d['stations']:
                    counts.add(s['count'])
                    if split == 'test':
                        query_counts.add(s['count'])
                    sides.add(s['side'])
                    branches.add(len(s['forks']))
                for o in d['objects']:
                    r0, _, c0, _ = o['box']
                    masks.add(tuple(sorted((r-r0, c-c0) for r, c in o['cells'])))
                records.append(d)
    _check(len(signatures) == 42, 'repeated puzzles')
    _check(len(sizes) >= 30 and len(geometries) >= 100, 'insufficient structural variation')
    _check(len(masks) >= 50 and {0, 1, 2, 3, 4} <= counts, 'insufficient census/shape variation')
    _check(len(inks) >= 7 and len(bgs) >= 7 and len(sides) >= 2, 'palette/size variation')
    _check({1, 2, 3, 4} <= branches, 'receiver-branch variation')
    _check({1, 2, 3, 4} <= query_counts, 'test census variation')
    _check(_signature(generate_puzzle(77, 5, 2)) == _signature(generate_puzzle(77, 5, 2)), 'nondeterminism')
    for nt, nq in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(91+nt, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(7, 2, 1)
    # Omit B: merge two equal-color components by replacing them with one
    # connected object of the same total area. The compatible fork must change.
    original = p['train'][1]['input']
    d = _analyze(original)
    s = next(s for s in d['stations'] if s['count'] == 2)
    objs = [o for o in d['objects'] if o['color'] == s['color']]
    changed = original.copy()
    for o in objs:
        _paint(changed, o['cells'], d['background'])
    area = sum(len(o['cells']) for o in objs)
    # Search legal placements of the merged bar without altering the apparatus.
    found = False
    for r in range(changed.shape[0]):
        if found:
            break
        for c in range(changed.shape[1]-area+1):
            trial = changed.copy()
            cells = {(r, c+j) for j in range(area)}
            if not all(trial[q] == d['background'] for q in cells):
                continue
            _paint(trial, cells, s['color'])
            try:
                dd = _analyze(trial)
            except AssertionError:
                continue
            ss = next(x for x in dd['stations'] if x['color'] == s['color'])
            if ss['count'] == 1 and ss['selected'] is not None:
                _check(ss['selected']['out'] != s['selected']['out'], 'count did not reroute')
                _check(not np.array_equal(solve(trial), solve(original)), 'omitting census leaves answer unchanged')
                found = True
                break
    _check(found, 'same-area intervention could not be placed')
    # Omit A: remove the matching fork but retain both the source and census.
    changed = original.copy()
    _paint(changed, s['selected']['cells'], d['background'])
    dd = _analyze(changed)
    ss = next(x for x in dd['stations'] if x['color'] == s['color'])
    _check(ss['count'] == 2 and ss['selected'] is None, 'missing fork did not disable packet')
    _check(not np.array_equal(solve(changed), solve(original)), 'omitting receiver leaves answer unchanged')
    # Moving the central stopping face changes the packet and swept distance
    # while leaving its census and its receiver choice intact.
    changed = original.copy()
    f = s['selected']
    center_start = (f['width']-f['span'])//2
    extension = {_cell(f, f['stop'], q) for q in range(center_start, center_start+f['span'])}
    _paint(changed, extension, d['ink'])
    dd = _analyze(changed)
    ss = next(x for x in dd['stations'] if x['color'] == s['color'])
    _check(ss['count'] == s['count'] and ss['selected']['capacity'] == f['capacity'], 'extension altered census/key')
    _check(ss['final'] != s['final'] and len(ss['trail']) == len(s['trail'])-s['count'], 'central stop has no effect')
    # Geometry and semantic palette roles also commute with an unannounced
    # quarter-turn and a permutation of the nine non-trail colors.
    _check(np.array_equal(solve(np.rot90(original)), np.rot90(solve(original))), 'rotation equivariance')
    palette = np.array([0, 6, 4, 8, 2, 9, 1, 5, 3, 7], dtype=np.int8)
    _check(np.array_equal(solve(palette[original]), palette[solve(original)]), 'color-role equivariance')
    # Exercise a compatible three-station input outside the sampler's usual
    # one/two-station presentations, with a known construction-based label.
    extra_rng = np.random.default_rng(248)
    for _ in range(100):
        plan = _layout(extra_rng, [_module(extra_rng, [1], [(0, 1)], side=5) for j in range(3)])
        if plan is None:
            continue
        pair = _place_pair(extra_rng, plan, [1, 2, 3], 9, 8,
                           [(c, _shape(extra_rng, 2)) for c in (1, 2, 3)], [1, 1, 1], 0)
        if pair is not None:
            break
    else:
        raise AssertionError('three-station check could not be constructed')
    _check(len(_analyze(pair['input'])['stations']) == 3, 'three-station parsing')
    _check(np.array_equal(solve(pair['input']), pair['output']), 'three-station solve')
    # Corruption must be rejected by the public validator.
    corrupt = {k: [{q: v.copy() for q, v in pair.items()} for pair in p[k]] for k in p}
    corrupt['test'][0]['output'] = corrupt['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('bad label accepted')
    return dict(seeds=42, pairs=len(records), canvas_shapes=len(sizes), fork_geometries=len(geometries),
                object_masks=len(masks), counts=sorted(counts), station_sides=sorted(sides),
                receiver_counts=sorted(branches), test_counts=sorted(query_counts),
                backgrounds=len(bgs), fork_inks=len(inks))
