"""Rendezvous scaffold: implicit ray intersections are rigid docking keys.

The solver reads pixels only. Generation constructs a geometric scene and then
scatters its junction pieces; it does not import either parent's implementation.
"""
from collections import Counter, deque
from itertools import combinations

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _box(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _normal(cells):
    t, l, _, _ = _box(cells)
    return tuple(sorted((r - t, c - l) for r, c in cells))


def _components(cells):
    left = set(cells)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in N4:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _buffer(cells):
    return {(r + a, c + b) for r, c in cells
            for a in (-1, 0, 1) for b in (-1, 0, 1)}


def _outline(center, radius):
    r, c = center
    return {(r + a, c + b) for a in range(-radius, radius + 1)
            for b in range(-radius, radius + 1) if abs(a) + abs(b) == radius}


def _cap(center, radius, direction):
    dr, dc = DIRECTIONS[direction]
    return [(center[0] + dr * (k + 1), center[1] + dc * (radius - k + 1))
            for k in range(radius + 1)]


def _base(shape, background):
    return np.asarray(background, dtype=np.int8)[np.indices(shape).sum(axis=0) & 1]


def _background(grid):
    parity = np.indices(grid.shape).sum(axis=0) & 1
    colors = []
    for p in (0, 1):
        histogram = np.bincount(grid[parity == p].astype(int), minlength=10)
        color = int(histogram.argmax())
        assert histogram[color] > np.count_nonzero(parity == p) // 2
        colors.append(color)
    assert colors[0] != colors[1]
    return tuple(colors)


def _footprint(emitter, shape):
    result = _outline(emitter['center'], emitter['radius']) | {emitter['center']}
    if not all(_inside(q, shape) for q in result):
        return None
    for d in emitter['caps']:
        visible = {q for q in _cap(emitter['center'], emitter['radius'], d)
                   if _inside(q, shape)}
        if len(visible) < 2:
            return None
        result |= visible
    return result


def _beams(emitters, shape):
    beams = []
    for i, e in enumerate(emitters):
        for d, color in sorted(e['caps'].items()):
            cap = _cap(e['center'], e['radius'], d)
            dr, dc = DIRECTIONS[d]
            lanes = []
            for endpoint in (cap[0], cap[-1]):
                path = []
                if _inside(endpoint, shape):
                    q = endpoint[0] + dr, endpoint[1] + dc
                    while _inside(q, shape):
                        path.append(q)
                        q = q[0] + dr, q[1] + dc
                lanes.append(path)
            beams.append(dict(emitter=i, direction=d, color=color, lanes=lanes,
                              cells=set(lanes[0]) | set(lanes[1])))
    return beams


def _network(emitters, shape):
    """Recover four-corner sockets from discrete endpoint rays, not filled strips."""
    footprints = [_footprint(e, shape) for e in emitters]
    assert all(q is not None for q in footprints)
    fixed = set().union(*footprints)
    assert sum(map(len, footprints)) == len(fixed)
    beams = _beams(emitters, shape)
    assert not any(b['cells'] & fixed for b in beams), 'ray hits a stationary object'
    docks, edges, used = [], [], set()
    for i, j in combinations(range(len(beams)), 2):
        a, b = beams[i], beams[j]
        hits = a['cells'] & b['cells']
        if not hits:
            continue
        va, vb = DIRECTIONS[a['direction']], DIRECTIONS[b['direction']]
        assert va[0] * va[1] != vb[0] * vb[1], 'parallel rays overlap'
        crossings = [set(x) & set(y) for x in a['lanes'] for y in b['lanes']]
        assert all(len(x) == 1 for x in crossings), 'partial four-corner socket'
        assert len(hits) == 4 and not hits & used, 'shared or triple crossing'
        assert a['emitter'] != b['emitter']
        used |= hits
        docks.append(hits)
        edges.append((a['emitter'], b['emitter']))
    assert docks
    reached = {0}
    for _ in emitters:
        reached |= {v for a, b in edges if a in reached or b in reached for v in (a, b)}
    assert len(reached) == len(emitters), 'the emitter network must be connected'
    return dict(fixed=fixed, beams=beams, docks=docks, edges=edges, pins=used,
                rays=set().union(*(b['cells'] for b in beams)))


def _analyze(input_grid):
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= n <= 30 for n in grid.shape)
    assert np.all((0 <= grid) & (grid <= 9))
    background = _background(grid)
    base = _base(grid.shape, background)
    emitters = []
    # A center dot, a complete monochrome outline and a cap identify an anchor.
    # Vectorized stencil detection keeps repeated input-only checks inexpensive.
    for radius in (2, 3, 4):
        h, w = grid.shape
        if min(h, w) < 2 * radius + 1:
            continue
        sh = h - 2 * radius, w - 2 * radius
        offsets = sorted(_outline((0, 0), radius))
        a, b = offsets[0]
        colors = grid[radius + a:radius + a + sh[0], radius + b:radius + b + sh[1]]
        centers = grid[radius:radius + sh[0], radius:radius + sh[1]]
        mask = ((colors != background[0]) & (colors != background[1]) &
                (centers != background[0]) & (centers != background[1]) &
                (centers != colors))
        for a, b in offsets[1:]:
            mask &= grid[radius + a:radius + a + sh[0],
                         radius + b:radius + b + sh[1]] == colors
        for a, b in np.argwhere(mask):
            center = int(a) + radius, int(b) + radius
            emitters.append(dict(center=center, radius=radius,
                                 color=int(grid[center[0] - radius, center[1]]), caps={}))
    emitters.sort(key=lambda e: e['center'])
    assert 2 <= len(emitters)
    marker_colors = {int(grid[e['center']]) for e in emitters}
    assert len(marker_colors) == 1
    marker = marker_colors.pop()
    covered = set()
    for e in emitters:
        for d in range(4):
            visible = [q for q in _cap(e['center'], e['radius'], d) if _inside(q, grid.shape)]
            values = {int(grid[q]) for q in visible}
            if len(visible) >= 2 and len(values) == 1:
                color = values.pop()
                if color not in (*background, marker, e['color']):
                    e['caps'][d] = color
        assert e['caps']
        cells = _footprint(e, grid.shape)
        assert not cells & covered
        covered |= cells
    network = _network(emitters, grid.shape)
    left = set(map(tuple, np.argwhere(grid != base))) - covered
    pieces = []
    for cells in _components(left):
        colors = {int(grid[q]) for q in cells}
        assert len(colors) == 2 and marker in colors
        color = (colors - {marker}).pop()
        assert color not in background
        pins = {q for q in cells if grid[q] == marker}
        body = cells - pins
        assert len(pins) == 4 and len(_components(body)) == 1
        pieces.append(dict(cells=cells, body=body, pins=pins, color=color))
    assert len(pieces) == len(network['docks'])
    return dict(background=background, marker=marker, emitters=emitters,
                pieces=pieces, network=network)


def _render(shape, background, marker, emitters, pieces, positions):
    out = _base(shape, background)
    for e in emitters:
        for q in _outline(e['center'], e['radius']):
            out[q] = e['color']
        out[e['center']] = marker
        for d, color in e['caps'].items():
            for q in _cap(e['center'], e['radius'], d):
                if _inside(q, shape):
                    out[q] = color
    network = _network(emitters, shape)
    for beam in network['beams']:
        for q in beam['cells']:
            out[q] = marker if q in network['pins'] else beam['color']
    for piece, delta in zip(pieces, positions):
        for q in _shift(piece['body'], delta):
            out[q] = piece['color']
        for q in _shift(piece['pins'], delta):
            out[q] = marker
    return out


def _reconstruct(grid, analysis=None):
    a = _analyze(grid) if analysis is None else analysis
    pieces, network = a['pieces'], a['network']
    options = []
    for p in pieces:
        choices = []
        pt, pl, _, _ = _box(p['pins'])
        for j, target in enumerate(network['docks']):
            if _normal(p['pins']) != _normal(target):
                continue
            tt, tl, _, _ = _box(target)
            delta = tt - pt, tl - pl
            body = _shift(p['body'], delta)
            if (all(_inside(q, grid.shape) for q in body) and
                    not body & (network['fixed'] | network['rays'])):
                choices.append((j, delta, body))
        assert choices, 'a rigid junction cannot fit any socket'
        options.append(choices)
    order = sorted(range(len(pieces)), key=lambda i: (len(options[i]), i))
    solutions = {}
    positions = [None] * len(pieces)

    def visit(k, used, occupied):
        if len(solutions) > 1:
            return
        if k == len(order):
            out = _render(grid.shape, a['background'], a['marker'], a['emitters'], pieces, positions)
            solutions.setdefault(out.tobytes(), (out, list(positions)))
            return
        i = order[k]
        for j, delta, body in options[i]:
            if j in used or body & occupied:
                continue
            positions[i] = delta
            visit(k + 1, used | {j}, occupied | body)
            positions[i] = None

    visit(0, set(), set())
    assert len(solutions) == 1, 'docking must determine one output grid'
    return next(iter(solutions.values()))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the docking sites and the uniquely translated pieces from this input."""
    grid = np.asarray(input_grid)
    return _reconstruct(grid)[0]


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _candidate(center, radius, direction):
    return dict(center=center, radius=radius, caps={direction: 1}, color=2)


def _grow(rng, shape, count, max_radius, root_radius, tree_only, cycle_only,
          force_leaf=False, different_keys=False):
    """Grow on free coordinates; a new source must make complete real crossings."""
    h, w = shape
    for restart in range(28):
        R = root_radius if root_radius is not None else int(rng.integers(2, max_radius + 1))
        root = _candidate((int(rng.integers(R + 2, R + 5)),
                           int(rng.integers(R + 2, R + 5))), R, 2)
        emitters = [root]
        fixed = _footprint(root, shape)
        if fixed is None:
            continue
        forbidden = _buffer(fixed)
        beams = _beams(emitters, shape)
        all_rays = set().union(*(b['cells'] for b in beams))
        for index in range(1, count):
            for attempt in range(900):
                radius = int(rng.integers(2, max_radius + 1))
                if min(shape) <= 2 * radius + 4:
                    continue
                center = (int(rng.integers(radius + 2, h - radius - 1)),
                          int(rng.integers(radius + 2, w - radius - 1)))
                d = 3 if force_leaf and index == 1 else int(rng.integers(4))
                e = _candidate(center, radius, d)
                cells = _footprint(e, shape)
                if cells is None or cells & (forbidden | all_rays):
                    continue
                new_ray = _beams([e], shape)[0]
                if new_ray['cells'] & fixed:
                    continue
                additions, good = [], True
                for old in beams:
                    hits = old['cells'] & new_ray['cells']
                    if not hits:
                        continue
                    if DIRECTIONS[old['direction']][0] * DIRECTIONS[old['direction']][1] == DIRECTIONS[d][0] * DIRECTIONS[d][1]:
                        good = False
                        break
                    corners = [set(x) & set(y) for x in old['lanes'] for y in new_ray['lanes']]
                    if not all(len(x) == 1 for x in corners):
                        good = False
                        break
                    additions.append(hits)
                if not good or not additions or (tree_only and len(additions) != 1):
                    continue
                proposal = emitters + [e]
                try:
                    net = _network(proposal, shape)
                except AssertionError:
                    continue
                if len(net['docks']) > 5:
                    continue
                emitters = proposal
                fixed |= cells
                forbidden |= _buffer(cells)
                beams = net['beams']
                all_rays = net['rays']
                break
            else:
                break
        if len(emitters) != count:
            continue
        net = _network(emitters, shape)
        if cycle_only and len(net['docks']) < count:
            continue
        if different_keys and len({_normal(q) for q in net['docks']}) < 2:
            continue
        return emitters
    return None


def _add_free_caps(rng, emitters, shape, max_extra):
    # Additional sides may be clipped or contribute zero new pixels. They are
    # accepted only if they add no extra socket or conflicting ray ownership.
    emitters = [dict(e, caps=dict(e['caps'])) for e in emitters]
    original = _network(emitters, shape)
    slots = [(i, d) for i, e in enumerate(emitters) for d in range(4) if d not in e['caps']]
    rng.shuffle(slots)
    added = 0
    for i, d in slots:
        if added >= max_extra:
            break
        emitters[i]['caps'][d] = 1
        try:
            now = _network(emitters, shape)
            if {frozenset(q) for q in now['docks']} != {frozenset(q) for q in original['docks']}:
                raise AssertionError('new socket')
        except AssertionError:
            del emitters[i]['caps'][d]
        else:
            added += 1
    return emitters


def _path_between(start, targets, domain):
    queue, previous = deque([start]), {start: None}
    finish = None
    while queue:
        q = queue.popleft()
        if q in targets:
            finish = q
            break
        for dr, dc in N4:
            p = q[0] + dr, q[1] + dc
            if p in domain and p not in previous:
                previous[p] = q
                queue.append(p)
    if finish is None:
        return None
    path = set()
    while finish is not None:
        path.add(finish)
        finish = previous[finish]
    return path


def _junction(rng, pins, network, shape, color):
    us = [r + c for r, c in pins]
    vs = [r - c for r, c in pins]
    t, l, b, rr = _box(pins)
    domain = {(r, c) for r in range(t, b + 1) for c in range(l, rr + 1)
              if min(us) < r + c < max(us) and min(vs) < r - c < max(vs)}
    domain -= network['rays'] | network['fixed']
    near = [{(r + dr, c + dc) for dr, dc in N4} & domain for r, c in sorted(pins)]
    compatible = [part for part in _components(domain) if all(part & ns for ns in near)]
    if not compatible:
        return None
    domain = max(compatible, key=len)
    style = int(rng.integers(4))
    if style == 0:
        body = set(domain)
    else:
        start = _pick(rng, sorted(domain))
        body = {start}
        for neighbors in near:
            targets = neighbors & domain
            route = _path_between(_pick(rng, sorted(targets)), body, domain)
            if route is None:
                return None
            body |= route
        if style == 3:
            ring = {q for q in domain if min(q[0] + q[1] - min(us), max(us) - q[0] - q[1],
                     q[0] - q[1] - min(vs), max(vs) - q[0] + q[1]) <= 2}
            if len(_components(ring)) == 1 and all(ring & ns for ns in near):
                body = ring
        if style == 2:
            target = int(rng.integers(len(body), len(domain) + 1))
            while len(body) < target:
                frontier = sorted({(r + dr, c + dc) for r, c in body for dr, dc in N4} & domain - body)
                if not frontier:
                    break
                body.add(_pick(rng, frontier))
    assert len(_components(body)) == 1
    assert len(_components(body | pins)) == 1
    return dict(body=body, pins=set(pins), cells=body | pins, color=color)


def _scatter(rng, shape, background, marker, emitters, pieces):
    fixed = set().union(*(_footprint(e, shape) for e in emitters))
    target_pin_sets = {frozenset(p['pins']) for p in pieces}
    for restart in range(24):
        forbidden = _buffer(fixed)
        positions = [None] * len(pieces)
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i]['cells']))
        for i in order:
            p = pieces[i]
            t, l, b, r = _box(p['cells'])
            for attempt in range(400):
                delta = (int(rng.integers(shape[0] - (b - t))) - t,
                         int(rng.integers(shape[1] - (r - l))) - l)
                if delta == (0, 0):
                    continue
                if frozenset(_shift(p['pins'], delta)) in target_pin_sets:
                    continue
                cells = _shift(p['cells'], delta)
                if cells & forbidden:
                    continue
                positions[i] = delta
                forbidden |= _buffer(cells)
                break
            else:
                break
        if all(d is not None for d in positions):
            inp = _base(shape, background)
            for e in emitters:
                for q in _outline(e['center'], e['radius']):
                    inp[q] = e['color']
                inp[e['center']] = marker
                for d, color in e['caps'].items():
                    for q in _cap(e['center'], e['radius'], d):
                        if _inside(q, shape):
                            inp[q] = color
            for p, delta in zip(pieces, positions):
                for q in _shift(p['body'], delta):
                    inp[q] = p['color']
                for q in _shift(p['pins'], delta):
                    inp[q] = marker
            return inp
    return None


def _teaching(a, output):
    evidence = set()
    base = _base(output.shape, a['background'])
    for e in a['emitters']:
        for d in e['caps']:
            dr, dc = DIRECTIONS[d]
            cap = _cap(e['center'], e['radius'], d)
            probes = [(r + dr * k, c + dc * k) for r, c in cap[1:-1] for k in (1, 2)]
            lane_probes = [(r + dr * k, c + dc * k) for r, c in (cap[0], cap[-1]) for k in (1, 2)]
            if (all(_inside(q, output.shape) and output[q] == base[q] for q in probes) and
                    all(_inside(q, output.shape) and output[q] == e['caps'][d] for q in lane_probes)):
                evidence.add(d)
    return evidence


def _coordinate_rays(a, shape):
    """Second ray calculation: signed coordinate equations, without path walks."""
    rows, cols = np.indices(shape)
    masks = []
    for e in a['emitters']:
        for d in sorted(e['caps']):
            dr, dc = DIRECTIONS[d]
            u = (rows - e['center'][0]) * dr
            v = (cols - e['center'][1]) * dc
            masks.append((np.abs(u - v) == e['radius']) & (np.minimum(u, v) >= 2))
    multiplicity = np.sum(masks, axis=0)
    assert np.all(multiplicity <= 2)
    return masks, multiplicity


def _make_pair(rng, params, count, witness=None, cycle=False, tree=False):
    for retry in range(64):
        shape = ((30, 30) if retry >= 8 else
                 tuple(int(rng.integers(28 if count >= 4 else 24, 31)) for _ in range(2)))
        root_radius = params['max_radius'] if witness == 0 else None
        es = _grow(rng, shape, count, params['max_radius'], root_radius,
                   tree, cycle, force_leaf=witness is not None,
                   different_keys=witness == 1)
        if es is None:
            continue
        # Translate the complete source arrangement within its feasible margins.
        cells = set().union(*(_footprint(e, shape) for e in es))
        t, l, b, r = _box(cells)
        delta = (int(rng.integers(shape[0] - (b - t))) - t,
                 int(rng.integers(shape[1] - (r - l))) - l)
        moved = [dict(e, center=(e['center'][0] + delta[0], e['center'][1] + delta[1])) for e in es]
        try:
            _network(moved, shape)
            es = moved
        except AssertionError:
            pass
        es = _add_free_caps(rng, es, shape, int(rng.integers(0, 4)))
        background = (params['background'] if params['fixed_palette'] else
                      tuple(map(int, rng.choice([c for c in range(10) if c != params['marker']], 2, replace=False))))
        marker = params['marker']
        available = [c for c in range(10) if c not in (*background, marker)]
        for e in es:
            e['color'] = int(rng.choice(available))
            cap_bank = [c for c in available if c != e['color']]
            e['caps'] = {d: int(rng.choice(cap_bank)) for d in e['caps']}
        network = _network(es, shape)
        body_bank = list(map(int, rng.choice(available, int(rng.integers(1, min(4, len(network['docks'])) + 1)), replace=False)))
        pieces = [_junction(rng, pins, network, shape, int(rng.choice(body_bank))) for pins in network['docks']]
        if any(p is None for p in pieces):
            continue
        all_bodies = set().union(*(p['body'] for p in pieces))
        if len(all_bodies) != sum(len(p['body']) for p in pieces):
            continue
        inp = _scatter(rng, shape, background, marker, es, pieces)
        if inp is None:
            continue
        out = _render(shape, background, marker, es, pieces, [(0, 0)] * len(pieces))
        k = ((params['orientation'] + 2 * witness) % 4 if witness is not None
             else int(rng.integers(4)))
        inp, out = np.rot90(inp, k).copy(), np.rot90(out, k).copy()
        flip = params['reflection'] if witness is not None else bool(rng.integers(2))
        if flip:
            inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
        try:
            analysis = _analyze(inp)
            predicted, positions = _reconstruct(inp, analysis)
            if not np.array_equal(predicted, out) or any(d == (0, 0) for d in positions):
                continue
            if witness is not None:
                # Both mandatory adjacent cap directions have endpoint-only
                # witnesses; the second demonstration supplies their opposites.
                primary = []
                for d in (2, 3):
                    v = np.array(DIRECTIONS[d])
                    for _ in range(k):
                        v = np.array([-v[1], v[0]])
                    if flip:
                        v[1] *= -1
                    primary.append(DIRECTIONS.index(tuple(map(int, v))))
                if not set(primary) <= _teaching(analysis, out):
                    continue
            return dict(input=inp, output=out)
        except AssertionError:
            continue
    raise RuntimeError('bounded rendezvous construction exhausted')


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' is outside its legal range')
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    marker = int(rng.integers(10))
    params = dict(marker=marker,
                  background=tuple(map(int, rng.choice([c for c in range(10) if c != marker], 2, replace=False))),
                  fixed_palette=bool(rng.integers(2)), max_radius=int(rng.choice([3, 4])),
                  orientation=int(rng.integers(4)), reflection=bool(rng.integers(2)),
                  cycle_regime=bool(rng.integers(2)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            witness = index if split == 'train' and index < 2 else None
            sources = (2 if witness == 0 else 3 if witness == 1 else
                       int(rng.integers(2, 5 if params['cycle_regime'] else 4)) if split == 'train' else 4)
            for _ in range(12):
                pair = _make_pair(rng, params, sources, witness,
                                  cycle=split == 'test' and params['cycle_regime'],
                                  tree=split == 'train')
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair, a=None, positions=None):
    a = _analyze(np.asarray(pair['input'])) if a is None else a
    net = a['network']
    degrees = Counter(v for edge in net['edges'] for v in edge)
    return dict(shape=list(np.asarray(pair['input']).shape), background=a['background'], marker=a['marker'],
                sources=len(a['emitters']), junctions=len(a['pieces']),
                cycle=len(net['edges']) >= len(a['emitters']), degree=max(degrees.values()),
                radii=sorted({e['radius'] for e in a['emitters']}),
                masks=[sum(1 << d for d in e['caps']) for e in a['emitters']],
                keys=[_normal(q) for q in net['docks']],
                body_shapes=[_normal(p['body']) for p in a['pieces']],
                body_areas=[len(p['body']) for p in a['pieces']],
                reuse=len(a['pieces']) - len({p['color'] for p in a['pieces']}),
                ray_lengths=[len(path) for b in net['beams'] for path in b['lanes']],
                clipped=sum(any(not _inside(q, np.asarray(pair['input']).shape)
                                for q in _cap(e['center'], e['radius'], d))
                            for e in a['emitters'] for d in e['caps']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence, key_witness = set(), set(), False
    features = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            for g in (inp, out):
                assert g.ndim == 2 and np.issubdtype(g.dtype, np.integer)
                assert all(1 <= n <= 30 for n in g.shape) and np.all((0 <= g) & (g <= 9))
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a = _analyze(inp)
            predicted, positions = _reconstruct(inp, a)
            assert np.array_equal(predicted, out)
            assert all(d != (0, 0) for d in positions)
            net = a['network']
            masks, multiplicity = _coordinate_rays(a, inp.shape)
            assert all(set(map(tuple, np.argwhere(mask))) == beam['cells']
                       for mask, beam in zip(masks, net['beams']))
            assert set(map(tuple, np.argwhere(multiplicity == 2))) == net['pins']
            # Whole rigid bodies, stationary geometry, pins and background
            # erasure are checked explicitly, beyond full-grid solver equality.
            for q in net['fixed']:
                assert out[q] == inp[q]
            claimed = set(net['fixed']) | net['rays']
            occupied = set()
            for p, delta in zip(a['pieces'], positions):
                moved_body = _shift(p['body'], delta)
                moved_pins = _shift(p['pins'], delta)
                assert moved_pins in net['docks']
                assert not moved_body & (occupied | net['fixed'] | net['rays'])
                assert all(out[q] == p['color'] for q in moved_body)
                assert all(out[q] == a['marker'] for q in moved_pins)
                occupied |= moved_body
                claimed |= moved_body | moved_pins
            base = _base(inp.shape, a['background'])
            untouched = np.ones(inp.shape, dtype=bool)
            for q in claimed:
                untouched[q] = False
            assert np.array_equal(out[untouched], base[untouched])
            assert all(out[q] == a['marker'] for q in net['pins'])
            f = _features(pair, a, positions)
            features[split].append(f)
            if split == 'train':
                evidence |= _teaching(a, out)
                key_witness |= len(set(f['keys'])) >= 2
    assert evidence == set(range(4)), 'training must teach endpoint-only extension in every direction'
    assert key_witness, 'equal-area, different-geometry docking must be demonstrated'
    train = features['train']
    tests = features['test']
    # Generated puzzles deliberately use either a source-count shift or cycles.
    assert all(t['sources'] > max(f['sources'] for f in train) or
               (t['cycle'] and not any(f['cycle'] for f in train)) for t in tests)
    assert max(r for f in tests for r in f['radii']) <= max(r for f in train for r in f['radii'])


def self_test():
    """Bounded 42-seed coverage and semantic negative checks; no external oracle."""
    records, fingerprints = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(pair['input']) for s in ('train', 'test') for pair in puzzle[s]))
        records.extend(_features(pair) for s in ('train', 'test') for pair in puzzle[s])
    assert len(fingerprints) == 42
    assert {f['sources'] for f in records} == {2, 3, 4}
    assert {r for f in records for r in f['radii']} == {2, 3, 4}
    assert {f['junctions'] for f in records} >= {1, 2, 3, 4}
    assert {f['cycle'] for f in records} == {False, True}
    assert max(f['degree'] for f in records) >= 3
    assert len({tuple(f['shape']) for f in records}) >= 15
    assert len({f['background'] for f in records}) >= 15
    assert len({f['marker'] for f in records}) >= 8
    assert len({key for f in records for key in f['keys']}) >= 7
    assert len({body for f in records for body in f['body_shapes']}) >= 30
    assert any(f['reuse'] > 0 for f in records)
    assert any(f['clipped'] > 0 for f in records)
    assert max(n for f in records for n in f['ray_lengths']) >= 20
    # Count boundaries and noninteger requests are part of the public contract.
    for nt, nq in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(8123, nt, nq)
        assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(0, 2, 1)
    again = generate_puzzle(0, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in ('train', 'test')
               for x, y in zip(p[s], again[s]) for k in ('input', 'output'))
    a = _analyze(p['train'][1]['input'])
    assert len({_normal(q) for q in a['network']['docks']}) >= 2
    # Without B, a bare ray scaffold loses every translated payload cell.
    for pair in p['train']:
        a = _analyze(pair['input'])
        predicted, positions = _reconstruct(pair['input'], a)
        bare = _render(pair['input'].shape, a['background'], a['marker'], a['emitters'], [], [])
        assert not np.array_equal(predicted, bare)
        # Grant B its correct translations, then remove A's actual extensions:
        # the transported pieces alone still fail to produce the full answer.
        without_extension = predicted.copy()
        base = _base(predicted.shape, a['background'])
        for q in a['network']['rays'] - a['network']['pins']:
            without_extension[q] = base[q]
        assert not np.array_equal(predicted, without_extension)
        # Without A, no beams or crossing socket exists; leaving scattered
        # pieces where found also differs. Interior-lane witnesses explicitly
        # refute whole-cap propagation, in all four directions across training.
        assert not np.array_equal(predicted, pair['input'])
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in ('train', 'test')}
    a = _analyze(damaged['test'][0]['input'])
    q = min(a['network']['pins'])
    damaged['test'][0]['output'][q] = a['background'][(q[0] + q[1]) & 1]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('missing crossing pin accepted')
    # Equal pin masks with distinguishable payloads really can be ambiguous:
    # neither payload hue nor scatter order is an authorized tie breaker.
    es = [dict(center=(4, 4), radius=2, color=2, caps={2: 3}),
          dict(center=(4, 24), radius=2, color=4, caps={3: 6}),
          dict(center=(24, 12), radius=2, color=8, caps={1: 3})]
    network = _network(es, (30, 30))
    assert len(network['docks']) == 2
    assert len({_normal(q) for q in network['docks']}) == 1
    fixture_rng = np.random.default_rng(417)
    parts = [_junction(fixture_rng, pins, network, (30, 30), color)
             for pins, color in zip(network['docks'], (1, 9))]
    ambiguous = _scatter(fixture_rng, (30, 30), (0, 7), 5, es, parts)
    assert ambiguous is not None
    _analyze(ambiguous)  # The negative fixture is geometrically well formed.
    try:
        solve(ambiguous)
    except AssertionError as error:
        assert 'docking must determine one output grid' in str(error)
    else:
        raise AssertionError('distinct-payload ambiguity was accepted')
    return dict(seeds=42, pairs=len(records), sources=sorted({f['sources'] for f in records}),
                junctions=sorted({f['junctions'] for f in records}),
                keys=len({key for f in records for key in f['keys']}),
                body_shapes=len({body for f in records for body in f['body_shapes']}),
                canvas_shapes=len({tuple(f['shape']) for f in records}),
                cycles=sum(f['cycle'] for f in records), clipped_caps=sum(f['clipped'] for f in records))


if __name__ == '__main__':
    print(self_test())
