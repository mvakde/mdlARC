"""One-way mirrors for hollow diamond beams; all conventions live in the input.

Only standard-library modules and NumPy are used. No parent code is imported.
"""
import numpy as np


VECTORS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
AXIAL = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= x <= 30 for x in g.shape), "invalid grid shape")
    _require(np.issubdtype(g.dtype, np.integer), "grid must contain integers")
    _require(np.all((0 <= g) & (g <= 9)), "invalid color")
    return g.astype(np.int8, copy=False)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    offsets = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                    if (r or c) and (diagonal or not (r and c)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, cells = [start], {start}
        while todo:
            r, c = todo.pop()
            for dr, dc in offsets:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    todo.append(p)
        result.append(cells)
    return result


def _field(cut):
    """Two topological regions and the local normal at straight seam cells."""
    cells = set(map(tuple, np.argwhere(cut)))
    _require(len(cells) >= 5, "short membrane")
    graph = {p: [tuple(p[i] + d[i] for i in (0, 1)) for d in AXIAL
                 if tuple(p[i] + d[i] for i in (0, 1)) in cells] for p in cells}
    ends = [p for p in cells if len(graph[p]) == 1]
    _require(len(ends) == 2 and all(1 <= len(graph[p]) <= 2 for p in cells),
             "membrane must be a simple orthogonal path")
    edge = lambda p: p[0] in (0, cut.shape[0] - 1) or p[1] in (0, cut.shape[1] - 1)
    _require(all(edge(p) for p in ends) and all(not edge(p) for p in cells - set(ends)),
             "only the membrane endpoints may touch the boundary")
    _require(len(_components(cut)) == 1, "disconnected membrane")
    comps = _components(~cut)
    _require(len(comps) == 2 and min(map(len, comps)) >= 8, "membrane must separate two regions")
    labels = np.full(cut.shape, -1, dtype=np.int8)
    for index, comp in enumerate(comps):
        for p in comp:
            labels[p] = index
    normals = {}
    for p, neighbors in graph.items():
        if len(neighbors) == 2:
            if neighbors[0][1] == neighbors[1][1] == p[1]:
                normals[p] = 1  # vertical membrane reverses column velocity
            elif neighbors[0][0] == neighbors[1][0] == p[0]:
                normals[p] = 0  # horizontal membrane reverses row velocity
    return labels, normals


def _base(shape, colors):
    return np.asarray(colors, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _outline(center, radius):
    r, c = center
    return {(r + dr, c + dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1) if abs(dr) + abs(dc) == radius}


def _cap(center, radius, direction):
    r, c = center
    dr, dc = VECTORS[direction]
    return [(r + dr * (k + 1), c + dc * (radius - k + 1)) for k in range(radius + 1)]


def _decode(g):
    """Recover the checkerboard, membrane, marker, and every emitter from pixels."""
    parity = np.indices(g.shape).sum(axis=0) % 2
    bg = []
    for k in (0, 1):
        counts = np.bincount(g[parity == k], minlength=10)
        color = int(counts.argmax())
        _require(counts[color] > np.count_nonzero(parity == k) / 2, "ambiguous checkerboard")
        bg.append(color)
    _require(bg[0] != bg[1], "checkerboard requires two colors")
    base = _base(g.shape, bg)
    controls = []
    for color in set(map(int, np.unique(g))) - set(bg):
        comps = sorted(_components(g == color, True), key=len)
        if len(comps) != 2 or len(comps[0]) != 1:
            continue
        cut = np.zeros(g.shape, dtype=bool)
        for p in comps[1]:
            cut[p] = True
        try:
            labels, normals = _field(cut)
        except AssertionError:
            continue
        marker = next(iter(comps[0]))
        if any(max(abs(marker[0] - p[0]), abs(marker[1] - p[1])) < 2 for p in comps[1]):
            continue
        controls.append((color, cut, marker, labels, normals))
    _require(len(controls) == 1, "membrane and isolated marker must be unique")
    control, cut, marker, labels, normals = controls[0]
    side = int(labels[marker])
    _require(side in (0, 1), "marker must lie inside one region")
    covered = set(map(tuple, np.argwhere(cut))) | {marker}
    emitters = []
    h, w = g.shape
    for radius in (2, 3, 4):
        if min(h, w) < 2 * radius + 1:
            continue
        sh = (h - 2 * radius, w - 2 * radius)
        offsets = sorted(_outline((0, 0), radius))
        dr, dc = offsets[0]
        colors = g[radius + dr:radius + dr + sh[0], radius + dc:radius + dc + sh[1]]
        candidates = ~np.isin(colors, [*bg, control])
        for dr, dc in offsets[1:]:
            candidates &= g[radius + dr:radius + dr + sh[0], radius + dc:radius + dc + sh[1]] == colors
        for rr, cc in np.argwhere(candidates):
            center = (int(rr) + radius, int(cc) + radius)
            emitters.append(dict(center=center, radius=radius, struct=int(colors[rr, cc]), caps={}))
    _require(1 <= len(emitters) <= 3, "need one to three complete diamonds")
    for e in emitters:
        outline = _outline(e['center'], e['radius'])
        _require(not outline & covered, "overlapping objects")
        covered |= outline
    for e in emitters:
        own = _outline(e['center'], e['radius'])
        for direction in range(4):
            cap = [p for p in _cap(e['center'], e['radius'], direction) if _inside(p, g.shape)]
            if len(cap) < 2:
                continue
            values = {int(g[p]) for p in cap}
            if len(values) == 1 and not values & {*bg, control, e['struct']}:
                _require(not set(cap) & covered, "overlapping cap")
                e['caps'][direction] = values.pop()
                covered.update(cap)
                own.update(cap)
        _require(e['caps'], "diamond has no active side")
        regions = {int(labels[p]) for p in own}
        _require(len(regions) == 1 and -1 not in regions, "emitter must lie wholly inside one region")
        e['region'] = regions.pop()
    _require(covered == set(map(tuple, np.argwhere(g != base))), "unexplained foreground")
    emitters.sort(key=lambda e: e['center'])
    return dict(bg=tuple(bg), base=base, control=control, cut=cut, marker=marker,
                labels=labels, normals=normals, side=side, emitters=emitters, occupied=covered)


def _walk(start, velocity, labels, normals, side):
    """Trace one endpoint. A contact uses the tangent of the cell ahead.

    Reflect before entering the membrane. Transmit by skipping its protected
    cell. A valid scene has no corner contact, loops, or ambiguous crossings.
    """
    p, v = start, velocity
    states, path, events = set(), [], []
    while True:
        state = (p, v)
        _require(state not in states, "cyclic ray")
        states.add(state)
        q = (p[0] + v[0], p[1] + v[1])
        if not _inside(q, labels.shape):
            return path, events
        if labels[q] == -1:
            _require(q in normals, "ray hits a membrane bend or endpoint")
            if labels[p] == side:
                axis = normals[q]
                vv = list(v)
                vv[axis] = -vv[axis]
                v = tuple(vv)
                events.append(('reflect', q, axis))
                q = (p[0] + v[0], p[1] + v[1])
                _require(_inside(q, labels.shape) and labels[q] == labels[p], "ambiguous reflection")
            else:
                events.append(('transmit', q, normals[q]))
                q = (q[0] + v[0], q[1] + v[1])
                if not _inside(q, labels.shape):
                    return path, events
                _require(labels[q] == side, "transmission must enter the marked region")
        else:
            _require(labels[q] == labels[p], "ray crosses the membrane without a visible contact")
        path.append(q)
        p = q


def _trace(info, starts='ends', forced_side=None):
    """Return claims and traces, with optional ablations for teaching checks."""
    claims, traces = {}, []
    for e in info['emitters']:
        for direction, color in sorted(e['caps'].items()):
            cap = _cap(e['center'], e['radius'], direction)
            launch = (cap[0], cap[-1]) if starts == 'ends' else cap
            for p in launch:
                if not _inside(p, info['cut'].shape):
                    continue
                path, events = _walk(p, VECTORS[direction], info['labels'], info['normals'],
                                     info['side'] if forced_side is None else forced_side)
                for q in path:
                    _require(q not in info['occupied'], "ray hits a preserved input object")
                    _require(q not in claims or claims[q] == color, "different-colored rays collide")
                    claims[q] = color
                traces.append(dict(start=p, path=path, events=events, direction=direction,
                                   radius=e['radius'], color=color))
    return claims, traces


def solve(input_grid: np.ndarray) -> np.ndarray:
    g = _grid(input_grid)
    info = _decode(g)
    claims, _ = _trace(info)
    out = g.copy()
    for p, color in claims.items():
        out[p] = color
    return out


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _sites(cut, labels, side, excluded):
    near = cut.copy()
    h, w = cut.shape
    for r, c in np.argwhere(cut):
        near[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(w, c + 2)] = True
    return [tuple(p) for p in np.argwhere((labels == side) & ~near) if tuple(p) not in excluded]


def _sample_cut(rng, shape, family):
    h, w = shape
    cut = np.zeros(shape, dtype=bool)
    if family == 'vertical':
        cut[:, int(rng.integers(5, w - 5))] = True
    elif family == 'horizontal':
        cut[int(rng.integers(5, h - 5)), :] = True
    elif family == 'elbow':
        r, c = int(rng.integers(7, h - 6)), int(rng.integers(6, w - 7))
        cut[:r + 1, c] = True
        cut[r, c:] = True
    elif family == 'stepped':
        r = int(rng.integers(7, h - 7))
        c1 = int(rng.integers(6, w - 10))
        c2 = int(rng.integers(c1 + 3, w - 6))
        cut[:r + 1, c1] = True
        cut[r, c1:c2 + 1] = True
        cut[r:, c2] = True
    else:
        raise ValueError('unknown membrane family')
    if family in ('elbow', 'stepped'):
        if rng.integers(2):
            cut = cut[::-1].copy()
        if rng.integers(2):
            cut = cut[:, ::-1].copy()
        if rng.integers(2):
            cut = cut.T.copy()
    _field(cut)
    return cut


def _paint(shape, bg, control, cut, marker, emitters):
    g = _base(shape, bg)
    g[cut] = control
    g[marker] = control
    for e in emitters:
        for p in _outline(e['center'], e['radius']):
            g[p] = e['struct']
        for direction, color in e['caps'].items():
            for p in _cap(e['center'], e['radius'], direction):
                if _inside(p, shape):
                    g[p] = color
    return g


def _geometry_traces(emitters, cut, side, field=None):
    labels, normals = _field(cut) if field is None else field
    occupied = set(map(tuple, np.argwhere(cut)))
    for e in emitters:
        occupied |= _outline(e['center'], e['radius'])
        occupied |= {p for d in e['caps'] for p in _cap(e['center'], e['radius'], d) if _inside(p, cut.shape)}
    info = dict(emitters=emitters, cut=cut, labels=labels, normals=normals, side=side, occupied=occupied)
    return _trace(info)


def _finish(rng, bg, control, cut, emitters, side, twin=False):
    labels, normals = _field(cut)
    occupied = set().union(*[_outline(e['center'], e['radius']) |
                            {p for d in e['caps'] for p in _cap(e['center'], e['radius'], d)
                             if _inside(p, cut.shape)} for e in emitters])
    sides = (side, 1 - side) if twin else (side,)
    paths = [_geometry_traces(emitters, cut, s, (labels, normals))[0] for s in sides]
    excluded = occupied | set().union(*(set(p) for p in paths))
    pairs = []
    for s in sides:
        sites = _sites(cut, labels, s, excluded)
        _require(sites, 'no isolated marker site away from the rays')
        marker = sites[int(rng.integers(len(sites)))]
        g = _paint(cut.shape, bg, control, cut, marker, emitters)
        pairs.append({'input': g, 'output': solve(g)})
    return pairs


def _witness(rng, radius, family):
    """A clean four-cap scene with both long reflected and transmitted beams."""
    # Construct in vertical coordinates, then transpose for a horizontal seam.
    h = int(rng.integers(max(19, 2 * radius + 11), 31))
    w = int(rng.integers(3 * radius + 15, 31))
    seam = int(rng.integers(3 * radius + 9, w - 4))
    cut = np.zeros((h, w), dtype=bool)
    cut[:, seam] = True
    r = int(rng.integers(radius + 4, h - radius - 4))
    # Keep the cap near enough to the seam for a visible encounter; leave two
    # untouched continuation steps before contact and several after it.
    c = seam - 2 * radius - int(rng.integers(4, 7))
    _require(c >= radius + 3, 'insufficient witness margin')
    palette = list(map(int, rng.permutation(10)))
    bg, control, struct = tuple(palette[:2]), palette[2], palette[3]
    caps = {d: palette[4 + d] for d in range(4)}
    e = dict(center=(r, c), radius=radius, struct=struct, caps=caps)
    if rng.integers(2):
        cut = cut[:, ::-1].copy()
        e['center'] = (r, w - 1 - c)
        e['caps'] = {(1 - d) % 4: color for d, color in caps.items()}
    if family == 'horizontal':
        cut = cut.T.copy()
        e['center'] = e['center'][::-1]
        e['caps'] = {(-d) % 4: color for d, color in e['caps'].items()}
    pairs = _finish(rng, bg, control, cut, [e], int(rng.integers(2)), True)
    event_kinds = set()
    erased = []
    for pair in pairs:
        info = _decode(pair['input'])
        _, traces = _trace(info)
        event_kinds |= {ev[0] for t in traces for ev in t['events']}
        out = pair['output'].copy()
        out[info['marker']] = info['base'][info['marker']]
        erased.append(out)
    _require(event_kinds == {'reflect', 'transmit'}, 'witness rays must reach the seam')
    _require(not np.array_equal(*erased), 'marker must change ray pixels')
    return pairs


def _random_scene(rng, max_radius, family, count, require_contact=True,
                  require_reflection=False, allow_clipping=True):
    """Bounded free placement with full geometric collision checks."""
    for restart in range(120):
        shape = (int(rng.integers(22, 31)), int(rng.integers(22, 31)))
        cut = _sample_cut(rng, shape, family)
        shape = cut.shape
        labels, normals = _field(cut)
        side = int(rng.integers(2))
        palette = list(map(int, rng.permutation(10)))
        bg, control = tuple(palette[:2]), palette[2]
        emitters, occupied = [], set()
        for _ in range(count):
            radius = int(rng.integers(2, max_radius + 1))
            # High cap counts remain available with multiple emitters.
            mask = int(rng.integers(1, 16))
            structural = int(rng.choice(palette[3:])) if rng.random() < .6 else palette[3]
            available = [color for color in palette[3:] if color != structural]
            bank = list(map(int, rng.choice(available, int(rng.integers(1, 5)), replace=False)))
            caps = {d: int(rng.choice(bank)) for d in range(4) if mask & (1 << d)}
            for attempt in range(160):
                center = (int(rng.integers(radius, shape[0] - radius)),
                          int(rng.integers(radius, shape[1] - radius)))
                own = _outline(center, radius)
                valid = True
                for d in caps:
                    visible = {p for p in _cap(center, radius, d) if _inside(p, shape)}
                    valid &= len(visible) >= 2
                    valid &= allow_clipping or len(visible) == radius + 1
                    own |= visible
                if not valid or own & occupied:
                    continue
                regions = {int(labels[p]) for p in own}
                if len(regions) != 1 or -1 in regions:
                    continue
                e = dict(center=center, radius=radius, struct=structural, caps=caps)
                try:
                    _geometry_traces(emitters + [e], cut, side, (labels, normals))
                except AssertionError:
                    continue
                emitters.append(e)
                occupied |= own
                break
            else:
                break
        if len(emitters) != count:
            continue
        try:
            claims, traces = _geometry_traces(emitters, cut, side, (labels, normals))
            if not claims or (require_contact and not any(t['events'] for t in traces)):
                continue
            if require_reflection and not any(ev[0] == 'reflect' for t in traces for ev in t['events']):
                continue
            return _finish(rng, bg, control, cut, emitters, side)[0]
        except AssertionError:
            continue
    raise RuntimeError('bounded placement search exhausted')


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    max_radius = int(rng.choice([2, 3, 4], p=[.35, .45, .20]))
    axis = str(rng.choice(['vertical', 'horizontal']))
    # Retry only clean witnesses, retaining sampled size and seam family.
    for _ in range(80):
        try:
            twins = _witness(rng, max_radius, axis)
            if _teaching(twins[0]) | _teaching(twins[1]) == set(range(4)):
                break
        except AssertionError:
            pass
    else:
        raise RuntimeError('could not construct endpoint witnesses')
    puzzle = {'train': twins, 'test': []}
    seen = {_key(p['input']) for p in twins}
    families = [axis]
    if nt >= 3:
        families.append('horizontal' if axis == 'vertical' else 'vertical')
    bent = str(rng.choice(['elbow', 'stepped']))
    if nt >= 4:
        families.append(bent)
    for index in range(2, nt):
        family = families[1] if index == 2 else (bent if index == 3 else str(rng.choice(families)))
        count = 1 if index == 2 else int(rng.integers(1, 4))
        for _ in range(20):
            pair = _random_scene(rng, max_radius, family, count, require_reflection=index == 2)
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError('unique input search exhausted')
        puzzle['train'].append(pair)
        seen.add(_key(pair['input']))
    allow_test_clipping = any(_metrics(pair)['clipped'] for pair in puzzle['train'])
    for _ in range(nq):
        # New interacting multi-object configurations, using trained tangent types.
        family = str(rng.choice(families))
        count = int(rng.integers(2, 4))
        for _ in range(20):
            pair = _random_scene(rng, max_radius, family, count, require_reflection=True,
                                 allow_clipping=allow_test_clipping)
            if (_key(pair['input']) not in seen
                    and not np.array_equal(_ignore_membrane(pair), pair['output'])):
                break
        else:
            raise RuntimeError('unique input search exhausted')
        seen.add(_key(pair['input']))
        puzzle['test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _teaching(pair):
    """Endpoint-only evidence: two changed end lanes and unchanged inner lanes."""
    g, out = _grid(pair['input']), _grid(pair['output'])
    info = _decode(g)
    evidence = set()
    for e in info['emitters']:
        for d, color in e['caps'].items():
            cap, v = _cap(e['center'], e['radius'], d), VECTORS[d]
            probes = [(p[0] + k * v[0], p[1] + k * v[1]) for p in cap for k in (1, 2)]
            if not all(_inside(p, g.shape) and g[p] == info['base'][p] for p in probes):
                continue
            ends = [(p[0] + k * v[0], p[1] + k * v[1]) for p in (cap[0], cap[-1]) for k in (1, 2)]
            inner = [(p[0] + k * v[0], p[1] + k * v[1]) for p in cap[1:-1] for k in (1, 2)]
            if all(out[p] == color for p in ends) and all(out[p] == g[p] for p in inner):
                evidence.add(d)
    return evidence


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence, events, taught_radii = set(), set(), set(), set()
    reflected_normals, tested_normals = set(), set()
    clipping = {'train': False, 'test': False}
    infos = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            g, out = _grid(pair['input']), _grid(pair['output'])
            _require(g.shape == out.shape and not np.array_equal(g, out), 'empty or resized answer')
            _require(_key(g) not in seen, 'duplicate input')
            seen.add(_key(g))
            info = _decode(g)
            clipping[split] |= any(any(not _inside(p, g.shape) for p in _cap(e['center'], e['radius'], d))
                                   for e in info['emitters'] for d in e['caps'])
            claims, traces = _trace(info)
            changed = set(map(tuple, np.argwhere(g != out)))
            _require(changed == set(claims), 'changed cells must be precisely endpoint paths')
            _require(all(out[p] == color for p, color in claims.items()), 'incorrect payload color')
            _require(np.array_equal(out[g != info['base']], g[g != info['base']]), 'input foreground was changed')
            # Input-only solution and pixel-preservation checks, not independent approval.
            _require(np.array_equal(solve(g), out), 'solve disagrees')
            infos.append(info)
            if split == 'train':
                ev = _teaching(pair)
                evidence |= ev
                if ev == set(range(4)):
                    taught_radii.update(e['radius'] for e in info['emitters'])
                events |= {event[0] for t in traces for event in t['events']}
                reflected_normals |= {ev[2] for t in traces for ev in t['events'] if ev[0] == 'reflect'}
            else:
                tested_normals |= {ev[2] for t in traces for ev in t['events'] if ev[0] == 'reflect'}
                _require(any(ev[0] == 'reflect' for t in traces for ev in t['events']),
                         'each test must visibly reflect an endpoint')
                _require(not np.array_equal(_ignore_membrane(pair), out),
                         'ignoring the membrane must change every test answer')
    _require(evidence == set(range(4)), 'training must expose all four endpoint directions')
    _require(events == {'reflect', 'transmit'}, 'training must demonstrate both membrane behaviors')
    _require(tested_normals <= reflected_normals, 'test reflection uses an undemonstrated tangent')
    _require(not clipping['test'] or clipping['train'], 'clipped tests need a clipped training example')
    _require(taught_radii and max(taught_radii) >= max(e['radius'] for i in infos for e in i['emitters']),
             'largest cap needs visible endpoint-only teaching')
    # Controlled marker comparison removes an orientation-only or color-only code.
    twins = False
    for i, a in enumerate(puzzle['train']):
        da = _decode(_grid(a['input']))
        ga = _grid(a['input']).copy()
        ga[da['marker']] = da['base'][da['marker']]
        for b in puzzle['train'][i + 1:]:
            db = _decode(_grid(b['input']))
            gb = _grid(b['input']).copy()
            gb[db['marker']] = db['base'][db['marker']]
            if ga.shape == gb.shape and np.array_equal(ga, gb) and da['side'] != db['side']:
                oa, ob = _grid(a['output']).copy(), _grid(b['output']).copy()
                oa[da['marker']] = da['base'][da['marker']]
                ob[db['marker']] = db['base'][db['marker']]
                twins |= not np.array_equal(oa, ob)
    _require(twins, 'need an effective opposite-marker demonstration')


def _metrics(pair):
    info = _decode(_grid(pair['input']))
    claims, traces = _trace(info)
    cut = info['cut']
    rows, cols = np.where(cut)
    family = 'horizontal' if len(set(rows)) == 1 else ('vertical' if len(set(cols)) == 1 else 'bent')
    return dict(shape=cut.shape, family=family, side=info['side'], count=len(info['emitters']),
                radii=sorted({e['radius'] for e in info['emitters']}),
                masks=[sum(1 << d for d in e['caps']) for e in info['emitters']],
                lengths=[len(t['path']) for t in traces], changed=len(claims),
                reflections=sum(ev[0] == 'reflect' for t in traces for ev in t['events']),
                transmissions=sum(ev[0] == 'transmit' for t in traces for ev in t['events']),
                clipped=sum(any(not _inside(p, cut.shape) for p in _cap(e['center'], e['radius'], d))
                            for e in info['emitters'] for d in e['caps']),
                layout=(cut.shape, cut.tobytes(), tuple((e['center'], e['radius'], tuple(e['caps']))
                                                       for e in info['emitters'])))


def _ignore_membrane(pair):
    """Ablation: Parent B's straight endpoint rays, preserving control pixels."""
    g = _grid(pair['input'])
    info = _decode(g)
    out = g.copy()
    for e in info['emitters']:
        for d, color in e['caps'].items():
            cap = _cap(e['center'], e['radius'], d)
            for p in (cap[0], cap[-1]):
                if not _inside(p, g.shape):
                    continue
                dr, dc = VECTORS[d]
                p = (p[0] + dr, p[1] + dc)
                while _inside(p, g.shape):
                    if g[p] == info['base'][p]:
                        out[p] = color
                    p = (p[0] + dr, p[1] + dc)
    return out


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features = [_metrics(pair) for p in puzzles for s in ('train', 'test') for pair in p[s]]
    _require(len({tuple(_key(pair['input']) for s in ('train', 'test') for pair in p[s]) for p in puzzles}) == 42,
             '42 seeds did not produce distinct puzzles')
    _require(len({f['layout'] for f in features}) >= 100, 'geometry variation collapsed')
    _require(len({f['shape'] for f in features}) >= 25, 'shape variation collapsed')
    _require({f['count'] for f in features} == {1, 2, 3}, 'emitter counts collapsed')
    _require({r for f in features for r in f['radii']} == {2, 3, 4}, 'radius variation collapsed')
    _require({m for f in features for m in f['masks']} == set(range(1, 16)), 'cap subset variation collapsed')
    _require(any(f['clipped'] for f in features), 'clipped caps absent')
    _require(any(f['family'] == 'bent' and f['reflections'] and f['transmissions'] for f in features),
             'combined bent-membrane behavior absent')
    _require(max(max(f['lengths']) for f in features) >= 15, 'long continuations absent')
    for p in puzzles:
        _require(any(not np.array_equal(_ignore_membrane(pair), pair['output']) for pair in p['train']),
                 'omitting Parent A did not change a demonstration')
        _require(set().union(*(_teaching(pair) for pair in p['train'])) == set(range(4)),
                 'omitting Parent B has no hollow-lane counterexample')
    repeated = generate_puzzle(0)
    _require(all(_key(a[k]) == _key(b[k]) for s in ('train', 'test')
                 for a, b in zip(puzzles[0][s], repeated[s]) for k in ('input', 'output')),
             'seed reproduction failed')
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(1000 + 10 * a + b, a, b)
        _require(len(p['train']) == a and len(p['test']) == b, 'count ignored')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Spatial symmetries and arbitrary role recoloring preserve the entire rule.
    rng = np.random.default_rng(90210)
    for p in puzzles[:6]:
        pair = p['test'][0]
        for g, answer in ((pair['input'].T, pair['output'].T),
                          (pair['input'][:, ::-1], pair['output'][:, ::-1]),
                          (pair['input'][::-1], pair['output'][::-1])):
            _require(np.array_equal(solve(g), answer), 'geometric covariance failed')
        permutation = rng.permutation(10)
        _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]),
                 'palette role covariance failed')
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    pair = damaged['test'][0]
    cell = tuple(np.argwhere(pair['input'] != pair['output'])[0])
    pair['output'][cell] = pair['input'][cell]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted missing continuation')
    pair = puzzles[0]['train'][0]
    info = _decode(pair['input'])
    malformed = pair['input'].copy()
    malformed[info['marker']] = info['base'][info['marker']]
    touching = next(p for p in map(tuple, np.argwhere(~info['cut']))
                    if malformed[p] == info['base'][p]
                    and any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) == 1
                            for q in map(tuple, np.argwhere(info['cut']))))
    malformed[touching] = info['control']
    try:
        solve(malformed)
    except AssertionError:
        pass
    else:
        raise AssertionError('solver accepted a marker touching the membrane')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds plus count boundaries)')
