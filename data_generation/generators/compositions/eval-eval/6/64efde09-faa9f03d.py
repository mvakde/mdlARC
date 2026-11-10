"""Cable-addressed broadcast switches. Standard library and NumPy only.

0 is empty, 8 is neutral cable, 9 marks a cable cell needing repair.
Colors 1..7 are assigned to bar patterns and terminal requests per input.
The decoder uses only pixels. Construction renders its own planned geometry.
"""
import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _add(p, d, n=1):
    return p[0] + n*d[0], p[1] + n*d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _edge(p, shape):
    return p[0] in (0, shape[0]-1) or p[1] in (0, shape[1]-1)


def _neighbors(p, cells):
    return [q for d in DIRS if (q := _add(p, d)) in cells]


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        component, stack = {p}, [p]
        while stack:
            for q in _neighbors(stack.pop(), remaining):
                remaining.remove(q)
                component.add(q)
                stack.append(q)
        result.append(component)
    return result


def _ray(p, d, shape):
    result = []
    p = _add(p, d)
    while _inside(p, shape):
        result.append(p)
        p = _add(p, d)
    return result


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    d = int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1]))
    n = abs(b[0]-a[0]) + abs(b[1]-a[1])
    return [_add(a, d, i) for i in range(n+1)]


def _polyline(points):
    result = [points[0]]
    for a, b in zip(points, points[1:]):
        result.extend(_line(a, b)[1:])
    return result


def _turns(path):
    ds = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(ds, ds[1:]))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _port(bar, side, slot, length):
    a = tuple(map(int, np.argwhere(bar['labels'] == side*length+slot)[0]))
    b = tuple(map(int, np.argwhere(bar['labels'] == (1-side)*length+slot)[0]))
    p = bar['r']+a[0], bar['c']+a[1]
    d = a[0]-b[0], a[1]-b[1]
    return p, d


def _decode(value, *, ablation=None):
    """Recover bars, topological addresses and routes from input pixels alone."""
    inp = _grid(value)
    assert np.count_nonzero(inp == 0) > inp.size/2
    colored = set(map(tuple, np.argwhere((inp >= 1) & (inp <= 7))))
    bars, terminals = [], {}
    for comp in _components(colored):
        if len(comp) == 1:
            p = next(iter(comp))
            assert _edge(p, inp.shape), 'a request must be a boundary terminal'
            terminals[p] = int(inp[p])
            continue
        rs, cs = zip(*comp)
        r, c = min(rs), min(cs)
        a = inp[r:max(rs)+1, c:max(cs)+1]
        assert a.size == len(comp) and min(a.shape) == 2
        assert 4 <= max(a.shape) <= 12
        bars.append({'r': r, 'c': c, 'arr': a.copy(), 'cells': comp})
    assert 2 <= len(bars) <= 4 and terminals
    canonical = bars[0]['arr']
    if canonical.shape[0] != 2:
        canonical = np.rot90(canonical)
    length = canonical.shape[1]
    forms = [_orient(canonical, t) for t in range(8)]
    assert len({_key(a) for a in forms}) == 8, 'bar correspondence is ambiguous'
    structural = set(map(int, canonical.flat))
    assert 2 <= len(structural) <= 3 and structural.isdisjoint(terminals.values())
    labels = np.arange(2*length).reshape(2, length)
    body_owner = {}
    for i, bar in enumerate(bars):
        matches = [t for t, a in enumerate(forms) if np.array_equal(a, bar['arr'])]
        assert len(matches) == 1
        bar['transform'] = matches[0]
        bar['labels'] = _orient(labels, matches[0])
        body_owner.update((p, i) for p in bar['cells'])

    # A single missing cell between opposed cable/waypoint cells is a gap.
    # All proposals are made from the unchanged observed mask.
    wire = set(map(tuple, np.argwhere((inp == 8) | (inp == 9)))) | set(terminals)
    gaps = set()
    for p in map(tuple, np.argwhere(inp == 0)):
        if any(_add(p, d) in wire and _add(p, (-d[0], -d[1])) in wire
               for d in ((1, 0), (0, 1))):
            gaps.add(p)
    wire |= gaps
    routes, sides, slots_by_bar, code = [], set(), [set() for _ in bars], {}
    for comp in _components(wire):
        contacts = [(p, q, body_owner[q]) for p in comp for d in DIRS
                    if (q := _add(p, d)) in body_owner]
        assert len(contacts) == 1, 'each cable needs exactly one port'
        root, body_cell, owner = contacts[0]
        degrees = {p: len(_neighbors(p, comp)) for p in comp}
        assert len(comp) >= 3 and all(n in (1, 2) for n in degrees.values())
        ends = [p for p in comp if degrees[p] == 1]
        assert len(ends) == 2 and root in ends
        endpoint = next(p for p in ends if p != root)
        assert [p for p in comp if _edge(p, inp.shape)] == [endpoint]
        bar = bars[owner]
        label = int(bar['labels'][body_cell[0]-bar['r'], body_cell[1]-bar['c']])
        side, slot = divmod(label, length)
        p, d = _port(bar, side, slot, length)
        assert p == body_cell and _add(p, d) == root, 'attach to a long face'
        assert slot not in slots_by_bar[owner]
        slots_by_bar[owner].add(slot)
        sides.add(side)
        path, previous, current = [root], None, root
        while current != endpoint:
            following = [q for q in _neighbors(current, comp) if q != previous]
            assert len(following) == 1
            previous, current = current, following[0]
            path.append(current)
        assert len(path) == len(comp)
        request = terminals.get(endpoint)
        assert not set(terminals).intersection(comp) - {endpoint}
        if request is not None:
            assert slot not in code or code[slot] == request, 'inconsistent requests'
            code[slot] = request
        routes.append({'owner': owner, 'slot': slot, 'side': side,
                       'cells': comp, 'path': path, 'endpoint': endpoint,
                       'request': request, 'bends': _turns([p]+path),
                       'displaced': endpoint != _ray(p, d, inp.shape)[-1]})
    assert len(sides) == 1
    side = sides.pop()
    slots = slots_by_bar[0]
    assert all(s == slots for s in slots_by_bar)
    assert 2 <= len(slots) <= 4 and 1 <= len(code) < len(slots)
    assert all(b-a >= 3 for a, b in zip(sorted(slots), sorted(slots)[1:]))
    assert set(code) <= slots
    assert all(inp[p] in (8, 9, 0) for r in routes for p in r['cells']-{r['endpoint']})

    out = np.zeros(inp.shape, dtype=np.int8)
    for bar in bars:
        for p in bar['cells']:
            out[p] = inp[p]
    kept, deleted, paint = set(), set(), {}
    allocations = [set() for _ in bars]
    for route in routes:
        if route['request'] is not None:
            allocations[route['owner']].add(route['slot'])
        selected = route['slot'] in code and ablation != 'append'
        if ablation == 'local':
            selected = selected and route['request'] is not None
        if selected:
            deleted |= route['cells']
        else:
            kept |= route['cells']
            for p in route['cells']:
                out[p] = 8
    ray_lengths = []
    for i, bar in enumerate(bars):
        for slot, color in code.items():
            if ablation == 'local' and slot not in allocations[i]:
                continue
            p, d = _port(bar, 1-side, slot, length)
            ray = _ray(p, d, inp.shape)
            assert ray and not set(ray).intersection(body_owner)
            ray_lengths.append(len(ray))
            for q in ray:
                assert q not in paint or paint[q] == color, 'different request rays conflict'
                paint[q] = color
                out[q] = color
    marked = set(map(tuple, np.argwhere(inp == 9)))
    commanded = [r for r in routes if r['slot'] in code]
    observed = [r for r in routes if r['request'] is not None]
    f = {'shape': inp.shape, 'length': length, 'bars': len(bars), 'slots': tuple(sorted(slots)),
         'code': tuple(sorted(code.items())), 'active': len(code), 'palette': tuple(sorted(structural)),
         'transforms': tuple(b['transform'] for b in bars),
         'directions': tuple(_port(b, 1-side, next(iter(code)), length)[1] for b in bars),
         'pooled': all(len(a) < len(code) for a in allocations),
         'unhinted_commands': sum(r['request'] is None for r in commanded),
         'retained_markers': len(marked & kept), 'deleted_markers': len(marked & deleted),
         'retained_gaps': len(gaps & kept), 'gaps': len(gaps),
         'bends': tuple(r['bends'] for r in routes),
         'bent_commands': sum(r['bends'] > 0 for r in commanded),
         'displaced_observations': sum(r['displaced'] for r in observed),
         'all_observations_displaced': all(r['displaced'] for r in observed),
         'observations': len(observed), 'ray_lengths': tuple(ray_lengths),
         'new_old_crossings': len(set(paint) & deleted), 'new_kept_crossings': len(set(paint) & kept),
         'asymmetric_code': code != {length-1-s: c for s, c in code.items()}}
    return out, f


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _decode(input_grid)[0]


def _template(rng, length, colors):
    for _ in range(100):
        a = rng.choice(colors, (2, length)).astype(np.int8)
        if rng.random() < .6:
            a[0] = colors[0]
            a[1] = colors[1]
            cut = int(rng.integers(1, length))
            a[1, :cut] = colors[-1]
            a[0, -1] = colors[1]
        for i, color in enumerate(colors):
            a[1, i] = color
        if len({_key(_orient(a, t)) for t in range(8)}) == 8:
            return a
    raise AssertionError('asymmetric template construction failed')


def _make_bar(template, t, r, c):
    a = _orient(template, t)
    return {'r': r, 'c': c, 'arr': a,
            'labels': _orient(np.arange(template.size).reshape(template.shape), t),
            'cells': {(r+i, c+j) for i in range(a.shape[0]) for j in range(a.shape[1])}}


def _route_candidates(rng, p, d, shape):
    straight = _ray(p, d, shape)
    options = [straight]
    perp = (-d[1], d[0])
    for _ in range(30):
        if len(straight) < 5:
            break
        distance = int(rng.integers(3, len(straight)-1))
        a = _add(p, d, distance)
        e = perp if rng.integers(2) else (-perp[0], -perp[1])
        bends = int(rng.integers(1, 4))
        points = [p, a]
        if bends >= 2:
            side_ray = _ray(a, e, shape)
            if len(side_ray) < 5:
                continue
            b = _add(a, e, int(rng.integers(3, len(side_ray)-1)))
            points.append(b)
            e = d if rng.random() < .8 else (-d[0], -d[1])
            if bends == 3:
                normal_ray = _ray(b, e, shape)
                if len(normal_ray) < 5:
                    continue
                points.append(_add(b, e, int(rng.integers(3, len(normal_ray)-1))))
                e = perp if rng.integers(2) else (-perp[0], -perp[1])
        tail = _ray(points[-1], e, shape)
        if len(tail) < 2:
            continue
        points.append(tail[-1])
        options.append(_polyline(points)[1:])
    rng.shuffle(options)
    return options


def _safe_route(path, p, d, shape, body, previous):
    s = set(path)
    if len(s) != len(path) or len(s) < 3 or path[0] != _add(p, d):
        return False
    if [q for q in path if _edge(q, shape)] != [path[-1]]:
        return False
    if any(not _inside(q, shape) or q in body for q in s):
        return False
    if any(_neighbors(q, body) != ([p] if q == path[0] else []) for q in s):
        return False
    if any(len(_neighbors(q, s)) != (1 if q in (path[0], path[-1]) else 2) for q in s):
        return False
    # Separation prevents accidental one-cell gap bridges between cables.
    forbidden = previous | {_add(q, e) for q in previous for e in DIRS}
    forbidden |= {_add(q, e) for q in list(forbidden) for e in DIRS}
    if s & forbidden:
        return False
    # A folded cable must not imply a shortcut across its own empty cells.
    for q in s:
        for e in ((1, 0), (0, 1)):
            middle = _add(q, e)
            if _add(q, e, 2) in s and middle not in s and middle not in body:
                return False
    return True


def _scene(rng, template, transforms, side, slots, code, shape):
    bars, body, paint = [], set(), {}
    length = template.shape[1]
    for t in transforms:
        h, w = _orient(template, t).shape
        for _ in range(40):
            r = int(rng.integers(2, shape[0]-h-1))
            c = int(rng.integers(2, shape[1]-w-1))
            bar = _make_bar(template, t, r, c)
            if bar['cells'] & (body | set(paint)):
                continue
            if any(_neighbors(p, body) for p in bar['cells']):
                continue
            fresh = {}
            valid = True
            for slot, color in code.items():
                p, d = _port(bar, 1-side, slot, length)
                ray = _ray(p, d, shape)
                if set(ray) & body or any(q in paint and paint[q] != color for q in ray):
                    valid = False
                    break
                fresh.update((q, color) for q in ray)
            if valid:
                bars.append(bar)
                body |= bar['cells']
                paint.update(fresh)
                break
        else:
            return None
    routes, previous = [], set()
    requests = [(i, s) for i in range(len(bars)) for s in slots]
    rng.shuffle(requests)
    for i, slot in requests:
        p, d = _port(bars[i], side, slot, length)
        for path in _route_candidates(rng, p, d, shape):
            if _safe_route(path, p, d, shape, body, previous):
                routes.append({'owner': i, 'slot': slot, 'path': path, 'cells': set(path),
                               'displaced': path[-1] != _ray(p, d, shape)[-1]})
                previous.update(path)
                break
        else:
            return None
    return bars, routes, paint


def _render_pair(rng, template, bars, routes, paint, code, witness, query):
    # Pick terminal observations. A query uses only displaced endpoints.
    observations = []
    for k, slot in enumerate(code):
        candidates = [r for r in routes if r['slot'] == slot and (not query or r['displaced'])]
        if witness == 'pool':
            candidates = [r for r in candidates if r['owner'] == k % len(bars)]
        if not candidates:
            return None
        if witness and k == 0:
            candidates = [r for r in candidates if r['displaced']]
            if not candidates:
                return None
        if witness == 'pool' and k == 1:
            candidates = [r for r in candidates if not r['displaced']]
            if not candidates:
                return None
        observations.append(candidates[int(rng.integers(len(candidates)))])
    # Consistent redundant terminal requests sometimes occur in later examples.
    if not witness and rng.random() < .25:
        candidates = [r for r in routes if r['slot'] in code and r not in observations
                      and (not query or r['displaced'])]
        if candidates:
            observations.append(candidates[int(rng.integers(len(candidates)))])
    shape = bars[0]['shape']
    inp = np.zeros(shape, dtype=np.int8)
    out = inp.copy()
    for bar in bars:
        r, c = bar['r'], bar['c']
        h, w = bar['arr'].shape
        inp[r:r+h, c:c+w] = bar['arr']
        out[r:r+h, c:c+w] = bar['arr']
    for route in routes:
        for p in route['path']:
            inp[p] = 8
            if route['slot'] not in code:
                out[p] = 8
    for route in observations:
        inp[route['path'][-1]] = code[route['slot']]
    for p, color in paint.items():
        out[p] = color
    # Include both a deleted annotation and a retained repair in every pair.
    groups = [[r for r in routes if (r['slot'] in code) == selected] for selected in (True, False)]
    for group in groups:
        route = group[int(rng.integers(len(group)))]
        candidates = route['path'][2:-2]
        bends = [p for j, p in enumerate(route['path'][2:-2], 2)
                 if route['path'][j-1][0] != route['path'][j+1][0]
                 and route['path'][j-1][1] != route['path'][j+1][1]]
        pool = bends if bends and rng.random() < .65 else candidates
        if not pool:
            return None
        inp[pool[int(rng.integers(len(pool)))]] = 9
    # Extra annotations and isolated straight gaps may occur anywhere on cables.
    candidates = [(r, j) for r in routes for j in range(2, len(r['path'])-2)]
    rng.shuffle(candidates)
    target, made = int(rng.integers(1, 5)), 0
    for route, j in candidates:
        p = route['path'][j]
        a, b = route['path'][j-1], route['path'][j+1]
        if inp[p] != 8 or inp[a] == 0 or inp[b] == 0:
            continue
        if a[0]+b[0] != 2*p[0] or a[1]+b[1] != 2*p[1]:
            continue
        inp[p] = 0
        made += 1
        if made >= target:
            break
    for route, j in candidates[:int(rng.integers(0, 5))]:
        if inp[route['path'][j]] == 8:
            inp[route['path'][j]] = 9
    if np.count_nonzero(inp == 0) <= inp.size/2:
        return None
    try:
        decoded, f = _decode(inp)
        assert np.array_equal(decoded, out)
        assert f['retained_markers'] and f['deleted_markers']
        assert f['bent_commands'] and f['displaced_observations']
        assert f['unhinted_commands']
        if witness == 'pool':
            assert f['pooled'] and f['asymmetric_code']
            assert not f['all_observations_displaced']
        if witness == 'rotate':
            assert f['new_kept_crossings'], 'demonstrate ray over retained cable'
        if query:
            assert f['all_observations_displaced']
    except AssertionError:
        return None
    return {'input': inp, 'output': out}


def _make_pair(rng, witness=None, query=False):
    for attempt in range(160):
        nslots = 3 if witness == 'pool' else int(rng.choice([3, 4], p=[.8, .2])) if query else int(rng.choice([2, 3, 4], p=[.3, .55, .15]))
        length = int(rng.integers(3*(nslots-1)+1, 13))
        # Sample nonuniform port spacings subject to cable separation.
        slots = sorted(map(int, rng.choice(length-2*(nslots-1), nslots, replace=False)))
        slots = tuple(s+2*i for i, s in enumerate(slots))
        nactive = 2 if witness == 'pool' else int(rng.integers(2 if query else 1, nslots))
        active = list(map(int, rng.choice(slots, nactive, replace=False)))
        palette = list(map(int, rng.permutation(np.arange(1, 8))))
        ns = int(rng.integers(2, 4))
        template = _template(rng, length, palette[:ns])
        code = dict(zip(active, palette[ns:ns+nactive]))
        if not witness and nactive > 1 and rng.random() < .25:
            code[active[-1]] = code[active[0]]
        nbar = 2 if witness else int(rng.choice([2, 3, 4], p=[.7, .27, .03]))
        transforms = [0, 4] if witness == 'pool' else [0, 1] if witness == 'rotate' else list(map(int, rng.integers(8, size=nbar)))
        side = int(rng.integers(2))
        # Richer configurations need more routing space; canvas is not an address.
        low = 17 if nslots <= 3 and nbar == 2 else 23
        shape = tuple(map(int, rng.integers(low, 31, size=2)))
        scene = _scene(rng, template, transforms, side, slots, code, shape)
        if scene is None:
            continue
        bars, routes, paint = scene
        for bar in bars:
            bar['shape'] = shape
        pair = _render_pair(rng, template, bars, routes, paint, code, witness, query)
        if pair is not None:
            # Whole-scene symmetry avoids a fixed compass orientation even in witnesses.
            t = int(rng.integers(8))
            return {role: _orient(a, t) for role, a in pair.items()}
    return _fallback(rng, witness, query)


def _fallback(rng, witness, query):
    """Spacious constructive witness, still the same port/connectivity rule."""
    shape = int(rng.integers(24, 31)), int(rng.integers(29, 31))
    four = witness is None and rng.random() < .55
    length = int(rng.integers(10, 13) if four else rng.integers(7, 10))
    slots = (0, 3, 6, 9) if four else (0, 3, 6)
    palette = list(map(int, rng.permutation(np.arange(1, 8))))
    template = _template(rng, length, palette[:int(rng.integers(2, 4))])
    code = {0: palette[3], slots[-1]: palette[4]}
    if four:
        code[3] = palette[5]
    r0 = int(rng.integers(13, 18))
    r1 = int(rng.integers(11, 15) if four else rng.integers(7, 12))
    bars = [_make_bar(template, 0, r0, 1 if four else 2)]
    if witness == 'rotate':
        bars.append(_make_bar(template, 3, 3, shape[1]-9))
    else:
        bars.append(_make_bar(template, 4, r1, shape[1]-length-2))
    routes = []
    for i, bar in enumerate(bars):
        bar['shape'] = shape
        for slot in slots:
            p, d = _port(bar, 0, slot, length)
            points = None
            if i == 0 and slot == 0:
                a = _add(p, d, 4)
                points = [p, a, (a[0], 0)]
            elif i == 1 and witness != 'rotate' and slot == 0:
                a = _add(p, d, 4)
                points = [p, a, (a[0], shape[1]-1)]
            elif i == 1 and witness != 'rotate' and four and slot == 3:
                a = _add(p, d, 8)
                b = _add(a, (0, 1), 3)
                points = [p, a, b, (0, b[1])]
            elif i == 1 and witness != 'rotate' and slot == slots[-1] and query:
                a = _add(p, d, 4)
                b = _add(a, (0, -1), 3)
                points = [p, a, b, (0, b[1])]
            path = _polyline(points)[1:] if points else _ray(p, d, shape)
            routes.append({'owner': i, 'slot': slot, 'path': path, 'cells': set(path),
                           'displaced': path[-1] != _ray(p, d, shape)[-1]})
    body = set().union(*(b['cells'] for b in bars))
    prior = set()
    for route in routes:
        p, d = _port(bars[route['owner']], 0, route['slot'], length)
        assert _safe_route(route['path'], p, d, shape, body, prior)
        prior |= route['cells']
    paint = {}
    for bar in bars:
        for slot, color in code.items():
            p, d = _port(bar, 1, slot, length)
            for q in _ray(p, d, shape):
                assert q not in body and (q not in paint or paint[q] == color)
                paint[q] = color
    for _ in range(20):
        pair = _render_pair(rng, template, bars, routes, paint, code, witness, query)
        if pair is not None:
            t = int(rng.integers(8))
            return {role: _orient(a, t) for role, a in pair.items()}
    raise AssertionError('constructive witness failed')


def _count(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            witness = ('pool', 'rotate')[i] if split == 'train' and i < 2 else None
            for _ in range(8):
                pair = _make_pair(rng, witness, split == 'test')
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct input search exhausted')
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            predicted, f = _decode(inp)
            assert np.array_equal(predicted, out), 'output violates cable-addressed switching'
            assert not np.array_equal(inp, out) and not np.any(out == 9)
            assert f['unhinted_commands'] and f['bent_commands']
            assert f['retained_markers'] and f['deleted_markers']
            assert f['displaced_observations']
            features[split].append(f)
    train = features['train']
    assert any(f['pooled'] and f['active'] >= 2 and f['asymmetric_code'] for f in train)
    assert any(any(t >= 4 for t in f['transforms']) and f['asymmetric_code'] for f in train)
    assert any(len({t % 2 for t in f['transforms']}) == 2 for f in train)
    assert any(f['new_kept_crossings'] for f in train)
    assert any(f['displaced_observations'] < f['observations'] for f in train)
    # Concrete ablations establish that both inherited mechanisms affect labels.
    for mode in ('local', 'append'):
        assert any(not np.array_equal(_decode(pair['input'], ablation=mode)[0], pair['output'])
                   for pair in puzzle['train']), f'{mode} shortcut not distinguished'
    assert all(f['all_observations_displaced'] for f in features['test'])


def self_test():
    puzzles, features = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        for pair in p['train']+p['test']:
            out, f = _decode(pair['input'])
            features.append(f)
            assert np.array_equal(out, pair['output'])
            assert np.array_equal(solve(_orient(pair['input'], 5)), _orient(out, 5))
            # Reassign every nonreserved color independently, preserving roles.
            perm = np.array([0, 7, 6, 5, 4, 3, 2, 1, 8, 9], dtype=np.int8)
            assert np.array_equal(solve(perm[pair['input']]), perm[out])
    assert len({tuple(_key(q['input']) for q in p['train']+p['test']) for p in puzzles}) == 42
    assert len({f['shape'] for f in features}) >= 35
    assert len({f['length'] for f in features}) >= 7
    assert len({f['slots'] for f in features}) >= 25
    assert {len(f['slots']) for f in features} == {2, 3, 4}
    assert {f['active'] for f in features} == {1, 2, 3}
    assert {f['bars'] for f in features} >= {2, 3}
    assert len({f['palette'] for f in features}) >= 20
    assert {d for f in features for d in f['directions']} == set(DIRS)
    assert {b for f in features for b in f['bends']} >= {0, 1, 2, 3}
    assert any(f['retained_gaps'] for f in features)
    assert any(f['new_old_crossings'] for f in features)
    # Negative checks target preservation, deletion, ray painting and repair.
    p = puzzles[0]
    q = p['test'][0]
    changed = list(map(tuple, np.argwhere(q['input'] != q['output'])))
    unchanged = list(map(tuple, np.argwhere(q['input'] == q['output'])))
    for site in changed[:3] + unchanged[:2]:
        bad = {s: [dict(pair) for pair in p[s]] for s in p}
        bad['test'][0]['output'] = q['output'].copy()
        bad['test'][0]['output'][site] = (int(q['output'][site])+1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupt output accepted')
    other = generate_puzzle(0)
    assert all(_key(a['input']) == _key(b['input']) and _key(a['output']) == _key(b['output'])
               for s in p for a, b in zip(p[s], other[s]))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return {'seeds': 42, 'pairs': len(features), 'shapes': len({f['shape'] for f in features}),
            'lengths': sorted({f['length'] for f in features}),
            'port_counts': sorted({len(f['slots']) for f in features}),
            'active_counts': sorted({f['active'] for f in features}),
            'bar_counts': sorted({f['bars'] for f in features}),
            'bend_counts': sorted({b for f in features for b in f['bends']}),
            'old_ray_interactions': sum(f['new_old_crossings'] > 0 for f in features),
            'retained_ray_interactions': sum(f['new_kept_crossings'] > 0 for f in features)}
