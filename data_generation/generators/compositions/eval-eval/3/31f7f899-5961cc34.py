"""Threaded sorting carrier. Standard library + NumPy; no external state."""
from itertools import combinations
import hashlib
import json

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _add(p, d, k=1):
    return p[0] + k * d[0], p[1] + k * d[1]


def _normal(d):
    return d[1], -d[0]


def _head(s):
    return {_add(s['hub'], s['d'], k) for k in (-1, 0, 1)}


def _cargo(s, span=None):
    lo, hi = s['span'] if span is None else span
    a, d, n = _add(s['hub'], s['d'], -2), s['d'], _normal(s['d'])
    return {_add(_add(a, d, -k), n, t)
            for k in range(s['width']) for t in range(lo, hi + 1)}


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, stack = {start}, [start]
        while stack:
            p = stack.pop()
            for d in DIRS:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    stack.append(q)
        result.append(component)
    return result


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError('grid must be an integer 1..30 by 1..30 array of colors 0..9')
    return g


def _decode(value):
    g = _grid(value)
    h, w = g.shape
    bg = int(g[0, 0])
    assert all(int(g[p]) == bg for p in ((0, w-1), (h-1, 0), (h-1, w-1)))
    parts = _components(map(tuple, np.argwhere(g != bg).tolist()))
    border = lambda p: p[0] in (0, h-1) or p[1] in (0, w-1)
    launchers = [c for c in parts if any(border(p) for p in c)]
    assert len(launchers) == 1
    launcher = launchers[0]
    starts = [p for p in launcher if border(p)]
    assert len(launcher) == 2 and len(starts) == 1
    start = starts[0]
    direction = (1, 0) if start[0] == 0 else (-1, 0) if start[0] == h-1 else (0, 1) if start[1] == 0 else (0, -1)
    cap = _add(start, direction)
    assert launcher == {start, cap}
    signal, port = int(g[start]), int(g[cap])
    assert len({bg, signal, port}) == 3
    stations, stem_colors, templates = [], set(), {}
    for component in parts:
        if component == launcher:
            continue
        ports = [p for p in component if int(g[p]) == port]
        assert len(ports) == 1
        p = ports[0]
        neighbors = [q for d in DIRS if (q := _add(p, d)) in component]
        assert len(neighbors) == 1
        hub = neighbors[0]
        d = p[0]-hub[0], p[1]-hub[1]
        stem = int(g[hub])
        assert stem not in (bg, signal, port)
        assert _add(hub, d, -1) in component and int(g[_add(hub, d, -1)]) == stem
        s = {'hub': hub, 'd': d}
        payload = component - _head(s)
        assert payload
        colors = {int(g[q]) for q in payload}
        assert len(colors) == 1
        color = next(iter(colors))
        assert color not in (bg, signal, port, stem)
        anchor, n = _add(hub, d, -2), _normal(d)
        local = [((anchor[0]-r)*d[0] + (anchor[1]-c)*d[1],
                  (r-anchor[0])*n[0] + (c-anchor[1])*n[1]) for r, c in payload]
        width = max(k for k, t in local) + 1
        lo, hi = min(t for k, t in local), max(t for k, t in local)
        assert width in (1, 2) and -3 <= lo <= 0 <= hi <= 3
        s.update(width=width, color=color, span=(lo, hi))
        assert _cargo(s) == payload
        assert all(0 < r < h-1 and 0 < c < w-1 for r, c in component)
        length = hi-lo+1
        assert templates.setdefault(length, (lo, hi)) == (lo, hi), 'ambiguous interval tie'
        stations.append(s)
        stem_colors.add(stem)
    assert stations and len(stem_colors) == 1
    return {'shape': g.shape, 'bg': bg, 'signal': signal, 'port': port,
            'stem': next(iter(stem_colors)), 'start': start, 'cap': cap,
            'initial': direction, 'stations': stations}


def _walk(scene):
    """Pixel traversal of original control heads, independent of construction."""
    stations, shape = scene['stations'], scene['shape']
    owner = {}
    for i, s in enumerate(stations):
        for p in _head(s):
            assert p not in owner
            owner[p] = i
    trace = {scene['start'], scene['cap']}
    route, origin, d = [], scene['cap'], scene['initial']
    for _ in range(len(stations) + 1):
        p = _add(origin, d)
        while 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]:
            trace.add(p)
            if p in owner:
                index = owner[p]
                assert index not in route, 'cycle or reactivation'
                route.append(index)
                s = stations[index]
                trace.update(_head(s))
                origin, d = _add(s['hub'], s['d']), s['d']
                break
            p = _add(p, d)
        else:
            return route, trace
    raise AssertionError('route does not terminate')


def _assigned(stations, route, mode='length'):
    indices = route
    if mode == 'all':
        indices = list(range(len(stations)))
    if mode == 'keep':
        return [stations[i]['span'] for i in route]
    if mode == 'reverse':
        return [stations[i]['span'] for i in route[::-1]]
    key = lambda i: stations[i]['span'][1] - stations[i]['span'][0] + 1
    if mode == 'area':
        key = lambda i: (stations[i]['span'][1]-stations[i]['span'][0]+1)*stations[i]['width']
    spans = [stations[i]['span'] for i in sorted(indices, key=key)][:len(route)]
    if mode == 'center':
        spans = [(-(int(b-a+1)//2), int(b-a)//2) for a, b in spans]
    if mode == 'mirror':
        spans = [(-b, -a) for a, b in spans]
    if mode in ('raster', 'column'):
        order = sorted(route, key=lambda i: stations[i]['hub'] if mode == 'raster' else stations[i]['hub'][::-1])
        by_index = dict(zip(order, spans))
        spans = [by_index[i] for i in route]
    return spans


def _render(scene, route, trace, spans):
    out = np.full(scene['shape'], scene['bg'], dtype=np.int8)
    for p in trace:
        out[p] = scene['signal']
    for index, span in zip(route, spans):
        s = scene['stations'][index]
        for p in _cargo(s, span):
            out[p] = s['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover all conventions from this input; use no seed or stored puzzle."""
    scene = _decode(input_grid)
    route, trace = _walk(scene)
    return _render(scene, route, trace, _assigned(scene['stations'], route))


def _graph(points):
    result = []
    for r, c in points:
        edges = {}
        for d in DIRS:
            candidates = [((rr-r)*d[0]+(cc-c)*d[1], j)
                          for j, (rr, cc) in enumerate(points)
                          if (rr-r)*d[1] == (cc-c)*d[0]
                          and (rr-r)*d[0]+(cc-c)*d[1] > 0]
            if candidates:
                edges[d] = min(candidates)[1]
        result.append(edges)
    return result


def _find_route(points, count, rng, witness):
    graph = _graph(points)
    starts = [(i, d) for i in range(len(points)) for d in DIRS
              if (-d[0], -d[1]) not in graph[i]]
    rng.shuffle(starts)

    def visit(path, initial, directions):
        i = path[-1]
        if len(path) == count:
            exits = [d for d in DIRS if d not in graph[i]]
            rng.shuffle(exits)
            for final in exits:
                ds = directions + [final]
                if witness:
                    if len({d[0] == 0 for d in ds}) < 2:
                        continue
                    if path in (sorted(path, key=lambda j: points[j]),
                                sorted(path, key=lambda j: points[j][::-1])):
                        continue
                return path, initial, ds
            return None
        edges = list(graph[i].items())
        rng.shuffle(edges)
        for d, j in edges:
            if j not in path:
                found = visit(path + [j], initial, directions + [d])
                if found is not None:
                    return found
        return None

    for i, initial in starts:
        found = visit([i], initial, [])
        if found is not None:
            return found
    return None


def _layout(rng, active, decoys, witness):
    total = active + decoys
    shapes = [(r, c) for r in (2, 3) for c in (2, 3) if r*c >= total]
    # Random attempts are bounded. Exhaustive subset fallback preserves counts.
    for attempt in range(49):
        nr, nc = shapes[int(rng.integers(len(shapes)))]
        available = [(r, c) for r in range(nr) for c in range(nc)]
        if attempt < 48:
            subset = sorted(int(x) for x in rng.choice(len(available), total, replace=False))
            points = [available[i] for i in subset]
            found = _find_route(points, active, rng, witness)
        else:
            nr = nc = 3
            available = [(r, c) for r in range(3) for c in range(3)]
            found = None
            for subset in combinations(range(9), total):
                points = [available[i] for i in subset]
                found = _find_route(points, active, rng, witness)
                if found is not None:
                    break
        if found is not None:
            break
    assert found is not None, 'finite route fallback failed'
    route, initial, directions = found

    def coordinates(count, along_launch):
        margin = 6 if along_launch else 4
        room = 30 - 2*margin - 1
        gaps = [8] * (count-1)
        extra = int(rng.integers(room-sum(gaps)+1))
        for _ in range(extra):
            gaps[int(rng.integers(len(gaps)))] += 1
        minimum = 2*margin + sum(gaps) + 1
        size = int(rng.integers(minimum, 31))
        first = margin + int(rng.integers(size-minimum+1))
        coords = [first]
        for gap in gaps:
            coords.append(coords[-1] + gap)
        return size, coords

    height, rows = coordinates(nr, initial[0] != 0)
    width, cols = coordinates(nc, initial[1] != 0)
    hubs = [(rows[r], cols[c]) for r, c in points]
    first = hubs[route[0]]
    if initial[0]:
        start = (0 if initial[0] == 1 else height-1, first[1])
    else:
        start = (first[0], 0 if initial[1] == 1 else width-1)
    outdirs = [DIRS[int(rng.integers(4))] for _ in points]
    for i, d in zip(route, directions):
        outdirs[i] = d
    return (height, width), hubs, outdirs, route, start, initial


def _sample_data(rng, count, witness, directions):
    for _ in range(160):
        if witness:
            evens = [int(x) for x in rng.choice([2, 4, 6], 2, replace=False)]
            rest = [x for x in range(1, 8) if x not in evens]
            lengths = evens + [int(x) for x in rng.choice(rest, count-2, replace=False)]
        else:
            if rng.random() < .55:
                lengths = [int(x) for x in rng.choice(np.arange(1, 8), count, replace=False)]
            else:
                distinct = int(rng.integers(1, count))
                selected = [int(x) for x in rng.choice(np.arange(1, 8), distinct, replace=False)]
                lengths = selected + [int(x) for x in rng.choice(selected, count-distinct)]
                rng.shuffle(lengths)
            mode = int(rng.choice(4, p=[.65, .10, .15, .10]))
            if mode:
                lengths.sort(reverse=mode == 2)
                if mode == 3:
                    j = int(rng.integers(count-1))
                    lengths[j], lengths[j+1] = lengths[j+1], lengths[j]
        templates = {}
        for length in sorted(set(lengths)):
            lower = max(-3, 1-length)
            upper = min(0, 4-length)
            lo = int(rng.integers(lower, upper+1))
            templates[length] = (lo, lo+length-1)
        if witness:
            negative, positive = evens
            a = [lo for lo in range(-3, 1) if 0 <= lo+negative-1 <= 3 and 2*lo+negative-1 < 0]
            b = [lo for lo in range(-3, 1) if 0 <= lo+positive-1 <= 3 and 2*lo+positive-1 > 0]
            lo1, lo2 = int(rng.choice(a)), int(rng.choice(b))
            templates[negative] = (lo1, lo1+negative-1)
            templates[positive] = (lo2, lo2+positive-1)
            ordered = sorted(lengths)
            rng.shuffle(lengths)
            if lengths == ordered or lengths[::-1] == ordered:
                continue
        spans = [templates[x] for x in lengths]
        widths = [int(x) for x in rng.integers(1, 3, size=count)]
        if witness:
            possible = [(a, b) for a in lengths for b in lengths if a < b < 2*a]
            if not possible:
                continue
            short, long = possible[int(rng.integers(len(possible)))]
            widths[lengths.index(short)], widths[lengths.index(long)] = 2, 1
            assigned = sorted(spans, key=lambda p: p[1]-p[0])
            # Both biased even intervals must move; at least one biased interval
            # must be transported to a genuinely different local orientation.
            if any(spans.index(templates[x]) == assigned.index(templates[x]) for x in evens):
                continue
            if not any(sum(span) != 0 and directions[j] != directions[spans.index(span)]
                       for j, span in enumerate(assigned)):
                continue
        return spans, widths, templates
    raise AssertionError('bounded cargo witness search failed')


def _analytic_trace(scene, route):
    """Constructor follows its abstract itinerary using projected distances."""
    trace = {scene['start'], scene['cap']}
    origin, d = scene['cap'], scene['initial']
    for index in route:
        s = scene['stations'][index]
        hits = [(p[0]-origin[0])*d[0]+(p[1]-origin[1])*d[1]
                for p in _head(s)
                if (p[0]-origin[0])*d[1] == (p[1]-origin[1])*d[0]
                and (p[0]-origin[0])*d[0]+(p[1]-origin[1])*d[1] > 0]
        assert hits
        for k in range(1, min(hits)):
            trace.add(_add(origin, d, k))
        trace.update(_head(s))
        origin, d = _add(s['hub'], s['d']), s['d']
    limit = origin[0]+1 if d == (-1, 0) else scene['shape'][0]-origin[0] if d == (1, 0) else origin[1]+1 if d == (0, -1) else scene['shape'][1]-origin[1]
    trace.update(_add(origin, d, k) for k in range(1, limit))
    return trace


def _make_pair(rng, active, decoys, witness, roles, accent_pool):
    shape, hubs, directions, route, start, initial = _layout(rng, active, decoys, witness)
    spans, widths, templates = _sample_data(rng, active, witness, [directions[i] for i in route])
    stations = [{'hub': p, 'd': d, 'color': int(rng.choice(accent_pool))}
                for p, d in zip(hubs, directions)]
    if witness and len({stations[i]['color'] for i in route}) < 2:
        stations[route[-1]]['color'] = int(rng.choice([c for c in accent_pool if c != stations[route[0]]['color']]))
    for index, span, width in zip(route, spans, widths):
        stations[index].update(span=span, width=width)
    for index, s in enumerate(stations):
        if index not in route:
            length = 1 if witness else int(rng.integers(1, 8))
            if length not in templates:
                lo = int(rng.integers(max(-3, 1-length), min(0, 4-length)+1))
                templates[length] = (lo, lo+length-1)
            s.update(span=templates[length], width=int(rng.integers(1, 3)))
    bg, signal, port, stem = roles
    scene = {'shape': shape, 'bg': bg, 'signal': signal, 'port': port, 'stem': stem,
             'start': start, 'cap': _add(start, initial), 'initial': initial, 'stations': stations}
    inp = np.full(shape, bg, dtype=np.int8)
    for s in stations:
        for p in _cargo(s):
            inp[p] = s['color']
        inp[s['hub']] = inp[_add(s['hub'], s['d'], -1)] = stem
        inp[_add(s['hub'], s['d'])] = port
    inp[start], inp[scene['cap']] = signal, port
    trace = _analytic_trace(scene, route)
    ordered = [spans[int(i)] for i in np.argsort([b-a+1 for a, b in spans], kind='stable')]
    return {'input': inp, 'output': _render(scene, route, trace, ordered)}


def _key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def _features(pair, check_output=True):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    scene = _decode(inp)
    route, trace = _walk(scene)
    stations = scene['stations']
    assigned = _assigned(stations, route)
    assert inp.shape == out.shape
    if check_output:
        # Relational invariants are checked independently of numeric argsort.
        remaining = [stations[i]['span'] for i in route]
        previous, colored = 0, set()
        for i, span in zip(route, assigned):
            assert span in remaining and span[1]-span[0]+1 >= previous
            remaining.remove(span)
            previous = span[1]-span[0]+1
            cells = _cargo(stations[i], span)
            assert not colored & cells
            assert all(0 < r < out.shape[0]-1 and 0 < c < out.shape[1]-1 for r, c in cells)
            assert not any(cells & _head(s) for s in stations)
            assert not any(cells & _cargo(s) for j, s in enumerate(stations) if j != i)
            assert all(int(out[p]) == stations[i]['color'] for p in cells)
            colored.update(cells)
        assert not remaining
        for p in np.ndindex(out.shape):
            if p not in colored:
                assert int(out[p]) == (scene['signal'] if p in trace else scene['bg'])
        assert np.array_equal(solve(inp.copy()), out)
    current = [stations[i]['span'] for i in route]
    directions = [stations[i]['d'] for i in route]
    moved_bias = {int(np.sign(sum(span))) for span, target in zip(current, assigned)
                  if span != target and (span[1]-span[0]+1) % 2 == 0 and sum(span) != 0}
    rotated = any(sum(span) != 0 and any(stations[i]['span'] == span and stations[i]['d'] != stations[j]['d']
                  for i in route) for j, span in zip(route, assigned))
    defeats = {mode: _assigned(stations, route, mode) != assigned
               for mode in ('keep', 'all', 'reverse', 'area', 'raster', 'column', 'center', 'mirror')}
    source_order = sorted(route, key=lambda i: stations[i]['span'][1]-stations[i]['span'][0])
    fixed_slot_witness = any((stations[i]['color'], stations[i]['width']) !=
                             (stations[j]['color'], stations[j]['width'])
                             for i, j in zip(route, source_order))
    behind = False
    origin, d = scene['cap'], scene['initial']
    for i in route:
        target = stations[i]['hub']
        near = (target[0]-origin[0])*d[0]+(target[1]-origin[1])*d[1]
        behind |= any((s['hub'][0]-origin[0])*d[1] == (s['hub'][1]-origin[1])*d[0]
                      and (s['hub'][0]-origin[0])*d[0]+(s['hub'][1]-origin[1])*d[1] > near
                      for j, s in enumerate(stations) if j != i)
        origin, d = _add(target, stations[i]['d']), stations[i]['d']
    return {'shape': inp.shape, 'active': len(route), 'decoys': len(stations)-len(route),
            'roles': tuple(scene[k] for k in ('bg', 'signal', 'port', 'stem')),
            'initial': scene['initial'], 'directions': tuple(directions),
            'widths': tuple(stations[i]['width'] for i in route),
            'lengths': tuple(b-a+1 for a, b in current), 'spans': tuple(current),
            'ties': len(set(current)) < len(current), 'moved_bias': moved_bias,
            'rotated': rotated, 'defeats': defeats, 'behind': behind,
            'fixed_slot_witness': fixed_slot_witness,
            'signature': hashlib.sha256(inp.astype(np.int8).tobytes()).hexdigest()}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, facts = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            key = _key(pair['input'])
            assert key not in seen
            seen.add(key)
            facts[split].append(_features(pair))
    train = facts['train']
    assert len({f['roles'] for rows in facts.values() for f in rows}) == 1
    assert {3, 4} <= {f['active'] for f in train}
    for f in train:
        assert 2 <= f['active'] <= 4 and 0 <= f['decoys'] <= 3
    for f in facts['test']:
        assert 5 <= f['active'] <= 6 and 0 <= f['decoys'] <= 3
    witnesses = [f for f in train if f['active'] in (3, 4) and f['decoys']
                 and f['moved_bias'] == {-1, 1} and f['rotated']
                 and f['fixed_slot_witness'] and all(f['defeats'].values())]
    assert {3, 4} <= {f['active'] for f in witnesses}, 'missing compositional demonstrations'


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
    palette = [int(x) for x in rng.permutation(10)]
    roles, pool = palette[:4], palette[4:]
    reuse = bool(rng.integers(2))
    shared = pool[:int(rng.integers(2, 7))]
    counts = [3, 4]
    rng.shuffle(counts)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            witness = split == 'train' and index < 2
            active = counts[index] if witness else int(rng.integers(2, 5) if split == 'train' else rng.integers(5, 7))
            decoys = int(rng.integers(1 if witness else 0, min(3, 9-active)+1))
            accent_pool = shared if reuse else list(rng.choice(pool, int(rng.integers(2, 7)), replace=False))
            for _ in range(32):
                pair = _make_pair(rng, active, decoys, witness, roles, accent_pool)
                key = _key(pair['input'])
                if key not in seen:
                    facts = _features(pair)
                    if not witness or (facts['moved_bias'] == {-1, 1} and facts['rotated']
                                       and facts['fixed_slot_witness'] and all(facts['defeats'].values())):
                        break
            else:
                raise AssertionError('bounded distinct demonstration search failed')
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_hash(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def _solver_regressions():
    # Explicit non-lattice example, independent of the random constructor.
    # Two reached heads: (8,5) east, (8,13) north. The decoy's cargo (8,9)
    # crosses the ray but its head at rows 4..6 does not intercept it.
    inp_rows = '''00000000000000000000
00000000000000000000
00000000000000000000
00000000000000000000
00000000020000000000
00000000030000000000
00000000030000000000
00040000060002000000
12043320060003000000
00040000000003000000
00040000000555000000
00000000000555000000
00000000000000000000
00000000000000000000'''
    out_rows = '''00000000000001000000
00000000000001000000
00000000000001000000
00000000000001000000
00000000000001000000
00000000000001000000
00040000000001000000
00040000000001000000
11141111111111000000
00000000000001000000
00000000000055550000
00000000000055550000
00000000000000000000
00000000000000000000'''
    parse = lambda rows: np.array([[int(c) for c in row] for row in rows.split()], dtype=np.int8)
    inp, out = parse(inp_rows), parse(out_rows)
    for turn in range(4):
        assert np.array_equal(solve(np.rot90(inp, turn).copy()), np.rot90(out, turn))
    miss = inp.copy()
    miss[8, :2], miss[1, :2] = 0, [1, 2]
    expected = np.zeros_like(miss)
    expected[1, :] = 1
    assert np.array_equal(solve(miss), expected)


def self_test():
    """Bounded sequential 42-seed sweep; no files, traces, or grids retained."""
    hashes, shapes, signatures, roles, directions = set(), set(), set(), set(), set()
    active_counts, widths, lengths, spans, decoy_counts = set(), set(), set(), set(), set()
    tied, first_hit, rotated, pairs = 0, 0, 0, 0
    seed_zero = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprint = _puzzle_hash(puzzle)
        hashes.add(fingerprint)
        if seed == 0:
            seed_zero = fingerprint
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair)
                shapes.add(f['shape'])
                signatures.add(f['signature'])
                roles.add(f['roles'])
                directions.update(f['directions'])
                active_counts.add(f['active'])
                decoy_counts.add(f['decoys'])
                widths.update(f['widths'])
                lengths.update(f['lengths'])
                spans.update(f['spans'])
                tied += f['ties']
                first_hit += f['behind']
                rotated += f['rotated']
                pairs += 1
    assert len(hashes) == 42 and seed_zero == _puzzle_hash(generate_puzzle(0))
    assert len(shapes) >= 30 and len(signatures) == pairs and len(roles) >= 35
    assert directions == set(DIRS) and widths == {1, 2} and lengths == set(range(1, 8))
    assert active_counts == set(range(2, 7)) and decoy_counts == set(range(4))
    assert len(spans) >= 14 and tied >= 20 and first_hit >= 20 and rotated >= 84
    _solver_regressions()
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(81000+10*nt+nq, nt, nq)
            assert len(puzzle['train']) == nt and len(puzzle['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(0, 2, 1)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return {'seeds': 42, 'pairs': pairs, 'puzzles': len(hashes), 'shapes': len(shapes),
            'input_signatures': len(signatures), 'role_palettes': len(roles),
            'active_counts': sorted(active_counts), 'decoy_counts': sorted(decoy_counts),
            'widths': sorted(widths), 'lengths': sorted(lengths), 'span_templates': len(spans),
            'tied_pairs': tied, 'nearer_hit_pairs': first_hit, 'rotated_transfer_pairs': rotated,
            'explicit_regression_rotations': 4, 'explicit_miss': True}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
