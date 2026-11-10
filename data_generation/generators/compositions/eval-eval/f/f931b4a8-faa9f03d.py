"""Walking ribbons: source-distance patterns over a replicated carrier.

No files, parent generators, seeds, or training labels are read by solve.
Generation retains at most one bounded scene/candidate pool; self_test streams.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
INKS = (1, 2, 3, 4)
REQUIRED = frozenset(("tile_carrier", "ignore_ribbon", "opaque_holes", "cartesian_phase",
                      "uncounted_gaps", "uncounted_markers", "ignore_redirect",
                      "keep_tail", "straight_delete", "reset_ray", "reset_turns",
                      "reverse_phase"))


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _neighbors(p, cells):
    return [q for d in DIRS if (q := _add(p, d)) in cells]


def _boundary(p, shape):
    return p[0] in (0, shape[0]-1) or p[1] in (0, shape[1]-1)


def _ray(p, d, shape):
    cells = []
    p = _add(p, d)
    while _inside(p, shape):
        cells.append(p)
        p = _add(p, d)
    return cells


def _simple(walk):
    cells = set(walk)
    if len(cells) != len(walk) or len(walk) < 2:
        return False
    return all(len(_neighbors(p, cells)) == (1 if i in (0, len(walk)-1) else 2)
               and (i == 0 or sum(abs(p[j]-walk[i-1][j]) for j in (0, 1)) == 1)
               for i, p in enumerate(walk))


def _separated(first, second):
    a, b = set(first), set(second)
    return not a & b and not any(_neighbors(p, b) for p in a)


def _bends(walk):
    ds = [(q[0]-p[0], q[1]-p[1]) for p, q in zip(walk, walk[1:])]
    return sum(a != b for a, b in zip(ds, ds[1:]))


def _read_legend(inp):
    separators = np.flatnonzero(np.all(inp == 5, axis=1))
    assert len(separators) == 1, 'one full-width legend separator required'
    sep = int(separators[0])
    a = sep-2
    assert 2 <= a <= 4
    top = inp[0]
    stops = np.flatnonzero(top != 5)
    assert len(stops)
    b = int(stops[0])-2
    assert 2 <= b <= 4
    left = b+3
    stop = left
    while stop < inp.shape[1] and top[stop] == 5:
        stop += 1
    length = stop-left-2
    assert 2 <= length <= 6
    carrier = inp[1:a+1, 1:b+1].copy()
    ribbon = inp[1, left+1:left+1+length].copy()
    assert np.all(np.isin(carrier, INKS)) and np.all(np.isin(ribbon, (0, *INKS)))
    assert np.any(ribbon == 0) and np.any(ribbon != 0)
    assert all(not (length % k == 0 and np.array_equal(ribbon, np.tile(ribbon[:k], length//k)))
               for k in range(1, length)), 'ribbon period must be primitive'
    expected = np.zeros((sep, inp.shape[1]), dtype=np.int8)
    expected[:a+2, :b+2] = 5
    expected[1:a+1, 1:b+1] = carrier
    expected[:3, left:left+length+2] = 5
    expected[1, left+1:left+length+1] = ribbon
    assert np.array_equal(inp[:sep], expected), 'malformed sample frames'
    board = inp[sep+1:].copy()
    assert board.size and np.all(np.isin(board, (0, 6, 7, 8, 9)))
    assert board.shape[0] % a == 0 and board.shape[1] % b == 0
    return carrier, ribbon, board


def _analyze(input_grid):
    inp = _grid(input_grid)
    carrier, ribbon, board = _read_legend(inp)
    occupied = {tuple(map(int, p)) for p in np.argwhere(np.isin(board, (6, 8, 9)))}
    gaps = set()
    for p0 in np.argwhere(board == 0):
        p = tuple(map(int, p0))
        ns = _neighbors(p, occupied)
        if (len(ns) == 2 and ns[0][0]+ns[1][0] == 2*p[0]
                and ns[0][1]+ns[1][1] == 2*p[1]):
            gaps.add(p)
    cells = occupied | gaps
    remaining = set(cells)
    routes = []
    while remaining:
        todo = [min(remaining)]
        component = set(todo)
        remaining.difference_update(component)
        while todo:
            p = todo.pop()
            for q in _neighbors(p, remaining):
                remaining.remove(q)
                component.add(q)
                todo.append(q)
        roots = [p for p in component if board[p] == 9]
        assert len(roots) == 1, 'each route needs exactly one visible source'
        root = roots[0]
        assert len(_neighbors(root, component)) == 1
        walk, previous, current = [], None, root
        while True:
            walk.append(current)
            following = [p for p in _neighbors(current, component) if p != previous]
            if not following:
                break
            assert len(following) == 1 and following[0] not in walk, 'not a simple path'
            previous, current = current, following[0]
        assert len(walk) == len(component) and _simple(walk)
        assert _boundary(walk[-1], board.shape), 'old endpoint must reach boundary'
        routes.append({'walk': walk, 'final': list(walk), 'pivot': None, 'old': [], 'ray': []})
    assert routes
    routes.sort(key=lambda r: r['walk'][0])
    markers = [tuple(map(int, p)) for p in np.argwhere(board == 7)]
    for q in markers:
        ns = _neighbors(q, cells)
        assert len(ns) == 1, 'direction marker must have one adjacent pivot'
        pivot = ns[0]
        route = next(r for r in routes if pivot in r['walk'])
        assert route['pivot'] is None, 'at most one redirect per route'
        walk = route['walk']
        index = walk.index(pivot)
        d = (q[0]-pivot[0], q[1]-pivot[1])
        assert 0 < index < len(walk)-1
        assert walk[index+1] == (pivot[0]-d[0], pivot[1]-d[1])
        before = (walk[index-1][0]-pivot[0], walk[index-1][1]-pivot[1])
        assert before[0]*d[0]+before[1]*d[1] == 0
        ray = _ray(pivot, d, board.shape)
        final = walk[:index+1]+ray
        assert _simple(final), 'redirect touches its own retained route'
        route.update(final=final, pivot=index, old=walk[index+1:], ray=ray, direction=d)
    for i, route in enumerate(routes):
        assert all(_separated(route['final'], r['final']) for r in routes[i+1:]), 'final routes must not touch'
    return {'carrier': carrier, 'ribbon': ribbon, 'board': board, 'gaps': gaps, 'routes': routes}


def _base(data, tiled=False):
    carrier, board = data['carrier'], data['board']
    h, w = board.shape
    a, b = carrier.shape
    if tiled:
        return carrier[np.arange(h)[:, None] % a, np.arange(w)[None, :] % b].copy()
    return np.repeat(np.repeat(carrier, h//a, axis=0), w//b, axis=1)


def _paint(data, variant=None):
    out = _base(data, variant == 'tile_carrier')
    ribbon, board = data['ribbon'], data['board']
    if variant == 'ignore_ribbon':
        return out
    for route in data['routes']:
        paths = [route['final']]
        if variant == 'ignore_redirect':
            paths = [route['walk']]
        elif variant in ('keep_tail', 'straight_delete'):
            old = route['walk']
            if variant == 'keep_tail':
                paths = [old, route['final']]
            elif route['pivot'] is not None:
                i = route['pivot']
                d = (old[i+1][0]-old[i][0], old[i+1][1]-old[i][1])
                j = i+1
                while j+1 < len(old) and (old[j+1][0]-old[j][0], old[j+1][1]-old[j][1]) == d:
                    j += 1
                # A faulty straight-only deletion leaves the rest of the old suffix.
                # Paint this faulty remnant separately with its original indices.
                for k in range(j+1, len(old)):
                    ink = int(ribbon[k % len(ribbon)])
                    if ink:
                        out[old[k]] = ink
                paths = [route['final']]
        for path in paths:
            phase = 0
            last_turn = 0
            for k, p in enumerate(path):
                index = k
                if variant == 'cartesian_phase':
                    index = p[0]+p[1]
                elif variant == 'reverse_phase':
                    index = len(path)-1-k
                elif variant == 'reset_turns':
                    if 0 < k < len(path)-1:
                        before = (p[0]-path[k-1][0], p[1]-path[k-1][1])
                        after = (path[k+1][0]-p[0], path[k+1][1]-p[1])
                        if before != after:
                            last_turn = k
                    index = k-last_turn
                elif variant in ('uncounted_gaps', 'uncounted_markers'):
                    index = phase
                    skip = p in data['gaps'] if variant == 'uncounted_gaps' else board[p] == 6
                    phase += int(not skip)
                elif variant == 'reset_ray' and route['pivot'] is not None and k > route['pivot']:
                    index = k-route['pivot']-1
                ink = int(ribbon[index % len(ribbon)])
                if ink or variant == 'opaque_holes':
                    out[p] = ink
    return out.astype(np.int8)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the material, geometry, direction and phase solely from this input."""
    return _paint(_analyze(input_grid))


def _evidence(data, out):
    found = set()
    for variant in REQUIRED:
        alternative = _paint(data, variant)
        changed = out != alternative
        if variant in ('uncounted_gaps', 'uncounted_markers'):
            # A phase error must be visible away from the damaged position itself.
            allowed = np.ones(out.shape, dtype=bool)
            for p in data['gaps']:
                allowed[p] = False
            allowed[data['board'] == 6] = False
            changed &= allowed
        if np.any(changed):
            found.add(variant)
    return found


def _line(a, b):
    dr, dc = int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1]))
    return [(a[0]+i*dr, a[1]+i*dc) for i in range(max(abs(a[0]-b[0]), abs(a[1]-b[1]))+1)]


def _random_walk(rng, shape, rich=False):
    side = int(rng.integers(4))
    h, w = shape
    if side in (0, 2):
        start = (0 if side == 0 else h-1, int(rng.integers(1, w-1)))
        axis = 0
    else:
        start = (int(rng.integers(1, h-1)), w-1 if side == 1 else 0)
        axis = 1
    walk, vertices = [start], [start]
    desired = int(rng.integers(4 if rich else 2, 7 if rich else 5))
    for _ in range(desired):
        p = walk[-1]
        choices = [n for n in range(1, shape[axis]-1) if abs(n-p[axis]) >= 3]
        rng.shuffle(choices)
        for n in choices:
            q = (n, p[1]) if axis == 0 else (p[0], n)
            candidate = walk+_line(p, q)[1:]
            if _simple(candidate):
                walk = candidate
                vertices.append(q)
                break
        else:
            break
        axis = 1-axis
    if len(vertices) < (5 if rich else 3):
        return None
    return list(reversed(walk)), vertices


def _route_option(rng, shape, redirect, rich=False):
    result = _random_walk(rng, shape, rich)
    if result is None:
        return None
    walk, vertices = result
    if not redirect:
        return {'walk': walk, 'final': list(walk), 'pivot': None, 'q': None}
    choices = []
    for pivot in vertices[1:-1]:
        i = walk.index(pivot)
        old_step = (walk[i+1][0]-pivot[0], walk[i+1][1]-pivot[1])
        d = (-old_step[0], -old_step[1])
        q = _add(pivot, d)
        if not _inside(q, shape) or q in walk or _neighbors(q, set(walk)) != [pivot]:
            continue
        final = walk[:i+1]+_ray(pivot, d, shape)
        if not _simple(final):
            continue
        if rich and (_bends(walk[i:]) < 1 or _bends(walk[:i+1]) < 1 or i < 6):
            continue
        choices.append({'walk': walk, 'final': final, 'pivot': i, 'q': q})
    if not choices:
        return None
    return choices[int(rng.integers(len(choices)))]


def _carrier(rng, palette, shape):
    a, b = shape
    k = int(rng.integers(2, len(palette)+1))
    inks = rng.choice(palette, k, replace=False)
    rr, cc = np.indices(shape)
    kind = int(rng.integers(6))
    if kind == 0:
        labels = np.tile(rng.integers(k, size=b), (a, 1))
    elif kind == 1:
        labels = np.tile(rng.integers(k, size=a)[:, None], (1, b))
    elif kind == 2:
        labels = (rr//int(rng.integers(1, a)) + cc//int(rng.integers(1, b))) % k
    elif kind == 3:
        labels = np.zeros(shape, dtype=int)
        r, c = int(rng.integers(a)), int(rng.integers(b))
        labels[(rr >= r) & (cc >= c)] = 1
    elif kind == 4:
        labels = (rr+int(rng.integers(1, k+1))*cc+int(rng.integers(k))) % k
    else:
        labels = rng.integers(k, size=shape)
    if len(np.unique(labels)) == 1:
        labels.flat[int(rng.integers(a*b))] = (int(labels.flat[0])+1) % k
    return inks[labels].astype(np.int8)


def _ribbon(rng, palette):
    for _ in range(32):
        length = int(rng.integers(2, 7))
        ribbon = rng.choice((0, *map(int, palette)), size=length).astype(np.int8)
        ribbon[int(rng.integers(length))] = 0
        if not np.any(ribbon):
            ribbon[int(rng.integers(length))] = int(rng.choice(palette))
        if np.any(ribbon == 0) and all(not (length % k == 0 and np.array_equal(
                ribbon, np.tile(ribbon[:k], length//k))) for k in range(1, length)):
            return ribbon
    return np.array([0, int(palette[0])], dtype=np.int8)


def _assemble(carrier, ribbon, board):
    a, b = carrier.shape
    left, length = b+3, len(ribbon)
    header = np.zeros((a+3, board.shape[1]), dtype=np.int8)
    header[:a+2, :b+2] = 5
    header[1:a+1, 1:b+1] = carrier
    header[:3, left:left+length+2] = 5
    header[1, left+1:left+length+1] = ribbon
    header[-1] = 5
    return np.concatenate((header, board), axis=0)


def _construct_pair(rng, palette, count, redirect, rich=False):
    # Every retry is discarded in memory. No payloads or traces are logged.
    for _scene_attempt in range(100):
        a, b = map(int, rng.integers(2, 5, size=2))
        ribbon = _ribbon(rng, palette)
        min_h, min_w = (10, 12) if count == 1 else (15, 18)
        hs = [a*k for k in range(3, 10) if min_h <= a*k <= 27-a]
        ws = [b*k for k in range(3, 16) if max(min_w, b+len(ribbon)+5) <= b*k <= 30]
        shape = (int(rng.choice(hs)), int(rng.choice(ws)))
        routes = []
        for index in range(count):
            wants_redirect = bool(redirect) if count == 1 else (index == 0 or rng.random() < .65)
            for _path_attempt in range(160):
                route = _route_option(rng, shape, wants_redirect, rich and index == 0)
                if route is None:
                    continue
                if any(not _separated(route['walk'], r['walk']) or not _separated(
                        route['final'], r['final']) for r in routes):
                    continue
                originals = set(route['walk']).union(*(set(r['walk']) for r in routes))
                qs = [r['q'] for r in routes+[route] if r['q'] is not None]
                if len(qs) != len(set(qs)) or any(len(_neighbors(q, originals)) != 1 for q in qs):
                    continue
                routes.append(route)
                break
            else:
                break
        if len(routes) != count:
            continue
        board = np.zeros(shape, dtype=np.int8)
        for route in routes:
            for p in route['walk']:
                board[p] = 8
            board[route['walk'][0]] = 9
        for route in routes:
            if route['q'] is not None:
                board[route['q']] = 7
            walk = route['walk']
            retained = walk if route['pivot'] is None else walk[:route['pivot']+1]
            # At least one retained waypoint; additional sites may be on discarded bends/endpoints.
            sites = list(retained[1:])
            if not sites:
                break
            board[sites[int(rng.integers(len(sites)))]] = 6
            for _ in range(int(rng.integers(0, 5))):
                p = walk[int(rng.integers(1, len(walk)))]
                board[p] = 6
            if route['pivot'] is not None and rng.random() < .4:
                board[walk[route['pivot']]] = 6
            gap_sites = []
            for i in range(1, len(retained)-1):
                p = retained[i]
                if (retained[i-1][0]+retained[i+1][0] == 2*p[0]
                        and retained[i-1][1]+retained[i+1][1] == 2*p[1] and board[p] == 8):
                    gap_sites.append(p)
            rng.shuffle(gap_sites)
            chosen = []
            for p in gap_sites:
                if all(sum(abs(p[j]-q[j]) for j in (0, 1)) > 2 for q in chosen):
                    chosen.append(p)
                    if len(chosen) >= int(rng.integers(1, 4)):
                        break
            if not chosen:
                break
            for p in chosen:
                board[p] = 0
        else:
            carrier = _carrier(rng, palette, (a, b))
            inp = _assemble(carrier, ribbon, board)
            try:
                recovered = _analyze(inp)
                planned = {tuple(r['walk']): tuple(r['final']) for r in routes}
                observed = {tuple(r['walk']): tuple(r['final']) for r in recovered['routes']}
                assert planned == observed
            except AssertionError:
                continue
            # Construct the label from planned geometry, not solve or reconstruction.
            out = np.repeat(np.repeat(carrier, shape[0]//a, axis=0), shape[1]//b, axis=1)
            for route in routes:
                path = route['final']
                for i, p in enumerate(path):
                    value = int(ribbon[i % len(ribbon)])
                    if value:
                        out[p] = value
            return {'input': inp, 'output': out.astype(np.int8)}, recovered
    raise RuntimeError('bounded scene construction exhausted')


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n_train, n_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = rng.choice(INKS, size=int(rng.integers(2, 5)), replace=False)
    # Keep only the best teaching candidates, rather than a pool of full grids.
    first = second = None
    first_cover = second_cover = set()
    for _ in range(32):
        pair, data = _construct_pair(rng, palette, 1, False)
        cover = _evidence(data, pair['output'])
        if first is None or len(cover) > len(first_cover):
            first, first_cover = pair, cover
        pair, data = _construct_pair(rng, palette, 1, True, rich=True)
        cover = _evidence(data, pair['output'])
        if second is None or len(cover) > len(second_cover):
            second, second_cover = pair, cover
        if REQUIRED <= first_cover | second_cover:
            break
    else:
        raise RuntimeError('bounded teaching selection exhausted')
    puzzle = {'train': [first, second], 'test': []}
    seen = {_key(first['input']), _key(second['input'])}
    max_count = 1
    for i in range(n_train-2):
        count = 1+int(rng.random() < .4)
        max_count = max(max_count, count)
        for _ in range(16):
            pair, _data = _construct_pair(rng, palette, count, bool(rng.integers(2)))
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError('bounded uniqueness selection exhausted')
        puzzle['train'].append(pair)
        seen.add(_key(pair['input']))
    for i in range(n_test):
        for _ in range(16):
            pair, data = _construct_pair(rng, palette, max_count+1, True)
            # Avoid a query whose redirected ink happens to be completely hidden
            # by matching material colors. Natural phase alignments remain legal.
            visible = _evidence(data, pair['output'])
            if (_key(pair['input']) not in seen and
                    {'ignore_redirect', 'ignore_ribbon', 'tile_carrier', 'uncounted_gaps'} <= visible):
                break
        else:
            raise RuntimeError('bounded uniqueness selection exhausted')
        puzzle['test'].append(pair)
        seen.add(_key(pair['input']))
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, evidence = set(), set()
    counts = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            data = _analyze(inp)
            assert out.shape == data['board'].shape and np.all(np.isin(out, INKS))
            assert np.array_equal(_paint(data), out), 'incorrect complete output'
            counts[split].append(len(data['routes']))
            if split == 'train':
                evidence |= _evidence(data, out)
            else:
                assert {'ignore_redirect', 'ignore_ribbon', 'tile_carrier', 'uncounted_gaps'} <= _evidence(data, out), \
                    'query hides a core operation'
    assert REQUIRED <= evidence, 'demonstrations miss: '+str(sorted(REQUIRED-evidence))
    assert all(n == max(counts['train'])+1 for n in counts['test']), 'missing route multiplicity shift'


def _signature(puzzle):
    return tuple(_key(pair[k]) for s in ('train', 'test') for pair in puzzle[s] for k in ('input', 'output'))


def _normalized(a, transparent=False):
    roles = {0: 0} if transparent else {}
    return a.shape, bytes(roles.setdefault(int(v), len(roles)) for v in a.flat)


def _features(data):
    return {'canvas': data['board'].shape, 'carrier_shape': data['carrier'].shape,
            'carrier': _key(data['carrier']), 'ribbon': tuple(map(int, data['ribbon'])),
            'period': len(data['ribbon']), 'routes': len(data['routes']),
            'gaps': len(data['gaps']), 'repairs': int(np.sum(data['board'] == 6)),
            'directions': {r['direction'] for r in data['routes'] if r['pivot'] is not None},
            'old_bends': {_bends(r['walk'][r['pivot']:]) for r in data['routes'] if r['pivot'] is not None},
            'ray_lengths': {len(r['ray']) for r in data['routes'] if r['pivot'] is not None},
            'turns': {_bends(r['walk']) for r in data['routes']}}


def self_test():
    """42 streaming seeds, structural metrics, meaningful mutations, count boundaries."""
    signatures, shapes, carriers, ribbons, periods, routes, directions = (set() for _ in range(7))
    carrier_structures, ribbon_structures, route_structures = set(), set(), set()
    bends, ray_lengths, turns, repair_counts, gap_counts = (set() for _ in range(5))
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        digest = hashlib.sha256(repr(_signature(puzzle)).encode()).hexdigest()
        signatures.add(digest)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                data = _analyze(pair['input'])
                f = _features(data)
                shapes.add(f['canvas'])
                carriers.add(f['carrier'])
                ribbons.add(f['ribbon'])
                carrier_structures.add(_normalized(data['carrier']))
                ribbon_structures.add(_normalized(data['ribbon'], transparent=True))
                geometry = (data['board'].shape, tuple(tuple(r['walk']) for r in data['routes']),
                            tuple(tuple(r['final']) for r in data['routes']))
                route_structures.add(hashlib.sha256(repr(geometry).encode()).digest())
                periods.add(f['period'])
                routes.add(f['routes'])
                directions |= f['directions']
                bends |= f['old_bends']
                ray_lengths |= f['ray_lengths']
                turns |= f['turns']
                repair_counts.add(f['repairs'])
                gap_counts.add(f['gaps'])
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
        if seed < 3:
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
            # Reassign all material inks bijectively; marker semantics stay visible.
            lut = np.arange(10, dtype=np.int8)
            lut[1:5] = [4, 3, 1, 2]
            for pair in puzzle['test']:
                assert np.array_equal(solve(lut[pair['input']]), lut[pair['output']])
            bad = {'train': puzzle['train'], 'test': [dict(p) for p in puzzle['test']]}
            bad['test'][0]['output'] = bad['test'][0]['output'].copy()
            bad['test'][0]['output'][0, 0] = 1+int(bad['test'][0]['output'][0, 0]) % 4
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('corrupt output accepted')
            # Reject errors in geometry and transported phase as well as a single changed pixel.
            data = _analyze(puzzle['test'][0]['input'])
            for variant in ('ignore_redirect', 'reset_ray', 'uncounted_gaps', 'tile_carrier'):
                wrong = _paint(data, variant)
                if np.array_equal(wrong, puzzle['test'][0]['output']):
                    continue
                bad['test'][0]['output'] = wrong
                try:
                    validate_generated_puzzle(bad)
                except AssertionError:
                    pass
                else:
                    raise AssertionError('semantic mutation accepted: '+variant)
    assert len(signatures) == 42 and len(shapes) >= 30
    assert len(carriers) >= 100 and len(ribbons) >= 80
    assert len(carrier_structures) >= 80 and len(ribbon_structures) >= 70
    assert len(route_structures) == 210
    assert periods == set(range(2, 7)) and routes == {1, 2, 3}
    assert directions == set(DIRS) and {0, 1, 2} <= bends
    assert len(turns) >= 4 and len(repair_counts) >= 6 and len(gap_counts) >= 4
    assert min(ray_lengths) == 1 and max(ray_lengths) >= 15
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(1000+10*nt+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return {'seeds': 42, 'distinct_puzzles': len(signatures), 'canvas_shapes': len(shapes),
            'carrier_patterns': len(carriers), 'ribbon_patterns': len(ribbons),
            'normalized_carrier_patterns': len(carrier_structures),
            'normalized_ribbon_patterns': len(ribbon_structures),
            'route_geometries': len(route_structures),
            'periods': sorted(periods), 'route_counts': sorted(routes),
            'directions': sorted(directions), 'old_branch_bends': sorted(bends),
            'ray_lengths': [min(ray_lengths), max(ray_lengths)]}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
