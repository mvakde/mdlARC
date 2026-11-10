"""Keyed relay circuits: one-bar lane changes make a directed station graph.

No training, random state, or generation plan is needed by solve().  Bars and
station borders share a visible carrier; the other bar color is the marker.
"""
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid using colors 0..9")
    return a.copy()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, cells = [first], []
        while stack:
            y, x = stack.pop()
            cells.append((int(y), int(x)))
            for dy, dx in DIRS:
                q = (y + dy, x + dx)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _parse(a):
    bg = int(a[0, 0])
    if not all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])):
        raise ValueError("background must form the outer frame")
    stations, raw_bars, tokens, signature = [], [], [], None
    for cells in _components(a != bg):
        ys, xs = zip(*cells)
        y, x, h, w = min(ys), min(xs), max(ys)-min(ys)+1, max(xs)-min(xs)+1
        if len(cells) == 1:
            tokens.append(cells[0])
        elif h == w and h in (3, 5) and len(cells) == h*w:
            carrier = int(a[y, x])
            block = a[y:y+h, x:x+w].copy()
            label = int(block[h//2, h//2])
            block[h//2, h//2] = carrier
            if label in (bg, carrier) or not np.all(block == carrier):
                raise ValueError("station must have a single centered label")
            sig = (h, carrier)
            if signature is not None and sig != signature:
                raise ValueError("stations must share size and border color")
            signature = sig
            stations.append(dict(center=(y+h//2, x+w//2), label=label, cells=cells))
        elif min(h, w) == 1 and len(cells) == max(h, w):
            raw_bars.append(cells)
        else:
            raise ValueError("foreground must consist of stations, bars, and key tokens")
    if signature is None or len(stations) < 3 or not raw_bars or not 3 <= len(tokens) <= 7:
        raise ValueError("missing relay circuit")
    size, carrier = signature
    tokens.sort(key=lambda p: p[1])
    ky = tokens[0][0]
    if (any(y != ky for y, x in tokens) or ky != a.shape[0]-2
            or any(q[1]-p[1] != 2 for p, q in zip(tokens, tokens[1:]))):
        raise ValueError("key is read left to right on the penultimate row")
    key = tuple(int(a[p]) for p in tokens)
    if len(set(key)) < 2 or carrier in key or bg in key:
        raise ValueError("key must contain distinct label roles")
    labels = {s['label'] for s in stations}
    bar_colors = {int(a[p]) for bar in raw_bars for p in bar}
    marker_colors = bar_colors - {carrier}
    if len(marker_colors) != 1 or carrier not in bar_colors:
        raise ValueError("bars share carrier and one instruction color")
    marker = marker_colors.pop()
    if marker in labels or marker in key or marker == bg or not set(key) <= labels:
        raise ValueError("marker, structure, and label roles must be disjoint")
    bars = []
    for cells in raw_bars:
        vertical = len({p[1] for p in cells}) == 1
        axis = 0 if vertical else 1
        cells.sort(key=lambda p: p[axis])
        ms = [p for p in cells if int(a[p]) == marker]
        n = len(ms)
        low = ms == cells[:2*n:2]
        high = ms == list(reversed(cells[-1:-(2*n+1):-2]))
        if not 1 <= n <= 3 or len(cells) < 2*n or low == high:
            raise ValueError("bar must have an asymmetric alternating endpoint instruction")
        bars.append(dict(cells=cells, markers=ms, axis=axis, n=n,
                         side=1 if low else -1))
    bodies = [p for s in stations for p in s['cells']] + [p for b in bars for p in b['cells']]
    if any(p[0] >= ky-1 for p in bodies):
        raise ValueError("a blank row must separate circuit and key")
    station_at = {p: i for i, s in enumerate(stations) for p in s['cells']}
    bar_at = {p: i for i, b in enumerate(bars) for p in b['cells']}
    return dict(bg=bg, carrier=carrier, marker=marker, size=size, stations=stations,
                bars=bars, key=key, ky=ky, station_at=station_at, bar_at=bar_at)


def _add(p, v, n=1):
    return (p[0]+v[0]*n, p[1]+v[1]*n)


def _line(p, q):
    if p[0] != q[0] and p[1] != q[1]:
        raise ValueError("nonorthogonal segment")
    delta = ((q[0] > p[0]) - (q[0] < p[0]), (q[1] > p[1]) - (q[1] < p[1]))
    return [_add(p, delta, i) for i in range(abs(q[0]-p[0])+abs(q[1]-p[1])+1)]


def _hop(a, info, source, direction):
    """Trace one heading until a station, with at most one encoded lane change."""
    bg, radius = info['bg'], info['size']//2
    start = _add(info['stations'][source]['center'], direction, radius+1)
    def inside(p):
        return 1 <= p[0] <= info['ky']-2 and 1 <= p[1] < a.shape[1]-1
    def ray(p):
        clear = []
        while inside(p):
            if int(a[p]) != bg:
                return p, clear
            clear.append(p)
            p = _add(p, direction)
        return None, clear
    hit, initial = ray(start)
    if hit is None:
        return None
    bar_id, path = None, initial
    if hit in info['bar_at']:
        bar_id = info['bar_at'][hit]
        bar = info['bars'][bar_id]
        if direction[bar['axis']] != 0:
            return None  # a longitudinal bar is a wall, not a command for this heading
        d = bar['n']+1
        turn = _add(hit, direction, -d)
        if sum((turn[k]-start[k])*direction[k] for k in (0, 1)) < 0:
            return None
        end = bar['cells'][-1] if bar['side'] == 1 else bar['cells'][0]
        lane = end[bar['axis']] + bar['side']*d
        bend = list(turn)
        bend[bar['axis']] = lane
        bend = tuple(bend)
        approach = _line(start, turn) + _line(turn, bend)[1:]
        if any(not inside(p) or int(a[p]) != bg for p in approach):
            return None
        hit, tail = ray(_add(bend, direction))
        path = approach + tail
    if hit is None or hit not in info['station_at']:
        return None
    target = info['station_at'][hit]
    if target == source or hit != _add(info['stations'][target]['center'], direction, -radius):
        return None
    if not path or len(path) != len(set(path)):
        return None
    return dict(source=source, target=target, path=tuple(path), bar=bar_id, direction=direction)


def _decode(a):
    info = _parse(a)
    graph = [[h for v in DIRS if (h := _hop(a, info, i, v)) is not None]
             for i in range(len(info['stations']))]
    answers = []
    def visit(route, hops, occupied, used_bars):
        if len(answers) >= 2:
            return
        if len(route) == len(info['key']):
            answers.append((tuple(route), tuple(hops)))
            return
        for h in graph[route[-1]]:
            j = h['target']
            if (j in route or info['stations'][j]['label'] != info['key'][len(route)]
                    or occupied.intersection(h['path'])
                    or (h['bar'] is not None and h['bar'] in used_bars)):
                continue
            visit(route+[j], hops+[h], occupied | set(h['path']),
                  used_bars | ({h['bar']} if h['bar'] is not None else set()))
    for i, station in enumerate(info['stations']):
        if station['label'] == info['key'][0]:
            visit([i], [], set(), set())
    if len(answers) != 1:
        raise ValueError("key must select exactly one nonintersecting distinct-station route")
    info['route'], info['hops'] = answers[0]
    info['graph'] = graph
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    info = _decode(a)
    b = a.copy()
    for h in info['hops']:
        for p in h['path']:
            b[p] = info['stations'][h['source']]['label']
        if h['bar'] is not None:
            for p in info['bars'][h['bar']]['markers']:
                b[p] = info['carrier']
    return b


def _square(center, size):
    r = size//2
    return [(y, x) for y in range(center[0]-r, center[0]+r+1)
            for x in range(center[1]-r, center[1]+r+1)]


def _stamp(a, center, size):
    for p in _square(center, size):
        a[p] = 1
    a[center] = 3


def _separate(a, cells, reserved):
    for p in cells:
        if not (1 <= p[0] < a.shape[0]-1 and 1 <= p[1] < a.shape[1]-1):
            return False
        if p in reserved or int(a[p]) != 0:
            return False
        if any(int(a[_add(p, v)]) != 0 for v in DIRS):
            return False
    return True


def _propose(rng, a, source, size, n, reserved, decoy=False, branch=False):
    radius = size//2
    v = DIRS[int(rng.integers(4))]
    lateral = (v[1], v[0])
    if n is None:
        distance = size + int(rng.integers(1, 8))
        target = _add(source, v, distance)
        path = _line(_add(source, v, radius+1), _add(target, v, -radius-1))
        bar_cells, markers, decoy_center = [], [], None
    else:
        d = n+1
        bar_distance = radius+1+d+int(rng.integers(0, 5))
        middle = _add(source, v, bar_distance)
        length = 2*n+int(rng.integers(0, 5))
        lo = -int(rng.integers(length))
        hi = lo+length-1
        side = 1 if rng.integers(2) else -1
        lane = hi+d if side == 1 else lo-d
        target = _add(_add(middle, v, radius+2+int(rng.integers(0, 5))), lateral, lane)
        bar_cells = [_add(middle, lateral, j) for j in range(lo, hi+1)]
        markers = [_add(middle, lateral, lo+2*j if side == 1 else hi-2*j) for j in range(n)]
        decoy_center = _add(target, lateral, -lane) if decoy else None
        # Cheap bounds precede segment construction and component checks.
        centers = [target] + ([decoy_center] if decoy_center is not None else [])
        if any(not (radius+1 <= c[0] < a.shape[0]-radius-1
                    and radius+1 <= c[1] < a.shape[1]-radius-1) for c in centers):
            return None
        if any(not (1 <= p[0] < a.shape[0]-1 and 1 <= p[1] < a.shape[1]-1) for p in bar_cells):
            return None
        turn = _add(middle, v, -d)
        bend = _add(turn, lateral, lane)
        path = (_line(_add(source, v, radius+1), turn)
                + _line(turn, bend)[1:]
                + _line(bend, _add(target, v, -radius-1))[1:])
    target_cells = _square(target, size)
    objects = target_cells + bar_cells
    if decoy_center is not None:
        objects += _square(decoy_center, size)
    if len(objects) != len(set(objects)) or not _separate(a, objects, reserved):
        return None
    # Keep distinct new components separated too.
    groups = [set(target_cells), set(bar_cells)]
    if decoy_center is not None:
        groups.append(set(_square(decoy_center, size)))
    if any(any(_add(p, v) in q for p in g for v in DIRS)
           for i, g in enumerate(groups) for q in groups[i+1:]):
        return None
    if any(not (1 <= p[0] < a.shape[0]-1 and 1 <= p[1] < a.shape[1]-1)
           or int(a[p]) != 0 or p in objects or (not branch and p in reserved) for p in path):
        return None
    b = a.copy()
    _stamp(b, target, size)
    if decoy_center is not None:
        _stamp(b, decoy_center, size)
    for p in bar_cells:
        b[p] = 1
    for p in markers:
        b[p] = 2
    return b, target, tuple(path), tuple(markers), decoy_center


def _make_pair(rng, ns, size, aligned_decoy, need_branch, test, colors):
    # Bounded geometric construction.  Later attempts use the largest scene,
    # but never change the requested rule, counting evidence, or key shift.
    for attempt in range(160):
        minimum = 25 if size == 5 else 20 if max(n or 0 for n in ns) >= 2 else 16
        h, w = ((27, 27) if attempt >= 30 else
                tuple(map(int, rng.integers(minimum, 28, size=2))))
        a = np.zeros((h, w), dtype=np.int8)
        r = size//2
        start = (int(rng.integers(r+1, h-r-1)), int(rng.integers(r+1, w-r-1)))
        _stamp(a, start, size)
        stations, paths, marker_sets, decoys, reserved = [start], [], [], [], set()
        for j, n in enumerate(ns):
            made = None
            for _ in range(220):
                made = _propose(rng, a, stations[-1], size, n, reserved,
                                decoy=aligned_decoy and j == 0)
                if made is not None:
                    break
            if made is None:
                break
            a, target, path, markers, decoy = made
            stations.append(target)
            paths.append(path)
            marker_sets.append(markers)
            reserved.update(path)
            if decoy is not None:
                decoys.append((decoy, j+1))
        if len(paths) != len(ns):
            continue
        branch_target = None
        if need_branch:
            for _ in range(1400):
                source = stations[int(rng.integers(len(stations)-1))]
                made = _propose(rng, a, source, size, int(rng.integers(1, 3)),
                                reserved, branch=True)
                if made is not None:
                    a, branch_target, _, _, _ = made
                    break
            if branch_target is None:
                continue
        # Random labels, with a readable training/test change in repetition.
        pool = [int(c) for c in rng.permutation(np.arange(3, 10))]
        labels = pool[:len(stations)]
        if test:
            labels[2] = labels[1]  # two distinct required relays now share a label
            if len(labels) > 3 and rng.random() < .45:
                labels[-1] = labels[0]
        elif len(labels) > 3 and rng.random() < .5:
            labels[-1] = labels[0]
        for center, label in zip(stations, labels):
            a[center] = label
        for center, target_index in decoys:
            a[center] = labels[target_index]
        if branch_target is not None:
            a[branch_target] = pool[len(stations)]
        b = a.copy()
        for label, path, ms in zip(labels, paths, marker_sets):
            for p in path:
                b[p] = label
            for p in ms:
                b[p] = 1
        # Scene symmetries leave the detached key's reading direction fixed.
        k, flip = int(rng.integers(4)), bool(rng.integers(2))
        a, b = np.rot90(a, k), np.rot90(b, k)
        if flip:
            a, b = a[:, ::-1], b[:, ::-1]
        row_extra = int(rng.integers(28-a.shape[0]))
        col_extra = int(rng.integers(31-a.shape[1]))
        top, left = int(rng.integers(row_extra+1)), int(rng.integers(col_extra+1))
        a = np.pad(a, ((top, row_extra-top+3), (left, col_extra-left)))
        b = np.pad(b, ((top, row_extra-top+3), (left, col_extra-left)))
        key_x = int(rng.integers(1, a.shape[1]-(2*len(labels)-1)))
        a[-2, key_x:key_x+2*len(labels):2] = labels
        b[-2, key_x:key_x+2*len(labels):2] = labels
        a, b = colors[a].astype(np.int8), colors[b].astype(np.int8)
        try:
            info = _decode(a)
        except ValueError:
            continue
        # The unique input-decoded route must reproduce the independently drawn plan.
        if not np.array_equal(solve(a), b):
            continue
        if need_branch and not any(len(info['graph'][i]) > 1 for i in info['route'][:-1]):
            continue
        return {'input': a, 'output': b}
    raise RuntimeError("bounded relay placement exhausted")


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _signature(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, 'num_train')
    nq = _count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    maximum = int(rng.choice((2, 3)))
    stable = bool(rng.random() < .35)
    palette = rng.permutation(10)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            if split == 'train' and index == 0:
                ns, size, decoy, branch = [1, None], 3, True, False
            elif split == 'train' and index == 1:
                ns, size, decoy, branch = [2, maximum], 3, True, True
            else:
                size = int(rng.choice((3, 3, 3, 5)))
                length = int(rng.choice((2, 2, 3))) if size == 3 else 2
                ns = [int(rng.integers(1, maximum+1)) for _ in range(length)]
                for j in range(1, length):
                    if rng.random() < .35:
                        ns[j] = None
                decoy, branch = bool(rng.random() < .65), bool(rng.random() < .55)
                # With key [a,b,b], a same-b decoy beside the first b is itself
                # a legal final relay.  That geometry has two answers.  Longer
                # keys can disambiguate it by their remaining suffix.
                if split == 'test' and length == 2:
                    decoy = False
            for _ in range(10):
                colors = palette if stable else rng.permutation(10)
                pair = _make_pair(rng, ns, size, decoy, branch, split == 'test', colors)
                sig = _signature(pair['input'])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unique inputs exhausted")
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(count)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    a, b = _grid(pair['input']), _grid(pair['output'])
    assert a.shape == b.shape
    info = _decode(a)
    expected = solve(a)
    assert np.array_equal(b, expected), "incorrect circuit drawing or marker activation"
    used = {h['bar'] for h in info['hops'] if h['bar'] is not None}
    painted = {p for h in info['hops'] for p in h['path']}
    changed = set(map(tuple, np.argwhere(a != b)))
    markers = {p for j in used for p in info['bars'][j]['markers']}
    assert changed == painted | markers and painted.isdisjoint(markers)
    assert all(int(a[p]) == info['bg'] for p in painted)
    assert all(int(b[p]) == info['carrier'] for p in markers)
    for j, bar in enumerate(info['bars']):
        if j not in used:
            assert all(int(a[p]) == int(b[p]) for p in bar['cells'])
    assert all(int(a[p]) == int(b[p]) for s in info['stations'] for p in s['cells'])
    assert np.array_equal(a[info['ky']], b[info['ky']])
    assert used and painted
    info['used'] = used
    info['unused_bars'] = len(info['bars'])-len(used)
    info['adjacent_equal'] = any(x == y for x, y in zip(info['key'], info['key'][1:]))
    info['counts'] = {info['bars'][j]['n'] for j in used}
    info['branching'] = any(len(info['graph'][i]) > 1 for i in info['route'][:-1])
    # An aligned same-label decoy is a witness against ignoring the lane change.
    decoy = False
    for hop in info['hops']:
        if hop['bar'] is None:
            continue
        s, t = (info['stations'][hop[k]] for k in ('source', 'target'))
        axis = 1 if hop['direction'][0] else 0
        other = 1-axis
        decoy |= any(j not in info['route'] and q['label'] == t['label']
                     and q['center'][axis] == s['center'][axis]
                     and q['center'][other] == t['center'][other]
                     for j, q in enumerate(info['stations']))
    info['aligned_decoy'] = bool(decoy)
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            info = _features(pair)
            sig = _signature(np.asarray(pair['input']))
            assert sig not in seen
            seen.add(sig)
            features[split].append(info)
    train, test = features['train'], features['test']
    assert any(f['aligned_decoy'] for f in train), "B must determine repeated-label identity"
    assert any(f['branching'] and f['unused_bars'] for f in train), "A must select gate activation"
    assert any(any(h['bar'] is None for h in f['hops']) for f in train)
    assert all(not f['adjacent_equal'] for f in train)
    assert all(f['adjacent_equal'] for f in test)
    counts = set().union(*(f['counts'] for f in train))
    assert {1, 2} <= counts
    assert set().union(*(f['counts'] for f in test)) <= counts
    assert any(any(x != y for x, y in zip(f['key'], f['key'][1:])) for f in train)


def self_test():
    _literal_geometry_tests()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    repeated = generate_puzzle(0)
    assert all(_signature(x[k]) == _signature(y[k])
               for s in puzzles[0] for x, y in zip(puzzles[0][s], repeated[s])
               for k in ('input', 'output'))
    shapes, structures, sizes, counts, headings, turns, roles, routes = set(), set(), set(), set(), set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in puzzle:
            for pair in puzzle[split]:
                f = _features(pair)
                shapes.add(pair['input'].shape)
                role_map = {}
                norm = tuple(role_map.setdefault(int(c), len(role_map)) for c in pair['input'].flat)
                structures.add((pair['input'].shape, norm))
                sizes.add(f['size'])
                counts.update(f['counts'])
                headings.update(h['direction'] for h in f['hops'])
                turns.update((x['direction'], y['direction']) for x, y in zip(f['hops'], f['hops'][1:]))
                roles.add((f['bg'], f['carrier'], f['marker']))
                routes.add(len(f['route']))
    assert len(shapes) >= 35 and len(structures) >= 180 and len(roles) >= 25
    assert sizes == {3, 5} and counts == {1, 2, 3} and headings == set(DIRS)
    assert routes >= {3, 4}
    assert any(a != b and a != (-b[0], -b[1]) for a, b in turns)
    # Every legal explicit count, including the two-demo boundary, is exercised.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(8100+10*nt+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal example counts accepted")
    # Tampering with active geometry, coloring, or unused commands must fail.
    p = puzzles[0]
    for kind in ('route', 'station', 'unused'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        for pair in bad['train']:
            f = _features(pair)
            if kind == 'route':
                pos = f['hops'][0]['path'][0]
                pair['output'][pos] = f['bg']
                break
            if kind == 'station':
                pos = f['stations'][f['route'][0]]['center']
                pair['output'][pos] = f['carrier']
                break
            candidates = [j for j in range(len(f['bars'])) if j not in f['used']]
            if candidates:
                pair['output'][f['bars'][candidates[0]]['markers'][0]] = f['carrier']
                break
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError(f"{kind} corruption was accepted")


def _literal_geometry_tests():
    """Hand-specified turns, captures and selective edits; no random constructor."""
    a = np.zeros((14, 23), dtype=np.int8)
    for center, label in (((3, 5), 3), ((11, 9), 4), ((11, 18), 5),
                          ((11, 5), 4), ((7, 13), 6)):
        _stamp(a, center, 3)
        a[center] = label
    a[8, 3:7] = (2, 1, 2, 1)  # n=2; turn row 5, resume in column 9
    a[2:6, 10] = (2, 1, 1, 1)  # unused branch: n=1, resume in row 7
    b = a.copy()
    b[5, 5:10] = 3
    b[6:10, 9] = 3
    b[11, 11:17] = 4
    b[8, (3, 5)] = 1
    cases = [(a, b, (3, 4, 5))]
    # Select the other gate.  Its two-step clearance and downstream capture
    # differ literally, while the n=2 gate now retains its instruction cells.
    a2 = a.copy()
    _stamp(a2, (11, 13), 3)
    a2[11, 13] = 7
    b2 = a2.copy()
    b2[3, 7:9] = 3
    b2[4:8, 8] = 3
    b2[7, 9:12] = 3
    b2[9, 13] = 6
    b2[2, 10] = 1
    cases.append((a2, b2, (3, 6, 7)))
    permutation = np.array((8, 0, 9, 2, 7, 5, 1, 6, 3, 4), dtype=np.int8)
    for a, b, key in cases:
        for rotation in range(4):
            for flip in (False, True):
                x, y = np.rot90(a, rotation), np.rot90(b, rotation)
                if flip:
                    x, y = x[:, ::-1], y[:, ::-1]
                x, y = np.pad(x, ((0, 3), (0, 0))), np.pad(y, ((0, 3), (0, 0)))
                x[-2, (2, 4, 6)] = key
                y[-2, (2, 4, 6)] = key
                assert np.array_equal(solve(x), y), "literal clearance/capture fixture failed"
                assert np.array_equal(solve(permutation[x]), permutation[y])
    # Removing a unary instruction makes its actual destination unreachable.
    x = np.pad(cases[0][0], ((0, 3), (0, 0)))
    x[-2, (2, 4, 6)] = (3, 4, 5)
    x[8, 5] = 1
    try:
        solve(x)
    except ValueError:
        pass
    else:
        raise AssertionError("changed clearance unexpectedly retained the original route")


if __name__ == '__main__':
    self_test()
