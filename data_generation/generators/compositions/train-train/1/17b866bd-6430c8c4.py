"""Paired sluice maps: corner paint sources compete through shared openings.

Only NumPy and the standard library are used.  The solver parses the two maps
from the input, reconstructs their room graph, and has no generation state.
"""

from collections import deque
import json
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low, high=None):
    _require(isinstance(value, (int, np.integer))
             and not isinstance(value, (bool, np.bool_)), name + " must be an integer")
    value = int(value)
    _require(value >= low and (high is None or value <= high), name + " out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= s <= 30 for s in a.shape), "bad grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad grid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _edges(rb, cb):
    """Each interface has one interval of eligible aperture cells per map."""
    nr, nc = len(rb) - 1, len(cb) - 1
    result = []
    for r in range(nr):
        for c in range(nc):
            u = r * nc + c
            if c + 1 < nc:
                cells = [(y, cb[c + 1]) for y in range(rb[r] + 2, rb[r + 1] - 1)]
                result.append((u, u + 1, cells))
            if r + 1 < nr:
                cells = [(rb[r + 1], x) for x in range(cb[c] + 2, cb[c + 1] - 1)]
                result.append((u, u + nc, cells))
    return result


def _scaffold(rb, cb, background, wall):
    a = np.full((rb[-1] + 1, cb[-1] + 1), background, dtype=np.int8)
    a[rb, :] = wall
    a[:, cb] = wall
    a[np.ix_(rb, cb)] = background
    return a


def _distances(n, edges, active, sources):
    adj = [[] for _ in range(n)]
    for (u, v, _), opened in zip(edges, active):
        if opened:
            adj[u].append(v)
            adj[v].append(u)
    d = np.full((len(sources), n), n + 1, dtype=int)
    for k, u in enumerate(sources):
        d[k, u] = 0
        q = deque([u])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if d[k, v] > d[k, u] + 1:
                    d[k, v] = d[k, u] + 1
                    q.append(v)
    return d


def _winners(n, edges, active, sources, background, distances=None):
    d = _distances(n, edges, active, sources) if distances is None else distances
    winners = np.full(n, background, dtype=np.int8)
    nearest = []
    colors = list(sources.values())
    for u in range(n):
        if len(sources) == 0 or int(d[:, u].min()) > n:
            nearest.append([])
            continue
        ks = list(map(int, np.flatnonzero(d[:, u] == d[:, u].min())))
        nearest.append(ks)
        paint = {colors[k] for k in ks}
        if len(paint) == 1:
            winners[u] = next(iter(paint))
    return winners, d, nearest


def _render(first, second, rb, cb, background, wall, winners):
    # Erase source slots before combining structural masks.  Open gates remain
    # background: only chamber interiors are colored by the graph computation.
    clean = first.copy()
    clean[np.ix_(rb, cb)] = background
    out = np.where((clean != background) | (second != background), wall, background).astype(np.int8)
    nc = len(cb) - 1
    for r in range(len(rb) - 1):
        for c in range(nc):
            out[rb[r] + 1:rb[r + 1], cb[c] + 1:cb[c + 1]] = winners[r * nc + c]
    return out


def _parse(input_grid):
    a = _grid(input_grid)
    candidates = []
    for axis in (0, 1):
        size = a.shape[axis]
        if size % 2 != 1 or size < 3:
            continue
        m = size // 2
        first = a[:m, :] if axis == 0 else a[:, :m]
        line = a[m, :] if axis == 0 else a[:, m]
        second = a[m + 1:, :] if axis == 0 else a[:, m + 1:]
        divider = int(line[0])
        if np.all(line == divider) and not np.any(first == divider) and not np.any(second == divider):
            candidates.append((axis, divider, first, second))
    _require(len(candidates) == 1, "divider must be uniquely visible")
    axis, divider, first, second = candidates[0]
    _require(min(first.shape) >= 5, "map too small")
    background = int(second[-1, -1])
    wall, other_wall = int(first[0, 1]), int(second[0, 1])
    _require(len({background, wall, other_wall, divider}) == 4, "structural roles must differ")
    rb = list(map(int, np.flatnonzero(second[:, -1] == background)))
    cb = list(map(int, np.flatnonzero(second[-1, :] == background)))
    _require(len(rb) >= 2 and len(cb) >= 2 and rb[0] == cb[0] == 0
             and rb[-1] == first.shape[0] - 1 and cb[-1] == first.shape[1] - 1,
             "missing boundary corners")
    _require(all(4 <= b - x <= 9 for x, b in zip(rb, rb[1:]))
             and all(4 <= b - x <= 9 for x, b in zip(cb, cb[1:])), "bad room dimensions")
    edges = _edges(rb, cb)
    expected1 = _scaffold(rb, cb, background, wall)
    expected2 = _scaffold(rb, cb, background, other_wall)
    holes1, holes2 = [], []
    for _, _, cells in edges:
        openings = []
        for panel, expected, color in ((first, expected1, wall), (second, expected2, other_wall)):
            _require(all(int(panel[p]) in (background, color) for p in cells), "invalid gate color")
            opened = [k for k, p in enumerate(cells) if panel[p] == background]
            _require(not opened or opened == list(range(opened[0], opened[-1] + 1)), "gate must be an interval")
            for k in opened:
                expected[cells[k]] = background
            openings.append(set(opened))
        holes1.append(openings[0])
        holes2.append(openings[1])
    nr, nc = len(rb) - 1, len(cb) - 1
    sources = {}
    for r in range(nr):
        for c in range(nc):
            paint = int(first[rb[r], cb[c]])
            if paint != background:
                _require(paint != divider, "source equals divider")
                sources[r * nc + c] = paint
                expected1[rb[r], cb[c]] = paint
    _require(np.array_equal(first, expected1) and np.array_equal(second, expected2), "invalid room scaffold")
    return dict(first=first, second=second, rb=rb, cb=cb, nr=nr, nc=nc, n=nr * nc,
                edges=edges, holes1=holes1, holes2=holes2,
                active=[bool(x & y) for x, y in zip(holes1, holes2)], sources=sources,
                background=background, wall=wall, other_wall=other_wall,
                divider=divider, axis=axis)


def _solve_info(info):
    w, _, _ = _winners(info['n'], info['edges'], info['active'], info['sources'], info['background'])
    return _render(info['first'], info['second'], info['rb'], info['cb'], info['background'], info['wall'], w)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return one merged, filled map using only the visible input."""
    return _solve_info(_parse(input_grid))


def _diagnostics(info):
    n, sources, bg = info['n'], info['sources'], info['background']
    w, d, nearest = _winners(n, info['edges'], info['active'], sources, bg)
    paint = list(sources.values())
    tie_diff = any(len({paint[k] for k in ks}) > 1 for ks in nearest)
    tie_same = any(len(ks) > 1 and len({paint[k] for k in ks}) == 1 for ks in nearest)
    flags = dict(tie_diff=tie_diff, tie_same=tie_same,
                 nonlocal_fill=any(u not in sources and w[u] != bg for u in range(n)),
                 unreachable=any(not ks for ks in nearest))
    variants = {'first_only': [bool(x) for x in info['holes1']],
                'second_only': [bool(x) for x in info['holes2']],
                'either_open': [bool(x | y) for x, y in zip(info['holes1'], info['holes2'])],
                'unaligned': [bool(x) and bool(y) for x, y in zip(info['holes1'], info['holes2'])]}
    for name, active in variants.items():
        alternative, _, _ = _winners(n, info['edges'], active, sources, bg)
        flags[name + '_differs'] = not np.array_equal(w, alternative)
    md = np.asarray([[abs(u // info['nc'] - s // info['nc']) + abs(u % info['nc'] - s % info['nc'])
                      for u in range(n)] for s in sources], dtype=int).reshape(len(sources), n)
    manhattan, _, _ = _winners(n, info['edges'], info['active'], sources, bg, md)
    flags['detour'] = bool(sources) and any(nearest[u] and w[u] != manhattan[u]
                                           and d[:, u].min() > md[:, u].min() for u in range(n))
    flags['states'] = {2 * int(k not in x) + int(k not in y)
                       for (_, _, cells), x, y in zip(info['edges'], info['holes1'], info['holes2'])
                       for k in range(len(cells))}
    flags['max_distance'] = max((int(d[:, u].min()) for u in range(n) if nearest[u]), default=0)
    return flags


def _breaks(rng, count, limit):
    # Independently sized rows/columns make gate-count distance visibly distinct
    # from pixel distance.  Always leave one wall cell beside every corner slot.
    lengths = np.full(count, 4, dtype=int)
    spare = limit - 1 - 4 * count
    for _ in range(int(rng.integers(spare + 1))):
        options = np.flatnonzero(lengths < 9)
        if len(options):
            lengths[int(rng.choice(options))] += 1
    return [0] + list(map(int, np.cumsum(lengths)))


def _interval(rng, size, pivot=None):
    if pivot is None:
        start = int(rng.integers(size))
        end = int(rng.integers(start + 1, size + 1))
    else:
        start = int(rng.integers(pivot + 1))
        end = int(rng.integers(pivot + 1, size + 1))
    return set(range(start, end))


def _sample_holes(rng, edges, p):
    a, b = [], []
    for _, _, cells in edges:
        size = len(cells)
        if rng.random() < p:
            pivot = int(rng.integers(size))
            x, y = _interval(rng, size, pivot), _interval(rng, size, pivot)
        else:
            kind = int(rng.integers(4 if size >= 2 else 3))
            if kind == 0:
                x, y = _interval(rng, size), set()
            elif kind == 1:
                x, y = set(), _interval(rng, size)
            elif kind == 2:
                x, y = set(), set()
            else:
                cut = int(rng.integers(1, size))
                x = set(range(int(rng.integers(cut)), cut))
                y = set(range(cut, int(rng.integers(cut + 1, size + 1))))
                if rng.integers(2):
                    x, y = y, x
        a.append(x)
        b.append(y)
    return a, b


def _pair(rng, axis, mode):
    # 'small' teaches unlike-color ties; 'teacher' teaches same-color ties,
    # aperture alignment and actual route distance.  Other maps are unrestricted.
    if mode == 'small':
        nr, nc = (2, 3) if axis == 0 else (3, 2)
    elif mode == 'teacher':
        nr, nc = 3, 3
    elif mode == 'large':
        short = int(rng.integers(2, 4))
        long = int(rng.integers(5 if short == 2 else 4, 8))
        nr, nc = (short, long) if axis == 0 else (long, short)
    else:
        while True:
            short = int(rng.integers(1, 4))
            long = int(rng.integers(1, 8))
            nr, nc = (short, long) if axis == 0 else (long, short)
            if mode != 'ordinary_train' or nr * nc <= 9:
                break
    hlimit, wlimit = (14, 30) if axis == 0 else (30, 14)
    # A wide chamber in the teacher permits visibly shifted, disjoint holes.
    for _ in range(100):
        rb, cb = _breaks(rng, nr, hlimit), _breaks(rng, nc, wlimit)
        edges = _edges(rb, cb)
        if mode != 'teacher' or any(len(cells) >= 2 for _, _, cells in edges):
            break
    else:
        raise RuntimeError('could not choose informative geometry')
    bg, wall, wall2, divider = map(int, rng.choice(10, size=4, replace=False))
    accents = [c for c in range(10) if c not in (bg, wall, divider)]
    n = nr * nc
    for attempt in range(20000):
        p = 0.48 if mode == 'small' else 0.62 if mode == 'teacher' else float(rng.choice((0.15, 0.35, 0.55, 0.75, 0.95)))
        holes1, holes2 = _sample_holes(rng, edges, p)
        if mode == 'small':
            count = 2
            paints = [wall, int(rng.choice(accents))]
        elif mode == 'teacher':
            count = 3
            accent = int(rng.choice(accents))
            paints = [accent, accent, wall]
        else:
            if mode == 'large':
                count = int(rng.integers(2, min(5, n) + 1))
            elif rng.random() < 0.09:
                count = 0
            elif rng.random() < 0.12:
                count = n
            else:
                count = int(rng.integers(1, min(6, n) + 1))
            palette = list(map(int, rng.choice([c for c in range(10) if c not in (bg, divider)],
                                              size=int(rng.integers(1, min(5, max(1, count)) + 1)), replace=False)))
            paints = [int(rng.choice(palette)) for _ in range(count)]
        positions = list(map(int, rng.choice(n, size=count, replace=False)))
        rng.shuffle(paints)
        sources = dict(zip(positions, paints))
        info = dict(rb=rb, cb=cb, nr=nr, nc=nc, n=n, edges=edges, holes1=holes1, holes2=holes2,
                    active=[bool(x & y) for x, y in zip(holes1, holes2)], sources=sources,
                    background=bg, wall=wall, other_wall=wall2, divider=divider, axis=axis)
        flags = _diagnostics(info)
        if mode == 'small' and not (flags['tie_diff'] and flags['nonlocal_fill'] and flags['unreachable']
                                     and flags['first_only_differs'] and flags['second_only_differs']
                                     and flags['states'] == {0, 1, 2, 3}):
            continue
        if mode == 'teacher' and not (flags['tie_same'] and flags['detour'] and flags['unaligned_differs']):
            continue
        if mode == 'large' and not (flags['nonlocal_fill'] and flags['max_distance'] >= 3):
            continue
        first, second = _scaffold(rb, cb, bg, wall), _scaffold(rb, cb, bg, wall2)
        for (_, _, cells), x, y in zip(edges, holes1, holes2):
            for k in x:
                first[cells[k]] = bg
            for k in y:
                second[cells[k]] = bg
        for u, paint in sources.items():
            first[rb[u // nc], cb[u % nc]] = paint
        info.update(first=first, second=second)
        winners, _, _ = _winners(n, edges, info['active'], sources, bg)
        output = _render(first, second, rb, cb, bg, wall, winners)
        line = np.full((1, first.shape[1]) if axis == 0 else (first.shape[0], 1), divider, dtype=np.int8)
        inp = np.concatenate((first, line, second), axis=axis)
        return {'input': inp, 'output': output}
    raise RuntimeError('could not sample informative ' + mode + ' map')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    ne = _integer(num_test, 'num_test', 1, 4)
    _require(nt > ne, 'num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    puzzle = {'train': [], 'test': []}
    seen = set()
    first_axis = int(rng.integers(2))
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            mode = ('small' if index == 0 else 'teacher' if index == 1 else 'ordinary_train') if split == 'train' else ('large' if index == 0 else 'ordinary_test')
            axis = first_axis if split == 'train' and index == 0 else 1 - first_axis if split == 'train' and index == 1 else int(rng.integers(2))
            for _ in range(100):
                pair = _pair(rng, axis, mode)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('failed to generate distinct input')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate labels, structure, propagation, and teaching counterexamples."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'illegal counts')
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'bad pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            info = _parse(inp)
            _require(out.shape == info['first'].shape, 'wrong output dimensions')
            _require(np.array_equal(out, _solve_info(info)), 'wrong propagation result')
            # Check task-specific invariants explicitly, including wall-colored
            # sources and gates which remain empty even between painted rooms.
            rb, cb = info['rb'], info['cb']
            _require(np.all(out[np.ix_(rb, cb)] == info['background']), 'source slot not erased')
            for r in range(info['nr']):
                for c in range(info['nc']):
                    cavity = out[rb[r] + 1:rb[r + 1], cb[c] + 1:cb[c + 1]]
                    _require(len(np.unique(cavity)) == 1, 'partially filled chamber')
                    u = r * info['nc'] + c
                    if u in info['sources']:
                        _require(np.all(cavity == info['sources'][u]), 'source lost its own chamber')
            for (_, _, cells), x, y in zip(info['edges'], info['holes1'], info['holes2']):
                for k, point in enumerate(cells):
                    _require(out[point] == (info['background'] if k in x & y else info['wall']), 'wrong shared aperture')
            info['flags'] = _diagnostics(info)
            infos[split].append(info)
    train = infos['train']
    for flag in ('tie_diff', 'tie_same', 'nonlocal_fill', 'unreachable', 'detour',
                 'first_only_differs', 'second_only_differs', 'either_open_differs', 'unaligned_differs'):
        _require(any(i['flags'][flag] for i in train), 'training lacks ' + flag)
    _require(any(i['wall'] in i['sources'].values() for i in train), 'no outline-colored marker demonstration')
    _require(any(len(set(i['sources'].values())) >= 2 for i in train), 'no color-copying contrast')
    _require(set.union(*(i['flags']['states'] for i in train)) == {0, 1, 2, 3}, 'incomplete gate truth table')
    _require({i['axis'] for i in train} == {0, 1}, 'both panel layouts must be demonstrated')
    _require(infos['test'][0]['n'] > max(i['n'] for i in train), 'missing room-count shift')


def self_test():
    """Bounded checks on 42 seeds, plus explicit count and mutation probes."""
    shapes, geometries, graphs, palettes, source_counts, room_counts = set(), set(), set(), set(), set(), set()
    cases = {'blank': False, 'full_sources': False, 'unequal_rooms': False, 'loop': False,
             'wall2_paint': False, 'wide_gate': False, 'shifted_open': False}
    fingerprints = set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _parse(pair['input'])
                _require(np.array_equal(solve(pair['input']), pair['output']), 'solver mismatch')
                shapes.add(pair['input'].shape)
                geometries.add((tuple(np.diff(info['rb'])), tuple(np.diff(info['cb']))))
                graphs.add((info['nr'], info['nc'], tuple(info['active'])))
                palettes.add((info['background'], info['wall'], info['other_wall'], info['divider']))
                source_counts.add(len(info['sources']))
                room_counts.add(info['n'])
                cases['blank'] |= not info['sources']
                cases['full_sources'] |= len(info['sources']) == info['n']
                cases['unequal_rooms'] |= len(set(np.diff(info['rb']))) > 1 or len(set(np.diff(info['cb']))) > 1
                cases['wall2_paint'] |= info['other_wall'] in info['sources'].values()
                cases['wide_gate'] |= any(len(x & y) > 1 for x, y in zip(info['holes1'], info['holes2']))
                cases['shifted_open'] |= any(x and y and not x & y for x, y in zip(info['holes1'], info['holes2']))
                # A component has a cycle iff its active-edge count >= vertices.
                parent = list(range(info['n']))
                def root(x):
                    while parent[x] != x:
                        x = parent[x]
                    return x
                for (u, v, _), opened in zip(info['edges'], info['active']):
                    if opened:
                        x, y = root(u), root(v)
                        if x == y:
                            cases['loop'] = True
                        parent[x] = y
    _require(len(fingerprints) == 42, 'seeds repeat puzzles')
    _require(len(shapes) >= 30 and len(geometries) >= 80 and len(graphs) >= 100, 'insufficient structural variation')
    _require(len(palettes) >= 100 and len(source_counts) >= 7 and len(room_counts) >= 10, 'insufficient role/count variation')
    _require(all(cases.values()), 'missing variation: ' + str(cases))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(123, nt, ne)
            _require((len(p['train']), len(p['test'])) == (nt, ne), 'count mismatch')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    a, b = generate_puzzle(9876), generate_puzzle(9876)
    _require(all(np.array_equal(x[k], y[k]) for s in ('train', 'test')
                 for x, y in zip(a[s], b[s]) for k in ('input', 'output')), 'not reproducible')
    before = a['test'][0]['input'].copy()
    solve(a['test'][0]['input'])
    _require(np.array_equal(before, a['test'][0]['input']), 'solver mutates input')
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    corrupted['test'][0]['output'][1, 1] = (int(corrupted['test'][0]['output'][1, 1]) + 1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError('validator accepted wrong chamber')
    return {'seeds': 42, 'input_shapes': len(shapes), 'geometries': len(geometries),
            'room_graphs': len(graphs), 'palettes': len(palettes),
            'source_counts': sorted(source_counts), 'room_counts': sorted(room_counts), 'cases': cases}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
