"""Key-controlled diagonal relays. Only NumPy and the standard library are used."""
from __future__ import annotations

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _segment(a, b):
    dr, dc = b[0]-a[0], b[1]-a[1]
    if dr == 0 or abs(dr) != abs(dc):
        raise ValueError("Seed endpoints must be diagonally aligned")
    sr, sc = (1 if dr > 0 else -1), (1 if dc > 0 else -1)
    return tuple((a[0]+sr*t, a[1]+sc*t) for t in range(abs(dr)+1))


def _line(p, slope, shape):
    h, w = shape
    return tuple((r, p[1]+slope*(r-p[0])) for r in range(h)
                 if 0 <= p[1]+slope*(r-p[0]) < w)


def _parse(grid):
    if grid.shape[0] < 7 or grid.shape[1] < 7:
        raise ValueError("Scene, divider and key do not fit")
    bg, carrier = int(grid[-1, 0]), int(grid[-2, 0])
    if bg == carrier or not np.all(grid[-2] == carrier):
        raise ValueError("Missing solid divider")
    xs = np.flatnonzero(grid[-1] != bg)
    if (not 1 <= len(xs) <= 7 or xs[0] == 0 or xs[-1] == grid.shape[1]-1
            or np.any(np.diff(xs) != 2)):
        raise ValueError("Key must have 1..7 separated left-to-right tokens")
    key = tuple(int(grid[-1, x]) for x in xs)
    if carrier in key:
        raise ValueError("Carrier cannot be a label")
    scene = grid[:-2]
    frames, centers, sizes = set(), [], []
    for cells in _components(scene == carrier):
        rs, cs = zip(*cells)
        r, c, size = min(rs), min(cs), max(rs)-min(rs)+1
        if size < 3 or size % 2 == 0 or max(cs)-c+1 != size:
            raise ValueError("Endpoint carriers must be odd squares")
        center = (r+size//2, c+size//2)
        square = {(y, x) for y in range(r, r+size) for x in range(c, c+size)}
        if cells != square-{center} or int(scene[center]) != key[0]:
            raise ValueError("Framed endpoint centers must match the first key token")
        frames.update(square)
        centers.append(center)
        sizes.append(size)
    if len(centers) != 2 or sizes[0] != sizes[1]:
        raise ValueError("Exactly two equal endpoint stations are required")
    centers.sort()
    seed = _segment(*centers)
    slope = 1 if centers[1][1] > centers[0][1] else -1
    points = {tuple(map(int, q)): int(scene[tuple(q)])
              for q in np.argwhere((scene != bg) & (scene != carrier))
              if tuple(q) not in frames}
    gap = set(seed)-frames
    if len(gap) < 1:
        raise ValueError("The seed needs a positive open gap")
    return dict(bg=bg, carrier=carrier, key=key, centers=tuple(centers),
                size=sizes[0], frames=frames, seed=seed, slope=slope,
                points=points, gap=gap, shape=scene.shape)


def _routes(info, key=None, limit=2):
    """Enumerate whole distinct-dot routes, not greedy nearest neighbors."""
    tokens = info['key'][1:] if key is None else tuple(key)
    candidates = [[p for p, color in sorted(info['points'].items()) if color == k]
                  for k in tokens]
    answers = []

    def visit(path, available, incoming):
        if len(answers) >= limit:
            return
        if len(path) == len(tokens):
            answers.append(tuple(path))
            return
        for p in candidates[len(path)]:
            if p in available and p not in path:
                visit(path+[p], set(_line(p, -incoming, info['shape'])), -incoming)

    visit([], info['gap'], info['slope'])
    return answers


def _draw(grid, info, route, key=None):
    """Chronological painting; every original non-background cell is protected."""
    tokens = info['key'] if key is None else tuple(key)
    output = grid.copy()
    layers = [(info['seed'], tokens[0])]
    slope = info['slope']
    for p, color in zip(route, tokens[1:]):
        slope = -slope
        layers.append((_line(p, slope, info['shape']), color))
    for cells, color in layers:
        for p in cells:
            if grid[p] == info['bg']:
                output[p] = color
    return output, layers


def _decode(grid):
    info = _parse(grid)
    routes = _routes(info)
    if len(routes) != 1:
        raise ValueError("Key must specify exactly one complete distinct-dot route")
    info['route'] = routes[0]
    output, info['layers'] = _draw(grid, info, routes[0])
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all roles, geometry and temporal order from this input alone."""
    return _decode(_grid(input_grid))[0]


def _signature(grid):
    return tuple(grid.shape), grid.astype(np.int8).tobytes()


def _safe(p, dots, frames):
    return (p not in frames and all(max(abs(p[0]-q[0]), abs(p[1]-q[1])) >= 2
                                   for q in dots))


def _geometry(rng, shape, size, n, distance_range, crossing, branching):
    h, w = shape
    m = size//2
    maximum = min(h, w)-size-2
    low, high = distance_range
    low, high = max(size+7, low), min(maximum, high)
    if low > high:
        return None
    distance = int(rng.integers(low, high+1))
    r = int(rng.integers(m+1, h-m-distance-1))
    x = int(rng.integers(m+1, w-m-distance-1))
    slope = int(rng.choice((-1, 1)))
    a = (r, x if slope == 1 else x+distance)
    b = (r+distance, a[1]+slope*distance)
    frames = {(y, z) for p in (a, b) for y in range(p[0]-m, p[0]+m+1)
              for z in range(p[1]-m, p[1]+m+1)}
    seed = _segment(a, b)
    gap = tuple(p for p in seed if p not in frames)
    route, incoming = [], slope
    available = gap
    for index in range(n):
        options = [p for p in available if _safe(p, route, frames)]
        if index == 0:
            # Leave room before the selected trigger for two distinct decoys.
            options = [p for p in options if p[0] >= gap[0][0]+4]
            if branching:
                options = [p for p in options if p[0] <= gap[-1][0]-2]
        if index == 2 and crossing:
            options = [p for p in options
                       if (set(_line(p, -incoming, shape)) & set(gap))-set(route)]
        if not options:
            return None
        p = options[int(rng.integers(len(options)))]
        route.append(p)
        incoming = -incoming
        available = _line(p, incoming, shape)
    return dict(centers=(a, b), size=size, frames=frames, seed=seed,
                gap=set(gap), slope=slope, shape=shape, route=tuple(route))


def _recency_cells(grid, info):
    old, evidence = {}, set()
    for cells, color in info['layers']:
        for p in cells:
            if grid[p] != info['bg']:
                continue
            if p in old and old[p] != color:
                evidence.add(p)
            old[p] = color
    return evidence


def _first_wins(grid, info):
    out, visited = grid.copy(), set()
    for cells, color in info['layers']:
        for p in cells:
            if grid[p] == info['bg'] and p not in visited:
                out[p] = color
                visited.add(p)
    return out


def _without_key(grid, info):
    """B-like one-wave alternative: every dot on the finite seed launches."""
    out, _ = _draw(grid, info, ())
    for p, color in info['points'].items():
        if p in info['gap']:
            for q in _line(p, -info['slope'], info['shape']):
                if grid[q] == info['bg']:
                    out[q] = color
    return out


def _without_turns(grid, info):
    """A-like alternative: follow the selected dots with finite straight links."""
    out, _ = _draw(grid, info, ())
    previous = info['route'][0] if info['route'] else None
    for index, p in enumerate(info['route']):
        if index:
            for q in _segment(previous, p):
                if grid[q] == info['bg']:
                    out[q] = info['key'][index]
        previous = p
    return out


def _sample_pair(rng, roles, mode, split, witness, used):
    bg, carrier = roles
    pool = [c for c in range(10) if c not in roles]
    for attempt in range(500):
        if witness:
            n = int(rng.choice((3, 4)))
        elif mode == 'long_route':
            n = int(rng.integers(5, 7)) if split == 'test' else int(rng.integers(0, 5))
        elif mode == 'short_query':
            n = int(rng.integers(0, 3)) if split == 'test' else int(rng.integers(3, 7))
        else:
            n = int(rng.integers(0, 5))
        if mode == 'long_seed' and split == 'test':
            shape = (int(rng.integers(23, 29)), int(rng.integers(23, 31)))
            size, distances = 3, (16, 24)
        elif mode == 'short_query' and split == 'test':
            shape = (int(rng.integers(15, 19)), int(rng.integers(15, 23)))
            size, distances = 3, (10, 14)
        else:
            lo = 22 if mode == 'short_query' else 15
            shape = (int(rng.integers(lo, 29)), int(rng.integers(max(lo, 2*n+3), 31)))
            size = int(rng.choice((3, 3, 5))) if min(shape) >= 19 else 3
            distances = (size+7, size+9) if mode == 'long_seed' else (size+7, 25)
        g = _geometry(rng, shape, size, n, distances,
                      witness == 'order', witness == 'choice')
        if g is None:
            continue
        colors = list(map(int, rng.permutation(pool)))
        key = colors[:n+1]
        # Repeated route labels coexist with repeated-color off-route candidates.
        if n >= 3 and rng.random() < .65:
            index = int(rng.integers(3, n+1))
            key[index] = key[int(rng.integers(0, index-1))]
        grid = np.full((shape[0]+2, shape[1]), bg, dtype=np.int8)
        grid[-2] = carrier
        m = size//2
        for p in g['centers']:
            grid[p[0]-m:p[0]+m+1, p[1]-m:p[1]+m+1] = carrier
            grid[p] = key[0]
        points = {p: color for p, color in zip(g['route'], key[1:])}
        if witness == 'choice':
            earlier = [p for p in sorted(g['gap']) if p[0] < g['route'][0][0]
                       and _safe(p, points, g['frames'])]
            if not earlier:
                continue
            points[earlier[0]] = key[1]
            later = [p for p in sorted(g['gap']) if p[0] > g['route'][0][0]
                     and _safe(p, points, g['frames'])]
            if not later:
                continue
            points[later[-1]] = key[1]
            unused = next(c for c in pool if c not in key)
            wrong_color = [p for p in sorted(g['gap']) if _safe(p, points, g['frames'])]
            if not wrong_color:
                continue
            points[wrong_color[0]] = unused
            # An apparently eligible matching dot beyond the finite seed's end.
            a = g['centers'][0]
            beyond = (a[0]-m-1, a[1]-g['slope']*(m+1))
            if not _safe(beyond, points, g['frames']):
                continue
            points[beyond] = key[1]
        extra_count = int(rng.integers(0, 6))
        for flat in rng.permutation(shape[0]*shape[1]):
            if extra_count == 0:
                break
            p = (int(flat)//shape[1], int(flat)%shape[1])
            if _safe(p, points, g['frames']):
                points[p] = int(rng.choice(pool))
                extra_count -= 1
        for p, color in points.items():
            grid[p] = color
        start = int(rng.integers(1, shape[1]-2*len(key)+1))
        grid[-1, start:start+2*len(key):2] = key
        info = dict(g, bg=bg, carrier=carrier, points=points, key=tuple(key))
        routes = _routes(info)
        if routes != [g['route']]:
            continue
        output, layers = _draw(grid, info, g['route'])
        info['layers'] = layers
        if witness == 'order' and not _recency_cells(grid, info):
            continue
        if witness == 'choice' and np.array_equal(_without_key(grid, info), output):
            continue
        if _signature(grid) in used:
            continue
        return {'input': grid, 'output': output}
    return _guaranteed_pair(roles, mode, split, witness, used)


def _guaranteed_pair(roles, mode, split, witness, used):
    """A finite emergency construction obeying the same rule and split shift.

    Six diagonal turns are available in a single scene. Distinct key labels
    make the intended route unique despite the deliberately failed first branch.
    Sixteen key translations exceed the thirteen possible previous inputs.
    """
    bg, carrier = roles
    colors = [c for c in range(10) if c not in roles]
    if mode == 'long_route':
        n = 5 if split == 'test' else 3
    elif mode == 'short_query':
        n = 2 if split == 'test' else 4
    else:
        n = 3
    a = (2, 2)
    b = (20, 20) if mode == 'long_seed' and split == 'test' else (14, 14)
    route = ((9, 9), (5, 13), (8, 16), (18, 6), (16, 4), (7, 13))[:n]
    key = colors[:n+1]
    base = np.full((30, 30), bg, dtype=np.int8)
    base[-2] = carrier
    for r, c in (a, b):
        base[r-1:r+2, c-1:c+2] = carrier
        base[r, c] = key[0]
    for p, color in zip(route, key[1:]):
        base[p] = color
    for p, color in (((4, 4), key[1]), ((11, 11), key[1]),
                     ((6, 6), colors[-1]), ((0, 0), key[1])):
        base[p] = color
    for start in range(1, 30-2*len(key)+1):
        grid = base.copy()
        grid[-1, start:start+2*len(key):2] = key
        if _signature(grid) not in used:
            info = _parse(grid)
            assert _routes(info) == [route]
            output, _ = _draw(grid, info, route)
            return {'input': grid, 'output': output}
    raise AssertionError("Emergency construction has more distinct inputs than a legal puzzle")


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError("Counts must be integers: train 2..10 and test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    varied_palette = bool(rng.integers(2))
    mode = str(rng.choice(('long_route', 'long_seed', 'short_query')))
    witnesses = ['choice', 'order'] + [None]*(int(num_train)-2)
    rng.shuffle(witnesses)
    puzzle, used = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            pair_roles = tuple(map(int, rng.choice(10, 2, replace=False))) if varied_palette else roles
            pair = _sample_pair(rng, pair_roles, mode, split,
                                witnesses[index] if split == 'train' else None, used)
            used.add(_signature(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _choice_evidence(info):
    if len(info['route']) < 2:
        return False
    selected = info['route'][0]
    same = [p for p in info['gap'] if info['points'].get(p) == info['key'][1]]
    wrong = [p for p in same if p != selected]
    has_other_color = any(p in info['gap'] and color != info['key'][1]
                          for p, color in info['points'].items())
    beyond = any(p not in set(info['seed']) and
                 abs(p[0]-info['centers'][0][0]) == abs(p[1]-info['centers'][0][1]) and
                 (p[1]-info['centers'][0][1]) == info['slope']*(p[0]-info['centers'][0][0])
                 and color == info['key'][1] for p, color in info['points'].items())
    return bool(wrong and min(same) < selected < max(same) and has_other_color and beyond)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    for split in infos:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            grid, output = _grid(pair['input']), _grid(pair['output'])
            assert grid.shape == output.shape
            signature = _signature(grid)
            assert signature not in seen, "Duplicate input"
            seen.add(signature)
            predicted, info = _decode(grid)
            assert np.array_equal(predicted, output), "Incorrect relay painting"
            assert not np.array_equal(grid, output)
            # No original station, dot, key or divider can be recolored.
            protected = grid != info['bg']
            assert np.array_equal(grid[protected], output[protected])
            assert np.array_equal(grid[-2:], output[-2:])
            allowed = set().union(*(set(cells) for cells, color in info['layers']))
            assert all(tuple(p) in allowed for p in np.argwhere(grid != output))
            # Every chosen dot is original and all launches reverse the slope.
            assert len(set(info['route'])) == len(info['route'])
            assert all(info['points'][p] == color for p, color in zip(info['route'], info['key'][1:]))
            assert all(len(cells) >= 1 for cells, color in info['layers'])
            info['input'], info['output'] = grid, output
            infos[split].append(info)
    train = infos['train']
    assert any(_choice_evidence(a) for a in train), "Need a failed local branch and finite-seed evidence"
    assert any(_recency_cells(a['input'], a) for a in train), "Need an unmarked chronological crossing"
    assert any(not np.array_equal(_first_wins(a['input'], a), a['output']) for a in train)
    assert any(not np.array_equal(_without_key(a['input'], a), a['output']) for a in train)
    assert any(not np.array_equal(_without_turns(a['input'], a), a['output']) for a in train)
    assert set(a['slope']*(-1)**(i+1) for a in train for i in range(len(a['route']))) == {-1, 1}
    test = infos['test']
    assert (min(len(a['route']) for a in test) > max(len(a['route']) for a in train)
            or max(len(a['route']) for a in test) < min(len(a['route']) for a in train)
            or min(len(a['seed']) for a in test) > max(len(a['seed']) for a in train)), "Missing structural split shift"


def measure_puzzle(puzzle):
    records = []
    for split in ('train', 'test'):
        for index, pair in enumerate(puzzle[split]):
            _, a = _decode(_grid(pair['input']))
            records.append(dict(split=split, index=index, shape=tuple(pair['input'].shape),
                                roles=(a['bg'], a['carrier']), size=a['size'],
                                seed_length=len(a['seed']), relays=len(a['route']),
                                dots=len(a['points']), key=a['key'], slope=a['slope'],
                                route=a['route'], unused=len(a['points'])-len(a['route']),
                                repeated_key=len(set(a['key'])) < len(a['key']),
                                branch=_choice_evidence(a),
                                recency=bool(_recency_cells(pair['input'], a))))
    return records


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records = [a for p in puzzles for a in measure_puzzle(p)]
    keys = {tuple(_signature(pair['input']) for s in ('train', 'test') for pair in p[s]) for p in puzzles}
    assert len(keys) == 42
    assert len({a['shape'] for a in records}) >= 45
    assert len({a['roles'] for a in records}) >= 25
    assert {a['size'] for a in records} == {3, 5}
    assert {a['slope'] for a in records} == {-1, 1}
    assert {a['relays'] for a in records} >= {0, 1, 2, 3, 4, 5, 6}
    assert len({a['seed_length'] for a in records}) >= 10
    assert len({a['route'] for a in records}) >= 140
    assert any(a['unused'] == 0 for a in records)
    assert any(a['repeated_key'] for a in records if a['split'] == 'test')
    for p in puzzles:
        validate_generated_puzzle(p)
    # Exercise the emergency path, rather than relying on it never being sampled.
    for mode in ('long_route', 'long_seed', 'short_query'):
        fallback = {'train': [], 'test': []}
        used = set()
        for split, count in (('train', 10), ('test', 4)):
            for i in range(count):
                pair = _guaranteed_pair((0, 8), mode, split,
                                        ('choice', 'order')[i % 2], used)
                used.add(_signature(pair['input']))
                fallback[split].append(pair)
        validate_generated_puzzle(fallback)
    repeated = generate_puzzle(0)
    assert all(np.array_equal(x[k], y[k]) for s in ('train', 'test')
               for x, y in zip(puzzles[0][s], repeated[s]) for k in ('input', 'output'))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            validate_generated_puzzle(generate_puzzle(8000+a*10+b, a, b))
    for a, b in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][-2, 0] = (int(bad['test'][0]['output'][-2, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted divider accepted")
    # Input-only equivariance checks: scene reflection and literal recoloring.
    for pair in puzzles[0]['train']:
        grid, expected = pair['input'], pair['output']
        reflected, reflected_output = grid.copy(), expected.copy()
        reflected[:-2] = grid[:-2, ::-1]
        reflected_output[:-2] = expected[:-2, ::-1]
        assert np.array_equal(solve(reflected), reflected_output)
        remap = np.array([9, 2, 7, 1, 8, 0, 5, 3, 6, 4], dtype=np.int8)
        assert np.array_equal(solve(remap[grid]), remap[expected])
    # Small member outside the sampler's size/span envelope, with a repeated
    # key and an inert original dot lying on an active broadcast.
    small = np.zeros((14, 12), dtype=np.int8)
    small[-2] = 8
    for r, c in ((2, 2), (9, 9)):
        small[r-1:r+2, c-1:c+2] = 8
        small[r, c] = 1
    for p, color in (((5, 5), 2), ((3, 7), 3), ((6, 10), 2), ((1, 9), 4)):
        small[p] = color
    small[-1, 1:8:2] = (1, 2, 3, 2)
    out = solve(small)
    assert out[4, 4] == 1 and out[0, 10] == 2 and out[0, 4] == 3 and out[5, 11] == 2
    assert out[1, 9] == 4 and out[8, 8] == 8
    assert np.array_equal(out[-2:], small[-2:])
    # Larger carrier margins do not change one-cell diagonal thickness.
    wide = np.zeros((23, 25), dtype=np.int8)
    wide[-2] = 8
    for r, c in ((4, 4), (16, 16)):
        wide[r-3:r+4, c-3:c+4] = 8
        wide[r, c] = 1
    wide[-1, 3] = 1
    out = solve(wide)
    assert np.sum(out != wide) == 5 and np.all([out[t, t] == 1 for t in range(8, 13)])
    return dict(seeds=42, pairs=len(records), shapes=len({a['shape'] for a in records}),
                role_assignments=len({a['roles'] for a in records}),
                route_geometries=len({a['route'] for a in records}))
