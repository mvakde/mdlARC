"""Breakable necklace: route-derived bonds, color fracture, rigid gravity.

Only NumPy and the Python standard library are used. Solving reads the pixels;
no sampled route, random state, examples, or stored answer is consulted.
"""
from dataclasses import dataclass
import itertools

import numpy as np


@dataclass(frozen=True)
class Body:
    pixels: tuple[tuple[int, int, int], ...]
    beads: tuple[int, ...]


def _integer(value, lo, hi, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _components(mask):
    seen = set()
    result = []
    h, w = mask.shape
    for rr, cc in zip(*np.nonzero(mask)):
        start = int(rr), int(cc)
        if start in seen:
            continue
        todo, component = [start], []
        seen.add(start)
        while todo:
            r, c = todo.pop()
            component.append((r, c))
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if (0 <= p[0] < h and 0 <= p[1] < w
                        and mask[p] and p not in seen):
                    seen.add(p)
                    todo.append(p)
        result.append(component)
    return result


def _band(a, b, size, patch):
    """Coordinates of the gap rectangle, never the station centers/margins."""
    offset = (size-patch)//2
    if a[0] == b[0] and abs(a[1]-b[1]) > size:
        return tuple((r, c) for r in range(a[0]+offset, a[0]+offset+patch)
                     for c in range(min(a[1], b[1])+size, max(a[1], b[1])))
    if a[1] == b[1] and abs(a[0]-b[0]) > size:
        return tuple((r, c) for r in range(min(a[0], b[0])+size, max(a[0], b[0]))
                     for c in range(a[1]+offset, a[1]+offset+patch))
    return ()


def _routes(graph, labels, key, limit=2):
    candidates = [[i for i, label in enumerate(labels) if label == color]
                  for color in key]
    found = []

    def visit(path, used):
        if len(found) >= limit:
            return
        if len(path) == len(key):
            found.append(tuple(path))
            return
        for i in candidates[len(path)]:
            if i not in used and (not path or i in graph[path[-1]]):
                visit(path+[i], used | {i})

    visit([], set())
    return found


def _decode(grid):
    grid = _grid(grid)
    h, w = grid.shape
    assert h >= 8 and w >= 7
    bg = int(grid[1, 0])
    assert np.all(grid[1] == bg) and np.all(grid[-1] == bg)
    carrier = int(grid[-3, 0])
    assert carrier != bg and np.all(grid[-3] == carrier)
    commands = tuple(int(c) for c in grid[0] if c != bg)
    assert 1 <= len(commands) <= 3 and len(set(commands)) == len(commands)
    assert carrier not in commands
    xs = np.flatnonzero(grid[-2] != bg)
    assert 3 <= len(xs) <= 12 and np.all(np.diff(xs) == 2)
    assert xs[0] >= 1 and xs[-1] < w-1
    key = tuple(int(grid[-2, x]) for x in xs)
    assert carrier not in key and len(set(key)) >= 2
    chamber = grid[2:-3]
    assert np.all(chamber[:, 0] == bg) and np.all(chamber[:, -1] == bg)
    stations, signature = [], None
    for component in _components(chamber != bg):
        ys, cs = zip(*component)
        y, x = min(ys)+2, min(cs)
        size = max(ys)-min(ys)+1
        assert 3 <= size <= 6 and max(cs)-min(cs)+1 == size
        assert len(component) == size*size
        block = grid[y:y+size, x:x+size]
        assert int(block[0, 0]) == carrier
        py, px = np.nonzero(block != carrier)
        assert len(py)
        off = int(py.min())
        patch = int(py.max())-off+1
        assert off > 0 and patch+2*off == size
        assert int(px.min()) == off and int(px.max()) == off+patch-1
        assert len(py) == patch*patch
        label = int(block[off, off])
        assert label not in (bg, carrier)
        assert np.all(block[off:off+patch, off:off+patch] == label)
        assert signature is None or signature == (size, patch)
        signature = size, patch
        stations.append((y, x, label))
    assert 3 <= len(stations) <= 16 and signature is not None
    size, patch = signature
    stations.sort()
    graph, bands = [[] for _ in stations], {}
    for i, a in enumerate(stations):
        for j, b in enumerate(stations):
            if i == j:
                continue
            band = _band(a, b, size, patch)
            if band and all(grid[p] == bg for p in band):
                graph[i].append(j)
                bands[i, j] = band
    routes = _routes(graph, [s[2] for s in stations], key)
    assert len(routes) == 1, "key must have one complete distinct-bead route"
    route = routes[0]
    occupied = set()
    for i, j in zip(route, route[1:]):
        cells = set(bands[i, j])
        assert not cells & occupied, "links must not overlap or cross"
        occupied |= cells
    return dict(grid=grid, background=bg, carrier=carrier, size=size, patch=patch,
                stations=stations, key=key, commands=commands, route=route,
                graph=graph, bands=bands, floor=h-3)


def _assemble(info, commands=None, *, target=False, loose=False):
    stations, size, patch = info['stations'], info['size'], info['patch']
    cut = set(info['commands'] if commands is None else commands)
    parent = list(range(len(stations)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    links = []
    if not loose:
        for i, j in zip(info['route'], info['route'][1:]):
            color = stations[j if target else i][2]
            if color not in cut:
                parent[root(j)] = root(i)
                links.append((i, j, color))
    pixels, beads = {}, {}
    off = (size-patch)//2
    for i, (y, x, label) in enumerate(stations):
        k = root(i)
        beads.setdefault(k, []).append(i)
        pixels.setdefault(k, []).extend(
            (r, c, label if y+off <= r < y+off+patch and x+off <= c < x+off+patch
             else info['carrier'])
            for r in range(y, y+size) for c in range(x, x+size))
    for i, j, color in links:
        pixels[root(i)].extend((r, c, color) for r, c in info['bands'][i, j])
    return [Body(tuple(pixels[k]), tuple(beads[k])) for k in sorted(pixels)]


def _settle(bodies, floor):
    cells = [frozenset((r, c) for r, c, color in body.pixels) for body in bodies]
    offsets = [0]*len(bodies)
    for _ in range(floor+1):
        current = [frozenset((r+offsets[i], c) for r, c in points)
                   for i, points in enumerate(cells)]
        targets = [frozenset((r+1, c) for r, c in points) for points in current]
        moving = {i for i, points in enumerate(targets) if max(r for r, c in points) < floor}
        while moving:
            stationary = set(range(len(bodies)))-moving
            blocked = {i for i in moving if any(targets[i] & current[j] for j in stationary)}
            if not blocked:
                break
            moving -= blocked
        if not moving:
            return tuple(offsets)
        for i in moving:
            offsets[i] += 1
    raise AssertionError('monotone gravity did not converge')


def _render(info, bodies, offsets):
    grid = np.full(info['grid'].shape, info['background'], dtype=np.int8)
    grid[:2] = info['grid'][:2]
    grid[-3:] = info['grid'][-3:]
    occupied = set()
    for body, dy in zip(bodies, offsets):
        for y, x, color in body.pixels:
            p = y+dy, x
            assert 2 <= p[0] < info['floor'] and 0 < p[1] < grid.shape[1]-1
            assert p not in occupied
            occupied.add(p)
            grid[p] = color
    return grid


def _prediction(info, commands=None, *, target=False, loose=False, gravity=True):
    bodies = _assemble(info, commands, target=target, loose=loose)
    offsets = _settle(bodies, info['floor']) if gravity else (0,)*len(bodies)
    return _render(info, bodies, offsets), bodies, offsets


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _prediction(_decode(input_grid))[0]


def _landings(bodies, offsets, info):
    return tuple(next(info['stations'][i][0]+d for b, d in zip(bodies, offsets) if i in b.beads)
                 for i in range(len(info['stations'])))


def _axis(rng, count, size, limit):
    gaps = [1]*(count-1)
    budget = int(rng.integers(0, min(10, limit-count*size-(count-1))+1))
    for j in rng.permutation(len(gaps)):
        extra = int(rng.integers(0, min(5, budget)+1))
        gaps[int(j)] += extra
        budget -= extra
    positions = [0]
    for gap in gaps:
        positions.append(positions[-1]+size+gap)
    return positions


def _visibility(cells):
    graph = [[] for _ in cells]
    for i, a in enumerate(cells):
        for j, b in enumerate(cells):
            if i == j:
                continue
            if a[0] == b[0]:
                axis = 1
            elif a[1] == b[1]:
                axis = 0
            else:
                continue
            lo, hi = sorted((a[axis], b[axis]))
            if not any(c[1-axis] == a[1-axis] and lo < c[axis] < hi for c in cells):
                graph[i].append(j)
    return graph


def _path(rng, cells, graph, length, bent):
    budget = 2500

    def cross(a, b, c, d):
        if a[0] == b[0] and c[1] == d[1]:
            return (min(a[1], b[1]) < c[1] < max(a[1], b[1])
                    and min(c[0], d[0]) < a[0] < max(c[0], d[0]))
        if a[1] == b[1] and c[0] == d[0]:
            return cross(c, d, a, b)
        return False

    def visit(path):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            axes = {cells[i][0] == cells[j][0] for i, j in zip(path, path[1:])}
            return path if not bent or len(axes) == 2 else None
        for nxt in rng.permutation(graph[path[-1]]):
            nxt = int(nxt)
            if nxt in path:
                continue
            a, b = cells[path[-1]], cells[nxt]
            if any(cross(a, b, cells[i], cells[j]) for i, j in zip(path, path[1:])):
                continue
            found = visit(path+[nxt])
            if found is not None:
                return found
        return None

    for i in rng.permutation(len(cells)):
        found = visit([int(i)])
        if found is not None:
            return found
        if budget < 0:
            break
    return None


def _labels(rng, graph, path, colors, anchor):
    unused = [i for i in range(len(graph)) if i not in path]
    for attempt in range(90):
        count = len(path) if len(path) <= len(colors) and rng.random() < .35 else int(
            rng.integers(min(3, len(path)), min(len(path), len(colors))+1))
        pool = [int(c) for c in rng.permutation(colors)[:count]]
        key = pool+[int(rng.choice(pool)) for _ in range(len(path)-count)]
        key = [int(c) for c in rng.permutation(key)]
        if anchor and len(set(key[:-1])) < 3:
            continue
        labels = [0]*len(graph)
        for i, color in zip(path, key):
            labels[i] = color
        absent = [c for c in colors if c not in key]
        for i in unused:
            labels[i] = int(rng.choice(absent if absent and rng.random() < .3 else key))
        if anchor:
            labels[unused[0]] = key[0]
        solutions = _routes(graph, labels, key)
        if len(solutions) == 1 and solutions[0] == tuple(path):
            return labels, key
    return None


def _draw(rng, cells, labels, key, size, patch, palette):
    bg, carrier = palette[:2]
    ys = _axis(rng, max(r for r, c in cells)+1, size, 25)
    xs = _axis(rng, max(c for r, c in cells)+1, size, 28)
    bh = max(ys[r] for r, c in cells)+size
    bw = max(xs[c] for r, c in cells)+size
    h = int(rng.integers(bh+5, 31))
    minw = max(bw+2, 2*len(key)+1)
    w = int(rng.integers(minw, 31))
    top = int(rng.integers(2, h-3-bh+1))
    left = int(rng.integers(1, w-bw))
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[-3] = carrier
    off = (size-patch)//2
    for (r, c), label in zip(cells, labels):
        y, x = top+ys[r], left+xs[c]
        grid[y:y+size, x:x+size] = carrier
        grid[y+off:y+off+patch, x+off:x+off+patch] = label
    keyx = int(rng.integers(1, w-(2*len(key)-1)))
    grid[-2, keyx:keyx+2*len(key):2] = key
    grid[0, int(rng.integers(w))] = palette[-1]
    return grid


def _scaffold(rng, size, patch, length, anchor=False, fallback=False):
    maxr, maxc = 26//(size+1), 29//(size+1)
    if fallback and anchor:
        cells = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2)]
        path = [0, 1, 3, 4]
        palette = [int(c) for c in rng.permutation(10)]
        labels = [palette[2], palette[3], palette[2], palette[4], palette[5]]
        return _draw(rng, cells, labels, [labels[i] for i in path], size, patch, palette)
    n = length+int(rng.integers(1 if anchor else 0, 4))
    n = min(n, maxr*maxc, 16)
    options = [(r, c) for r in range(1, maxr+1) for c in range(1, maxc+1)
               if n <= r*c <= n+7 and (not anchor or (r >= 2 and c >= 2))]
    if not options:
        return None
    r, c = options[int(rng.integers(len(options)))]
    universe = [(y, x) for y in range(r) for x in range(c)]
    if fallback:
        # A long snake supplies a fitting noncrossing route; label uniqueness
        # is still checked by the same general decoder, never assumed.
        snake = [(y, x) for y in range(r) for x in (range(c) if y % 2 == 0 else range(c-1, -1, -1))]
        chosen = snake[:length]
        rest = [p for p in universe if p not in chosen]
        cells = sorted(chosen+[rest[int(i)] for i in rng.permutation(len(rest))[:n-length]])
        path = [cells.index(p) for p in chosen]
    else:
        cells = sorted(universe[int(i)] for i in rng.choice(len(universe), n, replace=False))
        path = _path(rng, cells, _visibility(cells), length, anchor)
    if path is None:
        return None
    graph = _visibility(cells)
    palette = [int(color) for color in rng.permutation(10)]
    colors = palette[2:-1] if anchor else palette[2:]
    assigned = _labels(rng, graph, path, colors, anchor)
    if assigned is None:
        return None
    labels, key = assigned
    return _draw(rng, cells, labels, key, size, patch, palette)


def _commands(rng, grid, colors):
    result = grid.copy()
    bg = int(grid[1, 0])
    result[0] = bg
    positions = rng.choice(grid.shape[1], len(colors), replace=False)
    for x, color in zip(positions, rng.permutation(colors)):
        result[0, int(x)] = int(color)
    return result


def _pair(grid):
    return dict(input=grid, output=solve(grid))


def _anchor_pairs(rng, size, patch, length):
    for attempt in range(85):
        grid = _scaffold(rng, size, patch, length, anchor=True, fallback=attempt >= 65)
        if grid is None:
            continue
        info = _decode(grid)
        out, bodies, offsets = _prediction(info, commands=())
        _, loose, loose_offsets = _prediction(info, loose=True)
        if (not any(offsets) or _landings(bodies, offsets, info)
                == _landings(loose, loose_offsets, info)):
            continue
        present = sorted(set(info['key'][:-1]))
        choices = list(itertools.combinations(present, 2))
        for k in rng.permutation(len(choices)):
            cmd = choices[int(k)]
            if set(present) <= set(cmd):
                continue
            cutout, cutbodies, cutoffsets = _prediction(info, commands=cmd)
            if _landings(bodies, offsets, info) == _landings(cutbodies, cutoffsets, info):
                continue
            alt = _prediction(info, commands=cmd, target=True)[0]
            if np.array_equal(cutout, alt) or not any(cutoffsets):
                continue
            first = _pair(grid)
            second = _pair(_commands(rng, grid, cmd))
            assert np.array_equal(first['output'], out)
            return first, second
    raise AssertionError('bounded anchor construction failed')


def _sample_pair(rng, length, size, patch, forced=None):
    for attempt in range(70):
        grid = _scaffold(rng, size, patch, length, fallback=attempt >= 45)
        if grid is None:
            continue
        info = _decode(grid)
        outgoing = sorted(set(info['key'][:-1]))
        absent = [c for c in range(10) if c not in (info['background'], info['carrier']) and c not in outgoing]
        pool = outgoing+absent
        count = int(rng.integers(1, min(3, len(pool))+1))
        if forced == 'cut':
            colors = [int(c) for c in rng.choice(outgoing, min(count, len(outgoing)), replace=False)]
        elif rng.random() < .13 and absent:
            colors = [int(rng.choice(absent))]
        else:
            colors = [int(c) for c in rng.choice(pool, count, replace=False)]
        return _pair(_commands(rng, grid, colors))
    raise AssertionError('bounded scaffold construction failed')


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nt = _integer(num_train, 2, 10, 'num_train')
    nq = _integer(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    size = int(rng.choice((5, 6)))
    widths = (1, 3) if size == 5 else (2, 4)
    patch = int(rng.choice(widths))
    length = int(rng.integers(4, 8))
    train = list(_anchor_pairs(rng, size, patch, length))
    seen = {_signature(p['input']) for p in train}
    for index in range(2, nt):
        s = size if index == 2 else int(rng.choice((3, 4, 5, 6)))
        p = widths[1-widths.index(patch)] if index == 2 else int(rng.choice(tuple(range(1 if s % 2 else 2, s-1, 2))))
        for _ in range(20):
            pair = _sample_pair(rng, int(rng.integers(3, length+1)), s, p)
            if _signature(pair['input']) not in seen:
                train.append(pair)
                seen.add(_signature(pair['input']))
                break
        else:
            raise AssertionError('could not generate distinct training input')
    max_train = max(len(_decode(pair['input'])['key']) for pair in train)
    shown_widths = sorted({_decode(pair['input'])['patch'] for pair in train})
    tests = []
    for _ in range(nq):
        for _ in range(20):
            p = int(rng.choice(shown_widths))
            sizes = [s for s in (3, 4, 5, 6) if 0 < p < s and (s-p) % 2 == 0]
            s = int(rng.choice(sizes))
            capacity = (26//(s+1))*(29//(s+1))
            test_length = int(rng.integers(max_train+1, min(11, max_train+4, capacity)+1))
            pair = _sample_pair(rng, test_length, s, p, forced='cut')
            if _signature(pair['input']) not in seen:
                tests.append(pair)
                seen.add(_signature(pair['input']))
                break
        else:
            raise AssertionError('could not generate distinct test input')
    rng.shuffle(train)
    rng.shuffle(tests)
    puzzle = dict(train=train, test=tests)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    out, bodies, offsets = _prediction(info)
    intact, intact_bodies, intact_d = _prediction(info, commands=())
    loose, loose_bodies, loose_d = _prediction(info, loose=True)
    still = _prediction(info, gravity=False)[0]
    cut_edges = [int(info['stations'][i][2] in info['commands'])
                 for i, j in zip(info['route'], info['route'][1:])]
    used = set(info['route'])
    duplicate_unused = any(s[2] in info['key'] for i, s in enumerate(info['stations']) if i not in used)
    has_branch = any(sum(s[2] == color and j not in info['route'][:step]
                         and (not step or j in info['graph'][info['route'][step-1]])
                         for j, s in enumerate(info['stations'])) > 1
                     for step, color in enumerate(info['key']))
    landings = _landings(bodies, offsets, info)
    cuts_mechanical = landings != _landings(intact_bodies, intact_d, info)
    route_mechanical = landings != _landings(loose_bodies, loose_d, info)
    target_difference = not np.array_equal(out, _prediction(info, target=True)[0])
    axes = [info['stations'][i][0] == info['stations'][j][0]
            for i, j in zip(info['route'], info['route'][1:])]
    supported = 0
    final_cells = []
    for body, dy in zip(bodies, offsets):
        final_cells.append({(r+dy, c) for r, c, color in body.pixels})
        if max(r+dy for r, c, color in body.pixels) < info['floor']-1:
            supported += 1
    support_counts = [sum(bool({(r+1, c) for r, c in cells} & other)
                          for j, other in enumerate(final_cells) if j != i)
                      for i, cells in enumerate(final_cells)]
    gap_lengths = [len(info['bands'][i, j])//info['patch']
                   for i, j in zip(info['route'], info['route'][1:])]
    return dict(length=len(info['key']), stations=len(info['stations']),
                size=info['size'], patch=info['patch'], bodies=len(bodies),
                cuts=sum(cut_edges), kept=len(cut_edges)-sum(cut_edges),
                commands=len(info['commands']), duplicate_unused=duplicate_unused,
                branch=has_branch, cuts_mechanical=cuts_mechanical,
                route_mechanical=route_mechanical, target_difference=target_difference,
                gravity_difference=not np.array_equal(out, still),
                moved=sum(d > 0 for d in offsets), distances=tuple(sorted(set(offsets)-{0})),
                supported=supported, support_edges=sum(support_counts),
                max_supports=max(support_counts), fragment_sizes=tuple(sorted(len(b.beads) for b in bodies)),
                turns=sum(a != b for a, b in zip(axes, axes[1:])),
                gaps=tuple(gap_lengths), repeated=len(set(info['key'])) < len(info['key']),
                adjacent_equal=any(a == b for a, b in zip(info['key'], info['key'][1:])))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    infos, features, seen = {}, {}, set()
    for split in ('train', 'test'):
        infos[split], features[split] = [], []
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            signature = _signature(inp)
            assert signature not in seen
            seen.add(signature)
            info = _decode(inp)
            predicted, bodies, offsets = _prediction(info)
            assert np.array_equal(predicted, out), 'incorrect rigid-body result'
            assert np.array_equal(inp[:2], out[:2]) and np.array_equal(inp[-3:], out[-3:])
            assert _settle([Body(tuple((r+d, c, color) for r, c, color in body.pixels), body.beads)
                            for body, d in zip(bodies, offsets)], info['floor']) == (0,)*len(bodies)
            # Count conservation includes every bead pixel and only uncut links.
            expected_count = np.zeros(10, dtype=int)
            for body in bodies:
                assert len(body.pixels) == len({(r, c) for r, c, color in body.pixels})
                for r, c, color in body.pixels:
                    expected_count[color] += 1
            actual = np.bincount(out[2:-3].ravel(), minlength=10)
            actual[info['background']] = 0
            assert np.array_equal(actual, expected_count)
            infos[split].append(info)
            features[split].append(_features(info))
    train, test = features['train'], features['test']
    assert min(f['length'] for f in test) > max(f['length'] for f in train)
    assert {f['patch'] for f in test} <= {f['patch'] for f in train}
    assert any(f['cuts'] == 0 and f['route_mechanical'] and f['gravity_difference'] for f in train)
    assert any(f['commands'] == 2 and f['cuts'] >= 2 and f['kept'] > 0
               and f['cuts_mechanical'] and f['target_difference'] and f['moved'] for f in train)
    assert any(f['duplicate_unused'] and f['branch'] for f in train)
    assert any(np.array_equal(a['grid'][2:], b['grid'][2:])
               and set(a['commands']) != set(b['commands'])
               for a, b in itertools.combinations(infos['train'], 2)), 'counterfactual scaffold missing'
    if len(train) >= 3:
        assert any(a['size'] == b['size'] and a['patch'] != b['patch']
                   for a, b in itertools.combinations(train, 2))


def _fingerprint(puzzle):
    return tuple((_signature(pair['input']), _signature(pair['output']))
                 for split in ('train', 'test') for pair in puzzle[split])


def _mechanics_test():
    # Manually derived small scene. The loose lower-left bead stops the
    # released upper-left bead; the surviving vertical bond holds B higher.
    grid = np.zeros((16, 15), dtype=np.int8)
    grid[13] = 8
    grid[14, [1, 3, 5, 7]] = [2, 3, 4, 5]
    stations = [(2, 1, 2), (2, 6, 3), (7, 1, 2), (7, 6, 4), (7, 11, 5)]

    def square(canvas, y, x, color):
        canvas[y:y+3, x:x+3] = 8
        canvas[y+1, x+1] = color

    for y, x, color in stations:
        square(grid, y, x, color)
    grid[0, 0] = 9
    expected = np.zeros_like(grid)
    expected[:2] = grid[:2]
    expected[-3:] = grid[-3:]
    for y, x, color in stations:
        square(expected, y+3, x, color)
    expected[6, 4:6] = 2
    expected[8:10, 7] = 3
    expected[11, 9:11] = 4
    assert np.array_equal(solve(grid), expected)
    cut = grid.copy()
    cut[0] = 0
    cut[0, [0, 3]] = [2, 4]
    expected = np.zeros_like(grid)
    expected[:2] = cut[:2]
    expected[-3:] = cut[-3:]
    for i, (y, x, color) in enumerate(stations):
        square(expected, y+(5 if i == 0 else 3), x, color)
    expected[8:10, 7] = 3
    assert np.array_equal(solve(cut), expected)
    # A separately specified footprint diagnostic checks a bent body whose
    # distant arm supports its overhang, with another object moving with it.
    upper = Body(tuple((3, c, 1) for c in range(1, 6))+((4, 5, 1),), (0,))
    lower = Body(tuple((r, c, 2) for r in range(7, 10) for c in (1, 2)), (1,))
    cap = Body(tuple((r, c, 3) for r in (1, 2) for c in (4, 5)), (2,))
    assert _settle([upper, lower, cap], 12) == (5, 2, 5)


def self_test() -> None:
    _mechanics_test()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    records, shapes, layouts, palettes = [], set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _decode(pair['input'])
                f = _features(info)
                records.append(f)
                shapes.add(pair['input'].shape)
                ys = [s[0] for s in info['stations']]
                xs = [s[1] for s in info['stations']]
                layouts.add((info['size'], info['patch'],
                             tuple((y-min(ys), x-min(xs)) for y, x, label in info['stations']),
                             tuple(info['route']), tuple(s[2] in info['commands'] for s in info['stations'])))
                palettes.add((info['background'], info['carrier']))
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
    assert len({_fingerprint(puzzle) for puzzle in puzzles}) == 42
    assert len(shapes) >= 30 and len(layouts) >= 120 and len(palettes) >= 35
    assert {f['size'] for f in records} == {3, 4, 5, 6}
    assert {f['patch'] for f in records} == {1, 2, 3, 4}
    assert len({f['length'] for f in records}) >= 7
    assert len({f['stations'] for f in records}) >= 8
    assert len({f['bodies'] for f in records}) >= 6
    assert len({d for f in records for d in f['distances']}) >= 8
    assert any(len(f['distances']) >= 2 for f in records)
    assert any(f['supported'] >= 2 for f in records)
    assert any(f['turns'] >= 4 for f in records)
    assert any(len(set(f['gaps'])) >= 3 for f in records)
    assert any(f['repeated'] for f in records) and any(f['adjacent_equal'] for f in records)
    assert {f['commands'] for f in records} == {1, 2, 3}
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    # Color equality is the only color semantics; all ten role values can be
    # permuted without changing route topology, bonds or fall distances.
    color_rng = np.random.default_rng(98765)
    remap = color_rng.permutation(10).astype(np.int8)
    for pair in puzzles[0]['train']+puzzles[0]['test']:
        assert np.array_equal(solve(remap[pair['input']]), remap[pair['output']])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(7000+10*nt+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (None, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Output corruption must be detected, including preserved-band corruption.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][-3, 0] = (int(bad['test'][0]['output'][-3, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
