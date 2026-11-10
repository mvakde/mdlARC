"""Switch stations on latent clockwise pinwheel rails.

Only the minimum-station route is revealed. The input cross specifies the
unseen rail geometry; square switch stations join the lanes they cover.
All solving conventions are input-visible or fixed by this family.
"""
from collections import deque
import heapq
import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= x <= 30 for x in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(mask):
    """Four-neighbor components: corners do not make rail connections."""
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(np.asarray(cells, dtype=np.int16))
    return result


def _square(cells):
    span = cells.max(0) - cells.min(0) + 1
    return int(span[0]) if span[0] == span[1] and len(cells) == int(np.prod(span)) else 0


def _read(grid):
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 4 or np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("four distinct roles with a unique dominant background required")
    bg = int(colors[np.argmax(counts)])
    by_color = {int(c): _components(grid == c) for c in colors if c != bg}
    ep = [c for c, cc in by_color.items() if len(cc) == 2
          and _square(cc[0]) in (3, 5) and all(_square(x) == _square(cc[0]) for x in cc)]
    if len(ep) != 1:
        raise ValueError("exactly two endpoint squares required")
    ep = ep[0]
    size = _square(by_color[ep][0])
    ordinary = [c for c, cc in by_color.items() if c != ep and len(cc) >= 3
                and all(_square(x) == size for x in cc)]
    if len(ordinary) != 1:
        raise ValueError("at least three equal ordinary switch squares required")
    ordinary = ordinary[0]
    ink = next(c for c in by_color if c not in (ep, ordinary))
    mask = grid == ink
    n = int(mask.sum())
    if n % 4 or not 1 <= n // 4 <= 4:
        raise ValueError("cross length must be 1..4")
    length = n // 4
    yy = np.flatnonzero(mask.sum(1) == 2 * length)
    xx = np.flatnonzero(mask.sum(0) == 2 * length)
    if len(yy) != 1 or len(xx) != 1:
        raise ValueError("ambiguous hollow cross")
    center = int(yy[0]), int(xx[0])
    expected = np.zeros(grid.shape, bool)
    y, x = center
    if min(y, x, grid.shape[0]-1-y, grid.shape[1]-1-x) < length:
        raise ValueError("the four input spokes must be fully visible")
    for d in range(1, length + 1):
        expected[y-d, x] = expected[y+d, x] = True
        expected[y, x-d] = expected[y, x+d] = True
    if not np.array_equal(mask, expected) or grid[center] != bg:
        raise ValueError("ink must form exactly one hollow equal-spoke cross")
    stations = _components((grid == ordinary) | (grid == ep))
    if any(_square(cc) != size for cc in stations):
        raise ValueError("stations must be separated solid squares")
    endpoints = [i for i, cc in enumerate(stations) if grid[tuple(cc[0])] == ep]
    return dict(roles=(bg, ink, ordinary, ep), length=length, center=center,
                size=size, stations=stations, endpoints=endpoints, seed_mask=mask)


def _rails(length, shape, center):
    """Geometric continuation on the plane, clipping pixels, never path state.

    A north arm reaches radius R, then at stage k travels east/right/forward
    by 2+4k, R-2, R; the forward direction turns clockwise each stage.
    Rotated copies supply the other three phases. Jog witnesses are retained
    solely to schedule demonstrative samples, never to solve an input.
    """
    radius = max(2, length)
    h, w = shape
    cy, cx = center
    mask = np.zeros(shape, bool)
    jogs = []
    bound = max(cy, cx, h-1-cy, w-1-cx)
    stages = 4 * (bound // 8 + 2)

    def draw(a, b):
        y0, x0 = a[0]+cy, a[1]+cx
        y1, x1 = b[0]+cy, b[1]+cx
        if y0 == y1 and 0 <= y0 < h:
            lo, hi = max(0, min(x0, x1)), min(w-1, max(x0, x1))
            if lo <= hi:
                mask[y0, lo:hi+1] = True
        elif x0 == x1 and 0 <= x0 < w:
            lo, hi = max(0, min(y0, y1)), min(h-1, max(y0, y1))
            if lo <= hi:
                mask[lo:hi+1, x0] = True

    for dy, dx in _DIRS:
        p = radius*dy, radius*dx
        draw((dy, dx), p)
        dy, dx = dx, -dy
        for k in range(stages):
            a = p[0] + (2+4*k)*dy, p[1] + (2+4*k)*dx
            b = a[0] + (radius-2)*dx, a[1] - (radius-2)*dy
            q = b[0] + radius*dy, b[1] + radius*dx
            draw(p, a); draw(a, b); draw(b, q)
            if radius > 2:
                # Both corners and the adjacent forward cells must be exposed.
                jogs.append(tuple((r+cy, c+cx) for r, c in
                                  ((a[0]-dy, a[1]-dx), a, b,
                                   (b[0]+dy, b[1]+dx))))
            p = q
            dy, dx = dx, -dy
    return mask, jogs


def _rail_intervals(length, shape, center):
    """Closed-form membership check, separate from the procedural walker."""
    radius = max(2, length)
    y, x = np.indices(shape)
    y, x = y-center[0], x-center[1]
    mask = np.zeros(shape, bool)
    bound = max(int(abs(y).max()), int(abs(x).max()))
    for phase in range(4):
        mask |= (x == 0) & (y >= -radius) & (y <= -1)
        for q in range(bound // 8 + 3):
            starts = ((-radius-8*q, -8*q), (-2-8*q, radius+8*q+2),
                      (radius+8*q+4, 8*q+4), (8*q+6, -radius-8*q-6))
            for s, ((ay, ax), (dy, dx)) in enumerate(zip(starts, ((0, 1), (1, 0), (0, -1), (-1, 0)))):
                along = (y-ay)*dy + (x-ax)*dx
                across = (y-ay)*dx - (x-ax)*dy
                run = 16*q + 4*s + 2
                mask |= (across == 0) & (along >= 0) & (along <= run)
                mask |= (along == run) & (across >= 0) & (across <= radius-2)
                mask |= (across == radius-2) & (along >= run) & (along <= run+radius)
        y, x = x, -y
    return mask


def _network(mask, stations):
    labels = np.full(mask.shape, -1, dtype=np.int16)
    stripe = np.zeros((_square(stations[0]), _square(stations[0])), bool)
    stripe[::2] = True
    for i, cc in enumerate(stations):
        labels[tuple(cc.T)] = i
        r, c = cc.min(0)
        patch = mask[r:r+stripe.shape[0], c:c+stripe.shape[1]]
        if not (np.array_equal(patch, stripe) or np.array_equal(patch, stripe.T)):
            raise ValueError("each switch must bridge complete parallel alternating lanes")
    edges, cells, dangling = [], [], []
    h, w = mask.shape
    for cc in _components(mask & (labels < 0)):
        touching = set()
        for r, c in cc.tolist():
            for dr, dc in _DIRS:
                rr, zz = r+dr, c+dc
                if 0 <= rr < h and 0 <= zz < w and labels[rr, zz] >= 0:
                    touching.add(int(labels[rr, zz]))
        if len(touching) > 2:
            raise ValueError("a rail fragment branches outside a switch")
        if len(touching) == 2:
            edges.append(tuple(sorted(touching)))
            cells.append(cc)
        else:
            dangling.append(cc)
    adjacency = [[] for _ in stations]
    for e, (a, b) in enumerate(edges):
        adjacency[a].append((b, e))
        adjacency[b].append((a, e))
    reached, todo = {0}, [0]
    while todo:
        for b, e in adjacency[todo.pop()]:
            if b not in reached:
                reached.add(b); todo.append(b)
    if len(reached) != len(stations):
        raise ValueError("all switch stations must belong to one connected network")
    return dict(edges=edges, cells=cells, adjacency=adjacency,
                labels=labels, dangling=dangling)


def _shortest(net, start, target):
    n = len(net['adjacency'])
    distance, ways, prev = [-1]*n, [0]*n, [None]*n
    distance[start], ways[start] = 0, 1
    queue = deque([start])
    while queue:
        a = queue.popleft()
        for b, e in net['adjacency'][a]:
            if distance[b] < 0:
                distance[b] = distance[a]+1
                prev[b] = a, e
                queue.append(b)
            if distance[b] == distance[a]+1:
                ways[b] = min(2, ways[b]+ways[a])
    if ways[target] != 1:
        return None
    nodes, edges = [target], []
    while nodes[-1] != start:
        a, e = prev[nodes[-1]]
        nodes.append(a); edges.append(e)
    return nodes[::-1], edges[::-1]


def _checked_route(net, start, target):
    """Independent exhaustive minimum-hop search, including parallel rails."""
    # Only partial paths as short as the BFS distance can be optimal. Computing
    # that distance by a distance-matrix closure keeps this enumeration bounded.
    n = len(net['adjacency'])
    distances = np.full((n, n), n+1, dtype=int)
    np.fill_diagonal(distances, 0)
    for a, b in net['edges']:
        distances[a, b] = distances[b, a] = 1
    for k in range(n):
        distances = np.minimum(distances, distances[:, k, None]+distances[None, k, :])
    minimum = int(distances[start, target])
    found, stack = [], [(start, [start], [])]
    while stack:
        a, nodes, edges = stack.pop()
        if a == target:
            found.append((nodes, edges))
            if len(found) > 1:
                raise ValueError("minimum-station route is tied")
            continue
        for b, e in net['adjacency'][a]:
            if b not in nodes and len(edges)+1+int(distances[b, target]) == minimum:
                stack.append((b, nodes+[b], edges+[e]))
    if len(found) != 1:
        raise ValueError("endpoints are disconnected")
    return found[0]


def _pixel_cost(net, start, target):
    distance = [float('inf')]*len(net['adjacency'])
    distance[start] = 0
    todo = [(0, start)]
    while todo:
        cost, a = heapq.heappop(todo)
        if cost != distance[a]:
            continue
        if a == target:
            return cost
        for b, e in net['adjacency'][a]:
            next_cost = cost+len(net['cells'][e])
            if next_cost < distance[b]:
                distance[b] = next_cost
                heapq.heappush(todo, (next_cost, b))
    return float('inf')


def _route_mask(shape, net, path):
    mask = np.zeros(shape, bool)
    for e in path[1]:
        mask[tuple(net['cells'][e].T)] = True
    return mask


def _bend_count(cells):
    cc = set(map(tuple, cells.tolist()))
    count = 0
    for r, c in cc:
        ns = [(dr, dc) for dr, dc in _DIRS if (r+dr, c+dc) in cc]
        count += int(len(ns) == 2 and ns[0][0]+ns[1][0] != 0
                     and ns[0][1]+ns[1][1] != 0)
    return count


def _paint(grid, data, net, path):
    out = grid.copy()
    out[_route_mask(grid.shape, net, path)] = data['roles'][1]
    for n in path[0][1:-1]:
        out[tuple(data['stations'][n].T)] = data['roles'][3]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    data = _read(grid)
    mask, _ = _rails(data['length'], grid.shape, data['center'])
    net = _network(mask, data['stations'])
    path = _shortest(net, *data['endpoints'])
    if path is None:
        raise ValueError("minimum-station route must be unique, including rail choices")
    return _paint(grid, data, net, path)


def _candidates(mask, seed_mask, size):
    stripe = np.zeros((size, size), bool)
    stripe[::2] = True
    windows = np.lib.stride_tricks.sliding_window_view(mask, (size, size))
    good = np.all(windows == stripe, axis=(-2, -1)) | np.all(windows == stripe.T, axis=(-2, -1))
    # No switch covers or touches the calibration seed, including its hole.
    forbidden = seed_mask.copy()
    for dr, dc in _DIRS:
        if dr == 1: forbidden[1:] |= seed_mask[:-1]
        if dr == -1: forbidden[:-1] |= seed_mask[1:]
        if dc == 1: forbidden[:, 1:] |= seed_mask[:, :-1]
        if dc == -1: forbidden[:, :-1] |= seed_mask[:, 1:]
    seed_windows = np.lib.stride_tricks.sliding_window_view(forbidden, (size, size))
    good &= ~np.any(seed_windows, axis=(-2, -1))
    return [tuple(map(int, p)) for p in np.argwhere(good)]


def _jog_visible(mask, jogs):
    h, w = mask.shape
    return any(all(0 <= r < h and 0 <= c < w and mask[r, c] for r, c in jog) for jog in jogs)


def _parallel_visible(mask):
    stripe = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=bool)
    windows = np.lib.stride_tricks.sliding_window_view(mask, (3, 3))
    return bool(np.any(np.all(windows == stripe, axis=(-2, -1)))
                or np.any(np.all(windows == stripe.T, axis=(-2, -1))))


def _make_pair(rng, length, size, roles, witness, shift, is_train, corridor):
    # A bounded search over constructive parallel-lane switch placements.
    # The switch graph is reconstructed from the rendered geometry each time.
    for attempt in range(600):
        low = 23 if size == 5 else 15
        if shift == 'compact' and size == 3:
            shape = tuple(int(v) for v in rng.integers(15, 25, 2))
        elif shift == 'large':
            shape = tuple(int(v) for v in rng.integers(max(low, 27), 31, 2))
        else:
            shape = tuple(int(v) for v in rng.integers(low, 31, 2))
        if rng.random() < .1:
            shape = (30, 30)
        padding = 2 if is_train else 0
        center = [int(rng.integers(length+padding, side-length-padding)) for side in shape]
        if not is_train:
            axis = int(rng.integers(2))
            center[axis] = length if rng.random() < .5 else shape[axis]-1-length
        center = tuple(center)
        seed_mask = np.zeros(shape, bool)
        cy, cx = center
        for d in range(1, length+1):
            seed_mask[cy-d, cx] = seed_mask[cy+d, cx] = True
            seed_mask[cy, cx-d] = seed_mask[cy, cx+d] = True
        mask, jogs = _rails(length, shape, center)
        candidates = _candidates(mask, seed_mask, size)
        rng.shuffle(candidates)
        target = int(rng.integers(5, 9 if size == 5 else 14))
        boxes = []
        for p in candidates:
            if all(max(abs(p[0]-q[0]), abs(p[1]-q[1])) >= size+2 for q in boxes):
                boxes.append(p)
            if len(boxes) == target:
                break
        if len(boxes) < 5:
            continue
        grid = np.full(shape, roles[0], np.int8)
        grid[seed_mask] = roles[1]
        for r, c in boxes:
            grid[r:r+size, c:c+size] = roles[2]
        stations = _components(grid == roles[2])
        try:
            net = _network(mask, stations)
        except ValueError:
            continue
        choices = []
        for a in range(len(stations)):
            for b in range(a+1, len(stations)):
                path = _shortest(net, a, b)
                if path is None or len(path[0]) < 3:
                    continue
                if sum(_bend_count(net['cells'][e]) for e in path[1]) < 2:
                    continue
                route = _route_mask(shape, net, path)
                if corridor and not _parallel_visible(route):
                    continue
                if length >= 3 and not _jog_visible(route, jogs):
                    continue
                if witness and sum(len(net['cells'][e]) for e in path[1]) <= _pixel_cost(net, a, b):
                    continue
                choices.append((a, b, path))
        if not choices:
            continue
        a, b, path = choices[int(rng.integers(len(choices)))]
        for n in (a, b):
            grid[tuple(stations[n].T)] = roles[3]
        data = dict(roles=roles, stations=stations)
        return {'input': grid, 'output': _paint(grid, data, net, path)}
    raise RuntimeError("bounded search could not find a teachable switch network")


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain = _count(num_train, 'num_train', 2, 10)
    ntest = _count(num_test, 'num_test', 1, 4)
    if ntrain <= ntest:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    base = tuple(int(c) for c in rng.choice(10, 4, replace=False))
    palette_mode = int(rng.integers(3))  # common, swapping, fresh
    mixed_size = bool(rng.integers(2))
    regime = str(rng.choice(['recombine', 'large', 'compact']))
    lengths = [2, 3]
    lengths += [int(rng.choice([1, 2, 3])) for _ in range(ntrain-2)]
    sizes = [3, 5 if mixed_size else 3]
    sizes += [int(rng.choice([3, 5])) if mixed_size else 3 for _ in range(ntrain-2)]
    witness_index = 1  # L3 both exposes the inward jog and teaches station cost.
    plan = [('train', l, s, i == witness_index, 'recombine')
            for i, (l, s) in enumerate(zip(lengths, sizes))]
    for i in range(ntest):
        # Only >=3 demonstrations admit a new radius; the two compulsory
        # demonstrations expose no jog versus a one-cell jog on drawn routes.
        length = 4 if ntrain >= 3 and regime == 'large' else int(rng.choice(sorted(set(lengths))))
        size = int(rng.choice(sorted(set(sizes))))
        plan.append(('test', length, size, False, regime))
    puzzle, seen = {'train': [], 'test': []}, set()
    for i, (split, length, size, witness, shift) in enumerate(plan):
        if palette_mode == 0:
            roles = base
        elif palette_mode == 1:
            roles = base if i % 2 == 0 else (base[1], base[0], base[3], base[2])
        else:
            roles = tuple(int(c) for c in rng.choice(10, 4, replace=False))
        for retry in range(20):
            pair = _make_pair(rng, length, size, roles, witness, shift, split == 'train', i == 0)
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key); puzzle[split].append(pair)
                break
        else:
            raise RuntimeError("distinct input construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(grid):
    data = _read(grid)
    mask = _rail_intervals(data['length'], grid.shape, data['center'])
    net = _network(mask, data['stations'])
    path = _checked_route(net, *data['endpoints'])
    data.update(net=net, path=path, rails=mask)
    return data


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            grid, out = _grid(pair['input']), _grid(pair['output'])
            assert grid.shape == out.shape
            assert _key(grid) not in seen
            seen.add(_key(grid))
            data = _analyze(grid)
            net, path = data['net'], data['path']
            origins = [tuple(map(int, cc.min(0))) for cc in data['stations']]
            assert all(max(abs(r-rr), abs(c-zz)) >= data['size']+2
                       for i, (r, c) in enumerate(origins) for rr, zz in origins[i+1:])
            station_mask = net['labels'] >= 0
            for r, c in np.argwhere(data['seed_mask']):
                assert not any(0 <= r+dr < grid.shape[0] and 0 <= c+dc < grid.shape[1]
                               and station_mask[r+dr, c+dc] for dr, dc in _DIRS)
            assert len(path[0]) >= 3
            assert sum(_bend_count(net['cells'][e]) for e in path[1]) >= 2
            expected = _paint(grid, data, net, path)
            assert np.array_equal(out, expected), 'incorrect route or off-route change'
            drawn = _route_mask(grid.shape, net, path)
            lit = np.zeros(grid.shape, bool)
            for n in path[0][1:-1]:
                lit[tuple(data['stations'][n].T)] = True
            assert np.all(grid[drawn] == data['roles'][0])
            assert np.array_equal(out == data['roles'][1], data['seed_mask'] | drawn)
            assert np.all(out[lit] == data['roles'][3])
            assert np.array_equal(out[~(lit | drawn)], grid[~(lit | drawn)])
            assert np.all(out[data['seed_mask']] == data['roles'][1])
            assert out[data['center']] == data['roles'][0]
            assert set(map(int, np.unique(out))) <= set(data['roles'])
            cy, cx = data['center']
            length = data['length']
            margin = min(cy-length, cx-length, grid.shape[0]-1-cy-length,
                         grid.shape[1]-1-cx-length)
            assert (margin >= 2) if split == 'train' else (margin == 0)
            # Unchosen switches, dangling tails and every unchosen invisible
            # connector are checked by whole-grid equality, not only counts.
            assert np.array_equal(solve(grid), out)
            if data['length'] >= 3:
                _, jogs = _rails(data['length'], grid.shape, data['center'])
                assert _jog_visible(_route_mask(grid.shape, net, path), jogs)
            analyses[split].append(data)
    assert {2, 3} <= {d['length'] for d in analyses['train']}
    assert any(sum(len(d['net']['cells'][e]) for e in d['path'][1]) >
               _pixel_cost(d['net'], *d['endpoints']) for d in analyses['train']), 'missing station/pixel conflict'
    assert any(_parallel_visible(_route_mask(d['rails'].shape, d['net'], d['path']))
               for d in analyses['train']), 'missing visible one-cell corridor'
    train_sizes = {d['size'] for d in analyses['train']}
    train_lengths = {d['length'] for d in analyses['train']}
    assert all(d['size'] in train_sizes for d in analyses['test'])
    assert all(d['length'] in train_lengths or
               (d['length'] == 4 and len(puzzle['train']) >= 3) for d in analyses['test'])


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    """Bounded geometric, graph, mutation, determinism and 42-seed checks."""
    for length in range(1, 5):
        for shape, center in (((30, 30), (length, length)),
                              ((23, 29), (22-length, 28-length)),
                              ((19, 26), (9, 13))):
            walked, _ = _rails(length, shape, center)
            assert np.array_equal(walked, _rail_intervals(length, shape, center))
            # Re-entry remains valid when calculated on a larger plane window.
            big = _rail_intervals(length, (100, 100), (50, 50))
            top, left = 50-center[0], 50-center[1]
            assert np.array_equal(walked, big[top:top+shape[0], left:left+shape[1]])
            turned_center = (shape[1]-1-center[1], center[0])
            assert np.array_equal(np.rot90(walked), _rail_intervals(length, shape[::-1], turned_center))
    assert np.array_equal(_rails(1, (30, 30), (15, 15))[0], _rails(2, (30, 30), (15, 15))[0])
    # A diamond has two minimum-hop routes. Parallel rail choices also tie.
    for edges in ([(0, 1), (0, 2), (1, 3), (2, 3)], [(0, 1), (0, 1)]):
        n = max(max(e) for e in edges)+1
        net = {'edges': edges, 'adjacency': [[] for _ in range(n)]}
        for i, (a, b) in enumerate(edges):
            net['adjacency'][a].append((b, i)); net['adjacency'][b].append((a, i))
        assert _shortest(net, 0, n-1) is None
        try:
            _checked_route(net, 0, n-1)
        except ValueError:
            pass
        else:
            raise AssertionError('tied routes accepted')
    shapes, counts, lengths, sizes, selected_counts, loop_ranks, palettes = set(), set(), set(), set(), set(), set(), set()
    signatures = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        signatures.add(tuple(_key(pair[k]) for s in ('train', 'test') for pair in p[s] for k in ('input', 'output')))
        if seed < 2:
            assert _equal(p, generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in p[split]:
                d = _analyze(pair['input'])
                shapes.add(pair['input'].shape); counts.add(len(d['stations']))
                lengths.add(d['length']); sizes.add(d['size']); palettes.add(d['roles'])
                selected_counts.add(len(d['path'][0])-2)
                loop_ranks.add(len(d['net']['edges'])-len(d['stations'])+1)
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(counts) >= 5 and len(palettes) >= 30
    assert sizes == {3, 5} and lengths == {1, 2, 3, 4}
    assert len(selected_counts) >= 3 and max(loop_ranks) >= 3
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(1000+nt+ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(1, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    corrupted = generate_puzzle(42, 2, 1)
    corrupted['test'][0]['output'][0, 0] = (int(corrupted['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(corrupted)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('altered output accepted')
    return {'seeds': 42, 'shapes': len(shapes), 'station_counts': sorted(counts),
            'lengths': sorted(lengths), 'station_sizes': sorted(sizes),
            'selected_station_counts': sorted(selected_counts),
            'loop_ranks': sorted(loop_ranks), 'palettes': len(palettes)}
