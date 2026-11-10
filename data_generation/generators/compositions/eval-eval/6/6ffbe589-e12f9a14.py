"""Counter-steered relay cascades. Standard library and NumPy only.

The solver parses the input, builds the visible directed port graph, and takes
its reachability closure. Generation constructs geometry, never solver state.
"""
from collections import deque
import numpy as np


PORTS = ((0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 3),
         (2, 0), (2, 3), (3, 0), (3, 1), (3, 2), (3, 3))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = r + dr, c + dc
                    if q in remaining:
                        remaining.remove(q)
                        stack.append(q)
        result.append(cells)
    return result


def _parse(a):
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[counts.argmax()])
    _require(int(counts.max()) > a.size / 2, "background must occupy over half the input")
    relays, bars = [], {}
    for cells in _components(a != bg):
        coords = np.array(cells)
        r, c = map(int, coords.min(axis=0))
        hh, ww = map(int, coords.max(axis=0) - coords.min(axis=0) + 1)
        if (hh, ww) == (4, 4):
            tile = a[r:r+4, c:c+4].copy()
            rim = {int(tile[p]) for p in PORTS if tile[p] != bg}
            _require(len(rim) == 1, "relay rim has mixed colors")
            color = rim.pop()
            core = int(tile[1, 1])
            _require(core != bg and bool(np.all(tile[1:3, 1:3] == core)), "invalid relay core")
            gaps = [p for p in PORTS if tile[p] == bg]
            _require(1 <= len(gaps) <= 4, "relay needs one through four openings")
            _require(len(cells) == 16 - len(gaps), "unaccounted tile cells")
            relays.append(dict(pos=(r, c), color=color, core=core, tile=tile, gaps=gaps))
        else:
            palette = {int(a[p]) for p in cells}
            _require(len(palette) == 1 and (hh == 1 or ww == 1), "non-relay object must be a straight bar")
            color = palette.pop()
            _require(1 <= len(cells) <= 9 and len(cells) == hh * ww, "invalid counter length")
            _require(color not in bars, "at most one counter per relay color")
            bars[color] = dict(length=len(cells), cells=tuple(sorted(cells)),
                               orientation='single' if len(cells) == 1 else ('horizontal' if hh == 1 else 'vertical'))
    _require(2 <= len(relays) <= 7, "family has two through seven relays")
    colors = {t['color'] for t in relays}
    _require(set(bars) <= colors, "counter does not match a relay color")
    roots = [i for i, t in enumerate(relays) if t['core'] != t['color']]
    _require(len(roots) == 1, "exactly one relay has a contrasting starter core")
    root = roots[0]
    spark = relays[root]['core']
    _require(spark not in colors and spark != bg and int(np.sum(a == spark)) == 4,
             "starter color is reserved for its four core cells")
    owner = np.full(a.shape, -1, dtype=int)
    for i, t in enumerate(relays):
        r, c = t['pos']
        _require(bool(np.all(owner[r:r+4, c:c+4] == -1)), "relay footprints overlap")
        owner[r:r+4, c:c+4] = i
    return dict(background=bg, spark=spark, root=root, relays=relays, bars=bars, owner=owner)


def _direction(port):
    r, c = port
    return (-1 if r == 0 else 1 if r == 3 else 0,
            -1 if c == 0 else 1 if c == 3 else 0)


def _route(pos, port, owner):
    """Port itself is drawn; the first entered receiver footprint is not."""
    r, c = pos[0] + port[0], pos[1] + port[1]
    dr, dc = _direction(port)
    path = [(r, c)]
    h, w = owner.shape
    while True:
        r, c = r + dr, c + dc
        if not (0 <= r < h and 0 <= c < w):
            return path, None
        if owner[r, c] >= 0:
            return path, int(owner[r, c])
        path.append((r, c))


def _evaluate(a, *, variant='normal'):
    info = _parse(a)
    bg, relays = info['background'], info['relays']
    counts = {c: b['length'] for c, b in info['bars'].items()}
    common = next(iter(counts.values()), 0)
    tiles, routes = [], []
    for t in relays:
        k = counts.get(t['color'], 0)
        if variant == 'ignore_counts':
            k = 0
        elif variant == 'counterclockwise':
            k = -k
        elif variant == 'binary_count':
            k = int(k > 0)
        elif variant == 'clamp_count':
            k = min(k, 3)
        elif variant == 'shared_count':
            k = common
        elif variant == 'absent_one' and t['color'] not in counts:
            k = 1
        elif variant == 'bar_direction' and t['color'] in counts:
            k = 3 if info['bars'][t['color']]['orientation'] == 'vertical' else 1
        tile = np.rot90(t['tile'], -k).copy()
        tiles.append(tile)
        routes.append([_route(t['pos'], p, info['owner']) for p in PORTS if tile[p] == bg])
    # A graph closure is insensitive to signal travel time and processing order.
    active, depth = {info['root']}, {info['root']: 0}
    queue = deque([info['root']])
    if variant == 'all_active':
        active = set(range(len(relays)))
        depth = {i: 0 for i in active}
        queue = deque(sorted(active))
    while queue:
        i = queue.popleft()
        if variant == 'root_only':
            continue
        for _, j in routes[i]:
            if j is not None and j not in active:
                active.add(j)
                depth[j] = depth[i] + 1
                queue.append(j)
    out = a.copy()
    for b in info['bars'].values():
        for p in b['cells']:
            out[p] = bg
    for i in active:
        r, c = relays[i]['pos']
        out[r:r+4, c:c+4] = tiles[i]
    painted, segments = {}, {}
    edges, boundary, headings = [], 0, set()
    for i in sorted(active):
        color = relays[i]['color']
        for path, j in routes[i]:
            if j is None:
                boundary += 1
            else:
                edges.append((i, j))
            headings.add(_direction((path[0][0]-relays[i]['pos'][0], path[0][1]-relays[i]['pos'][1])))
            for p in path:
                _require(p not in painted or painted[p] == color, "different-colored rays overlap")
                painted[p] = color
                out[p] = color
            for p, q in zip(path, path[1:]):
                if p[0] != q[0] and p[1] != q[1]:
                    midpoint = p[0] + q[0], p[1] + q[1]
                    slope = (q[0]-p[0]) * (q[1]-p[1])
                    _require((midpoint, -slope) not in segments, "diagonal rays cross between cells")
                    segments[midpoint, slope] = color
    info.update(active=active, depths=depth, edges=edges, boundary_rays=boundary,
                headings=headings, routes=routes, rotated_tiles=tiles, painted=painted)
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve with fixed visible conventions, without any generation state."""
    return _evaluate(_grid(input_grid))[0]


def _different(a, correct, variant):
    try:
        predicted, _ = _evaluate(a, variant=variant)
        return not np.array_equal(predicted, correct)
    except ValueError:
        # A counterfactual can create an inadmissible crossing. It then cannot
        # account for this demonstration under the same collision-free grammar.
        return True


def _topology_tags(info):
    """Features reconstructed from visible, rotated-port geometry."""
    edges = set(info['edges'])
    tags = set()
    if any(len({i for i,j in edges if j == v}) >= 2 for v in info['active']):
        tags.add('merge')
    vertices = set(info['active'])
    while vertices:
        free = {v for v in vertices if not any(i in vertices and j == v for i,j in edges)}
        if not free:
            tags.add('cycle')
            break
        vertices -= free
    return tags


def _layout(rng, shape, n):
    h, w = shape
    positions = []
    for _ in range(500):
        if len(positions) == n:
            return positions
        r, c = int(rng.integers(h-3)), int(rng.integers(w-3))
        # A blank cell between footprints also separates 8-connected components.
        if all(abs(r-y) >= 5 or abs(c-x) >= 5 for y, x in positions):
            positions.append((r, c))
    return None


def _network(rng, shape, positions, n_active):
    owner = np.full(shape, -1, dtype=int)
    for i, (r, c) in enumerate(positions):
        owner[r:r+4, c:c+4] = i
    options = [{p: _route(pos, p, owner) for p in PORTS} for pos in positions]
    root = int(rng.integers(len(positions)))
    active, order, ports = {root}, [root], [set() for _ in positions]
    while len(active) < n_active:
        choices = [(i, p, j) for i in active if len(ports[i]) < 3
                   for p, (_, j) in options[i].items() if j is not None and j not in active]
        if not choices:
            return None
        # Both branching and long chains emerge from the same growth grammar.
        latest = [x for x in choices if x[0] == order[-1]]
        pool = latest if latest and rng.random() < .6 else choices
        i, p, j = pool[int(rng.integers(len(pool)))]
        ports[i].add(p)
        active.add(j)
        order.append(j)
    for i in range(len(positions)):
        exits = [p for p, (_, j) in options[i].items() if j is None]
        if not ports[i]:
            if i in active:
                if not exits:
                    return None
                ports[i].add(exits[int(rng.integers(len(exits)))])
            else:
                # A single opening is always asymmetric on dormant witnesses.
                ports[i].add(PORTS[int(rng.integers(len(PORTS)))])
        elif exits and rng.random() < .3:
            ports[i].add(exits[int(rng.integers(len(exits)))])
        # Additional contacts into the reached set make mergers and cycles.
        if i in active and len(ports[i]) < 4 and rng.random() < .35:
            returning = [p for p, (_, j) in options[i].items() if j in active and j != i and p not in ports[i]]
            if returning:
                ports[i].add(returning[int(rng.integers(len(returning)))])
    return root, order, ports


def _pack_bars(rng, a, bg, counts, forbidden, forced):
    blocked = forbidden.copy()
    blocked |= a != bg
    # Markers must have a blank halo from every tile and from other markers.
    foreground = a != bg
    for r, c in np.argwhere(foreground):
        blocked[max(0,r-1):r+2, max(0,c-1):c+2] = True
    for color in sorted(counts, key=lambda x: -counts[x]):
        length = counts[color]
        candidates = []
        for orientation in ('horizontal', 'vertical'):
            if color in forced and orientation != forced[color]:
                continue
            bh, bw = (1, length) if orientation == 'horizontal' else (length, 1)
            if bh > a.shape[0] or bw > a.shape[1]:
                continue
            rr, cc = np.indices((a.shape[0]-bh+1, a.shape[1]-bw+1))
            clear = np.ones(rr.shape, dtype=bool)
            for offset in range(length):
                clear &= ~blocked[rr+(offset if bh > 1 else 0), cc+(offset if bw > 1 else 0)]
            for r, c in np.argwhere(clear):
                candidates.append((orientation, int(r), int(c)))
        if not candidates:
            return None
        orientation, r, c = candidates[int(rng.integers(len(candidates)))]
        for offset in range(length):
            y, x = r+(offset if orientation == 'vertical' else 0), c+(offset if orientation == 'horizontal' else 0)
            a[y, x] = color
            blocked[max(0,y-1):y+2, max(0,x-1):x+2] = True
    return a


def _make_pair(rng, witness, test):
    for attempt in range(500):
        if witness == 0:
            n, n_active = 4, 3
        elif witness == 1:
            n, n_active = 5, 4
        else:
            n = int(rng.integers(3, 8))
            n_active = int(rng.integers(2, n))
        low = 18 if n <= 4 else 23
        shape = tuple(int(x) for x in rng.integers(low, 31, 2))
        positions = _layout(rng, shape, n)
        if positions is None:
            continue
        network = _network(rng, shape, positions, n_active)
        if network is None:
            continue
        root, order, ports = network
        palette = [int(x) for x in rng.permutation(10)]
        bg, spark = palette[:2]
        if witness >= 0:
            colors = palette[2:2+n]
        else:
            k = int(rng.integers(2, n+1))
            colors = [palette[2+i] for i in range(k)]
            colors += [int(rng.choice(colors)) for _ in range(n-k)]
            rng.shuffle(colors)
        counts = {c: int(rng.integers(0, 10 if test else 6)) for c in set(colors)}
        forced = {}
        if witness == 0:
            for i, length in zip(order, (1, 2, 2)):
                counts[colors[i]] = length
            dormant = next(i for i in range(n) if i not in order)
            counts[colors[dormant]] = 3
            orientations = ['horizontal', 'vertical']
            rng.shuffle(orientations)
            forced = {colors[i]: orientation for i, orientation in zip(order[1:], orientations)}
        elif witness == 1:
            for i, length in zip(order, (3, 4, 5, 0)):
                counts[colors[i]] = length
            dormant = next(i for i in range(n) if i not in order)
            counts[colors[dormant]] = int(rng.choice((1, 2, 3, 5)))
        elif test:
            counts[colors[order[0]]] = int(rng.integers(6, 10))
        counts = {c: k for c, k in counts.items() if k > 0}
        a = np.full(shape, bg, dtype=np.int8)
        for i, (r, c) in enumerate(positions):
            # Sample the desired *active* opening set, then undo its counter.
            # The input is still a complete specification of the relay graph.
            tile = np.full((4, 4), colors[i], dtype=np.int8)
            if i == root:
                tile[1:3, 1:3] = spark
            for p in ports[i]:
                tile[p] = bg
            a[r:r+4, c:c+4] = np.rot90(tile, counts.get(colors[i], 0))
        # Reserve the intended active routes so bars are visibly separate from
        # the signal drawing. Actual reachability is checked from the input.
        forbidden = np.zeros(shape, dtype=bool)
        owner = np.full(shape, -1, dtype=int)
        for i, (r, c) in enumerate(positions):
            owner[r:r+4, c:c+4] = i
        for i in order:
            for p in ports[i]:
                for q in _route(positions[i], p, owner)[0]:
                    forbidden[q] = True
        a = _pack_bars(rng, a, bg, counts, forbidden, forced)
        if a is None:
            continue
        try:
            out, info = _evaluate(a)
            if len(info['active']) != n_active:
                continue
            if witness >= 0:
                if any(any(np.array_equal(info['relays'][i]['tile'], np.rot90(info['relays'][i]['tile'], k))
                           for k in (1, 2, 3)) for i in info['active']):
                    continue
                if any(max(len(path) for path, _ in info['routes'][i]) < 3 for i in info['active']):
                    continue
            if witness == 0 and {sum(map(abs, d)) for d in info['headings']} != {1, 2}:
                continue
            if witness == 0 and max(len(info['routes'][i]) for i in info['active']) < 2:
                continue
            if not _different(a, out, 'ignore_counts'):
                continue
            if witness >= 0 and not _different(a, out, 'all_active'):
                continue
            if witness == 0 and not _different(a, out, 'counterclockwise'):
                continue
            if witness == 1 and (max(info['depths'].values()) < 2 or
                                 not _different(a, out, 'clamp_count')):
                continue
        except ValueError:
            continue
        return dict(input=a, output=out)
    raise RuntimeError("bounded relay construction exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    _require(_integer(num_train) and 2 <= num_train <= 10, "num_train must be an integer in 2..10")
    _require(_integer(num_test) and 1 <= num_test <= 4 and num_train > num_test,
             "num_test must be an integer in 1..4 smaller than num_train")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {'train': [], 'test': []}, set()
    taught = set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            witness = index if split == 'train' and index < 2 else -1
            for _ in range(20):
                pair = _make_pair(rng, witness, split == 'test')
                key = _key(pair['input'])
                tags = _topology_tags(_evaluate(pair['input'])[1])
                if split == 'test' and not tags <= taught:
                    continue
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    if split == 'train':
                        taught |= tags
                    break
            else:
                raise RuntimeError("bounded uniqueness sampling exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid example counts")
    seen, train_infos = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(_key(a) not in seen, "duplicate input")
            seen.add(_key(a))
            expected, info = _evaluate(a)
            _require(a.shape == b.shape and np.array_equal(expected, b), "incorrect relay output")
            _require(len(info['active']) >= 2 and bool(info['edges']), "no propagated activation")
            _require(_different(a, b, 'ignore_counts'), "counter rotation does not affect the answer")
            _require(_different(a, b, 'root_only'), "propagation does not affect the answer")
            # Relays absorb at their footprint, even if its entrance is a hole.
            for i, t in enumerate(info['relays']):
                r, c = t['pos']
                _require(np.array_equal(a[r+1:r+3, c+1:c+3], b[r+1:r+3, c+1:c+3]), "payload core changed")
                if i not in info['active']:
                    _require(np.array_equal(a[r:r+4, c:c+4], b[r:r+4, c:c+4]), "dormant relay changed")
                else:
                    desired = np.full((4,4), t['color'], dtype=np.int8)
                    desired[1:3,1:3] = t['core']
                    _require(np.array_equal(b[r:r+4,c:c+4], desired), "active relay openings are not all lit")
            for bar in info['bars'].values():
                _require(all(b[p] == info['background'] for p in bar['cells']), "counter was not erased")
            # Rotation conserves each relay's pre-emission area, including root.
            for t, tile in zip(info['relays'], info['rotated_tiles']):
                _require(int(np.sum(t['tile'] != info['background'])) == int(np.sum(tile != info['background'])),
                         "rotor area not conserved")
                # Check opening transport by coordinates rather than np.rot90.
                k = info['bars'].get(t['color'], {}).get('length', 0)
                moved = set()
                for r, c in t['gaps']:
                    for _ in range(k % 4):
                        r, c = c, 3-r
                    moved.add((r, c))
                _require(moved == {p for p in PORTS if tile[p] == info['background']},
                         "clockwise opening transport failed")
            reached = {info['root']}
            for _ in range(len(info['relays'])):
                reached |= {j for i in tuple(reached) for _,j in info['routes'][i] if j is not None}
            _require(reached == info['active'], "activation is not the least reachability closure")
            if split == 'train':
                train_infos.append(info)
    for variant in ('ignore_counts', 'counterclockwise', 'binary_count', 'clamp_count',
                    'shared_count', 'absent_one', 'bar_direction', 'all_active', 'root_only'):
        _require(any(_different(_grid(p['input']), _grid(p['output']), variant) for p in puzzle['train']),
                 "training fails to distinguish " + variant)
    informative = set()
    repeated_orientation = False
    cardinal, diagonal = False, False
    for info in train_infos:
        orient = {}
        for i in info['active']:
            t = info['relays'][i]
            k = info['bars'].get(t['color'], {}).get('length', 0)
            informative.add(k)
            if k > 1:
                orient.setdefault(k, set()).add(info['bars'][t['color']]['orientation'])
        repeated_orientation |= any({'horizontal', 'vertical'} <= s for s in orient.values())
        cardinal |= any(sum(map(abs, d)) == 1 for d in info['headings'])
        diagonal |= any(sum(map(abs, d)) == 2 for d in info['headings'])
    _require(set(range(6)) <= informative, "training must expose absent and counts 1..5 on active relays")
    _require(repeated_orientation, "training must expose equal counters in both orientations")
    _require(cardinal and diagonal, "training must demonstrate side and corner emission")
    _require(any(max(f['depths'].values()) >= 2 for f in train_infos), "training must expose a relay after a relay")
    _require(any(max(len(f['routes'][i]) for i in f['active']) >= 2 for f in train_infos),
             "training must expose simultaneous emission from multiple openings")
    taught = set().union(*(_topology_tags(f) for f in train_infos))
    _require(all(_topology_tags(_evaluate(_grid(p['input']))[1]) <= taught for p in puzzle['test']),
             "test merge or cycle needs a training witness")
    _require(all(any(b['length'] >= 6 and any(i in f['active'] and t['color'] == color
                 for i,t in enumerate(f['relays'])) for color,b in f['bars'].items())
                 for f in (_evaluate(_grid(p['input']))[1] for p in puzzle['test'])),
             "tests must exercise a long counter on a reached relay")


def _fixture_checks():
    """Small hand-specified answers: local turns, absorption at a hole, gating.

    Both counter lengths 1 and 5 have the same expected signal diagram. The
    receiver has an input hole on the incident ray, yet absorbs that ray. A
    marked dormant relay must keep its original opening.
    """
    for length in (1, 5):
        a = np.zeros((18,22), dtype=np.int8)
        for pos,color,gap,core in (((8,3),1,(0,2),9), ((8,12),2,(2,0),2), ((2,12),3,(2,0),3)):
            r,c = pos
            a[r:r+4,c:c+4] = color
            a[r+1:r+3,c+1:c+3] = core
            a[r+gap[0],c+gap[1]] = 0
        a[0,:length] = 1
        a[0,7:9] = 2
        a[0,12:15] = 3
        expected = a.copy()
        expected[0] = 0
        expected[8:12,3:7] = 1
        expected[9:11,4:6] = 9
        expected[8:12,12:16] = 2
        expected[10,7:12] = 1
        expected[9,16:22] = 2
        original = a.copy()
        predicted, info = _evaluate(a)
        assert np.array_equal(predicted, expected)
        assert np.array_equal(a, original), "solver mutated its input"
        assert len(info['active']) == 2
        assert _different(a, expected, 'ignore_counts')
        assert _different(a, expected, 'root_only')
        assert _different(a, expected, 'all_active')
        # The color layers share one canvas but rotate around different centers.
        for t in info['relays']:
            r,c = t['pos']
            assert np.array_equal(predicted[r+1:r+3,c+1:c+3], a[r+1:r+3,c+1:c+3])


def self_test():
    """Bounded 42-seed structural and semantic checks, with public measurements."""
    _fixture_checks()
    shapes, layouts, palettes, fingerprints = set(), set(), set(), set()
    relay_counts, active_counts, headings, lengths = set(), set(), set(), set()
    repeated = cycle = branching = merging = edge_touch = False
    for seed in range(42):
        p = generate_puzzle(seed)
        fingerprints.add(tuple(_key(x['input']) for s in ('train','test') for x in p[s]))
        for split in ('train', 'test'):
            for pair in p[split]:
                out, f = _evaluate(pair['input'])
                assert np.array_equal(out, pair['output'])
                shapes.add(pair['input'].shape)
                palettes.add((f['background'], f['spark'], tuple(sorted({t['color'] for t in f['relays']}))))
                layouts.add(tuple((t['pos'], tuple(sorted(t['gaps']))) for t in f['relays']))
                relay_counts.add(len(f['relays']))
                active_counts.add(len(f['active']))
                headings |= f['headings']
                lengths |= {b['length'] for b in f['bars'].values()}
                repeated |= len({t['color'] for t in f['relays']}) < len(f['relays'])
                indegree = [sum(j == k for i,j in f['edges']) for k in f['active']]
                outdegree = [sum(i == k for i,j in f['edges']) for k in f['active']]
                branching |= max(outdegree, default=0) >= 2
                merging |= max(indegree, default=0) >= 2
                # Kahn's algorithm detects a directed cycle, independently of BFS.
                vertices = set(f['active'])
                while vertices:
                    free = {v for v in vertices if not any(i in vertices and j == v for i,j in f['edges'])}
                    if not free:
                        cycle = True
                        break
                    vertices -= free
                h,w = pair['input'].shape
                edge_touch |= any(r == 0 or c == 0 or r+4 == h or c+4 == w for r,c in (t['pos'] for t in f['relays']))
    assert len(fingerprints) == 42 and len(shapes) >= 35 and len(layouts) >= 180
    assert relay_counts == set(range(3,8)) and len(active_counts) >= 4
    assert headings == {(r,c) for r in (-1,0,1) for c in (-1,0,1) if (r,c) != (0,0)}
    assert lengths == set(range(1,10)) and len(palettes) >= 150
    assert repeated and branching and merging and cycle and edge_touch
    for nt in range(2,11):
        for ne in range(1, min(4,nt-1)+1):
            generate_puzzle(100+nt*4+ne, nt, ne)
    for nt,ne in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.0),(None,1)):
        try:
            generate_puzzle(0,nt,ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p, q = generate_puzzle(17,2,1), generate_puzzle(17,2,1)
    assert all(_key(a[k]) == _key(b[k]) for s in p for a,b in zip(p[s],q[s]) for k in ('input','output'))
    bad = {s:[{k:v.copy() for k,v in x.items()} for x in p[s]] for s in p}
    bad['test'][0]['output'][0,0] = (int(bad['test'][0]['output'][0,0])+1)%10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return dict(seeds=42, shapes=len(shapes), layouts=len(layouts), palettes=len(palettes),
                relay_counts=sorted(relay_counts), active_counts=sorted(active_counts),
                headings=len(headings), counter_lengths=sorted(lengths), repeated_colors=repeated,
                branches=branching, merges=merging, cycles=cycle, edge_touch=edge_touch)
