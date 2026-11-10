"""Fewest-stop switchboards. No parent generators or external state are used.

Switch settings are decisions in the route search. Only used switches are
actuated. The input encodes all colors; zero is always the background.
"""
from collections import deque
from itertools import combinations
import heapq
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
HOST_SEEDS = (0, 67636673, 731838998, 356457734, 736397069, 354453366,
              109213060, 462777161, 321806462, 128548918, 181308276,
              63147569, 985943982, 224471273, 802743228, 253392759,
              850045678, 726033936, 994301971, 987012249, 149027790,
              309415612, 863289287, 751396213, 94952704, 612950272,
              897652999, 713014596, 642431159, 681638840, 193830282,
              635544939, 147487022, 267219721, 691909224, 705742327,
              183831655, 517215365, 407358012, 813111902, 247368010, 155080116)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape),
             "grid dimensions must be 1..30")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "grid cells must be integer colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _square(cells):
    rr, cc = zip(*cells)
    h, w = max(rr) - min(rr) + 1, max(cc) - min(cc) + 1
    return h if h == w and len(cells) == h * w else 0


def _switches(grid, wire, divider):
    switches = []
    covered = set()
    for cells in _components(grid == divider):
        rr, cc = zip(*cells)
        horizontal = len(set(rr)) == 1 and max(cc) - min(cc) == 2
        vertical = len(set(cc)) == 1 and max(rr) - min(rr) == 2
        _require(len(cells) == 3 and (horizontal or vertical), "divider must be a three-cell bar")
        r, c = sorted(cells)[1]
        _require(1 <= r < grid.shape[0] - 1 and 1 <= c < grid.shape[1] - 1,
                 "switch tile must fit in the canvas")
        region = {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        _require(not covered & region, "switch tiles must be disjoint")
        _require(all(grid[p] == wire for p in region - set(cells)),
                 "the other six cells in a switch tile must be track")
        covered.update(region)
        switches.append((r, c, bool(horizontal)))
    _require(1 <= len(switches) <= 4, "the family permits one to four switches")
    return switches


def _read(grid):
    colors = set(map(int, np.unique(grid)))
    _require(0 in colors and len(colors) == 5, "inputs have background 0 and four distinct nonzero roles")
    groups = {c: _components(grid == c) for c in colors - {0}}
    candidates = []
    for endpoint, blocks in groups.items():
        sizes = [_square(x) for x in blocks]
        if len(blocks) != 2 or len(set(sizes)) != 1 or sizes[0] not in (2, 3):
            continue
        size = sizes[0]
        for ordinary, bb in groups.items():
            if ordinary == endpoint or len(bb) < 3 or not all(_square(x) == size for x in bb):
                continue
            for divider in sorted(colors - {0, endpoint, ordinary}):
                wire = next(iter(colors - {0, endpoint, ordinary, divider}))
                try:
                    switches = _switches(grid, wire, divider)
                except ValueError:
                    continue
                stations = sorted(blocks + bb, key=lambda x: x[0])
                ends = [i for i, x in enumerate(stations) if grid[x[0]] == endpoint]
                candidates.append(dict(wire=wire, ordinary=ordinary, endpoint=endpoint,
                                       divider=divider, switches=switches, stations=stations,
                                       ends=ends, size=size))
    _require(len(candidates) == 1, "input roles must be uniquely recognizable by shape and count")
    return candidates[0]


def _turn(grid, data, bits):
    result = grid.copy()
    for i, (r, c, horizontal) in enumerate(data['switches']):
        if not bits & (1 << i):
            continue
        result[r - 1:r + 2, c - 1:c + 2] = data['wire']
        if horizontal:
            result[r - 1:r + 2, c] = data['divider']
        else:
            result[r, c - 1:c + 2] = data['divider']
    return result


def _network(grid, data, bits):
    """Recover edges from the rendered setting, including parallel edges.

    Track components are simple paths between station ports, or closed loops
    with no station. A path returning to the same station is an unused loop.
    """
    structural = _turn(grid, data, bits)
    components = _components(structural == data['wire'])
    station_at = {p: i for i, block in enumerate(data['stations']) for p in block}
    adjacency = [[] for _ in data['stations']]
    centers = [np.mean(block, axis=0) for block in data['stations']]
    edges = []
    for cells in components:
        cell_set = set(cells)
        touching, termini, terminal_cells = set(), [], []
        for r, c in cells:
            degree = sum((r + dr, c + dc) in cell_set for dr, dc in DIRS)
            _require(degree in (1, 2), "track outside stations must not branch or end in isolated pixels")
            neighbors = {station_at[p] for dr, dc in DIRS
                         if (p := (r + dr, c + dc)) in station_at}
            touching.update(neighbors)
            if degree == 1:
                _require(len(neighbors) == 1, "an exposed wire end must touch one station")
                termini.append(next(iter(neighbors)))
                terminal_cells.append((r, c))
            else:
                _require(not neighbors, "station attachment must be at a wire terminal")
        _require(len(termini) in (0, 2), "wire component must be a path or a closed loop")
        _require(len(touching) <= 2 and (termini or not touching), "ambiguous station connections")
        if len(touching) != 2:
            continue
        a, b = sorted(touching)
        e = len(edges)
        # Track steps plus Manhattan distance from each terminal pixel to its
        # station center measure routed geometric length, including station
        # scale. This is a diagnostic alternative, never the solving cost.
        length = len(cells) - 1 + sum(float(np.abs(np.array(p) - centers[n]).sum())
                                     for p, n in zip(terminal_cells, termini))
        edges.append(dict(nodes=(a, b), cells=cells, area=len(cells), length=length))
        adjacency[a].append((b, e))
        adjacency[b].append((a, e))
    return dict(bits=bits, adjacency=adjacency, edges=edges)


def _settings(grid, data):
    return [_network(grid, data, bits) for bits in range(1 << len(data['switches']))]


def _shortest_paths(network, a, b):
    """At most two minimum-hop routes; two suffice to establish a tie."""
    adjacency = network['adjacency']
    dist, parents = [-1] * len(adjacency), [[] for _ in adjacency]
    dist[a] = 0
    q = deque([a])
    while q:
        n = q.popleft()
        for other, e in adjacency[n]:
            if dist[other] < 0:
                dist[other] = dist[n] + 1
                q.append(other)
            if dist[other] == dist[n] + 1:
                parents[other].append((n, e))
    if dist[b] < 0:
        return []
    routes = []
    stack = [(b, [b], [])]
    while stack and len(routes) < 2:
        n, nodes, edges = stack.pop()
        if n == a:
            routes.append((nodes[::-1], edges[::-1]))
        else:
            for prev, edge in parents[n]:
                stack.append((prev, nodes + [prev], edges + [edge]))
    return routes


def _route(network, nodes, edges, data):
    cells = frozenset(p for e in edges for p in network['edges'][e]['cells'])
    used = [i for i, (r, c, _) in enumerate(data['switches'])
            if any((r + dr, c + dc) in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1))]
    turns = network['bits'] & sum(1 << i for i in used)
    return dict(nodes=tuple(nodes), edges=tuple(edges), cells=cells, used=tuple(used),
                turns=turns, stops=len(nodes) - 2, area=len(cells),
                length=sum(network['edges'][e]['length'] for e in edges),
                signature=(tuple(nodes), tuple(sorted(cells))))


def _best(settings, data, ends):
    best, routes = float('inf'), {}
    for network in settings:
        for nodes, edges in _shortest_paths(network, *ends):
            stops = len(nodes) - 2
            if stops > best:
                continue
            if stops < best:
                best, routes = stops, {}
            r = _route(network, nodes, edges, data)
            routes[r['signature']] = r
    return next(iter(routes.values())) if len(routes) == 1 else None


def _pixel_min(settings, ends, metric='area'):
    """Minimum connector area or routed center length, for shortcut witnesses."""
    best = float('inf')
    for network in settings:
        dist = [float('inf')] * len(network['adjacency'])
        dist[ends[0]] = 0
        queue = [(0, ends[0])]
        while queue:
            d, a = heapq.heappop(queue)
            if d != dist[a]:
                continue
            if a == ends[1]:
                best = min(best, d)
                break
            for b, e in network['adjacency'][a]:
                dd = d + network['edges'][e][metric]
                if dd < dist[b]:
                    dist[b] = dd
                    heapq.heappush(queue, (dd, b))
    return best


def _paint(grid, data, route):
    result = _turn(grid, data, route['turns'])
    for n in route['nodes'][1:-1]:
        for p in data['stations'][n]:
            result[p] = data['endpoint']
    for p in route['cells']:
        result[p] = data['ordinary']
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    data = _read(grid)
    route = _best(_settings(grid, data), data, data['ends'])
    _require(route is not None, "the feasible minimum-station route must be unique")
    return _paint(grid, data, route)


def _line(a, b):
    """Axis-aligned raster line, endpoints included."""
    _require(a[0] == b[0] or a[1] == b[1], "line must be orthogonal")
    dr, dc = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
    return [(a[0] + i * dr, a[1] + i * dc)
            for i in range(abs(b[0] - a[0]) + abs(b[1] - a[1]) + 1)]


def _polyline(points):
    return set(p for a, b in zip(points, points[1:]) for p in _line(a, b))


def _connected(vertices, edges):
    reached, stack = {vertices[0]}, [vertices[0]]
    adjacency = {v: [] for v in vertices}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    while stack:
        for b in adjacency[stack.pop()]:
            if b not in reached:
                reached.add(b)
                stack.append(b)
    return len(reached) == len(vertices)


def _layout(rng, switch_count, size):
    nr, nc = (3, 3) if switch_count == 1 and rng.random() < .7 else (3, 4)
    if switch_count >= 3:
        nr, nc = 4, 4
    vertices = [(r, c) for r in range(nr) for c in range(nc)]
    interior = [v for v in vertices if 0 < v[0] < nr - 1 and 0 < v[1] < nc - 1]
    switches = {interior[int(i)] for i in rng.choice(len(interior), switch_count, replace=False)}
    edges = [(a, b) for a, b in combinations(vertices, 2)
             if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
    # Keep every switch's four ports. Remove other edges while retaining the
    # underlying connected graph; individual settings may disconnect it.
    removal = float(rng.uniform(.15, .7))
    candidates = [e for e in edges if not any(v in switches for v in e)]
    rng.shuffle(candidates)
    for edge in candidates:
        if rng.random() < removal:
            proposed = [e for e in edges if e != edge]
            if _connected(vertices, proposed):
                edges = proposed
    neighbors = {v: [b if a == v else a for a, b in edges if v in (a, b)] for v in vertices}
    bends = {v for v in vertices if v not in switches and len(neighbors[v]) == 2
             and rng.random() < float(rng.uniform(.2, .8))}
    while len(vertices) - len(switches) - len(bends) < 5:
        bends.remove(sorted(bends)[int(rng.integers(len(bends)))])
    # Unequal pitches decouple physical route length from station count.
    def axis(n):
        gaps = rng.integers(5, 9, size=n - 1)
        return [0] + list(map(int, np.cumsum(gaps)))
    rows, cols = axis(nr), axis(nc)
    anchors = {v: (rows[v[0]], cols[v[1]]) for v in vertices}
    shape = rows[-1] + 3, cols[-1] + 3
    grid = np.zeros(shape, dtype=np.int8)
    def direction(a, b):
        return DIRS.index((int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))))
    def port(v, side):
        r, c = anchors[v]
        if v in switches:
            offsets = ((0, 0), (0, 2), (2, 2), (2, 0))
        elif size == 2:
            offsets = ((0, 1), (1, 1), (1, 0), (0, 0))
        else:
            offsets = ((0, 1), (1, 2), (2, 1), (1, 0))
        dr, dc = offsets[side]
        return r + dr, c + dc
    for a, b in edges:
        side = direction(a, b)
        pa, pb = port(a, side), port(b, (side + 2) % 4)
        if side in (1, 3):
            m = (pa[1] + pb[1]) // 2
            points = [pa, (pa[0], m), (pb[0], m), pb]
        else:
            m = (pa[0] + pb[0]) // 2
            points = [pa, (m, pa[1]), (m, pb[1]), pb]
        for p in _polyline(points):
            grid[p] = 1
    for v in bends:
        aa, bb = [port(v, direction(v, x)) for x in neighbors[v]]
        # Connect through the object's interior; this vertex is a bend, not a
        # station, and has exactly two ports.
        mid = (aa[0], bb[1]) if rng.random() < .5 else (bb[0], aa[1])
        for p in _polyline([aa, mid, bb]):
            grid[p] = 1
    for v in vertices:
        r, c = anchors[v]
        if v in switches:
            grid[r:r + 3, c:c + 3] = 1
            if rng.random() < .5:
                grid[r + 1, c:c + 3] = 4
            else:
                grid[r:r + 3, c + 1] = 4
        elif v not in bends:
            grid[r:r + size, c:c + size] = 2
    stations = _components(grid == 2)
    data = dict(wire=1, ordinary=2, endpoint=3, divider=4,
                stations=stations, switches=_switches(grid, 1, 4), size=size)
    return grid, data


def _initial_diff(settings, data, ends, route):
    initial = _shortest_paths(settings[0], *ends)
    return not initial or all(_route(settings[0], n, e, data)['signature'] != route['signature']
                              for n, e in initial)


def _make_pair(rng, switch_count, size, witness, palette):
    target = int(rng.choice([0, 1, 2, 3, 4])) if witness == 'any' else int(rng.integers(1, 4))
    for attempt in range(180):
        try:
            base, data = _layout(rng, switch_count, size)
            settings = _settings(base, data)
        except ValueError:
            continue
        choices = list(combinations(range(len(data['stations'])), 2))
        rng.shuffle(choices)
        eligible = []
        for ends in choices:
            route = _best(settings, data, ends)
            if route is None or not route['used'] or (witness != 'any' and route['stops'] < 1):
                continue
            if switch_count >= 3 and (len(route['used']) < 2 or not route['turns']):
                continue
            if witness == 'switch' and (not route['turns'] or not _shortest_paths(settings[0], *ends)
                                        or not _initial_diff(settings, data, ends, route)):
                continue
            if witness == 'distance' and route['area'] <= _pixel_min(settings, ends):
                continue
            if witness == 'distance' and route['length'] <= _pixel_min(settings, ends, 'length'):
                continue
            if witness == 'distance' and not (0 < route['turns'].bit_count() < len(route['used'])):
                continue
            eligible.append((abs(route['stops'] - target), ends))
            if route['stops'] == target:
                break
        if not eligible:
            continue
        _, ends = min(eligible, key=lambda x: x[0])
        base = base.copy()
        for n in ends:
            for p in data['stations'][n]:
                base[p] = 3
        # Rotate and reflect whole scenes: there is no hidden gravity.
        base = np.rot90(base, int(rng.integers(4))).copy()
        if rng.random() < .5:
            base = np.fliplr(base).copy()
        margins = []
        for dim in base.shape:
            total = int(rng.integers(min(5, 30 - dim) + 1))
            before = int(rng.integers(total + 1))
            margins.append((before, total - before))
        base = np.pad(base, margins)
        grid = np.asarray(palette, dtype=np.int8)[base]
        return {'input': grid, 'output': solve(grid)}
    raise RuntimeError(f'bounded construction exhausted for {switch_count} switches / {witness}')


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    _require(lo <= int(value) <= hi, f'{name} must be {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and int(seed) >= 0, 'seed must be a nonnegative integer')
    seed = int(seed)
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    _require(num_train > num_test, 'num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    palette = [0] + list(map(int, rng.choice(np.arange(1, 10), 4, replace=False)))
    vary_palette = rng.random() < .6
    schedule = list(map(int, rng.permutation(num_train)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            witness = 'switch' if split == 'train' and i == schedule[0] else (
                'distance' if split == 'train' and i == schedule[1] else 'any')
            switches = 1 if witness == 'switch' else 2 if witness == 'distance' else (
                int(rng.integers(1, 3)) if split == 'train' else int(rng.integers(3, 5)))
            size = int(rng.choice([2, 3]))
            roles = [0] + list(map(int, rng.choice(np.arange(1, 10), 4, replace=False))) if vary_palette else palette
            for _ in range(12):
                pair = _make_pair(rng, switches, size, witness, roles)
                signature = _key(pair['input'])
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded duplicate-input retries exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(grid):
    data = _read(grid)
    settings = _settings(grid, data)
    route = _best(settings, data, data['ends'])
    _require(route is not None, 'route must be unique')
    initial = _shortest_paths(settings[0], *data['ends'])
    area_conflict = route['area'] > _pixel_min(settings, data['ends'])
    length_conflict = route['length'] > _pixel_min(settings, data['ends'], 'length')
    return data, route, dict(switches=len(data['switches']), used=len(route['used']),
                            turned=route['turns'].bit_count(), stops=route['stops'],
                            stations=len(data['stations']), area=route['area'],
                            length=route['length'], area_conflict=area_conflict,
                            length_conflict=length_conflict,
                            initial_reachable=bool(initial),
                            initial_diff=_initial_diff(settings, data, data['ends'], route),
                            distance_conflict=area_conflict and length_conflict)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'exact train/test keys required')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _count(len(puzzle['train']), 'num_train', 2, 10)
    _count(len(puzzle['test']), 'num_test', 1, 4)
    _require(len(puzzle['train']) > len(puzzle['test']), 'more training than test examples required')
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'exact pair keys required')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'output preserves canvas dimensions')
            _require(_key(inp) not in seen, 'inputs must be distinct')
            seen.add(_key(inp))
            data, route, f = _features(inp)
            if split == 'train':
                _require(f['switches'] in (1, 2), 'training uses one or two switches')
            else:
                _require(f['switches'] in (3, 4) and f['used'] >= 2 and f['turned'] >= 1,
                         'test must combine multiple switches and actuate at least one')
            _require(np.array_equal(out, _paint(inp, data, route)), 'incorrect switch actuation or route highlighting')
            # Check preservation and role transfer directly, in addition to the
            # full expected-grid comparison.
            allowed = set(route['cells'])
            for n in route['nodes'][1:-1]:
                allowed.update(data['stations'][n])
            for i, (r, c, _) in enumerate(data['switches']):
                region = {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
                if route['turns'] & (1 << i):
                    allowed.update(region)
                elif i not in route['used']:
                    _require(all(inp[p] == out[p] for p in region), 'unused switch must be preserved')
            _require(all(tuple(p) in allowed for p in np.argwhere(inp != out)), 'off-route pixels changed')
            for n in data['ends']:
                _require(all(inp[p] == out[p] for p in data['stations'][n]), 'endpoint footprint changed')
            if split == 'train':
                training.append(f)
    _require(any(f['turned'] and f['initial_diff'] and f['initial_reachable'] for f in training),
             'training must distinguish an existing unturned route from a better switched route')
    _require(any(f['distance_conflict'] and 0 < f['turned'] < f['used'] for f in training),
             'training must demonstrate station counting together with mixed switch decisions')


def _exhaustive_best(settings, data, ends):
    """Self-test oracle for the route search: enumerate simple station routes.

    This is another authored check, with the same pixel parser, not an
    independent solver. Partial routes above the best stop count are pruned.
    """
    best, signatures = float('inf'), set()
    for network in settings:
        stack = [(ends[0], [ends[0]], [])]
        while stack:
            a, nodes, edges = stack.pop()
            if a == ends[1]:
                cost = len(nodes) - 2
                if cost < best:
                    best, signatures = cost, set()
                if cost == best:
                    signatures.add(_route(network, nodes, edges, data)['signature'])
                continue
            if len(nodes) - 1 > best:
                continue
            for b, e in network['adjacency'][a]:
                if b not in nodes:
                    stack.append((b, nodes + [b], edges + [e]))
    return signatures


def self_test():
    """Bounded, authored semantic and diversity checks over the 42 host seeds."""
    structures, shapes, palettes, signatures = set(), set(), set(), set()
    stats = []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p[k]) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output')))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                data, route, f = _features(inp)
                f.update(split=split, shape=inp.shape, size=data['size'])
                stats.append(f)
                shapes.add(inp.shape)
                roles = (0, data['wire'], data['ordinary'], data['endpoint'], data['divider'])
                palettes.add(roles)
                norm = np.zeros(inp.shape, dtype=np.int8)
                for i, c in enumerate(roles):
                    norm[inp == c] = i
                fg = np.argwhere(norm != 0)
                lo, hi = fg.min(0), fg.max(0) + 1
                structures.add(_key(norm[lo[0]:hi[0], lo[1]:hi[1]]))
                _require(np.array_equal(solve(inp.copy()), pair['output']), 'input-only prediction disagrees')
                # Color equivariance: geometry determines the four roles.
                if seed == 0:
                    perm = np.array([0, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int8)
                    _require(np.array_equal(solve(perm[inp]), perm[pair['output']]), 'palette equivariance failed')
                    moved_in, moved_out = np.fliplr(np.rot90(inp)), np.fliplr(np.rot90(pair['output']))
                    _require(np.array_equal(solve(moved_in), moved_out), 'spatial equivariance failed')
                    _require(_exhaustive_best(_settings(inp, data), data, data['ends']) == {route['signature']},
                             'BFS and exhaustive simple-route enumeration disagree')
    _require(len(signatures) == 42 and len(structures) >= 180, 'insufficient structural variation')
    _require(len(shapes) >= 50 and len(palettes) >= 35, 'insufficient geometry or palette variation')
    _require({f['switches'] for f in stats} == {1, 2, 3, 4}, 'switch-count coverage missing')
    _require({f['size'] for f in stats} == {2, 3}, 'station-scale coverage missing')
    _require(len({f['stations'] for f in stats}) >= 6 and len({f['stops'] for f in stats}) >= 4,
             'station or answer counts do not vary')
    _require(any(0 < f['turned'] < f['used'] for f in stats), 'mixed switch settings not demonstrated')
    _require(any(f['used'] < f['switches'] for f in stats), 'unused switches not demonstrated')
    a, b = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
                 for k in ('input', 'output')), 'seed determinism failed')
    generate_puzzle(123, 10, 4)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    corrupt = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    corrupt['test'][0]['output'][0, 0] = (int(corrupt['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError('altered output was accepted')
    # Relabel station endpoints in a real network to obtain a genuine tied
    # minimum, and check that it is rejected instead of silently tie-broken.
    tied = False
    for pair in a['train'] + a['test']:
        inp = pair['input']
        data = _read(inp)
        settings = _settings(inp, data)
        for ends in combinations(range(len(data['stations'])), 2):
            if _best(settings, data, ends) is not None or not any(_shortest_paths(n, *ends) for n in settings):
                continue
            altered = inp.copy()
            altered[altered == data['endpoint']] = data['ordinary']
            for n in ends:
                for p in data['stations'][n]:
                    altered[p] = data['endpoint']
            try:
                solve(altered)
            except ValueError:
                tied = True
            else:
                raise AssertionError('tied routes were accepted')
            break
        if tied:
            break
    _require(tied, 'no tied-route rejection case exercised')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds')
