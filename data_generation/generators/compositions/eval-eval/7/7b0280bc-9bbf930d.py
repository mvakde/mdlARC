"""Fewest interchanges in a network of stations and matching twin rails.

Only NumPy and the standard library are used. Solving reads the input pixels;
generation metadata, examples, and seed are never available to the solver.
"""
from collections import Counter, deque
from itertools import combinations
import heapq
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= x <= 30 for x in g.shape), "grid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _need(bool(np.all((g >= 0) & (g <= 9))), "ARC colors required")
    return g


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            p = stack.pop()
            cells.append(p)
            for d in DIRS:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _square(cells):
    rs, cs = zip(*cells)
    h, w = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
    return h if h == w and h >= 3 and h % 2 and len(cells) == h * w else 0


def _adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        a, b = e['ends']
        adj[a].append((b, i))
        adj[b].append((a, i))
    return adj


def _paths(adj, start, target):
    """Enumerate simple station itineraries; parallel tracks remain distinct."""
    result = []
    stack = [(start, (start,), ())]
    while stack:
        a, nodes, edges = stack.pop()
        if a == target:
            result.append((nodes, edges))
            continue
        for b, e in adj[a]:
            if b not in nodes:
                stack.append((b, nodes + (b,), edges + (e,)))
    return result


def _usable_routes(paths, edges, override=None):
    result = []
    for nodes, ids in paths:
        if any(not edges[e]['active'] and (override is None or e != override[0]) for e in ids):
            continue
        colors = tuple(override[1] if override is not None and e == override[0]
                       else edges[e]['colors'][0] for e in ids)
        changes = tuple(nodes[i] for i in range(1, len(nodes) - 1)
                        if colors[i - 1] != colors[i])
        runs = tuple(c for i, c in enumerate(colors) if not i or c != colors[i - 1])
        result.append(dict(nodes=nodes, edges=ids, colors=colors, changes=changes,
                           cost=len(changes), runs=runs,
                           length=sum(len(edges[e]['path']) for e in ids)))
    return sorted(result, key=lambda x: x['cost'])


def _unique(routes):
    return bool(routes) and (len(routes) == 1 or routes[0]['cost'] < routes[1]['cost'])


def _gate_witness(paths, edges, winner):
    """Can repairing one real mismatched pair change the unique optimum?"""
    for e, edge in enumerate(edges):
        if edge['active']:
            continue
        for color in edge['colors']:
            routes = _usable_routes(paths, edges, (e, color))
            if _unique(routes) and routes[0]['edges'] != winner['edges']:
                return e, color
    return None


def _read(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _need(all(int(g[p]) == bg for p in ((0, -1), (-1, 0), (-1, -1))), "blank corners")
    _need(np.all(g[0] == bg) and np.all(g[-1] == bg) and
          np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg), "blank outer border")
    by_color = {int(c): _components(g == c) for c in np.unique(g) if c != bg}
    block_colors = {c: cc for c, cc in by_color.items()
                    if cc and all(_square(x) for x in cc)}
    endpoints = [c for c, cc in block_colors.items() if len(cc) == 2]
    ordinary = [c for c, cc in block_colors.items() if len(cc) >= 3]
    _need(len(endpoints) == len(ordinary) == 1, "two terminal and at least three ordinary blocks")
    terminal, ordinary = endpoints[0], ordinary[0]
    stations = sorted(by_color[terminal] + by_color[ordinary], key=lambda x: x[0])
    sizes = {_square(x) for x in stations}
    _need(len(sizes) == 1 and len(stations) <= 20, "equal odd square stations")
    size = sizes.pop()
    centers = [(min(p[0] for p in s) + size // 2,
                min(p[1] for p in s) + size // 2) for s in stations]
    station_map = np.full(g.shape, -1, dtype=int)
    for i, cells in enumerate(stations):
        for p in cells:
            station_map[p] = i
    guides = set(by_color) - {terminal, ordinary}
    _need(2 <= len(guides) <= 7, "two to seven guide identities")

    def pair_at(p, d):
        normal = (-d[1], d[0])
        a, b = _add(p, normal), _add(p, (-normal[0], -normal[1]))
        if _inside(g, a) and _inside(g, b) and int(g[a]) in guides and int(g[b]) in guides:
            return int(g[a]), int(g[b])
        return None

    recovered = {}
    for node, center in enumerate(centers):
        for direction in DIRS:
            p = (center[0] + (size // 2 + 1) * direction[0],
                 center[1] + (size // 2 + 1) * direction[1])
            if not _inside(g, p) or int(g[p]) != bg or pair_at(p, direction) is None:
                continue
            path, states, pairs, bends = [p], set(), [], 0
            d = direction
            while True:
                _need((p, d) not in states, "track cycles outside stations")
                states.add((p, d))
                support = pair_at(p, d)
                if support is not None:
                    pairs.append(support)
                q = _add(p, d)
                _need(_inside(g, q), "track must terminate at a station")
                target = int(station_map[q])
                if target >= 0:
                    _need(target != node, "self loop")
                    break
                if int(g[q]) == bg:
                    _need(q not in path, "track revisits a free cell")
                    p = q
                    path.append(p)
                    continue
                choices = []
                for turn in ((-d[1], d[0]), (d[1], -d[0])):
                    q = _add(p, turn)
                    if _inside(g, q) and int(g[q]) == bg and pair_at(q, turn) is not None:
                        choices.append(turn)
                _need(len(choices) == 1, "track turn must be geometrically unique")
                d = choices[0]
                bends += 1
            _need(len(set(pairs)) == 1, "rail identities cannot change between stations")
            colors = tuple(sorted(pairs[0]))
            rail_cells = {q for cell in path for delta in DIRS
                          if _inside(g, q := _add(cell, delta)) and int(g[q]) in guides}
            _need({int(g[q]) for q in rail_cells} == set(colors), "foreign rail touching track")
            ends = tuple(sorted((node, target)))
            key = ends, tuple(sorted(path))
            if key in recovered:
                edge = recovered[key]
                _need(tuple(path) == edge['path'][::-1] and edge['colors'] == colors,
                      "reciprocal track geometry")
                edge['readings'] += 1
            else:
                recovered[key] = dict(ends=ends, path=tuple(path), colors=colors,
                                      active=colors[0] == colors[1], rails=rail_cells,
                                      bends=bends, readings=1)
    edges = list(recovered.values())
    _need(edges and all(e['readings'] == 2 for e in edges), "each track needs two ports")
    channels = [p for e in edges for p in e['path']]
    _need(len(channels) == len(set(channels)), "tracks cannot cross outside stations")
    rails = [p for e in edges for p in e['rails']]
    _need(len(rails) == len(set(rails)), "tracks must not share rail pixels")
    _need(set(rails) == {tuple(p) for p in np.argwhere(np.isin(g, list(guides))).tolist()},
          "unexplained guide pixels")
    adj = _adjacency(len(stations), edges)
    _need(all(1 <= len(x) <= 4 for x in adj), "station degree")
    reached, queue = {0}, [0]
    while queue:
        for b, _ in adj[queue.pop()]:
            if b not in reached:
                reached.add(b)
                queue.append(b)
    _need(len(reached) == len(stations), "geometric network must be connected")
    _need(len(edges) - len(stations) + 1 <= 5, "at most five independent loops")
    ends = [i for i, s in enumerate(stations) if int(g[s[0]]) == terminal]
    paths = _paths(adj, *ends)
    routes = _usable_routes(paths, edges)
    _need(_unique(routes), "unique minimum-interchange itinerary required")
    return dict(grid=g, background=bg, terminal=terminal, ordinary=ordinary,
                stations=stations, centers=centers, size=size, edges=edges,
                adjacency=adj, endpoints=ends, paths=paths, routes=routes)


def _paint(data, route=None):
    route = data['routes'][0] if route is None else route
    out = data['grid'].copy()
    for edge in route['edges']:
        for p in data['edges'][edge]['path']:
            out[p] = data['terminal']
    for node in route['changes']:
        for p in data['stations'][node]:
            out[p] = data['terminal']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Mark the cheapest simple journey using only the supplied input pixels."""
    return _paint(_read(input_grid))


def _polyline(points):
    result = [points[0]]
    for target in points[1:]:
        p = result[-1]
        d = (int(np.sign(target[0] - p[0])), int(np.sign(target[1] - p[1])))
        _need((d[0] == 0) != (d[1] == 0), "orthogonal polyline")
        while p != target:
            p = _add(p, d)
            result.append(p)
    return result


def _topology(rng, size, is_test, fallback=None):
    """A new lattice subgraph, with degree-two vertices optionally suppressed."""
    pitch = size + (2 if size == 5 else 3)
    if fallback in ('gate', 'scenic'):
        nr, nc = (3, 3) if fallback == 'gate' else (3, 4)
        active = {(r, c) for r in range(nr) for c in range(nc)}
        links = [((r, c), (r, c + 1)) for r in range(nr) for c in range(nc - 1)]
        links += [((r, c), (r + 1, c)) for c in (0, nc - 1) for r in range(nr - 1)]
        visible = active.copy()
    elif fallback == 'test':
        nr = nc = 3
        active = {(r, c) for r in range(3) for c in range(3)}
        links = [(a, b) for a, b in combinations(sorted(active), 2)
                 if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
        visible = active.copy()
    else:
        shapes = [(2, 4), (3, 3), (3, 4), (4, 3)] if not is_test else [
            (3, 3), (3, 4), (4, 3), (4, 4), (2, 5), (5, 2)]
        shapes = [(r, c) for r, c in shapes
                  if size + (max(r, c) - 1) * pitch <= 28]
        nr, nc = shapes[int(rng.integers(len(shapes)))]
        active = {(r, c) for r in range(nr) for c in range(nc)}
        if rng.random() < .4:
            corner = [(0, 0), (0, nc - 1), (nr - 1, 0), (nr - 1, nc - 1)][int(rng.integers(4))]
            active.remove(corner)
        possible = [(a, b) for a, b in combinations(sorted(active), 2)
                    if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
        maximum = len(possible) - len(active) + 1
        _need(maximum >= (3 if is_test else 2), "insufficient loops")
        target = int(rng.integers(3 if is_test else 2, min(maximum, 5 if is_test else 3) + 1))
        parent = {p: p for p in active}

        def root(p):
            while parent[p] != p:
                p = parent[p]
            return p

        links, extra = [], []
        for i in rng.permutation(len(possible)):
            a, b = possible[int(i)]
            ra, rb = root(a), root(b)
            if ra != rb:
                parent[ra] = rb
                links.append((a, b))
            else:
                extra.append((a, b))
        rng.shuffle(extra)
        links += extra[:target]
        degree = Counter(p for edge in links for p in edge)
        visible = {p for p in active if degree[p] != 2 or rng.random() < .55}
        for p in sorted(active):
            if len(visible) >= 5:
                break
            visible.add(p)
    adjacency = {p: [] for p in active}
    for a, b in links:
        adjacency[a].append(b)
        adjacency[b].append(a)
    order = sorted(visible)
    ids = {p: i for i, p in enumerate(order)}
    walks, used = [], set()
    for a in order:
        for b in sorted(adjacency[a]):
            if frozenset((a, b)) in used:
                continue
            walk, prev, cur = [a, b], a, b
            used.add(frozenset((a, b)))
            while cur not in visible:
                nxt = next(p for p in adjacency[cur] if p != prev)
                used.add(frozenset((cur, nxt)))
                walk.append(nxt)
                prev, cur = cur, nxt
            _need(cur != a, "suppressed self loop")
            walks.append((ids[a], ids[cur], walk))

    def axis(n):
        base = size + (n - 1) * pitch
        # The gate fallback must contain a long straight run for a visible gap.
        minimum = 6 if fallback == 'gate' else 0
        slack = int(rng.integers(minimum, 29 - base))
        gaps = pitch + rng.multinomial(slack, [1 / (n - 1)] * (n - 1))
        return [size // 2] + list(size // 2 + np.cumsum(gaps))

    rows, cols = axis(nr), axis(nc)
    centers = [(int(rows[p[0]]), int(cols[p[1]])) for p in order]
    station_cells = [tuple((r + dr, c + dc) for dr in range(-size // 2 + 1, size // 2 + 1)
                           for dc in range(-size // 2 + 1, size // 2 + 1)) for r, c in centers]
    blocks = set(p for cells in station_cells for p in cells)
    edges = []
    for a, b, walk in walks:
        full = _polyline([(int(rows[r]), int(cols[c])) for r, c in walk])
        path = tuple(p for p in full if p not in blocks)
        left, right = set(), set()
        for p, q in zip(full, full[1:]):
            d = q[0] - p[0], q[1] - p[1]
            n = -d[1], d[0]
            for point in (p, q):
                if point in blocks:
                    continue
                left.add(_add(point, n))
                right.add(_add(point, (-n[0], -n[1])))
        left -= blocks | set(path)
        right -= blocks | set(path)
        _need(not left & right, "self contacting rails")
        edges.append(dict(ends=(a, b), path=path, left=left, right=right))
    return dict(size=size, centers=centers, stations=station_cells, edges=edges,
                order=order, adjacency=_adjacency(len(order), edges))


def _select(rng, topo, kind, q, fallback=False):
    edges = topo['edges']
    all_paths = [(a, b, _paths(topo['adjacency'], a, b))
                 for a, b in combinations(range(len(topo['stations'])), 2)]
    if fallback:
        order = topo['order']
        if kind in ('gate', 'scenic', 'extra'):
            nc = max(c for r, c in order) + 1
            a, b = order.index((1, 0)), order.index((1, nc - 1))
            paths = next(p for x, y, p in all_paths if (x, y) == (a, b))
            upper = next(p for p in paths if order[p[0][1]][0] == 0)
            lower = next(p for p in paths if order[p[0][1]][0] == 2)
            middle = next(p for p in paths if len(p[1]) == nc - 1)
            colors = [0] * len(edges)
            if kind == 'gate':
                seq = [0, 0, 1, 0]
                for e, c in zip(upper[1], seq): colors[e] = c
                for i, e in enumerate(lower[1]): colors[e] = i % 2
                # Four lower edges must cost three, versus upper cost two.
                closed = middle[1][0]
            else:
                for e, c in zip(upper[1], [0, 0, 0, 1, 1]): colors[e] = c
                for e, c in zip(middle[1], [0, 1, 0]): colors[e] = c
                closed = lower[1][0]
            for e, c in enumerate(colors):
                edges[e].update(colors=(c, (c + 1) % q) if e == closed else (c, c), active=e != closed)
        else:
            order = topo['order']
            a, b = order.index((0, 0)), order.index((2, 2))
            paths = next(p for x, y, p in all_paths if (x, y) == (a, b))
            closed = None
            for e, edge in enumerate(edges):
                u, v = (order[x] for x in edge['ends'])
                # One A-to-B interchange on the outside L. Every competing
                # route must enter C and finish on B, so costs at least two.
                c = 0 if u[0] == v[0] == 0 else 1 if u[1] == v[1] == 2 else 2
                if (2, 2) in (u, v): c = 1
                if set((u, v)) == {(0, 0), (1, 0)}: closed = e
                edge.update(colors=(c, c), active=True)
            edges[closed].update(colors=(0, 2), active=False)
        routes = _usable_routes(paths, edges)
        _need(_unique(routes), "constructive fallback optimum")
        return a, b

    viable = [(a, b, p) for a, b, p in all_paths if len(p) >= (5 if kind == 'test' else 3)]
    _need(viable, "no candidate terminal pairs")
    for _ in range(65):
        colors = list(map(int, rng.integers(0, q, len(edges))))
        flags = list(map(bool, rng.random(len(edges)) > rng.uniform(.06, .25)))
        # Correlated line stretches are sampled on a real path, not on row bands.
        if rng.random() < .85:
            _, _, candidates = viable[int(rng.integers(len(viable)))]
            long = [p for p in candidates if len(p[1]) >= 4]
            if long:
                _, ids = long[int(rng.integers(len(long)))]
                cut1 = int(rng.integers(1, len(ids) - 1))
                cut2 = int(rng.integers(cut1 + 1, len(ids)))
                c1, c2 = map(int, rng.choice(q, 2, replace=False))
                for j, e in enumerate(ids):
                    colors[e] = c1 if j < cut1 or (j >= cut2 and kind == 'gate') else c2
                    flags[e] = True
        if all(flags):
            flags[int(rng.integers(len(edges)))] = False
        for e, c in enumerate(colors):
            other = int(rng.choice([x for x in range(q) if x != c]))
            edges[e].update(colors=(c, c) if flags[e] else tuple(sorted((c, other))), active=flags[e])
        for i in rng.permutation(len(viable)):
            a, b, paths = viable[int(i)]
            routes = _usable_routes(paths, edges)
            if not _unique(routes):
                continue
            winner = routes[0]
            if kind == 'test':
                if len(routes) < 5 or winner['cost'] < 1 or len(winner['nodes']) - 2 <= winner['cost']:
                    continue
            elif not 2 <= len(routes) <= 4:
                continue
            if kind in ('gate', 'scenic'):
                if winner['cost'] < 1 or len(winner['nodes']) - 2 <= winner['cost']:
                    continue
            if kind == 'gate':
                if len(winner['runs']) != 3 or winner['runs'][0] != winner['runs'][2]:
                    continue
                if _gate_witness(paths, edges, winner) is None:
                    continue
            if kind == 'scenic':
                if len(winner['edges']) <= min(len(p['edges']) for p in routes):
                    continue
                if winner['length'] <= min(p['length'] for p in routes):
                    continue
            return a, b
    raise ValueError("no suitably discriminating color assignment")


def _render(rng, topo, endpoints, roles, rotation, require_gap=False):
    bg, ordinary, terminal, palette = roles
    edge_data = topo['edges']
    blocks = {p for cells in topo['stations'] for p in cells}
    channels = {p for e in edge_data for p in e['path']}
    rails, owners, gap_count = {}, {}, 0
    gap_choices = []
    for eid, edge in enumerate(edge_data):
        for side, color in ((edge['left'], edge['colors'][0]), (edge['right'], edge['colors'][1])):
            for p in side:
                _need(p not in blocks | channels and p not in rails, "rail collision")
                rails[p], owners[p] = palette[color], eid
        path = edge['path']
        for i in range(2, len(path) - 2):
            ds = [(path[j + 1][0] - path[j][0], path[j + 1][1] - path[j][1])
                  for j in range(i - 2, i + 2)]
            if len(set(ds)) != 1:
                continue
            d = ds[0]
            n = -d[1], d[0]
            flanks = (_add(path[i], n), _add(path[i], (-n[0], -n[1])))
            if all(p in edge['left'] | edge['right'] for p in flanks):
                gap_choices.append((eid, flanks))
    rng.shuffle(gap_choices)
    for eid, flanks in gap_choices:
        if (require_gap and not gap_count) or rng.random() < .08:
            if all(p in rails for p in flanks):
                for p in flanks:
                    rails.pop(p)
                gap_count += 1
    _need(not require_gap or gap_count, "need a demonstrated straight rail gap")
    all_cells = blocks | set(rails) | channels
    high_r, high_c = max(r for r, c in all_cells), max(c for r, c in all_cells)
    base_h, base_w = high_r + 1, high_c + 1
    _need(base_h <= 28 and base_w <= 28, "drawing exceeds padding budget")
    h, w = int(rng.integers(base_h + 2, 31)), int(rng.integers(base_w + 2, 31))
    top, left = int(rng.integers(1, h - base_h)), int(rng.integers(1, w - base_w))
    g = np.full((h, w), bg, dtype=int)
    for (r, c), color in rails.items(): g[r + top, c + left] = color
    for node, cells in enumerate(topo['stations']):
        for r, c in cells:
            g[r + top, c + left] = terminal if node in endpoints else ordinary
    return np.rot90(g, rotation).copy()


def _pair(rng, size, kind, roles, rotation):
    q = len(roles[3])
    for attempt in range(26):
        fallback = attempt == 25
        try:
            fallback_kind = ('scenic' if kind == 'extra' else kind) if fallback else None
            topo = _topology(rng, size, kind == 'test', fallback_kind)
            endpoints = _select(rng, topo, kind, q, fallback)
            inp = _render(rng, topo, endpoints, roles, rotation, require_gap=kind == 'gate')
            data = _read(inp)
            routes, winner = data['routes'], data['routes'][0]
            _need((len(routes) >= 5) if kind == 'test' else (2 <= len(routes) <= 4), "itinerary shift")
            if kind == 'gate':
                _need(_gate_witness(data['paths'], data['edges'], winner) is not None and
                      len(winner['runs']) == 3 and winner['runs'][0] == winner['runs'][2], "gate witness")
            if kind == 'scenic':
                _need(len(winner['edges']) > min(len(r['edges']) for r in routes) and
                      winner['length'] > min(r['length'] for r in routes), "scenic witness")
            return dict(input=inp, output=_paint(data))
        except ValueError:
            if fallback:
                raise
    raise RuntimeError("bounded sampling exhausted")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if not low <= value <= high:
        raise ValueError(name + " out of range")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, m = _count(num_train, 2, 10, 'num_train'), _count(num_test, 1, 4, 'num_test')
    _need(n > m, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    size_mode = int(rng.integers(3))
    train_sizes = [3 if size_mode == 0 else 5 if size_mode == 1 else (3 if i % 2 == 0 else 5)
                   for i in range(n)]
    rng.shuffle(train_sizes)
    mixed_roles, mixed_rotation = bool(rng.integers(2)), bool(rng.integers(2))
    base_colors, base_rotation = list(map(int, rng.permutation(10))), int(rng.integers(4))
    kinds = ['gate', 'scenic'] + ['extra'] * (n - 2)
    rng.shuffle(kinds)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', n), ('test', m)):
        for i in range(count):
            size = train_sizes[i] if split == 'train' else int(rng.choice(train_sizes))
            kind = kinds[i] if split == 'train' else 'test'
            colors = list(map(int, rng.permutation(10))) if mixed_roles else base_colors
            q = int(rng.integers(3, 6))
            roles = (colors[0], colors[1], colors[2], colors[3:3 + q])
            rotation = int(rng.integers(4)) if mixed_rotation else (base_rotation + (split == 'test')) % 4
            for _ in range(12):
                pair = _pair(rng, size, kind, roles, rotation)
                key = pair['input'].shape, pair['input'].tobytes()
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("unique input sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _state_cost(data):
    """An independent state-graph shortest-cost check (no path enumeration)."""
    start, target = data['endpoints']
    best, heap = {(start, -1): 0}, [(0, start, -1)]
    while heap:
        cost, node, incoming = heapq.heappop(heap)
        if best[(node, incoming)] != cost:
            continue
        if node == target:
            return cost
        for nxt, eid in data['adjacency'][node]:
            e = data['edges'][eid]
            if not e['active']:
                continue
            color = e['colors'][0]
            new = cost + int(incoming != -1 and color != incoming)
            key = nxt, color
            if new < best.get(key, float('inf')):
                best[key] = new
                heapq.heappush(heap, (new, nxt, color))
    raise ValueError("unreachable terminals")


def _features(data):
    winner, routes = data['routes'][0], data['routes']
    gaps = 0
    for edge in data['edges']:
        path = edge['path']
        for a, p, b in zip(path, path[1:], path[2:]):
            d = p[0] - a[0], p[1] - a[1]
            if d != (b[0] - p[0], b[1] - p[1]):
                continue
            n = -d[1], d[0]
            if all(int(data['grid'][q]) == data['background']
                   for q in (_add(p, n), _add(p, (-n[0], -n[1])))):
                gaps += 1
    return dict(shape=list(data['grid'].shape), size=data['size'], stations=len(data['stations']),
                tracks=len(data['edges']), loops=len(data['edges']) - len(data['stations']) + 1,
                usable_routes=len(routes), transfers=winner['cost'], hops=len(winner['edges']),
                through=len(winner['nodes']) - 2 - winner['cost'],
                bends=sum(e['bends'] for e in data['edges']),
                gaps=gaps,
                mismatches=sum(not e['active'] for e in data['edges']),
                repeated=len(winner['runs']) >= 3 and len(set(winner['runs'])) < len(winner['runs']),
                scenic=len(winner['edges']) > min(len(r['edges']) for r in routes) and
                       winner['length'] > min(r['length'] for r in routes),
                gate=_gate_witness(data['paths'], data['edges'], winner) is not None)


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= n <= 10 and 1 <= m <= 4 and n > m, "pair counts")
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = inp.shape, inp.astype(np.int8).tobytes()
            _need(key not in seen, "duplicate input")
            seen.add(key)
            data = _read(inp)
            winner = data['routes'][0]
            _need(out.shape == inp.shape and np.array_equal(out, _paint(data)), "incorrect output")
            _need(_state_cost(data) == winner['cost'], "station-state cost disagrees")
            rail_mask = ~np.isin(inp, [data['background'], data['ordinary'], data['terminal']])
            _need(np.array_equal(inp[rail_mask], out[rail_mask]), "rails must be preserved")
            painted = set(p for e in winner['edges'] for p in data['edges'][e]['path'])
            painted |= set(p for node in winner['changes'] for p in data['stations'][node])
            changed = set(map(tuple, np.argwhere(inp != out).tolist()))
            _need(changed == painted, "only itinerary centers and transfer blocks change")
            _need(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))), "no unseen output ink")
            f = _features(data)
            _need(f['mismatches'] >= 1, "matched and mismatched tracks required")
            if split == 'train':
                _need(2 <= f['usable_routes'] <= 4, "training route count")
                training.append(f)
            else:
                _need(f['usable_routes'] >= 5 and f['transfers'] and f['through'],
                      "test recombines more alternatives with transfers and through stations")
    _need(any(f['gate'] and f['repeated'] and f['transfers'] == 2 and f['through'] and f['gaps']
              for f in training),
          "training needs a sensitive gate and A-B-A interchange witness")
    _need(any(f['scenic'] and f['transfers'] > 0 and f['through'] for f in training),
          "training needs a physically longer, more-station winner")


def self_test():
    records, shapes, structures, firsts, roles = [], set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for split in puzzle
                       for p, q in zip(puzzle[split], again[split]) for k in ('input', 'output'))
        first = puzzle['train'][0]['input']
        firsts.add((first.shape, first.tobytes()))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _read(pair['input'])
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                f = _features(d)
                f.update(seed=seed, split=split)
                records.append(f)
                shapes.add(tuple(f['shape']))
                roles.add((d['background'], d['ordinary'], d['terminal']))
                mapping = {d['background']: 0, d['ordinary']: 1, d['terminal']: 2}
                normalized = tuple(tuple(mapping.setdefault(int(c), len(mapping)) for c in row)
                                   for row in pair['input'])
                structures.add(normalized)
                if seed < 3:
                    # The roles and the cost rule are invariant under geometry
                    # symmetries and bijective relabeling of all ten colors.
                    assert np.array_equal(solve(np.rot90(pair['input'])), np.rot90(pair['output']))
                    assert np.array_equal(solve(pair['input'][:, ::-1]), pair['output'][:, ::-1])
                    perm = np.array([9, 2, 6, 0, 8, 4, 1, 7, 3, 5])
                    assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
                if seed < 3 and f['gate']:
                    eid, color = _gate_witness(d['paths'], d['edges'], d['routes'][0])
                    repaired = pair['input'].copy()
                    for cell in d['edges'][eid]['rails']:
                        repaired[cell] = color
                    repaired_data = _read(repaired)
                    assert repaired_data['routes'][0]['cost'] < d['routes'][0]['cost']
                    assert not np.array_equal(solve(repaired), pair['output'])
    assert len(firsts) == 42 and len(structures) >= 180 and len(shapes) >= 30
    assert len(roles) >= 30
    assert {3, 5} <= {f['size'] for f in records}
    assert len({f['stations'] for f in records}) >= 5
    assert len({f['hops'] for f in records}) >= 5
    assert any(f['bends'] for f in records)
    assert all(f['mismatches'] for f in records) and any(f['gaps'] for f in records)
    assert all(any(f['seed'] == seed and f['split'] == 'train' and f['scenic'] for f in records)
               for seed in range(42))
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(100 + n * 5 + m, n, m)
            assert len(p['train']) == n and len(p['test']) == m
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (None, 1), (4, False)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(89, 2, 1)
    p['test'][0]['output'] = p['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return dict(seeds=42, pairs=len(records), shapes=len(shapes), structures=len(structures),
                roles=len(roles), station_counts=sorted({f['stations'] for f in records}),
                chosen_hops=sorted({f['hops'] for f in records}), records=records)
