"""Turntable dispatch: minimum-station routing with a changing parcel shape.

Background 0 and signal trail 5 are fixed. Other roles are recognized by shape.
All analysis and solving use only the visible input. No parent module is imported.
"""
from collections import deque
from itertools import combinations
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _cells(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    components = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        stack, component = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    component.add(p)
                    stack.append(p)
        components.append(component)
    return components


def _box(cells):
    rows, cols = zip(*cells)
    return min(rows), min(cols), max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _rect(r, c, h, w):
    return {(r + i, c + j) for i in range(h) for j in range(w)}


def _ring(r, c, size):
    return _rect(r, c, size, size) - _rect(r + 1, c + 1, size - 2, size - 2)


def _put(grid, cells, color):
    for p in cells:
        grid[p] = color


def _fork(cells):
    """Recover the base, positive prongs, axis and stop without a shape bank."""
    parses = []
    for out in DIRS:
        perp = (-out[1], out[0])
        kmin = min(r * out[0] + c * out[1] for r, c in cells)
        qmin = min(r * perp[0] + c * perp[1] for r, c in cells)
        local = {(r * out[0] + c * out[1] - kmin,
                  r * perp[0] + c * perp[1] - qmin) for r, c in cells}
        width = max(q for k, q in local) + 1
        if not 5 <= width <= 7 or {q for k, q in local if k == 0} != set(range(width)):
            continue
        lengths = [sum((k, q) in local for k in range(1, 31)) for q in range(width)]
        expected = {(0, q) for q in range(width)} | {
            (k, q) for q, length in enumerate(lengths) for k in range(1, length + 1)}
        if expected != local:
            continue
        runs, q = [], 0
        while q < width:
            if lengths[q] == 0:
                q += 1
                continue
            start, length = q, lengths[q]
            while q < width and lengths[q] == length:
                q += 1
            runs.append((start, q, length))
        if len(runs) != 3 or runs[0][1] >= runs[1][0] or runs[1][1] >= runs[2][0]:
            continue
        if runs[1][0] + runs[1][1] != width:
            continue
        if runs[0][1] - runs[0][0] != 1 or runs[2][1] - runs[2][0] != 1:
            continue
        if not all(1 <= x[2] <= 4 for x in runs) or runs[1][2] > 3:
            continue
        parses.append(dict(out=out, perp=perp, kmin=kmin, qmin=qmin,
                           span=runs[1][1] - runs[1][0], q0=runs[1][0],
                           stop=runs[1][2] + 1, lengths=tuple(x[2] for x in runs)))
    assert len(parses) == 1, 'receiver must have a unique three-prong interpretation'
    return parses[0]


def _dock_cells(fork, length):
    return {((k + fork['kmin']) * fork['out'][0] + (q + fork['qmin']) * fork['perp'][0],
             (k + fork['kmin']) * fork['out'][1] + (q + fork['qmin']) * fork['perp'][1])
            for k in range(fork['stop'], fork['stop'] + length)
            for q in range(fork['q0'], fork['q0'] + fork['span'])}


def _adjacency(n, edges):
    result = [[] for _ in range(n)]
    for i, (a, b) in enumerate(edges):
        result[a].append((b, i))
        result[b].append((a, i))
    return result


def _routes(adjacency, source, docks, turns, span, length, *, ignore_fit=False, ignore_turns=False):
    """Enumerate simple station routes; a bend does not change the carried span."""
    result = []
    stack = [(source, [source], [], span, length)]
    while stack:
        a, nodes, edges, s, ell = stack.pop()
        if a in docks:
            if ignore_fit or docks[a]['span'] == s:
                result.append(dict(nodes=nodes, edges=edges, span=s, length=ell, dock=a))
            continue
        for b, e in adjacency[a]:
            if b in nodes:
                continue
            ss, ll = (ell, s) if b in turns and not ignore_turns else (s, ell)
            stack.append((b, nodes + [b], edges + [e], ss, ll))
    return result


def _choose(routes):
    assert routes, 'no shape-compatible destination'
    cost = min(len(p['nodes']) for p in routes)
    choices = [p for p in routes if len(p['nodes']) == cost]
    assert len(choices) == 1, 'minimum-station dispatch is tied'
    return choices[0]


def _read_input(input_grid):
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
    colors = set(map(int, np.unique(grid)))
    assert 0 in colors and 5 not in colors and len(colors) == 4
    candidates = []
    for color in sorted(colors - {0}):
        comps = _cells(grid == color)
        sources = [i for i, cc in enumerate(comps) if _box(cc)[2:] == (5, 5)
                   and cc == _ring(*_box(cc)[:2], 5)]
        if len(sources) == 1:
            candidates.append((color, comps, sources[0]))
    assert len(candidates) == 1, 'one parcel frame identifies the structural role'
    structure, entities, source = candidates[0]
    turns, docks, ordinary = set(), {}, set()
    for i, cc in enumerate(entities):
        if i == source:
            continue
        r, c, h, w = _box(cc)
        if (h, w) == (3, 3) and cc == _rect(r, c, 3, 3):
            ordinary.add(i)
        elif (h, w) == (3, 3) and cc == _ring(r, c, 3):
            turns.add(i)
        else:
            docks[i] = _fork(cc)
    assert 6 <= len(entities) <= 16 and len(ordinary) >= 1 and len(turns) >= 1
    assert len(ordinary | turns) >= 3 and 2 <= len(docks) <= 4
    r, c, _, _ = _box(entities[source])
    inside = _rect(r + 1, c + 1, 3, 3)
    parcel_colors = {int(grid[p]) for p in inside} - {0}
    assert len(parcel_colors) == 1
    parcel = parcel_colors.pop()
    assert parcel != structure
    tokens = _cells(grid == parcel)
    assert len(tokens) == 1 and tokens[0] <= inside
    tr, tc, th, tw = _box(tokens[0])
    assert tokens[0] == _rect(tr, tc, th, tw) and 1 <= th <= 3 and 1 <= tw <= 3 and th != tw
    wire = (colors - {0, structure, parcel}).pop()
    components = _cells(grid == wire)
    station_map = {p: i for i, cc in enumerate(entities) for p in cc}
    edges, departure = [], set()
    for cc in components:
        touching = set()
        for rr, col in cc:
            for dr, dc in DIRS:
                p = rr + dr, col + dc
                if p in station_map:
                    i = station_map[p]
                    touching.add(i)
                    if i == source:
                        if rr < r: departure.add((-1, 0))
                        elif rr >= r + 5: departure.add((1, 0))
                        elif col < c: departure.add((0, -1))
                        elif col >= c + 5: departure.add((0, 1))
        assert len(touching) == 2, 'a wire must join exactly two stations'
        edges.append(tuple(sorted(touching)))
    assert len(set(edges)) == len(edges), 'parallel wire components are not allowed'
    adj = _adjacency(len(entities), edges)
    assert len(adj[source]) == 1 and len(departure) == 1
    assert all(len(adj[n]) == 1 for n in docks)
    assert all(1 <= len(a) <= 4 for a in adj)
    reached, queue = {source}, deque([source])
    while queue:
        for b, e in adj[queue.popleft()]:
            if b not in reached:
                reached.add(b)
                queue.append(b)
    assert len(reached) == len(entities)
    assert len(edges) - len(entities) + 1 <= 3
    # A receiver's wire must enter its rear base, never an arm or the mouth.
    for n, fork in docks.items():
        e = adj[n][0][1]
        for p in components[e]:
            if any((p[0] + dr, p[1] + dc) in entities[n] for dr, dc in DIRS):
                k = p[0] * fork['out'][0] + p[1] * fork['out'][1] - fork['kmin']
                assert k == -1, 'receiver wires enter the back of the base'
        # All possible parcel lengths have visibly clear, in-bounds seating room.
        for p in _dock_cells(fork, 3):
            assert 0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1] and int(grid[p]) == 0
    direction = next(iter(departure))
    span, length = (tw, th) if direction[0] else (th, tw)
    routes = _routes(adj, source, docks, turns, span, length)
    selected = _choose(routes)
    return dict(entities=entities, source=source, ordinary=ordinary, turns=turns, docks=docks,
                parcel=parcel, token=tokens[0], structure=structure, wire=wire,
                components=components, edges=edges, adjacency=adj, direction=direction,
                span=span, length=length, routes=routes, selected=selected)


def _paint_answer(grid, data, path):
    out = np.array(grid, dtype=np.int8, copy=True)
    _put(out, data['token'], 0)
    for n in path['nodes'][1:-1]:
        _put(out, data['entities'][n], data['parcel'])
    for e in path['edges']:
        _put(out, data['components'][e], 5)
    _put(out, _dock_cells(data['docks'][path['dock']], path['length']), data['parcel'])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Dispatch using the input alone, including its initial parcel orientation."""
    data = _read_input(input_grid)
    return _paint_answer(input_grid, data, data['selected'])


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _topology(rng, richer):
    nr = int(rng.choice([2, 3, 4] if richer else [2, 3]))
    nc = int(rng.choice([2, 3, 4]))
    if nr == nc == 4:
        nc = 3
    if nr * nc < 6:
        nc = 3
    vertices = [(r, c) for r in range(nr) for c in range(nc)]
    possible = [(a, b) for a, b in combinations(vertices, 2)
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
    target = int(rng.integers(max(6, nr * nc - (1 if richer else 3)), nr * nc + 1))
    active = {vertices[int(rng.integers(len(vertices)))]}
    edges = []
    while len(active) < target:
        options = [(a, b) for a, b in possible if (a in active) != (b in active)]
        a, b = options[int(rng.integers(len(options)))]
        active.update((a, b))
        edges.append((a, b))
    extra = [e for e in possible if e[0] in active and e[1] in active and e not in edges]
    rng.shuffle(extra)
    for e in extra[:int(rng.integers(0, 4 if richer else 3))]:
        edges.append(e)
    neighbors = {v: [b if a == v else a for a, b in edges if v in (a, b)] for v in active}
    leaves = [v for v in sorted(active) if len(neighbors[v]) == 1]
    eligible = []
    for v in leaves:
        neighbor = neighbors[v][0]
        out = v[0] - neighbor[0], v[1] - neighbor[1]
        # The mouth points beyond the routing lattice, leaving unrestricted room.
        if not (0 <= v[0] + out[0] < nr and 0 <= v[1] + out[1] < nc):
            eligible.append(v)
    assert len(eligible) >= 2 and len(leaves) >= 3
    rng.shuffle(eligible)
    count = int(rng.integers(2, min(4, len(eligible), len(leaves) - 1) + 1))
    dock_vertices = eligible[:count]
    source_options = sorted(set(leaves) - set(dock_vertices))
    source = source_options[int(rng.integers(len(source_options)))]
    visible = set(active)
    for v in sorted(active):
        if len(neighbors[v]) == 2 and rng.random() < .55:
            visible.remove(v)
    normals = sorted(visible - set(dock_vertices) - {source})
    assert len(normals) >= 3
    turns = {v for v in normals if rng.random() < .38}
    if not turns:
        turns.add(normals[int(rng.integers(len(normals)))])
    if turns == set(normals):
        turns.remove(normals[0])
    return nr, nc, active, edges, neighbors, visible, source, dock_vertices, turns


def _render(rng, topology, roles, dims, spans, spacing=None, spoke_mode=None):
    nr, nc, active, raw_edges, neighbors, visible, source, docks, turns = topology
    # Independent variable spacings; reserve five pixels beyond a terminal base.
    def axis(n):
        maximum = 19 if n == 4 else 19 if n == 3 else 18
        slack = int(rng.integers(0, maximum - 6 * (n - 1) + 1))
        gaps = rng.multinomial(slack, [1 / (n - 1)] * (n - 1)) + 6
        return [0] + list(map(int, np.cumsum(gaps)))
    rows, cols = (axis(nr), axis(nc)) if spacing is None else spacing
    anchor = {v: (rows[v[0]], cols[v[1]]) for v in active}
    footprints, reserved, forks = {}, {}, {}
    for v in sorted(visible):
        r, c = anchor[v]
        if v == source:
            footprints[v] = _ring(r - 2, c - 2, 5)
            reserved[v] = _rect(r - 2, c - 2, 5, 5)
        elif v in docks:
            nb = neighbors[v][0]
            out = v[0] - nb[0], v[1] - nb[1]
            perp = (-out[1], out[0])
            span = spans[v]
            q0 = -(span // 2)
            if spoke_mode == 'standard':
                center = int(rng.integers(1, 3))
                left, right = map(int, rng.integers(center + 1, 5, size=2))
            elif spoke_mode == 'long_center':
                center = int(rng.integers(2, 4))
                left, right = map(int, rng.integers(1, center, size=2))
            else:
                left, center, right = map(int, rng.integers(1, 5, size=3))
                center = min(center, 3)
            local = {(0, q) for q in range(q0 - 2, q0 + span + 2)}
            for qa, qb, depth in ((q0 - 2, q0 - 1, left), (q0, q0 + span, center),
                                  (q0 + span + 1, q0 + span + 2, right)):
                local |= {(k, q) for k in range(1, depth + 1) for q in range(qa, qb)}
            footprints[v] = {(r + k * out[0] + q * perp[0], c + k * out[1] + q * perp[1])
                             for k, q in local}
            reserved[v] = footprints[v]
            forks[v] = _fork(footprints[v])
        else:
            reserved[v] = _rect(r - 1, c - 1, 3, 3)
            footprints[v] = _ring(r - 1, c - 1, 3) if v in turns else reserved[v]
    blocks = set().union(*reserved.values())
    lines = set()
    for a, b in raw_edges:
        ar, ac = anchor[a]
        br, bc = anchor[b]
        if ar == br:
            lo, hi = sorted((ac, bc))
            # Safe mid-corridor elbows, separate from station ports.
            shift = int(rng.integers(-1, 2)) if hi - lo >= 8 else 0
            pts = [(ar, lo), (ar, lo + 3), (ar + shift, lo + 3),
                   (ar + shift, hi - 3), (ar, hi - 3), (ar, hi)]
        else:
            lo, hi = sorted((ar, br))
            shift = int(rng.integers(-1, 2)) if hi - lo >= 8 else 0
            pts = [(lo, ac), (lo + 3, ac), (lo + 3, ac + shift),
                   (hi - 3, ac + shift), (hi - 3, ac), (hi, ac)]
        for p, q in zip(pts, pts[1:]):
            if p[0] == q[0]:
                lines |= {(p[0], c) for c in range(min(p[1], q[1]), max(p[1], q[1]) + 1)}
            else:
                lines |= {(r, p[1]) for r in range(min(p[0], q[0]), max(p[0], q[0]) + 1)}
    lines -= blocks
    r, c = anchor[source]
    h, w = dims
    token = _rect(r - 1 + int(rng.integers(4 - h)), c - 1 + int(rng.integers(4 - w)), h, w)
    # Reserve all possible final lengths at every receiver before choosing padding.
    fg = lines | set().union(*footprints.values()) | token
    room = fg | set().union(*(_dock_cells(f, 3) for f in forks.values()))
    low = np.min(list(room), axis=0)
    high = np.max(list(room), axis=0)
    shape = high - low + 1
    assert max(shape) <= 30
    margins = [int(rng.integers(min(3, 30 - int(s)) + 1)) for s in shape]
    offsets = [int(rng.integers(m + 1)) for m in margins]
    offset = -low + offsets
    grid = np.zeros(tuple(shape + margins), dtype=np.int8)
    def translated(cc):
        return {(int(p[0] + offset[0]), int(p[1] + offset[1])) for p in cc}
    structure, wire, parcel = roles
    _put(grid, translated(lines), wire)
    for cc in footprints.values():
        _put(grid, translated(cc), structure)
    _put(grid, translated(token), parcel)
    data = _read_input(grid)
    # Recover the intended compressed graph from raw edges; compare with pixels.
    vertex_to_id = {v: next(i for i, cc in enumerate(data['entities'])
                            if cc == translated(footprints[v])) for v in visible}
    expected = set()
    for v in visible:
        for nb in neighbors[v]:
            previous, current = v, nb
            while current not in visible:
                nxt = [x for x in neighbors[current] if x != previous]
                assert len(nxt) == 1
                previous, current = current, nxt[0]
            assert current != v
            expected.add(tuple(sorted((vertex_to_id[v], vertex_to_id[current]))))
    assert expected == set(data['edges'])
    assert sum(len(neighbors[v]) for v in visible) == 2 * len(expected)
    # Independently construct truth from the abstract topology and parcel state.
    ids = {v: i for i, v in enumerate(sorted(visible))}
    graph_edges = [(ids[v], ids[w]) for v, w in {
        tuple(sorted((next(x for x in visible if vertex_to_id[x] == a),
                      next(x for x in visible if vertex_to_id[x] == b)))) for a, b in expected}]
    graph_edges.sort()
    adj = _adjacency(len(ids), graph_edges)
    nb = neighbors[source][0]
    direction = nb[0] - source[0], nb[1] - source[1]
    span, length = (w, h) if direction[0] else (h, w)
    paths = _routes(adj, ids[source], {ids[v]: forks[v] for v in docks},
                    {ids[v] for v in turns}, span, length)
    chosen = _choose(paths)
    chosen_vertices = [sorted(visible)[i] for i in chosen['nodes']]
    assert [vertex_to_id[v] for v in chosen_vertices] == data['selected']['nodes']
    output = grid.copy()
    _put(output, translated(token), 0)
    for v in chosen_vertices[1:-1]:
        _put(output, translated(footprints[v]), parcel)
    chosen_pairs = {tuple(sorted((vertex_to_id[a], vertex_to_id[b])))
                    for a, b in zip(chosen_vertices, chosen_vertices[1:])}
    for cc, edge in zip(data['components'], data['edges']):
        if edge in chosen_pairs:
            _put(output, cc, 5)
    dest = chosen_vertices[-1]
    _put(output, translated(_dock_cells(forks[dest], chosen['length'])), parcel)
    return grid, output, data


def _witnesses(data):
    selected = data['selected']
    unfiltered = _routes(data['adjacency'], data['source'], data['docks'], data['turns'],
                         data['span'], data['length'], ignore_fit=True)
    best = min(len(p['nodes']) for p in unfiltered)
    shortest = [p for p in unfiltered if len(p['nodes']) == best]
    fit = len(shortest) == 1 and shortest[0]['nodes'] != selected['nodes']
    areas = [sum(len(data['components'][e]) for e in p['edges']) for p in data['routes']]
    chosen_area = sum(len(data['components'][e]) for e in selected['edges'])
    distance = chosen_area > min(areas)
    rotation = selected['span'] != data['span']
    return fit, distance, rotation


def _make_pair(rng, roles, purpose, richer):
    for attempt in range(1200):
        try:
            topology = _topology(rng, richer)
            dims = tuple(map(int, rng.choice([1, 2, 3], size=2, replace=False)))
            spans = {v: int(rng.integers(1, 4)) for v in topology[7]}
            grid, output, data = _render(rng, topology, roles, dims, spans,
                                         spoke_mode={'fit': 'standard', 'distance': 'long_center'}.get(purpose))
            assert len(data['routes']) >= 2
            fit, distance, rotation = _witnesses(data)
            if purpose == 'fit' and not (fit and len(set(data['selected']['nodes']) & data['turns']) == 2):
                continue
            if purpose == 'distance' and not (distance and rotation and len(set(data['selected']['nodes']) & data['turns']) == 1):
                continue
            if richer and purpose == 'test' and not (len(data['edges']) >= len(data['entities']) and rotation):
                continue
            k = int(rng.integers(4))
            grid, output = np.rot90(grid, k).copy(), np.rot90(output, k).copy()
            if rng.random() < .5:
                grid, output = grid[:, ::-1].copy(), output[:, ::-1].copy()
            assert np.array_equal(solve(grid), output)
            return {'input': grid, 'output': output}
        except AssertionError:
            continue
    return _fallback_pair(rng, roles, purpose)


def _fallback_pair(rng, roles, purpose):
    """Constructive witnesses after bounded random sampling; same visible rule."""
    dims = tuple(map(int, rng.choice([1, 2, 3], 2, replace=False)))
    h, w = dims
    if purpose == 'fit':
        # Three leaves, two turntables on the selected route, one on the longer
        # viable route, and a nearer receiver with the third (impossible) span.
        source = (2, 0)
        docks = [(0, 0), (0, 3), (2, 3)]
        turns = {(1, 1), (0, 1)}
        chains = [[source, (1, 0), (0, 0)],
                  [(1, 0), (1, 1), (0, 1), (0, 2), (0, 3)],
                  [(1, 1), (2, 1), (2, 2), (1, 2), (1, 3), (2, 3)]]
        visible = {p for chain in chains for p in chain}
        # Initial departure is vertical: transverse w, longitudinal h.
        spans = {(0, 0): 6 - h - w, (0, 3): w, (2, 3): h}
        spacing = ([0, 7, 14], [0, 6, 12, 18])
    else:
        # A sparse, long bent branch has two stations; the physically shorter
        # branch has three. The sparse branch contains the single turntable.
        source = (0, 0)
        docks = [(0, 3), (0, 2)]
        turns = {(2, 2)}
        chains = [[source, (1, 0), (2, 0), (2, 1), (2, 2), (2, 3), (1, 3), (0, 3)],
                  [(1, 0), (1, 1), (1, 2), (0, 2)]]
        if purpose == 'test':
            chains.append([(1, 2), (2, 2)])
        visible = {source, *docks, (1, 0), (1, 1), (1, 2), (2, 2)}
        spans = {(0, 3): h, (0, 2): w}
        spacing = ([0, 7, 14], [0, 6, 12, 20])
    edges = sorted({tuple(sorted((a, b))) for chain in chains for a, b in zip(chain, chain[1:])})
    active = {p for chain in chains for p in chain}
    neighbors = {v: [b if a == v else a for a, b in edges if v in (a, b)] for v in active}
    topology = (3, 4, active, edges, neighbors, visible, source, docks, turns)
    grid, out, data = _render(rng, topology, roles, dims, spans, spacing=spacing,
                              spoke_mode={'fit': 'standard', 'distance': 'long_center'}.get(purpose))
    assert len(data['routes']) >= 2
    fit, distance, rotation = _witnesses(data)
    if purpose == 'fit':
        assert fit and len(set(data['selected']['nodes']) & data['turns']) == 2
    elif purpose == 'distance':
        assert distance and rotation
    elif purpose == 'test':
        assert rotation and len(data['edges']) >= len(data['entities'])
    k = int(rng.integers(4))
    grid, out = np.rot90(grid, k).copy(), np.rot90(out, k).copy()
    if rng.random() < .5:
        grid, out = grid[:, ::-1].copy(), out[:, ::-1].copy()
    assert np.array_equal(solve(grid), out)
    return {'input': grid, 'output': out}


def _count(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, 'num_train')
    ne = _count(num_test, 1, 4, 'num_test')
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    pool = [c for c in range(1, 10) if c != 5]
    base_roles = tuple(map(int, rng.choice(pool, 3, replace=False)))
    fresh = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    purposes = ['fit', 'distance'] + ['free'] * (nt - 2)
    rng.shuffle(purposes)
    for split, purpose_list in (('train', purposes), ('test', ['test'] * ne)):
        for purpose in purpose_list:
            roles = tuple(map(int, rng.choice(pool, 3, replace=False))) if fresh else base_roles
            for retry in range(20):
                pair = _make_pair(rng, roles, purpose, richer=split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, witnesses, turn_counts, stop_modes = set(), [], [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for g in pair.values():
                a = np.asarray(g)
                assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
                assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            d = _read_input(inp)
            assert len(d['routes']) >= 2
            assert np.array_equal(out, _paint_answer(inp, d, d['selected']))
            path = d['selected']
            changed = set(d['token']) | _dock_cells(d['docks'][path['dock']], path['length'])
            for n in path['nodes'][1:-1]:
                changed |= d['entities'][n]
            for e in path['edges']:
                changed |= d['components'][e]
            assert changed == set(map(tuple, np.argwhere(inp != out).tolist()))
            assert all(int(out[p]) == d['structure'] for n in [d['source']] + list(d['docks']) for p in d['entities'][n])
            assert all(out[p] == 0 for p in d['token'])
            assert set(map(int, np.unique(out))) - set(map(int, np.unique(inp))) == {5}
            if split == 'train':
                witnesses.append(_witnesses(d))
                turn_counts.append(len(set(path['nodes']) & d['turns']))
                left, center, right = d['docks'][path['dock']]['lengths']
                stop_modes.append((center < min(left, right), center > max(left, right)))
            else:
                assert len(d['edges']) >= len(d['entities']) and _witnesses(d)[2]
    assert any(w[0] and t == 2 and mode[0] for w, t, mode in zip(witnesses, turn_counts, stop_modes)), 'training must expose gated routing, two turns and a short central stop'
    assert any(w[1] and w[2] and t == 1 and mode[1] for w, t, mode in zip(witnesses, turn_counts, stop_modes)), 'training must expose station cost, one turn and a long central stop'


def self_test():
    structures, palettes, shapes, graph_forms, directions, states, puzzles = set(), set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        if seed < 2:
            repeat = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], repeat[s]) for k in p)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                d = _read_input(inp)
                assert np.array_equal(solve(inp), out)
                shapes.add(inp.shape)
                palettes.add((d['structure'], d['wire'], d['parcel']))
                semantic = np.zeros_like(inp)
                for role, color in enumerate((d['structure'], d['wire'], d['parcel']), 1):
                    semantic[inp == color] = role
                fg = np.argwhere(inp != 0)
                lo, hi = fg.min(0), fg.max(0)
                structures.add(_key(semantic[lo[0]:hi[0]+1, lo[1]:hi[1]+1]))
                graph_forms.add((len(d['entities']), len(d['edges']), len(d['docks']), len(d['turns']),
                                 tuple(sorted(len(x) for x in d['adjacency'])), len(d['selected']['nodes']) - 2))
                directions.update(f['out'] for f in d['docks'].values())
                states.add((d['span'], d['length'], d['selected']['span'], d['selected']['length']))
    assert len(puzzles) == 42 and len(structures) >= 150 and len(shapes) >= 25
    assert len(palettes) >= 25 and len(graph_forms) >= 60 and directions == set(DIRS)
    assert len(states) >= 10
    # Force the otherwise rare constructive branch for every witness class.
    for seed in range(6):
        rng = np.random.default_rng(seed)
        for purpose in ('fit', 'distance', 'test', 'free'):
            pair = _fallback_pair(rng, (1, 2, 3), purpose)
            assert np.array_equal(solve(pair['input']), pair['output'])
    # Counts, corruption, and parser failures are semantic checks, not only schema checks.
    for nt, ne in ((2, 1), (10, 4), (5, 3)):
        validate_generated_puzzle(generate_puzzle(981 + nt, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(1, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    puzzle = generate_puzzle(77, 2, 1)
    inp = puzzle['test'][0]['input']
    d = _read_input(inp)
    f = d['docks'][d['selected']['dock']]
    puzzle['test'][0]['output'][min(_dock_cells(f, d['selected']['length']))] = 5
    try:
        validate_generated_puzzle(puzzle)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted final parcel accepted')
    # Explicit graph checks distinguish tie rejection and a rotation-sensitive gate.
    adj = _adjacency(6, [(0, 1), (1, 2), (2, 4), (0, 3), (3, 5)])
    paths = _routes(adj, 0, {4: {'span': 2}, 5: {'span': 3}}, {1}, 1, 2)
    assert len(paths) == 1 and paths[0]['dock'] == 4 and paths[0]['length'] == 1
    assert not _routes(adj, 0, {4: {'span': 2}, 5: {'span': 3}}, set(), 1, 2)
    tied = _routes(_adjacency(5, [(0, 1), (1, 3), (0, 2), (2, 4)]),
                   0, {3: {'span': 1}, 4: {'span': 1}}, set(), 1, 2)
    try:
        _choose(tied)
    except AssertionError:
        pass
    else:
        raise AssertionError('tied routing accepted')
    return {'seeds': 42, 'distinct_puzzles': len(puzzles), 'structures': len(structures),
            'shapes': len(shapes), 'palettes': len(palettes), 'graph_forms': len(graph_forms),
            'parcel_states': len(states)}


if __name__ == '__main__':
    print(self_test())
