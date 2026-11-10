"""Shortest routes through composed color relations.

The input contains a five-cell-wide cross source, a five-cell-wide square
target, centered three-cell-wide relation tokens, and neutral wires. A token
X->Y can transmit X as Z only when another original token Y->Z exists.
Find the unique fewest-token transmitting route. See task_spec.md for the
complete visible conventions; no generation parameters are used by solve.
"""
from collections import deque
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _components(cells, directions=D4):
    todo, result = set(cells), []
    while todo:
        p = min(todo)
        todo.remove(p)
        part, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _support(center, radius, kind):
    r, c = center
    return {(r + dr, c + dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1)
            if kind == 'square' or dr == 0 or dc == 0}


def _pixel_graph(cells):
    # Diagonal stretches count, but do not invent a shortcut across an elbow.
    return {p: {q for dr, dc in D8 if (q := (p[0] + dr, p[1] + dc)) in cells
                and not (dr and dc and ((p[0] + dr, p[1]) in cells
                                       or (p[0], p[1] + dc) in cells))}
            for p in cells}


def _parse_with_wire(grid, background, wire):
    components = _components(map(tuple, np.argwhere((grid != background) & (grid != wire))))
    nodes = []
    for cells in components:
        low = np.min(list(cells), axis=0)
        high = np.max(list(cells), axis=0)
        span = high - low + 1
        assert span[0] == span[1] and span[0] in (3, 5)
        center = tuple(map(int, (low + high) // 2))
        radius = int(span[0] // 2)
        forms = [k for k in ('square', 'cross') if cells == _support(center, radius, k)]
        assert len(forms) == 1
        bodies = {int(grid[p]) for p in cells if p != center}
        assert len(bodies) == 1
        nodes.append(dict(cells=cells, center=center, radius=radius, kind=forms[0],
                          body=bodies.pop(), value=int(grid[center])))
    terminals = [i for i, n in enumerate(nodes) if n['radius'] == 2]
    assert len(terminals) == 2
    sources = [i for i in terminals if nodes[i]['kind'] == 'cross']
    targets = [i for i in terminals if nodes[i]['kind'] == 'square']
    assert len(sources) == len(targets) == 1
    start, target = sources[0], targets[0]
    marker = nodes[start]['body']
    assert nodes[target]['body'] == marker
    ordinary = [i for i, n in enumerate(nodes) if n['radius'] == 1]
    assert len(ordinary) >= 3
    for i, n in enumerate(nodes):
        assert n['value'] not in (background, wire, marker)
        if i in ordinary:
            assert n['body'] not in (background, wire, marker)
    mapping = {}
    for i in ordinary:
        n = nodes[i]
        assert n['body'] not in mapping or mapping[n['body']] == n['value'], 'conflicting donors'
        mapping[n['body']] = n['value']
    for i in ordinary:
        n = nodes[i]
        n['donors'] = [j for j in ordinary if j != i and nodes[j]['body'] == n['value']]
        n['composed'] = nodes[n['donors'][0]]['value'] if n['donors'] else None
    node_at = {p: i for i, n in enumerate(nodes) for p in n['cells']}
    wires = _components(map(tuple, np.argwhere(grid == wire)), D8)
    assert wires
    edges, adjacency = [], [[] for _ in nodes]
    for e, cells in enumerate(wires):
        pg = _pixel_graph(cells)
        assert len(cells) >= 2 and all(1 <= len(v) <= 2 for v in pg.values())
        ends = {p for p, v in pg.items() if len(v) == 1}
        assert len(ends) == 2, 'wire is closed or branched'
        touching = {node_at[q] for r, c in cells for dr, dc in D8
                    if (q := (r + dr, c + dc)) in node_at}
        assert len(touching) == 2, 'wire must connect exactly two tokens'
        for p in ends:
            assert any((p[0] + dr, p[1] + dc) in node_at for dr, dc in D8)
        a, b = sorted(touching)
        edges.append((a, b))
        adjacency[a].append((b, e))
        adjacency[b].append((a, e))
    assert len(set(edges)) == len(edges), 'parallel edges are outside this family'
    assert all(len(a) <= 4 for a in adjacency)
    reached, stack = {start}, [start]
    while stack:
        for b, _ in adjacency[stack.pop()]:
            if b not in reached:
                reached.add(b)
                stack.append(b)
    assert target in reached
    assert all(i in reached or not adjacency[i] for i in ordinary), 'disconnected wire network'
    return dict(background=background, wire=wire, marker=marker, nodes=nodes,
                ordinary=ordinary, start=start, target=target, wires=wires,
                edges=edges, adjacency=adjacency)


def _read_input(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    background = int(colors[np.argmax(counts)])
    candidates = []
    for wire in colors:
        if wire == background:
            continue
        try:
            candidates.append(_parse_with_wire(grid, background, int(wire)))
        except AssertionError:
            pass
    assert len(candidates) == 1, 'visible roles must have one interpretation'
    return grid, candidates[0]


def _state(data, node, composed=True):
    if node == data['start']:
        return data['nodes'][node]['value']
    n = data['nodes'][node]
    return n['composed'] if composed else n['value']


def _can_step(data, a, b, composed=True):
    if b == data['start'] or a == data['target']:
        return False
    state = _state(data, a, composed)
    if state is None:
        return False
    n = data['nodes'][b]
    if b == data['target']:
        return state == n['value']
    return state == n['body'] and (n['composed'] is not None or not composed)


def _shortest(data, compatible=True, composed=True):
    """Unit station cost, counting shortest-route multiplicity explicitly."""
    start, target = data['start'], data['target']
    distance, ways, previous = {start: 0}, {start: 1}, {}
    queue = deque([start])
    while queue:
        a = queue.popleft()
        if a == target:
            continue
        for b, e in data['adjacency'][a]:
            if b == start or (compatible and not _can_step(data, a, b, composed)):
                continue
            if b not in distance:
                distance[b] = distance[a] + 1
                ways[b], previous[b] = 0, (a, e)
                queue.append(b)
            if distance[b] == distance[a] + 1:
                ways[b] = min(2, ways[b] + ways[a])
    if ways.get(target) != 1:
        return None
    nodes, edges = [target], []
    while nodes[-1] != start:
        a, e = previous[nodes[-1]]
        nodes.append(a)
        edges.append(e)
    return nodes[::-1], edges[::-1]


def _paths(data):
    result, stack = [], [(data['start'], [data['start']], [])]
    while stack:
        a, nodes, edges = stack.pop()
        if a == data['target']:
            result.append((nodes, edges))
            continue
        for b, e in data['adjacency'][a]:
            if b not in nodes:
                stack.append((b, nodes + [b], edges + [e]))
    return result


def _compatible(data, path):
    return all(_can_step(data, a, b) for a, b in zip(path[0], path[0][1:]))


def _paint(grid, data, route):
    selected, edges = route
    result = grid.copy()
    selected_set = set(selected)
    for i in data['ordinary']:
        n = data['nodes'][i]
        if i in selected_set:
            result[n['center']] = n['composed']
        else:
            for p in n['cells']:
                result[p] = data['background']
    for a, e in zip(selected, edges):
        for p in data['wires'][e]:
            result[p] = _state(data, a)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid, data = _read_input(input_grid)
    route = _shortest(data)
    assert route is not None, 'no unique minimum compatible route'
    return _paint(grid, data, route)


def _metrics(grid):
    _, d = _read_input(grid)
    route = _shortest(d)
    assert route is not None
    paths = _paths(d)
    valid = [p for p in paths if _compatible(d, p)]
    valid.sort(key=lambda p: len(p[0]))
    assert valid and (len(valid) == 1 or len(valid[0][0]) < len(valid[1][0]))
    assert valid[0] == route
    selected = set(route[0][1:-1])
    area = lambda p: sum(len(d['wires'][e]) for e in p[1])
    remote, retained = 0, 0
    for i in selected:
        donors = d['nodes'][i]['donors']
        adjacent = {j for j, _ in d['adjacency'][i]}
        remote += int(all(j not in adjacent and j not in selected for j in donors))
        retained += int(any(j in selected and d['nodes'][j]['value'] != d['nodes'][j]['composed']
                            for j in donors))
    low = np.min([n['center'] for n in d['nodes']], axis=0)
    layout = tuple((n['center'][0] - int(low[0]), n['center'][1] - int(low[1]),
                    n['radius'], n['kind']) for n in d['nodes'])
    return dict(shape=grid.shape, tokens=len(d['ordinary']), selected=len(selected),
                route_count=len(paths), valid_routes=len(valid),
                gate_conflict=min(len(p[0]) for p in paths) < len(route[0]),
                area_conflict=any(len(p[0]) > len(route[0]) and area(p) < area(route) for p in valid),
                remote_donors=remote, retained_donors=retained,
                compatible_deleted=sum(i not in selected and d['nodes'][i]['composed'] is not None
                                       for i in d['ordinary']),
                missing_donors=sum(d['nodes'][i]['composed'] is None for i in d['ordinary']),
                monochrome=sum(d['nodes'][i]['body'] == d['nodes'][i]['value'] for i in d['ordinary']),
                repeated_bodies=len(d['ordinary']) - len({d['nodes'][i]['body'] for i in d['ordinary']}),
                junctions=sum(len(a) >= 3 for a in d['adjacency']),
                dead_ends=sum(len(a) == 1 for a in d['adjacency']),
                isolates=sum(not a for a in d['adjacency']),
                loops=len(d['edges']) - sum(bool(a) for a in d['adjacency']) + 1,
                nonstraight_wires=sum(len(cells) != int(np.prod(np.max(list(cells), axis=0)
                                      - np.min(list(cells), axis=0) + 1)) for cells in d['wires']),
                diagonal_steps=sum(sum(p[0] != q[0] and p[1] != q[1] for q in adjacent)
                                   for cells in d['wires'] for p, adjacent in _pixel_graph(cells).items()) // 2,
                kinds=tuple(sorted({d['nodes'][i]['kind'] for i in d['ordinary']})),
                palette=(d['background'], d['wire'], d['marker']), layout=layout,
                changed_centers=sum(d['nodes'][i]['value'] != d['nodes'][i]['composed'] for i in selected))


def _line(a, b):
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [tuple(int(round(a[k] + (b[k] - a[k]) * t / steps)) for k in (0, 1))
            for t in range(steps + 1)] if steps else [a]


def _render(rng, nr, nc, active, edges, visible, start, target, relations, codes,
            palette, count_witness=False, straight=False):
    """Embed a lattice network, then rediscover every edge from actual pixels."""
    if count_witness:
        rows = [2, int(rng.integers(13, 16)), int(rng.integers(21, 23))]
        if nc == 4:
            gaps = rng.integers(6, 9, size=3)
        else:
            gaps = np.full(4, 6)
    else:
        rows = [2]
        gaps_r = rng.integers(6, 9, size=nr - 1)
        while int(sum(gaps_r)) > 24:
            gaps_r[int(np.argmax(gaps_r))] -= 1
        rows += [2 + int(v) for v in np.cumsum(gaps_r)]
        gaps = rng.integers(6, 9, size=nc - 1)
    while int(sum(gaps)) > 24:
        gaps[int(np.argmax(gaps))] -= 1
    cols = [2] + [2 + int(v) for v in np.cumsum(gaps)]
    centers = {v: (rows[v[0]], cols[v[1]]) for v in active | visible}
    # Free donor tokens share the geometric vocabulary but have no wire.
    kinds = {v: ('cross' if v == start else 'square') if v in (start, target)
             else str(rng.choice(['square', 'cross'])) for v in visible}
    supports = {v: _support(centers[v], 2 if v in (start, target) else 1, kinds[v]) for v in visible}
    occupied = set().union(*supports.values())
    wire_cells = set()
    for a, b in edges:
        pa, pb = centers[a], centers[b]
        direction = (int(np.sign(pb[0] - pa[0])), int(np.sign(pb[1] - pa[1])))
        qa = (pa[0] + 3 * direction[0], pa[1] + 3 * direction[1])
        qb = (pb[0] - 3 * direction[0], pb[1] - 3 * direction[1])
        points = [pa, qa]
        if not straight and max(abs(qb[0] - qa[0]), abs(qb[1] - qa[1])) >= 2 and rng.random() < .65:
            middle = ((qa[0] + qb[0]) // 2, (qa[1] + qb[1]) // 2)
            shift = int(rng.choice([-1, 1]))
            middle = (middle[0], middle[1] + shift) if direction[0] else (middle[0] + shift, middle[1])
            points.append(middle)
        points += [qb, pb]
        for p, q in zip(points, points[1:]):
            wire_cells.update(_line(p, q))
    wire_cells -= occupied
    foreground = wire_cells | occupied
    low, high = np.min(list(foreground), axis=0), np.max(list(foreground), axis=0)
    base_shape = high - low + 1
    assert max(base_shape) <= 30
    padding = [int(rng.integers(min(4, 31 - int(n)))) for n in base_shape]
    shape = tuple(int(n) + p for n, p in zip(base_shape, padding))
    if rng.random() < .07:
        shape = (30, 30)
    offset = tuple(int(rng.integers(shape[k] - int(base_shape[k]) + 1)) - int(low[k]) for k in (0, 1))
    bg, wire, marker, *identities = palette
    grid = np.full(shape, bg, dtype=np.int8)
    move = lambda p: (p[0] + offset[0], p[1] + offset[1])
    for p in wire_cells:
        grid[move(p)] = wire
    for v in visible:
        body, value = (marker, identities[codes[0 if v == start else 1]]) if v in (start, target) else (
            identities[relations[v][0]], identities[relations[v][1]])
        for p in supports[v]:
            grid[move(p)] = body
        grid[move(centers[v])] = value
    before = _parse_with_wire(grid, bg, wire)
    actual_edges = {tuple(sorted((before['nodes'][a]['center'], before['nodes'][b]['center'])))
                    for a, b in before['edges']}
    expected_edges = {tuple(sorted((move(centers[a]), move(centers[b]))))
                      for a, b in _compressed(active, edges, visible & active)}
    assert actual_edges == expected_edges, 'rendered topology differs from construction'
    grid = np.rot90(grid, int(rng.integers(4)))
    if rng.random() < .5:
        grid = np.fliplr(grid)
    grid = grid.copy()
    _, data = _read_input(grid)
    # Suppressed degree-two lattice vertices are wire bends, not extra stations.
    original_adj = {v: [] for v in active}
    for a, b in edges:
        original_adj[a].append(b)
        original_adj[b].append(a)
    expected_degrees = sorted(len(original_adj.get(v, [])) for v in visible)
    assert sorted(map(len, data['adjacency'])) == expected_degrees
    return grid


def _lattice(rng, long=False, richness=2):
    nr = int(rng.choice([3, 4])) if long else int(rng.choice([2, 3, 4], p=[.12, .63, .25]))
    nc = int(rng.choice([3, 4]))
    vertices = [(r, c) for r in range(nr) for c in range(nc)]
    candidates = [(v, (v[0] + dr, v[1] + dc)) for v in vertices for dr, dc in ((1, 0), (0, 1))
                  if v[0] + dr < nr and v[1] + dc < nc]
    active = {vertices[int(rng.integers(len(vertices)))]}
    edges = []
    desired = int(rng.integers(max(6, len(vertices) - 5), len(vertices) + 1))
    while len(active) < desired:
        boundary = [(a, b) for a, b in candidates if (a in active) != (b in active)]
        a, b = boundary[int(rng.integers(len(boundary)))]
        edges.append((a, b))
        active.update((a, b))
    extras = [(a, b) for a, b in candidates if a in active and b in active and (a, b) not in edges]
    rng.shuffle(extras)
    edges += extras[:int(rng.integers(1, min(len(extras), richness + 2) + 1))] if extras else []
    degrees = {v: sum(v in e for e in edges) for v in active}
    visible = {v for v in active if degrees[v] != 2 or rng.random() > .28}
    return nr, nc, active, edges, visible


def _compressed(active, edges, visible):
    adj = {v: [] for v in active}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    compressed = set()
    for a in visible:
        for nxt in adj[a]:
            previous, current = a, nxt
            seen = {a}
            while current not in visible:
                assert current not in seen
                seen.add(current)
                options = [v for v in adj[current] if v != previous]
                assert len(options) == 1
                previous, current = current, options[0]
            assert current != a
            compressed.add(tuple(sorted((a, current))))
    assert len(compressed) * 2 == sum(len(adj[v]) for v in visible), 'parallel suppressed routes'
    return sorted(compressed)


def _abstract_paths(visible, edges, start, target):
    adj = {v: [] for v in visible}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    result, stack = [], [[start]]
    while stack:
        p = stack.pop()
        if p[-1] == target:
            result.append(p)
        else:
            stack.extend(p + [v] for v in adj[p[-1]] if v not in p)
    return result


def _count_network(rng):
    nc = int(rng.choice([4, 5]))
    start, target = (1, 0), (1, nc - 1)
    sparse = [(0, 1)] if nc == 4 else [(0, 1), (0, 3)]
    dense = [(2, c) for c in range(nc)]
    upper = [start] + [(0, c) for c in range(nc)] + [target]
    lower = [start] + dense + [target]
    active = set(upper + lower)
    edges = list(zip(upper, upper[1:])) + list(zip(lower, lower[1:]))
    visible = {start, target, *sparse, *dense}
    # A dangling branch independently adds a junction and a compatible decoy.
    c = int(rng.integers(1, nc - 1))
    branch = (1, c)
    edges.append(((2, c), branch))
    active.add(branch)
    visible.add(branch)
    return 3, nc, active, edges, visible, start, target, [start] + sparse + [target]


def _assign(rng, nr, nc, active, visible, start, target, path, count_witness=False):
    # Five live identities at most leave a sixth for a missing-donor relation.
    if count_witness:
        mapping = {0: 1, 1: 2, 2: 0}
    else:
        kind = int(rng.integers(4))
        if kind == 0:
            mapping = {0: 1, 1: 2, 2: 3, 3: 1}  # tail into a cycle
        else:
            size = kind + 2
            mapping = {i: (i + 1) % size for i in range(size)}
    relations = {v: (int(rng.choice(list(mapping))), 0) for v in visible if v not in (start, target)}
    state = 0
    for v in path[1:-1]:
        relations[v] = (state, mapping[state])
        state = mapping[mapping[state]]
    target_state = state
    selected = set(path[1:-1])
    for v in relations:
        if v not in selected:
            b = relations[v][0]
            relations[v] = (b, mapping[b])
    if count_witness:
        state = 0
        for v in sorted(v for v in relations if v[0] == 2):
            relations[v] = (state, mapping[state])
            state = mapping[mapping[state]]
        assert state == target_state
    # Supply any missing live identity using an off-route node or an isolated
    # token in an unused lattice slot. These donors are visible evidence.
    required = {relations[v][1] for v in selected}
    spare = [v for v in sorted(relations) if v not in selected]
    rng.shuffle(spare)
    free = [(r, c) for r in range(nr) for c in range(nc) if (r, c) not in active]
    rng.shuffle(free)
    for b in sorted(required):
        if any(body == b for body, _ in relations.values()):
            continue
        # Do not overwrite the sole donor of a previously satisfied identity.
        choices = [v for v in spare if relations[v][0] not in required
                   or sum(body == relations[v][0] for body, _ in relations.values()) > 1]
        if choices:
            v = choices[0]
            spare.remove(v)
        else:
            assert free, 'not enough donor space'
            v = free.pop()
            visible.add(v)
        relations[v] = (b, mapping[b])
    # When space permits, include an explicitly nontransmitting orphan 5->4.
    # Its missing donor is role-independent; it is never silently made usable.
    if free and 4 not in mapping and rng.random() < .6:
        v = free.pop()
        visible.add(v)
        relations[v] = (5, 4)
    return relations, (0, target_state)


def _make_pair(rng, mode, palette, richness):
    for attempt in range(180):
        try:
            if mode == 'count':
                nr, nc, active, edges, visible, start, target, path = _count_network(rng)
            else:
                nr, nc, active, edges, visible = _lattice(rng, mode == 'long', richness)
                assert len(visible) >= 6
                ce = _compressed(active, edges, visible)
                order = list(sorted(visible))
                rng.shuffle(order)
                choices = []
                for start in order[:6]:
                    for target in order[-6:]:
                        if target == start:
                            continue
                        paths = _abstract_paths(visible, ce, start, target)
                        if not paths:
                            continue
                        minimum = min(len(p) for p in paths)
                        for p in paths:
                            k = len(p) - 2
                            if ((mode == 'gate' and 2 <= k <= 3 and minimum < len(p))
                                    or (mode == 'random' and 1 <= k <= 3)
                                    or (mode == 'long' and 4 <= k <= 6)):
                                choices.append((start, target, p))
                assert choices
                start, target, path = choices[int(rng.integers(len(choices)))]
            relations, codes = _assign(rng, nr, nc, active, visible, start, target, path, mode == 'count')
            grid = _render(rng, nr, nc, active, edges, visible, start, target,
                           relations, codes, palette, mode == 'count')
            f = _metrics(grid)
            assert f['changed_centers'] and f['compatible_deleted']
            if mode == 'gate':
                assert f['gate_conflict'] and 2 <= f['selected'] <= 3
                assert f['retained_donors'] and f['remote_donors']
            elif mode == 'count':
                assert f['area_conflict'] and f['remote_donors'] and 1 <= f['selected'] <= 2
            elif mode == 'long':
                assert 4 <= f['selected'] <= 6
            else:
                assert 1 <= f['selected'] <= 3
            return dict(input=grid, output=solve(grid))
        except AssertionError:
            continue
    return _fallback_pair(rng, mode, palette)


def _fallback_pair(rng, mode, palette):
    """Constructive backup with the same semantics, never an altered rule."""
    if mode in ('count', 'random'):
        nr, nc, active, edges, visible, start, target, path = _count_network(rng)
        relations, codes = _assign(rng, nr, nc, active, visible, start, target, path, True)
    else:
        nc = 4 if mode == 'gate' else int(rng.choice([4, 5]))
        nr, start, target = 3, (1, 0), (1, nc - 1)
        upper = [start] + [(0, c) for c in range(nc)] + [target]
        lower = [start] + [(2, c) for c in range(nc)] + [target]
        branch = (1, 1)
        edges = list(zip(upper, upper[1:])) + list(zip(lower, lower[1:]))
        edges.append(((2, 1), branch))
        active = set(upper + lower + [branch])
        visible = {start, target, (0, 1), branch, *((2, c) for c in range(nc))}
        if mode == 'gate':
            visible.remove((2, 2))
            mapping = {0: 1, 1: 2, 2: 3, 3: 1}
        else:
            mapping = {0: 1, 1: 2, 2: 0}
        relations = {branch: (1, 2), (0, 1): (3, 1) if mode == 'gate' else (1, 2)}
        state = 0
        for v in sorted(v for v in visible if v[0] == 2):
            relations[v] = (state, mapping[state])
            state = mapping[mapping[state]]
        codes = (0, state)
    grid = _render(rng, nr, nc, active, edges, visible, start, target,
                   relations, codes, palette, True, straight=True)
    f = _metrics(grid)
    assert f['compatible_deleted'] and f['changed_centers']
    if mode == 'gate':
        assert f['gate_conflict'] and f['remote_donors'] and f['retained_donors']
        assert f['selected'] == 3
    elif mode == 'count':
        assert f['area_conflict'] and f['remote_donors']
    elif mode == 'long':
        assert 4 <= f['selected'] <= 5
    return dict(input=grid, output=solve(grid))


def _key(grid):
    grid = np.ascontiguousarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    base_palette = tuple(map(int, rng.permutation(10)))
    varied_roles = bool(rng.random() < .7)
    richness = int(rng.integers(1, 4))
    modes = ['gate', 'count'] + ['random'] * (nt - 2)
    rng.shuffle(modes)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            for _ in range(8):
                palette = tuple(map(int, rng.permutation(10))) if varied_roles else (
                    base_palette[:3] + tuple(map(int, rng.permutation(base_palette[3:]))))
                mode = modes[i] if split == 'train' else 'long'
                pair = _make_pair(rng, mode, palette, richness)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input construction failed')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            grid, output = _grid(pair['input']), _grid(pair['output'])
            assert grid.shape == output.shape
            key = _key(grid)
            assert key not in seen
            seen.add(key)
            _, data = _read_input(grid)
            route = _shortest(data)
            assert route is not None
            f = _metrics(grid)
            assert 1 <= f['selected'] < f['tokens'] and f['changed_centers']
            assert np.array_equal(output, _paint(grid, data, route)), 'route, composition or preservation mismatch'
            # Explicit footprint and signal checks, in addition to full equality.
            for i in (data['start'], data['target']):
                assert all(output[p] == grid[p] for p in data['nodes'][i]['cells'])
            for a, e in zip(route[0], route[1]):
                assert {int(output[p]) for p in data['wires'][e]} == {_state(data, a)}
            for e in set(range(len(data['edges']))) - set(route[1]):
                assert all(output[p] == data['wire'] for p in data['wires'][e])
            assert np.all(output[grid == data['background']] == data['background'])
            features[split].append(f)
    train, test = features['train'], features['test']
    assert any(f['gate_conflict'] and f['retained_donors'] and f['remote_donors'] for f in train)
    assert any(f['area_conflict'] and f['valid_routes'] >= 2 for f in train)
    assert all(f['compatible_deleted'] for f in train + test)
    assert max(f['selected'] for f in train) <= 3
    assert min(f['selected'] for f in test) >= 4


def self_test():
    """Bounded pixel-derived checks over 42 seeds, plus negative cases."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signature = lambda p: tuple(_key(q[k]) for s in ('train', 'test') for q in p[s]
                                for k in ('input', 'output'))
    assert len({signature(p) for p in puzzles}) == 42
    assert signature(puzzles[0]) == signature(generate_puzzle(0))
    features = [_metrics(q['input']) for p in puzzles for s in ('train', 'test') for q in p[s]]
    assert len({f['shape'] for f in features}) >= 30
    assert len({f['layout'] for f in features}) >= 100
    assert len({f['palette'] for f in features}) >= 20
    assert len({f['tokens'] for f in features}) >= 6
    assert {1, 2, 3, 4, 5, 6} <= {f['selected'] for f in features}
    assert any(f['loops'] >= 2 for f in features)
    assert any(f['dead_ends'] and f['junctions'] for f in features)
    assert any(f['isolates'] and f['missing_donors'] for f in features)
    assert any(len(f['kinds']) == 2 for f in features)
    assert any(f['shape'] == (30, 30) for f in features)
    # Counterfactual outputs show why both inherited mechanisms matter.
    for puzzle in puzzles:
        gate_seen = count_seen = simultaneous_seen = False
        for pair in puzzle['train']:
            grid, d = _read_input(pair['input'])
            route = _shortest(d)
            valid = [p for p in _paths(d) if _compatible(d, p)]
            if min(len(p[0]) for p in _paths(d)) < len(route[0]):
                gate_seen = True  # Unconditioned station search has a different answer.
            for p in valid:
                if len(p[0]) > len(route[0]) and sum(len(d['wires'][e]) for e in p[1]) < sum(
                        len(d['wires'][e]) for e in route[1]):
                    assert not np.array_equal(_paint(grid, d, p), pair['output'])
                    count_seen = True  # Wire length and arbitrary compatible routing fail.
            selected = set(route[0][1:-1])
            for i in selected:
                n = d['nodes'][i]
                for j in n['donors']:
                    donor = d['nodes'][j]
                    if j in selected and donor['composed'] != donor['value']:
                        assert pair['output'][n['center']] != donor['composed']
                        simultaneous_seen = True
            assert any(i not in selected and d['nodes'][i]['composed'] is not None
                       for i in d['ordinary'])  # Updating every usable token fails.
        assert gate_seen and count_seen and simultaneous_seen
    rng = np.random.default_rng(170)
    for mode in ('gate', 'count', 'random', 'long'):
        q = _fallback_pair(rng, mode, tuple(range(10)))
        assert np.array_equal(solve(q['input']), q['output'])
    # The input-only solver also supports consistent maps outside the sampled
    # cycle/tail distribution: return cycles and monochrome fixed points.
    for mapping in ({0: 1, 1: 0}, {0: 1, 1: 2, 2: 2}):
        nr, nc, active, edges, visible, start, target, _ = _count_network(rng)
        relations = {v: (1, mapping[1]) for v in visible if v not in (start, target)}
        state = 0
        for v in sorted(v for v in visible if v[0] == 2):
            relations[v] = (state, mapping[state])
            state = mapping[mapping[state]]
        grid = _render(rng, nr, nc, active, edges, visible, start, target,
                       relations, (0, state), tuple(range(10)), True, straight=True)
        _, d = _read_input(grid)
        route = _shortest(d)
        assert route is not None and len(route[0]) - 2 == nc
        predicted = solve(grid)
        for i in route[0][1:-1]:
            n = d['nodes'][i]
            assert predicted[n['center']] == n['composed']
        if len(mapping) == 2:
            assert all(predicted[d['nodes'][i]['center']] == d['nodes'][i]['body']
                       for i in route[0][1:-1])
        else:
            assert sum(d['nodes'][i]['body'] == d['nodes'][i]['value']
                       for i in route[0][1:-1]) >= 3
    # A visible diamond with two equally short transmitting paths is rejected.
    active = {(0, 0), (0, 1), (1, 0), (1, 1)}
    edges = [((0, 0), (0, 1)), ((0, 0), (1, 0)),
             ((0, 1), (1, 1)), ((1, 0), (1, 1))]
    visible = active | {(0, 2)}
    tied = _render(rng, 2, 3, active, edges, visible, (0, 0), (1, 1),
                   {(0, 1): (0, 1), (1, 0): (0, 1), (0, 2): (1, 2)},
                   (0, 2), tuple(range(10)), straight=True)
    try:
        solve(tied)
    except AssertionError:
        pass
    else:
        raise AssertionError('tied route accepted')
    inconsistent = tied.copy()
    _, d = _read_input(inconsistent)
    twins = [i for i in d['ordinary'] if d['nodes'][i]['body'] == 3]
    assert len(twins) == 2
    inconsistent[d['nodes'][twins[0]]['center']] = 5
    try:
        _read_input(inconsistent)
    except AssertionError:
        pass
    else:
        raise AssertionError('conflicting donor values accepted')
    for counts in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(900 + sum(counts), *counts)
        assert (len(p['train']), len(p['test'])) == counts
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Corruption outside the route is caught, rather than tolerated as noise.
    bad = {s: [dict(input=q['input'].copy(), output=q['output'].copy()) for q in puzzles[0][s]]
           for s in ('train', 'test')}
    _, d = _read_input(bad['train'][0]['input'])
    cell = d['nodes'][d['start']]['center']
    bad['train'][0]['output'][cell] = d['background']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('terminal corruption accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds)')
