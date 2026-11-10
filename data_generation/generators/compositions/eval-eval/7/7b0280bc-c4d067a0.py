"""Selectively enlarge the unique minimum-station network joining built stations.

Only NumPy and the standard library are used. No saved examples, RNG state,
training labels, or puzzle-level color dictionary enter solve().
"""
from collections import Counter, deque
from itertools import combinations
from math import gcd
import heapq

import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


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
            for dr, dc in _D4:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(cells)
    return result


def _bbox(cells):
    rr, cc = zip(*cells)
    return min(rr), min(cc), max(rr) + 1, max(cc) + 1


def _solid_square(grid, cells):
    r, c, b, e = _bbox(cells)
    return (b - r >= 2 and b - r == e - c and len(cells) == (b-r)**2
            and np.all(grid[r:b, c:e] == grid[r, c]))


def _adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for e, (a, b) in enumerate(edges):
        adj[a].append((b, e))
        adj[b].append((a, e))
    return adj


def _blueprint(mini, bg):
    """Decode the graph from pixels; bends have wire color even at lattice sites."""
    h, w = mini.shape
    assert h % 2 == w % 2 == 1 and min(h, w) >= 5
    off_lattice = np.ones(mini.shape, dtype=bool)
    off_lattice[::2, ::2] = False
    wire_colors = set(map(int, mini[off_lattice & (mini != bg)]))
    assert len(wire_colors) == 1, 'one visible connector color is required'
    wire = wire_colors.pop()
    station_mask = (mini != bg) & (mini != wire)
    assert not np.any(station_mask & off_lattice)
    pixels = list(map(tuple, np.argwhere(station_mask).tolist()))
    assert 6 <= len(pixels) <= 18
    coords = [(r//2, c//2) for r, c in pixels]
    colors = [int(mini[p]) for p in pixels]
    ids = {p: i for i, p in enumerate(pixels)}
    wires = _components(mini == wire)
    edges = []
    for cells in wires:
        touching = set()
        for r, c in cells:
            for dr, dc in _D4:
                if (r+dr, c+dc) in ids:
                    touching.add(ids[r+dr, c+dc])
        assert len(touching) == 2, 'a connector must have exactly two stations'
        # No off-station branching, cycles, or ambiguous self-contact.
        cellset = set(cells)
        for r, c in cells:
            degree = sum((r+dr, c+dc) in cellset or (r+dr, c+dc) in ids
                         for dr, dc in _D4)
            assert degree == 2, 'only station cells may be junctions'
        edges.append(tuple(sorted(touching)))
    assert len(set(edges)) == len(edges), 'parallel compressed edges are excluded'
    adj = _adjacency(len(coords), edges)
    assert all(1 <= len(a) <= 4 for a in adj)
    seen, stack = {0}, [0]
    while stack:
        for b, _ in adj[stack.pop()]:
            if b not in seen:
                seen.add(b)
                stack.append(b)
    assert len(seen) == len(coords)
    assert 0 <= len(edges)-len(coords)+1 <= 3
    return dict(coords=coords, colors=colors, edges=edges, wires=wires, adj=adj,
                wire=wire, lattice=((h+1)//2, (w+1)//2))


def _shortest_two(data, terminals):
    a, target = terminals
    adj = data['adj']
    distance, ways, previous = [-1]*len(adj), [0]*len(adj), [None]*len(adj)
    distance[a], ways[a] = 0, 1
    queue = deque([a])
    while queue:
        x = queue.popleft()
        for y, e in adj[x]:
            if distance[y] < 0:
                distance[y] = distance[x]+1
                previous[y] = x, e
                queue.append(y)
            if distance[y] == distance[x]+1:
                ways[y] = min(2, ways[y]+ways[x])
    if ways[target] != 1:
        return None
    chosen, x = set(), target
    while x != a:
        x, e = previous[x]
        chosen.add(e)
    return frozenset(chosen)


def _minimum_tree(data, terminals):
    """Exact search. Every terminal tree extends to a full spanning tree.

    Delete cycle-rank edges, keep spanning trees, then prune nonterminal leaves.
    Different spanning trees may give the same answer; deduplicate edge sets.
    Ties between distinct minimum connecting networks are rejected.
    """
    terminals = tuple(sorted(terminals))
    if len(terminals) == 2:
        return _shortest_two(data, terminals)
    n, m = len(data['coords']), len(data['edges'])
    rank = m-n+1
    assert 0 <= rank <= 3
    best_cost, best = m+1, set()
    terminal_set = set(terminals)
    for removed_tuple in combinations(range(m), rank):
        removed = set(removed_tuple)
        parent = list(range(n))

        def root(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        retained, valid = set(), True
        for e, (a, b) in enumerate(data['edges']):
            if e in removed:
                continue
            ra, rb = root(a), root(b)
            if ra == rb:
                valid = False
                break
            parent[ra] = rb
            retained.add(e)
        if not valid:
            continue
        degree = [sum(e in retained for _, e in neighbors) for neighbors in data['adj']]
        queue = deque(i for i in range(n) if degree[i] == 1 and i not in terminal_set)
        while queue:
            a = queue.popleft()
            if degree[a] != 1:
                continue
            b, e = next((b, e) for b, e in data['adj'][a] if e in retained)
            retained.remove(e)
            degree[a] -= 1
            degree[b] -= 1
            if degree[b] == 1 and b not in terminal_set:
                queue.append(b)
        cost = len(retained)  # tree vertices = edges + 1; terminals are fixed
        key = frozenset(retained)
        if cost < best_cost:
            best_cost, best = cost, {key}
        elif cost == best_cost and len(best) < 2:
            best.add(key)
    return next(iter(best)) if len(best) == 1 else None


def _selected_nodes(data, chosen):
    return {v for e in chosen for v in data['edges'][e]}


def _distance_conflict(data, terminals, chosen):
    if len(terminals) != 2:
        return False
    weights = [len(w)+1 for w in data['wires']]
    a, target = terminals
    distances = [float('inf')]*len(data['coords'])
    distances[a] = 0
    queue = [(0, a)]
    while queue:
        cost, x = heapq.heappop(queue)
        if cost != distances[x]:
            continue
        for y, e in data['adj'][x]:
            if cost+weights[e] < distances[y]:
                distances[y] = cost+weights[e]
                heapq.heappush(queue, (distances[y], y))
    return sum(weights[e] for e in chosen) > distances[target]


def _registrations(data, offsets, colors=None):
    by_coord = {p: i for i, p in enumerate(data['coords'])}
    found = []
    for first, (r, c) in enumerate(data['coords']):
        nodes = [by_coord.get((r+dr, c+dc)) for dr, dc in offsets]
        if any(n is None for n in nodes):
            continue
        if colors is None or all(data['colors'][n] == color for n, color in zip(nodes, colors)):
            found.append((first, tuple(nodes)))
    return found


def _fits(base, lattice, side, stride, shape, mini_box):
    br, bc = base
    bh, bw = ((n-1)*stride+side for n in lattice)
    mr, mc, mb, me = mini_box
    return (0 <= br and 0 <= bc and br+bh <= shape[0] and bc+bw <= shape[1]
            and (br+bh < mr or mb < br or bc+bw < mc or me < bc))


def _recover(input_grid):
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer)
    assert all(1 <= n <= 30 for n in grid.shape) and np.all((0 <= grid) & (grid <= 9))
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > np.partition(counts, -2)[-2]
    components = _components(grid != bg)
    plans = [c for c in components if not _solid_square(grid, c)]
    squares = [c for c in components if _solid_square(grid, c)]
    assert len(plans) == 1 and 2 <= len(squares) <= 5
    mr, mc, mb, me = _bbox(plans[0])
    mini = grid[mr:mb, mc:me].copy()
    data = _blueprint(mini, bg)
    boxes = sorted(_bbox(c) for c in squares)
    side = boxes[0][2]-boxes[0][0]
    assert 2 <= side <= 4 and all(b-r == e-c == side for r, c, b, e in boxes)
    first_r, first_c = boxes[0][:2]
    stride = 0
    for r, c, _, _ in boxes:
        stride = gcd(stride, abs(r-first_r))
        stride = gcd(stride, abs(c-first_c))
    assert 1 <= stride-side <= 3
    assert any((r == rr and abs(c-cc) == stride) or (c == cc and abs(r-rr) == stride)
               for (r, c, _, _), (rr, cc, _, _) in combinations(boxes, 2)), 'consecutive landmark pair required'
    offsets = [((r-first_r)//stride, (c-first_c)//stride) for r, c, _, _ in boxes]
    seed_colors = [int(grid[r, c]) for r, c, _, _ in boxes]
    matches = _registrations(data, offsets, seed_colors)
    assert len(matches) == 1, 'landmarks must have a unique colored registration'
    first, terminals = matches[0]
    pr, pc = data['coords'][first]
    base = first_r-pr*stride, first_c-pc*stride
    bh = (data['lattice'][0]-1)*stride+side
    bw = (data['lattice'][1]-1)*stride+side
    br, bc = base
    assert _fits(base, data['lattice'], side, stride, grid.shape, (mr, mc, mb, me))
    chosen = _minimum_tree(data, terminals)
    assert chosen is not None, 'minimum connecting tree is tied'
    nodes = _selected_nodes(data, chosen)
    assert set(terminals) <= nodes
    geometric = _registrations(data, offsets)
    viable = [(n, ns) for n, ns in geometric if _fits(
        (first_r-data['coords'][n][0]*stride, first_c-data['coords'][n][1]*stride),
        data['lattice'], side, stride, grid.shape, (mr, mc, mb, me))]
    data.update(bg=bg, mini=mini, mini_origin=(mr, mc), base=base, stride=stride,
                side=side, terminals=terminals, chosen=chosen, selected=nodes,
                geometric_matches=len(geometric), viable_geometric_matches=len(viable), shape=grid.shape)
    # Nothing except the supplied squares may occupy the entire target box.
    target = grid[br:br+bh, bc:bc+bw].copy()
    for n in terminals:
        r, c = data['coords'][n]
        target[r*stride:r*stride+side, c*stride:c*stride+side] = bg
    assert np.all(target == bg)
    return data


def _paint(grid, data, chosen=None):
    """Scale only the selected graph. Half-open station squares stay literal."""
    chosen = data['chosen'] if chosen is None else chosen
    result = grid.copy()
    side, stride = data['side'], data['stride']
    br, bc = data['base']
    mid = side//2
    # A miniature pixel coordinate maps to a station center at even positions.
    # Consecutive wire pixels become contiguous horizontal/vertical line runs.
    def center(p):
        return br+(p[0]*stride)//2+mid, bc+(p[1]*stride)//2+mid

    for e in chosen:
        a, b = data['edges'][e]
        endpoints = [(2*data['coords'][v][0], 2*data['coords'][v][1]) for v in (a, b)]
        cells = set(data['wires'][e]) | set(endpoints)
        for r, c in cells:
            for dr, dc in ((1, 0), (0, 1)):
                if (r+dr, c+dc) not in cells:
                    continue
                y, x = center((r, c))
                yy, xx = center((r+dr, c+dc))
                result[y:yy+1, x:xx+1] = data['wire']
    for n in _selected_nodes(data, chosen):
        r, c = data['coords'][n]
        y, x = br+r*stride, bc+c*stride
        result[y:y+side, x:x+side] = data['colors'][n]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid)
    return _paint(grid, _recover(grid))


def _sample_skeleton(rng, nr, nc):
    vertices = [(r, c) for r in range(nr) for c in range(nc)]
    possible = [(p, (p[0]+dr, p[1]+dc)) for p in vertices for dr, dc in ((1, 0), (0, 1))
                if p[0]+dr < nr and p[1]+dc < nc]
    target = int(rng.integers(min(10, len(vertices)), min(22, len(vertices))+1))
    active = {vertices[int(rng.integers(len(vertices)))]}
    edges = []
    while len(active) < target:
        boundary = [(a, b) for a, b in possible if (a in active) != (b in active)]
        a, b = boundary[int(rng.integers(len(boundary)))]
        active.update((a, b))
        edges.append((a, b))
    extras = [(a, b) for a, b in possible if a in active and b in active and (a, b) not in edges]
    rng.shuffle(extras)
    edges += extras[:int(rng.integers(0, min(3, len(extras))+1))] if extras else []
    degrees = Counter(v for e in edges for v in e)
    visible = {v for v in sorted(active) if degrees[v] != 2 or rng.random() < rng.uniform(.25, .75)}
    others = sorted(active-visible)
    rng.shuffle(others)
    while len(visible) < 7 and others:
        visible.add(others.pop())
    return active, edges, visible


def _fallback_skeleton():
    # Two close leaves compete via a dense short top and a sparse long bottom.
    active = {(r, c) for r in range(4) for c in range(4) if r in (0, 3) or c in (0, 3)}
    edges = [(a, b) for a, b in combinations(sorted(active), 2)
             if abs(a[0]-b[0])+abs(a[1]-b[1]) == 1]
    for leaf, attach in (((1, 1), (1, 0)), ((1, 2), (1, 3)), ((2, 1), (2, 0))):
        active.add(leaf)
        edges.append((leaf, attach))
    visible = {(0, c) for c in range(4)} | {(1, 0), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (3, 1)}
    return active, edges, visible


def _witness_skeleton(rng):
    """Variable long/sparse versus short/dense routes, with optional branches.

    Close terminal leaves attach to opposite sides of a perimeter. The two
    connecting arcs differ in physical length. Independently choose the actual
    stations on those arcs, making the longer one cheaper. Extra leaf branches
    attach at already visible stations so they cannot change that comparison.
    """
    nr, nc = map(int, rng.integers(4, 6, size=2))
    row = int(rng.choice([1, nr-2]))
    split = int(rng.integers(1, nc-2))
    ring = ([(0, c) for c in range(nc)] + [(r, nc-1) for r in range(1, nr)]
            + [(nr-1, c) for c in range(nc-2, -1, -1)]
            + [(r, 0) for r in range(nr-2, 0, -1)])
    active = set(ring) | {(row, c) for c in range(nc)}
    edges = [(ring[i], ring[(i+1) % len(ring)]) for i in range(len(ring))]
    edges += [((row, c), (row, c+1)) for c in range(nc-1) if c != split]
    a, b = (row, 0), (row, nc-1)

    def arc(step):
        i, stop, result = ring.index(a), ring.index(b), []
        i = (i+step) % len(ring)
        while i != stop:
            result.append(ring[i])
            i = (i+step) % len(ring)
        return result

    near, far = sorted((arc(1), arc(-1)), key=len)
    assert len(near) < len(far)
    near_count = int(rng.integers(3, min(5, len(near))+1))
    far_count = int(rng.integers(1, min(2, near_count-1)+1))
    boundary_row = 0 if row == 1 else nr-1
    col = int(rng.integers(nc-1))
    near_stations = {(boundary_row, col), (boundary_row, col+1)}
    candidates = [v for v in near if v not in near_stations]
    near_stations.update(candidates[int(i)] for i in rng.choice(len(candidates), near_count-2, replace=False))
    far_stations = {far[int(i)] for i in rng.choice(len(far), far_count, replace=False)}
    visible = {a, b, (row, split), (row, split+1)} | near_stations | far_stations
    for v in sorted(active-set(ring)-visible):
        if rng.random() < .5:
            visible.add(v)
    for _ in range(int(rng.integers(4))):
        choices = [(v, (v[0]+dr, v[1]+dc)) for v in sorted(visible) for dr, dc in _D4
                   if 0 <= v[0]+dr < nr and 0 <= v[1]+dc < nc
                   and (v[0]+dr, v[1]+dc) not in active]
        if not choices:
            break
        v, leaf = choices[int(rng.integers(len(choices)))]
        active.add(leaf)
        visible.add(leaf)
        edges.append((v, leaf))
    return active, edges, visible


def _draw_skeleton(skeleton):
    active, edges, visible = skeleton
    low_r, low_c = min(r for r, c in active), min(c for r, c in active)
    h = 2*(max(r for r, c in active)-low_r)+1
    w = 2*(max(c for r, c in active)-low_c)+1
    mini = np.zeros((h, w), dtype=np.int8)
    for r, c in active:
        mini[2*(r-low_r), 2*(c-low_c)] = 2 if (r, c) in visible else 1
    for (r, c), (rr, cc) in edges:
        mini[r+rr-2*low_r, c+cc-2*low_c] = 1
    return mini


def _layout(rng, mini_shape, lattice, side, stride, registration_room=False):
    large = tuple((n-1)*stride+side for n in lattice)
    axes = [d for d in (0, 1) if mini_shape[d]+large[d]+1 <= 30 and max(large[1-d], mini_shape[1-d]) <= 30]
    assert axes
    d = int(rng.choice(axes))
    other = 1-d
    gap = int(rng.integers(1, min(4, 30-mini_shape[d]-large[d])+1))
    needed = [max(large[k], mini_shape[k]) for k in (0, 1)]
    needed[d] = mini_shape[d]+large[d]+gap
    lower = needed.copy()
    if registration_room:
        lower[other] = min(30, lower[other]+stride)
    shape = tuple(int(rng.integers(n, min(30, n+5)+1)) for n in lower)
    if rng.random() < .12:
        shape = (30, 30)
    elif rng.random() < .15:
        shape = (max(shape), max(shape))
    mini, base = [0, 0], [0, 0]
    first = int(rng.integers(shape[d]-needed[d]+1))
    if rng.integers(2):
        mini[d], base[d] = first, first+mini_shape[d]+gap
    else:
        base[d], mini[d] = first, first+large[d]+gap
    mini[other] = int(rng.integers(shape[other]-mini_shape[other]+1))
    base[other] = int(rng.integers(shape[other]-large[other]+1))
    return shape, tuple(mini), tuple(base)


def _has_branch(data, chosen, terminals):
    degree = Counter(v for e in chosen for v in data['edges'][e])
    return any(k >= 3 and v not in terminals for v, k in degree.items())


def _make_pair(rng, kind, side, terminal_count, forced_fallback=False):
    for attempt in range(121):
        fallback = forced_fallback or attempt >= 100
        if fallback:
            skeleton = _fallback_skeleton()
        elif kind == 'distance' and rng.random() < .8:
            skeleton = _witness_skeleton(rng)
        else:
            nr, nc = map(int, rng.integers(3, 6, size=2))
            skeleton = _sample_skeleton(rng, nr, nc)
        mini = _draw_skeleton(skeleton)
        mini = np.rot90(mini, int(rng.integers(4)))
        if rng.integers(2):
            mini = mini[:, ::-1]
        try:
            data = _blueprint(mini, 0)
        except AssertionError:
            continue
        coords = data['coords']
        adjacent = [(a, b) for a, b in combinations(range(len(coords)), 2)
                    if abs(coords[a][0]-coords[b][0])+abs(coords[a][1]-coords[b][1]) == 1]
        rng.shuffle(adjacent)
        selected = None
        for a, b in adjacent[:20]:
            remaining = [n for n in range(len(coords)) if n not in (a, b)]
            if kind == 'landmarks':
                # Repeated spatial motifs make query colors necessary too.
                # Choose a constellation that can translate by one lattice step
                # onto another set of actual stations before colors are read.
                by_coord = set(coords)
                domains = []
                for dr, dc in _D4:
                    domain = [n for n, (r, c) in enumerate(coords) if (r+dr, c+dc) in by_coord]
                    if a in domain and b in domain and len(domain) >= terminal_count:
                        domains.append(domain)
                if not domains:
                    continue
                domain = domains[int(rng.integers(len(domains)))]
                remaining = [n for n in domain if n not in (a, b)]
            extras = list(map(int, rng.choice(remaining, terminal_count-2, replace=False)))
            terminals = tuple(sorted([a, b]+extras))
            chosen = _minimum_tree(data, terminals)
            if chosen is None:
                continue
            nodes = _selected_nodes(data, chosen)
            if len(nodes-set(terminals)) < 2 or len(nodes) == len(coords):
                continue
            if kind == 'distance' and not _distance_conflict(data, terminals, chosen):
                continue
            if kind == 'branch' and not _has_branch(data, chosen, terminals):
                continue
            offsets = [(coords[n][0]-coords[terminals[0]][0], coords[n][1]-coords[terminals[0]][1])
                       for n in terminals]
            if kind == 'distance' and len(_registrations(data, offsets)) < 2:
                continue
            selected = terminals, chosen, nodes, offsets
            break
        if selected is None:
            continue
        terminals, chosen, nodes, offsets = selected
        # Literal color roles vary locally, including the connector/background.
        roles = list(map(int, rng.permutation(10)))
        bg, wire = roles[:2]
        payload = roles[2:2+int(rng.integers(3, 6))]
        colors = None
        for _ in range(80):
            trial = list(map(int, rng.choice(payload, len(coords))))
            data['colors'] = trial
            if len(_registrations(data, offsets, [trial[n] for n in terminals])) != 1:
                continue
            if len({trial[n] for n in nodes-set(terminals)}) < 2:
                continue
            if kind == 'distance' and any(trial.count(trial[n]) < 2 for n in terminals):
                continue
            colors = trial
            break
        if colors is None:
            continue
        colored = np.full(mini.shape, bg, dtype=np.int8)
        colored[mini == 1] = wire
        for (r, c), color in zip(coords, colors):
            colored[2*r, 2*c] = color
        # Larger station side need not imply a larger network or more terminals.
        gutters = [g for g in (1, 2, 3) if any(
            mini.shape[d]+(data['lattice'][d]-1)*(side+g)+side+1 <= 30
            and (data['lattice'][1-d]-1)*(side+g)+side <= 30 for d in (0, 1))]
        if not gutters:
            continue
        stride = side+int(rng.choice(gutters))
        shape, anchor, base = _layout(rng, mini.shape, data['lattice'], side, stride,
                                     registration_room=kind in ('distance', 'landmarks'))
        inp = np.full(shape, bg, dtype=np.int8)
        mr, mc = anchor
        inp[mr:mr+mini.shape[0], mc:mc+mini.shape[1]] = colored
        for n in terminals:
            r, c = coords[n]
            y, x = base[0]+r*stride, base[1]+c*stride
            inp[y:y+side, x:x+side] = colors[n]
        counts = np.bincount(inp.ravel().astype(int), minlength=10)
        if counts[bg] <= max(counts[c] for c in range(10) if c != bg):
            continue
        data.update(colors=colors, wire=wire, bg=bg, side=side, stride=stride, base=base, chosen=chosen)
        out = _paint(inp, data)
        # Verify geometry was encoded correctly, rather than passing latent data.
        recovered = _recover(inp)
        if kind in ('distance', 'landmarks') and recovered['viable_geometric_matches'] < 2:
            continue
        assert np.array_equal(out, _paint(inp, recovered))
        return {'input': inp, 'output': out}
    raise RuntimeError('bounded pair construction exhausted')


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def _key(grid):
    return tuple(grid.shape), np.asarray(grid, dtype=np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    first_side = int(rng.choice([2, 3, 4], p=[.5, .4, .1]))
    second_side = int(rng.choice([s for s in (2, 3, 4) if s != first_side]))
    train_requests = [('distance', first_side, 2), ('branch', second_side, 3)]
    train_requests += [('ordinary', int(rng.choice([2, 3, 4], p=[.5, .4, .1])), int(rng.integers(2, 5)))
                       for _ in range(num_train-2)]
    test_requests = [('landmarks', int(rng.choice([2, 3, 4], p=[.45, .4, .15])), int(rng.integers(4, 6)))
                     for _ in range(num_test)]
    for split, requests in (('train', train_requests), ('test', test_requests)):
        for kind, side, count in requests:
            for _ in range(16):
                pair = _make_pair(rng, kind, side, count)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded duplicate exclusion exhausted')
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def features(input_grid):
    """Bounded pixel-derived diagnostics; not a second solver."""
    d = _recover(input_grid)
    nodes = d['selected']
    return dict(shape=tuple(input_grid.shape), lattice=d['lattice'], stations=len(d['coords']),
                cycle_rank=len(d['edges'])-len(d['coords'])+1,
                terminal_count=len(d['terminals']), added_stations=len(nodes-set(d['terminals'])),
                omitted_stations=len(d['coords'])-len(nodes), side=d['side'], gutter=d['stride']-d['side'],
                background=d['bg'], connector=d['wire'], match=d['coords'][d['terminals'][0]],
                mini_origin=d['mini_origin'], base=d['base'], geometric_matches=d['geometric_matches'],
                viable_geometric_matches=d['viable_geometric_matches'],
                branch=_has_branch(d, d['chosen'], d['terminals']),
                distance_conflict=_distance_conflict(d, d['terminals'], d['chosen']),
                bent_edges=sum(len({r for r, c in cells}) > 1 and len({c for r, c in cells}) > 1
                               for cells in d['wires']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            d = _recover(inp)
            assert np.array_equal(out, _paint(inp, d)), 'incorrect construction or changed unrelated pixels'
            assert not np.array_equal(inp, out)
            assert len(d['selected']) < len(d['coords']), 'whole-blueprint copying must fail'
            additions = d['selected']-set(d['terminals'])
            assert len(additions) >= 2 and len({d['colors'][n] for n in additions}) >= 2
            # Construction is additive; preserve every already visible pixel.
            assert np.array_equal(inp[inp != d['bg']], out[inp != d['bg']])
            assert set(map(int, out.flat)) <= set(map(int, inp.flat))
            if split == 'train':
                training.append(d)
            else:
                assert len(d['terminals']) in (4, 5)
                assert d['viable_geometric_matches'] >= 2, 'test must require colored registration'
    assert any(len(d['terminals']) == 2 and _distance_conflict(d, d['terminals'], d['chosen'])
               and d['viable_geometric_matches'] >= 2
               and all(d['colors'].count(d['colors'][n]) >= 2 for n in d['terminals']) for d in training)
    assert any(len(d['terminals']) >= 3 and _has_branch(d, d['chosen'], d['terminals']) for d in training)
    assert len({d['side'] for d in training}) >= 2


def _signature(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Stream 42 seeds, check exact semantics, teaching witnesses, and variation."""
    signatures, records, structures = set(), [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
                f = features(pair['input'])
                f['split'] = split
                records.append(f)
                # Normalize literal colors in encounter order.
                roles = {}
                norm = np.array([roles.setdefault(int(c), len(roles)) for c in pair['input'].flat], dtype=np.int8)
                structures.add((pair['input'].shape, norm.tobytes()))
        if seed < 2:
            assert _signature(puzzle) == _signature(generate_puzzle(seed))
    assert len(signatures) == 42 and len(structures) >= 180
    for split in ('train', 'test'):
        rows = [r for r in records if r['split'] == split]
        assert len({r['shape'] for r in rows}) >= 20
        assert len({r['lattice'] for r in rows}) >= 5
        assert len({r['stations'] for r in rows}) >= 6
        assert len({r['match'] for r in rows}) >= 6
        assert len({r['background'] for r in rows}) >= 8
        assert {r['side'] for r in rows} == {2, 3, 4}
        assert {r['gutter'] for r in rows} == {1, 2, 3}
        assert any(r['bent_edges'] for r in rows)
        assert any(r['cycle_rank'] >= 2 for r in rows)
        assert any(r['branch'] for r in rows)
    assert {r['terminal_count'] for r in records} == {2, 3, 4, 5}
    assert any(r['shape'] == (30, 30) for r in records)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(731+nt*7+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # A square cycle has two different minimum connections between opposites.
    tied = dict(coords=[(0, 0), (0, 1), (1, 0), (1, 1)],
                edges=[(0, 1), (0, 2), (1, 3), (2, 3)])
    tied['adj'] = _adjacency(4, tied['edges'])
    assert _minimum_tree(tied, (0, 3)) is None
    assert _minimum_tree(tied, (0, 1, 2)) == frozenset((0, 1))
    assert _minimum_tree(tied, (0, 1, 2, 3)) is None
    # Exercise the finite fallback on both required witness types and rich clues.
    rng = np.random.default_rng(8031)
    for side in (2, 3, 4):
        for kind, count in (('distance', 2), ('branch', 3), ('landmarks', 5)):
            p = _make_pair(rng, kind, side, count, forced_fallback=True)
            assert np.array_equal(solve(p['input']), p['output'])
    damaged = generate_puzzle(9001, 2, 1)
    d = _recover(damaged['test'][0]['input'])
    mr, mc = d['mini_origin']
    r, c = d['coords'][0]
    damaged['test'][0]['output'][mr+2*r, mc+2*c] = d['bg']
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('changed blueprint accepted')
    # Remove the color information from an alignment witness. Even after
    # checking canvas fit, its geometry admits more than one registration.
    witness = next(p for p in generate_puzzle(0, 2, 1)['train']
                   if _recover(p['input'])['viable_geometric_matches'] > 1)
    ambiguous = witness['input'].copy()
    d = _recover(ambiguous)
    payload_mask = np.isin(ambiguous, d['colors'])
    ambiguous[payload_mask] = d['colors'][0]
    try:
        solve(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError('ambiguous landmark registration accepted')
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures), structures=len(structures))
