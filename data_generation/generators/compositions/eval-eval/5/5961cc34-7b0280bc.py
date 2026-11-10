"""Fewest handoffs: minimum-relay delivery in an implicit port/ray graph.

Only standard library and NumPy are used. No I/O is performed by generation,
validation, solving, or self_test. See task_spec.md for the family contract.
"""
from collections import deque
import hashlib
import heapq
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or
            not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        root = min(remaining)
        remaining.remove(root)
        found, stack = {root}, [root]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    found.add(p)
                    stack.append(p)
        result.append(found)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _decode(a):
    """Recover all roles and port geometry from this input alone."""
    a = _grid(a)
    h, w = a.shape
    bg = int(a[0, 0])
    assert all(int(a[p]) == bg for p in ((0, w-1), (h-1, 0), (h-1, w-1)))
    cc = _components(map(tuple, np.argwhere(a != bg).tolist()))
    launches = [c for c in cc if any(r in (0, h-1) or k in (0, w-1) for r, k in c)]
    assert len(launches) == 1
    launcher = launches[0]
    ends = [p for p in launcher if p[0] in (0, h-1) or p[1] in (0, w-1)]
    assert len(ends) == 1
    root = ends[0]
    d = 2 if root[0] == 0 else 0 if root[0] == h-1 else 1 if root[1] == 0 else 3
    dr, dc = DIRS[d]
    line = [(root[0]+k*dr, root[1]+k*dc) for k in range(len(launcher))]
    assert set(line) == launcher and 3 <= len(line) <= 6
    signal, goal_color = int(a[root]), int(a[line[-1]])
    assert signal != goal_color and all(int(a[p]) == signal for p in line[:-1])
    nodes, goal, body_colors, port_colors = [], None, set(), set()
    for component in cc:
        if component == launcher:
            continue
        box = _bbox(component)
        r0, r1, c0, c1 = box
        assert 0 < r0 <= r1 < h-1 and 0 < c0 <= c1 < w-1
        assert min(r1-r0+1, c1-c0+1) >= 3
        colors, counts = np.unique([a[p] for p in component], return_counts=True)
        ports = []
        if len(colors) == 1:
            assert int(colors[0]) == goal_color and goal is None
            assert len(component) == (r1-r0+1)*(c1-c0+1)
            goal = len(nodes)
        else:
            assert len(colors) == 2 and counts[0] != counts[1]
            pc, bc = int(colors[np.argmin(counts)]), int(colors[np.argmax(counts)])
            assert not {pc, bc} & {bg, signal, goal_color}
            body_colors.add(bc)
            port_colors.add(pc)
            groups = {}
            for r, c in sorted(component):
                if int(a[r, c]) != pc:
                    continue
                sides = [i for i, ok in enumerate((r == r0, c == c1, r == r1, c == c0)) if ok]
                assert len(sides) == 1, 'ports cannot occupy corners or interior cells'
                groups.setdefault(sides[0], set()).add((r, c))
            assert 1 <= len(groups) <= 3
            for side, cells in sorted(groups.items()):
                span = sorted(c if side % 2 == 0 else r for r, c in cells)
                assert 1 <= len(span) <= 2
                assert span == list(range(span[0], span[0]+len(span)))
                dr2, dc2 = DIRS[side]
                assert all(int(a[r-dr2, c-dc2]) == bc for r, c in cells)
                ports.append((side, frozenset(cells)))
        nodes.append({'cells': frozenset(component), 'box': box, 'ports': ports})
    assert goal is not None and len(nodes) >= 3
    assert len(body_colors) == len(port_colors) == 1
    assert len({bg, signal, goal_color, *body_colors, *port_colors}) == 5
    assert np.count_nonzero(a == bg) > a.size // 2
    # A full blank Chebyshev gap keeps independently drawn objects legible.
    owner = {p: i for i, n in enumerate(nodes) for p in n['cells']}
    for p, i in owner.items():
        for rr in (-1, 0, 1):
            for cc2 in (-1, 0, 1):
                neighbor = p[0]+rr, p[1]+cc2
                assert owner.get(neighbor, i) == i and neighbor not in launcher
    return {'grid': a, 'bg': bg, 'signal': signal, 'goal_color': goal_color,
            'body': next(iter(body_colors)), 'port': next(iter(port_colors)),
            'launcher': frozenset(launcher), 'cap': line[-1], 'direction': d,
            'nodes': nodes, 'goal': goal, 'owner': owner}


def _ray(shape, owner, origins, direction):
    """Advance a beam against original objects, never against drawn beams."""
    dr, dc = DIRS[direction]
    front, cells, distance = list(origins), set(), 0
    while True:
        front = [(r+dr, c+dc) for r, c in front]
        inside = [0 <= r < shape[0] and 0 <= c < shape[1] for r, c in front]
        assert all(inside) or not any(inside)
        if not any(inside):
            return None, frozenset(cells), distance
        hits = {owner[p] for p in front if p in owner}
        if hits:
            if hits == {-1}:
                return -1, frozenset(cells), distance
            assert len(hits) == 1 and all(p in owner for p in front), 'partial or simultaneous collision'
            return next(iter(hits)), frozenset(cells), distance
        cells.update(front)
        distance += 1


def _graph(scene):
    owner = dict(scene['owner'])
    owner.update({p: -1 for p in scene['launcher']})
    adj = [[] for _ in scene['nodes']]
    for i, n in enumerate(scene['nodes']):
        for direction, port in n['ports']:
            target, cells, gap = _ray(scene['grid'].shape, owner, port, direction)
            if target is not None and target >= 0:
                assert target != i
                adj[i].append({'to': target, 'direction': direction, 'width': len(port),
                               'cells': cells, 'gap': gap})
    start, cells, gap = _ray(scene['grid'].shape, owner, [scene['cap']], scene['direction'])
    assert start is not None and start >= 0, 'launcher must hit a station'
    return adj, start, cells


def _bfs(adj, start):
    dist, ways, prev = [10**6]*len(adj), [0]*len(adj), [None]*len(adj)
    dist[start], ways[start] = 0, 1
    queue = deque([start])
    while queue:
        a = queue.popleft()
        for e in adj[a]:
            b = e['to']
            if dist[b] == 10**6:
                dist[b], prev[b] = dist[a]+1, (a, e)
                queue.append(b)
            if dist[b] == dist[a]+1:
                ways[b] = min(2, ways[b]+ways[a])
    return dist, ways, prev


def _path(prev, start, goal):
    nodes, edges = [goal], []
    while nodes[-1] != start:
        a, e = prev[nodes[-1]]
        nodes.append(a)
        edges.append(e)
    return nodes[::-1], edges[::-1]


def _weighted(adj, start, metric):
    costs, previous = [float('inf')]*len(adj), [None]*len(adj)
    costs[start] = 0
    queue = [(0, start)]
    while queue:
        cost, a = heapq.heappop(queue)
        if cost != costs[a]:
            continue
        for e in adj[a]:
            b, candidate = e['to'], cost + e[metric]
            if candidate < costs[b]:
                costs[b], previous[b] = candidate, (a, e)
                heapq.heappush(queue, (candidate, b))
    return costs, previous


def _paint(scene, nodes, edges, initial):
    out = np.full(scene['grid'].shape, scene['bg'], dtype=np.int8)
    cells = set(scene['launcher']) | set(initial)
    for n in nodes:
        cells.update(scene['nodes'][n]['cells'])
    for e in edges:
        cells.update(e['cells'])
    for p in cells:
        out[p] = scene['signal']
    return out


def _analyze(input_grid):
    scene = _decode(np.asarray(input_grid))
    adj, start, initial = _graph(scene)
    dist, ways, previous = _bfs(adj, start)
    assert ways[scene['goal']] == 1, 'receiver unreachable or minimum-handoff route tied'
    nodes, edges = _path(previous, start, scene['goal'])
    return scene, adj, start, initial, nodes, edges


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene, _, _, initial, nodes, edges = _analyze(input_grid)
    return _paint(scene, nodes, edges, initial)


def _open_graph(scene):
    """Counterfactual: let every relay emit centrally on every side."""
    adj = [[] for _ in scene['nodes']]
    for i, n in enumerate(scene['nodes']):
        if i == scene['goal']:
            continue
        r0, r1, c0, c1 = n['box']
        midr, midc = (r0+r1+1)//2, (c0+c1+1)//2
        origins = ((r0, midc), (midr, c1), (r1, midc), (midr, c0))
        for d, p in enumerate(origins):
            target, cells, gap = _ray(scene['grid'].shape, scene['owner'], [p], d)
            if target is not None and target != i:
                adj[i].append({'to': target, 'cells': cells, 'gap': gap,
                               'area': len(cells), 'direction': d, 'width': 1})
    return adj


def _ablation_outputs(input_grid):
    """Concrete wrong-rule outputs for the mandatory demonstration witnesses."""
    scene, adj, start, initial, nodes, edges = _analyze(input_grid)
    for row in adj:
        for edge in row:
            edge['area'] = len(edge['cells'])
    result = {}
    opened = _open_graph(scene)
    dist, _, prev = _bfs(opened, start)
    if dist[scene['goal']] < 10**6:
        nn, ee = _path(prev, start, scene['goal'])
        result['ignore_ports'] = _paint(scene, nn, ee, initial)
    _, prev = _weighted(adj, start, 'area')
    nn, ee = _path(prev, start, scene['goal'])
    result['minimize_beam_area'] = _paint(scene, nn, ee, initial)
    target, cells, _ = _ray(scene['grid'].shape,
                           {p: scene['goal'] for p in scene['nodes'][scene['goal']]['cells']},
                           [scene['cap']], scene['direction'])
    if target == scene['goal']:
        result['skip_occluding_relays'] = _paint(scene, [target], [], cells)
    return result


def _facts(input_grid):
    scene, adj, start, initial, nodes, edges = _analyze(input_grid)
    for row in adj:
        for e in row:
            e['area'] = len(e['cells'])
    for i, row in enumerate(adj):
        r0, r1, c0, c1 = scene['nodes'][i]['box']
        for e in row:
            rr0, rr1, cc0, cc1 = scene['nodes'][e['to']]['box']
            e['length'] = (abs(r0+r1-rr0-rr1)+abs(c0+c1-cc0-cc1))/2
    goal = scene['goal']
    gap_cost = sum(e['gap'] for e in edges)
    area_cost = sum(e['area'] for e in edges)
    geometry_conflict = (gap_cost > _weighted(adj, start, 'gap')[0][goal] and
                         area_cost > _weighted(adj, start, 'area')[0][goal] and
                         sum(e['length'] for e in edges) > _weighted(adj, start, 'length')[0][goal])
    open_distance = _bfs(_open_graph(scene), start)[0][goal]
    direct_goal, _, _ = _ray(scene['grid'].shape, {p: goal for p in scene['nodes'][goal]['cells']},
                             [scene['cap']], scene['direction'])
    used_cells = [initial] + [e['cells'] for e in edges]
    reachable = {i for i, d in enumerate(_bfs(adj, start)[0]) if d < 10**6}
    return {'station_count': len(scene['nodes']), 'relays_used': len(nodes)-1,
            'shape': tuple(scene['grid'].shape), 'direction': scene['direction'],
            'widths': tuple(e['width'] for e in edges),
            'geometry_conflict': geometry_conflict, 'port_conflict': open_distance < len(edges),
            'occlusion': direct_goal == goal and start != goal,
            'gap_cost': gap_cost, 'area_cost': area_cost,
            'selected_turn': any(a != b for a, b in zip(
                [scene['direction']] + [e['direction'] for e in edges],
                [e['direction'] for e in edges])),
            'branch_on_route': any(len(adj[n]) >= 2 for n in nodes[:-1]),
            'discarded_reachable': len(reachable-set(nodes)),
            'crossing': any(a & b for i, a in enumerate(used_cells) for b in used_cells[i+1:]),
            'sizes': tuple(sorted({(n['box'][1]-n['box'][0]+1, n['box'][3]-n['box'][2]+1)
                                   for n in scene['nodes']})),
            'palette': (scene['bg'], scene['signal'], scene['goal_color'], scene['body'], scene['port'])}


def _axis(rng, n, size, before, after):
    minimum = before + n*size + (n-1)*2 + after
    assert minimum <= 30
    extra = int(rng.integers(0, 31-minimum))
    allocation = rng.multinomial(extra, np.full(n+1, 1/(n+1)))
    starts = [before+int(allocation[0])]
    for k in range(n-1):
        starts.append(starts[-1]+size+2+int(allocation[k+1]))
    return starts, starts[-1]+size+after+int(allocation[-1])


def _lattice_adjacency(pos, rows, cols, size, ports):
    ids = {p: i for i, p in enumerate(pos)}
    full, adj = [[] for _ in pos], [[] for _ in pos]
    for i, (r, c) in enumerate(pos):
        for d, (dr, dc) in enumerate(DIRS):
            rr, cc = r+dr, c+dc
            while 0 <= rr < len(rows) and 0 <= cc < len(cols):
                if (rr, cc) in ids:
                    gap = abs(rows[rr]-rows[r])+abs(cols[cc]-cols[c])-size
                    full[i].append({'to': ids[rr, cc], 'direction': d, 'gap': gap,
                                    'area': gap, 'length': gap+size})
                    break
                rr, cc = rr+dr, cc+dc
        for d, width in ports[i].items():
            e = next((e for e in full[i] if e['direction'] == d), None)
            if e is not None:
                adj[i].append(dict(e, width=width, area=e['gap']*width))
    return full, adj


def _random_plan(rng, kind, size):
    if kind == 'test':
        nr = nc = 4
        count = int(rng.integers(13, 17))
    else:
        options = [(3, 3), (3, 4), (4, 3)] if kind == 'witness' else [(2, 3), (3, 2), (3, 3), (3, 4), (4, 3)]
        options = [(r, c) for r, c in options if 4+c*size+(c-1)*2+1 <= 30 and r*size+(r-1)*2+2 <= 30]
        nr, nc = options[int(rng.integers(len(options)))]
        count = int(rng.integers(min(7 if kind == 'witness' else 5, nr*nc), nr*nc+1))
    pos = sorted(divmod(int(x), nc) for x in rng.choice(nr*nc, count, replace=False))
    rows, h = _axis(rng, nr, size, 1, 1)
    cols, w = _axis(rng, nc, size, 4, 1)
    ports = []
    for _ in pos:
        n = int(rng.choice([1, 2, 3], p=[.16, .54, .30]))
        sides = rng.choice(4, n, replace=False)
        ports.append({int(d): int(rng.integers(1, min(2, size-2)+1)) for d in sides})
    full, adj = _lattice_adjacency(pos, rows, cols, size, ports)
    starts = [i for i, (r, c) in enumerate(pos) if not any(rr == r and cc < c for rr, cc in pos)]
    rng.shuffle(starts)
    choices = []
    desired = int(rng.integers(3 if kind == 'test' else 1, 8 if kind == 'test' else 6))
    for start in starts:
        distances, ways, prev = _bfs(adj, start)
        open_dist = _bfs(full, start)[0]
        gap_best = _weighted(adj, start, 'gap')[0]
        area_best = _weighted(adj, start, 'area')[0]
        length_best = _weighted(adj, start, 'length')[0]
        for goal in range(len(pos)):
            if goal == start or ways[goal] != 1:
                continue
            nodes, edges = _path(prev, start, goal)
            if kind == 'witness':
                if not (open_dist[goal] < distances[goal] and
                        sum(e['gap'] for e in edges) > gap_best[goal] and
                        sum(e['area'] for e in edges) > area_best[goal] and
                        sum(e['length'] for e in edges) > length_best[goal] and
                        {e['width'] for e in edges} == {1, 2}):
                    continue
            if kind == 'occlusion' and not (pos[start][0] == pos[goal][0] and pos[start][1] < pos[goal][1]):
                continue
            choices.append((abs(distances[goal]-desired), start, goal, nodes, edges))
    if not choices:
        return None
    best = min(x[0] for x in choices)
    choices = [x for x in choices if x[0] == best]
    _, start, goal, nodes, edges = choices[int(rng.integers(len(choices)))]
    return {'pos': pos, 'rows': rows, 'cols': cols, 'size': size, 'shape': (h, w),
            'ports': ports, 'start': start, 'goal': goal, 'nodes': nodes, 'edges': edges}


def _fallback_plan(rng, kind, size):
    """Constructive bounded fallback; no redraw to an easier obligation."""
    if kind == 'test':
        pos = [(r, c) for r in range(4) for c in range(4)]
        rows, h = _axis(rng, 4, size, 1, 1)
        cols, w = _axis(rng, 4, size, 4, 1)
        path = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]
        ports = [{0: 1} for _ in pos]
        for a, b in zip(path, path[1:]):
            d = 1 if a[0] == b[0] else 2
            ports[pos.index(a)] = {d: int(rng.integers(1, min(2, size-2)+1))}
        start, goal = pos.index(path[0]), pos.index(path[-1])
    else:
        # Three-relay long upper route beats five-relay short lower route.
        # The unmarked middle shortcut is unavailable, and the receiver is
        # directly behind the first collision on the launcher's sightline.
        size = 4
        pos = [(0, 0), (0, 3), (1, 0), (1, 1), (1, 3),
               (2, 0), (2, 1), (2, 2), (2, 3)]
        rows, cols, h, w = [1, 14, 21], [4, 10, 16, 22], 27, 27
        ports = [{1: 2}, {2: 1}, {0: 1, 2: 1}, {3: 1}, {},
                 {1: 1}, {1: 1}, {1: 1}, {0: 1}]
        start, goal = 2, 4
    _, adj = _lattice_adjacency(pos, rows, cols, size, ports)
    _, ways, prev = _bfs(adj, start)
    assert ways[goal] == 1
    nodes, edges = _path(prev, start, goal)
    return {'pos': pos, 'rows': rows, 'cols': cols, 'size': size, 'shape': (h, w),
            'ports': ports, 'start': start, 'goal': goal, 'nodes': nodes, 'edges': edges}


def _render(rng, plan, palette):
    bg, signal, goal_color, carrier, marker = palette
    size, pos, rows, cols = plan['size'], plan['pos'], plan['rows'], plan['cols']
    a = np.full(plan['shape'], bg, dtype=np.int8)
    footprints, port_cells, boxes = [], [], []
    for i, (r, c) in enumerate(pos):
        r0, c0 = rows[r], cols[c]
        cells = {(r0+rr, c0+cc) for rr in range(size) for cc in range(size)}
        ports = {}
        if i != plan['goal']:
            for d, width in sorted(plan['ports'][i].items()):
                offset = int(rng.integers(1, size-width))
                points = ({(r0, c0+offset+k) for k in range(width)} if d == 0 else
                          {(r0+offset+k, c0+size-1) for k in range(width)} if d == 1 else
                          {(r0+size-1, c0+offset+k) for k in range(width)} if d == 2 else
                          {(r0+offset+k, c0) for k in range(width)})
                ports[d] = points
            # Independently clipped corners preserve flat interior beam lanes.
            corners = [(r0, c0), (r0, c0+size-1), (r0+size-1, c0), (r0+size-1, c0+size-1)]
            rng.shuffle(corners)
            maximum = min(4, size*size-2*sum(map(len, ports.values()))-1)
            for p in corners[:int(rng.integers(0, maximum+1))]:
                cells.remove(p)
        footprints.append(cells)
        port_cells.append(ports)
        boxes.append((r0, r0+size-1, c0, c0+size-1))
        for p in cells:
            a[p] = goal_color if i == plan['goal'] else carrier
        for points in ports.values():
            for p in points:
                a[p] = marker
    first = plan['start']
    r0, _, c0, _ = boxes[first]
    launch_row = r0+int(rng.integers(1, size-1))
    length = int(rng.integers(2, min(5, c0-1)))
    a[launch_row, :length] = signal
    a[launch_row, length] = goal_color
    out = np.full_like(a, bg)
    out[launch_row, :c0] = signal
    for i in plan['nodes']:
        for p in footprints[i]:
            out[p] = signal
    # Labels come from lattice first-neighbor geometry, not solve/_ray.
    for source, e in zip(plan['nodes'], plan['edges']):
        d, target = e['direction'], e['to']
        dr, dc = DIRS[d]
        for r, c in port_cells[source][d]:
            for k in range(1, e['gap']+1):
                out[r+k*dr, c+k*dc] = signal
    turns = int(rng.integers(4))
    a, out = np.rot90(a, turns).copy(), np.rot90(out, turns).copy()
    if rng.random() < .5:
        a, out = a[:, ::-1].copy(), out[:, ::-1].copy()
    return {'input': a, 'output': out}


def _make_pair(rng, kind, size, palette):
    for _ in range(1536 if kind == 'witness' else 96):
        plan = _random_plan(rng, kind, size)
        if plan is not None:
            return _render(rng, plan, palette)
    return _render(rng, _fallback_plan(rng, kind, size), palette)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    common = tuple(map(int, rng.choice(10, 5, replace=False)))
    vary_palette = bool(rng.random() < .75)
    ordinary_size = int(rng.choice([3, 4]))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            kind = 'test' if split == 'test' else 'witness' if i == 0 else 'occlusion' if i == 1 else 'ordinary'
            size = ordinary_size if split == 'test' or i == 1 else int(rng.choice([4, 5])) if i == 0 else int(rng.choice([3, 4, 5]))
            palette = tuple(map(int, rng.choice(10, 5, replace=False))) if vary_palette else common
            for _ in range(16):
                pair = _make_pair(rng, kind, size, palette)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == 'train' and i == 1:
                        # A constructive fallback may replace the proposed size.
                        # Queries reuse the actual demonstrated size.
                        box = _decode(pair['input'])['nodes'][0]['box']
                        ordinary_size = box[1]-box[0]+1
                    break
            else:
                raise RuntimeError('bounded distinct-input construction exhausted')
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, facts = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            scene, adj, start, initial, nodes, edges = _analyze(a)
            expected = _paint(scene, nodes, edges, initial)
            assert np.array_equal(b, expected) and not np.array_equal(a, b)
            assert set(map(int, np.unique(b))) == {scene['bg'], scene['signal']}
            for i, n in enumerate(scene['nodes']):
                if i not in nodes:
                    assert all(int(b[p]) == scene['bg'] for p in n['cells'])
            facts[split].append(_facts(a))
    train, test = facts['train'], facts['test']
    assert any(f['geometry_conflict'] and f['port_conflict'] for f in train)
    assert any(f['occlusion'] for f in train)
    assert {1, 2} <= {w for f in train for w in f['widths']}
    assert any(f['selected_turn'] and f['branch_on_route'] for f in train)
    assert max(f['station_count'] for f in train) <= 12
    assert min(f['station_count'] for f in test) >= 13
    assert all(f['station_count'] <= 16 for f in test)


def _enumerate_paths(adj, start, goal):
    result, stack = [], [(start, (start,), ())]
    while stack:
        a, nodes, edges = stack.pop()
        if a == goal:
            result.append((nodes, edges))
            continue
        for e in adj[a]:
            if e['to'] not in nodes:
                stack.append((e['to'], nodes+(e['to'],), edges+(e,)))
    return result


def self_test():
    """42 seeds, streaming measurements, exhaustive simple-path cross-checks."""
    signatures, structures, shapes, sizes, directions, palettes = set(), set(), set(), set(), set(), set()
    records, totals = [], {'pairs': 0, 'geometry_witnesses': 0, 'port_witnesses': 0,
                           'occlusion_witnesses': 0, 'crossings': 0, 'discarded_reachable': 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                scene, adj, start, initial, nodes, edges = _analyze(a)
                # Exhaustive search is a different optimality/multiplicity check.
                paths = _enumerate_paths(adj, start, scene['goal'])
                minimum = min(len(p[0]) for p in paths)
                shortest = [p for p in paths if len(p[0]) == minimum]
                assert len(shortest) == 1 and list(shortest[0][0]) == nodes
                assert np.array_equal(_paint(scene, list(shortest[0][0]), shortest[0][1], initial), pair['output'])
                f = _facts(a)
                if f['geometry_conflict'] and f['port_conflict']:
                    alternatives = _ablation_outputs(a)
                    assert not np.array_equal(alternatives['ignore_ports'], pair['output'])
                    assert not np.array_equal(alternatives['minimize_beam_area'], pair['output'])
                if f['occlusion']:
                    assert not np.array_equal(_ablation_outputs(a)['skip_occluding_relays'], pair['output'])
                records.append((split, f))
                totals['pairs'] += 1
                for name, field in (('geometry_witnesses', 'geometry_conflict'), ('port_witnesses', 'port_conflict'),
                                    ('occlusion_witnesses', 'occlusion'), ('crossings', 'crossing')):
                    totals[name] += int(f[field])
                totals['discarded_reachable'] += int(f['discarded_reachable'] > 0)
                shapes.add(f['shape']); sizes.update(f['sizes']); directions.add(f['direction']); palettes.add(f['palette'])
                semantic = np.zeros_like(a)
                for k, color in enumerate(f['palette']):
                    semantic[a == color] = k
                structures.add(_key(semantic))
                signature.update(str(a.shape).encode()); signature.update(a.tobytes()); signature.update(pair['output'].tobytes())
        signatures.add(signature.hexdigest())
        if seed < 2:
            other = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], other[s]) for k in p)
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 35 and len(sizes) >= 3 and directions == set(range(4))
    assert len(palettes) >= 30
    assert totals['crossings'] > 0 and totals['discarded_reachable'] >= 42
    assert len({f['relays_used'] for _, f in records}) >= 5
    # Directly exercise constructive fallbacks, including every orientation.
    rng = np.random.default_rng(91021)
    for kind, size in (('witness', 4), ('occlusion', 5), ('ordinary', 3), ('test', 3), ('test', 4)):
        pair = _render(rng, _fallback_plan(rng, kind, size), (0, 2, 4, 7, 8))
        assert np.array_equal(solve(pair['input']), pair['output'])
        if kind != 'test':
            f = _facts(pair['input'])
            assert f['geometry_conflict'] and f['port_conflict'] and f['occlusion']
    # An equal-handoff upper/lower diamond has no legitimate tie break.
    pos = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    tie = {'pos': pos, 'rows': [1, 8, 15], 'cols': [4, 14], 'size': 4,
           'shape': (21, 21), 'ports': [{1: 1}, {2: 1}, {0: 1, 2: 1}, {}, {1: 1}, {0: 1}],
           'start': 2, 'goal': 3, 'nodes': [2, 0, 1, 3],
           'edges': [{'to': 0, 'direction': 0, 'gap': 3},
                     {'to': 1, 'direction': 1, 'gap': 6},
                     {'to': 3, 'direction': 2, 'gap': 3}]}
    tied = _render(rng, tie, (0, 2, 4, 7, 8))['input']
    try:
        solve(tied)
    except AssertionError:
        pass
    else:
        raise AssertionError('tied optimum accepted')
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(19000+10*n+m, n, m)
            assert len(p['train']) == n and len(p['test']) == m
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(1, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Corruption must be detected; solve never writes into the caller's input.
    p = generate_puzzle(991, 2, 1)
    before = p['test'][0]['input'].copy()
    solve(p['test'][0]['input'])
    assert np.array_equal(before, p['test'][0]['input'])
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted label accepted')
    totals.update(distinct_puzzles=len(signatures), color_normalized_inputs=len(structures),
                  grid_shapes=len(shapes), station_sizes=sorted(sizes), launcher_directions=sorted(directions),
                  palettes=len(palettes))
    return totals
