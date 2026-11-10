"""Minimum-station paths in readable networks, including branches and loops.

The source's graph is not necessarily a cycle. Eight-neighbor connector
continuity includes diagonal attachments; only solid square blocks are stations.
The two reserved output roles are learned across each whole puzzle.
"""
from collections import deque
from itertools import combinations
import heapq
import numpy as np


_NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _check_count(value, name, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer or None")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return int(value)


def _array_key(array):
    return array.shape, np.ascontiguousarray(array, dtype=np.int8).tobytes()


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
            for dr, dc in _NEIGHBORS:
                point = r + dr, c + dc
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        result.append(np.array(cells, dtype=np.int16))
    return result


def _square_size(cells):
    shape = cells.max(axis=0) - cells.min(axis=0) + 1
    return int(shape[0]) if shape[0] == shape[1] and len(cells) == int(np.prod(shape)) else 0


def _adjacency(n, edges):
    adjacency = [[] for _ in range(n)]
    for e, (a, b) in enumerate(edges):
        adjacency[a].append((b, e))
        adjacency[b].append((a, e))
    return adjacency


def _shortest(adjacency, start, target):
    """BFS with multiplicity, rather than an arbitrary tie-breaking traversal."""
    distance = [-1] * len(adjacency)
    ways = [0] * len(adjacency)
    previous = [None] * len(adjacency)
    distance[start], ways[start] = 0, 1
    queue = deque([start])
    while queue:
        a = queue.popleft()
        for b, edge in adjacency[a]:
            if distance[b] < 0:
                distance[b] = distance[a] + 1
                previous[b] = a, edge
                queue.append(b)
            if distance[b] == distance[a] + 1:
                ways[b] = min(2, ways[b] + ways[a])
    if ways[target] != 1:
        return None
    nodes, edges = [target], []
    while nodes[-1] != start:
        a, e = previous[nodes[-1]]
        nodes.append(a)
        edges.append(e)
    return nodes[::-1], edges[::-1]


def _weighted_cost(adjacency, weights, start, target):
    best = [float('inf')] * len(adjacency)
    best[start] = 0.
    queue = [(0., start)]
    while queue:
        cost, a = heapq.heappop(queue)
        if cost != best[a]:
            continue
        if a == target:
            return cost
        for b, e in adjacency[a]:
            candidate = cost + weights[e]
            if candidate < best[b]:
                best[b] = candidate
                heapq.heappush(queue, (candidate, b))
    return float('inf')


def _paths(adjacency, start, target):
    """Independent exhaustive simple-path search used for semantic checks."""
    result = []
    stack = [(start, [start], [])]
    while stack:
        a, nodes, edges = stack.pop()
        if a == target:
            result.append((nodes, edges))
            continue
        for b, e in adjacency[a]:
            if b not in nodes:
                stack.append((b, nodes + [b], edges + [e]))
    return sorted(result, key=lambda p: len(p[0]))


def _geometry(grid, station_mask, connector):
    stations = _components(station_mask)
    sizes = [_square_size(c) for c in stations]
    assert len(stations) >= 5 and len(set(sizes)) == 1 and sizes[0] >= 2
    station_map = np.full(grid.shape, -1, dtype=np.int16)
    for i, cells in enumerate(stations):
        station_map[tuple(cells.T)] = i
    components = _components(grid == connector)
    edges = []
    h, w = grid.shape
    for cells in components:
        touching = set()
        for r, c in cells.tolist():
            for dr, dc in _NEIGHBORS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and station_map[rr, cc] >= 0:
                    touching.add(int(station_map[rr, cc]))
        assert len(touching) == 2, 'connector must join exactly two stations'
        edges.append(tuple(sorted(touching)))
    assert len(set(edges)) == len(edges), 'ambiguous parallel connectors'
    adjacency = _adjacency(len(stations), edges)
    assert all(1 <= len(a) <= 3 for a in adjacency)
    reached, stack = {0}, [0]
    while stack:
        for b, _ in adjacency[stack.pop()]:
            if b not in reached:
                reached.add(b)
                stack.append(b)
    assert len(reached) == len(stations)
    centers = [c.mean(axis=0) for c in stations]
    lengths = [float(np.linalg.norm(centers[a] - centers[b])) for a, b in edges]
    return dict(stations=stations, components=components, edges=edges,
                adjacency=adjacency, station_size=sizes[0],
                areas=[len(c) for c in components], lengths=lengths)


def _read_input(grid):
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 4
    # The background is dominant, but may be interrupted at any canvas edge.
    background = int(colors[int(np.argmax(counts))])
    by_color = {int(c): _components(grid == c) for c in colors if c != background}
    endpoint_candidates = [c for c, cc in by_color.items() if len(cc) == 2
                           and len(set(_square_size(x) for x in cc)) == 1
                           and _square_size(cc[0]) >= 2]
    assert len(endpoint_candidates) == 1
    endpoint = endpoint_candidates[0]
    size = _square_size(by_color[endpoint][0])
    ordinary_candidates = [c for c, cc in by_color.items() if c != endpoint
                           and len(cc) >= 3 and all(_square_size(x) == size for x in cc)]
    assert len(ordinary_candidates) == 1
    ordinary = ordinary_candidates[0]
    connector = next(c for c in by_color if c not in (endpoint, ordinary))
    data = _geometry(grid, (grid == endpoint) | (grid == ordinary), connector)
    endpoints = [i for i, cells in enumerate(data['stations']) if grid[tuple(cells[0])] == endpoint]
    assert len(endpoints) == 2
    path = _shortest(data['adjacency'], *endpoints)
    assert path is not None, 'shortest station route is tied'
    data.update(roles=(background, connector, ordinary, endpoint), endpoints=endpoints, path=path)
    return data


def _paint(grid, data, highlights, path=None):
    nodes, edges = data['path'] if path is None else path
    result = grid.copy()
    for n in nodes[1:-1]:
        result[tuple(data['stations'][n].T)] = highlights[0]
    for e in edges:
        result[tuple(data['components'][e].T)] = highlights[1]
    return result


def _learn_highlights(train):
    stations, connectors = set(), set()
    for pair in train:
        data = _read_input(pair['input'])
        for role, target in ((data['roles'][2], stations), (data['roles'][1], connectors)):
            changed = (pair['input'] == role) & (pair['input'] != pair['output'])
            target.update(int(c) for c in np.unique(pair['output'][changed]))
    assert len(stations) == len(connectors) == 1
    highlights = next(iter(stations)), next(iter(connectors))
    assert highlights[0] != highlights[1]
    return highlights


def solve(train_pairs, test_input):
    """Predict using demonstrations and a query input only; no seed or test label."""
    highlights = _learn_highlights(train_pairs)
    data = _read_input(test_input)
    paths = _paths(data['adjacency'], *data['endpoints'])
    assert paths and (len(paths) == 1 or len(paths[0][0]) < len(paths[1][0]))
    return _paint(test_input, data, highlights, paths[0])


def _witness(data, path, endpoints):
    if len(path[0]) < 3:
        return False
    # Both pixel area and center-to-center geometric distance must disagree.
    return all(sum(data[name][e] for e in path[1]) >
               _weighted_cost(data['adjacency'], data[name], *endpoints) + 0.01
               for name in ('areas', 'lengths'))


def _line(a, b):
    delta = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    if not delta:
        return [tuple(a)]
    return [tuple(int(round(a[k] + (b[k] - a[k]) * t / delta)) for k in (0, 1))
            for t in range(delta + 1)]


def _brush_path(points, width):
    cells = set()
    for a, b in zip(points, points[1:]):
        for r, c in _line(a, b):
            cells.update((r + dr, c + dc) for dr in range(width) for dc in range(width))
    return cells


def _sample_graph(rng, size, richness, easy=False, small=False):
    max_axis = min(5, ((19 if small else 28) - size) // (size + 3) + 1)
    nr, nc = int(rng.integers(2, max_axis + 1)), int(rng.integers(2, max_axis + 1))
    vertices = [(r, c) for r in range(nr) for c in range(nc)]
    candidates = [(a, b) for a, b in combinations(vertices, 2)
                  if abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1]
    if easy:
        active = {v for v in vertices if v[0] in (0, nr-1) or v[1] in (0, nc-1)}
        edges = [(a, b) for a, b in candidates if a in active and b in active
                 and ((a[0] == b[0] and a[0] in (0, nr-1))
                      or (a[1] == b[1] and a[1] in (0, nc-1)))]
    else:
        target = int(rng.integers(max(6, len(vertices)//2), len(vertices)+1)) if len(vertices) >= 6 else len(vertices)
        active = {vertices[int(rng.integers(len(vertices)))]}
        edges, degrees = [], {v: 0 for v in vertices}
        while len(active) < target:
            choices = [(a, b) for a, b in candidates if (a in active) != (b in active)
                       and degrees[a] < 3 and degrees[b] < 3]
            if not choices:
                break
            a, b = choices[int(rng.integers(len(choices)))]
            edges.append((a, b))
            active.update((a, b))
            degrees[a] += 1
            degrees[b] += 1
        extras = [(a, b) for a, b in candidates if a in active and b in active and (a, b) not in edges]
        rng.shuffle(extras)
        loops = int(rng.integers(1, richness + 3))
        for a, b in extras:
            if loops and degrees[a] < 3 and degrees[b] < 3:
                edges.append((a, b))
                degrees[a] += 1
                degrees[b] += 1
                loops -= 1
    degree = {v: sum(v in e for e in edges) for v in active}
    visible = {v for v in active if degree[v] != 2 or rng.random() > rng.uniform(.15, .65)}
    missing = sorted(active - visible)
    rng.shuffle(missing)
    while len(visible) < min(5, len(active)):
        visible.add(missing.pop())
    assert len(visible) >= 5
    return nr, nc, active, edges, visible


def _fallback_graph():
    # A guaranteed long/sparse versus short/dense witness, with an unused branch.
    active = {(r, c) for r in (0, 1) for c in range(4)} | {(2, 2)}
    edges = [((r, c), (r, c+1)) for r in (0, 1) for c in range(3)]
    edges += [((0, 0), (1, 0)), ((0, 3), (1, 3)), ((1, 2), (2, 2))]
    visible = {(0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (2, 2)}
    return 3, 4, active, edges, visible


def _embedding(rng, graph, size, width, fallback=False, small=False):
    nr, nc, active, edges, visible = graph
    pitch = size + 3
    if fallback:
        rows = [0, 2*pitch, 3*pitch]
        cols = [i*pitch for i in range(4)]
    else:
        def axis(n):
            base = (n-1)*pitch + size
            slack = 0 if rng.random() < .3 else int(rng.integers(0, max(1, (20 if small else 29)-base)))
            gaps = rng.multinomial(slack, [1/(n-1)]*(n-1)) + pitch
            return [0] + list(np.cumsum(gaps).astype(int))
        rows, cols = axis(nr), axis(nc)
    anchors = {v: (int(rows[v[0]]), int(cols[v[1]])) for v in active}
    roomy = min(np.diff(rows)) >= size+5 and min(np.diff(cols)) >= size+5
    if not fallback and roomy:
        # Independent local jitter creates diagonal runs without changing topology.
        anchors = {v: (p[0]+int(rng.integers(0, 2)), p[1]+int(rng.integers(0, 2)))
                   for v, p in anchors.items()}
    neighbors = {v: [b if a == v else a for a,b in edges if v in (a,b)] for v in active}
    offsets = {v: {'N': size-width, 'E': size-width, 'S': 0, 'W': 0} for v in active}
    def direction(a, b):
        return ('E' if b[1] > a[1] else 'W') if a[0] == b[0] else ('S' if b[0] > a[0] else 'N')
    def separated(v):
        # Port footprints of distinct incident edges must not meet diagonally
        # around a station corner. This allows straight and slanted edges to mix.
        external = []
        for other in neighbors[v]:
            side = direction(v, other); off = offsets[v][side]
            r,c = {'N': (-width,off), 'E': (off,size),
                   'S': (size,off), 'W': (off,-width)}[side]
            external.append([(r+dr,c+dc) for dr in range(width) for dc in range(width)])
        return all(max(abs(a[0]-b[0]),abs(a[1]-b[1])) > 1
                   for aa,bb in combinations(external,2) for a in aa for b in bb)
    order = list(edges)
    if not fallback:
        rng.shuffle(order)
        for a,b in order:
            if rng.random() > .7: continue
            da,db = direction(a,b),direction(b,a)
            axis = 0 if da in ('E','W') else 1
            lo,hi = max(anchors[a][axis],anchors[b][axis]), min(anchors[a][axis],anchors[b][axis])+size-width
            alignments = list(range(lo,hi+1)); rng.shuffle(alignments)
            old_a,old_b = offsets[a][da],offsets[b][db]
            for coordinate in alignments:
                offsets[a][da] = coordinate-anchors[a][axis]
                offsets[b][db] = coordinate-anchors[b][axis]
                if separated(a) and separated(b): break
                offsets[a][da],offsets[b][db] = old_a,old_b
    # Start from separated clockwise ports, then independently align safe edges.
    def port(v, direction):
        r, c = anchors[v]
        off = offsets[v][direction]
        return {'N': (r, c+off), 'E': (r+off, c+size-width),
                'S': (r+size-width, c+off), 'W': (r+off, c)}[direction]
    stubs, edge_cells = {v: [] for v in active}, []
    def diagonal_safe(v):
        ns = neighbors[v]
        return len(ns) == 1 or (len(ns) == 2 and
               ns[0][0]+ns[1][0] == 2*v[0] and ns[0][1]+ns[1][1] == 2*v[1])
    for a, b in edges:
        if a[0] == b[0]:
            if a[1] > b[1]: a, b = b, a
            pa, pb = port(a, 'E'), port(b, 'W')
            qa, qb = (pa[0], anchors[a][1]+size), (pb[0], anchors[b][1]-width)
            horizontal = True
        else:
            if a[0] > b[0]: a, b = b, a
            pa, pb = port(a, 'S'), port(b, 'N')
            qa, qb = (anchors[a][0]+size, pa[1]), (anchors[b][0]-width, pb[1])
            horizontal = False
        if width == 1 and not fallback:
            # Corner-only attachment is visible under the source's eight-neighbor
            # convention. Keep it separate from the adjacent clockwise port.
            if diagonal_safe(a) and rng.random() < .25:
                qa = (qa[0]+1, qa[1]) if horizontal else (qa[0], qa[1]-1)
            if diagonal_safe(b) and rng.random() < .25:
                qb = (qb[0]-1, qb[1]) if horizontal else (qb[0], qb[1]+1)
        points = [pa, qa]
        style = 0 if fallback else int(rng.integers(0, 4))
        room = abs(qb[1]-qa[1]) if horizontal else abs(qb[0]-qa[0])
        if style == 1 and room >= 3:
            # An elbow is connector geometry, not a station.
            points.append((qa[0], qb[1]) if horizontal else (qb[0], qa[1]))
        elif style >= 2 and room >= 4:
            middle = ((qa[0]+qb[0])//2, (qa[1]+qb[1])//2)
            shift = int(rng.integers(-1, 2))
            middle = (middle[0]+shift, middle[1]) if horizontal else (middle[0], middle[1]+shift)
            points.append(middle)
        points.extend([qb, pb])
        edge_cells.append(_brush_path(points, width))
        stubs[a].append(pa)
        stubs[b].append(pb)
    all_cells = set().union(*edge_cells)
    for v in active-visible:
        assert len(stubs[v]) == 2
        all_cells.update(_brush_path(stubs[v], width))
    blocks = {v: {(anchors[v][0]+r, anchors[v][1]+c) for r in range(size) for c in range(size)}
              for v in visible}
    foreground = all_cells | set().union(*blocks.values())
    low = np.min(list(foreground), axis=0)
    high = np.max(list(foreground), axis=0)
    shape = high-low+1
    assert max(shape) <= (20 if small else 30)
    grid = np.zeros(tuple(shape), dtype=np.int8)
    for r, c in all_cells: grid[r-low[0], c-low[1]] = 1
    for block in blocks.values():
        for r, c in block: grid[r-low[0], c-low[1]] = 2
    data = _geometry(grid, grid == 2, 1)
    # Check the compressed construction graph against actual pixel connectivity.
    adjacency = {v: [] for v in active}
    for a, b in edges: adjacency[a].append(b); adjacency[b].append(a)
    visible_order = sorted(visible, key=lambda v: anchors[v])
    ids = {v: i for i, v in enumerate(visible_order)}
    expected = set()
    for v in visible:
        for nxt in adjacency[v]:
            prev, cur = v, nxt
            while cur not in visible:
                choices = [x for x in adjacency[cur] if x != prev]
                assert len(choices) == 1
                prev, cur = cur, choices[0]
            assert cur != v
            expected.add(tuple(sorted((ids[v], ids[cur]))))
    assert len(expected)*2 == sum(len(adjacency[v]) for v in visible), 'parallel suppressed routes'
    assert set(data['edges']) == expected, 'accidental edge contact or missing connection'
    assert len(data['stations']) == len(visible)
    return grid, data


def _place(rng, grid, size_regime):
    grid = np.rot90(grid, int(rng.integers(4)))
    if rng.random() < .5: grid = grid[:, ::-1]
    h, w = grid.shape
    if size_regime == 'large':
        hh, ww = int(rng.integers(max(24, h), 31)), int(rng.integers(max(24, w), 31))
    elif size_regime == 'small':
        hh, ww = int(rng.integers(max(15, h), 21)), int(rng.integers(max(15, w), 21))
    else:
        # Includes compact drawings with generous and asymmetric empty margins.
        hh, ww = int(rng.integers(h, 31)), int(rng.integers(w, 31))
        if rng.random() < .3:
            hh, ww = min(30, h+int(rng.integers(4))), min(30, w+int(rng.integers(4)))
    if size_regime != 'small' and rng.random() < .1: hh = ww = 30
    top = int(rng.integers(hh-h+1)); left = int(rng.integers(ww-w+1))
    placed = np.zeros((hh, ww), dtype=np.int8)
    placed[top:top+h, left:left+w] = grid
    return placed


def _make_pair(rng, size, width, richness, highlights, roles, witness=False, large=False, small=False):
    target = int(rng.choice([0, 1, 2, 3, 4, 5, 6, 7], p=[.07, .17, .20, .18, .16, .10, .08, .04]))
    desired_count = int(rng.choice([5, 7, 10, 13, 16, 19], p=[.18, .15, .22, .20, .15, .10]))
    if small or size == 4: desired_count = min(desired_count, 13)
    for attempt in range(101):
        fallback = attempt == 100
        try:
            graph = _fallback_graph() if fallback else _sample_graph(rng, size, richness, rng.random() < .15, small)
            grid, data = _embedding(rng, graph, size, width, fallback, small)
            if attempt < 60 and len(data['stations']) < desired_count-1:
                continue
            choices = []
            for a, b in combinations(range(len(data['stations'])), 2):
                path = _shortest(data['adjacency'], a, b)
                if path is None or (witness and not _witness(data, path, (a, b))):
                    continue
                choices.append((abs(len(path[0])-2-target), a, b))
            if not choices: continue
            distance = min(x[0] for x in choices)
            if not witness and target >= 5 and distance and attempt < 85:
                continue
            choices = [x for x in choices if x[0] == distance]
            _, a, b = choices[int(rng.integers(len(choices)))]
            for n in (a, b): grid[tuple(data['stations'][n].T)] = 3
            grid = _place(rng, grid, 'large' if large else 'small' if small else 'any')
            lookup = np.array(roles, dtype=np.int8)
            inp = lookup[grid]
            data = _read_input(inp)
            assert not witness or _witness(data, data['path'], data['endpoints'])
            paths = _paths(data['adjacency'], *data['endpoints'])
            assert len(paths) == 1 or len(paths[0][0]) < len(paths[1][0])
            out = _paint(inp, data, highlights, paths[0])
            assert _array_key(inp) not in _SOURCE_INPUT_KEYS
            return {'input': inp, 'output': out}
        except AssertionError:
            if fallback: raise
    raise AssertionError('bounded construction exhausted')


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    num_train = _check_count(num_train, 'num_train', 2, 10)
    num_test = _check_count(num_test, 'num_test', 1, 4)
    rng = np.random.default_rng(seed)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test+1), 11))
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train-1)+1))
    if num_train <= num_test: raise ValueError('num_train must exceed num_test')
    highlights = tuple(int(x) for x in rng.choice(10, 2, replace=False))
    pool = [c for c in range(10) if c not in highlights]
    palette_mode = int(rng.choice(3, p=[.65, .20, .15]))  # fresh, recurring, consistent
    base_roles = tuple(int(x) for x in rng.choice(pool, 4, replace=False))
    regime = int(rng.integers(4))  # source shift, mixed, same scale, recombination
    richness = int(rng.integers(1, 4))
    common_size = int(rng.choice([2, 3, 4], p=[.45, .40, .15]))
    common_width = int(rng.integers(1, min(common_size, 3)))
    train_scales = []
    if regime == 0: train_scales = [(2, 1)] * num_train
    elif regime == 2: train_scales = [(common_size, common_width)] * num_train
    else:
        for _ in range(num_train):
            size = int(rng.choice([2, 3, 4], p=[.50, .40, .10]))
            train_scales.append((size, int(rng.integers(1, min(size, 3)))))
        if len(set(train_scales)) == 1:
            train_scales[-1] = (3, 1) if train_scales[0][0] == 2 else (2, 1)
    test_scales = []
    for _ in range(num_test):
        if regime == 0: scale = (3, 2)
        elif regime == 2: scale = (common_size, common_width)
        elif regime == 3: scale = train_scales[int(rng.integers(len(train_scales)))]
        else:
            size = int(rng.choice([2, 3, 4], p=[.4, .45, .15]))
            scale = size, int(rng.integers(1, min(size, 3)))
        test_scales.append(scale)
    witness_index = int(rng.integers(num_train))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, scales in (('train', train_scales), ('test', test_scales)):
        for i, (size, width) in enumerate(scales):
            roles = base_roles if palette_mode == 2 or (palette_mode == 1 and rng.random() < .65) else tuple(
                int(x) for x in rng.choice(pool, 4, replace=False))
            for _ in range(20):
                pair = _make_pair(rng, size, width, richness, highlights, roles,
                                  witness=(split == 'train' and i == witness_index) or
                                          (split == 'test' and rng.random() < .2),
                                  large=regime == 0 and split == 'test',
                                  small=regime == 0 and split == 'train')
                key = _array_key(pair['input'])
                if key not in seen:
                    seen.add(key); puzzle[split].append(pair); break
            else: raise AssertionError('duplicate inputs after bounded retries')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, analyses = set(), {'train': [], 'test': []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.all((0 <= grid) & (grid <= 9))
            assert pair['input'].shape == pair['output'].shape
            key = _array_key(pair['input'])
            assert key not in seen and key not in _SOURCE_INPUT_KEYS
            seen.add(key)
            analyses[split].append(_read_input(pair['input']))
    highlights = _learn_highlights(puzzle['train'])
    for split in puzzle:
        for pair, data in zip(puzzle[split], analyses[split]):
            assert set(highlights).isdisjoint(data['roles'])
            assert np.array_equal(pair['output'], _paint(pair['input'], data, highlights))
            assert not np.array_equal(pair['input'], pair['output'])
    assert any(_witness(d, d['path'], d['endpoints']) for d in analyses['train'])
    for pair in puzzle['test']:
        assert np.array_equal(solve(puzzle['train'], pair['input'].copy()), pair['output'])
    # Scale has no semantic effect. Source-shift, mixed and equal scales are legal
    # in both splits; all tests share the visible rule and the learned color roles.


def features(pair):
    """Task-specific measurements reconstructed from the returned pixels."""
    grid = pair['input']; d = _read_input(grid)
    paths = _paths(d['adjacency'], *d['endpoints'])
    nodes, edges = paths[0]
    degrees = [len(a) for a in d['adjacency']]
    fg = np.argwhere(grid != d['roles'][0]); low, high = fg.min(0), fg.max(0)
    nonrect, diagonal, diagonal_attachment = 0, 0, 0
    widths = []
    for cells in d['components']:
        span = cells.max(0)-cells.min(0)+1
        nonrect += int(len(cells) != int(np.prod(span)))
        mask = set(map(tuple, cells.tolist()))
        diagonal += int(any(not any((r+dr,c+dc) in mask for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)))
                            for r,c in mask) and len(mask)>1)
        # Maximum inscribed square detects the narrow-band scale even at bends.
        widths.append(max((k for k in range(1, 4) if any(
            all((r+dr,c+dc) in mask for dr in range(k) for dc in range(k)) for r,c in mask)), default=1))
    for e, (a,b) in enumerate(d['edges']):
        cc = set(map(tuple, d['components'][e].tolist()))
        for n in (a,b):
            ss = set(map(tuple, d['stations'][n].tolist()))
            if not any((r+dr,c+dc) in ss for r,c in cc for dr,dc in ((1,0),(-1,0),(0,1),(0,-1))):
                diagonal_attachment += 1
    common = set(edges)
    for _, other_edges in paths[1:]: common.intersection_update(other_edges)
    unused = sum(1 for n in nodes[1:-1] if degrees[n] >= 3 and any(e not in edges for _,e in d['adjacency'][n]))
    branch_lengths = []
    for leaf, degree in enumerate(degrees):
        if degree != 1: continue
        previous, current, length = leaf, d['adjacency'][leaf][0][0], 1
        while degrees[current] == 2:
            nxt = next(b for b,_ in d['adjacency'][current] if b != previous)
            previous, current, length = current, nxt, length+1
        branch_lengths.append(length)
    centers = [c.mean(axis=0) for c in d['stations']]
    connected = {tuple(sorted(e)) for e in d['edges']}
    proximity_decoy = any(
        tuple(sorted((a,x))) not in connected and x != a and
        np.linalg.norm(centers[a]-centers[x]) + .01 < np.linalg.norm(centers[a]-centers[b])
        for a,b in zip(nodes,nodes[1:]) for x in range(len(centers)))
    return dict(roles=list(d['roles']), station_size=d['station_size'],
                connector_width=min(widths), station_count=len(degrees),
                selected_stations=len(nodes)-2, route_counts=[len(p[0])-2 for p in paths],
                route_areas=[sum(d['areas'][e] for e in p[1]) for p in paths],
                route_lengths=[round(sum(d['lengths'][e] for e in p[1]), 3) for p in paths],
                route_count=len(paths), loop_rank=len(d['edges'])-len(degrees)+1,
                degrees=sorted(degrees), dead_ends=degrees.count(1), junctions=degrees.count(3),
                ordinary_dead_ends=sum(degrees[n] == 1 for n in range(len(degrees)) if n not in d['endpoints']),
                branch_lengths=sorted(branch_lengths),
                endpoint_junctions=sum(degrees[n] == 3 for n in d['endpoints']),
                shared_route_edges=len(common) if len(paths)>1 else 0,
                highlighted_unused_junctions=unused, nonrectangular_edges=nonrect,
                straight_edges=len(d['edges'])-nonrect,
                proximity_decoy=proximity_decoy,
                diagonal_edges=diagonal, diagonal_attachments=diagonal_attachment,
                witness=_witness(d, d['path'], d['endpoints']),
                margins=[int(low[0]),int(low[1]),int(grid.shape[0]-1-high[0]),int(grid.shape[1]-1-high[1])],
                shape=list(grid.shape))


def _puzzles_equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k]) for p,q in zip(a[s], b[s])
               for k in ('input','output')) for s in ('train','test'))


def self_test() -> None:
    # These source regressions include branching, diagonal attachments, a shared
    # route prefix/suffix, and the 26-versus-19-versus-9 connector-area witness.
    source = _source_pairs()
    for i, pair in enumerate(source):
        assert np.array_equal(solve([p for j,p in enumerate(source[:3]) if j != i], pair['input']), pair['output'])
    counts, test_counts, stats, structures, highlight_roles = set(), set(), [], set(), set()
    within = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        counts.add(len(puzzle['train'])); test_counts.add(len(puzzle['test']))
        highlight_roles.add(_learn_highlights(puzzle['train']))
        if seed < 3: assert _puzzles_equal(puzzle, generate_puzzle(seed))
        local = {'train': [], 'test': []}
        for split in ('train','test'):
            for pair in puzzle[split]:
                f = features(pair); f['split'] = split; stats.append(f)
                local[split].append(f)
                grid = pair['input']; fg = np.argwhere(grid != f['roles'][0]); lo,hi=fg.min(0),fg.max(0)
                semantic = np.zeros(grid.shape, dtype=np.int8)
                for role,c in enumerate(f['roles']): semantic[grid == c] = role
                structures.add(_array_key(semantic[lo[0]:hi[0]+1,lo[1]:hi[1]+1]))
                d = _read_input(grid); paths = _paths(d['adjacency'], *d['endpoints'])
                assert len(paths) == 1 or len(paths[0][0]) < len(paths[1][0])
                assert np.array_equal(_paint(grid,d,_learn_highlights(puzzle['train']),paths[0]), pair['output'])
        tr, te = local['train'], local['test']
        ts = {(s['station_size'],s['connector_width']) for s in tr}
        es = {(s['station_size'],s['connector_width']) for s in te}
        roles = {tuple(s['roles']) for s in tr+te}
        within.append(dict(mixed=len(ts)>1, same=len(ts|es)==1,
                           recombination=len(ts)>1 and es <= ts,
                           source_shift=ts == {(2,1)} and es == {(3,2)} and
                           all(15 <= n <= 20 for s in tr for n in s['shape']) and
                           all(24 <= n <= 30 for s in te for n in s['shape']),
                           consistent=len(roles)==1, recurring=1 < len(roles) < len(tr+te),
                           fresh=len({tuple(sorted(s['roles'])) for s in tr+te}) >= 3,
                           counts=len({s['station_count'] for s in tr+te}),
                           shapes=len({tuple(s['shape']) for s in tr+te})))
    assert len(counts) >= 4 and len(test_counts) >= 3 and len(structures) >= 200
    assert len(highlight_roles) >= 20
    for axis in ('mixed','same','recombination','source_shift','consistent','recurring','fresh'):
        assert any(p[axis] for p in within), axis
    assert sum(p['counts'] >= 3 for p in within) >= 30
    assert sum(p['shapes'] >= 3 for p in within) >= 30
    assert any(s['shape'] == [30,30] for s in stats)
    assert min(min(s['shape']) for s in stats) <= 14
    assert any(s['witness'] and s['junctions'] and s['proximity_decoy'] for s in stats if s['split']=='train')
    for split in ('train', 'test'):
        rows = [s for s in stats if s['split'] == split]
        assert any(s['loop_rank'] >= 2 and s['junctions'] and s['dead_ends'] for s in rows)
        assert any(s['route_count'] >= 3 for s in rows)
        assert any(s['shared_route_edges'] and s['highlighted_unused_junctions'] for s in rows)
        assert any(s['endpoint_junctions'] for s in rows)
        assert any(s['ordinary_dead_ends'] for s in rows)
        assert max(k for s in rows for k in s['branch_lengths']) >= 3
        assert any(s['selected_stations'] == 0 for s in rows)
        assert max(s['selected_stations'] for s in rows) >= 5
        assert max(s['station_count'] for s in rows) >= 14
        assert any(s['nonrectangular_edges'] and s['diagonal_edges'] for s in rows)
        assert any(s['nonrectangular_edges'] and s['straight_edges'] and s['junctions'] for s in rows)
        assert any(s['diagonal_attachments'] for s in rows)
        assert any(s['route_count'] > 1 and s['route_counts'][1]-s['route_counts'][0] == 1 for s in rows)
        assert any(s['route_count'] > 1 and s['route_counts'][1]-s['route_counts'][0] >= 3 for s in rows)
        assert any(0 in s['margins'] for s in rows)
        assert any(max(s['margins']) >= 5 for s in rows)
        assert {(2,1),(3,1),(3,2),(4,1)} <= {(s['station_size'],s['connector_width']) for s in rows}
    for i, (nt, ne) in enumerate(((2,1),(10,1),(10,4),(None,1),(None,4),(2,None),(10,None))):
        p = generate_puzzle(500+i, nt, ne)
        assert nt is None or len(p['train']) == nt
        assert ne is None or len(p['test']) == ne
        assert _puzzles_equal(p, generate_puzzle(500+i, nt, ne))
    for nt, ne in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(True,1)):
        try: generate_puzzle(10, nt, ne)
        except ValueError: continue
        raise AssertionError('illegal counts accepted')
    # Reject altered off-route pixels, partial blocks, and hidden output colors.
    p = generate_puzzle(81, 2, 1)
    p['test'][0]['output'][0,0] = (int(p['test'][0]['output'][0,0])+1)%10
    try: validate_generated_puzzle(p)
    except AssertionError: pass
    else: raise AssertionError('corrupted output accepted')
    found_tie = False
    for pair in source:
        d = _read_input(pair['input'])
        for a,b in combinations(range(len(d['stations'])), 2):
            if _shortest(d['adjacency'], a,b) is not None: continue
            tied = pair['input'].copy()
            tied[tied == d['roles'][3]] = d['roles'][2]
            for n in (a,b): tied[tuple(d['stations'][n].T)] = d['roles'][3]
            try: _read_input(tied)
            except AssertionError: found_tie = True
            else: raise AssertionError('tied shortest routes accepted')
    assert found_tie


# Exact source contents for regression and copy exclusion; dimensions stay legal.
_SOURCE_ROWS = ((('888888888888888888',
   '888888802288888118',
   '888888082288888118',
   '882280888088880808',
   '882208888088808808',
   '880888888088808808',
   '880888888088808808',
   '888088888088088808',
   '888088888808228808',
   '888118888808228808',
   '888118888808888808',
   '888088888808888088',
   '888808888808880888',
   '888808888822808888',
   '888822880022088888',
   '888822008888888888',
   '888888888888888888',
   '888888888888888888'),
  ('888888888888888888',
   '888888802288888118',
   '888888082288888118',
   '882280888088880858',
   '882208888088808858',
   '880888888088808858',
   '880888888088808858',
   '888088888088088858',
   '888088888808228858',
   '888118888808228858',
   '888118888808888858',
   '888588888808888588',
   '888858888808885888',
   '888858888833858888',
   '888833885533588888',
   '888833558888888888',
   '888888888888888888',
   '888888888888888888')),
 (('999999477499999',
   '999774977946699',
   '999779999996649',
   '999494999999994',
   '999499499999994',
   '946699949999949',
   '496699949999779',
   '499999977444779',
   '499999977999949',
   '497749994999949',
   '497794999499499',
   '499994999499499',
   '949997799779499',
   '994447799779779',
   '999999999999779'),
  ('999999533599999',
   '999335933956699',
   '999339999996649',
   '999594999999994',
   '999599499999994',
   '946699949999949',
   '496699949999779',
   '499999977444779',
   '499999977999949',
   '497749994999949',
   '497794999499499',
   '499994999499499',
   '949997799779499',
   '994447799779779',
   '999999999999779')),
 (('77777777777777777777',
   '77116666667777777777',
   '77117777776666777777',
   '77677777777777677777',
   '77677777777772277777',
   '77677777777762277777',
   '77677777777677677777',
   '77677777771177677777',
   '76677777771177677777',
   '76777777776777677777',
   '76777771176777767777',
   '76777761167777767777',
   '76777677777777776777',
   '76771177777777776777',
   '76771166777777767777',
   '76776777667777117777',
   '76767777776666117777',
   '72277777777777777777',
   '72277777777777777777',
   '77777777777777777777'),
  ('77777777777777777777',
   '77335555557777777777',
   '77337777775555777777',
   '77577777777777577777',
   '77577777777772277777',
   '77577777777762277777',
   '77577777777677677777',
   '77577777771177677777',
   '75577777771177677777',
   '75777777776777677777',
   '75777771176777767777',
   '75777761167777767777',
   '75777677777777776777',
   '75771177777777776777',
   '75771166777777767777',
   '75776777667777117777',
   '75767777776666117777',
   '72277777777777777777',
   '72277777777777777777',
   '77777777777777777777')),
 (('444444444444444444444444444444',
   '444444444444444444444444444444',
   '449991111111999111111114444444',
   '449991111111999111111114444444',
   '449994444444999444444114444444',
   '441144444441144444444999444444',
   '441144444411144444444999444444',
   '441144444111444444441999444444',
   '441144000114444444411411444444',
   '441144000444444444114441144444',
   '444114000444444441144444114444',
   '444114411444444411444444411444',
   '444114411444499914444444999444',
   '444114411444499944444444999444',
   '444114411444499944444444999444',
   '444999411444411444444444114444',
   '444999411444411444444441144444',
   '444999411444411444444411444444',
   '444444411444411444444000444444',
   '444444411444411444444000444444',
   '444444499944411444444000444444',
   '444444499944411444441144444444',
   '444444499944411444411444444444',
   '444444441144411444114444444444',
   '444444444114411441144444444444',
   '444444444411499911444444444444',
   '444444444441199914444444444444',
   '444444444444199944444444444444',
   '444444444444444444444444444444',
   '444444444444444444444444444444'),
  ('444444444444444444444444444444',
   '444444444444444444444444444444',
   '449991111111999111111114444444',
   '449991111111999111111114444444',
   '449994444444999444444114444444',
   '441144444441144444444999444444',
   '441144444411144444444999444444',
   '441144444111444444441999444444',
   '441144000114444444411411444444',
   '441144000444444444114441144444',
   '444114000444444441144444114444',
   '444114455444444411444444411444',
   '444114455444499914444444999444',
   '444114455444499944444444999444',
   '444114455444499944444444999444',
   '444999455444411444444444114444',
   '444999455444411444444441144444',
   '444999455444411444444411444444',
   '444444455444411444444000444444',
   '444444455444411444444000444444',
   '444444433344411444444000444444',
   '444444433344411444445544444444',
   '444444433344411444455444444444',
   '444444445544411444554444444444',
   '444444444554411445544444444444',
   '444444444455433355444444444444',
   '444444444445533354444444444444',
   '444444444444533344444444444444',
   '444444444444444444444444444444',
   '444444444444444444444444444444')))

def _source_pairs():
    return [{k: np.array([[int(c) for c in row] for row in rows], dtype=np.int8)
             for k, rows in zip(("input", "output"), pair)} for pair in _SOURCE_ROWS]

_SOURCE_INPUT_KEYS = frozenset(_array_key(p["input"]) for p in _source_pairs())
