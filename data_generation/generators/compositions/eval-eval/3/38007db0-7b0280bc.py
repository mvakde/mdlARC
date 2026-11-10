"""Repair the cheapest terminal-to-terminal route through four-window stations.

Only the standard library and NumPy are used. All solving conventions, including
the path highlight color, are visible in the query input. No parent is imported.
"""
from collections import Counter
from itertools import combinations
import heapq

import numpy as np


_D4 = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        group, stack = [first], [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D4:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    group.append(p)
                    stack.append(p)
        groups.append(np.array(sorted(group), dtype=np.int16))
    return groups


def _frame_mask(h, w):
    mask = np.zeros((h, w), dtype=bool)
    mask[[0, -1], :] = True
    mask[:, [0, -1]] = True
    mask[h // 2, :] = True
    mask[:, w // 2] = True
    return mask


def _frame_box(cells):
    low, high = cells.min(0), cells.max(0)
    h, w = map(int, high - low + 1)
    if h < 5 or w < 5 or h % 2 != 1 or w % 2 != 1:
        return None
    mask = _frame_mask(h, w)
    relative = cells - low
    if len(cells) != int(mask.sum()) or not np.all(mask[tuple(relative.T)]):
        return None
    return int(low[0]), int(low[1]), h, w


def _windows(box):
    r, c, h, w = box
    ph, pw = (h - 3) // 2, (w - 3) // 2
    return [(slice(r + 1 + i * (ph + 1), r + 1 + i * (ph + 1) + ph),
             slice(c + 1 + j * (pw + 1), c + 1 + j * (pw + 1) + pw))
            for i in range(2) for j in range(2)]


def _adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for e, (a, b) in enumerate(edges):
        adj[a].append((b, e))
        adj[b].append((a, e))
    return adj


def _shortest(adj, weights, start, target):
    """Node-weighted Dijkstra; count minimum routes, rejecting ties."""
    n = len(adj)
    dist, ways, previous = [float('inf')] * n, [0] * n, [None] * n
    dist[start], ways[start] = 0, 1
    queue = [(0, start)]
    while queue:
        cost, a = heapq.heappop(queue)
        if cost != dist[a] or a == target:
            continue
        for b, e in adj[a]:
            if b == start:
                continue
            candidate = cost + (0 if b == target else weights[b])
            if candidate < dist[b]:
                dist[b], ways[b], previous[b] = candidate, ways[a], (a, e)
                heapq.heappush(queue, (candidate, b))
            elif candidate == dist[b]:
                ways[b] = min(2, ways[b] + ways[a])
    if ways[target] != 1:
        return None
    nodes, edges = [target], []
    while nodes[-1] != start:
        a, e = previous[nodes[-1]]
        nodes.append(a)
        edges.append(e)
    return nodes[::-1], edges[::-1]


def _paths(adj, start, target):
    """Bounded exhaustive simple paths, used for generation and validation."""
    paths, stack = [], [(start, [start], [])]
    while stack:
        a, nodes, edges = stack.pop()
        if a == target:
            paths.append((nodes, edges))
        else:
            for b, e in adj[a]:
                if b not in nodes:
                    stack.append((b, nodes + [b], edges + [e]))
    return paths


def _read(input_grid):
    grid = _grid(input_grid)
    colors = list(map(int, np.unique(grid)))
    _require(len(colors) == 6, "six distinct semantic colors are required")
    groups = {c: _components(grid == c) for c in colors}
    frames = {}
    for color, cc in groups.items():
        boxes = [_frame_box(cells) for cells in cc]
        if boxes and all(b is not None for b in boxes) and len({b[2:] for b in boxes}) == 1:
            frames[color] = boxes
    ordinary = [c for c, boxes in frames.items() if len(boxes) >= 3]
    terminals = [c for c, boxes in frames.items() if len(boxes) == 2]
    _require(len(ordinary) == len(terminals) == 1, "station frame roles are ambiguous")
    ordinary, terminal = ordinary[0], terminals[0]
    _require(frames[ordinary][0][2:] == frames[terminal][0][2:], "unequal station sizes")
    boxes = sorted(frames[ordinary] + frames[terminal])
    _require(5 <= len(boxes) <= 12, "family has 5..12 stations")
    ph, pw = (boxes[0][2] - 3) // 2, (boxes[0][3] - 3) // 2
    _require(2 <= ph * pw <= 6, "window area must be 2..6")
    station_map = np.full(grid.shape, -1, dtype=np.int16)
    endpoints, weights, masters, odd, panels, interior = [], [], [], [], [], set()
    for i, box in enumerate(boxes):
        r, c, h, w = box
        _require(np.all(station_map[r:r+h, c:c+w] < 0), "overlapping stations")
        station_map[r:r+h, c:c+w] = i
        windows = _windows(box)
        values = [grid[window].copy() for window in windows]
        counts = Counter(_key(p) for p in values)
        master_key, multiplicity = counts.most_common(1)[0]
        master = next(p for p in values if _key(p) == master_key)
        _require(len(np.unique(master)) == 2, "master uses both interior colors")
        interior.update(map(int, np.unique(np.concatenate([p.ravel() for p in values]))))
        is_terminal = int(grid[r, c]) == terminal
        if is_terminal:
            _require(multiplicity == 4, "terminal windows must be identical")
            endpoints.append(i)
            odd.append(None)
            weights.append(0)
        else:
            _require(multiplicity == 3 and len(counts) == 2, "one visible odd window is required")
            index = next(j for j, p in enumerate(values) if not np.array_equal(p, master))
            odd.append(index)
            weights.append(int(np.count_nonzero(values[index] != master)))
        masters.append(master)
        panels.append(windows)
    _require(len(interior) == 2 and interior.isdisjoint({ordinary, terminal}), "bad interior palette")
    remaining = set(colors) - interior - {ordinary, terminal}
    _require(len(remaining) == 2, "bad color roles")
    candidates = []
    h, w = grid.shape
    for color in sorted(remaining):
        edges, edge_cells = [], groups[color]
        valid = True
        for cells in edge_cells:
            touching = set()
            for r, c in cells.tolist():
                for dr, dc in _D4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and station_map[rr, cc] >= 0:
                        touching.add(int(station_map[rr, cc]))
            if len(touching) != 2:
                valid = False
                break
            edges.append(tuple(sorted(touching)))
        if not valid or not edges or len(edges) != len(set(edges)):
            continue
        adj = _adjacency(len(boxes), edges)
        reached, stack = {0}, [0]
        while stack:
            for b, _ in adj[stack.pop()]:
                if b not in reached:
                    reached.add(b)
                    stack.append(b)
        if len(reached) == len(boxes) and all(1 <= len(a) <= 4 for a in adj):
            candidates.append((color, edges, edge_cells, adj))
    _require(len(candidates) == 1, "connector role or graph is ambiguous")
    connector, edges, edge_cells, adj = candidates[0]
    background = next(iter(remaining - {connector}))
    _require(bool(np.all(np.isin(grid[station_map < 0], [background, connector]))),
             "interior colors may not occur outside stations")
    path = _shortest(adj, weights, *endpoints)
    _require(path is not None, "minimum repair cost must be unique")
    return dict(grid=grid, boxes=boxes, panels=panels, masters=masters, odd=odd,
                weights=weights, endpoints=endpoints, edges=edges, edge_cells=edge_cells,
                adj=adj, path=path, terminal=terminal, ordinary=ordinary,
                connector=connector, background=background,
                interior=tuple(sorted(interior)))


def _paint(data, path=None):
    nodes, edges = data['path'] if path is None else path
    result = data['grid'].copy()
    for n in nodes[1:-1]:
        result[data['panels'][n][data['odd'][n]]] = data['masters'][n]
    for e in edges:
        result[tuple(data['edge_cells'][e].T)] = data['terminal']
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read local repetition, recover the graph, minimize repairs, and paint."""
    return _paint(_read(input_grid))


def _graph(rng, nr, nc, target):
    board = [(r, c) for r in range(nr) for c in range(nc)]
    possible = [(a, b) for a, b in combinations(board, 2)
                if abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1]
    active = {board[int(rng.integers(len(board)))]}
    edges = []
    while len(active) < target:
        choices = [(a, b) for a, b in possible if (a in active) != (b in active)]
        a, b = choices[int(rng.integers(len(choices)))]
        edges.append((a, b))
        active.update((a, b))
    extras = [(a, b) for a, b in possible if a in active and b in active and (a, b) not in edges]
    if not extras:
        return None
    rng.shuffle(extras)
    edges += extras[:int(rng.integers(1, len(extras) + 1))]
    return sorted(active), edges


def _route_choice(rng, vertices, edges, area, agree=False, conflict=False):
    ids = {v: i for i, v in enumerate(vertices)}
    adj = _adjacency(len(vertices), [(ids[a], ids[b]) for a, b in edges])
    endpoints = list(combinations(range(len(vertices)), 2))
    rng.shuffle(endpoints)
    weights = (list(map(int, rng.choice([1, 2, 3, 4], size=len(vertices), p=[.50, .10, .10, .30])))
               if conflict else list(map(int, rng.integers(1, area + 1, size=len(vertices)))))
    for a, b in endpoints:
        paths = _paths(adj, a, b)
        if len(paths) < 2 or min(len(p[0]) for p in paths) < 3:
            continue
        ordered = sorted(paths, key=lambda p: sum(weights[n] for n in p[0][1:-1]))
        best, second = ordered[:2]
        if sum(weights[n] for n in best[0][1:-1]) == sum(weights[n] for n in second[0][1:-1]):
            continue
        if len(best[0]) >= len(vertices):
            continue
        if agree and not (len(best[0]) == min(len(p[0]) for p in paths) < max(len(p[0]) for p in paths)):
            continue
        if agree and max(weights[n] for n in best[0][1:-1]) <= min(
                max(weights[n] for n in p[0][1:-1]) for p in paths):
            continue
        if conflict and len(best[0]) <= min(len(p[0]) for p in paths):
            continue
        return (a, b), weights, best
    return None


def _witness(rng):
    # A six-cycle: the one-station route costs 5, the three-station route 1+2+1.
    # An optional chord ends at the expensive station, so it cannot improve cost 4.
    ph, pw = [(2, 3), (3, 2), (1, 5), (5, 1)][int(rng.integers(4))]
    vertical = ph < pw
    nr, nc = (3, 2) if vertical else (2, 3)
    cycle = ([(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)] if vertical else
             [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)])
    vertices = sorted(cycle)
    edges = [(cycle[i], cycle[(i+1) % 6]) for i in range(6)]
    if rng.random() < .7:
        edges.append((cycle[1], cycle[4]))
    if (ph, pw) == (1, 5) and rng.random() < .65:
        nr = 4
        for col in range(2):
            if rng.random() < .75:
                vertices.append((3, col))
                edges.append(((2, col), (3, col)))
    if (ph, pw) == (5, 1) and rng.random() < .65:
        nc = 4
        for row in range(2):
            if rng.random() < .75:
                vertices.append((row, 3))
                edges.append(((row, 2), (row, 3)))
    vertices.sort()
    ids = {v: i for i, v in enumerate(vertices)}
    weights = [int(rng.integers(1, ph*pw+1)) for _ in vertices]
    for v, weight in zip(cycle, (0, 5, 0, 1, 2, 1)):
        weights[ids[v]] = weight
    modes = {ids[cycle[1]]: 'net_decoy', ids[cycle[3]]: 'add',
             ids[cycle[4]]: 'exchange', ids[cycle[5]]: 'remove'}
    return ph, pw, nr, nc, vertices, edges, (ids[cycle[0]], ids[cycle[2]]), weights, modes


def _panel(rng, ph, pw, cost=0, mode=None):
    area = ph * pw
    if mode is None:
        modes = ['flip']
        if 0 < cost < area:
            modes += ['add', 'remove']
        if cost > 0 and cost % 2 == 0:
            modes.append('exchange')
        mode = str(rng.choice(modes))
    low, high = 1, area - 1
    if mode == 'add':
        high = area - cost
    elif mode == 'remove':
        low = cost
    elif mode == 'exchange':
        low, high = cost // 2, area - cost // 2
    elif mode == 'net_decoy':
        low = high = area // 2
    count = int(rng.integers(low, high + 1))
    master = np.zeros(area, dtype=bool)
    master[rng.choice(area, count, replace=False)] = True
    variant = master.copy()
    if mode == 'add':
        variant[rng.choice(np.flatnonzero(~master), cost, replace=False)] = True
    elif mode == 'remove':
        variant[rng.choice(np.flatnonzero(master), cost, replace=False)] = False
    elif mode == 'exchange':
        k = cost // 2
        variant[rng.choice(np.flatnonzero(master), k, replace=False)] = False
        variant[rng.choice(np.flatnonzero(~master), k, replace=False)] = True
    elif cost:
        cells = rng.choice(area, cost, replace=False)
        variant[cells] = ~variant[cells]
    return master.reshape(ph, pw), variant.reshape(ph, pw)


def _axis(rng, count, size):
    slack = 30 - (count * size + 2 * (count - 1))
    extra = int(rng.integers(0, min(6, slack) + 1))
    gaps = 2 + rng.multinomial(extra, [1/(count-1)] * (count-1))
    starts = [0]
    for gap in gaps:
        starts.append(starts[-1] + size + int(gap))
    return starts


def _line(points):
    cells = set()
    for a, b in zip(points, points[1:]):
        _require(a[0] == b[0] or a[1] == b[1], "only orthogonal drawing steps")
        distance = abs(a[0]-b[0]) + abs(a[1]-b[1])
        dr, dc = np.sign(b[0]-a[0]), np.sign(b[1]-a[1])
        cells.update((int(a[0]+i*dr), int(a[1]+i*dc)) for i in range(distance+1))
    return cells


def _draw(rng, ph, pw, nr, nc, vertices, edges, endpoints, weights, modes):
    sh, sw = 2*ph+3, 2*pw+3
    rows, cols = _axis(rng, nr, sh), _axis(rng, nc, sw)
    shape = (rows[-1]+sh, cols[-1]+sw)
    grid = np.zeros(shape, dtype=np.int8)
    boxes = [(rows[r], cols[c], sh, sw) for r, c in vertices]
    ids = {v: i for i, v in enumerate(vertices)}
    edge_masks = []
    for a, b in edges:
        if a > b:
            a, b = b, a
        ra, ca, _, _ = boxes[ids[a]]
        rb, cb, _, _ = boxes[ids[b]]
        horizontal = a[0] == b[0]
        if horizontal:
            start, end = (ra+sh//2, ca+sw), (rb+sh//2, cb-1)
        else:
            start, end = (ra+sh, ca+sw//2), (rb-1, cb+sw//2)
        gap = abs(end[0]-start[0]) + abs(end[1]-start[1]) + 1
        points = [start, end]
        if gap >= 5 and rng.random() < .5:
            off = int(rng.choice([-1, 1]))
            r, c = start
            rr, cc = end
            points = ([start, (r, c+1), (r+off, c+1), (rr+off, cc-1), (rr, cc-1), end]
                      if horizontal else
                      [start, (r+1, c), (r+1, c+off), (rr-1, cc+off), (rr-1, cc), end])
        cells = _line(points)
        edge_masks.append(cells)
        for r, c in cells:
            grid[r, c] = 5
    masters, odd = [], []
    for n, box in enumerate(boxes):
        r, c, h, w = box
        frame = 2 if n in endpoints else 1
        grid[r:r+h, c:c+w][_frame_mask(h, w)] = frame
        master, variant = _panel(rng, ph, pw, 0 if n in endpoints else weights[n], modes.get(n))
        master, variant = 3 + master.astype(np.int8), 3 + variant.astype(np.int8)
        exception = int(rng.integers(4))
        for j, window in enumerate(_windows(box)):
            grid[window] = variant if n not in endpoints and j == exception else master
        masters.append(master)
        odd.append(exception)
    ordinary_ids = [n for n in range(len(vertices)) if n not in endpoints]
    if len({_key(masters[n]) for n in ordinary_ids}) == 1:
        # Guarantee visible evidence for local rather than whole-network templates.
        n = max(ordinary_ids, key=lambda n: weights[n])
        masters[n] = 7 - masters[n]
        for window in _windows(boxes[n]):
            grid[window] = 7 - grid[window]
    adj = _adjacency(len(vertices), [(ids[a], ids[b]) for a, b in edges])
    path = _shortest(adj, weights, *endpoints)
    _require(path is not None, "construction has tied cost")
    out = grid.copy()
    for n in path[0][1:-1]:
        out[_windows(boxes[n])[odd[n]]] = masters[n]
    for e in path[1]:
        for r, c in edge_masks[e]:
            out[r, c] = 2
    recovered = _read(grid)
    expected_edges = {tuple(sorted((ids[a], ids[b]))) for a, b in edges}
    _require(set(recovered['edges']) == expected_edges, "drawing changed intended graph")
    _require(np.array_equal(out, _paint(recovered)), "drawing changed intended answer")
    # Margins, rotations and reflections do not encode cost or output roles.
    pad_h = int(rng.integers(0, min(5, 30-grid.shape[0]) + 1))
    pad_w = int(rng.integers(0, min(5, 30-grid.shape[1]) + 1))
    top, left = int(rng.integers(pad_h+1)), int(rng.integers(pad_w+1))
    padding = ((top, pad_h-top), (left, pad_w-left))
    grid, out = np.pad(grid, padding), np.pad(out, padding)
    turns, reflect = int(rng.integers(4)), rng.random() < .5
    grid, out = np.rot90(grid, turns), np.rot90(out, turns)
    if reflect:
        grid, out = grid[:, ::-1], out[:, ::-1]
    palette = rng.choice(10, 6, replace=False).astype(np.int8)
    return {'input': palette[grid].copy(), 'output': palette[out].copy()}


def _ordinary_pair(rng, test=False, agree=False):
    conflict = test and rng.random() < .45
    cores = [(1, 2), (2, 1), (2, 2), (1, 3), (3, 1)]
    if test:
        cores = [(2, 2)] if conflict else [(1, 2), (2, 1), (2, 2)]
    options = [(ph, pw, nr, nc) for ph, pw in cores
               for nr in range(2, 5) for nc in range(2, 5)
               if nr*(2*ph+3)+2*(nr-1) <= 30
               and nc*(2*pw+3)+2*(nc-1) <= 30
               and nr*nc >= (9 if test else 6)]
    for _ in range(160):
        ph, pw, nr, nc = options[int(rng.integers(len(options)))]
        target = int(rng.integers(9 if test else 5, min(nr*nc, 12 if test else 8)+1))
        graph = _graph(rng, nr, nc, target)
        if graph is None:
            continue
        vertices, edges = graph
        chosen = _route_choice(rng, vertices, edges, ph*pw, agree, conflict)
        if chosen is None:
            continue
        endpoints, weights, _ = chosen
        return _draw(rng, ph, pw, nr, nc, vertices, edges, endpoints, weights, {})
    # A deterministic constructive fallback is still in the requested split.
    if agree:
        ph, pw, nr, nc = 1, 2, 2, 3
        cycle = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 1), (1, 0)]
        vertices = sorted(cycle)
        ids = {v: i for i, v in enumerate(vertices)}
        edges = [(cycle[i], cycle[(i+1) % 6]) for i in range(6)]
        weights = [1] * 6
        weights[ids[cycle[1]]] = 2
        return _draw(rng, ph, pw, nr, nc, vertices, edges,
                     (ids[cycle[0]], ids[cycle[2]]), weights, {})
    ph, pw, nr, nc = ((2, 2, 3, 3) if conflict else
                      (1, 2, 3, 3) if test else (2, 2, 2, 3))
    vertices = [(r, c) for r in range(nr) for c in range(nc)]
    edges = [(a, b) for a, b in combinations(vertices, 2)
             if abs(a[0]-b[0])+abs(a[1]-b[1]) == 1]
    # Adjacent sides of a square: one intermediate cheap station, one expensive.
    # Other lattice routes cannot improve on this cost of one.
    endpoints = (vertices.index((0, 0)), vertices.index((0, 2) if conflict else (1, 1)))
    weights = [4 if conflict else 2] * len(vertices)
    for v in ([(1, 0), (1, 1), (1, 2)] if conflict else [(0, 1)]):
        weights[vertices.index(v)] = 1
    return _draw(rng, ph, pw, nr, nc, vertices, edges, endpoints, weights, {})


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(name + ' out of range')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed))
    witness_index = int(rng.integers(nt))
    agree_index = (witness_index + int(rng.integers(1, nt))) % nt
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            for _ in range(24):
                pair = (_draw(rng, *_witness(rng)) if split == 'train' and i == witness_index else
                        _ordinary_pair(rng, test=split == 'test', agree=split == 'train' and i == agree_index))
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded uniqueness retries exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(data):
    paths = _paths(data['adj'], *data['endpoints'])
    costs = [sum(data['weights'][n] for n in p[0][1:-1]) for p in paths]
    best = int(np.argmin(costs))
    chosen = paths[best]
    hops = [len(p[0])-2 for p in paths]
    areas = [sum(len(data['edge_cells'][e]) for e in p[1]) for p in paths]
    net_weights = []
    for n, master in enumerate(data['masters']):
        if n in data['endpoints']:
            net_weights.append(0)
        else:
            variant = data['grid'][data['panels'][n][data['odd'][n]]]
            color = data['interior'][0]
            net_weights.append(abs(int(np.count_nonzero(master == color)) -
                                   int(np.count_nonzero(variant == color))))
    net_costs = [sum(net_weights[n] for n in p[0][1:-1]) for p in paths]
    maximum_costs = [max((data['weights'][n] for n in p[0][1:-1]), default=0) for p in paths]
    balanced = adds = removes = 0
    for n in chosen[0][1:-1]:
        master = data['masters'][n]
        variant = data['grid'][data['panels'][n][data['odd'][n]]]
        color = data['interior'][0]
        a = int(np.count_nonzero((master == color) & (variant != color)))
        b = int(np.count_nonzero((master != color) & (variant == color)))
        balanced += int(a == b and a > 0)
        adds += int(a > 0 and b == 0)
        removes += int(b > 0 and a == 0)
    # The two interior colors are interchangeable; require changes in both directions.
    return dict(stations=len(data['boxes']), routes=len(paths), costs=sorted(costs),
                chosen_cost=costs[best], chosen_hops=hops[best], min_hops=min(hops),
                hop_conflict=hops[best] > min(hops),
                pixel_conflict=areas[best] > min(areas),
                net_conflict=net_costs[best] > min(net_costs),
                maximum_conflict=maximum_costs[best] > min(maximum_costs),
                agree=hops[best] == min(hops) < max(hops),
                balanced=balanced, one_way_edits=adds+removes, adds=adds, removes=removes,
                core_shape=data['masters'][0].shape,
                degrees=sorted(map(len, data['adj'])),
                distinct_masters=len({_key(p) for n, p in enumerate(data['masters'])
                                      if n not in data['endpoints']}),
                selected_odd_positions=[data['odd'][n] for n in chosen[0][1:-1]])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "illegal pair counts")
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "bad pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "output must preserve dimensions")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _read(inp)
            paths = _paths(d['adj'], *d['endpoints'])
            ranked = sorted(paths, key=lambda p: sum(d['weights'][n] for n in p[0][1:-1]))
            _require(len(ranked) >= 2, "need competing routes")
            costs = [sum(d['weights'][n] for n in p[0][1:-1]) for p in ranked]
            _require(costs[0] < costs[1], "repair cost tie")
            _require(ranked[0] == d['path'], "Dijkstra disagrees with path enumeration")
            _require(3 <= len(d['path'][0]) < len(d['boxes']), "selected and unselected ordinary stations required")
            _require(np.array_equal(out, _paint(d, ranked[0])), "wrong repairs or changed off-route pixels")
            _require(not np.array_equal(inp, out), "identity output")
            f = _features(d)
            _require(f['distinct_masters'] >= 2, "demonstrate station-local masters")
            _require(5 <= f['stations'] <= 8 if split == 'train' else 9 <= f['stations'] <= 12,
                     "generation split station-count shift")
            features[split].append(f)
    train = features['train']
    _require(any(f['hop_conflict'] and f['pixel_conflict'] and f['net_conflict'] and f['balanced'] and
                 f['adds'] and f['removes'] and f['distinct_masters'] >= 2 for f in train),
             "missing cost-versus-hop/pixel witness with exchange and one-way repairs")
    _require(any(f['agree'] and f['maximum_conflict'] for f in train),
             "also demonstrate minimum total cost defeating minimum worst defect")
    return None


def _puzzle_key(puzzle):
    return tuple(tuple((_key(_grid(p['input'])), _key(_grid(p['output']))) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    """42-seed structural coverage, ablations, count boundaries and negative tests."""
    stats, structures, palettes, shapes, puzzles = [], set(), set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _read(pair['input'])
                f = _features(d)
                f['split'] = split
                stats.append(f)
                roles, normalized = {}, []
                for c in pair['input'].flat:
                    normalized.append(roles.setdefault(int(c), len(roles)))
                structures.add((pair['input'].shape, tuple(normalized)))
                shapes.add(pair['input'].shape)
                palettes.add((d['ordinary'], d['terminal'], d['connector'], d['background']))
                _require(np.array_equal(solve(pair['input'].tolist()), pair['output']), "list solve failed")
                # Omitting B repairs all ordinary stations, including an off-route exception.
                repair_all = pair['output'].copy()
                for n in range(len(d['boxes'])):
                    if n not in d['endpoints']:
                        repair_all[d['panels'][n][d['odd'][n]]] = d['masters'][n]
                _require(not np.array_equal(repair_all, pair['output']), "B ablation was invisible")
    _require(len(set(puzzles)) == 42 and len(structures) >= 190, "insufficient structural diversity")
    _require(len(shapes) >= 25 and len(palettes) >= 100, "insufficient geometry/palette diversity")
    _require(len({f['core_shape'] for f in stats}) >= 8, "insufficient panel geometry variation")
    _require(any(4 in f['degrees'] for f in stats), "no four-way junction")
    _require(any(1 in f['degrees'] for f in stats), "no dead end")
    _require(any(f['routes'] >= 4 for f in stats), "too few route alternatives")
    _require(any(f['hop_conflict'] for f in stats if f['split'] == 'test'), "no test hop conflict")
    _require(any(f['balanced'] for f in stats if f['split'] == 'test'), "no test balanced repair")
    for puzzle in (generate_puzzle(123, 2, 1), generate_puzzle(123, 10, 4)):
        _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(123, len(puzzle['train']), len(puzzle['test']))),
                 "seed reproducibility failed")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            generate_puzzle(400 + a*10 + b, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # A ablation: every minimum-hop path differs from the mandatory witness answer.
    base = generate_puzzle(0, 2, 1)
    for pair in base['train']:
        d = _read(pair['input'])
        f = _features(d)
        if f['hop_conflict']:
            paths = _paths(d['adj'], *d['endpoints'])
            minimum = min(len(p[0]) for p in paths)
            _require(all(not np.array_equal(_paint(d, p), pair['output'])
                         for p in paths if len(p[0]) == minimum), "A ablation was invisible")
    for defect in ('output', 'missing_odd', 'terminal'):
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in base[s]] for s in base}
        pair = bad['train'][0]
        d = _read(pair['input'])
        n = next(n for n in range(len(d['boxes'])) if n not in d['endpoints'])
        if defect == 'output':
            pair['output'][0, 0] = (int(pair['output'][0, 0]) + 1) % 10
        elif defect == 'missing_odd':
            pair['input'][d['panels'][n][d['odd'][n]]] = d['masters'][n]
        else:
            r, c, _, _ = d['boxes'][d['endpoints'][0]]
            pair['input'][r, c] = d['ordinary']
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('validator accepted ' + defect)
    # Explicit tied diamond, plus dangling nodes: cost equality must not use a tie-break.
    adj = _adjacency(5, [(0, 1), (1, 3), (0, 2), (2, 3), (2, 4)])
    _require(_shortest(adj, [0, 1, 1, 0, 2], 0, 3) is None, "tied routes accepted")
    return {'seeds': 42, 'pairs': len(stats), 'input_shapes': len(shapes),
            'normalized_inputs': len(structures), 'role_palettes': len(palettes),
            'panel_shapes': len({f['core_shape'] for f in stats})}


if __name__ == '__main__':
    print(self_test())
