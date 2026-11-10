"""Repeated transit tiles: exact substitution creates a minimum-station network.

All solving conventions are visible in a single input, except the fixed output
colors 8 (station) and 9 (wire). No parent code or fixtures are imported.
"""
from collections import deque
import heapq

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # N,E,S,W


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grids must be integer 1..30 rectangles with colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    left = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while left:
        start = min(left)
        left.remove(start)
        stack, cells = [start], []
        while stack:
            p = stack.pop()
            cells.append(p)
            for dr, dc in DIRS:
                q = p[0] + dr, p[1] + dc
                if q in left:
                    left.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _box(mask):
    cells = np.argwhere(mask)
    lo, hi = cells.min(0), cells.max(0) + 1
    return tuple(map(int, (lo[0], hi[0], lo[1], hi[1])))


def _crop(a, box):
    r0, r1, c0, c1 = box
    return a[r0:r1, c0:c1].copy()


def _separate(a, b):
    return a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2]


def _adjacent_pair(mask):
    return bool(np.any(mask[:-1] & mask[1:]) or np.any(mask[:, :-1] & mask[:, 1:]))


def _tile_info(tile, bg):
    """Recover the hub, relay role, wire role, and four actual arm paths."""
    colors = [int(c) for c in np.unique(tile) if c != bg]
    if len(colors) != 3:
        raise ValueError("tile needs hub, relay, and wire roles")
    anchors = [c for c in colors if np.count_nonzero(tile == c) == 1]
    wires = [c for c in colors if _adjacent_pair(tile == c)]
    if len(anchors) != 1 or len(wires) != 1 or anchors[0] == wires[0]:
        raise ValueError("tile dot/wire roles must be distinguishable")
    anchor, wire = anchors[0], wires[0]
    ordinary = next(c for c in colors if c not in (anchor, wire))
    if np.count_nonzero(tile == ordinary) < 2 or _adjacent_pair(tile == ordinary):
        raise ValueError("relay dots must be isolated and repeated")
    root = tuple(map(int, np.argwhere(tile == anchor)[0]))
    h, w = tile.shape
    if not (0 < root[0] < h - 1 and 0 < root[1] < w - 1):
        raise ValueError("hub must be interior")
    fg = set(map(tuple, np.argwhere(tile != bg).tolist()))
    ports = []
    for side in ([(0, c) for c in range(w)], [(r, w-1) for r in range(h)],
                 [(h-1, c) for c in range(w)], [(r, 0) for r in range(h)]):
        found = [p for p in side if p in fg]
        if len(found) != 1 or tile[found[0]] != wire:
            raise ValueError("each side needs exactly one wire port")
        ports.append(found[0])
    if len(set(ports)) != 4 or ports[0][1] != ports[2][1] or ports[1][0] != ports[3][0]:
        raise ValueError("opposite ports must match")
    neighbors = {p: [(p[0]+dr, p[1]+dc) for dr, dc in DIRS
                     if (p[0]+dr, p[1]+dc) in fg] for p in fg}
    if any(len(ns) != (4 if p == root else 1 if p in ports else 2)
           for p, ns in neighbors.items()):
        raise ValueError("tile must be a four-arm tree with no unmarked junctions")
    arms, reached = {}, {root}
    for nxt in neighbors[root]:
        prev, cur, path = root, nxt, [root, nxt]
        while cur not in ports:
            candidates = [p for p in neighbors[cur] if p != prev]
            if len(candidates) != 1 or candidates[0] in reached or candidates[0] in path:
                raise ValueError("tile has a loop or intersecting arms")
            prev, cur = cur, candidates[0]
            path.append(cur)
        side = ports.index(cur)
        arms[side] = path
        reached.update(path)
    if reached != fg or len(arms) != 4:
        raise ValueError("all tile content must belong to the four arms")
    relays = [sum(tile[p] == ordinary for p in arms[d]) for d in range(4)]
    return dict(root=root, anchor=anchor, ordinary=ordinary, wire=wire,
                arms=arms, ports=ports, relays=relays,
                wh=1+relays[1]+relays[3], wv=1+relays[0]+relays[2])


def _read_input(value):
    grid = _grid(value)
    colors, counts = np.unique(grid, return_counts=True)
    if len(colors) != 6 or np.any(colors >= 8) or counts.max() <= grid.size / 2:
        raise ValueError("input needs six distinct roles and a majority background")
    bg = int(colors[np.argmax(counts)])
    candidates = []
    endpoints = [int(c) for c, n in zip(colors, counts) if c != bg and n == 2]
    for endpoint in endpoints:
        for signal in map(int, colors):
            if signal in (bg, endpoint) or np.count_nonzero(grid == signal) < 3:
                continue
            mb = _box((grid == signal) | (grid == endpoint))
            sb = _box((grid != bg) & (grid != signal) & (grid != endpoint))
            if not _separate(mb, sb):
                continue
            mask, tile = _crop(grid, mb), _crop(grid, sb)
            if (not set(map(int, np.unique(mask))) <= {bg, signal, endpoint}
                    or signal in tile or endpoint in tile
                    or mask.shape[0]*tile.shape[0] > 30
                    or mask.shape[1]*tile.shape[1] > 30):
                continue
            active = mask != bg
            if len(_components(active)) != 1:
                continue
            try:
                info = _tile_info(tile, bg)
            except ValueError:
                continue
            info.update(grid=grid, bg=bg, signal=signal, endpoint=endpoint,
                        mask=mask, active=active, tile=tile, mask_box=mb, tile_box=sb,
                        terminals=list(map(tuple, np.argwhere(mask == endpoint).tolist())))
            candidates.append(info)
    if len(candidates) != 1:
        raise ValueError("encodings need exactly one visible interpretation")
    return candidates[0]


def _expand(d):
    tile, mask, bg = d['tile'], d['mask'], d['bg']
    h, w = tile.shape
    board = np.full((mask.shape[0]*h, mask.shape[1]*w), bg, dtype=np.int8)
    for r, c in np.argwhere(d['active']).tolist():
        board[r*h:(r+1)*h, c*w:(c+1)*w] = tile
    terminals = []
    for r, c in d['terminals']:
        p = r*h+d['root'][0], c*w+d['root'][1]
        board[p] = d['endpoint']
        terminals.append(p)
    return board, terminals


def _station_graph(board, d):
    """Contract actual rendered wire components; open stubs are retained separately."""
    station_mask = np.isin(board, [d['anchor'], d['ordinary'], d['endpoint']])
    nodes = list(map(tuple, np.argwhere(station_mask).tolist()))
    ids = {p: i for i, p in enumerate(nodes)}
    edges, edge_cells, stubs, seen = [], [], [], set()

    def add(a, b, cells):
        pair = tuple(sorted((a, b)))
        if pair in seen:
            raise ValueError("parallel or ambiguous station edges")
        seen.add(pair)
        edges.append(pair)
        edge_cells.append(cells)

    for cells in _components(board == d['wire']):
        touching = {ids[q] for r, c in cells for dr, dc in DIRS
                    if (q := (r+dr, c+dc)) in ids}
        if len(touching) == 2:
            a, b = sorted(touching)
            add(a, b, cells)
        elif len(touching) == 1:
            stubs.append(cells)
        else:
            raise ValueError("a wire component must be an edge or an open stub")
    for p, a in ids.items():
        for dr, dc in DIRS:
            q = p[0]+dr, p[1]+dc
            if q in ids and a < ids[q]:
                add(a, ids[q], [])
    adj = [[] for _ in nodes]
    for e, (a, b) in enumerate(edges):
        adj[a].append((b, e))
        adj[b].append((a, e))
    return dict(nodes=nodes, ids=ids, edges=edges, cells=edge_cells, stubs=stubs, adj=adj)


def _bfs(g, start, target):
    dist, ways, prev = [-1]*len(g['nodes']), [0]*len(g['nodes']), {}
    dist[start], ways[start] = 0, 1
    q = deque([start])
    while q:
        a = q.popleft()
        for b, e in g['adj'][a]:
            if dist[b] < 0:
                dist[b] = dist[a]+1
                prev[b] = a, e
                q.append(b)
            if dist[b] == dist[a]+1:
                ways[b] = min(2, ways[b]+ways[a])
    if any(x < 0 for x in dist) or ways[target] != 1:
        raise ValueError("network disconnected or minimum-station path tied")
    nodes, edges = [target], []
    while nodes[-1] != start:
        a, e = prev[nodes[-1]]
        nodes.append(a)
        edges.append(e)
    return nodes[::-1], edges[::-1]


def _solve_details(value):
    d = _read_input(value)
    board, terminals = _expand(d)
    g = _station_graph(board, d)
    nodes, edges = _bfs(g, *(g['ids'][p] for p in terminals))
    out, painted = board.copy(), set()
    for n in nodes[1:-1]:
        p = g['nodes'][n]
        out[p] = 8
        painted.add(p)
    for e in edges:
        for p in g['cells'][e]:
            out[p] = 9
            painted.add(p)
    return out, d, board, g, nodes, edges, painted


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve one input using rendered-pixel station BFS, without generation state."""
    return _solve_details(input_grid)[0]


def _coarse_search(active, start, wh, wv):
    cells = set(map(tuple, np.argwhere(active).tolist()))
    dist, ways, prev = {start: 0}, {start: 1}, {}
    q = [(0, start)]
    while q:
        cost, a = heapq.heappop(q)
        if cost != dist[a]:
            continue
        for dr, dc in DIRS:
            b = a[0]+dr, a[1]+dc
            if b not in cells:
                continue
            candidate = cost + (wv if dr else wh)
            if candidate < dist.get(b, float('inf')):
                dist[b], ways[b], prev[b] = candidate, ways[a], a
                heapq.heappush(q, (candidate, b))
            elif candidate == dist.get(b):
                ways[b] = min(2, ways[b]+ways[a])
    return dist, ways, prev


def _coarse_route(active, start, target, wh, wv):
    dist, ways, prev = _coarse_search(active, start, wh, wv)
    if ways.get(target) != 1:
        return None
    path = [target]
    while path[-1] != start:
        path.append(prev[path[-1]])
    return path[::-1], dist[target]


def _loop_rank(active):
    n = int(active.sum())
    edges = int(np.count_nonzero(active[:-1] & active[1:])
                + np.count_nonzero(active[:, :-1] & active[:, 1:]))
    return edges-n+1


def _is_witness(d):
    s, t = d['terminals']
    short = _coarse_route(d['active'], s, t, 1, 1)
    best = _coarse_route(d['active'], s, t, d['wh'], d['wv'])
    if short is None or best is None or len(best[0]) <= len(short[0]):
        return False
    def physical(path):
        return sum(d['tile'].shape[0] if a[0] != b[0] else d['tile'].shape[1]
                   for a, b in zip(path, path[1:]))
    def wire_walk(path):
        return sum(len(d['arms'][0])+len(d['arms'][2])-1 if a[0] != b[0]
                   else len(d['arms'][1])+len(d['arms'][3])-1
                   for a, b in zip(path, path[1:]))
    return physical(best[0]) > physical(short[0]) and wire_walk(best[0]) > wire_walk(short[0])


def _polyline(points):
    """Unit orthogonal steps, excluding the initial point and including the end."""
    path = []
    for a, b in zip(points, points[1:]):
        dr = (b[0] > a[0]) - (b[0] < a[0])
        dc = (b[1] > a[1]) - (b[1] < a[1])
        distance = abs(b[0]-a[0])+abs(b[1]-a[1])
        assert not (dr and dc)
        path.extend((a[0]+i*dr, a[1]+i*dc) for i in range(1, distance+1))
    return path


def _make_tile(rng, roles, kind):
    bg, _, _, anchor, ordinary, wire = roles
    if kind in ('witness', 'contrast'):
        h, w = int(rng.integers(5, 8)), 5
        root = int(rng.integers(2, h-2)), 2
        er, nc = root
        if kind == 'contrast' and min(root[0], h-1-root[0]) >= 3:
            nc += int(rng.choice([-1, 1]))
    else:
        shapes = [(h, w) for h in range(5, 11) for w in range(5, 11)]
        if kind == 'bent':
            shapes = [(h, w) for h, w in shapes if min(h, w) >= 7 and h != w]
        h, w = shapes[int(rng.integers(len(shapes)))]
        if kind == 'bent':
            roots = [(r, c) for r in range(3, h-3) for c in range(3, w-3)
                     if 2*r != h-1 or 2*c != w-1]
            root = roots[int(rng.integers(len(roots)))]
        else:
            root = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
        er, nc = root
        if min(root[1], w-1-root[1]) >= 3 and (kind == 'bent' or rng.random() < .6):
            er += int(rng.choice([x for x in (-1, 1) if 1 <= er+x <= h-2]))
        if min(root[0], h-1-root[0]) >= 3 and (kind == 'bent' or rng.random() < .6):
            nc += int(rng.choice([x for x in (-1, 1) if 1 <= nc+x <= w-2]))
    r, c = root
    # Outward arms use separate corridors and optionally turn near the sides.
    paths = [_polyline([root, (1, c), (1, nc), (0, nc)]),
             _polyline([root, (r, w-2), (er, w-2), (er, w-1)]),
             _polyline([root, (h-2, c), (h-2, nc), (h-1, nc)]),
             _polyline([root, (r, 1), (er, 1), (er, 0)])]
    for _ in range(80):
        tile = np.full((h, w), bg, dtype=np.int8)
        tile[root] = anchor
        for side, path in enumerate(paths):
            for p in path:
                tile[p] = wire
            eligible = path[:-1]
            if kind == 'witness':
                chosen = [eligible[int(rng.integers(len(eligible)))]] if side in (0, 2) else []
            elif kind == 'contrast':
                chosen = []
                if side in (0, 2):
                    target = int(rng.integers(1, 3))
                    for k in rng.permutation(len(eligible)):
                        p = eligible[int(k)]
                        if len(chosen) < target and all(abs(p[0]-q[0])+abs(p[1]-q[1]) > 1 for q in chosen):
                            chosen.append(p)
            else:
                chosen = []
                order = list(rng.permutation(len(eligible)))
                target = int(rng.integers(0, (len(eligible)+1)//2+1))
                for k in order:
                    p = eligible[int(k)]
                    if len(chosen) < target and all(abs(p[0]-q[0])+abs(p[1]-q[1]) > 1 for q in chosen):
                        chosen.append(p)
            for p in chosen:
                tile[p] = ordinary
        try:
            _tile_info(tile, bg)
        except ValueError:
            continue
        return tile
    raise RuntimeError("bounded tile construction exhausted")


def _make_mask(rng, max_shape, kind):
    if kind in ('witness', 'contrast'):
        base = np.array([[1,1,1,1,1,1], [1,0,0,0,1,0],
                         [1,1,1,0,1,0], [1,0,1,1,1,1]], dtype=bool)
        s, t = (2, 2), (0, 5)
        if kind == 'contrast':
            base[3, 1] = True  # Add a loop without changing the metric conflict.
        for _ in range(30):
            mask = base.copy()
            for r, c in np.argwhere(rng.random(base.shape) < (.16 if kind == 'contrast' else .08)).tolist():
                if (r, c) not in (s, t):
                    mask[r, c] = ~mask[r, c]
            if len(_components(mask)) != 1 or not all(mask.any(axis=a).all() for a in (0, 1)):
                continue
            if kind == 'contrast':
                if _loop_rank(mask) >= 2:
                    return mask, None
                continue
            short = _coarse_route(mask, s, t, 1, 1)
            best = _coarse_route(mask, s, t, 1, 3)
            if short and best and len(best[0]) > len(short[0]):
                return mask, (s, t)
        return base, None if kind == 'contrast' else (s, t)
    options = [(h, w) for h in range(2 if kind == 'tree' else 3, max_shape[0]+1)
               for w in range(2 if kind == 'tree' else 3, max_shape[1]+1) if h*w >= 6]
    for _ in range(100):
        h, w = options[int(rng.integers(len(options)))]
        if kind == 'tree':
            mask = np.zeros((h, w), dtype=bool)
            mask[int(rng.integers(h)), int(rng.integers(w))] = True
            target = int(rng.integers(5, max(6, h*w*3//4+1)))
            while int(mask.sum()) < target:
                choices = [(r, c) for r in range(h) for c in range(w) if not mask[r, c]
                           and sum(0 <= r+dr < h and 0 <= c+dc < w and mask[r+dr, c+dc]
                                   for dr, dc in DIRS) == 1]
                if not choices:
                    break
                mask[choices[int(rng.integers(len(choices)))]] = True
        else:
            mask = rng.random((h, w)) < float(rng.uniform(.58, .95))
            if mask.sum() < 5 or len(_components(mask)) != 1:
                continue
        if mask.sum() < 5:
            continue
        mask = _crop(mask, _box(mask))
        if min(mask.shape) < 2 or (kind == 'test' and _loop_rank(mask) < 2):
            continue
        return mask, None
    # Connected tree and multi-loop constructive fallbacks fit every size regime.
    mask = np.array([[1,1,0], [0,1,1], [0,0,1]], dtype=bool) if kind == 'tree' else np.ones((3,3), dtype=bool)
    if kind != 'tree':
        mask[0, 0] = False
    return mask, None


def _choose_endpoints(rng, active, tile, bg, both_axes=False, contrast=False):
    info = _tile_info(tile, bg)
    cells = list(map(tuple, np.argwhere(active).tolist()))
    candidates = []
    for i, a in enumerate(cells):
        _, ways, prev = _coarse_search(active, a, info['wh'], info['wv'])
        if contrast:
            short_dist, short_ways, _ = _coarse_search(active, a, 1, 1)
        for b in cells[i+1:]:
            if ways.get(b) != 1:
                continue
            path = [b]
            while path[-1] != a:
                path.append(prev[path[-1]])
            if not 3 <= len(path) < len(cells):
                continue
            if both_axes and (len({p[0] for p in path}) < 2 or len({p[1] for p in path}) < 2):
                continue
            if contrast and (short_ways.get(b) != 1 or len(path)-1 <= short_dist[b]):
                continue
            candidates.append((a, b))
    if not candidates:
        return None
    return candidates[int(rng.integers(len(candidates)))]


def _place(rng, mask, tile, bg):
    horizontal, reverse = bool(rng.integers(2)), bool(rng.integers(2))
    mh, mw = mask.shape
    th, tw = tile.shape
    gap = int(rng.integers(1, 5))
    base_h, base_w = (max(mh, th), mw+tw+gap) if horizontal else (mh+th+gap, max(mw, tw))
    h = int(rng.integers(base_h, min(30, base_h+8)+1))
    w = int(rng.integers(base_w, min(30, base_w+8)+1))
    foreground = int(np.count_nonzero(mask != bg)+np.count_nonzero(tile != bg))
    while h*w <= 2*foreground:
        if h < 30:
            h += 1
        else:
            w += 1
    if horizontal:
        offset = int(rng.integers(w-base_w+1))
        mc, tc = (offset+tw+gap, offset) if reverse else (offset, offset+mw+gap)
        mr, tr = int(rng.integers(h-mh+1)), int(rng.integers(h-th+1))
    else:
        offset = int(rng.integers(h-base_h+1))
        mr, tr = (offset+th+gap, offset) if reverse else (offset, offset+mh+gap)
        mc, tc = int(rng.integers(w-mw+1)), int(rng.integers(w-tw+1))
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[mr:mr+mh, mc:mc+mw] = mask
    grid[tr:tr+th, tc:tc+tw] = tile
    return grid


def _label_from_coarse_path(d):
    """Generation labels use tile-arm concatenation, not the solver's pixel BFS."""
    board, terminals = _expand(d)
    route = _coarse_route(d['active'], *d['terminals'], d['wh'], d['wv'])
    if route is None:
        raise ValueError("tied coarse weighted route")
    h, w = d['tile'].shape
    cells = set()
    for a, b in zip(route[0], route[0][1:]):
        direction = DIRS.index((b[0]-a[0], b[1]-a[1]))
        for cell, side in ((a, direction), (b, (direction+2) % 4)):
            cells.update((cell[0]*h+r, cell[1]*w+c) for r, c in d['arms'][side])
    out = board.copy()
    for p in cells:
        if p in terminals:
            continue
        out[p] = 9 if board[p] == d['wire'] else 8
    return out


def _make_pair(rng, roles, kind):
    bg, signal, endpoint, _, _, _ = roles
    for _ in range(100):
        tile_kind = kind if kind in ('witness', 'contrast') else 'bent' if kind == 'tree' else 'random'
        tile = _make_tile(rng, roles, tile_kind)
        active, ends = _make_mask(rng, (30//tile.shape[0], 30//tile.shape[1]), kind)
        ends = ends or _choose_endpoints(rng, active, tile, bg, kind in ('test', 'contrast'), kind == 'contrast')
        if ends is None:
            continue
        mask = np.where(active, signal, bg).astype(np.int8)
        for p in ends:
            mask[p] = endpoint
        turns = int(rng.integers(4))
        mask, tile = np.rot90(mask, turns), np.rot90(tile, turns)
        if rng.random() < .5:
            mask, tile = np.fliplr(mask), np.fliplr(tile)
        inp = _place(rng, mask, tile, bg)
        d = _read_input(inp)
        if kind in ('witness', 'contrast') and not _is_witness(d):
            continue
        return {'input': inp, 'output': _label_from_coarse_path(d)}
    raise RuntimeError("bounded pair construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    stable = bool(rng.random() < .2)
    common = tuple(map(int, rng.choice(8, 6, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            kind = ('witness' if i == 0 else 'tree' if i == 1 else str(rng.choice(['tree', 'random']))) if split == 'train' else ('contrast' if i == 0 else 'test')
            for _ in range(20):
                roles = common if stable else tuple(map(int, rng.choice(8, 6, replace=False)))
                pair = _make_pair(rng, roles, kind)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input construction exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _bent(d):
    return any(any(a[0] != b[0] for a, b in zip(p, p[1:]))
               and any(a[1] != b[1] for a, b in zip(p, p[1:])) for p in d['arms'].values())


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            expected, d, board, g, nodes, edges, painted = _solve_details(inp)
            assert np.array_equal(out, expected), "incorrect assembled route output"
            route = _coarse_route(d['active'], *d['terminals'], d['wh'], d['wv'])
            assert route is not None and route[1] == len(nodes)-1, "coarse metric/pixel graph disagreement"
            h, w = d['tile'].shape
            hubs = [(g['nodes'][n][0]//h, g['nodes'][n][1]//w) for n in nodes
                    if board[g['nodes'][n]] in (d['anchor'], d['endpoint'])]
            assert hubs == route[0], "selected hubs must follow the tile-induced route"
            changed = set(map(tuple, np.argwhere(out != board).tolist()))
            assert changed == painted and any(out[p] == 8 for p in painted) and any(out[p] == 9 for p in painted)
            assert len(route[0]) < int(d['active'].sum()), "show unused copies"
            assert all(out[g['nodes'][n]] == d['endpoint'] for n in (nodes[0], nodes[-1]))
            for cells in g['stubs']:
                assert all(out[p] == d['wire'] for p in cells), "open ports are preserved"
            # Seams join only occupied neighbors; there is exactly one port pair.
            for r, c in np.argwhere(d['active']).tolist():
                for side, (dr, dc) in enumerate(DIRS):
                    nr, nc = r+dr, c+dc
                    p = (r*h+d['ports'][side][0], c*w+d['ports'][side][1])
                    q = p[0]+dr, p[1]+dc
                    inside = 0 <= q[0] < board.shape[0] and 0 <= q[1] < board.shape[1]
                    neighbor = 0 <= nr < d['active'].shape[0] and 0 <= nc < d['active'].shape[1] and d['active'][nr, nc]
                    assert (inside and board[q] == d['wire']) == bool(neighbor)
            unused_at_selected = any(any(e not in edges for _, e in g['adj'][n]) for n in nodes[1:-1])
            analyses[split].append((d, unused_at_selected))
    train = [d for d, _ in analyses['train']]
    assert any(_is_witness(d) for d in train), "training must separate station and coarse distances"
    assert len({d['tile'].shape for d in train}) >= 2
    assert any(d['tile'].shape[0] != d['tile'].shape[1] for d in train)
    assert any(2*d['root'][0] != d['tile'].shape[0]-1 or 2*d['root'][1] != d['tile'].shape[1]-1 for d in train)
    assert any(unused for _, unused in analyses['train'])
    assert any(_loop_rank(d['active']) == 0 for d in train), "include a tree demonstration"
    assert all(_loop_rank(d['active']) >= 2 for d, _ in analyses['test'])
    assert any(_is_witness(d) for d, _ in analyses['test']), "test the tile-induced route metric"
    if any(_bent(d) for d, _ in analyses['test']):
        assert any(_bent(d) for d in train), "demonstrate bends before testing them"
    return None


def self_test():
    """Bounded 42-seed semantic, diversity, and counterfactual checks."""
    shapes, maps, motifs, topologies, relay_counts, lengths, palettes = set(), set(), set(), set(), set(), set(), set()
    bent = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed < 3:
            repeat = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], repeat[s]) for k in p)
        for split in puzzle:
            for pair in puzzle[split]:
                out, d, board, g, nodes, _, _ = _solve_details(pair['input'])
                assert np.array_equal(out, pair['output'])
                assert not np.array_equal(board, out), "omitting path selection must change the answer"
                shapes.add((pair['input'].shape, out.shape))
                maps.add(_key(d['active']))
                semantic_tile = np.zeros(d['tile'].shape, dtype=np.int8)
                for role, color in enumerate((d['anchor'], d['ordinary'], d['wire']), 1):
                    semantic_tile[d['tile'] == color] = role
                motifs.add(_key(semantic_tile))
                topologies.add((d['active'].shape, int(d['active'].sum()), _loop_rank(d['active'])))
                relay_counts.add(tuple(d['relays']))
                lengths.add(len(nodes)-2)
                palettes.add((d['bg'], d['signal'], d['endpoint'], d['anchor'], d['ordinary'], d['wire']))
                bent += _bent(d)
    assert len(shapes) >= 100 and len(maps) >= 80 and len(motifs) >= 120
    assert len(topologies) >= 25 and len(relay_counts) >= 40 and len(lengths) >= 10
    assert len(palettes) >= 80 and bent >= 42
    # Every legal explicit count, including minimum demonstrations and four tests.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1700+10*nt+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (4.0,1), (None,1), (4,None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Input palette equivariance; highlight colors remain fixed and input-only.
    p = generate_puzzle(111, 2, 1)
    pair = p['test'][0]
    lookup = np.array([6,4,1,7,0,2,5,3,8,9], dtype=np.int8)
    assert np.array_equal(solve(lookup[pair['input']]), lookup[pair['output']])
    # Corrupting a preserved pixel must be detected.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # A hand-authored member has two-cell port offsets, outside the sampler's
    # one-cell-offset vocabulary. Solving it uses only the visible pixels.
    tile = np.zeros((9, 9), dtype=np.int8)
    root = (4, 4)
    for points in ([root, (0,4)], [root, (4,7), (2,7), (2,8)],
                   [root, (8,4)], [root, (4,1), (2,1), (2,0)]):
        for cell in _polyline(points):
            tile[cell] = 5
    tile[root] = 3
    for cell in ((2,4), (3,7), (6,4), (4,2)):
        tile[cell] = 4
    mask = np.array([[2,1,1], [0,1,0], [0,1,2]], dtype=np.int8)
    handmade = _place(np.random.default_rng(92), mask, tile, 0)
    assert np.array_equal(solve(handmade), _label_from_coarse_path(_read_input(handmade)))
    # Full coarse grids have multiple equally costly opposite-corner routes.
    d = _read_input(p['train'][0]['input'])
    board_mask = np.full((3,3), d['signal'], dtype=np.int8)
    board_mask[0, 0] = board_mask[2, 2] = d['endpoint']
    tied = _place(np.random.default_rng(17), board_mask, d['tile'], d['bg'])
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError("tied minimum accepted")
    return {'seeds': 42, 'pairs': 210, 'input_output_shapes': len(shapes),
            'coarse_patterns': len(maps), 'normalized_tiles': len(motifs),
            'topology_signatures': len(topologies), 'relay_arrangements': len(relay_counts),
            'selected_station_counts': sorted(lengths), 'palette_roles': len(palettes),
            'bent_tiles': bent, 'independent_verification': False}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
