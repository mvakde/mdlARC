"""Shortest handshake routes through patterned 3x3 stations.

Connections are usable exactly when their facing tile samples agree. Choose the
unique minimum-station usable path, copy its intermediate tiles half-turned in
place, and propagate each handshake color along its used connector. Everything
is read from the input; no puzzle-wide hidden color convention is needed.
"""

from collections import deque
from itertools import combinations, permutations
import heapq
import numpy as np


_N4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
_N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.bool_),
             "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(mask, neighbors=_N8):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        cells, stack = [], [p]
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(sorted(cells))
    return result


def _geometry(a, bg, wire):
    """Recover tile boxes, edge masks, and attachment directions from pixels."""
    blocks = _components((a != bg) & (a != wire))
    _require(len(blocks) >= 5, "at least five stations required")
    centers = []
    labels = np.full(a.shape, -1, dtype=np.int16)
    for i, cells in enumerate(blocks):
        lo, hi = np.min(cells, axis=0), np.max(cells, axis=0)
        _require(len(cells) == 9 and tuple(hi - lo) == (2, 2), "stations must be full 3x3 tiles")
        centers.append((int(lo[0]) + 1, int(lo[1]) + 1))
        for p in cells:
            labels[p] = i
    masks = _components(a == wire)
    edges, ports = [], []
    for cells in masks:
        touching = set()
        cellset = set(cells)
        for r, c in cells:
            for dr, dc in _N8:
                rr, cc = r + dr, c + dc
                if 0 <= rr < a.shape[0] and 0 <= cc < a.shape[1] and labels[rr, cc] >= 0:
                    touching.add(int(labels[rr, cc]))
        _require(len(touching) == 2, "a connector must touch exactly two stations")
        u, v = sorted(touching)
        directions = []
        for n in (u, v):
            r, c = centers[n]
            exits = [d for d in _N4 if (r + 2*d[0], c + 2*d[1]) in cellset]
            _require(len(exits) == 1, "connector must use one cardinal midpoint port")
            # Forbid incidental corner contact with a different side of this tile.
            dr, dc = exits[0]
            nearby = [(rr-r, cc-c) for rr, cc in cells if max(abs(rr-r), abs(cc-c)) <= 2]
            _require(nearby == [(2*dr, 2*dc)], "connector grazes a station away from its port")
            directions.append(exits[0])
        edges.append((u, v))
        ports.append(tuple(directions))
    _require(len(set(edges)) == len(edges), "parallel connectors are excluded")
    adj = [[] for _ in centers]
    for e, (u, v) in enumerate(edges):
        adj[u].append((v, e))
        adj[v].append((u, e))
    _require(all(1 <= len(x) <= 4 for x in adj), "isolated tile or excessive degree")
    reached, stack = {0}, [0]
    while stack:
        for v, _ in adj[stack.pop()]:
            if v not in reached:
                reached.add(v)
                stack.append(v)
    _require(len(reached) == len(centers), "physical network must be connected")
    return dict(centers=centers, blocks=blocks, masks=masks, edges=edges,
                ports=ports, adj=adj, areas=[len(c) for c in masks])


def _shortest(data, start, target, usable=None):
    n = len(data['centers'])
    dist, ways, prev = [-1]*n, [0]*n, [None]*n
    dist[start], ways[start] = 0, 1
    queue = deque([start])
    while queue:
        u = queue.popleft()
        for v, e in data['adj'][u]:
            if usable is not None and not usable[e]:
                continue
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                prev[v] = (u, e)
                queue.append(v)
            if dist[v] == dist[u] + 1:
                ways[v] = min(2, ways[v] + ways[u])
    if ways[target] != 1:
        return None
    nodes, edges = [target], []
    while nodes[-1] != start:
        u, e = prev[nodes[-1]]
        nodes.append(u)
        edges.append(e)
    return nodes[::-1], edges[::-1]


def _weighted_min(data, start, target, usable):
    best = [float('inf')] * len(data['centers'])
    best[start] = 0
    queue = [(0, start)]
    while queue:
        cost, u = heapq.heappop(queue)
        if cost != best[u]:
            continue
        if u == target:
            return cost
        for v, e in data['adj'][u]:
            if not usable[e]:
                continue
            new = cost + data['areas'][e]
            if new < best[v]:
                best[v] = new
                heapq.heappush(queue, (new, v))
    return float('inf')


def _area_path(data):
    """Counterfactual: minimize connector cells instead of station visits."""
    start, target = data['endpoints']
    queue = [(0, start, (start,), ())]
    visited = set()
    while queue:
        cost, u, nodes, edges = heapq.heappop(queue)
        if u in visited:
            continue
        visited.add(u)
        if u == target:
            return list(nodes), list(edges)
        for v, e in data['adj'][u]:
            if data['usable'][e] and v not in visited:
                heapq.heappush(queue, (cost+data['areas'][e], v, nodes+(v,), edges+(e,)))
    raise ValueError("disconnected usable endpoints")


def _read_input(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(np.count_nonzero(counts == counts.max()) == 1, "background must be uniquely dominant")
    candidates = []
    for wire in colors:
        if wire == bg:
            continue
        # All filled tiles survive removal of the connector color intact.
        blocks = _components((a != bg) & (a != wire))
        if len(blocks) < 5 or any(len(b) != 9 or tuple(np.max(b, 0)-np.min(b, 0)) != (2, 2)
                                 for b in blocks):
            continue
        try:
            data = _geometry(a, bg, int(wire))
        except ValueError:
            continue
        candidates.append((int(wire), data))
    _require(len(candidates) == 1, "cannot uniquely identify the wire and tiles")
    wire, data = candidates[0]
    center_colors = [int(a[p]) for p in data['centers']]
    palette = set(center_colors)
    _require(len(palette) == 2, "station centers must have two role colors")
    marker = [c for c in palette if center_colors.count(c) == 2]
    _require(len(marker) == 1, "exactly two marked endpoint centers required")
    marker = marker[0]
    carrier = next(c for c in palette if c != marker)
    _require(bool(np.count_nonzero(a == marker) == 2), "endpoint marker occurs only at centers")
    samples, usable = [], []
    for (u, v), (du, dv) in zip(data['edges'], data['ports']):
        values = []
        for n, (dr, dc) in ((u, du), (v, dv)):
            r, c = data['centers'][n]
            sample = int(a[r+dr, c+dc])
            _require(sample not in (bg, wire, marker, carrier), "attached port must be a signal color")
            values.append(sample)
        samples.append(tuple(values))
        usable.append(values[0] == values[1])
    endpoints = [i for i, c in enumerate(center_colors) if c == marker]
    path = _shortest(data, *endpoints, usable)
    _require(path is not None, "usable endpoint route must exist and have a unique minimum")
    data.update(bg=bg, wire=wire, carrier=carrier, marker=marker,
                samples=samples, usable=usable, endpoints=endpoints, path=path)
    return data


def _paint(a, data, path=None, turn=2):
    nodes, edges = data['path'] if path is None else path
    out = a.copy()
    for e in edges:
        value = data['samples'][e][0]
        for p in data['masks'][e]:
            out[p] = value
    for n in nodes[1:-1]:
        r, c = data['centers'][n]
        tile = a[r-1:r+2, c-1:c+2]
        out[r-1:r+2, c-1:c+2] = np.rot90(tile, turn)
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve solely from visible input cells, evaluating gates before turning tiles."""
    a = _grid(input_grid)
    return _paint(a, _read_input(a))


def _witnesses(data):
    nodes, edges = data['path']
    a = (len(nodes) >= 3 and sum(data['areas'][e] for e in edges) >
         _weighted_min(data, *data['endpoints'], data['usable']))
    raw = _shortest(data, *data['endpoints'])
    b = raw is not None and len(raw[0]) < len(nodes)
    return bool(a), bool(b)


def _sample_layout(rng, larger=False):
    nr = int(rng.integers(3, 5)) if larger else int(rng.integers(2, 5))
    nc = int(rng.integers(3, 5)) if larger else int(rng.integers(2, 5))
    if nr * nc < 6:
        nc = 3
    vertices = [(r, c) for r in range(nr) for c in range(nc)]
    possible = [(u, v) for u, v in combinations(vertices, 2)
                if abs(u[0]-v[0]) + abs(u[1]-v[1]) == 1]
    # Random connected planar subgraphs, rather than a preselected loop outline.
    active = {vertices[int(rng.integers(len(vertices)))]}
    edges = []
    while len(active) < len(vertices):
        choices = [(u, v) for u, v in possible if (u in active) != (v in active)]
        u, v = choices[int(rng.integers(len(choices)))]
        active.update((u, v))
        edges.append((u, v))
    extras = [e for e in possible if e not in edges]
    rng.shuffle(extras)
    add = int(rng.integers(1, len(extras)+1)) if extras else 0
    edges.extend(extras[:add])
    degrees = {v: sum(v in e for e in edges) for v in vertices}
    visible = set(vertices)
    suppress = float(rng.uniform(.05, .48)) if not larger else float(rng.uniform(0, .2))
    for v in vertices:
        if len(visible) > (8 if larger else 5) and degrees[v] == 2 and rng.random() < suppress:
            visible.remove(v)
    def axis(n):
        gap = rng.integers(6, 9, size=n-1)
        return [1] + [1+int(x) for x in np.cumsum(gap)]
    rows, cols = axis(nr), axis(nc)
    centers = {v: (rows[v[0]], cols[v[1]]) for v in vertices}
    shape = rows[-1]+2, cols[-1]+2
    grid = np.zeros(shape, dtype=np.int8)
    for u, v in edges:
        p, q = centers[u], centers[v]
        if p[0] == q[0]:
            r, lo, hi = p[0], min(p[1], q[1]), max(p[1], q[1])
            for c in range(lo, hi+1):
                grid[r, c] = 1
            if hi-lo >= 7 and rng.random() < .45:
                shift = int(rng.choice([-1, 1]))
                # The ends have straight stubs; bends never graze the source tile.
                grid[r, lo+3:hi-2] = 0
                grid[r+shift, lo+3:hi-2] = 1
        else:
            c, lo, hi = p[1], min(p[0], q[0]), max(p[0], q[0])
            for r in range(lo, hi+1):
                grid[r, c] = 1
            if hi-lo >= 7 and rng.random() < .45:
                shift = int(rng.choice([-1, 1]))
                grid[lo+3:hi-2, c] = 0
                grid[lo+3:hi-2, c+shift] = 1
    for v in visible:
        r, c = centers[v]
        grid[r-1:r+2, c-1:c+2] = 2
    data = _geometry(grid, 0, 1)
    # Verify the contraction against the rendered connectors, rejecting parallel
    # routes without stations and incidental eight-neighbor contact.
    visible_order = sorted(visible, key=lambda v: centers[v])
    ids = {v: i for i, v in enumerate(visible_order)}
    adj = {v: [] for v in vertices}
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    expected = []
    for u in visible_order:
        for nxt in adj[u]:
            previous, current = u, nxt
            visited = {u}
            while current not in visible:
                _require(current not in visited, "hidden cycle")
                visited.add(current)
                onward = [x for x in adj[current] if x != previous]
                _require(len(onward) == 1, "only degree-two bends may be suppressed")
                previous, current = current, onward[0]
            _require(current != u, "self-loop")
            if ids[u] < ids[current]:
                expected.append((ids[u], ids[current]))
    _require(len(expected) == len(set(expected)) and set(expected) == set(data['edges']),
             "rendered graph differs from construction")
    return grid, data


def _fallback_layout(larger=False):
    """Three competing corridors: long/sparse, short/dense, and a blocked bypass."""
    a = np.zeros((23, 27), dtype=np.int8)
    a[1, 1:26] = 1
    a[21, 1:26] = 1
    a[11, 1:26] = 1
    a[1:22, 1] = 1
    a[1:22, 25] = 1
    for r, c in ((11, 1), (11, 25), (1, 13), (11, 7), (11, 13), (11, 19)):
        a[r-1:r+2, c-1:c+2] = 2
    if larger:
        for c in (7, 13, 19):
            a[13:17, c] = 1
            a[16:19, c-1:c+2] = 2
    return a, _geometry(a, 0, 1)


def _choose_path(rng, data, witness, fallback=False):
    edge_count = len(data['edges'])
    if fallback:
        # Suppressed bottom corners create the only direct endpoint connection.
        usable = [not (data['centers'][u] == (11, 1) and data['centers'][v] == (11, 25))
                  for u, v in data['edges']]
        endpoints = [data['centers'].index((11, 1)), data['centers'].index((11, 25))]
        data.update(usable=usable, endpoints=endpoints,
                    path=_shortest(data, *endpoints, usable))
        return data
    usable = list(rng.random(edge_count) < float(rng.uniform(.58, .95)))
    if all(usable) and rng.random() < .6:
        usable[int(rng.integers(edge_count))] = False
    choices = []
    for u, v in combinations(range(len(data['centers'])), 2):
        path = _shortest(data, u, v, usable)
        if path is None or (witness is not None and len(path[0]) < 3):
            continue
        probe = dict(data, usable=usable, endpoints=(u, v), path=path)
        wa, wb = _witnesses(probe)
        if witness in ('A', 'AB') and not wa or witness in ('B', 'AB') and not wb:
            continue
        choices.append((u, v, path))
    _require(bool(choices), "no endpoint pair has the requested witness")
    # Answer length is allowed to vary independently from tile count.
    target = int(rng.integers(0, 7))
    best = min(abs(len(p[2][0])-2-target) for p in choices)
    choices = [p for p in choices if abs(len(p[2][0])-2-target) == best]
    u, v, path = choices[int(rng.integers(len(choices)))]
    data.update(usable=usable, endpoints=(u, v), path=path)
    return data


def _d4(tile):
    return tuple(np.rot90(tile, k) for k in range(4)) + tuple(
        np.rot90(tile[:, ::-1], k) for k in range(4))


def _sample_candidates(train):
    """Infer possible tile sample coordinates per port from changed wires only."""
    candidates = {d: set(range(9)) for d in _N4}
    for pair in train:
        a, b = np.asarray(pair['input']), np.asarray(pair['output'])
        data = _read_input(a)
        for e in data['path'][1]:
            value = int(b[data['masks'][e][0]])
            for n, direction in zip(data['edges'][e], data['ports'][e]):
                r, c = data['centers'][n]
                tile = a[r-1:r+2, c-1:c+2]
                candidates[direction].intersection_update(i for i in range(9) if tile.flat[i] == value)
    return candidates


def _samples_taught(candidates):
    return all(candidates[d] == {(1+d[0])*3+1+d[1]} for d in _N4)


def _unique_ports(rng, pair):
    """Coverage repair: keep graph and gate states, make samples position-unique.

    Properly color the physical edges first, then change one half of each closed
    edge. Each tile gets distinct side colors and carrier-colored corners. This
    is a simpler motif witness, not a new solution convention.
    """
    a = pair['input'].copy()
    d = _read_input(a)
    pool = [c for c in range(10) if c not in (d['bg'], d['wire'], d['carrier'], d['marker'])]
    common = None
    for _ in range(32):
        assigned = [None]*len(d['edges'])
        at = [set() for _ in d['centers']]
        for e in rng.permutation(len(d['edges'])):
            e = int(e)
            u, v = d['edges'][e]
            choices = [c for c in pool if c not in at[u] | at[v]]
            if not choices:
                break
            value = int(rng.choice(choices))
            assigned[e] = value
            at[u].add(value)
            at[v].add(value)
        else:
            common = assigned
            break
    if common is None:
        return None
    values = [[c, c] for c in common]
    occupied = [set() for _ in d['centers']]
    for e, (u, v) in enumerate(d['edges']):
        occupied[u].add(common[e])
        occupied[v].add(common[e])
    for e, (u, v) in enumerate(d['edges']):
        if not d['usable'][e]:
            occupied[v].remove(common[e])
            choices = [c for c in pool if c not in occupied[v] and c != common[e]]
            values[e][1] = int(rng.choice(choices))
            occupied[v].add(values[e][1])
    for n, (r, c) in enumerate(d['centers']):
        center = int(a[r, c])
        a[r-1:r+2, c-1:c+2] = d['carrier']
        a[r, c] = center
        available = [x for x in pool if x not in occupied[n]]
        rng.shuffle(available)
        used_dirs = {direction for e in range(len(d['edges']))
                     for v, direction in zip(d['edges'][e], d['ports'][e]) if v == n}
        for dr, dc in _N4:
            if (dr, dc) not in used_dirs:
                a[r+dr, c+dc] = available.pop()
    for e in range(len(d['edges'])):
        for n, (dr, dc), color in zip(d['edges'][e], d['ports'][e], values[e]):
            r, c = d['centers'][n]
            a[r+dr, c+dc] = color
    result = dict(input=a, output=solve(a))
    rebuilt = _read_input(a)
    _require(rebuilt['usable'] == d['usable'] and rebuilt['path'] == d['path'], "port repair changed the route")
    return result


def _repair_sampling(rng, train):
    if _samples_taught(_sample_candidates(train)):
        return
    # Repair only when the existing demonstrations leave a position alternative.
    for i, pair in enumerate(train):
        replacement = _unique_ports(rng, pair)
        if replacement is None:
            d = _read_input(pair['input'])
            roles = d['bg'], d['wire'], d['carrier'], d['marker']
            base, info = _fallback_layout()
            info = _choose_path(rng, info, 'AB', fallback=True)
            inp = _place(rng, _decorate(rng, base, info, roles))
            replacement = _unique_ports(rng, dict(input=inp, output=solve(inp)))
            _require(replacement is not None, "degree-three fallback edge coloring failed")
        train[i] = replacement
    candidates = _sample_candidates(train)
    if not _samples_taught(candidates):
        # At least two sides occur in every positive route. A three-side fallback
        # can be oriented to include all missing sides without adding examples.
        original = train[0]
        d = _read_input(original['input'])
        roles = d['bg'], d['wire'], d['carrier'], d['marker']
        base, info = _fallback_layout()
        info = _choose_path(rng, info, 'AB', fallback=True)
        inp = _decorate(rng, base, info, roles)
        replacement = _unique_ports(rng, dict(input=inp, output=solve(inp)))
        _require(replacement is not None, "fallback port witness failed")
        for k in range(4):
            a = np.rot90(replacement['input'], k).copy()
            train[0] = dict(input=a, output=solve(a))
            if _samples_taught(_sample_candidates(train)):
                break
        else:
            raise RuntimeError("could not expose all four port directions")
    _require(_samples_taught(_sample_candidates(train)), "sampling convention remains ambiguous")


def _decorate(rng, base, data, roles, rich=False, force_turn_witness=False):
    bg, wire, carrier, marker = roles
    pool = [c for c in range(10) if c not in roles]
    signal_count = int(rng.integers(2, 7))
    signals = [int(x) for x in rng.choice(pool, signal_count, replace=False)]
    a = np.full(base.shape, bg, dtype=np.int8)
    a[base == 1] = wire
    for n, (r, c) in enumerate(data['centers']):
        tile = np.full((3, 3), carrier, dtype=np.int8)
        # Unused side samples and corner motifs may repeat signal colors.
        for dr, dc in _N4:
            tile[1+dr, 1+dc] = int(rng.choice(signals))
        motif = int(rng.integers(4))
        corners = [(0, 0), (0, 2), (2, 0), (2, 2)]
        if motif == 1:
            positions = [corners[int(rng.integers(4))]]
        elif motif == 2:
            positions = [corners[int(rng.integers(4))], corners[int(rng.integers(4))]]
        elif motif == 3:
            positions = corners if rich else [(0, 0), (2, 2)]
        else:
            positions = []
        for p in positions:
            tile[p] = int(rng.choice(pool if rich else signals))
        if n in data['endpoints']:
            tile[1, 1] = marker
        a[r-1:r+2, c-1:c+2] = tile
    for e, ((u, v), (du, dv)) in enumerate(zip(data['edges'], data['ports'])):
        values = [int(rng.choice(signals))]
        values.append(values[0] if data['usable'][e] else int(rng.choice([x for x in signals if x != values[0]])))
        for n, (dr, dc), value in zip((u, v), (du, dv), values):
            r, c = data['centers'][n]
            a[r+dr, c+dc] = value
    # Two adjacent, differently colored corners give eight distinct D4 images.
    # Only the scheduled demonstrations need this deliberate turn witness;
    # ordinary examples and queries retain their natural motif symmetries.
    if force_turn_witness and len(data['path'][0]) > 2:
        n = int(rng.choice(data['path'][0][1:-1]))
        r, c = data['centers'][n]
        corners = [(-1, -1), (-1, 1), (1, 1), (1, -1)]
        for dr, dc in corners:
            a[r+dr, c+dc] = carrier
        start = int(rng.integers(4))
        for (dr, dc), value in zip((corners[start], corners[(start+1)%4]),
                                   rng.choice(signals, 2, replace=False)):
            a[r+dr, c+dc] = value
    return a


def _place(rng, a, large=False):
    a = np.rot90(a, int(rng.integers(4)))
    if rng.random() < .5:
        a = a[:, ::-1]
    h, w = a.shape
    low_h, low_w = (max(27, h), max(27, w)) if large else (h, w)
    hh, ww = int(rng.integers(low_h, 31)), int(rng.integers(low_w, 31))
    top, left = int(rng.integers(hh-h+1)), int(rng.integers(ww-w+1))
    bg = int(np.argmax(np.bincount(a.ravel(), minlength=10)))
    out = np.full((hh, ww), bg, dtype=np.int8)
    out[top:top+h, left:left+w] = a
    return out


def _make_pair(rng, roles, witness=None, large=False):
    for attempt in range(121):
        fallback = attempt == 120
        try:
            base, data = _fallback_layout(larger=large) if fallback else _sample_layout(rng, larger=large)
            data = _choose_path(rng, data, witness, fallback)
            a = _decorate(rng, base, data, roles, rich=bool(rng.random() < .55),
                          force_turn_witness=witness in ('A', 'B'))
            a = _place(rng, a, large=large)
            parsed = _read_input(a)
            wa, wb = _witnesses(parsed)
            _require(witness not in ('A', 'AB') or wa, "lost count-versus-length witness")
            _require(witness not in ('B', 'AB') or wb, "lost closed-bypass witness")
            return dict(input=a, output=_paint(a, parsed))
        except ValueError:
            if fallback:
                raise
    raise RuntimeError("bounded sampler exhausted")


def _count(value, name, lo, hi):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             name + " must be an integer")
    value = int(value)
    _require(lo <= value <= hi, name + " outside its legal range")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)),
             "seed must be an integer")
    rng = np.random.default_rng(seed)
    common_roles = tuple(int(x) for x in rng.choice(10, 4, replace=False))
    fixed_roles = bool(rng.random() < .3)
    # Every minimum-size training set distinguishes both relevant shortcuts.
    requests = ['A', 'B'] + [None] * (num_train-2)
    rng.shuffle(requests)
    puzzle, seen = dict(train=[], test=[]), set()
    for split, items in (('train', requests), ('test', ['AB']*num_test)):
        for witness in items:
            for _ in range(20):
                roles = common_roles if fixed_roles else tuple(int(x) for x in rng.choice(10, 4, replace=False))
                pair = _make_pair(rng, roles, witness, large=split == 'test')
                k = _key(pair['input'])
                if k not in seen:
                    seen.add(k)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct-input retries exhausted")
    _repair_sampling(rng, puzzle['train'])
    # Repair may simplify two formerly distinct motifs into the same input.
    # Signal-only permutations preserve every gate, role, and geometry witness.
    seen = set()
    for split in ('train', 'test'):
        for i, pair in enumerate(puzzle[split]):
            if _key(pair['input']) in seen:
                d = _read_input(pair['input'])
                pool = [c for c in range(10) if c not in (d['bg'], d['wire'], d['carrier'], d['marker'])]
                for perm in permutations(pool):
                    mapping = np.arange(10, dtype=np.int8)
                    mapping[pool] = perm
                    inp = mapping[pair['input']]
                    if _key(inp) not in seen:
                        pair = dict(input=inp, output=mapping[pair['output']])
                        puzzle[split][i] = pair
                        break
                else:
                    raise RuntimeError("signal-permutation distinctness repair exhausted")
            seen.add(_key(pair['input']))
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
    a, b = _grid(pair['input']), _grid(pair['output'])
    _require(a.shape == b.shape, "dimensions changed")
    data = _read_input(a)
    nodes, edges = data['path']
    # Scalar coordinate validation separately checks exactly the assigned cells.
    assigned = set()
    for n in nodes[1:-1]:
        r, c = data['centers'][n]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                p = r+dr, c+dc
                _require(b[p] == a[r-dr, c-dc], "tile was not half-turned")
                assigned.add(p)
    for e in edges:
        _require(data['usable'][e], "selected edge has mismatching samples")
        for p in data['masks'][e]:
            _require(b[p] == data['samples'][e][0], "wrong propagated handshake")
            assigned.add(p)
    mask = np.zeros(a.shape, dtype=bool)
    for p in assigned:
        mask[p] = True
    _require(np.array_equal(a[~mask], b[~mask]), "off-route content changed")
    _require(not np.array_equal(a, b), "identity pair")
    _require(set(map(int, np.unique(b))) <= set(map(int, np.unique(a))), "new color introduced")
    return data


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "invalid counts")
    seen, witnesses, operations = set(), [False, False], set(range(8))
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            data = _check_pair(pair)
            a, b = np.asarray(pair['input']), np.asarray(pair['output'])
            k = _key(a)
            _require(k not in seen, "duplicate inputs")
            seen.add(k)
            if split == 'train':
                witnesses = [old or new for old, new in zip(witnesses, _witnesses(data))]
                for n in data['path'][0][1:-1]:
                    r, c = data['centers'][n]
                    tile = a[r-1:r+2, c-1:c+2]
                    target = b[r-1:r+2, c-1:c+2]
                    operations.intersection_update(i for i, t in enumerate(_d4(tile))
                                                   if np.array_equal(t, target))
            else:
                _require(all(_witnesses(data)), "tests must recombine both route-choice conflicts")
                _require(min(a.shape) >= 27, "tests must use the larger-canvas regime")
    _require(all(witnesses), "training must distinguish length and ungated-route shortcuts")
    _require(operations == {2}, "training must uniquely teach the half-turn among D4 operations")
    _require(_samples_taught(_sample_candidates(puzzle['train'])),
             "training must teach all four facing sample coordinates")


def _signature(puzzle):
    return tuple(_key(np.asarray(p[k])) for s in ('train', 'test')
                 for p in puzzle[s] for k in ('input', 'output'))


def self_test():
    """Bounded 42-seed checks, including topology, ablation, and corruption tests."""
    shapes, structures, station_counts, lengths, roles, degrees, signatures = set(), set(), set(), set(), set(), set(), set()
    closed, bent, branch, unused = 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        if seed < 3:
            _require(_signature(puzzle) == _signature(generate_puzzle(seed)), "nondeterministic seed")
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                d = _read_input(a)
                _require(np.array_equal(solve(a.copy()), b), "input-only solve failed")
                shapes.add(a.shape)
                station_counts.add(len(d['centers']))
                lengths.add(len(d['path'][0])-2)
                roles.add((d['bg'], d['wire'], d['carrier'], d['marker']))
                degrees.update(len(x) for x in d['adj'])
                closed += not all(d['usable'])
                branch += any(len(x) >= 3 for x in d['adj'])
                unused += any(any(e not in d['path'][1] for _, e in d['adj'][n]) for n in d['path'][0][1:-1])
                for cells in d['masks']:
                    rows, cols = zip(*cells)
                    bent += len(set(rows)) > 1 and len(set(cols)) > 1
                # Remove colors and cropping margins to measure actual geometry.
                semantic = np.zeros(a.shape, dtype=np.int8)
                semantic[a == d['wire']] = 1
                for n, cells in enumerate(d['blocks']):
                    for p in cells:
                        semantic[p] = 3 if n in d['endpoints'] else 2
                fg = np.argwhere(semantic)
                lo, hi = fg.min(0), fg.max(0)
                structures.add(_key(semantic[lo[0]:hi[0]+1, lo[1]:hi[1]+1]))
                if split == 'train':
                    wa, wb = _witnesses(d)
                    if wa:
                        _require(sum(d['areas'][e] for e in d['path'][1]) >
                                 _weighted_min(d, *d['endpoints'], d['usable']), "A ablation ineffective")
                        _require(not np.array_equal(_paint(a, d, _area_path(d)), b),
                                 "removing station minimization did not change the answer")
                    if wb:
                        raw = _shortest(d, *d['endpoints'])
                        _require(not np.array_equal(_paint(a, d, raw), b), "B ablation ineffective")
    _require(len(signatures) == 42 and len(structures) >= 100, "insufficient structural diversity")
    _require(len(shapes) >= 30 and len(roles) >= 30 and len(station_counts) >= 6 and len(lengths) >= 4,
             "variation axes collapsed")
    _require({1, 2, 3, 4} <= degrees and min(closed, bent, branch, unused) >= 20,
             "missing gate/bend/junction coverage")
    _require(0 in lengths, "direct-route case missing")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+ne, nt, ne)
            _require(len(p['train']) == nt and len(p['test']) == ne, "requested counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    good = generate_puzzle(2026, 2, 1)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in good[s]] for s in good}
    d = _read_input(bad['test'][0]['input'])
    n = d['endpoints'][0]
    bad['test'][0]['output'][d['centers'][n]] = d['carrier']
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("endpoint corruption accepted")
    # A square with a leaf has two equal two-edge routes between diagonal
    # endpoints. Reject it instead of selecting by BFS ordering or wire length.
    tied = np.zeros((13, 19), dtype=np.int8)
    tied[1, 1:18] = 1
    tied[11, 1:12] = 1
    tied[1:12, 1] = 1
    tied[1:12, 11] = 1
    for r, c in ((1, 1), (1, 11), (11, 1), (11, 11), (1, 17)):
        tied[r-1:r+2, c-1:c+2] = 2
        for dr, dc in _N4:
            tied[r+dr, c+dc] = 4
    tied[1, 1] = tied[11, 11] = 3
    try:
        solve(tied)
    except ValueError:
        pass
    else:
        raise AssertionError("tied minimum accepted")


if __name__ == '__main__':
    self_test()
