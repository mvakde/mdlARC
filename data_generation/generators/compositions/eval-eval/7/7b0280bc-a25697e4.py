"""Conducting locks: rigid key fitting changes which graph vertices can be used.

Only NumPy and the standard library are used. No parent module, examples, seed,
or generation metadata is consulted by solve(). See task_spec.md for the rule.
"""
from collections import deque
import heapq
import itertools
import numpy as np


DIR4 = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        seen, queue = {start}, [start]
        while queue:
            r, c = queue.pop()
            for dr, dc in DIR4:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    seen.add(p)
                    queue.append(p)
        result.append(frozenset(seen))
    return result


def _cells(mask):
    return frozenset(map(tuple, np.argwhere(mask).tolist()))


def _bbox(cells):
    rows, cols = zip(*cells)
    return min(rows), max(rows), min(cols), max(cols)


def _rectangle(box):
    r0, r1, c0, c1 = box
    return frozenset((r, c) for r in range(r0, r1 + 1)
                     for c in range(c0, c1 + 1))


def _shift(cells, delta):
    return frozenset((r + delta[0], c + delta[1]) for r, c in cells)


def _transform(cells, op):
    # Reflection in a vertical axis, followed by 0..3 quarter turns.
    out = set()
    for r, c in cells:
        if op >= 4:
            c = -c
        for _ in range(op % 4):
            r, c = c, -r
        out.add((r, c))
    return frozenset(out)


def _signature(cells):
    r0, _, c0, _ = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _key(array):
    a = np.asarray(array, dtype=np.int8)
    return a.shape, a.tobytes()


def _array(grid):
    a = np.asarray(grid)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('expected a 1..30 integer grid with colors 0..9')
    return a


def _adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for i, (a, b) in enumerate(edges):
        adj[a].append((b, i))
        adj[b].append((a, i))
    return adj


def _shortest(adj, endpoints, allowed):
    start, target = endpoints
    dist, ways, prev = {start: 0}, {start: 1}, {}
    queue = deque([start])
    while queue:
        a = queue.popleft()
        for b, e in adj[a]:
            if b not in allowed:
                continue
            if b not in dist:
                dist[b] = dist[a] + 1
                ways[b] = 0
                prev[b] = a, e
                queue.append(b)
            if dist[b] == dist[a] + 1:
                ways[b] = min(2, ways[b] + ways[a])
    if ways.get(target) != 1:
        return None
    nodes, edges = [target], []
    while nodes[-1] != start:
        a, e = prev[nodes[-1]]
        nodes.append(a)
        edges.append(e)
    return nodes[::-1], edges[::-1]


def _weighted_min(adj, endpoints, allowed, weights):
    start, target = endpoints
    best, queue = {start: 0}, [(0, start)]
    while queue:
        cost, a = heapq.heappop(queue)
        if cost != best[a]:
            continue
        if a == target:
            return cost
        for b, e in adj[a]:
            if b in allowed and cost + weights[e] < best.get(b, float('inf')):
                best[b] = cost + weights[e]
                heapq.heappush(queue, (best[b], b))
    return float('inf')


def _fits(inp, regions, colors, station, bg):
    slot, rectangle = station['slot'], station['rectangle']
    sr0, _, sc0, _ = _bbox(slot)
    candidates = {}
    for plug_index in (0, 1):
        for op in range(8):
            transformed = tuple(_transform(x, op) for x in regions)
            pr0, _, pc0, _ = _bbox(transformed[plug_index])
            absolute = tuple(_shift(x, (sr0 - pr0, sc0 - pc0)) for x in transformed)
            if absolute[plug_index] != slot:
                continue
            handle = absolute[1 - plug_index]
            if handle & rectangle:
                continue
            if any(not (0 <= r < inp.shape[0] and 0 <= c < inp.shape[1])
                   or inp[r, c] != bg for r, c in handle):
                continue
            new_colors = colors[::-1] if op >= 4 else colors
            pc, hc = new_colors[plug_index], new_colors[1 - plug_index]
            sig = (tuple(sorted(slot)), tuple(sorted(handle)), pc, hc)
            candidates[sig] = dict(plug=slot, handle=handle, plug_color=pc,
                                   handle_color=hc, parity=int(op >= 4), op=op,
                                   plug_index=plug_index)
    if len(candidates) != 1:
        raise ValueError('every lock needs one complete colored key placement')
    return next(iter(candidates.values()))


def _analyze(input_grid):
    """Reconstruct roles, locks, legal fits and graph solely from input pixels."""
    inp = _array(input_grid)
    colors, counts = np.unique(inp, return_counts=True)
    if len(colors) != 5 or np.sum(counts == counts.max()) != 1:
        raise ValueError('five roles and a uniquely dominant background required')
    bg = int(colors[np.argmax(counts)])
    fg = _cells(inp != bg)
    keys = []
    for component in _components(fg):
        cc = tuple(sorted({int(inp[p]) for p in component}))
        if len(cc) == 2:
            regions = tuple(frozenset(p for p in component if inp[p] == c) for c in cc)
            if all(len(_components(x)) == 1 for x in regions):
                keys.append((component, cc, regions))
    if len(keys) != 1:
        raise ValueError('exactly one isolated two-region master key required')
    key_cells, key_colors, key_regions = keys[0]
    net = fg - key_cells
    net_colors = {int(inp[p]) for p in net}
    shared = net_colors & set(key_colors)
    if len(net_colors) != 3 or len(shared) != 1:
        raise ValueError('one key color must also be the network wire color')
    wire = shared.pop()
    other = next(c for c in key_colors if c != wire)
    endpoint_candidates = []
    for color in net_colors - {wire}:
        components = _components(_cells(inp == color))
        if len(components) != 2:
            continue
        boxes = [_bbox(x) for x in components]
        sizes = [b[1] - b[0] + 1 for b in boxes]
        if (sizes[0] == sizes[1] and sizes[0] >= 3
                and all(x == _rectangle(b) and b[3] - b[2] + 1 == sizes[0]
                        for x, b in zip(components, boxes))):
            endpoint_candidates.append((color, sizes[0], components))
    if len(endpoint_candidates) != 1:
        raise ValueError('exactly two solid square endpoint stations required')
    endpoint_color, size, endpoint_cells = endpoint_candidates[0]
    lock_color = next(c for c in net_colors if c not in (wire, endpoint_color))
    fragments = _components(_cells(inp == lock_color))
    candidates = []
    for i, j in itertools.combinations(range(len(fragments)), 2):
        target = fragments[i] | fragments[j]
        box = _bbox(target)
        if box[1] - box[0] + 1 != size or box[3] - box[2] + 1 != size:
            continue
        rectangle = _rectangle(box)
        if any(inp[p] not in (bg, lock_color) for p in rectangle):
            continue
        if frozenset(p for p in rectangle if inp[p] == lock_color) != target:
            continue
        slot = rectangle - target
        if slot and len(_components(slot)) == 1:
            candidates.append((i, j, dict(rectangle=rectangle, target=target, slot=slot)))
    used = [i for i, j, _ in candidates] + [j for i, j, _ in candidates]
    if sorted(used) != list(range(len(fragments))) or len(candidates) < 3:
        raise ValueError('lock fragments must have one unambiguous square pairing')
    stations = [dict(rectangle=x, target=x, slot=None) for x in endpoint_cells]
    stations += [s for _, _, s in candidates]
    station_map = {}
    for i, s in enumerate(stations):
        if any(p in station_map for p in s['rectangle']):
            raise ValueError('overlapping station rectangles')
        station_map.update({p: i for p in s['rectangle']})
        if s['slot'] is not None:
            s['fit'] = _fits(inp, key_regions, key_colors, s, bg)
    connectors = _components(_cells(inp == wire) - key_cells)
    edges = []
    for component in connectors:
        touching = set()
        for r, c in component:
            for dr, dc in DIR4:
                p = r + dr, c + dc
                if p in station_map:
                    i = station_map[p]
                    if p not in stations[i]['target']:
                        raise ValueError('wires must attach to fixed station fragments')
                    touching.add(i)
        if len(touching) != 2:
            raise ValueError('a wire component must join exactly two stations')
        edges.append(tuple(sorted(touching)))
    if len(set(edges)) != len(edges):
        raise ValueError('parallel connectors not supported')
    adj = _adjacency(len(stations), edges)
    if not all(1 <= len(x) <= 4 for x in adj):
        raise ValueError('station degrees must be 1..4')
    reached, todo = {0}, [0]
    while todo:
        for b, _ in adj[todo.pop()]:
            if b not in reached:
                reached.add(b)
                todo.append(b)
    if len(reached) != len(stations):
        raise ValueError('the underlying station graph must be connected')
    allowed = {0, 1} | {i for i, s in enumerate(stations) if s['slot'] is not None
                       and s['fit']['plug_color'] == wire}
    path = _shortest(adj, (0, 1), allowed)
    if path is None:
        raise ValueError('conducting routes are absent or minimum station count ties')
    # All potential handles are disjoint. This is stronger than only checking
    # selected copies, and makes each local gate's state independent of the path.
    handles = set()
    for s in stations[2:]:
        if handles & s['fit']['handle']:
            raise ValueError('potential installed handles overlap')
        handles.update(s['fit']['handle'])
    return dict(bg=bg, wire=wire, other=other, lock_color=lock_color,
                endpoint_color=endpoint_color, size=size, stations=stations,
                key_cells=key_cells, key_regions=key_regions, key_colors=key_colors,
                connectors=connectors, edges=edges, adj=adj, allowed=allowed,
                endpoints=(0, 1), path=path)


def _paint(inp, d, path=None):
    nodes, edges = d['path'] if path is None else path
    out = np.array(inp, copy=True)
    for i in nodes[1:-1]:
        fit = d['stations'][i]['fit']
        for p in fit['plug']:
            out[p] = fit['plug_color']
        for p in fit['handle']:
            out[p] = fit['handle_color']
    for e in edges:
        for p in d['connectors'][e]:
            out[p] = d['other']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Install the unique minimum-station conducting route, preserving the key."""
    inp = _array(input_grid)
    return _paint(inp, _analyze(inp))


def _shape(rng, size):
    rectangle = _rectangle((0, size - 1, 0, size - 1))
    for _ in range(300):
        y = int(rng.integers(1, size - 1))
        slot = {(y, 0)}
        for c in range(1, size):
            yy = int(np.clip(y + rng.integers(-1, 2), 1, size - 2))
            slot.update((r, c) for r in range(min(y, yy), max(y, yy) + 1))
            y = yy
        slot = frozenset(slot)
        target = rectangle - slot
        if (len(_components(target)) != 2 or _bbox(target) != (0, size-1, 0, size-1)
                or len({_signature(_transform(slot, op)) for op in range(8)}) != 8):
            continue
        side = int(rng.integers(2))
        column = -1 if side == 0 else size
        border = 0 if side == 0 else size - 1
        starts = sorted(r for r, c in slot if c == border)
        handle = {(starts[int(rng.integers(len(starts)))], column)}
        length = int(rng.integers(2, min(2 * size - 1, len(slot) + 2) + 1))
        strip = {(r, c) for r in range(size)
                 for c in ((-2, -1) if side == 0 else (size, size + 1))}
        for _ in range(length - 1):
            frontier = sorted({(r + dr, c + dc) for r, c in handle for dr, dc in DIR4}
                              & strip - handle)
            handle.add(frontier[int(rng.integers(len(frontier)))])
        op = int(rng.integers(8))
        rr = _transform(rectangle, op)
        r0, _, c0, _ = _bbox(rr)
        return tuple(_shift(_transform(x, op), (-r0, -c0))
                     for x in (rectangle, slot, frozenset(handle)))
    raise RuntimeError('bounded asymmetric separator sampling failed')


def _random_graph(rng, size):
    nr, nc = ((2, 3), (3, 2), (3, 3), (2, 4), (4, 2))[int(rng.integers(5))]
    if size == 5 and max(nr, nc) == 4:
        nr, nc = 3, 2
    vertices = {(r, c) for r in range(nr) for c in range(nc)}
    choices = sorted((a, b) for a, b in itertools.combinations(sorted(vertices), 2)
                     if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1)
    # Random spanning tree, then independently add extra edges.
    reached = {min(vertices)}
    edges = []
    while reached != vertices:
        frontier = [e for e in choices if (e[0] in reached) != (e[1] in reached)]
        e = frontier[int(rng.integers(len(frontier)))]
        edges.append(e)
        reached.update(e)
    for e in choices:
        if e not in edges and rng.random() < .65:
            edges.append(e)
    degree = {v: sum(v in e for e in edges) for v in vertices}
    visible = {v for v in vertices if degree[v] != 2 or rng.random() < .65}
    if len(visible) < 5:
        visible.update(sorted(vertices - visible)[:5 - len(visible)])
    vv = sorted(visible)
    endpoints = tuple(vv[i] for i in rng.choice(len(vv), 2, replace=False))
    opened = {v for v in visible - set(endpoints) if rng.random() < .72}
    rows = [0]
    cols = [0]
    for _ in range(nr - 1):
        rows.append(rows[-1] + size + int(rng.integers(4, 7)))
    for _ in range(nc - 1):
        cols.append(cols[-1] + size + int(rng.integers(4, 7)))
    return vertices, edges, visible, endpoints, opened, rows, cols


def _witness_graph(profile, rng):
    vertices = {(r, c) for r in range(3) for c in range(3)}
    edges = [((r, c), (r, c+1)) for r in (0, 2) for c in range(2)]
    edges += [((r, c), (r+1, c)) for c in (0, 2) for r in range(2)]
    endpoints = ((1, 0), (1, 2))
    if profile == 'distance':
        visible = set(endpoints) | {(0, 1), (2, 0), (2, 1), (2, 2), (1, 1)}
        edges += [((1, 0), (1, 1))]  # irrelevant leaf at a selected junction
        opened = visible - set(endpoints)
        rows = [0, 16, 24]
    elif profile == 'gate':
        visible = set(endpoints) | {(0, 0), (0, 1), (2, 0), (2, 1), (2, 2), (1, 1)}
        edges += [((1, 0), (1, 1)), ((1, 1), (1, 2))]
        opened = visible - set(endpoints) - {(1, 1)}
        rows = [0, int(rng.integers(8, 11)), 20]
    else:  # three open ordinary stations; two shorter routes are blocked
        visible = set(endpoints) | {(0, 0), (0, 1), (0, 2), (2, 0), (2, 2), (1, 1)}
        edges += [((1, 0), (1, 1)), ((1, 1), (1, 2))]
        opened = {(0, 0), (0, 1), (0, 2), (2, 0)}
        rows = [0, int(rng.integers(8, 11)), 20]
    cols = [0, int(rng.integers(8, 10)), 18]
    return vertices, edges, visible, endpoints, opened, rows, cols


def _walk(allowed, starts, ends, rng):
    """Small randomized BFS for drawing a wire inside its own corridor."""
    starts = sorted(starts)
    ends = set(ends)
    rng.shuffle(starts)
    if not starts or not ends:
        raise ValueError('no free station port')
    prev = {p: None for p in starts}
    queue = deque(starts)
    directions = list(DIR4)
    rng.shuffle(directions)
    while queue:
        p = queue.popleft()
        if p in ends:
            path = set()
            while p is not None:
                path.add(p)
                p = prev[p]
            return frozenset(path)
        for dr, dc in directions:
            q = p[0] + dr, p[1] + dc
            if q in allowed and q not in prev:
                prev[q] = p
                queue.append(q)
    raise ValueError('reserved key handle blocks a wire corridor')


def _construct(rng, shape, graph, plug_is_wire, palette):
    rectangle, plug, handle = shape
    vertices, lattice_edges, visible, endpoints, opened, rows, cols = graph
    size = len({r for r, _ in rectangle})
    blocks, slots, handles, targets, operations = {}, {}, {}, {}, {}
    for v in sorted(visible):
        origin = rows[v[0]], cols[v[1]]
        blocks[v] = _shift(rectangle, origin)
        if v in endpoints:
            targets[v] = blocks[v]
            continue
        parity = int((v in opened) != plug_is_wire)
        op = int(rng.integers(4)) + 4 * parity
        rr = _transform(rectangle, op)
        r0, _, c0, _ = _bbox(rr)
        delta = origin[0] - r0, origin[1] - c0
        slots[v] = _shift(_transform(plug, op), delta)
        handles[v] = _shift(_transform(handle, op), delta)
        targets[v] = blocks[v] - slots[v]
        operations[v] = op
    reserved = set().union(*handles.values())
    forbidden_ports = {(r+dr, c+dc) for slot in slots.values()
                       for r, c in slot for dr, dc in DIR4}
    wires, ports = {}, {v: [] for v in vertices}
    for a, b in lattice_edges:
        ar, ac, br, bc = rows[a[0]], cols[a[1]], rows[b[0]], cols[b[1]]
        if a[0] == b[0]:
            if ac > bc:
                a, b, ar, ac, br, bc = b, a, br, bc, ar, ac
            corridor = {(r, c) for r in range(ar, ar+size) for c in range(ac+size, bc)}
            starts = {(r, ac+size) for r in range(ar, ar+size)
                      if a not in visible or (r, ac+size-1) in targets[a]}
            ends = {(r, bc-1) for r in range(br, br+size)
                    if b not in visible or (r, bc) in targets[b]}
            inside = lambda p, left: (p[0], p[1] - 1 if left else p[1] + 1)
        else:
            if ar > br:
                a, b, ar, ac, br, bc = b, a, br, bc, ar, ac
            corridor = {(r, c) for r in range(ar+size, br) for c in range(ac, ac+size)}
            starts = {(ar+size, c) for c in range(ac, ac+size)
                      if a not in visible or (ar+size-1, c) in targets[a]}
            ends = {(br-1, c) for c in range(bc, bc+size)
                    if b not in visible or (br, c) in targets[b]}
            inside = lambda p, left: (p[0] - 1 if left else p[0] + 1, p[1])
        corridor -= reserved | forbidden_ports
        path = _walk(corridor, starts & corridor, ends & corridor, rng)
        pa = min(path & starts)
        pb = min(path & ends)
        ports[a].append(inside(pa, True))
        ports[b].append(inside(pb, False))
        wires[tuple(sorted((a, b)))] = path
    hidden_wires = {}
    for v in vertices - visible:
        if len(ports[v]) != 2:
            raise ValueError('only degree-two bends may be suppressed')
        allowed = _rectangle((rows[v[0]], rows[v[0]]+size-1,
                              cols[v[1]], cols[v[1]]+size-1))
        hidden_wires[v] = _walk(allowed, [ports[v][0]], [ports[v][1]], rng)
    neighbors = {v: [] for v in vertices}
    for a, b in lattice_edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    order = list(endpoints) + sorted(visible - set(endpoints))
    ids = {v: i for i, v in enumerate(order)}
    edge_masks = {}
    for v in order:
        for nxt in neighbors[v]:
            old, cur = v, nxt
            mask = set(wires[tuple(sorted((old, cur)))])
            while cur not in visible:
                mask.update(hidden_wires[cur])
                following = next(x for x in neighbors[cur] if x != old)
                old, cur = cur, following
                mask.update(wires[tuple(sorted((old, cur)))])
            if cur == v:
                raise ValueError('suppressed self-loop')
            e = tuple(sorted((ids[v], ids[cur])))
            if e in edge_masks and edge_masks[e] != frozenset(mask):
                raise ValueError('suppressed parallel routes')
            edge_masks[e] = frozenset(mask)
    edges = sorted(edge_masks)
    allowed_ids = {0, 1} | {ids[v] for v in opened}
    planned_path = _shortest(_adjacency(len(order), edges), (0, 1), allowed_ids)
    if planned_path is None:
        raise ValueError('no unique planned conducting path')
    network = set().union(*blocks.values(), reserved, *edge_masks.values())
    low_r, hi_r, low_c, hi_c = _bbox(network)
    if hi_r - low_r >= 30 or hi_c - low_c >= 30:
        raise ValueError('network exceeds the canvas')
    delta = -low_r, -low_c
    blocks = {v: _shift(x, delta) for v, x in blocks.items()}
    targets = {v: _shift(x, delta) for v, x in targets.items()}
    slots = {v: _shift(x, delta) for v, x in slots.items()}
    handles = {v: _shift(x, delta) for v, x in handles.items()}
    edge_masks = {e: _shift(x, delta) for e, x in edge_masks.items()}
    inp = np.zeros((30, 30), dtype=np.int8)
    for x in edge_masks.values():
        for p in x:
            inp[p] = 1
    for v, x in targets.items():
        for p in x:
            inp[p] = 4 if v in endpoints else 3
    out = inp.copy()
    for i in planned_path[0][1:-1]:
        v = order[i]
        pc, hc = ((1, 2) if plug_is_wire else (2, 1))
        if operations[v] >= 4:
            pc, hc = hc, pc
        for p in slots[v]:
            out[p] = pc
        for p in handles[v]:
            out[p] = hc
    for e in planned_path[1]:
        for p in edge_masks[edges[e]]:
            out[p] = 2
    # Pack the reusable master key in any free area, with a four-neighbor moat.
    occupied = set().union(*blocks.values(), *handles.values(), *edge_masks.values())
    occupied |= {(r+dr, c+dc) for r, c in occupied for dr, dc in DIR4}
    r0, r1, c0, c1 = _bbox(plug | handle)
    key_regions = [_shift(x, (-r0, -c0)) for x in (plug, handle)]
    positions = [(r, c) for r in range(30-(r1-r0)) for c in range(30-(c1-c0))]
    rng.shuffle(positions)
    for origin in positions:
        master = [_shift(x, origin) for x in key_regions]
        if not (set().union(*master) & occupied):
            break
    else:
        raise ValueError('master key does not pack')
    for region, color in zip(master, (1, 2) if plug_is_wire else (2, 1)):
        for p in region:
            inp[p] = out[p] = color
    # Random global rigid changes do not change relative fitting parity.
    rotation = int(rng.integers(4))
    inp, out = np.rot90(inp, rotation), np.rot90(out, rotation)
    if rng.random() < .5:
        inp, out = inp[:, ::-1], out[:, ::-1]
    used = _cells((inp != 0) | (out != 0))
    r0, r1, c0, c1 = _bbox(used)
    margins = rng.integers(0, 4, size=4)
    r0, r1 = max(0, r0-int(margins[0])), min(29, r1+int(margins[1]))
    c0, c1 = max(0, c0-int(margins[2])), min(29, c1+int(margins[3]))
    inp, out = inp[r0:r1+1, c0:c1+1].copy(), out[r0:r1+1, c0:c1+1].copy()
    lookup = np.array(palette, dtype=np.int8)
    return {'input': lookup[inp], 'output': lookup[out]}


def _features(d):
    nodes, chosen_edges = d['path']
    ungated = _shortest(d['adj'], d['endpoints'], set(range(len(d['stations']))))
    weights = [len(x) for x in d['connectors']]
    centers = [np.array(tuple(s['rectangle']), dtype=float).mean(axis=0)
               for s in d['stations']]
    lengths = [float(np.linalg.norm(centers[a]-centers[b])) for a, b in d['edges']]
    return dict(selected=len(nodes)-2, station_count=len(d['stations']),
                closed=len(d['stations'])-len(d['allowed']),
                off_route_open=len(d['allowed'] - set(nodes)),
                gate_witness=ungated is not None and ungated[0] != nodes,
                distance_witness=sum(weights[e] for e in chosen_edges) >
                _weighted_min(d['adj'], d['endpoints'], d['allowed'], weights),
                center_distance_witness=sum(lengths[e] for e in chosen_edges) >
                _weighted_min(d['adj'], d['endpoints'], d['allowed'], lengths) + 1e-8,
                parities={d['stations'][i]['fit']['parity'] for i in nodes[1:-1]})


def _topology_eligible(graph, profile):
    """Reject unsuitable topology before spending time drawing its pixels.

    This only accelerates generation. Returned inputs are still reconstructed
    and all witness conditions are checked again using their actual geometry.
    """
    vertices, lattice_edges, visible, endpoints, opened, rows, cols = graph
    neighbors = {v: [] for v in vertices}
    for a, b in lattice_edges:
        neighbors[a].append(b)
        neighbors[b].append(a)
    order = list(endpoints) + sorted(visible - set(endpoints))
    ids = {v: i for i, v in enumerate(order)}
    traces = {}
    for v in order:
        for nxt in neighbors[v]:
            old, cur = v, nxt
            trace = {tuple(sorted((old, cur)))}
            while cur not in visible:
                if len(neighbors[cur]) != 2:
                    return False
                following = next(x for x in neighbors[cur] if x != old)
                old, cur = cur, following
                trace.add(tuple(sorted((old, cur))))
            if cur == v:
                return False
            edge = tuple(sorted((ids[v], ids[cur])))
            if edge in traces and traces[edge] != trace:
                return False
            traces[edge] = trace
    edges = sorted(traces)
    adj = _adjacency(len(order), edges)
    allowed = {0, 1} | {ids[v] for v in opened}
    path = _shortest(adj, (0, 1), allowed)
    if path is None or len(path[0]) < 3 or not (allowed - set(path[0])):
        return False
    if profile in ('gate', 'test'):
        ungated = _shortest(adj, (0, 1), set(range(len(order))))
        if ungated is None or ungated[0] == path[0]:
            return False
    if profile == 'test' and len(path[0]) < 5:
        return False
    if profile == 'distance':
        centers = [np.array((rows[v[0]], cols[v[1]]), dtype=float) for v in order]
        lengths = [float(np.linalg.norm(centers[a]-centers[b])) for a, b in edges]
        if sum(lengths[e] for e in path[1]) <= _weighted_min(adj, (0, 1), allowed, lengths) + 1e-8:
            return False
    return True


def _make_pair(rng, profile, plug_is_wire, palette):
    # Random graphs first; constructive profiles ensure the two demonstrations
    # and the test shift cannot disappear through unlucky sampling.
    for attempt in range(120):
        size = 4 if attempt >= 25 else int(rng.choice([4, 5]))
        shape = _shape(rng, size)
        graph = (_witness_graph(profile, rng) if attempt >= 25 and profile != 'general'
                 else _random_graph(rng, size))
        if not _topology_eligible(graph, profile):
            continue
        try:
            pair = _construct(rng, shape, graph, plug_is_wire, palette)
            d = _analyze(pair['input'])
            f = _features(d)
            if f['selected'] < 1 or not f['off_route_open']:
                continue
            if profile == 'gate' and not f['gate_witness']:
                continue
            if profile == 'distance' and not (f['distance_witness'] and f['center_distance_witness']):
                continue
            if profile == 'test' and (f['selected'] < 3 or not f['gate_witness']):
                continue
            if not np.array_equal(_paint(pair['input'], d), pair['output']):
                continue
            return pair
        except ValueError:
            continue
    raise RuntimeError('bounded scene construction exhausted')


def _is_int(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _is_int(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (not _is_int(num_train) or not _is_int(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4
            or num_train <= num_test):
        raise ValueError('require 2..10 training, 1..4 tests, and training > tests')
    rng = np.random.default_rng(seed)
    stable_palette = rng.random() < .25
    palette = tuple(int(x) for x in rng.choice(10, 5, replace=False))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            profile = ('gate' if i == 0 else 'distance' if i == 1 else 'general') if split == 'train' else 'test'
            plug_is_wire = (i == 0) if split == 'train' and i < 2 else bool(rng.integers(2))
            colors = palette if stable_palette else tuple(int(x) for x in rng.choice(10, 5, replace=False))
            for _ in range(12):
                pair = _make_pair(rng, profile, plug_is_wire, colors)
                k = _key(pair['input'])
                if k not in seen:
                    seen.add(k)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input retries exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train_features, parities = set(), [], set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _array(pair['input']), _array(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            d = _analyze(inp)
            assert np.array_equal(out, _paint(inp, d))
            f = _features(d)
            assert f['selected'] >= 1 and f['off_route_open'] >= 1
            # Check complete installations, exact off-route preservation, and
            # the fact that the reusable template and both endpoints stay put.
            changed = set()
            for i in d['path'][0][1:-1]:
                fit = d['stations'][i]['fit']
                assert fit['plug_color'] == d['wire'] and fit['handle_color'] == d['other']
                assert all(out[p] == d['wire'] for p in fit['plug'])
                assert all(out[p] == d['other'] for p in fit['handle'])
                changed.update(fit['plug'] | fit['handle'])
            for e in d['path'][1]:
                changed.update(d['connectors'][e])
                assert all(out[p] == d['other'] for p in d['connectors'][e])
            assert _cells(inp != out) == changed
            assert all(out[p] == inp[p] for p in d['key_cells'])
            assert all(out[p] == inp[p] for s in d['stations'] for p in s['target'])
            if split == 'train':
                train_features.append(f)
                parities.update(f['parities'])
            else:
                assert f['selected'] >= 3 and f['gate_witness']
    assert any(f['gate_witness'] for f in train_features)
    assert any(f['distance_witness'] and f['center_distance_witness'] for f in train_features)
    assert parities == {0, 1}


def self_test():
    """Bounded 42-seed semantic and diversity checks; no external judge."""
    layouts, dimensions, sizes, shapes, handles, palettes = set(), set(), set(), set(), set(), set()
    node_counts, route_counts, closed_counts, degree_sets, relations = set(), set(), set(), set(), set()
    all_parities, edge_contacts = set(), False
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        if seed < 2:
            q = generate_puzzle(seed)
            assert all(_key(a[k]) == _key(b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
        for split in p:
            for pair in p[split]:
                inp, out = pair['input'], pair['output']
                d = _analyze(inp)
                assert np.array_equal(solve(inp.copy()), out)
                assert np.array_equal(solve(inp.tolist()), out)
                f = _features(d)
                dimensions.add(inp.shape)
                sizes.add(d['size'])
                node_counts.add(f['station_count'])
                route_counts.add(f['selected'])
                closed_counts.add(f['closed'])
                degree_sets.add(tuple(sorted(len(a) for a in d['adj'])))
                palettes.add((d['bg'], d['wire'], d['other'], d['lock_color'], d['endpoint_color']))
                all_parities.update(f['parities'])
                semantic = np.zeros(inp.shape, dtype=np.int8)
                for role, color in enumerate((d['bg'], d['wire'], d['other'], d['lock_color'], d['endpoint_color'])):
                    semantic[inp == color] = role
                layouts.add(_key(semantic))
                for s in d['stations'][2:]:
                    fit = s['fit']
                    shapes.add(min(_signature(_transform(s['slot'], op)) for op in range(8)))
                    handles.add(min(_signature(_transform(fit['handle'], op)) for op in range(8)))
                    relations.add(int(np.sign(len(fit['plug']) - len(fit['handle']))))
                edge_contacts |= bool(np.any(inp[0] != d['bg']) or np.any(inp[-1] != d['bg'])
                                      or np.any(inp[:, 0] != d['bg']) or np.any(inp[:, -1] != d['bg']))
    assert len(layouts) >= 180 and len(dimensions) >= 35
    assert sizes == {4, 5} and len(shapes) >= 8 and len(handles) >= 15
    assert len(node_counts) >= 5 and len(route_counts) >= 3 and len(closed_counts) >= 3
    assert len(degree_sets) >= 15 and len(palettes) >= 80
    assert all_parities == {0, 1} and relations == {-1, 0, 1} and edge_contacts
    # Count boundaries; verify.py separately exercises all 30 legal combinations.
    for train, test in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(900 + 10*train + test, train, test)
        assert len(p['train']) == train and len(p['test']) == test
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    adj = _adjacency(4, [(0, 2), (2, 1), (0, 3), (3, 1)])
    assert _shortest(adj, (0, 1), set(range(4))) is None
    ambiguous = np.zeros((10, 12), dtype=np.int8)
    rectangle = _rectangle((3, 5, 3, 6))
    slot = frozenset((4, c) for c in range(3, 7))
    for p in rectangle - slot:
        ambiguous[p] = 3
    try:
        _fits(ambiguous, (frozenset((0, c) for c in range(4)),
                          frozenset({(0, 4), (1, 4)})), (1, 2),
              dict(slot=slot, rectangle=rectangle), 0)
    except ValueError:
        pass
    else:
        raise AssertionError('ambiguous complete colored fits accepted')
    p = generate_puzzle(123, 2, 1)
    p['test'][0]['output'] = p['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('identity output accepted')
    return {'seeds': 42, 'pairs': 210, 'structures': len(layouts), 'dimensions': len(dimensions),
            'slot_shapes': len(shapes), 'handle_shapes': len(handles),
            'station_counts': sorted(node_counts), 'selected_counts': sorted(route_counts)}


if __name__ == '__main__':
    print(self_test())
