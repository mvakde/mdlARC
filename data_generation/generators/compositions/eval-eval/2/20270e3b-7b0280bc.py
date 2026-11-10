"""Registered transit overlays: fuse two maps, extract a minimum-stop route.

Only standard-library modules and NumPy are used. The solver recovers ownership,
palette, registration, promoted stations, and graph edges from input pixels.
"""
from collections import deque
import heapq
import json

import numpy as np


BACKGROUND = 0
GUIDE = 8
DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))
INK_POOL = (1, 2, 3, 4, 5, 6, 7, 9)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid colors")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _components(cells):
    remaining = set(cells)
    parts = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        part, todo = {p}, [p]
        while todo:
            r, c = todo.pop()
            for dr, dc in DIRECTIONS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    todo.append(q)
        parts.append(part)
    return parts


def _cells(grid, color):
    return {tuple(map(int, p)) for p in np.argwhere(grid == color)}


def _bbox(cells):
    _require(bool(cells), "empty object")
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _square(part):
    t, b, l, r = _bbox(part)
    return b - t + 1 if b - t == r - l and len(part) == (b - t + 1) ** 2 else 0


def _translate(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _read_input(grid):
    """Identify the three diagram inks by complete square station components."""
    colors = set(map(int, np.unique(grid)))
    _require(len(colors) == 5 and {BACKGROUND, GUIDE} <= colors, "expected five input roles")
    inks = sorted(colors - {BACKGROUND, GUIDE})
    by_color = {c: _cells(grid, c) for c in inks + [GUIDE]}
    parts = {c: _components(by_color[c]) for c in inks}
    end_choices = [c for c in inks if len(parts[c]) == 2 and
                   len({_square(p) for p in parts[c]}) == 1 and _square(parts[c][0]) in (2, 3)]
    _require(len(end_choices) == 1, "endpoints must be uniquely recognizable")
    endpoint = end_choices[0]
    size = _square(parts[endpoint][0])
    stop_choices = [c for c in inks if c != endpoint and len(parts[c]) >= 3 and
                    all(_square(p) == size for p in parts[c])]
    _require(len(stop_choices) == 1, "ordinary station role is ambiguous")
    station = stop_choices[0]
    wire = next(c for c in inks if c not in (station, endpoint))
    guides = _components(by_color[GUIDE])
    _require(len(guides) == 2 and len(guides[0]) == len(guides[1]), "need two congruent bars")
    bar_shapes = []
    for g in guides:
        t, b, l, r = _bbox(g)
        _require(len(g) >= 2 and (t == b or l == r), "guides must be straight, nonsingleton bars")
        bar_shapes.append((b - t, r - l))
    _require(bar_shapes[0] == bar_shapes[1], "guide axes must agree")
    fragments = _components(by_color[GUIDE] | set.union(*(by_color[c] for c in inks)))
    _require(len(fragments) == 2, "need two separated connected maps")
    layers, local_guides = [], []
    for fragment in fragments:
        own = [g for g in guides if g <= fragment]
        _require(len(own) == 1, "each map must own one registration bar")
        _require(sum(p <= fragment for p in parts[endpoint]) == 1, "one endpoint per map")
        g = own[0]
        # The guide directly touches one complete station, not a wire or a second stop.
        touching = [p for p in parts[station] if any(
            (r + dr, c + dc) in p for r, c in g for dr, dc in DIRECTIONS)]
        _require(len(touching) == 1, "guide must attach to one ordinary station")
        _require(not any((r + dr, c + dc) in by_color[wire]
                         for r, c in g for dr, dc in DIRECTIONS), "guide touches a wire")
        origin = min(g)
        layers.append({c: _translate(by_color[c] & fragment, (-origin[0], -origin[1]))
                       for c in inks})
        local_guides.append(_translate(g, (-origin[0], -origin[1])))
    _require(local_guides[0] == local_guides[1], "registration mismatch")
    # Matching guide origins means translation only. Guides and blanks do not paste.
    fused_stops = layers[0][station] | layers[1][station]
    fused_ends = layers[0][endpoint] | layers[1][endpoint]
    _require(not fused_stops & fused_ends, "endpoints may not overlap stations")
    fused_wires = (layers[0][wire] | layers[1][wire]) - fused_stops - fused_ends
    return dict(roles=(station, wire, endpoint), size=size, layers=layers,
                guides=local_guides[0], stations=fused_stops, endpoints=fused_ends,
                wires=fused_wires, fragments=fragments)


def _network(station_cells, endpoint_cells, wires, size, connected=True):
    """Recover edges by removing square stations, then tracing wire components."""
    blocks = _components(station_cells | endpoint_cells)
    _require(all(_square(p) == size for p in blocks), "partial or touching station blocks")
    owners = {q: n for n, p in enumerate(blocks) for q in p}
    end_ids = [n for n, p in enumerate(blocks) if p <= endpoint_cells]
    _require(all(p <= station_cells or p <= endpoint_cells for p in blocks), "mixed station colors")
    edges, adjacency = [], [[] for _ in blocks]
    for component in _components(wires):
        touching = {owners[(r + dr, c + dc)] for r, c in component for dr, dc in DIRECTIONS
                    if (r + dr, c + dc) in owners}
        _require(len(touching) == 2, "every wire must connect exactly two stations")
        a, b = sorted(touching)
        e = len(edges)
        edges.append((a, b, component))
        adjacency[a].append((b, e))
        adjacency[b].append((a, e))
    _require(all(1 <= len(n) <= 4 for n in adjacency), "invalid station degree")
    reached, todo = {0}, [0]
    while todo:
        for b, e in adjacency[todo.pop()]:
            if b not in reached:
                reached.add(b)
                todo.append(b)
    _require(not connected or len(reached) == len(blocks), "assembled map is disconnected")
    return dict(blocks=blocks, end_ids=end_ids, edges=edges, adjacency=adjacency)


def _minimum_route(net):
    _require(len(net["end_ids"]) == 2, "need exactly two assembled endpoints")
    start, goal = net["end_ids"]
    dist, ways, previous = {start: 0}, {start: 1}, {}
    todo = deque([start])
    while todo:
        a = todo.popleft()
        for b, e in net["adjacency"][a]:
            if b not in dist:
                dist[b], ways[b], previous[b] = dist[a] + 1, 0, (a, e)
                todo.append(b)
            if dist[b] == dist[a] + 1:
                ways[b] = min(2, ways[b] + ways[a])
    _require(ways.get(goal) == 1, "minimum station route is absent or tied")
    nodes, edges = [goal], []
    while nodes[-1] != start:
        a, e = previous[nodes[-1]]
        nodes.append(a)
        edges.append(e)
    return nodes[::-1], edges[::-1]


def _render_route(data, net, route):
    station, wire, endpoint = data["roles"]
    nodes, edge_ids = route
    retained = set.union(*(net["blocks"][n] for n in nodes))
    retained |= set.union(*(net["edges"][e][2] for e in edge_ids))
    t, b, l, r = _bbox(retained)
    out = np.zeros((b - t + 1, r - l + 1), dtype=np.int8)
    for e in edge_ids:
        for y, x in net["edges"][e][2]:
            out[y - t, x - l] = wire
    for n in nodes:
        for y, x in net["blocks"][n]:
            out[y - t, x - l] = endpoint if n in net["end_ids"] else station
    return out


def _analyze(input_grid):
    data = _read_input(input_grid)
    station, wire, endpoint = data["roles"]
    # Check each visible map independently, not just the fused rendering.
    for layer in data["layers"]:
        _network(layer[station], layer[endpoint], layer[wire], data["size"])
    net = _network(data["stations"], data["endpoints"], data["wires"], data["size"])
    route = _minimum_route(net)
    data.update(net=net, route=route)
    return data


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the route entirely from one grid. No training, seed, or stored state."""
    data = _analyze(_grid(input_grid))
    return _render_route(data, data["net"], data["route"])


def _weighted_minimum(net, node_weights=None):
    start, goal = net["end_ids"]
    costs, todo = {start: 0}, [(0, start)]
    while todo:
        cost, a = heapq.heappop(todo)
        if cost != costs[a]:
            continue
        if a == goal:
            return cost
        for b, e in net["adjacency"][a]:
            candidate = cost + (len(net["edges"][e][2]) if node_weights is None else node_weights[b])
            if candidate < costs.get(b, float("inf")):
                costs[b] = candidate
                heapq.heappush(todo, (candidate, b))
    return float("inf")


def _features(data):
    station, wire, endpoint = data["roles"]
    net, (nodes, edges) = data["net"], data["route"]
    a, b = data["layers"]
    shared = [p for p in net["blocks"] if p <= a[station] and p <= b[station]]
    promoted = [p for p in net["blocks"] if
                (p <= a[station] and bool(p & b[wire])) or
                (p <= b[station] and bool(p & a[wire]))]
    route_sets = [net["edges"][e][2] for e in edges]
    only_a = any(p <= a[wire] and not p <= b[wire] for p in route_sets)
    only_b = any(p <= b[wire] and not p <= a[wire] for p in route_sets)
    promotion_transfers = 0
    for i, n in enumerate(nodes[1:-1], 1):
        if net["blocks"][n] not in promoted:
            continue
        incoming, outgoing = route_sets[i - 1], route_sets[i]
        exclusive = lambda p, first, second: p <= first[wire] and not p <= second[wire]
        if (exclusive(incoming, a, b) and exclusive(outgoing, b, a) or
                exclusive(incoming, b, a) and exclusive(outgoing, a, b)):
            promotion_transfers += 1
    selected_area = sum(len(p) for p in route_sets)
    copies = [sum(p <= layer[station] for layer in data["layers"])
              if n not in net["end_ids"] else 0 for n, p in enumerate(net["blocks"])]
    copy_cost = sum(copies[n] for n in nodes[1:-1])
    return dict(stations=len(net["blocks"]), intermediate=len(nodes) - 2,
                edges=len(net["edges"]), loops=len(net["edges"]) - len(net["blocks"]) + 1,
                shared_stations=len(shared), promoted_stations=len(promoted),
                selected_promotions=sum(net["blocks"][n] in promoted for n in nodes),
                promotion_transfers=promotion_transfers,
                selected_area=selected_area,
                distance_witness=selected_area > _weighted_minimum(net),
                fusion_witness=copy_cost > _weighted_minimum(net, copies),
                uses_both=only_a and only_b,
                spare_edges=len(net["edges"]) - len(edges),
                size=data["size"], guide_length=len(data["guides"]))


def _all_routes(net):
    """Small bounded family: exhaustive check of BFS's unique minimum."""
    start, goal = net["end_ids"]
    todo, routes = [(start, [start], [])], []
    while todo:
        a, nodes, edges = todo.pop()
        if a == goal:
            routes.append((nodes, edges))
        else:
            for b, e in net["adjacency"][a]:
                if b not in nodes:
                    todo.append((b, nodes + [b], edges + [e]))
        _require(len(todo) + len(routes) < 20000, "too many routes for bounded validation")
    return sorted(routes, key=lambda p: len(p[0]))


def _without_station_promotions(data):
    """Ablation graph: sheets can transfer only at explicitly duplicated squares."""
    station, wire, endpoint = data["roles"]
    blocks, identities, ends, edges, seen = [], {}, set(), [], set()
    for layer in data["layers"]:
        local = _network(layer[station], layer[endpoint], layer[wire], data["size"])
        ids = {}
        for n, block in enumerate(local["blocks"]):
            key = frozenset(block)
            if key not in identities:
                identities[key] = len(blocks)
                blocks.append(block)
            ids[n] = identities[key]
            if n in local["end_ids"]:
                ends.add(ids[n])
        for a, b, cells in local["edges"]:
            a, b = sorted((ids[a], ids[b]))
            key = a, b, frozenset(cells)
            if key not in seen:
                seen.add(key)
                edges.append((a, b, cells))
    adjacency = [[] for _ in blocks]
    for e, (a, b, cells) in enumerate(edges):
        adjacency[a].append((b, e))
        adjacency[b].append((a, e))
    return dict(blocks=blocks, edges=edges, adjacency=adjacency, end_ids=sorted(ends))


def _route_cells(net, route):
    nodes, edges = route
    return set.union(*(net["blocks"][n] for n in nodes), *(net["edges"][e][2] for e in edges))


def _lattice(rng, nr, nc):
    vertices = [(r, c) for r in range(nr) for c in range(nc)]
    possible = [(p, q) for p in vertices for q in vertices if p < q and
                abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1]

    def grow():
        active = {vertices[int(rng.integers(len(vertices)))]}
        target = int(rng.integers(4, min(len(vertices), 11) + 1))
        edges = []
        while len(active) < target:
            choices = [(p, q) for p, q in possible if (p in active) != (q in active)]
            e = choices[int(rng.integers(len(choices)))]
            edges.append(e)
            active.update(e)
        extras = [e for e in possible if set(e) <= active and e not in edges]
        rng.shuffle(extras)
        edges += extras[:int(rng.integers(min(3, len(extras)) + 1))]
        return active, edges

    aa, ae = grow()
    bb, be = grow()
    shared = aa & bb
    if len(shared) < 2 or not aa - bb or not bb - aa:
        return None
    endpoints = (sorted(aa - bb)[int(rng.integers(len(aa - bb)))],
                 sorted(bb - aa)[int(rng.integers(len(bb - aa)))])
    union_edges = set(ae) | set(be)
    available = []
    for v in sorted(shared):
        neighbors = {q if p == v else p for p, q in union_edges if v in (p, q)}
        for direction in DIRECTIONS:
            q = v[0] + direction[0], v[1] + direction[1]
            if q not in neighbors:
                available.append((v, direction))
    if not available:
        return None
    marker, direction = available[int(rng.integers(len(available)))]
    visible = []
    for active, edges, end in ((aa, ae, endpoints[0]), (bb, be, endpoints[1])):
        degree = {v: sum(v in e for e in edges) for v in active}
        visible.append({v for v in active if degree[v] != 2 or rng.random() > .48} | {marker, end})
    # Two invisible crossing bends may never form a junction. An actual square
    # on at least one sheet is what licenses a transfer.
    for v in sorted(shared):
        if v not in visible[0] | visible[1]:
            visible[int(rng.integers(2))].add(v)
    return ((aa, ae, visible[0]), (bb, be, visible[1])), endpoints, marker, direction


def _fallback(selected_promotion=False, rng=None):
    """Procedural interlocking maps guarantee the requested teaching witness.

    This is a procedural construction, not a saved input/output pair.
    """
    if not selected_promotion:
        # Two shared bottom squares beat three single-sheet top squares. Their
        # doubled copy cost loses 4 versus 3, and the bottom wires are longer.
        left = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 3)]
        top = [(0, 0), (0, 1), (0, 2), (0, 3)]
        right = [(0, 4), (1, 4), (2, 4), (2, 3), (2, 2), (2, 1)]
        inner = [(0, 4), (0, 3), (1, 3), (2, 3)]
        ae = list(zip(left, left[1:])) + list(zip(top, top[1:]))
        be = list(dict.fromkeys(tuple(sorted(e)) for e in
                               list(zip(right, right[1:])) + list(zip(inner, inner[1:]))))
        aa, bb = set(left + top), set(right + inner)
        av = {(0, 0), (0, 1), (0, 2), (0, 3), (2, 1), (2, 3)}
        bv = {(0, 4), (2, 1), (2, 3)}
        leaves = []
        for column in (1, 2):
            if rng is not None and rng.random() < .5:
                v = (1, column)
                aa.add(v)
                av.add(v)
                ae.append(((0, column), v))
                leaves.append(v)
        if len(leaves) == 2 and rng.random() < .5:
            ae.append(tuple(leaves))
        return ((aa, ae, av), (bb, be, bv)), ((0, 0), (0, 4)), (2, 3), (1, 0)
    left = [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)]
    top = [(0, 0), (0, 1), (0, 2)]
    right = [(0, 3), (1, 3), (2, 3), (2, 2)]
    inner = [(0, 3), (0, 2), (1, 2), (2, 2)]
    ae = list(zip(left, left[1:])) + list(zip(top, top[1:]))
    be = list(zip(right, right[1:])) + list(zip(inner, inner[1:]))
    aa, bb = set(left + top), set(right + inner)
    av = {(0, 0), (0, 1), (0, 2), (2, 2)}
    if selected_promotion:
        # Make the bottom arc dense so the promoted top bend lies on the winner.
        av |= {(1, 0), (2, 0)}
    bv = {(0, 3), (2, 2)}
    return ((aa, ae, av), (bb, be, bv)), ((0, 0), (0, 3)), (2, 2), (1, 0)


def _draw_maps(rng, construction, size, nr, nc, length, compact=False):
    graphs, endpoints, marker, direction = construction
    # Unequal row/column gaps vary geometric distance without changing topology.
    rows = [0]
    cols = [0]
    for _ in range(nr - 1):
        gap = (2 if size == 3 else int(rng.integers(2, 4))) if compact else int(rng.integers(2, 5))
        rows.append(rows[-1] + size + gap)
    for _ in range(nc - 1):
        cols.append(cols[-1] + size + int(rng.integers(2, 4)))
    origin = {v: (rows[v[0]], cols[v[1]]) for graph in graphs for v in graph[0]}
    # The fallback's south-facing guide can overhang a small square safely:
    # route east/west wires along its top, away from the temporary guide.
    offset = 0 if compact else int(rng.integers(size))
    blocks = {v: {(origin[v][0] + dr, origin[v][1] + dc)
                  for dr in range(size) for dc in range(size)} for v in origin}
    mr, mc = origin[marker]
    dr, dc = direction
    if dr:
        guide = {(mr - 1 if dr < 0 else mr + size, mc + k) for k in range(length)}
    else:
        guide = {(mr + k, mc - 1 if dc < 0 else mc + size) for k in range(length)}
    layers = []
    for index, (active, edges, visible) in enumerate(graphs):
        wires = set()
        for a, b in edges:
            ra, ca = origin[a]
            rb, cb = origin[b]
            if ra == rb:
                wires.update((ra + offset, c) for c in range(min(ca, cb) + offset, max(ca, cb) + offset + 1))
            else:
                wires.update((r, ca + offset) for r in range(min(ra, rb) + offset, max(ra, rb) + offset + 1))
        # Suppressed degree-two lattice points remain plain connector bends.
        stations = set.union(*(blocks[v] for v in visible - {endpoints[index]}))
        ends = blocks[endpoints[index]]
        wires -= stations | ends
        _require(not guide & (stations | ends | wires), "guide collision")
        layers.append(dict(stations=stations, wires=wires, endpoints=ends, guide=guide))
    return layers


def _stage(rng, layers, roles, stage_axis=None):
    station, wire, endpoint = roles
    normalized = []
    for layer in layers:
        all_cells = set.union(*layer.values())
        t, b, l, r = _bbox(all_cells)
        normalized.append(({k: _translate(v, (-t, -l)) for k, v in layer.items()}, b - t + 1, r - l + 1))
    if rng.random() < .5:
        normalized.reverse()
    a, ah, aw = normalized[0]
    b, bh, bw = normalized[1]
    axes = [stage_axis] if stage_axis is not None else list(map(int, rng.permutation(2)))
    for axis in axes:
        room = 30 - (ah + bh if axis == 0 else aw + bw)
        if room < 1:
            continue
        gap = int(rng.integers(1, min(3, room) + 1))
        if axis == 0:
            height, width = ah + bh + gap, max(aw, bw)
        else:
            height, width = max(ah, bh), aw + bw + gap
        if height > 30 or width > 30:
            continue
        extra_r = int(rng.integers(min(5, 30 - height) + 1))
        extra_c = int(rng.integers(min(5, 30 - width) + 1))
        h, w = height + extra_r, width + extra_c
        top, left = int(rng.integers(extra_r + 1)), int(rng.integers(extra_c + 1))
        if axis == 0:
            apos = (top, int(rng.integers(w - aw + 1)))
            bpos = (top + ah + gap, int(rng.integers(w - bw + 1)))
        else:
            apos = (int(rng.integers(h - ah + 1)), left)
            bpos = (int(rng.integers(h - bh + 1)), left + aw + gap)
        grid = np.zeros((h, w), dtype=np.int8)
        for layer, pos in ((a, apos), (b, bpos)):
            for name, color in (("stations", station), ("wires", wire), ("endpoints", endpoint), ("guide", GUIDE)):
                for r, c in layer[name]:
                    grid[r + pos[0], c + pos[1]] = color
        return grid
    return None


def _make_pair(rng, roles, nr, nc, size, length, required, used):
    random_budget = 40 if required == "witness" else 100
    for attempt in range(random_budget + 40):
        fallback = attempt >= random_budget
        construction = _fallback(required == "promotion", rng) if fallback else _lattice(rng, nr, nc)
        if construction is None:
            continue
        try:
            vertices = set.union(*(graph[0] for graph in construction[0]))
            render_nr = 1 + max(v[0] for v in vertices) if fallback else nr
            render_nc = 1 + max(v[1] for v in vertices) if fallback else nc
            layers = _draw_maps(rng, construction, size, render_nr, render_nc, length, compact=fallback)
            inp = _stage(rng, layers, roles)
            if inp is None:
                continue
            # Whole-scene transforms; neither layer ever rotates during solving.
            inp = np.rot90(inp, int(rng.integers(4))).copy()
            if rng.random() < .5:
                inp = inp[:, ::-1].copy()
            if _key(inp) in used:
                continue
            data = _analyze(inp)
            f = _features(data)
            if not f["uses_both"] or f["spare_edges"] == 0 or f["shared_stations"] == 0:
                continue
            if required == "witness" and not (f["distance_witness"] and f["fusion_witness"] and f["promoted_stations"]):
                continue
            if required == "promotion" and not f["promotion_transfers"]:
                continue
            if f["intermediate"] < 1:
                continue
            out = _render_route(data, data["net"], data["route"])
            return {"input": inp, "output": out}
        except ValueError:
            continue
    raise RuntimeError("bounded map construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, "num_train", 2, 10)
    num_test = _count(num_test, "num_test", 1, 4)
    _require(num_train > num_test, "num_train must exceed num_test")
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base_roles = tuple(map(int, rng.choice(INK_POOL, 3, replace=False)))
    changing_palette = bool(rng.integers(2))
    regime = int(rng.integers(2))
    # A visible, guaranteed shift: square scale or registration-length interpolation.
    common_size = int(rng.choice((2, 2, 3)))
    puzzle, used = {"train": [], "test": []}, set()
    witness_index = int(rng.integers(num_train))
    promotion_index = (witness_index + 1) % num_train
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            roles = tuple(map(int, rng.choice(INK_POOL, 3, replace=False))) if changing_palette else base_roles
            nr = int(rng.choice((3, 4, 5)))
            nc = 3
            size = (2 if split == "train" else 3) if regime == 0 else common_size
            length = 2 if regime == 0 else size
            if regime == 1:
                length = (2 if index % 2 == 0 else 4) if split == "train" else 3
            required = ("witness" if split == "train" and index == witness_index else
                        "promotion" if split == "train" and index == promotion_index else "any")
            pair = _make_pair(rng, roles, nr, nc, size, length, required, used)
            used.add(_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    ntr, nte = len(puzzle["train"]), len(puzzle["test"])
    _count(ntr, "num_train", 2, 10)
    _count(nte, "num_test", 1, 4)
    _require(ntr > nte, "invalid split sizes")
    seen, feats = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            data = _analyze(inp)
            expected = _render_route(data, data["net"], data["route"])
            _require(np.array_equal(expected, out), "incorrect route pixels or tight crop")
            _require(GUIDE not in out, "guide leaked into output")
            station, wire, endpoint = data["roles"]
            _require(len(_cells(out, endpoint)) == 2 * data["size"] ** 2, "endpoint footprint lost")
            # Extracted output is exactly one chain, not an overhighlighted network.
            outnet = _network(_cells(out, station), _cells(out, endpoint), _cells(out, wire), data["size"])
            _require(sorted(map(len, outnet["adjacency"])) == [1, 1] + [2] * (len(outnet["blocks"]) - 2),
                     "output is not one unbranched route")
            f = _features(data)
            _require(f["uses_both"] and f["shared_stations"] >= 1, "assembly has no substantive interaction")
            _require(f["spare_edges"] > 0 and f["intermediate"] >= 1, "route extraction is trivial")
            feats[split].append(f)
    _require(any(f["distance_witness"] and f["fusion_witness"] and f["promoted_stations"]
                 for f in feats["train"]),
             "training does not distinguish station count from wire length and duplicate-copy count")
    _require(any(f["promotion_transfers"] for f in feats["train"]),
             "training does not show an essential transfer at a promoted bend")


def self_test() -> dict:
    """Bounded semantic checks and actual pixel-derived diversity over 42 seeds."""
    signatures, structures, shapes, palettes = set(), set(), set(), set()
    stats = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p[k]) for s in ("train", "test") for p in puzzle[s] for k in ("input", "output")))
        if seed < 3:
            again = generate_puzzle(seed)
            _require(all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s])
                         for k in ("input", "output")), "nondeterministic generation")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp, out = pair["input"], pair["output"]
                data = _analyze(inp)
                f = _features(data)
                stats.append(f)
                _require(np.array_equal(solve(inp.copy()), out), "solve disagreement")
                paths = _all_routes(data["net"])
                _require(paths and (len(paths) == 1 or len(paths[0][0]) < len(paths[1][0])), "BFS missed a tie")
                _require(paths[0] == data["route"], "BFS and enumeration disagree")
                station, wire, endpoint = data["roles"]
                semantic = inp.copy()
                for c, role in ((station, 1), (wire, 2), (endpoint, 3)):
                    semantic[inp == c] = role
                structures.add(_key(semantic))
                shapes.add((inp.shape, out.shape))
                palettes.add(data["roles"])
                # Parent ablations: neither separate map has both endpoints;
                # removing the route criterion retains spare wire pixels.
                _require(all(len(_components(layer[endpoint])) == 1 for layer in data["layers"]),
                         "an unassembled sheet already supplies the complete route")
                full_foreground = data["stations"] | data["endpoints"] | data["wires"]
                kept = set.union(*(data["net"]["blocks"][n] for n in data["route"][0]))
                kept |= set.union(*(data["net"]["edges"][e][2] for e in data["route"][1]))
                _require(kept < full_foreground, "omitting B makes no difference")
                if split == "train" and f["promotion_transfers"]:
                    ablated = _without_station_promotions(data)
                    try:
                        wrong_route = _minimum_route(ablated)
                    except ValueError:
                        # Missing or tied routes also fail to specify the actual answer.
                        pass
                    else:
                        _require(_route_cells(ablated, wrong_route) != kept,
                                 "promotion ablation retained the same answer")
    _require(len(signatures) == 42 and len(structures) >= 150, "insufficient structural variation")
    _require(len(shapes) >= 80 and len(palettes) >= 30, "insufficient geometric/palette variation")
    _require({f["size"] for f in stats} == {2, 3}, "station size coverage missing")
    _require({f["guide_length"] for f in stats} >= {2, 3, 4}, "guide length coverage missing")
    _require(len({f["stations"] for f in stats}) >= 8, "station-count variation missing")
    _require(any(f["loops"] >= 2 for f in stats), "multiple-loop coverage missing")
    _require(any(f["selected_promotions"] >= 2 for f in stats), "combined promotions missing")
    # Altered labels must fail even when schema, dimensions, and colors are legal.
    mutant = generate_puzzle(7, 2, 1)
    mutant["test"][0]["output"][0, 0] = GUIDE
    try:
        validate_generated_puzzle(mutant)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    # A four-node diamond ties in station count regardless of geometry.
    tie = dict(end_ids=[0, 3], adjacency=[[(1, 0), (2, 1)], [(0, 0), (3, 2)],
                                       [(0, 1), (3, 3)], [(1, 2), (2, 3)]])
    try:
        _minimum_route(tie)
    except ValueError:
        pass
    else:
        raise AssertionError("tied paths accepted")
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    report = dict(seeds=42, pairs=len(stats), distinct_puzzles=len(signatures),
                  normalized_structures=len(structures), shape_pairs=len(shapes), palettes=len(palettes),
                  distance_witnesses=sum(f["distance_witness"] for f in stats),
                  fusion_witnesses=sum(f["fusion_witness"] for f in stats),
                  promoted_pairs=sum(f["promoted_stations"] > 0 for f in stats),
                  selected_promotion_pairs=sum(f["selected_promotions"] > 0 for f in stats),
                  essential_promotion_pairs=sum(f["promotion_transfers"] > 0 for f in stats),
                  station_counts=sorted({f["stations"] for f in stats}),
                  route_counts=sorted({f["intermediate"] for f in stats}),
                  loop_counts=sorted({f["loops"] for f in stats}))
    return report


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
