"""Cargo-specific minimum-station delivery to compatible colored terminal docks.

Structural roles: 0 background, 5 wire, 9 launch frame / square junctions.
All destination mappings and the direction of alignment are visible in the input.
"""
from collections import deque
import heapq
import json
import numpy as np

_FOUR = ((-1, 0), (0, -1), (0, 1), (1, 0))
_EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
_COLORS = (1, 2, 3, 4, 6, 7, 8)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer) and np.all((g >= 0) & (g <= 9)),
             "grid must contain integer colors 0..9")
    return g.astype(np.int8, copy=True)


def _components(points, neighbors=_FOUR):
    remaining = set(points)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], set()
        while stack:
            r, c = stack.pop()
            cells.add((r, c))
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _points(mask):
    return set(map(tuple, np.argwhere(mask).tolist()))


def _box(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _rectangle(box):
    t, b, l, r = box
    return {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)}


def _perimeter(box):
    t, b, l, r = box
    return {(y, x) for y, x in _rectangle(box) if y in (t, b) or x in (l, r)}


def _distances(adjacency, start, weighted=False):
    distance = [float("inf")] * len(adjacency)
    distance[start] = 0
    queue = [(0, start)]
    while queue:
        d, a = heapq.heappop(queue)
        if d != distance[a]:
            continue
        for b, length in adjacency[a]:
            nd = d + (length if weighted else 1)
            if nd < distance[b]:
                distance[b] = nd
                heapq.heappush(queue, (nd, b))
    return distance


def _scene(input_grid):
    """Recover the drawing, without requiring that the launch bay be occupied."""
    original = _grid(input_grid)
    nine = _components(_points(original == 9))
    frames = [c for c in nine if min(_box(c)[1] - _box(c)[0] + 1,
                                   _box(c)[3] - _box(c)[2] + 1) >= 5
              and c == _perimeter(_box(c))]
    _require(len(frames) == 1, "one rectangular launch frame is required")
    frame_box = _box(frames[0])
    t, b, l, r = frame_box
    exits = []
    for side, points in enumerate(([(t-1, c) for c in range(l, r+1)],
                                   [(y, r+1) for y in range(t, b+1)],
                                   [(b+1, c) for c in range(l, r+1)],
                                   [(y, l-1) for y in range(t, b+1)])):
        if any(0 <= y < original.shape[0] and 0 <= x < original.shape[1]
               and original[y, x] == 5 for y, x in points):
            exits.append(side)
    _require(len(exits) == 1, "launch frame must have one wire exit side")
    # np.rot90 moves N->W->S->E. Canonical delivery points downwards.
    turn = (exits[0] - 2) % 4
    g = np.rot90(original, turn).copy()
    nine = _components(_points(g == 9))
    frame = next(c for c in nine if c == _perimeter(_box(c)) and
                 min(_box(c)[1]-_box(c)[0]+1, _box(c)[3]-_box(c)[2]+1) >= 5)
    t, b, l, r = _box(frame)
    interior = {(y, x) for y in range(t+1, b) for x in range(l+1, r)}
    _require(not any(g[p] in (5, 9) for p in interior), "structural color inside cargo bay")
    stations = [c for c in nine if c != frame]
    sizes = []
    for c in stations:
        y0, y1, x0, x1 = _box(c)
        size = y1-y0+1
        _require(size == x1-x0+1 and 2 <= size <= 4 and c == _rectangle(_box(c)),
                 "transit stations must be solid squares of size 2..4")
        _require(y0 > b, "network must be on the exit side of the launch bay")
        sizes.append(size)
    _require(stations and len(set(sizes)) == 1, "equal-sized transit stations required")
    nodes = [{"kind": "source", "cells": frame, "box": (t, b, l, r)}]
    nodes += [{"kind": "station", "cells": c, "box": _box(c)} for c in stations]
    other = _points((g != 0) & (g != 5) & (g != 9)) - interior
    docks = []
    for c in _components(other):
        y0, y1, x0, x1 = _box(c)
        _require(c == _rectangle(_box(c)) and x1-x0 >= 3 and y0 > b,
                 "receiver must be a filled rectangle beyond the bay")
        route, rail = int(g[y0, x0+1]), int(g[y0, x0])
        _require(route in _COLORS and rail in _COLORS and route != rail,
                 "receiver has invalid route or rail color")
        _require(np.all(g[y0:y1+1, x0] == rail) and
                 np.all(g[y0:y1+1, x1] == rail) and
                 np.all(g[y0:y1+1, x0+1:x1] == route), "receiver rails/interior are not uniform")
        node = {"kind": "dock", "cells": c, "box": _box(c),
                "route": route, "rail": rail, "id": len(nodes)}
        docks.append(node)
        nodes.append(node)
    _require(len(docks) >= 2, "at least two terminal docks required")
    node_at = {p: i for i, node in enumerate(nodes) for p in node["cells"]}
    edge_masks = _components(_points(g == 5))
    adjacency = [[] for _ in nodes]
    edges = []
    for c in edge_masks:
        touching = {node_at[q] for y, x in c for dy, dx in _FOUR
                    if (q := (y+dy, x+dx)) in node_at}
        _require(len(touching) == 2, "each wire component must connect exactly two nodes")
        a, z = sorted(touching)
        _require((a, z) not in edges, "parallel connector components are excluded")
        edges.append((a, z))
        adjacency[a].append((z, len(c)))
        adjacency[z].append((a, len(c)))
    _require(len(adjacency[0]) == 1, "launch bay must be a leaf")
    _require(all(len(adjacency[d["id"]]) == 1 for d in docks), "receivers must be terminal leaves")
    _require(all(1 <= len(adjacency[i]) <= 4 for i, n in enumerate(nodes)
                 if n["kind"] == "station"), "invalid transit degree")
    distances = _distances(adjacency, 0)
    _require(all(np.isfinite(distances)), "disconnected delivery network")
    # Everything outside the bay belongs to a recognized node or wire.
    accounted = set(node_at) | set().union(*edge_masks) | interior
    _require(_points(g != 0) <= accounted, "unrecognized foreground")
    return {"grid": g, "turn": turn, "bay": (t, b, l, r), "interior": interior,
            "nodes": nodes, "docks": docks, "adjacency": adjacency, "edges": edges,
            "edge_masks": edge_masks, "distances": distances,
            "wire_distances": _distances(adjacency, 0, True)}


def _cargo(scene):
    g = scene["grid"]
    result = []
    for color in _COLORS:
        for cells in _components({p for p in scene["interior"] if g[p] == color}):
            result.append({"color": color, "cells": cells, "box": _box(cells)})
    return result


def _fits(comp, dock):
    top, bottom, left, right = comp["box"]
    dt, db, dl, dr = dock["box"]
    return bottom-top <= db-dt and dl < left and right < dr


def _delivery(scene, cargo):
    out = scene["grid"].copy()
    occupied, transfers = set(), []
    for comp in cargo:
        matching = [d for d in scene["docks"] if d["route"] == comp["color"]]
        eligible = [d for d in matching if _fits(comp, d)]
        _require(eligible, "component has no compatible matching receiver")
        best = min(scene["distances"][d["id"]] for d in eligible)
        nearest = [d for d in eligible if scene["distances"][d["id"]] == best]
        _require(len(nearest) == 1, "nearest compatible receiver is tied")
        dock = nearest[0]
        shift = dock["box"][1] - comp["box"][1]
        target = {(y+shift, x) for y, x in comp["cells"]}
        _require(not (target & occupied), "delivered masks collide")
        occupied.update(target)
        for p in comp["cells"]:
            out[p] = 0
        transfers.append({"component": comp, "dock": dock, "matching": matching,
                          "eligible": eligible, "target": target, "shift": shift})
    for transfer in transfers:
        for p in transfer["target"]:
            out[p] = transfer["dock"]["rail"]
    return out, transfers


def _analyze(input_grid):
    scene = _scene(input_grid)
    cargo = _cargo(scene)
    _require(cargo, "launch bay is empty")
    # Avoid undemonstrated four-versus-eight segmentation ambiguity.
    for color in _COLORS:
        pts = {p for c in cargo if c["color"] == color for p in c["cells"]}
        _require(len(_components(pts)) == len(_components(pts, _EIGHT)),
                 "diagonal-only same-color contact is excluded")
    output, transfers = _delivery(scene, cargo)
    scene.update(cargo=cargo, transfers=transfers,
                 output=np.rot90(output, -scene["turn"]).copy())
    return scene


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible pixels only. No seed, training examples or generation state."""
    return _analyze(input_grid)["output"]


def _key(g):
    return g.shape, np.ascontiguousarray(g, dtype=np.int8).tobytes()


def _geometric_cost(a, dock):
    """Squared distance between doubled node centers; exact integer comparison."""
    st, sb, sl, sr = a["bay"]
    dt, db, dl, dr = dock["box"]
    return (dt+db-st-sb)**2 + (dl+dr-sl-sr)**2


def _station_witness(a):
    return any(any(a["distances"][z["id"]] > a["distances"][t["dock"]["id"]] and
                       a["wire_distances"][z["id"]] < a["wire_distances"][t["dock"]["id"]] and
                       _geometric_cost(a, z) < _geometric_cost(a, t["dock"])
                       for z in t["eligible"]) for t in a["transfers"])


def _capacity_witness(a):
    return any(any(not _fits(t["component"], z) and
                       a["distances"][z["id"]] < a["distances"][t["dock"]["id"]]
                       for z in t["matching"]) for t in a["transfers"])


def _split_witness(a):
    g = a["grid"]
    cells = {p for c in a["cargo"] for p in c["cells"]}
    return any((y+dy, x+dx) in cells and g[y, x] != g[y+dy, x+dx]
               for y, x in cells for dy, dx in _FOUR)


def _independence_witness(a):
    return any(x["component"]["color"] == y["component"]["color"] and
               x["component"]["box"][1] != y["component"]["box"][1] and
               x["shift"] != y["shift"]
               for i, x in enumerate(a["transfers"]) for y in a["transfers"][i+1:])


def _raw_graph(rng, mode, reserve=False):
    """Build a network with terminal slots; never call either parent generator."""
    if reserve:
        nr, nc = 4, 3
        root = (0, 2)
        receivers = [(0, 0), (3, 0)] + ([(2, 0)] if mode == "capacity" else [])
        edges = [((0, 2), (0, 1)), ((0, 1), (0, 0))]
        edges += [((r, 2), (r+1, 2)) for r in range(3)]
        edges += [((3, 2), (3, 1)), ((3, 1), (3, 0))]
        edges += [((0, 1), (1, 1))]
        if mode == "capacity":
            edges += [((1, 1), (2, 1)), ((2, 1), (2, 0))]
        active = {v for e in edges for v in e}
        visible = {root, (0, 1), (1, 1)} | set(receivers)
        if mode == "capacity":
            visible.add((2, 1))
        return nr, nc, root, active, edges, visible, receivers
    nr = 4 if mode == "capacity" else int(rng.choice([2, 3, 4], p=[.15, .35, .5]))
    nc = int(rng.choice([3, 4], p=[.7, .3]))
    side = 0 if rng.random() < .5 else nc-1
    rows = list(map(int, rng.choice(nr, 3 if mode == "capacity" else 2, replace=False)))
    receivers = [(r, side) for r in rows]
    core = {(r, c) for r in range(nr) for c in range(nc)} - set(receivers)
    root = (0, int(rng.choice([c for c in range(nc) if (0, c) in core])))
    active, edges = {root}, []
    while active != core:
        choices = sorted((a, (a[0]+dy, a[1]+dx)) for a in active for dy, dx in _FOUR
                         if (a[0]+dy, a[1]+dx) in core-active)
        _require(choices, "disconnected core")
        e = choices[int(rng.integers(len(choices)))]
        edges.append(e)
        active.add(e[1])
    for v in receivers:
        neighbor = (v[0], 1 if side == 0 else nc-2)
        edges.append((v, neighbor))
        active.add(v)
    candidates = sorted((a, b) for a in core for b in core if a < b and
                        abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 and
                        (a, b) not in edges and (b, a) not in edges)
    rng.shuffle(candidates)
    for e in candidates:
        if rng.random() < (.4 if mode == "general" else .18):
            edges.append(e)
    degree = {v: sum(v in e for e in edges) for v in active}
    leaves = [v for v in sorted(core) if v != root and degree[v] == 1 and
              next(b if a == v else a for a, b in edges if v in (a, b))[0] == v[0]]
    rng.shuffle(leaves)
    receivers += leaves[:int(rng.integers(min(2, len(leaves))+1))]
    visible = {v for v in active if v == root or v in receivers or degree[v] != 2 or rng.random() < .45}
    ordinary = sorted(active-set(receivers))
    if len(visible-set(receivers)) < 2:
        visible.add(ordinary[0] if ordinary[0] != root else ordinary[-1])
    return nr, nc, root, active, edges, visible, receivers


def _draw_scene(rng, mode, reserve=False):
    nr, nc, root, active, edges, visible, receiver_slots = _raw_graph(rng, mode, reserve)
    pitch = int(rng.integers(7, 10)) if nc == 3 else 7
    width = min(30, (nc-1)*pitch + int(rng.integers(9, 12)))
    centers = [4+i*pitch for i in range(nc)]
    bay_height = int(rng.integers(7, 9))
    top_margin = int(rng.integers(2))
    first = top_margin + bay_height + int(rng.integers(1, 3))
    row_pitch = 5 if nr == 4 else int(rng.integers(5, 8))
    height = min(30, first + (nr-1)*row_pitch + 4 + int(rng.integers(3)))
    _require(first+(nr-1)*row_pitch+4 <= height, "scene too tall")
    g = np.zeros((height, width), dtype=np.int8)
    bay = (top_margin, top_margin+bay_height-1, 1, width-2)
    for p in _perimeter(bay):
        g[p] = 9
    palette = list(map(int, rng.permutation(_COLORS)))
    route_colors = palette[:3]
    docks, boxes = {}, {}
    for v in sorted(active):
        y, c = first+v[0]*row_pitch, centers[v[1]]
        if v in receiver_slots:
            index = receiver_slots.index(v)
            w = int(rng.integers(5, min(7, pitch-1)+1))
            h = int(rng.integers(1, 5))
            if index < 2:
                h = int(rng.integers(3, 5))
            box = (y, y+h-1, c-w//2, c-w//2+w-1)
            route = route_colors[0] if index < 2 else route_colors[1] if index == 2 else int(rng.choice(route_colors[1:]))
            rail = int(rng.choice([z for z in _COLORS if z != route]))
            docks[v] = {"box": box, "route": route, "rail": rail}
        else:
            box = (y, y+1, c-1, c)
        boxes[v] = box
    # For the capacity lesson, identify the nearer of the two X terminals in the
    # abstract compressed graph, then give it a shallower receiver.
    raw_adj = {v: [] for v in active}
    for a, b in edges:
        raw_adj[a].append(b)
        raw_adj[b].append(a)
    raw_distance = {root: 0}
    todo = deque([root])
    while todo:
        v = todo.popleft()
        for z in raw_adj[v]:
            d = raw_distance[v] + int(z in visible)
            if z not in raw_distance or d < raw_distance[z]:
                raw_distance[z] = d
                todo.append(z)
    if mode == "capacity":
        x, z = receiver_slots[:2]
        _require(raw_distance[x] != raw_distance[z], "capacity receivers tie")
        shallow = min((x, z), key=lambda v: raw_distance[v])
        box = boxes[shallow]
        boxes[shallow] = docks[shallow]["box"] = (box[0], box[0]+int(rng.integers(1, 3))-1, box[2], box[3])
    def port(v, side):
        t, b, l, r = boxes[v]
        return {"N": (t, r), "E": (b, r), "S": (b, l), "W": (t, l)}[side]
    def draw_path(points):
        for (y0, x0), (y1, x1) in zip(points, points[1:]):
            _require(y0 == y1 or x0 == x1, "wire leg must be axis aligned")
            for y in range(min(y0, y1), max(y0, y1)+1):
                for x in range(min(x0, x1), max(x0, x1)+1):
                    _require(0 <= y < height and 0 <= x < width, "wire outside canvas")
                    g[y, x] = 5
    for a, b in edges:
        if a[0] == b[0]:
            a, b = sorted((a, b), key=lambda v: v[1])
            p, q = port(a, "E"), port(b, "W")
            mid = (p[1]+q[1])//2
            draw_path([p, (p[0], mid), (q[0], mid), q])
        else:
            a, b = sorted((a, b), key=lambda v: v[0])
            p, q = port(a, "S"), port(b, "N")
            mid = (p[0]+q[0])//2
            draw_path([p, (mid, p[1]), (mid, q[1]), q])
    # Hidden degree-two lattice vertices are bends, not counted stations.
    for v in active-visible:
        for p in _rectangle(boxes[v]):
            g[p] = 5
    root_port = port(root, "N")
    draw_path([(bay[1], root_port[1]), root_port])
    for p in _perimeter(bay):
        g[p] = 9
    for v in visible:
        box = boxes[v]
        for p in _rectangle(box):
            g[p] = docks[v]["route"] if v in docks else 9
        if v in docks:
            t, b, l, r = box
            g[t:b+1, [l, r]] = docks[v]["rail"]
    scene = _scene(g)
    # Check that the drawing reconstructs the intended compressed network.
    ids = {v: next(i for i, n in enumerate(scene["nodes"]) if n["box"] == boxes[v]) for v in visible}
    expected = {tuple(sorted((0, ids[root])))}
    visits = 0
    for v in sorted(visible):
        for z in raw_adj[v]:
            prev, cur = v, z
            seen = {v}
            while cur not in visible:
                _require(cur not in seen, "suppressed loop without visible stations")
                seen.add(cur)
                choices = [q for q in raw_adj[cur] if q != prev]
                _require(len(choices) == 1, "suppressed branch")
                prev, cur = cur, choices[0]
            _require(cur != v, "self-loop in compressed network")
            expected.add(tuple(sorted((ids[v], ids[cur]))))
            visits += 1
    _require(visits == 2*(len(expected)-1) and set(scene["edges"]) == expected,
             "rendered graph differs from construction")
    if mode == "station":
        ds = [next(d for d in scene["docks"] if d["box"] == docks[v]["box"]) for v in receiver_slots[:2]]
        near = min(ds, key=lambda d: scene["distances"][d["id"]])
        far = next(d for d in ds if d != near)
        _require(scene["distances"][near["id"]] < scene["distances"][far["id"]] and
                 scene["wire_distances"][near["id"]] > scene["wire_distances"][far["id"]] and
                 _geometric_cost(scene, near) > _geometric_cost(scene, far),
                 "no station-versus-wire-distance witness")
    return g, scene, route_colors[0], route_colors[1]


def _normalize(cells):
    t, b, l, r = _box(cells)
    return {(y-t, x-l) for y, x in cells}


def _mask(rng, h, w):
    h, w = int(rng.integers(1, h+1)), int(rng.integers(1, w+1))
    kind = int(rng.integers(6))
    if kind == 0:
        return _rectangle((0, h-1, 0, w-1))
    if kind == 1:
        return {(r, 0) for r in range(h)} if rng.random() < .5 else {(0, c) for c in range(w)}
    if kind == 2 and h >= 3 and w >= 3:
        return _perimeter((0, h-1, 0, w-1))
    if kind == 3:
        cells = {(r, w//2) for r in range(h)}
        for row in range(h):
            if rng.random() < .5:
                cells.update((row, c) for c in range(w))
        return _normalize(cells)
    if kind == 4:
        y = x = 0
        cells = {(y, x)}
        while y < h-1 or x < w-1:
            if y < h-1 and (x == w-1 or rng.random() < .5):
                y += 1
            else:
                x += 1
            cells.add((y, x))
        return cells
    cells = {(int(rng.integers(h)), int(rng.integers(w)))}
    target = int(rng.integers(1, h*w+1))
    while len(cells) < target:
        frontier = sorted({(y+dy, x+dx) for y, x in cells for dy, dx in _FOUR
                           if 0 <= y+dy < h and 0 <= x+dx < w and (y+dy, x+dx) not in cells})
        cells.add(frontier[int(rng.integers(len(frontier)))])
    return _normalize(cells)


def _add_random_cargo(rng, g, scene, preferred=None):
    t, b, l, r = scene["bay"]
    for _ in range(60):
        docks = [d for d in scene["docks"] if preferred is None or d["route"] == preferred]
        dock = docks[int(rng.integers(len(docks)))]
        dt, db, dl, dr = dock["box"]
        mask = _mask(rng, min(4, b-t-1, db-dt+1), dr-dl-1)
        mt, mb, ml, mr = _box(mask)
        left = int(rng.integers(dl+1, dr-mr))
        top = int(rng.integers(t+1, b-mb))
        cells = {(top+y, left+x) for y, x in mask}
        color = dock["route"]
        if any(g[p] != 0 for p in cells):
            continue
        if any(0 <= y+dy < g.shape[0] and 0 <= x+dx < g.shape[1] and
               g[y+dy, x+dx] == color for y, x in cells for dy, dx in _EIGHT):
            continue
        candidate = g.copy()
        for p in cells:
            candidate[p] = color
        try:
            _analyze(candidate)
        except ValueError:
            continue
        g[:] = candidate
        return True
    return False


def _fill_scene(rng, g, scene, xcolor, ycolor, mode):
    t, b, l, r = scene["bay"]
    xdocks = [d for d in scene["docks"] if d["route"] == xcolor]
    lo = max(d["box"][2]+1 for d in xdocks)
    hi = min(d["box"][3]-1 for d in xdocks)
    if mode == "capacity":
        ydock = next(d for d in scene["docks"] if d["route"] == ycolor and
                     max(lo, d["box"][2]+1)+2 <= min(hi, d["box"][3]-1))
        lo = max(lo, ydock["box"][2]+1)
        hi = min(hi, ydock["box"][3]-1)
        frame_width = int(rng.integers(3, hi-lo+2))
        left = int(rng.integers(lo, hi-frame_width+2))
        low_first = bool(rng.integers(2))
        spare = b-t-1-5
        offset = int(rng.integers(spare+1))
        frame_top = t+1+offset+(2 if low_first else 0)
        small_top = t+1+offset+(0 if low_first else 4)
        for y, c in _perimeter((0, 2, 0, frame_width-1)):
            g[frame_top+y, left+c] = xcolor
        # A varied filled window is touching the surrounding X frame.
        fill_width = int(rng.integers(1, frame_width-1))
        start = int(rng.integers(1, frame_width-fill_width))
        g[frame_top+1, left+start:left+start+fill_width] = ycolor
        small_width = int(rng.integers(1, min(2, hi-lo+1)+1))
        small_left = int(rng.integers(lo, hi-small_width+2))
        g[small_top, small_left:small_left+small_width] = xcolor
    else:
        # A shipment fitting BOTH repeated-color terminals makes station distance
        # an observable choice, even in simple examples.
        h = min(d["box"][1]-d["box"][0]+1 for d in xdocks)
        mask = _mask(rng, min(h, 3), hi-lo+1)
        mt, mb, ml, mr = _box(mask)
        left = int(rng.integers(lo, hi-mr+1))
        top = int(rng.integers(t+1, b-mb))
        for y, c in mask:
            g[top+y, left+c] = xcolor
    analysis = _analyze(g)
    if mode == "station":
        _require(_station_witness(analysis), "station lesson lost")
    if mode == "capacity":
        _require(_capacity_witness(analysis) and _split_witness(analysis) and
                 _independence_witness(analysis), "capacity/splitting lesson lost")
    extras = int(rng.integers(0, 3)) if mode != "general" else int(rng.integers(2, 6))
    for _ in range(extras):
        _add_random_cargo(rng, g, scene)
    return g


def _pair(rng, mode, turn):
    last_error = None
    for attempt in range(121):
        try:
            g, scene, xcolor, ycolor = _draw_scene(rng, mode, reserve=attempt >= 110)
            g = _fill_scene(rng, g, scene, xcolor, ycolor, mode)
            inp = np.rot90(g, turn).copy()
            return {"input": inp, "output": solve(inp)}
        except (ValueError, StopIteration) as error:
            last_error = error
    raise RuntimeError(f"bounded construction failed for {mode}: {last_error}")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, "num_train", 2, 10)
    ne = _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    turns = list(map(int, rng.permutation(4)))
    # Two observed directions establish rotation covariance; tests hold out the
    # other two. The exit side visibly supplies the direction in every query.
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            mode = ("station" if i == 0 else "capacity" if i == 1 else "general") if split == "train" else "general"
            turn = turns[i % 2] if split == "train" else turns[2+i % 2]
            for _ in range(12):
                pair = _pair(rng, mode, turn)
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input reserve exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, ne = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, "num_train", 2, 10)
    _count(ne, "num_test", 1, 4)
    _require(nt > ne, "illegal train/test counts")
    seen, analyses = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "shape must be preserved")
            key = _key(inp)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            a = _analyze(inp)
            _require(np.array_equal(out, a["output"]), "incorrect delivery output")
            _require(not np.array_equal(inp, out), "identity pair")
            canonical_out = np.rot90(out, a["turn"])
            _require(np.all(canonical_out[a["grid"] == 5] == 5) and
                     np.all(canonical_out[a["grid"] == 9] == 9), "network was altered")
            _require(all(canonical_out[p] == 0 for p in a["interior"]), "cargo bay not cleared")
            for transfer in a["transfers"]:
                comp, dock = transfer["component"], transfer["dock"]
                _require(_normalize(transfer["target"]) == _normalize(comp["cells"]), "mask changed")
                _require(max(y for y, x in transfer["target"]) == dock["box"][1], "wrong far-edge alignment")
                _require({x for y, x in transfer["target"]} == {x for y, x in comp["cells"]}, "lateral coordinates changed")
                _require(all(canonical_out[p] == dock["rail"] for p in transfer["target"]), "wrong rail color")
            analyses[split].append(a)
    tr = analyses["train"]
    _require(any(_station_witness(a) for a in tr), "training lacks station-versus-wire-distance witness")
    _require(any(_capacity_witness(a) for a in tr), "training lacks incompatible nearer dock witness")
    _require(any(_split_witness(a) for a in tr), "training lacks multicolor splitting witness")
    _require(any(_independence_witness(a) for a in tr), "training lacks independent same-color alignment witness")
    # Required for this generator's deliberate rotation holdout, not a solve-time
    # assumption about what directions a valid individual scene may contain.
    if any(a["turn"] not in {z["turn"] for z in tr} for a in analyses["test"]):
        _require(len({a["turn"] for a in tr}) >= 2, "held-out direction needs two training directions")


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ("input", "output")) for s in ("train", "test"))


def self_test():
    """Bounded structural, semantic, omission and diversity checks on 42 seeds."""
    # Hand-specified pixels and expected placements check the intended rule
    # separately from construction. Small X fits two docks; tall X must bypass
    # the closer shallow one; its touching Y center has its own destination.
    hand = np.zeros((20, 18), dtype=np.int8)
    for p in _perimeter((0, 6, 1, 16)):
        hand[p] = 9
    for box in ((9, 10, 12, 13), (12, 13, 12, 13), (16, 17, 12, 13)):
        for p in _rectangle(box):
            hand[p] = 9
    hand[7:9, 13] = 5
    hand[11, 12:14] = 5
    hand[14:16, 13] = 5
    hand[14, 12] = 5
    hand[9, 9:12] = 5
    hand[10, 7:10] = 5
    hand[9:11, 9] = 5
    hand[12, 7:12] = 5
    hand[16, 9:12] = 5
    hand[17, 7:10] = 5
    hand[16:18, 9] = 5
    for box, route, rail in (((9, 10, 2, 6), 1, 4),
                             ((12, 12, 2, 6), 2, 3),
                             ((15, 17, 2, 6), 1, 6)):
        t, b, l, r = box
        hand[t:b+1, l:r+1] = route
        hand[t:b+1, [l, r]] = rail
    for p in _perimeter((1, 3, 3, 5)):
        hand[p] = 1
    hand[2, 4] = 2
    hand[5, 3] = 1
    expected = hand.copy()
    expected[1:6, 2:16] = 0
    for p in _perimeter((15, 17, 3, 5)):
        expected[p] = 6
    expected[12, 4] = 3
    expected[10, 3] = 4
    for turn in range(4):
        _require(np.array_equal(solve(np.rot90(hand, turn)), np.rot90(expected, turn)),
                 "hand-specified delivery regression failed")
    records, shapes, masks, graphs, first_inputs = [], set(), set(), set(), set()
    roles, sizes, docking_counts, component_counts, directions = set(), set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        if seed < 2:
            _require(_equal(p, generate_puzzle(seed)), "nondeterministic puzzle")
        first_inputs.add(_key(p["train"][0]["input"]))
        for split in ("train", "test"):
            for pair in p[split]:
                a = _analyze(pair["input"])
                _require(np.array_equal(solve(pair["input"]), pair["output"]), "solve mismatch")
                shapes.add(pair["input"].shape)
                directions.add(a["turn"])
                docking_counts.add(len(a["docks"]))
                component_counts.add(len(a["cargo"]))
                n = len(a["nodes"])
                # Floyd-Warshall confirms distances on recovered graphs; this is
                # a graph algorithm check, not an independent ARC solver.
                matrix = np.full((n, n), 10000, dtype=int)
                np.fill_diagonal(matrix, 0)
                for x, z in a["edges"]:
                    matrix[x, z] = matrix[z, x] = 1
                for k in range(n):
                    matrix = np.minimum(matrix, matrix[:, k:k+1]+matrix[k:k+1, :])
                _require(list(matrix[0]) == a["distances"], "incorrect station distances")
                degrees = tuple(sorted(len(q) for q in a["adjacency"]))
                loop_rank = len(a["edges"])-n+1
                graphs.add((n, degrees, loop_rank, tuple(sorted(a["distances"][d["id"]] for d in a["docks"]))))
                records.append((split, loop_rank, max(degrees), _station_witness(a), _capacity_witness(a)))
                for transfer in a["transfers"]:
                    c, d = transfer["component"], transfer["dock"]
                    masks.add(tuple(sorted(_normalize(c["cells"]))))
                    roles.add((c["color"], d["rail"]))
                    sizes.add((d["box"][1]-d["box"][0]+1, d["box"][3]-d["box"][2]+1))
    _require(len(first_inputs) == 42, "cross-seed duplicate puzzles")
    _require(len(shapes) >= 30 and len(graphs) >= 30 and len(masks) >= 30, "narrow structural variation")
    _require(len(roles) >= 20 and len(sizes) >= 8, "narrow palette/capacity variation")
    _require(directions == {0, 1, 2, 3} and len(docking_counts) >= 3 and len(component_counts) >= 4,
             "narrow orientations or counts")
    _require(any(r[1] >= 2 for r in records) and any(r[2] == 4 for r in records), "loops or four-way junctions absent")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+10*nt+ne, nt, ne)
            _require((len(p["train"]), len(p["test"])) == (nt, ne), "counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # All finite reserve paths must actually construct; they are not an exception
    # to the semantic or learning requirements.
    rng = np.random.default_rng(909)
    for mode in ("station", "capacity", "general"):
        for turn in range(4):
            g, a, x, y = _draw_scene(rng, mode, reserve=True)
            g = _fill_scene(rng, g, a, x, y, mode)
            canonical = solve(g)
            _require(np.array_equal(solve(np.rot90(g, turn)), np.rot90(canonical, turn)), "rotation covariance failed")
    # Wrong labels, same-distance receiver ties and corrupted rail conventions
    # must be caught. Ties use an explicit graph distance regression below.
    p = generate_puzzle(501, 2, 1)
    p["test"][0]["output"][0, 0] = (int(p["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    g, a, x, y = _draw_scene(np.random.default_rng(90), "general", reserve=True)
    g = _fill_scene(np.random.default_rng(91), g, a, x, y, "general")
    a = _analyze(g)
    comp = next(c for c in a["cargo"] if c["color"] == x and
                sum(d["route"] == x and _fits(c, d) for d in a["docks"]) == 2)
    ds = [d for d in a["docks"] if d["route"] == x and _fits(comp, d)]
    _require(len(ds) == 2, "tie test needs two candidates")
    a["distances"][ds[1]["id"]] = a["distances"][ds[0]["id"]]
    try:
        _delivery(a, [comp])
    except ValueError:
        pass
    else:
        raise AssertionError("receiver tie accepted")
    return {"seeds": 42, "pairs": len(records), "grid_shapes": len(shapes),
            "graph_signatures": len(graphs), "cargo_masks": len(masks),
            "route_rail_pairs": len(roles), "dock_sizes": len(sizes),
            "dock_counts": sorted(docking_counts), "cargo_counts": sorted(component_counts),
            "directions": sorted(directions), "max_cycle_rank": max(r[1] for r in records),
            "semantic_checks": "generator-owned; no independent or blind solver"}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
