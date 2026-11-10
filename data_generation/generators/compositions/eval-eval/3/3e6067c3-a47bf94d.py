"""Itineraries through a woven station network. No file or network I/O.

Construction compresses a coarse drawing; solving traces returned input pixels.
The key selects a unique simple station path, and each departure colors its wire.
"""
from hashlib import sha256
import numpy as np


_D = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "invalid cells")
    return a.astype(np.int8, copy=False)


def _components(cells):
    unseen = set(cells)
    found = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        stack, part = [p], {p}
        while stack:
            y, x = stack.pop()
            for dy, dx in _D:
                q = (y + dy, x + dx)
                if q in unseen:
                    unseen.remove(q)
                    part.add(q)
                    stack.append(q)
        found.append(part)
    return found


def _rectangles(g, color):
    rects = []
    for part in _components(map(tuple, np.argwhere(g == color))):
        ys, xs = zip(*part)
        y0, y1, x0, x1 = min(ys), max(ys), min(xs), max(xs)
        _require(y1 > y0 and x1 > x0 and len(part) == (y1-y0+1)*(x1-x0+1), "invalid mask")
        rects.append((y0, y1, x0, x1))
    return rects


def _trace_pixels(g, stations, wire, bridge, masks):
    """Recover edges with a directional graph; a bridge never acts as a junction."""
    visible = set(map(tuple, np.argwhere(g == wire)))
    if bridge is not None:
        visible.update(map(tuple, np.argwhere(g == bridge)))
    graph = {p: {} for p in visible}
    for p in sorted(visible):
        y, x = p
        for d, (dy, dx) in enumerate(_D):
            q = (y+dy, x+dx)
            if q in visible:
                graph[p][d] = (q, (d+2) % 4, None)
    starts = [[] for _ in stations]
    for i, (y, x, _) in enumerate(stations):
        for d, (dy, dx) in enumerate(_D):
            p = (y+2*dy, x+2*dx)
            if p in visible:
                inward = (d+2) % 4
                _require(inward not in graph[p], "station port overlaps a wire")
                graph[p][inward] = (i, -1, None)
                starts[i].append((p, inward))
        _require(1 <= len(starts[i]) <= 4, "unconnected station")
    mask_links = []
    for mi, (y0, y1, x0, x1) in enumerate(masks):
        perimeter = []
        for x in range(x0, x1+1):
            perimeter += [((y0-1, x), 2), ((y1+1, x), 0)]
        for y in range(y0, y1+1):
            perimeter += [((y, x0-1), 1), ((y, x1+1), 3)]
        # An odd degree needs an edge; a parallel wire merely beside a mask does not.
        ports = {(p, d) for p, d in perimeter if p in visible and len(graph[p]) % 2 == 1}
        _require(len(ports) >= 2 and len(ports) % 2 == 0, "invalid mask ports")
        links = []
        while ports:
            p, d = min(ports)
            ports.remove((p, d))
            candidates = [(q, e) for q, e in ports if e == (d+2) % 4
                          and q[1 if d % 2 == 0 else 0] == p[1 if d % 2 == 0 else 0]]
            _require(len(candidates) == 1, "mask must have opposite aligned ports")
            q, e = candidates[0]
            ports.remove((q, e))
            _require(d not in graph[p] and e not in graph[q], "occupied mask port")
            graph[p][d] = (q, e, mi)
            graph[q][e] = (p, d, mi)
            links.append((p, q))
        mask_links.append(links)
    for p, links in graph.items():
        if bridge is not None and g[p] == bridge:
            _require(set(links) == {0, 1, 2, 3}, "bridge must have four arms")
        else:
            _require(len(links) == 2, "wire branches or breaks outside a station")
    edges, endpoints, used = [], {}, set()
    for a in range(len(stations)):
        for start, incoming in starts[a]:
            p, cells, hidden, states = start, [], set(), set()
            while True:
                _require((p, incoming) not in states, "closed stationless cable")
                states.add((p, incoming))
                cells.append(p)
                used.add(p)
                links = graph[p]
                if len(links) == 4:
                    outgoing = (incoming+2) % 4
                else:
                    choices = [d for d in links if d != incoming]
                    _require(len(choices) == 1, "ambiguous cable")
                    outgoing = choices[0]
                q, incoming, mask = links[outgoing]
                if mask is not None:
                    hidden.add(mask)
                if isinstance(q, int):
                    b = q
                    break
                p = q
            _require(a != b, "self loop")
            key = tuple(sorted((a, b)))
            if key in endpoints:
                old = edges[endpoints[key]]
                _require(set(old["cells"]) == set(cells), "parallel station edges are excluded")
            else:
                endpoints[key] = len(edges)
                edges.append(dict(a=a, b=b, cells=tuple(cells), masks=tuple(sorted(hidden))))
    _require(used == visible, "unattached cable component")
    return edges, mask_links


def _parse(input_grid):
    g = _grid(input_grid)
    bg = int(g[0, 0])
    _require(all(np.all(e == bg) for e in (g[0], g[-1], g[:, 0], g[:, -1])), "background frame required")
    candidates = []
    for ring in sorted(set(map(int, np.unique(g))) - {bg}):
        stations, covered = [], np.zeros(g.shape, dtype=bool)
        for y, x in map(tuple, np.argwhere(g == ring)):
            if y+2 >= g.shape[0] or x+2 >= g.shape[1]:
                continue
            block = g[y:y+3, x:x+3]
            border = block.copy()
            border[1, 1] = ring
            if np.all(border == ring) and int(block[1, 1]) not in (bg, ring):
                stations.append((int(y+1), int(x+1), int(block[1, 1])))
                covered[y:y+3, x:x+3] = True
        if len(stations) >= 4 and np.all(covered[g == ring]):
            candidates.append((ring, stations, covered))
    _require(len(candidates) == 1, "station rings must be uniquely recognizable")
    ring, stations, footprint = candidates[0]
    last = max(y for y, x, c in stations) + 1
    ky = int(np.argwhere(g != bg)[:, 0].max())
    _require(ky >= last+2 and np.all(g[ky-1] == bg), "key must be below a blank separating row")
    key_cells = [(ky, int(x)) for x in np.flatnonzero(g[ky] != bg)]
    _require(3 <= len(key_cells) <= 10, "key length must be 3..10")
    key_cells.sort(key=lambda p: p[1])
    ky = key_cells[0][0]
    _require(all(y == ky for y, x in key_cells) and
             all(b[1]-a[1] == 2 for a, b in zip(key_cells, key_cells[1:])), "invalid detached key")
    key = [int(g[p]) for p in key_cells]
    labels = {s[2] for s in stations}
    _require(set(key) <= labels and len(set(key)) >= 2, "key labels missing from stations")
    key_mask = np.zeros(g.shape, dtype=bool)
    for p in key_cells:
        key_mask[p] = True
    outside = ~footprint & ~key_mask
    _require(not np.any(np.isin(g[outside], list(labels))), "label occurs outside stations/key")
    structural = sorted(set(map(int, np.unique(g[outside]))) - {bg, ring})
    wire_colors = set()
    for y, x, c in stations:
        for dy, dx in _D:
            color = int(g[y+2*dy, x+2*dx])
            if color != bg:
                wire_colors.add(color)
    _require(len(wire_colors) == 1, "station ports must share a wire color")
    wire = wire_colors.pop()
    _require(wire in structural and wire not in labels, "invalid wire role")
    mask, bridge, rects = None, None, []
    for color in structural:
        if color == wire:
            continue
        parts = _components(map(tuple, np.argwhere(g == color)))
        if all(len(p) == 1 for p in parts):
            _require(bridge is None, "multiple bridge colors")
            bridge = color
        else:
            _require(mask is None, "multiple mask colors")
            mask, rects = color, _rectangles(g, color)
    _require(len(structural) <= 3, "extra structural colors")
    _require(all(y1 < ky-1 for y0, y1, x0, x1 in rects), "mask touches key")
    edges, links = _trace_pixels(g, stations, wire, bridge, rects)
    return dict(grid=g, background=bg, ring=ring, wire=wire, bridge=bridge, mask=mask,
                stations=stations, footprint=footprint, key=key, key_cells=key_cells,
                edges=edges, rectangles=rects, mask_links=links)


def _routes(stations, edges, key, limit=2):
    adjacency = [[] for _ in stations]
    for ei, e in enumerate(edges):
        adjacency[e["a"]].append((e["b"], ei))
        adjacency[e["b"]].append((e["a"], ei))
    answers = []
    def visit(path, route_edges):
        if len(answers) >= limit:
            return
        if len(path) == len(key):
            answers.append((tuple(path), tuple(route_edges)))
            return
        for j, ei in adjacency[path[-1]]:
            if j not in path and stations[j][2] == key[len(path)]:
                visit(path+[j], route_edges+[ei])
    for i, s in enumerate(stations):
        if s[2] == key[0]:
            visit([i], [])
    return answers


def _interpret(input_grid):
    info = _parse(input_grid)
    routes = _routes(info["stations"], info["edges"], info["key"])
    _require(len(routes) == 1, "key must identify one simple station itinerary")
    info["route"], info["route_edges"] = routes[0]
    out = info["grid"].copy()
    for step, ei in enumerate(info["route_edges"]):
        for p in info["edges"][ei]["cells"]:
            if info["grid"][p] == info["wire"]:
                out[p] = info["key"][step]
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """All conventions, topology and labels are recovered from this input alone."""
    return _interpret(input_grid)[0]


def _coordinates(rng, n, limit, roomy=False):
    gaps = np.full(n-1, 4, dtype=int)
    gaps[int(rng.integers(n-1))] = 7
    budget = limit - 2 - int(gaps.sum())
    _require(budget >= 0, "coordinate budget")
    for i in rng.permutation(n-1):
        extra = int(rng.integers(min(3 if roomy else 2, budget)+1))
        gaps[i] += extra
        budget -= extra
    offset = int(rng.integers(min(2, budget)+1))
    return [2+offset] + [int(2+offset+x) for x in np.cumsum(gaps)]


def _network(rng, small=False):
    """Random spanning mesh plus extra edges, then distinguish stations/crossings."""
    nr = int(rng.integers(3, 5 if small else 6))
    nc = int(rng.integers(3, 5 if small else 6))
    ys, xs = _coordinates(rng, nr, 24), _coordinates(rng, nc, 27)
    nodes = [(r, c) for r in range(nr) for c in range(nc)]
    neighbors = {p: [(p[0]+dr, p[1]+dc) for dr, dc in _D
                     if 0 <= p[0]+dr < nr and 0 <= p[1]+dc < nc] for p in nodes}
    selected, seen = set(), {nodes[int(rng.integers(len(nodes)))]}
    while len(seen) < len(nodes):
        frontier = [(p, q) for p in sorted(seen) for q in neighbors[p] if q not in seen]
        p, q = frontier[int(rng.integers(len(frontier)))]
        selected.add(tuple(sorted((p, q))))
        seen.add(q)
    density = float(rng.uniform(.15, .85))
    for p in nodes:
        for q in neighbors[p]:
            if p < q and rng.random() < density:
                selected.add((p, q))
    crossing = (int(rng.integers(1, nr-1)), int(rng.integers(1, nc-1)))
    for q in neighbors[crossing]:
        selected.add(tuple(sorted((crossing, q))))
    # A corner conduit gives a genuine bent connection, not an aligned gap.
    for q in neighbors[(0, 0)]:
        selected.add(tuple(sorted(((0, 0), q))))
    adj = {p: [] for p in nodes}
    for a, b in sorted(selected):
        adj[a].append(b)
        adj[b].append(a)
    forced_stations = set(neighbors[crossing]) | {(0, 1), (1, 0)}
    station_nodes, bridges = [], set()
    for p in nodes:
        degree = len(adj[p])
        if p == crossing:
            bridges.add(p)
        elif p == (0, 0):
            continue
        elif p in forced_stations or degree in (1, 3) or rng.random() < .60:
            station_nodes.append(p)
        elif degree == 4:
            bridges.add(p)
    stations = [(ys[r], xs[c], 5) for r, c in station_nodes]
    ids = {p: i for i, p in enumerate(station_nodes)}
    edges, recorded, used = [], set(), set()
    for a in station_nodes:
        for first in sorted(adj[a]):
            previous, p, path = a, first, [a, first]
            trace_seen = set()
            while p not in ids:
                _require((previous, p) not in trace_seen, "coarse cable cycle")
                trace_seen.add((previous, p))
                if p in bridges:
                    q = (2*p[0]-previous[0], 2*p[1]-previous[1])
                    _require(q in adj[p], "broken coarse crossing")
                else:
                    choices = [q for q in adj[p] if q != previous]
                    _require(len(choices) == 1, "coarse conduit branches")
                    q = choices[0]
                path.append(q)
                previous, p = p, q
            for u, v in zip(path, path[1:]):
                used.add(tuple(sorted((u, v))))
            b = p
            _require(a != b, "coarse self loop")
            if ids[a] > ids[b]:
                continue
            endpoint = (ids[a], ids[b])
            _require(endpoint not in recorded, "coarse parallel edge")
            recorded.add(endpoint)
            pixels = []
            for u, v in zip(path, path[1:]):
                y, x, yy, xx = ys[u[0]], xs[u[1]], ys[v[0]], xs[v[1]]
                dy, dx = int(np.sign(yy-y)), int(np.sign(xx-x))
                pixels.extend((y+k*dy, x+k*dx) for k in range(abs(yy-y)+abs(xx-x)))
            pixels.append((ys[b[0]], xs[b[1]]))
            cells = tuple(pixels[2:-2])
            edges.append(dict(a=ids[a], b=ids[b], cells=cells,
                              bridge=any(q in bridges for q in path),
                              bent=a[0] != b[0] and a[1] != b[1]))
    _require(used == selected, "unused stationless loop")
    h = ys[-1] + int(rng.integers(5, 7))
    w = xs[-1] + int(rng.integers(3, min(5, 30-xs[-1])+1))
    g = np.zeros((h, w), dtype=np.int8)
    for e in edges:
        for p in e["cells"]:
            g[p] = 2
    for r, c in bridges:
        g[ys[r], xs[c]] = 3
    for y, x, color in stations:
        g[y-1:y+2, x-1:x+2] = 1
        g[y, x] = color
    return g, stations, edges


def _sample_path(rng, stations, edges, length, required):
    adj = [[] for _ in stations]
    for ei, e in enumerate(edges):
        adj[e["a"]].append((e["b"], ei))
        adj[e["b"]].append((e["a"], ei))
    budget = 12000
    def visit(path, links):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            if all(any(edges[i][flag] for i in links) for flag in required):
                return path, links
            return None
        for index in rng.permutation(len(adj[path[-1]])):
            j, ei = adj[path[-1]][int(index)]
            if j not in path:
                result = visit(path+[j], links+[ei])
                if result is not None:
                    return result
        return None
    for i in rng.permutation(len(stations)):
        result = visit([int(i)], [])
        if result is not None:
            return result
        if budget < 0:
            break
    return None


def _assign_labels(rng, stations, edges, route, repeat):
    n = len(route)
    for attempt in range(100):
        k = min(n, int(rng.integers(3, 5)))
        if repeat:
            k = min(k, n-1)
        elif n <= 4:
            k = n
        pool = list(map(int, rng.permutation(np.arange(5, 10))[:k]))
        key = pool + [int(rng.choice(pool)) for _ in range(n-k)]
        rng.shuffle(key)
        if repeat and len(set(key)) == len(key):
            continue
        if not any(a != b for a, b in zip(key, key[1:])):
            continue
        outside = [c for c in range(5, 10) if c not in key]
        labels = [int(rng.choice(outside if attempt >= 35 else list(range(5, 10)))) for _ in stations]
        for i, c in zip(route, key):
            labels[i] = c
        labeled = [(y, x, c) for (y, x, _), c in zip(stations, labels)]
        solutions = _routes(labeled, edges, key)
        if len(solutions) == 1 and solutions[0][0] == tuple(route):
            return labeled, key
    return None


def _inside(p, rect):
    y0, y1, x0, x1 = rect
    return y0 <= p[0] <= y1 and x0 <= p[1] <= x1


def _add_masks(rng, g, stations, edges, chosen, wanted):
    rects = []
    selected_cells = [p for ei in chosen for p in edges[ei]["cells"] if g[p] == 2]
    all_cells = [p for e in edges for p in e["cells"] if g[p] == 2]
    selected_hidden = False
    for attempt in range(160):
        if len(rects) >= wanted:
            break
        points = selected_cells if not selected_hidden else all_cells
        y, x = points[int(rng.integers(len(points)))]
        hh, ww = int(rng.integers(2, 5)), int(rng.integers(2, 7))
        y0, x0 = y-int(rng.integers(hh)), x-int(rng.integers(ww))
        rect = (y0, y0+hh-1, x0, x0+ww-1)
        a, b, c, d = rect
        if a < 1 or b >= g.shape[0]-3 or c < 1 or d >= g.shape[1]-1:
            continue
        if any(not (b+1 < yy-1 or yy+1 < a-1 or d+1 < xx-1 or xx+1 < c-1)
               for yy, xx, label in stations):
            continue
        if any(not (b+1 < p or q+1 < a or d+1 < r or s+1 < c) for p, q, r, s in rects):
            continue
        links, hidden_edges, valid = [], set(), True
        for ei, e in enumerate(edges):
            entry = None
            cells = e["cells"]
            if _inside(cells[0], rect) or _inside(cells[-1], rect):
                valid = False
                break
            for p, q in zip(cells, cells[1:]):
                if not _inside(p, rect) and _inside(q, rect):
                    entry = p
                elif _inside(p, rect) and not _inside(q, rect):
                    if entry is None or not (entry[0] == q[0] or entry[1] == q[1]):
                        valid = False
                        break
                    # Opposite faces, rather than adjacent-face hidden elbows.
                    if not ((entry[0] == a-1 and q[0] == b+1) or
                            (q[0] == a-1 and entry[0] == b+1) or
                            (entry[1] == c-1 and q[1] == d+1) or
                            (q[1] == c-1 and entry[1] == d+1)):
                        valid = False
                        break
                    links.append((entry, q))
                    hidden_edges.add(ei)
                    entry = None
            if not valid:
                break
        if not valid or not links or (not selected_hidden and not hidden_edges.intersection(chosen)):
            continue
        candidate = g.copy()
        candidate[a:b+1, c:d+1] = 4
        try:
            decoded, _ = _trace_pixels(candidate, stations, 2, 3, rects+[rect])
            actual = {tuple(sorted((e["a"], e["b"]))): set(e["cells"]) for e in decoded}
            for e in edges:
                expected = {p for p in e["cells"] if candidate[p] != 4}
                _require(actual[tuple(sorted((e["a"], e["b"])))] == expected, "mask changed topology")
        except (ValueError, KeyError):
            continue
        g = candidate
        rects.append(rect)
        selected_hidden |= bool(hidden_edges.intersection(chosen))
    return g, rects, selected_hidden


def _make_pair(rng, test, lesson, palette):
    for attempt in range(120):
        try:
            g, stations, edges = _network(rng, small=not test)
        except ValueError:
            continue
        length = int(rng.integers(6, 10)) if test else (4 if lesson == 0 else 5 if lesson == 1 else int(rng.integers(3, 6)))
        required = ("bridge",) if lesson == 0 else ("bent",) if lesson == 1 else ()
        path = _sample_path(rng, stations, edges, length, required)
        if path is None:
            continue
        route, chosen = path
        assigned = _assign_labels(rng, stations, edges, route, repeat=lesson == 1 or (test and rng.random() < .7))
        if assigned is None:
            continue
        stations, key = assigned
        wanted = 1 if lesson == 1 else int(rng.integers(0, 4 if test else 3))
        if wanted:
            g, rects, selected_hidden = _add_masks(rng, g, stations, edges, chosen, wanted)
            if lesson == 1 and not selected_hidden:
                continue
        for y, x, label in stations:
            g[y, x] = label
        ky = g.shape[0]-2
        kw = 2*len(key)-1
        if kw+2 > g.shape[1]:
            g = np.pad(g, ((0, 0), (0, kw+2-g.shape[1])))
        kx = int(rng.integers(1, g.shape[1]-kw))
        g[ky, kx:kx+kw:2] = key
        out = g.copy()
        for step, ei in enumerate(chosen):
            for p in edges[ei]["cells"]:
                if g[p] == 2:
                    out[p] = key[step]
        # The first lesson leaves a visible crossing, exposing strand independence.
        if lesson == 0:
            selected_bridges = {p for ei in chosen for p in edges[ei]["cells"] if g[p] == 3}
            if not any(any(p in e["cells"] for ei, e in enumerate(edges) if ei not in chosen)
                       for p in selected_bridges):
                continue
        lut = np.asarray(palette, dtype=np.int8)
        return {"input": lut[g], "output": lut[out]}
    raise RuntimeError("bounded construction exhausted")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    varied_palette = bool(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            lesson = index if split == "train" and index < 2 else -1
            for retry in range(32):
                colors = list(map(int, rng.permutation(10))) if varied_palette else palette
                pair = _make_pair(rng, split == "test", lesson, colors)
                sig = (pair["input"].shape, pair["input"].tobytes())
                if sig not in seen:
                    break
            else:
                raise RuntimeError("distinct input retry bound exhausted")
            seen.add(sig)
            puzzle[split].append(pair)
    return puzzle


def _features(info):
    g, route_edges = info["grid"], info["route_edges"]
    active = set(route_edges)
    selected_cells = {p for i in active for p in info["edges"][i]["cells"]}
    bent = 0
    for i in active:
        e = info["edges"][i]
        a, b = info["stations"][e["a"]], info["stations"][e["b"]]
        bent += a[0] != b[0] and a[1] != b[1]
    bridges = {p for p in selected_cells if g[p] == info["bridge"]}
    independent_crossing = any(any(p in e["cells"] for i, e in enumerate(info["edges"]) if i not in active)
                               for p in bridges)
    branch = max(sum(e["a"] == i or e["b"] == i for e in info["edges"]) for i in range(len(info["stations"])))
    return dict(shape=tuple(g.shape), stations=len(info["stations"]), edges=len(info["edges"]),
                key_length=len(info["key"]), key_colors=len(set(info["key"])),
                repeated=len(set(info["key"])) < len(info["key"]), bent_selected=bent,
                bridges=int(np.count_nonzero(g == info["bridge"])) if info["bridge"] is not None else 0,
                selected_bridges=len(bridges), independent_crossing=bool(independent_crossing),
                masks=len(info["rectangles"]), multiwire_masks=sum(len(x) > 1 for x in info["mask_links"]),
                selected_masks=len({m for i in active for m in info["edges"][i]["masks"]}),
                unused_edges=len(info["edges"])-len(active), max_degree=branch,
                roles=(info["background"], info["ring"], info["wire"]),
                layout=tuple((y, x) for y, x, c in info["stations"]))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _count(nt, "num_train", 2, 10)
    _count(nq, "num_test", 1, 4)
    _require(nt > nq, "invalid split counts")
    seen, metrics = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            expected, info = _interpret(inp)
            _require(np.array_equal(out, expected), "incorrect output")
            _require(inp.shape == out.shape and np.any(inp != out), "identity or resized answer")
            signature = (inp.shape, inp.tobytes())
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            changed = inp != out
            _require(np.all(inp[changed] == info["wire"]), "nonwire changed")
            selected = set(info["route_edges"])
            for step, ei in enumerate(info["route_edges"]):
                wire_cells = [p for p in info["edges"][ei]["cells"] if inp[p] == info["wire"]]
                _require(wire_cells and all(out[p] == info["key"][step] for p in wire_cells), "source color mismatch")
            for ei, edge in enumerate(info["edges"]):
                if ei not in selected:
                    _require(all(inp[p] == out[p] for p in edge["cells"] if inp[p] == info["wire"]), "inactive cable changed")
            m = _features(info)
            _require(m["unused_edges"] > 0, "no inactive network evidence")
            metrics[split].append(m)
    train = metrics["train"]
    _require(any(m["independent_crossing"] for m in train), "missing straight-through crossing lesson")
    _require(any(m["bent_selected"] for m in train), "missing bent-cable lesson")
    _require(any(m["repeated"] and m["selected_masks"] for m in train), "missing repeated-key/mask lesson")
    _require(max(m["key_length"] for m in train) < min(m["key_length"] for m in metrics["test"]), "missing length shift")


def self_test():
    """42 seeds, bounded aggregates; no samples/logs are saved by this module."""
    summaries, digests, structures = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = sha256()
        for split in ("train", "test"):
            for pair in puzzle[split]:
                out, info = _interpret(pair["input"])
                _require(np.array_equal(out, pair["output"]), "solver disagrees")
                summaries.append(_features(info))
                digest.update(pair["input"].tobytes())
                mapping = {}
                normalized = bytes(mapping.setdefault(int(c), len(mapping)) for c in pair["input"].flat)
                structures.add((pair["input"].shape, sha256(normalized).digest()))
                # Destination-colored and all-cable painting are rejected alternatives.
                wrong = out.copy()
                for step, ei in enumerate(info["route_edges"]):
                    for p in info["edges"][ei]["cells"]:
                        if info["grid"][p] == info["wire"]:
                            wrong[p] = info["key"][step+1]
                _require(not np.array_equal(wrong, out), "destination-color shortcut survived")
                _require(any(info["grid"][p] == out[p] == info["wire"]
                             for i, e in enumerate(info["edges"]) if i not in info["route_edges"] for p in e["cells"]),
                         "paint-all shortcut survived")
        digests.add(digest.digest())
    _require(len(digests) == 42 and len(structures) >= 180, "insufficient structural variation")
    _require(len({m["shape"] for m in summaries}) >= 35, "insufficient shape variation")
    _require(len({m["layout"] for m in summaries}) >= 150, "insufficient layout variation")
    _require(len({m["stations"] for m in summaries}) >= 8, "insufficient station counts")
    _require({m["key_length"] for m in summaries} >= {3, 4, 5, 6, 7, 8, 9}, "missing route lengths")
    _require(any(m["multiwire_masks"] for m in summaries), "no multiwire mask coverage")
    _require(any(m["selected_bridges"] >= 2 for m in summaries), "no multiple crossing route")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(5000+nt*10+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(0), generate_puzzle(0)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x), "nondeterminism")
    # Input-only metamorphisms: palette reassignment and a transposed network
    # with the left-to-right itinerary kept outside that geometric transform.
    inp, out = a["train"][0]["input"], a["train"][0]["output"]
    lut = np.random.default_rng(710).permutation(10).astype(np.int8)
    _require(np.array_equal(solve(lut[inp]), lut[out]), "palette dependence")
    _, info = _interpret(inp)
    ky = info["key_cells"][0][0]
    scene, answer = inp[:ky].T, out[:ky].T
    width = max(scene.shape[1], 2*len(info["key"])+1)
    if scene.shape[0]+3 <= 30 and width <= 30:
        ti = np.full((scene.shape[0]+3, width), info["background"], dtype=np.int8)
        to = ti.copy()
        ti[:scene.shape[0], :scene.shape[1]] = scene
        to[:scene.shape[0], :scene.shape[1]] = answer
        ti[-2, 1:2*len(info["key"]):2] = info["key"]
        to[-2, 1:2*len(info["key"]):2] = info["key"]
        _require(np.array_equal(solve(ti), to), "network orientation dependence")
    reversed_input = inp.copy()
    for p, c in zip(info["key_cells"], reversed(info["key"])):
        reversed_input[p] = c
    _require(not np.array_equal(solve(reversed_input), out), "key order has no effect")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    bad["test"][0]["output"][0, 0] = (int(bad["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        return
    raise AssertionError("nonwire corruption accepted")
