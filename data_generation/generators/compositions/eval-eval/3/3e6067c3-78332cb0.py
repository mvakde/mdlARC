"""Socket itinerary: key-selected routes and exact geometric socket bridges.

Only NumPy and the standard library are used. The solver resegments its input;
generation plans, seeds, and training examples are never solver arguments.
"""
from itertools import combinations

import numpy as np


# Up, right, down, left. Signatures use GLOBAL row/column coordinates.
_OPPOSITE = (2, 3, 0, 1)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        body, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    body.add(q)
                    stack.append(q)
        result.append(body)
    return result


def _parse(grid):
    bg = int(grid[0, 0])
    if not all(np.all(e == bg) for e in
               (grid[0], grid[-1], grid[:, 0], grid[:, -1])):
        raise ValueError("the outer frame must identify the background")
    hubs, tokens, convention = [], [], None
    for component in _components(grid != bg):
        if len(component) == 1:
            tokens.append(next(iter(component)))
            continue
        rr, cc = zip(*component)
        y, x = min(rr), min(cc)
        size = max(rr)-y+1
        if not (4 <= size <= 7 and max(cc)-x+1 == size
                and len(component) == size*size):
            raise ValueError("hubs must be separate, filled, equal square cards")
        block = grid[y:y+size, x:x+size]
        carrier = int(block[0, 0])
        center = (size-1)//2
        width = 1 if size % 2 else 2
        label = int(block[center, center])
        if (carrier == bg or label in (carrier, bg)
                or not np.all(block[center:center+width, center:center+width] == label)
                or not np.all(np.isin(block, (carrier, label)))
                or any(int(block[r, c]) != carrier for r, c in
                       ((0, 0), (0, -1), (-1, 0), (-1, -1)))
                or len(_components(block == label)) != 1):
            raise ValueError("each hub has a carrier corner and one connected label motif")
        current = (size, carrier)
        if convention is not None and convention != current:
            raise ValueError("size and carrier must be shared within an input")
        convention = current
        ports = (
            tuple(x+int(c) for c in np.flatnonzero(block[0] == label)),
            tuple(y+int(r) for r in np.flatnonzero(block[:, -1] == label)),
            tuple(x+int(c) for c in np.flatnonzero(block[-1] == label)),
            tuple(y+int(r) for r in np.flatnonzero(block[:, 0] == label)),
        )
        hubs.append(dict(pos=(y, x), label=label, ports=ports))
    if convention is None or not 3 <= len(hubs) <= 12 or not 3 <= len(tokens) <= 8:
        raise ValueError("expected 3..12 hubs and a 3..8 token itinerary")
    hubs.sort(key=lambda h: h["pos"])
    tokens.sort(key=lambda p: p[1])
    key_y = tokens[0][0]
    size, carrier = convention
    if (any(r != key_y for r, c in tokens)
            or any(b[1]-a[1] != 2 for a, b in zip(tokens, tokens[1:]))
            or key_y < max(h["pos"][0]+size for h in hubs)+1):
        raise ValueError("isolated itinerary tokens must be below all hubs")
    key = tuple(int(grid[p]) for p in tokens)
    if len(set(key)) < 2 or any(c in (bg, carrier) for c in key):
        raise ValueError("the key must contain at least two nonstructural colors")
    return dict(background=bg, carrier=carrier, size=size, hubs=hubs,
                key=key, tokens=tokens)


def _geometry(a, b, size, grid, background):
    """Available straight lanes between facing sides, without using sockets.

    Return (source side, allowed global coordinates, gap start, gap stop).
    Positive gaps and a nonempty overlap of side interiors are required.
    """
    y, x = a
    v, u = b
    r0, r1 = max(y, v)+1, min(y, v)+size-1
    c0, c1 = max(x, u)+1, min(x, u)+size-1
    if r0 < r1 and (x+size < u or u+size < x):
        d = 1 if x < u else 3
        lo, hi = min(x, u)+size, max(x, u)
        coords = tuple(r for r in range(r0, r1)
                       if np.all(grid[r, lo:hi] == background))
    elif c0 < c1 and (y+size < v or v+size < y):
        d = 2 if y < v else 0
        lo, hi = min(y, v)+size, max(y, v)
        coords = tuple(c for c in range(c0, c1)
                       if np.all(grid[lo:hi, c] == background))
    else:
        return None
    return (d, coords, lo, hi) if coords else None


def _graphs(grid, info):
    n, size, bg = len(info["hubs"]), info["size"], info["background"]
    geometric, exact, cardinality = ([[] for _ in range(n)] for _ in range(3))
    edges = {}
    for i, j in combinations(range(n), 2):
        a, b = info["hubs"][i], info["hubs"][j]
        e = _geometry(a["pos"], b["pos"], size, grid, bg)
        if e is None:
            continue
        d, allowed, lo, hi = e
        edges[i, j], edges[j, i] = e, (_OPPOSITE[d], allowed, lo, hi)
        geometric[i].append(j)
        geometric[j].append(i)
        sa, sb = a["ports"][d], b["ports"][_OPPOSITE[d]]
        if sa and len(sa) == len(sb):
            cardinality[i].append(j)
            cardinality[j].append(i)
        if sa and sa == sb and set(sa) <= set(allowed):
            exact[i].append(j)
            exact[j].append(i)
    return geometric, exact, cardinality, edges


def _routes(graph, labels, key, limit=2):
    """Enumerate distinct-hub paths; None tokens omit the color constraint."""
    solutions = []
    def visit(path, used):
        if len(solutions) >= limit:
            return
        if len(path) == len(key):
            solutions.append(tuple(path))
            return
        options = range(len(graph)) if not path else graph[path[-1]]
        token = key[len(path)]
        for j in options:
            if j not in used and (token is None or labels[j] == token):
                visit(path+[j], used | {j})
    visit([], set())
    return solutions


def _cells(edge, signature):
    d, _, lo, hi = edge
    if d in (1, 3):
        return [(r, c) for r in signature for c in range(lo, hi)]
    return [(r, c) for c in signature for r in range(lo, hi)]


def _paint(grid, info, route, edges):
    output, painted = grid.copy(), set()
    for i, j in zip(route, route[1:]):
        e = edges[i, j]
        signature = info["hubs"][i]["ports"][e[0]]
        lane = set(_cells(e, signature))
        if not lane or lane & painted or any(grid[p] != info["background"] for p in lane):
            raise ValueError("selected bridges must be nonempty, clear and nonoverlapping")
        for p in lane:
            output[p] = info["hubs"][i]["label"]
        painted |= lane
    return output, painted


def _inactive_clear(info, route, exact, edges, painted):
    selected = {frozenset((a, b)) for a, b in zip(route, route[1:])}
    for i, row in enumerate(exact):
        for j in row:
            if i >= j or frozenset((i, j)) in selected:
                continue
            e = edges[i, j]
            lane = set(_cells(e, info["hubs"][i]["ports"][e[0]]))
            if lane & painted:
                return False
    return True


def _decode(grid):
    info = _parse(grid)
    geometric, exact, cardinality, edges = _graphs(grid, info)
    labels = [h["label"] for h in info["hubs"]]
    routes = _routes(exact, labels, info["key"])
    if len(routes) != 1:
        raise ValueError("key and complete socket signatures must identify one route")
    route = routes[0]
    output, painted = _paint(grid, info, route, edges)
    if not _inactive_clear(info, route, exact, edges, painted):
        raise ValueError("selected strands must not cross an inactive compatible bridge")
    info.update(route=route, geometric=geometric, exact=exact,
                cardinality=cardinality, edges=edges, painted=painted)
    return output, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the entire answer from the visible input alone."""
    return _decode(_grid(input_grid))[0]


def _kind(signature):
    if len(signature) == 1:
        return "single"
    return "thick" if signature[-1]-signature[0]+1 == len(signature) else "separated"


def _signature(rng, coordinates, required=None):
    pool = [s for k in range(1, min(3, len(coordinates))+1)
            for s in combinations(coordinates, k)
            if required is None or _kind(s) == required]
    if not pool:
        return None
    # Sample categories separately: large sides do not force large ports.
    kinds = sorted({_kind(s) for s in pool})
    kind = required or kinds[int(rng.integers(len(kinds)))]
    pool = [s for s in pool if _kind(s) == kind]
    return pool[int(rng.integers(len(pool)))]


def _base_board(positions, size, length, rng):
    min_h = max(y for y, x in positions)+size+3
    min_w = max(max(x for y, x in positions)+size+1, 2*length+1)
    h = int(rng.integers(min_h, 31))
    w = int(rng.integers(min_w, 31))
    board = np.zeros((h, w), dtype=np.int8)
    for y, x in positions:
        board[y:y+size, x:x+size] = 1
    return board


def _position_graph(positions, size, board):
    edges, graph = {}, [[] for _ in positions]
    for i, j in combinations(range(len(positions)), 2):
        e = _geometry(positions[i], positions[j], size, board, 0)
        if e is not None:
            edges[i, j] = e
            edges[j, i] = (_OPPOSITE[e[0]], *e[1:])
            graph[i].append(j)
            graph[j].append(i)
    return edges, graph


def _sample_path(rng, graph, edges, length, both_axes):
    budget = 1500
    def visit(path):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            axes = {edges[a, b][0] % 2 for a, b in zip(path, path[1:])}
            return path if not both_axes or len(axes) == 2 else None
        for q in rng.permutation(graph[path[-1]]):
            q = int(q)
            if q in path:
                continue
            if len(path) > 1 and edges[path[-1], path[-2]][0] == edges[path[-1], q][0]:
                continue  # Two selected connections cannot share a hub side.
            answer = visit(path+[q])
            if answer is not None:
                return answer
        return None
    for start in rng.permutation(len(graph)):
        answer = visit([int(start)])
        if answer is not None:
            return answer
        if budget < 0:
            break
    return None


def _route_ports(rng, positions, edges, route, required):
    ports = [dict() for _ in positions]
    kinds = list(required)+[None]*(len(route)-1-len(required))
    rng.shuffle(kinds)
    occupied = set()
    for (a, b), kind in zip(zip(route, route[1:]), kinds):
        e = edges[a, b]
        signature = _signature(rng, e[1], kind)
        if signature is None:
            return None
        lane = set(_cells(e, signature))
        if lane & occupied:
            return None
        occupied |= lane
        ports[a][e[0]] = signature
        ports[b][_OPPOSITE[e[0]]] = signature
    return ports


def _extra_ports(rng, positions, size, edges, route, ports):
    route_edges = {frozenset((a, b)) for a, b in zip(route, route[1:])}
    added, target = 0, int(rng.integers(1, 4))
    candidates = [(a, b) for a, b in edges if a < b]
    rng.shuffle(candidates)
    for a, b in candidates:
        if frozenset((a, b)) in route_edges:
            continue
        d, allowed, _, _ = edges[a, b]
        sa, sb = ports[a].get(d), ports[b].get(_OPPOSITE[d])
        if sa is not None and sb is not None:
            continue
        signature = sa or sb or _signature(rng, allowed)
        if not set(signature) <= set(allowed):
            continue
        ports[a][d] = signature
        ports[b][_OPPOSITE[d]] = signature
        added += 1
        if added >= target:
            break
    if not added:
        return False
    # Unmatched sockets are legitimate inactive alternatives, not pixel noise.
    for (y, x), pp in zip(positions, ports):
        for d in range(4):
            if d not in pp and rng.random() < .20:
                origin = x if d in (0, 2) else y
                pp[d] = _signature(rng, tuple(range(origin+1, origin+size-1)))
    return True


def _visible_info(positions, size, ports, labels, key):
    return dict(size=size, background=0, carrier=1, key=tuple(key),
                hubs=[dict(pos=p, label=c, ports=tuple(pp.get(d, ()) for d in range(4)))
                      for p, pp, c in zip(positions, ports, labels)])


def _teaches(board, info, route, count_witness=False, graphs=None):
    geometric, exact, cardinality, edges = graphs if graphs is not None else _graphs(board, info)
    labels, key = [h["label"] for h in info["hubs"]], info["key"]
    if _routes(exact, labels, key) != [tuple(route)]:
        return False
    if len(_routes(geometric, labels, key)) < 2:
        return False
    if len(_routes(exact, labels, [None]*len(key), limit=3)) < 3:
        return False
    if count_witness and len(_routes(cardinality, labels, key)) < 2:
        return False
    used_edges = {frozenset((a, b)) for a, b in zip(route, route[1:])}
    if not any(frozenset((i, j)) not in used_edges for i, row in enumerate(exact) for j in row):
        return False
    try:
        _, painted = _paint(board, info, route, edges)
    except ValueError:
        return False
    if not _inactive_clear(info, route, exact, edges, painted):
        return False
    return True


def _motif(rng, size, pos, ports, richness):
    center = (size-1)//2
    width = 1 if size % 2 else 2
    core = {(r, c) for r in range(center, center+width)
            for c in range(center, center+width)}
    body = set(core)
    y, x = pos
    boundary, roots = set(), []
    for d, signature in sorted(ports.items()):
        for t in signature:
            if d == 0: p, root = (0, t-x), (1, t-x)
            elif d == 1: p, root = (t-y, size-1), (t-y, size-2)
            elif d == 2: p, root = (size-1, t-x), (size-2, t-x)
            else: p, root = (t-y, 0), (t-y, 1)
            boundary.add(p)
            roots.append(root)
    for root in roots:
        target = sorted(core)[int(rng.integers(len(core)))]
        waypoints = [target]
        if rng.random() < .20*richness:
            waypoints.insert(0, tuple(map(int, rng.integers(1, size-1, size=2))))
        r, c = root
        body.add((r, c))
        for tr, tc in waypoints:
            while (r, c) != (tr, tc):
                options = []
                if r != tr: options.append((r+int(np.sign(tr-r)), c))
                if c != tc: options.append((r, c+int(np.sign(tc-c))))
                r, c = options[int(rng.integers(len(options)))]
                body.add((r, c))
    for _ in range(int(rng.integers(1+richness*(size-2)))):
        frontier = sorted({q for r, c in body for q in
                           ((r-1, c), (r+1, c), (r, c-1), (r, c+1))
                           if 0 < q[0] < size-1 and 0 < q[1] < size-1 and q not in body})
        if not frontier:
            break
        body.add(frontier[int(rng.integers(len(frontier)))])
    return body | boundary


def _render(rng, board, positions, size, ports, labels, route, palette, richness):
    mapping = np.asarray(palette, dtype=np.int8)
    grid = mapping[board]
    for pos, pp, label in zip(positions, ports, labels):
        y, x = pos
        for r, c in _motif(rng, size, pos, pp, richness):
            grid[y+r, x+c] = mapping[label]
    key = [int(mapping[labels[i]]) for i in route]
    key_y = int(rng.integers(max(y for y, x in positions)+size+1, grid.shape[0]-1))
    key_x = int(rng.integers(1, grid.shape[1]-(2*len(key)-1)))
    tokens = [(key_y, key_x+2*i) for i in range(len(key))]
    for p, c in zip(tokens, key):
        grid[p] = c
    # Ground truth uses the constructed endpoint signatures; input decoding is
    # performed separately by validation and must recover this very route.
    info = _visible_info(positions, size, ports, [int(mapping[c]) for c in labels], key)
    info["background"] = int(mapping[0])
    _, _, _, edges = _graphs(grid, info)
    output, _ = _paint(grid, info, route, edges)
    return {"input": grid, "output": output}


def _sample_pair(rng, plan, palette):
    length = plan["length"]
    for _ in range(192):
        size = int(rng.choice((4, 5, 5, 6, 7)))
        n = length+int(rng.integers(0, min(3, 12-length)+1))
        layouts = [(r, c) for r in range(1, 5) for c in range(1, 5)
                   if n <= r*c and r*size+(r-1)*2+1 <= 26
                   and c*size+(c-1)*2+1 <= 28]
        if not layouts:
            continue
        nr, nc = layouts[int(rng.integers(len(layouts)))]
        def levels(count, limit):
            gaps = [2]*(count-1)
            spare = limit-(count*size+sum(gaps)+1)
            for j in rng.permutation(len(gaps)):
                extra = int(rng.integers(min(5, spare)+1))
                gaps[int(j)] += extra
                spare -= extra
            return [0]+list(map(int, np.cumsum(np.array(gaps)+size)))
        ys, xs = levels(nr, 26), levels(nc, 28)
        jitter = plan["offset"] or rng.random() < .60
        slots = sorted(map(int, rng.choice(nr*nc, n, replace=False)))
        positions = [(1+ys[k//nc]+(int(rng.integers(2)) if jitter else 0),
                      1+xs[k % nc]+(int(rng.integers(2)) if jitter else 0)) for k in slots]
        board = _base_board(positions, size, length, rng)
        edges, geometric = _position_graph(positions, size, board)
        route = _sample_path(rng, geometric, edges, length, plan["both"])
        if route is None:
            continue
        if plan["offset"] and not any(positions[a][0 if edges[a, b][0] % 2 else 1]
                                       != positions[b][0 if edges[a, b][0] % 2 else 1]
                                       for a, b in zip(route, route[1:])):
            continue
        ports = _route_ports(rng, positions, edges, route, plan["required"])
        if ports is None or not _extra_ports(rng, positions, size, edges, route, ports):
            continue
        # Socket geometry does not depend on the labels currently being sampled.
        # Compute it once instead of repeating pixel-clearance scans 32 times.
        graphs = _graphs(board, _visible_info(positions, size, ports, [2]*n, [2]*length))
        for _ in range(32):
            k = int(rng.integers(2, min(8, n)+1))
            labels = list(map(int, rng.integers(2, k+2, size=n)))
            key = [labels[i] for i in route]
            if len(set(key)) < 2:
                continue
            info = _visible_info(positions, size, ports, labels, key)
            if _teaches(board, info, route, plan["count_witness"], graphs):
                return _render(rng, board, positions, size, ports, labels,
                               route, palette, plan["richness"])
    return _fallback(rng, plan, palette)


def _fallback(rng, plan, palette):
    """A guaranteed graph construction under the same rule, after bounded search.

    An unused duplicate of the first label can replace the start if sockets are
    ignored. An unequal-position, equal-size socket pair permits that mistake
    under cardinality-only matching. A separate compatible chord is inactive.
    """
    size, length = 5, plan["length"]
    gap = int(rng.integers(2, 4))
    positions = [(1+r*(size+gap), 1+c*(size+gap)) for r in range(3) for c in range(3)]
    positions[1] = (positions[1][0]+1, positions[1][1])
    route = [0, 1, 4, 3, 6, 7, 8, 5][:length]
    board = _base_board(positions, size, length, rng)
    edges, _ = _position_graph(positions, size, board)
    # Edge 0--1 has only two common lanes; edge 1--4 has three.
    required = list(plan["required"])
    kinds = [None]*(length-1)
    if "thick" in required:
        kinds[0] = "thick"
    if "separated" in required:
        kinds[1] = "separated"
    if "single" in required:
        kinds[next(i for i in range(length-1) if kinds[i] is None)] = "single"
    ports = [dict() for _ in positions]
    for (a, b), kind in zip(zip(route, route[1:]), kinds):
        e = edges[a, b]
        signature = _signature(rng, e[1], kind)
        ports[a][e[0]], ports[b][_OPPOSITE[e[0]]] = signature, signature
    e = edges[0, 3]
    signature = _signature(rng, e[1])
    ports[0][2] = ports[3][0] = signature
    ports[1][1] = (positions[1][0]+1,)
    ports[2][3] = (positions[2][0]+1,)  # same count, different absolute row
    labels = [2]*9
    for index, hub in enumerate(route):
        labels[hub] = index+2
    labels[2] = labels[0]
    if 3 not in route:
        labels[3] = 9
    info = _visible_info(positions, size, ports, labels, [labels[i] for i in route])
    assert _teaches(board, info, route, plan["count_witness"])
    return _render(rng, board, positions, size, ports, labels, route, palette, plan["richness"])


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10),
                                   ("num_test", num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _array_key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    stable = rng.random() < .30
    base_palette = list(map(int, rng.permutation(10)))
    richness = int(rng.integers(3))
    plans = []
    for i in range(nt):
        plans.append(dict(length=int(rng.integers(3, 6)), richness=richness,
                          required=("single", "thick") if i == 0 else
                                   ("single", "separated") if i == 1 else (),
                          both=i == 1, offset=i == 1, count_witness=i == 1))
    rng.shuffle(plans)
    test_plans = [dict(length=int(rng.integers(6, 9)), richness=richness,
                       required=(), both=False, offset=False, count_witness=False) for _ in range(nq)]
    puzzle, seen = {"train": [], "test": []}, set()
    for split, part in (("train", plans), ("test", test_plans)):
        for plan in part:
            palette = base_palette if stable else list(map(int, rng.permutation(10)))
            pair = _sample_pair(rng, plan, palette)
            if _array_key(pair["input"]) in seen:
                # Forty injective color permutations suffice for <=14 inputs.
                found = False
                for mul in (1, 3, 7, 9):
                    for shift in range(10):
                        remap = (mul*np.arange(10)+shift) % 10
                        candidate = {k: remap[v].astype(np.int8) for k, v in pair.items()}
                        if _array_key(candidate["input"]) not in seen:
                            pair, found = candidate, True
                            break
                    if found:
                        break
                assert found
            seen.add(_array_key(pair["input"]))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    route, hubs, edges = info["route"], info["hubs"], info["edges"]
    links = [(a, b, edges[a, b]) for a, b in zip(route, route[1:])]
    sigs = [hubs[a]["ports"][e[0]] for a, b, e in links]
    route_edges = {frozenset((a, b)) for a, b, e in links}
    return dict(size=info["size"], stations=len(hubs), length=len(route),
                unused=len(hubs)-len(route), kinds=tuple(sorted({_kind(s) for s in sigs})),
                axes=tuple(sorted({e[0] % 2 for a, b, e in links})),
                offset=any(hubs[a]["pos"][0 if e[0] % 2 else 1] !=
                           hubs[b]["pos"][0 if e[0] % 2 else 1] for a, b, e in links),
                directions=tuple(e[0] for a, b, e in links),
                gaps=tuple(e[3]-e[2] for a, b, e in links),
                key_colors=len(set(info["key"])),
                repeated_key=len(set(info["key"])) < len(info["key"]),
                inactive=sum(frozenset((i, j)) not in route_edges
                             for i, row in enumerate(info["exact"]) for j in row if i < j),
                layout=tuple(h["pos"] for h in hubs))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    records, seen = {"train": [], "test": []}, set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            key = _array_key(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            expected, info = _decode(inp)
            assert np.array_equal(out, expected), "incorrect bridge cells or damaged hub/key"
            assert np.any(out != inp)
            assert all(inp[p] == info["background"] for p in info["painted"])
            changed = {tuple(map(int, p)) for p in np.argwhere(inp != out)}
            assert changed == info["painted"]
            for hub in info["hubs"]:
                y, x = hub["pos"]
                s = info["size"]
                assert np.array_equal(inp[y:y+s, x:x+s], out[y:y+s, x:x+s])
            assert all(inp[p] == out[p] for p in info["tokens"])
            labels = [h["label"] for h in info["hubs"]]
            assert len(_routes(info["geometric"], labels, info["key"])) >= 2, "sockets must disambiguate"
            assert len(_routes(info["exact"], labels, [None]*len(info["key"]), 3)) == 3, "key must select among geometric routes"
            # Reversing the visible key reverses the unique route and changes
            # source coloring. This is an input intervention, not latent state.
            reversed_input = inp.copy()
            for p, c in zip(info["tokens"], reversed(info["key"])):
                reversed_input[p] = c
            reversed_output, reversed_info = _decode(reversed_input)
            assert reversed_info["route"] == info["route"][::-1]
            assert any(reversed_output[p] != out[p] for p in info["painted"]), "the key must affect bridge coloring"
            f = _features(info)
            assert f["inactive"] >= 1, "show compatible but unselected bridges"
            f["count_witness"] = len(_routes(info["cardinality"], labels, info["key"])) >= 2
            records[split].append(f)
    train, test = records["train"], records["test"]
    demonstrated = {k for f in train for k in f["kinds"]}
    assert {"single", "thick", "separated"} <= demonstrated
    assert all(set(f["kinds"]) <= demonstrated for f in test)
    assert any(f["count_witness"] and f["offset"] and len(f["axes"]) == 2 for f in train)
    assert all(3 <= f["length"] <= 5 for f in train)
    assert all(6 <= f["length"] <= 8 for f in test)


def _puzzle_key(puzzle):
    return tuple(tuple((_array_key(p["input"]), _array_key(p["output"]))
                       for p in puzzle[s]) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed checks, ablations, count contracts and interventions."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    records, shapes, layouts, roles, motifs, structures = [], set(), set(), set(), set(), set()
    for p in puzzles:
        separated_witness = False
        for split in ("train", "test"):
            for pair in p[split]:
                inp = pair["input"]
                predicted, info = _decode(inp)
                assert np.array_equal(predicted, pair["output"])
                assert np.array_equal(solve(inp.tolist()), predicted)
                f = _features(info)
                f["split"] = split
                records.append(f)
                shapes.add(inp.shape)
                layouts.add((info["size"], f["layout"]))
                roles.add((info["background"], info["carrier"]))
                mapping = {}
                normalized = tuple(mapping.setdefault(int(c), len(mapping)) for c in inp.ravel())
                structures.add((inp.shape, normalized))
                if split == "train":
                    for i, j in zip(info["route"], info["route"][1:]):
                        e = info["edges"][i, j]
                        signature = info["hubs"][i]["ports"][e[0]]
                        if _kind(signature) == "separated":
                            wrong = predicted.copy()
                            for q in _cells(e, tuple(range(signature[0], signature[-1]+1))):
                                if inp[q] == info["background"]:
                                    wrong[q] = info["hubs"][i]["label"]
                            assert not np.array_equal(wrong, predicted), "filling a socket's interval must fail"
                            separated_witness = True
                for hub in info["hubs"]:
                    y, x = hub["pos"]
                    s = info["size"]
                    mask = inp[y:y+s, x:x+s] == hub["label"]
                    motifs.add((s, mask.tobytes()))
        assert separated_witness
    assert {f["size"] for f in records} == {4, 5, 6, 7}
    assert {f["length"] for f in records} == set(range(3, 9))
    assert len(shapes) >= 25 and len(layouts) >= 100
    assert len(roles) >= 25 and len(motifs) >= 250 and len(structures) >= 200
    assert {d for f in records for d in f["directions"]} == set(range(4))
    for split in ("train", "test"):
        group = [f for f in records if f["split"] == split]
        assert {"single", "thick", "separated"} <= {k for f in group for k in f["kinds"]}
        assert any(f["offset"] for f in group) and any(not f["offset"] for f in group)
        assert any(f["unused"] >= 2 for f in group)
        assert any(f["repeated_key"] for f in group)
        assert any(len(set(f["gaps"])) >= 2 for f in group)
    assert any(f["unused"] == 0 for f in records if f["split"] == "test")
    # Color permutation commutes with solving. The key always reads horizontally.
    pair = puzzles[0]["train"][0]
    remap = (3*np.arange(10)+7) % 10
    assert np.array_equal(solve(remap[pair["input"]]), remap[pair["output"]])
    # Exercise the finite fallback with every allowed route length and teaching
    # requirement, independently of whether rejection sampling reaches it.
    rng = np.random.default_rng(876)
    for length in range(3, 9):
        for required in ((), ("single", "thick"), ("single", "separated")):
            plan = dict(length=length, required=required, richness=1, count_witness=True)
            pair = _fallback(rng, plan, list(range(10)))
            assert np.array_equal(solve(pair["input"]), pair["output"])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1042+nt*10+nq, nt, nq)
            assert (len(p["train"]), len(p["test"])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # An output mutation must not pass preservation checks.
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ("train", "test")}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("damaged preserved background accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds)")
