"""Mirror toll stations: reflected data supplies costs and routes select repairs.

Only standard-library modules and NumPy are used. The solver reads exclusively
the supplied pixels. Generation paints labels from its complete patterns, then
checks them against input-derived reconstruction and graph recovery.
"""
from itertools import combinations
import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_AXES = ("lr", "tb", "diag", "anti")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid grid color")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = [], [start]
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _N4:
                point = r + dr, c + dc
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        result.append(np.asarray(sorted(cells), dtype=np.int16))
    return result


def _mate(r, c, n, axis):
    return ((r, n - 1 - c), (n - 1 - r, c),
            (c, r), (n - 1 - c, n - 1 - r))[_AXES.index(axis)]


def _reflect(a, axis):
    return {"lr": lambda: a[:, ::-1], "tb": lambda: a[::-1],
            "diag": lambda: a.T, "anti": lambda: a[::-1, ::-1].T}[axis]()


def _orbits(n, axis):
    return sorted({tuple(sorted({(r, c), _mate(r, c, n, axis)}))
                   for r in range(n) for c in range(n)})


def _ring(cells):
    lo, hi = cells.min(0), cells.max(0)
    size = int(hi[0] - lo[0] + 1)
    if size not in (5, 7) or int(hi[1] - lo[1] + 1) != size:
        return None
    expected = {(int(lo[0]) + r, int(lo[1]) + c)
                for r in range(size) for c in range(size)
                if r in (0, size - 1) or c in (0, size - 1)}
    if set(map(tuple, cells.tolist())) != expected:
        return None
    return int(lo[0]), int(lo[1]), size


def _all_paths(adjacency, start, end):
    paths, stack = [], [(start, (start,), ())]
    while stack:
        node, nodes, edges = stack.pop()
        if node == end:
            paths.append((nodes, edges))
            continue
        for nxt, e in adjacency[node]:
            if nxt not in nodes:
                stack.append((nxt, nodes + (nxt,), edges + (e,)))
    return paths


def _minimum(paths, costs):
    values = [sum(int(costs[n]) for n in nodes[1:-1]) for nodes, _ in paths]
    low = min(values)
    best = [i for i, value in enumerate(values) if value == low]
    return paths[best[0]] if len(best) == 1 else None, low


def _read_input(grid):
    """Recover walls, palette, reflection, complete interiors and graph from pixels."""
    a = _grid(grid)
    rings = {}
    for color in np.unique(a):
        components = _components(a == color)
        boxes = [_ring(cells) for cells in components]
        if len(boxes) >= 2 and all(box is not None for box in boxes):
            if len({box[2] for box in boxes}) == 1:
                rings[int(color)] = boxes
    ends = [c for c, boxes in rings.items() if len(boxes) == 2]
    normals = [c for c, boxes in rings.items() if len(boxes) >= 3]
    _require(len(ends) == len(normals) == 1, "ambiguous station walls")
    endpoint, wall = ends[0], normals[0]
    boxes = sorted(rings[wall] + rings[endpoint])
    _require(5 <= len(boxes) <= 9 and len({b[2] for b in boxes}) == 1,
             "room count or sizes invalid")
    room_map = np.full(a.shape, -1, dtype=np.int16)
    endpoint_ids = []
    interiors = []
    for i, (r, c, size) in enumerate(boxes):
        _require(bool(np.all(room_map[r:r+size, c:c+size] < 0)), "overlapping rooms")
        room_map[r:r+size, c:c+size] = i
        interiors.append(a[r+1:r+size-1, c+1:c+size-1].copy())
        if int(a[r, c]) == endpoint:
            endpoint_ids.append(i)
    colors, counts = np.unique(a[room_map < 0], return_counts=True)
    _require(len(colors) == 2 and counts[0] != counts[1], "ambiguous exterior palette")
    background = int(colors[int(np.argmax(counts))])
    wire = int(colors[int(np.argmin(counts))])
    _require(len({background, wire, wall, endpoint}) == 4, "role colors collide")
    interior_colors = set(int(v) for x in interiors for v in np.unique(x))
    unknown = sorted(interior_colors - {wire})
    _require(len(unknown) == 2 and not interior_colors.intersection({background, wall, endpoint}),
             "interior palette invalid")

    candidates = []
    n = boxes[0][2] - 2
    for mask, paper in (unknown, unknown[::-1]):
        masks = [x == mask for x in interiors]
        intact = [i for i, m in enumerate(masks) if not m.any()]
        if len(intact) < 2 or not all(i in intact for i in endpoint_ids):
            continue
        if not all(np.any(x == paper) for x in interiors):
            continue
        valid = True
        for m in masks:
            cells = np.argwhere(m)
            if len(cells):
                lo, hi = cells.min(0), cells.max(0)
                if len(cells) != int(np.prod(hi - lo + 1)):
                    valid = False
        if not valid or not any(m.any() for m in masks):
            continue
        for axis in _AXES:
            full = []
            for x, m in zip(interiors, masks):
                mate_values = _reflect(x, axis)
                mate_missing = _reflect(m, axis)
                visible = ~m & ~mate_missing
                if np.any(m & mate_missing) or np.any(x[visible] != mate_values[visible]):
                    break
                y = x.copy()
                y[m] = mate_values[m]
                if set(int(v) for v in np.unique(y)) != {paper, wire}:
                    break
                full.append(y)
            else:
                # Intact endpoint fields establish the axis, rather than a fit
                # to just the query windows or an axis supplied by the generator.
                if all(sum(np.array_equal(interiors[i], _reflect(interiors[i], ax))
                           for ax in _AXES) == 1 for i in endpoint_ids):
                    candidates.append((mask, paper, axis, masks, full))
    _require(bool(candidates), "unsupported reflection or erased orbit")
    first = candidates[0]
    _require(all(c[:2] == first[:2] and all(np.array_equal(x, y)
                 for x, y in zip(c[4], first[4])) for c in candidates),
             "ambiguous reflection completion")
    mask, paper, axis, masks, full = first

    wire_cells = _components((a == wire) & (room_map < 0))
    edges = []
    for cells in wire_cells:
        contacts = set()
        for r, c in cells.tolist():
            for dr, dc in _N4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < a.shape[0] and 0 <= cc < a.shape[1]:
                    if room_map[rr, cc] >= 0:
                        contacts.add(int(room_map[rr, cc]))
        _require(len(contacts) == 2, "wire does not join exactly two rooms")
        edges.append(tuple(sorted(contacts)))
    _require(len(set(edges)) == len(edges), "parallel wires")
    adjacency = [[] for _ in boxes]
    for e, (u, v) in enumerate(edges):
        adjacency[u].append((v, e))
        adjacency[v].append((u, e))
    reached, stack = {0}, [0]
    while stack:
        for nxt, _ in adjacency[stack.pop()]:
            if nxt not in reached:
                reached.add(nxt)
                stack.append(nxt)
    _require(len(reached) == len(boxes) and all(1 <= len(x) <= 4 for x in adjacency),
             "disconnected or invalid graph")
    costs = [int(np.count_nonzero(y == wire)) for y in full]
    for i in endpoint_ids:
        costs[i] = 0
    paths = _all_paths(adjacency, *endpoint_ids)
    path, total = _minimum(paths, costs)
    _require(path is not None, "minimum toll routes tie")
    return dict(grid=a, boxes=boxes, interiors=interiors, full=full, masks=masks,
                axis=axis, roles=(background, paper, wall, endpoint, wire, mask),
                endpoints=endpoint_ids, edges=edges, wire_cells=wire_cells,
                adjacency=adjacency, costs=costs, paths=paths, path=path, total=total)


def _paint(data):
    result = data["grid"].copy()
    nodes, edges = data["path"]
    endpoint = data["roles"][3]
    for i in nodes:
        r, c, side = data["boxes"][i]
        block = result[r:r+side, c:c+side]
        block[0, :] = block[-1, :] = endpoint
        block[:, 0] = block[:, -1] = endpoint
        block[1:-1, 1:-1] = data["full"][i]
    for e in edges:
        result[tuple(data["wire_cells"][e].T)] = endpoint
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the selected annotated network, using only input_grid."""
    return _paint(_read_input(input_grid))


def _pattern(rng, n, axis, target=None, pairs_only=False, singles_only=False):
    orbits = _orbits(n, axis)
    singles = [o for o in orbits if len(o) == 1]
    pairs = [o for o in orbits if len(o) == 2]
    rng.shuffle(singles)
    rng.shuffle(pairs)
    a = np.zeros((n, n), dtype=np.int8)
    if target is None:
        density = float(rng.uniform(.18, .72))
        for o in orbits:
            if rng.random() < density:
                for point in o:
                    a[point] = 1
        if not a.any() or a.all():
            # A finite repair of a degenerate sample, without recursive retries.
            a[:] = 0
            for point in orbits[int(rng.integers(len(orbits)))]:
                a[point] = 1
    else:
        if pairs_only:
            _require(target % 2 == 0 and target // 2 <= len(pairs), "invalid pair target")
            chosen = pairs[:target // 2]
        elif singles_only:
            _require(target <= len(singles), "invalid singleton target")
            chosen = singles[:target]
        else:
            # At least one genuine two-cell orbit for all costs >=2.
            k = min(len(pairs), target // 2)
            chosen = pairs[:k] + singles[:target - 2*k]
        for o in chosen:
            for point in o:
                a[point] = 1
        _require(int(a.sum()) == target, "unattainable dot target")
    return a


def _unique_pattern(rng, n, axis):
    for _ in range(100):
        a = _pattern(rng, n, axis)
        if sum(np.array_equal(a, _reflect(a, ax)) for ax in _AXES) == 1:
            return a
    # A single off-center fixed point has precisely the desired reflection.
    a = np.zeros((n, n), dtype=np.int8)
    point = {"lr": (0, n//2), "tb": (n//2, 0),
             "diag": (0, 0), "anti": (0, n-1)}[axis]
    a[point] = 1
    return a


def _windows(pattern, axis, hidden="any"):
    n = pattern.shape[0]
    result = []
    for r in range(n):
        for s in range(r + 1, n + 1):
            for c in range(n):
                for d in range(c + 1, n + 1):
                    cells = {(rr, cc) for rr in range(r, s) for cc in range(c, d)}
                    if any(_mate(rr, cc, n, axis) in cells for rr, cc in cells):
                        continue
                    dots = int(pattern[r:s, c:d].sum())
                    if hidden == "dot" and not dots:
                        continue
                    if hidden == "paper" and dots:
                        continue
                    result.append((r, s, c, d))
    return result


def _random_graph(rng):
    nr, nc = ((2, 3), (3, 2), (3, 3))[int(rng.integers(3))]
    vertices = [(r, c) for r in range(nr) for c in range(nc)]
    target = int(rng.integers(5, len(vertices) + 1))
    active = {vertices[int(rng.integers(len(vertices)))]}
    edges = []
    while len(active) < target:
        choices = [(a, b) for a in sorted(active) for b in vertices
                   if b not in active and abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1]
        a, b = choices[int(rng.integers(len(choices)))]
        active.add(b)
        edges.append(tuple(sorted((a, b))))
    extras = [(a, b) for a, b in combinations(sorted(active), 2)
              if abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1 and (a, b) not in edges]
    rng.shuffle(extras)
    if not extras:
        return None
    edges.extend(extras[:int(rng.integers(1, len(extras) + 1))])
    return sorted(active), edges


def _witness_graph(rng):
    vertices = [(r, c) for r in range(2) for c in range(3)]
    edges = [((r, c), (r, c+1)) for r in range(2) for c in range(2)]
    edges += [((0, 0), (1, 0)), ((0, 2), (1, 2))]
    if rng.random() < .5:
        vertices.append((2, 1))
        edges.append(((1, 1), (2, 1)))
    return vertices, edges


def _adjacency(vertices, edges):
    ids = {v: i for i, v in enumerate(vertices)}
    adj = [[] for _ in vertices]
    for e, (a, b) in enumerate(edges):
        u, v = ids[a], ids[b]
        adj[u].append((v, e))
        adj[v].append((u, e))
    return adj


def _render(rng, vertices, edges, patterns, masks, endpoints, path, roles):
    side = patterns[0].shape[0] + 2
    transpose_geometry = bool(rng.integers(2))
    locations = {v: (v[1], v[0]) if transpose_geometry else v for v in vertices}
    nr = max(v[0] for v in locations.values())+1
    nc = max(v[1] for v in locations.values())+1
    def coords(count):
        gaps = [int(rng.integers(2, 5)) for _ in range(count-1)]
        return [0] + [int(x) for x in np.cumsum([side+g for g in gaps])]
    rows, cols = coords(nr), coords(nc)
    height, width = rows[-1]+side, cols[-1]+side
    hh, ww = int(rng.integers(height, 31)), int(rng.integers(width, 31))
    top, left = int(rng.integers(hh-height+1)), int(rng.integers(ww-width+1))
    anchors = {v: (rows[locations[v][0]]+top, cols[locations[v][1]]+left) for v in vertices}
    background, paper, wall, endpoint, wire, mask_color = roles
    inp = np.full((hh, ww), background, dtype=np.int8)
    edge_masks = []
    half = side // 2
    for a, b in edges:
        ar, ac = anchors[a]
        br, bc = anchors[b]
        # Independent ports create a mixture of straight and elbow wires.
        off_a, off_b = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
        cells = set()
        if ar == br:
            if ac > bc:
                ar, ac, br, bc = br, bc, ar, ac
            y1, y2 = ar+half+off_a, br+half+off_b
            x1, x2 = ac+side, bc-1
            middle = int(rng.integers(x1, x2+1))
            cells.update((y1, x) for x in range(x1, middle+1))
            cells.update((y, middle) for y in range(min(y1,y2), max(y1,y2)+1))
            cells.update((y2, x) for x in range(middle, x2+1))
        else:
            if ar > br:
                ar, ac, br, bc = br, bc, ar, ac
            x1, x2 = ac+half+off_a, bc+half+off_b
            y1, y2 = ar+side, br-1
            middle = int(rng.integers(y1, y2+1))
            cells.update((y, x1) for y in range(y1, middle+1))
            cells.update((middle, x) for x in range(min(x1,x2), max(x1,x2)+1))
            cells.update((y, x2) for y in range(middle, y2+1))
        pixels = np.asarray(sorted(cells), dtype=np.int16)
        inp[tuple(pixels.T)] = wire
        edge_masks.append(pixels)
    for i, v in enumerate(vertices):
        r, c = anchors[v]
        inp[r:r+side, c:c+side] = endpoint if i in endpoints else wall
        inner = np.where(patterns[i], wire, paper).astype(np.int8)
        inp[r+1:r+side-1, c+1:c+side-1] = inner
        if i in masks:
            a, b, x, y = masks[i]
            inp[r+1+a:r+1+b, c+1+x:c+1+y] = mask_color
    # Label from known complete fields and the construction graph; no solve call.
    out = inp.copy()
    for i in path[0]:
        r, c = anchors[vertices[i]]
        block = out[r:r+side, c:c+side]
        block[0, :] = block[-1, :] = endpoint
        block[:, 0] = block[:, -1] = endpoint
        block[1:-1, 1:-1] = np.where(patterns[i], wire, paper)
    for e in path[1]:
        out[tuple(edge_masks[e].T)] = endpoint
    # Transpose the layout independently of interiors, then optionally reflect
    # the whole drawing. Geometry and the local symmetry axis are independent.
    # Before flips, compare the entire recovered graph, not only its degrees.
    parsed = _read_input(inp)
    by_anchor = {(r, c): i for i, (r, c, _) in enumerate(parsed["boxes"])}
    expected = {tuple(sorted((by_anchor[anchors[a]], by_anchor[anchors[b]])))
                for a, b in edges}
    _require(set(parsed["edges"]) == expected, "rendered edges differ from construction")
    if rng.random() < .5:
        inp, out = inp[::-1].copy(), out[::-1].copy()
    if rng.random() < .5:
        inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
    return {"input": inp, "output": out}


def _construct(rng, n, axis, roles, mode, test=False):
    """mode: A witness, orbit-count witness, or random graph."""
    if mode in ("A", "orbits"):
        vertices, edges = _witness_graph(rng)
        patterns = [_pattern(rng, n, axis) for _ in vertices]
        endpoints = [vertices.index((0, 0)), vertices.index((0, 2))]
        direct = vertices.index((0, 1))
        detour = [vertices.index((1, c)) for c in range(3)]
        if mode == "A":
            p = np.zeros((n, n), dtype=np.int8)
            if axis in ("lr", "tb"):
                p[:, 0] = p[:, -1] = 1
                window = (0, n, 0, 1)
                if axis == "tb":
                    p = p.T
                    window = (0, 1, 0, n)
            else:
                p[0, :] = p[:, 0] = 1
                window = (0, 1, 1, n)
                if axis == "anti":
                    p = p[:, ::-1]
                    window = (0, 1, 0, n-1)
            patterns[direct] = p
            for i, target in zip(detour, (1, 1, n-1)):
                patterns[i] = _pattern(rng, n, axis, target)
            # This window hides paper; keeping all detour dots visible makes the
            # visible-only minimum strictly wrong even in a 3x3 diagonal field.
            choices = _windows(patterns[detour[-1]], axis, "paper")
            _require(bool(choices), "no paper window in witness")
            masks = {direct: window, detour[-1]: choices[int(rng.integers(len(choices)))]}
        else:
            # direct: n two-cell orbits, cost 2n; detour: one pair, one fixed
            # dot, n-1 fixed dots, cost n+2. Orbit-count costs instead are n
            # versus n+1, so counting a symmetric pair only once is wrong.
            patterns[direct] = _pattern(rng, n, axis, 2*n, pairs_only=True)
            patterns[detour[0]] = _pattern(rng, n, axis, 2, pairs_only=True)
            patterns[detour[1]] = _pattern(rng, n, axis, 1, singles_only=True)
            patterns[detour[2]] = _pattern(rng, n, axis, n-1, singles_only=True)
            choices = _windows(patterns[detour[0]], axis, "dot")
            masks = {detour[0]: choices[int(rng.integers(len(choices)))]}
        for i in endpoints:
            patterns[i] = _unique_pattern(rng, n, axis)
    else:
        graph = _random_graph(rng)
        _require(graph is not None, "graph lacks competing routes")
        vertices, edges = graph
        patterns = [_pattern(rng, n, axis) for _ in vertices]
        adj = _adjacency(vertices, edges)
        options = []
        for a, b in combinations(range(len(vertices)), 2):
            paths = _all_paths(adj, a, b)
            if len(paths) < 2:
                continue
            path, _ = _minimum(paths, [p.sum() for p in patterns])
            if path is not None and len(path[0]) >= 3:
                ink_nodes = [i for i in path[0][1:-1] if _windows(patterns[i], axis, "dot")]
                if ink_nodes:
                    options.append((a, b, path, ink_nodes))
        _require(bool(options), "no unique useful endpoint route")
        a, b, path, ink_nodes = options[int(rng.integers(len(options)))]
        endpoints = [a, b]
        for i in endpoints:
            patterns[i] = _unique_pattern(rng, n, axis)
        selected = ink_nodes[int(rng.integers(len(ink_nodes)))]
        choices = _windows(patterns[selected], axis, "dot")
        masks = {selected: choices[int(rng.integers(len(choices)))]}
        others = [i for i in range(len(vertices)) if i not in endpoints and i != selected]
        rng.shuffle(others)
        wanted = int(rng.integers(3, len(vertices)-1)) if test else int(rng.integers(1, 3))
        for i in others[:wanted-1]:
            choices = _windows(patterns[i], axis)
            masks[i] = choices[int(rng.integers(len(choices)))]
    adj = _adjacency(vertices, edges)
    paths = _all_paths(adj, *endpoints)
    path, _ = _minimum(paths, [p.sum() for p in patterns])
    _require(path is not None, "construction tied")
    if test and len(masks) < 3:
        others = [i for i in range(len(vertices)) if i not in endpoints and i not in masks]
        rng.shuffle(others)
        for i in others[:3-len(masks)]:
            choices = _windows(patterns[i], axis)
            masks[i] = choices[int(rng.integers(len(choices)))]
    pair = _render(rng, vertices, edges, patterns, masks, endpoints, path, roles)
    data = _read_input(pair["input"])
    _require(len(data["boxes"]) == len(vertices), "lost room")
    # Compare recovered topology with construction after the placement flips.
    expected_degrees = sorted(len(x) for x in adj)
    _require(sorted(len(x) for x in data["adjacency"]) == expected_degrees
             and len(data["edges"]) == len(edges), "rendered graph changed")
    _require(np.array_equal(_paint(data), pair["output"]), "rendered semantics changed")
    if mode == "A":
        f = _features(data)
        _require(f["A_conflict"] and f["wire_area_conflict"], "A witness lost")
    if mode == "orbits":
        _require(_features(data)["orbit_conflict"], "orbit witness lost")
    return pair


def _features(data):
    wire = data["roles"][4]
    visible = [int(np.count_nonzero(x == wire)) for x in data["interiors"]]
    orbit_costs = [sum(int(p[o[0]] == wire) for o in _orbits(p.shape[0], data["axis"]))
                   for p in data["full"]]
    blob_costs = [len(_components(p == wire)) for p in data["full"]]
    visible_path, _ = _minimum(data["paths"], visible)
    hop_path, _ = _minimum(data["paths"], [1]*len(visible))
    orbit_path, _ = _minimum(data["paths"], orbit_costs)
    blob_path, _ = _minimum(data["paths"], blob_costs)
    area_scores = [sum(len(data["wire_cells"][e]) for e in edges)
                   for _, edges in data["paths"]]
    minimum_area = min(area_scores)
    area_best = [i for i, value in enumerate(area_scores) if value == minimum_area]
    area_path = data["paths"][area_best[0]] if len(area_best) == 1 else None
    chosen = data["path"]
    selected = set(chosen[0])
    masked = {i for i, m in enumerate(data["masks"]) if m.any()}
    return dict(axis=data["axis"], side=data["boxes"][0][2], rooms=len(visible),
                shape=data["grid"].shape, palette=data["roles"],
                route_count=len(data["paths"]), cost=data["total"],
                selected_rooms=len(chosen[0])-2, masked=len(masked),
                unused_masks=len(masked-selected), selected_masks=len(masked & selected),
                A_conflict=visible_path is not None and visible_path != chosen,
                hop_conflict=hop_path is not None and hop_path != chosen,
                orbit_conflict=orbit_path is not None and orbit_path != chosen,
                blob_conflict=blob_path is not None and blob_path != chosen,
                wire_area_conflict=area_path is not None and area_path != chosen,
                hidden_selected_dot=any(np.any(data["masks"][i] & (data["full"][i] == wire))
                                        for i in selected),
                degrees=tuple(sorted(len(x) for x in data["adjacency"])),
                bends=sum(len(np.unique(c[:, 0])) > 1 and len(np.unique(c[:, 1])) > 1
                          for c in data["wire_cells"]))


def _check_count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _check_count(num_train, "num_train", 2, 10)
    nq = _check_count(num_test, "num_test", 1, 4)
    _require(nt > nq, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    diagonal = bool(rng.integers(2))
    axes = ("diag", "anti") if diagonal else ("lr", "tb")
    scale_shift = bool(rng.integers(2))
    fresh = bool(rng.integers(2))
    common_roles = tuple(int(x) for x in rng.choice(10, 6, replace=False))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for i in range(count):
            mode = ("A" if i == 0 else "orbits") if split == "train" and i < 2 else "random"
            n = 3 if split == "train" and (scale_shift or i == 0) else 5
            if split == "test" and not scale_shift:
                n = int(rng.choice([3, 5]))
            axis = axes[i % 2] if split == "train" and i < 2 else axes[int(rng.integers(2))]
            for attempt in range(70):
                roles = tuple(int(x) for x in rng.choice(10, 6, replace=False)) if fresh else common_roles
                chosen_mode = mode if attempt < 55 or mode != "random" else "orbits"
                try:
                    pair = _construct(rng, n, axis, roles, chosen_mode, test=split == "test")
                    # Vertical/horizontal flips interchange the two diagonal
                    # axes. Correct that visibly by an extra flip when needed.
                    if _read_input(pair["input"])["axis"] != axis:
                        pair = {k: v[:, ::-1].copy() for k, v in pair.items()}
                    if _key(pair["input"]) in seen:
                        continue
                    if split == "train" and i == 1 and not _features(_read_input(pair["input"]))["blob_conflict"]:
                        continue
                    seen.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("bounded construction exhausted")
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "invalid puzzle counts")
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair keys")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "output shape changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            d = _read_input(inp)
            _require(np.array_equal(out, _paint(d)), "incorrect selective repair or highlighting")
            f = _features(d)
            _require(f["selected_masks"] >= 1, "no repaired window")
            features[split].append(f)
            # Explicitly certify whole-footprint preservation off the route.
            footprint = np.zeros(inp.shape, dtype=bool)
            for i in d["path"][0]:
                r, c, side = d["boxes"][i]
                footprint[r:r+side, c:c+side] = True
            for e in d["path"][1]:
                footprint[tuple(d["wire_cells"][e].T)] = True
            _require(np.array_equal(inp[~footprint], out[~footprint]), "off-route change")
    tr = features["train"]
    _require(any(f["A_conflict"] and f["hop_conflict"] and f["wire_area_conflict"]
                 and f["unused_masks"] for f in tr),
             "missing completed-cost versus visible/hop training witness")
    _require(any(f["orbit_conflict"] for f in tr), "missing cell-versus-orbit witness")
    _require(any(f["blob_conflict"] for f in tr), "missing cell-versus-blob witness")
    _require(any(f["hidden_selected_dot"] for f in tr), "hidden dots never repaired in training")
    _require({f["axis"] for f in features["test"]} <= {f["axis"] for f in tr},
             "test uses an undemonstrated reflection")


def self_test():
    """Bounded 42-seed coverage plus semantic mutation and equivariance checks."""
    signatures, shapes, structures, axes, palettes, rooms, stats = set(), set(), set(), set(), set(), set(), []
    for seed in range(42):
        p = generate_puzzle(seed)
        signature = tuple(tuple((_key(x["input"]), _key(x["output"])) for x in p[s])
                          for s in ("train", "test"))
        signatures.add(signature)
        if seed < 3:
            q = generate_puzzle(seed)
            assert signature == tuple(tuple((_key(x["input"]), _key(x["output"])) for x in q[s])
                                      for s in ("train", "test"))
        for split in ("train", "test"):
            for pair in p[split]:
                before = pair["input"].copy()
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"])
                d = _read_input(before)
                f = _features(d)
                f["split"] = split
                stats.append(f)
                shapes.add(before.shape)
                axes.add(f["axis"])
                palettes.add(f["palette"])
                rooms.add(f["rooms"])
                normalized = np.zeros(before.shape, dtype=np.int8)
                for i, color in enumerate(d["roles"]):
                    normalized[before == color] = i
                structures.add(_key(normalized))
    assert len(signatures) == 42 and len(structures) >= 170
    assert len(shapes) >= 25 and len(palettes) >= 40 and len(rooms) >= 4
    assert axes == set(_AXES)
    assert set(c for palette in palettes for c in palette) == set(range(10))
    assert any(4 in f["degrees"] for f in stats)
    assert any(1 in f["degrees"] and 3 in f["degrees"] for f in stats)
    assert any(f["route_count"] >= 3 for f in stats)
    assert any(f["bends"] >= 3 for f in stats)
    assert all(f["masked"] >= 3 for f in stats if f["split"] == "test")
    assert {f["side"] for f in stats} == {5, 7}
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(500+nt*5+nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(93, 2, 1)
    for turn in range(4):
        rotated = {s: [{k: np.rot90(v, turn).copy() for k, v in x.items()}
                       for x in p[s]] for s in p}
        validate_generated_puzzle(rotated)
    recolored = {s: [] for s in p}
    rng = np.random.default_rng(876)
    for s in p:
        for x in p[s]:
            perm = rng.permutation(10)
            recolored[s].append({k: perm[v].tolist() for k, v in x.items()})
    validate_generated_puzzle(recolored)
    assert np.array_equal(solve(recolored["test"][0]["input"]), recolored["test"][0]["output"])
    broken = {s: [{k: v.copy() for k, v in x.items()} for x in p[s]] for s in p}
    broken["test"][0]["output"][0, 0] = (int(broken["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("wrong output accepted")
    a = p["train"][0]["input"]
    d = _read_input(a)
    # Erase a singleton on the reflection line: its orbit has no visible mate.
    bad = a.copy()
    i = next(i for i in range(len(d["boxes"])) if i not in d["endpoints"])
    r, c, side = d["boxes"][i]
    bad[r+1:r+side-1, c+1:c+side-1] = d["full"][i]
    fixed = next(o[0] for o in _orbits(side-2, d["axis"]) if len(o) == 1)
    bad[r+1+fixed[0], c+1+fixed[1]] = d["roles"][5]
    try:
        solve(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("erased singleton accepted")
    # Break the input's supported reflection in one intact endpoint.
    bad = a.copy()
    r, c, side = d["boxes"][d["endpoints"][0]]
    orbit = next(o for o in _orbits(side-2, d["axis"]) if len(o) == 2)
    rr, cc = r+1+orbit[0][0], c+1+orbit[0][1]
    bad[rr, cc] = d["roles"][1] if bad[rr, cc] == d["roles"][4] else d["roles"][4]
    try:
        solve(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("damaged reflection accepted")
    # Create an exact toll tie in the three-room detour fixture.
    rng = np.random.default_rng(123)
    for axis in _AXES:
        pair = _construct(rng, 5, axis, (0, 1, 2, 3, 4, 5), "orbits")
        d = _read_input(pair["input"])
        bad = pair["input"].copy()
        i = d["path"][0][2]
        r, c, side = d["boxes"][i]
        added = d["costs"][i] + (2*5 - d["total"])
        bits = _pattern(rng, 5, axis, added)
        bad[r+1:r+side-1, c+1:c+side-1] = np.where(bits, 4, 1)
        try:
            solve(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("tied route accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds and family-specific checks")
