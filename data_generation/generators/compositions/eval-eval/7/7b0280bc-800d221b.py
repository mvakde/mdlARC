"""First-arrival quorum: shortest-station wavefronts with local port votes.

No parent code, fixtures, file access, latent seed, or demonstrations are used by
solve. Background is always 0; the other three colors are geometric roles.
"""
from collections import Counter, deque
import heapq
from numbers import Integral

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid of dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _neighbors(p, shape):
    r, c = p
    return [(r + dr, c + dc) for dr, dc in DIRS
            if 0 <= r + dr < shape[0] and 0 <= c + dc < shape[1]]


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, stack = {start}, [start]
        while stack:
            for q in _neighbors(stack.pop(), mask.shape):
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for e, (a, b) in enumerate(edges):
        adj[a].append((b, e))
        adj[b].append((a, e))
    return adj


def _propagate(n, edges, sources, weights=None, emissions=None):
    """Positive edge costs; one predecessor connector is one vote.

    Unit costs implement the family. Other costs are used only for ablations.
    Source panels are fixed emitters and never relay another panel's signal.
    """
    adj = _adjacency(n, edges)
    weights = [1] * len(edges) if weights is None else weights
    dist = [float("inf")] * n
    queue = []
    for a in sources:
        dist[a] = 0
        heapq.heappush(queue, (0, a))
    while queue:
        cost, a = heapq.heappop(queue)
        if cost != dist[a]:
            continue
        for b, e in adj[a]:
            new = cost + weights[e]
            if new < dist[b]:
                dist[b] = new
                heapq.heappush(queue, (new, b))
    if any(d == float("inf") for d in dist):
        raise ValueError("Every relay must be reachable from a panel")
    labels = dict(sources)
    incoming = {}
    for a in sorted(range(n), key=lambda x: (dist[x], x)):
        if a in sources:
            continue
        previous = [(b, e) for b, e in adj[a]
                    if dist[b] + weights[e] == dist[a]]
        votes = [emissions[e] if emissions is not None and b in sources
                 else labels[b] for b, e in previous]
        counts = Counter(votes)
        maximum = max(counts.values())
        winners = [c for c, count in counts.items() if count == maximum]
        if len(winners) != 1:
            raise ValueError("A relay's first-arrival port vote must be strict")
        labels[a] = winners[0]
        incoming[a] = previous
    return dist, labels, incoming


def _parse(input_grid):
    a = _grid(input_grid)
    colors = set(map(int, np.unique(a)))
    if len(colors) != 4 or 0 not in colors:
        raise ValueError("Expected background 0 and exactly three nonzero roles")
    rings_by_color = {}
    for color in sorted(colors - {0}):
        centers = []
        for r in range(1, a.shape[0] - 1):
            for c in range(1, a.shape[1] - 1):
                if a[r, c] == 0:
                    block = a[r-1:r+2, c-1:c+2]
                    if np.count_nonzero(block == color) == 8:
                        centers.append((r, c))
        if centers:
            rings_by_color[color] = centers
    if len(rings_by_color) != 1:
        raise ValueError("The hollow relay rims must identify one scaffold color")
    scaffold, centers = next(iter(rings_by_color.items()))
    signals = sorted(colors - {0, scaffold})
    objects, kinds, source_labels = [], [], {}
    for center in centers:
        r, c = center
        cells = {(r+dr, c+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        objects.append(cells)
        kinds.append("relay")
    panels = _components(np.isin(a, signals))
    if not 2 <= len(panels) <= 5:
        raise ValueError("Expected 2..5 rectangular two-color source panels")
    for cells in panels:
        rr, cc = zip(*cells)
        if len(cells) != (max(rr)-min(rr)+1) * (max(cc)-min(cc)+1):
            raise ValueError("Panels must be solid rectangles")
        counts = Counter(int(a[p]) for p in cells)
        if len(counts) != 2 or counts[signals[0]] == counts[signals[1]]:
            raise ValueError("Each panel needs a strict two-color local majority")
        source_labels[len(objects)] = max(counts, key=counts.get)
        objects.append(cells)
        kinds.append("panel")
    ids = np.full(a.shape, -1, dtype=np.int16)
    wire_mask = a == scaffold
    for i, cells in enumerate(objects):
        for p in cells:
            if ids[p] >= 0:
                raise ValueError("Objects overlap")
            ids[p] = i
            wire_mask[p] = False
    wires = _components(wire_mask)
    edges, contact_colors = [], {}
    for e, cells in enumerate(wires):
        degrees = [sum(q in cells for q in _neighbors(p, a.shape)) for p in cells]
        if (len(cells) > 1 and (degrees.count(1) != 2 or any(v not in (1, 2) for v in degrees))):
            raise ValueError("A connector must be a thin, unbranched pixel path")
        touching = set()
        contacts = {}
        for p in cells:
            for q in _neighbors(p, a.shape):
                if ids[q] >= 0:
                    i = int(ids[q])
                    touching.add(i)
                    contacts.setdefault(i, set()).add(q)
        if len(touching) != 2:
            raise ValueError("Each wire must join exactly two objects")
        if any(len(contacts[i]) != 1 for i in touching):
            raise ValueError("A wire must touch one pixel on each object")
        u, v = sorted(touching)
        if u in source_labels and v in source_labels:
            raise ValueError("Panels cannot connect directly to other panels")
        edges.append((u, v))
        for i in touching & source_labels.keys():
            if len(contacts[i]) != 1:
                raise ValueError("Each panel connection has one contact pixel")
            contact_colors[e] = int(a[next(iter(contacts[i]))])
    if len(set(edges)) != len(edges):
        raise ValueError("Parallel wires are outside this family's convention")
    adj = _adjacency(len(objects), edges)
    if not all(1 <= len(row) <= 4 for row in adj):
        raise ValueError("Object degrees must be 1..4")
    reached, queue = {0}, deque([0])
    while queue:
        for b, _ in adj[queue.popleft()]:
            if b not in reached:
                reached.add(b)
                queue.append(b)
    if len(reached) != len(objects):
        raise ValueError("The station network must be connected")
    dist, labels, incoming = _propagate(len(objects), edges, source_labels)
    return dict(grid=a, scaffold=scaffold, signals=signals, centers=centers,
                objects=objects, sources=source_labels, kinds=kinds,
                wires=wires, edges=edges, adjacency=adj, distance=dist,
                labels=labels, incoming=incoming, contacts=contact_colors)


def _paint(data):
    out = data["grid"].copy()
    for i, p in enumerate(data["centers"]):
        out[p] = data["labels"][i]
    for cells, (u, v) in zip(data["wires"], data["edges"]):
        if data["distance"][u] == data["distance"][v]:
            continue
        upstream = u if data["distance"][u] < data["distance"][v] else v
        for p in cells:
            out[p] = data["labels"][upstream]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Resolve roles, shortest arrival levels, causal votes, and full wire masks."""
    return _paint(_parse(input_grid))


def _witness_graph():
    # P forks through A and B. At T: P's two ports beat Q's one.
    # R approaches through a longer chain; a same-time edge stops transmission.
    vertices = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1),
                (1, 2), (2, 1), (2, 2), (2, 3), (3, 1), (3, 2)]
    edges = [((0, 0), (0, 1)), ((0, 0), (1, 0)),
             ((0, 1), (1, 1)), ((1, 0), (1, 1)),
             ((0, 2), (1, 2)), ((1, 2), (1, 1)),
             ((1, 1), (2, 1)), ((2, 1), (3, 1)),
             ((3, 1), (3, 2)), ((3, 2), (2, 2)), ((2, 2), (2, 3))]
    return vertices, edges, {(0, 0), (0, 2), (2, 3)}


def _nested_graph():
    # Two equal-X ports merge at M=(1,1); its one outgoing port has weight one
    # at N=(2,3), where two Y ports win. Counting all shortest paths ties 2:2.
    vertices = [(0, 0), (0, 1), (0, 3), (0, 4), (1, 0),
                (1, 1), (1, 3), (1, 4), (2, 3), (2, 4)]
    edges = [((0, 0), (0, 1)), ((0, 0), (1, 0)),
             ((0, 1), (1, 1)), ((1, 0), (1, 1)),
             ((1, 1), (2, 1)), ((2, 1), (2, 2)), ((2, 2), (2, 3)),
             ((0, 4), (0, 3)), ((0, 3), (1, 3)), ((1, 3), (2, 3)),
             ((0, 4), (1, 4)), ((1, 4), (2, 4)), ((2, 4), (2, 3))]
    return vertices, edges, {(0, 0), (0, 4)}


def _transfer_graph(rng):
    """Extend the nested election with variable branches and another loop."""
    vertices, edges, panels = _nested_graph()
    optional = [((0, 1), (0, 2)), ((1, 0), (2, 0)), ((1, 1), (1, 2))]
    for a, b in optional:
        if rng.random() < .65:
            vertices.append(b)
            edges.append((a, b))
    vertices.append((3, 3))
    edges.append(((2, 3), (3, 3)))
    if rng.random() < .65:
        vertices.append((3, 4))
        edges += [((3, 3), (3, 4)), ((3, 4), (2, 4))]
    if rng.random() < .55:
        vertices.append((3, 2))
        edges.append(((3, 3), (3, 2)))
    return sorted(vertices), edges, panels


def _random_graph(rng, harder):
    nr, nc = map(int, rng.integers(3, 5, size=2))
    if harder and nr == nc == 3:
        if rng.random() < .5:
            nr = 4
        else:
            nc = 4
    lattice = [(r, c) for r in range(nr) for c in range(nc)]
    candidates = [(a, b) for a in lattice for b in lattice
                  if a < b and abs(a[0]-b[0]) + abs(a[1]-b[1]) == 1]
    active = {lattice[int(rng.integers(len(lattice)))]}
    edges = []
    target = int(rng.integers(max(7, len(lattice)-4), len(lattice)+1))
    while len(active) < target:
        choices = [(a, b) for a, b in candidates if (a in active) != (b in active)]
        a, b = choices[int(rng.integers(len(choices)))]
        edges.append((a, b))
        active.update((a, b))
    extras = [e for e in candidates if e[0] in active and e[1] in active and e not in edges]
    rng.shuffle(extras)
    edges += extras[:int(rng.integers(min(1, len(extras)) if harder else 0,
                                    min(len(extras), 5)+1))]
    # Sources are independent nonadjacent lattice sites, with 2..5 emitters.
    order = sorted(active)
    rng.shuffle(order)
    panels = set()
    desired = int(rng.integers(2, (3 if harder else min(5, len(active)//2))+1))
    for v in order:
        if all(abs(v[0]-p[0])+abs(v[1]-p[1]) > 1 for p in panels):
            panels.add(v)
        if len(panels) == desired:
            break
    if len(panels) < 2:
        return None
    # Suppress a few degree-two sites into wire bends/long segments. They no
    # longer count as stations; the retained lattice edges still draw the wire.
    degrees = Counter(v for e in edges for v in e)
    retained = set(active)
    for v in sorted(active - panels):
        if (degrees[v] == 2 and len(retained) > len(panels)+4
                and rng.random() < (.45 if harder else .28)):
            retained.remove(v)
    return sorted(retained), edges, panels


def _compressed_paths(vertices, lattice_edges):
    adj = {}
    for a, b in lattice_edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    visible = set(vertices)
    paths = []
    for a in vertices:
        for b in sorted(adj[a]):
            path, previous, current = [a, b], a, b
            while current not in visible:
                choices = [v for v in adj[current] if v != previous]
                if len(choices) != 1:
                    return None
                previous, current = current, choices[0]
                path.append(current)
            if a == current:
                return None
            if a < current:
                paths.append(path)
    ends = [(p[0], p[-1]) for p in paths]
    return paths if len(set(ends)) == len(ends) else None


def _segment(a, b):
    dr = int(np.sign(b[0]-a[0]))
    dc = int(np.sign(b[1]-a[1]))
    n = abs(b[0]-a[0])+abs(b[1]-a[1])
    return {(a[0]+dr*i, a[1]+dc*i) for i in range(n+1)}


def _pattern(rng, h, w, majority, minority, forced=(), keep_majority=(), cap=None):
    area = h*w
    maximum = min((area-1)//2, area-len(keep_majority))
    if cap is not None:
        maximum = min(maximum, cap)
    count = int(rng.integers(max(1, len(forced)), maximum+1))
    block = np.full((h, w), majority, dtype=np.int8)
    occupied = set(forced)
    style = int(rng.integers(3))
    positions = [(r, c) for r in range(h) for c in range(w)
                 if (r, c) not in occupied and (r, c) not in keep_majority]
    if style == 0:
        rng.shuffle(positions)
    elif style == 1:
        r0, c0 = int(rng.integers(h)), int(rng.integers(w))
        positions.sort(key=lambda p: (abs(p[0]-r0)+abs(p[1]-c0), p))
    else:
        positions.sort(key=lambda p: ((p[0]+p[1]) % 2, p))
    occupied.update(positions[:count-len(occupied)])
    for p in occupied:
        block[p] = minority
    return block


def _construct(rng, palette, witness=None, harder=False, invert=False):
    for attempt in range(161):
        fallback = attempt == 160 and witness is None
        graph = (_witness_graph() if witness == "ports" else _nested_graph()
                 if witness == "nested" else _transfer_graph(rng)
                 if witness == "transfer" or fallback else _random_graph(rng, harder))
        if graph is None:
            continue
        vertices, lattice_edges, panels = graph
        paths = _compressed_paths(vertices, lattice_edges)
        if paths is None:
            continue
        ids = {v: i for i, v in enumerate(vertices)}
        edges = [(ids[p[0]], ids[p[-1]]) for p in paths]
        scaffold, s0, s1 = palette
        if witness:
            x, y = (s1, s0) if invert else (s0, s1)
            sources = {ids[v]: x if v == (0, 0) else y for v in panels}
        else:
            assignments = list(map(int, rng.integers(2, size=len(panels))))
            if len(set(assignments)) == 1:
                assignments[int(rng.integers(len(assignments)))] ^= 1
            sources = {ids[v]: (s0, s1)[bit] for v, bit in zip(sorted(panels), assignments)}
        try:
            dist, labels, incoming = _propagate(len(vertices), edges, sources)
        except ValueError:
            continue
        if harder and (max(dist) < 3 or len(edges)-len(vertices)+1 < 1
                       or not any(any(p[k-1][0] != p[k+1][0] and p[k-1][1] != p[k+1][1]
                                      for k in range(1, len(p)-1)) for p in paths)
                       or not any(len({labels[b] for b, _ in previous}) == 2
                                  for previous in incoming.values())):
            continue
        nr = max(v[0] for e in lattice_edges for v in e)+1
        nc = max(v[1] for e in lattice_edges for v in e)+1
        if witness == "ports":
            rows = [2, int(rng.integers(13, 15)), 20, 26]
            cols = [2, int(rng.integers(11, 13)), 18, 26]
        elif witness in ("nested", "transfer") or fallback:
            rows = ([2, int(rng.integers(8, 11)), int(rng.integers(17, 21))]
                    if nr == 3 else [2, int(rng.integers(8, 10)),
                                     int(rng.integers(15, 18)), int(rng.integers(24, 27))])
            cols = [2, 8, 14, 20, 26]
        else:
            rows = [2] + list(2+np.cumsum(rng.integers(6, 9, size=nr-1)))
            cols = [2] + list(2+np.cumsum(rng.integers(6, 9, size=nc-1)))
        centers = {v: (int(rows[v[0]]), int(cols[v[1]])) for v in vertices}
        shape = int(rows[-1])+3, int(cols[-1])+3
        inp = np.zeros(shape, dtype=np.int8)
        object_cells, boxes = {}, {}
        for v in vertices:
            r, c = centers[v]
            if v in panels:
                h, w = map(int, rng.integers(3, 6, size=2))
                if witness in ("ports", "nested"):
                    h, w = ((3, 3) if v == (0, 0) else (5, 5))
                r0, c0 = r-h//2, c-w//2
            else:
                h = w = 3
                r0, c0 = r-1, c-1
            boxes[v] = (r0, c0, h, w)
            object_cells[v] = {(rr, cc) for rr in range(r0, r0+h) for cc in range(c0, c0+w)}
        occupied = set().union(*object_cells.values())
        wire_cells = []
        for path in paths:
            pixels = set()
            for u, v in zip(path, path[1:]):
                a = (int(rows[u[0]]), int(cols[u[1]]))
                b = (int(rows[v[0]]), int(cols[v[1]]))
                pixels.update(_segment(a, b))
            wire_cells.append(pixels - occupied)
        for cells in wire_cells:
            for p in cells:
                inp[p] = scaffold
        for v in vertices:
            r0, c0, h, w = boxes[v]
            if v in panels:
                contacts = set()
                if witness in ("ports", "nested"):
                    for path, pixels in zip(paths, wire_cells):
                        if v not in (path[0], path[-1]):
                            continue
                        for p in pixels:
                            contacts.update(q for q in _neighbors(p, shape) if q in object_cells[v])
                forced = [(r-r0, c-c0) for r, c in sorted(contacts)]
                keep_majority = []
                if witness == "nested":
                    keep_majority = forced[1:]
                    forced = forced[:1]
                majority = sources[ids[v]]
                minority = s1 if majority == s0 else s0
                cap = 4 if witness == "ports" and v != (0, 0) else None
                inp[r0:r0+h, c0:c0+w] = _pattern(rng, h, w, majority, minority,
                                                forced, keep_majority, cap)
            else:
                inp[r0:r0+3, c0:c0+3] = scaffold
                inp[r0+1, c0+1] = 0
        # Construction labels use only the abstract graph. Pixel reconstruction
        # below verifies edges, first-arrival levels, and the entire result.
        out = inp.copy()
        for v in vertices:
            if v not in panels:
                out[centers[v]] = labels[ids[v]]
        for cells, (u, v) in zip(wire_cells, edges):
            if dist[u] != dist[v]:
                upstream = u if dist[u] < dist[v] else v
                for p in cells:
                    out[p] = labels[upstream]
        try:
            data = _parse(inp)
            assert len(data["objects"]) == len(vertices)
            assert len(data["edges"]) == len(edges)
            assert np.array_equal(_paint(data), out)
        except (ValueError, AssertionError):
            continue
        # Transform the complete scene; padding changes neither graph nor votes.
        k = int(rng.integers(4))
        inp, out = np.rot90(inp, k), np.rot90(out, k)
        if rng.random() < .5:
            inp, out = inp[:, ::-1], out[:, ::-1]
        pads = []
        for n in inp.shape:
            total = int(rng.integers(min(3, 30-n)+1))
            before = int(rng.integers(total+1))
            pads.append((before, total-before))
        inp, out = np.pad(inp, pads), np.pad(out, pads)
        return {"input": inp.copy(), "output": out.copy()}
    raise RuntimeError("Bounded construction exhausted")


def _check_counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not low <= value <= high:
            raise ValueError("Counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(np.arange(1, 10), 3, replace=False)))
    fresh_palette = bool(rng.integers(2))
    witness_slots = list(map(int, rng.choice(num_train, 2, replace=False)))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", num_train), ("test", num_test)):
        for index in range(count):
            roles = (tuple(map(int, rng.choice(np.arange(1, 10), 3, replace=False)))
                     if fresh_palette else palette)
            for _ in range(12):
                witness = (("ports" if index == witness_slots[0] else "nested"
                            if index == witness_slots[1] else None) if split == "train"
                           else "transfer" if rng.random() < .4 else None)
                pair = _construct(rng, roles, witness=witness,
                                  harder=split == "test", invert=bool(rng.integers(2)))
                key = _key(pair["input"])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Distinct-input retry bound exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _nearest_panel_ablation(data):
    """One vote per nearest source identity, discarding repeated incoming ports."""
    n = len(data["objects"])
    votes = [[] for _ in range(n)]
    nearest = [float("inf")] * n
    for source, color in data["sources"].items():
        distances = [float("inf")] * n
        distances[source] = 0
        queue = deque([source])
        while queue:
            a = queue.popleft()
            for b, _ in data["adjacency"][a]:
                if b in data["sources"] or distances[b] != float("inf"):
                    continue
                distances[b] = distances[a]+1
                queue.append(b)
        for a, d in enumerate(distances):
            if d < nearest[a]:
                nearest[a], votes[a] = d, [color]
            elif d == nearest[a] and d != float("inf"):
                votes[a].append(color)
    result = {}
    for a in range(len(data["centers"])):
        counts = Counter(votes[a])
        best = max(counts.values())
        result[a] = next(iter(counts)) if len(counts) == 1 else (
            max(counts, key=counts.get) if list(counts.values()).count(best) == 1 else 0)
    return result


def _path_count_ablation(data):
    counts = {i: Counter({color: 1}) for i, color in data["sources"].items()}
    for a in sorted(data["incoming"], key=lambda i: (data["distance"][i], i)):
        counts[a] = Counter()
        for b, _ in data["incoming"][a]:
            counts[a].update(counts[b])
    result = {}
    for a in data["incoming"]:
        maximum = max(counts[a].values())
        winners = [c for c, count in counts[a].items() if count == maximum]
        result[a] = winners[0] if len(winners) == 1 else 0
    return result


def features(input_grid):
    d = _parse(input_grid)
    n = len(d["objects"])
    try:
        _, physical, _ = _propagate(n, d["edges"], d["sources"],
                                    [len(cells)+1 for cells in d["wires"]])
        physical_conflict = any(physical[i] != d["labels"][i] for i in range(len(d["centers"])))
    except ValueError:
        physical_conflict = False  # An ambiguous ablation is not a counterexample.
    distinct = _nearest_panel_ablation(d)
    paths = _path_count_ablation(d)
    try:
        _, contact_labels, _ = _propagate(n, d["edges"], d["sources"], emissions=d["contacts"])
    except ValueError:
        contact_labels = None  # Do not count an ambiguous alternative as a win.
    semantic = np.zeros(d["grid"].shape, dtype=np.int8)
    semantic[d["grid"] == d["scaffold"]] = 1
    for role, color in enumerate(d["signals"], 2):
        semantic[d["grid"] == color] = role
    fg = np.argwhere(semantic)
    lo, hi = fg.min(axis=0), fg.max(axis=0)
    normalized = semantic[lo[0]:hi[0]+1, lo[1]:hi[1]+1]
    return dict(shape=d["grid"].shape, relays=len(d["centers"]), panels=len(d["sources"]),
                edges=len(d["edges"]), loops=len(d["edges"])-n+1,
                max_level=max(d["distance"]),
                max_incoming=max(map(len, d["incoming"].values())),
                opposed_elections=sum(len({d["labels"][b] for b, _ in previous}) == 2
                                      for previous in d["incoming"].values()),
                bent_wires=sum(len({r for r, c in cells}) > 1 and len({c for r, c in cells}) > 1
                               for cells in d["wires"]),
                equal_level_edges=sum(d["distance"][u] == d["distance"][v] for u, v in d["edges"]),
                physical_conflict=physical_conflict,
                port_conflict=any(distinct[i] != d["labels"][i] for i in distinct),
                nested_conflict=any(paths[i] != d["labels"][i] for i in paths),
                local_conflict=contact_labels is not None and any(
                    contact_labels[i] != d["labels"][i] for i in range(len(d["centers"]))),
                structure=_key(normalized), palette=(d["scaffold"], *d["signals"]))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, witnesses, nested_witnesses = set(), [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            d = _parse(inp)
            assert np.array_equal(out, _paint(d))
            assert not np.array_equal(inp, out)
            # Explicit preservation and complete-wire invariants supplement
            # whole-output equality; panel minority pixels are never repaired.
            for i in d["sources"]:
                assert all(out[p] == inp[p] for p in d["objects"][i])
            centers = set(d["centers"])
            for i in range(len(d["centers"])):
                assert all(out[p] == inp[p] for p in d["objects"][i] - centers)
            assert all(out[p] == 0 for p in map(tuple, np.argwhere(inp == 0)) if p not in centers)
            for cells, (u, v) in zip(d["wires"], d["edges"]):
                if d["distance"][u] == d["distance"][v]:
                    assert all(out[p] == d["scaffold"] for p in cells)
                else:
                    upstream = u if d["distance"][u] < d["distance"][v] else v
                    assert all(out[p] == d["labels"][upstream] for p in cells)
            if split == "train":
                f = features(inp)
                witnesses.append(f["physical_conflict"] and f["port_conflict"] and f["local_conflict"])
                nested_witnesses.append(f["nested_conflict"])
            else:
                assert max(d["distance"]) >= 3
                assert len(d["edges"]) - len(d["objects"]) + 1 >= 1
                assert any(len({d["labels"][b] for b, _ in previous}) == 2
                           for previous in d["incoming"].values())
                assert any(len({r for r, c in cells}) > 1 and len({c for r, c in cells}) > 1
                           for cells in d["wires"])
    assert any(witnesses), "Training must distinguish physical length, source counts, and contact sampling"
    assert any(nested_witnesses), "Training must distinguish local elections from total shortest-path counts"


def self_test():
    """Bounded: 210 main pairs across 42 seeds, count checks, and mutations.

    No files, logs, caches, or saved execution traces are produced.
    """
    structures, palettes, shapes, panel_counts, levels, loops, indegrees = set(), set(), set(), set(), set(), set(), set()
    equal_edges = nested_tests = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        for split in ("train", "test"):
            for pair in p[split]:
                assert np.array_equal(solve(pair["input"]), pair["output"])
                f = features(pair["input"])
                structures.add(f["structure"])
                palettes.add(f["palette"])
                shapes.add(f["shape"])
                panel_counts.add(f["panels"])
                levels.add(f["max_level"])
                loops.add(f["loops"])
                indegrees.add(f["max_incoming"])
                equal_edges += f["equal_level_edges"]
                nested_tests += split == "test" and f["nested_conflict"]
        if seed < 3:
            q = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
        # Palette permutations leave all geometry and causal decisions intact.
        pair = p["test"][0]
        lut = np.array([0, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int8)
        assert np.array_equal(solve(lut[pair["input"]]), lut[pair["output"]])
        assert np.array_equal(solve(np.rot90(pair["input"])), np.rot90(pair["output"]))
    assert len(structures) >= 180 and len(shapes) >= 35 and len(palettes) >= 25
    assert {2, 3, 4} <= panel_counts and max(levels) >= 4
    assert {0, 1, 2} <= loops and {1, 2, 3} <= indegrees and equal_edges >= 42
    assert nested_tests >= 6
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(12000+nt*10+ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError("Illegal counts accepted")
    # A relay tie is rejected; no arbitrary traversal or color order decides it.
    try:
        _propagate(3, [(0, 2), (1, 2)], {0: 1, 1: 2})
    except ValueError:
        pass
    else:
        raise AssertionError("Tied first-arrival votes accepted")
    p = generate_puzzle(41, 2, 1)
    d = _parse(p["test"][0]["input"])
    p["test"][0]["output"][d["centers"][0]] = 0
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted relay center accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed: 42 seeds, all 30 legal count combinations, metamorphic and mutation checks")
