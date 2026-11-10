"""Station-distance paint watersheds. No files, seeds or labels are used by solve.

Objects, rather than wire pixels, are the units of travel. Every wall object is
one station, including hollow rooms and solid relays. A room takes the nearest
reservoir's paint; equally near different reservoirs leave it empty.
"""
from collections import deque
from itertools import combinations
import hashlib
import heapq
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
AUDIT_SEEDS = (0, 682080286, 424808665, 196318031, 772732116, 174947549,
               360031335, 775831681, 35579297, 398282911, 800878768,
               579108234, 247258859, 892054648, 488832717, 643603722,
               909741655, 523300003, 436117247, 865371804, 430439119,
               844487974, 43825636, 279230096, 857361470, 745237719,
               433146192, 769169542, 682313396, 327881989, 721936909,
               268674109, 515015161, 976822820, 141112462, 57384491,
               875802742, 564736203, 305750869, 272708450, 312322963,
               752133328)


def _components(cells, steps=D8):
    left = set(cells)
    result = []
    while left:
        p = min(left)
        left.remove(p)
        group, todo = {p}, [p]
        for r, c in todo:
            for dr, dc in steps:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    group.add(q)
                    todo.append(q)
        result.append(group)
    return result


def _neighbors(cells, steps=D8):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps} - cells


def _holes(wall):
    """Find cavities of this wall alone, ignoring unrelated network faces."""
    r0, r1 = min(r for r, c in wall) - 1, max(r for r, c in wall) + 1
    c0, c1 = min(c for r, c in wall) - 1, max(c for r, c in wall) + 1
    space = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)} - wall
    return [g for g in _components(space, D4)
            if not any(r in (r0, r1) or c in (c0, c1) for r, c in g)]


def _solid(wall):
    rs, cs = zip(*wall)
    h, w = max(rs) - min(rs) + 1, max(cs) - min(cs) + 1
    return h == w == 3 and len(wall) == 9


def _read(input_grid):
    a = np.asarray(input_grid)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid")
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert sum(counts == counts[bg]) == 1, "background must be uniquely dominant"
    candidates = []
    for col in np.flatnonzero(counts):
        if col == bg:
            continue
        groups = _components(map(tuple, np.argwhere(a == col)))
        if len(groups) < 6:
            continue
        holes = [_holes(g) for g in groups]
        if all(len(h) == 1 or (not h and _solid(g)) for g, h in zip(groups, holes)):
            candidates.append((int(col), groups, holes))
    assert len(candidates) == 1, "one recognizable station-wall color required"
    wall_color, walls, holes = candidates[0]
    interiors = [h[0] if h else set() for h in holes]
    sources, rooms, relays = {}, [], []
    for i, inner in enumerate(interiors):
        if not inner:
            relays.append(i)
            continue
        colors = {int(a[p]) for p in inner}
        assert len(colors) == 1, "room interiors must be uniform"
        color = next(iter(colors))
        if color == bg:
            rooms.append(i)
        else:
            assert color != wall_color
            sources[i] = color
    assert 2 <= len(sources) <= 3 and len(set(sources.values())) == len(sources)
    wire_colors = set(int(c) for c in np.flatnonzero(counts)) - {bg, wall_color} - set(sources.values())
    assert len(wire_colors) == 1, "one wire role required"
    wire = wire_colors.pop()
    assert all(a[p] != wire for inner in interiors for p in inner)
    assert relays and rooms
    lookup = {p: i for i, cells in enumerate(walls) for p in cells}
    edge_cells = _components(map(tuple, np.argwhere(a == wire)))
    edges, adj = [], [[] for _ in walls]
    for e, cells in enumerate(edge_cells):
        attached = {lookup[p] for p in _neighbors(cells) if p in lookup}
        assert len(attached) == 2, "each wire must join two stations"
        u, v = sorted(attached)
        edges.append((u, v))
        adj[u].append((v, e))
        adj[v].append((u, e))
    assert len(set(edges)) == len(edges), "parallel edges are not generated"
    reached, todo = {next(iter(sources))}, [next(iter(sources))]
    for u in todo:
        for v, _ in adj[u]:
            if v not in reached:
                reached.add(v)
                todo.append(v)
    assert set(sources) <= reached
    assert all(i in reached or (not adj[i] and i in rooms) for i in range(len(walls)))
    assert sum(not n for n in adj) <= 2
    occupied = set(lookup) | set().union(*interiors) | set().union(*edge_cells)
    assert occupied == set(map(tuple, np.argwhere(a != bg))) | set().union(*interiors)
    assert all(len(n) <= 4 for n in adj)
    return dict(a=a, bg=bg, wall=wall_color, wire=wire, walls=walls,
                interiors=interiors, sources=sources, rooms=rooms, relays=relays,
                edge_cells=edge_cells, edges=edges, adj=adj)


def _distances(adj, start, weights=None):
    distances = [float("inf")] * len(adj)
    distances[start] = 0
    if weights is None:
        todo = deque([start])
        while todo:
            u = todo.popleft()
            for v, _ in adj[u]:
                if distances[v] == float("inf"):
                    distances[v] = distances[u] + 1
                    todo.append(v)
    else:
        todo = [(0, start)]
        while todo:
            cost, u = heapq.heappop(todo)
            if cost != distances[u]:
                continue
            for v, e in adj[u]:
                new = cost + weights[e]
                if new < distances[v]:
                    distances[v] = new
                    heapq.heappush(todo, (new, v))
    return distances


def _route_multiplicity(adj, start):
    """Count shortest routes up to two; source ties are a separate question."""
    distance = _distances(adj, start)
    ways = [0] * len(adj)
    ways[start] = 1
    for u in sorted(range(len(adj)), key=lambda v: distance[v]):
        if distance[u] == float("inf"):
            continue
        for v, _ in adj[u]:
            if distance[v] == distance[u] + 1:
                ways[v] = min(2, ways[v] + ways[u])
    return ways


def _owners(data, sources=None, weighted=False):
    sources = data['sources'] if sources is None else sources
    weights = [len(c) for c in data['edge_cells']] if weighted else None
    distances = {s: _distances(data['adj'], s, weights) for s in sources}
    result = []
    for v in range(len(data['walls'])):
        best = min(d[v] for d in distances.values())
        winners = tuple(s for s in sources if distances[s][v] == best)
        result.append(winners if best != float("inf") else ())
    return result, distances


def _ignore_station_owners(data, sources, ignored):
    """Counterfactual for validation: ignored stations consume zero stops."""
    distances = {}
    for s in sources:
        dist = [float('inf')] * len(data['walls'])
        dist[s] = 0
        queue = [(0, s)]
        while queue:
            cost, u = heapq.heappop(queue)
            if cost != dist[u]:
                continue
            for v, _ in data['adj'][u]:
                candidate = cost + (v not in ignored)
                if candidate < dist[v]:
                    dist[v] = candidate
                    heapq.heappush(queue, (candidate, v))
        distances[s] = dist
    result = []
    for v in range(len(data['walls'])):
        best = min(x[v] for x in distances.values())
        result.append(tuple(s for s in sources if distances[s][v] == best)
                      if best != float('inf') else ())
    return result


def _different_fill(a, b):
    return a != b and (len(a) == 1 or len(b) == 1)


def _shortcut_evidence(data, sources, owners, blank):
    centers = [np.mean(list(w), axis=0) for w in data['walls']]
    spatial = []
    for norm in (1, 2):
        changed = 0
        for v in blank:
            costs = {s: float(np.linalg.norm(centers[v] - centers[s], ord=norm)) for s in sources}
            best = min(costs.values())
            near = tuple(s for s in sources if abs(costs[s] - best) < 1e-9)
            changed += int(len(near) == len(owners[v]) == 1 and near != owners[v])
        spatial.append(changed)
    ignore_relay = _ignore_station_owners(data, sources, set(data['relays']))
    ignore_hollow = _ignore_station_owners(data, sources,
                                          {i for i, inner in enumerate(data['interiors']) if inner})
    return dict(manhattan_conflicts=spatial[0], euclidean_conflicts=spatial[1],
                ignoring_relays_changes=sum(_different_fill(owners[v], ignore_relay[v]) for v in blank),
                ignoring_hollow_changes=sum(_different_fill(owners[v], ignore_hollow[v]) for v in blank))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Copy only uniquely nearest reservoir colors into blank chamber interiors."""
    data = _read(input_grid)
    owners, _ = _owners(data)
    out = np.array(input_grid, copy=True)
    for v in data['rooms']:
        if len(owners[v]) == 1:
            color = data['sources'][owners[v][0]]
            for p in data['interiors'][v]:
                out[p] = color
    return out


def _ri(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def _sample_lattice(rng, large, fallback=0):
    if fallback == 1:
        # Seven visible stops; two very long edges precede four short ones.
        path = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3),
                (2, 2), (2, 1), (1, 1), (1, 0)]
        visible = {(0, 0), (0, 3), (2, 3), (2, 2), (2, 1), (1, 1), (1, 0)}
        return 3, 4, set(path), list(zip(path, path[1:])), visible
    if fallback == 2:
        # A four-station loop with a four-edge tail. A source on the loop
        # reaches its opposite room by two equal routes; the remote source
        # still creates a neutral room and its own productive territory.
        loop = [(0, 1), (0, 0), (1, 0), (1, 1)]
        tail = [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]
        active = set(loop + tail)
        edges = list(zip(loop, loop[1:] + loop[:1])) + list(zip(tail, tail[1:]))
        return 3, 4, active, edges, active
    nr, nc = (4, 4) if large else (3, _ri(rng, 3, 4))
    vertices = [(r, c) for r in range(nr) for c in range(nc)]
    potential = [(a, b) for a, b in combinations(vertices, 2)
                 if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
    active = {vertices[_ri(rng, 0, len(vertices) - 1)]}
    edges = []
    target = _ri(rng, 12 if large else 8, len(vertices))
    while len(active) < target:
        choices = [(a, b) for a, b in potential if (a in active) != (b in active)]
        a, b = choices[_ri(rng, 0, len(choices) - 1)]
        active.update((a, b))
        edges.append((a, b))
    extras = [(a, b) for a, b in potential if a in active and b in active and (a, b) not in edges]
    rng.shuffle(extras)
    edges.extend(extras[:_ri(rng, 0, min(4, len(extras)))])
    degree = {v: sum(v in e for e in edges) for v in active}
    visible = {v for v in active if degree[v] != 2 or rng.random() > .5}
    missing = sorted(active - visible)
    rng.shuffle(missing)
    while len(visible) < (10 if large else 7):
        visible.add(missing.pop())
    return nr, nc, active, edges, visible


def _shape(rng, relay):
    if relay:
        return {(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)}, set()
    kind = _ri(rng, 0, 3)
    if kind <= 1:
        h, w = _ri(rng, 1, 3), _ri(rng, 1, 3)
        inner = {(r, c) for r in range(-(h // 2), h - h // 2)
                 for c in range(-(w // 2), w - w // 2)}
        wall = _neighbors(inner)  # Full rectangular rim, including corners.
    elif kind == 2:
        inner = {(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)}
        wall = _neighbors(inner, D4)
    else:
        inner = {(-1, -1), (-1, 0), (0, -1), (0, 0), (0, 1), (1, 0), (1, 1)}
        wall = _neighbors(inner, D4)
        if rng.random() < .5:
            inner = {(r, -c) for r, c in inner}
            wall = {(r, -c) for r, c in wall}
    return wall, inner


def _render(rng, lattice, fallback=0):
    nr, nc, active, segments, visible = lattice
    # Centers retain enough clearance for all 3..5 cell chamber footprints.
    def axis(n):
        gaps = [7] * (n - 1) if fallback else [_ri(rng, 7, 8) for _ in range(n - 1)]
        return [2] + [2 + sum(gaps[:i]) for i in range(1, n)]
    rows, cols = axis(nr), axis(nc)
    grid = np.zeros((rows[-1] + 3, cols[-1] + 3), dtype=np.int8)
    for a, b in segments:
        ra, ca, rb, cb = rows[a[0]], cols[a[1]], rows[b[0]], cols[b[1]]
        if ra == rb:
            grid[ra, min(ca, cb):max(ca, cb) + 1] = 1
        else:
            grid[min(ra, rb):max(ra, rb) + 1, ca] = 1
    ordered = sorted(visible)
    # At least five hollow stations and at least one solid relay.
    nrelay = _ri(rng, 1, max(1, len(ordered) - 5))
    relay_ids = set(int(x) for x in rng.choice(len(ordered), nrelay, replace=False))
    if fallback:
        relay_ids = {1} if fallback == 1 else {0}
    objects = []
    for i, v in enumerate(ordered):
        wall, inner = _shape(rng, i in relay_ids)
        center = rows[v[0]], cols[v[1]]
        wall = {(r + center[0], c + center[1]) for r, c in wall}
        inner = {(r + center[0], c + center[1]) for r, c in inner}
        for p in wall:
            grid[p] = 2
        for p in inner:
            grid[p] = 0
        objects.append((v, wall, inner))
    # Empty rooms may be completely unconnected, so emptiness is not always a tie.
    unused = sorted({(r, c) for r in range(nr) for c in range(nc)} - active)
    rng.shuffle(unused)
    for v in unused[:_ri(rng, 0, min(2, len(unused)))]:
        wall, inner = _shape(rng, False)
        center = rows[v[0]], cols[v[1]]
        wall = {(r + center[0], c + center[1]) for r, c in wall}
        inner = {(r + center[0], c + center[1]) for r, c in inner}
        for p in wall:
            grid[p] = 2
        objects.append((v, wall, inner))
    # Recover every edge from pixels before choosing reservoir locations.
    walls = [x[1] for x in objects]
    interiors = [x[2] for x in objects]
    lookup = {p: i for i, w in enumerate(walls) for p in w}
    edge_cells = _components(map(tuple, np.argwhere(grid == 1)))
    edges, adj = [], [[] for _ in walls]
    for e, cells in enumerate(edge_cells):
        ends = {lookup[p] for p in _neighbors(cells) if p in lookup}
        assert len(ends) == 2
        u, v = sorted(ends)
        edges.append((u, v))
        adj[u].append((v, e))
        adj[v].append((u, e))
    assert len(set(edges)) == len(edges)
    # Compare with the intended lattice after suppressing bend-only vertices.
    original = {v: [] for v in active}
    for u, v in segments:
        original[u].append(v)
        original[v].append(u)
    ids = {v: i for i, (v, _, _) in enumerate(objects)}
    expected = set()
    incidences = 0
    for u in visible:
        for v in original[u]:
            prev, cur = u, v
            while cur not in visible:
                nxt = [x for x in original[cur] if x != prev]
                assert len(nxt) == 1
                prev, cur = cur, nxt[0]
            assert u != cur
            expected.add(tuple(sorted((ids[u], ids[cur]))))
            incidences += 1
    assert incidences == 2 * len(expected) and set(edges) == expected
    return grid, dict(walls=walls, interiors=interiors, edges=edges,
                      edge_cells=edge_cells, adj=adj,
                      relays=[i for i, x in enumerate(interiors) if not x],
                      rooms=[i for i, x in enumerate(interiors) if x])


def _choose_sources(rng, data, nsource, witness, cyclic):
    connected = [i for i in data['rooms'] if data['adj'][i]]
    loop_rank = len(data['edges']) - sum(bool(x) for x in data['adj']) + 1
    if cyclic and loop_rank < 1:
        return None
    choices = list(combinations(connected, nsource))
    rng.shuffle(choices)
    for src in choices:
        owners, distances = _owners(data, {s: 0 for s in src})
        blank = [i for i in data['rooms'] if i not in src]
        productive = {owners[i][0] for i in blank if len(owners[i]) == 1}
        ties = [i for i in blank if len(owners[i]) > 1]
        if len(productive) < 2 or not ties:
            continue
        # A third reservoir must influence some answer, even if it only
        # prevents another color from filling an equal-distance chamber.
        if any(s not in productive and not any(len(owners[v]) == 2 and s in owners[v]
                                               for v in blank) for s in src):
            continue
        # Even a two-demonstration task teaches propagation beyond a neighbor.
        if (witness or cyclic) and not any(
                len(owners[i]) == 1 and distances[owners[i][0]][i] >= 2 for i in blank):
            continue
        if cyclic:
            ways = {s: _route_multiplicity(data['adj'], s) for s in src}
            if not any(len(owners[i]) == 1 and ways[owners[i][0]][i] == 2 for i in blank):
                continue
        if witness:
            weighted, _ = _owners(data, {s: 0 for s in src}, weighted=True)
            if not any(len(owners[i]) == len(weighted[i]) == 1
                       and distances[owners[i][0]][i] >= 2
                       and owners[i] != weighted[i] for i in blank):
                continue
            evidence = _shortcut_evidence(data, src, owners, blank)
            if not all(evidence.values()):
                continue
        return src
    return None


def _pair(rng, nsource, witness=False, cyclic=False, large=False):
    for attempt in range(161):
        fallback = (1 if witness else 2) if attempt == 160 else 0
        try:
            lattice = _sample_lattice(rng, large, fallback)
            base, data = _render(rng, lattice, fallback)
            sources = _choose_sources(rng, data, nsource, witness, cyclic)
            if sources is None:
                continue
            # Independent role colors; no output-only or learned palette exists.
            palette = [int(c) for c in rng.permutation(10)]
            bg, wire, wall = palette[:3]
            a = np.array([bg, wire, wall], dtype=np.int8)[base]
            for s, color in zip(sources, palette[3:3 + nsource]):
                for p in data['interiors'][s]:
                    a[p] = color
            # Construct labels from the intended station graph, before solve.
            owners, _ = _owners(data, dict.fromkeys(sources, 0))
            color_for = dict(zip(sources, palette[3:3 + nsource]))
            b = a.copy()
            for v in data['rooms']:
                if v not in sources and len(owners[v]) == 1:
                    for p in data['interiors'][v]:
                        b[p] = color_for[owners[v][0]]
            rotation = _ri(rng, 0, 3)
            a, b = np.rot90(a, rotation), np.rot90(b, rotation)
            if rng.random() < .5:
                a, b = a[:, ::-1], b[:, ::-1]
            # Placement and margins are per example, with a hard 30-cell bound.
            h, w = a.shape
            hh, ww = _ri(rng, h, min(30, h + 3)), _ri(rng, w, min(30, w + 3))
            top, left = _ri(rng, 0, hh - h), _ri(rng, 0, ww - w)
            aa, bb = np.full((hh, ww), bg, dtype=np.int8), np.full((hh, ww), bg, dtype=np.int8)
            aa[top:top + h, left:left + w] = a
            bb[top:top + h, left:left + w] = b
            assert np.array_equal(solve(aa), bb)
            return {'input': aa, 'output': bb}
        except AssertionError:
            if fallback:
                raise
    raise AssertionError("bounded construction exhausted")


def _check_counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    # Test shift: richer source competition or larger station graph, sometimes both.
    shift = _ri(rng, 0, 2)
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            nsource = 3 if (split == 'test' and shift != 1) or (split == 'train' and i >= 2 and rng.random() < .35) else 2
            witness = split == 'train' and i == 0
            cyclic = split == 'train' and i == 1
            for _ in range(8):
                pair = _pair(rng, nsource, witness, cyclic,
                             large=split == 'test' or i >= 2)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError("distinct-input retries exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    d = _read(pair['input'])
    owners, distances = _owners(d)
    weighted, _ = _owners(d, weighted=True)
    blank = d['rooms']
    ties = [v for v in blank if len(owners[v]) > 1]
    inactive = [v for v in blank if not owners[v]]
    winners = [v for v in blank if len(owners[v]) == 1]
    conflict = [v for v in winners if len(weighted[v]) == 1 and owners[v] != weighted[v]]
    ways = {s: _route_multiplicity(d['adj'], s) for s in d['sources']}
    multiple = sum(ways[owners[v][0]][v] == 2 for v in winners)
    shortcuts = _shortcut_evidence(d, d['sources'], owners, blank)
    degrees = [len(x) for x in d['adj']]
    # Main connected component plus any isolated empty rooms.
    rank = len(d['edges']) - sum(x > 0 for x in degrees) + 1
    return dict(**shortcuts, shape=list(d['a'].shape), stations=len(degrees), edges=len(d['edges']),
                sources=len(d['sources']), chambers=len(blank), relays=len(d['relays']),
                ties=len(ties), inactive=len(inactive), filled=len(winners),
                filled_colors=len({d['sources'][owners[v][0]] for v in winners}),
                pixel_conflicts=len(conflict), cycle_rank=rank,
                multiple_routes_filled=multiple,
                junctions=sum(x >= 3 for x in degrees), dead_ends=degrees.count(1),
                max_distance=max(min(ds[v] for ds in distances.values()) for v in winners),
                cavity_shapes=sorted({tuple(sorted((r - min(x for x, y in cells),
                                                   c - min(y for x, y in cells)) for r, c in cells))
                                      for cells in d['interiors'] if cells}),
                background=d['bg'], wall=d['wall'], wire=d['wire'],
                paint_colors=sorted(d['sources'].values()))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, stats = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = np.asarray(pair['input']), np.asarray(pair['output'])
            for arr in (a, b):
                assert arr.ndim == 2 and all(1 <= n <= 30 for n in arr.shape)
                assert np.issubdtype(arr.dtype, np.integer) and np.all((0 <= arr) & (arr <= 9))
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            d = _read(a)
            owners, distances = _owners(d)
            expected = a.copy()
            for v in d['rooms']:
                cells = d['interiors'][v]
                values = {int(b[p]) for p in cells}
                if len(owners[v]) == 1:
                    s = owners[v][0]
                    assert values == {d['sources'][s]}
                    # Every other source is strictly farther away in station hops.
                    assert all(distances[s][v] < distances[t][v] for t in d['sources'] if t != s)
                    for p in cells:
                        expected[p] = d['sources'][s]
                else:
                    assert values == {d['bg']}
            assert np.array_equal(b, expected) and np.array_equal(solve(a), b)
            assert np.all(a[a != b] == d['bg'])
            assert all(np.array_equal(a[tuple(np.array(sorted(w)).T)], b[tuple(np.array(sorted(w)).T)])
                       for w in d['walls'])
            f = _features(pair)
            assert f['filled_colors'] >= 2 and f['ties'] >= 1
            stats[split].append(f)
    assert any(f['pixel_conflicts'] for f in stats['train']), "teach station count over wire length"
    for key in ('manhattan_conflicts', 'euclidean_conflicts',
                'ignoring_relays_changes', 'ignoring_hollow_changes'):
        assert stats['train'][0][key], f"first lesson must distinguish {key}"
    assert any(f['cycle_rank'] >= 1 for f in stats['train']), "teach cycles and unchanged network faces"
    assert any(f['multiple_routes_filled'] for f in stats['train']), "multiple paths from one source do not cancel"
    assert all(f['max_distance'] >= 2 for f in stats['train'][:2])
    return True


def self_test():
    """Bounded, streamed audit; retain aggregate counts, never 42 full puzzles."""
    shapes, structures, source_counts, sizes, topologies = set(), set(), set(), set(), set()
    paints, backgrounds, walls, wires = set(), set(), set(), set()
    totals = dict(pairs=0, conflicts=0, ties=0, inactive=0, cyclic=0, junctions=0)
    for variant, ns, witness, cyclic in ((1, 2, True, False), (2, 2, False, True), (2, 3, False, False)):
        rng = np.random.default_rng(100 + variant + ns)
        lattice = _sample_lattice(rng, False, variant)
        base, data = _render(rng, lattice, variant)
        src = _choose_sources(rng, data, ns, witness, cyclic)
        assert src is not None, 'constructive fallback must retain its teaching conditions'
        a = base.copy()
        for s, color in zip(src, range(3, 3 + ns)):
            for loc in data['interiors'][s]:
                a[loc] = color
        parsed = _read(a)
        ownership, _ = _owners(data, dict.fromkeys(src, 0))
        expected = a.copy()
        colors = dict(zip(src, range(3, 3 + ns)))
        for v in data['rooms']:
            if v not in src and len(ownership[v]) == 1:
                for loc in data['interiors'][v]:
                    expected[loc] = colors[ownership[v][0]]
        assert np.array_equal(solve(a), expected)
    for seed in AUDIT_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        # Floyd-Warshall independently checks the BFS distances on one drawing
        # per seed. This is an authored consistency check, not external review.
        d = _read(p['train'][0]['input'])
        n = len(d['walls'])
        matrix = np.full((n, n), 1000, dtype=int)
        np.fill_diagonal(matrix, 0)
        for u, v in d['edges']:
            matrix[u, v] = matrix[v, u] = 1
        for k in range(n):
            matrix = np.minimum(matrix, matrix[:, k, None] + matrix[None, k, :])
        for s in d['sources']:
            assert list(np.minimum(_distances(d['adj'], s), 1000)) == matrix[s].tolist()
        for split in ('train', 'test'):
            for pair in p[split]:
                f = _features(pair)
                totals['pairs'] += 1
                for key, field in (('conflicts', 'pixel_conflicts'), ('ties', 'ties'),
                                   ('inactive', 'inactive'), ('cyclic', 'cycle_rank'),
                                   ('junctions', 'junctions')):
                    totals[key] += int(f[field] > 0)
                shapes.add(tuple(f['shape']))
                source_counts.add(f['sources'])
                sizes.add(f['stations'])
                topologies.add((f['stations'], f['edges'], f['junctions'], f['dead_ends']))
                paints.update(f['paint_colors'])
                backgrounds.add(f['background']); walls.add(f['wall']); wires.add(f['wire'])
                a = pair['input']
                mapping, normalized = {}, []
                for x in a.flat:
                    normalized.append(mapping.setdefault(int(x), len(mapping)))
                # Small immutable signatures, not reconstructible grid payloads.
                structures.add(hashlib.sha256(bytes(normalized)).digest())
    assert totals['pairs'] == 210 and totals['conflicts'] >= 42 and totals['ties'] == 210
    assert totals['inactive'] >= 5 and totals['cyclic'] >= 42 and totals['junctions'] >= 42
    assert len(shapes) >= 20 and len(structures) >= 180 and len(sizes) >= 6
    assert len(topologies) >= 25 and source_counts == {2, 3}
    assert min(len(x) for x in (paints, backgrounds, walls, wires)) >= 8
    p, q = generate_puzzle(0, 2, 1), generate_puzzle(0, 2, 1)
    assert all(_key(a[k]) == _key(b[k]) for s in p for a, b in zip(p[s], q[s]) for k in ('input', 'output'))
    # Spatial and palette transformations preserve the input-only semantics.
    query = p['test'][0]
    for k in range(4):
        assert np.array_equal(solve(np.rot90(query['input'], k)), np.rot90(query['output'], k))
    permutation = np.array([7, 4, 8, 0, 9, 2, 6, 1, 3, 5], dtype=np.int8)
    assert np.array_equal(solve(permutation[query['input']]), permutation[query['output']])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(100 + nt * 10 + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Validators must detect a partly unpainted chamber and a changed wall.
    for kind in ('fill', 'wall'):
        bad = {s: [{k: a.copy() for k, a in pair.items()} for pair in p[s]] for s in p}
        pair = bad['test'][0]
        d = _read(pair['input'])
        if kind == 'fill':
            loc = tuple(np.argwhere(pair['input'] != pair['output'])[0])
            pair['output'][loc] = pair['input'][loc]
        else:
            loc = min(d['walls'][0])
            pair['output'][loc] = d['bg']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupted answer accepted")
    return dict(seeds=len(AUDIT_SEEDS), **totals, grid_shapes=len(shapes),
                normalized_structures=len(structures), station_counts=sorted(sizes),
                graph_signatures=len(topologies))
