"""Toll ruler: a spatial periodic tariff on a station graph.

Only standard library and NumPy. No parent imports, file access, global RNG,
training-state lookup or learned color convention is used by solve().
"""
from __future__ import annotations

import heapq
from itertools import combinations
import json
import numpy as np

_NEIGHBORS = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(mask):
    pending = set(map(tuple, np.argwhere(mask).tolist()))
    parts = []
    while pending:
        root = min(pending)
        pending.remove(root)
        stack, cells = [root], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _NEIGHBORS:
                p = r + dr, c + dc
                if p in pending:
                    pending.remove(p)
                    stack.append(p)
        parts.append(np.array(sorted(cells), dtype=np.int16))
    return parts


def _square(cells):
    span = cells.max(0) - cells.min(0) + 1
    return int(span[0]) if span[0] == span[1] and len(cells) == int(np.prod(span)) else 0


def _adjacency(n, edges):
    result = [[] for _ in range(n)]
    for e, (a, b) in enumerate(edges):
        result[a].append((b, e))
        result[b].append((a, e))
    return result


def _best(adjacency, endpoints, tolls, areas, mode="normal"):
    """Positive lexicographic Dijkstra, with multiplicity of the optimum."""
    start, target = endpoints
    zero = (0,) if mode == "hops" else (0, 0)
    costs, ways, previous = {start: zero}, {start: 1}, {}
    queue = [(zero, start)]
    while queue:
        cost, a = heapq.heappop(queue)
        if cost != costs[a]:
            continue
        if a == target:
            break
        for b, e in adjacency[a]:
            ordinary = int(b not in endpoints)
            if mode == "hops":
                # Count edges for strict positivity, then subtract one below.
                step = (1,)
            elif mode == "pixels":
                step = (int(tolls[b]) * ordinary, int(areas[e]))
            else:
                # Count edges as second coordinate; endpoints are common to all routes.
                step = (int(tolls[b]) * ordinary, 1)
            candidate = tuple(x + y for x, y in zip(cost, step))
            if b not in costs or candidate < costs[b]:
                costs[b], ways[b], previous[b] = candidate, ways[a], (a, e)
                heapq.heappush(queue, (candidate, b))
            elif candidate == costs[b]:
                ways[b] = min(2, ways[b] + ways[a])
    if target not in costs:
        raise ValueError("disconnected endpoints")
    nodes, edges = [target], []
    while nodes[-1] != start:
        a, e = previous[nodes[-1]]
        nodes.append(a)
        edges.append(e)
    final = costs[target]
    if mode == "hops":
        final = (final[0] - 1,)
    elif mode == "normal":
        final = final[0], final[1] - 1
    return dict(nodes=nodes[::-1], edges=edges[::-1], cost=final, ways=ways[target])


def _geometry(g, ordinary, endpoint, connector):
    stations = _components((g == ordinary) | (g == endpoint))
    sizes = [_square(c) for c in stations]
    if len(stations) < 5 or len(set(sizes)) != 1 or sizes[0] < 2:
        raise ValueError("stations must be at least five equal separated squares")
    station_map = np.full(g.shape, -1, dtype=np.int16)
    endpoints = []
    for n, cells in enumerate(stations):
        colors = np.unique(g[tuple(cells.T)])
        if len(colors) != 1:
            raise ValueError("mixed station block")
        if int(colors[0]) == endpoint:
            endpoints.append(n)
        station_map[tuple(cells.T)] = n
    if len(endpoints) not in (0, 2):
        raise ValueError("expected exactly two endpoint squares")
    components = _components(g == connector)
    edges = []
    h, w = g.shape
    for cells in components:
        touching = set()
        for r, c in cells.tolist():
            for dr, dc in _NEIGHBORS:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and station_map[rr, cc] >= 0:
                    touching.add(int(station_map[rr, cc]))
        if len(touching) != 2:
            raise ValueError("each connector component must touch exactly two stations")
        edges.append(tuple(sorted(touching)))
    if len(set(edges)) != len(edges):
        raise ValueError("parallel connectors are outside this drawing convention")
    adjacency = _adjacency(len(stations), edges)
    reached, stack = {0}, [0]
    while stack:
        for b, _ in adjacency[stack.pop()]:
            if b not in reached:
                reached.add(b)
                stack.append(b)
    if len(reached) != len(stations):
        raise ValueError("disconnected station network")
    tolls = [bool(np.any(cells[:, 0] % 6 == 4)) for cells in stations]
    return dict(stations=stations, components=components, edges=edges,
                adjacency=adjacency, endpoints=endpoints, tolls=tolls,
                areas=[len(c) for c in components], size=sizes[0])


def _read(input_grid):
    original = _grid(input_grid)
    colors, counts = np.unique(original, return_counts=True)
    if len(colors) != 6 or counts.max() <= original.size / 2:
        raise ValueError("six roles and a majority background are required")
    background = int(colors[np.argmax(counts)])
    guides = []
    # Normalize with rotations only. Lateral reflection needs no special handling.
    for rotation in range(4):
        g = np.rot90(original, rotation)
        if g.shape[0] < 5:
            continue
        for seed in map(int, colors):
            if seed == background:
                continue
            cells = np.argwhere(g == seed)
            if len(cells) < 2 or len(cells) % 2:
                continue
            span = len(cells) // 2
            cc = np.flatnonzero(g[0] == seed)
            if len(cc) != span or not np.array_equal(cc, np.arange(cc[0], cc[0] + span)):
                continue
            start = int(cc[0])
            if not np.all(g[2, start:start + span] == seed):
                continue
            allowed = np.zeros_like(g, dtype=bool)
            allowed[0, start:start + span] = allowed[2, start:start + span] = True
            if not np.array_equal(g == seed, allowed):
                continue
            special = int(g[4, start])
            special_mask = np.zeros_like(allowed)
            special_mask[4, start:start + span] = True
            if special in (background, seed) or not np.array_equal(g == special, special_mask):
                continue
            strip = g[:, start:start + span].copy()
            strip[[0, 2, 4]] = background
            if np.any(strip != background):
                continue
            guides.append((rotation, g, seed, special, start, span))
    if len(guides) != 1:
        raise ValueError("expected one unambiguous three-tick edge ruler")
    rotation, g, seed, special, start, span = guides[0]
    remaining = [int(c) for c in colors if c not in (background, seed, special)]
    parts = {c: _components(g == c) for c in remaining}
    possibilities = []
    for endpoint in remaining:
        pp = parts[endpoint]
        if len(pp) != 2 or _square(pp[0]) < 2 or _square(pp[0]) != _square(pp[1]):
            continue
        size = _square(pp[0])
        for ordinary in remaining:
            if ordinary == endpoint or len(parts[ordinary]) < 3:
                continue
            if all(_square(c) == size for c in parts[ordinary]):
                connector = next(c for c in remaining if c not in (endpoint, ordinary))
                possibilities.append((ordinary, endpoint, connector))
    if len(possibilities) != 1:
        raise ValueError("ambiguous station colors")
    ordinary, endpoint, connector = possibilities[0]
    d = _geometry(g, ordinary, endpoint, connector)
    path = _best(d['adjacency'], d['endpoints'], d['tolls'], d['areas'])
    if path['ways'] != 1:
        raise ValueError("tied minimum (tolls, intermediate stations)")
    d.update(grid=g, rotation=rotation, background=background, seed=seed, special=special,
             start=start, span=span, ordinary=ordinary, endpoint=endpoint,
             connector=connector, path=path)
    return d


def _paint(d, path=None):
    path = d['path'] if path is None else path
    out = d['grid'].copy()
    for n in path['nodes'][1:-1]:
        out[tuple(d['stations'][n].T)] = d['special']
    for e in path['edges']:
        out[tuple(d['components'][e].T)] = d['seed']
    a, b = d['start'], d['start'] + d['span']
    out[:, a:b] = d['background']
    for r in range(len(out)):
        if r % 6 in (0, 2):
            out[r, a:b] = d['seed']
        elif r % 6 == 4:
            out[r, a:b] = d['special']
    return np.rot90(out, -d['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive palette, periodic tariff, graph and unique optimum from input only."""
    return _paint(_read(input_grid))


def _features(d):
    p = d['path']
    hops = _best(d['adjacency'], d['endpoints'], d['tolls'], d['areas'], 'hops')
    pixels = _best(d['adjacency'], d['endpoints'], d['tolls'], d['areas'], 'pixels')
    area = sum(d['areas'][e] for e in p['edges'])
    degrees = [len(a) for a in d['adjacency']]
    edges = set(p['edges'])
    ordinary = set(range(len(degrees))) - set(d['endpoints'])
    return dict(a_witness=p['cost'][1] > hops['cost'][0],
                b_witness=area > pixels['cost'][1],
                tolls=p['cost'][0], stops=p['cost'][1],
                cheapest_pixel_area=pixels['cost'][1], selected_pixel_area=area,
                unweighted_stops=hops['cost'][0], nodes=len(degrees),
                loops=len(d['edges']) - len(degrees) + 1,
                leaves=degrees.count(1), junctions=sum(k >= 3 for k in degrees),
                unused_junction=any(degrees[n] >= 3 and any(e not in edges for _, e in d['adjacency'][n])
                                    for n in p['nodes'][1:-1]),
                off_path_toll=any(d['tolls'][n] for n in ordinary - set(p['nodes'])),
                hit_not_near_edge=any(d['tolls'][n] and int(d['stations'][n][:, 0].min()) % 6 != 4
                                      for n in ordinary),
                hit_not_center=any(d['tolls'][n] and float(d['stations'][n][:, 0].mean()) % 6 != 4
                                   for n in ordinary),
                shape=d['grid'].shape, size=d['size'], rotation=d['rotation'],
                span=d['span'], start=d['start'],
                roles=(d['background'], d['seed'], d['special'], d['connector'], d['ordinary'], d['endpoint']))


def _line(a, b):
    n = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [tuple(int(round(a[k] + (b[k] - a[k]) * t / n)) for k in (0, 1))
            for t in range(n + 1)] if n else [a]


def _drawing(rng, anchors, edges, size, shape, start, span, suppress=False):
    """Draw direct metadata masks, then certify the rendered graph separately."""
    g = np.zeros(shape, dtype=np.int8)
    blocks = [{(r + dr, c + dc) for dr in range(size) for dc in range(size)} for r, c in anchors]
    edge_masks = []
    for a, b in edges:
        ra, ca = anchors[a]
        rb, cb = anchors[b]
        if ra == rb:
            if ca > cb:
                ra, ca, rb, cb = rb, cb, ra, ca
            qa, qb = (ra + size - 1, ca + size), (rb, cb - 1)
            horizontal = True
        else:
            if ca != cb:
                raise ValueError("sampling anchors must share a routing axis")
            if ra > rb:
                ra, ca, rb, cb = rb, cb, ra, ca
            qa, qb = (ra + size, ca), (rb - 1, cb + size - 1)
            horizontal = False
        # Keep an axis-aligned stub outside each station before bending.
        # A bend immediately beside a square could touch an unused port.
        lead = (qa[0], qa[1] + 1) if horizontal else (qa[0] + 1, qa[1])
        retreat = (qb[0], qb[1] - 1) if horizontal else (qb[0] - 1, qb[1])
        points = [qa, lead]
        style = int(rng.integers(3))
        if style == 1:
            points.append((lead[0], retreat[1]) if horizontal else (retreat[0], lead[1]))
        elif style == 2:
            mid = ((lead[0] + retreat[0]) // 2, (lead[1] + retreat[1]) // 2)
            points.append(mid)
        points.extend([retreat, qb])
        cells = set()
        for p, q in zip(points, points[1:]):
            cells.update(_line(p, q))
        edge_masks.append(cells)
        for p in cells:
            g[p] = 1
    visible = set(range(len(blocks)))
    if suppress:
        # A removed degree-two block becomes a bend, reducing station count
        # without shortening its corridor. Reject collapsed parallel edges.
        order = rng.permutation(len(blocks)).tolist()
        for n in order:
            incident = [e for e, edge in enumerate(edges) if n in edge]
            if len(visible) <= 5 or len(incident) != 2 or rng.random() > .38:
                continue
            a, b = [next(v for v in edges[e] if v != n) for e in incident]
            if a == b or any(set(edge) == {a, b} for edge in edges):
                continue
            center = np.mean(list(blocks[n]), axis=0)
            tips = [min(edge_masks[e], key=lambda p: (sum((p[k] - center[k]) ** 2 for k in (0, 1)), p))
                    for e in incident]
            merged = edge_masks[incident[0]] | edge_masks[incident[1]] | set(_line(*tips))
            edges = [edge for e, edge in enumerate(edges) if e not in incident] + [(a, b)]
            edge_masks = [cells for e, cells in enumerate(edge_masks) if e not in incident] + [merged]
            visible.remove(n)
        ids = sorted(visible)
        remap = {old: i for i, old in enumerate(ids)}
        anchors = [anchors[n] for n in ids]
        blocks = [blocks[n] for n in ids]
        edges = [(remap[a], remap[b]) for a, b in edges]
    g[:] = 0
    for cells in edge_masks:
        for p in cells:
            g[p] = 1
    for cells in blocks:
        for p in cells:
            g[p] = 2
    g[0, start:start + span] = g[2, start:start + span] = 4
    g[4, start:start + span] = 5
    recovered = _geometry(g, 2, 3, 1)
    if len(recovered['stations']) != len(anchors) or set(recovered['edges']) != {tuple(sorted(e)) for e in edges}:
        raise ValueError("rendered graph differs from construction")
    if [set(map(tuple, c.tolist())) for c in recovered['stations']] != blocks:
        raise ValueError("station ordering or footprints changed")
    if set().union(*blocks) & set().union(*edge_masks):
        raise ValueError("connector overwrites a station")
    # Separate actual connector components must reproduce individual intended masks.
    got = {frozenset(map(tuple, c.tolist())) for c in recovered['components']}
    if got != {frozenset(c) for c in edge_masks}:
        raise ValueError("connector components merged")
    return g, blocks, edge_masks, anchors, edges


def _random_layout(rng, richer=False):
    size = int(rng.choice([2, 3]))
    start, span = int(rng.integers(0, 3)), int(rng.integers(1, 4))
    left = start + span + 3
    nr = int(rng.integers(2, 5))
    max_cols = min(4, 1 + (30 - left - size) // (size + 3))
    nc = int(rng.integers(2, max_cols + 1))
    vertices = [(r, c) for r in range(nr) for c in range(nc)]
    candidates = [(i, j) for i, a in enumerate(vertices) for j, b in enumerate(vertices)
                  if i < j and abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
    if len(vertices) < 5:
        raise ValueError("too few stations")
    target = int(rng.integers(5, len(vertices) + 1))
    active = {int(rng.integers(len(vertices)))}
    edges = []
    while len(active) < target:
        frontier = [(a, b) for a, b in candidates if (a in active) != (b in active)]
        a, b = frontier[int(rng.integers(len(frontier)))]
        active.update((a, b))
        edges.append((a, b))
    extras = [(a, b) for a, b in candidates if a in active and b in active and (a, b) not in edges]
    rng.shuffle(extras)
    loop_target = int(rng.integers(1 if richer else 0, 5 if richer else 4))
    edges += extras[:loop_target]

    def axis(n, low, maximum):
        base = (n - 1) * (size + 3) + size
        margin = int(rng.integers(low, min(low + 5, maximum - base + 1)))
        slack = int(rng.integers(maximum - base - margin + 1))
        gaps = rng.multinomial(slack, [1 / (n - 1)] * (n - 1)) + size + 3
        return [margin] + [margin + int(x) for x in np.cumsum(gaps)]

    rows = axis(nr, 1, 30)
    cols = axis(nc, left, 30)
    ids = sorted(active)
    remap = {v: n for n, v in enumerate(ids)}
    anchors = [(rows[vertices[v][0]], cols[vertices[v][1]]) for v in ids]
    edges = [(remap[a], remap[b]) for a, b in edges]
    h = max(6, max(r for r, _ in anchors) + size)
    w = max(c for _, c in anchors) + size
    shape = (h + int(rng.integers(min(4, 31 - h))), w + int(rng.integers(min(4, 31 - w))))
    return anchors, edges, size, shape, start, span


def _fallback_layout(rng, kind):
    """Guaranteed conflict witnesses; variable guide, margins and sparse-route depth."""
    upper = 7
    lower = int(rng.choice([16, 22])) if kind == 'A' else int(rng.choice([19, 25]))
    anchors = [(upper, c) for c in (6, 11, 16, 21, 26)]
    anchors += [(lower, 6), (lower, 26)]
    edges = [(i, i + 1) for i in range(4)] + [(0, 5), (5, 6), (4, 6)]
    if rng.random() < .7:
        anchors.append((12, 16))
        edges.append((2, 7))
    # Sort to match raster station ordering.
    order = sorted(range(len(anchors)), key=lambda n: anchors[n])
    remap = {old: n for n, old in enumerate(order)}
    anchors = [anchors[n] for n in order]
    edges = [(remap[a], remap[b]) for a, b in edges]
    start = int(rng.integers(0, 2))
    span = int(rng.integers(1, 4))
    shape = lower + int(rng.integers(3, 6)), int(rng.integers(28, 31))
    return (anchors, edges, 2, shape, start, span), (remap[0], remap[4])


def _candidate(rng, layout, kind, forced_endpoints=None):
    anchors, edges, size, shape, start, span = layout
    g, blocks, edge_masks, anchors, edges = _drawing(
        rng, anchors, edges, size, shape, start, span, suppress=forced_endpoints is None)
    adjacency = _adjacency(len(anchors), edges)
    tolls = [any((r + k) % 6 == 4 for k in range(size)) for r, _ in anchors]
    areas = list(map(len, edge_masks))
    choices = [forced_endpoints] if forced_endpoints else list(combinations(range(len(anchors)), 2))
    rng.shuffle(choices)
    for endpoints in choices:
        p = _best(adjacency, endpoints, tolls, areas)
        if p['ways'] != 1 or (kind in ('A', 'B') and len(p['nodes']) < 3):
            continue
        if kind == 'A':
            alternative = _best(adjacency, endpoints, tolls, areas, 'hops')
            if p['cost'][1] <= alternative['cost'][0]:
                continue
        if kind == 'B':
            alternative = _best(adjacency, endpoints, tolls, areas, 'pixels')
            if sum(areas[e] for e in p['edges']) <= alternative['cost'][1]:
                continue
        for n in endpoints:
            for point in blocks[n]:
                g[point] = 3
        # Label directly from construction graph and masks, not from solve().
        out = g.copy()
        for n in p['nodes'][1:-1]:
            for point in blocks[n]:
                out[point] = 5
        for e in p['edges']:
            for point in edge_masks[e]:
                out[point] = 4
        out[:, start:start + span] = 0
        for r in range(len(out)):
            if r % 2 == 0:
                out[r, start:start + span] = 5 if r % 6 == 4 else 4
        if np.count_nonzero(g == 0) <= g.size / 2:
            continue
        return g, out
    return None


def _make_pair(rng, kind, rotation, palette, richer=False):
    for attempt in range(65):
        try:
            if attempt < 64:
                layout, forced = _random_layout(rng, richer), None
            else:
                layout, forced = _fallback_layout(rng, kind if kind in ('A', 'B') else 'B')
            candidate = _candidate(rng, layout, kind, forced)
            if candidate is None:
                continue
            inp, out = candidate
            if rng.random() < .5:
                inp, out = inp[:, ::-1], out[:, ::-1]
            inp = np.rot90(palette[inp], rotation).copy()
            out = np.rot90(palette[out], rotation).copy()
            d = _read(inp)
            if not np.array_equal(_paint(d), out):
                raise ValueError("input reconstruction disagrees with construction label")
            return {'input': inp, 'output': out}
        except ValueError:
            if attempt == 64:
                raise
    raise RuntimeError("bounded construction failed")


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    base = int(rng.integers(4))
    kinds = ['A', 'B'] + [str(rng.choice(['normal', 'A', 'B'], p=[.7, .15, .15]))
                          for _ in range(num_train - 2)]
    rng.shuffle(kinds)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            kind = kinds[i] if split == 'train' else str(rng.choice(['normal', 'A', 'B'], p=[.5, .25, .25]))
            rotation = (base + (i % 2 if split == 'train' else 2 + i % 2)) % 4
            for _ in range(24):
                palette = rng.permutation(10)[:6].astype(np.int8)
                pair = _make_pair(rng, kind, rotation, palette, richer=split == 'test')
                key = _key(pair['input'])
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("bounded uniqueness construction failed")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError("puzzle must contain exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in ('train', 'test')):
        raise ValueError("splits must be lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
                raise ValueError("pair must contain exactly input and output")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            if inp.shape != out.shape or _key(inp) in seen:
                raise ValueError("shape mismatch or duplicate input")
            seen.add(_key(inp))
            d = _read(inp)
            if not np.array_equal(out, _paint(d)):
                raise ValueError("incorrect output, including off-path preservation")
            features[split].append(_features(d))
    if not any(f['a_witness'] for f in features['train']):
        raise ValueError("training must defeat unweighted station routing")
    if not any(f['b_witness'] for f in features['train']):
        raise ValueError("training must defeat pixel routing among minimum-toll paths")
    if {f['rotation'] for f in features['train']} & {f['rotation'] for f in features['test']}:
        raise ValueError("generated tests must use held-out ruler sides")
    return True


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def _simple_paths(adjacency, start, target):
    """Bounded generated graphs only: at most four independent cycles."""
    stack, result = [(start, [start], [])], []
    while stack:
        n, nodes, edges = stack.pop()
        if n == target:
            result.append((nodes, edges))
            continue
        for nxt, e in adjacency[n]:
            if nxt not in nodes:
                stack.append((nxt, nodes + [nxt], edges + [e]))
    return result


def self_test():
    """Stream 42 seeds; retain only aggregate sets and one puzzle at a time."""
    stats = []
    structures, palettes, signatures = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        assert validate_generated_puzzle(puzzle)
        if seed < 3:
            assert _equal(puzzle, generate_puzzle(seed))
        signatures.add(tuple(_key(p['input']) for s in puzzle for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                d = _read(inp)
                f = _features(d)
                f['split'] = split
                stats.append(f)
                norm = np.zeros_like(d['grid'])
                for i, c in enumerate(f['roles']):
                    norm[d['grid'] == c] = i
                structures.add(_key(norm))
                palettes.add(f['roles'])
                # Exhaustive route comparison is separate from Dijkstra.
                paths = _simple_paths(d['adjacency'], *d['endpoints'])
                costs = [((sum(d['tolls'][n] for n in nodes[1:-1]), len(nodes) - 2), nodes, edges)
                         for nodes, edges in paths]
                costs.sort(key=lambda item: item[0])
                assert costs[0][0] == d['path']['cost']
                assert len(costs) == 1 or costs[0][0] < costs[1][0]
                assert costs[0][1] == d['path']['nodes'] and costs[0][2] == d['path']['edges']
                assert np.array_equal(solve(inp), pair['output'])
                # Rotation and palette changes must transform the whole answer.
                if seed < 4:
                    for rotation in range(4):
                        assert np.array_equal(solve(np.rot90(inp, rotation)), np.rot90(pair['output'], rotation))
                    perm = np.random.default_rng(1000 + seed).permutation(10).astype(np.int8)
                    assert np.array_equal(solve(perm[inp]), perm[pair['output']])
                if f['a_witness']:
                    wrong = _best(d['adjacency'], d['endpoints'], d['tolls'], d['areas'], 'hops')
                    assert not np.array_equal(_paint(d, wrong), pair['output'])
                if f['b_witness']:
                    wrong = _best(d['adjacency'], d['endpoints'], d['tolls'], d['areas'], 'pixels')
                    assert not np.array_equal(_paint(d, wrong), pair['output'])
    assert len(signatures) == 42
    assert len(structures) >= 180 and len(palettes) >= 180
    assert len({f['shape'] for f in stats}) >= 50
    assert {f['size'] for f in stats} == {2, 3}
    assert len({f['nodes'] for f in stats}) >= 7
    assert {f['rotation'] for f in stats} == set(range(4))
    assert {f['span'] for f in stats} == {1, 2, 3}
    assert max(f['loops'] for f in stats) >= 3
    assert all(any(f[k] for f in stats) for k in ('leaves', 'junctions', 'unused_junction',
                                                'off_path_toll', 'hit_not_near_edge', 'hit_not_center'))
    for split in ('train', 'test'):
        rows = [f for f in stats if f['split'] == split]
        assert {f['rotation'] for f in rows} == set(range(4))
        assert any(f['a_witness'] for f in rows) and any(f['b_witness'] for f in rows)
    # Exercise every legal explicit count, rejecting noninteger and illegal values.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(42 + nt * 10 + ns, nt, ns)
            assert (len(p['train']), len(p['test'])) == (nt, ns)
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False),
                   (2.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Explicit constructive fallbacks certify both witnesses.
    rng = np.random.default_rng(719)
    for kind in ('A', 'B'):
        layout, endpoints = _fallback_layout(rng, kind)
        candidate = _candidate(rng, layout, kind, endpoints)
        assert candidate is not None
        d = _read(candidate[0])
        assert _features(d)['a_witness' if kind == 'A' else 'b_witness']
    # Equal ordered costs must be rejected, rather than resolved by traversal order.
    adj = _adjacency(4, [(0, 1), (1, 3), (0, 2), (2, 3)])
    assert _best(adj, (0, 3), [False] * 4, [1] * 4)['ways'] == 2
    # A real six-station image with equal routes must also fail input parsing.
    anchors = [(r, c) for r in (7, 19) for c in (6, 16, 26)]
    cycle = [(0, 1), (1, 2), (2, 5), (5, 4), (4, 3), (3, 0)]
    tied, blocks, _, _, _ = _drawing(rng, anchors, cycle, 2, (22, 29), 1, 2)
    for n in (0, 5):
        for point in blocks[n]:
            tied[point] = 3
    try:
        solve(tied)
    except ValueError as error:
        assert 'tied minimum' in str(error)
    else:
        raise AssertionError("tied image accepted")
    corrupt = generate_puzzle(81, 2, 1)
    q = corrupt['test'][0]['output']
    q[0, 0] = (int(q[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return dict(seeds=42, pairs=len(stats), color_normalized_structures=len(structures),
                shapes=len({f['shape'] for f in stats}), palettes=len(palettes),
                station_counts=sorted({f['nodes'] for f in stats}),
                loop_counts=sorted({f['loops'] for f in stats}),
                a_witnesses=sum(f['a_witness'] for f in stats),
                b_witnesses=sum(f['b_witness'] for f in stats))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(",", ":")))
