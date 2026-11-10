"""Capacity-constrained sorting routes. Standard library and NumPy only.

Blue (1) is start and the connector highlight; red (2) is finish.
Other geometric color roles are recovered separately from each input.
"""
from collections import deque
from itertools import combinations
import numpy as np

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= x <= 30 for x in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("grid must be an integer 1..30 by 1..30 array of colors 0..9")
    return g.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in _DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(np.array(sorted(cells), dtype=np.int16))
    return result


def _box(cells):
    r0, c0 = cells.min(axis=0)
    r1, c1 = cells.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_frame(cells):
    r0, c0, r1, c1 = _box(cells)
    return (r1-r0 >= 2 and c1-c0 >= 4
            and len(cells) == 2*(r1-r0+c1-c0)
            and all(r in (r0, r1) or c in (c0, c1) for r, c in cells))


def _bar(g, box, bg, track):
    r0, c0, r1, c1 = box
    anchor = r1-1
    width = c1-c0-3
    cap = r1-r0-1
    if g[anchor, c0+1] != track or g[anchor, c1-1] != track:
        raise ValueError("socket carrier must have two exposed ends")
    color = int(g[anchor, c0+2])
    if color in (bg, track, 1, 2):
        raise ValueError("bar color must differ from background, track and endpoints")
    occupied = np.flatnonzero(np.any(g[r0+1:r1, c0+2:c1-1] != bg, axis=1))
    if not len(occupied):
        raise ValueError("empty socket")
    length = cap-int(occupied[0])
    expected = np.full((cap, c1-c0-1), bg, dtype=np.int8)
    expected[-1, :] = track
    expected[-length:, 1:-1] = color
    if not np.array_equal(expected, g[r0+1:r1, c0+1:c1]):
        raise ValueError("bar must be a solid one-sided span above the carrier")
    return dict(box=box, anchor=anchor, width=width, capacity=cap,
                length=length, color=color)


def _paths(adj, start, end):
    result = []
    stack = [(start, [start], [])]
    while stack:
        a, nodes, edges = stack.pop()
        if a == end:
            result.append((nodes, edges))
            continue
        for b, e in adj[a]:
            if b not in nodes:
                stack.append((b, nodes+[b], edges+[e]))
    return result


def _usable(path, bars, by_area=False):
    nodes = path[0][1:-1]
    order = sorted(nodes, key=lambda n: bars[n]['length'] * (bars[n]['width'] if by_area else 1))
    lengths = [bars[n]['length'] for n in order]
    return all(length <= bars[n]['capacity'] for length, n in zip(lengths, nodes))


def _choose(data):
    feasible = [p for p in data['paths'] if _usable(p, data['bars'])]
    feasible.sort(key=lambda p: len(p[0]))
    if not feasible:
        raise ValueError("no capacity-feasible route")
    if len(feasible) > 1 and len(feasible[0][0]) == len(feasible[1][0]):
        raise ValueError("minimum-station feasible route is tied")
    return feasible[0], feasible


def _read_canonical(value):
    g = _grid(value)
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if bg in (1, 2) or np.count_nonzero(counts == counts[bg]) != 1:
        raise ValueError("background must be uniquely dominant and distinct from endpoints")
    endpoints = []
    for color in (1, 2):
        cc = _components(g == color)
        if len(cc) != 1 or len(cc[0]) != 9:
            raise ValueError("each endpoint must be one solid 3 by 3 square")
        r0, c0, r1, c1 = _box(cc[0])
        if r1-r0 != 2 or c1-c0 != 2:
            raise ValueError("malformed endpoint")
        endpoints.append(cc[0])
    frames = []
    for color in np.unique(g):
        if color in (bg, 1, 2):
            continue
        cc = _components(g == color)
        if len(cc) >= 2 and all(_is_frame(c) for c in cc):
            frames.append((int(color), cc))
    if len(frames) != 1:
        raise ValueError("hollow frame role must be unique")
    frame, cc = frames[0]
    if not 2 <= len(cc) <= 10:
        raise ValueError("family has 2..10 ordinary sockets")
    boxes = [_box(c) for c in cc]
    tracks = {int(g[r1-1, c0+1]) for r0, c0, r1, c1 in boxes}
    if len(tracks) != 1:
        raise ValueError("all local carriers share the track role")
    track = tracks.pop()
    if track in (bg, frame, 1, 2):
        raise ValueError("geometric roles must be distinct")
    bars = {i+2: _bar(g, box, bg, track) for i, box in enumerate(boxes)}
    if any(b['color'] == frame for b in bars.values()):
        raise ValueError("bar and frame roles must differ")
    nodes = endpoints+cc
    node_map = np.full(g.shape, -1, dtype=np.int16)
    internal = np.zeros(g.shape, dtype=bool)
    for i, cells in enumerate(nodes):
        node_map[tuple(cells.T)] = i
    for r0, c0, r1, c1 in boxes:
        internal[r0+1:r1, c0+1:c1] = True
    components = _components((g == track) & ~internal)
    edges = []
    for cells in components:
        touching = set()
        for r, c in cells.tolist():
            for dr, dc in _DIRS:
                rr, co = r+dr, c+dc
                if 0 <= rr < g.shape[0] and 0 <= co < g.shape[1] and node_map[rr, co] >= 0:
                    touching.add(int(node_map[rr, co]))
        if len(touching) != 2:
            raise ValueError("each external connector must meet exactly two stations")
        edges.append(tuple(sorted(touching)))
    if len(set(edges)) != len(edges):
        raise ValueError("parallel connectors are outside this family")
    adj = [[] for _ in nodes]
    for e, (a, b) in enumerate(edges):
        adj[a].append((b, e)); adj[b].append((a, e))
    reached, stack = {0}, [0]
    while stack:
        for b, _ in adj[stack.pop()]:
            if b not in reached:
                reached.add(b); stack.append(b)
    if len(reached) != len(nodes):
        raise ValueError("station network must be connected")
    rebuilt = np.full(g.shape, bg, dtype=np.int8)
    for color, cells in zip((1, 2), endpoints):
        rebuilt[tuple(cells.T)] = color
    for cells in cc:
        rebuilt[tuple(cells.T)] = frame
    for b in bars.values():
        r0, c0, r1, c1 = b['box']
        rebuilt[r1-1, c0+1:c1] = track
        rebuilt[r1-b['length']:r1, c0+2:c1-1] = b['color']
    for cells in components:
        rebuilt[tuple(cells.T)] = track
    if not np.array_equal(g, rebuilt):
        raise ValueError("unexplained input pixels")
    data = dict(grid=g, background=bg, frame=frame, track=track, nodes=nodes,
                bars=bars, edges=edges, components=components, adjacency=adj,
                paths=_paths(adj, 0, 1))
    data['chosen'], data['feasible'] = _choose(data)
    return data


def _read(value):
    g = _grid(value)
    errors = []
    for rotation in range(4):
        try:
            data = _read_canonical(np.rot90(g, rotation))
            data['rotation'] = rotation
            data['original_shape'] = g.shape
            return data
        except ValueError as error:
            errors.append(str(error))
    raise ValueError('no readable socket orientation: '+errors[0])


def _paint(g, data, path, by_area=False, reverse=False):
    out = g.copy()
    nodes, edges = path
    middle = nodes[1:-1]
    order = (middle[::-1] if reverse else sorted(middle, key=lambda n:
             data['bars'][n]['length'] * (data['bars'][n]['width'] if by_area else 1)))
    for n, source in zip(middle, order):
        b = data['bars'][n]
        r0, c0, r1, c1 = b['box']
        length = data['bars'][source]['length']
        if length > b['capacity']:
            raise ValueError("assigned span exceeds socket capacity")
        out[r0+1:r1, c0+1:c1] = data['background']
        out[r1-1, c0+1:c1] = data['track']
        out[r1-length:r1, c0+2:c1-1] = b['color']
    for e in edges:
        out[tuple(data['components'][e].T)] = 1
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, capacity, bar span and graph edge from input pixels."""
    data = _read(input_grid)
    return np.rot90(_paint(data['grid'], data, data['chosen']), -data['rotation']).copy()


def _abstract_paths(n, edges, start, finish):
    adj = [[] for _ in range(n)]
    for e, (a, b) in enumerate(edges):
        adj[a].append((b, e)); adj[b].append((a, e))
    return _paths(adj, start, finish)


def _lattice_graph(rng, nr, nc, rich):
    candidates = [(r*nc+c, rr*nc+co) for r in range(nr) for c in range(nc)
                  for rr, co in ((r+1, c), (r, c+1)) if rr < nr and co < nc]
    rng.shuffle(candidates)
    roots = list(range(nr*nc))
    def root(a):
        while roots[a] != a:
            a = roots[a]
        return a
    edges, extras = [], []
    for a, b in candidates:
        x, y = root(a), root(b)
        if x != y:
            roots[x] = y; edges.append((a, b))
        else:
            extras.append((a, b))
    rng.shuffle(extras)
    loops = int(rng.integers(1, min(len(extras), rich)+1))
    return edges+extras[:loops]


def _sample_abstract(rng, dims, witness):
    nr, nc = dims
    n = nr*nc
    # More than one endpoint placement per drawing makes the rare capacity
    # conflict practical without imposing a fixed topology or winning side.
    for attempt in range(4000):
        edges = _lattice_graph(rng, nr, nc, 5 if n == 12 else 3)
        capacities = rng.integers(1, 6, n).astype(int).tolist()
        lengths = [int(rng.integers(1, c+1)) for c in capacities]
        candidates = [(a, b) for a, b in combinations(range(n), 2)
                      if abs(a//nc-b//nc)+abs(a%nc-b%nc) >= (3 if witness == 'B' or n < 8 else 4)]
        rng.shuffle(candidates)
        for a, b in candidates:
            if rng.random() < .5:
                a, b = b, a
            paths = _abstract_paths(n, edges, a, b)
            bars = {i: dict(length=lengths[i], capacity=capacities[i], width=1) for i in range(n)}
            feasible = sorted([p for p in paths if _usable(p, bars)], key=lambda p: len(p[0]))
            if not feasible or (len(feasible) > 1 and len(feasible[0][0]) == len(feasible[1][0])):
                continue
            chosen = feasible[0]
            values = [lengths[i] for i in chosen[0][1:-1]]
            if witness and values == sorted(values):
                continue
            if witness == 'A':
                shortest = sorted(paths, key=lambda p: len(p[0]))
                if (len(shortest) < 2 or len(shortest[0][0]) == len(shortest[1][0])
                        or len(chosen[0]) <= len(shortest[0][0])
                        or values[::-1] == sorted(values)):
                    continue
            if witness == 'B' and len(feasible) < 2:
                continue
            if witness == 'T' and (len(feasible) < 2 or len(feasible) == len(paths)):
                continue
            widths = rng.integers(1, (3 if nc == 4 else 4), n).astype(int).tolist()
            if witness == 'A':
                max_width = 2 if nc == 4 else 3
                pairs = [(i, j) for i, j in combinations(chosen[0][1:-1], 2)
                         if lengths[i] != lengths[j] and min(lengths[i], lengths[j])*max_width > max(lengths[i], lengths[j])]
                if not pairs:
                    continue
                i, j = pairs[int(rng.integers(len(pairs)))]
                if lengths[i] > lengths[j]:
                    i, j = j, i
                widths[i], widths[j] = max_width, 1
            return dict(nr=nr, nc=nc, edges=edges, start=a, finish=b,
                        capacities=capacities, lengths=lengths, widths=widths,
                        chosen=chosen, feasible=feasible)
    raise RuntimeError("bounded abstract construction exhausted")


def _draw_abstract(rng, spec, roles, witness, transpose=None):
    nr, nc = spec['nr'], spec['nc']
    n = nr*nc
    start, finish = spec['start'], spec['finish']
    heights = [3 if i in (start, finish) else spec['capacities'][i]+2 for i in range(n)]
    widths = [3 if i in (start, finish) else spec['widths'][i]+4 for i in range(n)]
    row_heights = [max(heights[r*nc:(r+1)*nc]) for r in range(nr)]
    col_widths = [max(widths[c::nc]) for c in range(nc)]
    row_gaps = [int(rng.integers(2, 5)) for _ in range(nr-1)]
    col_gaps = [2]* (nc-1)
    slack = 30-sum(col_widths)-sum(col_gaps)
    if nc < 4 and slack > 0:
        for _ in range(int(rng.integers(min(slack, 4)+1))):
            col_gaps[int(rng.integers(nc-1))] += 1
    bottoms, centers = [], []
    y = 0
    for r, h in enumerate(row_heights):
        bottoms.append(y+h-1)
        y += h+(row_gaps[r] if r < nr-1 else 0)
    x = 0
    for c, w in enumerate(col_widths):
        centers.append(x+w//2)
        x += w+(col_gaps[c] if c < nc-1 else 0)
    h, w = y, x
    if h > 30 or w > 30:
        raise ValueError("embedding does not fit")
    g = np.zeros((h, w), dtype=np.int8)
    boxes = []
    for i in range(n):
        r, c = divmod(i, nc)
        r1 = bottoms[r]; r0 = r1-heights[i]+1
        c0 = centers[c]-widths[i]//2; c1 = c0+widths[i]-1
        boxes.append((r0, c0, r1, c1))
        if i in (start, finish):
            g[r0:r1+1, c0:c1+1] = 1 if i == start else 2
        else:
            g[r0:r1+1, c0] = g[r0:r1+1, c1] = 3
            g[r0, c0:c1+1] = g[r1, c0:c1+1] = 3
            g[r1-1, c0+1:c1] = 4
            # Temporary accents 5..9 are later mapped to local colors.
            color = int(rng.integers(5, 10))
            g[r1-spec['lengths'][i]:r1, c0+2:c1-1] = color
    selected_edges = set(spec['chosen'][1])
    for e, (a, b) in enumerate(spec['edges']):
        ar0, ac0, ar1, ac1 = boxes[a]
        br0, bc0, br1, bc1 = boxes[b]
        if a//nc == b//nc:
            if ac0 > bc0:
                a, b = b, a
                ar0, ac0, ar1, ac1 = boxes[a]; br0, bc0, br1, bc1 = boxes[b]
            rr = ar1-1
            left, right = ac1+1, bc0-1
            # U-shaped bends can be longer than a route with more stations.
            # For the counting witness, concentrate detours on chosen edges.
            detour = (e in selected_edges) if witness == 'B' else rng.random() < .45
            depth = min(rr, int(rng.integers(2, 6))) if detour and right-left >= 2 else 0
            g[rr-depth:rr+1, left] = 4
            g[rr-depth, left:right+1] = 4
            g[rr-depth:rr+1, right] = 4
        else:
            if ar0 > br0:
                a, b = b, a
                ar0, ac0, ar1, ac1 = boxes[a]; br0, bc0, br1, bc1 = boxes[b]
            cc = centers[a % nc]
            g[ar1+1:br0, cc] = 4
    # Reflection and transpose change placement without changing the explicit
    # start-to-finish sorting order, which is a graph itinerary.
    if rng.random() < .5:
        g = g[:, ::-1]
    if rng.random() < .5:
        g = g[::-1, :]
    use_transpose = bool(rng.integers(2)) if transpose is None else transpose
    if use_transpose:
        g = g.T
    hh, ww = g.shape
    extra_h = int(rng.integers(min(30-hh, 3)+1))
    extra_w = int(rng.integers(min(30-ww, 3)+1))
    padded = np.zeros((hh+extra_h, ww+extra_w), dtype=np.int8)
    top = int(rng.integers(extra_h+1)); left = int(rng.integers(extra_w+1))
    padded[top:top+hh, left:left+ww] = g
    lookup = np.array([roles[0], 1, 2, roles[1], roles[2]]+roles[3:], dtype=np.int8)
    return lookup[padded]


def _features(data):
    nodes, edges = data['chosen']
    middle = nodes[1:-1]
    values = [data['bars'][i]['length'] for i in middle]
    area_values = [data['bars'][i]['length'] for i in sorted(middle, key=lambda n:
                   data['bars'][n]['length']*data['bars'][n]['width'])]
    raw = sorted(data['paths'], key=lambda p: len(p[0]))
    chosen_area = sum(len(data['components'][e]) for e in edges)
    return dict(stations=len(data['nodes']), sockets=len(data['bars']),
                routes=len(data['paths']), feasible=len(data['feasible']),
                hops=len(edges), raw_hops=len(raw[0][1]),
                capacity_witness=(len(raw)>1 and len(raw[0][0]) < len(raw[1][0])
                                  and len(raw[0][0]) < len(nodes)),
                counting_witness=(len(data['feasible']) > 1 and chosen_area > min(
                    sum(len(data['components'][e]) for e in p[1]) for p in data['feasible'])),
                changed_bars=values != sorted(values), nonreverse=values[::-1] != sorted(values),
                area_disagrees=area_values != sorted(values),
                widths=tuple(b['width'] for b in data['bars'].values()),
                capacities=tuple(b['capacity'] for b in data['bars'].values()),
                lengths=tuple(b['length'] for b in data['bars'].values()),
                roles=(data['background'], data['frame'], data['track']),
                degree=tuple(sorted(len(a) for a in data['adjacency'])),
                loops=len(data['edges'])-len(data['nodes'])+1,
                shape=data['original_shape'], axis=data['rotation'] % 2)


def _make_pair(rng, dims, witness, roles, transpose=None):
    # Separate bounded searches for graph/span constraints and pixel geometry.
    for _ in range(100):
        spec = _sample_abstract(rng, dims, witness)
        for _ in range(12):
            try:
                inp = _draw_abstract(rng, spec, roles, witness, transpose)
                data = _read(inp)
                f = _features(data)
                if witness == 'A' and not (f['capacity_witness'] and f['area_disagrees'] and f['nonreverse']):
                    continue
                if witness == 'B' and not f['counting_witness']:
                    continue
                out = np.rot90(_paint(data['grid'], data, data['chosen']), -data['rotation']).copy()
                return {'input': inp, 'output': out}
            except ValueError:
                continue
    raise RuntimeError("bounded pixel construction exhausted")


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _key(g):
    g = np.asarray(g)
    return g.shape, g.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, 'num_train')
    nq = _count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    base_roles = [int(c) for c in rng.permutation([0, 3, 4, 5, 6, 7, 8, 9])]
    consistent = bool(rng.random() < .25)
    puzzle, seen = {'train': [], 'test': []}, set()
    a_dims = (3, 3) if rng.random() < .6 else (2, 4)
    b_dims = (2, 4) if a_dims == (3, 3) else (3, 3)
    first_axis = bool(rng.integers(2))
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            witness = ('A', 'B')[i] if split == 'train' and i < 2 else 'T' if split == 'test' else ''
            if witness in ('A', 'B'):
                dims = a_dims if witness == 'A' else b_dims
            elif split == 'test':
                dims = (3, 4)
            else:
                dims = [(2, 3), (2, 4), (3, 3), (3, 4)][int(rng.integers(4))]
            for _ in range(20):
                roles = base_roles if consistent else [int(c) for c in rng.permutation([0, 3, 4, 5, 6, 7, 8, 9])]
                axis = first_axis if witness == 'A' else not first_axis if witness == 'B' else None
                pair = _make_pair(rng, dims, witness, roles, axis)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key); puzzle[split].append(pair); break
            else:
                raise RuntimeError("bounded uniqueness construction exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    """Relational output checks: no call to solve or its output painter."""
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape
    data = _read(inp)
    inp = data['grid']
    out = np.rot90(out, data['rotation'])
    selected_edges = []
    allowed = np.zeros(inp.shape, dtype=bool)
    for e, cells in enumerate(data['components']):
        values = out[tuple(cells.T)]
        if np.all(values == 1):
            selected_edges.append(e); allowed[tuple(cells.T)] = True
        else:
            assert np.all(values == data['track']), "connector partially recolored or wrong color"
    expected_nodes, expected_edges = data['chosen']
    assert set(selected_edges) == set(expected_edges), "wrong capacity-feasible minimum route"
    remaining = [data['bars'][n]['length'] for n in expected_nodes[1:-1]]
    previous = 0
    for n in expected_nodes[1:-1]:
        b = data['bars'][n]
        r0, c0, r1, c1 = b['box']
        final = _bar(out, b['box'], data['background'], data['track'])
        assert final['width'] == b['width'] and final['color'] == b['color']
        assert previous <= final['length'] <= b['capacity'], "order or containment violation"
        assert final['length'] in remaining, "span multiset changed"
        remaining.remove(final['length']); previous = final['length']
        allowed[r0+1:r1, c0+1:c1] = True
    assert not remaining
    assert np.array_equal(inp[~allowed], out[~allowed]), "off-route pixels, frame or endpoint changed"
    return data


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            key = _key(_grid(pair['input']))
            assert key not in seen, "duplicate input"
            seen.add(key)
            features[split].append(_features(_check_pair(pair)))
    train = features['train']
    assert any(f['capacity_witness'] and f['changed_bars'] and f['nonreverse'] and f['area_disagrees'] for f in train)
    assert any(f['counting_witness'] and f['changed_bars'] for f in train)
    assert {f['axis'] for f in train} == {0, 1}, 'both bar axes must be demonstrated'
    assert {8, 9} <= {f['stations'] for f in train}, "both witness scales must be demonstrated"
    assert all(f['stations'] == 12 and f['changed_bars'] and
               2 <= f['feasible'] < f['routes'] for f in features['test']), 'test recombination lacks both constraints'


def self_test() -> dict:
    shapes, structures, palettes, hops, capacities, widths, degrees, loops = (set() for _ in range(8))
    signatures = set()
    pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_key(p[r]) for s in ('train', 'test') for p in puzzle[s] for r in ('input', 'output'))
        signatures.add(signature)
        if seed < 2:
            repeat = generate_puzzle(seed)
            assert signature == tuple(_key(p[r]) for s in ('train', 'test') for p in repeat[s] for r in ('input', 'output'))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                data = _read(pair['input']); f = _features(data)
                assert np.array_equal(solve(pair['input']), pair['output'])
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                shapes.add(f['shape']); palettes.add(f['roles']); hops.add(f['hops'])
                capacities.update(f['capacities']); widths.update(f['widths'])
                degrees.update(f['degree']); loops.add(f['loops'])
                g = pair['input']; mapping = {}; norm = np.array([mapping.setdefault(int(c), len(mapping)) for c in g.ravel()], dtype=np.int8).reshape(g.shape)
                structures.add(_key(norm)); pairs += 1
    assert len(signatures) == 42 and len(structures) >= 180
    assert len(shapes) >= 30 and len(palettes) >= 20
    assert capacities == {1, 2, 3, 4, 5} and widths == {1, 2, 3}
    assert {1, 2, 3} <= degrees and len(hops) >= 4 and max(loops) >= 3
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(901+nt*10+nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Rotations/reflections preserve the directed itinerary, rather than
    # replacing it with global left-to-right order.
    q = generate_puzzle(81, 2, 1)
    for pair in q['train']+q['test']:
        for k in range(4):
            for reflect in (False, True):
                a, b = np.rot90(pair['input'], k), np.rot90(pair['output'], k)
                if reflect:
                    a, b = a[:, ::-1], b[:, ::-1]
                assert np.array_equal(solve(a), b)
    # Find a raw hop tie, then make every socket unit-length: all routes
    # become usable, and an arbitrary traversal must not break the tie.
    rejected_tie = False
    for seed in range(8):
        q = generate_puzzle(seed, 2, 1)
        for pair in q['train']+q['test']:
            d = _read(pair['input'])
            raw = sorted(d['paths'], key=lambda p: len(p[0]))
            if len(raw) < 2 or len(raw[0][0]) != len(raw[1][0]):
                continue
            tied = d['grid'].copy()
            for b in d['bars'].values():
                r0, c0, r1, c1 = b['box']
                tied[r0+1:r1, c0+1:c1] = d['background']
                tied[r1-1, c0+1:c1] = d['track']
                tied[r1-1, c0+2:c1-1] = b['color']
            try:
                solve(tied)
            except ValueError:
                rejected_tie = True
            else:
                raise AssertionError('tied route accepted')
        if rejected_tie:
            break
    assert rejected_tie
    # Targeted output mutations must be rejected by the relational validator.
    p = generate_puzzle(0, 2, 1)
    for pair in p['train']:
        d = _read(pair['input'])
        f = _features(d)
        if f['nonreverse'] and f['changed_bars']:
            wrong = np.rot90(_paint(d['grid'], d, d['chosen'], reverse=True), -d['rotation']) if all(
                d['bars'][src]['length'] <= d['bars'][dst]['capacity']
                for src, dst in zip(d['chosen'][0][1:-1][::-1], d['chosen'][0][1:-1])) else pair['input'].copy()
            try:
                _check_pair({'input': pair['input'], 'output': wrong})
            except (AssertionError, ValueError):
                pass
            else:
                raise AssertionError("reversal/no-sort mutation accepted")
    corrupt = p['train'][0]['output'].copy()
    d = _read(p['train'][0]['input'])
    endpoint = np.argwhere(p['train'][0]['input'] == 1)[0]
    corrupt[tuple(endpoint)] = 2
    try:
        _check_pair({'input': p['train'][0]['input'], 'output': corrupt})
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("endpoint mutation accepted")
    return dict(seeds=42, pairs=pairs, shapes=len(shapes), structures=len(structures),
                role_palettes=len(palettes), route_hops=sorted(hops), capacities=sorted(capacities),
                widths=sorted(widths), degrees=sorted(degrees), loop_counts=sorted(loops))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
