"""Voting gates: shape-indexed elections determine a station network's routes.

No parent generator is imported. Generation uses a small routing lattice;
solving reconstructs frames, cavities, votes and edges from pixels alone.
All connectivity is orthogonal. No files are written by this module.
"""
from __future__ import annotations

from collections import Counter, deque
from functools import lru_cache
from itertools import combinations
import hashlib
import heapq
import numpy as np


DIRS = ((-1, 0), (0, -1), (0, 1), (1, 0))
HOST_SEEDS = (0,75883832,994849664,183037220,154339683,101236714,649327764,
              346565950,888833931,896515578,714221437,15586229,913038474,
              134241352,1793307,279634461,686999212,467882049,301682327,
              662661608,913607270,988724609,124628678,177727790,716564716,
              567571483,960264165,555941036,200521498,844004067,399058564,
              792246210,312481537,279288785,938846615,854247871,646799301,
              715370213,341167695,846104202,217513889,549159302)


def _norm(cells):
    cells = tuple((int(r), int(c)) for r, c in cells)
    r0, c0 = min(r for r, _ in cells), min(c for _, c in cells)
    return tuple(sorted((r-r0, c-c0) for r, c in cells))


def _rot(shape):
    return _norm((c, -r) for r, c in shape)


def _mirror(shape):
    return _norm((r, -c) for r, c in shape)


@lru_cache(maxsize=1024)
def _rotations(shape):
    result, s = set(), _norm(shape)
    for _ in range(4):
        result.add(s)
        s = _rot(s)
    return tuple(sorted(result))


@lru_cache(maxsize=1024)
def _variants(shape):
    return tuple(sorted(set(_rotations(shape)) | set(_rotations(_mirror(shape)))))


def _canon(cells):
    return min(_variants(_norm(cells)))


def _extent(shape):
    return max(r for r, _ in shape)+1, max(c for _, c in shape)+1


def _components(mask):
    left = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while left:
        start = min(left)
        left.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in DIRS:
                p = r+dr, c+dc
                if p in left:
                    left.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(cells)))
    return result


def _connected(cells):
    left = set(cells)
    if not left:
        return False
    stack = [left.pop()]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS:
            p = r+dr, c+dc
            if p in left:
                left.remove(p)
                stack.append(p)
    return not left


@lru_cache(maxsize=4)
def _pool(interior):
    # Exhaust the tiny 3x3 vocabulary. For 4x4, also include parameterized
    # larger arms/bars; all tests still use exact, unscaled cell geometry.
    points = [(r, c) for r in range(min(interior, 3))
              for c in range(min(interior, 3))]
    shapes = set()
    for n in range(1, min(7, len(points))+1):
        for cells in combinations(points, n):
            if not _connected(cells):
                continue
            shape = _canon(cells)
            h, w = _extent(shape)
            frame = {(r, c) for r in range(h+2) for c in range(w+2)}
            frame -= {(r+1, c+1) for r, c in shape}
            if _connected(frame):
                shapes.add(shape)
    if interior >= 4:
        for h in range(2, interior+1):
            for w in range(2, interior+1):
                shapes.add(_canon({(r, 0) for r in range(h)} |
                                  {(h-1, c) for c in range(w)}))
        shapes.add(_canon((0, c) for c in range(interior)))
    return tuple(sorted(shapes))


def _choice(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for e, (a, b) in enumerate(edges):
        adj[a].append((b, e))
        adj[b].append((a, e))
    return adj


def _shortest(adj, ends, allowed=None, omit=-1):
    """Return a unique minimum-hop route, or None for no route/a tie."""
    if allowed is None:
        allowed = set(range(len(adj)))
    start, target = ends
    dist, ways, prev = [-1]*len(adj), [0]*len(adj), [None]*len(adj)
    dist[start], ways[start] = 0, 1
    queue = deque([start])
    while queue:
        a = queue.popleft()
        for b, e in adj[a]:
            if e == omit or b not in allowed:
                continue
            if dist[b] < 0:
                dist[b], prev[b] = dist[a]+1, (a, e)
                queue.append(b)
            if dist[b] == dist[a]+1:
                ways[b] = min(2, ways[b]+ways[a])
    if ways[target] != 1:
        return None
    nodes, edges = [target], []
    while nodes[-1] != start:
        a, e = prev[nodes[-1]]
        nodes.append(a)
        edges.append(e)
    return nodes[::-1], edges[::-1]


def _weighted(adj, ends, allowed, weights):
    start, target = ends
    best, heap = {start: 0}, [(0, start)]
    while heap:
        cost, a = heapq.heappop(heap)
        if cost != best[a]:
            continue
        if a == target:
            return cost
        for b, e in adj[a]:
            candidate = cost+weights[e]
            if b in allowed and candidate < best.get(b, 10**9):
                best[b] = candidate
                heapq.heappush(heap, (candidate, b))
    return 10**9


def _reachable(adj, ends, allowed, omit=-1):
    reached, stack = {ends[0]}, [ends[0]]
    while stack:
        for b, e in adj[stack.pop()]:
            if e != omit and b in allowed and b not in reached:
                reached.add(b)
                stack.append(b)
    return ends[1] in reached


def _gate_set(raws, colors, ends, equivalence='free'):
    def signature(raw):
        if equivalence == 'area':
            return len(raw)
        if equivalence == 'box':
            return tuple(sorted(_extent(raw)))
        if equivalence == 'rotation':
            return min(_rotations(raw))
        if equivalence == 'translation':
            return raw
        return _canon(raw)
    ballots, keys = {}, {}
    for n, raw in raws.items():
        key = signature(raw)
        keys[n] = key
        ballots.setdefault(key, Counter())[colors[n]] += 1
    winners = {}
    for key, counts in ballots.items():
        top = max(counts.values())
        best = [c for c, count in counts.items() if count == top]
        if len(best) != 1:
            return None, ballots, None
        winners[key] = best[0]
    allowed = set(ends) | {n for n, key in keys.items() if colors[n] == winners[key]}
    return allowed, ballots, winners


def _geometry(grid, boxes, connector):
    ids = np.full(grid.shape, -1, dtype=np.int16)
    for n, (r, c, size) in enumerate(boxes):
        ids[r:r+size, c:c+size] = n
    components = _components(grid == connector)
    edges = []
    h, w = grid.shape
    for cells in components:
        touching = set()
        for r, c in cells:
            for dr, dc in DIRS:
                rr, cc = r+dr, c+dc
                if 0 <= rr < h and 0 <= cc < w and ids[rr, cc] >= 0:
                    touching.add(int(ids[rr, cc]))
        assert len(touching) == 2, 'a connector must join exactly two stations'
        edges.append(tuple(sorted(touching)))
    assert len(set(edges)) == len(edges), 'no parallel connectors'
    adj = _adjacency(len(boxes), edges)
    assert all(1 <= len(a) <= 4 for a in adj)
    assert all(_reachable(adj, (0, n), set(range(len(boxes)))) for n in range(len(boxes)))
    return dict(boxes=boxes, edge_cells=components, edges=edges, adj=adj,
                weights=[len(cells) for cells in components])


def _decode(input_grid):
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
    bg = int(grid[0, 0])
    assert all(np.all(side == bg) for side in (grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    frames, leftover = [], set()
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for cells in _components(grid == color):
            r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
            r1, c1 = max(r for r, c in cells), max(c for r, c in cells)
            box = grid[r0:r1+1, c0:c1+1]
            size = len(box)
            is_frame = (size == box.shape[1] and size >= 4 and
                        all(np.all(side == color) for side in
                            (box[0], box[-1], box[:, 0], box[:, -1])) and
                        set(map(int, np.unique(box))) <= {color, bg})
            if is_frame:
                holes = _components(box == bg)
                assert len(holes) <= 1, 'one cavity per ordinary station'
                frames.append((r0, c0, size, color, holes[0] if holes else None))
            else:
                leftover.add(color)
    assert len(frames) >= 6 and len({f[2] for f in frames}) == 1
    frames.sort()
    assert len(leftover) == 1
    connector = next(iter(leftover))
    assert connector not in {f[3] for f in frames}
    ends = [n for n, f in enumerate(frames) if f[4] is None]
    assert len(ends) == 2
    terminal = frames[ends[0]][3]
    assert frames[ends[1]][3] == terminal
    assert all(f[3] != terminal for n, f in enumerate(frames) if n not in ends)
    boxes = [f[:3] for f in frames]
    data = _geometry(grid, boxes, connector)
    raws = {n: _norm(f[4]) for n, f in enumerate(frames) if f[4] is not None}
    colors = {n: f[3] for n, f in enumerate(frames)}
    holes = {n: tuple((f[0]+r, f[1]+c) for r, c in f[4])
             for n, f in enumerate(frames) if f[4] is not None}
    allowed, ballots, winners = _gate_set(raws, colors, ends)
    assert allowed is not None, 'tied shape election'
    path = _shortest(data['adj'], ends, allowed)
    assert path is not None, 'no unique minimum-station route'
    data.update(bg=bg, connector=connector, terminal=terminal, ends=ends,
                raws=raws, colors=colors, holes=holes, allowed=allowed,
                ballots=ballots, winners=winners, path=path)
    return data


def _paint(grid, data, path):
    out = np.array(grid, copy=True)
    for n in path[0][1:-1]:
        for r, c in data['holes'][n]:
            out[r, c] = data['colors'][n]
    for e in path[1]:
        for r, c in data['edge_cells'][e]:
            out[r, c] = data['terminal']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only: count each perforated frame once, elect gates, find route."""
    data = _decode(input_grid)
    return _paint(input_grid, data, data['path'])


def _compressed(vertices, edges, visible):
    adj = {v: [] for v in vertices}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    visible = sorted(visible)
    ids, result = {v: i for i, v in enumerate(visible)}, []
    for v in visible:
        for neighbor in adj[v]:
            prev, cur = v, neighbor
            seen = {v}
            while cur not in ids:
                assert cur not in seen and len(adj[cur]) == 2
                seen.add(cur)
                nxt = next(x for x in adj[cur] if x != prev)
                prev, cur = cur, nxt
            assert cur != v
            if ids[v] < ids[cur]:
                result.append((ids[v], ids[cur]))
    assert len(set(result)) == len(result)
    return result


def _random_topology(rng, nr, nc, minimum, dense=False):
    vertices = [(r, c) for r in range(nr) for c in range(nc)]
    possible = [(v, (v[0]+dr, v[1]+dc)) for v in vertices
                for dr, dc in ((1, 0), (0, 1))
                if v[0]+dr < nr and v[1]+dc < nc]
    # Randomized Kruskal spanning tree followed by independently chosen loops.
    groups = {v: {v} for v in vertices}
    edges = []
    for i in rng.permutation(len(possible)):
        a, b = possible[int(i)]
        if groups[a] is not groups[b]:
            merged = groups[a] | groups[b]
            for v in merged:
                groups[v] = merged
            edges.append((a, b))
    extras = [e for e in possible if e not in edges]
    rng.shuffle(extras)
    count = int(rng.integers(2, len(extras)+1)) if len(extras) >= 2 else len(extras)
    if dense:
        count = max(count, len(extras)-2)
    edges += extras[:count]
    visible = set(vertices)
    candidates = [v for v in vertices if sum(v in e for e in edges) == 2]
    rng.shuffle(candidates)
    for v in candidates:
        if len(visible) <= minimum or rng.random() < .25:
            continue
        trial = visible-{v}
        try:
            _compressed(vertices, edges, trial)
        except AssertionError:
            continue
        visible = trial
    return vertices, edges, visible


def _witness_topology(rng):
    # A parameterized perimeter with a branching interior tree. Suppression
    # makes the physically longer lower arc have one station versus three.
    vertices = [(r, c) for r in range(4) for c in range(4)]
    ring = [(r, c) for r, c in vertices if r in (0, 3) or c in (0, 3)]
    edges = [(a, b) for a, b in combinations(ring, 2)
             if abs(a[0]-b[0])+abs(a[1]-b[1]) == 1]
    root_col = int(rng.integers(1, 3))
    other = 3-root_col
    edges += [((0, root_col), (1, root_col)),
              ((1, root_col), (1, other)),
              ((1, root_col), (2, root_col)),
              ((2, root_col), (2, other))]
    upper = [(0, c) for c in range(4)]
    upper.remove((0, other))
    visible = {(1, 0), (1, 3), (3, int(rng.integers(1, 3)))}
    visible.update(upper)
    visible.update((r, c) for r in (1, 2) for c in (1, 2))
    return vertices, edges, visible


def _draw_skeleton(rng, topology, size):
    vertices, edges, visible = topology
    nr, nc = max(r for r, _ in vertices)+1, max(c for _, c in vertices)+1
    def axis(n):
        slack = 28-(n*size+2*(n-1))
        extra = int(rng.integers(min(slack, 5)+1))
        gaps = rng.multinomial(extra, [1/(n-1)]*(n-1)) + size+2
        return [1] + list(1+np.cumsum(gaps).astype(int))
    rows, cols = axis(nr), axis(nc)
    grid = np.zeros((int(rows[-1]+size+1), int(cols[-1]+size+1)), dtype=np.int8)
    mid = size//2
    for a, b in edges:
        ra, ca, rb, cb = rows[a[0]]+mid, cols[a[1]]+mid, rows[b[0]]+mid, cols[b[1]]+mid
        if ra == rb:
            grid[ra, min(ca, cb):max(ca, cb)+1] = 1
        else:
            grid[min(ra, rb):max(ra, rb)+1, ca] = 1
    visible = sorted(visible)
    boxes = [(int(rows[r]), int(cols[c]), size) for r, c in visible]
    for r, c, s in boxes:
        grid[r:r+s, c:c+s] = 2
    data = _geometry(grid, boxes, 1)
    assert set(data['edges']) == set(_compressed(vertices, edges, visible))
    data['sites'] = visible
    return grid, data


@lru_cache(maxsize=24)
def _ballot_patterns(total):
    possibilities = []
    # All integer partitions with one strict winner and up to four colors.
    for high in range(1, total+1):
        for k in range(4):
            if k == 0:
                if high == total:
                    possibilities.append((high,))
                continue
            for losers in combinations(range(1, high+k-1), k):
                # Stars-and-bars encoded as nondecreasing values 1..high-1.
                vals = tuple(losers[i]-i for i in range(k))
                if max(vals) < high and high+sum(vals) == total:
                    possibilities.append((high,)+tuple(sorted(vals, reverse=True)))
    return tuple(possibilities)


def _ballot(rng, total):
    possibilities = _ballot_patterns(total)
    # Prefer actual competing ballots, while keeping single-color classes.
    contest = [x for x in possibilities if len(x) > 1]
    return _choice(rng, contest if contest and rng.random() < .9 else possibilities)


def _labels(rng, n, ends, shapes, palette, teaching, first_winners=None):
    nodes = [i for i in range(n) if i not in ends]
    rng.shuffle(nodes)
    k = len(shapes)
    if teaching == 'A':
        totals, patterns = [7, 7], [(3, 2, 2), (3, 2, 2)]
    elif teaching == 'B':
        totals, patterns = [4, 4], [(3, 1), (3, 1)]
    else:
        totals = [1]*k
        for _ in range(len(nodes)-k):
            totals[int(rng.integers(k))] += 1
        patterns = [_ballot(rng, count) for count in totals]
    colors, raws, cursor = {}, {}, 0
    for j, (shape, total, pattern) in enumerate(zip(shapes, totals, patterns)):
        local = list(palette)
        rng.shuffle(local)
        if teaching == 'A':
            # Unique area/box-pooled winner is palette[1]; shape 0 elects
            # palette[0]. Class 0's minority of palette[1] is also a genuine
            # reflection witness, rather than an unused orientation detail.
            local = [palette[0], palette[1], palette[2]] if j == 0 else [palette[1], palette[2], palette[3]]
        elif first_winners is not None:
            available = [c for c in local if c != first_winners[j]]
            win = _choice(rng, available)
            local = [win]+[c for c in local if c != win]
        specs = [color for color, count in zip(local, pattern) for _ in range(count)]
        rng.shuffle(specs)
        for node, color in zip(nodes[cursor:cursor+total], specs):
            if teaching == 'A':
                oriented = _mirror(shape) if j == 0 and color != palette[1] else shape
                raw = _choice(rng, _rotations(oriented))
            else:
                raw = _choice(rng, _variants(shape))
            colors[node], raws[node] = color, raw
        cursor += total
    assert cursor == len(nodes)
    return colors, raws


def _alternative_route(data, equivalence):
    if equivalence == 'ungated':
        allowed = set(range(len(data['adj'])))
    else:
        allowed, _, _ = _gate_set(data['raws'], data['colors'], data['ends'], equivalence)
        if allowed is None:
            return None
    return _shortest(data['adj'], data['ends'], allowed)


def _route_key(route):
    return None if route is None else (frozenset(route[0]), frozenset(route[1]))


def _is_length_witness(data):
    path = data['path']
    actual = sum(data['weights'][e] for e in path[1])
    return actual > _weighted(data['adj'], data['ends'], data['allowed'], data['weights'])


def _make_pair(rng, teaching, anchors, first_winners=None, test=False, known_shapes=()):
    roles = list(map(int, rng.permutation(10)))
    bg, connector, terminal = roles[:3]
    palette = roles[3:]
    for attempt in range(48):
        if teaching == 'A':
            size, nr, nc = 5, 4, 4
            topology = _random_topology(rng, nr, nc, 16, dense=True)
            shapes = anchors
        elif teaching == 'B':
            size = 5
            topology = _witness_topology(rng)
            shapes = anchors
        else:
            size = int(rng.choice([4, 5, 6], p=[.30, .55, .15]))
            max_axis = (28+2)//(size+2)
            nr, nc = int(rng.integers(2, max_axis+1)), int(rng.integers(2, max_axis+1))
            if nr*nc < 6:
                nr = 3
            topology = _random_topology(rng, nr, nc, min(nr*nc, int(rng.integers(6, 12))))
            pool = list(_pool(size-2))
            # Every query carries a new shape class, with its entire ballot
            # present locally. No shape-to-color lookup can solve the shift.
            if test:
                pool = [s for s in pool if s not in known_shapes]
            if not pool:
                continue
            k = int(rng.integers(1, min(3, len(pool), len(topology[2])-2)+1))
            shapes = [pool[int(i)] for i in rng.choice(len(pool), k, replace=False)]
        base, geo = _draw_skeleton(rng, topology, size)
        n = len(geo['boxes'])
        pairs = list(combinations(range(n), 2))
        if teaching == 'B':
            ends_fixed = [geo['sites'].index((1, 0)), geo['sites'].index((1, 3))]
        for trial in range(192):
            ends = ends_fixed if teaching == 'B' else list(_choice(rng, pairs))
            colors, raws = _labels(rng, n, ends, shapes, palette, teaching, first_winners)
            allowed, ballots, winners = _gate_set(raws, colors, ends)
            path = _shortest(geo['adj'], ends, allowed)
            if path is None or (len(path[0]) == 2 and rng.random() < .9):
                continue
            data = dict(geo, raws=raws, colors=colors, ends=ends, allowed=allowed,
                        ballots=ballots, winners=winners, path=path)
            if teaching == 'A':
                ungated = _alternative_route(data, 'ungated')
                rotated = _alternative_route(data, 'rotation')
                if ungated is None or rotated is None:
                    continue
                if len(ungated[0]) >= len(path[0]) or len(rotated[0]) >= len(path[0]):
                    continue
                if _route_key(_alternative_route(data, 'area')) == _route_key(path):
                    continue
            if teaching == 'B' and not _is_length_witness(data):
                continue
            inp = np.full(base.shape, bg, dtype=np.int8)
            inp[base == 1] = connector
            holes = {}
            for node, (r, c, s) in enumerate(geo['boxes']):
                color = terminal if node in ends else colors[node]
                inp[r:r+s, c:c+s] = color
                if node in ends:
                    continue
                raw = raws[node]
                hh, ww = _extent(raw)
                dr = int(rng.integers(1, s-hh))
                dc = int(rng.integers(1, s-ww))
                holes[node] = tuple((r+dr+y, c+dc+x) for y, x in raw)
                for y, x in holes[node]:
                    inp[y, x] = bg
            data.update(holes=holes, terminal=terminal)
            out = _paint(inp, data, path)
            # Independent margins, rigid transforms and palette roles per pair.
            turns, mirror = int(rng.integers(4)), bool(rng.integers(2))
            transformed = []
            for grid in (inp, out):
                grid = np.rot90(grid, turns)
                transformed.append(grid[:, ::-1] if mirror else grid)
            h, w = transformed[0].shape
            extra_h, extra_w = int(rng.integers(min(3, 30-h)+1)), int(rng.integers(min(3, 30-w)+1))
            top, left = int(rng.integers(extra_h+1)), int(rng.integers(extra_w+1))
            result = []
            for grid in transformed:
                placed = np.full((h+extra_h, w+extra_w), bg, dtype=np.int8)
                placed[top:top+h, left:left+w] = grid
                result.append(placed)
            return dict(zip(('input', 'output'), result))
    raise RuntimeError('bounded voting-network construction exhausted')


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.astype(np.int8).tobytes()


def _counts(num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError('counts must be integers')
        if not low <= int(value) <= high:
            raise ValueError('counts outside supported range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    pool = _pool(3)
    anchor_pairs = [(s, t) for s in pool if _mirror(s) not in _rotations(s)
                    for t in pool if t != s and len(t) == len(s) and
                    sorted(_extent(t)) == sorted(_extent(s))]
    anchors = _choice(rng, anchor_pairs)
    puzzle, seen, first_winners = {'train': [], 'test': []}, set(), None
    known_shapes = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            teaching = ('A' if i == 0 else 'B') if split == 'train' and i < 2 else None
            for _ in range(8):
                pair = _make_pair(rng, teaching, anchors, first_winners if teaching == 'B' else None,
                                  test=split == 'test', known_shapes=known_shapes)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    break
            else:
                raise RuntimeError('bounded distinct-input search exhausted')
            puzzle[split].append(pair)
            if split == 'train':
                d = _decode(pair['input'])
                known_shapes.update(d['winners'])
                if teaching == 'A':
                    first_winners = [d['winners'][s] for s in anchors]
    validate_generated_puzzle(puzzle)
    return puzzle


def features(input_grid):
    d = _decode(input_grid)
    ungated = _alternative_route(d, 'ungated')
    alternatives = {kind: _route_key(_alternative_route(d, kind)) != _route_key(d['path'])
                    for kind in ('area', 'box', 'rotation', 'translation')}
    return dict(shape=list(np.asarray(input_grid).shape), size=d['boxes'][0][2],
                stations=len(d['boxes']), classes=len(d['ballots']),
                selected=len(d['path'][0])-2, open=len(d['allowed'])-2,
                edges=len(d['edges']), cycles=len(d['edges'])-len(d['boxes'])+1,
                junctions=sum(len(a) >= 3 for a in d['adj']),
                dead_ends=sum(len(a) == 1 for a in d['adj']),
                shapes=tuple(sorted(d['ballots'])),
                ballots=tuple(sorted(tuple(sorted(c.values(), reverse=True)) for c in d['ballots'].values())),
                roles=(d['bg'], d['connector'], d['terminal']),
                gate_witness=ungated is not None and len(ungated[0]) < len(d['path'][0]),
                length_witness=_is_length_witness(d),
                alternatives=alternatives,
                unused_selected_branches=sum(any(e not in d['path'][1] for _, e in d['adj'][n])
                                             for n in d['path'][0][1:-1]))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train_features, history = set(), [], {}
    training_shapes, query_shapes = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            for grid in (inp, out):
                assert grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            d = _decode(inp)
            assert np.array_equal(_paint(inp, d, d['path']), out)
            assert not np.array_equal(inp, out)
            # Precisely the chosen holes and connector cells may change.
            expected_changes = set().union(*(set(d['holes'][n]) for n in d['path'][0][1:-1]),
                                           *(set(d['edge_cells'][e]) for e in d['path'][1]))
            assert set(map(tuple, np.argwhere(inp != out).tolist())) == expected_changes
            for n in d['path'][0][1:-1]:
                assert n in d['allowed']
                r, c, s = d['boxes'][n]
                assert np.all(out[r:r+s, c:c+s] == d['colors'][n])
            if split == 'train':
                train_features.append(features(inp))
                training_shapes.update(d['ballots'])
                for shape, winner in d['winners'].items():
                    history.setdefault(shape, set()).add(winner)
            else:
                query_shapes.append(set(d['ballots']))
    # These are evidence guarantees, not a split-specific solution rule.
    assert any(f['gate_witness'] for f in train_features)
    assert any(f['length_witness'] for f in train_features)
    for alternative in ('area', 'box', 'rotation', 'translation'):
        assert any(f['alternatives'][alternative] for f in train_features)
    assert any(any(len(b) >= 3 and b[0]*2 <= sum(b) for b in f['ballots']) for f in train_features)
    assert any(len(colors) > 1 for colors in history.values())
    assert all(shapes-training_shapes for shapes in query_shapes)


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                shape, payload = _key(pair[role])
                h.update(bytes(shape))
                h.update(payload)
    return h.hexdigest()


def self_test():
    """42 specified seeds, one puzzle resident at a time; no disk writes."""
    digests, shapes, vocabulary, sizes, counts, ballots, roles = set(), set(), set(), set(), set(), set(), set()
    witnesses = Counter()
    for index, seed in enumerate(HOST_SEEDS):
        puzzle = generate_puzzle(seed)
        digests.add(_digest(puzzle))
        if index < 2:
            assert _digest(puzzle) == _digest(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = features(pair['input'])
                shapes.add(tuple(f['shape']))
                vocabulary.update(f['shapes'])
                sizes.add(f['size'])
                counts.add(f['stations'])
                ballots.update(f['ballots'])
                roles.add(f['roles'])
                for name in ('gate_witness', 'length_witness', 'unused_selected_branches'):
                    witnesses[name] += bool(f[name])
        # Solving commutes with every rigid grid transformation and palette
        # permutation, including changing the literal background color.
        if index < 4:
            p = puzzle['test'][0]
            lut = np.array([3,7,1,9,0,5,8,2,6,4], dtype=np.int8)
            for turns in range(4):
                inp = lut[np.rot90(p['input'], turns)[:, ::-1]]
                expected = lut[np.rot90(p['output'], turns)[:, ::-1]]
                assert np.array_equal(solve(inp), expected)
    assert len(digests) == 42
    assert len(shapes) >= 20 and len(vocabulary) >= 20
    assert sizes == {4, 5, 6} and len(counts) >= 8 and len(ballots) >= 10
    assert len(roles) >= 50
    assert witnesses['gate_witness'] >= 42 and witnesses['length_witness'] >= 42
    assert witnesses['unused_selected_branches'] >= 20
    diamond = _adjacency(4, [(0,1),(1,3),(0,2),(2,3)])
    assert _shortest(diamond, (0,3)) is None  # a tied minimum is illegal
    assert _shortest(diamond, (0,3), {0,3}) is None  # gates can disconnect
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            _counts(a, b)
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    corrupted = generate_puzzle(0, 2, 1)
    corrupted['test'][0]['output'][0, 0] = (int(corrupted['test'][0]['output'][0, 0])+1)%10
    try:
        validate_generated_puzzle(corrupted)
    except AssertionError:
        pass
    else:
        raise AssertionError('off-route corruption accepted')
    return dict(seeds=42, pairs=210, distinct_puzzles=len(digests),
                grid_shapes=len(shapes), cavity_classes=len(vocabulary),
                station_counts=sorted(counts), frame_sizes=sorted(sizes),
                ballot_patterns=len(ballots), palette_roles=len(roles), witnesses=dict(witnesses))


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
