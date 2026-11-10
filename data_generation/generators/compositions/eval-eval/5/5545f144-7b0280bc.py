"""Keyed courier: shortest station journey subject to rigid-arrow docking.

Only standard-library modules and NumPy are used. The decoder recovers all roles,
ports, gates, and the actual arrow from pixels; generation state is never used by
solve. Coordinates are (row, column); directions are north, east, south, west.
"""
from collections import Counter
import heapq
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
# Right-facing, tip-relative geometries; shape identity is not used by solve.
TRAIN_MASKS = (
    ((0, 0), (-1, -1), (1, -1)),
    ((0, 0), (-1, -1), (0, -1), (1, -1)),
    ((0, 0), (0, -1), (-1, -2), (1, -2)),
    ((0, 0), (0, -1), (-1, -2), (0, -2), (1, -2)),
)
TEST_MASKS = (
    ((0, 0), (-1, -1), (1, -1), (-2, -2), (2, -2)),
    tuple([(0, 0)] + [(r, -d) for d in (1, 2) for r in range(-d, d + 1)]),
    ((0, 0), (0, -1), (-1, -2), (1, -2), (-2, -3), (2, -3)),
    tuple([(0, 0), (0, -1)] + [(r, -d-1) for d in (1, 2) for r in range(-d, d + 1)]),
)
HOST_SEEDS = (0, 879476808, 156369370, 77767103, 334163299, 486386366,
              448337585, 499954032, 87727046, 80074515, 882840901, 620212108,
              209623528, 801139083, 907107712, 324656115, 116571225, 579710962,
              449453511, 465520643, 91197484, 968452339, 724386232, 425879693,
              712056618, 308439205, 11780827, 188878002, 426789809, 612329962,
              374186021, 669635566, 800987143, 810899041, 745104546, 59088729,
              25968216, 292751092, 603772020, 868009455, 122075140, 764733762)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid dimensions'
    assert np.issubdtype(a.dtype, np.integer), 'noninteger cells'
    assert np.all((0 <= a) & (a <= 9)), 'invalid colors'
    return a


def _components(cells):
    remaining, result = set(cells), []
    while remaining:
        q = min(remaining)
        remaining.remove(q)
        component, stack = {q}, [q]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    component.add(p)
                    stack.append(p)
        result.append(component)
    return result


def _coords(mask):
    return set(map(tuple, np.argwhere(mask).tolist()))


def _rotate(p, turns):
    r, c = p
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _footprint(mask, center, direction):
    return {(center[0] + r, center[1] + c)
            for r, c in (_rotate(p, direction - 1) for p in mask)}


def _arrow_at(cells, center):
    """Infer pointing geometry relative to the visible dock center, without a bank."""
    candidates = []
    for direction in range(4):
        offsets = {_rotate((r-center[0], c-center[1]), 1-direction) for r, c in cells}
        if ((0, 0) in offsets and all(c <= 0 for r, c in offsets)
                and {p for p in offsets if p[1] == 0} == {(0, 0)}
                and any(r != 0 for r, c in offsets)
                and {(-r, c) for r, c in offsets} == offsets):
            candidates.append((tuple(sorted(offsets)), direction))
    assert len(candidates) == 1, 'arrow must have one projecting tip and symmetry axis'
    return candidates[0]


def _port(center, size, q):
    half = size // 2
    delta = (q[0]-center[0], q[1]-center[1])
    candidates = [d for d, (dr, dc) in enumerate(DIRS)
                  if delta == (dr*(half+1), dc*(half+1))]
    assert len(candidates) == 1, 'connector must meet the middle of a dock side'
    return candidates[0]


def _network(grid, rail):
    """Remove rails to segment filled square docks; then recover their ports."""
    blocks = _components(_coords((grid != 0) & (grid != rail)))
    assert 4 <= len(blocks) <= 16
    sizes, centers = [], []
    station_map = {}
    for i, block in enumerate(blocks):
        rows, cols = zip(*block)
        height, width = max(rows)-min(rows)+1, max(cols)-min(cols)+1
        assert height == width and height in (5, 7) and len(block) == height*width
        sizes.append(height)
        centers.append(((min(rows)+max(rows))//2, (min(cols)+max(cols))//2))
        station_map.update({p: i for p in block})
    assert len(set(sizes)) == 1
    size = sizes[0]
    rails = _components(_coords(grid == rail))
    edges, adjacency = [], [[] for _ in blocks]
    for e, cells in enumerate(rails):
        assert len(cells) >= 2
        # A connector is a nonbranching pixel chain; elbows do not add docks.
        degrees = {p: sum((p[0]+dr, p[1]+dc) in cells for dr, dc in DIRS) for p in cells}
        assert all(degree in (1, 2) for degree in degrees.values())
        ends = {p for p, degree in degrees.items() if degree == 1}
        assert len(ends) == 2, 'connector must be an open pixel chain'
        touched = {}
        for r, c in cells:
            for dr, dc in DIRS:
                n = station_map.get((r+dr, c+dc))
                if n is not None:
                    touched.setdefault(n, set()).add((r, c))
        assert len(touched) == 2 and all(len(v) == 1 for v in touched.values())
        assert set().union(*touched.values()) == ends, 'dock contacts must be rail endpoints'
        a, b = sorted(touched)
        da = _port(centers[a], size, next(iter(touched[a])))
        db = _port(centers[b], size, next(iter(touched[b])))
        edges.append((a, b))
        adjacency[a].append((b, e, da, (db+2) % 4))
        adjacency[b].append((a, e, db, (da+2) % 4))
    assert len(set(edges)) == len(edges), 'parallel connectors are excluded'
    assert all(1 <= len(a) <= 4 for a in adjacency)
    reached, stack = {0}, [0]
    while stack:
        for b, _, _, _ in adjacency[stack.pop()]:
            if b not in reached:
                reached.add(b)
                stack.append(b)
    assert len(reached) == len(blocks), 'disconnected network'
    return dict(blocks=blocks, centers=centers, size=size, rails=rails,
                edges=edges, adjacency=adjacency, lengths=[len(c) for c in rails])


def _read_input(value):
    grid = _grid(value)
    colors = set(map(int, np.unique(grid)))
    assert 0 in colors and len(colors) == 7, 'background plus six visible roles required'
    candidates = []
    for rail in sorted(colors - {0}):
        try:
            d = _network(grid, rail)
        except AssertionError:
            continue
        candidates.append((rail, d))
    assert len(candidates) == 1, 'rails and filled square docks must segment uniquely'
    rail, d = candidates[0]
    center_colors = [int(grid[p]) for p in d['centers']]
    counts = Counter(center_colors)
    markers = [c for c, n in counts.items() if n == len(d['blocks'])-2]
    assert len(markers) == 1
    marker = markers[0]
    specials = set(center_colors)-{marker}
    assert len(specials) == 2 and all(counts[c] == 1 for c in specials)
    goals = [c for c in specials if np.count_nonzero(grid == c) == 1]
    assert len(goals) == 1, 'goal is the single different center dot'
    goal = goals[0]
    ink = next(iter(specials-{goal}))
    start, target = center_colors.index(ink), center_colors.index(goal)
    arrow = _coords(grid == ink)
    assert len(arrow) >= 3 and arrow <= d['blocks'][start]
    mask, initial = _arrow_at(arrow, d['centers'][start])
    remaining = colors-{0, rail, marker, goal, ink}
    assert len(remaining) == 2
    floor = max(remaining, key=lambda c: np.count_nonzero(grid == c))
    wall = next(iter(remaining-{floor}))
    for n, block in enumerate(d['blocks']):
        assert sum(grid[p] == floor for p in block) > len(block)//2, 'floor must dominate each dock'
        for p in block:
            if p not in arrow and p != d['centers'][n]:
                assert int(grid[p]) in (floor, wall), 'extraneous dock marks'
    assert _coords(grid == marker) == set(d['centers'])-{d['centers'][start], d['centers'][target]}
    assert _coords(grid == goal) == {d['centers'][target]}
    walls = [_coords(grid == wall) & block for block in d['blocks']]
    fits = [[_footprint(mask, center, direction) <= block and
             not (_footprint(mask, center, direction) & obstacle)
             for direction in range(4)]
            for center, block, obstacle in zip(d['centers'], d['blocks'], walls)]
    assert fits[start][initial]
    assert any(entry[2] == initial for entry in d['adjacency'][start]), 'arrow points to no rail'
    d.update(mask=mask, initial=initial, start=start, target=target, walls=walls,
             fits=fits, roles=(0, floor, wall, rail, marker, goal, ink), arrow=arrow)
    return d


def _transitions(d, node, gates=True):
    for v, e, depart, arrive in d['adjacency'][node]:
        if v == d['start']:
            continue  # the courier cannot return to its already vacated origin
        if node == d['start'] and depart != d['initial']:
            continue
        if gates and not d['fits'][v][arrive]:
            continue
        yield v, e, arrive


def _shortest(d, gates=True, weighted=False):
    """Unique minimum-hop path, or a pixel-length optimum for an ablation."""
    n = len(d['blocks'])
    distance, ways, prev = [float('inf')]*n, [0]*n, [None]*n
    start, target = d['start'], d['target']
    distance[start], ways[start] = 0, 1
    queue = [(0, start)]
    while queue:
        cost, u = heapq.heappop(queue)
        if cost != distance[u]:
            continue
        if u == target:
            continue
        for v, e, arrive in _transitions(d, u, gates):
            nxt = cost + (d['lengths'][e] if weighted else 1)
            if nxt < distance[v]:
                distance[v], ways[v], prev[v] = nxt, ways[u], (u, e, arrive)
                heapq.heappush(queue, (nxt, v))
            elif nxt == distance[v]:
                ways[v] = min(2, ways[v]+ways[u])
    if not np.isfinite(distance[target]) or (not weighted and ways[target] != 1):
        return None
    nodes, edges, arrivals = [target], [], []
    while nodes[-1] != start:
        u, e, arrive = prev[nodes[-1]]
        nodes.append(u)
        edges.append(e)
        arrivals.append(arrive)
    return (nodes[::-1], edges[::-1], arrivals[::-1])


def _routes(d, gates=True, limit=20000):
    """Bounded exhaustive simple routes, used only for authored checks/construction."""
    found = []
    stack = [(d['start'], [d['start']], [], [])]
    while stack:
        u, nodes, edges, arrivals = stack.pop()
        if u == d['target']:
            found.append((nodes, edges, arrivals))
            assert len(found) <= limit, 'route enumeration bound exceeded'
            continue
        for v, e, arrive in _transitions(d, u, gates):
            if v not in nodes:
                stack.append((v, nodes+[v], edges+[e], arrivals+[arrive]))
    return sorted(found, key=lambda p: (len(p[1]), tuple(p[0])))


def _paint(grid, d, route):
    out = grid.copy()
    _, floor, _, _, marker, _, ink = d['roles']
    for p in d['arrow']:
        out[p] = floor
    out[d['centers'][d['start']]] = marker
    nodes, edges, arrivals = route
    for n, direction in zip(nodes[1:], arrivals):
        for p in _footprint(d['mask'], d['centers'][n], direction):
            out[p] = ink
    for e in edges:
        for p in d['rails'][e]:
            out[p] = ink
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    d = _read_input(input_grid)
    route = _shortest(d)
    assert route is not None, 'feasible minimum-station route must exist and be unique'
    return _paint(np.asarray(input_grid), d, route)


def _sample_graph(rng, size):
    max_axis = 4 if size == 5 else 3
    nr, nc = int(rng.integers(2, max_axis+1)), int(rng.integers(2, max_axis+1))
    if nr*nc < 6:
        return None
    vertices = [(r, c) for r in range(nr) for c in range(nc)]
    candidates = [(a, (a[0]+dr, a[1]+dc)) for a in vertices for dr, dc in ((1, 0), (0, 1))
                  if (a[0]+dr, a[1]+dc) in vertices]
    active = {vertices[int(rng.integers(len(vertices)))]}
    edges = []
    count = int(rng.integers(6, len(vertices)+1))
    while len(active) < count:
        options = [(a, b) for a, b in candidates if (a in active) != (b in active)]
        a, b = options[int(rng.integers(len(options)))]
        edges.append((a, b))
        active.update((a, b))
    extra = [(a, b) for a, b in candidates if a in active and b in active and (a, b) not in edges]
    rng.shuffle(extra)
    if not extra:
        return None
    edges += extra[:int(rng.integers(1, len(extra)+1))]
    adjacency = {v: [] for v in active}
    for a, b in edges:
        adjacency[a].append(b)
        adjacency[b].append(a)
    keep_probability = float(rng.uniform(.25, .8))
    visible = {v for v in active if len(adjacency[v]) != 2 or rng.random() < keep_probability}
    if len(visible) < 4:
        return None
    ids = {v: i for i, v in enumerate(sorted(visible))}
    compressed = []
    for a in sorted(visible):
        for nxt in adjacency[a]:
            chain, prev, cur = [a, nxt], a, nxt
            while cur not in visible:
                nxt = next(v for v in adjacency[cur] if v != prev)
                chain.append(nxt)
                prev, cur = cur, nxt
            if cur == a:
                return None
            if ids[a] < ids[cur]:
                compressed.append((ids[a], ids[cur], chain))
    if len({(a, b) for a, b, _ in compressed}) != len(compressed):
        return None
    def axis(n):
        slack = 30-(n*size+(n-1)*2)
        extra = int(rng.integers(slack+1))
        gaps = size+2+rng.multinomial(extra, [1/(n-1)]*(n-1))
        return [size//2]+list(size//2+np.cumsum(gaps))
    rows, cols = axis(nr), axis(nc)
    centers = [(int(rows[r]), int(cols[c])) for r, c in sorted(visible)]
    shape = (int(rows[-1])+size//2+1, int(cols[-1])+size//2+1)
    grid = np.zeros(shape, dtype=np.int8)
    half = size//2
    for r, c in centers:
        grid[r-half:r+half+1, c-half:c+half+1] = 1  # floor
        grid[r, c] = 4  # ordinary center
    for _, _, chain in compressed:
        for a, b in zip(chain, chain[1:]):
            p, q = (int(rows[a[0]]), int(cols[a[1]])), (int(rows[b[0]]), int(cols[b[1]]))
            dr = int(np.sign(q[0]-p[0])); dc = int(np.sign(q[1]-p[1]))
            for t in range(abs(q[0]-p[0])+abs(q[1]-p[1])+1):
                r, c = p[0]+t*dr, p[1]+t*dc
                if grid[r, c] == 0:
                    grid[r, c] = 3  # rail
    d = _network(grid, 3)
    assert d['centers'] == centers
    assert set(d['edges']) == {(a, b) for a, b, _ in compressed}
    return grid, d


def _protect(d, route):
    protected = [set() for _ in d['blocks']]
    protected[d['start']] = _footprint(d['mask'], d['centers'][d['start']], d['initial'])
    for node, direction in zip(route[0][1:], route[2]):
        protected[node].update(_footprint(d['mask'], d['centers'][node], direction))
    return protected


def _set_walls(rng, d, protected, forced=None):
    walls = [set() for _ in d['blocks']]
    if forced is not None:
        node, cell = forced
        walls[node].add(cell)
    for node, block in enumerate(d['blocks']):
        choices = sorted(block-protected[node]-{d['centers'][node]})
        # The floor remains the majority even in the arrow's origin dock.
        occupied = len(d['mask']) if node == d['start'] else 1
        cap = min(5, len(choices), len(block)-occupied-len(block)//2-1)
        number = int(rng.integers(0, cap+1))
        if choices and number:
            walls[node].update(choices[int(i)] for i in rng.choice(len(choices), number, replace=False))
    if not any(walls):
        for node, block in enumerate(d['blocks']):
            choices = sorted(block-protected[node]-{d['centers'][node]})
            if choices:
                walls[node].add(choices[int(rng.integers(len(choices)))])
                break
    d['walls'] = walls
    d['fits'] = [[not (_footprint(d['mask'], center, direction) & obstacle)
                  and _footprint(d['mask'], center, direction) <= block
                  for direction in range(4)]
                 for center, obstacle, block in zip(d['centers'], walls, d['blocks'])]


def _witnesses(d):
    chosen, ungated, geometric = _shortest(d), _shortest(d, gates=False), _shortest(d, weighted=True)
    assert chosen is not None
    gate = ungated is not None and len(chosen[1]) > len(ungated[1])
    cost = geometric is not None and sum(d['lengths'][e] for e in chosen[1]) > sum(d['lengths'][e] for e in geometric[1])
    return gate, cost


def _initial_witness(d):
    """A shorter journey would exist if the first departure could ignore the tip."""
    selected = _shortest(d)
    return any((other := _shortest(dict(d, initial=direction))) is not None
               and len(other[1]) < len(selected[1])
               for direction in range(4) if direction != d['initial'])


def _finish_pair(rng, base, d, roles):
    grid = base.copy()
    for wall in d['walls']:
        for p in wall:
            grid[p] = 2
    grid[d['centers'][d['target']]] = 5
    for p in _footprint(d['mask'], d['centers'][d['start']], d['initial']):
        grid[p] = 6
    lookup = np.array((0, *roles), dtype=np.int8)
    grid = np.rot90(lookup[grid], int(rng.integers(4)))
    if rng.random() < .5:
        grid = grid[:, ::-1]
    pads = []
    for n in grid.shape:
        total = int(rng.integers(min(4, 30-n)+1))
        low = int(rng.integers(total+1))
        pads.append((low, total-low))
    inp = np.pad(grid, pads, constant_values=0)
    decoded = _read_input(inp)
    actual = _shortest(decoded)
    assert actual is not None
    assert _witnesses(decoded) == _witnesses(d)
    return {'input': inp, 'output': _paint(inp, decoded, actual)}


def _fallback_pair(rng, mask, size, roles, mode):
    """Four docks on a three-row drawing: sparse detour versus dense direct leg.

    This is only a finite fallback, never a different transformation. S initially
    faces west toward A. A--G has a long left detour without intermediate docks;
    A--M--G is short but costs another stop. S also has a different first rail
    reaching G directly, which supplies the initial-facing witness.
    """
    pitch, half = size+2, size//2
    vertices = [(0, 1), (0, 2), (1, 1), (2, 1)]  # A, S, M, G
    centers = [(half+r*pitch, half+c*pitch) for r, c in vertices]
    base = np.zeros((3*size+4, 3*size+4), dtype=np.int8)
    for r, c in centers:
        base[r-half:r+half+1, c-half:c+half+1] = 1
        base[r, c] = 4
    chains = [[(0, 1), (0, 2)], [(0, 1), (0, 0), (1, 0), (2, 0), (2, 1)],
              [(0, 1), (1, 1)], [(1, 1), (2, 1)],
              [(0, 2), (1, 2), (2, 2), (2, 1)]]
    for chain in chains:
        for a, b in zip(chain, chain[1:]):
            p = (half+a[0]*pitch, half+a[1]*pitch)
            q = (half+b[0]*pitch, half+b[1]*pitch)
            dr, dc = int(np.sign(q[0]-p[0])), int(np.sign(q[1]-p[1]))
            for t in range(pitch+1):
                r, c = p[0]+t*dr, p[1]+t*dc
                if base[r, c] == 0:
                    base[r, c] = 3
    d = _network(base, 3)
    d.update(mask=mask, start=1, target=3, initial=3,
             fits=[[True]*4 for _ in vertices])
    short = _shortest(d, gates=False)
    dense = _shortest(d, gates=False, weighted=True)
    if mode == 'gate':
        protected = _protect(d, dense)
        last_direction = short[2][-1]
        choices = sorted(_footprint(mask, centers[3], last_direction)-protected[3]-{centers[3]})
        assert choices
        _set_walls(rng, d, protected, (3, choices[int(rng.integers(len(choices)))]))
        assert _witnesses(d)[0]
    else:
        direct = _shortest(dict(d, initial=2), gates=False)
        protected = [a | b | c for a, b, c in zip(_protect(d, short), _protect(d, dense), _protect(d, direct))]
        _set_walls(rng, d, protected)
        assert _witnesses(d)[1] and _initial_witness(d)
    return _finish_pair(rng, base, d, roles)


def _make_pair(rng, mask, size, roles, mode):
    # A small random network is searched for the requested teaching witness.
    # Fixed retry limits are followed by a constructive case of the same rule.
    for _ in range(160):
        sampled = _sample_graph(rng, size)
        if sampled is None:
            continue
        base, d = sampled
        d['mask'] = mask
        d['fits'] = [[True]*4 for _ in d['blocks']]
        options = [(s, t, depart) for s, adj in enumerate(d['adjacency'])
                   for v, e, depart, arrive in adj for t in range(len(d['blocks'])) if t not in (s, v)]
        rng.shuffle(options)
        for start, target, initial in options[:60]:
            d.update(start=start, target=target, initial=initial)
            short = _shortest(d, gates=False)
            if short is None or len(short[1]) < 2:
                continue
            if mode != 'gate' and not any(a != initial for a in short[2]):
                continue
            forced = None
            if mode == 'gate':
                routes = _routes(d, gates=False)
                alternatives = [p for p in routes if len(p[1]) > len(short[1])]
                rng.shuffle(alternatives)
                selected = None
                for candidate in alternatives[:30]:
                    protected = _protect(d, candidate)
                    choices = [(n, p) for n, direction in zip(short[0][1:], short[2])
                               for p in sorted(_footprint(mask, d['centers'][n], direction)-protected[n]-{d['centers'][n]})]
                    if choices:
                        selected = candidate
                        forced = choices[int(rng.integers(len(choices)))]
                        break
                if selected is None:
                    continue
                protected = _protect(d, selected)
            else:
                protected = _protect(d, short)
                if mode == 'count':
                    geometric = _shortest(d, gates=False, weighted=True)
                    if geometric is None or sum(d['lengths'][e] for e in short[1]) <= sum(d['lengths'][e] for e in geometric[1]):
                        continue
                    other = _protect(d, geometric)
                    protected = [a | b for a, b in zip(protected, other)]
            _set_walls(rng, d, protected, forced)
            route = _shortest(d)
            if route is None:
                continue
            gate, cost = _witnesses(d)
            if ((mode == 'gate' and not gate) or (mode == 'count' and (not cost or not _initial_witness(d)))
                    or not any(a != initial for a in route[2])):
                continue
            return _finish_pair(rng, base, d, roles)
    return _fallback_pair(rng, mask, size, roles, mode)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _key(value):
    a = np.asarray(value, dtype=np.int8)
    return a.shape, a.tobytes()


def _distinct_fallback(rng, mask, size, roles, mode, seen):
    pair = _fallback_pair(rng, mask, size, roles, mode)
    fg = np.argwhere(pair['input'] != 0)
    lo, hi = fg.min(0), fg.max(0)
    cropped = {k: v[lo[0]:hi[0]+1, lo[1]:hi[1]+1] for k, v in pair.items()}
    h, w = cropped['input'].shape
    # The fallback's occupied extent is at most 25 on either axis, giving at
    # least 36 translations on a 30x30 board against at most 13 earlier inputs.
    assert h <= 25 and w <= 25
    for top in range(31-h):
        for left in range(31-w):
            pads = ((top, 30-h-top), (left, 30-w-left))
            placed = {k: np.pad(v, pads, constant_values=0) for k, v in cropped.items()}
            if _key(placed['input']) not in seen:
                return placed
    raise AssertionError('more prior inputs than the legal count limit permits')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice(np.arange(1, 10), 6, replace=False)))
    shared_palette = bool(rng.random() < .2)
    train_masks = rng.permutation(len(TRAIN_MASKS)).tolist()
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, number in (('train', num_train), ('test', num_test)):
        for i in range(number):
            if split == 'train':
                mask = TRAIN_MASKS[train_masks[i]] if i < 2 else TRAIN_MASKS[int(rng.integers(len(TRAIN_MASKS)))]
                mode = ('gate', 'count')[i] if i < 2 else str(rng.choice(['free', 'gate', 'count'], p=[.6, .2, .2]))
            else:
                mask = TEST_MASKS[int(rng.integers(len(TEST_MASKS)))]
                mode = str(rng.choice(['free', 'gate', 'count'], p=[.4, .35, .25]))
            radius = max(max(abs(r), abs(c)) for r, c in mask)
            size = 7 if radius == 3 else int(rng.choice([5, 7], p=[.75, .25]))
            roles = palette if shared_palette else tuple(map(int, rng.choice(np.arange(1, 10), 6, replace=False)))
            for _ in range(12):
                pair = _make_pair(rng, mask, size, roles, mode)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                pair = _distinct_fallback(rng, mask, size, roles, mode, seen)
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, info = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            d = _read_input(inp)
            route = _shortest(d)
            assert route is not None
            assert np.array_equal(out, _paint(inp, d, route)), 'incorrect route or rigid stamps'
            assert not np.array_equal(inp, out)
            # Gate pixels and the full drawing outside the moved/stamped/selected
            # cells must be preserved. Every selected imprint is the actual mask.
            mutable = set(d['arrow']) | {d['centers'][d['start']]}
            for node, direction in zip(route[0][1:], route[2]):
                stamp = _footprint(d['mask'], d['centers'][node], direction)
                assert stamp <= d['blocks'][node] and not stamp & d['walls'][node]
                assert all(out[p] == d['roles'][-1] for p in stamp)
                mutable |= stamp
            for e in route[1]:
                mutable |= d['rails'][e]
            assert all(inp[r, c] == out[r, c] for r in range(inp.shape[0]) for c in range(inp.shape[1])
                       if (r, c) not in mutable), 'off-route alteration'
            assert out[d['centers'][d['start']]] == d['roles'][4]
            assert all(out[p] == d['roles'][2] for wall in d['walls'] for p in wall)
            assert np.count_nonzero(out == d['roles'][-1]) == (
                len(d['mask'])*len(route[1])+sum(d['lengths'][e] for e in route[1])), 'stamp/trace area mismatch'
            info[split].append(d)
    train = info['train']
    assert any(_witnesses(d)[0] for d in train), 'missing shape-gate training witness'
    assert any(_witnesses(d)[1] for d in train), 'missing station-count versus rail-length witness'
    assert any(_initial_witness(d) for d in train), 'initial facing must constrain a training answer'
    assert len({d['mask'] for d in train}) >= 2, 'must teach actual shape preservation'
    assert all(any(a != d['initial'] for a in _shortest(d)[2]) for d in train[:2])
    # Every query extends the visible head width, with no new convention.
    assert all(max(abs(r) for r, c in d['mask']) == 1 for d in train)
    assert all(max(abs(r) for r, c in d['mask']) == 2 for d in info['test'])
    return None


def _puzzle_key(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def _features(pair):
    inp = np.asarray(pair['input'])
    d = _read_input(inp)
    route = _shortest(d)
    normalized = np.zeros(inp.shape, dtype=np.int8)
    for role, color in enumerate(d['roles']):
        normalized[inp == color] = role
    fg = np.argwhere(inp != 0)
    lo, hi = fg.min(0), fg.max(0)
    normalized = normalized[lo[0]:hi[0]+1, lo[1]:hi[1]+1]
    bent = sum(len({r for r, c in cells}) > 1 and len({c for r, c in cells}) > 1 for cells in d['rails'])
    return dict(stations=len(d['blocks']), stops=len(route[1]), size=d['size'],
                mask=d['mask'], initial=d['initial'], final=route[2][-1],
                gate=_witnesses(d)[0], count=_witnesses(d)[1], bent=bent,
                degree=tuple(sorted(map(len, d['adjacency']))),
                loops=len(d['edges'])-len(d['blocks'])+1, roles=d['roles'],
                normalized=_key(normalized), shape=inp.shape)


def self_test():
    """Bounded 42-seed semantic/diversity audit; no external files or model calls."""
    records, signatures = [], set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _read_input(pair['input'])
                routes = _routes(d)
                assert routes and (len(routes) == 1 or len(routes[0][1]) < len(routes[1][1]))
                assert _shortest(d) == routes[0], 'shortest path differs from enumeration'
                assert np.array_equal(solve(pair['input']), pair['output'])
                gate, cost = _witnesses(d)
                if gate:
                    assert not np.array_equal(_paint(pair['input'], d, _shortest(d, gates=False)), pair['output'])
                if cost:
                    assert not np.array_equal(_paint(pair['input'], d, _shortest(d, weighted=True)), pair['output'])
                f = _features(pair)
                f['split'] = split
                records.append(f)
        if seed in HOST_SEEDS[:2]:
            assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
    assert len(signatures) == 42
    assert len({r['normalized'] for r in records}) >= 180
    assert len({r['shape'] for r in records}) >= 35
    assert len({r['stations'] for r in records}) >= 8
    assert len({r['stops'] for r in records}) >= 5
    assert len({r['roles'] for r in records}) >= 80
    for split in ('train', 'test'):
        rows = [r for r in records if r['split'] == split]
        assert {r['initial'] for r in rows} == set(range(4))
        assert {r['final'] for r in rows} == set(range(4))
        assert len({r['mask'] for r in rows}) == 4
        assert {r['size'] for r in rows} == {5, 7}
        assert any(r['bent'] for r in rows)
        assert any(1 in r['degree'] and max(r['degree']) >= 3 for r in rows)
        assert any(r['loops'] >= 2 for r in rows)
        assert any(r['gate'] for r in rows) and any(r['count'] for r in rows)
    # Certify every fallback geometry/size/mode combination, including the wider
    # query keys. The role colors are permuted to exercise the pixel decoder.
    rng = np.random.default_rng(420)
    for mask in TRAIN_MASKS+TEST_MASKS:
        for size in (5, 7):
            if max(max(abs(r), abs(c)) for r, c in mask) > size//2:
                continue
            for mode in ('free', 'gate', 'count'):
                roles = tuple(map(int, rng.choice(np.arange(1, 10), 6, replace=False)))
                pair = _fallback_pair(rng, mask, size, roles, mode)
                d = _read_input(pair['input'])
                assert np.array_equal(solve(pair['input']), pair['output'])
                assert _witnesses(d)[0 if mode == 'gate' else 1]
                if mode != 'gate':
                    assert _initial_witness(d)
    seen = set()
    for _ in range(14):
        pair = _distinct_fallback(rng, TRAIN_MASKS[0], 7, (1, 2, 3, 4, 5, 6), 'count', seen)
        assert _key(pair['input']) not in seen
        assert np.array_equal(solve(pair['input']), pair['output'])
        seen.add(_key(pair['input']))
    # Explicit count limits, types, partial-specified defaults, and determinism.
    for a, b in ((2, 1), (10, 1), (10, 4), (3, 2), (5, 4)):
        p = generate_puzzle(17, a, b)
        assert len(p['train']) == a and len(p['test']) == b
    assert len(generate_puzzle(19, num_train=6)['train']) == 6
    assert len(generate_puzzle(19, num_test=2)['test']) == 2
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # An equal-hop diamond after the forced first leg must be rejected.
    tied = dict(blocks=[set() for _ in range(5)], start=0, target=4, initial=1,
                fits=[[True]*4 for _ in range(5)], lengths=[1]*5,
                adjacency=[[(1, 0, 1, 1)], [(2, 1, 0, 0), (3, 2, 2, 2)],
                           [(4, 3, 1, 1)], [(4, 4, 1, 1)], []])
    assert _shortest(tied) is None, 'tied feasible routes accepted'
    assert len(_routes(tied)) == 2
    # A bad label is rejected; no second authoring implementation is claimed.
    p = generate_puzzle(0, 2, 1)
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
