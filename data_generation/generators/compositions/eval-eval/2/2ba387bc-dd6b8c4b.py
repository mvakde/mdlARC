"""Ranked one-way portals feeding a capacity-nine collector.

Only standard-library and NumPy dependencies. Solving never consults generation,
training examples, an RNG, or files. Diagnostics retain aggregate scalars only.
"""
from collections import deque
import hashlib
from heapq import heappop, heappush
import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 436762071, 15134908, 827735678, 874443387, 712478633,
              719312944, 971391819, 550060303, 416084687, 991998201,
              155038386, 406237833, 62018649, 475706123, 439736006,
              69119313, 360952989, 513594570, 895051092, 993039771,
              471009423, 411891226, 415823902, 949014790, 530874130,
              664767769, 151024426, 87488020, 502107018, 667103941,
              609222494, 899084284, 148706519, 347650258, 553464513,
              424856311, 371321145, 749471166, 419288169, 386972202,
              512173677)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not (_integer(a) and _integer(b) and 2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("Require integer counts 2..10 / 1..4 with num_train > num_test.")
    return int(a), int(b)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid grid size.")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)),
             "Grid must contain integer ARC colors.")
    return a


def _components(mask):
    h, w = mask.shape
    labels = np.full(mask.shape, -1, dtype=np.int16)
    groups = []
    for start in np.flatnonzero(mask):
        r, c = divmod(int(start), w)
        if labels[r, c] >= 0:
            continue
        label = len(groups)
        labels[r, c] = label
        cells = [(r, c)]
        for rr, cc in cells:
            for dr, dc in STEPS:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] < 0:
                    labels[nr, nc] = label
                    cells.append((nr, nc))
        groups.append(cells)
    return labels, groups


def _tile(size, color, hollow):
    a = np.full((size, size), color, dtype=np.int8)
    if hollow:
        a[1:-1, 1:-1] = 0
    return a


def _port_objects(grid, color):
    _, groups = _components(grid == color)
    objects = []
    for cells in groups:
        r = min(x[0] for x in cells)
        c = min(x[1] for x in cells)
        s = max(x[0] for x in cells) - r + 1
        if s < 3 or max(x[1] for x in cells) - c + 1 != s:
            return None
        patch = grid[r:r+s, c:c+s]
        hollow = np.array_equal(patch, _tile(s, color, True))
        if not hollow and not np.all(patch == color):
            return None
        objects.append((r, c, s, hollow))
    if not objects or len({p[2] for p in objects}) != 1:
        return None
    hollow = sorted(p for p in objects if p[3])
    solid = sorted(p for p in objects if not p[3])
    if not hollow or not solid:
        return None
    if any(len({p[0] for p in q}) != len(q) for q in (hollow, solid)):
        return None
    return hollow, solid


def _exterior(obj, floor):
    r, c, s, _ = obj
    h, w = floor.shape
    cells = [(r-1, c+i) for i in range(s)] + [(r+s, c+i) for i in range(s)]
    cells += [(r+i, c-1) for i in range(s)] + [(r+i, c+s) for i in range(s)]
    return tuple(rr*w+cc for rr, cc in cells
                 if 0 <= rr < h and 0 <= cc < w and floor[rr, cc])


def _scene(wall, hollow, solid, counter):
    floor = ~wall.copy()
    for r, c, s, _ in hollow + solid:
        floor[r:r+s, c:c+s] = False
    scene = dict(wall=wall, floor=floor, hollow=hollow, solid=solid, counter=counter)
    scene['entries'] = [_exterior(p, floor) for p in hollow]
    scene['exits'] = [_exterior(p, floor) for p in solid]
    return scene


def _parse(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    counters = []
    for color, count in zip(values, counts):
        if color in (0, 5) or count != 8:
            continue
        rr, cc = np.where(a == color)
        r, c = int(rr.min()), int(cc.min())
        if int(rr.max()) != r+2 or int(cc.max()) != c+2:
            continue
        m = int(a[r+1, c+1])
        if m not in (0, 5, color) and np.count_nonzero(a == m) == 1:
            counters.append((r, c, int(color), m))
    _require(len(counters) == 1, "Need one marked 3x3 collector.")
    r, c, slot, marker = counters[0]
    remaining = set(map(int, values)) - {0, 5, slot, marker}
    _require(len(remaining) == 2, "Need a portal color and a visible payload color.")
    interpretations = []
    for color in sorted(remaining):
        objects = _port_objects(a, color)
        if objects is not None:
            interpretations.append((color, objects))
    _require(len(interpretations) == 1, "Ambiguous portal vocabulary.")
    port, (hollow, solid) = interpretations[0]
    payload = next(iter(remaining - {port}))
    scene = _scene(a == 5, hollow, solid, (r, c))
    scene.update(grid=a, port=port, payload=payload, slot=slot, marker=marker)
    # All complete footprints have a one-cell empty halo. This makes all sides
    # unambiguous and prevents a wall from silently selecting a privileged gate.
    for rr, cc, s, _ in hollow + solid:
        halo = a[max(0, rr-1):min(a.shape[0], rr+s+1),
                 max(0, cc-1):min(a.shape[1], cc+s+1)].copy()
        halo[rr-max(0, rr-1):rr-max(0, rr-1)+s,
             cc-max(0, cc-1):cc-max(0, cc-1)+s] = 0
        _require(np.all((halo == 0) | (halo == payload)), "Portal halo is obstructed.")
    _require(np.all(scene['floor'][r:r+3, c:c+3]), "Collector overlaps a portal/wall.")
    return scene


def _links(scene, mode='rank'):
    hs, ss = scene['hollow'], scene['solid']
    pairs = list(zip(range(len(hs)), range(len(ss))))
    if mode == 'none':
        pairs = []
    elif mode == 'greedy':
        # Competing global scan from A's review: a repeated shape starts a
        # new row; earlier holes are never backfilled by the other class.
        events = sorted([(p[0], p[1], 'h', i) for i, p in enumerate(hs)] +
                        [(p[0], p[1], 's', i) for i, p in enumerate(ss)])
        pairs, row = [], {}
        for _, _, kind, index in events:
            if kind in row:
                if len(row) == 2:
                    pairs.append((row['h'], row['s']))
                row = {}
            row[kind] = index
        if len(row) == 2:
            pairs.append((row['h'], row['s']))
    elif mode == 'reverse':
        pairs = list(zip(range(len(hs)), reversed(range(len(ss)))))
    elif mode == 'horizontal':
        pairs = list(zip(sorted(range(len(hs)), key=lambda i: (hs[i][1], hs[i][0])),
                         sorted(range(len(ss)), key=lambda i: (ss[i][1], ss[i][0]))))
    elif mode == 'nearest':
        pairs = [(i, min(range(len(ss)), key=lambda j:
                        (abs(h[0]-ss[j][0])+abs(h[1]-ss[j][1]), ss[j][0], ss[j][1])))
                 for i, h in enumerate(hs)]
    elif mode == 'broadcast':
        pairs = [(i, j) for i in range(len(hs)) for j in range(len(ss))]
    links = {}
    for i, j in pairs:
        entry, exit_ = scene['entries'][i], scene['exits'][j]
        for cell in entry:
            links.setdefault(cell, set()).update(exit_)
        if mode == 'undirected':
            for cell in exit_:
                links.setdefault(cell, set()).update(entry)
    return {k: tuple(sorted(v)) for k, v in links.items()}


def _distances(scene, mode='rank', hop_limit=None, ignore_walls=False):
    floor = scene['floor'].copy()
    if ignore_walls:
        floor[scene['wall']] = True
    h, w = floor.shape
    links = _links(scene, mode)
    layers = 1 if hop_limit is None else hop_limit+1
    distance = np.full((layers, h*w), -1, dtype=np.int16)
    r, c = scene['counter']
    queue = deque()
    for rr in range(r, r+3):
        for cc in range(c, c+3):
            cell = rr*w+cc
            distance[0, cell] = 0
            queue.append((0, cell))
    flat_floor = floor.ravel()
    while queue:
        layer, cell = queue.popleft()
        rr, cc = divmod(cell, w)
        nd = int(distance[layer, cell])+1
        ordinary = []
        if rr: ordinary.append(cell-w)
        if rr+1 < h: ordinary.append(cell+w)
        if cc: ordinary.append(cell-1)
        if cc+1 < w: ordinary.append(cell+1)
        for dest in ordinary:
            if flat_floor[dest] and distance[layer, dest] < 0:
                distance[layer, dest] = nd
                queue.append((layer, dest))
        if hop_limit is None or layer < hop_limit:
            nxt = 0 if hop_limit is None else layer+1
            for dest in links.get(cell, ()):
                if distance[nxt, dest] < 0:
                    distance[nxt, dest] = nd
                    queue.append((nxt, dest))
    if layers == 1:
        return distance[0].reshape(h, w)
    best = np.where(distance < 0, 32000, distance).min(axis=0)
    best[best == 32000] = -1
    return best.reshape(h, w)


def _selection(scene, distance, strict=True, capacity=9):
    cargo = (scene['grid'] == scene['payload']) & (distance >= 0)
    cells = np.flatnonzero(cargo)
    ordered = sorted(map(int, cells), key=lambda i: (int(distance.flat[i]), i))
    if strict and len(ordered) > capacity:
        _require(distance.flat[ordered[capacity-1]] < distance.flat[ordered[capacity]],
                 "Capacity boundary cuts a distance tie.")
    return ordered[:capacity]


def _weighted_distances(scene, portal_cost):
    """Counterfactual route costs for demonstration sufficiency checks only."""
    floor = scene['floor']
    h, w = floor.shape
    r, c = scene['counter']
    distance = np.full(h*w, 32000, dtype=np.int32)
    queue = []
    for rr in range(r, r+3):
        for cc in range(c, c+3):
            cell = rr*w+cc
            distance[cell] = 0
            heappush(queue, (0, cell))
    links = _links(scene)
    while queue:
        depth, cell = heappop(queue)
        if distance[cell] != depth:
            continue
        rr, cc = divmod(cell, w)
        neighbors = [(nr*w+nc, 1) for dr, dc in STEPS
                     if 0 <= (nr := rr+dr) < h and 0 <= (nc := cc+dc) < w and floor[nr, nc]]
        neighbors += [(dest, portal_cost) for dest in links.get(cell, ())]
        for dest, cost in neighbors:
            if depth+cost < distance[dest]:
                distance[dest] = depth+cost
                heappush(queue, (depth+cost, dest))
    distance[distance == 32000] = -1
    return distance.reshape(h, w)


def _cost_relation(scene, cost, actual_distance=None):
    """Positive: cost is refuted. Zero: ambiguous tie. Negative: same answer."""
    actual = _distances(scene) if actual_distance is None else actual_distance
    selected = _selection(scene, actual)
    mask = np.zeros(actual.shape, dtype=bool)
    mask.flat[selected] = True
    alternative = _weighted_distances(scene, cost)
    retained = (scene['grid'] == scene['payload']) & (alternative >= 0) & ~mask
    if not retained.any() or not selected:
        return -1
    return int(alternative[mask].max()-alternative[retained].min())


def _render(scene, selected, column_major=False, fill_count=None):
    out = scene['grid'].copy()
    out.flat[selected] = 0
    r, c = scene['counter']
    count = len(selected) if fill_count is None else fill_count
    for i in range(min(9, count)):
        dr, dc = (i % 3, i // 3) if column_major else divmod(i, 3)
        out[r+dr, c+dc] = scene['payload']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive directed portal routes, nearest-nine collection and the prefix."""
    scene = _parse(input_grid)
    return _render(scene, _selection(scene, _distances(scene)))


def _alternative(scene, mode):
    distance = _distances(scene, mode if mode in
                          ('none', 'greedy', 'reverse', 'horizontal', 'nearest', 'broadcast', 'undirected')
                          else 'rank', hop_limit=1 if mode == 'one_hop' else None,
                          ignore_walls=mode == 'no_walls')
    if mode == 'manhattan':
        rr, cc = np.indices(distance.shape)
        r, c = scene['counter']
        direct = np.maximum(r-rr, 0)+np.maximum(rr-r-2, 0)
        direct += np.maximum(c-cc, 0)+np.maximum(cc-c-2, 0)
        distance = np.where(distance >= 0, direct, -1)
    selected = _selection(scene, distance, strict=False,
                          capacity=scene['grid'].size if mode == 'no_capacity' else 9)
    fill_count = None
    if mode == 'components':
        mask = np.zeros(scene['grid'].shape, dtype=bool)
        mask.flat[selected] = True
        fill_count = len(_components(mask)[1])
    return _render(scene, selected, mode == 'column_major', fill_count)


def _sites(blocked, size):
    """Legal square footprints with a clipped one-cell clear halo."""
    h, w = blocked.shape
    integral = np.pad(blocked.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    rr, cc = np.indices((h-size+1, w-size+1))
    r0, c0 = np.maximum(rr-1, 0), np.maximum(cc-1, 0)
    r1, c1 = np.minimum(rr+size+1, h), np.minimum(cc+size+1, w)
    sums = integral[r1, c1]-integral[r0, c1]-integral[r1, c0]+integral[r0, c0]
    return np.argwhere(sums == 0)


def _walls(rng, shape, size, broad):
    h, w = shape
    rooms = [(0, 0, h, w)]
    cuts = []
    minimum = size+3
    target = int(rng.integers(4, 8))
    while len(rooms) < target:
        choices = [(i, axis) for i, (r0, c0, r1, c1) in enumerate(rooms)
                   for axis, length in ((0, r1-r0), (1, c1-c0))
                   if length >= 2*minimum+1]
        if not choices:
            break
        i, axis = choices[int(rng.integers(len(choices)))]
        r0, c0, r1, c1 = rooms.pop(i)
        if axis == 0:
            cut = int(rng.integers(r0+minimum, r1-minimum))
            cuts.append((axis, cut, c0, c1))
            rooms.extend([(r0, c0, cut, c1), (cut+1, c0, r1, c1)])
        else:
            cut = int(rng.integers(c0+minimum, c1-minimum))
            cuts.append((axis, cut, r0, r1))
            rooms.extend([(r0, c0, r1, cut), (r0, cut+1, r1, c1)])
    wall = np.zeros(shape, dtype=bool)
    enclosure_mode = rng.random() < .25
    if enclosure_mode:
        for r0, c0, r1, c1 in rooms:
            if rng.random() < .85:
                wall[r0, c0:c1] = wall[r1-1, c0:c1] = True
                wall[r0:r1, c0] = wall[r0:r1, c1-1] = True
    else:
        for axis, cut, a, b in cuts:
            if axis == 0: wall[cut, a:b] = True
            else: wall[a:b, cut] = True
            if broad and rng.random() < .3:
                opening = int(rng.integers(a, b))
                length = int(rng.integers(1, 4))
                if axis == 0: wall[cut, opening:min(b, opening+length)] = False
                else: wall[opening:min(b, opening+length), cut] = False
    # Partial strokes can produce detours and dead ends inside the chambers.
    for _ in range(int(rng.integers(4))):
        r0, c0, r1, c1 = rooms[int(rng.integers(len(rooms)))]
        rr = int(rng.integers(r0+1, r1-1))
        cc = int(rng.integers(c0+1, c1-1))
        if rng.integers(2):
            wall[rr, cc:min(c1-1, cc+int(rng.integers(2, 7)))] = True
        else:
            wall[rr:min(r1-1, rr+int(rng.integers(2, 7))), cc] = True
    return wall


def _room_graph(scene, labels, groups, mode='rank'):
    edges = [set() for _ in groups]
    w = labels.shape[1]
    for entry, exits in _links(scene, mode).items():
        source = int(labels.flat[entry])
        for dest in exits:
            target = int(labels.flat[dest])
            if source >= 0 and target >= 0:
                edges[source].add(target)
    return edges


def _hop_distances(edges, source):
    distance = {source: 0}
    queue = [source]
    for node in queue:
        for neighbor in sorted(edges[node]):
            if neighbor not in distance:
                distance[neighbor] = distance[node]+1
                queue.append(neighbor)
    return distance


def _fallback(size, nh, ns):
    """A roomy constructive path of four chambers, still using the same rule."""
    wall = np.zeros((30, 30), dtype=bool)
    wall[14, :] = wall[:, 14] = True
    hs = [(1, 1, size, True), (2, 22, size, True),
          (17, 7, size, True), (23, 16, size, True)][:nh]
    ss = [(8, 16, size, False), (16, 1, size, False),
          (17, 16, size, False), (24, 22, size, False)][:ns]
    return wall, hs, ss


def _layout(rng, size, nh, ns, obligation, attempt):
    if attempt == 255:
        wall, hs, ss = _fallback(size, nh, ns)
    else:
        low = 25 if nh+ns >= 5 else max(17, 2*(size+3)+1)
        shape = tuple(int(x) for x in rng.integers(low, 31, size=2))
        wall = _walls(rng, shape, size, obligation == 'ordinary')
        blocked = wall.copy()
        hs, ss = [], []
        kinds = list(rng.permutation([True]*nh+[False]*ns))
        for hollow in kinds:
            sites = _sites(blocked, size)
            queue = hs if hollow else ss
            used_rows = {p[0] for p in queue}
            sites = np.array([p for p in sites if int(p[0]) not in used_rows], dtype=int)
            if not len(sites):
                return None
            r, c = map(int, sites[int(rng.integers(len(sites)))])
            queue.append((r, c, size, bool(hollow)))
            blocked[r:r+size, c:c+size] = True
        hs.sort()
        ss.sort()
    scene = _scene(wall, hs, ss, (0, 0))
    labels, groups = _components(scene['floor'])
    edges = _room_graph(scene, labels, groups)
    undir = _room_graph(scene, labels, groups, 'undirected')
    depth = {'partial': 1, 'excess': 2, 'shift': 3, 'ordinary': 0}[obligation]
    blocked = ~scene['floor']
    sites = _sites(blocked, 3)
    if not len(sites):
        return None
    eligible = []
    depths = {}
    for source in sorted({int(labels[r, c]) for r, c in sites}):
        hop = _hop_distances(edges, source)
        if max(hop.values()) < depth:
            continue
        if obligation in ('partial', 'excess') and len(hop) == len(groups):
            continue
        if obligation == 'partial' and len(_hop_distances(undir, source)) == len(hop):
            continue
        depths[source] = hop
        eligible.append(source)
    if not eligible:
        return None
    source = int(rng.choice(eligible))
    sites = np.array([p for p in sites if labels[tuple(p)] == source])
    r, c = map(int, sites[int(rng.integers(len(sites)))])
    scene['counter'] = (r, c)
    # Minimum teleport count is derived from the actual floor components.
    hop_map = np.full(wall.shape, -1, dtype=np.int16)
    for component, count in depths[source].items():
        hop_map[labels == component] = count
    scene['hops'] = hop_map
    return scene


def _choose(rng, pool, n, forced=()):
    selected = list(dict.fromkeys(map(int, forced)))
    cells = [int(x) for x in np.flatnonzero(pool) if int(x) not in selected]
    if len(selected) > n or len(cells)+len(selected) < n:
        return None
    if n > len(selected):
        selected.extend(map(int, rng.choice(cells, n-len(selected), replace=False)))
    return selected


def _supply(rng, scene, distance, load, obligation):
    outside = scene['floor'].copy()
    r, c = scene['counter']
    outside[max(0, r-1):r+4, max(0, c-1):c+4] = False
    active = outside & (distance >= 0)
    inactive = outside & (distance < 0)
    required_depth = {'partial': 1, 'excess': 2, 'shift': 3, 'ordinary': 0}[obligation]
    deep = active & (scene['hops'] >= required_depth)
    if not deep.any() or active.sum() < load:
        return None
    if load <= 9:
        forced = [int(rng.choice(np.flatnonzero(deep)))] if load else []
        if obligation == 'partial':
            # An adjacent selected pair visibly teaches cells versus objects.
            ar, ac = divmod(forced[0], distance.shape[1])
            neighbors = [(ar+dr, ac+dc) for dr, dc in STEPS
                         if 0 <= ar+dr < distance.shape[0] and
                         0 <= ac+dc < distance.shape[1] and active[ar+dr, ac+dc]]
            if not neighbors:
                return None
            br, bc = neighbors[int(rng.integers(len(neighbors)))]
            forced.append(br*distance.shape[1]+bc)
        chosen = _choose(rng, active, load, forced)
    else:
        rr, cc = np.indices(distance.shape)
        direct = np.maximum(r-rr, 0)+np.maximum(rr-r-2, 0)
        direct += np.maximum(c-cc, 0)+np.maximum(cc-c-2, 0)
        chosen = None
        for threshold in rng.permutation(np.unique(distance[active])):
            near, far = active & (distance <= threshold), active & (distance > threshold)
            if near.sum() < 9 or far.sum() < load-9 or not (deep & near).any():
                continue
            forced_near = [int(rng.choice(np.flatnonzero(deep & near)))]
            forced_far = []
            if obligation == 'excess':
                a = int(np.argmax(np.where(near, direct, -1)))
                options = far & (direct < direct.flat[a])
                if not options.any():
                    continue
                forced_near.append(a)
                forced_far.append(int(rng.choice(np.flatnonzero(options))))
            chosen = _choose(rng, near, 9, forced_near)
            retained = _choose(rng, far, load-9, forced_far)
            chosen += retained
            break
    if chosen is None:
        return None
    minimum = 1 if obligation in ('partial', 'excess') or load == 0 else 0
    if inactive.sum() < minimum:
        return None
    count = int(rng.integers(minimum, min(7, int(inactive.sum()))+1))
    force = []
    if obligation == 'partial':
        extra = inactive & (_distances(scene, 'undirected') >= 0)
        if not extra.any():
            return None
        force = [int(rng.choice(np.flatnonzero(extra)))]
    distractors = _choose(rng, inactive, count, force)
    return chosen+distractors


def _make_pair(rng, size, nh, ns, obligation, load, palette, untaught_costs=()):
    for attempt in range(256):
        scene = _layout(rng, size, nh, ns, obligation, attempt)
        if scene is None:
            continue
        d = _distances(scene)
        if obligation == 'partial':
            # Below capacity, ordering evidence must affect reachability itself.
            if any(np.array_equal(d >= 0, _distances(scene, mode) >= 0)
                   for mode in ('greedy', 'reverse', 'horizontal', 'nearest')):
                continue
        for _ in range(12):
            cargo = _supply(rng, scene, d, load, obligation)
            if cargo is None:
                break
            port, slot, marker, payload = palette
            a = np.zeros(scene['wall'].shape, dtype=np.int8)
            a[scene['wall']] = 5
            for rr, cc, s, hollow in scene['hollow']+scene['solid']:
                a[rr:rr+s, cc:cc+s] = _tile(s, port, hollow)
            r, c = scene['counter']
            a[r:r+3, c:c+3] = slot
            a[r+1, c+1] = marker
            a.flat[cargo] = payload
            parsed = _parse(a)
            selected = _selection(parsed, d)
            out = _render(parsed, selected)
            if any(_cost_relation(parsed, cost, d) >= 0 for cost in untaught_costs):
                continue
            obligations = ('none', 'greedy', 'reverse', 'horizontal', 'nearest', 'undirected') if obligation == 'partial' else (
                ('broadcast', 'manhattan', 'one_hop') if obligation == 'excess' else ())
            if any(np.array_equal(out, _alternative(parsed, mode)) for mode in obligations):
                continue
            return {'input': a, 'output': out}
    raise RuntimeError('Bounded scene construction exhausted; no ambiguous fallback label was emitted.')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.choice([1, 2, 3, 4, 6, 7, 8, 9], 4, replace=False)))
    varying_palette = bool(rng.integers(2))
    constant_scale = bool(rng.random() < .3)
    base_size = int(rng.integers(3, 6))
    specs = [('partial', 3, 2, int(rng.choice([2, 4]))),
             ('excess', 2, 3, int(rng.integers(10, 17)))]
    for _ in range(nt-2):
        specs.append(('ordinary', int(rng.integers(1, 5)), int(rng.integers(1, 5)),
                      int(rng.integers(0, 18))))
    specs = [specs[int(i)] for i in rng.permutation(nt)]
    specs += [('shift', int(rng.integers(3, 5)), int(rng.integers(3, 5)),
               int(rng.integers(1, 18))) for _ in range(ne)]
    puzzle = {'train': [], 'test': []}
    seen = set()
    untaught_costs = ()
    for index, (obligation, nh, ns, load) in enumerate(specs):
        if index == nt:
            scenes = [_parse(pair['input']) for pair in puzzle['train']]
            untaught_costs = tuple(cost for cost in (0, 2)
                                  if not any(_cost_relation(scene, cost) > 0 for scene in scenes))
        size = base_size if constant_scale else int(rng.integers(3, 6))
        colors = palette if not varying_palette else tuple(map(int,
            rng.choice([1, 2, 3, 4, 6, 7, 8, 9], 4, replace=False)))
        for _ in range(8):
            pair = _make_pair(rng, size, nh, ns, obligation, load, colors, untaught_costs)
            key = (pair['input'].shape, pair['input'].tobytes())
            if key not in seen:
                break
        else:
            raise RuntimeError('Duplicate inputs after bounded retries.')
        seen.add(key)
        puzzle['train' if index < nt else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(pair):
    scene = _parse(pair['input'])
    d = _distances(scene)
    selected = _selection(scene, d)
    cargo = scene['grid'] == scene['payload']
    none = _distances(scene, 'none')
    one = _distances(scene, hop_limit=1)
    two = _distances(scene, hop_limit=2)
    selected_mask = np.zeros(d.shape, bool)
    selected_mask.flat[selected] = True
    out = np.asarray(pair['output'])
    witnesses = {mode: not np.array_equal(out, _alternative(scene, mode))
                 for mode in ('none', 'greedy', 'reverse', 'horizontal', 'nearest', 'broadcast',
                              'undirected', 'no_walls', 'manhattan', 'no_capacity',
                              'column_major', 'one_hop', 'components')}
    retained = cargo & (d >= 0) & ~selected_mask
    _, rooms = _components(scene['floor'])
    return dict(shape=list(d.shape), size=scene['hollow'][0][2],
                hollow=len(scene['hollow']), solid=len(scene['solid']),
                selected=len(selected), reachable=int(np.sum(cargo & (d >= 0))),
                unreachable=int(np.sum(cargo & (d < 0))),
                remote_selected=int(np.sum(selected_mask & (none < 0))),
                two_hop_selected=int(np.sum(selected_mask & (one < 0))),
                three_hop_selected=int(np.sum(selected_mask & (two < 0))),
                capacity_gap=int(d[retained].min()-d[selected_mask].max()) if retained.any() else 0,
                floor_components=len(rooms), max_distance=int(d.max()),
                walls=int(scene['wall'].sum()), palette=[scene[k] for k in ('port', 'slot', 'marker', 'payload')],
                witnesses=witnesses)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle schema.')
    _require(all(isinstance(puzzle[k], list) for k in puzzle), 'Splits must be lists.')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, testing = set(), [], []
    cost_relations = {split: {cost: [] for cost in (0, 2)} for split in ('train', 'test')}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair schema.')
            a, b = _grid(pair['input']), _grid(pair['output'])
            _require(a.shape == b.shape, 'Output must preserve dimensions.')
            key = (a.shape, a.astype(np.int8).tobytes())
            _require(key not in seen, 'Duplicate input.')
            seen.add(key)
            scene = _parse(a)
            selected = _selection(scene, _distances(scene))
            _require(np.array_equal(b, _render(scene, selected)), 'Incorrect transfer or prefix.')
            # Exact conservation of non-marker cargo: marker is a distinct color.
            _require(np.sum(a == scene['payload']) == np.sum(b == scene['payload']), 'Cargo was created or lost.')
            _require(np.array_equal(a == 5, b == 5), 'Wall changed.')
            for r, c, s, _ in scene['hollow']+scene['solid']:
                _require(np.array_equal(a[r:r+s, c:c+s], b[r:r+s, c:c+s]), 'Portal changed.')
            features = pair_features(pair)
            (training if split == 'train' else testing).append(features)
            for cost in (0, 2):
                cost_relations[split][cost].append(_cost_relation(scene, cost))
    _require(any(f['selected'] in (2, 4) for f in training), 'Missing partial row-major evidence.')
    _require(any(f['reachable'] > 9 and f['two_hop_selected'] and f['witnesses']['manhattan']
                 for f in training), 'Missing chain/distance/capacity evidence.')
    _require(any(f['hollow'] > f['solid'] for f in training) and
             any(f['solid'] > f['hollow'] for f in training), 'Demonstrate unmatched ranks in both classes.')
    _require(any(f['unreachable'] for f in training), 'Missing preserved distractors.')
    for mode in training[0]['witnesses']:
        _require(any(f['witnesses'][mode] for f in training), 'Training admits shortcut: '+mode)
    _require(all(f['three_hop_selected'] for f in testing), 'Test must exercise the deeper portal chain.')
    for cost in (0, 2):
        if not any(relation > 0 for relation in cost_relations['train'][cost]):
            _require(all(relation < 0 for relation in cost_relations['test'][cost]),
                     'Untaught portal cost changes or ambiguates a test selection.')
    return None


def _signature(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for key in ('input', 'output'):
                a = np.asarray(pair[key], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed puzzles; no files, saved traces, or retained puzzle corpus."""
    # Numeric fixture: cut a connected line by capacity, preserve reachable
    # excess and a wall-separated cell, and count cells rather than components.
    line = np.zeros((12, 30), dtype=np.int8)
    line[:, 22] = 5
    line[1:4, 1:4] = 3
    line[2, 2] = 4
    line[8:11, 1:4] = _tile(3, 1, True)
    line[8:11, 8:11] = _tile(3, 1, False)
    line[2, 4:18] = 9
    line[5, 26] = 9
    expected = line.copy()
    expected[2, 4:13] = 0
    expected[1:4, 1:4] = 9
    _require(np.array_equal(solve(line), expected), 'Connected-line capacity regression.')
    signatures, shapes, sizes, counts, walls, palettes, loads, components = (set() for _ in range(8))
    max_route = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input'].copy()
                prediction = solve(a)
                _require(np.array_equal(a, pair['input']), 'solve mutated its input.')
                _require(np.array_equal(prediction, pair['output']), 'solve disagreed.')
                f = pair_features(pair)
                shapes.add(tuple(f['shape']))
                sizes.add(f['size'])
                counts.add((f['hollow'], f['solid']))
                walls.add((tuple(a.shape), (a == 5).tobytes()))
                palettes.add(tuple(f['palette']))
                loads.add(f['selected'])
                components.add(f['floor_components'])
                max_route = max(max_route, f['max_distance'])
    _require(len(signatures) == 42 and len(shapes) >= 20 and len(walls) >= 100,
             'Structural variation collapsed.')
    _require(sizes == {3, 4, 5} and len(counts) >= 10 and len(palettes) >= 30,
             'Scale, count or palette variation collapsed.')
    _require(len(loads) >= 8 and len(components) >= 3 and max_route >= 20,
             'Load or topology variation collapsed.')
    for a, b in ((2, 1), (10, 1), (10, 4)):
        first = generate_puzzle(12031, a, b)
        _require(_signature(first) == _signature(generate_puzzle(12031, a, b)), 'Nondeterminism.')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (4.0, 1), (True, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted.')
    broken = generate_puzzle(0, 2, 1)
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupt output accepted.')
    # A strict capacity tie is rejected, including a tie inside one cluster.
    scene = _parse(broken['train'][0]['input'])
    tied = np.full(scene['grid'].shape, -1, dtype=np.int16)
    cargo = np.flatnonzero(scene['grid'] == scene['payload'])
    scene['grid'] = scene['grid'].copy()
    # Use ten arbitrary exterior floor cells and no construction metadata.
    r, c = scene['counter']
    candidates = scene['floor'].copy()
    candidates[r:r+3, c:c+3] = False
    scene['grid'].flat[cargo] = 0
    cells = np.flatnonzero(candidates)[:10]
    scene['grid'].flat[cells] = scene['payload']
    tied.flat[cells] = 3
    try:
        _selection(scene, tied)
    except AssertionError:
        pass
    else:
        raise AssertionError('Capacity tie accepted.')
    return {'seeds': 42, 'pairs': 210, 'distinct_puzzles': len(signatures),
            'shapes': len(shapes), 'wall_layouts': len(walls), 'scales': sorted(sizes),
            'count_combinations': len(counts), 'palettes': len(palettes),
            'selected_loads': sorted(loads), 'floor_component_counts': sorted(components),
            'max_route': max_route, 'independent_solver': False}
