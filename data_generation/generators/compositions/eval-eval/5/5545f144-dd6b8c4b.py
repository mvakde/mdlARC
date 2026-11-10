"""Touring collector: a rigid arrow carries a shared nine-slot cargo counter.

Only NumPy and the standard library are used. No files, seeds, examples, or
generation metadata are consulted by solve(). All saved reports are optional
caller responsibilities; this module writes no files or execution traces.
"""
from collections import deque
import hashlib

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
SLOTS = tuple((r, c) for r in range(-1, 2) for c in range(-1, 2))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Invalid colors")
    return a


def _rotate(p, direction):
    r, c = p
    for _ in range(direction % 4):
        r, c = c, -r
    return r, c


def _body(tip, direction, shaft):
    dr, dc = DIRS[direction]
    center = (tip[0] - (shaft + 1)*dr, tip[1] - (shaft + 1)*dc)
    slots = [(center[0]+r, center[1]+c)
             for r, c in (_rotate(p, direction) for p in SLOTS)]
    nose = [(center[0]+k*dr, center[1]+k*dc) for k in range(2, shaft+2)]
    return center, slots, nose


def _fits(cells, wall):
    h, w = wall.shape
    return all(0 <= r < h and 0 <= c < w and not wall[r, c] for r, c in cells)


def _direction(a, b):
    dr, dc = b[0]-a[0], b[1]-a[1]
    _require((dr == 0) != (dc == 0), "Route must be orthogonal")
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _route_order(tip, beacons):
    remaining, points, directions = set(beacons), [tip], []
    while remaining:
        rays = {}
        for q in sorted(remaining):
            if q[0] == tip[0] or q[1] == tip[1]:
                rays.setdefault(_direction(tip, q), []).append(q)
        _require(len(rays) == 1, "Need exactly one occupied waypoint ray")
        d, candidates = next(iter(rays.items()))
        q = min(candidates, key=lambda p: abs(p[0]-tip[0])+abs(p[1]-tip[1]))
        points.append(q)
        directions.append(d)
        remaining.remove(q)
        tip = q
    return points, directions


def _parse(grid):
    a = _grid(grid)
    colors, counts = np.unique(a, return_counts=True)
    _require(len(colors) == 6, "Input needs six distinct visible color roles")
    bg = int(colors[np.argmax(counts)])
    _require(int(counts.max()) > a.size // 2, "Background must be a strict majority")
    wall_color = int(a[0, 0])
    _require(wall_color != bg and bool(np.all(a[0] == wall_color)) and
             bool(np.all(a[-1] == wall_color)) and bool(np.all(a[:, 0] == wall_color)) and
             bool(np.all(a[:, -1] == wall_color)), "Need a complete wall border")
    candidates = []
    for color, count in zip(colors, counts):
        if count != 8 or int(color) in (bg, wall_color):
            continue
        rr, cc = np.where(a == color)
        r, c = int(rr.min()), int(cc.min())
        if int(rr.max()) != r+2 or int(cc.max()) != c+2:
            continue
        marker = int(a[r+1, c+1])
        if marker == int(color) or int(np.count_nonzero(a == marker)) != 1:
            continue
        if set(zip(rr, cc)) != {(r+i, c+j) for i in range(3) for j in range(3)
                              if (i, j) != (1, 1)}:
            continue
        candidates.append(((r+1, c+1), int(color), marker))
    _require(len(candidates) == 1, "Need one marked 3x3 storage body")
    center, slot, marker = candidates[0]
    extras = set(int(c) for c in colors) - {bg, wall_color, slot, marker}
    _require(len(extras) == 2, "Role colors must differ")
    noses = []
    for d, (dr, dc) in enumerate(DIRS):
        p = (center[0]+2*dr, center[1]+2*dc)
        if 0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1] and int(a[p]) in extras:
            route_color = int(a[p])
            nose = []
            while 0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1] and a[p] == route_color:
                nose.append(p)
                p = (p[0]+dr, p[1]+dc)
            if 1 <= len(nose) <= 2:
                noses.append((nose, d, route_color))
    _require(len(noses) == 1, "Need one projecting nose; other body sides stay clear")
    nose, facing, route_color = noses[0]
    payload = next(iter(extras - {route_color}))
    beacons = set(zip(*np.where(a == route_color))) - set(nose)
    _require(len(beacons) >= 1, "Need at least one waypoint")
    points, directions = _route_order(nose[-1], beacons)
    _require(directions[0] == facing, "First leg must follow the visible nose")
    shaft = len(nose)
    wall = a == wall_color
    footprints = []
    for i, p in enumerate(points):
        _, slots, ns = _body(p, facing if i == 0 else directions[i-1], shaft)
        cells = set(slots+ns)
        _require(_fits(cells, wall), "Every docking footprint must fit off the walls")
        _require(all(a[q] != payload for q in cells), "Cargo may not occupy a docking footprint")
        _require(all(max(abs(q[0]-r), abs(q[1]-c)) > 1
                     for q in points[i+1:] for r, c in cells), "Future waypoints must be separated")
        footprints.append(cells)
    return dict(bg=bg, wall_color=wall_color, slot=slot, marker=marker,
                route_color=route_color, payload=payload, shaft=shaft,
                points=points, directions=directions, facing=facing,
                wall=wall, footprints=footprints)


def _distances(wall, starts):
    distance = np.full(wall.shape, -1, dtype=np.int16)
    queue = deque()
    for p in starts:
        distance[p] = 0
        queue.append(p)
    h, w = wall.shape
    while queue:
        r, c = queue.popleft()
        nd = int(distance[r, c])+1
        for dr, dc in DIRS:
            q = (r+dr, c+dc)
            if 0 <= q[0] < h and 0 <= q[1] < w and not wall[q] and distance[q] < 0:
                distance[q] = nd
                queue.append(q)
    return distance


def _manhattan(shape, slots):
    rr, cc = np.indices(shape)
    return np.minimum.reduce([abs(rr-r)+abs(cc-c) for r, c in slots])


def _select(candidates, distance, capacity, strict=True):
    candidates = sorted(candidates, key=lambda q: (int(distance[q]), q))
    k = min(capacity, len(candidates))
    if strict and 0 < k < len(candidates):
        _require(distance[candidates[k-1]] < distance[candidates[k]],
                 'Capacity frontier needs a strict path-distance gap')
    return candidates[:k], candidates[k:]


def _run(a, info, *, visits=None, ignore_walls=False, collect=True, strict=True):
    """Input-derived operation; diagnostic switches are never used by solve."""
    world = a.copy()
    for q in info['footprints'][0] | set(info['points'][1:]):
        world[q] = info['bg']
    wall = np.zeros(a.shape, dtype=bool) if ignore_walls else info['wall']
    events, cargo = [], 0
    if visits is None:
        visits = range(1, len(info['points']))
    for i in visits:
        _, slots, _ = _body(info['points'][i], info['facing'] if i == 0 else info['directions'][i-1], info['shaft'])
        distance = _distances(wall, slots)
        candidates = [tuple(map(int, q)) for q in np.argwhere((world == info['payload']) & (distance >= 0))]
        capacity = 9-cargo
        selected, retained = _select(candidates, distance, capacity if collect else 0, strict)
        k = len(selected)
        metric = _manhattan(a.shape, slots)
        reversal = bool(selected and retained and k < len(candidates) and
                        max(int(metric[q]) for q in selected) > min(int(metric[q]) for q in retained))
        for q in selected:
            world[q] = info['bg']
        cargo += k
        events.append(dict(index=i, selected=selected, retained=retained, capacity=capacity,
                           reachable=len(candidates), detour_reversal=reversal,
                           max_distance=max((int(distance[q]) for q in candidates), default=0)))
    _, slots, nose = _body(info['points'][-1], info['directions'][-1], info['shaft'])
    for j, q in enumerate(slots):
        world[q] = info['payload'] if j < cargo else info['marker'] if j == 4 else info['slot']
    for q in nose:
        world[q] = info['route_color']
    return world.astype(np.int8), events, cargo


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, route, collection, rotation and slot from this input."""
    a = _grid(input_grid)
    return _run(a, _parse(a))[0]


def _components(wall):
    labels = np.full(wall.shape, -1, dtype=np.int16)
    groups = []
    for p in map(tuple, np.argwhere(~wall)):
        if labels[p] >= 0:
            continue
        group, queue = [], deque([p])
        labels[p] = len(groups)
        while queue:
            q = queue.popleft()
            group.append(q)
            for dr, dc in DIRS:
                t = (q[0]+dr, q[1]+dc)
                if 0 <= t[0] < wall.shape[0] and 0 <= t[1] < wall.shape[1] and not wall[t] and labels[t] < 0:
                    labels[t] = len(groups)
                    queue.append(t)
        groups.append(group)
    return labels, groups


def _geometry(rng, large):
    h, w = (int(x) for x in rng.integers(23 if large else 20, 31, 2))
    wall = np.zeros((h, w), dtype=bool)
    wall[[0, -1], :] = True
    wall[:, [0, -1]] = True
    rooms = [(1, 1, h-2, w-2)]
    target = int(rng.integers(3, 6))
    while len(rooms) < target:
        choices = []
        for j, (r0, c0, r1, c1) in enumerate(rooms):
            if r1-r0+1 >= 15:
                choices.append((j, 0))
            if c1-c0+1 >= 15:
                choices.append((j, 1))
        if not choices:
            break
        j, axis = choices[int(rng.integers(len(choices)))]
        r0, c0, r1, c1 = rooms.pop(j)
        if axis == 0:
            cut = int(rng.integers(r0+6, r1-5))
            wall[cut, c0:c1+1] = True
            rooms.extend([(r0, c0, cut-1, c1), (cut+1, c0, r1, c1)])
        else:
            cut = int(rng.integers(c0+6, c1-5))
            wall[r0:r1+1, cut] = True
            rooms.extend([(r0, c0, r1, cut-1), (r0, cut+1, r1, c1)])
    # Partial baffles make path distance differ from geometric closeness.
    for r0, c0, r1, c1 in rooms:
        for _ in range(int(rng.integers(1, 3))):
            if rng.random() < .5:
                c = int(rng.integers(c0+2, c1-1))
                gap = int(rng.integers(2, max(3, (r1-r0+1)//2)))
                if rng.random() < .5:
                    wall[r0:r1-gap+1, c] = True
                else:
                    wall[r0+gap:r1+1, c] = True
            else:
                r = int(rng.integers(r0+2, r1-1))
                gap = int(rng.integers(2, max(3, (c1-c0+1)//2)))
                if rng.random() < .5:
                    wall[r, c0:c1-gap+1] = True
                else:
                    wall[r, c0+gap:c1+1] = True
    return wall


def _grow_route(rng, wall, shaft, target, labels, groups):
    valid = {}
    for d in range(4):
        valid[d] = {tuple(map(int, p)) for p in np.argwhere(~wall)
                    if _fits(_body(tuple(p), d, shaft)[1]+_body(tuple(p), d, shaft)[2], wall)}
    for _ in range(26):
        initial = int(rng.integers(4))
        if not valid[initial]:
            continue
        start = sorted(valid[initial])[int(rng.integers(len(valid[initial])))]
        points, ds = [start], []
        footprints = [set(sum(_body(start, initial, shaft)[1:], []))]
        while len(ds) < target:
            options = []
            r, c = points[-1]
            for d, (dr, dc) in enumerate(DIRS):
                if not ds and d != initial:
                    continue
                for length in range(3, max(wall.shape)):
                    q = (r+length*dr, c+length*dc)
                    if q not in valid[d]:
                        continue
                    if any(max(abs(q[0]-a), abs(q[1]-b)) <= 1 for cells in footprints for a, b in cells):
                        continue
                    okay = True
                    for p, nxt in zip(points[:-1], points[1:]):
                        if p[0] == q[0] or p[1] == q[1]:
                            if _direction(p, q) != _direction(p, nxt) or abs(q[0]-p[0])+abs(q[1]-p[1]) <= abs(nxt[0]-p[0])+abs(nxt[1]-p[1]):
                                okay = False
                                break
                    if okay:
                        options.append((q, d))
            if not options:
                break
            q, d = options[int(rng.integers(len(options)))]
            points.append(q)
            ds.append(d)
            footprints.append(set(sum(_body(q, d, shaft)[1:], [])))
        visited = list(dict.fromkeys(int(labels[q]) for q in points[1:]))
        if len(ds) == target and len(visited) >= 2 and len(visited) < len(groups):
            return points, ds, footprints, visited
    return None


def _pick(rng, cells, count, forced=()):
    cells = sorted(set(cells)-set(forced))
    if len(cells)+len(forced) < count:
        return None
    indices = rng.choice(len(cells), count-len(forced), replace=False)
    return list(forced)+[cells[int(i)] for i in indices]


def _frontier(rng, cells, distance, slots, count, excess, require_detour):
    metric = _manhattan(distance.shape, slots)
    levels = sorted({int(distance[q]) for q in cells if distance[q] >= 0})
    for index in rng.permutation(len(levels)):
        threshold = levels[int(index)]
        near = [q for q in cells if 0 <= distance[q] <= threshold]
        far = [q for q in cells if distance[q] > threshold]
        if len(near) < count or len(far) < excess:
            continue
        forced_near, forced_far = [], []
        if require_detour:
            low = min(int(metric[q]) for q in far)
            witnesses = [q for q in near if metric[q] > low]
            if not witnesses:
                continue
            qa = witnesses[int(rng.integers(len(witnesses)))]
            others = [q for q in far if metric[q] < metric[qa]]
            qb = others[int(rng.integers(len(others)))]
            forced_near, forced_far = [qa], [qb]
        return _pick(rng, near, count, forced_near)+_pick(rng, far, excess, forced_far)
    return None


def _make_pair(rng, mode, target, large, teach_rotation):
    # All attempts stay in memory. A failure is explicit; no unbounded retry.
    for _ in range(100):
        wall = _geometry(rng, large)
        labels, groups = _components(wall)
        shaft = int(rng.integers(1, 3))
        route = _grow_route(rng, wall, shaft, target, labels, groups)
        if route is None:
            continue
        points, ds, footprints, visited = route
        if teach_rotation and (ds[-1] == 0 or ds[-1] == ds[0]):
            continue
        if teach_rotation and int(labels[points[0]]) in visited:
            continue
        reserved = set().union(*footprints)
        for r, c in footprints[0]:
            reserved.update((r+dr, c+dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1))
        pools = {j: [q for q in group if q not in reserved] for j, group in enumerate(groups)}
        if teach_rotation and not pools[int(labels[points[0]])]:
            continue
        total = int(rng.choice([2, 4])) if mode == 'partial' else 9 if mode == 'saturated' else int(rng.integers(2, 10))
        stop_count = int(rng.integers(2, min(len(visited), total)+1))
        cuts = sorted(int(x) for x in rng.choice(np.arange(1, total), stop_count-1, replace=False))
        allocations = np.diff([0]+cuts+[total]).tolist()
        supply, success = [], True
        for j, amount in enumerate(allocations):
            component = visited[j]
            if mode == 'saturated' and j == stop_count-1:
                i = next(i for i in range(1, len(points)) if labels[points[i]] == component)
                _, slots, _ = _body(points[i], ds[i-1], shaft)
                distance = _distances(wall, slots)
                chosen = _frontier(rng, pools[component], distance, slots, amount,
                                   int(rng.integers(1, 8)), True)
            else:
                chosen = _pick(rng, pools[component], amount)
            if chosen is None:
                success = False
                break
            supply.extend(chosen)
        if not success:
            continue
        # After saturation, additional visited rooms may still hold cargo.
        if mode == 'saturated':
            for component in visited[stop_count:]:
                supply.extend(_pick(rng, pools[component], min(len(pools[component]), int(rng.integers(0, 6)))))
        unseen = [j for j in pools if j not in visited and pools[j]]
        if not unseen:
            continue
        # At least one untouched chamber visibly distinguishes global counting.
        distractors = []
        for j in unseen:
            distractors.extend(_pick(rng, pools[j], min(len(pools[j]), int(rng.integers(1, 7)))))
        bg, wc, slot, marker, ink, payload = [int(x) for x in rng.choice(10, 6, replace=False)]
        a = np.full(wall.shape, bg, dtype=np.int8)
        a[wall] = wc
        for q in supply+distractors:
            a[q] = payload
        center, slots, nose = _body(points[0], ds[0], shaft)
        for q in slots:
            a[q] = slot
        a[center] = marker
        for q in nose+points[1:]:
            a[q] = ink
        if np.count_nonzero(a == bg) <= a.size//2:
            continue
        info = _parse(a)
        out, events, cargo = _run(a, info)
        if mode == 'saturated' and not any(e['detour_reversal'] for e in events):
            continue
        return {'input': a, 'output': out}
    raise RuntimeError('Bounded touring-collector construction exhausted')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    # Two complementary evidence obligations survive even at the minimum count.
    requests = [('partial', int(rng.integers(2, 4)), False, True),
                ('saturated', int(rng.integers(2, 4)), False, False)]
    requests += [(str(rng.choice(['random', 'saturated'])), int(rng.integers(2, 4)), False, False)
                 for _ in range(num_train-2)]
    requests = [requests[int(i)] for i in rng.permutation(num_train)]
    test_requests = [(str(rng.choice(['random', 'saturated', 'partial'])), int(rng.integers(4, 7)), True, False)
                     for _ in range(num_test)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, specs in [('train', requests), ('test', test_requests)]:
        for spec in specs:
            for _ in range(8):
                pair = _make_pair(rng, *spec)
                sig = (pair['input'].shape, pair['input'].tobytes())
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Distinct-input retry bound exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(pair):
    a, b = _grid(pair['input']), _grid(pair['output'])
    info = _parse(a)
    expected, events, cargo = _run(a, info)
    _require(np.array_equal(b, expected), 'Output disagrees with touring collection')
    collected = {q for e in events for q in e['selected']}
    _require(len(collected) == cargo <= 9, 'Cargo must equal distinct cells erased')
    for q in collected:
        _require(a[q] == info['payload'] and b[q] == info['bg'], 'Collection mass mismatch')
    _require(bool(np.all(b[info['wall']] == info['wall_color'])), 'Walls must remain unchanged')
    _, slots, _ = _body(info['points'][-1], info['directions'][-1], info['shaft'])
    outside = np.ones(a.shape, dtype=bool)
    for q in slots:
        outside[q] = False
    _require(int(np.count_nonzero((b == info['payload']) & outside))+cargo ==
             int(np.count_nonzero(a == info['payload'])), 'Payload conservation failed')
    labels, groups = _components(info['wall'])
    visited = {int(labels[p]) for p in info['points'][1:]}
    unreachable = sum(a[q] == info['payload'] for j, group in enumerate(groups) if j not in visited for q in group)
    reverse = _run(a, info, visits=range(len(info['points'])-1, 0, -1), strict=False)[0]
    no_walls = _run(a, info, ignore_walls=True, strict=False)[0]
    final_only = _run(a, info, visits=[len(info['points'])-1], strict=False)[0]
    empty = _run(a, info, collect=False)[0]
    start_pickup = _run(a, info, visits=range(len(info['points'])), strict=False)[0]
    # Compare transported local serialization to a displayed row-major prefix.
    local = [tuple(q) for q in slots[:cargo]]
    displayed = sorted(slots)[:cargo]
    return dict(shape=list(a.shape), waypoints=len(info['directions']), shaft=info['shaft'],
                initial=info['facing'], final=info['directions'][-1], cargo=cargo,
                components=len(groups), visited_components=len(visited), unreachable=int(unreachable),
                per_stop=[len(e['selected']) for e in events],
                distance_max=max(e['max_distance'] for e in events),
                detour=any(e['detour_reversal'] for e in events),
                excess=any(e['capacity'] > 0 and e['reachable'] > e['capacity'] for e in events),
                prefix_rotates=set(local) != set(displayed),
                reverse_changes=not np.array_equal(reverse, b),
                wall_blind_changes=not np.array_equal(no_walls, b),
                final_only_changes=not np.array_equal(final_only, b),
                no_collection_changes=not np.array_equal(empty, b),
                start_pickup_changes=not np.array_equal(start_pickup, b),
                lengths=[abs(q[0]-p[0])+abs(q[1]-p[1]) for p, q in zip(info['points'], info['points'][1:])],
                turns=[(b-a) % 4 for a, b in zip(info['directions'], info['directions'][1:]) if a != b],
                palette=[info[k] for k in ('bg', 'wall_color', 'slot', 'marker', 'route_color', 'payload')])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'Invalid counts')
    seen, stats = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair schema')
            x, y = _grid(pair['input']), _grid(pair['output'])
            _require(x.shape == y.shape, 'The full canvas must be preserved')
            sig = (x.shape, x.astype(np.int8).tobytes())
            _require(sig not in seen, 'Duplicate input')
            seen.add(sig)
            f = pair_features(pair)
            _require(f['visited_components'] >= 2 and f['unreachable'] > 0, 'Need selective multi-room collection')
            _require(f['final_only_changes'] and f['no_collection_changes'], 'Both parents must affect each answer')
            stats[split].append(f)
    train = stats['train']
    _require(any(f['cargo'] in (2, 4) and f['prefix_rotates'] and f['initial'] != f['final'] for f in train),
             'Training must teach partial body-relative filling and rotation')
    _require(any(f['cargo'] == 9 and f['excess'] and f['detour'] and f['reverse_changes'] for f in train),
             'Training must teach shared capacity, order, and path-distance selection')
    _require(any(f['wall_blind_changes'] for f in train), 'Training must distinguish wall-blind counting')
    _require(any(f['start_pickup_changes'] and f['cargo'] in (2, 4) for f in train),
             'Training must show that pickup begins at the first waypoint')
    _require(max(f['waypoints'] for f in train) < min(f['waypoints'] for f in stats['test']),
             'Generated tests must extend the demonstrated journey length')
    return None


def _digest(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """42 streamed puzzles, no files/traces, at most two puzzles retained."""
    features, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_digest(puzzle))
        for split in ('train', 'test'):
            features.extend(dict(pair_features(p), split=split) for p in puzzle[split])
    _require(len(signatures) == 42, 'Cross-seed determinism/diversity failure')
    _require(_digest(generate_puzzle(717)) == _digest(generate_puzzle(717)), 'Nondeterministic puzzle')
    _require(len({tuple(f['shape']) for f in features}) >= 35, 'Insufficient canvas variation')
    _require({f['shaft'] for f in features} == {1, 2}, 'Both visible mover geometries needed')
    _require(len({f['cargo'] for f in features}) >= 7, 'Insufficient loads')
    _require(len({f['components'] for f in features}) >= 3, 'Insufficient room topology')
    _require(len({f['distance_max'] for f in features}) >= 12, 'Insufficient path geometry')
    for split in ('train', 'test'):
        rows = [f for f in features if f['split'] == split]
        _require({f['initial'] for f in rows} == set(range(4)), 'Initial-direction coverage')
        _require({f['final'] for f in rows} == set(range(4)), 'Final-direction coverage')
        _require(sum(f['detour'] for f in rows) >= 5, 'Missing nongeometric path selections')
    for role in range(6):
        _require({f['palette'][role] for f in features} == set(range(10)), 'Color-role coverage')
    for a, b in [(2, 1), (10, 4), (7, 3)]:
        generate_puzzle(42+a*4+b, a, b)
    for a, b in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (3.0, 1), (True, 1), (4, None)]:
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted')
    # Mutations must be detected, including a plausible empty moving collector.
    broken = generate_puzzle(18, 2, 1)
    p = broken['test'][0]
    original = p['input'].copy()
    expected = solve(original)
    for turns in range(4):
        _require(np.array_equal(solve(np.rot90(original, turns)), np.rot90(expected, turns)),
                 'Body-frame rule must commute with scene rotation')
    permutation = np.random.default_rng(81).permutation(10)
    _require(np.array_equal(solve(permutation[original]), permutation[expected]), 'Color-role equivariance')
    _require(np.array_equal(original, p['input']), 'Solver mutated its input')
    p['output'] = _run(p['input'], _parse(p['input']), collect=False)[0]
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('Missing collection was accepted')
    try:
        _route_order((4, 4), {(4, 8), (8, 4)})
    except AssertionError:
        pass
    else:
        raise AssertionError('Ambiguous route accepted')
    # Unit check: an equidistant excess frontier cannot use iteration order.
    wall = np.zeros((9, 9), dtype=bool)
    d = _distances(wall, [(4, 4)])
    try:
        _select([(3, 4), (4, 3), (4, 5)], d, 2)
    except AssertionError:
        pass
    else:
        raise AssertionError('Capacity tie was accepted')
    return {'seeds': 42, 'pairs': len(features), 'distinct_puzzles': len(signatures),
            'shapes': len({tuple(f['shape']) for f in features}),
            'loads': sorted({f['cargo'] for f in features}),
            'detour_pairs': sum(f['detour'] for f in features)}
