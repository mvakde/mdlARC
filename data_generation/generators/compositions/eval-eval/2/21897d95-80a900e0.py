"""Two-prong color couriers: original-component addressing by cap endpoints.

Only marker color 9 has a fixed role. All other colors are literal fills/payloads.
The decoder and predictor require only the input pixels. Construction uses
connected, weighted-growth territories and is deliberately separate from solve.
"""
from __future__ import annotations

import heapq
import numbers
import numpy as np

INK = 9
CARDINAL = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIAGONAL = ((-1, -1), (-1, 1), (1, 1), (1, -1))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _inside(point, shape):
    return 0 <= point[0] < shape[0] and 0 <= point[1] < shape[1]


def _add(point, direction, k=1):
    return point[0] + direction[0] * k, point[1] + direction[1] * k


def _ring(center, radius):
    return {_add(center, (r, c)) for r in range(-radius, radius + 1)
            for c in range(-radius, radius + 1) if abs(r) + abs(c) == radius}


def _cap(center, radius, direction):
    dr, dc = direction
    return [(center[0] + dr * (k + 1), center[1] + dc * (radius - k + 1))
            for k in range(radius + 1)]


def _components(grid):
    ids = np.full(grid.shape, -1, dtype=np.int16)
    colors, cells = [], []
    for start in np.ndindex(grid.shape):
        if ids[start] >= 0:
            continue
        label = len(colors)
        color = int(grid[start])
        colors.append(color)
        points, stack = set(), [start]
        ids[start] = label
        while stack:
            point = stack.pop()
            points.add(point)
            for direction in CARDINAL:
                neighbor = _add(point, direction)
                if (_inside(neighbor, grid.shape) and ids[neighbor] < 0
                        and grid[neighbor] == color):
                    ids[neighbor] = label
                    stack.append(neighbor)
        cells.append(points)
    return ids, colors, cells


def _first_hit(ids, start, direction, source):
    """Begin beyond the cap, skip original host cells, stop at first foreign ID."""
    path = []
    point = _add(start, direction)
    while _inside(point, ids.shape):
        path.append(point)
        if ids[point] != source:
            return int(ids[point]), path
        point = _add(point, direction)
    return None, path


def _read(input_grid):
    grid = np.asarray(input_grid)
    _need(grid.ndim == 2 and np.issubdtype(grid.dtype, np.integer), 'integer 2D input required')
    _need(all(1 <= n <= 30 for n in grid.shape), 'invalid dimensions')
    _need(bool(np.all((0 <= grid) & (grid <= 9))), 'invalid colors')
    h, w = grid.shape
    emitters = []
    # Complete outlines are recognizable without counting foreground components:
    # diamond pixels and cap pixels are diagonally, not orthogonally, connected.
    for radius in (2, 3, 4):
        if min(h, w) < 2 * radius + 1:
            continue
        sh = h - 2 * radius, w - 2 * radius
        mask = np.ones(sh, dtype=bool)
        for dr, dc in _ring((0, 0), radius):
            mask &= grid[radius + dr:radius + dr + sh[0],
                         radius + dc:radius + dc + sh[1]] == INK
        mask &= grid[radius:radius + sh[0], radius:radius + sh[1]] != INK
        for r, c in np.argwhere(mask):
            center = int(r) + radius, int(c) + radius
            host = int(grid[center])
            caps = []
            for direction in DIAGONAL:
                points = _cap(center, radius, direction)
                if not all(_inside(p, grid.shape) for p in points):
                    continue
                if grid[points[0]] != INK or grid[points[-1]] != INK:
                    continue
                tokens = [p for p in points[1:-1] if grid[p] != INK]
                _need(len(tokens) <= 1, 'a cap may have at most one interior token')
                caps.append(dict(direction=direction, points=points, radius=radius,
                                 explicit=bool(tokens), token=tokens[0] if tokens else None,
                                 payload=int(grid[tokens[0]]) if tokens else host))
            _need(bool(caps), 'diamond has no cap')
            emitters.append(dict(center=center, radius=radius, host=host, caps=caps))
    _need(bool(emitters), 'no courier diamonds')
    clean = grid.astype(np.int8, copy=True)
    covered, writes = set(), set()
    for emitter in emitters:
        footprint = _ring(emitter['center'], emitter['radius'])
        footprint.update(p for cap in emitter['caps'] for p in cap['points'])
        _need(not footprint & writes, 'overlapping annotations')
        writes.update(footprint)
        covered.update(p for p in footprint if grid[p] == INK)
        for point in footprint:
            clean[point] = emitter['host']
        emitter['write'] = footprint
    _need(covered == set(map(tuple, np.argwhere(grid == INK))), 'unexplained marker pixels')
    ids, colors, cells = _components(clean)
    deliveries = []
    assignments = {}
    for emitter in emitters:
        source = int(ids[emitter['center']])
        emitter['source'] = source
        _need(all(ids[p] == source for p in emitter['write']), 'annotation crosses a host boundary')
        # An orthogonal fill halo makes restoration a visible geometric fact.
        _need(all(ids[q] == source for p in emitter['write'] for d in CARDINAL
                  if _inside(q := _add(p, d), grid.shape)), 'annotation has no host halo')
        for cap in emitter['caps']:
            hits = [_first_hit(ids, p, cap['direction'], source) for p in cap['points']]
            endpoints = hits[0][0], hits[-1][0]
            target = endpoints[0] if endpoints[0] is not None and endpoints[0] == endpoints[1] else None
            cap.update(source=source, target=target, hits=hits)
            deliveries.append(cap)
            if target is not None:
                payload = cap['payload']
                _need(target not in assignments or assignments[target] == payload,
                      'conflicting successful deliveries')
                assignments[target] = payload
    return dict(clean=clean, ids=ids, colors=colors, cells=cells, emitters=emitters,
                deliveries=deliveries, assignments=assignments)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Paint original recipients simultaneously; do not draw the virtual probes."""
    info = _read(input_grid)
    output = info['clean'].copy()
    for target, payload in info['assignments'].items():
        output[info['ids'] == target] = payload
    return output


def _connected_points(points):
    if not points:
        return False
    unseen = set(points)
    stack = [unseen.pop()]
    while stack:
        point = stack.pop()
        for direction in CARDINAL:
            q = _add(point, direction)
            if q in unseen:
                unseen.remove(q)
                stack.append(q)
    return not unseen


def _grow(rng, radii, extra, compact=False, surround=False):
    """Connected weighted wavefront growth from protected source patches/seeds."""
    n = len(radii)
    lower = max(17, 2 * max(radii) + (15 if surround else 9)) if n == 1 else 25
    upper = 31 if surround else (27 if compact and n == 1 else 31)
    shape = tuple(map(int, rng.integers(lower, upper, size=2)))
    if n > 1 and max(radii) == 4:
        shape = (30, shape[1]) if rng.random() < .5 else (shape[0], 30)
    labels = np.full(shape, -1, dtype=np.int16)
    centers, protected = [], set()
    for i, radius in enumerate(radii):
        margin = radius + 2
        candidates = [(r, c) for r in range(margin + 1, shape[0] - margin - 1)
                      for c in range(margin + 1, shape[1] - margin - 1)]
        if surround:
            candidates = [p for p in candidates if all(abs(p[j] - shape[j]//2) <= 2 for j in (0,1))]
        rng.shuffle(candidates)
        for r, c in candidates:
            patch = labels[r-margin:r+margin+1, c-margin:c+margin+1]
            if np.any(patch >= 0):
                continue
            patch[:] = i
            centers.append((r, c))
            protected.update((rr, cc) for rr in range(r-margin, r+margin+1)
                             for cc in range(c-margin, c+margin+1))
            break
        else:
            return None
    for label in range(n, n + extra):
        choices = np.argwhere(labels < 0)
        if not len(choices):
            return None
        if surround:
            direction = DIAGONAL[label - n]
            choices = np.array([p for p in choices if all(
                p[j] <= 3 if direction[j] < 0 else p[j] >= shape[j]-4 for j in (0,1))])
            if not len(choices):
                return None
        point = tuple(choices[int(rng.integers(len(choices)))])
        labels[point] = label
    distance = np.full(shape, np.inf)
    queue = []
    for r, c in np.argwhere(labels >= 0):
        distance[r, c] = 0.
        queue.append((0., int(r), int(c), int(labels[r, c])))
    heapq.heapify(queue)
    # Undirected edge weights; each assigned pixel has a path to its own seed.
    weights = rng.uniform(.6, 2.4, size=(*shape, 2))
    while queue:
        cost, r, c, label = heapq.heappop(queue)
        if cost != distance[r, c] or labels[r, c] != label:
            continue
        for dr, dc in CARDINAL:
            q = r + dr, c + dc
            if not _inside(q, shape):
                continue
            edge = weights[min(r, q[0]), min(c, q[1]), 0 if dr else 1]
            new_cost = cost + edge
            if new_cost < distance[q]:
                distance[q] = new_cost
                labels[q] = label
                heapq.heappush(queue, (new_cost, *q, label))
    return labels, centers, protected


def _addresses(labels, centers, radii):
    result = []
    for source, (center, radius) in enumerate(zip(centers, radii)):
        for direction in DIAGONAL:
            points = _cap(center, radius, direction)
            hits = [_first_hit(labels, p, direction, source) for p in points]
            endpoints = hits[0][0], hits[-1][0]
            target = endpoints[0] if endpoints[0] is not None and endpoints[0] == endpoints[1] else None
            result.append(dict(source=source, center=center, radius=radius, direction=direction,
                               points=points, hits=hits, target=target))
    return result


def _middle_island(rng, labels, centers, radii, protected):
    """Carve a small recipient on an interior lane, preserving all endpoint hits."""
    addresses = _addresses(labels, centers, radii)
    if any(a['target'] is not None and any(h[0] != a['target'] for h in a['hits'][1:-1])
           for a in addresses):
        return labels
    endpoint_paths = {p for a in addresses for h in (a['hits'][0], a['hits'][-1]) for p in h[1]}
    candidates = []
    for a in addresses:
        if a['target'] is None:
            continue
        for hit, path in a['hits'][1:-1]:
            if hit is not None:
                candidates.extend(path)
    rng.shuffle(candidates)
    for point in candidates:
        if point in protected or point in endpoint_paths:
            continue
        old = int(labels[point])
        neighbors = [_add(point, d) for d in CARDINAL]
        rng.shuffle(neighbors)
        for neighbor in neighbors:
            patch = {point, neighbor}
            if (not _inside(neighbor, labels.shape) or labels[neighbor] != old
                    or patch & (protected | endpoint_paths)):
                continue
            remainder = set(map(tuple, np.argwhere(labels == old))) - patch
            if not _connected_points(remainder):
                continue
            result = labels.copy()
            result[point] = result[neighbor] = int(labels.max()) + 1
            after = _addresses(result, centers, radii)
            if any(a['target'] is not None and any(h[0] != a['target'] for h in a['hits'][1:-1])
                   for a in after):
                return result
    return None


def _adjacency(labels):
    n = int(labels.max()) + 1
    neighbors = [set() for _ in range(n)]
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:])):
        for x, y in zip(a[a != b], b[a != b]):
            neighbors[int(x)].add(int(y))
            neighbors[int(y)].add(int(x))
    return neighbors


def _palette(rng, labels, source_count, repeat_pair=None):
    n = int(labels.max()) + 1
    neighbors = _adjacency(labels)
    pool = list(map(int, rng.permutation(9)))
    # One color appears only as a literal payload if an explicit witness needs it.
    singleton = pool.pop()
    colors = [-1] * n
    if repeat_pair:
        a, b = repeat_pair
        if b in neighbors[a]:
            return None
        colors[a] = colors[b] = pool[0]
    for label in list(range(source_count)) + list(map(int, rng.permutation(np.arange(source_count, n)))):
        if colors[label] >= 0:
            continue
        forbidden = {colors[q] for q in neighbors[label]}
        if label < source_count:
            forbidden.update(colors[:source_count])
        choices = [c for c in pool if c not in forbidden]
        unused = [c for c in choices if c not in colors]
        if not choices:
            return None
        colors[label] = int(rng.choice(unused or choices))
    return np.asarray(colors, dtype=np.int8), singleton


def _features(info):
    deliveries = info['deliveries']
    assignments, colors = info['assignments'], info['colors']
    final = [assignments.get(i, c) for i, c in enumerate(colors)]
    successes = [a for a in deliveries if a['target'] is not None]
    split = any(a['hits'][0][0] is not None and a['hits'][-1][0] is not None
                and a['hits'][0][0] != a['hits'][-1][0]
                and any(final[h[0]] != a['payload'] for h in (a['hits'][0], a['hits'][-1]))
                for a in deliveries)
    # The complete-cap consensus shortcut would suppress these actual successes.
    interior = any(any(hit[0] != a['target'] for hit in a['hits'][1:-1])
                   and a['payload'] != colors[a['target']] for a in successes)
    chain = any(not a['explicit'] and not b['explicit'] and a['target'] == b['source']
                and colors[a['source']] != colors[b['source']]
                for a in successes for b in successes)
    locality = any(colors[a] == colors[b] and final[a] != final[b]
                   for a in range(len(colors)) for b in range(a))
    return dict(emitters=len(info['emitters']), radii=tuple(e['radius'] for e in info['emitters']),
                territories=len(colors), caps=len(deliveries), accepted=len(successes),
                failed=len(deliveries)-len(successes), split=split, interior=interior,
                chain=chain, locality=locality,
                singleton=any(a['explicit'] and a['payload'] not in colors for a in successes),
                explicit=sum(a['explicit'] for a in deliveries),
                directions={a['direction'] for a in deliveries},
                success_directions={a['direction'] for a in successes},
                lengths=[len(hit[1]) for a in deliveries for hit in (a['hits'][0], a['hits'][-1])],
                nonrectangular=sum(len(s) != (max(r for r,c in s)-min(r for r,c in s)+1)
                                     * (max(c for r,c in s)-min(c for r,c in s)+1)
                                     for s in info['cells']),
                multiple_incoming=len(successes) > len(assignments))


def _whole_cap_shortcut(info):
    """Wrong rule used solely to check that demonstrations expose endpoint use."""
    output = info['clean'].copy()
    assignments = {}
    for cap in info['deliveries']:
        hits = [hit[0] for hit in cap['hits']]
        if hits[0] is not None and all(t == hits[0] for t in hits):
            assignments[hits[0]] = cap['payload']
    for target, payload in assignments.items():
        output[info['ids'] == target] = payload
    return output


def _cascading_shortcut(info):
    """Wrong rule: emitters read their host fills after earlier deliveries."""
    output = info['clean'].copy()
    for emitter in info['emitters']:
        for cap in emitter['caps']:
            if cap['target'] is not None:
                payload = cap['payload'] if cap['explicit'] else int(output[emitter['center']])
                output[info['ids'] == cap['target']] = payload
    return output


def _build(rng, max_radius, mode, needed_directions=frozenset()):
    """Return a candidate or None. Teaching requests constrain evidence, not rule."""
    if mode == 'cycle':
        n, radii, extra = 2, [max_radius] * 2, int(rng.integers(1, 4))
    elif mode == 'literal':
        n, radii, extra = 1, [max_radius], 4
    else:
        n = int(rng.choice([2, 3], p=[.55, .45])) if mode == 'test' else int(
            rng.choice([1, 2, 3], p=[.4, .4, .2]))
        radii = list(map(int, rng.integers(2, max_radius + 1, size=n)))
        extra = int(rng.integers(2, 6))
    grown = _grow(rng, radii, extra, compact=mode == 'literal', surround=mode == 'literal')
    if grown is None:
        return None
    labels, centers, protected = grown
    if mode in ('cycle', 'literal') or rng.random() < .55:
        labels = _middle_island(rng, labels, centers, radii, protected)
        if labels is None:
            return None
    addresses = _addresses(labels, centers, radii)
    forced = set()
    if mode == 'cycle':
        forward = [i for i,a in enumerate(addresses) if a['source'] == 0 and a['target'] == 1]
        backward = [i for i,a in enumerate(addresses) if a['source'] == 1 and a['target'] == 0]
        if not forward or not backward:
            return None
        forced.update((int(rng.choice(forward)), int(rng.choice(backward))))
        # Keep the true endpoint-only witness active and an observable split.
        interior = [i for i,a in enumerate(addresses) if a['target'] is not None
                    and any(h[0] != a['target'] for h in a['hits'][1:-1])]
        split = [i for i,a in enumerate(addresses) if a['hits'][0][0] is not None
                 and a['hits'][-1][0] is not None and a['hits'][0][0] != a['hits'][-1][0]]
        if not interior or not split:
            return None
        forced.update((int(rng.choice(interior)), int(rng.choice(split))))
    if mode == 'literal':
        targets = {a['target'] for a in addresses if a['target'] is not None}
        actual_directions = {a['direction'] for a in addresses if a['target'] is not None}
        if not needed_directions <= actual_directions:
            return None
        adjacent = _adjacency(labels)
        repeat_candidates = [(a,b) for a in targets for b in range(n, int(labels.max())+1)
                             if b not in targets and b != a and b not in adjacent[a]]
        if not repeat_candidates:
            return None
        repeat_pair = repeat_candidates[int(rng.integers(len(repeat_candidates)))]
    else:
        repeat_pair = None
        if rng.random() < .55:
            adjacent = _adjacency(labels)
            choices = [(a,b) for a in range(n, len(adjacent)) for b in range(a)
                       if b not in adjacent[a] and not (a < n and b < n)]
            if choices:
                repeat_pair = choices[int(rng.integers(len(choices)))]
    palette = _palette(rng, labels, n, repeat_pair)
    if palette is None:
        return None
    colors, singleton = palette
    selected, assignments = [], {}
    order = sorted(forced) + [int(i) for i in rng.permutation(len(addresses)) if i not in forced]
    for index in order:
        a = dict(addresses[index])
        if mode == 'literal':
            include, explicit, payload = True, True, singleton
        else:
            include = index in forced or rng.random() < (.68 if mode == 'cycle' else .63)
            explicit = mode != 'cycle' and rng.random() < .45
            payload = int(rng.integers(9)) if explicit else int(colors[a['source']])
        if not include:
            continue
        target = a['target']
        if target is not None and target in assignments and assignments[target] != payload:
            if index in forced:
                return None
            if mode == 'cycle':
                continue
            explicit, payload = True, assignments[target]
        a.update(explicit=explicit, payload=payload)
        selected.append(a)
        if target is not None:
            assignments[target] = payload
    if {a['source'] for a in selected} != set(range(n)) or not assignments:
        return None
    clean = colors[labels]
    inp = clean.copy()
    for center, radius in zip(centers, radii):
        for point in _ring(center, radius):
            inp[point] = INK
    for a in selected:
        for point in a['points']:
            inp[point] = INK
        if a['explicit']:
            point = a['points'][int(rng.integers(1, len(a['points'])-1))]
            inp[point] = a['payload']
    expected = clean.copy()
    for target, payload in assignments.items():
        expected[labels == target] = payload
    try:
        info = _read(inp)
    except AssertionError:
        return None
    # Check recovered restoration and original component partition against the
    # construction. This comparison catches unintended equal-color mergers.
    if not np.array_equal(info['clean'], clean) or len(info['cells']) != len(colors):
        return None
    predicted = solve(inp)
    if not np.array_equal(expected, predicted):
        return None
    features = _features(info)
    if mode == 'cycle' and not all(features[k] for k in ('chain', 'interior', 'split')):
        return None
    if mode == 'literal' and not all(features[k] for k in ('locality', 'singleton', 'interior')):
        return None
    if mode in ('cycle', 'literal') and np.array_equal(_whole_cap_shortcut(info), expected):
        return None
    if mode == 'cycle' and np.array_equal(_cascading_shortcut(info), expected):
        return None
    if mode == 'test':
        successful = [a for a in info['deliveries'] if a['target'] is not None
                      and a['payload'] != info['colors'][a['target']]]
        if (not any(a['explicit'] and a['payload'] != info['colors'][a['source']] for a in successful)
                or not any(not a['explicit'] for a in successful)
                or len({a['target'] for a in successful}) < 2):
            return None
    # A counterpart to drawing rays: every accepted region contains multiple
    # cells off the actual probe paths, and at least one delivery changes a fill.
    if not any(a['payload'] != info['colors'][a['target']]
               for a in info['deliveries'] if a['target'] is not None):
        return None
    if any(len(info['cells'][t]) < 3 for t,p in info['assignments'].items() if p != info['colors'][t]):
        return None
    return dict(input=inp, output=expected), info


def _key(grid):
    array = np.asarray(grid, dtype=np.int8)
    return array.shape, array.tobytes()


def _integer(value):
    return isinstance(value, numbers.Integral) and not isinstance(value, (bool, np.bool_))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if (not _integer(num_train) or not _integer(num_test) or not 2 <= num_train <= 10
            or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError('require integer counts 2..10 / 1..4 with num_train > num_test')
    rng = np.random.default_rng(int(seed))
    max_radius = int(rng.integers(2, 5))
    puzzle, seen = {'train': [], 'test': []}, set()
    success_directions = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            mode = 'cycle' if split == 'train' and index == 0 else (
                'literal' if split == 'train' and index == 1 else split)
            needed = set(DIAGONAL) - success_directions if mode == 'literal' else set()
            for _ in range(600):
                result = _build(rng, max_radius, mode, needed)
                if result is None:
                    continue
                pair, info = result
                key = _key(pair['input'])
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                if split == 'train':
                    success_directions.update(_features(info)['success_directions'])
                break
            else:
                raise RuntimeError('bounded scene construction exhausted for ' + mode)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _closed_form_hits(ids, point, direction, source):
    """Second address calculation: enumerate the diagonal in one array operation."""
    steps = np.arange(1, 31)
    rows = point[0] + direction[0] * steps
    cols = point[1] + direction[1] * steps
    mask = (rows >= 0) & (rows < ids.shape[0]) & (cols >= 0) & (cols < ids.shape[1])
    rows, cols = rows[mask], cols[mask]
    foreign = np.flatnonzero(ids[rows, cols] != source)
    return int(ids[rows[foreign[0]], cols[foreign[0]]]) if len(foreign) else None


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'invalid counts')
    seen, teaching, radii = set(), [], {'train': set(), 'test': set()}
    endpoint_counterexample = cascade_counterexample = False
    taught_size_by_direction = {}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            for grid in (inp, out):
                _need(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape), 'invalid grid shape')
                _need(np.issubdtype(grid.dtype, np.integer), 'noninteger grid')
                _need(bool(np.all((0 <= grid) & (grid <= 9))), 'invalid color')
            _need(inp.shape == out.shape, 'output dimensions changed')
            _need(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            info = _read(inp)
            _need(not np.any(out == INK), 'annotation ink survived')
            predicted = info['clean'].copy()
            independent_assignments = {}
            for cap in info['deliveries']:
                hits = [_closed_form_hits(info['ids'], p, cap['direction'], cap['source'])
                        for p in (cap['points'][0], cap['points'][-1])]
                target = hits[0] if hits[0] is not None and hits[0] == hits[1] else None
                _need(target == cap['target'], 'endpoint address implementations disagree')
                if target is not None:
                    _need(target not in independent_assignments
                          or independent_assignments[target] == cap['payload'], 'conflict')
                    independent_assignments[target] = cap['payload']
            for target, payload in independent_assignments.items():
                predicted[info['ids'] == target] = payload
            _need(np.array_equal(out, predicted), 'wrong whole-territory result')
            _need(np.array_equal(out, solve(inp)), 'solve disagreement')
            changed_targets = [t for t,p in independent_assignments.items() if p != info['colors'][t]]
            _need(bool(changed_targets), 'no actual territory transfer')
            for label, color in enumerate(info['colors']):
                expected_color = independent_assignments.get(label, color)
                _need(bool(np.all(out[info['ids'] == label] == expected_color)), 'component partially painted')
            # Every changed target extends beyond the endpoint contact pixels.
            _need(all(len(info['cells'][t]) >= 3 for t in changed_targets), 'unhelpful tiny-only target')
            features = _features(info)
            radii[split].update(features['radii'])
            if split == 'train':
                teaching.append(features)
                endpoint_counterexample |= not np.array_equal(_whole_cap_shortcut(info), out)
                cascade_counterexample |= not np.array_equal(_cascading_shortcut(info), out)
                for cap in info['deliveries']:
                    if cap['target'] is not None:
                        d = cap['direction']
                        taught_size_by_direction[d] = max(taught_size_by_direction.get(d,0), cap['radius'])
            else:
                successes = [c for c in info['deliveries'] if c['target'] is not None
                             and c['payload'] != info['colors'][c['target']]]
                _need(features['emitters'] >= 2 and len({c['target'] for c in successes}) >= 2,
                      'test lacks multi-recipient recombination')
                _need(any(not c['explicit'] for c in successes)
                      and any(c['explicit'] and c['payload'] != info['colors'][c['source']] for c in successes),
                      'test does not combine both payload modes meaningfully')
    _need(any(f['chain'] for f in teaching), 'missing original-color dependency witness')
    _need(any(f['locality'] and f['singleton'] for f in teaching), 'missing literal/component witness')
    _need(any(f['split'] for f in teaching), 'missing paired-agreement witness')
    _need(any(f['interior'] for f in teaching), 'missing interior-lane counterexample')
    _need(endpoint_counterexample and cascade_counterexample, 'training admits a parent-omitting shortcut')
    _need(set.union(*(f['success_directions'] for f in teaching)) == set(DIAGONAL), 'directions not taught')
    _need(max(radii['test']) <= max(radii['train']), 'test size of cap is untaught')
    _need(all(taught_size_by_direction[d] >= max(radii['test']) for d in DIAGONAL),
          'a direction lacks a successful large-cap demonstration')


def self_test():
    """Bounded 42-seed geometry/palette checks and deliberate label damage."""
    puzzles, shapes, layouts, fields = [], set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(tuple(_key(p['input']) for s in ('train','test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _read(pair['input'])
                fields.append(_features(info))
                shapes.add(pair['input'].shape)
                layouts.add((_key(info['ids']), tuple((e['center'],e['radius']) for e in info['emitters'])))
        if seed == 0:
            again = generate_puzzle(seed)
            _need(all(_key(a[k]) == _key(b[k]) for s in puzzle for a,b in zip(puzzle[s],again[s])
                      for k in ('input','output')), 'nondeterministic seed')
    _need(len(set(puzzles)) == 42 and len(layouts) >= 150, 'collapsed structural variation')
    _need(len(shapes) >= 35, 'collapsed canvas geometry')
    _need({f['emitters'] for f in fields} == {1,2,3}, 'collapsed emitter counts')
    _need({r for f in fields for r in f['radii']} == {2,3,4}, 'collapsed radii')
    _need(len({f['territories'] for f in fields}) >= 5, 'collapsed territory counts')
    _need(sum(f['nonrectangular'] >= 3 for f in fields) >= 50, 'missing irregular partitions')
    _need(max(l for f in fields for l in f['lengths']) >= 12, 'missing long addresses')
    _need(any(f['failed'] and f['accepted'] for f in fields), 'missing mixed address outcomes')
    _need(any(len(set(f['radii'])) > 1 for f in fields), 'missing mixed radii')
    _need(any(f['multiple_incoming'] for f in fields), 'missing consistent multiple incoming deliveries')
    sample = generate_puzzle(0)
    permutation = np.array([4,7,1,8,0,5,2,6,3,9], dtype=np.int8)
    for split in ('train', 'test'):
        pair = sample[split][0]
        for turn in range(4):
            _need(np.array_equal(solve(np.rot90(pair['input'],turn)), np.rot90(pair['output'],turn)),
                  'geometric rule depends on canvas orientation')
        _need(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]),
              'literal fills/payloads depend on their color identities')
    # Direct geometric regressions have explicitly constructed expected arrays.
    # Equal endpoint colors are not equal component IDs.
    clean = np.full((17,17),6,dtype=np.int8)
    clean[:3,:3] = 4
    clean[10:12,12:15] = 2
    clean[12:15,10:12] = 2
    inp = clean.copy()
    center = (7,7)
    for point in _ring(center,2) | set(_cap(center,2,DIAGONAL[0])) | set(_cap(center,2,DIAGONAL[2])):
        inp[point] = INK
    inp[5,5] = 0
    expected = clean.copy()
    expected[:3,:3] = 0
    _need(np.array_equal(solve(inp),expected), 'component agreement or literal-zero regression')
    info = _read(inp)
    southeast = next(c for c in info['deliveries'] if c['direction'] == (1,1))
    _need(southeast['target'] is None and all(h[0] is not None for h in (southeast['hits'][0],southeast['hits'][-1])),
          'disconnected equal-color recipients were treated as one')
    # Complete cap against two edges: no endpoint has a next in-bounds cell.
    clean = np.full((13,15),6,dtype=np.int8)
    clean[8:,8:] = 4
    inp = clean.copy()
    center = (3,3)
    for point in _ring(center,2) | set(_cap(center,2,DIAGONAL[0])) | set(_cap(center,2,DIAGONAL[2])):
        inp[point] = INK
    _need(np.array_equal(solve(inp),np.full(clean.shape,6,dtype=np.int8)), 'boundary-miss regression')
    northwest = next(c for c in _read(inp)['deliveries'] if c['direction'] == (-1,-1))
    _need(northwest['target'] is None and northwest['hits'][0][0] is None
          and northwest['hits'][-1][0] is None, 'two misses were accepted as agreement')
    # A connected surrounding recipient receives conflicting explicit colors.
    inp = np.full((15,15),2,dtype=np.int8)
    inp[3:12,3:12] = 6
    center = (7,7)
    for point in _ring(center,2) | set(_cap(center,2,DIAGONAL[0])) | set(_cap(center,2,DIAGONAL[2])):
        inp[point] = INK
    inp[5,5], inp[9,9] = 0, 1
    try:
        solve(inp)
    except AssertionError:
        pass
    else:
        raise AssertionError('conflicting deliveries acquired an implicit priority')
    # The validator must catch an unpainted cell in a recipient, not only syntax.
    damaged = generate_puzzle(0, 2, 1)
    pair = damaged['test'][0]
    info = _read(pair['input'])
    target = next(t for t,p in info['assignments'].items() if p != info['colors'][t])
    point = next(iter(info['cells'][target]))
    pair['output'][point] = info['colors'][target]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a damaged recipient')
    for counts in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,False),(2.5,1),(None,1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
