"""Occlusion routes: a directed visibility graph of two-cell-thick strips.

The dashed border arm is the fast axis; the solid arm is the slow axis.
Only input pixels determine the graph, endpoint tags, proper rotation and width.
"""
from __future__ import annotations

from itertools import combinations
import math
import numpy as np


def _grid(value):
    array = np.asarray(value)
    if (array.ndim != 2 or not np.issubdtype(array.dtype, np.integer)
            or any(n < 1 or n > 30 for n in array.shape)
            or np.any((array < 0) | (array > 9))):
        raise ValueError('Expected a 1..30 integer grid with colors 0..9')
    return array.astype(np.int8, copy=True)


def _key(array):
    """Find the origin and arm roles from border geometry, not color identities."""
    candidates = []
    for bottom in (False, True):
        for right in (False, True):
            aligned = array[::-1] if bottom else array
            aligned = aligned[:, ::-1] if right else aligned
            if min(aligned.shape) < 5:
                continue
            marker, ink, background = map(int, (aligned[0, 0], aligned[0, 1], aligned[-1, -1]))
            if len({marker, ink, background}) != 3:
                continue
            if not (np.all(aligned[-1, 1:] == background)
                    and np.all(aligned[1:, -1] == background)):
                continue
            for axis in (0, 1):
                scene = aligned.T if axis else aligned
                dashed = np.where(np.arange(scene.shape[1]-1) % 2 == 0, ink, background)
                if np.array_equal(scene[0, 1:], dashed) and np.all(scene[1:, 0] == ink):
                    candidates.append((axis, bottom, right, background, ink, marker, scene))
    if len(candidates) != 1:
        raise ValueError('The border must determine exactly one dashed/solid coordinate key')
    return candidates[0]


def _rectangles(mask):
    """Four-neighbor components; payloads can be multicolor full rectangles."""
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    boxes = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        r0 = min(r for r, c in cells)
        c0 = min(c for r, c in cells)
        r1 = max(r for r, c in cells)+1
        c1 = max(c for r, c in cells)+1
        if len(cells) != (r1-r0)*(c1-c0) or r1-r0 != 2 or c1-c0 < 2:
            raise ValueError('Every payload must be a full two-row rectangle in keyed coordinates')
        boxes.append((r0, c0, c1-c0))
    return sorted(boxes)


def _visibility(boxes):
    """Each slow-direction ray connects consecutive hits, never skips a blocker."""
    adjacency = [set() for _ in boxes]
    for column in range(max(u+length for v, u, length in boxes)):
        hits = sorted((v, i) for i, (v, u, length) in enumerate(boxes)
                      if u <= column < u+length)
        for (_, a), (_, b) in zip(hits, hits[1:]):
            adjacency[a].add(b)
    return [sorted(neighbors) for neighbors in adjacency]


def _connected(adjacency):
    undirected = [set(a) for a in adjacency]
    for a, neighbors in enumerate(adjacency):
        for b in neighbors:
            undirected[b].add(a)
    reached, stack = {0}, [0]
    while stack:
        for b in undirected[stack.pop()]:
            if b not in reached:
                reached.add(b)
                stack.append(b)
    return len(reached) == len(adjacency)


def _shortest(adjacency, start, target):
    """DAG dynamic programming with path multiplicity capped at two."""
    distance = [100] * len(adjacency)
    ways = [0] * len(adjacency)
    previous = [-1] * len(adjacency)
    distance[start], ways[start] = 0, 1
    for a in range(start, len(adjacency)):
        if not ways[a]:
            continue
        for b in adjacency[a]:
            candidate = distance[a]+1
            if candidate < distance[b]:
                distance[b], ways[b], previous[b] = candidate, ways[a], a
            elif candidate == distance[b]:
                ways[b] = min(2, ways[b]+ways[a])
    if ways[target] != 1:
        return None
    path = [target]
    while path[-1] != start:
        path.append(previous[path[-1]])
    return path[::-1]


def _all_paths(adjacency, start, target):
    """Bounded by 15 nodes; used for generation evidence and validation only."""
    paths, stack = [], [(start, [start])]
    while stack:
        a, path = stack.pop()
        if a == target:
            paths.append(path)
        else:
            for b in adjacency[a]:
                if b <= target:
                    stack.append((b, path+[b]))
    return sorted(paths, key=lambda p: (len(p), p))


def _route_cost(boxes, path, euclidean=False):
    cost = 0.
    for a, b in zip(path, path[1:]):
        va, ua, la = boxes[a]
        vb, ub, lb = boxes[b]
        dv, du = vb-va, (ub+(lb-1)/2)-(ua+(la-1)/2)
        cost += math.hypot(dv, du) if euclidean else abs(dv)+abs(du)
    return cost


def _route_features(boxes, adjacency, path):
    paths = _all_paths(adjacency, path[0], path[-1])
    s, t = boxes[path[0]], boxes[path[-1]]
    overlap = max(s[1], t[1]) < min(s[1]+s[2], t[1]+t[2])
    shortest = len(paths) == 1 or len(paths[0]) < len(paths[1])
    count_witness = len(paths) > 1 and all(
        _route_cost(boxes, path, metric) > min(_route_cost(boxes, p, metric) for p in paths)+.01
        for metric in (False, True))
    selected = set(path)
    longest_off = max(b[2] for b in boxes) > max(boxes[i][2] for i in path)
    single_ray = max(boxes[i][1] for i in path) < min(boxes[i][1]+boxes[i][2] for i in path)
    return dict(paths=paths, shortest=shortest, count_witness=count_witness,
                blocked=overlap and path[-1] not in adjacency[path[0]],
                longest_off=longest_off, single_ray=single_ray,
                centering=len({boxes[i][2] for i in path}) > 1,
                skipped_lane=any(boxes[b][0]-boxes[a][0] > 3 for a, b in zip(path, path[1:])),
                off_route=len(boxes)-len(selected))


def _read_input(input_grid):
    array = _grid(input_grid)
    axis, bottom, right, background, ink, marker, scene = _key(array)
    mask = scene != background
    mask = mask.copy()
    mask[0, :], mask[:, 0] = False, False
    boxes = _rectangles(mask)
    if not 5 <= len(boxes) <= 15:
        raise ValueError('Expected 5..15 strip stations')
    lanes = sorted({v for v, u, length in boxes})
    if any(b-a < 3 for a, b in zip(lanes, lanes[1:])):
        raise ValueError('Lane bands must be aligned and separated')
    if any(v < 2 or u < 2 or v+2 >= scene.shape[0] or u+l >= scene.shape[1]
           for v, u, l in boxes):
        raise ValueError('A blank cell must separate payload from every outer border')
    if len({length % 2 for v, u, length in boxes}) != 1:
        raise ValueError('All strip lengths must share parity for exact centering')
    for v in lanes:
        intervals = sorted((u, u+l) for vv, u, l in boxes if vv == v)
        if any(b[0]-a[1] < 1 for a, b in zip(intervals, intervals[1:])):
            raise ValueError('Strips in a lane must be separated')
    raw = [scene[v:v+2, u:u+length].copy() for v, u, length in boxes]
    endpoints = []
    for i, block in enumerate(raw):
        tagged = np.flatnonzero(np.any(block == marker, axis=0))
        if len(tagged):
            if len(tagged) != 1 or not np.all(block[:, tagged[0]] == marker):
                raise ValueError('An endpoint tag is exactly one complete marker column')
            endpoints.append(i)
    if len(endpoints) != 2 or boxes[endpoints[0]][0] == boxes[endpoints[1]][0]:
        raise ValueError('Two tagged strips in different lanes define the endpoints')
    adjacency = _visibility(boxes)
    if not _connected(adjacency):
        raise ValueError('The visibility network must be weakly connected')
    path = _shortest(adjacency, *endpoints)
    if path is None:
        raise ValueError('Endpoint route is missing or the minimum station count ties')
    # Converting coordinate reflections/transposition to a proper rotation
    # requires correcting the thickness order; this never reflects a payload.
    reverse_rows = (bottom == right) if axis else (bottom != right)
    blocks = [b[::-1].copy() if reverse_rows else b for b in raw]
    return dict(boxes=boxes, adjacency=adjacency, path=path, blocks=blocks,
                raw=raw, background=background, ink=ink, marker=marker,
                configuration=(axis, bottom, right), endpoints=endpoints, lanes=lanes)


def _stack(blocks, path, background):
    width = max(blocks[i].shape[1] for i in path)
    output = np.full((2*len(path), width), background, dtype=np.int8)
    for j, i in enumerate(path):
        block = blocks[i]
        left = (width-block.shape[1])//2
        output[2*j:2*j+2, left:left+block.shape[1]] = block
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    data = _read_input(input_grid)
    return _stack(data['blocks'], data['path'], data['background'])


def _gaps(rng, count, spare):
    concentration = float(rng.choice([.4, 1., 3.]))
    return 1+rng.multinomial(spare, rng.dirichlet(np.full(count, concentration)))


def _layout(rng, larger=False):
    base = int(rng.choice([2, 3], p=[.72, .28]))
    n = int(rng.integers(8 if larger else 6, 16 if larger else 14))
    lanes = int(rng.integers(4 if larger else 3, min(8 if larger else 7, n)+1))
    counts = [1]*lanes
    capacity = 28//(base+1)
    preferences = rng.dirichlet(np.ones(lanes))
    for _ in range(n-lanes):
        eligible = [i for i, c in enumerate(counts) if c < min(capacity, 5)]
        weights = preferences[eligible]
        counts[int(rng.choice(eligible, p=weights/weights.sum()))] += 1
    lengths_by_lane = []
    for count in counts:
        ceiling = int(rng.choice([base+2*i for i in range(1, (27-base)//2+1)]))
        lengths = rng.choice(np.arange(base, ceiling+1, 2), size=count).astype(int).tolist()
        while sum(lengths) > 28-count:
            choices = [i for i, length in enumerate(lengths) if length > base]
            lengths[int(rng.choice(choices))] -= 2
        lengths_by_lane.append(lengths)
    min_width = max(sum(ls)+len(ls)+1 for ls in lengths_by_lane)
    fast_extent = int(rng.integers(min_width, 30))
    min_height = 3*lanes+1
    slow_extent = int(rng.integers(min_height, 30))
    slow_gaps = _gaps(rng, lanes+1, slow_extent-min_height)
    boxes, v = [], 1+int(slow_gaps[0])
    for j, lengths in enumerate(lengths_by_lane):
        fast_gaps = _gaps(rng, len(lengths)+1, fast_extent-sum(lengths)-len(lengths)-1)
        u = 1+int(fast_gaps[0])
        for k, length in enumerate(lengths):
            boxes.append((v, u, length))
            u += length+int(fast_gaps[k+1])
        v += 2+int(slow_gaps[j+1])
    return boxes, (slow_extent+1, fast_extent+1)


def _candidates(boxes, requirement):
    adjacency = _visibility(boxes)
    if not _connected(adjacency):
        return []
    result = []
    for s, t in combinations(range(len(boxes)), 2):
        if boxes[s][0] == boxes[t][0]:
            continue
        path = _shortest(adjacency, s, t)
        if path is None or len(path) < 3 or len(path) == len(boxes):
            continue
        f = _route_features(boxes, adjacency, path)
        if not f['centering']:
            continue
        if requirement == 'count' and not (len(path) == 3 and f['count_witness'] and f['blocked']):
            continue
        if requirement == 'width' and not (len(path) >= 4 and f['longest_off'] and not f['single_ray']):
            continue
        if requirement == 'test' and not (len(f['paths']) >= 2 and not f['single_ray']):
            continue
        if requirement == 'ordinary' and len(f['paths']) < 2:
            continue
        result.append(path)
    return result


def _fallback(rng, requirement):
    """Rule-consistent constructive witnesses, used only after bounded rejection."""
    if requirement == 'count':
        # S->A->T beats S->B->C->T by count, but loses on both geometric costs.
        boxes = [(2, 2, 10), (5, 2, 8), (8, 2, 8), (11, 5, 10), (14, 4, 2)]
    elif requirement == 'width':
        # Four route stations shift sideways; the long right-hand dead end
        # must not set the output width. S->B->C->T, plus S->unused.
        boxes = [(2, 6, 6), (5, 3, 6), (5, 11, 10), (8, 2, 4), (11, 2, 4)]
    else:
        boxes = [(2, 2, 4), (2, 8, 6), (5, 2, 8), (8, 5, 4),
                 (8, 11, 8), (11, 3, 4), (14, 4, 4), (14, 10, 10)]
    choices = _candidates(boxes, requirement)
    if not choices:
        raise AssertionError('Constructive fallback does not meet its evidence requirement')
    # Random translation and wider margins do not change visibility topology.
    dv, du = int(rng.integers(3)), int(rng.integers(4))
    boxes = [(v+dv, u+du, length) for v, u, length in boxes]
    shape = (max(v+2 for v, u, l in boxes)+2+int(rng.integers(3)),
             max(u+l for v, u, l in boxes)+2+int(rng.integers(3)))
    return boxes, shape, choices[int(rng.integers(len(choices)))]


def _sample_scene(rng, requirement):
    desired = 3 if requirement == 'count' else int(rng.integers(4 if requirement == 'width' else 3, 8))
    for attempt in range(240):
        boxes, shape = _layout(rng, requirement == 'test')
        choices = _candidates(boxes, requirement)
        if choices:
            if attempt < 65 and max(map(len, choices)) < desired:
                continue
            gap = min(abs(len(path)-desired) for path in choices)
            choices = [path for path in choices if abs(len(path)-desired) == gap]
            path = choices[int(rng.integers(len(choices)))]
            return boxes, shape, path
    return _fallback(rng, requirement)


def _asymmetric(block):
    return all(not np.array_equal(block, other)
               for other in (block[::-1], block[:, ::-1], block[::-1, ::-1]))


def _payloads(rng, boxes, path, roles, mode):
    background, ink, marker = roles
    pool = [c for c in range(10) if c not in (background, marker)]
    palette = rng.choice(pool, size=int(rng.integers(3, len(pool)+1)), replace=False)
    blocks = []
    for v, u, length in boxes:
        kind = int(rng.integers(4)) if mode else 0
        if kind == 0:
            block = np.full((2, length), rng.choice(palette), dtype=np.int8)
        elif kind == 1:
            block = np.repeat(rng.choice(palette, size=(1, length)), 2, axis=0).astype(np.int8)
        elif kind == 2:
            block = np.repeat(rng.choice(palette, size=(2, 1)), length, axis=1).astype(np.int8)
        else:
            block = rng.choice(palette, size=(2, length)).astype(np.int8)
        repeated = [b for b in blocks if b.shape == block.shape and marker not in b]
        if repeated and rng.random() < .25:
            block = repeated[int(rng.integers(len(repeated)))].copy()
        blocks.append(block)
    # A varying selected ordinary strip teaches proper rotation without a fixed
    # first-block signature. Endpoints cannot supply this witness by tag alone.
    index = int(rng.choice(path[1:-1]))
    for _ in range(32):
        block = rng.choice(palette, size=blocks[index].shape).astype(np.int8)
        if _asymmetric(block):
            break
    else:
        block = np.full(blocks[index].shape, palette[0], dtype=np.int8)
        block[0, 0], block[1, -1] = palette[1], palette[2]
    blocks[index] = block
    for i in (path[0], path[-1]):
        blocks[i][:, int(rng.integers(blocks[i].shape[1]))] = marker
    return blocks


def _paint(boxes, shape, raw, path, configuration, roles):
    axis, bottom, right = configuration
    background, ink, marker = roles
    scene = np.full(shape, background, dtype=np.int8)
    scene[0, 0] = marker
    scene[0, 1:] = np.where(np.arange(shape[1]-1) % 2 == 0, ink, background)
    scene[1:, 0] = ink
    for (v, u, length), block in zip(boxes, raw):
        scene[v:v+2, u:u+length] = block
    grid = scene.T if axis else scene
    if bottom:
        grid = grid[::-1]
    if right:
        grid = grid[:, ::-1]
    reverse_rows = (bottom == right) if axis else (bottom != right)
    normalized = [b[::-1] if reverse_rows else b for b in raw]
    # Label uses the intended sampled route; input-only reader independently
    # recovers station rectangles, first-hit edges and the unique minimum path.
    return {'input': grid.copy(), 'output': _stack(normalized, path, background)}


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def _array_key(array):
    return array.shape, array.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    first = (int(rng.integers(2)), bool(rng.integers(2)), bool(rng.integers(2)))
    second = (1-first[0], not first[1], not first[2])
    configs = [(a, b, r) for a in (0, 1) for b in (False, True) for r in (False, True)]
    # Both axes and both signs are taught. Queries hold out a combination of
    # these factors; extra demonstrations never erase that compositional shift.
    train_pool = [first, second]
    other = [c for c in configs if c not in train_pool]
    rng.shuffle(other)
    train_pool += other[:int(rng.integers(3))]
    test_pool = [c for c in configs if c not in train_pool]
    base_roles = tuple(map(int, rng.choice(10, 3, replace=False)))
    palette_mode = int(rng.integers(3))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            requirement = ('count' if i == 0 else 'width' if i == 1 else 'ordinary') if split == 'train' else 'test'
            config = first if split == 'train' and i == 0 else second if split == 'train' and i == 1 else (
                train_pool[int(rng.integers(len(train_pool)))] if split == 'train'
                else test_pool[int(rng.integers(len(test_pool)))])
            roles = base_roles if palette_mode == 0 or (palette_mode == 1 and rng.random() < .6) else tuple(
                map(int, rng.choice(10, 3, replace=False)))
            for _ in range(8):
                boxes, shape, path = _sample_scene(rng, requirement)
                raw = _payloads(rng, boxes, path, roles, bool(rng.integers(2)))
                pair = _paint(boxes, shape, raw, path, config, roles)
                key = _array_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('Could not construct distinct inputs in bounded retries')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError('Puzzle must have exactly train and test')
    if not all(isinstance(puzzle[s], list) for s in ('train', 'test')):
        raise ValueError('Splits must be lists')
    _count(len(puzzle['train']), 2, 10, 'num_train')
    _count(len(puzzle['test']), 1, 4, 'num_test')
    if len(puzzle['train']) <= len(puzzle['test']):
        raise ValueError('num_train must exceed num_test')
    seen, training, test_configs = set(), [], set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
                raise ValueError('Each pair must have exactly input/output')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _array_key(inp)
            if key in seen:
                raise ValueError('Duplicate input')
            seen.add(key)
            data = _read_input(inp)
            expected = _stack(data['blocks'], data['path'], data['background'])
            if not np.array_equal(expected, out):
                raise ValueError('Output is not the exact rotated/centered minimum-station route')
            # Exhaustive enumeration checks the DP's uniqueness and station cost.
            f = _route_features(data['boxes'], data['adjacency'], data['path'])
            if not f['shortest'] or f['paths'][0] != data['path']:
                raise ValueError('Route is not the unique station minimum')
            if len(data['path']) < 3 or not any(_asymmetric(data['blocks'][i]) for i in data['path'][1:-1]):
                raise ValueError('Generated examples need an ordinary asymmetric route station')
            if split == 'train':
                training.append((data, f))
            else:
                test_configs.add(data['configuration'])
    configurations = {d['configuration'] for d, f in training}
    if {c[0] for c in configurations} != {0, 1} or {c[1] for c in configurations} != {False, True} or {c[2] for c in configurations} != {False, True}:
        raise ValueError('Training must teach both axes and both origin signs')
    if not test_configs.isdisjoint(configurations):
        raise ValueError('Generated tests must hold out a keyed direction combination')
    if not any(f['count_witness'] and f['blocked'] and f['centering'] for d, f in training):
        raise ValueError('Training needs a station-count/geometry conflict with occlusion')
    if not any(f['longest_off'] and not f['single_ray'] for d, f in training):
        raise ValueError('Training needs a lateral route and a wider excluded strip')
    if len({len(d['path']) for d, f in training}) < 2:
        raise ValueError('Training must demonstrate different answer station counts')


def _expect_bad(call):
    try:
        call()
    except (ValueError, AssertionError):
        return
    raise AssertionError('Invalid case was accepted')


def self_test():
    """42-seed diversity, geometry metamorphisms, and deliberate bad cases."""
    puzzles, shapes, layouts, configurations, colors, lengths, counts = set(), set(), set(), set(), set(), set(), set()
    statistics = {'pairs': 0, 'count_witnesses': 0, 'blocked': 0, 'lateral': 0, 'off_width': 0}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signature = tuple(_array_key(np.asarray(p[k])) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output'))
        puzzles.add(signature)
        if seed < 2:
            repeated = generate_puzzle(seed)
            assert signature == tuple(_array_key(p[k]) for s in ('train', 'test') for p in repeated[s] for k in ('input', 'output'))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                d = _read_input(inp)
                f = _route_features(d['boxes'], d['adjacency'], d['path'])
                assert np.array_equal(solve(inp), out)
                # Proper whole-scene rotations move the key with the payload;
                # the normalized answer is invariant under all four rotations.
                if seed < 4:
                    for k in range(4):
                        assert np.array_equal(solve(np.rot90(inp, k)), out)
                    permutation = np.random.default_rng(seed+100).permutation(10)
                    assert np.array_equal(solve(permutation[inp]), permutation[out])
                shapes.add((inp.shape, out.shape))
                layouts.add((tuple(d['boxes']), tuple(tuple(a) for a in d['adjacency']), tuple(d['path'])))
                configurations.add(d['configuration'])
                colors.add((d['background'], d['ink'], d['marker']))
                lengths.update(b[2] for b in d['boxes'])
                counts.add((len(d['boxes']), len(d['path']), len(d['lanes'])))
                statistics['pairs'] += 1
                statistics['count_witnesses'] += f['count_witness']
                statistics['blocked'] += f['blocked']
                statistics['lateral'] += not f['single_ray']
                statistics['off_width'] += f['longest_off']
    assert len(puzzles) == 42 and len(layouts) >= 170
    assert len(shapes) >= 90 and len(colors) >= 35 and len(counts) >= 45
    assert len(configurations) == 8 and {l % 2 for l in lengths} == {0, 1}
    assert max(lengths) >= 20
    assert statistics['count_witnesses'] >= 42, statistics
    assert statistics['blocked'] >= 42, statistics
    assert statistics['lateral'] >= 84 and statistics['off_width'] >= 42, statistics
    # Every legal count combination and type/range/relation failures.
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+10*a+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.5), (None, 1)):
        _expect_bad(lambda a=a, b=b: generate_puzzle(0, a, b))
    damaged = generate_puzzle(500)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    _expect_bad(lambda: validate_generated_puzzle(damaged))
    duplicated = generate_puzzle(501)
    duplicated['test'][0] = duplicated['train'][0]
    _expect_bad(lambda: validate_generated_puzzle(duplicated))
    # A diamond really ties; BFS iteration order is never a tie breaker.
    assert _shortest([[1, 2], [3], [3], []], 0, 3) is None
    assert _shortest([[1, 2], [3], [4], [4], []], 0, 4) == [0, 2, 4]
    # Semantic members outside the informative random-demo schedule: uniform
    # strips, repeated occurrences, equal lengths, width 27 and nine lanes.
    for length in (2, 3, 26, 27):
        boxes = [(2+3*i, 2, length) for i in range(9)]
        raw = [np.full((2, length), 4, dtype=np.int8) for _ in boxes]
        raw[0][:, -1] = 2
        raw[-1][:, 0] = 2
        for axis in (0, 1):
            for bottom in (False, True):
                for right in (False, True):
                    pair = _paint(boxes, (29, length+3), raw, list(range(9)),
                                  (axis, bottom, right), (0, 1, 2))
                    assert np.array_equal(solve(pair['input']), pair['output'])
    boxes = [(2, 6, 6), (5, 3, 6), (5, 11, 10), (8, 2, 4), (11, 2, 4)]
    raw = [np.full((2, length), 4, dtype=np.int8) for v, u, length in boxes]
    raw[0][:, 1] = 2
    raw[2][:, -2] = 2
    direct = _paint(boxes, (15, 23), raw, [0, 2], (0, False, True), (0, 1, 2))
    assert np.array_equal(solve(direct['input']), direct['output'])
    # A rendered diamond tie must fail at input parsing, as must a payload hole
    # and a broken endpoint tag. No supplied output is consulted by these checks.
    boxes = [(2, 2, 8), (5, 2, 4), (5, 8, 4), (8, 6, 4), (11, 2, 8)]
    raw = [np.full((2, length), 4, dtype=np.int8) for v, u, length in boxes]
    raw[0][:, 0] = raw[-1][:, 0] = 2
    tied = _paint(boxes, (15, 15), raw, [0, 1, 4], (0, False, False), (0, 1, 2))
    _expect_bad(lambda: solve(tied['input']))
    good = generate_puzzle(502)['train'][0]['input']
    d = _read_input(good)
    axis, bottom, right, bg, ink, marker, scene = _key(good)
    hole = scene.copy()
    v, u, length = d['boxes'][0]
    hole[v, u] = bg
    _expect_bad(lambda: solve(hole))
    broken_tag = scene.copy()
    v, u, length = d['boxes'][d['endpoints'][0]]
    col = int(np.flatnonzero(scene[v, u:u+length] == marker)[0])+u
    broken_tag[v, col] = ink
    _expect_bad(lambda: solve(broken_tag))
    return dict(seeds=42, distinct_puzzles=len(puzzles), layouts=len(layouts),
                shape_pairs=len(shapes), role_assignments=len(colors),
                count_signatures=len(counts), lengths=sorted(lengths), **statistics)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
