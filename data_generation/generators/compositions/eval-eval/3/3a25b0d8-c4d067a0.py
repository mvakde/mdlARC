"""Complete a deformed loop of enclosed chambers from a painted fragment.

Only the standard library and NumPy are used. Construction paints the answer
from an abstract cyclic word; solving recovers enclosures and links from pixels.
"""
from functools import lru_cache
import numpy as np


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    cells.append(p)
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(points):
    r, c = points.min(axis=0)
    R, C = points.max(axis=0) + 1
    return int(r), int(c), int(R), int(C)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(int, copy=True)


def _key(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _holes(frame):
    r, c, R, C = _bbox(frame)
    mask = np.ones((R - r + 2, C - c + 2), dtype=bool)
    mask[frame[:, 0] - r + 1, frame[:, 1] - c + 1] = False
    return [p + (r - 1, c - 1) for p in _components(mask)
            if np.all(p > 0) and np.all(p < np.array(mask.shape) - 1)]


def _cycle(nodes, adjacency):
    start = min(nodes)
    order, previous, current = [start], start, min(adjacency[start])
    while current != start:
        assert current not in order, 'links must make a simple cycle'
        order.append(current)
        nxt = adjacency[current] - {previous}
        assert len(nxt) == 1
        previous, current = current, next(iter(nxt))
    assert set(order) == set(nodes)
    return order


def _completion_options(source, known, background):
    """Dihedral cyclic matches; color symmetries may give equivalent mappings."""
    n = len(source)
    answers = set()
    for sign in (-1, 1):
        for offset in range(n):
            values = tuple(int(source[(offset + sign * j) % n]) for j in range(n))
            if all(k == background or k == v for k, v in zip(known, values)):
                answers.add(values)
    return answers


def _recover(input_grid):
    inp = _grid(input_grid)
    bg = int(inp[0, 0])
    assert all(np.all(edge == bg) for edge in (inp[0], inp[-1], inp[:, 0], inp[:, -1])), \
        'the outer border identifies the background'
    objects = _components(inp != bg)
    assert len(objects) == 2, 'expected two linked chamber figures'

    # A frame color uniquely forms several disconnected closed rings. Payload
    # colors are solid, hole-free regions; link-color paths have open ends.
    candidates = []
    for color in sorted(set(map(int, inp.flat)) - {bg}):
        parts = _components(inp == color)
        holes = [_holes(p) for p in parts]
        if len(parts) >= 8 and all(len(h) == 1 for h in holes):
            candidates.append((color, parts, [h[0] for h in holes]))
    assert len(candidates) == 1, 'frame identity must be visible'
    wall, frames, rooms = candidates[0]
    frame_owner = np.full(inp.shape, -1, dtype=int)
    interior = np.zeros(inp.shape, dtype=bool)
    room_colors = []
    for i, (frame, room) in enumerate(zip(frames, rooms)):
        frame_owner[tuple(frame.T)] = i
        assert not np.any(interior[tuple(room.T)])
        interior[tuple(room.T)] = True
        color = np.unique(inp[tuple(room.T)])
        assert len(color) == 1 and color[0] != wall, 'uniform chamber interiors'
        room_colors.append(int(color[0]))
    outside = (inp != bg) & (inp != wall) & ~interior
    wire_colors = np.unique(inp[outside])
    assert len(wire_colors) == 1, 'one shared link color'
    wire = int(wire_colors[0])
    assert wire not in (bg, wall) and wire not in room_colors
    adjacency = [set() for _ in rooms]
    links = _components(outside)
    for link in links:
        link_set = set(map(tuple, link))
        degrees = [sum(p in link_set for p in ((r - 1, c), (r + 1, c),
                                              (r, c - 1), (r, c + 1))) for r, c in link]
        assert max(degrees) <= 2 and (len(link) == 1 or degrees.count(1) == 2), \
            'each link is an unbranched open pixel path'
        endpoints = set()
        for r, c in link:
            for y, x in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= y < inp.shape[0] and 0 <= x < inp.shape[1]:
                    owner = int(frame_owner[y, x])
                    if owner >= 0:
                        endpoints.add(owner)
        assert len(endpoints) == 2, 'each link joins two chamber rims'
        a, b = sorted(endpoints)
        assert b not in adjacency[a], 'no duplicate links'
        adjacency[a].add(b)
        adjacency[b].add(a)
    assert all(len(a) == 2 for a in adjacency)

    cycles, unvisited = [], set(range(len(rooms)))
    while unvisited:
        start = min(unvisited)
        stack, nodes = [start], {start}
        while stack:
            for other in adjacency[stack.pop()]:
                if other not in nodes:
                    nodes.add(other)
                    stack.append(other)
        unvisited -= nodes
        cycles.append(_cycle(nodes, adjacency))
    assert len(cycles) == 2 and len(cycles[0]) == len(cycles[1])
    n = len(cycles[0])
    assert 4 <= n <= 12
    donors = [cy for cy in cycles if all(room_colors[i] != bg for i in cy)]
    targets = [cy for cy in cycles if any(room_colors[i] == bg for i in cy)]
    assert len(donors) == len(targets) == 1
    donor, target = donors[0], targets[0]
    source = tuple(room_colors[i] for i in donor)
    known = tuple(room_colors[i] for i in target)
    assert len(set(source)) >= 2
    assert set(source).isdisjoint((bg, wall, wire))
    occupied = [k != bg for k in known]
    length = sum(occupied)
    assert 2 <= length <= min(4, n - 1)
    assert sum(occupied[j] and not occupied[(j - 1) % n] for j in range(n)) == 1, \
        'the painted clue is one contiguous cyclic run'
    choices = _completion_options(source, known, bg)
    assert len(choices) == 1, 'painted fragment must fix all missing colors'
    completion = next(iter(choices))
    out = inp.copy()
    blank_cells = np.zeros(inp.shape, dtype=bool)
    for i, color in zip(target, completion):
        if room_colors[i] == bg:
            blank_cells[tuple(rooms[i].T)] = True
        out[tuple(rooms[i].T)] = color

    # This intentionally wrong rule skips the cyclic fragment match.
    # Discovery order is spatial scan order, not a matching convention.
    scan_colors = [room_colors[i] for i in sorted(donor)]
    scan_differs = any(room_colors[i] == bg and scan_colors[j] != completion[target.index(i)]
                       for j, i in enumerate(sorted(target)))
    meta = dict(background=bg, wall=wall, wire=wire, count=n, length=length,
                source=source, known=known, completion=completion, donor=donor,
                target=target, frames=frames, rooms=rooms, links=links,
                blank_cells=blank_cells, scan_differs=scan_differs,
                large_blank=any(len(rooms[i]) > 1 for i in target if room_colors[i] == bg),
                room_areas=tuple(len(p) for p in rooms),
                nonrectangular=sum(len(p) != (_bbox(p)[2] - _bbox(p)[0]) *
                                   (_bbox(p)[3] - _bbox(p)[1]) for p in rooms))
    return out, meta


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the entire answer from visible enclosures, links, and seed colors."""
    return _recover(input_grid)[0]


@lru_cache(maxsize=1)
def _polygons():
    """Enumerate simple orthogonal boundaries of connected 2-by-4 polyominoes.

    This creates convex and concave loops, rather than a bank of finished grids.
    Vertex-touching boundaries and holes are excluded.
    """
    result = set()
    for bits in range(1, 256):
        cells = {(i // 4, i % 4) for i in range(8) if bits & (1 << i)}
        reached, stack = {min(cells)}, [min(cells)]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in cells and p not in reached:
                    reached.add(p)
                    stack.append(p)
        if reached != cells:
            continue
        edges = set()
        for r, c in cells:
            corners = ((r, c), (r, c + 1), (r + 1, c + 1), (r + 1, c))
            for j in range(4):
                edge = tuple(sorted((corners[j], corners[(j + 1) % 4])))
                if edge in edges:
                    edges.remove(edge)
                else:
                    edges.add(edge)
        graph = {}
        for a, b in edges:
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
        if any(len(v) != 2 for v in graph.values()):
            continue
        start = min(graph)
        order, prev, cur = [start], start, min(graph[start])
        while cur != start and cur not in order:
            order.append(cur)
            prev, cur = cur, next(iter(graph[cur] - {prev}))
        if cur != start or len(order) != len(graph):
            continue
        mr = min(p[0] for p in order)
        mc = min(p[1] for p in order)
        result.add(tuple((r - mr, c - mc) for r, c in order))
    return tuple(sorted(result))


def _word_and_fragment(rng, n, length, palette, bg, fallback=False):
    for attempt in range(64):
        if fallback or attempt == 63:
            source = [int(palette[2])] * n
            source[0], source[1] = map(int, palette[:2])
        else:
            source = list(map(int, rng.choice(palette, size=n)))
        if len(set(source)) < 2:
            continue
        sign, offset = int(rng.choice([-1, 1])), int(rng.integers(n))
        target = [source[(offset + sign * j) % n] for j in range(n)]
        starts = []
        for start in range(n):
            known = [bg] * n
            for k in range(length):
                j = (start + k) % n
                known[j] = target[j]
            if len(_completion_options(source, known, bg)) == 1:
                starts.append(start)
        if starts:
            start = int(rng.choice(starts))
            painted = {(start + k) % n for k in range(length)}
            return source, target, painted
    raise AssertionError('constructive cyclic word failed')


def _figure(rng, n, bg, wall, wire, colors, painted, compact=False,
            fallback=False, fixed=False):
    polygons = [p for p in _polygons() if len(p) >= n]
    poly = np.array(polygons[0 if fixed else int(rng.integers(len(polygons)))], dtype=int)
    # Reflections affect the embedding, never the abstract link order.
    if not fixed and rng.integers(2):
        poly[:, 0] *= -1
    if not fixed and rng.integers(2):
        poly[:, 1] *= -1
    poly -= poly.min(axis=0)
    selected = list(range(n)) if fixed else sorted(map(int, rng.choice(len(poly), n, replace=False)))
    pitch = 4 if compact else 5
    positions = poly * pitch + 1
    h, w = positions.max(axis=0) + (2 if compact else 3)
    grid = np.full((h, w), bg, dtype=int)
    for a, b in zip(positions, np.roll(positions, -1, axis=0)):
        r, c = map(int, a)
        R, C = map(int, b)
        if r == R:
            grid[r, min(c, C):max(c, C) + 1] = wire
        else:
            grid[min(r, R):max(r, R) + 1, c] = wire
    interiors = []
    for j, vertex in enumerate(selected):
        r, c = map(int, positions[vertex])
        rh = 1 if compact else int(rng.integers(1, 3))
        cw = 1 if compact else int(rng.integers(1, 3))
        if fallback and not compact:
            rh = cw = 2
        grid[r - 1:r + rh + 1, c - 1:c + cw + 1] = wall
        room = [(y, x) for y in range(r, r + rh) for x in range(c, c + cw)]
        if rh == cw == 2 and not fallback and rng.random() < .45:
            room.remove(room[int(rng.integers(4))])
        points = np.array(room, dtype=int)
        grid[tuple(points.T)] = colors[j] if j in painted else bg
        interiors.append(points)
    # Each figure is cropped before its independent transpose, preventing
    # arbitrary padding from becoming part of the object geometry.
    r, c, R, C = _bbox(np.argwhere(grid != bg))
    grid = grid[r:R, c:C]
    interiors = [p - (r, c) for p in interiors]
    return grid, interiors


def _layout(rng, donor, target, room_points, colors, bg):
    options = []
    for dt in (False, True):
        for tt in (False, True):
            ds = donor.T.shape if dt else donor.shape
            ts = target.T.shape if tt else target.shape
            for axis in (0, 1):
                shape = [max(ds[k], ts[k]) + 2 for k in (0, 1)]
                shape[axis] = ds[axis] + ts[axis] + 3
                if max(shape) <= 30:
                    options.append((dt, tt, axis, shape))
    assert options, 'compact donor and target must fit'
    dt, tt, axis, required = options[int(rng.integers(len(options)))]
    if dt:
        donor = donor.T.copy()
    if tt:
        target = target.T.copy()
        room_points = [p[:, ::-1] for p in room_points]
    shape = tuple(n + int(rng.integers(min(30 - n, 4) + 1)) for n in required)
    out_axis = 1 - axis
    dp, tp = [0, 0], [0, 0]
    dp[out_axis] = int(rng.integers(1, shape[out_axis] - donor.shape[out_axis]))
    tp[out_axis] = int(rng.integers(1, shape[out_axis] - target.shape[out_axis]))
    extra = shape[axis] - required[axis]
    first = int(rng.integers(extra + 1)) + 1
    if rng.integers(2):
        dp[axis] = first
        tp[axis] = first + donor.shape[axis] + 1
    else:
        tp[axis] = first
        dp[axis] = first + target.shape[axis] + 1
    inp = np.full(shape, bg, dtype=int)
    for figure, pos in ((donor, dp), (target, tp)):
        r, c = pos
        inp[r:r + figure.shape[0], c:c + figure.shape[1]] = figure
    out = inp.copy()
    for points, color in zip(room_points, colors):
        out[tuple((points + tp).T)] = color
    return {'input': inp, 'output': out}


def _constructive_pair(rng, n, length, teaching, excluded_bg):
    """Guaranteed fitting geometry, then exhaustive finite phase selection."""
    roles = list(map(int, rng.permutation(10)))
    if roles[0] == excluded_bg:
        roles[0], roles[1] = roles[1], roles[0]
    bg, wall, wire, a, b, c = roles[:6]
    source = [a, b] + [c] * (n - 2)
    donor, _ = _figure(rng, n, bg, wall, wire, source, set(range(n)), True, True, True)
    target, points = _figure(rng, n, bg, wall, wire, source, set(range(length)), False, True, True)
    base = _layout(rng, donor, target, points, source, bg)
    _, info = _recover(base['input'])
    # Layout may transpose either figure. Read the visible traversal once;
    # the fallback paints an explicitly chosen cyclic word on the other loop.
    source = info['source']
    for offset in range(n):
        for sign in (-1, 1):
            values = [source[(offset + sign * j) % n] for j in range(n)]
            known = values[:length] + [bg] * (n - length)
            if len(_completion_options(source, known, bg)) != 1:
                continue
            inp, out = base['input'].copy(), base['input'].copy()
            for j, room_index in enumerate(info['target']):
                room = info['rooms'][room_index]
                inp[tuple(room.T)] = known[j]
                out[tuple(room.T)] = values[j]
            expected, meta = _recover(inp)
            assert np.array_equal(expected, out)
            if not teaching or (meta['large_blank'] and meta['scan_differs']):
                return {'input': inp, 'output': out}
    raise AssertionError('constructive phase enumeration failed')


def _make_pair(rng, n, length, teaching, excluded_bg=None):
    for _ in range(64):
        roles = list(map(int, rng.permutation(10)))
        if roles[0] == excluded_bg:
            roles[0], roles[1] = roles[1], roles[0]
        bg, wall, wire = roles[:3]
        pool_size = int(rng.integers(3, min(7, n) + 1))
        source, target, painted = _word_and_fragment(rng, n, length, roles[3:3 + pool_size],
                                                    bg)
        # Larger donors are also allowed where the pair fits. The guaranteed
        # compact mode makes all 12-chamber count requests constructible.
        compact = bool(rng.random() < .65)
        donor, _ = _figure(rng, n, bg, wall, wire, source, set(range(n)), compact)
        target_grid, points = _figure(rng, n, bg, wall, wire, target, painted)
        try:
            pair = _layout(rng, donor, target_grid, points, target, bg)
            predicted, meta = _recover(pair['input'])
            assert np.array_equal(predicted, pair['output'])
            if teaching:
                assert meta['large_blank'] and meta['scan_differs']
                assert len(set(meta['source'])) < n, 'teach repeated donor colors'
            return pair
        except AssertionError:
            continue
    return _constructive_pair(rng, n, length, teaching, excluded_bg)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    ns = _count(num_test, 'num_test', 1, 4)
    if nt <= ns:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    maximum = int(rng.integers(6, 9))
    result, seen = {'train': [], 'test': []}, set()
    first_bg = None
    for split, count in (('train', nt), ('test', ns)):
        for index in range(count):
            if split == 'train':
                n = maximum if index == 0 else maximum - 2 if index == 1 else int(rng.integers(4, maximum + 1))
                length = 2 if index == 0 else 3 if index == 1 else int(rng.integers(2, min(4, n - 1) + 1))
            else:
                n = int(rng.integers(maximum + 1, 13))
                length = int(rng.integers(2, 5))
            pair = _make_pair(rng, n, length, split == 'train',
                              first_bg if split == 'train' and index == 1 else None)
            if _key(pair['input']) in seen:
                # Ninety injective role assignments exceed the maximum 14
                # examples. A duplicate can always be changed visibly.
                original = pair
                active = sorted(set(map(int, original['input'].flat)))
                domain = active + [c for c in range(10) if c not in active]
                for a in range(10):
                    for b in range(10):
                        if a == b:
                            continue
                        mapping = np.empty(10, dtype=int)
                        mapping[domain] = [a, b] + [c for c in range(10) if c not in (a, b)]
                        candidate = {k: mapping[v] for k, v in original.items()}
                        if _key(candidate['input']) not in seen and not (
                                split == 'train' and index == 1 and candidate['input'][0, 0] == first_bg):
                            pair = candidate
                            break
                    else:
                        continue
                    break
                assert _key(pair['input']) not in seen
            seen.add(_key(pair['input']))
            result[split].append(pair)
            if split == 'train' and index == 0:
                first_bg = int(pair['input'][0, 0])
        rng.shuffle(result[split])
    return result


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            expected, meta = _recover(inp)
            assert np.array_equal(out, expected), 'wrong cyclic completion'
            changes = inp != out
            assert np.array_equal(changes, meta['blank_cells'])
            assert np.all(inp[changes] == meta['background'])
            assert np.array_equal(inp[~changes], out[~changes]), 'walls, links, and donor must survive'
            assert changes.any()
            for room in meta['rooms']:
                assert len(np.unique(out[tuple(room.T)])) == 1
            records[split].append(meta)
    # The demonstrations teach both mechanisms even for the 2/1 request.
    assert all(m['large_blank'] and m['scan_differs'] for m in records['train'])
    assert all(len(set(m['source'])) < m['count'] for m in records['train'])
    assert len({m['count'] for m in records['train']}) >= 2
    assert {2, 3} <= {m['length'] for m in records['train']}
    assert len({m['background'] for m in records['train']}) >= 2
    assert min(m['count'] for m in records['test']) > max(m['count'] for m in records['train'])


def self_test():
    """Bounded semantic/variation checks over 42 seeds, plus negative checks."""
    shapes, roles, counts, lengths, structures, signatures = set(), set(), set(), set(), set(), set()
    records = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                predicted, m = _recover(pair['input'])
                assert np.array_equal(predicted, pair['output'])
                shapes.add(pair['input'].shape)
                roles.add((m['background'], m['wall'], m['wire']))
                counts.add(m['count'])
                lengths.add(m['length'])
                structure = np.zeros(pair['input'].shape, dtype=np.int8)
                structure[pair['input'] == m['wall']] = 1
                structure[pair['input'] == m['wire']] = 2
                structure[(pair['input'] != m['background']) & (structure == 0)] = 3
                structures.add(_key(structure))
                records.append(m)
    assert len(signatures) == 42 and len(structures) >= 150
    assert len(shapes) >= 25 and len(roles) >= 70
    assert counts == set(range(4, 13)) and lengths == {2, 3, 4}
    assert sum(m['nonrectangular'] > 0 for m in records) >= 25
    assert any(m['count'] == 12 and m['length'] == 2 for m in records)
    assert any(len(set(m['room_areas'])) >= 3 for m in records)
    a, b = generate_puzzle(372596936, 10, 4), generate_puzzle(372596936, 10, 4)
    assert all(_key(p[k]) == _key(q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    validate_generated_puzzle(a)
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1000 + 10 * nt + ns, nt, ns))
    for counts_bad in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, *counts_bad)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Unit-level cyclic ambiguity and a longer fragment resolving a short one.
    assert len(_completion_options([1, 2, 3, 1, 2, 4], [1, 2, 0, 0, 0, 0], 0)) > 1
    assert len(_completion_options([1, 2, 3, 1, 2, 4], [1, 2, 3, 0, 0, 0], 0)) == 1
    # Exercise the constructive path directly for every supported chamber and
    # fragment length, including the one-blank four-chamber case.
    fallback_rng = np.random.default_rng(2026)
    for n in range(4, 13):
        for length in range(2, min(4, n - 1) + 1):
            pair = _constructive_pair(fallback_rng, n, length, True, None)
            prediction, info = _recover(pair['input'])
            assert np.array_equal(prediction, pair['output'])
            assert info['large_blank'] and info['scan_differs']
    # Uniform wrong fill, changed frame, and changed donor are all rejected.
    base = generate_puzzle(999, 2, 1)
    _, meta = _recover(base['test'][0]['input'])
    for point in (tuple(np.argwhere(meta['blank_cells'])[0]), tuple(meta['frames'][0][0]),
                  tuple(meta['rooms'][meta['donor'][0]][0])):
        damaged = {s: [{k: v.copy() for k, v in p.items()} for p in base[s]] for s in base}
        damaged['test'][0]['output'][point] = (int(damaged['test'][0]['output'][point]) + 1) % 10
        try:
            validate_generated_puzzle(damaged)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupted output accepted')
    return {'seeds': 42, 'pairs': len(records), 'distinct_structures': len(structures),
            'canvas_shapes': len(shapes), 'role_palettes': len(roles),
            'room_counts': sorted(counts), 'seed_lengths': sorted(lengths)}
