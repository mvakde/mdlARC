"""Ranked vacancy routes on two geometrically different compartment maps.

Only standard-library modules and NumPy are used. The public solver reads
roles, room adjacency, endpoint labels, and operation order from the grid.
"""
from collections import deque
import json
import numpy as np


_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_HOST_SEEDS = (0, 729145298, 256876882, 844530403, 410593659, 760571764,
               518006029, 974333980, 269529072, 970106210, 165530362,
               35252495, 282738671, 831700334, 587315555, 124737579,
               252792339, 497998161, 410013399, 602723392, 720056443,
               599539097, 847978586, 399518980, 318884765, 411257802,
               927552927, 300956566, 54029907, 218857970, 526231503,
               393037212, 79018506, 338209123, 328394749, 256580131,
               817335908, 905193735, 60537897, 755610036, 5944998, 478360932)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    parts = []
    directions = _DIRS + (((-1, -1), (-1, 1), (1, -1), (1, 1))
                          if diagonal else ())
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, part = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    part.append(p)
        parts.append(np.array(part, dtype=int))
    return parts


def _bbox(points):
    r, c = points.min(axis=0)
    R, C = points.max(axis=0) + 1
    return int(r), int(c), int(R), int(C)


def _rooms(crop, wall):
    mask = np.pad(crop != wall, 1, constant_values=True)
    return [p - 1 for p in _components(mask)
            if np.all(p > 0) and np.all(p < np.array(mask.shape) - 1)]


def _room_graph(crop, wall, rooms):
    """Compartments are neighbors across one wall pixel in a compass direction."""
    ids = np.full(crop.shape, -1, dtype=int)
    for i, points in enumerate(rooms):
        ids[tuple(points.T)] = i
    graph = [dict() for _ in rooms]
    h, w = crop.shape
    for i, points in enumerate(rooms):
        for r, c in points:
            for dr, dc in _DIRS:
                R, C = r + 2 * dr, c + 2 * dc
                if 0 <= R < h and 0 <= C < w:
                    j = int(ids[R, C])
                    if j >= 0 and j != i and crop[r + dr, c + dc] == wall:
                        assert (dr, dc) not in graph[i] or graph[i][dr, dc] == j
                        graph[i][dr, dc] = j
    n = len(rooms)
    assert 3 <= n <= 8
    assert sum(map(len, graph)) == 2 * (n - 1), 'room adjacency must be a tree'
    coords = {0: (0, 0)}
    todo = [0]
    while todo:
        i = todo.pop()
        for delta, j in graph[i].items():
            assert graph[j].get((-delta[0], -delta[1])) == i
            expected = tuple(a + b for a, b in zip(coords[i], delta))
            if j in coords:
                assert coords[j] == expected
            else:
                coords[j] = expected
                todo.append(j)
    assert len(coords) == n and len(set(coords.values())) == n
    r0 = min(r for r, _ in coords.values())
    c0 = min(c for _, c in coords.values())
    coords = {i: (r - r0, c - c0) for i, (r, c) in coords.items()}
    neighbors = {coords[i]: tuple(sorted(coords[j] for j in graph[i].values()))
                 for i in range(n)}
    return coords, neighbors


def _tile(size, color, bg, hollow):
    a = np.full((size, size), color, dtype=np.int8)
    if hollow:
        a[1:-1, 1:-1] = bg
    return a


def _analyze(value):
    inp = _grid(value)
    bg = int(inp[0, 0])
    assert all(np.all(edge == bg) for edge in (inp[0], inp[-1], inp[:, 0], inp[:, -1]))
    figures, tokens = [], []
    for part in _components(inp != bg, diagonal=True):
        r, c, R, C = _bbox(part)
        crop = np.full((R - r, C - c), bg, dtype=np.int8)
        crop[part[:, 0] - r, part[:, 1] - c] = inp[tuple(part.T)]
        colors = set(map(int, crop.flat)) - {bg}
        s = R - r
        is_tile = len(colors) == 1 and s == C - c and s >= 3
        if is_tile:
            color = next(iter(colors))
            hollow = np.array_equal(crop, _tile(s, color, bg, True))
            solid = np.array_equal(crop, _tile(s, color, bg, False))
            is_tile = hollow or solid
        if is_tile:
            tokens.append(dict(top=r, left=c, color=color, hollow=hollow, size=s))
        else:
            figures.append((crop, (r, c, R, C), colors))
    assert len(figures) == 2
    target = [f for f in figures if len(f[2]) == 1]
    donor = [f for f in figures if len(f[2]) > 1]
    assert len(target) == len(donor) == 1
    target, tb, wall_set = target[0]
    donor, db, _ = donor[0]
    wall = next(iter(wall_set))
    dr, tr = _rooms(donor, wall), _rooms(target, wall)
    assert len(dr) == len(tr)
    dc, dg = _room_graph(donor, wall, dr)
    tc, tg = _room_graph(target, wall, tr)
    assert set(dc.values()) == set(tc.values()), 'different abstract maps'
    assert dg == tg, 'different compass edges'
    values, by_color = {}, {}
    for i, room in enumerate(dr):
        colors = np.unique(donor[tuple(room.T)])
        assert len(colors) == 1 and int(colors[0]) not in (bg, wall)
        color = int(colors[0])
        assert color not in by_color, 'room address colors must be unique'
        values[dc[i]] = color
        by_color[color] = dc[i]
    assert sum(map(len, dr)) == np.count_nonzero((donor != bg) & (donor != wall))
    assert tokens and len({t['size'] for t in tokens}) == 1
    assert 3 <= tokens[0]['size'] <= 4
    assert len({t['color'] for t in tokens}) == len(tokens)
    assert all(t['color'] in by_color for t in tokens)
    queues = []
    for hollow in (True, False):
        q = sorted((t for t in tokens if t['hollow'] == hollow),
                   key=lambda t: (t['top'], t['left']))
        assert 1 <= len(q) <= 4
        assert len({t['top'] for t in q}) == len(q), 'same-class top ties excluded'
        queues.append(q)
    assert all(np.all(target[tuple(room.T)] == bg) for room in tr)
    return dict(bg=bg, wall=wall, donor=donor, target=target, db=db, tb=tb,
                dr=dr, tr=tr, dc=dc, tc=tc, initial=values, by_color=by_color,
                tokens=tokens, queues=queues, graph=dg)


def _path(nodes, start, end, graph=None):
    parents = {start: None}
    todo = deque([start])
    while todo and end not in parents:
        node = todo.popleft()
        neighbors = graph[node] if graph is not None else [(node[0] + dr, node[1] + dc) for dr, dc in _DIRS]
        for other in neighbors:
            if other in nodes and other not in parents:
                parents[other] = node
                todo.append(other)
    assert end in parents
    route = [end]
    while route[-1] != start:
        route.append(parents[route[-1]])
    return route[::-1]


def _operations(hollow, solid):
    return [(hollow[i] if i < len(hollow) else None,
             solid[i] if i < len(solid) else None)
            for i in range(max(len(hollow), len(solid)))]


def _transport(initial, operations, bg, simultaneous=False, endpoints_only=False, graph=None):
    values = dict(initial)
    for receiver, source in operations:
        if receiver is None:  # A solid without a hollow has no destination.
            continue
        if source is None:    # A hollow without a solid receives background.
            values[receiver] = bg
            continue
        path = _path(initial, receiver, source, graph)
        old = initial if simultaneous else dict(values)
        if endpoints_only:
            values[receiver] = old[source]
        else:
            for a, b in zip(path, path[1:]):
                values[a] = old[b]
        values[source] = bg
    return values


def _render_answer(a, values, scan=False):
    out = a['target'].copy()
    if scan:
        order = [a['dc'][i] for i in range(len(a['dr']))]
        room_values = [values[node] for node in order]
    else:
        room_values = [values[a['tc'][i]] for i in range(len(a['tr']))]
    for room, color in zip(a['tr'], room_values):
        out[tuple(room.T)] = color
    return out


def _predict(a, variant='correct'):
    h, s = [list(q) for q in a['queues']]
    if variant in ('horizontal', 'horizontal_reverse'):
        h, s = [sorted(q, key=lambda t: (t['left'], t['top']),
                       reverse=variant == 'horizontal_reverse') for q in (h, s)]
    elif variant == 'reverse':
        h, s = h[::-1], s[::-1]
    elif variant == 'swap_roles':
        h, s = s, h
    if variant == 'nearest':
        remaining, ops = list(s), []
        for t in h:
            nearest = min(remaining, key=lambda v: (abs(t['top'] - v['top']) +
                          abs(t['left'] - v['left']), v['top'], v['left'])) if remaining else None
            ops.append((t, nearest))
            if nearest is not None:
                remaining.remove(nearest)
    elif variant == 'greedy':
        rows = [[None, None]]
        for t in sorted(a['tokens'], key=lambda t: (t['top'], t['left'])):
            col = 0 if t['hollow'] else 1
            if rows[-1][col] is not None:
                rows.append([None, None])
            rows[-1][col] = t
        ops = rows
    else:
        ops = _operations(h, s)
    operations = [(a['by_color'][x['color']] if x else None,
                   a['by_color'][y['color']] if y else None) for x, y in ops]
    if variant == 'reverse_operations':
        operations.reverse()
    values = (a['initial'] if variant == 'direct_transfer' else
              _transport(a['initial'], operations, a['bg'],
                         simultaneous=variant == 'simultaneous',
                         endpoints_only=variant == 'endpoints_only', graph=a['graph']))
    if variant in ('restore_unmatched_solids', 'clear_unmatched_solids'):
        for receiver, source in operations:
            if receiver is None and source is not None:
                values[source] = a['initial'][source] if variant == 'restore_unmatched_solids' else a['bg']
    return _render_answer(a, values, scan=variant == 'scan_correspondence')


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve without a seed, generated labels, training pairs, or mutable state."""
    return _predict(_analyze(input_grid))


def _tree(rng, n, force_top_pair=False):
    for _ in range(100):
        nodes = {(0, 0)}
        while len(nodes) < n:
            choices = set()
            for r, c in nodes:
                for dr, dc in _DIRS:
                    p = r + dr, c + dc
                    if p in nodes:
                        continue
                    if sum((p[0] + a, p[1] + b) in nodes for a, b in _DIRS) != 1:
                        continue
                    trial = nodes | {p}
                    if (max(r for r, _ in trial) - min(r for r, _ in trial) < 5 and
                            max(c for _, c in trial) - min(c for _, c in trial) < 5):
                        choices.add(p)
            if not choices:
                break
            choices = sorted(choices)
            nodes.add(choices[int(rng.integers(len(choices)))])
        r0, c0 = min(r for r, _ in nodes), min(c for _, c in nodes)
        nodes = {(r - r0, c - c0) for r, c in nodes}
        if len(nodes) == n and (not force_top_pair or sum(r == 0 for r, _ in nodes) >= 2):
            return sorted(nodes)
    # A bent chain is a valid tree for every supported room count.
    length = min(4, n)
    return [(0, c) for c in range(length)] + [(r, length - 1) for r in range(1, n - length + 1)]


def _geometry(rng, nodes, maximum=3, scan_change=False, plain=False, fixed=None):
    rows = max(r for r, _ in nodes) + 1
    cols = max(c for _, c in nodes) + 1
    if fixed is None:
        heights = rng.integers(1, maximum + 1, size=rows)
        widths = rng.integers(1, maximum + 1, size=cols)
    else:
        heights, widths = [np.array(v, dtype=int) for v in fixed]
    if maximum == 4:
        # Geometry curricula have an actual held-out span, not merely a
        # chance of drawing one. Keep this room intact below.
        heights[nodes[0][0]] = widths[nodes[0][1]] = 4
    if scan_change:
        heights[0] = max(2, int(heights[0]))
    ys, xs = np.r_[0, np.cumsum(heights + 1)], np.r_[0, np.cumsum(widths + 1)]
    labels = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    for i, (r, c) in enumerate(nodes):
        labels[ys[r] + 1:ys[r + 1], xs[c] + 1:xs[c + 1]] = i
    shifted_id = next(i for i, (r, _) in enumerate(nodes) if r == 0) if scan_change else None
    if not plain:
        for i, (r, c) in enumerate(nodes):
            if i == shifted_id or (maximum == 4 and i == 0):
                continue
            points = np.argwhere(labels == i)
            if len(points) > 1 and rng.random() < 0.45:
                R, C, RR, CC = _bbox(points)
                corner = ((R, C), (R, CC - 1), (RR - 1, C), (RR - 1, CC - 1))[int(rng.integers(4))]
                candidate = labels == i
                candidate[corner] = False
                if len(_components(candidate)) == 1:
                    labels[corner] = -1
    if scan_change:
        # Shift the first top-row room below its same-row neighbors. The
        # adjacency graph is subsequently checked, never assumed unchanged.
        i = shifted_id
        points = np.argwhere(labels == i)
        top = int(points[:, 0].min())
        labels[top, labels[top] == i] = -1
        assert np.any(labels == i)
    occupied = labels >= 0
    near = np.zeros(labels.shape, dtype=bool)
    padded = np.pad(occupied, 1)
    for dr in range(3):
        for dc in range(3):
            near |= padded[dr:dr + labels.shape[0], dc:dc + labels.shape[1]]
    labels[(labels < 0) & near] = -2
    r, c, R, C = _bbox(np.argwhere(labels != -1))
    return labels[r:R, c:C]


def _paint(labels, bg, wall, colors=None):
    out = np.full(labels.shape, bg, dtype=np.int8)
    out[labels == -2] = wall
    if colors is not None:
        for i, color in enumerate(colors):
            out[labels == i] = color
    return out


def _separated(rect, other):
    r, c, h, w = rect
    R, C, H, W = other
    return r + h < R or R + H < r or c + w < C or C + W < c


def _embed(rng, donor, target, token_colors, kinds, bg, size):
    """Rectangle separation leaves a visible gap; free sites vary the layout."""
    figure_shapes = [donor.shape, target.shape]
    canvas_h = int(rng.integers(max(22, donor.shape[0] + 2, target.shape[0] + 2), 31))
    canvas_w = int(rng.integers(max(22, donor.shape[1] + 2, target.shape[1] + 2), 31))
    # Side/stack/offset layouts are sampled from free translations. A compact
    # corner placement is tried regularly to retain enough space for markers.
    for attempt in range(40):
        boxes = []
        for j, (h, w) in enumerate(figure_shapes):
            if h > canvas_h - 2 or w > canvas_w - 2:
                return None
            options = [(r, c, h, w) for r in range(1, canvas_h - h)
                       for c in range(1, canvas_w - w)
                       if all(_separated((r, c, h, w), old) for old in boxes)]
            if not options:
                break
            if attempt % 4 == 0:
                edges = [b for b in options if b[0] in (1, canvas_h - 1 - h) and b[1] in (1, canvas_w - 1 - w)]
                if edges:
                    options = edges
            boxes.append(options[int(rng.integers(len(options)))])
        if len(boxes) != 2:
            continue
        slots = []
        for hollow in kinds:
            options = [(r, c, size, size) for r in range(1, canvas_h - size)
                       for c in range(1, canvas_w - size)
                       if all(_separated((r, c, size, size), old) for old in boxes + slots)
                       and all(r != old[0] for old, k in zip(slots, kinds) if k == hollow)]
            if not options:
                break
            slots.append(options[int(rng.integers(len(options)))])
        if len(slots) != len(kinds):
            continue
        # Colors are assigned to the class queues after placement, so the
        # independently sorted input ranks encode the requested operations.
        assigned = [None] * len(slots)
        for hollow in (True, False):
            indices = sorted((i for i, k in enumerate(kinds) if k == hollow),
                             key=lambda i: slots[i][:2])
            colors = [color for color, k in zip(token_colors, kinds) if k == hollow]
            for i, color in zip(indices, colors):
                assigned[i] = color
        canvas = np.full((canvas_h, canvas_w), bg, dtype=np.int8)
        for grid, (r, c, h, w) in zip((donor, target), boxes):
            canvas[r:r + h, c:c + w] = grid
        for (r, c, _, _), hollow, color in zip(slots, kinds, assigned):
            canvas[r:r + size, c:c + size] = _tile(size, color, bg, hollow)
        # Random exterior padding, still at least one full background pixel.
        used = np.argwhere(canvas != bg)
        r, c, R, C = _bbox(used)
        top = int(rng.integers(r))
        left = int(rng.integers(c))
        bottom = int(rng.integers(R + 1, canvas_h + 1))
        right = int(rng.integers(C + 1, canvas_w + 1))
        return canvas[top:bottom, left:right].copy()
    return None


_ALTERNATIVES = ('horizontal', 'horizontal_reverse', 'reverse', 'swap_roles',
                 'nearest', 'greedy', 'reverse_operations', 'simultaneous',
                 'endpoints_only', 'scan_correspondence', 'direct_transfer',
                 'restore_unmatched_solids', 'clear_unmatched_solids')


def _required_witnesses(h, s):
    return set(_ALTERNATIVES) - ({'restore_unmatched_solids', 'clear_unmatched_solids'} if h > s else set())


def _witnesses(a):
    correct = _predict(a)
    return {name: not np.array_equal(correct, _predict(a, name)) for name in _ALTERNATIVES}


def _routes(a):
    h, s = a['queues']
    operations = [(a['by_color'][x['color']] if x else None,
                   a['by_color'][y['color']] if y else None)
                  for x, y in _operations(h, s)]
    paths = [_path(a['initial'], x, y, a['graph']) for x, y in operations if x is not None and y is not None]
    return operations, paths


def _make_pair(rng, n, h, s, demo_color=None, demo_hollow=True,
               witness=False, maximum=3):
    kinds = [True] * h + [False] * s
    required = _required_witnesses(h, s) if witness else set()
    for attempt in range(384):
        nodes = _tree(rng, n, force_top_pair=witness)
        d = _geometry(rng, nodes, 3, plain=witness)
        t = _geometry(rng, nodes, maximum, scan_change=witness)
        palette = list(map(int, rng.permutation(10)))
        if demo_color is not None:
            palette.remove(demo_color)
        bg, wall = palette[:2]
        fill_pool = palette[2:]
        if demo_color is not None:
            fill_pool.insert(0, demo_color)
        colors = fill_pool[:n]
        endpoint_ids = list(map(int, rng.choice(n, size=h + s, replace=False)))
        if demo_color is not None:
            forced = endpoint_ids[0 if demo_hollow else h]
            colors[0], colors[forced] = colors[forced], colors[0]
        endpoints = [nodes[i] for i in endpoint_ids]
        initial = dict(zip(nodes, colors))
        ops = _operations(endpoints[:h], endpoints[h:])
        values = _transport(initial, ops, bg)
        donor, target = _paint(d, bg, wall, colors), _paint(t, bg, wall)
        expected = _paint(t, bg, wall, [values[node] for node in nodes])
        size = 3 if witness else int(rng.integers(3, 5))
        inp = _embed(rng, donor, target, [colors[i] for i in endpoint_ids], kinds, bg, size)
        if inp is None:
            continue
        try:
            a = _analyze(inp)
            assert a['initial'] == initial
            assert np.array_equal(_predict(a), expected)
            operations, paths = _routes(a)
            if witness:
                observed = _witnesses(a)
                assert all(observed[k] for k in required)
                assert max(map(len, paths)) >= 3
                used = set().union(*map(set, paths))
                extra = endpoints[min(h, s)] if h > s else endpoints[h + min(h, s)]
                if h > s:
                    assert extra not in used
                else:
                    # An unmatched solid can be changed by another route;
                    # ignoring its token must not reset or clear that room.
                    assert extra in used and values[extra] not in (bg, initial[extra])
                    assert set(initial) - used
            return dict(input=inp, output=expected)
        except AssertionError:
            continue
    return _fallback_pair(rng, n, h, s, demo_color, demo_hollow, witness, maximum)


def _fallback_pair(rng, n, h, s, demo_color, demo_hollow, witness, maximum):
    """Explicit fitting constructions make legal requests total and bounded.

    The witness construction has two overlapping routes, a changed scan order,
    and a diagnostic null partner. The ordinary construction is a bent chain.
    These are sampled only after random construction exhausts its budget.
    """
    if witness:
        assert (h, s) in ((3, 2), (2, 3))
        nodes = [(0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3)]
        n = 6
        d = _geometry(rng, nodes, plain=True, fixed=([3, 3], [2, 2, 2, 2]))
        t = _geometry(rng, nodes, plain=True, scan_change=True,
                      fixed=([3, 2], [1, 1, 2, 1]))
        h_ids = [0, 4] + ([1] if h == 3 else [])
        s_ids = [5, 3] + ([2] if s == 3 else [])
        slots_h = [(1, 17), (7, 13)] + ([(26, 24)] if h == 3 else [])
        slots_s = [(13, 4), (22, 17)] + ([(26, 24)] if s == 3 else [])
        placements = [(t, 1, 1), (d, 20, 1)]
    else:
        length = min(4, n)
        nodes = [(0, c) for c in range(length)] + [(r, length - 1) for r in range(1, n - length + 1)]
        nodes.sort()
        rows, cols = max(r for r, _ in nodes) + 1, max(c for _, c in nodes) + 1
        d = _geometry(rng, nodes, plain=True, fixed=([1] * rows, [1] * cols))
        t = _geometry(rng, nodes, maximum, plain=True, fixed=([1] * rows, [2] * cols))
        ids = list(map(int, rng.permutation(n)))
        h_ids, s_ids = ids[:h], ids[h:h + s]
        slots_h = [(14 + i, 1 + 4 * i) for i in range(h)]
        slots_s = [(21 + i, 1 + 4 * i) for i in range(s)]
        placements = [(d, 1, 1), (t, 1, d.shape[1] + 2)]
    palette = list(map(int, rng.permutation(10)))
    if demo_color is not None:
        palette.remove(demo_color)
    bg, wall = palette[:2]
    colors = palette[2:]
    if demo_color is not None:
        colors.insert(0, demo_color)
        forced = h_ids[0] if demo_hollow else s_ids[0]
        colors[0], colors[forced] = colors[forced], colors[0]
    colors = colors[:n]
    initial = dict(zip(nodes, colors))
    values = _transport(initial, _operations([nodes[i] for i in h_ids], [nodes[i] for i in s_ids]), bg)
    inp = np.full((30, 30), bg, dtype=np.int8)
    for labels, r, c in placements:
        grid = _paint(labels, bg, wall, colors if labels is d else None)
        inp[r:r + grid.shape[0], c:c + grid.shape[1]] = grid
    for hollow, ids, slots in ((True, h_ids, slots_h), (False, s_ids, slots_s)):
        for i, (r, c) in zip(ids, slots):
            inp[r:r + 3, c:c + 3] = _tile(3, colors[i], bg, hollow)
    expected = _paint(t, bg, wall, [values[node] for node in nodes])
    a = _analyze(inp)
    assert a['initial'] == initial and np.array_equal(_predict(a), expected)
    if witness:
        observed = _witnesses(a)
        assert all(observed[k] for k in _required_witnesses(h, s))
    return dict(input=inp, output=expected)


def _count(value, lo, hi, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(name + ' is outside the legal range')
    return value


def _distinct_recolor(pair, seen):
    a = _analyze(pair['input'])
    old = [a['bg'], a['wall']] + [c for c in range(10) if c not in (a['bg'], a['wall'])]
    # Fewer than 14 inputs exist; the 90 distinct ordered role pairs cannot
    # all collide. This bijection preserves every geometric witness.
    for bg in range(10):
        for wall in range(10):
            if bg == wall:
                continue
            new = [bg, wall] + [c for c in range(10) if c not in (bg, wall)]
            lookup = np.empty(10, dtype=np.int8)
            lookup[old] = new
            candidate = {k: lookup[v] for k, v in pair.items()}
            if _key(candidate['input']) not in seen:
                return candidate
    raise AssertionError('ordered role recolorings unexpectedly exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    shift = int(rng.integers(3))  # More routes, more rooms, or wider local deformation.
    demo_color = int(rng.integers(10))
    first_hollow_tail = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            witness = split == 'train' and index < 2
            if witness:
                h, s = (3, 2) if first_hollow_tail == (index == 0) else (2, 3)
                n = 6 if shift == 1 else int(rng.integers(6, 8))
            else:
                maximum_rooms = 6 if split == 'train' and shift == 1 else 7
                if split == 'test' and shift == 0:
                    h, s = ((3, 3), (4, 3), (3, 4))[int(rng.integers(3))]
                    n = int(rng.integers(max(h + s, 7), 9))
                else:
                    counts = ((1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (2, 3),
                              (3, 1), (1, 3), (3, 3))
                    options = [(H, S) for H, S in counts if H + S <= maximum_rooms]
                    if split == 'train' and shift == 0:
                        options = [(H, S) for H, S in options if min(H, S) <= 2]
                    h, s = options[int(rng.integers(len(options)))]
                    n = 8 if split == 'test' and shift == 1 else int(rng.integers(max(3, h + s), maximum_rooms + 1))
            maximum = 4 if split == 'test' and shift == 2 else 3
            for _ in range(20):
                pair = _make_pair(rng, n, h, s, demo_color if witness else None,
                                  demo_hollow=index == 0, witness=witness, maximum=maximum)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                pair = _distinct_recolor(pair, seen)
                key = _key(pair['input'])
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            a = _analyze(inp)
            expected = _predict(a)
            assert np.array_equal(out, expected), 'wrong routed colors or crop'
            assert np.array_equal(out == a['wall'], a['target'] == a['wall'])
            interior = np.zeros(out.shape, dtype=bool)
            for room in a['tr']:
                interior[tuple(room.T)] = True
                assert len(np.unique(out[tuple(room.T)])) == 1
            assert np.array_equal(out[~interior], a['target'][~interior]), 'exterior changed'
            assert np.any(out == a['bg']) and np.any(out != a['bg'])
            if split == 'train':
                training.append(a)
    # These are visible evidence checks, not sampler metadata assertions.
    witnesses = [_witnesses(a) for a in training]
    for name in _ALTERNATIVES:
        assert any(w[name] for w in witnesses), 'training does not distinguish ' + name
    imbalances = [len(a['queues'][0]) - len(a['queues'][1]) for a in training]
    assert min(imbalances) < 0 < max(imbalances), 'both null partner cases must be taught'
    roles = {}
    null_hollow, null_solid, off_route_transfer = False, False, False
    for a in training:
        for t in a['tokens']:
            roles.setdefault(t['color'], set()).add(t['hollow'])
        ops, paths = _routes(a)
        used = set().union(*map(set, paths))
        final = _transport(a['initial'], ops, a['bg'], graph=a['graph'])
        null_hollow |= any(x is not None and y is None and x not in used and final[x] == a['bg']
                           for x, y in ops)
        null_solid |= any(x is None and y is not None and y in used and final[y] not in (a['bg'], a['initial'][y])
                          for x, y in ops)
        cleared = {x for x, y in ops if x is not None and y is None}
        off_route_transfer |= any(k not in used and k not in cleared and final[k] == v
                                  for k, v in a['initial'].items())
    assert any(len(v) == 2 for v in roles.values()), 'missing geometry/color role switch'
    assert null_hollow and null_solid, 'null partners must visibly teach both cases'
    assert off_route_transfer, 'off-route room transfer must be demonstrated'


def self_test():
    signatures, structures, shapes, graphs, counts, roles = set(), set(), set(), set(), set(), set()
    path_lengths, overlaps, area_signs, nonrectangular = set(), 0, set(), 0
    for seed in _HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for split in ('train', 'test') for p in puzzle[split]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair['input']), pair['output'])
                a = _analyze(pair['input'])
                shapes.add((pair['input'].shape, pair['output'].shape))
                graphs.add(tuple(sorted(a['initial'])))
                counts.add(tuple(map(len, a['queues'])))
                roles.add((a['bg'], a['wall']))
                norm = {}
                structures.add(tuple(tuple(norm.setdefault(int(c), len(norm)) for c in row)
                                     for row in pair['input']))
                _, paths = _routes(a)
                path_lengths.update(map(len, paths))
                overlaps += int(any(set(x) & set(y) for i, x in enumerate(paths) for y in paths[i + 1:]))
                ds = {a['dc'][i]: len(room) for i, room in enumerate(a['dr'])}
                for i, room in enumerate(a['tr']):
                    area_signs.add(int(np.sign(len(room) - ds[a['tc'][i]])))
                    r, c, R, C = _bbox(room)
                    nonrectangular += int(len(room) < (R - r) * (C - c))
    assert len(signatures) == 42
    assert len(structures) >= 180 and len(shapes) >= 40 and len(graphs) >= 25
    assert len(counts) >= 9 and len(roles) >= 40
    assert len(path_lengths) >= 4 and max(path_lengths) >= 6
    assert overlaps >= 42 and area_signs == {-1, 0, 1} and nonrectangular >= 20
    a, b = generate_puzzle(177, 2, 1), generate_puzzle(177, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for split in a for x, y in zip(a[split], b[split]) for k in x)
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            validate_generated_puzzle(generate_puzzle(11, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    rng = np.random.default_rng(711)
    for h, s in ((3, 2), (2, 3)):
        for demo_hollow in (True, False):
            pair = _fallback_pair(rng, 6, h, s, 0, demo_hollow, True, 3)
            observed = _witnesses(_analyze(pair['input']))
            assert all(observed[k] for k in _required_witnesses(h, s))
    for n in range(3, 9):
        for h in range(1, 5):
            for s in range(1, 5):
                if h + s <= n:
                    for maximum in (3, 4):
                        pair = _fallback_pair(rng, n, h, s, None, True, False, maximum)
                        assert np.array_equal(solve(pair['input']), pair['output'])
    pair = a['test'][0]
    replacement = _distinct_recolor(pair, {_key(pair['input'])})
    assert _key(replacement['input']) != _key(pair['input'])
    assert np.array_equal(solve(replacement['input']), replacement['output'])
    perm = rng.permutation(10)
    assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
    # A compass tree can occupy a square of coarse coordinates while one
    # side has a thicker wall and is not an edge. Use the visible graph, not
    # presumed adjacency of the reconstructed coordinate positions.
    donor = np.array([[int(c) for c in row] for row in
                      ('1111111', '1221331', '1111331', '1111111',
                       '1551441', '1551441', '1111111')], dtype=np.int8)
    target = np.array([[int(c) for c in row] for row in
                       ('1111111', '1001001', '1111001', '1111111',
                        '1010011', '1110011', '1111111')], dtype=np.int8)
    expected = np.array([[int(c) for c in row] for row in
                         ('1111111', '1331441', '1111441', '1111111',
                          '1015511', '1115511', '1111111')], dtype=np.int8)
    query = np.zeros((14, 21), dtype=np.int8)
    query[1:8, 1:8], query[1:8, 11:18] = donor, target
    query[9:12, 2:5] = _tile(3, 2, 0, True)
    query[9:12, 16:19] = _tile(3, 5, 0, False)
    assert np.array_equal(solve(query), expected)
    damaged = {split: [{k: np.array(v, copy=True) for k, v in p.items()} for p in a[split]] for split in a}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt framework accepted')


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(',', ':')))
