"""A waypoint-driven printer: relative stencil rotation and opaque layering.

Only Python's standard library and NumPy are used. Solving reads one input;
the construction recipe, RNG, and training examples are never solver inputs.
"""
from collections import Counter

import numpy as np


_DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
_KEY_ARROW = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 0, 0]], bool)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _fingerprint(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _rotate_offset(point, clockwise):
    r, c = point
    for _ in range(clockwise % 4):
        r, c = c, -r
    return r, c


def _courier_cells(mask, tip, direction):
    return {(tip[0] + r, tip[1] + c)
            for r, c in (_rotate_offset(p, direction - 1) for p in mask)}


def _identify_courier(cells):
    matches = []
    for tip in sorted(cells):
        for d in range(4):
            offsets = {_rotate_offset((r - tip[0], c - tip[1]), 1 - d)
                       for r, c in cells}
            if any(c > 0 for r, c in offsets):
                continue
            if {p for p in offsets if p[1] == 0} != {(0, 0)}:
                continue
            if not any(r != 0 for r, c in offsets):
                continue
            if {(-r, c) for r, c in offsets} == offsets:
                matches.append((tuple(sorted(offsets)), tip, d))
    assert len(matches) == 1, "courier must have one geometric projecting tip"
    return matches[0]


def _direction(a, b):
    dr, dc = b[0] - a[0], b[1] - a[1]
    assert (dr == 0) != (dc == 0), "distinct orthogonal references required"
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _route(tip, stations):
    remaining, points, directions = set(stations), [tip], []
    while remaining:
        rays = {}
        for p in sorted(remaining):
            if p[0] == tip[0] or p[1] == tip[1]:
                rays.setdefault(_direction(tip, p), []).append(p)
        assert len(rays) == 1, "need one occupied ray, with no junction tie policy"
        d, candidates = next(iter(rays.items()))
        p = min(candidates, key=lambda q: abs(q[0] - tip[0]) + abs(q[1] - tip[1]))
        remaining.remove(p)
        points.append(p)
        directions.append(d)
        tip = p
    return points, directions


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        cells, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _box(cells):
    a = np.asarray(list(cells), dtype=int)
    assert a.size
    return (int(a[:, 0].min()), int(a[:, 0].max()) + 1,
            int(a[:, 1].min()), int(a[:, 1].max()) + 1)


def _frame(grid):
    candidates = []
    for color in np.unique(grid):
        cells = np.argwhere(grid == color)
        r, e, c, f = _box(cells)
        h, w = e - r, f - c
        if h < 9 or w < 9 or len(cells) != 2 * h + 2 * w - 4:
            continue
        if (np.all(grid[r, c:f] == color) and np.all(grid[e - 1, c:f] == color)
                and np.all(grid[r:e, c] == color) and np.all(grid[r:e, f - 1] == color)):
            candidates.append(((r, e, c, f), int(color)))
    assert len(candidates) == 1, "one exclusive-color rectangular frame required"
    return candidates[0]


def _key_tile(source, target, direction):
    return np.where(np.rot90(_KEY_ARROW, 1 - direction), source, target).astype(np.int8)


def _inside(cells, shape):
    return all(1 <= r < shape[0] - 1 and 1 <= c < shape[1] - 1 for r, c in cells)


def _overlaps(points, n):
    return sum(abs(a[0] - b[0]) < n and abs(a[1] - b[1]) < n
               for i, a in enumerate(points[1:]) for b in points[i + 2:])


def _analyze(input_grid):
    grid = _grid(input_grid)
    frame_box, frame_color = _frame(grid)
    fr, fe, fc, ff = frame_box
    corners = [int(grid[fr + 1, fc + 1]), int(grid[fr + 1, ff - 2]),
               int(grid[fe - 2, fc + 1]), int(grid[fe - 2, ff - 2])]
    assert len(set(corners)) == 1 and corners[0] != frame_color
    bg = corners[0]
    outside = grid != bg
    outside[fr:fe, fc:ff] = False
    key_cells = np.zeros(grid.shape, bool)
    mapping, refs, key_boxes = {}, {}, {}
    for cells in _components(outside):
        colors, counts = np.unique([grid[r, c] for r, c in cells], return_counts=True)
        if len(colors) == 1:
            continue
        assert len(colors) == 2 and set(map(int, counts)) == {5, 7}
        r, e, c, f = _box(cells)
        assert len(cells) == (e - r) * (f - c)
        tile = grid[r:e, c:f]
        source = int(colors[np.flatnonzero(counts == 7)[0]])
        target = int(colors[np.flatnonzero(counts == 5)[0]])
        matches = [d for d in range(4) if np.array_equal(tile, _key_tile(source, target, d))]
        assert len(matches) == 1 and source not in mapping
        assert bg not in (source, target) and frame_color not in (source, target)
        mapping[source], refs[source] = target, matches[0]
        key_boxes[source] = (r, e, c, f)
        key_cells[r:e, c:f] = True
    assert 1 <= len(mapping) <= 4
    remaining = outside & ~key_cells
    assert set(map(int, np.unique(grid[remaining]))) == set(mapping)
    stencils, stencil_boxes = {}, {}
    for source in sorted(mapping):
        cells = np.argwhere(remaining & (grid == source))
        r, e, c, f = _box(cells)
        n = e - r
        assert n == f - c and n in (3, 5, 7)
        tile = grid[r:e, c:f]
        assert np.all(np.isin(tile, [bg, source]))
        mask = tile == source
        assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
        stencils[source], stencil_boxes[source] = mask.copy(), (r, e, c, f)
    sizes = {m.shape[0] for m in stencils.values()}
    assert len(sizes) == 1
    n = sizes.pop()
    boxes = list(stencil_boxes.values()) + list(key_boxes.values())
    for i, (r, e, c, f) in enumerate(boxes):
        assert e < fr or fe < r or f < fc or ff < c, "clues need a blank gap to the frame"
        for rr, ee, cc, fff in boxes[i + 1:]:
            assert e < rr or ee < r or f < cc or fff < c, "complete clue boxes need a gap"
    board = grid[fr:fe, fc:ff]
    inside_colors = set(map(int, np.unique(board[1:-1, 1:-1])))
    inks = inside_colors - {bg} - set(mapping)
    assert len(inks) == 1
    ink = inks.pop()
    assert ink not in set(mapping.values()) | {bg, frame_color}
    courier = set(map(tuple, np.argwhere(board == ink)))
    mask, tip, initial = _identify_courier(courier)
    stations = {}
    for source in sorted(mapping):
        cells = list(map(tuple, np.argwhere(board == source)))
        assert cells, "every legend source must occur on the route"
        stations.update({p: source for p in cells})
    assert 3 <= len(stations) <= 9
    assert all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 2
               for i, a in enumerate(stations) for b in list(stations)[i + 1:])
    points, directions = _route(tip, stations)
    assert directions[0] == initial
    radius = n // 2
    for i, p in enumerate(points):
        d = initial if i == 0 else directions[i - 1]
        cells = _courier_cells(mask, p, d)
        assert _inside(cells, board.shape)
        assert not (cells & set(points[i + 1:])), "courier cannot cover future commands"
        if i:
            assert (1 <= p[0] - radius and p[0] + radius < board.shape[0] - 1
                    and 1 <= p[1] - radius and p[1] + radius < board.shape[1] - 1)
    initial_board = np.full(board.shape, bg, np.int8)
    _draw_frame(initial_board, 0, 0, *board.shape, frame_color)
    for r, c in courier:
        initial_board[r, c] = ink
    for p, source in stations.items():
        initial_board[p] = source
    assert np.array_equal(initial_board, board), "unexpected board content"
    return dict(bg=bg, frame=frame_color, ink=ink, shape=board.shape, frame_box=frame_box,
                mask=mask, initial=initial, points=points, directions=directions,
                stations=stations, stencils=stencils, mapping=mapping, refs=refs, n=n,
                key_boxes=key_boxes, stencil_boxes=stencil_boxes,
                overlap=_overlaps(points, n))


def _draw_frame(grid, r, c, h, w, color):
    grid[r, c:c + w] = grid[r + h - 1, c:c + w] = color
    grid[r:r + h, c] = grid[r:r + h, c + w - 1] = color


def _print_board(a, alternative=None):
    """Render a decoded recipe; alternatives are used only for witness checks."""
    out = np.full(a['shape'], a['bg'], np.int8)
    _draw_frame(out, 0, 0, *a['shape'], a['frame'])
    points, ds, n = a['points'], a['directions'], a['n']
    indices = list(range(1, len(points)))
    if alternative == 'reverse':
        indices.reverse()
    if alternative == 'last_only':
        indices = indices[-1:]
    for i in indices:
        r, c = points[i]
        source = a['stations'][points[i]]
        paint = a['mapping'][source]
        if alternative in ('source_paint', 'no_B'):
            paint = source
        elif alternative == 'twice_mapping':
            paint = a['mapping'].get(paint, paint)
        mask = a['stencils'][source]
        d = ds[i - 1]
        if alternative == 'displayed':
            d = a['refs'][source]
        elif alternative == 'fixed_facing':
            d = a['initial']
        elif alternative == 'outgoing' and i < len(ds):
            d = ds[i]
        turned = np.rot90(mask, a['refs'][source] - d)
        if alternative in ('solid', 'no_B'):
            turned = np.ones_like(mask)
        tile = out[r - n // 2:r + n // 2 + 1, c - n // 2:c + n // 2 + 1]
        if alternative != 'transparent':
            tile[:] = a['bg']
        tile[turned] = paint
    # Counterfactuals keep the real final courier: a shortcut must change visible
    # printing pixels, not merely fail to fit a differently oriented courier.
    final = ds[-1]
    for p in _courier_cells(a['mask'], points[-1], final):
        out[p] = a['ink']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the framed printed board entirely from one input grid."""
    return _print_board(_analyze(input_grid))


_WITNESSES = ('displayed', 'solid', 'source_paint', 'no_B', 'transparent',
              'reverse', 'outgoing', 'twice_mapping', 'last_only', 'fixed_facing')


def _witnesses(a):
    expected = _print_board(a)
    return {kind: not np.array_equal(expected, _print_board(a, kind)) for kind in _WITNESSES}


def _arrow(span, extension, filled, tail):
    cells = {(0, -i) for i in range(extension)}
    for depth in range(1, span + 1):
        rows = range(-depth, depth + 1) if filled else (-depth, depth)
        cells.update((r, -extension + 1 - depth) for r in rows)
    for depth in range(tail):
        cells.update((r, -extension - span - depth) for r in (-span, span))
    return tuple(sorted(cells))


_COURIERS = tuple(dict.fromkeys(_arrow(s, e, f, t)
                              for s in (1, 2) for e in (1, 2)
                              for f in (False, True) for t in (0, 1)))


def _asymmetric(mask):
    return all(not np.array_equal(mask, np.rot90(mask, d)) for d in (1, 2, 3))


def _stencil(rng, n, informative=False):
    y, x = np.indices((n, n))
    for _ in range(80):
        kind = 0 if informative else int(rng.integers(9))
        if kind == 0:
            m = rng.random((n, n)) < rng.uniform(.2, .85)
        elif kind == 1:
            m = (x == 0) | (x == n - 1) | (y == 0) | (y == n - 1)
        elif kind == 2:
            m = (x == n // 2) | (y == n // 2)
        elif kind == 3:
            m = (x + y) % 2 == 0
        elif kind == 4:
            m = x == y
        elif kind == 5:
            m = (y % 2 == 0)
        elif kind == 6:
            m = (x == 0) | (y == n - 1)
        elif kind == 7:
            m = rng.random((n, n)) < .9
        else:
            m = np.ones((n, n), bool)
        for edge in (m[0], m[-1], m[:, 0], m[:, -1]):
            if not edge.any():
                edge[int(rng.integers(n))] = True
        if not informative or (_asymmetric(m) and not m.all()):
            return np.rot90(m, int(rng.integers(4))).copy()
    # A square whose four edges and unequal limbs remain readable at every size.
    m = np.zeros((n, n), bool)
    m[0, :] = m[:, 0] = True
    m[-1, 1] = True
    return m


def _sample_mapping(rng, palette, k, teach):
    sources = list(map(int, palette[:k]))
    external = list(map(int, palette[k:]))
    order = list(map(int, rng.permutation(sources)))
    kind = 'chain' if teach else str(rng.choice(['disjoint', 'chain', 'cycle', 'many', 'mixed']))
    if kind == 'chain':
        mapping = {s: order[i + 1] if i + 1 < k else external[0]
                   for i, s in enumerate(order)}
    elif kind == 'cycle' and k > 1:
        mapping = {s: order[(i + 1) % k] for i, s in enumerate(order)}
    elif kind == 'many':
        target = int(rng.choice(palette))
        mapping = {s: external[0] if s == target else target for s in sources}
    elif kind == 'mixed':
        mapping = {s: int(rng.choice([t for t in palette if t != s])) for s in sources}
    else:
        mapping = {s: int(rng.choice(external)) for s in sources}
    return mapping


def _route_options(points, footprints, mask, shape, n):
    radius = n // 2
    options = []
    r, c = points[-1]
    for d, (dr, dc) in enumerate(_DIRS):
        initial = _courier_cells(mask, points[0], d) if len(points) == 1 else None
        if initial is not None and not _inside(initial, shape):
            continue
        old_footprints = [initial] if initial is not None else footprints
        for distance in range(2, max(shape)):
            p = (r + dr * distance, c + dc * distance)
            if not (radius + 1 <= p[0] < shape[0] - radius - 1
                    and radius + 1 <= p[1] < shape[1] - radius - 1):
                continue
            if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) < 2 for q in points):
                continue
            new = _courier_cells(mask, p, d)
            if not _inside(new, shape) or any(p in cells for cells in old_footprints):
                continue
            valid = True
            for a, b in zip(points[:-1], points[1:]):
                if a[0] == p[0] or a[1] == p[1]:
                    cross = (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])
                    beyond = (p[0] - b[0]) * (b[0] - a[0]) + (p[1] - b[1]) * (b[1] - a[1])
                    if cross or beyond <= 0:
                        valid = False
                        break
            if valid:
                options.append((d, distance, p, new))
    return options


def _grow_route(rng, shape, mask, n, count, teach, sparse=False):
    radius = n // 2
    for _ in range(20):
        points = [(int(rng.integers(radius + 1, shape[0] - radius - 1)),
                   int(rng.integers(radius + 1, shape[1] - radius - 1)))]
        ds, footprints = [], []
        while len(ds) < count:
            options = _route_options(points, footprints, mask, shape, n)
            if not options:
                break
            weights = np.array([(dist ** .7 if sparse else 1 / dist ** .7)
                                * (1.4 if ds and d == ds[-1] else 1)
                                for d, dist, p, cells in options], float)
            d, distance, p, cells = options[int(rng.choice(len(options), p=weights / weights.sum()))]
            if not ds:
                footprints.append(_courier_cells(mask, points[0], d))
            ds.append(d)
            points.append(p)
            footprints.append(cells)
        if len(ds) != count:
            continue
        if teach and (len(set(ds)) < 2 or not any(a == b for a, b in zip(ds, ds[1:]))):
            continue
        # Preserve the four background inner corners used for structural parsing.
        h, w = shape
        corners = {(1, 1), (1, w - 2), (h - 2, 1), (h - 2, w - 2)}
        if footprints[0] & corners:
            continue
        return points, ds
    return None


def _available(occupied, h, w):
    sums = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    totals = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
    return np.argwhere(totals == 0)


def _reserve(occupied, r, c, h, w):
    occupied[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1] = True


def _pack_input(rng, a, force_shelf=False):
    """Independent shuffled clues, with a finite, guaranteed shelf fallback."""
    h, w = a['shape']
    objects = []
    for s in rng.permutation(sorted(a['mapping'])):
        s = int(s)
        objects.append(('stencil', s, np.where(a['stencils'][s], s, a['bg']).astype(np.int8)))
    for s in rng.permutation(sorted(a['mapping'])):
        s = int(s)
        objects.append(('key', s, _key_tile(s, a['mapping'][s], a['refs'][s])))
    for _ in range(0 if force_shelf else 36):
        fr = int(rng.choice([0, 30 - h])) if rng.random() < .7 else int(rng.integers(31 - h))
        fc = int(rng.choice([0, 30 - w])) if rng.random() < .7 else int(rng.integers(31 - w))
        occupied = np.zeros((30, 30), bool)
        _reserve(occupied, fr, fc, h, w)
        places = []
        for kind, s, tile in objects:
            spots = _available(occupied, *tile.shape)
            if not len(spots):
                break
            r, c = map(int, spots[int(rng.integers(len(spots)))])
            places.append((r, c, tile))
            _reserve(occupied, r, c, *tile.shape)
        if len(places) == len(objects):
            grid = np.full((30, 30), a['bg'], np.int8)
            fallback = False
            break
    else:
        n = a['n']
        stencils = [o for o in objects if o[0] == 'stencil']
        keys = [o for o in objects if o[0] == 'key']
        key_height = max(o[2].shape[0] for o in keys)
        key_width = sum(o[2].shape[1] + 1 for o in keys) - 1
        width = max(w, len(stencils) * (n + 1) - 1, key_width)
        fr, fc = n + key_height + 2, int(rng.integers(width - w + 1))
        grid = np.full((fr + h, width), a['bg'], np.int8)
        assert max(grid.shape) <= 30
        places = [(0, i * (n + 1), tile) for i, (_, s, tile) in enumerate(stencils)]
        c = 0
        for _, s, tile in keys:
            places.append((n + 1, c, tile))
            c += tile.shape[1] + 1
        fallback = True
    _draw_frame(grid, fr, fc, h, w, a['frame'])
    for r, c, tile in places:
        grid[r:r + tile.shape[0], c:c + tile.shape[1]] = tile
    for r, c in _courier_cells(a['mask'], a['points'][0], a['initial']):
        grid[fr + r, fc + c] = a['ink']
    for (r, c), s in a['stations'].items():
        grid[fr + r, fc + c] = s
    output = _print_board(a)
    # The local key directions make full-scene rotation an exact rule symmetry.
    rotation = int(rng.integers(4))
    grid, output = np.rot90(grid, rotation).copy(), np.rot90(output, rotation).copy()
    r, e, c, f = _box(np.argwhere(grid != a['bg']))
    r, c = max(0, r - int(rng.integers(3))), max(0, c - int(rng.integers(3)))
    e, f = min(grid.shape[0], e + int(rng.integers(3))), min(grid.shape[1], f + int(rng.integers(3)))
    return {'input': grid[r:e, c:f].copy(), 'output': output}, fallback


def _make_pair(rng, n, count, mask, teach=False, max_overlap=None, min_overlap=0,
               sparse=False, force_shelf=False):
    for _ in range(160):
        h = int(rng.integers(max(12, n + 8), 25 - n))
        w = int(rng.integers(max(13, n + 10), 28))
        route = _grow_route(rng, (h, w), mask, n, count, teach, sparse)
        if route is None:
            continue
        points, ds = route
        overlap = _overlaps(points, n)
        if overlap < min_overlap or (max_overlap is not None and overlap > max_overlap):
            continue
        palette = list(map(int, rng.permutation(10)))
        bg, frame_color, ink = palette[:3]
        k = int(rng.integers(2, min(3, count) + 1)) if teach else int(rng.integers(1, min(3 if n == 7 else 4, count) + 1))
        mapping = _sample_mapping(rng, palette[3:], k, teach)
        sources = sorted(mapping)
        assignments = sources + list(map(int, rng.choice(sources, count - k)))
        rng.shuffle(assignments)
        stations = dict(zip(points[1:], assignments))
        if teach and not any(len({d for p, d in zip(points[1:], ds) if stations[p] == s}) > 1 for s in sources):
            continue
        stencils = {}
        for i, s in enumerate(sources):
            if i and not teach and rng.random() < .15:
                stencils[s] = stencils[sources[0]].copy()
            else:
                stencils[s] = _stencil(rng, n, informative=teach)
        a = dict(shape=(h, w), bg=bg, frame=frame_color, ink=ink, mask=mask,
                 points=points, directions=ds, initial=ds[0], stations=stations,
                 stencils=stencils, refs={s: int(rng.integers(4)) for s in sources},
                 mapping=mapping, n=n, overlap=overlap)
        if teach and not all(_witnesses(a).values()):
            continue
        pair, fallback = _pack_input(rng, a, force_shelf=force_shelf)
        decoded = _analyze(pair['input'])
        assert np.array_equal(_print_board(decoded), pair['output'])
        decoded['packing_fallback'] = fallback
        return pair, decoded
    raise RuntimeError('bounded route/stencil construction exhausted')


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= int(value) <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate distinct examples sharing fixed visible printing conventions."""
    tr, te = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(['length', 'scale', 'crowding']))
    first_n = int(rng.choice([3, 5]))
    sizes = [first_n, 8 - first_n]
    puzzle, seen, train_infos = {'train': [], 'test': []}, set(), []
    first_mask = None
    for split, amount in (('train', tr), ('test', te)):
        for i in range(amount):
            teach = split == 'train' and i == 0
            short = split == 'train' and i == 1
            n = sizes[i] if split == 'train' and i < 2 else int(rng.choice([3, 5] if shift == 'scale' else [3, 5, 7]))
            if split == 'test' and shift == 'scale':
                n = 7
            count = 6 if teach else int(rng.integers(3, 6 if short else 7))
            if split == 'test' and shift == 'length':
                count = int(rng.integers(8, 10))
            elif split == 'test' and shift == 'crowding':
                count = int(rng.integers(6, 10))
            pool = [m for m in _COURIERS if not short or m != first_mask]
            mask = pool[int(rng.integers(len(pool)))]
            if teach:
                first_mask = mask
            cap = None
            minimum = 1 if teach else 0
            sparse = False
            if shift == 'crowding':
                if split == 'train':
                    cap, sparse = (1 if short else 6), not teach
                else:
                    minimum = max(a['overlap'] for a in train_infos) + 1
            for _ in range(12):
                pair, info = _make_pair(rng, n, count, mask, teach, cap, minimum, sparse)
                key = _fingerprint(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == 'train':
                        train_infos.append(info)
                    break
            else:
                raise RuntimeError('distinct-input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check schema, visible commands, complete geometry, and teaching witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    infos, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _fingerprint(inp)
            assert key not in seen, 'inputs must be distinct across both splits'
            seen.add(key)
            a = _analyze(inp)
            expected = _print_board(a)
            assert np.array_equal(out, expected), 'incorrect chronological impressions or terminal courier'
            assert out.shape == a['shape']
            assert set(map(int, np.unique(out))).issubset({a['bg'], a['frame'], a['ink']} | set(a['mapping'].values()))
            assert set(map(tuple, np.argwhere(out == a['ink']))) == _courier_cells(a['mask'], a['points'][-1], a['directions'][-1])
            assert not np.array_equal(inp, out)
            infos[split].append(a)
    train, test = infos['train'], infos['test']
    assert all(_witnesses(train[0]).values()), 'first demonstration must distinguish every documented shortcut'
    assert len({a['n'] for a in train}) >= 2
    assert len({a['mask'] for a in train}) >= 2
    assert any(a == b for a, b in zip(train[0]['directions'], train[0]['directions'][1:]))
    assert len(set(train[0]['directions'])) > 1
    assert any(len({d for p, d in zip(train[0]['points'][1:], train[0]['directions'])
                        if train[0]['stations'][p] == s}) > 1 for s in train[0]['mapping'])
    # Every generated puzzle increases a demonstrated, input-recoverable demand.
    assert (min(a['n'] for a in test) > max(a['n'] for a in train)
            or min(len(a['directions']) for a in test) > max(len(a['directions']) for a in train)
            or min(a['overlap'] for a in test) > max(a['overlap'] for a in train)), 'missing train/test shift'


def measure_pair(pair):
    """Reconstruct diversity features from the grid, without generation metadata."""
    a = _analyze(pair['input'])
    points, ds, n = a['points'], a['directions'], a['n']
    routes = []
    for mirror in (1, -1):
        for d in range(4):
            routes.append(tuple(_rotate_offset((r - points[0][0], mirror * (c - points[0][1])), d)
                                for r, c in points))
    turns = [(b - x) % 4 for x, b in zip(ds, ds[1:]) if x != b]
    return dict(n=n, stations=len(ds), categories=len(a['mapping']), overlap=a['overlap'],
                courier=a['mask'], area=len(a['mask']), route=min(routes),
                distances=tuple(abs(b[0] - x[0]) + abs(b[1] - x[1]) for x, b in zip(points, points[1:])),
                initial=ds[0], final=ds[-1], mixed_turns=1 in turns and 3 in turns,
                straight=sum(x == b for x, b in zip(ds, ds[1:])),
                input_shape=np.asarray(pair['input']).shape, output_shape=a['shape'],
                palette=(a['bg'], a['frame'], a['ink']), refs=tuple(a['refs'].values()),
                map_overlap=sum(t in a['mapping'] for t in a['mapping'].values()),
                many_to_one=len(set(a['mapping'].values())) < len(a['mapping']),
                stencils=tuple((s, m.tobytes()) for s, m in sorted(a['stencils'].items())),
                disconnected=sum(len(_components(m)) > 1 for m in a['stencils'].values()),
                symmetric=sum(not _asymmetric(m) for m in a['stencils'].values()),
                solid=sum(m.all() for m in a['stencils'].values()))


def self_test():
    """Bounded 42-seed sweep, semantic mutations, count boundaries, and diversity."""
    puzzles, rows = [], []
    signatures = set()
    shift_counts = Counter()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        signatures.add(tuple((_fingerprint(x['input']), _fingerprint(x['output']))
                             for split in ('train', 'test') for x in p[split]))
        measured = {split: [measure_pair(x) for x in p[split]] for split in ('train', 'test')}
        rows.extend(m for split in ('train', 'test') for m in measured[split])
        tr, te = measured['train'], measured['test']
        if min(m['n'] for m in te) > max(m['n'] for m in tr):
            shift_counts['scale'] += 1
        if min(m['stations'] for m in te) > max(m['stations'] for m in tr):
            shift_counts['length'] += 1
        if min(m['overlap'] for m in te) > max(m['overlap'] for m in tr):
            shift_counts['crowding'] += 1
    assert len(signatures) == 42
    assert {m['n'] for m in rows} == {3, 5, 7}
    assert {m['categories'] for m in rows} == {1, 2, 3, 4}
    assert {m['stations'] for m in rows} == set(range(3, 10))
    assert {m['initial'] for m in rows} == {m['final'] for m in rows} == set(range(4))
    assert set(d for m in rows for d in m['refs']) == set(range(4))
    assert len({m['courier'] for m in rows}) >= 12
    assert len({m['route'] for m in rows}) >= 140
    assert len({m['input_shape'] for m in rows}) >= 20
    assert len({m['output_shape'] for m in rows}) >= 50
    assert len({(m['n'], x[1]) for m in rows for x in m['stencils']}) >= 100
    assert sum(m['mixed_turns'] for m in rows) >= 20
    assert sum(m['disconnected'] for m in rows) >= 30
    assert sum(m['symmetric'] for m in rows) >= 30
    assert sum(m['solid'] for m in rows) >= 5
    assert sum(m['many_to_one'] for m in rows) >= 5
    assert all({m['palette'][i] for m in rows} == set(range(10)) for i in range(3))
    assert all(shift_counts[k] >= 5 for k in ('scale', 'length', 'crowding'))
    # Exercise the packing fallback even if free packing succeeds on all seeds.
    # This changes only construction, never the input-only prediction rule.
    for n in (3, 5, 7):
        pair, info = _make_pair(np.random.default_rng(8020 + n), n, 4,
                                _COURIERS[0], force_shelf=True)
        assert info['packing_fallback']
        assert np.array_equal(solve(pair['input']), pair['output'])
    # Rule symmetries check reference directions, center offsets, and color roles.
    perm = np.random.default_rng(112358).permutation(10)
    for p in puzzles[:5]:
        for pair in p['train'] + p['test']:
            for d in (1, 2, 3):
                assert np.array_equal(solve(np.rot90(pair['input'], d)), np.rot90(pair['output'], d))
            assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
    # Rotate one displayed stencil and its reference arrow together: same print.
    pair = puzzles[0]['train'][0]
    a = _analyze(pair['input'])
    source = sorted(a['mapping'])[0]
    changed = pair['input'].copy()
    for box in (a['stencil_boxes'][source], a['key_boxes'][source]):
        r, e, c, f = box
        changed[r:e, c:f] = np.rot90(changed[r:e, c:f], 2)
    assert np.array_equal(solve(changed), pair['output'])
    # Relocating a complete legend clue must not move its board impressions.
    occupied = pair['input'] != a['bg']
    blocked = np.zeros(occupied.shape, bool)
    for r, c in np.argwhere(occupied):
        _reserve(blocked, int(r), int(c), 1, 1)
    fr, fe, fc, ff = a['frame_box']
    _reserve(blocked, fr, fc, fe - fr, ff - fc)
    for r, e, c, f in list(a['key_boxes'].values()) + list(a['stencil_boxes'].values()):
        _reserve(blocked, r, c, e - r, f - c)
    r, e, c, f = a['key_boxes'][source]
    spots = _available(blocked, e - r, f - c)
    if len(spots):
        rr, cc = map(int, spots[0])
        changed = pair['input'].copy()
        tile = changed[r:e, c:f].copy()
        changed[r:e, c:f] = a['bg']
        changed[rr:rr + e - r, cc:cc + f - c] = tile
        assert np.array_equal(solve(changed), pair['output'])
    assert tuple((_fingerprint(x['input']), _fingerprint(x['output'])) for s in ('train', 'test')
                 for x in generate_puzzle(17)[s]) == tuple((_fingerprint(x['input']), _fingerprint(x['output']))
                                                         for s in ('train', 'test') for x in puzzles[17][s])
    for tr, te in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(7100 + tr + te, tr, te)
        assert len(p['train']) == tr and len(p['test']) == te
    for tr, te in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    try:
        _route((5, 5), {(5, 7), (7, 5)})
    except AssertionError:
        pass
    else:
        raise AssertionError('ambiguous route accepted')
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
               for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, pairs=len(rows), distinct_routes=len({m['route'] for m in rows}),
                courier_geometries=len({m['courier'] for m in rows}),
                stencil_geometries=len({(m['n'], x[1]) for m in rows for x in m['stencils']}),
                input_shapes=len({m['input_shape'] for m in rows}),
                output_shapes=len({m['output_shape'] for m in rows}),
                stamp_sizes=sorted({m['n'] for m in rows}),
                station_counts=sorted({m['stations'] for m in rows}),
                source_counts=sorted({m['categories'] for m in rows}),
                disconnected_stencils=sum(m['disconnected'] for m in rows),
                symmetric_stencils=sum(m['symmetric'] for m in rows),
                solid_stencils=int(sum(m['solid'] for m in rows)),
                mixed_turn_routes=sum(m['mixed_turns'] for m in rows),
                shift_witness_counts=dict(shift_counts),
                shortcut_witnesses_per_first_demonstration=len(_WITNESSES))
