"""Woven interference: periodic motif occupancy intersected with route parity.

The full token edge supplies orientation, a hollow 5x5 frame supplies a 3x3
sample and the highlight color, and a clear approach column supplies background.
No output convention is learned from a seed or from stored demonstrations.
"""

from itertools import permutations

import numpy as np


_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))
_ORDERS = tuple((t, r, c) for t in (False, True)
                for r in permutations(range(3)) for c in permutations(range(3)))
_DIRECT = _ORDERS.index((False, (1, 2, 0), (1, 2, 0)))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _full_box(mask):
    return (2 <= int(mask.sum()) <= 8
            and mask[0].any() and mask[-1].any()
            and mask[:, 0].any() and mask[:, -1].any())


_MASKS = tuple(m for code in range(1, 511)
              if _full_box(m := np.array([(code >> i) & 1 for i in range(9)],
                                         dtype=bool).reshape(3, 3)))


def _ordered(mask, order):
    transpose, rows, cols = _ORDERS[order]
    return (mask.T if transpose else mask)[np.ix_(rows, cols)]


_AGREEMENT = np.array([[np.array_equal(_ordered(m, k), _ordered(m, _DIRECT))
                       for k in range(72)] for m in _MASKS])


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= x <= 30 for x in g.shape), 'grid dimensions')
    _require(np.issubdtype(g.dtype, np.integer), 'integer cells required')
    _require(np.all((0 <= g) & (g <= 9)), 'ARC colors required')
    return g


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _normalize(inp):
    candidates = []
    for k in range(4):
        g = np.rot90(inp, k)
        token = int(g[0, 0])
        if np.all(g[:, 0] == token) and np.count_nonzero(g == token) == len(g):
            candidates.append((g, k, token))
    _require(len(candidates) == 1, 'unique full-edge token strip required')
    g, k, token = candidates[0]
    _require(g.shape[1] >= 7, 'scene too narrow')
    bg = int(g[0, 1])
    _require(bg != token and np.all(g[:, 1] == bg), 'clear approach column required')
    return g, k, token, bg


def _sample_panel(g, token, bg):
    candidates = []
    for color in map(int, np.unique(g)):
        if color in (token, bg):
            continue
        coords = np.argwhere(g == color)
        if len(coords) != 16:
            continue
        r, c = map(int, coords.min(axis=0))
        b, e = map(int, coords.max(axis=0))
        if (b - r, e - c) != (4, 4):
            continue
        box = g[r:r + 5, c:c + 5]
        border = np.ones((5, 5), dtype=bool)
        border[1:4, 1:4] = False
        if not np.all(box[border] == color) or np.any(box[~border] == color):
            continue
        patch = box[1:4, 1:4]
        inks = set(map(int, np.unique(patch))) - {bg}
        if len(inks) != 1:
            continue
        ink = inks.pop()
        mask = patch == ink
        if (ink == token or not _full_box(mask)
                or np.count_nonzero(g == ink) != int(mask.sum())):
            continue
        candidates.append((r, c, color, ink, mask.copy()))
    _require(len(candidates) == 1, 'unique framed sample required')
    return candidates[0]


def _pair_at(g, p, d, guides):
    normal = (-d[1], d[0])
    a, b = _add(p, normal), _add(p, (-normal[0], -normal[1]))
    return (_inside(g, a) and _inside(g, b)
            and int(g[a]) in guides and g[a] == g[b])


def _trace(g, row, bg, guides):
    p, d = (row, 1), (0, 1)
    path, turns, seen = [p], [], set()
    while True:
        _require((p, d) not in seen, 'cyclic route')
        seen.add((p, d))
        q = _add(p, d)
        if not _inside(g, q):
            break
        if int(g[q]) == bg:
            p = q
            path.append(p)
            continue
        choices = []
        for nd in ((-d[1], d[0]), (d[1], -d[0])):
            q = _add(p, nd)
            if (_inside(g, q) and g[q] == bg and _pair_at(g, q, nd, guides)):
                choices.append(nd)
        _require(len(choices) <= 1, 'ambiguous turn')
        if not choices:
            break
        turns.append(p)
        d = choices[0]
    _require(len(set(path)) == len(path), 'self-intersecting route')
    return dict(start=(row, 0), path=path, end=p, turns=turns, direction=d)


def _analyze(inp, ignore_matching=False):
    g, rotation, token, bg = _normalize(_grid(inp))
    top, left, frame, ink, mask = _sample_panel(g, token, bg)
    guides = set(map(int, np.unique(g))) - {token, bg, frame, ink}
    _require(len(guides) >= 2, 'at least two guide identities required')
    routes, mismatches = [], []
    for r in range(1, len(g) - 1):
        for c in range(1, g.shape[1]):
            if g[r, c] != bg:
                break
            a, b = int(g[r - 1, c]), int(g[r + 1, c])
            if a in guides and b in guides:
                if a == b or ignore_matching:
                    route = _trace(g, r, bg, guides)
                    route['rail'] = a
                    routes.append(route)
                if a != b:
                    mismatches.append((r, c))
                break
    counts = np.zeros(g.shape, dtype=np.int8)
    visits = {}
    for route in routes:
        for i, p in enumerate(route['path']):
            counts[p] += 1
            before = route['path'][max(0, i - 1)]
            after = route['path'][min(len(route['path']) - 1, i + 1)]
            axis = int(before[0] != after[0]), int(before[1] != after[1])
            terminal = i in (0, len(route['path']) - 1) or p in route['turns']
            visits.setdefault(p, []).append((axis, terminal))
    if not ignore_matching:
        _require(len({r['end'] for r in routes}) == len(routes), 'terminal collision')
        for p, items in visits.items():
            if len(items) > 1:
                _require(len(items) == 2 and items[0][0] != items[1][0]
                         and not any(t for axis, t in items),
                         'crossings must be perpendicular straight interiors')
    writable = g == bg
    writable = writable.copy()
    writable[top:top + 5, left:left + 5] = False
    return dict(grid=g, rotation=rotation, token=token, bg=bg, frame=frame,
                ink=ink, mask=mask, panel=(top, left), guides=guides,
                routes=routes, mismatches=mismatches, counts=counts,
                writable=writable)


def _occupancy(a, order=_DIRECT, origin=(0, 0)):
    tile = _ordered(a['mask'], order)
    rows, cols = np.indices(a['grid'].shape)
    return tile[(rows - origin[0]) % 3, (cols - origin[1]) % 3]


def _render(a, order=_DIRECT, union=False, solid=False, origin=(0, 0)):
    out = a['grid'].copy()
    occupancy = np.ones(out.shape, dtype=bool) if solid else _occupancy(a, order, origin)
    paint = a['writable'] & occupancy
    out[paint] = a['ink']
    active = a['counts'] > 0 if union else a['counts'] % 2 == 1
    out[paint & active] = a['frame']
    for route in a['routes']:
        out[route['start']] = a['bg']
        out[route['end']] = a['token']
    return np.rot90(out, -a['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve using only the visible strip, sample, rail equalities and geometry."""
    return _render(_analyze(input_grid))


def _polyline(points):
    path = [points[0]]
    for target in points[1:]:
        p = path[-1]
        d = int(np.sign(target[0] - p[0])), int(np.sign(target[1] - p[1]))
        if (d[0] == 0) == (d[1] == 0):
            return None
        while p != target:
            p = _add(p, d)
            path.append(p)
    if len(set(path)) != len(path):
        return None
    cells = set(path)
    for i, p in enumerate(path):
        degree = sum(_add(p, d) in cells for d in _DIRS)
        if degree != (1 if i in (0, len(path) - 1) else 2):
            return None
    return path


def _random_root(rng, h, w, bends):
    row = int(rng.integers(2, h - 2))
    points, d = [(row, 1)], (0, 1)
    for i in range(bends + 1):
        p = points[-1]
        if i == bends and rng.random() < .6 and d != (0, -1):
            q = ((h - 1 if d[0] > 0 else 0), p[1]) if d[0] else (p[0], w - 1)
            if abs(q[0] - p[0]) + abs(q[1] - p[1]) < 3:
                return None
        else:
            options = []
            for length in range(3, max(h, w)):
                q = p[0] + length * d[0], p[1] + length * d[1]
                if 1 <= q[0] <= h - 2 and 7 <= q[1] <= w - 3:
                    options.append(q)
            if not options:
                return None
            q = options[int(rng.integers(len(options)))]
        points.append(q)
        d = (-d[1], d[0]) if rng.integers(2) else (d[1], -d[0])
    return _polyline(points)


def _walls(rng, path, palette, h, w, occupied, inactive=False, decorate=True):
    walls = {}
    rail = int(rng.choice(palette))
    other = int(rng.choice([x for x in palette if x != rail]))
    first_bend = next((p[1] for p, q in zip(path, path[1:]) if p[0] != q[0]), w)
    start = int(rng.integers(2, min(5, first_bend - 2)))
    end = min(first_bend - 2, start + int(rng.integers(2, 5)))
    for c in range(start, end):
        walls[(path[0][0] - 1, c)] = rail
        walls[(path[0][0] + 1, c)] = other if inactive else rail
    if not inactive:
        turn_number = 0
        for i in range(1, len(path) - 1):
            p, prev, nxt = path[i], path[i - 1], path[i + 1]
            before = p[0] - prev[0], p[1] - prev[1]
            after = nxt[0] - p[0], nxt[1] - p[1]
            if before != after:
                choices = [v for v in palette if v != rail] if turn_number == 0 else palette
                local = int(rng.choice(choices))
                turn_number += 1
                walls[_add(p, before)] = int(rng.choice(palette))
                normal = -after[1], after[0]
                walls[_add(nxt, normal)] = local
                walls[_add(nxt, (-normal[0], -normal[1]))] = local
    d = path[-1][0] - path[-2][0], path[-1][1] - path[-2][1]
    cap = _add(path[-1], d)
    if 0 <= cap[0] < h and 0 <= cap[1] < w:
        walls[cap] = int(rng.choice(palette))
    if any(q in occupied or not (0 <= q[0] < h and 2 <= q[1] < w)
           for q in walls):
        return None
    # Sparse optional paired evidence, with gaps and unsupported tails. It is
    # never permitted to obstruct a known route or its open-space crossing.
    for i in (range(3, len(path) - 3) if decorate else ()):
        p, prev, nxt = path[i], path[i - 1], path[i + 1]
        d = p[0] - prev[0], p[1] - prev[1]
        if d != (nxt[0] - p[0], nxt[1] - p[1]) or rng.random() > .12:
            continue
        n = -d[1], d[0]
        qs = [_add(p, n), _add(p, (-n[0], -n[1]))]
        if all(0 <= q[0] < h and 6 <= q[1] < w and q not in occupied
               and q not in walls for q in qs):
            walls[qs[0]] = rail
            walls[qs[1]] = other if inactive else rail
    return walls


def _paint(g, walls, bg):
    if walls is None or any(g[q] not in (bg, v) for q, v in walls.items()):
        return None
    result = g.copy()
    for q, v in walls.items():
        result[q] = v
    return result


def _read_without_panel(g, bg, palette):
    routes, mismatches = {}, []
    for r in range(1, len(g) - 1):
        for c in range(1, g.shape[1]):
            if g[r, c] != bg:
                break
            a, b = int(g[r - 1, c]), int(g[r + 1, c])
            if a in palette and b in palette:
                if a == b:
                    routes[r] = _trace(g, r, bg, set(palette))['path']
                else:
                    mismatches.append(r)
                break
    return routes, mismatches


def _fallback_root(h, w, bends):
    """A roomy procedural backbone for the bounded fallback, still same rule."""
    column = w // 3
    points = [(h - 3, 1), (h - 3, column), (3, column)]
    if bends == 2:
        points.append((3, w - 1))
    elif bends == 3:
        points.extend([(3, w - 7), (0, w - 7)])
    else:
        points.extend([(3, w - 7), (6, w - 7), (6, w - 1)])
    return _polyline(points)


def _build_scene(rng, h, w, roles, bends, cross_count, fallback=False):
    bg, token, frame, ink, palette = roles
    root = _fallback_root(h, w, bends) if fallback else _random_root(rng, h, w, bends)
    if root is None:
        return None
    g = np.full((h, w), bg, dtype=np.int8)
    g[:, 0] = token
    occupied = set(root)
    walls = _walls(rng, root, palette, h, w, occupied, decorate=not fallback)
    if walls is not None:
        for p, q in zip(root, root[1:]):
            if p[0] == q[0] and p[0] != root[0][0] and (p[0], 2) not in occupied:
                walls[(p[0], 2)] = int(rng.choice(palette))
    g = _paint(g, walls, bg)
    if g is None:
        return None
    wanted = {root[0][0]: root}
    candidate_rows = set()
    for i in range(2, len(root) - 2):
        p = root[i]
        if (root[i - 1][1] == p[1] == root[i + 1][1]
                and 1 < p[0] < h - 2 and abs(p[0] - root[0][0]) >= 3):
            candidate_rows.add(p[0])
    for row in rng.permutation(sorted(candidate_rows)):
        row = int(row)
        if len(wanted) >= cross_count + 1:
            break
        if any(abs(row - r) < 3 for r in wanted):
            continue
        if len(wanted) == 1:
            root_open = root[-1][0] in (0, h - 1) or root[-1][1] == w - 1
            end = w - 2 if root_open else w - 1
        else:
            end = w - 1 if rng.random() < .65 else w - 2
        path = _polyline([(row, 1), (row, end)])
        if any(g[q] != bg for q in path):
            continue
        walls = _walls(rng, path, palette, h, w, occupied | set(path), decorate=not fallback)
        candidate = _paint(g, walls, bg)
        if candidate is None:
            continue
        target = dict(wanted)
        target[row] = path
        try:
            actual, _ = _read_without_panel(candidate, bg, palette)
        except AssertionError:
            continue
        if actual != target:
            continue
        g, wanted = candidate, target
        occupied.update(path)
    if len(wanted) != cross_count + 1:
        return None
    # A mismatch has genuine channel geometry. The inactive ray's entire
    # potential route is left clear, so equality ablation is well-defined.
    inactive = 0
    for row in rng.permutation(np.arange(1, h - 1)):
        row = int(row)
        if row in wanted or any(abs(row - r) < 3 for r in wanted):
            continue
        end = int(rng.integers(6, w - 2))
        path = _polyline([(row, 1), (row, end)])
        if any(g[q] != bg for q in path):
            continue
        walls = _walls(rng, path, palette, h, w, occupied | set(path), True, decorate=not fallback)
        candidate = _paint(g, walls, bg)
        if candidate is None:
            continue
        try:
            actual, mismatch_rows = _read_without_panel(candidate, bg, palette)
        except AssertionError:
            continue
        if actual != wanted or row not in mismatch_rows:
            continue
        g = candidate
        occupied.update(path)  # Also reserve inactive potential rays for sample placement.
        inactive += 1
        if inactive >= int(rng.integers(1, 4)):
            break
    if not inactive:
        return None
    positions = [(r, c) for r in range(h - 4) for c in range(2, w - 4)
                 if np.all(g[r:r + 5, c:c + 5] == bg)
                 and not any((y, x) in occupied for y in range(r, r + 5)
                             for x in range(c, c + 5))]
    if not positions:
        return None
    r, c = positions[int(rng.integers(len(positions)))]
    g[r:r + 5, c:c + 5] = frame
    g[r + 1:r + 4, c + 1:c + 4] = bg
    return g, wanted, (r, c)


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def _make_pair(rng, roles, rotation, bends, cross_count, is_test, allowed, seen,
               position_witness=False):
    for attempt in range(240):
        h = int(rng.integers(23, 31)) if is_test else int(rng.integers(17, 27))
        w = int(rng.integers(20, 31)) if is_test else int(rng.integers(17, 28))
        fallback = is_test and attempt >= 80
        if fallback:
            h, w = 30, int(rng.integers(26, 31))
        built = _build_scene(rng, h, w, roles, bends, cross_count, fallback)
        if built is None:
            continue
        g, intended, (r, c) = built
        ids = list(map(int, rng.permutation(np.flatnonzero(allowed))))
        if position_witness and (r % 3, c % 3) in ((0, 0), (1, 1)):
            # Both proposed sample anchors can coincide with the canvas phase
            # at these origins. No motif can distinguish the corresponding one.
            continue
        for mask_id in ids:
            candidate = g.copy()
            candidate[r + 1:r + 4, c + 1:c + 4][_MASKS[mask_id]] = roles[3]
            inp = np.rot90(candidate, rotation).copy()
            if _key(inp) in seen:
                continue
            try:
                a = _analyze(inp)
                actual = {x['start'][0]: x['path'] for x in a['routes']}
                if actual != intended:
                    continue
                occupancy = _occupancy(a)
                cross = a['counts'] == 2
                odd = a['counts'] % 2 == 1
                if not (np.any(cross & occupancy & a['writable'])
                        and np.any(odd & occupancy) and np.any(odd & ~occupancy)):
                    continue
                wrong = _analyze(inp, ignore_matching=True)
                out = _render(a)
                if np.array_equal(_render(wrong), out):
                    continue
                if position_witness and any(np.array_equal(_render(a, origin=origin), out)
                                            for origin in ((r, c), (r + 2, c + 2))):
                    continue
            except AssertionError:
                continue
            return dict(input=inp, output=out), mask_id
    raise AssertionError(f'bounded scene construction exhausted: bends={bends}, crossings={cross_count}, test={is_test}')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, 'num_train', 2, 10)
    m = _count(num_test, 'num_test', 1, 4)
    if n <= m:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    colors = list(map(int, rng.permutation(10)))
    palette_size = int(rng.integers(2, 6))
    base_roles = (*colors[:4], colors[4:4 + palette_size])
    base_rotation = int(rng.integers(4))
    mixed_rotation = bool(rng.integers(2))
    mixed_roles = bool(rng.integers(2))
    puzzle, seen, train_ids = {'train': [], 'test': []}, set(), []
    for split, count in (('train', n), ('test', m)):
        for i in range(count):
            colors_i = list(map(int, rng.permutation(10))) if mixed_roles else colors
            roles = (*colors_i[:4], colors_i[4:4 + palette_size]) if mixed_roles else base_roles
            rotation = (base_rotation + i + (1 if split == 'test' else 0)) % 4 if mixed_rotation else base_rotation
            if split == 'train':
                bends = 1 if i == 0 else 2 if i == 1 else int(rng.integers(1, 3))
                crosses = 1 if i == 0 else 2 if i == 1 else int(rng.integers(1, 3))
            else:
                bends = int(rng.integers(2, 5))
                crosses = int(rng.integers(3, 6))
            allowed = np.ones(len(_MASKS), dtype=bool)
            if split == 'train' and i == 1:
                first = train_ids[0]
                allowed &= np.array([m.sum() != _MASKS[first].sum() for m in _MASKS])
                allowed &= (_AGREEMENT & _AGREEMENT[first]).sum(axis=1) == 1
            if split == 'test':
                allowed[train_ids] = False
            pair, mask_id = _make_pair(rng, roles, rotation, bends, crosses,
                                      split == 'test', allowed, seen,
                                      position_witness=split == 'train' and i == 0)
            puzzle[split].append(pair)
            seen.add(_key(pair['input']))
            if split == 'train':
                train_ids.append(mask_id)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists required')
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, 'pair counts')
    seen, train_counts, train_masks, densities = set(), [], set(), set()
    orders = set(range(72))
    sample_anchored = {0, 1}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'shape preservation')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            a = _analyze(inp)
            _require(a['mismatches'], 'missing mismatch witness')
            h, w = a['grid'].shape
            terminal_types = {r['end'][0] in (0, h - 1) or r['end'][1] == w - 1
                              for r in a['routes']}
            _require(terminal_types == {False, True}, 'open and closed terminal evidence')
            handoff = False
            for route in a['routes']:
                if route['turns']:
                    p = route['turns'][0]
                    q = route['path'][route['path'].index(p) + 1]
                    d = q[0] - p[0], q[1] - p[1]
                    handoff |= int(a['grid'][_add(q, (-d[1], d[0]))]) != route['rail']
            _require(handoff, 'local matching must be distinguished from remembering entrance color')
            expected = _render(a)
            _require(np.array_equal(out, expected), 'wrong woven output')
            norm_out = np.rot90(out, a['rotation'])
            occupancy = _occupancy(a)
            eligible = a['writable'].copy()
            for route in a['routes']:
                eligible[route['end']] = False
            cross = eligible & (a['counts'] == 2) & occupancy
            odd = eligible & (a['counts'] % 2 == 1)
            _require(cross.any(), 'occupied crossing must witness cancellation')
            _require(np.all(norm_out[cross] == a['ink']), 'crossing parity')
            _require((odd & occupancy).any() and (odd & ~occupancy).any(), 'both motif phases on routes')
            _require(np.all(norm_out[odd & ~occupancy] == a['bg']), 'holes must remain background')
            _require(np.array_equal(norm_out == a['frame'],
                                    (a['grid'] == a['frame']) | (odd & occupancy)), 'highlight footprint')
            _require(np.count_nonzero(inp == a['token']) == np.count_nonzero(out == a['token']), 'token conservation')
            fixed = (a['grid'] != a['bg']) & (a['grid'] != a['token'])
            _require(np.array_equal(norm_out[fixed], a['grid'][fixed]), 'sample and guide preservation')
            r, c = a['panel']
            _require(np.array_equal(norm_out[r:r + 5, c:c + 5],
                                    a['grid'][r:r + 5, c:c + 5]), 'sample gaps preserved')
            _require(not np.array_equal(_render(a, union=True), out), 'parity ablation must change output')
            _require(not np.array_equal(_render(a, solid=True), out), 'motif ablation must change output')
            wrong = _analyze(inp, ignore_matching=True)
            _require(not np.array_equal(_render(wrong), out), 'matching ablation must change output')
            if split == 'train':
                train_counts.append(len(a['routes']))
                train_masks.add(a['mask'].tobytes())
                densities.add(int(a['mask'].sum()))
                orders &= {k for k in orders if np.array_equal(_render(a, order=k), out)}
                sample_anchored &= {k for k, origin in enumerate(((r, c), (r + 2, c + 2)))
                                    if np.array_equal(_render(a, origin=origin), out)}
            else:
                _require(len(a['routes']) > max(train_counts), 'test route-count shift')
                _require(a['mask'].tobytes() not in train_masks, 'test motif must be novel')
    _require(len(set(train_counts)) >= 2 and len(densities) >= 2, 'training must vary counts and motifs')
    _require(orders == {_DIRECT}, 'training must establish texture orientation and phase')
    _require(not sample_anchored, 'training must distinguish sample-anchored texture')


def self_test():
    shapes, masks, geometries, palettes, rotations, structures = set(), set(), set(), set(), set(), set()
    active_counts, bends, panel_positions, density = set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        q = generate_puzzle(seed)
        for split in p:
            for pair, repeated in zip(p[split], q[split]):
                for key in pair:
                    _require(np.array_equal(pair[key], repeated[key]), 'seed determinism')
                _require(np.array_equal(solve(pair['input'].copy()), pair['output']), 'input-only solve')
                a = _analyze(pair['input'])
                shapes.add(a['grid'].shape)
                masks.add(a['mask'].tobytes())
                palettes.add((a['bg'], a['token'], a['frame'], a['ink']))
                rotations.add(a['rotation'])
                density.add(int(a['mask'].sum()))
                active_counts.add(len(a['routes']))
                bends.update(len(r['turns']) for r in a['routes'])
                panel_positions.add(a['panel'])
                geometries.add(tuple(tuple(r['path']) for r in a['routes']))
                roles = {a['bg']: 0, a['token']: 1, a['frame']: 2, a['ink']: 3}
                canonical = tuple(tuple(roles.setdefault(int(c), len(roles)) for c in row)
                                  for row in a['grid'])
                structures.add(canonical)
    _require(len(shapes) >= 35 and len(masks) >= 75, 'shape/motif diversity')
    _require(len(geometries) >= 150 and len(structures) >= 150, 'structural diversity')
    _require(len(palettes) >= 30 and rotations == {0, 1, 2, 3}, 'role/orientation diversity')
    _require(active_counts == set(range(2, 7)) and set(range(5)) <= bends, 'count/bend diversity')
    _require(len(panel_positions) >= 35 and len(density) >= 6, 'sample placement/density diversity')
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(12345 + n * 10 + m, n, m)
            _require(len(p['train']) == n and len(p['test']) == m, 'explicit counts')
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    damaged = generate_puzzle(42, 2, 1)
    damaged['test'][0]['output'] = damaged['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, explicit counts, semantic invariants and diversity')
