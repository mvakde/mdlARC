"""Compass odometers: partial docking along an ordered two-axis route.

Only NumPy and the standard library are used. 8 is spent, 9 is fuel, and
0 is background. All other roles are recovered from the current scene.
"""
from __future__ import annotations

import itertools
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))  # top, right, bottom, left
SPENT, FUEL, BG = 8, 9, 0


def _need(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(np.all((0 <= a) & (a <= 9)), "invalid color")
    return a.astype(np.int8, copy=True)


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _dot(p, q):
    return p[0] * q[0] + p[1] * q[1]


def _add(p, q):
    return p[0] + q[0], p[1] + q[1]


def _sub(p, q):
    return p[0] - q[0], p[1] - q[1]


def _shift(cells, delta):
    return {_add(p, delta): int(v) for p, v in cells.items()}


def _norm(points):
    r, c = min(p[0] for p in points), min(p[1] for p in points)
    return tuple(sorted((y - r, x - c) for y, x in points))


def _halo(points):
    return {(r + dr, c + dc) for r, c in points
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _components(points):
    todo, result = set(points), []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, part = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in itertools.product((-1, 0, 1), repeat=2):
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        result.append(part)
    return result


def _basis_point(p, old_f, old_s, new_f, new_s):
    a, b = _dot(p, old_f), _dot(p, old_s)
    return a * new_f[0] + b * new_s[0], a * new_f[1] + b * new_s[1]


def _d4_point(p, code):
    r, c = p
    if code >= 4:
        c = -c
    for _ in range(code % 4):
        r, c = c, -r
    return r, c


def _d4(cells, code):
    return {_d4_point(p, code): int(v) for p, v in cells.items()}


def _meter_order(cells, center, forward, side):
    points = [p for p, v in cells.items() if v in (SPENT, FUEL)]
    return sorted(points, key=lambda p: (-_dot(_sub(p, center), forward),
                                         _dot(_sub(p, center), side)))


def _route(a, b, forward, side):
    """Center offsets, including the initial position; one cell per tick."""
    points = [(0, 0)]
    for i in range(1, a + 1):
        points.append((i * forward[0], i * forward[1]))
    for j in range(1, b + 1):
        points.append((a * forward[0] + j * side[0],
                       a * forward[1] + j * side[1]))
    return points


def _structure(g):
    h, w = g.shape
    _need(h >= 7 and w >= 7, "scene is too small")
    north, south = int(g[0, 0]), int(g[-1, 0])
    _need(1 <= north <= 7 and 1 <= south <= 7, "invalid boundary colors")
    n = next((r for r in range(h) if not np.all(g[r] == north)), h)
    s = next((r for r in range(h) if not np.all(g[h - 1 - r] == south)), h)
    _need(n in (1, 2) and s in (1, 2), "invalid horizontal band thickness")
    middle = g[n:h - s]
    west, east = int(middle[0, 0]), int(middle[0, -1])
    l = next((c for c in range(w) if not np.all(middle[:, c] == west)), w)
    e = next((c for c in range(w) if not np.all(middle[:, w - 1 - c] == east)), w)
    _need(l in (1, 2) and e in (1, 2), "invalid vertical band thickness")
    colors = (north, east, south, west)
    _need(len(set(colors)) == 4 and all(1 <= v <= 7 for v in colors), "invalid compass")
    mask = np.ones(g.shape, dtype=bool)
    mask[n:h - s, l:w - e] = False
    return mask, dict(zip(colors, DIRS)), (n, h - s, l, w - e), (n, e, s, l)


def _read_cart(cells, frame, reg, compass):
    ring = [p for p, v in cells.items() if v == frame]
    _need(bool(ring), "missing cart frame")
    t, b = min(r for r, _ in ring), max(r for r, _ in ring)
    l, r = min(c for _, c in ring), max(c for _, c in ring)
    _need(b - t >= 2 and r - l >= 2 and (b - t) % 2 == (r - l) % 2 == 0,
          "cart frame must have odd dimensions and a nonempty interior")
    border = {(y, x) for y in range(t, b + 1) for x in range(l, r + 1)
              if y in (t, b) or x in (l, r)}
    _need(set(ring) == border, "broken rectangular frame")
    inside = {(y, x) for y in range(t + 1, b) for x in range(l + 1, r)}
    _need(all(cells.get(p) in (SPENT, FUEL, reg) for p in inside), "invalid meter interior")
    marks = {p for p, v in cells.items() if v == reg}
    _need(marks and marks <= inside and len(_components(marks)) == 1, "invalid embedded key")
    label_groups = {}
    for p, v in cells.items():
        if v not in compass:
            continue
        y, x = p
        possibilities = [d for d, yes in (((-1, 0), y == t - 1 and l <= x <= r),
                                           ((1, 0), y == b + 1 and l <= x <= r),
                                           ((0, -1), x == l - 1 and t <= y <= b),
                                           ((0, 1), x == r + 1 and t <= y <= b)) if yes]
        _need(len(possibilities) == 1, "label does not project from a frame side")
        label_groups.setdefault(possibilities[0], []).append((p, v))
    _need(sorted(map(len, label_groups.values())) == [1, 2], "need a double and a single label")
    old_f = next(d for d, ps in label_groups.items() if len(ps) == 2)
    old_s = next(d for d, ps in label_groups.items() if len(ps) == 1)
    _need(_dot(old_f, old_s) == 0, "labels must be on perpendicular sides")
    double = label_groups[old_f]
    _need(double[0][1] == double[1][1] and
          sum(abs(x - y) for x, y in zip(double[0][0], double[1][0])) == 1,
          "double label must be a contiguous same-color patch")
    new_f = compass[double[0][1]]
    new_s = compass[label_groups[old_s][0][1]]
    _need(_dot(new_f, new_s) == 0, "label identities must name perpendicular boundaries")
    _need(set(cells) == border | inside | {p for ps in label_groups.values() for p, _ in ps},
          "unexplained cart cells")
    center = (t + b) // 2, (l + r) // 2
    order = _meter_order(cells, center, old_f, old_s)
    _need(order, "cart has no counter cells")
    advanced = sum(cells[p] == SPENT for p in order)
    _need([cells[p] for p in order] == [SPENT] * advanced + [FUEL] * (len(order) - advanced),
          "counter is not a label-relative prefix")
    oriented = {_add(center, _basis_point(_sub(p, center), old_f, old_s, new_f, new_s)): v
                for p, v in cells.items()}
    mr = _basis_point((1, 0), old_f, old_s, new_f, new_s)
    mc = _basis_point((0, 1), old_f, old_s, new_f, new_s)
    det = mr[0] * mc[1] - mr[1] * mc[0]
    pose = "reflection" if det == -1 else "translation" if (mr, mc) == ((1, 0), (0, 1)) else "rotation"
    return dict(cells=cells, center=center, oriented=oriented, forward=new_f, side=new_s,
                source_forward=old_f, source_side=old_s, advanced=advanced,
                capacity=len(order), remaining=len(order) - advanced, pose=pose,
                interior=(b - t - 1, r - l - 1))


def _parse(g):
    structural, compass, bounds, thickness = _structure(g)
    points = {tuple(map(int, p)) for p in np.argwhere((g != BG) & ~structural)}
    components = _components(points)
    mono = [p for p in components if len({int(g[q]) for q in p}) == 1]
    multi = [p for p in components if len({int(g[q]) for q in p}) > 1]
    _need(mono and len(mono) == len(multi), "need one detached stencil per cart")
    regs = {int(g[min(p)]) for p in mono}
    _need(len(regs) == 1, "detached stencils must share a registration color")
    reg = regs.pop()
    nonbands = set(map(int, np.unique(g))) - set(compass) - {reg, BG, SPENT, FUEL}
    _need(1 <= reg <= 7 and reg not in compass and len(nonbands) == 1, "invalid role colors")
    frame = nonbands.pop()
    _need(1 <= frame <= 7, "invalid frame color")
    sources = [_read_cart({p: int(g[p]) for p in part}, frame, reg, compass) for part in multi]
    return dict(grid=g, structural=structural, compass=compass, bounds=bounds,
                thickness=thickness, reg=reg, frame=frame, sources=sources, targets=mono)


def _candidates(source, info):
    oriented = source['oriented']
    marks = {p for p, v in oriented.items() if v == info['reg']}
    others = set().union(*(set(s['cells']) for s in info['sources'] if s is not source))
    t, b, l, r = info['bounds']
    candidates = []
    for index, target in enumerate(info['targets']):
        if _norm(marks) != _norm(target):
            continue
        delta = _sub(min(target), min(marks))
        a, z = _dot(delta, source['forward']), _dot(delta, source['side'])
        if a < 0 or z < 0 or a + z == 0:
            continue
        path = _route(a, z, source['forward'], source['side'])
        sweep = set(source['cells']) | {_add(p, q) for p in oriented for q in path}
        if any(not (t <= y < b and l <= x < r) for y, x in sweep):
            continue
        other_targets = set().union(*(p for j, p in enumerate(info['targets']) if j != index))
        if sweep & (others | other_targets):
            continue
        movement = min(source['remaining'], a + z)
        updated = dict(oriented)
        order = _meter_order(updated, source['center'], source['forward'], source['side'])
        for p in order[source['advanced']:source['advanced'] + movement]:
            updated[p] = SPENT
        final = _shift(updated, path[movement])
        candidates.append(dict(target=index, first=a, second=z, distance=a + z,
                               movement=movement, final=final, sweep=sweep,
                               path=path, source=source))
    return candidates


def _render_plan(info, plan):
    g = info['grid']
    out = np.zeros_like(g)
    out[info['structural']] = g[info['structural']]
    for target in info['targets']:
        for p in target:
            out[p] = info['reg']
    for c in plan:
        for p, value in c['final'].items():
            out[p] = value
    return out


def _ablation_facts(candidate):
    """Concrete bad-rule comparisons on a parsed cart, not extra solvers."""
    s = candidate['source']
    m = candidate['movement']
    delta = candidate['path'][m]
    naive = dict(s['oriented'])
    # Tick pending cells in screen row order, discarding the colored compass.
    for p in sorted(p for p, v in naive.items() if v == FUEL)[:m]:
        naive[p] = SPENT
    frozen = dict(s['cells'])
    order = _meter_order(frozen, s['center'], s['source_forward'], s['source_side'])
    for p in order[s['advanced']:s['advanced'] + m]:
        frozen[p] = SPENT
    reverse_path = _route(candidate['second'], candidate['first'], s['side'], s['forward'])
    reset = dict(s['oriented'])
    carried_order = _meter_order(reset, s['center'], s['forward'], s['side'])
    first_ticks = min(m, candidate['first'])
    for p in carried_order[s['advanced']:s['advanced'] + first_ticks]:
        reset[p] = SPENT
    turn_order = _meter_order(reset, s['center'], s['side'],
                              (-s['forward'][0], -s['forward'][1]))
    for p in [p for p in turn_order if reset[p] == FUEL][:m - first_ticks]:
        reset[p] = SPENT
    return dict(global_raster=_shift(naive, delta) != candidate['final'],
                ignored_pose=_shift(frozen, delta) != candidate['final'],
                unlimited_docking=m < candidate['distance'],
                reversed_legs=reverse_path[m] != delta,
                scan_reset=_shift(reset, delta) != candidate['final'])


def _analyze(input_grid):
    info = _parse(_grid(input_grid))
    choices = [_candidates(s, info) for s in info['sources']]
    order = sorted(range(len(choices)), key=lambda i: len(choices[i]))
    outputs = {}

    def search(depth, used, swept, plan):
        if len(outputs) >= 2:
            return
        if depth == len(order):
            out = _render_plan(info, plan)
            outputs.setdefault(_key(out), (out, plan[:]))
            return
        for candidate in choices[order[depth]]:
            if candidate['target'] not in used and not swept & candidate['sweep']:
                search(depth + 1, used | {candidate['target']}, swept | candidate['sweep'],
                       plan + [candidate])

    search(0, set(), set(), [])
    _need(len(outputs) == 1, "input does not determine one output")
    out, plan = next(iter(outputs.values()))
    return out, info, plan


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from the current input alone."""
    return _analyze(input_grid)[0]


def _base(shape, palette, thickness):
    frame, reg, north, east, south, west = palette
    n, e, s, l = thickness
    g = np.zeros(shape, dtype=np.int8)
    g[:n, :] = north
    g[-s:, :] = south
    g[n:-s, :l] = west
    g[n:-s, -e:] = east
    return g


def _pattern(rng, ih, iw, max_size, min_size=1, exact_size=None):
    size = exact_size if exact_size is not None else int(rng.integers(min_size, min(max_size, ih * iw - 1) + 1))
    points = {(int(rng.integers(ih)), int(rng.integers(iw)))}
    for _ in range(size - 1):
        frontier = sorted({(r + dr, c + dc) for r, c in points
                           for dr, dc in itertools.product((-1, 0, 1), repeat=2)
                           if 0 <= r + dr < ih and 0 <= c + dc < iw} - points)
        if not frontier:
            break
        points.add(frontier[int(rng.integers(len(frontier)))])
    return {(r - ih // 2, c - iw // 2) for r, c in points}


def _canonical_cart(rng, ih, iw, pattern, remaining, palette, f, side):
    frame, reg = palette[:2]
    rh, cw = ih // 2 + 1, iw // 2 + 1
    cells = {(r, c): frame for r in range(-rh, rh + 1) for c in range(-cw, cw + 1)
             if abs(r) == rh or abs(c) == cw}
    meter = sorted((r, c) for r in range(-rh + 1, rh) for c in range(-cw + 1, cw)
                   if (r, c) not in pattern)
    advance = len(meter) - remaining
    cells.update({p: SPENT if i < advance else FUEL for i, p in enumerate(meter)})
    cells.update({p: reg for p in pattern})
    start = int(rng.integers(-cw, cw))
    label_row = int(rng.integers(-rh, rh + 1))
    cells[(-rh - 1, start)] = palette[2 + DIRS.index(f)]
    cells[(-rh - 1, start + 1)] = palette[2 + DIRS.index(f)]
    cells[(label_row, cw + 1)] = palette[2 + DIRS.index(side)]
    return {_basis_point(p, DIRS[0], DIRS[1], f, side): v for p, v in cells.items()}


def _sample_cart(rng, palette, lesson, compact, long_distance=None, forced_forward=None,
                 key_size=None):
    if lesson in ('dock', 'after'):
        ih, iw = ((3, 5), (5, 3))[int(rng.integers(2))]
    elif long_distance is not None:
        ih, iw = ((3, 5), (5, 5), (5, 7))[int(rng.integers(3))]
    elif compact:
        ih, iw = ((1, 3), (3, 1), (3, 3))[int(rng.choice(3, p=(.18, .18, .64)))]
    else:
        ih, iw = ((1, 5), (3, 3), (3, 5), (5, 3), (5, 5), (3, 7), (7, 3))[int(rng.integers(7))]
    if key_size is not None and key_size >= ih * iw:
        return None
    pattern = _pattern(rng, ih, iw, 3 if lesson else 6,
                       min_size=2 if lesson == 'dock' else 1, exact_size=key_size)
    capacity = ih * iw - len(pattern)
    f_index = int(rng.integers(4)) if forced_forward is None else forced_forward
    f = DIRS[f_index]
    side = DIRS[(f_index + int(rng.choice((-1, 1)))) % 4]
    if long_distance is not None:
        a = int(rng.integers(4, long_distance - 3))
        b = long_distance - a
    elif lesson == 'dock':
        a, b = int(rng.integers(2, 5)), int(rng.integers(3, 6))
    elif lesson == 'after':
        a, b = int(rng.integers(2, 5)), int(rng.integers(4, 7))
    elif lesson == 'before':
        a, b = int(rng.integers(4, 8)), int(rng.integers(2, 6))
    else:
        a, b = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        if not compact and rng.random() < .22:
            if rng.random() < .5:
                a = 0
                b += 4
            else:
                b = 0
                a += 4
    distance = a + b
    if lesson == 'dock':
        if capacity <= distance + 1:
            return None
        remaining = int(rng.integers(distance + 1, capacity))
        code = int(rng.integers(1, 4))
    elif lesson == 'after':
        if capacity <= a + 1:
            return None
        remaining = int(rng.integers(a + 1, min(capacity - 1, distance - 1) + 1))
        code = int(rng.integers(4, 8))
    elif lesson == 'before':
        remaining = int(rng.integers(1, min(capacity, a - 1) + 1))
        code = int(rng.integers(8))
    else:
        # Fresh, exhausted, exact, and partially used states are independent of pose.
        regime = int(rng.integers(5))
        if regime == 0:
            remaining = 0
        elif regime == 1:
            remaining = capacity
        elif regime == 2 and distance <= capacity:
            remaining = distance
        else:
            remaining = int(rng.integers(1, capacity + 1))
        code = int(rng.integers(8))
    goal = _canonical_cart(rng, ih, iw, pattern, remaining, palette, f, side)
    source = _d4(goal, code)
    delta = (a * f[0] + b * side[0], a * f[1] + b * side[1])
    target = {_add(p, delta): palette[1] for p, v in goal.items() if v == palette[1]}
    if set(target) & _halo(source):
        return None
    path = _route(a, b, f, side)
    sweep = set(source) | {_add(p, q) for p in goal for q in path}
    return dict(source=source, target=target, sweep=sweep, signature=_norm(target),
                canonical_key=min(_norm({_d4_point(p, k) for p in target}) for k in range(8)),
                key_size=len(pattern), forward=f_index, distance=distance)


def _origins(points, blocked, bounds):
    t, b, l, r = bounds
    minr, maxr = min(y for y, _ in points), max(y for y, _ in points)
    minc, maxc = min(x for _, x in points), max(x for _, x in points)
    ys, xs = np.meshgrid(np.arange(t - minr, b - maxr),
                         np.arange(l - minc, r - maxc), indexing='ij')
    ys, xs = ys.ravel(), xs.ravel()
    ok = np.ones(len(ys), dtype=bool)
    for y, x in sorted(points):
        ok &= ~blocked[ys + y, xs + x]
    return list(zip(ys[ok].tolist(), xs[ok].tolist()))


def _construct(rng, palette, count, lesson=None, long_distance=None, forced_forward=None,
               large=False):
    low = 27 if count >= 3 else 23 if count == 2 else 20 if long_distance else 16
    h = 30 if large else int(rng.integers(low, 31))
    w = 30 if large else int(rng.integers(low, 31))
    thickness = (1, 1, 1, 1) if large else tuple(map(int, rng.integers(1, 3, 4)))
    inp = _base((h, w), palette, thickness)
    _, _, bounds, _ = _structure(inp)
    blocked = np.zeros(inp.shape, dtype=bool)
    signatures, actual_forward = set(), None
    teaching_key_size, teaching_shape = None, None
    for i in range(count):
        placed = False
        for _ in range(45):
            cart = _sample_cart(rng, palette, lesson if i == 0 else None, compact=count >= 2,
                                long_distance=long_distance if i == 0 else None,
                                forced_forward=forced_forward if i == 0 else None,
                                key_size=teaching_key_size if lesson == 'dock' and i else None)
            if cart is None or cart['signature'] in signatures:
                continue
            if lesson == 'dock' and i and cart['canonical_key'] == teaching_shape:
                continue
            positions = _origins(cart['sweep'], blocked, bounds)
            if not positions:
                continue
            origin = positions[int(rng.integers(len(positions)))]
            for cells in (cart['source'], cart['target']):
                for p, value in _shift(cells, origin).items():
                    inp[p] = value
            for y, x in _halo(_shift(dict.fromkeys(cart['sweep'], 1), origin)):
                if 0 <= y < h and 0 <= x < w:
                    blocked[y, x] = True
            signatures.add(cart['signature'])
            if i == 0:
                actual_forward = cart['forward']
                teaching_key_size, teaching_shape = cart['key_size'], cart['canonical_key']
            placed = True
            break
        if not placed:
            return None
    try:
        out, _, plan = _analyze(inp)
    except ValueError:
        return None
    if lesson == 'dock' and not all(any(_ablation_facts(c)[name] for c in plan)
                                   for name in ('global_raster', 'scan_reset')):
        return None
    return dict(input=inp, output=out), actual_forward


def _compact_fallback(rng, palette, count):
    """Four small disjoint routes on a 30-square canvas; same visible rule."""
    inp = _base((30, 30), palette, (1, 1, 1, 1))
    patterns = ({(0, 0)}, {(-1, -1), (-1, 0)}, {(-1, -1), (0, -1)},
                {(-1, -1), (-1, 0), (0, -1)})
    origins = ((8, 4), (8, 18), (22, 4), (22, 18))
    for pattern, origin in zip(patterns[:count], origins):
        capacity = 9 - len(pattern)
        goal = _canonical_cart(rng, 3, 3, pattern, int(rng.integers(capacity + 1)),
                               palette, DIRS[0], DIRS[1])
        source = _shift(_d4(goal, int(rng.integers(8))), origin)
        target = _shift({p: palette[1] for p, v in goal.items() if v == palette[1]},
                        _add(origin, (-4, 5)))
        for cells in (source, target):
            for p, v in cells.items():
                inp[p] = v
    return dict(input=inp, output=solve(inp))


def _lesson_fallback(rng, palette, lesson, forced_forward):
    """Small explicit witnesses, used only after bounded random packing."""
    inp = _base((30, 30), palette, (1, 1, 1, 1))
    pattern = {(-1, -1), (-1, 0)}
    if lesson == 'dock':
        f_index, side, a, b, remaining, code = 2, DIRS[1], 4, 5, 10, 1
    else:
        f_index = 0 if forced_forward is None else forced_forward
        side = DIRS[(f_index + 1) % 4]
        a, b = (4, 5) if lesson == 'after' else (7, 3)
        remaining, code = (6, 4) if lesson == 'after' else (3, 0)
    f = DIRS[f_index]
    goal = _canonical_cart(rng, 3, 5, pattern, remaining, palette, f, side)
    source = _d4(goal, code)
    delta = (a * f[0] + b * side[0], a * f[1] + b * side[1])
    target = {_add(p, delta): palette[1] for p, v in goal.items() if v == palette[1]}
    sweep = set(source) | {_add(p, q) for p in goal for q in _route(a, b, f, side)}
    origin = (5, 5) if lesson == 'dock' else _origins(sweep, np.zeros_like(inp, dtype=bool), (1, 29, 1, 29))[0]
    for cells in (source, target):
        for p, value in _shift(cells, origin).items():
            inp[p] = value
    if lesson == 'dock':
        # A diagonal domino and an edge-adjacent domino have equal counts but
        # are not congruent. The second route is completely separate.
        goal2 = _canonical_cart(rng, 3, 3, {(-1, -1), (0, 0)}, 6, palette, DIRS[0], DIRS[1])
        source2 = _shift(_d4(goal2, 4), (22, 20))
        target2 = _shift({p: palette[1] for p, v in goal2.items() if v == palette[1]}, (18, 25))
        for cells in (source2, target2):
            for p, value in cells.items():
                inp[p] = value
    return dict(input=inp, output=solve(inp)), f_index


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10),
                                   (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f'{name} must be an integer')
        if not low <= value <= high:
            raise ValueError(f'{name} outside supported range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    common = list(map(int, rng.permutation(np.arange(1, 8))[:6]))
    pair_roles = bool(rng.integers(2))
    shift = str(rng.choice(('distance', 'count'), p=(.65, .35)))
    puzzle, seen = {'train': [], 'test': []}, set()
    max_distance, first_forward = 0, None
    for split, number in (('train', ntrain), ('test', ntest)):
        for index in range(number):
            colors = list(map(int, rng.permutation(np.arange(1, 8))))
            if not pair_roles:
                bands = [c for c in colors if c not in common[:2]]
                colors = common[:2] + bands[:4]
            else:
                colors = colors[:6]
            lesson = ('dock' if index == 0 else 'after' if index == 1 else
                      'before' if index == 2 else None) if split == 'train' else None
            count = (2 if lesson == 'dock' else
                     int(rng.integers(3, 5)) if split == 'test' and shift == 'count' else
                     1 if lesson or split == 'test' else int(rng.integers(1, 3)))
            long_distance = max_distance + int(rng.integers(2, 6)) if split == 'test' and shift == 'distance' else None
            forced = (int(rng.choice([d for d in range(4) if d != first_forward]))
                      if split == 'train' and index == 1 else None)
            pair = None
            for attempt in range(32):
                constructed = _construct(rng, colors, count, lesson, long_distance, forced,
                                         large=attempt >= 16)
                if constructed is None:
                    continue
                candidate, forward = constructed
                if _key(candidate['input']) not in seen:
                    pair = candidate
                    if split == 'train' and index == 0:
                        first_forward = forward
                    break
            if pair is None and lesson:
                pair, forward = _lesson_fallback(rng, colors, lesson, forced)
                if index == 0:
                    first_forward = forward
            elif pair is None and count >= 2:
                pair = _compact_fallback(rng, colors, count)
            _need(pair is not None, 'bounded construction failed')
            _need(_key(pair['input']) not in seen, 'duplicate input after bounded construction')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                max_distance = max(max_distance, max(c['distance'] for c in _analyze(pair['input'])[2]))
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp, out):
    predicted, info, plan = _analyze(inp)
    _need(np.array_equal(predicted, out), 'incorrect compass odometer output')
    _need(np.array_equal(inp[info['structural']], out[info['structural']]), 'boundary changed')
    records = []
    for c in plan:
        source = c['source']
        final = c['final']
        _need(len(final) == len(source['cells']), 'cart geometry changed')
        _need(sum(v == FUEL for v in final.values()) == source['remaining'] - c['movement'],
              'one fuel cell must pay for each step')
        _need(sum(v == SPENT for v in final.values()) == source['advanced'] + c['movement'],
              'spent count did not advance by movement')
        for color in range(1, 8):
            _need(sum(v == color for v in final.values()) ==
                  sum(v == color for v in source['cells'].values()), 'non-counter cart colors changed')
        shifted_center = _add(source['center'], c['path'][c['movement']])
        order = _meter_order(final, shifted_center, source['forward'], source['side'])
        _need([final[p] for p in order] == [SPENT] * (source['advanced'] + c['movement']) +
              [FUEL] * (source['remaining'] - c['movement']), 'output counter prefix broken')
        target = info['targets'][c['target']]
        uncovered = target - set(final)
        _need(all(out[p] == info['reg'] for p in uncovered), 'uncovered floor stencil erased')
        if c['movement'] == c['distance']:
            _need({p for p, v in final.items() if v == info['reg']} == target, 'full docking misaligned')
        else:
            _need({p for p, v in final.items() if v == info['reg']} != target, 'short journey falsely docked')
        phase = ('zero' if c['movement'] == 0 else 'dock' if c['movement'] == c['distance'] else
                 'before' if c['movement'] < c['first'] else 'bend' if c['movement'] == c['first'] else 'after')
        regime = ('zero' if source['remaining'] == 0 else 'surplus' if source['remaining'] > c['distance']
                  else 'exact' if source['remaining'] == c['distance'] else 'fuel')
        records.append(dict(phase=phase, regime=regime, pose=source['pose'],
                            first=c['first'], second=c['second'], movement=c['movement'],
                            distance=c['distance'], capacity=source['capacity'],
                            advanced=source['advanced'], remaining=source['remaining'],
                            interior=source['interior'], forward=source['forward'], side=source['side'],
                            pattern=_norm(target), occluded=len(set(final) & target) -
                            len({p for p, v in final.items() if v == info['reg']} & target),
                            **_ablation_facts(c)))
    return info, plan, records


def validate_generated_puzzle(puzzle) -> None:
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train, test = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, 'canvas dimensions changed')
            _need(_key(inp) not in seen, 'duplicate inputs')
            seen.add(_key(inp))
            info, plan, records = _features(inp, out)
            _need(1 <= len(plan) <= 4, 'invalid generated cart count')
            # Required demonstrations below, rather than colors or seed metadata,
            # establish that both mechanisms affect the answer.
            (train if split == 'train' else test).append(records)
    lessons = [r for rs in train for r in rs]
    _need(any(r['regime'] == 'surplus' and r['first'] > 0 and r['second'] > 0 for r in lessons),
          'training lacks complete two-leg docking with leftover fuel')
    _need(any(r['phase'] == 'after' and r['pose'] == 'reflection' for r in lessons),
          'training lacks reflected fuel exhaustion after a bend')
    _need(any(r['pose'] == 'rotation' and r['movement'] > 0 for r in lessons),
          'training lacks active rotation')
    _need(len({r['forward'] for r in lessons if r['movement'] > 0}) >= 2,
          'training lacks compass-axis variation')
    _need(any(0 < r['advanced'] < r['capacity'] and r['movement'] >= 3 for r in lessons),
          'training lacks informative counter progression')
    for wrong_rule in ('global_raster', 'ignored_pose', 'unlimited_docking', 'reversed_legs', 'scan_reset'):
        _need(any(r[wrong_rule] for r in lessons), f'training does not distinguish {wrong_rule}')
    def canonical(pattern):
        return min(_norm({_d4_point(p, k) for p in pattern}) for k in range(8))
    _need(any(len(a['pattern']) == len(b['pattern']) and canonical(a['pattern']) != canonical(b['pattern'])
              for rs in train for i, a in enumerate(rs) for b in rs[i + 1:]),
          'training lacks equal-count noncongruent stencil geometries')
    bigger_count = any(len(rs) > max(map(len, train)) for rs in test)
    longer = any(r['distance'] > max(q['distance'] for q in lessons) for rs in test for r in rs)
    _need(bigger_count or longer, 'missing train/test shift')


def self_test() -> dict:
    """Bounded grid-derived semantic checks and variation across 42 seeds."""
    shapes, inputs, patterns, interiors, capacities, palettes, counts = (set() for _ in range(7))
    phases, poses, regimes, axes, thicknesses = (set() for _ in range(5))
    all_records, by_split = [], {'train': [], 'test': []}
    signatures = set()
    equal_count_shapes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                info, _, rs = _features(inp, out)
                shapes.add(inp.shape)
                inputs.add(_key(inp))
                palettes.add((info['frame'], info['reg'], tuple(info['compass'])))
                counts.add(len(rs))
                thicknesses.add(info['thickness'])
                ps = [r['pattern'] for r in rs]
                equal_count_shapes += any(len(a) == len(b) and a != b
                                          for i, a in enumerate(ps) for b in ps[i + 1:])
                all_records.extend(rs)
                by_split[split].extend(rs)
                for r in rs:
                    patterns.add(r['pattern'])
                    interiors.add(r['interior'])
                    capacities.add(r['capacity'])
                    phases.add(r['phase'])
                    poses.add(r['pose'])
                    regimes.add(r['regime'])
                    axes.add((r['forward'], r['side']))
        if seed in (0, 17, 41):
            again = generate_puzzle(seed)
            _need(all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s])
                      for k in ('input', 'output')), 'nondeterministic generation')
    _need(len(signatures) == 42 and len(inputs) == 210, 'seed diversity collapsed')
    _need(len(shapes) >= 30 and len(patterns) >= 30 and len(interiors) >= 10, 'geometry diversity collapsed')
    _need(counts == {1, 2, 3, 4}, 'cart count diversity collapsed')
    _need(phases >= {'zero', 'before', 'after', 'dock'}, 'stopping phase diversity collapsed')
    _need(poses == {'translation', 'rotation', 'reflection'}, 'pose diversity collapsed')
    _need(regimes == {'zero', 'surplus', 'exact', 'fuel'}, 'fuel relationship diversity collapsed')
    _need(len(axes) == 8 and len(thicknesses) >= 8, 'compass or boundary diversity collapsed')
    _need(equal_count_shapes >= 5, 'stencil count became a sufficient matching shortcut')
    _need(any(r['advanced'] == 0 for r in all_records), 'no fresh counter')
    _need(any(r['occluded'] > 0 for r in all_records), 'floor-stencil occlusion untested')
    # Recolor structural roles arbitrarily; the input alone still provides the compass.
    p = generate_puzzle(0)
    perm = {0: 0, 8: 8, 9: 9, **dict(zip(range(1, 8), (6, 3, 1, 5, 2, 7, 4)))}
    for pair in p['train'] + p['test']:
        inp = np.array([[perm[int(v)] for v in row] for row in pair['input']], dtype=np.int8)
        out = np.array([[perm[int(v)] for v in row] for row in pair['output']], dtype=np.int8)
        _need(np.array_equal(solve(inp), out), 'role recoloring changed the rule')
    # Meaningful mutation: a counter tick or a boundary-cell change must be rejected.
    for color in (SPENT, FUEL):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        g = bad['test'][0]['output']
        point = tuple(map(int, np.argwhere(g == color)[0])) if np.any(g == color) else (0, 0)
        g[point] = FUEL if color == SPENT else SPENT
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('validator accepted a corrupt output')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted illegal counts')
    return dict(seeds=42, pairs=210, distinct_inputs=len(inputs), shapes=len(shapes),
                registration_poses=len(patterns), interiors=len(interiors), capacities=len(capacities),
                palettes=len(palettes), counts=sorted(counts), phases=sorted(phases),
                poses=sorted(poses), fuel_regimes=sorted(regimes), compass_axis_pairs=len(axes),
                equal_count_distinct_stencil_pairs=equal_count_shapes)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
