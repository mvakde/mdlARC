"""Ranked sluices: ordinal-selected closures change exterior reachability.

Only standard-library modules and NumPy are used. No parent module, fixture,
seed, room label, or generation record is consulted by solve().
"""
from __future__ import annotations

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _array(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid size")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(((a >= 0) & (a <= 9)).all()), "invalid color")
    return a.astype(np.int8, copy=False)


def _components(mask, neighbors=N4):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        stack, cells = [first], [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    cells.append(q)
                    stack.append(q)
        result.append(tuple(sorted(cells)))
    return result


def _holes(barrier, neighbors=N4):
    """Enclosed air; a virtual outside border handles clipping explicitly."""
    b = np.pad(barrier, 1, constant_values=False)
    reached = np.zeros_like(b)
    reached[0, 0] = True
    stack = [(0, 0)]
    h, w = b.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not b[rr, cc] and not reached[rr, cc]:
                reached[rr, cc] = True
                stack.append((rr, cc))
    return (~b & ~reached)[1:-1, 1:-1]


def _dilate(mask):
    p = np.pad(mask, 1)
    h, w = mask.shape
    return np.logical_or.reduce([p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
                                 for dr, dc in N8])


def _mark(shape, cells):
    mask = np.zeros(shape, bool)
    if cells:
        rr, cc = np.array(cells, int).T
        mask[rr, cc] = True
    return mask


def _straight(cells):
    rows = {r for r, c in cells}
    cols = {c for r, c in cells}
    return ((len(rows) == 1 and len(cells) == max(cols) - min(cols) + 1)
            or (len(cols) == 1 and len(cells) == max(rows) - min(rows) + 1))


def _payload_gates(a, bg, porous):
    h, w = a.shape
    gates = []
    for cells in _components(a == porous):
        _require(_straight(cells), "porous components must be straight")
        rr, cc = np.array(cells, int).T
        possibilities = []
        for dr, dc in ((1, 0), (0, 1)):
            if dr and len(set(cc)) != 1:
                continue
            if dc and len(set(rr)) != 1:
                continue
            first, last = min(cells), max(cells)
            caps = ((first[0] - dr, first[1] - dc), (last[0] + dr, last[1] + dc))
            if not all(0 <= r < h and 0 <= c < w for r, c in caps):
                continue
            control = int(a[caps[0]])
            if control != bg and control != porous and a[caps[1]] == control:
                possibilities.append(dict(cells=cells, caps=caps, control=control,
                                          step=(dr, dc), length=len(cells) + 2))
        _require(len(possibilities) == 1, "porous run must have one matching-cap axis")
        gates.append(possibilities[0])
    _require(gates, "no gates")
    controls = sorted({g['control'] for g in gates})
    remainder = set(map(int, np.unique(a))) - {bg, porous} - set(controls)
    _require(len(remainder) == 1, "one neutral wall identity is required")
    wall = remainder.pop()
    cap_cells = [q for g in gates for q in g['caps']]
    _require(len(cap_cells) == len(set(cap_cells)), "gates cannot share caps")
    world = (a == wall) | _mark(a.shape, cap_cells + [q for g in gates for q in g['cells']])
    keys, ranks = {}, {}
    for control in controls:
        endpoints = {q for g in gates if g['control'] == control for q in g['caps']}
        key = tuple(sorted(set(map(tuple, np.argwhere(a == control).tolist())) - endpoints))
        _require(key and _straight(key) and 1 <= len(key) <= 4, "invalid unary key")
        _require(not np.any(_mark(a.shape, key) & _dilate(world)), "key touches the wall network")
        family = [g for g in gates if g['control'] == control]
        _require(len(family) >= 2 and len(key) <= len(family), "rank lacks competitors")
        _require(len({g['length'] for g in family}) == len(family), "tied family lengths")
        keys[control], ranks[control] = key, len(key)
    for c, key in keys.items():
        others = [q for other, k in keys.items() if other != c for q in k]
        _require(not np.any(_mark(a.shape, key) & _dilate(_mark(a.shape, others))), "keys touch")
    for g in gates:
        dr, dc = g['step']
        first, last = g['caps']
        beyond = ((first[0] - dr, first[1] - dc), (last[0] + dr, last[1] + dc))
        _require(all(0 <= r < h and 0 <= c < w and a[r, c] == wall for r, c in beyond),
                 "gates must terminate in neutral walls")
    return dict(background=bg, porous=porous, wall=wall, gates=gates, keys=keys, ranks=ranks)


def _decode(input_grid):
    a = _array(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(np.count_nonzero(counts == counts[bg]) == 1, "background is ambiguous")
    candidates = []
    for porous in np.unique(a):
        if porous == bg:
            continue
        try:
            candidates.append(_payload_gates(a, bg, int(porous)))
        except AssertionError:
            pass
    _require(len(candidates) == 1, "input must have one wall/gate/key interpretation")
    return a, candidates[0]


def _selected(d, alternative=None):
    result = []
    for c, rank in d['ranks'].items():
        family = [g for g in d['gates'] if g['control'] == c]
        if alternative in ('forward', 'reverse'):
            ordered = sorted(family, key=lambda g: min(g['caps']), reverse=alternative == 'reverse')
        else:
            ordered = sorted(family, key=lambda g: g['length'], reverse=alternative != 'ascending')
        index = rank - 1
        if alternative == 'longest':
            index = 0
        elif alternative == 'shortest':
            index = -1
        elif alternative == 'second_shortest':
            index = -2
        result.append(ordered[index])
    return result


def _zones(a, d, chosen=None):
    chosen = _selected(d) if chosen is None else chosen
    endpoints = [q for g in d['gates'] for q in g['caps']]
    original = (a == d['wall']) | _mark(a.shape, endpoints)
    closure = _mark(a.shape, [q for g in chosen for q in g['cells']])
    final = original | closure
    old = _holes(original)
    enclosed = _holes(final)
    exterior = ~final & ~enclosed
    fresh = enclosed & ~old & (a == d['background'])
    outline = _dilate(final) & exterior & (a == d['background'])
    return dict(original=original, final=final, closure=closure, old=old,
                enclosed=enclosed, exterior=exterior, fresh=fresh, outline=outline, selected=chosen)


def _answer(a, d, z):
    out = a.copy()
    for g in z['selected']:
        rr, cc = np.array(g['cells']).T
        out[rr, cc] = g['control']
    out[z['fresh']] = d['porous']
    out[z['outline']] = d['wall']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode semantic roles and reachability entirely from the input pixels."""
    a, d = _decode(input_grid)
    return np.ascontiguousarray(_answer(a, d, _zones(a, d)))


def _signature(a):
    a = np.asarray(a, np.int8)
    return a.shape, a.tobytes()


def _checked_int(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if value < low or (high is not None and value > high):
        raise ValueError(name + " is outside its legal range")
    return int(value)


# Construction helpers below have no role in solving.
def _network(rng, layout):
    nr, nc = layout
    ys = [int(v) for v in rng.integers(8 if nr < 3 else 7, 12 if nr == 1 else 11, nr)]
    xs = [int(v) for v in rng.integers(7 if nc >= 3 else 8, 10 if nc >= 3 else 12, nc)]
    while sum(ys) > 23:
        ys[int(np.argmax(ys))] -= 1
    while sum(xs) > 28:
        xs[int(np.argmax(xs))] -= 1
    if nc == 4:
        xs = [7] * 4
    y = [0] + list(map(int, np.cumsum(ys)))
    x = [0] + list(map(int, np.cumsum(xs)))
    wall = np.zeros((y[-1] + 1, x[-1] + 1), bool)
    wall[y, :] = True
    wall[:, x] = True
    rooms = [(y[i], y[i + 1], x[j], x[j + 1]) for i in range(nr) for j in range(nc)]
    edges = []
    for i in range(nr + 1):
        for j in range(nc):
            sides = ((i - 1) * nc + j if i else -1, i * nc + j if i < nr else -1)
            edges.append(dict(axis=1, line=y[i], start=x[j], end=x[j + 1], sides=sides))
    for i in range(nr):
        for j in range(nc + 1):
            sides = (i * nc + j - 1 if j else -1, i * nc + j if j < nc else -1)
            edges.append(dict(axis=0, line=x[j], start=y[i], end=y[i + 1], sides=sides))
    return wall, rooms, edges


def _edge(edges, a, b):
    return [i for i, e in enumerate(edges) if set(e['sides']) == {a, b}]


def _cut(rng, wall_mask, edges, assignments, palette):
    bg, wall, porous = palette[:3]
    a = np.where(wall_mask, wall, bg).astype(np.int8)
    gates = []
    for edge_index, control, length in assignments:
        e = edges[edge_index]
        start = int(rng.integers(e['start'] + 1, e['end'] - length + 1))
        path = ([(e['line'], c) for c in range(start, start + length)] if e['axis']
                else [(r, e['line']) for r in range(start, start + length)])
        for q in path:
            a[q] = porous
        a[path[0]] = a[path[-1]] = control
        gates.extend(path)
    return a, _mark(a.shape, gates)


def _unambiguous(mask):
    a, b, c, d = mask[:-1, :-1], mask[:-1, 1:], mask[1:, :-1], mask[1:, 1:]
    return not np.any((a == d) & (b == c) & (a != b))


def _baffles(rng, a, rooms, gate_mask, palette, density):
    """Inward attached wall blocks preserve each room's connected air."""
    bg, wall, porous = palette[:3]
    protected = gate_mask | _dilate(gate_mask)
    for r0, r1, c0, c1 in rooms:
        for _ in range(int(rng.integers(0, density + 1))):
            side = int(rng.integers(4))
            depth = int(rng.integers(1, 4))
            if side < 2:
                start = int(rng.integers(c0 + 1, c1))
                stop = min(c1, start + int(rng.integers(1, 4)))
                block = (slice(r0 + 1, min(r1, r0 + depth + 1)) if side == 0
                         else slice(max(r0 + 1, r1 - depth), r1), slice(start, stop))
            else:
                start = int(rng.integers(r0 + 1, r1))
                stop = min(r1, start + int(rng.integers(1, 4)))
                block = (slice(start, stop), slice(c0 + 1, min(c1, c0 + depth + 1)) if side == 2
                         else slice(max(c0 + 1, c1 - depth), c1))
            if protected[block].any():
                continue
            candidate = a.copy()
            candidate[block] = wall
            air = candidate[r0 + 1:r1, c0 + 1:c1] == bg
            barrier = (candidate != bg) & (candidate != porous)
            if air.any() and len(_components(air)) == 1 and _unambiguous(barrier):
                a = candidate
    return a


def _key_plan(rng, shape, controls, max_rank, orientation=None, reflected=None):
    h, w = shape
    horizontal = bool(rng.integers(2))
    kh = 2 * len(controls) - 1 if horizontal else max_rank
    kw = max_rank if horizontal else 2 * len(controls) - 1
    if h + kh + 2 > 30:
        horizontal, kh, kw = False, max_rank, 2 * len(controls) - 1
    _require(h + kh + 2 <= 30 and w >= kw, "key/network exceeds canvas")
    extra_h = int(rng.integers(0, min(4, 30 - h - kh - 2) + 1))
    extra_w = int(rng.integers(0, min(4, 30 - w) + 1))
    top = int(rng.integers(extra_h + 1))
    left = int(rng.integers(extra_w + 1))
    key_x = int(rng.integers(w + extra_w - kw + 1))
    order = list(map(int, rng.permutation(controls)))
    positions = {c: (top + 2 * i, key_x) if horizontal else (top, key_x + 2 * i)
                 for i, c in enumerate(order)}
    return dict(shape=(h + kh + 2 + extra_h, w + extra_w), offset=(top + kh + 2, left),
                positions=positions, horizontal=horizontal,
                orientation=int(rng.integers(4)) if orientation is None else orientation,
                reflected=bool(rng.integers(2)) if reflected is None else reflected)


def _decorate(world, ranks, palette, plan):
    a = np.full(plan['shape'], palette[0], np.int8)
    r, c = plan['offset']
    a[r:r + world.shape[0], c:c + world.shape[1]] = world
    for control, rank in ranks.items():
        rr, cc = plan['positions'][control]
        for i in range(rank):
            if plan['horizontal']:
                a[rr, cc + i] = control
            else:
                a[rr + i, cc] = control
    if plan['reflected']:
        a = np.fliplr(a)
    return np.ascontiguousarray(np.rot90(a, plan['orientation']))


def _pair(world, ranks, palette, rng, plan=None):
    plan = _key_plan(rng, world.shape, list(ranks), max(ranks.values())) if plan is None else plan
    a = _decorate(world, ranks, palette, plan)
    a, d = _decode(a)
    z = _zones(a, d)
    _require(np.array_equal(z['old'], _holes(z['original'], N8)), "diagonal original leak")
    _require(np.array_equal(z['enclosed'], _holes(z['final'], N8)), "diagonal final leak")
    return {'input': a, 'output': _answer(a, d, z)}


def _anchors(rng, palette, density):
    ca, cb = palette[3:5]
    for attempt in range(32):
        wall, rooms, edges = _network(rng, (2, 2))
        limit = min(e['end'] - e['start'] - 1 for e in edges)
        la = sorted(map(int, rng.choice(np.arange(3, limit + 1), 3, replace=False)), reverse=True)
        lb = sorted(map(int, rng.choice(np.arange(3, limit + 1), 2, replace=False)), reverse=True)
        # A controls exterior entrances; B's longest is their shared partition.
        e0 = int(rng.choice(_edge(edges, 0, -1)))
        e1 = int(rng.choice(_edge(edges, 1, -1)))
        e2 = list(map(int, rng.permutation(_edge(edges, 2, -1))))
        assignments = [(e0, ca, la[0]), (e1, ca, la[1]), (e2[0], ca, la[2]),
                       (_edge(edges, 0, 1)[0], cb, lb[0]), (e2[1], cb, lb[1])]
        world, gates = _cut(rng, wall, edges, assignments, palette)
        world = _baffles(rng, world, rooms, gates, palette, density)
        plan = _key_plan(rng, world.shape, [ca, cb], 2)
        transformations = list(map(int, rng.permutation(8)))
        for transform in transformations:
            plan['orientation'], plan['reflected'] = transform % 4, bool(transform // 4)
            try:
                p1 = _pair(world, {ca: 1, cb: 1}, palette, rng, plan)
                p2 = _pair(world, {ca: 2, cb: 1}, palette, rng, plan)
            except AssertionError:
                continue
            _, d = _decode(p1['input'])
            family = sorted([g for g in d['gates'] if g['control'] == ca], key=lambda g: min(g['caps']))
            if family[1]['length'] != la[0]:
                continue
            return [p1, p2]
    raise AssertionError("could not build a readable spatially contrastive anchor")


def _ranks_for_targets(assignments, targets):
    result = {}
    for control, target_edge in targets.items():
        family = sorted([v for v in assignments if v[1] == control], key=lambda v: -v[2])
        result[control] = next(i + 1 for i, v in enumerate(family) if v[0] == target_edge)
    return result


def _cluster_scene(rng, palette, density, max_rank):
    wall, rooms, edges = _network(rng, (2, 2))
    ca, cb, cc = palette[3:6]
    e0, e1 = int(rng.choice(_edge(edges, 0, -1))), int(rng.choice(_edge(edges, 1, -1)))
    e2 = list(map(int, rng.permutation(_edge(edges, 2, -1))))
    shared = _edge(edges, 0, 1)[0]
    third = bool(rng.integers(2))
    if third:
        e3 = int(rng.choice(_edge(edges, 3, -1)))
        roles = [(e0, ca), (e1, cb), (e2[0], ca), (e2[1], cc), (e3, cb), (shared, cc)]
        targets = {ca: e0, cb: e1, cc: e2[1]}
    else:
        owner = ca if bool(rng.integers(2)) else cb
        roles = [(e0, ca), (e1, cb), (e2[0], ca), (e2[1], cb), (shared, owner)]
        targets = {ca: e0, cb: e1}
    limit = min(e['end'] - e['start'] - 1 for e in edges)
    assignments = []
    for control in targets:
        family_edges = [e for e, c in roles if c == control]
        lengths = sorted(map(int, rng.choice(np.arange(3, limit + 1), len(family_edges), replace=False)),
                         reverse=True)
        target_index = int(rng.integers(min(max_rank, len(lengths))))
        target_length = lengths.pop(target_index)
        rng.shuffle(lengths)
        for e in family_edges:
            assignments.append((e, control, target_length if e == targets[control] else lengths.pop()))
    ranks = _ranks_for_targets(assignments, targets)
    world, gates = _cut(rng, wall, edges, assignments, palette)
    world = _baffles(rng, world, rooms, gates, palette, density)
    return _pair(world, ranks, palette, rng)


def _ordinal_scene(rng, palette, density, rank, population=None):
    population = int(rng.integers(max(2, rank), 5)) if population is None else population
    count = min(4, population + int(rng.integers(2)))
    wall, rooms, edges = _network(rng, (1, count))
    candidates = [int(rng.choice(_edge(edges, i, -1))) for i in range(population)]
    limit = min(edges[i]['end'] - edges[i]['start'] - 1 for i in candidates)
    lengths = list(map(int, rng.choice(np.arange(3, limit + 1), population, replace=False)))
    control = palette[3]
    assignments = [(e, control, length) for e, length in zip(candidates, lengths)]
    world, gates = _cut(rng, wall, edges, assignments, palette)
    world = _baffles(rng, world, rooms, gates, palette, density)
    return _pair(world, {control: rank}, palette, rng)


def _random_scene(rng, palette, density, max_rank, chain=False, require_fill=False, reject_joined=False):
    layouts = [(1, 2), (1, 3), (1, 4), (2, 1), (2, 2), (3, 1)]
    if chain:
        layouts.remove((2, 2))
    for attempt in range(48):
        layout = layouts[int(rng.integers(len(layouts)))]
        wall, rooms, edges = _network(rng, layout)
        f = int(rng.integers(1, 4))
        populations = list(map(int, rng.integers(2, 5, f)))
        if sum(populations) > len(edges):
            continue
        chosen_edges = list(map(int, rng.choice(len(edges), sum(populations), replace=False)))
        assignments, ranks, cursor = [], {}, 0
        for control, n in zip(palette[3:3 + f], populations):
            family = chosen_edges[cursor:cursor + n]
            cursor += n
            limit = min(edges[i]['end'] - edges[i]['start'] - 1 for i in family)
            if limit - 2 < n:
                break
            lengths = list(map(int, rng.choice(np.arange(3, limit + 1), n, replace=False)))
            assignments.extend((e, control, length) for e, length in zip(family, lengths))
            ranks[control] = int(rng.integers(1, min(max_rank, n) + 1))
        else:
            world, gates = _cut(rng, wall, edges, assignments, palette)
            world = _baffles(rng, world, rooms, gates, palette, density)
            try:
                pair = _pair(world, ranks, palette, rng)
            except AssertionError:
                continue
            a, d = _decode(pair['input'])
            z = _zones(a, d)
            joined = any(any(z['enclosed'][q] and not z['old'][q] for q in g['cells'])
                         for g in d['gates'] if g not in z['selected'])
            if reject_joined and joined:
                continue
            if not require_fill or z['fresh'].any():
                return pair
    # Deterministic bounded fallback remains a chain with a real newly sealed room.
    return _ordinal_scene(rng, palette, density, min(2, max_rank), population=3)


def _unique(pair, seen):
    if _signature(pair['input']) not in seen:
        return pair
    # 40 distinct input-visible recolorings/orientations against <=13 prior
    # inputs. The detached key cluster distinguishes the four rotations.
    for shift in range(10):
        for rotation in range(4):
            a = np.ascontiguousarray((np.rot90(pair['input'], rotation).astype(int) + shift) % 10,
                                     dtype=np.int8)
            if _signature(a) not in seen:
                return {'input': a, 'output': solve(a)}
    raise AssertionError("unique variant fallback exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _checked_int('seed', seed, 0)
    nt = _checked_int('num_train', num_train, 2, 10)
    nq = _checked_int('num_test', num_test, 1, 4)
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    variable_palette = bool(rng.integers(2))
    density = int(rng.integers(0, 5))
    regime = str(rng.choice(['cluster', 'arrangement', 'ordinal']))
    max_rank = int(rng.integers(2, 5 if nt >= 3 else 4))
    if regime == 'ordinal':
        max_rank = max(3, max_rank)
    anchors = _anchors(rng, palette, density)
    if bool(rng.integers(2)):
        anchors.reverse()
    puzzle, seen = {'train': anchors, 'test': []}, {_signature(p['input']) for p in anchors}
    for split, count in (('train', nt - 2), ('test', nq)):
        for index in range(count):
            colors = list(map(int, rng.permutation(10))) if variable_palette else palette
            if split == 'train':
                if index == 0 and max_rank == 4:
                    pair = _ordinal_scene(rng, colors, density, 3)
                else:
                    pair = _random_scene(rng, colors, density, max_rank,
                                         reject_joined=regime == 'cluster')
            elif regime == 'cluster':
                pair = _cluster_scene(rng, colors, density, max_rank)
            elif regime == 'arrangement':
                pair = _random_scene(rng, colors, density, max_rank, chain=True, require_fill=True)
            else:
                rank = max_rank
                pair = _ordinal_scene(rng, colors, density, rank)
            pair = _unique(pair, seen)
            seen.add(_signature(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _cooperative(a, d, z):
    if not z['fresh'].any() or len(z['selected']) < 2:
        return False
    # At least one newly enclosed cell depends on two or more closures.
    independently = np.zeros(a.shape, bool)
    for g in z['selected']:
        independently |= _zones(a, d, [g])['fresh']
    return bool(np.any(z['fresh'] & ~independently))


def _shared_gate_witness(a, d, z):
    rr, cc = np.nonzero(z['original'])
    r0, r1, c0, c1 = rr.min(), rr.max(), cc.min(), cc.max()
    for g in z['selected']:
        # A selected internal passage separates a newly sealed basin and
        # another room which remains reachable from the outside.
        if not all(r0 < r < r1 and c0 < c < c1 for r, c in g['cells']):
            continue
        neighbors = np.zeros(a.shape, bool)
        for r, c in g['cells']:
            for dr, dc in N4:
                q = (r + dr, c + dc)
                if 0 <= q[0] < a.shape[0] and 0 <= q[1] < a.shape[1]:
                    neighbors[q] = True
        if np.any(neighbors & z['fresh']) and np.any(neighbors & z['exterior'] & (a == d['background'])):
            return True
    return False


def _features(a, d=None, z=None):
    if d is None:
        a, d = _decode(a)
    z = _zones(a, d) if z is None else z
    unselected = [g for g in d['gates'] if g not in z['selected']]
    internal_open = sum(any(z['enclosed'][q] and not z['old'][q] for q in g['cells']) for g in unselected)
    lengths = [g['length'] for g in d['gates']]
    return dict(shape=a.shape, families=len(d['ranks']), gates=len(d['gates']),
                populations=tuple(sorted(sum(g['control'] == c for g in d['gates']) for c in d['ranks'])),
                ranks=tuple(sorted(d['ranks'].values())), lengths=tuple(sorted(lengths)),
                axes=tuple(sorted({g['step'] for g in d['gates']})),
                old_area=int((z['old'] & (a == d['background'])).sum()),
                fresh_area=int(z['fresh'].sum()), internal_open=internal_open,
                cooperative=_cooperative(a, d, z), shared_gate=_shared_gate_witness(a, d, z),
                clipped=bool(z['final'][0].any() or z['final'][-1].any()
                             or z['final'][:, 0].any() or z['final'][:, -1].any()),
                palette=(d['background'], d['wall'], d['porous']),
                holes=len(_components(z['enclosed'])),
                irregular=any(len(cells) < (max(r for r, c in cells) - min(r for r, c in cells) + 1)
                              * (max(c for r, c in cells) - min(c for r, c in cells) + 1)
                              for cells in _components(z['fresh'])))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), "illegal counts")
    seen, train, ranks = set(), [], set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
            a, out = _array(pair['input']), _array(pair['output'])
            _require(a.shape == out.shape, "changed canvas dimensions")
            key = _signature(a)
            _require(key not in seen, "duplicate input")
            seen.add(key)
            _, d = _decode(a)
            z = _zones(a, d)
            _require(np.array_equal(z['old'], _holes(z['original'], N8)), "original diagonal leak")
            _require(np.array_equal(z['enclosed'], _holes(z['final'], N8)), "final diagonal leak")
            _require(not np.any(z['old'] & ~z['final'] & ~z['enclosed']), "closure opened old air")
            _require(not np.any(z['fresh'] & z['outline']), "interior/exterior overlap")
            _require(z['outline'].any(), "no visible exterior outline")
            edited = z['closure'] | z['fresh'] | z['outline']
            _require(np.array_equal(out[~edited], a[~edited]), "preserved pixels changed")
            for g in z['selected']:
                rr, cc = np.array(g['cells']).T
                _require(np.all(out[rr, cc] == g['control']), "wrong sluice closure")
            _require(np.all(out[z['fresh']] == d['porous']), "wrong new basin fill")
            _require(np.all(out[z['outline']] == d['wall']), "wrong exterior outline")
            _require(np.any(a != out), "identity example")
            if split == 'train':
                train.append((a, d, z))
                ranks.update(d['ranks'].values())
            else:
                for rank in d['ranks'].values():
                    _require(rank in ranks or set(range(1, rank)) <= ranks, "unshown ordinal progression")
    _require({1, 2} <= ranks, "training must show ranks one and two")
    history = {}
    for a, d, z in train:
        for c, rank in d['ranks'].items():
            history.setdefault(c, set()).add(rank)
    _require(any({1, 2} <= v for v in history.values()), "same cap must change ordinal")
    _require(any(rank == 2 and sum(g['control'] == c for g in d['gates']) >= 3
                 for a, d, z in train for c, rank in d['ranks'].items()), "no intermediate rank example")
    _require(any((z['old'] & (a == d['background'])).any() for a, d, z in train), "no old sealed pocket")
    _require(any(_cooperative(a, d, z) for a, d, z in train), "no cooperative closure evidence")
    _require(any(_shared_gate_witness(a, d, z) for a, d, z in train), "no still-open neighboring chamber")
    for shortcut in ('longest', 'shortest', 'second_shortest', 'ascending', 'forward', 'reverse'):
        _require(any({g['cells'] for g in _selected(d, shortcut)} != {g['cells'] for g in z['selected']}
                     for a, d, z in train), "training permits shortcut " + shortcut)
    _require(any(z['fresh'].any() for a, d, z in train), "omitting topology does not change training")


def self_test():
    """Bounded pixel-derived invariants, 42-seed coverage, and regressions."""
    records, puzzles, normalized = [], [], set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        q = generate_puzzle(seed)
        _require(all(_signature(x[k]) == _signature(y[k]) for s in p
                     for x, y in zip(p[s], q[s]) for k in ('input', 'output')), "nondeterminism")
        puzzles.append(p)
        for s in p:
            for pair in p[s]:
                a, d = _decode(pair['input'])
                z = _zones(a, d)
                _require(np.array_equal(solve(a.copy()), pair['output']), "input-only solve disagrees")
                records.append(_features(a, d, z))
                mapping, next_role = {}, 0
                colors = []
                for c in a.flat:
                    c = int(c)
                    if c not in mapping:
                        mapping[c] = next_role
                        next_role += 1
                    colors.append(mapping[c])
                normalized.add((a.shape, tuple(colors)))
    _require(len({r['shape'] for r in records}) >= 30, "canvas variation too narrow")
    _require(len(normalized) >= 150, "structural variation too narrow")
    _require({1, 2, 3} <= {r['families'] for r in records}, "cap family count coverage")
    _require({2, 3, 4} <= {n for r in records for n in r['populations']}, "population coverage")
    _require({1, 2, 3, 4} <= {n for r in records for n in r['ranks']}, "ordinal coverage")
    for name in ('cooperative', 'shared_gate', 'internal_open', 'old_area', 'irregular', 'clipped'):
        _require(sum(bool(r[name]) for r in records) >= 3, "missing feature " + name)
    _require(any(not r['clipped'] for r in records), "clipping must be optional")
    _require(any(r['fresh_area'] == 0 for r in records), "no ineffective closure scene")
    _require(any(len(r['axes']) == 2 for r in records), "no mixed gate axes")
    _require(len({r['palette'] for r in records}) >= 30, "palette role diversity")

    # Every legal count request, and strict rejection beyond integer ranges.
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + nt * 10 + nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, "counts not honored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")

    # Covariance: no stored palette, orientation, or generation state is needed.
    pair = puzzles[0]['train'][0]
    rename = np.array([8, 2, 7, 0, 5, 1, 9, 3, 6, 4], np.int8)
    for rotation in range(4):
        for flip in (False, True):
            a = np.rot90(np.fliplr(pair['input']) if flip else pair['input'], rotation)
            b = np.rot90(np.fliplr(pair['output']) if flip else pair['output'], rotation)
            _require(np.array_equal(solve(rename[a]), rename[b]), "geometric/color covariance")

    # All semantic zones are checked by deliberate label damage.
    a, d = _decode(pair['input'])
    z = _zones(a, d)
    kept_gate = _mark(a.shape, [q for g in d['gates'] if g not in z['selected'] for q in g['cells']])
    key_mask = _mark(a.shape, [q for key in d['keys'].values() for q in key])
    far = (a == d['background']) & ~z['old'] & ~z['fresh'] & ~z['outline']
    checks = [z['closure'], z['fresh'], z['outline'], z['old'] & (a == d['background']),
              kept_gate, key_mask, z['original'], far]
    for mask in checks:
        _require(mask.any(), "mutation witness missing")
        bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
        r, c = np.argwhere(mask)[0]
        bad['train'][0]['output'][r, c] = (int(bad['train'][0]['output'][r, c]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted semantic-zone damage")

    # An open one-cell bay becomes visually blocked by the outline. Its
    # still-exterior interior must remain unfilled after that thickening.
    rng = np.random.default_rng(904)
    palette = [0, 5, 6, 2, 3, 1, 4, 7, 8, 9]
    wall, rooms, edges = _network(rng, (1, 3))
    gate_edges = [i for i in range(len(edges)) if edges[i]['axis'] == 1 and edges[i]['line'] == 0]
    world, gate_mask = _cut(rng, wall, edges, [(e, 2, n) for e, n in zip(gate_edges, (3, 4, 5))], palette)
    # A plain background doorway is allowed elsewhere in the wall network.
    r0, r1, c0, c1 = rooms[0]
    world[r1, (c0 + c1) // 2] = 0
    regression = _pair(world, {2: 2}, palette, rng)
    a, d = _decode(regression['input'])
    z = _zones(a, d)
    visually_enclosed = _holes(z['final'] | z['outline']) & z['exterior'] & (a == d['background'])
    _require(visually_enclosed.any(), "outline-order regression not exercised")
    _require(np.all(regression['output'][visually_enclosed] == d['background']), "outline triggered filling")

    return dict(seeds=42, pairs=len(records), normalized_structures=len(normalized),
                shapes=len({r['shape'] for r in records}),
                family_counts=sorted({r['families'] for r in records}),
                populations=sorted({n for r in records for n in r['populations']}),
                ranks=sorted({n for r in records for n in r['ranks']}),
                features={name: sum(bool(r[name]) for r in records)
                          for name in ('cooperative', 'shared_gate', 'internal_open', 'old_area',
                                       'irregular', 'clipped')})


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
