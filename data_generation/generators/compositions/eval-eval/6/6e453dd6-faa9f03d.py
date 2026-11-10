"""Socket docking: leading cavities absorb selected, reconstructed wires.

Only Python's standard library and NumPy are used. The solver has no access to
generation state, examples, labels, random seeds, or external files.
"""
from collections import defaultdict
import numpy as np

REPAIR, SWITCH = 8, 9
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 215340168, 773607699, 776404878, 467263295, 186554877,
              828398191, 206290452, 168981438, 284678754, 557523937,
              631796834, 439673819, 666120455, 356489700, 90538546,
              961929318, 990751534, 136468097, 602943567, 829643682,
              476243233, 592911104, 933532571, 511652403, 273765360,
              436370414, 750918033, 496951576, 776142046, 497648578,
              831145763, 591925044, 275591762, 357980423, 805017329,
              553575654, 227110156, 829600466, 254543079, 733315169,
              828393639)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    return int(value)


def _check_grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), 'invalid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'grid must contain integers')
    _require(bool(np.all((a >= 0) & (a <= 9))), 'invalid color')
    return a.astype(np.int8, copy=True)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _neighbors(p, cells):
    return [q for d in DIRS if (q := _add(p, d)) in cells]


def _components(mask):
    pending = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while pending:
        p = min(pending)
        pending.remove(p)
        stack, cells = [p], {p}
        while stack:
            for q in _neighbors(stack.pop(), pending):
                pending.remove(q)
                cells.add(q)
                stack.append(q)
        result.append(cells)
    return result


def _holes(mask):
    h, w = mask.shape
    return [s for s in _components(~mask)
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in s)]


def _canonical(inp):
    _require(len({int(inp[p]) for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))}) == 1,
             'corners must establish the background')
    bg = int(inp[0, 0])
    candidates = []
    for turn in range(4):
        a = np.rot90(inp, turn).copy()
        h, w = a.shape
        for col in range(1, w - 1):
            rail = int(a[0, col])
            if rail in (bg, REPAIR, SWITCH):
                continue
            if not np.all(a[:, col] == rail) or np.count_nonzero(a == rail) != h:
                continue
            if np.any(np.isin(a[:, :col], (REPAIR, SWITCH))):
                continue
            if not np.any(a[:, :col] != bg) or not np.any(a[:, col + 1:] == SWITCH):
                continue
            candidates.append((a, turn, bg, rail, col))
    _require(len(candidates) == 1, 'rail-relative orientation must be unique')
    return candidates[0]


def _bodies(a, bg, col):
    bodies = []
    for cells in _components(a[:, :col] != bg):
        rr, cc = zip(*cells)
        r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
        colors = {int(a[p]) for p in cells}
        _require(len(colors) == 1, 'body must be monochrome')
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        cavities = _holes(mask)
        active = [s for s in cavities if any(c == mask.shape[1] - 2 for r, c in s)]
        recessed = [s for s in cavities if s not in active]
        for cavity in active:
            rows = {r for r, c in cavity}
            _require(rows == {r for r, c in cavity if c == mask.shape[1] - 2},
                     'partial-leading cavities are outside this family')
            for r in rows:
                cs = [c for rr, c in cavity if rr == r]
                _require(len(cs) == max(cs) - min(cs) + 1, 'nonconvex socket slice')
        enclosed = set().union(*cavities) if cavities else set()
        open_rows = {r0 + r for r in range(mask.shape[0]) if mask.shape[1] >= 3 and
                     not mask[r, -2] and (r, mask.shape[1] - 2) not in enclosed}
        bodies.append(dict(cells=cells, mask=mask, color=colors.pop(), r0=r0,
                           r1=r1, c0=c0, shift=col - 1 - c1, active=active,
                           recessed=recessed, open_rows=open_rows))
    for x, y in zip(bodies, bodies[1:]):
        _require(y['r0'] > x['r1'] + 1, 'body bands must be separated')
    return bodies


def _reconstruct(a, bg, rail, col):
    """Marked waypoints and at most two missing straight pixels, input only."""
    work = a.copy()
    colors = set(map(int, np.unique(a[:, col + 1:]))) - {bg, rail, REPAIR, SWITCH}
    pending = {tuple(map(int, p)) for p in np.argwhere(a == REPAIR)}
    for _ in range(len(pending) + 1):
        changed = False
        for p in sorted(pending):
            witnessed = {int(work[q]) for d in DIRS
                         if _inside(q := _add(p, d), a.shape) and q[1] > col and
                         int(work[q]) in colors}
            if len(witnessed) == 1:
                work[p] = next(iter(witnessed))
                pending.remove(p)
                changed = True
        if not changed:
            break
    _require(not pending, 'repair mark lacks a unique adjacent color witness')
    layers = {c: {tuple(map(int, p)) for p in np.argwhere(work == c) if p[1] > col}
              for c in colors}
    for color, cells in layers.items():
        visible = set(cells)
        for p in sorted(visible):
            for d in ((1, 0), (0, 1)):
                for distance in (2, 3):
                    q = p[0] + distance * d[0], p[1] + distance * d[1]
                    if q not in visible:
                        continue
                    perpendicular = (-d[1], d[0])
                    turns = lambda x: any(_add(x, t) in visible for t in
                                          (perpendicular, (-perpendicular[0], -perpendicular[1])))
                    if turns(p) and turns(q):
                        continue
                    between = _line(p, q)[1:-1]
                    if any(a[x] == SWITCH or int(work[x]) not in (bg, color) for x in between):
                        continue
                    cells.update(between)
    return layers


def _line(p, q):
    _require(p[0] == q[0] or p[1] == q[1], 'nonorthogonal segment')
    dr, dc = int(np.sign(q[0] - p[0])), int(np.sign(q[1] - p[1]))
    return [(p[0] + i * dr, p[1] + i * dc)
            for i in range(max(abs(q[0] - p[0]), abs(q[1] - p[1])) + 1)]


def _polyline(points):
    result = [points[0]]
    for p, q in zip(points, points[1:]):
        result.extend(_line(p, q)[1:])
    return result


def _turns(walk):
    ds = [(q[0] - p[0], q[1] - p[1]) for p, q in zip(walk, walk[1:])]
    return sum(a != b for a, b in zip(ds, ds[1:]))


def _wires(a, layers, col):
    occupied = set().union(*layers.values()) if layers else set()
    _require(occupied, 'no wires')
    mask = np.zeros(a.shape, dtype=bool)
    for p in occupied:
        mask[p] = True
    components = _components(mask)
    owner = {}
    for i, cells in enumerate(components):
        cs = [c for c, s in layers.items() if cells & s]
        _require(len(cs) == 1, 'input paths may not touch or cross')
        _require(all(len(_neighbors(p, cells)) <= 2 for p in cells), 'ambiguous path junction')
        _require(sum(len(_neighbors(p, cells)) == 1 for p in cells) == 2, 'wire must be a simple path')
        for p in cells:
            owner[p] = i
    wires, used = [], set()
    for mark0 in np.argwhere(a == SWITCH):
        mark = tuple(map(int, mark0))
        pivot, start = (mark[0], mark[1] + 1), (mark[0], mark[1] + 2)
        _require(pivot in owner, 'switch lacks pivot')
        i = owner[pivot]
        _require(i not in used, 'one switch per input path')
        used.add(i)
        cells = components[i]
        _require(start in cells and mark not in cells, 'switch must oppose old branch')
        _require(len(_neighbors(pivot, cells)) == 2 and
                 any((pivot[0] + dr, pivot[1]) in cells for dr in (-1, 1)), 'pivot must turn')
        _require(_neighbors(mark, occupied) == [pivot], 'switch must have one adjacent wire')
        previous, current, walk = pivot, start, [pivot]
        while True:
            _require(current not in walk, 'old branch cycles')
            walk.append(current)
            following = [p for p in _neighbors(current, cells) if p != previous]
            if not following:
                _require(current[0] in (0, a.shape[0] - 1) or current[1] == a.shape[1] - 1,
                         'old branch must reach a wire-side grid boundary')
                break
            _require(len(following) == 1, 'old branch is ambiguous')
            previous, current = current, following[0]
        color = next(c for c, s in layers.items() if pivot in s)
        wires.append(dict(mark=mark, pivot=pivot, color=color, cells=cells,
                          old=set(walk[1:]), walk=walk))
    _require(len(used) == len(components), 'every input path requires one switch')
    return wires


def _ports(bodies):
    ports = {}
    for body in bodies:
        for cavity in body['active']:
            for r in {r for r, c in cavity}:
                _require(body['r0'] + r not in ports, 'competing body slices')
                ports[body['r0'] + r] = body['c0'] + body['shift'] + min(c for rr, c in cavity if rr == r)
    return ports


def _plans(bodies, wires):
    ports, by_row = _ports(bodies), defaultdict(list)
    for i, wire in enumerate(wires):
        by_row[wire['pivot'][0]].append(i)
    plans = []
    for row, left in sorted(ports.items()):
        if by_row[row]:
            i = min(by_row[row], key=lambda k: wires[k]['pivot'][1])
            plans.append((i, row, left))
    return plans


def _render(shape, bg, rail, col, bodies, wires, plans):
    out = np.full(shape, bg, dtype=np.int8)
    out[:, col] = rail
    for body in bodies:
        for r, c in body['cells']:
            out[r, c + body['shift']] = body['color']
    selected = {i for i, row, left in plans}
    for i, wire in enumerate(wires):
        kept = wire['cells'] - wire['old'] if i in selected else wire['cells']
        for p in kept:
            out[p] = wire['color']
    for i, row, left in plans:
        out[row, left:wires[i]['pivot'][1] + 1] = wires[i]['color']
    return out


def _parse(inp):
    a, turn, bg, rail, col = _canonical(inp)
    bodies = _bodies(a, bg, col)
    layers = _reconstruct(a, bg, rail, col)
    wires = _wires(a, layers, col)
    plans = _plans(bodies, wires)
    return a, turn, bg, rail, col, bodies, wires, plans


def solve(input_grid: np.ndarray) -> np.ndarray:
    inp = _check_grid(input_grid)
    a, turn, bg, rail, col, bodies, wires, plans = _parse(inp)
    return np.rot90(_render(a.shape, bg, rail, col, bodies, wires, plans), -turn).copy()


def _rand(rng, low, high):
    return int(rng.integers(low, high + 1))


def _body_mask(rng, kind, height=None):
    h = height if height is not None else _rand(rng, 3, 7)
    width = _rand(rng, 3, 7)
    if kind == 'solid':
        mask = np.ones((h, width), dtype=bool)
        if rng.random() < .6:
            for r in range(h):
                mask[r, :_rand(rng, 0, width - 2)] = False
        return mask
    mask = np.ones((h, width), dtype=bool)
    if kind == 'multi':
        _require(h >= 5, 'multi body too short')
        separator = _rand(rng, 2, h - 3)
        intervals = [(1, separator), (separator + 1, h - 1)]
    else:
        intervals = [(1, h - 1)]
    # Width and stepped cavity back edges are independent of height/count.
    for first, stop in intervals:
        back = _rand(rng, 1, width - 2)
        stepped = stop - first > 1 and width > 3 and rng.random() < .5
        for r in range(first, stop):
            left = _rand(rng, 1, width - 2) if stepped else back
            mask[r, left:width - 1] = False
    if kind == 'open':
        # Deep openings on back, top, or bottom, all reaching the leading slice.
        side = _rand(rng, 0, 2)
        if side == 0:
            mask[1:-1, :width - 1] = False
        else:
            mask[0 if side == 1 else -1, width - 2] = False
    elif kind == 'recessed':
        extension = _rand(rng, 1, 3)
        extended = np.zeros((h, width + extension), dtype=bool)
        extended[:, :width] = mask
        extended[0 if rng.random() < .5 else -1, width:] = True
        mask = extended
    elif kind == 'mixed':
        # Two compartments: a leading socket and a separately recessed cavity.
        extra = _rand(rng, 2, 4)
        extended = np.ones((h, width + extra), dtype=bool)
        extended[:, extra:] = mask
        extended[1:-1, 1:extra] = False
        mask = extended
    return mask


def _palette(rng):
    colors = list(map(int, rng.permutation(8)))
    return colors[0], colors[1], colors[2:]


def _place_bodies(rng, shape, bg, rail, col, masks, starts):
    a = np.full(shape, bg, dtype=np.int8)
    a[:, col] = rail
    payloads = [c for c in range(8) if c not in (bg, rail)]
    for mask, row in zip(masks, starts):
        limit = col - mask.shape[1]
        _require(limit >= 0, 'body cannot fit')
        shift = 0 if rng.random() < .25 else _rand(rng, 1, max(1, limit))
        if limit == 0:
            shift = 0
        start = col - mask.shape[1] - shift
        color = int(rng.choice(payloads))
        a[row:row + mask.shape[0], start:start + mask.shape[1]][mask] = color
    bodies = _bodies(a, bg, col)
    return a, bodies


def _valid_wire(points, pivot, shape, col, existing, marks):
    walk = _polyline(points)
    cells = set(walk)
    mark = (pivot[0], pivot[1] - 1)
    if len(cells) != len(walk) or not all(_inside(p, shape) and p[1] > col for p in cells):
        return None
    if any(len(_neighbors(p, cells)) > 2 for p in cells) or mark in cells:
        return None
    if cells & existing or any(_neighbors(p, existing) for p in cells):
        return None
    if cells & set(marks) or any(_neighbors(p, cells) for p in marks) or mark in existing or mark in marks:
        return None
    if _neighbors(mark, cells) != [pivot] or _neighbors(mark, existing):
        return None
    if any(p == mark or mark in _neighbors(p, {mark}) for p in marks):
        return None
    at = walk.index(pivot)
    old_walk = walk[at:]
    if old_walk[1] != (pivot[0], pivot[1] + 1):
        return None
    return dict(mark=mark, pivot=pivot, cells=cells, old=set(old_walk[1:]), walk=old_walk)


def _random_wire(rng, shape, col, row, existing, marks):
    h, w = shape
    for _ in range(350):
        c = _rand(rng, col + 3, w - 4)
        pivot = row, c
        direction = int(rng.choice((-1, 1)))
        length = _rand(rng, 1, 4)
        rret = row + direction * length
        if not 1 <= rret < h - 1:
            continue
        # Retained side lengths, turns and end positions vary independently.
        retained = [(rret, c), pivot]
        if rng.random() < .7:
            endc = _rand(rng, col + 1, w - 2)
            if abs(endc - c) >= 2:
                retained.insert(0, (rret, endc))
        if len(retained) == 3 and rng.random() < .3:
            endr = _rand(rng, 1, h - 2)
            if abs(endr - rret) >= 2:
                retained.insert(0, (endr, retained[0][1]))
        mode = _rand(rng, 0, 2)
        if mode == 0:
            old = [pivot, (row, w - 1)]
        else:
            bendc = _rand(rng, c + 2, w - 2)
            side = int(rng.choice((-1, 1)))
            if mode == 1:
                old = [pivot, (row, bendc), (0 if side < 0 else h - 1, bendc)]
            else:
                bendr = row + side * _rand(rng, 2, 5)
                if not 1 <= bendr < h - 1:
                    continue
                old = [pivot, (row, bendc), (bendr, bendc), (bendr, w - 1)]
        wire = _valid_wire(retained + old[1:], pivot, shape, col, existing, marks)
        if wire is not None:
            return wire
    return None


def _draw_latent(a, wires, colors):
    for wire, color in zip(wires, colors):
        wire['color'] = color
        for p in wire['cells']:
            a[p] = color
        a[wire['mark']] = SWITCH


def _damage(rng, a, bg, rail, col, wires, required_pivot=False):
    """Keep only damage whose pixel reconstruction equals the latent network."""
    expected = {color: set().union(*(w['cells'] for w in wires if w['color'] == color))
                for color in {w['color'] for w in wires}}
    if _reconstruct(a, bg, rail, col) != expected:
        return None
    result = a.copy()
    candidates = list(set().union(*(w['cells'] for w in wires)))
    rng.shuffle(candidates)
    if required_pivot:
        candidates = [wires[0]['pivot']] + [p for p in candidates if p != wires[0]['pivot']]
    desired = _rand(rng, 2, min(9, max(2, len(candidates) // 5)))
    done = 0
    for p in candidates:
        old = int(result[p])
        result[p] = REPAIR
        try:
            accepted = _reconstruct(result, bg, rail, col) == expected
        except AssertionError:
            accepted = False
        if accepted:
            done += 1
            if done >= desired:
                break
        else:
            result[p] = old
    if done == 0 or required_pivot and result[wires[0]['pivot']] != REPAIR:
        return None
    candidates = [p for p in candidates if result[p] != REPAIR]
    rng.shuffle(candidates)
    desired = _rand(rng, 1, 4)
    done = 0
    for p in candidates:
        old = int(result[p])
        result[p] = bg
        try:
            accepted = _reconstruct(result, bg, rail, col) == expected
        except AssertionError:
            accepted = False
        if accepted:
            done += 1
            if done >= desired:
                break
        else:
            result[p] = old
    # Propose a neighboring second gap to exercise two-cell completion.
    if rng.random() < .5:
        for p in candidates:
            if result[p] != bg or p not in set().union(*expected.values()):
                continue
            for d in DIRS:
                q = _add(p, d)
                if not _inside(q, a.shape) or q[1] <= col or int(result[q]) not in expected:
                    continue
                old = int(result[q])
                result[q] = bg
                try:
                    accepted = _reconstruct(result, bg, rail, col) == expected
                except AssertionError:
                    accepted = False
                if not accepted:
                    result[q] = old
                else:
                    return result
    return result


def _witness(rng):
    """Geometric proof scene: competition, color recovery, crossing, open decoy.

    Coordinates and dimensions are sampled; the three paths' relationships are
    coverage witnesses. Other examples use independently sampled polylines.
    """
    bg, rail, payloads = _palette(rng)
    masks = [_body_mask(rng, 'active', _rand(rng, 5, 7)),
             _body_mask(rng, 'open', _rand(rng, 3, 6))]
    h = _rand(rng, 23, 30)
    col = _rand(rng, max(8, max(m.shape[1] for m in masks) + 1), 13)
    w = _rand(rng, col + 15, 30)
    top = _rand(rng, 2, 4)
    row = top + 3
    lower = _rand(rng, max(top + masks[0].shape[0] + 2, row + 6), h - masks[1].shape[0] - 2)
    a, bodies = _place_bodies(rng, (h, w), bg, rail, col, masks, [top, lower])
    # Guarantee visible travel in this witness without fixing it by object rank.
    if not any(b['shift'] for b in bodies):
        return None
    crossc = col + _rand(rng, 1, 2)
    nearc = col + _rand(rng, 6, 7)
    bendc = nearc + 2
    farc = _rand(rng, col + 12, w - 3)
    up = row - _rand(rng, 4, min(5, row - 1))
    nearret = row - 2
    paths = [([(nearret, crossc + 2), (nearret, nearc), (row, nearc),
               (row, bendc), (h - 1, bendc)], (row, nearc)),
             ([(row + 2, crossc), (up, crossc), (up, farc), (row, farc),
               (row, w - 1)], (row, farc))]
    open_row = int(rng.choice(sorted(bodies[1]['open_rows'])))
    quietc = farc
    retrow = open_row + 2 if open_row + 2 < h - 1 else open_row - 2
    paths.append(([(retrow, quietc), (open_row, quietc), (open_row, w - 1)],
                  (open_row, quietc)))
    wires, occupied, marks = [], set(), []
    for points, pivot in paths:
        wire = _valid_wire(points, pivot, a.shape, col, occupied, marks)
        if wire is None:
            return None
        wires.append(wire)
        occupied.update(wire['cells'])
        marks.append(wire['mark'])
    rng.shuffle(payloads)
    if payloads[0] == bodies[0]['color']:
        payloads[0], payloads[1] = payloads[1], payloads[0]
    _draw_latent(a, wires, payloads[:3])
    expected = _render(a.shape, bg, rail, col, bodies, wires, _plans(bodies, wires))
    damaged = _damage(rng, a, bg, rail, col, wires, required_pivot=True)
    if damaged is None:
        return None
    return {'input': damaged, 'output': expected}


def _generic(rng, kinds, query=False):
    bg, rail, payloads = _palette(rng)
    masks = [_body_mask(rng, k, _rand(rng, 5, 8) if k == 'multi' else _rand(rng, 3, 7))
             for k in kinds]
    base = sum(m.shape[0] for m in masks) + 2 * (len(masks) - 1) + 2
    if base > 30:
        return None
    h = _rand(rng, max(base, 16), 30)
    widest = max(m.shape[1] for m in masks)
    col = _rand(rng, widest + 1, min(15, widest + 5))
    if col + 9 > 30:
        return None
    w = _rand(rng, col + 9, 30)
    slack = h - base
    extras = list(map(int, rng.multinomial(slack, np.full(len(masks) + 1, 1 / (len(masks) + 1)))))
    starts, row = [], 1 + extras[0]
    for i, mask in enumerate(masks):
        starts.append(row)
        row += mask.shape[0] + 2 + extras[i + 1]
    a, bodies = _place_bodies(rng, (h, w), bg, rail, col, masks, starts)
    rows = []
    for body, kind in zip(bodies, kinds):
        if body['active']:
            for hole in body['active']:
                rows.append(body['r0'] + int(rng.choice(sorted({r for r, c in hole}))))
        elif body['recessed']:
            rows.append(body['r0'] + int(rng.choice(sorted({r for s in body['recessed'] for r, c in s}))))
        elif body['open_rows']:
            rows.append(int(rng.choice(sorted(body['open_rows']))))
        else:
            rows.append(_rand(rng, body['r0'], body['r1']))
    # Sometimes a second row of one tall socket receives a separate colored wire.
    if len(rows) < 4 and rng.random() < .35:
        options = sorted(set(_ports(bodies)) - set(rows))
        if options:
            rows.append(int(rng.choice(options)))
    if len(rows) < 4 and rng.random() < .35:
        options = [r for r in range(2, h - 2) if all(abs(r - s) >= 3 for s in rows)]
        if options:
            rows.append(int(rng.choice(options)))
    # Hardest rows first to avoid making compact multi-hole geometry impossible.
    rows.sort(key=lambda r: min((abs(r - s) for s in rows if s != r), default=h))
    wires, occupied, marks = [], set(), []
    for row in rows:
        wire = _random_wire(rng, a.shape, col, row, occupied, marks)
        if wire is None:
            return None
        wires.append(wire)
        occupied.update(wire['cells'])
        marks.append(wire['mark'])
    rng.shuffle(payloads)
    _draw_latent(a, wires, payloads[:len(wires)])
    plans = _plans(bodies, wires)
    expected = _render(a.shape, bg, rail, col, bodies, wires, plans)
    damaged = _damage(rng, a, bg, rail, col, wires)
    if damaged is None:
        return None
    return {'input': damaged, 'output': expected}


def _transform(pair, turn, reflect):
    return {k: np.rot90(np.flipud(a) if reflect else a, turn).copy()
            for k, a in pair.items()}


def _signature(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _puzzle_signature(p):
    return tuple(tuple((_signature(q['input']), _signature(q['output'])) for q in p[s])
                 for s in ('train', 'test'))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed')
    num_train = _integer(num_train, 'num_train')
    num_test = _integer(num_test, 'num_test')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('counts require 2..10 train, 1..4 test, and train > test')
    rng = np.random.default_rng(seed % (1 << 64))
    shift_mode = str(rng.choice(('split_cavity', 'orientation')))
    base_turn, reflect = _rand(rng, 0, 3), bool(rng.integers(2))
    test_turn = (base_turn + _rand(rng, 1, 3)) % 4
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            for _ in range(180):
                if split == 'train' and i == 0:
                    pair = _witness(rng)
                elif split == 'train' and i == 1:
                    pair = _generic(rng, ['active', 'recessed'])
                elif split == 'test' and shift_mode == 'split_cavity':
                    kinds = ['multi', str(rng.choice(('open', 'recessed', 'solid')))]
                    pair = _generic(rng, kinds, query=True)
                else:
                    count_bodies = _rand(rng, 1, 3)
                    pool = ('active', 'active', 'open', 'recessed', 'solid', 'mixed')
                    kinds = [str(rng.choice(pool)) for _ in range(count_bodies)]
                    # Orientation queries still ask simultaneous, established operations.
                    if split == 'test':
                        kinds[0] = 'mixed' if rng.random() < .5 else 'multi'
                    pair = _generic(rng, kinds, query=split == 'test')
                if pair is None:
                    continue
                turn = test_turn if split == 'test' and shift_mode == 'orientation' else base_turn
                if shift_mode == 'split_cavity' and split == 'train' and i >= 2 and rng.random() < .4:
                    turn = _rand(rng, 0, 3)
                pair = _transform(pair, turn, reflect)
                try:
                    predicted = solve(pair['input'])
                except AssertionError:
                    continue
                if not np.array_equal(predicted, pair['output']):
                    continue
                sig = _signature(pair['input'])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded socket scene construction exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(inp):
    a, turn, bg, rail, col, bodies, wires, plans = _parse(inp)
    selected = {i for i, row, left in plans}
    active_rows = set(_ports(bodies))
    switch_rows = {w['pivot'][0] for w in wires}
    open_rows = set().union(*(b['open_rows'] for b in bodies))
    recessed_rows = {b['r0'] + r for b in bodies for hole in b['recessed'] for r, c in hole}
    connections = {i: {(row, c) for c in range(left, wires[i]['pivot'][1] + 1)}
                   for i, row, left in plans}
    kept = {i: w['cells'] - w['old'] if i in selected else w['cells'] for i, w in enumerate(wires)}
    crossings = sum(bool(points & kept[j]) for i, points in connections.items()
                    for j in kept if i != j)
    return dict(shape=inp.shape, orientation=turn, roles=(bg, rail), body_count=len(bodies),
                wire_count=len(wires), shifts=tuple(b['shift'] for b in bodies),
                active_counts=tuple(len(b['active']) for b in bodies),
                mask_signatures=tuple(_signature(b['mask']) for b in bodies),
                cavity_widths=tuple(max(c for r, c in hole) - min(c for r, c in hole) + 1
                                    for b in bodies for hole in b['active'] + b['recessed']),
                selected=len(selected), open_contrast=bool((open_rows - active_rows) & switch_rows),
                recessed_contrast=bool((recessed_rows - active_rows) & switch_rows),
                pivot_repair=any(a[wires[i]['pivot']] == REPAIR for i in selected),
                bent_deletion=any(_turns(wires[i]['walk']) > 0 for i in selected),
                competition=any(sum(w['pivot'][0] == row for w in wires) > 1 for i, row, left in plans),
                crossing=crossings, old_bends=tuple(_turns(w['walk']) for w in wires),
                old_lengths=tuple(len(w['old']) for w in wires),
                partial_fill=any(set(b['r0'] + r for r, c in hole) & switch_rows and
                                 set(b['r0'] + r for r, c in hole) - switch_rows
                                 for b in bodies for hole in b['active']),
                repair_count=int(np.count_nonzero(a == REPAIR)),
                gaps=sum(a[p] == bg for w in wires for p in w['cells']),
                new_old_crossing=any(points & wires[j]['old'] for i, points in connections.items()
                                     for j in selected if i != j))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'invalid counts')
    seen, training = set(), []
    ablation_a = ablation_b = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            inp, out = _check_grid(pair['input']), _check_grid(pair['output'])
            _require(inp.shape == out.shape, 'shape changed')
            sig = _signature(inp)
            _require(sig not in seen, 'duplicate input')
            seen.add(sig)
            canonical, turn, bg, rail, col, bodies, wires, plans = _parse(inp)
            actual = np.rot90(out, turn)
            expected = _render(canonical.shape, bg, rail, col, bodies, wires, plans)
            _require(np.array_equal(actual, expected), 'output violates socket rewiring')
            _require(not np.any(np.isin(out, (REPAIR, SWITCH))), 'annotation survived')
            selected = {i for i, row, left in plans}
            drawn = set().union(*({(row, c) for c in range(left, wires[i]['pivot'][1] + 1)}
                                  for i, row, left in plans)) if plans else set()
            for body in bodies:
                for r, c in body['cells']:
                    target = r, c + body['shift']
                    _require(target in drawn or actual[target] == body['color'], 'body shape changed')
            for i, wire in enumerate(wires):
                for p in wire['cells']:
                    if p in drawn:
                        continue
                    color = bg if i in selected and p in wire['old'] else wire['color']
                    _require(actual[p] == color, 'branch deletion or preservation failed')
            for row in range(actual.shape[0]):
                _require((row, col) in drawn or actual[row, col] == rail, 'unplugged rail changed')
            if split == 'train':
                training.append(_features(inp))
                # Remove A's topological authorization and activate every switch.
                all_plans = [(i, w['pivot'][0], col) for i, w in enumerate(wires)]
                ablation_a |= not np.array_equal(_render(canonical.shape, bg, rail, col,
                                                         bodies, wires, all_plans), actual)
                # Remove B's branch replacement and keep every old terminal.
                no_deletion = [dict(w, old=set()) for w in wires]
                ablation_b |= not np.array_equal(_render(canonical.shape, bg, rail, col,
                                                         bodies, no_deletion, plans), actual)
    for name in ('open_contrast', 'recessed_contrast', 'pivot_repair', 'bent_deletion',
                 'competition', 'partial_fill'):
        _require(any(f[name] for f in training), 'training lacks ' + name)
    _require(any(f['selected'] for f in training), 'no positive socket')
    _require(any(max(f['shifts']) > 0 for f in training), 'no docking travel')
    _require(any(f['crossing'] for f in training), 'connection crossing priority not taught')
    _require(ablation_a and ablation_b, 'parent ablation leaves demonstrations unchanged')


def self_test():
    """Bounded, pixel-derived checks across all 42 specified host seeds."""
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    records = [_features(np.asarray(pair['input'])) for puzzle in puzzles
               for split in ('train', 'test') for pair in puzzle[split]]
    _require(len({_puzzle_signature(p) for p in puzzles}) == 42, 'seed diversity collapsed')
    _require(len({f['shape'] for f in records}) >= 35, 'dimension diversity collapsed')
    _require(len({m for f in records for m in f['mask_signatures']}) >= 80, 'body geometry collapsed')
    _require({f['orientation'] for f in records} == set(range(4)), 'missing orientation')
    _require(len({f['roles'] for f in records}) >= 25, 'palette roles collapsed')
    _require({1, 2, 3} <= {f['body_count'] for f in records}, 'body counts collapsed')
    _require({1, 2, 3, 4} <= {f['wire_count'] for f in records}, 'wire counts collapsed')
    _require({0, 1, 2} <= {x for f in records for x in f['old_bends']}, 'old paths collapsed')
    _require(any(0 in f['shifts'] and max(f['shifts']) > 0 for f in records), 'no travel mixture')
    _require(any(max(f['active_counts']) >= 2 for f in records), 'no split sockets')
    _require(any(f['selected'] >= 2 for f in records), 'no simultaneous connections')
    _require(any(f['new_old_crossing'] for f in records), 'no simultaneous removal interaction')
    _require(max(x for f in records for x in f['cavity_widths']) >= 4, 'cavities too narrow')
    _require(sum(f['gaps'] > 0 for f in records) >= 100, 'short-gap reconstruction unused')
    _require(_puzzle_signature(generate_puzzle(99173, 7, 3)) ==
             _puzzle_signature(generate_puzzle(99173, 7, 3)), 'nondeterminism')
    # Boundary/coverage cases here; verify.py independently covers every legal
    # explicit combination, avoiding a second expensive complete count sweep.
    for a, b in ((2, 1), (3, 2), (5, 4), (10, 1), (10, 4)):
        p = generate_puzzle(123456 + 10 * a + b, a, b)
        _require((len(p['train']), len(p['test'])) == (a, b), 'count ignored')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # Equivariance and input nonmutation are semantic checks, not new fixtures.
    for p in puzzles[:3]:
        for pair in p['train'] + p['test']:
            original = pair['input'].copy()
            for turn in range(4):
                q = _transform(pair, turn, True)
                _require(np.array_equal(solve(q['input']), q['output']), 'symmetry failed')
            _require(np.array_equal(pair['input'], original), 'solver mutated input')
    # Corrupt preserved, plug, deletion and repaired cells; all must be rejected.
    for pair in puzzles[0]['train']:
        a, turn, bg, rail, col, bodies, wires, plans = _parse(pair['input'])
        sites = [(0, col)] + [wires[i]['pivot'] for i, row, left in plans]
        sites += [min(wires[i]['old']) for i, row, left in plans]
        sites += [(row, left) for i, row, left in plans]
        for site in sites:
            wrong = np.rot90(pair['output'], turn).copy()
            wrong[site] = (int(wrong[site]) + 1) % 10
            bad_pair = {'input': pair['input'], 'output': np.rot90(wrong, -turn).copy()}
            bad = {'train': [bad_pair if q is pair else q for q in puzzles[0]['train']],
                   'test': puzzles[0]['test']}
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('corrupt label accepted')
