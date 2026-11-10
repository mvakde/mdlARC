"""Seal-and-transfer: geometric plugs gate compartment-color transfer.

Only standard-library Python and NumPy are used. Prediction reads the input
alone. Generation labels are checked against prediction, never used by it.
"""
from __future__ import annotations

import itertools
import numpy as np

STEPS4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
STEPS8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _parts(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    answer = []
    steps = STEPS8 if diagonal else STEPS4
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    cells.append(p)
        answer.append(np.array(cells, dtype=int))
    return answer


def _bbox(points):
    r, c = points.min(0)
    R, C = points.max(0) + 1
    return int(r), int(c), int(R), int(C)


def _rooms(barriers):
    free = np.pad(~barriers, 1, constant_values=True)
    return [p - 1 for p in _parts(free)
            if np.all(p > 0) and np.all(p < np.array(free.shape) - 1)]


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _canonical(mask):
    """Return the unique occupied-cell silhouette with its neck facing down."""
    options = {}
    for turn in range(4):
        b = np.rot90(mask, turn).copy()
        h, w = b.shape
        if h < 2 or w < 4 or w % 2:
            continue
        neck = np.zeros(w, dtype=bool)
        neck[w // 2 - 1:w // 2 + 1] = True
        if np.array_equal(b[-1], neck):
            options[_key(b)] = b
    assert len(options) == 1, 'missing or competing attachment necks'
    return next(iter(options.values()))


def _dock(mask, side, gaps):
    neck = np.zeros_like(mask)
    neck[-1, mask.shape[1] // 2 - 1:mask.shape[1] // 2 + 1] = True
    b = np.rot90(mask, -side)
    n = np.argwhere(np.rot90(neck, -side))
    shift = gaps.min(0) - n.min(0)
    assert set(map(tuple, n + shift)) == set(map(tuple, gaps))
    return np.argwhere(b) + shift


def _relations(rooms):
    boxes = [_bbox(p) for p in rooms]
    return [[tuple(-1 if a[k + 2] <= b[k] else
                   1 if b[k + 2] <= a[k] else 0 for k in (0, 1))
             for b in boxes] for a in boxes]


def _mapping(donor_rooms, target_rooms, colors):
    """Exact ordered projection relations; all supported maps must agree."""
    n = len(donor_rooms)
    assert n == len(target_rooms) and 2 <= n <= 6
    a, b = _relations(donor_rooms), _relations(target_rooms)
    answers = set()
    for perm in itertools.permutations(range(n)):
        if all(a[i][j] == b[perm[i]][perm[j]]
               for i in range(n) for j in range(i)):
            result = [None] * n
            for i, j in enumerate(perm):
                result[j] = colors[i]
            answers.add(tuple(result))
    assert len(answers) == 1, 'room arrangement has ambiguous payload colors'
    return next(iter(answers))


def _extract(input_grid):
    g = _grid(input_grid)
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > g.size / 2, 'background must be the strict majority'
    parts = _parts(g != bg, diagonal=True)
    # The donor is the unique complete framework whose non-wall pixels are
    # exactly two or more uniformly colored, enclosed rooms. Open target
    # corners may be separate components and must not be mistaken for keys.
    candidates = []
    for cells in parts:
        vals = set(map(int, g[tuple(cells.T)]))
        if len(vals) < 3:
            continue
        r, c, R, C = _bbox(cells)
        crop = np.full((R - r, C - c), bg, dtype=np.int8)
        crop[cells[:, 0] - r, cells[:, 1] - c] = g[tuple(cells.T)]
        for wall in sorted(vals):
            rooms = _rooms(crop == wall)
            if not 2 <= len(rooms) <= 6:
                continue
            colors, valid = [], True
            for room in rooms:
                v = np.unique(crop[tuple(room.T)])
                if len(v) != 1 or int(v[0]) in (bg, wall):
                    valid = False
                    break
                colors.append(int(v[0]))
            if (valid and len(set(colors)) >= 2 and
                    sum(map(len, rooms)) == np.count_nonzero((crop != bg) & (crop != wall))):
                candidates.append((cells, crop, wall, rooms, colors))
    assert len(candidates) == 1, 'one visibly colored donor is required'
    donor_cells, donor, wall, dr, colors = candidates[0]
    remaining = g != bg
    remaining[tuple(donor_cells.T)] = False
    framework = remaining & (g == wall)
    assert framework.any(), 'missing target framework'
    r, c, R, C = _bbox(np.argwhere(framework))
    target = g[r:R, c:C].copy()
    # Generation guarantees other objects do not occupy this rectangle.
    assert not set(map(tuple, donor_cells)) & set(itertools.product(range(r, R), range(c, C)))
    labels = sorted(set(map(int, target.flat)) - {bg, wall})
    assert 2 <= len(labels) <= 4 and not set(labels) & set(colors)
    sealed = target != bg
    ports = []
    for color in labels:
        markers = np.argwhere(target == color)
        assert len(markers) == 2
        delta = markers[1] - markers[0]
        assert sorted(map(abs, delta)) == [0, 3], 'port brackets flank a two-cell gap'
        unit = delta // 3
        gaps = np.array([markers[0] + unit, markers[0] + 2 * unit])
        assert np.all(target[tuple(gaps.T)] == bg)
        sealed[tuple(gaps.T)] = True
        ports.append(dict(color=color, markers=markers, gaps=gaps))
    tr = _rooms(sealed)
    payload = _mapping(dr, tr, colors)
    lookup = np.full(target.shape, -1, dtype=int)
    for i, room in enumerate(tr):
        lookup[tuple(room.T)] = i
    for port in ports:
        choices = []
        for side, (out_r, out_c) in enumerate(STEPS4):
            gap = port['gaps']
            inside = gap - (out_r, out_c)
            outside = gap + (out_r, out_c)
            if not np.all((inside >= 0) & (inside < np.array(target.shape))):
                continue
            ii = lookup[tuple(inside.T)]
            if ii[0] < 0 or not np.all(ii == ii[0]):
                continue
            if np.all((outside >= 0) & (outside < np.array(target.shape))):
                if np.any(lookup[tuple(outside.T)] >= 0) or np.any(sealed[tuple(outside.T)]):
                    continue
            choices.append((side, int(ii[0])))
        assert len(choices) == 1, 'port must open exactly one room to exterior'
        port['side'], port['room'] = choices[0]
        port['absolute_gaps'] = port['gaps'] + (r, c)
    remaining[r:R, c:C] = False
    keys = []
    for cells in _parts(remaining, diagonal=True):
        values = np.unique(g[tuple(cells.T)])
        assert len(values) == 1 and values[0] != wall
        rr, cc, RR, CC = _bbox(cells)
        mask = np.zeros((RR - rr, CC - cc), dtype=bool)
        mask[cells[:, 0] - rr, cells[:, 1] - cc] = True
        keys.append(dict(color=int(values[0]), cells=cells, mask=mask))
    return dict(grid=g, bg=bg, wall=wall, donor=donor, donor_bbox=_bbox(donor_cells), donor_rooms=dr,
                donor_colors=colors, target=target, target_bbox=(r, c, R, C),
                rooms=tr, payload=payload, ports=ports, keys=keys)


def _predict(input_grid):
    info = _extract(input_grid)
    g, bg = info['grid'], info['bg']
    r, c, R, C = info['target_bbox']
    out = np.full_like(g, bg)
    out[r:R, c:C] = info['target']
    matched, used = [], set()
    port_by_color = {p['color']: p for p in info['ports']}
    for key in info['keys']:
        if key['color'] not in port_by_color:
            continue
        assert key['color'] not in used, 'at most one matching key per port'
        used.add(key['color'])
        port = port_by_color[key['color']]
        canonical = _canonical(key['mask'])
        coords = _dock(canonical, port['side'], port['absolute_gaps'])
        assert np.all((coords >= 0) & (coords < np.array(g.shape))), 'no clipping'
        assert np.all(out[tuple(coords.T)] == bg), 'docked keys may not overlap'
        out[tuple(coords.T)] = key['color']
        rotations = {t for t in range(4) if np.array_equal(np.rot90(key['mask'], t),
                                                         np.rot90(canonical, -port['side']))}
        matched.append(dict(key=key, port=port, canonical=canonical,
                            coords=coords, rotations=rotations))
    open_rooms = {p['room'] for p in info['ports'] if p['color'] not in used}
    actual_rooms = _rooms(out != bg)
    closed = set(range(len(info['rooms']))) - open_rooms
    expected_sets = {frozenset(map(tuple, info['rooms'][i] + (r, c))) for i in closed}
    assert {frozenset(map(tuple, p)) for p in actual_rooms} == expected_sets, 'keys must only close marked openings'
    for i in closed:
        coords = info['rooms'][i] + (r, c)
        out[tuple(coords.T)] = info['payload'][i]
    rr, cc, RR, CC = _bbox(np.argwhere(out != bg))
    answer = out[rr:RR, cc:CC].copy()
    info.update(matched=matched, selected=used, open_rooms=open_rooms, closed=closed,
                output_bbox=(rr, cc, RR, CC), assembled=out)
    return answer, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from visible geometry and colors, with no state."""
    return _predict(input_grid)[0]


def _plan(rng, shape, count):
    plan = np.arange(np.prod(shape)).reshape(shape)
    while len(set(plan.flat)) > count:
        edges = [(int(plan[r, c]), int(plan[r + dr, c + dc]))
                 for r in range(shape[0]) for c in range(shape[1])
                 for dr, dc in ((1, 0), (0, 1))
                 if r + dr < shape[0] and c + dc < shape[1] and plan[r, c] != plan[r + dr, c + dc]]
        a, b = edges[int(rng.integers(len(edges)))]
        plan[plan == b] = a
    for i, old in enumerate(sorted(set(plan.flat))):
        plan[plan == old] = i
    return plan


def _geometry(rng, plan, widths, heights, detail):
    ys = np.r_[0, np.cumsum(np.array(heights) + 1)]
    xs = np.r_[0, np.cumsum(np.array(widths) + 1)]
    a = np.full((ys[-1] + 1, xs[-1] + 1), -1, dtype=int)
    for r, c in itertools.product(range(plan.shape[0]), range(plan.shape[1])):
        label = plan[r, c]
        a[ys[r] + 1:ys[r + 1], xs[c] + 1:xs[c + 1]] = label
        if r and plan[r - 1, c] == label:
            a[ys[r], xs[c] + 1:xs[c + 1]] = label
        if c and plan[r, c - 1] == label:
            a[ys[r] + 1:ys[r + 1], xs[c]] = label
        if r and c and np.all(plan[r - 1:r + 1, c - 1:c + 1] == label):
            a[ys[r], xs[c]] = label
    for label in sorted(set(plan.flat)):
        for _ in range(detail):
            cells = np.argwhere(a == label)
            if len(cells) < 6 or rng.random() < .5:
                continue
            r, c, R, C = _bbox(cells)
            choices = [(r, c), (r, C - 1), (R - 1, c), (R - 1, C - 1)]
            point = choices[int(rng.integers(4))]
            if a[point] != label:
                continue
            trial = a == label
            trial[point] = False
            pieces = _parts(trial)
            if len(pieces) == 1 and _bbox(pieces[0]) == (r, c, R, C):
                a[point] = -1
    return a


def _paint(labels, bg, wall, colors=None):
    a = np.full(labels.shape, wall, dtype=np.int8)
    for i in sorted(set(labels.flat) - {-1}):
        a[labels == i] = bg if colors is None else colors[i]
    return a


def _cap(rng, width=None, asymmetric=False):
    for _ in range(32):
        w = int(width or rng.choice((4, 6)))
        stem = int(rng.integers(1, 3))
        body = 2 if asymmetric else int(rng.integers(1, 3))
        mask = np.zeros((stem + body, w), dtype=bool)
        mask[:body] = True
        mask[body:, w // 2 - 1:w // 2 + 1] = True
        if body == 2 and (asymmetric or rng.random() < .6):
            mask[0, int(rng.choice((0, w - 1)))] = False
        try:
            canonical = _canonical(mask)
            if np.array_equal(mask, canonical):
                return mask
        except AssertionError:
            pass
    return np.array([[1, 1, 1, 1], [0, 1, 1, 0]], dtype=bool)


def _port_candidates(labels):
    h, w = labels.shape
    sites = []
    for side in range(4):
        length = w if side % 2 == 0 else h
        for pos in range(2, length - 3):
            if side == 0:
                gaps = np.array([[0, pos], [0, pos + 1]])
            elif side == 1:
                gaps = np.array([[pos, w - 1], [pos + 1, w - 1]])
            elif side == 2:
                gaps = np.array([[h - 1, pos], [h - 1, pos + 1]])
            else:
                gaps = np.array([[pos, 0], [pos + 1, 0]])
            inside = gaps - STEPS4[side]
            v = labels[tuple(inside.T)]
            if v[0] >= 0 and v[0] == v[1]:
                unit = gaps[1] - gaps[0]
                markers = np.array([gaps[0] - unit, gaps[1] + unit])
                if np.all(labels[tuple(markers.T)] == -1):
                    sites.append(dict(side=side, room=int(v[0]), gaps=gaps, markers=markers))
    return sites


def _choose_ports(rng, labels, mode):
    sites = _port_candidates(labels)
    upper_corner = mode == 0 or (mode == 3 and rng.random() < .5)
    sides = (0, 3) if upper_corner else (1, 2)
    corner_room = int(labels[1, 1] if upper_corner else labels[-2, -2])
    chosen = []
    for side in sides:
        options = [p for p in sites if p['room'] == corner_room and p['side'] == side]
        assert options, 'corner needs two independent exterior ports'
        chosen.append(options[int(rng.integers(len(options)))])
    if mode == 2 and rng.random() < .5:
        # Tests have two double-opening rooms on opposite corners.
        other = int(labels[1, 1])
        assert other != corner_room
        for side in (0, 3):
            options = [p for p in sites if p['room'] == other and p['side'] == side]
            assert options
            chosen.append(options[int(rng.integers(len(options)))])
    elif mode < 2:
        opposite = 2 if mode == 0 else 0
        options = [p for p in sites if p['room'] != corner_room and p['side'] == opposite]
        assert options
        chosen.append(options[int(rng.integers(len(options)))])
    else:
        desired = 4 if mode == 2 else int(rng.integers(2, 5))
        for index in rng.permutation(len(sites)):
            if len(chosen) == desired:
                break
            candidate = sites[int(index)]
            cells = set(map(tuple, np.r_[candidate['markers'], candidate['gaps']]))
            if all(cells.isdisjoint(set(map(tuple, np.r_[p['markers'], p['gaps']]))) for p in chosen):
                chosen.append(candidate)
        assert len(chosen) == desired
    cells = [set(map(tuple, np.r_[p['markers'], p['gaps']])) for p in chosen]
    assert all(cells[i].isdisjoint(cells[j]) for i in range(len(cells)) for j in range(i))
    return chosen


def _place(rng, target, donor, keys, relative_caps, bg):
    th, tw = target.shape
    points = [np.argwhere(target != bg)] + relative_caps
    all_points = np.concatenate(points)
    low = -all_points.min(0)
    maxpoint = all_points.max(0)
    min_shape = np.maximum(maxpoint + low + 1, (22, 22))
    for attempt in range(60):
        shape = tuple(int(rng.integers(int(x), 31)) if attempt < 35 else 30 for x in min_shape)
        shift = np.array([int(rng.integers(int(a), int(b) + 1))
                          for a, b in zip(low, np.array(shape) - 1 - maxpoint)])
        r, c = shift
        g = np.full(shape, bg, dtype=np.int8)
        g[r:r + th, c:c + tw] = target
        blocked = np.zeros(shape, dtype=bool)
        blocked[max(0, r - 1):min(shape[0], r + th + 1),
                max(0, c - 1):min(shape[1], c + tw + 1)] = True
        objects = [(donor != bg, donor)] + [(k['mask'], k['color']) for k in keys]
        success = True
        for mask, values in objects:
            h, w = mask.shape
            if h > shape[0] or w > shape[1]:
                success = False
                break
            local = np.argwhere(mask)
            tops, lefts = np.indices((shape[0] - h + 1, shape[1] - w + 1))
            rr = tops.ravel()[:, None] + local[:, 0]
            cc = lefts.ravel()[:, None] + local[:, 1]
            good = np.flatnonzero(~np.any(blocked[rr, cc], axis=1))
            if not len(good):
                success = False
                break
            index = int(rng.choice(good))
            g[rr[index], cc[index]] = values[tuple(local.T)] if isinstance(values, np.ndarray) else values
            for dr, dc in itertools.product((-1, 0, 1), repeat=2):
                a, b = rr[index] + dr, cc[index] + dc
                valid = (a >= 0) & (a < shape[0]) & (b >= 0) & (b < shape[1])
                blocked[a[valid], b[valid]] = True
        if success and np.count_nonzero(g == bg) > g.size / 2:
            return g, shift
    raise AssertionError('packing exhausted')


def _build(rng, mode, rotation, previous_bg, force_rotations=None):
    palette = list(map(int, rng.permutation(10)))
    if palette[0] == previous_bg:
        palette[0], palette[-1] = palette[-1], palette[0]
    bg, wall = palette[:2]
    if mode == 0:
        plan = np.array([[0, 1], [2, 2]])
        tw, tt = [4, 2], [4, 2]
        dw, dh = [2, 4], [2, 4]
        detail = 0
    elif mode == 1:
        plan = np.array([[0, 1], [2, 3]])
        tw, tt = [int(rng.integers(2, 5)), 4], [int(rng.integers(2, 5)), 4]
        dw, dh = rng.integers(1, 4, 2), rng.integers(1, 4, 2)
        detail = int(rng.integers(2))
    else:
        shape = ((2, 3) if rng.random() < .5 else (3, 2)) if mode == 2 else (
            (2, 2), (2, 3), (3, 2))[int(rng.integers(3))]
        plan = _plan(rng, shape, int(rng.integers(4 if mode == 2 else 3, np.prod(shape) + 1)))
        tw, tt = rng.integers(2, 5, shape[1]), rng.integers(2, 5, shape[0])
        tw[0] = tw[-1] = tt[0] = tt[-1] = 4
        dw, dh = rng.integers(1, 4, shape[1]), rng.integers(1, 4, shape[0])
        detail = int(rng.integers(1, 3))
    target_labels = _geometry(rng, plan, tw, tt, detail)
    donor_labels = _geometry(rng, plan, dw, dh, detail)
    # Ports need intact inner neighbors; notches are rejected if they spoil a
    # requested corner. Random plans may also be visibly ambiguous.
    ports = _choose_ports(rng, target_labels, mode)
    n = len(set(plan.flat))
    pcount = len(ports)
    port_colors = palette[2:2 + pcount]
    fill_pool = palette[2 + pcount:9]
    assert len(fill_pool) >= 3
    richness = int(rng.integers(2, min(3, n) + 1))
    colors = fill_pool[:richness] + [int(rng.choice(fill_pool[:richness])) for _ in range(n - richness)]
    rng.shuffle(colors)
    target = _paint(target_labels, bg, wall)
    donor = _paint(donor_labels, bg, wall, colors)
    for port, color in zip(ports, port_colors):
        target[tuple(port['markers'].T)] = color
        target[tuple(port['gaps'].T)] = bg
        port['color'] = color
    # Rotating both floorplans changes their common orientation, never their
    # correspondence. Locate rotated ports by their visible bracket colors.
    target = np.rot90(target, rotation).copy()
    donor = np.rot90(donor, rotation).copy()
    target_labels = np.rot90(target_labels, rotation).copy()
    for port in ports:
        markers = np.argwhere(target == port['color'])
        unit = (markers[1] - markers[0]) // 3
        port['gaps'] = np.array([markers[0] + unit, markers[0] + 2 * unit])
        port['side'] = (port['side'] - rotation) % 4
    if mode == 0:
        active = set(range(pcount))
    elif mode == 1:
        active = {0, 2}  # one of the double-opening room's ports remains open
    else:
        active = set(map(int, rng.choice(pcount, int(rng.integers(1, pcount + 1)), replace=False)))
    keys, relative_caps = [], []
    occupancy = target != bg
    occupied = set(map(tuple, np.argwhere(occupancy)))
    for j in sorted(active):
        port = ports[j]
        canonical = _cap(rng, asymmetric=mode == 0 and j == 1)
        coords = _dock(canonical, port['side'], port['gaps'])
        assert not occupied & set(map(tuple, coords)), 'cap collision'
        occupied.update(map(tuple, coords))
        relative_caps.append(coords)
        delta = (force_rotations or {}).get(j, int(rng.integers(4)))
        input_mask = np.rot90(canonical, (-port['side'] - delta) % 4).copy()
        keys.append(dict(mask=input_mask, color=port['color']))
    # Distractors have equally legitimate necks; shape oddness cannot select.
    distractor_count = 1 if mode == 0 else int(rng.integers(0, 3))
    for _ in range(distractor_count):
        keys.append(dict(mask=np.rot90(_cap(rng), int(rng.integers(4))).copy(), color=palette[9]))
    rng.shuffle(keys)
    inp, shift = _place(rng, target, donor, keys, relative_caps, bg)
    expected = np.full_like(inp, bg)
    r, c = shift
    expected[r:r + target.shape[0], c:c + target.shape[1]] = target
    for j, coords in zip(sorted(active), relative_caps):
        expected[tuple((coords + shift).T)] = ports[j]['color']
    unclosed = {ports[j]['room'] for j in range(pcount) if j not in active}
    assert len(unclosed) < n, 'each pair must demonstrate actual color transfer'
    for label, color in enumerate(colors):
        if label not in unclosed:
            coords = np.argwhere(target_labels == label) + shift
            expected[tuple(coords.T)] = color
    rr, cc, RR, CC = _bbox(np.argwhere(expected != bg))
    expected = expected[rr:RR, cc:CC].copy()
    predicted = solve(inp)
    assert np.array_equal(predicted, expected), 'visible rule must agree with construction'
    return dict(input=inp, output=expected)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nt = _count(num_train, 'num_train', 2, 10)
    nq = _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    rotation = int(rng.integers(4))
    puzzle, seen, previous_bg = dict(train=[], test=[]), set(), None
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            mode = index if split == 'train' and index < 2 else (3 if split == 'train' else 2)
            turns = rotation if mode < 2 else int(rng.integers(4))
            forced = {0: 0, 1: 1, 2: 2} if mode == 0 else {0: 3} if mode == 1 else None
            for _ in range(80):
                try:
                    pair = _build(rng, mode, turns, previous_bg, forced)
                    if _key(pair['input']) not in seen:
                        break
                except AssertionError:
                    continue
            else:
                raise RuntimeError('bounded construction attempts exhausted')
            seen.add(_key(pair['input']))
            previous_bg = int(np.bincount(pair['input'].ravel(), minlength=10).argmax())
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure(pair):
    predicted, info = _predict(pair['input'])
    out = _grid(pair['output'])
    assert np.array_equal(predicted, out), 'incorrect sealed-room transfer'
    target, bg, wall = info['target'], info['bg'], info['wall']
    r, c, R, C = info['target_bbox']
    assembled = info['assembled']
    # Every existing wall and marker stays at its original position.
    mask = target != bg
    assert np.array_equal(assembled[r:R, c:C][mask], target[mask])
    for i, room in enumerate(info['rooms']):
        values = assembled[tuple((room + (r, c)).T)]
        expected = info['payload'][i] if i in info['closed'] else bg
        assert np.all(values == expected), 'uniform fills and open rooms'
    for match in info['matched']:
        assert len(match['coords']) == int(match['key']['mask'].sum())
        assert np.all(assembled[tuple(match['coords'].T)] == match['key']['color'])
    # Keep a geometry-only assembly as the omission-of-B counterfactual.
    geometry_only = assembled.copy()
    for room in info['rooms']:
        geometry_only[tuple((room + (r, c)).T)] = bg
    assert not np.array_equal(geometry_only, assembled), 'B must change the answer'
    # Omission of A: discard all loose keys, fill only initially enclosed rooms.
    initially_open = {p['room'] for p in info['ports']}
    without_a = np.full_like(assembled, bg)
    without_a[r:R, c:C] = target
    for i, room in enumerate(info['rooms']):
        if i not in initially_open:
            without_a[tuple((room + (r, c)).T)] = info['payload'][i]
    rr, cc, RR, CC = _bbox(np.argwhere(without_a != bg))
    assert not np.array_equal(without_a[rr:RR, cc:CC], out), 'A must change the answer'
    sizes = [len(room) for room in info['rooms']]
    donor_sizes = [len(room) for room in info['donor_rooms']]
    rooms_by_port = {}
    for p in info['ports']:
        rooms_by_port.setdefault(p['room'], []).append(p)
    partial = any(len(ps) >= 2 and any(p['color'] in info['selected'] for p in ps)
                  and any(p['color'] not in info['selected'] for p in ps)
                  for ps in rooms_by_port.values())
    shape_signatures = {(m['canonical'].shape, m['canonical'].tobytes()) for m in info['matched']}
    return dict(bg=bg, wall=wall, rooms=len(sizes), ports=len(info['ports']),
                matches=len(info['matched']), distractors=len(info['keys']) - len(info['matched']),
                closed=len(info['closed']), partial=partial,
                all_closed=not info['open_rooms'],
                asymmetric_turn=any(not np.array_equal(m['canonical'], np.fliplr(m['canonical']))
                                    and not m['rotations'] & {0} for m in info['matched']),
                sides={m['port']['side'] for m in info['matched']},
                rotations=set().union(*(m['rotations'] for m in info['matched'])),
                nonrectangular=sum(len(p) != (_bbox(p)[2] - _bbox(p)[0]) * (_bbox(p)[3] - _bbox(p)[1])
                                   for p in info['rooms'] + info['donor_rooms']),
                shapes=shape_signatures, room_sizes=sizes + donor_sizes,
                target_structure=(target.shape, (target != bg).tobytes()),
                donor_shape=info['donor'].shape, target_shape=target.shape,
                labels=tuple(p['color'] for p in info['ports']),
                payload=tuple(info['payload']),
                port_allocation=tuple(sorted(len(p) for p in rooms_by_port.values())))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, metrics = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            assert not np.array_equal(inp, out)
            metrics[split].append(_measure(pair))
    train = metrics['train']
    assert any(m['all_closed'] and m['matches'] >= 3 for m in train)
    assert any(m['partial'] for m in train), 'show that one plug need not seal a room'
    assert any(m['distractors'] for m in train), 'show local color filtering'
    assert any(m['asymmetric_turn'] for m in train), 'show rigid rotation of an asymmetric key'
    assert set().union(*(m['sides'] for m in train)) == set(range(4))
    assert set().union(*(m['rotations'] for m in train)) == set(range(4))
    assert len({m['bg'] for m in train}) >= 2
    assert len({m['rooms'] for m in train}) >= 2
    assert any(m['donor_shape'] != m['target_shape'] for m in train)


def self_test():
    """Bounded semantic invariants, 42-seed variation, counts and mutations."""
    metrics, signatures, dimensions, structures, caps = [], set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(tuple(_key(p[k]) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output')))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                m = _measure(pair)
                metrics.append(m)
                dimensions.add(pair['input'].shape)
                structures.add(m['target_structure'])
                caps.update(m['shapes'])
    assert len(signatures) == 42
    assert len(dimensions) >= 25 and len(structures) >= 100 and len(caps) >= 12
    assert {m['rooms'] for m in metrics} == {3, 4, 5, 6}
    assert {m['ports'] for m in metrics} == {2, 3, 4}
    assert {m['matches'] for m in metrics} >= {1, 2, 3, 4}
    assert {m['distractors'] for m in metrics} >= {0, 1, 2}
    assert {m['bg'] for m in metrics} == set(range(10))
    assert {m['wall'] for m in metrics} == set(range(10))
    assert sum(m['nonrectangular'] > 0 for m in metrics) >= 30
    assert min(n for m in metrics for n in m['room_sizes']) == 1
    assert max(n for m in metrics for n in m['room_sizes']) >= 20
    assert sum(m['partial'] for m in metrics) >= 42
    a, b = generate_puzzle(991, 5, 3), generate_puzzle(991, 5, 3)
    assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(812 + nt * 10 + nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    p = generate_puzzle(0, 2, 1)
    _, info = _predict(p['train'][0]['input'])
    # Removing one of a room's two keys leaves the other docked but empties
    # that whole room. This checks the interaction, not only solve equality.
    double_room = next(i for i in info['closed'] if sum(q['room'] == i for q in info['ports']) == 2)
    removed = next(m for m in info['matched'] if m['port']['room'] == double_room)
    query = p['train'][0]['input'].copy()
    query[tuple(removed['key']['cells'].T)] = info['bg']
    changed, altered = _predict(query)
    assert double_room in altered['open_rooms']
    assert any(m['port']['room'] == double_room for m in altered['matched'])
    assert not np.array_equal(changed, p['train'][0]['output'])
    # Changing one uniformly colored donor compartment changes the matching
    # target compartment without changing any wall or key.
    rr, cc, _, _ = info['donor_bbox']
    query = p['train'][0]['input'].copy()
    room = info['donor_rooms'][0] + (rr, cc)
    old = info['donor_colors'][0]
    replacement = next(x for x in range(10) if x not in {info['bg'], info['wall'], old} | {q['color'] for q in info['ports']})
    query[tuple(room.T)] = replacement
    assert not np.array_equal(solve(query), p['train'][0]['output'])
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return
    raise AssertionError('damaged output accepted')
