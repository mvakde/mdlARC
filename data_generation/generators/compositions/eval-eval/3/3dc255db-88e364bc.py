"""Rigid payload packets: count, orient, and translate a finite body to contact.

Only NumPy and the standard library are used. The reader has no generation
metadata: keys are decoded by their strokes and nozzles by cross sections.
"""
from __future__ import annotations

import numpy as np


CARDINAL = ((-1, 0), (0, 1), (1, 0), (0, -1))
COMPASS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _components(cells, diagonal=False):
    unseen = set(cells)
    result = []
    steps = COMPASS if diagonal else CARDINAL
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(part)
    return result


def _array(grid):
    a = np.asarray(grid)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or not all(1 <= s <= 30 for s in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _bg(a):
    counts = np.bincount(a.ravel(), minlength=10)
    assert int(counts.max()) > a.size // 2, "background must be a strict majority"
    return int(counts.argmax())


def _bbox(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs), min(cs), max(cs)


def _key_read(a, background):
    found = []
    for part in _components(map(tuple, np.argwhere(a != background))):
        r0, r1, c0, c1 = _bbox(part)
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 4 or w < 4 or len(part) != h * w:
            continue
        sub = a[r0:r1 + 1, c0:c1 + 1]
        wall = int(sub[0, 0])
        border = np.concatenate((sub[0], sub[-1], sub[:, 0], sub[:, -1]))
        others = set(map(int, np.unique(sub))) - {wall}
        if np.any(border != wall) or len(others) != 2:
            continue
        interpretations = []
        for tip in sorted(others):
            trail = next(iter(others - {tip}))
            tips = set(map(tuple, np.argwhere(sub == tip)))
            trails = set(map(tuple, np.argwhere(sub == trail)))
            if len(tips) < 2:
                continue
            for dr, dc in COMPASS:
                used, lengths = set(), []
                for r, c in sorted(tips):
                    run = []
                    p = r - dr, c - dc
                    while p in trails:
                        run.append(p)
                        p = p[0] - dr, p[1] - dc
                    if len(run) < 2 or used.intersection(run):
                        break
                    used.update(run)
                    lengths.append(len(run))
                if len(lengths) == len(tips) and used == trails:
                    interpretations.append((tip, trail, (dr, dc), tuple(lengths)))
        if len(interpretations) == 1:
            tip, trail, direction, lengths = interpretations[0]
            found.append(dict(cells=part, bbox=(r0, r1, c0, c1), wall=wall,
                              tip_color=tip, trail=trail, direction=direction,
                              lengths=lengths, shape=(h, w)))
    assert len(found) == 1, "one unambiguous arena direction key is required"
    return found[0]


def _local(p, d):
    # Depth increases away from the tip; x is the transverse coordinate.
    r, c = p
    dr, dc = d
    return -dr * r - dc * c, -dc * r + dr * c


def _world(p, d):
    y, x = p
    dr, dc = d
    return -dr * y - dc * x, -dc * y + dr * x


def _nozzle_geometry(part, direction):
    local = {_local(p, direction) for p in part}
    y0 = min(y for y, _ in local)
    y1 = max(y for y, _ in local)
    if y1 - y0 < 2:
        return None
    rows = [sorted(x for y, x in local if y == level) for level in range(y0, y1 + 1)]
    if len(rows[0]) != 1:
        return None
    center = rows[0][0]
    left = right = center
    split, cavity = False, set()
    for i, row in enumerate(rows[1:], 1):
        if len(row) == 1 and not split and row[0] == center:
            continue
        if len(row) != 2:
            return None
        lo, hi = row
        if lo > left or hi < right or left - lo > 1 or hi - right > 1:
            return None
        if not lo < center < hi:
            return None
        left, right, split = lo, hi, True
        cavity.update(_world((y0 + i, x), direction) for x in range(lo + 1, hi))
    if not cavity or len(rows[-1]) != 2:
        return None
    return _world((y0, center), direction), cavity


def _nozzles_read(a, background, key):
    key_mask = np.zeros(a.shape, bool)
    for p in key['cells']:
        key_mask[p] = True
    found = []
    for color in sorted(set(map(int, np.unique(a))) - {background, key['wall']}):
        for part in _components(map(tuple, np.argwhere((a == color) & ~key_mask)), True):
            if len(part) < 5:
                continue
            for d in CARDINAL:
                geometry = _nozzle_geometry(part, d)
                if geometry is None:
                    continue
                tip, cavity = geometry
                if any(not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]) for r, c in cavity):
                    continue
                values = {int(a[p]) for p in cavity} - {background}
                if len(values) != 1 or color in values or key['wall'] in values:
                    continue
                payload_color = next(iter(values))
                payload = {p for p in cavity if a[p] == payload_color}
                if len(payload) == len(cavity):
                    continue
                r0, r1, c0, c1 = _bbox(part)
                box = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
                found.append(dict(frame=part, cavity=cavity, tip=tip, axis=d,
                                  frame_color=color, payload_color=payload_color,
                                  payload=payload, quantity=len(payload), box=box))
    found = [o for o in found if not any(o is not q and o['frame'] <= q['cavity'] for q in found)]
    assert found and len({frozenset(o['frame']) for o in found}) == len(found), "ambiguous nozzles"
    boxes = set()
    for o in found:
        assert boxes.isdisjoint(o['box']), "nozzle boxes must be separate"
        boxes.update(o['box'])
    return sorted(found, key=lambda o: min(o['frame']))


def _point_ray(free, start, direction, corner=True):
    """Reader's collision calculation for ONE part of a body."""
    h, w = free.shape
    dr, dc = direction
    r, c = start
    path = [(r, c)]
    while True:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w) or not free[nr, nc]:
            return path
        if corner and dr and dc and (not free[nr, c] or not free[r, nc]):
            return path
        r, c = nr, nc
        path.append((r, c))


def _read(a):
    background = _bg(a)
    key = _key_read(a, background)
    objects = _nozzles_read(a, background, key)
    wall_cells = set(map(tuple, np.argwhere(a == key['wall']))) - key['cells']
    assert wall_cells, "key carrier must match the arena walls"
    # Enclosure association is topological; the nozzle is not itself a wall.
    universe = {(r, c) for r in range(a.shape[0]) for c in range(a.shape[1])}
    enclosed = [part for part in _components(universe - wall_cells)
                if all(r not in (0, a.shape[0] - 1) and c not in (0, a.shape[1] - 1)
                       for r, c in part)]
    explained = key['cells'] | wall_cells
    all_frames = set().union(*(o['frame'] for o in objects))
    all_payload = set().union(*(o['payload'] for o in objects))
    explained |= all_frames | all_payload
    assert explained == set(map(tuple, np.argwhere(a != background))), "unexplained marks"
    free = np.ones(a.shape, bool)
    for p in key['cells'] | wall_cells | all_frames:
        free[p] = False
    swept, guards = set(), set()
    dr, dc = key['direction']
    for o in objects:
        ar, ac = o['axis']
        tr, tc = o['tip']
        initial = [(tr + i * ar, tc + i * ac) for i in range(1, o['quantity'] + 1)]
        assert all(0 <= r < a.shape[0] and 0 <= c < a.shape[1] and a[r, c] == background
                   for r, c in initial), "initial packet needs empty room beyond the tip"
        regions = [p for p in enclosed if o['frame'] | o['cavity'] | set(initial) <= p]
        assert len(regions) == 1, "nozzle must belong to a closed arena chamber"
        region = regions[0]
        paths = [_point_ray(free, p, (dr, dc)) for p in initial]
        distance = min(len(path) - 1 for path in paths)
        assert all(p in region for path in paths for p in path), "ray leaves its chamber"
        final = {(r + distance * dr, c + distance * dc) for r, c in initial}
        body_sweep = {path[t] for path in paths for t in range(distance + 1)}
        body_guard = body_sweep.copy()
        if dr and dc:
            body_guard.update((r + dr, c) for t in range(distance) for path in paths for r, c in [path[t]])
            body_guard.update((r, c + dc) for t in range(distance) for path in paths for r, c in [path[t]])
        assert guards.isdisjoint(body_guard), "packets must have disjoint swept corridors"
        guards.update(body_guard)
        swept.update(body_sweep)
        plain_distance = min(len(_point_ray(free, p, (dr, dc), False)) - 1 for p in initial)
        independent = {path[-1] for path in paths}
        o.update(initial=initial, final=final, distance=distance, sweep=body_sweep,
                 guard=body_guard, independent=independent,
                 rigid_witness=independent != final,
                 extent_witness=len(paths[0]) - 1 > distance,
                 corner_witness=plain_distance > distance)
    return dict(background=background, key=key, objects=objects, walls=wall_cells,
                free=free, swept=swept)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode only this input; never consult seeds, demonstrations, or samplers."""
    a = _array(input_grid)
    info = _read(a)
    out = a.copy()
    for o in info['objects']:
        for p in o['payload']:
            out[p] = info['background']
    for o in info['objects']:
        for p in o['final']:
            out[p] = o['payload_color']
    return out


def _profile(rng, anchor=False, small=False):
    a, b = (2, 2) if anchor else tuple(map(int, rng.integers(1, 2 if small else 4, size=2)))
    stem = 0 if small else int(rng.integers(0, 2 if anchor else 3))
    tail = int(rng.integers(3, 5)) if anchor else int(rng.integers(2, 4 if small else 7))
    rows, left, right = [(0, 0)] * (stem + 1), 0, 0
    while left < a or right < b:
        dl = left < a and (left == 0 or right == b or rng.random() < .8)
        dr = right < b and (right == 0 or left == a or rng.random() < .8)
        if not dl and not dr:
            dl = left < a
            dr = not dl
        left, right = left + int(dl), right + int(dr)
        rows.append((-left, right))
    rows.extend([(-a, b)] * tail)
    frame, cavity = set(), set()
    for y, (lo, hi) in enumerate(rows):
        frame.add((y, lo))
        frame.add((y, hi))
        cavity.update((y, x) for x in range(lo + 1, hi))
    return dict(frame=frame, cavity=cavity, tip=(0, 0))


def _orient_profile(shape, d):
    points = {p: _world(p, d) for p in shape['frame'] | shape['cavity']}
    mr = min(p[0] for p in points.values())
    mc = min(p[1] for p in points.values())
    def trans(p):
        r, c = _world(p, d)
        return r - mr, c - mc
    frame = {trans(p) for p in shape['frame']}
    return dict(frame=frame, cavity={trans(p) for p in shape['cavity']},
                tip=trans(shape['tip']), axis=d,
                height=max(r for r, _ in frame) + 1, width=max(c for _, c in frame) + 1)


def _payload(rng, cavity, q, pattern):
    cells = sorted(cavity)
    selected = {cells[int(rng.integers(len(cells)))]}
    while len(selected) < q:
        remaining = sorted(cavity - selected)
        if pattern == 'compact':
            choices = [p for p in remaining if any(abs(p[0] - r) + abs(p[1] - c) == 1 for r, c in selected)]
        elif pattern == 'sparse':
            distances = [min(max(abs(p[0] - r), abs(p[1] - c)) for r, c in selected) for p in remaining]
            choices = [p for p, d in zip(remaining, distances) if d == max(distances)]
        else:
            choices = remaining
        assert choices, "cavity must be connected"
        selected.add(choices[int(rng.integers(len(choices)))])
    return selected


def _make_key(rng, direction, wall, trail, tip):
    dr, dc = direction
    count, length = int(rng.choice([2, 2, 3])), int(rng.choice([2, 2, 3]))
    marks = []
    for i in range(count):
        offset = int(rng.integers(0, 2))
        base = (0, 2 * i) if dr else (2 * i, 0)
        for j in range(length + 1):
            marks.append((base[0] + (j + offset) * dr, base[1] + (j + offset) * dc,
                          tip if j == length else trail))
    r0, r1, c0, c1 = _bbox([(r, c) for r, c, _ in marks])
    a = np.full((r1 - r0 + 3, c1 - c0 + 3), wall, np.int8)
    for r, c, color in marks:
        a[r - r0 + 1, c - c0 + 1] = color
    return a


def _arena(rng, h, w, special=False, complexity=1, fallback=False):
    footprint = np.ones((h, w), bool)
    kind = 0 if fallback else int(rng.integers(4))
    if kind == 1:
        # Independently cut corners: bays change the first contact of a long rod.
        for corner in rng.choice(4, int(rng.integers(1, 4)), replace=False):
            dh, dw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
            rs = slice(0, dh) if corner < 2 else slice(h - dh, h)
            cs = slice(0, dw) if corner % 2 == 0 else slice(w - dw, w)
            footprint[rs, cs] = False
    elif kind == 2:
        yy, xx = np.indices((h, w))
        cut = int(rng.integers(2, 5))
        footprint &= (yy + xx >= cut) & (yy + w - 1 - xx >= cut)
        footprint &= (h - 1 - yy + xx >= cut) & (h + w - 2 - yy - xx >= cut)
    elif kind == 3:
        left = right = 0
        for r in range(h):
            left = int(np.clip(left + rng.integers(-1, 2), 0, 3))
            right = int(np.clip(right + rng.integers(-1, 2), 0, 3))
            footprint[r, :left] = False
            if right:
                footprint[r, w - right:] = False
    padded = np.pad(footprint, 1)
    inside = footprint.copy()
    for dr, dc in CARDINAL:
        inside &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    wall = footprint & ~inside
    if special:
        # A ledge end admits the next diagonal cell but blocks its side cell.
        col = 19 if fallback else int(rng.integers(16, w - 3))
        end = 10 if fallback else int(rng.integers(8, h - 5))
        wall[:end + 1, col] |= footprint[:end + 1, col]
    for _ in range(0 if fallback else complexity):
        if rng.random() < .5:
            r = int(rng.integers(3, h - 3))
            length = int(rng.integers(2, w // 2 + 1))
            cs = slice(0, length) if rng.random() < .5 else slice(w - length, w)
            wall[r, cs] |= footprint[r, cs]
        else:
            c = int(rng.integers(3, w - 3))
            length = int(rng.integers(2, h // 2 + 1))
            rs = slice(0, length) if rng.random() < .5 else slice(h - length, h)
            wall[rs, c] |= footprint[rs, c]
    return wall, footprint


def _body_motion(free, initial, direction):
    """Constructor translates a whole footprint, independently of point rays."""
    dr, dc = direction
    current = set(initial)
    sweep, guard = set(current), set(current)
    distance = 0
    h, w = free.shape
    while True:
        next_body = {(r + dr, c + dc) for r, c in current}
        probes = next_body.copy()
        if dr and dc:
            probes |= {(r + dr, c) for r, c in current} | {(r, c + dc) for r, c in current}
        if any(not (0 <= r < h and 0 <= c < w) or not free[r, c] for r, c in probes):
            # Keep empty blocker-neighbor cells free of later nozzles, too.
            guard.update(p for p in probes if 0 <= p[0] < h and 0 <= p[1] < w and free[p])
            return current, sweep, guard, distance
        current = next_body
        distance += 1
        sweep.update(current)
        guard.update(probes)


def _place_nozzles(rng, wall, footprint, direction, specs):
    free_base = footprint & ~wall
    frames, boxes, guards, records = set(), set(), set(), []
    h, w = wall.shape
    for index, spec in enumerate(specs):
        shape = _orient_profile(spec['shape'], spec['axis'])
        positions = [(r, c) for r in range(1, h - shape['height'])
                     for c in range(1, w - shape['width'])]
        accepted = None
        for raw in rng.permutation(len(positions)):
            r0, c0 = positions[int(raw)]
            frame = {(r + r0, c + c0) for r, c in shape['frame']}
            cavity = {(r + r0, c + c0) for r, c in shape['cavity']}
            box = {(r + r0, c + c0) for r in range(shape['height']) for c in range(shape['width'])}
            halo = {(r + dr, c + dc) for r, c in box for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            if (not box.isdisjoint(boxes | guards) or not halo.isdisjoint(frames)
                    or any(not free_base[p] for p in halo)):
                continue
            tip = shape['tip'][0] + r0, shape['tip'][1] + c0
            ar, ac = shape['axis']
            initial = [(tip[0] + i * ar, tip[1] + i * ac) for i in range(1, spec['q'] + 1)]
            free = free_base.copy()
            for p in frame | frames:
                free[p] = False
            if any(not (0 <= r < h and 0 <= c < w) or not free[r, c] for r, c in initial):
                continue
            final, sweep, guard, distance = _body_motion(free, initial, direction)
            if distance < 1 or not guard.isdisjoint(boxes | guards):
                continue
            # The arena, rather than the nozzle itself, must cause final contact.
            bare_final, _, _, bare_distance = _body_motion(free_base, initial, direction)
            if distance != bare_distance:
                continue
            independent = {_point_ray(free, p, direction)[-1] for p in initial}
            plain = min(len(_point_ray(free, p, direction, False)) - 1 for p in initial)
            if spec.get('rigid') and independent == final:
                continue
            if spec.get('extent') and len(_point_ray(free, initial[0], direction)) - 1 == distance:
                continue
            if spec.get('corner') and plain == distance:
                continue
            payload = _payload(rng, cavity, spec['q'], spec['pattern'])
            if spec.get('disconnected') and len(_components(payload, True)) == 1:
                continue
            accepted = dict(frame=frame, payload=payload, final=final, box=box,
                            tip=tip, spec=spec, sweep=sweep, guard=guard)
            break
        if accepted is None:
            if spec.get('required', True):
                return None
            continue
        frames.update(accepted['frame'])
        boxes.update(accepted['box'])
        guards.update(accepted['guard'])
        records.append(accepted)
    return records


def _assemble(rng, arena_input, arena_output, key_grid, rotation, reflect):
    h, w = arena_input.shape
    kh, kw = key_grid.shape
    layouts = []
    if h + kh + 1 <= 30:
        layouts += ['top', 'bottom']
    if w + kw + 1 <= 30:
        layouts += ['left', 'right']
    assert layouts
    side = str(rng.choice(layouts))
    if side in ('top', 'bottom'):
        min_h, min_w = h + kh + 1, max(w, kw)
    else:
        min_h, min_w = max(h, kh), w + kw + 1
    height, width = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
    background = _bg(arena_input)
    a = np.full((height, width), background, np.int8)
    b = a.copy()
    if side == 'top':
        kr, er = 0, kh + 1
        kc, ec = int(rng.integers(width - kw + 1)), int(rng.integers(width - w + 1))
    elif side == 'bottom':
        er, kr = 0, h + 1
        kc, ec = int(rng.integers(width - kw + 1)), int(rng.integers(width - w + 1))
    elif side == 'left':
        kc, ec = 0, kw + 1
        kr, er = int(rng.integers(height - kh + 1)), int(rng.integers(height - h + 1))
    else:
        ec, kc = 0, w + 1
        kr, er = int(rng.integers(height - kh + 1)), int(rng.integers(height - h + 1))
    a[er:er + h, ec:ec + w] = arena_input
    b[er:er + h, ec:ec + w] = arena_output
    a[kr:kr + kh, kc:kc + kw] = key_grid
    b[kr:kr + kh, kc:kc + kw] = key_grid
    if reflect:
        a, b = np.fliplr(a), np.fliplr(b)
    return dict(input=np.rot90(a, rotation).copy(), output=np.rot90(b, rotation).copy())


def _pair(rng, specs, direction, roles, rotation, reflect, test=False, special=False):
    background, wall_color, frame_color, payload_color, trail, terminal = roles
    for attempt in range(24):
        fallback = attempt == 23
        required_shapes = [_orient_profile(s['shape'], s['axis']) for s in specs if s.get('required', True)]
        min_h = max([18 if len(specs) > 1 or special else 14] +
                    [p['height'] + (s['q'] if s['axis'][0] else 0) + 4
                     for s, p in zip([s for s in specs if s.get('required', True)], required_shapes)])
        min_w = max([22 if special else 18] +
                    [p['width'] + (s['q'] if s['axis'][1] else 0) + 4
                     for s, p in zip([s for s in specs if s.get('required', True)], required_shapes)])
        h = max(20, min_h) if fallback else int(rng.integers(min_h, 22))
        w = max(23, min_w) if fallback else int(rng.integers(min_w, 25))
        wall, footprint = _arena(rng, h, w, special, int(rng.integers(0, 3 if test else 2)), fallback)
        records = _place_nozzles(rng, wall, footprint, direction, specs)
        if records is None:
            continue
        a = np.full((h, w), background, np.int8)
        a[wall] = wall_color
        b = a.copy()
        for record in records:
            spec = record['spec']
            for p in record['frame']:
                a[p] = b[p] = frame_color
            for p in record['payload']:
                a[p] = spec.get('color', payload_color)
            for p in record['final']:
                b[p] = spec.get('color', payload_color)
        key = _make_key(rng, direction, wall_color, trail, terminal)
        if h + key.shape[0] + 1 > 30 and w + key.shape[1] + 1 > 30:
            continue
        pair = _assemble(rng, a, b, key, rotation, reflect)
        try:
            info = _read(pair['input'])
        except AssertionError:
            continue
        if len(info['objects']) != len(records):
            continue
        assert np.array_equal(solve(pair['input']), pair['output']), "body construction and point reader differ"
        return pair
    raise AssertionError("bounded arena construction failed")


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    # These six literal roles are visibly decoded, never needed as solver state.
    palette = tuple(map(int, rng.choice(10, 6, replace=False)))
    anchor = _profile(rng, True)
    low, high = int(rng.integers(2, 4)), int(rng.integers(5, 7))
    rotation, reflect = int(rng.integers(4)), bool(rng.integers(2))
    base = dict(shape=anchor, axis=(0, 1), required=True)
    plans = [([dict(base, q=low, pattern='compact')], (1, 0), False),
             ([dict(base, q=high, pattern='random', rigid=True, extent=True, corner=True),
               dict(base, q=low, pattern='sparse', disconnected=True)], (1, 1), True)]
    plans = [plans[int(i)] for i in rng.permutation(2)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            if split == 'train' and index < 2:
                specs, direction, special = plans[index]
                roles = list(palette)
                roles[0] = int(rng.choice([v for v in range(10) if v not in palette[1:]]))
                rot, flip = rotation, reflect
            else:
                roles = list(map(int, rng.choice(10, 6, replace=False)))
                count_objects = int(rng.choice([1, 1, 2, 2, 3]))
                direction = COMPASS[int(rng.integers(8))]
                compatible_axes = [d for d in CARDINAL if d[0] * direction[0] + d[1] * direction[1] >= 0]
                specs = []
                for j in range(count_objects):
                    shape = _profile(rng, small=count_objects == 3 and j > 0)
                    capacity = len(shape['cavity'])
                    axial_depth = max(y for y, _ in shape['frame']) + 1
                    if j == 0 and split == 'test':
                        # Longer rods combine the learned count and first-contact
                        # rules. Training rods have at most seven cells.
                        for _ in range(100):
                            if capacity >= 9 and axial_depth <= 9:
                                break
                            shape = _profile(rng)
                            capacity = len(shape['cavity'])
                            axial_depth = max(y for y, _ in shape['frame']) + 1
                        else:
                            # A wide shallow profile is guaranteed to fit eight.
                            shape = dict(frame={(0, 0), (1, -1), (1, 1)} |
                                         {(y, x) for y in range(2, 6) for x in (-2, 2)},
                                         cavity={(1, 0)} | {(y, x) for y in range(2, 6) for x in (-1, 0, 1)},
                                         tip=(0, 0))
                            capacity, axial_depth = 13, 6
                        q = int(rng.integers(8, min(9, capacity - 1, 17 - axial_depth) + 1))
                    else:
                        q = int(rng.integers(1, min(7, capacity - 1, 17 - axial_depth) + 1))
                    if q == capacity:
                        q -= 1
                    specs.append(dict(shape=shape, axis=compatible_axes[int(rng.integers(len(compatible_axes)))], q=q,
                                      pattern=str(rng.choice(['compact', 'sparse', 'random'])),
                                      required=j == 0 or (count_objects == 3 and split == 'train'),
                                      rigid=j == 0 and split == 'test', extent=j == 0 and split == 'test'))
                    if j and rng.random() < .5:
                        specs[-1]['color'] = int(rng.choice([v for v in range(10) if v not in roles[:3]]))
                special = False
                rot, flip = int(rng.integers(4)), bool(rng.integers(2))
            for _ in range(5):
                pair = _pair(rng, specs, direction, roles, rot, flip, split == 'test', special)
                key = pair['input'].shape, pair['input'].tobytes()
                if key not in seen:
                    seen.add(key)
                    break
            else:
                raise AssertionError('could not make distinct inputs')
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _frame_signature(o):
    local = [_local(p, o['axis']) for p in o['frame']]
    r0, _, c0, _ = _bbox(local)
    return tuple(sorted((r - r0, c - c0) for r, c in local)), o['axis'], o['frame_color'], o['payload_color']


def _training_evidence(infos):
    quantities, matched, arrangements, directions = set(), {}, {}, set()
    corner = rigid = extent = compact = False
    for info in infos:
        directions.add(info['key']['direction'])
        for o in info['objects']:
            q = o['quantity']
            quantities.add(q)
            matched.setdefault(_frame_signature(o), set()).add(q)
            components = len(_components(o['payload'], True))
            arrangements.setdefault(q, set()).add(components)
            compact |= components < q
            corner |= o['corner_witness']
            rigid |= o['rigid_witness']
            extent |= o['extent_witness']
    assert len(quantities) >= 2 and any(len(v) >= 2 for v in matched.values()), "missing matched counting evidence"
    assert compact and any(len(v) >= 2 for v in arrangements.values()), "missing cell-count/arrangement evidence"
    assert len(directions) >= 2 and corner and rigid and extent, "missing direction, rigid-body, extent, or corner evidence"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    n, m = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _array(pair['input']), _array(pair['output'])
            assert a.shape == b.shape and not np.array_equal(a, b)
            fingerprint = a.shape, a.tobytes()
            assert fingerprint not in seen, "repeated input"
            seen.add(fingerprint)
            info = _read(a)
            assert 1 <= len(info['objects']) <= 3
            assert all(o['distance'] >= 1 for o in info['objects']), "packet must visibly travel"
            assert np.array_equal(solve(a), b), "wrong transformation"
            fixed = info['key']['cells'] | info['walls'] | set().union(*(o['frame'] for o in info['objects']))
            assert all(a[p] == b[p] for p in fixed), "structural cells changed"
            for o in info['objects']:
                final = o['final']
                assert len(final) == o['quantity'] and all(b[p] == o['payload_color'] for p in final)
                assert all(b[p] == info['background'] for p in o['payload']), "payload not cleared"
                dr, dc = info['key']['direction']
                blocked = False
                for r, c in final:
                    probes = [(r + dr, c + dc)]
                    if dr and dc:
                        probes += [(r + dr, c), (r, c + dc)]
                    blocked |= any(p in info['walls'] for p in probes)
                assert blocked, "final contact must be with an arena wall"
            for color in {o['payload_color'] for o in info['objects']}:
                assert np.count_nonzero(a == color) == np.count_nonzero(b == color), "payload quantity not conserved"
            if split == 'train':
                training.append(info)
            else:
                assert any(o['rigid_witness'] for o in info['objects']), "test must require rigid contact"
                assert any(o['extent_witness'] for o in info['objects']), "packet extent must change travel distance"
                assert max(o['quantity'] for o in info['objects']) >= 8, "missing longer-packet query shift"
    _training_evidence(training)


def _fingerprint(puzzle):
    return tuple((p[k].shape, p[k].tobytes()) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output'))


def self_test():
    """Bounded 42-seed variation audit, symmetry checks, and corruption checks."""
    # A concrete count/contact regression: the third cell catches a ledge
    # which the first two cells miss. This is not a generated output fixture.
    distances_by_count = []
    for quantity in (2, 3):
        arena = np.zeros((16, 18), np.int8)
        arena[0] = arena[-1] = 1
        arena[:, 0] = arena[:, -1] = 1
        arena[7, 10:] = 1
        shape = dict(frame={(0, 0), (1, -1), (1, 1)} |
                     {(y, x) for y in range(2, 5) for x in (-2, 2)},
                     cavity={(1, 0)} | {(y, x) for y in range(2, 5) for x in (-1, 0, 1)},
                     tip=(0, 0))
        for y, x in shape['frame']:
            arena[5 - x, 7 - y] = 2
        for p in [(4, 4), (4, 5), (6, 4)][:quantity]:
            arena[p] = 3
        key = _make_key(np.random.default_rng(8), (1, 0), 1, 4, 5)
        canvas = np.zeros((16 + key.shape[0] + 1, 18), np.int8)
        canvas[:16] = arena
        canvas[17:17 + key.shape[0], :key.shape[1]] = key
        info = _read(canvas)
        distances_by_count.append(info['objects'][0]['distance'])
        out = solve(canvas)
        assert np.count_nonzero(out == 3) == quantity
        expected_row = 14 if quantity == 2 else 6
        assert all(out[expected_row, c] == 3 for c in range(8, 8 + quantity))
    assert distances_by_count == [9, 1], "count must change which wall causes contact"
    fingerprints, sizes, profiles, colors, directions, axes, counts, quantities = (set() for _ in range(8))
    outlines, distances, relative_angles, key_shapes = set(), set(), set(), set()
    corner_cases = rigid_cases = mixed_axes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                info = _read(a)
                sizes.add(a.shape)
                colors.add(info['background'])
                directions.add(info['key']['direction'])
                key_shapes.add(info['key']['shape'])
                counts.add(len(info['objects']))
                r0, r1, c0, c1 = _bbox(info['walls'])
                outlines.add(tuple(sorted((r - r0, c - c0) for r, c in info['walls'])))
                mixed_axes += len({o['axis'] for o in info['objects']}) > 1
                for o in info['objects']:
                    profiles.add(_frame_signature(o)[0])
                    quantities.add(o['quantity'])
                    axes.add(o['axis'])
                    distances.add(o['distance'])
                    d, e = info['key']['direction'], o['axis']
                    relative_angles.add(d[0] * e[0] + d[1] * e[1])
                    corner_cases += o['corner_witness']
                    rigid_cases += o['rigid_witness']
        # Symmetry is tested on held-out input copies, with no sampler state.
        pair = puzzle['test'][0]
        for rotation in range(4):
            a, b = np.rot90(pair['input'], rotation), np.rot90(pair['output'], rotation)
            assert np.array_equal(solve(a), b)
            assert np.array_equal(solve(np.fliplr(a)), np.fliplr(b))
        permutation = np.random.default_rng(seed + 1000).permutation(10)
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    assert len(fingerprints) == 42 and len(sizes) >= 35 and len(outlines) >= 90
    assert directions == set(COMPASS) and axes == set(CARDINAL) and colors == set(range(10))
    assert counts == {1, 2, 3} and max(quantities) >= 8 and len(profiles) >= 35
    assert len(key_shapes) >= 10 and max(distances) >= 8 and mixed_axes >= 3
    assert relative_angles >= {0, 1} and corner_cases >= 42 and rigid_cases >= 84
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(42 + 10 * n + m, n, m)
            assert len(p['train']) == n and len(p['test']) == m
    assert _fingerprint(generate_puzzle(73)) == _fingerprint(generate_puzzle(73))
    for n, m in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    good = generate_puzzle(0)
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in good[s]] for s in good}
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
