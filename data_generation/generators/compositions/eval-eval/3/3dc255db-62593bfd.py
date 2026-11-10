"""Common charge cancellation: rotation classes of pointed hollow reservoirs.

Labels are painted from sampled construction records.  solve reconstructs shells
and charges from input cells; it does not consult construction or stored examples.
"""
from collections import Counter, defaultdict
import json

import numpy as np


_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        seen, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D8:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    seen.add(p)
                    stack.append(p)
        result.append(seen)
    return result


def _turn(point, k):
    r, c = point
    for _ in range(k % 4):
        r, c = -c, r
    return r, c


def _mask(cells):
    rs, cs = zip(*cells)
    r0, c0 = min(rs), min(cs)
    a = np.zeros((max(rs) - r0 + 1, max(cs) - c0 + 1), dtype=bool)
    for r, c in cells:
        a[r - r0, c - c0] = True
    return a


def _mask_key(mask):
    return mask.shape, bytes(mask.astype(np.uint8).ravel())


def _rotation_key(mask):
    return min(_mask_key(np.rot90(mask, k)) for k in range(4))


def _ray(shape, tip, direction, length=None):
    r, c = tip
    dr, dc = direction
    result = []
    while length is None or len(result) < length:
        r, c = r + dr, c + dc
        if not (0 <= r < shape[0] and 0 <= c < shape[1]):
            break
        result.append((r, c))
    return result


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or any(not 1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer color grid")
    return a


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    i = int(np.argmax(counts))
    if counts[i] * 2 <= grid.size:
        raise ValueError("background must occupy more than half the grid")
    return int(values[i])


def _contour(cells, k):
    """Cross-sectional recognition, not matching a generated stencil."""
    local = {_turn(p, k) for p in cells}
    ys = [r for r, _ in local]
    y0, y1 = min(ys), max(ys)
    if y1 - y0 < 2:
        return None
    rows = [sorted(c for r, c in local if r == y) for y in range(y0, y1 + 1)]
    if len(rows[0]) != 1 or any(not row for row in rows):
        return None
    tip_x = rows[0][0]
    previous_l = previous_r = tip_x
    split, cavity, run_counts = False, set(), []
    for offset, row in enumerate(rows):
        lo, hi = row[0], row[-1]
        if offset and not (0 <= previous_l - lo <= 1 and 0 <= hi - previous_r <= 1):
            return None
        previous_l, previous_r = lo, hi
        runs = []
        for c in row:
            if not runs or c != runs[-1][-1] + 1:
                runs.append([c])
            else:
                runs[-1].append(c)
        run_counts.append(len(runs))
        if len(runs) == 1:
            if split or len(row) > 4:
                return None
        elif len(runs) == 2 and all(len(run) <= 2 for run in runs):
            split = True
            cavity.update(_turn((y0 + offset, c), -k)
                          for c in range(runs[0][-1] + 1, runs[1][0]))
        else:
            return None
    if not cavity or run_counts[-2:] != [2, 2] or not rows[-1][0] < tip_x < rows[-1][-1]:
        return None
    tip = _turn((y0, tip_x), -k)
    return tip, _turn((-1, 0), -k), cavity


def _read_scene(grid):
    bg = _background(grid)
    candidates = []
    for color in sorted(int(c) for c in np.unique(grid) if c != bg):
        for cells in _components(map(tuple, np.argwhere(grid == color))):
            if len(cells) < 5:
                continue
            for k in range(4):
                geometry = _contour(cells, k)
                if geometry is None:
                    continue
                tip, direction, cavity = geometry
                values = {int(grid[p]) for p in cavity} - {bg}
                if len(values) != 1 or color in values:
                    continue
                pc = next(iter(values))
                payload = {p for p in cavity if grid[p] == pc}
                if len(payload) == len(cavity):
                    continue
                mask = _mask(cells)
                rs, cs = zip(*cells)
                box = {(r, c) for r in range(min(rs), max(rs) + 1)
                       for c in range(min(cs), max(cs) + 1)}
                candidates.append(dict(frame=cells, cavity=cavity, payload=payload,
                    tip=tip, direction=direction, frame_color=color, payload_color=pc,
                    q=len(payload), mask=mask, key=_rotation_key(mask), box=box))
    # Payloads may accidentally form a tiny cup. Their enclosing shell owns them.
    objects = [a for a in candidates if not any(a is not b and a['frame'] <= b['cavity']
                                               for b in candidates)]
    if not objects or len({frozenset(o['frame']) for o in objects}) != len(objects):
        raise ValueError("missing or ambiguous pointed shells")
    objects.sort(key=lambda o: min(o['frame']))
    boxes, foreground = set(), set()
    for o in objects:
        if o['box'] & boxes:
            raise ValueError("shell boxes overlap")
        boxes.update(o['box'])
        foreground.update(o['frame'] | o['payload'])
        if any(grid[p] != bg for p in o['box'] - o['frame'] - o['payload']):
            raise ValueError("unexpected cells in shell region")
    if foreground != set(map(tuple, np.argwhere(grid != bg))):
        raise ValueError("unowned foreground cells")
    minima = {}
    for o in objects:
        minima[o['key']] = min(minima.get(o['key'], o['q']), o['q'])
    rays = set()
    for o in objects:
        o['baseline'] = minima[o['key']]
        o['surplus'] = o['q'] - o['baseline']
        o['room'] = len(_ray(grid.shape, o['tip'], o['direction']))
        o['ray'] = _ray(grid.shape, o['tip'], o['direction'], o['surplus'])
        if set(o['ray']) & (boxes | rays) or any(grid[p] != bg for p in o['ray']):
            raise ValueError("emitted ray is obstructed or overlaps another ray")
        rays.update(o['ray'])
    return bg, objects


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    bg, objects = _read_scene(grid)
    output = grid.copy()
    for o in objects:
        for p in o['payload']:
            output[p] = bg
        for p in o['ray']:
            output[p] = o['payload_color']
    return output


def _frame_from_rows(rows, thickness):
    frame, cavity = set(), set()
    for r, (lo, hi) in enumerate(rows):
        left, right = lo + thickness - 1, hi - thickness + 1
        if left >= right:
            frame.update((r, c) for c in range(lo, hi + 1))
        else:
            frame.update((r, c) for c in range(lo, left + 1))
            frame.update((r, c) for c in range(right, hi + 1))
            cavity.update((r, c) for c in range(left + 1, right))
    shift = -min(c for _, c in frame)
    frame = {(r, c + shift) for r, c in frame}
    cavity = {(r, c + shift) for r, c in cavity}
    mask = _mask(frame)
    return dict(frame=frame, cavity=cavity, tip=(0, shift), mask=mask,
                key=_rotation_key(mask), height=mask.shape[0], width=mask.shape[1])


def _sample_frame(rng, capacity=3, forbidden=(), chiral=False, anchor=False, small=False):
    forbidden = set(forbidden)

    def allowed(base):
        return (len(base['cavity']) >= capacity and base['key'] not in forbidden
                and (not chiral or _rotation_key(np.fliplr(base['mask'])) != base['key'])
                and (not anchor or 5 <= base['width'] <= 7 and base['height'] <= 8)
                and base['height'] <= (6 if small else 10))

    for _ in range(180):
        reach = 2 if small else 3 if anchor else 4
        a, b = (int(rng.integers(1, reach + 1)) for _ in range(2))
        if rng.random() < .25 and not chiral:
            b = a
        stem = int(rng.integers(0, 2 if anchor or small else 3))
        tail = int(rng.integers(2, 6 if anchor else 7))
        thickness = 1 if anchor else int(rng.choice([1, 1, 1, 2]))
        rows = [(0, 0)] * (stem + 1)
        l = r = 0
        while l < a or r < b:
            dl = l < a and (r == b or rng.random() < .8)
            dr = r < b and (l == a or rng.random() < .8)
            if not dl and not dr:
                dl = l < a
                dr = not dl
            l, r = l + int(dl), r + int(dr)
            rows.append((-l, r))
        rows.extend([(-a, b)] * tail)
        base = _frame_from_rows(rows, thickness)
        if allowed(base):
            return base
    # A finite geometric fallback, not a stored puzzle or output.
    for a in range(1, 5):
        for b in range(1, 5):
            for tail in range(2, 7):
                rows = [(0, 0)] + [(-min(t, a), min(t, b)) for t in range(1, max(a, b) + 1)]
                rows.extend([(-a, b)] * tail)
                base = _frame_from_rows(rows, 1)
                if allowed(base):
                    return base
    raise RuntimeError("no compatible shell profile")


def _payload(rng, cavity, q, pattern):
    cells = sorted(cavity)
    if pattern == 'separated':
        # A pedagogical contrast must be guaranteed, not a likely side effect
        # of scattering. Isolate one cell from the remaining q-1 cells.
        for i in rng.permutation(len(cells)):
            point = cells[int(i)]
            distant = [p for p in cells if max(abs(p[0]-point[0]), abs(p[1]-point[1])) > 1]
            if q > 1 and len(distant) >= q - 1:
                return {point} | {distant[int(j)] for j in rng.choice(len(distant), q - 1, replace=False)}
        raise RuntimeError("cavity cannot display separated payload components")
    selected = {cells[int(rng.integers(len(cells)))]}
    while len(selected) < q:
        choices = sorted(cavity - selected)
        if pattern == 'compact':
            nearby = [p for p in choices if any(max(abs(p[0]-s[0]), abs(p[1]-s[1])) == 1
                                              for s in selected)]
            if nearby:
                choices = nearby
        elif pattern == 'scatter':
            distances = [min(max(abs(p[0]-s[0]), abs(p[1]-s[1])) for s in selected) for p in choices]
            best = max(distances)
            choices = [p for p, d in zip(choices, distances) if d == best]
        selected.add(choices[int(rng.integers(len(choices)))])
    return selected


def _object(rng, base, turn, q, colors, class_id, pattern='random', mode=None):
    rotated = {_turn(p, turn) for p in base['frame']}
    r0 = min(r for r, c in rotated)
    c0 = min(c for r, c in rotated)

    def transform(p):
        r, c = _turn(p, turn)
        return r - r0, c - c0

    payload = _payload(rng, base['cavity'], q, pattern)
    return dict(frame={transform(p) for p in base['frame']},
                payload={transform(p) for p in payload}, tip=transform(base['tip']),
                direction=_turn((-1, 0), turn), height=_mask(rotated).shape[0],
                width=_mask(rotated).shape[1], frame_color=colors[0], payload_color=colors[1],
                q=q, class_id=class_id, mode=mode, silent_guard=0)


def _colors(rng, bg):
    return tuple(int(c) for c in rng.choice([c for c in range(10) if c != bg], 2, replace=False))


def _mirror(base):
    w = base['width']
    frame = {(r, w - 1 - c) for r, c in base['frame']}
    return dict(frame=frame, cavity={(r, w - 1 - c) for r, c in base['cavity']},
                tip=(base['tip'][0], w - 1 - base['tip'][1]), mask=_mask(frame),
                key=_rotation_key(_mask(frame)), height=base['height'], width=w)


def _finish_objects(rng, objects):
    minima = {}
    for o in objects:
        minima[o['class_id']] = min(minima.get(o['class_id'], o['q']), o['q'])
    for o in objects:
        o['surplus'] = o['q'] - minima[o['class_id']]
        if not o['surplus']:
            o['mode'] = 'any'
        elif o['mode'] is None:
            o['mode'] = str(rng.choice(['free', 'any', 'exact', 'clip', 'zero'],
                                       p=[.40, .25, .15, .17, .03]))
        if o['mode'] == 'clip' and o['surplus'] == 1:
            o['mode'] = 'zero'
    return objects


def _witness_objects(rng, base, b, which, bg):
    palette = [int(c) for c in rng.permutation(10) if c != bg]
    f, p, s = palette[:3]
    start = int(rng.integers(4))
    objects = []
    used = {base['key'], _mirror(base)['key']}
    if which == 0:
        charges = [b, b + 1, b + 1, b + 3]
        turns = [start, (start + 1) % 4, start, (start + 2) % 4]
        colors = [(f, p), (p, f), (f, p), (s, p)]
        patterns = ['compact', 'compact', 'separated', 'random']
        for q, k, color, pattern in zip(charges, turns, colors, patterns):
            objects.append(_object(rng, base, k, q, color, 0, pattern, 'free'))
        # The repeated middle quantity has different component counts.
        assert len(_components(objects[1]['payload'])) == 1
        assert len(_components(objects[2]['payload'])) > 1
        objects.append(_object(rng, _mirror(base), int(rng.integers(4)), b + 1,
                               colors[1], 1, 'random'))
        # Silence must be observable: an edge tip would hide the incorrect
        # one-cell signal produced by area/reflection matching.
        objects[-1]['silent_guard'] = 2
    else:
        for i, q in enumerate([b + 1, b + 3, b + 6]):
            objects.append(_object(rng, base, (start + i) % 4, q,
                                   [(f, p), (s, f), (p, s)][i], 0, 'random', 'free'))
        small = _sample_frame(rng, capacity=4, forbidden=used, small=True)
        used.add(small['key'])
        used.add(_mirror(small)['key'])
        for i, q in enumerate([1, 3]):
            objects.append(_object(rng, small, int(rng.integers(4)), q,
                                   (f, p) if i == 0 else (p, f), 1, 'compact',
                                   'free' if i == 0 else 'clip'))
        single = _sample_frame(rng, capacity=3, forbidden=used, small=True)
        objects.append(_object(rng, single, int(rng.integers(4)),
                               int(rng.integers(1, min(5, len(single['cavity'])))),
                               (f, p), 2))
        objects[-1]['silent_guard'] = 2
    return _finish_objects(rng, objects)


def _ordinary_objects(rng, bg, test, forbidden=()):
    objects, used = [], set(forbidden)
    class_count = int(rng.choice([1, 2, 2, 3]))
    slots = 8
    for cid in range(class_count):
        if slots <= 0:
            break
        size = int(rng.integers(2, min(4, slots) + 1)) if cid == 0 else int(rng.integers(1, min(3, slots) + 1))
        low = int(rng.integers(1, 4))
        spread = int(rng.integers(6, 10)) if test and cid == 0 else int(rng.integers(1, 6))
        high = low + spread
        base = _sample_frame(rng, capacity=high + 1, forbidden=used,
                             small=cid > 0 and high <= 4)
        used.add(base['key'])
        quantities = [low, high] + [int(rng.integers(low, high + 1)) for _ in range(size - 2)] if size > 1 else [low]
        turns = [int(rng.integers(4)) for _ in range(size)]
        common = _colors(rng, bg)
        for i, q in enumerate(quantities):
            color = common if rng.random() < .35 else _colors(rng, bg)
            objects.append(_object(rng, base, turns[i], q, color, cid,
                                   str(rng.choice(['random', 'compact', 'scatter'])),
                                   'free' if test and cid == 0 and i == 1 else None))
        slots -= size
    if slots and rng.random() < .8:
        base = _sample_frame(rng, capacity=3, forbidden=used, small=True)
        q = int(rng.integers(1, min(6, len(base['cavity']))))
        objects.append(_object(rng, base, int(rng.integers(4)), q, _colors(rng, bg), class_count))
    return _finish_objects(rng, objects)


def _positions(rng, o, shape):
    candidates = []
    for r in range(shape[0] - o['height'] + 1):
        for c in range(shape[1] - o['width'] + 1):
            tip = o['tip'][0] + r, o['tip'][1] + c
            dr, dc = o['direction']
            room = (tip[0] if dr == -1 else shape[0] - 1 - tip[0] if dr == 1
                    else tip[1] if dc == -1 else shape[1] - 1 - tip[1])
            surplus, mode = o['surplus'], o['mode']
            if (room < o['silent_guard'] or (mode == 'free' and room <= surplus) or (mode == 'exact' and room != surplus)
                    or (mode == 'clip' and not 0 < room < surplus)
                    or (mode == 'zero' and room != 0)):
                continue
            candidates.append((r, c, tip))
    return [candidates[int(i)] for i in rng.permutation(len(candidates))]


def _pack(rng, objects, shape):
    order = sorted(range(len(objects)), key=lambda i: -(objects[i]['height'] * objects[i]['width']
                                                     + objects[i]['surplus'] * 3 + float(rng.random())))
    boxes, guards, rays, placed = set(), set(), set(), {}
    for i in order:
        o = objects[i]
        accepted = None
        for r, c, tip in _positions(rng, o, shape):
            box = {(r+y, c+x) for y in range(o['height']) for x in range(o['width'])}
            if box & (guards | rays):
                continue
            # Reserve an empty endpoint witness for every complete positive ray,
            # even if it happened to fit under an unconstrained placement mode.
            length = max(o['surplus'] + int(o['surplus'] > 0), o['silent_guard'])
            ray_guard = set(_ray(shape, tip, o['direction'], length))
            if ray_guard & (boxes | rays):
                continue
            accepted = r, c, tip, box, ray_guard
            break
        if accepted is None:
            return None
        r, c, tip, box, ray_guard = accepted
        placed[i] = r, c, tip
        boxes.update(box)
        guards.update((r+y, c+x) for y in range(-1, o['height'] + 1)
                      for x in range(-1, o['width'] + 1))
        rays.update(ray_guard)
    return placed


def _render(rng, objects, bg):
    occupied = sum(len(o['frame']) + o['q'] for o in objects)
    for attempt in range(26):
        if attempt >= 8:
            shape = (30, 30)
        else:
            shape = int(rng.integers(13, 31)), int(rng.integers(13, 31))
        if occupied * 2 >= shape[0] * shape[1]:
            continue
        placed = _pack(rng, objects, shape)
        if placed is None:
            continue
        inp = np.full(shape, bg, dtype=np.int8)
        out = inp.copy()
        for i, o in enumerate(objects):
            r, c, tip = placed[i]
            for y, x in o['frame']:
                inp[r+y, c+x] = out[r+y, c+x] = o['frame_color']
            for y, x in o['payload']:
                inp[r+y, c+x] = o['payload_color']
            for point in _ray(shape, tip, o['direction'], o['surplus']):
                out[point] = o['payload_color']
        observed_bg, observed = _read_scene(inp)
        expected = {(placed[i][2], o['direction'], o['q'], o['surplus'],
                     o['frame_color'], o['payload_color']) for i, o in enumerate(objects)}
        actual = {(o['tip'], o['direction'], o['q'], o['surplus'],
                   o['frame_color'], o['payload_color']) for o in observed}
        assert observed_bg == bg and expected == actual, "construction/reader disagreement"
        assert np.array_equal(solve(inp), out), "directly painted label disagrees with input-only solve"
        return {'input': inp, 'output': out}
    return None


def _count(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.uint8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    stable_bg = int(rng.integers(10)) if rng.random() < .3 else None
    base = _sample_frame(rng, capacity=12, chiral=True, anchor=True)
    b = int(rng.integers(1, 3))
    witness_order = list(int(i) for i in rng.permutation(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for _ in range(40):
                bg = stable_bg if stable_bg is not None else int(rng.integers(10))
                if split == 'train' and index < 2:
                    objects = _witness_objects(rng, base, b, witness_order[index], bg)
                else:
                    objects = _ordinary_objects(rng, bg, split == 'test', forbidden={base['key']})
                pair = _render(rng, objects, bg)
                if pair is not None and _fingerprint(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError("bounded scene construction could not find a distinct fit")
            seen.add(_fingerprint(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _alternative(grid, name, scene=None):
    """Finite counterexample checks; this is not an independent reference solver."""
    bg, objects = _read_scene(grid) if scene is None else scene
    charges, keys = [], []
    for o in objects:
        charges.append(len(_components(o['payload'])) if name == 'components' else o['q'])
        if name == 'global':
            key = 0
        elif name == 'area':
            key = len(o['frame'])
        elif name == 'color':
            key = o['frame_color']
        elif name == 'reflection':
            key = min(o['key'], _rotation_key(np.fliplr(o['mask'])))
        elif name == 'identity':
            key = _mask_key(o['mask'])
        elif name == 'combined':
            key = _rotation_key(_mask(o['frame'] | o['payload']))
        elif name == 'independent':
            key = min(o['frame'])
        else:
            key = o['key']
        keys.append(key)
    groups = defaultdict(list)
    for key, q in zip(keys, charges):
        groups[key].append(q)
    output = grid.copy()
    for o in objects:
        for point in o['payload']:
            output[point] = bg
    for o, key, q in zip(objects, keys, charges):
        qs = groups[key]
        baseline = min(qs)
        length = q - baseline
        if name == 'whole_count':
            length = q
        elif name == 'fixed':
            length = int(length > 0)
        elif name == 'maximum':
            length = max(qs) - q
        elif name == 'mean':
            length = max(0, q - sum(qs) // len(qs))
        elif name == 'mode':
            counts = Counter(qs)
            mode = min(q for q in counts if counts[q] == max(counts.values()))
            length = max(0, q - mode)
        elif name == 'exactly_two' and len(qs) != 2:
            length = 0
        elif name == 'to_edge' and length:
            length = o['room']
        for point in _ray(grid.shape, o['tip'], o['direction'], length):
            output[point] = o['payload_color']
    return output


def _training_evidence(train):
    full_surpluses, multiplicities = set(), set()
    clipped = False
    arrangements, changed_baselines = defaultdict(set), defaultdict(set)
    scenes = [_read_scene(np.asarray(p['input'])) for p in train]
    for pair, (bg, objects) in zip(train, scenes):
        inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
        multiplicities.update(Counter(o['key'] for o in objects).values())
        for o in objects:
            clipped |= 0 < o['room'] < o['surplus']
            if o['surplus'] and o['room'] > o['surplus']:
                full_surpluses.add(o['surplus'])
                after = _ray(inp.shape, o['tip'], o['direction'], o['surplus'] + 1)[-1]
                assert out[after] == bg, "missing background after a complete teaching ray"
                arrangements[(o['key'], o['q'], o['surplus'])].add(len(_components(o['payload'])))
                changed_baselines[(o['key'], o['q'])].add(o['surplus'])
            else:
                changed_baselines[(o['key'], o['q'])].add(o['surplus'])
    assert {1, 2, 3, 5} <= full_surpluses, "count subtraction is not sufficiently visible"
    assert clipped, "boundary clipping is not demonstrated"
    assert {1, 3, 4} <= multiplicities, "singleton/triplet/quartet evidence missing"
    assert any(len(v) > 1 for v in arrangements.values()), "equal charges lack arrangement contrast"
    assert any(len(v) > 1 for v in changed_baselines.values()), "same shell/count never changes baseline"
    alternatives = ('whole_count', 'fixed', 'global', 'area', 'color', 'reflection',
                    'identity', 'combined', 'components', 'maximum', 'mean', 'mode',
                    'exactly_two', 'independent', 'to_edge')
    for name in alternatives:
        assert any(not np.array_equal(_alternative(np.asarray(p['input']), name, scene), p['output'])
                   for p, scene in zip(train, scenes)), f"training admits {name} shortcut"


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _fingerprint(inp)
            assert key not in seen, "duplicate input"
            seen.add(key)
            bg, objects = _read_scene(inp)
            assert 1 <= len(objects) <= 8
            assert max(Counter(o['key'] for o in objects).values()) <= 4
            unchanged = np.ones(inp.shape, dtype=bool)
            for o in objects:
                for p in o['frame']:
                    assert out[p] == o['frame_color'], "shell changed"
                for p in o['payload']:
                    assert out[p] == bg, "payload not cleared"
                    unchanged[p] = False
                for p in o['ray']:
                    assert out[p] == o['payload_color'], "surplus ray has wrong color/length"
                    unchanged[p] = False
                for other in objects:
                    if other is o:
                        continue
                    assert all((r+dr, c+dc) not in other['box']
                               for r, c in o['box'] for dr, dc in _D8), "shell boxes touch"
            assert np.array_equal(inp[unchanged], out[unchanged]), "unrelated cell changed"
            assert np.array_equal(solve(inp), out)
            if split == 'test':
                assert any(o['surplus'] >= 6 and o['room'] > o['surplus']
                           and out[_ray(inp.shape, o['tip'], o['direction'], o['surplus'] + 1)[-1]] == bg
                           for o in objects), "test hides the extrapolated surplus"
            else:
                assert max(o['surplus'] for o in objects) <= 5
    _training_evidence(puzzle['train'])


def _puzzle_key(puzzle):
    return tuple((_fingerprint(p['input']), _fingerprint(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded 42-seed evidence, diversity, equivariance and interface checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    dimensions, profiles, backgrounds, directions = set(), set(), set(), set()
    counts, multiplicities, quantities, surpluses, components, modes = set(), set(), set(), set(), set(), set()
    within_profiles = 0
    for puzzle in puzzles:
        local_profiles = set()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                bg, objects = _read_scene(inp)
                dimensions.add(inp.shape)
                backgrounds.add(bg)
                counts.add(len(objects))
                class_sizes = tuple(sorted(Counter(o['key'] for o in objects).values()))
                multiplicities.update(class_sizes)
                local_profiles.add(class_sizes)
                for o in objects:
                    profiles.add(o['key'])
                    directions.add(o['direction'])
                    quantities.add(o['q'])
                    surpluses.add(o['surplus'])
                    components.add(len(_components(o['payload'])))
                    if o['surplus']:
                        modes.add('zero' if not o['room'] else 'clip' if o['room'] < o['surplus']
                                  else 'exact' if o['room'] == o['surplus'] else 'free')
        within_profiles += len(local_profiles) > 1
    assert len(dimensions) >= 25 and len(profiles) >= 60
    assert backgrounds == set(range(10)) and len(directions) == 4
    assert len(counts) >= 4 and multiplicities == {1, 2, 3, 4}
    assert max(quantities) >= 10 and {0, 1, 2, 3, 5, 6} <= surpluses
    assert max(components) >= 3 and within_profiles == 42
    assert modes == {'zero', 'clip', 'exact', 'free'}
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(281005803, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        for turn in range(4):
            inp = (np.rot90(pair['input'], turn).astype(int) + 3) % 10
            expected = (np.rot90(pair['output'], turn).astype(int) + 3) % 10
            assert np.array_equal(solve(inp), expected), "rotation/palette equivariance failed"
    # Change only a minimum companion. A remote member must change its ray.
    pair = next(p for p in puzzles[0]['train']
                if 4 in Counter(o['key'] for o in _read_scene(p['input'])[1]).values())
    inp = pair['input'].copy()
    _, objects = _read_scene(inp)
    groups = defaultdict(list)
    for o in objects:
        groups[o['key']].append(o)
    group = next(g for g in groups.values() if len(g) == 4)
    lowest = min(group, key=lambda o: o['q'])
    remote = max(group, key=lambda o: o['q'])
    spare = sorted(lowest['cavity'] - lowest['payload'])[0]
    inp[spare] = lowest['payload_color']
    prediction = solve(inp)
    assert all(inp[p] == pair['input'][p] for p in remote['box'])
    assert any(prediction[p] != pair['output'][p] for p in remote['ray']), "companionship had no quantitative effect"
    bad = {s: [{r: a.copy() for r, a in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("corrupted label accepted")
    return {'seeds': 42, 'pairs': sum(len(p[s]) for p in puzzles for s in p),
            'dimensions': len(dimensions), 'shell_profiles': len(profiles),
            'object_counts': sorted(counts), 'payload_counts': sorted(quantities),
            'surpluses': sorted(surpluses), 'clearance_modes': sorted(modes)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
