"""Registered color carry: translate a reservoir, then propagate contact colors.

Only standard-library modules and NumPy are used. The solver never calls the
sampler, reads files, uses a seed, or learns conventions from stored examples.
"""
from collections import deque
import numpy as np


ORTH = ((-1, 0), (0, 1), (1, 0), (0, -1))
EIGHT = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0, 788017129, 494292226, 22305269, 946542015, 548225057,
              265711560, 971788447, 293791877, 183194933, 568867960,
              781399656, 59731009, 155788916, 312519388, 501974574,
              8566151, 497101065, 748406198, 990316055, 880689913,
              91705974, 284942321, 85049902, 362024157, 448038202,
              498015853, 688176913, 899055448, 78413135, 745811957,
              379484006, 964915646, 940337384, 477618365, 412732444,
              715357435, 651321561, 690922686, 93309177, 484113092,
              796618861)


def _add(p, q):
    return p[0] + q[0], p[1] + q[1]


def _sub(p, q):
    return p[0] - q[0], p[1] - q[1]


def _components(cells, directions=EIGHT):
    remaining = set(cells)
    parts = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        part, stack = {start}, [start]
        while stack:
            p = stack.pop()
            for step in directions:
                q = _add(p, step)
                if q in remaining:
                    remaining.remove(q)
                    part.add(q)
                    stack.append(q)
        parts.append(part)
    return parts


def _bbox(cells):
    return (min(p[0] for p in cells), max(p[0] for p in cells),
            min(p[1] for p in cells), max(p[1] for p in cells))


def _key(grid):
    a = np.asarray(grid)
    return a.shape, tuple(map(int, a.ravel()))


def _check_grid(grid):
    a = np.asarray(grid)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('grid must be an integer 1..30 by 1..30 array with colors 0..9')
    return a


def _trace(cells):
    cells = set(cells)
    if not 2 <= len(cells) <= 30:
        raise ValueError('carrier length must be 2..30')
    neighbors = {p: [_add(p, d) for d in ORTH if _add(p, d) in cells] for p in cells}
    ends = sorted(p for p, ns in neighbors.items() if len(ns) == 1)
    if (len(ends) != 2 or any(len(ns) not in (1, 2) for ns in neighbors.values())
            or ends[0][0] == ends[1][0]):
        raise ValueError('carrier must be a simple chain with unequal endpoint rows')
    start = min(ends, key=lambda p: p[0])
    ordered, visited = [], set()
    previous, current = None, start
    while current not in visited:
        ordered.append(current)
        visited.add(current)
        options = [q for q in neighbors[current] if q != previous]
        if not options:
            break
        previous, current = current, options[0]
    if len(ordered) != len(cells):
        raise ValueError('carrier is disconnected or contains a cycle')
    return ordered


def _bar(cells):
    cells = sorted(cells)
    if len(cells) < 2:
        raise ValueError('a bar must expose its axis with at least two cells')
    if len({p[0] for p in cells}) == 1:
        axis, normal = 0, (1, 0)
    elif len({p[1] for p in cells}) == 1:
        axis, normal = 1, (0, 1)
    else:
        raise ValueError('guide must be straight')
    tangent = 1 - axis
    values = sorted(p[tangent] for p in cells)
    if values != list(range(values[0], values[0] + len(cells))):
        raise ValueError('guide must be contiguous')
    return axis, normal, cells[0]


def _parse(input_grid):
    """Recover geometry, roles, registration and path solely from input pixels."""
    grid = _check_grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] * 2 <= grid.size:
        raise ValueError('background must be a strict majority')
    objects = _components(map(tuple, np.argwhere(grid != bg)))
    if len(objects) != 2:
        raise ValueError('input must contain two separated eight-connected objects')
    color_sets = [{int(grid[p]) for p in obj} for obj in objects]
    common = color_sets[0] & color_sets[1]
    if len(common) != 1:
        raise ValueError('objects must share only their guide color')
    guide = next(iter(common))
    carrier_ids = [i for i, cs in enumerate(color_sets) if len(cs - {guide}) == 1]
    if len(carrier_ids) != 1:
        raise ValueError('exactly one object must have a monochrome carrier')
    ci = carrier_ids[0]
    pi = 1 - ci
    if len(color_sets[pi] - {guide}) < 2:
        raise ValueError('reservoir must contain at least two pigment colors')
    carrier_color = next(iter(color_sets[ci] - {guide}))
    chain = {p for p in objects[ci] if int(grid[p]) != guide}
    paint = {p: int(grid[p]) for p in objects[pi] if int(grid[p]) != guide}
    bars = [{p for p in obj if int(grid[p]) == guide} for obj in (objects[ci], objects[pi])]
    axis, normal, ca = _bar(bars[0])
    pa_axis, _, pa = _bar(bars[1])
    if axis != pa_axis or len(bars[0]) != len(bars[1]):
        raise ValueError('bars must be parallel and congruent')
    signs = []
    for bar, ordinary in zip(bars, (chain, set(paint))):
        neg = any(_sub(p, normal) in ordinary for p in bar)
        pos = any(_add(p, normal) in ordinary for p in bar)
        if neg == pos:
            raise ValueError('each bar must have one backing side and one clear facing side')
        signs.append(-1 if neg else 1)
    if signs[0] == signs[1]:
        raise ValueError('backing sides must be opposite')
    backing = (signs[0] * normal[0], signs[0] * normal[1])
    delta = _sub(_add(ca, backing), pa)
    registered = {_add(p, delta): color for p, color in paint.items()}
    registered_guide = {_add(p, delta) for p in bars[1]}
    path = _trace(chain)
    return {'grid': grid, 'background': bg, 'guide': guide, 'carrier': carrier_color,
            'path': path, 'paint': paint, 'registered': registered, 'delta': delta,
            'normal': normal, 'backing': backing, 'axis': axis, 'length': len(bars[0]),
            'bars': bars, 'objects': objects, 'registered_guide': registered_guide}


def _read(order, paint, carrier, persistence=True):
    current = carrier
    sequence = []
    for p in order:
        if p in paint:
            current = paint[p]
        elif not persistence:
            current = carrier
        sequence.append(int(current))
    return sequence


def solve(input_grid: np.ndarray) -> np.ndarray:
    p = _parse(input_grid)
    return np.asarray(_read(p['path'], p['registered'], p['carrier']), dtype=int).reshape(-1, 1)


def _scan_orders(path):
    orders = []
    for axis in (0, 1):
        for sign_a in (-1, 1):
            for sign_b in (-1, 1):
                orders.append(sorted(path, key=lambda q: (sign_a * q[axis], sign_b * q[1 - axis])))
    columns = sorted({p[1] for p in path})
    for cs in (columns, columns[::-1]):
        for reverse_last in (False, True):
            orders.append([p for c in cs for p in sorted(
                (p for p in path if p[1] == c), reverse=reverse_last and c == cs[-1])])
    return orders


def _alternatives(p):
    order, paint, carrier = p['path'], p['registered'], p['carrier']
    result = {'no_registration': _read(order, p['paint'], carrier),
              'no_propagation': _read(order, paint, carrier, False),
              'reverse': _read(order[::-1], paint, carrier)}
    # Move the correctly registered reservoir back onto the anchor bar, or to
    # the other adjacent line. Every coordinate changes, not just guide pixels.
    for name, factor in (('coincident_guides', 1), ('opposite_seam', 2)):
        correction = (-factor * p['backing'][0], -factor * p['backing'][1])
        wrong_paint = {_add(q, correction): c for q, c in paint.items()}
        result[name] = _read(order, wrong_paint, carrier)
    for i, scan in enumerate(_scan_orders(order)):
        result[f'scan_{i}'] = _read(scan, paint, carrier)
    return result


def _features(p):
    order, paint, carrier = p['path'], p['registered'], p['carrier']
    seq = _read(order, paint, carrier)
    events = [(i, paint[q]) for i, q in enumerate(order) if q in paint]
    steps = [_sub(b, a) for a, b in zip(order, order[1:])]
    runs = [c for i, c in enumerate(seq) if i == 0 or c != seq[i - 1]]
    event_runs = [c for i, (_, c) in enumerate(events) if i == 0 or c != events[i - 1][1]]
    colors_at_contacts = {c for _, c in events}
    boxes = [_bbox(obj) for obj in p['objects']]
    a, b = boxes
    xs = a[3] < b[2] or b[3] < a[2]
    ys = a[1] < b[0] or b[1] < a[0]
    stage = 'diagonal' if xs and ys else 'horizontal' if xs else 'vertical' if ys else 'interleaved'
    return {'n': len(order), 'contacts': len(events), 'events': events, 'sequence': seq,
            'turns': sum(a != b for a, b in zip(steps, steps[1:])),
            'above': min(q[0] for q in order) < order[0][0],
            'reversal': (0, 1) in steps and (0, -1) in steps,
            'internal_up': any(a == (-1, 0) and b != a for a, b in zip(steps, steps[1:])),
            'propagated': sum(q not in paint and c != carrier for q, c in zip(order, seq)),
            'off_path_colors': set(paint.values()) - colors_at_contacts,
            'recurrence': len(runs) > len(set(runs)),
            'contact_recurrence': len(event_runs) > len(set(event_runs)),
            'guide_preserved': len(set(order) & p['registered_guide']),
            'axis': p['axis'], 'bar': p['length'], 'stage': stage,
            'paint_size': len(paint), 'input_shape': p['grid'].shape,
            'roles': (p['background'], carrier, p['guide']),
            'edge': any(q[0] in (0, p['grid'].shape[0] - 1) or
                        q[1] in (0, p['grid'].shape[1] - 1)
                        for obj in p['objects'] for q in obj)}


def _witness(p):
    answer = _read(p['path'], p['registered'], p['carrier'])
    return all(answer != alternative for alternative in _alternatives(p).values())


def _walk(rng, n):
    path = [(0, 0)]
    cells = set(path)
    last = None
    persistence = float(rng.uniform(.6, 8.0))
    for _ in range(n - 1):
        options, weights = [], []
        for d in ORTH:
            q = _add(path[-1], d)
            if q in cells or sum(_add(q, step) in cells for step in ORTH) != 1:
                continue
            options.append((q, d))
            weights.append(persistence if d == last else 1.0)
        if not options:
            return None
        weights = np.asarray(weights)
        q, last = options[int(rng.choice(len(options), p=weights / weights.sum()))]
        path.append(q)
        cells.add(q)
    return path


def _normalize_at_bar(rng, path, length):
    cells = set(path)
    options = []
    for r, c in sorted(cells):
        if not all((r, c + i) in cells for i in range(length)):
            continue
        for sign in (-1, 1):
            marker_row = r + sign
            if any((r + sign, c + i) in cells or (r + 2 * sign, c + i) in cells
                   for i in range(length)):
                continue
            options.append((marker_row, c, sign))
    if not options:
        return None
    r0, c0, sign = options[int(rng.integers(len(options)))]
    return [(sign * (r - r0), c - c0) for r, c in path]


def _rotate_point(p, rotation):
    r, c = p
    for _ in range(rotation):
        r, c = -c, r
    return r, c


def _orient_for_scene(path, rotation):
    a, b = _rotate_point(path[0], rotation), _rotate_point(path[-1], rotation)
    if a[0] == b[0]:
        return None
    return path if a[0] < b[0] else path[::-1]


def _above_in_scene(path, rotation):
    points = [_rotate_point(p, rotation) for p in path]
    return min(p[0] for p in points) < min(points[0][0], points[-1][0])


def _fallback_paths(length, minimum):
    """Construct fresh geometric chains; no saved pairs or colors are used."""
    width = length + 1
    segment_lists = [
        [((0, 1), width), ((1, 0), 3), ((0, -1), width - 1),
         ((1, 0), 2), ((0, 1), 1)],
        [((1, 0), 3), ((0, 1), width), ((-1, 0), 2),
         ((0, 1), 2), ((1, 0), 3)],
        [((-1, 0), 2), ((0, 1), width + 1), ((1, 0), 4),
         ((0, -1), width), ((1, 0), 2)],
    ]
    for segments in segment_lists:
        path = [(0, 0)]
        for step, count in segments:
            for _ in range(count):
                path.append(_add(path[-1], step))
        # Extend an exposed endpoint while preserving an induced graph.
        while len(path) < minimum:
            occupied = set(path)
            candidates = [_add(path[-1], d) for d in ((1, 0), (0, 1), (0, -1), (-1, 0))]
            options = [q for q in candidates if q not in occupied
                       and sum(_add(q, d) in occupied for d in ORTH) == 1]
            if not options:
                break
            path.append(options[0])
        if len(path) <= 30:
            yield path


def _connect(rng, mask, target, allowed):
    if target in mask:
        return True
    queue = deque([target])
    parent = {target: None}
    directions = [EIGHT[int(i)] for i in rng.permutation(len(EIGHT))]
    finish = None
    while queue:
        p = queue.popleft()
        if p in mask:
            finish = p
            break
        for d in directions:
            q = _add(p, d)
            if q in allowed and q not in parent:
                parent[q] = p
                queue.append(q)
    if finish is None:
        return False
    while finish is not None:
        mask.add(finish)
        finish = parent[finish]
    return True


def _reservoir(rng, path, length, rotation, palette, growth, witness):
    ordered = _orient_for_scene(path, rotation)
    if ordered is None:
        return None
    # These are the reservoir's guide and clear facing strip after registration.
    forbidden = {(r, c) for r in (-1, -2) for c in range(length)}
    available_indices = [i for i, q in enumerate(ordered) if q not in forbidden]
    if witness:
        available_indices = [i for i in available_indices if i > 0]
    if len(available_indices) < 3:
        return None
    count = int(rng.integers(3, min(8, len(available_indices)) + 1))
    indices = sorted(map(int, rng.choice(available_indices, count, replace=False)))
    contacts = {ordered[i] for i in indices}
    t, b, l, r = _bbox(set(path) | {(0, c) for c in range(length)})
    domain = {(y, x) for y in range(t - 2, b + 3) for x in range(l - 2, r + 3)}
    allowed = domain - forbidden - (set(path) - contacts)
    # At least one backing seed is present; full and partial backing both occur.
    backed = sorted(map(int, rng.choice(length, int(rng.integers(1, length + 1)), replace=False)))
    mask = {(0, c) for c in backed}
    for q in sorted(contacts):
        if not _connect(rng, mask, q, allowed):
            return None
    # Vary outline, area and thickness without permitting extra hidden contacts.
    for _ in range(growth):
        frontier = sorted({_add(q, d) for q in mask for d in EIGHT} & allowed - mask)
        if not frontier:
            break
        mask.add(frontier[int(rng.integers(len(frontier)))])
    if len(_components(mask)) != 1:
        return None
    decoy = palette[-1]
    used = palette[:-1]
    centers = [sorted(mask)[int(rng.integers(len(mask)))] for _ in used]
    paint = {q: used[min(range(len(used)), key=lambda j:
                        abs(q[0] - centers[j][0]) + abs(q[1] - centers[j][1]))] for q in mask}
    contact_colors = [used[int(rng.integers(len(used)))] for _ in indices]
    # Recurrence is explicit in the combined witness; elsewhere contacts are free.
    if witness:
        contact_colors[:3] = [used[0], used[1], used[0]]
    else:
        contact_colors[:2] = [used[0], used[1]]
    for i, color in zip(indices, contact_colors):
        paint[ordered[i]] = color
    off_path = sorted(mask - set(path))
    if not off_path:
        return None
    # Decoy pigment has a spatial region but can never be a contact.
    center = off_path[int(rng.integers(len(off_path)))]
    radius = int(rng.integers(0, 3))
    for q in off_path:
        if abs(q[0] - center[0]) + abs(q[1] - center[1]) <= radius:
            paint[q] = decoy
    return paint


def _stage(rng, path, paint, length, colors, stage):
    bg, carrier, guide = colors
    guides = {(0, c) for c in range(length)}
    carrier_object = {q: carrier for q in path}
    carrier_object.update({q: guide for q in guides})
    # Paint is currently registered; move its guide back to local row zero.
    reservoir_object = {(q[0] + 1, q[1]): color for q, color in paint.items()}
    reservoir_object.update({q: guide for q in guides})
    objects = []
    for obj in (carrier_object, reservoir_object):
        t, b, l, r = _bbox(obj)
        objects.append(({(y - t, x - l): color for (y, x), color in obj.items()},
                        b - t + 1, r - l + 1))
    a, ah, aw = objects[0]
    b, bh, bw = objects[1]
    avoid = {_add(q, d) for q in a for d in EIGHT + ((0, 0),)}
    candidates = []
    if stage == 'interleaved':
        options = [(dy, dx) for dy in range(-bh + 1, ah) for dx in range(-bw + 1, aw)]
        for i in rng.permutation(len(options))[:120]:
            candidates.append(options[int(i)])
    for _ in range(20):
        gap = int(rng.integers(1, 7))
        if stage == 'horizontal':
            offset = (int(rng.integers(-bh + 1, ah)), aw + gap)
        elif stage == 'diagonal':
            offset = (ah + gap, aw + gap)
        else:
            offset = (ah + gap, int(rng.integers(-bw + 1, aw)))
        if rng.integers(2):
            offset = (-offset[0] - bh + ah, -offset[1] - bw + aw)
        candidates.append(offset)
    for offset in candidates:
        shifted = {_add(q, offset): color for q, color in b.items()}
        if set(shifted) & avoid:
            continue
        combined = dict(a)
        combined.update(shifted)
        t, bottom, l, right = _bbox(combined)
        h, w = bottom - t + 1, right - l + 1
        if h > 30 or w > 30:
            continue
        pads = [int(rng.integers(0, 4)) for _ in range(4)]
        pads[0] = min(pads[0], 30 - h)
        pads[1] = min(pads[1], 30 - h - pads[0])
        pads[2] = min(pads[2], 30 - w)
        pads[3] = min(pads[3], 30 - w - pads[2])
        height, width = h + pads[0] + pads[1], w + pads[2] + pads[3]
        if rng.random() < .15:
            if rng.integers(2):
                height = 30
            else:
                width = 30
        while height * width <= 2 * len(combined):
            if height < 30 and (height <= width or width == 30):
                height += 1
            elif width < 30:
                width += 1
            else:
                break
        if height * width <= 2 * len(combined):
            continue
        grid = np.full((height, width), bg, dtype=int)
        for (y, x), color in combined.items():
            grid[y - t + pads[0], x - l + pads[2]] = color
        return grid
    return None


def _make_pair(rng, profile, used):
    for attempt in range(300):
        rotation = profile['axis'] + 2 * int(rng.integers(2))
        if attempt < 240:
            path = _walk(rng, int(rng.integers(profile['low'], profile['high'] + 1)))
        else:
            fallbacks = list(_fallback_paths(profile['bar'], profile['low']))
            path = fallbacks[(attempt - 240) % len(fallbacks)] if fallbacks else None
        if path is None or len(path) > profile['high']:
            continue
        path = _normalize_at_bar(rng, path, profile['bar'])
        if path is None or _orient_for_scene(path, rotation) is None:
            continue
        above = _above_in_scene(path, rotation)
        if (above and not profile['allow_above']) or (profile['need_above'] and not above):
            continue
        growth = int(rng.integers(*profile['growth']))
        paint = _reservoir(rng, path, profile['bar'], rotation,
                           profile['pigments'], growth, profile['witness'])
        if paint is None:
            continue
        stage = profile['stage']
        if stage == 'free':
            stage = str(rng.choice(('horizontal', 'vertical', 'diagonal', 'interleaved')))
        staged = _stage(rng, path, paint, profile['bar'], profile['colors'], stage)
        if staged is None:
            continue
        grid = staged
        grid = np.rot90(grid, rotation).copy()
        if rng.integers(2):
            grid = np.fliplr(grid).copy()
        if _key(grid) in used:
            continue
        try:
            parsed = _parse(grid)
        except ValueError:
            continue
        # The output uses known construction contacts and their oriented path;
        # a separate assertion checks recovery from actual staged input pixels.
        ordered = _orient_for_scene(path, rotation)
        expected = np.asarray(_read(ordered, paint, profile['colors'][1]), dtype=int).reshape(-1, 1)
        if not np.array_equal(expected, solve(grid)):
            raise AssertionError('registration or construction oracle disagrees with input pixels')
        f = _features(parsed)
        if len({c for _, c in f['events']}) < 2 or f['propagated'] == 0:
            continue
        if profile['witness'] and not _witness(parsed):
            continue
        if profile['train_first'] and (expected[0, 0] != profile['colors'][1]
                                      or not f['contact_recurrence'] or not f['off_path_colors']):
            continue
        return {'input': grid, 'output': expected}
    raise RuntimeError('bounded construction exhausted; no semantic exception was substituted')


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f'illegal {name}')
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(('length', 'bar', 'placement', 'density')))
    changing_palette = bool(rng.integers(2))
    mixed_axis = bool(rng.integers(2))
    allow_above = bool(rng.random() < .4)
    axis = int(rng.integers(2))
    shared = list(map(int, rng.permutation(10)))
    preferred_stage = str(rng.choice(('horizontal', 'vertical')))
    puzzle, used = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            test = split == 'test'
            first = index == 0
            palette = list(map(int, rng.permutation(10))) if changing_palette else shared
            npigment = int(rng.integers(3, 7))
            low, high = (10, 24)
            if regime == 'length':
                low, high = (24, 30) if test else (10, 20)
            if first:
                low = max(14, low)
            bar = int(rng.integers(2, 6))
            if regime == 'bar':
                bar = 3 if test else (2 if index % 2 == 0 else 4)
            if regime == 'length' and not test:
                bar = int(rng.integers(2, 4))
            growth = (0, 45)
            if regime == 'density':
                growth = (45, 76) if test else (0, 12)
            stage = 'free'
            if regime == 'placement':
                stage = ('vertical' if preferred_stage == 'horizontal' else 'horizontal') if test else preferred_stage
            profile = {'low': low, 'high': high, 'bar': bar,
                       'colors': tuple(palette[:3]), 'pigments': palette[3:3 + npigment],
                       'axis': (axis + index % 2) % 2 if mixed_axis else axis,
                       'allow_above': allow_above, 'need_above': allow_above and first,
                       'witness': first, 'train_first': not test and first,
                       'growth': growth, 'stage': stage}
            pair = _make_pair(rng, profile, used)
            puzzle[split].append(pair)
            used.add(_key(pair['input']))
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _check_grid(pair['input']), _check_grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            p = _parse(inp)
            f = _features(p)
            assert out.shape == (f['n'], 1), 'one row per carrier cell'
            assert np.array_equal(out[:, 0], f['sequence']), 'wrong contact carry sequence'
            assert len({c for _, c in f['events']}) >= 2, 'unrevealing contact coloring'
            assert f['propagated'] > 0, 'no visible persistence witness'
            assert f['guide_preserved'] > 0, 'no ordinary carrier under the registered guide'
            assert p['guide'] not in out, 'guides must be temporary'
            assert p['background'] not in out, 'background is never loaded'
            assert all(color not in out for color in f['off_path_colors'])
            # Every guide-covered carrier cell contributes exactly one output
            # row. Pigment at the other guide's positions remains ordinary paint.
            assert set(p['registered_guide']) & set(p['path'])
            records[split].append((p, f))
    train, test = records['train'], records['test']
    assert any(_witness(p) for p, _ in train), 'missing combined training witness'
    assert any(_witness(p) for p, _ in test), 'test admits named shortcuts'
    assert any(f['sequence'][0] == p['carrier'] and f['contact_recurrence']
               and f['off_path_colors'] for p, f in train), 'missing initialization/recurrence/gating evidence'
    assert {f['axis'] for _, f in test} <= {f['axis'] for _, f in train}
    if any(f['above'] for _, f in test):
        assert any(f['above'] and f['sequence'] != f['sequence'][::-1] for _, f in train)


def self_test():
    """Bounded 42-seed semantic, evidence, diversity and interface checks."""
    # Explicit, hand-computed domain fixtures do not satisfy the sampler's rich
    # evidence conditions. They must still be solved by the input-only solver.
    fixtures = [
        ([(-2, 0), (-1, 0)], {(0, 0): 4, (0, 1): 5}, 2, [1, 1]),
        ([(-2, -1), (-1, -1), (-1, 0)],
         {(-2, -1): 4, (-2, -2): 5, (-1, -2): 5, (0, -1): 5, (0, 0): 5},
         2, [4, 4, 4]),
        ([(-1, 0), (-1, -1), (0, -1), (1, -1)],
         {(0, -1): 4, (0, 0): 5, (0, 1): 5}, 2, [1, 1, 4, 4]),
        ([(-1, 0), (-1, -1), (0, -1), (1, -1)],
         {(0, -1): 4, **{(0, c): 5 for c in range(6)}}, 6, [1, 1, 4, 4]),
    ]
    for path, paint, length, expected in fixtures:
        grid = np.zeros((16, 20), dtype=int)
        for r, c in path:
            grid[r + 5, c + 5] = 1
        for (r, c), color in paint.items():
            grid[r + 11, c + 11] = color
        for c in range(length):
            grid[5, c + 5] = grid[10, c + 11] = 2
        assert solve(grid).ravel().tolist() == expected
        assert solve(np.fliplr(grid)).ravel().tolist() == expected
    puzzles, features, layouts, paints = [], [], set(), set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                p = _parse(pair['input'])
                features.append(_features(p))
                origin = p['path'][0]
                layouts.add(tuple(_sub(q, origin) for q in p['path']))
                t, b, l, r = _bbox(p['paint'])
                paints.add(tuple(sorted((y - t, x - l) for y, x in p['paint'])))
                assert np.array_equal(solve(np.array(pair['input'].tolist(), dtype=int)), pair['output'])
                # A color bijection must preserve semantics without role tables.
                lut = (np.arange(10) + 3) % 10
                assert np.array_equal(solve(lut[pair['input']]), lut[pair['output']])
                # Horizontal reflection preserves higher-endpoint traversal.
                assert np.array_equal(solve(np.fliplr(pair['input'])), pair['output'])
    fingerprints = {tuple(_key(pair[k]) for s in ('train', 'test')
                          for pair in puzzle[s] for k in ('input', 'output')) for puzzle in puzzles}
    assert len(fingerprints) == 42
    assert len(layouts) >= 120 and len(paints) >= 160
    assert len({f['input_shape'] for f in features}) >= 60
    assert len({f['n'] for f in features}) >= 15
    assert len({f['turns'] for f in features}) >= 8
    assert {f['axis'] for f in features} == {0, 1}
    assert {f['bar'] for f in features} >= {2, 3, 4, 5}
    assert len({f['roles'] for f in features}) >= 40
    assert len({f['contacts'] for f in features}) >= 5
    assert len({f['paint_size'] for f in features}) >= 40
    assert any(f['edge'] for f in features) and any(30 in f['input_shape'] for f in features)
    assert any(f['above'] for f in features) and any(f['internal_up'] for f in features)
    assert any(f['contact_recurrence'] for f in features)
    assert {'horizontal', 'vertical', 'diagonal'} <= {f['stage'] for f in features}
    repeat = generate_puzzle(HOST_SEEDS[7])
    assert all(np.array_equal(a[k], b[k]) for s in repeat
               for a, b in zip(repeat[s], puzzles[7][s]) for k in ('input', 'output'))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(0, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    for seed in (-1, True, 0.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            continue
        raise AssertionError('illegal seed accepted')
    # Incorrect output and input topology/color-role mutations must be rejected.
    example = puzzles[0]
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in example[s]] for s in example}
    bad['test'][0]['output'][0, 0] = (bad['test'][0]['output'][0, 0] + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong output accepted')
    p = _parse(example['train'][0]['input'])
    for wrong in _alternatives(p).values():
        mutant = {s: list(example[s]) for s in example}
        mutant['train'][0] = {'input': example['train'][0]['input'],
                              'output': np.asarray(wrong, dtype=int).reshape(-1, 1)}
        try:
            validate_generated_puzzle(mutant)
        except AssertionError:
            pass
        else:
            raise AssertionError('named semantic alternative accepted')
    broken = example['train'][0]['input'].copy()
    broken[p['path'][len(p['path']) // 2]] = p['background']
    try:
        solve(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('broken chain accepted')
    for cells in ({(1, 1), (1, 2), (1, 3)},
                  {(1, 2), (2, 1), (2, 2), (2, 3)},
                  {(1, 1), (1, 2), (2, 1), (2, 2)},
                  {(1, 1), (2, 1), (4, 4)}):
        try:
            _trace(cells)
        except ValueError:
            continue
        raise AssertionError('invalid carrier graph accepted')
    return {'seeds': 42, 'pairs': len(features), 'path_layouts': len(layouts),
            'reservoir_masks': len(paints), 'input_shapes': len({f['input_shape'] for f in features}),
            'path_lengths': sorted({f['n'] for f in features}),
            'contact_counts': sorted({f['contacts'] for f in features})}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
