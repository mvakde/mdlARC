"""Toll-curtain travel: object extrapolation drives a color-selective odometer.

Only standard-library Python and NumPy are used. Generation paints latent strokes
and runs a cell-by-cell odometer; solve fits the input and uses cumulative tolls.
"""
from __future__ import annotations

import numpy as np


def _need(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _need(np.all((0 <= a) & (a <= 9)), "invalid ARC color")
    return a.astype(np.int8, copy=True)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name + " out of range")
    return int(value)


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.flat))


def _marker_cells(bb):
    t, b, l, r = bb
    return {(t, l), (t, l + 1), (t + 1, l),
            (b, r), (b, r - 1), (b - 1, r)}


def _bounds(mask):
    points = np.argwhere(mask)
    _need(len(points) > 0, "empty bounding box")
    t, l = map(int, points.min(axis=0))
    b, r = map(int, points.max(axis=0))
    return t, b, l, r


def _find_marker(a):
    matches = []
    for color in np.unique(a):
        if color == 0 or np.count_nonzero(a == color) != 6:
            continue
        bb = _bounds(a == color)
        t, b, l, r = bb
        if b - t < 2 or r - l < 2:
            continue
        if set(map(tuple, np.argwhere(a == color))) == _marker_cells(bb):
            matches.append((int(color), bb))
    _need(len(matches) == 1, "expected one canonical pair of L markers")
    return matches[0]


def _put_markers(a, bb, marker):
    for p in _marker_cells(bb):
        a[p] = marker


def _mask(style, h, w, spacing=2):
    """Input-side cell predicates, separate from the latent stroke painter."""
    y, x = np.indices((h, w))
    if style == "filled":
        return np.ones((h, w), dtype=bool)
    if style == "hollow":
        return (y == 0) | (y == h - 1) | (x == 0) | (x == w - 1)
    if style == "comb":
        return (x == 0) | (y % spacing == 0)
    if style == "ladder":
        return (x == 0) | (x == w - 1) | (y % spacing == 0)
    raise AssertionError("unknown curtain style")


def _stroke(style, h, w, spacing, color):
    """Generator-side explicit strokes."""
    a = np.zeros((h, w), dtype=np.int8)
    if style == "filled":
        a[:] = color
    elif style == "hollow":
        a[0] = a[-1] = color
        a[:, 0] = a[:, -1] = color
    elif style in ("comb", "ladder"):
        a[:, 0] = color
        if style == "ladder":
            a[:, -1] = color
        for y in range(0, h, spacing):
            a[y] = color
    else:
        raise AssertionError("unknown curtain style")
    return a


def _linear(values):
    values = list(map(int, values))
    _need(len(values) >= 3, "not enough geometric evidence")
    delta = values[1] - values[0]
    _need(all(v == values[0] + i * delta for i, v in enumerate(values)),
          "non-arithmetic curtain geometry")
    return values[0], delta


def _indices(start, period, width, left, right):
    first = -(-(left - width + 1 - start) // period)
    last = (right - start) // period
    return range(first, last + 1)


def _field(model, bb):
    t, b, l, r = bb
    answer = np.zeros((b - t + 1, r - l + 1), dtype=np.int8)
    for i in _indices(model['start'], model['period'], model['width'], l, r):
        h = model['height'] + i * model['dh']
        y = model['top'] + i * model['dy']
        x = model['start'] + i * model['period']
        _need(3 <= h <= 30, "curtain height leaves the supported domain")
        lo, hi = max(t, y), min(b + 1, y + h)
        xl, xr = max(l, x), min(r + 1, x + model['width'])
        if lo >= hi or xl >= xr:
            continue
        mask = _mask(model['style'], h, model['width'], model['spacing'])
        part = mask[lo - y:hi - y, xl - x:xr - x]
        target = answer[lo - t:hi - t, xl - l:xr - l]
        target[part] = model['palette'][i % len(model['palette'])]
    return answer


def _fit_terrain(a, bb, marker, frame):
    t, b, l, r = bb
    evidence = a.copy()
    evidence[t:b + 1, l:r + 1] = 0
    evidence[evidence == marker] = 0
    _need(not np.any(evidence == frame), "vehicle color occurs in the evidence")
    points = np.argwhere(evidence != 0)
    _need(len(points) > 0, "missing curtain evidence")
    _need(points[:, 1].max() < l or points[:, 1].min() > r,
          "evidence must be on one axial side of the crop")
    cols = np.flatnonzero(np.any(evidence != 0, axis=0))
    groups = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
    _need(4 <= len(groups) <= 6, "expected four to six visible curtains")
    patches, tops, starts, colors = [], [], [], []
    for group in groups:
        ys = np.flatnonzero(np.any(evidence[:, group] != 0, axis=1))
        patch = evidence[ys[0]:ys[-1] + 1, group[0]:group[-1] + 1]
        vals = np.unique(patch[patch != 0])
        _need(len(vals) == 1, "each curtain must be monochrome")
        patches.append(patch)
        tops.append(int(ys[0]))
        starts.append(int(group[0]))
        colors.append(int(vals[0]))
    widths = {p.shape[1] for p in patches}
    _need(len(widths) == 1, "curtain widths must be constant")
    width = widths.pop()
    _need(1 <= width <= 3, "unsupported curtain width")
    start, period = _linear(starts)
    top, dy = _linear(tops)
    height, dh = _linear([p.shape[0] for p in patches])
    _need(period > width and min(p.shape[0] for p in patches) >= 3,
          "curtains must have positive separating gaps and height at least three")
    palette = None
    for n in (2, 3):
        if (len(colors) >= 2 * n and len(set(colors[:n])) == n and
                all(c == colors[i % n] for i, c in enumerate(colors))):
            palette = tuple(colors[:n])
            break
    _need(palette is not None, "color cycle needs two complete repetitions")
    base = dict(start=start, period=period, top=top, dy=dy, height=height,
                dh=dh, width=width, palette=palette, visible=len(groups))
    predictions = []
    for style in ("filled", "hollow", "comb", "ladder"):
        if style == "hollow" and width < 3:
            continue
        if style == "comb" and width < 2 or style == "ladder" and width != 3:
            continue
        for spacing in (2, 3) if style in ("comb", "ladder") else (2,):
            if not all(np.array_equal(p != 0, _mask(style, *p.shape, spacing)) for p in patches):
                continue
            model = dict(base, style=style, spacing=spacing)
            # Validate all traversed objects, including unseen skipped ones.
            needed = list(_indices(start, period, width, l, r)) + list(range(len(groups)))
            _need(all(3 <= height + i * dh <= 30
                      for i in range(min(needed), max(needed) + 1)),
                  "invalid intermediate curtain")
            predictions.append((_field(model, bb), model))
    _need(bool(predictions), "unrecognized curtain construction")
    _need(all(np.array_equal(predictions[0][0], p[0]) for p in predictions),
          "visible shape explanations disagree on the hidden continuation")
    return predictions[0]


def _parse_canonical(a, marker, bb):
    """Read one scene with the travel axis pointing right."""
    t, b, l, r = bb
    clean = a.copy()
    clean[clean == marker] = 0
    outside = clean.copy()
    outside[t:b + 1, l:r + 1] = 0
    road_colors = set(map(int, np.unique(outside))) - {0}
    cars = []
    for color in set(map(int, np.unique(clean))) - road_colors - {0}:
        yt, yb, xl, xr = _bounds(clean == color)
        if not (t < yt < yb < b and l < xl < xr < r and yb - yt >= 2 and xr - xl >= 2):
            continue
        block = clean[yt:yb + 1, xl:xr + 1]
        if not (np.all(block[0] == color) and np.all(block[-1] == color) and
                np.all(block[:, 0] == color) and np.all(block[:, -1] == color)):
            continue
        interior = block[1:-1, 1:-1]
        if not np.all((interior == 0) | (interior == color)):
            continue
        front = xr + 1
        route = int(clean[yt, front])
        if route not in road_colors or not np.all(clean[yt:yb + 1, front] == route):
            continue
        candidates = [x for x in range(front + 1, r)
                      if np.all(clean[yt:yb + 1, x] == route)]
        if len(candidates) != 1:
            continue
        goal = candidates[0]
        expected = np.zeros((b - t + 1, r - l + 1), dtype=np.int8)
        expected[yt - t:yb - t + 1, xl - l:xr - l + 1] = block
        expected[yt - t:yb - t + 1, front - l] = route
        expected[yt - t:yb - t + 1, goal - l] = route
        if not np.array_equal(expected, clean[t:b + 1, l:r + 1]):
            continue
        scan = np.rot90(interior, 1).ravel()
        advanced = int(np.count_nonzero(scan == 0))
        if not np.array_equal(scan, np.array([0] * advanced + [color] * (scan.size - advanced))):
            continue
        cars.append(dict(top=yt, bottom=yb, left=xl, right=xr, front=front,
                         goal=goal, route=route, frame=color, advanced=advanced,
                         capacity=int(scan.size), distance=goal - front,
                         ih=interior.shape[0], iw=interior.shape[1]))
    _need(len(cars) == 1, "expected one framed vehicle with a rightward bar pair")
    car = cars[0]
    field, model = _fit_terrain(clean, bb, marker, car['frame'])
    return car, field, model


def _analyze(value):
    original = _grid(value)
    marker, original_bb = _find_marker(original)
    found = []
    for turns in range(4):
        a = np.rot90(original, turns).copy()
        bb = _bounds(a == marker)
        try:
            car, field, model = _parse_canonical(a, marker, bb)
            found.append(dict(grid=a, bounds=bb, car=car, field=field,
                              model=model, turns=turns, marker=marker,
                              original_bounds=original_bb))
        except AssertionError:
            continue
    _need(len(found) == 1, "input must establish one complete directed explanation")
    return found[0]


def _tolls(field, car, bb):
    t, b, l, r = bb
    rows = field[car['top'] - t:car['bottom'] - t + 1]
    return np.count_nonzero(rows[:, car['front'] + 1 - l:car['goal'] + 1 - l] == car['route'], axis=0)


def _bulk_trip(costs, remaining):
    if remaining == 0:
        return 0, 0
    sums = np.cumsum(costs)
    too_much = np.flatnonzero(sums > remaining)
    exhausted = np.flatnonzero(sums == remaining)
    move = len(costs)
    if len(too_much):
        move = min(move, int(too_much[0]))
    if len(exhausted):
        move = min(move, int(exhausted[0]) + 1)
    spent = int(sums[move - 1]) if move else 0
    return move, spent


def _predicted_output(info, move, spent):
    car, bb = info['car'], info['bounds']
    t, b, l, r = bb
    answer = info['field'].copy()
    yt, yb = car['top'] - t, car['bottom'] - t
    answer[yt:yb + 1, car['goal'] - l] = car['route']
    xl = car['left'] + move - l
    xr = car['right'] + move - l
    answer[yt:yb + 1, xl:xr + 1] = car['frame']
    # In the travel-up view, zero is the spent prefix and frame color is fuel.
    canonical = np.full((car['iw'], car['ih']), car['frame'], dtype=np.int8)
    canonical.flat[:car['advanced'] + spent] = 0
    answer[yt + 1:yb, xl + 1:xr] = np.rot90(canonical, -1)
    answer[yt:yb + 1, car['front'] + move - l] = car['route']
    return np.rot90(answer, -info['turns']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every convention, terrain pixel, toll and counter state from input."""
    info = _analyze(input_grid)
    car = info['car']
    move, spent = _bulk_trip(_tolls(info['field'], car, info['bounds']),
                             car['capacity'] - car['advanced'])
    return _predicted_output(info, move, spent)


def _initial_block(ih, iw, frame, remaining):
    a = np.full((ih + 2, iw + 2), frame, dtype=np.int8)
    used = ih * iw - remaining
    for x in range(iw, 0, -1):
        for y in range(1, ih + 1):
            if used:
                a[y, x] = 0
                used -= 1
    return a


def _unit_trip(terrain, car, block):
    """Ground-truth simulation: pay one literal counter cell per terrain pixel."""
    block = block.copy()
    pending = [(y, x) for x in range(block.shape[1] - 2, 0, -1)
               for y in range(1, block.shape[0] - 1) if block[y, x] == car['frame']]
    moves, paid = 0, 0
    while moves < car['distance'] and pending:
        x = car['front'] + moves + 1
        cost = sum(int(terrain[y, x]) == car['route']
                   for y in range(car['top'], car['bottom'] + 1))
        if cost > len(pending):
            break
        for _ in range(cost):
            y, col = pending.pop(0)
            block[y, col] = 0
            paid += 1
        moves += 1
    return moves, paid, block


def _paint_latent(shape, model, indices):
    terrain = np.zeros(shape, dtype=np.int8)
    for i in indices:
        h = model['height'] + i * model['dh']
        y = model['top'] + i * model['dy']
        x = model['start'] + i * model['period']
        patch = _stroke(model['style'], h, model['width'], model['spacing'],
                        model['palette'][i % len(model['palette'])])
        for py, px in np.argwhere(patch != 0):
            yy, xx = y + int(py), x + int(px)
            if 0 <= yy < shape[0] and 0 <= xx < shape[1]:
                terrain[yy, xx] = patch[py, px]
    return terrain


def _assemble(model, bb, shape, car, remaining, marker, turns):
    needed = list(_indices(model['start'], model['period'], model['width'], bb[2], bb[3]))
    all_indices = range(min([0] + needed), max([model['visible'] - 1] + needed) + 1)
    terrain = _paint_latent(shape, model, all_indices)
    initial = _paint_latent(shape, model, range(model['visible']))
    t, b, l, r = bb
    initial[t:b + 1, l:r + 1] = 0
    block = _initial_block(car['ih'], car['iw'], car['frame'], remaining)
    initial[car['top']:car['bottom'] + 1, car['left']:car['right'] + 1] = block
    initial[car['top']:car['bottom'] + 1, car['front']] = car['route']
    initial[car['top']:car['bottom'] + 1, car['goal']] = car['route']
    move, spent, new_block = _unit_trip(terrain, car, block)
    latent_output = terrain.copy()
    latent_output[car['top']:car['bottom'] + 1, car['goal']] = car['route']
    latent_output[car['top']:car['bottom'] + 1, car['left'] + move:car['right'] + move + 1] = new_block
    latent_output[car['top']:car['bottom'] + 1, car['front'] + move] = car['route']
    answer = latent_output[t:b + 1, l:r + 1].copy()
    _put_markers(initial, bb, marker)
    initial = np.rot90(initial, turns).copy()
    # Rotating geometry must not rotate the canonical annotation convention.
    transformed_bb = _bounds(initial == marker)
    initial[initial == marker] = 0
    _put_markers(initial, transformed_bb, marker)
    answer = np.rot90(answer, turns).copy()
    return dict(input=initial, output=answer), move, spent


def _sample_pair(rng, roles, lesson, turns, test=False, allow_unaffordable=True):
    marker, frame, pool = roles
    k = int(rng.choice([2, 3], p=[.78, .22]))
    palette = tuple(map(int, rng.choice(pool, k, replace=False)))
    n = 2 * k if k == 3 else int(rng.integers(4, 7))
    style = str(rng.choice(['filled', 'hollow', 'comb', 'ladder']))
    width = int(rng.integers(1, 4)) if style == 'filled' else (3 if style in ('hollow', 'ladder') else int(rng.integers(2, 4)))
    spacing = int(rng.integers(2, 4))
    period = width + int(rng.integers(1, 3))
    ih = int(rng.integers(2, 5)) if lesson else int(rng.integers(1, 6))
    iw = int(rng.integers(2, 5)) if lesson else int(rng.integers(1, 5))
    distance = int(rng.integers(7, 11)) if test else int(rng.integers(4, 7))
    window_width = iw + 2 + distance + 3
    backward = bool(rng.integers(2))
    skip = int(rng.choice([0, 1], p=[.8, .2]))
    offset = int(rng.integers(1, period + 1)) + skip * period
    if backward:
        left, start = 0, window_width + offset
    else:
        start, left = 0, (n - 1) * period + width + offset
    right = left + window_width - 1
    canvas_width = max(right + 1, start + (n - 1) * period + width)
    if canvas_width > 30:
        return None
    domain = list(_indices(start, period, width, left, right)) + list(range(n))
    lo, hi = min(domain), max(domain)
    dh = int(rng.choice([-1, 1])) if lesson else int(rng.choice([-1, 0, 1, 2], p=[.25, .25, .40, .10]))
    hmin = int(rng.integers(5 if style in ('comb', 'ladder') else 3, 8))
    h0 = hmin - (lo if dh > 0 else hi) * dh
    dy = int(rng.choice([0, -dh, -1, 1]))
    ys = [i * dy for i in range(lo, hi + 1)]
    heights = [h0 + i * dh for i in range(lo, hi + 1)]
    if min(heights) < 3 or max(heights) > 30:
        return None
    ymin = min(ys)
    ymax = max(y + h for y, h in zip(ys, heights))
    top = -ymin + int(rng.integers(0, 3))
    canvas_height = max(ymax + top, ih + 4) + int(rng.integers(0, 3))
    if canvas_height > 30:
        return None
    wh = int(rng.integers(ih + 4, canvas_height + 1))
    wt = int(rng.integers(0, canvas_height - wh + 1))
    bb = (wt, wt + wh - 1, left, right)
    car_top = int(rng.integers(wt + 1, wt + wh - ih - 2))
    car_left = left + 1
    car = dict(top=car_top, bottom=car_top + ih + 1, left=car_left,
               right=car_left + iw + 1, front=car_left + iw + 2,
               goal=car_left + iw + 2 + distance, distance=distance,
               ih=ih, iw=iw, capacity=ih * iw, frame=frame,
               route=int(rng.choice(palette)))
    model = dict(start=start, period=period, top=top, dy=dy, height=h0, dh=dh,
                 width=width, palette=palette, style=style, spacing=spacing, visible=n)
    shape = (canvas_height, canvas_width)
    terrain = _paint_latent(shape, model, range(lo, hi + 1))
    costs = np.array([sum(terrain[y, car['front'] + j] == car['route']
                          for y in range(car['top'], car['bottom'] + 1))
                      for j in range(1, distance + 1)], dtype=int)
    total, capacity = int(costs.sum()), ih * iw
    if lesson == 'destination':
        if total < 1 or total + 2 > capacity:
            return None
        remaining = int(rng.integers(total + 1, capacity))
    elif lesson in ('exhausted', 'unaffordable'):
        if total < 2 or capacity < 3:
            return None
        choices = []
        for fuel in range(1, min(total, capacity - 1) + 1):
            m, p = _bulk_trip(costs, fuel)
            if m == 0 or p == 0 or m == distance:
                continue
            if lesson == 'exhausted' and fuel == p and costs[m] == 0:
                choices.append(fuel)
            if lesson == 'unaffordable' and fuel > p:
                choices.append(fuel)
        if not choices:
            return None
        remaining = int(rng.choice(choices))
    else:
        mode = int(rng.integers(5))
        remaining = (0 if mode == 0 else capacity if mode == 1 else
                     total if mode == 2 and total <= capacity else int(rng.integers(1, capacity + 1)))
    pair, move, spent = _assemble(model, bb, shape, car, remaining, marker, turns)
    if not allow_unaffordable and move < distance and remaining > spent:
        return None
    if lesson:
        if move == 0 or spent == 0 or not np.any(costs == 0):
            return None
        if not np.any(costs[:move] >= 2):
            return None
        if lesson == 'destination' and not (move == distance and remaining > spent):
            return None
        if lesson in ('exhausted', 'unaffordable') and not (move < distance):
            return None
        # Demonstrate why this is not B's ordinary one-unit-per-cell journey.
        if (move, spent) == (min(remaining, distance), min(remaining, distance)):
            return None
        # A's changing object geometry must affect motion/payment, not merely scenery.
        frozen = dict(model, height=h0 + (n - 1) * dh, top=top + (n - 1) * dy, dh=0, dy=0)
        fixed = _paint_latent(shape, frozen, range(lo, hi + 1))
        fm, fp, _ = _unit_trip(fixed, car, _initial_block(ih, iw, frame, remaining))
        if (fm, fp) == (move, spent):
            return None
        if lesson == 'destination':
            all_colors = terrain.copy()
            all_colors[all_colors != 0] = car['route']
            am, ap, _ = _unit_trip(all_colors, car, _initial_block(ih, iw, frame, remaining))
            if (am, ap) == (move, spent):
                return None
    try:
        info = _analyze(pair['input'])
        predicted = solve(pair['input'])
        if not np.array_equal(predicted, pair['output']):
            return None
        if info['model']['dh'] != dh:
            return None
    except AssertionError:
        return None
    return pair


def _fallback(roles, lesson, turns, test, variant):
    """32 distinct constructive witnesses; no stored grids or output lookups."""
    marker, frame, pool = roles
    palette = tuple(pool[:2])
    distance = 9 if test else 6
    bb = (0, 17 + variant % 11, 9, 9 + 5 + distance + 2)
    model = dict(start=0, period=2, top=0, dy=0, height=5, dh=1,
                 width=1, palette=palette, style='filled', spacing=2, visible=4)
    ct = 8 + variant // 11
    car = dict(top=ct, bottom=ct + 4, left=10, right=14, front=15,
               goal=15 + distance, distance=distance, ih=3, iw=3,
               capacity=9, frame=frame, route=palette[1])
    first_toll = min(5, 14 - ct)
    if lesson == 'unaffordable':
        car['route'] = palette[0]
        remaining = min(5, 13 - ct) + 1
    else:
        remaining = first_toll + 1 if lesson == 'destination' else first_toll
    pair, _, _ = _assemble(model, bb, (bb[1] + 1, bb[3] + 1), car, remaining, marker, turns)
    return pair


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    colors = list(map(int, rng.permutation(np.arange(1, 10))))
    shared = (colors[0], colors[1], tuple(colors[2:]))
    varying_palette = bool(rng.integers(2))
    orientation = int(rng.integers(4))
    other = (orientation + int(rng.choice([1, 3]))) % 4
    order = list(map(int, rng.permutation(num_train)))
    lessons = {order[0]: ('destination', orientation), order[1]: ('exhausted', other)}
    if num_train >= 3:
        lessons[order[2]] = ('unaffordable', int(rng.integers(4)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            lesson, turns = lessons.get(index, (None, int(rng.integers(4)))) if split == 'train' else (None, int(rng.choice([orientation, other])))
            if varying_palette:
                colors = list(map(int, rng.permutation(np.arange(1, 10))))
                roles = (colors[0], colors[1], tuple(colors[2:]))
            else:
                roles = shared
            pair = None
            for attempt in range(544):
                candidate = (_sample_pair(rng, roles, lesson, turns, split == 'test', num_train >= 3) if attempt < 512 else
                             _fallback(roles, lesson, turns, split == 'test', attempt - 512))
                if candidate is None:
                    continue
                key = _key(candidate['input'])
                if key not in seen:
                    pair = candidate
                    seen.add(key)
                    break
            _need(pair is not None, 'bounded construction exhausted')
            puzzle[split].append(pair)
    return puzzle


def _features(pair):
    info = _analyze(pair['input'])
    car, model = info['car'], info['model']
    remaining = car['capacity'] - car['advanced']
    costs = _tolls(info['field'], car, info['bounds'])
    move, spent = _bulk_trip(costs, remaining)
    residual = remaining - spent
    stop = ('destination' if move == car['distance'] and residual > 0 else
            'exact' if move == car['distance'] else 'zero' if remaining == 0 else
            'unaffordable' if residual > 0 else 'exhausted')
    return dict(info=info, move=move, spent=spent, remaining=remaining,
                residual=residual, costs=costs, stop=stop,
                backward=model['start'] > info['bounds'][3])


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _need(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4 and
          len(puzzle['train']) > len(puzzle['test']), 'invalid pair counts')
    seen, train_features, test_features = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            a, out = _grid(pair['input']), _grid(pair['output'])
            _need(_key(a) not in seen, 'duplicate input')
            seen.add(_key(a))
            f = _features(pair)
            info, car = f['info'], f['info']['car']
            t, b, l, r = info['original_bounds']
            _need(out.shape == (b - t + 1, r - l + 1), 'output must be the inclusive crop')
            _need(not np.any(out == info['marker']), 'L annotations leaked into answer')
            _need(np.array_equal(solve(a), out), 'incorrect output')
            # Rotated unit-step simulation validates the bulk cumulative-toll engine.
            block = info['grid'][car['top']:car['bottom'] + 1, car['left']:car['right'] + 1]
            terrain = np.zeros(info['grid'].shape, dtype=np.int8)
            ct, cb, cl, cr = info['bounds']
            terrain[ct:cb + 1, cl:cr + 1] = info['field']
            m, p, updated = _unit_trip(terrain, car, block)
            _need((m, p) == (f['move'], f['spent']), 'unit and cumulative toll calculations disagree')
            _need(np.count_nonzero(updated[1:-1, 1:-1] == car['frame']) == f['residual'],
                  'odometer did not consume exactly the toll')
            if f['move'] < car['distance'] and f['residual'] > 0:
                _need(int(f['costs'][f['move']]) > f['residual'], 'unjustified early stop')
            if f['remaining'] == 0:
                _need(f['move'] == 0, 'empty counter moved')
            if f['move'] and f['residual'] == 0:
                _need(int(np.cumsum(f['costs'])[f['move'] - 1]) == f['remaining'], 'wrong exhaustion boundary')
            (train_features if split == 'train' else test_features).append(f)
    _need(any(f['stop'] == 'destination' and f['spent'] > 0 for f in train_features),
          'training must show paid travel with surplus fuel at the destination')
    _need(any(f['stop'] in ('exhausted', 'unaffordable') and f['spent'] > 0 for f in train_features),
          'training must show a paid fuel-limited journey')
    _need(any(f['stop'] == 'exhausted' and f['spent'] > 0 and
              f['costs'][f['move']] == 0 for f in train_features),
          'training must show exhaustion stopping before a free step')
    if len(train_features) >= 3:
        _need(any(f['stop'] == 'unaffordable' and f['spent'] > 0 for f in train_features),
              'training must show an atomic unpaid toll with fuel left')
    else:
        _need(not any(f['stop'] == 'unaffordable' for f in test_features),
              'two-example tests must use demonstrated stopping cases')
    _need(len({f['info']['turns'] for f in train_features if f['spent'] > 0}) >= 2,
          'training must expose rotation of the counter')
    informative = [f for f in train_features if f['spent'] > 0]
    _need(any((f['move'], f['spent']) != (min(f['remaining'], f['info']['car']['distance']),) * 2
              for f in informative), 'ordinary B motion is not distinguished')
    growth_witness = False
    selector_witness = False
    for f in informative:
        info, model, car = f['info'], f['info']['model'], f['info']['car']
        n = model['visible'] - 1
        frozen = dict(model, height=model['height'] + n * model['dh'],
                      top=model['top'] + n * model['dy'], dh=0, dy=0)
        fixed = _field(frozen, info['bounds'])
        altered = _bulk_trip(_tolls(fixed, car, info['bounds']), f['remaining'])
        growth_witness |= altered != (f['move'], f['spent'])
        all_colors = info['field'].copy()
        all_colors[all_colors != 0] = car['route']
        selector_witness |= _bulk_trip(_tolls(all_colors, car, info['bounds']), f['remaining']) != (f['move'], f['spent'])
    _need(growth_witness, 'changing curtain geometry must affect a demonstrated trip')
    _need(selector_witness, 'training must distinguish color selection from charging every color')
    _need(any(np.any(f['costs'][:f['move']] >= 2) for f in informative),
          'training must show more than one fuel cell paid at a single step')
    _need(min(f['info']['car']['distance'] for f in test_features) >
          max(f['info']['car']['distance'] for f in train_features), 'missing longer-journey test shift')


def self_test():
    """Bounded semantic regressions and input-derived variation over 42 seeds."""
    # Atomic tolls, free motion, exact exhaustion, unaffordable residual, empty fuel.
    for costs, fuel, expected in [([0, 3, 0, 2], 4, (3, 3)),
                                  ([0, 3, 0, 2], 3, (2, 3)),
                                  ([0, 3, 0, 2], 6, (4, 5)),
                                  ([0, 0, 2], 0, (0, 0)),
                                  ([2, 0], 1, (0, 0)),
                                  ([0, 0], 1, (2, 0))]:
        _need(_bulk_trip(np.array(costs), fuel) == expected, 'toll boundary regression')
    puzzles, features = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        features.extend(_features(pair) for split in p.values() for pair in split)
    _need(len({tuple(_key(pair['input']) for split in p.values() for pair in split) for p in puzzles}) == 42,
          'seed diversity collapsed')
    _need(len({pair['input'].shape for p in puzzles for split in p.values() for pair in split}) >= 40,
          'canvas geometry diversity collapsed')
    _need({f['info']['model']['style'] for f in features} == {'filled', 'hollow', 'comb', 'ladder'},
          'curtain shape diversity collapsed')
    _need({f['info']['turns'] for f in features} == {0, 1, 2, 3}, 'travel directions collapsed')
    _need({f['backward'] for f in features} == {False, True}, 'extrapolation direction collapsed')
    _need({len(f['info']['model']['palette']) for f in features} == {2, 3}, 'cycle lengths collapsed')
    _need({np.sign(f['info']['model']['dh']) for f in features} == {-1, 0, 1}, 'height progressions collapsed')
    _need(len({(f['info']['model']['dh'], f['info']['model']['dy']) for f in features}) >= 8,
          'growth/placement interactions collapsed')
    _need(len({f['info']['car']['capacity'] for f in features}) >= 9, 'counter shapes collapsed')
    _need(any(f['remaining'] == f['info']['car']['capacity'] for f in features) and
          any(f['remaining'] == 0 for f in features), 'uniform counter states missing')
    _need(any(f['stop'] == 'unaffordable' for f in features), 'positive-residual stop missing')
    _need(any(f['move'] > f['spent'] > 0 for f in features) and
          any(f['spent'] > f['move'] > 0 for f in features), 'weighted/free motion range missing')
    _need(any(f['info']['car']['advanced'] // f['info']['car']['ih'] <
              (f['info']['car']['advanced'] + f['spent']) // f['info']['car']['ih'] and
              (f['info']['car']['advanced'] + f['spent']) % f['info']['car']['ih']
              for f in features), 'counter wrapping with a partial final row missing')
    repeat = generate_puzzle(17)
    _need(all(_key(x[k]) == _key(y[k]) for s in repeat for x, y in zip(repeat[s], puzzles[17][s])
              for k in ('input', 'output')), 'whole-puzzle nondeterminism')
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(73, nt, ne)
            _need(len(p['train']) == nt and len(p['test']) == ne, 'counts ignored')
            validate_generated_puzzle(p)
    for nt, ne in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)]:
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Rotation consistency and palette-permutation equivariance, including markers.
    p = puzzles[0]
    sample = p['test'][0]
    inp, output = sample['input'], sample['output']
    marker, _ = _find_marker(inp)
    for turns in range(4):
        rotated = np.rot90(inp, turns).copy()
        bb = _bounds(rotated == marker)
        rotated[rotated == marker] = 0
        _put_markers(rotated, bb, marker)
        _need(np.array_equal(solve(rotated), np.rot90(output, turns)), 'rotation consistency failed')
    mapping = np.array([0, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=np.int8)
    _need(np.array_equal(solve(mapping[inp]), mapping[output]), 'palette roles depend on literal color')
    roles = (9, 8, (1, 2, 3, 4, 5, 6, 7))
    for lesson in ('destination', 'exhausted', 'unaffordable'):
        keys = set()
        for variant in range(32):
            pair = _fallback(roles, lesson, variant % 4, False, variant)
            _need(np.array_equal(solve(pair['input']), pair['output']), 'constructive fallback disagreement')
            _need(_features(pair)['stop'] == lesson, 'constructive fallback changed its lesson')
            keys.add(_key(pair['input']))
        _need(len(keys) == 32, 'fallback pool cannot guarantee unique inputs')
    broken = {s: [{k: np.array(g, copy=True) for k, g in pair.items()} for pair in p[s]] for s in p}
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a corrupted label')
