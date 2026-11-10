"""A diagonal aperture activates colors in corresponding, deformed rooms.

Only NumPy and the standard library are used. The solver reads visible walls,
room arrangements, uniform source colors, and finite diagonal segments.
"""
from collections import Counter
from itertools import combinations
import numpy as np


HOST_SEEDS = (0, 294651257, 868713709, 993775783, 256090447, 949449932,
              769369486, 719136606, 536543498, 554903736, 397954027,
              168402798, 309984070, 253409420, 111359120, 280995811,
              901316973, 463119470, 486363884, 758197137, 628072233,
              4874456, 789397860, 557090794, 329044516, 487134033,
              158780572, 301012371, 51262502, 565633960, 609955505,
              462499152, 470249820, 434298269, 931395499, 192557183,
              739116796, 617475052, 855652234, 604363172, 114316611,
              769700767)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _parts(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    if diagonal:
        steps += ((1, 1), (1, -1), (-1, 1), (-1, -1))
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    cells.append(p)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _box(points):
    lo, hi = points.min(axis=0), points.max(axis=0) + 1
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _enclosed(walls):
    # Four-connected nonwalls are interiors unless connected to padded exterior.
    padded = np.pad(~walls, 1, constant_values=True)
    limit = np.array(padded.shape) - 1
    return [p - 1 for p in _parts(padded)
            if np.all(p > 0) and np.all(p < limit)]


def _relations(rooms):
    boxes = [_box(p) for p in rooms]
    return tuple(tuple(tuple(-1 if a[k + 2] <= b[k] else
                             1 if b[k + 2] <= a[k] else 0
                             for k in (0, 1)) for b in boxes) for a in boxes)


def _mappings(source, target):
    """Exact isomorphisms of signed above/below and left/right projections.

    Area, discovery order, color, and sampling labels do not enter matching.
    """
    a, b = _relations(source), _relations(target)
    n = len(a)
    assert n == len(b)
    sa = [Counter(row) for row in a]
    sb = [Counter(row) for row in b]
    domains = [[j for j in range(n) if sa[i] == sb[j]] for i in range(n)]
    result, visits = [], 0

    def search(mapping, used):
        nonlocal visits
        visits += 1
        assert visits < 10000, "room arrangement is insufficiently distinguished"
        if len(mapping) == n:
            result.append(dict(mapping))
            return
        choices = {i: [j for j in domains[i] if j not in used and
                       all(a[i][k] == b[j][v] for k, v in mapping.items())]
                   for i in range(n) if i not in mapping}
        i = min(choices, key=lambda k: (len(choices[k]), k))
        for j in choices[i]:
            mapping[i] = j
            search(mapping, used | {j})
            del mapping[i]

    search({}, set())
    assert result, "frameworks do not have corresponding room arrangements"
    return result


def _read_strokes(points):
    candidates = []
    for slope in (-1, 1):
        lines = {}
        for r, c in points:
            lines.setdefault(int(c - slope * r), []).append(int(r))
        strokes = []
        valid = True
        for q, rows in sorted(lines.items()):
            rows.sort()
            groups = []
            for r in rows:
                if not groups or r != groups[-1][-1] + 1:
                    groups.append([])
                groups[-1].append(r)
            for run in groups:
                if len(run) < 2:
                    valid = False
                    break
                strokes.append((q, 2 * run[0] + slope * q,
                                2 * run[-1] + slope * q))
        if valid and len(strokes) >= 2:
            candidates.append((slope, sorted(strokes)))
    assert len(candidates) == 1, "diagonal segments need one visible orientation"
    return candidates[0]


def _parse(inp):
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] * 2 > inp.size, "background must occupy a strict majority"
    candidates = []
    for wall in sorted(set(map(int, inp.flat)) - {bg}):
        components = _parts(inp == wall, diagonal=True)
        if len(components) != 2:
            continue
        figures = []
        for cells in components:
            r, c, R, C = _box(cells)
            walls = np.zeros((R - r, C - c), dtype=bool)
            walls[cells[:, 0] - r, cells[:, 1] - c] = True
            rooms = _enclosed(walls)
            if not 3 <= len(rooms) <= 10:
                break
            crop = np.full(walls.shape, bg, dtype=np.int8)
            crop[walls] = wall
            values = []
            valid = True
            for p in rooms:
                colors = np.unique(inp[p[:, 0] + r, p[:, 1] + c])
                if len(colors) != 1 or colors[0] == wall:
                    valid = False
                    break
                values.append(int(colors[0]))
                crop[tuple(p.T)] = colors[0]
            if not valid:
                break
            figures.append(dict(crop=crop, walls=walls, rooms=rooms,
                                colors=values, origin=(r, c), cells=cells))
        if len(figures) != 2:
            continue
        donors = [f for f in figures if all(v != bg for v in f['colors'])]
        targets = [f for f in figures if all(v == bg for v in f['colors'])]
        if len(donors) != 1 or len(targets) != 1:
            continue
        donor, target = donors[0], targets[0]
        if len(donor['rooms']) != len(target['rooms']):
            continue
        used = np.zeros(inp.shape, dtype=bool)
        for f in figures:
            used[tuple(f['cells'].T)] = True
            for p in f['rooms']:
                used[p[:, 0] + f['origin'][0], p[:, 1] + f['origin'][1]] = True
        # All remaining foreground is one distinct shutter role.
        remaining = np.argwhere((inp != bg) & ~used)
        if not len(remaining):
            continue
        markers = np.unique(inp[tuple(remaining.T)])
        if len(markers) != 1 or markers[0] in {bg, wall, *donor['colors']}:
            continue
        try:
            slope, strokes = _read_strokes(remaining)
            mappings = _mappings(donor['rooms'], target['rooms'])
        except AssertionError:
            continue
        candidates.append(dict(bg=bg, wall=wall, marker=int(markers[0]),
                               donor=donor, target=target, slope=slope,
                               strokes=strokes, mappings=mappings))
    assert len(candidates) == 1, "input must have one unambiguous role interpretation"
    return candidates[0]


def _band(shape, slope, strokes):
    """Input-only solver: sweep all integer longitudinal coordinates."""
    r, c = np.indices(shape)
    q, t = c - slope * r, r + slope * c
    band = np.zeros(shape, dtype=bool)
    for level in range(int(t.min()), int(t.max()) + 1):
        active = [u for u, lo, hi in strokes if lo <= level <= hi]
        if len(active) >= 2:
            band |= (t == level) & (q > min(active)) & (q < max(active))
    return band


def _pair_band(shape, slope, strokes, alternative=''):
    """Geometric audit: union finite parallelograms, with weaker alternatives."""
    r, c = np.indices(shape)
    q, t = c - slope * r, r + slope * c
    pairs = list(combinations(range(len(strokes)), 2))
    if alternative == 'longitudinal_only':
        count = sum(((t >= lo) & (t <= hi)).astype(int) for _, lo, hi in strokes)
        return count >= 2
    if alternative == 'consecutive':
        # Permanently adjacent strokes within each interval-overlap component.
        remaining, groups = set(range(len(strokes))), []
        while remaining:
            stack = [min(remaining)]
            remaining.remove(stack[0])
            group = []
            while stack:
                i = stack.pop()
                group.append(i)
                neighbors = [j for j in sorted(remaining)
                             if max(strokes[i][1], strokes[j][1]) <=
                             min(strokes[i][2], strokes[j][2])]
                remaining.difference_update(neighbors)
                stack.extend(neighbors)
            groups.append(sorted(group, key=lambda i: strokes[i][0]))
        pairs = [p for group in groups for p in zip(group, group[1:])]
    band = np.zeros(shape, dtype=bool)
    for i, j in pairs:
        a, b = strokes[i], strokes[j]
        mask = (q > min(a[0], b[0])) & (q < max(a[0], b[0]))
        if alternative != 'unbounded':
            mask &= (t >= max(a[1], b[1])) & (t <= min(a[2], b[2]))
        if alternative == 'pixel_centers':
            mask &= ((t - a[0]) % 2 == 0) & ((t - b[0]) % 2 == 0)
        band |= mask
    return band


def _answer(info, band, alternative=''):
    d, target = info['donor'], info['target']
    active = []
    for room in d['rooms']:
        p = room + d['origin']
        flags = band[tuple(p.T)]
        if alternative == 'all_rooms':
            activate = True
        elif alternative == 'majority':
            activate = int(flags.sum()) * 2 > len(flags)
        elif alternative == 'center':
            center = np.rint(p.mean(axis=0)).astype(int)
            activate = bool(band[tuple(center)])
        else:
            activate = bool(flags.any())
        active.append(activate)
    answers = []
    for mapping in info['mappings']:
        out = target['crop'].copy()
        for i, j in mapping.items():
            if active[i]:
                out[tuple(target['rooms'][j].T)] = d['colors'][i]
        answers.append(out)
    assert all(np.array_equal(answers[0], a) for a in answers[1:]), "ambiguous transfer"
    return answers[0], active


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the isolated receiver, using no training, seed or stored state."""
    inp = _grid(input_grid)
    info = _parse(inp)
    return _answer(info, _band(inp.shape, info['slope'], info['strokes']))[0]


def _plan(rng, count):
    options = [(r, c) for r, c in ((2, 2), (2, 3), (3, 2), (3, 3),
                                  (2, 4), (4, 2), (3, 4), (4, 3)) if r * c >= count]
    rows, cols = options[int(rng.integers(len(options)))]
    plan = np.arange(rows * cols).reshape(rows, cols)
    for corner in rng.permutation(4):
        r, c = ((0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1))[int(corner)]
        if np.count_nonzero(plan >= 0) > count and rng.random() < .35:
            trial = plan.copy()
            trial[r, c] = -1
            if len(_parts(trial >= 0)) == 1:
                plan = trial
    while len(set(plan.flat) - {-1}) > count:
        edges = [(int(plan[r, c]), int(plan[r + dr, c + dc]))
                 for r in range(rows) for c in range(cols) for dr, dc in ((1, 0), (0, 1))
                 if r + dr < rows and c + dc < cols and plan[r, c] >= 0
                 and plan[r + dr, c + dc] >= 0 and plan[r, c] != plan[r + dr, c + dc]]
        a, b = edges[int(rng.integers(len(edges)))]
        plan[plan == b] = a
    names = sorted(set(plan.flat) - {-1})
    return np.array([[names.index(int(v)) if v >= 0 else -1 for v in row]
                     for row in plan], dtype=int)


def _embedding(rng, plan, cap):
    for _ in range(20):
        heights = rng.integers(1, 4, size=plan.shape[0])
        widths = rng.integers(1, 4, size=plan.shape[1])
        if sum(heights + 1) + 1 <= cap and sum(widths + 1) + 1 <= cap:
            break
    else:
        heights = np.ones(plan.shape[0], dtype=int)
        widths = np.ones(plan.shape[1], dtype=int)
    ys = [0, *np.cumsum(heights + 1)]
    xs = [0, *np.cumsum(widths + 1)]
    labels = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    for r, c in np.ndindex(plan.shape):
        name = int(plan[r, c])
        if name < 0:
            continue
        labels[ys[r] + 1:ys[r + 1], xs[c] + 1:xs[c + 1]] = name
        if r and plan[r - 1, c] == name:
            labels[ys[r], xs[c] + 1:xs[c + 1]] = name
        if c and plan[r, c - 1] == name:
            labels[ys[r] + 1:ys[r + 1], xs[c]] = name
        if r and c and np.all(plan[r - 1:r + 1, c - 1:c + 1] == name):
            labels[ys[r], xs[c]] = name
    # Local corner bites preserve room bounding boxes and connectivity.
    for name in sorted(set(plan.flat) - {-1}):
        for _ in range(int(rng.integers(3))):
            p = np.argwhere(labels == name)
            r, c, R, C = _box(p)
            if len(p) < 5 or R - r < 2 or C - c < 2:
                continue
            point = ((r, c), (r, C - 1), (R - 1, c), (R - 1, C - 1))[int(rng.integers(4))]
            if labels[point] != name:
                continue
            trial = labels == name
            trial[point] = False
            parts = _parts(trial)
            if len(parts) == 1 and _box(parts[0]) == (r, c, R, C):
                labels[point] = -1
    occupied = labels >= 0
    padded = np.pad(occupied, 1)
    near = np.zeros(labels.shape, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            near |= padded[dr:dr + len(labels), dc:dc + labels.shape[1]]
    labels[(labels < 0) & near] = -2
    r, c, R, C = _box(np.argwhere(labels != -1))
    return labels[r:R, c:C]


def _paint(labels, bg, wall, colors=None, active=None):
    out = np.full(labels.shape, bg, dtype=np.int8)
    out[labels == -2] = wall
    if colors is not None:
        for i, color in enumerate(colors):
            if active is None or active[i]:
                out[labels == i] = color
    return out


def _stroke_cells(stroke):
    q, lo, hi = stroke
    return np.array([((t - q) // 2, (t + q) // 2)
                     for t in range(lo, hi + 1, 2)], dtype=int)


def _parity_interval(q, lo, hi):
    lo = int(lo) - ((int(lo) - q) % 2)
    hi = int(hi) + ((q - int(hi)) % 2)
    return q, lo, max(hi, lo + 2)


def _shutters(rng, labels, count, witness):
    occupied = np.argwhere(labels != -1)
    qmin, qmax = int((occupied[:, 1] - occupied[:, 0]).min()), int((occupied[:, 1] - occupied[:, 0]).max())
    tmax = int(occupied.sum(axis=1).max())
    if witness == 'windows':
        # Two longitudinally disjoint apertures. Each pair must transmit some
        # room colors that the other pair cannot transmit (checked downstream).
        room_t = np.argwhere(labels >= 0).sum(axis=1)
        early = int(room_t.min()) + int(rng.integers(3))
        late = int(room_t.max()) - int(rng.integers(3))
        assert late - early >= 6
        specs = [_parity_interval(qmin - 3, early - 2, early),
                 _parity_interval(qmax + 3, early - 2, early),
                 _parity_interval(qmin - 7, late, late + 2),
                 _parity_interval(qmax + 7, late, late + 2)]
        if count == 5:
            # An isolated short segment tests that count alone is not exposure.
            options = [q for q in range(qmin, qmax + 1)
                       if all(abs(q - s[0]) >= 3 for s in specs)]
            q = int(rng.choice(options))
            hi = -4 - q % 2
            specs.append((q, hi - 2 * int(rng.integers(1, 4)), hi))
        return specs
    left = qmin - (6 if witness == 'skip' else int(rng.integers(3, 7)))
    right = qmax + int(rng.integers(3, 7))
    if witness:
        # Opposite endpoint phases expose continuous rather than center-only rays.
        right += (right - left + 1) % 2
    room_points = np.argwhere(labels >= 0)
    first = int(room_points.sum(axis=1).min())
    if witness == 'tip':
        high = first
        low = -int(rng.integers(4, 9))
    elif witness == 'skip':
        high = int(rng.integers(first + 1, max(first + 2, tmax)))
        low = -int(rng.integers(7, 11))
    else:
        low = int(rng.integers(-6, max(1, tmax - 3)))
        high = int(rng.integers(max(first, low + 2), tmax + 5))
    specs = [_parity_interval(left, low, high), _parity_interval(right, low, high)]
    if not witness:
        specs[0] = _parity_interval(left, low - int(rng.integers(5)), high + int(rng.integers(5)))
        specs[1] = _parity_interval(right, low - int(rng.integers(5)), high + int(rng.integers(5)))
    forbidden = {(int(r + dr), int(c + dc)) for r, c in occupied
                 for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    if witness == 'skip':
        # A short intermediate boundary sits outside the donor on the left.
        # Early on, the outer strokes bypass it; later, it and the left stroke
        # are active on the same side of the donor, so mere interval counts fail.
        mid = qmin - 3
        middle = _parity_interval(mid, high - 2, high + int(rng.integers(3, 7)))
        specs[0] = _parity_interval(left, low, middle[2] + int(rng.integers(2, 5)))
        specs.append(middle)
    # Extra shutters can run beside the figure or stop before a concave contour.
    for _ in range(200):
        if len(specs) == count:
            break
        q = int(rng.integers(qmin - 9, qmax + 10))
        if any(abs(q - s[0]) < 3 for s in specs):
            continue
        ts = list(range(-10 - ((-10 - q) % 2), tmax + 11, 2))
        runs = []
        for t in ts:
            if ((t - q) // 2, (t + q) // 2) in forbidden:
                continue
            if not runs or t != runs[-1][-1] + 2:
                runs.append([])
            runs[-1].append(t)
        runs = [run for run in runs if len(run) >= 2]
        if not runs:
            continue
        run = runs[int(rng.integers(len(runs)))]
        length = int(rng.integers(2, min(11, len(run)) + 1))
        start = int(rng.integers(len(run) - length + 1))
        specs.append((q, run[start], run[start + length - 1]))
    assert len(specs) == count
    for s in specs:
        assert not any(tuple(p) in forbidden for p in _stroke_cells(s))
    return specs


def _place(rng, donor, target, bg, marker, specs):
    # Coordinates are donor-local, allowing shutters on all four sides.
    a = np.argwhere(donor != bg)
    beams = np.concatenate([_stroke_cells(s) for s in specs])
    points = np.concatenate((a, beams))
    r0, c0, R0, C0 = _box(points)
    if R0 - r0 > 30 or C0 - c0 > 30:
        raise AssertionError('shutters exceed the canvas')
    forbidden = {(int(r + dr), int(c + dc)) for r, c in points
                 for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
    b = np.argwhere(target != bg)
    positions = [(int(rng.integers(r0 - target.shape[0] - 2, R0 + 3)),
                  int(rng.integers(c0 - target.shape[1] - 2, C0 + 3)))
                 for _ in range(140)]
    positions += [(r0, C0 + 1), (R0 + 1, c0),
                  (r0, c0 - target.shape[1] - 1),
                  (r0 - target.shape[0] - 1, c0)]
    for tr, tc in positions:
        shifted = b + (tr, tc)
        p = np.concatenate((points, shifted))
        r, c, R, C = _box(p)
        h, w = R - r, C - c
        if h > 30 or w > 30 or any(tuple(v) in forbidden for v in shifted):
            continue
        # Padding keeps the semantic field a strict majority, without fixed size.
        need = 2 * len(p) + 1
        h += int(rng.integers(min(4, 30 - h) + 1))
        w += int(rng.integers(min(4, 30 - w) + 1))
        while h * w < need:
            if h == w == 30:
                break
            if h < 30 and (h <= w or w == 30):
                h += 1
            else:
                w += 1
        if h * w < need:
            continue
        top = int(rng.integers(h - (R - r) + 1)) - r
        left = int(rng.integers(w - (C - c) + 1)) - c
        inp = np.full((h, w), bg, dtype=np.int8)
        inp[a[:, 0] + top, a[:, 1] + left] = donor[tuple(a.T)]
        inp[beams[:, 0] + top, beams[:, 1] + left] = marker
        inp[shifted[:, 0] + top, shifted[:, 1] + left] = target[tuple(b.T)]
        return inp, (top, left)
    raise AssertionError('no separated receiver placement fits')


def _transform(grid, rotation, reflection):
    out = np.rot90(grid, rotation)
    if reflection:
        out = out[:, ::-1]
    return out.copy()


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _alternatives(inp, info, correct):
    actual = _pair_band(inp.shape, info['slope'], info['strokes'])
    flags = {}
    for name in ('all_rooms', 'center', 'majority', 'consecutive', 'pixel_centers', 'unbounded', 'longitudinal_only'):
        band = _pair_band(inp.shape, info['slope'], info['strokes'], name) if name in (
            'consecutive', 'pixel_centers', 'unbounded', 'longitudinal_only') else actual
        other = _answer(info, band, name)[0]
        flags[name] = not np.array_equal(correct, other)
    d, target = info['donor'], info['target']
    # Ignore geometric correspondence and match rooms by ascending area instead.
    order_d = sorted(range(len(d['rooms'])), key=lambda i: (len(d['rooms'][i]), _box(d['rooms'][i])))
    order_t = sorted(range(len(target['rooms'])), key=lambda i: (len(target['rooms'][i]), _box(target['rooms'][i])))
    _, active = _answer(info, actual)
    area_answer = target['crop'].copy()
    for i, j in zip(order_d, order_t):
        if active[i]:
            area_answer[tuple(target['rooms'][j].T)] = d['colors'][i]
    flags['area_match'] = not np.array_equal(correct, area_answer)
    # Apply the aperture to the receiver instead of the colored source.
    receiver_answer = target['crop'].copy()
    for i, j in info['mappings'][0].items():
        p = target['rooms'][j] + target['origin']
        if actual[tuple(p.T)].any():
            receiver_answer[tuple(target['rooms'][j].T)] = d['colors'][i]
    flags['receiver_gate'] = not np.array_equal(correct, receiver_answer)
    return flags


def _make_pair(rng, rooms, strokes, witness, seen, excluded_roles=None, excluded_structure=None):
    for _ in range(180):
        try:
            plan = _plan(rng, rooms)
            dlabels = _embedding(rng, plan, 11 if witness == 'windows' else 13)
            tlabels = _embedding(rng, plan, 13 if witness == 'windows' else 15)
            palette = list(map(int, rng.permutation(10)))
            bg, wall, marker = palette[:3]
            if excluded_roles is not None and (bg == excluded_roles[0] or wall == excluded_roles[1]):
                continue
            richness = int(rng.integers(2, min(rooms, 7) + 1))
            colors = palette[3:3 + richness]
            colors += [int(rng.choice(colors)) for _ in range(rooms - richness)]
            rng.shuffle(colors)
            donor = _paint(dlabels, bg, wall, colors)
            target = _paint(tlabels, bg, wall)
            specs = _shutters(rng, dlabels, strokes, witness)
            inp, origin = _place(rng, donor, target, bg, marker, specs)
            # Labels use the construction's room identities, separately from matching.
            local_band = _pair_band(inp.shape, 1,
                                    [(q + origin[1] - origin[0],
                                      lo + sum(origin), hi + sum(origin)) for q, lo, hi in specs])
            active = [bool(local_band[(np.argwhere(dlabels == i) + origin).T[0],
                                      (np.argwhere(dlabels == i) + origin).T[1]].any())
                      for i in range(rooms)]
            expected = _paint(tlabels, bg, wall, colors, active)
            rotation, reflection = int(rng.integers(4)), bool(rng.integers(2))
            inp = _transform(inp, rotation, reflection)
            expected = _transform(expected, rotation, reflection)
            if _key(inp) in seen:
                continue
            info = _parse(inp)
            if excluded_structure == (info['target']['crop'].shape, info['target']['walls'].tobytes()):
                continue
            answer, visible_active = _answer(info, _band(inp.shape, info['slope'], info['strokes']))
            if not np.array_equal(answer, expected) or not any(visible_active):
                continue
            if np.array_equal(donor == wall, target == wall):
                continue
            flags = _alternatives(inp, info, expected)
            if witness == 'tip' and not all(flags[n] for n in ('all_rooms', 'center', 'majority', 'pixel_centers', 'unbounded', 'receiver_gate')):
                continue
            nonrectangular = any(len(p) != (_box(p)[2] - _box(p)[0]) * (_box(p)[3] - _box(p)[1])
                                 for p in info['donor']['rooms'] + info['target']['rooms'])
            if witness == 'skip' and not (flags['consecutive'] and flags['pixel_centers'] and flags['area_match'] and flags['longitudinal_only'] and nonrectangular):
                continue
            if witness == 'skip' and not 2 <= sum(visible_active) < rooms:
                continue
            if witness == 'windows':
                if sum(visible_active) == rooms:
                    continue
                effects = []
                for k in range(len(info['strokes'])):
                    subset = info['strokes'][:k] + info['strokes'][k + 1:]
                    other = _answer(info, _pair_band(inp.shape, info['slope'], subset))[0]
                    effects.append(not np.array_equal(other, expected))
                if sum(effects) < 4:
                    continue
            return dict(input=inp, output=expected)
        except AssertionError:
            continue
    raise RuntimeError('bounded aperture construction exhausted')


def _counts(a, b):
    for name, value, low, high in (('num_train', a, 2, 10), ('num_test', b, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if a <= b:
        raise ValueError('num_train must be greater than num_test')
    return int(a), int(b)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    # Only sampling curricula change between puzzles; the input-only rule is fixed.
    room_shift = bool(rng.integers(2))
    puzzle, seen = dict(train=[], test=[]), set()
    previous_roles, previous_structure, first_rooms = None, None, None
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            if split == 'train':
                rooms = int(rng.integers(3, 7 if room_shift else 10))
                if index == 1:
                    choices = [n for n in range(5, 7 if room_shift else 10) if n != first_rooms]
                    rooms = int(rng.choice(choices))
                strokes = (2, 3)[index] if index < 2 else int(rng.integers(2, 4))
                witness = ('tip', 'skip')[index] if index < 2 else ''
            else:
                rooms = int(rng.integers(7, 11)) if room_shift else int(rng.integers(3, 10))
                strokes = int(rng.integers(2, 5)) if room_shift else int(rng.integers(4, 6))
                witness = '' if room_shift else 'windows'
            pair = _make_pair(rng, rooms, strokes, witness, seen,
                              previous_roles if split == 'train' and index == 1 else None,
                              previous_structure if split == 'train' and index == 1 else None)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train' and index == 0:
                info = _parse(pair['input'])
                previous_roles = info['bg'], info['wall']
                previous_structure = info['target']['crop'].shape, info['target']['walls'].tobytes()
                first_rooms = rooms
    return puzzle


def measure_pair(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    info = _parse(inp)
    band = _pair_band(inp.shape, info['slope'], info['strokes'])
    expected, active = _answer(info, band)
    assert np.array_equal(out, expected), 'wrong room activation, transfer or crop'
    assert np.array_equal(band, _band(inp.shape, info['slope'], info['strokes']))
    assert np.array_equal(out == info['wall'], info['target']['walls']), 'receiver walls changed'
    room_mask = np.zeros(out.shape, dtype=bool)
    for p in info['target']['rooms']:
        room_mask[tuple(p.T)] = True
    assert np.array_equal(out[~room_mask], info['target']['crop'][~room_mask])
    donor, target = info['donor'], info['target']
    owned = []
    for figure in (donor, target):
        owned.append(set(map(tuple, figure['cells'])) |
                     {tuple(p) for room in figure['rooms'] for p in room + figure['origin']})
    markers = set(map(tuple, np.argwhere(inp == info['marker'])))
    for i, cells in enumerate(owned):
        near = {(int(r + dr), int(c + dc)) for r, c in cells
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
        assert not (near & (owned[1 - i] | markers)), 'figures and shutters touch'
    sizes = [len(p) for p in donor['rooms']]
    mapping = info['mappings'][0]
    signs = {int(np.sign(len(target['rooms'][mapping[i]]) - sizes[i])) for i in range(len(sizes))}
    db = _box(donor['cells'])
    tb = _box(target['cells'])
    row_overlap = min(db[2], tb[2]) > max(db[0], tb[0])
    col_overlap = min(db[3], tb[3]) > max(db[1], tb[1])
    layout = 'interleaved' if row_overlap and col_overlap else 'side' if row_overlap else 'stack' if col_overlap else 'diagonal'
    flags = _alternatives(inp, info, out)
    effective = 0
    for k in range(len(info['strokes'])):
        subset = info['strokes'][:k] + info['strokes'][k + 1:]
        other = _answer(info, _pair_band(inp.shape, info['slope'], subset))[0]
        effective += int(not np.array_equal(other, out))
    return dict(rooms=len(sizes), strokes=len(info['strokes']), active=sum(active),
                effective_strokes=effective,
                bg=info['bg'], wall=info['wall'], marker=info['marker'],
                slope=info['slope'], layout=layout, area_signs=tuple(sorted(signs)),
                colors=len(set(donor['colors'])), reused=len(set(donor['colors'])) < len(sizes),
                nonrectangular=sum(len(p) != (_box(p)[2] - _box(p)[0]) * (_box(p)[3] - _box(p)[1])
                                   for p in donor['rooms'] + target['rooms']),
                equal_area_colors=any(sizes[i] == sizes[j] and donor['colors'][i] != donor['colors'][j]
                                      for i in range(len(sizes)) for j in range(i)),
                deformation=(_relations(donor['rooms']), donor['crop'].shape, target['crop'].shape),
                target_structure=(out.shape, target['walls'].tobytes()), **flags)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp = _grid(pair['input'])
            key = _key(inp)
            assert key not in seen, 'repeated input'
            seen.add(key)
            features[split].append(measure_pair(pair))
            assert np.array_equal(solve(inp), np.asarray(pair['output']))
    train = features['train']
    # Input-role colors are local; demonstrations make this visibly literal.
    assert len({m['bg'] for m in train}) >= 2
    assert len({m['wall'] for m in train}) >= 2
    assert len({m['target_structure'] for m in train}) >= 2
    for name in ('all_rooms', 'center', 'majority', 'unbounded', 'consecutive', 'pixel_centers', 'area_match', 'receiver_gate', 'longitudinal_only'):
        assert any(m[name] for m in train), f'missing {name} training counterexample'
    assert any(m['nonrectangular'] for m in train), 'missing deformed room evidence'


def self_test():
    # Hand-checked inclusive endpoints, strict transverse boundaries, and a
    # scanline falling between one segment's actual pixel centers.
    strokes = [(0, 0, 8), (5, 5, 9)]
    band = _band((8, 8), 1, strokes)
    assert band[2, 3] and band[3, 5] and band[1, 5]
    assert not band[4, 5] and not band[3, 3] and not band[1, 6]
    assert not _pair_band((8, 8), 1, strokes, 'pixel_centers').any()
    assert np.array_equal(band, _pair_band((8, 8), 1, strokes))
    structures, shapes, profiles, roles, records = set(), set(), set(), set(), []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                m = measure_pair(pair)
                records.append(m)
                structures.add(m['target_structure'])
                profiles.add(m['deformation'])
                roles.add((m['bg'], m['wall'], m['marker']))
                shapes.add(np.asarray(pair['input']).shape)
    assert len(structures) >= 140 and len(profiles) >= 100
    assert len(shapes) >= 50 and len(roles) >= 100
    assert {m['slope'] for m in records} == {-1, 1}
    assert {m['rooms'] for m in records} >= set(range(3, 11))
    assert {m['strokes'] for m in records} >= set(range(2, 6))
    assert sum(m['effective_strokes'] >= 4 for m in records) >= 10
    assert any(m['reused'] and m['nonrectangular'] for m in records)
    assert any(m['area_signs'] == (-1, 0, 1) for m in records)
    assert any(m['equal_area_colors'] for m in records)
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(31, a, b)
            validate_generated_puzzle(p)
            assert (len(p['train']), len(p['test'])) == (a, b)
    p, q = generate_puzzle(42, 2, 1), generate_puzzle(42, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in p for x, y in zip(p[s], q[s]) for k in ('input', 'output'))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Correct labels alone are insufficient: two simple two-shutter examples
    # never demonstrate bypassing an inactive middle shutter.
    a = generate_puzzle(0, 2, 1)
    first = measure_pair(a['train'][0])
    for seed in range(1, 12):
        b = generate_puzzle(seed, 2, 1)
        second = measure_pair(b['train'][0])
        if first['bg'] != second['bg'] and first['wall'] != second['wall'] and first['target_structure'] != second['target_structure']:
            break
    inadequate = dict(train=[a['train'][0], b['train'][0]], test=a['test'])
    try:
        validate_generated_puzzle(inadequate)
    except AssertionError as error:
        assert 'consecutive' in str(error)
    else:
        raise AssertionError('missing middle-shutter demonstration accepted')
    # Negative semantic check: a correctly shaped but altered answer must fail.
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted receiver accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds and all legal counts')
