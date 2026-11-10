"""Aperture census: continuous diagonal exposure selects enclosed rooms.

Only standard-library modules and NumPy are used. Solving reads the input;
generation and validation do not supply it with a palette, seed, or latent mask.
"""
import numbers

import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return g


def _key(g):
    return g.shape, np.asarray(g, dtype=np.int8).tobytes()


def _strokes(g, color, slope):
    lines = {}
    for r, c in np.argwhere(g == color):
        r, c = int(r), int(c)
        lines.setdefault(c - slope * r, []).append(r)
    result = []
    for q, rows in sorted(lines.items()):
        rows.sort()
        runs = []
        for r in rows:
            if not runs or r != runs[-1][-1] + 1:
                runs.append([])
            runs[-1].append(r)
        for run in runs:
            if len(run) < 2:
                return None
            result.append((q, 2 * run[0] + slope * q,
                           2 * run[-1] + slope * q))
    return result if len(result) >= 2 else None


def _bounded_regions(wall):
    """Bounded four-connected free components, with explicit outside padding."""
    # A free exterior ring plus a blocked sentinel ring eliminates per-neighbor
    # array indexing and boundary tests. All traversal state remains input-local.
    stride = wall.shape[1] + 4
    free = [False] * stride
    for row in np.pad(~wall, 1, constant_values=True).tolist():
        free.extend([False, *row, False])
    free.extend([False] * stride)
    start = stride + 1
    free[start] = False
    todo = [start]
    for cell in todo:
        for neighbor in (cell - stride, cell + stride, cell - 1, cell + 1):
            if free[neighbor]:
                free[neighbor] = False
                todo.append(neighbor)
    regions = []
    for start, available in enumerate(free):
        if not available:
            continue
        todo = [start]
        free[start] = False
        cells = []
        for cell in todo:
            y, x = divmod(cell, stride)
            cells.append((y - 2, x - 2))
            for neighbor in (cell - stride, cell + stride, cell - 1, cell + 1):
                if free[neighbor]:
                    free[neighbor] = False
                    todo.append(neighbor)
        regions.append(cells)
    return regions


def _room_boxes(wall):
    rooms = []
    for cells in _bounded_regions(wall):
        a = np.asarray(cells)
        top, left = map(int, a.min(axis=0))
        bottom, right = map(int, a.max(axis=0))
        if len(cells) != (bottom - top + 1) * (right - left + 1):
            raise ValueError("elementary rooms must be rectangular")
        if bottom - top < 1 or right - left < 1:
            raise ValueError("room interiors need at least two cells on each axis")
        rooms.append((top, left, bottom, right))
    return sorted(rooms)


def _walls(shape, boxes):
    wall = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:
        wall[t, l:r + 1] = wall[b, l:r + 1] = True
        wall[t:b + 1, l] = wall[t:b + 1, r] = True
    return wall


def _recover(g, color, guide):
    visible = g == color
    cells = np.argwhere(visible)
    if not len(cells):
        raise ValueError("empty structural role")
    t, l = map(int, cells.min(axis=0))
    b, r = map(int, cells.max(axis=0))
    if b - t < 3 or r - l < 3:
        raise ValueError("frame too small")
    wall = visible.copy()
    outer = _walls(g.shape, [(t, l, b, r)])
    if np.any(outer & ~((g == color) | (g == guide))):
        raise ValueError("outer boundary has an unsupported gap")
    wall |= outer
    # Outer corners provide endpoints. Bridge only guide-colored interruptions
    # between two collinear wall supports, never an observed background gap.
    local_wall = wall[t:b + 1, l:r + 1]
    local_grid = g[t:b + 1, l:r + 1]
    for axis in (0, 1):
        field = local_wall if axis == 0 else local_wall.T
        observed = local_grid if axis == 0 else local_grid.T
        allowed = (observed == color) | (observed == guide)
        segments = np.cumsum(~allowed, axis=1)
        left = np.maximum.accumulate(np.where(field, segments, -1), axis=1)
        right = np.minimum.accumulate(
            np.where(field, segments, field.shape[1] + 1)[:, ::-1], axis=1
        )[:, ::-1]
        # Equal segment IDs mean both supports lie in one uninterrupted run
        # of wall/guide cells. This is exactly the supported-gap bridge above.
        field |= allowed & (left == right)
    local_rooms = _room_boxes(wall[t:b + 1, l:r + 1])
    rooms = [(a + t, x + l, z + t, y + l) for a, x, z, y in local_rooms]
    if not rooms or len(rooms) > 30:
        raise ValueError("missing or excessive compartments")
    boundaries = [(a - 1, x - 1, z + 1, y + 1) for a, x, z, y in rooms]
    if not np.array_equal(wall, _walls(g.shape, boundaries)):
        raise ValueError("unexplained wall fragments")
    return dict(color=color, box=(t, l, b, r), wall=wall, rooms=rooms)


def _overlap(a, b):
    t, l, z, r = a
    y, x, v, w = b
    return not (z < y or v < t or r < x or w < l)


def _decode(input_grid):
    g = _grid(input_grid)
    colors, counts = np.unique(g, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if counts.max() <= g.size / 2:
        raise ValueError("background must occupy a strict majority")
    candidates = []
    for color in colors:
        if color == bg:
            continue
        for slope in (-1, 1):
            strokes = _strokes(g, int(color), slope)
            if strokes is not None:
                candidates.append((int(color), slope, strokes))
    if len(candidates) != 1:
        raise ValueError("guide role and slope must be unique")
    guide, slope, strokes = candidates[0]
    # Adjacent diagonal lines cannot be confused with thick or crossing strokes.
    distinct_q = sorted({s[0] for s in strokes})
    if any(y - x < 3 for x, y in zip(distinct_q, distinct_q[1:])):
        raise ValueError("guide lines are too close")
    networks = [_recover(g, int(color), guide)
                for color in colors if color not in (bg, guide)]
    if not 1 <= len(networks) <= 8:
        raise ValueError("expected one to eight differently colored networks")
    for i, n in enumerate(networks):
        if any(_overlap(n['box'], other['box']) for other in networks[:i]):
            raise ValueError("frame bounding boxes must be disjoint")
    return dict(grid=g, background=bg, guide=guide, slope=slope,
                strokes=strokes, networks=networks)


def _groups(strokes):
    unseen = set(range(len(strokes)))
    groups = []
    while unseen:
        group = [unseen.pop()]
        for i in group:
            a = strokes[i]
            neighbors = [j for j in unseen
                         if max(a[1], strokes[j][1]) <= min(a[2], strokes[j][2])]
            unseen.difference_update(neighbors)
            group.extend(neighbors)
        groups.append(group)
    return groups


def _band(info, alternative=''):
    rr, cc = np.indices(info['grid'].shape)
    s = info['slope']
    q, t = cc - s * rr, rr + s * cc
    strokes = info['strokes']
    if alternative == 'infinite':
        return (q > min(a[0] for a in strokes)) & (q < max(a[0] for a in strokes))
    if alternative in ('consecutive', 'pixel_centers'):
        pairs = [(a, b) for i, a in enumerate(strokes) for b in strokes[i + 1:]]
        if alternative == 'consecutive':
            pairs = []
            for ids in _groups(strokes):
                ordered = sorted((strokes[i] for i in ids))
                pairs.extend(zip(ordered, ordered[1:]))
        result = np.zeros(q.shape, dtype=bool)
        for a, b in pairs:
            m = ((t >= max(a[1], b[1])) & (t <= min(a[2], b[2]))
                 & (q > min(a[0], b[0])) & (q < max(a[0], b[0])))
            if alternative == 'pixel_centers':
                m &= ((t - s * a[0]) % 2 == 0) & ((t - s * b[0]) % 2 == 0)
            result |= m
        return result
    # Inclusive continuous intervals include scanlines between marker centers.
    low = np.full(q.shape, 1000)
    high = np.full(q.shape, -1000)
    active = np.zeros(q.shape, dtype=int)
    for x, a, b in strokes:
        m = (t >= a) & (t <= b)
        active += m
        low[m] = np.minimum(low[m], x)
        high[m] = np.maximum(high[m], x)
    return (active >= 2) & (q > low) & (q < high)


def _counts(info, alternative='', geometry=None):
    if geometry is None:
        geometry = _band(info, alternative)
    light = geometry & (info['grid'] == info['background'])
    result = []
    for n in info['networks']:
        if alternative == 'fragments':
            t, l, b, r = n['box']
            result.append(_patch_count(light[t:b + 1, l:r + 1]))
            continue
        if alternative == 'unrepaired':
            t, l, b, r = n['box']
            visible = info['grid'][t:b + 1, l:r + 1] == n['color']
            regions = _bounded_regions(visible)
            result.append(sum(any(light[y + t, x + l] for y, x in region)
                              for region in regions))
            continue
        if alternative == 'full_lattice':
            t, l, b, r = n['box']
            wall = n['wall'][t:b + 1, l:r + 1]
            hs = np.flatnonzero(np.any(wall[:, :-1] & wall[:, 1:], axis=1))
            vs = np.flatnonzero(np.any(wall[:-1, :] & wall[1:, :], axis=0))
            lattice = np.zeros(wall.shape, dtype=bool)
            lattice[hs, :] = True
            lattice[:, vs] = True
            result.append(sum(any(light[y + t, x + l] for y, x in region)
                              for region in _bounded_regions(lattice)))
            continue
        votes = 0
        for t, l, b, r in n['rooms']:
            touched = bool(light[t:b + 1, l:r + 1].any())
            if alternative == 'total_rooms':
                touched = True
            elif alternative == 'center':
                touched = bool(geometry[(t + b) // 2, (l + r) // 2])
            elif alternative == 'contained':
                touched = bool(geometry[t:b + 1, l:r + 1].all())
            votes += touched
        result.append(min(votes, 1) if alternative == 'whole_frame' else votes)
    return result


def _patch_count(mask):
    """A deliberately weaker alternative: count exposed pixel components."""
    remaining = set(map(tuple, np.argwhere(mask)))
    count = 0
    while remaining:
        count += 1
        todo = [remaining.pop()]
        for y, x in todo:
            for cell in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if cell in remaining:
                    remaining.remove(cell)
                    todo.append(cell)
    return count


def _encode(info, counts, positive_spatial=False):
    ordered = sorted(zip(info['networks'], counts),
                     key=lambda a: (int(a[1] > 0) if positive_spatial else a[1],
                                    a[0]['box'][1], a[0]['box'][0]))
    out = np.full((len(ordered), max(1, max(counts))), info['guide'], dtype=np.int8)
    for row, (n, count) in enumerate(ordered):
        out[row, :count] = n['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _decode(input_grid)
    return _encode(info, _counts(info))


_ALTERNATIVES = ('total_rooms', 'whole_frame', 'center', 'contained',
                 'infinite', 'consecutive', 'pixel_centers', 'fragments', 'unrepaired',
                 'positive_spatial', 'full_lattice')


def _evidence(info, names=None, counts=None):
    geometry = _band(info)
    if counts is None:
        counts = _counts(info, geometry=geometry)
    answer = _encode(info, counts)
    flags = dict(zero=0 in counts,
                 tie=len(set(x for x in counts if x > 0)) < sum(x > 0 for x in counts),
                 padding=min(counts) < max(counts))
    spatial = sorted(zip(info['networks'], counts), key=lambda x: x[0]['box'][1])
    flags['order'] = [v for _, v in spatial] != sorted(counts)
    flags['occlusion'] = any(np.any(n['wall'] & (info['grid'] == info['guide']))
                             for n in info['networks'])
    flags['partial'] = any(_partial(n) for n in info['networks'])
    requested = set(names) if names is not None else set(_ALTERNATIVES)
    if names is not None and any(not flags[name] for name in requested if name in flags):
        return flags
    # Cheap necessary witnesses reject a candidate before expensive topology
    # alternatives. Checking fewer predicates cannot change accepted inputs.
    order = ('total_rooms', 'whole_frame', 'positive_spatial', 'center', 'contained',
             'infinite', 'consecutive', 'pixel_centers', 'fragments', 'unrepaired',
             'full_lattice')
    for name in order:
        if name not in requested:
            continue
        alt_geometry = None if name in ('infinite', 'consecutive', 'pixel_centers') else geometry
        prediction = _encode(info, _counts(info, name, geometry=alt_geometry),
                             positive_spatial=name == 'positive_spatial')
        flags[name] = not np.array_equal(answer, prediction)
        if names is not None and not flags[name]:
            return flags
    return flags


def _partial(n):
    t, l, b, r = n['box']
    wall = n['wall'][t:b + 1, l:r + 1]
    hs = np.flatnonzero(np.any(wall[:, :-1] & wall[:, 1:], axis=1))
    vs = np.flatnonzero(np.any(wall[:-1, :] & wall[1:, :], axis=0))
    return any(not wall[y].all() for y in hs) or any(not wall[:, x].all() for x in vs)


def _tie_clear(info, counts):
    for i, a in enumerate(info['networks']):
        for j, b in enumerate(info['networks'][:i]):
            if counts[i] == counts[j] and counts[i] > 0:
                if not (a['box'][3] < b['box'][1] or b['box'][3] < a['box'][1]):
                    return False
    return True


def _partition(rng, h, w, count, style):
    outer = (0, 0, h - 1, w - 1)
    capacity = lambda box: ((box[2] - box[0]) // 3) * ((box[3] - box[1]) // 3)
    if capacity(outer) < count:
        return None
    factors = [(a, count // a) for a in range(1, count + 1)
               if count % a == 0 and 3 * a < h and 3 * (count // a) < w
               and (style == 'grid' or a in (1, count))]
    if style in ('strip', 'grid') and factors:
        ny, nx = factors[int(rng.integers(len(factors)))]
        def offsets(n, span):
            gaps = np.full(n, 3, dtype=int)
            for _ in range(span - 3 * n):
                gaps[int(rng.integers(n))] += 1
            return [0] + list(map(int, np.cumsum(gaps)))
        ys, xs = offsets(ny, h - 1), offsets(nx, w - 1)
        boxes = [(ys[i], xs[j], ys[i + 1], xs[j + 1])
                 for i in range(ny) for j in range(nx)]
    else:
        boxes = [outer]
        first_axis = None
        while len(boxes) < count:
            choices = []
            for i, (t, l, b, r) in enumerate(boxes):
                for axis, lo, hi in ((0, t, b), (1, l, r)):
                    if style == 'partial' and len(boxes) == 2 and axis == first_axis:
                        continue
                    for cut in range(lo + 3, hi - 2):
                        halves = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else ((t, l, b, cut), (t, cut, b, r))
                        cap = sum(capacity(x) for j, x in enumerate(boxes) if j != i)
                        if cap + sum(capacity(x) for x in halves) >= count:
                            choices.append((i, axis, halves))
            if not choices:
                return None
            i, axis, halves = choices[int(rng.integers(len(choices)))]
            if len(boxes) == 1:
                first_axis = axis
            boxes[i:i + 1] = halves
    return _walls((h, w), boxes)


def _frames(rng, shape, number, colors, maximum, force_partial=False):
    h, w = shape
    frames = []
    for index in range(number):
        count = int(rng.integers(1, maximum + 1))
        style = str(rng.choice(['strip', 'grid', 'recursive', 'partial']))
        if index == 0 and force_partial:
            count = int(rng.integers(3, maximum + 1))
            style = 'partial'
        for _ in range(110):
            cap_h = min(h, 20 if number <= 2 else 15)
            cap_w = min(w, 20 if number <= 2 else 15)
            low = 7 if style == 'partial' and count >= 3 else 4
            bh, bw = int(rng.integers(low, cap_h + 1)), int(rng.integers(low, cap_w + 1))
            if ((bh - 1) // 3) * ((bw - 1) // 3) < count:
                continue
            top, left = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
            box = (top, left, top + bh - 1, left + bw - 1)
            if any(_overlap(box, n['box']) for n in frames):
                continue
            wall = _partition(rng, bh, bw, count, style)
            if wall is None:
                continue
            full = np.zeros(shape, dtype=bool)
            full[top:top + bh, left:left + bw] = wall
            frames.append(dict(color=colors[index], box=box, wall=full))
            break
        else:
            return None
    return frames


def _guides(rng, shape, number, witness=False, split_run=False, odd_gap=False):
    h, w = shape
    n = number - int(split_run)
    gaps = list(map(int, rng.integers(3, 8, size=n - 1)))
    if odd_gap and n == 2:
        gaps[0] = int(rng.choice([3, 5, 7, 9, 11]))
    span = sum(gaps)
    lower, upper = -h + 3, w - 4 - span
    if lower > upper:
        return None
    center = (w - h - span) // 2
    base = int(rng.integers(max(lower, center - 5), min(upper, center + 5) + 1))
    qs = [base]
    for gap in gaps:
        qs.append(qs[-1] + gap)
    limits = [(max(0, -q), min(h - 1, w - 1 - q)) for q in qs]
    low = max(2 * a + q for q, (a, b) in zip(qs, limits))
    high = min(2 * b + q for q, (a, b) in zip(qs, limits))
    core = int(rng.integers(low, high + 1)) if low <= high else (h + w) // 2
    runs = []
    for i, (q, (a, b)) in enumerate(zip(qs, limits)):
        available = b - a + 1
        if available < 2:
            return None
        if witness and i in (0, n - 1):
            start = a + int(rng.integers(0, min(3, available - 2) + 1))
            end = b - int(rng.integers(0, min(3, b - start - 1) + 1))
        else:
            length = int(rng.integers(2, min(available, 3 if witness else 18) + 1))
            if rng.random() < .75 and not witness:
                anchor = min(b, max(a, (core - q) // 2))
                lo = max(a, anchor - length + 1)
                hi = min(anchor, b - length + 1)
                start = int(rng.integers(lo, hi + 1))
            else:
                start = int(rng.integers(a, b - length + 2))
            end = start + length - 1
        runs.append((q, start, end))
    if split_run:
        eligible = [i for i, (_, a, b) in enumerate(runs) if b - a + 1 >= 6]
        if not eligible:
            return None
        i = int(rng.choice(eligible))
        q, a, b = runs.pop(i)
        cut = int(rng.integers(a + 2, b - 2))
        runs.extend([(q, a, cut - 1), (q, cut + 2, b)])
    mask = np.zeros(shape, dtype=bool)
    for q, a, b in runs:
        rows = np.arange(a, b + 1)
        mask[rows, rows + q] = True
    return mask


def _make_pair(rng, roles, number_strokes, required, split_run=False, witness=False):
    bg, guide, pool = roles
    for _ in range(4000):
        number = int(rng.choice([1, 2, 2, 3, 3, 4]))
        minimum = 12 if number == 1 else 18 if number <= 3 else 23
        shape = (int(rng.integers(minimum, 31)), int(rng.integers(minimum, 31)))
        colors = list(map(int, rng.choice(pool, size=number, replace=False)))
        frames = _frames(rng, shape, number, colors, int(rng.integers(4, 8)),
                         force_partial='partial' in required)
        if frames is None:
            continue
        guides = _guides(rng, shape, number_strokes, witness, split_run,
                         odd_gap=number_strokes == 2)
        if guides is None:
            continue
        g = np.full(shape, bg, dtype=np.int8)
        for n in frames:
            g[n['wall']] = n['color']
        g[guides] = guide
        try:
            info = _decode(g)
        except ValueError:
            continue
        latent = {n['color']: n['wall'] for n in frames}
        if any(not np.array_equal(n['wall'], latent[n['color']]) for n in info['networks']):
            continue
        counts = _counts(info)
        if max(counts) == 0 or not _tie_clear(info, counts):
            continue
        flags = _evidence(info, names=required, counts=counts) if required else {}
        if not all(flags.get(name, False) for name in required):
            continue
        # Both slopes occur among the first two training examples: reflection
        # is applied by the caller. No geometric direction is a hidden setting.
        return dict(input=g, output=_encode(info, counts))
    raise RuntimeError("bounded aperture-scene sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    ne = _integer(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError("num_train must be greater than num_test")
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    roles = (palette[0], palette[1], palette[2:])
    base_slope = int(rng.choice([-1, 1]))
    split_runs = nt >= 3 and bool(rng.integers(2))
    puzzle = dict(train=[], test=[])
    seen = set()
    for split, count in (('train', nt), ('test', ne)):
        for i in range(count):
            if split == 'train' and i == 0:
                required = ('consecutive', 'total_rooms', 'whole_frame',
                            'center', 'contained', 'fragments', 'unrepaired',
                            'positive_spatial', 'full_lattice', 'partial', 'occlusion')
                nstrokes, witness, split_run = 3, True, False
            elif split == 'train' and i == 1:
                required = ('pixel_centers', 'infinite', 'zero', 'order', 'padding')
                nstrokes, witness, split_run = 2, False, False
            else:
                required = ('total_rooms', 'whole_frame') if split == 'test' else ()
                nstrokes = int(rng.integers(2, 4)) if split == 'train' else int(rng.integers(4, 7))
                witness = False
                split_run = split_runs and (i == 2 if split == 'train' else bool(rng.integers(2)))
                if split_run and nstrokes < 3:
                    nstrokes = 3
            for _ in range(20):
                pair = _make_pair(rng, roles, nstrokes, required, split_run, witness)
                slope = base_slope * (-1 if split == 'train' and i % 2 else 1)
                if split == 'test':
                    slope *= int(rng.choice([-1, 1]))
                if slope == -1:
                    pair = {k: np.fliplr(v).copy() for k, v in pair.items() if k == 'input'}
                    pair['output'] = solve(pair['input'])
                info = _decode(pair['input'])
                flags = _evidence(info)
                # Reflection changes spatial ordering: check the returned scene.
                if not all(flags[x] for x in required):
                    continue
                key = _key(pair['input'])
                if key in seen:
                    continue
                seen.add(key)
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError("distinct demonstrated scene construction exhausted")
    # Combine a needed tie witness with the two-guide demonstration, preserving
    # its zero, finite-endpoint, continuous-scanline, and ordering evidence.
    if any(_evidence(_decode(p['input']))['tie'] for p in puzzle['test']):
        if not any(_evidence(_decode(p['input']))['tie'] for p in puzzle['train']):
            required = ('pixel_centers', 'infinite', 'zero', 'order', 'padding', 'tie')
            for _ in range(20):
                pair = _make_pair(rng, roles, 2, required)
                if base_slope == 1:
                    pair['input'] = np.fliplr(pair['input']).copy()
                    pair['output'] = solve(pair['input'])
                if (_key(pair['input']) not in seen
                        and all(_evidence(_decode(pair['input']))[x] for x in required)):
                    puzzle['train'][1] = pair
                    break
            else:
                raise RuntimeError("tie witness construction exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, roles, evidence = set(), set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(g) not in seen, 'duplicate input'
            seen.add(_key(g))
            info = _decode(g)
            counts = _counts(info)
            roles.add((info['background'], info['guide']))
            assert _tie_clear(info, counts), 'positive ties need separated horizontal extents'
            assert np.array_equal(out, _encode(info, counts)), 'incorrect census'
            assert out.shape == (len(info['networks']), max(1, max(counts)))
            flags = _evidence(info)
            evidence[split].append(flags)
            if split == 'test':
                assert flags['total_rooms'] and flags['whole_frame'], 'both parent concepts must affect each test answer'
            # Topological count has an independent planar graph identity.
            for n in info['networks']:
                wall = n['wall']
                edges = np.count_nonzero(wall[:, :-1] & wall[:, 1:])
                edges += np.count_nonzero(wall[:-1, :] & wall[1:, :])
                tiny = np.count_nonzero(wall[:-1, :-1] & wall[1:, :-1]
                                        & wall[:-1, 1:] & wall[1:, 1:])
                assert int(edges - wall.sum() + 1 - tiny) == len(n['rooms'])
    assert len(roles) == 1, 'background and guide roles stay fixed within a puzzle'
    for name in _ALTERNATIVES + ('zero', 'order', 'padding', 'partial', 'occlusion'):
        assert any(e[name] for e in evidence['train']), f'missing training witness: {name}'
    if any(e['tie'] for e in evidence['test']):
        assert any(e['tie'] for e in evidence['train']), 'missing tie demonstration'
    assert {_decode(p['input'])['slope'] for p in puzzle['train']} == {-1, 1}


def _fingerprint(puzzle):
    return tuple((_key(p['input']), _key(p['output']))
                 for s in ('train', 'test') for p in puzzle[s])


def self_test():
    # Analytical fixture: the left-bottom room is outside the aperture; the
    # upper-left and right rooms vote once each. The separate small frame is dark.
    # The short interior guide ends before the right room's first contact.
    def fixture(extra_divider=False, end=9):
        g = np.zeros((15, 22), dtype=np.int8)
        g[_walls(g.shape, [(2, 2, 8, 7), (8, 2, 11, 7), (2, 7, 11, 13)])] = 2
        if extra_divider:
            g[5, 2:8] = 2
        g[_walls(g.shape, [(0, 15, 3, 20)])] = 3
        for q, a, b in ((-3, 3, 13), (0, 3, 6), (4, 0, end)):
            rows = np.arange(a, b + 1)
            g[rows, rows + q] = 7
        return g
    g = fixture()
    assert np.array_equal(solve(g), [[7, 7], [2, 2]])
    info = _decode(g)
    assert _counts(info, 'fragments') == [3, 0]
    assert _band(info)[5, 8] and not _band(info, 'consecutive')[5, 8]
    assert np.array_equal(solve(fixture(extra_divider=True)), [[7, 7, 7], [2, 2, 2]])
    assert np.array_equal(solve(fixture(end=4)), [[7], [2]])
    puzzles, shapes, palettes, topologies, stroke_counts, room_counts = [], set(), set(), set(), set(), set()
    partial = occluded = zeros = ties = edges = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(_fingerprint(p))
        for split in ('train', 'test'):
            for pair in p[split]:
                info = _decode(pair['input'])
                counts = _counts(info)
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
                shapes.add(pair['input'].shape)
                palettes.add((info['background'], info['guide']))
                stroke_counts.add(len(info['strokes']))
                flags = _evidence(info)
                partial += flags['partial']
                occluded += flags['occlusion']
                zeros += flags['zero']
                ties += flags['tie']
                for n in info['networks']:
                    rooms = n['rooms']
                    room_counts.add(len(rooms))
                    ys = sorted({v for a, _, b, _ in rooms for v in (a - 1, b + 1)})
                    xs = sorted({v for _, a, _, b in rooms for v in (a - 1, b + 1)})
                    topologies.add(tuple(sorted((ys.index(a - 1), xs.index(x - 1),
                                                ys.index(b + 1), xs.index(y + 1))
                                               for a, x, b, y in rooms)))
                    t, l, b, r = n['box']
                    edges += t == 0 or l == 0 or b == pair['input'].shape[0] - 1 or r == pair['input'].shape[1] - 1
    assert len(set(puzzles)) == 42
    assert len(shapes) >= 40 and len(palettes) >= 25 and len(topologies) >= 20
    assert {2, 3, 4, 5, 6} <= stroke_counts
    assert {1, 2, 3, 4, 5, 6} <= room_counts
    assert partial >= 42 and occluded >= 42 and zeros >= 42 and ties > 0 and edges > 0
    assert _fingerprint(generate_puzzle(0)) == puzzles[0]
    # Boundary requests; verify.py separately exercises every legal combination.
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(1000 + 10 * nt + ne, nt, ne)
        assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Color relabeling and spatial symmetries commute with input-only solving.
    p = generate_puzzle(7, 2, 1)
    permutation = np.asarray([7, 0, 9, 3, 6, 1, 8, 4, 2, 5], dtype=np.int8)
    for pair in p['train'] + p['test']:
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
        info = _decode(pair['input'])
    if not _evidence(info)['tie']:
            for g in (np.rot90(pair['input']), pair['input'].T, np.fliplr(pair['input'])):
                assert np.array_equal(solve(g), pair['output'])
    # Correct labels on two one-network inputs cannot teach positive sorting.
    teaching = []
    for g in (fixture(), np.fliplr(fixture(end=4)).copy()):
        g[g == 3] = 0
        teaching.append(dict(input=g, output=solve(g)))
    g = fixture()
    inadequate = dict(train=teaching, test=[dict(input=g, output=solve(g))])
    try:
        validate_generated_puzzle(inadequate)
    except AssertionError as exc:
        assert 'missing training witness: positive_spatial' in str(exc)
    else:
        raise AssertionError('inadequate demonstration set accepted')
    # Negative checks cover labels and actual teaching sufficiency.
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
