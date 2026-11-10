"""Census shadows: a global object census written into local lantern shadows.

Only the standard library and NumPy are used. Prediction reads the input only;
construction carries explicit geometry/objects, which the validator reconstructs.
"""

from collections import Counter
import numpy as np


HOST_SEEDS = (0, 334749851, 126868453, 621890626, 931396489, 38417068,
              221080433, 722309310, 355852345, 161293519, 762825054,
              767159463, 245338501, 297274244, 762002697, 396876505,
              219221716, 505603220, 99401420, 949226633, 977048114,
              41709770, 209790807, 996369468, 724706581, 329762055,
              216722534, 285610332, 611908134, 319541950, 295261698,
              835081035, 372864295, 205673891, 818861481, 551668568,
              795970038, 280562492, 242573811, 522252436, 84769255,
              345987266)
DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= s <= 30 for s in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("Expected an integer ARC grid with bounds 1..30 and colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _puzzle_key(p):
    return tuple(tuple((_key(np.asarray(q['input'])), _key(np.asarray(q['output'])))
                       for q in p[s]) for s in ('train', 'test'))


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
             if (dy or dx) and (diagonal or abs(dy) + abs(dx) == 1)]
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        group, stack = {start}, [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in steps:
                q = (y + dy, x + dx)
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(frozenset((int(y), int(x)) for y, x in group))
    return groups


def _fit_terminal(values, sign):
    """Fit the longest suffix containing at least two outward changes."""
    xs = np.arange(len(values)) - (len(values) - 1)
    candidates = []
    for period in (1, 2, 3):
        for phase in range(period):
            ys = values[-1] + sign * ((xs + phase) // period)
            length = 0
            for equal in (ys == values)[::-1]:
                if not equal:
                    break
                length += 1
            if length >= 3 and len(set(values[-length:])) >= 3:
                candidates.append((length, period, phase))
    if not candidates:
        raise ValueError("Unreadable terminal rail")
    longest = max(t[0] for t in candidates)
    best = [t for t in candidates if t[0] == longest]
    futures = {tuple(int(values[-1] + sign * ((d + phase) // period))
                     for d in range(1, 31)) for _, period, phase in best}
    if len(futures) != 1:
        raise ValueError("Ambiguous rail continuation")
    _, period, phase = min(best, key=lambda t: (t[1], t[2]))
    return period, phase, longest


def _read_input(input_grid):
    a0 = _grid(input_grid)
    frequency = np.bincount(a0.ravel(), minlength=10)
    bg = int(np.argmax(frequency))
    if frequency[bg] <= a0.size / 2:
        raise ValueError("The background must dominate")
    matches = []
    for rotation in range(4):
        a = np.rot90(a0, -rotation)
        for wall in map(int, np.unique(a)):
            if wall == bg:
                continue
            wallmask = a == wall
            points = np.argwhere(wallmask)
            if len(points) < 11 or len(_components(wallmask)) != 1:
                continue
            rear, mouth = int(points[:, 1].min()), int(points[:, 1].max())
            cap = np.flatnonzero(wallmask[:, rear])
            if len(cap) < 3 or np.any(np.diff(cap) != 1) or mouth - rear < 4:
                continue
            upper, lower = [], []
            for x in range(rear + 1, mouth + 1):
                rows = np.flatnonzero(wallmask[:, x])
                if len(rows) != 2 or rows[1] - rows[0] < 2:
                    break
                upper.append(int(rows[0]))
                lower.append(int(rows[1]))
            else:
                if (abs(upper[0] - int(cap[0])) > 1
                        or abs(lower[0] - int(cap[-1])) > 1
                        or np.any(np.abs(np.diff(upper)) > 1)
                        or np.any(np.abs(np.diff(lower)) > 1)):
                    continue
                try:
                    fits = (_fit_terminal(np.array(upper), -1),
                            _fit_terminal(np.array(lower), 1))
                except ValueError:
                    continue
                footprint = np.zeros(a.shape, bool)
                bounds = []
                for x in range(rear + 1, a.shape[1]):
                    if x <= mouth:
                        lo, hi = upper[x - rear - 1], lower[x - rear - 1]
                    else:
                        d = x - mouth
                        lo = upper[-1] - (d + fits[0][1]) // fits[0][0]
                        hi = lower[-1] + (d + fits[1][1]) // fits[1][0]
                    footprint[:, x] = (np.arange(a.shape[0]) > lo) & (np.arange(a.shape[0]) < hi)
                    bounds.append((lo, hi))
                chamber = footprint.copy()
                chamber[:, mouth + 1:] = False
                if len(_components(chamber, False)) != 1:
                    continue
                interior_colors = set(map(int, a[chamber])) - {bg}
                if len(interior_colors) != 1:
                    continue
                payload = interior_colors.pop()
                patch = a == payload
                if (payload == wall or np.any(patch & ~chamber)
                        or len(_components(patch, False)) != 1):
                    continue
                objects = []
                for color in sorted(set(map(int, np.unique(a))) - {bg, wall, payload}):
                    objects.extend((color, cells) for cells in _components(a == color))
                matches.append(dict(a=a.copy(), background=bg, wall=wall, payload=payload,
                                    rotation=rotation, rear=rear, mouth=mouth, bounds=tuple(bounds),
                                    fits=fits, footprint=footprint, chamber=chamber, objects=objects))
    if len(matches) != 1:
        raise ValueError("Input must identify exactly one chamber and payload")
    return matches[0]


def _paint(a, footprint, objects, background, payload, census=None):
    """First-hit lane ownership, chamber lighting, and fixed-phase census stamps."""
    counts = Counter(color for color, _ in objects) if census is None else census
    owner = np.full(a.shape, -1, int)
    for i, (_, cells) in enumerate(objects):
        for y, x in cells:
            owner[y, x] = i
    output = a.copy()
    first_owner = np.full(a.shape[0], -1, int)
    anchors, scheduled = {}, []
    for y in range(a.shape[0]):
        lane = np.flatnonzero(footprint[y])
        hits = [int(x) for x in lane if owner[y, x] >= 0]
        if not hits:
            output[y, lane[a[y, lane] == background]] = payload
            continue
        first = min(hits)
        i = int(owner[y, first])
        first_owner[y] = i
        color, cells = objects[i]
        before = lane[(lane < first) & (a[y, lane] == background)]
        output[y, before] = payload
        tail = max(x for yy, x in cells if yy == y and footprint[y, x])
        anchors[y] = tail
        for k in range(counts[color]):
            x = tail + 1 + 2 * k
            if x < a.shape[1] and footprint[y, x]:
                scheduled.append((y, x, color))
                if a[y, x] == background:
                    output[y, x] = color
    return output, dict(owner=owner, first_owner=first_owner, anchors=anchors,
                        scheduled=scheduled, counts=dict(counts))


def solve(input_grid: np.ndarray) -> np.ndarray:
    m = _read_input(input_grid)
    out, _ = _paint(m['a'], m['footprint'], m['objects'], m['background'], m['payload'])
    return np.rot90(out, m['rotation']).copy()


def _halo(mask, cells):
    for y, x in cells:
        mask[max(0, y - 1):y + 2, max(0, x - 1):x + 2] = True


def _normalize(cells):
    lo_y, lo_x = min(y for y, x in cells), min(x for y, x in cells)
    return tuple(sorted((y - lo_y, x - lo_x) for y, x in cells))


def _shape(rng, rich=False):
    choice = int(rng.integers(7))
    if choice == 0:
        cells = DIAMOND
    elif choice == 1:
        cells = SQUARE
    elif choice == 2:
        cells = tuple((0, x) for x in range(int(rng.integers(1, 7 if rich else 5))))
    elif choice == 3:
        cells = ((0, 0), (1, 1), (2, 2))
    elif choice == 4:
        cells = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
    else:
        side = 5 if rich else 4
        target = int(rng.integers(1, 11 if rich else 8))
        occupied = {(side // 2, side // 2)}
        while len(occupied) < target:
            frontier = sorted({(y + dy, x + dx) for y, x in occupied
                               for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                               if 0 <= y + dy < side and 0 <= x + dx < side} - occupied)
            occupied.add(frontier[int(rng.integers(len(frontier)))])
        cells = occupied
    if rng.integers(2):
        cells = [(y, -x) for y, x in cells]
    for _ in range(int(rng.integers(4))):
        cells = [(-x, y) for y, x in cells]
    return _normalize(cells)


def _geometry(rng, teaching=False, long_strip=False):
    periods = [int(rng.integers(1, 3 if teaching or long_strip else 4)) for _ in range(2)]
    phases = [int(rng.integers(p)) for p in periods]
    rear_length = int(rng.integers(2, 5 if teaching or long_strip else 6))
    profiles = []
    for _ in range(2):
        profile = [int(rng.integers(2, 5))]
        for _ in range(rear_length - 1):
            profile.append(int(np.clip(profile[-1] + rng.integers(-1, 2), 2, 4)))
        profiles.append(profile)
    flare_length = 2 * max(periods) + 1
    rails = [profile + [profile[-1] + (t + phase) // p for t in range(1, flare_length + 1)]
             for profile, p, phase in zip(profiles, periods, phases)]
    span = max(rails[0]) + max(rails[1]) + 1
    height = int(rng.integers(span + 6, 31))
    center = int(rng.integers(max(rails[0]) + 3, height - max(rails[1]) - 3))
    rear = int(rng.integers(0, 3 if teaching or long_strip else 5))
    mouth = rear + len(rails[0])
    projection_min = 16 if teaching else (13 if long_strip else 6)
    width = int(rng.integers(mouth + projection_min + 1, 31))
    wall = np.zeros((height, width), bool)
    footprint = np.zeros_like(wall)
    wall[center - profiles[0][0]:center + profiles[1][0] + 1, rear] = True
    for j, (up, down) in enumerate(zip(*rails), 1):
        x = rear + j
        wall[center - up, x] = wall[center + down, x] = True
        footprint[center - up + 1:center + down, x] = True
    for x in range(mouth + 1, width):
        t = flare_length + x - mouth
        lo = center - profiles[0][-1] - (t + phases[0]) // periods[0]
        hi = center + profiles[1][-1] + (t + phases[1]) // periods[1]
        footprint[max(0, lo + 1):min(height, hi), x] = True
    chamber = footprint.copy()
    chamber[:, mouth + 1:] = False
    return dict(wallmask=wall, footprint=footprint, chamber=chamber, rear=rear,
                mouth=mouth, center=center, patch_end=rear + rear_length)


def _payload(rng, grid, geometry, color):
    allowed = geometry['chamber'].copy()
    allowed[:, geometry['patch_end'] + 1:] = False
    options = list(map(tuple, np.argwhere(allowed)))
    patch = {options[int(rng.integers(len(options)))]}
    for _ in range(int(rng.integers(0, 6))):
        frontier = sorted({(y + dy, x + dx) for y, x in patch
                           for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
                           if 0 <= y + dy < grid.shape[0] and 0 <= x + dx < grid.shape[1]
                           and allowed[y + dy, x + dx]} - patch)
        if not frontier:
            break
        patch.add(frontier[int(rng.integers(len(frontier)))])
    for y, x in patch:
        grid[y, x] = color


def _place(rng, grid, blocked, geometry, color, shape, objects, relation='any', exact=None):
    shape = _normalize(shape)
    sh, sw = 1 + max(y for y, x in shape), 1 + max(x for y, x in shape)
    nh, nw = grid.shape[0] - sh + 1, grid.shape[1] - sw + 1
    if min(nh, nw) <= 0:
        raise ValueError("Object does not fit")
    available = ~(blocked | geometry['chamber'])
    valid = np.ones((nh, nw), bool)
    inside = np.zeros((nh, nw), int)
    for y, x in shape:
        valid &= available[y:y + nh, x:x + nw]
        inside += geometry['footprint'][y:y + nh, x:x + nw]
    if relation == 'outside':
        valid &= inside == 0
    elif relation in ('inside', 'early'):
        valid &= inside == len(shape)
    elif relation == 'straddle':
        valid &= (inside > 0) & (inside < len(shape))
    if relation in ('inside', 'early', 'straddle'):
        valid[:, :geometry['mouth'] + 1] = False
    if relation == 'early':
        valid[:, geometry['mouth'] + 4:] = False
    if exact is None:
        options = np.argwhere(valid)
        if not len(options):
            raise ValueError("No separated object placement")
        y0, x0 = map(int, options[int(rng.integers(len(options)))])
    else:
        y0, x0 = exact
        if not (0 <= y0 < nh and 0 <= x0 < nw and valid[y0, x0]):
            raise ValueError("Witness placement unavailable")
    cells = frozenset((y0 + y, x0 + x) for y, x in shape)
    for y, x in cells:
        grid[y, x] = color
    _halo(blocked, cells)
    objects.append((color, cells))


def _construct_pair(rng, palette, rotation, kind, regime):
    teaching = kind in ('census', 'occlusion')
    count_transfer = kind == 'test' and regime == 0
    g = _geometry(rng, teaching, count_transfer)
    bg, wall, payload, c, d = map(int, palette[:5])
    a = np.full(g['wallmask'].shape, bg, np.int8)
    a[g['wallmask']] = wall
    _payload(rng, a, g, payload)
    blocked = np.zeros(a.shape, bool)
    _halo(blocked, np.argwhere(a != bg))
    objects = []
    put = lambda color, shape, relation='any', exact=None: _place(
        rng, a, blocked, g, color, shape, objects, relation, exact)
    y, x = g['center'], g['mouth']
    if kind == 'census':
        put(c, DIAMOND, 'inside', (y - 1, x + 2))
        put(c, SQUARE, 'outside')
        put(c, ((0, 0),), 'outside')
        for _ in range(int(rng.integers(1, 4))):
            put(d, _shape(rng), 'outside')
    elif kind == 'occlusion':
        put(c, ((0, 0), (1, 0)), 'inside', (y, x + 2))
        put(c, ((0, 0),), 'inside', (y, x + 8))
        put(d, ((0, 0), (1, 1)), 'inside', (y + 1, x + 5))
        put(c, _shape(rng), 'outside')
        put(d, _shape(rng), 'outside')
        if rng.integers(2):
            put(int(palette[5]), _shape(rng), 'outside')
    else:
        ncolors = int(rng.integers(1, 5))
        colors = list(map(int, palette[3:3 + ncolors]))
        counts = [int(rng.integers(1, 6 if kind == 'test' else 5)) for _ in colors]
        if count_transfer:
            counts[0] = 5
        if kind == 'test' and regime == 2:
            ncolors = max(2, ncolors)
            colors = list(map(int, palette[3:3 + ncolors]))
            counts = [int(rng.integers(1, 5)) for _ in colors]
        while sum(counts) > 16:
            eligible = [i for i, n in enumerate(counts) if n > 1 and not (count_transfer and i == 0)]
            counts[int(rng.choice(eligible))] -= 1
        put(colors[0], DIAMOND if count_transfer else _shape(rng, kind == 'test'), 'early')
        for i, (color, count) in enumerate(zip(colors, counts)):
            for j in range(1 if i == 0 else 0, count):
                if kind == 'test' and regime == 2 and i == 1 and j == 0:
                    put(color, ((0, 0), (1, 0), (2, 0)), 'straddle')
                else:
                    relation = 'outside' if count_transfer else str(rng.choice(
                        ['any', 'inside', 'outside'], p=[.45, .30, .25]))
                    put(color, _shape(rng, kind == 'test'), relation)
    out, _ = _paint(a, g['footprint'], objects, bg, payload)
    inp = np.rot90(a, rotation).copy()
    output = np.rot90(out, rotation).copy()
    meta = _read_input(inp)
    if (not np.array_equal(meta['a'], a)
            or not np.array_equal(meta['footprint'], g['footprint'])
            or set(meta['objects']) != set(objects)):
        raise ValueError("Input reconstruction changed geometry or object identity")
    predicted, info = _paint(meta['a'], meta['footprint'], meta['objects'], bg, payload)
    if not np.array_equal(np.rot90(predicted, rotation), output):
        raise ValueError("Geometry reconstruction disagrees")
    if count_transfer and not _complete_strip(meta, info, 5):
        raise ValueError("Count-transfer test needs a complete five-mark strip")
    return {'input': inp, 'output': output}


def _check_integer(value, name, low, high=None):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < low or (high is not None and value > high)):
        raise ValueError(name + " is outside its legal integer range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _check_integer(seed, 'seed', 0)
    num_train = _check_integer(num_train, 'num_train', 2, 10)
    num_test = _check_integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    shared_palette = rng.random() < .20
    shared_direction = rng.random() < .25
    preferred = int(rng.integers(4))
    test_direction = (preferred + int(rng.integers(1, 4))) % 4
    palette = rng.permutation(10)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', num_train), ('test', num_test)):
        for index in range(n):
            kind = ('census' if index == 0 else 'occlusion' if index == 1 else 'extra') if split == 'train' else 'test'
            colors = palette if shared_palette else rng.permutation(10)
            if regime == 1:
                rotation = preferred if split == 'train' else test_direction
            else:
                rotation = preferred if shared_direction else int(rng.integers(4))
            for _ in range(100):
                try:
                    pair = _construct_pair(rng, colors, rotation, kind, regime)
                except (ValueError, AssertionError):
                    continue
                key = _key(pair['input'])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("Bounded scene construction exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _complete_strip(m, info, minimum=3):
    a, fp = m['a'], m['footprint']
    for y, tail in info['anchors'].items():
        i = int(info['first_owner'][y])
        color = m['objects'][i][0]
        n = info['counts'][color]
        positions = [tail + 1 + 2 * k for k in range(n)]
        if n >= minimum and all(x < a.shape[1] and fp[y, x] and a[y, x] == m['background'] for x in positions):
            return True
    return False


def _features(m, out, info):
    a, fp, bg = m['a'], m['footprint'], m['background']
    objects = m['objects']
    active = Counter(color for color, cells in objects if any(fp[y, x] for y, x in cells))
    areas = Counter()
    four = Counter()
    for color, cells in objects:
        areas[color] += len(cells)
    for color in areas:
        four[color] = len(_components(a == color, False))
    alternatives = {}
    without_owner = {c: n - 1 for c, n in info['counts'].items()}
    for name, counts in (('one', {c: 1 for c in areas}), ('active', active), ('area', areas),
                         ('four', four), ('without_owner', without_owner)):
        alternative, _ = _paint(a, fp, objects, bg, m['payload'], counts)
        alternatives[name] = not np.array_equal(out, alternative)
    unoccluded = a.copy()
    unoccluded[fp & (a == bg)] = m['payload']
    alternatives['no_opacity'] = not np.array_equal(out, unoccluded)
    partial, hidden, straddle = False, False, False
    for i, (_, cells) in enumerate(objects):
        hits = [(y, x) for y, x in cells if fp[y, x]]
        owned = [q for q in hits if info['first_owner'][q[0]] == i]
        partial |= bool(owned) and len(owned) < len(hits)
        hidden |= bool(hits) and not owned
        straddle |= bool(hits) and len(hits) < len(cells)
    wallpoints = np.argwhere(a == m['wall'])
    crop = (a == m['wall'])[wallpoints[:, 0].min():wallpoints[:, 0].max() + 1,
                            wallpoints[:, 1].min():wallpoints[:, 1].max() + 1]
    widths = [hi - lo for lo, hi in m['bounds'][:m['mouth'] - m['rear']]]
    return dict(alternatives=alternatives, partial=partial, hidden=hidden, straddle=straddle,
                covered=any(a[y, x] != bg for y, x, _ in info['scheduled']),
                complete=_complete_strip(m, info), shape=a.shape, direction=m['rotation'],
                palette=(bg, m['wall'], m['payload']), counts=tuple(sorted(info['counts'].values())),
                periods=tuple(f[0] for f in m['fits']), phases=tuple(f[1] for f in m['fits']),
                outline=_key(crop.astype(np.int8)), glyphs=tuple(_normalize(s) for _, s in objects),
                nonmonotone=any(b < a for a, b in zip(widths, widths[1:])),
                asymmetric=len({lo + hi for lo, hi in m['bounds'][:m['mouth'] - m['rear']]}) > 1,
                clipping=any(tail + 2 * info['counts'][objects[int(info['first_owner'][y])][0]] - 1 >= a.shape[1]
                             for y, tail in info['anchors'].items()))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, output = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == output.shape and _key(inp) not in seen
            seen.add(_key(inp))
            m = _read_input(inp)
            a = m['a']
            owner = np.full(a.shape, -1, int)
            for i, (_, cells) in enumerate(m['objects']):
                for y, x in cells:
                    owner[y, x] = i
                assert not any(m['chamber'][y, x] for y, x in cells)
            for y, x in np.argwhere(owner >= 0):
                neighborhood = owner[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
                assert np.all((neighborhood < 0) | (neighborhood == owner[y, x]))
                sub = a[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
                assert not np.any((sub == m['wall']) | (sub == m['payload']))
            expected, info = _paint(a, m['footprint'], m['objects'], m['background'], m['payload'])
            assert np.array_equal(np.rot90(expected, m['rotation']), output)
            assert np.array_equal(inp[inp != m['background']], output[inp != m['background']])
            assert np.any(inp != output)
            canonical_output = np.rot90(output, -m['rotation'])
            assert np.array_equal(a[~m['footprint']], canonical_output[~m['footprint']])
            if split == 'train':
                training.append(_features(m, expected, info))
    for name in ('one', 'active', 'area', 'four', 'without_owner', 'no_opacity'):
        assert any(f['alternatives'][name] for f in training), 'Missing distinguishing witness: ' + name
    assert any(f['complete'] for f in training)
    assert any(f['partial'] and f['hidden'] and f['covered'] for f in training)


def self_test():
    # A small hand-written scene: independent explicit rail phase expectations.
    for p, phase in ((1, 0), (2, 0), (2, 1), (3, 0), (3, 2)):
        length = 2 * p + 2
        a = np.zeros((25, length + 14), np.int8)
        center, half = 12, 2
        a[center - half:center + half + 1, 0] = 1
        for x in range(1, length + 1):
            h = half + (x + phase) // p
            a[center - h, x] = a[center + h, x] = 1
        a[center, 1] = 2
        # Two singleton category objects; the later one is fully shadowed.
        a[center, length + 2] = a[center, length + 8] = 3
        predicted = solve(a)
        next_h = half + (length + 1 + phase) // p
        assert predicted[center - next_h, length + 1] == 0
        assert predicted[center - next_h + 1, length + 1] == 2
        assert predicted[center, length + 3] == predicted[center, length + 5] == 3
        assert predicted[center, length + 9] == 0  # Later object does not restart.
        for r in range(4):
            assert np.array_equal(solve(np.rot90(a, r)), np.rot90(predicted, r))
        for flip in (np.flipud, np.fliplr):
            assert np.array_equal(solve(flip(a)), flip(predicted))
        # One component straddles a virtual rail. Only the inside row shadows.
        # A different, later component covers a tally slot without moving it.
        b = a.copy()
        x = length + 2
        lo = center - half - (x + phase) // p
        b[lo:lo + 2, x] = 4
        b[center, length + 5] = 5
        stamped = solve(b)
        assert stamped[lo, x + p] == 2  # Outside half cannot start a shadow.
        assert stamped[lo + 1, x + 1] == 4
        assert stamped[center, length + 3] == 3
        assert stamped[center, length + 5] == 5
        assert stamped[center, length + 7] == stamped[center, length + 9] == 0
    records, keys, seed_zero = [], set(), None
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        if seed == 0:
            seed_zero = _puzzle_key(puzzle)
        keys.add(_puzzle_key(puzzle))
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                m = _read_input(pair['input'])
                out, info = _paint(m['a'], m['footprint'], m['objects'], m['background'], m['payload'])
                records.append((split, _features(m, out, info)))
    assert len(keys) == 42
    assert _puzzle_key(generate_puzzle(0)) == seed_zero
    features = [f for _, f in records]
    assert len({f['shape'] for f in features}) >= 35
    assert len({f['outline'] for f in features}) >= 80
    assert len({s for f in features for s in f['glyphs']}) >= 60
    assert {p for f in features for p in f['periods']} == {1, 2, 3}
    assert all({f['palette'][i] for f in features} == set(range(10)) for i in range(3))
    assert {n for f in features for n in f['counts']} >= {1, 2, 3, 4, 5}
    for split in ('train', 'test'):
        assert {f['direction'] for s, f in records if s == split} == set(range(4))
    for flag in ('partial', 'hidden', 'covered', 'straddle', 'clipping', 'nonmonotone', 'asymmetric'):
        assert any(f[flag] for f in features), flag
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(17 + nt * 10 + nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted')
    good = generate_puzzle(0, 2, 1)
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in good[s]] for s in ('train', 'test')}
    pair = damaged['test'][0]
    y, x = np.argwhere(pair['input'] != pair['output'])[0]
    pair['output'][y, x] = pair['input'][y, x]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return
    raise AssertionError('Corrupted label accepted')
