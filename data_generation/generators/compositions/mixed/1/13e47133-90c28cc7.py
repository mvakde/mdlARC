"""Keyed contour expansion followed by independent full-vector run collapse.

Only standard library and NumPy are used. No parent code or artifacts are loaded.
Generation and all tests are in memory; this module writes no files.
"""
from collections import deque
import hashlib
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc)
VERIFICATION_SEEDS = (0, 811982374, 284504537, 498613345, 531616887,
    692217935, 510548454, 596039523, 782895336, 4456065, 924281418,
    358186426, 674028771, 483432360, 957486135, 567337032, 101136388,
    599832970, 764829938, 789838683, 185649238, 606252597, 673618897,
    29363248, 111006190, 739916004, 723498967, 254694557, 399945872,
    489298690, 727104126, 712132808, 596006811, 249743607, 189554575,
    270252755, 187990198, 819734042, 228808575, 91920228, 30021745, 486315935)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _check_counts(train, test):
    if not _integer(train) or not 2 <= train <= 10:
        raise ValueError('num_train must be an integer in 2..10')
    if not _integer(test) or not 1 <= test <= 4:
        raise ValueError('num_test must be an integer in 1..4')
    if train <= test:
        raise ValueError('num_train must exceed num_test')


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError('expected a 1..30 by 1..30 integer color grid')
    return a.astype(np.int8, copy=True)


def _signature(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        part = np.zeros(mask.shape, dtype=bool)
        while q:
            r, c = q.popleft()
            part[r, c] = True
            for dr, dc in D4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        result.append(part)
    return result


def _erosion(mask, diagonal=True):
    """Input-only solver: successive erosions, exterior counted as outside."""
    h, w = mask.shape
    remaining = mask.copy()
    depth = np.full(mask.shape, -1, dtype=np.int16)
    d = 0
    while remaining.any():
        padded = np.pad(remaining, 1)
        inside = remaining.copy()
        for dr, dc in D8 if diagonal else D4:
            inside &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        depth[remaining & ~inside] = d
        remaining = inside
        d += 1
    return depth


def _bfs_depth(mask):
    """Separate construction path: eight-neighbor distance from region boundary."""
    h, w = mask.shape
    depth = np.full(mask.shape, -1, dtype=np.int16)
    q = deque()
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if any(not (0 <= r + dr < h and 0 <= c + dc < w and mask[r + dr, c + dc])
               for dr, dc in D8):
            depth[r, c] = 0
            q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in D8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and depth[nr, nc] == -1:
                depth[nr, nc] = depth[r, c] + 1
                q.append((nr, nc))
    return depth


def _inside(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _paths(mask, depth):
    paths = []
    for rr, cc in np.argwhere(mask & (depth == 0)):
        r, c = int(rr), int(cc)
        for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            if (_inside(mask, r - dr, c) or _inside(mask, r, c - dc)
                    or not all(_inside(mask, nr, nc) for nr, nc in
                               ((r + dr, c), (r, c + dc), (r + dr, c + dc)))):
                continue
            length = 0
            while (_inside(mask, r + length * dr, c + length * dc)
                   and depth[r + length * dr, c + length * dc] == length):
                length += 1
            paths.append((r, c, dr, dc, length))
    return paths


def _roles(a):
    counts = np.bincount(a.ravel(), minlength=10)
    order = np.argsort(-counts, kind='stable')
    neutral, wall = map(int, order[:2])
    if not counts[neutral] > counts[wall] > counts[order[2]]:
        raise ValueError('neutral and divider must have strictly ordered frequencies')
    return neutral, wall


def _decode(a, mask, depth, neutral):
    cues = [(int(r), int(c)) for r, c in np.argwhere(mask & (a != neutral))]
    if not cues:
        return (neutral,), ()
    length = max(int(depth[r, c]) for r, c in cues) + 1
    readings = []
    for r, c, dr, dc, capacity in _paths(mask, depth):
        if capacity >= length and all((rr, cc) == (r + int(depth[rr, cc]) * dr,
                                                    c + int(depth[rr, cc]) * dc)
                                      for rr, cc in cues):
            period = tuple(int(a[r + i * dr, c + i * dc]) for i in range(length))
            readings.append((period, (dr, dc)))
    periods = {p for p, _ in readings}
    if len(periods) != 1:
        raise ValueError('a region key must determine one inward color period')
    return next(iter(periods)), tuple(sorted({direction for _, direction in readings}))


def _collapse(a):
    rows = np.r_[0, 1 + np.flatnonzero(np.any(a[1:] != a[:-1], axis=1))]
    cols = np.r_[0, 1 + np.flatnonzero(np.any(a[:, 1:] != a[:, :-1], axis=0))]
    out = a[np.ix_(rows, cols)].copy()
    return out, np.diff(np.r_[rows, a.shape[0]]), np.diff(np.r_[cols, a.shape[1]]), rows, cols


def _separated_recurrence(a):
    # Applied AFTER maximal runs have been collapsed, so any remaining equal
    # vectors are necessarily separated and must both survive.
    return (len({row.tobytes() for row in a}) < len(a),
            len({col.tobytes() for col in a.T}) < a.shape[1])


def _local_seam(a, wall):
    for work in (a, a.T):
        for first, second in zip(work[:-1], work[1:]):
            content = (first != wall) & (second != wall)
            if np.any(content & (first == second)) and np.any(content & (first != second)):
                return True
    return False


def _expand_from_input(a):
    neutral, wall = _roles(a)
    wall_mask = a == wall
    if len(_components(wall_mask)) != 1:
        raise ValueError('divider must be connected')
    if np.any(wall_mask[:-1, :-1] & wall_mask[1:, :-1] & wall_mask[:-1, 1:] & wall_mask[1:, 1:]):
        raise ValueError('divider must be one cell wide')
    masks = _components(~wall_mask)
    if len(masks) < 2:
        raise ValueError('divider must separate at least two regions')
    expanded = a.copy()
    info = []
    for mask in masks:
        depth = _erosion(mask)
        period, directions = _decode(a, mask, depth, neutral)
        expanded[mask] = np.array(period, dtype=np.int8)[depth[mask] % len(period)]
        rr, cc = np.nonzero(mask)
        concave = int(mask.sum()) != (int(rr.max() - rr.min()) + 1) * (int(cc.max() - cc.min()) + 1)
        info.append({'mask': mask, 'depth': depth, 'period': period,
                     'directions': directions, 'concave': concave})
    return expanded, neutral, wall, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fill contours at physical resolution; then collapse maximal full-vector runs."""
    expanded, _, _, _ = _expand_from_input(_grid(input_grid))
    return _collapse(expanded)[0]


def _partition(rng, shape, target):
    """Composed guillotine cuts, with one protected large room for a full cycle."""
    h, w = shape
    wall = np.zeros(shape, dtype=bool)
    vertical = bool(rng.integers(2))
    extent = w if vertical else h
    cut = int(rng.integers(7, extent - 6))
    if vertical:
        wall[:, cut] = True
        rooms = [(0, h - 1, 0, cut - 1), (0, h - 1, cut + 1, w - 1)]
    else:
        wall[cut] = True
        rooms = [(0, cut - 1, 0, w - 1), (cut + 1, h - 1, 0, w - 1)]
    # Protect one side; recursively divide the other side in either direction.
    reserved = int(rng.integers(2))
    protected = rooms[reserved]
    while len(rooms) < target:
        options = []
        for i, (a, b, c, d) in enumerate(rooms):
            if (a, b, c, d) == protected:
                continue
            if b - a + 1 >= 11:
                options.append((i, 0))
            if d - c + 1 >= 11:
                options.append((i, 1))
        if not options:
            break
        i, axis = options[int(rng.integers(len(options)))]
        a, b, c, d = rooms.pop(i)
        if axis == 0:
            q = int(rng.integers(a + 5, b - 4))
            wall[q, max(0, c - 1):min(w, d + 2)] = True
            rooms.extend(((a, q - 1, c, d), (q + 1, b, c, d)))
        else:
            q = int(rng.integers(c + 5, d - 4))
            wall[max(0, a - 1):min(h, b + 2), q] = True
            rooms.extend(((a, b, c, q - 1), (a, b, q + 1, d)))
    return wall, rooms


def _box(rng, wall, rooms):
    """Attach a U to a wall, creating an enclosed room and an indented neighbor."""
    options = []
    for a, b, c, d in rooms:
        for transpose in (False, True):
            aa, bb, cc, dd = (c, d, a, b) if transpose else (a, b, c, d)
            work = wall.T if transpose else wall
            if bb - aa + 1 < 13 or dd - cc + 1 < 9:
                continue
            for sign in (-1, 1):
                edge = cc - 1 if sign == 1 else dd + 1
                if 0 <= edge < work.shape[1] and work[aa:bb + 1, edge].all():
                    options.append((transpose, aa, bb, cc, dd, sign, edge))
    if not options:
        return False
    transpose, a, b, c, d, sign, edge = options[int(rng.integers(len(options)))]
    work = wall.T if transpose else wall
    top = int(rng.integers(max(a + 2, 3), b - 8 + 1))
    bottom = int(rng.integers(top + 6, min(b - 2, work.shape[0] - 4) + 1))
    width = int(rng.integers(5, d - c - 2 + 1))
    far = edge + sign * (width + 1)
    lo, hi = sorted((edge, far))
    work[top, lo:hi + 1] = True
    work[bottom, lo:hi + 1] = True
    work[top:bottom + 1, far] = True
    return True


def _step(rng, wall, budget):
    for _ in range(budget):
        for _attempt in range(40):
            work = wall.T if rng.integers(2) else wall
            points = np.argwhere(work)
            r, c = map(int, points[int(rng.integers(len(points)))])
            end = r + int(rng.integers(4, 10))
            shift = int(rng.choice((-2, -1, 1, 2)))
            lo, hi = sorted((c, c + shift))
            if r < 3 or end >= work.shape[0] - 3 or lo < 3 or hi >= work.shape[1] - 3:
                continue
            block = work[r - 1:end + 2, lo - 1:hi + 2]
            expected = np.zeros(block.shape, dtype=bool)
            expected[:, c - lo + 1] = True
            if not np.array_equal(block, expected):
                continue
            work[r + 1:end, c] = False
            work[r, lo:hi + 1] = True
            work[end, lo:hi + 1] = True
            work[r:end + 1, c + shift] = True
            break


def _touches_edge(mask):
    return any((mask[0].any(), mask[-1].any(), mask[:, 0].any(), mask[:, -1].any()))


def _period(rng, neutral, wall, capacity, boundary, force_cycle=False):
    colors = [int(c) for c in rng.permutation([c for c in range(10) if c not in (neutral, wall)])]
    if force_cycle:
        return (colors[0], colors[0], colors[1])
    if rng.random() < .19:
        return (neutral,)
    if capacity < 3 or rng.random() < .20:
        return (colors[0],)
    if boundary:
        length = int(rng.integers(3, min(4, capacity) + 1))
        # Equal adjacent depths form actual run-length bands; use different
        # thicknesses, colors and final entries in different regions.
        if length == 3:
            period = [colors[0], colors[0], colors[1]]
        elif rng.random() < .5:
            period = [colors[0], colors[0], colors[0], colors[1]]
        else:
            period = [colors[0], colors[0], colors[1], colors[2]]
        if rng.random() < .13:
            period[0] = period[1] = neutral
        return tuple(period)
    length = int(rng.integers(2, min(4, capacity) + 1))
    period = colors[:length]
    if rng.random() < .35:
        period[int(rng.integers(length - 1))] = neutral
    elif rng.random() < .3:
        period[1] = period[0]
    return tuple(period)


def _make_pair(rng, shape, neutral, wall_color, mode, target):
    wall, rooms = _partition(rng, shape, target)
    if mode != 'rectangular':
        _box(rng, wall, rooms)
        _step(rng, wall, 2 if mode == 'concave' else 4)
    if rng.integers(2):
        wall = wall[::-1].copy()
    if rng.integers(2):
        wall = wall[:, ::-1].copy()
    masks = _components(~wall)
    depths = [_bfs_depth(mask) for mask in masks]
    choices = [_paths(mask, depth) for mask, depth in zip(masks, depths)]
    if not all(choices):
        raise ValueError('region has no diagonal corner')
    capacities = [max(path[4] for path in ps) for ps in choices]
    cycle_options = [i for i in range(len(masks)) if capacities[i] >= 4]
    if not cycle_options:
        raise ValueError('no region can demonstrate a complete repeated cycle')
    # In the concave demonstration, prioritize a deep colored concavity.
    concave_options = [i for i in cycle_options if int(masks[i].sum()) <
                       (np.ptp(np.nonzero(masks[i])[0]) + 1) *
                       (np.ptp(np.nonzero(masks[i])[1]) + 1)]
    options = concave_options if mode == 'concave' and concave_options else cycle_options
    cycle_index = int(rng.choice(options))
    # The two core demonstrations teach neutral entries as well as the two
    # parent mechanisms, even when only two training pairs are requested.
    specifications = {}
    if mode == 'rectangular':
        options = [i for i in range(len(masks)) if i != cycle_index and capacities[i] >= 3]
        if not options:
            raise ValueError('leading-neutral demonstration needs a second deep room')
        i = int(rng.choice(options))
        color = int(rng.choice([c for c in range(10) if c not in (neutral, wall_color)]))
        specifications[i] = (neutral, neutral, color)
    elif mode == 'concave':
        options = [i for i in range(len(masks)) if i != cycle_index
                   and not _touches_edge(masks[i]) and capacities[i] >= 3]
        if not options:
            raise ValueError('internal-neutral demonstration needs an enclosed room')
        i = int(rng.choice(options))
        colors = rng.choice([c for c in range(10) if c not in (neutral, wall_color)],
                            size=2, replace=False)
        specifications[i] = (int(colors[0]), neutral, int(colors[1]))
        options = [j for j in range(len(masks)) if j not in (cycle_index, i)]
        if not options:
            raise ValueError('unmarked demonstration needs a third room')
        specifications[int(rng.choice(options))] = (neutral,)
    inp = np.full(shape, neutral, dtype=np.int8)
    expanded = inp.copy()
    inp[wall] = expanded[wall] = wall_color
    for i, (mask, depth, paths, capacity) in enumerate(zip(masks, depths, choices, capacities)):
        period = specifications.get(i)
        if period is None:
            period = _period(rng, neutral, wall_color, capacity, _touches_edge(mask), i == cycle_index)
        if period != (neutral,):
            paths = [p for p in paths if p[4] >= len(period)]
            r, c, dr, dc, _ = paths[int(rng.integers(len(paths)))]
            for d, color in enumerate(period):
                inp[r + dr * d, c + dc * d] = color
        expanded[mask] = np.array(period, dtype=np.int8)[depth[mask] % len(period)]
    if _roles(inp) != (neutral, wall_color):
        raise ValueError('insufficient color-frequency separation')
    return {'input': inp, 'output': _collapse(expanded)[0]}


def _concavity_witness(expanded, wall, regions):
    """After B, a genuine contour must differ from bbox and Manhattan shortcuts."""
    actual = _collapse(expanded)[0]
    for info in regions:
        if not info['concave'] or len(set(info['period'])) < 2:
            continue
        mask, period = info['mask'], np.array(info['period'], dtype=np.int8)
        rr, cc = np.nonzero(mask)
        box = np.minimum.reduce((rr - rr.min(), rr.max() - rr, cc - cc.min(), cc.max() - cc))
        wrong_box = expanded.copy()
        wrong_l1 = expanded.copy()
        wrong_box[mask] = period[box % len(period)]
        wrong_l1[mask] = period[_erosion(mask, diagonal=False)[mask] % len(period)]
        if (not np.array_equal(actual, _collapse(wrong_box)[0])
                and not np.array_equal(actual, _collapse(wrong_l1)[0])):
            return True
    return False


def _analyze(inp, out):
    expanded, neutral, wall, regions = _expand_from_input(inp)
    predicted, rows, cols, ri, ci = _collapse(expanded)
    assert np.array_equal(predicted, out), 'output disagrees with contour-then-collapse rule'
    assert max(rows) > 1 and max(cols) > 1, 'both axes must actually collapse'
    assert out.shape[0] < inp.shape[0] and out.shape[1] < inp.shape[1]
    assert np.array_equal(np.repeat(np.repeat(out, rows, 0), cols, 1), expanded)
    assert not np.any(np.all(out[1:] == out[:-1], axis=1))
    assert not np.any(np.all(out[:, 1:] == out[:, :-1], axis=0))
    # Independent row/column scan confirms the vectorized grouping.
    kept_r = [i for i in range(len(expanded)) if i == 0 or not np.array_equal(expanded[i], expanded[i - 1])]
    kept_c = [i for i in range(expanded.shape[1]) if i == 0 or not np.array_equal(expanded[:, i], expanded[:, i - 1])]
    assert kept_r == ri.tolist() and kept_c == ci.tolist()
    assert not np.array_equal(_collapse(inp)[0], out), 'A must affect the final output'
    assert not np.array_equal(expanded, out), 'B must affect the final output'
    # Divider topology is preserved under precisely the SAME quotient map.
    logical_wall = out == wall
    assert np.array_equal(np.repeat(np.repeat(logical_wall, rows, 0), cols, 1), inp == wall)
    assert len(_components(logical_wall)) == 1
    assert len(_components(~logical_wall)) == len(regions)
    # Every surviving color was present in this input; there is no palette code.
    assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
    cyclic = False
    wrong_once = expanded.copy()
    for info in regions:
        mask, depth, period = info['mask'], info['depth'], np.array(info['period'], dtype=np.int8)
        assert np.array_equal(depth, _bfs_depth(mask))
        if len(set(period)) > 1 and depth.max() >= len(period):
            cyclic = True
        wrong_once[mask] = period[np.minimum(depth[mask], len(period) - 1)]
    assert cyclic
    assert not np.array_equal(out, _collapse(wrong_once)[0]), 'cycle repetition must be observable after collapse'
    return {'expanded': expanded, 'neutral': neutral, 'wall': wall, 'regions': regions,
            'rows': tuple(map(int, rows)), 'cols': tuple(map(int, cols)),
            'row_starts': ri, 'col_starts': ci,
            'separated_recurrence': _separated_recurrence(out),
            'local_seam': _local_seam(out, wall),
            'concavity_witness': _concavity_witness(expanded, wall, regions)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    _check_counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle = {'train': [], 'test': []}
    used = set()
    # The two guaranteed demonstrations may occur in either order.
    modes = ['rectangular', 'concave']
    rng.shuffle(modes)
    modes += ['free'] * (int(num_train) - 2)
    role_order = [int(c) for c in rng.permutation(10)]
    offset = int(rng.integers(1, 10))
    for index in range(int(num_train) + int(num_test)):
        test = index >= num_train
        mode = 'free' if test else modes[index]
        neutral = role_order[index % 10]
        wall = role_order[(index + offset) % 10]
        accepted = False
        for attempt in range(96):
            low = 22 if test else (21 if mode == 'concave' else 16)
            h, w = map(int, rng.integers(low, 31, size=2))
            if test and rng.random() < .16:
                h = w = 30
            target = int(rng.integers(3 if test else 2, 7 if test else 5))
            try:
                pair = _make_pair(rng, (h, w), neutral, wall, mode, target)
                signature = _signature(pair['input'])
                if signature in used:
                    continue
                analysis = _analyze(pair['input'], pair['output'])
                if mode == 'concave' and not analysis['concavity_witness']:
                    continue
                # A fixed scale or tied row/column factors must fail in at least
                # one of the two core demonstrations.
                if mode == 'rectangular' and (max(analysis['rows']) == max(analysis['cols'])
                        or not all(analysis['separated_recurrence'])
                        or not analysis['local_seam']):
                    continue
            except (ValueError, AssertionError):
                continue
            puzzle['test' if test else 'train'].append(pair)
            used.add(signature)
            accepted = True
            break
        if not accepted:
            raise RuntimeError('bounded construction exhausted; no silent rule fallback')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen = set()
    train = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            sig = _signature(inp)
            assert sig not in seen, 'duplicate input'
            seen.add(sig)
            analysis = _analyze(inp, out)
            if split == 'train':
                train.append(analysis)
    assert any(a['concavity_witness'] for a in train), 'training must expose true contour geometry'
    assert any(max(a['rows']) != max(a['cols']) and all(a['separated_recurrence'])
               and a['local_seam'] for a in train)
    assert len({a['neutral'] for a in train}) >= 2
    assert len({a['wall'] for a in train}) >= 2
    periods = [(a['neutral'], region['period']) for a in train for region in a['regions']]
    assert any(p == (n,) for n, p in periods), 'training must expose unmarked neutral regions'
    assert any(len(p) > 1 and p[0] == n for n, p in periods), 'training must expose leading neutral entries'
    assert any(n in p[1:-1] for n, p in periods), 'training must expose internal neutral entries'


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 specified seeds, bounded memory, no retained puzzles or disk output."""
    # A hand-computed witness, independent of sampled output construction:
    # left 9x8 room has period (2,2,3); right 9x8 room has period (4,).
    manual = np.zeros((9, 17), dtype=np.int8)
    manual[:, 8] = 1
    manual[0, 0] = manual[1, 1] = 2
    manual[2, 2] = 3
    manual[0, 9] = 4
    expected = np.array([[int(c) for c in row] for row in
                         ('2222214', '2333214', '2323214', '2333214', '2222214')], dtype=np.int8)
    assert np.array_equal(solve(manual), expected)
    witness = _analyze(manual, expected)
    assert witness['rows'] == (2, 1, 3, 1, 2)
    assert witness['cols'] == (2, 1, 2, 1, 2, 1, 8)
    assert all(witness['separated_recurrence'])
    shapes, geometries, outputs, profiles, counts, roles, periods = (set() for _ in range(7))
    digests = set()
    directions = set()
    full = 0
    for seed in VERIFICATION_SEEDS:
        puzzle = generate_puzzle(seed)
        digests.add(_puzzle_digest(puzzle))
        assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                a = _analyze(inp, out)
                assert np.array_equal(solve(inp), out)
                shapes.add(inp.shape)
                geometries.add(hashlib.sha256((inp == a['wall']).tobytes() + bytes(inp.shape)).hexdigest())
                outputs.add(out.shape)
                profiles.add((a['rows'], a['cols']))
                counts.add(len(a['regions']))
                roles.add((a['neutral'], a['wall']))
                full += inp.shape == (30, 30)
                for region in a['regions']:
                    periods.add(len(region['period']))
                    directions.update(region['directions'])
    assert len(digests) == 42
    assert len(shapes) >= 45 and len(geometries) >= 100
    assert len(outputs) >= 35 and len(profiles) >= 100
    assert len(counts) >= 4 and len(roles) >= 50
    assert {1, 3, 4} <= periods and len(directions) == 4 and full > 0
    for train in range(2, 11):
        for test in range(1, min(4, train - 1) + 1):
            puzzle = generate_puzzle(1234567, train, test)
            assert len(puzzle['train']) == train and len(puzzle['test']) == test
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                        (True, 1), (3.5, 1), (4, None), (None, 1), (4, False)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Confirm that validation uses returned grids, not generation parameters.
    puzzle = generate_puzzle(5, 2, 1)
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzle[s]] for s in puzzle}
    out = damaged['test'][0]['output']
    out[0, 0] = (int(out[0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError('corrupt output accepted')
