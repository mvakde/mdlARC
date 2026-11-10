"""Dock-to-close chambers: packing makes the addresses for color transfer.

Only standard-library modules and NumPy are used.  The input carries its own
direction key (two colored rails); neither solve nor validation needs a seed.
"""
from collections import Counter, deque
import numpy as np


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask)))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, points = [start], [start]
        while stack:
            r, c = stack.pop()
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in unseen:
                    unseen.remove(p)
                    points.append(p)
                    stack.append(p)
        result.append(np.asarray(points, dtype=int))
    return result


def _box(points):
    return tuple(map(int, points.min(0))) + tuple(map(int, points.max(0) + 1))


def _holes(barrier):
    free = np.pad(~barrier, 1, constant_values=True)
    return [p - 1 for p in _components(free)
            if np.all(p > 0) and np.all(p < np.array(free.shape) - 1)]


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= d <= 30 for d in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((0 <= g) & (g <= 9))
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8, copy=False).tobytes()


def _u_patch(h, w, direction, frame, bg, ink, pins):
    patch = np.full((h, w), bg, dtype=np.int8)
    patch[0] = patch[-1] = frame
    patch[:, -1 if direction == 0 else 0] = frame
    for r, c in pins:
        patch[r, c] = ink
    return patch


def _decode_canonical(g):
    """Find the rail board from its complete caps, then read open frames."""
    bg = int(g[0, 0])
    assert np.all(g[0] == bg) and np.all(g[-1] == bg)
    assert np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)
    candidates = []
    H, W = g.shape
    for r in range(1, H - 4):
        c = 1
        while c < W - 1:
            ink = int(g[r, c])
            C = c + 1
            while C < W - 1 and g[r, C] == ink:
                C += 1
            if ink != bg and C - c >= 7:
                for R in range(r + 4, H):
                    if not np.all(g[R - 1, c:C] == ink):
                        continue
                    left, right = int(g[r + 1, c]), int(g[r + 1, C - 1])
                    if len({bg, ink, left, right}) != 4:
                        continue
                    if (np.all(g[r + 1:R - 1, c] == left)
                            and np.all(g[r + 1:R - 1, C - 1] == right)):
                        board = g[r:R, c:C].copy()
                        if set(map(int, board.flat)) <= {bg, ink, left, right}:
                            candidates.append((board, (r, c, R, C), bg, ink, left, right))
            c = C
    assert len(candidates) == 1, "expected one board with two unlike rails"
    board, bounds, bg, ink, left, right = candidates[0]
    inner = board[1:-1, 1:-1]
    objects = []
    covered = np.zeros(inner.shape, dtype=bool)
    for direction, frame in enumerate((left, right)):
        for points in _components(inner == frame):
            r, c, R, C = _box(points)
            h, w = R - r, C - c
            assert h >= 3 and w >= 3
            patch = inner[r:R, c:C].copy()
            pins = [tuple(map(int, p)) for p in np.argwhere(patch == ink)]
            assert pins and all(0 < y < h - 1 and 0 < x < w - 1 for y, x in pins)
            expected = _u_patch(h, w, direction, frame, bg, ink, pins)
            assert np.array_equal(patch, expected), "not a complete open U frame"
            assert not covered[r:R, c:C].any(), "overlapping rectangular footprints"
            covered[r:R, c:C] = True
            objects.append(dict(r=r + 1, c=c + 1, h=h, w=w, d=direction,
                                patch=patch, pins=pins))
    assert 2 <= len(objects) <= 8
    assert np.all((inner == bg) | covered), "stray board marks"
    outside = g.copy()
    r, c, R, C = bounds
    outside[r:R, c:C] = bg
    points = np.argwhere(outside != bg)
    assert len(points)
    a, b, A, B = _box(points)
    donor = outside[a:A, b:B]
    rooms = _holes(donor == ink)
    assert len(rooms) == len(objects)
    colors, room_mask = [], np.zeros(donor.shape, dtype=bool)
    for room in rooms:
        values = np.unique(donor[tuple(room.T)])
        assert len(values) == 1 and int(values[0]) not in (bg, ink, left, right)
        colors.append(int(values[0]))
        room_mask[tuple(room.T)] = True
    assert np.all((donor == bg) | (donor == ink) | room_mask)
    assert len(set(colors)) >= 2
    assert len(_components(donor != bg)) == 1, "donor must be one connected atlas"
    return dict(grid=g, board=board, bounds=bounds, bg=bg, ink=ink,
                rails=(left, right), objects=objects, donor=donor,
                donor_rooms=rooms, colors=colors)


def _decode(g):
    possibilities = []
    for transpose in (False, True):
        try:
            info = _decode_canonical(g.T if transpose else g)
            info['transpose'] = transpose
            possibilities.append(info)
        except AssertionError:
            pass
    assert len(possibilities) == 1, "board axis is not uniquely visible"
    return possibilities[0]


def _band(a, b):
    return max(a['r'], b['r']) < min(a['r'] + a['h'], b['r'] + b['h'])


def _overlap(a, b, margin=0):
    return (a['r'] < b['r'] + b['h'] + margin
            and b['r'] < a['r'] + a['h'] + margin
            and a['c'] < b['c'] + b['w'] + margin
            and b['c'] < a['c'] + a['w'] + margin)


def _pack(objects, width):
    """Preserve every overlapping predecessor; never search past a blocker."""
    result = [dict(o) for o in objects]
    for d in (0, 1):
        indices = [i for i, o in enumerate(objects) if o['d'] == d]
        indices.sort(key=lambda i: (objects[i]['c'] if d == 0 else
                                   -objects[i]['c'] - objects[i]['w'], objects[i]['r']))
        settled = []
        for i in indices:
            o = result[i]
            blockers = [p for p in settled if _band(o, p)]
            o['c'] = (max((p['c'] + p['w'] for p in blockers), default=1)
                      if d == 0 else
                      min((p['c'] for p in blockers), default=width - 1) - o['w'])
            assert 1 <= o['c'] and o['c'] + o['w'] <= width - 1
            settled.append(o)
    assert not any(_overlap(a, b) for i, a in enumerate(result) for b in result[i + 1:])
    return result


def _render_board(shape, bg, ink, rails, objects):
    b = np.full(shape, bg, dtype=np.int8)
    b[0] = b[-1] = ink
    b[1:-1, 0], b[1:-1, -1] = rails
    for o in objects:
        r, c, h, w = (o[k] for k in ('r', 'c', 'h', 'w'))
        b[r:r+h, c:c+w] = o['patch']
    return b


def _chambers(board, objects, rails):
    """Caps and cargo ink are permeable for enclosure; colored walls are not."""
    rooms = _holes(np.isin(board, rails))
    expected = []
    for o in objects:
        r, c, h, w, d = (o[k] for k in ('r', 'c', 'h', 'w', 'd'))
        columns = range(c, c + w - 1) if d == 0 else range(c + 1, c + w)
        expected.append(np.asarray([(y, x) for y in range(r + 1, r + h - 1)
                                    for x in columns], dtype=int))
    actual_sets = {frozenset(map(tuple, p)) for p in rooms}
    assert actual_sets == {frozenset(map(tuple, p)) for p in expected}, "a mouth did not close"
    return expected


def _relations(rooms):
    boxes = [_box(p) for p in rooms]
    return [[tuple(-1 if a[k + 2] <= b[k] else
                   1 if b[k + 2] <= a[k] else 0 for k in (0, 1))
             for b in boxes] for a in boxes]


def _match_colors(donor_rooms, target_rooms, colors):
    """Exact orientation-preserving spatial graph matching, allowing symmetry
    only when every permitted mapping gives the very same color assignment.
    """
    a, b = _relations(donor_rooms), _relations(target_rooms)
    n = len(a)
    signatures_a = [Counter(row) for row in a]
    signatures_b = [Counter(row) for row in b]
    options = [[j for j in range(n) if signatures_a[i] == signatures_b[j]] for i in range(n)]
    answer, solutions = None, 0

    def search(mapping, used):
        nonlocal answer, solutions
        if len(mapping) == n:
            candidate = [None] * n
            for i, j in mapping.items():
                candidate[j] = colors[i]
            if answer is None:
                answer = candidate
            assert answer == candidate, "ambiguous donor color correspondence"
            solutions += 1
            return
        remaining = {}
        for i in range(n):
            if i not in mapping:
                remaining[i] = [j for j in options[i] if j not in used and
                                all(a[i][k] == b[j][v] for k, v in mapping.items())]
        i = min(remaining, key=lambda k: len(remaining[k]))
        for j in remaining[i]:
            mapping[i] = j
            search(mapping, used | {j})
            del mapping[i]

    search({}, set())
    assert solutions, "donor and docked room arrangements differ"
    return answer


def _analyze(g):
    info = _decode(g)
    settled = _pack(info['objects'], info['board'].shape[1])
    docked = _render_board(info['board'].shape, info['bg'], info['ink'], info['rails'], settled)
    rooms = _chambers(docked, settled, info['rails'])
    colors = _match_colors(info['donor_rooms'], rooms, info['colors'])
    output = docked.copy()
    for room, color in zip(rooms, colors):
        y, x = room.T
        available = output[y, x] == info['bg']
        output[y[available], x[available]] = color
    info.update(settled=settled, docked=docked, rooms=rooms, target_colors=colors,
                output=output)
    return info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from one visible input; no generator state or demonstrations."""
    info = _analyze(_grid(input_grid))
    return (info['output'].T if info['transpose'] else info['output']).copy()


def _compress_plan(plan):
    rows = [0] + [r for r in range(1, len(plan)) if not np.array_equal(plan[r], plan[r - 1])]
    small = plan[rows]
    cols = [0] + [c for c in range(1, small.shape[1])
                  if not np.array_equal(small[:, c], small[:, c - 1])]
    small = small[:, cols]
    r, c, R, C = _box(np.argwhere(small >= 0))
    return small[r:R, c:C]


def _make_donor(rng, plan, bg, ink, colors, max_height, scan_axis=None):
    small = _compress_plan(plan)
    yr = rng.integers(1, 3, size=small.shape[0])
    xr = rng.integers(1, 3, size=small.shape[1])
    while int(yr.sum()) + 2 > max_height and np.any(yr > 1):
        yr[int(rng.choice(np.flatnonzero(yr > 1)))] -= 1
    while int(xr.sum()) + 2 > 28 and np.any(xr > 1):
        xr[int(rng.choice(np.flatnonzero(xr > 1)))] -= 1
    assert int(yr.sum()) + 2 <= max_height
    labels = np.pad(np.repeat(np.repeat(small, yr, axis=0), xr, axis=1), 1,
                    constant_values=-1)
    original_rooms = [np.argwhere(labels == i) for i in range(len(colors))]
    relations = _relations(original_rooms)
    def scan_key(room):
        r, c = _box(room)[:2]
        return (c, r) if scan_axis == 1 else (r, c)

    old_order = sorted(range(len(colors)), key=lambda i: scan_key(original_rooms[i]))
    # Independently retreat local room boundaries while retaining the spatial
    # graph. Overlapping vertical spans may exchange their discovery order;
    # a global coordinate warp alone could never demonstrate that distinction.
    changed = False
    for _ in range(48 if scan_axis is not None else 12):
        i = int(rng.integers(len(colors)))
        r, c, R, C = _box(np.argwhere(labels == i))
        edge = int(rng.integers(4))
        if (edge < 2 and R-r < 2) or (edge >= 2 and C-c < 2):
            continue
        trial = labels.copy()
        if edge == 0:
            trial[r, c:C] = -1
        elif edge == 1:
            trial[R-1, c:C] = -1
        elif edge == 2:
            trial[r:R, c] = -1
        else:
            trial[r:R, C-1] = -1
        trial_rooms = [np.argwhere(trial == j) for j in range(len(colors))]
        if _relations(trial_rooms) != relations:
            continue
        labels = trial
        new_order = sorted(range(len(colors)), key=lambda j: scan_key(trial_rooms[j]))
        changed = [colors[j] for j in new_order] != [colors[j] for j in old_order]
        if changed and scan_axis is not None:
            break
    assert scan_axis is None or changed, 'need a changed room-discovery order'
    occupied = labels >= 0
    walls = np.zeros(labels.shape, dtype=bool)
    pad = np.pad(occupied, 1)
    for dy in range(3):
        for dx in range(3):
            walls |= pad[dy:dy+labels.shape[0], dx:dx+labels.shape[1]]
    walls &= ~occupied
    # Join separated room frameworks by shortest exterior paths. Paths never
    # cross rooms. A visible-hole check below rejects accidental extra pockets.
    for _ in range(len(colors)):
        parts = _components(walls)
        if len(parts) == 1:
            break
        first = set(map(tuple, parts[0]))
        other = set(map(tuple, np.concatenate(parts[1:])))
        queue = deque(sorted(first))
        previous = {p: None for p in first}
        end = None
        while queue and end is None:
            y, x = queue.popleft()
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if (not 0 <= p[0] < labels.shape[0] or not 0 <= p[1] < labels.shape[1]
                        or occupied[p] or p in previous):
                    continue
                previous[p] = (y, x)
                if p in other:
                    end = p
                    break
                queue.append(p)
        assert end is not None
        while end not in first:
            walls[end] = True
            end = previous[end]
    donor = np.full(labels.shape, bg, dtype=np.int8)
    donor[walls] = ink
    for i, color in enumerate(colors):
        donor[labels == i] = color
    assert len(_holes(walls)) == len(colors)
    assert len(_components(donor != bg)) == 1
    return donor


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _layout(rng, purpose):
    if purpose == 'cross':
        counts, split_counts = [2], [(1, 1)]
    elif purpose == 'teach':
        counts = [3, 1]
        split_counts = [(2, 1), (0, 1)]
        if rng.integers(2):
            split_counts = [(b, a) for a, b in split_counts]
    else:
        total = 6 if purpose == 'test' else int(rng.integers(2, 6))
        bands = 2 if total >= 5 or rng.random() < 0.55 else 1
        cut = int(rng.integers(1, total)) if bands == 2 else total
        counts = [cut, total - cut] if bands == 2 else [total]
        split_counts = []
        for count in counts:
            # Depth is limited by the room-height budget, rather than a fixed
            # final-column template. Single-class banks are allowed.
            allowed = [n for n in range(count + 1) if max(n, count - n) <= 3]
            low = _pick(rng, allowed)
            split_counts.append((low, count - low))
    width = int(rng.integers(18, 27))
    objects, row = [], 1 + int(rng.integers(2))
    for bank_index, ((low, high), count) in enumerate(zip(split_counts, counts)):
        depth = max(low, high)
        base_h = int(rng.integers(max(3, depth + 2), max(3, depth + 2) + 3))
        groups = []
        for d, number in enumerate((low, high)):
            group, r, h = [], row, base_h
            for level in range(number):
                if level:
                    shrink = int(rng.integers(0, min(2, h - 3) + 1))
                    r += int(rng.integers(shrink + 1))
                    h -= shrink
                group.append(dict(r=r, h=h, w=int(rng.integers(3, 7)), d=d))
            groups.append(group if d == 0 else list(reversed(group)))
        if purpose == 'cross':
            # Same dimensions and cargo under opposite instructions.
            groups[1][0].update(r=groups[0][0]['r'], h=groups[0][0]['h'],
                                w=groups[0][0]['w'])
            order = groups[1] + groups[0]
        else:
            order = []
            while groups[0] or groups[1]:
                available = [d for d in (0, 1) if groups[d]]
                d = _pick(rng, available)
                order.append(groups[d].pop(0))
        # All initial footprints in a bank are separated; final ones may dock.
        needed = sum(o['w'] for o in order) + count - 1
        if needed > width - 2:
            for o in order:
                o['w'] = 3
            needed = 4 * count - 1
        assert needed <= width - 2
        gaps = rng.multinomial(width - 2 - needed, np.full(count + 1, 1 / (count + 1)))
        col = 1 + int(gaps[0])
        for i, o in enumerate(order):
            o['c'] = col
            objects.append(o)
            col += o['w'] + 1 + int(gaps[i + 1])
        row += base_h + int(rng.integers(1, 3))
    height = row + int(rng.integers(0, 3))
    assert height <= 18
    return (height, width), objects


def _pair(rng, purpose, transpose):
    for _ in range(160):
        try:
            palette = list(map(int, rng.permutation(10)))
            bg, ink, left, right = palette[:4]
            shape, objects = _layout(rng, purpose)
            for o in objects:
                h, w = o['h'], o['w']
                all_pins = [(y, x) for y in range(1, h - 1) for x in range(1, w - 1)]
                n = min(len(all_pins), int(rng.integers(1, 4)))
                chosen = rng.choice(len(all_pins), n, replace=False)
                o['pins'] = [all_pins[int(i)] for i in chosen]
                o['patch'] = _u_patch(h, w, o['d'], (left, right)[o['d']], bg, ink, o['pins'])
            if purpose == 'cross':
                for o in objects:
                    o['pins'] = [(1, 1)]
                    o['patch'] = _u_patch(o['h'], o['w'], o['d'], (left, right)[o['d']],
                                          bg, ink, o['pins'])
            settled = _pack(objects, shape[1])
            board = _render_board(shape, bg, ink, (left, right), objects)
            docked = _render_board(shape, bg, ink, (left, right), settled)
            rooms = _chambers(docked, settled, (left, right))
            # Leave at least one cell to reveal the transferred color per room.
            assert all(np.any(docked[tuple(p.T)] == bg) for p in rooms)
            colors_n = int(rng.integers(2, min(6, len(objects)) + 1))
            colors = palette[4:4 + colors_n] + [int(rng.choice(palette[4:4 + colors_n]))
                                              for _ in range(len(objects) - colors_n)]
            rng.shuffle(colors)
            plan = np.full(shape, -1, dtype=int)
            expected = docked.copy()
            for i, (room, color) in enumerate(zip(rooms, colors)):
                plan[tuple(room.T)] = i
                y, x = room.T
                blank = expected[y, x] == bg
                expected[y[blank], x[blank]] = color
            scan_axis = (0 if purpose == 'cross' and not transpose else
                         1 if purpose == 'teach' and transpose else None)
            donor = _make_donor(rng, plan, bg, ink, colors, 27 - shape[0], scan_axis)
            # Layout is independent of the board's motion axis whenever both
            # arrangements fit; neither object has a fixed side in the canvas.
            side_fits = shape[1] + donor.shape[1] + 3 <= 30
            side = side_fits and bool(rng.integers(2))
            summed = shape[1] + donor.shape[1] if side else shape[0] + donor.shape[0]
            gap = int(rng.integers(1, min(3, 28 - summed) + 1))
            H = (max(shape[0], donor.shape[0]) if side else summed + gap) + 2
            W = (summed + gap if side else max(shape[1], donor.shape[1])) + 2
            H += int(rng.integers(min(2, 30-H) + 1))
            W += int(rng.integers(min(2, 30-W) + 1))
            assert H <= 30 and W <= 30
            inp = np.full((H, W), bg, dtype=np.int8)
            first_donor = bool(rng.integers(2))
            if side:
                c_d = 1 if first_donor else 1 + shape[1] + gap
                c_b = 1 + donor.shape[1] + gap if first_donor else 1
                r_d = int(rng.integers(1, H - donor.shape[0]))
                r_b = int(rng.integers(1, H - shape[0]))
            else:
                r_d = 1 if first_donor else 1 + shape[0] + gap
                r_b = 1 + donor.shape[0] + gap if first_donor else 1
                c_d = int(rng.integers(1, W - donor.shape[1]))
                c_b = int(rng.integers(1, W - shape[1]))
            inp[r_d:r_d+donor.shape[0], c_d:c_d+donor.shape[1]] = donor
            inp[r_b:r_b+shape[0], c_b:c_b+shape[1]] = board
            if transpose:
                inp, expected = inp.T.copy(), expected.T.copy()
            assert np.array_equal(solve(inp), expected)
            return {'input': inp, 'output': expected}
        except AssertionError:
            continue
    raise RuntimeError('bounded dock-and-enclose construction exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, low, high in ((num_train, 'num_train', 2, 10),
                                    (num_test, 'num_test', 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    transpose = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            purpose = ('test' if split == 'test' else 'cross' if index == 0 else
                       'teach' if index == 1 else 'free')
            for _ in range(32):
                pair = _pair(rng, purpose, transpose)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input construction exhausted')
    return puzzle


def _unit_pack(info):
    """A separate bounded one-cell simulation checks the boundary recurrence."""
    objects = [dict(o) for o in info['objects']]
    width = info['board'].shape[1]
    for d, delta in ((0, -1), (1, 1)):
        group = [o for o in objects if o['d'] == d]
        for _ in range(width * len(group) + 1):
            moved = False
            for o in group:
                proposed = dict(o, c=o['c'] + delta)
                if (1 <= proposed['c'] and proposed['c'] + o['w'] <= width - 1
                        and not any(p is not o and _overlap(proposed, p) for p in group)):
                    o['c'] += delta
                    moved = True
            if not moved:
                break
        else:
            raise AssertionError('unit simulation did not settle')
    assert not any(_overlap(a, b) for i, a in enumerate(objects) for b in objects[i + 1:])
    return objects


def _measure(info):
    def scan_key(room):
        r, c = _box(room)[:2]
        return (c, r) if info['transpose'] else (r, c)

    order = sorted(range(len(info['rooms'])), key=lambda i: scan_key(info['rooms'][i]))
    donor_order = sorted(range(len(info['donor_rooms'])),
                         key=lambda i: scan_key(info['donor_rooms'][i]))
    naive = [None] * len(order)
    for i, j in zip(order, donor_order):
        naive[i] = info['colors'][j]
    return dict(count=len(info['objects']), axis=int(info['transpose']),
                shape=info['board'].shape, donor_shape=info['donor'].shape,
                palette=(info['bg'], info['ink'], *info['rails']),
                footprints=tuple(sorted({(o['h'], o['w']) for o in info['objects']})),
                moves=tuple(abs(a['c']-b['c']) for a, b in zip(info['objects'], info['settled'])),
                initially_closed=len(_holes(np.isin(info['board'], info['rails']))),
                colors=len(set(info['colors'])),
                scan_shortcut_differs=naive != info['target_colors'],
                relation=tuple(tuple(row) for row in _relations(info['rooms'])))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            actual = out.T if info['transpose'] else out
            assert np.array_equal(actual, info['output']), 'incorrect docking or color transfer'
            simulated = _unit_pack(info)
            assert [o['c'] for o in simulated] == [o['c'] for o in info['settled']]
            # Full frame/cargo preservation and filling only formerly free cells.
            fixed = info['docked'] != info['bg']
            assert np.array_equal(actual[fixed], info['docked'][fixed])
            allowed = np.zeros(actual.shape, dtype=bool)
            for room, color in zip(info['rooms'], info['target_colors']):
                allowed[tuple(room.T)] = True
                vals = actual[tuple(room.T)]
                assert set(map(int, vals)) <= {color, info['ink']}
                assert np.any(vals == color)
            assert np.array_equal(actual[~allowed], info['docked'][~allowed])
            # A-ablated answer: pack and carry every pin, but never transfer.
            assert not np.array_equal(actual, info['docked'])
            # B-ablated oracle: assign the correct color to each U, but leave
            # its complete frame and cargo at the original position.
            unmoved = info['board'].copy()
            for o, color in zip(info['objects'], info['target_colors']):
                r, c, h, w, d = (o[k] for k in ('r', 'c', 'h', 'w', 'd'))
                cols = range(c, c+w-1) if d == 0 else range(c+1, c+w)
                for y in range(r+1, r+h-1):
                    for x in cols:
                        if unmoved[y, x] == info['bg']:
                            unmoved[y, x] = color
            if any(a['c'] != b['c'] for a, b in zip(info['objects'], info['settled'])):
                assert not np.array_equal(actual, unmoved)
            records[split].append(_measure(info))
    assert len({r['axis'] for s in records.values() for r in s}) == 1
    assert records['train'][0]['count'] == 2 and records['train'][1]['count'] == 4
    assert all(r['count'] <= 5 for r in records['train'])
    assert all(r['count'] == 6 for r in records['test'])
    assert all(records['train'][0]['moves']), 'both rail instructions must move'
    assert records['train'][0]['initially_closed'] == 0
    assert any(r['scan_shortcut_differs'] for r in records['train'][:2])
    assert len({r['relation'] for r in records['train']}) >= 2


def self_test():
    """Bounded, reproducible 42-seed semantic and structural checks."""
    fingerprints, shapes, relations, palettes, footprints, axes = set(), set(), set(), set(), set(), set()
    measurements = []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprint = tuple((_key(pair['input']), _key(pair['output']))
                            for split in ('train', 'test') for pair in p[split])
        fingerprints.add(fingerprint)
        if seed in (0, 41):
            q = generate_puzzle(seed)
            assert fingerprint == tuple((_key(pair['input']), _key(pair['output']))
                                        for split in ('train', 'test') for pair in q[split])
        for split in ('train', 'test'):
            for pair in p[split]:
                info = _analyze(pair['input'])
                m = _measure(info)
                measurements.append(m)
                shapes.add(pair['input'].shape)
                relations.add(m['relation'])
                palettes.add(m['palette'])
                footprints.update(m['footprints'])
                axes.add(m['axis'])
        # The same input-only rule is equivariant to visible axis and colors.
        pair = p['test'][0]
        assert np.array_equal(solve(pair['input'].T), pair['output'].T)
        perm = np.random.default_rng(1000 + seed).permutation(10)
        assert np.array_equal(solve(perm[pair['input']]), perm[pair['output']])
    assert len(fingerprints) == 42 and axes == {0, 1}
    assert len(shapes) >= 30 and len(relations) >= 30 and len(palettes) >= 100
    assert len(footprints) >= 12
    assert {2, 3, 4, 5, 6} <= {m['count'] for m in measurements}
    assert any(0 in m['moves'] for m in measurements)
    assert any(m['initially_closed'] > 0 for m in measurements)
    # The host verifier enumerates every legal count combination. Keep this
    # embedded test bounded by checking its four useful boundary cases.
    for a, b in ((2, 1), (10, 1), (10, 4), (5, 4)):
        validate_generated_puzzle(generate_puzzle(12345, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (2.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    broken = generate_puzzle(42, 2, 1)
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted wall accepted')
    return dict(seeds=42, puzzles=len(fingerprints), input_shapes=len(shapes),
                spatial_arrangements=len(relations), role_palettes=len(palettes),
                frame_footprints=len(footprints))


if __name__ == '__main__':
    print(self_test())
