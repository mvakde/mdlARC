"""Mirror-certified pigment labels transferred between deformed compartments.

Only standard-library modules and NumPy are used. No parent code, examples,
random state, or training labels enter solve().
"""
from __future__ import annotations

import copy
import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask, diagonal=False):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    steps = _STEPS + (((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else ())
    parts = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, part = [start], [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    part.append(p)
        parts.append(np.asarray(part, dtype=int))
    return parts


def _bbox(points):
    return tuple(map(int, points.min(0))) + tuple(map(int, points.max(0) + 1))


def _rooms(walls):
    empty = np.pad(~walls, 1, constant_values=True)
    shape = np.array(empty.shape)
    return [p - 1 for p in _components(empty)
            if np.all(p > 0) and np.all(p < shape - 1)]


def _background(grid):
    border = np.concatenate((grid[0], grid[-1], grid[:, 0], grid[:, -1]))
    counts = np.bincount(border, minlength=10)
    best = np.flatnonzero(counts == counts.max())
    assert len(best) == 1 and counts[best[0]] > len(border) / 2
    return int(best[0])


def _extract(grid):
    """Discover the common wall color and isolate both compartment objects."""
    bg = _background(grid)
    choices = []
    for wall in map(int, np.unique(grid)):
        if wall == bg:
            continue
        parts = _components(grid == wall, diagonal=True)
        if len(parts) != 2:
            continue
        objects, covered = [], grid == wall
        valid = True
        for part in parts:
            r, c, R, C = _bbox(part)
            walls = np.zeros((R - r, C - c), dtype=bool)
            walls[part[:, 0] - r, part[:, 1] - c] = True
            rooms = _rooms(walls)
            if not 3 <= len(rooms) <= 6:
                valid = False
                break
            crop = np.full(walls.shape, bg, dtype=int)
            crop[walls] = wall
            for room in rooms:
                crop[tuple(room.T)] = grid[room[:, 0] + r, room[:, 1] + c]
                covered[room[:, 0] + r, room[:, 1] + c] = True
            colored = np.any((crop != bg) & (crop != wall))
            objects.append((crop, rooms, (r, c, R, C), bool(colored)))
        if (not valid or len(objects) != 2 or
                len(objects[0][1]) != len(objects[1][1]) or
                sum(o[3] for o in objects) != 1 or
                np.any((grid != bg) & ~covered)):
            continue
        donor = next(o for o in objects if o[3])
        target = next(o for o in objects if not o[3])
        choices.append((bg, wall, donor, target))
    assert len(choices) == 1, "framework roles are not uniquely visible"
    return choices[0]


def _axis(room):
    """The enclosure, not a seed or a learned orientation, defines the frame."""
    points = set(map(tuple, room.tolist()))
    r, c, R, C = _bbox(room)
    options = []
    a = c + C - 1
    if {(y, a - x) for y, x in points} == points:
        options.append((0, a))  # vertical, doubled column coordinate
    a = r + R - 1
    if {(a - y, x) for y, x in points} == points:
        options.append((1, a))  # horizontal, doubled row coordinate
    assert len(options) == 1, "donor chamber must have exactly one mirror axis"
    return options[0]


def _reflection(point, axis):
    direction, doubled = axis
    r, c = point
    return (r, doubled - c) if direction == 0 else (doubled - r, c)


def _elect(crop, room, bg, wall):
    axis = _axis(room)
    colors = set(map(int, crop[tuple(room.T)])) - {bg, wall}
    assert colors, "a donor chamber cannot be empty"
    scores, counts, supported = {}, {}, {}
    for color in sorted(colors):
        cells = {tuple(p) for p in room.tolist() if crop[tuple(p)] == color}
        keep = {p for p in cells if _reflection(p, axis) in cells}
        scores[color], counts[color], supported[color] = len(keep), len(cells), keep
    maximum = max(scores.values())
    winners = [color for color in scores if scores[color] == maximum]
    assert maximum > 0 and len(winners) == 1, "tied pigment certificates"
    return winners[0], axis, scores, counts, supported


def _relations(rooms):
    boxes = [_bbox(p) for p in rooms]
    return [[tuple(-1 if a[k + 2] <= b[k] else 1 if b[k + 2] <= a[k] else 0
                   for k in (0, 1)) for b in boxes] for a in boxes]


def _match(drooms, trooms, dyes):
    """Find best non-reversing spatial bijections; all must paint identically."""
    a, b = _relations(drooms), _relations(trooms)
    n = len(a)
    best_score, assignments = -1, []

    def search(mapping, used, score):
        nonlocal best_score, assignments
        if len(mapping) == n:
            result = tuple(dyes[mapping.index(j)] for j in range(n))
            if score > best_score:
                best_score, assignments = score, [result]
            elif score == best_score:
                assignments.append(result)
            return
        i = len(mapping)
        for j in range(n):
            if j in used:
                continue
            if any(any(u and v and u != v for u, v in zip(a[i][k], b[j][l]))
                   for k, l in enumerate(mapping)):
                continue
            extra = sum(sum(u == v for u, v in zip(a[i][k], b[j][l]))
                        for k, l in enumerate(mapping))
            search(mapping + [j], used | {j}, score + extra)

    search([], set(), 0)
    assert assignments and len(set(assignments)) == 1, "ambiguous chamber correspondence"
    return assignments[0]


def _predict(grid):
    bg, wall, donor, target = _extract(grid)
    dcrop, drooms, db, _ = donor
    tcrop, trooms, tb, _ = target
    certificates = [_elect(dcrop, p, bg, wall) for p in drooms]
    dyes = [e[0] for e in certificates]
    target_dyes = _match(drooms, trooms, dyes)
    out = tcrop.copy()
    for room, color in zip(trooms, target_dyes):
        out[tuple(room.T)] = color
    return out, (bg, wall, donor, target, certificates, target_dyes)


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(input_grid)
    if (grid.ndim != 2 or not np.issubdtype(grid.dtype, np.integer) or
            not all(1 <= d <= 30 for d in grid.shape) or
            np.any((grid < 0) | (grid > 9))):
        raise ValueError("input must be an integer ARC grid")
    return _predict(grid.astype(int, copy=True))[0]


def _plan(rng, count):
    shapes = [(2, 2), (2, 3), (3, 2), (3, 3)]
    rows, cols = shapes[int(rng.choice([i for i, s in enumerate(shapes)
                                       if s[0] * s[1] >= count]))]
    labels = np.arange(rows * cols).reshape(rows, cols)
    corners = [(0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1)]
    for i in rng.permutation(4):
        if np.count_nonzero(labels >= 0) > count and rng.random() < .28:
            trial = labels.copy()
            trial[corners[int(i)]] = -1
            if len(_components(trial >= 0)) == 1:
                labels = trial
    while len(set(labels.flat) - {-1}) > count:
        edges = set()
        for r, c in np.argwhere(labels >= 0):
            for dr, dc in ((1, 0), (0, 1)):
                if r + dr < rows and c + dc < cols:
                    x, y = int(labels[r, c]), int(labels[r + dr, c + dc])
                    if y >= 0 and x != y:
                        points = np.argwhere((labels == x) | (labels == y))
                        a, b, A, B = _bbox(points)
                        if len(points) == (A - a) * (B - b):
                            edges.add(tuple(sorted((x, y))))
        assert edges, "partition cannot reach requested count"
        x, y = sorted(edges)[int(rng.integers(len(edges)))]
        labels[labels == y] = x
    result = np.full(labels.shape, -1, dtype=int)
    for new, old in enumerate(sorted(set(labels.flat) - {-1})):
        result[labels == old] = new
    return result


def _render(rng, plan, widths, heights, donor, parity=None):
    ys = np.concatenate(([0], np.cumsum(heights + 1)))
    xs = np.concatenate(([0], np.cumsum(widths + 1)))
    labels = np.full((int(ys[-1]) + 1, int(xs[-1]) + 1), -1, dtype=int)
    for r, c in np.argwhere(plan >= 0):
        labels[ys[r]:ys[r + 1] + 1, xs[c]:xs[c + 1] + 1] = -2
    for label in range(int(plan.max()) + 1):
        r, c, R, C = _bbox(np.argwhere(plan == label))
        y0, x0 = int(ys[r]) + 1, int(xs[c]) + 1
        h, w = int(ys[R] - ys[r]) - 1, int(xs[C] - xs[c]) - 1
        if donor:
            # A single vertical axis, while top and bottom profiles differ.
            h = max(5, h - int(rng.integers(min(2, h - 4))))
            w = max(5, w - int(rng.integers(min(2, w - 4))))
            if label == 0 and parity is not None and (w - 1) % 2 != parity:
                w = w - 1 if w > 5 else w + 1
                assert w <= int(xs[C] - xs[c]) - 1
            for attempt in range(20):
                levels = rng.integers(0, w // 2, size=h)
                levels[0] = 0
                levels[-1] = int(rng.integers(1, w // 2))
                lengths = w - 2 * levels
                if sum(int(v) // 2 for v in lengths) >= 8:
                    break
            else:
                levels = np.zeros(h, dtype=int)
                levels[-1] = 1
            for y, level in enumerate(levels):
                labels[y0 + y, x0 + level:x0 + w - level] = label
        else:
            dy = 0 if rng.random() < .45 else int(rng.integers(min(3, h)))
            dx = 0 if rng.random() < .45 else int(rng.integers(min(3, w)))
            h = h - dy if rng.random() < .65 else int(rng.integers(1, h - dy + 1))
            w = w - dx if rng.random() < .65 else int(rng.integers(1, w - dx + 1))
            y0, x0 = y0 + dy, x0 + dx
            spine = int(rng.integers(w))
            for y in range(h):
                left = int(rng.integers(spine + 1))
                right = int(rng.integers(spine + 1, w + 1))
                if rng.random() < .6:
                    left, right = 0, w
                labels[y0 + y, x0 + left:x0 + right] = label
    r, c, R, C = _bbox(np.argwhere(labels != -1))
    return labels[r:R, c:C]


def _stamp(rng, grid, room, winner, pigments, bg, wall, misleading, rich=False, heavy=False):
    axis = _axis(room)
    pairs, central = [], []
    for p in map(tuple, room.tolist()):
        q = _reflection(p, axis)
        if p < q:
            pairs.append((p, q))
        elif p == q:
            central.append(p)
    rng.shuffle(pairs)
    assert len(pairs) >= 8
    rival = int(rng.choice([c for c in pigments if c != winner]))
    third = next((c for c in pigments if c not in (winner, rival)), None)
    limit = min(4, (len(pairs) - 2) // 3) if misleading else min(4, len(pairs) - 3)
    q = int(rng.integers(2, limit + 1))
    for pair in pairs[:q]:
        for p in pair:
            grid[p] = winner
    # An unpaired winner mark defeats "choose the entirely symmetric hue".
    extra_color = third if rich else winner
    grid[pairs[q][int(rng.integers(2))]] = extra_color
    # A weaker genuine rival certificate defeats "choose any paired hue".
    for p in pairs[q + 1]:
        grid[p] = rival
    rest = pairs[q + 2:]
    if misleading:
        noise_count = int(rng.integers(2 * q, len(rest) + 1))
        for pair in rest[:noise_count]:
            grid[pair[int(rng.integers(2))]] = rival
        rest = rest[noise_count:]
    for pair in rest:
        if rng.random() < (.9 if heavy else .45):
            pool = [c for c in pigments if c != winner] if misleading else pigments
            color = int(rng.choice(pool))
            grid[pair[int(rng.integers(2))]] = color
    if not misleading:
        for p in central[:int(rng.integers(min(3, len(central)) + 1))]:
            grid[p] = winner
    certificate = _elect(grid, room, bg, wall)
    assert certificate[0] == winner
    if misleading:
        counts = certificate[3]
        assert counts[rival] > counts[winner]


def _paint(rng, labels, bg, wall, dyes=None, pigments=None, misleading=False,
           rich=False, heavy=False):
    grid = np.full(labels.shape, bg, dtype=int)
    grid[labels == -2] = wall
    if dyes is not None:
        for label, dye in enumerate(dyes):
            room = np.argwhere(labels == label)
            if pigments is None:
                grid[tuple(room.T)] = dye
            else:
                _stamp(rng, grid, room, dye, pigments, bg, wall,
                       misleading and label == 0, rich, heavy)
    return grid


def _embed(rng, donor, target, bg, layout):
    dh, dw = donor.shape
    th, tw = target.shape
    gap = int(rng.integers(1, 4))
    if layout == 'side':
        assert dw + tw + gap <= 30
        height = max(dh, th)
        dr = int(rng.integers(height - dh + 1))
        tr = int(rng.integers(height - th + 1))
        dc, tc = 0, dw + gap
        width = dw + tw + gap
        if rng.random() < .5:
            dc, tc = tw + gap, 0
    elif layout == 'stack':
        assert dh + th + gap <= 30
        width = max(dw, tw)
        dc = int(rng.integers(width - dw + 1))
        tc = int(rng.integers(width - tw + 1))
        dr, tr = 0, dh + gap
        height = dh + th + gap
        if rng.random() < .5:
            dr, tr = th + gap, 0
    else:
        assert dh + th + gap <= 30 and dw + tw + gap <= 30
        height, width = dh + th + gap, dw + tw + gap
        dr, dc, tr, tc = 0, 0, dh + gap, dw + gap
        if rng.random() < .5:
            dr, tr = th + gap, 0
        if rng.random() < .5:
            dc, tc = tw + gap, 0
    ph = int(rng.integers(min(30 - height, 7) + 1))
    pw = int(rng.integers(min(30 - width, 7) + 1))
    top, left = int(rng.integers(ph + 1)), int(rng.integers(pw + 1))
    inp = np.full((height + ph, width + pw), bg, dtype=int)
    for grid, r, c in ((donor, dr, dc), (target, tr, tc)):
        inp[r + top:r + top + grid.shape[0], c + left:c + left + grid.shape[1]] = grid
    _background(inp)
    return inp


def _pair(rng, count, colors, layout, parity, rich=False, heavy=False,
          fallback_only=False, require_scan_error=False, scan_transpose=False):
    bg, wall = map(int, colors[:2])
    pigments = list(map(int, colors[2:]))
    dyes = [pigments[0], pigments[1]] + [int(rng.choice(pigments)) for _ in range(count - 2)]
    rng.shuffle(dyes)
    if dyes[0] == dyes[1]:
        dyes[1] = int(rng.choice([c for c in pigments if c != dyes[0]]))
    if require_scan_error and dyes[2] == dyes[0]:
        dyes[2] = int(rng.choice([c for c in pigments if c != dyes[0]]))
    for attempt in ([79] if fallback_only else range(80)):
        try:
            if attempt == 79:
                # A bounded fitting fallback; inference conventions stay identical.
                plan = np.arange(6).reshape(3, 2)
                for label in range(count, 6):
                    plan[plan == label] = count - 1
                # Ensure the last room is rectangular for counts 3 and 5.
                if count == 3:
                    plan = np.array([[0, 1], [2, 2]])
                elif count == 4:
                    plan = np.array([[0, 1], [2, 3]])
                dwidths = np.full(plan.shape[1], 6)
                dheights = np.full(plan.shape[0], 5)
                twidths = np.full(plan.shape[1], 2)
                theights = np.full(plan.shape[0], 1)
                theights[0] = 2
                chosen_layout = layout
            else:
                plan = _plan(rng, count)
                dwidths = rng.integers(5, 8, size=plan.shape[1])
                dheights = rng.integers(5, 8, size=plan.shape[0])
                if parity is not None:
                    dwidths[:] = 7 if parity == 0 else 6
                twidths = rng.integers(1, 8, size=plan.shape[1])
                theights = rng.integers(1, 8, size=plan.shape[0])
                chosen_layout = layout
            d = _render(rng, plan, dwidths, dheights, True, parity)
            t = _render(rng, plan, twidths, theights, False)
            if attempt == 79:
                # Right-hand top pocket starts before the left-hand one.
                # Spatial left/right relations identify their different dyes.
                t[(t == 0) | (t == 1) | (t == 2)] = -2
                t[2, 2], t[1, 4], t[4, 1] = 0, 1, 2
            donor = _paint(rng, d, bg, wall, dyes, pigments, True, rich, heavy)
            target = _paint(rng, t, bg, wall)
            expected = _paint(rng, t, bg, wall, dyes)
            if attempt == 79:
                inp = np.full((30, 30), bg, dtype=int)
                dh, dw = donor.shape
                th, tw = target.shape
                tr, tc = (1, dw + 2) if layout == 'side' else (dh + 2, 1) if layout == 'stack' else (dh + 2, dw + 2)
                assert tr + th < 30 and tc + tw < 30
                inp[1:dh + 1, 1:dw + 1] = donor
                inp[tr:tr + th, tc:tc + tw] = target
            else:
                inp = _embed(rng, donor, target, bg, chosen_layout)
            predicted, info = _predict(inp)
            assert np.array_equal(predicted, expected)
            assert len(info[2][1]) == count
            if require_scan_error:
                scan_info = _predict(inp.T)[1] if scan_transpose else info
                tcrop, trooms = scan_info[3][:2]
                scan = tcrop.copy()
                for room, e in zip(trooms, scan_info[4]):
                    scan[tuple(room.T)] = e[0]
                assert not np.array_equal(scan, expected.T if scan_transpose else expected)
            return {'input': inp, 'output': expected}
        except AssertionError:
            continue
    raise AssertionError('bounded compartment construction failed')


def _key(grid):
    return grid.shape, grid.astype(np.uint8).tobytes()


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or
                not low <= value <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    shift = str(rng.choice(['count', 'pigments', 'layout']))
    transpose = bool(rng.integers(2))
    train_layout = str(rng.choice(['side', 'stack']))
    puzzle, seen = {'train': [], 'test': []}, set()
    previous_roles = None
    first_count = int(rng.integers(3, 5 if shift == 'count' else 7))
    for split, length in (('train', num_train), ('test', num_test)):
        for index in range(length):
            if shift == 'count':
                count = int(rng.integers(3, 5)) if split == 'train' else int(rng.integers(5, 7))
            else:
                count = int(rng.integers(3, 7))
            if split == 'train' and index == 0:
                count = first_count
            if split == 'train' and index == 1:
                options = [c for c in range(3, 5 if shift == 'count' else 7) if c != first_count]
                count = int(rng.choice(options))
            layout = (('stack' if train_layout == 'side' else 'side') if split == 'test' else train_layout) if shift == 'layout' else str(rng.choice(['side', 'stack', 'diagonal']))
            richness = int(rng.integers(3, 6)) if split == 'test' and shift == 'pigments' else 2 if shift == 'pigments' else int(rng.integers(2, 6))
            palette = list(map(int, rng.permutation(10)))
            if split == 'train' and index == 1:
                bg = next(c for c in palette if c != previous_roles[0])
                wall = next(c for c in palette if c not in (bg, previous_roles[1]))
                palette = [bg, wall] + [c for c in palette if c not in (bg, wall)]
            colors = palette[:2 + richness]
            parity = index if split == 'train' and index < 2 else None
            for _ in range(12):
                pair = _pair(rng, count, colors, layout, parity,
                             rich=(shift == 'pigments' and split == 'test'),
                             heavy=(split == 'test' and shift == 'pigments'),
                             require_scan_error=(split == 'train' and index == 0),
                             scan_transpose=transpose)
                if transpose:
                    pair = {k: v.T.copy() for k, v in pair.items()}
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError('duplicate inputs after bounded retries')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train' and index == 0:
                previous_roles = colors[:2]
    return puzzle


def measure_pair(pair):
    grid = np.asarray(pair['input'])
    predicted, info = _predict(grid)
    bg, wall, donor, target, certificates, target_dyes = info
    dcrop, drooms, db, _ = donor
    tcrop, trooms, tb, _ = target
    out = np.asarray(pair['output'])
    assert np.array_equal(predicted, out)
    interior = np.zeros(tcrop.shape, dtype=bool)
    for room, dye in zip(trooms, target_dyes):
        interior[tuple(room.T)] = True
        assert np.all(out[tuple(room.T)] == dye)
    assert np.array_equal(out == wall, tcrop == wall)
    assert np.array_equal(out[~interior], tcrop[~interior])
    assert np.all(tcrop[interior] == bg)
    assert np.array_equal(out != tcrop, interior)
    assert len(set(e[0] for e in certificates)) >= 2
    majority_errors = sum(e[3][e[0]] < max(e[3].values()) for e in certificates)
    assert majority_errors >= 1
    assert any(sum(v > 0 for v in e[2].values()) >= 2 for e in certificates)
    row_overlap = min(db[2], tb[2]) > max(db[0], tb[0])
    col_overlap = min(db[3], tb[3]) > max(db[1], tb[1])
    layout = 'side' if row_overlap else 'stack' if col_overlap else 'diagonal'
    # Scan-order painting is a concrete, inferior correspondence shortcut.
    scan = tcrop.copy()
    for room, e in zip(trooms, certificates):
        scan[tuple(room.T)] = e[0]
    global_dye = max((int(c) for c in np.unique(dcrop) if c not in (bg, wall)),
                     key=lambda c: np.count_nonzero(dcrop == c))
    assert any(dye != global_dye for dye in target_dyes)
    return {'rooms': len(drooms), 'background': bg, 'wall': wall,
            'layout': layout, 'input_shape': tuple(grid.shape), 'output_shape': tuple(out.shape),
            'orientations': tuple(sorted({e[1][0] for e in certificates})),
            'parities': tuple(sorted({e[1][1] % 2 for e in certificates})),
            'pigments': max(len(e[2]) for e in certificates),
            'majority_errors': majority_errors,
            'unpaired_winner': any(e[3][e[0]] > e[2][e[0]] for e in certificates),
            'scan_error': not np.array_equal(scan, out),
            'nonrectangular': sum(len(p) != (_bbox(p)[2] - _bbox(p)[0]) * (_bbox(p)[3] - _bbox(p)[1]) for p in drooms + trooms),
            'edge': any(b[0] == 0 or b[1] == 0 or b[2] == grid.shape[0] or b[3] == grid.shape[1] for b in (db, tb)),
            'areas_d': tuple(sorted(map(len, drooms))), 'areas_t': tuple(sorted(map(len, trooms))),
            'structure': (tcrop.shape, (tcrop == wall).tobytes())}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, metrics = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for value in pair.values():
                grid = np.asarray(value)
                assert grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            key = _key(np.asarray(pair['input']))
            assert key not in seen
            seen.add(key)
            metrics[split].append(measure_pair(pair))
    training = metrics['train']
    assert len({m['rooms'] for m in training}) >= 2
    assert len({m['structure'] for m in training}) >= 2
    assert len({m['background'] for m in training}) >= 2
    assert len({m['wall'] for m in training}) >= 2
    assert all(m['majority_errors'] and m['unpaired_winner'] for m in training[:2])
    assert any(m['scan_error'] for m in training), 'training never refutes scan-order transfer'
    assert {0, 1} <= {p for m in training[:2] for p in m['parities']}
    assert len({m['orientations'] for s in metrics for m in metrics[s]}) == 1
    # Check the observable single-factor query shift, without generation metadata.
    count_shift = min(m['rooms'] for m in metrics['test']) > max(m['rooms'] for m in training)
    pigment_shift = max(m['pigments'] for m in training) == 2 and min(m['pigments'] for m in metrics['test']) >= 3
    layout_shift = not ({m['layout'] for m in training} & {m['layout'] for m in metrics['test']})
    assert count_shift or pigment_shift or layout_shift, 'no meaningful observable test shift'
    return None


def self_test():
    # A small explicit certificate: seven rival pixels lose to four reflected
    # winner pixels, despite the winner also having an unmatched addition.
    mask = np.ones((5, 5), dtype=bool)
    mask[4, [0, 4]] = False
    room = np.argwhere(mask)
    fixture = np.zeros((5, 5), dtype=int)
    for p in ((0, 0), (0, 4), (2, 1), (2, 3), (4, 1)):
        fixture[p] = 1
    for p in ((1, 0), (1, 4), (0, 1), (1, 1), (2, 0), (3, 0), (3, 1)):
        fixture[p] = 2
    e = _elect(fixture, room, 0, 9)
    assert e[0] == 1 and e[2] == {1: 4, 2: 2} and e[3] == {1: 5, 2: 7}
    assert _elect(fixture.T, room[:, ::-1], 0, 9)[0] == 1
    axis_cell = fixture.copy()
    axis_cell[1, 2] = 1
    assert _elect(axis_cell, room, 0, 9)[2][1] == 5
    tied = fixture.copy()
    tied[3, 4] = 2
    try:
        _elect(tied, room, 0, 9)
    except AssertionError:
        pass
    else:
        raise AssertionError('tied certificates accepted')
    for count in range(3, 7):
        for parity in (0, 1):
            for layout in ('side', 'stack', 'diagonal'):
                for transpose in (False, True):
                    pair = _pair(np.random.default_rng(count), count, [0, 9, 1, 2, 3],
                                 layout, parity, rich=True, fallback_only=True,
                                 require_scan_error=True, scan_transpose=transpose)
                    assert np.array_equal(solve(pair['input']), pair['output'])
    records, fingerprints, shapes, structures = [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                before = pair['input'].copy()
                assert np.array_equal(solve(before), pair['output'])
                assert np.array_equal(before, pair['input'])
                m = measure_pair(pair)
                records.append(m)
                shapes.add(m['input_shape'])
                structures.add(m['structure'])
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(p[k], q[k]) for s in puzzle
                       for p, q in zip(puzzle[s], again[s]) for k in p)
    assert len(fingerprints) == 42
    assert {m['rooms'] for m in records} == {3, 4, 5, 6}
    assert {m['orientations'] for m in records} == {(0,), (1,)}
    assert {m['background'] for m in records} == set(range(10))
    assert len(shapes) >= 40 and len(structures) >= 100
    assert any(m['scan_error'] for m in records)
    assert any(m['edge'] for m in records)
    assert max(m['pigments'] for m in records) >= 4
    assert all(m['nonrectangular'] >= m['rooms'] for m in records)
    # All legal count combinations and non-integral/boolean/illegal cases.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            validate_generated_puzzle(generate_puzzle(6000 + a * 10 + b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Mutating a transferred dye or a preserved wall must be rejected.
    puzzle = generate_puzzle(0, 2, 1)
    for point in ((0, 0), tuple(np.argwhere(puzzle['test'][0]['output'] == _extract(puzzle['test'][0]['input'])[1])[0])):
        bad = copy.deepcopy(puzzle)
        out = bad['test'][0]['output']
        out[point] = (int(out[point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('mutated output accepted')
    return {'seeds': 42, 'pairs': len(records), 'input_shapes': len(shapes),
            'target_structures': len(structures), 'scan_order_counterexamples': sum(m['scan_error'] for m in records),
            'majority_counterexamples': sum(m['majority_errors'] for m in records)}


if __name__ == '__main__':
    print(self_test())
