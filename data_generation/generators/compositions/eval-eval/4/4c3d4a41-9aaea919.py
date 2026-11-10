"""Comb-driven overflow counter. All solving conventions are visible in one input.

Canonical coordinates have an external comb on the left and a clipping roof
above the pins. Whole scenes may undergo any dihedral transformation. The
constructor draws from explicit objects; solve moves/counts the parsed pixels.
Only the standard library and NumPy are used. No parent code is imported.
"""
from __future__ import annotations

import json
import numpy as np


def _runs(values):
    edges = np.diff(np.r_[False, values, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def _orient(grid, code):
    return np.rot90(np.fliplr(grid) if code >= 4 else grid, code % 4).copy()


def _unorient(grid, code):
    a = np.rot90(grid, -(code % 4))
    return (np.fliplr(a) if code >= 4 else a).copy()


def _connected(mask):
    remaining = set(zip(*np.nonzero(mask)))
    if not remaining:
        return False
    pending = [remaining.pop()]
    while pending:
        y, x = pending.pop()
        for p in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if p in remaining:
                pending.append(p)
                remaining.remove(p)
    return not remaining


def _draw_mask(grid, y, x, mask, color, roof=None):
    for dy, dx in zip(*np.nonzero(mask)):
        row = y + int(dy)
        if roof is None or row > roof:
            assert 0 <= row < grid.shape[0]
            grid[row, x + int(dx)] = color


def _housing(grid, f):
    t, b, l, r, s = (f[k] for k in ('top', 'bottom', 'left', 'right', 'structure'))
    grid[t, l:r + 1] = grid[b, l:r + 1] = s
    grid[t:b + 1, l] = grid[t:b + 1, r] = s
    grid[f['base'] - max(f['teeth']):f['base'] + 1, l] = f['background']


def _comb(grid, f, start):
    base, s = f['base'], f['structure']
    grid[base, start:start + f['span']] = s
    for lane, height in zip(f['lanes'], f['teeth']):
        x = start + lane['x'] - f['lanes'][0]['x']
        if height:
            grid[base - height:base, x:x + f['width']] = s


def _input_from_objects(f):
    a = np.full(f['shape'], f['background'], dtype=np.int8)
    _housing(a, f)
    _comb(a, f, f['key_left'])
    for i, lane in enumerate(f['lanes']):
        for y in lane['ys']:
            _draw_mask(a, y, lane['x'], lane['mask'], lane['color'])
        if i in f['sources']:
            a[f['base'] + 2, lane['x']:lane['x'] + f['width']] = f['marker']
    dst = f['lanes'][f['destination']]
    x, w, b = dst['x'], f['width'], f['base']
    a[b - 1, x - 1:x + w + 1] = f['marker']
    a[b - 2, (x - 1, x + w)] = f['marker']
    return a


def _output_from_objects(f):
    """Constructor path: place objects at their explicit final coordinates."""
    a = np.full(f['shape'], f['background'], dtype=np.int8)
    _housing(a, f)
    _comb(a, f, f['lanes'][0]['x'])
    total = sum(f['lanes'][i]['loss'] for i in f['sources'])
    for i, lane in enumerate(f['lanes']):
        if i == f['destination']:
            bottom_start = lane['ys'][-1]
            for k in range(len(lane['ys']) + total):
                _draw_mask(a, bottom_start - k * f['pitch'], lane['x'],
                           lane['mask'], lane['color'])
        else:
            color = f['marker'] if i in f['sources'] else lane['color']
            for y in lane['ys']:
                _draw_mask(a, y - lane['shift'], lane['x'], lane['mask'],
                           color, f['top'])
    return a


def _wall_candidates(a, bg):
    groups = {}
    for y, row in enumerate(a):
        for c in np.unique(row):
            if c == bg:
                continue
            for x, stop in _runs(row == c):
                if stop - x >= 7:
                    groups.setdefault((int(c), x, stop - 1), []).append(y)
    return [(s, l, r, t, b) for (s, l, r), ys in groups.items()
            for t in ys for b in ys if b > t]


def _read_canonical(a, bg, wall):
    """Parse only geometric/semantic invariants, not the sampler's schedules."""
    s, left, right, top, bottom = wall
    assert np.all(a[top:bottom + 1, right] == s)
    edge = a[top + 1:bottom, left]
    opening = np.flatnonzero(edge == bg) + top + 1
    assert len(opening) >= 2
    assert np.array_equal(opening, np.arange(opening[0], opening[-1] + 1))
    base = int(opening[-1])
    assert bottom - base >= 4 and base - top >= 5
    outside = np.flatnonzero(a[base, :left] != bg)
    assert len(outside) >= 3
    kl, kr = int(outside[0]), int(outside[-1])
    assert kr < left - 1 and len(outside) == kr - kl + 1
    assert np.all(a[base, kl:kr + 1] == s)
    # The band ends above the cup, so its column projection contains tokens only.
    band = a[top + 1:base - 2, left + 1:right]
    spans = _runs(np.any(band != bg, axis=0))
    assert 2 <= len(spans) <= 6
    widths = {stop - x for x, stop in spans}
    assert len(widths) == 1
    w = widths.pop()
    assert 1 <= w <= 3
    xs = [left + 1 + x for x, stop in spans]
    assert xs[0] >= left + 2 and xs[-1] + w <= right - 1
    assert xs[-1] + w - xs[0] == kr - kl + 1
    teeth = []
    for x in xs:
        col = kl + x - xs[0]
        height = 0
        while base - height - 1 >= 0 and a[base - height - 1, col] == s:
            height += 1
        assert 0 <= height <= base - top - 2
        teeth.append(height)
    assert teeth.count(0) == 1 and max(teeth) > 0
    dst = teeth.index(0)
    assert int(opening[0]) == base - max(teeth)
    marker_values = set()
    sources = []
    for i, x in enumerate(xs):
        bar = a[base + 2, x:x + w]
        if np.any(bar != bg):
            assert np.all(bar == bar[0]) and i != dst
            marker_values.add(int(bar[0]))
            sources.append(i)
    assert sources and len(marker_values) == 1
    marker = marker_values.pop()
    assert len({bg, s, marker}) == 3
    assert np.all(a[base - 1, xs[dst] - 1:xs[dst] + w + 1] == marker)
    assert np.all(a[base - 2, (xs[dst] - 1, xs[dst] + w)] == marker)
    lanes, pitches = [], set()
    for i, x in enumerate(xs):
        strip = a[top + 1:base - 2, x:x + w]
        runs = _runs(np.any(strip != bg, axis=1))
        assert runs
        ys, masks, colors = [], [], set()
        for start, stop in runs:
            block = strip[start:stop]
            mask = block != bg
            assert 1 <= mask.shape[0] <= 3 and _connected(mask)
            assert np.all(np.any(mask, axis=0)) and np.all(np.any(mask, axis=1))
            masks.append(mask)
            colors.update(int(c) for c in block[mask])
            ys.append(top + 1 + start)
        assert len(colors) == 1 and colors.isdisjoint({bg, s, marker})
        assert all(np.array_equal(masks[0], m) for m in masks)
        pitches.update(b - a for a, b in zip(ys, ys[1:]))
        h = masks[0].shape[0]
        if i == dst:
            assert ys[-1] + h - 1 == base - 3
            shift = loss = 0
        else:
            assert ys[0] == top + 1
            pin_bottom = ys[-1] + h - 1
            shift = pin_bottom - (base - teeth[i] - 1)
            assert shift >= 0  # contact or overlap; no unsupported attraction
            assert pin_bottom - shift > top
            loss = sum(y + h - 1 - shift <= top for y in ys)
        lanes.append(dict(x=x, ys=ys, mask=masks[0], color=colors.pop(),
                          shift=shift, loss=loss))
    assert len(pitches) == 1, 'each input must visibly specify a common pitch'
    pitch = pitches.pop()
    assert all(pitch > lane['mask'].shape[0] for lane in lanes)
    final_count = len(lanes[dst]['ys']) + sum(lanes[i]['loss'] for i in sources)
    assert lanes[dst]['ys'][-1] - (final_count - 1) * pitch > top
    f = dict(shape=a.shape, background=bg, structure=s, marker=marker,
             top=top, bottom=bottom, left=left, right=right, base=base,
             key_left=kl, span=kr - kl + 1, width=w, teeth=teeth,
             lanes=lanes, sources=sources, destination=dst, pitch=pitch)
    # This also rejects excess teeth, malformed gaps, noise, and extra controls.
    assert np.array_equal(a, _input_from_objects(f))
    return f


def _parse_input(grid):
    a = np.asarray(grid)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    bg = int(a[0, 0])
    assert all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1]))
    candidates = []
    for code in range(8):
        view = _unorient(a, code)
        for wall in _wall_candidates(view, bg):
            try:
                f = _read_canonical(view, bg, wall)
            except (AssertionError, IndexError):
                continue
            f['orientation'] = code
            candidates.append(f)
    assert len(candidates) == 1, f'expected one interpretation, got {len(candidates)}'
    return candidates[0]


def _move_and_count(grid, f):
    """Solver path: move actual input pixels; count completely expelled groups."""
    a = _unorient(np.asarray(grid), f['orientation'])
    out = a.copy()
    bg, roof, base, w = f['background'], f['top'], f['base'], f['width']
    external = np.argwhere((a != bg) & (np.indices(a.shape)[1] < f['left']))
    dx = f['lanes'][0]['x'] - f['key_left']
    out[:, :f['left']] = bg
    # Remove the cup and the source bars.
    rx = f['lanes'][f['destination']]['x']
    out[base - 2, (rx - 1, rx + w)] = bg
    out[base - 1, rx - 1:rx + w + 1] = bg
    for i in f['sources']:
        x = f['lanes'][i]['x']
        out[base + 2, x:x + w] = bg
    for i, lane in enumerate(f['lanes']):
        if i != f['destination']:
            out[roof + 1:base - 2, lane['x']:lane['x'] + w] = bg
    for y, x in external:
        out[y, x + dx] = a[y, x]
    total = 0
    for i, lane in enumerate(f['lanes']):
        if i == f['destination']:
            continue
        x = lane['x']
        # The inserted tooth's top and the original pin bottom determine motion.
        tooth_top = base
        while out[tooth_top - 1, x] == f['structure']:
            tooth_top -= 1
        occupied = np.argwhere(a[roof + 1:base - 2, x:x + w] != bg)
        occupied[:, 0] += roof + 1
        shift = int(occupied[:, 0].max()) - tooth_top + 1
        for y in lane['ys']:
            block = a[y:y + lane['mask'].shape[0], x:x + w]
            cells = np.argwhere(block != bg)
            translated_rows = y + cells[:, 0] - shift
            if i in f['sources'] and np.all(translated_rows <= roof):
                total += 1
            for (dy, dc), row in zip(cells, translated_rows):
                if row > roof:
                    color = f['marker'] if i in f['sources'] else int(block[dy, dc])
                    out[row, x + dc] = color
    receiver = f['lanes'][f['destination']]
    first_y = receiver['ys'][0]
    token = a[first_y:first_y + receiver['mask'].shape[0], rx:rx + w]
    for k in range(1, total + 1):
        y = first_y - k * f['pitch']
        for dy, dc in np.argwhere(token != bg):
            out[y + dy, rx + dc] = token[dy, dc]
    return _orient(out, f['orientation'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from this input alone; no training or seed."""
    return _move_and_count(input_grid, _parse_input(input_grid))


def _mask(rng, height, width):
    # A full left spine with nonempty contiguous row segments is connected.
    # Reflection gives left/right and top/bottom asymmetric shapes as well.
    mask = np.zeros((height, width), dtype=bool)
    for y in range(height):
        mask[y, :int(rng.integers(1, width + 1))] = True
    mask[int(rng.integers(height)), :] = True
    if rng.integers(2):
        mask = mask[:, ::-1]
    if rng.integers(2):
        mask = mask[::-1]
    return mask.copy()


def _counts(num_train, num_test):
    for name, v, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not lo <= v <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def _sample_objects(rng, n, losses, receiver_count, depth_bounds, colors,
                    orientation, demonstration=''):
    max_depth = depth_bounds[1]
    source_count = len(losses)
    assert n >= source_count + 1
    order = rng.permutation(n).tolist()
    dst, sources = order[0], order[1:source_count + 1]
    neutrals = order[source_count + 1:]
    widths = [w for w in (1, 2, 3) if 2 * (n * w + n - 1) + 7 <= 30]
    w = int(rng.choice(widths))
    gaps = [1] * (n - 1)
    # Width is spent on meaningful lane spacing before exterior padding.
    span = n * w + sum(gaps)
    for i in rng.permutation(n - 1):
        extra = int(rng.integers(0, min(2, (30 - 7) // 2 - span) + 1))
        gaps[i] += extra
        span += extra
    total = sum(losses)
    final = receiver_count + total
    source_min_count = max((q + (2 if demonstration == 'partial' else 1) for q in losses), default=1)
    possible_pitches = [p for p in (3, 4, 5)
                        if 4 + (final - 1) * p <= max_depth
                        and 5 + (source_min_count - 1) * p <= max_depth]
    assert possible_pitches
    pitch = int(rng.choice(possible_pitches))
    hmax = min(3, pitch - 1)
    q_by_lane = [0] * n
    for i, q in zip(sources, losses):
        q_by_lane[i] = int(q)
    for i in neutrals:
        q_by_lane[i] = int(rng.integers(0, 3))
    if demonstration == 'partial':
        q_by_lane[neutrals[0]] = int(rng.integers(1, 3))
    elif demonstration == 'aggregate':
        q_by_lane[neutrals[0]] = 0
    # Choose depth after semantic operands. No truncation or silent count repair.
    min_counts = [q + 1 for q in q_by_lane]
    min_counts[dst] = receiver_count
    if demonstration == 'partial':
        min_counts[sources[0]] += 1
    heights = []
    for i in range(n):
        needed = final if i == dst else min_counts[i]
        cap = min(hmax, max_depth - 3 - (needed - 1) * pitch)
        lower = 2 if demonstration == 'partial' and i == sources[0] else 1
        heights.append(int(rng.integers(lower, cap + 1)))
    if demonstration == 'partial':
        heights[dst] = 1
    required = max(heights[dst] + 3 + (final - 1) * pitch,
                   max(heights[i] + 3 + (min_counts[i] - 1) * pitch for i in range(n)))
    assert required <= max_depth
    depth = int(rng.integers(max(depth_bounds[0], required), max_depth + 1))
    count = []
    for i in range(n):
        capacity = 1 + (depth - 3 - heights[i]) // pitch
        count.append(int(rng.integers(min_counts[i], min(capacity, min_counts[i] + 3) + 1)))
    count[dst] = receiver_count
    # Keep the pitch locally observable even in zero-credit/singleton cases.
    if max(count) == 1:
        other = next(i for i in range(n) if i != dst)
        count[other] = 2
    shifts = [0] * n
    teeth = [0] * n
    for i in range(n):
        if i == dst:
            continue
        h, q = heights[i], q_by_lane[i]
        if demonstration == 'partial' and i == sources[0]:
            shift = q * pitch + int(rng.integers(1, h))
        elif demonstration == 'aggregate' and i == neutrals[0]:
            shift = 0
        elif q == 0:
            shift = int(rng.integers(0, h))
        else:
            # All phases with exactly q complete losses: includes gap cuts and
            # partial next tokens. A positive survivor is always retained.
            shift = int(rng.integers((q - 1) * pitch + h, q * pitch + h))
        shifts[i] = shift
        pin_span = (count[i] - 1) * pitch + h
        teeth[i] = depth - 1 - pin_span + shift
        assert 1 <= teeth[i] <= depth - 2
    floor_clearance = int(rng.integers(4, min(5, 27 - depth) + 1))
    # Minimum one-cell border makes background directly visible under rotation.
    vertical_slack = 30 - (depth + floor_clearance + 3)
    top = 1 + int(rng.integers(0, min(2, vertical_slack) + 1))
    bottom_margin = 1 + int(rng.integers(0, min(2, vertical_slack - top + 1) + 1))
    base = top + depth
    bottom = base + floor_clearance
    extras = [0] * 5
    budget = 30 - (2 * span + 7)
    for i in rng.permutation(5):
        extras[i] = int(rng.integers(0, min(2, budget) + 1))
        budget -= extras[i]
    kl = 1 + extras[0]
    left = kl + span + 1 + extras[1]
    x = left + 2 + extras[2]
    xs = [x]
    for gap in gaps:
        x += w + gap
        xs.append(x)
    right = xs[-1] + w + 1 + extras[3]
    bg, structure, marker = colors
    pool = [c for c in range(10) if c not in colors]
    palette = rng.choice(pool, int(rng.integers(1, min(n, len(pool)) + 1)), replace=False)
    payloads = rng.choice(palette, n, replace=True).tolist()
    if demonstration == 'partial':
        payloads[neutrals[0]] = payloads[sources[0]]
    lanes = []
    for i, x in enumerate(xs):
        mask = _mask(rng, heights[i], w)
        start = base - 2 - heights[i] - (count[i] - 1) * pitch if i == dst else top + 1
        ys = [start + k * pitch for k in range(count[i])]
        lanes.append(dict(x=x, mask=mask, ys=ys, color=int(payloads[i]),
                          shift=shifts[i], loss=q_by_lane[i]))
    return dict(shape=(bottom + bottom_margin + 1, right + 2 + extras[4]),
                top=top, bottom=bottom, base=base, left=left, right=right,
                key_left=kl, width=w, span=span, teeth=teeth, lanes=lanes,
                sources=sources, destination=dst, pitch=pitch,
                background=bg, structure=structure, marker=marker, orientation=orientation)


def _grid_key(a):
    a = np.asarray(a)
    return (a.shape, a.astype(np.int8).tobytes())


def _new_pair(f, seen):
    # Payload recolorings supply at least 49 distinct inputs with fixed geometry
    # and role colors, more than the maximum 14 examples. This fallback is finite.
    pool = [c for c in range(10) if c not in (f['background'], f['structure'], f['marker'])]
    original = [lane['color'] for lane in f['lanes']]
    for da in range(len(pool)):
        for db in range(len(pool)):
            for i, lane in enumerate(f['lanes']):
                lane['color'] = pool[(pool.index(original[i]) + (da if i == f['destination'] else db)) % len(pool)]
            a = _orient(_input_from_objects(f), f['orientation'])
            if _grid_key(a) not in seen:
                return {'input': a, 'output': _orient(_output_from_objects(f), f['orientation'])}
    raise AssertionError('finite uniqueness fallback exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    shift_axis = str(rng.choice(('lanes', 'depth')))
    orientation_mode = str(rng.choice(('fixed', 'mixed', 'transfer')))
    fixed_orientation = int(rng.integers(8))
    fixed_colors = tuple(int(c) for c in rng.choice(10, 3, replace=False))
    local_colors = bool(rng.integers(2))
    # Joint arithmetic evidence, including independent receiver operands.
    choices = [(q, a, b) for q in (1, 2) for a in range(1, 4) for b in (1, 2)
               if a != b and a + q != b + 3 and q * b != 3 * a
               and not (2 * a == a + q and 2 * b == b + 3)
               and not (a * q == a + q and b * 3 == b + 3)]
    q, a, b = choices[int(rng.integers(len(choices)))]
    operands = [([q], a, 'partial'), ([1, 2], b, 'aggregate')]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, amount in (('train', num_train), ('test', num_test)):
        for index in range(amount):
            depth_bounds = (12, 20) if shift_axis == 'depth' and split == 'train' else (12, 23)
            if shift_axis == 'depth' and split == 'test':
                depth_bounds = (22, 23)
            demonstration = ''
            if split == 'train' and index < 2:
                losses, receiver, demonstration = operands[index]
                n = int(rng.choice((3, 4))) if index == 0 else 4
            else:
                n = 5 if shift_axis == 'lanes' and split == 'test' else int(rng.integers(2, 5 if split == 'train' else 6))
                nsrc = int(rng.integers(1, min(3, n - 1) + 1))
                # Sum and output capacity restrict counts, not the split label.
                options = [(qs, r) for qs in _loss_vectors(nsrc)
                           for r in range(1, 4) if sum(qs) + r <= 6]
                losses, receiver = options[int(rng.integers(len(options)))]
            orientation = fixed_orientation
            if orientation_mode == 'mixed':
                orientation = (fixed_orientation + (index if index < 2 else int(rng.integers(8)))) % 8
            elif orientation_mode == 'transfer' and split == 'test':
                orientation = (fixed_orientation + int(rng.integers(1, 8))) % 8
            colors = fixed_colors
            if local_colors:
                colors = tuple(int(c) for c in rng.choice(10, 3, replace=False))
            f = _sample_objects(rng, n, losses, receiver, depth_bounds, colors,
                                orientation, demonstration)
            pair = _new_pair(f, seen)
            seen.add(_grid_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    return puzzle


def _loss_vectors(n):
    if n == 0:
        return [()]
    return [(q,) + rest for q in range(4) for rest in _loss_vectors(n - 1)
            if q + sum(rest) <= 5]


def _features(f):
    sources = [f['lanes'][i] for i in f['sources']]
    neutral = [lane for i, lane in enumerate(f['lanes'])
               if i not in f['sources'] and i != f['destination']]
    dst = f['lanes'][f['destination']]
    partial = lambda lane: any(y - lane['shift'] <= f['top'] < y + lane['mask'].shape[0] - 1 - lane['shift']
                               for y in lane['ys'])
    return dict(lanes=len(f['lanes']), sources=len(sources), neutrals=len(neutral),
                total=sum(lane['loss'] for lane in sources), receiver=len(dst['ys']),
                losses=tuple(lane['loss'] for lane in sources),
                counts=tuple(len(lane['ys']) for lane in sources),
                partial=any(partial(lane) for lane in sources),
                neutral_overflow=any(lane['loss'] for lane in neutral),
                stationary=any(lane['shift'] == 0 for lane in neutral),
                repeated_roles=any(a['color'] == b['color'] for a in sources for b in neutral),
                depth=f['base'] - f['top'], width=f['width'], pitch=f['pitch'],
                orientation=f['orientation'], profile=tuple(h for h in f['teeth'] if h),
                masks=tuple((lane['mask'].shape, tuple(lane['mask'].flat)) for lane in f['lanes']),
                colors=(f['background'], f['structure'], f['marker']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training, tests = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = np.asarray(pair['input']), np.asarray(pair['output'])
            for grid in (a, b):
                assert grid.ndim == 2 and all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9))
            assert a.shape == b.shape and not np.array_equal(a, b)
            assert _grid_key(a) not in seen
            seen.add(_grid_key(a))
            f = _parse_input(a)
            assert np.array_equal(_move_and_count(a, f), b), 'wrong movement, credit, recoloring, or erasure'
            view = _unorient(b, f['orientation'])
            # Count actual output receiver components, and check preservation of
            # roof/floor and of unmarked pin colors independently of the renderer.
            dst = f['lanes'][f['destination']]
            x, w = dst['x'], f['width']
            groups = _runs(np.any(view[f['top'] + 1:f['base'] - 2, x:x + w] != f['background'], axis=1))
            expected = len(dst['ys']) + sum(f['lanes'][i]['loss'] for i in f['sources'])
            assert len(groups) == expected
            assert np.all(view[f['top'], f['left']:f['right'] + 1] == f['structure'])
            assert np.all(view[f['bottom'], f['left']:f['right'] + 1] == f['structure'])
            for i, lane in enumerate(f['lanes']):
                if i == f['destination']:
                    continue
                section = view[f['top'] + 1:f['base'] - f['teeth'][i], lane['x']:lane['x'] + w]
                colors = set(int(c) for c in section.flat) - {f['background']}
                assert colors == {f['marker'] if i in f['sources'] else lane['color']}
            (training if split == 'train' else tests).append(_features(f))
    # Collective evidence is required even for the minimum two demonstrations.
    assert any(d['partial'] and d['neutral_overflow'] and d['repeated_roles'] for d in training)
    assert any(len(d['losses']) >= 2 and min(d['losses']) > 0 and len(set(d['losses'])) > 1
               and d['stationary'] for d in training)
    assert len({d['total'] for d in training}) >= 2
    assert len({d['receiver'] for d in training}) >= 2
    assert len({d['total'] + d['receiver'] for d in training}) >= 2
    assert len({(d['total'], d['receiver']) for d in training}) >= 2
    first = training[0]
    assert any(d['total'] * first['receiver'] != first['total'] * d['receiver'] for d in training)
    rivals = (lambda d: 2 * d['receiver'], lambda d: d['receiver'] + len(d['losses']),
              lambda d: d['receiver'] + max(d['losses']), lambda d: d['receiver'] + sum(d['counts']),
              lambda d: d['receiver'] * d['total'], lambda d: d['total'])
    assert all(any(rival(d) != d['receiver'] + d['total'] for d in training) for rival in rivals)
    # Every puzzle has an actual size extrapolation, not just a nominal mode.
    assert (all(d['lanes'] > max(t['lanes'] for t in training) for d in tests)
            or all(d['depth'] > max(t['depth'] for t in training) for d in tests))


def _puzzle_key(puzzle):
    return tuple(_grid_key(p[k]) for split in ('train', 'test')
                 for p in puzzle[split] for k in ('input', 'output'))


def self_test():
    """Bounded checks: 42 diverse seeds, count API, symmetries, and mutations."""
    shapes, structures, profiles, masks, pitches, widths, colors, orientations = (set() for _ in range(8))
    all_features, keys = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        keys.add(_puzzle_key(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _parse_input(pair['input'])
                d = _features(f)
                all_features.append(d)
                shapes.add(pair['input'].shape)
                roles = {}
                structures.add(tuple(roles.setdefault(int(c), len(roles)) for c in pair['input'].flat))
                profiles.add(d['profile'])
                masks.update(d['masks'])
                pitches.add(d['pitch'])
                widths.add(d['width'])
                colors.add(d['colors'])
                orientations.add(d['orientation'])
    assert len(keys) == 42 and len(shapes) >= 25 and len(structures) >= 150
    assert len(profiles) >= 70 and len(masks) >= 15
    assert pitches == {3, 4, 5} and widths == {1, 2, 3} and orientations == set(range(8))
    assert len(colors) >= 30 and {d['sources'] for d in all_features} >= {1, 2, 3}
    assert {d['lanes'] for d in all_features} >= {2, 3, 4, 5}
    assert any(d['total'] == 0 for d in all_features)
    assert any(len(set(d['profile'])) < len(d['profile']) for d in all_features)
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(generate_puzzle(0))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(89, a, b)
            assert len(p['train']) == a and len(p['test']) == b
            validate_generated_puzzle(p)
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    puzzle = generate_puzzle(0, 2, 1)
    pair = puzzle['train'][0]
    for code in range(8):
        assert np.array_equal(solve(_orient(pair['input'], code)), _orient(pair['output'], code))
    # Whole-scene recoloring must not change the operation.
    mapping = np.array([7, 2, 8, 0, 9, 4, 1, 6, 3, 5])
    assert np.array_equal(solve(mapping[pair['input']]), mapping[pair['output']])
    # Check that the validator catches receiver arithmetic, source processing,
    # marker erasure, and neutral preservation defects.
    for defect in ('receiver', 'source', 'marker', 'neutral'):
        p = {s: [{k: v.copy() for k, v in item.items()} for item in pairs] for s, pairs in puzzle.items()}
        f = _parse_input(p['train'][0]['input'])
        out = _unorient(p['train'][0]['output'], f['orientation'])
        if defect == 'receiver':
            lane = f['lanes'][f['destination']]
            y = lane['ys'][0] - f['pitch']
            dy, dx = np.argwhere(lane['mask'])[0]
            out[y + dy, lane['x'] + dx] = f['background']
        elif defect == 'marker':
            x = f['lanes'][f['sources'][0]]['x']
            out[f['base'] + 2, x] = f['marker']
        else:
            i = f['sources'][0] if defect == 'source' else next(i for i in range(len(f['lanes'])) if i not in f['sources'] and i != f['destination'])
            lane = f['lanes'][i]
            y = lane['ys'][-1] - lane['shift']
            dy, dx = np.argwhere(lane['mask'])[-1]
            out[y + dy, lane['x'] + dx] = lane['color'] if defect == 'source' else f['marker']
        p['train'][0]['output'] = _orient(out, f['orientation'])
        try:
            validate_generated_puzzle(p)
        except AssertionError:
            pass
        else:
            raise AssertionError('accepted incorrect ' + defect)
    return {'seeds': 42, 'pairs': len(all_features), 'shapes': len(shapes),
            'structures': len(structures), 'profiles': len(profiles), 'masks': len(masks),
            'pitches': sorted(pitches), 'widths': sorted(widths), 'orientations': sorted(orientations)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
