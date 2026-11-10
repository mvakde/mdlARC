"""Sliding phase lock: contact motion changes fixed-probe/cadence alignment.

Only NumPy and the standard library are used. No parent module, file, seed,
example cache, or generator state participates in solve().
"""

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.tobytes()


def _orient(a, code):
    return np.rot90(np.fliplr(a) if code >= 4 else a, code % 4).copy()


def _active(lane, displacement, capacity):
    probe = lane['probe']
    if probe is None:
        return False
    t = probe['t']
    u, v = lane['a'] - displacement, lane['b'] - displacement
    return (0 <= t < capacity and (t - u) % (v - u) == 0
            and (t < u or t > v))


def _frame(f):
    a = np.full(f['shape'], f['bg'], dtype=np.int8)
    t, floor, left, right = f['top'], f['floor'], f['left'], f['right']
    s = f['s']
    a[t, left:right + 1] = s
    a[t:floor + 1, left] = a[t:floor + 1, right] = s
    first, last = f['cols'][0], f['cols'][-1]
    a[floor, left:first] = a[floor, last + 1:right + 1] = s
    for col, lane in zip(f['cols'], f['lanes']):
        probe = lane['probe']
        if probe is not None:
            a[t + 1 + probe['t'], col + probe['side']] = probe['color']
    return a


def _input(f):
    a = _frame(f)
    first, last = f['cols'][0], f['cols'][-1]
    a[f['base'], first:last + 1] = f['s']
    for col, lane in zip(f['cols'], f['lanes']):
        a[f['base'] - lane['h']:f['base'], col] = f['s']
        start = f['top'] + 1
        a[start:start + lane['L'], col] = f['s']
        a[start + lane['a'], col] = a[start + lane['b'], col] = lane['ref']
    return a


def _label(f, omit_pattern=False, omit_transport=False):
    """Direct geometric renderer used for generation and ablation diagnostics."""
    out = _frame(f)
    t, floor, s = f['top'], f['floor'], f['s']
    out[floor, f['cols'][0]:f['cols'][-1] + 1] = s
    for col, lane in zip(f['cols'], f['lanes']):
        c = floor - t - lane['h'] - 1
        d = lane['L'] - c
        out[floor - lane['h']:floor, col] = s
        out[t + 1:t + 1 + c, col] = s
        if omit_pattern:
            for x in (lane['a'] - d, lane['b'] - d):
                if 0 <= x < c:
                    out[t + 1 + x, col] = lane['ref']
            continue
        shift = 0 if omit_transport else d
        u, v = lane['a'] - shift, lane['b'] - shift
        p = v - u
        active = _active(lane, shift, c)
        lo, hi = (min(u, lane['probe']['t']), max(v, lane['probe']['t'])) if active else (0, c - 1)
        color = lane['probe']['color'] if active else lane['ref']
        for x in range(c):
            if lo <= x <= hi and (x - u) % p == 0:
                out[t + 1 + x, col] = color
    return out


def _runs(row, bg):
    breaks = np.flatnonzero(row[1:] != row[:-1]) + 1
    ends = [0, *breaks.tolist(), len(row)]
    return [(int(row[l]), l, r - 1) for l, r in zip(ends, ends[1:])
            if r - l >= 8 and int(row[l]) != bg]


def _read_canonical(a, bg, top, s, left, right):
    """Recover a complete scene from one putative roof; reject extra pixels."""
    height, width = a.shape
    ends = []
    for col in (left, right):
        row = top
        while row + 1 < height and a[row + 1, col] == s:
            row += 1
        ends.append(row)
    assert ends[0] == ends[1]
    floor = ends[0]
    span = floor - top
    assert span >= 6
    opening = np.flatnonzero(a[floor, left:right + 1] == bg) + left
    assert len(opening) >= 4
    first, last = int(opening[0]), int(opening[-1])
    assert first - left >= 2 and right - last >= 2
    assert np.array_equal(opening, np.arange(first, last + 1))
    below = np.argwhere(a[floor + 1:] != bg)
    assert len(below)
    base = floor + 1 + int(below[:, 0].max())
    assert np.all(a[base, first:last + 1] == s)
    cols = (np.flatnonzero(a[base - 1, first:last + 1] == s) + first).tolist()
    assert 2 <= len(cols) <= 8 and cols[0] == first and cols[-1] == last
    assert np.all(np.diff(cols) >= 3)
    lanes = []
    for col in cols:
        row = base - 1
        while row >= 0 and a[row, col] == s:
            row -= 1
        h = base - row - 1
        assert h > 0 and base - h > floor
        capacity = span - h - 1
        assert capacity >= 3
        segment = a[top + 1:floor, col]
        occupied = np.flatnonzero(segment != bg)
        length = len(occupied)
        assert length >= 2 and np.array_equal(occupied, np.arange(length))
        marks = np.flatnonzero(segment[:length] != s)
        assert len(marks) == 2
        ref = int(segment[marks[0]])
        assert ref not in (bg, s) and segment[marks[1]] == ref
        displacement = length - capacity
        assert 0 <= displacement <= h
        probes = []
        for side in (-1, 1):
            for x in np.flatnonzero(a[top + 1:floor, col + side] != bg):
                color = int(a[top + 1 + x, col + side])
                assert color not in (bg, s, ref)
                probes.append(dict(t=int(x), side=side, color=color))
        assert len(probes) <= 1
        lanes.append(dict(h=h, L=length, a=int(marks[0]), b=int(marks[1]),
                          ref=ref, probe=probes[0] if probes else None))
    f = dict(shape=a.shape, bg=bg, s=s, top=top, floor=floor, left=left,
             right=right, base=base, cols=cols, lanes=lanes)
    assert np.array_equal(a, _input(f)), "unexplained pixel or malformed geometry"
    return f


def _parse(grid):
    a = _grid(grid)
    colors, counts = np.unique(a, return_counts=True)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must be uniquely most frequent")
    bg = int(colors[np.argmax(counts)])
    candidates = []
    # A reflected canonical scene is itself canonical: gaps and probe sides
    # have no handedness convention. Only four roof directions are needed.
    for turn in range(4):
        canonical = np.rot90(a, -turn)
        for top, row in enumerate(canonical):
            for s, left, right in _runs(row, bg):
                try:
                    f = _read_canonical(canonical, bg, top, s, left, right)
                except (AssertionError, IndexError):
                    continue
                f['turn'] = turn
                candidates.append(f)
    if len(candidates) != 1:
        raise ValueError("expected one roof/opening/key interpretation")
    return candidates[0]


def _simulate(a, f):
    """Move actual rod cells stepwise; infer cadence from their input marks."""
    result = a.copy()
    bg, s, top, floor = f['bg'], f['s'], f['top'], f['floor']
    external = np.argwhere(a[floor + 1:] != bg)
    for rr, col in external:
        result[floor + 1 + rr, col] = bg
    for col, lane in zip(f['cols'], f['lanes']):
        cells = [(int(r), int(a[r, col])) for r in range(top + 1, floor)
                 if a[r, col] != bg]
        result[top + 1:floor, col] = bg
        tooth_top = floor - lane['h']
        steps = 0
        while cells[-1][0] >= tooth_top:
            cells = [(r - 1, color) for r, color in cells if r - 1 > top]
            steps += 1
        assert cells and cells[-1][0] == tooth_top - 1
        for row, color in cells:
            result[row, col] = color
        # Complete on surviving material, after motion; probes stay put.
        mark_rows = [r for r in range(top + 1, floor) if a[r, col] == lane['ref']]
        u, v = mark_rows[0] - steps, mark_rows[1] - steps
        period = v - u
        probe = lane['probe']
        active = False
        if probe is not None:
            pr = top + 1 + probe['t']
            active = (top < pr < tooth_top and (pr - u) % period == 0
                      and (pr < u or pr > v))
        low, high = (min(u, pr), max(v, pr)) if active else (top + 1, tooth_top - 1)
        paint = probe['color'] if active else lane['ref']
        for row, _ in cells:
            result[row, col] = paint if low <= row <= high and (row - u) % period == 0 else s
    # Translate the entire observed external key, including its base.
    for rr, col in external:
        row = floor + 1 + rr
        result[row - (f['base'] - floor), col] = a[row, col]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    f = _parse(a)
    canonical = np.rot90(a, -f['turn'])
    return np.rot90(_simulate(canonical, f), f['turn']).copy()


def _pick(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _profile(rng, n, cap):
    mode = _pick(rng, ('flat', 'rising', 'falling', 'levels', 'valley', 'plateau', 'random'))
    h = rng.integers(1, cap + 1, n).tolist()
    if mode == 'flat':
        h = [h[0]] * n
    elif mode in ('rising', 'falling'):
        h.sort(reverse=mode == 'falling')
    elif mode == 'levels':
        lo, hi = sorted(rng.choice(np.arange(1, cap + 1), 2, replace=False).tolist())
        h = [lo, hi] + rng.choice([lo, hi], n - 2).tolist()
        rng.shuffle(h)
    elif mode == 'valley' and n >= 3:
        h[0] = h[-1] = cap
        h[n // 2] = 1
    elif mode == 'plateau' and n >= 3:
        start = int(rng.integers(n - 1))
        h[start:start + 2] = [cap, cap]
    return h


def _probe(rng, lane, capacity, span, desired=None):
    choices = []
    for t in range(span - 1):
        temporary = dict(lane, probe=dict(t=t))
        if _active(temporary, lane['L'] - capacity, capacity) == desired:
            choices.append(t)
    if not choices:
        return None
    return dict(t=int(_pick(rng, choices)), side=int(_pick(rng, (-1, 1))), color=0)


def _sample(rng, settings, split, witness=None):
    shift = settings['shift']
    if shift == 'lanes':
        n = int(rng.integers(3, 5) if split == 'train' else rng.integers(5, 8))
    else:
        n = int(rng.integers(3, 6) if witness == 'switch' else rng.integers(2, 8))
    top = int(rng.integers(3))
    gap = int(rng.integers(1, 4))
    span = int(rng.integers(11, 17) if witness == 'switch' else rng.integers(9, 19))
    if split == 'test' and shift == 'period':
        span = int(rng.integers(14, 19))
    cap = min(span - 5, 28 - top - span - gap)
    # The height budget reserves a row for the exterior base; cap >= 3.
    assert cap >= 3
    heights = _profile(rng, n, cap)
    order = rng.permutation(n).tolist()
    if witness == 'switch':
        for i in order[:3]:
            heights[i] = int(rng.integers(1, min(cap, span - 9) + 1))
    if witness == 'clip':
        heights[order[0]] = int(rng.integers(2, cap + 1))
    if split == 'test' and shift == 'erased':
        heights[order[0]] = int(rng.integers(3, cap + 1))
    if split == 'test' and shift == 'period':
        heights[order[0]] = int(rng.integers(1, min(cap, span - 9) + 1))
    left_margin, right_margin = int(rng.integers(3)), int(rng.integers(3))
    il, ir = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    gaps = [3] * (n - 1)
    slack = 30 - (left_margin + right_margin + il + ir + 1 + sum(gaps))
    for i in rng.permutation(n - 1):
        extra = int(rng.integers(min(2, slack) + 1))
        gaps[i] += extra
        slack -= extra
    left = left_margin
    cols = [left + il]
    for distance in gaps:
        cols.append(cols[-1] + distance)
    right = cols[-1] + ir
    floor = top + span
    base = floor + max(heights) + gap
    bottom_margin = int(rng.integers(min(2, 29 - base) + 1))
    bg, s = settings['colors'] if settings['fixed_colors'] else tuple(int(v) for v in rng.choice(10, 2, replace=False))
    available = [v for v in range(10) if v not in (bg, s)]
    palette = rng.choice(available, int(rng.integers(1, min(6, n + 1) + 1)), replace=False).tolist()
    lanes = []
    probe_mode = _pick(rng, ('none', 'mixed', 'mixed', 'all'))
    contact_mode = _pick(rng, ('mixed', 'random', 'stationary', 'moving'))
    for h in heights:
        c = span - h - 1
        d = int(rng.integers(h + 1))
        if contact_mode == 'stationary':
            d = 0
        elif contact_mode == 'moving':
            d = int(rng.integers(1, h + 1))
        length = c + d
        pcap = min(8, c // 2)
        if split == 'train' and shift == 'period':
            pcap = min(pcap, 3)
        p = int(rng.integers(1, pcap + 1))
        # In the erased-reference transfer, training never removes both marks.
        # One lost mark is taught explicitly; complete loss is held out.
        amin = max(0, d - p) if split == 'train' and shift == 'erased' else 0
        a = int(rng.integers(amin, length - p))
        lane = dict(h=h, L=length, a=a, b=a + p, ref=int(_pick(rng, palette)), probe=None)
        if probe_mode == 'all' or (probe_mode == 'mixed' and rng.random() < .65):
            lane['probe'] = _probe(rng, lane, c, span, desired=bool(rng.integers(2)))
        lanes.append(lane)
    if witness == 'switch':
        still, activate, deactivate = order[:3]
        lane = lanes[still]
        c = span - lane['h'] - 1
        a = int(rng.integers(c - 1))
        lane.update(L=c, a=a, b=a + 1, probe=None)
        for i, activating in ((activate, True), (deactivate, False)):
            lane = lanes[i]
            c = span - lane['h'] - 1
            # Leave a cadence site above the pair to expose bounded stopping
            # and bilateral extrapolation. Vary period, offset, and motion.
            eligible = [p for p in (2, 3) if c >= 3 * p + (1 if activating else 2)]
            p = int(_pick(rng, eligible))
            dcap = min(lane['h'], p - 1, c - 1 - 3 * p) if not activating else min(lane['h'], p - 1)
            d = int(rng.integers(1, dcap + 1))
            umax = c - 1 - 2 * p - (0 if activating else d)
            u = int(rng.integers(p, umax + 1))
            t = u + 2 * p + (0 if activating else d)
            lane.update(L=c + d, a=u + d, b=u + d + p,
                        probe=dict(t=t, side=int(_pick(rng, (-1, 1))), color=0))
    elif witness == 'clip':
        lane = lanes[order[0]]
        c = span - lane['h'] - 1
        d = int(rng.integers(2, lane['h'] + 1))
        lane.update(L=c + d, a=d - 1, b=d + 1, probe=None)
    if split == 'test' and shift == 'erased':
        lane = lanes[order[0]]
        c = span - lane['h'] - 1
        d = int(rng.integers(3, lane['h'] + 1))
        lane.update(L=c + d, a=d - 3, b=d - 1, probe=None)
    elif split == 'test' and shift == 'period':
        lane = lanes[order[0]]
        c = span - lane['h'] - 1
        p = int(rng.integers(4, min(6, c // 2) + 1))
        a = int(rng.integers(lane['L'] - p))
        lane.update(a=a, b=a + p, probe=None)
    for lane in lanes:
        if lane['probe'] is not None:
            other = [v for v in available if v != lane['ref']]
            lane['probe']['color'] = int(_pick(rng, other))
    f = dict(shape=(base + bottom_margin + 1, right + right_margin + 1),
             top=top, floor=floor, base=base, left=left, right=right,
             cols=cols, lanes=lanes, bg=bg, s=s)
    code = settings['orientation']
    if shift == 'orientation' and split == 'test':
        code = (code // 4) * 4 + (code + 1) % 4
    elif settings['mixed_orientation'] and shift != 'orientation':
        code = int(rng.integers(8))
    return {'input': _orient(_input(f), code), 'output': _orient(_label(f), code)}


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    settings = dict(shift=_pick(rng, ('erased', 'period', 'lanes', 'orientation')),
                    colors=tuple(int(v) for v in rng.choice(10, 2, replace=False)),
                    fixed_colors=bool(rng.random() < .3), orientation=int(rng.integers(8)),
                    mixed_orientation=bool(rng.random() < .6))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            witness = ('switch', 'clip')[index] if split == 'train' and index < 2 else None
            for _ in range(100):
                pair = _sample(rng, settings, split, witness)
                key = _key(pair['input'])
                if key not in seen:
                    # Colors can make the structure unusually dense; require
                    # the background convention to be visible in actual cells.
                    try:
                        _parse(pair['input'])
                    except ValueError:
                        continue
                    break
            else:
                raise RuntimeError('bounded distinct-input construction exhausted')
            seen.add(key)
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    return puzzle


def _features(f):
    records = []
    span = f['floor'] - f['top']
    for lane in f['lanes']:
        c = span - lane['h'] - 1
        d = lane['L'] - c
        p = lane['b'] - lane['a']
        u, v = lane['a'] - d, lane['b'] - d
        active = _active(lane, d, c)
        before = _active(lane, 0, lane['L'])
        phase = [x for x in range(c) if (x - u) % p == 0]
        bounded_stop = False
        if active:
            lo, hi = min(u, lane['probe']['t']), max(v, lane['probe']['t'])
            bounded_stop = any(x < lo or x > hi for x in phase)
        records.append(dict(period=p, shift=d, capacity=c, active=active,
                            activated=active and not before, deactivated=before and not active,
                            clipped=int(u < 0) + int(v < 0), bounded_stop=bounded_stop,
                            bilateral=not active and any(x < u for x in phase) and any(x > v for x in phase),
                            probe=lane['probe'] is not None,
                            probe_side=lane['probe']['side'] if lane['probe'] else 0))
    return records


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, demonstrations = set(), []
    ablations = [False, False]
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and not np.array_equal(a, b)
            key = _key(a)
            assert key not in seen
            seen.add(key)
            f = _parse(a)
            canonical = np.rot90(a, -f['turn'])
            expected = np.rot90(b, -f['turn'])
            assert np.array_equal(_simulate(canonical, f), expected), 'incorrect contact/phase result'
            assert np.array_equal(_label(f), expected), 'formula and stepwise paths disagree'
            t, floor = f['top'], f['floor']
            # Fixed hardware and probes must survive exactly; exterior removed.
            frame = _frame(f)
            fixed = frame != f['bg']
            assert np.array_equal(expected[fixed], frame[fixed])
            assert np.all(expected[floor + 1:] == f['bg'])
            for col, lane in zip(f['cols'], f['lanes']):
                c = floor - t - lane['h'] - 1
                assert np.all(expected[t + 1:t + 1 + c, col] != f['bg'])
                assert np.all(expected[floor - lane['h']:floor + 1, col] == f['s'])
            if split == 'train':
                demonstrations.extend(_features(f))
                ablations[0] |= not np.array_equal(expected, _label(f, omit_pattern=True))
                ablations[1] |= not np.array_equal(expected, _label(f, omit_transport=True))
    for field in ('activated', 'deactivated', 'bounded_stop', 'bilateral'):
        assert any(d[field] for d in demonstrations), f'missing training evidence: {field}'
    assert any(d['shift'] == 0 for d in demonstrations)
    assert any(d['shift'] > 0 for d in demonstrations)
    assert any(d['clipped'] == 1 for d in demonstrations)
    assert any(d['period'] == 1 for d in demonstrations)
    assert any(d['period'] > 1 for d in demonstrations)
    assert all(ablations), 'omitting either parent must alter a training answer'


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed sweep, targeted counterfactuals, and public count checks."""
    # Literal expected carrier sequences for activation, zero contact, and a
    # clipped mark. These are not predictions made by the label renderer.
    fixture = dict(shape=(18, 13), bg=0, s=5, top=1, floor=12, base=16,
                   left=1, right=11, cols=[3, 6, 9], lanes=[
                       dict(h=1, L=9, a=4, b=5, ref=2, probe=None),
                       dict(h=2, L=9, a=3, b=5, ref=3,
                            probe=dict(t=6, side=1, color=4)),
                       dict(h=3, L=10, a=2, b=4, ref=6,
                            probe=dict(t=0, side=-1, color=7))])
    answer = solve(_input(fixture))
    assert answer[2:11, 3].tolist() == [2] * 9
    assert answer[2:10, 6].tolist() == [5, 5, 4, 5, 4, 5, 4, 5]
    assert answer[2:9, 9].tolist() == [5, 6, 5, 6, 5, 6, 5]
    assert answer[8, 7] == 4 and answer[2, 8] == 7
    assert np.all(answer[13:] == 0)
    malformed = _input(fixture)
    malformed[1, 2] = 0
    try:
        solve(malformed)
    except ValueError:
        pass
    else:
        raise AssertionError('broken roof accepted')
    signatures, shapes, layouts, palettes, profiles, periods = set(), set(), set(), set(), set(), set()
    orientations, backgrounds, counts, records = set(), set(), set(), []
    shifts = set()
    for seed in range(42):
        nt = 1 + seed % 4
        nr = nt + 1 + (seed // 4) % (10 - nt)
        puzzle = generate_puzzle(seed, nr, nt)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        counts.add((nr, nt))
        readings = {s: [_parse(p['input']) for p in puzzle[s]] for s in puzzle}
        transfer = _pick(np.random.default_rng(seed), ('erased', 'period', 'lanes', 'orientation'))
        shifts.add(transfer)
        if transfer == 'erased':
            assert all(d['clipped'] <= 1 for f in readings['train'] for d in _features(f))
            assert all(any(d['clipped'] == 2 for d in _features(f)) for f in readings['test'])
        elif transfer == 'period':
            assert max(d['period'] for f in readings['train'] for d in _features(f)) <= 3
            assert all(any(d['period'] >= 4 for d in _features(f)) for f in readings['test'])
        elif transfer == 'lanes':
            assert max(len(f['lanes']) for f in readings['train']) <= 4
            assert min(len(f['lanes']) for f in readings['test']) >= 5
        else:
            train_turns = {f['turn'] for f in readings['train']}
            assert len(train_turns) == 1
            assert all(f['turn'] not in train_turns for f in readings['test'])
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                original = a.copy()
                assert np.array_equal(solve(a), pair['output'])
                assert np.array_equal(a, original), 'solve must not mutate input'
                f = _parse(a)
                shapes.add(a.shape)
                orientations.add(f['turn'])
                backgrounds.add(f['bg'])
                palettes.add(tuple(np.unique(a).tolist()))
                profiles.add(tuple(lane['h'] for lane in f['lanes']))
                layouts.add((f['shape'], tuple(np.diff(f['cols'])), f['floor'] - f['top']))
                features = _features(f)
                records.extend(features)
                periods.update(d['period'] for d in features)
    assert len(signatures) == 42
    assert shifts == {'erased', 'period', 'lanes', 'orientation'}
    assert len(shapes) >= 60 and len(layouts) >= 100 and len(profiles) >= 100
    assert backgrounds == set(range(10)) and orientations == set(range(4))
    assert len(palettes) >= 40 and {1, 2, 3, 4, 5, 6} <= periods
    for field in ('active', 'activated', 'deactivated', 'bounded_stop', 'bilateral'):
        assert sum(d[field] for d in records) >= 42, field
    assert {d['clipped'] for d in records} == {0, 1, 2}
    assert {d['probe_side'] for d in records} == {-1, 0, 1}
    assert any(d['shift'] >= 5 for d in records)
    # Full public count matrix, including the smallest demonstrations.
    for nr in range(2, 11):
        for nt in range(1, min(4, nr - 1) + 1):
            p = generate_puzzle(1000 + nr * 10 + nt, nr, nt)
            assert (len(p['train']), len(p['test'])) == (nr, nt)
            validate_generated_puzzle(p)
    for nr, nt in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nr, nt)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    assert _puzzle_key(generate_puzzle(312, 2, 1)) == _puzzle_key(generate_puzzle(312, 2, 1))
    # Test the input-only path under every rigid symmetry and a role permutation.
    fixture = generate_puzzle(77, 2, 1)['train'][0]
    permutation = np.array([3, 8, 1, 6, 9, 2, 0, 5, 7, 4], dtype=np.int8)
    for code in range(8):
        a = _orient(permutation[fixture['input']], code)
        b = _orient(permutation[fixture['output']], code)
        assert np.array_equal(solve(a), b)
    return dict(seeds=42, distinct_puzzles=len(signatures), input_shapes=len(shapes),
                layouts=len(layouts), profiles=len(profiles), palettes=len(palettes),
                periods=sorted(periods), lanes_checked=len(records),
                activated=sum(d['activated'] for d in records),
                deactivated=sum(d['deactivated'] for d in records),
                both_references_clipped=sum(d['clipped'] == 2 for d in records),
                train_test_transfers=sorted(shifts),
                legal_count_combinations=sum(min(4, nr - 1) for nr in range(2, 11)),
                independent_solver=False)
