"""Occluded diagonal trellis: infer hidden finite bridges, then fire sensors.

No file I/O, example cache, or random state is used by solve. Ground truth is
painted with strokes; the input-side model uses coordinate equations.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _corners(bb):
    r0, r1, c0, c1 = bb
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _annotations(a):
    candidates = []
    for color in np.unique(a):
        cells = np.argwhere(a == color)
        if len(cells) != 6:
            continue
        r0, c0 = map(int, cells.min(axis=0))
        r1, c1 = map(int, cells.max(axis=0))
        bb = (r0, r1, c0, c1)
        if min(r1 - r0, c1 - c0) < 2:
            continue
        if set(map(tuple, cells)) == _corners(bb):
            candidates.append((int(color), bb))
    assert len(candidates) == 1, 'Two unique canonical L annotations required'
    return candidates[0]


def _linear(values):
    d = values[1] - values[0]
    assert all(v == values[0] + i * d for i, v in enumerate(values))
    return values[0], d


def _geometry(model, i):
    y, x, length, dy, dl, gap, slope, palette = model
    return (y + i * dy, x + i * (length + gap) + i * (i - 1) // 2 * dl,
            length + i * dl, slope, palette[i % len(palette)])


def _on(point, bridge, finite=True):
    r, c = point
    y, x, length, slope, _ = bridge
    return r - y == slope * (c - x) and (not finite or 0 <= c - x < length)


def _equation_render(bb, background, signal, bridges, pins, *, fire=True,
                     all_pins=False, infinite=False, endpoints_only=False):
    """Input-side prediction and targeted counterfactuals, in global coordinates."""
    r0, r1, c0, c1 = bb
    rr, cc = np.indices((r1 - r0 + 1, c1 - c0 + 1))
    rr, cc = rr + r0, cc + c0
    out = np.full(rr.shape, background, dtype=np.int8)
    for y, x, length, slope, color in bridges:
        on = (rr - y == slope * (cc - x)) & (cc >= x) & (cc < x + length)
        if endpoints_only:
            on &= (cc == x) | (cc == x + length - 1)
        out[on] = color
    active = [p for p in pins if all_pins or any(_on(p, b, not infinite) for b in bridges)]
    if fire:
        # All bridges in an example are parallel. Signals have the other slope.
        slope = bridges[0][3]
        for r, c in active:
            out[rr - r == -slope * (cc - c)] = signal
    for r, c in pins:
        out[r - r0, c - c0] = signal
    return out, active


def _axis_model(a, bb, endpoints, background, signal):
    """Pair same-colored dots before fitting the object progression."""
    bridges = []
    for color in endpoints:
        points = [tuple(map(int, p)) for p in np.argwhere(a == color)]
        degree = dict.fromkeys(points, 0)
        for j, p in enumerate(points):
            for q in points[j + 1:]:
                if abs(p[0] - q[0]) == abs(p[1] - q[1]):
                    degree[p] += 1
                    degree[q] += 1
                    left, right = sorted((p, q), key=lambda v: v[1])
                    length = right[1] - left[1] + 1
                    assert length >= 3
                    slope = 1 if right[0] > left[0] else -1
                    bridges.append((*left, length, slope, color))
        assert points and all(v == 1 for v in degree.values()), 'Endpoint partners must be unique'
    bridges.sort(key=lambda v: v[1])
    n = len(bridges)
    assert 3 <= n <= 6
    assert len({b[3] for b in bridges}) == 1
    assert all(a0[1] + a0[2] < a1[1] for a0, a1 in zip(bridges, bridges[1:]))
    ys, xs, lengths, colors = ([b[k] for b in bridges] for k in (0, 1, 2, 4))
    y, dy = _linear(ys)
    length, dl = _linear(lengths)
    gaps = [xs[i + 1] - xs[i] - lengths[i] for i in range(n - 1)]
    assert len(set(gaps)) == 1 and gaps[0] >= 1
    cycles = [p for p in range(1, 4) if n >= max(3, 2 * p)
              and len(set(colors[:p])) == p
              and all(c == colors[i % p] for i, c in enumerate(colors))]
    assert cycles
    palette = tuple(colors[:min(cycles)])
    model = (y, xs[0], length, dy, dl, gaps[0], bridges[0][3], palette)
    r0, r1, c0, c1 = bb
    if c0 > xs[-1] + lengths[-1] - 1:
        indices = range(n, 64)
        forward = True
    else:
        assert c1 < xs[0]
        indices = range(-1, -64, -1)
        forward = False
    hidden = []
    for i in indices:
        b = _geometry(model, i)
        if (forward and b[1] > c1) or (not forward and b[1] + b[2] - 1 < c0):
            break
        assert b[2] >= 3, 'Continuation may not cross a degenerate bridge'
        if b[1] <= c1 and b[1] + b[2] > c0:
            hidden.append((i, b))
    assert hidden
    pins = [tuple(map(int, p)) for p in np.argwhere(a == signal)]
    pred, active = _equation_render(bb, background, signal, [b for i, b in hidden], pins)
    return pred, dict(model=model, visible=bridges, hidden=hidden, pins=pins,
                      active=active, bb=bb, background=background, signal=signal)


def _analyze(value):
    a = _grid(value)
    marker, bb = _annotations(a)
    vals, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    background = int(vals[np.argmax(counts)])
    assert background != marker and np.count_nonzero(a == background) > a.size / 2
    r0, r1, c0, c1 = bb
    inside = a[r0:r1 + 1, c0:c1 + 1]
    signal_colors = set(map(int, np.unique(inside))) - {background, marker}
    assert len(signal_colors) == 1, 'One surviving sensor color inside the window'
    signal = signal_colors.pop()
    outside = a.copy()
    outside[r0:r1 + 1, c0:c1 + 1] = background
    assert not np.any(outside == signal), 'Sensors belong to the requested window'
    endpoints = set(map(int, np.unique(outside))) - {background, marker}
    assert 1 <= len(endpoints) <= 3
    assert np.all(np.isin(inside, [background, marker, signal]))
    answers = []
    for transpose in (False, True):
        canon = a.T if transpose else a
        bounds = (c0, c1, r0, r1) if transpose else bb
        try:
            pred, info = _axis_model(canon, bounds, endpoints, background, signal)
        except AssertionError:
            continue
        answers.append((pred.T.copy() if transpose else pred, info, transpose))
    assert answers, 'No supported endpoint progression'
    assert all(np.array_equal(v[0], answers[0][0]) for v in answers), 'Ambiguous continuation'
    pred, info, transpose = answers[0]
    info.update(marker=marker, transpose=transpose, bounds=bb)
    return pred, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the crop solely from the original input; never fire new crossings."""
    return _analyze(input_grid)[0]


def _stroke_truth(shape, bb, background, signal, bridges, pins):
    """Generation labels: explicit strokes, then full diagonal signal walks."""
    canvas = np.full(shape, background, dtype=np.int8)
    for y, x, length, slope, color in bridges:
        for t in range(length):
            r, c = y + slope * t, x + t
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                canvas[r, c] = color
    active = []
    for point in pins:
        r, c = point
        for y, x, length, slope, color in bridges:
            cells = {(y + slope * t, x + t) for t in range(length)}
            if point in cells:
                active.append(point)
                for direction in (-1, 1):
                    rr, cc = r, c
                    while 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                        canvas[rr, cc] = signal
                        rr += direction
                        cc -= direction * slope
                break
    for r, c in pins:
        canvas[r, c] = signal
    r0, r1, c0, c1 = bb
    return canvas[r0:r1 + 1, c0:c1 + 1].copy()


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _proposal(rng, roles, plan):
    p = 1 if plan['growth'] else (2 if plan['cycle'] else int(rng.choice([1, 1, 2, 3])))
    skip, span = plan['skip'], plan['span']
    # Choose feasible arithmetic geometry first, rather than repeatedly drawing
    # impossible long sequences. These are parameter tuples, not stored grids.
    options = []
    for nv0 in range(max(3, 2 * p), 7):
        if plan['cycle'] and span == 1 and nv0 % p == 0:
            continue  # A phase-zero single target would not teach color phase.
        total0 = nv0 + skip + span
        for delta in ((-1, 1) if plan['growth'] else (-1, 0, 0, 0, 1)):
            for smallest0 in range(3, 7):
                start = smallest0 + max(0, -delta) * (total0 - 1)
                width_sum = total0 * start + delta * total0 * (total0 - 1) // 2
                for gap0 in range(1, 4):
                    if width_sum + (total0 - 1) * gap0 <= 29:
                        options.append((nv0, delta, start, gap0))
    if not options:
        return None
    nv, dl, l0, gap = options[int(rng.integers(len(options)))]
    total = nv + skip + span
    lengths = [l0 + i * dl for i in range(total)]
    dy = int(rng.integers(-1, 2))
    xs, x = [], 0
    for length in lengths:
        xs.append(x)
        x += length + gap
    if x - gap > 29:
        return None
    ys = [i * dy for i in range(total)]
    # Start with positive slope. A reflection supplies either visible slope.
    bridges = [(ys[i], xs[i], lengths[i], 1, roles[3 + i % p]) for i in range(total)]
    targets = bridges[nv + skip:]
    pad_low, pad_high = (3, 5) if plan['witness'] else (0, 3)
    r0 = min(b[0] for b in targets) - int(rng.integers(pad_low, pad_high))
    r1 = max(b[0] + b[2] - 1 for b in targets) + int(rng.integers(pad_low, pad_high))
    previous = nv + skip - 1
    c0 = max(xs[previous] + lengths[previous], targets[0][1] - int(rng.integers(0, 3)))
    c1 = targets[-1][1] + targets[-1][2] - 1 + min(gap, int(rng.integers(int(plan['witness']), 3)))
    if plan['partial']:
        if rng.random() < .5:
            r0 += int(rng.integers(1, max(2, (r1 - r0) // 2 + 1)))
        else:
            c0 += int(rng.integers(1, max(2, (c1 - c0) // 2 + 1)))
            c0 = min(c0, targets[0][1] + targets[0][2] - 1)
    if min(r1 - r0, c1 - c0) < 2:
        return None
    ymin = min(r0, min(b[0] for b in bridges[:nv]))
    ymax = max(r1, max(b[0] + b[2] - 1 for b in bridges[:nv]))
    height, width = ymax - ymin + 1, c1 + 1
    if max(height, width) > 30:
        return None
    top = int(rng.integers(0, min(2, 30 - height) + 1))
    left = int(rng.integers(0, min(2, 30 - width) + 1))
    bottom = int(rng.integers(0, min(2, 30 - height - top) + 1))
    right = int(rng.integers(0, min(2, 30 - width - left) + 1))
    shift_r, shift_c = top - ymin, left
    bridges = [(y + shift_r, x + shift_c, length, s, color) for y, x, length, s, color in bridges]
    bb = (r0 + shift_r, r1 + shift_r, c0 + shift_c, c1 + shift_c)
    r0, r1, c0, c1 = bb
    bg, marker, signal = roles[:3]
    shape = (height + top + bottom, width + left + right)
    hidden = [b for b in bridges[nv:] if b[1] <= c1 and b[1] + b[2] > c0]
    corners = _corners(bb)
    available = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                 if (r, c) not in corners]
    active_pool = [q for q in available if any(_on(q, b) for b in hidden)]
    inert_pool = [q for q in available if not any(_on(q, b) for b in hidden)]
    beyond = [q for q in inert_pool if any(_on(q, b, False) for b in hidden)]
    nactive = plan['active']
    ninert = max(1, int(rng.integers(0, 4))) if plan['witness'] else int(rng.integers(0, 4))
    if not nactive and not ninert:
        ninert = 1
    if len(active_pool) < nactive or len(inert_pool) < ninert or (plan['witness'] and not beyond):
        return None
    pins = [active_pool[int(i)] for i in rng.permutation(len(active_pool))[:nactive]]
    inert = [beyond[int(rng.integers(len(beyond)))]] if plan['witness'] else []
    for i in rng.permutation(len(inert_pool)):
        if len(inert) == ninert:
            break
        if inert_pool[int(i)] not in inert:
            inert.append(inert_pool[int(i)])
    pins += inert
    a = np.full(shape, bg, dtype=np.int8)
    for y, x, length, slope, color in bridges[:nv]:
        a[y, x] = color
        a[y + slope * (length - 1), x + length - 1] = color
    for q in pins:
        a[q] = signal
    out = _stroke_truth(shape, bb, bg, signal, bridges, pins)
    # Keep annotation orientation canonical after every scene symmetry.
    if plan['slope'] == -1:
        a, out = a[::-1].copy(), out[::-1].copy()
        bb = (shape[0] - 1 - bb[1], shape[0] - 1 - bb[0], bb[2], bb[3])
    if plan['reverse']:
        a, out = a[:, ::-1].copy(), out[:, ::-1].copy()
        bb = (bb[0], bb[1], shape[1] - 1 - bb[3], shape[1] - 1 - bb[2])
    if plan['axis']:
        a, out = a.T.copy(), out.T.copy()
        bb = (bb[2], bb[3], bb[0], bb[1])
    for q in _corners(bb):
        a[q] = marker
    try:
        prediction, info = _analyze(a)
    except AssertionError:
        return None
    if not np.array_equal(prediction, out):
        return None
    metrics = _metrics(info, prediction)
    if not (metrics['primary_survives'] and metrics['no_A_changes'] and metrics['endpoints_only_changes']):
        return None
    if metrics['active'] and not metrics['no_B_changes']:
        return None
    if plan['witness'] and not (metrics['no_B_changes'] and metrics['infinite_changes']
                                and metrics['all_sensors_changes']):
        return None
    if plan['growth'] and not metrics['freeze_length_changes']:
        return None
    if plan['cycle'] and not metrics['freeze_color_changes']:
        return None
    return {'input': a, 'output': out}


def _metrics(info, prediction):
    # Work in the fitted axis so measurements are reconstructed from input.
    out = prediction.T if info['transpose'] else prediction
    bb, bg, signal = info['bb'], info['background'], info['signal']
    hidden = [b for i, b in info['hidden']]
    pins = info['pins']
    def render(bs=hidden, **kwargs):
        return _equation_render(bb, bg, signal, bs, pins, **kwargs)[0]
    pin_only = np.full(out.shape, bg, dtype=np.int8)
    for r, c in pins:
        pin_only[r - bb[0], c - bb[2]] = signal
    last_length = info['visible'][0 if bb[3] < info['visible'][0][1] else -1][2]
    length_frozen = [(y, x, last_length, s, color) for y, x, length, s, color in hidden]
    color_frozen = [(y, x, length, s, info['visible'][0][4]) for y, x, length, s, color in hidden]
    crossing = any(bb[0] <= y + s * t <= bb[1] and bb[2] <= x + t <= bb[3]
                   and (y + s * t, x + t) != q
                   and y + s * t - q[0] == -s * (x + t - q[1])
                   for q in info['active'] for y, x, length, s, color in hidden
                   for t in range(length))
    return dict(visible=len(info['visible']), hidden=len(hidden), signal_crosses_bridge=crossing,
                length_delta=info['model'][4], drift=info['model'][3], gap=info['model'][5],
                cycle=len(info['model'][7]), active=len(info['active']),
                inert=len(pins) - len(info['active']), slope=hidden[0][3],
                axis=int(info['transpose']), background=bg, marker=info['marker'], signal=signal,
                before=bb[3] < info['visible'][0][1],
                skipped=max(0, min(abs(i - (len(info['visible']) if i >= 0 else -1))
                                   for i, b in info['hidden'])),
                partial=any(bb[0] > min(y, y + s * (length - 1)) or
                            bb[1] < max(y, y + s * (length - 1)) or
                            bb[2] > x or bb[3] < x + length - 1 for y, x, length, s, color in hidden),
                primary_survives=bool(np.any((out != bg) & (out != signal))),
                no_A_changes=not np.array_equal(pin_only, out),
                no_B_changes=not np.array_equal(render(fire=False), out),
                endpoints_only_changes=not np.array_equal(render(endpoints_only=True, fire=False), out),
                infinite_changes=not np.array_equal(render(infinite=True), out),
                all_sensors_changes=not np.array_equal(render(all_pins=True), out),
                freeze_length_changes=not np.array_equal(render(length_frozen), out),
                freeze_color_changes=not np.array_equal(render(color_frozen), out))


def _integer(name, value, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' out of range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    nt = _integer('num_train', num_train, 2, 10)
    ne = _integer('num_test', num_test, 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    shared_roles = list(map(int, rng.permutation(10)))
    vary_roles = bool(rng.integers(2))
    # Each puzzle isolates a visible generalization: axis, distance, or crop span.
    shift = int(rng.integers(3))
    axis = int(rng.integers(2))
    witness_slots = list(map(int, rng.choice(nt, 2, replace=False)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for j in range(count):
            test = split == 'test'
            witness = not test and j in witness_slots
            growth = not test and j == witness_slots[0]
            cycle = not test and j == witness_slots[1]
            plan = dict(witness=witness, growth=growth, cycle=cycle,
                        axis=(1 - axis if shift == 0 and test else axis),
                        skip=(int(rng.integers(1, 3)) if shift == 1 and test else 0),
                        span=(2 if shift == 2 and test else int(rng.choice([1, 1, 2]))),
                        partial=bool(rng.integers(2)), active=(1 if growth else 2 if cycle else int(rng.integers(0, 4))),
                        slope=(1 if j == witness_slots[0] else -1) if witness else int(rng.choice([-1, 1])),
                        reverse=bool(rng.integers(2)))
            if shift == 2 and not test:
                plan['span'] = 1
            if shift == 2 and test:
                plan['partial'] = False
            if witness and plan['reverse']:
                plan['slope'] *= -1
            # Bounded rejection, no retained retries. Witness growth uses short
            # sequences so a 30-cell canvas leaves space for the full lesson.
            for attempt in range(2400):
                roles = list(map(int, rng.permutation(10))) if vary_roles else shared_roles
                pair = _proposal(rng, roles, plan)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('No unique geometric proposal within the finite budget')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, measurements = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert _key(a) not in seen
            seen.add(_key(a))
            prediction, info = _analyze(a)
            assert np.array_equal(prediction, b), 'Wrong hidden bridge or signal crop'
            assert info['marker'] not in set(map(int, np.unique(b)))
            assert set(map(int, np.unique(b))) <= set(map(int, np.unique(a)))
            m = _metrics(info, prediction)
            assert m['primary_survives'] and m['no_A_changes'] and m['endpoints_only_changes']
            if m['active']:
                assert m['no_B_changes']
            measurements[split].append(m)
    # Two distributed witnesses suffice even at the minimum legal counts.
    training, testing = measurements['train'], measurements['test']
    assert any(m['length_delta'] and m['freeze_length_changes'] for m in training)
    assert any(m['cycle'] >= 2 and m['freeze_color_changes'] for m in training)
    assert {m['slope'] for m in training} == {-1, 1}
    assert sum(m['no_B_changes'] and m['infinite_changes'] and m['all_sensors_changes']
               for m in training) >= 2
    axis_shift = {m['axis'] for m in training}.isdisjoint({m['axis'] for m in testing})
    distance_shift = min(m['skipped'] for m in testing) > max(m['skipped'] for m in training)
    span_shift = min(m['hidden'] for m in testing) > max(m['hidden'] for m in training)
    assert axis_shift or distance_shift or span_shift, 'A visible train/test shift is required'


def self_test():
    """42 streaming seeds, ablations, symmetries, counts, corruption rejection."""
    signatures, shapes, structures = set(), set(), set()
    values = {k: set() for k in ('visible', 'hidden', 'length_delta', 'drift', 'gap',
                                 'cycle', 'active', 'inert', 'slope', 'axis', 'background',
                                 'marker', 'signal', 'before', 'skipped', 'partial',
                                 'signal_crosses_bridge')}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                digest.update(bytes(a.shape) + a.tobytes() + bytes(b.shape) + b.tobytes())
                shapes.add((a.shape, b.shape))
                pred, info = _analyze(a)
                for k, v in _metrics(info, pred).items():
                    if k in values:
                        values[k].add(v)
                mapping = {}
                norm = bytes(mapping.setdefault(int(c), len(mapping)) for c in a.flat)
                structures.add((a.shape, norm))
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(shapes) >= 100 and len(structures) >= 180
    for k in ('length_delta', 'drift'):
        assert {-1, 0, 1} <= values[k], (k, values[k])
    assert values['cycle'] == {1, 2, 3}
    assert values['axis'] == {0, 1} and values['slope'] == {-1, 1}
    assert values['before'] == {False, True} and values['partial'] == {False, True}
    assert len(values['visible']) >= 3 and len(values['hidden']) >= 2
    assert {0, 1, 2, 3} <= values['active'] and {0, 1, 2, 3} <= values['inert']
    assert values['signal_crosses_bridge'] == {False, True}
    assert max(values['skipped']) >= 1 and len(values['background']) >= 6
    # Exercise all count combinations without retaining their grid payloads.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(1953 + nt * 10 + ne, nt, ne))
    for nt, ne in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)]:
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    first, second = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    assert all(_key(p[k]) == _key(q[k]) for s in first for p, q in zip(first[s], second[s])
               for k in ('input', 'output'))
    # Solver must leave its argument untouched and work with plain nested lists.
    for pair in first['train'] + first['test']:
        a = pair['input'].copy()
        assert np.array_equal(solve(a.tolist()), pair['output'])
        assert np.array_equal(a, pair['input'])
    # A compatible arithmetic member outside the sampler's +/-1 growth support.
    a = np.zeros((12, 30), dtype=np.int8)
    bridges = [(0, 0, 3, 1, 1), (0, 5, 5, 1, 1),
               (0, 12, 7, 1, 1), (0, 21, 9, 1, 1)]
    bb, pins = (0, 10, 20, 29), [(4, 25), (10, 23)]
    for y, x, length, slope, color in bridges[:3]:
        a[y, x] = color
        a[y + length - 1, x + length - 1] = color
    for q in pins:
        a[q] = 8
    for q in _corners(bb):
        a[q] = 9
    expected = _stroke_truth(a.shape, bb, 0, 8, bridges, pins)
    assert _analyze(a)[1]['model'][4] == 2
    assert np.array_equal(solve(a), expected)
    first['test'][0]['output'][0, 0] = (int(first['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(first)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupted label accepted')
