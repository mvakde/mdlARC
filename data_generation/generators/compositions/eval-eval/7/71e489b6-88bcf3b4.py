"""Safety thread: unary slack length routed outside reconstructed defect halos.

Only NumPy and the standard library are used. The solver reads input pixels;
construction uses explicit plate/defect masks and a separate clearance sweep.
"""
from __future__ import annotations

import json
import numpy as np


FIELD = 0
SUPPORT = 8
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int64, copy=True)


def _components(cells, diagonal=True):
    unseen = set(cells)
    result = []
    directions = DIR8 if diagonal else DIR4
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        todo, component = [start], {start}
        while todo:
            u, v = todo.pop()
            for du, dv in directions:
                q = u + du, v + dv
                if q in unseen:
                    unseen.remove(q)
                    component.add(q)
                    todo.append(q)
        result.append(component)
    return result


def _bbox(cells):
    return (min(u for u, v in cells), max(u for u, v in cells),
            min(v for u, v in cells), max(v for u, v in cells))


def _local(p, end, forward, side):
    dr, dc = p[0] - end[0], p[1] - end[1]
    return dr * forward[0] + dc * forward[1], dr * side[0] + dc * side[1]


def _global(p, end, forward, side):
    u, v = p
    return (end[0] + u * forward[0] + v * side[0],
            end[1] + u * forward[1] + v * side[1])


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _halo(holes):
    # Deliberately unbounded: a visible defect has its full conceptual halo,
    # even if some of those cells lie outside the actual canvas.
    return {(u + du, v + dv) for u, v in holes
            for du in (-1, 0, 1) for dv in (-1, 0, 1)}


def _route(holes, length, buffered=True):
    """Least lateral, 1-Lipschitz profile outside all safety zones.

    A halo requires lateral clearance h_v+1 at its axial row h_u.
    Its influence declines by one per axial step. Max combines requirements.
    The attachment line v=1 is the lower bound, including after departure.
    """
    hazards = _halo(holes) if buffered else set(holes)
    return [(u, max([1] + [v + 1 - abs(u - t) for t, v in hazards]))
            for u in range(length)]


def _construction_route(holes, length):
    """Separate label construction: rasterize halos, then clearance sweeps."""
    far = max(u for u, v in holes) + 1
    size = max(length, far + 2)
    profile = np.ones(size, dtype=int)
    for u, v in holes:
        # The outward edge of the Moore halo spans three axial rows.
        for axial in range(u - 1, u + 2):
            if 0 <= axial < size:
                profile[axial] = max(int(profile[axial]), v + 2)
    for u in range(1, size):
        profile[u] = max(int(profile[u]), int(profile[u - 1]) - 1)
    for u in range(size - 2, -1, -1):
        profile[u] = max(int(profile[u]), int(profile[u + 1]) - 1)
    return [(u, int(v)) for u, v in enumerate(profile[:length])]


def _decode(value):
    inp = _grid(value)
    support = set(map(tuple, np.argwhere(inp == SUPPORT)))
    if not 3 <= len(support) <= 7:
        raise ValueError("one straight support of length 3..7 is required")
    r0, r1, c0, c1 = _bbox(support)
    if r0 == r1 and len(support) == c1 - c0 + 1:
        positive = (0, 1)
    elif c0 == c1 and len(support) == r1 - r0 + 1:
        positive = (1, 0)
    else:
        raise ValueError("support must be a contiguous straight line")
    colors = set(map(int, np.unique(inp))) - {FIELD, SUPPORT}
    if len(colors) != 2 or not np.any(inp == FIELD):
        raise ValueError("exactly field, support, plate and thread colors required")
    candidates = []
    for end, forward in ((min(support), tuple(-x for x in positive)),
                         (max(support), positive)):
        for side in ((forward[1], -forward[0]), (-forward[1], forward[0])):
            attachment = _global((0, 1), end, forward, side)
            if not _inside(attachment, inp.shape) or int(inp[attachment]) not in colors:
                continue
            thread_color = int(inp[attachment])
            slack = set(map(tuple, np.argwhere(inp == thread_color)))
            local_slack = {_local(p, end, forward, side) for p in slack}
            if (len(slack) < 2 or len(_components(slack)) != 1
                    or any(u > 0 or v < 1 for u, v in local_slack)):
                continue
            plate_color = next(iter(colors - {thread_color}))
            observed = {_local(tuple(p), end, forward, side)
                        for p in np.argwhere(inp == plate_color)}
            base, extras, boxes = set(), set(), []
            valid = True
            for comp in _components(observed):
                if len(comp) <= 2:
                    extras.update(comp)
                    continue
                lo, hi, left, right = _bbox(comp)
                box = {(u, v) for u in range(lo, hi + 1)
                       for v in range(left, right + 1)}
                corners = {(lo, left), (lo, right), (hi, left), (hi, right)}
                if (hi - lo < 3 or right - left < 3 or lo < 2
                        or not corners <= comp or len(comp) < .7 * len(box)):
                    valid = False
                    break
                if any(max(abs(u - t), abs(v - w)) <= 1
                       for u, v in box for t, w in base):
                    valid = False
                    break
                base.update(box)
                boxes.append((lo, hi, left, right))
            holes = base - observed
            if (not valid or not 1 <= len(boxes) <= 4 or not holes
                    or any(u < 3 or v < 1 or v > u - 2 for u, v in holes)):
                continue
            if any(inp[_global(p, end, forward, side)] != FIELD for p in holes):
                continue
            path = _route(holes, len(slack))
            if path[0] != (0, 1) or set(path) & _halo(holes):
                continue
            candidates.append(dict(input=inp, support=support, endpoint=end,
                                   forward=forward, side=side, thread=thread_color,
                                   plate=plate_color, slack=slack, base=base,
                                   extras=extras, holes=holes, boxes=boxes, path=path))
    if len(candidates) != 1:
        raise ValueError("input must have one unambiguous attached thread")
    return candidates[0]


def _render_prediction(d, path=None):
    out = d['input'].copy()
    for p in d['slack']:
        out[p] = FIELD
    for p in d['extras']:
        out[_global(p, d['endpoint'], d['forward'], d['side'])] = FIELD
    for p in d['path'] if path is None else path:
        q = _global(p, d['endpoint'], d['forward'], d['side'])
        if _inside(q, out.shape):
            out[q] = d['thread']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer exclusively from visible cells and fixed conventions."""
    return _render_prediction(_decode(input_grid))


def _slack(rng, length, depth, width, style, diagonal_lesson=False):
    cells = {(0, 1)}
    # In the lesson, the attachment's only thread neighbor is diagonal.
    forbidden = {(0, 2), (-1, 1)} if diagonal_lesson else set()
    if diagonal_lesson:
        cells.add((-1, 2))
    if style == 1:
        for k in range(1, min(depth + 1, width, length)):
            cells.add((-k, k + 1))
            if len(cells) == length:
                return cells
    while len(cells) < length:
        frontier = sorted({(u + du, v + dv) for u, v in cells
                           for du, dv in (DIR4 if style == 0 and not diagonal_lesson else DIR8)
                           if -depth <= u + du <= 0 and 1 <= v + dv <= width}
                          - cells - forbidden)
        if not frontier:
            raise ValueError("insufficient slack area")
        weights = []
        for u, v in frontier:
            neighbors = sum((u + du, v + dv) in cells for du, dv in DIR8)
            weights.append((neighbors + 1) ** (2 if style == 0 else -2 if style == 2 else 0))
        weights = np.asarray(weights, dtype=float)
        cells.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
    return cells


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _make_pair(rng, colors, lesson, count, rotation, reflect, length_cap=None, reach_last=False):
    """Build one scene, not a composition of whole parent generators."""
    plate_color, thread_color = colors
    for attempt in range(180):
        peak = int(rng.integers(2, 6))
        widths = [int(rng.integers(4, min(8, peak + 4))) for _ in range(count)]
        heights = list(map(int, rng.integers(4, 7, size=count)))
        gaps = list(map(int, rng.integers(2, 5, size=max(0, count - 1))))
        start = int(rng.integers(max(3, peak - 1), max(4, peak + 2)))
        if start + sum(heights) + sum(gaps) > 23:
            continue
        base, holes, boxes = set(), set(), []
        cursor = start
        for j, (height, width) in enumerate(zip(heights, widths)):
            high = peak + 1 if lesson == 'combined' else peak + int(rng.integers(0, 3))
            low = high - width + 1
            box = {(u, v) for u in range(cursor, cursor + height)
                   for v in range(low, high + 1)}
            center = (cursor + int(rng.integers(1, height - 1)),
                      peak if lesson == 'combined' else int(rng.integers(max(1, low + 1), high + 1)))
            if center[1] > center[0] - 2:
                break
            base.update(box)
            boxes.append((cursor, cursor + height - 1, low, high))
            holes.add(center)
            # Sparse clusters vary separately from rectangle dimensions.
            motif = int(rng.integers(6))
            offsets = ((), ((0, -1),), ((1, -1),), ((0, -1), (1, -1)),
                       ((0, -2),), ((2, -1),))[motif]
            corners = {(cursor, low), (cursor, high),
                       (cursor + height - 1, low), (cursor + height - 1, high)}
            for du, dv in offsets:
                p = center[0] + du, center[1] + dv
                if p in box and p not in corners and 1 <= p[1] <= p[0] - 2:
                    holes.add(p)
            cursor += height + (gaps[j] if j < len(gaps) else 0)
        if len(boxes) != count:
            continue
        # Recovered plate components must not split under the corruption.
        if len(_components(base - holes)) != count:
            continue
        last = max(u for u, v in holes)
        if lesson == 'complete':
            length = max(u + v + 3 for u, v in holes) + int(rng.integers(1, 3))
        elif lesson == 'combined':
            length = last + int(rng.integers(2, 5))
        elif reach_last:
            length = last + int(rng.integers(2, 5))
        else:
            first_influence = min(u - v - 1 for u, v in holes)
            low_n = max(3, first_influence + 2)
            upper = min(27, last + peak + 5, 27 if length_cap is None else length_cap + 1)
            if low_n >= upper:
                continue
            length = int(rng.integers(low_n, upper))
        depth = int(rng.integers(3, 7))
        support_length = int(rng.integers(3, 8))
        depth = max(depth, support_length - 1)
        width = int(rng.integers(3, peak + 2)) if lesson == 'combined' else int(rng.integers(3, 9))
        if length > (depth + 1) * width - (2 if lesson == 'complete' else 0):
            continue
        try:
            slack = _slack(rng, length, depth, width, int(rng.integers(4)), lesson == 'complete')
        except ValueError:
            continue
        support = {(-u, 0) for u in range(support_length)}
        extras = set()
        # A conspicuous false peg ahead of the complete lesson. It is a solid
        # speck, not a missing cell, and must not alter the thread's roof.
        if lesson == 'complete':
            center = min(holes)
            extra_v = max(v for u, v in base) + 3
            # Even the wrong interpretation as a safety peg has an admissible
            # attachment here, so geometry alone cannot dismiss that shortcut.
            extras.add((max(center[0], extra_v + 2), extra_v))
        desired_extras = int(rng.integers(1, 6))
        choices = [(u, v) for u in range(1, cursor) for v in range(-5, 0)]
        rng.shuffle(choices)
        for p in choices:
            if len(extras) >= desired_extras:
                break
            if (p not in base and p not in slack and p not in support
                    and all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1
                            for q in base | support | slack | extras)):
                extras.add(p)
                q = (p[0] + 1, p[1])
                if (len(extras) < desired_extras and rng.random() < .28 and q[0] < cursor
                        and all(max(abs(q[0] - x[0]), abs(q[1] - x[1])) > 1
                                for x in base | support | slack | (extras - {p}))):
                    extras.add(q)
        path = _construction_route(holes, length)
        if path[0] != (0, 1) or path != _route(holes, length):
            continue
        input_cells = base | support | slack | extras
        ulo, uhi, vlo, vhi = _bbox(input_cells)
        clipped = lesson == 'combined' or (lesson == 'free' and rng.random() < .35)
        if not clipped:
            _, puhi, pvlo, pvhi = _bbox(path)
            uhi, vlo, vhi = max(uhi, puhi), min(vlo, pvlo), max(vhi, pvhi)
            if lesson == 'complete':
                uhi = max(uhi, length)  # show an unused cell beyond the thread
        else:
            # Keep all input material. Individual off-grid thread cells vanish;
            # later cells can re-enter after a lateral excursion.
            uhi = max(uhi, min(length - 1, 23))
        limits = [ulo, uhi, vlo, vhi]
        for axis in (0, 1):
            size = limits[2 * axis + 1] - limits[2 * axis] + 1
            budget = min(30 - size, int(rng.integers(0, 5)))
            if budget < 0:
                break
            for _ in range(budget):
                edge = 2 * axis + int(rng.integers(2))
                # Preserve the learned clipped outer side for this lesson.
                if lesson == 'combined' and edge == 3:
                    edge = 2
                limits[edge] += -1 if edge % 2 == 0 else 1
        if any(limits[2 * a + 1] - limits[2 * a] + 1 > 30 for a in (0, 1)):
            continue
        ulo, uhi, vlo, vhi = limits
        shape = uhi - ulo + 1, vhi - vlo + 1
        inp = np.zeros(shape, dtype=np.int64)
        for cells, color in ((base - holes, plate_color), (extras, plate_color),
                             (support, SUPPORT), (slack, thread_color)):
            for u, v in cells:
                inp[u - ulo, v - vlo] = color
        out = np.zeros(shape, dtype=np.int64)
        for u, v in base - holes:
            out[u - ulo, v - vlo] = plate_color
        for u, v in support:
            out[u - ulo, v - vlo] = SUPPORT
        for u, v in path:
            if ulo <= u <= uhi and vlo <= v <= vhi:
                out[u - ulo, v - vlo] = thread_color
        def transform(a):
            return np.rot90(np.fliplr(a) if reflect else a, rotation).copy()
        pair = {'input': transform(inp), 'output': transform(out)}
        try:
            d = _decode(pair['input'])
        except ValueError:
            continue
        if (d['holes'] != holes or d['base'] != base
                or not np.array_equal(_render_prediction(d), pair['output'])):
            continue
        m = _measure(d)
        if lesson == 'complete' and (m['clipped'] or not m['returned'] or not m['diagonal_slack']):
            continue
        if lesson == 'combined' and (not m['clipped'] or m['peaks'] < 2 or not m['reentered']):
            continue
        return pair
    raise RuntimeError("bounded scene construction exhausted")


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    colors = tuple(map(int, rng.choice([1, 2, 3, 4, 5, 6, 7, 9], 2, replace=False)))
    regime = int(rng.integers(3))
    train_axis = int(rng.integers(2))
    lessons = ['complete', 'combined'] + ['free'] * (nt - 2)
    rng.shuffle(lessons)
    puzzle, seen, core_lengths = {'train': [], 'test': []}, set(), []
    for split, number in (('train', nt), ('test', nq)):
        for i in range(number):
            lesson = lessons[i] if split == 'train' else 'free'
            count = 1 if lesson == 'complete' else 2 if lesson == 'combined' else int(rng.integers(1, 4))
            if split == 'train' and regime == 0 and lesson == 'free':
                count = int(rng.integers(1, 3))
            if split == 'test' and regime == 0:
                count = 3
            rotation = train_axis + 2 * int(rng.integers(2)) if regime == 1 else int(rng.integers(4))
            if split == 'test' and regime == 1:
                rotation = (rotation + 1) % 4
            reflect = bool(rng.integers(2))
            length_cap = (min(core_lengths) - 1
                          if split == 'test' and regime == 2 else None)
            for _ in range(24):
                pair = _make_pair(rng, colors, lesson, count, rotation, reflect,
                                  length_cap, split == 'test' and regime == 0)
                key = _key(pair['input'])
                duplicate_anchor_length = (split == 'train' and lesson in ('complete', 'combined')
                                           and any(len(_decode(p['input'])['slack']) ==
                                                   len(_decode(pair['input'])['slack'])
                                                   for p in puzzle['train']))
                if key not in seen and not duplicate_anchor_length:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == 'train' and lesson in ('complete', 'combined'):
                        core_lengths.append(len(_decode(pair['input'])['slack']))
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _measure(d):
    visible = [_inside(_global(p, d['endpoint'], d['forward'], d['side']), d['input'].shape)
               for p in d['path']]
    levels = [v for u, v in d['path']]
    rises = [b > a for a, b in zip(levels, levels[1:])]
    peaks = sum(rise and (i == 0 or not rises[i - 1]) for i, rise in enumerate(rises))
    return dict(plates=len(d['boxes']), holes=len(d['holes']), extras=len(d['extras']),
                length=len(d['slack']), visible=sum(visible), clipped=not all(visible),
                reentered=any(not visible[i] and any(visible[i + 1:]) for i in range(len(visible))),
                returned=levels[-1] == 1 and max(levels) > 1,
                diagonal_slack=len(_components(d['slack'], False)) > 1,
                peaks=peaks, max_lift=max(levels) - 1,
                groups=len(_components(d['holes'])),
                route_difference=any(a != b for a, b in zip(d['path'], _route(d['holes'], len(d['slack']), False))),
                shape=d['input'].shape, forward=d['forward'], side=d['side'])


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, palettes, training = set(), set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            d = _decode(inp)
            palettes.add((d['plate'], d['thread']))
            assert np.array_equal(_render_prediction(d), out), 'input-only prediction mismatch'
            assert all(out[p] == SUPPORT for p in d['support'])
            assert all(out[_global(p, d['endpoint'], d['forward'], d['side'])] == FIELD for p in d['holes'])
            assert not set(d['path']) & _halo(d['holes']), 'thread enters a safety zone'
            levels = [v for u, v in d['path']]
            assert all(v >= 1 for v in levels) and levels[0] == 1
            assert all(abs(a - b) <= 1 for a, b in zip(levels, levels[1:]))
            # Every elevated route cell is necessary: lowering the profile at
            # that cell violates either halo clearance or neighboring steps.
            clearance = {}
            for u, v in _halo(d['holes']):
                clearance[u] = max(clearance.get(u, 1), v + 1)
            for u, v in d['path']:
                if v > 1:
                    assert (v == clearance.get(u, 1)
                            or (u > 0 and levels[u - 1] == v + 1)
                            or (u + 1 < len(levels) and levels[u + 1] == v + 1)
                            or any(w + 1 - abs(u - t) == v for t, w in _halo(d['holes'])))
            m = _measure(d)
            assert m['visible'] == int(np.count_nonzero(out == d['thread']))
            expected_material = d['base'] - d['holes'] - set(d['path'])
            assert all(out[_global(p, d['endpoint'], d['forward'], d['side'])] == d['plate']
                       for p in expected_material)
            assert all(out[_global(p, d['endpoint'], d['forward'], d['side'])] ==
                       (d['thread'] if p in d['path'] else FIELD) for p in d['extras'])
            if split == 'train':
                training.append((d, m))
    assert len(palettes) == 1, 'palette conventions must be puzzle-wide'
    assert any(m['returned'] and not m['clipped'] and m['diagonal_slack'] for d, m in training)
    assert any(m['clipped'] and m['reentered'] and m['peaks'] >= 2 for d, m in training)
    assert any(m['route_difference'] for d, m in training), 'missing safety-radius demonstration'
    assert len({m['length'] for d, m in training}) >= 2
    # Ablations are applied to the same decoded input, never to latent state.
    assert any(not np.array_equal(_render_prediction(d, [(u, 1) for u in range(len(d['slack']))]),
                                 _render_prediction(d)) for d, m in training), 'A does not affect the route'
    assert any(not np.array_equal(_render_prediction(d, _route(d['holes'], len(d['slack']) + 1)),
                                 _render_prediction(d)) for d, m in training), 'B length is not observable'
    assert any(not np.array_equal(_render_prediction(d, _route(d['holes'] | d['extras'], len(d['slack']))),
                                 _render_prediction(d)) for d, m in training), 'speck/defect distinction unexposed'


def self_test():
    """Bounded checks of geometry, causal contributions, and 42-seed variety."""
    assert _route({(4, 2)}, 10) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 4),
                                  (5, 4), (6, 3), (7, 2), (8, 1), (9, 1)]
    for holes in ({(4, 2)}, {(4, 2), (5, 1)}, {(4, 2), (12, 2)}, {(7, 3), (8, 4), (9, 3)}):
        for length in (2, 8, 16, 24):
            assert _route(holes, length) == _construction_route(holes, length)
    # A valid four-plate input, beyond the sampler's usual 1..3. Later defects
    # alter the ninth route cell even though the thread has not reached them.
    fixture = np.zeros((29, 17), dtype=int)
    end, f, side = (3, 5), (1, 0), (0, 1)
    for start in (3, 9, 15, 21):
        for u in range(start, start + 4):
            for v in range(4):
                fixture[_global((u, v), end, f, side)] = 4
        fixture[_global((start + 1, 2), end, f, side)] = FIELD
    fixed = {(-k, 0) for k in range(4)}
    slack = {(0, 1), (-1, 2), (-1, 3), (-1, 4), (-2, 2),
             (-2, 3), (-2, 4), (-3, 3), (-3, 4)}
    extras = {(8, -3), (8, -2)}
    for cells, color in ((fixed, SUPPORT), (slack, 6), (extras, 4)):
        for p in cells:
            fixture[_global(p, end, f, side)] = color
    expected = fixture.copy()
    for p in slack | extras:
        expected[_global(p, end, f, side)] = FIELD
    for u, v in enumerate([1, 2, 3, 4, 4, 4, 3, 2, 3]):
        expected[_global((u, v), end, f, side)] = 6
    assert len(_decode(fixture)['boxes']) == 4
    assert np.array_equal(solve(fixture), expected)
    shapes, structures, palettes, rows = set(), set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                d = _decode(pair['input'])
                rows.append(_measure(d))
                shapes.add(d['input'].shape)
                palettes.add((d['plate'], d['thread']))
                structures.add((tuple(sorted(d['base'])), tuple(sorted(d['holes'])), len(d['slack'])))
                # Rule equivariance also checks attachment identification.
                for k in range(4):
                    x, y = np.rot90(pair['input'], k), np.rot90(pair['output'], k)
                    assert np.array_equal(solve(x), y)
                assert np.array_equal(solve(np.fliplr(pair['input'])), np.fliplr(pair['output']))
        if seed < 2:
            repeat = generate_puzzle(seed)
            assert all(np.array_equal(a[k], b[k]) for s in puzzle
                       for a, b in zip(puzzle[s], repeat[s]) for k in ('input', 'output'))
    assert len(shapes) >= 50 and len(structures) >= 150 and len(palettes) >= 20
    assert {m['plates'] for m in rows} == {1, 2, 3}
    assert len({m['holes'] for m in rows}) >= 6 and len({m['length'] for m in rows}) >= 15
    assert {m['forward'] for m in rows} == {(-1, 0), (1, 0), (0, -1), (0, 1)}
    assert any(m['peaks'] >= 3 for m in rows)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            generate_puzzle(919, nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (True, 1), (3.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    damaged = generate_puzzle(121, 2, 1)
    d = _decode(damaged['test'][0]['input'])
    damaged['test'][0]['output'][min(d['support'])] = FIELD
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a damaged fixed support')
    return {'seeds': 42, 'pairs': len(rows), 'shapes': len(shapes),
            'scaffold_defect_length_structures': len(structures), 'palettes': len(palettes),
            'plate_counts': sorted({m['plates'] for m in rows}),
            'strand_lengths': sorted({m['length'] for m in rows}),
            'hole_counts': sorted({m['holes'] for m in rows})}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
