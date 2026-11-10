"""Chromatic caustics: invisible reflected flights bracket colored light sheets.

Only standard-library modules and NumPy are used. No parent code, stored pairs,
seed-dependent semantics, or training labels enter solve().
"""
from __future__ import annotations

from itertools import combinations

import numpy as np


class SceneError(ValueError):
    """The input does not have the family's unambiguous optical geometry."""


def _need(condition, message):
    if not condition:
        raise SceneError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "integer colors required")
    _need(np.all((0 <= a) & (a <= 9)), "colors must be 0..9")
    return a.astype(int, copy=True)


def _inside(point, shape):
    return all(0 <= x < n for x, n in zip(point, shape))


def _parse(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _need(counts[bg] > a.size / 2, "a unique majority field is required")
    remaining = {tuple(map(int, p)) for p in np.argwhere(a != bg)}
    sources, mirrors = [], []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], {start}
        color = int(a[start])
        while stack:
            r, c = stack.pop()
            for point in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if point in remaining and a[point] == color:
                    remaining.remove(point)
                    cells.add(point)
                    stack.append(point)
        rows = {p[0] for p in cells}
        cols = {p[1] for p in cells}
        if len(cells) == 3 and len(rows) == len(cols) == 2:
            missing = next(iter({(r, c) for r in rows for c in cols} - cells))
            elbow = (sum(rows)-missing[0], sum(cols)-missing[1])
            direction = (elbow[0]-missing[0], elbow[1]-missing[1])
            sources.append(dict(cells=cells, elbow=elbow, direction=direction, color=color))
        else:
            _need(len(cells) % 2 == 1 and (len(rows) == 1 or len(cols) == 1),
                  "objects must be L triominoes or odd straight bars")
            _need(len(cells) == max(max(rows)-min(rows), max(cols)-min(cols))+1,
                  "bars must be contiguous")
            midpoint = ((min(rows)+max(rows))//2, (min(cols)+max(cols))//2)
            axes = (0, 1) if len(cells) == 1 else (0,) if len(rows) == 1 else (1,)
            mirrors.append(dict(cells=cells, midpoint=midpoint, axes=axes, color=color))
    _need(bool(sources), "at least one directional source is required")
    return bg, sources, mirrors


def _flight(start, end, direction, color, source_color, source_id):
    slope = direction[0]*direction[1]
    lo, hi = sorted((start[0]+slope*start[1], end[0]+slope*end[1]))
    return dict(start=start, end=end, direction=direction, slope=slope,
                q=start[1]-slope*start[0], lo=lo, hi=hi, color=color,
                source_color=source_color, source_id=source_id)


def _trace(a):
    """Trace geometry without painting any rays or using generation metadata."""
    bg, sources, mirrors = _parse(a)
    flights, routes, used = [], [], set()
    for source_id, source in enumerate(sources):
        anchor = position = source['elbow']
        direction, color = source['direction'], source['color']
        seen, contacts = set(), []
        route_start = len(flights)
        while True:
            next_point = (position[0]+direction[0], position[1]+direction[1])
            if not _inside(next_point, a.shape):
                flights.append(_flight(anchor, position, direction, color, source['color'], source_id))
                break
            state = (next_point, direction)
            _need(state not in seen, "ray loop")
            seen.add(state)
            _need(a[next_point] == bg, "a flight hits an object rather than a midpoint face")
            hits = []
            for mirror_id, mirror in enumerate(mirrors):
                midpoint = mirror['midpoint']
                for axis in mirror['axes']:
                    if (next_point[1-axis] == midpoint[1-axis]
                            and midpoint[axis]-next_point[axis] == direction[axis]):
                        hits.append((mirror_id, axis))
            _need(len(hits) <= 1, "competing midpoint contacts")
            if hits:
                # Incoming color ends at the predecessor. The contact belongs
                # to the new, reflected flight and immediately copies its bar.
                flights.append(_flight(anchor, position, direction, color, source['color'], source_id))
                mirror_id, axis = hits[0]
                used.add(mirror_id)
                old_color = color
                color = mirrors[mirror_id]['color']
                contacts.append(dict(point=next_point, axis=axis, mirror=mirror_id,
                                     incoming=old_color, color=color))
                d = list(direction)
                d[axis] *= -1
                direction = tuple(d)
                anchor = next_point
            position = next_point
        routes.append(dict(source=source_id, contacts=contacts,
                           flights=list(range(route_start, len(flights)))))
    _need(used == set(range(len(mirrors))), "every mirror must be contacted")
    return dict(background=bg, sources=sources, mirrors=mirrors, flights=flights, routes=routes)


def _sweep_claims(a, info):
    """Primary implementation: outermost active intervals of each color/slope."""
    claims = np.zeros((10, *a.shape), dtype=bool)
    rows, cols = np.indices(a.shape)
    groups = {}
    for f in info['flights']:
        groups.setdefault((f['slope'], f['color']), []).append(f)
    for (slope, color), flights in groups.items():
        q = cols-slope*rows
        t = rows+slope*cols
        for level in range(int(t.min()), int(t.max())+1):
            active = [f['q'] for f in flights if f['lo'] <= level <= f['hi']]
            if len(active) >= 2 and min(active) < max(active):
                claims[color] |= ((a == info['background']) & (t == level)
                                  & (q > min(active)) & (q < max(active)))
    return claims


def _pair_claims(a, info, alternative=''):
    """Check via unions of pairwise parallelograms; also expose shortcuts.

    This geometric implementation does not call the scanline painter. It shares
    the input parser/tracer and is a self-authored consistency check, not an
    independent reference solver.
    """
    flights = info['flights']
    if alternative == 'ignore_mirrors':
        flights = []
        for i, source in enumerate(info['sources']):
            start, d = source['elbow'], source['direction']
            distance = min(a.shape[axis]-1-start[axis] if d[axis] > 0 else start[axis]
                           for axis in (0, 1))
            end = tuple(start[axis]+distance*d[axis] for axis in (0, 1))
            flights.append(_flight(start, end, d, source['color'], source['color'], i))
    elif alternative == 'keep_source_color':
        flights = [dict(f, color=f['source_color']) for f in flights]
    rows, cols = np.indices(a.shape)
    claims = np.zeros((10, *a.shape), dtype=bool)
    groups = {}
    for f in flights:
        key = (f['slope'], -1 if alternative == 'ungated' else f['color'])
        groups.setdefault(key, []).append(f)
    for (slope, color), group in groups.items():
        ordered_q = sorted({f['q'] for f in group})
        neighbors = set(zip(ordered_q, ordered_q[1:]))
        q, t = cols-slope*rows, rows+slope*cols
        for first, second in combinations(group, 2):
            q1, q2 = sorted((first['q'], second['q']))
            if q1 == q2:
                continue
            if alternative == 'consecutive' and (q1, q2) not in neighbors:
                continue
            if alternative == 'directed' and first['direction'] != second['direction']:
                continue
            mask = ((a == info['background']) & (q > q1) & (q < q2)
                    & (t >= max(first['lo'], second['lo']))
                    & (t <= min(first['hi'], second['hi'])))
            if alternative == 'pixel_centers':
                mask &= ((t-first['q']) % 2 == 0) & ((t-second['q']) % 2 == 0)
            claims[0 if color == -1 else color] |= mask
    return claims


def _finish(a, claims):
    _need(np.all(np.sum(claims, axis=0) <= 1), "different-colored light sheets conflict")
    output = a.copy()
    for color in range(10):
        output[claims[color]] = color
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from visible objects and fixed family rules."""
    a = _grid(input_grid)
    info = _trace(a)
    return _finish(a, _sweep_claims(a, info))


_ALTERNATIVES = ('ignore_mirrors', 'keep_source_color', 'ungated',
                 'consecutive', 'pixel_centers', 'directed')


def _evidence(a, info=None, claims=None):
    info = _trace(a) if info is None else info
    claims = _pair_claims(a, info) if claims is None else claims
    flags = {}
    for alternative in _ALTERNATIVES:
        weaker = _pair_claims(a, info, alternative)
        if alternative == 'ungated':
            flags[alternative] = not np.array_equal(claims.any(axis=0), weaker.any(axis=0))
        else:
            flags[alternative] = not np.array_equal(claims, weaker)
    ray_cells = np.zeros(a.shape, dtype=bool)
    for f in info['flights']:
        length = abs(f['end'][0]-f['start'][0])
        for k in range(length+1):
            ray_cells[f['start'][0]+k*f['direction'][0],
                      f['start'][1]+k*f['direction'][1]] = True
    lit = claims.any(axis=0)
    flags['dark_boundary'] = bool(np.any(ray_cells & (a == info['background']) & ~lit))
    flags['off_ray_fill'] = bool(np.any(lit & ~ray_cells))
    return flags


def _key(a):
    return a.shape, a.astype(np.uint8).tobytes()


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be {low}..{high}')
    return int(value)


def _transform(a, symmetry):
    b = np.rot90(a, symmetry % 4)
    return (b[:, ::-1] if symmetry >= 4 else b).copy()


def _layout(rng, n, bounces, mode, colors, safe=False, variant=0):
    """Construct new geometry, never run or chain parent generators.

    Normal construction uses a fan of offset launch intervals. Mirror planes
    can be shared or staggered. Cropping retains all visible object endpoints;
    tails are then traced afresh to the actual canvas boundary.
    """
    bg, source_color, mirror_color, other_color = colors
    qs = [0]
    for _ in range(n-1):
        qs.append(qs[-1] + (4 if safe else int(rng.integers(3, 7))))
    starts = [0 if safe else int(rng.integers(-2, 3)) for _ in qs]
    if safe:
        turn_row = 8 if mode == 'skip' else 5
        left = -2
    else:
        turn_row = int(rng.integers(5, 10 if n < 4 else 8))
        left = -int(rng.integers(1, 5))
    objects = []
    for i, q in enumerate(qs):
        if mode in ('join', 'scatter'):
            shade = source_color if i % 2 == 0 else other_color
        else:
            shade = source_color
        r, c = starts[i], q+starts[i]
        objects.append(([(r, c), (r-1, c), (r, c-1)], shade))
        count = bounces
        if mode == 'partial' and not safe:
            count = int(rng.integers(0, bounces+1))
        if not count:
            continue
        early = mode == 'skip' and i == n//2
        d = turn_row
        if early:
            d = 3 if safe else turn_row-int(rng.integers(3, 6))
        elif mode in ('scatter', 'partial') and not safe:
            d += int(rng.integers(-2, 3))
        new_color = other_color if early else mirror_color
        if mode == 'split':
            new_color = mirror_color if i % 2 == 0 else other_color
        elif mode == 'scatter' and not safe:
            new_color = int(rng.choice(colors[1:]))
        radius = 0 if early or safe else int(rng.choice([0, 1, 2, 3]))
        objects.append(([(d+k, q+d+1) for k in range(-radius, radius+1)], new_color))
        if count >= 2 and not early:
            col = left
            if mode in ('scatter', 'partial') and not safe:
                col += int(rng.integers(-1, 2))
            row = q+2*d-col+1
            radius = 1 if safe else int(rng.choice([0, 1, 2, 3]))
            final_color = source_color
            if mode == 'scatter' and not safe:
                final_color = int(rng.choice(colors[1:]))
            objects.append(([(row, col+k) for k in range(-radius, radius+1)], final_color))
    cells = [point for points, _ in objects for point in points]
    rmin, cmin = np.min(cells, axis=0)
    rmax, cmax = np.max(cells, axis=0)
    spans = (int(rmax-rmin+1), int(cmax-cmin+1))
    _need(max(spans) <= 30, 'construction does not fit')
    dims, offsets = [], []
    for axis, span in enumerate(spans):
        if safe:
            extra = min(30-span, 4 + variant % 3)
            offset = (variant // (3 if axis else 1)) % (extra+1)
        else:
            maximum = min(30-span, 16 if bounces == 0 else 8)
            minimum = min(maximum, 3)
            extra = int(rng.integers(minimum, maximum+1))
            offset = int(rng.integers(extra+1))
        dims.append(span+extra)
        offsets.append(offset)
    a = np.full(tuple(dims), bg, dtype=int)
    for points, color in objects:
        for r, c in points:
            point = (int(r-rmin+offsets[0]), int(c-cmin+offsets[1]))
            _need(a[point] == bg, 'construction objects overlap')
            a[point] = color
    return a


def _append_mirror(rng, a, palette):
    """Broaden flights beyond the fan scaffold with one input-visible bounce."""
    info = _trace(a)
    tails = [info['flights'][route['flights'][-1]] for route in info['routes']]
    options = []
    for f in tails:
        length = abs(f['end'][0]-f['start'][0])
        options.extend((f, step, axis) for step in range(2, length)
                       for axis in (0, 1))
    rng.shuffle(options)
    for f, step, axis in options[:60]:
        contact = [f['start'][i]+step*f['direction'][i] for i in (0, 1)]
        midpoint = contact.copy()
        midpoint[axis] += f['direction'][axis]
        radius = int(rng.choice([0, 1, 2]))
        points = [tuple(midpoint[i]+(k if i != axis else 0) for i in (0, 1))
                  for k in range(-radius, radius+1)]
        if not all(_inside(p, a.shape) and a[p] == info['background'] for p in points):
            continue
        trial = a.copy()
        shade = int(rng.choice(palette))
        for point in points:
            trial[point] = shade
        try:
            new_info = _trace(trial)
            if len(new_info['mirrors']) != len(info['mirrors'])+1:
                continue
            _finish(trial, _sweep_claims(trial, new_info))
        except SceneError:
            continue
        return trial
    return None


def _make_pair(rng, plan, colors, seen, required=()):
    # Random attempts are bounded; the fallback enumerates clear layouts and
    # placements. Required evidence is always checked on visible arrays.
    for attempt in range(256):
        safe = attempt >= 160
        if safe:
            n, bounces, mode = plan['fallback']
        else:
            n = int(rng.choice(plan['counts']))
            bounces = int(rng.choice(plan['bounces']))
            mode = str(rng.choice(plan['modes']))
        try:
            a = _layout(rng, n, bounces, mode, colors, safe, attempt-160)
            a = _transform(a, int(rng.integers(8)))
            if not safe and plan.get('extend', False):
                extended = _append_mirror(rng, a, colors[1:])
                if extended is None and plan.get('must_extend', False):
                    continue
                if extended is not None:
                    a = extended
            info = _trace(a)
            claims = _sweep_claims(a, info)
            output = _finish(a, claims)
        except SceneError:
            continue
        if _key(a) in seen or np.array_equal(a, output):
            continue
        if required:
            flags = _evidence(a, info, claims)
            if not all(flags[name] for name in required):
                continue
        seen.add(_key(a))
        return {'input': a, 'output': output}
    raise RuntimeError('bounded construction exhausted its valid placements')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    palette = [int(c) for c in rng.permutation(10)]
    bg1, s, c, d, bg2 = palette[:5]
    seen = set()
    skip_plan = dict(counts=[3], bounces=[1], modes=['skip'], fallback=(3, 1, 'skip'))
    join_plan = dict(counts=[2], bounces=[2], modes=['join'], fallback=(2, 2, 'join'))
    first = _make_pair(rng, skip_plan, [bg1, s, c, d], seen,
                       ('consecutive', 'pixel_centers', 'ignore_mirrors',
                        'dark_boundary', 'off_ray_fill'))
    second = _make_pair(rng, join_plan, [bg2, s, d, c], seen,
                        ('keep_source_color', 'ungated', 'directed',
                         'dark_boundary', 'off_ray_fill'))
    puzzle = {'train': [first, second], 'test': []}
    regime = str(rng.choice(['fan', 'echo', 'uneven', 'fold'], p=[.35, .20, .30, .15]))
    for split, count in (('train', num_train), ('test', num_test)):
        while len(puzzle[split]) < count:
            local = [int(x) for x in rng.permutation(10)]
            # Visible copying allows palette/background changes at pair scope.
            # Extras also include unchanged-color reflections and monochrome scenes.
            if rng.random() < .25:
                local[2] = local[1]
            if split == 'train':
                plan = dict(counts=[1, 2, 3, 4, 5], bounces=[0, 1, 2],
                            modes=['ribbon', 'join', 'split', 'partial', 'scatter'],
                            fallback=(2, 1, 'ribbon'), extend=bool(rng.random() < .18))
            elif regime == 'fan':
                plan = dict(counts=[4, 5, 6], bounces=[1],
                            modes=['ribbon', 'join', 'split', 'partial'],
                            fallback=(4, 1, 'ribbon'))
            elif regime == 'echo':
                plan = dict(counts=[1], bounces=[2], modes=['ribbon'],
                            fallback=(1, 2, 'ribbon'))
            elif regime == 'uneven':
                plan = dict(counts=[2, 3, 4, 5], bounces=[1, 2],
                            modes=['scatter', 'partial', 'skip'],
                            fallback=(3, 1, 'skip'))
                # A short interior flight needs its own color to stay unmatched.
                local[2], local[3] = [x for x in range(10) if x not in (local[0], local[1])][:2]
            else:
                plan = dict(counts=[1, 2, 3], bounces=[2], modes=['ribbon', 'join'],
                            fallback=(2, 2, 'join'), extend=True, must_extend=True)
            puzzle[split].append(_make_pair(rng, plan, local[:4], seen))
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    ntrain, ntest = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= ntrain <= 10 and 1 <= ntest <= 4 and ntrain > ntest
    seen, flags = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, output = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == output.shape
            assert _key(a) not in seen, 'duplicate input'
            seen.add(_key(a))
            info = _trace(a)
            geometric = _pair_claims(a, info)
            assert np.array_equal(output, _finish(a, geometric)), 'incorrect light sheet'
            assert np.array_equal(geometric, _sweep_claims(a, info)), 'sweep/pair disagreement'
            occupied = a != info['background']
            assert np.array_equal(a[occupied], output[occupied]), 'changed source or mirror'
            assert set(np.unique(output)) <= set(np.unique(a)), 'a hidden paint color was introduced'
            for f in info['flights']:
                assert f['q'] == f['end'][1]-f['slope']*f['end'][0]
                assert (f['lo']-f['q']) % 2 == (f['hi']-f['q']) % 2 == 0
                assert _inside(f['start'], a.shape) and _inside(f['end'], a.shape)
            flags[split].append(_evidence(a, info, geometric))
    for name in _ALTERNATIVES:
        if any(f[name] for f in flags['test']):
            assert any(f[name] for f in flags['train']), f'missing training witness: {name}'
    assert any(f['ignore_mirrors'] for f in flags['train']), 'A must affect the answer'
    assert any(f['off_ray_fill'] for f in flags['train']), 'B must add actual between-flight area'
    assert any(f['dark_boundary'] for f in flags['train']), 'must teach that rays are invisible'


def _fingerprint(puzzle):
    return tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                 for split in ('train', 'test') for p in puzzle[split])


def _normalized(a):
    bg = int(np.argmax(np.bincount(a.ravel(), minlength=10)))
    rows, cols = np.where(a != bg)
    crop = a[rows.min():rows.max()+1, cols.min():cols.max()+1]
    mapping = {bg: 0}
    return crop.shape, bytes(mapping.setdefault(int(x), len(mapping)) for x in crop.flat)


def _regressions():
    # Hand-calculated continuous-interval geometry, including odd scanlines.
    a = np.zeros((5, 5), dtype=int)
    fs = [_flight((0, 0), (2, 2), (1, 1), 3, 3, 0),
          _flight((0, 4), (0, 4), (1, 1), 3, 3, 1)]
    info = dict(background=0, flights=fs)
    expected = np.zeros_like(a)
    expected[1, 3] = 3  # t=4, strictly 0<q=2<4; endpoints inclusive.
    assert np.array_equal(_finish(a, _sweep_claims(a, info)), expected)
    assert np.array_equal(_finish(a, _pair_claims(a, info)), expected)
    # Two long parallel flights bracket a short middle flight even after it ends.
    fs = [_flight((0, 0), (4, 4), (1, 1), 3, 3, 0),
          _flight((0, 2), (1, 3), (1, 1), 3, 3, 1),
          _flight((0, 4), (4, 8), (1, 1), 3, 3, 2)]
    a = np.zeros((5, 9), dtype=int)
    info = dict(background=0, flights=fs)
    true = _pair_claims(a, info)
    assert true[3, 2, 4] and not _pair_claims(a, info, 'consecutive')[3, 2, 4]
    # At (2,3), t=5 lies between the boundary pixel centers t=4 and t=6.
    assert true[3, 2, 3] and not _pair_claims(a, info, 'pixel_centers')[3, 2, 3]
    # Color disagreement prevents a bracket even when its geometry is identical.
    info['flights'] = [fs[0], dict(fs[2], color=7)]
    assert not _pair_claims(a, info).any()
    assert _pair_claims(a, info, 'ungated').any()
    # A literal optical fixture, computed without generating a puzzle.
    a = np.zeros((10, 12), dtype=int)
    a[0, 1] = a[1, 0] = a[1, 1] = 4
    a[6, 3:8] = 7
    info = _trace(a)
    flights = info['flights']
    assert [(f['start'], f['end'], f['color']) for f in flights] == [
        ((1, 1), (4, 4), 4), ((5, 5), (0, 10), 7)]
    assert np.array_equal(solve(a), a)  # One flight of each slope cannot bracket.
    # Conflict is rejected rather than resolved by color number or iteration order.
    claims = np.zeros((10, 2, 2), dtype=bool)
    claims[3, 0, 0] = claims[7, 0, 0] = True
    try:
        _finish(np.zeros((2, 2), dtype=int), claims)
    except SceneError:
        pass
    else:
        raise AssertionError('ambiguous paint order accepted')


def self_test():
    """Bounded semantic, determinism, count, and actual variation checks."""
    _regressions()
    fingerprints, shapes, structures = set(), set(), set()
    sources, lengths, bounces, backgrounds, directions = set(), set(), set(), set(), set()
    changed = echo_pairs = extended_pairs = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                info = _trace(a)
                shapes.add(a.shape)
                structures.add(_normalized(a))
                sources.add(len(info['sources']))
                backgrounds.add(info['background'])
                lengths.update(len(m['cells']) for m in info['mirrors'])
                directions.update(s['direction'] for s in info['sources'])
                bounces.update(len(r['contacts']) for r in info['routes'])
                changed += int(np.count_nonzero(a != pair['output']))
                echo_pairs += int(len(info['sources']) == 1 and np.any(a != pair['output']))
                extended_pairs += any(len(r['contacts']) >= 3 for r in info['routes'])
                assert np.array_equal(solve(a.copy()), pair['output'])
    assert len(fingerprints) == 42 and len(structures) >= 150 and len(shapes) >= 35
    assert {1, 2, 3, 4} <= sources and max(sources) >= 5
    assert {1, 3, 5, 7} <= lengths
    assert {0, 1, 2, 3} <= bounces
    assert backgrounds == set(range(10)) and len(directions) == 4
    assert echo_pairs >= 3 and extended_pairs >= 1
    # Every legal explicit count combination, including the fourteen-pair maximum.
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain-1)+1):
            p = generate_puzzle(101, ntrain, ntest)
            assert len(p['train']) == ntrain and len(p['test']) == ntest
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Correct-looking but incomplete labels and inadequate demonstrations fail.
    p = generate_puzzle(7, 2, 1)
    bad = {split: [dict(pair) for pair in p[split]] for split in p}
    bad['test'][0]['output'] = bad['test'][0]['output'].copy()
    changed_point = tuple(np.argwhere(bad['test'][0]['input'] != bad['test'][0]['output'])[0])
    bad['test'][0]['output'][changed_point] = bad['test'][0]['input'][changed_point]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('missing light cell accepted')
    # Replace training with visibly ungated monochrome fans; witnesses disappear.
    rng, seen = np.random.default_rng(23), set()
    plan = dict(counts=[2], bounces=[0], modes=['ribbon'], fallback=(2, 0, 'ribbon'))
    inadequate = dict(train=[_make_pair(rng, plan, [0, 3, 4, 5], seen) for _ in range(2)],
                      test=p['test'])
    try:
        validate_generated_puzzle(inadequate)
    except AssertionError:
        pass
    else:
        raise AssertionError('insufficient training accepted')
    return dict(seeds=42, distinct_puzzles=len(fingerprints), grid_shapes=len(shapes),
                normalized_input_structures=len(structures), source_counts=sorted(sources),
                mirror_lengths=sorted(lengths), reflection_counts=sorted(bounces),
                self_bracketing_pairs=echo_pairs, extended_pairs=extended_pairs,
                changed_cells=changed, independent_solver=False)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
