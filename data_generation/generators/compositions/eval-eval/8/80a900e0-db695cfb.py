"""Diagonal loom: finite threads stitch the interiors of cap-defined channels.

Only NumPy and the standard library are used. No files, examples, or RNG state
are consulted by solve(). Generation and self_test() retain no execution traces.
"""

from collections import Counter
import hashlib
import json

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
RADII = (3, 4, 5)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _outline(center, radius):
    r, c = center
    return {(r + a, c + b) for a in range(-radius, radius + 1)
            for b in range(-radius, radius + 1) if abs(a) + abs(b) == radius}


def _interior(center, radius):
    r, c = center
    return {(r + a, c + b) for a in range(-radius + 1, radius)
            for b in range(-radius + 1, radius) if abs(a) + abs(b) < radius}


def _cap(center, radius, direction):
    r, c = center
    dr, dc = direction
    return [(r + dr * (k + 1), c + dc * (radius - k + 1))
            for k in range(radius + 1)]


def _base(shape, colors):
    return np.asarray(colors, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _background(grid):
    parity = np.indices(grid.shape).sum(axis=0) % 2
    colors = []
    for k in (0, 1):
        values = grid[parity == k]
        counts = np.bincount(values.astype(int), minlength=10)
        color = int(counts.argmax())
        assert counts[color] * 2 > len(values), "checkerboard majority is not unique"
        colors.append(color)
    assert colors[0] != colors[1]
    return tuple(colors)


def _channel(point, center, radius, caps):
    """Return the unique outward hollow channel, excluding cap and rails."""
    for direction in caps:
        u = (point[0] - center[0]) * direction[0]
        v = (point[1] - center[1]) * direction[1]
        if u + v > radius + 2 and abs(u - v) < radius:
            return direction
    return None


def _rail_paths(center, radius, caps, shape):
    paths = {}
    for direction in caps:
        cap = _cap(center, radius, direction)
        pair = []
        for endpoint in (cap[0], cap[-1]):
            p = (endpoint[0] + direction[0], endpoint[1] + direction[1])
            path = []
            while _inside(p, shape):
                path.append(p)
                p = (p[0] + direction[0], p[1] + direction[1])
            pair.append(path)
        paths[direction] = pair
    return paths


def _bar(point, direction, center, radius, shape):
    """Walk across a channel; rail cells themselves retain their rail color."""
    result = {point}
    transverse = (direction[0], -direction[1])
    for sign in (-1, 1):
        p = point
        while True:
            p = (p[0] + sign * transverse[0], p[1] + sign * transverse[1])
            if not _inside(p, shape):
                break
            u = (p[0] - center[0]) * direction[0]
            v = (p[1] - center[1]) * direction[1]
            if abs(u - v) >= radius:
                break
            result.add(p)
    return result


def _segment(a, b):
    assert a[0] != b[0] and abs(a[0] - b[0]) == abs(a[1] - b[1])
    steps = abs(a[0] - b[0])
    dr = 1 if b[0] > a[0] else -1
    dc = 1 if b[1] > a[1] else -1
    return [(a[0] + k * dr, a[1] + k * dc) for k in range(steps + 1)]


def _slope(segment):
    a, b = segment[0], segment[-1]
    return 1 if (b[0] - a[0]) * (b[1] - a[1]) > 0 else -1


def _grid(value):
    grid = np.asarray(value)
    assert grid.ndim == 2 and all(1 <= x <= 30 for x in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((grid >= 0) & (grid <= 9))
    return grid.astype(np.int8, copy=False)


def _analyze(grid):
    """Segment the input and derive every role from visible geometry."""
    grid = _grid(grid)
    shape = grid.shape
    background = _background(grid)
    base = _base(shape, background)
    foreground = set(map(tuple, np.argwhere(grid != base)))
    assert all(int(grid[p]) not in background for p in foreground)
    diamonds = []
    for radius in RADII:
        h, w = shape[0] - 2 * radius, shape[1] - 2 * radius
        if h <= 0 or w <= 0:
            continue
        offsets = sorted(_outline((0, 0), radius))
        a, b = offsets[0]
        colors = grid[radius + a:radius + a + h, radius + b:radius + b + w]
        mask = (colors != background[0]) & (colors != background[1])
        for a, b in offsets[1:]:
            mask &= grid[radius + a:radius + a + h, radius + b:radius + b + w] == colors
        for r, c in np.argwhere(mask):
            diamonds.append(((int(r) + radius, int(c) + radius), radius, int(colors[r, c])))
    assert len(diamonds) == 1, "one complete diamond identifies the primary dot color"
    center, radius, structural = diamonds[0]
    interior = _interior(center, radius)
    assert not interior & foreground, "the loom head stays hollow"
    core = _outline(center, radius)
    caps = {}
    for direction in DIRECTIONS:
        points = _cap(center, radius, direction)
        assert all(_inside(p, shape) for p in points), "caps are never clipped in the input"
        values = [int(grid[p]) for p in points]
        if len(set(values)) == 1 and values[0] not in (*background, structural):
            caps[direction] = values[0]
            core.update(points)
    assert caps
    dots = foreground - core
    primary = sorted(p for p in dots if grid[p] == structural)
    secondary = {p: int(grid[p]) for p in dots if grid[p] != structural}
    degree = Counter()
    segments = []
    for i, a in enumerate(primary):
        for b in primary[i + 1:]:
            if a[0] != b[0] and abs(a[0] - b[0]) == abs(a[1] - b[1]):
                degree[a] += 1
                degree[b] += 1
                span = _segment(a, b)
                assert len(span) >= 3
                segments.append(span)
    assert segments and all(n == 1 for n in degree.values()), "endpoint partners must be unique"
    endpoints = {p for s in segments for p in (s[0], s[-1])}
    span_union = set()
    for s in segments:
        assert not set(s) & (core | interior | span_union)
        span_union.update(s)
    paths = _rail_paths(center, radius, caps, shape)
    rails = {p for pair in paths.values() for path in pair for p in path}
    assert not rails & (foreground | span_union)
    # The finite spans lie on the same parity lattice as the outline. This
    # makes a transverse stitch meet both conceptual rails exactly.
    for s in segments:
        assert (sum(s[0]) - sum(center) - radius) % 2 == 0
    active, bars = {}, {}
    for p, color in secondary.items():
        containing = [s for s in segments if p in s[1:-1]]
        direction = _channel(p, center, radius, caps)
        if not containing or direction is None:
            continue
        assert len(containing) == 1
        assert _slope(containing[0]) == direction[0] * direction[1]
        assert (sum(p) - sum(center) - radius) % 2 == 0
        bar = _bar(p, direction, center, radius, shape)
        assert not bar & endpoints
        assert bar & set(secondary) == {p}, "a stitch cannot overwrite another original dot"
        assert all(not bar & old for old in bars.values())
        active[p] = direction
        bars[p] = bar
    assert active, "each generated pair contains a useful stitch"
    return dict(grid=grid, base=base, background=background, center=center,
                radius=radius, structural=structural, caps=caps, core=core,
                primary=set(primary), secondary=secondary, segments=segments,
                span_union=span_union, endpoints=endpoints, paths=paths,
                rails=rails, active=active, bars=bars)


def _paint(a):
    out = a['grid'].copy()
    for p in a['span_union'] - set(a['secondary']):
        out[p] = a['structural']
    for direction, pair in a['paths'].items():
        for path in pair:
            for p in path:
                out[p] = a['caps'][direction]
    for p, bar in a['bars'].items():
        for q in bar:
            out[q] = a['secondary'][p]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete rails and finite threads, then stitch only qualifying pins."""
    return _paint(_analyze(input_grid))


def _cellwise_prediction(a):
    """Closed-form masks check the walking renderer, sharing only segmentation.

    This is an internal consistency check, not an independent semantic oracle.
    """
    grid = a['grid']
    rows, cols = np.indices(grid.shape)
    out = grid.copy()
    for span in a['segments']:
        p, q = sorted((span[0], span[-1]))
        mask = ((cols - p[1] == _slope(span) * (rows - p[0])) &
                (rows >= p[0]) & (rows <= q[0]) & (grid == a['base']))
        out[mask] = a['structural']
    for direction, color in a['caps'].items():
        u = (rows - a['center'][0]) * direction[0]
        v = (cols - a['center'][1]) * direction[1]
        rails = (abs(u - v) == a['radius']) & (np.minimum(u, v) >= 2)
        out[rails] = color
    for p, direction in a['active'].items():
        u = (rows - a['center'][0]) * direction[0]
        v = (cols - a['center'][1]) * direction[1]
        total = ((p[0] - a['center'][0]) * direction[0] +
                 (p[1] - a['center'][1]) * direction[1])
        out[(u + v == total) & (abs(u - v) < a['radius'])] = a['secondary'][p]
    return out


def _compatible(span, spans, occupied):
    if set(span) & occupied:
        return False
    for old in spans:
        if set(span) & set(old):
            return False
        for p in (span[0], span[-1]):
            for q in (old[0], old[-1]):
                if abs(p[0] - q[0]) == abs(p[1] - q[1]):
                    return False
    return True


def _lane_candidates(shape, center, radius, direction):
    lanes = {}
    for r in range(shape[0]):
        for c in range(shape[1]):
            u = (r - center[0]) * direction[0]
            v = (c - center[1]) * direction[1]
            if u + v > radius + 2 and abs(u - v) < radius and (u - v - radius) % 2 == 0:
                lanes.setdefault(u - v, []).append((r, c))
    result = []
    for lane in lanes.values():
        lane.sort(key=lambda p: (p[0] - center[0]) * direction[0])
        for start in range(len(lane) - 2):
            for end in range(start + 2, len(lane)):
                result.append(lane[start:end + 1])
    return result


def _evidence(a, out):
    """Pixel witnesses for the two gates and endpoint-only propagation."""
    base, grid = a['base'], a['grid']
    positive, bounded, contrast = set(), set(), False
    for p, direction in a['active'].items():
        if any(q != p and grid[q] == base[q] and out[q] == a['secondary'][p]
               for q in a['bars'][p]):
            positive.add(direction)
            contrast |= a['secondary'][p] != a['caps'][direction]
            transverse = -direction[0] * direction[1]
            if any(c - p[1] == transverse * (r - p[0]) and
                   (r, c) not in a['bars'][p] | a['rails'] and
                   grid[r, c] == base[r, c] and out[r, c] == base[r, c]
                   for r in range(grid.shape[0]) for c in range(grid.shape[1])):
                bounded.add(direction)
    inside_inert = outside_on_span = beyond = False
    for p, color in a['secondary'].items():
        if p in a['active']:
            continue
        direction = _channel(p, a['center'], a['radius'], a['caps'])
        if direction is not None and p not in a['span_union']:
            bar = _bar(p, direction, a['center'], a['radius'], grid.shape)
            witness = any(q != p and grid[q] == base[q] and out[q] == base[q] for q in bar)
            inside_inert |= witness
            beyond |= witness and any(
                _slope(s) == direction[0] * direction[1] and
                abs(p[0] - s[0][0]) == abs(p[1] - s[0][1]) for s in a['segments'])
        if p in a['span_union'] and direction is None:
            s = next(s for s in a['segments'] if p in s)
            transverse = -_slope(s)
            outside_on_span |= any(
                grid[r, c] == base[r, c] and out[r, c] == base[r, c]
                for r in range(grid.shape[0]) for c in range(grid.shape[1])
                if c - p[1] == transverse * (r - p[0]))
    hollow = set()
    for direction in a['caps']:
        if any(_channel((r, c), a['center'], a['radius'], a['caps']) == direction and
               (r + c - sum(a['center']) - a['radius']) % 2 == 0 and
               grid[r, c] == base[r, c] and out[r, c] == base[r, c]
               for r in range(grid.shape[0]) for c in range(grid.shape[1])):
            hollow.add(direction)
    cross_thread = any(q in a['span_union'] and q != p and a['grid'][q] == base[q]
                       for p, bar in a['bars'].items() for q in bar)
    return dict(positive=positive, bounded=bounded, contrast=contrast, inside_inert=inside_inert,
                outside_on_span=outside_on_span, beyond=beyond,
                hollow=hollow, cross_thread=cross_thread)


def _make_pair(rng, max_radius, mask, teaching, edge, need_cross):
    """Bounded free placement; no stored layout templates or parent imports."""
    for attempt in range(160):
        radius = max_radius if teaching else int(rng.integers(3, max_radius + 1))
        low = 2 * radius + 11
        shape = tuple(int(rng.integers(low, 31)) for _ in range(2))
        margin = radius + 5
        center = [int(rng.integers(margin, extent - margin)) for extent in shape]
        if edge:
            axis = int(rng.integers(2))
            center[axis] = radius + 1 if rng.random() < .5 else shape[axis] - radius - 2
        center = tuple(center)
        palette = list(map(int, rng.permutation(10)))
        background, structural = tuple(palette[:2]), palette[2]
        directions = [d for i, d in enumerate(DIRECTIONS) if mask & (1 << i)]
        cap_bank = palette[3:3 + int(rng.integers(1, min(4, len(directions)) + 1))]
        caps = {d: int(rng.choice(cap_bank)) for d in directions}
        marker_bank = list(map(int, rng.choice(palette[3:], int(rng.integers(1, 4)), replace=False)))
        grid = _base(shape, background)
        core = _outline(center, radius)
        for p in core:
            grid[p] = structural
        for direction, color in caps.items():
            for p in _cap(center, radius, direction):
                core.add(p)
                grid[p] = color
        paths = _rail_paths(center, radius, caps, shape)
        rails = {p for pair in paths.values() for path in pair for p in path}
        occupied = core | rails | _interior(center, radius)
        candidates = {d: _lane_candidates(shape, center, radius, d) for d in directions}
        available = [d for d in directions if candidates[d]]
        if teaching and len(available) != len(directions) or not available:
            continue
        required = directions.copy() if teaching else list(
            rng.choice(len(available), int(rng.integers(1, len(available) + 1)), replace=False))
        if not teaching:
            required = [available[int(i)] for i in required]
        rng.shuffle(required)
        spans, tags = [], []
        failed = False
        # Every required cap gets one finite longitudinal thread.
        for direction in required:
            pool = candidates[direction]
            for index in rng.permutation(len(pool)):
                span = pool[int(index)]
                if _compatible(span, spans, occupied):
                    spans.append(span)
                    tags.append(direction)
                    break
            else:
                failed = True
                break
        if failed:
            continue
        # Occasionally add a second/third lane, with crossbars allowed to
        # overpaint newly completed thread cells, as in a woven crossing.
        extra = 1 if need_cross else int(rng.choice((0, 0, 1, 2)))
        for _ in range(extra):
            direction = required[int(rng.integers(len(required)))]
            pool = candidates[direction]
            for index in rng.permutation(len(pool)):
                span = pool[int(index)]
                if _compatible(span, spans, occupied):
                    spans.append(span)
                    tags.append(direction)
                    break
        if need_cross and len(spans) == len(required):
            continue
        outside_required = teaching or rng.random() < .65
        if outside_required:
            for _ in range(700):
                steps = int(rng.integers(2, min(10, *shape)))
                r = int(rng.integers(shape[0] - steps))
                slope = int(rng.choice((-1, 1)))
                c = int(rng.integers(shape[1] - steps)) + (steps if slope == -1 else 0)
                span = _segment((r, c), (r + steps, c + slope * steps))
                if (sum(span[0]) - sum(center) - radius) % 2:
                    continue
                if not any(_channel(p, center, radius, caps) is None for p in span[1:-1]):
                    continue
                if _compatible(span, spans, occupied):
                    spans.append(span)
                    tags.append(None)
                    break
            else:
                continue
        endpoints = {p for s in spans for p in (s[0], s[-1])}
        span_union = {p for s in spans for p in s}
        dots, bar_union = {}, set()
        active_directions = set()
        span_order = list(range(len(spans)))
        rng.shuffle(span_order)
        for index in span_order:
            span, direction = spans[index], tags[index]
            if direction is None:
                pool = [p for p in span[1:-1] if _channel(p, center, radius, caps) is None]
                dots[pool[int(rng.integers(len(pool)))]] = int(rng.choice(marker_bank))
                continue
            pool = []
            for p in span[1:-1]:
                bar = _bar(p, direction, center, radius, shape)
                if len(bar) >= 2 and not bar & (endpoints | bar_union | set(dots)):
                    pool.append((p, bar))
            if not pool:
                continue
            count = 1 if direction not in active_directions else int(rng.integers(0, 3))
            for j in rng.permutation(len(pool)):
                if count == 0:
                    break
                p, bar = pool[int(j)]
                if bar & bar_union or p in bar_union:
                    continue
                dots[p] = int(rng.choice(marker_bank))
                bar_union.update(bar)
                active_directions.add(direction)
                count -= 1
        if not set(required) <= active_directions:
            continue
        # A pin beyond a finite thread, still inside its channel, is a direct
        # witness against infinite primary lines and unconditional stitching.
        inert_pool = []
        for direction in available:
            for lane_span in candidates[direction]:
                for p in (lane_span[0], lane_span[-1]):
                    if p in span_union | occupied | bar_union | set(dots):
                        continue
                    bar = _bar(p, direction, center, radius, shape)
                    if bar & (endpoints | set(dots)) or len(bar) < 2:
                        continue
                    if any(_slope(s) == direction[0] * direction[1] and
                           abs(p[0] - s[0][0]) == abs(p[1] - s[0][1]) for s in spans):
                        inert_pool.append(p)
        if teaching and not inert_pool:
            continue
        if inert_pool:
            dots[inert_pool[int(rng.integers(len(inert_pool)))]] = int(rng.choice(marker_bank))
        # Independently optional unpaired primary dots and off-span pins.
        all_primary = list(endpoints)
        n_primary = int(rng.integers(3))
        n_secondary = int(rng.integers(4))
        for flat in rng.permutation(grid.size):
            if n_primary == 0 and n_secondary == 0:
                break
            p = (int(flat) // shape[1], int(flat) % shape[1])
            if p in occupied | span_union | bar_union | set(dots):
                continue
            if n_primary and not any(abs(p[0] - q[0]) == abs(p[1] - q[1]) for q in all_primary):
                all_primary.append(p)
                n_primary -= 1
            elif n_secondary:
                dots[p] = int(rng.choice(marker_bank))
                n_secondary -= 1
        for p in all_primary:
            grid[p] = structural
        for p, color in dots.items():
            grid[p] = color
        try:
            a = _analyze(grid)
            if a['center'] != center or a['radius'] != radius or a['caps'] != caps:
                continue
            if {frozenset(s) for s in a['segments']} != {frozenset(s) for s in spans}:
                continue
            out = _paint(a)
            evidence = _evidence(a, out)
            if teaching and not (set(directions) <= evidence['positive'] & evidence['hollow'] & evidence['bounded'] and
                                 evidence['contrast'] and evidence['inside_inert'] and
                                 evidence['outside_on_span'] and evidence['beyond']):
                continue
            if need_cross and not evidence['cross_thread']:
                continue
            return {'input': grid, 'output': out}
        except AssertionError:
            continue
    raise RuntimeError('bounded loom construction exhausted')


def _counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError('illegal example count')
    if num_train is None:
        if num_test is None:
            num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    num_train, num_test = _counts(rng, num_train, num_test)
    max_radius = int(rng.integers(3, 6))
    # Radius three has only two interior lanes of either slope. Opposite
    # caps consume both unique endpoint lines, leaving no third line for a
    # cross-thread teaching witness. Put that witness on a feasible mask.
    first = int(rng.choice([m for m in range(1, 15)
                            if max_radius > 3 or m not in (5, 10)]))
    masks = [first, 15 ^ first]
    witnesses = list(map(int, rng.choice(num_train, 2, replace=False)))
    teaching = dict(zip(witnesses, masks))
    puzzle = {'train': [], 'test': []}
    used = set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            witness = split == 'train' and index in teaching
            mask = teaching[index] if witness else int(rng.integers(1, 16))
            for _ in range(16):
                pair = _make_pair(rng, max_radius, mask, witness, split == 'test',
                                  witness and index == witnesses[0])
                key = _key(pair['input'])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct input budget exhausted')
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, positive, hollow, bounded = set(), {}, {}, {}
    flags = Counter()
    max_test = 0
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            a = _analyze(inp)
            assert np.array_equal(out, _paint(a))
            assert np.array_equal(out, _cellwise_prediction(a))
            foreground = inp != a['base']
            assert np.array_equal(out[foreground], inp[foreground])
            expected_changed = a['rails'] | a['span_union'] | set().union(*a['bars'].values())
            actual_changed = set(map(tuple, np.argwhere(inp != out)))
            assert actual_changed <= expected_changed
            evidence = _evidence(a, out)
            if split == 'train':
                for d in evidence['positive']:
                    positive[d] = max(positive.get(d, 0), a['radius'])
                for d in evidence['hollow']:
                    hollow[d] = max(hollow.get(d, 0), a['radius'])
                for d in evidence['bounded']:
                    bounded[d] = max(bounded.get(d, 0), a['radius'])
                flags.update({k: int(evidence[k]) for k in
                              ('inside_inert', 'outside_on_span', 'beyond', 'cross_thread', 'contrast')})
            else:
                max_test = max(max_test, a['radius'])
                r, c = a['center']
                assert min(r, c, inp.shape[0] - 1 - r, inp.shape[1] - 1 - c) == a['radius'] + 1
    assert set(positive) == set(hollow) == set(bounded) == set(DIRECTIONS)
    assert min(positive.values()) >= max_test and min(hollow.values()) >= max_test
    assert all(flags[name] for name in ('inside_inert', 'outside_on_span', 'beyond', 'cross_thread', 'contrast'))
    return True


def _measure(puzzle):
    records = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = _analyze(pair['input'])
            counts = Counter(a['active'].values())
            e = _evidence(a, pair['output'])
            records.append(dict(split=split, shape=pair['input'].shape, radius=a['radius'],
                                caps=len(a['caps']), mask=sum(1 << DIRECTIONS.index(d) for d in a['caps']),
                                segments=len(a['segments']), lengths=[len(s) for s in a['segments']],
                                colors=len(set(a['secondary'].values())), active=len(a['active']),
                                unpaired=len(a['primary'] - a['endpoints']),
                                inactive=len(a['secondary']) - len(a['active']),
                                cross_thread=e['cross_thread'],
                                max_triggers_per_cap=max(counts.values()),
                                rail_lengths=[len(p) for pair in a['paths'].values() for p in pair],
                                layout=(pair['input'].shape, a['center'], a['radius'],
                                        tuple(sorted(a['caps'])), tuple(tuple(s) for s in a['segments']))))
    return records


def self_test():
    """42 streamed seeds, count boundaries, mutations, and geometric variation.

    Ordinary puzzles are streamed. At most three puzzle objects are retained
    during determinism/mutation checks (<=75,600 raw int8 cell bytes). Only
    hashes/features are retained across seeds. Nothing is written here.
    """
    records, fingerprints = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        records.extend(_measure(puzzle))
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                digest.update(str(pair['input'].shape).encode())
                digest.update(pair['input'].tobytes())
                digest.update(pair['output'].tobytes())
        fingerprints.add(digest.digest())
    assert len(fingerprints) == 42
    assert {r['radius'] for r in records} == set(RADII)
    assert {r['caps'] for r in records} == {1, 2, 3, 4}
    assert {r['colors'] for r in records} >= {1, 2, 3}
    assert len({r['shape'] for r in records}) >= 40
    assert len({r['layout'] for r in records}) >= 180
    assert max(r['segments'] for r in records) >= 5
    assert max(r['active'] for r in records) >= 4
    assert max(n for r in records for n in r['lengths']) >= 10
    assert any(r['cross_thread'] for r in records if r['split'] == 'test')
    assert any(0 in r['rail_lengths'] for r in records if r['split'] == 'test')
    assert max(n for r in records for n in r['rail_lengths']) >= 12
    assert {r['unpaired'] for r in records} >= {0, 1, 2}
    assert max(r['max_triggers_per_cap'] for r in records) >= 3
    # A wide channel's stitch is clipped by the top canvas edge. Its input
    # cap is complete; one endpoint ray has zero continuation. This fixture
    # exercises solve outside the sampler's ordinary central teaching scenes.
    fixture = _base((23, 28), (0, 1))
    center, radius, direction = (6, 15), 5, (-1, -1)
    for p in _outline(center, radius):
        fixture[p] = 3
    for p in _cap(center, radius, direction):
        fixture[p] = 4
    fixture[0, 8] = fixture[2, 10] = 3
    fixture[1, 9] = 2
    a = _analyze(fixture)
    assert a['bars'][(1, 9)] == {(0, 10), (1, 9), (2, 8)}
    clipped = solve(fixture)
    assert clipped[0, 10] == clipped[2, 8] == 2 and clipped[3, 7] == 4
    assert np.array_equal(clipped, _cellwise_prediction(a))
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            puzzle = generate_puzzle(100 + a * 4 + b, a, b)
            assert (len(puzzle['train']), len(puzzle['test'])) == (a, b)
    for a, b in ((None, 1), (None, 4), (2, None), (10, None), (None, None)):
        puzzle = generate_puzzle(12345, a, b)
        assert (a is None or len(puzzle['train']) == a) and (b is None or len(puzzle['test']) == b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (4, 1.5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    original = generate_puzzle(0, 2, 1)
    repeat = generate_puzzle(0, 2, 1)
    assert all(_key(p[x]) == _key(q[x]) for s in original for p, q in
               zip(original[s], repeat[s]) for x in ('input', 'output'))
    # Mutate an actual stitch-only cell, checking that semantic damage is caught.
    a = _analyze(original['test'][0]['input'])
    out = original['test'][0]['output']
    p = next(q for pin, bar in a['bars'].items() for q in bar
             if q != pin and a['grid'][q] == a['base'][q])
    out[p] = a['grid'][p]
    try:
        validate_generated_puzzle(original)
    except AssertionError:
        pass
    else:
        raise AssertionError('missing stitch cell was accepted')
    return dict(seeds=42, pairs=len(records), shapes=len({r['shape'] for r in records}),
                layouts=len({r['layout'] for r in records}),
                radii=sorted({r['radius'] for r in records}),
                cap_counts=sorted({r['caps'] for r in records}),
                segment_counts=sorted({r['segments'] for r in records}),
                max_triggers_per_cap=max(r['max_triggers_per_cap'] for r in records),
                cross_thread_pairs=sum(r['cross_thread'] for r in records),
                clipped_stitch_fixture=True)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
