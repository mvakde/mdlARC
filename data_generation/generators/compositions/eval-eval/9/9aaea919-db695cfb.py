"""Diagonal harvest: finite bridges select perpendicular unary additions.

Only NumPy and the standard library are used. No files are read or written.
The pixel parser, construction and checks do not retain generation state.
"""
from __future__ import annotations

import hashlib
import numpy as np


_MASKS = (
    ((1,),),
    ((1, 1, 1),),
    ((1,), (1,), (1,)),
    ((0, 1, 0), (1, 1, 1), (0, 1, 0)),
    ((1, 1, 1), (1, 1, 1), (1, 1, 1)),
    ((1, 0, 0), (1, 0, 0), (1, 1, 1)),
)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected a 1..30 by 1..30 integer color grid")
    return a.astype(np.int8, copy=False)


def _components(a, bg):
    remaining = {(int(r), int(c)) for r, c in np.argwhere(a != bg)}
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        color = int(a[first])
        cells, queue = {first}, [first]
        while queue:
            r, c = queue.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in remaining and int(a[q]) == color:
                    remaining.remove(q)
                    cells.add(q)
                    queue.append(q)
        r0, c0 = map(min, zip(*cells))
        r1, c1 = map(max, zip(*cells))
        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        result.append(dict(color=color, cells=cells, mask=mask,
                           center=((r0 + r1) // 2, (c0 + c1) // 2)))
    return result


def _is_bar(component):
    return component['mask'].shape in ((1, 3), (3, 1)) and len(component['cells']) == 3


def _segment(a, b):
    if a > b:
        a, b = b, a
    dr, dc = b[0] - a[0], b[1] - a[1]
    if dr < 4 or abs(dc) != dr:
        raise ValueError("Endpoint centers must span at least five diagonal cells")
    slope = 1 if dc > 0 else -1
    return tuple((a[0] + k, a[1] + slope * k) for k in range(dr + 1)), slope


def _token_cells(center, mask):
    r, c = center
    h, w = mask.shape
    return {(r + int(y) - h // 2, c + int(x) - w // 2)
            for y, x in np.argwhere(mask)}


def _bundles(lower):
    """Maximal consecutive pitch positions; each bundle must be monochrome."""
    groups = []
    for k, token in sorted(lower.items()):
        if not groups or k != groups[-1][-1][0] + 1:
            groups.append([])
        groups[-1].append((k, token))
    if any(len({t['color'] for _, t in group}) != 1 for group in groups):
        raise ValueError("A donor bundle must be monochrome")
    return groups


def _parse(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if int(counts.max()) * 2 <= a.size:
        raise ValueError("The background must occupy more than half the input")
    components = _components(a, bg)
    candidates = []
    for color in values:
        if int(color) == bg:
            continue
        parts = [p for p in components if p['color'] == int(color)]
        bars = [p for p in parts if _is_bar(p)]
        dots = [p for p in parts if len(p['cells']) == 1]
        if len(bars) == 2 and dots and len(bars) + len(dots) == len(parts):
            candidates.append((int(color), bars, dots))
    if len(candidates) != 1:
        raise ValueError("A unique marker color must have two bars and isolated gates")
    marker, bars, dots = candidates[0]
    if bars[0]['mask'].shape != bars[1]['mask'].shape:
        raise ValueError("Both endpoint bars must use the same orientation")
    segment, slope = _segment(bars[0]['center'], bars[1]['center'])
    segment_set = set(segment)
    tokens = [p for p in components if p['color'] != marker]
    if not tokens:
        raise ValueError("Missing token lanes")
    mask = tokens[0]['mask']
    if mask.shape not in ((1, 1), (1, 3), (3, 1), (3, 3)):
        raise ValueError("Tokens have odd bounding dimensions of one or three")
    if not all(np.array_equal(t['mask'], mask) for t in tokens):
        raise ValueError("Every token in an input must have one common silhouette")
    pitch = 2 if mask.shape == (1, 1) else 3
    token_pixels = set().union(*(t['cells'] for t in tokens))
    if token_pixels & segment_set:
        raise ValueError("The bridge cannot pass through a token")
    assigned = {t['center']: 0 for t in tokens}
    channels = []
    for dot in dots:
        gate = dot['center']
        active = gate in segment_set
        if active and min(abs(gate[0] - segment[0][0]),
                          abs(gate[0] - segment[-1][0])) < 2:
            raise ValueError("Gates must be separated from endpoint bars")
        possibilities = []
        for s in ((slope,) if active else (-1, 1)):
            upper, lower = {}, {}
            for token in tokens:
                dr = token['center'][0] - gate[0]
                dc = token['center'][1] - gate[1]
                if dr and dc == -s * dr and dr % pitch == 0:
                    (upper if dr < 0 else lower)[abs(dr) // pitch] = token
            if (upper and lower and set(upper) == set(range(1, len(upper) + 1))
                    and len({t['color'] for t in upper.values()}) == 1):
                groups = _bundles(lower)
                possibilities.append((s, upper, lower, groups))
        if len(possibilities) != 1:
            raise ValueError("Each gate must have one readable perpendicular token lane")
        s, upper, lower, groups = possibilities[0]
        for token in list(upper.values()) + list(lower.values()):
            assigned[token['center']] += 1
        channels.append(dict(gate=gate, active=active, slope=s, upper=upper,
                             lower=lower, groups=groups, d=len(upper), total=len(lower)))
    if not all(n == 1 for n in assigned.values()):
        raise ValueError("Each token must belong to exactly one gate lane")
    if not any(c['active'] for c in channels):
        raise ValueError("At least one gate must lie on the finite bridge")
    additions = set()
    for channel in channels:
        if not channel['active']:
            continue
        r, c = channel['gate']
        for k in range(channel['d'] + 1, channel['d'] + channel['total'] + 1):
            center = (r - pitch * k, c + channel['slope'] * pitch * k)
            cells = _token_cells(center, mask)
            if any(not (0 <= y < a.shape[0] and 0 <= x < a.shape[1]) for y, x in cells):
                raise ValueError("The complete sum must fit without clipping")
            if cells & (segment_set | additions) or any(a[q] != bg for q in cells):
                raise ValueError("A new token cannot overwrite existing material")
            additions.update(cells)
    return dict(grid=a, bg=bg, marker=marker, bars=bars, segment=segment,
                slope=slope, channels=channels, mask=mask, pitch=pitch,
                tokens=tokens, additions=additions)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles from bar/gate geometry, then count and grow active lanes."""
    p = _parse(input_grid)
    output = p['grid'].copy()
    for q in p['segment']:
        output[q] = p['marker']
    for channel in p['channels']:
        if not channel['active']:
            continue
        for token in channel['lower'].values():
            for q in token['cells']:
                output[q] = p['marker']
        color = channel['upper'][1]['color']
        r, c = channel['gate']
        for k in range(channel['d'] + 1, channel['d'] + channel['total'] + 1):
            center = (r - k * p['pitch'], c + channel['slope'] * k * p['pitch'])
            for q in _token_cells(center, p['mask']):
                output[q] = color
    return output


def _indices(rng, total, extent=None, fragmented=False):
    if extent is None:
        extent = total + int(fragmented)
    if extent < total:
        raise ValueError("Donor extent cannot be shorter than its count")
    if total == 1:
        return [extent]
    middle = sorted(int(x) for x in rng.choice(np.arange(1, extent), total - 1, replace=False))
    return middle + [extent]


def _random_mask(rng):
    mask = np.array(_MASKS[int(rng.integers(len(_MASKS)))], dtype=bool)
    return np.rot90(mask, int(rng.integers(4))).copy()


def _make_pair(rng, marker, bg, mask, slope, specs, inert, endpoint_axis, compact=False):
    """Construct output from explicit counts; parsing is only an acceptance check.

    Canonical bridge centers lie on r=c; then mirror, crop and pad the scene.
    No parent generator or stored puzzle is involved.
    """
    pitch = 2 if mask.shape == (1, 1) else 3
    gap_low = 2 if pitch == 2 else 4
    offsets = [0]
    for _ in specs[1:]:
        offsets.append(offsets[-1] + (gap_low if compact else int(rng.integers(gap_low, gap_low + 2))))
    ends = (-2, offsets[-1] + 2) if compact else (
        -int(rng.integers(2, 6)), offsets[-1] + int(rng.integers(2, 6)))
    lanes = [(t, True, d, indices) for t, (d, indices) in zip(offsets, specs)]
    if inert:
        side = int(rng.integers(2))
        distance = 2 if compact else int(rng.integers(2, 5))
        t = ends[side] + (-1 if side == 0 else 1) * distance
        lanes.append((t, False, 1 if compact else int(rng.integers(1, 3)), [1]))
    pool = [c for c in range(10) if c not in (marker, bg)]
    # A small local palette permits repeated donor/destination colors.
    pool = [int(c) for c in rng.choice(pool, int(rng.integers(1, 5)), replace=False)]
    before, after = {}, {}

    def paint(target, cells, color):
        if any(q in target for q in cells):
            raise ValueError("Object overlap")
        for q in cells:
            target[q] = color

    bar_mask = np.ones((1, 3) if endpoint_axis == 0 else (3, 1), dtype=bool)
    for t in ends:
        cells = _token_cells((t, t), bar_mask)
        paint(before, cells, marker)
        paint(after, cells, marker)
    for t, active, d, indices in lanes:
        paint(before, {(t, t)}, marker)
        after[(t, t)] = marker
        destination = int(rng.choice(pool))
        # Bundle colors can repeat, but adjacent donor positions share a color.
        previous, donor = None, None
        for k in indices:
            if previous is None or k != previous + 1:
                donor = int(rng.choice(pool))
            cells = _token_cells((t + k * pitch, t - k * pitch), mask)
            paint(before, cells, donor)
            paint(after, cells, marker if active else donor)
            previous = k
        for k in range(1, d + 1):
            cells = _token_cells((t - k * pitch, t + k * pitch), mask)
            paint(before, cells, destination)
            paint(after, cells, destination)
        if active:
            for k in range(d + 1, d + len(indices) + 1):
                paint(after, _token_cells((t - k * pitch, t + k * pitch), mask), destination)
    for t in range(ends[0], ends[1] + 1):
        q = (t, t)
        if q in before and before[q] != marker:
            raise ValueError("Bridge intersects a token")
        after[q] = marker
    all_points = set(before) | set(after)
    r0, c0 = map(min, zip(*all_points))
    r1, c1 = map(max, zip(*all_points))
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if max(h, w) > 30:
        raise ValueError("Scene exceeds the canvas")
    extra_h = int(rng.integers(0, min(5, 30 - h) + 1))
    extra_w = int(rng.integers(0, min(5, 30 - w) + 1))
    top, left = int(rng.integers(extra_h + 1)), int(rng.integers(extra_w + 1))
    a = np.full((h + extra_h, w + extra_w), bg, dtype=np.int8)
    b = a.copy()
    for mapping, grid in ((before, a), (after, b)):
        for (r, c), color in mapping.items():
            grid[r - r0 + top, c - c0 + left] = color
    if slope == -1:
        a, b = a[:, ::-1].copy(), b[:, ::-1].copy()
    parsed = _parse(a)
    if not np.array_equal(solve(a), b):
        raise ValueError("Construction does not match visible semantics")
    return {'input': a, 'output': b}, parsed


def _count(value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError("Illegal example count")
    return int(value)


def _key(grid):
    a = _grid(grid)
    return a.shape, a.tobytes()


def _emit(rng, seen, marker, bg, mask, slope, specs, inert, endpoint_axis):
    # Layout retries are bounded; no per-attempt files, logs or retained arrays.
    for _ in range(80):
        try:
            pair, parsed = _make_pair(rng, marker, bg, mask, slope, specs, inert, endpoint_axis)
        except ValueError:
            continue
        key = _key(pair['input'])
        if key not in seen:
            seen.add(key)
            return pair, parsed
    # Minimum bridge margins and lane separation fit every generated descriptor.
    # There are 64 visible upper/lower recolorings, versus at most 13 used inputs.
    pair, p = _make_pair(rng, marker, bg, mask, slope, specs, inert, endpoint_axis, compact=True)
    pool = [c for c in range(10) if c not in (marker, bg)]
    for upper_color in pool:
        for lower_color in pool:
            a, b = pair['input'].copy(), pair['output'].copy()
            for channel in p['channels']:
                for token in channel['upper'].values():
                    for q in token['cells']:
                        a[q] = b[q] = upper_color
                for token in channel['lower'].values():
                    for q in token['cells']:
                        a[q] = lower_color
                        b[q] = marker if channel['active'] else lower_color
                if channel['active']:
                    r, c = channel['gate']
                    for k in range(channel['d'] + 1, channel['d'] + channel['total'] + 1):
                        center = (r - k * p['pitch'], c + channel['slope'] * k * p['pitch'])
                        for q in _token_cells(center, p['mask']):
                            b[q] = upper_color
            key = _key(a)
            if key not in seen:
                seen.add(key)
                return {'input': a, 'output': b}, _parse(a)
    raise AssertionError("Finite recoloring capacity exceeded")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    marker, bg = (int(c) for c in rng.choice(10, 2, replace=False))
    mask = _random_mask(rng)
    point = mask.shape == (1, 1)
    endpoint_axis = int(rng.integers(2))
    mode = str(rng.choice(('more_gates', 'longer_gaps', 'larger_sum')))
    first_slope = int(rng.choice((-1, 1)))
    limit = 4 if mode == 'larger_sum' else (6 if point else 5)
    options_a = [(d, s) for d in (1, 2) for s in range(3, 5 if point else 4)
                 if d + s <= limit]
    d1, s1 = options_a[int(rng.integers(len(options_a)))]
    options_b = [(d, s) for d in (1, 2, 3) for s in (1, 2, 3)
                 if d + s <= limit and d != d1 and s != s1 and s * d1 != s1 * d]
    d2, s2 = options_b[int(rng.integers(len(options_b)))]
    # Unequal donor bundles, with a genuine missing pitch position.
    if s1 == 3:
        source_a = [1, 2, 4] if rng.integers(2) else [1, 3, 4]
    else:
        source_a = [1, 2, 3, 5] if rng.integers(2) else [1, 3, 4, 5]
    extra_d = int(rng.integers(1, 3))
    descriptors = [
        (first_slope, [(d1, source_a)], True),
        (-first_slope, [(d2, _indices(rng, s2)), (extra_d, [1])], False),
    ]
    for _ in range(2, num_train):
        # Compatible local geometry varies; the two core witnesses carry evidence.
        total = int(rng.integers(1, 4))
        d = int(rng.integers(1, min(3, limit - total) + 1))
        specs = [(d, _indices(rng, total, fragmented=bool(total > 1 and rng.integers(2))))]
        if rng.random() < .45:
            specs.append((int(rng.integers(1, 3)), [1]))
        descriptors.append((int(rng.choice((-1, 1))), specs, bool(rng.random() < .4)))
    puzzle, seen, training_parses = {'train': [], 'test': []}, set(), []
    for slope, specs, inert in descriptors:
        pair, parsed = _emit(rng, seen, marker, bg, mask, slope, specs, inert, endpoint_axis)
        puzzle['train'].append(pair)
        training_parses.append(parsed)
    max_extent = max(max(c['lower']) for p in training_parses for c in p['channels'] if c['active'])
    max_final = max(c['d'] + c['total'] for p in training_parses for c in p['channels'] if c['active'])
    for _ in range(num_test):
        if mode == 'more_gates':
            count = int(rng.integers(3, 5))
            specs = [(1, [1]) for _ in range(count)]
            # One unequal channel prevents a uniform destination-length template.
            index = int(rng.integers(count))
            specs[index] = (1, [1, 2])
            inert = False
        elif mode == 'longer_gaps':
            extent = max_extent + int(rng.integers(1, 3) if point else 1)
            total = int(rng.integers(2, 4))
            specs = [(1, _indices(rng, total, extent))]
            inert = bool(rng.random() < .5)
        else:
            final = max_final + int(rng.integers(1, 3))
            total = int(rng.integers(1, 4))
            specs = [(final - total, _indices(rng, total))]
            inert = bool(rng.random() < .5)
        pair, _ = _emit(rng, seen, marker, bg, mask, int(rng.choice((-1, 1))),
                        specs, inert, endpoint_axis)
        puzzle['test'].append(pair)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(p):
    active = [c for c in p['channels'] if c['active']]
    return dict(shape=p['grid'].shape, marker=p['marker'], bg=p['bg'],
                mask=(p['mask'].shape, p['mask'].tobytes()), pitch=p['pitch'],
                slope=p['slope'], bridge_length=len(p['segment']),
                active=len(active), inactive=len(p['channels']) - len(active),
                totals=tuple(c['total'] for c in active),
                destinations=tuple(c['d'] for c in active),
                finals=tuple(c['d'] + c['total'] for c in active),
                extents=tuple(max(c['lower']) for c in active),
                bundles=tuple(tuple(len(g) for g in c['groups']) for c in active))


def validate_generated_puzzle(puzzle):
    if not isinstance(puzzle, dict) or set(puzzle) != {'train', 'test'}:
        raise ValueError("Puzzle keys must be exactly train and test")
    if not all(isinstance(puzzle[s], list) for s in puzzle):
        raise ValueError("Splits must be lists")
    nt, nq = _count(len(puzzle['train']), 2, 10), _count(len(puzzle['test']), 1, 4)
    if nt <= nq:
        raise ValueError("More training examples are required")
    seen, analyses = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            if not isinstance(pair, dict) or set(pair) != {'input', 'output'}:
                raise ValueError("Pair keys must be exactly input and output")
            a, b = _grid(pair['input']), _grid(pair['output'])
            p = _parse(a)
            if a.shape != b.shape or not np.array_equal(solve(a), b):
                raise ValueError("Output violates the harvest rule")
            key = _key(a)
            if key in seen:
                raise ValueError("Duplicate input")
            seen.add(key)
            # Check the full edit footprint as well as equality to solve.
            expected_changed = {q for q in p['segment'] if a[q] != p['marker']}
            for c in p['channels']:
                if c['active']:
                    expected_changed.update(set().union(*(t['cells'] for t in c['lower'].values())))
            expected_changed.update(p['additions'])
            actual_changed = {tuple(map(int, q)) for q in np.argwhere(a != b)}
            if actual_changed != expected_changed:
                raise ValueError("Incorrect edits or changes outside the selected channels")
            area = int(p['mask'].sum())
            expected_new = area * sum(c['total'] for c in p['channels'] if c['active'])
            actual_new = sum(a[q] == p['bg'] and b[q] not in (p['bg'], p['marker'])
                             for q in np.ndindex(a.shape))
            if actual_new != expected_new:
                raise ValueError("Destination extension must copy tokens, not count pixels")
            analyses[split].append(p)
    if len({p['marker'] for s in analyses.values() for p in s}) != 1:
        raise ValueError("Marker convention must be coherent within the puzzle")
    train = analyses['train']
    channels = [c for p in train for c in p['channels'] if c['active']]
    if {p['slope'] for p in train} != {-1, 1}:
        raise ValueError("Training must demonstrate both growth slopes")
    if not any(not c['active'] for p in train for c in p['channels']):
        raise ValueError("Training must demonstrate finite-span gating")
    if not any(sum(c['active'] for c in p['channels']) >= 2 for p in train):
        raise ValueError("Training must demonstrate simultaneous gates")
    if not any(len(c['groups']) >= 2 and len({len(g) for g in c['groups']}) >= 2 for c in channels):
        raise ValueError("Training must demonstrate addition of unequal donor bundles")
    operands = [(c['d'], c['total']) for c in channels]
    if len({d for d, s in operands}) < 2 or len({s for d, s in operands}) < 2:
        raise ValueError("Operands must vary independently")
    if not any(s * operands[0][0] != operands[0][1] * d for d, s in operands):
        raise ValueError("Training cannot identify addition from a constant multiplier")
    rivals = [lambda c: 2 * c['d'], lambda c: c['total'],
              lambda c: c['d'] * c['total'], lambda c: max(c['d'], c['total']),
              lambda c: c['d'] + len(c['groups']),
              lambda c: c['d'] + max(len(g) for g in c['groups']),
              lambda c: c['d'] + max(c['lower'])]
    if any(all(r(c) == c['d'] + c['total'] for c in channels) for r in rivals):
        raise ValueError("Training leaves a listed arithmetic shortcut unresolved")
    return True


def self_test():
    """Stream 42 seeds, retain aggregate signatures, and exercise corruptions."""
    fingerprints, shapes, masks, palettes, slopes, active_counts, bridge_lengths = (set() for _ in range(7))
    totals, destinations, extents, bundle_patterns, shifts = (set() for _ in range(5))
    records = 0
    sample = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        features = {'train': [], 'test': []}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                p = _parse(pair['input'])
                f = _features(p)
                features[split].append(f)
                digest.update(repr(pair['input'].shape).encode())
                digest.update(pair['input'].tobytes())
                digest.update(pair['output'].tobytes())
                shapes.add(f['shape'])
                masks.add(f['mask'])
                palettes.add((f['marker'], f['bg']))
                slopes.add(f['slope'])
                active_counts.add(f['active'])
                bridge_lengths.add(f['bridge_length'])
                totals.update(f['totals'])
                destinations.update(f['destinations'])
                extents.update(f['extents'])
                bundle_patterns.update(f['bundles'])
                records += 1
        fingerprints.add(digest.hexdigest())
        tr, te = features['train'], features['test']
        if min(f['active'] for f in te) > max(f['active'] for f in tr):
            shifts.add('more_gates')
        if min(min(f['extents']) for f in te) > max(max(f['extents']) for f in tr):
            shifts.add('longer_gaps')
        if min(min(f['finals']) for f in te) > max(max(f['finals']) for f in tr):
            shifts.add('larger_sum')
        if seed == 0:
            sample = puzzle
    assert len(fingerprints) == 42
    assert len(shapes) >= 35 and len(masks) >= 6 and len(palettes) >= 25
    assert slopes == {-1, 1} and {1, 2, 3, 4} <= active_counts
    assert len(bridge_lengths) >= 10 and len(bundle_patterns) >= 5
    assert max(totals) >= 4 and max(destinations) >= 4 and max(extents) >= 6
    assert shifts == {'more_gates', 'longer_gaps', 'larger_sum'}
    again = generate_puzzle(0)
    assert all(np.array_equal(a[k], b[k]) for s in sample for a, b in zip(sample[s], again[s])
               for k in ('input', 'output'))
    # Causal A check: deleting one donor changes the arithmetic by one token.
    for pair in sample['train']:
        p = _parse(pair['input'])
        choices = [c for c in p['channels'] if c['active'] and c['total'] > 1]
        if not choices:
            continue
        channel = choices[0]
        a = pair['input'].copy()
        for q in channel['lower'][max(channel['lower'])]['cells']:
            a[q] = p['bg']
        p2 = _parse(a)
        assert len(p['additions']) - len(p2['additions']) == int(p['mask'].sum())
        r, c = channel['gate']
        k = channel['d'] + channel['total']
        top_cells = _token_cells((r - k * p['pitch'], c + channel['slope'] * k * p['pitch']), p['mask'])
        assert all(solve(a)[q] == p['bg'] for q in top_cells)
        break
    else:
        raise AssertionError("Missing donor-count counterfactual")
    # Causal B check: move one endpoint past the first of two gates. Its entire
    # payload is now unchanged, although all of its donor tokens remain present.
    for pair in sample['train']:
        p = _parse(pair['input'])
        channels = sorted((c for c in p['channels'] if c['active']), key=lambda c: c['gate'])
        if len(channels) < 2 or channels[1]['gate'][0] - channels[0]['gate'][0] < 4:
            continue
        a = pair['input'].copy()
        old = min(p['bars'], key=lambda b: b['center'])
        for q in old['cells']:
            a[q] = p['bg']
        r, c = channels[0]['gate']
        for q in _token_cells((r + 2, c + 2 * p['slope']), old['mask']):
            a[q] = p['marker']
        p2 = _parse(a)
        assert sum(c['active'] for c in p2['channels']) == len(channels) - 1
        b = solve(a)
        for token in list(channels[0]['upper'].values()) + list(channels[0]['lower'].values()):
            assert all(b[q] == a[q] for q in token['cells'])
        break
    else:
        raise AssertionError("Missing finite-bridge counterfactual")
    # Check the tight constructive placement independently of normal retries.
    for index, template in enumerate(_MASKS):
        m = np.array(template, dtype=bool)
        descriptors = ([(2, [1, 2, 4])], [(1, [1]), (1, [1]), (1, [1]), (1, [1, 2])],
                       [(3, [1, 2, 3])], [(1, [1, 3, 5])])
        for specs in descriptors:
            pair, _ = _make_pair(np.random.default_rng(500 + index), 8, 0, m, -1,
                                 specs, False, index % 2, compact=True)
            assert np.array_equal(solve(pair['input']), pair['output'])
    # Deliberately break addition, highlighting, bridge completion and inertness.
    for defect in ('growth', 'highlight', 'bridge', 'inactive'):
        broken = {s: [{k: g.copy() for k, g in pair.items()} for pair in sample[s]] for s in sample}
        changed = False
        for pair in broken['train']:
            p = _parse(pair['input'])
            if defect == 'growth':
                q = next(iter(p['additions']))
            elif defect == 'highlight':
                c = next(c for c in p['channels'] if c['active'])
                q = next(iter(next(iter(c['lower'].values()))['cells']))
            elif defect == 'bridge':
                cells = [q for q in p['segment'] if pair['input'][q] == p['bg']]
                if not cells:
                    continue
                q = cells[0]
            else:
                inactive = [c for c in p['channels'] if not c['active']]
                if not inactive:
                    continue
                q = next(iter(next(iter(inactive[0]['lower'].values()))['cells']))
            pair['output'][q] = p['bg'] if defect != 'inactive' else p['marker']
            changed = True
            break
        assert changed
        try:
            validate_generated_puzzle(broken)
        except ValueError:
            pass
        else:
            raise AssertionError("Failed to reject " + defect)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("Illegal count accepted")
    return dict(seeds=42, pairs=records, distinct_puzzles=len(fingerprints),
                shapes=len(shapes), masks=len(masks), palettes=len(palettes),
                active_counts=sorted(active_counts), bridge_lengths=len(bridge_lengths),
                source_totals=sorted(totals), destinations=sorted(destinations),
                donor_extents=sorted(extents), bundle_patterns=len(bundle_patterns),
                shifts=sorted(shifts))
