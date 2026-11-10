"""Unary stack addition, with visible controls and recoverable token geometry."""
import numpy as np


def _draw_token(grid, y, x, color, mask):
    h, w = mask.shape
    assert 0 <= y and y + h <= grid.shape[0]
    grid[y:y + h, x:x + w][mask] = color


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(),
                    np.flatnonzero(edges == -1).tolist()))


def _connected(mask):
    remaining = set(zip(*np.nonzero(mask)))
    if not remaining:
        return False
    queue = [remaining.pop()]
    while queue:
        y, x = queue.pop()
        for point in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if point in remaining:
                remaining.remove(point)
                queue.append(point)
    return not remaining


def _parse_input(grid):
    """Segment spatial lanes and separated components, never group by color."""
    bg = int(grid[0, 0])
    assert np.all(grid[0] == bg)
    lanes = _runs(np.any(grid != bg, axis=0))
    assert len(lanes) >= 2
    stacks, controls, pitches, baselines, control_rows = [], {}, set(), set(), set()
    for x, end in lanes:
        strip = grid[:, x:end]
        groups = _runs(np.any(strip != bg, axis=1))
        assert groups
        if groups[-1][1] - groups[-1][0] == 1:
            y, _ = groups.pop()
            assert np.all(strip[y] == strip[y, 0])
            controls[x] = int(strip[y, 0])
            control_rows.add(y)
        assert groups
        masks, colors = [], set()
        for y, stop in groups:
            block = strip[y:stop]
            mask = block != bg
            assert 2 <= mask.shape[0] <= 5 and 2 <= mask.shape[1] <= 5
            assert _connected(mask)
            assert np.all(np.any(mask, axis=0)) and np.all(np.any(mask, axis=1))
            colors.update(int(v) for v in np.unique(block[mask]))
            masks.append(mask)
        assert len(colors) == 1
        assert all(np.array_equal(masks[0], mask) for mask in masks)
        starts = [a for a, _ in groups]
        pitches.update(b - a for a, b in zip(starts, starts[1:]))
        baselines.add(groups[-1][1])
        stacks.append(dict(x=x, color=colors.pop(), height=len(groups),
                           ys=starts, mask=masks[0], bottom_y=starts[-1]))
    assert len(control_rows) == 1 and len(baselines) == 1
    assert len(set(controls.values())) == 2
    assert len(pitches) <= 1
    baseline, control_y = baselines.pop(), control_rows.pop()
    assert control_y > baseline  # at least one blank row beneath tokens
    assert np.all(grid[baseline:control_y] == bg)
    assert np.all(grid[control_y + 1:] == bg)
    mask = stacks[0]['mask']
    assert all(np.array_equal(s['mask'], mask) for s in stacks)
    pitch = next(iter(pitches), None)
    assert pitch is None or pitch > mask.shape[0]
    return dict(background=bg, stacks=stacks, controls=controls,
                control_y=control_y, baseline=baseline, mask=mask, pitch=pitch)


def _roles(parsed, source, destination):
    stacks = parsed['stacks']
    src = [s for s in stacks if parsed['controls'].get(s['x']) == source]
    dst = [s for s in stacks if parsed['controls'].get(s['x']) == destination]
    assert src and len(dst) == 1
    assert len(src) + 1 == len(parsed['controls'])
    return src, dst[0]


def _expected_output(grid, parsed, conventions, pitch):
    source, destination, highlight = conventions
    src, dst = _roles(parsed, source, destination)
    assert pitch is not None and pitch > parsed['mask'].shape[0]
    assert all(s['color'] != highlight for s in src)
    total = sum(s['height'] for s in src)
    assert dst['bottom_y'] - (dst['height'] + total - 1) * pitch >= 1
    result = grid.copy()
    result[parsed['control_y']] = parsed['background']
    for s in src:
        for y in s['ys']:
            _draw_token(result, y, s['x'], highlight, s['mask'])
    for k in range(dst['height'], dst['height'] + total):
        _draw_token(result, dst['bottom_y'] - k * pitch, dst['x'],
                    dst['color'], dst['mask'])
    return result


def _learn(train_pairs):
    parses = [_parse_input(p['input']) for p in train_pairs]
    visible = {p['pitch'] for p in parses if p['pitch'] is not None}
    assert visible, 'training must reveal spacing in its inputs'
    shared = next(iter(visible)) if len(visible) == 1 else None
    palette = set(parses[0]['controls'].values())
    assert all(set(p['controls'].values()) == palette for p in parses)
    candidates = []
    for source in sorted(palette):
        destination = next(c for c in palette if c != source)
        try:
            src, _ = _roles(parses[0], source, destination)
            s = src[0]
            h, w = s['mask'].shape
            values = train_pairs[0]['output'][s['bottom_y']:s['bottom_y'] + h,
                                               s['x']:s['x'] + w][s['mask']]
            assert len(np.unique(values)) == 1
            highlight = int(values[0])
            conventions = (source, destination, highlight)
            assert len(set(conventions)) == 3
            for pair, p in zip(train_pairs, parses):
                expected = _expected_output(pair['input'], p, conventions,
                                            p['pitch'] or shared)
                assert np.array_equal(expected, pair['output'])
            candidates.append(conventions)
        except (AssertionError, IndexError, ValueError):
            pass
    assert len(candidates) == 1, 'training must identify the control meanings'
    return candidates[0], shared, parses


def predict(train_pairs, test_input):
    """Prediction uses only demonstrations and the query grid, with no seed."""
    conventions, shared, _ = _learn(train_pairs)
    parsed = _parse_input(test_input)
    return _expected_output(test_input, parsed, conventions, parsed['pitch'] or shared)


def _grid_key(grid):
    return (grid.shape, tuple(int(v) for v in grid.flat))


def _source_keys():
    # Exact sparse contents of the staged source; dimensions are never forbidden.
    mask = np.array([[0, 1, 1, 1, 0], [1, 1, 1, 1, 1], [0, 1, 1, 1, 0]], bool)
    descriptions = [
        (7, [9, 0, 1, 4], [2, 1, 1, 3], {3: 2, 15: 3}),
        (1, [0, 8, 4, 6], [3, 2, 2, 1], {3: 3, 21: 2}),
        (6, [7, 2, 3, 0], [1, 3, 3, 1], {9: 3, 15: 2}),
        (4, [1, 9, 6, 8], [1, 1, 1, 1], {3: 2, 9: 2, 21: 3}),
    ]
    result = set()
    for bg, colors, counts, controls in descriptions:
        a = np.full((30, 30), bg, dtype=np.int8)
        for x, color, count in zip((3, 9, 15, 21), colors, counts):
            for k in range(count):
                _draw_token(a, 25 - k * 4, x, color, mask)
        for x, color in controls.items():
            a[29, x:x + 5] = color
        b = _expected_output(a, _parse_input(a), (2, 3, 5), 4)
        result.add((_grid_key(a), _grid_key(b)))
    return result


_SOURCE_KEYS = _source_keys()


def _arithmetic_evidence(operands):
    # Reject a destination multiplier, fixed increment, source multiplier,
    # multiplication, maximum, replacement, and counting source bars.
    totals = [sum(s) for s, d in operands]
    dests = [d for s, d in operands]
    assert len(set(totals)) >= 2 and len(set(dests)) >= 2
    assert any(totals[0] * d != s * dests[0] for s, d in zip(totals, dests))
    rivals = (
        lambda s, d: 2 * d, lambda s, d: 2 * sum(s),
        lambda s, d: sum(s) * d, lambda s, d: max(sum(s), d),
        lambda s, d: sum(s), lambda s, d: d + len(s),
    )
    for rival in rivals:
        assert any(rival(s, d) != sum(s) + d for s, d in operands)
    if any(len(s) > 1 for s, d in operands):
        assert any(len(s) > 1 and len(set(s)) > 1 for s, d in operands)
        assert any(d + max(s) != d + sum(s) for s, d in operands)


def _analyze_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in puzzle)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    for pair in puzzle['train'] + puzzle['test']:
        assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
        for grid in pair.values():
            assert isinstance(grid, np.ndarray) and grid.ndim == 2
            assert all(1 <= size <= 30 for size in grid.shape)
            assert np.issubdtype(grid.dtype, np.integer)
            assert np.all((0 <= grid) & (grid <= 9))
        assert pair['input'].shape == pair['output'].shape
        assert not np.array_equal(pair['input'], pair['output'])
        key = _grid_key(pair['input'])
        assert key not in seen
        assert (key, _grid_key(pair['output'])) not in _SOURCE_KEYS
        seen.add(key)
    conventions, shared, train = _learn(puzzle['train'])
    test = [_parse_input(p['input']) for p in puzzle['test']]
    source, destination, highlight = conventions
    operands, train_unmarked, train_overlap, train_repeat = [], False, False, False
    earlier_pitch = False
    for split, parses in (('train', train), ('test', test)):
        for pair, p in zip(puzzle[split], parses):
            assert p['background'] not in conventions
            assert highlight not in pair['input']
            if p['pitch'] is None:
                assert shared is not None
                if split == 'train':
                    assert earlier_pitch, 'singleton demonstrations must follow visible pitch'
                assert all(q['pitch'] in (None, shared) for q in train + test)
            else:
                earlier_pitch = True
            expected = _expected_output(pair['input'], p, conventions, p['pitch'] or shared)
            assert np.array_equal(expected, pair['output'])
            src, dst = _roles(p, source, destination)
            unmarked = [s for s in p['stacks'] if s['x'] not in p['controls']]
            overlap = any(s['color'] in (source, destination) for s in p['stacks'])
            repeat = any(a['color'] == b['color'] for a in src for b in unmarked)
            if split == 'train':
                operands.append(([s['height'] for s in src], dst['height']))
                train_unmarked |= bool(unmarked)
                train_overlap |= overlap
                train_repeat |= repeat
            else:
                assert not unmarked or train_unmarked
                assert not overlap or train_overlap
                assert not repeat or train_repeat
                # The source-supported extrapolation is one source -> two;
                # higher source counts require an aggregation demonstration.
                assert len(src) <= 2 or any(len(s) > 1 for s, d in operands)
    _arithmetic_evidence(operands)
    return dict(conventions=conventions, shared_pitch=shared, train=train, test=test)


def validate_generated_puzzle(puzzle: dict) -> None:
    _analyze_puzzle(puzzle)


def _resolve_counts(rng, num_train, num_test):
    for value, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError('invalid pair count')
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1)) if num_train is not None else int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _sample_mask(rng, max_height=5, familiar=False):
    family = 0 if familiar else int(rng.integers(0, 5))
    if family == 0 and max_height >= 3:
        w = int(rng.choice([3, 5]))
        mask = np.ones((3, w), bool)
        mask[0, (0, -1)] = False
        mask[-1, (0, -1)] = False
        return mask
    h = int(rng.integers(2, max_height + 1))
    w = int(rng.integers(2, 6))
    if family == 1:
        return np.ones((h, w), bool)
    if family == 2:
        radius = int(rng.integers(1, min((max_height - 1) // 2, 2) + 1)) if max_height >= 3 else 0
        if radius:
            yy, xx = np.indices((2 * radius + 1, 2 * radius + 1))
            return abs(yy - radius) + abs(xx - radius) <= radius
    if family == 3:
        mask = np.zeros((h, w), bool)
        mask[:, int(rng.integers(w))] = True
        mask[int(rng.integers(h)), :] = True
        return mask
    # A connected, full-span asymmetric silhouette: an L spine plus
    # randomized adjoining row segments. This is compositional, not a mask bank.
    mask = np.zeros((h, w), bool)
    mask[:, 0] = True
    mask[-1, :] = True
    for y in range(h - 1):
        mask[y, :int(rng.integers(1, w + 1))] = True
    if rng.integers(2):
        mask = mask[:, ::-1]
    if rng.integers(2):
        mask = mask[::-1]
    return mask.copy()


def _geometry(rng, params):
    if params['geometry'] == 'shared':
        mask, pitch = params['mask'], params['pitch']
    elif params['geometry'] == 'shared_pitch':
        pitch = params['pitch']
        mask = _sample_mask(rng, min(5, pitch - 1))
    else:
        mask = _sample_mask(rng)
        pitch = mask.shape[0] + int(rng.integers(1, 4))
        # Two informative demonstrations need room for at least four tokens.
        pitch = min(pitch, (27 - mask.shape[0]) // 3)
    capacity = 1 + (27 - mask.shape[0]) // pitch
    if params['simple']:
        capacity = min(capacity, 6)
    return dict(mask=mask, pitch=pitch, capacity=capacity,
                max_stacks=(31 // (mask.shape[1] + 1)))


def _options(geometry, mode, regime):
    maximum = min(4, geometry['max_stacks'] - 1, geometry['capacity'] - 1)
    if mode == 'original':
        sources = [1] if regime == 'source_train' else [2]
    elif mode == 'single':
        sources = [1]
    else:
        sources = range(1, maximum + 1)
    return [(n, total, dest) for n in sources
            for total in range(n, geometry['capacity'])
            for dest in range(1, geometry['capacity'] - total + 1)]


def _demonstration_choices(rng, g1, g2, mode, anchor):
    options = []
    for i, geometry in enumerate((g1, g2)):
        options.append([o for o in _options(geometry, mode, 'source_train')
                        if o[1] + o[2] > o[0] + 1 and
                        (i != anchor or o[0] + 2 <= geometry['max_stacks'])])
    a, b = options
    compatible = []
    for first in a:
        n1, s1, d1 = first
        for second in b:
            n2, s2, d2 = second
            if s1 == s2 or d1 == d2 or s1 * d2 == s2 * d1:
                continue
            if s1 * d1 == s1 + d1 and s2 * d2 == s2 + d2:
                continue
            if s1 == n1 and s2 == n2:
                continue
            if mode == 'mixed' and not ((n1 > 1 and s1 > n1) or (n2 > 1 and s2 > n2)):
                continue
            compatible.append((first, second))
    assert compatible
    return compatible[int(rng.integers(len(compatible)))]


def _partition(rng, total, count):
    if count == 1:
        return [total]
    cuts = sorted(rng.choice(np.arange(1, total), count - 1, replace=False).tolist())
    values = np.diff([0] + cuts + [total]).tolist()
    if total > count and len(set(values)) == 1:
        values[0] -= 1
        values[1] += 1
    rng.shuffle(values)
    return values


def _choose_layout(rng, count, width, full, compact=False):
    minimum = count * width + count - 1
    upper = min(30, minimum + 3) if compact else 30
    canvas = 30 if full else int(rng.integers(minimum, upper + 1))
    extra = rng.multinomial(canvas - minimum, np.full(count + 1, 1 / (count + 1)))
    xs, x = [], int(extra[0])
    for i in range(count):
        xs.append(x)
        x += width + (1 + int(extra[i + 1]) if i < count - 1 else 0)
    return canvas, xs


def _build_pair(rng, params, geometry, choice, anchor=False, singleton=False):
    nsrc, total, dest_height = choice
    compact = not params['simple'] and rng.random() < 0.25
    mask, pitch = geometry['mask'], geometry['pitch']
    h, w = mask.shape
    max_stacks = geometry['max_stacks']
    if not params['repeats']:
        max_stacks = min(max_stacks, 6)
    if params['simple']:
        count = 4
    else:
        lower = 1 if anchor else 0
        unmarked = int(rng.integers(lower, min(2 if compact else 4, max_stacks - nsrc - 1) + 1))
        count = nsrc + 1 + unmarked
    order = rng.permutation(count).tolist()
    dst, src = order[0], order[1:nsrc + 1]
    unmarked = order[nsrc + 1:]
    unmarked_limit = min(geometry['capacity'], max(2, dest_height + total)) if compact else geometry['capacity']
    counts = rng.integers(1, unmarked_limit + 1, size=count).tolist()
    counts[dst] = dest_height
    for i, value in zip(src, _partition(rng, total, nsrc)):
        counts[i] = value
    if singleton:
        assert total == nsrc and dest_height == 1
        counts = [1] * count
    assert singleton or max(counts) > 1
    final_count = dest_height + total
    tallest = max(final_count, max(counts))
    required = 1 + h + (tallest - 1) * pitch + 2
    assert required <= 30
    spare = 30 - required
    control_gap = int(rng.integers(1, min(2 if compact else 4, spare + 1) + 1))
    spare -= control_gap - 1
    bottom_margin = int(rng.integers(0, min(1 if compact else 4, spare) + 1))
    minimum_height = required + control_gap - 1 + bottom_margin
    full = params['simple'] or (not compact and bool(rng.random() < 0.18))
    upper_height = min(30, minimum_height + 2) if compact else 30
    height = 30 if full else int(rng.integers(minimum_height, upper_height + 1))
    width, xs = _choose_layout(rng, count, w, full, compact)
    if params['simple']:
        xs = params['xs']
        control_gap, bottom_margin = params['control_gap'], params['bottom_margin']
        # Familiar layouts use their own capacity, with these margins budgeted.
    control_y = height - 1 - bottom_margin
    bottom_y = control_y - control_gap - h
    assert bottom_y - (tallest - 1) * pitch >= 1
    source, destination, highlight = params['conventions']
    bg = int(rng.choice([c for c in range(10) if c not in params['conventions']]))
    overlap = params['overlap'] and (anchor or rng.random() < 0.5)
    excluded = {bg, highlight} if overlap else {bg, highlight, source, destination}
    pool = [c for c in range(10) if c not in excluded]
    if params['repeats']:
        local_pool = rng.choice(pool, int(rng.integers(1, min(len(pool), count) + 1)), replace=False)
        colors = rng.choice(local_pool, count, replace=True).tolist()
    else:
        colors = rng.choice(pool, count, replace=False).tolist()
    if anchor and params['overlap']:
        colors[src[0]], colors[dst] = destination, source
        if not params['repeats']:
            remaining = [c for c in pool if c not in (source, destination)]
            others = [i for i in range(count) if i not in (src[0], dst)]
            for i, c in zip(others, rng.choice(remaining, len(others), replace=False)):
                colors[i] = int(c)
    if anchor and params['repeats']:
        colors[unmarked[0]] = colors[src[0]]
    a = np.full((height, width), bg, dtype=np.int8)
    b = a.copy()
    for i, x in enumerate(xs):
        for k in range(counts[i]):
            _draw_token(a, bottom_y - k * pitch, x, colors[i], mask)
        output_count = final_count if i == dst else counts[i]
        for k in range(output_count):
            _draw_token(b, bottom_y - k * pitch, x, highlight if i in src else colors[i], mask)
        if i == dst or i in src:
            a[control_y, x:x + w] = destination if i == dst else source
    return {'input': a, 'output': b}


def _unique_pair(pair, seen, conventions):
    def available(p):
        return _grid_key(p['input']) not in seen and (_grid_key(p['input']), _grid_key(p['output'])) not in _SOURCE_KEYS
    if available(pair):
        return pair
    # A bounded deterministic fallback provides 7 backgrounds x 6 destination
    # colors = 42 distinct pairs, exceeding 14 puzzle pairs plus 4 source pairs.
    parsed = _parse_input(pair['input'])
    _, dst = _roles(parsed, *conventions[:2])
    bg = parsed['background']
    colors = [c for c in range(10) if c not in conventions]
    for new_bg in colors:
        lookup = np.arange(10)
        lookup[bg], lookup[new_bg] = new_bg, bg
        for new_dest in colors:
            if new_dest == new_bg:
                continue
            result = {k: lookup[v].astype(np.int8) for k, v in pair.items()}
            x, w = dst['x'], dst['mask'].shape[1]
            for grid in result.values():
                lane = grid[:parsed['baseline'], x:x + w]
                lane[lane != new_bg] = new_dest
            if available(result):
                return result
    raise AssertionError('finite uniqueness construction exhausted')


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    num_train, num_test = _resolve_counts(rng, num_train, num_test)
    simple = bool(rng.random() < 0.15)
    mode = str(rng.choice(['original', 'single', 'mixed'], p=[0.25, 0.15, 0.60]))
    if simple:
        mode = str(rng.choice(['original', 'single']))
    geometry_mode = 'shared' if simple else str(rng.choice(['shared', 'shared_pitch', 'local'], p=[0.4, 0.25, 0.35]))
    mask = _sample_mask(rng, familiar=simple)
    pitch = min(mask.shape[0] + int(rng.integers(1, 4)), (27 - mask.shape[0]) // 3)
    params = dict(conventions=tuple(int(v) for v in rng.choice(10, 3, replace=False)),
                  simple=simple, geometry=geometry_mode, mask=mask, pitch=pitch,
                  overlap=bool(rng.random() < 0.75), repeats=bool(rng.random() < 0.75),
                  xs=[3, 9, 15, 21], control_gap=1, bottom_margin=0)
    geometries = [_geometry(rng, params) for _ in range(num_train + num_test)]
    anchor = int(rng.integers(2))
    first, second = _demonstration_choices(rng, geometries[0], geometries[1], mode, anchor)
    choices = [first, second]
    singletons = [False, False]
    for index, geo in enumerate(geometries[2:], 2):
        regime = 'source_train' if index < num_train else 'target_test'
        options = _options(geo, mode, regime)
        singleton = geometry_mode != 'local' and rng.random() < (0.18 if regime == 'source_train' else 0.35)
        if singleton:
            options = [o for o in options if o[1] == o[0] and o[2] == 1]
        else:
            options = [o for o in options if o[1] + o[2] > o[0] + 1]
        choices.append(options[int(rng.integers(len(options)))])
        singletons.append(singleton)
    # Anchor placement and operand order vary. Singleton demonstrations remain
    # after the two informative, visibly repeated inputs.
    puzzle, seen = {'train': [], 'test': []}, set()
    for index, (geo, choice, singleton) in enumerate(zip(geometries, choices, singletons)):
        pair = _build_pair(rng, params, geo, choice, index == anchor, singleton)
        pair = _unique_pair(pair, seen, params['conventions'])
        seen.add(_grid_key(pair['input']))
        puzzle['train' if index < num_train else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def review_features(pair, parsed, conventions, shared_pitch=None):
    """Measurements reconstructed from returned pixels, for bounded self-tests."""
    source, destination, _ = conventions
    src, dst = _roles(parsed, source, destination)
    unmarked = [s for s in parsed['stacks'] if s['x'] not in parsed['controls']]
    heights = [s['height'] for s in src]
    pitch = parsed['pitch'] or shared_pitch
    h, w = parsed['mask'].shape
    all_colors = [s['color'] for s in parsed['stacks']]
    return dict(shape=pair['input'].shape, sources=len(src), unmarked=len(unmarked),
                source_heights=tuple(heights), source_total=sum(heights), destination=dst['height'],
                final=dst['height'] + sum(heights), singleton=all(s['height'] == 1 for s in parsed['stacks']),
                unequal_sources=len(set(heights)) > 1,
                unmarked_heights=tuple(s['height'] for s in unmarked),
                repeated=len(set(all_colors)) < len(all_colors),
                source_unmarked_match=any(a['color'] == b['color'] for a in src for b in unmarked),
                source_source_match=len({s['color'] for s in src}) < len(src),
                source_destination_match=any(s['color'] == dst['color'] for s in src),
                overlap=any(c in (source, destination) for c in all_colors),
                controlled_overlap=any(s['color'] in (source, destination) for s in src + [dst]),
                mask=(h, w, tuple(int(v) for v in parsed['mask'].flat)), pitch=pitch,
                control_gap=parsed['control_y'] - parsed['baseline'],
                bottom_margin=pair['input'].shape[0] - 1 - parsed['control_y'],
                headroom=dst['bottom_y'] - (dst['height'] + sum(heights) - 1) * pitch,
                positions=tuple(s['x'] for s in parsed['stacks']))


def _puzzles_equal(a, b):
    return all(len(a[k]) == len(b[k]) and all(np.array_equal(p[r], q[r])
               for p, q in zip(a[k], b[k]) for r in ('input', 'output')) for k in ('train', 'test'))


def self_test() -> None:
    assert _puzzles_equal(generate_puzzle(73), generate_puzzle(73))
    assert not _puzzles_equal(generate_puzzle(73), generate_puzzle(74))
    for i, (a, b) in enumerate([(2, 1), (10, 1), (10, 4), (7, 3),
                                (None, 1), (None, 4), (2, None), (10, None)]):
        p = generate_puzzle(900 + i, a, b)
        assert a is None or len(p['train']) == a
        assert b is None or len(p['test']) == b
        assert _puzzles_equal(p, generate_puzzle(900 + i, a, b))
    for a, b in [(1, None), (11, None), (None, 0), (None, 5), (2, 2),
                 (3, 4), (False, 1), (3.0, 1)]:
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    rows, train_counts, test_counts, conventions, fingerprints = [], set(), set(), set(), set()
    within = {k: 0 for k in ('shape', 'sources', 'unmarked', 'mask', 'pitch', 'positions', 'overlap')}
    original_shift = single_test = mixed_training = singleton_train = singleton_test = 0
    for seed in range(20000, 20042):
        puzzle = generate_puzzle(seed)
        a = _analyze_puzzle(puzzle)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        conventions.add(a['conventions'])
        fingerprints.add(tuple(_grid_key(p['input']) for p in puzzle['train'] + puzzle['test']))
        local = []
        for split in ('train', 'test'):
            for pair, parsed in zip(puzzle[split], a[split]):
                f = review_features(pair, parsed, a['conventions'], a['shared_pitch'])
                rows.append(f)
                local.append(f)
                if split == 'test':
                    assert np.array_equal(predict(puzzle['train'], pair['input'].copy()), pair['output'])
                    singleton_test += f['singleton']
                else:
                    singleton_train += f['singleton']
        nt = len(puzzle['train'])
        ns = [f['sources'] for f in local]
        original_shift += set(ns[:nt]) == {1} and set(ns[nt:]) == {2}
        single_test += 1 in ns[nt:]
        mixed_training += max(ns[:nt]) > 1
        for name in within:
            within[name] += len({f[name] for f in local}) > 1
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(fingerprints) == 42 and len(conventions) >= 25
    assert len({f['shape'] for f in rows}) >= 70
    assert len({f['mask'] for f in rows}) >= 25
    assert len({f['positions'] for f in rows}) >= 70
    assert {1, 2, 3, 4} <= {f['sources'] for f in rows}
    assert {0, 1, 2, 3, 4} <= {f['unmarked'] for f in rows}
    assert original_shift >= 3 and single_test >= 8 and mixed_training >= 10
    assert singleton_train >= 5 and singleton_test >= 8
    assert max(f['source_total'] for f in rows) >= 6
    assert max(f['destination'] for f in rows) >= 4
    assert max(f['final'] for f in rows) >= 8
    assert max(v for f in rows for v in f['unmarked_heights']) >= 5
    assert any(f['shape'] == (30, 30) for f in rows)
    assert any(max(f['shape']) <= 18 for f in rows)
    assert min(f['headroom'] for f in rows) == 1 and max(f['headroom'] for f in rows) >= 12
    assert {1, 2, 3, 4} <= {f['control_gap'] for f in rows}
    assert {0, 1, 2, 3, 4} <= {f['bottom_margin'] for f in rows}
    assert len({f['pitch'] for f in rows}) >= 5
    for name in ('repeated', 'source_unmarked_match', 'source_source_match',
                 'source_destination_match', 'overlap', 'controlled_overlap', 'unequal_sources'):
        assert sum(f[name] for f in rows) >= 15
        assert any(not f[name] for f in rows)
    assert within['shape'] >= 30 and within['unmarked'] >= 25
    assert within['sources'] >= 20 and within['mask'] >= 12
    assert within['pitch'] >= 6 and within['positions'] >= 30 and within['overlap'] >= 15
    assert sum(f['sources'] >= 2 and f['repeated'] and f['unmarked'] > 0 and
               f['mask'][0] != 3 and f['bottom_margin'] > 0 for f in rows) >= 8
    # Mutation checks exercise sum, highlighting, controls, and untouched stacks.
    original = generate_puzzle(31, 2, 1)
    analysis = _analyze_puzzle(original)
    for defect in ('sum', 'highlight', 'control', 'unmarked'):
        broken = {k: [{r: g.copy() for r, g in p.items()} for p in v] for k, v in original.items()}
        changed = False
        for pair, parsed in zip(broken['train'], analysis['train']):
            src, dst = _roles(parsed, *analysis['conventions'][:2])
            if defect == 'sum':
                pitch = parsed['pitch'] or analysis['shared_pitch']
                y = dst['ys'][0] - pitch
                _draw_token(pair['output'], y, dst['x'], parsed['background'], dst['mask'])
            elif defect == 'highlight':
                s = src[0]
                _draw_token(pair['output'], s['bottom_y'], s['x'], s['color'], s['mask'])
            elif defect == 'control':
                pair['output'][parsed['control_y']] = pair['input'][parsed['control_y']]
            else:
                unmarked = [s for s in parsed['stacks'] if s['x'] not in parsed['controls']]
                if not unmarked:
                    continue
                s = unmarked[0]
                _draw_token(pair['output'], s['bottom_y'], s['x'], analysis['conventions'][2], s['mask'])
            changed = True
            break
        assert changed
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError('incorrect ' + defect + ' accepted')
