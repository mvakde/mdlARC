"""Cadence collectors: a phase-gated transfer of unary token counts.

Only the three isolated control dots encode roles. No palette, pitch, rotation,
or source/destination association is learned from a seed or another example.
"""
from __future__ import annotations

import numpy as np


HOST_SEEDS = (0, 73772842, 684977141, 235903929, 288523259, 19993142,
              637844934, 640220443, 812332879, 46542417, 669160838, 229710880,
              333218496, 546544724, 40859373, 885910549, 829079545, 338283721,
              708502125, 743324173, 724240754, 654023277, 232484347, 210914147,
              894405241, 372998953, 666667497, 951167949, 85401841, 579270977,
              635862193, 228825067, 285557451, 727089911, 107625137, 4965914,
              304304286, 34582857, 499280193, 478361028, 546816932, 282025497)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid")
    return a.astype(np.int8, copy=True)


def _components(grid, bg):
    remaining = set(zip(*np.nonzero(grid != bg)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue, cells = [start], [start]
        while queue:
            y, x = queue.pop()
            for point in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if point in remaining:
                    remaining.remove(point)
                    queue.append(point)
                    cells.append(point)
        ys, xs = zip(*cells)
        top, left, bottom, right = map(int, (min(ys), min(xs), max(ys) + 1, max(xs) + 1))
        block = grid[top:bottom, left:right]
        mask = block != bg
        colors = np.unique(block[mask])
        if len(colors) != 1:
            raise ValueError("each connected object must be monochrome")
        result.append(dict(top=top, bottom=bottom, left=left, right=right,
                           mask=mask.copy(), color=int(colors[0]), cells=cells))
    return result


def _parse(input_grid):
    raw = _grid(input_grid)
    colors, counts = np.unique(raw, return_counts=True)
    if counts.max() <= raw.size // 2:
        raise ValueError("background must occupy more than half the input")
    bg = int(colors[np.argmax(counts)])
    candidates = []
    for turns in range(4):
        g = np.rot90(raw, turns).copy()
        components = _components(g, bg)
        dots = [c for c in components if len(c['cells']) == 1]
        tokens = [c for c in components if len(c['cells']) > 1]
        if (len(dots) != 3 or not tokens
                or len({c['top'] for c in dots}) != 1):
            continue
        marker_y = dots[0]['top']
        if max(c['bottom'] for c in tokens) >= marker_y:
            continue
        candidates.append((turns, g, dots, tokens, marker_y))
    if len(candidates) != 1:
        raise ValueError("controls must uniquely identify the growth direction")
    turns, g, dots, tokens, marker_y = candidates[0]
    if not (np.all(g[0] == bg) and np.all(g[-1] == bg)
            and np.all(g[:, 0] == bg) and np.all(g[:, -1] == bg)):
        raise ValueError("the outer border must be background")
    dot_colors = [d['color'] for d in dots]
    repeated = [c for c in set(dot_colors) if dot_colors.count(c) == 2]
    if len(repeated) != 1 or len(set(dot_colors)) != 2:
        raise ValueError("controls need two equal references and one terminal")
    ref = repeated[0]
    a, b = sorted(d['left'] for d in dots if d['color'] == ref)
    terminal = next(d for d in dots if d['color'] != ref)
    t, highlight = terminal['left'], terminal['color']
    period = b - a
    if period < 3 or a < t < b or (t - a) % period:
        raise ValueError("terminal must lie beyond the references on their phase")
    by_center = {}
    for c in tokens:
        h, w = c['mask'].shape
        if h not in (2, 3) or w not in (1, 3):
            raise ValueError("tokens need height 2..3 and width 1 or 3")
        x = (c['left'] + c['right'] - 1) // 2
        by_center.setdefault(x, []).append(c)
    stacks, pitches, baselines = [], set(), set()
    for x, objects in sorted(by_center.items()):
        objects.sort(key=lambda c: c['top'])
        first = objects[0]
        if any(c['color'] != first['color'] or c['left'] != first['left']
               or not np.array_equal(c['mask'], first['mask']) for c in objects):
            raise ValueError("each stack must repeat exactly one token")
        ys = [c['top'] for c in objects]
        pitches.update(v - u for u, v in zip(ys, ys[1:]))
        baselines.add(objects[-1]['bottom'])
        stacks.append(dict(x=x, left=first['left'], right=first['right'],
                           ys=ys, mask=first['mask'], color=first['color'],
                           count=len(objects)))
    if len(pitches) != 1 or len(baselines) != 1:
        raise ValueError("a common baseline and locally visible pitch are required")
    pitch = pitches.pop()
    if pitch <= max(s['mask'].shape[0] for s in stacks):
        raise ValueError("tokens require a positive background gap")
    if any(u['right'] >= v['left'] for u, v in zip(stacks, stacks[1:])):
        raise ValueError("stacks require a blank column between bounding boxes")
    indexed = {s['x']: s for s in stacks}
    if any(x not in indexed for x in (a, b, t)):
        raise ValueError("all control dots must align with existing stacks")
    low, high = min(a, t), max(b, t)
    src = [s for s in stacks if low <= s['x'] <= high
           and (s['x'] - a) % period == 0 and s['x'] != t]
    dst = indexed[t]
    if any(s['color'] == highlight for s in src):
        raise ValueError("contributors must visibly change color")
    total = sum(s['count'] for s in src)
    if dst['ys'][0] - total * pitch < 1:
        raise ValueError("the complete transfer must fit inside the border")
    return dict(grid=g, turns=turns, background=bg, dots=dots, marker_y=marker_y,
                a=a, b=b, t=t, period=period, highlight=highlight, reference=ref,
                stacks=stacks, sources=src, destination=dst, pitch=pitch,
                total=total, low=low, high=high)


def _paint(grid, y, left, mask, color):
    h, w = mask.shape
    if y < 0 or y + h > grid.shape[0] or left < 0 or left + w > grid.shape[1]:
        raise ValueError("token would be clipped")
    grid[y:y + h, left:left + w][mask] = color


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, count, spacing and silhouette from this input only."""
    p = _parse(input_grid)
    result = p['grid'].copy()
    for d in p['dots']:
        result[d['top'], d['left']] = p['background']
    for s in p['sources']:
        for y in s['ys']:
            _paint(result, y, s['left'], s['mask'], p['highlight'])
    dst = p['destination']
    for step in range(1, p['total'] + 1):
        _paint(result, dst['ys'][0] - step * p['pitch'], dst['left'],
               dst['mask'], dst['color'])
    return np.rot90(result, -p['turns']).copy()


def _pick(rng, values):
    return values[int(rng.integers(len(values)))]


def _mask(rng, max_height=3, narrow=False):
    h = int(rng.integers(2, max_height + 1))
    w = 1 if narrow else int(_pick(rng, (1, 3, 3)))
    mask = np.ones((h, w), dtype=bool)
    if w == 3:
        mode = int(rng.integers(4))
        if mode == 1:
            mask[:-1, 1:] = False  # connected L
        elif mode == 2:
            mask[0, 0] = mask[-1, -1] = False
        elif mode == 3:
            mask[:] = False
            mask[:, 1] = True
            mask[h // 2, :] = True
    if rng.integers(2):
        mask = mask[:, ::-1]
    if rng.integers(2):
        mask = mask[::-1]
    return mask.copy()


def _partition(rng, total, count, unequal=False):
    # A bounded constructive composition; unequal counts when requested.
    result = [1] * count
    for _ in range(total - count):
        result[int(rng.integers(count))] += 1
    if unequal and len(set(result)) == 1:
        result[0] -= 1
        result[1] += 1
    rng.shuffle(result)
    return result


def _operand_choices(rng):
    first = [(s, d) for s in range(3, 6) for d in range(1, 4)
             if s + d <= 6 and (s > 3 or d > 1)]
    second = [(s, d) for s in range(3, 6) for d in range(1, 4) if s + d <= 6]
    compatible = [(u, v) for u in first for v in second
                  if u[0] != v[0] and u[1] != v[1]
                  and sum(u) != sum(v) and u[0] * v[1] != v[0] * u[1]]
    return _pick(rng, compatible)


def _build_pair(rng, kind, operands, roles, turns, reverse):
    """Labels are painted from constructed objects, independently of solve."""
    if kind == 'implicit':
        nsrc, pitch = 3, 4
    elif kind == 'filter':
        nsrc, pitch = 2, 4
    else:
        nsrc = int(_pick(rng, (3, 4, 4))) if kind == 'test' else int(rng.integers(2, 5))
        pitch = int(_pick(rng, (3, 4, 4, 5)))
    max_h = min(3, pitch - 1)
    masks = [_mask(rng, max_h) for _ in range(nsrc + 1)]
    if kind == 'filter' and all(np.array_equal(m, masks[nsrc]) for m in masks[:nsrc]):
        other_width = 1 if masks[nsrc].shape[1] == 3 else 3
        masks[nsrc] = np.ones((max_h, other_width), dtype=bool)
    max_w = max(m.shape[1] for m in masks)
    half = max_w // 2
    capacity = 1 + (26 - max_h) // pitch
    if operands is None:
        options = [(s, d) for s in range(nsrc, capacity)
                   for d in range(1, min(3, capacity - s) + 1)
                   if s > nsrc or d > 1]
        if not options:
            return None
        total, dcount = _pick(rng, options)
    else:
        total, dcount = operands
    if total + dcount > capacity:
        return None
    # Reserve an out-of-span lattice station in the filter demonstration.
    # Other pairs may independently contain one when it fits.
    outside = kind == 'filter' or (kind != 'implicit' and rng.random() < .35)
    max_period = 8 if kind == 'filter' else 7
    minimum_period = 6 if kind == 'filter' else max(3, max_w + 1)
    options = []
    for period in range(minimum_period, max_period + 1):
        min_steps = nsrc + 1 if kind in ('implicit', 'test') else nsrc
        for steps in range(min_steps, 9):
            span = (steps + int(outside)) * period
            if span + 2 * half + 3 <= 30:
                options.append((period, steps))
    if not options:
        return None
    period, steps = _pick(rng, options)
    stations = [0, 1]
    if nsrc > 2:
        stations.extend(sorted(int(v) for v in rng.choice(
            np.arange(2, steps), nsrc - 2, replace=False)))
    all_stations = stations + [steps]
    if outside:
        all_stations.append(-1)
    min_station = min(all_stations)
    span = (steps - min_station) * period
    minimum_width = span + 2 * half + 3
    width = int(rng.integers(minimum_width, 31))
    slack = width - minimum_width
    left_margin = 1 + int(rng.integers(slack + 1))
    a = left_margin + half - min_station * period
    centers = [a + index * period for index in all_stations]
    source_counts = _partition(rng, total, nsrc, unequal=(kind == 'filter'))
    counts = source_counts + [dcount]
    if outside:
        masks.append(_mask(rng, max_h, narrow=(max_w == 1)))
        counts.append(int(rng.integers(1, capacity + 1)))
    bg, ref, highlight = roles
    source_pool = [c for c in range(10) if c not in (bg, highlight)]
    local_pool = rng.choice(source_pool, int(rng.integers(1, 5)), replace=False)
    payload = [int(_pick(rng, local_pool)) for _ in range(nsrc)]
    dest_pool = source_pool if kind == 'filter' else [c for c in range(10) if c != bg]
    payload.append(int(_pick(rng, dest_pool)))
    if outside:
        payload.append(payload[0])  # a same-hue on-phase negative witness
    # Off-phase objects are genuine stacks, not arbitrary noise. They can sit
    # inside the collection span, but cannot touch another stack's box.
    def available(x, w=1):
        lo, hi = x - w // 2, x + w // 2
        return lo >= 1 and hi < width - 1 and all(
            hi + 1 < c - m.shape[1] // 2 or lo - 1 > c + m.shape[1] // 2
            for c, m in zip(centers, masks))
    off_count = 1 if kind == 'filter' else int(_pick(rng, (0, 0, 1, 2)))
    for _ in range(off_count):
        choices = [x for x in range(a + 1, a + steps * period)
                   if (x - a) % period and available(x)]
        if not choices:
            if kind == 'filter':
                return None
            break
        centers.append(int(_pick(rng, choices)))
        masks.append(_mask(rng, max_h, narrow=True))
        counts.append(int(rng.integers(1, capacity + 1)))
        payload.append(payload[0])
    final_counts = counts.copy()
    final_counts[nsrc] += total
    extent = max(m.shape[0] + (c - 1) * pitch for m, c in zip(masks, final_counts))
    # Positive outer margins and a positive gap above the controls.
    spare = 30 - (1 + extent + 1 + 1 + 1)
    if spare < 0:
        return None
    gap = 1 + int(rng.integers(min(2, spare) + 1))
    spare -= gap - 1
    bottom_margin = 1 + int(rng.integers(min(2, spare) + 1))
    required = 1 + extent + gap + 1 + bottom_margin
    height = int(rng.integers(required, 31))
    marker_y = height - 1 - bottom_margin
    baseline = marker_y - gap
    inp = np.full((height, width), bg, dtype=np.int8)
    out = inp.copy()
    for i, (x, m, count, color) in enumerate(zip(centers, masks, counts, payload)):
        left, bottom_y = x - m.shape[1] // 2, baseline - m.shape[0]
        for k in range(count):
            _paint(inp, bottom_y - k * pitch, left, m, color)
        for k in range(final_counts[i]):
            _paint(out, bottom_y - k * pitch, left, m, highlight if i < nsrc else color)
    inp[marker_y, a] = inp[marker_y, a + period] = ref
    inp[marker_y, a + steps * period] = highlight
    if reverse:
        inp, out = inp[:, ::-1], out[:, ::-1]
    return {'input': np.rot90(inp, turns).copy(), 'output': np.rot90(out, turns).copy()}


def _key(grid):
    a = np.asarray(grid)
    return a.shape, tuple(int(v) for v in a.flat)


def _counts(num_train, num_test):
    for v, low, high in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not low <= v <= high:
            raise ValueError("illegal example counts")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    roles = tuple(int(v) for v in rng.choice(10, 3, replace=False))
    stable_roles = bool(rng.random() < .75)
    first, second = _operand_choices(rng)
    configs = [('implicit', first), ('filter', second)]
    configs.extend(('ordinary', None) for _ in range(num_train - 2))
    rng.shuffle(configs)
    train_turns = [int(rng.integers(4)) for _ in range(num_train)]
    if rng.random() < .5:
        train_turns = [train_turns[0]] * num_train
    first_direction = bool(rng.integers(2))
    directions = [first_direction, not first_direction]
    directions.extend(bool(rng.integers(2)) for _ in range(num_train - 2))
    configs.extend(('test', None) for _ in range(num_test))
    turns = train_turns + [int(_pick(rng, train_turns)) for _ in range(num_test)]
    directions.extend(bool(rng.integers(2)) for _ in range(num_test))
    puzzle, seen = {'train': [], 'test': []}, set()
    for index, ((kind, operands), rotation, reverse) in enumerate(zip(configs, turns, directions)):
        pair_roles = roles if stable_roles else tuple(int(v) for v in rng.choice(10, 3, replace=False))
        for _ in range(160):
            pair = _build_pair(rng, kind, operands, pair_roles, rotation, reverse)
            if pair is not None and _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError("bounded scene construction exhausted")
        seen.add(_key(pair['input']))
        puzzle['train' if index < num_train else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _inspect_pair(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape
    p = _parse(inp)
    assert np.array_equal(solve(inp), out), "wrong transfer or unchanged cells"
    canonical_output = np.rot90(out, p['turns'])
    objects = _components(canonical_output, p['background'])
    assert all(len(c['cells']) > 1 for c in objects), "controls must disappear"
    out_by_x = {}
    for c in objects:
        x = (c['left'] + c['right'] - 1) // 2
        out_by_x.setdefault(x, []).append(c)
    src_x = {s['x'] for s in p['sources']}
    dst = p['destination']
    assert set(out_by_x) == {s['x'] for s in p['stacks']}
    for s in p['stacks']:
        cs = sorted(out_by_x[s['x']], key=lambda c: c['top'])
        expected_count = s['count'] + (p['total'] if s['x'] == dst['x'] else 0)
        assert len(cs) == expected_count, "sum of eligible token counts required"
        expected_color = p['highlight'] if s['x'] in src_x else s['color']
        assert all(c['color'] == expected_color and c['left'] == s['left']
                   and np.array_equal(c['mask'], s['mask']) for c in cs)
        expected_ys = list(s['ys'])
        if s['x'] == dst['x']:
            expected_ys += [s['ys'][0] - j * p['pitch'] for j in range(1, p['total'] + 1)]
        assert [c['top'] for c in cs] == sorted(expected_ys)
    inactive = [s for s in p['stacks'] if s['x'] not in src_x and s['x'] != dst['x']]
    p['implicit'] = sum(s['x'] not in (p['a'], p['b']) for s in p['sources'])
    p['offphase_inside'] = any(p['low'] < s['x'] < p['high']
                              and (s['x'] - p['a']) % p['period'] for s in inactive)
    p['outside_onphase'] = any((s['x'] < p['low'] or s['x'] > p['high'])
                              and (s['x'] - p['a']) % p['period'] == 0 for s in inactive)
    p['holes'] = len(range(p['low'], p['high'] + 1, p['period'])) - len(p['sources']) - 1
    p['inactive'] = inactive
    return p


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, analyses = set(), {'train': [], 'test': []}
    for split in analyses:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            key = _key(_grid(pair['input']))
            assert key not in seen, "duplicate inputs"
            seen.add(key)
            analyses[split].append(_inspect_pair(pair))
    train = analyses['train']
    assert any(p['implicit'] for p in train), "teach unmarked on-phase donation"
    assert any(p['offphase_inside'] for p in train), "teach phase filtering"
    assert any(p['outside_onphase'] for p in train), "teach terminal span filtering"
    assert any(p['holes'] >= 1 for p in train), "empty stations contribute zero"
    assert any(p['destination']['color'] != p['highlight'] for p in train), "teach receiver color retention"
    assert any(any(not np.array_equal(s['mask'], p['destination']['mask'])
                   for s in p['sources']) for p in train), "teach receiver silhouette copying"
    assert any(len({s['count'] for s in p['sources']}) > 1 for p in train)
    totals = [p['total'] for p in train]
    dests = [p['destination']['count'] for p in train]
    assert len(set(totals)) > 1 and len(set(dests)) > 1
    assert len({s + d for s, d in zip(totals, dests)}) > 1
    assert any(totals[0] * d != s * dests[0] for s, d in zip(totals, dests))
    for rival in (lambda s, d, n: 2 * d, lambda s, d, n: s * d,
                  lambda s, d, n: max(s, d), lambda s, d, n: s,
                  lambda s, d, n: d + n, lambda s, d, n: d + 2):
        assert any(rival(p['total'], p['destination']['count'], len(p['sources']))
                   != p['total'] + p['destination']['count'] for p in train)
    # Removing A's geometric gate changes a donor set; removing B's unary
    # arithmetic changes its numerical result. These checks concern pixels,
    # rather than labels kept by the sampler.
    assert any(p['inactive'] for p in train)
    assert any(p['total'] != len(p['sources']) for p in train)
    assert {p['turns'] for p in analyses['test']} <= {p['turns'] for p in train}
    assert all(p['holes'] >= 1 and len(p['sources']) >= 3 for p in analyses['test'])


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s])
                 for s in ('train', 'test'))


def self_test():
    """Bounded checks, including 42 seeds and semantic output mutations."""
    puzzles = [generate_puzzle(seed) for seed in HOST_SEEDS]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    features, structures = [], set()
    for puzzle in puzzles:
        for pair in puzzle['train'] + puzzle['test']:
            p = _inspect_pair(pair)
            features.append(p)
            g, roles = pair['input'], {}
            normalized = tuple(roles.setdefault(int(v), len(roles)) for v in g.flat)
            structures.add((g.shape, normalized))
    assert len(structures) >= 180
    assert len({p['grid'].shape for p in features}) >= 30
    assert {p['turns'] for p in features} == {0, 1, 2, 3}
    assert {len(p['sources']) for p in features} >= {2, 3, 4}
    assert {p['pitch'] for p in features} >= {3, 4, 5}
    assert {p['period'] for p in features} >= {4, 5, 6, 7, 8}
    assert len({p['background'] for p in features}) >= 8
    assert len({(s['mask'].shape, tuple(s['mask'].flat))
                for p in features for s in p['stacks']}) >= 10
    assert any(not p['inactive'] for p in features)
    assert any(p['holes'] >= 3 for p in features)
    assert any(p['t'] < p['a'] for p in features)
    assert any(p['t'] > p['b'] for p in features)
    assert any(len({(s['mask'].shape, tuple(s['mask'].flat)) for s in p['stacks']}) > 1
               for p in features)
    # All legal explicit counts; determinism includes every returned output.
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            generate_puzzle(0, a, b)
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    for defect in ('growth', 'highlight', 'control', 'inactive'):
        broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]]
                  for s in ('train', 'test')}
        for pair in broken['train']:
            p = _parse(pair['input'])
            g = np.rot90(pair['output'], p['turns']).copy()
            if defect == 'growth':
                s = p['destination']
                _paint(g, s['ys'][0] - p['pitch'], s['left'], s['mask'], p['background'])
            elif defect == 'highlight':
                s = p['sources'][0]
                _paint(g, s['ys'][-1], s['left'], s['mask'], s['color'])
            elif defect == 'control':
                d = p['dots'][0]
                g[d['top'], d['left']] = d['color']
            else:
                source_xs = {s['x'] for s in p['sources']}
                inactive = [s for s in p['stacks'] if s['x'] not in source_xs
                            and s['x'] != p['t']]
                if not inactive:
                    continue
                s = inactive[0]
                _paint(g, s['ys'][0], s['left'], s['mask'], (s['color'] + 1) % 10)
            pair['output'] = np.rot90(g, -p['turns']).copy()
            break
        try:
            validate_generated_puzzle(broken)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("mutation accepted: " + defect)
    return {'seeds': 42, 'pairs': len(features), 'structures': len(structures),
            'shapes': len({p['grid'].shape for p in features})}
