"""Read-back ribbons: endpoint queries into a raster-ordered input history.

Only Python's standard library and NumPy are used. No parent generator is imported.
"""
from __future__ import annotations

import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= d <= 30 for d in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((0 <= g) & (g <= 9))
    return g.astype(np.int8, copy=False)


def _key(g):
    """Recover scan coordinates from geometry, independent of color names."""
    candidates = []
    for bottom in (False, True):
        for right in (False, True):
            a = g[::-1] if bottom else g
            a = a[:, ::-1] if right else a
            for axis in (0, 1):
                b = a.T if axis else a
                if min(b.shape) < 7:
                    continue
                origin, fast, slow, bg = map(int, (b[0, 0], b[0, 1], b[1, 0], b[-1, -1]))
                if len({origin, fast, slow, bg}) != 4:
                    continue
                arm = np.array([fast if i % 2 else origin for i in range(1, b.shape[1])])
                if (np.array_equal(b[0, 1:], arm) and np.all(b[1:, 0] == slow)
                        and np.all(b[-1, 1:] == bg) and np.all(b[1:, -1] == bg)):
                    candidates.append((b.copy(), (axis, bottom, right), (bg, origin, fast, slow)))
    assert len(candidates) == 1, 'Expected one alternating-fast / solid-slow corner key'
    return candidates[0]


def _components(cells):
    remaining = set(cells)
    parts = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        stack, part = [first], {first}
        while stack:
            r, c = stack.pop()
            for dr, dc in _D4:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
                    part.add(q)
        parts.append(part)
    return parts


def _parse_line(a, bg, line):
    mask = (a != bg) & (a != line)
    mask[0, :] = False
    mask[:, 0] = False
    records = []
    for cells in _components(zip(*np.nonzero(mask))):
        r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
        r1, c1 = max(r for r, c in cells) + 1, max(c for r, c in cells) + 1
        length = c1 - c0
        assert r1 - r0 == 3 and length >= 3 and length % 2 == 1
        assert len(cells) == 3 * length, 'Ribbon must be a full rectangle'
        assert r0 >= 2 and c0 >= 2 and r1 <= a.shape[0] - 2 and c1 <= a.shape[1] - 1
        block = a[r0:r1, c0:c1].copy()
        body = int(block[0, 0])
        assert np.all(block[0] == body) and np.all(block[2] == body)
        center = (r0 + 1, c0 + length // 2)
        records.append(dict(box=(r0, r1, c0, c1), block=block, body=body,
                            value=int(a[center]), center=center, lane=r0))
    assert 2 <= len(records) <= 12
    lanes = sorted({t['lane'] for t in records})
    assert len(lanes) <= 4 and all(y - x >= 6 for x, y in zip(lanes, lanes[1:]))
    records.sort(key=lambda t: (t['lane'], t['box'][2]))
    for lane in lanes:
        boxes = [t['box'] for t in records if t['lane'] == lane]
        assert all(y[2] - x[3] >= 1 for x, y in zip(boxes, boxes[1:]))
    lm = a == line
    lm[0, :] = False
    lm[:, 0] = False
    line_cells = set(zip(*np.nonzero(lm)))
    assert line_cells and all(r > 0 and c > 0 for r, c in line_cells)
    graph = {p: {(p[0] + dr, p[1] + dc) for dr, dc in _D4} & line_cells for p in line_cells}
    assert all(len(adj) in (1, 2) for adj in graph.values()), 'Selector lines must be open paths'
    ends = {p for p, adj in graph.items() if len(adj) == 1}
    for part in _components(line_cells):
        assert len(part & ends) == 2, 'No closed or branching selector lines'
    for t in records:
        r, c = t['center']
        contacts = {(r - 2, c), (r + 2, c)} & line_cells
        t['eligible'] = bool(contacts & ends)
        t['interior_contact'] = bool(contacts - ends)
    assert any(t['eligible'] for t in records)
    return records, graph, lanes


def _read(input_grid):
    g = _grid(input_grid)
    a, config, roles = _key(g)
    bg = roles[0]
    choices = []
    for line in map(int, np.unique(a[1:, 1:])):
        if line in roles:
            continue
        try:
            records, graph, lanes = _parse_line(a, bg, line)
        except AssertionError:
            continue
        choices.append((line, records, graph, lanes))
    assert len(choices) == 1, 'Line color and ribbon geometry must be uniquely recoverable'
    line, records, graph, lanes = choices[0]
    for i, t in enumerate(records):
        donors = [j for j in range(i) if records[j]['body'] == t['value']]
        t['donor'] = donors[-1] if donors else None
        t['retained'] = t['eligible'] and bool(donors)
    return dict(records=records, background=bg, roles=roles, line=line,
                configuration=config, lanes=lanes, graph=graph, shape=g.shape)


def _stack(blocks, bg):
    if not blocks:
        return np.full((1, 1), bg, dtype=np.int8)
    width = max(b.shape[1] for b in blocks)
    result = np.full((3 * len(blocks), width), bg, dtype=np.int8)
    for i, b in enumerate(blocks):
        offset = (width - b.shape[1]) // 2
        result[3*i:3*i+3, offset:offset+b.shape[1]] = b
    return result


def _predict(d, alternative='rule'):
    """Also evaluate concrete simpler hypotheses for demonstration checks."""
    records, bg = d['records'], d['background']
    order = list(range(len(records)))
    if alternative == 'reverse_fast':
        order = [i for lane in d['lanes'] for i in order[::-1] if records[i]['lane'] == lane]
    elif alternative == 'reverse_slow':
        order = [i for lane in d['lanes'][::-1] for i in order if records[i]['lane'] == lane]
    elif alternative == 'fast_major':
        order.sort(key=lambda i: (records[i]['box'][2], records[i]['lane']))
    blocks, updated = [], {}
    for position, i in enumerate(order):
        t = records[i]
        selected = (t['eligible'] or alternative == 'all_queries' or
                    (alternative == 'touching_queries' and t['interior_contact']))
        if not selected:
            continue
        prior = order[:position]
        if alternative == 'future_allowed':
            prior = [j for j in order if j != i]
        elif alternative == 'lane_local':
            prior = [j for j in prior if records[j]['lane'] == t['lane']]
        donors = [j for j in prior if records[j]['value' if alternative == 'reverse_lookup' else 'body'] == t['value']]
        if not donors:
            continue
        donor = donors[0] if alternative == 'first_donor' else donors[-1]
        if alternative == 'spatial_nearest':
            donor = min(donors, key=lambda j: (sum(abs(x-y) for x, y in
                        zip(records[j]['center'], t['center'])), j))
        value = updated.get(donor, records[donor]['value']) if alternative == 'updated_donor' else records[donor]['value']
        if alternative == 'reverse_lookup':
            value = records[donor]['body']
        if alternative == 'no_lookup':
            value = t['value']
        b = t['block'].copy()
        b[1, b.shape[1] // 2] = value
        blocks.append(b)
        updated[i] = value
    result = _stack(blocks, bg)
    if alternative == 'global_width' and blocks:
        width = max(t['block'].shape[1] for t in records)
        if result.shape[1] < width:
            pad = (width - result.shape[1]) // 2
            result = np.pad(result, ((0, 0), (pad, pad)), constant_values=bg)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Resolve the visible prefix history; no generation or training state is read."""
    return _predict(_read(input_grid))


def _counts(rng, num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1)) if num_train is not None else int(rng.integers(1, 5))
    if num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _sequence(rng, mode):
    # Identities are locally relabeled after building these relational witnesses.
    if mode == 'shadow':
        records = [(1, 2, 0), (1, 3, 0), (0, 1, 1), (1, 4, 0),
                   (2, 1, 2), (0, 1, 1), (3, 4, 1), (4, 0, 0)]
    elif mode == 'snapshot':
        records = [(1, 2, 0), (2, 3, 0), (1, 2, 1), (0, 1, 1),
                   (4, 1, 2), (3, 3, 0), (2, 3, 1), (0, 4, 1)]
    else:
        n = int(rng.integers(2, 13))
        k = int(rng.integers(2, 9))
        records = [(int(rng.integers(k)), int(rng.integers(k)),
                    int(rng.choice([0, 1, 2], p=[.3, .55, .15]))) for _ in range(n)]
        # Guarantee a visible successful query, while leaving other relations free.
        donor, query = sorted(map(int, rng.choice(n, 2, replace=False)))
        a, b, _ = records[query]
        records[query] = (a, records[donor][0], 1)
    if mode in ('shadow', 'snapshot'):
        # Insert irrelevant identities anywhere, moving witness positions without
        # disturbing precedence among its five key identities.
        for _ in range(int(rng.integers(5))):
            body = int(rng.integers(5, 8))
            records.insert(int(rng.integers(len(records) + 1)), (body, int(rng.integers(8)), int(rng.integers(3))))
    return records


def _gaps(rng, occupied, count, minimum, limit=29):
    base = occupied + minimum * (count - 1) + 4
    assert base <= limit
    extent = base + int(rng.integers(limit - base + 1))
    gaps = [2] + [minimum] * (count - 1) + [2]
    extra = rng.multinomial(extent - base, rng.dirichlet(np.full(count + 1, .7)))
    return [int(x + y) for x, y in zip(gaps, extra)], extent


def _layout(rng, n, mode, sequence):
    min_lanes = max(2 if mode != 'random' else 1, (n + 5) // 6)
    lanes = int(rng.integers(min_lanes, min(4, n) + 1))
    counts = [1] * lanes
    preference = rng.dirichlet(np.ones(lanes))
    for _ in range(n - lanes):
        possible = [i for i, c in enumerate(counts) if c < 6]
        p = preference[possible]
        counts[int(rng.choice(possible, p=p / p.sum()))] += 1
    ceiling = int(rng.choice([5, 7, 9, 13, 17, 23]))
    lengths = [int(rng.choice(np.arange(3, ceiling + 1, 2))) for _ in range(n)]
    if mode == 'random' and rng.random() < .2:
        cap = min((29 - 4 - (count - 1)) // count for count in counts)
        common = int(rng.choice(np.arange(3, min(cap, ceiling) + 1, 2)))
        lengths = [common] * n
    if mode == 'shadow':
        # Make successful ribbons shorter than an ignored definition. This is
        # joint evidence for lookup, filtering, and survivor-dependent width.
        lengths = [3 if kind == 1 else int(rng.choice([3, 5, 7, 9])) for _, _, kind in sequence]
        query_indices = [i for i, (_, _, kind) in enumerate(sequence) if kind == 1]
        lengths[query_indices[1]] = 5
    start = 0
    for count in counts:
        budget = 29 - 4 - (count - 1)
        while sum(lengths[start:start+count]) > budget:
            choices = [i for i in range(start, start+count) if lengths[i] > 3]
            lengths[int(rng.choice(choices))] -= 2
        start += count
    slow_gaps, h = _gaps(rng, 3 * lanes, lanes, 3)
    minimum_w, start = 0, 0
    for count in counts:
        minimum_w = max(minimum_w, sum(lengths[start:start+count]) + count - 1 + 4)
        start += count
    w = minimum_w + int(rng.integers(30 - minimum_w))
    positions, r, start = [], slow_gaps[0] + 1, 0
    for lane, count in enumerate(counts):
        gaps, _ = _gaps(rng, sum(lengths[start:start+count]), count, 1, limit=w)
        # _gaps samples its extent; distribute the unsampled remainder too.
        spare = w - (sum(lengths[start:start+count]) + sum(gaps))
        gaps[-1] += spare
        c = gaps[0] + 1
        for j in range(count):
            i = start + j
            positions.append((r, c, lengths[i]))
            c += lengths[i] + gaps[j+1]
        start += count
        r += 3 + slow_gaps[lane+1]
    return positions, (h + 1, w + 1)


def _paint(rng, sequence, positions, shape, configuration, roles, patterned):
    bg, origin, fast, slow, line = roles
    identities = list(map(int, rng.permutation([c for c in range(10) if c not in (bg, line)])))
    a = np.full(shape, bg, dtype=np.int8)
    a[0, 0] = origin
    a[0, 1:] = [fast if i % 2 else origin for i in range(1, shape[1])]
    a[1:, 0] = slow
    blocks, lookup, successful = [], {}, []
    for i, ((source, value, kind), (r, c, length)) in enumerate(zip(sequence, positions)):
        body, center = identities[source], identities[value]
        b = np.full((3, length), body, dtype=np.int8)
        if patterned and (rng.random() < .45 or not blocks):
            indices = [j for j in range(length) if j != length // 2]
            for j in rng.choice(indices, size=int(rng.integers(1, len(indices) + 1)), replace=False):
                b[1, j] = rng.choice(identities)
        b[1, length // 2] = center
        a[r:r+3, c:c+length] = b
        blocks.append(b)
        mid = c + length // 2
        if kind:
            side = r - 1 if rng.random() < .5 else r + 3
            if kind == 1:
                extent = int(rng.integers(1, length // 2 + 1))
                left = bool(rng.integers(2))
                lo, hi = (mid - extent, mid) if left else (mid, mid + extent)
            else:
                lo = mid - int(rng.integers(1, length // 2 + 1))
                hi = mid + int(rng.integers(1, length // 2 + 1))
            a[side, lo:hi+1] = line
        if kind == 1 and value in lookup:
            out = b.copy()
            out[1, length // 2] = identities[lookup[value]]
            successful.append(out)
        lookup[source] = value  # Every input record, including failed queries.
    output = _stack(successful, bg)
    axis, bottom, right = configuration
    if axis:
        a = a.T
    if bottom:
        a = a[::-1]
    if right:
        a = a[:, ::-1]
    return dict(input=a.copy(), output=output)


_ALTERNATIVES = ('first_donor', 'future_allowed', 'all_queries', 'touching_queries', 'updated_donor',
                 'lane_local', 'reverse_fast', 'reverse_slow', 'fast_major',
                 'no_lookup', 'global_width', 'spatial_nearest', 'reverse_lookup')


def _evidence(descriptors):
    for alt in _ALTERNATIVES:
        assert any(not np.array_equal(_predict(d), _predict(d, alt)) for d in descriptors), alt
    records = [t for d in descriptors for t in d['records']]
    assert any(t['interior_contact'] and t['donor'] is not None for t in records)
    assert any(t['eligible'] and t['donor'] is None for t in records)
    assert any(t['retained'] and t['value'] == d['records'][t['donor']]['value']
               for d in descriptors for t in d['records'])
    assert any(t['body'] == t['value'] for t in records)
    assert any(len({t['block'].shape[1] for t in d['records'] if t['retained']}) > 1 for d in descriptors)
    assert any(t['retained'] and not np.array_equal(t['block'], t['block'][:, ::-1]) for t in records)
    # Rule out nearest-donor lookup even if a competitor chooses a different
    # tie break: an earlier conflicting donor is strictly closer than the last.
    strict_spatial_contrast = False
    for d in descriptors:
        for i, t in enumerate(d['records']):
            if not t['retained']:
                continue
            donor = d['records'][t['donor']]
            distance = sum(abs(x-y) for x, y in zip(t['center'], donor['center']))
            strict_spatial_contrast |= any(
                earlier['body'] == t['value'] and earlier['value'] != donor['value']
                and sum(abs(x-y) for x, y in zip(t['center'], earlier['center'])) < distance
                for earlier in d['records'][:i])
    assert strict_spatial_contrast, 'Spatial precedence needs a strict-distance contrast'
    configs = [d['configuration'] for d in descriptors]
    for factor in range(3):
        assert len({c[factor] for c in configs}) == 2


def _grid_key(g):
    return g.shape, g.tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, descriptions = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _grid_key(inp)
            assert key not in seen, 'Repeated input'
            seen.add(key)
            d = _read(inp)
            selected = [t for t in d['records'] if t['retained']]
            assert 1 <= len(selected) <= 10
            assert out.shape == (3 * len(selected), max(t['block'].shape[1] for t in selected))
            assert np.array_equal(_predict(d), out), 'Incorrect precedence, snapshot lookup, or copied ribbon'
            # Cell-level invariants: only the center may change; all engraving is copied.
            width = out.shape[1]
            for i, t in enumerate(selected):
                b = t['block']
                offset = (width - b.shape[1]) // 2
                actual = out[3*i:3*i+3, offset:offset+b.shape[1]]
                keep = np.ones(b.shape, dtype=bool)
                keep[1, b.shape[1] // 2] = False
                assert np.array_equal(actual[keep], b[keep])
                assert int(actual[1, b.shape[1] // 2]) == d['records'][t['donor']]['value']
                assert np.all(out[3*i:3*i+3, :offset] == d['background'])
                assert np.all(out[3*i:3*i+3, offset+b.shape[1]:] == d['background'])
            descriptions[split].append(d)
    _evidence(descriptions['train'])
    training_configs = {d['configuration'] for d in descriptions['train']}
    assert all(d['configuration'] not in training_configs for d in descriptions['test'])


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    nt, ne = _counts(rng, num_train, num_test)
    all_configs = [(axis, bottom, right) for axis in (0, 1)
                   for bottom in (False, True) for right in (False, True)]
    first = all_configs[int(rng.integers(8))]
    second = (1 - first[0], not first[1], not first[2])
    held = [c for c in all_configs if c not in (first, second)]
    test_configs = [held[int(rng.integers(6))] for _ in range(ne)]
    train_pool = [c for c in all_configs if c not in test_configs]
    train_configs = [first, second] + [train_pool[int(rng.integers(len(train_pool)))] for _ in range(nt - 2)]
    stable_roles = tuple(map(int, rng.choice(10, size=5, replace=False)))
    varied_roles = bool(rng.random() < .65)
    patterned = bool(rng.random() < .7)
    puzzle = {'train': [], 'test': []}
    used = set()
    # Rebuild the two witnesses only if accidental length or geometry choices
    # leave one of the documented competing hypotheses indistinguishable.
    for _ in range(128):
        demos = []
        for mode, config in zip(('shadow', 'snapshot'), train_configs[:2]):
            seq = _sequence(rng, mode)
            pos, shape = _layout(rng, len(seq), mode, seq)
            roles = tuple(map(int, rng.choice(10, size=5, replace=False))) if varied_roles else stable_roles
            pair = _paint(rng, seq, pos, shape, config, roles, True)
            demos.append(pair)
        try:
            ds = [_read(p['input']) for p in demos]
            _evidence(ds)
            assert all(1 <= sum(t['retained'] for t in d['records']) <= 10 for d in ds)
        except AssertionError:
            continue
        puzzle['train'] = demos
        used.update(_grid_key(p['input']) for p in demos)
        break
    else:
        raise AssertionError('Unable to construct complete training evidence')
    for split, configs in (('train', train_configs[2:]), ('test', test_configs)):
        for config in configs:
            for _ in range(128):
                seq = _sequence(rng, 'random')
                pos, shape = _layout(rng, len(seq), 'random', seq)
                roles = tuple(map(int, rng.choice(10, size=5, replace=False))) if varied_roles else stable_roles
                pair = _paint(rng, seq, pos, shape, config, roles, patterned)
                d = _read(pair['input'])
                count = sum(t['retained'] for t in d['records'])
                key = _grid_key(pair['input'])
                if 1 <= count <= 10 and key not in used:
                    assert np.array_equal(pair['output'], _predict(d))
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('Unable to produce a distinct query history')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def puzzle_features(puzzle):
    result = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            d = _read(pair['input'])
            ts = d['records']
            selected = [t for t in ts if t['retained']]
            result.append(dict(split=split, configuration=d['configuration'], roles=d['roles'] + (d['line'],),
                shape=d['shape'], lanes=tuple(sum(t['lane'] == lane for t in ts) for lane in d['lanes']),
                lengths=tuple(t['block'].shape[1] for t in ts), count=len(ts), survivors=len(selected),
                output_shape=pair['output'].shape,
                shadowing=sum(t['retained'] and len({x['value'] for x in ts[:i] if x['body'] == t['value']}) > 1
                              for i, t in enumerate(ts)),
                cross_lane=sum(t['retained'] and ts[t['donor']]['lane'] != t['lane'] for t in ts),
                retained_donor=sum(t['retained'] and ts[t['donor']]['retained'] for t in ts),
                failed=sum(t['eligible'] and not t['retained'] for t in ts),
                interior=sum(t['interior_contact'] and t['donor'] is not None for t in ts),
                monochrome=sum(np.all(t['block'] == t['body']) for t in ts),
                unchanged=sum(t['retained'] and t['value'] == ts[t['donor']]['value'] for t in ts),
                solid_result=sum(t['retained'] and t['body'] != t['value'] and t['body'] == ts[t['donor']]['value'] for t in ts),
                engraved=sum(not np.all(np.delete(t['block'][1], t['block'].shape[1]//2) == t['body']) for t in ts),
                ignored_max=max(t['block'].shape[1] for t in ts) > max(t['block'].shape[1] for t in selected)))
    return result


def self_test():
    """Bounded 42-seed suite, invariants, factor diversity, and negative checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    keys = {tuple(_grid_key(p['input']) for s in ('train', 'test') for p in task[s]) for task in puzzles}
    assert len(keys) == 42
    rows = [row for p in puzzles for row in puzzle_features(p)]
    assert len({r['shape'] for r in rows}) >= 50
    assert len({r['lanes'] for r in rows}) >= 20
    assert len({r['lengths'] for r in rows}) >= 150
    assert len({r['roles'] for r in rows}) >= 35
    assert {r['configuration'] for r in rows} == {(a, b, c) for a in (0, 1) for b in (False, True) for c in (False, True)}
    assert max(r['count'] for r in rows) == 12 and min(r['count'] for r in rows) <= 3
    assert max(max(r['lengths']) for r in rows) >= 19
    assert max(max(r['lanes']) for r in rows) >= 5
    assert any(len(r['lanes']) == 1 for r in rows) and any(len(r['lanes']) == 4 for r in rows)
    for feature in ('shadowing', 'cross_lane', 'retained_donor', 'failed', 'interior', 'unchanged', 'engraved', 'ignored_max'):
        assert sum(bool(r[feature]) for r in rows) >= 10, feature
    assert any(r['monochrome'] for r in rows) and any(r['solid_result'] for r in rows)
    for task in puzzles:
        validate_generated_puzzle(task)
        for split in ('train', 'test'):
            for pair in task[split]:
                assert np.array_equal(solve(pair['input']), pair['output'])
                # Proper whole-input rotations leave the extracted reading sequence invariant.
                for rotation in (1, 2, 3):
                    assert np.array_equal(solve(np.rot90(pair['input'], rotation)), pair['output'])
    again = generate_puzzle(0)
    for split in ('train', 'test'):
        for p, q in zip(puzzles[0][split], again[split]):
            assert np.array_equal(p['input'], q['input']) and np.array_equal(p['output'], q['output'])
    # Literal color values carry no semantics: a bijection must commute with solve.
    recolor = np.array([7, 2, 9, 0, 5, 1, 8, 4, 6, 3], dtype=np.int8)
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        assert np.array_equal(solve(recolor[pair['input']]), recolor[pair['output']])
    # Direct boundary histories cover extremes that need not occur in 42 draws.
    boundary_rng = np.random.default_rng(808)
    wide = _paint(boundary_rng, [(0, 1, 0), (2, 0, 1)],
                  [(3, 3, 3), (9, 3, 23)], (14, 29),
                  (0, False, False), (0, 1, 2, 3, 4), True)
    assert wide['output'].shape == (3, 23)
    assert np.array_equal(solve(wide['input']), wide['output'])
    seq = [(0, 0, 0)] + [(0, 0, 1)] * 10
    positions = [(3 + 6 * (i // 4), 3 + 4 * (i % 4), 3) for i in range(11)]
    tall = _paint(boundary_rng, seq, positions, (20, 20),
                  (1, True, True), (0, 1, 2, 3, 4), False)
    assert tall['output'].shape == (30, 3)
    assert np.array_equal(solve(tall['input']), tall['output'])
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            task = generate_puzzle(12345 + 10*a + b, a, b)
            assert len(task['train']) == a and len(task['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Incorrect output accepted')
