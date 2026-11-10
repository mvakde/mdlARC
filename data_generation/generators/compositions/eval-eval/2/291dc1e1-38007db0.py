"""Directed local exceptions registered by their first template-relative fault.

Only NumPy and the standard library are used. The solver reads no examples,
generation parameters, files, seed, or mutable global state.
"""

from collections import Counter
import copy
import numpy as np


def _integer(name, value, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}: {value}")
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions"
    assert np.issubdtype(a.dtype, np.integer), "integer cells required"
    assert np.all((0 <= a) & (a <= 9)), "colors must be 0..9"
    return a.astype(np.int8, copy=False)


def _grid_key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _find_key(a):
    """Identify roles from edge geometry, not from fixed color identities."""
    assert min(a.shape) >= 5, "insufficient clearance for key and payload"
    candidates = []
    for bottom in (False, True):
        for right in (False, True):
            b = a[::-1] if bottom else a
            b = b[:, ::-1] if right else b
            origin, bg = int(b[0, 0]), int(b[-1, -1])
            if not (np.all(b[-1, 1:] == bg) and np.all(b[1:, -1] == bg)):
                continue
            for axis in (0, 1):
                solid = b[0, 1:] if axis == 0 else b[1:, 0]
                dotted = b[1:, 0] if axis == 0 else b[0, 1:]
                fast, slow = int(solid[0]), int(dotted[0])
                if len({origin, bg, fast, slow}) != 4:
                    continue
                expected = np.where(np.arange(len(dotted)) % 2 == 0, slow, bg)
                if np.all(solid == fast) and np.array_equal(dotted, expected):
                    candidates.append(dict(axis=axis, bottom=bottom, right=right,
                                           origin=origin, background=bg, fast=fast, slow=slow))
    assert len(candidates) == 1, "key must identify one origin and ordered axis pair"
    return candidates[0]


def _rectangles(mask):
    todo = set(map(tuple, np.argwhere(mask)))
    boxes = []
    while todo:
        first = min(todo)
        todo.remove(first)
        pending, points = [first], [first]
        while pending:
            r, c = pending.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in todo:
                    todo.remove(q)
                    pending.append(q)
                    points.append(q)
        r0, c0 = np.min(points, axis=0)
        r1, c1 = np.max(points, axis=0) + 1
        assert len(points) == (r1-r0)*(c1-c0), "payload has a hole or nonrectangular shape"
        boxes.append(tuple(map(int, (r0, r1, c0, c1))))
    return boxes


def _parse(input_grid):
    a = _grid(input_grid)
    info = _find_key(a)
    axis, bottom, right = (info[k] for k in ("axis", "bottom", "right"))
    bg = info['background']
    mask = a != bg
    mask[-1 if bottom else 0, :] = False
    mask[:, -1 if right else 0] = False
    boxes = _rectangles(mask)
    lanes = {}
    for r0, r1, c0, c1 in boxes:
        assert r0 >= 1 + int(not bottom) and r1 <= a.shape[0]-1-int(bottom), "row clearance"
        assert c0 >= 1 + int(not right) and c1 <= a.shape[1]-1-int(right), "column clearance"
        thickness = c1-c0 if axis else r1-r0
        length = r1-r0 if axis else c1-c0
        assert thickness == 2 and 2 <= length <= 8, "thin strip dimensions"
        lanes.setdefault(c0 if axis else r0, []).append((r0, r1, c0, c1))
    assert 1 <= len(lanes) <= 9, "lane count"
    starts = sorted(lanes)
    assert all(y-x >= 3 for x, y in zip(starts, starts[1:])), "overlapping/staggered lanes"
    slow_reverse = right if axis else bottom
    fast_reverse = bottom if axis else right
    # np.rot90 sends down to right with k=1 and up to right with k=3.
    rotation = (3 if bottom else 1) if axis else (2 if right else 0)
    records = []
    for v in sorted(lanes, reverse=slow_reverse):
        row_boxes = sorted(lanes[v], key=lambda b: b[0] if axis else b[2], reverse=fast_reverse)
        intervals = sorted((b[0], b[1]) if axis else (b[2], b[3]) for b in row_boxes)
        assert all(y[0]-x[1] >= 1 for x, y in zip(intervals, intervals[1:])), "strip clearance"
        strips = [np.rot90(a[r:s, c:d], rotation).copy() for r, s, c, d in row_boxes]
        assert 3 <= len(strips) <= 9, "local majority requires at least three copies"
        assert len({s.shape for s in strips}) == 1, "unequal strip sizes within a lane"
        counts = Counter(_grid_key(s) for s in strips)
        assert len(counts) == 2, "exactly one exception bitmap per lane"
        winner, frequency = counts.most_common(1)[0]
        assert frequency == len(strips)-1, "unique repeated local master required"
        master = next(s for s in strips if _grid_key(s) == winner)
        index = next(i for i, s in enumerate(strips) if _grid_key(s) != winner)
        exception = strips[index]
        palette = set(map(int, np.unique(master)))
        assert len(palette) == 2 and bg not in palette, "two-color local master"
        assert set(map(int, np.unique(exception))) <= palette, "exception palette"
        changed = np.flatnonzero(np.any(exception != master, axis=0))
        assert len(changed), "visible exception required"
        records.append(dict(master=master, exception=exception, anchor=int(changed[0]),
                            last=int(changed[-1]), length=master.shape[1], copies=len(strips),
                            index=index, boxes=row_boxes))
    return info | dict(records=records, input_shape=a.shape)


def _register(records, background, *, mode='first', selection='exception',
              reflection=None, reverse=False):
    """Registration, also used for explicit competing-rule measurements."""
    blocks, anchors = [], []
    for record in records[::-1] if reverse else records:
        master, exception = record['master'], record['exception']
        if reflection == 'rows':
            master, exception = master[::-1], exception[::-1]
        elif reflection == 'columns':
            master, exception = master[:, ::-1], exception[:, ::-1]
        block = exception if selection == 'exception' else master
        columns = np.flatnonzero(np.any(master != exception, axis=0))
        blocks.append(block)
        anchors.append(int(columns[-1] if mode == 'last' else columns[0]))
    if mode == 'center':
        width = max(b.shape[1] for b in blocks)
        offsets = [(width-b.shape[1])//2 for b in blocks]
    else:
        left = max(anchors)
        right = max(b.shape[1]-1-a for b, a in zip(blocks, anchors))
        width = left+1+right
        offsets = [left-a for a in anchors]
    result = np.full((2*len(blocks), width), background, dtype=np.int8)
    for i, (block, start) in enumerate(zip(blocks, offsets)):
        result[2*i:2*i+2, start:start+block.shape[1]] = block
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive selection, orientation, anchors and dimensions entirely from input."""
    info = _parse(input_grid)
    return _register(info['records'], info['background'])


def _sample_master(rng, length, colors):
    style = int(rng.integers(6))
    mask = np.zeros((2, length), dtype=bool)
    if style == 0:
        mask.flat[rng.choice(mask.size, int(rng.integers(1, length+1)), replace=False)] = True
    elif style == 1:
        mask[:] = rng.integers(2, size=length).astype(bool)
    elif style == 2:
        mask[0] = True
        if rng.random() < .5:
            mask = mask[::-1].copy()
    elif style == 3:
        r, c = np.indices(mask.shape)
        mask = (r+c) % int(rng.integers(2, 4)) == 0
        if rng.random() < .5:
            mask |= mask[:, ::-1]
    elif style == 4:
        point = (int(rng.integers(2)), int(rng.integers(length)))
        mask[point] = True
        for _ in range(int(rng.integers(1, mask.size))):
            frontier = [(r, c) for r in range(2) for c in range(length) if not mask[r, c]
                        and any(0 <= rr < 2 and 0 <= cc < length and mask[rr, cc]
                                for rr, cc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1)))]
            if frontier:
                mask[frontier[int(rng.integers(len(frontier)))]] = True
    else:
        mask = rng.random((2, length)) < rng.uniform(.15, .85)
        if rng.random() < .3:
            mask |= mask[:, ::-1]
    if np.all(mask) or not np.any(mask):
        mask.flat[int(rng.integers(mask.size))] = not bool(mask.flat[0])
    return np.where(mask, colors[1], colors[0]).astype(np.int8)


def _make_lane(rng, length, colors, kind=None, anchor=None, spread=False):
    master = _sample_master(rng, length, colors)
    kind = str(rng.choice(['add', 'remove', 'balanced', 'scatter', 'shift'])) if kind is None else kind
    anchor = int(rng.integers(length)) if anchor is None else anchor
    pool = [(r, c) for r in range(2) for c in range(anchor, length)]
    first = (int(rng.integers(2)), anchor)
    cells = [first]
    if spread:
        cells.append((int(rng.integers(2)), int(rng.integers(anchor+2, length))))
    if kind == 'balanced':
        high = min(3, length//2, len(pool)//2)
        count = 2 * int(rng.integers(1, high+1))
    else:
        count = int(rng.integers(len(cells), min(length, len(pool))+1))
    remaining = [p for p in pool if p not in cells]
    if count > len(cells):
        cells.extend(remaining[int(i)] for i in rng.choice(len(remaining), count-len(cells), replace=False))
    if kind in ('add', 'remove', 'balanced'):
        if kind == 'balanced':
            values = [colors[0]]*(count//2) + [colors[1]]*(count//2)
            rng.shuffle(values)
        else:
            values = [colors[0] if kind == 'add' else colors[1]] * count
        for point, value in zip(cells, values):
            master[point] = value
        # A pure edit may overwrite the last cell of one template color.
        if len(np.unique(master)) == 1:
            untouched = [(r, c) for r in range(2) for c in range(length) if (r, c) not in cells]
            p = untouched[int(rng.integers(len(untouched)))]
            master[p] = colors[1] if master[p] == colors[0] else colors[0]
    exception = master.copy()
    for point in cells:
        exception[point] = colors[1] if master[point] == colors[0] else colors[0]
    if kind == 'shift':
        shifted = np.roll(master, int(rng.choice([-1, 1])), axis=1)
        changes = int(np.count_nonzero(master != shifted))
        if 1 <= changes <= length:
            exception = shifted
    columns = np.flatnonzero(np.any(master != exception, axis=0))
    return dict(master=master, exception=exception, anchor=int(columns[0]),
                last=int(columns[-1]), length=length)


def _gaps(rng, occupied, count, extent):
    spare = extent-occupied-count-1
    assert spare >= 0
    weights = rng.dirichlet(np.full(count+1, float(rng.choice([.4, 1., 3.]))))
    return (1+rng.multinomial(spare, weights)).tolist()


def _extent(rng, minimum):
    spare = 29-minimum
    assert spare >= 0
    cap = min(6, spare) if rng.random() < .65 else spare
    return minimum + int(rng.integers(cap+1))


def _paint(rng, records, configuration, roles):
    axis, bottom, right = configuration
    bg, origin, fast, slow = roles
    n = len(records)
    slow_extent = _extent(rng, 3*n+1)
    fast_extent = _extent(rng, max(r['copies']*(r['length']+1)+1 for r in records))
    interior = np.full((slow_extent, fast_extent), bg, dtype=np.int8)
    # Coordinates run inward from the origin along slow and fast. Depending on
    # their handedness, slow may oppose the proper output thickness basis.
    reverse_thickness = (bottom != right) if axis == 0 else (bottom == right)
    lane_gaps = _gaps(rng, 2*n, n, slow_extent)
    v = lane_gaps[0]
    for record, gap in zip(records, lane_gaps[1:]):
        length, copies = record['length'], record['copies']
        fast_gaps = _gaps(rng, length*copies, copies, fast_extent)
        u = fast_gaps[0]
        for index in range(copies):
            block = record['exception'] if index == record['index'] else record['master']
            interior[v:v+2, u:u+length] = block[::-1] if reverse_thickness else block
            u += length+fast_gaps[index+1]
        v += 2+gap
    a = np.full((slow_extent+1, fast_extent+1), bg, dtype=np.int8)
    a[1:, 1:] = interior
    a[0, 0] = origin
    a[0, 1:] = fast
    a[1:, 0] = np.where(np.arange(slow_extent) % 2 == 0, slow, bg)
    if axis:
        a = a.T
    if bottom:
        a = a[::-1]
    if right:
        a = a[:, ::-1]
    return {'input': a.copy(), 'output': _register(records, bg)}


def _histogram_same(record):
    return Counter(map(int, record['master'].flat)) == Counter(map(int, record['exception'].flat))


def _witness_ok(records, bg):
    answer = _register(records, bg)
    if answer.shape[1] <= max(r['length'] for r in records):
        return False
    alternatives = [dict(reverse=True), dict(reflection='rows'), dict(reflection='columns'), dict(mode='last')]
    if any(np.array_equal(answer, _register(records, bg, **a)) for a in alternatives):
        return False
    # Equal-length masters must be genuinely local, rather than global copies.
    masters = {}
    for r in records:
        masters.setdefault(r['length'], set()).add(_grid_key(r['master']))
    return any(len(m) >= 2 for m in masters.values())


def _example(rng, configuration, *, witness=False, lanes=None, duplicate=False, shared=None):
    roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    bg = roles[0]
    pool = [c for c in range(10) if c != bg]
    common = tuple(map(int, rng.choice(pool, 2, replace=False)))
    shared = bool(rng.random() < .35) if shared is None else shared
    lanes = int(rng.integers(1, 10)) if lanes is None else lanes
    for _ in range(64):
        records = []
        if witness:
            long = int(rng.integers(5, 9))
            short = int(rng.integers(4, long))
            kinds = ['add', 'remove']
            rng.shuffle(kinds)
            records.append(_make_lane(rng, long, common, kinds[0], 0))
            records.append(_make_lane(rng, long, common, kinds[1], long-1))
            records.append(_make_lane(rng, short, common, 'balanced',
                                      int(rng.integers(1, short-2)), spread=True))
            if duplicate:
                records.append(copy.deepcopy(records[int(rng.integers(3))]))
        while len(records) < lanes:
            length = int(rng.integers(2, 9))
            earlier = [r for r in records if r['length'] == length]
            if earlier and rng.random() < .20:
                records.append(copy.deepcopy(earlier[int(rng.integers(len(earlier)))]))
            else:
                colors = common if shared else tuple(map(int, rng.choice(pool, 2, replace=False)))
                records.append(_make_lane(rng, length, colors))
        rng.shuffle(records)
        for record in records:
            maximum = 28//(record['length']+1)
            record['copies'] = int(rng.integers(3, maximum+1))
            record['index'] = int(rng.integers(record['copies']))
        if witness:
            # Condition only the demonstrations against fixed-rank shortcuts.
            if len({r['index'] for r in records}) == 1:
                records[0]['index'] = (records[0]['index']+1) % records[0]['copies']
            if not any(0 < r['index'] < r['copies']-1 for r in records):
                records[int(rng.integers(lanes))]['index'] = 1
            if not any(r['index'] > 0 for r in records):
                records[0]['index'] = 1
            if not _witness_ok(records, bg):
                continue
        return _paint(rng, records, configuration, roles)
    raise AssertionError('bounded witness construction failed')


def _configuration(info):
    return info['axis'], info['bottom'], info['right']


def _training_evidence(descriptors):
    assert {d['axis'] for d in descriptors} == {0, 1}, 'teach both fast axes'
    assert {d['bottom'] for d in descriptors} == {False, True}, 'teach opposite origin rows'
    assert {d['right'] for d in descriptors} == {False, True}, 'teach opposite origin columns'
    assert len({len(d['records']) for d in descriptors}) >= 2, 'vary lane count'
    all_records = [r for d in descriptors for r in d['records']]
    assert any(_histogram_same(r) and r['last']-r['anchor'] >= 2 and 0 < r['anchor'] < r['length']-1
               for r in all_records), 'balanced multi-column interior fault evidence'
    pure_directions = set()
    for r in all_records:
        change = r['master'] != r['exception']
        old = set(map(int, r['master'][change]))
        if len(old) == 1:
            color = min(map(int, np.unique(r['master'])))
            pure_directions.add(int(np.count_nonzero(r['exception'] == color) -
                                    np.count_nonzero(r['master'] == color)) > 0)
    assert pure_directions == {False, True}, 'teach additions and removals'
    assert any(len({_grid_key(r['exception']) for r in d['records']}) < len(d['records'])
               for d in descriptors), 'preserve repeated selected occurrences'
    assert any(len({r['index'] for r in d['records']}) > 1 and
               any(0 < r['index'] < r['copies']-1 for r in d['records']) for d in descriptors)
    for alternative in (dict(reverse=True), dict(reflection='rows'), dict(reflection='columns'),
                        dict(selection='master'), dict(mode='center'), dict(mode='last')):
        assert any(not np.array_equal(_register(d['records'], d['background']),
                                      _register(d['records'], d['background'], **alternative))
                   for d in descriptors), f'undistinguished alternative: {alternative}'
    assert any(_register(d['records'], d['background']).shape[1] >
               max(r['length'] for r in d['records']) for d in descriptors), 'faults change envelope'


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema'
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'split lists'
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4, 'pair counts'
    assert len(puzzle['train']) > len(puzzle['test']), 'training count must exceed test count'
    seen, descriptions = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema'
            a, b = _grid(pair['input']), _grid(pair['output'])
            key = _grid_key(a)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            d = _parse(a)
            records, bg = d['records'], d['background']
            p = max(r['anchor'] for r in records)
            q = max(r['length']-1-r['anchor'] for r in records)
            assert b.shape == (2*len(records), p+1+q), 'tight fault envelope dimensions'
            for i, r in enumerate(records):
                start = p-r['anchor']
                block = b[2*i:2*i+2]
                assert np.array_equal(block[:, start:start+r['length']], r['exception']), 'exact exception copy'
                assert np.all(block[:, :start] == bg) and np.all(block[:, start+r['length']:] == bg), 'padding'
                first = np.flatnonzero(np.any(r['master'] != block[:, start:start+r['length']], axis=0))[0]
                assert start+first == p, 'shared first-fault column'
            assert np.any(b[:, 0] != bg) and np.any(b[:, -1] != bg), 'minimal side bounds'
            assert set(map(int, np.unique(b))) <= set(map(int, np.unique(a))), 'no invented colors'
            assert np.array_equal(solve(a), b), 'input-only solution disagrees'
            descriptions[split].append(d)
    _training_evidence(descriptions['train'])
    training_configurations = {_configuration(d) for d in descriptions['train']}
    assert _configuration(descriptions['test'][0]) not in training_configurations, 'missing combination shift'


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer('seed', seed, 0)
    nt = _integer('num_train', num_train, 2, 10)
    ne = _integer('num_test', num_test, 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    configurations = [(a, b, r) for a in (0, 1) for b in (False, True) for r in (False, True)]
    first = configurations[int(rng.integers(8))]
    second = (1-first[0], not first[1], not first[2])
    rest = [c for c in configurations if c not in (first, second)]
    held = [rest[int(i)] for i in rng.choice(len(rest), 2, replace=False)]
    train_pool = [c for c in configurations if c not in held]
    configs = [first, second] + [train_pool[int(rng.integers(len(train_pool)))] for _ in range(nt-2)]
    tests = [held[int(rng.integers(2))]]
    for _ in range(ne-1):
        pool = held if rng.random() < .65 else configs
        tests.append(pool[int(rng.integers(len(pool)))])
    first_lanes = int(rng.integers(4, 7))
    second_lanes = int(rng.choice([n for n in range(3, 7) if n != first_lanes]))
    shared_preference = bool(rng.random() < .3)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, settings in (('train', configs), ('test', tests)):
        for i, configuration in enumerate(settings):
            witness = split == 'train' and i < 2
            lanes = (first_lanes if i == 0 else second_lanes) if witness else None
            for _ in range(64):
                pair = _example(rng, configuration, witness=witness, lanes=lanes,
                                duplicate=witness and i == 0, shared=shared_preference)
                key = _grid_key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise AssertionError('unique input construction exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def puzzle_features(puzzle):
    """Measurements of actual visible arrays, for bounded diversity audits."""
    rows = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            d = _parse(pair['input'])
            records = d['records']
            rows.append(dict(split=split, configuration=_configuration(d),
                             roles=tuple(d[k] for k in ('background', 'origin', 'fast', 'slow')),
                             input_shape=tuple(np.asarray(pair['input']).shape),
                             output_shape=tuple(np.asarray(pair['output']).shape),
                             lengths=tuple(r['length'] for r in records),
                             populations=tuple(r['copies'] for r in records),
                             ranks=tuple(r['index'] for r in records),
                             anchors=tuple(r['anchor'] for r in records),
                             spans=tuple(r['last']-r['anchor']+1 for r in records),
                             edits=tuple(int(np.count_nonzero(r['master'] != r['exception'])) for r in records),
                             balanced=sum(_histogram_same(r) for r in records),
                             repeats=len(records)-len({_grid_key(r['exception']) for r in records}),
                             lanes=len(records),
                             mixed_parity=len({r['length'] % 2 for r in records}) == 2,
                             masters=tuple(_grid_key(r['master']) for r in records)))
    return rows


def _puzzle_key(puzzle):
    return tuple(tuple((_grid_key(np.asarray(p['input'])), _grid_key(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def self_test():
    # Direct family members beyond the witness sampler: maximal fast packing,
    # mixed parity, terminal anchors, two-row faults, and complete recoloring.
    rng = np.random.default_rng(501)
    edge_records = []
    for length, anchor, copies in ((2,0,9), (3,1,5), (8,7,3), (8,0,3)):
        rr, cc = np.indices((2, length))
        master = np.where((rr+cc) % 2 == 0, 3, 4).astype(np.int8)
        exception = master.copy()
        exception[:, anchor] = 7-exception[:, anchor]
        edge_records.append(dict(master=master, exception=exception, length=length,
                                 anchor=anchor, last=anchor, copies=copies, index=copies//2))
    expected = np.full((8, 15), 0, dtype=np.int8)
    for i, r in enumerate(edge_records):
        start = 7-r['anchor']
        expected[2*i:2*i+2, start:start+r['length']] = r['exception']
    for axis in (0, 1):
        for bottom in (False, True):
            for right in (False, True):
                configuration = (axis, bottom, right)
                pair = _paint(rng, edge_records, configuration, (0,5,6,7))
                assert np.array_equal(solve(pair['input']), expected)
                single = dict(edge_records[0], exception=7-edge_records[0]['master'], last=1)
                pair = _paint(rng, [single], configuration, (0,5,6,7))
                assert np.array_equal(solve(pair['input']), single['exception'])
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    rows = [r for p in puzzles for r in puzzle_features(p)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len({r['input_shape'] for r in rows}) >= 50
    assert len({r['output_shape'] for r in rows}) >= 35
    assert len({(r['configuration'], r['lengths'], r['populations'], r['anchors']) for r in rows}) >= 180
    assert len({r['roles'] for r in rows}) >= 100
    assert len({m for r in rows for m in r['masters']}) >= 150
    assert {r['configuration'] for r in rows} == {(a,b,c) for a in (0,1) for b in (False,True) for c in (False,True)}
    assert {r['roles'][0] for r in rows} == set(range(10))
    assert {r['roles'][1] for r in rows} == set(range(10))
    assert {v for r in rows for v in r['lengths']} == set(range(2, 9))
    assert {v for r in rows for v in r['populations']} == set(range(3, 10))
    assert any(r['lanes'] == 1 for r in rows) and any(r['lanes'] == 9 for r in rows)
    assert sum(r['mixed_parity'] for r in rows) >= 60
    assert sum(r['balanced'] > 0 for r in rows) >= 85
    assert sum(r['repeats'] > 0 for r in rows) >= 42
    for p in puzzles:
        validate_generated_puzzle(p)
        for pair in p['train'] + p['test']:
            # Entire-input rotation keeps reading and output coordinates intact.
            for k in range(4):
                assert np.array_equal(solve(np.rot90(pair['input'], k)), pair['output'])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+10*nt+ne, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    assert _puzzle_key(generate_puzzle(91827, 10, 4)) == _puzzle_key(generate_puzzle(91827, 10, 4))
    for nt, ne in ((1,1), (11,1), (4,0), (5,5), (4,5), (2,2), (3,4), (True,1), (4,False), (4,1.5), (None,1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for seed in (-1, 1.5, True):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal seed accepted')
    # Color permutations are local roles, not semantic color dictionaries.
    lookup = np.array([8, 4, 0, 7, 2, 9, 5, 1, 6, 3], dtype=np.int8)
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        assert np.array_equal(solve(lookup[pair['input']]), lookup[pair['output']])
    # Reject altered output, lost exception, damaged full rectangle and key.
    base = puzzles[0]
    for defect in ('output', 'missing', 'hole', 'key', 'duplicate'):
        bad = copy.deepcopy(base)
        pair = bad['train'][0]
        if defect == 'output':
            pair['output'][0, 0] = (int(pair['output'][0, 0])+1) % 10
        elif defect == 'duplicate':
            bad['test'][0] = copy.deepcopy(bad['train'][0])
        elif defect == 'key':
            pair['input'][0, :] = pair['input'][1, :]
        else:
            d = _parse(pair['input'])
            r = d['records'][0]
            r0, r1, c0, c1 = r['boxes'][r['index']]
            if defect == 'missing':
                rotation = (3 if d['bottom'] else 1) if d['axis'] else (2 if d['right'] else 0)
                pair['input'][r0:r1, c0:c1] = np.rot90(r['master'], -rotation)
            else:
                pair['input'][r0, c0] = d['background']
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('accepted mutation: '+defect)


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, all legal counts, registration and mutation checks')
