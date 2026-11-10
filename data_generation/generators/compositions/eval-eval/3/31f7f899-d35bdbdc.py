"""Relay sorter: color-addressed queues carry ranked bar spans and beads.

Only NumPy and the standard library are used. The solver parses visible cells;
it does not call generation, read fixtures, or retain any puzzle state.
"""
from __future__ import annotations

from collections import Counter
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 by 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(zip(*np.where(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], {p}
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _parse(input_grid):
    grid = _grid(input_grid)
    colors, counts = np.unique(grid, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1, "background must be unique mode"
    bg = int(colors[np.argmax(counts)])
    rows = np.flatnonzero(np.all(grid != bg, axis=1))
    cols = np.flatnonzero(np.all(grid != bg, axis=0))
    assert len(rows) + len(cols) == 1, "one full carrier axis is required"
    vertical = bool(len(cols))
    canonical = grid.T.copy() if vertical else grid.copy()
    row = int(cols[0] if vertical else rows[0])
    assert 0 < row < canonical.shape[0]-1
    # Bars have background margins along the carrier, so its end pixel names it.
    carrier = int(canonical[row, 0])
    assert carrier != bg and int(canonical[row, -1]) == carrier
    assert not np.any(canonical[np.arange(len(canonical)) != row] == carrier)
    records = []
    for cells in _components((canonical != bg) & (canonical != carrier)):
        top = min(r for r, c in cells)
        bottom = max(r for r, c in cells)
        left = min(c for r, c in cells)
        right = max(c for r, c in cells)
        width = right-left+1
        assert width in (1, 3, 5), "slot width must identify a central bead"
        assert 0 < left <= right < canonical.shape[1]-1
        assert 0 < top <= row <= bottom < canonical.shape[0]-1
        assert len(cells) == width*(bottom-top+1) >= 3
        bead = (row, (left+right)//2)
        bodies = {int(canonical[p]) for p in cells if p != bead}
        assert len(bodies) == 1, "bar must be solid apart from its bead"
        body = bodies.pop()
        value = int(canonical[bead])
        assert body != value, "input relations have distinct body and bead"
        assert body not in (bg, carrier) and value not in (bg, carrier)
        records.append(dict(x=left, width=width, body=body, value=value,
                            lo=top-row, hi=bottom-row, bead=bead))
    records.sort(key=lambda b: b['x'])
    assert 2 <= len(records) <= 7
    assert all(a['x']+a['width'] < b['x'] for a, b in zip(records, records[1:]))
    rebuilt = _paint(canonical.shape, row, bg, carrier, records)
    assert np.array_equal(canonical, rebuilt), "unexplained foreground cells"
    tied = {}
    for b in records:
        key = b['body'], b['hi']-b['lo']+1
        payload = b['lo'], b['hi'], b['value']
        assert tied.setdefault(key, payload) == payload, "ambiguous equal-length donors"
    return dict(grid=canonical, vertical=vertical, row=row, bg=bg,
                carrier=carrier, bars=records)


def _paint(shape, row, bg, carrier, bars):
    grid = np.full(shape, bg, dtype=np.int8)
    grid[row, :] = carrier
    for b in bars:
        grid[row+b['lo']:row+b['hi']+1, b['x']:b['x']+b['width']] = b['body']
        grid[row, b['x']+b['width']//2] = b['value']
    return grid


def _route(a, mode='length'):
    """Return donor indices, with None for an exhausted or absent queue."""
    bars = a['bars']
    if mode == 'global':
        return sorted(range(len(bars)), key=lambda i: bars[i]['hi']-bars[i]['lo'])
    supply = {}
    for i, b in enumerate(bars):
        supply.setdefault(b['body'], []).append(i)
    for donors in supply.values():
        if mode == 'length':
            donors.sort(key=lambda i: bars[i]['hi']-bars[i]['lo'])
        elif mode == 'area':
            donors.sort(key=lambda i: bars[i]['width']*(bars[i]['hi']-bars[i]['lo']+1))
        elif mode == 'reverse':
            donors.reverse()
        else:
            assert mode == 'source'
    demand = Counter()
    plan = []
    for b in bars:
        color = b['value']
        rank = demand[color]
        demand[color] += 1
        donors = supply.get(color, [])
        plan.append(donors[rank] if rank < len(donors) else None)
    return plan


def _render(a, plan, update_donors=False, surviving_donors=False):
    result = []
    bars = a['bars']
    for i, j in enumerate(plan):
        if j is None or (surviving_donors and plan[j] is None):
            continue
        dest, source = bars[i], bars[j]
        value = source['value']
        if update_donors and plan[j] is not None:
            value = bars[plan[j]]['value']
        result.append(dict(x=dest['x'], width=dest['width'], body=dest['body'],
                           lo=source['lo'], hi=source['hi'], value=value))
    out = _paint(a['grid'].shape, a['row'], a['bg'], a['carrier'], result)
    return out.T.copy() if a['vertical'] else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Compose X->Y with the kth shortest Y->Z donor, reading inputs once."""
    a = _parse(input_grid)
    return _render(a, _route(a))


def _check_pair(pair):
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape
    a = _parse(inp)
    canonical = out.T if a['vertical'] else out
    bars = a['bars']
    covered = np.zeros(canonical.shape, dtype=bool)
    covered[a['row']] = True
    used = Counter()
    chosen = []
    # Validate rank by counting strictly shorter input donors. This does not
    # call the solver's routing or rendering helpers.
    for b in bars:
        covered[:, b['x']:b['x']+b['width']] = True
        donors = [d for d in bars if d['body'] == b['value']]
        k = used[b['value']]
        used[b['value']] += 1
        expected = np.full((len(canonical), b['width']), a['bg'], dtype=np.int8)
        expected[a['row']] = a['carrier']
        if k < len(donors):
            for d in donors:
                length = d['hi']-d['lo']+1
                below = sum(e['hi']-e['lo']+1 < length for e in donors)
                equal = sum(e['hi']-e['lo']+1 == length for e in donors)
                if below <= k < below+equal:
                    # Identical payload ties do not affect the answer. Select
                    # distinct physical entries for the diagnostic role counts.
                    equals = [e for e in donors if e['hi']-e['lo']+1 == length]
                    d = equals[k-below]
                    expected[a['row']+d['lo']:a['row']+d['hi']+1] = b['body']
                    expected[a['row'], b['width']//2] = d['value']
                    chosen.append(bars.index(d))
                    break
            else:
                raise AssertionError("rank not found")
        else:
            chosen.append(None)
        assert np.array_equal(canonical[:, b['x']:b['x']+b['width']], expected), \
            "wrong donor rank, span, fixed slot, bead, or deletion"
    assert np.all(canonical[~covered] == a['bg']), "new cells outside slots"
    gaps = np.ones(canonical.shape[1], dtype=bool)
    for b in bars:
        gaps[b['x']:b['x']+b['width']] = False
    assert np.all(canonical[a['row'], gaps] == a['carrier']), "carrier changed"
    return a, chosen


def _features(pair):
    a, plan = _check_pair(pair)
    bars = a['bars']
    out = np.asarray(pair['output'])
    supply, demand = Counter(b['body'] for b in bars), Counter(b['value'] for b in bars)
    missing = sum(v for c, v in demand.items() if c not in supply)
    overflow = sum(max(0, v-supply[c]) for c, v in demand.items() if c in supply)
    queues = {}
    for b in bars:
        queues.setdefault(b['body'], []).append(b)
    moved = [i for i, j in enumerate(plan) if j is not None
             and (bars[i]['lo'], bars[i]['hi']) != (bars[j]['lo'], bars[j]['hi'])]
    return dict(shape=tuple(int(n) for n in out.shape), vertical=a['vertical'],
                background=a['bg'], carrier=a['carrier'], count=len(bars),
                parallel_size=a['grid'].shape[1],
                max_output_length=max((bars[j]['hi']-bars[j]['lo']+1
                                       for j in plan if j is not None), default=0),
                survivors=sum(j is not None for j in plan), missing=missing, overflow=overflow,
                mixed_widths=len({b['width'] for b in bars}) > 1,
                widths=tuple(b['width'] for b in bars),
                lengths=tuple(b['hi']-b['lo']+1 for b in bars),
                ties=any(len({b['hi']-b['lo'] for b in q}) < len(q) for q in queues.values()),
                conflicting_donors=any(len({b['value'] for b in q}) > 1 for q in queues.values()),
                moved=len(moved),
                moved_upper=any(bars[plan[i]]['lo']+bars[plan[i]]['hi'] < 0 for i in moved),
                moved_lower=any(bars[plan[i]]['lo']+bars[plan[i]]['hi'] > 0 for i in moved),
                deleted_donors=sum(j is not None and plan[j] is None for j in plan),
                updated_donors=sum(j is not None and plan[j] is not None
                                   and bars[plan[j]]['value'] != bars[j]['value'] for j in plan),
                solid=sum(j is not None and bars[j]['value'] == bars[i]['body']
                          for i, j in enumerate(plan)),
                no_rank_differs=not np.array_equal(out, _render(a, _route(a, 'source'))),
                no_relation_differs=not np.array_equal(out, _render(a, _route(a, 'global'))),
                area_differs=not np.array_equal(out, _render(a, _route(a, 'area'))),
                reverse_differs=not np.array_equal(out, _render(a, _route(a, 'reverse'))),
                updated_read_differs=not np.array_equal(out, _render(a, plan, update_donors=True)),
                deleted_read_differs=not np.array_equal(out, _render(a, plan, surviving_donors=True)),
                layout=(a['vertical'], a['grid'].shape, a['row'],
                        tuple((b['x'], b['width'], b['lo'], b['hi']) for b in bars)),
                relation=tuple((b['body'], b['value']) for b in bars))


def _span(rng, length, policy, bias=None):
    if length == 1:
        return 0, 0
    if policy == 'centered':
        lo = -(length//2)
    elif bias is not None:
        lo = 1-length if bias == 'upper' else 0
    else:
        # A shared feasible envelope permits transfer between every slot while
        # leaving background margins, even when a test uses length sixteen.
        low = max(1-length, -12)
        high = min(0, 13-length)
        lo = int(rng.integers(low, high+1))
    return lo, lo+length-1


def _sample_relations(rng, mode):
    if mode == 'rank':
        rel = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        n = int(rng.integers(5, 8))
        if n == 5:
            rel.pop()
        elif n == 7:
            rel.append((3, 1))
    elif mode == 'shortage':
        rel = [(0, 1), (0, 3), (1, 0), (1, 0), (2, 1), (2, 0)]
        if rng.random() < .5:
            rel.append((4, 1))
    else:
        n = int(rng.integers(2, 8))
        k = int(rng.integers(2, min(5, n)+1))
        bodies = [int(x) for x in rng.integers(k, size=n)]
        # Force at least two different bodies, without imposing balanced counts.
        bodies[0], bodies[1] = 0, 1
        rel = [(b, int(rng.choice([x for x in range(k+1) if x != b]))) for b in bodies]
    return [rel[int(i)] for i in rng.permutation(len(rel))]


def _make_pair(rng, config, mode, vertical, split):
    rel = _sample_relations(rng, mode)
    n = len(rel)
    palette = (list(map(int, rng.permutation(10))) if config['varied_roles'] else
               list(config['roles']) + list(map(int, rng.permutation(
                   [c for c in range(10) if c not in config['roles']]))))
    bg, carrier = palette[:2]
    widths = list(map(int, rng.choice([1, 3, 5], n, p=[.5, .4, .1])))
    # Larger tests shift the numerical scale, with all geometry conventions
    # already present in the two required demonstrations.
    cap = 16 if split == 'test' and config['shift'] == 'length' else 12
    lengths = list(map(int, rng.choice(np.arange(1, cap+1), n, replace=False)))
    if split == 'test' and config['shift'] == 'length':
        supply = Counter(b for b, v in rel)
        demand = Counter(v for b, v in rel)
        # Every donor of these bodies is used, so the larger span must actually
        # influence the answer rather than appearing only on an unused token.
        eligible = [i for i, (b, v) in enumerate(rel) if demand[b] >= supply[b]]
        if eligible:
            lengths[int(rng.choice(eligible))] = int(rng.integers(13, 17))
    if mode == 'rank':
        ai = [i for i, (b, v) in enumerate(rel) if b == 0]
        bi = [i for i, (b, v) in enumerate(rel) if b == 1]
        lengths[ai[0]], lengths[ai[1]] = 9, 5
        widths[ai[0]], widths[ai[1]] = 1, 3
        lengths[bi[0]], lengths[bi[1]] = 3, 7
        # Remaining bodies occur once or twice; remove accidentally duplicated
        # lengths within a body without constraining ties across different bodies.
    for body in sorted({b for b, v in rel}):
        indices = [i for i, (b, v) in enumerate(rel) if b == body]
        seen = set()
        for i in indices:
            if lengths[i] in seen:
                lengths[i] = next(x for x in range(1, cap+1) if x not in seen)
            seen.add(lengths[i])
    spans = [_span(rng, l, config['spans']) for l in lengths]
    if mode == 'rank' and config['spans'] == 'free':
        ai = [i for i, (b, v) in enumerate(rel) if b == 0]
        spans[ai[0]] = _span(rng, lengths[ai[0]], 'free', 'upper')
        spans[ai[1]] = _span(rng, lengths[ai[1]], 'free', 'lower')
    # Equal-length donors can tie only with an identical bead and interval.
    # The random subfamily occasionally demonstrates such indistinguishable ties.
    if mode == 'random' and rng.random() < .3:
        compatible = [(i, j) for i in range(n) for j in range(i+1, n) if rel[i] == rel[j]]
        if compatible:
            i, j = compatible[int(rng.integers(len(compatible)))]
            lengths[j], spans[j] = lengths[i], spans[i]
    for i in range(n):
        if lengths[i] < 3 and widths[i] == 1:
            widths[i] = 3
    locked = {i: 3 for i in range(n) if lengths[i] < 3}
    if mode == 'rank':
        ai = [i for i, (b, v) in enumerate(rel) if b == 0]
        locked[ai[0]], locked[ai[1]] = 1, 3
    # Gaps separate entire two-color bars; the bead is never mistaken for a slot.
    width_limit = 26 if split == 'train' and config['shift'] == 'spacing' else 30
    while sum(widths)+n+1 > width_limit:
        candidates = [i for i in range(n) if widths[i] > locked.get(i, 1)]
        assert candidates
        i = max(candidates, key=lambda i: widths[i])
        widths[i] -= 2
    min_width = sum(widths)+n+1
    min_lo, max_hi = min(s[0] for s in spans), max(s[1] for s in spans)
    min_height = max(7, max_hi-min_lo+5)
    assert min_height <= 30
    if split == 'test' and config['shift'] == 'spacing':
        width = int(rng.integers(max(min_width, 28), 31))
    else:
        width = int(rng.integers(min_width, min(width_limit, min_width+8)+1))
    height = int(rng.integers(min_height, min(30, min_height+6)+1))
    if rng.random() < .1 and max(height, width) <= width_limit:
        height = width = max(height, width)
    row = int(rng.integers(1-min_lo, height-1-max_hi))
    gaps = [1]* (n+1)
    extra = width-min_width
    for i, amount in enumerate(rng.multinomial(extra, [1/(n+1)]*(n+1))):
        gaps[i] += int(amount)
    bars, x = [], gaps[0]
    for i, ((b, v), (lo, hi)) in enumerate(zip(rel, spans)):
        bars.append(dict(x=x, width=widths[i], body=palette[b+2], value=palette[v+2], lo=lo, hi=hi))
        x += widths[i]+gaps[i+1]
    inp = _paint((height, width), row, bg, carrier, bars)
    # Construct labels from sampled visible relations, rather than calling solve.
    recipients = {}
    routed = []
    for dest in bars:
        key = dest['value']
        k = recipients.get(key, 0)
        recipients[key] = k+1
        sources = [b for b in bars if b['body'] == key]
        order = np.argsort([b['hi']-b['lo']+1 for b in sources], kind='stable')
        if k < len(sources):
            source = sources[int(order[k])]
            routed.append(dict(x=dest['x'], width=dest['width'], body=dest['body'],
                               lo=source['lo'], hi=source['hi'], value=source['value']))
    out = _paint((height, width), row, bg, carrier, routed)
    if vertical:
        inp, out = inp.T.copy(), out.T.copy()
    return dict(input=inp, output=out)


def _key(grid):
    grid = np.asarray(grid)
    return grid.shape, grid.astype(np.int8).tobytes()


def _puzzle_key(p):
    return tuple(tuple((_key(q['input']), _key(q['output'])) for q in p[s])
                 for s in ('train', 'test'))


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count('num_train', num_train, 2, 10), _count('num_test', num_test, 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    config = dict(roles=tuple(map(int, rng.choice(10, 2, replace=False))),
                  varied_roles=bool(rng.random() < .7),
                  spans=str(rng.choice(['centered', 'free'], p=[.3, .7])),
                  shift=str(rng.choice(['length', 'spacing'])))
    axis = str(rng.choice(['horizontal', 'vertical', 'mixed'], p=[.35, .25, .4]))
    train_axes = ([False, True]+list(map(bool, rng.integers(2, size=nt-2)))
                  if axis == 'mixed' else [axis == 'vertical']*nt)
    test_axes = (list(map(bool, rng.integers(2, size=ne)))
                 if axis == 'mixed' else [axis == 'vertical']*ne)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, axes in (('train', train_axes), ('test', test_axes)):
        for i, vertical in enumerate(axes):
            mode = ['rank', 'shortage'][i] if split == 'train' and i < 2 else 'random'
            for attempt in range(160):
                pair = _make_pair(rng, config, mode, vertical, split)
                try:
                    f = _features(pair)
                    assert f['survivors'] > 0 and f['moved'] > 0
                    assert _key(pair['input']) not in seen
                    if mode == 'rank':
                        assert f['no_rank_differs'] and f['area_differs'] and f['reverse_differs']
                        assert f['updated_read_differs'] and f['conflicting_donors']
                        if config['spans'] == 'free':
                            assert f['moved_upper'] and f['moved_lower']
                    if mode == 'shortage':
                        assert f['missing'] and f['overflow'] and f['deleted_read_differs']
                        assert f['no_rank_differs']
                    if split == 'test' and config['shift'] == 'length':
                        assert f['max_output_length'] > 12
                    if config['varied_roles'] and split == 'train' and i == 1:
                        first = _features(puzzle['train'][0])
                        assert (f['background'], f['carrier']) != (first['background'], first['carrier'])
                except AssertionError:
                    continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError(f'could not construct {mode} example in bounded attempts')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, measured = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            _grid(pair['input'])
            _grid(pair['output'])
            assert _key(pair['input']) not in seen, 'duplicate input'
            seen.add(_key(pair['input']))
            f = _features(pair)
            assert f['survivors'] > 0 and f['moved'] > 0
            measured[split].append(f)
    train = measured['train']
    for name in ('no_rank_differs', 'no_relation_differs', 'area_differs', 'reverse_differs',
                 'updated_read_differs', 'deleted_read_differs', 'missing', 'overflow',
                 'conflicting_donors', 'mixed_widths'):
        assert any(f[name] for f in train), f'training lacks {name} witness'
    axes = {f['vertical'] for f in train}
    assert all(f['vertical'] in axes for f in measured['test'])
    max_train_length = max(max(f['lengths']) for f in train)
    max_train_parallel = max(f['parallel_size'] for f in train)
    assert (all(f['max_output_length'] > max_train_length for f in measured['test'])
            or all(f['parallel_size'] > max_train_parallel for f in measured['test'])), \
        'test lacks the promised length or carrier-scale shift'
    if len({(f['background'], f['carrier']) for f in train}) == 1:
        assert {(f['background'], f['carrier']) for f in measured['test']} <= \
            {(train[0]['background'], train[0]['carrier'])}
    if any(f['moved_upper'] and f['moved_lower'] for f in train):
        return
    # Centered-only puzzles do not require general offsets at test.
    for pair in puzzle['test']:
        a = _parse(pair['input'])
        assert all(b['lo'] == -((b['hi']-b['lo']+1)//2) for b in a['bars'])


def self_test():
    """A bounded 42-seed sweep, semantic mutations, counts, and diversity."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    features = []
    for p in puzzles:
        validate_generated_puzzle(p)
        for split in ('train', 'test'):
            for q in p[split]:
                original = np.asarray(q['input']).copy()
                result = solve(q['input'])
                assert np.array_equal(result, q['output'])
                assert np.array_equal(original, q['input']), 'solve mutated its input'
                features.append(_features(q))
    assert len({f['layout'] for f in features}) >= 180
    assert len({f['shape'] for f in features}) >= 65
    assert {f['count'] for f in features} == set(range(2, 8))
    assert {f['vertical'] for f in features} == {False, True}
    assert {w for f in features for w in f['widths']} == {1, 3, 5}
    assert len({f['survivors'] for f in features}) >= 5
    assert len({(f['background'], f['carrier']) for f in features}) >= 40
    assert any(f['ties'] for f in features)
    assert any(1 in f['lengths'] for f in features)
    assert any(f['solid'] for f in features)
    for name in ('missing', 'overflow', 'conflicting_donors', 'deleted_donors',
                 'updated_donors', 'moved_upper', 'moved_lower'):
        assert sum(bool(f[name]) for f in features) >= 20, name
    # Every legal explicit count combination and representative invalid types.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+10*nt+ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.5, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Actual incorrect answers must fail the geometric/rank invariant checker.
    rejected = set()
    for q in puzzles[0]['train']:
        a = _parse(q['input'])
        for mode in ('source', 'area', 'reverse', 'global'):
            bad = _render(a, _route(a, mode))
            if np.array_equal(bad, q['output']):
                continue
            try:
                _check_pair(dict(input=q['input'], output=bad))
            except AssertionError:
                rejected.add(mode)
            else:
                raise AssertionError(f'{mode} mutation accepted')
    assert rejected == {'source', 'area', 'reverse', 'global'}
    return dict(seeds=42, pairs=len(features), layouts=len({f['layout'] for f in features}),
                shapes=len({f['shape'] for f in features}),
                counts=sorted({f['count'] for f in features}),
                survivor_counts=sorted({f['survivors'] for f in features}),
                role_pairs=len({(f['background'], f['carrier']) for f in features}),
                ties=sum(f['ties'] for f in features),
                missing=sum(bool(f['missing']) for f in features),
                overflow=sum(bool(f['overflow']) for f in features))
