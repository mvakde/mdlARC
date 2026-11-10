"""Remote mirror partners: ordinal square pairs share reflected satellite support.

Only standard-library modules and NumPy are used. Generation never reads parents,
and solve never uses random state, demonstration labels, or construction metadata.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a


def _key(a):
    return tuple(a.shape), a.astype(np.uint8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        todo, comp = [p], [p]
        while todo:
            r, c = todo.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
                    comp.append(q)
        result.append(comp)
    return result


def _distance(p, obj):
    r, c = p
    t, l, s = obj['top'], obj['left'], obj['size']
    return max(t-r, r-(t+s-1), l-c, c-(l+s-1), 0)


def _analyze(value):
    """Parse visible objects and unique satellite ownership, input only."""
    a = _grid(value)
    counts = np.bincount(a.ravel().astype(int), minlength=10)
    bg = int(counts.argmax())
    assert counts[bg] > a.size / 2, 'background is a unique majority'
    objects, dots = [], {}
    for comp in _components(a != bg):
        if len(comp) == 1:
            p = comp[0]
            dots[p] = int(a[p])
            continue
        rr, cc = zip(*comp)
        top, left = min(rr), min(cc)
        s = max(rr)-top+1
        assert 3 <= s <= 5 and max(cc)-left+1 == s
        tile = a[top:top+s, left:left+s]
        color = int(tile[0, 0])
        assert color != bg
        rim = np.zeros((s, s), dtype=bool)
        rim[0, :] = rim[-1, :] = True
        rim[:, 0] = rim[:, -1] = True
        assert np.all(tile[rim] == color)
        if np.all(tile == color):
            kind = 1  # solid
        else:
            assert np.all(tile[~rim] == bg)
            kind = 0  # hollow
        assert len(comp) == int(np.sum(tile == color))
        objects.append(dict(top=top, left=left, size=s, color=color, kind=kind))
    objects.sort(key=lambda o: (o['top'], o['left']))
    assert 2 <= len(objects) <= 9
    assert len({o['size'] for o in objects}) == 1
    assert len({o['color'] for o in objects}) == 1
    marker = objects[0]['color']
    assert all(color != marker for color in dots.values())
    queues = [[i for i, o in enumerate(objects) if o['kind'] == k] for k in (0, 1)]
    assert all(queues)
    for queue in queues:
        assert len({objects[i]['top'] for i in queue}) == len(queue), 'same-class top tie'
    owners = {}
    for p in dots:
        ds = [_distance(p, o) for o in objects]
        distance = min(ds)
        assert 2 <= distance <= 3 and ds.count(distance) == 1, 'unclear ownership'
        owners[p] = ds.index(distance)
        r, c = p
        assert all((r+dr, c+dc) not in dots for dr in (-1, 0, 1)
                   for dc in (-1, 0, 1) if dr or dc), 'satellites must be isolated'
    return dict(grid=a, background=bg, marker=marker, objects=objects,
                dots=dots, owners=owners, queues=queues)


def _rank_partners(a):
    partners = {}
    for i, j in zip(*a['queues']):
        partners[i], partners[j] = j, i
    return partners


def _filter(a, partners, *, local=False, colorblind=False):
    out = a['grid'].copy()
    objects, dots, owners = a['objects'], a['dots'], a['owners']
    for p, color in dots.items():
        i = owners[p]
        j = i if local else partners.get(i)
        supported = False
        if j is not None:
            src, dst = objects[i], objects[j]
            q = (dst['top'] + p[0] - src['top'],
                 dst['left'] + src['size'] - 1 - (p[1] - src['left']))
            supported = (owners.get(q) == j and q in dots and
                         (colorblind or dots[q] == color))
        if not supported:
            out[p] = a['background']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Retain only same-colored dots supported by the rank-matched remote mirror."""
    a = _analyze(input_grid)
    return _filter(a, _rank_partners(a))


def _alternative(a, name):
    """Concrete competing rules used to enforce informative demonstrations."""
    objects, qs = a['objects'], a['queues']
    if name == 'local':
        return _filter(a, {}, local=True)
    if name == 'colorblind':
        return _filter(a, _rank_partners(a), colorblind=True)
    if name == 'keep_all':
        return a['grid'].copy()
    if name == 'delete_all':
        return _filter(a, {})
    if name in ('any_opposite', 'best_support'):
        # These omit the ordinal concept, using payload agreement to find mates.
        out, choices = a['grid'].copy(), {}
        for i, src in enumerate(objects):
            scores = []
            for j, dst in enumerate(objects):
                if src['kind'] == dst['kind']:
                    continue
                supported = set()
                for p, color in a['dots'].items():
                    if a['owners'][p] != i:
                        continue
                    q = (dst['top']+p[0]-src['top'],
                         dst['left']+src['size']-1-(p[1]-src['left']))
                    if a['owners'].get(q) == j and a['dots'].get(q) == color:
                        supported.add(p)
                scores.append((len(supported), -j, j))
                if name == 'any_opposite':
                    for p in supported:
                        choices[p] = True
            if name == 'best_support':
                choices[i] = max(scores)[2]
        if name == 'best_support':
            return _filter(a, choices)
        for p in a['dots']:
            if p not in choices:
                out[p] = a['background']
        return out
    if name in ('horizontal', 'horizontal_reverse', 'reverse'):
        key = ((lambda i: objects[i]['top']) if name == 'reverse' else
               (lambda i: (objects[i]['left'], objects[i]['top'])))
        lists = [sorted(q, key=key, reverse=name != 'horizontal') for q in qs]
        pairs = list(zip(*lists))
    elif name == 'nearest':
        available = list(qs[1])
        pairs = []
        for i in qs[0]:
            if not available:
                break
            j = min(available, key=lambda j: (abs(objects[i]['top']-objects[j]['top']) +
                                             abs(objects[i]['left']-objects[j]['left']),
                                             objects[j]['top'], objects[j]['left']))
            pairs.append((i, j))
            available.remove(j)
    elif name == 'greedy':
        rows = [[None, None]]
        for i, obj in enumerate(objects):
            k = obj['kind']
            if rows[-1][k] is not None:
                rows.append([None, None])
            rows[-1][k] = i
        pairs = [(i, j) for i, j in rows if i is not None and j is not None]
    else:
        raise ValueError(name)
    partners = {i: j for i, j in pairs}
    partners.update({j: i for i, j in pairs})
    return _filter(a, partners)


_ALTERNATIVES = ('horizontal', 'horizontal_reverse', 'reverse', 'nearest',
                 'greedy', 'any_opposite', 'best_support', 'local',
                 'colorblind', 'keep_all', 'delete_all')


def _partition(rng, total, parts, capacity):
    amounts = [1] * parts
    for _ in range(total-parts):
        choices = [i for i, n in enumerate(amounts) if n < capacity]
        amounts[int(rng.choice(choices))] += 1
    return amounts


def _capacity(s, radius):
    e = s+2*radius
    cap = 31//(e+1)
    return max(n for n in range(2, 10)
               if ((n+cap-1)//cap)*e+n-1 <= 30 and n <= cap*cap)


def _layout(rng, s, radius, n, fallback=False):
    """Invisible disjoint neighborhoods; free relocation removes shelf alignment."""
    e = s+2*radius
    if fallback:
        # Five staggered sites, genuinely non-nearest and non-horizontal matching.
        sites = [(0, 2*(e+1)), (1, 0), (2, e+1), (e+3, e+1), (e+4, 0)]
        return (2*e+4, 3*e+2), [(r+radius, c+radius) for r, c in sites]
    cap = 31//(e+1)
    bands = [b for b in range((n+cap-1)//cap, min(n, cap)+1) if b*e+n-1 <= 30]
    b = int(rng.choice(bands))
    amounts = _partition(rng, n, b, cap)
    min_h, min_w = b*e+n-1, max(amounts)*(e+1)-1
    h = int(rng.integers(min_h, 31))
    w = int(rng.integers(min_w, 31))
    gaps = rng.multinomial(h-min_h, np.full(b+1, 1/(b+1)))
    sites, top = [], int(gaps[0])
    for band, count in enumerate(amounts):
        xgaps = rng.multinomial(w-(count*(e+1)-1), np.full(count+1, 1/(count+1)))
        left = int(xgaps[0])
        for column, offset in enumerate(rng.permutation(count)):
            sites.append((top+int(offset), left))
            left += e+1+int(xgaps[column+1])
        top += e+count+int(gaps[band+1])
    rr, cc = np.indices((h-e+1, w-e+1))
    for _ in range(2):
        for i in rng.permutation(n):
            allowed = np.ones(rr.shape, dtype=bool)
            for j, (r, c) in enumerate(sites):
                if i != j:
                    allowed &= ((np.abs(rr-r) > e) | (np.abs(cc-c) > e)) & (rr != r)
            choices = np.argwhere(allowed)
            r, c = choices[int(rng.integers(len(choices)))]
            sites[int(i)] = int(r), int(c)
    return (h, w), sorted((r+radius, c+radius) for r, c in sites)


def _reflect(p, s):
    return p[0], s-1-p[1]


def _ring(s, radius):
    return [(r, c) for r in range(-radius, s+radius) for c in range(-radius, s+radius)
            if 2 <= max(-r, r-s+1, -c, c-s+1, 0) <= radius]


def _spread(rng, candidates, used, amount):
    result = []
    for index in rng.permutation(len(candidates)):
        p = candidates[int(index)]
        if all(max(abs(p[0]-q[0]), abs(p[1]-q[1])) >= 2 for q in used+result):
            result.append(p)
            if len(result) >= amount:
                break
    return result


def _clouds(rng, kinds, s, radius, colors, witness):
    """Construct input clouds and their intended supported subsets in local space."""
    n = len(kinds)
    raw, clean = [dict() for _ in kinds], [dict() for _ in kinds]
    queues = [[i for i, k in enumerate(kinds) if k == kind] for kind in (0, 1)]
    paired = set()
    candidates = _ring(s, radius)
    for rank, (i, j) in enumerate(zip(*queues)):
        paired.update((i, j))
        p0 = (-radius, -radius)
        p1, p2 = (s+radius-1, -radius), (s+radius-1, s+radius-1)
        p3 = ((s-1)//2, -radius)
        # Each pair has genuine support; the first has explicit contradictory local evidence.
        color = colors[rank % len(colors)]
        raw[i][p0] = clean[i][p0] = color
        q0 = _reflect(p0, s)
        raw[j][q0] = clean[j][q0] = color
        used = [p0, q0]
        if rank == 0:
            raw[i][p1] = raw[i][p2] = colors[0]
            raw[i][p3] = colors[0]
            if len(colors) >= 2:
                raw[j][_reflect(p3, s)] = colors[1]
            used += [p1, p2, p3]
        for p in _spread(rng, candidates, used, int(rng.integers(2, 7))):
            q = _reflect(p, s)
            state = int(rng.choice(4, p=(.50, .20, .20, .10)))
            color = int(rng.choice(colors))
            if state == 0:
                raw[i][p] = clean[i][p] = color
                raw[j][q] = clean[j][q] = color
            elif state == 1:
                raw[i][p] = color
            elif state == 2:
                raw[j][q] = color
            else:
                raw[i][p] = color
                if len(colors) > 1:
                    raw[j][q] = int(rng.choice([c for c in colors if c != color]))
        # Independent flips commute with horizontal reflection, moving the evidence motifs.
        flip_r, flip_c = bool(rng.integers(2)), bool(rng.integers(2))
        def transform(p):
            return (s-1-p[0] if flip_r else p[0], s-1-p[1] if flip_c else p[1])
        for index in (i, j):
            raw[index] = {transform(p): c for p, c in raw[index].items()}
            clean[index] = {transform(p): c for p, c in clean[index].items()}
    for i in set(range(n))-paired:
        for p in _spread(rng, candidates, [], int(rng.integers(3, 8))):
            raw[i][p] = int(rng.choice(colors))
        if witness:
            # False friend: this unpaired cloud is a better visual mirror of
            # the opposite first beacon than that beacon's true noisy partner.
            # It still has no ordinal partner, so all its dots must disappear.
            other = queues[1-kinds[i]][0]
            raw[i] = {_reflect(p, s): color for p, color in raw[other].items()}
    return raw, clean


def _make_pair(rng, s, radius, counts, bg, witness=False, fallback=False):
    n = sum(counts)
    shape, positions = _layout(rng, s, radius, n, fallback)
    if fallback:
        kinds = [0, 0, 1, 1, 1] if counts == (2, 3) else [1, 1, 0, 0, 0]
    else:
        kinds = list(map(int, rng.permutation([0]*counts[0]+[1]*counts[1])))
    palette = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
    marker = palette[0]
    colors = palette[1:1+int(rng.integers(2 if witness else 1, 5))]
    raw, clean = _clouds(rng, kinds, s, radius, colors, witness)
    inp = np.full(shape, bg, dtype=np.int8)
    out = inp.copy()
    for i, ((top, left), kind) in enumerate(zip(positions, kinds)):
        for grid in (inp, out):
            grid[top:top+s, left:left+s] = marker
            if kind == 0:
                grid[top+1:top+s-1, left+1:left+s-1] = bg
        for grid, clouds in ((inp, raw), (out, clean)):
            for (r, c), color in clouds[i].items():
                grid[top+r, left+c] = color
    return dict(input=inp, output=out)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(['count', 'scale', 'reach']))
    base_size = int(rng.integers(3, 6)) if shift == 'count' else 3
    first_counts = (2, 3) if rng.integers(2) else (3, 2)
    puzzle, seen = dict(train=[], test=[]), set()
    first_bg = int(rng.integers(10))
    for split, amount in (('train', num_train), ('test', num_test)):
        for index in range(amount):
            target = split == 'test'
            s = int(rng.integers(4, 6)) if target and shift == 'scale' else base_size
            radius = 3 if target and shift == 'reach' else 2
            witness = split == 'train' and index < 2
            if witness:
                counts = first_counts if index == 0 else first_counts[::-1]
            else:
                n = (int(rng.integers(6, _capacity(s, radius)+1)) if target and shift == 'count'
                     else int(rng.integers(2, 6)))
                h = int(rng.integers(1, n))
                counts = h, n-h
            bg = (first_bg if split == 'train' and index == 0 else
                  int(rng.choice([c for c in range(10) if c != first_bg]))
                  if split == 'train' and index == 1 else int(rng.integers(10)))
            for attempt in range(64):
                fallback = witness and attempt >= 48
                pair = _make_pair(rng, s, radius, counts, bg, witness, fallback)
                a = _analyze(pair['input'])
                assert np.array_equal(_filter(a, _rank_partners(a)), pair['output'])
                if _key(pair['input']) in seen:
                    continue
                if witness and not all(not np.array_equal(_alternative(a, name), pair['output'])
                                       for name in _ALTERNATIVES):
                    continue
                break
            else:
                raise AssertionError('bounded scene construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _reference_keep(a):
    """Local-coordinate reconstruction for validation, without calling solve/filter."""
    local = [dict() for _ in a['objects']]
    for (r, c), color in a['dots'].items():
        i = a['owners'][(r, c)]
        o = a['objects'][i]
        local[i][(r-o['top'], c-o['left'])] = color
    keep = set()
    for i, j in zip(*a['queues']):
        s = a['objects'][i]['size']
        for p, color in local[i].items():
            q = _reflect(p, s)
            if local[j].get(q) == color:
                for index, point in ((i, p), (j, q)):
                    obj = a['objects'][index]
                    keep.add((obj['top']+point[0], obj['left']+point[1]))
    return keep


def _features(a):
    return dict(shape=tuple(a['grid'].shape), size=a['objects'][0]['size'],
                count=len(a['objects']), counts=tuple(map(len, a['queues'])),
                background=a['background'], marker=a['marker'],
                radius=max(_distance(p, a['objects'][i]) for p, i in a['owners'].items()),
                layout=tuple((o['top'], o['left'], o['kind']) for o in a['objects']),
                edge=any(r in (0, a['grid'].shape[0]-1) or c in (0, a['grid'].shape[1]-1)
                         for r, c in a['dots']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, rejected, unpaired_kinds = set(), set(), set()
    features = dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            a = _analyze(inp)
            features[split].append(_features(a))
            keep = _reference_keep(a)
            assert keep and len(keep) < len(a['dots']), 'retain and erase actual satellites'
            expected = inp.copy()
            for p in set(a['dots'])-keep:
                expected[p] = a['background']
            assert np.array_equal(out, expected), 'wrong reciprocal support or altered beacons'
            assert np.all((out == inp) | (out == a['background'])), 'only erase'
            assert all(out[p] == inp[p] for p in keep)
            if split == 'train':
                for name in _ALTERNATIVES:
                    if not np.array_equal(_alternative(a, name), out):
                        rejected.add(name)
                matched = _rank_partners(a)
                for i, obj in enumerate(a['objects']):
                    if i not in matched and i in a['owners'].values():
                        unpaired_kinds.add(obj['kind'])
    assert rejected == set(_ALTERNATIVES), 'insufficient evidence against shortcuts'
    assert unpaired_kinds == {0, 1}, 'teach both unmatched tails'
    assert len({f['background'] for f in features['train']}) >= 2
    train, test = features['train'], features['test']
    shifted = (all(f['count'] > max(t['count'] for t in train) for f in test) or
               all(f['size'] > max(t['size'] for t in train) for f in test) or
               all(f['radius'] > max(t['radius'] for t in train) for f in test))
    assert shifted, 'missing visible train/test shift'


def self_test():
    """Bounded 42-seed coverage, count handling, and semantic negative controls."""
    records, signatures, shifts = [], set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple(_key(pair['input']) for split in p for pair in p[split]))
        local = {split: [_features(_analyze(pair['input'])) for pair in p[split]] for split in p}
        records.extend(f for fs in local.values() for f in fs)
        for feature in ('count', 'size', 'radius'):
            if all(f[feature] > max(t[feature] for t in local['train']) for f in local['test']):
                shifts.add(feature)
        for split in p:
            for pair in p[split]:
                before = pair['input'].copy()
                assert np.array_equal(solve(before), pair['output'])
                assert np.array_equal(before, pair['input']), 'solve mutated its input'
        if seed < 3:
            again = generate_puzzle(seed)
            assert all(np.array_equal(x[k], y[k]) for split in p
                       for x, y in zip(p[split], again[split]) for k in x)
    assert len(signatures) == 42
    assert shifts == {'count', 'size', 'radius'}
    assert {f['size'] for f in records} == {3, 4, 5}
    assert {f['background'] for f in records} == set(range(10))
    assert {f['marker'] for f in records} == set(range(10))
    assert len({f['shape'] for f in records}) >= 35
    assert len({f['layout'] for f in records}) >= 150
    assert len({f['counts'] for f in records}) >= 15
    assert any(f['count'] >= 8 for f in records)
    assert any(f['edge'] for f in records)
    # The fixed random suite reaches eight. Exercise the feasible nine-beacon
    # boundary directly, including tails in either class, instead of claiming
    # that every maximum must occur accidentally in those 42 puzzles.
    for counts in ((1, 8), (4, 5), (8, 1)):
        pair = _make_pair(np.random.default_rng(sum(counts)+counts[0]), 3, 2, counts, 0)
        assert len(_analyze(pair['input'])['objects']) == 9
        assert np.array_equal(solve(pair['input']), pair['output'])
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(2000+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted illegal counts')
    bad = generate_puzzle(3000, 2, 1)
    a = _analyze(bad['test'][0]['input'])
    p = next(iter(_reference_keep(a)))
    bad['test'][0]['output'][p] = a['background']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('accepted erasure of supported dot')
    _handwritten_checks()
    return dict(seeds=42, pairs=len(records), distinct_puzzles=len(signatures),
                input_shapes=len({f['shape'] for f in records}),
                layouts=len({f['layout'] for f in records}),
                count_combinations=len({f['counts'] for f in records}),
                sizes=sorted({f['size'] for f in records}), shifts=sorted(shifts))


def _handwritten_checks():
    # An even-sized square has a half-column center, with no rounding choice.
    grid = np.zeros((20, 24), dtype=np.int8)
    grid[4:8, 3:7] = 7
    grid[5:7, 4:6] = 0  # frame
    grid[11:15, 16:20] = 7  # solid
    grid[2, 1] = grid[9, 21] = 2  # (-2,-2) <-> (-2,5)
    grid[9, 1] = grid[9, 8] = 3  # locally symmetric but unsupported
    grid[5, 1], grid[12, 21] = 4, 5  # wrong-color counterparts
    expected = grid.copy()
    for p in ((9, 1), (9, 8), (5, 1), (12, 21)):
        expected[p] = 0
    assert np.array_equal(solve(grid), expected)
    # A partner's registered coordinate can lie outside the canvas: erase it.
    grid = np.zeros((15, 18), dtype=np.int8)
    grid[3:6, 3:6] = 7
    grid[4, 4] = 0
    grid[10:13, 0:3] = 7
    grid[3, 7] = 2  # reflects to column -2 relative to the partner at column 0
    expected = grid.copy()
    expected[3, 7] = 0
    assert np.array_equal(solve(grid), expected)


if __name__ == '__main__':
    report = self_test()
    Path('scratch').mkdir(exist_ok=True)
    Path('scratch/self_test.json').write_text(json.dumps(report, separators=(',', ':'))+'\n')
    print(json.dumps(report, separators=(',', ':')))
