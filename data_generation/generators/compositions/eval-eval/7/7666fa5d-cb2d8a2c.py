"""Tip-projected bridges. Only NumPy and the Python standard library are used.

No parent module, stored sample, seed or sampling plan is consulted by solve.
Coordinates: q=c-s*r, t=r+s*c; consecutive diagonal pixels differ by 2 in t.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= k <= 30 for k in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer color grid with dimensions 1..30")
    return a


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _read(input_grid):
    """Decode roles, whole strokes and endpoint codes from the input alone."""
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    if len(values) != 3 or counts.max() <= a.size // 2:
        raise ValueError("three colors and a majority background are required")
    bg = int(values[counts.argmax()])
    other = [int(v) for v in values if v != bg]
    cells = [(int(r), int(c)) for r, c in np.argwhere(a != bg)]
    candidates = []
    for s in (-1, 1):
        lines = {}
        for r, c in cells:
            lines.setdefault(c - s*r, []).append((r, c))
        runs = []
        for q, points in sorted(lines.items()):
            points.sort()
            for r, c in points:
                if not runs or runs[-1][0] != q or r != runs[-1][1][-1][0] + 1:
                    runs.append((q, []))
                runs[-1][1].append((r, c))
        for bar, marker in (other, other[::-1]):
            strokes = []
            for q, points in runs:
                n = sum(int(a[r, c]) == marker for r, c in points)
                if not 1 <= n <= 4 or len(points) < 2*n + 1:
                    break
                marked = [i for i, (r, c) in enumerate(points) if a[r, c] == marker]
                low = marked == list(range(0, 2*n, 2))
                high = marked == list(range(len(points)-2*n+1, len(points), 2))
                if low == high:
                    break
                lo = points[0][0] + s*points[0][1]
                hi = points[-1][0] + s*points[-1][1]
                direction = 1 if low else -1
                d = n + 1
                el, eh = (hi+2, hi+2*d) if low else (lo-2*d, lo-2)
                strokes.append(dict(q=q, lo=lo, hi=hi, n=n, d=d,
                                    direction=direction, el=el, eh=eh,
                                    length=len(points)))
            else:
                if len(strokes) >= 2:
                    candidates.append(dict(bg=bg, bar=bar, marker=marker,
                                           slope=s, strokes=strokes))
    if len(candidates) != 1:
        raise ValueError("diagonal orientation and cap roles must decode uniquely")
    return a, candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Paint continuous scanline brackets between finite virtual tip extensions."""
    a, info = _read(input_grid)
    rr, cc = np.indices(a.shape)
    q, t = cc-info['slope']*rr, rr+info['slope']*cc
    b = a.copy()
    b[a == info['marker']] = info['bar']
    for level in range(int(t.min()), int(t.max()) + 1):
        active = [p['q'] for p in info['strokes'] if p['el'] <= level <= p['eh']]
        if len(active) >= 2:
            b[(a == info['bg']) & (t == level)
              & (min(active) < q) & (q < max(active))] = info['marker']
    return b


def _prediction(a, info, alternative=''):
    """Pairwise parallelogram union checks the sweep via another construction.

    The alternatives are deliberately plausible weaker interpretations. This
    checker shares the decoder; it is authored validation, not an external oracle.
    """
    rr, cc = np.indices(a.shape)
    q, t = cc-info['slope']*rr, rr+info['slope']*cc
    specs = []
    for p in info['strokes']:
        lo, hi = p['el'], p['eh']
        if alternative == 'visible':
            lo, hi = p['lo'], p['hi']
        elif alternative == 'unit':
            lo, hi = ((p['hi']+2, p['hi']+4) if p['direction'] == 1
                      else (p['lo']-4, p['lo']-2))
        elif alternative == 'marked':
            lo, hi = ((p['lo']-2*p['d'], p['lo']-2) if p['direction'] == 1
                      else (p['hi']+2, p['hi']+2*p['d']))
        elif alternative == 'no_plus_one':
            lo, hi = ((p['hi']+2, p['hi']+2*p['n']) if p['direction'] == 1
                      else (p['lo']-2*p['n'], p['lo']-2))
        elif alternative == 'tip_included':
            lo, hi = ((p['hi'], p['hi']+2*p['n']) if p['direction'] == 1
                      else (p['lo']-2*p['n'], p['lo']))
        specs.append((p['q'], lo, hi))
    pairs = [(i, j) for i in range(len(specs)) for j in range(i+1, len(specs))]
    if alternative == 'consecutive':
        pairs = [(i, i+1) for i in range(len(specs)-1)]
    paint = np.zeros(a.shape, dtype=bool)
    for i, j in pairs:
        qa, la, ha = specs[i]
        qb, lb, hb = specs[j]
        mask = ((max(la, lb) <= t) & (t <= min(ha, hb))
                & (min(qa, qb) < q) & (q < max(qa, qb)))
        if alternative == 'centers':
            mask &= ((t-qa) % 2 == 0) & ((t-qb) % 2 == 0)
        paint |= mask
    b = a.copy()
    b[a == info['marker']] = info['bar']
    b[paint & (a == info['bg'])] = info['marker']
    return b


_ALTERNATIVES = ('visible', 'unit', 'marked', 'consecutive', 'centers',
                 'no_plus_one', 'tip_included')


def _features(pair):
    a, info = _read(pair['input'])
    b = np.asarray(pair['output'])
    return dict(info=info,
                witnesses={name for name in _ALTERNATIVES
                           if not np.array_equal(b, _prediction(a, info, name))},
                paint=int(np.count_nonzero((a == info['bg']) & (b == info['marker']))))


def _specs(rng, kind):
    """Sample virtual intervals first, then attach coded physical strokes.

    This is geometric construction, not a composition of parent generators.
    The solver always recovers these intervals from visible endpoints instead.
    """
    if kind == 'skip':
        count = int(rng.integers(3, 7))
    elif kind == 'mixed':
        count = int(rng.integers(2, 6))
    else:
        count = int(rng.integers(2, 9))
    qs = [0]
    for _ in range(count-1):
        qs.append(qs[-1] + int(rng.integers(3, 8)))
    ns = list(map(int, rng.integers(1, 5, size=count)))
    if kind == 'skip':
        ns[0], ns[-1] = (3, 4) if rng.integers(2) else (4, 3)
        ns[int(rng.integers(1, count-1))] = 1
    if kind == 'mixed':
        ns = list(map(int, rng.integers(1, 3, size=count)))
        ns[0], ns[-1] = 1, 2
    sign = int(rng.choice((-1, 1)))
    directions = [sign]*count
    if kind == 'mixed' or (kind == 'free' and rng.random() < .55):
        directions = list(map(int, rng.choice((-1, 1), size=count)))
        directions[0], directions[-1] = -1, 1
    mode = int(rng.integers(4))
    starts = []
    for i, (q, n) in enumerate(zip(qs, ns)):
        if kind == 'skip':
            start = (0 if i in (0, count-1) else 2) + q % 2
        elif mode == 0:
            start = q % 2  # aligned up to the necessary pixel phase
        elif mode == 1:
            start = 2*int(rng.integers(-3, 4)) + q % 2
        elif mode == 2:
            start = 2*i + q % 2  # staggered chain; common overlap can be absent
        else:
            start = 2*(0 if i < count//2 else 5) + q % 2
        starts.append(start)
    result = []
    for q, n, start, direction in zip(qs, ns, starts, directions):
        # Two consecutive permanent pixels at the other end prevent role ties.
        extra = int(rng.integers(0, 4 if len(set(directions)) == 1 else 2))
        length = 2*n+1+extra
        el, eh = start, start+2*n
        if direction == 1:
            hi = el-2
            lo = hi-2*(length-1)
        else:
            lo = eh+2
            hi = lo+2*(length-1)
        result.append(dict(q=q, lo=lo, hi=hi, n=n, direction=direction,
                           el=el, eh=eh, length=length))
    return result


def _render(rng, specs, slope, colors):
    bg, bar, marker = colors
    physical, extent = [], []
    for p in specs:
        stroke = []
        for t in range(p['lo'], p['hi']+1, 2):
            r, c = (t-slope*p['q'])//2, (p['q']+slope*t)//2
            stroke.append((r, c))
        physical.append(stroke)
        extent.extend(stroke)
        for t in (p['el'], p['eh']):
            extent.append(((t-slope*p['q'])//2, (p['q']+slope*t)//2))
    r0, c0 = np.min(extent, axis=0)
    r1, c1 = np.max(extent, axis=0)
    h, w = int(r1-r0+1), int(c1-c0+1)
    if max(h, w) > 30:
        return None
    height = h+int(rng.integers(0, min(6, 30-h)+1))
    width = w+int(rng.integers(0, min(6, 30-w)+1))
    top = int(rng.integers(0, height-h+1))
    left = int(rng.integers(0, width-w+1))
    a = np.full((height, width), bg, dtype=np.int8)
    for p, stroke in zip(specs, physical):
        for i, (r, c) in enumerate(stroke):
            index = i if p['direction'] == 1 else len(stroke)-1-i
            code = index % 2 == 0 and index < 2*p['n']
            a[int(r-r0)+top, int(c-c0)+left] = marker if code else bar
    try:
        _, info = _read(a)
    except ValueError:
        return None
    # Labels use the pairwise geometry; solve uses the scanline construction.
    return dict(input=a, output=_prediction(a, info))


def _make_pair(rng, kind, slope, colors, seen):
    required = ({'visible', 'unit', 'consecutive', 'centers',
                 'no_plus_one', 'tip_included'} if kind == 'skip'
                else {'visible', 'marked'} if kind == 'mixed' else set())
    for _ in range(384):
        pair = _render(rng, _specs(rng, kind), slope, colors)
        if pair is None or _key(pair['input']) in seen:
            continue
        f = _features(pair)
        if f['paint'] == 0 or not required <= f['witnesses']:
            continue
        seen.add(_key(pair['input']))
        return pair
    raise RuntimeError('bounded geometric sampling exhausted')


def _count(value, lower, upper, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not lower <= int(value) <= upper):
        raise ValueError(f'{name} must be an integer in {lower}..{upper}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError('seed must be a nonnegative integer')
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(int(seed))
    colors = tuple(map(int, rng.choice(10, size=3, replace=False)))
    slope = int(rng.choice((-1, 1)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for i in range(num_train):
        kind = 'skip' if i == 0 else 'mixed' if i == 1 else 'free'
        puzzle['train'].append(_make_pair(rng, kind, slope, colors, seen))
    for _ in range(num_test):
        puzzle['test'].append(_make_pair(rng, 'free', -slope, colors, seen))
    for split in puzzle:
        puzzle[split] = [puzzle[split][int(i)] for i in rng.permutation(len(puzzle[split]))]
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Validate displayed grids, semantic invariants and teaching witnesses."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, roles = set(), set()
    features = {'train': [], 'test': []}
    for split in puzzle:
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape
            key = _key(a)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            _, info = _read(a)
            roles.add((info['bg'], info['bar'], info['marker']))
            assert np.array_equal(b[a == info['bar']], a[a == info['bar']])
            assert np.all(b[a == info['marker']] == info['bar'])
            assert np.array_equal(b, _prediction(a, info)), 'incorrect bridge geometry'
            assert np.array_equal(b, solve(a)), 'scanline and pairwise checks differ'
            f = _features(pair)
            assert f['paint'] > 0, 'must exhibit a nonempty bridge'
            features[split].append(f)
    assert len(roles) == 1, 'color conventions must agree within a puzzle'
    witnesses = set().union(*(f['witnesses'] for f in features['train']))
    assert set(_ALTERNATIVES) <= witnesses, 'missing shortcut counterexample in training'
    train_counts = {p['n'] for f in features['train'] for p in f['info']['strokes']}
    test_counts = {p['n'] for f in features['test'] for p in f['info']['strokes']}
    assert test_counts <= train_counts, 'test clearance not demonstrated'
    assert any(len({p['direction'] for p in f['info']['strokes']}) == 2
               for f in features['train']), 'missing mixed-end demonstration'
    return True


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for p in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(p[role])
                digest.update(bytes(a.shape))
                digest.update(a.astype(np.int8).tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded 42-seed measurements, analytical fixture and rejection checks.

    At most two puzzles are held at once. No files, JSON or execution traces
    are written. Returned measurements are small, reconstructible aggregates.
    """
    # Literal 7x9 fixture, q=0 and 4, positive launches, n=1 and 2.
    a = np.full((7, 9), 0, dtype=np.int8)
    a[(0, 1, 2), (0, 1, 2)] = (3, 7, 7)
    a[(0, 1, 2, 3, 4), (4, 5, 6, 7, 8)] = (3, 7, 3, 7, 7)
    b = a.copy()
    b[a == 3] = 7
    # First echo t=6..8, second t=14..18: no shared active interval.
    assert np.array_equal(solve(a), b)
    # Two strokes end on t=12 but have different lengths and unary counts.
    c = np.zeros((10, 12), dtype=np.int8)
    c[(4, 5, 6), (4, 5, 6)] = (3, 7, 7)
    c[(0, 1, 2, 3, 4), (4, 5, 6, 7, 8)] = (3, 7, 3, 7, 7)
    expected = c.copy()
    expected[c == 3] = 7
    # Echo overlap t=14..16; strict 0<q<4 gives four paint cells.
    expected[(6, 7, 6, 7), (8, 8, 9, 9)] = 3
    assert np.array_equal(solve(c), expected), 'analytical continuous-interval fixture'
    assert expected[6, 9] == 3  # t=15 lies between both echo pixel centers
    assert np.array_equal(solve(c[:8, :9]), expected[:8, :9]), 'clipped virtual support'
    palette = np.array([8, 5, 2, 0, 4, 3, 1, 9, 7, 6], dtype=np.int8)
    assert np.array_equal(solve(palette[c]), palette[expected]), 'palette equivariance'
    for k in range(4):
        assert np.array_equal(solve(np.rot90(c, k)), np.rot90(expected, k))
        assert np.array_equal(solve(np.rot90(c[:, ::-1], k)),
                              np.rot90(expected[:, ::-1], k))
    shapes, layouts, signatures, ns, ends, counts, slopes, palettes = (set() for _ in range(8))
    records = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        if seed in (0, 21, 41):
            assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
        for split in puzzle:
            for pair in puzzle[split]:
                a, info = _read(pair['input'])
                shapes.add(a.shape)
                ns.update(p['n'] for p in info['strokes'])
                ends.update(p['direction'] for p in info['strokes'])
                counts.add(len(info['strokes']))
                slopes.add(info['slope'])
                palettes.add((info['bg'], info['bar'], info['marker']))
                normalized = np.empty_like(a)
                for i, color in enumerate((info['bg'], info['bar'], info['marker'])):
                    normalized[a == color] = i
                layouts.add(_key(normalized))
                records += 1
    assert len(signatures) == 42 and len(layouts) >= 180 and len(shapes) >= 40
    assert ns == {1, 2, 3, 4} and ends == {-1, 1} and slopes == {-1, 1}
    assert {2, 3, 4, 5, 6} <= counts and len(palettes) >= 30
    for ntrain in range(2, 11):
        for ntest in range(1, min(4, ntrain-1)+1):
            generate_puzzle(79, ntrain, ntest)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    damaged = generate_puzzle(0, 2, 1)
    p = damaged['test'][0]
    _, info = _read(p['input'])
    r, c = np.argwhere((p['input'] == info['bg']) & (p['output'] == info['marker']))[0]
    p['output'][r, c] = info['bg']
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('missing bridge cell accepted')
    return dict(seeds=42, pairs=records, distinct_puzzles=len(signatures),
                shapes=len(shapes), normalized_layouts=len(layouts),
                stroke_counts=sorted(counts), marker_counts=sorted(ns),
                palettes=len(palettes), independent_semantic_verification=False)
