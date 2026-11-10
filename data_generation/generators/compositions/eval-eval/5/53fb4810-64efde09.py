"""Keyed bars broadcast a pooled cyclic code as rolling, wrapping ribbons.

Only Python's standard library and NumPy are used. No parent generator, saved
example, seed, or construction record is consulted by the input-only decoder.
"""

from itertools import permutations

import numpy as np


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _orient(a, t):
    return np.ascontiguousarray(np.rot90(np.fliplr(a) if t >= 4 else a, t % 4))


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _point(frame, slot, distance):
    m, e, n = frame['m'], frame['e'], frame['n']
    return (m[0] + slot*e[0] + distance*n[0],
            m[1] + slot*e[1] + distance*n[1])


def _frame(a, top, left, shape):
    """The unique singleton structural color is an intrinsic corner key."""
    h, w = a.shape
    assert min(h, w) == 2 and 4 <= max(h, w) <= 9
    colors, counts = np.unique(a, return_counts=True)
    assert len(colors) == 3 and sum(counts == 1) == 1
    marker = int(colors[np.flatnonzero(counts == 1)[0]])
    r, c = map(int, np.argwhere(a == marker)[0])
    assert r in (0, h-1) and c in (0, w-1)
    if h == 2:
        e = (0, 1 if c == 0 else -1)
        n = (-1 if r == 0 else 1, 0)
    else:
        e = (1 if r == 0 else -1, 0)
        n = (0, -1 if c == 0 else 1)
    m = (top+r, left+c)
    maximum = (m[0] if n == (-1, 0) else shape[0]-1-m[0]
               if n == (1, 0) else m[1] if n == (0, -1)
               else shape[1]-1-m[1])
    length = max(h, w)
    cells = {(top+i, left+j) for i in range(h) for j in range(w)}
    frame = dict(m=m, e=e, n=n, maximum=maximum, length=length,
                 cells=cells, arr=a.copy(), top=top, left=left, marker=marker)
    frame['band'] = {_point(frame, s, d) for s in range(length)
                     for d in range(1, maximum+1)}
    # This normal form is independent of the object's screen orientation.
    frame['template'] = tuple(tuple(int(a[r+s*e[0]-k*n[0],
                                             c+s*e[1]-k*n[1]])
                                   for s in range(length)) for k in range(2))
    return frame


def _paint_code(grid, frames, code, *, roll=True, per_frame=None):
    out = grid.copy()
    for i, f in enumerate(frames):
        local_code = code if per_frame is None else per_frame[i]
        length = f['length']
        for d in range(1, f['maximum']+1):
            for s in range(length):
                residue = (s-(d-1)) % length if roll else s
                if residue in local_code:
                    out[_point(f, s, d)] = local_code[residue]
    return out


def _decode(input_grid):
    grid = np.asarray(input_grid)
    assert grid.ndim == 2 and all(1 <= v <= 30 for v in grid.shape)
    assert np.issubdtype(grid.dtype, np.integer)
    assert np.all((0 <= grid) & (grid <= 9))
    colors, counts = np.unique(grid, return_counts=True)
    assert sum(counts == counts.max()) == 1
    background = int(colors[counts.argmax()])
    frames, clues = [], []
    for cells in _components(grid != background):
        if len(cells) == 1:
            r, c = next(iter(cells))
            assert r in (0, grid.shape[0]-1) or c in (0, grid.shape[1]-1)
            clues.append((r, c, int(grid[r, c])))
            continue
        rows, cols = zip(*cells)
        t, l, b, z = min(rows), min(cols), max(rows), max(cols)
        a = grid[t:b+1, l:z+1]
        assert a.size == len(cells), 'bar must be a filled rectangle'
        frames.append(_frame(a, t, l, grid.shape))
    assert 2 <= len(frames) <= 5 and clues
    reference = frames[0]['template']
    assert all(f['template'] == reference for f in frames), 'not exact keyed copies'
    length = frames[0]['length']
    structural = {v for row in reference for v in row}
    assert background not in structural
    assert structural.isdisjoint(v for _, _, v in clues)
    occupied = set().union(*(f['cells'] for f in frames))
    bands, endpoints = set(), {}
    for i, f in enumerate(frames):
        assert f['maximum'] >= 2
        assert not occupied.intersection(f['band']), 'ribbon hits a bar'
        assert not bands.intersection(f['band']), 'ribbons intersect'
        bands.update(f['band'])
        for s in range(length):
            p = _point(f, s, f['maximum'])
            assert p not in endpoints
            endpoints[p] = (i, s)
    code, observations = {}, []
    local_codes = [{} for _ in frames]
    for r, c, color in clues:
        assert (r, c) in endpoints, 'clue outside an active ribbon end'
        i, s = endpoints[r, c]
        residue = (s-(frames[i]['maximum']-1)) % length
        assert residue not in code or code[residue] == color, 'conflicting phase clues'
        code[residue] = color
        local_codes[i][residue] = color
        observations.append((i, s, residue, color))
    assert 1 <= len(code) < length
    # Check isolation even when diagonal adjacency would not merge components.
    for i, (r, c, _) in enumerate(clues):
        assert all(max(abs(r-u), abs(c-v)) > 1 for u, v in occupied)
        assert all(max(abs(r-u), abs(c-v)) > 1 for u, v, _ in clues[:i])
    expected = _paint_code(grid, frames, code)
    assert np.any(expected != grid)
    return expected, dict(background=background, frames=frames, code=code,
                          local_codes=local_codes, observations=observations,
                          clues=clues, length=length, bands=bands)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode visible keys and boundary phases, pool, and roll to the edge."""
    return _decode(input_grid)[0]


def _template(rng, length, colors):
    a, b, marker = colors
    result = np.full((2, length), a, dtype=np.int8)
    result[0] = b
    # Runs and blocks can change, while the corner key stays uniquely readable.
    style = int(rng.integers(4))
    if style == 1 and length >= 5:
        result[0, int(rng.integers(2, length)): ] = a
    elif style == 2:
        for s in range(2, length):
            if rng.random() < .4:
                result[0, s] = a
    elif style == 3 and length >= 6:
        start = int(rng.integers(2, length-1))
        result[0, start:min(length, start+2)] = a
    result[0, 0] = marker
    if np.count_nonzero(result == b) < 2:
        result[0, 1:3] = b
    return result


def _compatible(f, previous):
    for g in previous:
        if (f['cells'] & g['cells'] or f['band'] & g['cells']
                or f['cells'] & g['band'] or f['band'] & g['band']):
            return False
        if any((r+dr, c+dc) in g['cells'] for r, c in f['cells']
               for dr in (-1, 0, 1) for dc in (-1, 0, 1)):
            return False
    return True


def _clue_allocation(rng, frames, residues, mode, redundant):
    occupied = set().union(*(f['cells'] for f in frames))
    options = {}
    length = frames[0]['length']
    for residue in residues:
        choices = []
        for i, f in enumerate(frames):
            s = (residue+f['maximum']-1) % length
            p = _point(f, s, f['maximum'])
            if all(max(abs(p[0]-r), abs(p[1]-c)) > 1 for r, c in occupied):
                choices.append((i, residue, p))
        rng.shuffle(choices)
        options[residue] = choices
        if not choices:
            return None
    groups = [[i] for i in rng.permutation(len(frames))] if mode == 'full' else [list(range(len(frames)))]
    for group in groups:
        answer, nodes = [], 0

        def search(k):
            nonlocal nodes
            nodes += 1
            if nodes > 4000:
                return False
            if k == len(residues):
                return mode != 'pool' or len({x[0] for x in answer}) >= 2
            for entry in options[residues[k]]:
                if entry[0] not in group:
                    continue
                if any(max(abs(entry[2][0]-x[2][0]), abs(entry[2][1]-x[2][1])) <= 1
                       for x in answer):
                    continue
                answer.append(entry)
                if search(k+1):
                    return True
                answer.pop()
            return False

        if search(0):
            if redundant:
                extras = [v for choices in options.values() for v in choices]
                rng.shuffle(extras)
                for v in extras:
                    if all(max(abs(v[2][0]-x[2][0]), abs(v[2][1]-x[2][1])) > 1
                           for x in answer):
                        answer.append(v)
                        break
            return answer
    return None


def _features(a):
    frames, length, code = a['frames'], a['length'], a['code']
    allocations = tuple(len(x) for x in a['local_codes'])
    normals = {f['n'] for f in frames}
    handedness = {f['e'][0]*f['n'][1]-f['e'][1]*f['n'][0] for f in frames}
    return dict(length=length, bars=len(frames), code_count=len(code),
                allocations=allocations, pooled=all(n < len(code) for n in allocations),
                unhinted=0 in allocations, reflected=len(handedness) > 1,
                rotated=len({n[0] == 0 for n in normals}) > 1,
                directions=tuple(sorted(normals)),
                distances=tuple(f['maximum'] for f in frames),
                adjacent=any((s+1) % length in code for s in code),
                repeated_color=len(set(code.values())) < len(code),
                redundant=len(a['clues']) > len(code),
                phase_shifted=any(s != r for _, s, r, _ in a['observations']),
                wrap=any(f['maximum'] > length for f in frames),
                clipped=any(f['maximum'] % length != 0 for f in frames),
                background=a['background'])


def _pack(rng, template, transforms, shape, residues, mode, anchor):
    # Bands sharing a normal require disjoint transverse intervals. Check this
    # necessary capacity constraint before attempting any random placement.
    normals = []
    for t in transforms:
        normals.append(_frame(_orient(template, t), 0, 0, shape)['n'])
    for n in set(normals):
        transverse = shape[1] if n[0] else shape[0]
        if normals.count(n)*template.shape[1] > transverse:
            return None
    for _ in range(4):
        frames = []
        for t in transforms:
            a = _orient(template, t)
            h, w = a.shape
            for _ in range(64):
                top = int(rng.integers(shape[0]-h+1))
                left = int(rng.integers(shape[1]-w+1))
                f = _frame(a, top, left, shape)
                if f['maximum'] < 2 or not _compatible(f, frames):
                    continue
                frames.append(f)
                break
            else:
                break
        if len(frames) != len(transforms):
            continue
        if anchor and (not any(f['maximum'] >= f['length']+2 for f in frames)
                       or len({f['maximum'] % f['length'] for f in frames}) < 2):
            continue
        clues = _clue_allocation(rng, frames, residues, mode,
                                 redundant=not anchor and rng.random() < .3)
        if anchor and clues is not None and not any(
                (residue+frames[i]['maximum']-1) % frames[i]['length'] != residue
                for i, residue, _ in clues):
            continue
        if clues is not None:
            return frames, clues
    return None


def _fallback(rng, template, residues, mode, anchor, count=2):
    """Guaranteed separated full bands; only used after bounded random packing."""
    length = template.shape[1]
    shape = (30, 30)
    if not anchor:
        placements = ((0, 3, 0), (4, 3, 20), (2, 25, 0),
                      (6, 25, 20), (1, 10, 12))[:count]
    elif anchor == 'rotate':
        placements = ((0, 12, 0), (1, 21, 20))
    else:
        placements = ((0, 14, 0), (4, 17, length+2))
    frames = [_frame(_orient(template, t), r, c, shape) for t, r, c in placements]
    assert all(_compatible(f, frames[:i]) for i, f in enumerate(frames))
    clue_mode = 'pool' if len(residues) > 1 else 'any'
    clues = _clue_allocation(rng, frames, residues, clue_mode, False)
    if clues is None:
        # Even the densest permitted code (five of nine residues) can be
        # evidenced on three separated boundaries with different phases.
        frames = [_frame(_orient(template, t), r, c, shape)
                  for t, r, c in ((0, 12, 0), (4, 15, 10), (0, 18, 20))]
        assert all(_compatible(f, frames[:i]) for i, f in enumerate(frames))
        clues = _clue_allocation(rng, frames, residues, clue_mode, False)
    assert clues is not None
    return shape, frames, clues


def _generate_pair(rng, anchor=None, shift=False, previous_background=None):
    palette = list(map(int, rng.permutation(10)))
    if palette[0] == previous_background:
        palette[0], palette[-1] = palette[-1], palette[0]
    background = palette[0]
    # A visible, meaningful shift: longer keyed cycles in tests. Every training
    # pair already teaches wrapping and distances, rather than a hidden code.
    length = int(rng.integers(7, 10) if shift else rng.integers(4, 7))
    count = int(rng.choice([2, 2, 3] if anchor else [2, 3, 3, 4, 5]))
    code_count = int(rng.integers(2 if anchor else 1, min(5, length-1)+1))
    residues = tuple(sorted(map(int, rng.choice(length, code_count, replace=False))))
    signal_count = code_count if anchor else int(rng.integers(1, code_count+1))
    signals = palette[4:4+signal_count]
    signal_values = signals + list(map(int, rng.choice(signals, code_count-signal_count)))
    rng.shuffle(signal_values)
    code = dict(zip(residues, signal_values))
    template = _template(rng, length, palette[1:4])
    base = int(rng.integers(8))
    if anchor == 'reflect':
        transforms = [base, base ^ 4] + ([base] if count == 3 else [])
    elif anchor == 'rotate':
        transforms = [base, (base//4)*4+(base+1) % 4] + ([base ^ 4] if count == 3 else [])
    else:
        pattern = int(rng.integers(4))
        transforms = ([base]*count if pattern == 0 else
                      [base if i % 2 == 0 else base ^ 4 for i in range(count)] if pattern == 1
                      else list(map(int, rng.integers(8, size=count))))
    mode = 'pool' if anchor == 'reflect' or shift else str(rng.choice(['full', 'pool', 'any']))
    if code_count == 1:
        mode = 'any'
    elif mode == 'full' and any((s+1) % length in residues for s in residues):
        mode = 'any'
    result = None
    for attempt in range(10):
        low = max(11, length+4, 19 if count >= 4 else 14 if anchor else 11)
        shape = (int(rng.integers(low, 31)), int(rng.integers(low, 31)))
        if attempt >= 5:
            shape = (30, 30)
            if not anchor:
                transforms = list(map(int, rng.integers(8, size=count)))
        result = _pack(rng, template, transforms, shape, residues, mode, anchor)
        if result is not None:
            break
        if attempt == 4 and mode == 'full':
            mode = 'pool'
    if result is None:
        shape, frames, clues = _fallback(rng, template, residues, mode, anchor, count)
    else:
        frames, clues = result
    grid = np.full(shape, background, dtype=np.int8)
    for f in frames:
        t, l = f['top'], f['left']
        h, w = f['arr'].shape
        grid[t:t+h, l:l+w] = f['arr']
    for _, residue, p in clues:
        grid[p] = code[residue]
    # Construction uses np.roll of the actual sampled complete vector.
    # Decoder instead derives residue classes from returned boundary pixels.
    output = grid.copy()
    vector = np.array([code.get(s, background) for s in range(length)], dtype=np.int8)
    for f in frames:
        for d in range(1, f['maximum']+1):
            for s, color in enumerate(np.roll(vector, d-1)):
                output[_point(f, s, d)] = color
    pair = {'input': grid, 'output': output}
    expected, analysis = _decode(grid)
    assert np.array_equal(expected, output)
    if anchor:
        features = _features(analysis)
        assert features['phase_shifted'] and features['wrap']
        if anchor == 'reflect':
            assert features['pooled'] and features['reflected']
        else:
            assert features['rotated']
    return pair


def _check_count(name, value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _check_count('num_train', num_train, 2, 10)
    num_test = _check_count('num_test', num_test, 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    puzzle, used = {'train': [], 'test': []}, set()
    previous_background = None
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            anchor = ('reflect', 'rotate')[i] if split == 'train' and i < 2 else None
            pair = _generate_pair(rng, anchor, shift=split == 'test',
                                  previous_background=previous_background)
            if _key(pair['input']) in used:
                colors = list(map(int, np.unique(pair['input'])))
                for values in permutations(range(10), len(colors)):
                    mapping = np.arange(10, dtype=np.int8)
                    mapping[colors] = values
                    candidate = {k: mapping[v] for k, v in pair.items()}
                    if _key(candidate['input']) not in used:
                        pair = candidate
                        break
                else:
                    raise AssertionError('finite recoloring space exhausted')
            used.add(_key(pair['input']))
            previous_background = _decode(pair['input'])[1]['background']
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    a, b = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= a <= 10 and 1 <= b <= 4 and a > b
    used, features, analyses = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            for g in (inp, out):
                assert g.ndim == 2 and all(1 <= v <= 30 for v in g.shape)
                assert np.issubdtype(g.dtype, np.integer) and np.all((0 <= g) & (g <= 9))
            assert inp.shape == out.shape
            assert _key(inp) not in used
            used.add(_key(inp))
            expected, analysis = _decode(inp)
            assert np.array_equal(expected, out), 'incorrect rolling ribbon'
            assert all(out[p] == inp[p] for f in analysis['frames'] for p in f['cells'])
            assert all(out[r, c] == color for r, c, color in analysis['clues'])
            protected = np.ones(inp.shape, dtype=bool)
            for p in analysis['bands']:
                protected[p] = False
            assert np.array_equal(out[protected], inp[protected])
            features.append(_features(analysis))
            analyses.append((inp, out, analysis))
    train = features[:a]
    assert any(f['pooled'] and f['reflected'] for f in train), 'teach pooled reflected transfer'
    assert any(f['rotated'] for f in train), 'teach quarter turns'
    assert any(f['wrap'] and f['phase_shifted'] for f in train), 'teach periodic boundary phase'
    assert min(f['length'] for f in features[a:]) > max(f['length'] for f in train)
    assert any(not np.array_equal(_paint_code(inp, data['frames'], data['code'], roll=False), out)
               for inp, out, data in analyses[:a]), 'A ablation must change a demonstration'
    assert any(not np.array_equal(_paint_code(inp, data['frames'], data['code'],
                                            per_frame=data['local_codes']), out)
               for inp, out, data in analyses[:a]), 'B ablation must change a demonstration'
    assert any(_wrong_phase_differs(inp, out, data) for inp, out, data in analyses[:a]), \
        'teach distance-corrected boundary clues'


def _wrong_phase_differs(inp, out, data):
    wrong = {}
    for _, slot, _, color in data['observations']:
        if slot in wrong and wrong[slot] != color:
            return True
        wrong[slot] = color
    return not np.array_equal(_paint_code(inp, data['frames'], wrong), out)


def _hand_regression():
    """A literal five-slice example, including a reflected transfer."""
    inp = np.zeros((15, 17), dtype=np.int8)
    bar = np.array([[3, 2, 2, 2, 2], [1, 1, 1, 1, 1]], dtype=np.int8)
    inp[4:6, 1:6] = bar
    inp[9:11, 9:14] = np.fliplr(bar)
    inp[0, 1], inp[0, 3], inp[0, 10] = 6, 7, 4
    expected = inp.copy()
    slices = ((4, 0, 6, 0, 7), (7, 4, 0, 6, 0), (0, 7, 4, 0, 6),
              (6, 0, 7, 4, 0), (0, 6, 0, 7, 4))
    for d in range(1, 5):
        expected[4-d, 1:6] = slices[(d-1) % 5]
    for d in range(1, 10):
        expected[9-d, 9:14] = slices[(d-1) % 5][::-1]
    assert np.array_equal(solve(inp), expected)
    return {'input': inp, 'output': expected}


def self_test():
    """Bounded semantic, equivariance, count, and 42-seed variation checks."""
    _hand_regression()
    # Exercise the escape path explicitly, including one active residue and
    # dense adjacent codes; ordinary random samples need not use a fallback.
    for length in range(4, 10):
        rng = np.random.default_rng(600+length)
        template = _template(rng, length, [1, 2, 3])
        for anchor in (None, 'reflect', 'rotate'):
            residues = tuple(range(min(5, length-1))) if anchor else (0,)
            shape, frames, clues = _fallback(rng, template, residues, 'pool', anchor)
            grid = np.zeros(shape, dtype=np.int8)
            for f in frames:
                h, w = f['arr'].shape
                grid[f['top']:f['top']+h, f['left']:f['left']+w] = f['arr']
            for _, residue, p in clues:
                grid[p] = 4+residue
            expected = _paint_code(grid, frames, {s: 4+s for s in residues})
            assert np.array_equal(solve(grid), expected)
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures = {tuple(_key(pair['input']) for split in ('train', 'test')
                        for pair in p[split]) for p in puzzles}
    assert len(signatures) == 42
    records, shapes, structures = [], set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                expected, a = _decode(inp)
                assert np.array_equal(expected, out)
                records.append(_features(a))
                shapes.add(inp.shape)
                structures.add(_key(inp != a['background']))
                # An arbitrary global isometry/recoloring cannot change the rule.
                t = len(records) % 8
                recolor = np.roll(np.arange(10, dtype=np.int8), len(records) % 10)
                assert np.array_equal(solve(_orient(recolor[inp], t)), _orient(recolor[out], t))
    assert len(shapes) >= 35 and len(structures) >= 180
    assert {f['length'] for f in records} == set(range(4, 10))
    assert {f['bars'] for f in records} >= {2, 3, 4, 5}
    assert {f['code_count'] for f in records} >= {1, 2, 3, 4, 5}
    assert {f['background'] for f in records} == set(range(10))
    assert len({f['allocations'] for f in records}) >= 15
    assert len({f['distances'] for f in records}) >= 100
    assert {n for f in records for n in f['directions']} == {(-1, 0), (1, 0), (0, -1), (0, 1)}
    for flag in ('pooled', 'unhinted', 'reflected', 'rotated', 'adjacent',
                 'repeated_color', 'redundant', 'phase_shifted', 'wrap', 'clipped'):
        assert any(f[flag] for f in records), flag
    assert tuple(_key(p['input']) for p in puzzles[0]['train']+puzzles[0]['test']) == tuple(
        _key(p['input']) for p in generate_puzzle(0)['train']+generate_puzzle(0)['test'])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(1917+nt*10+ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]]
               for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('output mutation accepted')
