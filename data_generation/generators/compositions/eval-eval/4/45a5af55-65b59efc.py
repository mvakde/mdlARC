"""Nested transparent stencils: ordered depth codes, keyed motifs, and reveal.

Only NumPy and the standard library are used. Generation writes no files.
The input-only solver searches visible ink from the innermost covering layer.
"""

import hashlib
import numpy as np


HOST_SEEDS = (0,975324646,18923994,550840799,841285318,227413550,323676891,
              463695482,112196280,186627460,568824005,786587897,596886786,
              746024460,315233716,121372875,14074571,371374296,127809191,
              183136575,971287525,285557183,954978650,221403612,188415833,
              40185618,874662400,366740979,746937526,230364871,189255384,
              173490217,230796528,423721500,45586024,376841290,935254460,
              390397386,474460346,647716913,812271762,770714732)


def _count(value, low, high, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= v <= 30 for v in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("expected an integer ARC grid, sides 1..30, colors 0..9")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _parse(input_grid):
    """Decode the spine, stripe code, bitmap library and aligned target bars."""
    a = _grid(input_grid)
    bg = int(a[0, 0])
    spine = np.flatnonzero(a[0] != bg)
    assert len(spine) == 1, "one spine crosses the upper background border"
    divider = int(spine[0])
    w = divider - 1
    assert 1 <= w <= 4 and a.shape[0] >= 6
    sep = int(a[0, divider])
    assert sep != bg and np.all(a[:, divider] == sep)
    assert np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)
    assert np.all(np.delete(a[-1], divider) == bg)
    code_box = a[1:-1, 1:divider]
    assert np.all(code_box == code_box[:, :1])
    active = np.flatnonzero(code_box[:, 0] != bg)
    assert len(active) >= 3 and np.array_equal(active, np.arange(len(active)))
    code = code_box[:len(active), 0].copy()
    assert 3 <= len(code) <= 16 and code[-1] == code[-2]
    assert sep not in code
    lib = a[1:-1, divider+1:-1]
    rows = np.flatnonzero(np.all(lib == sep, axis=1))
    assert len(rows) >= 1
    n = int(rows[0])
    assert 2 <= n <= 5 and lib.shape[0] >= n+2
    assert (lib.shape[1]+1) % (n+1) == 0
    k = (lib.shape[1]+1)//(n+1)
    assert 1 <= k <= 8
    masks, targets, order = {}, {}, []
    for j in range(k):
        c = j*(n+1)
        panel = lib[:n, c:c+n]
        colors = np.unique(panel[panel != bg])
        assert len(colors) == 1
        label = int(colors[0])
        assert label != sep and label not in masks
        target = int(lib[n+1, c])
        assert target != bg and np.all(lib[n+1, c:c+n] == target)
        masks[label] = panel == label
        targets[label] = target
        order.append(label)
        if j < k-1:
            assert np.all(lib[:n+2, c+n] == sep)
    assert np.all(lib[n+2:] == bg)
    assert set(map(int, code)) == set(order), "every library identity is used"
    assert a.shape[0] == max(len(code), n+2)+2
    return dict(bg=bg, sep=sep, w=w, n=n, code=code, masks=masks,
                targets=targets, order=order)


def _encode(code, masks, targets, order, bg, sep, w):
    n = next(iter(masks.values())).shape[0]
    h = max(len(code), n+2)
    library_width = len(order)*(n+1)-1
    a = np.full((h+2, w+library_width+3), bg, dtype=np.int8)
    a[:, w+1] = sep
    a[1:len(code)+1, 1:w+1] = np.asarray(code)[:, None]
    start = w+2
    for j, label in enumerate(order):
        c = start+j*(n+1)
        a[1:n+1, c:c+n] = np.where(masks[label], label, bg)
        a[n+2, c:c+n] = targets[label]
        if j < len(order)-1:
            a[1:n+3, c+n] = sep
    a[n+1, start:-1] = sep
    return a


def _transform(mask, index):
    return np.rot90(np.fliplr(mask) if index >= 4 else mask, index % 4)


def _paint(p, alternative=None):
    """Forward compositing, also used to evaluate explicit wrong hypotheses."""
    layers = list(map(int, p['code'][:-1]))
    if alternative == 'reverse':
        layers.reverse()
    elif alternative == 'collapse':
        layers = [x for i, x in enumerate(layers) if i == 0 or x != layers[i-1]]
    side, n = 2*len(layers), p['n']
    out = np.full((side, side), p['bg'], dtype=np.int8)
    owner = np.full((side, side), -1, dtype=np.int8)
    correspondence = dict(zip(dict.fromkeys(layers), p['order']))
    if alternative == 'first_only':
        layers = [layers[0]]
    elif alternative == 'last_only':
        layers = [layers[-1]]
    for d, label in enumerate(layers):
        if alternative == 'position':
            label = correspondence[label]
        mask = p['masks'][label]
        if alternative and alternative.startswith('transform_'):
            mask = _transform(mask, int(alternative.split('_')[1]))
        size = side-2*d
        coordinate = np.arange(size)
        if alternative == 'global_phase':
            coordinate = coordinate+d
        tiled = mask[coordinate[:, None] % n, coordinate[None, :] % n]
        if alternative == 'solid':
            tiled = np.ones((size, size), dtype=bool)
        if alternative == 'shell':
            shell = np.ones((size, size), dtype=bool)
            shell[1:-1, 1:-1] = False
            tiled = tiled & shell
        target = label if alternative == 'source_color' else p['targets'][label]
        region = out[d:side-d, d:side-d]
        if alternative == 'opaque':
            region[:] = p['bg']
        region[tiled] = target
        owner[d:side-d, d:side-d][tiled] = d
    return out, owner


def solve(input_grid: np.ndarray) -> np.ndarray:
    """For each pixel, find its newest covering ink using only the input."""
    p = _parse(input_grid)
    layers = p['code'][:-1]
    n, side = p['n'], 2*len(layers)
    output = np.full((side, side), p['bg'], dtype=np.int8)
    for r in range(side):
        for c in range(side):
            depth = min(r, c, side-1-r, side-1-c)
            for d in range(depth, -1, -1):
                label = int(layers[d])
                if p['masks'][label][(r-d) % n, (c-d) % n]:
                    output[r, c] = p['targets'][label]
                    break
    return output


_ALTERNATIVES = ('solid', 'opaque', 'global_phase', 'shell', 'reverse',
                 'collapse', 'first_only', 'last_only', 'position', 'source_color')
_ORIENTATIONS = tuple('transform_'+str(i) for i in range(1, 8))


def _wrong(p, expected, alternative):
    candidate, _ = _paint(p, alternative)
    return candidate.shape != expected.shape or not np.array_equal(candidate, expected)


def _motif(rng, n):
    rr, cc = np.indices((n, n))
    style = int(rng.integers(9))
    if style == 0:
        mask = np.ones((n, n), dtype=bool)
    elif style == 1:
        mask = (rr == 0) | (cc == 0) | (rr == n-1) | (cc == n-1)
    elif style == 2:
        mask = (rr+cc) % 2 == int(rng.integers(2))
    elif style == 3:
        mask = (rr == cc) | ((rr+cc == n-1) & bool(rng.integers(2)))
    elif style == 4:
        mask = (rr == int(rng.integers(n))) | (cc == int(rng.integers(n)))
    elif style == 5:
        mask = np.broadcast_to(rng.random((n, 1)) < .5, (n, n)).copy()
    elif style == 6:
        mask = np.broadcast_to(rng.random((1, n)) < .5, (n, n)).copy()
    else:
        mask = rng.random((n, n)) < rng.uniform(.15, .85)
    # At least one ink site on the top/left edge of the final 2x2 crop.
    # This makes the last occurrence of every generated identity observable.
    if not (mask[0, 0] or mask[0, 1] or mask[1, 0]):
        r, c = ((0, 0), (0, 1), (1, 0))[int(rng.integers(3))]
        mask[r, c] = True
    return mask.copy()


def _make_code(rng, labels, length, role):
    order = [int(x) for x in rng.permutation(labels)]
    if role == 0:
        # A genuine return plus repeated layers, with every identity exercised.
        layers = order+[order[0], order[0]]
        while len(layers) < length:
            layers.insert(int(rng.integers(len(layers)+1)), int(rng.choice(labels)))
    elif role == 1:
        # A color first introduced at the innermost layer.
        layers = order[:-1]
        while len(layers) < length-1:
            layers.insert(int(rng.integers(len(layers)+1)), int(rng.choice(order[:-1])))
        layers.append(order[-1])
    else:
        style = int(rng.integers(4))
        layers = order.copy()
        while len(layers) < length:
            if style == 0:
                index = int(rng.integers(len(layers)))
                layers.insert(index, layers[index])
            elif style == 1:
                layers.append(order[len(layers) % len(order)])
            else:
                layers.insert(int(rng.integers(len(layers)+1)), int(rng.choice(labels)))
    return np.array(layers+[layers[-1]], dtype=np.int8)


def _make_pair(rng, length, n, role, seen):
    for attempt in range(192):
        bg, sep = map(int, rng.choice(10, 2, replace=False))
        w = int(rng.integers(1, 5))
        capacity = min(6, (28-w)//(n+1), length-(2 if role == 0 else 1 if role == 1 else 0))
        k = int(rng.integers(2, capacity+1))
        labels = [int(c) for c in rng.choice([c for c in range(10) if c not in (bg, sep)], k, replace=False)]
        code = _make_code(rng, labels, length, role)
        masks = {label: _motif(rng, n) for label in labels}
        if role >= 2 and rng.random() < .25:
            masks[labels[-1]] = masks[labels[0]].copy()
        pool = [c for c in range(10) if c != bg]
        values = rng.choice(pool, k, replace=role >= 2)
        targets = dict(zip(labels, map(int, values)))
        order = [int(x) for x in rng.permutation(labels)]
        p = dict(bg=bg, sep=sep, w=w, n=n, code=code, masks=masks,
                 targets=targets, order=order)
        output, _ = _paint(p)
        # The first demonstration is an observed counterexample to every
        # listed competing interpretation, not just a sampler-style promise.
        if role == 0 and not all(_wrong(p, output, a) for a in _ALTERNATIVES+_ORIENTATIONS):
            continue
        inp = _encode(code, masks, targets, order, bg, sep, w)
        key = _key(inp)
        if key in seen:
            continue
        seen.add(key)
        return {'input': inp, 'output': output}
    raise RuntimeError("bounded informative-example sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    training_lengths = []
    for i in range(num_train):
        if regime == 0:
            choices = list(range(5, 10))
        elif regime == 1:
            choices = list(range(8, 16))
        elif regime == 2:
            choices = list(range(5, 16))
        else:
            choices = [4] if i == 0 else [12] if i == 1 else [4, 12]
        if i == 1 and regime != 3:
            choices = [x for x in choices if x != training_lengths[0]]
        length = int(rng.choice(choices))
        n = int(rng.choice([2, 3] if regime == 2 else [2, 3, 4, 5]))
        puzzle['train'].append(_make_pair(rng, length, n, i, seen))
        training_lengths.append(length)
    rng.shuffle(puzzle['train'])
    for i in range(num_test):
        if regime == 0:
            choices = range(10, 16)
        elif regime == 1:
            choices = range(3, 8)
        elif regime == 2:
            choices = range(3, 16)
        else:
            choices = range(5, 12)
        length = int(rng.choice(list(choices)))
        n = int(rng.choice([4, 5] if regime == 2 else [2, 3, 4, 5]))
        puzzle['test'].append(_make_pair(rng, length, n, 2, seen))
    validate_generated_puzzle(puzzle)
    return puzzle


def measure_pair(pair):
    """Small input-derived feature record; no grids or traces are retained."""
    p = _parse(pair['input'])
    out, owner = _paint(p)
    side = out.shape[0]
    rr, cc = np.indices(out.shape)
    geometric_depth = np.minimum.reduce([rr, cc, side-1-rr, side-1-cc])
    layers = p['code'][:-1]
    normalized = {}
    normalized_code = tuple(normalized.setdefault(int(c), len(normalized)) for c in layers)
    return dict(length=len(layers), n=p['n'], k=len(p['order']), width=p['w'],
                bg=p['bg'], sep=p['sep'],
                repeat=bool(np.any(layers[1:] == layers[:-1])),
                return_color=len(dict.fromkeys(map(int, layers))) < len([x for i,x in enumerate(layers) if i == 0 or x != layers[i-1]]),
                new_center=bool(layers[-1] not in layers[:-1]),
                revealed_pixels=int(np.sum((owner >= 0) & (owner < geometric_depth))),
                shared_target=len(set(p['targets'].values())) < len(p['targets']),
                duplicate_motif=len({m.tobytes() for m in p['masks'].values()}) < len(p['masks']),
                full_motif=any(m.all() for m in p['masks'].values()),
                density=round(float(np.mean([m.mean() for m in p['masks'].values()])), 4),
                code_signature=normalized_code)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            p = _parse(inp)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            expected, owner = _paint(p)
            assert np.array_equal(out, expected)
            assert np.array_equal(out, solve(inp))
            assert out.shape == (2*(len(p['code'])-1),)*2
            assert set(map(int, np.unique(out))) <= {p['bg'], *p['targets'].values()}
            # Recover actual visible ownership, including a revisited identity.
            visible = {int(p['code'][d]) for d in np.unique(owner) if d >= 0}
            assert visible == set(p['order'])
            # No parent input has this crossing-spine/row-code/library format.
            assert not np.array_equal(inp, out)
            if split == 'train':
                training.append((p, out))
    for alternative in _ALTERNATIVES+_ORIENTATIONS:
        assert any(_wrong(p, out, alternative) for p, out in training), alternative
    assert len({len(p['code']) for p, _ in training}) >= 2
    assert any(p['code'][-1] not in p['code'][:-2] for p, _ in training)
    assert any(p['code'][-1] in p['code'][:-2] for p, _ in training)
    # Every generated puzzle has a real extrapolation/interpolation of code
    # length, or a larger motif pitch after demonstrations at smaller pitches.
    t_lengths = {len(p['code'])-1 for p, _ in training}
    t_pitches = {p['n'] for p, _ in training}
    for pair in puzzle['test']:
        q = _parse(pair['input'])
        assert len(q['code'])-1 not in t_lengths or q['n'] > max(t_pitches)


def _signature(puzzle):
    return tuple(_key(p[k]) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output'))


def self_test():
    """42 streamed seeds, bounded counterfactuals, fixtures and count checks."""
    signatures, shapes, motifs, codes, pitches, counts, roles = (set() for _ in range(7))
    features = []
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        signatures.add(hashlib.sha256(repr(_signature(puzzle)).encode()).hexdigest())
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                p = _parse(pair['input'])
                f = measure_pair(pair)
                features.append(f)
                shapes.add(pair['input'].shape)
                motifs.update((p['n'], m.tobytes()) for m in p['masks'].values())
                codes.add((f['length'], f['code_signature']))
                pitches.add(f['n'])
                counts.add(f['k'])
                roles.add((f['bg'], f['sep']))
    assert len(signatures) == 42 and len(shapes) >= 30
    assert len(motifs) >= 150 and len(codes) >= 100
    assert pitches == {2, 3, 4, 5} and {2, 3, 4, 5} <= counts
    assert len({f['length'] for f in features}) >= 11 and len(roles) >= 40
    for name in ('repeat', 'return_color', 'new_center', 'shared_target', 'duplicate_motif', 'full_motif'):
        assert sum(bool(f[name]) for f in features) >= 5, name
    assert sum(f['revealed_pixels'] > 0 for f in features) >= 100
    assert max(f['density'] for f in features)-min(f['density'] for f in features) > .3
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            generate_puzzle(81000+10*a+b, a, b)
    for a, b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),
                 (4,False),(2.5,1),(4,1.5),(None,1),(4,None),('4',1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    assert _signature(generate_puzzle(12345, 10, 4)) == _signature(generate_puzzle(12345, 10, 4))
    # Hand-computed transparent-window fixture: holes in the inner anti-
    # diagonal reveal the outer diagonal, rather than becoming background.
    masks = {1: np.eye(2, dtype=bool), 2: ~np.eye(2, dtype=bool)}
    fixture = _encode(np.array([1, 2, 2]), masks, {1: 3, 2: 4}, [2, 1], 0, 5, 1)
    expected = np.array([[3,0,3,0],[0,3,4,3],[3,4,3,0],[0,3,0,3]])
    assert np.array_equal(solve(fixture), expected)
    wider = _encode(np.array([1,2,2]), masks, {1:3,2:4}, [2,1], 0,5,4)
    assert np.array_equal(solve(wider), expected)
    # A bijective palette relabeling must commute with the full solution.
    permutation = np.array([9,2,6,1,8,0,5,3,7,4])
    assert np.array_equal(solve(permutation[fixture]), permutation[expected])
    # Non-generated but valid library: no ink in the terminal 2x2 crop.
    sparse = np.zeros((5,5), dtype=bool)
    sparse[4,4] = True
    custom = _encode(np.array([1,2,2]), {1:sparse,2:sparse}, {1:3,2:4}, [1,2], 0,5,2)
    assert not solve(custom).any()
    # Exercise the specified family beyond the sampler's 2..6 identities:
    # a single identity and the full eight-label / width-30 boundary.
    single = _encode(np.array([1,1,1]), {1:np.ones((3,3),dtype=bool)}, {1:7}, [1], 0,5,3)
    assert np.array_equal(solve(single), np.full((4,4),7))
    boundary_masks = {i:np.array([[1,i%2],[(i//2)%2,(i//4)%2]],dtype=bool) for i in range(8)}
    boundary_code = np.array([0,1,2,3,4,5,6,7,0,2,4,6,1,3,5,5])
    boundary = _encode(boundary_code, boundary_masks, {i:(i+1)%9 for i in range(8)}, list(range(7,-1,-1)), 9,8,4)
    assert boundary.shape == (18,30)
    assert solve(boundary).shape == (30,30)
    assert np.array_equal(solve(boundary), _paint(_parse(boundary))[0])
    sample = generate_puzzle(24680, 2, 1)
    for kind in ('output', 'spine', 'key'):
        bad = {s: [{k: np.array(v, copy=True) for k,v in p.items()} for p in sample[s]] for s in sample}
        pair = bad['test'][0]
        if kind == 'output':
            pair['output'][0,0] = (int(pair['output'][0,0])+1) % 10
        elif kind == 'spine':
            a = pair['input']
            a[0, np.flatnonzero(a[0] != a[0,0])[0]] = a[0,0]
        else:
            a = pair['input']
            p = _parse(a)
            a[p['n']+2, p['w']+2] = p['bg']
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError('validator accepted '+kind+' corruption')
    return {'seeds':42, 'pairs':len(features), 'input_shapes':len(shapes),
            'motifs':len(motifs), 'normalized_codes':len(codes),
            'lengths':sorted({f['length'] for f in features}),
            'pitches':sorted(pitches), 'library_counts':sorted(counts),
            'palette_roles':len(roles),
            'transparent_examples':sum(f['revealed_pixels'] > 0 for f in features)}
