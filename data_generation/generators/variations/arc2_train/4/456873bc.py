import numpy as np


_SOURCE_ROLES = (0, 2, 3, 8)
_SOURCE_SPECS = {
    (3, 3, 'V', 2, (1, 1, 0, 1, 0, 1, 0, 1, 0)),
    (3, 3, 'V', 2, (1, 0, 1, 0, 1, 0, 1, 1, 0)),
    (4, 4, 'H', 1, (0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0)),
    (4, 4, 'H', 2, (1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1)),
}
_SHAPES = ((2, 2), (3, 3), (4, 4), (5, 5), (2, 3), (3, 2),
           (3, 4), (4, 3), (2, 4), (4, 2), (3, 5), (5, 3),
           (4, 5), (5, 4))


def _counts(seed, num_train, num_test):
    for value, name, low, high in ((num_train, 'num_train', 2, 10),
                                   (num_test, 'num_test', 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                  not isinstance(value, (int, np.integer)) or
                                  not low <= value <= high):
            raise ValueError(name)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _key(grid):
    return grid.shape, grid.astype(np.int8, copy=False).tobytes()


def _render(seed, roles, direction, band, row_gap, col_gap, row_pad, col_pad):
    background, foreground, mask, highlight = roles
    h, w = seed.shape
    height = h * h + (h - 1) * row_gap + 2 * row_pad
    width = w * w + (w - 1) * col_gap + 2 * col_pad
    clean = np.full((height, width), background, dtype=np.int8)
    tile = np.where(seed, foreground, background)
    output = clean.copy()
    for i, j in zip(*np.nonzero(seed)):
        r = row_pad + i * (h + row_gap)
        c = col_pad + j * (w + col_gap)
        clean[r:r+h, c:c+w] = tile
        output[r:r+h, c:c+w] = tile
        output[r+i, c+j] = highlight
    inp = clean.copy()
    if direction == 'H':
        r = row_pad + band * (h + row_gap)
        inp[r:r+h, :] = mask
    else:
        c = col_pad + band * (w + col_gap)
        inp[:, c:c+w] = mask
    return inp, output


def _geometry_options(side):
    return [(n, gap, pad) for n in range(2, 6)
            for gap in range(1, 3) for pad in range(3)
            if n*n + (n-1)*gap + 2*pad == side]


def _solutions(inp, highlight):
    height, width = inp.shape
    row_options = _geometry_options(height)
    col_options = _geometry_options(width)
    if not row_options or not col_options:
        return []
    colors = [int(x) for x in np.unique(inp)]
    if len(colors) != 3 or highlight in colors:
        return []
    answers = []
    for h, rg, rp in row_options:
        for w, cg, cp in col_options:
            for direction in ('H', 'V'):
                for band in range(h if direction == 'H' else w):
                    expected_mask = np.zeros(inp.shape, dtype=bool)
                    if direction == 'H':
                        start = rp + band * (h + rg)
                        expected_mask[start:start+h, :] = True
                    else:
                        start = cp + band * (w + cg)
                        expected_mask[:, start:start+w] = True
                    for mask in colors:
                        if not np.array_equal(inp == mask, expected_mask):
                            continue
                        others = [x for x in colors if x != mask]
                        for background, foreground in (others, others[::-1]):
                            patterns = []
                            for i in range(h):
                                for j in range(w):
                                    if (direction == 'H' and i == band) or (direction == 'V' and j == band):
                                        continue
                                    r = rp + i * (h + rg)
                                    c = cp + j * (w + cg)
                                    tile = inp[r:r+h, c:c+w]
                                    if not np.all((tile == background) | (tile == foreground)):
                                        break
                                    pattern = tile == foreground
                                    if pattern.any():
                                        patterns.append(pattern)
                                else:
                                    continue
                                break
                            else:
                                if len(patterns) < 2 or not all(np.array_equal(patterns[0], x) for x in patterns[1:]):
                                    continue
                                seed = patterns[0]
                                roles = (background, foreground, mask, highlight)
                                rebuilt, output = _render(seed, roles, direction, band, rg, cg, rp, cp)
                                if np.array_equal(rebuilt, inp):
                                    hidden = int(seed[band, :].sum() if direction == 'H' else seed[:, band].sum())
                                    answers.append((output, {'seed': seed, 'roles': roles,
                                        'direction': direction, 'band': band, 'h': h, 'w': w,
                                        'row_gap': rg, 'col_gap': cg, 'row_pad': rp, 'col_pad': cp,
                                        'density': int(seed.sum()), 'hidden': hidden}))
    return answers


def _infer_highlight(pair):
    input_colors = set(int(x) for x in np.unique(pair['input']))
    output_colors = set(int(x) for x in np.unique(pair['output']))
    added = output_colors - input_colors
    assert len(added) == 1
    return next(iter(added))


def _analyze(inp, highlight):
    answers = _solutions(inp, highlight)
    assert len(answers) == 1, 'input must uniquely identify roles, geometry, and motif'
    return answers[0]


def _motif(rng, h, w, kind):
    motif = np.zeros((h, w), dtype=bool)
    if kind == 'diagonal':
        for i in range(min(h, w)):
            motif[i, i] = True
        if h != w:
            motif[int(rng.integers(h)), int(rng.integers(w))] = True
    elif kind == 'bar':
        if rng.random() < .5:
            motif[int(rng.integers(h)), :] = True
        else:
            motif[:, int(rng.integers(w))] = True
    elif kind == 'cross':
        motif[int(rng.integers(h)), :] = True
        motif[:, int(rng.integers(w))] = True
    elif kind == 'ring':
        motif[0, :] = motif[-1, :] = True
        motif[:, 0] = motif[:, -1] = True
    elif kind == 'near_full':
        motif[:] = True
        holes = int(rng.integers(0, max(2, h*w//5 + 1)))
        if holes:
            motif.flat[rng.choice(h*w, size=holes, replace=False)] = False
    elif kind == 'empty_band':
        motif = rng.random((h, w)) < rng.uniform(.38, .75)
        if rng.random() < .5:
            motif[int(rng.integers(h)), :] = False
        else:
            motif[:, int(rng.integers(w))] = False
    elif kind == 'symmetric':
        motif = rng.random((h, w)) < rng.uniform(.3, .7)
        if rng.random() < .5:
            motif = motif | motif[:, ::-1]
        else:
            motif = motif | motif[::-1, :]
    else:
        probability = rng.uniform(.25, .83)
        motif = rng.random((h, w)) < probability
    if rng.random() < .5:
        motif = motif[::-1, :]
    if rng.random() < .5:
        motif = motif[:, ::-1]
    return motif


def _motif_classes(seed):
    h, w = seed.shape
    classes = set()
    if h == w and (np.array_equal(seed, np.eye(h, dtype=bool)) or
                   np.array_equal(seed, np.fliplr(np.eye(h, dtype=bool)))):
        classes.add('diagonal')
    if any(np.array_equal(seed, np.broadcast_to(np.arange(h)[:, None] == i, (h, w)))
           for i in range(h)) or any(
           np.array_equal(seed, np.broadcast_to(np.arange(w)[None, :] == j, (h, w)))
           for j in range(w)):
        classes.add('bar')
    if any(np.array_equal(seed, (np.arange(h)[:, None] == i) |
                          (np.arange(w)[None, :] == j))
           for i in range(h) for j in range(w)):
        classes.add('cross')
    ring = np.zeros((h, w), dtype=bool)
    ring[0, :] = ring[-1, :] = True
    ring[:, 0] = ring[:, -1] = True
    if np.array_equal(seed, ring):
        classes.add('ring')
    if 0 < h*w-int(seed.sum()) <= max(2, h*w//5):
        classes.add('near_full')
    return classes


def _source_copy(meta):
    return (meta['roles'] == _SOURCE_ROLES and meta['h'] == meta['w']
            and meta['row_gap'] == meta['col_gap'] == 1
            and meta['row_pad'] == meta['col_pad'] == 0
            and (meta['h'], meta['w'], meta['direction'], meta['band'],
                 tuple(int(x) for x in meta['seed'].flat)) in _SOURCE_SPECS)


def _check_output_cells(output, meta):
    seed = meta['seed']
    h, w = seed.shape
    background, foreground, mask, highlight = meta['roles']
    expected = np.full(output.shape, background, dtype=np.int8)
    for i in range(h):
        for j in range(w):
            if not seed[i, j]:
                continue
            top = meta['row_pad'] + i * (h + meta['row_gap'])
            left = meta['col_pad'] + j * (w + meta['col_gap'])
            for local_row in range(h):
                for local_col in range(w):
                    if not seed[local_row, local_col]:
                        continue
                    expected[top+local_row, left+local_col] = (
                        highlight if (local_row, local_col) == (i, j)
                        else foreground)
    assert np.array_equal(output, expected)
    assert np.count_nonzero(output == highlight) == int(seed.sum())
    assert not np.any(output == mask)


def _choose_shape(rng, mode, fixed_shape, split):
    if fixed_shape is not None:
        return fixed_shape
    if mode == 'source_like':
        return _SHAPES[int(rng.choice((1, 2)))]
    if mode == 'extrapolate':
        return _SHAPES[int(rng.choice((0, 1, 2, 4, 5, 6, 7)) if split == 'train'
                                  else rng.choice((3, 8, 9, 10, 11, 12, 13)))]
    return _SHAPES[int(rng.choice(len(_SHAPES), p=np.array(
        (8, 16, 17, 8, 8, 8, 7, 7, 4, 4, 4, 4, 3, 2)) / 100))]


def _make_pair(rng, highlight, palette, fixed_roles, direction, shape,
               mode, split, used_inputs, fixed_layout=None, special=False):
    h, w = shape
    for attempt in range(250):
        if fixed_layout is not None:
            rg, cg, rp, cp = fixed_layout
        elif mode == 'source_like':
            rg = cg = 1
            rp = cp = 0
        else:
            rg = 1 if rng.random() < .7 else 2
            cg = 1 if rng.random() < .7 else 2
            rp = int(rng.choice((0, 1, 2), p=(.65, .25, .10)))
            cp = int(rng.choice((0, 1, 2), p=(.65, .25, .10)))
        if h*h + (h-1)*rg + 2*rp > 30 or w*w + (w-1)*cg + 2*cp > 30:
            continue
        if special and mode not in ('source_like', 'uniform_size') and \
                rg == cg == 1 and rp == cp == 0:
            continue
        band = int(rng.integers(h if direction == 'H' else w))
        kinds = ('random', 'random', 'random', 'diagonal', 'bar', 'cross',
                 'ring', 'symmetric', 'near_full', 'empty_band')
        kind = 'random' if special else kinds[int(rng.integers(len(kinds)))]
        motif = _motif(rng, h, w, kind)
        density = int(motif.sum())
        hidden = int(motif[band, :].sum() if direction == 'H' else motif[:, band].sum())
        if density < 3 or density - hidden < 2:
            continue
        if special and (hidden == 0 or density == h*w or
                        np.array_equal(motif, motif[::-1, :]) or
                        np.array_equal(motif, motif[:, ::-1])):
            continue
        if fixed_roles is None:
            colors = [int(x) for x in rng.choice([x for x in range(10) if x != highlight],
                                                  size=3, replace=False)]
            roles = (*colors, highlight)
            if palette is not None and roles[:3] == palette:
                continue
        else:
            roles = fixed_roles
        inp, out = _render(motif, roles, direction, band, rg, cg, rp, cp)
        if _key(inp) in used_inputs:
            continue
        answers = _solutions(inp, highlight)
        if len(answers) != 1 or not np.array_equal(answers[0][0], out):
            continue
        if _source_copy(answers[0][1]):
            continue
        used_inputs.add(_key(inp))
        return {'input': inp, 'output': out}
    raise RuntimeError('could not construct a unique pair')


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(seed, num_train, num_test)
    highlight = int(rng.integers(10))
    mode = str(rng.choice(('source_like', 'general', 'general', 'general',
                           'uniform_size', 'extrapolate')))
    direction_mode = str(rng.choice(('mixed', 'mixed', 'consistent', 'holdout')))
    base_direction = 'H' if rng.random() < .5 else 'V'
    fixed_shape = _SHAPES[int(rng.choice((1, 2, 3, 6, 7, 10, 11, 12, 13)))] if mode == 'uniform_size' else None
    fixed_layout = None
    if fixed_shape is not None:
        h, w = fixed_shape
        layouts = [(rg, cg, rp, cp) for rg in (1, 2) for cg in (1, 2)
                   for rp in (0, 1, 2) for cp in (0, 1, 2)
                   if h*h+(h-1)*rg+2*rp <= 30 and w*w+(w-1)*cg+2*cp <= 30]
        fixed_layout = layouts[int(rng.integers(len(layouts)))]
    fixed_roles = None
    if rng.random() < .25:
        colors = [int(x) for x in rng.choice([x for x in range(10) if x != highlight],
                                              size=3, replace=False)]
        fixed_roles = (*colors, highlight)
    train_directions = []
    test_directions = []
    if direction_mode == 'mixed':
        train_directions = [base_direction, 'V' if base_direction == 'H' else 'H']
        train_directions += ['H' if rng.random() < .5 else 'V' for _ in range(train_count-2)]
        test_directions = ['H' if rng.random() < .5 else 'V' for _ in range(test_count)]
    else:
        train_directions = [base_direction] * train_count
        test_direction = ('V' if base_direction == 'H' else 'H') if direction_mode == 'holdout' else base_direction
        test_directions = [test_direction] * test_count
    pair_seeds = rng.integers(0, np.iinfo(np.uint64).max,
                              size=train_count+test_count, dtype=np.uint64)
    puzzle = {'train': [], 'test': []}
    used_inputs = set()
    first_palette = None
    for split, count, directions, offset in (('train', train_count, train_directions, 0),
                                              ('test', test_count, test_directions, train_count)):
        for index in range(count):
            pair_rng = np.random.default_rng(pair_seeds[offset+index])
            shape = _choose_shape(pair_rng, mode, fixed_shape, split)
            if split == 'train' and index == 0:
                if mode in ('general', 'extrapolate'):
                    shape = _SHAPES[int(pair_rng.choice(
                        (6, 7, 10, 11, 12, 13) if mode == 'general' else (6, 7)))]
                elif min(shape) < 3:
                    shape = _SHAPES[int(pair_rng.choice((1, 2, 6, 7)))]
            palette_exclusion = first_palette if fixed_roles is None and index == 1 and split == 'train' else None
            pair = _make_pair(pair_rng, highlight, palette_exclusion, fixed_roles,
                              directions[index], shape, mode, split, used_inputs,
                              fixed_layout=fixed_layout,
                              special=(split == 'train' and index == 0 and min(shape) >= 3))
            if split == 'train' and index == 0:
                answer = _analyze(pair['input'], highlight)
                first_palette = answer[1]['roles'][:3]
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    used_inputs, used_pairs = set(), set()
    highlights = []
    for pair in puzzle['train']:
        assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
        highlights.append(_infer_highlight(pair))
    assert len(set(highlights)) == 1
    highlight = highlights[0]
    train_metadata, test_metadata = [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in (pair['input'], pair['output']):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((grid >= 0) & (grid <= 9))
            inp, out = pair['input'], pair['output']
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            key = _key(inp)
            pair_key = (key, _key(out))
            assert key not in used_inputs and pair_key not in used_pairs
            used_inputs.add(key)
            used_pairs.add(pair_key)
            predicted, meta = _analyze(inp, highlight)
            assert np.array_equal(predicted, out)
            _check_output_cells(out, meta)
            assert not _source_copy(meta)
            if split == 'train':
                train_metadata.append(meta)
            else:
                test_metadata.append(meta)
    assert train_metadata and test_metadata
    first = train_metadata[0]
    assert first['hidden'] > 0 and not first['seed'].all()
    assert not np.array_equal(first['seed'], first['seed'][::-1, :])
    assert not np.array_equal(first['seed'], first['seed'][:, ::-1])
    assert all(meta['density'] >= 3 and meta['density']-meta['hidden'] >= 2
               for meta in train_metadata+test_metadata)
    assert all(meta['roles'][3] == highlight for meta in train_metadata+test_metadata)


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({tuple(_key(x['input']) for split in ('train', 'test') for x in p[split])
                for p in puzzles}) == 42
    shapes, directions, bands, gaps, pads, densities, motifs = set(), set(), set(), set(), set(), set(), set()
    classes = {'empty': 0, 'hidden_zero': 0, 'full': 0, 'sparse': 0, 'dense': 0}
    varied_palette = mixed_train = mixed_test = constant_direction = holdout = 0
    uniform_size = holdout_hv = holdout_vh = 0
    motif_classes = set()
    square_bands = set()
    equal_density = sparse_test = dense_train = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        highlight = _infer_highlight(puzzle['train'][0])
        train = [_analyze(pair['input'], highlight)[1] for pair in puzzle['train']]
        test = [_analyze(pair['input'], highlight)[1] for pair in puzzle['test']]
        if len({m['roles'][:3] for m in train+test}) > 1:
            varied_palette += 1
        td = {m['direction'] for m in train}
        sd = {m['direction'] for m in test}
        mixed_train += len(td) > 1
        mixed_test += len(sd) > 1
        constant_direction += len(td | sd) == 1
        holdout += len(td) == len(sd) == 1 and td != sd
        holdout_hv += td == {'H'} and sd == {'V'}
        holdout_vh += td == {'V'} and sd == {'H'}
        uniform_size += len({pair['input'].shape for split in ('train', 'test')
                             for pair in puzzle[split]}) == 1
        equal_density += bool({m['density'] for m in train} & {m['density'] for m in test})
        sparse_test += any(m['density'] <= m['h']*m['w']//2 for m in test)
        dense_train += any(m['density'] >= 3*m['h']*m['w']//4 for m in train)
        for m in train+test:
            h, w = m['h'], m['w']
            s = m['seed']
            shapes.add((h, w))
            directions.add(m['direction'])
            bands.add((m['direction'], m['band'], h if m['direction'] == 'H' else w))
            gaps.add((m['row_gap'], m['col_gap']))
            pads.add((m['row_pad'], m['col_pad']))
            densities.add((h, w, m['density']))
            motifs.add((h, w, tuple(int(v) for v in s.flat)))
            motif_classes.update(_motif_classes(s))
            if h == w and h in (3, 4):
                square_bands.add((h, m['direction'], m['band']))
            classes['empty'] += not s.any(axis=0).all() or not s.any(axis=1).all()
            classes['hidden_zero'] += m['hidden'] == 0
            classes['full'] += s.all()
            classes['sparse'] += m['density'] <= h*w//2
            classes['dense'] += m['density'] >= 3*h*w//4
    assert len(shapes) >= 8 and (2, 2) in shapes and (5, 5) in shapes
    assert directions == {'H', 'V'} and len(bands) >= 12
    assert len(gaps) >= 3 and len(pads) >= 3
    assert len(densities) >= 30 and len(motifs) >= 90
    assert motif_classes == {'diagonal', 'bar', 'cross', 'ring', 'near_full'}
    assert all((n, direction, band) in square_bands
               for n in (3, 4) for direction in ('H', 'V') for band in range(n))
    assert all(value > 0 for value in classes.values())
    assert varied_palette >= 15 and mixed_train >= 5
    assert constant_direction >= 3 and holdout >= 3
    assert mixed_test >= 3 and uniform_size >= 3
    assert holdout_hv >= 2 and holdout_vh >= 2
    assert equal_density >= 2 and sparse_test >= 5 and dense_train >= 5
    for seed, ntrain, ntest in ((7654, 6, 2), (8000, 2, 1), (8001, 10, 1), (8002, 10, 4),
                                (9000, None, 1), (9001, None, 4), (9002, 2, None), (9003, 10, None)):
        p = generate_puzzle(seed, ntrain, ntest)
        validate_generated_puzzle(p)
        if ntrain is not None:
            assert len(p['train']) == ntrain
        if ntest is not None:
            assert len(p['test']) == ntest
    for request in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0}, {'num_test': 5},
                    {'num_train': 2, 'num_test': 2}, {'num_train': 3, 'num_test': 4}):
        try:
            generate_puzzle(77, **request)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for args in ((123456, 7, 3), (17, None, None)):
        a, b = generate_puzzle(*args), generate_puzzle(*args)
        assert all(np.array_equal(x[role], y[role])
                   for split in ('train', 'test') for x, y in zip(a[split], b[split])
                   for role in ('input', 'output'))
