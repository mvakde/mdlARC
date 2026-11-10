import hashlib

import numpy as np


# SHA-256 of the six original input grids in row-major uint8 form.  Shapes are
# retained so source dimensions remain available for new configurations.
_SOURCE_INPUTS = {
    ((12, 12), 'bee19007bbf98847fe46e5907c22f3b5874f7a95ea1a26132fb27e950ce9eaed'),
    ((10, 10), '68e92f744de3850d84156741299334bbb65d37085698d65bf8e1336f144b91d8'),
    ((12, 12), '7b7852939da39be12cef7876ec03bbf162270421bb1e1256fb28922556020cba'),
    ((15, 15), 'f559d89b8da777d1bbfed99af65aa304fe130c056ffd61d2658c825fc6efe472'),
    ((25, 15), '0e600bcc46fc123ef95683fef75973475b5958f75f7e3e1c0530a55be18daf24'),
    ((10, 10), '8558df4b06389fc39cbd3af1873593ef8583c2924b4637bd98b5dfc2b66f5678'),
}


def _source_key(grid):
    return (grid.shape, hashlib.sha256(grid.astype(np.uint8).tobytes()).hexdigest())


def _count(name, value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if name == 'num_train' and not 2 <= value <= 10:
        raise ValueError('num_train must be in 2..10')
    if name == 'num_test' and not 1 <= value <= 4:
        raise ValueError('num_test must be in 1..4')
    return value


def _resolve_counts(rng, num_train, num_test):
    num_train, num_test = _count('num_train', num_train), _count('num_test', num_test)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return num_train, num_test


def _choice(rng, values):
    return values[int(rng.integers(0, len(values)))]


def _balanced(rng, values, count):
    result = [values[i % len(values)] for i in range(count)]
    rng.shuffle(result)
    return result


def _dimensions(rng, split):
    band = _choice(rng, ('compact',) * (6 if split == 'train' else 4)
                   + ('medium',) * 2 + ('large',) * (2 if split == 'train' else 4))
    if band == 'compact':
        low, high = 10, 16
    elif band == 'medium':
        low, high = 14, 23
    else:
        low, high = 20, 31
    return int(rng.integers(low, high)), int(rng.integers(10 if band == 'compact' else low, high))


def _end(rng, nearest, outer):
    # The endpoint, a canvas edge, and intermediate positions are all possible.
    mode = int(rng.integers(0, 4))
    if mode == 0:
        return nearest
    if mode == 1:
        return outer
    return int(rng.integers(min(nearest, outer), max(nearest, outer) + 1))


def _sample_spec(rng, split, controls=None):
    direction, phase, parity, relation = controls or (
        _choice(rng, (-1, 1)), _choice(rng, ('L', 'T')),
        _choice(rng, (0, 1)), _choice(rng, ('left', 'right', 'straddle')),
    )
    for _ in range(200):
        h, w = _dimensions(rng, split)
        far_low = 4 if split == 'train' else 4
        far_high = min(h - 2, 18 if split == 'train' else 23)
        if far_high < far_low:
            continue
        far = int(rng.integers(far_low, far_high + 1))
        near_options = [d for d in range(1, far) if d % 2 == parity]
        near = int(_choice(rng, near_options))
        r = int(rng.integers(far, h)) if direction == -1 else int(rng.integers(0, h - far))
        endpoint = r + direction * far
        top, bottom = min(r, endpoint), max(r, endpoint)
        v0, v1 = _end(rng, top, 0), _end(rng, bottom, h - 1)
        if v1 - v0 + 1 < 7:
            v0, v1 = 0, h - 1

        max_span = min(w - 2, 11 if split == 'train' else 12)
        if max_span < 3:
            continue
        span = int(rng.integers(3, max_span + 1))
        if relation == 'right':
            qlo = int(rng.integers(1, w - span + 1))
            qhi = qlo + span - 1
            c = int(rng.integers(0, qlo))
        elif relation == 'left':
            qlo = int(rng.integers(0, w - span))
            qhi = qlo + span - 1
            c = int(rng.integers(qhi + 1, w))
        else:
            qlo = int(rng.integers(0, w - span + 1))
            qhi = qlo + span - 1
            c = int(rng.integers(qlo + 1, qhi))
        left, right = min(c, qlo), max(c, qhi)
        h0, h1 = _end(rng, left, 0), _end(rng, right, w - 1)
        if h1 - h0 + 1 < 7:
            h0, h1 = 0, w - 1
        return dict(h=h, w=w, r=r, c=c, v0=v0, v1=v1, h0=h0, h1=h1,
                    direction=direction, near=near, far=far, qlo=qlo, qhi=qhi, phase=phase)
    raise ValueError('no valid geometry')


def _palette(rng, anchor, mode):
    if mode == 'simple' or int(rng.integers(0, 3)) != 0:
        background, pool = anchor
        if int(rng.integers(0, 5)) < 2:
            order = pool
        else:
            order = tuple(int(x) for x in rng.permutation(pool))
    else:
        colors = tuple(int(x) for x in rng.choice(10, size=5, replace=False))
        background, order = colors[0], colors[1:]
    return dict(background=background, L=order[0], P=order[1], T=order[2], Q=order[3])


def _build_input(spec, palette):
    grid = np.full((spec['h'], spec['w']), palette['background'], dtype=np.int8)
    r, c = spec['r'], spec['c']
    grid[spec['v0']:spec['v1'] + 1, c] = palette['L']
    grid[r, spec['h0']:spec['h1'] + 1] = palette['T']
    grid[r + spec['direction'] * spec['near'], c] = palette['P']
    grid[r + spec['direction'] * spec['far'], c] = palette['P']
    grid[r, spec['qlo']] = palette['Q']
    grid[r, spec['qhi']] = palette['Q']
    grid[r, c] = palette[spec['phase']]
    return grid


def _base_marker(values):
    colors, counts = np.unique(values, return_counts=True)
    assert len(colors) == 2
    marker = np.flatnonzero(counts == 2)
    assert len(marker) == 1
    mi = int(marker[0])
    assert counts[1 - mi] >= 3
    return int(colors[1 - mi]), int(colors[mi])


def _analyze_input(grid):
    h, w = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    assert len(colors) == 5 and np.count_nonzero(counts == counts.max()) == 1
    background = int(colors[int(np.argmax(counts))])
    occupied = grid != background
    row_counts, col_counts = occupied.sum(axis=1), occupied.sum(axis=0)
    assert np.count_nonzero(row_counts == row_counts.max()) == 1
    assert np.count_nonzero(col_counts == col_counts.max()) == 1
    r, c = int(np.argmax(row_counts)), int(np.argmax(col_counts))
    vr, hc = np.flatnonzero(occupied[:, c]), np.flatnonzero(occupied[r, :])
    assert len(vr) >= 7 and len(hc) >= 7
    assert np.all(np.diff(vr) == 1) and np.all(np.diff(hc) == 1)
    v0, v1, h0, h1 = int(vr[0]), int(vr[-1]), int(hc[0]), int(hc[-1])
    expected = np.zeros_like(occupied)
    expected[v0:v1 + 1, c] = True
    expected[r, h0:h1 + 1] = True
    assert np.array_equal(occupied, expected)
    l, p = _base_marker(np.array([grid[y, c] for y in vr if y != r]))
    t, q = _base_marker(np.array([grid[r, x] for x in hc if x != c]))
    assert len({background, l, p, t, q}) == 5
    x = int(grid[r, c])
    assert x in (l, t)
    p_rows = np.flatnonzero(grid[:, c] == p)
    assert len(p_rows) == 2
    deltas = [int(y) - r for y in p_rows]
    assert deltas[0] * deltas[1] > 0
    direction = 1 if deltas[0] > 0 else -1
    near, far = sorted(map(abs, deltas))
    assert 1 <= near < far and far >= 4
    q_cols = np.flatnonzero(grid[r, :] == q)
    assert len(q_cols) == 2
    qlo, qhi = map(int, q_cols)
    assert qlo < qhi and qhi - qlo + 1 >= 3 and c not in (qlo, qhi)
    relation = 'right' if c < qlo else 'left' if qhi < c else 'straddle'
    return dict(h=h, w=w, background=background, l=l, p=p, t=t, q=q, x=x,
                r=r, c=c, v0=v0, v1=v1, h0=h0, h1=h1, direction=direction,
                near=near, far=far, qlo=qlo, qhi=qhi, relation=relation)


def _render(analysis):
    a = analysis
    output = np.full((a['h'], a['w']), a['background'], dtype=np.int8)
    for d in range(a['far'] + 1):
        row = a['r'] + a['direction'] * d
        for col in range(a['qlo'], a['qhi'] + 1):
            s = col - a['qlo']
            if s % 2 == 0:
                value = a['t'] if d % 2 else a['q'] if d < a['near'] else a['p']
            else:
                value = a['x'] if d % 2 or d == a['near'] else a['l']
            output[row, col] = value
    return output


def _pair(spec, palette):
    input_grid = _build_input(spec, palette)
    return {'input': input_grid, 'output': _render(_analyze_input(input_grid))}


def _fingerprint(grid):
    return (grid.shape, grid.tobytes())


def _validate(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[k], list) for k in ('train', 'test'))
    n, m = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= n <= 10 and 1 <= m <= 4 and n > m
    seen = set()
    analyses = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            for grid in (inp, out):
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= side <= 30 for side in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) <= int(grid.max()) <= 9
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            fp = _fingerprint(inp)
            assert fp not in seen and _source_key(inp) not in _SOURCE_INPUTS
            seen.add(fp)
            a = _analyze_input(inp)
            assert np.array_equal(out, _render(a))
            assert set(map(int, np.unique(out))) == {a['background'], a['l'], a['p'], a['t'], a['q']}
            endpoint = a['r'] + a['direction'] * a['far']
            assert a['v0'] <= endpoint <= a['v1']
            assert a['h0'] <= a['qlo'] < a['qhi'] <= a['h1']
            analyses[split].append(a)
    # The transformation has no hidden color or geometry convention. Show both
    # crossing phases in training, and vary the other observable controls.
    phases = {a['x'] == a['l'] for a in analyses['train']}
    assert phases == {False, True}
    assert {a['direction'] for a in analyses['train']} == {-1, 1}
    assert {a['near'] % 2 for a in analyses['train']} == {0, 1}
    assert len({a['relation'] for a in analyses['train']}) >= 2


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        _validate(puzzle)
    except AssertionError:
        raise
    except Exception as exc:
        raise AssertionError('invalid generated puzzle') from exc


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    n, m = _resolve_counts(rng, num_train, num_test)
    colors = tuple(int(x) for x in rng.choice(10, size=5, replace=False))
    anchor = (colors[0], colors[1:])
    mode = _choice(rng, ('simple', 'mixed'))
    puzzle = {'train': [], 'test': []}
    seen = set()
    independent = n - (1 if n >= 3 else 0) - (1 if n >= 5 else 0)
    directions = _balanced(rng, (-1, 1), independent)
    phases = _balanced(rng, ('L', 'T'), independent)
    parities = _balanced(rng, (0, 1), independent)
    relations = _balanced(rng, ('left', 'right', 'straddle'), independent)
    controlled_sources = []
    for i in range(n + m):
        split = 'train' if i < n else 'test'
        controlled = split == 'train' and i >= independent
        for attempt in range(250):
            if controlled:
                source_spec, palette = controlled_sources[0]
                spec = dict(source_spec)
                if i == independent:
                    spec['phase'] = 'T' if spec['phase'] == 'L' else 'L'
                else:
                    choices = [d for d in range(1, spec['far']) if d != spec['near']]
                    spec['near'] = int(_choice(rng, choices))
            else:
                controls = (directions[i], phases[i], parities[i], relations[i]) if split == 'train' else None
                spec = _sample_spec(rng, split, controls)
                palette = _palette(rng, anchor, mode)
            pair = _pair(spec, palette)
            fp = _fingerprint(pair['input'])
            if fp not in seen and _source_key(pair['input']) not in _SOURCE_INPUTS:
                break
        else:
            raise ValueError('could not construct a unique pair')
        seen.add(fp)
        puzzle[split].append(pair)
        if split == 'train' and not controlled:
            controlled_sources.append((spec, palette))
    validate_generated_puzzle(puzzle)
    return puzzle


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(
        np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s]) for k in ('input', 'output'))
        for s in ('train', 'test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert _equal(generate_puzzle(123, 7, 3), generate_puzzle(123, 7, 3))
    assert not _equal(generate_puzzle(123, 7, 3), generate_puzzle(124, 7, 3))
    for n, m in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(9000 + n + m, n, m)
        assert (len(p['train']), len(p['test'])) == (n, m)
    for n, m in ((None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(9100 + (n or 0) + (m or 0), n, m))
    for n, m in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4), (4, 4)):
        try:
            generate_puzzle(777, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('expected ValueError')
    all_analyses = [(s, _analyze_input(pair['input'])) for p in puzzles
                    for s in ('train', 'test') for pair in p[s]]
    changed_background_puzzles = 0
    repeated_palette_puzzles = 0
    controlled_phase_puzzles = 0
    controlled_switch_puzzles = 0
    for puzzle in puzzles:
        aa = [_analyze_input(pair['input']) for split in ('train', 'test') for pair in puzzle[split]]
        train = [_analyze_input(pair['input']) for pair in puzzle['train']]
        assert {a['direction'] for a in train} == {-1, 1}
        assert {a['near'] % 2 for a in train} == {0, 1}
        assert {a['x'] == a['l'] for a in train} == {False, True}
        assert len({a['relation'] for a in train}) >= 2
        palettes = [(a['background'], a['l'], a['p'], a['t'], a['q']) for a in aa]
        changed_background_puzzles += len({a['background'] for a in aa}) > 1
        repeated_palette_puzzles += len(set(palettes)) < len(palettes)
        for i, a in enumerate(train):
            for b in train[i + 1:]:
                geometry = ('h', 'w', 'r', 'c', 'v0', 'v1', 'h0', 'h1',
                            'direction', 'far', 'qlo', 'qhi')
                if all(a[k] == b[k] for k in geometry) and \
                        (a['background'], a['l'], a['p'], a['t'], a['q']) == \
                        (b['background'], b['l'], b['p'], b['t'], b['q']):
                    controlled_phase_puzzles += a['near'] == b['near'] and a['x'] != b['x']
                    controlled_switch_puzzles += a['near'] != b['near'] and a['x'] == b['x']
    assert changed_background_puzzles >= 5
    assert repeated_palette_puzzles >= 20
    assert controlled_phase_puzzles >= 20
    assert controlled_switch_puzzles >= 10
    for split in ('train', 'test'):
        aa = [a for s, a in all_analyses if s == split]
        assert any(a['near'] == 1 for a in aa)
        assert any(a['near'] % 2 for a in aa)
        assert {a['relation'] for a in aa} == {'left', 'right', 'straddle'}
        assert {a['x'] == a['l'] for a in aa} == {False, True}
        assert {a['direction'] for a in aa} == {-1, 1}
        assert any(a['h'] == a['w'] for a in aa)
        assert any(a['v0'] > 0 or a['v1'] < a['h'] - 1 for a in aa)
        assert any(a['h0'] > 0 or a['h1'] < a['w'] - 1 for a in aa)
        assert any(a['r'] + a['direction'] * a['far'] in (0, a['h'] - 1) for a in aa)
        assert any(a['far'] == (a['v1'] - a['r'] if a['direction'] == 1 else a['r'] - a['v0']) for a in aa)
        assert any(a['qhi'] - a['qlo'] + 1 == 6 for a in aa)
        assert any(a['qhi'] - a['qlo'] + 1 >= 7 for a in aa)
        assert any(a['h0'] == 0 and a['h1'] == a['w'] - 1 and
                   a['v0'] == 0 and a['v1'] == a['h'] - 1 for a in aa)
        assert any(a['h0'] > 0 and a['h1'] < a['w'] - 1 and
                   a['v0'] > 0 and a['v1'] < a['h'] - 1 for a in aa)
        assert any((a['v0'] == 0 and a['v1'] == a['h'] - 1) !=
                   (a['h0'] == 0 and a['h1'] == a['w'] - 1) for a in aa)
        assert any(a['relation'] != 'straddle' and
                   (a['qlo'] - a['c'] if a['relation'] == 'right' else
                    a['c'] - a['qhi']) >= 3 for a in aa)
        assert any(a['near'] == 1 and a['qhi'] - a['qlo'] + 1 >= 7 and
                   (a['v0'] > 0 or a['v1'] < a['h'] - 1 or
                    a['h0'] > 0 or a['h1'] < a['w'] - 1) for a in aa)
    assert len({(a['h'], a['w']) for _, a in all_analyses}) >= 35
    assert len({(a['h'], a['w'], a['far'], a['qlo'], a['qhi'], a['near']) for _, a in all_analyses}) >= 180
