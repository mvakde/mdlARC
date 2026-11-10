"""Generate pair-local ordered color codebooks and decoded motif tilings."""
import hashlib
import numpy as np


_SOURCE_INPUT_HASHES = frozenset({
    '3a6e4ec438a372df0e9e86e72cb4feb5149aaf000bd62ee0b4176be3a21bbf4b',
    '5d7e7f32dfaab52ad172685ea530e791f936e4c062f427a8041e61bcc7ae554d',
    '06679148a25c790e4853530a85ebeebf54dd576d334394f5421d984b3dc79e6b',
    'fa59661fd7685970c7c377a245332b96c9eb84dd0050f0f2a387594373b1577b',
    'fb7e481fd96e65761ac59bb247d10437751b10dbf21f004d7bd57cd556cf3c5e',
})


def _counts(rng, a, b):
    for value, name, lo, hi in ((a, 'num_train', 2, 10), (b, 'num_test', 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi):
            raise ValueError(name)
    if a is None and b is None:
        b = int(rng.integers(1, 5))
        a = int(rng.integers(max(2, b + 1), 11))
    elif a is None:
        a = int(rng.integers(max(2, int(b) + 1), 11))
    elif b is None:
        b = int(rng.integers(1, min(4, int(a) - 1) + 1))
    if int(a) <= int(b):
        raise ValueError('num_train must exceed num_test')
    return int(a), int(b)


def _asymmetric(m):
    variants = [np.flipud(m), np.fliplr(m), np.rot90(m, 2)]
    if m.shape[0] == m.shape[1]:
        variants += [np.rot90(m), np.rot90(m, 3), m.T]
    return all(not np.array_equal(m, v) for v in variants)


def _components(m):
    remaining = set(map(tuple, np.argwhere(m)))
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            y, x = stack.pop()
            for neighbor in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
    return count


def _motif_candidate(rng, h, w):
    yy, xx = np.indices((h, w))
    mode = int(rng.integers(10))
    if mode == 0:  # near-filled, including source-style 3x3 masks
        m = np.ones((h, w), dtype=bool)
        holes = int(rng.integers(1, max(2, h * w // 4 + 1)))
        m.flat[rng.choice(h * w, holes, replace=False)] = False
    elif mode == 1:
        m = rng.random((h, w)) < float(rng.uniform(.22, .85))
    elif mode == 2:  # closed or open frame
        m = (yy == 0) | (yy == h - 1) | (xx == 0) | (xx == w - 1)
        m.flat[int(rng.integers(h * w))] = False
    elif mode == 3:  # steps
        phase = int(rng.integers(max(h, w)))
        m = ((xx + yy + phase) % int(rng.integers(2, 4))) == 0
        m |= yy == (xx + phase) % h
    elif mode == 4:  # reflection
        half = rng.random((h, (w + 1) // 2)) < float(rng.uniform(.35, .8))
        m = np.concatenate((half, np.fliplr(half[:, :w // 2])), axis=1)
    elif mode == 5:  # crossing strokes
        m = (yy == xx * h // w) | (yy == h - 1 - xx * h // w)
        m |= xx == int(rng.integers(w))
    elif mode == 6:  # row-column bars
        m = (yy == int(rng.integers(h))) | (xx == int(rng.integers(w)))
        m |= rng.random((h, w)) < .12
    elif mode == 7:
        m = ((yy + 2 * xx + int(rng.integers(4))) % 4) < 2
        m ^= rng.random((h, w)) < .15
    else:  # thin stepped or diagonal paths that still expose every slot edge
        m = np.zeros((h, w), dtype=bool)
        for row in range(h):
            m[row, row * w // h] = True
        for col in range(w):
            m[col * h // w, col] = True
        if mode == 9:
            m[int(rng.integers(h)), int(rng.integers(w))] = True
        if rng.random() < .5:
            m = np.fliplr(m)
    return m.astype(np.int8)


def _motifs(rng, n, h, w, force_asym):
    motifs, seen = [], set()
    for _ in range(3000):
        m = _motif_candidate(rng, h, w)
        if not (np.any(m) and np.any(m == 0)):
            continue
        # Every row and column has visible support, fixing common slot bounds.
        if not (np.all(m.any(axis=0)) and np.all(m.any(axis=1))):
            continue
        if m.tobytes() in seen or (force_asym and not motifs and not _asymmetric(m)):
            continue
        seen.add(m.tobytes())
        motifs.append(m)
        if len(motifs) == n:
            return tuple(motifs)
    raise RuntimeError('motif construction failed')


def _matrix(rng, palette, h, w):
    n = len(palette)
    r, c = np.indices((h, w))
    mode, phase = int(rng.integers(8)), int(rng.integers(n))
    if mode == 0:  # uneven frequencies
        p = rng.dirichlet(np.full(n, float(rng.uniform(.3, 2.5))))
        ids = rng.choice(n, (h, w), p=p)
    elif mode == 1:
        ids = (r + phase) % n
    elif mode == 2:
        ids = (c + phase) % n
    elif mode == 3:
        ids = (r + int(rng.integers(1, 4)) * c + phase) % n
    elif mode == 4:  # patch
        ids = rng.integers(n, size=(h, w))
        r0, c0 = int(rng.integers(h - 1)), int(rng.integers(w - 1))
        ids[r0:int(rng.integers(r0 + 2, h + 1)), c0:int(rng.integers(c0 + 2, w + 1))] = phase
    elif mode == 5:  # border
        ids = rng.integers(n, size=(h, w))
        ids[0, :] = ids[-1, :] = ids[:, 0] = ids[:, -1] = phase
    elif mode == 6:  # repeating tile
        tile = rng.integers(n, size=(int(rng.integers(2, 4)), int(rng.integers(2, 4))))
        ids = tile[r % tile.shape[0], c % tile.shape[1]]
    else:  # alternating rows plus a solid row
        ids = (r + c + phase) % n
        ids[int(rng.integers(h)), :] = int(rng.integers(n))
    ids = np.array(ids, dtype=np.int8, copy=True)
    if len(np.unique(ids)) < 2:
        ids[int(rng.integers(h)), int(rng.integers(w))] = (int(ids[0, 0]) + 1) % n
    return np.asarray(palette, dtype=np.int8)[ids]


def _expand(matrix, palette, motifs, bg):
    mh, mw = motifs[0].shape
    out = np.full((matrix.shape[0] * mh, matrix.shape[1] * mw), bg, dtype=np.int8)
    lookup = {int(color): motif for color, motif in zip(palette, motifs)}
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            color = int(matrix[r, c])
            out[r * mh:(r + 1) * mh, c * mw:(c + 1) * mw] = np.where(lookup[color], color, bg)
    return out


def _dimensions(rng, regime, count, force_asym, direction):
    for _ in range(100):
        if regime == 'source_train':
            mh, mw = ((int(rng.choice([3, 5])),) * 2 if rng.random() < .48
                      else (int(rng.integers(2, 6)), int(rng.integers(2, 6))))
            sh, sw = int(rng.integers(3, 8)), int(rng.integers(3, 8))
        else:
            mh, mw = ((int(rng.choice([3, 6])),) * 2 if rng.random() < .38
                      else (int(rng.integers(2, 7)), int(rng.integers(2, 7))))
            sh, sw = int(rng.integers(3, 9)), int(rng.integers(3, 9))
        if force_asym and mh == mw == 2:
            continue
        if (sh * mh <= 30 and sw * mw <= 30 and count * mw + count - 1 <= 28
                and count * mh + count - 1 <= 28
                and (direction != 'side' or count * mw + 4 * count + 1 <= 30)):
            return mh, mw, sh, sw
    return 3, 3, 4, 4


def _layout(rng, direction, palette, motifs, matrix, bg, marker):
    n, (mh, mw) = len(palette), motifs[0].shape
    sh = int(rng.integers(2, min(3, mh) + 1))
    sw = 2 if direction == 'side' else int(rng.integers(2, min(3, mw) + 1))
    for _ in range(80):
        gap, band_gap, matrix_gap = int(rng.integers(1, 3)), int(rng.integers(1, 4)), int(rng.integers(1, 4))
        if direction == 'horizontal':
            cat_w = n * mw + (n - 1) * gap
            need_h = sh + band_gap + mh + matrix_gap + matrix.shape[0] + 2
            need_w = max(cat_w, matrix.shape[1]) + 2
            if need_h > 30 or need_w > 30:
                continue
            height, width = int(rng.integers(need_h, 31)), int(rng.integers(need_w, 31))
            sy = int(rng.integers(1, height - (need_h - 2)))
            sx = int(rng.integers(1, width - cat_w))
            cy, my = sy + sh + band_gap, sy + sh + band_gap + mh + matrix_gap
            mx = int(rng.integers(1, width - matrix.shape[1]))
            grid = np.full((height, width), bg, dtype=np.int8)
            for i, (color, motif) in enumerate(zip(palette, motifs)):
                x = sx + i * (mw + gap)
                grid[sy:sy + sh, x:x + sw] = color
                grid[cy:cy + mh, x:x + mw] = np.where(motif, marker, bg)
            grid[my:my + matrix.shape[0], mx:mx + matrix.shape[1]] = matrix
        elif direction == 'vertical':
            cat_h = n * mh + (n - 1) * gap
            need_w = sw + band_gap + mw + matrix_gap + matrix.shape[1] + 2
            need_h = max(cat_h, matrix.shape[0]) + 2
            if need_w > 30 or need_h > 30:
                continue
            height, width = int(rng.integers(need_h, 31)), int(rng.integers(need_w, 31))
            sx = int(rng.integers(1, width - (need_w - 2)))
            sy = int(rng.integers(1, height - cat_h))
            cx, mx = sx + sw + band_gap, sx + sw + band_gap + mw + matrix_gap
            my = int(rng.integers(1, height - matrix.shape[0]))
            grid = np.full((height, width), bg, dtype=np.int8)
            for i, (color, motif) in enumerate(zip(palette, motifs)):
                y = sy + i * (mh + gap)
                grid[y:y + sh, sx:sx + sw] = color
                grid[y:y + mh, cx:cx + mw] = np.where(motif, marker, bg)
            grid[my:my + matrix.shape[0], mx:mx + matrix.shape[1]] = matrix
        else:  # source-like adjacent swatch and motif lists
            swatch_w = n * sw + (n - 1) * gap
            cat_w = n * mw + (n - 1) * gap
            need_w = swatch_w + band_gap + cat_w + 2
            need_h = mh + matrix_gap + matrix.shape[0] + 2
            if need_w > 30 or need_h > 30:
                continue
            height, width = int(rng.integers(need_h, 31)), int(rng.integers(need_w, 31))
            hy = int(rng.integers(1, height - (need_h - 2)))
            sx = int(rng.integers(1, width - (need_w - 2)))
            cx = sx + swatch_w + band_gap
            my = hy + mh + matrix_gap
            mx = int(rng.integers(1, width - matrix.shape[1]))
            grid = np.full((height, width), bg, dtype=np.int8)
            for i, (color, motif) in enumerate(zip(palette, motifs)):
                grid[hy:hy + sh, sx + i * (sw + gap):sx + i * (sw + gap) + sw] = color
                grid[hy:hy + mh, cx + i * (mw + gap):cx + i * (mw + gap) + mw] = np.where(motif, marker, bg)
            grid[my:my + matrix.shape[0], mx:mx + matrix.shape[1]] = matrix
        return grid
    raise RuntimeError('layout construction failed')


def _pair(rng, regime, direction, force_asym=False):
    colors = [int(v) for v in rng.permutation(10)]
    bg, marker = colors[:2]
    count = int(rng.integers(2, 4 if regime == 'source_train' else 5))
    palette = tuple(colors[2:2 + count])
    mh, mw, h, w = _dimensions(rng, regime, count, force_asym, direction)
    motifs = _motifs(rng, count, mh, mw, force_asym)
    matrix = _matrix(rng, palette, h, w)
    return {'input': _layout(rng, direction, palette, motifs, matrix, bg, marker),
            'output': _expand(matrix, palette, motifs, bg)}


def _runs(mask):
    pos = np.flatnonzero(mask)
    if not len(pos):
        return []
    return [(int(p[0]), int(p[-1]) + 1) for p in np.split(pos, np.where(np.diff(pos) > 1)[0] + 1)]


def _inspect(grid, direction):
    if direction == 'side':
        return _inspect_side(grid)
    bg = int(grid[0, 0])
    horizontal = direction == 'horizontal'
    bands = _runs(np.any(grid != bg, axis=1 if horizontal else 0))
    assert len(bands) == 3
    def region(b):
        return grid[b[0]:b[1], :] if horizontal else grid[:, b[0]:b[1]]
    sr, cr, mr = map(region, bands)
    swatches = _runs(np.any(sr != bg, axis=0 if horizontal else 1))
    slots = _runs(np.any(cr != bg, axis=0 if horizontal else 1))
    assert 2 <= len(swatches) <= 4 and len(swatches) == len(slots)
    assert [a for a, _ in swatches] == [a for a, _ in slots]
    marker_values = set(int(v) for v in np.unique(cr)) - {bg}
    assert len(marker_values) == 1
    marker = marker_values.pop()
    mh = bands[1][1] - bands[1][0] if horizontal else slots[0][1] - slots[0][0]
    mw = slots[0][1] - slots[0][0] if horizontal else bands[1][1] - bands[1][0]
    expected = np.full_like(grid, bg)
    palette, motifs = [], []
    for (s0, s1), (m0, m1) in zip(swatches, slots):
        sw = sr[:, s0:s1] if horizontal else sr[s0:s1, :]
        mo = cr[:, m0:m1] if horizontal else cr[m0:m1, :]
        color = int(sw[0, 0])
        assert color not in (bg, marker) and np.all(sw == color)
        assert mo.shape == (mh, mw) and np.all((mo == bg) | (mo == marker))
        motif = (mo == marker).astype(np.int8)
        assert np.any(motif == 0) and np.all(motif.any(axis=0)) and np.all(motif.any(axis=1))
        palette.append(color)
        motifs.append(motif)
        if horizontal:
            expected[bands[0][0]:bands[0][1], s0:s1] = color
            expected[bands[1][0]:bands[1][1], m0:m1] = mo
        else:
            expected[s0:s1, bands[0][0]:bands[0][1]] = color
            expected[m0:m1, bands[1][0]:bands[1][1]] = mo
    assert len(set(palette)) == len(palette) and len({m.tobytes() for m in motifs}) == len(motifs)
    occupied = np.argwhere(mr != bg)
    assert len(occupied)
    y0, x0 = occupied.min(axis=0)
    y1, x1 = occupied.max(axis=0) + 1
    matrix = mr[y0:y1, x0:x1]
    assert min(matrix.shape) >= 3 and np.all(matrix != bg)
    assert 2 <= len(np.unique(matrix)) and set(int(v) for v in np.unique(matrix)) <= set(palette)
    if horizontal:
        expected[bands[2][0] + y0:bands[2][0] + y1, x0:x1] = matrix
    else:
        expected[y0:y1, bands[2][0] + x0:bands[2][0] + x1] = matrix
    assert np.array_equal(grid, expected)
    return {'background': bg, 'marker': marker, 'palette': tuple(palette),
            'motifs': tuple(motifs), 'matrix': matrix, 'motif_shape': (mh, mw),
            'matrix_shape': matrix.shape, 'orientation': direction, 'swatch_shape': sw.shape,
            'band_gaps': (bands[1][0] - bands[0][1], bands[2][0] - bands[1][1])}


def _inspect_side(grid):
    bg = int(grid[0, 0])
    bands = _runs(np.any(grid != bg, axis=1))
    assert len(bands) == 2
    header = grid[bands[0][0]:bands[0][1], :]
    groups = _runs(np.any(header != bg, axis=0))
    assert len(groups) in (4, 6, 8)
    n = len(groups) // 2
    swatches, slots = groups[:n], groups[n:]
    marker_values = set(int(v) for x0, x1 in slots for v in np.unique(header[:, x0:x1])) - {bg}
    assert len(marker_values) == 1
    marker = marker_values.pop()
    mh, mw = header.shape[0], slots[0][1] - slots[0][0]
    expected = np.full_like(grid, bg)
    palette, motifs = [], []
    swatch_shape = None
    for s0, s1 in swatches:
        block = header[:, s0:s1]
        active = _runs(np.any(block != bg, axis=1))
        assert len(active) == 1 and active[0][0] == 0
        sh = active[0][1]
        swatch = block[:sh, :]
        color = int(swatch[0, 0])
        assert color not in (bg, marker) and np.all(swatch == color)
        assert np.all(block[sh:, :] == bg)
        if swatch_shape is None:
            swatch_shape = swatch.shape
        assert swatch.shape == swatch_shape
        palette.append(color)
        expected[bands[0][0]:bands[0][0] + sh, s0:s1] = color
    for m0, m1 in slots:
        block = header[:, m0:m1]
        assert block.shape == (mh, mw) and np.all((block == bg) | (block == marker))
        motif = (block == marker).astype(np.int8)
        assert np.any(motif == 0) and np.all(motif.any(axis=0)) and np.all(motif.any(axis=1))
        motifs.append(motif)
        expected[bands[0][0]:bands[0][1], m0:m1] = block
    assert len(set(palette)) == n and len({m.tobytes() for m in motifs}) == n
    mr = grid[bands[1][0]:bands[1][1], :]
    occupied = np.argwhere(mr != bg)
    assert len(occupied)
    y0, x0 = occupied.min(axis=0)
    y1, x1 = occupied.max(axis=0) + 1
    matrix = mr[y0:y1, x0:x1]
    assert min(matrix.shape) >= 3 and np.all(matrix != bg)
    assert 2 <= len(np.unique(matrix)) and set(int(v) for v in np.unique(matrix)) <= set(palette)
    expected[bands[1][0] + y0:bands[1][0] + y1, x0:x1] = matrix
    assert np.array_equal(grid, expected)
    return {'background': bg, 'marker': marker, 'palette': tuple(palette),
            'motifs': tuple(motifs), 'matrix': matrix, 'motif_shape': (mh, mw),
            'matrix_shape': matrix.shape, 'orientation': 'side', 'swatch_shape': swatch_shape,
            'band_gaps': (slots[0][0] - swatches[-1][1], bands[1][0] - bands[0][1])}


def _grid_check(g):
    assert isinstance(g, np.ndarray) and g.ndim == 2 and np.issubdtype(g.dtype, np.integer)
    assert all(1 <= d <= 30 for d in g.shape) and np.all((g >= 0) & (g <= 9))


def _key(g):
    return g.shape, g.tobytes()


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    inputs, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = pair['input'], pair['output']
            _grid_check(inp)
            _grid_check(out)
            assert _key(inp) not in inputs
            inputs.add(_key(inp))
            assert not np.array_equal(inp, out)
            digest = hashlib.sha256(str(inp.shape).encode() + inp.astype(np.int8).tobytes()).hexdigest()
            assert digest not in _SOURCE_INPUT_HASHES
            assert inp[0, 0] == inp[0, -1] == inp[-1, 0] == inp[-1, -1]
            parses = []
            for direction in ('horizontal', 'vertical', 'side'):
                try:
                    parses.append(_inspect(inp, direction))
                except (AssertionError, IndexError, ValueError):
                    pass
            assert len(parses) == 1
            info = parses[0]
            expected = _expand(info['matrix'], info['palette'], info['motifs'], info['background'])
            assert np.array_equal(out, expected)
            assert info['marker'] not in set(int(v) for v in np.unique(out))
            assert len(info['palette']) <= (3 if split == 'train' else 4)
            assert out.shape == (info['matrix_shape'][0] * info['motif_shape'][0],
                                 info['matrix_shape'][1] * info['motif_shape'][1])
            infos[split].append(info)
    assert len({i['orientation'] for s in infos.values() for i in s}) == 1
    assert any(_asymmetric(m) for i in infos['train'] for m in i['motifs'])


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed) % (1 << 64))
    train_count, test_count = _counts(rng, num_train, num_test)
    direction = ('horizontal', 'vertical', 'side')[int(rng.choice(3, p=[.4, .3, .3]))]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count, regime in (('train', train_count, 'source_train'), ('test', test_count, 'target_test')):
        for index in range(count):
            for _ in range(30):
                pair_rng = np.random.default_rng(int(rng.integers(0, 1 << 63)))
                pair = _pair(pair_rng, regime, direction, split == 'train' and index == 0)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('duplicate input retries exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({tuple(_key(q['input']) for s in ('train', 'test') for q in p[s]) for p in puzzles}) >= 40
    roles, shapes, layouts, masks, patterns = set(), set(), set(), set(), set()
    directions, backgrounds, markers = set(), set(), set()
    dense3 = symmetric = uneven = unused = singleton = rectangular_motifs = rectangular_matrices = combined = 0
    sparse = dense = connected = disconnected = frame_like = 0
    row_stripes = column_stripes = other_matrices = uniform_border = mixed_uniform_row = 0
    larger_test_palette = larger_test_motif = smaller_test_output = overlapping_test_output = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        puzzle_infos = {'train': [], 'test': []}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                parsed = []
                for direction in ('horizontal', 'vertical', 'side'):
                    try:
                        parsed.append(_inspect(inp, direction))
                    except (AssertionError, IndexError, ValueError):
                        pass
                assert len(parsed) == 1
                info = parsed[0]
                puzzle_infos[split].append(info)
                roles.add((info['background'], info['marker']))
                directions.add(info['orientation'])
                backgrounds.add(info['background'])
                markers.add(info['marker'])
                shapes.add((inp.shape, pair['output'].shape))
                layouts.add((info['orientation'], info['swatch_shape'], info['band_gaps'],
                             len(info['palette']), info['motif_shape'], info['matrix_shape']))
                masks.update((m.shape, m.tobytes()) for m in info['motifs'])
                patterns.add((info['matrix'].shape, info['matrix'].tobytes()))
                dense3 += sum(m.shape == (3, 3) and int(m.sum()) >= 7 for m in info['motifs'])
                symmetric += sum(not _asymmetric(m) for m in info['motifs'])
                for motif in info['motifs']:
                    sparse += np.mean(motif) <= .4
                    dense += np.mean(motif) >= .75
                    components = _components(motif)
                    connected += components == 1
                    disconnected += components > 1
                    border = np.r_[motif[0], motif[-1], motif[:, 0], motif[:, -1]]
                    frame_like += np.mean(border) >= .75 and np.any(motif == 0)
                matrix = info['matrix']
                freq = [int(np.count_nonzero(matrix == color)) for color in info['palette']]
                uneven += max(freq) - min(freq) >= 3
                unused += 0 in freq
                singleton += 1 in freq
                rectangular_motifs += info['motif_shape'][0] != info['motif_shape'][1]
                rectangular_matrices += matrix.shape[0] != matrix.shape[1]
                combined += (info['motif_shape'][0] != info['motif_shape'][1]
                             and max(freq) - min(freq) >= 3
                             and info['background'] != 0 and info['marker'] != 5)
                row = bool(np.all(matrix == matrix[:, :1]))
                column = bool(np.all(matrix == matrix[:1, :]))
                row_stripes += row
                column_stripes += column
                other_matrices += not row and not column
                uniform_border += len(set(np.r_[matrix[0], matrix[-1], matrix[:, 0], matrix[:, -1]])) == 1
                mixed_uniform_row += any(len(set(r)) == 1 for r in matrix) and any(len(set(r)) > 1 for r in matrix)
        all_infos = puzzle_infos['train'] + puzzle_infos['test']
        assert len({i['background'] for i in all_infos}) >= 2
        assert len({i['marker'] for i in all_infos}) >= 2
        assert len({i['motif_shape'] for i in all_infos}) >= 2
        assert len({i['matrix_shape'] for i in all_infos}) >= 2
        assert len({len(i['palette']) for i in all_infos}) >= 2
        for info in puzzle_infos['test']:
            train = puzzle_infos['train']
            larger_test_palette += len(info['palette']) > max(len(t['palette']) for t in train)
            larger_test_motif += np.prod(info['motif_shape']) > max(np.prod(t['motif_shape']) for t in train)
            area = np.prod(info['matrix_shape']) * np.prod(info['motif_shape'])
            train_areas = [np.prod(t['matrix_shape']) * np.prod(t['motif_shape']) for t in train]
            smaller_test_output += area < min(train_areas)
            overlapping_test_output += any(info['matrix_shape'][0] * info['motif_shape'][0] == t['matrix_shape'][0] * t['motif_shape'][0]
                                           and info['matrix_shape'][1] * info['motif_shape'][1] == t['matrix_shape'][1] * t['motif_shape'][1]
                                           for t in train)
    assert len(roles) >= 18 and len(shapes) >= 80 and len(layouts) >= 100
    assert len(masks) >= 120 and len(patterns) >= 100
    assert directions == {'horizontal', 'vertical', 'side'}
    assert len(backgrounds) == len(markers) == 10
    assert dense3 >= 30 and symmetric >= 50
    assert sparse >= 50 and dense >= 100 and connected >= 100 and disconnected >= 100 and frame_like >= 100
    assert uneven >= 100 and unused >= 5 and singleton >= 5
    assert rectangular_motifs >= 50 and rectangular_matrices >= 100 and combined >= 20
    assert min(row_stripes, column_stripes) >= 10 and other_matrices >= 100
    assert uniform_border >= 10 and mixed_uniform_row >= 30
    assert larger_test_palette >= 5 and larger_test_motif >= 5
    assert smaller_test_output >= 3 and overlapping_test_output >= 2
    assert any(a != b for _, _, _, _, (a, b), _ in layouts)
    assert any(a != b for _, _, _, _, _, (a, b) in layouts)
    p, q = generate_puzzle(1001, 6, 3), generate_puzzle(1001, 6, 3)
    assert all(np.array_equal(x[k], y[k]) for s in ('train', 'test') for x, y in zip(p[s], q[s]) for k in ('input', 'output'))
    for a, b in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        validate_generated_puzzle(generate_puzzle(7000 + (a or 0) + (b or 0), a, b))
    for kw in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0}, {'num_test': 5},
               {'num_train': 2, 'num_test': 2}, {'num_train': 3, 'num_test': 4}):
        try:
            generate_puzzle(42, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid count accepted')
