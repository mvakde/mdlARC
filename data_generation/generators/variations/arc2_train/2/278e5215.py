import hashlib
import numpy as np


_SOURCE_INPUT_DIGESTS = {
    '0d612f421dfe8224bf7f5cdfa21e2e882c37045bb1633592cd41f6a384b1036f',
    '343d1d52ae70059ffd29805dc80ec5002d815b3f006f1fc6365801b2c515ee9c',
    '58235f11a01326afb9c58d89f7d057311af2561f0a9afbd09fe49a6bd88dedf5',
    '512e2c60c8112bd62f46801ef12167730a57482ffc2ef76ae4b2935b625d18df',
}
_LAYOUTS = ('above_left', 'above_aligned', 'above_right',
            'below_left', 'below_aligned', 'below_right',
            'side_left', 'side_right')


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer or None')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _checked_count('num_train', num_train, 2, 10)
    test = _checked_count('num_test', num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        low = max(2, test + 1)
        train = low + (seed // 4 + 2 * seed) % (11 - low)
    elif train is None:
        low = max(2, test + 1)
        train = low + (3 * seed + 1) % (11 - low)
    elif test is None:
        test = 1 + (5 * seed + 2) % min(4, train - 1)
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return train, test


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    groups = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        group = []
        while stack:
            y, x = stack.pop()
            group.append((y, x))
            for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if (0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1]
                        and mask[yy, xx] and not seen[yy, xx]):
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        groups.append(group)
    return groups


def _path(mask, start, end, rng):
    y, x = start
    ey, ex = end
    mask[y, x] = True
    while (y, x) != (ey, ex):
        axes = []
        if y != ey:
            axes.append(0)
        if x != ex:
            axes.append(1)
        if int(rng.choice(axes)) == 0:
            y += 1 if ey > y else -1
        else:
            x += 1 if ex > x else -1
        mask[y, x] = True


def _make_stencil(rng, height, width, style, require_hole=False):
    mask = np.zeros((height, width), dtype=bool)
    if style == 'solid':
        mask[:] = True
    elif style == 'frame':
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        for _ in range(int(rng.integers(1, max(3, width // 2 + 1)))):
            y, x = int(rng.integers(height)), int(rng.integers(width))
            length = int(rng.integers(1, max(3, min(height, width) // 2 + 1)))
            if rng.random() < .5:
                mask[y, x:min(width, x + length)] = True
            else:
                mask[y:min(height, y + length), x] = True
    elif style in ('paths', 'blob'):
        anchors = [(0, int(rng.integers(width))),
                   (int(rng.integers(height)), width-1),
                   (height-1, int(rng.integers(width))),
                   (int(rng.integers(height)), 0)]
        center = (int(rng.integers(height)), int(rng.integers(width)))
        for anchor in anchors:
            _path(mask, center, anchor, rng)
        if style == 'paths':
            for _ in range(int(rng.integers(2, 9))):
                points = np.argwhere(mask)
                start = tuple(int(v) for v in points[int(rng.integers(len(points)))])
                end = (int(rng.integers(height)), int(rng.integers(width)))
                _path(mask, start, end, rng)
        else:
            desired = int(height * width * rng.uniform(.32, .72))
            while int(mask.sum()) < desired:
                points = np.argwhere(mask)
                y, x = (int(v) for v in points[int(rng.integers(len(points)))])
                dy, dx = ((-1,0),(1,0),(0,-1),(0,1))[int(rng.integers(4))]
                if 0 <= y+dy < height and 0 <= x+dx < width:
                    mask[y+dy, x+dx] = True
                if len(points) >= height * width:
                    break
    elif style == 'pieces' and max(height, width) >= 6:
        if height >= 6 and (width < 6 or rng.random() < .5):
            band = int(rng.integers(2, height-2))
            mask[0:2, 0:2] = True
            mask[height-2:height, width-2:width] = True
            for _ in range(int(rng.integers(2, 7))):
                y0, y1 = (0, band) if rng.random() < .5 else (band+1, height)
                if y1-y0 >= 2:
                    y = int(rng.integers(y0, y1-1))
                    x = int(rng.integers(0, width-1))
                    mask[y:y+2, x:x+2] = True
            mask[band, :] = False
        else:
            band = int(rng.integers(2, width-2))
            mask[0:2, 0:2] = True
            mask[height-2:height, width-2:width] = True
            for _ in range(int(rng.integers(2, 7))):
                x0, x1 = (0, band) if rng.random() < .5 else (band+1, width)
                if x1-x0 >= 2:
                    y = int(rng.integers(0, height-1))
                    x = int(rng.integers(x0, x1-1))
                    mask[y:y+2, x:x+2] = True
            mask[:, band] = False
    else:
        mask = rng.random((height, width)) < float(rng.uniform(.17, .38))
    # A tight crop needs occupied extremes, not occupied interior bands.
    if style not in ('paths', 'blob', 'frame', 'solid'):
        mask[0, int(rng.integers(width))] = True
        mask[-1, int(rng.integers(width))] = True
        mask[int(rng.integers(height)), 0] = True
        mask[int(rng.integers(height)), -1] = True
    if require_hole and np.all(mask):
        mask[int(rng.integers(1, height-1)), int(rng.integers(1, width-1))] = False
    return mask


def _make_sequence(rng, width, available, asymmetric=False):
    colors = [int(v) for v in available]
    assert len(colors) >= 2
    mode = str(rng.choice(('runs', 'runs', 'alternating', 'irregular',
                           'return', 'palindrome')))
    if asymmetric and mode == 'palindrome':
        mode = 'runs'
    count = int(rng.integers(2, min(width, len(colors), 7) + 1))
    chosen = [int(v) for v in rng.choice(colors, count, replace=False)]
    seq = np.empty(width, dtype=np.int8)
    if mode == 'alternating':
        count = int(rng.integers(2, min(4, len(chosen)) + 1))
        chosen = chosen[:count]
        seq[:] = [chosen[j % count] for j in range(width)]
    elif mode == 'irregular':
        seq[:] = rng.choice(chosen, width)
        if width >= count:
            seq[rng.choice(width, count, replace=False)] = chosen
    elif mode == 'palindrome':
        half = (width + 1) // 2
        left = rng.choice(chosen, half)
        seq[:] = np.concatenate((left, left[:width//2][::-1]))
    else:
        if mode == 'return' and width >= 4:
            chosen = chosen[:max(2, min(len(chosen), width-1))]
            run_colors = [chosen[0], chosen[1], chosen[0]]
            if len(chosen) > 2:
                run_colors += chosen[2:]
        else:
            run_count = int(rng.integers(2, width + 1))
            run_colors = [int(rng.choice(chosen))]
            while len(run_colors) < run_count:
                run_colors.append(int(rng.choice([v for v in chosen if v != run_colors[-1]])))
        run_count = min(width, len(run_colors))
        cuts = sorted(int(v) for v in rng.choice(np.arange(1, width), run_count-1, replace=False))
        bounds = [0] + cuts + [width]
        for color, left, right in zip(run_colors, bounds[:-1], bounds[1:]):
            seq[left:right] = color
    if len(np.unique(seq)) < 2:
        seq[-1] = next(v for v in colors if v != seq[0])
    if asymmetric and seq[0] == seq[-1]:
        seq[-1] = next(v for v in colors if v != seq[0])
    return seq


def _choose_dimensions(rng, regime, shift, train_widths, layout):
    if regime == 'target_test' and shift in ('width', 'mixed') and rng.random() < .8:
        low = max(train_widths) + 1
        width = int(rng.integers(low, min(19, low+5))) if low <= 18 else 18
    elif regime == 'target_test' and rng.random() < .32:
        width = int(rng.choice(train_widths))
    else:
        width = int(rng.integers(4, 19))
    if layout.startswith('side'):
        width = min(width, 12)
    height = int(rng.integers(3, 17))
    return height, width


def _place(rng, height, width, layout, large_canvas=False):
    gap = int(rng.integers(1, 7))
    if layout.startswith('above') or layout.startswith('below'):
        if layout.endswith('aligned'):
            dx = 0
        else:
            dx = int(rng.integers(1, min(9, 31-width)))
            if layout.endswith('left'):
                dx = -dx
        if layout.startswith('above'):
            ky, sy = 0, 3+gap
        else:
            sy, ky = 0, height+gap
        kx, sx = 0, -dx  # dx is key displacement relative to stencil
    else:
        sy = 0
        ky = int(rng.integers(-min(4, height-1), min(5, height)))
        sx = 0
        kx = -width-gap if layout == 'side_left' else width+gap
    min_y, min_x = min(ky, sy), min(kx, sx)
    ky, sy, kx, sx = ky-min_y, sy-min_y, kx-min_x, sx-min_x
    footprint_h = max(ky+3, sy+height)
    footprint_w = max(kx+width, sx+width)
    assert footprint_h <= 30 and footprint_w <= 30
    def canvas_span(footprint):
        if large_canvas and rng.random() < .45:
            return 30
        spare = min(30-footprint, 10)
        low = 2 if spare >= 2 and rng.random() < .7 else 0
        return footprint + int(rng.integers(low, spare+1))
    H, W = canvas_span(footprint_h), canvas_span(footprint_w)
    def offset(spare):
        p = rng.random()
        if p < .17:
            return 0
        if p < .34:
            return spare
        return int(rng.integers(1, spare)) if spare >= 2 else int(rng.integers(spare+1))
    oy, ox = offset(H-footprint_h), offset(W-footprint_w)
    return (H, W), (sy+oy, sx+ox), (ky+oy, kx+ox)


def _input_digest(grid):
    return hashlib.sha256(bytes(grid.shape) + grid.astype(np.uint8).tobytes()).hexdigest()


def _inspect_input(grid):
    candidates = []
    for bg in np.unique(grid):
        bg = int(bg)
        non = grid != bg
        components = _components(non)
        for group in components:
            values = {int(grid[y, x]) for y, x in group}
            if len(values) < 2:
                continue
            yy = [p[0] for p in group]
            xx = [p[1] for p in group]
            ky0, ky1, kx0, kx1 = min(yy), max(yy), min(xx), max(xx)
            w = kx1-kx0+1
            if ky1-ky0+1 != 3 or len(group) != 3*w or w < 2:
                continue
            key = grid[ky0:ky1+1, kx0:kx1+1]
            if (np.any(key == bg) or not np.array_equal(key[0], key[1])
                    or not np.all(key[2] == key[2, 0])):
                continue
            seq, fill = key[0], int(key[2, 0])
            if len(np.unique(seq)) < 2 or fill in seq:
                continue
            other = non.copy()
            other[ky0:ky1+1, kx0:kx1+1] = False
            coords = np.argwhere(other)
            if len(coords) == 0 or len(np.unique(grid[other])) != 1:
                continue
            stencil = int(grid[other][0])
            if stencil in values:
                continue
            (sy0, sx0), (sy1, sx1) = coords.min(axis=0), coords.max(axis=0)
            sy0, sx0, sy1, sx1 = map(int, (sy0, sx0, sy1, sx1))
            if sx1-sx0+1 != w:
                continue
            if not (ky1+1 < sy0 or sy1+1 < ky0 or kx1+1 < sx0 or sx1+1 < kx0):
                continue
            local = grid[sy0:sy1+1, sx0:sx1+1]
            if np.any((local != bg) & (local != stencil)):
                continue
            mask = local == stencil
            if ky1 < sy0:
                position = 'above'
            elif sy1 < ky0:
                position = 'below'
            else:
                position = 'side'
            if position == 'side':
                relation = position + ('_left' if kx0 < sx0 else '_right')
            else:
                relation = position + ('_left' if kx0 < sx0 else
                                       '_right' if kx0 > sx0 else '_aligned')
            expected = np.where(mask, seq[np.newaxis, :], fill).astype(np.int8)
            candidates.append({'background': bg, 'stencil': stencil,
                               'sequence': tuple(int(v) for v in seq), 'fill': fill,
                               'height': mask.shape[0], 'width': w,
                               'mask': mask, 'expected': expected,
                               'components': len(_components(mask)),
                               'relation': relation, 'key_pos': (ky0, kx0),
                               'stencil_pos': (sy0, sx0),
                               'edge': bool(np.any(non[0]) or np.any(non[-1]) or
                                            np.any(non[:, 0]) or np.any(non[:, -1]))})
    assert len(candidates) == 1, f'expected one readable key and stencil, found {len(candidates)}'
    return candidates[0]


def _inspect_pair(input_grid, output_grid):
    info = _inspect_input(input_grid)
    assert output_grid.shape == info['expected'].shape
    assert np.array_equal(output_grid, info['expected'])
    return info


def _grid_signature(grid):
    return (grid.shape, tuple(int(v) for v in grid.ravel()))


def _direction_disagreement(info):
    seq = np.asarray(info['sequence'])
    return int(np.count_nonzero(info['mask'] & (seq != seq[::-1])[np.newaxis, :]))


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen = set()
    train_info, test_info = [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= n <= 30 for n in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert np.all((0 <= grid) & (grid <= 9))
            sig = _grid_signature(pair['input'])
            assert sig not in seen
            seen.add(sig)
            assert _input_digest(pair['input']) not in _SOURCE_INPUT_DIGESTS
            info = _inspect_pair(pair['input'], pair['output'])
            assert not np.array_equal(pair['input'], pair['output'])
            (train_info if split == 'train' else test_info).append(info)
    assert any(np.any(~x['mask']) for x in train_info), 'training must show fill in holes'
    assert any(_direction_disagreement(x) > 0 for x in train_info), 'key direction ambiguous'
    assert len({x['relation'] for x in train_info}) >= 2
    # Roles are visible in each input; the only learned convention is the
    # repeated-row/third-row interpretation and local column direction.
    assert all(x['width'] == len(x['sequence']) for x in train_info + test_info)


def _pair_palette(rng, params):
    base = params['base']
    mode = params['palette_mode']
    if mode == 'stable':
        background, stencil = base[:2]
    elif mode == 'mixed':
        background = base[0]
        stencil = int(rng.choice([v for v in range(10) if v != background]))
    else:
        background, stencil = (int(v) for v in rng.choice(10, 2, replace=False))
    return background, stencil


def _generate_pair(rng, regime, params, index, train_widths, train_info):
    layout = params['first_layouts'][index] if regime == 'source_train' and index < 2 else str(rng.choice(_LAYOUTS))
    if regime == 'target_test' and params['shift'] in ('placement', 'mixed') and rng.random() < .7:
        used = {x['relation'] for x in train_info}
        fresh = [v for v in _LAYOUTS if v not in used]
        if fresh:
            layout = str(rng.choice(fresh))
    height, width = _choose_dimensions(rng, regime, params['shift'], train_widths, layout)
    background, stencil = _pair_palette(rng, params)
    choices = [v for v in range(10) if v not in (background, stencil)]
    train_fills = {x['fill'] for x in train_info}
    train_seq = {v for x in train_info for v in x['sequence']}
    if regime == 'target_test' and rng.random() < .45:
        familiar = [v for v in choices if v in train_fills]
        novel = [v for v in choices if v not in train_fills]
        target = familiar if rng.random() < .5 else novel
        fill = int(rng.choice(target or choices))
    else:
        fill = int(rng.choice(choices))
    available = [v for v in choices if v != fill]
    seq = _make_sequence(rng, width, available, asymmetric=regime == 'source_train' and index == 0)
    if regime == 'target_test' and params['shift'] in ('palette', 'mixed') and rng.random() < .6:
        unseen = [v for v in available if v not in train_seq]
        if unseen:
            seq[int(rng.integers(width))] = int(rng.choice(unseen))
    style = str(rng.choice(('paths', 'paths', 'blob', 'blob', 'frame',
                            'pieces', 'pieces', 'sparse', 'solid')))
    if regime == 'source_train' and index == 0:
        style = str(rng.choice(('paths', 'paths', 'blob', 'blob', 'frame')))
    mask = _make_stencil(rng, height, width, style,
                         require_hole=regime == 'source_train' and index == 0)
    shape, (sy, sx), (ky, kx) = _place(rng, height, width, layout,
                                      large_canvas=rng.random() < .32)
    inp = np.full(shape, background, dtype=np.int8)
    view = inp[sy:sy+height, sx:sx+width]
    view[mask] = stencil
    inp[ky:ky+2, kx:kx+width] = seq
    inp[ky+2, kx:kx+width] = fill
    out = np.where(mask, seq[np.newaxis, :], fill).astype(np.int8)
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    ntrain, ntest = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    first_layouts = rng.choice(_LAYOUTS, 2, replace=False)
    params = {'base': tuple(int(v) for v in rng.permutation(10)),
              'palette_mode': str(rng.choice(('stable', 'mixed', 'variable', 'variable'))),
              'shift': str(rng.choice(('width', 'placement', 'palette', 'mixed', 'familiar'))),
              'first_layouts': tuple(str(v) for v in first_layouts)}
    pair_seeds = rng.integers(0, np.iinfo(np.int64).max, ntrain+ntest, dtype=np.int64)
    puzzle = {'train': [], 'test': []}
    seen = set()
    train_widths = []
    train_info = []
    for split, count, start in (('train', ntrain, 0), ('test', ntest, ntrain)):
        for index in range(count):
            pair_rng = np.random.default_rng(int(pair_seeds[start+index]))
            for _ in range(30):
                pair = _generate_pair(pair_rng, 'source_train' if split == 'train' else 'target_test',
                                      params, index, train_widths, train_info)
                sig = _grid_signature(pair['input'])
                if sig not in seen and _input_digest(pair['input']) not in _SOURCE_INPUT_DIGESTS:
                    break
            else:
                raise RuntimeError('unable to construct a distinct pair')
            seen.add(sig)
            puzzle[split].append(pair)
            if split == 'train':
                info = _inspect_pair(pair['input'], pair['output'])
                train_info.append(info)
                train_widths.append(info['width'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _puzzle_signature(puzzle):
    return tuple((split, tuple((_grid_signature(p['input']), _grid_signature(p['output']))
                              for p in puzzle[split])) for split in ('train', 'test'))


def self_test() -> None:
    for seed, train, test in ((17, None, None), (17, 7, 3), (102, 2, 1),
                              (1021, 2, 1), (1101, 10, 1), (1104, 10, 4),
                              (2000, None, 1), (2001, None, 4),
                              (2002, 2, None), (2003, 10, None)):
        first = generate_puzzle(seed, train, test)
        second = generate_puzzle(seed, train, test)
        assert _puzzle_signature(first) == _puzzle_signature(second)
        validate_generated_puzzle(first)
        if train is not None:
            assert len(first['train']) == train
        if test is not None:
            assert len(first['test']) == test
    for kwargs in ({'num_train': 1}, {'num_train': 11}, {'num_test': 0},
                   {'num_test': 5}, {'num_train': 2, 'num_test': 2},
                   {'num_train': 3, 'num_test': 4}, {'num_train': 2.0}):
        try:
            generate_puzzle(0, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    train_counts, test_counts, signatures = set(), set(), set()
    shapes, sequences, masks, relations, palettes, components = (set() for _ in range(6))
    connected = edge = blank = solid = five = 0
    review = {name: 0 for name in (
        'palindrome', 'long_run', 'single_cell_runs', 'many_runs',
        'early_return', 'fixed_hole_1_filled', 'fixed_hole_2_filled',
        'blank_row', 'blank_col', 'few_pieces', 'full_canvas',
        'source_shape', 'five_connected', 'edge_side', 'new_sequence',
        'familiar_fill', 'novel_fill', 'width_overlap', 'width_extrapolation',
        'smaller_test', 'background_changes', 'stencil_changes',
        'multiple_test_fills', 'three_relations')}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        train_counts.add(len(puzzle['train']))
        test_counts.add(len(puzzle['test']))
        signatures.add(_puzzle_signature(puzzle))
        train = [_inspect_pair(p['input'], p['output']) for p in puzzle['train']]
        test = [_inspect_pair(p['input'], p['output']) for p in puzzle['test']]
        all_info = train + test
        train_colors = {v for x in train for v in x['sequence']}
        train_fills = {x['fill'] for x in train}
        review['new_sequence'] += any(set(x['sequence']) - train_colors for x in test)
        review['familiar_fill'] += any(x['fill'] in train_fills for x in test)
        review['novel_fill'] += any(x['fill'] not in train_fills for x in test)
        review['width_overlap'] += any(x['width'] <= max(y['width'] for y in train) for x in test)
        review['width_extrapolation'] += all(x['width'] > max(y['width'] for y in train) for x in test)
        review['smaller_test'] += any(x['width'] < min(y['width'] for y in train) for x in test)
        review['background_changes'] += len({x['background'] for x in all_info}) > 1
        review['stencil_changes'] += len({x['stencil'] for x in all_info}) > 1
        review['multiple_test_fills'] += len({x['fill'] for x in test}) > 1
        review['three_relations'] += len({x['relation'] for x in all_info}) >= 3
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _inspect_pair(pair['input'], pair['output'])
                shapes.add((pair['input'].shape, pair['output'].shape))
                sequences.add(info['sequence'])
                masks.add((info['mask'].shape, tuple(bool(v) for v in info['mask'].ravel())))
                relations.add(info['relation'])
                palettes.add((info['background'], info['stencil'], info['fill']))
                components.add(info['components'])
                connected += info['components'] == 1
                edge += info['edge']
                blank += bool(np.any(~np.any(info['mask'], axis=0)) or
                              np.any(~np.any(info['mask'], axis=1)))
                solid += bool(np.all(info['mask']))
                five += len(set(info['sequence'])) >= 5
                mask, seq = info['mask'], info['sequence']
                runs = [seq[0]]
                lengths = [1]
                for color in seq[1:]:
                    if color == runs[-1]:
                        lengths[-1] += 1
                    else:
                        runs.append(color)
                        lengths.append(1)
                review['palindrome'] += seq == seq[::-1]
                review['long_run'] += max(lengths) >= 4
                review['single_cell_runs'] += max(lengths) == 1
                review['many_runs'] += len(runs) > len(set(seq)) + 3
                review['early_return'] += any(
                    runs[i] in runs[:i] and len(set(runs[:i])) < len(set(seq))
                    for i in range(1, len(runs)))
                review['blank_row'] += bool(np.any(~np.any(mask, axis=1)))
                review['blank_col'] += bool(np.any(~np.any(mask, axis=0)))
                review['few_pieces'] += 2 <= info['components'] <= 5
                review['full_canvas'] += 30 in pair['input'].shape
                review['source_shape'] += pair['input'].shape in {
                    (15, 10), (15, 15), (15, 16), (18, 18)}
                review['five_connected'] += len(set(seq)) >= 5 and info['components'] == 1
                review['edge_side'] += info['edge'] and info['relation'].startswith('side')
                if mask.shape[0] > 2 and mask.shape[1] > 2:
                    review['fixed_hole_1_filled'] += bool(mask[1, 1])
                    review['fixed_hole_2_filled'] += bool(mask[-2, -2])
    assert len(train_counts) >= 4 and len(test_counts) >= 3
    assert len(signatures) == 42
    assert len(shapes) >= 70 and len(sequences) >= 100 and len(masks) >= 100
    assert len(palettes) >= 50 and len(relations) == 8
    assert connected >= 40 and edge >= 20 and blank >= 5 and solid >= 1 and five >= 10
    assert 1 in components and max(components) >= 2
    assert all(value > 0 for value in review.values()), review
