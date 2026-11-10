import numpy as np


# The five source inputs are recorded as foreground color and the six optional
# cells of each comb. This permits exact source-input exclusion without I/O.
_SOURCE_INPUTS = {
    (1, ('001111', '100001', '000000', '000100', '111011')),
    (7, ('101001', '000100', '101111', '001100', '111010')),
    (1, ('001111', '011111', '101100', '000000', '000010')),
    (6, ('001110', '101000', '101011', '000000', '000010')),
    (6, ('011110', '101001', '011111', '100000', '000011')),
}


def _checked_count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return value


def _resolve_counts(seed, num_train, num_test):
    train = _checked_count('num_train', num_train, 2, 10)
    test = _checked_count('num_test', num_test, 1, 4)
    if train is None and test is None:
        test = 1 + seed % 4
        minimum = max(2, test + 1)
        train = minimum + (seed // 4) % (11 - minimum)
    elif train is None:
        minimum = max(2, test + 1)
        train = minimum + seed % (11 - minimum)
    elif test is None:
        test = 1 + seed % min(4, train - 1)
    if train <= test:
        raise ValueError('num_train must exceed num_test')
    return train, test


def _geometry(rng, n=None, n_exclude=None, source_style=False):
    if n is None:
        choices = [v for v in range(3, 8) if v != n_exclude]
        n = int(rng.choice(choices))
    if source_style and n == 5:
        height, step, phase = 13, 2, 0
    else:
        candidates = [(h, s, p) for h in range(9, 23)
                      for s in (2, 3, 4) for p in range(s)
                      if h - len(range(p, h, s)) >= n - 1
                      and len(range(p, h, s)) >= 2]
        height, step, phase = candidates[int(rng.integers(len(candidates)))]
    gap_choices = [g for g in (1, 2, 3) if 2*n + g*(n-1) <= 30]
    gap = int(rng.choice(gap_choices))
    remaining = min(6, 30 - (2*n + gap*(n-1)))
    margin_choices = [(a, b) for a in range(4) for b in range(4)
                      if a+b <= remaining]
    left, right = margin_choices[int(rng.integers(len(margin_choices)))]
    top = int(rng.integers(0, min(3, (30-height)) + 1))
    bottom = int(rng.integers(0, min(3, (30-height-top)) + 1))
    return dict(n=n, height=height, step=step, phase=phase, gap=gap,
                left=left, right=right, top=top, bottom=bottom)


def _counts(rng, n, m):
    chosen = set()
    if rng.random() < 0.55:
        chosen.add(0)
    if rng.random() < 0.55:
        chosen.add(m)
    pool = [x for x in range(m+1) if x not in chosen]
    chosen.update(int(x) for x in rng.choice(pool, size=n-len(chosen), replace=False))
    return tuple(sorted(chosen))


def _masks(rng, optional_rows, counts, anchor=None):
    m = len(optional_rows)
    result = []
    for count in counts:
        selected = set(int(x) for x in rng.choice(m, size=count, replace=False))
        result.append(selected)
    if anchor is not None:
        # A shifted count is not a literal one-cell augmentation of every
        # anchor bitmap. Change at least one old optional cell as well.
        eligible = [i for i, old in enumerate(anchor)
                    if 0 < len(old) and m-len(old) >= 2]
        if eligible:
            i = int(rng.choice(eligible))
            old = set(anchor[i])
            removed = int(rng.choice(sorted(old)))
            added = set(int(x) for x in rng.choice(sorted(set(range(m))-old),
                                                    size=2, replace=False))
            result[i] = (old-{removed}) | added
    return tuple(result)


def _permutation(rng, n, ascending=False, nontrivial=False):
    if ascending:
        return tuple(range(n))
    order = tuple(int(x) for x in rng.permutation(n))
    if nontrivial and order == tuple(range(n)):
        order = tuple(reversed(order))
    return order


def _make_pair(geometry, background, colors, counts, masks, orientation, mirror):
    n = geometry['n']
    h = geometry['height']
    gap = geometry['gap']
    width = geometry['left'] + 2*n + gap*(n-1) + geometry['right']
    shape = (geometry['top']+h+geometry['bottom'], width)
    inp = np.full(shape, background, dtype=np.int8)
    tooth_rows = range(geometry['phase'], h, geometry['step'])
    optional_rows = [r for r in range(h) if r not in tooth_rows]
    assert len(counts) == len(colors) == len(masks) == n
    for slot in range(n):
        col = geometry['left'] + slot*(2+gap)
        spine, tooth = (col+1, col) if mirror else (col, col+1)
        inp[geometry['top']:geometry['top']+h, spine] = colors[slot]
        inp[[geometry['top']+r for r in tooth_rows], tooth] = colors[slot]
        inp[[geometry['top']+optional_rows[k] for k in masks[slot]], tooth] = colors[slot]
    out = np.full(shape, background, dtype=np.int8)
    for dest, source in enumerate(sorted(range(n), key=lambda i: counts[i])):
        dst_col = geometry['left'] + dest*(2+gap)
        src_col = geometry['left'] + source*(2+gap)
        out[:, dst_col:dst_col+2] = inp[:, src_col:src_col+2]
    if orientation == 'horizontal':
        inp, out = inp.T.copy(), out.T.copy()
    return {'input': inp, 'output': out}


def _pair_spec(rng, background, n=None, n_exclude=None, orientation=None,
               mirror=None, multi=None, force_nontrivial=False, anchor=None):
    if anchor is None:
        source_style = rng.random() < 0.18
        if source_style:
            n = 5 if n is None and n_exclude != 5 else n
        geometry = _geometry(rng, n, n_exclude, source_style)
        m = geometry['height'] - len(range(geometry['phase'], geometry['height'], geometry['step']))
        sorted_counts = _counts(rng, geometry['n'], m)
        order = _permutation(rng, geometry['n'], ascending=(not force_nontrivial and rng.random()<0.16),
                             nontrivial=force_nontrivial)
        counts = tuple(sorted_counts[j] for j in order)
        optional_rows = [r for r in range(geometry['height'])
                         if r not in range(geometry['phase'], geometry['height'], geometry['step'])]
        masks = _masks(rng, optional_rows, counts)
    else:
        base = anchor
        geometry = _geometry(rng, n=base['geometry']['n'], source_style=False)
        for key in ('height', 'step', 'phase'):
            geometry[key] = base['geometry'][key]
        counts = tuple(x+1 for x in base['counts'])
        optional_rows = [r for r in range(geometry['height'])
                         if r not in range(geometry['phase'], geometry['height'], geometry['step'])]
        masks = _masks(rng, optional_rows, counts, base['masks'])
    if orientation is None:
        orientation = 'horizontal' if rng.random()<0.5 else 'vertical'
    if mirror is None:
        mirror = bool(rng.integers(0, 2))
    if multi is None:
        multi = bool(rng.integers(0, 2))
    available = [x for x in range(10) if x != background]
    if multi:
        colors = tuple(int(x) for x in rng.choice(available, size=geometry['n'], replace=False))
    else:
        colors = (int(rng.choice(available)),)*geometry['n']
    pair = _make_pair(geometry, background, colors, counts, masks, orientation, mirror)
    return pair, dict(geometry=geometry, counts=counts, masks=masks)


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    train_count, test_count = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    background_order = tuple(int(x) for x in rng.permutation(10))
    orientation_order = ('horizontal', 'vertical') if rng.random()<0.5 else ('vertical', 'horizontal')
    mirror_order = (False, True) if rng.random()<0.5 else (True, False)
    pairs, specs = [], []
    seen = set()
    for index in range(train_count+test_count):
        is_train = index < train_count
        pair_rng = np.random.default_rng(np.random.SeedSequence(
            [seed, index, int(rng.integers(0, 2**32, dtype=np.uint64))]))
        background = background_order[index % 10]
        anchor = None
        if not is_train and pair_rng.random()<0.2:
            eligible = [spec for spec in specs if
                        max(spec['counts']) < spec['geometry']['height']-
                        len(range(spec['geometry']['phase'], spec['geometry']['height'], spec['geometry']['step']))]
            if eligible:
                anchor = eligible[int(pair_rng.integers(len(eligible)))]
        for attempt in range(12):
            forced_orientation = orientation_order[index] if index < 2 else None
            forced_mirror = mirror_order[index] if index < 2 else None
            pair, spec = _pair_spec(
                pair_rng, background,
                n_exclude=specs[0]['geometry']['n'] if index == 1 else None,
                orientation=forced_orientation, mirror=forced_mirror,
                multi=(index == 0) if index < 2 else None,
                force_nontrivial=(index == 0), anchor=anchor)
            key = (pair['input'].shape, pair['input'].tobytes())
            if key not in seen and not _source_copy(pair['input']):
                break
        else:
            raise AssertionError('Could not construct a unique pair')
        pairs.append(pair)
        seen.add(key)
        if is_train:
            specs.append(spec)
    puzzle = {'train': pairs[:train_count], 'test': pairs[train_count:]}
    validate_generated_puzzle(puzzle)
    return puzzle


def _source_copy(grid):
    if grid.shape != (13, 14) or np.any(grid[:, (2, 5, 8, 11)] != 0):
        return False
    fg = int(grid[0, 0])
    masks = tuple(''.join(str(int(grid[r, 3*j+1] == fg)) for r in range(1, 13, 2))
                  for j in range(5))
    return (fg, masks) in _SOURCE_INPUTS


def _parse_scene(grid):
    findings = []
    for orientation in ('vertical', 'horizontal'):
        view = grid if orientation == 'vertical' else grid.T
        for background in (int(x) for x in np.unique(view)):
            occupied = view != background
            rows, cols = np.nonzero(occupied)
            if not len(rows):
                continue
            top, bottom = int(rows.min()), int(rows.max())+1
            left, right = int(cols.min()), int(cols.max())+1
            h = bottom-top
            if not (7 <= h <= 23 and top <= 3 and view.shape[0]-bottom <= 3
                    and left <= 3 and view.shape[1]-right <= 3):
                continue
            bands = []
            active = np.any(occupied, axis=0)
            c = left
            while c < right:
                if not active[c]:
                    c += 1
                    continue
                start = c
                while c < right and active[c]:
                    c += 1
                bands.append((start, c))
            n = len(bands)
            if not 3 <= n <= 7 or any(b-a != 2 for a, b in bands):
                continue
            gaps = [bands[i+1][0]-bands[i][1] for i in range(n-1)]
            if len(set(gaps)) != 1 or not 1 <= gaps[0] <= 3:
                continue
            if np.any(occupied[:top]) or np.any(occupied[bottom:]):
                continue
            if any(np.any(occupied[:, bands[i][1]:bands[i+1][0]]) for i in range(n-1)):
                continue
            for mirror in (False, True):
                blocks, colors = [], []
                valid = True
                for a, b in bands:
                    block = view[top:bottom, a:b]
                    spine = block[:, 1 if mirror else 0]
                    tooth = block[:, 0 if mirror else 1]
                    fg = int(spine[0])
                    if fg == background or not np.all(spine == fg) or not np.all((tooth == fg) | (tooth == background)):
                        valid = False
                        break
                    blocks.append(block.copy())
                    colors.append(fg)
                if not valid:
                    continue
                if not any(all(np.all(block[list(range(p, h, s)), 0 if mirror else 1] == color)
                               for block, color in zip(blocks, colors))
                           and h-len(range(p, h, s)) >= n-1 and len(range(p, h, s)) >= 2
                           for s in (2, 3, 4) for p in range(s)):
                    continue
                areas = tuple(int(np.count_nonzero(block != background)) for block in blocks)
                if len(set(areas)) != n:
                    continue
                findings.append(dict(orientation=orientation, background=background,
                                     top=top, bottom=bottom, bands=bands, gap=gaps[0],
                                     mirror=mirror, blocks=blocks, colors=tuple(colors),
                                     areas=areas, n=n, height=h))
    assert len(findings) == 1, 'scene must have a unique comb interpretation'
    return findings[0]


def _analyze_pair(pair):
    inp, out = pair['input'], pair['output']
    assert inp.shape == out.shape
    info = _parse_scene(inp)
    bg = info['background']
    view = inp if info['orientation'] == 'vertical' else inp.T
    expected = np.full(view.shape, bg, dtype=inp.dtype)
    order = sorted(range(info['n']), key=lambda i: info['areas'][i])
    for dst, src in enumerate(order):
        da, db = info['bands'][dst]
        expected[info['top']:info['bottom'], da:db] = info['blocks'][src]
    if info['orientation'] == 'horizontal':
        expected = expected.T
    assert np.array_equal(out, expected), 'whole combs must be sorted by total area'
    output_info = _parse_scene(out)
    assert output_info['orientation'] == info['orientation']
    assert output_info['background'] == bg
    assert output_info['mirror'] == info['mirror']
    assert output_info['bands'] == info['bands']
    assert output_info['areas'] == tuple(sorted(info['areas']))
    return info


def validate_generated_puzzle(puzzle: dict) -> None:
    try:
        assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
        assert all(isinstance(puzzle[k], list) for k in ('train', 'test'))
        train, test = puzzle['train'], puzzle['test']
        assert 2 <= len(train) <= 10 and 1 <= len(test) <= 4 and len(train) > len(test)
        seen = set()
        infos = []
        for pair in train+test:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert all(1 <= d <= 30 for d in grid.shape)
                assert np.issubdtype(grid.dtype, np.integer)
                assert 0 <= int(grid.min()) and int(grid.max()) <= 9
            key = (pair['input'].shape, pair['input'].tobytes())
            assert key not in seen
            seen.add(key)
            assert not _source_copy(pair['input'])
            infos.append(_analyze_pair(pair))
        training = infos[:len(train)]
        assert any(tuple(sorted(range(i['n']), key=lambda j: i['areas'][j])) != tuple(range(i['n']))
                   for i in training)
        assert {'vertical', 'horizontal'} <= {i['orientation'] for i in training}
        assert {False, True} <= {i['mirror'] for i in training}
        assert any(len(set(i['colors'])) > 1 for i in training)
        assert any(len(set(i['colors'])) == 1 for i in training)
        assert len({i['n'] for i in training}) >= 2
    except AssertionError:
        raise
    except Exception as error:
        raise AssertionError('Malformed or invalid puzzle') from error


def _signature(puzzle):
    return tuple((pair['input'].shape, pair['input'].tobytes(), pair['output'].tobytes())
                 for split in ('train', 'test') for pair in puzzle[split])


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(generate_puzzle(12345)) == _signature(generate_puzzle(12345))
    assert _signature(generate_puzzle(12345, 7, 3)) == _signature(generate_puzzle(12345, 7, 3))
    assert _signature(generate_puzzle(12345, 7, 3)) != _signature(generate_puzzle(12346, 7, 3))
    for train, test in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(9000+train*10+test, train, test)
        assert (len(p['train']), len(p['test'])) == (train, test)
    for train, test in ((None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(8000+(train or 0)+(test or 0), train, test)
        assert train is None or len(p['train']) == train
        assert test is None or len(p['test']) == test
    for train, test in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(99, train, test)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal pair counts accepted')
    try:
        validate_generated_puzzle({'train': [], 'test': []})
    except AssertionError:
        pass
    else:
        raise AssertionError('invalid puzzle accepted')
    dimensions, ns, gaps, mirrors, orientations, bgs = set(), set(), set(), set(), set(), set()
    count_sets, ranks, palettes = set(), set(), set()
    ascending = full = multi = 0
    margins = novel_test_ranks = new_test_colors = shifted_tests = fresh_shifts = 0
    tooth_profiles = set()
    within_n = within_shape = within_bg = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        local = []
        train_infos = [_analyze_pair(pair) for pair in puzzle['train']]
        train_ranks = {
            tuple(sorted(range(i['n']), key=lambda j: i['areas'][j]))
            for i in train_infos
        }
        train_colors = {color for i in train_infos for color in i['colors']}
        for index, pair in enumerate(puzzle['train']+puzzle['test']):
            info = _analyze_pair(pair)
            local.append(info)
            dimensions.add(pair['input'].shape)
            ns.add(info['n']); gaps.add(info['gap']); mirrors.add(info['mirror'])
            orientations.add(info['orientation']); bgs.add(info['background'])
            count_sets.add(tuple(sorted(info['areas'])))
            ranks.add(tuple(sorted(range(info['n']), key=lambda j: info['areas'][j])))
            palettes.add(tuple(sorted(set(info['colors']))))
            ascending += info['areas'] == tuple(sorted(info['areas']))
            full += any(a == info['height']*2 for a in info['areas'])
            multi += len(set(info['colors'])) > 1
            view = pair['input'] if info['orientation'] == 'vertical' else pair['input'].T
            margins += (info['top'] > 0 or info['bottom'] < view.shape[0]
                        or info['bands'][0][0] > 0 or info['bands'][-1][1] < view.shape[1])
            tooth_side = 0 if info['mirror'] else 1
            tooth_profiles.add(tuple(
                int(all(block[row, tooth_side] != info['background'] for block in info['blocks']))
                for row in range(info['height'])))
            if index >= len(puzzle['train']):
                rank = tuple(sorted(range(info['n']), key=lambda j: info['areas'][j]))
                novel_test_ranks += rank not in train_ranks
                new_test_colors += bool(set(info['colors']) - train_colors)
                for anchor in train_infos:
                    anchor_rank = tuple(sorted(range(anchor['n']), key=lambda j: anchor['areas'][j]))
                    if (info['n'] == anchor['n'] and info['height'] == anchor['height']
                            and rank == anchor_rank
                            and all(a == b+1 for a, b in zip(info['areas'], anchor['areas']))):
                        shifted_tests += 1
                        old_side = 0 if anchor['mirror'] else 1
                        if any(not np.all((old[:, old_side] == anchor['background']) |
                                          (new[:, tooth_side] != info['background']))
                               for old, new in zip(anchor['blocks'], info['blocks'])):
                            fresh_shifts += 1
                        break
        within_n += len({i['n'] for i in local}) > 1
        within_shape += len({p['input'].shape for p in puzzle['train']+puzzle['test']}) > 1
        within_bg += len({i['background'] for i in local}) > 1
    assert len(dimensions) >= 35 and len(ns) == 5 and len(gaps) == 3
    assert len(mirrors) == len(orientations) == 2 and len(bgs) == 10
    assert len(count_sets) >= 30 and len(ranks) >= 30 and len(palettes) >= 20
    assert ascending > 0 and full > 0 and multi > 0
    assert margins > 0 and len(tooth_profiles) >= 20
    assert novel_test_ranks > 0 and new_test_colors > 0
    assert shifted_tests > 0 and fresh_shifts > 0
    assert within_n == within_shape == within_bg == 42
