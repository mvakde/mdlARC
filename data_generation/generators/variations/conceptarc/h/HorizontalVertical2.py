import numpy as np


_SOURCE_INPUTS = (
    ('23232200000000','23232200000000','23232202222220','23232202222220','23232203333330','23232202222220','00000003333330','02323202222220','02323200000000','02323200000000','02323200000000'),
    ('000000000000','044444000000','088888016161','044444016161','088888016161','044444016161','088888016161','044444016161','044444016161','044444016161','000000000000','000000000000'),
    ('00000000001414','00000000001414','00000000001414','06161600001414','06161600001414','06161600001414','06161600000000','06161600000000','00000000001111','00555555007777','00111111007777','00555555007777','00111111001111','00555555007777'),
    ('000000000000','000000000000','011611100000','011611100000','011611100000','011611100000','011611100000','011611100000','000000000000','000000000000','000004444460','000004444460','000004444460','000004444460','000004444460','000004444460'),
    ('000000000000000030000','000088888000032333000','008888888000032333000','004444444000032330000','008888888000032330000','008888888000000330000','004444444000000000000','008888888880000000000','000088888888000000000','000088888000000000000','000000880000000000000','000000000000000000000'),
)


def _resolve_counts(rng, num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if value is not None:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
                raise ValueError(name)
    if num_train is None and num_test is None:
        num_test = int(rng.integers(1, 5))
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_train is None:
        num_train = int(rng.integers(max(2, num_test + 1), 11))
    elif num_test is None:
        if num_train <= 1:
            raise ValueError('incompatible counts')
        num_test = int(rng.integers(1, min(4, num_train - 1) + 1))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _puzzle_colors(rng):
    colors = [int(x) for x in rng.permutation(10)]
    train_size = int(rng.integers(3, 7))
    train_colors = colors[:train_size]
    reserved = colors[train_size:train_size + 2]
    backgrounds = colors[train_size + 2:]
    return train_colors, reserved, backgrounds


def _sequence(rng, length, colors, simple=False):
    n = len(colors)
    if simple and n == 2:
        boundary = int(rng.integers(1, length))
        return np.array([colors[0]] * boundary + [colors[1]] * (length - boundary), dtype=np.int8)
    # At least one band of each selected color is visible. Extra runs can repeat colors.
    run_count = int(rng.integers(n, min(length, n + 3) + 1))
    cuts = sorted(int(v) for v in rng.choice(np.arange(1, length), size=run_count - 1, replace=False))
    run_colors = [int(x) for x in rng.permutation(colors)]
    for _ in range(run_count - n):
        alternatives = [int(c) for c in colors if c != run_colors[-1]]
        run_colors.append(alternatives[int(rng.integers(len(alternatives)))])
    bounds = [0] + cuts + [length]
    seq = np.empty(length, dtype=np.int8)
    for i, color in enumerate(run_colors):
        seq[bounds[i]:bounds[i + 1]] = color
    return seq


def _mask(rng, height, width, style):
    mask = np.ones((height, width), dtype=bool)
    if style == 'rectangle':
        return mask
    anchor = int(rng.integers(1, width - 1))
    left_max, right_max = anchor, width - anchor - 1
    left = np.zeros(height, dtype=int)
    right = np.zeros(height, dtype=int)
    if style == 'jagged':
        for row in range(1, height):
            left[row] = int(rng.integers(0, min(2, left_max) + 1))
            right[row] = int(rng.integers(0, min(2, right_max) + 1))
    elif style == 'step':
        side = int(rng.integers(0, 2))
        split = int(rng.integers(1, height))
        if side == 0:
            left[split:] = int(rng.integers(1, left_max + 1))
        else:
            right[split:] = int(rng.integers(1, right_max + 1))
        if height >= 6 and rng.random() < 0.5:
            split2 = int(rng.integers(split, height))
            if side == 0:
                left[split2:] = min(left_max, left[split2] + int(rng.integers(0, left_max + 1)))
            else:
                right[split2:] = min(right_max, right[split2] + int(rng.integers(0, right_max + 1)))
    elif style == 'taper':
        mode = int(rng.integers(0, 3))
        if mode in (0, 2):
            left[:] = np.rint(np.linspace(0, int(rng.integers(1, left_max + 1)), height)).astype(int)
        if mode in (1, 2):
            right[:] = np.rint(np.linspace(0, int(rng.integers(1, right_max + 1)), height)).astype(int)
        if rng.random() < 0.5:
            left = left[::-1].copy()
            right = right[::-1].copy()
    else:
        raise ValueError('unknown outline')
    if not np.any(left + right):
        left[-1] = 1
    mask[:] = False
    for row in range(height):
        mask[row, left[row]:width - right[row]] = True
    return mask


def _sample_specs(rng, regime, index, multicolor, train_colors, reserved):
    if regime == 'source_train':
        count = int(rng.integers(2, 5))
        orientations = ['H', 'V'] + [str(rng.choice(['H', 'V'])) for _ in range(count - 2)]
        rng.shuffle(orientations)
    else:
        count = int(rng.choice([2, 3, 4], p=[0.5, 0.25, 0.25]))
        mode = str(rng.choice(['mixed', 'vertical', 'horizontal'], p=[0.55, 0.30, 0.15]))
        if mode == 'mixed':
            orientations = ['H', 'V'] + [str(rng.choice(['H', 'V'])) for _ in range(count - 2)]
            rng.shuffle(orientations)
        else:
            orientations = ['V' if mode == 'vertical' else 'H'] * count
    specs = []
    shared = None
    if regime == 'source_train' and index == 0 and rng.random() < 0.4:
        size = 3 if multicolor else 2
        shared = tuple(int(c) for c in rng.choice(train_colors, size=size, replace=False))
    for i, orientation in enumerate(orientations):
        if regime == 'source_train' and index == 0:
            n_colors = 3 if multicolor else 2
        else:
            n_colors = 3 if multicolor and rng.random() < 0.36 else 2
        if shared is not None and i < 2:
            colors = shared
        else:
            pool = list(train_colors)
            if regime == 'target_test' and rng.random() < 0.52:
                pool += list(reserved)
            colors = tuple(int(c) for c in rng.choice(pool, size=n_colors, replace=False))
        if regime == 'source_train' and index == 0:
            style = 'rectangle' if i == 0 else str(rng.choice(['step', 'taper', 'jagged'])) if i == 1 else str(rng.choice(['rectangle', 'step', 'taper', 'jagged']))
        else:
            style = str(rng.choice(['rectangle', 'jagged', 'step', 'taper'], p=[0.38, 0.18, 0.22, 0.22]))
        simple = n_colors == 2 and (index == 1 and regime == 'source_train' or rng.random() < 0.36)
        specs.append((orientation, colors, style, simple))
    return specs


def _separated(a, b):
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    return ar + ah + 1 <= br or br + bh + 1 <= ar or ac + aw + 1 <= bc or bc + bw + 1 <= ac


def _layout(rng, count):
    compact = count == 2 and rng.random() < 0.42
    for _ in range(500):
        if compact:
            short = int(rng.integers(10, 17))
            long = int(rng.integers(11, 26))
            height, width = (short, long) if rng.random() < 0.5 else (long, short)
        else:
            low = 15 if count == 2 else 17 if count == 3 else 20
            height = int(rng.integers(low, 31))
            width = int(rng.integers(low, 31))
        limit = 7 if compact else 10 if count == 2 else 8
        sizes = [(int(rng.integers(4, min(limit, height - 1) + 1)), int(rng.integers(4, min(limit, width - 1) + 1))) for _ in range(count)]
        if sum(h * w for h, w in sizes) > 0.48 * height * width:
            continue
        boxes = []
        for h, w in sizes:
            for _ in range(90):
                candidate = (int(rng.integers(0, height - h + 1)), int(rng.integers(0, width - w + 1)), h, w)
                if all(_separated(candidate, other) for other in boxes):
                    boxes.append(candidate)
                    break
            else:
                break
        if len(boxes) != count:
            continue
        top = min(b[0] for b in boxes)
        bottom = max(b[0] + b[2] for b in boxes)
        left = min(b[1] for b in boxes)
        right = max(b[1] + b[3] for b in boxes)
        span_fraction = 0.40 if compact else 0.48
        if bottom - top < span_fraction * height or right - left < span_fraction * width:
            continue
        if max(top, height - bottom) > max(5, int(0.37 * height)):
            continue
        if max(left, width - right) > max(5, int(0.37 * width)):
            continue
        return height, width, boxes
    # This fixed-size fallback is only reached if randomized packing is exhausted.
    height = width = 24
    slots = [int(v) for v in rng.permutation(4)[:count]]
    boxes = [(1 + 12 * (s // 2), 1 + 12 * (s % 2), 7, 7) for s in slots]
    return height, width, boxes


def _generate_pair(rng, regime, index, multicolor, train_colors, reserved, backgrounds):
    specs = _sample_specs(rng, regime, index, multicolor, train_colors, reserved)
    height, width, boxes = _layout(rng, len(specs))
    background = int(rng.choice(backgrounds))
    inp = np.full((height, width), background, dtype=np.int8)
    out = inp.copy()
    for (orientation, colors, style, simple), (top, left, h, w) in zip(specs, boxes):
        mask = _mask(rng, h, w, style)
        seq = _sequence(rng, h if orientation == 'H' else w, colors, simple)
        for row, col in np.argwhere(mask):
            color = seq[row if orientation == 'H' else col]
            inp[top + row, left + col] = color
            if orientation == 'V':
                out[top + row, left + col] = background
            else:
                out[top + row, left + col] = color
    return inp, out


def _components(grid, background):
    visited = np.zeros(grid.shape, dtype=bool)
    groups = []
    for sr, sc in np.argwhere(grid != background):
        sr, sc = int(sr), int(sc)
        if visited[sr, sc]:
            continue
        stack = [(sr, sc)]
        visited[sr, sc] = True
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and not visited[rr,cc] and grid[rr,cc] != background:
                    visited[rr,cc] = True
                    stack.append((rr,cc))
        rows = [r for r,c in cells]
        cols = [c for r,c in cells]
        top, bottom, left, right = min(rows), max(rows), min(cols), max(cols)
        assert bottom - top + 1 >= 4 and right - left + 1 >= 4
        occupied = set(cells)
        for r in range(top, bottom + 1):
            cs = [c for rr,c in cells if rr == r]
            assert cs and len(cs) == max(cs) - min(cs) + 1
        row_pure = all(len({int(grid[r,c]) for rr,c in cells if rr == r}) == 1 for r in range(top,bottom+1))
        col_pure = all(len({int(grid[rr,c]) for rr,c in cells if c == cc}) == 1 for cc in range(left,right+1))
        assert row_pure != col_pure
        orientation = 'H' if row_pure else 'V'
        assert any(all((r,c) in occupied for r in range(top,bottom+1)) for c in range(left,right+1))
        if orientation == 'V':
            assert any(all((r,c) in occupied for c in range(left,right+1)) for r in range(top,bottom+1))
        colors = frozenset(int(grid[r,c]) for r,c in cells)
        assert 2 <= len(colors) <= 4
        groups.append({'orientation': orientation, 'cells': cells, 'colors': colors,
                       'bbox': (top,left,bottom,right),
                       'rectangle': len(cells) == (bottom-top+1)*(right-left+1)})
    return groups


def _grid_key(grid):
    return grid.shape, grid.tobytes()


def _source_key(rows):
    return (len(rows), len(rows[0])), bytes(int(ch) for row in rows for ch in row)


_SOURCE_KEYS = {_source_key(rows) for rows in _SOURCE_INPUTS}


def _analyze_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = pair['input'], pair['output']
    for grid in (inp, out):
        assert isinstance(grid, np.ndarray) and grid.ndim == 2
        assert np.issubdtype(grid.dtype, np.integer)
        assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
        assert np.all((grid >= 0) & (grid <= 9))
    assert inp.shape == out.shape
    values, counts = np.unique(inp, return_counts=True)
    background = int(values[np.argmax(counts)])
    assert int(max(counts)) > inp.size // 2
    assert _grid_key(inp) not in _SOURCE_KEYS
    groups = _components(inp, background)
    assert 2 <= len(groups) <= 4
    expected = np.full(inp.shape, background, dtype=inp.dtype)
    for group in groups:
        if group['orientation'] == 'H':
            for r,c in group['cells']:
                expected[r,c] = inp[r,c]
    assert np.array_equal(out, expected)
    return background, groups


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    assert 2 <= len(puzzle['train']) <= 10
    assert 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    inputs = set()
    pairs = set()
    train_multicolor = {'H': False, 'V': False}
    test_multicolor = set()
    train_mixed_outline = False
    train_colors = set()
    for regime in ('train', 'test'):
        for pair in puzzle[regime]:
            background, groups = _analyze_pair(pair)
            key = _grid_key(pair['input'])
            pair_key = key, _grid_key(pair['output'])
            assert key not in inputs and pair_key not in pairs
            inputs.add(key)
            pairs.add(pair_key)
            orientations = {g['orientation'] for g in groups}
            if regime == 'train':
                assert orientations == {'H', 'V'}
                train_mixed_outline |= any(g['rectangle'] for g in groups) and any(not g['rectangle'] for g in groups)
                for g in groups:
                    train_colors.update(g['colors'])
                    if len(g['colors']) >= 3:
                        train_multicolor[g['orientation']] = True
            else:
                for g in groups:
                    if len(g['colors']) >= 3:
                        test_multicolor.add(g['orientation'])
    assert train_mixed_outline
    assert all(train_multicolor[o] for o in test_multicolor)


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    rng = np.random.default_rng(seed)
    train_count, test_count = _resolve_counts(rng, num_train, num_test)
    train_colors, reserved, backgrounds = _puzzle_colors(rng)
    if rng.random() < 0.65:
        backgrounds = [int(rng.choice(backgrounds))]
    multicolor = bool(rng.random() < 0.52)
    puzzle = {'train': [], 'test': []}
    for regime, count, key in (('source_train',train_count,'train'), ('target_test',test_count,'test')):
        for index in range(count):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))
            inp, out = _generate_pair(pair_rng, regime, index, multicolor, train_colors, reserved, backgrounds)
            puzzle[key].append({'input': inp, 'output': out})
    validate_generated_puzzle(puzzle)
    return puzzle


def _signature(puzzle):
    return tuple(tuple((_grid_key(pair['input']),_grid_key(pair['output'])) for pair in puzzle[part]) for part in ('train','test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    assert len({_signature(p) for p in puzzles}) == 42
    layouts, shapes, palettes, backgrounds = set(), set(), set(), set()
    styles = {'train': set(), 'test': set()}
    modes = set()
    short = 0
    novel = 0
    compact = 0
    wide_outline = 0
    thin_row = 0
    mixed_outline_pairs = 0
    shared_orientation_colors = 0
    disjoint_orientation_colors = 0
    combined_shift = 0
    blank_tests = 0
    unchanged_tests = 0
    varying_background_puzzles = 0
    varying_train_counts = 0
    multicolor = {'H':0,'V':0}
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        training_colors = set()
        puzzle_backgrounds = set()
        training_group_counts = set()
        for pair in puzzle['train']:
            bg, groups = _analyze_pair(pair)
            training_group_counts.add(len(groups))
            for g in groups:
                training_colors.update(g['colors'])
        for regime in ('train','test'):
            for pair in puzzle[regime]:
                bg, groups = _analyze_pair(pair)
                inp = pair['input']
                backgrounds.add(bg)
                puzzle_backgrounds.add(bg)
                shapes.add(inp.shape)
                palettes.add(frozenset(int(x) for x in np.unique(inp)))
                layouts.add((inp.shape, tuple(sorted((g['orientation'],g['bbox']) for g in groups))))
                styles[regime].update(g['rectangle'] for g in groups)
                compact += min(inp.shape) <= 14
                mixed_outline_pairs += {g['rectangle'] for g in groups} == {True,False}
                for first_index, first in enumerate(groups):
                    for second in groups[first_index + 1:]:
                        if first['orientation'] != second['orientation']:
                            shared_orientation_colors += bool(first['colors'] & second['colors'])
                            disjoint_orientation_colors += not bool(first['colors'] & second['colors'])
                if regime == 'test':
                    modes.add(frozenset(g['orientation'] for g in groups))
                    new_color = bool(set().union(*(g['colors'] for g in groups)) - training_colors)
                    novel += new_color
                    combined_shift += new_color and any(not g['rectangle'] for g in groups) and {g['orientation'] for g in groups} == {'H','V'}
                    blank_tests += bool(np.all(pair['output'] == bg))
                    unchanged_tests += bool(np.array_equal(inp, pair['output']))
                for g in groups:
                    multicolor[g['orientation']] += len(g['colors']) >= 3
                    top,left,bottom,right = g['bbox']
                    row_widths = [sum(r == row for r,c in g['cells']) for row in range(top,bottom+1)]
                    wide_outline += max(row_widths) - min(row_widths) >= 3
                    thin_row += min(row_widths) <= 2
                    if len(g['colors']) == 2:
                        if g['orientation'] == 'H':
                            seq = [int(inp[r,next(c for c in range(left,right+1) if (r,c) in g['cells'])]) for r in range(top,bottom+1)]
                        else:
                            seq = [int(inp[next(r for r in range(top,bottom+1) if (r,c) in g['cells']),c]) for c in range(left,right+1)]
                        short += sum(a != b for a,b in zip(seq,seq[1:])) == 1
        varying_background_puzzles += len(puzzle_backgrounds) > 1
        varying_train_counts += len(training_group_counts) > 1
    assert len(layouts) >= 150 and len(shapes) >= 30 and len(palettes) >= 25
    assert len(backgrounds) >= 5
    assert styles['train'] == {True,False} and styles['test'] == {True,False}
    assert modes == {frozenset({'H','V'}),frozenset({'H'}),frozenset({'V'})}
    assert short >= 30 and novel >= 5 and min(multicolor.values()) >= 10
    assert compact >= 10 and wide_outline >= 30 and thin_row >= 20
    assert mixed_outline_pairs >= 20 and shared_orientation_colors >= 20
    assert disjoint_orientation_colors >= 10 and combined_shift >= 2
    assert blank_tests >= 5 and unchanged_tests >= 2
    assert varying_background_puzzles >= 5 and varying_train_counts >= 20
    for seed, nt, ns in ((100,2,1),(101,10,1),(102,10,4),(103,None,1),(104,None,4),(105,2,None),(106,10,None)):
        p = generate_puzzle(seed, nt, ns)
        if nt is not None:
            assert len(p['train']) == nt
        if ns is not None:
            assert len(p['test']) == ns
    for args in ((17,None,None),(18,5,2),(19,2,None),(20,None,4)):
        assert _signature(generate_puzzle(*args)) == _signature(generate_puzzle(*args))
    for nt, ns in ((1,None),(11,None),(None,0),(None,5),(2,2),(3,4),(4,4)):
        try:
            generate_puzzle(999,nt,ns)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid count accepted')
    corrupt = generate_puzzle(12345,2,1)
    corrupt['test'][0]['output'] = corrupt['test'][0]['output'].copy()
    corrupt['test'][0]['output'][0,0] = (int(corrupt['test'][0]['output'][0,0]) + 1) % 10
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError('invalid label accepted')
