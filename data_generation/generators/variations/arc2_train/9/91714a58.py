"""Filter noise by retaining the unique substantial filled rectangle."""
import hashlib
import numpy as np

SOURCE_DIGESTS = {
    '729426f8905c78d0cab3bd6890595cddc81f85697948cbfbd648e52c069659d1',
    '8acb00453319c97e2abf93efb52b974f701efcfca45d79f2871e63fb9eb1713d',
    'f7f53ae2735184f4bc630cb9abfbeb9d097a2b84c5c33acec09c0bd8',
    '10dc14dae562e39992ecb380d18fd8f09607cc6ffb39fcff14bf5f490b7d436e',
}


def _count(value, name):
    if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be an integer')
    return int(value)


def _resolve_counts(seed, num_train, num_test):
    if num_train is not None:
        num_train = _count(num_train, 'num_train')
        if not 2 <= num_train <= 10:
            raise ValueError('num_train outside 2..10')
    if num_test is not None:
        num_test = _count(num_test, 'num_test')
        if not 1 <= num_test <= 4:
            raise ValueError('num_test outside 1..4')
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + (5 * seed + 3) % (10 - num_test)
    elif num_test is None:
        num_test = 1 + (7 * seed + 1) % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError('train count must exceed test count')
    return num_train, num_test


def _find_core(grid):
    """Search solid cores, even when same-color noise touches an edge."""
    counts = np.bincount(grid.ravel().astype(np.int64), minlength=10)
    background = int(np.argmax(counts))
    if counts[background] <= int(np.max(np.delete(counts, background))):
        return None
    height = grid.shape[0]
    candidates = []
    for color in range(10):
        if color == background or counts[color] < 4:
            continue
        colored = grid == color
        for r0 in range(height - 1):
            common = colored[r0].copy()
            for r1 in range(r0 + 1, height):
                common &= colored[r1]
                if np.count_nonzero(common) < 2:
                    break
                starts = np.flatnonzero(common & ~np.r_[False, common[:-1]])
                ends = np.flatnonzero(common & ~np.r_[common[1:], False]) + 1
                for c0, c1 in zip(starts, ends):
                    if c1 - c0 >= 2:
                        candidates.append((r0, r1 + 1, int(c0), int(c1), color))
    if not candidates:
        return None
    candidates.sort(key=lambda b: (b[1]-b[0])*(b[3]-b[2]), reverse=True)
    maximal = []
    for b in candidates:
        if any(a[4] == b[4] and a[0] <= b[0] and a[1] >= b[1]
               and a[2] <= b[2] and a[3] >= b[3] for a in maximal):
            continue
        maximal.append(b)
    best = maximal[0]
    area = (best[1]-best[0])*(best[3]-best[2])
    if area < 9:
        return None
    if len(maximal) > 1:
        runner = maximal[1]
        runner_area = (runner[1]-runner[0])*(runner[3]-runner[2])
        if runner_area > max(4, int(0.5*area)):
            return None
    return background, best


def _shape(rng, category, canvas):
    height, width = canvas
    if category == 'wide':
        rh = int(rng.integers(2, min(4, height-2)+1))
        rw = int(rng.integers(max(rh+1, 5), min(10, width-2)+1))
    elif category == 'tall':
        rw = int(rng.integers(2, min(4, width-2)+1))
        rh = int(rng.integers(max(rw+1, 5), min(10, height-2)+1))
    else:
        side = int(rng.integers(3, min(6, height-2, width-2)+1))
        rh = rw = side
    return rh, rw


def _make_pair(rng, background, color, category, band, signal_noise,
               force_contact=False):
    for _ in range(70):
        canvas = (16, 16) if rng.random() < 0.09 else (
            int(rng.integers(12, 25)), int(rng.integers(12, 25)))
        rh, rw = _shape(rng, category, canvas)
        if rh*rw > 0.28*canvas[0]*canvas[1]:
            continue
        r0 = int(rng.integers(canvas[0]-rh+1))
        c0 = int(rng.integers(canvas[1]-rw+1))
        grid = np.full(canvas, background, dtype=np.int8)
        grid[r0:r0+rh, c0:c0+rw] = color
        outside = [(r, c) for r in range(canvas[0]) for c in range(canvas[1])
                   if not (r0 <= r < r0+rh and c0 <= c < c0+rw)]
        ranges = {'sparse': (0.10, 0.17), 'moderate': (0.18, 0.26),
                  'crowded': (0.27, 0.37)}
        low, high = ranges[band]
        target = max(8, int(round(float(rng.uniform(low, high))*len(outside))))
        other_colors = [c for c in range(10) if c not in (background, color)]
        rng.shuffle(other_colors)
        palette = other_colors[:int(rng.integers(3, 9))]
        if signal_noise:
            palette.append(color)
        placed = set()
        if rng.random() < 0.65:
            patterns = (((0, 0), (0, 1)),
                        ((0, 0), (0, 1), (1, 0)),
                        ((0, 0), (0, 1), (0, 2), (1, 0), (1, 2)))
            pattern = patterns[int(rng.integers(len(patterns)))]
            motif_color = int(palette[int(rng.integers(len(palette)))])
            for _ in range(20):
                rr = int(rng.integers(canvas[0]-1))
                cc = int(rng.integers(canvas[1]-2))
                cells = {(rr+dr, cc+dc) for dr, dc in pattern}
                if all((r, c) in outside for r, c in cells):
                    for r, c in cells:
                        grid[r, c] = motif_color
                    placed.update(cells)
                    break
        if signal_noise and (force_contact or rng.random() < 0.5):
            border = ([(r0-1, c) for c in range(c0, c0+rw) if r0 > 0]
                      + [(r0+rh, c) for c in range(c0, c0+rw) if r0+rh < canvas[0]]
                      + [(r, c0-1) for r in range(r0, r0+rh) if c0 > 0]
                      + [(r, c0+rw) for r in range(r0, r0+rh) if c0+rw < canvas[1]])
            if border:
                cell = border[int(rng.integers(len(border)))]
                grid[cell] = color
                placed.add(cell)
        remaining = [cell for cell in outside if cell not in placed]
        count = max(0, min(target-len(placed), len(remaining)))
        for index in rng.choice(len(remaining), size=count, replace=False):
            r, c = remaining[int(index)]
            grid[r, c] = int(palette[int(rng.integers(len(palette)))])
        if signal_noise and not any(grid[cell] == color for cell in outside):
            cell = outside[int(rng.integers(len(outside)))]
            grid[cell] = color
        if _find_core(grid) != (background, (r0, r0+rh, c0, c0+rw, color)):
            continue
        output = np.full_like(grid, background)
        output[r0:r0+rh, c0:c0+rw] = color
        return {'input': grid, 'output': output}
    raise RuntimeError('failed to construct a unique core')


def _key(grid):
    return grid.shape, grid.tobytes()


def _digest(pair):
    return hashlib.sha256(pair['input'].astype(np.uint8).tobytes()
                          + pair['output'].astype(np.uint8).tobytes()).hexdigest()


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_)):
        raise ValueError('seed must be an integer')
    seed = int(seed)
    if seed < 0:
        raise ValueError('seed must be nonnegative')
    num_train, num_test = _resolve_counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    fixed_background = bool(rng.integers(2))
    base_background = int(rng.integers(10))
    backgrounds = [base_background]*(num_train+num_test)
    if not fixed_background:
        backgrounds[1] = int(rng.choice([c for c in range(10) if c != base_background]))
        for i in range(2, len(backgrounds)):
            backgrounds[i] = int(rng.integers(10))
    categories = ['wide', 'tall' if rng.integers(2) else 'square']
    if num_train >= 3:
        categories[1] = 'tall'
        categories.append('square')
    while len(categories) < num_train:
        categories.append(('wide', 'tall', 'square')[int(rng.integers(3))])
    rng.shuffle(categories)
    test_categories = list(rng.permutation(['wide', 'tall', 'square']))
    while len(test_categories) < num_test:
        test_categories.append(('wide', 'tall', 'square')[int(rng.integers(3))])
    train_colors = []
    colors = []
    for i, bg in enumerate(backgrounds):
        available = [c for c in range(10) if c != bg]
        if i == 1:
            available = [c for c in available if c != train_colors[0]]
        elif i >= num_train and rng.random() < 0.5:
            repeated = [c for c in train_colors if c != bg]
            if repeated:
                available = repeated
        color = int(available[int(rng.integers(len(available)))])
        colors.append(color)
        if i < num_train:
            train_colors.append(color)
    bands = ['sparse', 'crowded']
    while len(bands) < num_train:
        bands.append(('sparse', 'moderate', 'crowded')[int(rng.integers(3))])
    rng.shuffle(bands)
    puzzle = {'train': [], 'test': []}
    seen = set()
    test_shapes = set()
    for i in range(num_train+num_test):
        section = 'train' if i < num_train else 'test'
        category = categories[i] if section == 'train' else test_categories[i-num_train]
        band = bands[i] if section == 'train' else (
            'sparse', 'moderate', 'crowded')[int(rng.integers(3))]
        signal_noise = i == 0 or bool(rng.random() < 0.6)
        for _ in range(60):
            pair_rng = np.random.default_rng(int(rng.integers(0, 2**63)))
            pair = _make_pair(pair_rng, backgrounds[i], colors[i], category,
                              band, signal_noise, force_contact=i == 0)
            key = _key(pair['input'])
            if key in seen or _digest(pair) in SOURCE_DIGESTS:
                continue
            _, core = _find_core(pair['input'])
            shape = (core[1]-core[0], core[3]-core[2])
            if section == 'test' and shape in test_shapes:
                continue
            seen.add(key)
            if section == 'test':
                test_shapes.add(shape)
            puzzle[section].append(pair)
            break
        else:
            raise RuntimeError('failed to construct distinct pairs')
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze(pair):
    inp, out = pair['input'], pair['output']
    assert inp.shape == out.shape
    found = _find_core(inp)
    assert found is not None
    background, (r0, r1, c0, c1, color) = found
    expected = np.full_like(inp, background)
    expected[r0:r1, c0:c1] = color
    assert np.array_equal(out, expected) and not np.array_equal(inp, out)
    assert _digest(pair) not in SOURCE_DIGESTS
    area = (r1-r0)*(c1-c0)
    density = (np.count_nonzero(inp != background)-area)/(inp.size-area)
    assert 0.08 <= density <= 0.39
    contact = any(inp[r, c] == color for r in range(inp.shape[0])
                  for c in range(inp.shape[1])
                  if (r0 <= r < r1 and c in (c0-1, c1))
                  or (c0 <= c < c1 and r in (r0-1, r1)))
    return {'background': background, 'color': color, 'shape': (r1-r0, c1-c0),
            'density': density, 'position': (r0, c0),
            'contact': contact,
            'signal_noise': int(np.count_nonzero(inp == color)) > area,
            'canvas': inp.shape}


def validate_generated_puzzle(puzzle: dict) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list)
    nt, ns = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ns <= 4 and nt > ns
    seen_inputs, seen_pairs = set(), set()
    analyses = {'train': [], 'test': []}
    for section in ('train', 'test'):
        for pair in puzzle[section]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            for grid in pair.values():
                assert isinstance(grid, np.ndarray) and grid.ndim == 2
                assert np.issubdtype(grid.dtype, np.integer)
                assert 1 <= grid.shape[0] <= 30 and 1 <= grid.shape[1] <= 30
                assert np.all((grid >= 0) & (grid <= 9))
            ik = _key(pair['input'])
            pk = (ik, _key(pair['output']))
            assert ik not in seen_inputs and pk not in seen_pairs
            seen_inputs.add(ik)
            seen_pairs.add(pk)
            analyses[section].append(_analyze(pair))
    train, tests = analyses['train'], analyses['test']
    assert len({x['color'] for x in train}) >= 2
    assert len({'wide' if h < w else 'tall' if h > w else 'square'
                for h, w in (x['shape'] for x in train)}) >= 2
    assert any(x['density'] < 0.18 for x in train)
    assert any(x['density'] >= 0.26 for x in train)
    assert any(x['contact'] for x in train)
    assert len({x['shape'] for x in tests}) == len(tests)
    backgrounds = {x['background'] for x in train}
    if len(backgrounds) == 1:
        assert all(x['background'] in backgrounds for x in tests)


def _signature(puzzle):
    return tuple((section, tuple((_key(p['input']), _key(p['output']))
                                 for p in puzzle[section])) for section in ('train', 'test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    assert len({len(p['train']) for p in puzzles}) >= 4
    assert len({len(p['test']) for p in puzzles}) >= 3
    for seed, nt, ns in ((101, 2, 1), (102, 10, 1), (103, 10, 4), (104, 6, 2),
                         (201, None, 1), (202, None, 4), (203, 2, None),
                         (204, 10, None)):
        p = generate_puzzle(seed, nt, ns)
        if nt is not None:
            assert len(p['train']) == nt
        if ns is not None:
            assert len(p['test']) == ns
        validate_generated_puzzle(p)
    assert _signature(generate_puzzle(777, 7, 3)) == _signature(generate_puzzle(777, 7, 3))
    assert _signature(generate_puzzle(777, 7, 3)) != _signature(generate_puzzle(778, 7, 3))
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(300, *args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid count accepted')
    infos = [_analyze(pair) for puzzle in puzzles for section in ('train', 'test')
             for pair in puzzle[section]]
    assert len({x['canvas'] for x in infos}) >= 45
    assert len({x['shape'] for x in infos}) >= 25
    assert len({x['position'] for x in infos}) >= 55
    assert len({x['background'] for x in infos}) >= 8
    assert len({x['color'] for x in infos}) >= 8
    assert any(x['canvas'] == (16, 16) for x in infos)
    assert any(x['signal_noise'] for x in infos)
    assert any(not x['signal_noise'] for x in infos)
    assert min(x['density'] for x in infos) < 0.15
    assert max(x['density'] for x in infos) > 0.32
    assert {(3, 5), (5, 2), (2, 6)}.issubset({x['shape'] for x in infos})
    assert any(len({q['input'].shape for section in ('train', 'test')
                    for q in p[section]}) >= 3 for p in puzzles)
    assert any(len({ _analyze(q)['background'] for q in p['train']}) > 1
               for p in puzzles)
    assert any(len({ _analyze(q)['background'] for q in p['train']}) == 1
               for p in puzzles)
    assert any(len(np.unique(q['input'])) < 10 for p in puzzles
               for section in ('train', 'test') for q in p[section])
    repeated, novel = 0, 0
    adjacent, contacts = 0, 0
    for p in puzzles:
        train_colors = {_analyze(q)['color'] for q in p['train']}
        for q in p['test']:
            if _analyze(q)['color'] in train_colors:
                repeated += 1
            else:
                novel += 1
        for section in ('train', 'test'):
            for q in p[section]:
                a = _analyze(q)
                contacts += a['contact']
                bg, (r0, r1, c0, c1, _) = _find_core(q['input'])
                noise = q['input'].copy()
                noise[r0:r1, c0:c1] = bg
                adjacent += int(np.count_nonzero((noise[:-1] == noise[1:])
                                                  & (noise[:-1] != bg)))
                adjacent += int(np.count_nonzero((noise[:, :-1] == noise[:, 1:])
                                                  & (noise[:, :-1] != bg)))
    assert repeated >= 5 and novel >= 5
    assert adjacent >= 100 and contacts >= 42
