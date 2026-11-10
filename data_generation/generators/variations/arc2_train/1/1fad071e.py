"""Generate unary tallies of separated, solid target-color 2 by 2 squares."""
import numpy as np

SOURCE = (
    ("110200002", "110001100", "000201100", "000000001", "011022000", "011022002", "000000000", "000220110", "010220110"),
    ("000022001", "011022000", "011000022", "000000022", "102200000", "002201100", "000001100", "000000000", "010000001"),
    ("220110000", "220110011", "100000011", "022000000", "022011010", "000011000", "000020000", "011000022", "011001022"),
    ("000002201", "110102200", "110000000", "000001100", "022001100", "022000000", "100000220", "220110220", "220110000"),
)
SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
FRAGMENTS = (
    ((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)), ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (1, 0), (2, 0), (3, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
)
FRAG_BY_SIZE = {k: tuple(f for f in FRAGMENTS if len(f) == k) for k in range(1, 5)}


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _key(grid):
    return grid.shape, tuple(int(x) for x in grid.flat)


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def _source_keys():
    return {((9, 9), tuple(int(c) for row in rows for c in row)) for rows in SOURCE}


def _count_arg(name, value, lo, hi):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def _pair_counts(rng, seed, num_train, num_test):
    train = _count_arg('num_train', num_train, 2, 10)
    test = _count_arg('num_test', num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError('num_train must exceed num_test')
    elif train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = int(rng.integers(test + 1, 11))
    else:
        test = int(rng.integers(1, min(4, train - 1) + 1))
    return train, test


def _components(grid, color):
    mask = grid == color
    seen = np.zeros(grid.shape, dtype=bool)
    result = []
    for rr, cc in zip(*np.where(mask)):
        rr, cc = int(rr), int(cc)
        if seen[rr, cc]:
            continue
        seen[rr, cc] = True
        stack, cells = [(rr, cc)], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]
                            and mask[nr, nc] and not seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
        result.append(cells)
    return result


def _is_square(cells):
    if len(cells) != 4:
        return False
    top, left = min(r for r, _ in cells), min(c for _, c in cells)
    return set(cells) == {(top + dr, left + dc) for dr, dc in SQUARE}


def _windows(grid, color):
    m = grid == color
    return int(np.count_nonzero(m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]))


def _tally(count, bg, target):
    out = np.full((1, 5), bg, dtype=np.int8)
    out[0, :count] = target
    return out


def _exact_noise(rng, pixels):
    result = []
    while pixels:
        size = min(4, pixels)
        if pixels > 4 and rng.random() < 0.4:
            size = int(rng.integers(1, min(4, pixels) + 1))
        options = FRAG_BY_SIZE[size]
        result.append(options[int(rng.integers(len(options)))])
        pixels -= size
    return result


def _noise(rng, maximum=4):
    if maximum >= 4 and rng.random() < 0.08:
        return [FRAGMENTS[0]] * 4
    return [FRAGMENTS[int(rng.integers(len(FRAGMENTS)))]
            for _ in range(int(rng.integers(maximum + 1)))]


def _size(rng, load, contrast=False):
    if contrast:
        return int(rng.integers(13, 19)), int(rng.integers(13, 19))
    mode = int(rng.integers(6))
    if mode == 0 and load <= 15:
        return int(rng.integers(7, 11)), int(rng.integers(7, 11))
    if mode == 2:
        return int(rng.integers(17, 25)), int(rng.integers(17, 25))
    if mode == 3:
        return int(rng.integers(8, 14)), int(rng.integers(17, 31))
    if mode == 4:
        return int(rng.integers(17, 31)), int(rng.integers(8, 14))
    if mode == 5:
        return int(rng.integers(25, 31)), int(rng.integers(25, 31))
    return int(rng.integers(10, 17)), int(rng.integers(10, 17))


def _put(grid, rng, color, bg, piece):
    height, width = 1 + max(r for r, _ in piece), 1 + max(c for _, c in piece)
    h, w = grid.shape
    if height > h or width > w:
        return False
    cols = w - width + 1
    for i in rng.permutation((h - height + 1) * cols):
        r, c = divmod(int(i), cols)
        if any(grid[r + dr, c + dc] != bg for dr, dc in piece):
            continue
        if np.any(grid[max(0, r-1):min(h, r+height+1),
                       max(0, c-1):min(w, c+width+1)] == color):
            continue
        for dr, dc in piece:
            grid[r + dr, c + dc] = color
        return True
    return False


def _build(rng, plan, target):
    bg, ds = plan['bg'], plan['dcolors']
    objects = [(target, SQUARE)] * plan['target_count']
    objects += [(ds[i % len(ds)], SQUARE) for i in range(plan['dcount'])]
    objects += [(target, f) for f in plan['tnoise']]
    objects += [(ds[(i + plan['dcount']) % len(ds)], f) for i, f in enumerate(plan['dnoise'])]
    for attempt in range(12):
        h, w = plan['size']
        if attempt >= 6:
            h, w = max(h, 16), max(w, 16)
        if attempt >= 10:
            h, w = max(h, 22), max(w, 22)
        grid = np.full((h, w), bg, dtype=np.int8)
        ordered = [objects[int(i)] for i in rng.permutation(len(objects))]
        ordered.sort(key=lambda obj: -len(obj[1]))
        if all(_put(grid, rng, color, bg, piece) for color, piece in ordered):
            if _windows(grid, target) == plan['target_count']:
                return grid
    raise RuntimeError('could not pack separated objects')


def _background(grid):
    values, counts = np.unique(grid, return_counts=True)
    _require(np.count_nonzero(counts == max(counts)) == 1, 'background is not uniquely dominant')
    return int(values[int(np.argmax(counts))])


def _plan(rng, target, count, *, size=None, bg=None, dcolors=None, dcount=None,
          tnoise=None, dnoise=None, contrast=False):
    if bg is None:
        bg = int(rng.choice([c for c in range(10) if c != target]))
    if dcount is None:
        dcount = int(rng.integers(0, 6))
    if tnoise is None:
        tnoise = _noise(rng)
    if dnoise is None:
        dnoise = _noise(rng)
    if dcolors is None:
        available = [c for c in range(10) if c not in (target, bg)]
        n = 2 if dcount + len(dnoise) >= 2 and rng.random() < 0.38 else 1
        dcolors = tuple(int(x) for x in rng.choice(available, size=n, replace=False))
    load = 4*(count+dcount) + sum(map(len, tnoise)) + sum(map(len, dnoise))
    if size is None:
        size = _size(rng, load, contrast)
    return dict(bg=bg, dcolors=tuple(dcolors), size=size, target_count=count,
                dcount=dcount, tnoise=tnoise, dnoise=dnoise)


def _make_pair(rng, plan, target):
    grid = _build(rng, plan, target)
    _require(_background(grid) == plan['bg'], 'foreground outnumbered background')
    count = _windows(grid, target)
    _require(count == plan['target_count'], 'square count changed in packing')
    return {'input': grid, 'output': _tally(count, plan['bg'], target)}


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    ntrain, ntest = _pair_counts(rng, seed, num_train, num_test)
    target = int(rng.integers(10))
    if seed % 5 == 0:
        a, b = 2, 4
    else:
        a = int(rng.integers(1, 4))
        b = int(rng.integers(a + 1, min(5, a + 2) + 1))
    if rng.random() < 0.5:
        a, b = b, a
    anoise = _exact_noise(rng, 4*max(0, b-a))
    bnoise = _exact_noise(rng, 4*max(0, a-b))
    same_hist = seed % 3 == 0
    bg = int(rng.choice([c for c in range(10) if c != target]))
    dc = int(rng.choice([c for c in range(10) if c not in (target, bg)]))
    dcount = int(rng.integers(1, 4))
    size = _size(rng, 0, contrast=True)
    shared_dnoise = _noise(rng, 2) if same_hist else None
    train_plans = [
        _plan(rng, target, a, size=size, bg=bg, dcolors=(dc,), dcount=dcount,
              tnoise=anoise, dnoise=shared_dnoise),
        _plan(rng, target, b, size=size, bg=bg, dcolors=(dc,) if same_hist else None,
              dcount=dcount if same_hist else None, tnoise=bnoise, dnoise=shared_dnoise),
    ]
    for index in range(2, ntrain):
        if index == 2 and seed % 5 == 0:
            # Keep some compact examples with the source's square-and-dot vocabulary.
            train_plans.append(_plan(
                rng, target, 2, size=(9, 9), dcount=2,
                tnoise=[FRAGMENTS[0], FRAGMENTS[0]],
                dnoise=[FRAGMENTS[0]],
            ))
        else:
            train_plans.append(_plan(rng, target, int(rng.integers(6))))
    first_test = 3 if seed % 5 == 0 else seed % 6
    remaining = [c for c in range(6) if c != first_test]
    rng.shuffle(remaining)
    test_counts = [first_test] + remaining[:ntest-1]
    test_plans = []
    for i, count in enumerate(test_counts):
        other_bg = int(rng.choice([c for c in range(10) if c not in (target, bg)])) if i == 0 else None
        test_plans.append(_plan(rng, target, count, bg=other_bg))
    puzzle = {'train': [], 'test': []}
    seen, sources = set(), _source_keys()
    for split, plans in (('train', train_plans), ('test', test_plans)):
        for plan in plans:
            for _ in range(8):
                pair = _make_pair(rng, plan, target)
                key = _key(pair['input'])
                if key not in seen and key not in sources:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('could not produce a unique input')
    validate_generated_puzzle(puzzle)
    return puzzle


def _infer_target(train):
    candidates = set()
    for pair in train:
        bg = _background(pair['input'])
        candidates.update(int(c) for c in np.unique(pair['output']) if c != bg)
    _require(len(candidates) == 1, 'training must reveal one tally color')
    return next(iter(candidates))


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list), 'split types')
    _require(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
             and len(puzzle['train']) > len(puzzle['test']), 'pair counts')
    seen, source = set(), _source_keys()
    for pair in puzzle['train'] + puzzle['test']:
        _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
        for name in ('input', 'output'):
            grid = pair[name]
            _require(isinstance(grid, np.ndarray) and grid.ndim == 2, 'grid type')
            _require(all(1 <= d <= 30 for d in grid.shape) and
                     np.issubdtype(grid.dtype, np.integer) and
                     np.all((grid >= 0) & (grid <= 9)), 'grid bounds')
        key = _key(pair['input'])
        _require(key not in seen and key not in source, 'duplicate or source input')
        seen.add(key)
        _require(pair['output'].shape == (1, 5), 'output dimensions')
    target = _infer_target(puzzle['train'])
    train_counts = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            grid = pair['input']
            bg = _background(grid)
            _require(bg != target, 'target equals background')
            count = 0
            for color in (int(c) for c in np.unique(grid) if c != bg):
                pieces = _components(grid, color)
                squares = sum(_is_square(p) for p in pieces)
                _require(all(len(p) <= 4 for p in pieces), 'oversized same-color object')
                _require(squares == _windows(grid, color), 'overlapping or accidental square')
                if color == target:
                    count = squares
            _require(0 <= count <= 5 and np.array_equal(pair['output'], _tally(count, bg, target)),
                     'incorrect target square tally')
            if split == 'train':
                train_counts.append(count)
    _require(any(0 < c < 5 for c in train_counts) and len(set(train_counts)) >= 2,
             'training does not reveal tally and variable count')
    first, second = puzzle['train'][:2]
    _require(np.count_nonzero(first['input'] == target) ==
             np.count_nonzero(second['input'] == target) and
             train_counts[0] != train_counts[1], 'missing equal-pixel-count contrast')


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_puzzle_key(p) for p in puzzles}) >= 40, 'puzzle diversity')
    _require(len({len(p['train']) for p in puzzles}) >= 4 and
             len({len(p['test']) for p in puzzles}) >= 3, 'pair-count diversity')
    train_counts, test_counts, sizes, roles, layouts, fragment_sizes = set(), set(), set(), set(), set(), set()
    no_noise = multi_dcolor = same_hist = False
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        target = _infer_target(puzzle['train'])
        roles.add(target)
        x, y = puzzle['train'][:2]
        histogram = lambda p: tuple(int(np.count_nonzero(p['input'] == c)) for c in range(10))
        same_hist |= histogram(x) == histogram(y)
        for split, counts in (('train', train_counts), ('test', test_counts)):
            for pair in puzzle[split]:
                grid = pair['input']
                bg = _background(grid)
                count = _windows(grid, target)
                counts.add(count)
                sizes.add(grid.shape)
                normalized = np.where(grid == bg, 0, np.where(grid == target, 1, 2))
                layouts.add(_key(normalized))
                noise_count, dcolor_count = 0, 0
                for color in (int(c) for c in np.unique(grid) if c != bg):
                    if color != target:
                        dcolor_count += 1
                    for piece in _components(grid, color):
                        if not _is_square(piece):
                            fragment_sizes.add(len(piece))
                            noise_count += 1
                no_noise |= noise_count == 0
                multi_dcolor |= dcolor_count >= 2
                _require(np.array_equal(pair['output'], _tally(count, bg, target)),
                         'window scan disagrees with output')
    _require(train_counts == set(range(6)) and test_counts == set(range(6)), 'count coverage')
    _require(fragment_sizes == {1, 2, 3, 4} and no_noise and multi_dcolor,
             'noise or distractor-color coverage')
    _require(same_hist and len(sizes) >= 30 and len(layouts) >= 100 and len(roles) >= 8,
             'histogram, layout, or role diversity')
    _require(any(h > w for h, w in sizes) and any(w > h for h, w in sizes)
             and any(h >= 25 and w >= 25 for h, w in sizes)
             and any(h <= 10 and w <= 10 for h, w in sizes), 'size regimes')
    _require(_puzzle_key(generate_puzzle(777, 7, 3)) == _puzzle_key(generate_puzzle(777, 7, 3))
             and _puzzle_key(generate_puzzle(777, 7, 3)) != _puzzle_key(generate_puzzle(778, 7, 3)),
             'determinism')
    for tr, te in ((2, 1), (10, 1), (10, 4), (None, 1), (None, 4), (2, None), (10, None)):
        p = generate_puzzle(1000 + (tr or 0) + (te or 0), tr, te)
        _require((tr is None or len(p['train']) == tr) and (te is None or len(p['test']) == te),
                 'requested counts ignored')
    for tr, te in ((1, None), (11, None), (None, 0), (None, 5),
                   (2, 2), (2, 3), (3, 4), (4, 4)):
        try:
            generate_puzzle(99, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal pair count accepted')
