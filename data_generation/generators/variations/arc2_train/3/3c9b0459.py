"""Generate solved whole-grid half-turn ARC puzzles."""
import numpy as np

SOURCE = frozenset({
    (8, 8, 8, 5, 5, 8, 8, 5, 5),
    (9, 2, 4, 2, 4, 4, 2, 9, 2),
    (3, 2, 9, 9, 9, 9, 2, 3, 3),
    (2, 2, 1, 2, 1, 2, 2, 8, 1),
    (6, 4, 4, 6, 6, 4, 4, 6, 7),
})


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _count(name, value, low, high):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be between {low} and {high}')
    return value


def _counts(seed, num_train, num_test):
    train = _count('num_train', num_train, 2, 10)
    test = _count('num_test', num_test, 1, 4)
    if train is not None and test is not None:
        if train <= test:
            raise ValueError('num_train must exceed num_test')
    elif train is None and test is None:
        test = 1 + seed % 4
        train = test + 1 + (seed // 4) % (10 - test)
    elif train is None:
        train = test + 1 + seed % (10 - test)
    else:
        test = 1 + seed % min(4, train - 1)
    return train, test


def _shape(rng, category=None):
    if category is None:
        category = rng.choice(('source', 'small', 'medium', 'large', 'max', 'elongated'),
                              p=(.24, .23, .20, .14, .09, .10))
    if category == 'source':
        return 3, 3
    if category == 'elongated':
        short, long = int(rng.integers(2, 9)), int(rng.integers(10, 31))
        return (short, long) if rng.integers(2) else (long, short)
    low, high = {'small': (3, 8), 'medium': (7, 16),
                 'large': (15, 24), 'max': (25, 30)}[category]
    if rng.random() < .38:
        side = int(rng.integers(low, high + 1))
        return side, side
    return int(rng.integers(low, high + 1)), int(rng.integers(low, high + 1))


def _shapes(rng, nt, nq):
    mode = rng.choice(('same', 'mixed', 'extrapolate'), p=(.32, .46, .22))
    if mode == 'same':
        shape = _shape(rng)
        return [shape] * nt, [shape] * nq
    if mode == 'extrapolate':
        train = [(3, 3)] + [_shape(rng, 'small') for _ in range(nt - 1)]
        test = []
        largest_train_side = max(max(shape) for shape in train)
        largest_train_area = max(h * w for h, w in train)
        for _ in range(nq):
            for attempt in range(32):
                candidate = _shape(rng, rng.choice(('medium', 'large', 'max', 'elongated')))
                if max(candidate) > largest_train_side and candidate[0] * candidate[1] > largest_train_area:
                    break
            else:
                candidate = (largest_train_side + 1, largest_train_side + 1)
            test.append(candidate)
        return train, test
    train = [_shape(rng) for _ in range(nt)]
    test = [_shape(rng, rng.choice(('source', 'small', 'medium', 'large', 'max', 'elongated'),
                                   p=(.08, .17, .25, .20, .15, .15))) for _ in range(nq)]
    if len(set(train + test)) == 1:
        test[0] = _shape(rng, 'medium')
    return train, test


def _key(grid):
    return (grid.shape[0], grid.shape[1], *map(int, grid.flat))


def _layout_key(grid):
    alternatives = []
    for variant in _transforms(grid):
        labels = {}
        pattern = []
        for raw in variant.flat:
            color = int(raw)
            if color not in labels:
                labels[color] = len(labels)
            pattern.append(labels[color])
        alternatives.append((variant.shape, tuple(pattern)))
    return min(alternatives)


def _transforms(grid):
    return (grid, np.rot90(grid), np.rot90(grid, 2), np.rot90(grid, 3),
            np.fliplr(grid), np.flipud(grid), grid.T, np.rot90(grid.T, 2))


def _diagnostic(grid):
    variants = _transforms(grid)
    return all(variants[2].shape != other.shape or not np.array_equal(variants[2], other)
               for i, other in enumerate(variants) if i != 2)


def _learned_transform_count(train):
    return sum(all(np.array_equal(_transforms(pair['input'])[i], pair['output'])
                   for pair in train) for i in range(8))


def _palette_size(rng, limit, style, shape, diagnostic):
    maximum = min(limit, shape[0] * shape[1], 10)
    if style == 'uniform':
        return 1
    if style == 'sparse':
        maximum = min(maximum, 4)
    if style == 'stripes':
        maximum = min(maximum, min(shape))
    if style == 'symmetric':
        maximum = min(maximum, (shape[0] * shape[1] + 1) // 2)
    minimum = 2 if diagnostic else 1
    if maximum < minimum:
        return maximum
    choices = np.arange(minimum, maximum + 1)
    weights = np.array([.035, .30, .28, .16, .10, .06, .035, .018, .01, .007])[choices - 1]
    return int(rng.choice(choices, p=weights / weights.sum()))


def _palette(rng, pool, size, relationship):
    known = list(pool)
    novel = [color for color in range(10) if color not in known]
    if relationship == 'unseen':
        selected = rng.choice(novel, size=min(size, len(novel)), replace=False)
    elif relationship == 'partial' and size >= 2:
        old_count = min(int(rng.integers(1, size)), len(known))
        new_count = min(size - old_count, len(novel))
        selected = np.r_[rng.choice(known, old_count, replace=False),
                         rng.choice(novel, new_count, replace=False)]
    else:
        selected = rng.choice(known, size=min(size, len(known)), replace=False)
    return tuple(map(int, selected))


def _arrange(rng, shape, palette, style):
    h, w = shape
    colors = np.asarray(palette, dtype=np.int8)
    k = len(colors)
    if style == 'uniform':
        return np.full(shape, colors[0], dtype=np.int8)
    if style == 'random':
        weights = rng.dirichlet(np.full(k, float(rng.choice((.3, .7, 1.5, 4.0)))))
        grid = rng.choice(colors, size=shape, p=weights).astype(np.int8)
    elif style == 'sparse':
        grid = np.full(shape, colors[0], dtype=np.int8)
        maximum = max(k - 1, min(h * w - 1, max(2, h * w // 7)))
        count = int(rng.integers(k - 1, maximum + 1))
        positions = rng.choice(h * w, size=count, replace=False)
        accents = np.resize(colors[1:], count)
        rng.shuffle(accents)
        grid.flat[positions] = accents
    elif style == 'stripes':
        axis = int(rng.integers(2))
        length = h if axis == 0 else w
        bands = int(rng.integers(k, min(length, max(k, 7)) + 1))
        boundaries = np.sort(rng.choice(np.arange(1, length), bands - 1, replace=False))
        band_colors = np.resize(colors, bands)
        rng.shuffle(band_colors)
        line = np.repeat(band_colors, np.diff(np.r_[0, boundaries, length]))
        grid = np.repeat(line[:, None], w, axis=1) if axis == 0 else np.repeat(line[None, :], h, axis=0)
    elif style == 'patch':
        grid = np.full(shape, colors[0], dtype=np.int8)
        for color in colors[1:]:
            for _ in range(int(rng.integers(1, 4))):
                rh, rw = int(rng.integers(1, h + 1)), int(rng.integers(1, w + 1))
                row, col = int(rng.integers(h - rh + 1)), int(rng.integers(w - rw + 1))
                grid[row:row + rh, col:col + rw] = color
    elif style == 'tile':
        th, tw = int(rng.integers(2, min(4, h) + 1)), int(rng.integers(2, min(4, w) + 1))
        tile = rng.choice(colors, size=(th, tw)).astype(np.int8)
        for index, color in enumerate(colors[:th * tw]):
            tile.flat[index] = color
        tile = rng.permutation(tile.reshape(-1)).reshape(th, tw)
        grid = np.tile(tile, ((h + th - 1) // th, (w + tw - 1) // tw))[:h, :w].copy()
    elif style == 'margin' and h >= 3 and w >= 3:
        grid = np.full(shape, colors[0], dtype=np.int8)
        top = int(rng.integers(1, max(2, h // 3 + 1)))
        left = int(rng.integers(1, max(2, w // 3 + 1)))
        bottom, right = max(top + 1, h - top), max(left + 1, w - left)
        grid[top:bottom, left:right] = rng.choice(colors, size=(bottom - top, right - left))
    elif style == 'symmetric':
        grid = np.empty(shape, dtype=np.int8)
        cells = h * w
        for index in range((cells + 1) // 2):
            color = colors[index % k] if index < k else colors[int(rng.integers(k))]
            grid.flat[index] = grid.flat[cells - 1 - index] = color
    else:
        grid = rng.choice(colors, size=shape).astype(np.int8)
    missing = [color for color in colors if not np.any(grid == color)]
    counts = np.bincount(grid.reshape(-1), minlength=10)
    for color in missing:
        candidates = np.flatnonzero(counts[grid.reshape(-1)] > 1)
        position = int(rng.choice(candidates))
        old = int(grid.flat[position])
        grid.flat[position] = color
        counts[old] -= 1
        counts[int(color)] += 1
    return grid


def _pair(rng, shape, pool, relationship, diagnostic, target, seen):
    styles = ('random', 'patch', 'sparse', 'stripes', 'tile', 'margin', 'symmetric', 'uniform')
    weights = ((.29, .19, .13, .12, .10, .08, .06, .03) if not target else
               (.23, .16, .15, .10, .10, .10, .11, .05))
    available = (len(pool) if relationship == 'reuse' else
                 10 - len(pool) if relationship == 'unseen' else 10)
    for attempt in range(160):
        style = str(rng.choice(styles, p=weights))
        if diagnostic and style in ('uniform', 'symmetric'):
            continue
        size = _palette_size(rng, available, style, shape, diagnostic)
        palette = _palette(rng, pool, size, relationship)
        grid = _arrange(rng, shape, palette, style)
        if diagnostic and not _diagnostic(grid):
            continue
        key = _key(grid)
        if key in seen or (shape == (3, 3) and tuple(map(int, grid.flat)) in SOURCE):
            continue
        seen.add(key)
        return {'input': grid, 'output': grid[::-1, ::-1].copy()}
    for _ in range(1000):
        palette = _palette(rng, pool, min(3, available), relationship)
        grid = _arrange(rng, shape, palette, 'random')
        key = _key(grid)
        if key not in seen and (not diagnostic or _diagnostic(grid)) and not (
                shape == (3, 3) and tuple(map(int, grid.flat)) in SOURCE):
            seen.add(key)
            return {'input': grid, 'output': grid[::-1, ::-1].copy()}
    raise RuntimeError('could not construct a distinct input')


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    nt, nq = _counts(seed, num_train, num_test)
    train_shapes, test_shapes = _shapes(rng, nt, nq)
    pool = tuple(map(int, rng.choice(10, size=int(rng.integers(4, 8)), replace=False)))
    palette_regime = str(rng.choice(('reuse', 'partial', 'unseen', 'mixed')))
    seen = set()
    train = [_pair(rng, shape, pool, 'reuse', i < 2, False, seen)
             for i, shape in enumerate(train_shapes)]
    test = [_pair(rng, shape, pool,
                  str(rng.choice(('reuse', 'partial', 'unseen'))) if palette_regime == 'mixed' else palette_regime,
                  False, True, seen) for shape in test_shapes]
    puzzle = {'train': train, 'test': test}
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle schema')
    _require(isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list), 'splits must be lists')
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, 'illegal pair counts')
    seen = set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair schema')
            a, b = pair['input'], pair['output']
            for grid in (a, b):
                _require(isinstance(grid, np.ndarray) and grid.ndim == 2, 'grid must be a 2D NumPy array')
                _require(all(1 <= d <= 30 for d in grid.shape), 'grid outside ARC size limit')
                _require(np.issubdtype(grid.dtype, np.integer), 'grid must contain integers')
                _require(int(grid.min()) >= 0 and int(grid.max()) <= 9, 'illegal ARC color')
            _require(a.shape == b.shape and np.array_equal(b, a[::-1, ::-1]), 'incorrect half-turn')
            key = _key(a)
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            _require(not (a.shape == (3, 3) and tuple(map(int, a.flat)) in SOURCE), 'copied source pair')
    _require(_learned_transform_count(puzzle['train']) == 1,
             'training does not uniquely establish the half-turn among D4 transforms')


def _equal(first, second):
    return all(len(first[split]) == len(second[split]) and all(
        np.array_equal(a[name], b[name]) for a, b in zip(first[split], second[split])
        for name in ('input', 'output')) for split in ('train', 'test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        learned = [i for i in range(8) if all(
            np.array_equal(_transforms(pair['input'])[i], pair['output'])
            for pair in puzzle['train'])]
        _require(len(learned) == 1, 'training leaves an alternative D4 rule')
        for pair in puzzle['test']:
            predicted = _transforms(pair['input'].copy())[learned[0]]
            _require(np.array_equal(predicted, pair['output']),
                     'training-only prediction failed')
    _require(all(not _equal(puzzles[0], puzzle) for puzzle in puzzles[1:]),
             'different seeds produced the same puzzle')
    _require(len({len(p['train']) for p in puzzles}) >= 4, 'train counts lack diversity')
    _require(len({len(p['test']) for p in puzzles}) >= 3, 'test counts lack diversity')
    pairs = [pair for puzzle in puzzles for split in ('train', 'test') for pair in puzzle[split]]
    shapes = {pair['input'].shape for pair in pairs}
    _require((3, 3) in shapes and any(h != w for h, w in shapes) and
             any(h == w and h > 3 for h, w in shapes) and
             any(max(h, w) == 30 for h, w in shapes), 'dimension range too narrow')
    _require(any(len({p['input'].shape for split in ('train', 'test') for p in puzzle[split]}) > 1
                 for puzzle in puzzles), 'no mixed-size puzzle')
    _require(any(all(p['input'].shape == puzzle['train'][0]['input'].shape
                     for split in ('train', 'test') for p in puzzle[split]) for puzzle in puzzles),
             'no same-size puzzle')
    _require(any(len(np.unique(p['input'])) >= 4 for p in pairs), 'no rich palette')
    _require(any(p['input'].shape == (3, 3) and len(np.unique(p['input'])) >= 4
                 for p in pairs), 'no rich source-size grid')
    _require(len({_layout_key(p['input']) for p in pairs}) >= 100,
             'too few color-normalized layouts')
    _require(len({(p['input'].size, tuple(sorted(np.unique(
        p['input'], return_counts=True)[1]))) for p in pairs}) >= 40,
             'too few color-frequency profiles')
    _require(any(np.array_equal(p['input'], p['output']) for p in pairs), 'no symmetric pair')
    _require(any(len(np.unique(p['input'])) == 1 for p in pairs), 'no uniform pair')
    _require(any(np.count_nonzero(p['input'] != np.bincount(
        p['input'].flat, minlength=10).argmax()) == 1 for p in pairs), 'no lone accent')
    _require(any(h > w for h, w in shapes) and any(w > h for h, w in shapes) and
             any(h % 2 == 0 or w % 2 == 0 for h, w in shapes) and
             any(h % 2 and w % 2 for h, w in shapes), 'aspect or parity range too narrow')
    _require(any(min(p['input'].size for p in puzzle['test']) >
                 max(p['input'].size for p in puzzle['train']) and
                 min(max(p['input'].shape) for p in puzzle['test']) >
                 max(max(p['input'].shape) for p in puzzle['train'])
                 for puzzle in puzzles), 'no size extrapolation')
    palette_relations = set()
    different_test_palettes = False
    for puzzle in puzzles:
        train_colors = set().union(*(set(map(int, p['input'].flat))
                                     for p in puzzle['train']))
        test_palettes = [set(map(int, p['input'].flat)) for p in puzzle['test']]
        different_test_palettes |= len({frozenset(p) for p in test_palettes}) > 1
        for palette in test_palettes:
            if palette <= train_colors:
                palette_relations.add('reuse')
            elif palette.isdisjoint(train_colors):
                palette_relations.add('unseen')
            else:
                palette_relations.add('partial')
    _require(palette_relations == {'reuse', 'partial', 'unseen'} and
             different_test_palettes, 'test palette relationships lack diversity')
    for seed, train, test in ((725, None, None), (72, 10, 4),
                              (20000, 2, 1), (20001, 10, 1), (20002, 10, 4),
                              (30001, None, 1), (30002, None, 4),
                              (30003, 2, None), (30004, 10, None)):
        p = generate_puzzle(seed, train, test)
        validate_generated_puzzle(p)
        _require((train is None or len(p['train']) == train) and
                 (test is None or len(p['test']) == test), 'requested count not honored')
        _require(_equal(p, generate_puzzle(seed, train, test)), 'nondeterministic puzzle')
    for args in ((1, None), (11, None), (None, 0), (None, 5),
                 (2, 2), (3, 4), (2.0, 1)):
        try:
            generate_puzzle(99, *args)
        except ValueError:
            continue
        raise AssertionError('illegal count was accepted')
