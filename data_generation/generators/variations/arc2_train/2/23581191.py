"""ARC puzzles where singleton markers project full orthogonal crosses."""
import numpy as np

_SOURCE = ((1, 3, 7, 6), (2, 2, 6, 6), (1, 4, 6, 1))


def _check(ok, message):
    if not bool(ok):
        raise AssertionError(message)


def _optional_count(name, value, lo, hi):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if not lo <= value <= hi:
        raise ValueError(f'{name} outside {lo}..{hi}')
    return value


def _counts(rng, seed, num_train, num_test):
    t = _optional_count('num_train', num_train, 2, 10)
    q = _optional_count('num_test', num_test, 1, 4)
    if t is None and q is None:
        q = 1 + seed % 4
        t = q + 1 + (seed // 4) % (10 - q)
    elif t is None:
        t = int(rng.integers(q + 1, 11))
    elif q is None:
        q = int(rng.integers(1, min(4, t - 1) + 1))
    if t <= q:
        raise ValueError('num_train must exceed num_test')
    return t, q


def _shape(rng, n, target):
    lower = max(3, n + 1)
    mode = int(rng.integers(0, 7))
    if mode == 0:
        h = w = int(rng.integers(lower, 16))
    elif mode == 1:
        h = int(rng.integers(lower, 9))
        w = int(rng.integers(h + 1, 31))
    elif mode == 2:
        w = int(rng.integers(lower, 9))
        h = int(rng.integers(w + 1, 31))
    elif mode == 3:
        h, w = [int(rng.integers(lower, 31)) for _ in range(2)]
    elif mode == 4:
        h, w = [int(rng.integers(lower, 11)) for _ in range(2)]
    elif mode == 5:
        h, w = [int(rng.integers(max(12, lower), 31)) for _ in range(2)]
    else:
        upper = 9 if target else 14
        h, w = [int(rng.integers(lower, upper)) for _ in range(2)]
    return h, w


def _axis(length, n, mode, rng):
    if mode == 0 and length >= n + 2:
        pool = np.arange(1, length - 1)
        return sorted(int(x) for x in rng.choice(pool, n, replace=False))
    if mode == 1:
        start = int(rng.integers(0, length - n + 1))
        return list(range(start, start + n))
    if mode == 2 and n >= 2:
        middle = rng.choice(np.arange(1, length - 1), n - 2, replace=False)
        return sorted([0, length - 1] + [int(x) for x in middle])
    return sorted(int(x) for x in rng.choice(length, n, replace=False))


def _coords(h, w, n, orientation, rng):
    mode = int(rng.choice(4, p=[0.37, 0.22, 0.16, 0.25]))
    rows, cols = _axis(h, n, mode, rng), _axis(w, n, mode, rng)
    if n == 1:
        return [(rows[0], cols[0])]
    if n == 2:
        return list(zip(rows, cols if orientation > 0 else cols[::-1]))
    first = [cols[0], cols[-1]] if orientation > 0 else [cols[-1], cols[0]]
    rest = [c for c in cols if c not in first]
    rng.shuffle(rest)
    return list(zip(rows, first + rest))


def _build(grid, background, markers, overlap):
    h, w = grid.shape
    output = np.full((h, w), background, dtype=np.int8)
    coverage = np.zeros((h, w), dtype=np.int8)
    masks = []
    for color, (row, col) in markers:
        mask = np.zeros((h, w), dtype=bool)
        mask[row, :] = True
        mask[:, col] = True
        coverage += mask
        masks.append((color, mask))
    for color, mask in masks:
        output[mask & (coverage == 1)] = color
    output[coverage >= 2] = overlap
    return output


def _source_input(grid):
    if grid.shape != (9, 9):
        return False
    for r1, c1, r2, c2 in _SOURCE:
        source = np.zeros((9, 9), dtype=np.int8)
        source[r1, c1], source[r2, c2] = 8, 7
        if np.array_equal(grid, source):
            return True
    return False


def _key(grid):
    return grid.shape, grid.tobytes()


def _orientation(split, index, policy, rng):
    if policy == 0:
        return 1 if split == 'train' else -1
    if policy == 1:
        return -1 if split == 'train' else 1
    if policy == 2:
        return 1
    if policy == 3:
        return -1
    if split == 'train' and index < 2:
        return 1 if index == 0 else -1
    return int(rng.choice((-1, 1)))


def _marker_counts(style, train_count, test_count, rng):
    if style == 0:
        return [2] * train_count, [2] * test_count
    train = [2, int(rng.integers(2, 4) if style == 1 else rng.integers(3, 6))]
    limit = 3 if style == 1 else 5
    train.extend(int(rng.integers(1, limit + 1)) for _ in range(train_count - 2))
    test = [int(rng.integers(2, max(train) + 1))]
    test.extend(int(rng.integers(1, max(train) + 1)) for _ in range(test_count - 1))
    return train, test


def _palette(rng, overlap, n, style, first_colors, first_bg, split, index):
    available = [c for c in range(10) if c != overlap]
    if first_colors is not None and style == 0:
        return first_bg, first_colors[:]
    if split == 'train' and index == 1:
        bg = int(rng.choice([c for c in available if c != first_bg and c not in first_colors]))
        colors = list(reversed(first_colors[:2]))
        extras = [c for c in available if c != bg and c not in colors]
        colors.extend(int(c) for c in rng.choice(extras, n - 2, replace=False))
        return bg, colors
    bg = int(rng.choice(available))
    colors = [int(c) for c in rng.choice([c for c in available if c != bg], n, replace=False)]
    return bg, colors


def _pair(rng, n, style, overlap, first_colors, first_bg, split, index, policy):
    bg, colors = _palette(rng, overlap, n, style, first_colors, first_bg, split, index)
    h, w = _shape(rng, n, split == 'test')
    coords = _coords(h, w, n, _orientation(split, index, policy, rng), rng)
    grid = np.full((h, w), bg, dtype=np.int8)
    markers = list(zip(colors, coords))
    for color, (row, col) in markers:
        grid[row, col] = color
    return grid, _build(grid, bg, markers, overlap), bg, colors


def generate_puzzle(seed: int, num_train: int | None = None,
                    num_test: int | None = None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a non-negative integer')
    seed = int(seed)
    rng = np.random.default_rng(seed)
    train_count, test_count = _counts(rng, seed, num_train, num_test)
    overlap = int(rng.integers(0, 10))
    style, policy = int(rng.integers(0, 3)), int(rng.integers(0, 5))
    train_ns, test_ns = _marker_counts(style, train_count, test_count, rng)
    puzzle, used = {'train': [], 'test': []}, set()
    first_colors = first_bg = None
    for split, ns in (('train', train_ns), ('test', test_ns)):
        for index, n in enumerate(ns):
            for attempt in range(80):
                child = np.random.default_rng(int(rng.integers(0, 2**63)))
                grid, output, bg, colors = _pair(child, n, style, overlap,
                                                  first_colors, first_bg, split, index, policy)
                if _key(grid) in used or _source_input(grid):
                    continue
                used.add(_key(grid))
                puzzle[split].append({'input': grid, 'output': output})
                if first_colors is None:
                    first_colors, first_bg = colors, bg
                break
            else:
                raise RuntimeError('could not construct a unique pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def _grid(grid, label):
    _check(isinstance(grid, np.ndarray), f'{label} must be an ndarray')
    _check(grid.ndim == 2, f'{label} must be two-dimensional')
    _check(all(1 <= x <= 30 for x in grid.shape), f'{label} has invalid dimensions')
    _check(np.issubdtype(grid.dtype, np.integer), f'{label} must be integer')
    _check(np.all((grid >= 0) & (grid <= 9)), f'{label} has invalid colors')


def _decode(grid):
    h, w = grid.shape
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[np.argmax(counts)])
    marker_colors = [int(v) for v, count in zip(values, counts) if v != bg and count == 1]
    n = len(marker_colors)
    _check(1 <= n <= 5 and len(values) == n + 1, 'invalid singleton colors')
    _check(int(np.max(counts)) == h * w - n, 'background is not uniform')
    markers = []
    for color in marker_colors:
        row, col = np.argwhere(grid == color)[0]
        markers.append((color, (int(row), int(col))))
    _check(len({pos[0] for _, pos in markers}) == n, 'marker rows repeat')
    _check(len({pos[1] for _, pos in markers}) == n, 'marker columns repeat')
    return bg, markers


def _inspect(grid, output):
    _check(grid.shape == output.shape, 'dimensions changed')
    h, w = grid.shape
    _check(3 <= h <= 30 and 3 <= w <= 30, 'axes must be 3 to 30')
    _check(not np.array_equal(grid, output), 'no cross expansion')
    bg, markers = _decode(grid)
    n = len(markers)
    _check(h > n and w > n, 'crosses leave no background')
    added = set(int(x) for x in np.unique(output)) - set(int(x) for x in np.unique(grid))
    _check(len(added) == (1 if n > 1 else 0), 'wrong number of new colors')
    overlap = next(iter(added)) if added else None
    for row in range(h):
        for col in range(w):
            hits = [color for color, (mr, mc) in markers if row == mr or col == mc]
            expected = bg if not hits else (hits[0] if len(hits) == 1 else overlap)
            _check(int(output[row, col]) == expected, f'incorrect projection at {(row, col)}')
    if n > 1:
        _check(np.count_nonzero(output == overlap) == n * (n - 1), 'wrong crossing count')
    _check(np.count_nonzero(output == bg) == (h - n) * (w - n), 'wrong background extent')
    return {'shape': (h, w), 'background': bg, 'markers': markers, 'overlap': overlap}


def _predict(train_pairs, test_input):
    extras = set()
    for pair in train_pairs:
        extras.update(set(int(x) for x in np.unique(pair['output'])) -
                      set(int(x) for x in np.unique(pair['input'])))
    _check(len(extras) == 1, 'training does not reveal overlap color')
    overlap = next(iter(extras))
    bg, markers = _decode(test_input)
    predicted = np.full(test_input.shape, bg, dtype=np.int8)
    for row in range(test_input.shape[0]):
        for col in range(test_input.shape[1]):
            hits = [color for color, (mr, mc) in markers if row == mr or col == mc]
            if hits:
                predicted[row, col] = hits[0] if len(hits) == 1 else overlap
    return predicted


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _check(isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list), 'splits must be lists')
    t, q = len(puzzle['train']), len(puzzle['test'])
    _check(2 <= t <= 10 and 1 <= q <= 4 and t > q, 'invalid pair counts')
    used_inputs, used_pairs = set(), set()
    records = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            grid, output = pair['input'], pair['output']
            _grid(grid, 'input')
            _grid(output, 'output')
            key, pair_key = _key(grid), (_key(grid), _key(output))
            _check(key not in used_inputs and pair_key not in used_pairs, 'duplicate pair or input')
            used_inputs.add(key)
            used_pairs.add(pair_key)
            _check(not _source_input(grid), 'source input copied')
            records[split].append(_inspect(grid, output))
    train_overlap = {r['overlap'] for r in records['train'] if r['overlap'] is not None}
    _check(len(train_overlap) == 1, 'training needs one overlap color')
    overlap = next(iter(train_overlap))
    _check(all(r['overlap'] in (None, overlap) for rs in records.values() for r in rs),
           'overlap convention changed')
    _check(any(len(r['markers']) == 2 for r in records['train']), 'training needs a two-marker example')
    maximum = max(len(r['markers']) for r in records['train'])
    _check(all(len(r['markers']) <= maximum for r in records['test']),
           'test marker count exceeds demonstrations')
    for pair in puzzle['test']:
        _check(np.array_equal(_predict(puzzle['train'], pair['input']), pair['output']),
               'test not recoverable from training')


def _signature(puzzle):
    return tuple((split, _key(pair['input']), _key(pair['output']))
                 for split in ('train', 'test') for pair in puzzle[split])


def self_test() -> None:
    cases = ((17, None, None), (314159, 7, 3), (500, 2, 1), (501, 10, 1),
             (502, 10, 4), (800, None, 1), (801, None, 4),
             (802, 2, None), (803, 10, None))
    for seed, t, q in cases:
        first, second = generate_puzzle(seed, t, q), generate_puzzle(seed, t, q)
        _check(_signature(first) == _signature(second), 'nondeterministic puzzle')
        if t is not None:
            _check(len(first['train']) == t, 'train count ignored')
        if q is not None:
            _check(len(first['test']) == q, 'test count ignored')
        validate_generated_puzzle(first)
    illegal = ({'num_train': 1}, {'num_train': 11}, {'num_test': 0},
               {'num_test': 5}, {'num_train': 2, 'num_test': 2},
               {'num_train': 2, 'num_test': 3}, {'num_train': 3, 'num_test': 4})
    for kwargs in illegal:
        try:
            generate_puzzle(99, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _check(len({_signature(p) for p in puzzles}) == 42, 'seed diversity collapsed')
    trains, tests, shapes, counts, overlaps, backgrounds = (set() for _ in range(6))
    orientations = set()
    interior = adjacent = small = mixed = varied_bg = 0
    stable = varied_counts = swapped = thin = combined = 0
    for puzzle in puzzles:
        trains.add(len(puzzle['train']))
        tests.add(len(puzzle['test']))
        records = {split: [_inspect(p['input'], p['output']) for p in puzzle[split]]
                   for split in ('train', 'test')}
        overlaps.add(next(r['overlap'] for r in records['train'] if r['overlap'] is not None))
        all_records = records['train'] + records['test']
        varied_bg += len({r['background'] for r in all_records}) > 1
        varied_counts += len({len(r['markers']) for r in all_records}) > 1
        stable += all(len(r['markers']) == 2 for r in all_records) and len({
            (r['background'], tuple(c for c, _ in sorted(r['markers'],
                                                    key=lambda item: item[1][0])))
            for r in all_records}) == 1
        first_two = [sorted(r['markers'], key=lambda item: item[1][0])[:2]
                     for r in records['train'][:2]]
        swapped += [c for c, _ in first_two[0]] == [c for c, _ in first_two[1]][::-1]
        combined += (len({r['background'] for r in all_records}) > 1
                     and any(len(r['markers']) >= 3 for r in all_records)
                     and len({r['shape'] for r in all_records}) > 1)
        signs = set()
        for rs in records.values():
            for r in rs:
                h, w = r['shape']
                shapes.add((h, w))
                backgrounds.add(r['background'])
                n = len(r['markers'])
                counts.add(n)
                rows = sorted(pos[0] for _, pos in r['markers'])
                cols = sorted(pos[1] for _, pos in r['markers'])
                interior += all(0 < row < h - 1 and 0 < col < w - 1
                                for _, (row, col) in r['markers'])
                adjacent += any(b - a == 1 for a, b in zip(rows, rows[1:])) or any(
                    b - a == 1 for a, b in zip(cols, cols[1:]))
                small += min(h, w) <= 4
                thin += max(h, w) >= 2 * min(h, w)
                if n >= 2:
                    top = sorted(r['markers'], key=lambda x: x[1][0])[:2]
                    sign = 1 if top[0][1][1] < top[1][1][1] else -1
                    signs.add(sign)
                    orientations.add(sign)
        mixed += len(signs) == 2
        validate_generated_puzzle(puzzle)
    _check(len(trains) >= 4 and len(tests) >= 3, 'pair-count diversity collapsed')
    _check(len(shapes) >= 40 and counts == {1, 2, 3, 4, 5}, 'shape/count diversity collapsed')
    _check(len(overlaps) >= 8 and len(backgrounds) >= 8, 'color diversity collapsed')
    _check(orientations == {-1, 1} and mixed >= 5, 'orientation diversity collapsed')
    _check(interior >= 20 and adjacent >= 20 and small >= 8,
           'placement or small-grid diversity collapsed')
    _check(varied_bg >= 10, 'within-puzzle background diversity collapsed')
    _check(stable >= 4 and varied_counts >= 10 and swapped >= 10,
           'palette and marker-count regimes collapsed')
    _check(thin >= 15 and combined >= 10, 'compatible variations stopped combining')
