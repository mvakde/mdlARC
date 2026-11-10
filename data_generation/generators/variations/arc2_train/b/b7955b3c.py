"""Generate layered rectangles damaged by five-cell plus stamps."""

import hashlib
import numpy as np

SOURCE_HASHES = {
    'e1e7f438e8742adc2a120759a4f600135b07a6fc6a94cec7d8b0123142525418',
    '39261ee8b906cb65145590242a698121db7258a88fccfaf7cc7388525380c8fd',
    'dabe59fd70315527e61f706a18424f9811f2ebca6eba55d5c332549c671f8dfd',
    'a7fe1dbf8ba13b6eec6ef6c4e44fc956ad35d70dd73fc6ce2af3e64493ef47a5',
    'e2e56a299ed733bee8184994c029325727caea37d4539eea9696bda1b4a400a0',
}


def _check(ok, message):
    if not ok:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(seed, num_train, num_test):
    if num_train is not None:
        if not _integer(num_train) or not 2 <= int(num_train) <= 10:
            raise ValueError('num_train must be from 2 to 10')
        num_train = int(num_train)
    if num_test is not None:
        if not _integer(num_test) or not 1 <= int(num_test) <= 4:
            raise ValueError('num_test must be from 1 to 4')
        num_test = int(num_test)
    if num_train is None and num_test is None:
        num_test = 1 + seed % 4
        num_train = num_test + 1 + (seed // 4) % (10 - num_test)
    elif num_train is None:
        num_train = num_test + 1 + seed % (10 - num_test)
    elif num_test is None:
        num_test = 1 + seed % min(4, num_train - 1)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return num_train, num_test


def _cross(r, c):
    return ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _centers(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), bool)
    centers = []
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        component = set()
        stack = [(row, col)]
        seen[row, col] = True
        while stack:
            r, c = stack.pop()
            component.add((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        possible = [p for p in component if set(_cross(*p)) == component]
        _check(len(component) == 5 and len(possible) == 1, 'marker is not separate plus stamps')
        centers.append(possible[0])
    return centers


def _solve_input(grid, marker):
    """Infer background, exposed bounds, and depth solely from the damaged input."""
    h, w = grid.shape
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    border = border[border != marker]
    _check(len(border) > 0, 'no background evidence')
    background = int(np.argmax(np.bincount(border.astype(int), minlength=10)))
    colors = [int(x) for x in np.unique(grid) if int(x) not in (background, marker)]
    _check(len(colors) >= 2, 'too few visible rectangles')
    bounds, masks = [], []
    for color in colors:
        points = np.argwhere(grid == color)
        r0, c0 = points.min(axis=0)
        r1, c1 = points.max(axis=0) + 1
        bounds.append((int(r0), int(r1), int(c0), int(c1)))
        mask = np.zeros((h, w), bool)
        mask[r0:r1, c0:c1] = True
        masks.append(mask)
    n = len(colors)
    above = np.zeros((n, n), bool)
    for i, color in enumerate(colors):
        visible = grid == color
        for j, cover in enumerate(masks):
            if i != j and np.any(visible & cover):
                above[i, j] = True
    reach = above.copy()
    for k in range(n):
        reach |= reach[:, k, None] & reach[k, None, :]
    _check(not np.any(np.diag(reach)), 'cyclic overlap evidence')
    result = np.full((h, w), background, np.int8)
    covered = np.zeros((h, w), bool)
    winners = np.zeros((h, w), np.int8)
    for i, mask in enumerate(masks):
        covered |= mask
        top = mask.copy()
        for j, other in enumerate(masks):
            if i != j and not reach[i, j]:
                top &= ~other
        result[top] = colors[i]
        winners += top.astype(np.int8)
    _check(np.all(winners[covered] == 1), 'ambiguous overlap order')
    known = grid != marker
    _check(np.array_equal(result[known], grid[known]), 'visible cells contradict rectangles')
    return result, background, colors, bounds


def _overlap(a, b):
    return max(a[0], b[0]) < min(a[1], b[1]) and max(a[2], b[2]) < min(a[3], b[3])


def _rectangle(rng, h, w, previous):
    kind = int(rng.integers(3))
    if kind == 0:
        rh = int(rng.integers(1, min(4, h - 2) + 1))
        rw = int(rng.integers(3, max(4, int(.86 * w)) + 1))
    elif kind == 1:
        rh = int(rng.integers(3, max(4, int(.86 * h)) + 1))
        rw = int(rng.integers(1, min(4, w - 2) + 1))
    else:
        rh = int(rng.integers(2, max(3, int(.7 * h)) + 1))
        rw = int(rng.integers(2, max(3, int(.7 * w)) + 1))
    rh, rw = min(rh, h), min(rw, w)
    relation = int(rng.choice(3, p=[.48, .24, .28]))
    chosen = None
    for _ in range(24):
        if relation == 0 and previous:
            target = previous[int(rng.integers(len(previous)))]
            mid_r, mid_c = (target[0] + target[1]) // 2, (target[2] + target[3]) // 2
            r0 = int(np.clip(mid_r - rh // 2 + rng.integers(-rh // 2 - 1, rh // 2 + 2), 0, h - rh))
            c0 = int(np.clip(mid_c - rw // 2 + rng.integers(-rw // 2 - 1, rw // 2 + 2), 0, w - rw))
        else:
            r0 = int(rng.integers(h - rh + 1))
            c0 = int(rng.integers(w - rw + 1))
        if rng.random() < .14:
            r0 = 0 if rng.random() < .5 else h - rh
        if rng.random() < .14:
            c0 = 0 if rng.random() < .5 else w - rw
        chosen = (r0, r0 + rh, c0, c0 + rw)
        if chosen in previous:
            continue
        if relation != 1 or not previous or all(not _overlap(chosen, p) for p in previous):
            return chosen
    return chosen


def _make_clean(rng, regime, marker, require_overlap):
    if regime == 'source_train':
        h, w, limit = int(rng.integers(8, 23)), int(rng.integers(8, 23)), 6
    else:
        h, w, limit = int(rng.integers(9, 30)), int(rng.integers(9, 31)), 7
    n = int(rng.integers(2, min(limit, max(2, 1 + h * w // 45)) + 1))
    palette = [v for v in range(10) if v != marker]
    chosen = rng.choice(palette, n + 1, replace=False)
    background, colors = int(chosen[0]), [int(v) for v in chosen[1:]]
    rectangles = [_rectangle(rng, h, w, [])]
    for _ in range(n - 1):
        rectangles.append(_rectangle(rng, h, w, rectangles))
    if len(set(rectangles)) != n:
        raise ValueError('duplicate rectangle')
    if require_overlap and not any(_overlap(a, b) for i, a in enumerate(rectangles) for b in rectangles[i + 1:]):
        raise ValueError('training lacks occlusion')
    clean = np.full((h, w), background, np.int8)
    for index in rng.permutation(n):
        r0, r1, c0, c1 = rectangles[int(index)]
        clean[r0:r1, c0:c1] = colors[int(index)]
    inferred, bg, visible, bounds = _solve_input(clean, marker)
    if bg != background or len(visible) != n or not np.array_equal(inferred, clean):
        raise ValueError('rectangle evidence insufficient')
    if require_overlap and not any(_overlap(a, b) for i, a in enumerate(bounds) for b in bounds[i + 1:]):
        raise ValueError('exposed rectangles do not demonstrate overlap')
    return clean


def _stamp(rng, clean, marker, count, require_mixed):
    h, w = clean.shape
    candidates = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            values = {int(clean[rr, cc]) for rr, cc in _cross(r, c)}
            edge = r == 1 or r == h - 2 or c == 1 or c == w - 2
            candidates.append((r, c, len(values) > 1, edge))
    for _ in range(45):
        chosen = []
        for index in range(count):
            possible = [p for p in candidates if all(abs(p[0] - q[0]) + abs(p[1] - q[1]) >= 4 for q in chosen)]
            if not possible:
                break
            want_mixed = True if index == 0 and require_mixed else rng.random() < .58
            preferred = [p for p in possible if p[2] == want_mixed]
            if preferred:
                possible = preferred
            if rng.random() < .24:
                edge = [p for p in possible if p[3]]
                if edge:
                    possible = edge
            chosen.append(possible[int(rng.integers(len(possible)))])
        if len(chosen) != count:
            continue
        damaged = clean.copy()
        for r, c, _, _ in chosen:
            for rr, cc in _cross(r, c):
                damaged[rr, cc] = marker
        try:
            inferred, _, _, _ = _solve_input(damaged, marker)
            if np.array_equal(inferred, clean) and len(_centers(damaged == marker)) == count:
                return damaged
        except AssertionError:
            pass
    raise ValueError('no recoverable stamp placement')


def _generate_pair(rng, regime, marker, require_overlap=False, require_mixed=False):
    for _ in range(260):
        try:
            clean = _make_clean(rng, regime, marker, require_overlap)
            count = int(rng.integers(1, min(6, max(1, clean.size // 36)) + 1))
            damaged = _stamp(rng, clean, marker, count, require_mixed)
            if require_overlap and _pair_stats(damaged, clean, marker)['overlaps'] == 0:
                raise ValueError('stamp hid the overlap evidence')
            return damaged, clean
        except (AssertionError, ValueError):
            pass
    raise RuntimeError('could not generate a recoverable pair')


def _grid_key(grid):
    return grid.shape, tuple(int(x) for x in grid.ravel())


def _source_hash(inp, out):
    text = '|'.join('/'.join(''.join(str(int(v)) for v in row) for row in grid) for grid in (inp, out))
    return hashlib.sha256(text.encode()).hexdigest()


def _pair_stats(inp, out, marker):
    mask = inp == marker
    centers = _centers(mask)
    _check(bool(centers), 'pair has no plus stamp')
    _check(np.array_equal(inp != out, mask), 'only marker cells may change')
    _check(not np.any(out == marker), 'marker survives in clean scene')
    inferred, background, colors, bounds = _solve_input(inp, marker)
    _check(np.array_equal(inferred, out), 'output cannot be reconstructed from input')
    h, w = inp.shape
    tall = [b for b in bounds if b[1] - b[0] > b[3] - b[2]]
    wide = [b for b in bounds if b[3] - b[2] > b[1] - b[0]]
    incomplete = any(not _overlap(a, b) for a in tall for b in wide)
    partial = any(_overlap(a, b) and (
        a[0] < b[0] < a[1] < b[1] or b[0] < a[0] < b[1] < a[1] or
        a[2] < b[2] < a[3] < b[3] or b[2] < a[2] < b[3] < a[3])
        for i, a in enumerate(bounds) for b in bounds[i + 1:])
    return {
        'shape': (h, w), 'background': background, 'rectangle_count': len(colors),
        'cross_count': len(centers), 'bounds': tuple(bounds),
        'overlaps': sum(_overlap(a, b) for i, a in enumerate(bounds) for b in bounds[i + 1:]),
        'incomplete_intersections': incomplete, 'partial_intersection': partial,
        'uniform_crosses': sum(len({int(out[r, c]) for r, c in _cross(*p)}) == 1 for p in centers),
        'mixed_crosses': sum(len({int(out[r, c]) for r, c in _cross(*p)}) > 1 for p in centers),
        'three_color_crosses': sum(len({int(out[r, c]) for r, c in _cross(*p)}) >= 3 for p in centers),
        'edge_crosses': sum(r == 1 or r == h - 2 or c == 1 or c == w - 2 for r, c in centers),
        'edge_content': bool(np.any(out[0] != background) or np.any(out[-1] != background) or
                             np.any(out[:, 0] != background) or np.any(out[:, -1] != background)),
        'thin': sum(min(r1-r0, c1-c0) == 1 for r0, r1, c0, c1 in bounds),
        'broad': sum(min(r1-r0, c1-c0) >= 4 for r0, r1, c0, c1 in bounds),
    }


def generate_puzzle(seed: int, num_train: int | None = None, num_test: int | None = None) -> dict:
    if not _integer(seed) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    seed = int(seed)
    num_train, num_test = _counts(seed, num_train, num_test)
    rng = np.random.default_rng(seed)
    marker = int(rng.integers(10))
    puzzle = {'train': [], 'test': []}
    keys = set()
    for section, count, regime in (('train', num_train, 'source_train'), ('test', num_test, 'target_test')):
        for index in range(count):
            for _ in range(50):
                pair_rng = np.random.default_rng(int(rng.integers(0, np.iinfo(np.int64).max)))
                inp, out = _generate_pair(pair_rng, regime, marker, section == 'train' and index == 0,
                                          section == 'train' and index == 0)
                key = _grid_key(inp)
                if key not in keys and _source_hash(inp, out) not in SOURCE_HASHES:
                    keys.add(key)
                    puzzle[section].append({'input': inp, 'output': out})
                    break
            else:
                raise RuntimeError('could not generate distinct pair')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle: dict) -> None:
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle schema')
    _check(isinstance(puzzle['train'], list) and isinstance(puzzle['test'], list), 'sections must be lists')
    nt, nv = len(puzzle['train']), len(puzzle['test'])
    _check(2 <= nt <= 10 and 1 <= nv <= 4 and nt > nv, 'illegal pair counts')
    inputs, pairs, marker = set(), set(), None
    stats = {'train': [], 'test': []}
    for section in ('train', 'test'):
        for pair in puzzle[section]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair schema')
            for grid in pair.values():
                _check(isinstance(grid, np.ndarray) and grid.ndim == 2, 'grid must be 2D NumPy array')
                h, w = grid.shape
                _check(1 <= h <= 30 and 1 <= w <= 30, 'grid dimensions out of range')
                _check(np.issubdtype(grid.dtype, np.integer) and np.all((0 <= grid) & (grid <= 9)),
                       'grid must contain integer ARC colors')
            inp, out = pair['input'], pair['output']
            _check(inp.shape == out.shape, 'dimensions changed')
            changed = inp != out
            _check(np.any(changed), 'unchanged example')
            changed_colors = set(int(x) for x in inp[changed])
            _check(len(changed_colors) == 1, 'multiple marker colors')
            current = next(iter(changed_colors))
            if marker is None:
                marker = current
            _check(current == marker, 'marker varies within puzzle')
            input_key = _grid_key(inp)
            pair_key = input_key, _grid_key(out)
            _check(input_key not in inputs and pair_key not in pairs, 'duplicate pair')
            inputs.add(input_key)
            pairs.add(pair_key)
            _check(_source_hash(inp, out) not in SOURCE_HASHES, 'source pair copied')
            info = _pair_stats(inp, out, marker)
            _check(2 <= info['rectangle_count'] <= (6 if section == 'train' else 7), 'rectangle count out of range')
            _check((8 <= inp.shape[0] <= 22 and 8 <= inp.shape[1] <= 22) if section == 'train'
                   else (9 <= inp.shape[0] <= 29 and 9 <= inp.shape[1] <= 30), 'shape out of regime')
            stats[section].append(info)
    _check(any(s['mixed_crosses'] for s in stats['train']), 'training lacks cellwise restoration')
    _check(any(s['overlaps'] for s in stats['train']), 'training lacks occlusion evidence')


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k])
               for x, y in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test() -> None:
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
    _check(len({len(p['train']) for p in puzzles}) >= 4, 'train counts lack diversity')
    _check(len({len(p['test']) for p in puzzles}) >= 3, 'test counts lack diversity')
    _check(_equal(generate_puzzle(9137, 6, 3), generate_puzzle(9137, 6, 3)), 'not deterministic')
    _check(not _equal(generate_puzzle(9137, 6, 3), generate_puzzle(9138, 6, 3)), 'seeds not distinct')
    for t, v in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(20000 + 10*t + v, t, v)
        _check((len(p['train']), len(p['test'])) == (t, v), 'explicit count ignored')
    for seed, t, v in ((31001, None, 1), (31004, None, 4), (32002, 2, None), (32010, 10, None)):
        p = generate_puzzle(seed, t, v)
        _check((t is None or len(p['train']) == t) and (v is None or len(p['test']) == v), 'partial count ignored')
    for args in ((1, None), (11, None), (None, 0), (None, 5), (2, 2), (3, 4)):
        try:
            generate_puzzle(7, *args)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid count accepted')
    stats, markers = [], set()
    within_shapes = within_counts = similar_scales = larger_tests = mixed_regimes = 0
    complex_train = simple_test = 0
    for p in puzzles:
        inp, out = p['train'][0]['input'], p['train'][0]['output']
        marker = int(inp[inp != out][0])
        markers.add(marker)
        train_stats = [_pair_stats(pair['input'], pair['output'], marker) for pair in p['train']]
        test_stats = [_pair_stats(pair['input'], pair['output'], marker) for pair in p['test']]
        current = train_stats + test_stats
        stats.extend(current)
        within_shapes += len({s['shape'] for s in current}) > 1
        within_counts += len({s['rectangle_count'] for s in current}) > 1
        mixed_regimes += any(s['overlaps'] == 0 for s in current) and any(s['overlaps'] for s in current)
        similar_scales += any(abs(t['shape'][0]*t['shape'][1] - v['shape'][0]*v['shape'][1]) /
                              max(t['shape'][0]*t['shape'][1], v['shape'][0]*v['shape'][1]) < .2
                              for t in train_stats for v in test_stats)
        larger_tests += any(v['shape'][0]*v['shape'][1] > 1.5*t['shape'][0]*t['shape'][1]
                            for t in train_stats for v in test_stats)
        complex_train += any(s['rectangle_count'] >= 5 for s in train_stats)
        simple_test += any(s['rectangle_count'] <= 3 for s in test_stats)
    _check(len(markers) >= 7 and 8 in markers, 'marker recoloring lacks coverage')
    _check(len({s['shape'] for s in stats}) >= 50, 'shapes lack diversity')
    _check(len({s['rectangle_count'] for s in stats}) >= 5, 'rectangle counts lack diversity')
    _check(len({(s['shape'], s['bounds']) for s in stats}) >= 150, 'geometry lacks diversity')
    for feature in ('uniform_crosses', 'mixed_crosses', 'edge_crosses', 'edge_content', 'thin', 'broad'):
        _check(any(s[feature] for s in stats), feature + ' absent')
    _check(sum(s['incomplete_intersections'] for s in stats) >= 30, 'incomplete crossings too rare')
    _check(sum(s['partial_intersection'] for s in stats) >= 20, 'partial crossings too rare')
    _check(sum(s['three_color_crosses'] for s in stats) >= 5, 'multilayer stamps too rare')
    _check(sum(s['thin'] > 0 and s['broad'] > 0 and s['uniform_crosses'] > 0 for s in stats) >= 10,
           'compatible variation axes rarely combine')
    _check(within_shapes == 42 and within_counts >= 35 and mixed_regimes >= 20,
           'within-puzzle geometry lacks variation')
    _check(similar_scales >= 15 and larger_tests >= 20 and complex_train >= 10 and simple_test >= 10,
           'train/test progression is too rigid')
    _check(any(s['cross_count'] == 1 for s in stats) and any(s['cross_count'] >= 5 for s in stats),
           'stamp counts lack range')
    _check(any(s['overlaps'] == 0 for s in stats) and any(s['overlaps'] >= 3 for s in stats),
           'overlap counts lack range')
