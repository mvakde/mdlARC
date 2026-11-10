"""Calibrate colored comb branches from the ordinal middle's object count.

Only standard-library modules and NumPy are used. The solver reads one input;
no sampling descriptor, demonstration, random state, or label is consulted.
"""
from fractions import Fraction
import json

import numpy as np


_DIAMOND = ((0, 1), (1, 0), (1, 2), (2, 1))
_SQUARE = ((0, 0), (0, 1), (1, 0), (1, 1))
_X = ((0, 0), (0, 2), (1, 1), (2, 0), (2, 2))
_CROSS = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))
_GLYPHS = (_DIAMOND, _SQUARE, _X, _CROSS, ((0, 0),),
           ((0, 0), (0, 1)), ((0, 0), (1, 1)),
           ((0, 0), (1, 0), (1, 1)), ((0, 0), (0, 1), (0, 2)))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("Expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, np.ascontiguousarray(a).tobytes()


def _puzzle_key(p):
    return tuple(tuple((_key(q['input']), _key(q['output'])) for q in p[s])
                 for s in ('train', 'test'))


def _components(mask, diagonal=True):
    remaining = set(map(tuple, np.argwhere(mask)))
    steps = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
             if (dr or dc) and (diagonal or abs(dr) + abs(dc) == 1)]
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr, dc in steps:
                point = (r + dr, c + dc)
                if point in remaining:
                    remaining.remove(point)
                    stack.append(point)
        result.append(tuple(sorted(cells)))
    return result


def _halo(mask):
    h, w = mask.shape
    padded = np.pad(mask, 1)
    answer = np.zeros(mask.shape, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            answer |= padded[dr:dr + h, dc:dc + w]
    return answer


def _ray(anchor, direction, length):
    r, c = anchor
    dr, dc = direction
    return tuple((r + j * dr, c + j * dc) for j in range(length))


def _render(grid, meta, targets=None):
    """Edit only branch rays, preserving every other cell."""
    out = grid.copy()
    for branch in meta['branches']:
        for r, c in branch['cells']:
            out[r, c] = meta['background']
    spans = meta['targets'] if targets is None else targets
    for branch, span in zip(meta['branches'], spans):
        for r, c in _ray(branch['anchor'], meta['direction'], int(span)):
            if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                raise ValueError("Target does not fit")
            out[r, c] = branch['color']
    return out


def _complete_candidate(grid, bg, marker, spine, direction):
    h, w = grid.shape
    branches = []
    comb = np.zeros(grid.shape, dtype=bool)
    for r, c in spine:
        comb[r, c] = True
    for index, (r, c) in enumerate(spine):
        ar, ac = r + direction[0], c + direction[1]
        if not (0 <= ar < h and 0 <= ac < w) or grid[ar, ac] == bg:
            continue
        color = int(grid[ar, ac])
        if color == marker or index in (0, len(spine) - 1):
            raise ValueError("Branches must use payload colors at interior anchors")
        cells = []
        rr, cc = ar, ac
        while 0 <= rr < h and 0 <= cc < w and grid[rr, cc] == color:
            cells.append((rr, cc))
            rr += direction[0]
            cc += direction[1]
        for rr, cc in cells:
            comb[rr, cc] = True
        branches.append(dict(anchor=(ar, ac), color=color, length=len(cells),
                             cells=tuple(cells), position=index))
    n = len(branches)
    positions = tuple(b['position'] for b in branches)
    if n not in (3, 5, 7, 9) or any(b - a < 2 for a, b in zip(positions, positions[1:])):
        raise ValueError("Need an odd, separated branch array")
    # All foreground touching the comb, of any color, must belong to it.
    loose = (grid != bg) & ~comb
    if np.any(loose & _halo(comb)):
        raise ValueError("Unexpected foreground attached to the comb")
    tokens, counts = [], {}
    for cells in _components(loose):
        colors = {int(grid[r, c]) for r, c in cells}
        if len(colors) != 1:
            raise ValueError("Loose objects must be monochrome")
        color = next(iter(colors))
        if color == marker:
            raise ValueError("Backbone color cannot occur in loose objects")
        box = tuple(max(p[k] for p in cells) - min(p[k] for p in cells) + 1 for k in (0, 1))
        if max(box) > 5:
            raise ValueError("Loose object bounding box exceeds five")
        counts[color] = counts.get(color, 0) + 1
        tokens.append(dict(color=color, cells=cells))
    if any(b['color'] not in counts for b in branches):
        raise ValueError("Every key color needs loose objects")
    middle = branches[n // 2]
    denominator = counts[middle['color']]
    if middle['length'] % denominator:
        raise ValueError("The reference must define an integral unit")
    unit = middle['length'] // denominator
    if unit < 1:
        raise ValueError("Unit must be positive")
    targets = tuple(unit * counts[b['color']] for b in branches)
    union = comb.copy()
    for branch, span in zip(branches, targets):
        for r, c in _ray(branch['anchor'], direction, span):
            if not (0 <= r < h and 0 <= c < w):
                raise ValueError("Complete calibrated branch must fit")
            union[r, c] = True
    if np.any(loose & _halo(union)):
        raise ValueError("Loose objects must be separated from target rays")
    return dict(background=bg, marker=marker, spine=spine, direction=direction,
                branches=branches, positions=positions, comb=comb, loose=loose,
                tokens=tokens, counts=counts, unit=unit, targets=targets,
                middle_count=denominator, reference=middle['length'], union=union)


def _analyze(input_grid):
    grid = _grid(input_grid)
    frequencies = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(frequencies))
    if frequencies[bg] <= grid.size / 2:
        raise ValueError("Input must have a majority background")
    candidates = []
    for marker in np.unique(grid):
        marker = int(marker)
        if marker == bg:
            continue
        cells = tuple((int(r), int(c)) for r, c in np.argwhere(grid == marker))
        if len(cells) < 7:
            continue
        rows, cols = {r for r, c in cells}, {c for r, c in cells}
        if len(cols) == 1 and max(rows) - min(rows) + 1 == len(cells):
            spine = tuple(sorted(cells))
            directions = ((0, 1), (0, -1))
        elif len(rows) == 1 and max(cols) - min(cols) + 1 == len(cells):
            spine = tuple(sorted(cells, key=lambda p: p[1]))
            directions = ((1, 0), (-1, 0))
        else:
            continue
        for direction in directions:
            try:
                candidates.append(_complete_candidate(grid, bg, marker, spine, direction))
            except ValueError:
                continue
    if len(candidates) != 1:
        raise ValueError("Expected one unique calibrated comb")
    return candidates[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    return _render(grid, _analyze(grid))


def _normalize(cells):
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _shape(rng):
    if rng.random() < .6:
        cells = _GLYPHS[int(rng.integers(len(_GLYPHS)))]
    else:
        occupied = {(1, 1)}
        area = int(rng.integers(1, 10))
        while len(occupied) < area:
            frontier = sorted({(r + dr, c + dc) for r, c in occupied
                               for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                               if 0 <= r + dr < 3 and 0 <= c + dc < 3} - occupied)
            occupied.add(frontier[int(rng.integers(len(frontier)))])
        cells = tuple(occupied)
    for _ in range(int(rng.integers(4))):
        cells = tuple((-c, r) for r, c in cells)
    if rng.integers(2):
        cells = tuple((r, -c) for r, c in cells)
    return _normalize(cells)


def _sample_lengths(rng, targets, witness=False):
    n, middle = len(targets), len(targets) // 2
    lengths = list(targets)
    others = [int(i) for i in rng.permutation(n) if i != middle]
    capacity = min(20, max(targets) + int(rng.integers(2, 6)))
    if witness:
        for i in others:
            lengths[i] = int(rng.integers(1, targets[i]))
        i = others[0]
        lengths[i] = int(rng.integers(targets[i] + 1, capacity + 1))
        return lengths
    mode = str(rng.choice(('mixed', 'extend', 'trim', 'one')))
    changes = others[:1] if mode == 'one' else others
    for i in changes:
        short, long = targets[i] > 1, targets[i] < capacity
        if rng.random() < .23 and mode != 'one':
            continue
        do_short = short and (not long or mode == 'extend' or
                             (mode in ('mixed', 'one') and rng.random() < .5))
        if do_short:
            lengths[i] = int(rng.integers(1, targets[i]))
        elif long:
            lengths[i] = int(rng.integers(targets[i] + 1, capacity + 1))
        elif short:
            lengths[i] = int(rng.integers(1, targets[i]))
    if lengths == list(targets):
        i = others[0]
        lengths[i] = targets[i] - 1 if targets[i] > 1 else targets[i] + 1
    return lengths


def _place_object(grid, blocked, color, shape, rng):
    sh, sw = 1 + max(r for r, c in shape), 1 + max(c for r, c in shape)
    nh, nw = grid.shape[0] - sh + 1, grid.shape[1] - sw + 1
    allowed = np.ones((nh, nw), dtype=bool)
    for r, c in shape:
        allowed &= ~blocked[r:r + nh, c:c + nw]
    positions = np.argwhere(allowed)
    if not len(positions):
        return False
    r0, c0 = map(int, positions[int(rng.integers(len(positions)))])
    for dr, dc in shape:
        r, c = r0 + dr, c0 + dc
        grid[r, c] = color
        blocked[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True
    return True


def _make_pair(rng, spec):
    colors, counts, unit, n = spec['colors'], spec['counts'], spec['unit'], spec['n']
    targets = [unit * counts[c] for c in colors]
    lengths = _sample_lengths(rng, targets, spec['witness'])
    objects = []
    middle_color = colors[n // 2]
    for color, count in counts.items():
        for j in range(count):
            if spec['witness'] and color == middle_color:
                shape = (_DIAMOND, _SQUARE, ((0, 0),))[j]
            elif spec['witness'] and color in colors:
                # Their orthogonal counts stay equal to their object counts,
                # so the central diamond cannot accidentally cancel out.
                shape = (_SQUARE, ((0, 0),), ((0, 0), (1, 0), (1, 1)))[j % 3]
            else:
                shape = _shape(rng)
            objects.append((color, shape))
    rng.shuffle(objects)
    objects.sort(key=lambda p: -(1 + max(r for r, c in p[1])) * (1 + max(c for r, c in p[1])))
    if spec['witness']:
        gaps = [2] * 4
        gaps[0 if rng.integers(2) else -1] += int(rng.choice((5, 7)))
    elif rng.random() < .35:
        gap = int(rng.integers(2, min(5, 23 // (n - 1)) + 1))
        gaps = [gap] * (n - 1)
    else:
        gaps = [2] * (n - 1)
        for _ in range(int(rng.integers(1, min(8, 24 - sum(gaps)) + 1))):
            gaps[int(rng.integers(n - 1))] += 1
    lead, trail = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    spine_len = max(7, lead + sum(gaps) + trail + 1)
    if spine_len > 30:
        trail = 1
        lead = min(lead, 28 - sum(gaps))
        spine_len = lead + sum(gaps) + trail + 1
    relative = np.cumsum([lead] + gaps).astype(int).tolist()
    span = max(lengths + targets)
    max_cloud = 28 - span
    columns = int(rng.integers(2, min(4, (max_cloud + 1) // 4) + 1))
    cloud = 4 * columns - 1
    cloud += int(rng.integers(min(3, max_cloud - cloud) + 1))
    spine_col = cloud + 1
    min_width = spine_col + span + 1
    width = int(rng.integers(min_width, 31))
    ink = sum(len(s) for c, s in objects) + spine_len + max(sum(lengths), sum(targets))
    min_height = max(spine_len, 4 * ((len(objects) + columns - 1) // columns) - 1,
                     (2 * ink) // width + 1)
    if min_height > 30:
        raise AssertionError("Construction budget exceeded")
    height = int(rng.integers(min_height, 31))
    top = int(rng.integers(height - spine_len + 1))
    rows = [top + p for p in relative]
    inp = np.full((height, width), spec['background'], dtype=np.int8)
    out = inp.copy()
    union = np.zeros(inp.shape, dtype=bool)
    for a in (inp, out):
        a[top:top + spine_len, spine_col] = spec['marker']
    union[top:top + spine_len, spine_col] = True
    for row, color, initial, target in zip(rows, colors, lengths, targets):
        inp[row, spine_col + 1:spine_col + 1 + initial] = color
        out[row, spine_col + 1:spine_col + 1 + target] = color
        union[row, spine_col + 1:spine_col + 1 + max(initial, target)] = True
    fixed_input, fixed_output = inp.copy(), out.copy()
    for attempt in range(7):
        inp = fixed_input.copy()
        blocked = _halo(union)
        if attempt < 6:
            if not all(_place_object(inp, blocked, c, s, rng) for c, s in objects):
                continue
        else:
            # Guaranteed fallback: slots have a one-cell Chebyshev halo.
            # It never changes counts or replaces a difficult glyph.
            slots = [(r, c) for r in range(0, height - 2, 4)
                     for c in range(0, 4 * columns, 4)]
            assert len(slots) >= len(objects)
            rng.shuffle(slots)
            for (color, shape), (r0, c0) in zip(objects, slots):
                sh, sw = 1 + max(r for r, c in shape), 1 + max(c for r, c in shape)
                r0 += int(rng.integers(4 - sh))
                c0 += int(rng.integers(4 - sw))
                for dr, dc in shape:
                    inp[r0 + dr, c0 + dc] = color
        out = fixed_output.copy()
        loose = (inp != spec['background']) & ~union
        out[loose] = inp[loose]
        if spec['reflect']:
            inp, out = np.flipud(inp), np.flipud(out)
        inp = np.rot90(inp, spec['rotation']).copy()
        out = np.rot90(out, spec['rotation']).copy()
        assert np.array_equal(solve(inp), out), "Construction and visible calibration disagree"
        return {'input': inp, 'output': out}
    raise AssertionError("Unreachable placement fallback")


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not low <= value <= high):
            raise ValueError(name + " must be an integer in its legal range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def _sample_spec(rng, split, index, shift, base, orientation, old_color, first_unit):
    witness = split == 'train' and index == 0
    role_witness = split == 'train' and index == 1
    protected = old_color if witness or role_witness else None
    structural = [c for c in range(10) if c != protected]
    bg, marker = map(int, rng.choice(structural, 2, replace=False))
    pool = [c for c in range(10) if c not in (bg, marker, protected)]
    rng.shuffle(pool)
    if witness:
        a, b = pool[:2]
        others = [a, b, a, b]
        rng.shuffle(others)
        colors = others[:2] + [old_color] + others[2:]
        counts = {a: 1, b: 2, old_color: 3}
        unit, n = first_unit, 5
        rest = pool[2:]
    elif role_witness:
        n, unit = 3, 5 - first_unit
        a, b = pool[:2]
        colors = [a, b, a]
        counts = {a: 3, b: 1 if first_unit == 2 else int(rng.integers(1, 3))}
        counts[old_color] = int(rng.integers(1, 4))
        rest = pool[2:]
    else:
        train = split == 'train'
        informative_test = not train and index == 0
        choices = (3, 5) if train and shift == 1 else (7, 9) if not train and shift == 1 else (3, 5, 7, 9)
        n = int(rng.choice(choices))
        k = int(rng.integers(2 if informative_test else 1, min(4, n) + 1))
        keys = pool[:k]
        colors = keys + [int(rng.choice(keys)) for _ in range(n - k)]
        rng.shuffle(colors)
        cap = 3 if train and shift == 0 else 5 if not train and shift in (0, 2) else 4
        counts = {c: int(rng.integers(1, cap + 1)) for c in keys}
        forced = int(rng.choice(keys)) if split == 'test' and shift == 0 else None
        if forced is not None:
            counts[forced] = 5
        while sum(counts.values()) > 11:
            eligible = [c for c in keys if counts[c] > 1 and c != forced]
            counts[int(rng.choice(eligible))] -= 1
        if informative_test and len(set(counts.values())) == 1:
            color = next(c for c in reversed(keys) if c != forced)
            counts[color] = 1 if counts[color] > 1 else 2
        unit = 4 if split == 'test' and shift == 2 else int(rng.integers(2 if informative_test else 1, 4))
        rest = pool[k:]
    nd = int(rng.integers(0, min(2, len(rest)) + 1))
    budget = min(3, 14 - sum(counts.values()))
    for color in rest[:min(nd, budget)]:
        amount = int(rng.integers(1, budget - min(nd, budget) + 2)) if nd == 1 else 1
        counts[color] = amount
        budget -= amount
    if shift == 3 or orientation == 0:
        rotation = base if split == 'train' else (base + 1) % 4 if shift == 3 else base
    elif orientation == 1:
        rotation = (base + index) % 4 if split == 'train' else int(rng.integers(4))
    else:
        rotation = base if split == 'train' else (base + 1) % 4
    return dict(n=n, colors=colors, counts=counts, unit=unit, witness=witness,
                background=bg, marker=marker, rotation=rotation,
                reflect=bool(rng.integers(2)))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift, base = int(rng.integers(4)), int(rng.integers(4))
    orientation = int(rng.choice(3, p=(.3, .5, .2)))
    old_color, first_unit = int(rng.integers(10)), int(rng.choice((2, 3)))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            spec = _sample_spec(rng, split, index, shift, base, orientation, old_color, first_unit)
            for _ in range(8):
                pair = _make_pair(rng, spec)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise AssertionError("Failed to obtain a distinct input")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(grid):
    m = _analyze(grid)
    colors = tuple(b['color'] for b in m['branches'])
    lengths = tuple(b['length'] for b in m['branches'])
    n, middle = len(colors), len(colors) // 2
    ratios = tuple(Fraction(l, m['counts'][c]) for l, c in zip(lengths, colors))
    dist = [abs(2 * p - m['positions'][0] - m['positions'][-1]) for p in m['positions']]
    nearest = [i for i, d in enumerate(dist) if d == min(dist)]
    counts4 = {c: len(_components((np.asarray(grid) == c) & m['loose'], False)) for c in m['counts']}
    shapes = {c: [_normalize(t['cells']) for t in m['tokens'] if t['color'] == c] for c in m['counts']}
    m.update(colors=colors, lengths=lengths, ratios=ratios, n=n,
             gaps=tuple(b - a for a, b in zip(m['positions'], m['positions'][1:])),
             counts4=counts4, shapes=shapes,
             nonmedian=sorted(lengths)[middle] != m['reference'],
             nonmedian_ratio=sorted(ratios)[middle] != m['unit'],
             geometric_witness=len(nearest) == 1 and nearest[0] != middle
                               and ratios[nearest[0]] != m['unit'],
             extends=any(a < b for a, b in zip(lengths, m['targets'])),
             trims=any(a > b for a, b in zip(lengths, m['targets'])),
             distractors=tuple(sorted(set(m['counts']) - set(colors))))
    return m


def _training_witness(m):
    c = m['colors'][m['n'] // 2]
    shapes = m['shapes'][c]
    return (m['n'] == 5 and m['middle_count'] == 3 and m['unit'] > 1
            and _DIAMOND in shapes and _SQUARE in shapes and ((0, 0),) in shapes
            and m['nonmedian'] and m['nonmedian_ratio'] and m['geometric_witness']
            and m['extends'] and m['trims'] and len(set(m['colors'])) < m['n']
            and len({m['counts'][x] for x in m['colors']}) > 1)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train, test = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            m = _features(inp)
            assert np.array_equal(out, _render(inp, m)), "Incorrect calibrated tails"
            assert not np.array_equal(inp, out)
            middle = m['branches'][m['n'] // 2]
            assert m['targets'][m['n'] // 2] == middle['length']
            for r, c in m['spine']:
                assert inp[r, c] == out[r, c] == m['marker']
            assert np.array_equal(inp[m['loose']], out[m['loose']])
            # Each common prefix and anchor is kept; only free tails change.
            for branch, target in zip(m['branches'], m['targets']):
                for r, c in branch['cells'][:target]:
                    assert inp[r, c] == out[r, c] == branch['color']
                for r, c in branch['cells'][target:]:
                    assert out[r, c] == m['background']
                assert out[branch['anchor']] == branch['color']
            assert not np.any((inp != out) & ~m['union'])
            if split == 'train':
                train.append(m)
            else:
                test.append(m)
    assert any(_training_witness(m) for m in train), "Missing joint counting/reference witness"
    assert len({m['unit'] for m in train}) >= 2
    assert len({m['reference'] for m in train}) >= 2
    assert len({m['n'] for m in train}) >= 2
    assert any(set(a['colors']) & set(b['distractors']) for a in train for b in train)
    assert any(m['targets'] != (m['reference'],) * m['n'] for m in train), "A omission invisible"
    assert any(m['targets'] != tuple(m['counts'][c] for c in m['colors']) for m in train), "B omission invisible"
    assert test[0]['unit'] > 1 and len(set(test[0]['targets'])) > 1, "First test must require both parents"


def _counterfactuals(pair):
    inp = np.asarray(pair['input']).copy()
    m = _features(inp)
    middle = m['branches'][m['n'] // 2]
    # Remove N(middle) cells to reduce cells-per-object by exactly one.
    shortened = inp.copy()
    for r, c in middle['cells'][-m['middle_count']:]:
        shortened[r, c] = m['background']
    changed = _analyze(shortened)
    assert changed['unit'] == m['unit'] - 1
    out = solve(shortened)
    assert not np.array_equal(out, pair['output'])
    assert any(changed['targets'][i] != m['targets'][i] for i in range(m['n']) if i != m['n'] // 2)
    # Remove one token of a noncentral repeated color with two loose objects.
    color = next(c for c in m['colors'] if c != middle['color'] and m['counts'][c] > 1)
    token = next(t for t in m['tokens'] if t['color'] == color)
    removed = inp.copy()
    for r, c in token['cells']:
        removed[r, c] = m['background']
    a = _analyze(removed)
    assert a['unit'] == m['unit']
    for i, branch in enumerate(m['branches']):
        assert a['targets'][i] == m['targets'][i] - (m['unit'] if branch['color'] == color else 0)
    assert not np.array_equal(solve(removed), pair['output'])
    # Four-neighbor counting changes at least one target, including via the
    # central denominator; compare rational quantities before any rounding.
    scale4 = Fraction(m['reference'], m['counts4'][middle['color']])
    assert tuple(scale4 * m['counts4'][c] for c in m['colors']) != m['targets']


def _check_beyond_sampler():
    """Hand-built members cover larger quantities, glyphs, and units.

    Labels come from the explicitly drawn algebraic construction. This is a
    family-membership regression, not an independent solver or blind prediction.
    """
    diamond5 = tuple((r, c) for r in range(5) for c in range(5)
                     if abs(r - 2) + abs(c - 2) == 2)
    for unit, quantity in ((2, 6), (6, 2)):
        inp = np.zeros((24, 30), dtype=np.int8)
        out = inp.copy()
        inp[1:14, 0] = out[1:14, 0] = 9
        for row, color, initial, count in ((3, 2, 7, quantity), (8, 5, 3 * unit, 3), (11, 8, 10, 2)):
            inp[row, 1:1 + initial] = color
            out[row, 1:1 + unit * count] = color
        tokens = [(2, diamond5, (17, 0))]
        loose_colors = [2] * (quantity - 1) + [5] * 3 + [8] * 2
        slots = [(r, c) for r in (17, 21) for c in (8, 12, 16, 20, 24)]
        tokens += [(color, ((0, 0),), slot) for color, slot in zip(loose_colors, slots)]
        for color, shape, (r0, c0) in tokens:
            for dr, dc in shape:
                inp[r0 + dr, c0 + dc] = out[r0 + dr, c0 + dc] = color
        assert np.array_equal(solve(inp), out)
        assert _analyze(inp)['unit'] == unit


def self_test():
    """Bounded semantic checks and measured diversity across 42 fixed seeds."""
    rng = np.random.default_rng(20261008)
    _check_beyond_sampler()
    signatures, shapes, structures, geometries, glyphs = set(), set(), set(), set(), set()
    backgrounds, markers, directions, units, ns, patterns = set(), set(), set(), set(), set(), set()
    quantities, repetitions, spacing = set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        assert _puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed))
        signatures.add(_puzzle_key(puzzle))
        witness = next(p for p in puzzle['train'] if _training_witness(_features(p['input'])))
        _counterfactuals(witness)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                m = _features(inp)
                assert np.array_equal(solve(inp.copy()), out)
                shapes.add(inp.shape)
                mapping = {}
                norm = tuple(mapping.setdefault(int(c), len(mapping)) for c in inp.ravel())
                structures.add((inp.shape, norm))
                geometries.add((m['n'], m['gaps'], m['lengths'], m['targets']))
                glyphs.update(s for v in m['shapes'].values() for s in v)
                backgrounds.add(m['background'])
                markers.add(m['marker'])
                directions.add(m['direction'])
                units.add(m['unit'])
                ns.add(m['n'])
                quantities.update(m['counts'].values())
                repetitions.add(len(set(m['colors'])) < m['n'])
                spacing.add(len(set(m['gaps'])) > 1)
                patterns.add('mixed' if m['extends'] and m['trims'] else 'extend' if m['extends'] else 'trim')
        # Whole-scene palette and D4 transformations commute with solve.
        pair = puzzle['test'][0]
        palette = rng.permutation(10).astype(np.int8)
        assert np.array_equal(solve(palette[pair['input']]), palette[pair['output']])
        for turns in range(4):
            for mirror in (False, True):
                inp, out = np.rot90(pair['input'], turns), np.rot90(pair['output'], turns)
                if mirror:
                    inp, out = np.fliplr(inp), np.fliplr(out)
                assert np.array_equal(solve(inp), out)
    assert len(signatures) == 42 and len(shapes) >= 35 and len(structures) >= 150
    assert len(geometries) >= 150 and len(glyphs) >= 30
    assert backgrounds == markers == set(range(10))
    assert len(directions) == 4 and units == {1, 2, 3, 4} and ns == {3, 5, 7, 9}
    assert quantities >= {1, 2, 3, 4, 5}
    assert patterns == {'mixed', 'extend', 'trim'} and repetitions == spacing == {False, True}
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(314159, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted illegal counts")
    broken = {s: [{k: a.copy() for k, a in p.items()} for p in puzzle[s]] for s in puzzle}
    p = broken['test'][0]
    m = _analyze(p['input'])
    p['output'][m['branches'][0]['anchor']] = m['background']
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("Validator accepted a damaged anchor")
    return {'seeds': 42, 'pairs': 210, 'distinct_puzzles': len(signatures),
            'grid_shapes': len(shapes), 'normalized_structures': len(structures),
            'comb_geometries': len(geometries), 'token_shapes': len(glyphs),
            'branch_counts': sorted(ns), 'units': sorted(units),
            'correction_patterns': sorted(patterns)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
