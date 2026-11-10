"""Odd-overlap cross stencils registered by a partial enlarged specimen.

No generation metadata enters solve(). All semantic colors are visible locally.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np

DIRECTIONS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('expected an integer ARC grid')
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _coverage(plan, bg):
    """Simultaneous monochrome centers and number of covering stencils."""
    count = np.zeros(plan.shape, dtype=np.int16)
    centers = []
    for r in range(1, plan.shape[0] - 1):
        for c in range(1, plan.shape[1] - 1):
            v = int(plan[r, c])
            if v != bg and all(plan[r + dr, c + dc] == v for dr, dc in DIRECTIONS):
                centers.append((r, c))
                count[r, c] += 1
                for dr, dc in DIRECTIONS:
                    count[r + dr, c + dc] += 1
    return count, centers


def _print(plan, bg):
    count, centers = _coverage(plan, bg)
    result = np.full_like(plan, bg)
    result[count % 2 == 1] = plan[count % 2 == 1]
    return result, count, centers


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    groups = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in DIRECTIONS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        groups.append(cells)
    return groups


def _frame(inp, bg):
    candidates = []
    for color in set(map(int, inp.flat)) - {bg}:
        points = np.argwhere(inp == color)
        r, c = map(int, points.min(0))
        end_r, end_c = map(int, points.max(0) + 1)
        if end_r - r < 5 or end_c - c < 5:
            continue
        patch = inp[r:end_r, c:end_c]
        rim = np.ones(patch.shape, dtype=bool)
        rim[1:-1, 1:-1] = False
        if np.array_equal(patch == color, rim):
            candidates.append((r, c, end_r, end_c, color))
    if len(candidates) != 1:
        raise ValueError('expected exactly one exclusive rectangular frame')
    return candidates[0]


def _matches(plan, sequence, axis):
    length = len(sequence)
    answer = []
    for r in range(plan.shape[0] - (length - 1 if axis == 0 else 0)):
        for c in range(plan.shape[1] - (length - 1 if axis == 1 else 0)):
            seq = tuple(int(plan[r + k * (axis == 0), c + k * (axis == 1)])
                        for k in range(length))
            if seq == sequence:
                answer.append((r, c))
    return answer


def _overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _recover(input_grid):
    inp = _grid(input_grid)
    hist = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(hist))
    if hist[bg] * 2 <= inp.size:
        raise ValueError('canvas background must occupy more than half the input')
    fr, fc, er, ec, marker = _frame(inp, bg)
    plan = inp[fr + 1:er - 1, fc + 1:ec - 1].copy()
    if marker in plan:
        raise ValueError('frame color is reserved')
    printing, coverage, centers = _print(plan, bg)
    outside = inp != bg
    outside[fr:er, fc:ec] = False
    squares = []
    for cells in _components(outside):
        points = np.array(cells)
        r, c = map(int, points.min(0))
        br, bc = map(int, points.max(0) + 1)
        side = br - r
        color = int(inp[r, c])
        if (side < 2 or bc - c != side or len(cells) != side * side
                or color == marker or not np.all(inp[r:br, c:bc] == color)):
            raise ValueError('outside the frame, only solid specimen squares are allowed')
        squares.append((r, c, side, color))
    if len(squares) < 2:
        raise ValueError('specimen needs at least two squares')
    if len({s[0] for s in squares}) == 1:
        axis = 1
    elif len({s[1] for s in squares}) == 1:
        axis = 0
    else:
        raise ValueError('specimen must form one straight run')
    squares.sort(key=lambda s: s[axis])
    side = squares[0][2]
    differences = np.diff([s[axis] for s in squares])
    stride = int(differences[0])
    if (any(s[2] != side for s in squares) or stride <= side
            or not np.all(differences == stride)):
        raise ValueError('inconsistent square size or positive gutter')
    sequence = tuple(s[3] for s in squares)
    positions = _matches(printing, sequence, axis)
    if len(positions) != 1:
        raise ValueError('specimen must uniquely match the odd-coverage print')
    match = positions[0]
    base = (squares[0][0] - match[0] * stride,
            squares[0][1] - match[1] * stride)
    bottom = base[0] + (plan.shape[0] - 1) * stride + side
    right = base[1] + (plan.shape[1] - 1) * stride + side
    if not (0 <= base[0] < bottom <= inp.shape[0]
            and 0 <= base[1] < right <= inp.shape[1]):
        raise ValueError('complete print must fit; clipping and wrapping are forbidden')
    if _overlap((base[0], base[1], bottom, right), (fr, fc, er, ec)):
        raise ValueError('complete print and framed blueprint must be separate')
    seed_sites = {(match[0] + k * (axis == 0), match[1] + k * (axis == 1))
                  for k in range(len(squares))}
    for (r, c), v in np.ndenumerate(printing):
        y, x = base[0] + r * stride, base[1] + c * stride
        expected = int(v) if (r, c) in seed_sites else bg
        if not np.all(inp[y:y + side, x:x + side] == expected):
            raise ValueError('only the specimen may prefill the print rectangle')
    meta = dict(background=bg, marker=marker, frame=(fr, fc, er, ec),
                plan=plan, printing=printing, coverage=coverage, centers=centers,
                axis=axis, sequence=sequence, match=match, side=side,
                stride=stride, base=base, seed_sites=seed_sites)
    return inp, meta


def _render(inp, plan, bg, base, side, stride):
    out = inp.copy()
    for (r, c), color in np.ndenumerate(plan):
        if color != bg:
            y, x = base[0] + r * stride, base[1] + c * stride
            out[y:y + side, x:x + side] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the frame, stencils, parity, and specimen alignment from pixels."""
    inp, m = _recover(input_grid)
    return _render(inp, m['printing'], m['background'], m['base'], m['side'], m['stride'])


def _asymmetric(a):
    variants = [a[::-1], a[:, ::-1], a[::-1, ::-1]]
    if a.shape[0] == a.shape[1]:
        variants.extend([a.T, np.rot90(a), np.rot90(a, 3), a.T[::-1, ::-1]])
    return all(not np.array_equal(a, other) for other in variants)


def _sequential_print(plan, bg):
    # A deliberately wrong alternative used only for explanatory witness checks.
    remaining = plan.copy()
    marked = np.full_like(plan, bg)
    for r in range(1, plan.shape[0] - 1):
        for c in range(1, plan.shape[1] - 1):
            v = int(remaining[r, c])
            if v != bg and all(remaining[r + dr, c + dc] == v for dr, dc in DIRECTIONS):
                for p in [(r, c)] + [(r + dr, c + dc) for dr, dc in DIRECTIONS]:
                    marked[p] = v
                    remaining[p] = bg
    return marked


def _features(m):
    p, q, cov, bg = m['plan'], m['printing'], m['coverage'], m['background']
    missing = [(r, c) for r, c in zip(*np.nonzero(q != bg)) if (r, c) not in m['seed_sites']]
    return dict(plan_shape=p.shape, side=m['side'], gutter=m['stride'] - m['side'],
                background=bg, marker=m['marker'], axis=m['axis'], length=len(m['sequence']),
                match=m['match'], base=m['base'], frame=m['frame'], centers=len(m['centers']),
                canceled=int(np.count_nonzero((cov > 0) & (cov % 2 == 0))),
                unused=int(np.count_nonzero((p != bg) & (cov == 0))),
                max_coverage=int(cov.max()), odd_high=bool(np.any((cov >= 3) & (cov % 2 == 1))),
                raw_alias=len(_matches(p, m['sequence'], m['axis'])) > 1,
                sequential=not np.array_equal(q, _sequential_print(p, bg)),
                asymmetric=_asymmetric(q), added=len(missing),
                added_colors=len({int(q[r, c]) for r, c in missing}),
                interior=(0 < m['match'][m['axis']] and m['match'][m['axis']] + len(m['sequence']) < p.shape[m['axis']]),
                cross_interior=0 < m['match'][1 - m['axis']] < p.shape[1 - m['axis']] - 1)


@lru_cache(maxsize=None)
def _options():
    choices = []
    for h in range(3, 9):
        for w in range(3, 9):
            for side in range(2, 5):
                for gutter in range(1, 4):
                    box = ((h - 1) * (side + gutter) + side,
                           (w - 1) * (side + gutter) + side)
                    frame = (h + 2, w + 2)
                    for d in (0, 1):
                        need = [max(box[k], frame[k]) for k in (0, 1)]
                        need[d] = box[d] + frame[d] + 1
                        if max(need) <= 30:
                            choices.append((h, w, side, gutter, d))
    return tuple(choices)


def _stamp(plan, r, c, color):
    plan[r, c] = color
    for dr, dc in DIRECTIONS:
        plan[r + dr, c + dc] = color


def _random_plan(rng, shape, bg, palette, overlap):
    h, w = shape
    p = np.full(shape, bg, dtype=np.int8)
    style = int(rng.integers(3))
    if style == 0:
        # Colored regions make dense embedded centers without a fixed template.
        d = int(rng.integers(2))
        cut = int(rng.integers(1, shape[d]))
        for r in range(h):
            for c in range(w):
                if rng.random() < rng.uniform(.6, 1.):
                    p[r, c] = palette[0 if (r, c)[d] < cut else 1]
    elif style == 1:
        for _ in range(int(rng.integers(2, 5))):
            bh, bw = int(rng.integers(2, h + 1)), int(rng.integers(2, w + 1))
            y, x = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
            p[y:y + bh, x:x + bw] = rng.choice(palette)
    # Paint crosses in arbitrary order, then analyze the finished blueprint.
    for _ in range(int(rng.integers(2, min(9, h * w // 3) + 1))):
        r, c = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
        color = int(rng.choice(palette))
        _stamp(p, r, c, color)
        if overlap and rng.random() < .65:
            neighbors = [(r + dr, c + dc) for dr, dc in DIRECTIONS
                         if 0 < r + dr < h - 1 and 0 < c + dc < w - 1]
            if neighbors:
                nr, nc = neighbors[int(rng.integers(len(neighbors)))]
                _stamp(p, nr, nc, color)
    # Extensions, clipped crosses and distractor strokes can create or destroy
    # centers. No planted-center metadata is trusted afterward.
    for _ in range(int(rng.integers(max(2, h * w // 5)))):
        r, c = int(rng.integers(h)), int(rng.integers(w))
        p[r, c] = rng.choice(palette)
    return p


def _segments(q, bg, axis, lengths):
    results = []
    for length in lengths:
        if length > q.shape[axis]:
            continue
        sequences = {}
        for r in range(q.shape[0] - (length - 1 if axis == 0 else 0)):
            for c in range(q.shape[1] - (length - 1 if axis == 1 else 0)):
                seq = tuple(int(q[r + k * (axis == 0), c + k * (axis == 1)]) for k in range(length))
                if bg not in seq:
                    sequences.setdefault(seq, []).append((r, c))
        for seq, sites in sequences.items():
            if len(sites) == 1:
                results.append((sites[0], seq))
    return results


def _place(rng, shape, side, gutter, d):
    frame = (shape[0] + 2, shape[1] + 2)
    box = tuple((n - 1) * (side + gutter) + side for n in shape)
    gap = int(rng.integers(1, min(4, 30 - frame[d] - box[d]) + 1))
    need = [max(frame[k], box[k]) for k in (0, 1)]
    need[d] = frame[d] + box[d] + gap
    if rng.random() < .18:
        canvas = (30, 30)
    elif rng.random() < .25:
        n = int(rng.integers(max(need), 31))
        canvas = (n, n)
    else:
        canvas = tuple(int(rng.integers(n, min(n + 5, 30) + 1)) for n in need)
    anchor, base = [0, 0], [0, 0]
    slack = canvas[d] - need[d]
    margin = int(rng.integers(slack + 1))
    if rng.integers(2):
        anchor[d] = margin
        base[d] = margin + frame[d] + gap
    else:
        base[d] = margin
        anchor[d] = margin + box[d] + gap
    other = 1 - d
    anchor[other] = int(rng.integers(canvas[other] - frame[other] + 1))
    base[other] = int(rng.integers(canvas[other] - box[other] + 1))
    return canvas, tuple(anchor), tuple(base)


def _draw(rng, p, bg, marker, geometry, axis, match, seq):
    h, w, side, gutter, d = geometry
    canvas, anchor, base = _place(rng, (h, w), side, gutter, d)
    inp = np.full(canvas, bg, dtype=np.int8)
    r, c = anchor
    inp[r:r + h + 2, c:c + w + 2] = marker
    inp[r + 1:r + h + 1, c + 1:c + w + 1] = p
    stride = side + gutter
    for k, color in enumerate(seq):
        y = base[0] + (match[0] + k * (axis == 0)) * stride
        x = base[1] + (match[1] + k * (axis == 1)) * stride
        inp[y:y + side, x:x + side] = color
    q, _, _ = _print(p, bg)
    return {'input': inp, 'output': _render(inp, q, bg, base, side, stride)}


def _acceptable(p, bg, axis, match, seq, kind):
    if bg in seq:
        return False
    q, count, centers = _print(p, bg)
    seed = {(match[0] + k * (axis == 0), match[1] + k * (axis == 1)) for k in range(len(seq))}
    added = {(int(r), int(c)) for r, c in np.argwhere(q != bg)} - seed
    if len(added) < 3 or len({int(q[r, c]) for r, c in added}) < 2 or not _asymmetric(q):
        return False
    if kind == 'clean' and count.max() > 1:
        return False
    if kind in ('witness', 'test'):
        if not np.any((count > 0) & (count % 2 == 0)) or not np.any((p != bg) & (count == 0)):
            return False
        if np.array_equal(q, _sequential_print(p, bg)):
            return False
    if kind == 'witness':
        if len(_matches(p, seq, axis)) <= 1:
            return False
        if not np.any((count >= 3) & (count % 2 == 1)):
            return False
    if kind == 'test':
        start = match[axis]
        if not (len(centers) >= 4 and 0 < start and start + len(seq) < p.shape[axis]):
            return False
    return True


def _fallback(bg, palette, axis, kind):
    # Constructive witnesses; geometric rotation is applied to both
    # representations, never inferred secretly at solve time.
    if kind in ('test', 'witness'):
        p = np.full((4, 8), bg, dtype=np.int8)
        p[0:3, 0:5] = palette[0]
        _stamp(p, 1, 6, palette[1])
        p[3, :3] = (palette[0], palette[1], palette[1])
        match, length = (1, 4), 3 if kind == 'test' else 2
    elif kind == 'clean':
        p = np.full((4, 6), bg, dtype=np.int8)
        _stamp(p, 1, 1, palette[0])
        _stamp(p, 1, 4, palette[1])
        p[3, 3] = palette[1]
        # Touching arms of two different-color stencils form one unique AB.
        match, length = (1, 2), 2
    else:
        p = np.full((4, 7), bg, dtype=np.int8)
        _stamp(p, 1, 1, palette[0])
        _stamp(p, 1, 2, palette[0])
        _stamp(p, 1, 5, palette[1])
        p[3, :2] = (palette[0], palette[1])
        match, length = (1, 3), 2
    if axis == 0:
        p = p.T.copy()
        match = match[::-1]
    q, _, _ = _print(p, bg)
    seq = tuple(int(q[match[0] + k * (axis == 0), match[1] + k * (axis == 1)]) for k in range(length))
    return p, match, seq


def _make_pair(rng, kind, axis, avoid_side=None, avoid_bg=None):
    colors = list(map(int, rng.permutation(10)))
    if colors[0] == avoid_bg:
        colors[0], colors[1] = colors[1], colors[0]
    bg, marker = colors[:2]
    palette = colors[2:2 + int(rng.integers(2, 5))]
    choices = [x for x in _options() if x[2] != avoid_side]
    for attempt in range(240):
        geometry = choices[int(rng.integers(len(choices)))]
        p = _random_plan(rng, geometry[:2], bg, palette, kind != 'clean')
        q, _, _ = _print(p, bg)
        segments = _segments(q, bg, axis, range(3, 6) if kind == 'test' else (2,))
        rng.shuffle(segments)
        for match, seq in segments:
            if not _acceptable(p, bg, axis, match, seq, kind):
                continue
            pair = _draw(rng, p, bg, marker, geometry, axis, match, seq)
            try:
                _recover(pair['input'])
            except ValueError:
                continue
            return pair
    p, match, seq = _fallback(bg, palette, axis, kind)
    assert _acceptable(p, bg, axis, match, seq, kind)
    geometries = [x for x in choices if x[:2] == p.shape]
    assert geometries
    rng.shuffle(geometries)
    for geometry in geometries:
        pair = _draw(rng, p, bg, marker, geometry, axis, match, seq)
        try:
            _recover(pair['input'])
        except ValueError:
            continue
        return pair
    raise RuntimeError('constructive layout failed')


def _counts(num_train, num_test):
    for name, v, low, high in [('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)]:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not low <= v <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ns = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    first_axis = int(rng.integers(2))
    tasks = [('train', 'witness', first_axis), ('train', 'clean', 1 - first_axis)]
    tasks.extend(('train', str(rng.choice(['mixed', 'witness', 'clean'], p=[.55, .25, .2])), int(rng.integers(2)))
                 for _ in range(nt - 2))
    tasks.extend(('test', 'test', int(rng.integers(2))) for _ in range(ns))
    puzzle, seen, first = {'train': [], 'test': []}, set(), None
    for index, (split, kind, axis) in enumerate(tasks):
        for _ in range(40):
            pair = _make_pair(rng, kind, axis,
                              first['side'] if index == 1 else None,
                              first['background'] if index == 1 else None)
            key = _key(pair['input'])
            if key not in seen:
                break
        else:
            # Injective palette permutations provide at least 90 distinct
            # alternatives, exceeding the fourteen-pair maximum.
            original = pair
            for a in range(10):
                for b in range(10):
                    if a == b:
                        continue
                    mapping = np.array([a, b] + [v for v in range(10) if v not in (a, b)], dtype=np.int8)
                    candidate = {k: mapping[v] for k, v in original.items()}
                    if _key(candidate['input']) not in seen:
                        if index == 1 and _recover(candidate['input'])[1]['background'] == first['background']:
                            continue
                        pair = candidate
                        break
                else:
                    continue
                break
            else:
                raise RuntimeError('uniqueness bound exceeded')
            key = _key(pair['input'])
        seen.add(key)
        puzzle[split].append(pair)
        if index == 0:
            first = _recover(pair['input'])[1]
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            _, m = _recover(inp)
            bg, p, cov, q = m['background'], m['plan'], m['coverage'], m['printing']
            # Exact patch-level checks cover blank sites, positive sites,
            # specimen preservation, miniature preservation, and gutters.
            expected = inp.copy()
            touched = np.zeros(inp.shape, dtype=bool)
            for (r, c), color in np.ndenumerate(q):
                y, x = m['base'][0] + r * m['stride'], m['base'][1] + c * m['stride']
                assert np.all(out[y:y + m['side'], x:x + m['side']] == color)
                if color != bg:
                    expected[y:y + m['side'], x:x + m['side']] = color
                    touched[y:y + m['side'], x:x + m['side']] = True
            assert np.array_equal(expected, out), 'incorrect parity print or unrelated change'
            assert np.array_equal(inp[~touched], out[~touched])
            assert np.array_equal(out[inp != bg], inp[inp != bg]), 'preserve every supplied mark'
            assert np.all(q[cov % 2 == 0] == bg)
            assert np.all(q[cov % 2 == 1] == p[cov % 2 == 1])
            assert cov.max() <= 5, 'one site can belong to at most five cardinal stencils'
            f = _features(m)
            assert f['added'] >= 3 and f['added_colors'] >= 2 and f['asymmetric']
            features[split].append(f)
    train = features['train']
    assert {f['axis'] for f in train} == {0, 1}, 'demonstrate both clue axes'
    assert any(f['canceled'] and f['unused'] and f['raw_alias'] and f['sequential'] for f in train)
    assert any(f['odd_high'] for f in train), 'distinguish parity from single coverage'
    assert any(not f['canceled'] and f['max_coverage'] == 1 for f in train)
    assert len({f['side'] for f in train}) >= 2
    assert len({f['background'] for f in train}) >= 2
    assert all(f['length'] == 2 for f in train)
    assert all(f['length'] >= 3 and f['centers'] >= 4 and f['canceled'] and f['unused'] and f['interior']
               for f in features['test'])


def _signature(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded 42-seed diversity checks plus targeted cancellation regressions."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_signature(p) for p in puzzles}) == 42
    assert _signature(puzzles[0]) == _signature(generate_puzzle(0))
    records, shapes, normalized = [], set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, m = _recover(pair['input'])
                assert np.array_equal(solve(inp), pair['output'])
                records.append((split, _features(m)))
                shapes.add(inp.shape)
                roles = {}
                normalized.add((inp.shape, tuple(roles.setdefault(int(v), len(roles)) for v in inp.flat)))
    assert len(shapes) >= 45 and len(normalized) >= 160
    for key, minimum in [('plan_shape', 15), ('match', 15), ('base', 30), ('frame', 35),
                         ('background', 9), ('marker', 9), ('centers', 8)]:
        assert len({f[key] for _, f in records}) >= minimum, key
    assert {f['side'] for _, f in records} == {2, 3, 4}
    assert {f['gutter'] for _, f in records} == {1, 2, 3}
    assert any(f['odd_high'] for _, f in records), 'odd triple overlap must survive'
    assert any(f['max_coverage'] >= 4 for _, f in records)
    assert any(h == w for h, w in shapes) and any(30 in s for s in shapes)
    assert {f['axis'] for split, f in records if split == 'test'} == {0, 1}
    # A 3x4 block has two centers; the shared centers cancel, six arms survive.
    block = np.ones((3, 4), dtype=np.int8)
    q, count, centers = _print(block, 0)
    assert len(centers) == 2 and (count > 0).sum() == 8 and (q != 0).sum() == 6
    assert count[1, 1] == count[1, 2] == 2 and q[1, 1] == q[1, 2] == 0
    # A third coverage survives, rather than keeping only singly covered sites.
    block = np.ones((3, 5), dtype=np.int8)
    q, count, centers = _print(block, 0)
    assert count[1, 2] == 3 and q[1, 2] == 1
    # Clipped crosses do not wrap; multi-color cardinal neighborhoods fail.
    clipped = np.array([[1, 1, 1], [0, 1, 0], [0, 0, 0]], dtype=np.int8)
    assert not np.any(_print(clipped, 0)[0])
    mixed = np.array([[0, 1, 0], [1, 2, 1], [0, 1, 0]], dtype=np.int8)
    assert not np.any(_print(mixed, 0)[0])
    # Forced constructive fallbacks must be valid for both specimen axes.
    for kind in ('witness', 'clean', 'test', 'mixed'):
        for axis in (0, 1):
            p, match, seq = _fallback(0, [2, 3], axis, kind)
            assert _acceptable(p, 0, axis, match, seq, kind)
            assert _matches(_print(p, 0)[0], seq, axis) == [match]
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    _, m = _recover(bad['test'][0]['input'])
    r, c = map(int, np.argwhere((m['coverage'] > 0) & (m['coverage'] % 2 == 0))[0])
    y, x = m['base'][0] + r * m['stride'], m['base'][1] + c * m['stride']
    bad['test'][0]['output'][y:y + m['side'], x:x + m['side']] = m['plan'][r, c]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('accepted union instead of cancellation')
    # All explicit legal counts; nonintegral, boolean, None and bounds rejected.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(9137 + 10 * nt + ns, nt, ns)
            assert len(p['train']) == nt and len(p['test']) == ns
    for nt, ns in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4., 1), (None, 1), (4, None)]:
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            continue
        raise AssertionError('accepted illegal counts')
    return {'seeds': 42, 'pairs': len(records), 'shapes': len(shapes),
            'normalized_structures': len(normalized),
            'plan_shapes': len({f['plan_shape'] for _, f in records}),
            'center_counts': sorted({f['centers'] for _, f in records}),
            'max_coverage': max(f['max_coverage'] for _, f in records)}
