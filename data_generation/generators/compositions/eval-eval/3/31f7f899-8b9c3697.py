"""Loaded-port recalibration: matching selects a length-sort; rank sets stops.

Only NumPy and the standard library are used. No parent code or evidence is
loaded. All role colors, approach directions and stopping surfaces are visible.
"""
from __future__ import annotations

from collections import Counter
import copy

import numpy as np


HOST_SEEDS = (0, 962052744, 815784595, 408814888, 766612650, 77945114,
              597459390, 555572037, 133160541, 296232575, 211593967,
              818945632, 357385592, 955322265, 756186141, 635956162,
              369157033, 723503653, 710803054, 358119613, 90843218,
              641857141, 1362915, 427873144, 950490693, 554128849,
              479883725, 62736170, 263591109, 465483485, 826303873,
              656595504, 949742420, 504841328, 554448571, 565366661,
              772139736, 612248843, 137303160, 474172034, 226447369,
              518488023)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _components(mask):
    unseen = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        cells, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _rect_cells(r0, r1, c0, c1):
    return {(r, c) for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)}


def _depth_cells(row, slot, lo, hi):
    return {(row + slot['side'] * k, c)
            for k in range(lo, hi + 1)
            for c in range(slot['x'], slot['x'] + slot['width'])}


def _paint(grid, cells, color):
    for point in cells:
        grid[point] = color


def _analyze(input_grid):
    """Parse the input's common rail, adjustable middles, ribs and rectangles."""
    grid = _grid(input_grid)
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > grid.size // 2, 'background must occupy a majority'
    rows = np.flatnonzero(np.all(grid != bg, axis=1))
    cols = np.flatnonzero(np.all(grid != bg, axis=0))
    assert len(rows) + len(cols) == 1, 'one full rail required'
    vertical = bool(len(cols))
    a = grid.T.copy() if vertical else grid
    row = int(cols[0] if vertical else rows[0])
    assert 0 < row < a.shape[0] - 1
    line_counts = np.bincount(a[row], minlength=10)
    rail = int(np.argmax(line_counts))
    assert np.count_nonzero(line_counts == line_counts[rail]) == 1
    assert rail != bg
    slots = []
    x = 0
    while x < a.shape[1]:
        color, end = int(a[row, x]), x + 1
        while end < a.shape[1] and int(a[row, end]) == color:
            end += 1
        if color != rail:
            occupied = np.flatnonzero(np.any(a[:, x:end] == color, axis=1))
            top, bottom = int(occupied[0]), int(occupied[-1])
            assert top <= row <= bottom
            assert np.all(a[top:bottom + 1, x:end] == color)
            assert (top == row) != (bottom == row), 'positive one-sided prong'
            side = 1 if top == row else -1
            depth = bottom - top  # Base cells do not count as spoke depth.
            assert 1 <= depth <= 7 and 1 <= end - x <= 4
            slots.append(dict(x=x, width=end - x, color=color,
                              side=side, depth=depth))
        x = end
    assert 2 <= len(slots) <= 5
    center_colors = {s['color'] for s in slots}
    assert len(center_colors) >= 2 and bg not in center_colors
    skeleton = set(map(tuple, np.argwhere(a == rail)))
    explained = set(skeleton)
    ribs = []
    for c in range(a.shape[1]):
        rr = np.flatnonzero(a[:, c] == rail)
        if not len(rr):
            continue
        assert row in rr and np.all(np.diff(rr) == 1)
        if len(rr) > 1:
            assert max(row - int(rr[0]), int(rr[-1]) - row) <= 7
            ribs.append(c)
    assert len(ribs) == len(slots) + 1
    assert all(b - a > 1 for a, b in zip(ribs, ribs[1:]))
    for i, s in enumerate(slots):
        assert ribs[i] + 1 < s['x']
        assert s['x'] + s['width'] < ribs[i + 1]
        assert all(a[row + s['side'], c] == rail
                   for c in (ribs[i], ribs[i + 1])), 'two positive guard arms'
        cells = _depth_cells(row, s, 0, s['depth'])
        assert explained.isdisjoint(cells)
        explained.update(cells)
    assert {p for p in map(tuple, np.argwhere(np.isin(a, list(center_colors))))} <= explained
    tokens = []
    for color in sorted(set(map(int, np.unique(a))) - center_colors - {bg, rail}):
        for cells in _components(a == color):
            rr, cc = zip(*cells)
            r0, r1, c0, c1 = min(rr), max(rr), min(cc), max(cc)
            assert cells == _rect_cells(r0, r1, c0, c1), 'solid token required'
            assert row < r0 or r1 < row
            tokens.append(dict(cells=cells, color=color, r0=r0, r1=r1,
                               c0=c0, c1=c1))
            assert explained.isdisjoint(cells)
            explained.update(cells)
    assert tokens
    max_structure_depth = max(abs(int(r) - row) for r, c in explained
                              if int(a[r, c]) in center_colors | {rail})
    assert all(min(abs(t['r0'] - row), abs(t['r1'] - row)) > max_structure_depth + 1
               for t in tokens), 'tokens must start beyond every possible stop and guard'
    assert explained == set(map(tuple, np.argwhere(a != bg)))
    matches, used = {}, set()
    max_depth = max(s['depth'] for s in slots)
    for i, s in enumerate(slots):
        candidates = []
        for j, token in enumerate(tokens):
            if (token['c0'], token['c1']) != (s['x'], s['x'] + s['width'] - 1):
                continue
            depths = [s['side'] * (r - row) for r in (token['r0'], token['r1'])]
            k0, k1 = min(depths), max(depths)
            if k0 <= max_depth + 1:
                continue
            corridor = _depth_cells(row, s, 1, k1)
            own_prong = _depth_cells(row, s, 1, s['depth'])
            if any(int(a[p]) != bg and p not in own_prong and p not in token['cells']
                   for p in corridor):
                continue
            candidates.append(dict(token=j, k0=k0, k1=k1, length=k1 - k0 + 1))
        assert len(candidates) <= 1, 'ambiguous loading'
        if candidates:
            m = candidates[0]
            assert m['token'] not in used
            used.add(m['token'])
            matches[i] = m
    assert len(matches) >= 2
    return dict(grid=a, vertical=vertical, row=row, background=bg, rail=rail,
                slots=slots, skeleton=skeleton, tokens=tokens, matches=matches,
                unmatched=len(tokens) - len(used), ribs=ribs)


def _render(analysis, assigned, *, dock=True, original_stop=False):
    """Render specified depths; used by solve and ablation experiments only."""
    a = analysis
    out = np.full_like(a['grid'], a['background'])
    _paint(out, a['skeleton'], a['rail'])
    for i, s in enumerate(a['slots']):
        _paint(out, _depth_cells(a['row'], s, 0, assigned[i]), s['color'])
        if i in a['matches'] and dock:
            m = a['matches'][i]
            stop = (s['depth'] if original_stop else assigned[i]) + 1
            final = _depth_cells(a['row'], s, stop, stop + m['length'] - 1)
            swept = _depth_cells(a['row'], s, stop, m['k1'])
            _paint(out, swept - final, a['rail'])
            _paint(out, final, a['tokens'][m['token']]['color'])
    return out.T.copy() if a['vertical'] else out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _analyze(input_grid)
    assigned = [s['depth'] for s in a['slots']]
    active = sorted(a['matches'])
    lengths = sorted(assigned[i] for i in active)
    for i, depth in zip(active, lengths):
        assigned[i] = depth
    return _render(a, assigned)


def _construct_pair(rng, *, test, witness, vertical, require_sort=False):
    n = 4 if witness else int(rng.integers(4, 6) if test else rng.integers(2, 5))
    if witness:
        inactive = int(rng.choice([1, 2]))
        active = [i for i in range(n) if i != inactive]
        values = sorted(map(int, rng.choice(np.arange(2, 7), 3, replace=False)))
        order = values[1:] + values[:1] if rng.integers(2) else values[-1:] + values[:-1]
        depths = [1] * n
        for i, d in zip(active, order):
            depths[i] = d
    else:
        active = sorted(map(int, rng.choice(n, int(rng.integers(3 if test else 2, n + 1)),
                                            replace=False)))
        depths = list(map(int, rng.integers(1, 8 if test else 7, n)))
        if len(set(depths[i] for i in active)) == 1 and (require_sort or rng.random() < .75):
            depths[active[-1]] = depths[active[0]] % (7 if test else 6) + 1
        if require_sort:
            order = [depths[i] for i in active]
            ordered = sorted(order)
            if order == ordered or order[::-1] == ordered:
                options = [ordered[k:] + ordered[:k] for k in range(1, len(active))
                           if ordered[k:] + ordered[:k] != ordered and
                           (ordered[k:] + ordered[:k])[::-1] != ordered]
                assert options
                order = options[int(rng.integers(len(options)))]
                for i, d in zip(active, order):
                    depths[i] = d
    sides = list(map(int, rng.choice([-1, 1], n)))
    if witness:
        sides[active[0]], sides[active[-1]] = -1, 1
    widths = list(map(int, rng.integers(1, 4, n)))
    locked = {}
    if witness:
        short = min(active, key=lambda i: depths[i])
        long = max(active, key=lambda i: depths[i])
        locked = {short: depths[long] // depths[short] + 1, long: 1}
        for i, w in locked.items():
            widths[i] = w
    while sum(widths) + 3 * n + 1 > 28:
        candidates = [i for i in range(n) if widths[i] > locked.get(i, 1)]
        assert candidates
        i = max(candidates, key=lambda i: widths[i])
        widths[i] -= 1
    minimum = sum(widths) + 3 * n + 1
    width = int(rng.integers(minimum, 31))
    extra = list(map(int, rng.multinomial(width - minimum, [1 / (2 * n + 2)] * (2 * n + 2))))
    ribs, starts = [extra[0]], []
    position = ribs[0]
    for i in range(n):
        start = position + 2 + extra[1 + 2 * i]
        right = start + widths[i] + 1 + extra[2 + 2 * i]
        starts.append(start)
        ribs.append(right)
        position = right
    assert ribs[-1] + 1 + extra[-1] == width
    wall = {q: {-1: 0, 1: 0} for q in ribs}
    for i, side in enumerate(sides):
        for q in (ribs[i], ribs[i + 1]):
            wall[q][side] = max(wall[q][side], int(rng.integers(1, 8)))
    maximum = max(max(depths), max(d for v in wall.values() for d in v.values()))
    palette = list(map(int, rng.permutation(10)))
    bg, rail = palette[:2]
    nc = int(rng.integers(2, min(n, 3) + 1))
    centers = palette[2:2 + nc]
    colors = (centers * n)[:n]
    rng.shuffle(colors)
    ntc = int(rng.integers(1, 4))
    token_colors = palette[2 + nc:2 + nc + ntc]
    slots = [dict(x=x, width=w, color=c, side=s, depth=d)
             for x, w, c, s, d in zip(starts, widths, colors, sides, depths)]
    tokens = []

    def add_token(i, side, kind, force_length=None):
        length = force_length or int(rng.integers(1, 5 if test else 3))
        if witness and kind == 'active':
            length = 1 + active.index(i) % 2
        extra_distance = int(rng.integers(0, min(3, 13 - maximum - length)))
        leading = maximum + 2 + extra_distance
        x, w = starts[i], widths[i]
        if kind == 'wide':
            x -= 1
            w += 2  # Same center line, wrong cross-section: alignment alone fails.
        elif kind == 'shifted':
            x += 1
        tokens.append(dict(slot=i, side=side, x=x, width=w, length=length,
                           start=leading, color=int(rng.choice(token_colors)),
                           active=(kind == 'active')))

    for i in range(n):
        kind = ('active' if i in active else 'wide' if witness == 1 else
                'shifted' if witness == 2 else str(rng.choice(['wide', 'shifted'])))
        forced = int(rng.integers(3, 5)) if test and i == active[0] else None
        add_token(i, sides[i], kind, forced)
    # Wrong-side distractors test approach direction without obstructing a route.
    extra_count = (int(rng.integers(1, min(n, 3) + 1)) if test else
                   int(rng.integers(1 if witness == 2 else 0, 3)))
    for i in map(int, rng.choice(n, extra_count, replace=False)):
        add_token(i, -sides[i], 'opposite')
    extents = {-1: 0, 1: 0}
    for s in slots:
        extents[s['side']] = max(extents[s['side']], s['depth'])
    for v in wall.values():
        for side in (-1, 1):
            extents[side] = max(extents[side], v[side])
    for t in tokens:
        extents[t['side']] = max(extents[t['side']], t['start'] + t['length'] - 1)
    # Sorting can move the largest depth to either side of the rail.
    for i in active:
        extents[sides[i]] = max(extents[sides[i]], max(depths[j] for j in active))
    height_min = extents[-1] + extents[1] + 3
    assert height_min <= 29
    height = int(rng.integers(max(16, height_min), 31))
    row = int(rng.integers(extents[-1] + 1, height - extents[1] - 1))
    inp = np.full((height, width), bg, dtype=np.int8)
    inp[row] = rail
    for q, v in wall.items():
        for side, depth in v.items():
            for k in range(1, depth + 1):
                inp[row + side * k, q] = rail
    fixed = inp.copy()
    for s in slots:
        _paint(inp, _depth_cells(row, s, 0, s['depth']), s['color'])
    for t in tokens:
        for k in range(t['start'], t['start'] + t['length']):
            inp[row + t['side'] * k, t['x']:t['x'] + t['width']] = t['color']
    assigned = depths.copy()
    for i, depth in zip(active, sorted(depths[j] for j in active)):
        assigned[i] = depth
    out = fixed.copy()
    for i, s in enumerate(slots):
        _paint(out, _depth_cells(row, s, 0, assigned[i]), s['color'])
    for t in tokens:
        if t['active']:
            s = slots[t['slot']]
            stop = assigned[t['slot']] + 1
            # Directly construct the final footprint and vacated swept strip.
            for k in range(stop, t['start'] + t['length']):
                color = t['color'] if k < stop + t['length'] else rail
                out[row + s['side'] * k, s['x']:s['x'] + s['width']] = color
    if vertical:
        inp, out = inp.T.copy(), out.T.copy()
    return dict(input=inp, output=out)


def _make_unique(pair, seen):
    if _key(pair['input']) not in seen:
        return pair
    a = _analyze(pair['input'])
    old = [a['background'], a['rail']] + sorted(set(map(int, np.unique(pair['input']))) -
                                              {a['background'], a['rail']})
    # Ninety different visible background/rail assignments exceed 13 exclusions.
    for bg in range(10):
        for rail in range(10):
            if bg == rail:
                continue
            new = [bg, rail] + [c for c in range(10) if c not in (bg, rail)]
            mapping = np.arange(10, dtype=np.int8)
            for source, target in zip(old, new):
                mapping[source] = target
            candidate = {k: mapping[v] for k, v in pair.items()}
            if _key(candidate['input']) not in seen:
                return candidate
    raise AssertionError('finite recoloring exhausted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (('num_train', num_train, 2, 10),
                                   ('num_test', num_test, 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or
                not isinstance(value, (int, np.integer)) or not low <= value <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(int(seed))
    puzzle, seen = {'train': [], 'test': []}, set()
    first_vertical = bool(rng.integers(2))
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(count):
            witness = i + 1 if split == 'train' and i < 2 else 0
            vertical = first_vertical ^ bool(i) if witness else bool(rng.integers(2))
            pair = _construct_pair(rng, test=(split == 'test'), witness=witness, vertical=vertical,
                                   require_sort=(split == 'test' and i == 0))
            pair = _make_unique(pair, seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    """Relational checks: read output depths, then check conservation and motion.

    This deliberately does not call solve, sort the input depths, or use the
    constructor/renderer's output. The input parser is shared and is not an
    independent semantic judge.
    """
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape
    a = _analyze(inp)
    b = out.T.copy() if a['vertical'] else out
    assert not np.array_equal(inp, out)
    assigned = []
    explanation = np.full_like(b, a['background'])
    _paint(explanation, a['skeleton'], a['rail'])
    for i, s in enumerate(a['slots']):
        k = 0
        while (0 <= a['row'] + s['side'] * k < b.shape[0] and
               np.all(b[a['row'] + s['side'] * k,
                        s['x']:s['x'] + s['width']] == s['color'])):
            k += 1
        depth = k - 1
        assert 1 <= depth <= 7
        if i not in a['matches']:
            assert depth == s['depth'], 'inactive prong changed'
        assigned.append(depth)
        _paint(explanation, _depth_cells(a['row'], s, 0, depth), s['color'])
    remaining = Counter(a['slots'][i]['depth'] for i in a['matches'])
    previous = 0
    routes = set()
    for i, m in sorted(a['matches'].items()):
        s, depth = a['slots'][i], assigned[i]
        assert depth >= previous, 'active lengths are not increasing'
        assert remaining[depth] > 0, 'active depth multiset not conserved'
        remaining[depth] -= 1
        previous = depth
        final = _depth_cells(a['row'], s, depth + 1, depth + m['length'])
        swept = _depth_cells(a['row'], s, depth + 1, m['k1'])
        assert all(0 <= r < b.shape[0] for r, _ in swept)
        assert swept.isdisjoint(routes) and swept.isdisjoint(a['skeleton'])
        routes.update(swept)
        color = a['tokens'][m['token']]['color']
        assert all(int(b[p]) == color for p in final), 'rigid token or new stop incorrect'
        assert all(int(b[p]) == a['rail'] for p in swept - final), 'swept strip incorrect'
        _paint(explanation, swept - final, a['rail'])
        _paint(explanation, final, color)
    assert not +remaining
    assert np.array_equal(explanation, b), 'unexplained cells or failed distractor erasure'
    active = sorted(a['matches'])
    source_depths = [a['slots'][i]['depth'] for i in active]
    target_depths = [assigned[i] for i in active]
    all_depths = [s['depth'] for s in a['slots']]
    area_order = sorted(active, key=lambda i: a['slots'][i]['depth'] * a['slots'][i]['width'])
    return a, dict(changed_sort=(source_depths != target_depths),
                   non_reversal=(source_depths[::-1] != target_depths),
                   area_disagrees=([a['slots'][i]['depth'] for i in area_order] != target_depths),
                   all_sort_disagrees=(sorted(all_depths) != assigned), assigned=assigned)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence, test_features = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            key = _key(_grid(pair['input']))
            assert key not in seen, 'duplicate input'
            seen.add(key)
            a, f = _check_pair(pair)
            (evidence if split == 'train' else test_features).append((a, f))
    # Both axes and both approach sides have a consequential demonstrated sort.
    witnesses = [(a, f) for a, f in evidence if f['changed_sort'] and f['non_reversal']
                 and {a['slots'][i]['side'] for i in a['matches']} == {-1, 1}]
    assert {a['vertical'] for a, _ in witnesses} == {False, True}
    assert any(f['area_disagrees'] for _, f in evidence)
    assert any(f['all_sort_disagrees'] and a['unmatched'] > 0 for a, f in evidence)
    assert set().union(*(_unmatched_kinds(a) for a, _ in evidence)) >= {'wide', 'shifted', 'opposite'}
    assert {m['length'] for a, _ in evidence for m in a['matches'].values()} >= {1, 2}
    for a, _ in test_features:
        assert len(a['slots']) >= 4
        assert len(a['matches']) >= 3
        assert any(m['length'] >= 3 for m in a['matches'].values())
        assert a['unmatched'] >= 1
    assert test_features[0][1]['changed_sort'] and test_features[0][1]['non_reversal']


def _unmatched_kinds(a):
    used = {m['token'] for m in a['matches'].values()}
    kinds = set()
    for j, t in enumerate(a['tokens']):
        if j in used:
            continue
        for i, s in enumerate(a['slots']):
            if not a['ribs'][i] < t['c0'] <= t['c1'] < a['ribs'][i + 1]:
                continue
            span = t['c1'] - t['c0'] + 1
            same_side = (t['r0'] > a['row']) == (s['side'] > 0)
            exact = (t['c0'], span) == (s['x'], s['width'])
            if exact and not same_side:
                kinds.add('opposite')
            elif same_side and span == s['width'] and not exact:
                kinds.add('shifted')
            elif (same_side and span != s['width'] and
                  t['c0'] + t['c1'] == 2 * s['x'] + s['width'] - 1):
                kinds.add('wide')
    return kinds


def _normalize(grid):
    colors = {}
    return grid.shape, tuple(colors.setdefault(int(c), len(colors)) for c in grid.ravel())


def self_test():
    """Bounded 42-seed sweep, all explicit counts, and semantic mutations."""
    puzzles, shapes, normalized, widths, depths = set(), set(), set(), set(), set()
    slot_counts, active_counts, sides, token_lengths, role_colors = set(), set(), set(), set(), set()
    movements, tied_pairs, statistics = set(), 0, Counter()
    first = None
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        sig = tuple((_key(pair['input']), _key(pair['output']))
                    for split in ('train', 'test') for pair in p[split])
        puzzles.add(sig)
        if seed == 0:
            first = p
            assert sig == tuple((_key(pair['input']), _key(pair['output']))
                                for split in ('train', 'test')
                                for pair in generate_puzzle(seed)[split])
        for split in ('train', 'test'):
            for pair in p[split]:
                before = np.asarray(pair['input']).copy()
                assert np.array_equal(solve(before), pair['output'])
                assert np.array_equal(before, pair['input']), 'solve modified input'
                a, f = _check_pair(pair)
                shapes.add(pair['input'].shape)
                normalized.add(_normalize(pair['input']))
                slot_counts.add(len(a['slots']))
                active_counts.add(len(a['matches']))
                role_colors.add((a['background'], a['rail']))
                widths.update(s['width'] for s in a['slots'])
                depths.update(s['depth'] for s in a['slots'])
                sides.update((a['vertical'], a['slots'][i]['side']) for i in a['matches'])
                token_lengths.update(m['length'] for m in a['matches'].values())
                movements.update(m['k0'] - f['assigned'][i] - 1 for i, m in a['matches'].items())
                active_depths = [a['slots'][i]['depth'] for i in a['matches']]
                tied_pairs += len(set(active_depths)) < len(active_depths)
                statistics['pairs'] += 1
                statistics['area_disagreements'] += int(f['area_disagrees'])
                statistics['gate_disagreements'] += int(f['all_sort_disagrees'])
    assert len(puzzles) == 42 and len(shapes) >= 25 and len(normalized) >= 180
    assert slot_counts == {2, 3, 4, 5} and {2, 3, 4} <= active_counts
    assert widths == {1, 2, 3, 4} and depths == set(range(1, 8))
    assert token_lengths == {1, 2, 3, 4}
    assert sides == {(False, -1), (False, 1), (True, -1), (True, 1)}
    assert len(role_colors) >= 40 and len(movements) >= 6 and tied_pairs >= 10
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(91821, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Alter solved outputs in ways that mimic tempting simpler rules.
    pair = next(pair for pair in first['train'] if _check_pair(pair)[1]['area_disagrees'])
    a, f = _check_pair(pair)
    original = [s['depth'] for s in a['slots']]
    active = sorted(a['matches'])
    area = original.copy()
    for i, j in zip(active, sorted(active, key=lambda j: original[j] * a['slots'][j]['width'])):
        area[i] = original[j]
    mutations = [_render(a, original), _render(a, sorted(original)),
                 _render(a, area), _render(a, f['assigned'], dock=False),
                 _render(a, f['assigned'], original_stop=True)]
    for out in mutations:
        assert not np.array_equal(out, pair['output'])
        try:
            _check_pair(dict(input=pair['input'], output=out))
        except AssertionError:
            pass
        else:
            raise AssertionError('semantic mutation accepted')
    damaged = copy.deepcopy(first)
    ap = _analyze(damaged['test'][0]['input'])
    i, m = next(iter(ap['matches'].items()))
    s = ap['slots'][i]
    # A corruption of an actual solved token footprint must be rejected.
    aa, ff = _check_pair(damaged['test'][0])
    cell = next(iter(_depth_cells(aa['row'], s, ff['assigned'][i] + 1, ff['assigned'][i] + 1)))
    if aa['vertical']:
        cell = cell[::-1]
    damaged['test'][0]['output'][cell] = aa['background']
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('damaged final token accepted')
    return dict(seeds=42, distinct_puzzles=len(puzzles), shapes=len(shapes),
                color_normalized_inputs=len(normalized), tied_pairs=tied_pairs,
                movements=list(map(int, sorted(movements))), role_assignments=len(role_colors),
                **statistics)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
