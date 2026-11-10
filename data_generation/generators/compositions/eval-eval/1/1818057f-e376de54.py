"""Plus-anchored tail equalization. Standard library and NumPy only.

No files, latent puzzle conventions or generation state enter solve().
All tests stream puzzles; this module writes no artifacts or execution traces.
"""
from __future__ import annotations

import hashlib
from collections import deque

import numpy as np


_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
HOST_SEEDS = (0, 554512021, 627948364, 289607258, 237684291, 828369646,
              538333042, 481127935, 833618734, 770313317, 972931297,
              855240708, 9128985, 985529801, 396902951, 719141558,
              427868996, 14026693, 993196591, 599696190, 229473265,
              586104140, 484797938, 514937674, 75063946, 10105629,
              272762700, 232721344, 196848561, 878210395, 161242397,
              452776069, 966428473, 585681989, 135471369, 101293089,
              234511760, 658241458, 106445595, 982432308, 777370442,
              975981810)


def _integer(value, name, lower, upper=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < lower or (upper is not None and value > upper):
        raise ValueError(f"{name} is out of range")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError("expected an integer ARC grid of size 1..30, colors 0..9")
    return a


def _key(a):
    a = np.asarray(a, dtype=np.uint8)
    return (a.shape, a.tobytes())


def _components(a, bg):
    remaining = set(map(tuple, np.argwhere(a != bg).tolist()))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue, cells = deque([start]), {start}
        while queue:
            r, c = queue.popleft()
            for dr, dc in _STEPS:
                p = (r + dr, c + dc)
                if p in remaining:
                    remaining.remove(p)
                    cells.add(p)
                    queue.append(p)
        result.append(cells)
    return result


def _centers(a, bg):
    """All monochrome complete pluses, detected simultaneously on the input."""
    points = set()
    for r in range(1, a.shape[0] - 1):
        for c in range(1, a.shape[1] - 1):
            color = a[r, c]
            if color != bg and all(a[r + dr, c + dc] == color for dr, dc in _STEPS):
                points.add((r, c))
    return points


def _parse(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    if np.count_nonzero(a == bg) * 2 <= a.size:
        raise ValueError("background must occupy a strict majority")
    components = _components(a, bg)
    if not 3 <= len(components) <= 7:
        raise ValueError("expected three through seven separated backbones")
    centers = _centers(a, bg)
    lanes = []
    for cells in components:
        palette = {int(a[p]) for p in cells}
        if len(palette) != 1:
            raise ValueError("each decorated backbone must be monochrome")
        rs, cs = zip(*cells)
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        candidates = []
        # A backbone spans its component's long axis; twigs are at most one
        # cell to either side. Width >=5 rules out an alternative twig axis.
        if c1 - c0 + 1 >= 5 and r1 - r0 <= 2:
            for r in range(r0, r1 + 1):
                if (all((r, c) in cells for c in range(c0, c1 + 1))
                        and all(abs(s - r) <= 1 for s in rs)):
                    candidates.append((1, r, c0, c1))
        if r1 - r0 + 1 >= 5 and c1 - c0 <= 2:
            for c in range(c0, c1 + 1):
                if (all((r, c) in cells for r in range(r0, r1 + 1))
                        and all(abs(s - c) <= 1 for s in cs)):
                    candidates.append((0, c, r0, r1))
        if len(candidates) != 1:
            raise ValueError("backbone axis must be unambiguous")
        axis, transverse, low, high = candidates[0]
        hits = cells & centers
        if len(hits) > 1:
            raise ValueError("at most one complete plus per backbone")
        center = next(iter(hits)) if hits else None
        if center is not None and center[1 - axis] != transverse:
            raise ValueError("plus must be centered on its backbone")
        lanes.append(dict(axis=axis, transverse=transverse, low=low, high=high,
                          cells=cells, color=palette.pop(), center=center))
    if len({x['axis'] for x in lanes}) != 1:
        raise ValueError("all backbones must be parallel")
    axis = lanes[0]['axis']
    lanes.sort(key=lambda x: x['transverse'])
    if any(y['transverse'] - x['transverse'] < 4 for x, y in zip(lanes, lanes[1:])):
        raise ValueError("backbone centerlines need a gap of at least four")
    common_low = len({x['low'] for x in lanes}) == 1
    common_high = len({x['high'] for x in lanes}) == 1
    if common_low == common_high:
        raise ValueError("exactly one shared endpoint baseline is required")
    sign = 1 if common_low else -1
    active = [i for i, x in enumerate(lanes) if x['center'] is not None]
    if len(active) not in (3, 5, 7):
        raise ValueError("the number of plus-bearing backbones must be odd, 3..7")
    for x in lanes:
        x['anchor'] = x['low'] if sign == 1 else x['high']
        x['end'] = x['high'] if sign == 1 else x['low']
        x['length'] = x['high'] - x['low'] + 1
        if x['center'] is not None:
            q = x['center'][axis]
            x['prefix'] = sign * (q - x['anchor'])
            x['tail'] = sign * (x['end'] - q)
            if x['prefix'] < 1 or x['tail'] < 1:
                raise ValueError("a plus must have both axial arms")
            if any(p[1 - axis] != x['transverse'] and sign * (p[axis] - q) > 0
                   for p in x['cells']):
                raise ValueError("eligible tail must be bare beyond its plus")
    reference = active[len(active) // 2]
    target = lanes[reference]['tail']
    for i in active:
        end = lanes[i]['center'][axis] + sign * target
        if not 0 <= end < a.shape[axis]:
            raise ValueError("the target tail must fit without clipping")
    return dict(background=bg, axis=axis, sign=sign, lanes=lanes,
                active=active, reference=reference, target=target)


def _point(a, lane, axial):
    return (lane['transverse'], axial) if a['axis'] == 1 else (axial, lane['transverse'])


def _edit(input_grid, a, target=None, baseline=False, all_lanes=False):
    """Edit cores only. Optional arguments are explicit shortcut diagnostics."""
    out = np.array(input_grid, copy=True)
    target = a['target'] if target is None else target
    selected = range(len(a['lanes'])) if all_lanes else a['active']
    for i in selected:
        x = a['lanes'][i]
        if baseline:
            anchor, old, new = x['anchor'], x['length'] - 1, target - 1
        else:
            anchor, old, new = x['center'][a['axis']], x['tail'], target
        for k in range(new + 1, old + 1):
            out[_point(a, x, anchor + a['sign'] * k)] = a['background']
        for k in range(old + 1, new + 1):
            p = _point(a, x, anchor + a['sign'] * k)
            if not (0 <= p[0] < out.shape[0] and 0 <= p[1] < out.shape[1]):
                raise ValueError("tail does not fit")
            out[p] = x['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Resize eligible bare tails relative to the ordinal middle plus."""
    return _edit(input_grid, _parse(input_grid))


def _orient(grid, direction, reverse_order):
    if reverse_order:
        grid = np.flipud(grid)
    if direction == 0:
        return grid.copy()
    if direction == 1:
        return grid.T.copy()
    if direction == 2:
        return np.fliplr(grid).copy()
    return np.flipud(grid.T).copy()


def _make_pair(rng, n, m, direction, target, informative=False, mono=False,
               background=None, simple=False):
    if informative:
        # Five eligible objects at positions 0,1,2,3,6. Their ordinal middle
        # is object 2; the geometric midpoint and all-object middle select 3.
        active = [0, 1, 2, 3, 6]
    else:
        active = sorted(map(int, rng.choice(n, size=m, replace=False)))
    reference = active[len(active) // 2]
    prefixes = list(map(int, rng.integers(3, 10, size=n)))
    if len({prefixes[i] for i in active}) == 1:
        prefixes[active[0]] = 3 + (prefixes[active[0]] - 2) % 7
    # Every stored state is at most 30x30; no unbounded trial or retained run.
    cap = 18
    tails = list(map(int, rng.integers(1, cap + 1, size=n)))
    tails[reference] = target
    others = [int(i) for i in rng.permutation(active) if i != reference]
    mode = str(rng.choice(('mixed', 'extend', 'trim', 'single')))
    if informative:
        majority_short = bool(rng.integers(2))
        for i in others:
            tails[i] = int(rng.integers(1, target)) if majority_short else int(rng.integers(target + 1, cap + 1))
        i = others[0]
        tails[i] = int(rng.integers(target + 1, cap + 1)) if majority_short else int(rng.integers(1, target))
    else:
        for i in active:
            if i == reference:
                continue
            if mode == 'single' or rng.random() < .2:
                tails[i] = target
            elif mode == 'extend' or (mode == 'mixed' and rng.random() < .5):
                tails[i] = int(rng.integers(1, target))
            else:
                tails[i] = int(rng.integers(target + 1, cap + 1))
        i = others[0]
        if all(tails[j] == target for j in others):
            tails[i] = int(rng.integers(1, target)) if rng.random() < .5 else int(rng.integers(target + 1, cap + 1))
    # At least one excluded T has a visibly different tail and would be
    # changed by an "equalize every branch" shortcut.
    inactive = [i for i in range(n) if i not in active]
    if inactive:
        tails[inactive[0]] = target + 1
    lengths = [p + t + 1 for p, t in zip(prefixes, tails)]
    if len(set(lengths)) == 1:
        i = inactive[0] if inactive else others[0]
        tails[i] = tails[i] - 1 if tails[i] > 1 else tails[i] + 1
        lengths[i] = prefixes[i] + tails[i] + 1
    gaps = [4] * (n - 1)
    spare = 27 - sum(gaps)
    # Seven-lane informative examples allow at most three extra gap cells,
    # preserving their strict geometric-midpoint counterexample.
    for _ in range(int(rng.integers(min(spare, 7) + 1))):
        gaps[int(rng.integers(n - 1))] += 1
    rows = [1] + [1 + int(v) for v in np.cumsum(gaps)]
    need_h = rows[-1] + 2
    need_w = max(prefixes[i] + max(tails[i], target if i in active else 0) + 1
                 for i in range(n))
    bg = int(rng.integers(10)) if background is None else background
    pool = [c for c in range(10) if c != bg]
    size = 1 if mono else int(rng.integers(2, min(n, 9) + 1))
    palette = list(map(int, rng.choice(pool, size=size, replace=False)))
    colors = palette + list(map(int, rng.choice(palette, size=n - size)))
    rng.shuffle(colors)
    # Decoration never enters the bare tails. At most one side is occupied
    # at any prefix coordinate except for the one qualifying plus center.
    extras = []
    for i, p in enumerate(prefixes):
        side = int(rng.choice((-1, 1)))
        cells = {(rows[i] + side, p)}
        if i in active:
            cells.add((rows[i] - side, p))
        richness = 0.0 if simple else float(rng.choice((.15, .45, .8)))
        for q in range(1, p):
            if rng.random() < richness:
                cells.add((rows[i] + int(rng.choice((-1, 1))), q))
        extras.append(cells)
    ink = sum(lengths) + sum(map(len, extras))
    out_ink = ink + sum(target - tails[i] for i in active)
    shapes = [(h, w) for h in range(need_h, 31) for w in range(need_w, 31)
              if h * w > 2 * max(ink, out_ink)]
    style = int(rng.integers(4))
    preferred = ([s for s in shapes if s[0] == s[1]] if style == 0 else
                 [s for s in shapes if s[0] <= need_h + 2 and s[1] <= need_w + 2] if style == 1 else
                 [s for s in shapes if max(s[0] - need_h, s[1] - need_w) >= 6] if style == 2 else shapes)
    choices = preferred or shapes
    shape = choices[int(rng.integers(len(choices)))]
    dr, dc = int(rng.integers(shape[0] - need_h + 1)), int(rng.integers(shape[1] - need_w + 1))
    inp = np.full(shape, bg, dtype=np.int8)
    out = inp.copy()
    # Labels come from the sampled geometric construction, not solve().
    for i, (r, p, t, color) in enumerate(zip(rows, prefixes, tails, colors)):
        inp[r + dr, dc:dc + p + t + 1] = color
        final_tail = target if i in active else t
        out[r + dr, dc:dc + p + final_tail + 1] = color
        for a, b in extras[i]:
            inp[a + dr, b + dc] = out[a + dr, b + dc] = color
    reverse_order = bool(rng.integers(2))
    return {k: _orient(v, direction, reverse_order) for k, v in (('input', inp), ('output', out))}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(('reverse', 'count', 'span')))
    base = int(rng.integers(4))
    first_target = int(rng.integers(2, 6 if shift == 'span' else 11))
    puzzle, seen = {'train': [], 'test': []}, set()
    first_bg = int(rng.integers(10))
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            informative = split == 'train' and index == 0
            if informative:
                n, m = 7, 5
            elif split == 'train' and index == 1:
                n, m = 5, 3
            elif split == 'test' and shift == 'count':
                n, m = 7, 7
            else:
                m = int(rng.choice((3, 5)))
                n = int(rng.integers(m, 8))
            if informative:
                target = first_target
            else:
                options = range(6, 14) if split == 'test' and shift == 'span' else range(2, 6) if shift == 'span' else range(2, 14)
                eligible = [t for t in options if not (split == 'train' and index == 1 and t == first_target)]
                target = int(rng.choice(eligible))
            direction = ((base + int(rng.integers(2)) + (2 if split == 'test' else 0)) % 4
                         if shift == 'reverse' else int(rng.integers(4)))
            bg = first_bg if informative else (first_bg + int(rng.integers(1, 10))) % 10 if split == 'train' and index == 1 else int(rng.integers(10))
            for _ in range(64):
                pair = _make_pair(rng, n, m, direction, target, informative,
                                  mono=informative or (index != 1 and rng.random() < .3),
                                  background=bg, simple=split == 'train' and index == 1)
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded uniqueness search exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def features(input_grid):
    """Bounded measurements from returned cells, with no sampled metadata."""
    a = _parse(input_grid)
    xs = [a['lanes'][i] for i in a['active']]
    tails = [x['tail'] for x in xs]
    positions = [x['transverse'] for x in xs]
    distances = [abs(2 * p - positions[0] - positions[-1]) for p in positions]
    nearest = [i for i, d in enumerate(distances) if d == min(distances)]
    geo = len(nearest) == 1 and tails[nearest[0]] != a['target']
    out = _edit(input_grid, a)
    raw_ref = a['lanes'][len(a['lanes']) // 2]['length']
    raw = _edit(input_grid, a, raw_ref, baseline=True, all_lanes=True)
    global_ref = a['lanes'][a['reference']]['length']
    global_out = _edit(input_grid, a, global_ref, baseline=True)
    return dict(shape=tuple(np.asarray(input_grid).shape), count=len(a['lanes']),
                active=len(xs), target=a['target'], direction=(a['axis'], a['sign']),
                background=a['background'], palette=len({x['color'] for x in a['lanes']}),
                gaps=tuple(y['transverse'] - x['transverse'] for x, y in zip(a['lanes'], a['lanes'][1:])),
                prefixes=tuple(x['prefix'] for x in xs), tails=tuple(tails),
                extend=any(t < a['target'] for t in tails), trim=any(t > a['target'] for t in tails),
                nonmedian=sorted(tails)[len(tails) // 2] != a['target'], geometric_witness=geo,
                excludes_middle=(len(a['lanes']) % 2 == 1 and a['reference'] != len(a['lanes']) // 2),
                rejects_no_A=not np.array_equal(raw, out), rejects_global=not np.array_equal(global_out, out),
                decorated=any(len(x['cells']) > x['length'] + (2 if x['center'] is not None else 1) for x in a['lanes']),
                changed=int(np.count_nonzero(np.asarray(input_grid) != out)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train_features = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            a = _parse(inp)
            assert np.array_equal(solve(inp), out), 'wrong tail equalization'
            assert not np.array_equal(inp, out), 'sampling must include an actual edit'
            # Check preservation and distance invariants directly on the cells.
            changed = set(map(tuple, np.argwhere(inp != out).tolist()))
            allowed = set()
            for i, x in enumerate(a['lanes']):
                if i not in a['active']:
                    assert all(out[p] == inp[p] for p in x['cells']), 'excluded component changed'
                    continue
                q, target = x['center'][a['axis']], a['target']
                for p in x['cells']:
                    if p[1 - a['axis']] != x['transverse'] or a['sign'] * (p[a['axis']] - q) <= 0:
                        assert out[p] == inp[p], 'prefix or branch changed'
                for k in range(1, max(x['tail'], target) + 1):
                    p = _point(a, x, q + a['sign'] * k)
                    assert out[p] == (x['color'] if k <= target else a['background'])
                    allowed.add(p)
            assert changed <= allowed
            assert all(out[p] == inp[p] for p in a['lanes'][a['reference']]['cells'])
            if split == 'train':
                train_features.append(features(inp))
    fs = train_features
    assert len({f['target'] for f in fs}) >= 2, 'training must reject a fixed span'
    assert len({f['count'] for f in fs}) >= 2
    assert len({f['background'] for f in fs}) >= 2
    assert any(f['palette'] == 1 for f in fs) and any(f['palette'] > 1 for f in fs)
    for name in ('extend', 'trim', 'nonmedian', 'geometric_witness', 'excludes_middle',
                 'rejects_no_A', 'rejects_global'):
        assert any(f[name] for f in fs), 'missing training evidence: ' + name
    assert any(f['active'] < f['count'] for f in fs)


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.uint8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _counterfactual_checks():
    # Hand-drawn regular lanes, staggered internal anchors, and excluded Ts.
    # The center of all five lanes is an excluded T; eligible middle is lane 1.
    g = np.zeros((19, 22), dtype=np.int8)
    rows, prefixes, tails = [1, 5, 9, 13, 17], [3, 5, 4, 6, 7], [2, 4, 8, 7, 6]
    for i, (r, p, t) in enumerate(zip(rows, prefixes, tails)):
        g[r, :p + t + 1] = 2
        g[r - 1, p] = 2
        if i in (0, 1, 4):
            g[r + 1, p] = 2
    out = solve(g)
    a = _parse(g)
    assert a['reference'] == 1 and a['target'] == 4
    assert [sum(out[r] == 2) for r in rows] == [8, 10, 13, 14, 12]
    # Swap just two certification arms: same strokes and tail lengths, but
    # a different eligible middle. A remote eligible lane must change answer.
    swapped = g.copy()
    swapped[rows[1] + 1, prefixes[1]] = 0
    swapped[rows[2] + 1, prefixes[2]] = 2
    b = _parse(swapped)
    assert b['reference'] == 2 and b['target'] == 8
    assert not np.array_equal(solve(swapped)[rows[0]], out[rows[0]])
    # Moving only the reference free endpoint changes other eligible tails.
    longer = g.copy()
    longer[rows[1], prefixes[1] + tails[1] + 1] = 2
    assert _parse(longer)['target'] == 5
    assert not np.array_equal(solve(longer)[rows[0]], out[rows[0]])
    assert np.array_equal(solve(longer)[rows[2]], g[rows[2]])
    # The full rule includes a one-cell tail although the sampler uses T>=2.
    # Its sole forward arm must survive trimming, including at an edge lane.
    shortest = g.copy()
    shortest[rows[1], prefixes[1] + 2:prefixes[1] + tails[1] + 1] = 0
    c = _parse(shortest)
    assert c['target'] == 1
    tiny_out = solve(shortest)
    for i in c['active']:
        assert tiny_out[rows[i], prefixes[i] + 1] == 2
        assert not np.any(tiny_out[rows[i], prefixes[i] + 2:] == 2)
    assert _centers(tiny_out, 0) == _centers(shortest, 0)
    # Every possible pair of opposite side-neighbor bits obeys the plus test;
    # diagonal camouflage may be present without invalidating the true center.
    for top in (0, 1):
        for bottom in (0, 1):
            patch = np.array([[2, 2 * top, 0], [2, 2, 2], [0, 2 * bottom, 2]])
            assert ((1, 1) in _centers(patch, 0)) == bool(top and bottom)


def self_test():
    """42 seeds, semantic mutations, all count combinations, no saved payloads."""
    records, hashes = [], set()
    rng = np.random.default_rng(821032)
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprint = _fingerprint(puzzle)
        assert fingerprint == _fingerprint(generate_puzzle(seed))
        hashes.add(fingerprint)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                records.append((split, features(pair['input'])))
        # A color permutation and every D4 symmetry preserve semantics.
        pair = puzzle['test'][0]
        permutation = rng.permutation(10)
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
        for k in range(4):
            for flip in (False, True):
                x, y = np.rot90(pair['input'], k), np.rot90(pair['output'], k)
                if flip:
                    x, y = np.fliplr(x), np.fliplr(y)
                assert np.array_equal(solve(x), y)
    assert len(hashes) == 42
    for split in ('train', 'test'):
        fs = [f for s, f in records if s == split]
        assert len({f['shape'] for f in fs}) >= 20
        assert len({f['direction'] for f in fs}) == 4
        assert len({f['target'] for f in fs}) >= 6
        assert {3, 5} <= {f['active'] for f in fs}
        assert len({(f['count'], f['prefixes'], f['tails'], f['gaps']) for f in fs}) >= 30
        assert any(f['decorated'] for f in fs)
        assert any(len(set(f['gaps'])) > 1 for f in fs)
        assert any(f['active'] < f['count'] for f in fs)
        assert any(f['palette'] == 1 for f in fs) and any(f['palette'] > 1 for f in fs)
    _counterfactual_checks()
    for n in range(2, 11):
        for m in range(1, min(n - 1, 4) + 1):
            p = generate_puzzle(8172 + n * 10 + m, n, m)
            assert len(p['train']) == n and len(p['test']) == m
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal example counts accepted')
    # Ensure the validator detects corruption, including a changed distractor.
    p = generate_puzzle(0)
    a = _parse(p['train'][0]['input'])
    lane = next((x for x in a['lanes'] if x['center'] is None), a['lanes'][0])
    cell = _point(a, lane, lane['anchor'])
    p['train'][0]['output'][cell] = a['background']
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError('validator accepted a corrupted prefix/distractor')


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, all legal counts, mutations and equivariance')
