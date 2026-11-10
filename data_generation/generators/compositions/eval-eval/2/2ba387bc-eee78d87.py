"""Remote looms: rank-paired solid centers phase-lock periodic frame fillings.

Only the standard library and NumPy are used. Generation writes no files.
See task_spec.md for the family, evidence requirements and deliberate shift.
"""
from itertools import permutations
import hashlib
import json

import numpy as np


HOST_SEEDS = (0,536368071,139033510,172183897,752713829,2671841,2449736,249012700,829105317,211008803,592171546,728153205,766537315,624642951,367661489,38142998,384211621,73800992,36940487,292277378,752018757,689116589,937592445,74504198,825478512,848756644,319985024,654103062,499939561,524093826,807315793,416588573,643132401,391879590,375310546,677417658,605260317,643122679,732337848,174149236,162480641,176164948)
_PERMS = tuple(permutations(range(3)))
_ORDERS = tuple((t, p, q) for t in (False, True) for p in _PERMS for q in _PERMS)
_IDENTITY = _ORDERS.index((False, (0, 1, 2), (0, 1, 2)))


def _full_box(m):
    return (m.shape == (3, 3) and 2 <= int(m.sum()) <= 8
            and all((m[0].any(), m[-1].any(), m[:, 0].any(), m[:, -1].any())))


_MASKS = tuple(m for code in range(1, 511)
               if _full_box(m := np.array([(code >> k) & 1 for k in range(9)],
                                          dtype=bool).reshape(3, 3)))


def _transform(m, order):
    transpose, rows, cols = _ORDERS[order]
    return (m.T if transpose else m)[np.ix_(rows, cols)]


_AGREE = np.array([[np.array_equal(m, _transform(m, k)) for k in range(72)]
                   for m in _MASKS], dtype=bool)


def _array(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or a.dtype.kind not in 'iu' or np.any((a < 0) | (a > 9))):
        raise ValueError('expected a 1..30 integer grid with colors 0..9')
    return a.astype(np.int8, copy=False)


def _key(a):
    a = _array(a)
    return a.shape, a.tobytes()


def _tile(s, color, bg, kind):
    a = np.full((s, s), color, dtype=np.int8)
    if kind == 'H':
        a[1:-1, 1:-1] = bg
    return a


def _separate(a, b):
    r, c, h, w = a
    y, x, hh, ww = b
    return r + h < y or y + hh < r or c + w < x or x + ww < c


def _analyze(input_grid):
    """Read only pixels; the swatch can have disconnected accent fragments."""
    a = _array(input_grid)
    bg = int(a[0, 0])
    if not all(int(a[r, c]) == bg for r, c in
               ((0, 0), (0, -1), (-1, 0), (-1, -1))):
        raise ValueError('all four corners must show the carrier background')
    candidates = []
    for color in np.unique(a):
        if color == bg:
            continue
        coords = np.argwhere(a == color)
        top, left = map(int, coords.min(axis=0))
        bottom, right = map(int, coords.max(axis=0))
        if (bottom - top, right - left) == (2, 2):
            mask = a[top:top + 3, left:left + 3] == color
            patch = a[top:top + 3, left:left + 3]
            if _full_box(mask) and np.all((patch == color) | (patch == bg)):
                candidates.append((int(color), top, left, mask.copy()))
    if len(candidates) != 1:
        raise ValueError('expected one uniquely colored full-box 3x3 swatch')
    accent, mt, ml, mask = candidates[0]
    remaining = set(map(tuple, np.argwhere((a != bg) & (a != accent))))
    objects = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, points = [start], []
        while stack:
            r, c = stack.pop()
            points.append((r, c))
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        top, left = map(int, np.min(points, axis=0))
        bottom, right = map(int, np.max(points, axis=0))
        s = bottom - top + 1
        if s not in (5, 7, 9) or right - left + 1 != s:
            raise ValueError('ordinary objects must be odd 5/7/9 squares')
        tile = a[top:top + s, left:left + s]
        color = int(tile[0, 0])
        kind = 'S' if np.all(tile == color) else 'H'
        if color in (bg, accent) or not np.array_equal(tile, _tile(s, color, bg, kind)):
            raise ValueError('square is neither a solid nor a one-cell hollow rim')
        objects.append(dict(top=top, left=left, size=s, kind=kind, color=color))
    if not objects or len({o['size'] for o in objects}) != 1:
        raise ValueError('one common square size is required')
    rectangles = [(o['top'], o['left'], o['size'], o['size']) for o in objects]
    rectangles.append((mt, ml, 3, 3))
    if any(not _separate(x, y) for i, x in enumerate(rectangles)
           for y in rectangles[i + 1:]):
        raise ValueError('bounding rectangles need a full carrier gap')
    objects.sort(key=lambda o: (o['top'], o['left']))
    h = [o for o in objects if o['kind'] == 'H']
    s = [o for o in objects if o['kind'] == 'S']
    if not all(1 <= len(q) <= 4 for q in (h, s)):
        raise ValueError('both queues need one through four objects')
    # The family avoids row ties, so no learned tie-breaking convention is needed.
    if any(len({o['top'] for o in q}) != len(q) for q in (h, s)):
        raise ValueError('within-class top-row ties are outside the family')
    return dict(grid=a, background=bg, accent=accent, motif_top=mt, motif_left=ml,
                mask=mask, objects=objects, hollows=h, solids=s, size=objects[0]['size'])


def _render(info, mode='correct', mask=None):
    """Counterfactual modes are diagnostics, never generation conventions."""
    h, s = list(info['hollows']), list(info['solids'])
    if mode in ('reverse_h', 'reverse_both'):
        h.reverse()
    if mode in ('reverse_s', 'reverse_both'):
        s.reverse()
    for kind, queue in (('h', h), ('s', s)):
        if mode in (kind + '_left', 'both_left'):
            queue.sort(key=lambda o: (o['left'], o['top']))
        if mode in (kind + '_right', 'both_right'):
            queue.sort(key=lambda o: (o['left'], o['top']), reverse=True)
    assignment = list(zip(h, s))
    if mode in ('nearest', 'nearest_unused'):
        available = list(s)
        assignment = []
        for f in h:
            if not available:
                break
            d = min(available, key=lambda d: (abs(f['top']-d['top']) +
                                             abs(f['left']-d['left']), d['top'], d['left']))
            assignment.append((f, d))
            if mode == 'nearest_unused':
                available.remove(d)
    if mode == 'first_donor':
        assignment = [(f, s[0]) for f in h]
    if mode == 'greedy':
        rows, occupied = [], {}
        for o in info['objects']:
            if o['kind'] in occupied:
                rows.append(occupied)
                occupied = {}
            occupied[o['kind']] = o
        rows.append(occupied)
        assignment = [(row['H'], row['S']) for row in rows if set(row) == {'H', 'S'}]
    out = np.full(info['grid'].shape, info['background'], dtype=np.int8)
    for f in h:
        t, l, z = f['top'], f['left'], f['size']
        out[t:t + z, l:l + z] = _tile(z, f['color'], info['background'], 'H')
    m = info['mask'] if mask is None else mask
    for f, d in assignment:
        t, l, z = f['top'], f['left'], f['size']
        ar, ac = d['top'] + z // 2, d['left'] + z // 2
        if mode == 'frame_phase':
            ar, ac = t + z // 2, l + z // 2
        elif mode == 'swatch_phase':
            ar, ac = info['motif_top'] + 1, info['motif_left'] + 1
        elif mode == 'donor_corner':
            ar, ac = d['top'], d['left']
        elif mode == 'canvas_phase':
            ar, ac = 0, 0
        phase = 0 if mode == 'no_normalization' else 1
        rr = (np.arange(t + 1, t + z - 1) - ar + phase) % 3
        cc = (np.arange(l + 1, l + z - 1) - ac + phase) % 3
        occupancy = m[rr[:, None], cc[None, :]]
        if mode == 'solid_fill':
            occupancy = np.ones((z - 2, z - 2), dtype=bool)
        color = f['color'] if mode == 'own_color' else d['color']
        out[t + 1:t + z - 1, l + 1:l + z - 1][occupancy] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Pure input-only solver: no RNG, demonstrations, files or cached examples."""
    return _render(_analyze(input_grid))


_ALTERNATIVES = ('reverse_h','reverse_s','reverse_both','h_left','h_right',
                 's_left','s_right','both_left','both_right','nearest',
                 'nearest_unused','first_donor','greedy','frame_phase',
                 'swatch_phase','donor_corner','canvas_phase',
                 'no_normalization','solid_fill','own_color')


def _evidence(infos):
    witnesses = {mode: any(not np.array_equal(_render(i), _render(i, mode))
                           for i in infos) for mode in _ALTERNATIVES}
    orders = [k for k in range(72) if all(np.array_equal(i['mask'], _transform(i['mask'], k))
                                        for i in infos)]
    return witnesses, orders


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not lo <= int(value) <= hi):
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def _place(rng, size, count):
    """Construct square shelves plus a swatch, then freely relocate rectangles.

    At most 3 sweeps over <=9 rectangles and <=784 possible sites each.
    Initial shelves guarantee feasibility; accepted moves preserve all gaps.
    """
    capacity = 31 // (size + 1)
    options = []
    for columns in range(1, min(capacity, count) + 1):
        bands = (count + columns - 1) // columns
        min_h = bands * size + count + 5  # staggered shelves and swatch footer
        min_w = columns * (size + 1) - 1
        if min_h <= 30 and min_w <= 30:
            options.append((columns, min_h, min_w))
    if not options:
        raise ValueError('square count/scale has no supported construction')
    columns, min_h, min_w = options[int(rng.integers(len(options)))]
    height = int(rng.integers(min_h, 31))
    width = int(rng.integers(max(min_w, 5), 31))
    rects = []
    top, pending = 1, count
    while pending:
        amount = min(columns, pending)
        left = int(rng.integers(width - (amount * (size + 1) - 1) + 1))
        offsets = rng.permutation(amount)
        for k in range(amount):
            rects.append((top + int(offsets[k]), left + k * (size + 1), size, size))
        top += size + amount
        pending -= amount
    # top is one row below the last occupied square; bottom border is free.
    rects.append((top, int(rng.integers(width - 2)), 3, 3))
    for _ in range(3):
        for idx in rng.permutation(len(rects)):
            idx = int(idx)
            _, _, h, w = rects[idx]
            rr, cc = np.indices((height - h + 1, width - w + 1))
            legal = np.ones(rr.shape, dtype=bool)
            # No rectangle may cover one of the four background corners.
            for cr, cl in ((0, 0), (0, width-1), (height-1, 0), (height-1, width-1)):
                legal &= ~((rr <= cr) & (cr < rr+h) & (cc <= cl) & (cl < cc+w))
            for j, (r, c, hh, ww) in enumerate(rects):
                if j == idx:
                    continue
                legal &= ((rr + h < r) | (r + hh < rr) | (cc + w < c) | (c + ww < cc))
                if idx < count and j < count:
                    legal &= rr != r
            sites = np.argwhere(legal)
            if not len(sites):
                raise AssertionError('initial feasible packing was lost')
            r, c = map(int, sites[int(rng.integers(len(sites)))])
            rects[idx] = (r, c, h, w)
    return (height, width), sorted(rects[:count]), rects[-1]


def _choose_mask(rng, allowed=None):
    ids = np.arange(len(_MASKS)) if allowed is None else np.flatnonzero(allowed)
    densities = np.array([int(_MASKS[k].sum()) for k in ids])
    density = int(rng.choice(np.unique(densities)))
    return int(rng.choice(ids[densities == density]))


def _make_pair(rng, size, counts, mask_id, demonstration, index):
    nh, ns = counts
    shape, sites, swatch = _place(rng, size, nh + ns)
    bg = int(rng.choice([c for c in range(10) if c != demonstration]))
    accent = int(rng.choice([c for c in range(10) if c not in (bg, demonstration)]))
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c not in (bg, accent)])]
    if index < 2:
        # Early same-class runs expose independent matching versus greedy pairing.
        kinds = ['H'] * nh + ['S'] * ns if index == 0 else ['S'] * ns + ['H'] * nh
    else:
        mode = int(rng.integers(3))
        if mode == 0:
            kinds = list(rng.permutation(['H'] * nh + ['S'] * ns))
        elif mode == 1:
            kinds = ['H'] * nh + ['S'] * ns
            if rng.integers(2):
                kinds.reverse()
        else:
            kinds = []
            for rank in range(max(nh, ns)):
                if rank < nh:
                    kinds.append('H')
                if rank < ns:
                    kinds.append('S')
    # Early donor colors are distinct, making their independent ordering visible.
    solid_colors = palette[:ns] if index < 2 or rng.random() < 0.65 else [int(rng.choice(palette[:2])) for _ in range(ns)]
    hollow_colors = [int(rng.choice(palette[:int(rng.integers(2, 9))])) for _ in range(nh)]
    if index == 0:
        hollow_colors[0] = demonstration
    if index == 1:
        if demonstration in solid_colors:
            j = solid_colors.index(demonstration)
            solid_colors[0], solid_colors[j] = solid_colors[j], solid_colors[0]
        else:
            solid_colors[0] = demonstration
    inp = np.full(shape, bg, dtype=np.int8)
    ranks = {'H': 0, 'S': 0}
    for (top, left, _, _), kind in zip(sites, kinds):
        color = (hollow_colors if kind == 'H' else solid_colors)[ranks[kind]]
        ranks[kind] += 1
        inp[top:top + size, left:left + size] = _tile(size, color, bg, kind)
    top, left, _, _ = swatch
    inp[top:top + 3, left:left + 3][_MASKS[mask_id]] = accent
    return {'input': inp, 'output': solve(inp)}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or int(seed) < 0):
        raise ValueError('seed must be a non-negative integer')
    seed = int(seed)
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    demonstration = int(rng.integers(10))
    # One previously unseen size in tests: larger, smaller or intermediate.
    held_size = int(rng.choice((5, 7, 9), p=(0.2, 0.2, 0.6)))
    train_sizes = [s for s in (5, 7, 9) if s != held_size]
    rng.shuffle(train_sizes)
    # At most 64 two-pair witness trials, no retained rejected grids or logs.
    for _ in range(64):
        first = _choose_mask(rng)
        eligible = (_AGREE & _AGREE[first]).sum(axis=1) == 1
        eligible[first] = False
        second = _choose_mask(rng, eligible)
        first_two = [_make_pair(rng, train_sizes[0], (2, 3), first, demonstration, 0),
                     _make_pair(rng, train_sizes[1], (3, 2), second, demonstration, 1)]
        witnesses, orders = _evidence([_analyze(p['input']) for p in first_two])
        if all(witnesses.values()) and orders == [_IDENTITY]:
            break
    else:
        raise RuntimeError('bounded informative-demonstration search exhausted')
    puzzle = {'train': first_two, 'test': []}
    used = {_key(p['input']) for p in first_two}
    train_mask_keys = {_analyze(p['input'])['mask'].tobytes() for p in first_two}
    for split, count in (('train', num_train - 2), ('test', num_test)):
        for local in range(count):
            size = int(rng.choice(train_sizes)) if split == 'train' else held_size
            limit = 4 if size == 5 else 3
            for attempt in range(32):
                counts = (int(rng.integers(1, limit + 1)), int(rng.integers(1, limit + 1)))
                mask_id = _choose_mask(rng)
                if split == 'test' and _MASKS[mask_id].tobytes() in train_mask_keys:
                    # Finite eligible space, not an unbounded motif retry loop.
                    allowed = np.array([m.tobytes() not in train_mask_keys for m in _MASKS])
                    mask_id = _choose_mask(rng, allowed)
                index = len(puzzle['train']) + len(puzzle['test'])
                pair = _make_pair(rng, size, counts, mask_id, demonstration, index)
                if _key(pair['input']) not in used:
                    break
            else:
                raise RuntimeError('bounded distinct-input search exhausted')
            used.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                train_mask_keys.add(_MASKS[mask_id].tobytes())
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = _array(pair['input']), _array(pair['output'])
    info = _analyze(inp)
    assert out.shape == inp.shape
    assert np.array_equal(out, solve(inp))
    # Check physical support, preserved rims and conditional carrier footprint.
    allowed = np.zeros(inp.shape, dtype=bool)
    for f in info['hollows']:
        t, l, z = f['top'], f['left'], f['size']
        allowed[t:t + z, l:l + z] = True
        assert np.array_equal(out[t, l:l + z], inp[t, l:l + z])
        assert np.array_equal(out[t+z-1, l:l + z], inp[t+z-1, l:l + z])
        assert np.array_equal(out[t:t + z, l], inp[t:t + z, l])
        assert np.array_equal(out[t:t + z, l+z-1], inp[t:t + z, l+z-1])
    assert np.all(out[~allowed] == info['background'])
    for k, f in enumerate(info['hollows']):
        t, l, z = f['top'], f['left'], f['size']
        interior = out[t + 1:t + z - 1, l + 1:l + z - 1]
        if k >= len(info['solids']):
            assert np.all(interior == info['background'])
            continue
        donor = info['solids'][k]
        # Direct pointwise invariant rather than a second vectorized rendering.
        ar, ac = donor['top'] + z // 2, donor['left'] + z // 2
        for y in range(z - 2):
            for x in range(z - 2):
                occupied = info['mask'][(t + y + 2 - ar) % 3, (l + x + 2 - ac) % 3]
                assert interior[y, x] == (donor['color'] if occupied else info['background'])
    return info


def validate_generated_puzzle(puzzle):
    """Validate the generation contract, including actual training counterexamples."""
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    infos, seen = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(pair['input'])
            assert key not in seen, 'inputs must be distinct'
            seen.add(key)
            infos[split].append(info)
    train = infos['train']
    differences = [len(i['solids']) - len(i['hollows']) for i in train]
    assert min(differences) < 0 < max(differences), 'teach both unmatched cases'
    roles = {}
    for i in train:
        for o in i['objects']:
            roles.setdefault(o['color'], set()).add(o['kind'])
    assert any(v == {'H', 'S'} for v in roles.values()), 'geometry must outrank color'
    witnesses, orders = _evidence(train)
    assert all(witnesses.values()), f'missing counterexamples: {witnesses}'
    assert orders == [_IDENTITY], 'training leaves motif orientation ambiguous'
    sizes = {i['size'] for i in train}
    assert len(sizes) == 2
    assert all(i['size'] not in sizes for i in infos['test']), 'missing size shift'
    masks = {i['mask'].tobytes() for i in train}
    assert all(i['mask'].tobytes() not in masks for i in infos['test'])


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        h.update(split.encode())
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = _array(pair[role])
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def _metamorphic_checks():
    # A roomy, human-readable fixture allows legal one-cell and period moves.
    def fixture(mask, donor_top=3, swatch_left=3):
        a = np.zeros((28, 27), dtype=np.int8)
        for t, l, color, kind in ((2,2,2,'H'), (10,2,3,'H'),
                                   (donor_top,18,5,'S'), (18,18,6,'S')):
            a[t:t+5,l:l+5] = _tile(5, color, 0, kind)
        a[24:27,swatch_left:swatch_left+3][mask] = 8
        return a
    m = np.array([[1,0,0],[1,1,0],[0,0,1]], dtype=bool)
    a = fixture(m)
    out = solve(a)
    assert np.array_equal(out[3:6,3:6], [[5,0,0],[0,5,0],[0,5,5]])
    assert np.array_equal(solve(fixture(m, donor_top=6)), out), 'three-cell period'
    moved = solve(fixture(m, donor_top=4))
    assert not np.array_equal(moved, out), 'one-cell donor shift must affect phase'
    assert np.array_equal(moved[11:14,3:6], out[11:14,3:6]), 'other donor stays independent'
    assert np.array_equal(solve(fixture(m, swatch_left=10)), out), 'swatch position is irrelevant'
    assert np.array_equal(solve(np.pad(a, 1)), np.pad(out, 1)), 'joint spatial translation'
    perm = np.array([7,4,0,9,3,1,8,2,5,6], dtype=np.int8)
    assert np.array_equal(solve(perm[a]), perm[out]), 'semantic color roles'
    for motif in _MASKS:
        pair = {'input': fixture(motif), 'output': solve(fixture(motif))}
        _check_pair(pair)
    return len(_MASKS)


def self_test():
    """Bounded streaming audit over the exact 42 host seeds; no file writes."""
    stats = {k: set() for k in ('fingerprints','shapes','layouts','masks','outputs',
                                'sizes','backgrounds','accents','counts','shifts','phases')}
    repeated = edges = 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        stats['fingerprints'].add(_fingerprint(puzzle))
        train_sizes = sorted({_analyze(p['input'])['size'] for p in puzzle['train']})
        test_size = _analyze(puzzle['test'][0]['input'])['size']
        stats['shifts'].add((tuple(train_sizes), test_size))
        for split in ('train', 'test'):
            for p in puzzle[split]:
                info = _check_pair(p)
                stats['shapes'].add(p['input'].shape)
                stats['layouts'].add(tuple((o['top'],o['left'],o['kind']) for o in info['objects']))
                stats['masks'].add(info['mask'].tobytes())
                stats['outputs'].add(hashlib.sha256(p['output'].tobytes()).hexdigest())
                for key, value in (('sizes',info['size']),('backgrounds',info['background']),
                                   ('accents',info['accent']),('counts',(len(info['hollows']),len(info['solids'])))):
                    stats[key].add(value)
                repeated += len({o['color'] for o in info['objects']}) < len(info['objects'])
                edges += any(o['top'] == 0 or o['left'] == 0 or
                             o['top']+o['size'] == p['input'].shape[0] or
                             o['left']+o['size'] == p['input'].shape[1] for o in info['objects'])
                for f, d in zip(info['hollows'], info['solids']):
                    stats['phases'].add(((f['top']-d['top']) % 3, (f['left']-d['left']) % 3))
        assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(seed))
    assert len(stats['fingerprints']) == 42
    assert stats['sizes'] == {5, 7, 9} and len(stats['shifts']) == 3
    assert len(stats['masks']) >= 65 and len(stats['layouts']) >= 180
    assert len(stats['shapes']) >= 30 and len(stats['outputs']) >= 200
    assert len(stats['backgrounds']) == len(stats['accents']) == 10
    assert len(stats['counts']) >= 12 and len(stats['phases']) == 9
    assert repeated >= 25 and edges >= 20
    # Every legal count combination, repeated deterministically at a fixed seed.
    explicit = 0
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(1000 + explicit, a, b)
            assert len(p['train']) == a and len(p['test']) == b
            assert _fingerprint(p) == _fingerprint(generate_puzzle(1000 + explicit, a, b))
            explicit += 1
    illegal = [(1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,True),
               (2.0,1),(4,1.0),(None,1),(4,None)]
    for a, b in illegal:
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    # Mutation and geometric invariance checks use one retained puzzle only.
    p = generate_puzzle(7, 2, 1)
    info = _analyze(p['train'][0]['input'])
    untouched = info['grid'].copy()
    solve(info['grid'])
    assert np.array_equal(info['grid'], untouched), 'solver mutated its input'
    for f, d in zip(info['hollows'], info['solids']):
        # Each >=3 interior displays every motif phase, with carrier and ink.
        z, t, l = f['size'], f['top'], f['left']
        values = set(map(int, p['train'][0]['output'][t+1:t+z-1,l+1:l+z-1].flat))
        assert values == {info['background'], d['color']}
    damaged = {s: [{k: v.copy() for k,v in pair.items()} for pair in p[s]] for s in p}
    damaged['test'][0]['output'][0,0] = (int(damaged['test'][0]['output'][0,0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    full_mask_checks = _metamorphic_checks()
    return {**{k: len(v) for k,v in stats.items()}, 'repeated_color_pairs': int(repeated),
            'edge_pairs': int(edges), 'explicit_count_combinations': explicit,
            'exhaustive_fixture_masks': full_mask_checks,
            'seeds_checked': len(HOST_SEEDS), 'semantic_scope': 'authored invariants and alternatives'}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
