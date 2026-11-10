"""ARC family: broadcast the differences between repeated capped diamonds.

Only the standard library and NumPy are used. Generation recipes are never
consulted by solve; the repeated visible motifs supply its reference template.
No files, traces, caches, or logs are written by this module.
"""
from __future__ import annotations

from collections import Counter
import hashlib

import numpy as np


# Clockwise order, with row coordinates increasing downward.
VECTORS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
RADII = (2, 3)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _background(shape, colors):
    return np.asarray(colors, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _outline(center, radius):
    r, c = center
    return [(r + dr, c + dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1) if abs(dr) + abs(dc) == radius]


def _cap(center, radius, direction):
    r, c = center
    vr, vc = VECTORS[direction]
    return [(r + vr * (k + 1), c + vc * (radius - k + 1))
            for k in range(radius + 1)]


def _footprint(center, radius, states):
    cells = set(_outline(center, radius))
    for d, color in enumerate(states):
        if color is not None:
            cells.update(_cap(center, radius, d))
    return cells


def _changes(master, variant):
    # Absence is a state, not a literal background color. A removal broadcasts
    # the old color; an addition/recolor broadcasts the new color.
    return [(d, old if new is None else new,
             'remove' if new is None else 'add' if old is None else 'recolor')
            for d, (old, new) in enumerate(zip(master, variant)) if old != new]


def _paths(center, radius, direction, shape, all_lanes=False):
    cap = _cap(center, radius, direction)
    starts = cap if all_lanes else (cap[0], cap[-1])
    vr, vc = VECTORS[direction]
    paths = []
    for r, c in starts:
        path = []
        r, c = r + vr, c + vc
        while _inside((r, c), shape):
            path.append((r, c))
            r, c = r + vr, c + vc
        paths.append(path)
    return paths


def _ray_map(center, radius, changes, shape):
    result = {}
    for d, color, _ in changes:
        for path in _paths(center, radius, d, shape):
            for p in path:
                assert p not in result
                result[p] = color
    return result


def _recover(value):
    """Recover every role, emitter, reference state, and discrepancy from pixels."""
    grid = _grid(value)
    h, w = grid.shape
    parity = np.indices(grid.shape).sum(axis=0) % 2
    colors = []
    for p in (0, 1):
        counts = np.bincount(grid[parity == p], minlength=10)
        color = int(counts.argmax())
        assert counts[color] * 2 > int(np.count_nonzero(parity == p))
        colors.append(color)
    assert colors[0] != colors[1]
    base = _background(grid.shape, colors)
    emitters = []
    for radius in RADII:
        sh = (h - 2 * radius, w - 2 * radius)
        if min(sh) <= 0:
            continue
        offsets = _outline((0, 0), radius)
        dr, dc = offsets[0]
        values = grid[radius + dr:radius + dr + sh[0],
                      radius + dc:radius + dc + sh[1]]
        mask = (values != colors[0]) & (values != colors[1])
        for dr, dc in offsets[1:]:
            mask &= grid[radius + dr:radius + dr + sh[0],
                         radius + dc:radius + dc + sh[1]] == values
        for rr, cc in np.argwhere(mask):
            emitters.append({'center': (int(rr) + radius, int(cc) + radius),
                             'radius': radius, 'struct': int(values[rr, cc])})
    assert 3 <= len(emitters) <= 5
    assert len({e['radius'] for e in emitters}) == 1
    assert len({e['struct'] for e in emitters}) == 1
    covered = set()
    for e in emitters:
        states = []
        cells = set(_outline(e['center'], e['radius']))
        for d in range(4):
            cap = _cap(e['center'], e['radius'], d)
            assert all(_inside(p, grid.shape) for p in cap), 'caps are not clipped'
            values = [int(grid[p]) for p in cap]
            if all(grid[p] == base[p] for p in cap):
                states.append(None)
            else:
                assert len(set(values)) == 1
                assert values[0] not in (*colors, e['struct'])
                states.append(values[0])
                cells.update(cap)
        assert not cells & covered, 'overlapping motifs'
        covered.update(cells)
        e['states'] = tuple(states)
    assert covered == set(map(tuple, np.argwhere(grid != base)))
    counts = Counter(e['states'] for e in emitters)
    assert len(counts) == 2
    master, n = counts.most_common(1)[0]
    assert n == len(emitters) - 1 and n >= 2
    exceptional = [e for e in emitters if e['states'] != master]
    assert len(exceptional) == 1
    odd = exceptional[0]
    changes = _changes(master, odd['states'])
    rays = _ray_map(odd['center'], odd['radius'], changes, grid.shape)
    assert rays and all(grid[p] == base[p] for p in rays), 'rays must cross background'
    return {'grid': grid, 'base': base, 'background': tuple(colors),
            'emitters': emitters, 'master': master, 'odd': odd,
            'changes': changes, 'rays': rays}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the repeated template, then extend just discrepant cap endpoints."""
    info = _recover(input_grid)
    out = info['grid'].copy()
    for p, color in info['rays'].items():
        out[p] = color
    return out


def _analytic_output(grid, center, radius, changes):
    """Closed-form ray membership, independent of the walking implementation."""
    rows, cols = np.indices(grid.shape)
    out = grid.copy()
    used = np.zeros(grid.shape, dtype=bool)
    for d, color, _ in changes:
        vr, vc = VECTORS[d]
        u, v = (rows - center[0]) * vr, (cols - center[1]) * vc
        mask = (np.abs(u - v) == radius) & (np.minimum(u, v) >= 2)
        assert not np.any(mask & used)
        out[mask] = color
        used |= mask
    return out


def _teaching(info, out):
    grid, base, odd = info['grid'], info['base'], info['odd']
    center, radius = odd['center'], odd['radius']
    evidence = set()
    for d, color, _ in info['changes']:
        paths = _paths(center, radius, d, grid.shape)
        vr, vc = VECTORS[d]
        interior = [(r + k * vr, c + k * vc)
                    for r, c in _cap(center, radius, d)[1:-1] for k in (1, 2)]
        if (min(map(len, paths)) >= 2
                and all(_inside(p, grid.shape) and grid[p] == base[p] == out[p]
                        for p in interior)
                and all(out[p] == color for path in paths for p in path[:2])):
            evidence.add(d)
    # Explicitly observed but unchanged caps on the odd emitter stay silent.
    silent = 0
    for d, state in enumerate(odd['states']):
        if state is not None and state == info['master'][d]:
            silent += sum(grid[p] == base[p] == out[p]
                          for path in _paths(center, radius, d, grid.shape)
                          for p in path[:2])
    # A shortcut firing every visible cap would add these unwanted pixels.
    all_observed = set()
    for e in info['emitters']:
        for d, state in enumerate(e['states']):
            if state is not None:
                all_observed.update(p for path in _paths(e['center'], radius, d, grid.shape)
                                    for p in path if grid[p] == base[p])
    extra = len(all_observed - set(info['rays']))
    missing = len(set(info['rays']) - all_observed)
    return evidence, int(silent), extra, missing


def _sample_states(rng, bank, mode, directions):
    order = list(map(int, directions))
    if mode == 'exchange':
        # Same cap count; one absence and one addition are both meaningful.
        removed, added, keep1, keep2 = order
        master = [int(rng.choice(bank)) for _ in range(4)]
        master[added] = None
        variant = master.copy()
        variant[removed] = None
        # Distinct colors make copying the new cap's color into both channels
        # fail: the absent channel must recover its own old reference color.
        variant[added] = int(rng.choice([c for c in bank if c != master[removed]]))
    elif mode == 'swap':
        # Full foreground color histogram is unchanged, so counting cannot
        # find the odd motif or decide its two emission colors.
        a, b, keep, absent = order
        master = [None] * 4
        x, y = map(int, rng.choice(bank, 2, replace=False))
        master[a], master[b], master[keep] = x, y, int(rng.choice(bank))
        variant = master.copy()
        variant[a], variant[b] = y, x
    else:
        master = [None] * 4
        active = rng.choice(4, int(rng.integers(1, 5)), replace=False)
        for d in active:
            master[int(d)] = int(rng.choice(bank))
        variant = master.copy()
        changed = rng.choice(4, int(mode), replace=False)
        for d in changed:
            d = int(d)
            if master[d] is None:
                variant[d] = int(rng.choice(bank))
            else:
                choices = [None] + [int(c) for c in bank if c != master[d]]
                variant[d] = choices[int(rng.integers(len(choices)))]
    return tuple(master), tuple(variant)


def _render(shape, centers, radius, background, structural, master, variant):
    grid = _background(shape, background)
    for i, center in enumerate(centers):
        states = variant if i == 0 else master
        for p in _outline(center, radius):
            grid[p] = structural
        for d, color in enumerate(states):
            if color is not None:
                for p in _cap(center, radius, d):
                    grid[p] = color
    # Labels use recipe states and a coordinate equation, not solve/recovery.
    out = _analytic_output(grid, centers[0], radius, _changes(master, variant))
    return {'input': grid, 'output': out}


def _place(rng, count, radius, master, variant, teaching):
    margin = radius + 1
    changes = _changes(master, variant)
    low = max(19 + 4 * (radius - 2), 3 * count + 10)
    low = min(low, 29)
    initial = tuple(map(int, rng.integers(low, 31, size=2)))
    for attempt in range(12):
        shape = initial if attempt < 6 else (30, 30)
        m = radius + 3 if teaching else margin
        odd = tuple(int(rng.integers(m, n - m)) for n in shape)
        rays = _ray_map(odd, radius, changes, shape)
        if not rays:
            continue
        reserved = set(rays)
        if teaching:
            # Reserve first two interior-lane probes for endpoint-only evidence,
            # plus silent endpoint probes for every unchanged visible cap.
            for d in range(4):
                vr, vc = VECTORS[d]
                cap = _cap(odd, radius, d)
                starts = cap[1:-1] if master[d] != variant[d] else (cap[0], cap[-1])
                reserved.update((r + k * vr, c + k * vc)
                                for r, c in starts for k in (1, 2))
        # Inactive cap sites must also remain readable as background. Reserving
        # the complete four-cap footprint prevents a neighbor from imitating a
        # partial cap at an otherwise absent side.
        occupied = _footprint(odd, radius, (0, 0, 0, 0))
        centers = [odd]
        candidates = [(r, c) for r in range(margin, shape[0] - margin)
                      for c in range(margin, shape[1] - margin)]
        rng.shuffle(candidates)
        for center in candidates:
            if any(abs(center[0] - p[0]) + abs(center[1] - p[1]) < 2 * radius + 4
                   for p in centers):
                continue
            cells = _footprint(center, radius, (0, 0, 0, 0))
            if cells & (occupied | reserved):
                continue
            centers.append(center)
            occupied.update(cells)
            if len(centers) == count:
                return shape, centers
    # Guaranteed full-count fallback: cardinal masters lie outside every
    # diagonal endpoint corridor, even with radius-three four-sided templates.
    masters = [(4, 14), (25, 14), (14, 4), (14, 25)]
    rng.shuffle(masters)
    return (30, 30), [(14, 14)] + masters[:count - 1]


def _unique(pair, seen):
    if _key(pair['input']) not in seen:
        return pair
    # Twenty distinct color bijections, more than the maximum 13 prior inputs.
    # With at least three colors the dihedral action on 0..9 has no stabilizer.
    for sign in (1, -1):
        for shift in range(10):
            lookup = ((sign * np.arange(10) + shift) % 10).astype(np.int8)
            candidate = {k: lookup[g] for k, g in pair.items()}
            if _key(candidate['input']) not in seen:
                return candidate
    raise AssertionError('unreachable: twenty colorings versus thirteen exclusions')


def _count(value, name, lo, hi):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi):
        raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    ne = _count(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    seed = int(seed) % (1 << 64)
    rng = np.random.default_rng(seed)
    shared_palette = bool(rng.integers(2))
    palette = list(map(int, rng.choice(10, 7, replace=False)))
    max_radius = int(rng.choice(RADII))
    slots = list(map(int, rng.choice(nt, 2, replace=False)))
    directions = list(map(int, rng.permutation(4)))
    # Witness 1 changes the first two directions; witness 2 changes the other
    # two, with their colors exchanged rather than their occupancy changed.
    witnesses = {slots[0]: ('exchange', directions),
                 slots[1]: ('swap', directions[2:] + directions[:2])}
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', nt), ('test', ne)):
        for i in range(n):
            if not shared_palette:
                palette = list(map(int, rng.choice(10, 7, replace=False)))
            background, structural = tuple(palette[:2]), palette[2]
            bank = palette[3:3 + int(rng.integers(2, 5))]
            teaching = split == 'train' and i in witnesses
            mode, order = witnesses[i] if teaching else (
                str(int(rng.integers(1, 3) if split == 'train' else rng.integers(3, 5))),
                list(map(int, rng.permutation(4))))
            master, variant = _sample_states(rng, bank, mode, order)
            radius = max_radius if teaching else int(rng.integers(2, max_radius + 1))
            count = int(rng.integers(3, 6))
            shape, centers = _place(rng, count, radius, master, variant, teaching)
            pair = _unique(_render(shape, centers, radius, background, structural,
                                   master, variant), seen)
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    info = _recover(pair['input'])
    odd = info['odd']
    evidence, silent, extra, missing = _teaching(info, np.asarray(pair['output']))
    lengths = [len(path) for d, _, _ in info['changes']
               for path in _paths(odd['center'], odd['radius'], d, info['grid'].shape)]
    kinds = tuple(k for _, _, k in info['changes'])
    master, variant = info['master'], odd['states']
    histogram = lambda states: Counter(c for c in states if c is not None)
    return {'count': len(info['emitters']), 'radius': odd['radius'],
            'shape': info['grid'].shape, 'background': info['background'],
            'master_mask': sum(1 << d for d, c in enumerate(master) if c is not None),
            'changed_mask': sum(1 << d for d, _, _ in info['changes']),
            'kinds': kinds, 'directions': evidence, 'silent': silent,
            'all_changes_taught': evidence == {d for d, _, _ in info['changes']},
            'omit_a_extra': extra, 'omit_a_missing': missing,
            'balanced_exchange': ('add' in kinds and 'remove' in kinds
                                  and sum(c is not None for c in master)
                                  == sum(c is not None for c in variant)
                                  and len({color for _, color, _ in info['changes']}) >= 2),
            'color_swap': ('recolor' in kinds and histogram(master) == histogram(variant)),
            'ray_lengths': lengths,
            'layout': (info['grid'].shape, odd['radius'],
                       tuple(sorted(e['center'] for e in info['emitters'])),
                       odd['center'], tuple(c is not None for c in master),
                       tuple(d for d, _, _ in info['changes']))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train_features, test_features = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _recover(inp)
            assert np.array_equal(out, solve(inp))
            assert np.array_equal(out, _analytic_output(inp, info['odd']['center'],
                                                       info['odd']['radius'], info['changes']))
            foreground = inp != info['base']
            assert np.array_equal(inp[foreground], out[foreground])
            assert set(map(tuple, np.argwhere(inp != out))) == set(info['rays'])
            features = _features(pair)
            (train_features if split == 'train' else test_features).append(features)
    # These are demonstration/shift requirements, not hidden solving conventions.
    assert set.union(*(f['directions'] for f in train_features)) == set(range(4))
    assert any(f['balanced_exchange'] and f['all_changes_taught'] for f in train_features)
    assert any(f['color_swap'] and f['all_changes_taught'] for f in train_features)
    assert any('remove' in f['kinds'] and f['directions'] for f in train_features)
    assert any(f['silent'] >= 2 for f in train_features)
    assert any(f['omit_a_extra'] > 0 for f in train_features)
    assert all(1 <= len(f['kinds']) <= 2 for f in train_features)
    assert all(3 <= len(f['kinds']) <= 4 for f in test_features)
    for d in range(4):
        assert max(f['radius'] for f in train_features if d in f['directions']) >= max(
            f['radius'] for f in test_features)


def self_test():
    """Bounded 42-seed diagnostics, streamed in memory; returns compact metrics."""
    shapes, layouts, palettes, signatures = set(), set(), set(), set()
    counts, radii, master_masks, changed_masks, kinds = set(), set(), set(), set(), set()
    max_ray, zero_rays, pairs = 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair)
                digest.update(pair['input'].tobytes())
                digest.update(pair['output'].tobytes())
                shapes.add(f['shape'])
                layouts.add(f['layout'])
                palettes.add(f['background'])
                counts.add(f['count'])
                radii.add(f['radius'])
                master_masks.add(f['master_mask'])
                changed_masks.add(f['changed_mask'])
                kinds.update(f['kinds'])
                max_ray = max(max_ray, max(f['ray_lengths']))
                zero_rays += f['ray_lengths'].count(0)
                pairs += 1
        signatures.add(digest.digest())
    assert len(signatures) == 42 and len(layouts) >= 100 and len(shapes) >= 20
    assert counts == {3, 4, 5} and radii == {2, 3}
    assert len(master_masks) >= 12 and len(changed_masks) == 15
    assert kinds == {'add', 'remove', 'recolor'} and len(palettes) >= 30
    assert max_ray >= 12 and zero_rays > 0
    a, b = generate_puzzle(436463143, 2, 1), generate_puzzle(436463143, 2, 1)
    assert all(_key(x[k]) == _key(y[k]) for s in ('train', 'test')
               for x, y in zip(a[s], b[s]) for k in ('input', 'output'))
    # The rule has no privileged location, orientation, or literal color.
    for pair in a['train'] + a['test']:
        for rotation in range(4):
            for reflect in (False, True):
                inp, out = (np.rot90(pair[k], rotation) for k in ('input', 'output'))
                if reflect:
                    inp, out = inp[:, ::-1], out[:, ::-1]
                assert np.array_equal(solve(inp), out)
                lookup = np.array([7, 4, 9, 0, 8, 2, 5, 1, 6, 3], dtype=np.int8)
                assert np.array_equal(solve(lookup[inp]), lookup[out])
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            generate_puzzle(1000 + 10 * nt + ne, nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Label corruption, whole-cap beams, and missing-template evidence must fail.
    base = generate_puzzle(0, 2, 1)
    for defect in ('ray', 'interior', 'reference'):
        bad = {s: [{k: g.copy() for k, g in pair.items()} for pair in base[s]]
               for s in ('train', 'test')}
        pair = bad['train'][0]
        info = _recover(pair['input'])
        if defect == 'ray':
            p = next(iter(info['rays']))
            pair['output'][p] = pair['input'][p]
        elif defect == 'interior':
            d, color, _ = info['changes'][0]
            path = _paths(info['odd']['center'], info['odd']['radius'], d,
                          pair['input'].shape, all_lanes=True)[1]
            p = next(p for p in path if pair['input'][p] == info['base'][p]
                     and p not in info['rays'])
            pair['output'][p] = color
        else:
            e = next(e for e in info['emitters'] if e is not info['odd'])
            d = next(d for d, c in enumerate(e['states']) if c is not None)
            for p in _cap(e['center'], e['radius'], d):
                pair['input'][p] = info['base'][p]
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError('validator accepted ' + defect)
    return {'seeds': 42, 'pairs': pairs, 'distinct_puzzles': len(signatures),
            'shapes': len(shapes), 'layouts_without_colors': len(layouts),
            'ordered_backgrounds': len(palettes), 'emitter_counts': sorted(counts),
            'radii': sorted(radii), 'template_masks': len(master_masks),
            'changed_side_masks': len(changed_masks), 'max_ray_length': max_ray,
            'zero_length_endpoint_rays': zero_rays}
