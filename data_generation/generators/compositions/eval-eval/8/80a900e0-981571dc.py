"""Selective mirror repair reached by occluded diamond-cap endpoint probes.

Only standard-library modules and NumPy are used. No parent files are loaded.
"""
from __future__ import annotations

from functools import lru_cache
import copy
import numpy as np

GAP, STRUCT, CAP = 0, 8, 9
DIRECTIONS = ((-1, -1), (-1, 1), (1, 1), (1, -1))


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError('expected an integer ARC grid with axes 1..30 and colors 0..9')
    return g.astype(np.int8, copy=True)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _outline(center, radius):
    r, c = center
    return tuple((r + dr, c + dc)
                 for dr in range(-radius, radius + 1)
                 for dc in range(-radius, radius + 1)
                 if abs(dr) + abs(dc) == radius)


def _cap(center, radius, direction):
    r, c = center
    dr, dc = direction
    return tuple((r + dr * (k + 1), c + dc * (radius - k + 1))
                 for k in range(radius + 1))


def _orbit(p, shape, axes):
    r, c = p
    rr, cc = 2 * axes[0] - r, 2 * axes[1] - c
    rows = {r, rr} if 0 <= rr < shape[0] else {r}
    cols = {c, cc} if 0 <= cc < shape[1] else {c}
    return tuple(sorted((y, x) for y in rows for x in cols))


@lru_cache(maxsize=256)
def _orbits(shape, axes):
    seen, result = set(), []
    for r in range(shape[0]):
        for c in range(shape[1]):
            if (r, c) not in seen:
                o = _orbit((r, c), shape, axes)
                seen.update(o)
                result.append(o)
    return tuple(result)


def _analyze(g):
    rows = np.flatnonzero(np.all(g == STRUCT, axis=1))
    cols = np.flatnonzero(np.all(g == STRUCT, axis=0))
    assert len(rows) == len(cols) == 1, 'one full row and column must mark the folds'
    axes = int(rows[0]), int(cols[0])
    seam = np.zeros(g.shape, bool)
    seam[axes[0], :] = seam[:, axes[1]] = True
    emitters, covered = [], set(map(tuple, np.argwhere(seam)))
    h, w = g.shape
    for radius in (2, 3):
        if min(h, w) < 2 * radius + 1:
            continue
        sh = h - 2 * radius, w - 2 * radius
        mask = np.ones(sh, bool)
        for dr, dc in _outline((0, 0), radius):
            mask &= g[radius + dr:radius + dr + sh[0],
                      radius + dc:radius + dc + sh[1]] == STRUCT
        for y, x in np.argwhere(mask):
            center = int(y) + radius, int(x) + radius
            points = set(_outline(center, radius))
            if any(seam[p] for p in points):
                continue
            assert not points & covered, 'overlapping or ambiguous outlines'
            covered.update(points)
            emitters.append({'center': center, 'radius': radius, 'caps': []})
    assert 1 <= len(emitters) <= 4, 'expected one through four complete diamonds'
    for e in emitters:
        for d in DIRECTIONS:
            points = _cap(e['center'], e['radius'], d)
            if all(_inside(p, g.shape) and g[p] == CAP for p in points):
                assert not set(points) & covered, 'overlapping caps'
                covered.update(points)
                e['caps'].append(d)
        assert e['caps'], 'each diamond must have an activating cap'
    assert covered == set(map(tuple, np.argwhere((g == STRUCT) | (g == CAP)))), \
        'unexplained structural or cap marker'
    emitters.sort(key=lambda e: e['center'])
    return axes, emitters


def _lane(endpoint, direction, g, gaps_only):
    """Use the ORIGINAL input for every stopping decision."""
    dr, dc = direction
    p = endpoint[0] + dr, endpoint[1] + dc
    result = []
    while _inside(p, g.shape):
        if (g[p] != GAP if gaps_only else g[p] in (STRUCT, CAP)):
            break
        result.append(p)
        p = p[0] + dr, p[1] + dc
    return tuple(result)


def _targets(g, emitters):
    result = set()
    for e in emitters:
        for d in e['caps']:
            cap = _cap(e['center'], e['radius'], d)
            for endpoint in (cap[0], cap[-1]):
                result.update(_lane(endpoint, d, g, True))
    return result


def _witness(g, point, axes):
    colors = {int(g[p]) for p in _orbit(point, g.shape, axes) if 1 <= g[p] <= 7}
    assert len(colors) == 1, 'missing or contradictory reflection witness'
    return colors.pop()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive all geometry, stopping positions and values from the input alone."""
    g = _grid(input_grid)
    axes, emitters = _analyze(g)
    result = g.copy()
    for p in _targets(g, emitters):
        result[p] = _witness(g, p, axes)
    return result


def _footprint(e):
    result = set(_outline(e['center'], e['radius']))
    for d in e['caps']:
        result.update(_cap(e['center'], e['radius'], d))
    return result


def _markers(shape, axes, emitters):
    g = np.ones(shape, dtype=np.int8)
    g[axes[0], :] = g[:, axes[1]] = STRUCT
    for e in emitters:
        for p in _outline(e['center'], e['radius']):
            g[p] = STRUCT
        for d in e['caps']:
            for p in _cap(e['center'], e['radius'], d):
                g[p] = CAP
    return g


def _layout(rng, shape, axes, maximum, count, teaching):
    specs, occupied = [], set()
    for i in range(count):
        radius = maximum if i == 0 and teaching is not None else int(rng.integers(2, maximum + 1))
        caps = list(teaching) if i == 0 and teaching is not None else []
        # Sample an independent nonempty subset; no cap-color direction code.
        if not (i == 0 and teaching is not None):
            mask = int(rng.integers(1, 16))
            caps = [d for j, d in enumerate(DIRECTIONS) if mask & (1 << j)]
        positions = [(r, c) for r in range(radius + 1, shape[0] - radius - 1)
                     for c in range(radius + 1, shape[1] - radius - 1)]
        rng.shuffle(positions)
        for center in positions:
            e = {'center': center, 'radius': radius, 'caps': caps}
            fp = _footprint(e)
            if (any(not _inside(p, shape) or p[0] == axes[0] or p[1] == axes[1] for p in fp)
                    or fp & occupied):
                continue
            if any(sum(abs(v - u) for v, u in zip(center, old['center']))
                   < radius + old['radius'] + 4 for old in specs):
                continue
            trial = _markers(shape, axes, specs + [e])
            if i == 0 and teaching is not None:
                paths = [_lane(p, d, trial, False) for d in caps
                         for p in (_cap(center, radius, d)[0], _cap(center, radius, d)[-1])]
                lengths = list(map(len, paths))
                if (min(lengths) < 2 or
                        not (sum(n >= 4 for n in lengths) >= 2 or max(lengths) >= 5)):
                    continue
                if any(len(_orbit(p, shape, axes)) < 2 for path in paths for p in path[:2]):
                    continue
            specs.append(e)
            occupied.update(fp)
            break
        else:
            return None
    return specs


def _field(rng, shape, axes, palette):
    style = int(rng.integers(5))
    bh, bw = int(rng.integers(2, 5)), int(rng.integers(2, 5))
    shift = int(rng.integers(len(palette)))
    coarse = rng.integers(len(palette), size=shape)
    origin = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
    g = np.empty(shape, np.int8)
    for o in _orbits(shape, axes):
        r, c = o[0]
        if style == 0:
            v = int(rng.integers(len(palette)))
        elif style == 1:
            v = int(coarse[r // bh, c // bw])
        elif style == 2:
            v = (r // bh + c // bw // 2 + shift) % len(palette)
        elif style == 3:
            v = ((r + c) // bw + shift) % len(palette)
        else:
            v = (max(abs(r - origin[0]), abs(c - origin[1])) // bh + shift) % len(palette)
        if rng.random() < .1:
            v = int(rng.integers(len(palette)))
        for p in o:
            g[p] = palette[v]
    return g


def _add_masks(rng, mask, usable, forbidden):
    shape = mask.shape
    if rng.random() < .5:
        proposed = rng.random(shape) < rng.uniform(.025, .12)
    else:
        proposed = np.zeros(shape, bool)
        for _ in range(int(rng.integers(1, 5))):
            r, c = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
            h, w = int(rng.integers(1, 5)), int(rng.integers(1, 6))
            if rng.random() < .45:
                for dr in range(h):
                    proposed[r + dr:min(r + dr + 1, shape[0]), c:min(c + dr + 1, c + w, shape[1])] = True
            else:
                proposed[r:min(r + h, shape[0]), c:min(c + w, shape[1])] = True
    mask |= proposed & usable & ~forbidden


def _teaching_features(inp, out):
    axes, emitters = _analyze(inp)
    targets = _targets(inp, emitters)
    evidence, colorful, behind, long = set(), False, False, False
    for e in emitters:
        for d in e['caps']:
            cap = _cap(e['center'], e['radius'], d)
            paths = [_lane(p, d, inp, True) for p in (cap[0], cap[-1])]
            probes = [(p[0] + d[0], p[1] + d[1]) for p in cap[1:-1]]
            if (min(map(len, paths)) >= 2 and any(
                    _inside(p, inp.shape) and inp[p] == out[p] == GAP for p in probes)):
                evidence.add((d, e['radius']))
            colorful |= any(len(path) >= 2 and out[path[0]] != out[path[1]] for path in paths)
            long |= any(len(path) >= 3 for path in paths)
            for endpoint, actual in zip((cap[0], cap[-1]), paths):
                geom = _lane(endpoint, d, inp, False)
                n = len(actual)
                if n < len(geom) and 1 <= inp[geom[n]] <= 7:
                    behind |= any(inp[p] == out[p] == GAP for p in geom[n + 1:])
    diagonal_only = []
    for r, c in sorted(targets):
        rr, cc = 2 * axes[0] - r, 2 * axes[1] - c
        if (0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1]
                and rr != r and cc != c and inp[rr, c] == inp[r, cc] == GAP
                and 1 <= inp[rr, cc] <= 7):
            diagonal_only.append((r, c))
    return {'evidence': evidence, 'colorful': colorful, 'behind': behind, 'long': long,
            'diagonal_only': diagonal_only, 'untouched': int(np.count_nonzero(out == GAP)),
            'targets': targets, 'axes': axes, 'emitters': emitters}


def _make_pair(rng, maximum, palette_pool, count, teaching=None, hard=False, test=False):
    """Bounded free placement and mask construction, with no hidden solve state."""
    for attempt in range(160):
        if teaching is not None:
            low = 25 if maximum == 2 else 29
        else:
            low = 17 if count == 1 else (23 if count <= 3 else 27)
        shape = int(rng.integers(low, 31)), int(rng.integers(low, 31))
        axes = tuple(int(np.clip(n // 2 + int(rng.integers(-2, 3)), 7, n - 8)) for n in shape)
        specs = _layout(rng, shape, axes, maximum, count, teaching)
        if specs is None:
            continue
        k = int(rng.integers(2, len(palette_pool) + 1))
        palette = list(map(int, rng.choice(palette_pool, k, replace=False)))
        base = _field(rng, shape, axes, palette)
        marker = _markers(shape, axes, specs)
        usable = marker == 1
        mask = np.zeros(shape, bool)
        mandatory = np.zeros(shape, bool)
        forbidden = ~usable
        records = []
        long_token = behind_token = None
        if teaching is not None:
            e = specs[0]
            options = [(d, endpoint, _lane(endpoint, d, marker, False))
                       for d in e['caps'] for endpoint in
                       (_cap(e['center'], e['radius'], d)[0],
                        _cap(e['center'], e['radius'], d)[-1])]
            # Distinguish a fixed two-step shortcut, while retaining a visible
            # intact stopper with a missing cell behind it.
            choices = [o for o in options if len(o[2]) >= 4]
            rng.shuffle(choices)
            long_choice = choices[0] if len(choices) > 1 else next(o for o in choices if len(o[2]) >= 5)
            other = [o for o in choices if o[:2] != long_choice[:2]]
            behind_choice = other[0] if other else long_choice
            long_token, behind_token = long_choice[:2], behind_choice[:2]
        for i, e in enumerate(specs):
            for d in e['caps']:
                cap = _cap(e['center'], e['radius'], d)
                for endpoint in (cap[0], cap[-1]):
                    lane = _lane(endpoint, d, marker, False)
                    if not lane:
                        continue
                    teach = teaching is not None and i == 0
                    if teach:
                        length = int(rng.integers(2, min(5, len(lane)) + 1))
                        if (d, endpoint) == long_token:
                            length = 3
                        if (d, endpoint) == behind_token:
                            length = 3 if behind_token == long_token else 2
                    else:
                        upper = min(len(lane), 7 if test else 5)
                        length = int(rng.integers(0, upper + 1))
                    for p in lane[:length]:
                        mask[p] = True
                        if teach:
                            mandatory[p] = True
                    if length < len(lane):
                        forbidden[lane[length]] = True
                        if length + 1 < len(lane):
                            mask[lane[length + 1]] = True
                            if teach:
                                mandatory[lane[length + 1]] = True
                    records.append((i, d, lane, length))
                # Interior lanes are visible negative evidence, not rays.
                for p in cap[1:-1]:
                    q = p[0] + d[0], p[1] + d[1]
                    if _inside(q, shape) and usable[q]:
                        mask[q] = True
                        if teaching is not None and i == 0:
                            mandatory[q] = True
        # Crossing paths must not erase another path's reserved intact stopper.
        if np.any(mandatory & forbidden):
            continue
        mask &= ~forbidden
        _add_masks(rng, mask, usable, forbidden)
        # Guarantee a multicolor endpoint trace in each teaching witness.
        if teaching is not None:
            lane = next(rec[2] for rec in records if rec[0] == 0 and len(rec[2]) >= 2)
            for p, color in zip(lane[:2], palette[:2]):
                for q in _orbit(p, shape, axes):
                    base[q] = color
        # Hard repairs keep the double reflection while erasing both direct ones.
        hard_count = 0
        if hard:
            candidates = [(p, _orbit(p, shape, axes)) for rec in records for p in rec[2][:rec[3]]]
            rng.shuffle(candidates)
            wanted = int(rng.integers(2, 5)) if test else 1
            used_orbits = set()
            for p, o in candidates:
                if len(o) != 4 or o in used_orbits or not all(usable[q] for q in o):
                    continue
                r, c = p
                survivor = 2 * axes[0] - r, 2 * axes[1] - c
                erased = [q for q in o if q != survivor]
                if mandatory[survivor] or any(forbidden[q] for q in erased):
                    continue
                for q in erased:
                    mask[q] = mandatory[q] = True
                mask[survivor] = False
                forbidden[survivor] = True
                used_orbits.add(o)
                hard_count += 1
                if hard_count >= wanted:
                    break
            if hard_count == 0:
                continue
        # Enforce recoverability everywhere, including gaps not selected by rays.
        possible = True
        for o in _orbits(shape, axes):
            points = [p for p in o if usable[p]]
            if points and all(mask[p] for p in points):
                choices = [p for p in points if not mandatory[p]]
                if not choices:
                    possible = False
                    break
                p = choices[int(rng.integers(len(choices)))]
                mask[p] = False
                forbidden[p] = True
        if not possible:
            continue
        inp = base.copy()
        inp[~usable] = marker[~usable]
        inp[mask] = GAP
        try:
            out = solve(inp)
            f = _teaching_features(inp, out)
        except AssertionError:
            continue
        if not f['targets'] or f['untouched'] == 0 or len(np.unique(inp[(inp >= 1) & (inp <= 7)])) < 2:
            continue
        if teaching is not None:
            expected = {(d, maximum) for d in teaching}
            if (not expected <= f['evidence'] or not f['colorful']
                    or not f['behind'] or not f['long']):
                continue
        if hard and not f['diagonal_only']:
            continue
        return {'input': inp, 'output': out}
    # Search failure is explicit: no silent substitution of an easier rule.
    raise RuntimeError('bounded placement/mask search exhausted')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if not low <= value <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return value


def _key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 'num_train', 2, 10)
    ns = _count(num_test, 'num_test', 1, 4)
    if nt <= ns:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    maximum = int(rng.integers(2, 4))
    held_out = int(rng.integers(1, 8)) if rng.random() < .45 else None
    training_pool = [c for c in range(1, 8) if c != held_out]
    teaching_slots = list(map(int, rng.choice(nt, 2, replace=False)))
    first_mask = int(rng.integers(1, 15))
    second_mask = (15 ^ first_mask) | int(rng.integers(0, 16))
    teaching = {teaching_slots[0]: [d for j, d in enumerate(DIRECTIONS) if first_mask & (1 << j)],
                teaching_slots[1]: [d for j, d in enumerate(DIRECTIONS) if second_mask & (1 << j)]}
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ns)):
        for index in range(count):
            sides = teaching.get(index) if split == 'train' else None
            n = 1 if sides is not None else int(rng.integers(2 if split == 'test' else 1, 5))
            hard = split == 'test' or index == teaching_slots[1]
            pool = list(range(1, 8)) if split == 'test' else training_pool
            for _ in range(12):
                pair = _make_pair(rng, maximum, pool, n, sides, hard, split == 'test')
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input search exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence, test_radii = set(), set(), set()
    colorful = behind = diagonal = long = False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen, 'duplicate input'
            seen.add(key)
            f = _teaching_features(inp, out)
            axes, emitters, targets = f['axes'], f['emitters'], f['targets']
            assert targets and f['untouched'] > 0, 'require repairs and untouched gaps'
            assert len(np.unique(inp[(inp >= 1) & (inp <= 7)])) >= 2
            for o in _orbits(inp.shape, axes):
                values = {int(inp[p]) for p in o if 1 <= inp[p] <= 7}
                assert len(values) <= 1, 'contradictory field orbit'
                if any(inp[p] == GAP for p in o):
                    assert len(values) == 1, 'unrecoverable erased orbit'
            changed = set(map(tuple, np.argwhere(inp != out)))
            assert changed == targets, 'changed cells must be exactly original-input endpoint prefixes'
            for p in targets:
                assert inp[p] == GAP and out[p] == _witness(inp, p, axes)
            assert np.array_equal(out[inp != GAP], inp[inp != GAP]), 'visible cells changed'
            assert np.array_equal(solve(inp), out)
            if split == 'train':
                evidence |= f['evidence']
                colorful |= f['colorful']
                long |= f['long']
                behind |= f['behind']
                diagonal |= bool(f['diagonal_only'])
            else:
                test_radii.update(e['radius'] for e in emitters)
    assert colorful and behind and diagonal and long, 'missing distinguishing training witness'
    for d in DIRECTIONS:
        assert any(direction == d and radius >= max(test_radii) for direction, radius in evidence), \
            'training must demonstrate all directions and tested cap sizes'


def _signature(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def _metrics(pair):
    inp, out = pair['input'], pair['output']
    f = _teaching_features(inp, out)
    lengths, zero, subsets = [], 0, []
    for e in f['emitters']:
        subsets.append(sum(1 << DIRECTIONS.index(d) for d in e['caps']))
        for d in e['caps']:
            cap = _cap(e['center'], e['radius'], d)
            for endpoint in (cap[0], cap[-1]):
                n = len(_lane(endpoint, d, inp, True))
                lengths.append(n)
                zero += n == 0
    return {'shape': inp.shape, 'axes': f['axes'], 'emitters': len(f['emitters']),
            'radii': tuple(e['radius'] for e in f['emitters']), 'subsets': tuple(subsets),
            'lengths': tuple(lengths), 'zero_endpoints': zero, 'repairs': len(f['targets']),
            'untouched': f['untouched'], 'diagonal_only': len(f['diagonal_only']),
            'palette': tuple(int(c) for c in np.unique(inp) if 1 <= c <= 7),
            'offset': tuple(2 * a - n + 1 for a, n in zip(f['axes'], inp.shape)),
            'layout': (inp.shape, f['axes'], tuple((e['center'], e['radius'], tuple(e['caps']))
                                                for e in f['emitters']))}


def _regression_tests():
    """Small explicit path expectations, including cases not forced by sampling."""
    def fixture(axes, emitters, gaps, targets):
        shape = (30, 30)
        base = np.empty(shape, np.int8)
        for o in _orbits(shape, axes):
            color = 1 + (2 * o[0][0] + 3 * o[0][1]) % 7
            for p in o:
                base[p] = color
        marker = _markers(shape, axes, emitters)
        inp = base.copy()
        inp[marker != 1] = marker[marker != 1]
        for p in gaps:
            assert marker[p] == 1
            inp[p] = GAP
        expected = inp.copy()
        for p in targets:
            expected[p] = base[p]
        actual = solve(inp)
        assert np.array_equal(actual, expected), 'handcrafted path regression failed'
        for p in gaps:
            assert _witness(inp, p, axes) == base[p]
        return inp, actual

    # Both direct mirrors are absent; a middle lane and a gap behind a stopper stay absent.
    targets = {(8, 10), (9, 11), (10, 8), (11, 9)}
    fixture((14, 14), [{'center': (6, 6), 'radius': 2, 'caps': [(1, 1)]}],
            targets | {(20, 10), (8, 18), (9, 9), (11, 13)}, targets)
    # Bottom fringe: no horizontal partner, but a vertical donor still repairs the tile.
    targets = {(27, 4), (28, 3), (29, 2), (29, 6)}
    inp, out = fixture((12, 12), [{'center': (25, 8), 'radius': 2, 'caps': [(1, -1)]}],
                       targets, targets)
    assert len(_orbit((29, 2), inp.shape, (12, 12))) == 2
    assert out[29, 29] == inp[29, 29]  # True singleton remains visible.
    # A complete cap touches the edge: one endpoint exits immediately.
    targets = {(1, 7), (0, 8)}
    fixture((12, 12), [{'center': (3, 3), 'radius': 2, 'caps': [(-1, 1)]}],
            targets | {(0, 6)}, targets)
    # Two rays meet at (8,9). Reading a progressively recolored grid would stop one early.
    targets = {(7, 8), (8, 9), (9, 10), (7, 10), (9, 8)}
    fixture((18, 18), [
        {'center': (5, 4), 'radius': 2, 'caps': [(1, 1)]},
        {'center': (5, 14), 'radius': 2, 'caps': [(1, -1)]}], targets | {(8, 7)}, targets)
    # Seams stop both lanes. An 8-covered direct donor is ignored rather than copied.
    targets = {(10, 12), (11, 13), (12, 10), (13, 11)}
    fixture((14, 14), [
        {'center': (8, 8), 'radius': 2, 'caps': [(1, 1)]},
        {'center': (10, 18), 'radius': 2, 'caps': [(-1, -1)]}],
        targets | {(18, 12), (11, 11)}, targets)
    # The rule has no sampler's length ceiling: a one-fold fringe supports a ten-cell lane.
    targets = {(12 + k, 7 + k) for k in range(8)} | {(14 + k, 5 + k) for k in range(10)}
    fixture((7, 15), [{'center': (10, 3), 'radius': 2, 'caps': [(1, 1)]}], targets, targets)


def self_test() -> None:
    _regression_tests()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert _signature(puzzles[0]) == _signature(generate_puzzle(0))
    assert len({_signature(p) for p in puzzles}) == 42
    features = [_metrics(pair) for p in puzzles for s in ('train', 'test') for pair in p[s]]
    assert {r for f in features for r in f['radii']} == {2, 3}
    assert {f['emitters'] for f in features} == {1, 2, 3, 4}
    assert len({f['shape'] for f in features}) >= 25
    assert len({f['layout'] for f in features}) >= 150
    assert {m for f in features for m in f['subsets']} == set(range(1, 16))
    assert len({f['palette'] for f in features}) >= 20
    assert any(f['zero_endpoints'] for f in features)
    assert any(max(f['lengths']) >= 5 for f in features)
    assert any(f['shape'][0] != f['shape'][1] for f in features)
    assert {int(np.sign(f['offset'][0])) for f in features} == {-1, 0, 1}
    assert {int(np.sign(f['offset'][1])) for f in features} == {-1, 0, 1}
    for p in puzzles:
        for pair in p['test']:
            assert _metrics(pair)['diagonal_only'] > 0
    # Explicit count support includes every legal combination, not just defaults.
    for nt in range(2, 11):
        for ns in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(100 + 10 * nt + ns, nt, ns)
            assert len(p['train']) == nt and len(p['test']) == ns
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    # Mutating a ray color, restoring an untouched gap, or altering a visible cell fails.
    for kind in ('ray', 'untouched', 'visible'):
        bad = copy.deepcopy(puzzles[0])
        pair = bad['test'][0]
        inp, out = pair['input'], pair['output']
        if kind == 'ray':
            p = tuple(np.argwhere(inp != out)[0])
            out[p] = 1 + int(out[p]) % 7
        elif kind == 'untouched':
            p = tuple(np.argwhere(out == GAP)[0])
            axes, _ = _analyze(inp)
            out[p] = _witness(inp, p, axes)
        else:
            p = tuple(np.argwhere((inp >= 1) & (inp <= 7))[0])
            out[p] = GAP
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f'validator accepted {kind} corruption')
