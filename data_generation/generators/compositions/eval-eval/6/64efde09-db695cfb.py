"""Shared stencils, local chords. All inference is from one input's pixels.

The two-cell antenna registers each island and names its endpoint color. Raw
colored dots form a shared stencil; different local chords gate its emissions.
No files, caches, parent imports, or generation state are used by solve().
"""
from __future__ import annotations

import hashlib
from itertools import combinations

import numpy as np


HOST_SEEDS = (0, 218935391, 583583414, 580811479, 871666771, 893729231,
              278873471, 370249748, 627938939, 605142531, 608056272, 900522457,
              145053753, 62864450, 280770125, 865577085, 670743129, 878057046,
              731226528, 34553292, 136134976, 233264868, 576228084, 32624865,
              639724609, 46782473, 468460190, 213327990, 586177145, 627837892,
              835989251, 828842417, 901014472, 150731365, 832277770, 206785013,
              141025119, 707666622, 381366877, 691750425, 454941561, 938473054)


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _undo(a, t):
    a = np.rot90(a, -t % 4)
    return np.fliplr(a).copy() if t >= 4 else a.copy()


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= x <= 30 for x in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in unseen:
                    unseen.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _line(domain, p, slope):
    """The maximal contiguous opposite diagonal through p inside the body."""
    result = {p}
    for sign in (-1, 1):
        r, c = p
        while True:
            r, c = r+sign, c+sign*slope
            if not (0 <= r < domain.shape[0] and 0 <= c < domain.shape[1]):
                break
            if not domain[r, c]:
                break
            result.add((r, c))
    return result


def _segment(a, b):
    assert a != b and abs(a[0]-b[0]) == abs(a[1]-b[1])
    n = abs(a[0]-b[0])
    dr, dc = (1 if b[0] > a[0] else -1), (1 if b[1] > a[1] else -1)
    return tuple((a[0]+i*dr, a[1]+i*dc) for i in range(n+1))


def _geometry(domain, primary_points, code):
    """Decode all unique endpoint pairs and validate their combined geometry."""
    points = sorted(primary_points)
    assert set(points).isdisjoint(code)
    edges, degree = [], dict.fromkeys(points, 0)
    for a, b in combinations(points, 2):
        if abs(a[0]-b[0]) == abs(a[1]-b[1]):
            cells = _segment(a, b)
            assert len(cells) >= 3 and all(domain[p] for p in cells)
            edges.append(cells)
            degree[a] += 1
            degree[b] += 1
    assert edges and max(degree.values()) <= 1, 'nonunique endpoint partners'
    union = set().union(*(set(s) for s in edges))
    active, painted, lines = {}, {}, []
    for p, color in code.items():
        owners = [s for s in edges if p in s]
        assert len(owners) <= 1, 'two-slope trigger'
        if not owners:
            continue
        s = owners[0]
        slope = 1 if s[-1][1] > s[0][1] else -1
        line = _line(domain, p, -slope)
        assert line & union == {p}, 'secondary line crosses another bridge'
        assert line.isdisjoint(primary_points), 'secondary line erases an endpoint'
        for q in line:
            assert q not in code or code[q] == color, 'secondary line erases a dot'
            assert q not in painted or painted[q] == color, 'different-color crossing'
            painted[q] = color
        active[p] = slope
        lines.append((p, color, line))
    # Every bridge has a genuinely missing primary cell, including short bridges.
    assert all(any(p not in primary_points and p not in code for p in s) for s in edges)
    return {'segments': edges, 'primary_union': union, 'active': active,
            'lines': lines, 'painted': painted,
            'unpaired': {p for p, d in degree.items() if d == 0}}


def _decode(value):
    a = _grid(value)
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    assert int(counts.max()) * 2 > a.size, 'background must occupy over half the input'
    islands, code = [], {}
    for cells in _components(a != bg):
        rs, cs = zip(*cells)
        r, c = min(rs), min(cs)
        crop = a[r:max(rs)+1, c:max(cs)+1].copy()
        matches = []
        for t in range(8):
            b = _orient(crop, t)
            if not (8 <= b.shape[0] <= 12 and 6 <= b.shape[1] <= 10):
                continue
            mask = b != bg
            if (np.count_nonzero(mask[0]) == np.count_nonzero(mask[1]) == 1
                    and mask[0, 0] and mask[1, 0] and np.all(mask[2])
                    and b[0, 0] == b[1, 0] and b[2, 0] != b[0, 0]):
                matches.append((t, b))
        assert len(matches) == 1, 'island registration is ambiguous'
        t, b = matches[0]
        primary, fill = int(b[0, 0]), int(b[2, 0])
        assert len({bg, primary, fill}) == 3
        domain = b[2:] != bg
        assert np.all(domain[0]) and np.all(np.sum(domain, axis=1) >= 2)
        primary_points = set(map(tuple, np.argwhere(b[2:] == primary).tolist()))
        local = {(int(y), int(x)): int(b[2+y, x]) for y, x in np.argwhere(domain)
                 if int(b[2+y, x]) not in (primary, fill)}
        for p, color in local.items():
            assert p not in code or code[p] == color, 'conflicting stencil observations'
            code[p] = color
        islands.append({'r': r, 'c': c, 'crop': crop, 't': t, 'canonical': b,
                        'domain': domain, 'primary': primary, 'fill': fill,
                        'points': primary_points, 'local': local})
    assert 2 <= len(islands) <= 4 and 1 <= len(code) <= 4
    reference = islands[0]
    for island in islands:
        assert np.array_equal(island['domain'], reference['domain'])
        assert (island['primary'], island['fill']) == (reference['primary'], reference['fill'])
        assert all(island['domain'][p] for p in code)
        island['geometry'] = _geometry(island['domain'], island['points'], code)
    return a, bg, islands, code


def _render(a, islands, code, *, transfer=True, chords=True):
    out = a.copy()
    for island in islands:
        b = island['canonical'].copy()
        stencil = code if transfer else island['local']
        if chords:
            geom = _geometry(island['domain'], island['points'], stencil)
            for p in geom['primary_union']:
                b[2+p[0], p[1]] = island['primary']
        for p, color in stencil.items():
            b[2+p[0], p[1]] = color
        if chords:
            for p, color in geom['painted'].items():
                b[2+p[0], p[1]] = color
        crop = _undo(b, island['t'])
        r, c = island['r'], island['c']
        out[r:r+crop.shape[0], c:c+crop.shape[1]] = crop
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Pool raw dots, complete local bridges, then emit locally clipped diagonals."""
    a, _, islands, code = _decode(input_grid)
    return _render(a, islands, code)


def _features(value):
    a, bg, islands, code = _decode(value)
    slopes = set()
    source_only_inactive, active_to_inactive, missing_active = False, False, 0
    multi = False
    zero = 0
    for p in code:
        sources = [i for i in islands if p in i['local']]
        if (all(p not in i['geometry']['active'] for i in sources)
                and any(p in i['geometry']['active'] for i in islands)):
            source_only_inactive = True
        if (any(p in i['geometry']['active'] for i in sources)
                and any(p not in i['local'] and p not in i['geometry']['active'] for i in islands)):
            active_to_inactive = True
    for i in islands:
        g = i['geometry']
        slopes.update(g['active'].values())
        missing_active += len(set(g['active']) - set(i['local']))
        for s in g['segments']:
            n = sum(p in code for p in s)
            multi |= n >= 2
            zero += n == 0
    return {'shape': tuple(a.shape), 'body': tuple(islands[0]['domain'].shape),
            'islands': len(islands), 'code_count': len(code),
            'code_colors': len(set(code.values())), 'slopes': tuple(sorted(slopes)),
            'segments': tuple(len(i['geometry']['segments']) for i in islands),
            'lengths': tuple(len(s) for i in islands for s in i['geometry']['segments']),
            'observations': tuple(len(i['local']) for i in islands),
            'unpaired': sum(len(i['geometry']['unpaired']) for i in islands),
            'zero': zero, 'multi': multi, 'missing_active': missing_active,
            'inactive_source_activates_elsewhere': source_only_inactive,
            'active_source_inactive_elsewhere': active_to_inactive,
            'transforms': tuple(i['t'] for i in islands),
            'palette': (bg, islands[0]['fill'], islands[0]['primary'], tuple(sorted(set(code.values())))),
            'cut_cells': int(np.sum(~islands[0]['domain']))}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    used, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, output = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == output.shape and _key(a) not in used
            used.add(_key(a))
            a, bg, islands, code = _decode(a)
            expected = _render(a, islands, code)
            assert np.array_equal(expected, output), 'incorrect shared-stencil construction'
            markers = (a != bg) & (a != islands[0]['fill'])
            assert np.array_equal(output[markers], a[markers]), 'original markers must survive'
            assert np.array_equal(output[a == bg], a[a == bg]), 'outside the islands must survive'
            # Primary markers survive; fill cells are allowed to change.
            assert not np.array_equal(a, output)
            assert not np.array_equal(_render(a, islands, code, transfer=False), output), 'A has no effect'
            assert not np.array_equal(_render(a, islands, code, chords=False), output), 'B has no effect'
            if split == 'train':
                training.append(_features(a))
    assert any(f['inactive_source_activates_elsewhere'] for f in training)
    assert any(f['active_source_inactive_elsewhere'] for f in training)
    assert any(f['zero'] for f in training)
    assert set().union(*(set(f['slopes']) for f in training)) == {-1, 1}
    assert any(len({t >= 4 for t in f['transforms']}) == 2 for f in training)
    assert any(len({t % 2 for t in f['transforms']}) == 2 for f in training)


def _domain(rng, h, w):
    domain = np.ones((h, w), dtype=bool)
    for side in (0, 1):
        depth = int(rng.integers(3))
        for k in range(depth):
            if side == 0:
                domain[h-1-k, :depth-k] = False
            else:
                domain[h-1-k, w-depth+k:] = False
    if rng.random() < .45:
        row = int(rng.integers(2, h-2))
        domain[row, -int(rng.integers(1, 3)):] = False
    return domain


def _candidates(domain, code):
    h, w = domain.shape
    candidates = []
    for r, c in np.argwhere(domain):
        a = int(r), int(c)
        if a == (0, 0) or a in code:
            continue
        for slope in (-1, 1):
            for steps in range(2, min(h, w)):
                b = a[0]+steps, a[1]+slope*steps
                if not (0 <= b[0] < h and 0 <= b[1] < w):
                    break
                if b in code or not domain[b]:
                    continue
                points = {a, b}
                try:
                    geom = _geometry(domain, points, code)
                except AssertionError:
                    continue
                candidates.append((points, geom))
    return candidates


def _choose_geometry(rng, domain, code, candidates, target=None, slope=None,
                     need_zero=False, max_segments=3):
    options = [x for x in candidates if (target is None or set(x[1]['active']) == {target})
               and (slope is None or target in x[1]['active'] and x[1]['active'][target] == slope)]
    rng.shuffle(options)
    for points, geom in options[:100]:
        points = set(points)
        desired = int(rng.integers(1, max_segments+1))
        desired = max(desired, 2 if need_zero else 1)
        for index in rng.permutation(len(candidates)):
            if len(geom['segments']) >= desired and (not need_zero or any(
                    not set(s).intersection(code) for s in geom['segments'])):
                break
            extra, g = candidates[int(index)]
            if points & extra:
                continue
            if need_zero and not any(not set(s).intersection(code) for s in geom['segments']):
                if g['active']:
                    continue
            try:
                combined = _geometry(domain, points | extra, code)
            except AssertionError:
                continue
            if len(combined['segments']) != len(geom['segments'])+1:
                continue
            if target is not None and set(combined['active']) != {target}:
                continue
            points, geom = points | extra, combined
        if need_zero and not any(not set(s).intersection(code) for s in geom['segments']):
            continue
        # Optional isolated primary dot, with no new diagonal partner.
        if rng.random() < .35:
            for index in rng.permutation(domain.size):
                p = int(index)//domain.shape[1], int(index)%domain.shape[1]
                if not domain[p] or p == (0, 0) or p in code or p in geom['primary_union'] or p in geom['painted']:
                    continue
                if any(abs(p[0]-q[0]) == abs(p[1]-q[1]) for q in points):
                    continue
                points.add(p)
                break
        return points
    return None


def _place(rng, crops, occupied_count):
    # No arrays or attempts are retained. At most 24*4*150 small placements.
    for _ in range(24):
        min_area = 2*occupied_count+1
        h, w = map(int, rng.integers(18 if len(crops) == 2 else 25, 31, size=2))
        if h*w < min_area:
            h = w = 30
        used = np.zeros((h, w), dtype=bool)
        positions = []
        for crop in crops:
            ch, cw = crop.shape
            for _ in range(150):
                r, c = int(rng.integers(h-ch+1)), int(rng.integers(w-cw+1))
                if np.any(used[max(0, r-1):min(h, r+ch+1), max(0, c-1):min(w, c+cw+1)]):
                    continue
                positions.append((r, c))
                used[r:r+ch, c:c+cw] = True
                break
            else:
                break
        if len(positions) == len(crops):
            return (h, w), positions
    # A bounded packing fallback only affects placement, not the sampled rule.
    return (30, 30), [(1+15*(i//2), 1+15*(i % 2)) for i in range(len(crops))]


def _assemble(rng, domain, code, points, witness, palette):
    bg, fill, primary = palette[:3]
    count = len(points)
    base = int(rng.integers(8))
    ts = list(map(int, rng.integers(8, size=count)))
    if witness == 'active':
        ts[:2] = [base, base ^ 4]
    elif witness == 'inactive':
        ts[:2] = [base, (base//4)*4+(base+1) % 4]
    geoms = [_geometry(domain, p, code) for p in points]
    observed = [dict() for _ in points]
    slots = list(code)
    if witness:
        # Targets 0 and 1 are active only in islands 0 and 1, respectively.
        for j, p in enumerate(slots):
            index = j if witness == 'active' else 1-j
            observed[index][p] = code[p]
    else:
        mode = int(rng.integers(3))
        full = int(rng.integers(count))
        for p, color in code.items():
            index = full if mode == 0 else int(rng.integers(count))
            observed[index][p] = color
            if mode == 2 and rng.random() < .4:
                observed[int(rng.integers(count))][p] = color
        # At least one missing observation keeps transfer substantive.
        if all(len(o) == len(code) for o in observed):
            observed[-1].pop(slots[0])
    crops, expected_crops = [], []
    for j in range(count):
        b = np.full((domain.shape[0]+2, domain.shape[1]), bg, dtype=np.int8)
        b[2:][domain] = fill
        b[:2, 0] = primary
        for r, c in points[j]:
            b[2+r, c] = primary
        for (r, c), color in observed[j].items():
            b[2+r, c] = color
        expected = b.copy()
        for r, c in geoms[j]['primary_union']:
            expected[2+r, c] = primary
        for (r, c), color in code.items():
            expected[2+r, c] = color
        for (r, c), color in geoms[j]['painted'].items():
            expected[2+r, c] = color
        crops.append(_orient(b, ts[j]))
        expected_crops.append(_orient(expected, ts[j]))
    shape, positions = _place(rng, crops, count*(int(np.sum(domain))+2))
    a, out = (np.full(shape, bg, dtype=np.int8) for _ in range(2))
    for b, expected, (r, c) in zip(crops, expected_crops, positions):
        h, w = b.shape
        a[r:r+h, c:c+w] = b
        out[r:r+h, c:c+w] = expected
    return {'input': a, 'output': out}


def _fallback(rng, witness, count, palette):
    domain = np.ones((9, 9), dtype=bool)
    p, q = (2, 2), (5, 2)
    code = {p: palette[3], q: palette[4]}
    first = {(1, 1), (4, 4), (4, 6), (6, 8)}
    second = {(3, 4), (7, 0)}
    points = [first, second] + [first.copy() if j % 2 == 0 else second.copy() for j in range(count-2)]
    return _assemble(rng, domain, code, points, witness, palette)


def _pair(rng, witness, count, code_count, max_segments):
    palette = list(map(int, rng.permutation(10)))
    all_triggered = not witness and rng.random() < .25
    for _ in range(64):
        h, w = map(int, rng.integers(7 if witness else 6, 11, size=2))
        domain = _domain(rng, h, w)
        available = [(r, c) for r in range(1, h-1) for c in range(1, w-1) if domain[r, c]]
        chosen = [available[int(j)] for j in rng.choice(len(available), code_count, replace=False)]
        if witness and abs(chosen[0][0]-chosen[1][0]) == abs(chosen[0][1]-chosen[1][1]):
            continue
        values = palette[3:3+code_count]
        if not witness and code_count > 1 and rng.random() < .3:
            values = list(map(int, rng.choice(values, code_count)))
        code = dict(zip(chosen, values))
        candidates = _candidates(domain, code)
        points = []
        for j in range(count):
            target = (chosen[j] if witness and j < 2 else
                      chosen[int(rng.integers(len(chosen)))] if all_triggered else None)
            slope = (1 if j == 0 else -1) if witness and j < 2 else None
            result = _choose_geometry(rng, domain, code, candidates, target, slope,
                                      need_zero=bool(witness and j == 0),
                                      max_segments=1 if all_triggered else max_segments)
            if result is None:
                break
            points.append(result)
        if len(points) == count:
            return _assemble(rng, domain, code, points, witness, palette)
    return _fallback(rng, witness, count, palette)


def _count(value, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError('example counts must be integers')
    if not lo <= int(value) <= hi:
        raise ValueError('example count outside supported range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, 2, 10), _count(num_test, 1, 4)
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    profile = int(rng.integers(3))
    puzzle, used = {'train': [], 'test': []}, set()
    witnesses = ['active', 'inactive'] + [None]*(nt-2)
    rng.shuffle(witnesses)
    for split, n in (('train', nt), ('test', ne)):
        for index in range(n):
            witness = witnesses[index] if split == 'train' else None
            count = int(rng.integers(2, 4)) if split == 'train' else (
                4 if profile == 0 else int(rng.integers(2 if profile == 1 else 3, 5)))
            code_count = 2 if witness else (int(rng.integers(1, 4)) if split == 'train'
                                            else 4 if profile else int(rng.integers(1, 4)))
            for _ in range(8):
                pair = _pair(rng, witness, count, code_count, 2 if witness else 3)
                key = _key(pair['input'])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded distinct-input budget exhausted')
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(a.shape))
                digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streaming puzzles; bounded semantic, diversity and mutation checks."""
    fingerprints, shapes, palettes, silhouettes = set(), set(), set(), set()
    code_counts, receivers, lengths, segments, slopes = set(), set(), set(), set(), set()
    flags = {'multi': 0, 'zero': 0, 'cut': 0, 'unpaired': 0, 'missing_active': 0,
             'all_bridges_triggered': 0}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        fingerprints.add(_fingerprint(puzzle))
        for pair in puzzle['train']+puzzle['test']:
            a, _, islands, _ = _decode(pair['input'])
            f = _features(a)
            shapes.add(f['shape'])
            palettes.add(f['palette'])
            silhouettes.add(_key(islands[0]['domain']))
            code_counts.add(f['code_count'])
            receivers.add(f['islands'])
            lengths.update(f['lengths'])
            segments.update(f['segments'])
            slopes.update(f['slopes'])
            for flag in flags:
                value = f['cut_cells'] if flag == 'cut' else not f['zero'] if flag == 'all_bridges_triggered' else f[flag]
                flags[flag] += bool(value)
            # Coordinate equivariance and palette independence on a fresh view.
            lut = np.roll(np.arange(10, dtype=np.int8), (seed % 9)+1)
            t = seed % 8
            assert np.array_equal(solve(_orient(lut[a], t)), _orient(lut[pair['output']], t))
    assert len(fingerprints) == 42 and len(shapes) >= 35
    assert len(palettes) >= 100 and len(silhouettes) >= 40
    assert code_counts == {1, 2, 3, 4} and receivers == {2, 3, 4}
    assert {3, 4, 5, 6, 7, 8} <= lengths and {1, 2, 3} <= segments
    assert slopes == {-1, 1} and all(flags.values()), flags
    puzzle = generate_puzzle(0)
    assert _fingerprint(puzzle) == _fingerprint(generate_puzzle(0))
    # All finite emergency cases preserve the same two teaching witnesses.
    rng = np.random.default_rng(903)
    fallback = {'train': [_fallback(rng, w, 2, list(range(10))) for w in ('active', 'inactive')],
                'test': [_fallback(rng, None, 4, list(range(10)))]}
    validate_generated_puzzle(fallback)
    # A side notch cuts a diagonal that would reenter the body farther along.
    domain = np.ones((10, 10), dtype=bool)
    domain[3, 8:] = False
    assert _line(domain, (5, 6), -1) == {(4, 7), (5, 6), (6, 5), (7, 4), (8, 3), (9, 2)}
    clipped = _assemble(rng, domain, {(5, 6): 4},
                        [{(3, 4), (7, 8)}, {(1, 2), (3, 4)}], None, list(range(10)))
    assert np.array_equal(solve(clipped['input']), clipped['output'])
    _, _, registered, _ = _decode(clipped['input'])
    for island in registered:
        r, c = island['r'], island['c']
        h, w = island['crop'].shape
        body = _orient(clipped['output'][r:r+h, c:c+w], island['t'])[2:]
        assert body[5, 6] == 4
        if (7, 8) in island['points']:
            assert body[4, 7] == 4 and body[2, 9] == 1
        else:
            # Same infinite primary diagonal, but the dot is beyond its endpoint.
            assert not island['geometry']['active'] and body[4, 7] == 1
    # Two distinct colors on one primary bridge create parallel full lines.
    domain = np.ones((9, 9), dtype=bool)
    g = _geometry(domain, {(1, 1), (7, 7)}, {(3, 3): 4, (5, 5): 6})
    assert len(g['lines']) == 2 and g['painted'][(2, 4)] == 4 and g['painted'][(4, 6)] == 6
    crossing = {(1, 1), (7, 7), (1, 7), (7, 1)}
    _geometry(domain, crossing, {(2, 5): 4})
    try:
        _geometry(domain, crossing, {(4, 4): 4})
    except AssertionError:
        pass
    else:
        raise AssertionError('a two-slope trigger was accepted')
    # Same stencil position cannot carry conflicting observations in two copies.
    a, _, registered, _ = _decode(fallback['train'][0]['input'])
    for j, island in enumerate(registered):
        b = island['canonical'].copy()
        b[4, 2] = 3+j
        crop = _undo(b, island['t'])
        r, c = island['r'], island['c']
        a[r:r+crop.shape[0], c:c+crop.shape[1]] = crop
    try:
        solve(a)
    except AssertionError:
        pass
    else:
        raise AssertionError('conflicting stencil observations were accepted')
    bad = {'train': puzzle['train'], 'test': [{'input': puzzle['test'][0]['input'].copy(),
                                               'output': puzzle['test'][0]['output'].copy()}]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output was accepted')
    # Conflicting observations and nonunique endpoint pairings are invalid.
    domain = np.ones((8, 8), dtype=bool)
    try:
        _geometry(domain, {(1, 1), (3, 3), (5, 5)}, {(2, 5): 4})
    except AssertionError:
        pass
    else:
        raise AssertionError('ambiguous endpoint pairing was accepted')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
