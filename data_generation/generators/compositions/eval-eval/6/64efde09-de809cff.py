"""Defect sonar: pool reconstructed holes, reflect them, emit material echoes.

All solving conventions are input-readable. No parent module, stored example,
randomness, output label, or puzzle context is used by solve().
"""
from collections import Counter

import numpy as np


def _orient(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any(a < 0) or np.any(a > 9)):
        raise ValueError('Expected an integer ARC grid with axes 1..30')
    return a.astype(np.int8, copy=True)


def _key_mask(h, w):
    a = np.zeros((h, w), dtype=bool)
    a[h-1, :2] = True
    a[h-2, 0] = True
    return a


def _components(a, bg):
    seen = np.zeros(a.shape, dtype=bool)
    result = []
    for rr, cc in np.argwhere(a != bg):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        seen[r, c] = True
        stack, points = [(r, c)], []
        while stack:
            p = stack.pop()
            points.append(p)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                q = (p[0]+dr, p[1]+dc)
                if (0 <= q[0] < a.shape[0] and 0 <= q[1] < a.shape[1]
                        and not seen[q] and a[q] != bg):
                    seen[q] = True
                    stack.append(q)
        result.append(points)
    return result


def _point(tile, r, c):
    o, u, v = tile['origin'], tile['u'], tile['v']
    return (o[0]+r*u[0]+c*v[0], o[1]+r*u[1]+c*v[1])


def _make_tile(clean, t, r, c):
    labels = _orient(np.arange(clean.size).reshape(clean.shape), t)
    positions = {}
    for rr, cc in np.ndindex(labels.shape):
        positions[int(labels[rr, cc])] = (r+rr, c+cc)
    origin = positions[0]
    p, q = positions[clean.shape[1]], positions[1]
    return {'origin': origin, 'u': (p[0]-origin[0], p[1]-origin[1]),
            'v': (q[0]-origin[0], q[1]-origin[1]),
            'cells': set(positions.values()), 'transform': t}


def _fit_rectangle(observations, bg, highlight):
    """Exhaustive fit of the family geometry, scored on observed pixels only."""
    h, w = observations[0].shape
    km = _key_mask(h, w)
    # The intact back edge identifies the base material, regardless of palette.
    bases = [set(map(int, a[-1, 2:])) for a in observations]
    assert all(len(s) == 1 and s == bases[0] for s in bases)
    base = next(iter(bases[0]))
    colors = set(map(int, np.unique(np.stack(observations)))) - {bg, highlight}
    assert len(colors) == 2 and base in colors
    overlay = next(c for c in colors if c != base)
    candidates, boxes = [], []
    for r0 in range(h-2):
        for r1 in range(r0+2, h):
            for c0 in range(1, w-2):
                for c1 in range(c0+3, w+1):
                    area = (r1-r0)*(c1-c0)
                    if not 8 <= area <= .60*h*w:
                        continue
                    clean = np.full((h, w), base, dtype=np.int8)
                    clean[r0:r1, c0:c1] = overlay
                    clean[km] = highlight
                    candidates.append(clean)
                    boxes.append((r0, c0, r1, c1))
    assert candidates
    stack, obs = np.stack(candidates), np.stack(observations)
    losses = np.sum((stack[:, None] != obs[None]) & (obs[None] != bg),
                    axis=(1, 2, 3))
    order = np.argsort(losses, kind='stable')
    best = int(order[0])
    assert len(order) == 1 or int(losses[order[1]]) >= int(losses[best])+2, 'Ambiguous laminate'
    clean = stack[best]
    border = np.zeros(clean.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    for a in observations:
        assert np.array_equal(a[border], clean[border]), 'Damaged outer contour'
        for color in colors:
            mask = clean == color
            assert np.count_nonzero(a[mask] == color) >= .65*np.count_nonzero(mask)
        assert np.count_nonzero(a != clean) <= max(4, .20*clean.size)
    return clean, (base, overlay), boxes[best], int(losses[best])


def _local_effects(clean, code, fills, same_fill=False):
    """Beams, rings and centers in the canonical tile's unbounded plane."""
    centers = {(-r-1, c) for r, c in code}
    beams, rings = {}, {}

    def demand(mapping, p, color):
        if p in centers:
            return
        assert p not in mapping or mapping[p] == color, 'Contradictory echoes'
        mapping[p] = color

    for r, c in sorted(code):
        assert 1 <= r <= clean.shape[0]-2 and 1 <= c <= clean.shape[1]-2
        material = int(clean[r, c])
        assert material in fills
        color = material if same_fill else next(x for x in fills if x != material)
        endpoint = (-r-1, c)
        for depth in range(1, r+2):
            demand(beams, (-depth, c), color)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    demand(rings, (endpoint[0]+dr, endpoint[1]+dc), color)
    return beams, rings, centers


def _world_effects(tile, effects, shape):
    beams, rings, centers = effects
    inside = lambda p: 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]
    wc = {_point(tile, *p) for p in centers}
    assert all(inside(p) for p in wc), 'Echo center outside canvas'
    wb = {_point(tile, *p): color for p, color in beams.items()}
    assert all(inside(p) for p in wb)
    wr = {_point(tile, *p): color for p, color in rings.items() if inside(_point(tile, *p))}
    return wb, wr, wc


def _merge_effects(worlds, occupied):
    centers = set().union(*(x[2] for x in worlds))
    beams, rings = {}, {}
    assert not centers & occupied
    for wb, wr, _ in worlds:
        for target, source in ((beams, wb), (rings, wr)):
            assert not set(source) & occupied, 'Echo intersects a tile body'
            for p, color in source.items():
                if p in centers:
                    continue
                assert p not in target or target[p] == color, 'Conflicting same-stage colors'
                target[p] = color
    return beams, rings, centers


def _render(data, codes=None, same_fill=False):
    clean, tiles = data['clean'], data['tiles']
    if codes is None:
        codes = [data['code']]*len(tiles)
    result = np.full(data['shape'], data['background'], dtype=np.int8)
    occupied = set().union(*(tile['cells'] for tile in tiles))
    for tile in tiles:
        for r, c in np.ndindex(clean.shape):
            result[_point(tile, r, c)] = clean[r, c]
    worlds = [_world_effects(tile, _local_effects(clean, code, data['fills'], same_fill), data['shape'])
              for tile, code in zip(tiles, codes)]
    beams, rings, centers = _merge_effects(worlds, occupied)
    for p, color in beams.items():
        result[p] = color
    for p, color in rings.items():
        result[p] = color
    for p in centers:
        result[p] = data['highlight']
    return result


def _analyze(input_grid):
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    assert np.count_nonzero(counts == counts.max()) == 1
    bg = int(values[np.argmax(counts)])
    assert np.count_nonzero(a == bg) > a.size/2, 'Background must be a strict majority'
    crops, tiles, speckles = [], [], []
    for points in _components(a, bg):
        if len(points) == 1:
            speckles.append(points[0])
            continue
        rs, cs = zip(*points)
        r, c = min(rs), min(cs)
        crop = a[r:max(rs)+1, c:max(cs)+1].copy()
        h, w = sorted(crop.shape)
        assert 4 <= h <= 7 and h+2 <= w <= 10
        assert len(points) >= .80*crop.size
        crops.append(crop)
        tiles.append({'bbox': (r, c)})
    assert 2 <= len(tiles) <= 4
    assert len({tuple(sorted(c.shape)) for c in crops}) == 1
    h, w = sorted(crops[0].shape)
    km = _key_mask(h, w)
    # A key color occurs precisely in this intact L in every crop.
    candidates = []
    for color in values:
        if color == bg:
            continue
        matches = []
        for crop in crops:
            choices = [t for t in range(8) if _orient(crop, t).shape == (h, w)
                       and np.array_equal(_orient(crop, t) == color, km)]
            if len(choices) != 1:
                break
            matches.append(choices[0])
        if len(matches) == len(crops):
            candidates.append((int(color), matches))
    assert len(candidates) == 1, 'Missing or ambiguous orientation key'
    highlight, transforms = candidates[0]
    observations = []
    for tile, crop, t in zip(tiles, crops, transforms):
        observations.append(_orient(crop, t))
        labels = _orient(np.arange(crop.size).reshape(crop.shape), t)
        r, c = tile['bbox']

        def pos(rr, cc):
            x, y = divmod(int(labels[rr, cc]), crop.shape[1])
            return r+x, c+y

        origin, p, q = pos(0, 0), pos(1, 0), pos(0, 1)
        tile.update(origin=origin, u=(p[0]-origin[0], p[1]-origin[1]),
                    v=(q[0]-origin[0], q[1]-origin[1]),
                    cells={pos(rr, cc) for rr, cc in np.ndindex((h, w))}, transform=t)
    clean, fills, rectangle, loss = _fit_rectangle(observations, bg, highlight)
    assert set(map(int, values)) == {bg, highlight, *fills}
    occupied = set().union(*(t['cells'] for t in tiles))
    for p in speckles:
        assert int(a[p]) in fills
        assert all(abs(p[0]-q[0])+abs(p[1]-q[1]) > 1 for q in occupied)
    local_codes = [{tuple(map(int, p)) for p in np.argwhere(obs == bg)} for obs in observations]
    code = set().union(*local_codes)
    assert 2 <= len(code) <= 5
    wrong = [{tuple(map(int, p)) for p in np.argwhere((obs != bg) & (obs != clean))}
             for obs in observations]
    data = {'shape': a.shape, 'background': bg, 'highlight': highlight, 'tiles': tiles,
            'clean': clean, 'fills': fills, 'rectangle': rectangle, 'code': code,
            'local_codes': local_codes, 'wrong': wrong, 'speckles': speckles, 'fit_loss': loss}
    data['expected'] = _render(data)
    return data


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from input pixels alone; never mutate input."""
    return _analyze(input_grid)['expected']


def _sample_clean(rng, bg, fills, highlight, min_height=4):
    for _ in range(80):
        h = int(rng.integers(min_height, 8))
        w = int(rng.integers(h+2, 11))
        r0 = int(rng.integers(h-2))
        r1 = int(rng.integers(r0+2, h))
        c0 = int(rng.integers(1, w-2))
        c1 = int(rng.integers(c0+3, w+1))
        if not 8 <= (r1-r0)*(c1-c0) <= .60*h*w:
            continue
        clean = np.full((h, w), fills[0], dtype=np.int8)
        clean[r0:r1, c0:c1] = fills[1]
        clean[_key_mask(h, w)] = highlight
        interior = clean[1:-1, 1:-1]
        if all(np.count_nonzero(interior == color) >= 3 for color in fills):
            return clean
    # Parameterized geometry, not a stored puzzle or an answer-dependent fallback.
    clean = np.full((6, 9), fills[0], dtype=np.int8)
    clean[0:4, 4:9] = fills[1]
    clean[_key_mask(6, 9)] = highlight
    return clean


def _sample_code(rng, clean, fills, total, cluster):
    pool = [(r, c) for r in range(1, clean.shape[0]-1) for c in range(1, clean.shape[1]-1)]
    for _ in range(100):
        rng.shuffle(pool)
        code = set()
        if cluster:
            pairs = [(p, q) for p in pool for q in pool if p[1] == q[1]
                     and p[0] < q[0] <= p[0]+2 and clean[p] == clean[q]]
            if not pairs:
                return None
            code.update(pairs[int(rng.integers(len(pairs)))])
        for color in fills:
            if any(clean[p] == color for p in code):
                continue
            for p in pool:
                if clean[p] != color:
                    continue
                try:
                    _local_effects(clean, code | {p}, fills)
                except AssertionError:
                    continue
                code.add(p)
                break
        if {int(clean[p]) for p in code} != set(fills):
            continue
        for p in pool:
            if len(code) >= total:
                break
            try:
                _local_effects(clean, code | {p}, fills)
            except AssertionError:
                continue
            code.add(p)
        if len(code) == total:
            signature = {(r, c, int(clean[r, c])) for r, c in code}
            reverse = {(r, clean.shape[1]-1-c, int(clean[r, c])) for r, c in code}
            if signature != reverse:
                return code
    return None


def _place(rng, clean, code, fills, transforms, shape):
    effects = _local_effects(clean, code, fills)
    for _ in range(12):
        placed, worlds = [], []
        occupied = set()
        for t in transforms:
            oh, ow = _orient(clean, t).shape
            blocked = set(occupied)
            for pr, pc in occupied:
                blocked.update(((pr-1, pc), (pr+1, pc), (pr, pc-1), (pr, pc+1)))
            for _ in range(100):
                r = int(rng.integers(shape[0]-oh+1))
                c = int(rng.integers(shape[1]-ow+1))
                tile = _make_tile(clean, t, r, c)
                if tile['cells'] & blocked:
                    continue
                try:
                    world = _world_effects(tile, effects, shape)
                    _merge_effects(worlds+[world], occupied | tile['cells'])
                except AssertionError:
                    continue
                placed.append(tile)
                worlds.append(world)
                occupied |= tile['cells']
                break
            else:
                break
        if len(placed) == len(transforms) and len(occupied) < .46*shape[0]*shape[1]:
            return placed
    return None


def _make_pair(rng, mode):
    for attempt in range(140):
        bg, a, b, highlight = map(int, rng.choice(10, 4, replace=False))
        fills = (a, b)
        clean = _sample_clean(rng, bg, fills, highlight, 5 if mode == 'depth' else 4)
        n = int(rng.choice([2, 3])) if mode in ('pool', 'depth') else int(rng.integers(2, 5))
        if mode == 'test':
            n = int(rng.integers(3, 5))
        total = 3 if mode == 'pool' else int(rng.integers(3, 6))
        code = _sample_code(rng, clean, fills, total, mode == 'depth' or rng.random() < .45)
        if code is None:
            continue
        base = int(rng.integers(8))
        if mode == 'pool':
            ts = [base, base ^ 4]
        elif mode == 'depth':
            ts = [base, (base//4)*4+(base+1)%4]
        elif mode == 'test':
            ts = [base, base ^ 4, (base//4)*4+(base+1)%4]
        else:
            ts = list(map(int, rng.integers(8, size=min(2, n))))
        ts += list(map(int, rng.integers(8, size=n-len(ts))))
        rng.shuffle(ts)
        low = 23 if n >= 3 else 18
        shape = tuple(map(int, rng.integers(low, 31, size=2)))
        if rng.random() < .18:
            shape = (shape[0], shape[0])
        tiles = _place(rng, clean, code, fills, ts, shape)
        if tiles is None:
            continue
        allocations = [set() for _ in tiles]
        points = sorted(code)
        rng.shuffle(points)
        if mode == 'depth' or (mode == 'random' and rng.random() < .35):
            allocations[int(rng.integers(n))] = set(points)
        else:
            recipients = n-1 if mode == 'test' else min(n, 2)
            for index, p in enumerate(points):
                allocations[index % recipients].add(p)
            if mode not in ('pool', 'test') and rng.random() < .3:
                allocations[-1].add(points[0])
        inp = np.full(shape, bg, dtype=np.int8)
        observations = []
        for tile, holes in zip(tiles, allocations):
            obs = clean.copy()
            for p in holes:
                obs[p] = bg
            # Wrong-fill errors never provide erasure evidence. Intact contour
            # and key remain reliable; seam and interior errors are both sampled.
            pool = [(r, c) for r in range(1, clean.shape[0]-1)
                    for c in range(1, clean.shape[1]-1) if (r, c) not in code]
            rng.shuffle(pool)
            if rng.random() < .6 or mode in ('pool', 'depth'):
                target = int(rng.integers(1, 3)) if mode == 'test' else 1
                for p in pool[:target]:
                    obs[p] = next(x for x in fills if x != clean[p])
            observations.append(obs)
            for r, c in np.ndindex(clean.shape):
                inp[_point(tile, r, c)] = obs[r, c]
        occupied = set().union(*(t['cells'] for t in tiles))
        exterior = [tuple(map(int, p)) for p in np.argwhere(inp == bg)
                    if tuple(map(int, p)) not in occupied]
        rng.shuffle(exterior)
        speckles = []
        for p in exterior:
            if len(speckles) >= int(1 + n//2):
                break
            if all(abs(p[0]-q[0])+abs(p[1]-q[1]) > 1 for q in occupied | set(speckles)):
                inp[p] = int(rng.choice(fills))
                speckles.append(p)
        hidden = {'shape': shape, 'background': bg, 'highlight': highlight,
                  'clean': clean, 'fills': fills, 'tiles': tiles, 'code': code}
        expected = _render(hidden)
        try:
            decoded = _analyze(inp)
            assert np.array_equal(decoded['clean'], clean)
            assert decoded['code'] == code
            assert np.array_equal(decoded['expected'], expected)
            assert not np.array_equal(_render(decoded, decoded['local_codes']), expected)
            if mode == 'pool':
                assert all(len(x) < len(code) for x in decoded['local_codes'])
        except AssertionError:
            continue
        return {'input': inp, 'output': expected}
    raise RuntimeError('Bounded layout search exhausted')


def _count(name, value, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count('num_train', num_train, 2, 10)
    ne = _count('num_test', num_test, 1, 4)
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            mode = ('pool', 'depth')[index] if split == 'train' and index < 2 else ('test' if split == 'test' else 'random')
            for _ in range(8):
                pair = _make_pair(rng, mode)
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError('Could not obtain a distinct input')
            seen.add(key)
            puzzle[split].append(pair)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, analyses = set(), []
    for pair in puzzle['train']+puzzle['test']:
        assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
        inp, out = _grid(pair['input']), _grid(pair['output'])
        assert inp.shape == out.shape
        key = inp.shape, inp.tobytes()
        assert key not in seen
        seen.add(key)
        data = _analyze(inp)
        assert np.array_equal(out, data['expected']), 'Incorrect defect sonar answer'
        assert not np.array_equal(out, inp)
        # Explicit parent ablations: no pooling, or no complementary material.
        assert not np.array_equal(_render(data, data['local_codes']), out), 'A has no observable role'
        assert not np.array_equal(_render(data, same_fill=True), out), 'B has no observable role'
        analyses.append(data)
    train = analyses[:nt]
    assert any(all(len(x) < len(d['code']) for x in d['local_codes']) for d in train)
    assert any(any(not x for x in d['local_codes']) for d in train)
    assert any(any(a['u'] == b['u'] and a['v'] == tuple(-x for x in b['v'])
                   for a in d['tiles'] for b in d['tiles']) for d in train), 'No reflected transfer demonstration'
    assert any(any(sum(x*y for x, y in zip(a['u'], b['u'])) == 0
                   for a in d['tiles'] for b in d['tiles']) for d in train), 'No quarter-turn demonstration'
    assert any(any(d['wrong']) for d in train)
    assert all({int(d['clean'][p]) for p in d['code']} == set(d['fills']) for d in analyses)
    assert any(any(p[1] == q[1] and p != q for p in d['code'] for q in d['code']) for d in train)


def _features(data):
    centers = set().union(*(_world_effects(t, _local_effects(data['clean'], data['code'], data['fills']),
                                         data['shape'])[2] for t in data['tiles']))
    directions = [tuple(-x for x in t['u']) for t in data['tiles']]
    return {'shape': data['shape'], 'tile_shape': data['clean'].shape,
            'tiles': len(data['tiles']), 'code_count': len(data['code']),
            'code': tuple(sorted(data['code'])), 'rectangle': data['rectangle'],
            'background': data['background'], 'highlight': data['highlight'],
            'fills': data['fills'], 'directions': tuple(directions),
            'allocations': tuple(len(x) for x in data['local_codes']),
            'wrong': sum(map(len, data['wrong'])), 'speckles': len(data['speckles']),
            'clipped': any(r in (0, data['shape'][0]-1) or c in (0, data['shape'][1]-1) for r, c in centers),
            'cluster': any(p != q and max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 2
                           and data['clean'][p] == data['clean'][q] for p in data['code'] for q in data['code'])}


def self_test():
    """Bounded 42-seed semantics, structure variation, and metamorphic checks."""
    fingerprints, features, clean_shapes = set(), [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(tuple((np.asarray(p['input']).shape, np.asarray(p['input']).tobytes())
                               for p in puzzle['train']+puzzle['test']))
        for pair in puzzle['train']+puzzle['test']:
            d = _analyze(pair['input'])
            features.append(_features(d))
            clean_shapes.add((d['clean'].shape, (d['clean'] == d['fills'][1]).tobytes()))
    assert len(fingerprints) == 42
    assert len(clean_shapes) >= 35
    assert len({f['shape'] for f in features}) >= 35
    assert len({f['tile_shape'] for f in features}) >= 10
    assert {f['tiles'] for f in features} == {2, 3, 4}
    assert {f['code_count'] for f in features} == {3, 4, 5}
    assert len({f['rectangle'] for f in features}) >= 25
    assert len({f['code'] for f in features}) >= 70
    assert {f['background'] for f in features} == set(range(10))
    assert {f['highlight'] for f in features} == set(range(10))
    assert any(f['clipped'] for f in features)
    assert any(not f['clipped'] for f in features)
    assert any(f['wrong'] == 0 for f in features)
    assert any(f['cluster'] for f in features)
    # Recoloring and all eight world isometries must commute with the rule.
    pair = generate_puzzle(1042, 2, 1)['test'][0]
    perm = np.random.default_rng(190).permutation(10)
    for t in range(8):
        inp = _orient(perm[pair['input']], t)
        before = inp.copy()
        assert np.array_equal(solve(inp), _orient(perm[pair['output']], t))
        assert np.array_equal(inp, before)
    first, second = generate_puzzle(9), generate_puzzle(9)
    assert all(np.array_equal(a[k], b[k]) for s in ('train', 'test')
               for a, b in zip(first[s], second[s]) for k in ('input', 'output'))
    # Label corruption must not alter reconstruction and must be rejected.
    bad = generate_puzzle(12, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Mutated output accepted')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    return {'seeds_checked': 42, 'pairs_checked': len(features),
            'distinct_clean_laminates': len(clean_shapes),
            'shapes': len({f['shape'] for f in features}),
            'tile_shapes': len({f['tile_shape'] for f in features}),
            'codes': len({f['code'] for f in features}),
            'tile_counts': dict(Counter(f['tiles'] for f in features)),
            'clipped_pairs': sum(f['clipped'] for f in features),
            'semantic_checks': 'Author-owned input reconstruction and ablations'}
