"""Paired-lane relation beacons. Standard library and NumPy only.

No imports of parent generators, files, or generation state are used by solve.
Generation and tests keep one puzzle at a time and write no artifacts.
"""

from collections import Counter
import numpy as np


_DIRS = ((-1, -1), (-1, 1), (1, 1), (1, -1))
_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _base(shape, colors):
    return np.asarray(colors, dtype=np.int8)[np.indices(shape).sum(axis=0) % 2]


def _outline(center, radius):
    r, c = center
    return {(r + dr, c + dc) for dr in range(-radius, radius + 1)
            for dc in range(-radius, radius + 1) if abs(dr) + abs(dc) == radius}


def _cap(center, radius, direction):
    r, c = center
    dr, dc = _DIRS[direction]
    return [(r + dr * (k + 1), c + dc * (radius - k + 1))
            for k in range(radius + 1)]


def _lanes(cap, shape):
    dr, dc = _DIRS[cap['direction']]
    cells = _cap(cap['center'], cap['radius'], cap['direction'])
    lanes = []
    for r, c in (cells[0], cells[-1]):
        lane = []
        r, c = r + dr, c + dc
        while _inside((r, c), shape):
            lane.append((r, c))
            r, c = r + dr, c + dc
        lanes.append(lane)
    return lanes


def _support(center, kind):
    r, c = center
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if kind == 'square' or dr == 0 or dc == 0}


def _halo(cells):
    return {(r + dr, c + dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _components(cells):
    todo = set(cells)
    components = []
    while todo:
        first = min(todo)
        todo.remove(first)
        part, stack = {first}, [first]
        while stack:
            r, c = stack.pop()
            for dr, dc in _D4:
                q = r + dr, c + dc
                if q in todo:
                    todo.remove(q)
                    part.add(q)
                    stack.append(q)
        components.append(part)
    return components


def _array(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9))
    return a


def _parse(input_grid):
    """Recover all geometry, color roles, and lane memberships from pixels."""
    grid = _array(input_grid)
    parity = np.indices(grid.shape).sum(axis=0) % 2
    colors = []
    for p in (0, 1):
        counts = np.bincount(grid[parity == p].astype(int), minlength=10)
        color = int(counts.argmax())
        assert counts[color] > np.count_nonzero(parity == p) / 2
        colors.append(color)
    assert colors[0] != colors[1]
    base = _base(grid.shape, colors)
    foreground = set(map(tuple, np.argwhere(grid != base)))
    emitters = []
    h, w = grid.shape
    # A genuine outline surrounds an untouched checkerboard interior.
    for radius in (2, 3, 4):
        if min(h, w) < 2 * radius + 1:
            continue
        sh = h - 2 * radius, w - 2 * radius
        offsets = sorted(_outline((0, 0), radius))
        dr, dc = offsets[0]
        values = grid[radius + dr:radius + dr + sh[0],
                      radius + dc:radius + dc + sh[1]]
        mask = ~np.isin(values, colors)
        for dr, dc in offsets[1:]:
            mask &= grid[radius + dr:radius + dr + sh[0],
                         radius + dc:radius + dc + sh[1]] == values
        for rr, cc in np.argwhere(mask):
            center = int(rr) + radius, int(cc) + radius
            interior = [(center[0] + dr, center[1] + dc)
                        for dr in range(-radius + 1, radius)
                        for dc in range(-radius + 1, radius)
                        if abs(dr) + abs(dc) < radius]
            if all(grid[p] == base[p] for p in interior):
                emitters.append(dict(center=center, radius=radius,
                                     struct=int(values[rr, cc])))
    assert 1 <= len(emitters) <= 3
    assert len({e['struct'] for e in emitters}) == 1
    structural = emitters[0]['struct']
    covered, caps = set(), []
    for e in emitters:
        cells = _outline(e['center'], e['radius'])
        assert not covered & cells
        covered.update(cells)
    for e in emitters:
        found = 0
        for direction in range(4):
            cells = _cap(e['center'], e['radius'], direction)
            if not all(_inside(p, grid.shape) for p in cells):
                continue
            values = {int(grid[p]) for p in cells}
            if len(values) == 1 and not values & {*colors, structural}:
                assert not covered.intersection(cells)
                covered.update(cells)
                caps.append(dict(center=e['center'], radius=e['radius'],
                                 direction=direction, key=values.pop()))
                found += 1
        assert found
    assert 1 <= len(caps) <= 8
    tokens = []
    for cells in _components(foreground - covered):
        rows, cols = zip(*cells)
        assert max(rows) - min(rows) == max(cols) - min(cols) == 2
        center = (min(rows) + 1, min(cols) + 1)
        kinds = [kind for kind in ('square', 'cross') if cells == _support(center, kind)]
        assert len(kinds) == 1
        bodies = {int(grid[p]) for p in cells if p != center}
        assert len(bodies) == 1 and center in cells
        body, value = bodies.pop(), int(grid[center])
        assert body not in (*colors, structural) and value not in (*colors, structural)
        assert not cells & _halo(covered)
        tokens.append(dict(center=center, cells=cells, kind=kinds[0], body=body, value=value))
    assert 2 <= len(tokens) <= 24
    assert foreground == covered | set().union(*(t['cells'] for t in tokens))
    for i, t in enumerate(tokens):
        assert all(not t['cells'] & _halo(u['cells']) for u in tokens[:i])
        t['membership'] = None
    used = set()
    for i, cap in enumerate(caps):
        cap['lanes'] = _lanes(cap, grid.shape)
        for lane, cells in enumerate(cap['lanes']):
            assert not used.intersection(cells), 'rays must have unique ownership'
            assert not covered.intersection(cells), 'ray hits an emitter'
            used.update(cells)
            centers = set(cells)
            for t in tokens:
                if t['center'] in centers:
                    assert t['membership'] is None
                    t['membership'] = i, lane
    return dict(base=base, background=tuple(colors), emitters=emitters,
                caps=caps, tokens=tokens, objects=covered)


def _results(analysis):
    """One two-edge relation composition, scoped to the two opposite rails."""
    results = []
    for i, cap in enumerate(analysis['caps']):
        chains = []
        for q in analysis['tokens']:
            if q['membership'] is None or q['membership'][0] != i or q['body'] != cap['key']:
                continue
            opposite = i, 1 - q['membership'][1]
            chains.extend((q, d, d['value']) for d in analysis['tokens']
                          if d['membership'] == opposite and d['body'] == q['value'])
        values = {z for _, _, z in chains}
        assert len(values) <= 1, 'conflicting successful chains within one cap'
        results.append((next(iter(values)) if values else None, chains))
    return results


def _paint(grid, base, emitters, caps, tokens, results):
    out = grid.copy()
    for token in tokens:
        for p in token['cells']:
            out[p] = base[p]
    for cap, color in zip(caps, results):
        if color is not None:
            for lane in _lanes(cap, grid.shape):
                for p in lane:
                    out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Erase the relation tiles and emit each successfully decoded pair of rays."""
    grid = _array(input_grid)
    a = _parse(grid)
    return _paint(grid, a['base'], a['emitters'], a['caps'], a['tokens'],
                  [z for z, _ in _results(a)])


def _closed_form_prediction(grid, a):
    """A second rasterization: diagonal equations, rather than walking rays."""
    out = grid.copy()
    for t in a['tokens']:
        for p in t['cells']:
            out[p] = a['base'][p]
    rows, cols = np.indices(grid.shape)
    for cap, (color, _) in zip(a['caps'], _results(a)):
        if color is None:
            continue
        dr, dc = _DIRS[cap['direction']]
        u = (rows - cap['center'][0]) * dr
        v = (cols - cap['center'][1]) * dc
        out[(np.abs(u - v) == cap['radius']) & (np.minimum(u, v) >= 2)] = color
    return out


def _plans(rng, mode, identity, directions):
    """Latent construction records; solve never sees these records."""
    x, y, z, w, v, u, t = identity
    plans = []
    if mode == 'geometry':
        # A complete chain in an interior lane must not activate its cap.
        plans = [(directions[0], x, 'success', x, y, z),
                 (directions[1], w, 'interior', w, v, u)]
    elif mode == 'scope':
        # Equal keys and equal intermediate identities, different local answers.
        plans = [(d, x, 'success' if j < 3 else 'same', x, y, (z, w, v, u)[j])
                 for j, d in enumerate(directions)]
    else:
        modes = ['success'] + [str(rng.choice(['success', 'same', 'interior', 'missing', 'key', 'empty']))
                              for _ in directions[1:]]
        rng.shuffle(modes)
        for d, kind in zip(directions, modes):
            ids = list(map(int, rng.permutation(identity)))
            px, py, pz, pw = ids[:4]
            # Some valid donors are monochrome; some final rays reuse the key.
            if kind == 'success' and rng.random() < .28:
                pz = int(rng.choice([px, py]))
            plans.append((d, px, kind, pw if kind == 'key' else px, py, pz))
    return plans


def _build_pair(rng, mode, max_radius, regime, split, success_direction=None):
    identity_palette = list(map(int, rng.permutation(10)))
    background, structural, identity = tuple(identity_palette[:2]), identity_palette[2], identity_palette[3:]
    if mode == 'geometry':
        others = [d for d in range(4) if d != success_direction]
        direction_groups = [[success_direction, int(rng.choice(others))]]
        n = 1
    elif mode == 'scope':
        # Three active directions complement the first witness's active one.
        active = [d for d in range(4) if d != success_direction]
        rng.shuffle(active)
        direction_groups = [active + [success_direction]]
        n = 1
    else:
        if split == 'test' and regime in ('count', 'scale'):
            n = 2
        elif regime == 'count':
            n = 1
        else:
            n = int(rng.choice([1, 2], p=[.75, .25]))
        direction_groups = []
        for _ in range(n):
            count = int(rng.integers(1, (4 if n == 1 else 2) + 1))
            direction_groups.append(list(map(int, rng.choice(4, count, replace=False))))
    if mode in ('geometry', 'scope'):
        radii = [max_radius]
    elif regime == 'scale':
        radii = [max_radius] * n if split == 'train' else [2, max_radius]
    else:
        radii = [int(rng.integers(2, max_radius + 1)) for _ in range(n)]
    plans = [_plans(rng, mode if n == 1 else 'random', identity, ds) for ds in direction_groups]
    # At least one success per emitter in random layouts, including both test beacons.
    for attempt in range(180):
        if mode == 'scope':
            low = 2 * max_radius + 20
            shape = tuple(int(rng.integers(low, 31)) for _ in range(2))
        elif n == 2 or split == 'test' and regime == 'distance':
            shape = tuple(int(rng.integers(28, 31)) for _ in range(2))
        else:
            low = 2 * max(radii) + (12 if len(direction_groups[0]) == 1 else 15)
            shape = tuple(int(rng.integers(low, 31)) for _ in range(2))
        emitters, caps, expected = [], [], []
        occupied, rays = set(), set()
        ok = True
        for e_index, (radius, local_plans) in enumerate(zip(radii, plans)):
            if mode == 'scope':
                center = tuple(extent // 2 + int(rng.integers(-1, 1)) for extent in shape)
            else:
                center = tuple(int(rng.integers(radius + 2, extent - radius - 2)) for extent in shape)
            emitter = dict(center=center, radius=radius, struct=structural)
            here = _outline(center, radius)
            new_caps = []
            for direction, key, kind, px, py, pz in local_plans:
                cells = _cap(center, radius, direction)
                if not all(_inside(p, shape) for p in cells):
                    ok = False
                    break
                here.update(cells)
                new_caps.append(dict(center=center, radius=radius, direction=direction,
                                     key=key, plan=(kind, px, py, pz)))
            if not ok:
                break
            new_rays = set(p for cap in new_caps for lane in _lanes(cap, shape) for p in lane)
            # Parallel lanes from one beacon are disjoint. Also reject crossings
            # between beacons, even when their eventual colors would agree.
            if (here & _halo(occupied) or here & rays or new_rays & occupied or new_rays & rays):
                ok = False
                break
            occupied.update(here)
            rays.update(new_rays)
            emitters.append(emitter)
            caps.extend(new_caps)
            expected.extend(pz if kind == 'success' else None
                            for _, _, kind, _, _, pz in local_plans)
        if not ok:
            continue
        # Requests encode geometry only; their colors are independently rendered.
        requests = []
        all_lanes = [_lanes(cap, shape) for cap in caps]
        for i, cap in enumerate(caps):
            kind, px, py, pz = cap['plan']
            if kind == 'empty':
                continue
            lane = int(rng.integers(2))
            first_lane = 'interior' if kind == 'interior' else lane
            second_lane = lane if kind == 'same' else 1 - lane
            requests.append((i, first_lane, px, py, False))
            donor_body = int(rng.choice([c for c in identity if c not in (px, py)])) if kind == 'missing' else py
            far = split == 'test' and regime == 'distance' and kind == 'success'
            requests.append((i, second_lane, donor_body, pz, far))
            if mode == 'scope' and kind == 'same':
                # X->Y and Y->U share one lane; U->T occupies the other.
                # Ignoring the cap key would incorrectly use Y->U->T.
                requests.append((i, 1 - lane, pz, identity[6], False))
        choices = []
        for i, lane, body, value, far in requests:
            cap = caps[i]
            dr, dc = _DIRS[cap['direction']]
            origin = (_cap(cap['center'], cap['radius'], cap['direction'])[cap['radius'] // 2]
                      if lane == 'interior' else
                      _cap(cap['center'], cap['radius'], cap['direction'])[0 if lane == 0 else -1])
            candidates = []
            for k in range(8 if far else 3, 30):
                if split == 'train' and regime == 'distance' and k > 7:
                    break
                center = origin[0] + k * dr, origin[1] + k * dc
                if not _inside(center, shape):
                    break
                kind = str(rng.choice(['square', 'cross']))
                cells = _support(center, kind)
                if not all(_inside(p, shape) for p in cells) or cells & _halo(occupied):
                    continue
                membership = [(j, side) for j, lanes in enumerate(all_lanes)
                              for side, line in enumerate(lanes) if center in line]
                if membership != ([] if lane == 'interior' else [(i, lane)]):
                    continue
                candidates.append(dict(center=center, kind=kind, cells=cells, body=body, value=value))
            rng.shuffle(candidates)
            choices.append(candidates)
        if any(not c for c in choices):
            continue
        # Tiny bounded backtracking avoids a near card blocking a farther one.
        order = sorted(range(len(choices)), key=lambda j: len(choices[j]))
        placed = [None] * len(choices)
        nodes = [0]

        def place(depth, taken):
            nodes[0] += 1
            if nodes[0] > 2500:
                return False
            if depth == len(order):
                return True
            index = order[depth]
            for token in choices[index]:
                if not token['cells'] & taken:
                    placed[index] = token
                    if place(depth + 1, taken | _halo(token['cells'])):
                        return True
            placed[index] = None
            return False

        if not place(0, set()):
            continue
        tokens = placed
        if mode == 'geometry':
            # Explicitly contradict pairing cards at equal distance from a cap.
            q, d = tokens[:2]
            first_lane, second_lane = requests[0][1], requests[1][1]
            if all_lanes[0][first_lane].index(q['center']) == all_lanes[0][second_lane].index(d['center']):
                continue
        # Optional isolated, compatible decoys. They disappear like all tiles.
        for _ in range(int(rng.integers(3))):
            blocked = _halo(occupied | set().union(*(t['cells'] for t in tokens)))
            for _ in range(40):
                center = int(rng.integers(1, shape[0] - 1)), int(rng.integers(1, shape[1] - 1))
                kind = str(rng.choice(['square', 'cross']))
                cells = _support(center, kind)
                if cells & blocked or center in rays:
                    continue
                body, value = map(int, rng.choice(identity, 2, replace=False))
                tokens.append(dict(center=center, cells=cells, kind=kind, body=body, value=value))
                break
        grid = _base(shape, background)
        for e in emitters:
            for p in _outline(e['center'], e['radius']):
                grid[p] = structural
        for cap in caps:
            for p in _cap(cap['center'], cap['radius'], cap['direction']):
                grid[p] = cap['key']
        for token in tokens:
            for p in token['cells']:
                grid[p] = token['body']
            grid[token['center']] = token['value']
        output = _paint(grid, _base(shape, background), emitters, caps, tokens, expected)
        try:
            a = _parse(grid)
            assert np.array_equal(output, solve(grid))
            assert sum(z is not None for z, _ in _results(a)) == sum(z is not None for z in expected)
        except AssertionError:
            continue
        return {'input': grid, 'output': output}
    raise RuntimeError('bounded paired-lane placement exhausted')


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(name + ' outside legal range')
    return int(value)


def _key(grid):
    grid = np.asarray(grid, dtype=np.int8)
    return grid.shape, grid.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(['count', 'scale', 'distance']))
    max_radius = int(rng.integers(3 if regime == 'scale' else 2, 5))
    success_direction = int(rng.integers(4))
    modes = ['geometry', 'scope'] + ['random'] * (num_train - 2)
    rng.shuffle(modes)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            for _ in range(12):
                pair = _build_pair(rng, modes[index] if split == 'train' else 'random',
                                   max_radius, regime, split, success_direction)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('unique-input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(grid):
    a = _parse(grid)
    results = _results(a)
    successes = [i for i, (z, _) in enumerate(results) if z is not None]
    failures = [i for i, (z, _) in enumerate(results) if z is None]
    middle_failure = same_failure = wrong_key_failure = 0
    for i in failures:
        cap = a['caps'][i]
        dr, dc = _DIRS[cap['direction']]
        interior = _cap(cap['center'], cap['radius'], cap['direction'])[1:-1]
        for q in a['tokens']:
            if q['body'] != cap['key'] and q['membership'] is not None and q['membership'][0] == i:
                wrong_key_failure += int(any(d['body'] == q['value'] and
                                         d['membership'] == (i, 1 - q['membership'][1])
                                         for d in a['tokens']))
            if q['body'] != cap['key']:
                continue
            on_interior = any((q['center'][0] - p[0]) * dr == (q['center'][1] - p[1]) * dc >= 1
                              for p in interior)
            middle_failure += int(on_interior and any(d['body'] == q['value'] and
                                  d['membership'] is not None and d['membership'][0] == i
                                  for d in a['tokens']))
            if q['membership'] is not None and q['membership'][0] == i:
                same_failure += int(any(d is not q and d['body'] == q['value'] and
                                        d['membership'] == q['membership'] for d in a['tokens']))
    scoped = any(a['caps'][i]['key'] == a['caps'][j]['key'] and results[i][0] != results[j][0]
                 for i in successes for j in successes if i < j)
    distances, unequal_distance = [], 0
    for i, (z, chains) in enumerate(results):
        for q, d, _ in chains:
            unequal_distance += int(a['caps'][i]['lanes'][q['membership'][1]].index(q['center']) !=
                                    a['caps'][i]['lanes'][d['membership'][1]].index(d['center']))
            for token in (q, d):
                lane = token['membership'][1]
                distances.append(a['caps'][i]['lanes'][lane].index(token['center']) + 1)
    return dict(emitters=len(a['emitters']), caps=len(a['caps']), tokens=len(a['tokens']),
                successes=len(successes), failures=len(failures), middle_failure=middle_failure,
                same_failure=same_failure, wrong_key_failure=wrong_key_failure, scoped=scoped,
                radii=tuple(sorted({e['radius'] for e in a['emitters']})),
                directions=tuple(sorted(a['caps'][i]['direction'] for i in successes)),
                kinds=tuple(sorted({t['kind'] for t in a['tokens']})),
                background=a['background'], shape=grid.shape,
                unequal_distance=unequal_distance,
                max_distance=max(distances, default=0),
                max_ray=max(len(lane) for c in a['caps'] for lane in c['lanes']),
                layout=(grid.shape, tuple((e['center'], e['radius']) for e in a['emitters']),
                        tuple((c['direction'], c['center']) for c in a['caps']),
                        tuple(sorted((t['center'], t['kind']) for t in a['tokens']))))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            grid, out = _array(pair['input']), _array(pair['output'])
            assert grid.shape == out.shape and _key(grid) not in seen
            seen.add(_key(grid))
            a = _parse(grid)
            assert np.array_equal(out, solve(grid))
            assert np.array_equal(out, _closed_form_prediction(grid, a))
            assert not np.array_equal(grid, out)
            for p in a['objects']:
                assert out[p] == grid[p], 'emitter or cap changed'
            for t in a['tokens']:
                # Deletion restores alternating cells, including a tile's old body.
                for p in t['cells']:
                    if not any(z is not None and any(p in lane for lane in c['lanes'])
                               for c, (z, _) in zip(a['caps'], _results(a))):
                        assert out[p] == a['base'][p]
            f = _features(grid)
            assert f['successes'] >= 1
            features[split].append(f)
    train, test = features['train'], features['test']
    assert any(f['middle_failure'] for f in train), 'interior-lane contrast missing'
    assert any(f['same_failure'] for f in train), 'opposite-lane contrast missing'
    assert any(f['wrong_key_failure'] for f in train), 'cap-key contrast missing'
    assert any(f['scoped'] for f in train), 'cap-local donor lookup not demonstrated'
    assert any(f['unequal_distance'] for f in train), 'unequal-range lookup not demonstrated'
    assert {d for f in train for d in f['directions']} == set(range(4))
    assert max(r for f in test for r in f['radii']) <= max(r for f in train for r in f['radii'])
    assert len({f['successes'] for f in train}) >= 2


def self_test() -> dict:
    """42 seeds, streamed; no file output, retained puzzles, or execution traces."""
    shapes, layouts, backgrounds, kinds, radii, emitters, caps, successes = (set() for _ in range(8))
    signatures = set()
    tallies = Counter()
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signature = tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s])
        assert signature not in signatures
        signatures.add(signature)
        if seed == 0:
            first = puzzle
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _features(pair['input'])
                shapes.add(f['shape'])
                layouts.add(f['layout'])
                backgrounds.add(f['background'])
                kinds.update(f['kinds'])
                radii.update(f['radii'])
                emitters.add(f['emitters'])
                caps.add(f['caps'])
                successes.add(f['successes'])
                for name in ('middle_failure', 'same_failure', 'wrong_key_failure', 'scoped', 'unequal_distance'):
                    tallies[name] += int(bool(f[name]))
                tallies['pairs'] += 1
                tallies['long_distance'] += f['max_distance'] >= 8
                tallies['mixed_radii'] += len(f['radii']) > 1
                tallies['long_ray'] += f['max_ray'] >= 12
        # Probe immutability and generation independence of solve.
        inp = puzzle['test'][0]['input'].copy()
        snapshot = inp.copy()
        assert np.array_equal(solve(inp), puzzle['test'][0]['output'])
        assert np.array_equal(inp, snapshot)
    repeated = generate_puzzle(0)
    assert all(_key(p[role]) == _key(q[role]) for s in ('train', 'test')
               for p, q in zip(first[s], repeated[s]) for role in ('input', 'output'))
    assert len(shapes) >= 30 and len(layouts) >= 180 and len(backgrounds) >= 45
    assert kinds == {'square', 'cross'} and radii == {2, 3, 4}
    assert emitters == {1, 2} and {1, 2, 3, 4} <= caps and {1, 2, 3} <= successes
    assert all(tallies[n] >= 42 for n in ('middle_failure', 'same_failure', 'wrong_key_failure', 'scoped', 'unequal_distance'))
    assert tallies['mixed_radii'] >= 5 and tallies['long_distance'] >= 8 and tallies['long_ray'] >= 30
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(9000 + 10 * a + b, a, b)
            assert (len(p['train']), len(p['test'])) == (a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (4, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (2.5, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('accepted illegal pair counts')
    # Damaging one ray label must fail semantic validation.
    bad = {s: [{r: g.copy() for r, g in p.items()} for p in first[s]] for s in ('train', 'test')}
    p = bad['test'][0]
    cell = tuple(np.argwhere(p['input'] != p['output'])[0])
    p['output'][cell] = p['input'][cell]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    # Hand-built boundary and tie cases beyond the ordinary sampler's two-card
    # success pattern. These exercise solve without invoking generation.
    fixture = _base((30, 30), (0, 1))
    center, radius, direction = (14, 14), 4, 1
    for p in _outline(center, radius):
        fixture[p] = 2
    for p in _cap(center, radius, direction):
        fixture[p] = 3
    cap = dict(center=center, radius=radius, direction=direction, key=3)
    lanes = _lanes(cap, fixture.shape)
    query_centers = [lanes[0][2], lanes[0][6]]
    donor_centers = [lanes[1][2], lanes[1][6]]
    for point in query_centers + donor_centers:
        body, value = (3, 4) if point in query_centers else (4, 5)
        for p in _support(point, 'square'):
            fixture[p] = body
        fixture[point] = value
    assert [z for z, _ in _results(_parse(fixture))] == [5]
    assert len(_results(_parse(fixture))[0][1]) == 4
    assert np.array_equal(solve(fixture), _closed_form_prediction(fixture, _parse(fixture)))
    for final_color in (3, 4):
        changed = fixture.copy()
        for point in donor_centers:
            changed[point] = final_color
        answer = solve(changed)
        assert answer[lanes[0][0]] == answer[lanes[1][-1]] == final_color
    conflicting = fixture.copy()
    conflicting[donor_centers[0]] = 6
    try:
        solve(conflicting)
    except AssertionError:
        pass
    else:
        raise AssertionError('conflicting within-cap results were accepted')
    edge = _base((18, 18), (0, 1))
    for p in _outline((3, 3), 2):
        edge[p] = 2
    for direction in (0, 2):
        for p in _cap((3, 3), 2, direction):
            edge[p] = 3
    for point, body, value in (((7, 9), 3, 4), ((13, 11), 4, 5)):
        for p in _support(point, 'cross'):
            edge[p] = body
        edge[point] = value
    edge_answer = solve(edge)
    assert edge_answer[0, 0] == edge[0, 0] == 0
    assert sorted(z for z, _ in _results(_parse(edge)) if z is not None) == [5]
    return dict(seeds=42, pairs=tallies['pairs'], shapes=len(shapes), layouts=len(layouts),
                backgrounds=len(backgrounds), kinds=sorted(kinds), radii=sorted(radii),
                emitter_counts=sorted(emitters), cap_counts=sorted(caps),
                success_counts=sorted(successes), witnesses=dict(tallies),
                hand_cases=['agreeing_duplicate_chains', 'monochrome_donor',
                            'return_to_key', 'conflicting_results_rejected', 'zero_length_cap'])
