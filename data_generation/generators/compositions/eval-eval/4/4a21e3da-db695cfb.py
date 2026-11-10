"""Diagonal ribbon release. Standard library + NumPy; no file I/O on generation."""
from __future__ import annotations

import hashlib
import json

import numpy as np


HOST_SEEDS = (0, 448697017, 732078809, 129852709, 732789324, 371294728,
              88463900, 731164737, 326887339, 961737919, 231351378,
              671373502, 200963883, 148475095, 776541337, 322549206,
              657700241, 62828058, 245007661, 506088404, 238032750,
              141543396, 828279502, 553139050, 606161421, 33151996,
              440195894, 180681889, 21991017, 838235654, 113129382,
              891927407, 229252187, 438749621, 565297583, 863051499,
              413296079, 811089392, 531101011, 915508763, 36547493, 278693666)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _neighbors(p):
    r, c = p
    return ((r-1, c), (r+1, c), (r, c-1), (r, c+1))


def _components(points):
    todo, result = set(points), []
    while todo:
        p = min(todo)
        todo.remove(p)
        found, stack = {p}, [p]
        while stack:
            for q in _neighbors(stack.pop()):
                if q in todo:
                    todo.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _points(grid, color):
    return {tuple(map(int, p)) for p in np.argwhere(grid == color)}


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid grid shape')
    _require(np.issubdtype(a.dtype, np.integer) and a.dtype.kind != 'b', 'noninteger cells')
    _require(bool(np.all((a >= 0) & (a <= 9))), 'invalid colors')
    return a


def _analyze(input_grid):
    """Read all roles, carrier geometry and active cuts from this input alone."""
    g = _grid(input_grid)
    h, w = g.shape
    bg = int(g[0, 0])
    _require(all(int(g[p]) == bg for p in ((0, 0), (0, w-1), (h-1, 0), (h-1, w-1))),
             'corners must show background')
    _require(2 * int(np.count_nonzero(g == bg)) > g.size, 'background must dominate')
    border = np.zeros(g.shape, dtype=bool)
    border[[0, -1], :] = True
    border[:, [0, -1]] = True
    rim = np.argwhere(border & (g != bg))
    _require(len(rim) == 1, 'exactly one border endpoint required')
    border_point = tuple(map(int, rim[0]))
    primary = int(g[border_point])
    endpoints = sorted(_points(g, primary))
    _require(len(endpoints) == 2, 'exactly two carrier endpoints required')
    a, b = endpoints
    dr, dc = b[0] - a[0], b[1] - a[1]
    _require(dr >= 3 and abs(dc) == dr, 'endpoints must define a nontrivial diagonal')
    slope = 1 if dc > 0 else -1
    carrier = {(a[0]+i, a[1]+slope*i) for i in range(dr+1)}
    others = set(map(int, np.unique(g))) - {bg, primary}
    _require(1 <= len(others) <= 2, 'one payload and optional secondary role required')
    adjacent = [color for color in sorted(others)
                if any(q in _points(g, color) for p in _points(g, color) for q in _neighbors(p))]
    _require(len(adjacent) == 1, 'payload is the unique interior role with an orthogonal pair')
    payload = adjacent[0]
    secondary = next(iter(others - {payload}), None)
    cloud = _points(g, payload)
    dots = _points(g, secondary) if secondary is not None else set()
    _require(all(not border[p] for p in cloud | dots), 'payload and secondary dots must be interior')
    _require(not any(q in dots for p in dots for q in _neighbors(p)), 'secondary dots must be isolated')
    active = dots & carrier
    # Endpoints have the primary color, so intersection is automatically strict.
    projection = lambda p: p[0] + slope*p[1]
    cuts = sorted({projection(p) for p in active})
    if cuts:
        low = {p for p in cloud if projection(p) < cuts[0]}
        high = {p for p in cloud if projection(p) > cuts[-1]}
        fixed = cloud - low - high
    else:
        low, high, fixed = set(), set(), cloud.copy()
    return dict(grid=g, bg=bg, primary=primary, payload=payload, secondary=secondary,
                endpoints=endpoints, border_point=border_point, slope=slope,
                carrier=carrier, cloud=cloud, dots=dots, active=active, cuts=cuts,
                low=low, high=high, fixed=fixed)


def _translate(cells, shape, slope, high):
    if not cells:
        return set(), (0, 0)
    h, w = shape
    rows, cols = zip(*cells)
    # Increasing r+s*c points toward SE for s=+1, SW for s=-1.
    bottom = high
    right = high if slope == 1 else not high
    dr = h-1-max(rows) if bottom else -min(rows)
    dc = w-1-max(cols) if right else -min(cols)
    return {(r+dr, c+dc) for r, c in cells}, (dr, dc)


def _render(info, *, activation='finite', relocate=True, fragments=False, first_only=False):
    """Rendering plus explicit wrong-rule switches used only by validation tests."""
    g, s = info['grid'], info['slope']
    cloud, dots = info['cloud'], info['dots']
    if activation == 'finite':
        active = info['active']
    elif activation == 'infinite':
        ar, ac = info['endpoints'][0]
        active = {p for p in dots if p[1]-ac == s*(p[0]-ar)}
    else:
        active = dots
    cuts = sorted({r+s*c for r, c in active})
    if first_only and cuts:
        cuts = cuts[:1]
    if cuts and relocate:
        low = {p for p in cloud if p[0]+s*p[1] < cuts[0]}
        high = {p for p in cloud if p[0]+s*p[1] > cuts[-1]}
        fixed = cloud - low - high
        placed = set(fixed)
        for group, is_high in ((low, False), (high, True)):
            for part in (_components(group) if fragments else [group]):
                placed.update(_translate(part, g.shape, s, is_high)[0])
    else:
        placed = set(cloud)
    out = np.full(g.shape, info['bg'], dtype=np.int8)
    for p in info['carrier']:
        out[p] = info['primary']
    if cuts:
        rr, cc = np.indices(g.shape)
        out[np.isin(rr+s*cc, cuts)] = info['secondary']
    # Dots survive, including inert dots on or off a constructed continuation.
    if info['secondary'] is not None:
        for p in dots:
            out[p] = info['secondary']
    # Payload is opaque: a guide fills gaps rather than repainting parcel cells.
    for p in placed:
        out[p] = info['payload']
    for p in info['endpoints']:
        out[p] = info['primary']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible geometry; no seed, training pairs or generation state."""
    return _render(_analyze(input_grid))


def _cloud(rng, domain, fragments, budget, *, witness=False):
    """Grow separated small silhouettes within a geometric region."""
    domain = set(domain)
    result = set()
    for index in range(fragments):
        safe = domain - result - {q for p in result for q in _neighbors(p)}
        starts = sorted(p for p in safe if index or any(q in safe for q in _neighbors(p)))
        if not starts:
            break
        p = starts[int(rng.integers(len(starts)))]
        part = {p}
        target = (2 if witness and index == 0 else
                  int(rng.integers(1 if index else 2, max(3, budget//fragments + 1))))
        for _ in range(target-1):
            frontier = sorted({q for p in part for q in _neighbors(p) if q in safe - part})
            if not frontier:
                break
            if rng.random() < .5:
                degree = [sum(q in part for q in _neighbors(p)) for p in frontier]
                frontier = [p for p, n in zip(frontier, degree) if n == min(degree)]
            part.add(frontier[int(rng.integers(len(frontier)))])
        result.update(part)
    return result


def _transform(grid, orientation):
    a = np.fliplr(grid) if orientation >= 4 else grid
    return np.rot90(a, orientation % 4).copy()


def _orientation_slope(orientation):
    return (1 if orientation < 4 else -1) * (-1 if orientation % 2 else 1)


def _sample_pair(rng, shape, k, roles, orientation, witness, rich, used):
    """At most 96 proposals, each <=900 cells; rejected candidates stay in RAM."""
    h, w = shape
    bg, primary, secondary, payload = roles
    for attempt in range(96):
        cap = int(rng.integers(2, min(8, w-10)+1))
        maximum = min(h-3, w-cap-3)
        minimum = max(7, k+3)
        if maximum < minimum:
            continue
        length = int(rng.integers(minimum, maximum+1))
        endpoints = {(0, cap), (length, cap+length)}
        beyond = (length+1, cap+length+1)
        if k >= 3:
            inside = list(range(3, length-2))
            indices = [2, length-2] + [int(i) for i in rng.choice(inside, k-2, replace=False)]
        elif k:
            indices = [int(i) for i in rng.choice(range(2, length-1), k, replace=False)]
        else:
            indices = []
        active = {(i, cap+i) for i in indices}
        reserved = active | endpoints | {beyond}
        domain = {(r, c) for r in range(1, h-1) for c in range(1, w-1)} - reserved
        if active:
            lo, hi = min(r+c for r, c in active), max(r+c for r, c in active)
            depth = int(rng.integers(3, 10))
            low_dom = {p for p in domain if lo-depth <= sum(p) < lo}
            high_dom = {p for p in domain if hi < sum(p) <= hi+depth}
            mid_dom = {p for p in domain if lo <= sum(p) <= hi}
            if witness:
                low_fragments = 2
            else:
                low_fragments = int(rng.integers(1, 5 if rich else 4))
            low = _cloud(rng, low_dom, low_fragments, 12 if not rich else 24, witness=witness)
            high = _cloud(rng, high_dom, int(rng.integers(1, 5 if rich else 4)), 18 if not rich else 30)
            fixed = (_cloud(rng, mid_dom, int(rng.integers(1, 4)), 15 if not rich else 24)
                     if k > 1 else {p for p in sorted(mid_dom) if rng.random() < .12})
            if not low or not high:
                continue
            # Teach that fixed payload can hide a cut, and that strict ribbon
            # interiors on BOTH sides of an intermediate cut are all stationary.
            if witness:
                on_cut = sorted(p for p in mid_dom if sum(p) == lo)
                if not on_cut:
                    continue
                fixed.add(on_cut[int(rng.integers(len(on_cut)))])
                if k >= 3:
                    mid = sorted(r+c for r, c in active)[1]
                    for left, right in ((lo, mid), (mid, hi)):
                        choices = sorted(p for p in mid_dom if left < sum(p) < right)
                        if not choices:
                            break
                        fixed.add(choices[int(rng.integers(len(choices)))])
                    else:
                        choices = sorted(p for p in high_dom if sum(p) == hi+1)
                        if choices:
                            high.add(choices[int(rng.integers(len(choices)))])
                    if not any(lo < sum(p) < mid for p in fixed) or not any(mid < sum(p) < hi for p in fixed):
                        continue
                if len(_components(low)) < 2:
                    continue
            cloud = low | high | fixed
            target_lo = _translate(low, shape, 1, False)[0]
            target_hi = _translate(high, shape, 1, True)[0]
            targets = target_lo | target_hi
        else:
            cloud = _cloud(rng, domain, int(rng.integers(2, 6)), 35)
            targets = set()
        if targets & (endpoints | {beyond}) or 2*(len(cloud)+k+8) >= h*w:
            continue
        if not any(q in cloud for p in cloud for q in _neighbors(p)):
            continue
        dots = set(active)
        # Every teaching scene includes a strictly beyond-endpoint inert dot.
        if witness or rng.random() < .55:
            dots.add(beyond)
        count = int(rng.integers(0, 6 if rich else 4))
        candidates = sorted(domain - cloud - targets - dots)
        for i in rng.permutation(len(candidates)):
            if count <= 0:
                break
            p = candidates[int(i)]
            if p[1]-p[0] == cap and 0 <= p[0] <= length:
                continue
            if any(q in dots for q in _neighbors(p)):
                continue
            dots.add(p)
            count -= 1
        g = np.full(shape, bg, dtype=np.int8)
        for p in endpoints:
            g[p] = primary
        for p in dots:
            g[p] = secondary
        for p in cloud:
            g[p] = payload
        g = _transform(g, orientation)
        if _key(g) in used:
            continue
        info = _analyze(g)
        out = _render(info)
        if np.count_nonzero((out == primary) & (g == bg)) < 1:
            continue
        if witness:
            if np.array_equal(_render(info, activation='infinite') == payload, out == payload):
                continue
            if np.array_equal(_render(info, fragments=True) == payload, out == payload):
                continue
            if k >= 3 and np.array_equal(_render(info, first_only=True) == payload, out == payload):
                continue
        return {'input': g, 'output': out}
    raise RuntimeError('96 scene proposals exhausted; no fallback rule is substituted')


def _key(g):
    return tuple(g.shape), g.astype(np.int8, copy=False).tobytes()


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    ntrain, ntest = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    # Mode changes only the sampling distribution, never the transformation.
    mode = int(rng.integers(3))  # more launches, larger canvas, richer fragments
    roles = tuple(map(int, rng.choice(10, 4, replace=False)))
    vary_palette = rng.random() < .55
    first = int(rng.integers(8))
    other_orientations = [i for i in range(8) if _orientation_slope(i) != _orientation_slope(first)]
    second = int(rng.choice(other_orientations))
    plan = [(1, first, True), (3, second, True)]
    plan += [(int(rng.integers(0, 4)), int(rng.integers(8)), False) for _ in range(ntrain-2)]
    rng.shuffle(plan)
    puzzle, used = {'train': [], 'test': []}, set()
    for split in ('train', 'test'):
        schedule = plan if split == 'train' else [
            (int(rng.integers(4, 6) if mode == 0 else rng.integers(1, 5)), int(rng.integers(8)), False)
            for _ in range(ntest)]
        for k, orientation, witness in schedule:
            if mode == 1:
                low, high = (14, 22) if split == 'train' else (23, 30)
            elif mode == 2:
                low, high = (18, 30) if split == 'train' else (14, 26)
            else:
                low, high = 16, 30
            h, w = map(int, rng.integers(low, high+1, size=2))
            if rng.random() < .2:
                w = h
            if vary_palette:
                roles = tuple(map(int, rng.choice(10, 4, replace=False)))
            pair = _sample_pair(rng, (h, w), k, roles, orientation, witness,
                                mode == 2 and split == 'test', used)
            puzzle[split].append(pair)
            used.add(_key(pair['input']))
    validate_generated_puzzle(puzzle)
    return puzzle


def pair_features(grid):
    a = _analyze(grid)
    h, w = a['grid'].shape
    s = a['slope']
    strict_fixed = sum(a['cuts'][0] < r+s*c < a['cuts'][-1] for r, c in a['fixed']) if a['cuts'] else 0
    endpoint = a['border_point']
    side = ('top' if endpoint[0] == 0 else 'bottom' if endpoint[0] == h-1
            else 'left' if endpoint[1] == 0 else 'right')
    return dict(shape=[h, w], slope=s, side=side,
                roles=[a['bg'], a['primary'], a['secondary'], a['payload']],
                carrier_length=len(a['carrier']), active=len(a['active']), inert=len(a['dots']-a['active']),
                ribbon_width=a['cuts'][-1]-a['cuts'][0] if a['cuts'] else None,
                payload=len(a['cloud']), moving=[len(a['low']), len(a['high'])],
                fragments=[len(_components(a['low'])), len(_components(a['high']))],
                fixed=len(a['fixed']), strict_fixed=strict_fixed,
                opaque_cut=sum(r+s*c in a['cuts'] for r, c in a['fixed']))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'bad pair schema')
            g, out = _grid(pair['input']), _grid(pair['output'])
            _require(g.shape == out.shape and _key(g) not in seen, 'shape change or duplicate input')
            seen.add(_key(g))
            a = _analyze(g)
            _require(np.array_equal(solve(g), out), 'wrong output')
            _require(not np.array_equal(g, out), 'identity pair')
            targets = []
            for name, high in (('low', False), ('high', True)):
                target, shift = _translate(a[name], g.shape, a['slope'], high)
                targets.append(target)
                _require(not target & (a['dots'] | set(a['endpoints'])), 'relocation would cover a marker')
                if target:
                    rs, cs = zip(*target)
                    _require((max(rs) == g.shape[0]-1 if high else min(rs) == 0), 'vertical corner alignment')
                    right = high if a['slope'] == 1 else not high
                    _require((max(cs) == g.shape[1]-1 if right else min(cs) == 0), 'horizontal corner alignment')
            expected_cloud = a['fixed'] | targets[0] | targets[1]
            _require(len(expected_cloud) == len(a['cloud']), 'payload lost or collided')
            _require(_points(out, a['payload']) == expected_cloud, 'payload motion mismatch')
            _require(all(int(out[p]) == a['primary'] for p in a['endpoints']), 'endpoint changed')
            _require(all(int(out[p]) == a['secondary'] for p in a['dots']), 'dot changed')
            if a['cuts']:
                _require(bool(a['low']) and bool(a['high']), 'both exterior parcels must be occupied')
            if split == 'train':
                training.append(a)
            elif not a['cuts']:
                _require(any(not t['cuts'] for t in training), 'untriggered query needs demonstration')
    _require({a['slope'] for a in training} == {-1, 1}, 'training must show both slopes')
    _require(any(len(a['active']) == 1 for a in training) and any(len(a['active']) >= 3 for a in training),
             'single-cut and envelope demonstrations required')
    # Compare payload masks, so these witnesses establish motion, not just line paint.
    for switch in ({'relocate': False}, {'activation': 'infinite'}, {'activation': 'all'},
                   {'fragments': True}, {'first_only': True}):
        _require(any(not np.array_equal(_render(a, **switch) == a['payload'],
                                        _render(a) == a['payload']) for a in training),
                 f'missing teaching witness for {switch}')
    _require(any(any(r+a['slope']*c in a['cuts'] for r, c in a['fixed']) for a in training),
             'need payload preserved on a cut')
    _require(any(a['cuts'] and any(a['cuts'][0] < r+a['slope']*c < a['cuts'][-1]
                                 for r, c in a['fixed']) for a in training), 'need protected ribbon interior')


def _puzzle_digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                g = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """42 seeds streamed; retain only small feature sets, never grids or traces."""
    shapes, roles, structures, slopes, sides, launches, widths, digests = (set() for _ in range(8))
    total, fragments, opaque, no_launch, changed_b = 0, 0, 0, 0, 0
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        _require(_puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed)), 'nondeterministic')
        digests.add(_puzzle_digest(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                g, out = pair['input'], pair['output']
                a, f = _analyze(g), pair_features(g)
                if split == 'test':
                    _require(bool(a['active'] and a['low'] and a['high']), 'query must exercise both parents')
                shapes.add(tuple(f['shape']))
                roles.add(tuple(f['roles']))
                slopes.add(f['slope'])
                sides.add(f['side'])
                launches.add(f['active'])
                widths.add(f['ribbon_width'])
                total += 1
                fragments += max(f['fragments']) > 1
                opaque += f['opaque_cut'] > 0
                no_launch += f['active'] == 0
                changed_b += not np.array_equal(_render(a, activation='all') == a['payload'], out == a['payload'])
                normalized = np.full(g.shape, 0, dtype=np.int8)
                for color, role in ((a['primary'], 1), (a['secondary'], 2), (a['payload'], 3)):
                    if color is not None:
                        normalized[g == color] = role
                structures.add(hashlib.sha256(normalized.tobytes()+bytes(g.shape)).hexdigest())
                # Rotation, reflection and arbitrary recoloring preserve the rule.
                for transform in (1, 4):
                    _require(np.array_equal(solve(_transform(g, transform)), _transform(out, transform)),
                             'symmetry failure')
                permutation = np.roll(np.arange(10, dtype=np.int8), 3)
                _require(np.array_equal(solve(permutation[g]), permutation[out]), 'palette dependence')
        if seed == 0:
            bad = {s: [{'input': p['input'].copy(), 'output': p['output'].copy()} for p in puzzle[s]]
                   for s in ('train', 'test')}
            bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
            try:
                validate_generated_puzzle(bad)
            except ValueError:
                pass
            else:
                raise AssertionError('corrupted output accepted')
    _require(len(digests) == 42 and len(structures) >= total*.95, 'insufficient structure diversity')
    _require(len(shapes) >= 50 and len(roles) >= 30, 'insufficient geometry/palette diversity')
    _require(slopes == {-1, 1} and len(sides) == 4 and {0, 1, 2, 3, 4, 5} <= launches, 'missing geometric modes')
    _require(fragments >= 80 and opaque >= 80 and no_launch >= 5 and changed_b >= 70, 'missing interactions')
    _require(len(widths) >= 10, 'insufficient ribbon variation')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1701+10*nt+nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, 'wrong requested counts')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Absent-secondary/no-launch family member, plus a manually fixed motion case.
    hand = np.zeros((14, 14), dtype=np.int8)
    hand[0, 3] = hand[8, 11] = 1
    hand[3, 6] = 2
    for p in ((1, 2), (1, 3), (3, 2), (8, 8), (8, 9), (4, 5)):
        hand[p] = 3
    expected_payload = {(0, 0), (0, 1), (2, 0), (13, 12), (13, 13), (4, 5)}
    predicted = solve(hand)
    _require(_points(predicted, 3) == expected_payload, 'manual translation regression')
    _require(predicted[1, 4] == 1 and predicted[8, 11] == 1 and predicted[9, 12] == 0,
             'manual finite-carrier regression')
    _require(predicted[0, 9] == 2 and predicted[9, 0] == 2 and predicted[10, 0] == 0
             and predicted[4, 5] == 3, 'manual full-perpendicular and opaque-payload regression')
    hand[3, 6] = 0
    _require(_points(solve(hand), 3) == _points(hand, 3), 'absent-secondary behavior')
    return {'seeds': 42, 'pairs': total, 'shapes': len(shapes), 'role_assignments': len(roles),
            'normalized_structures': len(structures), 'fragmented_pairs': fragments,
            'opaque_cut_pairs': opaque, 'untriggered_pairs': no_launch, 'B_changes_payload': changed_b,
            'launch_counts': sorted(launches), 'ribbon_widths': sorted(w for w in widths if w is not None)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
