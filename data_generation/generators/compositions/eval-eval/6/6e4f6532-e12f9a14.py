"""Ray rendezvous: projected beams supply targets and compasses for loose joints.

No parent module, stored example, file, or random state is used by solve().
Only the standard library and NumPy are required. Nothing is saved by this module.
"""
from collections import Counter
from fractions import Fraction
import hashlib
import numpy as np


DIRS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
        (0, 1), (1, -1), (1, 0), (1, 1))
PORTS = ((-1, -1), (-1, 0), (-1, 1), (-1, 2), (0, -1),
         (0, 2), (1, -1), (1, 2), (2, -1), (2, 0), (2, 1), (2, 2))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _add(p, d, k=1):
    return p[0] + k * d[0], p[1] + k * d[1]


def _direction(p):
    return tuple(-1 if v == -1 else 1 if v == 2 else 0 for v in p)


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _intersection(a, b, shape):
    """Exact spatial intersection, independent of distances from the ports."""
    p, u = a['p'], a['d']
    q, v = b['p'], b['d']
    det = _cross(u, v)
    if not det:
        # Collinear forward overlap is deliberately outside this family.
        if not _cross((q[0] - p[0], q[1] - p[1]), u):
            raise ValueError('collinear rays are not allowed')
        return None
    delta = q[0] - p[0], q[1] - p[1]
    t = Fraction(_cross(delta, v), det)
    s = Fraction(_cross(delta, u), det)
    point = _add(p, u, t)
    if t <= 0 or s <= 0 or not _inside(point, shape):
        return None
    return t, s, point


def _rays(specs):
    return [{'owner': i, 'color': spec['color'], 'p': _add(spec['core'], off),
             'd': _direction(off)}
            for i, spec in enumerate(specs) for off in sorted(spec['ports'])]


def _rendezvous(rays, shape):
    """Pair mutual first spatial encounters; no clock and no hidden pairing."""
    hits = [[] for _ in rays]
    for i, a in enumerate(rays):
        for j in range(i):
            b = rays[j]
            hit = _intersection(a, b, shape)
            if hit is not None:
                t, s, point = hit
                hits[i].append((t, j, point))
                hits[j].append((s, i, point))
    first = []
    for row in hits:
        _require(bool(row), 'a ray has no rendezvous')
        row.sort()
        _require(len(row) == 1 or row[0][0] < row[1][0], 'tied first encounter')
        first.append(row[0])
    joints = []
    color_pairs = set()
    for i, (t, j, point) in enumerate(first):
        _require(first[j][1] == i, 'encounters are not mutual first intersections')
        _require(all(v.denominator == 1 for v in point), 'half-cell rendezvous')
        _require(rays[i]['owner'] != rays[j]['owner'], 'self rendezvous')
        _require(rays[i]['color'] != rays[j]['color'], 'indistinguishable pin colors')
        if i < j:
            colors = frozenset((rays[i]['color'], rays[j]['color']))
            _require(colors not in color_pairs, 'repeated color-pair rendezvous')
            color_pairs.add(colors)
            joints.append({'rays': (i, j), 'q': tuple(int(v) for v in point),
                           'distances': (int(t), int(first[j][0])), 'colors': colors})
    _require(len(joints) * 2 == len(rays), 'unpaired ray')
    return joints


def _halo(points):
    return {_add(p, d) for p in points for d in DIRS} | set(points)


def _components(points):
    remaining, result = set(points), []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        group, stack = {p}, [p]
        while stack:
            p = stack.pop()
            for d in DIRS:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _transform(p, t):
    r, c = p
    if t >= 4:
        c = -c
    for _ in range(t % 4):
        r, c = c, -r
    return r, c


def _mapped(cells, t=0, offset=(0, 0)):
    return {_add(_transform(p, t), offset): v for p, v in cells.items()}


def _key(grid):
    return grid.shape, np.asarray(grid, dtype=np.uint8).tobytes()


def _signature(cells):
    r0 = min(p[0] for p in cells)
    c0 = min(p[1] for p in cells)
    return tuple(sorted((r - r0, c - c0, v) for (r, c), v in cells.items()))


def _parse(input_grid):
    g = np.asarray(input_grid)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), 'invalid grid dimensions')
    _require(np.issubdtype(g.dtype, np.integer) and np.all((g >= 0) & (g <= 9)),
             'invalid grid colors')
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(sum(counts == counts[bg]) == 1, 'ambiguous background')
    candidates = []
    h, w = g.shape
    for r in range(1, h - 2):
        for c in range(1, w - 2):
            payload = int(g[r, c])
            if payload == bg or not np.all(g[r:r+2, c:c+2] == payload):
                continue
            ring = [int(g[_add((r, c), p)]) for p in PORTS]
            other = set(ring) - {bg}
            if len(other) != 1 or not 1 <= ring.count(bg) <= 3:
                continue
            structure = next(iter(other))
            if structure == payload:
                continue
            candidates.append({'core': (r, c), 'color': payload,
                               'structure': structure,
                               'ports': tuple(p for p, v in zip(PORTS, ring) if v == bg)})
    _require(len(candidates) >= 2, 'missing launcher vocabulary')
    structures = {s['structure'] for s in candidates}
    _require(len(structures) == 1, 'ambiguous structural role')
    structure = structures.pop()
    payloads = [s['color'] for s in candidates]
    _require(len(set(payloads)) == len(payloads), 'launcher payloads must be distinct')
    footprints, covered = set(), set()
    for s in candidates:
        r, c = s['core']
        footprint = {(a, b) for a in range(r-1, r+3) for b in range(c-1, c+3)}
        _require(not footprints & footprint, 'overlapping launchers')
        footprints |= footprint
        covered |= {p for p in footprint if g[p] != bg}
    loose = set(zip(*np.nonzero(g != bg))) - covered
    groups = _components(loose)
    _require(bool(groups), 'missing loose joints')
    cells = [{p: int(g[p]) for p in group} for group in groups]
    roles = set().union(*(set(piece.values()) for piece in cells)) - set(payloads)
    _require(len(roles) == 2 and structure not in roles, 'ambiguous piece roles')
    anchors = [v for v in roles if all(list(p.values()).count(v) == 1 for p in cells)]
    _require(len(anchors) == 1, 'registration must be the unique singleton piece role')
    reg = anchors[0]
    body = (roles - {reg}).pop()
    pieces = []
    for piece in cells:
        hist = Counter(piece.values())
        _require(hist[body] >= 4 and hist[reg] == 1, 'malformed piece body/registration')
        pins = {p: v for p, v in piece.items() if v in payloads}
        _require(len(pins) == 2 and len(set(pins.values())) == 2, 'two distinct pins required')
        anchor = next(p for p, v in piece.items() if v == reg)
        relative = {_add(p, anchor, -1): v for p, v in piece.items()}
        pin_data = {}
        for p, color in pins.items():
            delta = _add(p, anchor, -1)
            length = max(map(abs, delta))
            _require(2 <= length <= 3 and all(v in (-length, 0, length) for v in delta),
                     'a pin must lie on a compass ray, two or three cells from its anchor')
            pin_data[color] = (tuple(v // length for v in delta), length)
        directions = [v[0] for v in pin_data.values()]
        _require(_cross(*directions) != 0, 'collinear piece pins')
        pieces.append({'cells': relative, 'source': piece, 'anchor': anchor, 'pins': pin_data})
    rays = _rays(candidates)
    joints = _rendezvous(rays, g.shape)
    _require(len(pieces) == len(joints), 'piece/rendezvous count mismatch')
    _require({frozenset(p['pins']) for p in pieces} == {j['colors'] for j in joints},
             'piece colors do not bijectively identify rendezvous')
    return {'grid': g, 'bg': bg, 'structure': structure, 'body': body, 'reg': reg,
            'specs': candidates, 'footprints': footprints, 'covered': covered,
            'rays': rays, 'joints': joints, 'pieces': pieces}


def _assemble(info, use_original_pose=False, draw_beams=True):
    """Input-only reconstruction, also used for explicit concept ablations."""
    g, rays = info['grid'], info['rays']
    out = g.copy()
    for piece in info['pieces']:
        for p in piece['source']:
            out[p] = info['bg']
    placements, routes, occupied = [], [], set(info['covered'])
    for joint in info['joints']:
        piece = next(p for p in info['pieces'] if frozenset(p['pins']) == joint['colors'])
        transforms = []
        for t in range(8):
            if all(_transform(piece['pins'][rays[i]['color']][0], t) ==
                   tuple(-v for v in rays[i]['d']) for i in joint['rays']):
                transforms.append(t)
        _require(len(transforms) == 1, 'pins do not determine a unique rigid orientation')
        t = 0 if use_original_pose else transforms[0]
        final = _mapped(piece['cells'], t, joint['q'])
        _require(all(_inside(p, g.shape) for p in final), 'docked joint is clipped')
        _require(not set(final) & occupied and not set(final) & info['footprints'],
                 'docked joints overlap launchers or each other')
        occupied |= set(final)
        placements.append((piece, joint, transforms[0], final))
        if not use_original_pose:
            for i, distance in zip(joint['rays'], joint['distances']):
                ray = rays[i]
                length = piece['pins'][ray['color']][1]
                _require(distance >= length + 2, 'insufficient approach length')
                route = [_add(ray['p'], ray['d'], k) for k in range(distance - length + 1)]
                _require(final.get(route[-1]) == ray['color'], 'ray misses its pin')
                routes.append((i, route))
    for _, _, _, final in placements:
        for p, value in final.items():
            out[p] = value
    if draw_beams and not use_original_pose:
        painted, mids = set(), set()
        for i, route in routes:
            color = rays[i]['color']
            for p in route[:-1]:
                _require(p not in occupied and p not in painted, 'beam overwrites a joint or trail')
                _require(p not in info['footprints'] or p == rays[i]['p'], 'beam enters a launcher')
                painted.add(p)
                out[p] = color
            for a, b in zip(route, route[1:]):
                if a[0] != b[0] and a[1] != b[1]:
                    mid = _add(a, b)
                    slope = (b[0]-a[0]) * (b[1]-a[1])
                    _require((mid, -slope) not in mids, 'unresolved between-cell crossing')
                    mids.add((mid, slope))
    return out, placements, routes


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Discover port intersections; orient and dock pieces; connect to their pins."""
    return _assemble(_parse(input_grid))[0]


def _base(shape, specs):
    g = np.zeros(shape, dtype=np.int8)
    occupied = set()
    for s in specs:
        r, c = s['core']
        footprint = {(a, b) for a in range(r-1, r+3) for b in range(c-1, c+3)}
        _require(all(_inside(p, shape) for p in footprint), 'ring clipped')
        _require(not footprint & _halo(occupied), 'launcher footprints too close')
        occupied |= footprint
        g[r:r+2, c:c+2] = s['color']
        for off in PORTS:
            if off not in s['ports']:
                g[_add(s['core'], off)] = 1
    return g


def _spec(point, direction, color, rng):
    options = [p for p in PORTS if _direction(p) == direction]
    off = options[int(rng.integers(len(options)))]
    return {'core': _add(point, off, -1), 'color': color, 'ports': (off,)}


def _network_ok(shape, specs, count):
    _base(shape, specs)
    rays = _rays(specs)
    joints = _rendezvous(rays, shape)
    _require(len(joints) == count, 'wrong joint count')
    _require(all(min(j['distances']) >= 5 for j in joints), 'short approach')
    _require(all(3 <= j['q'][a] < shape[a] - 3 for j in joints for a in (0, 1)),
             'joint too close to edge')
    for i, j in enumerate(joints):
        _require(all(max(abs(j['q'][a] - k['q'][a]) for a in (0, 1)) >= 7
                     for k in joints[:i]), 'crowded rendezvous')
    # Reserve the whole possible body envelope before accepting an emitter.
    # Previously a valid ray network could place another launcher immediately
    # beside an old joint, leaving no space for almost any sampled silhouette.
    # Planning both mechanisms together avoids wasting outer construction
    # attempts; this is a packing check, not a new transformation convention.
    footprints = {p for s in specs
                  for r in range(s['core'][0]-1, s['core'][0]+3)
                  for c in range(s['core'][1]-1, s['core'][1]+3) for p in [(r, c)]}
    reserved = []
    for joint in joints:
        q = joint['q']
        envelope = {_add(q, (r, c)) for r in range(-2, 3) for c in range(-2, 3)}
        envelope |= {_add(q, rays[i]['d'], -3) for i in joint['rays']}
        _require(not envelope & _halo(footprints), 'no reserved space for a joint')
        _require(all(not envelope & _halo(other) for other in reserved),
                 'joint reserves conflict')
        reserved.append(envelope)
    for j, joint in enumerate(joints):
        other_pieces = set().union(*(s for k, s in enumerate(reserved) if k != j))
        for i, distance in zip(joint['rays'], joint['distances']):
            ray = rays[i]
            route = {_add(ray['p'], ray['d'], k) for k in range(distance - 1)}
            _require(not route & other_pieces, 'approach crosses another joint reserve')
            _require(not (route - {ray['p']}) & footprints, 'approach enters a launcher')
    return joints


def _network(rng, shape, count, teaching):
    pairs = [(a, b) for a in DIRS for b in DIRS if _cross(a, b)]
    if teaching == 'cardinal':
        pairs = [(a, b) for a, b in pairs if sum(map(abs, a)) == sum(map(abs, b)) == 1]
    elif teaching == 'diagonal':
        pairs = [(a, b) for a, b in pairs if sum(map(abs, a)) == sum(map(abs, b)) == 2]
    specs = None
    for _ in range(240):
        q = tuple(int(rng.integers(3, n-3)) for n in shape)
        a, b = pairs[int(rng.integers(len(pairs)))]
        distances = [int(v) for v in rng.integers(5, 18, 2)]
        trial = [_spec(_add(q, d, -k), d, c, rng)
                 for d, k, c in zip((a, b), distances, (2, 3))]
        try:
            _network_ok(shape, trial, 1)
            specs = trial
            break
        except ValueError:
            pass
    _require(specs is not None, 'primary geometry exhausted')
    # Every additional rendezvous shares an existing emitter, making a tree
    # of interacting objects rather than a stack of independent mini scenes.
    for n in range(2, count + 1):
        found = False
        for _ in range(500):
            owner = int(rng.integers(len(specs)))
            s = specs[owner]
            options = [p for p in PORTS if p not in s['ports']]
            if len(s['ports']) >= 3:
                continue
            off = options[int(rng.integers(len(options)))]
            p, a = _add(s['core'], off), _direction(off)
            k = int(rng.integers(5, 23))
            q = _add(p, a, k)
            bs = [d for d in DIRS if _cross(a, d)]
            b = bs[int(rng.integers(len(bs)))]
            new = _spec(_add(q, b, -int(rng.integers(5, 23))), b, n+2, rng)
            trial = [dict(s) for s in specs] + [new]
            trial[owner]['ports'] = s['ports'] + (off,)
            try:
                _network_ok(shape, trial, n)
                specs, found = trial, True
                break
            except ValueError:
                pass
        _require(found, 'shared-emitter extension exhausted')
    return specs


def _piece(rng, joint, rays):
    """Procedural silhouette, independent of the launcher placement/template."""
    pins, body = {}, {(0, 0)}
    for i in joint['rays']:
        direction = tuple(-v for v in rays[i]['d'])
        length = int(rng.integers(2, 4))
        pins[_add((0, 0), direction, length)] = rays[i]['color']
        body |= {_add((0, 0), direction, k) for k in range(1, length)}
    # Preserve the one-cell aperture to each pin; no body cell can be in the
    # incoming external beam corridor. All actual empty cells remain empty.
    forbidden = set()
    for p in pins:
        length = max(map(abs, p))
        d = tuple(v // length for v in p)
        forbidden |= {_add((0, 0), d, k) for k in range(length, 4)}
    target_size = int(rng.integers(max(5, len(body)), 16))
    for _ in range(30):
        if len(body) >= target_size:
            break
        frontier = sorted({q for p in body for d in DIRS for q in [_add(p, d)]
                           if max(map(abs, q)) <= 2 and q not in body and q not in forbidden})
        if not frontier:
            break
        body.add(frontier[int(rng.integers(len(frontier)))])
    cells = {p: 8 for p in body}
    cells[(0, 0)] = 9
    cells.update(pins)
    return cells


def _construct_pair(rng, count, teaching, pose, larger):
    low, high = (24, 30) if larger else (21, 29)
    shape = tuple(int(v) for v in rng.integers(low, high+1, 2))
    specs = _network(rng, shape, count, teaching)
    inp = _base(shape, specs)
    rays = _rays(specs)
    joints = _network_ok(shape, specs, count)
    finals, relative_pieces = [], []
    footprints = {(r, c) for s in specs for r in range(s['core'][0]-1, s['core'][0]+3)
                  for c in range(s['core'][1]-1, s['core'][1]+3)}
    final_occupied = set(footprints)
    for joint in joints:
        piece = _piece(rng, joint, rays)
        final = _mapped(piece, offset=joint['q'])
        _require(not set(final) & _halo(final_occupied), 'joint/launcher packing conflict')
        final_occupied |= set(final)
        relative_pieces.append(piece)
        finals.append(final)
    # Sources can lie on future beams and old/new footprints can intersect.
    blocked = _halo(footprints)
    source_cells = []
    for index, piece in enumerate(relative_pieces):
        if index == 0 and pose == 'rotate':
            t = int(rng.integers(1, 4))
        elif index == 0 and pose == 'reflect':
            t = int(rng.integers(4, 8))
        else:
            t = int(rng.integers(8))
        tile = _mapped(piece, t)
        r0, r1 = min(p[0] for p in tile), max(p[0] for p in tile)
        c0, c1 = min(p[1] for p in tile), max(p[1] for p in tile)
        origins = [(r, c) for r in range(-r0, shape[0]-r1)
                   for c in range(-c0, shape[1]-c1)]
        rng.shuffle(origins)
        for origin in origins:
            source = _mapped(tile, offset=origin)
            if set(source) & blocked or source == finals[index]:
                continue
            blocked |= _halo(source)
            source_cells.append(source)
            break
        else:
            raise ValueError('cannot pack loose source')
    expected = inp.copy()
    for final in finals:
        for p, value in final.items():
            expected[p] = value
    for joint, final in zip(joints, finals):
        for i, distance in zip(joint['rays'], joint['distances']):
            ray = rays[i]
            pin = next(p for p, v in final.items() if v == ray['color'])
            length = max(abs(pin[a] - joint['q'][a]) for a in (0, 1))
            for k in range(distance - length + 1):
                p = _add(ray['p'], ray['d'], k)
                _require(int(expected[p]) in (0, ray['color']), 'constructed beam obstructed')
                expected[p] = ray['color']
    for source in source_cells:
        for p, value in source.items():
            inp[p] = value
    # Pair-local palette and scene pose do not encode any hidden conventions.
    palette = np.asarray(rng.permutation(10), dtype=np.int8)
    inp, expected = palette[inp], palette[expected]
    t = int(rng.integers(8))
    inp, expected = np.rot90(inp, t % 4), np.rot90(expected, t % 4)
    if t >= 4:
        inp, expected = np.fliplr(inp), np.fliplr(expected)
    inp, expected = np.ascontiguousarray(inp), np.ascontiguousarray(expected)
    info = _parse(inp)
    out, placements, _ = _assemble(info)
    _require(np.array_equal(out, expected), 'construction disagrees with input-only rule')
    if pose:
        _require(any(_signature(p['source']) != _signature(final)
                     for p, _, _, final in placements), 'pose demonstration is degenerate')
    return {'input': inp, 'output': expected}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, lo, hi, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        _require(isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
                 and lo <= value <= hi, name + ' is outside its legal integer range')
    _require(num_train > num_test, 'num_train must exceed num_test')
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and seed >= 0, 'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, number in (('train', num_train), ('test', num_test)):
        for index in range(number):
            teaching = 'cardinal' if split == 'train' and index == 0 else (
                       'diagonal' if split == 'train' and index == 1 else None)
            pose = ('rotate' if index == 0 else 'reflect') if split == 'train' and index < 2 else None
            count = (1 if index == 0 else 2) if split == 'train' else (
                    3 if index == 0 else int(rng.integers(2, 4)))
            if split == 'train' and index > 1:
                count = int(rng.integers(1, 3))
            rejections = Counter()
            for attempt in range(80):
                try:
                    pair = _construct_pair(rng, count, teaching, pose, split == 'test')
                    key = _key(pair['input'])
                    _require(key not in seen, 'duplicate input')
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
                except ValueError as error:
                    rejections[str(error)] += 1
                    continue
            else:
                raise RuntimeError(f'bounded construction exhausted at seed {seed}/{split}/{index}, '
                                   f'joints={count}: {rejections.most_common(4)}')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info, placements, routes):
    angles = set()
    for joint in info['joints']:
        a, b = [info['rays'][i]['d'] for i in joint['rays']]
        angles.add((sum(x*y for x, y in zip(a, b)), abs(_cross(a, b))))
    moves = []
    for piece, _, _, final in placements:
        sig = _signature(final)
        possibilities = [t for t in range(8) if _signature(_mapped(piece['source'], t)) == sig]
        moves.append('translation' if 0 in possibilities else 'rotation' if any(t < 4 for t in possibilities)
                     else 'reflection')
    return {'shape': tuple(info['grid'].shape), 'joints': len(info['joints']),
            'launchers': len(info['specs']), 'port_counts': tuple(len(s['ports']) for s in info['specs']),
            'angles': tuple(sorted(angles)), 'moves': tuple(moves),
            'unequal': sum(a != b for a, b in (j['distances'] for j in info['joints'])),
            'approaches': tuple(len(route) for _, route in routes),
            'body_sizes': tuple(sum(v == info['body'] for v in p['cells'].values()) for p in info['pieces']),
            'bg': info['bg'], 'structure': info['structure'],
            'emission': tuple(sorted({sum(map(abs, r['d'])) for r in info['rays']})),
            'body_shapes': tuple(_signature({p: 1 for p in piece['cells']}) for piece in info['pieces'])}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'illegal puzzle counts')
    seen, train_features = set(), []
    compass_needed, emission_needed = False, False
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            inp, out = (np.asarray(pair[k]) for k in ('input', 'output'))
            info = _parse(inp)
            _require(out.shape == inp.shape and np.issubdtype(out.dtype, np.integer)
                     and np.all((out >= 0) & (out <= 9)), 'invalid output grid')
            key = _key(inp)
            _require(key not in seen, 'duplicate puzzle input')
            seen.add(key)
            expected, placements, routes = _assemble(info)
            _require(np.array_equal(expected, out), 'output violates rendezvous docking')
            _require(not np.array_equal(inp, out), 'unchanged pair')
            _require(all(int(out[p]) == int(inp[p]) for p in info['covered']), 'launcher changed')
            for piece, joint, _, final in placements:
                _require(int(out[joint['q']]) == info['reg'], 'registration misses virtual target')
                _require(Counter(piece['source'].values()) == Counter(final.values()), 'piece cells lost')
                for p, v in final.items():
                    _require(int(out[p]) == v, 'piece overwritten by a beam')
            if split == 'train':
                feature = _features(info, placements, routes)
                train_features.append(feature)
                # Removing A keeps source pose at the ray-derived target.
                try:
                    wrong = _assemble(info, use_original_pose=True)[0]
                    compass_needed |= not np.array_equal(wrong, out)
                except ValueError:
                    compass_needed = True
                # Removing B's emission leaves correctly moved but disconnected joints.
                emission_needed |= not np.array_equal(_assemble(info, draw_beams=False)[0], out)
    _require(compass_needed and emission_needed, 'training fails a parent-concept ablation')
    _require({1, 2} <= set().union(*(set(f['emission']) for f in train_features)),
             'training must expose side and corner ports')
    _require(any(max(f['port_counts']) >= 2 for f in train_features), 'no shared-emitter demonstration')
    _require(any('rotation' in f['moves'] for f in train_features), 'no rotation demonstration')
    _require(any('reflection' in f['moves'] for f in train_features), 'no reflection demonstration')


def self_test():
    """Bounded: one puzzle at a time, 42 seeds, no grid archives or trace files."""
    # Explicit pixel fixture: the eastward and northward approaches reach
    # (8,12) after nine and eleven steps. Their last cells are colored pins,
    # not the registration cell. No simulation clock or constructed target
    # object supplies the crossing to solve().
    specs = [{'core': (8, 1), 'color': 2, 'ports': ((0, 2),)},
             {'core': (20, 11), 'color': 3, 'ports': ((-1, 1),)}]
    fixture = _base((24, 24), specs)
    target = {(8, 12): 9, (8, 11): 8, (8, 10): 2, (9, 12): 8,
              (10, 12): 8, (11, 12): 3, (9, 13): 8, (7, 12): 8}
    # Handwritten reflected/translated copy of precisely those colored cells.
    source = {(16, 5): 9, (16, 6): 8, (16, 7): 2, (17, 5): 8,
              (18, 5): 8, (19, 5): 3, (17, 4): 8, (15, 5): 8}
    expected = fixture.copy()
    for p, v in target.items():
        expected[p] = v
    expected[8, 3:11] = 2
    expected[11:20, 12] = 3
    for p, v in source.items():
        fixture[p] = v
    before = fixture.copy()
    assert np.array_equal(solve(fixture), expected)
    assert np.array_equal(fixture, before), 'solve mutated its input'
    info = _parse(fixture)
    assert info['joints'][0]['q'] == (8, 12)
    assert sorted(info['joints'][0]['distances']) == [9, 11]
    # Ignoring the precise side gaps and projecting from core centers moves
    # the crossing to (8.5,11.5), demonstrating a causal role for negative space.
    center_rays = [{'p': (Fraction(17, 2), Fraction(3, 2)), 'd': (0, 1)},
                   {'p': (Fraction(41, 2), Fraction(23, 2)), 'd': (-1, 0)}]
    assert _intersection(*center_rays, (24, 24))[2] != (8, 12)
    features, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _parse(pair['input'])
                out, placements, routes = _assemble(info)
                assert np.array_equal(out, pair['output'])
                digest.update(bytes(pair['input'].shape))
                digest.update(pair['input'].tobytes())
                features.append(_features(info, placements, routes))
        signatures.add(digest.hexdigest())
    assert len(signatures) == 42
    assert len({f['shape'] for f in features}) >= 30
    assert {f['joints'] for f in features} == {1, 2, 3}
    assert {f['launchers'] for f in features} == {2, 3, 4}
    assert {f['bg'] for f in features} == set(range(10))
    assert len({s for f in features for s in f['body_shapes']}) >= 100
    assert {'rotation', 'reflection', 'translation'} <= {m for f in features for m in f['moves']}
    assert sum(f['unequal'] for f in features) >= 100
    assert max(max(f['port_counts']) for f in features) == 3
    assert len({angle for f in features for angle in f['angles']}) == 4
    a, b = generate_puzzle(7, 2, 1), generate_puzzle(7, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s])
               for k in ('input', 'output'))
    # The host exercises every legal count combination. Keep this local
    # self-test focused on both count boundaries rather than repeating it.
    for nt, ne in ((2, 1), (10, 4)):
        p = generate_puzzle(1700+nt*10+ne, nt, ne)
        assert len(p['train']) == nt and len(p['test']) == ne
    # Original full-verifier failure: individually valid beams crowded out the
    # third installation. Keep the seed/count request as a packing regression.
    validate_generated_puzzle(generate_puzzle(1743, 4, 3))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None), (4, False)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = generate_puzzle(8, 2, 1)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('validator accepted a corrupted answer')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds plus explicit-count and corruption checks)')
