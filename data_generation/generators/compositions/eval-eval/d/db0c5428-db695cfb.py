"""Diagonal looms: endpoint corner samples turn clockwise at active relays.

Only NumPy and the standard library are used. No files or random state are read
by solve(). All diagnostics in self_test() are bounded and kept in memory.
"""
from __future__ import annotations

import hashlib
import numbers

import numpy as np


HOST_SEEDS = (0, 602355708, 500633466, 420980626, 348138853, 844276233,
              335492990, 620418194, 559482503, 374083205, 225547576,
              715392262, 249208441, 401841581, 403548703, 63431305,
              747542871, 712923906, 607903282, 193400737, 427848869,
              103134356, 564333072, 654581055, 696419859, 330247497,
              792555902, 821293049, 344200804, 66480033, 101717507,
              672634380, 36557974, 733944933, 954814588, 57874883,
              582740780, 87301361, 173037095, 318044098, 293054697, 153037792)
CORNERS = ((0, 0), (0, 2), (2, 0), (2, 2))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid ARC color")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _box(p):
    return {(p[0] + y, p[1] + x) for y in (-1, 0, 1) for x in (-1, 0, 1)}


def _near(cells):
    return cells | {(r + dr, c + dc) for r, c in cells
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))}


def _aligned(a, b):
    return a != b and abs(a[0] - b[0]) == abs(a[1] - b[1])


def _segment(a, b):
    dr = 1 if b[0] > a[0] else -1
    dc = 1 if b[1] > a[1] else -1
    return tuple((a[0] + i * dr, a[1] + i * dc)
                 for i in range(abs(b[0] - a[0]) + 1))


def _ray(p, direction, shape):
    r, c = p[0] + direction[0], p[1] + direction[1]
    cells = []
    while 0 <= r < shape[0] and 0 <= c < shape[1]:
        cells.append((r, c))
        r, c = r + direction[0], c + direction[1]
    return tuple(cells)


def _components(mask):
    todo = {tuple(map(int, p)) for p in np.argwhere(mask)}
    components = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in todo:
                    todo.remove(q)
                    cells.add(q)
                    stack.append(q)
        components.append(cells)
    return components


def _analyze(input_grid):
    """Parse the visible objects and all geometry, without generation metadata."""
    a = _grid(input_grid)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    _require(int(counts.max()) * 2 > a.size, "background must occupy more than half the input")
    tiles, markers, blocks = {}, {}, set()
    for cells in _components(a != bg):
        if len(cells) == 1:
            p = next(iter(cells))
            markers[p] = int(a[p])
            continue
        rows, cols = zip(*cells)
        p = (min(rows) + 1, min(cols) + 1)
        _require(len(cells) == 9 and cells == _box(p), "objects must be full 3x3 tiles or dots")
        tiles[p] = a[p[0] - 1:p[0] + 2, p[1] - 1:p[1] + 2].copy()
        blocks.update(cells)
    _require(len(tiles) >= 2, "at least two tiles are required")
    tile_values = np.concatenate([t.ravel() for t in tiles.values()])
    v, n = np.unique(tile_values, return_counts=True)
    carrier = int(v[np.argmax(n)])
    _require(all(np.count_nonzero(t == carrier) >= 5 for t in tiles.values()),
             "one common carrier must dominate every tile")
    _require(len(set(markers.values())) <= 1, "all relay dots must share a color")
    marker_color = next(iter(markers.values())) if markers else None
    _require(marker_color not in set(map(int, tile_values)), "relay color must not occur in tiles")
    centers = sorted(tiles)
    degrees = dict.fromkeys(centers, 0)
    segments, paired = [], set()
    for i, p in enumerate(centers):
        for q in centers[i + 1:]:
            if _aligned(p, q):
                degrees[p] += 1
                degrees[q] += 1
                cells = _segment(p, q)
                _require(len(cells) >= 6, "paired tile centers must be at least five steps apart")
                _require(not ((set(cells) & blocks) - _box(p) - _box(q)), "bridge passes through a third tile")
                for source, other in ((p, q), (q, p)):
                    dr, dc = int(np.sign(other[0] - source[0])), int(np.sign(other[1] - source[1]))
                    _require(tiles[source][1 + dr, 1 + dc] != carrier,
                             "paired inward corners must contain detail colors")
                segments.append((p, q, cells))
                paired.update((p, q))
    _require(segments and max(degrees.values()) <= 1, "tile partners must be unique")
    primary = set().union(*(set(s[2]) for s in segments))
    active, rays = {}, []
    for point in sorted(markers):
        containing = [j for j, s in enumerate(segments) if point in s[2]]
        _require(len(containing) <= 1, "a relay cannot lie on two bridges")
        if not containing:
            continue
        j = containing[0]
        active[point] = j
        p, q, _ = segments[j]
        for source, other in ((p, q), (q, p)):
            inward = (int(np.sign(other[0] - source[0])), int(np.sign(other[1] - source[1])))
            ink = int(tiles[source][1 + inward[0], 1 + inward[1]])
            _require(ink != carrier, "paired inward corners must contain detail colors")
            toward_source = (-inward[0], -inward[1])
            direction = (toward_source[1], -toward_source[0])  # Clockwise quarter-turn.
            cells = _ray(point, direction, a.shape)
            _require(not (set(cells) & primary), "a secondary ray crosses another primary span")
            _require(not (set(cells) & blocks), "a secondary ray crosses a tile")
            _require(not (set(cells) & set(markers)), "a secondary ray crosses an original relay")
            rays.append(dict(origin=point, source=source, inward=inward,
                             direction=direction, ink=ink, cells=cells))
    assigned = {}
    for ray in rays:
        for point in ray['cells']:
            _require(point not in assigned or assigned[point] == ray['ink'], "unequal ray inks collide")
            assigned[point] = ray['ink']
    return dict(grid=a, background=bg, carrier=carrier, marker_color=marker_color,
                tiles=tiles, blocks=blocks, markers=markers, segments=segments,
                primary=primary, active=active, rays=rays, ray_inks=assigned,
                unpaired=set(tiles) - paired)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read endpoint corners, complete finite bridges, and route active inks."""
    info = _analyze(input_grid)
    a = info['grid']
    out = a.copy()
    for point in info['primary']:
        if int(a[point]) == info['background']:
            out[point] = info['carrier']
    for point, ink in info['ray_inks'].items():
        out[point] = ink
    return out


def _finite_primary_foil(info, output):
    for p, q, _ in info['segments']:
        slope = 1 if q[1] > p[1] else -1
        outside = _ray(p, (-1, -slope), output.shape) + _ray(q, (1, slope), output.shape)
        if any(output[x] == info['background'] for x in outside):
            return True
    return False


def _geometry(rng, shape, specs, step_limits, unpaired_count, beyond, inert_count):
    """Bounded free placement. Colors are assigned only after geometry is fixed."""
    h, w = shape
    centers, blocks, primary, secondary = [], set(), set(), set()
    segments, triggers = [], []
    for slope, trigger_count in specs:
        for _ in range(100):
            lower, upper = step_limits
            upper = min(upper, h - 3, w - 3)
            if upper < lower:
                return None
            steps = int(rng.integers(lower, upper + 1))
            r = int(rng.integers(1, h - steps - 1))
            c = int(rng.integers(1, w - steps - 1)) + (steps if slope == -1 else 0)
            p, q = (r, c), (r + steps, c + slope * steps)
            new_blocks = _box(p) | _box(q)
            cells = _segment(p, q)
            span = set(cells)
            if any(_aligned(x, y) for x in (p, q) for y in centers):
                continue
            if new_blocks & (_near(blocks) | primary | secondary):
                continue
            if span & (blocks | secondary):
                continue
            eligible = []
            for point in cells[2:-2]:
                line = set(_ray(point, (1, -slope), shape)) | set(_ray(point, (-1, slope), shape))
                if line & (primary | span) or line & (blocks | new_blocks):
                    continue
                eligible.append((point, line))
            if len(eligible) < trigger_count:
                continue
            chosen = [eligible[int(i)] for i in rng.permutation(len(eligible))[:trigger_count]]
            # Distinct original dots may be diagonally adjacent, but not 4-adjacent.
            new_triggers = {x[0] for x in chosen}
            old_triggers = {p for p, _ in triggers}
            if new_triggers & _near(old_triggers):
                continue
            index = len(segments)
            segments.append((p, q, cells))
            centers.extend((p, q))
            blocks.update(new_blocks)
            primary.update(span)
            for point, line in chosen:
                triggers.append((point, index))
                secondary.update(line)
            break
        else:
            return None
    for _ in range(unpaired_count):
        for _ in range(100):
            point = (int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1)))
            cells = _box(point)
            if any(_aligned(point, c) for c in centers):
                continue
            if cells & (_near(blocks) | primary | secondary | _near({p for p, _ in triggers})):
                continue
            centers.append(point)
            blocks.update(cells)
            break
        else:
            return None
    dots = {p for p, _ in triggers}
    forbidden = primary | secondary | _near(blocks) | _near(dots)
    inert = []
    if beyond:
        candidates = set()
        for p, q, _ in segments:
            slope = 1 if q[1] > p[1] else -1
            outside = _ray(p, (-1, -slope), shape) + _ray(q, (1, slope), shape)
            for point in outside:
                line = set(_ray(point, (1, -slope), shape)) | set(_ray(point, (-1, slope), shape))
                # At least four currently blank cells ensure an observable foil
                # even after up to three optional inert dots have been added.
                if len(line - blocks - primary - secondary - dots) >= 4:
                    candidates.add(point)
        candidates = sorted(candidates - forbidden)
        if not candidates:
            return None
        point = candidates[int(rng.integers(len(candidates)))]
        inert.append(point)
        forbidden.update(_near({point}))
    free = [(r, c) for r in range(h) for c in range(w) if (r, c) not in forbidden]
    for i in rng.permutation(len(free)):
        if len(inert) >= max(inert_count, int(beyond)):
            break
        point = free[int(i)]
        if point in forbidden:
            continue
        inert.append(point)
        forbidden.update(_near({point}))
    if len(inert) < max(inert_count, int(beyond)):
        return None
    return dict(centers=centers, segments=segments, triggers=triggers,
                blocks=blocks, inert=inert)


def _paint(rng, shape, geometry, palette, witness):
    bg, carrier, marker, *details = palette
    centers, segments = geometry['centers'], geometry['segments']
    # At a crossing, both rays must carry the same visible sampled ink. Union
    # their source identities before sampling corner colors; no priority rule.
    parent = {p: p for p in centers}

    def root(p):
        while parent[p] != p:
            p = parent[p]
        return p

    occupied = {}
    for point, j in geometry['triggers']:
        p, q, _ = segments[j]
        for source, other in ((p, q), (q, p)):
            inward = (int(np.sign(other[0] - source[0])), int(np.sign(other[1] - source[1])))
            direction = (-inward[1], inward[0])
            for cell in _ray(point, direction, shape):
                if cell in occupied:
                    parent[root(source)] = root(occupied[cell])
                occupied[cell] = source
    inks = {root(p): int(rng.choice(details)) for p in centers}
    if witness:
        # The required active segment is first. Its two different inks expose
        # chirality and both half-lines, even at the minimum training count.
        p, q, _ = segments[0]
        if root(p) == root(q):
            return None
        colors = rng.choice(details, size=2, replace=False)
        inks[root(p)], inks[root(q)] = map(int, colors)
    inward_by_source = {}
    for p, q, _ in segments:
        inward_by_source[p] = (int(np.sign(q[0] - p[0])), int(np.sign(q[1] - p[1])))
        inward_by_source[q] = (-inward_by_source[p][0], -inward_by_source[p][1])
    a = np.full(shape, bg, dtype=np.int8)
    for p in centers:
        tile = np.full((3, 3), carrier, dtype=np.int8)
        if witness and p in segments[0][:2]:
            colors = list(map(int, rng.choice(details, size=4, replace=False)))
            for corner, color in zip(CORNERS, colors):
                tile[corner] = color
        else:
            # Repeated corners, sparse corners, and off-corner decoration all
            # occur. A strict carrier majority leaves the tile role visible.
            count = int(rng.integers(1, 5))
            positions = list(CORNERS) if rng.random() < .65 else list(np.ndindex(3, 3))
            rng.shuffle(positions)
            repeated = rng.random() < .45
            color = int(rng.choice(details))
            for pos in positions[:count]:
                tile[pos] = color if repeated else int(rng.choice(details))
        if p in inward_by_source:
            dr, dc = inward_by_source[p]
            corner = (1 + dr, 1 + dc)
            tile[corner] = inks[root(p)]
            # Preserve the five-cell carrier majority after forcing the sample.
            extras = [pos for pos in np.ndindex(3, 3) if pos != corner and tile[pos] != carrier]
            rng.shuffle(extras)
            for pos in extras[3:]:
                tile[pos] = carrier
            if witness and p in segments[0][:2]:
                remaining = [x for x in details if x != inks[root(p)]]
                rng.shuffle(remaining)
                for pos, color in zip([x for x in CORNERS if x != corner], remaining):
                    tile[pos] = color
        a[p[0] - 1:p[0] + 2, p[1] - 1:p[1] + 2] = tile
    for p in geometry['inert'] + [p for p, _ in geometry['triggers']]:
        a[p] = marker
    if np.count_nonzero(a == bg) * 2 <= a.size:
        return None
    try:
        out = solve(a)
    except ValueError:
        return None
    return dict(input=a, output=out)


def _new_pair(rng, regime, test, palette, witness, used):
    for attempt in range(160):
        if regime == 'context' and test:
            shape = (int(rng.integers(14, 21)), int(rng.integers(26, 31)))
            if rng.random() < .5:
                shape = shape[::-1]
        else:
            shape = tuple(map(int, rng.integers(17 if not test else 19, 31, size=2)))
        if regime == 'context' and not test:
            shape = tuple(map(int, rng.integers(17, 25, size=2)))
        if regime == 'span':
            limits = (8, 17) if test else (5, 7)
        else:
            limits = (5, min(shape) - 3)
        if regime == 'multiplicity':
            count = 3 if test else int(rng.integers(1, 3))
        else:
            count = int(rng.choice((1, 1, 2, 2, 3)))
        specs = []
        if witness:
            specs.append((witness['slope'], witness['triggers']))
        if witness and witness.get('zero'):
            specs.append((int(rng.choice((-1, 1))), 0))
        count = max(count, len(specs))
        # Additional segments can independently be untriggered or multi-triggered.
        parallel = int(rng.choice((-1, 1))) if rng.random() < .45 else None
        while len(specs) < count:
            specs.append((parallel or int(rng.choice((-1, 1))),
                          int(rng.choice((0, 0, 0, 1, 1, 2, 3)))))
        if test and rng.random() < .15:
            specs = [(s, 0) for s, _ in specs]
        unpaired = int(witness.get('unpaired', 0)) if witness else int(rng.choice((0, 0, 1)))
        beyond = bool(witness and witness.get('beyond'))
        inert = int(rng.choice((0, 0, 1, 2, 3)))
        geometry = _geometry(rng, shape, specs, limits, unpaired, beyond, inert)
        if geometry is None:
            continue
        pair = _paint(rng, shape, geometry, palette, bool(witness))
        if pair is not None and _key(pair['input']) not in used:
            if beyond and not _finite_primary_foil(_analyze(pair['input']), pair['output']):
                continue
            return pair
    # A finite construction is a last resort, not a bank of whole puzzles.
    # Reuse the same requested semantics and shift, on a roomier canvas, with
    # fewer optional interactions. Witness slots still expose every mechanism.
    shape = (30, 30)
    if regime == 'context' and not test:
        shape = (24, 24)
    if regime == 'context' and test:
        shape = (20, 30)
    limits = ((8, 10) if test else (5, 7)) if regime == 'span' else (5, 9)
    specs = [(witness['slope'], witness['triggers'])] if witness else [(1, 1)]
    if witness and witness.get('zero'):
        specs.append((1, 0))
    if regime == 'multiplicity' and test:
        specs = [(1, 1), (1, 0), (1, 0)]
    for _ in range(400):
        geometry = _geometry(rng, shape, specs, limits,
                             int(bool(witness and witness.get('unpaired'))),
                             bool(witness and witness.get('beyond')), 0)
        if geometry is not None:
            pair = _paint(rng, shape, geometry, palette, bool(witness))
            if pair is not None and _key(pair['input']) not in used:
                if witness and witness.get('beyond') and not _finite_primary_foil(
                        _analyze(pair['input']), pair['output']):
                    continue
                return pair
    raise RuntimeError('bounded geometric construction exhausted')


def _counts(num_train, num_test):
    for value in (num_train, num_test):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
            raise ValueError('counts must be integers')
    if not (2 <= num_train <= 10 and 1 <= num_test <= 4 and num_train > num_test):
        raise ValueError('require 2..10 train, 1..4 test, and num_train > num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, m = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(('span', 'multiplicity', 'context')))
    order = list(map(int, rng.permutation(10)))
    # Carrier and relay colors stay fixed within a puzzle; the background policy
    # is independently fixed or variable. All ten colors can hold every role.
    carrier, marker = order[:2]
    background_pool = order[2:5]
    details = order[5:]
    fixed_bg = rng.random() < .5
    slots = list(map(int, rng.choice(n, size=2, replace=False)))
    slopes = list(map(int, rng.permutation((-1, 1))))
    witnesses = {slots[0]: dict(slope=slopes[0], triggers=2, beyond=True, unpaired=1),
                 slots[1]: dict(slope=slopes[1], triggers=1, zero=True)}
    puzzle, used = dict(train=[], test=[]), set()
    for split, count in (('train', n), ('test', m)):
        for i in range(count):
            bg = background_pool[0] if fixed_bg else int(rng.choice(background_pool))
            palette = [bg, carrier, marker] + details
            pair = _new_pair(rng, regime, split == 'test', palette,
                             witnesses.get(i) if split == 'train' else None, used)
            used.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
    a, b = _grid(pair['input']), _grid(pair['output'])
    _require(a.shape == b.shape, 'output shape changed')
    info = _analyze(a)
    # Cellwise verification of preservation, exact support and the two ray inks.
    touched = set()
    for p in info['primary']:
        if a[p] == info['background']:
            _require(b[p] == info['carrier'], 'finite bridge has wrong ink')
            touched.add(p)
    for ray in info['rays']:
        source = ray['source']
        dr, dc = ray['inward']
        ink = int(a[source[0] + dr, source[1] + dc])
        for p in ray['cells']:
            _require(int(b[p]) == ink, 'ray does not carry its source corner sample')
            touched.add(p)
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            p = (r, c)
            if p not in touched:
                _require(a[p] == b[p], 'unassigned input cell changed')
    _require(bool(np.all(b[a != info['background']] == a[a != info['background']])),
             'an original tile or relay changed')
    _require(not np.array_equal(a, b), 'identity pair')
    _require(set(map(int, np.unique(b))) <= set(map(int, np.unique(a))), 'new literal color')
    return info


def _teaching_checks(train):
    infos = [_analyze(p['input']) for p in train]
    _require({r['inward'] for a in infos for r in a['rays']} ==
             {(-1, -1), (-1, 1), (1, -1), (1, 1)}, 'both active slopes must be taught')
    _require(any(max((sum(j == k for j in a['active'].values())
                          for k in range(len(a['segments']))), default=0) >= 2 for a in infos),
             'multiple relays on one bridge must be taught')
    _require(any(any(k not in a['active'].values() for k in range(len(a['segments']))) for a in infos),
             'an untriggered bridge must be taught')
    _require(any(a['unpaired'] for a in infos), 'an unpaired tile must be taught')
    _require(any(_finite_primary_foil(a, _grid(p['output'])) for a, p in zip(infos, train)),
             'finite versus full primary extension must be observable')
    beyond = False
    unique_positions = {d: set(range(9)) for d in ((-1, -1), (-1, 1), (1, -1), (1, 1))}
    chirality = False
    for pair, a in zip(train, infos):
        output = _grid(pair['output'])
        for ray in a['rays']:
            tile = a['tiles'][ray['source']]
            unique_positions[ray['inward']].intersection_update(
                i for i in range(9) if int(tile.flat[i]) == ray['ink'])
        for point, j in a['active'].items():
            inks = {r['ink'] for r in a['rays'] if r['origin'] == point}
            chirality |= len(inks) == 2
        for point in set(a['markers']) - set(a['active']):
            for p, q, cells in a['segments']:
                if _aligned(point, p) and _aligned(point, q) and point not in cells:
                    # Ignoring finiteness would create an observable extra line.
                    slope = 1 if q[1] > p[1] else -1
                    line = _ray(point, (1, -slope), a['grid'].shape) + _ray(point, (-1, slope), a['grid'].shape)
                    beyond |= any(output[x] == a['background'] for x in line)
    _require(beyond, 'a visible beyond-endpoint activation foil is required')
    _require(chirality, 'different inks must distinguish clockwise from counterclockwise')
    for d, candidates in unique_positions.items():
        _require(candidates == {(1 + d[0]) * 3 + 1 + d[1]}, 'training leaves a sample-position ambiguity')


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    used, carriers, markers = set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = _check_pair(pair)
            key = _key(a['grid'])
            _require(key not in used, 'duplicate input')
            used.add(key)
            carriers.add(a['carrier'])
            if a['marker_color'] is not None:
                markers.add(a['marker_color'])
    _require(len(carriers) == 1 and len(markers) == 1, 'puzzle role colors changed')
    _teaching_checks(puzzle['train'])


def measure_puzzle(puzzle):
    """Small observables, never a duplicate archive of grids."""
    result = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            a = _analyze(pair['input'])
            counts = tuple(sum(j == k for j in a['active'].values())
                           for k in range(len(a['segments'])))
            result.append(dict(split=split, shape=a['grid'].shape,
                               background=a['background'], carrier=a['carrier'],
                               segments=len(a['segments']), unpaired=len(a['unpaired']),
                               steps=tuple(len(s[2]) - 1 for s in a['segments']),
                               slopes=tuple(1 if q[1] > p[1] else -1 for p, q, _ in a['segments']),
                               triggers=counts, inactive=len(a['markers']) - len(a['active']),
                               inks=len({r['ink'] for r in a['rays']}),
                               crossings=sum(sum(p in r['cells'] for r in a['rays']) > 1
                                             for p in a['ray_inks']),
                               edge_tile=any(r in (1, a['grid'].shape[0] - 2) or
                                             c in (1, a['grid'].shape[1] - 2) for r, c in a['tiles']),
                               repeated_sample=any(np.count_nonzero(a['tiles'][r['source']] == r['ink']) > 1
                                                   for r in a['rays'])))
    return result


def _regression_fixture():
    """A hand-specified loom, not a generated pair or a parent fixture."""
    a = np.zeros((13, 15), dtype=np.int8)
    a[2:5, 2:5] = [[1, 8, 3], [8, 8, 8], [4, 8, 2]]
    a[8:11, 8:11] = [[7, 8, 4], [8, 8, 8], [3, 8, 6]]
    a[2:5, 12:15] = 8
    a[3, 13] = 3  # Unpaired, with irrelevant decoration.
    a[6, 6] = a[1, 1] = 5
    b = a.copy()
    b[5, 5] = b[7, 7] = 8
    for i in range(1, 7):
        b[6 - i, 6 + i] = 2
        b[6 + i, 6 - i] = 7
    return dict(input=a, output=b)


def self_test():
    fixture = _regression_fixture()
    _check_pair(fixture)
    for k in range(4):
        a, b = np.rot90(fixture['input'], k), np.rot90(fixture['output'], k)
        _require(np.array_equal(solve(a), b), 'clockwise routing is not rotation-equivariant')
    for offset in range(10):
        a, b = (fixture['input'] + offset) % 10, (fixture['output'] + offset) % 10
        _require(np.array_equal(solve(a), b), 'literal-color dependence')
    tight = np.zeros((8, 8), dtype=np.int8)
    tight[:3, :3] = fixture['input'][2:5, 2:5]
    tight[5:8, 5:8] = fixture['input'][8:11, 8:11]
    tight[3, 3] = tight[4, 4] = 5
    tight_expected = tight.copy()
    for r, c in ((3, 3), (4, 4)):
        for i in range(1, 4):
            tight_expected[r - i, c + i] = 2
            tight_expected[r + i, c - i] = 7
    _require(np.array_equal(solve(tight), tight_expected), 'tight boundary construction failed')
    a = fixture['input'].copy()
    a[4, 4] = 6  # Change just the upper endpoint's inward sample.
    b = fixture['output'].copy()
    b[4, 4] = 6
    for i in range(1, 7):
        b[6 - i, 6 + i] = 6
    _require(np.array_equal(solve(a), b), 'sample perturbation did not follow its routed half-line')
    a = fixture['input'].copy()
    a[6, 6] = 0
    b = a.copy()
    b[5, 5] = b[6, 6] = b[7, 7] = 8
    _require(np.array_equal(solve(a), b), 'inactive loom still emitted ink')
    # Two finite bridges can cross in carrier ink, but a relay there has two
    # incompatible launch directions and must be rejected.
    a[2:5, 12:15] = 0
    for r, c in ((2, 10), (10, 2)):
        a[r - 1:r + 2, c - 1:c + 2] = 8
    a[3, 9] = a[9, 3] = 4
    _require(solve(a)[6, 6] == 8, 'unmarked carrier crossing failed')
    a[6, 6] = 5
    try:
        solve(a)
    except ValueError:
        pass
    else:
        raise AssertionError('two-span relay ambiguity accepted')
    signatures, rows, structures = set(), [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _grid(pair['input'])
                digest.update(bytes(a.shape) + a.tobytes())
                bg = _analyze(a)['background']
                crop = np.argwhere(a != bg)
                lo, hi = crop.min(axis=0), crop.max(axis=0) + 1
                labels = {bg: 0}
                norm = tuple(labels.setdefault(int(x), len(labels))
                             for x in a[lo[0]:hi[0], lo[1]:hi[1]].flat)
                structures.add((tuple(hi - lo), norm))
                original = a.copy()
                _require(np.array_equal(solve(a), pair['output']) and np.array_equal(a, original),
                         'solver mismatch or input mutation')
        signatures.add(digest.digest())
        rows.extend(measure_puzzle(puzzle))
    _require(len(signatures) == 42 and len(structures) >= 150, 'insufficient structural diversity')
    _require(len({r['shape'] for r in rows}) >= 50, 'insufficient canvas diversity')
    _require({r['segments'] for r in rows} >= {1, 2, 3}, 'segment counts collapsed')
    _require({n for r in rows for n in r['triggers']} >= {0, 1, 2, 3}, 'trigger counts collapsed')
    _require({r['background'] for r in rows} == set(range(10)), 'background role coverage')
    _require(len({r['carrier'] for r in rows}) >= 8, 'carrier role coverage')
    for feature in ('repeated_sample', 'edge_tile', 'crossings'):
        _require(any(r[feature] for r in rows), 'missing family variation: ' + feature)
    _require(any(0 in r['triggers'] and max(r['triggers']) > 0 for r in rows), 'no mixed activation')
    _require(any(max(r['triggers']) == 0 for r in rows if r['split'] == 'test'), 'no untriggered queries')
    # Reproducibility includes every output and all explicit count combinations.
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(7100 + 10 * n + m, n, m)
            _require(len(p['train']) == n and len(p['test']) == m, 'count request ignored')
    p, q = generate_puzzle(2026, 6, 3), generate_puzzle(2026, 6, 3)
    _require(all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], q[s])
                 for k in ('input', 'output')), 'seed is not reproducible')
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                 (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Corrupt a ray, bridge, original source and untouched cell independently.
    pair = p['train'][0]
    info = _analyze(pair['input'])
    points = [next(iter(info['blocks'])), next(iter(info['primary'] - info['blocks'] - set(info['markers'])))]
    if info['ray_inks']:
        points.append(next(iter(info['ray_inks'])))
    untouched = {(r, c) for r in range(pair['input'].shape[0]) for c in range(pair['input'].shape[1])}
    untouched -= info['primary'] | info['blocks'] | set(info['markers']) | set(info['ray_inks'])
    points.append(min(untouched))
    for point in points:
        bad = dict(input=pair['input'], output=pair['output'].copy())
        bad['output'][point] = (int(bad['output'][point]) + 1) % 10
        try:
            _check_pair(bad)
        except ValueError:
            continue
        raise AssertionError('corrupted output accepted')


if __name__ == '__main__':
    self_test()
