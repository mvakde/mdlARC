"""Boundary-probed local ports gate inward sampling and radial tile echoes.

Only Python's standard library and NumPy are used. No files, global random state,
parent modules, saved examples, or generation metadata are needed by solve().
"""
from __future__ import annotations

import hashlib
import json

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
HOST_SEEDS = (0, 790151492, 434366251, 637378614, 389871903, 569768445,
              413945184, 162855285, 768797195, 64802560, 84778233, 878788295,
              526630817, 937008546, 113180423, 215567168, 358923055, 574563624,
              513623688, 269528044, 809349414, 479960392, 620093700, 300903898,
              56951292, 981135289, 68759784, 509068840, 711491106, 271441426,
              787230078, 213985365, 367539026, 649777546, 14327469, 165756548,
              188835561, 186114167, 352308105, 720716207, 681055509, 975924231)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "invalid color")
    return a


def _orient(a, transform):
    return np.rot90(np.fliplr(a) if transform >= 4 else a, transform % 4).copy()


def _key(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, cells = [start], []
        while stack:
            y, x = stack.pop()
            cells.append((int(y), int(x)))
            for q in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _copy_cells(hub, tiles):
    result = set()
    for tile in tiles:
        dr, dc = tile['direction']
        y = hub[0] + 2 * tile['radius'] * dr
        x = hub[1] + 2 * tile['radius'] * dc
        result.update((y+dy, x+dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    return result


def _probes(shape, occupied, hub, tiles):
    """All uniquely attributable clear cardinal probes, without signal colors.

    Endpoint exclusion reserves every potential echo footprint. Markers thus
    survive even when a footprint touches the canvas boundary.
    """
    h, w = shape
    forbidden = _copy_cells(hub, tiles)
    endpoints = {}
    for index, tile in enumerate(tiles):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            for slot in range(3):
                ly, lx = ((0 if dr < 0 else 2), slot) if dr else (slot, (0 if dc < 0 else 2))
                y, x = tile['top'] + ly + dr, tile['left'] + lx + dc
                corridor = []
                while 0 <= y < h and 0 <= x < w:
                    corridor.append((y, x))
                    y, x = y + dr, x + dc
                if len(corridor) < 2 or any(p in occupied for p in corridor):
                    continue
                endpoint = corridor[-1]
                if endpoint in forbidden:
                    continue
                entry = dict(tile=index, port=int(tile['labels'][ly, lx]), world_port=3*ly+lx,
                             endpoint=endpoint, direction=(dr, dc), corridor=tuple(corridor))
                endpoints.setdefault(endpoint, []).append(entry)
    return {p: entries[0] for p, entries in endpoints.items() if len(entries) == 1}


def _parse(value):
    a = _grid(value)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    _require(int(counts.max()) > a.size // 2, "background must occupy a strict majority")
    h, w = a.shape
    interior = a != bg
    boundary = np.zeros(a.shape, dtype=bool)
    boundary[0] = boundary[-1] = True
    boundary[:, 0] = boundary[:, -1] = True
    markers = [tuple(map(int, p)) for p in np.argwhere(interior & boundary)]
    interior = interior & ~boundary
    singletons, tiles = [], []
    for cells in _components(interior):
        if len(cells) == 1:
            singletons.append(cells[0])
            continue
        ys, xs = zip(*cells)
        y, x = min(ys), min(xs)
        _require(len(cells) == 9 and max(ys)-y == 2 and max(xs)-x == 2,
                 "source stamps must be complete separated 3x3 blocks")
        tiles.append(dict(top=y, left=x, array=a[y:y+3, x:x+3].copy(), cells=set(cells)))
    _require(len(singletons) == 1, "exactly one interior one-pixel hub is required")
    _require(3 <= len(tiles) <= 8 and markers, "expected 3..8 stamps and boundary probes")
    hub = singletons[0]
    _require(0 < hub[0] < h-1 and 0 < hub[1] < w-1, "hub must be interior")
    carrier = int(a[hub])
    canonical = tiles[0]['array']
    forms = [_orient(canonical, t) for t in range(8)]
    _require(len({t.tobytes() for t in forms}) == 8, "template must have eight distinct D4 forms")
    structural = set(map(int, canonical.flat))
    _require(2 <= len(structural) <= 4 and bg not in structural, "invalid structural palette")
    _require(int(np.count_nonzero(canonical == carrier)) >= 5, "hub color must dominate the stamp")
    occupied, directions = {hub}, set()
    for tile in tiles:
        matches = [t for t, form in enumerate(forms) if np.array_equal(form, tile['array'])]
        _require(len(matches) == 1, "all stamps must be exact transformed copies")
        tile['transform'] = matches[0]
        tile['labels'] = _orient(np.arange(9).reshape(3, 3), matches[0])
        dy, dx = tile['top']+1-hub[0], tile['left']+1-hub[1]
        _require((dy != 0 or dx != 0) and (dy == 0 or dx == 0 or abs(dy) == abs(dx)),
                 "stamp is not on a compass spoke")
        d = max(abs(dy), abs(dx))
        direction = (int(np.sign(dy)), int(np.sign(dx)))
        _require(4 <= d <= 6 and direction not in directions, "invalid radius or repeated spoke")
        directions.add(direction)
        tile['direction'], tile['radius'] = direction, d
        dr, dc = direction
        tile['inward'] = int(tile['labels'][1-dr, 1-dc])
        tile['outward'] = int(tile['labels'][1+dr, 1+dc])
        cy, cx = hub[0]+2*d*dr, hub[1]+2*d*dc
        _require(1 <= cy < h-1 and 1 <= cx < w-1, "potential echo exceeds canvas")
        occupied.update(tile['cells'])
    copies = _copy_cells(hub, tiles)
    _require(len(copies) == 9*len(tiles) and not copies.intersection(occupied),
             "source/echo footprints must be disjoint")
    _require(all(int(a[p]) == bg for p in copies), "potential echo footprints must be empty")
    _require(all(int(a[hub[0]+dr, hub[1]+dc]) == bg for dr, dc in DIRECTIONS),
             "hub neighbors must initially be empty")
    options = _probes(a.shape, occupied, hub, tiles)
    code, observations = {}, []
    for p in markers:
        _require(p in options, "boundary probe has no unique clear stamp alignment")
        entry = options[p]
        _require(all(int(a[q]) == bg for q in entry['corridor'][:-1]), "probe corridor is blocked")
        color = int(a[p])
        _require(color not in structural and color != bg, "signal and structural colors must differ")
        port = entry['port']
        _require(port not in code or code[port] == color, "conflicting observations of a local port")
        code[port] = color
        observations.append(entry)
    _require(1 <= len(code) <= 5 and 4 not in code, "expected 1..5 activated perimeter ports")
    _require(any(t['inward'] in code for t in tiles), "at least one inward port must be active")
    return dict(array=a, background=bg, carrier=carrier, hub=hub, tiles=tiles,
                code=code, observations=observations, structural=structural)


def _render(a, hub, tiles, code, sample='inward', halfturn=True, double=True):
    out = a.copy()
    for tile in tiles:
        port = tile[sample]
        if port not in code:
            continue
        dr, dc = tile['direction']
        out[hub[0]+dr, hub[1]+dc] = code[port]
        factor = 2 if double else 1
        cy, cx = hub[0]+factor*tile['radius']*dr, hub[1]+factor*tile['radius']*dc
        stamp = tile['array'][::-1, ::-1] if halfturn else tile['array']
        out[cy-1:cy+2, cx-1:cx+2] = stamp
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from visible input pixels, independent of generation or the seed."""
    info = _parse(input_grid)
    return _render(info['array'], info['hub'], info['tiles'], info['code'])


def _features(info):
    a, tiles, code = info['array'], info['tiles'], info['code']
    active = [i for i, t in enumerate(tiles) if t['inward'] in code]
    inactive = [i for i, t in enumerate(tiles) if t['inward'] not in code]
    allocations = [set() for _ in tiles]
    observers = {}
    for entry in info['observations']:
        allocations[entry['tile']].add(entry['port'])
        observers.setdefault(entry['port'], set()).add(entry['tile'])
    inward_values = [int(t['array'][1-t['direction'][0], 1-t['direction'][1]]) for t in tiles]
    # A concrete orientation-blind alternative: pool the observed world-facing
    # cell coordinates directly. This test is invariant to the arbitrary choice
    # of canonical stamp and to rotations/reflections of the entire grid.
    world_code, world_conflict = {}, False
    for entry in info['observations']:
        p, color = entry['world_port'], code[entry['port']]
        world_conflict |= p in world_code and world_code[p] != color
        world_code[p] = color
    canonical_sensitive = world_conflict or any(
        code.get(t['inward']) != world_code.get((1-t['direction'][0])*3+1-t['direction'][1])
        for t in tiles)
    used_ports = {tiles[i]['inward'] for i in active}
    result = dict(shape=a.shape, source_count=len(tiles), active_count=len(active),
                  port_count=len(code), clue_count=len(info['observations']),
                  background=info['background'], carrier=info['carrier'],
                  structural_count=len(info['structural']),
                  signal_count=len(set(code.values())),
                  radii=tuple(sorted({t['radius'] for t in tiles})),
                  active_radii=tuple(sorted({tiles[i]['radius'] for i in active})),
                  directions=tuple(sorted(t['direction'] for t in tiles)),
                  transforms=tuple(t['transform'] for t in tiles),
                  pooled=all(len(s) < len(code) for s in allocations),
                  pooled_used=all(not used_ports.issubset(s) for s in allocations),
                  complete_on_one=any(len(s) == len(code) for s in allocations),
                  remote=any(i not in observers[t['inward']] for i, t in enumerate(tiles)
                             if t['inward'] in code),
                  repeated_signal=len(set(code.values())) < len(code),
                  redundant=len(info['observations']) > len(code),
                  active_cardinal=any(0 in tiles[i]['direction'] for i in active),
                  active_diagonal=any(0 not in tiles[i]['direction'] for i in active),
                  raw_color_collision=any(inward_values[i] == inward_values[j] for i in active for j in inactive),
                  canonical_sensitive=canonical_sensitive,
                  inward_sensitive=any(code.get(t['inward']) != code.get(t['outward']) for t in tiles))
    return result


def _witness(f):
    return (f['active_count'] >= 2 and f['active_count'] < f['source_count']
            and f['port_count'] >= 2 and f['signal_count'] >= 2
            and f['pooled'] and f['pooled_used'] and f['remote'] and f['active_cardinal'] and f['active_diagonal']
            and len(f['active_radii']) >= 2 and f['raw_color_collision']
            and f['canonical_sensitive'] and f['inward_sensitive'])


def _make_template(rng, structural):
    for _ in range(48):
        tile = np.full((3, 3), structural[0], dtype=np.int8)
        count = int(rng.integers(max(2, len(structural)-1), 5))
        positions = rng.choice(9, size=count, replace=False)
        colors = list(structural[1:])
        colors += list(map(int, rng.choice(structural[1:], size=count-len(colors))))
        rng.shuffle(colors)
        for p, c in zip(positions, colors):
            tile.flat[int(p)] = c
        if len({_orient(tile, t).tobytes() for t in range(8)}) == 8:
            return tile
    tile = np.full((3, 3), structural[0], dtype=np.int8)
    tile[0, 0], tile[0, 1] = structural[1], structural[1]
    for index, c in enumerate(structural[2:]):
        tile[(1, 2) if index == 0 else (2, 1)] = c
    _require(len({_orient(tile, t).tobytes() for t in range(8)}) == 8, "fallback template lost asymmetry")
    return tile


def _base(rng, template, directions, radii, transforms, bg, fixed_shape=False):
    ey = max(2*d*abs(v[0])+1 for d, v in zip(radii, directions))
    ex = max(2*d*abs(v[1])+1 for d, v in zip(radii, directions))
    h = 30 if fixed_shape else int(rng.integers(2*ey+1, 31))
    w = 30 if fixed_shape else int(rng.integers(2*ex+1, 31))
    hub = (int(rng.integers(ey, h-ey)), int(rng.integers(ex, w-ex)))
    a = np.full((h, w), bg, dtype=np.int8)
    carrier_values, carrier_counts = np.unique(template, return_counts=True)
    carrier = int(carrier_values[np.argmax(carrier_counts)])
    a[hub] = carrier
    tiles, occupied = [], {hub}
    for direction, d, transform in zip(directions, radii, transforms):
        dr, dc = direction
        y, x = hub[0]+d*dr-1, hub[1]+d*dc-1
        stamp, labels = _orient(template, transform), _orient(np.arange(9).reshape(3, 3), transform)
        a[y:y+3, x:x+3] = stamp
        cells = {(y+dy, x+dx) for dy in range(3) for dx in range(3)}
        occupied.update(cells)
        tiles.append(dict(top=y, left=x, array=stamp, labels=labels, cells=cells,
                          direction=direction, radius=d, transform=transform,
                          inward=int(labels[1-dr, 1-dc]), outward=int(labels[1+dr, 1+dc])))
    return a, hub, tiles, _probes(a.shape, occupied, hub, tiles)


def _choose_clues(rng, code, options, witness):
    by_port = {port: [e for e in options.values() if e['port'] == port] for port in code}
    if any(not entries for entries in by_port.values()):
        return None
    # Sometimes observe the whole code on one stamp; witness examples pool it.
    common = set.intersection(*({e['tile'] for e in entries} for entries in by_port.values()))
    full_tile = int(rng.choice(sorted(common))) if common and not witness and rng.random() < .35 else None
    for _ in range(32):
        selected = []
        for port, entries in by_port.items():
            choices = [e for e in entries if full_tile is None or e['tile'] == full_tile]
            selected.append(choices[int(rng.integers(len(choices)))])
        if witness:
            allocations = {}
            for e in selected:
                allocations.setdefault(e['tile'], set()).add(e['port'])
            if any(len(s) == len(code) for s in allocations.values()):
                continue
        if not witness and rng.random() < .35:
            extra = [e for entries in by_port.values() for e in entries
                     if e['endpoint'] not in {s['endpoint'] for s in selected}]
            rng.shuffle(extra)
            selected += extra[:int(rng.integers(1, 4))]
        return selected
    return None


def _finish(a, hub, tiles, code, clues):
    a = a.copy()
    for entry in clues:
        a[entry['endpoint']] = code[entry['port']]
    return dict(input=a, output=_render(a, hub, tiles, code))


def _fallback(rng, pool, required_radius, sparse, palette):
    """A spacious member of the same family; no special solving convention."""
    bg, carrier, s1, s2, signal1, signal2 = palette[:6]
    template = np.full((3, 3), carrier, dtype=np.int8)
    template[0, 0], template[0, 1] = s1, s2
    directions = list(DIRECTIONS)
    if sparse:
        directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)]
    transform_map = {(-1, 0): 2, (0, 1): 1}
    radii = [pool[i % len(pool)] for i in range(len(directions))]
    radii[directions.index((-1, 0))] = required_radius or pool[0]
    radii[directions.index((1, 0))] = pool[-1]
    # Distinct active radii are guaranteed for the training fallback.
    if required_radius is None:
        radii[directions.index((-1, 0))] = pool[0]
    base, hub, tiles, options = _base(rng, template, directions, radii,
                                     [transform_map.get(d, 0) for d in directions], bg, True)
    code = {1: signal1, 6: signal2}
    clues = _choose_clues(rng, code, options, not sparse and required_radius is None)
    _require(clues is not None, "fallback probes unavailable")
    return _finish(base, hub, tiles, code, clues)


def _new_pair(rng, pool, required_radius=None, witness=False, source_range=(3, 8)):
    for _ in range(256):
        palette = list(map(int, rng.permutation(10)))
        structural_count = int(rng.integers(2, 5))
        template = _make_template(rng, palette[1:1+structural_count])
        n = int(rng.integers(source_range[0], source_range[1]+1))
        directions = [DIRECTIONS[int(i)] for i in rng.choice(8, n, replace=False)]
        if not any(0 in d for d in directions) or not any(0 not in d for d in directions):
            continue
        radii = list(map(int, rng.choice(pool, n)))
        if required_radius is not None:
            radii[int(rng.integers(n))] = required_radius
        transforms = list(map(int, rng.integers(8, size=n)))
        a, hub, tiles, options = _base(rng, template, directions, radii, transforms, palette[0])
        available = sorted({e['port'] for e in options.values()})
        count = int(rng.integers(2 if witness else 1, min(5, len(available))+1))
        ports = list(map(int, rng.choice(available, count, replace=False)))
        signal_palette = palette[1+structural_count:]
        if witness:
            colors = signal_palette[:count]
        else:
            color_count = int(rng.integers(1, min(count, len(signal_palette))+1))
            colors = signal_palette[:color_count]
            colors += list(map(int, rng.choice(colors, count-color_count)))
            rng.shuffle(colors)
        code = dict(zip(ports, colors))
        active = [t for t in tiles if t['inward'] in code]
        if len(active) < 2 or len(active) == n:
            continue
        if required_radius is not None and not any(t['radius'] == required_radius for t in active):
            continue
        clues = _choose_clues(rng, code, options, witness)
        if clues is None:
            continue
        pair = _finish(a, hub, tiles, code, clues)
        info = _parse(pair['input'])
        if witness and not _witness(_features(info)):
            continue
        return pair
    pair = _fallback(rng, pool, required_radius, source_range[1] <= 5, list(map(int, rng.permutation(10))))
    _require(not witness or _witness(_features(_parse(pair['input']))), "fallback lacks training evidence")
    return pair


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must be in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n = _count(num_train, 'num_train', 2, 10)
    m = _count(num_test, 'num_test', 1, 4)
    if n <= m:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    # Expansion, contraction, or fewer-spoke queries. The rule is identical.
    train_pool = (4, 5) if regime == 0 else (5, 6) if regime == 1 else (4, 5, 6)
    test_pool = (4, 5, 6) if regime == 0 else (4, 5) if regime == 1 else (4, 5, 6)
    required = 6 if regime == 0 else 4 if regime == 1 else None
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', n), ('test', m)):
        for index in range(count):
            witness = split == 'train' and index < 2
            source_range = (6, 8) if witness or (regime == 2 and split == 'train') else (3, 5) if regime == 2 else (3, 8)
            for _ in range(16):
                pair = _new_pair(rng, train_pool if split == 'train' else test_pool,
                                 required if split == 'test' else None, witness, source_range)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input construction exhausted')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "invalid pair keys")
    a, b = _grid(pair['input']), _grid(pair['output'])
    _require(a.shape == b.shape, "output dimensions changed")
    info = _parse(a)
    hub, bg, code = info['hub'], info['background'], info['code']
    assigned = np.zeros(a.shape, dtype=bool)
    active_count = 0
    # Scalar correspondence check, separate from _render's slice construction.
    for tile in info['tiles']:
        if tile['inward'] not in code:
            continue
        active_count += 1
        dr, dc = tile['direction']
        q = (hub[0]+dr, hub[1]+dc)
        assigned[q] = True
        _require(int(b[q]) == code[tile['inward']], "wrong hub signal")
        cy, cx = hub[0]+2*tile['radius']*dr, hub[1]+2*tile['radius']*dc
        for y in range(3):
            for x in range(3):
                destination = (cy+1-y, cx+1-x)
                _require(not assigned[destination], "echo destinations overlap")
                assigned[destination] = True
                _require(b[destination] == a[tile['top']+y, tile['left']+x], "wrong half-turn echo")
    _require(np.array_equal(a[~assigned], b[~assigned]), "source, inactive spoke, marker or unused background changed")
    _require(bool(np.all(a[assigned] == bg)), "completion must write initially empty cells")
    _require(int(assigned.sum()) == 10*active_count == int(np.count_nonzero(a != b)), "wrong change count")
    _require(np.array_equal(np.unique(a), np.unique(b)), "palette must be preserved")
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "splits must be lists")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "invalid example counts")
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(info['array'])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            f = _features(info)
            _require(2 <= f['active_count'] < f['source_count'], "generated examples need active and inactive spokes")
            features[split].append(f)
    training, queries = features['train'], features['test']
    _require(sum(_witness(f) for f in training) >= 2, "two informative witness demonstrations are required")
    train_radii = set().union(*(set(f['radii']) for f in training))
    expansion = 6 not in train_radii and all(6 in f['active_radii'] for f in queries)
    contraction = 4 not in train_radii and all(4 in f['active_radii'] for f in queries)
    sparse = all(6 <= f['source_count'] <= 8 for f in training) and all(3 <= f['source_count'] <= 5 for f in queries)
    _require(expansion or contraction or sparse, "missing demonstrated scale/occupancy shift")


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role])
                digest.update(bytes(a.shape))
                digest.update(a.astype(np.int8).tobytes())
    return digest.hexdigest()


def self_test():
    """Bounded streaming audit: 42 seeds; no retained puzzle corpus or files."""
    signatures, shapes, structures, layouts = set(), set(), set(), set()
    counts, ports, palettes, radii, active_counts = set(), set(), set(), set(), set()
    totals = dict(pooled=0, pooled_used=0, complete_on_one=0, repeated_signal=0, redundant=0,
                  remote=0, canonical_sensitive=0, inward_sensitive=0)
    split_shapes = {'train': set(), 'test': set()}
    shift_counts = dict(expansion=0, contraction=0, sparse=0)
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        _require(_digest(puzzle) == _digest(generate_puzzle(seed)), "seed is not reproducible")
        signatures.add(_digest(puzzle))
        features = {'train': [], 'test': []}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                saved = a.copy()
                _require(np.array_equal(solve(a), pair['output']) and np.array_equal(a, saved), "solver failure or input mutation")
                info, f = _parse(a), _features(_parse(a))
                features[split].append(f)
                shapes.add(a.shape)
                split_shapes[split].add(a.shape)
                counts.add(f['source_count'])
                active_counts.add(f['active_count'])
                ports.add(f['port_count'])
                radii.update(f['radii'])
                palettes.add((f['background'], f['carrier']))
                layouts.add((a.shape, info['hub'], f['directions'], tuple(t['radius'] for t in info['tiles'])))
                roles = {}
                structures.add((a.shape, bytes(roles.setdefault(int(c), len(roles)) for c in a.flat)))
                for name in totals:
                    totals[name] += bool(f[name])
                _require(not np.array_equal(_render(a, info['hub'], info['tiles'], info['code'], halfturn=False), pair['output']),
                         "half-turn is never observable")
        train_radii = set().union(*(set(f['radii']) for f in features['train']))
        query_active = set().union(*(set(f['active_radii']) for f in features['test']))
        expansion = 6 not in train_radii and 6 in query_active
        contraction = 4 not in train_radii and 4 in query_active
        sparse = max(f['source_count'] for f in features['test']) < min(f['source_count'] for f in features['train'])
        _require(expansion or contraction or sparse, "missing meaningful train/test shift")
        shift_counts['expansion'] += expansion
        shift_counts['contraction'] += contraction
        shift_counts['sparse'] += sparse
    _require(len(signatures) == 42 and len(structures) >= 180, "structural diversity collapsed")
    _require(len(shapes) >= 35 and len(layouts) >= 180, "geometric variation collapsed")
    _require(counts == set(range(3, 9)) and ports == set(range(1, 6)) and radii == {4, 5, 6}, "supported counts/radii omitted")
    _require(len(active_counts) >= 4 and len(palettes) >= 50, "selection or palette variation missing")
    _require(min(totals.values()) >= 3 and min(shift_counts.values()) >= 5, "variation mode missing")
    _require(all(len(s) >= 20 for s in split_shapes.values()), "split shapes collapsed")
    # All explicit legal count combinations, including the maximum 14 pairs.
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            p = generate_puzzle(6000+10*n+m, n, m)
            _require(len(p['train']) == n and len(p['test']) == m, "counts ignored")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (4.0, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Equivariance for every global D4 operation, plus an arbitrary palette permutation.
    original = generate_puzzle(321, 2, 1)
    permutation = np.array([8, 2, 7, 0, 6, 9, 3, 1, 5, 4], dtype=np.int8)
    for t in range(8):
        changed = {s: [{role: _orient(permutation[np.asarray(g)], t) for role, g in pair.items()}
                       for pair in original[s]] for s in ('train', 'test')}
        validate_generated_puzzle(changed)
        for split in ('train', 'test'):
            for pair in changed[split]:
                _require(np.array_equal(solve(pair['input']), pair['output']), "D4/recoloring equivariance failed")
    # Relocating a probe to the same local port of another copy preserves the
    # decisions and echoes. Only the two visible boundary endpoints change.
    moved = False
    for pair in original['train'] + original['test']:
        info = _parse(pair['input'])
        occupied = {info['hub']}.union(*(t['cells'] for t in info['tiles']))
        options = _probes(info['array'].shape, occupied, info['hub'], info['tiles'])
        for old in info['observations']:
            alternatives = [e for e in options.values() if e['port'] == old['port']
                            and e['tile'] != old['tile'] and info['array'][e['endpoint']] == info['background']]
            if not alternatives:
                continue
            new = alternatives[0]
            a, b = pair['input'].copy(), pair['output'].copy()
            for grid in (a, b):
                grid[old['endpoint']] = info['background']
                grid[new['endpoint']] = info['code'][old['port']]
            _require(np.array_equal(solve(a), b), "equivalent probe relocation changed growth")
            moved = True
            break
        if moved:
            break
    _require(moved, "no probe relocation witness found")
    # A signal-color change propagates precisely to the matching hub cells;
    # source patterns and entire echo blocks retain their structural colors.
    pair = original['test'][0]
    info = _parse(pair['input'])
    port = next(t['inward'] for t in info['tiles'] if t['inward'] in info['code'])
    color = next(c for c in range(10) if c not in info['structural'] | {info['background'], info['code'][port]})
    a, b = pair['input'].copy(), pair['output'].copy()
    for entry in info['observations']:
        if entry['port'] == port:
            a[entry['endpoint']] = b[entry['endpoint']] = color
    for tile in info['tiles']:
        if tile['inward'] == port:
            dr, dc = tile['direction']
            b[info['hub'][0]+dr, info['hub'][1]+dc] = color
    _require(np.array_equal(solve(a), b), "port recoloring did not propagate exactly")
    # Forced construction fallback and mutations of every distinct output role.
    rng = np.random.default_rng(517)
    for pool, required, sparse in (((4, 5), None, False), ((5, 6), None, False),
                                   ((4, 5, 6), 6, False), ((4, 5), 4, False),
                                   ((4, 5, 6), None, True)):
        pair = _fallback(rng, pool, required, sparse, list(range(10)))
        _check_pair(pair)
        _require(np.array_equal(solve(pair['input']), pair['output']), "fallback solve failed")
        if required is None and not sparse:
            _require(_witness(_features(_parse(pair['input']))), "fallback witness failed")
    info = _parse(original['test'][0]['input'])
    selected = next(t for t in info['tiles'] if t['inward'] in info['code'])
    dr, dc = selected['direction']
    points = [info['hub'], (info['hub'][0]+dr, info['hub'][1]+dc),
              (selected['top'], selected['left']), info['observations'][0]['endpoint'],
              (info['hub'][0]+2*selected['radius']*dr, info['hub'][1]+2*selected['radius']*dc)]
    inactive = next(t for t in info['tiles'] if t['inward'] not in info['code'])
    dy, dx = inactive['direction']
    points.append((info['hub'][0]+2*inactive['radius']*dy, info['hub'][1]+2*inactive['radius']*dx))
    unused = np.argwhere((original['test'][0]['input'] == info['background'])
                         & (original['test'][0]['output'] == info['background']))[0]
    points.append(tuple(map(int, unused)))
    for point in points:
        broken = {s: [{k: v.copy() for k, v in p.items()} for p in original[s]] for s in original}
        b = broken['test'][0]['output']
        b[point] = (int(b[point])+1) % 10
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            pass
        else:
            raise AssertionError('corrupted output accepted')
    return dict(seeds=42, pairs=210, shapes=len(shapes), normalized_structures=len(structures),
                layouts=len(layouts), source_counts=sorted(counts), port_counts=sorted(ports),
                active_counts=sorted(active_counts), radii=sorted(radii), palette_roles=len(palettes),
                feature_pairs=totals, train_test_shifts=shift_counts,
                semantic_verification='generator-owned checks only')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
