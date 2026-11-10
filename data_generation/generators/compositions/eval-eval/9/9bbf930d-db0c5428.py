"""Assemble a rosette from donors selected and oriented by paired-rail routes.

Only NumPy and the standard library are used. No parent code is imported.
"""
from __future__ import annotations

import json

import numpy as np


# Clockwise compass directions. Coordinates, rather than literal colors, give roles.
DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _need(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _need(bool(np.all((a >= 0) & (a <= 9))), "ARC colors required")
    return a


def _add(p, d, n=1):
    return p[0] + n * d[0], p[1] + n * d[1]


def _inside(a, p):
    return 0 <= p[0] < a.shape[0] and 0 <= p[1] < a.shape[1]


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _read_objects(value):
    a = _grid(value)
    bg = int(a[0, 0])
    _need(all(np.all(edge == bg) for edge in (a[0], a[-1], a[:, 0], a[:, -1])),
          "input has a clear background border")
    isolated = []
    for r, c in np.argwhere(a != bg):
        r, c = int(r), int(c)
        if 0 < r < len(a) - 1 and 0 < c < a.shape[1] - 1:
            window = a[r - 1:r + 2, c - 1:c + 2]
            if np.count_nonzero(window != bg) == 1:
                isolated.append((r, c))
    _need(len(isolated) == 1, "exactly one isolated collector is required")
    hub = isolated[0]
    carrier = int(a[hub])
    donors = {}
    occupied = set()
    for r, c in np.argwhere(a == carrier):
        r, c = int(r), int(c)
        if r < 1 or c < 1 or r + 1 >= len(a) or c + 1 >= a.shape[1]:
            continue
        t = a[r - 1:r + 2, c - 1:c + 2]
        if np.all(t != bg):
            _need(np.count_nonzero(t == carrier) >= 5, "donor carrier majority")
            center = r, c
            cells = {(r + y, c + x) for y in (-1, 0, 1) for x in (-1, 0, 1)}
            _need(not cells & occupied, "donors overlap or have ambiguous centers")
            occupied.update(cells)
            donors[center] = t.copy()
    _need(1 <= len(donors) <= 4, "one to four donors required")
    _need(sum(int(v == carrier) for v in a.flat) ==
          1 + sum(int(np.count_nonzero(t == carrier)) for t in donors.values()),
          "carrier occurs only in donors and collector")
    return dict(grid=a, background=bg, carrier=carrier, hub=hub, donors=donors,
                donor_cells=occupied)


def _flanks(info, p, d):
    a, bg, carrier = info['grid'], info['background'], info['carrier']
    n = -d[1], d[0]
    q, z = _add(p, n), _add(p, n, -1)
    if not (_inside(a, q) and _inside(a, z)):
        return None
    v, w = int(a[q]), int(a[z])
    if v in (bg, carrier) or w in (bg, carrier):
        return None
    return v, w


def _trace(info, port):
    """Directional reading: keep going across gaps; turn only when blocked."""
    a, bg, hub = info['grid'], info['background'], info['hub']
    d = DIRS[port]
    p = _add(hub, d, 2)
    _need(_inside(a, p) and a[p] == bg, "port center must be free")
    path, headings, seen = [p], [], set()
    while True:
        _need((p, d) not in seen, "cyclic route")
        seen.add((p, d))
        q = _add(p, d)
        _need(_inside(a, q), "a route must end at a donor, not outside the canvas")
        if a[q] == bg:
            p = q
            path.append(p)
            headings.append(d)
            continue
        # A donor is reached through the middle of a face, never a corner.
        center = _add(p, d, 2)
        if center in info['donors']:
            _need(q in info['donor_cells'], "donor face missing")
            break
        _need(q not in info['donor_cells'], "route hit a donor off center")
        options = []
        i = DIRS.index(d)
        for turn in (-1, 1):
            nd = DIRS[(i + turn) % 4]
            z = _add(p, nd)
            f = _flanks(info, z, nd)
            if _inside(a, z) and a[z] == bg and f is not None and f[0] == f[1]:
                options.append(nd)
        _need(len(options) == 1, "blocked route needs one visibly matched turn")
        d = options[0]
    _need(len(set(path)) == len(path), "route revisits itself")
    final = DIRS.index(d)
    tile = info['donors'][center]
    sample = int(tile[1 - d[0], 1 - d[1]])
    _need(sample != info['carrier'], "connected donor face must be marked")
    # np.rot90 rotates counterclockwise. Sample face -d must become outward DIRS[port].
    rotation = (final + 2 - port) % 4
    turns = sum(x != y for x, y in zip(headings, headings[1:]))
    supported = []
    for q, nd in zip(path[1:], headings):
        f = _flanks(info, q, nd)
        supported.append(f is not None and f[0] == f[1])
    gaps = 0
    for j in range(1, len(supported) - 1):
        if (not supported[j] and supported[j - 1] and
                headings[j - 1] == headings[j] == headings[j + 1]):
            gaps += 1
    return dict(port=port, center=center, path=path, final=final,
                relative=(final - port) % 4, sample=sample, rotation=rotation,
                tile=tile, turns=turns, gaps=gaps, length=len(path))


def _analyse(value, include_inactive=False):
    info = _read_objects(value)
    routes, inactive, present = [], [], []
    for port, d in enumerate(DIRS):
        p = _add(info['hub'], d, 2)
        f = _flanks(info, p, d)
        if f is None:
            continue
        _need(info['grid'][p] == info['background'], "blocked collector port")
        present.append(port)
        if f[0] == f[1]:
            route = _trace(info, port)
            route['gate'] = f[0]
            routes.append(route)
        else:
            inactive.append(_trace(info, port) if include_inactive else dict(port=port))
    info.update(routes=routes, inactive=inactive, present=present)
    return info


def _assemble(info, *, all_ports=False, no_rotation=False, wrong_sample=False):
    out = np.full((9, 9), info['background'], dtype=np.int8)
    out[4, 4] = info['carrier']
    routes = info['routes'] + (info['inactive'] if all_ports else [])
    for route in routes:
        dr, dc = DIRS[route['port']]
        tile = route['tile']
        sample = route['sample']
        if wrong_sample:
            sample = int(tile[1 - dr, 1 - dc])
        out[4 + dr, 4 + dc] = sample
        top, left = 3 + 3 * dr, 3 + 3 * dc
        out[top:top + 3, left:left + 3] = (
            tile if no_rotation else np.rot90(tile, route['rotation']))
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer collector, sources, gate equality, routes, samples and rotations from input."""
    return _assemble(_analyse(input_grid))


def _polyline(points):
    path = [points[0]]
    for target in points[1:]:
        p = path[-1]
        d = int(np.sign(target[0] - p[0])), int(np.sign(target[1] - p[1]))
        if (d[0] == 0) == (d[1] == 0):
            return None
        while p != target:
            p = _add(p, d)
            path.append(p)
    cells = set(path)
    if len(cells) != len(path):
        return None
    for i, p in enumerate(path):
        neighbors = sum(_add(p, d) in cells for d in DIRS)
        if neighbors != (1 if i in (0, len(path) - 1) else 2):
            return None
    return path


def _random_route(rng, shape, hub, port, relative, rich):
    # Enumerate turn sequences having the requested final heading. Lengths vary independently.
    base = (0, 1, 2, 1)[relative]
    bends = base + (2 if rich else 0)
    choices = []
    for mask in range(1 << bends):
        signs = [1 if mask & (1 << j) else -1 for j in range(bends)]
        if sum(signs) % 4 == relative:
            choices.append(signs)
    signs = choices[int(rng.integers(len(choices)))]
    current = _add(hub, DIRS[port], 2)
    points, heading = [current], port
    for j in range(bends + 1):
        direction = DIRS[heading]
        maximum = 10 if j == 0 else 8
        valid = []
        for length in range(3, maximum):
            q = _add(current, direction, length)
            if not (2 <= q[0] <= shape[0] - 3 and 2 <= q[1] <= shape[1] - 3):
                continue
            if j == bends:
                center = _add(q, direction, 2)
                if not (2 <= center[0] <= shape[0] - 3 and 2 <= center[1] <= shape[1] - 3):
                    continue
            valid.append(q)
        if not valid:
            return None
        current = valid[int(rng.integers(len(valid)))]
        points.append(current)
        if j < bends:
            heading = (heading + signs[j]) % 4
    path = _polyline(points)
    return None if path is None else (path, heading)


def _tile(rng, carrier, details, final, witness):
    d = DIRS[final]
    face = 1 - d[0], 1 - d[1]
    tile = np.full((3, 3), carrier, dtype=np.int8)
    tile[face] = int(rng.choice(details))
    cells = [(y, x) for y in range(3) for x in range(3)
             if (y, x) not in (face, (1, 1))]
    rng.shuffle(cells)
    count = int(rng.integers(0, 4))
    for q in cells[:count]:
        tile[q] = int(rng.choice(details))
    if witness:
        # A unique face mark and a distinct off-axis mark have eight distinct D4 views.
        sample, extra = map(int, rng.choice(details, 2, replace=False))
        for q in cells:
            candidate = np.full((3, 3), carrier, dtype=np.int8)
            candidate[face], candidate[q] = sample, extra
            if len({t.tobytes() for t in _d4(candidate)}) == 8:
                tile = candidate
                break
    return tile


def _d4(tile):
    return [np.rot90(tile, k) for k in range(4)] + [
        np.rot90(tile[:, ::-1], k) for k in range(4)]


def _place(rng, grid, hub, carrier, bg, rails, details, used, port,
           relative, active, rich, witness):
    candidate_path = _random_route(rng, grid.shape, hub, port, relative, rich)
    if candidate_path is None:
        return None
    path, final = candidate_path
    cells = set(path)
    d = DIRS[final]
    center = _add(path[-1], d, 2)
    tile_cells = {(center[0] + y, center[1] + x) for y in (-1, 0, 1) for x in (-1, 0, 1)}
    protected = {(hub[0] + y, hub[1] + x) for y in (-1, 0, 1) for x in (-1, 0, 1)}
    if (cells | tile_cells) & (used | protected):
        return None
    if any(grid[q] != bg for q in cells | tile_cells):
        return None
    color = int(rng.choice(rails))
    walls = {}
    for p in path:
        for nd in DIRS:
            q = _add(p, nd)
            if q in cells | tile_cells | protected:
                continue
            if not (1 <= q[0] < grid.shape[0] - 1 and 1 <= q[1] < grid.shape[1] - 1):
                return None
            walls[q] = color
    headings = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    # Remove aligned straight flanks, retaining first gate, every turn and final face approach.
    eligible = [j for j in range(3, len(path) - 3)
                if len(set(headings[j - 2:j + 2])) == 1]
    if eligible and rng.random() < .7:
        j = int(rng.choice(eligible))
        nd = headings[j]
        n = -nd[1], nd[0]
        walls.pop(_add(path[j], n), None)
        walls.pop(_add(path[j], n, -1), None)
    if not active:
        # The gate is geometrically identical; only equality of the two flanks changes.
        n = -DIRS[port][1], DIRS[port][0]
        other = int(rng.choice([v for v in rails if v != color]))
        walls[_add(path[0], n)] = other
        walls[_add(path[1], n)] = other
    if any(q in used or (grid[q] != bg and grid[q] != v) for q, v in walls.items()):
        return None
    tile = _tile(rng, carrier, details, final, witness)
    result = grid.copy()
    for q, v in walls.items():
        result[q] = v
    r, c = center
    result[r - 1:r + 2, c - 1:c + 2] = tile
    return result, used | cells | tile_cells, center, path


def _scene(rng, requests, rich, colors, shape=None):
    bg, carrier = colors[:2]
    rail_count = int(rng.integers(2, 5))
    rails, details = colors[2:2 + rail_count], colors[2 + rail_count:]
    if shape is None:
        shape = tuple(map(int, rng.integers(24 if len(requests) >= 3 else 20, 31, size=2)))
    hub = tuple(int(rng.integers(max(8, n // 2 - 2), min(n - 8, n // 2 + 3))) for n in shape)
    grid = np.full(shape, bg, dtype=np.int8)
    grid[hub] = carrier
    used, expected = set(), {}
    ports = list(map(int, rng.permutation(4)[:len(requests)]))
    # Place return/bent paths before straight ones to avoid easy routes consuming all space.
    order = sorted(range(len(requests)), key=lambda j: (requests[j][0] == 0, -requests[j][0]))
    for j in order:
        relative, active, witness = requests[j]
        port = ports[j]
        for _ in range(130):
            proposal = _place(rng, grid, hub, carrier, bg, rails, details, used,
                              port, relative, active, rich and active, witness)
            if proposal is None:
                continue
            candidate, reserved, center, path = proposal
            wanted = dict(expected)
            wanted[port] = (center, active, path)
            try:
                info = _analyse(candidate, include_inactive=True)
            except AssertionError:
                continue
            actual = {r['port']: (r['center'], True, r['path']) for r in info['routes']}
            actual.update({r['port']: (r['center'], False, r['path']) for r in info['inactive']})
            if actual != wanted or len(info['donors']) != len(wanted):
                continue
            grid, used, expected = candidate, reserved, wanted
            break
        else:
            return None
    info = _analyse(grid, include_inactive=True)
    # A color-blind shortcut must fail even in simple geometry.
    if np.array_equal(_assemble(info), _assemble(info, all_ports=True)):
        return None
    return dict(input=grid, output=_assemble(info))


def _counts(num_train, num_test):
    for v, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if (isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                or not low <= int(v) <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, m = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle, seen = dict(train=[], test=[]), set()
    base_colors = list(map(int, rng.permutation(10)))
    mix_colors = bool(rng.integers(0, 2))
    # The first two witnesses collectively expose all four relative arrival headings.
    relationships = list(map(int, rng.permutation(4)))
    profiles = [[(relationships[0], True, True), (relationships[1], True, True),
                 (int(rng.integers(4)), False, False)],
                [(relationships[2], True, True), (relationships[3], True, True),
                 (int(rng.integers(4)), False, False)]]
    if rng.random() < .6:
        profiles[int(rng.integers(2))].append(
            (int(rng.integers(4)), bool(rng.integers(2)), False))
    for split, count in (('train', n), ('test', m)):
        for index in range(count):
            if split == 'train' and index < 2:
                requests = profiles[index]
            else:
                donor_count = int(rng.integers(2, 4 if split == 'test' else 5))
                active_count = int(rng.integers(1, donor_count))
                requests = [(int(rng.integers(4)), j < active_count, False)
                            for j in range(donor_count)]
            # Tests combine extra bends with fewer donors: a compositional geometry shift.
            rich = split == 'test'
            colors = list(map(int, rng.permutation(10))) if mix_colors else base_colors
            for attempt in range(180):
                if attempt and attempt % 30 == 0:
                    # A randomly requested collection of returns may not pack.
                    # Resample compatible optional headings, retaining both mandatory
                    # witnesses, donor count, active count and the test bend regime.
                    keep = 2 if split == 'train' and index < 2 else 0
                    requests = requests[:keep] + [
                        (int(rng.integers(4)), active, witness)
                        for _, active, witness in requests[keep:]]
                pair = _scene(rng, requests, rich, colors)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError(f'bounded route packing exhausted: {seed}/{split}/{index}/{requests}')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    # Equality should be evidenced across more than one rail identity. Changing
    # only a matched gate's two flanks leaves its entire trajectory unchanged.
    gates = {r['gate'] for p in puzzle['train'] for r in _analyse(p['input'])['routes']}
    if len(gates) == 1:
        pair = puzzle['train'][0]
        info = _analyse(pair['input'], include_inactive=True)
        route = info['routes'][0]
        port = route['port']
        inactive_port = info['inactive'][0]['port']
        colors_at_inactive = _flanks(info, _add(info['hub'], DIRS[inactive_port], 2),
                                      DIRS[inactive_port])
        replacement = next(v for v in colors_at_inactive if v != route['gate'])
        normal = -DIRS[port][1], DIRS[port][0]
        for p in route['path'][:2]:
            pair['input'][_add(p, normal)] = replacement
            pair['input'][_add(p, normal, -1)] = replacement
        pair['output'] = solve(pair['input'])
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _check_pair(pair):
    _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
    a, b = _grid(pair['input']), _grid(pair['output'])
    _need(b.shape == (9, 9), "output is a 9x9 rosette")
    info = _analyse(a, include_inactive=True)
    _need(info['routes'] and info['inactive'], "matched and mismatched gates both required")
    routes = info['routes'] + info['inactive']
    _need(len(routes) == len(info['donors']), "each donor has exactly one route")
    _need(len({r['center'] for r in routes}) == len(routes), "donor used more than once")
    paths = [p for r in routes for p in r['path']]
    _need(len(paths) == len(set(paths)), "route centerlines must be disjoint")
    bg, carrier = info['background'], info['carrier']
    rail_mask = (a != bg) & (a != carrier)
    for center in info['donors']:
        r, c = center
        rail_mask[r - 1:r + 2, c - 1:c + 2] = False
    rail_values = set(map(int, np.unique(a[rail_mask])))
    detail_values = {int(v) for t in info['donors'].values() for v in t.flat if v != carrier}
    _need(not rail_values & detail_values, "rail and donor-detail palettes must be disjoint")
    assigned = np.zeros((9, 9), dtype=bool)
    assigned[4, 4] = True
    _need(b[4, 4] == carrier, "center carrier")
    # Scalar destination checks avoid using _assemble as the validator's answer oracle.
    for route in info['routes']:
        dr, dc = DIRS[route['port']]
        _need(b[4 + dr, 4 + dc] == route['sample'], "wrong routed face sample")
        assigned[4 + dr, 4 + dc] = True
        rot = route['rotation']
        top, left = 3 + 3 * dr, 3 + 3 * dc
        for y in range(3):
            for x in range(3):
                sy, sx = y, x
                for _ in range(rot):
                    sy, sx = sx, 2 - sy
                _need(b[top + y, left + x] == route['tile'][sy, sx], "wrong oriented donor copy")
                assigned[top + y, left + x] = True
        dy, dx = DIRS[route['final']]
        # The sample appears at the outward midpoint of the copied tile as well.
        _need(b[top + 1 + dr, left + 1 + dc] ==
              route['tile'][1 - dy, 1 - dx], "connected face must point outward")
    _need(np.all(b[~assigned] == bg), "unused slots and collector cells stay background")
    _need(np.count_nonzero(b != bg) == 1 + 10 * len(info['routes']), "rosette occupancy")
    _need(not np.array_equal(b, _assemble(info, all_ports=True)), "rail equality ablation must fail")
    return info


def _orientation_candidates(train):
    # Test all eight D4 operations independently for each relative arrival heading.
    # The relative heading determines the operation independently of absolute compass direction.
    candidates = [set(range(8)) for _ in range(4)]
    for pair in train:
        info = _analyse(pair['input'])
        b = np.asarray(pair['output'])
        for r in info['routes']:
            dr, dc = DIRS[r['port']]
            target = b[3 + 3 * dr:6 + 3 * dr, 3 + 3 * dc:6 + 3 * dc]
            tiles = _d4(r['tile'])
            candidates[r['relative']].intersection_update(
                k for k, tile in enumerate(tiles) if np.array_equal(tile, target))
    _need(all(candidates), "contradictory rotation demonstrations")
    return candidates


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _counts(n, m)
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(pair['input'])
            _need(key not in seen, "duplicate input")
            seen.add(key)
            if split == 'train':
                training.append(info)
    demonstrated = {r['relative'] for a in training for r in a['routes']}
    _need(demonstrated == set(range(4)), "demonstrate all four arrival relationships")
    _need(len({r['gate'] for a in training for r in a['routes']}) >= 2,
          "demonstrate equality on multiple rail identities")
    candidates = _orientation_candidates(puzzle['train'])
    for pair in puzzle['test']:
        info = _analyse(pair['input'])
        for r in info['routes']:
            images = _d4(r['tile'])
            _need(all(np.array_equal(images[k], np.rot90(r['tile'], r['rotation']))
                      for k in candidates[r['relative']]), "training leaves a D4 ambiguity")
    _need(any(not np.array_equal(_assemble(a), _assemble(a, wrong_sample=True))
              for a in training), "demonstrate sampling by donor face rather than slot direction")
    _need(any(not np.array_equal(_assemble(a), _assemble(a, no_rotation=True))
              for a in training), "demonstrate orientation rather than unchanged tile copying")


def measurements(puzzle):
    result = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _analyse(pair['input'], include_inactive=True)
            routes = info['routes'] + info['inactive']
            result.append(dict(split=split, shape=list(info['grid'].shape), hub=list(info['hub']),
                               background=info['background'], carrier=info['carrier'],
                               donors=len(routes), active=len(info['routes']),
                               ports=[r['port'] for r in routes],
                               relatives=[r['relative'] for r in routes],
                               turns=[r['turns'] for r in routes],
                               lengths=[r['length'] for r in routes],
                               gaps=sum(r['gaps'] for r in routes),
                               samples=[r['sample'] for r in info['routes']]))
    return result


def self_test():
    """Bounded 42-seed checks, mutations, equivariance, determinism and count contracts."""
    records, signatures, structures = [], set(), set()
    first = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in puzzle for p in puzzle[s]))
        records.extend(measurements(puzzle))
        for split in puzzle:
            for pair in puzzle[split]:
                a, b = pair['input'], pair['output']
                _need(np.array_equal(solve(a), b), "input-only solve")
                mapping = {}
                structures.add((a.shape, tuple(mapping.setdefault(int(v), len(mapping)) for v in a.flat)))
        if seed == 0:
            first = puzzle
    _need(len(signatures) == 42, "cross-seed puzzle diversity")
    _need(len(structures) >= 150, "structural variation beyond palette changes")
    _need(len({tuple(r['shape']) for r in records}) >= 30, "canvas variation")
    _need({r['donors'] for r in records} == {2, 3, 4}, "donor counts")
    _need({r['active'] for r in records} == {1, 2, 3}, "active counts")
    _need({v for r in records for v in r['relatives']} == set(range(4)), "arrival directions")
    _need({v for r in records for v in r['turns']} >= set(range(5)), "bend counts")
    _need(any(r['gaps'] for r in records), "interrupted rails")
    _need({r['background'] for r in records} == set(range(10)), "background role variation")
    _need({r['carrier'] for r in records} == set(range(10)), "carrier role variation")
    for pair in first['train'] + first['test']:
        for k in range(4):
            _need(np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k)),
                  "global rotation equivariance")
        recolor = np.array([7, 5, 2, 9, 0, 4, 1, 8, 6, 3], dtype=np.int8)
        _need(np.array_equal(solve(recolor[pair['input']]), recolor[pair['output']]),
              "palette equivariance")
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            p = generate_puzzle(73 + n * 5 + m, n, m)
            _need(len(p['train']) == n and len(p['test']) == m, "explicit counts")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    repeated = generate_puzzle(0)
    _need(all(_key(a[k]) == _key(b[k]) for s in first
              for a, b in zip(first[s], repeated[s]) for k in a), "whole-puzzle determinism")
    # Regression for a previously un-packable optional collection of three returns.
    generate_puzzle(548, 4, 1)
    # A connected-face edit must affect its central summary and one copied pixel.
    info = _analyse(first['train'][0]['input'], include_inactive=True)
    route = info['routes'][0]
    changed = first['train'][0]['input'].copy()
    nd = DIRS[route['final']]
    face = _add(route['center'], nd, -1)
    old = int(changed[face])
    alternatives = [int(v) for t in info['donors'].values() for v in t.flat
                    if v not in (info['carrier'], old)]
    _need(bool(alternatives), "sample mutation needs another visible detail")
    changed[face] = alternatives[0]
    revised = solve(changed)
    _need(np.count_nonzero(revised != first['train'][0]['output']) == 2,
          "face edit must change exactly summary and copy")
    # Removing the color comparison admits a previously inactive donor.
    changed = first['train'][0]['input'].copy()
    inactive = info['inactive'][0]
    port = inactive['port']
    p, nd = _add(info['hub'], DIRS[port], 2), DIRS[port]
    normal = -nd[1], nd[0]
    changed[_add(p, normal)] = changed[_add(p, normal, -1)]
    _need(np.array_equal(solve(changed), _assemble(info, all_ports=True))
          if len(info['inactive']) == 1 else not np.array_equal(solve(changed), _assemble(info)),
          "gate equality must alter admission")
    broken = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
    broken['test'][0]['output'][4, 4] = (int(broken['test'][0]['output'][4, 4]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt collector accepted')
    return dict(seeds=42, pairs=len(records), structural_variants=len(structures),
                shapes=len({tuple(r['shape']) for r in records}),
                donor_counts=sorted({r['donors'] for r in records}),
                bend_counts=sorted({v for r in records for v in r['turns']}))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
