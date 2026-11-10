"""Hidden-address atlas: sample local hubs, then dock their radial tile echoes.

Only Python/NumPy are used. Generation coordinates are discarded before the
input-only solver and demonstration-coverage checks run. No disk writes occur.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import itertools

import numpy as np


DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
              (0, 1), (1, -1), (1, 0), (1, 1))
HOST_SEEDS = (0, 75918821, 688978019, 747927043, 390730884, 160704925,
              22017618, 176669106, 213460375, 97397466, 526399870, 476140523,
              150337976, 318355672, 777267665, 599330501, 75429205, 629228829,
              661798782, 448284958, 715059248, 439387055, 945341179, 512211005,
              980055044, 174870214, 532458511, 366667491, 924949901, 624675493,
              928160064, 707700035, 666582137, 46039856, 714972453, 402919321,
              704167151, 409223292, 632350237, 505469833, 943811648, 225868875)


def _require(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grids required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "ARC colors required")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.astype(np.int8, copy=False).tobytes()


def _groups(points):
    remaining, result = set(points), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRECTIONS:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
                    group.add(p)
        result.append(group)
    return result


def _noncollinear(indices):
    return any(DIRECTIONS[i][0] * DIRECTIONS[j][1] !=
               DIRECTIONS[i][1] * DIRECTIONS[j][0] for i in indices for j in indices)


def _read_station(a, points, pin, carrier):
    centers = [p for p in points if int(a[p]) == pin]
    if len(centers) != 1:
        return None
    r, c = centers[0]
    used, ports = {(r, c)}, []
    for i, (dr, dc) in enumerate(DIRECTIONS):
        found = []
        for length in (3, 4, 5):
            box = {(r + length * dr + y, c + length * dc + x)
                   for y in (-1, 0, 1) for x in (-1, 0, 1)}
            ray = {(r + k * dr, c + k * dc) for k in range(1, length - 1)}
            if box <= points and ray <= points and all(int(a[p]) == carrier for p in ray):
                found.append((length, box, ray))
        if len(found) > 1:
            return None
        if found:
            length, box, ray = found[0]
            used.update(box | ray)
            top, left = r + length * dr - 1, c + length * dc - 1
            tile = a[top:top + 3, left:left + 3].copy()
            if not np.any(tile == carrier):
                return None
            ports.append((i, length, tile))
    if used != points or not 2 <= len(ports) <= 8 or not _noncollinear([p[0] for p in ports]):
        return None
    glyph = np.full((3, 3), carrier, dtype=np.int8)
    for i, length, tile in ports:
        dr, dc = DIRECTIONS[i]
        glyph[1 + dr, 1 + dc] = tile[1 - dr, 1 - dc]
    if np.count_nonzero(glyph != carrier) < 2:
        return None
    return dict(center=(r, c), ports=ports, glyph=glyph, points=points)


def _parse(input_grid):
    a = _grid(input_grid)
    colors, counts = np.unique(a, return_counts=True)
    _require(int(counts.max()) * 2 > a.size, "background must be a strict majority")
    bg = int(colors[counts.argmax()])
    parts = _groups(zip(*np.nonzero(a != bg)))
    _require(2 <= len(parts) <= 5, "need two to five stations")
    frequencies = [Counter(int(a[p]) for p in points) for points in parts]
    pin_candidates = set.intersection(*({v for v, n in f.items() if n == 1} for f in frequencies))
    solutions = []
    for pin in sorted(pin_candidates):
        for carrier in sorted(set(map(int, colors)) - {bg, pin}):
            stations = [_read_station(a, points, pin, carrier) for points in parts]
            if all(s is not None for s in stations):
                glyphs = [s['glyph'].tobytes() for s in stations]
                if len(set(glyphs)) == len(glyphs):
                    solutions.append((pin, carrier, stations))
    _require(len(solutions) == 1, "station geometry or visible roles are ambiguous")
    pin, carrier, stations = solutions[0]
    return dict(input=a, background=bg, pin=pin, carrier=carrier, stations=stations)


def _render_tiles(tiles, bg):
    top = min(p[0] for p in tiles) - 1
    left = min(p[1] for p in tiles) - 1
    h = max(p[0] for p in tiles) - top + 2
    w = max(p[1] for p in tiles) - left + 2
    _require(h <= 30 and w <= 30, "atlas exceeds ARC bounds")
    result = np.full((h, w), bg, dtype=np.int8)
    for (r, c), tile in tiles.items():
        result[r - top - 1:r - top + 2, c - left - 1:c - left + 2] = tile
    return result


def _assemble(info):
    stations = info['stations']
    addresses = {s['glyph'].tobytes(): i for i, s in enumerate(stations)}
    adjacency = [[] for _ in stations]
    edges = []
    for i, station in enumerate(stations):
        for d, length, tile in station['ports']:
            j = addresses.get(tile[::-1, ::-1].tobytes())
            if j is not None:
                _require(i != j, "a station cannot dock to itself")
                dr, dc = DIRECTIONS[d]
                delta = 2 * length * dr, 2 * length * dc
                adjacency[i].append((j, delta))
                adjacency[j].append((i, (-delta[0], -delta[1])))
                edges.append((i, j, d, length))
    offsets, stack = {0: (0, 0)}, [0]
    while stack:
        i = stack.pop()
        for j, delta in adjacency[i]:
            position = offsets[i][0] + delta[0], offsets[i][1] + delta[1]
            if j in offsets:
                _require(offsets[j] == position, "inconsistent docking cycle")
            else:
                offsets[j] = position
                stack.append(j)
    _require(len(offsets) == len(stations), "disconnected docking graph")
    _require(len(set(offsets.values())) == len(stations), "coincident station addresses")
    tiles = {}

    def put(p, tile):
        if p in tiles:
            _require(np.array_equal(tiles[p], tile), "unequal coincident tiles")
        else:
            _require(all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= 3 for q in tiles),
                     "partial tile collisions are outside the family")
            tiles[p] = tile.copy()

    for i, s in enumerate(stations):
        put(offsets[i], s['glyph'])
    for i, s in enumerate(stations):
        r, c = offsets[i]
        for d, length, tile in s['ports']:
            dr, dc = DIRECTIONS[d]
            put((r + 2 * length * dr, c + 2 * length * dc), tile[::-1, ::-1])
    info.update(offsets=offsets, edges=edges, tiles=tiles)
    return _render_tiles(tiles, info['background'])


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct solely from visible station shapes, samples and address matches."""
    return _assemble(_parse(input_grid))


def _views(tile):
    return tuple(np.rot90(tile, k) for k in range(4)) + tuple(
        np.rot90(tile[:, ::-1], k) for k in range(4))


def _coverage(train_infos, query_infos):
    """Sample-coordinate and per-direction D4 alternatives taught by training."""
    positions = [set(range(9)) for _ in DIRECTIONS]
    operations = [set(range(8)) for _ in DIRECTIONS]
    for info in train_infos:
        top = min(p[0] for p in info['tiles']) - 1
        left = min(p[1] for p in info['tiles']) - 1
        labeled = info['output']
        for i, s in enumerate(info['stations']):
            r, c = info['offsets'][i]
            for d, length, tile in s['ports']:
                dr, dc = DIRECTIONS[d]
                sample = int(labeled[r - top + dr, c - left + dc])
                positions[d].intersection_update(j for j in range(9) if tile.flat[j] == sample)
                er, ec = r + 2 * length * dr - top, c + 2 * length * dc - left
                echo = labeled[er - 1:er + 2, ec - 1:ec + 2]
                operations[d].intersection_update(k for k, t in enumerate(_views(tile))
                                                  if np.array_equal(t, echo))
    _require(all(positions) and all(operations), "contradictory training labels")
    missing = set()
    for info in query_infos:
        for s in info['stations']:
            for d, length, tile in s['ports']:
                dr, dc = DIRECTIONS[d]
                sample = int(s['glyph'][1 + dr, 1 + dc])
                if any(int(tile.flat[j]) != sample for j in positions[d]):
                    missing.add(d)
                echo = tile[::-1, ::-1]
                if any(not np.array_equal(_views(tile)[k], echo) for k in operations[d]):
                    missing.add(d)
    return missing


def _normalize(cells):
    r0 = min(p[0] for p in cells)
    c0 = min(p[1] for p in cells)
    return {(r - r0, c - c0): v for (r, c), v in cells.items()}


def _pack(parts, bg, rng):
    """Random shelf partitions/orientations; no fixed slots or adjacency encoding."""
    parts = [_normalize(p) for p in parts]
    shapes = [(max(r for r, c in p) + 1, max(c for r, c in p) + 1) for p in parts]
    total = sum(map(len, parts))
    orders = list(itertools.permutations(range(len(parts))))
    rng.shuffle(orders)
    breaks = list(range(1 << (len(parts) - 1)))
    rng.shuffle(breaks)
    axes = [0, 1]
    rng.shuffle(axes)
    for axis in axes:
        sizes = [s if axis == 0 else s[::-1] for s in shapes]
        for order in orders:
            for mask in breaks:
                rows = [[order[0]]]
                for k, i in enumerate(order[1:]):
                    if mask & (1 << k):
                        rows.append([])
                    rows[-1].append(i)
                heights = [max(sizes[i][0] for i in row) for row in rows]
                widths = [sum(sizes[i][1] for i in row) + len(row) - 1 for row in rows]
                h0, w0 = sum(heights) + len(rows) - 1, max(widths)
                if h0 > 30 or w0 > 30:
                    continue
                canvases = [(h, w) for h in range(h0, 31) for w in range(w0, 31)
                            if h * w > 2 * total]
                if not canvases:
                    continue
                h, w = canvases[int(rng.integers(len(canvases)))]
                result = np.full((h, w) if axis == 0 else (w, h), bg, dtype=np.int8)
                y = int(rng.integers(h - h0 + 1))
                for row, rh, rw in zip(rows, heights, widths):
                    x = int(rng.integers(w - rw + 1))
                    for i in row:
                        dy = int(rng.integers(rh - sizes[i][0] + 1))
                        for (r, c), value in parts[i].items():
                            p = (y + dy + r, x + c) if axis == 0 else (x + r, y + dy + c)
                            result[p] = value
                        x += sizes[i][1] + 1
                    y += rh + 1
                return result
    raise AssertionError("bounded shelf packer found no fit")


def _new_geometry(rng, n, lengths, cover):
    nodes, ports = [(12, 12)], [{} for _ in range(n)]
    for j in range(1, n):
        proposals = []
        for i, (r, c) in enumerate(nodes):
            for d, (dr, dc) in enumerate(DIRECTIONS):
                if d in ports[i]:
                    continue
                for length in lengths:
                    p = r + 2 * length * dr, c + 2 * length * dc
                    if all(0 <= x <= 24 for x in p) and all(
                            max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= 3 for q in nodes):
                        proposals.append((i, d, length, p))
        _require(bool(proposals), "node growth stuck")
        i, d, length, p = proposals[int(rng.integers(len(proposals)))]
        ports[i][d] = length
        nodes.append(p)
    endpoints = set(nodes)
    for i, (r, c) in enumerate(nodes):
        order = list(range(8))
        rng.shuffle(order)
        target = 8 if cover and i == 0 else int(rng.integers(3 if i == 0 else 2, 9))
        for d in order:
            if len(ports[i]) >= target:
                break
            if d in ports[i]:
                continue
            dr, dc = DIRECTIONS[d]
            options = list(lengths)
            rng.shuffle(options)
            for length in options:
                p = r + 2 * length * dr, c + 2 * length * dc
                if not all(0 <= x <= 24 for x in p):
                    continue
                if any(p != q and max(abs(p[0] - q[0]), abs(p[1] - q[1])) < 3
                       for q in endpoints):
                    continue
                ports[i][d] = length
                endpoints.add(p)
                break
        _require(2 <= len(ports[i]) <= 8 and _noncollinear(ports[i]), "insufficient local directions")
    return nodes, ports


def _make_glyphs(rng, nodes, ports, carrier, details):
    n = len(nodes)
    glyphs = [np.full((3, 3), carrier, dtype=np.int8) for _ in nodes]
    lookup = {p: i for i, p in enumerate(nodes)}
    for d, (dr, dc) in enumerate(DIRECTIONS):
        parent = list(range(n))

        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i

        def join(i, j):
            parent[find(i)] = find(j)

        same_endpoint = {}
        for i, (r, c) in enumerate(nodes):
            if d in ports[i]:
                length = ports[i][d]
                p = r + 2 * length * dr, c + 2 * length * dc
                if p in lookup:
                    join(i, lookup[p])
                if p in same_endpoint:
                    join(i, same_endpoint[p])
                same_endpoint[p] = i
        groups = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        for group in groups.values():
            value = int(rng.choice(details)) if all(d in ports[i] for i in group) else carrier
            for i in group:
                glyphs[i][1 + dr, 1 + dc] = value
    _require(all(np.count_nonzero(g != carrier) >= 2 for g in glyphs), "weak address")
    _require(len({g.tobytes() for g in glyphs}) == n, "duplicate addresses")
    return glyphs


def _leaf_tile(rng, constraints, carrier, details, forbidden):
    for _ in range(32):
        tile = np.full((3, 3), carrier, dtype=np.int8)
        kind = int(rng.integers(5))
        if kind == 0:
            locations = list(rng.choice(9, int(rng.integers(0, 7)), replace=False))
        elif kind == 1:
            axis, fixed = int(rng.integers(2)), int(rng.integers(3))
            locations = [r * 3 + c for r in range(3) for c in range(3) if (r, c)[axis] == fixed]
        elif kind == 2:
            locations = [0, 2, 6, 8] if rng.random() < .5 else [1, 3, 5, 7]
        elif kind == 3:
            locations = list(range(9))
        else:
            locations = [int(rng.integers(9))]
            locations += [8 - locations[0]]
        color = int(rng.choice(details))
        for flat in locations:
            tile.flat[flat] = int(rng.choice(details)) if kind in (0, 3) else color
        tile[1, 1] = carrier
        for p, value in constraints.items():
            tile[p] = value
        if tile.tobytes() not in forbidden:
            return tile
    raise AssertionError("accidental address could not be removed")


def _new_pair(rng, n, lengths, roles, cover=False, require_five=False):
    for _ in range(100):
        try:
            if roles is None:
                bg, carrier, pin = map(int, rng.choice(10, 3, replace=False))
            else:
                bg, carrier, pin = roles
            available = [v for v in range(10) if v not in (bg, carrier, pin)]
            details = list(map(int, rng.choice(available, int(rng.integers(2, 8)), replace=False)))
            nodes, ports = _new_geometry(rng, n, lengths, cover)
            if require_five and not any(5 in p.values() for p in ports):
                continue
            glyphs = _make_glyphs(rng, nodes, ports, carrier, details)
            tiles = {p: g for p, g in zip(nodes, glyphs)}
            constraints = {}
            for i, (r, c) in enumerate(nodes):
                for d, length in ports[i].items():
                    dr, dc = DIRECTIONS[d]
                    p = r + 2 * length * dr, c + 2 * length * dc
                    q = 1 + dr, 1 + dc
                    value = int(glyphs[i][q])
                    old = constraints.setdefault(p, {}).get(q, value)
                    _require(old == value, "incompatible sample constraints")
                    constraints[p][q] = value
            forbidden = {g.tobytes() for g in glyphs}
            for p in sorted(constraints):
                if p not in tiles:
                    tiles[p] = _leaf_tile(rng, constraints[p], carrier, details, forbidden)
                _require(all(int(tiles[p][q]) == value for q, value in constraints[p].items()),
                         "address/sample inconsistency")
            parts = []
            for i, (r, c) in enumerate(nodes):
                cells = {(0, 0): pin}
                for d, length in ports[i].items():
                    dr, dc = DIRECTIONS[d]
                    for k in range(1, length - 1):
                        cells[k * dr, k * dc] = carrier
                    p = r + 2 * length * dr, c + 2 * length * dc
                    tile = tiles[p][::-1, ::-1]
                    for y in (-1, 0, 1):
                        for x in (-1, 0, 1):
                            cells[length * dr + y, length * dc + x] = int(tile[1 + y, 1 + x])
                parts.append(cells)
            inp = _pack(parts, bg, rng)
            out = _render_tiles(tiles, bg)
            info = _parse(inp)
            predicted = _assemble(info)
            _require(np.array_equal(predicted, out), "input-only reconstruction differs from construction")
            _require(any(not np.array_equal(t, t[::-1, ::-1])
                         for s in info['stations'] for d, length, t in s['ports']), "no half-turn witness")
            info['output'] = out
            return dict(input=inp, output=out), info
        except AssertionError:
            continue
    raise RuntimeError("bounded station construction exhausted")


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError(f"{name} must be an integer")
        if not low <= value <= high:
            raise ValueError(f"{name} must be in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(seed)
    mode = int(rng.integers(2))
    roles = tuple(map(int, rng.choice(10, 3, replace=False))) if rng.random() < .35 else None
    train_lengths = (3, 4, 5) if mode == 0 else (3, 4)
    test_lengths = (3, 4, 5) if mode == 0 else (4, 5)
    train, train_infos, tests, query_infos = [], [], [], []
    for i in range(nt):
        n = 3 if i == 0 else int(rng.integers(2, 4 if mode == 0 else 5))
        p, info = _new_pair(rng, n, train_lengths, roles)
        train.append(p)
        train_infos.append(info)
    for i in range(nq):
        n = int(rng.integers(4, 6)) if mode == 0 else int(rng.integers(2, 5))
        p, info = _new_pair(rng, n, test_lengths, roles, require_five=mode == 1)
        tests.append(p)
        query_infos.append(info)
    for attempt in range(64):
        observed_lengths = {length for info in train_infos for s in info['stations']
                            for d, length, tile in s['ports']}
        if not _coverage(train_infos, query_infos) and len(observed_lengths) >= 2:
            break
        index = attempt % nt
        n = 3 if index == 0 else int(rng.integers(2, 4 if mode == 0 else 5))
        train[index], train_infos[index] = _new_pair(rng, n, train_lengths, roles, cover=attempt >= 16)
    else:
        raise RuntimeError("bounded demonstration coverage repair exhausted")
    rng.shuffle(train)
    result = dict(train=train, test=tests)
    seen = set()
    for split in ('train', 'test'):
        for pair in result[split]:
            _require(_key(pair['input']) not in seen, "duplicate generated input")
            seen.add(_key(pair['input']))
    return result


def _check_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
    a, b = _grid(pair['input']), _grid(pair['output'])
    info = _parse(a)
    expected = _assemble(info)
    _require(np.array_equal(expected, b), "incorrect atlas")
    # A second, scalar coordinate construction checks sampling, copying and crop.
    cells = {}

    def put(p, value):
        _require(p not in cells or cells[p] == value, "scalar collision")
        cells[p] = int(value)

    for i, s in enumerate(info['stations']):
        r, c = info['offsets'][i]
        for y in (-1, 0, 1):
            for x in (-1, 0, 1):
                put((r + y, c + x), int(s['glyph'][1 + y, 1 + x]))
        sr, sc = s['center']
        for d, length, tile in s['ports']:
            dr, dc = DIRECTIONS[d]
            _require(b is not a, "output aliases input")
            _require(int(s['glyph'][1 + dr, 1 + dc]) ==
                     int(a[sr + (length - 1) * dr, sc + (length - 1) * dc]), "wrong inward sample")
            for y in (-1, 0, 1):
                for x in (-1, 0, 1):
                    put((r + 2 * length * dr - y, c + 2 * length * dc - x),
                        a[sr + length * dr + y, sc + length * dc + x])
    cells = _normalize(cells)
    scalar = np.full((max(r for r, c in cells) + 1, max(c for r, c in cells) + 1),
                     info['background'], dtype=np.int8)
    for p, value in cells.items():
        scalar[p] = value
    _require(np.array_equal(scalar, b), "scalar geometry check failed")
    _require(info['pin'] not in b, "pin must disappear")
    _require(set(map(int, np.unique(a))) - {info['pin']} == set(map(int, np.unique(b))),
             "unexpected palette loss or invention")
    original = [s['center'] for s in info['stations']]
    _require(any((original[j][0] - original[i][0], original[j][1] - original[i][1]) !=
                 (info['offsets'][j][0] - info['offsets'][i][0],
                  info['offsets'][j][1] - info['offsets'][i][1]) for i, j, d, length in info['edges']),
             "input packing already solves docking")
    _require(len(info['tiles']) > len(info['stations']), "no non-address emissions")
    _require(any(not np.array_equal(t, t[::-1, ::-1])
                 for s in info['stations'] for d, length, t in s['ports']), "half-turn is vacuous")
    info['output'] = b
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "split lists required")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _check_pair(pair)
            key = _key(info['input'])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(info)
    _require(not _coverage(infos['train'], infos['test']), "query-relevant training ambiguity")
    lengths = {length for info in infos['train'] for s in info['stations'] for d, length, t in s['ports']}
    _require(len(lengths) >= 2, "doubled variable distances not demonstrated")
    return None


def _signature(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                a = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(a.shape))
                h.update(a.tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 seeds; retain only small signatures/counters, never puzzle corpora."""
    signatures, shapes, out_shapes, structures = set(), set(), set(), set()
    backgrounds, pins, carriers, counts, lengths, degrees = set(), set(), set(), set(), set(), set()
    cycles = branches = convergences = sample_repeats = 0
    shifts = set()
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_signature(puzzle))
        profiles = {'train': [], 'test': []}
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _check_pair(pair)
                profiles[split].append((len(info['stations']),
                                        {length for s in info['stations'] for d, length, t in s['ports']}))
                shapes.add(pair['input'].shape)
                out_shapes.add(pair['output'].shape)
                backgrounds.add(info['background'])
                pins.add(info['pin'])
                carriers.add(info['carrier'])
                counts.add(len(info['stations']))
                mapping = {}
                structures.add(hashlib.sha256(bytes(mapping.setdefault(int(v), len(mapping))
                                                     for v in pair['output'].flat)).hexdigest())
                undirected = {tuple(sorted((i, j))) for i, j, d, length in info['edges']}
                cycles += len(undirected) >= len(info['stations'])
                incidence = Counter(i for edge in undirected for i in edge)
                branches += any(n >= 3 for n in incidence.values())
                emissions = sum(len(s['ports']) for s in info['stations']) + len(info['stations'])
                convergences += emissions - len(info['tiles']) > len(info['stations']) - 1
                for s in info['stations']:
                    degrees.add(len(s['ports']))
                    for d, length, tile in s['ports']:
                        lengths.add(length)
                        dr, dc = DIRECTIONS[d]
                        sample_repeats += np.count_nonzero(tile == tile[1 - dr, 1 - dc]) > 1
        count_shift = max(n for n, ls in profiles['train']) <= 3 and min(
            n for n, ls in profiles['test']) >= 4
        reach_shift = all(ls <= {3, 4} for n, ls in profiles['train']) and all(
            ls <= {4, 5} and 5 in ls for n, ls in profiles['test'])
        _require(count_shift or reach_shift, "missing documented train/test shift")
        shifts.add('count' if count_shift else 'reach')
    _require(len(signatures) == 42, "seed diversity")
    _require(len(shapes) >= 20 and len(out_shapes) >= 15 and len(structures) >= 150, "structural diversity")
    _require(backgrounds == pins == carriers == set(range(10)), "palette role diversity")
    _require(counts == {2, 3, 4, 5} and lengths == {3, 4, 5}, "count/reach diversity")
    _require(len(degrees) >= 5 and cycles >= 1 and branches >= 1 and convergences >= 10,
             "graph/topology diversity")
    _require(sample_repeats >= 30, "repeated spatial samples missing")
    _require(shifts == {'count', 'reach'}, "shift diversity")
    p = generate_puzzle(314159, 2, 1)
    _require(_signature(p) == _signature(generate_puzzle(314159, 2, 1)), "determinism")
    # Repacking and complete palette permutations must leave the semantics intact.
    pair = p['train'][0]
    info = _check_pair(pair)
    parts = [{point: int(pair['input'][point]) for point in s['points']} for s in info['stations']]
    repacked = _pack(parts, info['background'], np.random.default_rng(180))
    _require(np.array_equal(solve(repacked), pair['output']), "packing leaked an address")
    permutation = np.array([7, 0, 9, 3, 1, 8, 2, 6, 4, 5], dtype=np.int8)
    _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]),
             "literal color dependence")
    copied = pair['input'].copy()
    solve(copied)
    _require(np.array_equal(copied, pair['input']), "solver mutated input")
    # Destroy all links of a two-station input without changing its sample hubs.
    linked, linked_info = _new_pair(np.random.default_rng(170), 2, (3, 4), (0, 1, 2))
    disconnected = linked['input'].copy()
    for i, j, d, length in linked_info['edges']:
        s = linked_info['stations'][i]
        dr, dc = DIRECTIONS[d]
        r, c = s['center']
        # Preserve center carrier and the inward sample, altering another glyph cell.
        y, x = next((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)
                    if (y, x) not in ((0, 0), (-dr, -dc)))
        point = r + length * dr + y, c + length * dc + x
        old = int(disconnected[point])
        disconnected[point] = next(v for v in range(10) if v not in (0, 2, old))
    try:
        solve(disconnected)
    except AssertionError:
        pass
    else:
        raise AssertionError("disconnected docking accepted")
    bad = {s: [{k: a.copy() for k, a in pair.items()} for pair in p[s]] for s in p}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt atlas accepted")
    for n in range(2, 11):
        for m in range(1, min(4, n - 1) + 1):
            validate_generated_puzzle(generate_puzzle(8000 + n * 10 + m, n, m))
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, True), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return None
