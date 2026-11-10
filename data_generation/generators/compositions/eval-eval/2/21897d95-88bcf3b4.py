"""Perimeter-addressed paint tethers. Standard library + NumPy only.

0 = exterior, 8 = T-dock ink, 9 = flexible strand, 1..7 = fills/payloads.
Every result is decoded from input geometry; no generation state is consulted.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np

D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
HOST_SEEDS = (0, 21300155, 143753639, 957202031, 48320994, 678279200,
              326511574, 836891280, 905138008, 215296069, 58277330,
              927642187, 403263111, 403126880, 417450846, 747425589,
              421557997, 339356607, 372553374, 508441220, 550588377,
              917298042, 508070375, 112194628, 222588321, 294898393,
              269579667, 983957387, 178411629, 554607331, 277797960,
              619358157, 26983465, 729085436, 424967835, 383790649,
              64600535, 567975614, 190330695, 724120461, 22814199, 35668491)


def _need(ok, message):
    if not ok:
        raise AssertionError(message)


def _int(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _add(p, d, k=1):
    return p[0] + k * d[0], p[1] + k * d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(grid, diagonal=False, include=None):
    """Return monochrome components, optionally restricted to selected cells."""
    g = np.asarray(grid)
    ids = np.full(g.shape, -1, dtype=np.int16)
    groups, colors = [], []
    directions = D8 if diagonal else D4
    for p in np.ndindex(g.shape):
        if ids[p] >= 0 or (include is not None and not include[p]):
            continue
        color = int(g[p])
        label = len(groups)
        ids[p] = label
        todo, cells = [p], set()
        while todo:
            q = todo.pop()
            cells.add(q)
            for d in directions:
                z = _add(q, d)
                if (_inside(z, g.shape) and ids[z] < 0 and g[z] == color
                        and (include is None or include[z])):
                    ids[z] = label
                    todo.append(z)
        groups.append(cells)
        colors.append(color)
    return ids, colors, groups


def _connected(cells, diagonal=False):
    if not cells:
        return False
    reached = {min(cells)}
    todo = list(reached)
    while todo:
        p = todo.pop()
        for d in D8 if diagonal else D4:
            q = _add(p, d)
            if q in cells and q not in reached:
                reached.add(q)
                todo.append(q)
    return reached == cells


def _perimeter(mask):
    """Clockwise boundary edges, mapped to exterior cells with runs collapsed.

    Edge orientation keeps the island on the right in row-down coordinates.
    Restrict the family to a single simple contour and a nonrepeating cell ring.
    """
    edges = {}
    for r, c in map(tuple, np.argwhere(mask)):
        options = (((r - 1, c), (r, c), (r, c + 1)),
                   ((r, c + 1), (r, c + 1), (r + 1, c + 1)),
                   ((r + 1, c), (r + 1, c + 1), (r + 1, c)),
                   ((r, c - 1), (r + 1, c), (r, c)))
        for outside, start, end in options:
            if not _inside(outside, mask.shape) or not mask[outside]:
                _need(_inside(outside, mask.shape), "island needs an exterior margin")
                _need(start not in edges, "non-simple boundary vertex")
                edges[start] = (end, outside)
    _need(bool(edges), "missing island contour")
    start = min(edges)
    vertex, ring, visited = start, [], set()
    while vertex not in visited:
        _need(vertex in edges, "broken contour")
        visited.add(vertex)
        vertex, outside = edges[vertex]
        if not ring or outside != ring[-1]:
            ring.append(outside)
    _need(vertex == start and len(visited) == len(edges), "island has holes or separate pieces")
    if ring[0] == ring[-1]:
        ring.pop()
    _need(len(set(ring)) == len(ring), "exterior contour revisits a cell")
    _need(all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1
              for a, b in zip(ring, ring[1:] + ring[:1])), "disconnected exterior ring")
    return ring


def _dock_geometry(center, direction):
    side = (-direction[1], direction[0])
    tip = _add(center, direction)
    return dict(center=center, direction=direction, tip=tip,
                arms=(_add(center, side), _add(center, side, -1)),
                flanks=(_add(tip, side), _add(tip, side, -1)),
                anchor=_add(center, direction, 2),
                write={center, tip, _add(center, side), _add(center, side, -1)})


def _recipient(point, ids):
    neighbors = {int(ids[q]) for d in D4 if _inside(q := _add(point, d), ids.shape)
                 and ids[q] >= 0}
    return next(iter(neighbors)) if len(neighbors) == 1 else None


def _decode(input_grid):
    g = np.asarray(input_grid)
    _need(g.ndim == 2 and np.issubdtype(g.dtype, np.integer), "expected integer grid")
    _need(all(1 <= n <= 30 for n in g.shape) and np.all((0 <= g) & (g <= 9)), "grid range")
    _need(np.all(g[0] == 0) and np.all(g[-1] == 0)
          and np.all(g[:, 0] == 0) and np.all(g[:, -1] == 0), "blank outside border required")
    docks = []
    for center in np.ndindex(g.shape):
        if not 1 <= g[center] <= 8:
            continue
        for d in D4:
            a = _dock_geometry(center, d)
            if not all(_inside(p, g.shape) for p in a['write'] | set(a['flanks']) | {a['anchor']}):
                continue
            if not all(g[p] == 8 for p in (*a['arms'], a['tip'])) or g[a['anchor']] != 9:
                continue
            host = int(g[a['flanks'][0]])
            if not 1 <= host <= 7 or g[a['flanks'][1]] != host:
                continue
            a.update(host=host, payload=host if g[center] == 8 else int(g[center]),
                     explicit=bool(g[center] != 8))
            docks.append(a)
    _need(1 <= len(docks) <= 3, "expected one to three unambiguous docks")
    used = set()
    for a in docks:
        _need(not used & a['write'], "overlapping docks")
        used.update(a['write'])
    _need({p for p in used if g[p] == 8} == set(map(tuple, np.argwhere(g == 8))), "unexplained dock ink")
    clean = g.copy()
    clean[clean == 9] = 0
    for a in docks:
        for p in a['write']:
            clean[p] = a['host']
    island = clean != 0
    ring = _perimeter(island)
    index = {p: i for i, p in enumerate(ring)}
    ids, colors, regions = _components(clean, include=island)
    strand_ids, _, strands = _components(g, diagonal=True, include=g == 9)
    _need(len(strands) == len(docks), "each dock must have its own slack component")
    assigned_strands, assignments = set(), {}
    for a in docks:
        source = int(ids[a['center']])
        _need(source >= 0 and all(ids[p] == source for p in a['write'] | set(a['flanks'])), "dock crosses regions")
        sid = int(strand_ids[a['anchor']])
        _need(sid >= 0 and sid not in assigned_strands, "ambiguous strand attachment")
        assigned_strands.add(sid)
        slack = strands[sid]
        _need(a['anchor'] in index and 2 <= len(slack) < len(ring), "invalid tether length or anchor")
        # A slack cloud lies in the outward half-plane of its dock's tip.
        _need(all((p[0] - a['tip'][0]) * a['direction'][0]
                  + (p[1] - a['tip'][1]) * a['direction'][1] >= 1 for p in slack), "slack crosses dock plane")
        start = index[a['anchor']]
        path = [ring[(start + k) % len(ring)] for k in range(len(slack))]
        target = _recipient(path[-1], ids)
        _need(target is not None and target != source, "endpoint needs one distinct recipient")
        _need(a['payload'] != colors[target], "delivery must change the recipient")
        _need(target not in assignments or assignments[target] == a['payload'], "conflicting deliveries")
        assignments[target] = a['payload']
        a.update(source=source, target=target, slack=slack, path=path, length=len(slack), start=start)
    out = clean.copy()
    for target, payload in assignments.items():
        out[ids == target] = payload
    for a in docks:
        for p in a['path']:
            _need(not island[p], "tether enters island")
            out[p] = 9
    return out.astype(np.int8), dict(clean=clean, island=island, ring=ring, ids=ids,
                                   colors=colors, regions=regions, docks=docks, assignments=assignments)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input alone; no seed, puzzle, or stored labels."""
    return _decode(input_grid)[0]


def _island(rng, large, style):
    axis = int(rng.integers(2))
    major = int(rng.integers(16, 21) if large else rng.integers(13, 18))
    minor = int(rng.integers(11, 18) if large else rng.integers(9, 16))
    h, w = (major, minor) if axis == 0 else (minor, major)
    margins = []
    for n in (h, w):
        budget = min(30 - n, int(rng.integers(9, 14)))
        before = int(rng.integers(4, budget - 3))
        margins.extend((before, budget - before))
    top, bottom, left, right = margins
    shape = h + top + bottom, w + left + right
    mask = np.zeros(shape, dtype=bool)
    cuts = [0, 0, 0, 0] if style == 0 else list(map(int, rng.integers(0, min(4, w // 3) + 1, size=4)))
    for r in range(h):
        lo = max(0, cuts[0] - r, cuts[2] - (h - 1 - r))
        hi = w - max(0, cuts[1] - r, cuts[3] - (h - 1 - r))
        mask[top + r, left + lo:left + hi] = True
    # Unequal stepped strips, with occasional additional subdivision.
    count = int(rng.integers(4, min(6, major // 3) + 1))
    widths = np.full(count, 3)
    for _ in range(major - 3 * count):
        widths[int(rng.integers(count))] += 1
    boundaries = np.cumsum(widths)[:-1]
    rr, cc = np.indices(shape)
    along = rr - top if axis == 0 else cc - left
    across = cc - left if axis == 0 else rr - top
    labels = np.zeros(shape, dtype=np.int16)
    for boundary in boundaries:
        pivot = int(rng.integers(2, minor - 1))
        delta = int(rng.choice((-1, 0, 0, 1))) if style else 0
        labels += along >= boundary + delta * (across >= pivot)
    labels[~mask] = -1
    if rng.random() < .45:
        rid = int(rng.integers(count))
        cut = minor // 2 + int(rng.integers(-1, 2))
        part = (labels == rid) & (across >= cut)
        if part.sum() >= 8 and ((labels == rid) & ~part).sum() >= 8:
            labels[part] = count
            count += 1
    for rid in range(count):
        cells = set(map(tuple, np.argwhere(labels == rid)))
        if not _connected(cells):
            return None
    try:
        ring = _perimeter(mask)
    except AssertionError:
        return None
    return labels, ring


def _palette(rng, labels, repeat):
    n = int(labels.max()) + 1
    colors = list(map(int, rng.choice(np.arange(1, 8), n, replace=False)))
    if repeat:
        adjacent = set()
        for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:])):
            adjacent.update(tuple(sorted((int(x), int(y)))) for x, y in zip(a.ravel(), b.ravel()) if x >= 0 and y >= 0 and x != y)
        choices = [(a, b) for a in range(n) for b in range(a) if (b, a) not in adjacent]
        if not choices:
            return None
        a, b = choices[int(rng.integers(len(choices)))]
        colors[a] = colors[b]
    return colors


def _candidates(labels, ring):
    index = {p: i for i, p in enumerate(ring)}
    geometries, options = [], []
    for center in map(tuple, np.argwhere(labels >= 0)):
        source = int(labels[center])
        for direction in D4:
            a = _dock_geometry(center, direction)
            if a['anchor'] not in index or not all(_inside(p, labels.shape) for p in a['write'] | set(a['flanks'])):
                continue
            if not all(labels[p] == source for p in a['write'] | set(a['flanks'])):
                continue
            a.update(source=source, start=index[a['anchor']])
            gi = len(geometries)
            geometries.append(a)
            for length in range(3, min(49, len(ring))):
                target = _recipient(ring[(a['start'] + length - 1) % len(ring)], labels)
                if target is not None and target != source:
                    options.append((gi, length, source, target))
    return geometries, options


def _grow_slack(rng, geometry, length, mask, blocked, diagonal):
    anchor, tip, d = geometry['anchor'], geometry['tip'], geometry['direction']
    h, w = mask.shape
    allowed = set()
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            p = r, c
            normal = (r - tip[0]) * d[0] + (c - tip[1]) * d[1]
            side = abs((r - tip[0]) * d[1] - (c - tip[1]) * d[0])
            if normal >= 1 and side <= 7 and not mask[p] and p not in blocked:
                allowed.add(p)
    if anchor not in allowed:
        return None
    cells = {anchor}
    directions = D8 if diagonal else D4
    # A diagonal-only first link gives an explicit eight-connectivity lesson.
    if diagonal:
        side = (-d[1], d[0])
        choices = [_add(_add(anchor, d), side, sign) for sign in (-1, 1)]
        choices = [p for p in choices if p in allowed]
        if not choices:
            return None
        q = choices[int(rng.integers(len(choices)))]
        cells.add(q)
        allowed -= {_add(anchor, z) for z in D4}
        # Prevent later growth from restoring a four-connected bridge.
    for _ in range(length - len(cells)):
        frontier = sorted({q for p in cells for z in directions if (q := _add(p, z)) in allowed} - cells)
        if not frontier:
            return None
        weights = np.array([1 / (1 + abs(p[0] - anchor[0]) + abs(p[1] - anchor[1])) for p in frontier])
        cells.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
    return cells


def _locality(colors, assignments):
    return any(colors[a] == colors[b] and assignments.get(a, colors[a]) != assignments.get(b, colors[b])
               for a in range(len(colors)) for b in range(a))


def _plan(rng, labels, ring, colors, count, lesson, length_limit, minimum_length=3):
    geometries, all_options = _candidates(labels, ring)
    options = [o for o in all_options if minimum_length <= o[1] <= length_limit and colors[o[2]] != colors[o[3]]]
    if not options:
        return None
    for _ in range(100):
        first = options[int(rng.integers(len(options)))]
        chosen = [first]
        assignments = {first[3]: colors[first[2]]}
        if lesson == 'address':
            a = geometries[first[0]]
            reverse_target = _recipient(ring[(a['start'] - first[1] + 1) % len(ring)], labels)
            if reverse_target in (None, first[3]):
                continue
            visited = {_recipient(ring[(a['start'] + k) % len(ring)], labels) for k in range(first[1] - 1)}
            if not visited - {None, first[2], first[3]}:
                continue
            path = [ring[(a['start'] + k) % len(ring)] for k in range(first[1])]
            steps = {(q[0] - p[0], q[1] - p[1]) for p, q in zip(path, path[1:])}
            if len(steps) < 2:
                continue
        if lesson == 'simultaneous':
            pool = [o for o in options if o[2] == first[3] and o[3] != first[3]
                    and _locality(colors, {first[3]: colors[first[2]], o[3]: colors[o[2]]})]
            if not pool:
                continue
            chosen.append(pool[int(rng.integers(len(pool)))])
            assignments[chosen[-1][3]] = colors[chosen[-1][2]]
        while len(chosen) < count:
            pool = [o for o in options if o[0] not in {x[0] for x in chosen}
                    and (o[3] not in assignments or assignments[o[3]] == colors[o[2]])]
            if not pool:
                break
            nxt = pool[int(rng.integers(len(pool)))]
            chosen.append(nxt)
            assignments[nxt[3]] = colors[nxt[2]]
        if len(chosen) != count:
            continue
        docks, blocked, writes = [], set(), set()
        valid = True
        for j, (gi, length, source, target) in enumerate(chosen):
            a = dict(geometries[gi])
            context = a['write'] | set(a['flanks'])
            if writes & context or any(max(abs(a['anchor'][0] - b['anchor'][0]), abs(a['anchor'][1] - b['anchor'][1])) < 5 for b in docks):
                valid = False
                break
            diagonal = lesson == 'address' or bool(rng.random() < .5)
            slack = _grow_slack(rng, a, length, labels >= 0, blocked, diagonal)
            if slack is None:
                valid = False
                break
            for p in slack:
                blocked.update(_add(p, d) for d in D8)
                blocked.add(p)
            writes.update(context)
            payload, explicit = colors[source], False
            if lesson != 'simultaneous' and (lesson == 'address' or rng.random() < .45):
                pool = [c for c in range(1, 8) if c not in (colors[source], colors[target])]
                if lesson == 'address':
                    pool = [c for c in pool if c not in colors]
                payload, explicit = int(rng.choice(pool)), True
            # Later explicit payloads must agree with any already selected target.
            existing = [b['payload'] for b in docks if b['target'] == target]
            if existing:
                payload = existing[0]
                explicit = payload != colors[source]
            path = [ring[(a['start'] + k) % len(ring)] for k in range(length)]
            a.update(target=target, length=length, slack=slack, payload=payload,
                     explicit=explicit, path=path)
            docks.append(a)
        if valid:
            final = {a['target']: a['payload'] for a in docks}
            if lesson == 'simultaneous' and not _locality(colors, final):
                continue
            return docks
    return None


def _make_pair(rng, count, lesson, large, style, length_limit, minimum_length=3, minimum_area=0):
    for _ in range(100):
        built = _island(rng, large, style)
        if built is None:
            continue
        labels, ring = built
        if int((labels >= 0).sum()) <= minimum_area:
            continue
        colors = _palette(rng, labels, repeat=lesson == 'simultaneous' or rng.random() < .6)
        if colors is None:
            continue
        docks = _plan(rng, labels, ring, colors, count, lesson, length_limit, minimum_length)
        if docks is None:
            continue
        clean = np.zeros(labels.shape, dtype=np.int8)
        for rid, color in enumerate(colors):
            clean[labels == rid] = color
        inp, out = clean.copy(), clean.copy()
        assignments = {a['target']: a['payload'] for a in docks}
        for target, payload in assignments.items():
            out[labels == target] = payload
        for a in docks:
            for p in a['write']:
                inp[p] = 8
            if a['explicit']:
                inp[a['center']] = a['payload']
            for p in a['slack']:
                inp[p] = 9
            for p in a['path']:
                out[p] = 9
        try:
            predicted, info = _decode(inp)
            if not np.array_equal(predicted, out) or not np.array_equal(info['clean'], clean):
                continue
            return dict(input=inp, output=out)
        except AssertionError:
            continue
    raise AssertionError('bounded construction exhausted')


def _key(g):
    a = np.asarray(g, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _int(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if not _int(num_train) or not 2 <= num_train <= 10:
        raise ValueError('num_train must be an integer in 2..10')
    if not _int(num_test) or not 1 <= num_test <= 4 or num_train <= num_test:
        raise ValueError('num_test must be in 1..4 and smaller than num_train')
    rng = np.random.default_rng(int(seed))
    regime = int(rng.integers(3))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, number in (('train', int(num_train)), ('test', int(num_test))):
        for i in range(number):
            lesson = ('address' if i == 0 else 'simultaneous' if i == 1 else None) if split == 'train' else None
            count = 1 if lesson == 'address' else 2 if lesson else int(rng.choice((1, 2)))
            if split == 'test' and regime == 1:
                count = 3
            large = split == 'test' and regime == 0
            style = int(rng.choice((0, 1, 1, 1)))
            # The scale regime is geometric; the count regime recombines three docks.
            # The remaining regime tests contour lengths beyond demonstration range.
            limit = 34 if split == 'train' else 47 if regime == 2 else 38
            minimum_length = 35 if split == 'test' and regime == 2 else 3
            minimum_area = (max(int(_decode(p['input'])[1]['island'].sum()) for p in puzzle['train'])
                            if split == 'test' and regime == 0 else 0)
            for _ in range(8):
                pair = _make_pair(rng, count, lesson, large, style, limit, minimum_length, minimum_area)
                key = _key(pair['input'])
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise AssertionError('distinct input construction exhausted')
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    ds, colors, assignments = info['docks'], info['colors'], info['assignments']
    chain = any(a is not b and not a['explicit'] and not b['explicit']
                and a['target'] == b['source'] and a['payload'] != b['payload'] for a in ds for b in ds)
    address = False
    for a in ds:
        reverse = _recipient(info['ring'][(a['start'] - a['length'] + 1) % len(info['ring'])], info['ids'])
        passed = {_recipient(p, info['ids']) for p in a['path'][:-1]}
        steps = {(q[0] - p[0], q[1] - p[1]) for p, q in zip(a['path'], a['path'][1:])}
        if len(ds) == 1 and len(steps) > 1 and reverse is not None and reverse != a['target'] and passed - {None, a['source'], a['target']}:
            address = True
    return dict(count=len(ds), lengths=[a['length'] for a in ds],
                explicit=sum(a['explicit'] for a in ds), chain=chain,
                singleton=any(a['explicit'] and a['payload'] not in colors for a in ds),
                locality=_locality(colors, assignments), address=address,
                diagonal=any(not _connected(a['slack']) for a in ds),
                regions=len(colors), perimeter=len(info['ring']),
                nonrectangular=int(info['island'].sum()) != int(np.prod(np.ptp(np.argwhere(info['island']), axis=0) + 1)))


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'illegal pair counts')
    seen, training, details = set(), [], {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair schema')
            for g in pair.values():
                g = np.asarray(g)
                _need(g.ndim == 2 and np.issubdtype(g.dtype, np.integer), 'integer 2D grids required')
                _need(all(1 <= n <= 30 for n in g.shape) and np.all((0 <= g) & (g <= 9)), 'grid range')
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            _need(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            predicted, info = _decode(inp)
            details[split].append(info)
            _need(np.array_equal(predicted, out), 'wrong delivery or tightened strand')
            _need(inp.shape == out.shape and not np.array_equal(inp, out), 'shape/identity')
            _need(not np.any(out == 8), 'dock ink survived')
            path_union = {p for d in info['docks'] for p in d['path']}
            _need(path_union == set(map(tuple, np.argwhere(out == 9))), 'strand union or length wrong')
            for rid, cells in enumerate(info['regions']):
                expected = info['assignments'].get(rid, info['colors'][rid])
                _need(all(out[p] == expected for p in cells), 'component-local painting failed')
            if split == 'train':
                training.append(_features(info))
    _need(any(f['address'] and f['singleton'] and f['diagonal'] for f in training), 'missing explicit length/direction lesson')
    _need(any(f['chain'] and f['locality'] for f in training), 'missing simultaneous/locality lesson')
    train, test = details['train'], details['test']
    scale_shift = all(int(i['island'].sum()) > max(int(t['island'].sum()) for t in train) for i in test)
    count_shift = all(len(i['docks']) > max(len(t['docks']) for t in train) for i in test)
    length_shift = all(min(a['length'] for a in i['docks']) > max(a['length'] for t in train for a in t['docks']) for i in test)
    _need(scale_shift or count_shift or length_shift, 'missing measured train/test shift')
    return None


def self_test():
    """42 seeds; retain only aggregate features/hashes, no execution traces."""
    counts, shapes, silhouettes, structures, palettes, lengths = set(), set(), set(), set(), set(), set()
    frequencies = Counter()
    directions = {'train': set(), 'test': set()}
    for seed in HOST_SEEDS:
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                predicted, info = _decode(inp.copy())
                _need(np.array_equal(predicted, out), 'input-only prediction mismatch')
                f = _features(info)
                directions[split].update(a['direction'] for a in info['docks'])
                counts.add(f['count'])
                shapes.add(inp.shape)
                palettes.add(tuple(info['colors']))
                lengths.update(f['lengths'])
                coordinates = np.argwhere(info['island'])
                origin = coordinates.min(axis=0)
                silhouettes.add(tuple(map(tuple, coordinates - origin)))
                # Region IDs are already canonical in row-major traversal.
                structures.add(hashlib.sha256(info['ids'].tobytes()).hexdigest())
                for feature in ('explicit', 'chain', 'locality', 'address', 'diagonal', 'nonrectangular'):
                    frequencies[feature] += bool(f[feature])
                frequencies[split + '_pairs'] += 1
                # Omitting A leaves every region unchanged; generated pairs refute it.
                no_paint = info['clean'].copy()
                for dock in info['docks']:
                    for p in dock['path']:
                        no_paint[p] = 9
                _need(not np.array_equal(no_paint, out), 'A contributes no change')
                # A bounding-box contour is a distinguishable B ablation on bevels.
                if f['nonrectangular']:
                    box = np.zeros(inp.shape, dtype=bool)
                    lo, hi = coordinates.min(axis=0), coordinates.max(axis=0)
                    box[lo[0]:hi[0]+1, lo[1]:hi[1]+1] = True
                    box_ring = _perimeter(box)
                    box_index = {p: i for i, p in enumerate(box_ring)}
                    different = False
                    for a in info['docks']:
                        if a['anchor'] not in box_index:
                            different = True
                        else:
                            path = [box_ring[(box_index[a['anchor']] + k) % len(box_ring)] for k in range(a['length'])]
                            different |= path != a['path']
                    frequencies['contour_vs_box'] += different
                if f['address'] and f['singleton'] and f['diagonal']:
                    a = info['docks'][0]
                    # Concrete wrong-rule predictions on the unoverlapped lesson.
                    alternatives = {}
                    reverse_path = [info['ring'][(a['start'] - k) % len(info['ring'])] for k in range(a['length'])]
                    reverse_target = _recipient(reverse_path[-1], info['ids'])
                    reverse = info['clean'].copy()
                    reverse[info['ids'] == reverse_target] = a['payload']
                    for p in reverse_path:
                        reverse[p] = 9
                    alternatives['counterclockwise'] = reverse
                    four_part = {a['anchor']}
                    todo = list(four_part)
                    while todo:
                        p = todo.pop()
                        for d in D4:
                            q = _add(p, d)
                            if q in a['slack'] and q not in four_part:
                                four_part.add(q)
                                todo.append(q)
                    short_path = a['path'][:len(four_part)]
                    short = info['clean'].copy()
                    short_target = _recipient(short_path[-1], info['ids'])
                    if short_target is not None:
                        short[info['ids'] == short_target] = a['payload']
                    for p in short_path:
                        short[p] = 9
                    alternatives['four_connected_count'] = short
                    touched = out.copy()
                    for p in a['path']:
                        rid = _recipient(p, info['ids'])
                        if rid is not None:
                            touched[info['ids'] == rid] = a['payload']
                    alternatives['all_contacts'] = touched
                    host = out.copy()
                    host[info['ids'] == a['target']] = a['host']
                    alternatives['ignore_payload'] = host
                    for name, alternative in alternatives.items():
                        _need(not np.array_equal(alternative, out), 'shortcut survived: ' + name)
                        frequencies['refutes_' + name] += 1
                if seed in (0, 21300155):
                    _need(np.array_equal(solve(np.rot90(inp)), np.rot90(out)), 'rotation covariance failed')
                    palette_map = np.array([0, 7, 6, 5, 4, 3, 2, 1, 8, 9], dtype=np.int8)
                    _need(np.array_equal(solve(palette_map[inp]), palette_map[out]), 'fill permutation failed')
        if seed == 0:
            again = generate_puzzle(seed)
            _need(all(_key(p[k]) == _key(q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s]) for k in p), 'nondeterministic')
            bad = {s: [{k: g.copy() for k, g in p.items()} for p in puzzle[s]] for s in puzzle}
            _, detail = _decode(bad['test'][0]['input'])
            p = next(iter(detail['regions'][detail['docks'][0]['target']]))
            bad['test'][0]['output'][p] = 8
            try:
                validate_generated_puzzle(bad)
            except AssertionError:
                pass
            else:
                raise AssertionError('validator accepted corrupted target')
    _need(counts == {1, 2, 3}, 'missing tether counts')
    _need(all(ds == set(D4) for ds in directions.values()), 'missing dock sides')
    _need(len(shapes) > 30 and len(silhouettes) > 50 and len(structures) > 100, 'collapsed geometry')
    _need(len(palettes) > 80 and len(lengths) > 15, 'collapsed palette/lengths')
    _need(all(frequencies[f] >= 20 for f in ('explicit', 'chain', 'locality', 'address', 'diagonal', 'nonrectangular', 'contour_vs_box')), 'missing mechanism evidence')
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(42, a, b)
            _need(len(p['train']) == a and len(p['test']) == b, 'counts ignored')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    return dict(seeds=42, counts=sorted(counts), grid_shapes=len(shapes), silhouettes=len(silhouettes),
                partitions=len(structures), palettes=len(palettes), lengths=sorted(lengths), features=dict(frequencies))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
