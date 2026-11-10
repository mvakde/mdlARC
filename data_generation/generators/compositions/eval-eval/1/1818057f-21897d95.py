"""Cross switches: local plus recognition controls simultaneous island exchange.

0 is empty space; 5 is erasable switch ink. All other colors are territory
colors. No generator metadata, examples, or seed is used by solve().
"""
from __future__ import annotations

import hashlib
from collections import Counter

import numpy as np


D = ((-1, 0), (0, 1), (1, 0), (0, -1))
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
AXES = ((0, 2), (1, 3))


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= n <= 30 for n in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((0 <= g) & (g <= 9))):
        raise ValueError("expected a 1..30 by 1..30 integer ARC grid")
    return g


def _add(p, d, k=1):
    return p[0] + k*d[0], p[1] + k*d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _cross(p):
    return {p, *(_add(p, d) for d in D)}


def _centers(ink):
    mask = np.zeros(ink.shape, dtype=bool)
    if min(ink.shape) >= 3:
        mask[1:-1, 1:-1] = (ink[1:-1, 1:-1] & ink[:-2, 1:-1]
                                & ink[2:, 1:-1] & ink[1:-1, :-2]
                                & ink[1:-1, 2:])
    return [tuple(map(int, p)) for p in np.argwhere(mask)]


def _territories(g):
    """Original four-connected colored islands; ink never belongs to them."""
    labels = np.full(g.shape, -1, dtype=np.int16)
    colors, cells = [], []
    for start in np.ndindex(g.shape):
        if g[start] in (0, 5) or labels[start] >= 0:
            continue
        label, color = len(colors), int(g[start])
        colors.append(color)
        points, stack = [], [start]
        labels[start] = label
        while stack:
            p = stack.pop()
            points.append(p)
            for d in D:
                q = _add(p, d)
                if (_inside(q, g.shape) and labels[q] < 0 and g[q] == color):
                    labels[q] = label
                    stack.append(q)
        cells.append(points)
    return labels, colors, cells


def _ray(labels, p, d):
    """Return islands in first-encounter order, and their first distances."""
    seen, hits = set(), []
    for k in range(1, max(labels.shape)):
        q = _add(p, d, k)
        if not _inside(q, labels.shape):
            break
        label = int(labels[q])
        if label >= 0 and label not in seen:
            seen.add(label)
            hits.append((label, k))
    return hits


def _links(labels, centers):
    links, rays = [], []
    for p in centers:
        arms = [_ray(labels, p, d) for d in D]
        rays.append(arms)
        for a, b in AXES:
            if arms[a] and arms[b]:
                links.append((arms[a][0][0], arms[b][0][0]))
    return links, rays


def _paint(g, labels, colors, centers):
    links, rays = _links(labels, centers)
    assignments = {}
    for a, b in links:
        for target, source in ((a, b), (b, a)):
            payload = colors[source]
            if target in assignments and assignments[target] != payload:
                raise ValueError("conflicting incoming colors: outside the family")
            assignments[target] = payload
    out = g.copy()
    out[g == 5] = 0
    for target, payload in assignments.items():
        out[labels == target] = payload
    return out, links, rays, assignments


def _decode(input_grid):
    g = _grid(input_grid)
    labels, colors, cells = _territories(g)
    centers = _centers(g == 5)
    out, links, rays, assignments = _paint(g, labels, colors, centers)
    return out, dict(labels=labels, colors=colors, cells=cells, centers=centers,
                     links=links, rays=rays, assignments=assignments)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read only this input; simultaneously exchange opposite first-hit islands."""
    return _decode(input_grid)[0]


def _features(g, info=None):
    if info is None:
        _, info = _decode(g)
    labels, colors, cells = info['labels'], info['colors'], info['cells']
    centers, rays = info['centers'], info['rays']
    final = [info['assignments'].get(i, c) for i, c in enumerate(colors)]
    union = set().union(*(_cross(p) for p in centers)) if centers else set()
    remaining, destructive = (g == 5).copy(), []
    for p in centers:
        if all(remaining[q] for q in _cross(p)):
            destructive.append(p)
            for q in _cross(p):
                remaining[q] = False
    sequential = _paint(g, labels, colors, destructive)[0]
    correct = _paint(g, labels, colors, centers)[0]
    near_miss, occlusion = False, False
    for p in map(tuple, np.argwhere(g == 5)):
        degree = sum(_inside(_add(p, d), g.shape) and g[_add(p, d)] == 5 for d in D)
        if degree == 3:
            arms = [_ray(labels, p, d) for d in D]
            for a, b in AXES:
                if arms[a] and arms[b]:
                    x, y = arms[a][0][0], arms[b][0][0]
                    near_miss |= colors[y] != final[x] or colors[x] != final[y]
    for arms in rays:
        for a, b in AXES:
            if not arms[a] or not arms[b]:
                continue
            for dest, source in ((a, b), (b, a)):
                first, payload = arms[dest][0][0], colors[arms[source][0][0]]
                if colors[first] == payload:
                    continue
                occlusion |= any(final[i] == colors[i] and colors[i] != payload
                                 for i, _ in arms[dest][1:])
    incidence = Counter(i for edge in info['links'] for i in set(edge))
    return dict(
        shape=tuple(g.shape), centers=len(centers), territories=len(colors),
        active_axes=len(info['links']), changed=sum(a != b for a, b in zip(colors, final)),
        near_miss=bool(near_miss), occlusion=bool(occlusion),
        locality=any(colors[a] == colors[b] and final[a] != final[b]
                     for a in range(len(colors)) for b in range(a)),
        whole_region=any(len(cells[i]) >= 3 and colors[i] != final[i] for i in range(len(colors))),
        overlap=5*len(centers) > len(union),
        overlap_effect=not np.array_equal(sequential, correct),
        extended=any(_inside(_add(p, d, 2), g.shape) and g[_add(p, d, 2)] == 5
                     for p in centers for d in D),
        both_axes=any(all(arms[d] for d in range(4)) for arms in rays),
        unpaired=any(bool(arms[a]) != bool(arms[b]) for arms in rays for a, b in AXES),
        shared=any(v >= 2 for v in incidence.values()),
        branch=any(len({b if a == i else a for a, b in info['links'] if i in (a, b)}) >= 2
                   for i in incidence),
        distances=sorted({distance for arms in rays for ray in arms for _, distance in ray[:1]}),
        areas=[len(c) for c in cells], palette=sorted(set(colors)),
        irregular=any(len(c) != (max(r for r, _ in c)-min(r for r, _ in c)+1)
                               * (max(v for _, v in c)-min(v for _, v in c)+1) for c in cells))


def _free(labels, ink):
    occupied = labels >= 0
    blocked = occupied.copy()
    blocked[1:] |= occupied[:-1]
    blocked[:-1] |= occupied[1:]
    blocked[:, 1:] |= occupied[:, :-1]
    blocked[:, :-1] |= occupied[:, 1:]
    return ~blocked & ~ink


def _island(rng, labels, ink, anchor, *, narrow=False, forbidden=None):
    allowed = _free(labels, ink)
    if forbidden is not None:
        allowed &= ~forbidden
    if not _inside(anchor, labels.shape) or not allowed[anchor]:
        return None
    count = int(rng.integers(2, 13))
    kind = int(rng.integers(5))
    points = {anchor}
    # Lines, rectangles, hooks, and organic connected growth. The placement
    # anchor need not be the center or a corner of the resulting silhouette.
    if narrow:
        allowed[:, :anchor[1]] = False
        allowed[:, anchor[1]+1:] = False
        count = int(rng.integers(2, 5))
    elif kind in (0, 1):
        axis = int(rng.integers(2))
        thickness = 1 if kind == 0 else int(rng.integers(2, 4))
        length = int(rng.integers(2, 5))
        offsets = [(a, b) for a in range(thickness) for b in range(length)]
        pivot = offsets[int(rng.integers(len(offsets)))]
        stamp = {(anchor[0]+(a-pivot[0] if axis else b-pivot[1]),
                  anchor[1]+(b-pivot[1] if axis else a-pivot[0])) for a, b in offsets}
        if all(_inside(p, labels.shape) and allowed[p] for p in stamp):
            points = stamp
            count = len(points)
    elif kind == 2:
        a, b = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        stamp = {anchor} | {_add(anchor, D[0], k) for k in range(1, a)}
        stamp |= {_add(anchor, D[1], k) for k in range(1, b)}
        if all(_inside(p, labels.shape) and allowed[p] for p in stamp):
            points = stamp
    while len(points) < count:
        frontier = sorted({_add(p, d) for p in points for d in D
                           if _inside(_add(p, d), labels.shape)
                           and allowed[_add(p, d)] and _add(p, d) not in points})
        if not frontier:
            break
        points.add(frontier[int(rng.integers(len(frontier)))])
    if len(points) < 2:
        return None
    label = int(labels.max()) + 1
    for p in points:
        labels[p] = label
    return label


def _endpoint(rng, labels, ink, p, direction, cap, narrow=False):
    if _ray(labels, p, direction):
        return
    options = [_add(p, direction, k) for k in range(3, cap+1)
               if _inside(_add(p, direction, k), labels.shape)
               and labels[_add(p, direction, k)] < 0 and not ink[_add(p, direction, k)]]
    for i in rng.permutation(len(options)):
        if _island(rng, labels, ink, options[int(i)], narrow=narrow) is not None:
            return


def _bipartite_palette(rng, labels, links):
    count = int(labels.max()) + 1
    neighbors = [set() for _ in range(count)]
    for a, b in links:
        if a != b:
            neighbors[a].add(b)
            neighbors[b].add(a)
    parity, colors = {}, [-1]*count
    for start in range(count):
        if start in parity:
            continue
        pair = list(map(int, rng.choice(PALETTE, size=2, replace=False)))
        stack, parity[start] = [start], 0
        while stack:
            a = stack.pop()
            colors[a] = pair[parity[a]]
            for b in sorted(neighbors[a]):
                if b in parity:
                    if parity[b] == parity[a]:
                        return None
                else:
                    parity[b] = 1-parity[a]
                    stack.append(b)
    return colors


def _candidate(rng, kind, target_count):
    witness = kind in ('contrast', 'overlap')
    low, high = (19, 27) if witness else ((23, 31) if kind == 'test' else (13, 29))
    h, w = map(int, rng.integers(low, high, size=2))
    if rng.random() < .15:
        w = h
    ink = np.zeros((h, w), dtype=bool)
    planned = []
    if kind == 'contrast':
        planned = [(int(rng.integers(6, h-6)), int(rng.integers(6, w-6)))]
    elif kind == 'overlap':
        r, c = int(rng.integers(5, h-5)), int(rng.integers(5, w-7))
        planned = [(r, c), (r, c+2)]
    elif kind == 'test':
        r, c = int(rng.integers(5, h-5)), int(rng.integers(4, w-11))
        planned = [(r, c), (r, c+6)]
    for _ in range(80):
        if len(planned) >= target_count:
            break
        p = int(rng.integers(2, h-2)), int(rng.integers(2, w-2))
        if all(abs(p[0]-q[0])+abs(p[1]-q[1]) >= 5 for q in planned):
            planned.append(p)
    if len(planned) < target_count:
        return None
    for p in planned:
        for q in _cross(p):
            ink[q] = True
    for p in planned:
        if rng.random() < .75 or kind == 'contrast':
            # One or several extended arms, without an isolation assumption.
            for i in rng.permutation(4)[:int(rng.integers(1, 4))]:
                for k in range(2, int(rng.integers(2, 5))+1):
                    q = _add(p, D[int(i)], k)
                    if _inside(q, ink.shape):
                        ink[q] = True
    # A disconnected T is a deliberate incomplete switch, not an arrow here.
    t_candidates = [(r, c) for r in range(3, h-3) for c in range(3, w-3)
                    if not ink[r-2:r+3, c-2:c+3].any()]
    if not t_candidates:
        return None
    t = t_candidates[int(rng.integers(len(t_candidates)))]
    missing = int(rng.integers(4))
    for q in {t, *(_add(t, d) for i, d in enumerate(D) if i != missing)}:
        ink[q] = True
    # Small additional near misses: diagonal Xs, hollow 3x3 rings, clipped edge
    # crosses. Every genuine plus is read again after these marks are drawn.
    if not witness and rng.random() < .6:
        r, c = int(rng.integers(1, h-1)), int(rng.integers(1, w-1))
        patch = ink[max(0,r-2):min(h,r+3), max(0,c-2):min(w,c+3)]
        if not patch.any():
            offsets = ([(0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1)]
                       if rng.random() < .5 else
                       [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)])
            for a, b in offsets:
                ink[r+a, c+b] = True
    if not witness and rng.random() < .35:
        c = int(rng.integers(1, w-1))
        if not ink[:3, c-1:c+2].any():
            ink[0, c-1:c+2] = True
            ink[1, c] = True
    centers = _centers(ink)
    labels = np.full((h, w), -1, dtype=np.int16)
    # A test's two aligned hubs can share a middle island. That island receives
    # consistent transfers from two different opposite endpoints.
    if kind == 'test':
        _island(rng, labels, ink, _add(planned[0], D[1], 3), narrow=True)
    cap = 13 if kind == 'test' else 8
    for index, p in enumerate(centers):
        directions = list(range(4))
        if kind == 'overlap' and index == 0:
            directions.remove(2)
        elif not witness and rng.random() < .3:
            directions.remove(int(rng.integers(4)))
        for i in rng.permutation(directions):
            _endpoint(rng, labels, ink, p, D[int(i)], cap,
                      narrow=(kind == 'overlap' and int(i) in (0, 2)))
    # Targets on the incomplete arm axis make accepting a T visibly wrong.
    for i in (missing, (missing+2) % 4):
        _endpoint(rng, labels, ink, t, D[i], 8)
    # Farther islands are geometrically present but occluded by the first hit.
    for p in centers[:2]:
        for i in rng.permutation(4):
            hits = _ray(labels, p, D[int(i)])
            if not hits:
                continue
            first_distance = hits[0][1]
            for k in range(first_distance+4, max(h, w)):
                q = _add(p, D[int(i)], k)
                if not _inside(q, labels.shape):
                    break
                if _island(rng, labels, ink, q, narrow=int(i) in (0, 2)) is not None:
                    break
    # A remote component with an already-used color witnesses component-local
    # transfer. Its entire silhouette avoids every genuine switch sightline.
    forbidden = np.zeros((h, w), dtype=bool)
    for r, c in centers:
        forbidden[r, :] = True
        forbidden[:, c] = True
    remote = None
    candidates = np.argwhere(_free(labels, ink) & ~forbidden)
    for index in rng.permutation(len(candidates))[:80]:
        p = tuple(map(int, candidates[int(index)]))
        remote = _island(rng, labels, ink, p, forbidden=forbidden)
        if remote is not None:
            break
    links, _ = _links(labels, centers)
    if not links:
        return None
    colors = _bipartite_palette(rng, labels, links)
    if colors is None:
        return None
    changed_ids = {a for a, b in links if colors[a] != colors[b]}
    changed_ids |= {b for a, b in links if colors[a] != colors[b]}
    if not changed_ids:
        return None
    if remote is not None:
        colors[remote] = colors[min(changed_ids)]
    inp = np.zeros((h, w), dtype=np.int8)
    for label, color in enumerate(colors):
        inp[labels == label] = color
    inp[ink] = 5
    # Construction labels provide the expected original component masks;
    # validation reconstructs them afresh from the returned input.
    out = inp.copy()
    out[ink] = 0
    assignments = {}
    for a, b in links:
        assignments[a], assignments[b] = colors[b], colors[a]
    for label, color in assignments.items():
        out[labels == label] = color
    turns, reflect = int(rng.integers(4)), bool(rng.integers(2))
    inp, out = np.rot90(inp, turns), np.rot90(out, turns)
    if reflect:
        inp, out = np.fliplr(inp), np.fliplr(out)
    inp, out = inp.copy(), out.copy()
    try:
        predicted, info = _decode(inp)
    except ValueError:
        return None
    if not np.array_equal(predicted, out):
        return None
    f = _features(inp, info)
    required = {'contrast': ('near_miss', 'occlusion', 'locality', 'whole_region', 'both_axes', 'extended'),
                'overlap': ('overlap_effect', 'locality', 'whole_region', 'unpaired'),
                'test': ('shared', 'whole_region'), 'extra': ('whole_region',)}[kind]
    if not all(f[name] for name in required):
        return None
    return dict(input=inp, output=out)


def _key(g):
    a = np.asarray(g, dtype=np.uint8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if (not _integer(num_train) or not _integer(num_test)
            or not 2 <= num_train <= 10 or not 1 <= num_test <= 4 or num_train <= num_test):
        raise ValueError("need 2..10 train, 1..4 test, and num_train > num_test")
    rng = np.random.default_rng(int(seed))
    puzzle, seen = dict(train=[], test=[]), set()
    requests = [('contrast', 1), ('overlap', 2)]
    requests += [('extra', int(rng.integers(1, 4))) for _ in range(num_train-2)]
    max_train_centers = 0
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            kind, target_count = (requests[index] if split == 'train'
                                  else ('test', max_train_centers+int(rng.integers(1, 3))))
            for _ in range(180):
                pair = _candidate(rng, kind, target_count)
                if pair is None or _key(pair['input']) in seen:
                    continue
                n = len(_centers(pair['input'] == 5))
                if split == 'test' and n <= max_train_centers:
                    continue
                break
            else:
                raise RuntimeError('bounded scene construction exhausted')
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
            if split == 'train':
                max_train_centers = max(max_train_centers, n)
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'illegal counts')
    seen, features = set(), dict(train=[], test=[])
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, 'dimensions must be preserved')
            _need(_key(inp) not in seen, 'duplicate inputs')
            seen.add(_key(inp))
            expected, info = _decode(inp)
            _need(np.array_equal(expected, out), 'incorrect original-component exchange')
            _need(not np.any(out == 5), 'ink survived erasure')
            _need(np.array_equal(inp == 0, (out == 0) & (inp != 5)), 'empty space changed')
            _need(np.array_equal(inp > 0, (out > 0) | (inp == 5)), 'territory support changed')
            _need(all(out[p] == 0 for p in zip(*np.nonzero(inp == 5))), 'ink is not empty in output')
            for label, points in enumerate(info['cells']):
                values = {int(out[p]) for p in points}
                _need(values == {info['assignments'].get(label, info['colors'][label])},
                      'partial territory repaint or nonlocal color substitution')
            f = _features(inp, info)
            _need(f['centers'] >= 1 and f['changed'] >= 2 and f['whole_region'], 'vacuous switch scene')
            features[split].append(f)
    for feature in ('near_miss', 'occlusion', 'locality', 'whole_region', 'both_axes',
                    'extended', 'overlap_effect', 'unpaired'):
        _need(any(f[feature] for f in features['train']), 'missing training contrast: '+feature)
    _need(min(f['centers'] for f in features['test']) > max(f['centers'] for f in features['train']),
          'missing router-count train/test shift')


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                shape, data = _key(pair[role])
                digest.update(bytes(shape))
                digest.update(data)
    return digest.hexdigest()


def self_test():
    """Bounded, no file writes: 42 puzzles, local motif and semantic regressions."""
    fingerprints, shapes, structures, palettes, counts, features = set(), set(), set(), set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        if seed in (0, 17):
            _need(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), 'nondeterministic seed')
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp = pair['input']
                saved = inp.copy()
                _need(np.array_equal(solve(inp), pair['output']), 'input-only solve failed')
                _need(np.array_equal(inp, saved), 'solve mutated its input')
                _, info = _decode(inp)
                f = _features(inp, info)
                features.append(f)
                shapes.add(inp.shape)
                structures.add((inp.shape, info['labels'].tobytes(), (inp == 5).tobytes()))
                palettes.add(tuple(f['palette']))
                counts.add((f['centers'], f['territories'], f['active_axes']))
    _need(len(fingerprints) == 42 and len(shapes) >= 50, 'collapsed puzzle or shape variation')
    _need(len(structures) >= 180 and len(counts) >= 50 and len(palettes) >= 20, 'collapsed structural variation')
    _need(any(30 in s for s in shapes), 'missing large dimensions')
    for feature in ('near_miss', 'occlusion', 'locality', 'overlap_effect', 'unpaired',
                    'extended', 'both_axes', 'irregular', 'shared', 'branch'):
        _need(sum(bool(f[feature]) for f in features) >= 3, 'unobserved variation: '+feature)
    _need(set.union(*(set(f['palette']) for f in features)) == set(PALETTE), 'collapsed palette')
    # All binary 3x3 neighborhoods: no diagonal or extension condition is added.
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], dtype=bool).reshape(3, 3)
        expected = bool(mask[1, 1] and mask[0, 1] and mask[2, 1] and mask[1, 0] and mask[1, 2])
        _need(bool(_centers(mask)) == expected, 'wrong five-cell motif')
    # A hand-constructed exchange also fixes boundary, occlusion, annotation
    # erasure, simultaneous swapping, and same-color disconnected locality.
    g = np.zeros((11, 15), dtype=np.int8)
    for p in _cross((5, 6)):
        g[p] = 5
    g[5, 1:3], g[5, 10:12], g[5, 14] = 2, 7, 3
    g[1, 1:3] = 2
    expected = g.copy()
    expected[expected == 5] = 0
    expected[5, 1:3], expected[5, 10:12] = 7, 2
    _need(np.array_equal(solve(g), expected), 'swap/occlusion/locality regression')
    missing = g.copy()
    missing[4, 6] = 0
    cleaned = missing.copy()
    cleaned[cleaned == 5] = 0
    _need(np.array_equal(solve(missing), cleaned), 'T incorrectly fired')
    # Reflecting/rotating a scene must transform its answer in the same way.
    for k in range(4):
        _need(np.array_equal(solve(np.rot90(g, k)), np.rot90(expected, k)), 'rotation equivariance')
        _need(np.array_equal(solve(np.fliplr(np.rot90(g, k))), np.fliplr(np.rot90(expected, k))), 'reflection equivariance')
    clipped = np.zeros((5, 9), dtype=np.int8)
    clipped[0, 3:6], clipped[1, 4] = 5, 5
    clipped[0, 0:2], clipped[0, 7:9] = 2, 7
    cleaned = clipped.copy()
    cleaned[cleaned == 5] = 0
    _need(np.array_equal(solve(clipped), cleaned), 'clipped edge cross incorrectly fired')
    # Two hubs address a shared middle island with incompatible original fills.
    conflict = np.zeros((11, 25), dtype=np.int8)
    for center in ((5, 5), (5, 15)):
        for q in _cross(center):
            conflict[q] = 5
    conflict[5, 1:3], conflict[5, 10:12], conflict[5, 20:22] = 2, 7, 3
    try:
        solve(conflict)
    except ValueError:
        pass
    else:
        raise AssertionError('conflicting incoming payloads were accepted')
    # With matching outside fills the same geometry is a valid shared target.
    conflict[5, 20:22] = 2
    expected_shared = conflict.copy()
    expected_shared[expected_shared == 5] = 0
    expected_shared[5, 1:3], expected_shared[5, 10:12], expected_shared[5, 20:22] = 7, 2, 7
    _need(np.array_equal(solve(conflict), expected_shared), 'consistent shared target rejected')
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            _need(len(p['train']) == a and len(p['test']) == b, 'explicit counts ignored')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for seed in (-1, True, 3.5):
        try:
            generate_puzzle(seed)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal seed accepted')
    # Labels and examples are validated, not silently regenerated.
    p = generate_puzzle(42, 2, 1)
    p['test'][0]['output'][0, 0] = (int(p['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(p)
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError('validator accepted corrupt output')
