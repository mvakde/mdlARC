"""Registered probes select original connected territories for simultaneous fills.

Only NumPy and the standard library are used. Generation retains component IDs;
solve reconstructs annotations and components entirely from its argument.
No files, parent modules, random state, or generated examples are read by solve.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
NEAR = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1))
FILLS = (2, 3, 4, 5, 6, 7, 9)
ANNOTATIONS = (0, 1, 8)


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _add(p, d, k=1):
    return p[0] + k * d[0], p[1] + k * d[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _halo(cells):
    return {_add(p, d) for p in cells for d in NEAR}


def _components(grid):
    ids = np.full(grid.shape, -1, dtype=np.int16)
    colors, groups = [], []
    for p in np.ndindex(grid.shape):
        if ids[p] >= 0:
            continue
        label, color = len(groups), int(grid[p])
        stack, cells = [p], set()
        ids[p] = label
        while stack:
            q = stack.pop()
            cells.add(q)
            for d in DIRS:
                v = _add(q, d)
                if _inside(v, grid.shape) and ids[v] < 0 and grid[v] == color:
                    ids[v] = label
                    stack.append(v)
        colors.append(color)
        groups.append(cells)
    return ids, colors, groups


def _groups8(points):
    remaining, groups = set(points), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, group = [start], {start}
        while stack:
            p = stack.pop()
            for d in NEAR:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    group.add(q)
                    stack.append(q)
        groups.append(group)
    return groups


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), 'grid dimensions')
    _need(np.issubdtype(g.dtype, np.integer), 'integer colors required')
    _need(bool(np.all((g >= 0) & (g <= 9))), 'color range')
    return g.astype(np.int8, copy=True)


def _geometry(center, d, shape):
    side = (-d[1], d[0])
    arms = (_add(center, side), _add(center, side, -1))
    tip, landing = _add(center, d), _add(center, d, 2)
    flanks = (_add(tip, side), _add(tip, side, -1))
    write = {center, tip, *arms}
    if not all(_inside(p, shape) for p in write | {landing, *flanks}):
        return None
    return dict(center=center, direction=d, arms=arms, tip=tip,
                landing=landing, flanks=flanks, write=write)


def _project(ids, landing, offsets):
    probes = {_add(landing, d) for d in offsets}
    inside = {p for p in probes if _inside(p, ids.shape)}
    return {int(ids[p]) for p in inside}, inside, len(probes - inside)


def _paint(clean, ids, arrows, mode='normal'):
    """Apply a parsed plan; alternate modes are diagnostic counter-rules."""
    assignments = {}
    for a in arrows:
        if mode == 'direct':
            targets = {int(ids[a['landing']])}
        elif mode in ('box', 'center'):
            offsets = a['offsets']
            r0, r1 = min(r for r, c in offsets), max(r for r, c in offsets)
            c0, c1 = min(c for r, c in offsets), max(c for r, c in offsets)
            if mode == 'box':
                offsets = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
            else:
                # The tempting but incorrect bounding-box-center registration.
                offsets = {(r - (r0 + r1) // 2, c - (c0 + c1) // 2) for r, c in offsets}
            targets, _, _ = _project(ids, a['landing'], offsets)
        else:
            targets = a['targets']
        for label in targets:
            if label in assignments and assignments[label] != a['payload']:
                return None
            assignments[label] = a['payload']
    out = clean.copy()
    if mode == 'pixels':
        for a in arrows:
            for p in a['probes']:
                out[p] = a['payload']
    else:
        for label, color in assignments.items():
            out[ids == label] = color
    return out


def _decode(input_grid):
    raw = _grid(input_grid)
    stencil_points = {tuple(map(int, p)) for p in np.argwhere((raw == 0) | (raw == 8))}
    stencils = []
    for cells in _groups8(stencil_points):
        registrations = [p for p in cells if raw[p] == 0]
        mask = {p for p in cells if raw[p] == 8}
        _need(len(registrations) == 1 and bool(mask), 'one registration per nonempty stencil')
        _need(len(_groups8(mask)) == 1, 'gray stencil must be eight-connected')
        q = registrations[0]
        _need(any(max(abs(r - q[0]), abs(c - q[1])) <= 1 for r, c in mask),
              'registration must touch its stencil')
        border = _halo(cells) - cells
        _need(all(_inside(p, raw.shape) for p in border), 'stencil needs a visible surrounding fill')
        values = {int(raw[p]) for p in border}
        _need(len(values) == 1 and next(iter(values)) in FILLS, 'stencil host is not unambiguous')
        stencils.append(dict(cells=cells, mask=mask, registration=q, host=values.pop(),
                             offsets=tuple(sorted((r-q[0], c-q[1]) for r, c in mask))))
    _need(bool(stencils), 'no probe stencil')

    arrows = []
    ink = {tuple(map(int, p)) for p in np.argwhere(raw == 1)}
    possible_centers = sorted({_add(p, d) for p in ink for d in DIRS
                               if _inside(_add(p, d), raw.shape)})
    for center in possible_centers:
        for d in DIRS:
            a = _geometry(center, d, raw.shape)
            if a is None or not all(raw[p] == 1 for p in (*a['arms'], a['tip'])):
                continue
            if raw[center] not in (1, *FILLS):
                continue
            x, y = a['flanks']
            if raw[x] != raw[y] or raw[x] not in FILLS or raw[a['landing']] not in FILLS:
                continue
            host = int(raw[x])
            if raw[a['landing']] == host:
                continue
            a.update(host=host, explicit=bool(raw[center] != 1),
                     payload=int(raw[center]) if raw[center] != 1 else host)
            arrows.append(a)
    _need(bool(arrows), 'no transfer arrow')
    used = set()
    for a in arrows:
        _need(not used & a['write'], 'overlapping arrow readings')
        used |= a['write']
    _need(used & ink == ink, 'unexplained blue ink')
    _need(not used & stencil_points, 'overlapping annotations')
    clean = raw.copy()
    for rec in stencils + arrows:
        for p in rec.get('cells', rec.get('write')):
            clean[p] = rec['host']
    _need(set(map(int, np.unique(clean))) <= set(FILLS), 'annotation survived restoration')
    ids, colors, components = _components(clean)
    by_source = {}
    for s in stencils:
        source = int(ids[s['registration']])
        _need(all(ids[p] == source for p in s['cells']), 'stencil crosses a territory')
        _need(source not in by_source, 'multiple stencils in one territory')
        s['source'] = source
        by_source[source] = s
    assignments = {}
    for a in arrows:
        source = int(ids[a['center']])
        _need(all(ids[p] == source for p in a['write']), 'arrow crosses its host boundary')
        _need(source in by_source, 'arrow host has no stencil')
        _need(ids[a['landing']] != source, 'arrow does not cross a boundary')
        offsets = by_source[source]['offsets']
        targets, probes, clipped = _project(ids, a['landing'], offsets)
        _need(bool(targets), 'all probes lie outside the canvas')
        a.update(source=source, offsets=offsets, targets=targets, probes=probes, clipped=clipped)
        for target in targets:
            _need(target not in assignments or assignments[target] == a['payload'],
                  'conflicting simultaneous transfers')
            assignments[target] = a['payload']
    _need({a['source'] for a in arrows} == set(by_source), 'unused stencil')
    out = _paint(clean, ids, arrows)
    _need(out is not None, 'conflicting paint')
    return out, dict(clean=clean, ids=ids, colors=colors, components=components,
                     stencils=stencils, arrows=arrows, assignments=assignments)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Read fixed annotation conventions and original components from this input."""
    return _decode(input_grid)[0]


def _choose(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _partition(rng, shape, count):
    """Unequal discrete Voronoi regions, sometimes with an enclosed island."""
    h, w = shape
    seeds = []
    for _ in range(150):
        p = int(rng.integers(h)), int(rng.integers(w))
        if all(abs(p[0]-q[0]) + abs(p[1]-q[1]) >= 5 for q in seeds):
            seeds.append(p)
            if len(seeds) == count:
                break
    if len(seeds) != count:
        return None
    rr, cc = np.indices(shape)
    power = int(_choose(rng, (1, 2, 2)))
    ratio = float(rng.uniform(.65, 1.55))
    distances = np.array([ratio * abs(rr-r)**power + abs(cc-c)**power for r, c in seeds])
    labels = distances.argmin(axis=0).astype(np.int16)
    ids, _, groups = _components(labels)
    if len(groups) != count or min(map(len, groups)) < 9:
        return None
    labels = ids
    if rng.random() < .4:
        rh, cw = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        possible = []
        for r in range(1, h-rh):
            for c in range(1, w-cw):
                patch = labels[r-1:r+rh+1, c-1:c+cw+1]
                if np.all(patch == patch[0, 0]):
                    possible.append((r, c))
        if possible:
            r, c = _choose(rng, possible)
            labels[r:r+rh, c:c+cw] = count
    return _components(labels)[0]


def _candidates(labels):
    # Test the small support stencil in array slices rather than rebuilding
    # geometry at every pixel. Keep the original row/column/direction order,
    # so the subsequent shuffle consumes the same RNG stream.
    h, w = labels.shape
    valid = np.zeros((h, w, len(DIRS)), dtype=bool)
    for k, d in enumerate(DIRS):
        dr, dc = d
        pr, pc = -dc, dr
        support = ((0, 0), (dr, dc), (pr, pc), (-pr, -pc),
                   (dr+pr, dc+pc), (dr-pr, dc-pc))
        bounds = support + ((2*dr, 2*dc),)
        r0, r1 = max(0, -min(r for r, c in bounds)), h-max(r for r, c in bounds)
        c0, c1 = max(0, -min(c for r, c in bounds)), w-max(c for r, c in bounds)
        if r0 >= r1 or c0 >= c1:
            continue
        center = labels[r0:r1, c0:c1]
        matches = labels[r0+2*dr:r1+2*dr, c0+2*dc:c1+2*dc] != center
        for r, c in support:
            matches &= labels[r0+r:r1+r, c0+c:c1+c] == center
        valid[r0:r1, c0:c1, k] = matches
    result = []
    for r, c, k in np.argwhere(valid):
        center = int(r), int(c)
        a = _geometry(center, DIRS[int(k)], labels.shape)
        a.update(source=int(labels[center]), neighbor=int(labels[a['landing']]))
        result.append(a)
    return result


def _rotate_offsets(offsets, d):
    # Canonical mask points right. Rotation acts on the stored geometry too.
    dr, dc = d
    return tuple(sorted((r*dc + c*dr, -r*dr + c*dc) for r, c in offsets))


def _mask(rng, d, richer=False):
    above, below = int(rng.integers(0, 3)), int(rng.integers(0, 3))
    width = int(rng.integers(2, 6 if richer else 5))
    mode = _choose(rng, ('grow', 'grow', 'branch', 'line', 'outline'))
    if mode == 'line':
        cells = {(r, 1) for r in range(-above, below+1)}
        if len(cells) == 1:
            cells.add((0, 2))
    elif mode == 'branch':
        cells = {(r, 1) for r in range(-above, below+1)}
        cells |= {(int(_choose(rng, (-above, below))), c) for c in range(1, width)}
    elif mode == 'outline':
        cells = {(r, c) for r in range(-above, below+1) for c in range(width)
                 if r in (-above, below) or c in (0, width-1)} - {(0, 0)}
        # A degenerate outline may split at the registration; retain one mask.
        if len(_groups8(cells)) != 1:
            cells = {(r, 1) for r in range(-above, below+1)} | {(0, 2)}
    else:
        cells = {(0, 1)}
        desired = int(rng.integers(2, max(3, (above+below+1)*width)))
        for _ in range(80):
            if len(cells) >= desired:
                break
            p = _choose(rng, sorted(cells))
            q = _add(p, _choose(rng, NEAR[0:4] + NEAR[5:]))
            if -above <= q[0] <= below and 0 <= q[1] < width and q != (0, 0):
                cells.add(q)
    return _rotate_offsets(cells, d)


def _stencil_location(rng, labels, source, offsets, forbidden):
    relative = set(offsets) | {(0, 0)}
    support = _halo(relative)
    h, w = labels.shape
    r0, r1 = max(0, -min(r for r, c in support)), h-max(r for r, c in support)
    c0, c1 = max(0, -min(c for r, c in support)), w-max(c for r, c in support)
    if r0 >= r1 or c0 >= c1:
        return None
    allowed = np.ones((r1-r0, c1-c0), dtype=bool)
    for r, c in support:
        allowed &= labels[r0+r:r1+r, c0+c:c1+c] == source
    blocked = np.zeros(labels.shape, dtype=bool)
    for p in forbidden:
        if _inside(p, labels.shape):
            blocked[p] = True
    for r, c in relative:
        allowed &= ~blocked[r0+r:r1+r, c0+c:c1+c]
    possibilities = np.argwhere(allowed)
    if not len(possibilities):
        return None
    # argwhere is row-major, matching the former scalar scan exactly.
    row, col = _choose(rng, possibilities)
    q = r0+int(row), c0+int(col)
    return q, {_add(q, d) for d in relative}


def _adjacent(labels):
    edges = set()
    for a, b in ((labels[:-1], labels[1:]), (labels[:, :-1], labels[:, 1:])):
        different = a != b
        edges |= {tuple(sorted((int(x), int(y)))) for x, y in zip(a[different], b[different])}
    return edges


def _palette(rng, labels, selected, donor_ids, locality):
    count = int(labels.max()) + 1
    edges = _adjacent(labels)
    repeat = None
    choices = [(a, b) for a in sorted(selected) for b in range(count)
               if b not in selected and a not in donor_ids and b not in donor_ids
               and tuple(sorted((a, b))) not in edges]
    if locality and not choices:
        return None
    if choices and (locality or rng.random() < .65):
        repeat = _choose(rng, choices)
    pool = [int(x) for x in rng.permutation(FILLS)]
    colors = [-1] * count
    if repeat:
        colors[repeat[0]] = colors[repeat[1]] = pool[0]
    for label in list(donor_ids) + list(map(int, rng.permutation(count))):
        if colors[label] >= 0:
            continue
        forbidden = {colors[j] for j in range(count) if tuple(sorted((label, j))) in edges}
        if label in donor_ids:
            forbidden |= {colors[j] for j in donor_ids if colors[j] >= 0}
        options = [c for c in pool if c not in forbidden]
        unused = [c for c in options if c not in colors]
        if not options:
            return None
        colors[label] = _choose(rng, unused or options)
    return np.array(colors, dtype=np.int8)


def _signature(offsets):
    # Keep registration fixed at the origin while factoring out a global
    # rotation/reflection. Test novelty cannot be satisfied by a mere turn.
    variants = []
    for sign in (-1, 1):
        reflected = [(sign*r, c) for r, c in offsets]
        variants.extend(_rotate_offsets(reflected, d) for d in DIRS)
    return min(variants)


def _features(info, out):
    clean, ids, colors = info['clean'], info['ids'], info['colors']
    final = [info['assignments'].get(i, c) for i, c in enumerate(colors)]
    def differs(mode):
        alt = _paint(clean, ids, info['arrows'], mode)
        return alt is None or not np.array_equal(alt, out)
    # Color-wide recoloring cannot represent two equal original fills that
    # end with different colors, regardless of which replacement it chooses.
    locality = any(colors[i] == colors[j] and final[i] != final[j]
                   for i in range(len(colors)) for j in range(i))
    donors = {a['source'] for a in info['arrows']}
    dependency = any(a['source'] in info['assignments'] and
                     info['assignments'][a['source']] != a['host'] and not a['explicit']
                     and any(final[t] == a['host'] and colors[t] != a['host'] for t in a['targets'])
                     for a in info['arrows'])
    return dict(shape=clean.shape, territories=len(colors), donors=len(donors),
                arrows=len(info['arrows']), stencil_sizes=tuple(len(s['offsets']) for s in info['stencils']),
                signatures=tuple(_signature(s['offsets']) for s in info['stencils']),
                directions=tuple(sorted({a['direction'] for a in info['arrows']})),
                explicit=any(a['explicit'] for a in info['arrows']),
                implicit=any(not a['explicit'] for a in info['arrows']),
                clipped=any(a['clipped'] for a in info['arrows']), locality=locality,
                mixed=any(len({a['direction'] for a in info['arrows'] if a['source'] == source}) > 1
                          for source in donors),
                dependency=dependency, multi=any(len(a['targets'] - {a['source']}) >= 2 for a in info['arrows']),
                skip=any(int(ids[a['landing']]) not in a['targets'] for a in info['arrows']),
                no_a=differs('pixels'), no_b=differs('direct'), box=differs('box'), center=differs('center'),
                nonrectangular=sum(len(cells) != (max(r for r,c in cells)-min(r for r,c in cells)+1)*
                                     (max(c for r,c in cells)-min(c for r,c in cells)+1)
                                     for cells in info['components']))


def _attempt_pair(rng, teaching, testing, familiar):
    swap = teaching == 1
    two_donors = swap or (teaching is None and rng.random() < .4)
    shape = (int(rng.integers(23 if two_donors else 18, 31)),
             int(rng.integers(23 if two_donors else 18, 31)))
    count = int(rng.integers(4, 7) if two_donors else rng.integers(5 if testing else 4, 9))
    labels = _partition(rng, shape, count)
    if labels is None:
        return None
    candidates = _candidates(labels)
    rng.shuffle(candidates)
    if not candidates:
        return None
    # Try several visible transfer graphs on one partition, without saving
    # unsuccessful candidates or any execution trace.
    for first in candidates[:28]:
        offsets = _mask(rng, first['direction'], testing)
        if testing and _signature(offsets) in familiar:
            continue
        targets, probes, clipped = _project(labels, first['landing'], offsets)
        if first['neighbor'] not in targets or first['source'] in targets:
            continue
        first = dict(first, offsets=offsets, targets=targets, probes=probes, clipped=clipped)
        arrows, masks = [first], {first['source']: offsets}
        if two_donors:
            first_halo = _halo(first['write'] | set(first['flanks']) | {first['landing']})
            reverse = [a for a in candidates if a['source'] != first['source'] and
                       (not swap or (a['source'] == first['neighbor'] and a['neighbor'] == first['source'])) and
                       not a['write'] & first_halo]
            found = False
            for second0 in reverse[:14]:
                offsets2 = _mask(rng, second0['direction'])
                ts, ps, cl = _project(labels, second0['landing'], offsets2)
                if (not ts - {second0['source']} or ts & targets or second0['source'] in ts
                        or (swap and first['source'] not in ts)):
                    continue
                arrows.append(dict(second0, offsets=offsets2, targets=ts, probes=ps, clipped=cl))
                masks[second0['source']] = offsets2
                found = True
                break
            if not found:
                continue
        elif teaching != 0 and rng.random() < .4:
            # Extra arrows reuse a donor's one stencil; they are not new
            # instructions associated merely by matching colors.
            desired = int(rng.integers(2, 5))
            for a in candidates:
                if len(arrows) >= desired:
                    break
                if a['source'] != first['source']:
                    continue
                occupied = set().union(*(b['write'] | set(b['flanks']) | {b['landing']} for b in arrows))
                occupied_halo = _halo(occupied)
                if a['write'] & occupied_halo or (set(a['flanks']) | {a['landing']}) & occupied_halo:
                    continue
                ts, ps, cl = _project(labels, a['landing'], offsets)
                if ts - {first['source']}:
                    arrows.append(dict(a, offsets=offsets, targets=ts, probes=ps, clipped=cl))
        forbidden = _halo(set().union(*(a['write'] | set(a['flanks']) | {a['landing']} for a in arrows)))
        stencils = []
        for source, off in masks.items():
            loc = _stencil_location(rng, labels, source, off, forbidden)
            if loc is None:
                break
            q, cells = loc
            stencils.append(dict(source=source, registration=q, cells=cells,
                                 mask={_add(q, d) for d in off}, offsets=off))
            forbidden |= _halo(cells)
        if len(stencils) != len(masks):
            continue
        selected = set().union(*(a['targets'] for a in arrows))
        colors = _palette(rng, labels, selected, sorted(masks), locality=teaching == 0)
        if colors is None:
            continue
        clean = colors[labels]
        inp = clean.copy()
        payloads = {}
        for source in masks:
            explicit = teaching == 0 or (not swap and rng.random() < .4)
            choices = [c for c in FILLS if c != colors[source]]
            # Singletons and colors already present in territories both fit.
            unused = [c for c in choices if c not in colors]
            payloads[source] = (int(_choose(rng, unused or choices)) if explicit else int(colors[source]), explicit)
        assignments = {}
        valid = True
        for a in arrows:
            payload, explicit = payloads[a['source']]
            a.update(host=int(colors[a['source']]), payload=payload, explicit=explicit)
            for target in a['targets']:
                if target in assignments and assignments[target] != payload:
                    valid = False
                assignments[target] = payload
            for p in a['write']:
                inp[p] = 1
            if explicit:
                inp[a['center']] = payload
        if not valid:
            continue
        for s in stencils:
            for p in s['mask']:
                inp[p] = 8
            inp[s['registration']] = 0
        out = clean.copy()
        for target, payload in assignments.items():
            out[labels == target] = payload
        # Reject insufficient teaching contrasts before the more expensive
        # input reconstruction. This uses the construction plan only here.
        direct = _paint(clean, labels, arrows, 'direct')
        if direct is not None and np.array_equal(direct, out):
            continue
        if teaching == 0:
            final = [assignments.get(j, int(c)) for j, c in enumerate(colors)]
            if not any(colors[j] == colors[k] and final[j] != final[k]
                       for j in range(len(colors)) for k in range(j)):
                continue
            if any((alt := _paint(clean, labels, arrows, mode)) is not None and np.array_equal(alt, out)
                   for mode in ('box', 'center')):
                continue
        try:
            decoded, info = _decode(inp)
        except AssertionError:
            continue
        if not np.array_equal(decoded, out) or not np.array_equal(info['clean'], clean):
            continue
        feature = _features(info, out)
        if not feature['no_a'] or not feature['no_b']:
            continue
        if teaching == 0 and not (feature['locality'] and feature['box'] and feature['center']):
            continue
        if swap and not feature['dependency']:
            continue
        return dict(input=inp, output=out), feature
    return None


def _int(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if value < low or (high is not None and value > high):
        raise ValueError(name + ' out of range')
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _int(seed, 'seed', 0)
    num_train = _int(num_train, 'num_train', 2, 10)
    num_test = _int(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    puzzle, seen, familiar = {'train': [], 'test': []}, set(), set()
    training_features = []
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            teaching = i if split == 'train' and i < 2 else None
            for _ in range(180):
                result = _attempt_pair(rng, teaching, split == 'test', familiar)
                if result is None:
                    continue
                pair, feature = result
                if _key(pair['input']) in seen:
                    continue
                # These spatial edge cases need an observed demonstration,
                # especially when only two training examples were requested.
                if split == 'test' and any(feature[k] and not taught[k] for k in ('clipped', 'skip', 'mixed')):
                    continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                if split == 'train':
                    familiar.update(feature['signatures'])
                    training_features.append(feature)
                break
            else:
                raise RuntimeError('bounded construction exhausted')
        if split == 'train':
            taught = {k: any(f[k] for f in training_features) for k in ('clipped', 'skip', 'mixed')}
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'split types')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'counts')
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape, 'shape changed')
            _need(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            predicted, info = _decode(inp)
            _need(np.array_equal(predicted, out), 'incorrect probe-and-flood output')
            _need(not np.array_equal(inp, out), 'unchanged example')
            f = _features(info, out)
            _need(f['no_a'] and f['no_b'], 'one parent has no visible effect')
            # Every original component remains monochromatic, including its
            # restored stencil and arrow locations, even when outputs merge.
            for label, color in enumerate(info['colors']):
                expected = info['assignments'].get(label, color)
                _need(bool(np.all(out[info['ids'] == label] == expected)), 'component closure or preservation')
            features[split].append(f)
    train = features['train']
    for name in ('locality', 'box', 'center', 'dependency', 'explicit', 'implicit'):
        _need(any(f[name] for f in train), 'missing teaching contrast: ' + name)
    familiar = {s for f in train for s in f['signatures']}
    for f in features['test']:
        _need(any(s not in familiar for s in f['signatures']), 'test has no new registered mask')
        for k in ('clipped', 'skip', 'mixed'):
            _need(not f[k] or any(t[k] for t in train), 'spatial case not taught: ' + k)


def _manual_regressions():
    """Small specified arrays with expected answers written without a plan."""
    g = np.full((19, 21), 3, dtype=np.int8)
    g[:, 10] = 2
    g[:, 11:15] = 4
    g[:, 15:] = 6
    g[3, 10:] = 5
    expected = g.copy()
    expected[4:, 11:15] = 3
    # A sparse diagonal stencil registered at an empty point, not its center.
    # The landing (7,10) is in the thin color-2 neighbor. All three probes
    # instead touch the lower color-4 territory, sparing the upper color 4.
    g[7, 3] = 0
    g[6, 4] = g[7, 5] = g[8, 6] = 8
    g[6, 8] = g[7, 8] = g[8, 8] = g[7, 9] = 1
    _need(np.array_equal(solve(g), expected), 'manual registration/locality/skip regression')
    literal = g.copy()
    literal[7, 8] = 9
    literal_expected = expected.copy()
    literal_expected[4:, 11:15] = 9
    _need(np.array_equal(solve(literal), literal_expected), 'manual singleton payload regression')

    g = np.full((19, 21), 3, dtype=np.int8)
    g[:, 10:] = 4
    expected = np.full_like(g, 4)
    expected[:, 10:] = 3
    g[12, 3] = 0
    g[12, 4] = g[13, 5] = 8
    g[5, 15] = 0
    g[5, 14] = g[6, 13] = 8
    for p in ((3, 8), (4, 8), (5, 8), (4, 9), (12, 11), (13, 11), (14, 11), (13, 10)):
        g[p] = 1
    _need(np.array_equal(solve(g), expected), 'manual simultaneous swap regression')
    bad = g.copy()
    for p in ((16, 8), (17, 8), (18, 8), (17, 9)):
        bad[p] = 1
    bad[17, 8] = 9
    try:
        solve(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('contradictory transfers accepted')

    g = np.full((19, 21), 3, dtype=np.int8)
    g[:, 10:] = 4
    expected = np.full_like(g, 3)
    g[8, 3] = 0
    g[6:9, 4] = 8
    for p in ((0, 8), (1, 8), (2, 8), (1, 9)):
        g[p] = 1
    result, info = _decode(g)
    _need(info['arrows'][0]['clipped'] == 1, 'manual clip count')
    _need(np.array_equal(result, expected), 'manual clipping regression')


def self_test():
    """Bounded streamed checks; keep hashes/sets rather than stored puzzles."""
    _manual_regressions()
    shapes, signatures, structures, directions, fingerprints = set(), set(), set(), set(), set()
    counts, donors, sizes = set(), set(), set()
    observed = {k: 0 for k in ('locality', 'dependency', 'box', 'center', 'clipped', 'multi', 'skip')}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                predicted, info = _decode(pair['input'])
                assert np.array_equal(predicted, pair['output'])
                feature = _features(info, predicted)
                digest.update(repr(_key(pair['input'])).encode())
                digest.update(predicted.tobytes())
                shapes.add(pair['input'].shape)
                signatures.update(feature['signatures'])
                directions.update(feature['directions'])
                counts.add(feature['arrows'])
                donors.add(feature['donors'])
                sizes.update(feature['stencil_sizes'])
                structures.add(hashlib.sha256(info['ids'].tobytes()).hexdigest())
                for name in observed:
                    observed[name] += bool(feature[name])
        fingerprints.add(digest.hexdigest())
    _need(len(fingerprints) == 42, 'seed diversity')
    _need(len(shapes) >= 40 and len(structures) >= 150 and len(signatures) >= 60, 'structural diversity')
    _need(directions == set(DIRS) and donors == {1, 2}, 'direction or donor diversity')
    _need(len(counts) >= 3 and len(sizes) >= 6, 'count or stencil-size diversity')
    _need(all(observed[k] >= 10 for k in ('locality', 'dependency', 'box', 'center', 'multi')), 'missing contrast')
    # Determinism, explicit count boundaries, rejection of corrupted answers.
    x, y = generate_puzzle(91, 2, 1), generate_puzzle(91, 2, 1)
    _need(all(np.array_equal(p[k], q[k]) for s in x for p, q in zip(x[s], y[s]) for k in p), 'determinism')
    for a, b in ((10, 4), (5, 4), (3, 2)):
        p = generate_puzzle(300+a, a, b)
        _need(len(p['train']) == a and len(p['test']) == b, 'explicit counts')
    for a, b in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (4.0, 1), (True, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    bad = generate_puzzle(92, 2, 1)
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('incorrect output accepted')
    return dict(seeds=42, distinct_puzzles=len(fingerprints), shapes=len(shapes),
                partitions=len(structures), registered_masks=len(signatures), arrow_counts=sorted(counts),
                donor_counts=sorted(donors), stencil_sizes=sorted(sizes), witnesses=observed)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
