"""ARC family: arrow impacts transported through exact rotational shape matches.

Only NumPy and the Python standard library are used. The solver reads pixels;
generation plans are never passed to it. Color 9 means >=2 distinct arrows.
"""
from collections import Counter, defaultdict

import numpy as np


DISTANCE = 5
COLLISION = 9
DIRS = ((-1, 0), (-1, 1), (0, 1), (1, 1),
        (1, 0), (1, -1), (0, -1), (-1, -1))
NEIGHBORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                  if (r, c) != (0, 0))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "Invalid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Cells must be integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "Invalid color")
    return a.astype(np.int8, copy=False)


def _arrow_cells(tip, direction):
    r, c = tip
    dr, dc = direction
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return {(r + y, c + x) for y, x in offsets}


def _halo(cells):
    return {(r + y, c + x) for r, c in cells
            for y in (-1, 0, 1) for x in (-1, 0, 1)}


def _mask_key(mask):
    return mask.shape, mask.tobytes()


def _canonical(mask):
    return min(_mask_key(np.rot90(mask, k).copy()) for k in range(4))


def _rot_point(point, shape, k):
    r, c = point
    h, w = shape
    for _ in range(k % 4):
        r, c, h, w = w - 1 - c, r, w, h
    return int(r), int(c)


def _parse(grid):
    values, counts = np.unique(grid, return_counts=True)
    bg = int(values[int(counts.argmax())])
    _require(int(counts.max()) * 2 > grid.size and bg != COLLISION,
             "Background must occupy a strict majority and differ from 9")
    _require(not np.any(grid == COLLISION), "9 is reserved for output collisions")
    remaining = set(map(tuple, np.argwhere(grid != bg).tolist()))
    arrows, bodies, occupied = [], [], set()
    while remaining:
        start = min(remaining)
        component, stack = {start}, [start]
        remaining.remove(start)
        while stack:
            r, c = stack.pop()
            fresh = {(r + y, c + x) for y, x in NEIGHBORS} & remaining
            remaining -= fresh
            component |= fresh
            stack.extend(sorted(fresh))
        colors = {int(grid[p]) for p in component}
        _require(len(colors) == 1, "Every separated component must be monochrome")
        color = next(iter(colors))
        occupied |= component
        if len(component) == 3:
            matches = [(tip, d) for tip in sorted(component) for d in DIRS
                       if _arrow_cells(tip, d) == component]
            _require(len(matches) == 1, "Ambiguous or invalid arrowhead")
            tip, direction = matches[0]
            endpoint = tuple(tip[a] + DISTANCE * direction[a] for a in (0, 1))
            arrows.append(dict(tip=tip, direction=direction, cells=component,
                               color=color, endpoint=endpoint))
        else:
            _require(len(component) >= 4, "Bodies need at least four cells")
            top, left = (min(p[a] for p in component) for a in (0, 1))
            h = max(r for r, c in component) - top + 1
            w = max(c for r, c in component) - left + 1
            mask = np.zeros((h, w), dtype=bool)
            for r, c in component:
                mask[r - top, c - left] = True
            bodies.append(dict(top=top, left=left, cells=component, mask=mask,
                               color=color, canonical=_canonical(mask)))
    _require(bool(arrows and bodies), "Need at least one arrow and one body")
    arrow_cells = set().union(*(a['cells'] for a in arrows))
    for arrow in arrows:
        r, c = arrow['endpoint']
        _require(0 <= r < grid.shape[0] and 0 <= c < grid.shape[1], "Shot leaves canvas")
        _require((r, c) not in arrow_cells, "Endpoints cannot hit arrows")
    return bg, arrows, bodies


def _events(arrows, bodies, *, transfer=True, reflections=False,
            identity=False, single_rotation=False):
    """Each arrival stores the identity of its original shooting arrow."""
    owner = {cell: i for i, body in enumerate(bodies) for cell in body['cells']}
    arrivals = defaultdict(set)
    deliveries = []
    for i, arrow in enumerate(arrows):
        endpoint = arrow['endpoint']
        if endpoint not in owner or not transfer:
            cells = {endpoint}
        else:
            source = bodies[owner[endpoint]]
            local = endpoint[0] - source['top'], endpoint[1] - source['left']
            cells = set()
            variants = [(source['mask'], local)]
            if reflections:
                variants.append((np.fliplr(source['mask']),
                                 (local[0], source['mask'].shape[1] - 1 - local[1])))
            for target in bodies:
                for mask, p in variants:
                    for k in (range(1) if identity else range(4)):
                        rotated = np.rot90(mask, k)
                        if rotated.shape == target['mask'].shape and np.array_equal(rotated, target['mask']):
                            r, c = _rot_point(p, mask.shape, k)
                            cells.add((target['top'] + r, target['left'] + c))
                            if single_rotation:
                                break
        for p in cells:
            arrivals[p].add(i)
        deliveries.append(cells)
    return dict(arrivals), deliveries


def _render(grid, arrows, arrivals):
    out = grid.copy()
    for cell, ids in arrivals.items():
        out[cell] = COLLISION if len(ids) > 1 else arrows[next(iter(ids))]['color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    grid = _grid(input_grid)
    _, arrows, bodies = _parse(grid)
    arrivals, _ = _events(arrows, bodies)
    return _render(grid, arrows, arrivals)


def _alternative(grid, kind):
    """Finite shortcut probes, supporting evidence rather than an independent solver."""
    _, arrows, bodies = _parse(grid)
    kwargs = {}
    if kind == 'direct':
        kwargs['transfer'] = False
    elif kind == 'identity':
        kwargs['identity'] = True
    elif kind == 'reflection':
        kwargs['reflections'] = True
    elif kind == 'one_rotation':
        kwargs['single_rotation'] = True
    elif kind == 'first_obstacle':
        owner = set().union(*(b['cells'] for b in bodies))
        changed = []
        for a in arrows:
            a = dict(a)
            for step in range(1, DISTANCE + 1):
                p = tuple(a['tip'][j] + step * a['direction'][j] for j in (0, 1))
                if p in owner:
                    a['endpoint'] = p
                    break
            changed.append(a)
        arrows = changed
    elif kind in ('distance4', 'distance6'):
        d = int(kind[-1])
        changed = []
        for a in arrows:
            p = tuple(a['tip'][j] + d * a['direction'][j] for j in (0, 1))
            if not (0 <= p[0] < grid.shape[0] and 0 <= p[1] < grid.shape[1]):
                return None
            changed.append(dict(a, endpoint=p))
        arrows = changed
    arrivals, deliveries = _events(arrows, bodies, **kwargs)
    if kind == 'raw_collision':
        # Copy each raw endpoint's paint without recounting transported hits.
        counts = Counter(a['endpoint'] for a in arrows)
        out = grid.copy()
        for i, cells in enumerate(deliveries):
            for p in cells:
                out[p] = COLLISION if counts[arrows[i]['endpoint']] > 1 else arrows[i]['color']
        return out
    if kind == 'whole_class':
        expanded = defaultdict(set)
        owner = {p: b for b in bodies for p in b['cells']}
        for i, cells in enumerate(deliveries):
            for p in cells:
                for q in owner[p]['cells'] if p in owner else (p,):
                    expanded[q].add(i)
        arrivals = expanded
    return _render(grid, arrows, arrivals)


def _connected(mask, neighbors=NEIGHBORS):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    start = min(cells)
    seen, stack = {start}, [start]
    while stack:
        r, c = stack.pop()
        fresh = {(r + y, c + x) for y, x in neighbors} & cells - seen
        seen |= fresh
        stack.extend(sorted(fresh))
    return seen == cells


def _sample_mask(rng, *, chiral=False, diagonal=False, limit=5, forbidden=()):
    for _ in range(500):
        cells = {(0, 0)}
        target = int(rng.integers(6, min(17, limit * limit) + 1))
        neighbors = NEIGHBORS if diagonal or rng.random() < .5 else DIRS[::2]
        while len(cells) < target:
            choices = set()
            for r, c in sorted(cells):
                for y, x in neighbors:
                    p = r + y, c + x
                    trial = cells | {p}
                    if p not in cells and all(max(q[a] for q in trial) - min(q[a] for q in trial) < limit
                                              for a in (0, 1)):
                        choices.add(p)
            if not choices:
                break
            choices = sorted(choices)
            cells.add(choices[int(rng.integers(len(choices)))])
        r0, c0 = (min(p[a] for p in cells) for a in (0, 1))
        mask = np.zeros((max(r for r, c in cells) - r0 + 1,
                         max(c for r, c in cells) - c0 + 1), dtype=bool)
        for r, c in cells:
            mask[r - r0, c - c0] = True
        if len({_mask_key(np.rot90(mask, k).copy()) for k in range(4)}) != 4:
            continue
        if _canonical(mask) in forbidden:
            continue
        if chiral and _canonical(mask) == _canonical(np.fliplr(mask)):
            continue
        if diagonal and _connected(mask, DIRS[::2]):
            continue
        return mask
    raise RuntimeError('Asymmetric mask sampling exhausted')


def _symmetric_mask(rng):
    kind = int(rng.integers(5))
    n = int(rng.integers(3, 6))
    if kind == 0:
        return np.ones((n, n), dtype=bool)
    if kind == 1:
        mask = np.ones((n, n), dtype=bool)
        mask[1:-1, 1:-1] = False
        return mask
    if kind == 2:
        return np.ones((2, n), dtype=bool)
    if kind == 3:
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, :] = mask[:, 1] = True
        return mask
    return np.ones((1, int(rng.integers(4, 7))), dtype=bool)


def _choose(rng, seq):
    return seq[int(rng.integers(len(seq)))]


def _make_plan(rng, mode, palette):
    """Masks and local addresses, before any canvas positions are chosen."""
    bodies, shots = [], []
    used = set()

    def add_class(count, *, symmetric=False, chiral=False, diagonal=False, turns=None):
        base = (_symmetric_mask(rng) if symmetric else
                _sample_mask(rng, chiral=chiral, diagonal=diagonal,
                             limit=int(rng.integers(3, 7)), forbidden=used))
        if _canonical(base) in used:
            return add_class(count, symmetric=False, chiral=chiral, diagonal=diagonal, turns=turns)
        used.add(_canonical(base))
        ids = []
        turns = list(map(int, rng.integers(4, size=count))) if turns is None else turns
        for k in turns:
            ids.append(len(bodies))
            bodies.append(dict(mask=np.rot90(base, k).copy(), color=int(_choose(rng, palette)),
                               turn=k, base=base))
        return base, ids

    def shoot(index, point, ink, through=False):
        b = bodies[index]
        shots.append(dict(body=index, point=_rot_point(point, b['base'].shape, b['turn']),
                          color=int(ink), through=through))

    if mode == 'teach_address':
        base, ids = add_class(2, chiral=True, diagonal=True, turns=[0, int(_choose(rng, (1, 3)))])
        # Same-area mirror and reused body color refute reflection/area/color matching.
        mirror = np.rot90(np.fliplr(base), int(rng.integers(4))).copy()
        bodies.append(dict(mask=mirror, color=palette[0], turn=0, base=mirror))
        bodies[0]['color'], bodies[1]['color'] = palette[0], palette[1]
        p = tuple(map(int, _choose(rng, np.argwhere(base).tolist())))
        shoot(ids[0], p, palette[2], through=True)
        return bodies, shots, 1
    if mode == 'teach_merge':
        base, ids = add_class(3, turns=[0, 1, int(rng.integers(4))])
        points = [tuple(map(int, p)) for p in np.argwhere(base)]
        p, q = [points[int(i)] for i in rng.choice(len(points), size=2, replace=False)]
        for i in ids:
            bodies[i]['color'] = palette[0]
        shoot(ids[0], p, palette[1])
        shoot(ids[1], p, palette[2])
        shoot(ids[0], q, palette[3])
        sym, sym_ids = add_class(1, symmetric=True)
        # Pick a noncentral orbit, so one arrow visibly marks multiple cells.
        candidates = [tuple(map(int, p)) for p in np.argwhere(sym)
                      if len({_rot_point(tuple(p), sym.shape, k) for k in range(4)
                              if np.array_equal(np.rot90(sym, k), sym)}) > 1]
        shoot(sym_ids[0], _choose(rng, candidates), palette[4])
        bodies[sym_ids[0]]['color'] = palette[0]
        return bodies, shots, 0

    count = int(rng.integers(2, 5))
    base, ids = add_class(count, symmetric=mode == 'orbit')
    points = [tuple(map(int, p)) for p in np.argwhere(base)]
    p = _choose(rng, points)
    shoot(ids[0], p, palette[0])
    if mode in ('collision', 'mixed', 'direct_collision'):
        shoot(ids[0] if mode == 'direct_collision' else ids[1], p, palette[1])
    elif mode == 'separate':
        q = _choose(rng, [q for q in points if q != p])
        shoot(ids[1], q, palette[1])
    if mode == 'mixed':
        other, more = add_class(2)
        shoot(more[0], tuple(map(int, _choose(rng, np.argwhere(other).tolist()))), palette[2])
    singles = int(rng.integers(0, 3)) if len(bodies) <= 4 else 0
    if mode == 'unmatched':
        singles = max(1, singles)
    for _ in range(singles):
        other, more = add_class(1)
        if mode == 'unmatched' or rng.random() < .4:
            shoot(more[0], tuple(map(int, _choose(rng, np.argwhere(other).tolist()))),
                  _choose(rng, palette))
    return bodies, shots, int(rng.random() < .55)


def _module(rng, body, plans, directions):
    cells = {tuple(map(int, p)): body['color'] for p in np.argwhere(body['mask'])}
    original = set(cells)
    for plan in plans:
        choices = []
        for direction in directions:
            endpoint = plan['point']
            tip = tuple(endpoint[a] - DISTANCE * direction[a] for a in (0, 1))
            arrow = _arrow_cells(tip, direction)
            if arrow & _halo(cells):
                continue
            through = any(tuple(tip[a] + step * direction[a] for a in (0, 1)) in original
                          for step in range(1, DISTANCE))
            if plan['through'] and not through:
                continue
            choices.append(arrow)
        if not choices:
            return None
        arrow = _choose(rng, choices)
        cells.update({p: plan['color'] for p in arrow})
    r0, c0 = (min(p[a] for p in cells) for a in (0, 1))
    return {(r - r0, c - c0): color for (r, c), color in cells.items()}


def _pack(rng, modules, shape):
    occupied = {}
    # Largest spans first, randomized ties; placement uses occupied cells, not boxes.
    modules = [modules[int(i)] for i in rng.permutation(len(modules))]
    modules.sort(key=lambda m: -(max(r for r, c in m) + 1) * (max(c for r, c in m) + 1))
    for module in modules:
        mh, mw = max(r for r, c in module) + 1, max(c for r, c in module) + 1
        if mh > shape[0] or mw > shape[1]:
            return None
        blocked = _halo(occupied)
        positions = [(r, c) for r in range(shape[0] - mh + 1) for c in range(shape[1] - mw + 1)]
        for index in rng.permutation(len(positions)):
            r0, c0 = positions[int(index)]
            shifted = {(r + r0, c + c0): color for (r, c), color in module.items()}
            if not blocked & shifted.keys():
                occupied.update(shifted)
                break
        else:
            return None
    return occupied


def _add_miss(rng, occupied, shape, color, directions):
    candidates = [(r, c, d) for r in range(shape[0]) for c in range(shape[1]) for d in directions]
    blocked = _halo(occupied)
    for index in rng.permutation(len(candidates)):
        r, c, direction = candidates[int(index)]
        cells = _arrow_cells((r, c), direction)
        endpoint = r + DISTANCE * direction[0], c + DISTANCE * direction[1]
        if (all(0 <= y < shape[0] and 0 <= x < shape[1] for y, x in cells | {endpoint})
                and not cells & blocked and endpoint not in occupied):
            occupied.update({p: color for p in cells})
            return True
    return False


def _make_pair(rng, mode, directions, background=None):
    for attempt in range(150):
        bg = int(rng.integers(9)) if background is None else background
        palette = [int(c) for c in rng.permutation(9) if c != bg]
        bodies, shots, misses = _make_plan(rng, mode, palette)
        modules = [_module(rng, b, [s for s in shots if s['body'] == i], directions)
                   for i, b in enumerate(bodies)]
        if any(m is None for m in modules):
            continue
        for placement in range(12):
            floor = 19 if mode == 'teach_merge' or len(bodies) >= 5 else 13
            mh = max(max(r for r, c in m) + 1 for m in modules)
            mw = max(max(c for r, c in m) + 1 for m in modules)
            shape = (int(rng.integers(max(floor, mh), 31)), int(rng.integers(max(floor, mw), 31)))
            if placement >= 9:
                shape = (30, 30)
            elif rng.random() < .2:
                shape = (max(shape), max(shape))
            occupied = _pack(rng, modules, shape)
            if occupied is None or len(occupied) * 2 >= shape[0] * shape[1]:
                continue
            success = True
            for _ in range(misses):
                success &= _add_miss(rng, occupied, shape, palette[-1], directions)
            if not success:
                continue
            inp = np.full(shape, bg, dtype=np.int8)
            for p, color in occupied.items():
                inp[p] = color
            out = solve(inp)
            if np.array_equal(inp, out):
                continue
            # Every returned scene must have visible pixels contributed by the
            # shape correspondence, even when some ink equals a body's color.
            if np.array_equal(_alternative(inp, 'direct'), out):
                continue
            if mode == 'teach_address':
                alternatives = ('direct', 'identity', 'reflection', 'first_obstacle', 'distance4', 'distance6')
            elif mode == 'teach_merge':
                alternatives = ('raw_collision', 'whole_class', 'one_rotation')
            else:
                alternatives = ()
            if any((pred := _alternative(inp, name)) is not None and np.array_equal(pred, out)
                   for name in alternatives):
                continue
            return {'input': inp, 'output': out}
    raise RuntimeError(f'Bounded scene construction exhausted for {mode}')


def _check_counts(num_train, num_test):
    for name, n, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not low <= int(n) <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must be greater than num_test')


def _fingerprint(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError('seed must be a non-negative integer')
    rng = np.random.default_rng(seed)
    # Hold out one cardinal and one diagonal direction, in a rotated frame.
    offset = int(rng.integers(4)) * 2
    heldout = {DIRS[offset], DIRS[(offset + 1) % 8]}
    train_dirs = tuple(d for d in DIRS if d not in heldout)
    test_dirs = DIRS
    stable_bg = int(rng.integers(9)) if rng.random() < .2 else None
    modes = ('echo', 'collision', 'mixed', 'orbit', 'direct_collision', 'separate', 'unmatched')
    train_modes = ['teach_address', 'teach_merge'] + [str(_choose(rng, modes)) for _ in range(num_train - 2)]
    train_modes = [train_modes[int(i)] for i in rng.permutation(num_train)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, specs, directions in (('train', train_modes, train_dirs),
                                      ('test', [str(_choose(rng, modes)) for _ in range(num_test)], test_dirs)):
        for mode in specs:
            for _ in range(20):
                pair = _make_pair(rng, mode, directions, stable_bg)
                key = _fingerprint(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Distinct input sampling exhausted')
    # Require a real held-out-direction query, not merely a larger available pool.
    if not any(a['direction'] in heldout for p in puzzle['test'] for a in _parse(p['input'])[1]):
        for _ in range(40):
            pair = _make_pair(rng, 'echo', tuple(d for d in DIRS if d in heldout), stable_bg)
            if _fingerprint(pair['input']) not in seen:
                puzzle['test'][0] = pair
                break
        else:
            raise RuntimeError('Held-out direction construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(grid):
    bg, arrows, bodies = _parse(grid)
    arrivals, deliveries = _events(arrows, bodies)
    owner = {p: i for i, b in enumerate(bodies) for p in b['cells']}
    raw = Counter(a['endpoint'] for a in arrows)
    classes = Counter(b['canonical'] for b in bodies)
    remote_collision = any(len(ids) > 1 and len({arrows[i]['endpoint'] for i in ids}) > 1
                           for ids in arrivals.values())
    differently_colored_matches = any(a['canonical'] == b['canonical'] and a['color'] != b['color']
                                     for i, a in enumerate(bodies) for b in bodies[i + 1:])
    same_color_nonmatches = any(a['canonical'] != b['canonical'] and a['color'] == b['color']
                               for i, a in enumerate(bodies) for b in bodies[i + 1:])
    mirror_negative = any(a['canonical'] != b['canonical'] and
                          _canonical(np.fliplr(a['mask'])) == b['canonical']
                          for i, a in enumerate(bodies) for b in bodies[i + 1:])
    return dict(shape=grid.shape, background=bg, arrows=len(arrows), bodies=len(bodies),
                classes=tuple(sorted(classes.values())), directions=tuple(sorted({a['direction'] for a in arrows})),
                areas=tuple(sorted({int(b['mask'].sum()) for b in bodies})),
                masks=tuple(sorted({b['canonical'] for b in bodies})),
                misses=sum(a['endpoint'] not in owner for a in arrows),
                raw_collision=any(n > 1 for n in raw.values()), remote_collision=remote_collision,
                symmetry=any(len(cells) > classes[bodies[owner[a['endpoint']]]['canonical']]
                             for a, cells in zip(arrows, deliveries) if a['endpoint'] in owner),
                echo=any(any(p != a['endpoint'] for p in cells) for a, cells in zip(arrows, deliveries)),
                diagonal=any(not _connected(b['mask'], DIRS[::2]) for b in bodies),
                extent=max(max(b['mask'].shape) for b in bodies),
                palette=len({b['color'] for b in bodies}),
                differently_colored_matches=differently_colored_matches,
                same_color_nonmatches=same_color_nonmatches, mirror_negative=mirror_negative,
                collisions=sum(len(ids) > 1 for ids in arrivals.values()))


SHORTCUTS = ('direct', 'identity', 'reflection', 'first_obstacle', 'distance4',
             'distance6', 'raw_collision', 'whole_class', 'one_rotation')


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, train_features, test_features = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair keys')
            inp, out = (_grid(pair[k]) for k in ('input', 'output'))
            _require(inp.shape == out.shape, 'Canvas dimensions change')
            key = _fingerprint(inp)
            _require(key not in seen, 'Duplicate input')
            seen.add(key)
            bg, arrows, bodies = _parse(inp)
            expected = solve(inp)
            _require(np.array_equal(expected, out), 'Incorrect output')
            _require(not np.array_equal(inp, out), 'Uninformative identity example')
            _require(not np.array_equal(_alternative(inp, 'direct'), out),
                     'Shape correspondence has no visible contribution')
            arrivals, _ = _events(arrows, bodies)
            untouched = np.ones(inp.shape, dtype=bool)
            for p in arrivals:
                untouched[p] = False
            _require(np.array_equal(inp[untouched], out[untouched]), 'Unexpected path or object paint')
            _require(all(out[p] == a['color'] for a in arrows for p in a['cells']), 'Arrow changed')
            _require(bool(np.all(out[inp != bg] != bg)), 'Body cells erased')
            _require(all(out[p] == COLLISION for p, ids in arrivals.items() if len(ids) > 1), 'Collision wrong')
            features = _features(inp)
            (train_features if split == 'train' else test_features).append(features)
    for name in SHORTCUTS:
        _require(any((p := _alternative(_grid(pair['input']), name)) is None or
                     not np.array_equal(p, _grid(pair['output'])) for pair in puzzle['train']),
                 f'Training fails to refute {name}')
    _require(any(f['remote_collision'] for f in train_features), 'Need transported collision teaching')
    _require(any(f['symmetry'] for f in train_features), 'Need symmetry-orbit teaching')
    _require(any(f['misses'] for f in train_features), 'Need visible five-step background endpoint')
    _require(any(f['diagonal'] for f in train_features), 'Need eight-connectivity teaching')
    _require(any(3 in f['classes'] for f in train_features), 'Need class threshold teaching')
    for name in ('differently_colored_matches', 'same_color_nonmatches', 'mirror_negative'):
        _require(any(f[name] for f in train_features), f'Need {name} teaching')
    train_directions = {d for f in train_features for d in f['directions']}
    test_directions = {d for f in test_features for d in f['directions']}
    _require(bool(test_directions - train_directions), 'Need orientation shift')


def _regression_tests():
    # Three differently oriented L tetrominoes. Two distant endpoints address
    # the same cell; a third addresses another cell. Expected pixels are explicit.
    inp = np.zeros((23, 26), dtype=np.int8)
    base = np.array([[1, 0], [1, 0], [1, 1]], dtype=bool)
    for top, left, k in ((8, 8, 0), (15, 15, 1), (3, 19, 0)):
        for r, c in np.argwhere(np.rot90(base, k)):
            inp[top + r, left + c] = 6
    for tip, direction, ink in (((10, 4), (0, 1), 2), ((10, 17), (1, 0), 3),
                                ((3, 8), (1, 0), 4)):
        for p in _arrow_cells(tip, direction):
            inp[p] = ink
    expected = inp.copy()
    for p in ((10, 9), (15, 17), (5, 20)):
        expected[p] = 9
    for p in ((8, 8), (16, 15), (3, 19)):
        expected[p] = 4
    assert np.array_equal(solve(inp), expected)
    assert not np.array_equal(_alternative(inp, 'direct'), expected)
    assert not np.array_equal(_alternative(inp, 'raw_collision'), expected)
    # Rotating or reflecting the entire scene preserves the rule (only bodies'
    # relation is rotation-only). This also exercises all compass directions.
    for k in range(4):
        assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(expected, k))
        assert np.array_equal(solve(np.fliplr(np.rot90(inp, k))), np.fliplr(np.rot90(expected, k)))
    # A square ring's corner orbit has four cells, all single impacts.
    inp = np.zeros((16, 16), dtype=np.int8)
    inp[8:11, 8:11] = 6
    inp[9, 9] = 0
    for p in _arrow_cells((3, 8), (1, 0)):
        inp[p] = 2
    expected = inp.copy()
    for p in ((8, 8), (8, 10), (10, 8), (10, 10)):
        expected[p] = 2
    assert np.array_equal(solve(inp), expected)
    assert not np.any(solve(inp) == 9)
    # Four valid rotations taking a square's center to itself still count once.
    inp = np.zeros((16, 16), dtype=np.int8)
    inp[8:11, 8:11] = 6
    for p in _arrow_cells((4, 9), (1, 0)):
        inp[p] = 2
    expected = inp.copy()
    expected[9, 9] = 2
    assert np.array_equal(solve(inp), expected)


def self_test():
    _regression_tests()
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    keys = {tuple(_fingerprint(pair['input']) for s in ('train', 'test') for pair in p[s]) for p in puzzles}
    assert len(keys) == 42
    records = []
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                # Solver must accept a fresh pixel-only nested-list copy as well.
                assert np.array_equal(solve(pair['input'].tolist()), pair['output'])
                records.append(_features(pair['input']))
    assert len({f['shape'] for f in records}) >= 40
    assert len({f['masks'] for f in records}) >= 100
    assert {f['background'] for f in records} == set(range(9))
    assert {d for f in records for d in f['directions']} == set(DIRS)
    assert len({f['classes'] for f in records}) >= 8
    assert {2, 3, 4} <= {n for f in records for n in f['classes']}
    assert len({f['arrows'] for f in records}) >= 4
    assert len({f['bodies'] for f in records}) >= 4
    assert max(f['extent'] for f in records) >= 6
    for name in ('remote_collision', 'raw_collision', 'symmetry', 'diagonal', 'misses'):
        assert sum(bool(f[name]) for f in records) >= 8, name
    for tr in range(2, 11):
        for te in range(1, min(4, tr - 1) + 1):
            p = generate_puzzle(91 + tr * 10 + te, tr, te)
            assert (len(p['train']), len(p['test'])) == (tr, te)
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('Illegal counts accepted')
    repeat = generate_puzzle(0)
    assert all(np.array_equal(puzzles[0][s][i][k], repeat[s][i][k])
               for s in ('train', 'test') for i in range(len(repeat[s])) for k in ('input', 'output'))
    # A collision-color corruption must be caught even on a background pixel.
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('Corrupted output accepted')
    return {'seeds': 42, 'pairs': len(records), 'shapes': len({f['shape'] for f in records}),
            'class_profiles': len({f['classes'] for f in records}),
            'body_vocabularies': len({f['masks'] for f in records})}
