"""Forecast volleys: accelerated formation spacing becomes projectile range.

The single dot of each content color is its time origin. Three or four arrow
snapshots establish motion; only the next snapshot exists in the output scene.
No training labels, seed, or generator state are used by solve().
"""
from collections import Counter
from itertools import product

import numpy as np


VECTORS = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)
ROLE_POOL = (0, 1, 2, 3, 4, 5, 6, 9)
SINGLE, COLLISION = 8, 7


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _cells(tip, heading):
    r, c = tip
    dr, dc = heading
    if dr == 0:
        offsets = ((0, 0), (-1, -dc), (1, -dc))
    elif dc == 0:
        offsets = ((0, 0), (-dr, -1), (-dr, 1))
    else:
        offsets = ((0, 0), (-dr, 0), (0, -dc))
    return tuple((r + a, c + b) for a, b in offsets)


def _plus(point, scale, vector):
    return tuple(int(point[j] + scale * vector[j]) for j in (0, 1))


def _marker_cells(bounds):
    r0, r1, c0, c1 = bounds
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo, component = [first], {first}
        while todo:
            r, c = todo.pop()
            new = {(r + a, c + b) for a, b in product((-1, 0, 1), repeat=2)} & remaining
            remaining -= new
            component |= new
            todo.extend(new)
        result.append(component)
    return result


def _parse(input_grid):
    a = np.asarray(input_grid)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "bad grid shape")
    _require(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _require(bool(np.all((0 <= a) & (a <= 9))), "bad color")
    colors, counts = np.unique(a, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _require(int(max(counts)) > a.size / 2, "background must be a strict majority")
    _require(not np.any((a == SINGLE) | (a == COLLISION)), "impact colors are reserved")
    candidates = []
    for color, count in zip(colors, counts):
        if count != 6:
            continue
        points = np.argwhere(a == color)
        r0, c0 = map(int, points.min(axis=0))
        r1, c1 = map(int, points.max(axis=0))
        bb = r0, r1, c0, c1
        if r1 - r0 >= 2 and c1 - c0 >= 2 and set(map(tuple, points)) == _marker_cells(bb):
            candidates.append((int(color), bb))
    _require(len(candidates) == 1, "need two canonical corner L markers")
    marker, bounds = candidates[0]
    forecasts = []
    for color in colors:
        color = int(color)
        if color in (bg, marker):
            continue
        components = _components(map(tuple, np.argwhere(a == color)))
        dots = [c for c in components if len(c) == 1]
        heads = [c for c in components if len(c) == 3]
        _require(len(dots) == 1 and len(heads) in (3, 4) and len(components) == len(heads) + 1,
                 "each stream needs one origin dot and three or four separated arrows")
        origin = next(iter(dots[0]))
        observations = []
        for component in heads:
            matches = [(tip, v) for tip in sorted(component) for v in VECTORS
                       if set(_cells(tip, v)) == component]
            _require(len(matches) == 1, "arrow direction must be unique")
            observations.append(matches[0])
        heading = observations[0][1]
        _require(all(v == heading for _, v in observations), "a stream keeps its heading")
        observations.sort(key=lambda x: max(abs(x[0][j] - origin[j]) for j in (0, 1)))
        first = observations[0][0]
        unit = tuple(int(np.sign(first[j] - origin[j])) for j in (0, 1))
        distances = [max(abs(tip[j] - origin[j]) for j in (0, 1)) for tip, _ in observations]
        _require(unit in VECTORS and all(tip == _plus(origin, s, unit)
                                       for (tip, _), s in zip(observations, distances)),
                 "tips and origin must lie on one directed compass line")
        gaps = np.diff([0] + distances).tolist()
        delta = gaps[1] - gaps[0]
        _require(delta in (-1, 0, 1) and min(gaps) >= 3 and
                 all(g == gaps[0] + i * delta for i, g in enumerate(gaps)),
                 "successive spacings must form an arithmetic progression")
        upcoming = gaps[-1] + delta
        _require(upcoming >= 3, "next spacing must remain at least three")
        last = observations[-1][0]
        forecasts.append(dict(color=color, tip=_plus(last, upcoming, unit), heading=heading,
                              distance=upcoming, unit=unit, delta=delta, gaps=tuple(gaps),
                              last=last, count=len(heads)))
    _require(1 <= len(forecasts) <= 4, "need one to four streams")
    return a, bg, marker, bounds, forecasts


def _events(objects):
    owner = {}
    for i, obj in enumerate(objects):
        for p in _cells(obj['tip'], obj['heading']):
            _require(p not in owner, "forecast arrows may not overlap")
            owner[p] = i
    counts = Counter(_plus(o['tip'], o['distance'], o['heading']) for o in objects)
    hit_roles = {}
    for point, n in counts.items():
        if point in owner:
            hit_roles.setdefault(owner[point], set()).add(n > 1)
    _require(all(len(v) == 1 for v in hit_roles.values()), "incompatible whole-arrow hit roles")
    return owner, counts, hit_roles


def _scene(shape, bg, objects, strict=True, point_hits=False, ignore_collisions=False):
    """Paint one simultaneous future snapshot, never the historical traces."""
    owner, counts, hits = _events(objects)
    if strict:
        _require(all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in list(owner) + list(counts)),
                 "forecast and endpoints must be within the input frame")
    out = np.full(shape, bg, dtype=np.int8)
    def paint(point, color):
        r, c = point
        if 0 <= r < shape[0] and 0 <= c < shape[1]:
            out[r, c] = color
    for p, i in owner.items():
        paint(p, objects[i]['color'])
    for p, n in counts.items():
        color = COLLISION if n > 1 and not ignore_collisions else SINGLE
        targets = _cells(objects[owner[p]]['tip'], objects[owner[p]]['heading']) if p in owner and not point_hits else (p,)
        for cell in targets:
            paint(cell, color)
    return out


def _crop(scene, bounds):
    r0, r1, c0, c1 = bounds
    return scene[r0:r1 + 1, c0:c1 + 1].copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer every next pose and range exclusively from this input's pixels."""
    a, bg, _, bounds, objects = _parse(input_grid)
    return _crop(_scene(a.shape, bg, objects), bounds)


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _separated(objects):
    used = set()
    for o in objects:
        cells = _cells(o['tip'], o['heading'])
        if any((r + a, c + b) in used for r, c in cells for a, b in product((-1, 0, 1), repeat=2)):
            return False
        used.update(cells)
    return True


def _profile(rng, n, growth=False):
    delta = int(_choose(rng, (-1, 1) if growth else (-1, 0, 0, 0, 1)))
    # All observed gaps and the next gap are >= 3. The maximum path is 25.
    if delta == 1:
        gap = n + 3
    elif delta == -1:
        gap = 3
    else:
        gap = int(_choose(rng, (3, 3, 4, 4, 5)))
    return dict(distance=gap, delta=delta, count=n)


def _module(rng, kind, n):
    k = 1 if kind == 'single' else 3 if kind in ('triple', 'collision_strike', 'multi') else 2
    objects = [_profile(rng, n, growth=(kind == 'collision' and i == 0)) for i in range(k)]
    for obj in objects:
        obj.update(tip=(0, 0), heading=_choose(rng, VECTORS))
    if kind in ('collision', 'triple'):
        headings = [VECTORS[int(i)] for i in rng.choice(len(VECTORS), size=k, replace=False)]
        for obj, v in zip(objects, headings):
            obj.update(tip=_plus((0, 0), -obj['distance'], v), heading=v)
    elif kind in ('strike', 'collision_strike', 'multi'):
        target = objects[0]
        cells = _cells(target['tip'], target['heading'])
        hit = _choose(rng, cells)
        for i, shooter in enumerate(objects[1:]):
            destination = cells[(cells.index(hit) + i) % 3] if kind == 'multi' else hit
            shooter['tip'] = _plus(destination, -shooter['distance'], shooter['heading'])
    elif kind == 'mutual':
        objects[1].update(distance=objects[0]['distance'], delta=objects[0]['delta'],
                          heading=tuple(-x for x in objects[0]['heading']),
                          tip=_plus((0, 0), objects[0]['distance'], objects[0]['heading']))
    elif kind == 'through':
        first = objects[0]
        objects[1]['tip'] = _plus(first['tip'], max(2, first['distance'] - 1), first['heading'])
    elif kind == 'cross':
        first, second = objects
        # Interior rays meet at the origin, with distinct endpoints.
        first['tip'] = _plus((0, 0), -2, first['heading'])
        second['tip'] = _plus((0, 0), -2, second['heading'])
    if not _separated(objects):
        return None
    try:
        owner, counts, hits = _events(objects)
    except ValueError:
        return None
    if kind in ('collision', 'triple') and (not any(v > 1 for v in counts.values()) or hits):
        return None
    if kind in ('strike', 'multi', 'mutual') and (not hits or any(v > 1 for v in counts.values())):
        return None
    if kind == 'collision_strike' and not any(n > 1 and p in owner for p, n in counts.items()):
        return None
    if kind == 'multi' and not any(sum(p in _cells(objects[i]['tip'], objects[i]['heading'])
                                      for p in counts) > 1 for i in hits):
        return None
    return objects


def _history(obj, unit):
    n, delta, next_gap = obj['count'], obj['delta'], obj['distance']
    first_gap = next_gap - n * delta
    length = (n + 1) * first_gap + delta * n * (n + 1) // 2
    origin = _plus(obj['tip'], -length, unit)
    entries = {origin}
    tips = []
    walked = 0
    for i in range(n):
        walked += first_gap + i * delta
        tip = _plus(origin, walked, unit)
        tips.append(tip)
        entries.update(_cells(tip, obj['heading']))
    # Do not permit the origin/arrow components of this color to merge.
    if sorted(map(len, _components(entries))) != [1] + [3] * n:
        return None
    return entries, tips, origin


def _extent(points):
    return tuple(min(p[j] for p in points) for j in (0, 1)), tuple(max(p[j] for p in points) for j in (0, 1))


def _assemble(rng, objects, bg, marker, colors, crop_mode, units=None, padding=None, allowed_units=None):
    """Explicit latent histories and future poses provide the construction label."""
    future = {p for o in objects for p in _cells(o['tip'], o['heading'])}
    endpoints = {_plus(o['tip'], o['distance'], o['heading']) for o in objects}
    forecast_points = future | endpoints
    source = {}
    all_points = set(forecast_points)
    # Search the eight history directions per stream, with immediate bounding
    # and overlap pruning. This is finite geometry, not unbounded rejection.
    direction_pool = VECTORS if allowed_units is None else allowed_units
    choices = [([units[i]] if units else [direction_pool[int(j)] for j in rng.permutation(len(direction_pool))])
               for i in range(len(objects))]
    picked = []
    def place(index):
        if index == len(objects):
            return True
        obj = objects[index]
        for unit in choices[index]:
            history = _history(obj, unit)
            if history is None:
                continue
            points, tips, origin = history
            if any(p in source for p in points):
                continue
            low, high = _extent(all_points | points)
            if max(high[j] - low[j] + 1 for j in (0, 1)) > 28:
                continue
            color = colors[index]
            for p in points:
                source[p] = color
            old = set(all_points)
            all_points.update(points)
            picked.append((unit, tips, origin))
            if place(index + 1):
                return True
            picked.pop()
            all_points.clear()
            all_points.update(old)
            for p in points:
                del source[p]
        return False
    if not place(0):
        return None
    low, high = _extent(forecast_points)
    if crop_mode == 'whole':
        base = (low[0], high[0], low[1], high[1])
    else:
        # Tight views of an impact or forecast arrow; the other emitters still
        # contribute even when they are outside this requested window.
        center = _choose(rng, sorted(endpoints if crop_mode == 'impact' else future))
        hr, hc = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        base = (center[0] - hr, center[0] + hr, center[1] - hc, center[1] + hc)
    for _ in range(24):
        r0, r1, c0, c1 = base
        if r1 - r0 < 2:
            r0 -= 1
            r1 += 1
        if c1 - c0 < 2:
            c0 -= 1
            c1 += 1
        # Optional margins, with no mandatory blank edge in the answer.
        margins = rng.integers(0, 3 if crop_mode == 'whole' else 2, size=4)
        bounds = (r0 - int(margins[0]), r1 + int(margins[1]),
                  c0 - int(margins[2]), c1 + int(margins[3]))
        if crop_mode != 'whole' and all(bounds[0] <= p[0] <= bounds[1] and
                                        bounds[2] <= p[1] <= bounds[3] for p in forecast_points):
            continue
        if any(p in source for p in _marker_cells(bounds)):
            continue
        lo, hi = _extent(all_points | _marker_cells(bounds))
        size = [hi[j] - lo[j] + 1 for j in (0, 1)]
        if max(size) > 30:
            continue
        if padding is None:
            pads = [int(rng.integers(min(2, 30 - size[j]) + 1)) for j in (0, 1)]
            before = [int(rng.integers(v + 1)) for v in pads]
        else:
            before, pads = padding
            if any(size[j] + pads[j] > 30 for j in (0, 1)):
                continue
        offset = tuple(before[j] - lo[j] for j in (0, 1))
        shape = tuple(size[j] + pads[j] for j in (0, 1))
        grid = np.full(shape, bg, dtype=np.int8)
        for p, color in source.items():
            grid[_plus(p, 1, offset)] = color
        bb = (bounds[0] + offset[0], bounds[1] + offset[0],
              bounds[2] + offset[1], bounds[3] + offset[1])
        for p in _marker_cells(bb):
            grid[p] = marker
        moved = [dict(o, color=colors[i], tip=_plus(o['tip'], 1, offset), unit=picked[i][0])
                 for i, o in enumerate(objects)]
        answer = _crop(_scene(shape, bg, moved), bb)
        if not np.any(answer != bg):
            continue
        return {'input': grid, 'output': answer}
    return None


def _transform(pair, turns, reflect):
    # Rotate/reflect the scene, then redraw the canonical annotation orientation.
    a, bg, marker, bounds, _ = _parse(pair['input'])
    mask = np.zeros(a.shape, dtype=np.int8)
    r0, r1, c0, c1 = bounds
    mask[r0:r1 + 1, c0:c1 + 1] = 1
    clean = a.copy()
    clean[clean == marker] = bg
    grids = [clean, mask, pair['output']]
    grids = [np.rot90(x, turns).copy() for x in grids]
    if reflect:
        grids = [x[:, ::-1].copy() for x in grids]
    clean, mask, out = grids
    points = np.argwhere(mask)
    r0, c0 = map(int, points.min(axis=0))
    r1, c1 = map(int, points.max(axis=0))
    bb = r0, r1, c0, c1
    if any(clean[p] != bg for p in _marker_cells(bb)):
        return None
    for p in _marker_cells(bb):
        clean[p] = marker
    return {'input': clean, 'output': out}


def _counterfactual(a, bg, bounds, objects, mode):
    changed = []
    for o in objects:
        v = dict(o)
        if mode == 'no_motion':
            v['tip'] = o['last']
        elif mode == 'constant_gap':
            v.update(tip=_plus(o['last'], o['gaps'][-1], o['unit']), distance=o['gaps'][-1])
        elif mode == 'fixed_five':
            v['distance'] = 5
        changed.append(v)
    try:
        return _crop(_scene(a.shape, bg, changed, strict=False,
                            point_hits=(mode == 'point_hits'), ignore_collisions=(mode == 'no_collision')), bounds)
    except ValueError:
        return None


def _teaches(pair, kind):
    a, bg, _, bounds, objects = _parse(pair['input'])
    out = pair['output']
    if not np.array_equal(solve(a), out):
        return False
    if kind == 'collision':
        if not np.any(out == COLLISION) or not any(o['delta'] for o in objects):
            return False
        if not any(o['unit'][0] * o['heading'][1] != o['unit'][1] * o['heading'][0] for o in objects):
            return False
        for mode in ('no_motion', 'constant_gap', 'fixed_five', 'no_collision'):
            alternative = _counterfactual(a, bg, bounds, objects, mode)
            if alternative is None or np.array_equal(alternative, out):
                return False
    if kind == 'strike':
        _, counts, hits = _events(objects)
        if not hits or not any(p not in _events(objects)[0] and n == 1 for p, n in counts.items()):
            return False
        if np.array_equal(_counterfactual(a, bg, bounds, objects, 'point_hits'), out):
            return False
    return True


def _fallback(rng, kind, n, bg, marker, colors, variant, crop_mode='whole'):
    # Two directly constructible teaching arrangements. Their histories move
    # vertically while shots travel horizontally, so time order != heading.
    if kind == 'collision':
        g = n + 3
        objects = [dict(tip=(0, -g), heading=(0, 1), distance=g, delta=1, count=n),
                   dict(tip=(0, 4), heading=(0, -1), distance=4, delta=0, count=n)]
    else:
        objects = [dict(tip=(0, 0), heading=(1, 0), distance=4, delta=0, count=n),
                   dict(tip=(0, -4), heading=(0, 1), distance=4, delta=0, count=n)]
    # Thirty-two distinct padding arrangements, larger than the maximum puzzle.
    pads = [variant // 16, variant % 4 + (variant // 4) % 4]
    before = [variant // 16, variant % 4]
    return _assemble(rng, objects, bg, marker, colors[:2], crop_mode,
                     units=[(1, 0), (1, 0)], padding=(before, pads))


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(name + ' is out of range or not an integer')
    return int(value)


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 2, 10, 'num_train')
    num_test = _count(num_test, 1, 4, 'num_test')
    _require(num_train > num_test, 'num_train must exceed num_test')
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_)) and seed >= 0,
             'seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    roles = list(map(int, rng.permutation(ROLE_POOL)))
    bg, marker = roles[:2]
    color_pool = roles[2:]
    shift = _choose(rng, ('turn', 'history', 'crop', 'combined'))
    turn = int(rng.integers(4))
    reflect = bool(rng.integers(2))
    train_kinds = ['collision', 'strike'] + [_choose(rng, ('single', 'collision', 'strike', 'mutual',
                                                           'multi', 'triple', 'collision_strike', 'through', 'cross'))
                                               for _ in range(num_train - 2)]
    rng.shuffle(train_kinds)
    tests = [_choose(rng, ('single', 'collision', 'strike', 'mutual', 'multi', 'triple',
                           'collision_strike', 'through', 'cross')) for _ in range(num_test)]
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, kinds in (('train', train_kinds), ('test', tests)):
        for kind in kinds:
            n = 4 if split == 'test' and shift in ('history', 'combined') else 3
            crop_mode = 'impact' if split == 'test' and shift in ('crop', 'combined') else 'whole'
            turns = (turn + int(split == 'test')) % 4 if shift in ('turn', 'combined') else int(rng.integers(4))
            for attempt in range(160):
                colors = [int(c) for c in rng.permutation(color_pool)]
                if attempt < 128:
                    objects = _module(rng, kind, n)
                    if objects is None:
                        continue
                    # A fourth, independent moving emitter can interact with the
                    # event or remain an ordinary shot in richer examples.
                    if len(objects) < 4 and kind not in ('collision', 'strike') and rng.random() < .22:
                        extra = _profile(rng, n)
                        extra.update(tip=(int(rng.integers(-7, 8)), int(rng.integers(-7, 8))), heading=_choose(rng, VECTORS))
                        if _separated(objects + [extra]):
                            try:
                                _events(objects + [extra])
                                objects.append(extra)
                            except ValueError:
                                pass
                    pair = _assemble(rng, objects, bg, marker, colors, crop_mode,
                                     allowed_units=((1, 0), (-1, 0)) if shift == 'turn' else None)
                else:
                    fallback_kind = 'collision' if kind in ('collision', 'triple') else 'strike'
                    pair = _fallback(rng, fallback_kind, n, bg, marker, colors, attempt - 128, crop_mode)
                if pair is None:
                    continue
                pair = _transform(pair, turns, reflect)
                if pair is None or _key(pair['input']) in seen:
                    continue
                if not _teaches(pair, kind if split == 'train' and kind in ('collision', 'strike') else ''):
                    continue
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
            else:
                raise RuntimeError('bounded construction exhausted')
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'bad puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _count(len(puzzle['train']), 2, 10, 'num_train')
    _count(len(puzzle['test']), 1, 4, 'num_test')
    _require(len(puzzle['train']) > len(puzzle['test']), 'bad split counts')
    seen, roles, teaching = set(), set(), []
    train_axes, test_records = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'bad pair keys')
            for grid in pair.values():
                a = np.asarray(grid)
                _require(a.ndim == 2 and all(1 <= v <= 30 for v in a.shape) and
                         np.issubdtype(a.dtype, np.integer) and bool(np.all((0 <= a) & (a <= 9))), 'bad grid')
            a, bg, marker, bounds, objects = _parse(pair['input'])
            roles.add((bg, marker))
            _require(_key(a) not in seen, 'duplicate input')
            seen.add(_key(a))
            _require(_separated(objects), 'forecast glyphs must be separated')
            predicted = solve(a.copy())
            _require(np.array_equal(predicted, pair['output']), 'wrong output')
            _require(marker not in predicted and np.any(predicted != bg), 'empty or annotated output')
            _require(predicted.shape == (bounds[1] - bounds[0] + 1, bounds[3] - bounds[2] + 1), 'wrong crop')
            if split == 'train':
                teaching.append(pair)
                _require(all(o['count'] == 3 for o in objects), 'training uses three observed poses')
                train_axes.update(tuple(abs(v) for v in o['unit']) for o in objects)
            else:
                points = list(_events(objects)[0]) + list(_events(objects)[1])
                partial = any(not (bounds[0] <= p[0] <= bounds[1] and bounds[2] <= p[1] <= bounds[3]) for p in points)
                test_records.append((objects, partial))
    _require(len(roles) == 1, 'background and marker are stable within a puzzle')
    _require(any(_teaches(p, 'collision') for p in teaching), 'need acceleration/range/collision evidence')
    _require(any(_teaches(p, 'strike') for p in teaching), 'need whole-arrow strike evidence')
    for objects, partial in test_records:
        new_axes = {tuple(abs(v) for v in o['unit']) for o in objects}
        _require(all(o['count'] == 4 for o in objects) or partial or not (new_axes & train_axes),
                 'test must shift history length, crop coverage, or motion axis')


def self_test():
    """Bounded semantic regressions, ablations, counts, and 42-seed diversity."""
    shapes, stream_counts, headings, units, deltas, ranges, views, palettes = [set() for _ in range(8)]
    keys = set()
    features = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        keys.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a, bg, marker, bounds, objects = _parse(pair['input'])
                shapes.add(a.shape)
                views.add(pair['output'].shape)
                palettes.add((bg, marker, tuple(sorted(o['color'] for o in objects))))
                stream_counts.add(len(objects))
                owner, counts, hits = _events(objects)
                features['collisions'] += any(n > 1 for n in counts.values())
                features['strikes'] += bool(hits)
                features['multiple_hits'] += any(sum(p in _cells(objects[i]['tip'], objects[i]['heading']) for p in counts) > 1 for i in hits)
                features['triple'] += any(n >= 3 for n in counts.values())
                features['source_inside_crop'] += bool(np.any((a[bounds[0]:bounds[1]+1, bounds[2]:bounds[3]+1] != bg) &
                                                            (a[bounds[0]:bounds[1]+1, bounds[2]:bounds[3]+1] != marker)))
                features['outside_emitter'] += any(not (bounds[0] <= o['tip'][0] <= bounds[1] and bounds[2] <= o['tip'][1] <= bounds[3]) for o in objects)
                for o in objects:
                    headings.add(o['heading'])
                    units.add(o['unit'])
                    deltas.add(o['delta'])
                    ranges.add(o['distance'])
                    features['four_observations'] += o['count'] == 4
                    features['sideways'] += o['unit'][0]*o['heading'][1] != o['unit'][1]*o['heading'][0]
                # Snapshot semantics must not depend on iteration order.
                assert np.array_equal(_scene(a.shape, bg, objects), _scene(a.shape, bg, objects[::-1]))
    assert len(keys) == 42 and len(shapes) >= 50 and len(views) >= 30 and len(palettes) >= 80
    assert stream_counts >= {1, 2, 3} and headings == set(VECTORS) and units == set(VECTORS)
    assert deltas == {-1, 0, 1} and len(ranges) >= 4
    assert all(features[k] >= 2 for k in ('collisions', 'strikes', 'multiple_hits', 'triple',
                                         'four_observations', 'sideways', 'source_inside_crop', 'outside_emitter')), features
    # Exact event regressions: a struck shooter still fires, all three target
    # pixels change, distinct single hits are not counted as a collision.
    target = dict(tip=(8, 8), heading=(0, 1), color=2, distance=5)
    objs = [target, dict(tip=(13, 8), heading=(-1, 0), color=3, distance=5),
            dict(tip=(12, 2), heading=(-1, 1), color=4, distance=5)]
    out = _scene((20, 20), 0, objs)
    assert all(out[p] == SINGLE for p in _cells(target['tip'], target['heading']))
    assert out[8, 13] == SINGLE and COLLISION not in out
    # Crossing interiors leave the intersection unpainted.
    objs = [dict(tip=(8, 5), heading=(0, 1), color=1, distance=5),
            dict(tip=(5, 8), heading=(1, 0), color=2, distance=5)]
    out = _scene((16, 16), 0, objs)
    assert out[8, 8] == 0 and out[8, 10] == SINGLE and out[10, 8] == SINGLE
    # A path may pass through another forecast arrow without striking it.
    objs = [dict(tip=(9, 3), heading=(0, 1), color=1, distance=5),
            dict(tip=(9, 6), heading=(-1, 0), color=2, distance=5)]
    out = _scene((16, 16), 0, objs)
    assert all(out[p] == 2 for p in _cells((9, 6), (-1, 0)))
    assert out[9, 8] == SINGLE and out[4, 6] == SINGLE and out[9, 5] == 0
    # Input-only inference also covers valid members outside the constructor's
    # preferred profiles: mixed history lengths and a next gap of eight.
    rng = np.random.default_rng(52)
    mixed = [dict(tip=(0, 0), heading=(0, 1), distance=7, delta=1, count=3),
             dict(tip=(4, 0), heading=(0, 1), distance=3, delta=0, count=4)]
    pair = _assemble(rng, mixed, 0, 9, [1, 2], 'whole', units=[(1, 0), (0, 1)])
    assert pair is not None and np.array_equal(solve(pair['input']), pair['output'])
    assert {o['count'] for o in _parse(pair['input'])[4]} == {3, 4}
    pair = _assemble(rng, [dict(tip=(0, 0), heading=(0, 1), distance=8, delta=1, count=3)],
                     0, 9, [1], 'whole', units=[(1, 0)])
    assert pair is not None and np.array_equal(solve(pair['input']), pair['output'])
    assert _parse(pair['input'])[4][0]['distance'] == 8
    for n in range(2, 11):
        for t in range(1, min(4, n - 1) + 1):
            validate_generated_puzzle(generate_puzzle(101, n, t))
    first, second = generate_puzzle(813, 2, 1), generate_puzzle(813, 2, 1)
    assert all(_key(a[k]) == _key(b[k]) for s in first for a, b in zip(first[s], second[s]) for k in a)
    for n, t in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, n, t)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    broken = generate_puzzle(44, 2, 1)
    broken['test'][0]['output'][0, 0] = (int(broken['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupted label accepted')
    return {'seeds': 42, 'input_shapes': len(shapes), 'output_shapes': len(views),
            'stream_counts': sorted(stream_counts), 'ranges': sorted(ranges), 'features': dict(features)}
