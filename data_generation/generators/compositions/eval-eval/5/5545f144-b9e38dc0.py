"""A lantern paints finite, shadowed exposures along a tip-led waypoint route.

All solving state is reconstructed from the input. No parent code is imported.
The sampler renders unions of geometric masks; solve uses per-exposure lane scans.
"""
from __future__ import annotations

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer 2D arrays, 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _rot(point, turns):
    r, c = point
    for _ in range(turns % 4):
        r, c = c, -r
    return int(r), int(c)


def _pose(mask, tip, direction):
    return {(tip[0] + r, tip[1] + c)
            for r, c in (_rot(p, direction - 1) for p in mask)}


def _box(mask, tip, direction):
    cells = _pose(mask, tip, direction) | {tip}
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _fits(box, shape):
    return 0 <= box[0] <= box[1] < shape[0] and 0 <= box[2] <= box[3] < shape[1]


def _near_box(point, box, padding=0):
    r, c = point
    return box[0]-padding <= r <= box[1]+padding and box[2]-padding <= c <= box[3]+padding


def _direction(a, b):
    dr, dc = b[0]-a[0], b[1]-a[1]
    if (dr == 0) == (dc == 0):
        raise ValueError("route points must be distinct and orthogonal")
    return (1 if dc > 0 else 3) if dr == 0 else (2 if dr > 0 else 0)


def _distance(a, b):
    return abs(b[0]-a[0]) + abs(b[1]-a[1])


def _route(tip, waypoints):
    points, remaining, directions = [tip], set(waypoints), []
    while remaining:
        rays = {}
        for q in sorted(remaining):
            if q[0] == tip[0] or q[1] == tip[1]:
                rays.setdefault(_direction(tip, q), []).append(q)
        if len(rays) != 1:
            raise ValueError("exactly one orthogonal ray must contain remaining beacons")
        d, candidates = next(iter(rays.items()))
        tip = min(candidates, key=lambda q: _distance(points[-1], q))
        remaining.remove(tip)
        points.append(tip)
        directions.append(d)
    return points, directions


def _fit_stair(heights):
    """Infer a terminal thin stair from at least two complete width increases."""
    end = int(heights[-1])
    candidates = []
    xs = np.arange(len(heights)) - len(heights) + 1
    for run in (1, 2):
        for phase in range(run):
            expected = end + (xs + phase) // run
            length = 0
            for actual, predicted in zip(reversed(heights), reversed(expected)):
                if actual != predicted:
                    break
                length += 1
            if length >= 2*run+1 and heights[-1]-heights[-length] >= 2:
                candidates.append((length, -run, phase))
    if not candidates:
        raise ValueError("lantern needs two visible outward stair steps")
    length, negative_run, phase = max(candidates)
    return -negative_run, phase, end, length


def _check_route(points, directions, mask, shape):
    assert directions and directions[0] == 1
    recovered, ds = _route(points[0], points[1:])
    assert recovered == points and ds == directions
    boxes = []
    for i, p in enumerate(points):
        poses = {directions[max(0, i-1)]}
        if i < len(directions):
            poses.add(directions[i])
        for d in poses:
            box = _box(mask, p, d)
            assert _fits(box, shape), "all incoming and outgoing poses must fit"
            assert all(not _near_box(q, box, 1) for q in points[i+1:])
            boxes.append(box)
    return boxes


def _read_input(value):
    original = _grid(value)
    counts = np.bincount(original.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= original.size // 2:
        raise ValueError("input background must occupy more than half the grid")
    matches = []
    for initial in range(4):
        # In this canonical view the initial lantern faces right.
        a = np.rot90(original, initial-1).copy()
        for wall in map(int, np.unique(a)):
            if wall == bg:
                continue
            cells = np.argwhere(a == wall)
            if len(cells) < 9:
                continue
            rear, mouth = int(cells[:, 1].min()), int(cells[:, 1].max())
            if mouth-rear < 3 or mouth+1 >= a.shape[1]:
                continue
            cap = np.flatnonzero(a[:, rear] == wall)
            if (len(cap) < 3 or len(cap) % 2 != 1 or np.any(np.diff(cap) != 1)):
                continue
            center = int((cap[0]+cap[-1])//2)
            heights = []
            for x in range(rear+1, mouth+1):
                rows = np.flatnonzero(a[:, x] == wall)
                if len(rows) != 2 or rows[0]+rows[1] != 2*center:
                    break
                heights.append(center-int(rows[0]))
            if len(heights) != mouth-rear or min(heights) < 1:
                continue
            profile = [(len(cap)-1)//2] + heights
            if any(abs(b-c) > 1 for b, c in zip(profile, profile[1:])):
                continue
            try:
                run, phase, half, evidence = _fit_stair(heights)
            except ValueError:
                continue
            tip = (center, mouth+1)
            ink = int(a[tip])
            if ink in (bg, wall):
                continue
            waypoints = set(map(tuple, np.argwhere(a == ink))) - {tip}
            if not 2 <= len(waypoints) <= 8:
                continue
            if any(max(abs(p[0]-q[0]), abs(p[1]-q[1])) <= 1
                   for p in waypoints for q in waypoints if p != q):
                continue
            mask = tuple(sorted((int(r)-center, int(c)-tip[1]) for r, c in cells))
            try:
                points, directions = _route(tip, waypoints)
                boxes = _check_route(points, directions, mask, a.shape)
            except (ValueError, AssertionError):
                continue
            extras = (a != bg) & (a != wall) & (a != ink)
            protected = _protected(a.shape, points, directions, boxes)
            if np.any(extras & protected):
                continue
            matches.append(dict(a=a, bg=bg, wall=wall, ink=ink, mask=mask,
                                points=points, directions=directions, boxes=boxes,
                                run=run, phase=phase, half=half, evidence=evidence,
                                extras=extras, initial=initial, profile=tuple(profile)))
    if len(matches) != 1:
        raise ValueError(f"input must identify one lantern and route; got {len(matches)}")
    return matches[0]


def _protected(shape, points, directions, boxes):
    protected = np.zeros(shape, dtype=bool)
    for r0, r1, c0, c1 in boxes:
        protected[r0:r1+1, c0:c1+1] = True
    # Travel is unobstructed on its centerline; side lanes can be occluded.
    for p, q, d in zip(points, points[1:], directions):
        dr, dc = DIRS[d]
        for t in range(_distance(p, q)+1):
            protected[p[0]+dr*t, p[1]+dc*t] = True
    return protected


def _footprints(shape, points, directions, run, phase, half):
    ys, xs = np.indices(shape)
    result = []
    for p, q, d in zip(points, points[1:], directions):
        dr, dc = DIRS[d]
        axial = (ys-p[0])*dr + (xs-p[1])*dc
        lateral = -(ys-p[0])*dc + (xs-p[1])*dr
        width = half + (axial+1+phase)//run
        footprint = (axial >= 0) & (axial <= _distance(p, q)) & (np.abs(lateral) < width)
        result.append((footprint, axial, lateral))
    return result


def _mask_exposures(footprints, extras):
    exposures, shadows = [], []
    for footprint, axial, lateral in footprints:
        shadow = np.zeros(extras.shape, dtype=bool)
        for r, c in np.argwhere(extras & footprint):
            shadow |= footprint & (lateral == lateral[r, c]) & (axial > axial[r, c])
        exposures.append(footprint & ~shadow & ~extras)
        shadows.append(shadow & ~extras)
    return exposures, shadows


def _scan_exposures(info):
    """The solving path enumerates lanes; it does not use construction masks."""
    a, points, ds = info['a'], info['points'], info['directions']
    exposures = []
    for p, q, d in zip(points, points[1:], ds):
        dr, dc = DIRS[d]
        painted = np.zeros(a.shape, dtype=bool)
        blocked_lanes = set()
        for t in range(_distance(p, q)+1):
            half = info['half'] + (t+1+info['phase'])//info['run']
            for lane in range(-half+1, half):
                r = p[0] + dr*t - dc*lane
                c = p[1] + dc*t + dr*lane
                if not (0 <= r < a.shape[0] and 0 <= c < a.shape[1]):
                    continue
                if info['extras'][r, c]:
                    blocked_lanes.add(lane)
                elif lane not in blocked_lanes:
                    painted[r, c] = True
        exposures.append(painted)
    return exposures


def _finish(info, light):
    out = np.full(info['a'].shape, info['bg'], dtype=np.int8)
    out[light & ~info['extras']] = info['ink']
    out[info['extras']] = info['a'][info['extras']]
    final = info['points'][-1]
    for p in _pose(info['mask'], final, info['directions'][-1]):
        out[p] = info['wall']
    out[final] = info['ink']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    info = _read_input(input_grid)
    light = np.logical_or.reduce(_scan_exposures(info))
    return np.rot90(_finish(info, light), 1-info['initial']).copy()


def _sample_housing(rng):
    run = int(rng.integers(1, 3))
    phase = int(rng.integers(run))
    cap = int(rng.integers(1, 3))
    heights, last = [], cap
    for _ in range(int(rng.integers(1, 3))):
        last = int(rng.choice([h for h in (last-1, last, last+1) if 1 <= h <= 3]))
        heights.append(last)
    start = heights[-1]
    heights.extend(start + (t+phase)//run for t in range(1, 2*run+1))
    length = len(heights)+1
    mask = {(r, -length) for r in range(-cap, cap+1)}
    for x, h in zip(range(-length+1, 0), heights):
        mask.update({(-h, x), (h, x)})
    inferred = _fit_stair(heights)
    assert inferred[:3] == (run, phase, heights[-1])
    return tuple(sorted(mask)), inferred[:3]


def _grow_route(rng, shape, mask, target, style):
    length = -min(c for r, c in mask)
    span = max(abs(r) for r, c in mask)
    straight_weight = float(rng.uniform(0.5, 2.0))
    for _ in range(90):
        tip = (int(rng.integers(span, shape[0]-span)),
               int(rng.integers(length, shape[1]-2)))
        points, ds = [tip], []
        boxes = [_box(mask, tip, 1)]
        for step in range(target):
            options = {}
            for d, (dr, dc) in enumerate(DIRS):
                if step == 0 and d != 1:
                    continue
                outgoing = _box(mask, points[-1], d)
                if not _fits(outgoing, shape):
                    continue
                for distance in range(2, max(shape)):
                    q = (points[-1][0]+dr*distance, points[-1][1]+dc*distance)
                    incoming = _box(mask, q, d)
                    if not _fits(incoming, shape):
                        continue
                    if any(_near_box(q, box, 1) for box in boxes+[outgoing]):
                        continue
                    valid = True
                    for old, nxt in zip(points[:-1], points[1:]):
                        if q[0] == old[0] or q[1] == old[1]:
                            if _direction(old, q) != _direction(old, nxt) or _distance(old, q) <= _distance(old, nxt):
                                valid = False
                                break
                    if valid:
                        options.setdefault(d, []).append((q, outgoing, incoming))
            if not options:
                break
            choices = sorted(options)
            weights = np.array([straight_weight if ds and ds[-1] == d else 1.0 for d in choices])
            d = int(rng.choice(choices, p=weights/weights.sum()))
            possibilities = options[d]
            weights = np.array([1/_distance(points[-1], v[0]) if target >= 5 else 1.0 for v in possibilities])
            q, outgoing, incoming = possibilities[int(rng.choice(len(possibilities), p=weights/weights.sum()))]
            boxes.extend([outgoing, incoming])
            points.append(q)
            ds.append(d)
        if len(ds) != target or len(set(ds)) < 2:
            continue
        if style == 'clean_teach' and not any(a == b for a, b in zip(ds, ds[1:])):
            continue
        if style == 'shadow_teach' and max(_distance(p, q) for p, q in zip(points, points[1:])) < length+3:
            continue
        _check_route(points, ds, mask, shape)
        return points, ds
    return None


def _fallback_route(rng, mask, target, style):
    """A finite randomized bend with subdivided legs, used after failed walks."""
    shape = (30, 30)
    length = -min(c for r, c in mask)
    span = max(abs(r) for r, c in mask)
    r, c = max(length, span+1), length+1
    first = target-1 if style == 'shadow_teach' else int(rng.integers(2, target))
    # Enough axial distance before the bend for a blocker outside every housing.
    horizontal = [2]*(first-1) + [max(2, 22-c-2*(first-1))]
    vertical = [span+2] + [2]*(target-first-1)
    points = [(r, c)]
    for distance in horizontal:
        c += distance
        points.append((r, c))
    for distance in vertical:
        r += distance
        points.append((r, c))
    ds = [1]*first + [2]*(target-first)
    _check_route(points, ds, mask, shape)
    return shape, points, ds


def _union(masks):
    return np.logical_or.reduce(masks)


def _exposure_stats(footprints, extras, final_box):
    exposures, shadows = _mask_exposures(footprints, extras)
    ideal = _union([f[0] for f in footprints])
    light, shadow = _union(exposures), _union(shadows)
    visible = np.ones(extras.shape, dtype=bool)
    r0, r1, c0, c1 = final_box
    visible[r0:r1+1, c0:c1+1] = False
    visible &= ~extras
    return dict(exposures=exposures, light=light,
                lost=int(np.count_nonzero(ideal & ~light & visible)),
                rescued=int(np.count_nonzero(shadow & light & visible)),
                early=int(np.count_nonzero(light & ~exposures[-1] & visible)))


def _opacity_effects(info, footprints, extras):
    """Count visible answer changes if each opaque color becomes transparent."""
    light = _union(_mask_exposures(footprints, extras)[0])
    visible = ~extras.copy()
    r0, r1, c0, c1 = _box(info['mask'], info['points'][-1], info['directions'][-1])
    visible[r0:r1+1, c0:c1+1] = False
    effective = extras & _union([f[0] for f in footprints])
    effects = []
    for color in np.unique(info['a'][effective]):
        remaining = extras & (info['a'] != color)
        transparent = _union(_mask_exposures(footprints, remaining)[0])
        effects.append(int(np.count_nonzero(transparent & ~light & visible)))
    return tuple(sorted(effects))


def _place_objects(rng, info, style, colors):
    a, points, ds = info['a'], info['points'], info['directions']
    footprints = _footprints(a.shape, points, ds, info['run'], info['phase'], info['half'])
    ideal = _union([f[0] for f in footprints])
    protected = _protected(a.shape, points, ds, info['boxes'])
    free = ~protected & (a == info['bg'])
    extras = np.zeros(a.shape, dtype=bool)
    if style == 'clean_teach' or style == 'clean':
        return extras, footprints
    final_box = _box(info['mask'], points[-1], ds[-1])
    candidates = np.argwhere(free & ideal)
    if not len(candidates):
        return None
    if style == 'shadow_teach':
        chosen = False
        for i in rng.permutation(len(candidates))[:220]:
            r, c = candidates[int(i)]
            extras[r, c] = True
            stats = _exposure_stats(footprints, extras, final_box)
            if stats['lost'] >= 4 and stats['rescued'] >= 2:
                a[r, c] = colors[0]
                chosen = True
                break
            extras[r, c] = False
        if not chosen:
            return None
        # A second color with an effective cell establishes color-independent opacity.
        chosen = False
        for i in rng.permutation(len(candidates))[:160]:
            r, c = candidates[int(i)]
            if extras[r, c]:
                continue
            extras[r, c] = True
            a[r, c] = colors[1]
            stats = _exposure_stats(footprints, extras, final_box)
            effects = _opacity_effects(info, footprints, extras)
            if stats['lost'] >= 4 and stats['rescued'] >= 2 and sum(v >= 2 for v in effects) >= 2:
                chosen = True
                break
            extras[r, c] = False
            a[r, c] = info['bg']
        if not chosen:
            return None
    else:
        groups = int(rng.integers(1, 3) if style == 'light' else rng.integers(3, 6))
        shapes = (((0, 0),), ((0, 0), (1, 0)), ((0, 0), (0, 1)),
                  ((0, 0), (1, 0), (1, 1)), ((0, 0), (0, 1), (1, 0), (1, 1)))
        for index in range(groups):
            offsets = shapes[int(rng.integers(len(shapes)))]
            for i in rng.permutation(len(candidates))[:100]:
                r, c = map(int, candidates[int(i)])
                cells = [(r+dr, c+dc) for dr, dc in offsets]
                if all(0 <= y < a.shape[0] and 0 <= x < a.shape[1]
                       and free[y, x] and not extras[y, x] for y, x in cells):
                    for y, x in cells:
                        extras[y, x] = True
                        a[y, x] = colors[index % len(colors)]
                    break
        if _exposure_stats(footprints, extras, final_box)['lost'] == 0:
            return None
    # Optional inert groups use the same ordinary colors, with no new role legend.
    outside = np.argwhere(free & ~ideal & ~extras)
    for i in rng.permutation(len(outside))[:int(rng.integers(0, 4))]:
        r, c = outside[int(i)]
        extras[r, c] = True
        a[r, c] = colors[int(rng.integers(len(colors)))]
    return extras, footprints


def _make_pair(rng, palette, target, style, excluded_mask=None):
    for attempt in range(65):
        mask, (run, phase, half) = _sample_housing(rng)
        if mask == excluded_mask:
            continue
        shape = (int(rng.integers(22, 31)), int(rng.integers(22, 31)))
        route = _grow_route(rng, shape, mask, target, style) if attempt < 45 else None
        if route is None:
            shape, points, ds = _fallback_route(rng, mask, target, style)
        else:
            points, ds = route
        boxes = _check_route(points, ds, mask, shape)
        bg, wall, ink = map(int, palette[:3])
        a = np.full(shape, bg, dtype=np.int8)
        for p in _pose(mask, points[0], 1):
            a[p] = wall
        for p in points:
            a[p] = ink
        info = dict(a=a, bg=bg, wall=wall, ink=ink, mask=mask, run=run,
                    phase=phase, half=half, points=points, directions=ds, boxes=boxes)
        placed = _place_objects(rng, info, style, list(map(int, palette[3:6])))
        if placed is None:
            continue
        extras, footprints = placed
        info['extras'] = extras
        exposures, _ = _mask_exposures(footprints, extras)
        out = _finish(info, _union(exposures))
        rotation = int(rng.integers(4))
        pair = {'input': np.rot90(a, rotation).copy(), 'output': np.rot90(out, rotation).copy()}
        decoded = _read_input(pair['input'])
        assert decoded['mask'] == mask
        if style == 'clean_teach' and not _reset_witness(decoded):
            continue
        stats = _measure(decoded)
        if stats['early'] == 0:
            continue
        assert np.array_equal(solve(pair['input']), pair['output'])
        return pair, mask
    raise RuntimeError("bounded lantern construction exhausted")


def _reset_witness(info):
    points, ds = info['points'], info['directions']
    merged_points, merged_ds = [points[0]], []
    for i, d in enumerate(ds):
        if merged_ds and d == merged_ds[-1]:
            merged_points[-1] = points[i+1]
        else:
            merged_ds.append(d)
            merged_points.append(points[i+1])
    original = _footprints(info['a'].shape, points, ds, info['run'], info['phase'], info['half'])
    merged = _footprints(info['a'].shape, merged_points, merged_ds, info['run'], info['phase'], info['half'])
    light = _union(_mask_exposures(original, info['extras'])[0])
    shortcut = _union(_mask_exposures(merged, info['extras'])[0])
    return bool(np.any(_finish(info, light) != _finish(info, shortcut)))


def _measure(info):
    footprints = _footprints(info['a'].shape, info['points'], info['directions'],
                             info['run'], info['phase'], info['half'])
    final_box = _box(info['mask'], info['points'][-1], info['directions'][-1])
    stats = _exposure_stats(footprints, info['extras'], final_box)
    centerlines = np.zeros(info['a'].shape, dtype=bool)
    for p, q, d in zip(info['points'], info['points'][1:], info['directions']):
        dr, dc = DIRS[d]
        for t in range(_distance(p, q)+1):
            centerlines[p[0]+dr*t, p[1]+dc*t] = True
    off_axis = stats['light'] & ~centerlines & ~info['extras']
    r0, r1, c0, c1 = final_box
    off_axis[r0:r1+1, c0:c1+1] = False
    stats.update(mask=info['mask'], run=info['run'], phase=info['phase'],
                 profile=info['profile'], count=len(info['directions']),
                 distances=tuple(_distance(a, b) for a, b in zip(info['points'], info['points'][1:])),
                 initial=info['initial'], final=(info['directions'][-1]+info['initial']-1)%4,
                 palette=(info['bg'], info['wall'], info['ink']), shape=info['a'].shape,
                 effective_colors=len(np.unique(info['a'][info['extras'] & _union([f[0] for f in footprints])])),
                 route=tuple((r-info['points'][0][0], c-info['points'][0][1]) for r, c in info['points']),
                 reset=_reset_witness(info), opacity_effects=_opacity_effects(info, footprints, info['extras']),
                 off_axis=int(np.count_nonzero(off_axis)))
    return stats


def _key(a):
    a = np.asarray(a)
    return a.shape, a.tobytes()


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shared_palette = bool(rng.random() < 0.2)
    palette = rng.permutation(10)
    puzzle, seen, first_mask = {'train': [], 'test': []}, set(), None
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            if split == 'train' and i == 0:
                target, style = 3, 'clean_teach'
            elif split == 'train' and i == 1:
                target, style = 4, 'shadow_teach'
            else:
                target = int(rng.integers(2, 5) if split == 'train' else rng.integers(5, 8))
                style = str(rng.choice(['clean', 'light', 'dense'], p=[0.25, 0.45, 0.30]))
            for _ in range(12):
                colors = palette if shared_palette else rng.permutation(10)
                pair, mask = _make_pair(rng, colors, target, style,
                                        first_mask if split == 'train' and i == 1 else None)
                key = _key(pair['input'])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not make distinct inputs")
            seen.add(key)
            if split == 'train' and i == 0:
                first_mask = mask
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _read_input(inp)
            expected = solve(inp)
            assert np.array_equal(expected, out), "incorrect exposure union or final lantern"
            canonical_out = np.rot90(out, info['initial']-1)
            assert np.array_equal(canonical_out[info['extras']], info['a'][info['extras']])
            final_cells = _pose(info['mask'], info['points'][-1], info['directions'][-1])
            assert set(map(tuple, np.argwhere(canonical_out == info['wall']))) == final_cells
            assert len(final_cells) == len(info['mask'])
            assert not np.array_equal(inp, out)
            # The differently implemented lane scan must agree with mask construction.
            footprints = _footprints(info['a'].shape, info['points'], info['directions'],
                                     info['run'], info['phase'], info['half'])
            masks = _mask_exposures(footprints, info['extras'])[0]
            assert all(np.array_equal(a, b) for a, b in zip(masks, _scan_exposures(info)))
            record = _measure(info)
            assert record['early'] > 0, "earlier hops must visibly contribute"
            assert record['off_axis'] > 0, "cone geometry must differ from centerline painting"
            records[split].append(record)
    train, test = records['train'], records['test']
    assert any(m['effective_colors'] == 0 and m['reset'] for m in train)
    assert any(m['lost'] >= 4 and m['rescued'] >= 2
               and sum(v >= 2 for v in m['opacity_effects']) >= 2 for m in train)
    assert len({m['mask'] for m in train}) >= 2
    assert len({m['count'] for m in train}) >= 2
    assert max(m['count'] for m in train) <= 4
    assert min(m['count'] for m in test) >= 5
    return None


def self_test():
    """Bounded 42-seed semantic/diversity audit plus hand-written regressions."""
    records, puzzles, shapes, normalized, routes = [], [], set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                info = _read_input(pair['input'])
                m = _measure(info)
                records.append((split, m))
                shapes.add(m['shape'])
                routes.add(m['route'])
                # Color-normalized grid structures, not just grid hashes.
                roles, values = {}, []
                for color in pair['input'].flat:
                    values.append(roles.setdefault(int(color), len(roles)))
                normalized.add((pair['input'].shape, tuple(values)))
    assert len(set(puzzles)) == 42
    assert len(shapes) >= 25 and len(normalized) >= 150 and len(routes) >= 100
    assert len({m['mask'] for s, m in records}) >= 15
    assert {m['run'] for s, m in records} == {1, 2}
    assert {m['count'] for s, m in records} == set(range(2, 8))
    for split in ('train', 'test'):
        ms = [m for s, m in records if s == split]
        assert {m['initial'] for m in ms} == set(range(4))
        assert {m['final'] for m in ms} == set(range(4))
        assert any(m['lost'] > 0 for m in ms)
        assert any(m['rescued'] > 0 for m in ms)
        assert any(m['effective_colors'] == 0 for m in ms)
    assert any(any(b < a for a, b in zip(m['profile'], m['profile'][1:])) for s, m in records)
    assert len({m['palette'] for s, m in records}) >= 80
    assert len({m['distances'] for s, m in records}) >= 100
    # Exact thin-stair continuation, including both phases of period two.
    assert _fit_stair([1, 2, 3])[:3] == (1, 0, 3)
    assert _fit_stair([1, 1, 2, 2, 3])[:3] == (2, 0, 3)
    assert _fit_stair([1, 2, 2, 3, 3])[:3] == (2, 1, 3)
    # Two perpendicular finite beams: one lane is shadowed, then illuminated anew.
    points, ds = [(5, 2), (5, 12), (12, 12)], [1, 2]
    extras = np.zeros((18, 18), dtype=bool)
    extras[7, 6] = True
    fps = _footprints(extras.shape, points, ds, 1, 0, 2)
    exposures, shadows = _mask_exposures(fps, extras)
    assert shadows[0][7, 12] and exposures[1][7, 12]
    assert not _union(exposures)[7, 7] and _union(exposures)[7, 12]
    # Destination plane and virtual boundary are excluded beyond their exact limits.
    fp = _footprints((15, 15), [(7, 2), (7, 6)], [1], 2, 0, 2)[0][0]
    assert fp[7, 6] and not fp[7, 7]
    assert not fp[4, 4] and fp[5, 4]
    # Eight beacons are allowed by the specification even though the sampler
    # uses at most seven. The input-only solver must still handle such a member.
    pair, _ = _make_pair(np.random.default_rng(123456), np.arange(10), 8, 'clean')
    assert len(_read_input(pair['input'])['directions']) == 8
    assert np.array_equal(solve(pair['input']), pair['output'])
    # Counts, determinism, NumPy integer arguments, and deliberate bad labels.
    for a, b in ((2, 1), (10, 1), (10, 4)):
        p = generate_puzzle(9000+a*10+b, a, b)
        assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.0), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    a, b = generate_puzzle(73, np.int64(2), np.int64(1)), generate_puzzle(73, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1)%10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted label accepted")
    try:
        _route((3, 3), {(3, 6), (6, 3)})
    except ValueError:
        pass
    else:
        raise AssertionError("competing occupied rays accepted")
    return {'seeds': 42, 'pairs': len(records), 'normalized_inputs': len(normalized),
            'shapes': len(shapes), 'housing_masks': len({m['mask'] for s, m in records}),
            'routes': len(routes), 'semantic_checks': 'generator-owned'}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
