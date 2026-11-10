"""Funnel-filtered blank-window union. Standard library and NumPy only.

The decoder receives only a grid. Construction masks are never used by solve.
No files, persistent caches, stored examples, or execution traces are created.
"""
from __future__ import annotations

import hashlib
import numpy as np


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(f"illegal {name}")
    return value


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, sides 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _connected(mask):
    cells = set(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    queue = [cells.pop()]
    while queue:
        y, x = queue.pop()
        for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if q in cells:
                cells.remove(q)
                queue.append(q)
    return not cells


def _fit_rail(values, sign):
    """Infer a thin terminal staircase, including its phase, from visible cells."""
    values = np.asarray(values, dtype=int)
    t = np.arange(len(values)) - len(values) + 1
    fits = []
    for period in (1, 2, 3):
        for phase in range(period):
            predicted = values[-1] + sign * ((t + phase) // period)
            length = 0
            for same in (values == predicted)[::-1]:
                if not same:
                    break
                length += 1
            if length >= 2 * period + 1:
                fits.append((length, period, phase))
    if not fits:
        raise ValueError("insufficient terminal staircase evidence")
    longest = max(v[0] for v in fits)
    best = [v for v in fits if v[0] == longest]
    futures = {tuple((np.arange(1, 31) + phase) // period)
               for _, period, phase in best}
    if len(futures) != 1:
        raise ValueError("ambiguous terminal continuation")
    _, period, phase = min(best, key=lambda v: v[1:])
    return period, phase


def _decode(input_grid):
    """Recover the unique closed-back emitter, marker and unoccluded footprint."""
    inp = _grid(input_grid)
    counts = np.bincount(inp.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if counts[bg] <= inp.size // 2:
        raise ValueError("background must occupy more than half the input")
    matches = []
    for rotation in range(4):
        a = np.rot90(inp, -rotation)
        h, w = a.shape
        for wall in map(int, np.flatnonzero(counts >= 9)):
            if wall == bg:
                continue
            positions = np.argwhere(a == wall)
            rear, mouth = int(positions[:, 1].min()), int(positions[:, 1].max())
            cap = np.flatnonzero(a[:, rear] == wall)
            if (len(cap) < 3 or np.any(np.diff(cap) != 1)
                    or mouth - rear < 3 or mouth >= w - 1):
                continue
            lows, highs = [], []
            for x in range(rear + 1, mouth + 1):
                rows = np.flatnonzero(a[:, x] == wall)
                if len(rows) != 2 or rows[1] - rows[0] < 2:
                    break
                lows.append(int(rows[0]))
                highs.append(int(rows[1]))
            if len(lows) != mouth - rear:
                continue
            if (cap[0] != lows[0] or cap[-1] != highs[0]
                    or np.any(np.abs(np.diff(lows)) > 1)
                    or np.any(np.abs(np.diff(highs)) > 1)):
                continue
            try:
                fits = (_fit_rail(lows, -1), _fit_rail(highs, 1))
            except ValueError:
                continue
            footprint = np.zeros(a.shape, dtype=bool)
            bounds = []
            for x in range(rear + 1, w):
                if x <= mouth:
                    lo, hi = lows[x - rear - 1], highs[x - rear - 1]
                else:
                    p, phase = fits[0]
                    lo = lows[-1] - (x - mouth + phase) // p
                    p, phase = fits[1]
                    hi = highs[-1] + (x - mouth + phase) // p
                bounds.append((lo, hi))
                footprint[max(0, lo + 1):min(h, hi), x] = True
            chamber = footprint.copy()
            chamber[:, mouth + 1:] = False
            markers = np.argwhere(chamber & (a != bg) & (a != wall))
            if not len(markers):
                continue
            first_x = int(markers[:, 1].min())
            first_colors = np.unique(a[markers[markers[:, 1] == first_x, 0], first_x])
            if len(first_colors) != 1:
                continue
            payload = int(first_colors[0])
            patch = a == payload
            if np.any(patch & ~chamber) or not _connected(patch):
                continue
            extras = (a != bg) & (a != wall) & (a != payload)
            internal = np.argwhere(extras & chamber)
            if len(internal) and internal[:, 1].min() <= np.argwhere(patch)[:, 1].max():
                continue
            matches.append(dict(grid=a, rotation=rotation, bg=bg, wall=wall,
                                payload=payload, rear=rear, mouth=mouth, fits=fits,
                                bounds=bounds, footprint=footprint, chamber=chamber,
                                extras=extras))
    if len(matches) != 1:
        raise ValueError(f"expected one readable emitter, found {len(matches)}")
    return matches[0]


def _window_union(mask, size=3):
    """All windows are tested simultaneously on the original mask."""
    h, w = mask.shape
    result = np.zeros(mask.shape, dtype=bool)
    if h < size or w < size:
        return result
    counts = np.zeros((h - size + 1, w - size + 1), dtype=np.int16)
    for dy in range(size):
        for dx in range(size):
            counts += mask[dy:dy + h - size + 1, dx:dx + w - size + 1]
    starts = counts == size * size
    for dy in range(size):
        for dx in range(size):
            result[dy:dy + starts.shape[0], dx:dx + starts.shape[1]] |= starts
    return result


def _light(a):
    # Input-only lane scan: outside objects never start a blocked lane.
    light = a['footprint'].copy()
    g = a['grid']
    for y in range(g.shape[0]):
        blocked = False
        for x in range(a['rear'] + 1, g.shape[1]):
            if not a['footprint'][y, x]:
                continue
            if a['extras'][y, x]:
                blocked = True
            if blocked:
                light[y, x] = False
    return light


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _decode(input_grid)
    out = a['grid'].copy()
    eligible = _light(a) & (a['grid'] == a['bg'])
    out[_window_union(eligible)] = a['payload']
    return np.rot90(out, a['rotation']).copy()


def _geometry(rng, projection_range):
    periods = [int(rng.integers(1, 4))] * 2
    if rng.random() < .35:
        periods[1] = int(rng.integers(1, 4))
    phases = [int(rng.integers(p)) for p in periods]
    n = int(rng.integers(3, 7))
    flare = 2 * max(periods) + 1 + int(rng.integers(2))
    profiles = []
    for _ in range(2):
        start = int(rng.integers(2, 5))
        neck = int(rng.integers(1, 3))
        if rng.random() < .45:
            peak = min(5, start + 1)
            ys = np.rint(np.interp(np.arange(n), [0, n // 2, n - 1],
                                  [start, peak, neck])).astype(int)
        else:
            ys = np.rint(np.linspace(start, neck, n)).astype(int)
        if np.any(np.abs(np.diff(ys)) > 1):
            return None
        profiles.append(ys)
    extensions = [np.array(list(profile) + [int(profile[-1]) + (t + phase) // period
                                          for t in range(1, flare + 1)])
                  for profile, period, phase in zip(profiles, periods, phases)]
    axial = n + flare
    min_proj, max_proj = projection_range
    rear_max = min(5, 29 - axial - min_proj)
    if rear_max < 0:
        return None
    rear = int(rng.integers(rear_max + 1))
    mouth = rear + axial
    projection = int(rng.integers(min_proj, min(max_proj, 29 - mouth) + 1))
    w = mouth + projection + 1
    upper, lower = int(extensions[0].max()), int(extensions[1].max())
    hmin = max(14, upper + lower + 3)
    if hmin > 30:
        return None
    h = int(rng.integers(hmin, 31))
    center = int(rng.integers(upper + 1, h - lower - 1))
    walls = np.zeros((h, w), dtype=bool)
    footprint = np.zeros((h, w), dtype=bool)
    walls[center - profiles[0][0]:center + profiles[1][0] + 1, rear] = True
    for j, (u, d) in enumerate(zip(*extensions), 1):
        x = rear + j
        walls[center - u, x] = walls[center + d, x] = True
        footprint[center - u + 1:center + d, x] = True
    for x in range(mouth + 1, w):
        t = x - rear - n
        lo = center - profiles[0][-1] - (t + phases[0]) // periods[0]
        hi = center + profiles[1][-1] + (t + phases[1]) // periods[1]
        footprint[max(0, lo + 1):min(h, hi), x] = True
    return walls, footprint, rear, mouth, rear + n


def _grow_marker(rng, mask):
    options = sorted(map(tuple, np.argwhere(mask)))
    points = {options[int(rng.integers(len(options)))]}
    for _ in range(int(rng.integers(4))):
        frontier = sorted({q for y, x in points for q in
                           ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                           if 0 <= q[0] < mask.shape[0] and 0 <= q[1] < mask.shape[1]
                           and mask[q] and q not in points})
        if not frontier:
            break
        points.add(frontier[int(rng.integers(len(frontier)))])
    return sorted(points)


_SHAPES = (((0, 0),), ((0, 0), (1, 0)),
           ((0, 0), (0, 1), (1, 0)), ((0, 0), (0, 1), (1, 0), (1, 1)))


def _place_obstacle(rng, grid, footprint, bg, color, minimum_x, offsets, used_lanes=None):
    h, w = grid.shape
    options = []
    for y, x in np.argwhere(footprint & (grid == bg)):
        y, x = int(y), int(x)
        if x < minimum_x or x > w - 4:
            continue
        points = [(y + dy, x + dx) for dy, dx in offsets]
        if any(yy >= h or xx >= w or grid[yy, xx] != bg for yy, xx in points):
            continue
        if used_lanes is not None and any(yy in used_lanes for yy, _ in points):
            continue
        options.append(points)
    if not options:
        return False
    for y, x in options[int(rng.integers(len(options)))]:
        grid[y, x] = color
        if used_lanes is not None:
            used_lanes.add(y)
    return True


def _construction_output(grid, footprint, bg, wall, payload):
    # Construction uses an explicit union of rays, distinct from solve's scan.
    shadow = np.zeros(grid.shape, dtype=bool)
    obstacles = footprint & (grid != bg) & (grid != wall) & (grid != payload)
    for y, x in np.argwhere(obstacles):
        shadow[y, x:] = True
    result = grid.copy()
    result[_window_union(footprint & ~shadow & (grid == bg))] = payload
    return result


def _features(a):
    g, bg, footprint = a['grid'], a['bg'], a['footprint']
    light = _light(a)
    blank = g == bg
    valid = blank & light
    chosen = _window_union(valid)
    without_shadows = _window_union(blank & footprint)
    global_windows = _window_union(blank)
    post_filter = global_windows & valid
    # A different implementation of the nine-cell predicate for validation.
    reference = np.zeros(g.shape, dtype=bool)
    starts = []
    for y in range(g.shape[0] - 2):
        for x in range(g.shape[1] - 2):
            if np.all(valid[y:y + 3, x:x + 3]):
                reference[y:y + 3, x:x + 3] = True
                starts.append((y, x))
    assert np.array_equal(chosen, reference)
    expected = g.copy()
    expected[reference] = a['payload']
    return dict(mask=chosen, expected=expected, starts=starts,
                no_a=int(np.count_nonzero(valid & ~chosen)),
                no_b=int(np.count_nonzero(global_windows & ~chosen)),
                no_shadow=int(np.count_nonzero(without_shadows & ~chosen)),
                post_filter=int(np.count_nonzero(post_filter & ~chosen)),
                size2=int(np.count_nonzero(_window_union(valid, 2) ^ chosen)),
                size4=int(np.count_nonzero(_window_union(valid, 4) ^ chosen)),
                effective=int(np.count_nonzero(a['extras'] & footprint)),
                repeated_lane=bool(np.any((a['extras'] & footprint).sum(axis=1) > 1)),
                blocker_colors=len(np.unique(g[a['extras'] & footprint])),
                shadow_cells=int(np.count_nonzero(footprint & ~light & blank)))


def _build_pair(rng, rotation, style, projection_range, palette):
    bg, wall, payload = map(int, palette[:3])
    geometry = _geometry(rng, projection_range)
    if geometry is None:
        return None
    walls, footprint, rear, mouth, neck = geometry
    grid = np.full(walls.shape, bg, dtype=np.int8)
    grid[walls] = wall
    candidates = footprint.copy()
    candidates[:, neck:] = False
    patch = _grow_marker(rng, candidates)
    for point in patch:
        grid[point] = payload
    minimum_x = max(x for _, x in patch) + 1
    if style != 'clean':
        lanes = set() if style in ('teach', 'light') else None
        count = 2 if style == 'teach' else int(rng.integers(2, 6))
        for j in range(count):
            shape_index = (0 if style == 'teach' else int(rng.integers(2, 4))
                           if style == 'groups' else int(rng.integers(2)))
            if not _place_obstacle(rng, grid, footprint, bg, int(palette[3 + j % 3]),
                                   minimum_x, _SHAPES[shape_index], lanes):
                return None
        if style == 'groups':
            # Repeated blockers in an already occupied lane do not widen its shadow.
            blocked = np.argwhere(footprint & (grid != bg) & (grid != wall) & (grid != payload))
            options = [(int(y), xx) for y, x in blocked
                       for xx in range(int(x) + 1, grid.shape[1] - 1)
                       if footprint[y, xx] and grid[y, xx] == bg]
            if options:
                grid[options[int(rng.integers(len(options)))]] = int(palette[4])
            else:
                return None
    # These can share the colors of effective blockers; they stay inert.
    outside = np.argwhere(~footprint & ~walls & (grid == bg))
    for j in rng.choice(len(outside), size=min(len(outside), int(rng.integers(1, 5))), replace=False):
        grid[tuple(outside[int(j)])] = int(palette[3 + int(rng.integers(3))])
    output = _construction_output(grid, footprint, bg, wall, payload)
    inp = np.rot90(grid, rotation).copy()
    out = np.rot90(output, rotation).copy()
    try:
        a = _decode(inp)
        f = _features(a)
    except (ValueError, AssertionError):
        return None
    if not np.array_equal(np.rot90(f['expected'], a['rotation']), out):
        return None
    # Every generated pair separates the inherited mechanisms from ablations.
    if (f['mask'].sum() <= 9 or not f['no_a'] or not f['no_b']
            or not f['post_filter'] or not f['size2'] or not f['size4']):
        return None
    if style != 'clean' and not f['no_shadow']:
        return None
    if style == 'teach' and (f['blocker_colors'] < 2 or f['effective'] != 2):
        return None
    if style == 'groups' and not f['repeated_lane']:
        return None
    return {'input': inp, 'output': out}


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    regime = str(rng.choice(('turn', 'range', 'occlusion')))
    directions = list(map(int, rng.permutation(4)))
    shared_palette = rng.random() < .25
    palette = rng.permutation(10)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            style = ('clean' if split == 'train' and index == 0 else
                     'teach' if split == 'train' and index == 1 else
                     'groups' if split == 'test' and regime == 'occlusion' else
                     str(rng.choice(('clean', 'light', 'groups'), p=(.2, .45, .35))))
            if regime == 'occlusion' and split == 'train' and index >= 2:
                style = 'light'
            rotation = (directions[index % 2] if split == 'train' else directions[2 + index % 2]
                        ) if regime == 'turn' else int(rng.integers(4))
            projection = ((5, 8) if split == 'train' else (10, 16)) if regime == 'range' else (5, 16)
            colors = palette if shared_palette else rng.permutation(10)
            for _ in range(120):
                pair = _build_pair(rng, rotation, style, projection, colors)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded pair construction exhausted")
    rng.shuffle(puzzle['train'])
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, training, testing = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _key(inp)
            assert key not in seen
            seen.add(key)
            a = _decode(inp)
            f = _features(a)
            f['rotation'] = a['rotation']
            f['projection'] = a['grid'].shape[1] - a['mouth'] - 1
            assert np.array_equal(out, np.rot90(f['expected'], a['rotation']))
            changed = np.rot90(inp != out, -a['rotation'])
            assert np.array_equal(changed, f['mask'])
            assert f['mask'].sum() > 9 and len(f['starts']) >= 2
            assert f['no_a'] and f['no_b'] and f['post_filter']
            assert f['size2'] and f['size4']
            assert np.array_equal(inp[inp != a['bg']], out[inp != a['bg']])
            assert not np.any(f['mask'] & ~_light(a))
            if f['effective']:
                assert f['no_shadow'] > 0
            if split == 'train':
                training.append(f)
            else:
                testing.append(f)
    assert any(f['effective'] == 0 for f in training)
    assert any(f['blocker_colors'] >= 2 and f['no_shadow'] > 0 for f in training)
    # Verify an actual observable shift, without obtaining the sampled regime.
    turns = {f['rotation'] for f in testing}.isdisjoint(f['rotation'] for f in training)
    longer = min(f['projection'] for f in testing) > max(f['projection'] for f in training)
    repeated = (all(f['repeated_lane'] for f in testing)
                and not any(f['repeated_lane'] for f in training))
    assert turns or longer or repeated, 'no observable train/test shift'


def _regressions():
    # Explicit mathematical tests of exhaustive union, whole-window eligibility,
    # overlapping windows, boundary clipping, and one/two-cell-wide light slivers.
    patch = np.zeros((7, 11), dtype=bool)
    patch[1:4, 0:4] = True
    patch[3:6, 8:11] = True
    expected = patch.copy()
    assert np.array_equal(_window_union(patch), expected)
    patch[0, 5:7] = True
    assert np.array_equal(_window_union(patch), expected)
    lit = np.ones((3, 4), dtype=bool)
    lit[0, 0] = False
    expected = np.zeros_like(lit)
    expected[:, 1:] = True
    assert np.array_equal(_window_union(lit), expected)
    assert not _window_union(np.ones((2, 8), dtype=bool)).any()
    # Four thin-wall corner phases, supplied directly as visible boundary rows.
    for rail, expected_next in (((5, 4, 3), 2), ((6, 6, 5, 5, 4), 4),
                                ((6, 5, 5, 4, 4), 3), ((7, 7, 7, 6, 6, 6, 5), 5)):
        p, phase = _fit_rail(rail, -1)
        assert rail[-1] - (1 + phase) // p == expected_next
    # A manually specified emitter, not sampled by the geometry constructor.
    g = np.zeros((17, 20), dtype=np.int8)
    g[6:11, 1] = 1
    for x, y in enumerate((6, 6, 5, 5, 4, 4, 3), 2):
        g[y, x] = g[16 - y, x] = 1
    g[8, 2] = 2
    g[8, 10] = 3
    g[8, 13] = 4  # A second obstacle in the same lane.
    g[0, 9] = 4  # Outside now; this lane enters the cone later.
    out = solve(g)
    assert np.all(out[8, 14:] == 0), 'straight downstream shadow'
    assert out[7, 18] == out[9, 18] == 2, 'shadow must not widen'
    assert out[0, 18] == 2, 'outside obstacle must not cast a shadow'
    assert out[2, 10] == 0, 'virtual boundary stays blank'
    assert out[7, 2] == 0, 'lit rear fringe cannot hold a full blank window'
    assert out[4, 12] == 2, 'complete nearby window must highlight'
    assert np.array_equal(g[g != 0], out[g != 0])
    erased = g.copy()
    erased[8, 13] = erased[0, 9] = 0
    expected = out.copy()
    expected[8, 13] = expected[0, 9] = 0
    assert np.array_equal(solve(erased), expected), 'redundant and exterior obstacles'


def self_test():
    """Bounded 42-seed sweep; aggregate fingerprints only, no saved payloads."""
    _regressions()
    shapes, palettes, outlines, hashes = set(), set(), set(), set()
    rotations, periods, projections, counts = set(), set(), set(), set()
    overlap = clipped = grouped = asymmetric = nonmonotone = 0
    no_a = no_b = no_shadow = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                assert np.array_equal(solve(inp), out)
                assert np.array_equal(solve(inp.tolist()), out)
                digest.update(str(inp.shape).encode())
                digest.update(inp.tobytes())
                digest.update(out.tobytes())
                a = _decode(inp)
                f = _features(a)
                shapes.add(inp.shape)
                palettes.add((a['bg'], a['wall'], a['payload']))
                rotations.add(a['rotation'])
                periods.update(p for p, _ in a['fits'])
                projections.add(a['grid'].shape[1] - a['mouth'] - 1)
                counts.add(len(f['starts']))
                wall = a['grid'] == a['wall']
                ys, xs = np.nonzero(wall)
                crop = wall[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
                outlines.add(_key(crop))
                overlap += int(f['mask'].sum() < 9 * len(f['starts']))
                clipped += int(np.any(a['footprint'][0]) or np.any(a['footprint'][-1]))
                effective = a['extras'] & a['footprint']
                grouped += int(np.any(effective[:-1] & effective[1:]) or
                               np.any(effective[:, :-1] & effective[:, 1:]))
                bounds = np.array(a['bounds'][:a['mouth'] - a['rear']])
                asymmetric += int(np.ptp(bounds.sum(axis=1)) > 0)
                nonmonotone += int(np.any(np.diff(bounds[:, 1] - bounds[:, 0]) < 0))
                no_a += f['no_a']
                no_b += f['no_b']
                no_shadow += f['no_shadow']
        hashes.add(digest.hexdigest())
        if seed in (0, 13, 41):
            again = generate_puzzle(seed)
            assert all(np.array_equal(x[k], y[k]) for s in ('train', 'test')
                       for x, y in zip(puzzle[s], again[s]) for k in ('input', 'output'))
        if seed == 0:
            inp, out = puzzle['test'][0]['input'], puzzle['test'][0]['output']
            before = inp.copy()
            for r in range(4):
                assert np.array_equal(solve(np.rot90(inp, r)), np.rot90(out, r))
            permutation = np.array([7, 3, 9, 0, 5, 1, 8, 2, 6, 4], dtype=np.int8)
            assert np.array_equal(solve(permutation[inp]), permutation[out])
            assert np.array_equal(inp, before), 'solve mutated its input'
    assert len(hashes) == 42 and len(shapes) >= 50 and len(outlines) >= 70
    assert len(palettes) >= 60 and len(counts) >= 25
    assert rotations == {0, 1, 2, 3} and periods == {1, 2, 3}
    assert min(projections) == 5 and max(projections) >= 14
    assert overlap > 150 and clipped > 30 and grouped > 20
    assert asymmetric > 50 and nonmonotone > 50
    assert no_a and no_b and no_shadow
    for nt, nq in ((2, 1), (10, 1), (10, 4), (5, 4)):
        p = generate_puzzle(817 + nt, nt, nq)
        assert (len(p['train']), len(p['test'])) == (nt, nq)
        validate_generated_puzzle(p)
    for nt, nq in ((1, 1), (11, 1), (2, 2), (4, 0), (6, 5),
                   (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    for seed in (-1, 1.5, True):
        try:
            generate_puzzle(seed)
        except ValueError:
            continue
        raise AssertionError("illegal seed accepted")
    corrupt = generate_puzzle(99, 2, 1)
    p = corrupt['test'][0]
    point = tuple(np.argwhere(p['input'] != p['output'])[0])
    p['output'][point] = p['input'][point]
    try:
        validate_generated_puzzle(corrupt)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupted output accepted")
    return {'seeds': 42, 'pairs': 210, 'shapes': len(shapes), 'outlines': len(outlines),
            'palettes': len(palettes), 'overlapping_window_pairs': overlap,
            'clipped_cones': clipped, 'grouped_blocker_pairs': grouped,
            'asymmetric_emitters': asymmetric, 'nonmonotone_emitters': nonmonotone,
            'ablation_changed_cells': {'omit_A': no_a, 'omit_B': no_b,
                                       'omit_shadows': no_shadow}}
