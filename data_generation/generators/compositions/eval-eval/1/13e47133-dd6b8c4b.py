"""Invisible contour tracks feeding a capacity-nine sampler.

Only Python's standard library and NumPy are used. No parent module is imported.
The solver reads roles, keys, terrain, routes and capacity from the input alone.
"""
from collections import deque
import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'Grid dimensions must be 1..30.')
    _require(np.issubdtype(a.dtype, np.integer), 'Grid cells must be integers.')
    _require(np.all((a >= 0) & (a <= 9)), 'Colors must be 0..9.')
    return a.astype(np.int8, copy=False)


def _signature(grid):
    return grid.shape, grid.astype(np.int8).tobytes()


def _puzzle_signature(puzzle):
    return tuple(tuple((_signature(np.asarray(p['input'])), _signature(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def _inside(mask, r, c):
    return 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1] and bool(mask[r, c])


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        cells = [(r, c)]
        seen[r, c] = True
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if _inside(mask, nr, nc) and not seen[nr, nc]:
                    seen[nr, nc] = True
                    cells.append((nr, nc))
        component = np.zeros(mask.shape, dtype=bool)
        for cell in cells:
            component[cell] = True
        result.append(component)
    return result


def _depths(mask, diagonal=True):
    """BFS distance inward from the erosion boundary; layer numbering starts at 0."""
    steps = DIR8 if diagonal else DIR4
    depth = np.full(mask.shape, -1, dtype=np.int16)
    queue = deque()
    for rr, cc in np.argwhere(mask):
        r, c = int(rr), int(cc)
        if any(not _inside(mask, r + dr, c + dc) for dr, dc in steps):
            depth[r, c] = 0
            queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for dr, dc in steps:
            nr, nc = r + dr, c + dc
            if _inside(mask, nr, nc) and depth[nr, nc] < 0:
                depth[nr, nc] = depth[r, c] + 1
                queue.append((nr, nc))
    return depth


def _erosion_depths(mask):
    """Separate array erosion implementation for returned-grid validation."""
    h, w = mask.shape
    current = mask.copy()
    result = np.full(mask.shape, -1, dtype=np.int16)
    level = 0
    while current.any():
        padded = np.pad(current, 1)
        interior = current.copy()
        for dr, dc in DIR8:
            interior &= padded[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
        result[current & ~interior] = level
        current = interior
        level += 1
    return result


def _paths(mask, depth):
    """Convex-corner diagonals with consecutive erosion depths."""
    result = []
    for rr, cc in np.argwhere(mask & (depth == 0)):
        r, c = int(rr), int(cc)
        for dr in (-1, 1):
            for dc in (-1, 1):
                if _inside(mask, r - dr, c) or _inside(mask, r, c - dc):
                    continue
                if not all(_inside(mask, r + x, c + y) for x, y in ((dr, 0), (0, dc), (dr, dc))):
                    continue
                length = 0
                while _inside(mask, r + length * dr, c + length * dc):
                    if depth[r + length * dr, c + length * dc] != length:
                        break
                    length += 1
                result.append((r, c, dr, dc, length))
    return result


def _collector_mask(shape, pos):
    mask = np.zeros(shape, dtype=bool)
    r, c = pos
    mask[r:r + 3, c:c + 3] = True
    return mask


def _roles(grid):
    wall = int(grid[0, 0])
    _require(all(np.all(edge == wall) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             'The outside frame must be uniform wall.')
    counts = np.bincount(grid.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(bg != wall and np.count_nonzero(counts == counts[bg]) == 1,
             'The background must be the unique most frequent input color.')
    candidates = []
    for r in range(grid.shape[0] - 2):
        for c in range(grid.shape[1] - 2):
            patch = grid[r:r + 3, c:c + 3]
            rim, payload = int(patch[0, 0]), int(patch[1, 1])
            if rim in (bg, wall) or payload in (bg, wall, rim):
                continue
            if np.all(patch.ravel()[[0, 1, 2, 3, 5, 6, 7, 8]] == rim):
                candidates.append(((r, c), rim, payload))
    _require(len(candidates) == 1, 'Exactly one two-color 3x3 sampler is required.')
    pos, target, payload = candidates[0]
    return bg, wall, pos, target, payload


def _decode_key(grid, mask, depth, bg, payload, collector):
    cues = [tuple(map(int, x)) for x in np.argwhere(mask & ~collector & (grid != bg) & (grid != payload))]
    if not cues:
        return (bg,), (), ()
    length = max(int(depth[p]) for p in cues) + 1
    _require(1 <= length <= 4, 'Keys have at most four positions.')
    candidates = []
    for r, c, dr, dc, capacity in _paths(mask, depth):
        positions = tuple((r + i * dr, c + i * dc) for i in range(length))
        if capacity < length or any(collector[p] for p in positions):
            continue
        if all(p == (r + int(depth[p]) * dr, c + int(depth[p]) * dc) for p in cues):
            period = tuple(int(grid[p]) for p in positions)
            if payload not in period:
                candidates.append((period, positions, (dr, dc)))
    _require(candidates and len({x[0] for x in candidates}) == 1,
             'The diagonal must determine one unambiguous period.')
    period = candidates[0][0]
    _require(period[-1] != bg, 'The final key entry must be visible.')
    # Multiple geometric readings are accepted only when the resulting program is identical.
    return period, candidates[0][1], tuple(sorted({x[2] for x in candidates}))


def _distance(traversable, collector, wave=False):
    distance = np.full(traversable.shape, -1, dtype=np.int16)
    distance[collector] = 0
    if wave:
        seen, frontier = collector.copy(), collector.copy()
        for level in range(1, traversable.size + 1):
            spread = np.zeros(traversable.shape, dtype=bool)
            spread[1:] |= frontier[:-1]
            spread[:-1] |= frontier[1:]
            spread[:, 1:] |= frontier[:, :-1]
            spread[:, :-1] |= frontier[:, 1:]
            frontier = spread & traversable & ~seen
            if not frontier.any():
                break
            distance[frontier] = level
            seen |= frontier
        return distance
    queue = deque(tuple(map(int, p)) for p in np.argwhere(collector))
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if _inside(traversable, nr, nc) and distance[nr, nc] < 0:
                distance[nr, nc] = distance[r, c] + 1
                queue.append((nr, nc))
    return distance


def _selection(grid, payload, collector, distance, strict=True):
    reachable = (grid == payload) & ~collector & (distance >= 0)
    values = np.sort(distance[reachable])
    if strict and len(values) > 9:
        _require(values[8] < values[9], 'A capacity tie would make selection ambiguous.')
    if not len(values):
        return reachable
    if not strict:
        # Used only to compare a deliberately simpler alternative, with a defined tie rule.
        cells = sorted((int(distance[p]), p[0], p[1])
                       for p in map(tuple, np.argwhere(reachable)))[:9]
        selected = np.zeros(grid.shape, dtype=bool)
        for _, r, c in cells:
            selected[r, c] = True
        return selected
    return reachable & (distance <= values[min(8, len(values) - 1)])


def _analyze(grid, audit=False):
    grid = _grid(grid)
    bg, wall, pos, target, payload = _roles(grid)
    collector = _collector_mask(grid.shape, pos)
    regions = []
    terrain = np.full(grid.shape, wall, dtype=np.int8)
    all_depths = np.full(grid.shape, -1, dtype=np.int16)
    for mask in _components(grid != wall):
        depth = _erosion_depths(mask) if audit else _depths(mask)
        period, key, directions = _decode_key(grid, mask, depth, bg, payload, collector)
        terrain[mask] = np.asarray(period, dtype=np.int8)[depth[mask] % len(period)]
        all_depths[mask] = depth[mask]
        regions.append(dict(mask=mask, depth=depth, period=period, key=key, directions=directions))
    active_region = next(region for region in regions if region['mask'][pos])
    _require(target in active_region['period'], 'The sampler rim must occur in its region key.')
    tracks = (terrain == target) & (grid != wall)
    distance = _distance(tracks | collector, collector, wave=audit)
    selected = _selection(grid, payload, collector, distance)
    return dict(grid=grid, bg=bg, wall=wall, pos=pos, target=target, payload=payload,
                collector=collector, regions=regions, active_region=active_region,
                terrain=terrain, tracks=tracks, depth=all_depths, distance=distance, selected=selected)


def _output(a):
    out = a['grid'].copy()
    out[a['selected']] = a['bg']
    r, c = a['pos']
    for i in range(int(a['selected'].sum())):
        out[r + i // 3, c + i % 3] = a['payload']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Reconstruct the invisible terrain; collect without any generation state."""
    return _output(_analyze(input_grid))


def _alternative_tracks(a, metric):
    result = np.zeros(a['grid'].shape, dtype=bool)
    for region in a['regions']:
        mask = region['mask']
        if metric == 'manhattan_depth':
            depth = _depths(mask, diagonal=False)
        else:
            rr, cc = np.nonzero(mask)
            rows, cols = np.indices(mask.shape)
            depth = np.minimum.reduce((rows - rr.min(), rr.max() - rows,
                                       cols - cc.min(), cc.max() - cols))
        colors = np.asarray(region['period'])[depth[mask] % len(region['period'])]
        result[mask] = colors == a['target']
    return result


def pair_features(input_grid):
    """Task measurements computed only from the supplied input."""
    a = _analyze(input_grid)
    source = (a['grid'] == a['payload']) & ~a['collector']
    selected = a['selected']
    reachable = source & (a['distance'] >= 0)
    retained = reachable & ~selected
    r, c = a['pos']
    rr, cc = np.indices(a['grid'].shape)
    direct = np.maximum(r - rr, 0) + np.maximum(rr - r - 2, 0) + np.maximum(c - cc, 0) + np.maximum(cc - c - 2, 0)
    wall_distance = _distance(a['grid'] != a['wall'], a['collector'])
    without_a = _selection(a['grid'], a['payload'], a['collector'], wall_distance, strict=False)
    matching = source & a['tracks']
    f = dict(height=a['grid'].shape[0], width=a['grid'].shape[1],
             roles=(a['bg'], a['wall'], a['target'], a['payload']),
             regions=len(a['regions']), period=len(a['active_region']['period']),
             selected=int(selected.sum()), reachable=int(reachable.sum()),
             excess=int(retained.sum()),
             wrong_band=int((source & a['active_region']['mask'] & ~a['tracks']).sum()),
             disconnected_matching=int((matching & a['active_region']['mask'] & (a['distance'] < 0)).sum()),
             wall_disconnected=int((source & ~a['active_region']['mask']).sum()),
             selected_depths=tuple(sorted(set(map(int, a['depth'][selected])))),
             active_depths=tuple(sorted(set(map(int, a['depth'][(a['distance'] > 0)])))),
             repeated_selected=bool(np.any(a['depth'][selected] >= len(a['active_region']['period']))),
             without_a_diff=not np.array_equal(without_a, selected),
             without_b_diff=not np.array_equal(matching, selected),
             path_beats_manhattan=bool(retained.any() and selected.any()
                                      and direct[selected].max() > direct[retained].min()),
             max_route=int(a['distance'].max()),
             frontier_gap=int(a['distance'][retained].min() - a['distance'][selected].max()) if retained.any() else 0,
             neutral_entry=any(a['bg'] in region['period'] and len(region['period']) > 1 for region in a['regions']),
             orientations=tuple(sorted({d for region in a['regions'] for d in region['directions']})))
    for metric in ('manhattan_depth', 'box_depth'):
        tracks = _alternative_tracks(a, metric)
        dist = _distance(tracks | a['collector'], a['collector'])
        other = _selection(a['grid'], a['payload'], a['collector'], dist, strict=False)
        f[metric + '_diff'] = not np.array_equal(other, selected)
    return f


def _rectangle(mask, r0, c0, r1, c1):
    mask[r0, c0:c1 + 1] = True
    mask[r1, c0:c1 + 1] = True
    mask[r0:r1 + 1, c0] = True
    mask[r0:r1 + 1, c1] = True


def _layout(rng, shape, complex_scene):
    """Compose cuts, attached notches, wall islands, loops and partial rails."""
    h, w = shape
    wall = np.zeros(shape, dtype=bool)
    _rectangle(wall, 0, 0, h - 1, w - 1)
    if rng.random() < .45:
        vertical = bool(rng.integers(2))
        work = wall if vertical else wall.T
        cut = int(rng.integers(5, work.shape[1] - 5))
        work[:, cut] = True
        if rng.random() < .35:
            at = int(rng.integers(3, work.shape[0] - 3))
            work[at:at + int(rng.integers(1, 4)), cut] = False
    count = int(rng.integers(1 if complex_scene else 0, 5 if complex_scene else 3))
    for _ in range(count):
        work = wall if rng.integers(2) else wall.T
        if rng.integers(2):
            work = work[::-1]
        x = int(rng.integers(4, work.shape[1] - 4))
        end = int(rng.integers(3, max(4, work.shape[0] // 2)))
        work[:end + 1, x] = True
        if rng.random() < .6:
            shift = int(rng.choice((-2, -1, 1, 2)))
            lo, hi = sorted((x, x + shift))
            work[end, lo:hi + 1] = True
            length = int(rng.integers(1, 4))
            work[end:min(work.shape[0] - 2, end + length + 1), x + shift] = True
    if complex_scene and rng.random() < .5:
        r0, c0 = int(rng.integers(4, h - 6)), int(rng.integers(4, w - 6))
        r1, c1 = min(h - 3, r0 + int(rng.integers(2, 6))), min(w - 3, c0 + int(rng.integers(2, 6)))
        _rectangle(wall, r0, c0, r1, c1)
        if rng.random() < .45:
            wall[r0, c0 + 1:c1] = False
    return wall


def _fallback_layout():
    wall = np.zeros((25, 27), dtype=bool)
    _rectangle(wall, 0, 0, 24, 26)
    wall[0:8, 13] = True
    wall[7, 13:17] = True
    return wall


def _pick(rng, mask, count, style, forced=()):
    chosen = list(dict.fromkeys(map(int, forced)))
    available = set(map(int, np.flatnonzero(mask)))
    if not set(chosen) <= available or not len(chosen) <= count <= len(available):
        return None
    available.difference_update(chosen)
    if not count:
        return []
    anchor = chosen[0] if chosen else int(rng.choice(sorted(available)))
    ar, ac = divmod(anchor, mask.shape[1])
    horizontal = bool(rng.integers(2))
    while len(chosen) < count:
        cells = np.array(sorted(available), dtype=int)
        rr, cc = cells // mask.shape[1], cells % mask.shape[1]
        if style == 'dispersed':
            cell = int(rng.choice(cells))
        else:
            score = ((rr - ar) ** 2 + (cc - ac) ** 2) if style == 'cluster' else (
                np.abs(rr - ar) * 30 + np.abs(cc - ac) if horizontal
                else np.abs(cc - ac) * 30 + np.abs(rr - ar))
            cell = int(rng.choice(cells[score == score.min()]))
        chosen.append(cell)
        available.remove(cell)
    return chosen


def _supply(rng, a, allowed, count, obligation, style):
    distance = a['distance']
    active = allowed & (distance > 0)
    if active.sum() < min(count, 9):
        return None
    if count <= 9:
        forced = []
        if obligation == 'partial':
            deep = active & (a['depth'] >= len(a['active_region']['period']))
            if not deep.any() or len(set(map(int, a['depth'][active]))) < 2:
                return None
            forced.append(int(rng.choice(np.flatnonzero(deep))))
            other = active & (a['depth'] != a['depth'].flat[forced[0]])
            if other.any():
                forced.append(int(rng.choice(np.flatnonzero(other))))
        return _pick(rng, active, count, style, forced)
    center = np.zeros(active.shape, dtype=bool)
    r, c = a['pos']
    center[r + 1, c + 1] = True
    center_distance = _distance(a['tracks'] | a['collector'], center)
    r, c = a['pos']
    rr, cc = np.indices(active.shape)
    direct = np.maximum(r - rr, 0) + np.maximum(rr - r - 2, 0) + np.maximum(c - cc, 0) + np.maximum(cc - c - 2, 0)
    sensitive = {}
    if obligation == 'route':
        for metric in ('manhattan_depth', 'box_depth'):
            sensitive[metric] = active & ~_alternative_tracks(a, metric)
            if not sensitive[metric].any():
                return None
    for cell in rng.permutation(np.flatnonzero(active)):
        cell = int(cell)
        # Teach path ranking without introducing an unobserved origin convention.
        near = active & (distance <= distance.flat[cell]) & (center_distance <= center_distance.flat[cell])
        far = active & (distance > distance.flat[cell]) & (center_distance > center_distance.flat[cell])
        if near.sum() < 9 or far.sum() < count - 9:
            continue
        forced = [cell]
        if obligation == 'route':
            far_choices = far & (direct < direct.flat[cell])
            if not far_choices.any() or any(not (mask & near).any() for mask in sensitive.values()):
                continue
            for mask in sensitive.values():
                forced.append(int(rng.choice(np.flatnonzero(mask & near))))
        else:
            far_choices = far
        retained_cell = int(rng.choice(np.flatnonzero(far_choices)))
        selected = _pick(rng, near, 9, style, forced)
        retained = _pick(rng, far, count - 9, style, [retained_cell])
        if selected is not None and retained is not None:
            return selected + retained
    return None


def _build_pair(rng, request, test, roles):
    bg, wall_color, payload, palette = roles
    obligation, count, length = request
    for attempt in range(100):
        if attempt == 99:
            wall = _fallback_layout()
        else:
            low, high = ((22, 31) if test else (18, 29)) if obligation else ((18, 31) if test else (13, 29))
            shape = tuple(map(int, rng.integers(low, high, size=2)))
            wall = _layout(rng, shape, complex_scene=bool(test or obligation == 'route' or rng.random() < .55))
        regions = _components(~wall)
        depths = [_depths(mask) for mask in regions]
        options = [_paths(mask, depth) for mask, depth in zip(regions, depths)]
        viable = [i for i, paths in enumerate(options) if any(p[4] >= max(length, 5 if obligation == 'partial' else 4) for p in paths)]
        if not viable:
            continue
        main = int(rng.choice(viable))
        grid = np.full(wall.shape, bg, dtype=np.int8)
        grid[wall] = wall_color
        keys = np.zeros(wall.shape, dtype=bool)
        key_buffer = np.zeros(wall.shape, dtype=bool)
        periods = []
        for i, paths in enumerate(options):
            if i == main:
                n = length
            elif not paths or rng.random() < .3:
                periods.append((bg,))
                continue
            else:
                n = int(rng.integers(1, min(3, max(p[4] for p in paths)) + 1))
            choices = [p for p in paths if p[4] >= n]
            if test and i == main:
                upward = [p for p in choices if p[2] == -1]
                if upward:
                    choices = upward
            elif not test and i == main:
                downward = [p for p in choices if p[2] == 1]
                if downward:
                    choices = downward
            path = choices[int(rng.integers(len(choices)))]
            period = list(map(int, rng.choice(palette, n, replace=False)))
            if i == main and obligation == 'partial':
                period[0] = bg
            elif i == main and obligation == 'route':
                period[1] = bg
            elif n > 1 and rng.random() < .3:
                period[int(rng.integers(n - 1))] = bg
            periods.append(tuple(period))
            r, c, dr, dc, _ = path
            for j, color in enumerate(period):
                rr, cc = r + j * dr, c + j * dc
                grid[rr, cc] = color
                keys[rr, cc] = True
                key_buffer[max(0, rr - 1):rr + 2, max(0, cc - 1):cc + 2] = True
        period = periods[main]
        target = period[0] if obligation == 'route' and period[0] != bg else int(rng.choice([x for x in period if x != bg]))
        terrain = np.full(wall.shape, wall_color, dtype=np.int8)
        for mask, depth, p in zip(regions, depths, periods):
            terrain[mask] = np.asarray(p)[depth[mask] % len(p)]
        positions = []
        for rr, cc in np.argwhere(regions[main]):
            r, c = int(rr), int(cc)
            if r + 3 <= grid.shape[0] and c + 3 <= grid.shape[1]:
                if np.all(regions[main][r:r + 3, c:c + 3]) and not key_buffer[r:r + 3, c:c + 3].any():
                    if (terrain[r:r + 3, c:c + 3] == target).any():
                        positions.append((r, c))
        rng.shuffle(positions)
        for pos in positions[:80]:
            collector = _collector_mask(grid.shape, pos)
            temp = grid.copy()
            r, c = pos
            temp[r:r + 3, c:c + 3] = target
            temp[r + 1, c + 1] = payload
            distance = _distance((terrain == target) | collector, collector)
            a = dict(grid=temp, bg=bg, wall=wall_color, pos=pos, target=target, payload=payload,
                     collector=collector, terrain=terrain, tracks=terrain == target,
                     depth=np.maximum.reduce(depths), distance=distance,
                     regions=[dict(mask=m, depth=d, period=p) for m, d, p in zip(regions, depths, periods)],
                     active_region=dict(mask=regions[main], depth=depths[main], period=period))
            allowed = ~wall & ~collector & ~key_buffer
            style = str(rng.choice(('dispersed', 'cluster', 'line')))
            supply = _supply(rng, a, allowed, count, obligation, style)
            if supply is None:
                continue
            wrong = allowed & regions[main] & (terrain != target)
            disconnected = allowed & regions[main] & (terrain == target) & (distance < 0)
            if obligation and (not wrong.any() or not disconnected.any()):
                continue
            other = allowed & (distance < 0)
            distraction = []
            if wrong.any():
                distraction += _pick(rng, wrong, int(rng.integers(1, min(6, int(wrong.sum())) + 1)), style)
            if disconnected.any():
                distraction += _pick(rng, disconnected, int(rng.integers(1, min(5, int(disconnected.sum())) + 1)), style)
            remaining = other.copy()
            remaining.flat[distraction] = False
            if remaining.any() and rng.random() < .65:
                distraction += _pick(rng, remaining, int(rng.integers(0, min(7, int(remaining.sum())) + 1)), style)
            temp.flat[supply + distraction] = payload
            try:
                info = _analyze(temp)
                out = _output(info)
                f = pair_features(temp)
                if obligation:
                    _require(f['without_a_diff'] and f['without_b_diff'], 'Missing parental witnesses.')
                if obligation == 'partial':
                    _require(f['selected'] in (2, 4) and f['repeated_selected'] and len(f['active_depths']) >= 2,
                             'The partial demonstration must show a bridge between repeating tracks.')
                if obligation == 'route':
                    _require(f['excess'] > 0 and f['path_beats_manhattan'] and f['manhattan_depth_diff'] and f['box_depth_diff'],
                             'The route demonstration must distinguish path and contour shortcuts.')
            except AssertionError:
                continue
            return dict(input=temp, output=out)
    raise RuntimeError('Bounded construction failed for ' + str(request))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if not _integer(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if not _integer(value) or not lo <= value <= hi:
            raise ValueError(name + ' is outside its legal integer range.')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test.')
    rng = np.random.default_rng(seed)
    colors = list(map(int, rng.permutation(10)))
    # All conventions are visible. Palette roles may be reassigned for each pair.
    varying = bool(rng.random() < .75)
    requests = [('partial', int(rng.choice((2, 4))), 2),
                ('route', int(rng.integers(10, 21)), 3)]
    for _ in range(num_train - 2):
        load = int(rng.integers(1, 10)) if rng.random() < .7 else int(rng.integers(10, 24))
        requests.append(('', load, int(rng.integers(1, 4))))
    requests = [requests[int(i)] for i in rng.permutation(num_train)]
    tests = [('', int(rng.integers(1, 10)) if rng.random() < .6 else int(rng.integers(10, 26)),
              int(rng.integers(2, 5))) for _ in range(num_test)]
    puzzle = dict(train=[], test=[])
    seen = set()
    for split, specs in (('train', requests), ('test', tests)):
        for request in specs:
            for _ in range(10):
                local = list(map(int, rng.permutation(10))) if varying else colors
                roles = (local[0], local[1], local[2], tuple(local[3:]))
                pair = _build_pair(rng, request, split == 'test', roles)
                sig = _signature(pair['input'])
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Could not construct distinct inputs.')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle keys.')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists.')
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, 'Invalid pair counts.')
    seen, train_features = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair keys.')
            grid, out = _grid(pair['input']), _grid(pair['output'])
            _require(grid.shape == out.shape, 'Dimensions must be preserved.')
            sig = _signature(grid)
            _require(sig not in seen, 'Duplicate input.')
            seen.add(sig)
            a = _analyze(grid, audit=True)
            selected = a['selected']
            _require(np.array_equal(out, _output(a)), 'Incorrect contour/route/capacity/prefix output.')
            _require(np.array_equal(out[grid == a['wall']], grid[grid == a['wall']]), 'Walls changed.')
            unchanged = ~selected & ~a['collector']
            _require(np.array_equal(out[unchanged], grid[unchanged]), 'Unselected field cells changed.')
            _require(int(selected.sum()) <= 9, 'Capacity exceeded.')
            _require(np.all(a['terrain'][selected] == a['target']), 'Collected from a forbidden contour.')
            _require(np.all(out[selected] == a['bg']), 'Selected cells must be erased.')
            for region in a['regions']:
                _require(np.array_equal(region['depth'], _depths(region['mask'])), 'Erosion/BFS disagreement.')
                for cell in region['key']:
                    _require(out[cell] == grid[cell], 'A key changed.')
            _require(np.array_equal(a['distance'], _distance(a['tracks'] | a['collector'], a['collector'])),
                     'Wavefront/BFS disagreement.')
            center = np.zeros(grid.shape, dtype=bool)
            r, c = a['pos']
            center[r + 1, c + 1] = True
            cd = _distance(a['tracks'] | a['collector'], center)
            _require(np.array_equal(selected, _selection(grid, a['payload'], a['collector'], cd)),
                     'Generated examples must avoid a distance-origin ambiguity.')
            f = pair_features(grid)
            if split == 'train':
                train_features.append(f)
    _require(any(f['selected'] in (2, 4) and f['period'] == 2 and f['repeated_selected']
                 and len(f['active_depths']) >= 2 for f in train_features), 'Missing partial bridge demonstration.')
    _require(any(f['period'] == 3 and f['excess'] and f['path_beats_manhattan']
                 and f['manhattan_depth_diff'] and f['box_depth_diff'] for f in train_features),
             'Missing contour and distance demonstration with excess.')
    _require(any(f['disconnected_matching'] and f['wrong_band'] and f['without_a_diff']
                 and f['without_b_diff'] for f in train_features), 'Both parental concepts must affect answers.')
    _require(any(f['neutral_entry'] for f in train_features), 'Training must reveal neutral entries in keys.')


def self_test():
    """Bounded 42-seed checks, mechanism ablations, corruptions and legal counts."""
    shapes, roles, periods, regions, selected_counts, signatures, geometry, orientations = (set() for _ in range(8))
    max_route, neutral, bridges, reversals = 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_puzzle_signature(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _analyze(pair['input'])
                f = pair_features(pair['input'])
                _require(np.array_equal(solve(pair['input']), pair['output']), 'Input-only solver mismatch.')
                shapes.add(pair['input'].shape)
                roles.add(f['roles'])
                periods.add(f['period'])
                regions.add(f['regions'])
                selected_counts.add(f['selected'])
                geometry.add(_signature((pair['input'] == a['wall']).astype(np.int8)))
                orientations.update(f['orientations'])
                max_route = max(max_route, f['max_route'])
                neutral += f['neutral_entry']
                bridges += len(f['active_depths']) >= 2 and f['period'] > 1
                reversals += f['path_beats_manhattan']
    _require(len(signatures) == 42 and len(geometry) >= 100, 'Cross-seed structural variation collapsed.')
    _require(len(shapes) >= 60 and len(roles) >= 50, 'Size/palette variation collapsed.')
    _require(periods == {1, 2, 3, 4} and len(regions) >= 3, 'Period/region variation collapsed.')
    _require(set(range(1, 10)) <= selected_counts, 'Loads must vary beyond full counters.')
    _require(orientations == {(dr, dc) for dr in (-1, 1) for dc in (-1, 1)}, 'Key orientation variation missing.')
    _require(max_route >= 30 and neutral >= 10 and bridges >= 42 and reversals >= 42, 'Core structural features missing.')
    first = generate_puzzle(713, 2, 1)
    _require(_puzzle_signature(first) == _puzzle_signature(generate_puzzle(713, 2, 1)), 'Nondeterminism.')
    for nt, ne in ((2, 1), (10, 1), (10, 4), (5, 4)):
        puzzle = generate_puzzle(891 + nt + ne, nt, ne)
        _require(len(puzzle['train']) == nt and len(puzzle['test']) == ne, 'Explicit counts ignored.')
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1), (4, None)):
        try:
            generate_puzzle(4, nt, ne)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted.')
    # Corrupt a wall and a selected source separately: labels are checked cell by cell.
    for source in (False, True):
        broken = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in first[s]] for s in first}
        pair = broken['train'][0]
        a = _analyze(pair['input'])
        pos = tuple(map(int, np.argwhere(a['selected'])[0])) if source else (0, 0)
        pair['output'][pos] = a['payload'] if source else a['bg']
        try:
            validate_generated_puzzle(broken)
        except AssertionError:
            continue
        raise AssertionError('Corrupt output accepted.')
    # A twelve-way nearest tie across the ninth slot must be rejected.
    grid = np.full((9, 9), 0, dtype=np.int8)
    collector = _collector_mask(grid.shape, (3, 3))
    distance = _distance(np.ones(grid.shape, dtype=bool), collector)
    grid[distance == 1] = 2
    try:
        _selection(grid, 2, collector, distance)
    except AssertionError:
        pass
    else:
        raise AssertionError('Ambiguous capacity frontier accepted.')
    # Input-only support beyond the positive-load, distinct-key sampling policy.
    partial = next(p for p in first['train'] if pair_features(p['input'])['selected'] < 9)
    a = _analyze(partial['input'])
    zero = a['grid'].copy()
    zero[(zero == a['payload']) & ~a['collector']] = a['bg']
    _require(np.array_equal(solve(zero), zero), 'Zero-collection input should stay unchanged.')
    repeated = zero.copy()
    for cell in a['active_region']['key']:
        repeated[cell] = a['target']
    available = a['active_region']['mask'] & ~a['collector']
    for cell in a['active_region']['key']:
        available[cell] = False
    repeated.flat[np.flatnonzero(available)[:3]] = a['payload']
    b = _analyze(repeated)
    _require(len(b['active_region']['period']) == 2 and len(set(b['active_region']['period'])) == 1,
             'Repeated entries must remain a valid visibly terminated key.')
    _require(int(b['selected'].sum()) == 3, 'Repeated-key tracks should collect all three sources.')
    _require(np.array_equal(solve(repeated), _output(b)), 'Repeated-key input-only solution failed.')
    return {'seeds': 42, 'pairs': 210, 'shapes': len(shapes), 'role_assignments': len(roles),
            'wall_geometries': len(geometry), 'periods': sorted(periods), 'region_counts': sorted(regions),
            'selected_counts': sorted(selected_counts), 'max_route': max_route,
            'neutral_key_pairs': neutral, 'bridge_pairs': bridges, 'distance_reversals': reversals}
