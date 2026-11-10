"""Spotlight audit: count completely exposed rooms in transparent partitions.

The solver reads only pixels. Construction uses explicit room rectangles, while
decoding reconstructs wall masks and flood-fills their bounded components.
"""
from __future__ import annotations

import json
import numpy as np


HOST_SEEDS = (0, 121351144, 822391258, 99297163, 795685553, 370063645,
              189339636, 53300566, 165575160, 454403729, 711719053,
              516101458, 16654621, 711748840, 406214077, 743232448,
              993113391, 257445465, 209822235, 827559275, 105179585,
              988226288, 511693375, 659573134, 678792375, 43166985,
              929178941, 837919262, 839544595, 889161585, 27355475,
              992647478, 146563301, 536977735, 291480192, 187199945,
              830464402, 668111421, 790330046, 866788018, 667634302,
              330350547)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid with dimensions 1..30")
    return a.astype(np.int8, copy=False)


def _key(a):
    return a.shape, a.tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    h, w = mask.shape
    for row, col in np.argwhere(mask):
        row, col = int(row), int(col)
        if seen[row, col]:
            continue
        cells, stack = [], [(row, col)]
        seen[row, col] = True
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        result.append(cells)
    return result


def _bounded_rooms(walls):
    rooms = []
    free = np.pad(~walls, 1, constant_values=True)
    for cells in _components(free):
        if any(y in (0, free.shape[0] - 1) or x in (0, free.shape[1] - 1)
               for y, x in cells):
            continue
        ys, xs = zip(*cells)
        t, l, b, r = min(ys) - 1, min(xs) - 1, max(ys) - 1, max(xs) - 1
        if len(cells) != (b - t + 1) * (r - l + 1):
            raise ValueError("room interiors must be rectangles")
        rooms.append((t, l, b, r))
    return sorted(rooms)


def _walls(shape, boxes):
    mask = np.zeros(shape, dtype=bool)
    for t, l, b, r in boxes:
        mask[t, l:r + 1] = mask[b, l:r + 1] = True
        mask[t:b + 1, l] = mask[t:b + 1, r] = True
    return mask


def _repair(grid, color, excluded):
    visible = grid == color
    horizontal, vertical = np.zeros_like(visible), np.zeros_like(visible)
    horizontal[:, 1:-1] = visible[:, :-2] & visible[:, 2:]
    vertical[1:-1, :] = visible[:-2, :] & visible[2:, :]
    # Exactly one visibly occupied missing pixel, with original supports on
    # both sides. Repairs are simultaneous; repaired pixels are not supports.
    repair = (horizontal | vertical) & ~np.isin(grid, excluded) & ~visible
    return visible | repair, repair


def _recover_frame(grid, color, background, emitter, payload):
    positions = np.argwhere(grid == color)
    if len(positions) < 8:
        return None
    t, l = map(int, positions.min(axis=0))
    b, r = map(int, positions.max(axis=0))
    if b - t < 2 or r - l < 2:
        return None
    if any(grid[y, x] != color for y, x in ((t, l), (t, r), (b, l), (b, r))):
        return None
    mask, repairs = _repair(grid, color, (background, emitter, payload))
    if not (mask[t, l:r + 1].all() and mask[b, l:r + 1].all()
            and mask[t:b + 1, l].all() and mask[t:b + 1, r].all()):
        return None
    try:
        local = _bounded_rooms(mask[t:b + 1, l:r + 1])
    except ValueError:
        return None
    rooms = [(y1 + t, x1 + l, y2 + t, x2 + l) for y1, x1, y2, x2 in local]
    if not rooms:
        return None
    boxes = [(y1 - 1, x1 - 1, y2 + 1, x2 + 1) for y1, x1, y2, x2 in rooms]
    if not np.array_equal(mask, _walls(grid.shape, boxes)):
        return None
    return dict(color=int(color), box=(t, l, b, r), rooms=rooms,
                mask=mask, repairs=repairs)


def _fit_rail(ys, sign):
    j = np.arange(len(ys))
    fits = []
    for period in (1, 2, 3):
        for phase in range(period):
            expected = ys[0] + sign * ((j + phase) // period)
            if np.array_equal(expected, ys) and len(set(ys)) >= 3:
                fits.append((period, phase))
    if len(fits) != 1:
        raise ValueError("terminal rail must identify one cadence and phase")
    return fits[0]


def _emitter(input_grid):
    frequencies = np.bincount(input_grid.ravel(), minlength=10)
    background = int(np.argmax(frequencies))
    if frequencies[background] <= input_grid.size // 2:
        raise ValueError("background must occupy more than half of the input")
    matches = []
    for rotation in range(4):
        g = np.rot90(input_grid, -rotation)
        for color in np.unique(g):
            color = int(color)
            if color == background:
                continue
            points = np.argwhere(g == color)
            if len(points) < 13:
                continue
            rear, mouth = int(points[:, 1].min()), int(points[:, 1].max())
            if mouth - rear < 5:
                continue
            cap = np.flatnonzero(g[:, rear] == color)
            if len(cap) < 3 or not np.all(np.diff(cap) == 1):
                continue
            rails = [np.flatnonzero(g[:, x] == color) for x in range(rear + 1, mouth + 1)]
            if any(len(v) != 2 or v[1] - v[0] < 2 for v in rails):
                continue
            upper = np.array([v[0] for v in rails])
            lower = np.array([v[1] for v in rails])
            if (cap[0], cap[-1]) != (upper[0], lower[0]):
                continue
            try:
                fits = (_fit_rail(upper, -1), _fit_rail(lower, 1))
            except ValueError:
                continue
            footprint = np.zeros(g.shape, dtype=bool)
            for x in range(rear + 1, g.shape[1]):
                j = x - rear - 1
                up = int(upper[0]) - (j + fits[0][1]) // fits[0][0]
                down = int(lower[0]) + (j + fits[1][1]) // fits[1][0]
                footprint[:, x] = (np.arange(g.shape[0]) > up) & (np.arange(g.shape[0]) < down)
            chamber = footprint.copy()
            chamber[:, mouth + 1:] = False
            interior_colors = set(map(int, np.unique(g[chamber]))) - {background}
            if len(interior_colors) != 1:
                continue
            payload = interior_colors.pop()
            patch = g == payload
            if np.any(patch & ~chamber) or len(_components(patch)) != 1:
                continue
            matches.append(dict(grid=g, rotation=rotation, background=background,
                                emitter=color, payload=payload, rear=rear, mouth=mouth,
                                fits=fits, footprint=footprint))
    if len(matches) != 1:
        raise ValueError("input must identify exactly one spotlight")
    return matches[0]


def _light(footprint, blockers):
    # Construction/analysis uses a union of complete downstream shadows.
    lit = footprint.copy()
    for y, x in np.argwhere(blockers & footprint):
        lit[y, x:] = False
    return lit


def _counts(rooms_by_frame, light, any_hit=False):
    return [sum(bool(np.any(light[t:b + 1, l:r + 1]) if any_hit else
                     np.all(light[t:b + 1, l:r + 1])) for t, l, b, r in rooms)
            for rooms in rooms_by_frame]


def _report(background, payload, frames, counts, rotation=0):
    # Row ordering is ordinary top-to-bottom, then left-to-right in the actual
    # presented input, independent of the spotlight's direction.
    height, width = frames[0]['mask'].shape
    positioned = []
    for frame, count in zip(frames, counts):
        t, l, b, r = frame['box']
        corners = np.zeros((height, width), dtype=bool)
        corners[t, l] = corners[b, r] = corners[t, r] = corners[b, l] = True
        actual = np.argwhere(np.rot90(corners, rotation))
        top, left = map(int, actual.min(axis=0))
        positioned.append((int(count), top, left, frame['color']))
    out = np.full((len(frames), 1 + max(counts)), background, dtype=np.int8)
    for row, (count, _, _, color) in enumerate(sorted(positioned)):
        out[row, 0] = color
        out[row, 1:1 + count] = payload
    return out


def _analyze(input_grid):
    a = _emitter(_grid(input_grid))
    g = a['grid']
    frames, leftovers = [], []
    excluded = (a['background'], a['emitter'], a['payload'])
    for color in np.unique(g):
        if int(color) in excluded:
            continue
        frame = _recover_frame(g, int(color), *excluded)
        if frame is None:
            leftovers.append(int(color))
        else:
            frames.append(frame)
    if not 1 <= len(frames) <= 4 or len(leftovers) > 1:
        raise ValueError("expected 1..4 partitions and at most one opacity color")
    frames.sort(key=lambda f: f['color'])
    union = np.logical_or.reduce([f['mask'] for f in frames])
    for i, frame in enumerate(frames):
        if frame['box'][1] < a['mouth'] + 2:
            raise ValueError("partitions must be downstream of the spotlight")
        for other in frames[:i]:
            t, l, b, r = frame['box']
            tt, ll, bb, rr = other['box']
            if not (b + 1 < tt or bb + 1 < t or r + 1 < ll or rr + 1 < l):
                raise ValueError("partitions need a separating background lane")
    opacity = leftovers[0] if leftovers else None
    if any(np.any(f['repairs'] & (g != opacity)) for f in frames):
        raise ValueError("only the opacity color may interrupt a frame")
    blockers = np.zeros_like(union) if opacity is None else (g == opacity) & ~union
    if np.any(blockers[:, :a['mouth'] + 1] & a['footprint'][:, :a['mouth'] + 1]):
        raise ValueError("no opaque cells in the spotlight chamber")
    a.update(frames=frames, union=union, opacity=opacity, blockers=blockers)
    a['light'] = _light(a['footprint'], blockers)
    a['counts'] = _counts([f['rooms'] for f in frames], a['light'])
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _analyze(input_grid)
    # Prediction instead scans each lane. A repaired wall interruption never
    # sets the blocked flag; boundaries themselves are transparent.
    light = np.zeros_like(a['footprint'])
    for y in range(light.shape[0]):
        blocked = False
        for x in range(a['rear'] + 1, light.shape[1]):
            if a['footprint'][y, x] and a['blockers'][y, x]:
                blocked = True
            light[y, x] = a['footprint'][y, x] and not blocked
    counts = _counts([f['rooms'] for f in a['frames']], light)
    return _report(a['background'], a['payload'], a['frames'], counts, a['rotation'])


def _partition(rng, box, count, style):
    rooms = [box]
    first_axis = None
    for step in range(count - 1):
        options = []
        for index, (t, l, b, r) in enumerate(rooms):
            for axis, start, end in ((0, t, b), (1, l, r)):
                if style == 'partial' and step == 1 and axis == first_axis:
                    continue
                if style == 'strip' and first_axis is not None and axis != first_axis:
                    continue
                for cut in range(start + 2, end - 1):
                    halves = ((t, l, cut, r), (cut, l, b, r)) if axis == 0 else ((t, l, b, cut), (t, cut, b, r))
                    capacity = sum(((bb - tt) // 2) * ((rr - ll) // 2)
                                   for tt, ll, bb, rr in rooms[:index] + rooms[index + 1:] + list(halves))
                    if capacity >= count:
                        options.append((index, axis, halves))
        if not options:
            return None
        index, axis, halves = options[int(rng.integers(len(options)))]
        if first_axis is None:
            first_axis = axis
        rooms[index:index + 1] = halves
    return rooms


def _sample_scene(rng, palette, number, max_rooms, rotation, force_partial=False, large_count=False):
    compact = not (force_partial or large_count) and rng.random() < 0.5
    h = int(rng.integers(12 if compact else 21, 24 if compact else 31))
    w = int(rng.integers(18 if compact else 23, 28 if compact else 31))
    bg, emitter, payload, opacity = map(int, palette[:4])
    periods = rng.integers(1, 4, size=2)
    phases = [int(rng.integers(p)) for p in periods]
    length = int(rng.integers(max(5, 2 * int(max(periods)) + 1), 10))
    rear = int(rng.integers(0, 3))
    widths = rng.integers(1, 4, size=2)
    up_span = int(widths[0]) + (length - 1 + phases[0]) // int(periods[0])
    down_span = int(widths[1]) + (length - 1 + phases[1]) // int(periods[1])
    if up_span + down_span + 1 > h:
        return None
    center = int(rng.integers(up_span, h - down_span))
    mouth = rear + length
    if w - mouth - 2 < (5 if force_partial else 3):
        return None
    grid = np.full((h, w), bg, dtype=np.int8)
    grid[center - widths[0]:center + widths[1] + 1, rear] = emitter
    footprint = np.zeros((h, w), dtype=bool)
    for x in range(rear + 1, w):
        j = x - rear - 1
        upper = center - int(widths[0]) - (j + phases[0]) // int(periods[0])
        lower = center + int(widths[1]) + (j + phases[1]) // int(periods[1])
        footprint[:, x] = (np.arange(h) > upper) & (np.arange(h) < lower)
        if x <= mouth:
            grid[upper, x] = grid[lower, x] = emitter
    # Connected, off-center payload patches of 1..4 pixels.
    sites = [(int(y), int(x)) for y, x in np.argwhere(footprint)
             if rear + 1 <= x <= min(rear + 3, mouth - 1)]
    patch = {sites[int(rng.integers(len(sites)))]}
    for _ in range(int(rng.integers(0, 4))):
        frontier = sorted({(yy, xx) for y, x in patch
                           for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))
                           if (yy, xx) in sites and (yy, xx) not in patch})
        if frontier:
            patch.add(frontier[int(rng.integers(len(frontier)))])
    for y, x in patch:
        grid[y, x] = payload
    frames = []
    for index in range(number):
        for _ in range(70):
            max_h = h if number == 1 else min(20 if number == 2 else 15, h)
            max_w = w - mouth - 2 if number == 1 else min(15 if number == 2 else 12, w - mouth - 2)
            bh = int(rng.integers(5 if force_partial and index == 0 else 3, max_h + 1))
            bw = int(rng.integers(5 if force_partial and index == 0 else 3, max_w + 1))
            t = int(rng.integers(0, h - bh + 1))
            l = int(rng.integers(mouth + 2, w - bw + 1))
            box = (t, l, t + bh - 1, l + bw - 1)
            if any(not (box[2] + 1 < f['box'][0] or f['box'][2] + 1 < box[0]
                            or box[3] + 1 < f['box'][1] or f['box'][3] + 1 < box[1]) for f in frames):
                continue
            capacity = ((bh - 1) // 2) * ((bw - 1) // 2)
            low = 7 if large_count and index == 0 else 3 if force_partial and index == 0 else 1
            if capacity < low:
                continue
            count = int(rng.integers(low, min(capacity, max_rooms) + 1))
            style = 'partial' if force_partial and index == 0 else str(rng.choice(['mixed', 'mixed', 'strip']))
            boxes = _partition(rng, box, count, style)
            if boxes is None:
                continue
            mask = _walls(grid.shape, boxes)
            frame = dict(color=int(palette[4 + index]), box=box, mask=mask,
                         rooms=[(a + 1, b + 1, c - 1, d - 1) for a, b, c, d in boxes])
            frames.append(frame)
            grid[mask] = frame['color']
            break
        else:
            return None
    if np.count_nonzero(grid == bg) <= grid.size // 2:
        return None
    union = np.logical_or.reduce([f['mask'] for f in frames])
    return dict(grid=grid, frames=frames, footprint=footprint, union=union,
                background=bg, emitter=emitter, payload=payload, opacity=opacity,
                rear=rear, mouth=mouth, rotation=rotation)


def _scene_counts(scene, blockers=None):
    if blockers is None:
        blockers = np.zeros_like(scene['union'])
    return _counts([f['rooms'] for f in scene['frames']], _light(scene['footprint'], blockers))


def _pair(scene, grid, blockers):
    counts = _scene_counts(scene, blockers)
    return {'input': np.rot90(grid, scene['rotation']).copy(),
            'output': _report(scene['background'], scene['payload'], scene['frames'], counts, scene['rotation'])}


def _lattice_counts(scene, light):
    result = []
    for f in scene['frames']:
        ys = sorted({v for t, l, b, r in f['rooms'] for v in (t - 1, b + 1)})
        xs = sorted({v for t, l, b, r in f['rooms'] for v in (l - 1, r + 1)})
        interiors = [(t + 1, l + 1, b - 1, r - 1) for t, b in zip(ys, ys[1:])
                     for l, r in zip(xs, xs[1:])]
        result.append(_counts([interiors], light)[0])
    return result


def _tie_witness(scene, counts):
    frames = scene['frames']
    for i, f in enumerate(frames):
        for j, other in enumerate(frames[:i]):
            if counts[i] != counts[j]:
                continue
            # The demonstrated top-first convention must differ from left-first.
            masks = [np.argwhere(np.rot90(v['mask'], scene['rotation'])) for v in (f, other)]
            p, q = [v.min(axis=0) for v in masks]
            if (p[0] - q[0]) * (p[1] - q[1]) < 0:
                return True
    return False


def _add_distractors(rng, scene, grid, amount=2):
    sites = np.argwhere((grid == scene['background']) & ~scene['footprint'] & ~scene['union'])
    if len(sites):
        for y, x in sites[rng.choice(len(sites), size=min(amount, len(sites)), replace=False)]:
            grid[y, x] = scene['opacity']


def _damage_options(scene, grid):
    options = []
    for f in scene['frames']:
        color = f['color']
        for y, x in np.argwhere(f['mask'] & (grid == color)):
            # One-cell repair evidence must remain visible on both sides.
            horizontal = 0 < x < grid.shape[1] - 1 and grid[y, x - 1] == grid[y, x + 1] == color
            vertical = 0 < y < grid.shape[0] - 1 and grid[y - 1, x] == grid[y + 1, x] == color
            degree = sum(0 <= yy < grid.shape[0] and 0 <= xx < grid.shape[1] and grid[yy, xx] == color
                         for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)))
            if degree == 2 and horizontal != vertical:
                options.append((int(y), int(x)))
    return options


def _anchor(rng, palette, rotation):
    for _ in range(1600):
        scene = _sample_scene(rng, palette, 3, 6, rotation, force_partial=True)
        if scene is None:
            continue
        clean = _scene_counts(scene)
        room_lists = [f['rooms'] for f in scene['frames']]
        if (max(clean) < 2 or len(set(clean)) < 2 or clean == [len(v) for v in room_lists]
                or clean == _counts(room_lists, scene['footprint'], any_hit=True)
                or clean == _lattice_counts(scene, scene['footprint'])):
            continue
        spatial = sorted(zip(scene['frames'], clean), key=lambda z: tuple(
            np.argwhere(np.rot90(z[0]['mask'], rotation)).min(axis=0)))
        if [v for _, v in spatial] == sorted(clean):
            continue
        grid = scene['grid'].copy()
        _add_distractors(rng, scene, grid)
        sites = [(int(y), int(x)) for y, x in np.argwhere(scene['footprint'] & (grid == scene['background']) & ~scene['union'])
                 if scene['mouth'] < x < grid.shape[1] - 2
                 and not any(t <= y <= b and l <= x <= r for f in scene['frames'] for t, l, b, r in [f['box']])]
        rng.shuffle(sites)
        for y, x in sites[:100]:
            blocker = np.zeros_like(scene['union'])
            blocker[y, x] = True
            occluded = _scene_counts(scene, blocker)
            if occluded == clean or max(occluded) < 1:
                continue
            if not (_tie_witness(scene, clean) or _tie_witness(scene, occluded)):
                continue
            edited = grid.copy()
            edited[y, x] = scene['opacity']
            # A repaired ink pixel that would have changed the answer if it
            # were an ordinary blocker provides a causal reconstruction witness.
            damage = _damage_options(scene, edited)
            rng.shuffle(damage)
            for yy, xx in damage:
                naive = blocker.copy()
                naive[yy, xx] = True
                if _scene_counts(scene, naive) == occluded:
                    continue
                edited[yy, xx] = scene['opacity']
                first, second = _pair(scene, grid, np.zeros_like(blocker)), _pair(scene, edited, blocker)
                try:
                    for pair in (first, second):
                        analysis = _analyze(pair['input'])
                        intended = {f['color']: f['mask'] for f in scene['frames']}
                        if ({f['color'] for f in analysis['frames']} != set(intended)
                                or not all(np.array_equal(f['mask'], intended[f['color']]) for f in analysis['frames'])):
                            raise ValueError('spurious repair')
                        if not np.array_equal(solve(pair['input']), pair['output']):
                            raise ValueError('construction mismatch')
                except ValueError:
                    break
                return [first, second]
    raise RuntimeError('bounded teaching-scene construction exhausted')


def _ordinary(rng, palette, rotation, shift, is_test):
    for _ in range(500):
        scene = _sample_scene(rng, palette, int(rng.integers(1, 5)),
                              8 if is_test and shift == 'count' else 6, rotation,
                              large_count=is_test and shift == 'count')
        if scene is None:
            continue
        grid = scene['grid'].copy()
        blockers = np.zeros_like(scene['union'])
        sites = [(int(y), int(x)) for y, x in np.argwhere(scene['footprint'] & (grid == scene['background']))
                 if x > scene['mouth'] and x < grid.shape[1] - 2]
        rng.shuffle(sites)
        target = int(rng.integers(0, 5))
        used_rows = set()
        for y, x in sites:
            if y in used_rows:
                continue
            blockers[y, x] = True
            grid[y, x] = scene['opacity']
            used_rows.add(y)
            if len(used_rows) >= target:
                break
        if target == 0:
            grid[blockers] = scene['background']
            blockers[:] = False
        if is_test and shift == 'occlusion':
            candidates = [(y, x, xx) for y, x in sites for xx in range(x + 2, grid.shape[1] - 1)
                          if grid[y, xx] == scene['background'] and not scene['union'][y, x]
                          and scene['footprint'][y, xx]]
            rng.shuffle(candidates)
            for y, x, xx in candidates:
                trial = blockers.copy()
                trial[y, x] = trial[y, xx] = True
                if _scene_counts(scene, trial) != _scene_counts(scene):
                    blockers = trial
                    grid[y, x] = grid[y, xx] = scene['opacity']
                    break
            else:
                continue
        _add_distractors(rng, scene, grid, int(rng.integers(0, 4)))
        options = _damage_options(scene, grid)
        rng.shuffle(options)
        for y, x in options[:int(rng.integers(0, 5))]:
            # Recheck support after each mutation; simultaneous decoding cannot
            # rely on a second damaged pixel as a support.
            candidate = grid.copy()
            candidate[y, x] = scene['opacity']
            if all(np.array_equal(_repair(candidate, f['color'],
                                          (scene['background'], scene['emitter'], scene['payload']))[0], f['mask'])
                   for f in scene['frames']):
                grid = candidate
        pair = _pair(scene, grid, blockers)
        try:
            a = _analyze(pair['input'])
            intended = {f['color']: f['mask'] for f in scene['frames']}
            if ({f['color'] for f in a['frames']} != set(intended)
                    or not all(np.array_equal(f['mask'], intended[f['color']]) for f in a['frames'])):
                continue
            if not np.array_equal(solve(pair['input']), pair['output']):
                continue
        except ValueError:
            continue
        return pair
    raise RuntimeError('bounded ordinary-scene construction exhausted')


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _integer(num_train, 'num_train', 2, 10)
    ne = _integer(num_test, 'num_test', 1, 4)
    if nt <= ne:
        raise ValueError('num_train must be greater than num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = rng.permutation(10)
    shared_palette = bool(rng.random() < 0.3)
    shift = str(rng.choice(['count', 'orientation', 'occlusion']))
    held = int(rng.integers(4))
    training_rotations = [r for r in range(4) if r != held] if shift == 'orientation' else list(range(4))
    rotation = int(rng.choice(training_rotations))
    puzzle = {'train': _anchor(rng, palette, rotation), 'test': []}
    seen = {_key(p['input']) for p in puzzle['train']}
    for split, number in (('train', nt - 2), ('test', ne)):
        for _ in range(number):
            for _ in range(12):
                colors = palette if shared_palette else rng.permutation(10)
                r = held if split == 'test' and shift == 'orientation' else int(rng.choice(training_rotations))
                pair = _ordinary(rng, colors, r, shift, split == 'test')
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('duplicate inputs after bounded retries')
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    rooms = [f['rooms'] for f in a['frames']]
    counts = a['counts']
    no_shadow = _counts(rooms, a['footprint'])
    naive = _light(a['footprint'], a['grid'] == a['opacity']) if a['opacity'] is not None else a['footprint']
    partial = any(len(f['rooms']) != len({t for t, l, b, r in f['rooms']}) *
                  len({l for t, l, b, r in f['rooms']}) for f in a['frames'])
    spatial = sorted(zip(a['frames'], counts), key=lambda z: tuple(
        np.argwhere(np.rot90(z[0]['mask'], a['rotation'])).min(axis=0)))
    return dict(rotation=a['rotation'], shape=a['grid'].shape,
                palette=(a['background'], a['emitter'], a['payload']),
                counts=tuple(counts), totals=tuple(len(v) for v in rooms),
                max_rooms=max(map(len, rooms)), frames=len(rooms),
                full_vs_any=counts != _counts(rooms, a['light'], any_hit=True),
                omit_B=counts != [len(v) for v in rooms],
                opacity_matters=counts != no_shadow,
                repair_matters=counts != _counts(rooms, naive),
                partial=partial, lattice_matters=counts != _lattice_counts(a, a['light']),
                sorting_matters=[c for _, c in spatial] != sorted(counts),
                clean=not np.any(a['blockers'] & a['footprint']),
                repeated=bool(np.any((a['blockers'] & a['footprint']).sum(axis=1) >= 2)),
                tie=_tie_witness(a, counts),
                repaired=int(sum(f['repairs'].sum() for f in a['frames'])))


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(g) not in seen, 'duplicate input'
            seen.add(_key(g))
            a = _analyze(g)
            assert np.array_equal(solve(g), out), 'incorrect exposure report'
            assert out.shape == (len(a['frames']), 1 + max(a['counts']))
            assert set(map(int, out[:, 0])) == {f['color'] for f in a['frames']}
            assert set(map(int, np.unique(out[:, 1:]))) <= {a['background'], a['payload']}
            assert all(0 <= c <= len(f['rooms']) for c, f in zip(a['counts'], a['frames']))
            for f in a['frames']:
                # Euler count is an additional topology check, separate from
                # flood filling (subtract solid 2x2 faces with no room pixels).
                m = f['mask']
                edges = int((m[:, :-1] & m[:, 1:]).sum() + (m[:-1] & m[1:]).sum())
                unit_faces = int((m[:-1, :-1] & m[1:, :-1] & m[:-1, 1:] & m[1:, 1:]).sum())
                assert edges - int(m.sum()) + 1 - unit_faces == len(f['rooms'])
                assert all(t <= b and l <= r for t, l, b, r in f['rooms'])
                for y, x in np.argwhere(f['repairs']):
                    assert sum(0 <= yy < m.shape[0] and 0 <= xx < m.shape[1] and m[yy, xx]
                               for yy, xx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))) == 2, 'only straight wall pixels may be erased'
            features[split].append(_features(a))
    train = features['train']
    for key in ('clean', 'partial', 'full_vs_any', 'omit_B', 'opacity_matters', 'repair_matters', 'lattice_matters', 'sorting_matters', 'tie'):
        assert any(f[key] for f in train), f'missing training evidence: {key}'
    # Every test has at least one of the explicitly supported structural shifts.
    for f in features['test']:
        assert (f['max_rooms'] > max(t['max_rooms'] for t in train)
                or f['rotation'] not in {t['rotation'] for t in train}
                or f['repeated'] and not any(t['repeated'] for t in train)), 'missing train/test shift'


def self_test():
    """Bounded population audit, counterfactuals, symmetry and corruption checks."""
    # Hand-written pixels and answers, without the scene/partition constructors:
    # a three-room T partition and a one-room receiver with conflicting spatial
    # orders. A single shadow kills both a tiny room and a large spanning room.
    fixture = np.zeros((13, 23), dtype=np.int8)
    fixture[5:8, 0] = 1
    for x, (up, down) in enumerate(zip((5, 5, 4, 4, 3, 3, 2),
                                      (7, 7, 8, 8, 9, 9, 10)), 1):
        fixture[up, x] = fixture[down, x] = 1
    fixture[6, 1] = 2
    fixture[4, 10:15] = fixture[8, 10:15] = 3
    fixture[4:9, 10] = fixture[4:9, 14] = 3
    fixture[4:9, 12] = 3
    fixture[6, 10:13] = 3
    fixture[0, 17:22] = fixture[4, 17:22] = 4
    fixture[0:5, 17] = fixture[0:5, 21] = 4
    clean_answer = np.array([[4, 2, 0, 0], [3, 2, 2, 2]], dtype=np.int8)
    assert np.array_equal(solve(fixture), clean_answer)
    # The virtual thin upper boundary at (1,9) is excluded; its inner neighbor
    # is included. This checks phase without reusing profile construction.
    fa = _analyze(fixture)
    assert not fa['footprint'][1, 9] and fa['footprint'][2, 9]
    fixture[5, 10] = 5  # supported frame ink, not an opaque lane
    assert np.array_equal(solve(fixture), clean_answer)
    fixture[7, 9] = 5   # detached opaque point: rooms go from three to one
    blocked_answer = np.array([[4, 2], [3, 2]], dtype=np.int8)
    assert np.array_equal(solve(fixture), blocked_answer)
    fixture[7, 16] = 5  # repeated downstream blocker has no extra effect
    assert np.array_equal(solve(fixture), blocked_answer)
    fixture[1:4, 16] = 5
    assert np.array_equal(solve(fixture), np.array([[4, 0], [3, 2]], dtype=np.int8))
    fixture[5, 9] = 5
    assert np.array_equal(solve(fixture), np.array([[4], [3]], dtype=np.int8))
    records, signatures, topologies = [], set(), set()
    for seed in HOST_SEEDS:
        p = generate_puzzle(seed)
        signatures.add(tuple(_key(v) for s in ('train', 'test') for pair in p[s] for v in pair.values()))
        for split in ('train', 'test'):
            for pair in p[split]:
                a = _analyze(pair['input'])
                records.append((split, _features(a)))
                for f in a['frames']:
                    ys = sorted({v for t, l, b, r in f['rooms'] for v in (t - 1, b + 1)})
                    xs = sorted({v for t, l, b, r in f['rooms'] for v in (l - 1, r + 1)})
                    topologies.add(tuple(sorted((ys.index(t - 1), xs.index(l - 1), ys.index(b + 1), xs.index(r + 1))
                                                for t, l, b, r in f['rooms'])))
        if seed == 0:
            again = generate_puzzle(seed)
            assert all(np.array_equal(x[k], y[k]) for s in p for x, y in zip(p[s], again[s]) for k in x)
            # Recoloring commutes with solving, including nonzero backgrounds.
            permutation = np.array([9, 7, 5, 3, 1, 8, 6, 4, 2, 0], dtype=np.int8)
            for pair in p['train'] + p['test']:
                assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
                for turns in (1, 2, 3):
                    rotated = np.rot90(pair['input'], turns).copy()
                    ra = _analyze(rotated)
                    assert sorted(ra['counts']) == sorted(_analyze(pair['input'])['counts'])
                    assert np.array_equal(solve(rotated), _report(ra['background'], ra['payload'], ra['frames'], ra['counts'], ra['rotation']))
    fs = [f for _, f in records]
    assert len(signatures) == 42
    assert len({f['shape'] for f in fs}) >= 30
    assert len(topologies) >= 30
    assert len({f['palette'] for f in fs}) >= 60
    assert {f['rotation'] for f in fs} == set(range(4))
    assert {f['frames'] for f in fs} == {1, 2, 3, 4}
    assert any(0 in f['counts'] for f in fs)
    assert any(f['max_rooms'] >= 8 for f in fs)
    assert any(f['repaired'] == 0 for f in fs) and any(f['repaired'] >= 3 for f in fs)
    for key in ('partial', 'full_vs_any', 'omit_B', 'repair_matters', 'opacity_matters', 'lattice_matters', 'repeated'):
        assert sum(f[key] for f in fs) >= 5, key
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(991 + nt * 7 + ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal count accepted')
    damaged = generate_puzzle(77, 2, 1)
    damaged['test'][0]['output'][0, 0] = int((damaged['test'][0]['output'][0, 0] + 1) % 10)
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted report accepted')
    return dict(seeds=42, pairs=len(records), shapes=len({f['shape'] for f in fs}),
                topologies=len(topologies), palettes=len({f['palette'] for f in fs}))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
