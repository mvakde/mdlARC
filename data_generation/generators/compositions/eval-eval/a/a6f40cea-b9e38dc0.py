"""Light through reconstructed, transparent, periodically colored rectangle loops.

Only the standard library and NumPy are used. The input reader is separate from
the scene sampler and the coordinate-wise construction of labels.
"""
from __future__ import annotations

import copy
import hashlib
import json
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
    if (a.ndim != 2 or not all(1 <= d <= 30 for d in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("grids must be integer rectangles, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _perimeter(box):
    t, b, l, r = box
    return ({(t, x) for x in range(l, r + 1)} | {(b, x) for x in range(l, r + 1)}
            | {(y, l) for y in range(t + 1, b)} | {(y, r) for y in range(t + 1, b)})


def _bbox(points):
    ys, xs = zip(*points)
    return min(ys), max(ys), min(xs), max(xs)


def _mask(shape, box, strict=False):
    a = np.zeros(shape, dtype=bool)
    t, b, l, r = box
    d = int(strict)
    a[t + d:b + 1 - d, l + d:r + 1 - d] = True
    return a


def _components(points):
    remaining, result = set(points), []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        pending, group = [start], {start}
        while pending:
            y, x = pending.pop()
            for q in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if q in remaining:
                    remaining.remove(q)
                    pending.append(q)
                    group.add(q)
        result.append(group)
    return result


def _halo(points):
    return {q for y, x in points for q in
            ((y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1))}


def _screen(a):
    bg = int(a[0, 0])
    if any(int(a[p]) != bg for p in ((0, -1), (-1, 0), (-1, -1))):
        raise ValueError("background corners must agree")
    candidates = []
    for color in np.unique(a):
        color = int(color)
        if color == bg:
            continue
        points = set(map(tuple, np.argwhere(a == color)))
        if not points:
            continue
        box = _bbox(points)
        t, b, l, r = box
        if b - t < 5 or r - l < 5 or t == 0 or l == 0 or b == a.shape[0]-1 or r == a.shape[1]-1:
            continue
        if points != _perimeter(box):
            continue
        inside = a[t + 1:b, l + 1:r]
        counts = np.bincount(inside.ravel(), minlength=10)
        fill = int(np.argmax(counts))
        if fill == color or counts[fill] <= inside.size * 0.75:
            continue
        blockers = set(map(int, np.unique(inside))) - {fill}
        if blockers & {color, bg}:
            continue
        candidates.append((box, color, fill, blockers, bg))
    if len(candidates) != 1:
        raise ValueError("one uniquely readable screen is required")
    return candidates[0]


def _fit_flank(values, sign):
    """Fit the longest terminal suffix with at least two outward steps."""
    ys = np.asarray(values, dtype=int)
    xs = np.arange(len(ys)) - len(ys) + 1
    candidates = []
    for period in (1, 2, 3):
        for phase in range(period):
            predicted = ys[-1] + sign * ((xs + phase) // period)
            length = 0
            for same in (predicted == ys)[::-1]:
                if not same:
                    break
                length += 1
            if length >= 2 * period + 1 and len(set(ys[-length:])) >= 3:
                candidates.append((length, -period, phase))
    if not candidates:
        raise ValueError("insufficient terminal cadence evidence")
    length, negative_period, phase = max(candidates)
    return -negative_period, phase, int(ys[-1]), length


def _emitter(inp):
    matches = []
    for rotation in range(4):
        a = np.rot90(inp, -rotation).copy()
        box, frame, fill, blockers, bg = _screen(a)
        t, b, l, r = box
        for color in np.unique(a):
            wall = int(color)
            if wall in {bg, fill, frame} | blockers:
                continue
            positions = np.argwhere(a == wall)
            if len(positions) < 9:
                continue
            rear, mouth = int(positions[:, 1].min()), int(positions[:, 1].max())
            if mouth >= l - 1 or mouth - rear < 5:
                continue
            cap = np.flatnonzero(a[:, rear] == wall)
            if len(cap) < 3 or np.any(np.diff(cap) != 1):
                continue
            upper, lower = [], []
            for x in range(rear + 1, mouth + 1):
                rows = np.flatnonzero(a[:, x] == wall)
                if len(rows) != 2 or rows[1] - rows[0] < 2:
                    break
                upper.append(int(rows[0]))
                lower.append(int(rows[1]))
            else:
                if (upper[0] != cap[0] or lower[0] != cap[-1]
                        or np.any(np.abs(np.diff(upper)) > 1)
                        or np.any(np.abs(np.diff(lower)) > 1)):
                    continue
                try:
                    fits = (_fit_flank(upper, -1), _fit_flank(lower, 1))
                except ValueError:
                    continue
                footprint = np.zeros(a.shape, dtype=bool)
                for x in range(rear + 1, a.shape[1]):
                    if x <= mouth:
                        lo, hi = upper[x - rear - 1], lower[x - rear - 1]
                    else:
                        lo = fits[0][2] - (x - mouth + fits[0][1]) // fits[0][0]
                        hi = fits[1][2] + (x - mouth + fits[1][1]) // fits[1][0]
                    footprint[:, x] = (np.arange(a.shape[0]) > lo) & (np.arange(a.shape[0]) < hi)
                chamber = footprint.copy()
                chamber[:, mouth + 1:] = False
                patch_points = np.argwhere(chamber & (a != bg) & (a != wall))
                patch_colors = {int(a[tuple(p)]) for p in patch_points}
                if len(patch_colors) != 1:
                    continue
                payload = patch_colors.pop()
                if payload in {bg, fill, frame, wall} | blockers:
                    continue
                patch = set(map(tuple, np.argwhere(a == payload)))
                if (not patch or any(not chamber[p] for p in patch)
                        or len(_components(patch)) != 1):
                    continue
                matches.append(dict(a=a, rotation=rotation, box=box, frame=frame,
                                    fill=fill, blockers=blockers, bg=bg, wall=wall,
                                    payload=payload, rear=rear, mouth=mouth, fits=fits,
                                    footprint=footprint, chamber=chamber,
                                    profile=(tuple(upper), tuple(lower)), patch=patch))
    if len(matches) != 1:
        raise ValueError("input must identify one emitter direction and payload")
    return matches[0]


def _periodic_completion(a, box, visible, hidden):
    predictions, periods = set(), []
    for period in range(1, 5):
        word, counts = {}, [0] * period
        for y, x in sorted(visible):
            residue, color = (y + x) % period, int(a[y, x])
            if residue in word and word[residue] != color:
                break
            word[residue] = color
            counts[residue] += 1
        else:
            if min(counts) >= 2:
                predictions.add(tuple(word[(y + x) % period] for y, x in sorted(hidden)))
                periods.append(period)
    if len(predictions) != 1:
        return None
    assignments = {p: int(a[p]) for p in visible}
    assignments.update(zip(sorted(hidden), next(iter(predictions))))
    return assignments, min(periods)


def _read_input(inp):
    scene = _emitter(inp)
    a, box = scene['a'], scene['box']
    closed = _mask(a.shape, box)
    excluded = {scene[k] for k in ('bg', 'wall', 'payload', 'frame', 'fill')} | scene['blockers']
    visible = {tuple(map(int, p)) for p in np.argwhere(~closed & ~np.isin(a, list(excluded)))}
    parts = _components(visible)
    candidates = []
    for i, first in enumerate(parts):
        for j in range(i, len(parts)):
            points = first if i == j else first | parts[j]
            rect = _bbox(points)
            t, b, l, r = rect
            if b - t < 2 or r - l < 2:
                continue
            full = _perimeter(rect)
            if {p for p in full if not closed[p]} != points:
                continue
            hidden = {p for p in full if closed[p]}
            fitted = _periodic_completion(a, rect, points, hidden)
            if fitted is None:
                continue
            assignments, period = fitted
            candidates.append(dict(parts=frozenset((i, j)), box=rect,
                                   points=full, assignments=assignments, period=period))
    covers = []

    def cover(remaining, chosen, occupied):
        if len(covers) > 1:
            return
        if not remaining:
            covers.append(chosen)
            return
        first = min(remaining)
        for c in candidates:
            if first in c['parts'] and c['parts'] <= remaining and not (c['points'] & _halo(occupied)):
                cover(remaining - c['parts'], chosen + [c], occupied | c['points'])

    cover(set(range(len(parts))), [], set())
    if len(covers) != 1 or not covers[0]:
        raise ValueError("rectangle completion must be unique")
    loops = covers[0]
    glass = np.full(a.shape, -1, dtype=np.int8)
    for loop in loops:
        for p, color in loop['assignments'].items():
            glass[p] = color
    opaque = np.isin(a, list(scene['blockers']))
    if np.any((glass >= 0) & opaque):
        raise ValueError("opaque objects cannot overwrite glass evidence")
    if np.any((glass >= 0) & scene['chamber']):
        raise ValueError("filters must be downstream of the chamber")
    scene.update(loops=loops, glass=glass, opaque=opaque)
    return scene


def _scan(scene, glass=None, opaque=None, first_contact=False):
    """Input-only lane transport; no construction descriptors are accepted."""
    a, field = scene['a'], scene['footprint']
    glass = scene['glass'] if glass is None else glass
    opaque = scene['opaque'] if opaque is None else opaque
    out = a.copy()
    t, b, l, r = scene['box']
    out[t + 1:b, l + 1:r] = scene['fill']
    for y, x in np.argwhere(glass >= 0):
        out[y, x] = glass[y, x]
    out[opaque] = a[opaque]
    for y in range(a.shape[0]):
        tint, blocked, contacted = scene['payload'], False, False
        for x in range(scene['rear'] + 1, a.shape[1]):
            if not field[y, x]:
                continue
            if opaque[y, x]:
                blocked = True
            elif glass[y, x] >= 0:
                if not first_contact or not contacted:
                    tint = int(glass[y, x])
                    contacted = True
            elif not blocked and t < y < b and l < x < r:
                out[y, x] = tint
    cropped = out[t + 1:b, l + 1:r]
    return np.rot90(cropped, scene['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive every role, hidden contour, beam bound and shadow from the input."""
    return _scan(_read_input(_grid(input_grid)))


def _oracle(a, box, fill, payload, field, glass, opaque):
    """Construction truth: last upstream contact and first opaque intersection.

    This is coordinate-wise rather than the reader's persistent-state lane scan.
    It receives the complete sampled layer, never a predicted completion.
    """
    t, b, l, r = box
    out = np.full((b - t - 1, r - l - 1), fill, dtype=np.int8)
    for y in range(t + 1, b):
        contacts = np.flatnonzero((glass[y] >= 0) & field[y])
        stops = np.flatnonzero(opaque[y] & field[y])
        stop = int(stops[0]) if len(stops) else a.shape[1]
        for x in range(l + 1, r):
            q = y - t - 1, x - l - 1
            if opaque[y, x]:
                out[q] = a[y, x]
            elif glass[y, x] >= 0:
                out[q] = glass[y, x]
            elif field[y, x] and x < stop:
                prior = contacts[contacts < x]
                out[q] = glass[y, int(prior[-1])] if len(prior) else payload
    return out


def _geometry(rng, kind, count, clean=False):
    if kind == 'v':
        ow = int(rng.integers(4 * count + 3, min(16, 4 * count + 7)))
        oh = int(rng.integers(12, 18) if clean else rng.integers(6, 13))
        top_margin, bottom_margin = (int(rng.integers(2 * count + 1, 2 * count + 4)) for _ in range(2))
        gap, right_margin = int(rng.integers(3, 6)), int(rng.integers(1, 4))
    elif kind == 'h':
        oh = int(rng.integers(4 * count + 3, min(17, 4 * count + 7)))
        ow = int(rng.integers(6, 11))
        top_margin, bottom_margin = (int(rng.integers(2, 6)) for _ in range(2))
        gap, right_margin = (int(rng.integers(2 * count + 1, 2 * count + 4)) for _ in range(2))
    else:
        oh, ow = (int(rng.integers(10, 15)) for _ in range(2))
        top_margin, bottom_margin = (int(rng.integers(3, 7)) for _ in range(2))
        gap, right_margin = int(rng.integers(3, 6)), int(rng.integers(3, 6))
    height = top_margin + oh + 2 + bottom_margin
    max_mouth = 28 - gap - ow - right_margin
    if height > 30 or max_mouth < 5:
        return None
    rear = int(rng.integers(0, min(3, max_mouth - 5) + 1))
    rear_length = int(rng.integers(2, min(5, max_mouth - rear - 3) + 1))
    max_run = min(3, (max_mouth - rear - rear_length - 1) // 2)
    runs = [int(rng.integers(1, max_run + 1)) for _ in range(2)]
    phases = [int(rng.integers(run)) for run in runs]
    flare_length = 2 * max(runs) + 1
    depths = []
    for run, phase in zip(runs, phases):
        start, neck = int(rng.integers(1, 4)), int(rng.integers(1, 3))
        peak = min(4, start + 1, neck + rear_length - 2)
        profile = np.rint(np.interp(np.arange(rear_length),
                                   [0, max(1, rear_length // 2), rear_length - 1],
                                   [start, peak, neck])).astype(int)
        # Interpolation with two sections may have repeated knot coordinates;
        # explicitly retain the end values and ensure thin connected rails.
        profile[0], profile[-1] = start, neck
        if np.any(np.abs(np.diff(profile)) > 1):
            return None
        extension = [neck + (j + phase) // run for j in range(1, flare_length + 1)]
        depths.append(np.array(list(profile) + extension, dtype=int))
    mouth = rear + rear_length + flare_length
    left = mouth + gap
    box = top_margin, top_margin + oh + 1, left, left + ow + 1
    width = box[3] + right_margin + 1
    low_center, high_center = int(depths[0].max()) + 1, height - int(depths[1].max()) - 2
    if low_center > high_center:
        return None
    center = int(np.clip((box[0] + box[1]) // 2 + int(rng.integers(-3, 4)), low_center, high_center))
    wall, field = np.zeros((height, width), bool), np.zeros((height, width), bool)
    wall[center - depths[0][0]:center + depths[1][0] + 1, rear] = True
    for j, (up, down) in enumerate(zip(*depths), 1):
        x = rear + j
        wall[center - up, x] = wall[center + down, x] = True
        field[center - up + 1:center + down, x] = True
    for x in range(mouth + 1, width):
        elapsed = flare_length + x - mouth
        lo = center - int(depths[0][rear_length - 1]) - (elapsed + phases[0]) // runs[0]
        hi = center + int(depths[1][rear_length - 1]) + (elapsed + phases[1]) // runs[1]
        field[:, x] = (np.arange(height) > lo) & (np.arange(height) < hi)
    return dict(box=box, wall=wall, field=field, rear=rear, mouth=mouth,
                rear_length=rear_length, kind=kind, count=count)


def _boxes(rng, geometry, clean):
    t, b, l, r = geometry['box']
    height, width = geometry['field'].shape
    count, kind, mouth = geometry['count'], geometry['kind'], geometry['mouth']
    boxes = []
    if kind == 'v':
        left_shift, right_shift = int(rng.integers(2)), int(rng.integers(2))
        for i in range(count):
            top = t - 2 * (count - i) - (int(rng.integers(2)) if i == 0 else 0)
            bottom = b + 2 * (count - i) + (int(rng.integers(2)) if i == 0 else 0)
            xl, xr = l + 2 + 2*i + left_shift, r - 2 - 2*i - right_shift
            if clean and (xr - xl) % 2 == 0:
                xr -= 1
            boxes.append((top, bottom, xl, xr))
    elif kind == 'h':
        up_shift, down_shift = int(rng.integers(2)), int(rng.integers(2))
        for i in range(count):
            xl = l - 2 * (count - i) - (int(rng.integers(2)) if i == 0 else 0)
            xr = r + 2 * (count - i) + (int(rng.integers(2)) if i == 0 else 0)
            boxes.append((t + 2 + 2*i + up_shift, b - 2 - 2*i - down_shift, xl, xr))
    else:
        my, mx = (t + b) // 2, (l + r) // 2
        corners = list(rng.choice(4, count, replace=False))
        for corner in corners:
            north, west = corner < 2, corner % 2 == 0
            yt = int(rng.integers(1, t - 1)) if north else int(rng.integers(my + 1, b - 1))
            yb = int(rng.integers(t + 2, my)) if north else int(rng.integers(b + 2, height - 1))
            xl = int(rng.integers(mouth + 1, l - 1)) if west else int(rng.integers(mx + 1, r - 1))
            xr = int(rng.integers(l + 2, mx)) if west else int(rng.integers(r + 2, width - 1))
            boxes.append((yt, yb, xl, xr))
    return boxes


def _patch(rng, candidates, size):
    available = set(map(tuple, np.argwhere(candidates)))
    if not available:
        return []
    ordered = sorted(available)
    points = {ordered[int(rng.integers(len(ordered)))]}
    for _ in range(size - 1):
        frontier = sorted(_halo(points) & available - points)
        if not frontier:
            break
        points.add(frontier[int(rng.integers(len(frontier)))])
    return sorted(points)


def _place_blocks(rng, a, geometry, glass, colors, style):
    inside = _mask(a.shape, geometry['box'], True)
    field = geometry['field']
    opaque = np.zeros(a.shape, dtype=bool)
    if style in ('clean', 'clear'):
        return opaque
    t, b, l, r = geometry['box']
    groups = 2 if style == 'teaching' else int(rng.integers(1, 4))
    used_rows = set()
    for i in range(groups):
        offsets = ((0, 0),) if style == 'teaching' else (
            ((0, 0),), ((0, 0), (1, 0)), ((0, 0), (0, 1), (1, 0), (1, 1)),
            ((0, 0), (1, 0), (1, 1)), ((0, 0), (0, 2)))[int(rng.integers(5))]
        choices = []
        for y in range(t + 1, b):
            for x in range(l + 2, r - 2):
                points = [(y + dy, x + dx) for dy, dx in offsets]
                if any(yy >= b or xx >= r or not inside[yy, xx] or glass[yy, xx] >= 0 or opaque[yy, xx] for yy, xx in points):
                    continue
                if not any(field[p] for p in points):
                    continue
                if style == 'teaching' and y in used_rows:
                    continue
                if not any(np.any(field[yy, xx + 1:r] & (glass[yy, xx + 1:r] < 0)) for yy, xx in points if field[yy, xx]):
                    continue
                if style == 'teaching' and not np.any(field[y, x + 1:r] & (glass[y, x + 1:r] >= 0)):
                    continue
                choices.append(points)
        if not choices:
            return None
        points = choices[int(rng.integers(len(choices)))]
        color = colors[i % len(colors)]
        for p in points:
            a[p] = color
            opaque[p] = True
            used_rows.add(p[0])
    # Same opaque color can occur on an inert exterior object.
    if rng.random() < 0.65:
        choices = np.argwhere(~field & ~_mask(a.shape, geometry['box']) & (glass < 0)
                              & ~geometry['wall'] & (a == int(a[0, -1])))
        choices = [tuple(map(int, p)) for p in choices if 0 < p[0] < a.shape[0]-1 and 0 < p[1] < a.shape[1]-1]
        if choices:
            p = choices[int(rng.integers(len(choices)))]
            a[p] = colors[0]
            opaque[p] = True
    return opaque


def _make_pair(rng, kind, count, style, rotation, palette, max_period, forced_period=None):
    g = _geometry(rng, kind, count, clean=style == 'clean')
    if g is None:
        return None
    bg, fill, frame, wall, payload = map(int, palette[:5])
    if style == 'clean':
        fill = bg
    a = np.full(g['field'].shape, bg, dtype=np.int8)
    a[g['wall']] = wall
    candidate_patch = g['field'].copy()
    candidate_patch[:, g['rear'] + g['rear_length'] + 1:] = False
    for p in _patch(rng, candidate_patch, int(rng.integers(1, 6))):
        a[p] = payload
    glass = np.full(a.shape, -1, dtype=np.int8)
    occupied = set()
    screen_mask = _mask(a.shape, g['box'])
    boxes = _boxes(rng, g, style == 'clean')
    for i, box in enumerate(boxes):
        yt, yb, xl, xr = box
        if yt < 1 or xl < 1 or yb >= a.shape[0]-1 or xr >= a.shape[1]-1 or yb - yt < 2 or xr - xl < 2:
            return None
        full = _perimeter(box)
        if full & _halo(occupied) or any(g['wall'][p] for p in full):
            return None
        exposed = {p for p in full if not screen_mask[p]}
        if not exposed or _bbox(exposed) != box or not any(screen_mask[p] for p in full):
            return None
        period = forced_period if i == 0 and forced_period is not None else int(rng.integers(1, max_period + 1))
        filters = [int(c) for c in rng.permutation(palette[5:8])]
        if period == 1:
            word = [filters[0]]
        elif period == 2:
            word = filters[:2]
        elif period == 3:
            word = filters[:3] if rng.random() < 0.5 else [filters[0], filters[0], filters[1]]
        else:
            word = [filters[0], filters[1], filters[0], filters[2]]
        phase = int(rng.integers(period))
        for y, x in full:
            color = word[(y + x + phase) % period]
            glass[y, x] = color
            if not screen_mask[y, x]:
                a[y, x] = color
        occupied.update(full)
    t, b, l, r = g['box']
    a[t + 1:b, l + 1:r] = fill
    for p in _perimeter(g['box']):
        a[p] = frame
    block_colors = list(map(int, palette[8:10]))
    opaque = _place_blocks(rng, a, g, glass, block_colors, style)
    if opaque is None:
        return None
    output = _oracle(a, g['box'], fill, payload, g['field'], glass, opaque)
    pair = {'input': np.rot90(a, rotation).copy(), 'output': np.rot90(output, rotation).copy()}
    try:
        scene = _read_input(pair['input'])
        if len(scene['loops']) != count or not np.array_equal(_scan(scene), pair['output']):
            return None
        f = _features(scene)
        if f['tinted_blank'] < 3 or f['payload_blank'] < 2 or not f['a_matters'] or not f['b_matters']:
            return None
        if style == 'clean' and (not f['last_contact_matters'] or f['periods'] != (2,)
                                 or f['unlit_blank'] < 3 or f['hidden_tint_blank'] < 3):
            return None
        if style == 'teaching' and (f['shadow_blank'] < 3 or f['blocker_colors'] < 2 or f['unlit_glass'] < 1):
            return None
        if forced_period is not None and forced_period not in f['periods']:
            return None
    except (ValueError, AssertionError):
        return None
    return pair


def _key(a):
    a = np.asarray(a)
    return tuple(a.shape), a.tobytes()


def _puzzle_digest(p):
    value = {s: [{k: np.asarray(v).tolist() for k, v in pair.items()} for pair in p[s]] for s in ('train', 'test')}
    return hashlib.sha256(json.dumps(value, separators=(',', ':')).encode()).hexdigest()


def _features(scene):
    a, glass, field = scene['a'], scene['glass'], scene['footprint']
    t, b, l, r = scene['box']
    section = np.s_[t+1:b, l+1:r]
    inside = _mask(a.shape, scene['box'], True)
    blank = (glass < 0) & ~scene['opaque'] & inside
    out = np.rot90(_scan(scene), -scene['rotation'])
    unblocked = np.rot90(_scan(scene, opaque=np.zeros(a.shape, bool)), -scene['rotation'])
    shadow = np.zeros(a.shape, bool)
    for y, x in np.argwhere(scene['opaque'] & field):
        shadow[y, x+1:] = True
    visible_only = glass.copy()
    visible_only[_mask(a.shape, scene['box'])] = -1
    no_a = _scan(scene, glass=visible_only)
    no_a_canonical = np.rot90(no_a, -scene['rotation'])
    static = a[section].copy()
    static[glass[section] >= 0] = glass[section][glass[section] >= 0]
    loops = tuple(sorted(loop['box'] for loop in scene['loops']))
    profile = scene['profile']
    profile_relative = tuple(tuple(int(v - ys[0]) for v in ys) for ys in profile)
    return dict(rotation=scene['rotation'], shape=tuple(a.shape), output_shape=tuple(out.shape),
                loop_count=len(loops), periods=tuple(sorted(loop['period'] for loop in scene['loops'])),
                layout=tuple((yt-t, yb-t, xl-l, xr-l) for yt, yb, xl, xr in loops),
                profile=profile_relative, runs=tuple(fit[0] for fit in scene['fits']),
                phases=tuple(fit[1] for fit in scene['fits']),
                patch_shape=tuple(sorted((y-min(p[0] for p in scene['patch']), x-min(p[1] for p in scene['patch'])) for y,x in scene['patch'])),
                palette=tuple(scene[k] for k in ('bg', 'fill', 'frame', 'wall', 'payload')),
                fill_matches=scene['bg'] == scene['fill'],
                blocker_colors=len(scene['blockers']), opaque_cells=int(scene['opaque'][inside].sum()),
                tinted_blank=int(np.sum(blank[section] & (out != scene['fill']) & (out != scene['payload']))),
                payload_blank=int(np.sum(blank[section] & (out == scene['payload']))),
                shadow_blank=int(np.sum(blank[section] & (out != unblocked))),
                unlit_blank=int(np.sum(blank & ~field)),
                hidden_tint_blank=int(np.sum(blank[section] & (out != no_a_canonical))),
                unlit_glass=int(np.sum(inside & shadow & (glass >= 0) & field)),
                clipped=bool(np.any(field[0]) or np.any(field[-1])),
                unequal_runs=scene['fits'][0][0] != scene['fits'][1][0],
                grouped=any(len(c)>1 for c in _components(set(map(tuple, np.argwhere(scene['opaque'] & inside))))),
                repeated_lane=bool(np.any((scene['opaque'] & inside).sum(axis=1) > 1)),
                inert=bool(np.any(scene['opaque'] & ~field)),
                a_matters=not np.array_equal(_scan(scene), no_a),
                b_matters=not np.array_equal(out, static),
                last_contact_matters=not np.array_equal(_scan(scene), _scan(scene, first_contact=True)))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    num_train = _integer(num_train, 'num_train', 2, 10)
    num_test = _integer(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    shared = bool(rng.random() < 0.2)
    shared_palette = rng.permutation(10)
    rotations = list(map(int, rng.permutation(4)))
    max_period = 3 if num_train == 2 else 4
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', num_train), ('test', num_test)):
        for i in range(n):
            if split == 'train' and i == 0:
                kind, count, style, forced = 'v', 1, 'clean', 2
            elif split == 'train' and i == 1:
                kind, count, style, forced = 'v', 2, 'teaching', 3
            elif split == 'train' and i == 2:
                kind, count, style, forced = 'c', 2, 'blocked', 4
            elif split == 'train' and i == 3:
                kind, count, style, forced = 'h', 1, 'blocked', None
            elif split == 'train':
                kind, count = str(rng.choice(['v', 'h', 'c'])), int(rng.integers(1, 3))
                if kind == 'c':
                    count = 2
                style, forced = str(rng.choice(['clear', 'blocked'])), None
            else:
                kind = 'v' if num_train == 2 else str(rng.choice(['v', 'h', 'c']))
                count, style, forced = 3, str(rng.choice(['clear', 'blocked'], p=[0.25, 0.75])), None
            palette = shared_palette if shared else rng.permutation(10)
            rotation = rotations[i % 4] if split == 'train' else int(rng.integers(4))
            for attempt in range(300):
                pair = _make_pair(rng, kind, count, style, rotation, palette, max_period, forced)
                if pair is None or _key(pair['input']) in seen:
                    continue
                puzzle[split].append(pair)
                seen.add(_key(pair['input']))
                break
            else:
                raise RuntimeError(f"bounded sampling exhausted for {kind}/{count}/{style}")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            scene = _read_input(inp)
            expected = _scan(scene)
            assert np.array_equal(expected, out), 'incorrect reconstructed light crop'
            t, b, l, r = scene['box']
            canonical_out = np.rot90(out, -scene['rotation'])
            glass = scene['glass'][t+1:b, l+1:r]
            opaque = scene['opaque'][t+1:b, l+1:r]
            assert np.array_equal(canonical_out[glass >= 0], glass[glass >= 0]), 'glass must be restored even in shadow'
            assert np.array_equal(canonical_out[opaque], scene['a'][t+1:b, l+1:r][opaque]), 'opaque cells must survive'
            # Coordinate-wise check uses recovered input layers, independent of
            # the stateful transport loop. No scene-generation data is retained.
            direct = _oracle(scene['a'], scene['box'], scene['fill'], scene['payload'],
                             scene['footprint'], scene['glass'], scene['opaque'])
            assert np.array_equal(direct, canonical_out)
            f = _features(scene)
            assert f['a_matters'] and f['b_matters'] and f['tinted_blank'] >= 3 and f['payload_blank'] >= 2
            features[split].append(f)
    train = features['train']
    assert any(f['loop_count'] == 1 and f['fill_matches'] and f['opaque_cells'] == 0
               and f['last_contact_matters'] and f['periods'] == (2,)
               and f['unlit_blank'] >= 3 and f['hidden_tint_blank'] >= 3 for f in train)
    assert any(f['loop_count'] == 2 and f['shadow_blank'] >= 3 and f['blocker_colors'] == 2
               and f['unlit_glass'] > 0 and 3 in f['periods'] for f in train)
    if nt >= 3:
        assert any(4 in f['periods'] for f in train)
    assert all(f['loop_count'] == 3 for f in features['test'])
    assert all(max(f['periods']) <= (3 if nt == 2 else 4) for fs in features.values() for f in fs)


def self_test() -> dict:
    """Bounded, reproducible variation and family-contract checks over 42 seeds."""
    # Hand-written inputs and expected rows, without sampling or label helpers:
    # glass transmits; exit contact replaces tint; glass in shadow stays visible;
    # that contact cannot restart light after an opaque stop.
    fixture = np.zeros((19, 23), dtype=np.int8)
    fixture[7:12, 0] = 2
    for x, (lo, hi) in enumerate(zip((7,7,6,5,4), (11,11,12,13,14)), 1):
        fixture[lo, x] = fixture[hi, x] = 2
    fixture[9, 1] = 3
    for y in range(1, 18):
        for x in range(13, 19):
            if y in (1, 17) or x in (13, 18):
                if not (3 <= y <= 15 and 10 <= x <= 21):
                    fixture[y, x] = 4 + (y + x) % 2
    fixture[3, 10:22] = fixture[15, 10:22] = 1
    fixture[3:16, 10] = fixture[3:16, 21] = 1
    fixture[9, 15], fixture[10, 16] = 6, 7
    answer = solve(fixture)
    assert answer[4].tolist() == [3,3,5,5,5,5,5,4,4,4]
    assert answer[5].tolist() == [3,3,4,4,6,0,0,5,0,0]
    assert answer[6].tolist() == [3,3,5,5,5,7,0,4,0,0]
    # A separate period-two flank fixture checks the exact next thin boundary.
    fixture = np.zeros((21, 24), dtype=np.int8)
    fixture[8:13, 0] = 2
    for x, (lo, hi) in enumerate(zip((8,8,8,7,7,6,6), (12,12,12,13,13,14,14)), 1):
        fixture[lo, x] = fixture[hi, x] = 2
    fixture[10, 1] = 3
    for y in range(1, 20):
        for x in range(15, 21):
            if (y in (1, 19) or x in (15, 20)) and not (2 <= y <= 18 and 11 <= x <= 22):
                fixture[y, x] = 4 + (y + x) % 2
    fixture[2, 11:23] = fixture[18, 11:23] = 1
    fixture[2:19, 11] = fixture[2:19, 22] = 1
    answer = solve(fixture)
    assert answer[0, :3].tolist() == [0,0,3]
    assert answer[1, 0] == 3
    records, digests = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        digests.add(_puzzle_digest(puzzle))
        if seed in (0, 13, 41):
            assert _puzzle_digest(puzzle) == _puzzle_digest(generate_puzzle(seed))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                original = pair['input'].copy()
                assert np.array_equal(solve(original), pair['output'])
                assert np.array_equal(original, pair['input']), 'solve mutated its input'
                scene = _read_input(original)
                records.append((split, _features(scene)))
                # Rotation equivariance is a property of the whole rule,
                # including the oriented coordinate phase and output crop.
                if seed < 3:
                    for turns in (1, 2, 3):
                        assert np.array_equal(solve(np.rot90(original, turns)), np.rot90(pair['output'], turns))
    fs = [f for _, f in records]
    summary = dict(seeds=42, pairs=len(fs), distinct_puzzles=len(digests),
                   input_shapes=len({f['shape'] for f in fs}), output_shapes=len({f['output_shape'] for f in fs}),
                   contour_layouts=len({f['layout'] for f in fs}), chamber_profiles=len({f['profile'] for f in fs}),
                   payload_shapes=len({f['patch_shape'] for f in fs}),
                   grouped_pairs=sum(f['grouped'] for f in fs), repeated_lane_pairs=sum(f['repeated_lane'] for f in fs),
                   unequal_cadence_pairs=sum(f['unequal_runs'] for f in fs), clipped_pairs=sum(f['clipped'] for f in fs))
    assert len(digests) == 42
    assert summary['input_shapes'] >= 40 and summary['output_shapes'] >= 25
    assert summary['contour_layouts'] >= 90 and summary['chamber_profiles'] >= 50
    assert summary['payload_shapes'] >= 8
    assert {f['rotation'] for f in fs} == {0, 1, 2, 3}
    assert {p for f in fs for p in f['periods']} == {1, 2, 3, 4}
    assert {f['loop_count'] for f in fs} == {1, 2, 3}
    assert {p for f in fs for p in f['runs']} == {1, 2, 3}
    assert summary['grouped_pairs'] >= 10 and summary['repeated_lane_pairs'] >= 10
    assert summary['unequal_cadence_pairs'] >= 5 and summary['clipped_pairs'] >= 20
    assert all({f['palette'][i] for f in fs} == set(range(10)) for i in (0, 2, 3, 4))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(10000 + nt * 10 + ne, nt, ne)
            assert len(p['train']) == nt and len(p['test']) == ne
    for nt, ne in ((1,1), (11,1), (4,0), (6,5), (2,2), (3,4), (True,1), (4,1.0), (None,1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = copy.deepcopy(generate_puzzle(101, 2, 1))
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return summary


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
