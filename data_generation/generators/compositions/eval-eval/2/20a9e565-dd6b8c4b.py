"""Growing gated reservoirs: reconstruct geometry that controls routing and capacity.

Generation paints explicit latent rectangles; solve fits the visible input only.
The two paths use separate wavefront/BFS distance implementations.
"""
from collections import deque
import numpy as np

_STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'Invalid grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'Grid must have integer cells')
    _require(np.all((0 <= a) & (a <= 9)), 'Invalid colors')
    return a


def _corners(bb):
    r0, r1, c0, c1 = bb
    return {(r0, c0), (r0, c0+1), (r0+1, c0),
            (r1, c1), (r1, c1-1), (r1-1, c1)}


def _annotations(a):
    candidates = []
    for raw in np.unique(a):
        points = np.argwhere(a == raw)
        if len(points) != 6:
            continue
        r0, c0 = map(int, points.min(axis=0))
        r1, c1 = map(int, points.max(axis=0))
        bb = (r0, r1, c0, c1)
        if r1-r0 >= 2 and c1-c0 >= 2 and set(map(tuple, points)) == _corners(bb):
            candidates.append((int(raw), bb))
    _require(len(candidates) == 1, 'Need unique opposite L annotations')
    return candidates[0]


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        queue = [(r, c)]
        seen[r, c] = True
        for y, x in queue:
            for dy, dx in _STEPS:
                yy, xx = y+dy, x+dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    queue.append((yy, xx))
        yield queue


def _open_frame(points):
    """Recognize a complete visible perimeter with one noncorner cell missing."""
    ys, xs = zip(*points)
    y, x, h, w = min(ys), min(xs), max(ys)-min(ys)+1, max(xs)-min(xs)+1
    if min(h, w) < 3:
        return None
    border = {(y+r, x+c) for r in range(h) for c in range(w)
              if r in (0, h-1) or c in (0, w-1)}
    actual = set(points)
    missing = border-actual
    if not actual <= border or len(missing) != 1:
        return None
    gy, gx = missing.pop()
    if gy == y and x < gx < x+w-1:
        side, offset = 0, gx-x
    elif gx == x+w-1 and y < gy < y+h-1:
        side, offset = 1, gy-y
    elif gy == y+h-1 and x < gx < x+w-1:
        side, offset = 2, gx-x
    elif gx == x and y < gy < y+h-1:
        side, offset = 3, gy-y
    else:
        return None
    return (y, x, h, w, side, offset)


def _linear(values):
    delta = int(values[1])-int(values[0])
    if all(int(v) == int(values[0])+i*delta for i, v in enumerate(values)):
        return int(values[0]), delta
    return None


def _frame(model, i):
    n, y0, dy, x0, h0, dh, w0, dw, gap, dg, side, alt, off, do = model
    h, w = h0+i*dh, w0+i*dw
    y = y0+i*dy
    x = x0+i*(w0+gap)+i*(i-1)//2*(dw+dg)
    s, o = (side+2*(i % 2)) % 4 if alt else side, off+i*do
    _require(min(h, w) >= 3, 'Nonpositive extrapolated frame interior')
    _require(1 <= o <= (w if s in (0, 2) else h)-2, 'Door leaves its side')
    return (y, x, h, w, s, o)


def _door(f):
    y, x, h, w, side, offset = f
    return ((y, x+offset), (y+offset, x+w-1),
            (y+h-1, x+offset), (y+offset, x))[side]


def _border_mask(shape, f):
    """Solver-side set of perimeter cells, minus the fitted aperture."""
    y, x, h, w, _, _ = f
    _require(0 <= y and y+h <= shape[0] and 0 <= x and x+w <= shape[1], 'Frame leaves canvas')
    rr, cc = np.indices(shape)
    inside = (y <= rr) & (rr < y+h) & (x <= cc) & (cc < x+w)
    border = inside & ((rr == y) | (rr == y+h-1) | (cc == x) | (cc == x+w-1))
    border[_door(f)] = False
    return border


def _interior(shape, f):
    y, x, h, w, _, _ = f
    mask = np.zeros(shape, bool)
    mask[y+1:y+h-1, x+1:x+w-1] = True
    return mask


def _fit_frames(frames, bb, shape):
    """Fit in displayed coordinates; transposition is internal, never the answer."""
    for transpose in (False, True):
        if transpose:
            fs = [(x, y, w, h, {0:3, 1:2, 2:1, 3:0}[s], o) for y, x, h, w, s, o in frames]
            cb = (bb[2], bb[3], bb[0], bb[1])
        else:
            fs, cb = frames, bb
        fs = sorted(fs, key=lambda f: f[1])
        if any(fs[i][1]+fs[i][3] >= fs[i+1][1] for i in range(len(fs)-1)):
            continue
        if not (cb[2] > fs[-1][1]+fs[-1][3]-1 or cb[3] < fs[0][1]):
            continue
        values = [_linear([f[k] for f in fs]) for k in (0, 2, 3, 5)]
        gaps = [fs[i+1][1]-fs[i][1]-fs[i][3] for i in range(len(fs)-1)]
        gl = _linear(gaps)
        sides = [f[4] for f in fs]
        alt = len(set(sides)) > 1
        if any(s != (sides[0]+2*(i % 2)) % 4 if alt else s != sides[0] for i, s in enumerate(sides)):
            continue
        if any(v is None for v in values) or gl is None:
            continue
        (y0, dy), (h0, dh), (w0, dw), (off, do) = values
        model = (len(fs), y0, dy, fs[0][1], h0, dh, w0, dw, *gl, sides[0], alt, off, do)
        forward = cb[2] > fs[-1][1]
        indices = range(len(fs), len(fs)+30) if forward else range(-1, -31, -1)
        hidden = []
        try:
            for i in indices:
                # Only required objects must remain valid. A shrinking frame
                # or moving door can cease to exist AFTER the requested crop.
                # Check axial extent before validating that unseen successor.
                x = fs[0][1]+i*(w0+gl[0])+i*(i-1)//2*(dw+gl[1])
                w = w0+i*dw
                if (forward and x > cb[3]) or (not forward and x+w-1 < cb[2]):
                    break
                f = _frame(model, i)
                y, x, h, w, side, offset = f
                gap_index = i-1 if forward else i
                _require(gl[0]+gap_index*gl[1] >= 1, 'Frames meet during continuation')
                _require(cb[0] <= y and y+h-1 <= cb[1] and cb[2] <= x and x+w-1 <= cb[3],
                         'Marked window must contain complete hidden frames')
                if transpose:
                    f = (x, y, w, h, {0:3, 1:2, 2:1, 3:0}[side], offset)
                _border_mask(shape, f)
                hidden.append(f)
        except AssertionError:
            continue
        if not 1 <= len(hidden) <= 3:
            continue
        yield hidden, {'axis': int(transpose), 'forward': forward, 'visible': len(fs),
                       'dh': dh, 'dw': dw, 'drift': dy, 'gap': gl[0], 'gap_delta': gl[1],
                       'alternating': alt, 'door_delta': do}


def _decode(a):
    marker, bb = _annotations(a)
    vals, counts = np.unique(a, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, 'Background must be uniquely dominant')
    bg = int(vals[counts.argmax()])
    _require(bg != marker, 'Annotations cannot be background')
    singleton = [int(v) for v, c in zip(vals, counts) if c == 1]
    _require(len(singleton) == 1, 'Need one unique interior reservoir key')
    key = singleton[0]
    kp = tuple(map(int, np.argwhere(a == key)[0]))
    r0, r1, c0, c1 = bb
    _require(r0 < kp[0] < r1 and c0 < kp[1] < c1, 'Key must lie inside marked window')
    fits = []
    for wall in set(map(int, vals))-{bg, marker, key}:
        frames = []
        for comp in _components(a == wall):
            # Only whole components wholly outside the marked rectangle can be evidence.
            if any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in comp):
                continue
            f = _open_frame(comp)
            if f is not None:
                frames.append(f)
        if not 3 <= len(frames) <= 6:
            continue
        for hidden, info in _fit_frames(frames, bb, a.shape):
            active = [f for f in hidden if _interior(a.shape, f)[kp]]
            if len(active) != 1 or min(active[0][2:4]) < 4:
                continue
            remainder = set(map(int, vals))-{bg, marker, key, wall}
            if len(remainder) != 1:
                continue
            payload = remainder.pop()
            if np.count_nonzero(a == payload) < 2:
                continue
            predicted = np.zeros(a.shape, bool)
            for f in hidden:
                predicted |= _border_mask(a.shape, f)
            # Erased sequence walls are absent, even when other static walls remain.
            clean = a.copy()
            clean[clean == marker] = bg
            if np.any(predicted & (clean != bg)):
                continue
            interior = _interior(a.shape, active[0])
            if np.any(interior & ((clean != bg) & (clean != key))):
                continue
            clean[predicted] = wall
            info.update(background=bg, marker=marker, key=key, payload=payload, wall=wall,
                        bounds=bb, active=active[0], hidden=hidden, capacity=int(interior.sum()),
                        key_position=kp)
            fits.append((clean, interior, predicted, info))
    _require(fits, 'No supported frame progression and keyed reservoir')
    _require(len(fits) == 1, 'Ambiguous frame interpretation')
    return fits[0]


def _bfs(wall, starts):
    h, w = wall.shape
    distance = np.full(wall.shape, -1, dtype=int)
    queue = deque()
    for r, c in np.argwhere(starts):
        distance[r, c] = 0
        queue.append((int(r), int(c)))
    while queue:
        r, c = queue.popleft()
        for dr, dc in _STEPS:
            rr, cc = r+dr, c+dc
            if 0 <= rr < h and 0 <= cc < w and not wall[rr, cc] and distance[rr, cc] < 0:
                distance[rr, cc] = distance[r, c]+1
                queue.append((rr, cc))
    return distance


def _wave(wall, starts):
    distance = np.full(wall.shape, -1, dtype=int)
    seen, frontier = starts.copy(), starts.copy()
    distance[starts] = 0
    for d in range(1, wall.size+1):
        spread = np.zeros(wall.shape, bool)
        spread[1:] |= frontier[:-1]
        spread[:-1] |= frontier[1:]
        spread[:, 1:] |= frontier[:, :-1]
        spread[:, :-1] |= frontier[:, 1:]
        frontier = spread & ~wall & ~seen
        if not frontier.any():
            break
        distance[frontier] = d
        seen |= frontier
    return distance


def _select(payload, distance, capacity):
    eligible = payload & (distance >= 0)
    ds = np.sort(distance[eligible])
    if len(ds) > capacity:
        _require(ds[capacity-1] < ds[capacity], 'Ambiguous tie across capacity frontier')
    threshold = int(ds[min(capacity, len(ds))-1]) if len(ds) else -1
    return eligible & (distance <= threshold)


def _store(clean, interior, selected, bg, payload):
    out = clean.copy()
    out[selected] = bg
    slots = np.flatnonzero(interior)
    out.flat[slots[:int(selected.sum())]] = payload
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    a = _grid(input_grid)
    clean, interior, _, info = _decode(a)
    distance = _bfs(clean == info['wall'], interior)
    selected = _select(a == info['payload'], distance, info['capacity'])
    full = _store(clean, interior, selected, info['background'], info['payload'])
    r0, r1, c0, c1 = info['bounds']
    return full[r0:r1+1, c0:c1+1].copy()


def pair_features(input_grid):
    a = _grid(input_grid)
    clean, interior, predicted, info = _decode(a)
    payload = a == info['payload']
    wall = clean == info['wall']
    d = _bfs(wall, interior)
    selected = _select(payload, d, info['capacity'])
    retained = payload & (d >= 0) & ~selected
    missing_ignored = _bfs(wall & ~predicted, interior)
    y, x, h, w, _, _ = info['active']
    rr, cc = np.indices(a.shape)
    direct = np.maximum(y+1-rr, 0)+np.maximum(rr-(y+h-2), 0)+np.maximum(x+1-cc, 0)+np.maximum(cc-(x+w-2), 0)
    def reversal(metric):
        return bool(selected.any() and retained.any() and metric[selected].max() > metric[retained].min())
    r0, r1, c0, c1 = info['bounds']
    window = np.zeros(a.shape, bool)
    window[r0:r1+1, c0:c1+1] = True
    # Certify a crop-visible membership change even if the counterfactual
    # ranking has ties. These inequalities force membership under every tie
    # resolution; queue order is never evidence for the witness.
    reachable_without_missing = payload & (missing_ignored >= 0)
    omission_crop = False
    for r, c in np.argwhere(payload & window):
        value = missing_ignored[r, c]
        if value < 0:
            continue
        before = int(np.count_nonzero(reachable_without_missing & (missing_ignored < value)))
        through = int(np.count_nonzero(reachable_without_missing & (missing_ignored <= value)))
        if (selected[r, c] and before >= info['capacity']) or (not selected[r, c] and through <= info['capacity']):
            omission_crop = True
            break
    info = dict(info)
    info.update(selected=int(selected.sum()), reachable=int((payload & (d >= 0)).sum()),
                unreachable=int((payload & (d < 0)).sum()), excess=int(retained.sum()),
                path_beats_manhattan=reversal(direct), omission_A_changes_selection=reversal(missing_ignored),
                omission_A_changes_crop=omission_crop,
                selected_in_crop=int((selected & window).sum()), excess_in_crop=int((retained & window).sum()),
                unreachable_in_crop=int((payload & (d < 0) & window).sum()),
                max_distance=int(d.max()), capacity_gap=int(d[retained].min()-d[selected].max()) if retained.any() else 0,
                key_preserved=int(selected.sum()) <= int(np.flatnonzero(interior).tolist().index(np.ravel_multi_index(info['key_position'], a.shape))))
    return info


def _paint_frame(wall, f):
    """Generator-side explicit strokes, not solver cell predicates."""
    y, x, h, w, _, _ = f
    wall[y, x:x+w] = True
    wall[y+h-1, x:x+w] = True
    wall[y:y+h, x] = True
    wall[y:y+h, x+w-1] = True
    wall[_door(f)] = False


def _reflect(a, bb, frames, axis, flip_r, flip_c):
    h, w = a.shape
    def point(r, c):
        r, c = (h-1-r if flip_r else r), (w-1-c if flip_c else c)
        return (c, r) if axis else (r, c)
    transformed = []
    for f in frames:
        y, x, fh, fw, _, _ = f
        corners = [point(y, x), point(y+fh-1, x+fw-1)]
        yy, xx = min(p[0] for p in corners), min(p[1] for p in corners)
        hh, ww = abs(corners[1][0]-corners[0][0])+1, abs(corners[1][1]-corners[0][1])+1
        gy, gx = point(*_door(f))
        if gy == yy:
            s, o = 0, gx-xx
        elif gx == xx+ww-1:
            s, o = 1, gy-yy
        elif gy == yy+hh-1:
            s, o = 2, gx-xx
        else:
            s, o = 3, gy-yy
        transformed.append((yy, xx, hh, ww, s, o))
    points = [point(bb[0], bb[2]), point(bb[1], bb[3])]
    new_bb = (min(p[0] for p in points), max(p[0] for p in points),
              min(p[1] for p in points), max(p[1] for p in points))
    out = a[::-1 if flip_r else 1, ::-1 if flip_c else 1]
    if axis:
        out = out.T
    return out.copy(), new_bb, transformed


def _candidate(rng, colors, hidden_count, axis, axial_flip, obligation):
    bg, wall_color, payload_color, marker, key = colors
    n = int(rng.integers(3, 5))
    total = n+hidden_count
    h0, w0 = int(rng.integers(4, 9)), int(rng.integers(4, 6))
    dh = int(rng.choice([-1, 0, 1, 2]))
    dw = int(rng.choice([-1, 0, 0, 0, 1]))
    if dh < 0:
        h0 += total-1
    if dw < 0:
        w0 += total-1
    gap, dg = int(rng.integers(1, 4)), int(rng.choice([0, 0, 0, 1]))
    dy = int(rng.integers(-1, 2))
    side, alt = int(rng.integers(4)), bool(rng.integers(2))
    do = int(rng.choice([-1, 0, 0, 0, 1]))
    limit = min((w0+i*dw if side in (0, 2) else h0+i*dh)-2-i*do for i in range(total))
    lower = max(1-i*do for i in range(total))
    if limit < lower:
        return None
    off = int(rng.integers(lower, limit+1))
    model = (n, 0, dy, 0, h0, dh, w0, dw, gap, dg, side, alt, off, do)
    frames = [_frame(model, i) for i in range(total)]
    active_index = int(rng.integers(n, total))
    active = frames[active_index]
    if min(active[2:4]) < 4:
        return None
    target = frames[n:]
    top_extra, bottom_extra = int(rng.integers(1, 6)), int(rng.integers(1, 6))
    overloaded = obligation == 'excess' or (obligation == 'free' and rng.random() < .3)
    if overloaded:
        top_extra = max(3, top_extra)
        bottom_extra = max(3, bottom_extra)
    r0 = min(f[0] for f in target)-top_extra
    r1 = max(f[0]+f[2]-1 for f in target)+bottom_extra
    c0 = target[0][1]-int(rng.integers(0, min(gap+(n-1)*dg, 2)+1))
    c1 = target[-1][1]+target[-1][3]-1+int(rng.integers(1, 5))
    ymin = min(r0, min(f[0] for f in frames))-int(rng.integers(0, 3))
    ymax = max(r1, max(f[0]+f[2]-1 for f in frames))+int(rng.integers(0, 3))
    xpad = int(rng.integers(0, 3))
    shape = (ymax-ymin+1, c1+xpad+1+int(rng.integers(0, 3)))
    if max(shape) > 30:
        return None
    frames = [(y-ymin, x+xpad, h, w, s, o) for y, x, h, w, s, o in frames]
    active = frames[active_index]
    bb = (r0-ymin, r1-ymin, c0+xpad, c1+xpad)
    footprints = np.zeros(shape, bool)
    guarded = np.zeros(shape, bool)
    for i, f in enumerate(frames):
        y, x, h, w, _, _ = f
        footprints[y:y+h, x:x+w] = True
        if i < n:
            guarded[max(0, y-1):min(shape[0], y+h+1), max(0, x-1):min(shape[1], x+w+1)] = True
    reserved = footprints | guarded
    for p in _corners(bb):
        reserved[p] = True
    static = np.zeros(shape, bool)
    # Closed pockets visibly distinguish unreachable supply from reachable excess.
    for _ in range(1 if obligation == 'excess' else int(rng.integers(0, 3))):
        placements = []
        ph, pw = int(rng.integers(3, 6)), int(rng.integers(3, 6))
        for y in range(bb[0], bb[1]-ph+2):
            for x in range(bb[2], bb[3]-pw+2):
                if not (reserved | static)[y:y+ph, x:x+pw].any():
                    placements.append((y, x))
        if not placements:
            if obligation == 'excess':
                return None
            break
        y, x = placements[int(rng.integers(len(placements)))]
        static[y, x:x+pw] = static[y+ph-1, x:x+pw] = True
        static[y:y+ph, x] = static[y:y+ph, x+pw-1] = True
    for _ in range(int(rng.integers(0, 6))):
        y, x = int(rng.integers(shape[0])), int(rng.integers(shape[1]))
        segment = np.zeros(shape, bool)
        length = int(rng.integers(2, 9))
        if rng.random() < .5:
            segment[y, x:min(shape[1], x+length)] = True
            if rng.random() < .3:
                segment[y:min(shape[0], y+3), x] = True
        else:
            segment[y:min(shape[0], y+length), x] = True
        if not np.any(segment & reserved):
            static |= segment
    visible = np.full(shape, bg, dtype=np.int8)
    observed_wall = static.copy()
    for f in frames[:n]:
        _paint_frame(observed_wall, f)
    visible[observed_wall] = wall_color
    # Transform geometry before serialization: prefix always follows displayed rows.
    visible, bb, frames = _reflect(visible, bb, frames, axis, bool(rng.integers(2)), axial_flip)
    # Recover the transformed static wall from the observed grid, then paint
    # the latent target frames from the separately transformed geometry.
    full_wall = visible == wall_color
    for f in frames[n:]:
        _paint_frame(full_wall, f)
    active = frames[active_index]
    interior = _interior(visible.shape, active)
    capacity = int(interior.sum())
    slots = np.flatnonzero(interior)
    key_index = int(rng.integers(2, capacity))
    visible.flat[slots[key_index]] = key
    corner_mask = np.zeros(visible.shape, bool)
    for p in _corners(bb):
        corner_mask[p] = True
    available = ~full_wall & ~interior & ~corner_mask
    distance = _wave(full_wall, interior)
    active_pool = available & (distance >= 0)
    inactive_pool = available & (distance < 0)
    window = np.zeros(visible.shape, bool)
    window[bb[0]:bb[1]+1, bb[2]:bb[3]+1] = True
    rr, cc = np.indices(visible.shape)
    y, x, h, w, _, _ = active
    direct = np.maximum(y+1-rr, 0)+np.maximum(rr-(y+h-2), 0)+np.maximum(x+1-cc, 0)+np.maximum(cc-(x+w-2), 0)
    if overloaded:
        choices = rng.permutation(np.flatnonzero(active_pool & window))
        picked = None
        for a in choices:
            near = active_pool & (distance <= distance.flat[a])
            far = active_pool & (distance > distance.flat[a])
            bs = far & window & (direct < direct.flat[a]) if obligation == 'excess' else far
            if near.sum() < capacity or not bs.any():
                continue
            b = int(rng.choice(np.flatnonzero(bs)))
            rest = np.flatnonzero(near)
            rest = rest[rest != a]
            selected_cells = [int(a)]+list(map(int, rng.choice(rest, capacity-1, replace=False)))
            far_cells = np.flatnonzero(far)
            far_cells = far_cells[far_cells != b]
            excess = int(rng.integers(1, min(9, len(far_cells)+1)+1))
            retained_cells = [b]+list(map(int, rng.choice(far_cells, excess-1, replace=False)))
            picked = selected_cells+retained_cells
            break
        if picked is None or (obligation == 'excess' and not (inactive_pool & window).any()):
            return None
        selected_cells = picked
    else:
        count = int(rng.integers(2, min(4, key_index, capacity-1)+1)) if obligation == 'partial' else int(rng.integers(1, capacity+1))
        if active_pool.sum() < count or not (active_pool & window).any():
            return None
        anchor = int(rng.choice(np.flatnonzero(active_pool & window)))
        others = np.flatnonzero(active_pool)
        others = others[others != anchor]
        selected_cells = [anchor]+list(map(int, rng.choice(others, count-1, replace=False)))
    distraction = []
    if inactive_pool.any():
        forced = int(rng.choice(np.flatnonzero(inactive_pool & window))) if (inactive_pool & window).any() else int(rng.choice(np.flatnonzero(inactive_pool)))
        choices = np.flatnonzero(inactive_pool)
        choices = choices[choices != forced]
        amount = int(rng.integers(1, min(7, len(choices)+1)+1))
        distraction = [forced]+list(map(int, rng.choice(choices, amount-1, replace=False)))
    if len(selected_cells)+len(distraction) < 2:
        return None
    visible.flat[selected_cells+distraction] = payload_color
    for p in _corners(bb):
        visible[p] = marker
    clean = visible.copy()
    clean[clean == marker] = bg
    clean[full_wall] = wall_color
    selected = _select(visible == payload_color, distance, capacity)
    full = _store(clean, interior, selected, bg, payload_color)
    output = full[bb[0]:bb[1]+1, bb[2]:bb[3]+1].copy()
    try:
        feature = pair_features(visible)
        if not np.array_equal(solve(visible), output):
            return None
        if obligation == 'excess' and not (feature['capacity'] != 9 and feature['excess_in_crop'] and feature['unreachable_in_crop']
            and feature['path_beats_manhattan'] and feature['omission_A_changes_selection'] and feature['omission_A_changes_crop']):
            return None
        if obligation == 'partial' and not (feature['selected'] < feature['capacity'] and feature['key_preserved']):
            return None
    except AssertionError:
        return None
    return {'input': visible, 'output': output}


def _counts(num_train, num_test):
    for name, value, low, high in [('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(name+' has illegal count or type')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _signature(a):
    return a.shape, tuple(map(int, a.flat))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    colors = tuple(map(int, rng.choice(10, 5, replace=False)))
    recolor = bool(rng.integers(2))
    base_axis = int(rng.integers(2))
    shift = str(rng.choice(['axis', 'direction', 'span']))
    obligations = ['partial', 'excess']+['free']*(num_train-2)
    rng.shuffle(obligations)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, specs in [('train', obligations), ('test', ['free']*num_test)]:
        for index, obligation in enumerate(specs):
            test = split == 'test'
            hidden_count = 2 if test and shift == 'span' else 1
            axis = 1-base_axis if test and shift == 'axis' else base_axis
            flip = test if shift == 'direction' else bool(rng.integers(2))
            for _ in range(700):
                palette = tuple(map(int, rng.choice(10, 5, replace=False))) if recolor else colors
                pair = _candidate(rng, palette, hidden_count, axis, flip, obligation)
                if pair is not None and split == 'train' and index == len(specs)-1:
                    capacities = {pair_features(p['input'])['capacity'] for p in puzzle['train']}
                    if len(capacities) == 1 and pair_features(pair['input'])['capacity'] in capacities:
                        continue
                if pair is not None and _signature(pair['input']) not in seen:
                    seen.add(_signature(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('Bounded reservoir construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'Invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'Splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'Invalid pair keys')
            a, b = _grid(pair['input']), _grid(pair['output'])
            sig = _signature(a)
            _require(sig not in seen, 'Duplicate input')
            seen.add(sig)
            clean, interior, predicted, info = _decode(a)
            distance = _wave(clean == info['wall'], interior)
            _require(np.array_equal(distance, _bfs(clean == info['wall'], interior)), 'BFS/wavefront disagree')
            selected = _select(a == info['payload'], distance, info['capacity'])
            full = _store(clean, interior, selected, info['background'], info['payload'])
            _require(np.count_nonzero(full == info['payload']) == np.count_nonzero(a == info['payload']),
                     'Collection must conserve payload cells on the full scene')
            _require(np.all(full[clean == info['wall']] == info['wall']), 'A wall was consumed')
            r0, r1, c0, c1 = info['bounds']
            _require(np.array_equal(b, full[r0:r1+1, c0:c1+1]), 'Incorrect restoration/collection/crop')
            _require(np.array_equal(b, solve(a.copy())), 'Input-only solve disagrees')
            _require(info['marker'] not in b, 'Annotation remains in output')
            _require(set(map(int, np.unique(b))) <= set(map(int, np.unique(a))), 'New literal output color')
            _require(np.count_nonzero(b == info['payload']) >= int(selected.sum()), 'Storage loses collected payload')
            _require(not np.any(selected & interior), 'Source payload already in active reservoir')
            _require(np.any(predicted), 'Missing frames are empty')
            features[split].append(pair_features(a))
    train = features['train']
    _require(any(2 <= f['selected'] <= 4 and f['selected'] < f['capacity'] and f['key_preserved']
                 and f['selected_in_crop'] > 0 for f in train),
             'Training must demonstrate a partial prefix with preserved key')
    _require(len({f['capacity'] for f in train}) >= 2, 'Training must expose different geometric capacities')
    _require(any(f['capacity'] != 9 and f['selected_in_crop'] and f['excess_in_crop'] and f['unreachable_in_crop'] and f['path_beats_manhattan']
                 and f['omission_A_changes_selection'] and f['omission_A_changes_crop'] and not f['key_preserved'] for f in train),
             'Training must distinguish capacity, topology, shortest paths and key overwrite')
    _require(all(f['capacity'] >= 4 for f in train+features['test']), 'Reservoir capacity below family minimum')


def _family_regressions():
    """Explicit semantic cases, including valid members outside the sampler.

    Source selections below are specified by coordinates, not computed by
    solve, BFS, wavefront, _select or _store. Check all eight scene symmetries,
    rebuilding the expected prefix in displayed order after each transform.
    """
    def check(shape, frames, visible_count, bb, active_index, key_position,
              sources, collected, static=()):
        latent = np.zeros(shape, dtype=int)
        observed = latent.copy()
        for i, (y, x, h, w, side, offset) in enumerate(frames):
            stroke = np.zeros(shape, dtype=bool)
            for r in range(y, y+h):
                for c in range(x, x+w):
                    if r in (y, y+h-1) or c in (x, x+w-1):
                        stroke[r, c] = True
            aperture = ((y, x+offset), (y+offset, x+w-1),
                        (y+h-1, x+offset), (y+offset, x))[side]
            stroke[aperture] = False
            latent[stroke] = 1
            if i < visible_count:
                observed[stroke] = 1
        for cell in static:
            latent[cell] = observed[cell] = 1
        latent[key_position] = observed[key_position] = 4
        for cell in sources:
            latent[cell] = observed[cell] = 2
        removed = np.zeros(shape, dtype=bool)
        for cell in collected:
            removed[cell] = True
        for axis in (0, 1):
            for flip_r in (False, True):
                for flip_c in (False, True):
                    a, bounds, fs = _reflect(observed, bb, frames, axis, flip_r, flip_c)
                    full, _, _ = _reflect(latent, bb, frames, axis, flip_r, flip_c)
                    erased, _, _ = _reflect(removed, bb, frames, axis, flip_r, flip_c)
                    full[erased] = 0
                    y, x, h, w, _, _ = fs[active_index]
                    slots = [(r, c) for r in range(y+1, y+h-1) for c in range(x+1, x+w-1)]
                    for cell in slots[:len(collected)]:
                        full[cell] = 2
                    for cell in _corners(bounds):
                        a[cell] = 3
                    r0, r1, c0, c1 = bounds
                    expected = full[r0:r1+1, c0:c1+1]
                    _require(np.array_equal(solve(a), expected), 'Explicit family regression failed')

    # The requested height-four frame and door offset one are valid. The
    # unused successor would have height two and a negative door offset.
    shrinking = [(1, 1+5*i, 10-2*i, 4, 1, 7-2*i) for i in range(4)]
    check((12, 22), shrinking, 3, (0, 6, 15, 20), 3, (2, 17),
          [(0, 17), (0, 18)], [(0, 17), (0, 18)])
    # Three missing frames, with the middle one keyed. Supply inside another
    # inferred frame is traversable, erased and counted in the active frame.
    alternating = [(2, 5*i, 4, 4, 2*(i % 2), 1) for i in range(6)]
    check((8, 30), alternating, 3, (0, 7, 15, 29), 4, (4, 22),
          [(1, 18), (1, 24), (3, 26)], [(1, 18), (1, 24), (3, 26)])
    # Five visible frames fit at the extreme of the 30-cell size limit.
    check((8, 30), alternating, 5, (0, 7, 25, 29), 5, (4, 27),
          [(1, 26), (1, 28)], [(1, 26), (1, 28)])
    # Nonactive target frames can reach the minimum size of three, while
    # the keyed frame still has at least two interior rows and columns.
    tapering = [(1, 5*i, 8-i, 4, 1, 1) for i in range(6)]
    check((10, 30), tapering, 3, (0, 7, 15, 29), 3, (3, 17),
          [(0, 18), (2, 26)], [(0, 18), (2, 26)])
    # All required gaps are positive; the unused next gap would be zero.
    closing_gaps = [(2, x, 4, 4, 2, 1) for x in (0, 8, 15, 21, 26)]
    check((8, 30), closing_gaps, 4, (0, 7, 26, 29), 4, (4, 28),
          [(1, 27), (1, 28)], [(1, 27), (1, 28)])
    frames = [(2, 5*i, 4, 4, 2, 1) for i in range(4)]
    # A static cell immediately outside the sole door seals the reservoir.
    check((8, 20), frames, 3, (0, 7, 15, 19), 3, (3, 17),
          [(0, 17), (1, 17)], [], static=[(6, 16)])
    # A connected six-cell line crosses a strict capacity-four frontier.
    # The first four disappear, the last two remain; payload is traversable.
    line = [(r, 16) for r in range(6, 12)]
    check((13, 20), frames, 3, (0, 12, 15, 19), 3, (3, 17), line, line[:4])


def self_test():
    _family_regressions()
    puzzles, features = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        features.extend(pair_features(pair['input']) for s in ('train', 'test') for pair in p[s])
    _require(len({tuple(_signature(q['input']) for s in ('train', 'test') for q in p[s]) for p in puzzles}) == 42,
             'Cross-seed diversity collapsed')
    for name, minimum in [('capacity', 8), ('selected', 8), ('max_distance', 15), ('gap', 2), ('background', 8)]:
        _require(len({f[name] for f in features}) >= minimum, 'Insufficient family variation: '+name)
    _require({f['axis'] for f in features} == {0, 1} and {f['forward'] for f in features} == {False, True}, 'Missing directions')
    _require({len(f['hidden']) for f in features} >= {1, 2}, 'No multiple-frame continuation')
    _require({f['alternating'] for f in features} == {False, True}, 'No door phase variation')
    _require(any(f['dh'] < 0 or f['dw'] < 0 for f in features), 'Shrinking frames absent')
    _require(any(f['dh'] > 0 or f['dw'] > 0 for f in features), 'Growing frames absent')
    _require(any(f['door_delta'] != 0 for f in features) and any(f['drift'] != 0 for f in features), 'Door/position displacement absent')
    _require(sum(f['omission_A_changes_selection'] for f in features) >= 42, 'A does not control selection')
    _require(sum(f['omission_A_changes_crop'] for f in features) >= 42, 'A witnesses lie outside the returned crop')
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(4200+10*nt+ne, nt, ne)
            _require((len(p['train']), len(p['test'])) == (nt, ne), 'Explicit counts not honored')
    a, b = generate_puzzle(90210, 10, 4), generate_puzzle(90210, 10, 4)
    _require(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x), 'Not deterministic')
    for nt, ne in [(1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1), (4, False)]:
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            continue
        raise AssertionError('Illegal count accepted')
    tied_payload = np.array([[True, True, True]])
    try:
        _select(tied_payload, np.array([[1, 1, 1]]), 2)
    except AssertionError:
        pass
    else:
        raise AssertionError('Capacity tie accepted')
    bad = generate_puzzle(42, 2, 1)
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('Corrupt label accepted')
