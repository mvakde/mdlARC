"""Shadow-switched light: one widening emitter feeds a keyed optical circuit.

All recognition and tracing use the input. No parent module, source fixture,
sampled geometry, seed, or puzzle-level state is consulted by solve().
"""
from __future__ import annotations

import json
import numpy as np


# Clockwise screen-coordinate headings: north, east, south, west.
_DIR = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid size")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((a >= 0) & (a <= 9)), "color range")
    return a.astype(np.int8, copy=True)


def _signature(a):
    return a.shape, a.astype(np.int8).tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, todo = [start], [start]
        while todo:
            y, x = todo.pop()
            for dy, dx in _DIR:
                q = y + dy, x + dx
                if q in remaining:
                    remaining.remove(q)
                    cells.append(q)
                    todo.append(q)
        result.append(cells)
    return result


def _header_info(grid):
    frame = int(grid[0, 0])
    _require(np.all(grid[0] == frame), "top key frame")
    bottom_rows = np.flatnonzero(np.all(grid[1:] == frame, axis=1)) + 1
    _require(len(bottom_rows) > 0, "bottom key frame")
    bottom = int(bottom_rows[0])
    _require(2 <= bottom < len(grid) - 1, "key height")
    band = grid[1:bottom]
    _require(np.all(band == band[0]), "vertical key bars")
    row = band[0]
    _require(row[0] == row[-1] == frame, "key side frames")
    intervals, x = [], 0
    while x < len(row):
        if row[x] == frame:
            x += 1
            continue
        begin = x
        while x < len(row) and row[x] != frame:
            x += 1
        intervals.append((begin, x))
    _require(intervals and all(b - a >= 3 for a, b in intervals), "legible panels")
    interior = int(row[intervals[0][0] + 1])
    turns = {}
    for a, b in intervals:
        _require(np.all(row[a + 1:b - 1] == interior), "common panel interior")
        left, right = int(row[a]), int(row[b - 1])
        _require((left == interior) != (right == interior), "one attached bar")
        color, side = (right, 1) if left == interior else (left, -1)
        _require(color not in turns and color not in (frame, interior), "unique key")
        turns[color] = side
    return bottom + 1, turns, tuple(b - a + 2 for a, b in intervals)


def _fit_flank(values, sign):
    """Fit the longest terminal stair; two complete step intervals are required.

    The least period breaks a redundant fit. Equal best fits must predict the
    same continuation, so recognition cannot silently choose a different cone.
    """
    values = np.asarray(values, dtype=int)
    offsets = np.arange(len(values)) - len(values) + 1
    candidates = []
    for period in (1, 2, 3):
        for phase in range(period):
            prediction = values[-1] + sign * ((offsets + phase) // period)
            length = 0
            for equal in (prediction == values)[::-1]:
                if not equal:
                    break
                length += 1
            if length >= 2 * period + 1 and abs(values[-1] - values[-length]) >= 2:
                candidates.append((length, -period, phase))
    _require(candidates, "two readable terminal step intervals")
    best_length = max(q[0] for q in candidates)
    best_period = max(q[1] for q in candidates if q[0] == best_length)
    best = [q for q in candidates if q[:2] == (best_length, best_period)]
    futures = {tuple((np.arange(1, 31) + q[2]) // -q[1]) for q in best}
    _require(len(futures) == 1, "ambiguous terminal phase")
    length, negative_period, phase = best[0]
    return -negative_period, phase, length


def _decode(input_grid):
    grid = _grid(input_grid)
    crop, turns, panel_widths = _header_info(grid)
    field = grid[crop:]
    counts = np.bincount(field.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    _require(counts[bg] > field.size // 2 and bg not in turns, "dominant background")
    matches = []
    for rotation in range(4):
        a = np.rot90(field, -rotation).copy()
        for wall in map(int, np.unique(a)):
            if wall == bg or wall in turns:
                continue
            pos = np.argwhere(a == wall)
            if len(pos) < 11:
                continue
            rear, mouth = map(int, (pos[:, 1].min(), pos[:, 1].max()))
            cap = np.flatnonzero(a[:, rear] == wall)
            if len(cap) < 3 or np.any(np.diff(cap) != 1) or mouth - rear < 5:
                continue
            lows, highs = [], []
            valid = True
            for x in range(rear + 1, mouth + 1):
                rows = np.flatnonzero(a[:, x] == wall)
                if len(rows) != 2 or rows[1] - rows[0] < 2:
                    valid = False
                    break
                lows.append(int(rows[0]))
                highs.append(int(rows[1]))
            if not valid:
                continue
            if abs(lows[0] - cap[0]) > 1 or abs(highs[0] - cap[-1]) > 1:
                continue
            if any(abs(v) > 1 for v in np.diff(lows)) or any(abs(v) > 1 for v in np.diff(highs)):
                continue
            try:
                fits = _fit_flank(lows, -1), _fit_flank(highs, 1)
            except AssertionError:
                continue
            chamber = np.zeros(a.shape, dtype=bool)
            for x, lo, hi in zip(range(rear + 1, mouth + 1), lows, highs):
                chamber[lo + 1:hi, x] = True
            markers = np.argwhere(chamber & (a != bg))
            if not len(markers):
                continue
            payload_colors = set(map(int, a[markers[:, 0], markers[:, 1]]))
            if len(payload_colors) != 1:
                continue
            payload = payload_colors.pop()
            if payload in turns or payload == wall:
                continue
            patch = a == payload
            if np.any(patch & ~chamber) or len(_components(patch)) != 1:
                continue
            if len(_components(chamber)) != 1:
                continue
            cone = np.zeros(a.shape, dtype=bool)
            bounds = []
            for x in range(mouth + 1, a.shape[1]):
                t = x - mouth
                lo = lows[-1] - (t + fits[0][1]) // fits[0][0]
                hi = highs[-1] + (t + fits[1][1]) // fits[1][0]
                cone[max(0, lo + 1):min(len(a), hi), x] = True
                bounds.append((lo, hi))
            births = np.zeros(a.shape, dtype=bool)
            for y in range(len(a)):
                xs = np.flatnonzero(cone[y])
                if len(xs):
                    births[y, xs[0]] = True
            if not np.any(births) or np.any(a[births] != bg):
                continue
            matches.append(dict(a=a, crop=crop, rotation=rotation, bg=bg,
                                wall=wall, payload=payload, rear=rear, mouth=mouth,
                                lows=tuple(lows), highs=tuple(highs), fits=fits,
                                chamber=chamber, cone=cone, births=births,
                                bounds=tuple(bounds), turns=turns, panels=panel_widths))
    _require(len(matches) == 1, "one input-derived emitter interpretation")
    info = matches[0]
    # Each keyed connected component is a solid rectangle. Repeated colors and
    # unused keys are valid; no key can also be the payload or the background.
    rectangles = []
    for color in turns:
        for cells in _components(info['a'] == color):
            ys, xs = zip(*cells)
            box = min(ys), max(ys) + 1, min(xs), max(xs) + 1
            _require(len(cells) == (box[1] - box[0]) * (box[3] - box[2]), "solid relay")
            rectangles.append((*box, color))
    info['rectangles'] = tuple(rectangles)
    return info


def _trace(info, *, widen=True, absorb_keys=False, transparent_plain=False,
           max_turns=None, glyph_mapping=None, ungated=False):
    """Finite directed reachability. Each lane is independent; paint is a union.

    The optional switches are concrete shortcut hypotheses used by validation,
    never conventions or hidden inputs used by the public solver.
    """
    a, bg, payload = info['a'], info['bg'], info['payload']
    h, w = a.shape
    out = a.copy()
    out[info['chamber'] & (a == bg)] = payload
    births = info['births']
    if not widen:
        births = np.zeros_like(births)
        births[info['lows'][-1] + 1:info['highs'][-1], info['mouth'] + 1] = True
    if ungated:
        births = births.copy()
        for y0, y1, x0, x1, color in info['rectangles']:
            if x0 > 0:
                for y in range(y0, y1):
                    if a[y, x0 - 1] == bg:
                        births[y, x0 - 1] = True
    turns = {} if absorb_keys else info['turns']
    if glyph_mapping is not None:
        turns = {c: glyph_mapping[side] for c, side in turns.items()}
    events, paths, cycles, max_depth = [], [], False, 0
    directed = np.zeros((4, h, w), dtype=bool)
    for yy, xx in np.argwhere(births):
        y, x, d, depth = int(yy), int(xx), 1, 0
        seen, path = set(), []
        while True:
            state = y, x, d
            if state in seen:
                cycles = True
                break
            seen.add(state)
            directed[d, y, x] = True
            path.append((y, x, d, depth))
            if a[y, x] == bg:
                out[y, x] = payload
            dy, dx = _DIR[d]
            ny, nx = y + dy, x + dx
            if not (0 <= ny < h and 0 <= nx < w):
                break
            color = int(a[ny, nx])
            if color in turns:
                if max_turns is not None and depth >= max_turns:
                    break
                side = turns[color]
                events.append((ny, nx, color, d, side, y, x, depth, int(yy)))
                depth += 1
                max_depth = max(max_depth, depth)
                d = (d + side) % 4
                continue
            if color not in (bg, payload):
                # Structural walls always absorb, even in the no-shadow rival.
                if not transparent_plain or color == info['wall'] or color in info['turns']:
                    break
            y, x = ny, nx
        paths.append(tuple(path))
    return out, dict(events=tuple(events), paths=tuple(paths), cycles=cycles,
                     depth=max_depth, directed=directed)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the transformed field, discarding the visible top key band."""
    info = _decode(input_grid)
    out, _ = _trace(info)
    return np.rot90(out, info['rotation']).copy()


def _make_emitter(rng, rotation, projection_range, lesson=False):
    # Both dimensions reserve room for the unrotated top instruction band.
    max_h, max_w = (30, 27) if rotation % 2 else (27, 30)
    for _ in range(400):
        ptop = int(rng.integers(1, 4))
        pbottom = int(rng.integers(1, 4))
        if projection_range[1] <= 11:
            ptop = int(rng.integers(1, 3))
            pbottom = int(rng.integers(1, 3))
        phase = [int(rng.integers(p)) for p in (ptop, pbottom)]
        reservoir = int(rng.integers(2, 6))
        profiles = []
        for _ in range(2):
            start = int(rng.integers(1, 4))
            profile = [start]
            for j in range(reservoir - 1):
                step = int(rng.choice([-1, 0, 1]))
                profile.append(min(4, max(1, profile[-1] + step)))
            profiles.append(profile)
        # >= three outward steps makes fitting and phase inference explicit.
        flare = 3 * max(ptop, pbottom) + int(rng.integers(1, 3))
        up = profiles[0] + [profiles[0][-1] + (t + phase[0]) // ptop for t in range(1, flare + 1)]
        down = profiles[1] + [profiles[1][-1] + (t + phase[1]) // pbottom for t in range(1, flare + 1)]
        rear = int(rng.integers(0, 4))
        mouth = rear + len(up)
        min_projection = max(projection_range[0], 3 * ptop + 5 if lesson else 3)
        max_projection = min(projection_range[1], max_w - mouth - 1)
        span = max(up) + max(down) + 1
        if max_projection < min_projection or span + 4 > max_h:
            continue
        projection = int(rng.integers(min_projection, max_projection + 1))
        w = max(mouth + projection + 1, 15)
        h = int(rng.integers(max(15, span + 4), max_h + 1))
        center = int(rng.integers(max(up) + 2, h - max(down) - 1))
        grid = np.zeros((h, w), dtype=np.int8)
        grid[center - up[0]:center + down[0] + 1, rear] = 1
        chamber = np.zeros_like(grid, dtype=bool)
        for j, (u, v) in enumerate(zip(up, down), rear + 1):
            grid[center - u, j] = grid[center + v, j] = 1
            chamber[center - u + 1:center + v, j] = True
        candidates = chamber.copy()
        candidates[:, rear + reservoir + 1:] = False
        cells = set(map(tuple, np.argwhere(candidates)))
        start = sorted(cells)[int(rng.integers(len(cells)))]
        patch = {start}
        for _ in range(int(rng.integers(0, 6))):
            frontier = sorted({(y + dy, x + dx) for y, x in patch for dy, dx in _DIR
                               if (y + dy, x + dx) in cells and (y + dy, x + dx) not in patch})
            if not frontier:
                break
            patch.add(frontier[int(rng.integers(len(frontier)))])
        for y, x in patch:
            grid[y, x] = 2
        low = [center - v for v in up]
        high = [center + v for v in down]
        cone, births = np.zeros_like(chamber), np.zeros_like(chamber)
        for x in range(mouth + 1, w):
            t = x - (rear + reservoir)
            lo = center - profiles[0][-1] - (t + phase[0]) // ptop
            hi = center + profiles[1][-1] + (t + phase[1]) // pbottom
            cone[max(0, lo + 1):min(h, hi), x] = True
        for y in range(h):
            xs = np.flatnonzero(cone[y])
            if len(xs):
                births[y, xs[0]] = True
        return grid, dict(mouth=mouth, rear=rear, chamber=chamber, cone=cone,
                          births=births, lows=low, highs=high, ptop=ptop,
                          reservoir=reservoir)
    raise AssertionError("bounded emitter construction exhausted")


def _put_rectangle(field, box, color, chamber, births):
    y0, y1, x0, x1 = box
    h, w = field.shape
    if not (0 <= y0 < y1 <= h and 0 <= x0 < x1 <= w):
        return False
    if np.any(field[y0:y1, x0:x1] != 0):
        return False
    if np.any(chamber[y0:y1, x0:x1] | births[y0:y1, x0:x1]):
        return False
    if color >= 4 and np.any(field[max(0, y0 - 1):min(h, y1 + 1),
                                     max(0, x0 - 1):min(w, x1 + 1)] == color):
        return False
    field[y0:y1, x0:x1] = color
    return True


def _make_header(rng, width, height, turns):
    # Logical palette: background 0, wall/frame 1, payload 2, absorber 3,
    # relays 4..7. Actual colors are assigned only after construction.
    n = len(turns)
    sizes = np.full(n, 5, dtype=int)
    _require(width >= 5 * n, "panel fit")
    if width % n == 0 and rng.random() < .35:
        sizes[:] = width // n
    else:
        sizes += rng.multinomial(width - 5 * n, np.full(n, 1 / n))
    header = np.zeros((height, width), dtype=np.int8)
    header[[0, -1]] = 1
    x = 0
    for c, size in zip(rng.permutation(list(turns)), sizes):
        header[:, x] = header[:, x + size - 1] = 1
        header[1:-1, x + 1 if turns[int(c)] == -1 else x + size - 2] = c
        x += int(size)
    return header


def _pack(rng, field, turns, rotation, palette):
    oriented = np.rot90(field, rotation).copy()
    max_keys = min(4, oriented.shape[1] // 5)
    turns = {c: side for c, side in turns.items() if c < 4 + max_keys}
    _require(set(map(int, np.unique(field))) & set(range(4, 8)) <= set(turns), "field key fit")
    hh = int(rng.integers(3, min(6, 30 - oriented.shape[0]) + 1))
    header = _make_header(rng, oriented.shape[1], hh, turns)
    inp = palette[np.vstack((header, oriented))].astype(np.int8)
    info = _decode(inp)
    _require(info['rotation'] == rotation, "sampled/inferred orientation agreement")
    output, trace = _trace(info)
    _require(not trace['cycles'], "sampler avoids cyclic circuits")
    return dict(input=inp, output=np.rot90(output, rotation).copy()), info, trace


def _lesson(rng, rotation, projection_range, mirrored, palette):
    for _ in range(160):
        field, geom = _make_emitter(rng, rotation, projection_range, lesson=True)
        h, w = field.shape
        mouth = geom['mouth']
        first_x = int(rng.integers(mouth + 3 * geom['ptop'], w - 4))
        ys = np.flatnonzero(geom['cone'][:, first_x])
        first_y = int(ys[0]) + 1
        # A newly exposed lane must have travelled at least one clear cell.
        if first_y >= geom['lows'][-1] + 1 or geom['births'][first_y, first_x - 1]:
            continue
        corner_x = first_x - 1
        outside_y = int(np.flatnonzero(geom['cone'][:, corner_x])[0]) - 3
        absorber_y = int(rng.integers(geom['lows'][-1] + 2, geom['highs'][-1] - 1))
        first_height = int(rng.integers(1, 4))
        abs_height = int(rng.integers(1, min(3, geom['highs'][-1] - absorber_y)))
        boxes = [
            ((first_y, first_y + first_height, first_x, first_x + int(rng.integers(1, 4))), 4),
            ((outside_y - int(rng.integers(0, 2)), outside_y + 1,
              corner_x, corner_x + int(rng.integers(1, 4))), 5),
            ((absorber_y, absorber_y + abs_height, mouth + 2,
              min(first_x - 1, mouth + 2 + int(rng.integers(1, 4)))), 3),
            ((absorber_y, absorber_y + abs_height, first_x + 2,
              min(w, first_x + 2 + int(rng.integers(1, 3)))), 5),
        ]
        if not all(_put_rectangle(field, box, c, geom['chamber'], geom['births']) for box, c in boxes):
            continue
        turns = {4: -1, 5: 1}
        for c in range(6, int(rng.integers(6, 9))):
            turns[c] = int(rng.choice([-1, 1]))
        if mirrored:
            field = field[::-1].copy()
            turns = {c: -side for c, side in turns.items()}
        try:
            pair, info, trace = _pack(rng, field, turns, rotation, palette)
            intended_cone = geom['cone'][::-1] if mirrored else geom['cone']
            _require(np.array_equal(info['cone'], intended_cone), "lesson continuation agreement")
            f = _features(info, trace, pair['output'])
            if not (f['dormant_shadow'] and f['off_cone_hit'] and f['widen_relay'] and f['shadow']):
                continue
            if any(np.array_equal(_trace(info, **kw)[0], np.rot90(pair['output'], -rotation))
                   for kw in ({'absorb_keys': True}, {'widen': False},
                              {'transparent_plain': True}, {'max_turns': 1}, {'ungated': True})):
                continue
            return pair
        except AssertionError:
            continue
    raise AssertionError("bounded optical lesson construction exhausted")


def _free_pair(rng, rotation, projection_range, palette, style):
    for _ in range(50):
        field, geom = _make_emitter(rng, rotation, projection_range)
        h, w = field.shape
        max_keys = min(4, (h if rotation % 2 else w) // 5)
        n = int(rng.integers(2, max_keys + 1))
        turns = {c: int(rng.choice([-1, 1])) for c in range(4, 4 + n)}
        if rng.random() < .25:
            turns = {c: int(rng.choice([-1, 1])) for c in turns} if rng.random() < .5 else {c: -1 for c in turns}
        target = {'clean': 0, 'light': int(rng.integers(1, 4)),
                  'free': int(rng.integers(2, 7)), 'dense': int(rng.integers(5, 10))}[style]
        try:
            _, info, trace = _pack(rng, field, turns, rotation, palette)
        except AssertionError:
            continue
        # Grow routes from real reachable segments. New objects can reroute or
        # extinguish old routes; each candidate is retraced after insertion.
        added = 0
        for attempt in range(12 * target + 1):
            if added >= target:
                break
            color = 3 if rng.random() < .28 else int(rng.choice(list(turns)))
            depth = int(rng.integers(1, 4))
            if rng.random() < .2:
                y = int(rng.integers(h)); x = int(rng.integers(w))
                box = y, min(h, y + depth), x, min(w, x + int(rng.integers(1, 5)))
            else:
                # States in canonical logical coordinates, independently of
                # recoloring. Avoid initial birth cells and tight double turns.
                states = [s for path in trace['paths'] for s in path
                          if s[1] > geom['mouth'] and not geom['births'][s[0], s[1]]]
                if not states:
                    break
                recursive = [s for s in states if s[3] > 0]
                if recursive and rng.random() < .65:
                    states = recursive
                y, x, d, _ = states[int(rng.integers(len(states)))]
                dy, dx = _DIR[d]
                ny, nx = y + dy, x + dx
                before, after = int(rng.integers(0, 3)), int(rng.integers(0, 3))
                if d == 0:
                    box = max(0, ny - depth + 1), ny + 1, max(0, nx - before), min(w, nx + after + 1)
                elif d == 2:
                    box = ny, min(h, ny + depth), max(0, nx - before), min(w, nx + after + 1)
                elif d == 1:
                    box = max(0, ny - before), min(h, ny + after + 1), nx, min(w, nx + depth)
                else:
                    box = max(0, ny - before), min(h, ny + after + 1), max(0, nx - depth + 1), nx + 1
            trial = field.copy()
            if not _put_rectangle(trial, box, color, geom['chamber'], geom['births']):
                continue
            try:
                pair, candidate_info, candidate_trace = _pack(rng, trial, turns, rotation, palette)
            except AssertionError:
                continue
            field, info, trace = trial, candidate_info, candidate_trace
            added += 1
        try:
            pair, info, trace = _pack(rng, field, turns, rotation, palette)
        except AssertionError:
            continue
        # Verify inferred continuation against the actual sampled thin rails.
        _require(np.array_equal(info['cone'], geom['cone']), "sampled/inferred cone agreement")
        return pair
    raise AssertionError("bounded free scene construction exhausted")


def _features(info, trace, output=None):
    a = info['a']
    bg, wall, payload = info['bg'], info['wall'], info['payload']
    out = _trace(info)[0] if output is None else np.rot90(output, -info['rotation'])
    hit = {(e[0], e[1]) for e in trace['events']}
    off_hit = any(not info['cone'][y, x] for y, x in hit)
    dormant_shadow = False
    plain = (a != bg) & (a != wall) & (a != payload) & ~np.isin(a, list(info['turns']))
    shadow_mask = np.zeros_like(plain)
    for y, x in np.argwhere(plain & info['cone']):
        shadow_mask[y, x + 1:] = True
    for y0, y1, x0, x1, c in info['rectangles']:
        cells = {(y, x) for y in range(y0, y1) for x in range(x0, x1)}
        if not cells & hit and any(info['cone'][y, x] and shadow_mask[y, x] for y, x in cells):
            dormant_shadow = True
    positions = np.argwhere(a == wall)
    y0, x0 = positions.min(axis=0); y1, x1 = positions.max(axis=0) + 1
    patch = np.argwhere(a == payload)
    py0, px0 = patch.min(axis=0); py1, px1 = patch.max(axis=0) + 1
    head = {side: {e[3] for e in trace['events'] if e[4] == side} for side in (-1, 1)}
    return dict(shape=a.shape, rotation=info['rotation'], projection=a.shape[1] - info['mouth'] - 1,
                outline=_signature((a[y0:y1, x0:x1] == wall).astype(np.int8)),
                patch=_signature((a[py0:py1, px0:px1] == payload).astype(np.int8)),
                palette=(bg, wall, payload), periods=tuple(q[0] for q in info['fits']),
                panels=info['panels'], objects=len(info['rectangles']) + len(_components(plain)),
                depth=trace['depth'], head=head, dormant_shadow=dormant_shadow,
                off_cone_hit=off_hit, shadow=bool(np.any(shadow_mask & info['cone'] & (a == bg) & (out == bg))),
                widen_relay=any(e[3] == 1 and e[7] == 0 and
                                not info['lows'][-1] < e[0] < info['highs'][-1]
                                for e in trace['events']),
                crossing=bool(np.any((trace['directed'][0] | trace['directed'][2]) &
                                    (trace['directed'][1] | trace['directed'][3]))),
                nonmonotone=bool(np.any(np.diff(np.array(info['highs']) - np.array(info['lows'])) < 0)),
                asymmetric=bool(len(set(np.array(info['lows']) + np.array(info['highs']))) > 1),
                unused_keys=len(set(info['turns']) - {c for *_, c in info['rectangles']}),
                repeated_colors=len(info['rectangles']) > len({c for *_, c in info['rectangles']}))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "counts")
    seen, infos, features = set(), {'train': [], 'test': []}, {'train': [], 'test': []}
    ablations = {name: False for name in ('absorb_keys', 'widen', 'transparent_plain', 'max_turns', 'ungated')}
    glyph_fits = {(-1, -1), (-1, 1), (1, -1), (1, 1)}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair keys")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            key = _signature(inp)
            _require(key not in seen, "distinct inputs")
            seen.add(key)
            info = _decode(inp)
            expected, trace = _trace(info)
            canonical_out = np.rot90(out, -info['rotation'])
            _require(np.array_equal(expected, canonical_out), "input-only solution mismatch")
            _require(not trace['cycles'], "generated cycle")
            a = info['a']
            _require(np.array_equal(a[a != info['bg']], expected[a != info['bg']]), "object preservation")
            changed = a != expected
            _require(np.any(changed) and np.all(expected[changed] == info['payload']), "payload fill")
            _require(np.all(expected[info['chamber'] & (a == info['bg'])] == info['payload']), "chamber fill")
            infos[split].append(info)
            features[split].append(_features(info, trace, out))
            if split == 'train':
                for name, value in (('absorb_keys', True), ('widen', False),
                                    ('transparent_plain', True), ('max_turns', 1), ('ungated', True)):
                    ablations[name] |= not np.array_equal(_trace(info, **{name: value})[0], expected)
                glyph_fits = {s for s in glyph_fits if np.array_equal(
                    _trace(info, glyph_mapping={-1: s[0], 1: s[1]})[0], expected)}
            else:
                for kw in ({'absorb_keys': True}, {'widen': False}):
                    _require(not np.array_equal(_trace(info, **kw)[0], expected),
                             "query must use widening and keyed turning")
    _require(all(ablations.values()), "training must distinguish both parents, shadows and recursion")
    _require(glyph_fits == {(-1, 1)}, "uniquely demonstrated left/right meaning")
    train = features['train']
    for side in (-1, 1):
        _require(len(set.union(*(f['head'][side] for f in train))) >= 2, "relative turns from multiple headings")
    for flag in ('dormant_shadow', 'off_cone_hit', 'widen_relay', 'shadow'):
        _require(any(f[flag] for f in train), "missing optical interaction: " + flag)
    _require(len({f['outline'] for f in train}) >= 2, "different teaching emitters")
    _require(any(len({q['turns'][color] for q in infos['train'] if color in q['turns']}) == 2
                 for color in range(10)), "training must distinguish local keys from a fixed color map")
    # Every query shifts an observed structural feature, while the local rule
    # stays fixed: a held-out direction or a longer forward projection.
    directions = {f['rotation'] for f in train}
    longest = max(f['projection'] for f in train)
    _require(all(f['rotation'] not in directions or f['projection'] > longest
                 for f in features['test']), "meaningful query shift")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base = rng.permutation(10)
    shared = bool(rng.random() < .25)
    preferred = int(rng.integers(4))
    held = int(rng.choice([r for r in range(4) if r != preferred]))
    shift = str(rng.choice(['direction', 'projection']))
    train_range = (8, 20) if shift == 'direction' else (8, 11)
    test_range = (4, 20) if shift == 'direction' else (12, 20)
    puzzle, seen = {'train': [], 'test': []}, set()

    def colors():
        palette = base.copy()
        if not shared:
            # Recurring relay colors retain visible identity, while their bar
            # meanings change. Other semantic colors are locally recognizable.
            palette[:4] = rng.permutation(base[:4])
        return palette

    for i in range(num_train + num_test):
        split = 'train' if i < num_train else 'test'
        for attempt in range(40):
            palette = colors()
            if i < 2:
                pair = _lesson(rng, preferred, train_range, bool(i), palette)
            else:
                allowed = [r for r in range(4) if r != held] if shift == 'direction' and split == 'train' else list(range(4))
                rotation = held if shift == 'direction' and split == 'test' else int(rng.choice(allowed))
                if split == 'test' and attempt >= 30:
                    # A randomized circuit lesson is a bounded fallback, with
                    # the same required visible shift and no latent convention.
                    pair = _lesson(rng, rotation, test_range, bool(rng.integers(2)), palette)
                else:
                    style = str(rng.choice(['clean', 'light', 'free', 'dense'], p=[.2, .3, .3, .2]))
                    pair = _free_pair(rng, rotation, train_range if split == 'train' else test_range, palette, style)
                if split == 'test':
                    info = _decode(pair['input'])
                    answer = _trace(info)[0]
                    if any(np.array_equal(_trace(info, **kw)[0], answer)
                           for kw in ({'absorb_keys': True}, {'widen': False})):
                        continue
            key = _signature(pair['input'])
            if i == 1:
                earlier = _decode(puzzle['train'][0]['input'])
                current = _decode(pair['input'])
                if _features(earlier, _trace(earlier)[1])['outline'] == _features(current, _trace(current)[1])['outline']:
                    continue
            if key not in seen:
                seen.add(key)
                puzzle[split].append(pair)
                break
        else:
            raise AssertionError("bounded unique-input construction exhausted")
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def self_test():
    """Bounded 42-seed structural/semantic checks; no independent solver claim."""
    puzzles, records = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        puzzles.append(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                _require(np.array_equal(solve(pair['input']), pair['output']), "solve check")
                info = _decode(pair['input'])
                records.append((split, _features(info, _trace(info)[1], pair['output'])))
    _require(len({tuple(_signature(q['input']) for s in ('train', 'test') for q in p[s])
                  for p in puzzles}) == 42, "seed diversity")
    fs = [f for _, f in records]
    for name, minimum in (('outline', 70), ('shape', 45), ('patch', 12), ('palette', 45), ('panels', 20)):
        _require(len({f[name] for f in fs}) >= minimum, "structural diversity: " + name)
    _require({r for f in fs for r in f['periods']} == {1, 2, 3}, "cadence variation")
    _require(len({f['objects'] for f in fs}) >= 6, "object-count variation")
    for flag in ('nonmonotone', 'asymmetric', 'crossing', 'unused_keys', 'repeated_colors'):
        _require(sum(bool(f[flag]) for f in fs) >= 5, "compatible variation: " + flag)
    for split in ('train', 'test'):
        _require({f['rotation'] for s, f in records if s == split} == set(range(4)), "orientation coverage")
    for seed in (0, 17, 41):
        again = generate_puzzle(seed)
        _require(all(np.array_equal(q[k], r[k]) for s in ('train', 'test')
                     for q, r in zip(puzzles[seed][s], again[s]) for k in ('input', 'output')), "determinism")
        # These transformations preserve the rule without consulting its tracer.
        # Mirroring the whole input reverses each visible bar's handedness too.
        permutation = np.random.default_rng(4700 + seed).permutation(10)
        for pair in (puzzles[seed]['train'][0], puzzles[seed]['test'][0]):
            inp, out = pair['input'], pair['output']
            frozen = inp.copy()
            _require(np.array_equal(solve(inp.tolist()), out), "nested-list input")
            _require(np.array_equal(inp, frozen), "solver must preserve its input")
            _require(np.array_equal(solve(permutation[inp]), permutation[out]), "semantic color roles")
            _require(np.array_equal(solve(inp[:, ::-1]), out[:, ::-1]), "mirrored cone and keyed turns")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(100 + nt * 10 + nq, nt, nq)
            _require(len(p['train']) == nt and len(p['test']) == nq, "explicit counts")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    _semantic_regressions()
    return {'seeds': 42, 'pairs': len(fs), 'outlines': len({f['outline'] for f in fs}),
            'shapes': len({f['shape'] for f in fs}), 'independent_solver': False}


def _semantic_regressions():
    # Hand-authored elementary circuit: east -> north -> east, plus absorption.
    a = np.zeros((12, 15), dtype=np.int8)
    a[6, 10] = 4
    a[2, 9] = 5
    a[8, 6] = 3
    births = np.zeros_like(a, dtype=bool)
    births[6, 4] = births[8, 4] = True
    info = dict(a=a, bg=0, payload=2, wall=1, births=births,
                chamber=np.zeros_like(births), turns={4: -1, 5: 1})
    out, trace = _trace(info)
    expected = a.copy()
    expected[6, 4:10] = 2
    expected[3:7, 9] = 2
    expected[3, 9:] = 2
    expected[8, 4:6] = 2
    _require(np.array_equal(out, expected) and trace['depth'] == 2, "hand circuit")
    # A closed route has a finite reachable-cell union, independent of order.
    a = np.zeros((10, 10), dtype=np.int8)
    for y, x in ((4, 7), (7, 6), (6, 3), (3, 4)):
        a[y, x] = 4
    births = np.zeros_like(a, dtype=bool); births[4, 4] = True
    info.update(a=a, births=births, chamber=np.zeros_like(births), turns={4: 1})
    out, trace = _trace(info)
    _require(trace['cycles'] and np.count_nonzero(out == 2) == 8, "finite cyclic union")
    # Read all phases of each supported thin terminal cadence.
    for p in (1, 2, 3):
        for phase in range(p):
            values = [10 - (t + phase) // p for t in range(3 * p + 2)]
            fit = _fit_flank(values, -1)
            for t in range(1, 8):
                inferred = values[-1] - (t + fit[1]) // fit[0]
                literal = 10 - (len(values) - 1 + t + phase) // p
                _require(inferred == literal, "thin boundary phase")
    p = generate_puzzle(73, 2, 1)
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    pair = damaged['test'][0]
    info = _decode(pair['input'])
    field = pair['input'][info['crop']:]
    y, x = np.argwhere(field != pair['output'])[0]
    pair['output'][y, x] = field[y, x]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        return
    raise AssertionError("corrupt label accepted")


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
