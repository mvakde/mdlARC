"""Sealed-chamber routing: enclosure owns a beacon; pocket area chooses a guide.

Only NumPy and the standard library are used. The solver has no generation state.
"""
from __future__ import annotations

import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _check(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _check(np.issubdtype(a.dtype, np.integer), "grid must contain integers")
    _check(bool(np.all((a >= 0) & (a <= 9))), "invalid color")
    return a.astype(np.int16, copy=False)


def _regions(points, directions=D4):
    left = set(points)
    result = []
    while left:
        p = min(left)
        left.remove(p)
        region, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in directions:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    region.add(q)
                    stack.append(q)
        result.append(region)
    return result


def _cavities(body, directions=D4):
    """Use an outside padding ring, including for a body touching the canvas."""
    r0, r1 = min(r for r, c in body) - 1, max(r for r, c in body) + 1
    c0, c1 = min(c for r, c in body) - 1, max(c for r, c in body) + 1
    empty = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)} - body
    regions = _regions(empty, directions)
    return [s for s in regions if (r0, c0) not in s]


def _shift(points, delta):
    dr, dc = delta
    return {(r + dr, c + dc) for r, c in points}


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _decode(input_grid):
    g = _grid(input_grid)
    h, w = g.shape
    colors, counts = np.unique(g, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    _check(counts.max() > g.size / 2, "background must occupy more than half the input")
    candidates = []
    for col in colors:
        if col == bg:
            continue
        m = g == col
        if np.any(m[1:] & m[:-1]) or np.any(m[:, 1:] & m[:, :-1]):
            candidates.append(int(col))
    _check(len(candidates) == 1, "exactly one color must form adjacent body cells")
    wall = candidates[0]
    bodies = _regions(map(tuple, np.argwhere(g == wall).tolist()))
    _check(all(len(s) >= 2 for s in bodies), "body singletons are outside this family")
    _check(len(bodies) == len(_regions(set().union(*bodies), D8)), "diagonal body contact")
    signals = {tuple(p): int(g[tuple(p)]) for p in np.argwhere((g != bg) & (g != wall)).tolist()}
    anchors = {}
    for p, color in signals.items():
        if p[0] in (0, h - 1) or p[1] in (0, w - 1):
            anchors.setdefault(color, set()).add(p)
    _check(anchors, "no boundary guide")
    guides = {}
    for color, ps in anchors.items():
        horizontal = len({r for r, c in ps}) == 1 and all(c in (0, w - 1) for r, c in ps)
        vertical = len({c for r, c in ps}) == 1 and all(r in (0, h - 1) for r, c in ps)
        _check(horizontal != vertical, "inconsistent or corner guide")
        axis = 0 if horizontal else 1
        coordinate = next(iter(ps))[axis]
        _check(0 < coordinate < g.shape[axis] - 1, "corner guide")
        guides[color] = (axis, coordinate)
    _check(set(signals.values()) <= set(guides), "signal lacks a same-colored guide")
    owned, objects = set(), []
    for body in bodies:
        holes = _cavities(body)
        _check({frozenset(s) for s in holes} == {frozenset(s) for s in _cavities(body, D8)},
               "diagonal-only cavity leak")
        pockets = []
        for hole in holes:
            _check(all(_inside(p, g.shape) and g[p] != wall for p in hole), "nested body")
            _check(not (hole & owned), "overlapping cavity ownership")
            beacons = hole & signals.keys()
            _check(len(beacons) == 1, "each cavity needs exactly one beacon")
            beacon = next(iter(beacons))
            pockets.append((hole, beacon, signals[beacon]))
            owned.update(hole)
        if pockets:
            sizes = [len(x[0]) for x in pockets]
            _check(sizes.count(max(sizes)) == 1, "largest cavity must be unique")
            control = pockets[int(np.argmax(sizes))]
            axis, guide = guides[control[2]]
            delta = [0, 0]
            delta[axis] = guide - control[1][axis]
            delta = tuple(delta)
            paint = control[2]
        else:
            control, delta, paint = None, (0, 0), wall
        empty_inside = set().union(*(x[0] for x in pockets)) if pockets else set()
        halo = {(r + dr, c + dc) for r, c in body for dr, dc in D8} - body - empty_inside
        objects.append(dict(body=body, pockets=pockets, control=control, delta=delta,
                            paint=paint, halo=halo))
    stationary = {p: col for p, col in signals.items() if p not in owned}
    return dict(grid=g, background=bg, wall=wall, guides=guides, anchors=anchors,
                stationary=stationary, objects=objects)


def _render(scene):
    shape = scene['grid'].shape
    out = np.full(shape, scene['background'], np.int16)
    payload = dict(scene['stationary'])
    halos = set()
    for obj in scene['objects']:
        delta = obj['delta']
        halos.update(p for p in _shift(obj['halo'], delta) if _inside(p, shape))
        zones = [(obj['body'], obj['paint'])] + [(s, col) for s, mark, col in obj['pockets']]
        for points, color in zones:
            for p in _shift(points, delta):
                if _inside(p, shape):
                    _check(p not in payload, "destination overlaps a body, cavity or fixed signal")
                    payload[p] = color
    for p in halos - payload.keys():
        out[p] = scene['wall']
    for p, color in payload.items():
        out[p] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover walls, pockets, beacons and guides solely from the input."""
    return _render(_decode(input_grid))


def _crop(mask, marks):
    rr, cc = np.nonzero(mask)
    r, c = int(rr.min()), int(cc.min())
    return mask[r:rr.max() + 1, c:cc.max() + 1].copy(), {(a-r, b-c): v for (a, b), v in marks.items()}


def _orient(rng, mask, marks):
    for _ in range(int(rng.integers(4))):
        w = mask.shape[1]
        marks = {(w - 1 - c, r): v for (r, c), v in marks.items()}
        mask = np.rot90(mask)
    if rng.random() < .5:
        w = mask.shape[1]
        marks = {(r, w - 1 - c): v for (r, c), v in marks.items()}
        mask = np.fliplr(mask)
    return _crop(mask, marks)


def _pocket(rng, h, w, irregular=False):
    m = np.ones((h, w), bool)
    if irregular and h >= 2 and w >= 2:
        # Connected L-shaped or stepped empty regions, without diagonal contacts.
        a, b = int(rng.integers(1, h)), int(rng.integers(1, w))
        m[a:, b:] = False
    return m


def _body(rng, pocket_masks, beacon_colors, decorate=True):
    """Carve separate pockets; sculpt the exterior without altering enclosure."""
    top, bottom, left, right = map(int, rng.integers(1, 3, size=4))
    offsets = list(map(int, rng.integers(0, 3, size=len(pocket_masks))))
    gaps = list(map(int, rng.integers(1, 3, size=max(0, len(pocket_masks) - 1))))
    h = top + max(m.shape[0] + d for m, d in zip(pocket_masks, offsets)) + bottom
    w = left + sum(m.shape[1] for m in pocket_masks) + sum(gaps) + right
    mask = np.ones((h, w), bool)
    marks, c = {}, left
    for j, (pocket, col) in enumerate(zip(pocket_masks, beacon_colors)):
        r = top + offsets[j]
        ph, pw = pocket.shape
        mask[r:r + ph, c:c + pw][pocket] = False
        choices = np.argwhere(pocket)
        mr, mc = map(int, choices[int(rng.integers(len(choices)))])
        marks[r + mr, c + mc] = col
        c += pw + (gaps[j] if j < len(gaps) else 0)
    if decorate:
        # Uneven lobes and bent tails change body geometry independently of pockets.
        pad = int(rng.integers(1, 4))
        enlarged = np.pad(mask, ((0, pad), (0, pad)))
        marks0 = marks.copy()
        if rng.random() < .7:
            c = int(rng.integers(w))
            thick = int(rng.integers(1, min(3, w - c) + 1))
            enlarged[h - 1:, c:c + thick] = True
            if rng.random() < .5:
                enlarged[-1, c:min(w + pad, c + thick + pad)] = True
        if rng.random() < .6:
            r = int(rng.integers(h))
            enlarged[r:min(h, r + int(rng.integers(1, 4))), w - 1:] = True
        mask = enlarged
        # Small open notches on the outside are permitted if pockets stay unchanged.
        before = {frozenset(s) for s in _cavities(set(map(tuple, np.argwhere(mask).tolist())))}
        for _ in range(3):
            trial = mask.copy()
            if rng.random() < .5:
                trial[0, int(rng.integers(trial.shape[1]))] = False
            else:
                trial[int(rng.integers(trial.shape[0])), 0] = False
            points = set(map(tuple, np.argwhere(trial).tolist()))
            if (len(_regions(points)) == 1 and before == {frozenset(s) for s in _cavities(points)}
                    and before == {frozenset(s) for s in _cavities(points, D8)}):
                mask = trial
        marks = marks0
    return _orient(rng, mask, marks)


def _open_copy(mask, marks):
    """Cut a straight passage to a pocket while keeping the beacon and body."""
    for turn in range(4):
        m = np.rot90(mask, turn)
        h, w = mask.shape
        transformed = dict(marks)
        for _ in range(turn):
            transformed = {(w - 1 - c, r): v for (r, c), v in transformed.items()}
            h, w = w, h
        points = set(map(tuple, np.argwhere(m).tolist()))
        hole = _cavities(points)[0]
        for c in sorted({c for r, c in hole}):
            r = min(r for r, cc in hole if cc == c)
            if any(cc == c and rr <= r for rr, cc in transformed):
                continue
            trial = m.copy()
            trial[:r + 1, c] = False
            ps = set(map(tuple, np.argwhere(trial).tolist()))
            if len(_regions(ps)) == 1 and not _cavities(ps) and not _cavities(ps, D8):
                return np.rot90(trial, -turn), dict(marks)
    return None


def _plain(rng):
    h, w = map(int, rng.integers(2, 7, size=2))
    mask = np.ones((h, w), bool)
    if h >= 3 and w >= 3 and rng.random() < .7:
        mask[:int(rng.integers(1, h)), 1:-1] = False
    elif rng.random() < .4:
        mask = np.ones((1, int(rng.integers(2, 8))), bool)
    return _orient(rng, mask, {})


def _local(mask, marks):
    body = set(map(tuple, np.argwhere(mask).tolist()))
    holes = _cavities(body)
    if holes:
        sizes = [len(s) for s in holes]
        if sizes.count(max(sizes)) != 1:
            return None
        hole = holes[int(np.argmax(sizes))]
        ps = hole & marks.keys()
        if len(ps) != 1:
            return None
        mark = next(iter(ps))
        control = (mark, marks[mark])
    else:
        control = None
    footprint = body | set().union(*holes) if holes else body
    if not holes:
        footprint |= marks.keys()
    return dict(mask=mask, marks=marks, body=body, holes=holes, control=control, footprint=footprint)


def _dilate(m):
    p = np.pad(m, 1)
    h, w = m.shape
    return np.logical_or.reduce([p[1+dr:1+dr+h, 1+dc:1+dc+w] for dr, dc in ((0, 0),) + D8])


def _place(rng, obj, inp, source_reserved, target_used, bg, wall, guides, moving):
    h, w = inp.shape
    mh, mw = obj['mask'].shape
    if mh > h or mw > w:
        return False
    rr, cc = np.meshgrid(np.arange(h - mh + 1), np.arange(w - mw + 1), indexing='ij')
    rr, cc = rr.ravel(), cc.ravel()
    # Reserve source bounding boxes to exclude nested and interlocking bodies.
    pref = np.pad(source_reserved.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = pref[mh:, mw:] - pref[:-mh, mw:] - pref[mh:, :-mw] + pref[:-mh, :-mw]
    valid = sums.ravel() == 0
    if obj['control']:
        mark, col = obj['control']
        axis, guide = guides[col]
        reg = rr + mark[0] if axis == 0 else cc + mark[1]
        if moving:
            valid &= reg != guide
        tr0 = np.full_like(rr, guide - mark[0]) if axis == 0 else rr
        tc0 = cc if axis == 0 else np.full_like(cc, guide - mark[1])
    else:
        tr0, tc0 = rr, cc
    for r, c in obj['marks']:
        # Only boundary signals define guides, so all local signals stay interior.
        valid &= (rr+r > 0) & (rr+r < h-1) & (cc+c > 0) & (cc+c < w-1)
    for r, c in obj['footprint']:
        tr, tc = tr0 + r, tc0 + c
        inside = (tr >= 0) & (tr < h) & (tc >= 0) & (tc < w)
        valid &= ~inside | ~target_used[np.clip(tr, 0, h-1), np.clip(tc, 0, w-1)]
    choices = np.flatnonzero(valid)
    if not len(choices):
        return False
    k = int(rng.choice(choices))
    r0, c0, dr0, dc0 = int(rr[k]), int(cc[k]), int(tr0[k]), int(tc0[k])
    for r, c in obj['body']:
        inp[r0+r, c0+c] = wall
    for (r, c), col in obj['marks'].items():
        inp[r0+r, c0+c] = col
    source_reserved[max(0, r0-1):min(h, r0+mh+1), max(0, c0-1):min(w, c0+mw+1)] = True
    for r, c in obj['footprint']:
        p = dr0+r, dc0+c
        if _inside(p, inp.shape):
            target_used[p] = True
    return True


def _make_objects(rng, phase, route_colors):
    a, b = route_colors[:2]
    if phase == 0:
        cavity = np.ones(tuple(map(int, rng.integers(3, 5, size=2))), bool)
        closed = _body(rng, [cavity], [a])
        opened = _open_copy(*closed)
        if opened is None:
            return None
        objects = [closed, opened]
        if rng.random() < .3:
            objects.append(_plain(rng))
    elif phase == 1:
        minor = np.zeros((3, 3), bool)
        minor[0, :] = True
        minor[:, 0] = True  # area 5, bounding-box area 9
        major = np.ones((2, int(rng.integers(3, 5))), bool)  # area 6 or 8
        objects = [_body(rng, [major, minor], [b, a])]
        if rng.random() < .55:
            objects.append(_plain(rng))
    else:
        n = 3 if phase == 3 else int(rng.integers(1, 3))
        pockets = []
        for j in range(n):
            h, w = map(int, rng.integers(1, 4, size=2))
            pockets.append(_pocket(rng, h, w, rng.random() < .5))
        sizes = [int(s.sum()) for s in pockets]
        if sizes.count(max(sizes)) != 1:
            return None
        cols = list(route_colors)
        rng.shuffle(cols)
        objects = [_body(rng, pockets, cols[:n])]
        for _ in range(int(rng.integers(0, 3))):
            if rng.random() < .5:
                objects.append(_plain(rng))
            else:
                small = np.ones((int(rng.integers(1, 4)), int(rng.integers(1, 4))), bool)
                objects.append(_body(rng, [small], [int(rng.choice(route_colors))]))
    result = [_local(*x) for x in objects]
    return None if any(x is None for x in result) else result


def _scene(rng, phase, palette, base_axis, clipped):
    bg, wall = palette[:2]
    routes = palette[2:5] if phase in (2, 3) else palette[2:4]
    objs = _make_objects(rng, phase, routes)
    if objs is None:
        return None
    maxh, maxw = max(o['mask'].shape[0] for o in objs), max(o['mask'].shape[1] for o in objs)
    area = sum(o['mask'].size for o in objs)
    ratio = float(np.exp(rng.uniform(-.65, .65)))
    target = area / float(rng.uniform(.18, .33))
    h = min(30, max(maxh + 2, int(round(np.sqrt(target * ratio))), 9))
    w = min(30, max(maxw + 2, int(round(np.sqrt(target / ratio))), 9))
    if rng.random() < .25:
        h, w = min(30, h + 3), min(30, w + 3)
    if maxh > h or maxw > w:
        return None
    main_color = objs[0]['control'][1]
    if phase == 0:
        # A full movement demonstration must distinguish both body-center
        # alignment and cavity-center alignment, including integer rounding.
        mark = objs[0]['control'][0]
        main_hole = next(s for s in objs[0]['holes'] if mark in s)
        for points in (objs[0]['body'], main_hole):
            center2 = min(p[base_axis] for p in points) + max(p[base_axis] for p in points)
            if mark[base_axis] in (center2 // 2, (center2 + 1) // 2):
                return None
    axes = {c: int(rng.integers(2)) for c in routes}
    axes[main_color] = base_axis if phase == 0 else 1-base_axis if phase == 1 else int(rng.integers(2))
    guides = {}
    for color in routes:
        axis = axes[color]
        options = list(range(1, (h, w)[axis] - 1))
        controlled = [o for o in objs if o['control'] and o['control'][1] == color]
        keep = []
        for coord in options:
            clips = []
            for o in controlled:
                mark = o['control'][0]
                xs = [p[axis] + coord - mark[axis] for p in o['footprint']]
                clips.append(min(xs) < 0 or max(xs) >= (h, w)[axis])
            if color == main_color and clipped:
                ok = bool(clips and clips[0])
            else:
                ok = not any(clips)
            if ok and phase == 1 and color == main_color:
                # Even the minimum two-demo puzzle must teach the smaller
                # pocket's independent fill color. Clipping may remove walls,
                # but cannot hide either teaching beacon or all its new fill.
                o = objs[0]
                origin = coord - o['control'][0][axis]
                for hole in o['holes']:
                    mark = next(iter(hole & o['marks'].keys()))
                    visible = sum(0 <= p[axis] + origin < (h, w)[axis] for p in hole)
                    ok &= 0 <= mark[axis] + origin < (h, w)[axis] and visible >= 2
            if ok:
                keep.append(coord)
        if not keep:
            return None
        guides[color] = axis, int(rng.choice(keep))
    inp = np.full((h, w), bg, np.int16)
    for color, (axis, coord) in guides.items():
        ends = [int(rng.integers(2))]
        if rng.random() < .3:
            ends = [0, 1]
        for end in ends:
            p = (coord, 0 if end == 0 else w-1) if axis == 0 else (0 if end == 0 else h-1, coord)
            if inp[p] != bg:
                return None
            inp[p] = color
    reserved = _dilate(inp != bg)
    used = inp != bg
    # Sealed carriers first; stationary bodies must avoid all transported payloads.
    order = sorted(range(len(objs)), key=lambda i: (not bool(objs[i]['control']), -objs[i]['mask'].size))
    for j in order:
        if not _place(rng, objs[j], inp, reserved, used, bg, wall, guides,
                      moving=(j == 0 or rng.random() > .15)):
            return None
    if np.mean(inp == bg) <= .5:
        return None
    try:
        scene = _decode(inp)
        out = _render(scene)
    except ValueError:
        return None
    return {'input': inp, 'output': out}


def _fallback(phase, palette, base_axis, clipped, variant):
    """Finite spacious construction preserving both mandatory teaching contrasts."""
    bg, wall, a, b, c = palette[:5]
    g = np.full((30, 30), bg, np.int16)
    if phase == 0:
        closed = np.ones((5, 5), bool)
        closed[1:4, 1:4] = False
        opened = closed.copy()
        opened[0, 1] = False
        # Seventeen positions, excluding the zero shift, leave >14 unique choices.
        origins = [r for r in range(2, 20) if r != 6]
        entries = [(closed, {(1, 2): a}, (origins[variant % len(origins)], 3)),
                   (opened, {(1, 2): a}, (20, 18))]
        anchors = {(7, 0): a, (29, 22): b}
    elif phase == 1:
        mask = np.ones((5, 9), bool)
        mask[1:3, 1:4] = False
        mask[1, 5:8] = False
        mask[1:4, 5] = False
        coord = 24 if clipped else 16
        origins = [x for x in range(2, 20) if x + 2 != coord]
        entries = [(mask, {(2, 2): b, (3, 5): a}, (3, origins[variant % len(origins)]))]
        anchors = {(0, coord): b, (18, 0): a}
    else:
        mask = np.ones((5, 12), bool)
        mask[1:3, 1:4] = False
        mask[1:3, 5:7] = False
        if phase == 3:
            mask[2, 8:10] = False
            marks = {(2, 2): a, (1, 6): b, (2, 9): c}
        else:
            marks = {(2, 2): a, (1, 6): b}
        coord = 1 if clipped else 11
        origins = [r for r in range(3, 23) if r + 2 != coord]
        entries = [(mask, marks, (origins[variant % len(origins)], 3))]
        anchors = {(coord, 0): a, (29, 22): b, (18, 29): c}
    for mask, marks, (r0, c0) in entries:
        view = g[r0:r0+mask.shape[0], c0:c0+mask.shape[1]]
        view[mask] = wall
        for (r, col), color in marks.items():
            g[r0+r, c0+col] = color
    for p, color in anchors.items():
        g[p] = color
    if base_axis:
        g = g.T.copy()
    return {'input': g, 'output': solve(g)}


def _signature(grid):
    g = np.asarray(grid, np.int8)
    return g.shape, g.tobytes()


def _counts(num_train, num_test):
    for value, lo, hi, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10)))
    stable = rng.random() < .3
    base_axis = int(rng.integers(2))
    show_clipping = bool(rng.random() < .55)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            phase = (0 if i == 0 else 1 if i == 1 else 2) if split == 'train' else 3
            colors = palette if stable else tuple(map(int, rng.permutation(10)))
            clip = show_clipping and (phase == 1 or phase >= 2 and rng.random() < .55)
            pair = None
            for _ in range(48):
                candidate = _scene(rng, phase, colors, base_axis, clip)
                if candidate is not None and _signature(candidate['input']) not in seen:
                    pair = candidate
                    break
            if pair is None:
                for variant in range(18):
                    candidate = _fallback(phase, colors, base_axis, clip, variant)
                    if _signature(candidate['input']) not in seen:
                        pair = candidate
                        break
            if pair is None:
                raise RuntimeError('finite uniqueness fallback exhausted')
            seen.add(_signature(pair['input']))
            puzzle[split].append(pair)
    return puzzle


def _features(scene):
    objects = scene['objects']
    closed = [o for o in objects if o['control']]
    open_beacon = False
    for o in objects:
        if o['pockets']:
            continue
        r0, r1 = min(r for r, c in o['body']), max(r for r, c in o['body'])
        c0, c1 = min(c for r, c in o['body']), max(c for r, c in o['body'])
        open_beacon |= any(r0 < r < r1 and c0 < c < c1 for r, c in scene['stationary'])
    bbox_counter = False
    eccentric_body, eccentric_cavity = False, False
    for o in closed:
        sizes = [len(s) for s, p, col in o['pockets']]
        boxes = [(max(r for r, c in s)-min(r for r, c in s)+1) *
                 (max(c for r, c in s)-min(c for r, c in s)+1) for s, p, col in o['pockets']]
        if len(sizes) > 1:
            bbox_counter |= boxes[int(np.argmax(sizes))] < max(boxes)
        axis = scene['guides'][o['control'][2]][0]
        mark = o['control'][1]
        if o['delta'] != (0, 0):
            for points, name in ((o['body'], 'body'), (o['control'][0], 'cavity')):
                center2 = min(p[axis] for p in points) + max(p[axis] for p in points)
                different = mark[axis] not in (center2 // 2, (center2 + 1) // 2)
                if name == 'body':
                    eccentric_body |= different
                else:
                    eccentric_cavity |= different
    return dict(objects=len(objects), sealed=len(closed), open_beacon=open_beacon,
                hole_counts=tuple(len(o['pockets']) for o in objects),
                areas=tuple(sorted(len(s) for o in objects for s, p, col in o['pockets'])),
                axes={scene['guides'][o['control'][2]][0] for o in closed},
                bbox_counter=bbox_counter,
                eccentric_body=eccentric_body, eccentric_cavity=eccentric_cavity,
                minor_fill_witness=any(col != o['paint'] and any(p != mark and _inside(
                    (p[0]+o['delta'][0], p[1]+o['delta'][1]), scene['grid'].shape) for p in hole)
                    for o in closed for hole, mark, col in o['pockets']),
                complete_mover=any(o['delta'] != (0, 0) and all(_inside(p, scene['grid'].shape)
                                  for p in _shift(o['body'] | set().union(*(s for s, p, col in o['pockets'])), o['delta'])) for o in closed),
                clips=sum(any(not _inside(p, scene['grid'].shape) for p in _shift(o['body'], o['delta'])) for o in closed))


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle keys')
    _check(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _check(inp.shape == out.shape, 'dimensions changed')
            key = _signature(inp)
            _check(key not in seen, 'duplicate input')
            seen.add(key)
            scene = _decode(inp)
            _check(np.array_equal(_render(scene), out), 'wrong routing, painting, fill, outline or erasure')
            _check(not np.array_equal(inp, out), 'unchanged example')
            for p, col in scene['stationary'].items():
                _check(out[p] == col, 'fixed signal changed')
            for obj in scene['objects']:
                if not obj['control']:
                    _check(all(out[p] == scene['wall'] for p in obj['body']), 'cavity-free body changed')
                else:
                    mark, color = obj['control'][1:]
                    axis, guide = scene['guides'][color]
                    target_mark = (mark[0]+obj['delta'][0], mark[1]+obj['delta'][1])
                    _check(target_mark[axis] == guide and target_mark[1-axis] == mark[1-axis], 'bad projection')
            features[split].append(_features(scene))
    training = features['train']
    _check(any(f['open_beacon'] for f in training), 'training lacks open-bay decoy')
    _check(any(f['complete_mover'] for f in training), 'training lacks complete rigid movement')
    _check(any(f['eccentric_body'] for f in training) and any(f['eccentric_cavity'] for f in training),
           'training permits center alignment instead of beacon registration')
    _check(any(f['bbox_counter'] for f in training), 'training permits bounding-box pocket selection')
    _check(any(f['minor_fill_witness'] for f in training), 'training hides every independent smaller-pocket fill')
    _check(set().union(*(f['axes'] for f in training)) == {0, 1}, 'training must show both axes')
    _check(all(3 in f['hole_counts'] for f in features['test']), 'missing largest-of-three test shift')
    if any(f['clips'] for f in features['test']):
        _check(any(f['clips'] for f in training), 'test clipping lacks a demonstration')
    return True


def self_test():
    """Bounded pixel-derived diversity, corruption and metamorphic checks."""
    shapes, signatures, body_shapes, area_vectors, palettes, counts = set(), set(), set(), set(), set(), set()
    clipping, open_decoys, multiple, all_axes, all_sides = 0, 0, 0, set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprint = tuple(_signature(p[k]) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output'))
        _check(fingerprint not in signatures, 'repeated puzzle across seeds')
        signatures.add(fingerprint)
        for s in ('train', 'test'):
            for pair in puzzle[s]:
                scene = _decode(pair['input'])
                f = _features(scene)
                shapes.add(scene['grid'].shape)
                counts.add(f['objects'])
                area_vectors.add(f['areas'])
                palettes.add((scene['background'], scene['wall'], tuple(sorted(scene['guides']))))
                all_axes.update(f['axes'])
                clipping += f['clips']
                open_decoys += f['open_beacon']
                multiple += sum(n >= 2 for n in f['hole_counts'])
                for ps in scene['anchors'].values():
                    for r, c in ps:
                        all_sides.update(name for name, hit in (('top', r == 0), ('bottom', r == scene['grid'].shape[0]-1),
                                                               ('left', c == 0), ('right', c == scene['grid'].shape[1]-1)) if hit)
                for o in scene['objects']:
                    r0, c0 = min(r for r, c in o['body']), min(c for r, c in o['body'])
                    body_shapes.add(frozenset((r-r0, c-c0) for r, c in o['body']))
        if seed in (0, 21, 41):
            again = generate_puzzle(seed)
            _check(all(np.array_equal(p[k], q[k]) for s in puzzle for p, q in zip(puzzle[s], again[s])
                       for k in ('input', 'output')), 'nondeterminism')
    _check(len(shapes) >= 50 and len(body_shapes) >= 150 and len(area_vectors) >= 20, 'insufficient geometric variation')
    _check(len(palettes) >= 30 and {1, 2, 3} <= counts, 'insufficient palette/count variation')
    _check(clipping >= 10 and open_decoys >= 42 and multiple >= 84, 'missing conceptual witnesses')
    _check(all_axes == {0, 1} and len(all_sides) == 4, 'orientation/edge coverage')
    # Color permutations and rotations preserve ownership, area and projection.
    rng = np.random.default_rng(777)
    p = generate_puzzle(105, 2, 1)
    for turn in (1, 2, 3):
        colors = rng.permutation(10)
        q = {s: [{k: colors[np.rot90(v, turn)] for k, v in pair.items()} for pair in p[s]] for s in p}
        validate_generated_puzzle(q)
    # Corrupt each important output zone; the validator must reject it.
    pair = p['train'][0]
    scene = _decode(pair['input'])
    obj = next(o for o in scene['objects'] if o['control'])
    locations = [next(iter(scene['stationary'])), next(iter(_shift(obj['body'], obj['delta']))),
                 next(iter(_shift(obj['pockets'][0][0], obj['delta'])))]
    halo = [x for x in _shift(obj['halo'], obj['delta']) if _inside(x, pair['output'].shape)
            and pair['output'][x] == scene['wall']]
    if halo:
        locations.append(halo[0])
    for point in locations:
        q = {s: [{k: np.asarray(v).copy() for k, v in z.items()} for z in p[s]] for s in p}
        q['train'][0]['output'][point] = (int(q['train'][0]['output'][point]) + 1) % 10
        try:
            validate_generated_puzzle(q)
        except ValueError:
            pass
        else:
            raise AssertionError('corrupted output accepted')
    # Concrete ablations of inherited mechanisms, with fixed input-visible
    # geometry. These are deliberately wrong reconstructions, not solvers.
    teaching = _fallback(0, tuple(range(10)), 0, False, 0)
    scene = _decode(teaching['input'])
    def changed_scene():
        return dict(scene, objects=[dict(o) for o in scene['objects']])
    without_a = changed_scene()
    for o in without_a['objects']:
        o['delta'] = (0, 0)
    _check(not np.array_equal(_render(without_a), teaching['output']), 'projection omission has no effect')
    without_body_recolor = changed_scene()
    for o in without_body_recolor['objects']:
        o['paint'] = scene['wall']
    _check(not np.array_equal(_render(without_body_recolor), teaching['output']), 'whole-body recoloring omission has no effect')
    without_fill = changed_scene()
    for o in without_fill['objects']:
        o['pockets'] = [({mark}, mark, col) for hole, mark, col in o['pockets']]
    _check(not np.array_equal(_render(without_fill), teaching['output']), 'cavity-fill omission has no effect')
    without_ownership = changed_scene()
    opened = next(o for o in without_ownership['objects'] if not o['control'])
    r0, r1 = min(r for r, c in opened['body']), max(r for r, c in opened['body'])
    c0, c1 = min(c for r, c in opened['body']), max(c for r, c in opened['body'])
    beacon = next(p for p in scene['stationary'] if r0 < p[0] < r1 and c0 < p[1] < c1)
    axis, guide = scene['guides'][scene['stationary'][beacon]]
    delta = [0, 0]
    delta[axis] = guide - beacon[axis]
    opened['delta'], opened['paint'] = tuple(delta), scene['stationary'][beacon]
    _check(not np.array_equal(_render(without_ownership), teaching['output']), 'open-bay ownership omission has no effect')
    # The halo actually seals this one-cell passage. Original topology must win.
    _check(not opened['pockets'] and bool(_cavities(opened['body'] | opened['halo'])),
           'outline-before-topology counterexample lost')
    # Exercise the guaranteed fallback at every teaching phase, clipping mode and axis.
    colors = tuple(range(10))
    for axis in (0, 1):
        for clip in (False, True):
            q = {'train': [_fallback(0, colors, axis, False, 0), _fallback(1, colors, axis, clip, 0)],
                 'test': [_fallback(3, colors, axis, clip, 0)]}
            validate_generated_puzzle(q)
            for phase in range(4):
                keys = {_signature(_fallback(phase, colors, axis, clip and phase != 0, i)['input']) for i in range(18)}
                _check(len(keys) >= 15, 'fallback has too few distinct inputs')
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, grid_shapes=len(shapes), body_shapes=len(body_shapes), area_vectors=len(area_vectors),
                palettes=len(palettes), object_counts=sorted(counts), clipped_objects=clipping,
                open_bay_examples=open_decoys, multi_cavity_objects=multiple)


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
