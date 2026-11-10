"""Closed-loop dye circuits. No parent imports or generation state in solve.

The large connected components are walls; isolated dots of their color are
endpoints. The remaining nonbackground color, when present, is dye. Markers
are transparent to the ORIGINAL floor topology. See task_spec.md.
"""
from __future__ import annotations

import itertools
from collections import deque

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= a) & (a <= 9)), "color bounds")
    return a.astype(np.int8, copy=False)


def _regions(mask, neighbors=N4):
    labels = np.full(mask.shape, -1, dtype=int)
    regions = []
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        k = len(regions)
        labels[r, c] = k
        queue = deque([(r, c)])
        cells = []
        while queue:
            p = queue.popleft()
            cells.append(p)
            for dr, dc in neighbors:
                rr, cc = p[0] + dr, p[1] + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and labels[rr, cc] < 0:
                    labels[rr, cc] = k
                    queue.append((rr, cc))
        regions.append(tuple(cells))
    return labels, regions


def _dilate(mask):
    h, w = mask.shape
    p = np.pad(mask, 1)
    return np.logical_or.reduce([p[1 + r:1 + r + h, 1 + c:1 + c + w] for r, c in N8])


def _clear_topology(mask):
    a, b, c, d = mask[:-1, :-1], mask[:-1, 1:], mask[1:, :-1], mask[1:, 1:]
    return not np.any((a == d) & (b == c) & (a != b))


def _holes(mask):
    padded = np.pad(mask, 1)
    labels, regions = _regions(~padded)
    return ((labels != labels[0, 0]) & ~padded)[1:-1, 1:-1]


def _segment(a, b):
    steps = abs(b[0] - a[0])
    dr = 1 if b[0] > a[0] else -1
    dc = 1 if b[1] > a[1] else -1
    return tuple((a[0] + k * dr, a[1] + k * dc) for k in range(steps + 1))


def _ray(point, slope, floor_labels, region):
    """Both directions, stopping at the first cell outside this floor region."""
    h, w = floor_labels.shape
    cells = {point}
    for sign in (-1, 1):
        r, c = point
        while True:
            r, c = r + sign, c + sign * slope
            if not (0 <= r < h and 0 <= c < w) or floor_labels[r, c] != region:
                break
            cells.add((r, c))
    return cells


def _analyze(value):
    g = _grid(value)
    h, w = g.shape
    corners = {int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])}
    _require(len(corners) == 1, "four corners must identify the background")
    bg = corners.pop()
    colors = set(map(int, np.unique(g))) - {bg}
    large = []
    for color in sorted(colors):
        _, components = _regions(g == color)
        if any(len(cells) >= 8 for cells in components):
            large.append(color)
    _require(len(large) == 1, "wall color must be recognizable by connected bodies")
    primary = large[0]
    dye_colors = colors - {primary}
    _require(len(dye_colors) <= 1, "at most one dye color")
    dye = next(iter(dye_colors)) if dye_colors else None

    _, components = _regions(g == primary)
    walls = np.zeros(g.shape, bool)
    endpoints = []
    for cells in components:
        _require(len(cells) == 1 or len(cells) >= 8, "endpoint/body size ambiguity")
        if len(cells) == 1:
            endpoints.append(cells[0])
        else:
            for p in cells:
                walls[p] = True
    _require(_clear_topology(walls), "diagonal-only topology ambiguity")
    wall_labels, bodies = _regions(walls)
    _require(len(bodies) == len(_regions(walls, N8)[1]), "diagonal wall contacts")
    near_wall = _dilate(walls)
    _require(all(not near_wall[p] for p in endpoints), "endpoints must be visibly detached")
    _require(all(max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1
                 for a, b in itertools.combinations(endpoints, 2)), "endpoint dots must be separate")

    # Dots never obstruct the floor: it is the complement of the wall bodies.
    floor_labels, floors = _regions(~walls)
    floor8, _ = _regions(~walls, N8)
    enclosed, owners = set(), {}
    exterior = np.zeros(g.shape, bool)
    for k, cells in enumerate(floors):
        edge = any(r in (0, h - 1) or c in (0, w - 1) for r, c in cells)
        if edge:
            for p in cells:
                exterior[p] = True
        else:
            enclosed.add(k)
            neighbors = set()
            for r, c in cells:
                for dr, dc in N4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and walls[rr, cc]:
                        neighbors.add(int(wall_labels[rr, cc]))
            _require(len(neighbors) == 1, "cavity must belong to one connected wall")
            owners[k] = neighbors.pop()
        _require(len({int(floor8[p]) for p in cells}) == 1, "floor connectivity")
    # Four- and eight-connected floor partitions must coincide, not just holes.
    _require(len(floors) == len(_regions(~walls, N8)[1]), "diagonally joined floor regions")

    endpoints.sort()
    segments, slopes = [], []
    degree = dict.fromkeys(endpoints, 0)
    for a, b in itertools.combinations(endpoints, 2):
        if abs(a[0] - b[0]) != abs(a[1] - b[1]) or abs(a[0] - b[0]) < 2:
            continue
        cells = _segment(a, b)
        if any(walls[p] for p in cells):
            continue  # A wall blocks pairing, even within one open floor region.
        _require(len({int(floor_labels[p]) for p in cells}) == 1, "bridge changes floor region")
        segments.append(cells)
        slopes.append(1 if b[1] > a[1] else -1)
        degree[a] += 1
        degree[b] += 1
    _require(segments and max(degree.values(), default=0) <= 1, "unique endpoint partners required")
    segment_union = set().union(*map(set, segments))
    secondary = set(map(tuple, np.argwhere(g == dye).tolist())) if dye is not None else set()
    active, ray_union, active_bodies = set(), set(), set()
    on_open = set()
    rays = []
    for p in sorted(secondary):
        memberships = [i for i, cells in enumerate(segments) if p in cells[1:-1]]
        _require(len(memberships) <= 1, "two-slope marker ambiguity")
        if not memberships:
            continue
        i = memberships[0]
        region = int(floor_labels[p])
        if region not in enclosed:
            on_open.add(p)
            continue
        line = _ray(p, -slopes[i], floor_labels, region)
        _require(line & segment_union == {p}, "dye cannot cross another primary bridge")
        _require(not line.intersection(endpoints), "dye cannot overwrite an endpoint")
        rays.append(line)
        active.add(p)
        ray_union.update(line)
        active_bodies.add(owners[region])
    active_wall = np.isin(wall_labels, list(active_bodies)) if active_bodies else np.zeros(g.shape, bool)
    outline = _dilate(active_wall) & exterior & (g == bg)
    return dict(grid=g, background=bg, primary=primary, dye=dye, walls=walls,
                wall_labels=wall_labels, bodies=bodies, floor_labels=floor_labels,
                floors=floors, enclosed=enclosed, owners=owners, exterior=exterior,
                endpoints=endpoints, unpaired={p for p, d in degree.items() if d == 0},
                segments=segments, slopes=slopes, secondary=secondary, active=active,
                on_open=on_open, rays=rays, ray_union=ray_union,
                active_bodies=active_bodies, active_wall=active_wall, outline=outline)


def _render(a):
    out = a['grid'].copy()
    for segment in a['segments']:
        for p in segment:
            if out[p] == a['background']:
                out[p] = a['primary']
    for p in a['ray_union']:
        out[p] = a['dye']
    if a['active_bodies']:
        out[a['active_wall']] = a['dye']
        out[a['outline']] = a['primary']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer all roles, topology, pairs and dye propagation from this input only."""
    return _render(_analyze(input_grid))


def _pocket(rng, h, w, irregular):
    floor = np.ones((h, w), bool)
    if irregular:
        # A stepped corner/bay; retain a >=5x5 core for readable diagonals.
        if rng.random() < .4 and min(h, w) >= 7:
            # A side bay can interrupt a diagonal which re-enters the same
            # cavity farther on. A beam must stop at its FIRST wall encounter.
            depth = int(rng.integers(1, w - 4))
            top = int(rng.integers(1, h - 2))
            bottom = min(h - 1, top + int(rng.integers(1, 3)))
            floor[top:bottom, w - depth:] = False
            if rng.random() < .5:
                floor = np.fliplr(floor).copy()
        else:
            corner = int(rng.integers(4))
            dh, dw = int(rng.integers(1, max(2, h - 4))), int(rng.integers(1, max(2, w - 4)))
            rr = slice(0, dh) if corner < 2 else slice(h - dh, h)
            cc = slice(0, dw) if corner % 2 == 0 else slice(w - dw, w)
            floor[rr, cc] = False
    return floor


def _body(rng, rooms=1, opened=(), small=False, plain=False):
    """A connected wall with independent pockets, lobes and a possible door."""
    low, high = (5, 7) if small else ((6, 7) if rooms == 3 else (7, 10))
    pockets = [_pocket(rng, int(rng.integers(low, high)), int(rng.integers(low, high)),
                       not plain and rng.random() < .55) for _ in range(rooms)]
    thickness = 1 if small or rooms == 3 else int(rng.choice([1, 1, 2]))
    gap = int(rng.integers(1, 3))
    # Pockets are side by side inside ONE conductive body, not puzzle panels.
    height = max(p.shape[0] for p in pockets) + 2 * thickness
    width = sum(p.shape[1] for p in pockets) + gap * (rooms - 1) + 2 * thickness
    mask = np.ones((height, width), bool)
    targets = []
    col = thickness
    for pocket in pockets:
        rr = thickness + int(rng.integers(height - 2 * thickness - pocket.shape[0] + 1))
        target = np.zeros(mask.shape, bool)
        target[rr:rr + pocket.shape[0], col:col + pocket.shape[1]] = pocket
        mask[target] = False
        targets.append(target)
        col += pocket.shape[1] + gap
    # Attached lobes expose whole-body propagation beyond the thin loop.
    pad = 0 if plain or small else 1
    mask = np.pad(mask, pad)
    targets = [np.pad(t, pad) for t in targets]
    if pad:
        for _ in range(int(rng.integers(1, 5))):
            side = int(rng.integers(4))
            if side < 2:
                start = int(rng.integers(1, mask.shape[1] - 1))
                stop = min(mask.shape[1] - 1, start + int(rng.integers(1, 5)))
                if side == 0:
                    mask[:2, start:stop] = True
                else:
                    mask[-2:, start:stop] = True
            else:
                start = int(rng.integers(1, mask.shape[0] - 1))
                stop = min(mask.shape[0] - 1, start + int(rng.integers(1, 5)))
                if side == 2:
                    mask[start:stop, :2] = True
                else:
                    mask[start:stop, -2:] = True
    for index in opened:
        # A single orthogonal passage to the outside; topology precedes drawing.
        rows, cols = np.nonzero(targets[index])
        top = int(rows.min())
        choices = np.flatnonzero(targets[index][top])
        col = int(rng.choice(choices))
        mask[:top + 1, col] = False
    # Random rotations/reflections also vary the pocket alignment direction.
    rotation, flip = int(rng.integers(4)), bool(rng.integers(2))
    def orient(m):
        a = np.rot90(m, rotation)
        return np.fliplr(a).copy() if flip else a.copy()
    mask, targets = orient(mask), [orient(t) for t in targets]
    _require(len(_regions(mask)[1]) == 1 and _clear_topology(mask), "body construction topology")
    return mask, targets


def _decorate(rng, mask, targets, modes, primary, dye, bg, slope=None, witness=False):
    local = np.full(mask.shape, bg, np.int8)
    local[mask] = primary
    labels, _ = _regions(~mask)
    eligible = ~mask & ~_dilate(mask)
    segments, desired_rays = [], set()
    endpoints = set()
    for room, mode in enumerate(modes):
        if mode == 'empty':
            continue
        minimum_dots = 2 if mode == 'two' else (1 if mode in ('active', 'open') else 0)
        min_length = 3 + minimum_dots
        pool = list(map(tuple, np.argwhere(targets[room] & eligible).tolist()))
        candidates = []
        for a, b in itertools.combinations(pool, 2):
            if abs(a[0] - b[0]) != abs(a[1] - b[1]) or abs(a[0] - b[0]) + 1 < min_length:
                continue
            s = 1 if b[1] > a[1] else -1
            if slope is not None and s != slope:
                continue
            cells = _segment(a, b)
            if not all(targets[room][p] for p in cells):
                continue
            candidates.append((cells, s))
        if not candidates:
            return None
        # Length is a structural draw, rather than always the full room diagonal.
        lengths = [len(cells) for cells, s in candidates]
        roll = rng.random()
        if roll < .3:
            choices = [i for i, n in enumerate(lengths) if n == max(lengths)]
        elif roll < .6:
            choices = [i for i, n in enumerate(lengths) if n == min(lengths)]
        else:
            choices = list(range(len(candidates)))
        choices = rng.permutation(choices)
        accepted = False
        for index in choices:
            cells, s = candidates[int(index)]
            if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1
                   for p in (cells[0], cells[-1]) for q in endpoints):
                continue
            interior = list(cells[1:-1])
            cap = len(cells) - 3  # retain at least one newly visible primary pixel
            number = 0 if not minimum_dots else int(rng.integers(minimum_dots, min(3, cap) + 1))
            chosen = [interior[int(i)] for i in rng.permutation(len(interior))[:number]]
            new_rays = set()
            if mode in ('active', 'two'):
                for p in chosen:
                    new_rays.update(_ray(p, -s, labels, int(labels[p])))
            previous = set().union(*map(set, segments)) if segments else set()
            if new_rays & previous or desired_rays & set(cells):
                continue
            segments.append(cells)
            endpoints.update((cells[0], cells[-1]))
            desired_rays.update(new_rays)
            local[cells[0]] = local[cells[-1]] = primary
            for p in chosen:
                local[p] = dye
            accepted = True
            break
        if not accepted:
            return None
    # A visible inert dot inside a closed cavity disproves unconditional emission.
    if witness:
        first_slope = 1 if segments[0][-1][1] > segments[0][0][1] else -1
        pool = [tuple(p) for p in np.argwhere(targets[0]).tolist()
                if local[tuple(p)] == bg and tuple(p) not in desired_rays
                and not any(tuple(p) in cells for cells in segments)
                and any(q != tuple(p) and local[q] == bg and q not in desired_rays
                        for q in _ray(tuple(p), -first_slope, labels, int(labels[tuple(p)])))]
        if not pool:
            return None
        local[pool[int(rng.integers(len(pool)))]] = dye
    elif rng.random() < .35:
        pool = [tuple(p) for t in targets for p in np.argwhere(t).tolist()
                if local[tuple(p)] == bg and not any(tuple(p) in cells for cells in segments)]
        if pool:
            local[pool[int(rng.integers(len(pool)))]] = dye
    return local, segments


def _pack(rng, parts, gap, margin):
    area = sum(p.size for p in parts)
    aspect = float(np.exp(rng.uniform(-.65, .65)))
    target = area / float(rng.uniform(.4, .65))
    h = min(30, max(max(p.shape[0] for p in parts) + 2 * margin, int(np.sqrt(target * aspect))))
    w = min(30, max(max(p.shape[1] for p in parts) + 2 * margin, int(np.sqrt(target / aspect))))
    if h < max(p.shape[0] for p in parts) or w < max(p.shape[1] for p in parts):
        return None
    for shape in ((h, w), (min(30, h + 5), min(30, w + 5)), (30, 30)):
        h, w = shape
        order = sorted(range(len(parts)), key=lambda i: -parts[i].size)
        for _ in range(6):
            used = np.zeros(shape, bool)
            placements = [None] * len(parts)
            for i in order:
                ph, pw = parts[i].shape
                candidates = []
                for r in range(margin, h - ph - margin + 1):
                    for c in range(margin, w - pw - margin + 1):
                        # All corners visibly encode the background.
                        if (r == 0 and c == 0 or r == 0 and c + pw == w
                                or r + ph == h and c == 0 or r + ph == h and c + pw == w):
                            continue
                        if not used[r:r + ph, c:c + pw].any():
                            candidates.append((r, c))
                if not candidates:
                    break
                r, c = candidates[int(rng.integers(len(candidates)))]
                placements[i] = (r, c)
                used[max(0, r - gap):min(h, r + ph + gap),
                     max(0, c - gap):min(w, c + pw + gap)] = True
            else:
                return shape, placements
    # Finite shelf packing is a capacity fallback, never a normal spatial template.
    for order in itertools.permutations(range(len(parts))):
        placements = [None] * len(parts)
        r, c, shelf = 1, 1, 0
        for i in order:
            ph, pw = parts[i].shape
            if c + pw > 29:
                r, c, shelf = r + shelf + 1, 1, 0
            if r + ph > 29:
                break
            placements[i] = (r, c)
            c += pw + 1
            shelf = max(shelf, ph)
        else:
            return (30, 30), placements
    return None


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _fallback_pair(kind, primary, dye, bg, slope, used):
    """Finite emergency geometry with >14 distinct inputs and all witnesses."""
    g = np.full((30, 30), bg, np.int8)
    s = slope or 1

    def body(r, c, rooms=1, opened=False):
        width = 9 if rooms == 1 else (21 if rooms == 2 else 28)
        offsets = (1,) if rooms == 1 else ((1, 13) if rooms == 2 else (1, 10, 19))
        g[r:r + 9, c:c + width] = primary
        for offset in offsets:
            g[r + 1:r + 8, c + offset:c + offset + 7] = bg
        if opened:
            g[r, c + offsets[-1] + 3] = bg
        return [(r + 1, c + offset) for offset in offsets]

    def bridge(origin, triggers=0):
        r, c = origin
        a = (r + 1, c + (1 if s == 1 else 5))
        b = (r + 5, c + (5 if s == 1 else 1))
        cells = _segment(a, b)
        g[a] = g[b] = primary
        for k in ((2,) if triggers == 1 else (1, 3) if triggers == 2 else ()):
            g[cells[k]] = dye

    if kind == 'contrast':
        left = body(2, 2)[0]
        right = body(2, 16, opened=True)[0]
        bridge(left, 1)
        bridge(right, 1)
        g[4, 7 if s == 1 else 5] = dye
    elif kind == 'shared':
        targets = body(2, 2, 2)
        bridge(targets[0], 2)
        bridge(targets[1])
        bridge(body(17, 2)[0])
        g[4, 7 if s == 1 else 5] = dye
    elif kind == 'three':
        for i, target in enumerate(body(2, 1, 3)):
            bridge(target, 1 if i == 0 else 0)
    elif kind == 'mixed':
        targets = body(2, 2, 2, opened=True)
        bridge(targets[0], 1)
        bridge(targets[1], 1)
    else:
        bridge(body(2, 2)[0], 0 if kind == 'dry' else 1)
    a = _analyze(g)
    # One extra unpaired dot varies the input without adding an emission or
    # breaking a dry query. A single choice suffices: at most 13 are already used.
    pool = [tuple(p) for p in np.argwhere(a['exterior'] & ~_dilate(a['walls']) & (g == bg)).tolist()
            if p[0] >= 27 and 1 <= p[1] <= 28
            and all(abs(p[0] - q[0]) != abs(p[1] - q[1]) for q in a['endpoints'])
            and all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1 for q in a['endpoints'])]
    _require(len(pool) >= 14, 'fallback uniqueness capacity')
    for p in pool:
        candidate = g.copy()
        candidate[p] = primary
        if _key(candidate) not in used:
            return {'input': candidate, 'output': solve(candidate)}
    raise AssertionError('maximum puzzle has only fourteen pairs')


def _make_pair(rng, kind, primary, dye, bg, slope, used, allow_reentry=True):
    """The two teaching witnesses have requirements, but no fixed positions."""
    for attempt in range(140):
        plain = attempt >= 90
        if kind == 'contrast':
            # Reuse a silhouette and its targets; cut an actual door in one copy.
            mask, targets = _body(rng, plain=plain)
            opened = mask.copy()
            # Open along a visible orthogonal route from the pocket to an edge.
            t = targets[0]
            rr, cc = np.argwhere(t)[0]
            opened[:int(rr) + 1, int(cc)] = False
            if len(_regions(opened)[1]) != 1 or not _clear_topology(opened) or _holes(opened).any():
                continue
            specs = [(mask, targets, ['active'], True), (opened, targets, ['open'], False)]
        elif kind == 'shared':
            a, ts = _body(rng, rooms=2, plain=plain)
            b, us = _body(rng, small=True, plain=plain)
            specs = [(a, ts, ['two', 'zero'], True), (b, us, ['zero'], False)]
        elif kind == 'three':
            a, ts = _body(rng, rooms=3, plain=plain)
            modes = ['active', str(rng.choice(['zero', 'active', 'empty'])), 'zero']
            rng.shuffle(modes)
            specs = [(a, ts, modes, False)]
            if rng.random() < .5:
                b, us = _body(rng, small=True, plain=plain)
                specs.append((b, us, ['zero'], False))
        elif kind == 'mixed':
            a, ts = _body(rng, rooms=2, opened=(1,), plain=plain)
            specs = [(a, ts, ['active', 'open'], False)]
            if rng.random() < .5:
                b, us = _body(rng, small=True, plain=plain)
                specs.append((b, us, ['zero'], False))
        else:
            specs = []
            count = int(rng.choice([1, 1, 2, 2, 3, 4]))
            for _ in range(count):
                small = count >= 3 or kind == 'dry'
                rooms = 2 if not small and rng.random() < .3 else 1
                # Shared open/sealed bodies are reserved for the recombination
                # query. Ordinary training bodies are all-open or all-closed.
                opening = tuple(range(rooms)) if rng.random() < .3 else ()
                a, ts = _body(rng, rooms=rooms, opened=opening, small=small, plain=plain)
                modes = [('zero' if kind == 'dry' or small else
                          'open' if j in opening else str(rng.choice(['active', 'zero', 'empty'])))
                         for j in range(rooms)]
                specs.append((a, ts, modes, False))
        if kind in ('contrast', 'shared') and rng.random() < .35:
            a, ts = _body(rng, small=True, plain=plain)
            specs.append((a, ts, ['zero'], False))
        parts, expected_local = [], []
        for mask, targets, modes, witness in specs:
            decorated = _decorate(rng, mask, targets, modes, primary, dye, bg,
                                   slope=slope if kind in ('contrast', 'shared') else None,
                                   witness=witness)
            if decorated is None:
                break
            part, segs = decorated
            if kind == 'dry':
                part[part == dye] = bg
            parts.append(part)
            expected_local.append(segs)
        else:
            packed = _pack(rng, parts, int(rng.choice([1, 1, 2, 3])), int(rng.choice([0, 1, 1, 2])))
            if packed is None:
                continue
            shape, placements = packed
            inp = np.full(shape, bg, np.int8)
            expected = set()
            for part, segs, (r, c) in zip(parts, expected_local, placements):
                ph, pw = part.shape
                inp[r:r + ph, c:c + pw] = part
                expected.update(tuple((rr + r, cc + c) for rr, cc in s) for s in segs)
            if not expected:
                continue
            try:
                a = _analyze(inp)
            except AssertionError:
                continue
            if set(a['segments']) != expected:
                continue
            if kind == 'ordinary' and a['dye'] is None:
                # Dry queries introduce absence of dye as a single-factor shift.
                pool = [tuple(p) for p in np.argwhere(a['exterior'] & (inp == bg)).tolist()
                        if tuple(p) not in ((0, 0), (0, shape[1] - 1), (shape[0] - 1, 0), (shape[0] - 1, shape[1] - 1))
                        and not any(tuple(p) in cells for cells in a['segments'])]
                if not pool:
                    continue
                inp[pool[int(rng.integers(len(pool)))]] = dye
                a = _analyze(inp)
            # An exterior unpaired endpoint makes its sparse marker role explicit.
            if kind == 'contrast' or rng.random() < .25:
                pool = [tuple(p) for p in np.argwhere(a['exterior'] & (inp == bg) & ~_dilate(a['walls'])).tolist()
                        if tuple(p) not in set().union(*map(set, a['segments']))
                        and all(abs(p[0] - q[0]) != abs(p[1] - q[1]) for q in a['endpoints'])
                        and all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) > 1 for q in a['endpoints'])
                        and tuple(p) not in ((0, 0), (0, shape[1] - 1), (shape[0] - 1, 0), (shape[0] - 1, shape[1] - 1))]
                if pool:
                    inp[pool[int(rng.integers(len(pool)))]] = primary
                    a = _analyze(inp)
                elif kind == 'contrast':
                    continue
            if kind == 'contrast' and (not a['active'] or not a['on_open']):
                continue
            if kind == 'shared' and (len(a['active']) < 2 or len(a['active_bodies']) != 1):
                continue
            if kind == 'mixed' and (not a['active'] or not a['on_open']):
                continue
            if not allow_reentry and _reentry(a):
                continue
            if _key(inp) in used:
                if kind == 'dry':
                    continue
                # Unconditionally inert exterior dye dots provide finite uniqueness
                # without changing the transformation or the teaching witnesses.
                for p in map(tuple, np.argwhere(a['exterior'] & (inp == bg) & ~a['outline']).tolist()):
                    if p in ((0, 0), (0, shape[1] - 1), (shape[0] - 1, 0), (shape[0] - 1, shape[1] - 1)):
                        continue
                    if any(p in s for s in a['segments']):
                        continue
                    candidate = inp.copy()
                    candidate[p] = dye
                    if _key(candidate) not in used:
                        inp, a = candidate, _analyze(candidate)
                        break
                else:
                    continue
            out = _render(a)
            if np.array_equal(inp, out):
                continue
            return {'input': inp, 'output': out}
    return _fallback_pair(kind, primary, dye, bg, slope, used)


def _counts(num_train, num_test):
    for name, n, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not lo <= int(n) <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    primary, dye = palette[:2]
    variable_bg = bool(rng.integers(2))
    query_kind = str(rng.choice(['three', 'mixed', 'dry']))
    slope = int(rng.choice([-1, 1]))
    kinds = [('contrast', slope), ('shared', -slope)]
    kinds += [('ordinary', None) for _ in range(nt - 2)]
    rng.shuffle(kinds)
    puzzle, used = {'train': [], 'test': []}, set()
    allow_reentry = True
    for split, requests in (('train', kinds), ('test', [(query_kind, None)] * nq)):
        for kind, needed_slope in requests:
            bg = int(rng.choice(palette[2:])) if variable_bg else palette[2]
            pair = _make_pair(rng, kind, primary, dye, bg, needed_slope, used, allow_reentry)
            used.add(_key(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
        if split == 'train':
            allow_reentry = any(_reentry(_analyze(pair['input'])) for pair in puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    holes_per_body = [sum(a['owners'][k] == i for k in a['enclosed']) for i in range(len(a['bodies']))]
    triggered_regions = {int(a['floor_labels'][p]) for p in a['active']}
    inactive_closed = a['secondary'] - a['active'] - a['ray_union']
    return dict(shape=a['grid'].shape, bodies=len(a['bodies']), holes=len(a['enclosed']),
                holes_per_body=tuple(sorted(holes_per_body)), segments=len(a['segments']),
                lengths=tuple(sorted(map(len, a['segments']))), active=len(a['active']),
                open_trigger=len(a['on_open']), unpaired=len(a['unpaired']),
                inert_closed=sum(int(a['floor_labels'][p]) in a['enclosed'] for p in inactive_closed),
                active_slopes={s for seg, s in zip(a['segments'], a['slopes']) if set(seg) & a['active']},
                dry_body=any(i not in a['active_bodies'] and n > 0 for i, n in enumerate(holes_per_body)),
                shared_untriggered=any(a['owners'][k] in a['active_bodies'] and k not in triggered_regions
                                       for k in a['enclosed']),
                irregular=sum(len(a['floors'][k]) != ((max(r for r, c in a['floors'][k]) - min(r for r, c in a['floors'][k]) + 1)
                              * (max(c for r, c in a['floors'][k]) - min(c for r, c in a['floors'][k]) + 1))
                              for k in a['enclosed']),
                clipped=bool(a['walls'][0].any() or a['walls'][-1].any()
                             or a['walls'][:, 0].any() or a['walls'][:, -1].any()),
                absent_dye=a['dye'] is None, reentry=_reentry(a))


def _reentry(a):
    """Does a blocked ray have more cells of its original cavity beyond a wall?"""
    h, w = a['grid'].shape
    for p in a['active']:
        i = next(i for i, cells in enumerate(a['segments']) if p in cells)
        slope, region = -a['slopes'][i], int(a['floor_labels'][p])
        for sign in (-1, 1):
            r, c, blocked = p[0], p[1], False
            while True:
                r, c = r + sign, c + sign * slope
                if not (0 <= r < h and 0 <= c < w):
                    break
                if a['floor_labels'][r, c] != region:
                    blocked = True
                elif blocked:
                    return True
    return False


def _ablation_changes(a):
    """Two precise shortcuts, tested against returned training pixels.

    Remove A's enclosure gate: on-bridge exterior dots emit through their floor.
    Remove B's between-endpoints test: every dye dot in a paired sealed room
    emits along that room's demonstrated opposite diagonal.
    These are counter-rules, not independent semantic certification.
    """
    intended = _render(a)
    no_a, no_b = intended.copy(), intended.copy()
    if a['dye'] is None:
        return False, False
    for p in a['on_open']:
        i = next(i for i, cells in enumerate(a['segments']) if p in cells)
        region = int(a['floor_labels'][p])
        for q in _ray(p, -a['slopes'][i], a['floor_labels'], region):
            no_a[q] = a['dye']
    for p in a['secondary'] - a['active']:
        region = int(a['floor_labels'][p])
        if region not in a['enclosed']:
            continue
        for cells, slope in zip(a['segments'], a['slopes']):
            if int(a['floor_labels'][cells[0]]) == region:
                for q in _ray(p, -slope, a['floor_labels'], region):
                    no_b[q] = a['dye']
    return not np.array_equal(intended, no_a), not np.array_equal(intended, no_b)


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists')
    try:
        _counts(len(puzzle['train']), len(puzzle['test']))
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc
    seen, roles = set(), set()
    features, test_features, ablations = [], [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, 'dimensions must be preserved')
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            a = _analyze(inp)
            roles.add((a['primary'], a['dye']))
            _require(np.array_equal(_render(a), out), 'incorrect dye-circuit output')
            _require(not np.array_equal(inp, out), 'each example must expose a change')
            # Zone invariants reconstructed from the actual returned grids.
            _require(all(out[p] == inp[p] for p in a['endpoints']), 'endpoint preservation')
            _require(all(out[p] == inp[p] for p in a['secondary']), 'original dye preservation')
            _require(np.all(out[a['walls'] & ~a['active_wall']] == a['primary']), 'inactive walls changed')
            _require(not np.any(a['outline'] & ~a['exterior']), 'outline inside cavity')
            changed = out != inp
            allowed = a['active_wall'] | a['outline']
            for cells in a['segments'] + a['rays']:
                for p in cells:
                    allowed[p] = True
            _require(not np.any(changed & ~allowed), 'paint outside authorized zones')
            if split == 'train':
                features.append(_features(a))
                ablations.append(_ablation_changes(a))
            else:
                test_features.append(_features(a))
    # A missing dye role is legal in individual pairs, but roles stay coherent.
    primaries = {p for p, s in roles}
    dyes = {s for p, s in roles if s is not None}
    _require(len(primaries) == len(dyes) == 1, 'puzzle role consistency')
    _require(set().union(*(f['active_slopes'] for f in features)) == {-1, 1}, 'teach both slopes')
    for name in ('open_trigger', 'inert_closed', 'dry_body', 'shared_untriggered', 'unpaired'):
        _require(any(f[name] for f in features), 'missing teaching witness: ' + name)
    _require(any(f['active'] >= 2 for f in features), 'teach multiple triggers')
    _require(all(any(changes[i] for changes in ablations) for i in (0, 1)),
             'training must distinguish omission of A and of B')
    _require(not any(f['reentry'] for f in test_features) or any(f['reentry'] for f in features),
             'first-wall occlusion with re-entry must be demonstrated')


def _fingerprint(p):
    return tuple(tuple((_key(x['input']), _key(x['output'])) for x in p[s]) for s in ('train', 'test'))


def _regression():
    # Hand-specified sealed/open twins, with known finite and bounded strokes.
    g = np.zeros((13, 25), np.int8)
    for c0 in (1, 14):
        g[2:11, c0:c0 + 9] = 5
        g[3:10, c0 + 1:c0 + 8] = 0
        g[4, c0 + 2] = g[8, c0 + 6] = 5
        g[6, c0 + 4] = 2
    g[2, 18] = 0  # The right twin's one-cell door makes its bay exterior.
    g[4, 6] = 2   # An off-segment dye dot, also outside the active ray.
    out = solve(g)
    _require(np.all(out[2, 1:10] == 2), 'whole sealed wall must take dye')
    _require(out[2, 14] == 5 and out[7, 19] == 5, 'open wall preserved, bridge completed')
    _require(out[4, 3] == 5 and out[5, 4] == 5 and out[6, 5] == 2 and out[8, 7] == 5,
             'finite primary bridge')
    _require(out[3, 8] == 2 and out[9, 2] == 2 and out[1, 10] == 5,
             'bounded opposite ray and exterior outline')
    _require(out[3, 3] == 0 and out[4, 6] == 2 and out[3, 7] == 0, 'no whole-cavity fill or inert launch')
    _require(out[2, 18] == 0 and out[5, 17] == 5, 'original open topology, no open dye emission')
    # Color equivariance: roles are geometry-derived, including zero as wall/dye.
    for permutation in (np.arange(10)[::-1], np.array([3, 8, 0, 4, 6, 1, 2, 9, 7, 5])):
        _require(np.array_equal(solve(permutation[g]), permutation[out]), 'palette equivariance')
    for k in range(4):
        _require(np.array_equal(solve(np.rot90(g, k)), np.rot90(out, k)), 'rotation equivariance')
    _require(np.array_equal(solve(np.fliplr(g)), np.fliplr(out)), 'reflection equivariance')
    # A concave cavity: the diagonal meets a wall and would later re-enter.
    notch = np.zeros((15, 16), np.int8)
    notch[2:13, 2:13] = 5
    notch[3:12, 3:12] = 0
    notch[5:7, 9:12] = 5
    notch[5, 6] = notch[9, 10] = 5
    notch[7, 8] = 2
    predicted = solve(notch)
    _require(predicted[4, 11] == 0 and predicted[8, 7] == 2,
             'first-wall termination; no re-entry after an occluding wall')
    _require(predicted[6, 7] == 5 and predicted[8, 9] == 5, 'primary finite bridge through concave room')
    return g


def self_test():
    _regression()
    fingerprints, shapes, layouts, roles, counts, lengths = set(), set(), set(), set(), set(), set()
    aggregate = {k: 0 for k in ('open_trigger', 'irregular', 'clipped', 'absent_dye')}
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _analyze(pair['input'])
                f = _features(a)
                shapes.add(f['shape'])
                roles.add((a['primary'], a['dye']))
                counts.add((f['bodies'], f['holes'], f['segments']))
                lengths.update(f['lengths'])
                norm = np.zeros(a['grid'].shape, np.int8)
                norm[a['walls']] = 1
                for p in a['endpoints']:
                    norm[p] = 2
                for p in a['secondary']:
                    norm[p] = 3
                layouts.add(_key(norm))
                for key in aggregate:
                    aggregate[key] += bool(f[key])
                # Derived masks also ensure no rays escape a sealed floor region.
                for line in a['rays']:
                    _require(len({int(a['floor_labels'][p]) for p in line}) == 1, 'ray leaks')
                _require(np.array_equal(solve(pair['input'].copy()), pair['output']), 'input-only solve')
        if seed in (0, 21, 41):
            _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), 'determinism')
    _require(len(fingerprints) == 42, '42 distinct seed puzzles')
    _require(len(shapes) >= 25 and len(layouts) >= 170, 'geometric/layout variation')
    _require(len(roles) >= 25 and len(counts) >= 12 and len(lengths) >= 5, 'role/count/length variation')
    _require(all(n >= 3 for n in aggregate.values()), 'missing structural variation')
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            validate_generated_puzzle(generate_puzzle(917 + nt * 10 + nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Validation must reject wrong labels, including a changed original marker.
    p = generate_puzzle(0, 2, 1)
    for zone in ('active_wall', 'outline', 'ray_union', 'endpoints'):
        bad = {s: [{k: v.copy() for k, v in x.items()} for x in p[s]] for s in p}
        for pair in bad['train']:
            a = _analyze(pair['input'])
            cells = list(map(tuple, np.argwhere(a[zone]).tolist())) if zone in ('active_wall', 'outline') else list(a[zone])
            if cells:
                point = cells[0]
                pair['output'][point] = (int(pair['output'][point]) + 1) % 10
                break
        try:
            validate_generated_puzzle(bad)
        except AssertionError:
            continue
        raise AssertionError('corrupted ' + zone + ' accepted')
    return dict(seeds=42, distinct_puzzles=len(fingerprints), shapes=len(shapes),
                normalized_inputs=len(layouts), role_assignments=len(roles),
                count_profiles=len(counts), lengths=sorted(lengths), feature_pairs=aggregate)


if __name__ == '__main__':
    print(self_test())
