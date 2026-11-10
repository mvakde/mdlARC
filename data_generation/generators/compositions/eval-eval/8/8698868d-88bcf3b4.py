"""Cavity-addressed spools: topology chooses an anchor, material becomes cord.

Only NumPy and the standard library are used. No files, seeds, examples or
construction metadata are consulted by solve(). Diagnostics stay in memory.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from itertools import permutations

import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _choose(rng, values):
    return values[int(rng.integers(len(values)))]


def _components(cells, diagonal=False):
    unseen = set(cells)
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        part, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in D8 if diagonal else D4:
                p = r + dr, c + dc
                if p in unseen:
                    unseen.remove(p)
                    part.add(p)
                    stack.append(p)
        result.append(part)
    return result


def _bbox(cells):
    return (min(r for r, c in cells), max(r for r, c in cells),
            min(c for r, c in cells), max(c for r, c in cells))


def _global(p, end, forward, lateral):
    u, v = p
    return (end[0] + u * forward[0] + v * lateral[0],
            end[1] + u * forward[1] + v * lateral[1])


def _local(p, end, forward, lateral):
    r, c = p[0] - end[0], p[1] - end[1]
    return r * forward[0] + c * forward[1], r * lateral[0] + c * lateral[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= s <= 30 for s in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _route(obstacle, length):
    """One forward step per cord cell; actual-cell exterior clearance envelope.

    Local endpoint is (0,0), attachment (0,1), ticket behind/left (u<0,v<=0).
    Admitted convex digital contours have slopes +1,0,-1 in that order.
    """
    peak = max(v for u, v in obstacle)
    far = max(u for u, v in obstacle)
    assert min(u for u, v in obstacle) >= 2 and peak >= 0
    assert max(v - u for u, v in obstacle) == 0 or peak == 0
    first_peak = min(u for u, v in obstacle if v == peak)
    route = []
    for u in range(max(length, far + 4)):
        v = max(b + 1 - abs(u - a) for a, b in obstacle)
        if peak == 0 and u <= first_peak:
            v = 1
        route.append((u, v))
    slopes = [b[1] - a[1] for a, b in zip(route, route[1:])]
    assert route[0] == (0, 1)
    assert all(s in (-1, 0, 1) for s in slopes)
    assert slopes == sorted(slopes, reverse=True)
    assert not set(route) & obstacle
    return route[:length]


def _ticket(cells):
    """Recognize a punctured rectangle with a two-cell stalk at one corner."""
    if len(_components(cells)) != 1:
        return None
    interpretations = []
    for f in D4:
        front = max(r * f[0] + c * f[1] for r, c in cells)
        tips = [p for p in cells if p[0] * f[0] + p[1] * f[1] == front]
        if len(tips) != 1:
            continue
        end = tips[0]
        for l in ((f[1], -f[0]), (-f[1], f[0])):
            local = {_local(p, end, f, l) for p in cells}
            u0, u1, v0, v1 = _bbox(local)
            h, w = -u0 - 1, 1 - v0
            if not (3 <= h <= 6 and 3 <= w <= 6 and u1 == v1 == 0):
                continue
            rectangle = {(u, v) for u in range(-h - 1, -1) for v in range(1 - w, 1)}
            holes = rectangle - local
            expected = rectangle | {(-1, 0), (0, 0)}
            if local != expected - holes or len(holes) > 4:
                continue
            if any(u in (-h - 1, -2) or v in (1 - w, 0) for u, v in holes):
                continue
            if any((u + du, v + dv) in holes for u, v in holes for du, dv in D4):
                continue
            interpretations.append(dict(end=end, f=f, l=l, count=len(holes),
                                        cells=cells, plate=(h, w), clues=holes))
    if len(interpretations) == 1:
        return interpretations[0]
    return None


def _spool(cells):
    r0, r1, c0, c1 = _bbox(cells)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    assert 3 <= h <= 6 and 3 <= w <= 6 and 8 <= len(cells) <= 24
    assert len(_components(cells)) == 1
    rectangle = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    holes = rectangle - cells
    assert all(r0 < r < r1 and c0 < c < c1 for r, c in holes)
    parts = _components(holes)
    assert len(parts) <= 4
    return dict(cells=cells, count=len(parts), length=len(cells), holes=holes,
                cavity_sizes=tuple(sorted(map(len, parts))),
                eight_count=len(_components(holes, True)), box=(r0, r1, c0, c1))


def _decode(input_grid):
    a = _grid(input_grid)
    counts = np.bincount(a.ravel(), minlength=10)
    background = int(counts.argmax())
    assert counts[background] > a.size / 2, "background must be a strict majority"
    records = []
    for color in range(10):
        if color != background:
            for cells in _components(map(tuple, np.argwhere(a == color).tolist()), True):
                records.append((color, cells))
    tickets = []
    for index, (color, cells) in enumerate(records):
        t = _ticket(cells)
        if t is not None:
            t.update(color=color, index=index)
            tickets.append(t)
    assert 1 <= len(tickets) <= 3
    assert len({t['color'] for t in tickets}) == len(tickets)
    assert len({t['count'] for t in tickets}) == len(tickets)
    used = set()
    for t in tickets:
        same = [i for i, (c, cells) in enumerate(records) if c == t['color'] and i != t['index']]
        assert len(same) == 1, "each ticket has one separate same-color obstacle"
        oi = same[0]
        obstacle = records[oi][1]
        assert len(obstacle) <= 18
        assert len(_components(obstacle, True)) == 1
        t['obstacle'] = obstacle
        t['local_obstacle'] = {_local(p, t['end'], t['f'], t['l']) for p in obstacle}
        _route(t['local_obstacle'], 2)
        used.update((oi, t['index']))
    spools = []
    for i, (color, cells) in enumerate(records):
        if i not in used:
            s = _spool(cells)
            s['color'] = color
            assert color not in {t['color'] for t in tickets}
            spools.append(s)
    assert len(spools) == len(tickets)
    assert len({s['count'] for s in spools}) == len(spools)
    assert {s['count'] for s in spools} == {t['count'] for t in tickets}
    # Different input components must not touch, even when differently colored.
    occupied = set()
    for color, cells in records:
        assert not cells & occupied
        occupied.update((r + dr, c + dc) for r, c in cells for dr, dc in D8 + ((0, 0),))
    for t in tickets:
        s = next(s for s in spools if s['count'] == t['count'])
        t['spool'] = s
        t['path'] = [_global(p, t['end'], t['f'], t['l'])
                     for p in _route(t['local_obstacle'], s['length'])]
        t['visible'] = {p for p in t['path'] if _inside(p, a.shape)}
        assert len(t['visible']) >= 3 and _inside(t['path'][0], a.shape)
    return a, background, tickets, spools


def _render(a, background, tickets, spools):
    out = a.copy()
    for s in spools:
        for p in s['cells']:
            out[p] = background
    occupied = set()
    for t in tickets:
        for p in t['visible']:
            assert a[p] == background and p not in occupied, "cord collision"
            out[p] = t['spool']['color']
        occupied.update(t['visible'])
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the visible addresses, amounts and geometry from this input only."""
    a, bg, tickets, spools = _decode(input_grid)
    return _render(a, bg, tickets, spools)


def _isolated_clues(rng, h, w, k):
    candidates = [(u, v) for u in range(-h, -2) for v in range(2 - w, 0)]
    rectangle = {(u, v) for u in range(-h - 1, -1) for v in range(1 - w, 1)}
    for _ in range(16):
        chosen = set()
        for index in rng.permutation(len(candidates)):
            p = candidates[int(index)]
            if not any((p[0] + du, p[1] + dv) in chosen for du, dv in D4):
                if len(_components(rectangle - chosen - {p})) != 1:
                    continue
                chosen.add(p)
                if len(chosen) == k:
                    return chosen
        if k == 0:
            return set()
    # Checkerboard fallback only; ordinary draws use arbitrary independent sets.
    chosen = [p for p in candidates if (p[0] + p[1]) % 2 == 0][:k]
    assert len(chosen) == k
    assert len(_components(rectangle - set(chosen))) == 1
    return set(chosen)


def _make_spool(rng, k, witness=False, large=False):
    if witness:
        mask = np.ones((4, 4), dtype=bool)
        if k == 1:
            mask[1:3, 1:3] = False
        else:
            mask[1, 1] = mask[2, 2] = False
        return np.rot90(mask, int(rng.integers(4))).copy()
    if large and k == 1:
        # A visible scale shift: every such query has a six-cell spool axis.
        w = int(rng.integers(4, 7))
        mask = np.ones((6, w), dtype=bool)
        mask[1:-1, 1:-1] = False
        return np.rot90(mask, int(rng.integers(4))).copy()
    if k == 0:
        h, w = _choose(rng, [(3, 3), (3, 4), (4, 3), (4, 4)])
        return np.ones((h, w), dtype=bool)
    minimum = 5 if k >= 3 else 4
    for _ in range(64):
        h = int(rng.integers(minimum, 7 if large else 6))
        w = int(rng.integers(minimum, 7 if large else 6))
        mask = np.zeros((h, w), dtype=bool)  # True means background cavity.
        if k == 1 and rng.random() < .45:
            mask[1:-1, 1:-1] = True
        else:
            inner = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
            for idx in rng.permutation(len(inner)):
                p = inner[int(idx)]
                if not any(0 <= p[0] + dr < h and 0 <= p[1] + dc < w
                           and mask[p[0] + dr, p[1] + dc] for dr, dc in D4):
                    mask[p] = True
                    if int(mask.sum()) == k:
                        break
            if int(mask.sum()) != k:
                continue
            target = int(rng.integers(max(k, h * w - 24), (h - 2) * (w - 2) + 1))
            for idx in rng.permutation(len(inner)):
                if int(mask.sum()) >= target:
                    break
                p = inner[int(idx)]
                trial = mask.copy()
                trial[p] = True
                holes = set(map(tuple, np.argwhere(trial).tolist()))
                fg = set(map(tuple, np.argwhere(~trial).tolist()))
                if len(_components(holes)) == k and len(_components(fg)) == 1:
                    mask = trial
        cells = set(map(tuple, np.argwhere(~mask).tolist()))
        if 8 <= len(cells) <= 24 and len(_components(cells)) == 1:
            if len(_components(map(tuple, np.argwhere(mask).tolist()))) == k:
                return ~mask
    # Finite, valid construction: four- or five-square with k separated holes.
    size = 4 if k <= 2 else 5
    result = np.ones((size, size), dtype=bool)
    positions = [(1, 1), (2, 2)] if k <= 2 else [(1, 1), (1, 3), (3, 1), (3, 3)]
    for p in positions[:k]:
        result[p] = False
    return result


def _make_obstacle(rng, kind=None, parallel=None, remote=False):
    kind = int(rng.integers(4)) if kind is None else kind
    if kind == 0:
        raw = {(0, 0)}
    elif kind == 1:
        h, w = int(rng.integers(1, 4)), int(rng.integers(1, 3))
        raw = {(u, v) for u in range(h) for v in range(w)}
    elif kind == 2:
        radius = int(rng.integers(1, 3))
        raw = {(u, radius - abs(u - radius)) for u in range(2 * radius + 1)}
        if rng.random() < .5:
            raw |= {(u, v) for u in range(2 * radius + 1)
                    for v in range(radius - abs(u - radius) + 1)}
    else:
        plateau, ramp = int(rng.integers(1, 4)), int(rng.integers(1, 3))
        raw = {(u, v) for u in range(plateau + ramp)
               for v in range(ramp - max(0, u - plateau + 1) + 1)}
    near = int(rng.integers(9, 13) if remote else rng.integers(2, 5))
    if remote:
        parallel = True
    parallel = rng.random() < .35 if parallel is None else parallel
    offset = -max(v for u, v in raw) if parallel else near - max(v - u for u, v in raw)
    obstacle = {(u + near, v + offset) for u, v in raw}
    _route(obstacle, 24)
    return obstacle


def _reserve(reserved, cells):
    for r, c in cells:
        for dr, dc in D8 + ((0, 0),):
            p = r + dr, c + dc
            if _inside(p, reserved.shape):
                reserved[p] = True


def _place_station(rng, ticket, obstacle, length, axes, reserved, visibility, clear_tail=False):
    # Eight local orientations, independent of palette and address.
    orientations = [(f, l) for f in D4 if (0 if f[0] else 1) in axes
                    for l in ((f[1], -f[0]), (-f[1], f[0]))]
    path = _route(obstacle, max(length, 16) if clear_tail else length)
    for oi in rng.permutation(len(orientations)):
        f, l = orientations[int(oi)]
        local_input = ticket | obstacle
        input_deltas = np.array([_global(p, (0, 0), f, l) for p in sorted(local_input)], dtype=int)
        path_deltas = np.array([_global(p, (0, 0), f, l) for p in path], dtype=int)
        extent = np.concatenate((input_deltas, path_deltas)) if visibility == 'full' else input_deltas
        low, high = -extent.min(axis=0), 29 - extent.max(axis=0)
        if np.any(high < low):
            continue
        nr, nc = high - low + 1
        for flat in rng.permutation(int(nr * nc)):
            end = low + (int(flat) // int(nc), int(flat) % int(nc))
            ip = input_deltas + end
            pp = path_deltas + end
            inside = np.all((pp >= 0) & (pp < 30), axis=1)
            if not inside[0] or inside.sum() < 4 or (visibility == 'clip' and inside.all()):
                continue
            if visibility == 'clip' and not inside[:max(u for u, v in obstacle) + 2].all():
                continue
            footprint = np.concatenate((ip, pp[inside]))
            if np.any(reserved[footprint[:, 0], footprint[:, 1]]):
                continue
            end = tuple(map(int, end))
            result = {'ticket': {_global(p, end, f, l) for p in ticket},
                      'obstacle': {_global(p, end, f, l) for p in obstacle},
                      'path': [tuple(map(int, p)) for p in pp[:length]],
                      'clearance': [tuple(map(int, p)) for p in pp], 'end': end}
            _reserve(reserved, map(tuple, footprint.tolist()))
            return result
    return None


def _place_spools(rng, masks, reserved):
    placements = []
    # Larger rectangles first; identity/order is never a matching convention.
    for j in sorted(range(len(masks)), key=lambda i: -masks[i].size):
        h, w = masks[j].shape
        free = [(r, c) for r in range(31 - h) for c in range(31 - w)
                if not np.any(reserved[r:r + h, c:c + w])]
        if not free:
            return None
        r, c = _choose(rng, free)
        box = {(rr, cc) for rr in range(r, r + h) for cc in range(c, c + w)}
        _reserve(reserved, box)
        placements.append((j, r, c))
    return placements


def _construct_pair(rng, addresses, axes, lesson=None, large=False):
    masks = [_make_spool(rng, k, witness=lesson == 'full', large=large) for k in addresses]
    bg = int(rng.integers(10))
    palette = list(map(int, rng.permutation([c for c in range(10) if c != bg])))
    carriers = palette[:len(addresses)]
    spool_pool = palette[len(addresses):]
    spool_colors = [_choose(rng, spool_pool) for _ in addresses]
    # Witness colors differ, so an incorrect address assignment is observable.
    if lesson == 'full':
        spool_colors = spool_pool[:len(addresses)]
    for attempt in range(64):
        reserved = np.zeros((30, 30), dtype=bool)
        stations = []
        for j, (k, mask) in enumerate(zip(addresses, masks)):
            h = int(rng.integers(5 if k >= 3 else 4, 7))
            w = int(rng.integers(5 if k >= 3 else 4, 7))
            clues = _isolated_clues(rng, h, w, k)
            ticket = {(u, v) for u in range(-h - 1, -1) for v in range(1 - w, 1)} - clues
            ticket |= {(-1, 0), (0, 0)}
            # The two fully visible lessons show a rectangle and a genuine cap.
            kind = (1 if j == 0 else 2) if lesson == 'full' else None
            parallel = False if lesson == 'full' else True if lesson == 'clip' and j == len(addresses) - 1 else None
            remote = lesson is None and len(addresses) == 1 and rng.random() < .45
            obs = _make_obstacle(rng, kind, parallel=parallel, remote=remote)
            visibility = 'full' if lesson == 'full' else 'clip' if lesson == 'clip' and j == len(addresses) - 1 else 'either'
            station = _place_station(rng, ticket, obs, int(mask.sum()), axes, reserved, visibility,
                                     clear_tail=lesson == 'full')
            if station is None:
                break
            stations.append(station)
        if len(stations) != len(addresses):
            continue
        spool_places = _place_spools(rng, masks, reserved)
        if spool_places is None:
            continue
        if lesson == 'full':
            # Identical 4x4 boxes permit exchanging locations without repacking.
            # Make the unique minimum-total-distance assignment contradict the
            # topological assignment, so geography cannot replace the address.
            positions = {j: (r, c) for j, r, c in spool_places}
            def distance(j, t):
                r, c = positions[j]
                ar, ac = stations[t]['path'][0]
                return abs(2 * r + 3 - 2 * ar) + abs(2 * c + 3 - 2 * ac)
            own = distance(0, 0) + distance(1, 1)
            other = distance(0, 1) + distance(1, 0)
            if own == other:
                continue
            if own < other:
                spool_places = [(1 - j, r, c) for j, r, c in spool_places]
        inp = np.full((30, 30), bg, dtype=np.int8)
        for station, color in zip(stations, carriers):
            for p in station['ticket'] | station['obstacle']:
                inp[p] = color
        for j, r, c in spool_places:
            h, w = masks[j].shape
            inp[r:r + h, c:c + w][masks[j]] = spool_colors[j]
        out = inp.copy()
        for j, r, c in spool_places:
            h, w = masks[j].shape
            out[r:r + h, c:c + w] = bg
        visible = set()
        for station, color in zip(stations, spool_colors):
            for p in station['path']:
                if _inside(p, (30, 30)):
                    assert inp[p] == bg
                    out[p] = color
                    visible.add(p)
        all_cells = set(map(tuple, np.argwhere(inp != bg).tolist())) | visible
        if lesson == 'full':
            all_cells.update(p for station in stations for p in station['clearance'])
        r0, r1, c0, c1 = _bbox(all_cells)
        # Include every already visible route cell; crop adds no hidden clipping.
        r0 = int(rng.integers(r0 + 1))
        c0 = int(rng.integers(c0 + 1))
        r1 = int(rng.integers(r1, 30))
        c1 = int(rng.integers(c1, 30))
        pair = {'input': inp[r0:r1 + 1, c0:c1 + 1].copy(),
                'output': out[r0:r1 + 1, c0:c1 + 1].copy()}
        a, b, ts, ss = _decode(pair['input'])
        assert np.array_equal(_render(a, b, ts, ss), pair['output'])
        return pair
    raise RuntimeError('bounded placement exhausted')


def _check_counts(num_train, num_test):
    for name, value, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(name + ' is outside its legal integer range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))  # address count, reservoir size, or orientation shift
    base_axis = int(rng.integers(2))
    puzzle = {'train': [], 'test': []}
    used = set()
    for split, number in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(number):
            lesson = ('full' if index == 0 else 'clip' if index == 1 else None) if split == 'train' else None
            if lesson == 'full':
                addresses = [1, 2]
            elif lesson == 'clip':
                addresses = [0, int(rng.integers(3, 5))]
            else:
                n = 3 if split == 'test' and regime == 0 else int(rng.integers(1, 3))
                addresses = list(map(int, rng.choice(5, n, replace=False)))
                if split == 'test' and regime == 1:
                    addresses = [1] + list(map(int, rng.choice([0, 2, 3, 4], n - 1, replace=False)))
            axes = [base_axis if split == 'train' else 1 - base_axis] if regime == 2 else [0, 1]
            for _ in range(8):
                pair = _construct_pair(rng, addresses, axes, lesson, large=split == 'test' and regime == 1)
                key = _key(pair['input'])
                if key not in used:
                    used.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded unique-input construction exhausted')
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(int(num_train))]
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    a, bg, tickets, spools = _decode(pair['input'])
    ordered = sorted(spools, key=lambda s: s['count'])
    area_inversion = any(s['count'] < t['count'] and len(s['holes']) > len(t['holes'])
                         for s in spools for t in spools)
    geometry = False
    costs = []
    for order in permutations(range(len(spools))):
        cost = 0
        for t, si in zip(tickets, order):
            r0, r1, c0, c1 = spools[si]['box']
            ar, ac = t['path'][0]
            cost += abs(r0 + r1 - 2 * ar) + abs(c0 + c1 - 2 * ac)
        correct = all(t['count'] == spools[si]['count'] for t, si in zip(tickets, order))
        costs.append((cost, correct))
    nearest_wrong = len(costs) > 1 and min(c for c, ok in costs if ok) > min(c for c, ok in costs)
    all_visible = set().union(*(t['visible'] for t in tickets))
    clear_tails = all(s['length'] < 16 for s in spools)
    if clear_tails:
        for t in tickets:
            extra = [_global(p, t['end'], t['f'], t['l'])
                     for p in _route(t['local_obstacle'], 16)[t['spool']['length']:]]
            clear_tails &= all(_inside(p, a.shape) and a[p] == bg and p not in all_visible for p in extra)
    for t in tickets:
        obs = t['local_obstacle']
        u0, u1, v0, v1 = _bbox(obs)
        box = {(u, v) for u in range(u0, u1 + 1) for v in range(v0, v1 + 1)}
        envelope = [(u, max(v + 1 - abs(u - q) for q, v in box)) for u in range(t['spool']['length'])]
        if max(v for u, v in obs) > 0 and envelope != _route(obs, t['spool']['length']):
            geometry = True
    return dict(background=bg, shape=a.shape, station_count=len(tickets),
                addresses=tuple(s['count'] for s in ordered),
                spool_shapes=tuple((s['box'][1] - s['box'][0] + 1, s['box'][3] - s['box'][2] + 1) for s in ordered),
                lengths=tuple(s['length'] for s in ordered),
                cavity_sizes=tuple(s['cavity_sizes'] for s in ordered),
                area_inversion=area_inversion,
                connectivity=any(s['eight_count'] < s['count'] for s in spools),
                full=any(len(t['visible']) == t['spool']['length'] for t in tickets),
                all_full=all(len(t['visible']) == t['spool']['length'] for t in tickets),
                clipped=any(len(t['visible']) < t['spool']['length'] for t in tickets),
                parallel=any(max(v for u, v in t['local_obstacle']) == 0 for t in tickets),
                silhouette_witness=geometry,
                nearest_wrong=nearest_wrong,
                clear_tails=clear_tails,
                axes=tuple(sorted({0 if t['f'][0] else 1 for t in tickets})),
                directions=tuple(sorted({t['f'] for t in tickets})),
                obstacles=tuple(sorted(tuple(sorted(t['local_obstacle'])) for t in tickets)))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _check_counts(len(puzzle['train']), len(puzzle['test']))
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, out = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == out.shape and not np.array_equal(a, out)
            assert _key(a) not in seen
            seen.add(_key(a))
            aa, bg, ts, ss = _decode(a)
            assert np.array_equal(_render(aa, bg, ts, ss), out), 'incorrect unwind'
            for t in ts:
                assert all(out[p] == t['color'] for p in t['cells'] | t['obstacle'])
                assert out[_global((0, 1), t['end'], t['f'], t['l'])] == t['spool']['color']
                assert len(t['visible']) <= t['spool']['length']
            for color in {s['color'] for s in ss}:
                expected_material = sum(len(t['visible']) for t in ts if t['spool']['color'] == color)
                assert int(np.count_nonzero(out == color)) == expected_material
            assert all(out[p] == bg for s in ss for p in s['cells'])
            if split == 'train':
                training.append(_features(pair))
    assert any(f['area_inversion'] and f['connectivity'] and f['all_full']
               and f['nearest_wrong'] and f['clear_tails'] for f in training)
    assert any(f['silhouette_witness'] for f in training)
    assert any(f['clipped'] for f in training)
    assert any(f['parallel'] for f in training)
    assert any(0 in f['addresses'] for f in training)
    return None


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p, q in zip(a[s], b[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def _counterfactual(input_grid, mode):
    """Deliberately wrong rule variants for checking the teaching witnesses."""
    a, bg, tickets, spools = _decode(input_grid)
    out = a.copy()
    for s in spools:
        for p in s['cells']:
            out[p] = bg
    by_count = {s['count']: s for s in spools}
    if mode == 'area_matching':
        by_count = dict(zip(sorted(by_count), sorted(spools, key=lambda s: len(s['holes']))))
    for t in tickets:
        s = by_count[t['count']]
        r0, r1, c0, c1 = s['box']
        h, w = r1 - r0 + 1, c1 - c0 + 1
        length = 2 * h + 2 * w - 4 if mode == 'perimeter_length' else h * w if mode == 'box_length' else s['length']
        route = [(u, 1) for u in range(length)] if mode == 'straight' else _route(t['local_obstacle'], length)
        for p in route:
            q = _global(p, t['end'], t['f'], t['l'])
            if _inside(q, a.shape):
                out[q] = s['color']
    return out


def self_test():
    """42 streaming seeds, small regressions and API boundaries; writes no files."""
    shapes, structures, obstacles, addresses, lengths, backgrounds = set(), set(), set(), set(), set(), set()
    totals = Counter()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        witness = next(p for p in puzzle['train'] if _features(p)['area_inversion']
                       and _features(p)['connectivity'] and _features(p)['all_full']
                       and _features(p)['nearest_wrong'] and _features(p)['clear_tails'])
        for mode in ('area_matching', 'perimeter_length', 'box_length', 'straight'):
            assert not np.array_equal(_counterfactual(witness['input'], mode), witness['output']), mode
        # Rotation/reflection and color relabeling have known transformed labels.
        pair = puzzle['test'][0]
        transformed = {k: np.rot90(v.T, seed % 4).copy() for k, v in pair.items()}
        lookup = np.roll(np.arange(10, dtype=np.int8), seed % 9 + 1)
        assert np.array_equal(solve(lookup[transformed['input']]), lookup[transformed['output']])
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                assert np.array_equal(solve(pair['input'].copy()), pair['output'])
                f = _features(pair)
                shapes.add(f['shape'])
                addresses.add(f['addresses'])
                lengths.update(f['lengths'])
                backgrounds.add(f['background'])
                obstacles.update(f['obstacles'])
                roles = {}
                norm = bytes(roles.setdefault(int(c), len(roles)) for c in pair['input'].ravel())
                structures.add(hashlib.sha256(bytes(pair['input'].shape) + norm).digest())
                for name in ('clipped', 'full', 'area_inversion', 'connectivity', 'silhouette_witness'):
                    totals[name] += f[name]
                totals['three_stations'] += f['station_count'] == 3
    assert len(shapes) >= 20 and len(structures) >= 150
    assert len(addresses) >= 12 and len(lengths) >= 10
    assert len(obstacles) >= 35 and len(backgrounds) == 10
    assert all(totals[n] >= 10 for n in ('clipped', 'full', 'area_inversion', 'connectivity', 'silhouette_witness'))
    assert totals['three_stations'] >= 5
    assert _equal(generate_puzzle(17, 2, 1), generate_puzzle(17, 2, 1))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(59, nt, ne)
            assert (len(p['train']), len(p['test'])) == (nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # The two source-supported tangent types and a nonrectangular contour.
    assert _route({(2, 0)}, 6) == [(0, 1), (1, 1), (2, 1), (3, 0), (4, -1), (5, -2)]
    assert _route({(2, 2), (3, 3), (4, 2)}, 6) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 3), (5, 2)]
    # Labels are not used for role inference; malformed material must be rejected.
    damaged = generate_puzzle(0, 2, 1)
    wrong = next(p for p in damaged['train'] if _features(p)['connectivity'] and _features(p)['all_full'])
    wrong['output'] = _counterfactual(wrong['input'], 'perimeter_length')
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('wrong material length accepted')
    return {'seeds': 42, 'input_shapes': len(shapes), 'structures': len(structures),
            'address_sets': len(addresses), 'length_values': len(lengths),
            'obstacle_geometries': len(obstacles), 'backgrounds': len(backgrounds),
            'witness_totals': dict(totals)}
