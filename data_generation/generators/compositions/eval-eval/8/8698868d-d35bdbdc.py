"""Topology-typed relay stamps. See task_spec.md for the input-visible rule.

Only standard-library modules and NumPy are used. Construction records never
enter solve(): the parser rediscovers plaques, their types, and line endpoints.
"""
from __future__ import annotations

import heapq
from itertools import combinations

import numpy as np


_D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _choice(rng, values):
    return values[int(rng.integers(len(values)))]


def _components(mask, diagonal=False):
    pending = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    steps = _D8 if diagonal else _D4
    while pending:
        p = min(pending)
        pending.remove(p)
        cells, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in pending:
                    pending.remove(q)
                    cells.add(q)
                    stack.append(q)
        result.append(cells)
    return result


def _box(cells):
    rs, cs = zip(*cells)
    return min(rs), max(rs) + 1, min(cs), max(cs) + 1


def _perimeter(grid):
    return np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))


def _transforms(mask):
    return [np.rot90(mask, k) for k in range(4)] + [np.rot90(mask.T, k) for k in range(4)]


def _grid(value):
    arr = np.asarray(value)
    assert arr.ndim == 2 and all(1 <= n <= 30 for n in arr.shape), 'invalid grid shape'
    assert np.issubdtype(arr.dtype, np.integer), 'noninteger grid'
    assert np.all((0 <= arr) & (arr <= 9)), 'invalid color'
    return arr


def _ports(box):
    r, endr, c, endc = box
    m = (endr - r) // 2
    return ((r - 1, c + m), (endr, c + m), (r + m, c - 1), (r + m, endc))


def _parse_roles(grid, bg, line):
    plaques = []
    for cells in _components((grid != bg) & (grid != line)):
        box = _box(cells)
        r, endr, c, endc = box
        s = endr - r
        assert s in (7, 9, 11) and endc - c == s
        patch = grid[r:endr, c:endc]
        body = int(patch[0, 0])
        assert body not in (bg, line) and np.all(_perimeter(patch) == body)
        middle = s // 2
        tag = int(patch[middle, middle])
        assert tag not in (bg, line)
        untagged = patch.copy()
        untagged[middle, middle] = body
        assert set(map(int, np.unique(untagged))) <= {body, bg}
        assert np.all(untagged[middle-1:middle+2, middle-1:middle+2] == body)
        holes = patch == bg
        assert len(_components(~holes)) == 1
        assert len(cells) == int((patch != bg).sum()), 'disconnected plaque'
        plaques.append(dict(box=box, side=s, body=body, tag=tag, holes=holes,
                            count=len(_components(holes)), center=(r+middle, c+middle)))
    assert 2 <= len(plaques) <= 9
    assert len({t['side'] for t in plaques}) == 1
    for a, b in combinations(plaques, 2):
        r, s, c, d = a['box']
        e, f, g, h = b['box']
        assert s < e or f < r or d < g or h < c, 'plaque boxes require a gap'
    lookup = {}
    for i, t in enumerate(plaques):
        key = (t['body'], t['count'])
        assert key not in lookup, 'ambiguous donor key'
        lookup[key] = i
    line_mask = grid == line
    line_parts = _components(line_mask)
    assert 1 <= len(line_parts) <= 2
    pixels = set().union(*line_parts)
    graph = {p: {q for dr, dc in _D4 if (q := (p[0]+dr, p[1]+dc)) in pixels}
             for p in pixels}
    assert all(len(v) in (1, 2) for v in graph.values()), 'branched or isolated line'
    endpoints = {p for p, neighbors in graph.items() if len(neighbors) == 1}
    assert all(len(part) >= 3 and len(part & endpoints) == 2 for part in line_parts)
    for t in plaques:
        t['ends'] = set(_ports(t['box'])) & endpoints
        assert len(t['ends']) <= 1
        t['active'] = bool(t['ends'])
        j = lookup.get((t['tag'], t['count']))
        t['donor'] = j if j is not None and plaques[j] is not t else None
    assert all(sum(p in t['ends'] for t in plaques) <= 1 for p in endpoints)
    assert all(any(t['ends'] & part for t in plaques) for part in line_parts)
    return dict(background=bg, line=line, plaques=plaques, graph=graph,
                parts=line_parts, endpoints=endpoints)


def _interpret(input_grid):
    grid = _grid(input_grid)
    edge = _perimeter(grid)
    assert np.all(edge == edge[0]), 'blank outer perimeter required'
    bg = int(edge[0])
    candidates = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        try:
            candidates.append(_parse_roles(grid, bg, color))
        except (AssertionError, IndexError, ValueError):
            continue
    assert len(candidates) == 1, 'line/plaque interpretation must be unique'
    return grid, candidates[0]


def _stamp(receiver, donor, ink=None, mask=None):
    result = np.full((receiver['side'], receiver['side']), receiver['body'], dtype=np.int64)
    holes = donor['holes'] if mask is None else mask
    color = donor['tag'] if ink is None else ink
    result[1:-1, 1:-1] = np.where(holes[1:-1, 1:-1], receiver['body'], color)
    return result


def _render(grid, info, alternative=None):
    """Input-derived rendering; optional counterfactuals are used only in tests."""
    result = grid.copy()
    plaques, bg = info['plaques'], info['background']
    for i, t in enumerate(plaques):
        r, s, c, d = t['box']
        result[r:s, c:d] = bg
        active = t['active']
        if alternative == 'all':
            active = True
        elif alternative == 'touching':
            active = any(r <= p[0]+dr < s and c <= p[1]+dc < d
                         for p in info['graph'] for dr, dc in _D4)
        if not active:
            continue
        j = t['donor']
        if alternative == 'color_only':
            donors = [q for q, other in enumerate(plaques) if q != i and other['body'] == t['tag']]
            j = min(donors, key=lambda q: plaques[q]['count']) if donors else None
        elif alternative == 'count_only':
            donors = [q for q, other in enumerate(plaques) if q != i and other['count'] == t['count']]
            j = min(donors, key=lambda q: (plaques[q]['body'], plaques[q]['box'])) if donors else None
        elif alternative == 'eight':
            k = len(_components(t['holes'], diagonal=True))
            donors = [q for q, other in enumerate(plaques) if q != i and other['body'] == t['tag']
                      and len(_components(other['holes'], diagonal=True)) == k]
            j = donors[0] if donors else None
        if j is None:
            continue
        donor = plaques[j]
        ink, mask = donor['tag'], donor['holes']
        if alternative == 'updated' and donor['active'] and donor['donor'] is not None:
            ink = plaques[donor['donor']]['tag']
        if alternative == 'rotate':
            mask = np.rot90(mask)
        if alternative == 'self_pattern':
            mask = t['holes']
        result[r:s, c:d] = _stamp(t, donor, ink, mask)
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer exclusively from the supplied grid, without mutation."""
    grid, info = _interpret(input_grid)
    return _render(grid, info)


def _features(grid, info=None):
    if info is None:
        grid, info = _interpret(grid)
    ts = info['plaques']
    successes = [t for t in ts if t['active'] and t['donor'] is not None]
    touching = lambda t: any(t['box'][0] <= p[0]+dr < t['box'][1]
                             and t['box'][2] <= p[1]+dc < t['box'][3]
                             for p in info['graph'] for dr, dc in _D4)
    stamp_diff = lambda t, d: not np.array_equal(_stamp(t, ts[t['donor']]), _stamp(t, d))
    return dict(
        side=ts[0]['side'], plaques=len(ts), shape=grid.shape,
        background=info['background'], line=info['line'], lines=len(info['parts']),
        active=sum(t['active'] for t in ts), successful=len(successes),
        failed=sum(t['active'] and t['donor'] is None for t in ts),
        bare_ends=len(info['endpoints'])-sum(t['active'] for t in ts),
        counts=tuple(sorted({t['count'] for t in ts})),
        body_reuse=len(ts)-len({t['body'] for t in ts}),
        remote=sum(not ts[t['donor']]['active'] for t in successes),
        retained_donor=sum(ts[t['donor']]['active'] and ts[t['donor']]['donor'] is not None
                           for t in successes),
        original_tag_witness=sum(ts[t['donor']]['active'] and ts[t['donor']]['donor'] is not None
                                 and ts[ts[t['donor']]['donor']]['tag'] != ts[t['donor']]['tag']
                                 for t in successes),
        unchanged=sum(ts[t['donor']]['tag'] == t['tag'] for t in successes),
        solid=sum(ts[t['donor']]['tag'] == t['body'] for t in successes),
        monochrome=sum(t['body'] == t['tag'] for t in ts),
        interior_compatible=sum(not t['active'] and t['donor'] is not None and touching(t) for t in ts),
        compatible_inactive=sum(not t['active'] and t['donor'] is not None for t in ts),
        count_sensitive=any(d['body'] == t['tag'] and d['count'] != t['count'] and stamp_diff(t, d)
                            for t in successes for d in ts),
        color_sensitive=any(d is not t and d['body'] != t['tag'] and d['count'] == t['count']
                            and stamp_diff(t, d) for t in successes for d in ts),
        diagonal=any(len(_components(t['holes'], True)) < t['count'] for t in ts),
        multi_cell=any(t['holes'].sum() > t['count'] for t in ts),
        area_inversion=any(a['body'] == b['body'] and a['count'] < b['count'] and a['holes'].sum() > b['holes'].sum()
                           for a, b in combinations(sorted(ts, key=lambda t: t['count']), 2)),
        orientation=any(ts[t['donor']]['tag'] != t['body'] and
                        all(not np.array_equal(ts[t['donor']]['holes'], v)
                            for v in _transforms(ts[t['donor']]['holes'])[1:]) for t in successes),
        nonpositional=any(not np.array_equal(t['holes'], ts[t['donor']]['holes']) for t in successes),
        bends=sum(len(adj) == 2 and tuple(map(sum, zip(*adj))) != (2*p[0], 2*p[1])
                  for p, adj in info['graph'].items()),
        centers=tuple(t['center'] for t in ts),
    )


def _allowed_holes(side):
    mask = np.zeros((side, side), dtype=bool)
    mask[1:-1, 1:-1] = True
    m = side // 2
    mask[m-1:m+2, m-1:m+2] = False
    return mask


def _isolated_mask(rng, side, count, separate_diagonals=False):
    if not count:
        return np.zeros((side, side), dtype=bool)
    cells = list(map(tuple, np.argwhere(_allowed_holes(side)).tolist()))
    neighbors = _D8 if separate_diagonals else _D4
    for _ in range(64):
        mask = np.zeros((side, side), dtype=bool)
        for index in rng.permutation(len(cells)):
            r, c = cells[int(index)]
            if any(mask[r+dr, c+dc] for dr, dc in neighbors):
                continue
            mask[r, c] = True
            if len(_components(~mask)) != 1:
                mask[r, c] = False
                continue
            if int(mask.sum()) == count:
                return mask
    raise AssertionError('could not place isolated cavities')


def _cavity_mask(rng, side, count):
    mask = _isolated_mask(rng, side, count)
    if not count or rng.random() < .25:
        return mask
    allowed = _allowed_holes(side)
    target = int(rng.integers(count, int(allowed.sum())+1))
    for _ in range(int(allowed.sum())):
        if int(mask.sum()) >= target:
            break
        grown = False
        for index in rng.permutation(side*side):
            r, c = divmod(int(index), side)
            if not allowed[r, c] or mask[r, c] or not any(mask[r+dr, c+dc] for dr, dc in _D4):
                continue
            mask[r, c] = True
            if len(_components(mask)) == count and len(_components(~mask)) == 1:
                grown = True
                break
            mask[r, c] = False
        if not grown:
            break
    return mask


def _make_relations(rng, mode, side, cap, allow_solid):
    if mode == 'typed':
        bodies, tags, counts = [0, 1, 1, 2, 3], [1, 3, 1, 1, 3], [2, 2, 1, 1, 2]
        groups, via = [[0, 3]], 1
    elif mode == 'chain':
        bodies, tags, counts = [0, 1, 2, 3], [1, 2, 3, 3], [0]*4
        groups, via = [[0, 1], [3]], None
    elif mode == 'mutual':
        bodies, tags, counts = [0, 1, 2], [1, 0, 1], [1]*3
        groups, via = [[0, 1]], None
    else:
        load = mode == 'load'
        maximum = (5 if mode == 'train' else 8) if side == 7 else 4
        n = int(rng.integers(6, 9)) if load else int(rng.integers(2, maximum+1))
        q = int(rng.integers(4 if load else 3, 9))
        k = int(rng.integers(3, 5)) if mode == 'count' else int(rng.integers(cap+1))
        keys = [(b, k) for b in range(4 if load else 2)]
        others = [(b, t) for b in range(q) for t in range(cap+1) if (b, t) not in keys]
        keys += [_choice(rng, others)] if n > len(keys) else []
        while len(keys) < n:
            key = _choice(rng, others)
            if key not in keys:
                keys.append(key)
        bodies, counts = map(list, zip(*keys))
        tags = list(map(int, rng.integers(q, size=n)))
        tags[0] = 1
        if load:
            tags[:4] = [1, 2, 3, 0]
            active = list(range(4))
        else:
            active_count = int(rng.integers(1, min(n, 2 if mode == 'train' else 4)+1))
            active = [0] + list(map(int, rng.choice(np.arange(1, n), active_count-1, replace=False)))
            for i in active[1:]:
                possible = [j for j in range(n) if j != i and counts[j] == counts[i]]
                if possible and rng.random() < .65:
                    tags[i] = bodies[_choice(rng, possible)]
        lookup = {key: i for i, key in enumerate(keys)}
        if not allow_solid:
            for _ in range(4):
                bad = [(i, lookup[(tags[i], counts[i])]) for i in active
                       if (tags[i], counts[i]) in lookup and lookup[(tags[i], counts[i])] != i
                       and tags[lookup[(tags[i], counts[i])]] == bodies[i]]
                if not bad:
                    break
                for i, j in bad:
                    if j == 0:
                        # Preserve the guaranteed 0 -> 1 relay; the competing
                        # receiver can instead have a visibly failed self-key.
                        tags[i] = bodies[i]
                    else:
                        tags[j] = bodies[j]
        rng.shuffle(active)
        lines = 2 if len(active) > 2 or (len(active) == 2 and rng.random() < .35) else 1
        groups = [[active[i]] for i in range(lines)]
        for i in active[lines:]:
            eligible = [g for g in groups if len(g) == 1]
            _choice(rng, eligible).append(i)
        via = None
    masks = [_cavity_mask(rng, side, k) for k in counts]
    if mode == 'typed':
        masks[0] = _isolated_mask(rng, side, 2, separate_diagonals=True)
        masks[3] = _isolated_mask(rng, side, 1)
        masks[4] = _isolated_mask(rng, side, 2)
        masks[1][:] = False
        masks[1][1, 2] = masks[1][1, 3] = masks[1][2, 1] = True
        masks[2][:] = False
        for p in ((1, 1), (1, 2), (1, 3), (1, 4), (2, 1)):
            masks[2][p] = True
        masks[1] = _choice(rng, _transforms(masks[1])).copy()
        masks[2] = _choice(rng, _transforms(masks[2])).copy()
    return bodies, tags, counts, masks, groups, via


def _random_boxes(rng, n, side):
    maximum = 3 if side == 7 else 2
    shelves = [(r, c) for r in range(1, maximum+1) for c in range(1, maximum+1) if r*c >= n]
    rcount, ccount = _choice(rng, shelves)
    gapr = int(rng.integers(1, max(1, (28-rcount*side)//max(1, rcount-1))+1)) if rcount > 1 else 1
    gapc = int(rng.integers(1, max(1, (28-ccount*side)//max(1, ccount-1))+1)) if ccount > 1 else 1
    min_h, min_w = rcount*side+(rcount-1)*gapr+2, ccount*side+(ccount-1)*gapc+2
    h, w = int(rng.integers(min_h, 31)), int(rng.integers(min_w, 31))
    for _ in range(8):
        boxes = []
        for _ in range(n):
            free = [(r, r+side, c, c+side) for r in range(1, h-side)
                    for c in range(1, w-side)
                    if all(r+side < a or b < r or c+side < e or f < c for a, b, e, f in boxes)]
            if not free:
                break
            boxes.append(_choice(rng, free))
        if len(boxes) == n:
            return (h, w), boxes
    # Shelf packing has random internal gaps, origins, identities and omissions.
    extra_r = h-min_h
    top = int(rng.integers(extra_r+1)) + 1
    slots = []
    for row in range(rcount):
        extra_c = w-min_w
        left = int(rng.integers(extra_c+1)) + 1
        for col in range(ccount):
            r, c = top+row*(side+gapr), left+col*(side+gapc)
            slots.append((r, r+side, c, c+side))
    return (h, w), [slots[int(i)] for i in rng.permutation(len(slots))[:n]]


def _blocked_for(shape, boxes):
    h, w = shape
    blocked = {(r, c) for r in range(h) for c in range(w) if r in (0, h-1) or c in (0, w-1)}
    for a, b, e, f in boxes:
        blocked.update((r, c) for r in range(a, b) for c in range(e, f))
    return blocked


def _route(rng, shape, blocked, start, finish):
    h, w = shape
    if start == finish or start in blocked or finish in blocked:
        return None
    costs = rng.uniform(1., 1.45, size=shape)
    distance, previous = {start: 0.}, {}
    queue = [(abs(start[0]-finish[0])+abs(start[1]-finish[1]), 0., start)]
    while queue:
        _, value, p = heapq.heappop(queue)
        if value != distance[p]:
            continue
        if p == finish:
            path = [p]
            while p != start:
                p = previous[p]
                path.append(p)
            return path[::-1]
        for dr, dc in _D4:
            q = (p[0]+dr, p[1]+dc)
            if not (0 <= q[0] < h and 0 <= q[1] < w) or q in blocked:
                continue
            new = value+float(costs[q])
            if new < distance.get(q, float('inf')):
                distance[q], previous[q] = new, p
                heuristic = abs(q[0]-finish[0])+abs(q[1]-finish[1])
                heapq.heappush(queue, (new+heuristic, new, q))
    return None


def _inflate(cells, steps=_D4):
    return set(cells) | {(r+dr, c+dc) for r, c in cells for dr, dc in steps}


def _path_valid(path):
    if path is None or len(path) < 3 or len(set(path)) != len(path):
        return False
    pixels = set(path)
    degrees = {p: sum((p[0]+dr, p[1]+dc) in pixels for dr, dc in _D4) for p in pixels}
    return {p for p, deg in degrees.items() if deg == 1} == {path[0], path[-1]} and all(d in (1, 2) for d in degrees.values())


def _route_groups(rng, shape, boxes, groups, via):
    base = _blocked_for(shape, boxes)
    expanded_base = _inflate(base)
    gray = set()
    for group_index, group in enumerate(groups):
        found = None
        for _ in range(14):
            blocked = base | _inflate(gray, _D8)
            possible = [p for p in _ports(boxes[group[0]]) if p not in blocked]
            if not possible:
                continue
            start = _choice(rng, possible)
            if len(group) == 2:
                possible = [p for p in _ports(boxes[group[1]]) if p not in blocked]
            else:
                reserved = set().union(*(set(_ports(b)) for b in boxes))
                possible = [(r, c) for r in range(1, shape[0]-1) for c in range(1, shape[1]-1)
                            if (r, c) not in expanded_base and (r, c) not in blocked and (r, c) not in reserved]
            if not possible:
                continue
            end = _choice(rng, possible)
            if via is not None and group_index == 0:
                mids = [p for p in _ports(boxes[via]) if p not in blocked and p not in (start, end)]
                if not mids:
                    continue
                middle = _choice(rng, mids)
                prefix = _route(rng, shape, blocked, start, middle)
                if prefix is None:
                    continue
                blocked2 = blocked | _inflate(prefix[:-1])
                blocked2.discard(middle)
                suffix = _route(rng, shape, blocked2, middle, end)
                path = prefix+suffix[1:] if suffix is not None else None
            else:
                path = _route(rng, shape, blocked, start, end)
            if not _path_valid(path):
                continue
            ends = {path[0], path[-1]}
            attached = {i for i, box in enumerate(boxes) if ends & set(_ports(box))}
            if attached != set(group) or any(sum(p in _ports(box) for box in boxes) > 1 for p in ends):
                continue
            found = path
            break
        if found is None:
            return None
        gray.update(found)
    return gray


def _fallback_layout(rng, n, side, groups, via):
    if via is not None:
        boxes = [(2, 9, 2, 9), (2, 9, 12, 19), (2, 9, 22, 29),
                 (20, 27, 22, 29), (20, 27, 2, 9)]
        waypoints = ((5, 9), (9, 9), (9, 20), (18, 20), (18, 25), (19, 25))
        gray = set()
        for a, b in zip(waypoints, waypoints[1:]):
            dr, dc = int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1]))
            p = a
            gray.add(p)
            while p != b:
                p = (p[0]+dr, p[1]+dc)
                gray.add(p)
        return (30, 30), boxes, gray
    origins = (1, 11, 21) if side == 7 else (2, 28-side)
    slots = [(r, r+side, c, c+side) for r in origins for c in origins]
    boxes, used, gray = [None]*n, set(), set()
    for line_index, group in enumerate(groups):
        row = 0 if line_index == 0 else len(origins)-1
        first_slot = row*len(origins)
        first = slots[first_slot]
        boxes[group[0]] = first
        used.add(first_slot)
        middle_r = first[0]+side//2
        if len(group) == 2:
            second_slot = first_slot+1
            second = slots[second_slot]
            boxes[group[1]] = second
            used.add(second_slot)
            gray.update((middle_r, c) for c in range(first[3], second[2]))
        else:
            middle_c = first[2]+side//2
            port_r = first[1] if row == 0 else first[0]-1
            direction = 1 if row == 0 else -1
            gray.update(((port_r, middle_c), (port_r+direction, middle_c), (port_r+direction, middle_c+1)))
    available = [i for i in range(len(slots)) if i not in used]
    rng.shuffle(available)
    for i in range(n):
        if boxes[i] is None:
            boxes[i] = slots[available.pop()]
    return (30, 30), boxes, gray


def _layout(rng, n, side, groups, via):
    for _ in range(10):
        shape, boxes = _random_boxes(rng, n, side)
        gray = _route_groups(rng, shape, boxes, groups, via)
        if gray is not None:
            return shape, boxes, gray
    return _fallback_layout(rng, n, side, groups, via)


def _make_pair(rng, mode, side, cap, roles, allow_solid):
    bodies, tags, counts, masks, groups, via = _make_relations(rng, mode, side, cap, allow_solid)
    shape, boxes, gray = _layout(rng, len(bodies), side, groups, via)
    bg, line = roles
    identities = list(map(int, rng.permutation([c for c in range(10) if c not in roles])))
    inp = np.full(shape, bg, dtype=np.int64)
    out = np.full(shape, bg, dtype=np.int64)
    active = set(i for g in groups for i in g)
    keys = {(b, k): i for i, (b, k) in enumerate(zip(bodies, counts))}
    for i, (r, s, c, d) in enumerate(boxes):
        patch = np.full((side, side), identities[bodies[i]], dtype=np.int64)
        patch[masks[i]] = bg
        patch[side//2, side//2] = identities[tags[i]]
        inp[r:s, c:d] = patch
        j = keys.get((tags[i], counts[i]))
        if i in active and j is not None and j != i:
            # Construct directly from original records, independently of parsing.
            answer = np.full((side, side), identities[bodies[i]], dtype=np.int64)
            answer[1:-1, 1:-1] = np.where(masks[j][1:-1, 1:-1], identities[bodies[i]], identities[tags[j]])
            out[r:s, c:d] = answer
    for p in gray:
        inp[p] = out[p] = line
    k = int(rng.integers(4))
    inp, out = np.rot90(inp, k), np.rot90(out, k)
    if rng.random() < .5:
        inp, out = np.fliplr(inp), np.fliplr(out)
    return {'input': inp.copy(), 'output': out.copy()}


def _key(grid):
    arr = np.asarray(grid, dtype=np.int8)
    return arr.shape, arr.tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, analyses = set(), {'train': [], 'test': []}
    decoded = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp, out)
            assert _key(inp) not in seen, 'duplicate input'
            seen.add(_key(inp))
            _, info = _interpret(inp)
            decoded[split].append((inp, out, info))
            predicted = _render(inp, info)
            assert np.array_equal(predicted, out), 'incorrect simultaneous typed stamp'
            assert np.all(out[inp == info['line']] == info['line']), 'line changed'
            assert np.all(out[inp == info['background']] != info['line']), 'new line'
            assert np.all(_perimeter(out) == info['background'])
            f = _features(inp, info)
            assert f['successful'] >= 1
            analyses[split].append(f)
    train, test = analyses['train'], analyses['test']
    for name in ('count_sensitive', 'color_sensitive', 'diagonal', 'multi_cell', 'area_inversion',
                 'orientation', 'nonpositional', 'interior_compatible', 'unchanged', 'monochrome',
                 'remote', 'retained_donor', 'original_tag_witness', 'failed', 'bare_ends'):
        assert any(f[name] for f in train), 'missing training witness: '+name
    assert max(f['lines'] for f in test) <= max(f['lines'] for f in train)
    if any(f['solid'] for f in test):
        assert any(f['solid'] for f in train), 'solid results must be demonstrated'
    for f in test:
        assert (f['side'] > max(t['side'] for t in train)
                or max(f['counts']) > max(max(t['counts']) for t in train)
                or (f['plaques'] > max(t['plaques'] for t in train)
                    and f['successful'] > max(t['successful'] for t in train))), 'missing structural test shift'
    # Concrete competing rules must already disagree with demonstrations.
    for alternative in ('color_only', 'count_only', 'eight', 'all', 'touching', 'updated', 'rotate', 'self_pattern'):
        assert any(not np.array_equal(_render(inp, info, alternative), out)
                   for inp, out, info in decoded['train']), 'training permits shortcut: '+alternative
    return None


def _check_counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(name+' must be an integer in '+str((low, high)))
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _check_counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = _choice(rng, ('scale', 'count', 'load'))
    varied_roles = bool(rng.random() < .75)
    base_roles = tuple(map(int, rng.choice(10, 2, replace=False)))
    allow_solid = bool(num_train >= 3 and rng.random() < .4)
    jobs = [('train', 'typed', 7), ('train', 'chain', 7 if shift in ('scale', 'load') else _choice(rng, (9, 11)))]
    if allow_solid:
        jobs.append(('train', 'mutual', 7))
    while len(jobs) < num_train:
        jobs.append(('train', 'train', 7))
    for _ in range(num_test):
        jobs.append(('test', shift, _choice(rng, (9, 11)) if shift == 'scale' else 7))
    puzzle, seen = {'train': [], 'test': []}, set()
    previous_roles = None
    for split, mode, side in jobs:
        roles = tuple(map(int, rng.choice(10, 2, replace=False))) if varied_roles else base_roles
        if varied_roles and previous_roles is not None and roles == previous_roles:
            roles = (roles[1], roles[0])
        previous_roles = roles
        for _ in range(12):
            pair = _make_pair(rng, mode, side, 4 if mode == 'count' else 2, roles, allow_solid)
            _, info = _interpret(pair['input'])
            assert np.array_equal(_render(pair['input'], info), pair['output'])
            if _key(pair['input']) not in seen:
                seen.add(_key(pair['input']))
                puzzle[split].append(pair)
                break
        else:
            raise AssertionError('bounded unique-input generation exhausted')
    puzzle['train'] = [puzzle['train'][int(i)] for i in rng.permutation(num_train)]
    validate_generated_puzzle(puzzle)
    return puzzle


def variation_metrics(puzzles):
    """Descriptive metrics reconstructed from grids rather than latent records."""
    rows, patterns, layouts = [], set(), set()
    shifts = {'scale': 0, 'count': 0, 'load': 0}
    for puzzle in puzzles:
        tr = [_features(p['input']) for p in puzzle['train']]
        for split in ('train', 'test'):
            for p in puzzle[split]:
                inp, info = _interpret(p['input'])
                f = _features(inp, info)
                rows.append(f)
                for t in info['plaques']:
                    patterns.add((t['side'], min(v.tobytes() for v in _transforms(t['holes']))))
                pts = f['centers']
                variants = []
                for swap in (False, True):
                    for sr in (-1, 1):
                        for sc in (-1, 1):
                            points = [(sr*(c if swap else r), sc*(r if swap else c)) for r, c in pts]
                            mr, mc = min(r for r, c in points), min(c for r, c in points)
                            variants.append(tuple(sorted((r-mr, c-mc) for r, c in points)))
                layouts.add(min(variants))
                if split == 'test':
                    if f['side'] > max(t['side'] for t in tr):
                        shifts['scale'] += 1
                    elif max(f['counts']) > max(max(t['counts']) for t in tr):
                        shifts['count'] += 1
                    else:
                        shifts['load'] += 1
    return dict(puzzles=len(puzzles), pairs=len(rows), canvas_shapes=len({f['shape'] for f in rows}),
                plaque_sides=sorted({f['side'] for f in rows}), plaque_counts=sorted({f['plaques'] for f in rows}),
                cavity_counts=sorted({k for f in rows for k in f['counts']}),
                role_pairs=len({(f['background'], f['line']) for f in rows}),
                normalized_layouts=len(layouts), cavity_shapes_modulo_symmetry=len(patterns),
                successful_counts=sorted({f['successful'] for f in rows}), shifts=shifts,
                feature_pairs={name: sum(bool(f[name]) for f in rows) for name in
                               ('body_reuse', 'remote', 'retained_donor', 'original_tag_witness',
                                'unchanged', 'solid', 'monochrome', 'interior_compatible',
                                'count_sensitive', 'color_sensitive', 'diagonal', 'multi_cell',
                                'area_inversion', 'orientation', 'failed', 'bare_ends', 'bends')})


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    metrics = variation_metrics(puzzles)
    assert metrics['puzzles'] == 42 and metrics['pairs'] == 210
    assert metrics['canvas_shapes'] >= 20
    assert metrics['plaque_sides'] == [7, 9, 11]
    assert {2, 3, 4, 5, 6, 7, 8} <= set(metrics['plaque_counts'])
    assert {0, 1, 2, 3, 4} <= set(metrics['cavity_counts'])
    assert metrics['normalized_layouts'] >= 60
    assert metrics['cavity_shapes_modulo_symmetry'] >= 80
    assert metrics['role_pairs'] >= 40
    assert all(metrics['shifts'][k] >= 5 for k in metrics['shifts'])
    for name, count in metrics['feature_pairs'].items():
        assert count >= 5, (name, count)
    assert len({tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]) for puzzle in puzzles}) == 42
    a, b = generate_puzzle(0), generate_puzzle(0)
    assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    for puzzle in puzzles:
        for s in puzzle:
            for pair in puzzle[s]:
                original = pair['input'].copy()
                assert np.array_equal(solve(pair['input']), pair['output'])
                assert np.array_equal(pair['input'], original)
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            puzzle = generate_puzzle(12345+nt*10+ne, nt, ne)
            assert len(puzzle['train']) == nt and len(puzzle['test']) == ne
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1),
                   (4, False), (2.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted: '+str(counts))
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in a[s]] for s in a}
    _, info = _interpret(bad['test'][0]['input'])
    cell = min(info['graph'])
    bad['test'][0]['output'][cell] = info['background']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted scaffold accepted')
    return metrics


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(), separators=(',', ':')))
