"""Contour-sensing wires: spatial contour palettes sampled by repaired routes.

Only NumPy and the Python standard library are used. No parent code is imported.
The public solver receives one input, and all conventions are visible or fixed.
"""
from collections import deque
from numbers import Integral
import numpy as np

REPAIR, DIRECTION = 2, 3
D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9))
    return a.astype(np.int8, copy=True)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(p, shape):
    return 1 <= p[0] < shape[0] - 1 and 1 <= p[1] < shape[1] - 1


def _terminal(p, shape):
    return _inside(p, shape) and (p[0] in (1, shape[0]-2) or p[1] in (1, shape[1]-2))


def _neighbors(p, cells):
    return [_add(p, d) for d in D4 if _add(p, d) in cells]


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    d = (int(np.sign(b[0]-a[0])), int(np.sign(b[1]-a[1])))
    return [(a[0]+i*d[0], a[1]+i*d[1])
            for i in range(max(abs(b[0]-a[0]), abs(b[1]-a[1]))+1)]


def _polyline(vertices):
    result = [vertices[0]]
    for a, b in zip(vertices, vertices[1:]):
        result.extend(_line(a, b)[1:])
    return result


def _ray(p, d, shape):
    result = set()
    p = _add(p, d)
    while _inside(p, shape):
        result.add(p)
        p = _add(p, d)
    return result


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask)))
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        cells, pending = {start}, [start]
        while pending:
            p = pending.pop()
            for q in _neighbors(p, remaining):
                remaining.remove(q)
                cells.add(q)
                pending.append(q)
        a = np.zeros(mask.shape, dtype=bool)
        for p in cells:
            a[p] = True
        result.append(a)
    return result


def _depth(mask, diagonal=True):
    """Repeated erosion; walls, other rooms, and off-canvas positions are outside."""
    current = mask.copy()
    depth = np.full(mask.shape, -1, dtype=np.int16)
    level, h, w = 0, *mask.shape
    while current.any():
        padded = np.pad(current, 1)
        inner = current.copy()
        for dr, dc in D8 if diagonal else D4:
            inner &= padded[1+dr:1+dr+h, 1+dc:1+dc+w]
        depth[current & ~inner] = level
        current = inner
        level += 1
    return depth


def _bfs_depth(mask):
    """Construction depth, separately checked against iterative erosion."""
    h, w = mask.shape
    depth = np.full(mask.shape, -1, dtype=np.int16)
    queue = deque()
    for r, c in np.argwhere(mask):
        p = int(r), int(c)
        if any(not (0 <= r+dr < h and 0 <= c+dc < w and mask[r+dr, c+dc])
               for dr, dc in D8):
            depth[p] = 0
            queue.append(p)
    while queue:
        r, c = queue.popleft()
        for dr, dc in D8:
            q = r+dr, c+dc
            if 0 <= q[0] < h and 0 <= q[1] < w and mask[q] and depth[q] < 0:
                depth[q] = depth[r, c] + 1
                queue.append(q)
    return depth


def _roles(inp):
    border = np.concatenate((inp[0], inp[-1], inp[:, 0], inp[:, -1]))
    assert len(set(map(int, border))) == 1, 'outer wall must be intact'
    wall = int(border[0])
    assert wall not in (REPAIR, DIRECTION)
    counts = np.bincount(inp.ravel(), minlength=10)
    order = sorted((c for c in range(10) if c not in (wall, REPAIR, DIRECTION)),
                   key=lambda c: (-int(counts[c]), c))
    bg, carrier = order[:2]
    assert counts[bg] > inp.size//2, 'empty color must occupy a strict majority'
    assert counts[carrier] > max(int(counts[c]) for c in order[2:]), 'carrier must outrank every cue color'
    assert counts[carrier] >= 5
    return bg, wall, carrier


def _reconstruct(inp, bg, wall, carrier):
    """Markers are waypoints; bridge only one/two-cell aligned gaps from a snapshot."""
    observed = set(map(tuple, np.argwhere((inp == carrier) | (inp == REPAIR))))
    cells = set(observed)
    for p in sorted(observed):
        for d in ((1, 0), (0, 1)):
            for distance in (2, 3):
                q = p[0]+distance*d[0], p[1]+distance*d[1]
                if q not in observed:
                    continue
                between = _line(p, q)[1:-1]
                if all(int(inp[x]) in (bg, wall) for x in between):
                    perpendicular = (-d[1], d[0])
                    def turning(x):
                        return any(_add(x, t) in observed for t in (perpendicular, (-perpendicular[0], -perpendicular[1])))
                    if not (turning(p) and turning(q)):
                        cells.update(between)
    assert cells and all(_inside(p, inp.shape) for p in cells)
    assert all(len(_neighbors(p, cells)) <= 2 for p in cells), 'input paths must not branch'
    for component in _components(_mask(inp.shape, cells)):
        s = set(map(tuple, np.argwhere(component)))
        assert len(s) >= 3 and sum(len(_neighbors(p, s)) == 1 for p in s) == 2, 'input component must be a simple path'
    return cells


def _mask(shape, cells):
    result = np.zeros(shape, dtype=bool)
    for p in cells:
        result[p] = True
    return result


def _edits(inp, cells):
    result = []
    for q in map(tuple, np.argwhere(inp == DIRECTION)):
        candidates = []
        for d in D4:
            pivot = q[0]-d[0], q[1]-d[1]
            start = pivot[0]-d[0], pivot[1]-d[1]
            if pivot not in cells or start not in cells or q in cells:
                continue
            side = (-d[1], d[0])
            if not any(_add(pivot, t) in cells for t in (side, (-side[0], -side[1]))):
                continue
            old, previous, current = [], pivot, start
            while current not in old:
                old.append(current)
                following = [p for p in _neighbors(current, cells) if p != previous]
                if not following:
                    if _terminal(current, inp.shape):
                        candidates.append({'pivot': pivot, 'direction': d, 'old': set(old),
                                           'walk': [pivot]+old, 'new': _ray(pivot, d, inp.shape)})
                    break
                if len(following) != 1:
                    break
                previous, current = current, following[0]
        assert len(candidates) == 1, 'direction marker must identify exactly one terminal branch'
        result.append(candidates[0])
    for i, a in enumerate(result):
        for b in result[i+1:]:
            assert not a['old'] & b['old']
            assert a['pivot'] not in b['old'] and b['pivot'] not in a['old']
    return result


def _final(cells, edits):
    removed = set().union(*(e['old'] for e in edits)) if edits else set()
    rays = set().union(*(e['new'] for e in edits)) if edits else set()
    return (cells - removed) | rays


def _key_paths(mask, depth):
    h, w = mask.shape
    def inside(r, c):
        return 0 <= r < h and 0 <= c < w and mask[r, c]
    result = []
    for r0, c0 in np.argwhere(mask & (depth == 0)):
        r, c = int(r0), int(c0)
        for dr in (-1, 1):
            for dc in (-1, 1):
                if inside(r-dr, c) or inside(r, c-dc):
                    continue
                if not all(inside(*q) for q in ((r+dr, c), (r, c+dc), (r+dr, c+dc))):
                    continue
                length = 0
                while inside(r+length*dr, c+length*dc) and depth[r+length*dr, c+length*dc] == length:
                    length += 1
                result.append((r, c, dr, dc, length))
    return result


def _decode(inp, region, depth, roles):
    bg, wall, carrier = roles
    cues = set(map(tuple, np.argwhere(region & ~np.isin(inp, (bg, wall, carrier, REPAIR, DIRECTION)))))
    if not cues:
        return (bg,), set(), ()
    length = max(int(depth[p]) for p in cues)+1
    possibilities = []
    for r, c, dr, dc, capacity in _key_paths(region, depth):
        if capacity < length:
            continue
        positions = [(r+d*dr, c+d*dc) for d in range(length)]
        if not cues <= set(positions):
            continue
        period = tuple(int(inp[p]) for p in positions)
        if any(v in (wall, carrier, REPAIR, DIRECTION) for v in period) or period[-1] == bg:
            continue
        possibilities.append((period, set(positions), (dr, dc)))
    assert possibilities and len({p[0] for p in possibilities}) == 1, 'key must uniquely determine its period'
    return possibilities[0][0], set().union(*(p[1] for p in possibilities)), tuple(sorted({p[2] for p in possibilities}))


def _render(inp, roles, final, field):
    bg, wall, carrier = roles
    out = inp.copy()
    out[np.isin(inp, (carrier, REPAIR, DIRECTION))] = bg
    for p in final:
        if inp[p] != wall:
            out[p] = field[p]
    return out


def _analyze(inp):
    roles = _roles(inp)
    bg, wall, carrier = roles
    regions = []
    field = np.full(inp.shape, bg, dtype=np.int8)
    for region in _components(inp != wall):
        depth = _depth(region)
        period, key, orientations = _decode(inp, region, depth, roles)
        field[region] = np.asarray(period, dtype=np.int8)[depth[region] % len(period)]
        regions.append({'mask': region, 'depth': depth, 'period': period, 'key': key,
                        'orientations': orientations})
    cells = _reconstruct(inp, *roles)
    edits = _edits(inp, cells)
    final = _final(cells, edits)
    return {'roles': roles, 'regions': regions, 'field': field, 'cells': cells,
            'edits': edits, 'final': final, 'prediction': _render(inp, roles, final, field)}


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode roles, walls, keys and route edits exclusively from this input."""
    return _analyze(_grid(input_grid))['prediction']


def _walls(rng, shape, target, reserve=False):
    h, w = shape
    walls = np.zeros(shape, dtype=bool)
    walls[0] = walls[-1] = True
    walls[:, 0] = walls[:, -1] = True
    cut = int(rng.integers(10, min(15, w-6))) if reserve else int(rng.integers(4 if w <= 14 else 6, w-(4 if w <= 14 else 6)))
    walls[:, cut] = True
    rooms = [(1, h-2, 1, cut-1), (1, h-2, cut+1, w-2)]
    reserved = rooms[0] if reserve else None
    while len(rooms) < target:
        options = [(i, axis) for i, room in enumerate(rooms) if room != reserved
                   for axis, length in ((0, room[1]-room[0]+1), (1, room[3]-room[2]+1)) if length >= 9]
        if not options:
            break
        i, axis = options[int(rng.integers(len(options)))]
        a, b, c, d = rooms.pop(i)
        if axis == 0:
            q = int(rng.integers(a+4, b-3))
            walls[q, c-1:d+2] = True
            rooms.extend(((a, q-1, c, d), (q+1, b, c, d)))
        else:
            q = int(rng.integers(c+4, d-3))
            walls[a-1:b+2, q] = True
            rooms.extend(((a, b, c, q-1), (a, b, q+1, d)))
    # Offset isolated intervals of any straight wall. This composes concavities
    # with subdivisions without replacing the scene by a fixed layout bank.
    for _ in range(int(rng.integers(1, 5))):
        for _ in range(35):
            work = walls.T if rng.integers(2) else walls
            r, c = map(int, np.argwhere(work)[int(rng.integers(work.sum()))])
            end = r+int(rng.integers(4, 10))
            shift = int(rng.choice((-2, -1, 1, 2)))
            lo, hi = sorted((c, c+shift))
            if r < 2 or end+1 >= work.shape[0]-1 or lo < 3 or hi+2 >= work.shape[1]-1:
                continue
            block = work[r-1:end+2, lo-1:hi+2]
            expected = np.zeros(block.shape, bool)
            expected[:, c-lo+1] = True
            if not np.array_equal(block, expected):
                continue
            work[r+1:end, c] = False
            work[r, lo:hi+1] = work[end, lo:hi+1] = True
            work[r:end+1, c+shift] = True
            break
    return walls


def _path(rng, shape):
    h, w = shape
    side = int(rng.integers(4))
    if side < 2:
        start = (1 if side == 0 else h-2, int(rng.integers(2, w-2)))
        axis = 0
    else:
        start = (int(rng.integers(2, h-2)), 1 if side == 2 else w-2)
        axis = 1
    vertices = [start]
    for _ in range(int(rng.integers(2, 6))):
        options = [v for v in range(2, shape[axis]-2) if abs(v-vertices[-1][axis]) >= 3]
        rng.shuffle(options)
        for v in options:
            q = list(vertices[-1])
            q[axis] = v
            candidate = vertices+[tuple(q)]
            walk = _polyline(candidate)
            cells = set(walk)
            if len(cells) == len(walk) and all(len(_neighbors(p, cells)) <= 2 for p in cells):
                vertices = candidate
                break
        else:
            break
        axis = 1-axis
    return vertices


def _routes(rng, walls, count, redirect_count):
    cells, edits = set(), []
    for index in range(count):
        accepted = False
        for _ in range(90):
            vertices = _path(rng, walls.shape)
            if len(vertices) < 3:
                continue
            walk = _polyline(vertices)
            s = set(walk)
            if any(p in cells or _neighbors(p, cells) for p in s):
                continue
            options = []
            if index < redirect_count:
                for j in range(1, len(vertices)-1):
                    pivot, previous = vertices[j], vertices[j-1]
                    d = (int(np.sign(pivot[0]-previous[0])), int(np.sign(pivot[1]-previous[1])))
                    q = _add(pivot, d)
                    if walls[pivot] or walls[q] or q in s or q in cells:
                        continue
                    if _neighbors(q, cells | s) != [pivot]:
                        continue
                    at = walk.index(pivot)
                    options.append({'pivot': pivot, 'direction': d, 'old': set(walk[:at]),
                                    'walk': list(reversed(walk[:at+1])), 'new': _ray(pivot, d, walls.shape)})
                if not options:
                    continue
                edits.append(options[int(rng.integers(len(options)))])
            cells.update(s)
            accepted = True
            break
        if not accepted:
            return None
    return cells, edits


def _turns(walk):
    directions = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(walk, walk[1:])]
    return sum(a != b for a, b in zip(directions, directions[1:]))


def _same_edits(a, b):
    return {(e['pivot'], e['direction'], frozenset(e['old'])) for e in a} == {
        (e['pivot'], e['direction'], frozenset(e['old'])) for e in b}


def _witness(a, kind):
    final_mask = _mask(a['field'].shape, a['final'])
    for region in a['regions']:
        period = np.asarray(region['period'])
        sample = region['mask'] & final_mask
        if len(period) < 2 or not sample.any():
            continue
        rr, cc = np.nonzero(region['mask'])
        if kind == 'box':
            r, c = np.indices(region['mask'].shape)
            alternative = np.minimum.reduce((r-rr.min(), rr.max()-r, c-cc.min(), cc.max()-c))
        elif kind == 'manhattan':
            alternative = _depth(region['mask'], False)
        elif kind == 'uniform':
            alternative = np.zeros(region['mask'].shape, dtype=int)
        else:
            raise ValueError(kind)
        if np.any(period[alternative[sample] % len(period)] != a['field'][sample]):
            return True
    return False


def _clock_witness(a):
    # Adjacent samples at the same contour depth disprove a moving step clock.
    for r in a['regions']:
        period = r['period']
        if len(period) < 2 or any(period[i] == period[(i+1) % len(period)] for i in range(len(period))):
            continue
        for p in a['final']:
            if r['mask'][p] and any(q in a['final'] and r['mask'][q] and r['depth'][q] == r['depth'][p]
                                     for q in _neighbors(p, a['final'])):
                return True
    return False


def _repair_witness(a, inp):
    bg, wall, _ = a['roles']
    return all(any(inp[p] == value and a['field'][p] != bg and p in a['cells']
                   for p in a['final'] if inp[p] != wall) for value in (bg, REPAIR))


def _each_edit_matters(a, inp):
    return all(not np.array_equal(a['prediction'], _render(inp, a['roles'],
               _final(a['cells'], a['edits'][:i]+a['edits'][i+1:]), a['field']))
               for i in range(len(a['edits'])))


def _simultaneous_witness(a, inp):
    bg, wall, _ = a['roles']
    return any(inp[p] != wall and a['field'][p] != bg for i, e in enumerate(a['edits'])
               for j, f in enumerate(a['edits']) if i != j for p in e['new'] & f['old']
               if p not in f['new'])


def _deleted_witness(a, inp):
    bg, wall, _ = a['roles']
    for e in a['edits']:
        walk = e['walk']
        bends = [i for i in range(1, len(walk)-1) if
                 (walk[i][0]-walk[i-1][0], walk[i][1]-walk[i-1][1]) !=
                 (walk[i+1][0]-walk[i][0], walk[i+1][1]-walk[i][1])]
        if bends and any(p not in a['final'] and inp[p] != wall and a['field'][p] != bg
                         for p in walk[bends[0]+1:]):
            return True
    return False


def _cross_room(a):
    # Two distinct keyed rooms contribute visible samples to the same new ray.
    bg = a['roles'][0]
    for e in a['edits']:
        periods = {r['period'] for r in a['regions'] if r['key'] and any(
            r['mask'][p] and a['field'][p] != bg for p in e['new'])}
        if len(periods) >= 2:
            return True
    return False


def _no_reroute(a, inp):
    return _render(inp, a['roles'], a['cells'], a['field'])


def _make_pair(rng, index, is_test=False):
    """Bounded rejection of ambiguities; labels are drawn from latent geometry."""
    first, second = not is_test and index == 0, not is_test and index == 1
    for attempt in range(500):
        if second:
            shape = int(rng.integers(24, 31)), int(rng.integers(24, 31))
            target, count, redirects = int(rng.integers(3, 6)), 2, 2
        elif first:
            shape = int(rng.integers(18, 29)), int(rng.integers(18, 29))
            target, count, redirects = 2, 1, 1
        else:
            low = 18 if is_test else 12
            shape = int(rng.integers(low, 31)), int(rng.integers(low, 31))
            target = int(rng.integers(2, 7))
            count = int(rng.integers(1, 4))
            redirects = int(rng.integers(1 if is_test else 0, count+1))
        colors = list(map(int, rng.permutation([0, 1, 4, 5, 6, 7, 8, 9])))
        bg, wall, carrier = colors[:3]
        palette = colors[3:]
        walls = _walls(rng, shape, target, second)
        scene = _routes(rng, walls, count, redirects)
        if scene is None:
            continue
        cells, edits = scene
        if second and not any(walls[p] for p in cells):
            continue
        if second and (not any(_turns(e['walk']) for e in edits) or not any(
                a['new'] & b['old'] for i, a in enumerate(edits) for j, b in enumerate(edits) if i != j)):
            continue
        inp = np.full(shape, bg, dtype=np.int8)
        inp[walls] = wall
        for p in cells:
            if not walls[p]:
                inp[p] = carrier
        for e in edits:
            inp[_add(e['pivot'], e['direction'])] = DIRECTION
        try:
            assert _reconstruct(inp, bg, wall, carrier) == cells
            assert _same_edits(_edits(inp, cells), edits)
        except AssertionError:
            continue
        final = _final(cells, edits)
        regions = _components(~walls)
        depths = [_bfs_depth(region) for region in regions]
        occupied = cells | {_add(e['pivot'], e['direction']) for e in edits}
        paths = []
        for region, depth in zip(regions, depths):
            options = []
            for r, c, dr, dc, capacity in _key_paths(region, depth):
                if dr != (-1 if is_test else 1):
                    continue
                usable = 0
                for d in range(min(capacity, 4)):
                    if (r+d*dr, c+d*dc) in occupied:
                        break
                    usable += 1
                if usable:
                    options.append((r, c, dr, dc, usable))
            paths.append(options)
        capacity = [max((p[4] for p in ps), default=0) for ps in paths]
        sample_depth = [max((int(depth[p]) for p in final if region[p]), default=-1)
                        for region, depth in zip(regions, depths)]
        assignments = {}
        if first or second:
            length = 3 if second else 2
            primary = [i for i in range(len(regions)) if capacity[i] >= length and sample_depth[i] >= length]
            if not primary:
                continue
            primary = int(rng.choice(primary))
            assignments[primary] = (length, 1 if second else None)
            others = [i for i in range(len(regions)) if i != primary and capacity[i] >= 1]
            if not others or (second and len(regions) < 3):
                continue
            assignments[others[0]] = (min(2 if second else 1, capacity[others[0]]), None)
            if second:
                blank = next(i for i in range(len(regions)) if i not in assignments)
                assignments[blank] = (0, None)
        if is_test:
            candidates = [i for i in range(len(regions)) if capacity[i] >= 2 and sample_depth[i] >= 1]
            if not candidates:
                continue
            primary = int(rng.choice(candidates))
            assignments[primary] = (int(rng.integers(2, min(4, capacity[primary])+1)), None)
        periods = []
        for i, (region, depth, options) in enumerate(zip(regions, depths, paths)):
            length, gap = assignments.get(i, (0 if not options or rng.random() < .16 else
                                             int(rng.integers(1, min(4 if is_test else 3, capacity[i])+1)), None))
            if length == 0:
                periods.append((bg,))
                continue
            available = [p for p in options if p[4] >= length]
            r, c, dr, dc, _ = available[int(rng.integers(len(available)))]
            period = list(map(int, rng.choice(palette, length, replace=False)))
            if gap is not None:
                period[gap] = bg
            elif length >= 2 and rng.random() < .28:
                period[int(rng.integers(length-1))] = bg
            for d, color in enumerate(period):
                inp[r+d*dr, c+d*dc] = color
            periods.append(tuple(period))
        # Waypoint markers may annotate bends, endpoints, pivots, or straight cells.
        candidates = [p for p in cells if inp[p] == carrier]
        rng.shuffle(candidates)
        mandatory = [edits[0]['pivot']] if second else []
        if second:
            old_visible = [p for e in edits for p in e['old'] if inp[p] == carrier]
            if old_visible:
                mandatory.append(old_visible[int(rng.integers(len(old_visible)))])
        marked = list(dict.fromkeys(mandatory+candidates))[:int(rng.integers(2, min(9, len(candidates))+1))]
        for p in marked:
            inp[p] = REPAIR
        # Accept omissions only if snapshot reconstruction recovers all paths.
        gap_sites = []
        rng.shuffle(candidates)
        desired = int(rng.integers(1, 5))
        for p in candidates:
            ns = _neighbors(p, cells)
            if inp[p] != carrier or len(ns) != 2 or (ns[0][0]+ns[1][0], ns[0][1]+ns[1][1]) != (2*p[0], 2*p[1]):
                continue
            inp[p] = bg
            try:
                assert _reconstruct(inp, bg, wall, carrier) == cells
            except AssertionError:
                inp[p] = carrier
            else:
                gap_sites.append(p)
                if len(gap_sites) >= desired:
                    break
        if (first or second) and not gap_sites:
            continue
        field = np.full(shape, bg, dtype=np.int8)
        for region, depth, period in zip(regions, depths, periods):
            field[region] = np.asarray(period, dtype=np.int8)[depth[region] % len(period)]
        out = _render(inp, (bg, wall, carrier), final, field)
        try:
            analysis = _analyze(inp)
            assert analysis['cells'] == cells and _same_edits(analysis['edits'], edits)
            assert np.array_equal(analysis['prediction'], out)
            assert not np.array_equal(inp, out)
            assert _each_edit_matters(analysis, inp)
            if first:
                assert _witness(analysis, 'uniform') and _witness(analysis, 'box') and _witness(analysis, 'manhattan')
                assert _cross_room(analysis) and _clock_witness(analysis) and _repair_witness(analysis, inp)
            if second:
                assert _deleted_witness(analysis, inp) and _simultaneous_witness(analysis, inp)
                assert any(len(r['period']) == 3 and r['period'][1] == bg and
                           any(r['mask'][p] and r['depth'][p] >= 3 for p in final) for r in analysis['regions'])
            if is_test:
                assert any(len(r['period']) >= 2 and r['orientations'] and all(d[0] == -1 for d in r['orientations'])
                           for r in analysis['regions'])
        except AssertionError:
            continue
        return {'input': inp, 'output': out}
    raise RuntimeError('bounded construction search exhausted')


def _integer(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or not low <= value <= high:
        raise ValueError(f'{name} must be an integer in {low}..{high}')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            for _ in range(20):
                pair = _make_pair(rng, i, split == 'test')
                key = _fingerprint(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input search exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, train_analyses = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = _grid(pair['input']), _grid(pair['output'])
            assert inp.shape == out.shape
            key = _fingerprint(inp)
            assert key not in seen
            seen.add(key)
            a = _analyze(inp)
            bg, wall, carrier = a['roles']
            wall_mask = inp == wall
            assert len(_components(wall_mask)) == 1, 'wall network must be connected'
            assert not np.any(wall_mask[:-1, :-1] & wall_mask[1:, :-1] & wall_mask[:-1, 1:] & wall_mask[1:, 1:])
            assert np.array_equal(out, a['prediction']), 'complete output disagrees with input semantics'
            assert np.array_equal(out == wall, inp == wall), 'walls must stay visible over every crossing'
            assert not np.any(np.isin(out, (carrier, REPAIR, DIRECTION)))
            assert not np.array_equal(inp, out)
            assert all(len(r['period']) <= 4 for r in a['regions'])
            for r in a['regions']:
                assert np.array_equal(_bfs_depth(r['mask']), r['depth'])
                for p in r['key']:
                    assert out[p] == inp[p], 'calibration key must be unchanged'
            if a['edits']:
                assert not np.array_equal(_no_reroute(a, inp), out), 'rerouting must affect the visible answer'
                assert _each_edit_matters(a, inp), 'every direction marker must affect the visible answer'
            if split == 'train':
                train_analyses.append(a)
    # Evidence requirements across the training set, never a requirement that
    # every generic family member has the generator's demonstration layout.
    assert any(_witness(a, 'box') and _witness(a, 'manhattan') for a in train_analyses)
    assert any(_cross_room(a) and _clock_witness(a) for a in train_analyses)
    assert any(_repair_witness(a, np.asarray(p['input'])) for p, a in zip(puzzle['train'], train_analyses))
    assert any(_deleted_witness(a, np.asarray(p['input'])) and _simultaneous_witness(a, np.asarray(p['input']))
               for p, a in zip(puzzle['train'], train_analyses))
    assert any(any(np.asarray(p['input'])[q] == a['roles'][1] for q in a['cells'])
               for p, a in zip(puzzle['train'], train_analyses))
    for length in (1, 2, 3):
        assert any(len(r['period']) == length and r['key'] for a in train_analyses for r in a['regions'])
    assert any(len(r['period']) == 3 and r['period'][1] == a['roles'][0] for a in train_analyses for r in a['regions'])
    assert any(not r['key'] for a in train_analyses for r in a['regions'])
    assert any(len(a['edits']) >= 2 and any(_turns(e['walk']) for e in a['edits']) for a in train_analyses)
    assert any(inp[e['pivot']] == REPAIR for inp, a in
               ((np.asarray(p['input']), x) for p, x in zip(puzzle['train'], train_analyses)) for e in a['edits'])


def self_test():
    """Bounded semantic, negative, equivariance, and 42-seed diversity checks."""
    puzzles = [generate_puzzle(s) for s in range(42)]
    shapes, walls, roles, region_counts, directions, periods = (set() for _ in range(6))
    signatures, route_counts, bends, repair_sites = set(), set(), set(), set()
    crossing_rays, cross_room_rays, gaps, neutral_samples = 0, 0, 0, 0
    for puzzle in puzzles:
        signatures.add(tuple(_fingerprint(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                a = _analyze(inp)
                bg, wall, carrier = a['roles']
                shapes.add(inp.shape)
                walls.add(_fingerprint((inp == wall).astype(np.int8)))
                roles.add(a['roles'])
                region_counts.add(len(a['regions']))
                route_counts.add(len(_components(_mask(inp.shape, a['cells']))))
                directions.update(e['direction'] for e in a['edits'])
                bends.update(_turns(e['walk']) for e in a['edits'])
                periods.update(len(r['period']) for r in a['regions'])
                crossing_rays += any(e['new'] & f['new'] for i, e in enumerate(a['edits']) for f in a['edits'][i+1:])
                for e in a['edits']:
                    cross_room_rays += sum(bool(e['new'] & set(map(tuple, np.argwhere(r['mask'])))) for r in a['regions']) >= 2
                gaps += sum(inp[p] == bg for p in a['cells'])
                neutral_samples += sum(inp[p] != wall and a['field'][p] == bg for p in a['final'])
                for p in map(tuple, np.argwhere(inp == REPAIR)):
                    ns = _neighbors(p, a['cells'])
                    repair_sites.add('endpoint' if len(ns) == 1 else 'straight' if
                                     (ns[0][0]+ns[1][0], ns[0][1]+ns[1][1]) == (2*p[0], 2*p[1]) else 'bend')
                assert a['final'] == _final(a['cells'], list(reversed(a['edits'])))
                # Symmetry must commute with the complete transformation.
                for k in (1, 2, 3):
                    assert np.array_equal(solve(np.rot90(inp, k)), np.rot90(out, k))
                assert np.array_equal(solve(np.fliplr(inp)), np.fliplr(out))
    assert len(signatures) == 42 and len(shapes) >= 65 and len(walls) >= 150
    assert len(roles) >= 50 and len(region_counts) >= 4
    assert directions == set(D4) and periods == {1, 2, 3, 4}
    assert route_counts == {1, 2, 3} and {0, 1, 2} <= bends
    assert repair_sites == {'endpoint', 'straight', 'bend'}
    assert crossing_rays >= 5 and cross_room_rays >= 60 and gaps >= 200 and neutral_samples >= 200
    assert max(max(s) for s in shapes) == 30 and min(min(s) for s in shapes) <= 14
    assert tuple(_fingerprint(p['input']) for p in generate_puzzle(8)['train']) == tuple(
        _fingerprint(p['input']) for p in puzzles[8]['train'])
    # Legal count boundaries and invalid types/ranges/relations.
    for nt, nq in ((2, 1), (10, 1), (10, 4), (5, 4)):
        validate_generated_puzzle(generate_puzzle(71+nt+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Negative labels at preserved walls, keys, deletions, samples, and empty cells.
    for puzzle in puzzles[:3]:
        pair = puzzle['test'][0]
        inp, out = pair['input'], pair['output']
        a = _analyze(inp)
        sites = [(0, 0), min(a['final'])]
        sites += [p for r in a['regions'] for p in sorted(r['key'])[:1]]
        sites += [min(e['old']) for e in a['edits']]
        sites += [tuple(map(int, p)) for p in np.argwhere(inp == out)[:1]]
        for p in sites:
            corrupted = out.copy()
            corrupted[p] = (int(out[p])+1) % 10
            broken = {'train': puzzle['train'], 'test': [{'input': inp, 'output': corrupted}]}
            try:
                validate_generated_puzzle(broken)
            except AssertionError:
                pass
            else:
                raise AssertionError('corrupt label accepted')
    return {'seeds': 42, 'pairs': 210, 'shapes': len(shapes), 'wall_geometries': len(walls),
            'role_assignments': len(roles), 'region_counts': sorted(region_counts),
            'route_counts': sorted(route_counts), 'old_bend_counts': sorted(bends),
            'crossing_ray_pairs': crossing_rays, 'cross_room_rays': cross_room_rays,
            'recoverable_empty_gaps': gaps, 'neutral_wire_samples': neutral_samples}
