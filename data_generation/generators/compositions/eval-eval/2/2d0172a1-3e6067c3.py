"""Keyed journeys through shrinking, color-locked rooms.

No parent imports, files, or generation metadata are used by the solver.
All geometry and conventions needed for decoding are visible in each input.
"""
from __future__ import annotations

import copy
import numpy as np


def _check_grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= d <= 30 for d in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    h, w = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(np.array(cells, dtype=int))
    return result


def _hole(mask):
    a = np.pad(mask, 1)
    outside = np.zeros(a.shape, bool)
    outside[0, 0] = True
    stack = [(0, 0)]
    h, w = a.shape
    while stack:
        r, c = stack.pop()
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= rr < h and 0 <= cc < w and not a[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                stack.append((rr, cc))
    return (~a & ~outside)[1:-1, 1:-1]


def _segment(a, b, shape):
    """Open center-to-center segment, excluding the two terminal centers."""
    mask = np.zeros(shape, bool)
    if a[0] == b[0]:
        mask[a[0], min(a[1], b[1])+1:max(a[1], b[1])] = True
    elif a[1] == b[1]:
        mask[min(a[0], b[0])+1:max(a[0], b[0]), a[1]] = True
    else:
        return None
    return mask


def _parse(grid):
    g = _check_grid(grid)
    bg = int(g[0, 0])
    assert all(np.all(edge == bg) for edge in (g[0], g[-1], g[:, 0], g[:, -1]))
    occupied_rows = np.flatnonzero(np.any(g != bg, axis=1))
    assert len(occupied_rows)
    ky = int(occupied_rows[-1])
    xs = np.flatnonzero(g[ky] != bg)
    assert 3 <= len(xs) <= 8 and np.all(np.diff(xs) >= 2)
    assert ky >= 2 and np.all(g[ky-1] == bg)
    key = [int(g[ky, x]) for x in xs]
    assert len(set(key)) >= 2
    scene = g.copy()
    scene[ky] = bg
    rooms, dots = [], []
    owner = np.zeros(g.shape, dtype=int)
    for serial, cells in enumerate(_components(scene != bg), 1):
        lo, hi = cells.min(0), cells.max(0)
        mask = np.zeros(g.shape, bool)
        mask[cells[:, 0], cells[:, 1]] = True
        owner[mask] = serial
        local = mask[lo[0]:hi[0]+1, lo[1]:hi[1]+1]
        inside = _hole(local)
        record = dict(mask=mask, lo=tuple(lo), hi=tuple(hi))
        if inside.any():
            assert len(_components(inside)) == 1
            hole = np.zeros(g.shape, bool)
            hole[lo[0]:hi[0]+1, lo[1]:hi[1]+1] = inside
            colors, counts = np.unique(scene[mask], return_counts=True)
            wall = int(colors[np.argmax(counts)])
            assert counts.max() > len(cells)/2
            rooms.append(dict(record, hole=hole, wall=wall))
        else:
            colors = np.unique(scene[mask])
            assert len(colors) == 1
            dims = hi-lo+1
            assert np.all(dims % 2 == 1) and np.all(dims <= 3)
            assert mask.sum() >= np.prod(dims)/2
            center = tuple(((lo+hi)//2).tolist())
            assert mask[center]
            dots.append(dict(record, center=center, color=int(colors[0])))
    # Components may not touch even at a corner; gates belong to their wall.
    for a, b in ((owner[:-1], owner[1:]), (owner[:, :-1], owner[:, 1:]),
                 (owner[:-1, :-1], owner[1:, 1:]), (owner[:-1, 1:], owner[1:, :-1])):
        assert not np.any((a > 0) & (b > 0) & (a != b))
    assert 1 <= len(rooms) <= 3 and 4 <= len(dots) <= 9
    wall = rooms[0]['wall']
    assert wall != bg and all(r['wall'] == wall for r in rooms)
    assert all(d['color'] not in (bg, wall) for d in dots)
    assert all(k not in (bg, wall) for k in key)
    assert max(r['hi'][0] for r in rooms) < ky-1
    assert max(d['hi'][0] for d in dots) < ky-1
    # Actual enclosed space, rather than a bounding-box containment heuristic.
    for r in rooms:
        for other in rooms + dots:
            if other is r:
                continue
            flags = r['hole'][other['mask']]
            assert flags.all() or not flags.any()
        assert any(r['hole'][d['center']] for d in dots), 'empty input room'
        for d in dots:
            y, x = d['center']
            in_box = r['lo'][0] < y < r['hi'][0] and r['lo'][1] < x < r['hi'][1]
            assert in_box == bool(r['hole'][y, x]), 'a deep concavity changes rectangle containment'
    for i, a in enumerate(rooms):
        for b in rooms[i+1:]:
            apart = a['hi'][0] < b['lo'][0] or b['hi'][0] < a['lo'][0] or a['hi'][1] < b['lo'][1] or b['hi'][1] < a['lo'][1]
            ainb = all(b['lo'][k] < a['lo'][k] <= a['hi'][k] < b['hi'][k] for k in (0, 1))
            bina = all(a['lo'][k] < b['lo'][k] <= b['hi'][k] < a['hi'][k] for k in (0, 1))
            assert apart or ainb or bina
    assert not ({d['center'][0] for d in dots} &
                {v for r in rooms for v in (r['lo'][0], r['hi'][0])})
    assert not ({d['center'][1] for d in dots} &
                {v for r in rooms for v in (r['lo'][1], r['hi'][1])})
    dots.sort(key=lambda d: d['center'])
    rooms.sort(key=lambda r: (r['lo'], r['hi']))
    return dict(grid=g, background=bg, wall=wall, key=key, key_y=ky,
                rooms=rooms, dots=dots)


def _graph(info, ignore_locks=False):
    dots, g = info['dots'], info['grid']
    graph = [[] for _ in dots]
    walls = np.zeros(g.shape, bool)
    for room in info['rooms']:
        walls |= room['mask']
    for i, a in enumerate(dots):
        for j, b in enumerate(dots):
            if i == j:
                continue
            line = _segment(a['center'], b['center'], g.shape)
            if line is None or not line.any():
                continue
            if any(np.any(line & d['mask']) for k, d in enumerate(dots) if k not in (i, j)):
                continue
            if ignore_locks or np.all(g[line & walls] == a['color']):
                graph[i].append(j)
    return graph


def _routes(graph, labels, key, limit=2):
    result = []
    candidates = [[i for i, color in enumerate(labels) if color == k] for k in key]
    def visit(path, used):
        if len(result) >= limit:
            return
        if len(path) == len(key):
            result.append(tuple(path))
            return
        for i in candidates[len(path)]:
            if i not in used and (not path or i in graph[path[-1]]):
                visit(path+[i], used | {i})
    visit([], set())
    return result


def _retained(info, route):
    centers = [info['dots'][i]['center'] for i in route]
    lines = [_segment(a, b, info['grid'].shape) for a, b in zip(centers, centers[1:])]
    return [r for r in info['rooms'] if any(r['hole'][p] for p in centers)
            or any(np.any(r['hole'] & line) for line in lines)]


def _render(info, route, keep_all=False, target_colors=False, omit_rooms=False):
    rooms = [] if omit_rooms else info['rooms'] if keep_all else _retained(info, route)
    dots = info['dots'] if keep_all else [info['dots'][i] for i in route]
    rows = sorted({d['center'][0] for d in dots} |
                  {v for r in rooms for v in (r['lo'][0], r['hi'][0])})
    cols = sorted({d['center'][1] for d in dots} |
                  {v for r in rooms for v in (r['lo'][1], r['hi'][1])})
    ry, cx = ({v: 1+2*i for i, v in enumerate(axis)} for axis in (rows, cols))
    out = np.full((2*len(rows)+1, 2*len(cols)+1), info['background'], np.int8)
    assert max(out.shape) <= 30
    for r in rooms:
        y0, x0 = ry[r['lo'][0]], cx[r['lo'][1]]
        y1, x1 = ry[r['hi'][0]], cx[r['hi'][1]]
        out[y0, x0:x1+1] = out[y1, x0:x1+1] = info['wall']
        out[y0:y1+1, x0] = out[y0:y1+1, x1] = info['wall']
    for d in dots:
        y, x = d['center']
        out[ry[y], cx[x]] = d['color']
    painted = np.zeros(out.shape, bool)
    for i, j in zip(route, route[1:]):
        a, b = info['dots'][i], info['dots'][j]
        ay, ax = a['center']; by, bx = b['center']
        line = _segment((ry[ay], cx[ax]), (ry[by], cx[bx]), out.shape)
        assert not np.any(painted & line), 'crossing or overlapping route'
        painted |= line
        out[line] = b['color'] if target_colors else a['color']
    return out


def _decode(grid):
    info = _parse(grid)
    labels = [d['color'] for d in info['dots']]
    routes = _routes(_graph(info), labels, info['key'])
    assert len(routes) == 1, 'key must select one distinct-terminal journey'
    return info, routes[0]


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from pixels only: gates select the journey that selects the map."""
    info, route = _decode(input_grid)
    return _render(info, route)


def _compatible(a, b):
    apart = a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]
    ainb = b[0] <= a[0] <= a[2] <= b[2] and b[1] <= a[1] <= a[3] <= b[3]
    bina = a[0] <= b[0] <= b[2] <= a[2] and a[1] <= b[1] <= b[3] <= a[3]
    return apart or ainb or bina


def _boxes(rng, cells, count, witness):
    nr, nc = max(r for r, c in cells)+1, max(c for r, c in cells)+1
    options = [(a, b, c, d) for a in range(nr) for b in range(nc)
               for c in range(a, nr) for d in range(b, nc)
               if any(a <= r <= c and b <= s <= d for r, s in cells)]
    if witness == 1:
        # Target the geometric relation needed for a transit demonstration,
        # rather than relying on a rare accidental empty position in a room.
        transit = []
        for box in options:
            for i, a in enumerate(cells):
                for b in cells[i+1:]:
                    horizontal = a[0] == b[0]
                    vertical = a[1] == b[1]
                    if not (horizontal or vertical):
                        continue
                    axis, cross = (1, 0) if horizontal else (0, 1)
                    lo, hi = sorted((a[axis], b[axis]))
                    if any(p[cross] == a[cross] and lo < p[axis] < hi for p in cells):
                        continue
                    if box[cross] <= a[cross] <= box[cross+2] and lo < box[axis] <= box[axis+2] < hi:
                        transit.append(box)
                        break
                else:
                    continue
                break
        if not transit:
            return None
        for index in rng.permutation(len(transit)):
            a = transit[int(index)]
            apart = [b for b in options if a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1]]
            if apart:
                b = apart[int(rng.integers(len(apart)))]
                return [(10*r0-4, 10*c0-4, 10*r1+4, 10*c1+4) for r0, c0, r1, c1 in (a, b)]
        return None
    for _ in range(40):
        chosen = [options[int(rng.integers(len(options)))]]
        for _ in range(count-1):
            eligible = [b for b in options if all(_compatible(a, b) for a in chosen)]
            chosen.append(eligible[int(rng.integers(len(eligible)))])
        # Equal ranges are separate nested rooms, with larger first index outside.
        depths = [sum(a != b and a[0] <= b[0] <= b[2] <= a[2]
                      and a[1] <= b[1] <= b[3] <= a[3] or a == b and j < i
                      for j, a in enumerate(chosen) if i != j)
                  for i, b in enumerate(chosen)]
        if witness == 0 and max(depths) != 1:
            continue
        if witness == 1 and max(depths) != 0:
            continue
        return [(10*a-(4-depth), 10*b-(4-depth),
                 10*c+(4-depth), 10*d+(4-depth))
                for (a, b, c, d), depth in zip(chosen, depths)]
    return None


def _axis(rng, events, maximum):
    events = sorted(set(events))
    steps = np.full(len(events)-1, 2, dtype=int)
    spare = maximum-(2*len(events)+1)
    if spare < 0:
        return None
    for j in rng.permutation(len(steps)):
        extra = int(rng.integers(0, min(3, spare)+1))
        steps[j] += extra
        spare -= extra
    values = [1] + (1+np.cumsum(steps)).tolist()
    return dict(zip(events, values))


def _room_mask(shape, box, cut):
    y0, x0, y1, x1 = box
    h, w = y1-y0+1, x1-x0+1
    yy, xx = np.indices((h, w))
    filled = np.minimum(yy, h-1-yy) + np.minimum(xx, w-1-xx) >= cut
    p = np.pad(filled, 1)
    # Eight-neighbor erosion makes staircase corners four-connected: using
    # only cardinal erosion would leave the four sides touching diagonally.
    eroded = filled.copy()
    for dy in range(3):
        for dx in range(3):
            eroded &= p[dy:dy+h, dx:dx+w]
    local = filled & ~eroded
    mask, hole = np.zeros(shape, bool), np.zeros(shape, bool)
    mask[y0:y1+1, x0:x1+1] = local
    hole[y0:y1+1, x0:x1+1] = eroded
    return mask, hole


def _path(rng, graph, dots, length, prefix=None):
    budget = 500
    def visit(path):
        nonlocal budget
        budget -= 1
        if budget < 0:
            return None
        if len(path) == length:
            return path
        for j in rng.permutation(graph[path[-1]]):
            j = int(j)
            if j in path:
                continue
            line = _segment(dots[path[-1]]['center'], dots[j]['center'], (30, 30))
            if any(np.any(line & _segment(dots[a]['center'], dots[b]['center'], (30, 30)))
                   for a, b in zip(path, path[1:])):
                continue
            answer = visit(path+[j])
            if answer is not None:
                return answer
        return None
    starts = [[int(start)] for start in rng.permutation(len(dots))] if prefix is None else [prefix]
    for start in starts:
        answer = visit(start)
        if answer is not None:
            return answer
    return None


def _candidate(rng, roles, test, witness):
    bg, wall = roles
    nr, nc = [(2, 3), (3, 2), (3, 3), (2, 4), (4, 2)][int(rng.integers(5))]
    universe = [(r, c) for r in range(nr) for c in range(nc)]
    low = 6 if test else 4
    n = int(rng.integers(low, min(9 if test else 8, len(universe))+1))
    cells = sorted(universe[int(i)] for i in rng.choice(len(universe), n, replace=False))
    count = 3 if test else 2 if witness is not None else int(rng.integers(1, 3))
    boxes = _boxes(rng, cells, count, witness)
    if boxes is None:
        return None
    ry = _axis(rng, [10*r for r, c in cells]+[v for b in boxes for v in (b[0], b[2])], 26)
    cx = _axis(rng, [10*c for r, c in cells]+[v for b in boxes for v in (b[1], b[3])], 28)
    if ry is None or cx is None:
        return None
    length = int(rng.integers(5 if test else 3, min(7 if test else 5, n-1)+1))
    h = max(ry.values())+5
    w = max(max(cx.values())+2, 2*length+1)
    h += int(rng.integers(0, min(2, 30-h)+1))
    w += int(rng.integers(0, min(3, 30-w)+1))
    grid = np.full((h, w), bg, np.int8)
    rooms = []
    taken = np.zeros(grid.shape, bool)
    for box in boxes:
        a, b, c, d = box
        mapped = (ry[a], cx[b], ry[c], cx[d])
        cut = int(rng.choice((0, 1, 1, 2)))
        mask, hole = _room_mask(grid.shape, mapped, cut)
        if np.any(mask & taken):
            return None
        taken |= mask
        grid[mask] = wall
        rooms.append(dict(mask=mask, hole=hole, lo=mapped[:2], hi=mapped[2:], wall=wall))
    dots = []
    palette = [int(c) for c in rng.permutation([c for c in range(10) if c not in roles])]
    k = int(rng.integers(2, min(5, n)+1))
    for r, c in cells:
        y, x = ry[10*r], cx[10*c]
        # Variable compact hole-free blobs. Geometry never supplies a label.
        sy = 3 if min([abs(y-v) for v in ry.values() if v != y] or [3]) >= 3 and rng.random() < .6 else 1
        sx = 3 if min([abs(x-v) for v in cx.values() if v != x] or [3]) >= 3 and rng.random() < .6 else 1
        mask = np.zeros(grid.shape, bool)
        mask[y-sy//2:y+sy//2+1, x-sx//2:x+sx//2+1] = True
        if sy == sx == 3 and rng.random() < .7:
            for yy, xx in ((y-1, x-1), (y-1, x+1), (y+1, x-1), (y+1, x+1)):
                if rng.random() < .65:
                    mask[yy, xx] = False
        color = palette[int(rng.integers(k))]
        grid[mask] = color
        dots.append(dict(mask=mask, center=(y, x), color=color,
                         lo=(y-sy//2, x-sx//2), hi=(y+sy//2, x+sx//2)))
    info = dict(grid=grid, rooms=rooms, dots=dots, background=bg, wall=wall)
    graph = _graph(info, ignore_locks=True)
    prefix = None
    if witness == 1:
        possibilities = []
        for i, neighbors in enumerate(graph):
            for j in neighbors:
                line = _segment(dots[i]['center'], dots[j]['center'], grid.shape)
                if not any(r['hole'][dots[v]['center']] for r in rooms for v in (i, j)):
                    if np.any(line & rooms[0]['hole']) and not np.any(line & rooms[1]['hole']):
                        possibilities.append([i, j])
        if not possibilities:
            return None
        prefix = possibilities[int(rng.integers(len(possibilities)))]
    path = _path(rng, graph, dots, length, prefix)
    if path is None:
        return None
    key = [dots[i]['color'] for i in path]
    if len(set(key)) < 2 or all(a == b for a, b in zip(key, key[1:])):
        return None
    info['key'] = key
    walls = taken
    required = np.zeros(grid.shape, bool)
    for i, j in zip(path, path[1:]):
        line = _segment(dots[i]['center'], dots[j]['center'], grid.shape)
        gates = line & walls
        if np.any(gates & required):
            return None
        grid[gates] = dots[i]['color']
        required |= gates
    if not required.any():
        return None
    # Gates for unused possibilities are legitimate distractors, not pixel noise.
    extra = np.zeros(grid.shape, bool)
    for i, neighbors in enumerate(graph):
        for j in neighbors:
            if rng.random() < .2:
                extra |= _segment(dots[i]['center'], dots[j]['center'], grid.shape) & walls & ~required
    for y, x in zip(*np.nonzero(extra)):
        grid[y, x] = palette[int(rng.integers(k))]
    labels = [d['color'] for d in dots]
    answers = _routes(_graph(info), labels, key)
    if answers != [tuple(path)]:
        return None
    retained = _retained(info, path)
    if witness == 0:
        if len(retained) != 2 or len(_routes(graph, labels, key)) < 2:
            return None
        if sum(np.any(r['mask'] & required) for r in rooms) < 2:
            return None
        axes = {dots[i]['center'][0] == dots[j]['center'][0] for i, j in zip(path, path[1:])}
        if len(axes) != 2:
            return None
    if witness == 1:
        if len(retained) >= count:
            return None
        if not any(not any(r['hole'][dots[i]['center']] for i in path) for r in retained):
            return None
    ky = h-2
    kx = int(rng.integers(1, w-2*length+1))
    grid[ky, kx:kx+2*length:2] = key
    try:
        out = _render(info, path)
        decoded, route = _decode(grid)
        assert np.array_equal(out, _render(decoded, route))
        assert not np.array_equal(out, _render(decoded, route, keep_all=True))
    except AssertionError:
        return None
    return dict(input=grid, output=out)


def _fallback(rng, roles, test, witness):
    """A proven-fit construction after bounded search, never a parent fixture."""
    bg, wall = roles
    labels = [int(c) for c in rng.permutation([c for c in range(10) if c not in roles])]
    if test:
        positions = [(7, 7), (7, 19), (17, 19), (23, 19), (23, 7), (17, 7)]
        colors = [labels[0], labels[1], labels[2], labels[3], labels[0], labels[2]]
        boxes = [(1, 1, 13, 13), (3, 3, 11, 11), (15, 5, 19, 9)]
        route = list(range(5))
    elif witness == 1:
        positions = [(6, 3), (6, 21), (18, 21), (18, 12), (18, 3)]
        colors = [labels[0], labels[1], labels[2], labels[3], labels[2]]
        boxes = [(2, 8, 22, 16), (14, 1, 22, 5)]
        route = [0, 1, 2]
    else:
        positions = [(7, 7), (7, 19), (17, 19), (17, 7)]
        colors = [labels[0], labels[1], labels[0], labels[2]]
        boxes = [(1, 1, 13, 13), (3, 3, 11, 11)]
        route = [0, 1, 2]
    grid = np.full((30, 30), bg, np.int8)
    wall_mask = np.zeros(grid.shape, bool)
    for box in boxes:
        mask, _ = _room_mask(grid.shape, box, 0)
        grid[mask] = wall
        wall_mask |= mask
    for p, color in zip(positions, colors):
        grid[p] = color
    for i, j in zip(route, route[1:]):
        line = _segment(positions[i], positions[j], grid.shape)
        grid[line & wall_mask] = colors[i]
    # The scene has clearance for any symmetry of its upper 25-square area.
    grid[:25, :25] = np.rot90(grid[:25, :25], int(rng.integers(4)))
    if rng.random() < .5:
        grid[:25, :25] = grid[:25, :25][:, ::-1]
    grid[27, 1:2*len(route):2] = [colors[i] for i in route]
    return dict(input=grid, output=solve(grid))


def _make_pair(rng, roles, test=False, witness=None):
    for _ in range(400):
        pair = _candidate(rng, roles, test, witness)
        if pair is not None:
            return pair
    return _fallback(rng, roles, test, witness)


def _counts(num_train, num_test):
    for name, n, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not low <= n <= high:
            raise ValueError(name+' must be a legal integer count')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette_policy = int(rng.integers(3))
    base = tuple(int(c) for c in rng.choice(10, 2, replace=False))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for i in range(count):
            roles = base if palette_policy == 0 else base[::-1] if palette_policy == 1 and split == 'test' else (
                base if palette_policy == 1 else tuple(int(c) for c in rng.choice(10, 2, replace=False)))
            witness = i if split == 'train' and i < 2 else None
            for _ in range(20):
                pair = _make_pair(rng, roles, split == 'test', witness)
                if _signature(pair['input']) not in seen:
                    break
            else:
                # Forty distinct visible color permutations exceed the at most
                # thirteen previously used inputs. This cannot exhaust a seed.
                original = pair
                for multiplier in (1, 3, 7, 9):
                    for shift in range(10):
                        remap = (multiplier*np.arange(10)+shift) % 10
                        pair = {k: remap[v].astype(np.int8) for k, v in original.items()}
                        if _signature(pair['input']) not in seen:
                            break
                    else:
                        continue
                    break
                else:
                    raise AssertionError('color orbit unexpectedly exhausted')
            seen.add(_signature(pair['input']))
            puzzle[split].append(pair)
        rng.shuffle(puzzle[split])
    return puzzle


def _features(info, route):
    rooms, dots = info['rooms'], info['dots']
    retained = _retained(info, route)
    depth = max(1+sum(other['hole'][tuple(r['lo'])] for other in rooms if other is not r) for r in rooms)
    crosses = sum(int(np.any(_segment(dots[i]['center'], dots[j]['center'], info['grid'].shape) & r['mask']))
                  for i, j in zip(route, route[1:]) for r in rooms)
    return dict(rooms=len(rooms), depth=int(depth), terminals=len(dots), length=len(route),
                retained=len(retained), removed_rooms=len(rooms)-len(retained),
                transit_rooms=sum(not any(r['hole'][dots[i]['center']] for i in route) for r in retained),
                repeated_key=len(set(info['key'])) < len(info['key']),
                unlocked_routes=len(_routes(_graph(info, True), [d['color'] for d in dots], info['key'])),
                crossings=crosses,
                axes=len({dots[i]['center'][0] == dots[j]['center'][0] for i, j in zip(route, route[1:])}))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = (_check_grid(pair[k]) for k in ('input', 'output'))
            sig = _signature(inp)
            assert sig not in seen
            seen.add(sig)
            info, route = _decode(inp)
            expected = _render(info, route)
            assert np.array_equal(out, expected), 'incorrect journey map'
            assert all(np.all(edge == info['background']) for edge in (out[0], out[-1], out[:, 0], out[:, -1]))
            assert len(set(route)) == len(info['key'])
            assert any(d['color'] != e['color'] for d, e in zip([info['dots'][i] for i in route],
                                                              [info['dots'][i] for i in route[1:]]))
            assert not np.array_equal(out, _render(info, route, target_colors=True))
            assert not np.array_equal(out, _render(info, route, keep_all=True))
            assert not np.array_equal(out, _render(info, route, omit_rooms=True))
            f = _features(info, route)
            assert f['crossings'] >= 1 and f['terminals'] > f['length']
            features[split].append(f)
    train = features['train']
    assert any(f['depth'] >= 2 and f['axes'] == 2 and f['unlocked_routes'] >= 2
               and f['crossings'] >= 2 for f in train)
    assert any(f['removed_rooms'] >= 1 and f['transit_rooms'] >= 1 for f in train)
    assert all(f['rooms'] == 3 and f['length'] >= 5 for f in features['test'])


def self_test():
    """Bounded structural, dependency, corruption, and metamorphic checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    shapes, structures, palettes, depths, lengths, room_shapes, blob_shapes = (set() for _ in range(7))
    tallies = dict(pairs=0, nonrectangular_rooms=0, transit_rooms=0, straight_routes=0,
                   repeated_keys=0, pruned_room_pairs=0, adjacent_equal_keys=0)
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                info, route = _decode(inp)
                f = _features(info, route)
                tallies['pairs'] += 1
                tallies['repeated_keys'] += f['repeated_key']
                tallies['pruned_room_pairs'] += f['removed_rooms'] > 0
                tallies['straight_routes'] += f['axes'] == 1
                tallies['adjacent_equal_keys'] += any(a == b for a, b in zip(info['key'], info['key'][1:]))
                for r in info['rooms']:
                    y0, x0 = r['lo']; y1, x1 = r['hi']
                    tallies['nonrectangular_rooms'] += not r['mask'][y0, x0]
                tallies['transit_rooms'] += sum(not any(r['hole'][info['dots'][i]['center']] for i in route)
                                                for r in _retained(info, route))
                shapes.add(inp.shape)
                mapping = {}
                structures.add((out.shape, tuple(mapping.setdefault(int(c), len(mapping)) for c in out.ravel())))
                palettes.add((info['background'], info['wall']))
                depths.add(f['depth']); lengths.add(f['length'])
                room_shapes.update((r['hi'][0]-r['lo'][0]+1, r['hi'][1]-r['lo'][1]+1) for r in info['rooms'])
                blob_shapes.update((int(d['hi'][0]-d['lo'][0]+1), int(d['hi'][1]-d['lo'][1]+1)) for d in info['dots'])
        # Color roles are geometric; translate or recolor without a dictionary.
        pair = puzzle['test'][0]
        rng = np.random.default_rng(800+len(structures))
        permutation = rng.permutation(10)
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
        inp = pair['input']
        info, route = _decode(inp)
        if max(inp.shape) < 30:
            translated = np.pad(inp, ((1, 0), (1, 0)), constant_values=info['background'])
            assert np.array_equal(solve(translated), pair['output'])
        # Stretch a safe vertical-wall row. No terminal or horizontal side is
        # duplicated, so the same enclosure and event order must survive.
        if inp.shape[0] < 30:
            excluded = {info['key_y']} | {v for r in info['rooms'] for v in (r['lo'][0], r['hi'][0])}
            for y in range(1, info['key_y']-1):
                if y in excluded or any(np.any(d['mask'][y]) for d in info['dots']):
                    continue
                if any(np.any(r['mask'][y]) for r in info['rooms']):
                    stretched = np.insert(inp, y, inp[y], axis=0)
                    assert np.array_equal(solve(stretched), pair['output'])
                    break
        # Blocking one used gate with wall color deletes an essential edge and
        # cannot add an edge: the unique original journey must then vanish.
        intro = next(q for q in puzzle['train']
                     if _features(*_decode(q['input']))['unlocked_routes'] == 2)
        ii, rr = _decode(intro['input'])
        blocked = intro['input'].copy()
        for a, b in zip(rr, rr[1:]):
            line = _segment(ii['dots'][a]['center'], ii['dots'][b]['center'], blocked.shape)
            gate = line & np.logical_or.reduce([r['mask'] for r in ii['rooms']])
            if gate.any():
                blocked[gate] = ii['wall']
                break
        bi = _parse(blocked)
        assert not _routes(_graph(bi), [d['color'] for d in bi['dots']], bi['key'])
    assert len(shapes) >= 35 and len(structures) >= 90 and len(palettes) >= 25
    assert depths >= {1, 2, 3} and lengths >= {3, 4, 5, 6, 7}
    assert len(room_shapes) >= 30 and len(blob_shapes) >= 3
    assert tallies['nonrectangular_rooms'] >= 100 and tallies['pruned_room_pairs'] >= 42
    assert tallies['straight_routes'] >= 1 and tallies['adjacent_equal_keys'] >= 1
    repeated = generate_puzzle(0)
    assert all(_signature(puzzles[0][s][i]['input']) == _signature(repeated[s][i]['input'])
               for s in ('train', 'test') for i in range(len(puzzles[0][s])))
    for nt, nq in ((2, 1), (10, 1), (10, 4), (7, 3)):
        validate_generated_puzzle(generate_puzzle(12000+nt*10+nq, nt, nq))
    for test, witness in ((False, 0), (False, 1), (False, None), (True, None)):
        pair = _fallback(np.random.default_rng(904), (0, 1), test, witness)
        info, route = _decode(pair['input'])
        assert np.array_equal(pair['output'], _render(info, route))
        if witness == 1:
            f = _features(info, route)
            assert f['transit_rooms'] == 1 and f['removed_rooms'] == 1
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = copy.deepcopy(puzzles[0])
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corruption accepted')
    return dict(seeds=42, input_shapes=len(shapes), normalized_outputs=len(structures),
                ordered_palettes=len(palettes), depths=sorted(depths), route_lengths=sorted(lengths),
                room_shapes=len(room_shapes), blob_shapes=sorted(blob_shapes),
                **{k: int(v) for k, v in tallies.items()})


if __name__ == '__main__':
    self_test()
    print('self_test passed: 42 seeds, count boundaries, dependency and color checks')
