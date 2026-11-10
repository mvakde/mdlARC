"""Fuel-gated jigsaw docking cascades. Only NumPy and the standard library."""
from collections import deque
import json
import numpy as np

SPENT, FUEL = 8, 9
DELTAS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
TURNS = {'U': 0, 'D': 2, 'L': 3, 'R': 1}


def _need(ok, message):
    if not ok:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), 'invalid dimensions')
    _need(np.issubdtype(g.dtype, np.integer), 'noninteger cells')
    _need(bool(np.all((g >= 0) & (g <= 9))), 'invalid colors')
    return g.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        todo, cells = [first], []
        for y, x in todo:
            cells.append((y, x))
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if p in unseen:
                    unseen.remove(p)
                    todo.append(p)
        result.append(cells)
    return result


def _marker(point, corner):
    y, x = point
    dy = -1 if corner[0] == 'T' else 1
    dx = -1 if corner[1] == 'L' else 1
    return [(y+dy, x), (y, x+dx), (y+dy, x+dx)]


def _extract(g):
    border = np.concatenate((g[0], g[-1], g[:, 0], g[:, -1]))
    _need(len(set(map(int, border))) == 1, 'input needs a blank outer border')
    bg = int(border[0])
    _need(bg not in (SPENT, FUEL), 'background conflicts with counter states')
    explanations = []
    for color in sorted(set(map(int, np.unique(g))) - {bg, SPENT, FUEL}):
        cells = np.argwhere(g == color)
        if len(cells) != 3 or tuple(np.ptp(cells, axis=0)) != (1, 1):
            continue
        y, x = map(int, cells.min(axis=0))
        missing = {(y, x), (y+1, x), (y, x+1), (y+1, x+1)} - set(map(tuple, cells.tolist()))
        point = missing.pop()
        if int(g[point]) in (bg, color, SPENT, FUEL):
            continue
        corner = {(1, 1): 'TL', (1, 0): 'TR', (0, 1): 'BL', (0, 0): 'BR'}[(point[0]-y, point[1]-x)]
        pieces, anchor = [], None
        good = True
        for component in _components((g != bg) & (g != color)):
            q = np.array(component)
            t, l = map(int, q.min(axis=0))
            b, r = map(int, q.max(axis=0) + 1)
            m = g[t:b, l:r].copy()
            if min(m.shape) < 3 or len(component) != m.size:
                good = False
                break
            edge = np.concatenate((m[0], m[-1], m[:, 0], m[:, -1]))
            if np.any(np.isin(edge, (bg, color, SPENT, FUEL))) or not np.all(np.isin(m[1:-1, 1:-1], (SPENT, FUEL))):
                good = False
                break
            if point in component:
                anchor = len(pieces)
            pieces.append((m, (t, l)))
        if not good or not 3 <= len(pieces) <= 7 or anchor is None:
            continue
        m, (t, l) = pieces[anchor]
        expected = (t + (m.shape[0]-1 if corner[0] == 'B' else 0),
                    l + (m.shape[1]-1 if corner[1] == 'R' else 0))
        if point == expected:
            explanations.append((bg, color, point, corner, pieces, anchor))
    _need(len(explanations) == 1, 'missing or ambiguous framed-piece explanation')
    return explanations[0]


def _travel(source, target):
    dy, dx = target[0]-source[0], target[1]-source[1]
    if dy and dx:
        return None
    if not dy and not dx:
        return ('U', 0)
    direction = ('D' if dy > 0 else 'U') if dy else ('R' if dx > 0 else 'L')
    return direction, abs(dy)+abs(dx)


def _counter(m, direction):
    interior = np.rot90(m, TURNS[direction])[1:-1, 1:-1]
    values = interior.ravel()
    advanced = int(np.count_nonzero(values == SPENT))
    _need(np.array_equal(values, np.array([SPENT]*advanced + [FUEL]*(values.size-advanced))),
          'counter is not a travel-relative prefix')
    return advanced, int(values.size)-advanced


def _contact(a, pa, b, pb, check_colors=True):
    y, x = pa
    v, u = pb
    h, w = a.shape
    k, z = b.shape
    if x+w == u or u+z == x:
        start, stop = max(y, v), min(y+h, v+k)
        if start < stop:
            aa = a[start-y:stop-y, -1 if x+w == u else 0]
            bb = b[start-v:stop-v, 0 if x+w == u else -1]
            return True, not check_colors or np.array_equal(aa, bb)
    if y+h == v or v+k == y:
        start, stop = max(x, u), min(x+w, u+z)
        if start < stop:
            aa = a[-1 if y+h == v else 0, start-x:stop-x]
            bb = b[0 if y+h == v else -1, start-u:stop-u]
            return True, not check_colors or np.array_equal(aa, bb)
    return False, True


def _assembly(g, parsed=None, match_edges=True):
    """Complete first-uncovered-cell rectangle packing; no generation metadata."""
    parsed = _extract(g) if parsed is None else parsed
    bg, marker, point, corner, pieces, anchor = parsed
    area = sum(m.size for m, _ in pieces)
    answers = []
    for height in range(3, g.shape[0]-1):
        if area % height:
            continue
        width = area // height
        oy = point[0] - (height-1 if corner[0] == 'B' else 0)
        ox = point[1] - (width-1 if corner[1] == 'R' else 0)
        if width < 3 or ox < 1 or oy < 1 or ox+width >= g.shape[1] or oy+height >= g.shape[0]:
            continue
        if any(m.shape[0] > height or m.shape[1] > width for m, _ in pieces):
            continue
        positions = [None]*len(pieces)
        ay, ax = pieces[anchor][1][0]-oy, pieces[anchor][1][1]-ox
        ah, aw = pieces[anchor][0].shape
        if ay < 0 or ax < 0 or ay+ah > height or ax+aw > width:
            continue
        positions[anchor] = (ay, ax)
        occupied = sum(((1 << aw)-1) << ((ay+r)*width+ax) for r in range(ah))
        full = (1 << area)-1

        def visit(occ, left):
            if len(answers) > 1:
                return
            if not left:
                absolute = tuple((p[0]+oy, p[1]+ox) for p in positions)
                if absolute not in answers:
                    answers.append(absolute)
                return
            free = full ^ occ
            cell = (free & -free).bit_length()-1
            row, col = divmod(cell, width)
            for i in left:
                m, source = pieces[i]
                h, w = m.shape
                if row+h > height or col+w > width:
                    continue
                travel = _travel(source, (row+oy, col+ox))
                if travel is None:
                    continue
                if travel[1]:
                    try:
                        _counter(m, travel[0])
                    except AssertionError:
                        continue
                bit = sum(((1 << w)-1) << ((row+r)*width+col) for r in range(h))
                if bit & occ:
                    continue
                if match_edges and any(not _contact(m, (row, col), pieces[j][0], p)[1]
                       for j, p in enumerate(positions) if p is not None):
                    continue
                positions[i] = (row, col)
                visit(occ | bit, tuple(j for j in left if j != i))
                positions[i] = None
        visit(occupied, tuple(i for i in range(len(pieces)) if i != anchor))
    _need(len(answers) == 1, 'intended assembly is absent or ambiguous')
    targets = answers[0]
    neighbors = [set() for _ in pieces]
    for i in range(len(pieces)):
        for j in range(i):
            if _contact(pieces[i][0], targets[i], pieces[j][0], targets[j])[0]:
                neighbors[i].add(j)
                neighbors[j].add(i)
    return parsed, targets, neighbors


def _consume(m, direction, amount):
    canonical = np.rot90(m, TURNS[direction]).copy()
    interior = canonical[1:-1, 1:-1].copy()
    pending = np.flatnonzero(interior.ravel() == FUEL)
    interior.flat[pending[:amount]] = SPENT
    canonical[1:-1, 1:-1] = interior
    return np.rot90(canonical, -TURNS[direction]).copy()


def _coordinate_ticks(m, direction, amount):
    """Validation uses direct coordinates rather than rotating/flattening a block."""
    h, w = m.shape
    rows, cols = list(range(1, h-1)), list(range(1, w-1))
    if direction == 'U':
        cells = [(y, x) for y in rows for x in cols]
    elif direction == 'D':
        cells = [(y, x) for y in rows[::-1] for x in cols[::-1]]
    elif direction == 'L':
        cells = [(y, x) for x in cols for y in rows[::-1]]
    else:
        cells = [(y, x) for x in cols[::-1] for y in rows]
    result = m.copy()
    for p in cells:
        if amount and result[p] == FUEL:
            result[p] = SPENT
            amount -= 1
    _need(amount == 0, 'counter overspent')
    return result


def _run(g, explanation=None, ungated=False):
    parsed, targets, neighbors = _assembly(g) if explanation is None else explanation
    bg, marker, point, corner, pieces, anchor = parsed
    docked, attempted = {anchor}, {anchor}
    matrices = [m.copy() for m, _ in pieces]
    positions = [p for _, p in pieces]
    records = []
    wave = 0
    while True:
        active = [i for i in range(len(pieces)) if i not in attempted and
                  (ungated or bool(neighbors[i] & docked))]
        if not active:
            break
        wave += 1
        newly = set()
        for i in active:
            m, source = pieces[i]
            direction, distance = _travel(source, targets[i])
            advanced, fuel = _counter(m, direction) if distance else (int(np.sum(m[1:-1, 1:-1] == SPENT)), int(np.sum(m[1:-1, 1:-1] == FUEL)))
            movement = min(distance, fuel)
            dy, dx = DELTAS[direction]
            positions[i] = (source[0]+movement*dy, source[1]+movement*dx)
            matrices[i] = _consume(m, direction, movement)
            attempted.add(i)
            if movement == distance:
                newly.add(i)
            records.append(dict(piece=i, wave=wave, direction=direction, distance=distance,
                                fuel=fuel, advanced=advanced, movement=movement,
                                docked=movement == distance))
        docked |= newly
    out = np.full_like(g, bg)
    occupied = np.zeros_like(g, dtype=bool)
    for m, (y, x) in zip(matrices, positions):
        h, w = m.shape
        _need(0 <= y and 0 <= x and y+h <= g.shape[0] and x+w <= g.shape[1], 'motion left canvas')
        _need(not occupied[y:y+h, x:x+w].any(), 'pieces collided')
        occupied[y:y+h, x:x+w] = True
        out[y:y+h, x:x+w] = m
    idle = set(range(len(pieces))) - attempted
    return out, dict(parsed=parsed, targets=targets, neighbors=neighbors,
                     records=records, idle=idle, docked=docked, positions=positions)


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _run(_grid(input_grid))[0]


def _waiting_can_move(info):
    """Ignoring activation would spend a positive number of extra fuel cells."""
    pieces = info['parsed'][4]
    for i in info['idle']:
        m, source = pieces[i]
        direction, distance = _travel(source, info['targets'][i])
        if distance and _counter(m, direction)[1] > 0:
            return True
    return False


def _adjacency(rects):
    n = len(rects)
    neighbors = [set() for _ in rects]
    for i in range(n):
        for j in range(i):
            a, b = rects[i], rects[j]
            if _contact(np.zeros(a[2:]), a[:2], np.zeros(b[2:]), b[:2], False)[0]:
                neighbors[i].add(j)
                neighbors[j].add(i)
    return neighbors


def _layout(rng, count, fallback=False):
    if fallback or count == 3 or rng.random() < .10:
        h = int(rng.integers(6, 9))
        minimum = 4 if count <= 4 else 3
        widths = [minimum]*count
        for _ in range(int(rng.integers(0, 27-minimum*count))):
            widths[int(rng.integers(count))] += 1
        x, rects = 0, []
        for w in widths:
            rects.append((0, x, h, w))
            x += w
        return (h, x), rects
    h, w = int(rng.integers(9, 18)), int(rng.integers(9, 19))
    rects = [(0, 0, h, w)]
    minimum = 4 if count <= 4 else 3
    for _ in range(count-1):
        choices = [(i, axis) for i, (_, _, ph, pw) in enumerate(rects)
                   for axis, length in enumerate((ph, pw)) if length >= 2*minimum]
        if not choices:
            return None
        i, axis = choices[int(rng.integers(len(choices)))]
        y, x, ph, pw = rects.pop(i)
        cut = int(rng.integers(minimum, (ph if axis == 0 else pw)-minimum+1))
        if axis == 0:
            rects += [(y, x, cut, pw), (y+cut, x, ph-cut, pw)]
        else:
            rects += [(y, x, ph, cut), (y, x+cut, ph, pw-cut)]
    if any(y > 0 and x > 0 and y+ph < h and x+pw < w for y, x, ph, pw in rects):
        return None  # Every movable tile needs an outward unobstructed lane.
    return (h, w), rects


def _separated(a, b):
    y, x, h, w = a
    v, u, k, z = b
    return y+h < v or v+k < y or x+w < u or u+z < x


def _overlap(a, b):
    y, x, h, w = a
    v, u, k, z = b
    return y < v+k and v < y+h and x < u+z and u < x+w


def _sweep(rect, source):
    y, x, h, w = rect
    v, u = source
    return min(y, v), min(x, u), h+abs(y-v), w+abs(x-u)


def _paint(rng, rects, palette):
    """Repeated/quiet frame patterns, with seam equalities applied simultaneously."""
    matrices, coordinates, parent = [], [], []
    for y, x, h, w in rects:
        m = np.full((h, w), FUEL, dtype=np.int8)
        for r in range(h):
            for c in range(w):
                if r in (0, h-1) or c in (0, w-1):
                    m[r, c] = palette[0] if r in (0, h-1) and c in (0, w-1) else int(rng.choice(palette))
                    coordinates.append((len(matrices), r, c, y+r, x+c))
        matrices.append(m)
    parent = list(range(len(coordinates)))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    lookup = {(y, x): k for k, (_, _, _, y, x) in enumerate(coordinates)}
    for k, (i, r, c, y, x) in enumerate(coordinates):
        for p in ((y+1, x), (y, x+1)):
            other = lookup.get(p)
            if other is not None and coordinates[other][0] != i:
                parent[root(other)] = root(k)
    groups = {}
    for k, coord in enumerate(coordinates):
        groups.setdefault(root(k), []).append(coord)
    for group in groups.values():
        corner = any(r in (0, matrices[i].shape[0]-1) and c in (0, matrices[i].shape[1]-1)
                     for i, r, c, _, _ in group)
        color = palette[0] if corner else int(rng.choice(palette))
        for i, r, c, _, _ in group:
            matrices[i][r, c] = color
    return matrices


def _make_pair(rng, count, lesson, bg, marker, palette, fallback=False):
    if lesson == 'complete':
        return _stack_pair(rng, count, lesson, bg, marker, palette)
    if not fallback and lesson == 'mixed' and rng.random() < .20:
        return _stack_pair(rng, count, lesson, bg, marker, palette)
    layout = _layout(rng, count, fallback)
    if layout is None:
        return None
    (height, width), local = layout
    anchor = next(i for i, (y, x, _, _) in enumerate(local) if y == x == 0)
    neighbors = _adjacency(local)
    depths = {anchor: 0}
    queue = deque([anchor])
    while queue:
        i = queue.popleft()
        for j in neighbors[i] - depths.keys():
            depths[j] = depths[i]+1
            queue.append(j)
    if max(depths.values()) < 2:
        return None
    oy = height+3 if fallback else (int(rng.integers(8, max(9, 22-height))) if height <= 13 else (30-height)//2)
    ox = int(rng.integers(2, 30-width))
    rects = [(y+oy, x+ox, h, w) for y, x, h, w in local]
    sources = [None]*count
    sources[anchor] = rects[anchor][:2]
    sweeps = {anchor: rects[anchor]}
    placed = {anchor: rects[anchor]}
    order = [i for i in range(count) if i != anchor]
    if not fallback:
        rng.shuffle(order)
    for i in order:
        y, x, h, w = local[i]
        outward = []
        if y == 0:
            outward.append('U')
        if y+h == height:
            outward.append('D')
        if x == 0:
            outward.append('L')
        if x+w == width:
            outward.append('R')
        options = []
        for d in outward:
            dy, dx = DELTAS[d]
            for distance in range(1, 15):
                if fallback and (d != ('U' if i % 2 else 'D') or distance != height+1):
                    continue
                if lesson == 'complete' and distance > (h-2)*(w-2):
                    continue
                if lesson == 'blocked' and i in neighbors[anchor] and distance < 2:
                    continue
                sy, sx = rects[i][0]+distance*dy, rects[i][1]+distance*dx
                source = (sy, sx, h, w)
                if sy < 1 or sx < 1 or sy+h >= 30 or sx+w >= 30:
                    continue
                swept = _sweep(rects[i], (sy, sx))
                if any(_overlap(swept, rects[j]) for j in range(count) if j != i):
                    continue
                if any(_overlap(swept, s) for s in sweeps.values()):
                    continue
                if not all(_separated(source, s) for s in placed.values()):
                    continue
                if any(sy-1 <= my <= sy+h and sx-1 <= mx <= sx+w for my, mx in _marker(rects[anchor][:2], 'TL')):
                    continue
                options.append(((sy, sx), swept, source))
        if not options:
            return None
        choice = options[int(rng.integers(len(options)))]
        sources[i], sweeps[i], placed[i] = choice
    matrices = _paint(rng, local, palette)
    for i, m in enumerate(matrices):
        capacity = (m.shape[0]-2)*(m.shape[1]-2)
        if i == anchor:
            advanced = int(rng.integers(capacity+1))
            m[1:-1, 1:-1] = np.array([SPENT]*advanced+[FUEL]*(capacity-advanced)).reshape(m.shape[0]-2, m.shape[1]-2)
            continue
        direction, distance = _travel(sources[i], rects[i][:2])
        if lesson == 'complete':
            fuel = int(rng.integers(distance, capacity+1))
        elif lesson == 'blocked' and i in neighbors[anchor]:
            fuel = int(rng.integers(1, min(distance-1, capacity)+1))
        elif lesson == 'blocked':
            fuel = int(rng.integers(min(distance, capacity), capacity+1))
        else:
            fuel = int(rng.integers(capacity+1))
        canonical = np.rot90(m, TURNS[direction]).copy()
        canonical[1:-1, 1:-1] = np.array([SPENT]*(capacity-fuel)+[FUEL]*fuel).reshape(canonical.shape[0]-2, canonical.shape[1]-2)
        matrices[i] = np.rot90(canonical, -TURNS[direction]).copy()
    g = np.full((30, 30), bg, dtype=np.int8)
    for m, (y, x) in zip(matrices, sources):
        g[y:y+m.shape[0], x:x+m.shape[1]] = m
    for p in _marker(rects[anchor][:2], 'TL'):
        g[p] = marker
    # Retain the intended target and all swept corridors while cropping blank space.
    all_rects = list(sweeps.values())
    t = max(0, min(r[0] for r in all_rects)-int(rng.integers(1, 4)))
    l = max(0, min(r[1] for r in all_rects)-int(rng.integers(1, 4)))
    b = min(30, max(r[0]+r[2] for r in all_rects)+int(rng.integers(1, 4)))
    r = min(30, max(r[1]+r[3] for r in all_rects)+int(rng.integers(1, 4)))
    # The marker needs its own blank exterior cell, too.
    t, l = min(t, rects[anchor][0]-2), min(l, rects[anchor][1]-2)
    if t < 0 or l < 0:
        return None
    g = np.rot90(g[t:b, l:r], int(rng.integers(4))).copy()
    try:
        out, info = _run(g)
        if lesson == 'complete' and len(info['docked']) != count:
            return None
        if lesson == 'blocked':
            if not info['idle'] or not any(r['movement'] > 0 and not r['docked'] for r in info['records']):
                return None
            if not _waiting_can_move(info):
                return None
        return dict(input=g, output=out)
    except AssertionError:
        return None


def _stack_pair(rng, count, lesson, bg, marker, palette):
    """Shuffle pieces along a common line: edge patterns must recover their order.

    Translations are endpoint operations, not an obstacle-avoidance simulation.
    This sampler may have crossing travel intervals; final masks must be disjoint.
    """
    minimum = 4 if lesson == 'complete' else 3
    heights = [minimum]*count
    for _ in range(0 if lesson == 'complete' else int(rng.integers(0, max(1, 25-count-count*minimum)))):
        heights[int(rng.integers(count))] += 1
    width = int(rng.integers(9, 12)) if lesson == 'complete' else int(rng.integers(7, 11))
    y, rects = 0, []
    for h in heights:
        rects.append((y, 0, h, width))
        y += h
    matrices = _paint(rng, rects, palette)
    sources = [(2, 3)] + [None]*(count-1)
    gap = 2 if lesson == 'complete' else 1
    position = 2+heights[0]+gap
    order = list(map(int, rng.permutation(np.arange(2, count)))) + [1] if lesson == 'complete' else list(map(int, rng.permutation(np.arange(1, count))))
    for i in order:
        i = int(i)
        sources[i] = (position, 3)
        position += heights[i]+gap
    if position+1 > 30:
        return None
    targets = [(r+2, 3) for r, _, _, _ in rects]
    for i, m in enumerate(matrices):
        capacity = (m.shape[0]-2)*(width-2)
        direction, distance = _travel(sources[i], targets[i])
        if i == 0:
            fuel = int(rng.integers(capacity+1))
        elif lesson == 'complete':
            if distance > capacity:
                return None
            fuel = capacity
        elif lesson == 'blocked' and i == 1:
            if distance < 2:
                return None
            fuel = 1  # A visible partial attempt, inside the surrounding blank gap.
        elif lesson == 'blocked':
            fuel = capacity
        else:
            fuel = int(rng.integers(capacity+1))
        canonical = np.rot90(m, TURNS[direction]).copy()
        canonical[1:-1, 1:-1] = np.array([SPENT]*(capacity-fuel)+[FUEL]*fuel).reshape(canonical.shape[0]-2, canonical.shape[1]-2)
        matrices[i] = np.rot90(canonical, -TURNS[direction]).copy()
    canvas = (position+1, width+int(rng.integers(6, 11)))
    g = np.full(canvas, bg, dtype=np.int8)
    for m, (r, c) in zip(matrices, sources):
        g[r:r+m.shape[0], c:c+width] = m
    for p in _marker((2, 3), 'TL'):
        g[p] = marker
    g = np.rot90(g, int(rng.integers(4))).copy()
    try:
        out, info = _run(g)
        if lesson == 'complete' and len(info['docked']) != count:
            return None
        if lesson == 'complete':
            try:
                _assembly(g, match_edges=False)
            except AssertionError:
                pass
            else:
                return None  # Training must make seam evidence necessary.
        if lesson == 'blocked' and (not info['idle'] or not _waiting_can_move(info)):
            return None
        return dict(input=g, output=out)
    except AssertionError:
        return None


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, low, high, name in ((num_train, 2, 10, 'num_train'), (num_test, 1, 4, 'num_test')):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shared = bool(rng.integers(2))
    base_roles = tuple(map(int, rng.choice(8, 2, replace=False)))
    plans = ['complete', 'blocked'] + ['mixed']*(num_train-2)
    rng.shuffle(plans)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            lesson = plans[index] if split == 'train' else 'mixed'
            if split == 'train':
                pieces = 4 if lesson == 'blocked' else int(rng.choice((3, 4), p=(.25, .75)))
            else:
                pieces = int(rng.choice((5, 6, 7), p=(.50, .35, .15)))
            bg, marker = base_roles if shared else tuple(map(int, rng.choice(8, 2, replace=False)))
            colors = [c for c in range(8) if c not in (bg, marker)]
            palette = list(map(int, rng.choice(colors, int(rng.integers(2, 5)), replace=False)))
            for attempt in range(160):
                pair = _make_pair(rng, pieces, lesson, bg, marker, palette, fallback=attempt >= 60)
                if pair is None:
                    continue
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded construction failed')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'split is not a list')
    _need(2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4 and
          len(puzzle['train']) > len(puzzle['test']), 'illegal counts')
    seen, train_info, counts = set(), [], {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair schema')
            g, out = _grid(pair['input']), _grid(pair['output'])
            _need(g.shape == out.shape, 'canvas changed')
            key = (g.shape, g.tobytes())
            _need(key not in seen, 'duplicate input')
            seen.add(key)
            predicted, info = _run(g)
            _need(np.array_equal(predicted, out), 'wrong docking cascade')
            bg, marker, point, corner, pieces, anchor = info['parsed']
            _need(not np.any(out == marker), 'marker not erased')
            _need(info['positions'][anchor] == pieces[anchor][1], 'anchor moved')
            _need(np.array_equal(out[pieces[anchor][1][0]:pieces[anchor][1][0]+pieces[anchor][0].shape[0],
                                      pieces[anchor][1][1]:pieces[anchor][1][1]+pieces[anchor][0].shape[1]], pieces[anchor][0]), 'anchor changed')
            used = sum(r['movement'] for r in info['records'])
            cin, cout = np.bincount(g.ravel(), minlength=10), np.bincount(out.ravel(), minlength=10)
            _need(cout[FUEL] == cin[FUEL]-used and cout[SPENT] == cin[SPENT]+used, 'one fuel tick per translated cell')
            _need(cout[bg] == cin[bg]+3, 'payload area not conserved')
            _need(all(cin[c] == cout[c] for c in range(10) if c not in (bg, marker, SPENT, FUEL)), 'frame colors changed')
            records = {r['piece']: r for r in info['records']}
            for i, (m, source) in enumerate(pieces):
                record = records.get(i)
                movement = record['movement'] if record else 0
                direction = record['direction'] if record else 'U'
                expected_tile = _coordinate_ticks(m, direction, movement)
                y, x = info['positions'][i]
                _need(np.array_equal(out[y:y+m.shape[0], x:x+m.shape[1]], expected_tile), 'wrong directional tick order')
                if i in info['idle']:
                    y, x = source
                    _need(np.array_equal(out[y:y+m.shape[0], x:x+m.shape[1]], m), 'waiting tile changed')
                    _need(not (info['neighbors'][i] & info['docked']), 'reachable tile left waiting')
            reached = {anchor}
            for wave in range(1, max((r['wave'] for r in info['records']), default=0)+1):
                wave_records = [r for r in info['records'] if r['wave'] == wave]
                _need(all(info['neighbors'][r['piece']] & reached for r in wave_records), 'tile activated without a docked neighbor')
                reached.update(r['piece'] for r in wave_records if r['docked'])
            counts[split].append(len(pieces))
            if split == 'train':
                train_info.append((g, out, info))
    _need(min(counts['test']) > max(counts['train']), 'count extrapolation missing')
    _need(any(len(info['docked']) == len(info['parsed'][4]) and max(r['wave'] for r in info['records']) >= 2
              for _, _, info in train_info), 'no successful multihop demonstration')
    _need(any(r['movement'] > 0 and r['fuel'] > r['distance'] and
              min(info['parsed'][4][r['piece']][0].shape) >= 4
              for _, _, info in train_info for r in info['records']), 'no informative two-dimensional counter')
    pattern_witness = False
    for g, _, _ in train_info:
        try:
            _assembly(g, match_edges=False)
        except AssertionError:
            pattern_witness = True
    _need(pattern_witness, 'training never needs frame patterns')
    _need(any(info['idle'] and any(r['movement'] > 0 and not r['docked'] for r in info['records']) and
              _waiting_can_move(info) for g, out, info in train_info), 'no counter/gating witness')


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    features, shapes, geometries, palettes, inputs = [], set(), set(), set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                g = pair['input']
                out, info = _run(g)
                _need(np.array_equal(solve(g.tolist()), out), 'list input differs')
                bg, marker, _, corner, pieces, _ = info['parsed']
                shapes.add(g.shape)
                palettes.add((bg, marker, tuple(map(int, np.unique(g)))))
                geometries.add((tuple(m.shape for m, _ in pieces), tuple(info['targets']), corner))
                inputs.add((g.shape, g.tobytes()))
                features.append((split, corner, len(pieces), info))
    _need(len(inputs) == 210 and len(shapes) >= 30 and len(geometries) >= 100 and len(palettes) >= 30, 'variation collapsed')
    _need({f[1] for f in features} == {'TL', 'TR', 'BL', 'BR'}, 'missing corners')
    _need({f[2] for f in features} == set(range(3, 8)), 'missing piece counts')
    records = [r for _, _, _, info in features for r in info['records']]
    _need({r['direction'] for r in records} == set(DELTAS), 'missing directions')
    _need(any(r['fuel'] == 0 for r in records) and any(r['advanced'] == 0 for r in records), 'missing uniform counters')
    _need(any(r['fuel'] == r['distance'] for r in records) and any(r['fuel'] > r['distance'] for r in records), 'missing exact/surplus fuel')
    _need(any(len(info['idle']) >= 2 for _, _, _, info in features), 'missing blocked branches')
    _need(any(max((r['wave'] for r in info['records']), default=0) >= 3 for _, _, _, info in features), 'missing deeper cascade')
    branching = sum(max(map(len, info['neighbors'])) >= 3 for _, _, _, info in features)
    _need(branching >= 20, 'branching variation collapsed')
    bypasses = 0
    for _, _, _, info in features:
        by_piece = {r['piece']: r for r in info['records']}
        for r in info['records']:
            if any(j in by_piece and not by_piece[j]['docked'] and by_piece[j]['wave'] < r['wave']
                   for j in info['neighbors'][r['piece']]):
                bypasses += 1
    _need(bypasses > 0, 'no alternate route around a failed neighbor')
    # Reordering an inventory never chooses a packing or an activation order.
    pair = puzzles[0]['train'][-1]
    parsed = _extract(pair['input'])
    bg, marker, point, corner, pieces, anchor = parsed
    reversed_parsed = (bg, marker, point, corner, list(reversed(pieces)), len(pieces)-1-anchor)
    reversed_explanation = _assembly(pair['input'], reversed_parsed)
    _need(np.array_equal(_run(pair['input'], reversed_explanation)[0], pair['output']), 'piece indexing changes answer')
    # Removing the informative seams creates a tie, rather than an arbitrary winner.
    teaching = next(g for g, _, info in [(q['input'], q['output'], _run(q['input'])[1]) for q in puzzles[0]['train']]
                    if all(r['advanced'] == 0 for r in info['records']) and len(info['docked']) == len(info['parsed'][4]))
    ambiguous = teaching.copy()
    bg, marker, _, _, pieces, _ = _extract(ambiguous)
    structural = sorted(set(map(int, np.unique(ambiguous))) - {bg, marker, SPENT, FUEL})
    ambiguous[np.isin(ambiguous, structural)] = structural[0]
    try:
        solve(ambiguous)
    except AssertionError:
        pass
    else:
        raise AssertionError('ambiguous seams accepted')
    a, b = generate_puzzle(123, 10, 4), generate_puzzle(123, 10, 4)
    _need(all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x), 'nondeterministic')
    for tr in range(2, 11):
        for te in range(1, min(4, tr-1)+1):
            generate_puzzle(42+tr+te, tr, te)
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupt output accepted')
    return {'seeds': 42, 'pairs': len(inputs), 'shapes': len(shapes),
            'geometries': len(geometries), 'palettes': len(palettes),
            'branching_scenes': branching, 'activations_beside_failed_neighbors': bypasses}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
