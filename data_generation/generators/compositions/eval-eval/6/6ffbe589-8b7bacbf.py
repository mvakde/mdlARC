"""Rotary paint manifold: independent masks gate stationary paint circuits.

Only Python's standard library and NumPy are used. Generation constructs labels
from intended post-turn masks and contour ownership; solve reconstructs both
from the input, using coordinate arithmetic rather than generation metadata.
"""
from collections import deque
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _integer(x):
    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))


def _counts(a, b):
    if not _integer(a) or not _integer(b):
        raise ValueError("example counts must be integers")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    return int(a), int(b)


def _grid(a):
    a = np.asarray(a)
    if (a.ndim != 2 or not all(1 <= x <= 30 for x in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.asarray(a, dtype=np.int8).tobytes()


def _components(cells, steps=D8):
    left = set(cells)
    groups = []
    while left:
        root = min(left)
        left.remove(root)
        group = {root}
        queue = [root]
        for r, c in queue:
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in left:
                    left.remove(q)
                    group.add(q)
                    queue.append(q)
        groups.append(group)
    return groups


def _near(cells, steps=D8):
    cells = set(cells)
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps} - cells


def _turn(p, n, k):
    r, c = p
    for _ in range(k % 4):
        r, c = c, n - 1 - r
    return r, c


def _port(p, r, c, n):
    """Map a noncorner rotor-edge cell to its rim socket and outward step."""
    y, x = p
    if y == 0 and 0 < x < n - 1:
        return (r, c + 1 + x), (-1, 0)
    if x == n - 1 and 0 < y < n - 1:
        return (r + 1 + y, c + n + 1), (0, 1)
    if y == n - 1 and 0 < x < n - 1:
        return (r + n + 1, c + 1 + x), (1, 0)
    if x == 0 and 0 < y < n - 1:
        return (r + 1 + y, c), (0, -1)
    raise ValueError("sockets face noncorner rotor-edge cells")


def _locate(a):
    values, counts = np.unique(a, return_counts=True)
    if np.sum(counts == counts.max()) != 1:
        raise ValueError("background must be the unique most frequent color")
    bg = int(values[counts.argmax()])
    h, w = a.shape
    candidates = []
    for n in range(5, 12):
        d = n + 2
        if min(h, w) < d:
            continue
        for r, c in np.argwhere(a[:h-d+1, :w-d+1] != bg):
            r, c = int(r), int(c)
            f = int(a[r, c])
            if not (a[r, c+d-1] == a[r+d-1, c] == a[r+d-1, c+d-1] == f):
                continue
            border = np.concatenate((a[r, c:c+d], a[r+d-1, c:c+d],
                                     a[r+1:r+d-1, c], a[r+1:r+d-1, c+d-1]))
            others = set(map(int, np.unique(border))) - {f}
            if len(others) != 1 or bg in others:
                continue
            wire = others.pop()
            source = a[r+1:r+n+1, c+1:c+n+1]
            palette = set(map(int, np.unique(source))) - {bg}
            if not 2 <= len(palette) <= 4 or palette & {f, wire}:
                continue
            sockets = []
            for y, x in np.argwhere(a[r:r+d, c:c+d] == wire):
                y, x = int(y), int(x)
                if y not in (0, d-1) and x not in (0, d-1):
                    break
                if y == 0:
                    inward = (0, x-1)
                elif y == d-1:
                    inward = (n-1, x-1)
                elif x == 0:
                    inward = (y-1, 0)
                else:
                    inward = (y-1, n-1)
                try:
                    absolute, delta = _port(inward, r, c, n)
                except ValueError:
                    break
                outward = (absolute[0] + delta[0], absolute[1] + delta[1])
                if not (0 <= outward[0] < h and 0 <= outward[1] < w
                        and a[outward] == wire):
                    break
                sockets.append((absolute, inward))
            else:
                if sockets:
                    candidates.append(dict(bg=bg, frame=f, wire=wire, r=r, c=c,
                                           n=n, source=source.copy(),
                                           paints=tuple(sorted(palette)), sockets=sockets))
    if len(candidates) != 1:
        raise ValueError("expected one geometrically identifiable rotary hub")
    return candidates[0]


def _analyze(input_grid):
    a = _grid(input_grid)
    info = _locate(a)
    bg, r, c, n = info['bg'], info['r'], info['c'], info['n']
    rotor = {(y, x) for y in range(r+1, r+n+1) for x in range(c+1, c+n+1)}
    housing = {(y, x) for y in range(r, r+n+2) for x in range(c, c+n+2)}
    bars, lengths, orientations = set(), {}, {}
    outside = a.copy()
    outside[r+1:r+n+1, c+1:c+n+1] = bg
    for paint in info['paints']:
        mask = info['source'] == paint
        perimeter = np.zeros((n, n), dtype=bool)
        perimeter[0] = perimeter[-1] = True
        perimeter[:, 0] = perimeter[:, -1] = True
        if int(np.sum(mask & perimeter)) != 1:
            raise ValueError('each paint mask must have one rotor-edge nozzle')
        cells = set(map(tuple, np.argwhere(outside == paint).tolist()))
        if not cells:
            lengths[paint] = 0
            continue
        rows = {y for y, x in cells}
        cols = {x for y, x in cells}
        horizontal = len(rows) == 1 and max(cols)-min(cols)+1 == len(cells)
        vertical = len(cols) == 1 and max(rows)-min(rows)+1 == len(cells)
        if not (horizontal or vertical) or not 1 <= len(cells) <= 9:
            raise ValueError("each paint has at most one detached straight 1..9-cell counter")
        if _near(cells) & housing:
            raise ValueError("counter touches housing")
        neighbors = _near(cells)
        if any(0 <= y < a.shape[0] and 0 <= x < a.shape[1] and a[y, x] != bg
               for y, x in neighbors):
            raise ValueError("counter touches another object")
        bars |= cells
        lengths[paint] = len(cells)
        orientations[paint] = 'single' if len(cells) == 1 else ('H' if horizontal else 'V')
    clean = a.copy()
    for p in bars:
        clean[p] = bg
    routes = _components(map(tuple, np.argwhere(clean == info['wire']).tolist()))
    lookup = {p: i for i, cells in enumerate(routes) for p in cells}
    route_ports = [[] for _ in routes]
    for absolute, inward in info['sockets']:
        route_ports[lookup[absolute]].append(inward)
    if any(len(ports) != 1 for ports in route_ports):
        raise ValueError("every separate route must have exactly one hub socket")
    cavities = []
    for cells in _components(map(tuple, np.argwhere(clean == bg).tolist()), D4):
        if cells & rotor:
            if not cells <= rotor:
                raise ValueError("hub rim must enclose its interior")
            continue
        if any(y in (0, a.shape[0]-1) or x in (0, a.shape[1]-1) for y, x in cells):
            continue
        rim = _near(cells, D4)
        colors = {int(clean[p]) for p in rim}
        if len(colors) != 1 or colors & ({bg, info['wire']} | set(info['paints'])):
            raise ValueError("unintended enclosure or ambiguous target shell")
        color = colors.pop()
        boundary = {p for p in _near(cells) if clean[p] == color}
        owners = {lookup[p] for p in _near(boundary) if p in lookup}
        if len(owners) > 1:
            raise ValueError("a contour must attach to at most one route")
        cavities.append(dict(cells=cells, boundary=boundary, color=color,
                             owner=next(iter(owners)) if owners else None))
    if not cavities:
        raise ValueError("scene must have hollow target contours")
    info.update(a=a, clean=clean, bars=bars, lengths=lengths, orientations=orientations,
                rotor_cells=rotor, routes=routes, route_ports=route_ports, cavities=cavities)
    return info


def _rotated(info, direction=1, ignore_counts=False, global_count=None):
    source, n, bg = info['source'], info['n'], info['bg']
    turned = np.full((n, n), bg, dtype=np.int8)
    for color in info['paints']:
        count = 0 if ignore_counts else info['lengths'][color]
        if global_count is not None:
            count = global_count
        for y, x in np.argwhere(source == color):
            p = _turn((int(y), int(x)), n, direction * count)
            if turned[p] != bg:
                raise ValueError("independently rotated masks conflict")
            turned[p] = color
    return turned


def _result(info, turned, paint=True):
    out = info['clean'].copy()
    r, c, n, bg = info['r'], info['c'], info['n'], info['bg']
    out[r+1:r+n+1, c+1:c+n+1] = turned
    supplies = [int(turned[ports[0]]) for ports in info['route_ports']]
    if paint:
        for cavity in info['cavities']:
            owner = cavity['owner']
            if owner is not None and supplies[owner] != bg:
                for p in cavity['cells']:
                    out[p] = supplies[owner]
    return out, supplies


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from input alone; no training examples, seed, or generation state."""
    info = _analyze(input_grid)
    return _result(info, _rotated(info))[0]


def _orbits(n, interior=True):
    left = {(r, c) for r in range(1, n-1) for c in range(1, n-1)} if interior else {
        (r, c) for r in range(n) for c in range(n)}
    groups = []
    while left:
        p = min(left)
        group = sorted({_turn(p, n, k) for k in range(4)})
        left.difference_update(group)
        groups.append(group)
    return groups


def _masks(rng, n, bg, paints, anchors):
    """Own whole rotation orbits, then draw different motifs inside each territory."""
    final = np.full((n, n), bg, dtype=np.int8)
    groups = _orbits(n)
    # Distance to the rotating boundary anchors varies territories geometrically.
    owner_groups = [[] for _ in paints]
    for j in rng.permutation(len(groups)):
        group = groups[int(j)]
        scores = [min(abs(y-u)+abs(x-v) for y, x in group
                      for u, v in {_turn(anchor, n, k) for k in range(4)})
                  + float(rng.uniform(0, 2.5)) for anchor in anchors]
        owner_groups[int(np.argmin(scores))].append(group)
    for i, color in enumerate(paints):
        final[anchors[i]] = color
        motif = int(rng.integers(0, 5))
        density = float(rng.uniform(.25, .9))
        for group in owner_groups[i]:
            if motif == 0:
                chosen = [p for p in group if rng.random() < density]
            elif motif == 1:
                chosen = sorted(group, key=lambda p: abs(p[0]-anchors[i][0])+abs(p[1]-anchors[i][1]))[:int(rng.integers(1, len(group)+1))]
            elif motif == 2:
                chosen = [p for p in group if (p[0]+i) % 2 == 0]
            elif motif == 3:
                chosen = group if rng.random() < density else []
            else:
                phase = int(rng.integers(len(group)))
                chosen = [group[phase]]
                if len(group) == 4 and rng.random() < .5:
                    chosen.append(_turn(group[phase], n, 2))
            for p in chosen:
                final[p] = color
    return final


def _contour(rng, tiny=False):
    h = int(rng.integers(1, 3 if tiny else 5))
    w = int(rng.integers(1, 4 if tiny else 6))
    kind = int(rng.integers(0, 5))
    cells = set()
    if kind == 4:
        radius = 1 if tiny else int(rng.integers(1, 3))
        cells = {(r, c) for r in range(-radius, radius+1)
                 for c in range(-radius, radius+1) if abs(r)+abs(c) <= radius}
    else:
        left = 0
        for r in range(h):
            if kind == 1 and r:
                left += int(rng.choice([-1, 0, 1]))
            if kind == 2:
                left = r // 2
            width = max(2, w) if kind in (1, 2) else w
            if kind == 3 and 0 < r < h-1:
                width = max(1, w-1)
            cells.update((r, c) for c in range(left, left+width))
    if rng.random() < .5:
        cells = {(c, r) for r, c in cells}
    boundary = _near(cells, D4)
    if kind == 0 and rng.random() < .5:
        boundary = _near(cells)
    r0 = min(r for r, c in boundary)
    c0 = min(c for r, c in boundary)
    return {(r-r0, c-c0) for r, c in cells}, {(r-r0, c-c0) for r, c in boundary}


def _path(rng, starts, goals, allowed):
    queue = deque(sorted(starts))
    previous = {p: None for p in starts}
    steps = [D8[int(i)] for i in rng.permutation(8)]
    end = None
    while queue:
        p = queue.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in steps:
            q = (p[0]+dr, p[1]+dc)
            if q in allowed and q not in previous:
                previous[q] = p
                queue.append(q)
    if end is None:
        return None
    result = set()
    while end is not None:
        result.add(end)
        end = previous[end]
    return result


def _bars(rng, a, bg, housing, paints, lengths, forced):
    """Pack independent horizontal/vertical counters in any available exterior space."""
    h, w = a.shape
    for i in sorted(range(len(paints)), key=lambda j: -lengths[j]):
        length = lengths[i]
        if length == 0:
            continue
        occupied = set(map(tuple, np.argwhere(a != bg).tolist())) | housing
        forbidden = occupied | _near(occupied)
        candidates = []
        axes = [forced[i]] if i in forced else [0, 1]
        for axis in axes:
            for r in range(h - (length-1 if axis == 0 else 0)):
                for c in range(w - (length-1 if axis == 1 else 0)):
                    cells = {(r+j if axis == 0 else r, c+j if axis == 1 else c) for j in range(length)}
                    if not cells & forbidden:
                        candidates.append(cells)
        if not candidates:
            return False
        cells = candidates[int(rng.integers(len(candidates)))]
        for p in cells:
            a[p] = paints[i]
    return True


def _scene(rng, lesson, test, previous_bg, attempt):
    # Geometry is sampled independently of counter lengths and literal palette.
    n = int(rng.integers(7, 12)) if lesson >= 0 else int(rng.integers(5, 12))
    h, w = int(rng.integers(25, 31)), int(rng.integers(25, 31))
    if lesson < 0 and rng.random() < .35:
        h, w = int(rng.integers(22, 31)), int(rng.integers(22, 31))
    r = int(rng.integers(4, h-n-5))
    c = int(rng.integers(4, w-n-5))
    palette = [int(x) for x in rng.permutation(10)]
    if palette[0] == previous_bg:
        palette[0], palette[1] = palette[1], palette[0]
    bg, frame, wire, extra = palette[:4]
    pcount = 3 if lesson >= 0 else int(rng.integers(2, min(4, n-2)+1))
    paints = palette[4:4+pcount]
    if lesson == 0:
        lengths, forced = [1, 2, 2], {1: 0, 2: 1}
    elif lesson == 1:
        lengths, forced = [3, 4, 0], {}
    else:
        lengths = [int(rng.integers(0, 5)) for _ in paints]
        lengths[0] = int(rng.choice([1, 2, 3]))
        forced = {}
    if test:
        lengths[0] = int(rng.choice([5, 6, 7, 9]))
        for i in range(1, pcount):
            lengths[i] = int(rng.integers(0, 10))
    # Interior territories own separate orbits. Boundary nozzles may share an
    # orbit, allowing a stationary socket to change paint, not just activation.
    seeds = rng.choice(np.arange(1, n-1), pcount, replace=False)
    anchors = [_turn((0, int(x)), n, int(rng.integers(4))) for x in seeds]
    if lesson == 0 or (lesson < 0 and rng.random() < .4):
        phase = lengths[1] % 4
        if phase not in (0, -lengths[0] % 4):
            # Color 1 starts at color 0's final socket, then turns elsewhere.
            anchors[1] = _turn(anchors[0], n, phase)
    # The old position of color 0 is deliberately wired to an inactive circuit.
    ports = anchors + [_turn(anchors[0], n, -lengths[0])]
    if lesson < 0 and pcount < 4 and rng.random() < .4:
        choices = [(0, x) for x in range(1, n-1)]
        choices = [_turn(p, n, k) for p in choices for k in range(4)]
        choices = [p for p in choices if p not in ports]
        ports.append(choices[int(rng.integers(len(choices)))])
    sockets = [_port(p, r, c, n) for p in ports]
    tips = [p for p, delta in sockets]
    initial = [{p, (p[0]+delta[0], p[1]+delta[1]),
                (p[0]+2*delta[0], p[1]+2*delta[1])} for p, delta in sockets]
    if any((cells | _near(cells)) & other for i, cells in enumerate(initial) for other in initial[:i]):
        return None
    housing = {(y, x) for y in range(r, r+n+2) for x in range(c, c+n+2)}
    final = _masks(rng, n, bg, paints, anchors)
    source = np.full((n, n), bg, dtype=np.int8)
    for color, count in zip(paints, lengths):
        mask = np.rot90(final == color, count % 4)
        if np.any((source != bg) & mask):
            return None
        source[mask] = color
    a = np.full((h, w), bg, dtype=np.int8)
    a[r:r+n+2, c:c+n+2] = frame
    a[r+1:r+n+1, c+1:c+n+1] = source
    for cells in initial:
        for p in cells:
            a[p] = wire
    canvas = {(y, x) for y in range(h) for x in range(w)}
    reserved = housing | _near(housing) | set().union(*initial)
    reserved |= _near(set().union(*initial))
    owners = []
    for i in range(len(ports)):
        count = int(rng.integers(1, 3 if attempt > 25 else 4))
        if pcount == 4:
            count = 1
        if lesson >= 0:
            count = 2 if lesson == 0 and i == 0 else 1
        owners.extend([i]*count)
    # One touching unconnected target teaches that shells do not conduct paint.
    touching = lesson == 0 or (lesson < 0 and rng.random() < .35)
    detached = int(rng.integers(0, 2)) if lesson < 0 else 0
    owners.extend([None] * (detached + int(touching)))
    regions = []
    occupied = set()
    for index, owner in enumerate(owners):
        inner, edge = _contour(rng, tiny=lesson >= 0 or attempt > 25 or len(owners) > 7)
        lh, lw = max(y for y, x in edge)+1, max(x for y, x in edge)+1
        is_touch = touching and index == len(owners)-1
        placed = False
        for _ in range(90):
            rr, cc = int(rng.integers(1, h-lh)), int(rng.integers(1, w-lw))
            # Locality is a soft packing preference, not a circuit-membership
            # rule. Unrestricted draws still interleave route bounds, and the
            # socket/source colors are independent of destination placement.
            if owner is not None and rng.random() < .85:
                socket, delta = sockets[owner]
                distance = int(rng.integers(5, 13))
                sideways = int(rng.integers(-5, 6))
                center_y = socket[0] + distance*delta[0] + sideways*delta[1]
                center_x = socket[1] + distance*delta[1] - sideways*delta[0]
                rr = max(1, min(h-lh-1, center_y-lh//2))
                cc = max(1, min(w-lw-1, center_x-lw//2))
            if is_touch:
                base = regions[0]
                p = sorted(base['edge'])[int(rng.integers(len(base['edge'])))]
                q = sorted(edge)[int(rng.integers(len(edge)))]
                dr, dc = D8[int(rng.integers(8))]
                rr, cc = p[0]+dr-q[0], p[1]+dc-q[1]
            ins = {(y+rr, x+cc) for y, x in inner}
            border = {(y+rr, x+cc) for y, x in edge}
            cells = ins | border
            if not all(0 < y < h-1 and 0 < x < w-1 for y, x in cells):
                continue
            if cells & (reserved | occupied) or _near(cells) & reserved:
                continue
            near_old = _near(cells) & occupied
            if not is_touch and near_old:
                continue
            if is_touch and (not near_old or not near_old <= regions[0]['edge'] or len(near_old) > 2):
                continue
            color = regions[0]['color'] if is_touch else (frame if rng.random() < .6 else extra)
            regions.append(dict(inner=ins, edge=border, owner=owner, color=color))
            occupied |= cells
            placed = True
            break
        if not placed:
            return None
    wires = [set(cells) for cells in initial]
    for owner in rng.permutation(len(wires)):
        owner = int(owner)
        own = wires[owner]
        others = set().union(*(cells for i, cells in enumerate(wires) if i != owner))
        targets = [i for i, q in enumerate(regions) if q['owner'] == owner]
        for index in rng.permutation(targets):
            target = regions[int(index)]
            otherloops = occupied - target['inner'] - target['edge']
            forbidden = housing | _near(housing) | occupied | _near(otherloops) | others | _near(others)
            allowed = (canvas - forbidden) | own
            goals = (_near(target['edge']) & canvas) - forbidden
            path = _path(rng, own, goals, allowed)
            if path is None:
                return None
            own |= path
        wires[owner] = own
    for q in regions:
        for p in q['edge']:
            a[p] = q['color']
    for cells in wires:
        for p in cells:
            a[p] = wire
    if not _bars(rng, a, bg, housing, paints, lengths, forced):
        return None
    b = a.copy()
    b[r+1:r+n+1, c+1:c+n+1] = final
    for q in regions:
        if q['owner'] is not None and q['owner'] < pcount:
            for p in q['inner']:
                b[p] = paints[q['owner']]
    # Erase only exterior counter cells, retaining every part of the machine.
    exterior = np.ones(a.shape, dtype=bool)
    exterior[r+1:r+n+1, c+1:c+n+1] = False
    b[exterior & np.isin(a, paints)] = bg
    rotation = int(rng.integers(4))
    a, b = np.rot90(a, rotation).copy(), np.rot90(b, rotation).copy()
    try:
        info = _analyze(a)
        predicted = _result(info, _rotated(info))[0]
        if not np.array_equal(predicted, b) or len(info['cavities']) != len(regions):
            return None
        # This comparison requires an actual change of external route ownership.
        old = _result(info, info['source'])[0]
        cells = set().union(*(q['cells'] for q in info['cavities']))
        if not any(old[p] != b[p] for p in cells):
            return None
        if lesson == 0:
            features = _features(info)
            if not features['touching_inactive'] or not features['diagonal'] or not features['recolored_routes']:
                return None
    except ValueError:
        return None
    return dict(input=a, output=b)


def _fallback(rng, lesson, test, previous_bg):
    """Sparse, guaranteed packing if the free-placement attempts are exhausted.

    Three nozzles supply three radial circuits. In lesson 0, two share an edge
    orbit to demonstrate a route changing paint. The old nozzle of the first
    mask is on a fourth circuit. Its three possible positions are
    separated from the active sockets and targets for every nonzero residue.
    """
    palette = [int(x) for x in rng.permutation(10)]
    if palette[0] == previous_bg:
        palette[0], palette[1] = palette[1], palette[0]
    bg, frame, wire, extra = palette[:4]
    paints = palette[4:7]
    lengths = [1, 2, 2] if lesson == 0 else ([3, 4, 0] if lesson == 1 else
              [int(rng.integers(1, 4)), int(rng.integers(5)), int(rng.integers(5))])
    if test:
        lengths = [int(rng.choice([5, 6, 7, 9])), int(rng.integers(10)), int(rng.integers(10))]
    n, r, c = 11, 9, 9
    anchors = [(0, 1), (10, 9) if lesson == 0 else (5, 10), (10, 1)]
    ports = anchors + [_turn(anchors[0], n, -lengths[0])]
    final = _masks(rng, n, bg, paints, anchors)
    source = np.full((n, n), bg, dtype=np.int8)
    for color, count in zip(paints, lengths):
        source[np.rot90(final == color, count % 4)] = color
    a = np.full((30, 30), bg, dtype=np.int8)
    a[r:r+n+2, c:c+n+2] = frame
    a[r+1:r+n+1, c+1:c+n+1] = source
    centers, regions = [], []
    for owner, p in enumerate(ports):
        socket, delta = _port(p, r, c, n)
        distance = 6 if delta == (-1, 0) else 5 if delta in ((1, 0), (0, 1)) else 6
        center = (socket[0]+distance*delta[0], socket[1]+distance*delta[1])
        centers.append(center)
        for step in range(distance-1):
            a[socket[0]+step*delta[0], socket[1]+step*delta[1]] = wire
        regions.append((center, owner, frame if owner != 1 else extra))
    # A branch to a second target, with a detached diagonal-touching neighbor.
    for p in ((6, 11), (6, 10), (7, 9), (7, 8), (7, 7), (6, 6), (6, 5), (5, 4)):
        a[p] = wire
    regions.extend([((3, 4), 0, extra), ((6, 14), None, frame)])
    for (y, x), owner, color in regions:
        a[y-1:y+2, x-1:x+2] = color
        a[y, x] = bg
    marker_cells = set()
    for i, (color, length) in enumerate(zip(paints, lengths)):
        cells = ({(0, j) for j in range(length)} if i == 0 else
                 {(0, 12+j) for j in range(length)} if i == 1 else
                 {(1+j, 29) for j in range(length)})
        for p in cells:
            a[p] = color
        marker_cells |= cells
    b = a.copy()
    b[r+1:r+n+1, c+1:c+n+1] = final
    for center, owner, color in regions:
        if owner is not None and owner < 3:
            b[center] = paints[owner]
    for p in marker_cells:
        b[p] = bg
    k = int(rng.integers(4))
    a, b = np.rot90(a, k).copy(), np.rot90(b, k).copy()
    assert np.array_equal(solve(a), b), 'fallback geometry must follow the same rule'
    return dict(input=a, output=b)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    puzzle = {'train': [], 'test': []}
    seen = set()
    previous_bg = None
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            lesson = index if split == 'train' and index < 2 else -1
            for attempt in range(160):
                pair = _scene(rng, lesson, split == 'test', previous_bg, attempt)
                if pair is not None and _key(pair['input']) not in seen:
                    break
            else:
                pair = _fallback(rng, lesson, split == 'test', previous_bg)
                # A second palette/mask draw handles any accidental duplicate.
                for _ in range(20):
                    if _key(pair['input']) not in seen:
                        break
                    pair = _fallback(rng, lesson, split == 'test', previous_bg)
                else:
                    raise RuntimeError("could not produce a distinct fallback input")
            seen.add(_key(pair['input']))
            previous_bg = _locate(pair['input'])['bg']
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    turned = _rotated(info)
    _, supplies = _result(info, turned)
    old_supplies = [int(info['source'][ports[0]]) for ports in info['route_ports']]
    shapes = []
    counts = [0]*len(info['routes'])
    touching_inactive = 0
    for i, q in enumerate(info['cavities']):
        cells = q['cells']
        r0, c0 = min(y for y, x in cells), min(x for y, x in cells)
        shapes.append(tuple(sorted((y-r0, x-c0) for y, x in cells)))
        if q['owner'] is not None:
            counts[q['owner']] += 1
        active = q['owner'] is not None and supplies[q['owner']] != info['bg']
        for other in info['cavities'][:i]:
            other_active = other['owner'] is not None and supplies[other['owner']] != info['bg']
            touching_inactive += bool(active != other_active and q['boundary'] & _near(other['boundary']))
    diagonal = 0
    for cells in info['routes']:
        diagonal += sum((y+1, x+dc) in cells and (y+1, x) not in cells and (y, x+dc) not in cells
                        for y, x in cells for dc in (-1, 1))
    asymmetry = []
    for color in info['paints']:
        mask = info['source'] == color
        asymmetry.append(not np.array_equal(mask, np.rot90(mask, 2)))
    return dict(n=info['n'], shape=info['a'].shape, bg=info['bg'], layers=len(info['paints']),
                routes=len(info['routes']), cavities=len(shapes), contour_shapes=shapes,
                recolored_routes=sum(old != new and old != info['bg'] and new != info['bg']
                                     for old, new in zip(old_supplies, supplies)),
                counts=counts, inactive=sum(col == info['bg'] for col in supplies),
                detached=sum(q['owner'] is None for q in info['cavities']),
                touching_inactive=touching_inactive, diagonal=diagonal,
                lengths=tuple(info['lengths'][col] for col in info['paints']),
                orientations=set(info['orientations'].values()),
                mask_areas=tuple(int(np.sum(info['source'] == col)) for col in info['paints']),
                asymmetric=asymmetry)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, teaching = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _analyze(a)
            turned = _rotated(info)
            expected, supplies = _result(info, turned)
            assert np.array_equal(b, expected), 'wrong mask turn, gate supply, or cavity fill'
            r, c, n = info['r'], info['c'], info['n']
            for color in info['paints']:
                assert np.sum(info['source'] == color) == np.sum(b[r+1:r+n+1, c+1:c+n+1] == color)
            allowed_changes = info['rotor_cells'] | info['bars'] | set().union(*(q['cells'] for q in info['cavities']))
            assert set(map(tuple, np.argwhere(a != b).tolist())) <= allowed_changes
            for q in info['cavities']:
                fill = info['bg'] if q['owner'] is None else supplies[q['owner']]
                assert all(b[p] == fill for p in q['cells'])
            assert any(col != info['bg'] for col in supplies)
            assert any(col == info['bg'] for col in supplies)
            assert not np.array_equal(_result(info, turned, paint=False)[0], b), 'B must affect the answer'
            old = _result(info, info['source'])[0]
            assert any(old[p] != b[p] for q in info['cavities'] for p in q['cells']), 'A must change circuit supply'
            if split == 'train':
                teaching.append(info)
    # The first two lessons fit even the minimum 2/1 request.
    short = {length for info in teaching for length in info['lengths'].values()}
    assert {0, 1, 2, 3, 4} <= short
    assert any(_features(info)['touching_inactive'] for info in teaching)
    assert any(_features(info)['recolored_routes'] for info in teaching)
    assert any(_features(info)['diagonal'] for info in teaching)
    assert any(_features(info)['orientations'] >= {'H', 'V'} for info in teaching)
    assert any(max(_features(info)['counts']) >= 2 for info in teaching)
    direction_witnesses = []
    for info in teaching:
        try:
            wrong = _result(info, _rotated(info, -1))[0]
        except ValueError:
            continue  # Seek a direction witness that does not rely on a clash.
        direction_witnesses.append(not np.array_equal(wrong, _result(info, _rotated(info))[0]))
    assert any(direction_witnesses)
    assert any(len({k % 4 for k in info['lengths'].values()}) >= 2 for info in teaching)


def self_test():
    """Bounded 42-seed geometry/mechanism audit, plus targeted negative checks."""
    for seed in range(12):
        rng = np.random.default_rng(seed)
        for lesson, test in ((0, False), (1, False), (-1, True)):
            pair = _fallback(rng, lesson, test, None)
            info = _analyze(pair['input'])
            assert _features(info)['touching_inactive'] >= 1
            old, _ = _result(info, info['source'])
            assert any(old[p] != pair['output'][p] for q in info['cavities'] for p in q['cells'])
    metrics, fingerprints, structures = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        fingerprints.add(tuple(_key(q['input']) for s in ('train', 'test') for q in puzzle[s]))
        for split in ('train', 'test'):
            for q in puzzle[split]:
                info = _analyze(q['input'])
                f = _features(info)
                metrics.append(f)
                roles = {}
                norm = tuple(roles.setdefault(int(x), len(roles)) for x in q['input'].flat)
                structures.add((q['input'].shape, norm))
                untouched = q['input'].copy()
                assert np.array_equal(solve(untouched), q['output'])
                assert np.array_equal(untouched, q['input']), 'solve modified input'
                if split == 'test':
                    assert max(info['lengths'].values()) > 4
    assert len(fingerprints) == 42 and len(structures) >= 180
    assert len({f['shape'] for f in metrics}) >= 25
    assert {f['n'] for f in metrics} == set(range(5, 12))
    assert {f['layers'] for f in metrics} == {2, 3, 4}
    assert len({f['bg'] for f in metrics}) >= 8
    assert len({s for f in metrics for s in f['contour_shapes']}) >= 35
    assert max(f['cavities'] for f in metrics) >= 8
    assert max(max(f['counts']) for f in metrics) >= 3
    assert any(f['detached'] for f in metrics)
    assert sum(f['touching_inactive'] > 0 for f in metrics) >= 42
    assert sum(f['diagonal'] > 0 for f in metrics) >= 150
    assert {k for f in metrics for k in f['lengths']} == set(range(10))
    assert max(v for f in metrics for v in f['mask_areas']) >= 20
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(9000+a*10+b, a, b)
            assert len(p['train']) == a and len(p['test']) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    one, two = generate_puzzle(811, 2, 1), generate_puzzle(811, 2, 1)
    assert all(np.array_equal(a[k], b[k]) for s in one for a, b in zip(one[s], two[s]) for k in a)
    bad = {s: [{k: a.copy() for k, a in q.items()} for q in one[s]] for s in one}
    q = bad['test'][0]
    info = _analyze(q['input'])
    filled = next(cav for cav in info['cavities'] if any(q['input'][p] != q['output'][p] for p in cav['cells']))
    p = min(filled['cells'])
    q['output'][p] = info['bg']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('partial cavity fill accepted')
    return dict(seeds=42, pairs=len(metrics), distinct_puzzles=len(fingerprints),
                normalized_structures=len(structures),
                canvas_shapes=len({f['shape'] for f in metrics}),
                hub_sizes=sorted({f['n'] for f in metrics}),
                backgrounds=sorted({f['bg'] for f in metrics}),
                layer_counts=sorted({f['layers'] for f in metrics}),
                route_counts=sorted({f['routes'] for f in metrics}),
                cavity_counts=sorted({f['cavities'] for f in metrics}),
                distinct_cavity_shapes=len({s for f in metrics for s in f['contour_shapes']}),
                max_targets_per_route=max(max(f['counts']) for f in metrics),
                raw_counts=sorted({k for f in metrics for k in f['lengths']}),
                mask_area_range=[min(v for f in metrics for v in f['mask_areas']),
                                 max(v for f in metrics for v in f['mask_areas'])],
                pairs_with_diagonal_steps=sum(f['diagonal'] > 0 for f in metrics),
                pairs_with_touching_inactive=sum(f['touching_inactive'] > 0 for f in metrics),
                pairs_with_detached=sum(f['detached'] > 0 for f in metrics),
                note='Generator-owned semantic checks; no independent solver or blind prediction.')


if __name__ == '__main__':
    self_test()
    print('self_test passed (42 seeds, all legal counts, mechanism and diversity checks)')
