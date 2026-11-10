"""Paint circuit cartridges: rigid docking completes a selective paint network.

All solving conventions are observable in the input. Only standard library and
NumPy are used. Sources are cupped; walls are passive, conductors are not walls.
"""
from collections import deque
import json
import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(cells, steps=D8):
    left = set(cells)
    groups = []
    while left:
        root = min(left)
        left.remove(root)
        group = {root}
        todo = [root]
        for r, c in todo:
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in left:
                    left.remove(q)
                    group.add(q)
                    todo.append(q)
        groups.append(group)
    return groups


def _near(cells, steps=D8):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps} - set(cells)


def _cells(a, color):
    return set(map(tuple, np.argwhere(a == color)))


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _grid(a):
    a = np.asarray(a)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or np.any((a < 0) | (a > 9))):
        raise ValueError('expected a 1..30 integer grid with colors 0..9')
    return a.astype(int, copy=True)


def _port_cells(s):
    j = s // 2 - 1
    return ({(0, j), (0, j + 1)}, {(j, s - 1), (j + 1, s - 1)},
            {(s - 1, j), (s - 1, j + 1)}, {(j, 0), (j + 1, 0)})


def _hub(s, bg, wire, labels, links):
    a = np.full((s, s), bg, dtype=int)
    j = s // 2 - 1
    anchors = ({(1, j), (1, j + 1)}, {(j, s - 2), (j + 1, s - 2)},
               {(s - 2, j), (s - 2, j + 1)}, {(j, 1), (j + 1, 1)})
    # Clockwise quarter-ring links. A missing link leaves a genuine gap, even
    # under eight-neighbor conductor connectivity, for either supported size.
    corner = ({(1, c) for c in range(j + 2, s - 1)}
              | {(r, s - 2) for r in range(2, j)},
              {(r, s - 2) for r in range(j + 2, s - 1)}
              | {(s - 2, c) for c in range(j + 2, s - 2)},
              {(s - 2, c) for c in range(1, j)}
              | {(r, 1) for r in range(j + 2, s - 2)},
              {(r, 1) for r in range(1, j)}
              | {(1, c) for c in range(2, j)})
    for cells in anchors:
        for p in cells:
            a[p] = wire
    for on, cells in zip(links, corner):
        if on:
            for p in cells:
                a[p] = wire
    for color, cells in zip(labels, _port_cells(s)):
        for p in cells:
            a[p] = color
    return a


def _find_hub(a, bg):
    found = []
    for s in (8, 10):
        for r in range(a.shape[0] - s + 1):
            for c in range(a.shape[1] - s + 1):
                j = s // 2 - 1
                if a[r, c + j] == bg or a[r, c + j] != a[r, c + j + 1]:
                    continue
                b = a[r:r + s, c:c + s]
                labels = []
                for cells in _port_cells(s):
                    colors = {int(b[p]) for p in cells}
                    if len(colors) != 1 or bg in colors:
                        break
                    labels.append(colors.pop())
                if len(labels) != 4 or len(set(labels)) != 4:
                    continue
                ports = set().union(*_port_cells(s))
                border = {(y, x) for y in range(s) for x in range(s)
                          if y in (0, s - 1) or x in (0, s - 1)} - ports
                if any(b[p] != bg for p in border):
                    continue
                colors = set(map(int, b[1:-1, 1:-1].flat)) - {bg}
                if len(colors) != 1 or colors & set(labels):
                    continue
                wire = colors.pop()
                # Exactly a subset of the four quarter-ring links is encoded.
                for bits in range(16):
                    links = tuple(bool(bits & (1 << k)) for k in range(4))
                    if np.array_equal(b, _hub(s, bg, wire, labels, links)):
                        found.append((r, c, s, wire, tuple(labels), links))
                        break
    if len(found) != 1:
        raise ValueError('one uniquely recognizable wired hub is required')
    return found[0]


def _canonical(module, bg, wire):
    # A plug is the unique two-cell color pair on a sole extreme edge. Its
    # terminal geometry, not any sampler-selected orientation, defines motion.
    candidates = []
    for color in sorted(set(map(int, module.flat)) - {bg, wire}):
        if np.count_nonzero(module == color) != 2:
            continue
        for k in range(4):
            b = np.rot90(module, k)
            pin = np.argwhere(b == color)
            if (np.all(pin[:, 0] == b.shape[0] - 1)
                    and abs(int(pin[0, 1] - pin[1, 1])) == 1
                    and np.count_nonzero(b[-1] != bg) == 2
                    and b.shape[0] >= 3):
                candidates.append((b.copy(), color, k))
    if len(candidates) != 1:
        raise ValueError('cartridge must have one terminal colored plug')
    return candidates[0]


def _parse(a):
    bg = int(np.bincount(a.ravel(), minlength=10).argmax())
    if np.count_nonzero(a == bg) <= a.size // 2:
        raise ValueError('background must occupy a strict majority')
    r, c, s, wire, labels, links = _find_hub(a, bg)
    hubcells = {(r + y, c + x) for y, x in np.argwhere(a[r:r+s, c:c+s] != bg)}
    outside = set(map(tuple, np.argwhere(a != bg))) - hubcells
    modules = []
    boundary_colors = set()
    used = set()
    for group in _components(outside):
        if _near(group) & hubcells:
            raise ValueError('loose cartridges must be separated from the hub')
        yy, xx = zip(*group)
        top, left = min(yy), min(xx)
        b = np.full((max(yy) - top + 1, max(xx) - left + 1), bg, dtype=int)
        for y, x in group:
            b[y - top, x - left] = a[y, x]
        canonical, plug, undo = _canonical(b, bg, wire)
        extras = set(map(int, b.flat)) - {bg, wire, plug}
        walls = {color for color in extras if np.count_nonzero(b == color) >= 4}
        if len(walls) > 1:
            raise ValueError('only one chamber wall color is allowed')
        boundary_colors.update(walls)
        side = labels.index(plug) if plug in labels else None
        if side is not None:
            if side in used:
                raise ValueError('at most one cartridge per socket')
            used.add(side)
        modules.append(dict(cells=group, canonical=canonical, plug=plug,
                            side=side, undo=undo))
    if not modules or len(boundary_colors) != 1:
        raise ValueError('at least one chamber and one shared wall color are required')
    wall = boundary_colors.pop()
    if wall in {bg, wire, *labels}:
        raise ValueError('wall, wire, background and socket roles must differ')
    return dict(bg=bg, row=r, col=c, size=s, wire=wire, wall=wall,
                labels=labels, links=links, modules=modules)


def _dock(module, side, r, c, s, bg):
    oriented = np.rot90(module, -side)
    # Mark only the terminal pair; rotation carries its offset and all contents.
    pins = np.zeros(module.shape, dtype=bool)
    pins[-1] = module[-1] != bg
    pp = np.argwhere(np.rot90(pins, -side))
    target = np.array(sorted(_port_cells(s)[side])) + (r, c)
    delta = ((-1, 0), (0, 1), (1, 0), (0, -1))[side]
    target += delta
    shift = target.min(axis=0) - pp.min(axis=0)
    return oriented, shift


def _paint(board, info):
    bg, wire, wall = info['bg'], info['wire'], info['wall']
    labels = info['labels']
    conductors = set(map(tuple, np.argwhere(np.isin(board, (wire, *labels)))))
    groups = _components(conductors)
    membership = {p: i for i, group in enumerate(groups) for p in group}
    supplies = {}
    sources = []
    for color in sorted(set(map(int, board.flat)) - {bg, wire, wall, *labels}):
        for singleton in _components(_cells(board, color)):
            if len(singleton) != 1:
                raise ValueError('paint sources must be isolated one-cell markers')
            p = next(iter(singleton))
            nb = {q for q in _near(singleton) if 0 <= q[0] < board.shape[0]
                  and 0 <= q[1] < board.shape[1] and board[q] != bg}
            if len(nb) not in (3, 5) or any(board[q] != wire for q in nb):
                raise ValueError('source must sit in a three- or five-neighbor wire cup')
            owner = {membership[q] for q in nb}
            if len(owner) != 1:
                raise ValueError('source has ambiguous conductor ownership')
            owner = owner.pop()
            if owner in supplies and supplies[owner] != color:
                raise ValueError('conflicting paints on one circuit')
            supplies[owner] = color
            sources.append((p, color, owner))
    out = board.copy()
    loops = []
    background_groups = _components(_cells(board, bg), D4)
    for interior in background_groups:
        if any(y in (0, board.shape[0] - 1) or x in (0, board.shape[1] - 1)
               for y, x in interior):
            continue
        border = _near(interior, D4)
        if not border or any(board[q] != wall for q in border):
            continue  # The ring's background hole is not a chamber.
        shell = {q for q in _near(interior) if board[q] == wall}
        owners = {membership[q] for q in _near(shell) & conductors}
        if len(owners) > 1:
            raise ValueError('chamber is attached to multiple circuits')
        owner = next(iter(owners)) if owners else None
        fill = supplies.get(owner)
        if fill is not None:
            for p in interior:
                out[p] = fill
        loops.append(dict(interior=interior, shell=shell, owner=owner, fill=fill))
    return out, dict(groups=groups, supplies=supplies, sources=sources, loops=loops)


def _analyze(a):
    info = _parse(a)
    bg, r, c, s = info['bg'], info['row'], info['col'], info['size']
    board = np.full_like(a, bg)
    board[r:r+s, c:c+s] = a[r:r+s, c:c+s]
    for module in info['modules']:
        if module['side'] is None:
            continue
        b, shift = _dock(module['canonical'], module['side'], r, c, s, bg)
        local = np.argwhere(b != bg)
        coords = local + shift
        if (np.any(coords < 0) or np.any(coords >= np.array(a.shape))
                or any(board[tuple(p)] != bg for p in coords)):
            raise ValueError('docked cartridge clips or overlaps the assembly')
        for (y, x), (dy, dx) in zip(local, coords):
            board[dy, dx] = b[y, x]
        module['destination'] = set(map(tuple, coords))
        module['rotation'] = (module['undo'] - module['side']) % 4
    out, electrical = _paint(board, info)
    info.update(electrical)
    info['assembled'] = board
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover object motion and electrical ownership entirely from this input."""
    return _analyze(_grid(input_grid))[0]


def _path(rng, starts, goals, allowed):
    todo = deque(sorted(starts))
    parent = {p: None for p in starts}
    dirs = [D8[i] for i in rng.permutation(8)]
    end = None
    while todo:
        p = todo.popleft()
        if p in goals:
            end = p
            break
        for dr, dc in dirs:
            q = (p[0] + dr, p[1] + dc)
            if q in allowed and q not in parent:
                parent[q] = p
                todo.append(q)
    if end is None:
        return None
    result = set()
    while end is not None:
        result.add(end)
        end = parent[end]
    return result


def _cavity(rng):
    h, w = int(rng.integers(1, 4)), int(rng.integers(1, 4))
    kind = int(rng.integers(4))
    cells = set()
    for r in range(h):
        shift = (r // 2) if kind == 1 else 0
        width = max(1, w - (kind == 2 and r == h // 2))
        cells.update((r, c + shift) for c in range(width))
    if kind == 3 and h > 1 and w > 1:
        cells.discard((0, w - 1))
    wall = _near(cells, D4)
    y0, x0 = min(y for y, x in wall), min(x for y, x in wall)
    return ({(y-y0, x-x0) for y, x in cells},
            {(y-y0, x-x0) for y, x in wall})


def _cup(style, rotation):
    src = {(0, 0)} if style == 3 else {(0, 1)}
    wire = {(0, 1), (1, 0), (1, 1)} if style == 3 else {
        (0, 0), (0, 2), (1, 0), (1, 1), (1, 2)}
    for _ in range(rotation):
        src = {(-x, y) for y, x in src}
        wire = {(-x, y) for y, x in wire}
    y0, x0 = min(y for y, x in src | wire), min(x for y, x in src | wire)
    return ({(y-y0, x-x0) for y, x in src}, {(y-y0, x-x0) for y, x in wire})


def _module(rng, s, bg, wire, wall, plug, paint=None, count=1,
            passive=False, style=3):
    """Free small contour placement with geometry-checked wiring and cups."""
    for attempt in range(160):
        h = int(rng.integers(5, 9))
        w = int(rng.integers(5, min(8, s) + 1))
        j = w // 2 - 1
        pins = {(h-1, j), (h-1, j+1)}
        root = {(h-2, j), (h-2, j+1)}
        canvas = {(y, x) for y in range(h-1) for x in range(w)}
        regions = []
        occupied = set()
        for i in range(count):
            inner, shell = _cavity(rng)
            lh = max(y for y, x in shell)+1
            lw = max(x for y, x in shell)+1
            if lh > h-2 or lw > w:
                break
            placed = False
            for _ in range(30):
                dy = int(rng.integers(0, h-1-lh))
                dx = int(rng.integers(0, w-lw+1))
                ins = {(y+dy, x+dx) for y, x in inner}
                edge = {(y+dy, x+dx) for y, x in shell}
                if (ins | edge) & (occupied | root | _near(root)):
                    continue
                if i == count-1 and passive:
                    if not _near(edge) & occupied:
                        continue
                    if _near(ins, D4) & occupied:
                        continue
                elif _near(ins | edge) & occupied:
                    continue
                regions.append((ins, edge, not (passive and i == count-1)))
                occupied |= ins | edge
                placed = True
                break
            if not placed:
                break
        if len(regions) != count:
            continue
        src = set()
        cupwire = set()
        if paint is not None:
            source, cup = _cup(style, int(rng.integers(4)))
            lh = max(y for y, x in source | cup)+1
            lw = max(x for y, x in source | cup)+1
            for _ in range(40):
                dy = int(rng.integers(0, h-1-lh+1))
                dx = int(rng.integers(0, w-lw+1))
                src = {(y+dy, x+dx) for y, x in source}
                cupwire = {(y+dy, x+dx) for y, x in cup}
                if not (src | cupwire | _near(src | cupwire)) & (occupied | root | pins):
                    break
            else:
                continue
        wires = set(root)
        blocked = occupied | src | _near(src) | pins
        allowed = canvas - blocked | wires | cupwire
        if cupwire:
            path = _path(rng, wires, cupwire, allowed)
            if path is None:
                continue
            wires |= path | cupwire
        good = True
        for ins, edge, active in regions:
            if not active:
                continue
            other = set().union(*(ii | ee for ii, ee, aa in regions if ee is not edge))
            allowed = canvas - occupied - src - _near(src) - _near(other) | wires
            goals = (_near(edge) & allowed) - _near(pins)
            path = _path(rng, wires, goals, allowed)
            if path is None:
                good = False
                break
            wires |= path
        if not good or any(not active and _near(edge) & (wires | pins)
                           for ins, edge, active in regions):
            continue
        a = np.full((h, w), bg, dtype=int)
        for ins, edge, active in regions:
            for p in edge:
                a[p] = wall
        for p in wires:
            a[p] = wire
        for p in src:
            a[p] = paint
        for p in pins:
            a[p] = plug
        fg = set(map(tuple, np.argwhere(a != bg)))
        if len(_components(fg)) != 1:
            continue
        yy, xx = zip(*fg)
        a = a[min(yy):max(yy)+1, min(xx):max(xx)+1]
        try:
            _canonical(a, bg, wire)
            # A temporary board supplies local checks without a generation mask.
            pad = np.pad(a, 1, constant_values=bg)
            _, info = _paint(pad, dict(bg=bg, wire=wire, wall=wall, labels=(plug,)))
            if len(info['loops']) != count or len(info['sources']) != (paint is not None):
                continue
            if passive and sum(q['owner'] is None for q in info['loops']) != 1:
                continue
        except ValueError:
            continue
        return a
    return _fallback_module(bg, wire, wall, plug, paint, count, passive, style)


def _fallback_module(bg, wire, wall, plug, paint, count, passive, style):
    # Sparse two-wall-contact construction, still respecting the requested
    # mechanisms. It never silently replaces an isolation witness.
    a = np.full((8, 8), bg, dtype=int)
    a[7, 3:5] = plug
    wires = {(6, 3), (6, 4), (5, 3), (4, 3)}
    interiors = [{(1, 2), (2, 2)}] if count else []
    if count == 2:
        interiors.append({(1, 5), (2, 5)})
    for ins in interiors:
        for p in _near(ins, D4):
            a[p] = wall
    if count == 2 and not passive:
        wires |= {(4, 4), (4, 5)}
    if paint is not None:
        # Paint at (5,0); the 3/5-neighbor cup opens left, away from contours.
        a[5, 0] = paint
        wires |= {(4, 0), (4, 1), (5, 1)}
        if style == 5:
            wires |= {(6, 0), (6, 1)}
        wires |= {(5, 2)}
    for p in wires:
        a[p] = wire
    yy, xx = np.nonzero(a != bg)
    return a[yy.min():yy.max()+1, xx.min():xx.max()+1]


def _place(rng, shape, bg, hub, pos, objects):
    a = np.full(shape, bg, dtype=int)
    r, c = pos
    s = hub.shape[0]
    a[r:r+s, c:c+s] = hub
    blocked = np.zeros(shape, dtype=bool)
    blocked[max(0, r-1):min(shape[0], r+s+1),
            max(0, c-1):min(shape[1], c+s+1)] = True
    order = list(map(int, rng.permutation(len(objects))))
    order.sort(key=lambda i: -objects[i].size)
    for i in order:
        b = objects[i]
        hh, ww = b.shape
        if hh > shape[0] or ww > shape[1]:
            return None
        local = np.argwhere(b != bg)
        candidates = np.indices((shape[0]-hh+1, shape[1]-ww+1)).reshape(2, -1).T
        rr = candidates[:, 0, None] + local[:, 0]
        cc = candidates[:, 1, None] + local[:, 1]
        good = np.flatnonzero(~blocked[rr, cc].any(axis=1))
        if not len(good):
            return None
        k = int(rng.choice(good))
        for (y, x), dy, dx in zip(local, rr[k], cc[k]):
            a[dy, dx] = b[y, x]
            blocked[max(0, dy-1):dy+2, max(0, dx-1):dx+2] = True
    return a


def _make_pair(rng, config, preferred=None):
    palette = list(map(int, rng.permutation(10)))
    if preferred is not None:
        palette.remove(preferred)
        palette.insert(8, preferred)
    bg, wire, wall = palette[:3]
    labels, unmatched, paints = palette[3:7], palette[7], palette[8:10]
    s = int(rng.choice((8, 10)))
    hub = _hub(s, bg, wire, labels, config['links'])
    objects = []
    for side in sorted(config['matches']):
        source = config['sources'].get(side)
        count = int(rng.choice((0, 1))) if source is not None else int(rng.choice((1, 1, 2)))
        passive = side == config.get('passive')
        if passive:
            count = 2
        module = _module(rng, s, bg, wire, wall, labels[side],
                         None if source is None else paints[source], count, passive,
                         config['style'])
        rotation = config.get('turns', {}).get(side, int(rng.integers(4)))
        objects.append(np.rot90(module, -side - rotation).copy())
    for _ in range(config['distractors']):
        module = _module(rng, s, bg, wire, wall, unmatched, paints[1],
                         int(rng.choice((0, 1))), False, config['style'])
        objects.append(np.rot90(module, int(rng.integers(4))).copy())
    # Actual output geometry controls feasible socket position, not the initial
    # locations of objects. Place anywhere in that feasible region.
    assembled_cells = set(map(tuple, np.argwhere(hub != bg)))
    for b in objects:
        canonical, plug, _ = _canonical(b, bg, wire)
        if plug not in labels:
            continue
        oriented, shift = _dock(canonical, labels.index(plug), 0, 0, s, bg)
        coords = np.argwhere(oriented != bg) + shift
        assembled_cells.update(map(tuple, coords))
    yy, xx = zip(*assembled_cells)
    minimum = max(18, max(yy)-min(yy)+1), max(18, max(xx)-min(xx)+1)
    if max(minimum) > 30:
        raise ValueError('assembly is too large')
    for attempt in range(32):
        h = int(rng.integers(minimum[0], 31))
        w = int(rng.integers(minimum[1], 31))
        if attempt > 12:
            h = w = 30
        low = (-min(yy), -min(xx))
        high = (h-1-max(yy), w-1-max(xx))
        pos = tuple(int(rng.integers(lo, hi+1)) for lo, hi in zip(low, high))
        a = _place(rng, (h, w), bg, hub, pos, objects)
        if a is None:
            continue
        try:
            b, info = _analyze(a)
        except ValueError:
            continue
        if not any(q['fill'] is not None for q in info['loops']):
            continue
        if config.get('passive') is not None and not any(q['owner'] is None for q in info['loops']):
            continue
        return dict(input=a, output=b)
    raise ValueError('could not pack a circuit scene')


def _counts(num_train, num_test):
    for n, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not lo <= n <= hi:
            raise ValueError('counts must be integers: train 2..10, test 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    phase = int(rng.integers(4))
    side = lambda x: (x + phase) % 4
    first = dict(links=tuple(k in (phase, (phase+1) % 4) for k in range(4)),
                 matches={side(0), side(2), side(3)}, sources={side(0): 0},
                 distractors=1, style=3)
    second = dict(links=tuple(k in (phase, (phase+2) % 4) for k in range(4)),
                  matches=set(range(4)), sources={side(0): 0, side(2): 1},
                  passive=side(1), distractors=0, style=5,
                  turns={side(k): k for k in range(4)})
    configs = [first, second]
    preferred = list(map(int, rng.choice(10, 2, replace=False)))
    for _ in range(num_train-2):
        p = int(rng.integers(4))
        links = tuple(k == p or (rng.random() < .4 and k == (p+1) % 4) for k in range(4))
        source = p
        receiver = (p+1) % 4
        others = {k for k in range(4) if k not in (source, receiver) and rng.random() < .55}
        configs.append(dict(links=links, matches={source, receiver} | others,
                            sources={source: 0}, distractors=int(rng.integers(2)),
                            style=int(rng.choice((3, 5)))))
        preferred.append(None)
    # Training order is sampled. The two teaching scenes have complementary
    # evidence rather than defining an ascending object-count template.
    order = list(map(int, rng.permutation(num_train)))
    configs = [configs[i] for i in order]
    preferred = [preferred[i] for i in order]
    for _ in range(num_test):
        gap = int(rng.integers(4))
        links = tuple(k != gap or rng.random() < .25 for k in range(4))
        source = int(rng.integers(4))
        n = int(rng.integers(2, 5))
        rest = [k for k in range(4) if k != source]
        matches = {source} | set(map(int, rng.choice(rest, n-1, replace=False)))
        configs.append(dict(links=links, matches=matches, sources={source: 0},
                            distractors=int(rng.integers(2)),
                            style=int(rng.choice((3, 5)))))
        preferred.append(None)
    puzzle = {'train': [], 'test': []}
    seen = set()
    for i, (config, paint) in enumerate(zip(configs, preferred)):
        for _ in range(20):
            try:
                pair = _make_pair(rng, config, paint)
            except ValueError:
                continue
            if _key(pair['input']) not in seen:
                break
        else:
            raise RuntimeError('bounded scene construction failed')
        seen.add(_key(pair['input']))
        puzzle['train' if i < num_train else 'test'].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in ('train', 'test'))
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen = set()
    analyses = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            expected, info = _analyze(a)
            assert np.array_equal(expected, b), 'wrong docking or chamber fill'
            assert not np.array_equal(a, b)
            assert info['sources'] and any(q['fill'] is not None for q in info['loops'])
            # Filling changes only chamber interiors. Every moved structural
            # pixel, passive wall, plug and source remains exactly preserved.
            changed = set(map(tuple, np.argwhere(info['assembled'] != b)))
            wanted = set().union(*(q['interior'] for q in info['loops'] if q['fill'] is not None))
            assert changed == wanted
            assert all(info['assembled'][p] == info['bg'] for p in changed)
            # Remote propagation is necessary in every pair: a receiver module
            # with no source marker has a supplied chamber.
            remote = False
            for module in info['modules']:
                dest = module.get('destination', set())
                if not dest:
                    continue
                has_source = any(p in dest for p, color, owner in info['sources'])
                has_fill = any(q['fill'] is not None and q['shell'] <= dest for q in info['loops'])
                remote |= has_fill and not has_source
            assert remote, 'each scene must demonstrate a remote receiver'
            analyses[split].append(info)
    train = analyses['train']
    assert set().union(*(set(m['side'] for m in a['modules'] if m['side'] is not None) for a in train)) == set(range(4))
    assert set().union(*(set(m['rotation'] for m in a['modules'] if m['side'] is not None) for a in train)) == set(range(4))
    assert len({col for a in train for p, col, owner in a['sources']}) >= 2
    assert any(len(a['supplies']) >= 2 for a in train), 'demonstrate independent paint circuits'
    assert any(m['side'] is None for a in train for m in a['modules']), 'demonstrate filtering'
    assert any(q['fill'] is None and q['owner'] is not None for a in train for q in a['loops']), 'demonstrate inactive circuits'
    assert any(q['owner'] is None and any(q['shell'] & _near(other['shell']) and other['fill'] is not None for other in a['loops'])
               for a in train for q in a['loops']), 'demonstrate passive wall contact'
    assert all(sum(a['links']) <= 2 for a in train)
    assert all(sum(a['links']) >= 3 for a in analyses['test'])


def _puzzle_key(puzzle):
    return tuple(tuple((_key(p['input']), _key(p['output'])) for p in puzzle[s]) for s in ('train', 'test'))


def self_test():
    """Bounded semantic, mutation, equivariance and 42-seed diversity checks."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    signatures, shapes, module_shapes, loop_shapes = set(), set(), set(), set()
    backgrounds, hub_sizes, counts, links, cup_degrees = set(), set(), set(), set(), set()
    for puzzle in puzzles:
        signatures.add(_puzzle_key(puzzle))
        for pair in puzzle['train'] + puzzle['test']:
            a, b = pair['input'], pair['output']
            _, info = _analyze(a)
            shapes.add(a.shape)
            backgrounds.add(info['bg'])
            hub_sizes.add(info['size'])
            counts.add(len(info['modules']))
            links.add(info['links'])
            for m in info['modules']:
                mask = m['canonical'] != info['bg']
                module_shapes.add((mask.shape, mask.tobytes()))
            for q in info['loops']:
                y0 = min(y for y, x in q['interior'])
                x0 = min(x for y, x in q['interior'])
                loop_shapes.add(tuple(sorted((y-y0, x-x0) for y, x in q['interior'])))
            for p, color, owner in info['sources']:
                cup_degrees.add(int(sum(info['assembled'][q] == info['wire'] for q in _near({p})
                                        if 0 <= q[0] < a.shape[0] and 0 <= q[1] < a.shape[1])))
        # Quarter-turn and palette equivariance ensure the family does not
        # consult absolute orientation, literal paint or seed state.
        pair = puzzle['test'][0]
        assert np.array_equal(solve(np.rot90(pair['input'])), np.rot90(pair['output']))
        permutation = np.array((7, 3, 9, 0, 8, 2, 6, 1, 5, 4))
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    assert len(signatures) == 42
    assert len(shapes) >= 25 and len(module_shapes) >= 80 and len(loop_shapes) >= 10
    assert len(backgrounds) >= 8 and hub_sizes == {8, 10} and len(counts) >= 3
    assert len(links) >= 12 and cup_degrees == {3, 5}
    assert _puzzle_key(puzzles[0]) == _puzzle_key(generate_puzzle(0))
    # Counterfactuals isolate the interaction: same plugs, objects, locations,
    # and paint, but new hub links. Removing links prevents remote painting;
    # completing the ring supplies the previously inactive receiver.
    teaching = next(p for p in puzzles[0]['train']
                    if len(_analyze(p['input'])[1]['sources']) == 1
                    and any(q['fill'] is None and q['owner'] is not None
                            for q in _analyze(p['input'])[1]['loops']))
    a = teaching['input']
    expected, info = _analyze(a)
    r, c, s = info['row'], info['col'], info['size']
    for on in (False, True):
        altered = a.copy()
        altered[r:r+s, c:c+s] = _hub(s, info['bg'], info['wire'], info['labels'], (on,)*4)
        answer, alternate = _analyze(altered)
        lookup = {frozenset(q['interior']): q for q in alternate['loops']}
        for q in info['loops']:
            if on and q['owner'] is not None and q['fill'] is None:
                assert lookup[frozenset(q['interior'])]['fill'] is not None
            if not on and q['fill'] is not None:
                source_here = any(p in m.get('destination', set())
                                  for p, color, owner in info['sources']
                                  for m in info['modules']
                                  if q['shell'] <= m.get('destination', set()))
                if not source_here:
                    assert lookup[frozenset(q['interior'])]['fill'] is None
        assert not np.array_equal(answer, expected)
    # Ablating B yields the unpainted assembly; ablating A leaves independent
    # loose receivers. Neither can produce the intended answer.
    assert not np.array_equal(info['assembled'], expected)
    undocked = a.copy()
    for m in info['modules']:
        if m['side'] is None:
            for p in m['cells']:
                undocked[p] = info['bg']
    local_answer, local_info = _paint(undocked, info)
    assert not np.array_equal(local_answer, expected)
    assert sum(q['fill'] is not None for q in local_info['loops']) < sum(q['fill'] is not None for q in info['loops'])
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            p = generate_puzzle(10000+nt*10+nq, nt, nq)
            assert (len(p['train']), len(p['test'])) == (nt, nq)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ('train', 'test')}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError('damaged output accepted')
    return dict(seeds=42, distinct_puzzles=len(signatures), grid_shapes=len(shapes),
                cartridge_shapes=len(module_shapes), cavity_shapes=len(loop_shapes),
                backgrounds=len(backgrounds), hub_sizes=sorted(hub_sizes),
                cartridge_counts=sorted(counts), link_patterns=len(links),
                cup_degrees=sorted(cup_degrees))


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
