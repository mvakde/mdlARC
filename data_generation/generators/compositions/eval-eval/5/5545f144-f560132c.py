"""Routed cargo: jointly choose docking poses and read a transported legend.

Only the standard library and NumPy are used. There are no parent imports,
stored puzzles, seed-dependent solving conventions, or solver caches.
"""
import itertools
import numpy as np

DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
T = ((0, 0), (-1, -1), (0, -1), (1, -1))  # tip-led right arrow


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _rot(point, turns):
    r, c = point
    for _ in range(turns % 4):
        r, c = c, -r
    return r, c


def _direction(a, b):
    r, c = b[0] - a[0], b[1] - a[1]
    _require((r == 0) != (c == 0), "leg must be orthogonal and nonzero")
    return (1 if c > 0 else 3) if r == 0 else (2 if r > 0 else 0)


def _shift(p, d, length):
    dr, dc = DIRS[d]
    return p[0] + dr * length, p[1] + dc * length


def _components(cells, diagonal=True):
    left, result = set(cells), []
    neighbors = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1)
                 if (r or c) and (diagonal or abs(r) + abs(c) == 1)]
    while left:
        start = min(left)
        left.remove(start)
        comp, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    comp.add(q)
                    stack.append(q)
        result.append(comp)
    return result


def _arrow(points):
    answers = []
    for tip in sorted(points):
        for d in range(4):
            cells = {(tip[0] + r, tip[1] + c)
                     for r, c in (_rot(p, d - 1) for p in T)}
            if cells == points:
                answers.append((tip, d))
    _require(len(answers) == 1, "arrow must have one tip and facing")
    return answers[0]


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= a) & (a <= 9))), "grid colors")
    return a


def _parse(value):
    grid = _grid(value)
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    bg = int(np.bincount(border, minlength=10).argmax())
    _require(np.count_nonzero(border == bg) > len(border) // 2, "readable background")
    comps = _components((int(r), int(c)) for r, c in np.argwhere(grid != bg))
    frames = []
    for comp in comps:
        if len(comp) < 12:
            continue
        r0, r1 = min(r for r, c in comp), max(r for r, c in comp)
        c0, c1 = min(c for r, c in comp), max(c for r, c in comp)
        perimeter = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
                     if r in (r0, r1) or c in (c0, c1)}
        if r1 - r0 == c1 - c0 and comp == perimeter and len({int(grid[p]) for p in comp}) == 1:
            frames.append((comp, (r0 + 1, c0 + 1, r1 - r0 - 1)))
    _require(len(frames) == 1, "exactly one square dock frame")
    frame, dock = frames[0]
    carrier = int(grid[min(frame)])
    cargos, dots, anchor = [], {}, None
    for comp in comps:
        if comp == frame:
            continue
        if len(comp) == 1:
            point = next(iter(comp))
            dots.setdefault(int(grid[point]), set()).add(point)
            continue
        _require(len(_components(comp, False)) == 1, "cargo footprint connected")
        colors = {int(grid[p]) for p in comp}
        _require(carrier in colors, "cargo shares frame carrier")
        marker_candidates = []
        for color in sorted(colors - {carrier}):
            points = {p for p in comp if int(grid[p]) == color}
            if len(points) == 4:
                try:
                    tip, facing = _arrow(points)
                except AssertionError:
                    continue
                marker_candidates.append((color, tip, facing))
        _require(len(marker_candidates) == 1, "exactly one embedded arrow per cargo")
        color, tip, facing = marker_candidates[0]
        extras = colors - {carrier, color}
        key = None
        if extras:
            _require(anchor is None and len(extras) == 4, "one four-color legend cargo")
            points = {p for p in comp if int(grid[p]) in extras}
            r, c = min(p[0] for p in points), min(p[1] for p in points)
            _require(points == {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)}, "2 by 2 legend")
            key = grid[r:r + 2, c:c + 2].copy()
            _require(len(set(map(int, key.flat))) == 4, "distinct legend entries")
            anchor = len(cargos)
        cargos.append(dict(tip=tip, facing=facing, color=color, key=key,
                           offsets=tuple(sorted((r-tip[0], c-tip[1]) for r, c in comp))))
    _require(len(cargos) == 4 and anchor is not None, "four cargos and one legend")
    route_colors = {p['color'] for p in cargos}
    key_colors = set(map(int, cargos[anchor]['key'].flat))
    _require(len(route_colors) == 4 and set(dots) == route_colors, "four matched waypoint networks")
    _require(len({bg, carrier} | route_colors | key_colors) == 10, "disjoint semantic palette")
    r, c, n = dock
    _require(sum(len(p['offsets']) for p in cargos) == n*n, "cargo area equals dock area")
    for cargo in cargos:
        cargo['dots'] = dots[cargo['color']]
        _require(1 <= len(cargo['dots']) <= 8, 'bounded waypoint networks')
    return dict(grid=grid, bg=bg, carrier=carrier, frame=frame, dock=dock,
                cargos=cargos, anchor=anchor)


def _pose(cargo, tip, facing):
    if 'routing_offsets' in cargo:
        return frozenset((tip[0]+r, tip[1]+c) for r, c in cargo['routing_offsets'][facing])
    return frozenset((tip[0]+r, tip[1]+c)
                     for r, c in (_rot(p, facing-cargo['facing']) for p in cargo['offsets']))


def _corner(cells, dock):
    r, c, n = dock
    corners = ((r, c), (r, c+n-1), (r+n-1, c), (r+n-1, c+n-1))
    hits = [i for i, p in enumerate(corners) if p in cells]
    return hits[0] if len(hits) == 1 else None


def _inside(cells, dock):
    r, c, n = dock
    return all(r <= a < r+n and c <= b < c+n for a, b in cells)


def _legal_pose(cells, shape, frame):
    return all(0 <= r < shape[0] and 0 <= c < shape[1] for r, c in cells) and not cells.intersection(frame)


def _halo(cells):
    return {(r+dr, c+dc) for r, c in cells
            for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _prepare_routing(cargo, shape, frame):
    """Constructor-local exact geometry; no seeds, examples or solver results.

    A table entry is true exactly when the translated rotated footprint is
    in bounds and misses the frame. Shifted array slices compute all tip
    positions together instead of rebuilding the footprint for every trial.
    """
    h, w = shape
    blocked = np.zeros(shape, dtype=bool)
    for r, c in frame:
        blocked[r, c] = True
    offsets, bounds, halos, tables = [], [], [], []
    for facing in range(4):
        rotated = tuple(_rot(p, facing-cargo['facing']) for p in cargo['offsets'])
        r0, r1 = min(r for r, c in rotated), max(r for r, c in rotated)
        c0, c1 = min(c for r, c in rotated), max(c for r, c in rotated)
        valid = np.zeros(shape, dtype=bool)
        lo_r, hi_r = max(0, -r0), min(h, h-r1)
        lo_c, hi_c = max(0, -c0), min(w, w-c1)
        if lo_r < hi_r and lo_c < hi_c:
            region = valid[lo_r:hi_r, lo_c:hi_c]
            region[:] = True
            for dr, dc in rotated:
                region &= ~blocked[lo_r+dr:hi_r+dr, lo_c+dc:hi_c+dc]
        offsets.append(rotated)
        bounds.append((r0, c0, r1, c1))
        halos.append(_halo(rotated))
        tables.append(valid)
    cargo['routing_offsets'] = offsets
    cargo['routing_bounds'] = bounds
    cargo['routing_halos'] = halos
    cargo['routing_legal'] = tables


def _routing_legal(cargo, tip, facing):
    table = cargo['routing_legal'][facing]
    r, c = tip
    return 0 <= r < table.shape[0] and 0 <= c < table.shape[1] and bool(table[r, c])


def _routing_inside(cargo, tip, facing, dock):
    r, c, n = dock
    r0, c0, r1, c1 = cargo['routing_bounds'][facing]
    return r <= tip[0]+r0 and tip[0]+r1 < r+n and c <= tip[1]+c0 and tip[1]+c1 < c+n


def _routing_near_dots(cargo, tip, facing, dots):
    halo = cargo['routing_halos'][facing]
    return any((r-tip[0], c-tip[1]) in halo for r, c in dots)


def _arrivals(cargo, analysis):
    """Enumerate local hypotheses; docking, not exhaustion, stops a route."""
    answers, visited = {}, set()
    dock, frame, shape = analysis['dock'], analysis['frame'], analysis['grid'].shape

    def visit(tip, facing, remaining, route, initial=False):
        state = (tip, facing, frozenset(remaining))
        if state in visited:
            return
        visited.add(state)
        cells = _pose(cargo, tip, facing)
        if not _legal_pose(cells, shape, frame):
            return
        if _inside(cells, dock):
            if initial:
                return
            corner = _corner(cells, dock)
            if corner is not None:
                turns = (facing-cargo['facing']) % 4
                key = None if cargo['key'] is None else np.rot90(cargo['key'], -turns).copy()
                sig = (cells, None if key is None else tuple(map(int, key.flat)))
                answers.setdefault(sig, dict(cells=cells, corner=corner, key=key,
                                             tip=tip, facing=facing, turns=turns, route=route))
            return
        rays = {}
        for q in remaining:
            if (q[0] == tip[0]) != (q[1] == tip[1]):
                d = _direction(tip, q)
                distance = abs(q[0]-tip[0]) + abs(q[1]-tip[1])
                if d not in rays or distance < rays[d][0]:
                    rays[d] = (distance, q)
        for d in sorted(rays):
            if initial and d != facing:
                continue
            q = rays[d][1]
            visit(q, d, remaining-{q}, route+(q,))

    visit(cargo['tip'], cargo['facing'], set(cargo['dots']), (cargo['tip'],), True)
    return list(answers.values())


def _assemble(analysis, arrivals, limit=2):
    """Joint exact cover. Route history duplicates have already been collapsed."""
    r, c, n = analysis['dock']
    full = frozenset((a, b) for a in range(r, r+n) for b in range(c, c+n))
    order = sorted(range(4), key=lambda i: len(arrivals[i]))
    solutions, chosen = [], [None]*4

    def search(depth, covered, corners):
        if len(solutions) >= limit:
            return
        if depth == 4:
            if covered == full:
                solutions.append(chosen.copy())
            return
        i = order[depth]
        for pose in arrivals[i]:
            if pose['corner'] in corners or covered.intersection(pose['cells']):
                continue
            chosen[i] = pose
            search(depth+1, covered | pose['cells'], corners | {pose['corner']})

    search(0, frozenset(), set())
    return solutions


def _corner_choices(arrivals, limit=2):
    """Count a weaker hypothesis that ignores every footprint except corners."""
    answers = []
    def visit(i, chosen, corners):
        if len(answers) >= limit:
            return
        if i == 4:
            answers.append(chosen)
            return
        for pose in arrivals[i]:
            if pose['corner'] not in corners:
                visit(i+1, chosen+[pose], corners | {pose['corner']})
    visit(0, [], set())
    return answers


def _analyze(input_grid):
    a = _parse(input_grid)
    candidates = [_arrivals(p, a) for p in a['cargos']]
    _require(all(candidates), "every cargo must have a possible arrival")
    solutions = _assemble(a, candidates)
    _require(len(solutions) == 1, "joint dock assembly must be unique")
    a['candidates'], a['chosen'] = candidates, solutions[0]
    return a


def _paint_answer(a):
    r, c, n = a['dock']
    key = a['chosen'][a['anchor']]['key'].reshape(-1)
    out = np.full((n, n), -1, dtype=np.int8)
    for pose in a['chosen']:
        for x, y in pose['cells']:
            out[x-r, y-c] = key[pose['corner']]
    _require(bool(np.all(out >= 0)), "complete square required")
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer solely from the visible input pixels."""
    return _paint_answer(_analyze(input_grid))


def _marker_locations(cells, d, dock):
    r, c, n = dock
    offsets = [_rot(p, d-1) for p in T]
    return [p for p in sorted(cells) if r+1 <= p[0] < r+n-1 and c+1 <= p[1] < c+n-1
            and {(p[0]+a, p[1]+b) for a, b in offsets} <= cells]


def _partition(rng, n):
    """Independent stepped seams, plus connected local cell exchanges."""
    h = int(rng.integers(3, n-2))
    owner = np.empty((n, n), dtype=np.int8)
    for lo, hi, labels in ((0, h, (0, 1)), (h, n, (2, 3))):
        cut = int(rng.integers(3, n-2))
        for row in range(lo, hi):
            if row > lo and rng.random() < 0.65:
                cut = int(np.clip(cut + rng.choice([-1, 1]), 3, n-3))
            owner[row, :cut], owner[row, cut:] = labels
    for _ in range(int(rng.integers(0, n+1))):
        row, col = int(rng.integers(1, n-1)), int(rng.integers(1, n-1))
        old = int(owner[row, col])
        dr, dc = DIRS[int(rng.integers(4))]
        new = int(owner[row+dr, col+dc])
        owner[row, col] = new
        if any(len(_components(map(tuple, np.argwhere(owner == k)), False)) != 1 for k in {old, new}):
            owner[row, col] = old
    return np.rot90(owner, int(rng.integers(4))).copy()


def _make_geometry(rng, n, dock, two_forks=False):
    owner = _partition(rng, n)
    r, c, _ = dock
    shapes = [{(int(x)+r, int(y)+c) for x, y in np.argwhere(owner == k)} for k in range(4)]
    corners = [_corner(s, dock) for s in shapes]
    if any(k is None for k in corners):
        return None
    choices, branches = [], []
    for i, cells in enumerate(shapes):
        corner = corners[i]
        dirs = (2 if corner < 2 else 0, 1 if corner % 2 == 0 else 3)
        local = []
        for d in dirs:
            tips = _marker_locations(cells, d, dock)
            if not tips:
                continue
            minr, maxr = min(x for x, y in cells), max(x for x, y in cells)
            minc, maxc = min(y for x, y in cells), max(y for x, y in cells)
            delta = (0, (n-(maxc-minc+1))*(1 if corner % 2 == 0 else -1)) if d % 2 == 0 else ((n-(maxr-minr+1))*(1 if corner < 2 else -1), 0)
            shifted = {(x+delta[0], y+delta[1]) for x, y in cells}
            branch = max(abs(delta[0]), abs(delta[1])) >= 2 and _inside(shifted, dock) and _corner(shifted, dock) is not None
            local.append((d, tips, delta if branch else None))
            if branch:
                branches.append((i, d, tips, delta))
        if not local:
            return None
        choices.append(local)
    if not branches:
        return None
    pairs = []
    for a, b in itertools.combinations(branches, 2):
        if a[0] == b[0] or a[1] != b[1]:
            continue
        ca = _corner({(x+a[3][0], y+a[3][1]) for x, y in shapes[a[0]]}, dock)
        cb = _corner({(x+b[3][0], y+b[3][1]) for x, y in shapes[b[0]]}, dock)
        if ca == corners[b[0]] and cb == corners[a[0]]:
            pairs.append((a, b))
    if two_forks:
        if not pairs:
            return None
        selected = pairs[int(rng.integers(len(pairs)))]
    else:
        selected = (branches[int(rng.integers(len(branches)))],)
    return owner, shapes, corners, choices, {p[0]: p for p in selected}


def _build_route(rng, cargo, target_tip, d, analysis, prefix_count, branch_delta=None):
    """Grow a reverse exterior walk, then independently check its visible network."""
    cells = _pose(cargo, target_tip, d)
    r, c, n = analysis['dock']
    if d == 2:
        distance = max(x for x, y in cells) - (r-3)
    elif d == 0:
        distance = (r+n+2) - min(x for x, y in cells)
    elif d == 1:
        distance = max(y for x, y in cells) - (c-3)
    else:
        distance = (c+n+2) - min(y for x, y in cells)
    q = _shift(target_tip, d, -(distance + int(rng.integers(0, 2))))
    if not _routing_legal(cargo, q, d):
        return None
    reverse = [target_tip, q]
    extra = []
    if branch_delta is not None:
        e = (q[0]+branch_delta[0], q[1]+branch_delta[1])
        u = (target_tip[0]+branch_delta[0], target_tip[1]+branch_delta[1])
        extra = [e, u]
        if not _routing_legal(cargo, e, _direction(q, e)):
            return None
    for step in range(prefix_count):
        options = []
        for nd in range(4):
            if step == 0 and nd % 2 == d % 2 and (branch_delta is not None or rng.random() < 0.7):
                continue  # an explicit turn before docking
            for length in range(3, 12):
                p = _shift(reverse[-1], nd, length)
                facing = _direction(p, reverse[-1])
                if not _routing_legal(cargo, p, facing) or _routing_inside(cargo, p, facing, analysis['dock']):
                    continue
                if not _routing_legal(cargo, reverse[-1], facing):
                    continue
                if any(max(abs(p[0]-x), abs(p[1]-y)) < 2 for x, y in reverse + extra):
                    continue
                # The desired first leg may not encounter another dot nearer.
                if any((x == p[0] or y == p[1]) and (x, y) != reverse[-1]
                       and _direction(p, (x, y)) == facing
                       and abs(x-p[0])+abs(y-p[1]) < length for x, y in reverse[:-1]+extra):
                    continue
                if step == 0 and branch_delta is not None:
                    # Approach the fork from the opposite side of its branch.
                    if (p[0]-q[0])*branch_delta[0] + (p[1]-q[1])*branch_delta[1] >= 0:
                        continue
                if _routing_near_dots(cargo, p, facing, reverse+extra):
                    continue
                options.append((p, facing))
        if not options:
            return None
        p, facing = options[int(rng.integers(len(options)))]
        reverse.append(p)
    route = list(reversed(reverse))
    start, facing = route[0], _direction(route[0], route[1])
    dots = set(route[1:] + extra)
    if _routing_near_dots(cargo, start, facing, dots):
        return None
    return start, facing, dots


def _make_pair(rng, n, prefix_count, force_anchor_turn=False, need_geometry=False):
    # The dock is offset independently; rendering is cropped/padded afterwards.
    dock = (int(rng.integers(8, 11)), int(rng.integers(8, 11)), n)
    r, c, _ = dock
    frame = {(x, y) for x in range(r-1, r+n+1) for y in range(c-1, c+n+1)
             if x in (r-1, r+n) or y in (c-1, c+n)}
    base = dict(dock=dock, frame=frame, grid=np.zeros((30, 30), dtype=np.int8))
    two_forks = need_geometry or rng.random() < 0.6
    geometry = _make_geometry(rng, n, dock, two_forks)
    if geometry is None:
        return None
    owner, shapes, corners, choices, branches = geometry
    anchor = int(rng.integers(4))
    cargos, markers, keys = [], [], []
    for i, cells in enumerate(shapes):
        local = branches[i][1:] if i in branches else choices[i][int(rng.integers(len(choices[i])))]
        d, tips, _ = local
        viable = []
        for tip in tips:
            arrow_cells = {(tip[0]+x, tip[1]+y) for x, y in (_rot(p, d-1) for p in T)}
            blocks = [(x, y) for x, y in sorted(cells)
                      if {(x, y), (x+1, y), (x, y+1), (x+1, y+1)} <= cells-arrow_cells]
            if i != anchor or blocks:
                viable.append((tip, arrow_cells, blocks))
        if not viable:
            return None
        tip, arrow_cells, blocks = viable[int(rng.integers(len(viable)))]
        cargo = dict(tip=tip, facing=d, offsets=tuple((x-tip[0], y-tip[1]) for x, y in sorted(cells)))
        _prepare_routing(cargo, (30, 30), frame)
        cargos.append(cargo)
        markers.append(arrow_cells)
        keys.append(blocks[int(rng.integers(len(blocks)))] if i == anchor else None)
    routes = []
    occupied = set(frame)
    forbidden = _halo(occupied)
    for i, cargo in enumerate(cargos):
        for attempt in range(35):
            route = _build_route(rng, cargo, cargo['tip'], cargo['facing'], base,
                                 prefix_count, branches[i][3] if i in branches else None)
            if route is None:
                continue
            if force_anchor_turn and i == anchor and route[1] == cargo['facing']:
                continue
            start, facing, dots = route
            group = set(_pose(cargo, start, facing)) | dots
            if group.intersection(forbidden):
                continue
            occupied |= group
            forbidden |= _halo(group)
            break
        else:
            return None
        routes.append(route)
    palette = [int(v) for v in rng.permutation(10)]
    bg, carrier = palette[:2]
    inks, key = palette[2:6], np.array(palette[6:10], dtype=np.int8).reshape(2, 2)
    grid = np.full((30, 30), bg, dtype=np.int8)
    all_groups = [frame]
    paintings = [(frame, carrier)]
    for i, cargo in enumerate(cargos):
        start, facing, dots = routes[i]
        cells = _pose(cargo, start, facing)
        all_groups.append(cells)
        all_groups.extend({q} for q in sorted(dots))
        paintings.append((cells, carrier))
        paintings.append((dots, inks[i]))
        turns = facing-cargo['facing']
        for x, y in markers[i]:
            a, b = _rot((x-cargo['tip'][0], y-cargo['tip'][1]), turns)
            paintings.append(({(start[0]+a, start[1]+b)}, inks[i]))
        if i == anchor:
            x, y = keys[i]
            for a in range(2):
                for b in range(2):
                    dr, dc = _rot((x+a-cargo['tip'][0], y+b-cargo['tip'][1]), turns)
                    paintings.append(({(start[0]+dr, start[1]+dc)}, int(key[a, b])))
    separation = set()
    for group in all_groups:
        if group.intersection(separation):
            return None
        separation |= _halo(group)
    for cells, color in paintings:
        for point in cells:
            grid[point] = color
    # Keep some varied margins. Frame and cargo bounding boxes are never cropped.
    points = set().union(*all_groups)
    lo = [min(p[a] for p in points) for a in (0, 1)]
    hi = [max(p[a] for p in points)+1 for a in (0, 1)]
    lo = [max(0, v-int(rng.integers(1, 4))) for v in lo]
    hi = [min(30, v+int(rng.integers(1, 4))) for v in hi]
    grid = grid[lo[0]:hi[0], lo[1]:hi[1]].copy()
    try:
        a = _analyze(grid)
    except AssertionError:
        return None
    if max(map(len, a['candidates'])) < 2:
        return None
    if two_forks and len(_corner_choices(a['candidates'])) < 2:
        return None
    # Ground truth is the intended partition; the input decoder must recover it.
    out = key.reshape(-1)[np.array(corners, dtype=int)[owner]].astype(np.int8)
    if not np.array_equal(out, _paint_answer(a)):
        return None
    if prefix_count >= 2 and min(len(p['route'])-1 for p in a['chosen']) <= 2:
        return None
    return {'input': grid, 'output': out}


def _valid_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not _valid_int(num_train) or not 2 <= num_train <= 10:
        raise ValueError('num_train must be an integer in 2..10')
    if not _valid_int(num_test) or not 1 <= num_test <= 4:
        raise ValueError('num_test must be an integer in 1..4')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def _fingerprint(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    if not _valid_int(seed) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            prefix = 1 if split == 'train' and i < 2 else (2 if split == 'train' else int(rng.integers(2, 4)))
            for attempt in range(4000):
                # The extra-long setting is an emphasis, not a hidden rule.
                # Relax it after a bounded number of failed geometric samples.
                length = 2 if split == 'test' and attempt >= 24 else prefix
                pair = _make_pair(rng, int(rng.integers(7, 11)), length,
                                  split == 'train' and i == 0, split == 'train' and i == 0)
                if pair is not None and _fingerprint(pair['input']) not in seen:
                    seen.add(_fingerprint(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('bounded route construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'split lists required')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, info = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            fp = _fingerprint(inp)
            _require(fp not in seen, 'distinct inputs required')
            seen.add(fp)
            a = _analyze(inp)
            _require(np.array_equal(out, _paint_answer(a)), 'incorrect transported-key assembly')
            _require(max(map(len, a['candidates'])) >= 2, 'assembly must resolve route ambiguity')
            key = a['chosen'][a['anchor']]['key']
            _require(set(map(int, out.flat)) == set(map(int, key.flat)), 'all four key colors used')
            _require(out.size == sum(len(p['offsets']) for p in a['cargos']), 'area preserved')
            # The ambiguity is functional: changing one local arrival destroys
            # the cover supplied by the other three chosen arrivals.
            for j, candidates in enumerate(a['candidates']):
                for other in candidates:
                    if other is a['chosen'][j]:
                        continue
                    counterfactual = [[p] for p in a['chosen']]
                    counterfactual[j] = [other]
                    _require(not _assemble(a, counterfactual), 'alternative arrival must change feasibility')
            info[split].append(a)
    _require(any(a['chosen'][a['anchor']]['turns'] != 0 for a in info['train']), 'train demonstrates key rotation')
    _require(len(_corner_choices(info['train'][0]['candidates'])) >= 2, 'first demonstration must require footprint geometry, not only corners')
    short = [max(len(p['route'])-1 for p in a['chosen']) for a in info['train'][:2]]
    _require(all(min(len(p['route'])-1 for p in a['chosen']) > max(short) for a in info['test']), 'test routes extend demonstrated short routes')


def _greedy_forward(cargo, analysis):
    """A deliberately incorrect ablation: prefer continuing straight at forks."""
    tip, facing, remaining = cargo['tip'], cargo['facing'], set(cargo['dots'])
    first = True
    while True:
        cells = _pose(cargo, tip, facing)
        if not _legal_pose(cells, analysis['grid'].shape, analysis['frame']):
            return None
        if _inside(cells, analysis['dock']):
            return cells
        rays = {}
        for q in remaining:
            if (q[0] == tip[0]) != (q[1] == tip[1]):
                d = _direction(tip, q)
                distance = abs(q[0]-tip[0])+abs(q[1]-tip[1])
                if d not in rays or distance < rays[d][0]:
                    rays[d] = (distance, q)
        if not rays or (first and facing not in rays):
            return None
        d = facing if facing in rays else min(rays, key=lambda k: ((k-facing) % 4, k))
        tip, facing, first = rays[d][1], d, False
        remaining.remove(tip)


def self_test():
    """A fixed 42-seed sweep, ablations, equivariance, and API boundary checks."""
    rows, seen, palette_roles, profiles, route_profiles = [], set(), [set() for _ in range(3)], set(), set()
    forward_failures = 0
    geometry_ambiguities = 0
    fork_counts = set()
    samples = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed < 3:
            samples.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = _analyze(pair['input'])
                rows.append((split, a))
                signature = _fingerprint(pair['input'])
                _require(signature not in seen, 'cross-seed duplicate input')
                seen.add(signature)
                palette_roles[0].add(a['bg'])
                palette_roles[1].add(a['carrier'])
                palette_roles[2].update(map(int, a['cargos'][a['anchor']]['key'].flat))
                profiles.add(tuple(sorted(len(p['offsets']) for p in a['cargos'])))
                geometry_ambiguities += len(_corner_choices(a['candidates'])) > 1
                fork_counts.add(sum(len(c) > 1 for c in a['candidates']))
                for p in a['chosen']:
                    points = p['route']
                    directions = tuple(_direction(x, y) for x, y in zip(points, points[1:]))
                    distances = tuple(abs(x[0]-y[0])+abs(x[1]-y[1]) for x, y in zip(points, points[1:]))
                    route_profiles.add((directions, distances))
                failed = any(_greedy_forward(p, a) != q['cells'] for p, q in zip(a['cargos'], a['chosen']))
                forward_failures += failed
                if split == 'train' and pair is puzzle['train'][0]:
                    _require(failed, 'first example must defeat forward priority')
                # Ignoring routes admits global rotations of the full packing.
                # All four distinct corner colors make each rotated answer differ.
                _require(not np.array_equal(pair['output'], np.rot90(pair['output'])), 'free packing must have an alternative answer')
                if a['chosen'][a['anchor']]['turns']:
                    wrong = np.empty_like(pair['output'])
                    r, c, n = a['dock']
                    key = a['cargos'][a['anchor']]['key'].reshape(-1)
                    for p in a['chosen']:
                        for x, y in p['cells']:
                            wrong[x-r, y-c] = key[p['corner']]
                    _require(not np.array_equal(wrong, pair['output']), 'unrotated key shortcut must fail')
    _require(len({a['dock'][2] for s, a in rows}) == 4, 'dock sizes vary')
    _require(len({a['grid'].shape for s, a in rows}) >= 12, 'canvas geometry varies')
    _require(len(profiles) >= 30 and len(route_profiles) >= 100, 'structural variation missing')
    _require(all(len(roles) == 10 for roles in palette_roles), 'palette roles must vary')
    _require({p['corner'] for s, a in rows for p in [a['chosen'][a['anchor']]]} == set(range(4)), 'mobile legend must reach all corners')
    _require({p['turns'] for s, a in rows for p in a['chosen']} == set(range(4)), 'net rotation spectrum missing')
    _require(forward_failures >= 42, 'route forks do not defeat the local shortcut')
    _require(geometry_ambiguities >= 42 and len(fork_counts) >= 2, 'joint footprint constraints or fork-count variation missing')
    # Whole-scene rotations and palette permutations provide input-only checks
    # for family members outside the constructor's immediate presentation.
    for puzzle in samples:
        for pair in puzzle['train'] + puzzle['test']:
            before = pair['input'].copy()
            for k in (1, 2, 3):
                _require(np.array_equal(solve(np.rot90(before, k)), np.rot90(pair['output'], k)), 'rotation equivariance')
            mapping = np.arange(9, -1, -1, dtype=np.int8)
            _require(np.array_equal(solve(mapping[before]), mapping[pair['output']]), 'palette equivariance')
            _require(np.array_equal(before, pair['input']), 'solve must not mutate input')
    first = generate_puzzle(12345, 2, 1)
    again = generate_puzzle(12345, 2, 1)
    _require(all(_fingerprint(p[k]) == _fingerprint(q[k]) for s in first for p, q in zip(first[s], again[s]) for k in p), 'determinism')
    for tr, te in ((2, 1), (10, 4)):
        p = generate_puzzle(2718+tr, tr, te)
        _require(len(p['train']) == tr and len(p['test']) == te, 'explicit counts')
    for tr, te in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5), (True, 1), (4, False), (2.0, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    damaged = {s: [{k: v.copy() for k, v in p.items()} for p in first[s]] for s in first}
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return dict(seeds=42, pairs=len(rows), canvas_shapes=len({a['grid'].shape for s, a in rows}),
                area_profiles=len(profiles), route_profiles=len(route_profiles), forward_shortcut_failures=forward_failures,
                corner_only_ambiguities=geometry_ambiguities, ambiguous_cargo_counts=sorted(fork_counts),
                square_sides=sorted({a['dock'][2] for s, a in rows}),
                anchor_corners=sorted({a['chosen'][a['anchor']]['corner'] for s, a in rows}),
                net_rotations=sorted({p['turns'] for s, a in rows for p in a['chosen']}))


if __name__ == '__main__':
    self_test()
