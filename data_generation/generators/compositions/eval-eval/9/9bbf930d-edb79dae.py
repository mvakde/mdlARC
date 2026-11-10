"""Rail-addressed stamps: routing is the legend and carries its orientation.

Only NumPy and the standard library are used. All semantic decoding in solve is
input-only. Construction computes labels from its sampled objects; validation
then reconstructs those objects from the actual pixels.
"""
from collections import deque
import numpy as np


_DIRS = ((0, 1), (1, 0), (0, -1), (-1, 0))  # clockwise from east


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "colors must be 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _components(mask):
    remaining = set(zip(*np.nonzero(mask)))
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        pending, cells = deque([p]), []
        while pending:
            p = pending.popleft()
            cells.append((int(p[0]), int(p[1])))
            for d in _DIRS:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    pending.append(q)
        result.append(cells)
    return result


def _normalize(inp):
    candidates = []
    for rotation in range(4):
        g = np.rot90(inp, rotation)
        strip = int(g[0, 0])
        if (g.shape[1] >= 3 and np.all(g[:, 0] == strip)
                and np.count_nonzero(g == strip) == g.shape[0]):
            bg = int(g[0, 1])
            if bg != strip and np.all(g[:, 1] == bg):
                candidates.append((g.copy(), rotation, strip, bg))
    _require(len(candidates) == 1, "one exclusive edge strip with a clear approach is required")
    return candidates[0]


def _tile_frames(g, strip, bg):
    candidates = []
    for color in map(int, np.unique(g)):
        if color in (strip, bg):
            continue
        boxes, sizes = [], set()
        valid = True
        for cells in _components(g == color):
            rr, cc = zip(*cells)
            r, c = min(rr), min(cc)
            h, w = max(rr) - r + 1, max(cc) - c + 1
            if not (h == w and 4 <= h <= 7 and len(cells) == 4 * h - 4
                    and all(a in (r, r+h-1) or b in (c, c+h-1) for a, b in cells)):
                valid = False
                break
            boxes.append((r, c))
            sizes.add(h)
        if valid and len(boxes) >= 3 and len(sizes) == 1:
            candidates.append((next(iter(sizes)), color, sorted(boxes)))
    _require(candidates, "missing equal-sized square tile frames")
    # Payload rings can form smaller nested square borders. The outermost,
    # largest common border is the tile role, rather than a payload feature.
    largest = max(x[0] for x in candidates)
    candidates = [x for x in candidates if x[0] == largest]
    _require(len(candidates) == 1, "ambiguous outer tile frame role")
    side, frame, boxes = candidates[0]
    tiles, border_index = [], {}
    for i, (r, c) in enumerate(boxes):
        payload = g[r+1:r+side-1, c+1:c+side-1]
        colors = set(map(int, np.unique(payload))) - {bg}
        _require(len(colors) == 1 and not colors & {frame, strip}, "tile needs one payload color")
        color = next(iter(colors))
        tiles.append(dict(top=r, left=c, color=color, mask=(payload == color).copy()))
        for a in range(r, r+side):
            for b in range(c, c+side):
                if a in (r, r+side-1) or b in (c, c+side-1):
                    border_index[a, b] = i
    return frame, side, tiles, border_index


def _paired(g, p, d, forbidden):
    normal = (-d[1], d[0])
    a, b = _add(p, normal), _add(p, (-normal[0], -normal[1]))
    if not (_inside(g, a) and _inside(g, b)):
        return None
    x, y = int(g[a]), int(g[b])
    return x if x == y and x not in forbidden else None


def _front_cell(tile, side, d):
    r, c, mid = tile['top'], tile['left'], side // 2
    return {(0, 1): (r+mid, c), (0, -1): (r+mid, c+side-1),
            (1, 0): (r, c+mid), (-1, 0): (r+side-1, c+mid)}[d]


def _trace(g, row, bg, strip, frame, side, tiles, borders):
    p, d = (row, 1), (0, 1)
    path, seen, turns = [p], set(), []
    forbidden = {bg, strip, frame}
    while True:
        _require((p, d) not in seen, "route cycle")
        seen.add((p, d))
        q = _add(p, d)
        _require(_inside(g, q), "a selected guide must terminate at a tile")
        if int(g[q]) == frame:
            _require(q in borders, "unknown border cell")
            index = borders[q]
            _require(q == _front_cell(tiles[index], side, d), "route must enter a tile at the side midpoint")
            _require(len(path) == len(set(path)), "route revisits a cell")
            return dict(path=path, terminal=index, heading=d, turns=turns)
        if int(g[q]) == bg:
            p = q
            path.append(p)
            continue
        choices = []
        for e in ((-d[1], d[0]), (d[1], -d[0])):
            q = _add(p, e)
            if (_inside(g, q) and int(g[q]) == bg
                    and _paired(g, q, e, forbidden) is not None):
                choices.append(e)
        _require(len(choices) == 1, "blocked route needs exactly one visibly paired turn")
        turns.append((p, d, choices[0]))
        d = choices[0]


def _analyze(inp):
    g, rotation, strip, bg = _normalize(_grid(inp))
    frame, side, tiles, borders = _tile_frames(g, strip, bg)
    forbidden = {strip, bg, frame}
    routes = []
    for row in range(1, g.shape[0]-1):
        for col in range(1, g.shape[1]):
            if int(g[row, col]) != bg:
                break
            x, y = int(g[row-1, col]), int(g[row+1, col])
            if x not in forbidden and y not in forbidden:
                route = _trace(g, row, bg, strip, frame, side, tiles, borders)
                route.update(row=row, entrance=col, source=x, other=y, active=x == y)
                routes.append(route)
                break  # first two-sided opening decides, including a mismatch
    _require(routes and any(x['active'] for x in routes), "no active rail definitions")
    _require(len({r['terminal'] for r in routes}) == len(routes), "terminal tile shared by routes")
    paths = [p for r in routes for p in r['path']]
    _require(len(paths) == len(set(paths)), "routes must be disjoint")
    terminal_tiles = {r['terminal'] for r in routes}
    requests = [i for i in range(len(tiles)) if i not in terminal_tiles]
    _require(requests and all(tiles[i]['mask'].all() for i in requests), "unrouted tiles must be solid requests")
    mapping = {}
    for route in routes:
        if route['active']:
            source = route['source']
            _require(source not in mapping, "a source has two active definitions")
            mapping[source] = route
    request_colors = {tiles[i]['color'] for i in requests}
    _require(set(mapping) <= request_colors, "active definition has no request")
    return dict(grid=g, rotation=rotation, strip=strip, background=bg, frame=frame,
                side=side, n=side-2, tiles=tiles, requests=requests, routes=routes, mapping=mapping)


def _gap_count(a, route):
    """Count actual paired gaps with straight supporting evidence on both sides."""
    line = route['path']
    directions = [(b[0]-p[0], b[1]-p[1]) for p, b in zip(line, line[1:])]
    forbidden = {a['strip'], a['background'], a['frame']}
    count = 0
    for i in range(2, len(line)-2):
        if len(set(directions[i-2:i+2])) != 1:
            continue
        d = directions[i]
        count += (_paired(a['grid'], line[i], d, forbidden) is None
                  and _paired(a['grid'], line[i-1], d, forbidden) is not None
                  and _paired(a['grid'], line[i+1], d, forbidden) is not None)
    return count


def _render(a, *, ignore_matching=False, ignore_stencil=False, ignore_rotation=False,
            keep_source_color=False, remap_again=False, ignore_bent_routes=False):
    out, bg, n = a['grid'].copy(), a['background'], a['n']
    definitions = dict(a['mapping'])
    if ignore_matching:
        for r in a['routes']:
            definitions.setdefault(r['source'], r)
    for i in a['requests']:
        request = a['tiles'][i]
        source = request['color']
        if source not in definitions:
            continue
        route = definitions[source]
        if ignore_bent_routes and route['turns']:
            continue
        stencil = a['tiles'][route['terminal']]
        mask = stencil['mask']
        if not ignore_rotation:
            mask = np.rot90(mask, _DIRS.index(route['heading']))
        if ignore_stencil:
            mask = np.ones((n, n), dtype=bool)
        paint = source if keep_source_color else stencil['color']
        if remap_again and paint in definitions:
            paint = a['tiles'][definitions[paint]['terminal']]['color']
        r, c = request['top']+1, request['left']+1
        out[r:r+n, c:c+n] = np.where(mask, paint, bg)
    return np.rot90(out, -a['rotation']).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the launch side, matching rails, terminal patterns and requests."""
    return _render(_analyze(input_grid))


def _polyline(points):
    path = [points[0]]
    for q in points[1:]:
        p = path[-1]
        dr, dc = q[0]-p[0], q[1]-p[1]
        if (dr == 0) == (dc == 0):
            return None
        d = (int(np.sign(dr)), int(np.sign(dc)))
        while path[-1] != q:
            path.append(_add(path[-1], d))
    cells = set(path)
    if len(cells) != len(path):
        return None
    for i, p in enumerate(path):
        if sum(_add(p, d) in cells for d in _DIRS) != (1 if i in (0, len(path)-1) else 2):
            return None
    return path


def _path_and_plate(rng, h, w, side, turn_signs):
    p = (int(rng.integers(1, h-1)), 1)
    points, heading = [p], 0
    for j in range(len(turn_signs)+1):
        choices = []
        for length in range(4 if j == 0 else 3, max(h, w)):
            q = (p[0]+length*_DIRS[heading][0], p[1]+length*_DIRS[heading][1])
            if not (1 <= q[0] < h-1 and 4 <= q[1] < w-1):
                continue
            if j == len(turn_signs):
                mid = side // 2
                r, c = {0: (q[0]-mid, q[1]+1), 1: (q[0]+1, q[1]-mid),
                        2: (q[0]-mid, q[1]-side), 3: (q[0]-side, q[1]-mid)}[heading]
                if not (0 <= r <= h-side and 3 <= c <= w-side):
                    continue
            choices.append(q)
        if not choices:
            return None
        # Short and long segments both occur; fitting restrictions are explicit.
        distances = np.array([abs(q[0]-p[0])+abs(q[1]-p[1]) for q in choices], float)
        weights = 1 / np.sqrt(distances)
        p = choices[int(rng.choice(len(choices), p=weights/weights.sum()))]
        points.append(p)
        if j < len(turn_signs):
            heading = (heading + turn_signs[j]) % 4
    path = _polyline(points)
    if path is None:
        return None
    mid = side // 2
    r, c = {0: (p[0]-mid, p[1]+1), 1: (p[0]+1, p[1]-mid),
            2: (p[0]-mid, p[1]-side), 3: (p[0]-side, p[1]-mid)}[heading]
    area = {(a, b) for a in range(r, r+side) for b in range(c, c+side)}
    if set(path) & area:
        return None
    return path, (r, c), heading


def _frame(g, top, left, side, color):
    g[top, left:left+side] = g[top+side-1, left:left+side] = color
    g[top:top+side, left] = g[top:top+side, left+side-1] = color


def _mask(rng, n, informative=False):
    if informative:
        for _ in range(100):
            m = rng.random((n, n)) < rng.uniform(.25, .7)
            if m.any() and not m.all() and all(not np.array_equal(m, np.rot90(m, k)) for k in (1, 2, 3)):
                return m
        m = np.zeros((n, n), bool)
        m[0, :] = True
        m[-1, 0] = m[-2, 1] = True
        return m
    y, x = np.indices((n, n))
    kind = int(rng.integers(9))
    if kind == 0:
        m = (x == 0) | (y == 0) | (x == n-1) | (y == n-1)
    elif kind == 1:
        m = (x == n//2) | (y == n//2)
    elif kind == 2:
        m = (x == y)
    elif kind == 3:
        m = (x+y) % 2 == 0
    elif kind == 4:
        m = np.zeros((n, n), bool)
        m[int(rng.integers(n)), int(rng.integers(n))] = True
    elif kind == 5:
        m = (y % 2 == 0)
    elif kind == 6:
        m = np.ones((n, n), bool)
    else:
        m = rng.random((n, n)) < rng.uniform(.15, .9)
    if not m.any():
        m[int(rng.integers(n)), int(rng.integers(n))] = True
    return np.rot90(m, int(rng.integers(4))).copy()


def _scene(rng, n, active_count, inactive_count, request_count, signs, colors,
           rotation, demonstration=False):
    """Pack complete objects; sampled geometry is never used by solve."""
    side = n+2
    crowded = active_count+inactive_count >= 4 or request_count >= 6
    h = int(rng.integers(28 if n >= 4 else (27 if crowded else 20), 31))
    w = int(rng.integers(28 if n >= 4 else (27 if crowded else 23), 31))
    bg, strip, frame = colors[:3]
    sources = colors[3:3+active_count+inactive_count]
    active = sources[:active_count]
    inactive = sources[active_count:]
    paint_pool = colors[3:]
    paints = [int(rng.choice(paint_pool)) for _ in sources]
    if demonstration:
        # A -> B -> C supplies an observable one-step-versus-recursive contrast.
        paints[0], paints[1] = active[1], inactive[0]
    masks = [_mask(rng, n, informative=demonstration or i == 0) for i in range(len(sources))]
    if not demonstration and len(sources) > 1 and rng.random() < .2:
        masks[-1] = masks[0].copy()
    g = np.full((h, w), bg, dtype=np.int8)
    g[:, 0] = strip
    reserved = set()
    plates, paths = {}, {}
    # Complicated routes get first choice of space, without reserving row bands.
    order = sorted(range(len(sources)), key=lambda i: -len(signs[i]))
    for i in order:
        placed = False
        for _ in range(420):
            proposal = _path_and_plate(rng, h, w, side, signs[i])
            if proposal is None:
                continue
            path, (r, c), heading = proposal
            cells = set(path)
            area = {(a, b) for a in range(r, r+side) for b in range(c, c+side)}
            rim = {(a, b) for a in range(max(0, r-1), min(h, r+side+1))
                   for b in range(max(2, c-1), min(w, c+side+1))}
            if (cells & reserved or area & reserved or any(g[p] != bg for p in cells | area)
                    or any(g[p] == frame for p in rim)):
                continue
            walls = {}
            for p in path:
                for d in _DIRS:
                    q = _add(p, d)
                    if _inside(g, q) and q[1] >= 2 and q not in cells and q not in area:
                        walls[q] = sources[i]
            if i >= active_count:
                # Two distinct colors on the initial strands give a convincing
                # but inactive cable, not a fixed-color inactivity command.
                row = path[0][0]
                first_turn_col = next((a[1] for a, b in zip(path, path[1:]) if a[0] != b[0]), path[-1][1])
                for col in range(2, min(first_turn_col, 5)+1):
                    if (row+1, col) in walls:
                        walls[row+1, col] = active[i % active_count]
            # Omit aligned support only in the middle of straight runs. The
            # bend blockers, entrance evidence and terminal ports remain intact.
            directions = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
            eligible = [j for j in range(4, len(path)-4)
                        if path[j][1] >= 5 and len(set(directions[j-2:j+2])) == 1]
            if demonstration and i == 0 and not eligible:
                continue
            if eligible and (demonstration and i == 0 or rng.random() < .6):
                j = int(rng.choice(eligible))
                d = directions[j]
                normal = (-d[1], d[0])
                for q in (_add(path[j], normal), _add(path[j], (-normal[0], -normal[1]))):
                    walls.pop(q, None)
            if any(q in reserved or (g[q] != bg and g[q] != value) for q, value in walls.items()):
                continue
            for q, value in walls.items():
                g[q] = value
            _frame(g, r, c, side, frame)
            g[r+1:r+side-1, c+1:c+side-1] = np.where(masks[i], paints[i], bg)
            reserved.update(cells | area)
            plates[i] = (r, c, heading)
            paths[i] = path
            placed = True
            break
        if not placed:
            return None
    assignments = list(sources)
    extra = request_count-len(sources)
    if demonstration and extra:
        assignments.append(active[0])
        extra -= 1
    assignments += list(map(int, rng.choice(sources, extra)))
    rng.shuffle(assignments)
    requests = []
    for source in assignments:
        blocked = g != bg
        for q in reserved:
            blocked[q] = True
        padded = np.pad(blocked.astype(int), 1)
        summed = np.pad(padded, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        window = side+2
        totals = (summed[window:, window:] - summed[:-window, window:]
                  - summed[window:, :-window] + summed[:-window, :-window])
        positions = [(int(r), int(c)) for r, c in np.argwhere(totals == 0) if c >= 3]
        if not positions:
            return None
        r, c = positions[int(rng.integers(len(positions)))]
        _frame(g, r, c, side, frame)
        g[r+1:r+side-1, c+1:c+side-1] = source
        reserved.update((a, b) for a in range(r, r+side) for b in range(c, c+side))
        requests.append((r, c, source))
    # Compute ground truth from sampled requests and terminal definitions.
    out = g.copy()
    for r, c, source in requests:
        i = sources.index(source)
        if i < active_count:
            out[r+1:r+side-1, c+1:c+side-1] = np.where(np.rot90(masks[i], plates[i][2]), paints[i], bg)
    # Remove unused far-right space; keep a visible variable outer margin.
    occupied = np.argwhere(g[:, 2:] != bg)
    right = min(w, int(occupied[:, 1].max())+3+int(rng.integers(0, 3)))
    g, out = g[:, :right].copy(), out[:, :right].copy()
    inp, answer = np.rot90(g, rotation).copy(), np.rot90(out, rotation).copy()
    try:
        a = _analyze(inp)
        actual = {r['row']: (r['source'], r['active'], a['tiles'][r['terminal']]['top'],
                             a['tiles'][r['terminal']]['left'], r['heading']) for r in a['routes']}
        intended = {paths[i][0][0]: (sources[i], i < active_count, plates[i][0], plates[i][1], _DIRS[plates[i][2]])
                    for i in range(len(sources))}
        if actual != intended or not np.array_equal(_render(a), answer):
            return None
        if demonstration and _gap_count(a, a['mapping'][active[0]]) == 0:
            return None
        if demonstration and (np.array_equal(_render(a, ignore_matching=True), answer)
                              or np.array_equal(_render(a, ignore_stencil=True), answer)
                              or np.array_equal(_render(a, ignore_rotation=True), answer)
                              or np.array_equal(_render(a, remap_again=True), answer)):
            return None
    except AssertionError:
        return None
    return {'input': inp, 'output': answer}


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def _configuration(rng, shift, is_test, demo_index):
    if demo_index is not None:
        n, na, ni = 3, 2, 1
        nr = 4 if demo_index == 0 else int(rng.integers(3, 5))
        sign = 1 if rng.integers(2) else -1
        turns = [[], [sign], []] if demo_index == 0 else [[sign, sign], [sign, -sign], []]
    else:
        n = int(rng.integers(2, 4 if shift in ('scale', 'repetition', 'turns') else 5))
        if is_test and shift == 'scale':
            n = int(rng.integers(4, 6))
        na = int(rng.integers(1, 3 if n >= 4 else 4))
        ni = int(rng.integers(1, 2 if n >= 4 or na == 3 else 3))
        minimum = na+ni
        nr = int(rng.integers(minimum, max(minimum, 4 if shift == 'repetition' else 6)+1))
        if n >= 4:
            nr = minimum + (int(rng.integers(2)) if n == 4 else 0)
        if is_test and shift == 'turns':
            # Hold out composition depth without also holding out scene density.
            na, ni, nr = 1, 1, int(rng.integers(2, 5))
        if is_test and shift == 'repetition':
            na, ni, nr = 2, 1, int(rng.integers(6, 9))
        max_turns = 4 if is_test and shift == 'turns' else 2
        turns = [list(map(int, rng.choice([-1, 1], int(rng.integers(max_turns+1))))) for _ in range(na)]
        if is_test and shift == 'turns':
            turns[0] = list(map(int, rng.choice([-1, 1], int(rng.integers(3, 5)))))
        turns += [[] for _ in range(ni)]
    return n, na, ni, nr, turns


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(['scale', 'turns', 'repetition', 'launch_side']))
    rotation = int(rng.integers(4))
    palette = list(map(int, rng.permutation(10)))
    fixed_palette = bool(rng.integers(2))
    puzzle, seen = {'train': [], 'test': []}, set()
    first_colors = None
    for split, count in (('train', num_train), ('test', num_test)):
        for index in range(count):
            demo = split == 'train' and index < 2
            is_test = split == 'test'
            local_rotation = rotation if shift == 'launch_side' else int(rng.integers(4))
            if is_test and shift == 'launch_side':
                local_rotation = (rotation+int(rng.integers(1, 4))) % 4
            colors = (palette[:3] + list(map(int, rng.permutation(palette[3:])))
                      if fixed_palette else list(map(int, rng.permutation(10))))
            if split == 'train' and index == 1 and fixed_palette:
                # A previously inactive literal becomes active, and a formerly
                # active literal becomes inactive, under the same visible roles.
                old_active, old_inactive = first_colors[3], first_colors[5]
                remaining = [x for x in colors[3:] if x not in (old_active, old_inactive)]
                colors = colors[:3] + [old_inactive, remaining[0], old_active] + remaining[1:]
            if split == 'train' and index == 0:
                first_colors = colors[:]
            for attempt in range(100):
                # Reject complete incompatible structural configurations, not
                # just their positions. The chosen test holdout stays fixed.
                n, na, ni, nr, turns = _configuration(rng, shift, is_test, index if demo else None)
                pair = _scene(rng, n, na, ni, nr, turns, colors, local_rotation, demo)
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError(f"bounded packing exhausted for {split} {index}, n={n}, turns={turns}")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in ('train', 'test')), "split lists")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "pair counts")
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "canvas dimensions must be preserved")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            a = _analyze(inp)
            _require(np.array_equal(_render(a), out), "output contradicts visible routed stencil")
            normalized_out = np.rot90(out, a['rotation'])
            editable = np.zeros(a['grid'].shape, bool)
            for i in a['requests']:
                tile = a['tiles'][i]
                if tile['color'] in a['mapping']:
                    r, c = tile['top']+1, tile['left']+1
                    editable[r:r+a['n'], c:c+a['n']] = True
            _require(np.array_equal(a['grid'][~editable], normalized_out[~editable]), "clues or inactive requests changed")
            _require(set(map(int, np.unique(out))) <= set(map(int, np.unique(inp))), "invented color")
            if split == 'train':
                train.append((a, out))
    for option, message in (({'ignore_matching': True}, "matching must affect a demonstration"),
                            ({'ignore_bent_routes': True}, "tracing bends must affect a demonstration"),
                            ({'ignore_stencil': True}, "stencil holes must affect a demonstration"),
                            ({'ignore_rotation': True}, "heading must affect a demonstration"),
                            ({'keep_source_color': True}, "terminal paint must affect a demonstration"),
                            ({'remap_again': True}, "direct substitution must affect a demonstration")):
        _require(any(not np.array_equal(_render(a, **option), out) for a, out in train), message)
    headings = {r['heading'] for a, _ in train for r in a['routes'] if r['active']}
    _require((0, 1) in headings and (0, -1) in headings and any(d[0] for d in headings), "training heading coverage")
    _require(all(any(_gap_count(a, r) for r in a['routes'] if r['active']) for a, _ in train[:2]),
             "teaching scenes must actually expose continuation through gaps")
    _require(any(any(sum(a['tiles'][i]['color'] == source for i in a['requests']) > 1
                         for source in a['mapping']) for a, _ in train),
             "training must demonstrate repeated active requests")


def self_test():
    """Bounded 42-seed semantic, variation, contract and equivariance checks."""
    shapes, masks, structures, headings, counts, sizes, signatures = set(), set(), set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                before = inp.copy()
                _require(np.array_equal(solve(inp), out) and np.array_equal(inp, before), "input-only solve or mutation")
                a = _analyze(inp)
                shapes.add(inp.shape)
                sizes.add(a['n'])
                counts.add((len(a['mapping']), len(a['requests'])))
                headings.update(r['heading'] for r in a['routes'] if r['active'])
                masks.update(t['mask'].tobytes() for t in a['tiles'] if not t['mask'].all())
                roles = {}
                structures.add(tuple(roles.setdefault(int(v), len(roles)) for v in inp.flat))
        pair = puzzle['test'][0]
        # Global rotation and a bijective recoloring cannot change the rule.
        permutation = np.random.default_rng(7700+seed).permutation(10)
        _require(np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']]), "color equivariance")
        _require(np.array_equal(solve(np.rot90(pair['input'])), np.rot90(pair['output'])), "rotation equivariance")
    _require(len(signatures) == 42 and len(structures) >= 150, "structural variation collapse")
    _require(len(shapes) >= 20 and sizes == {2, 3, 4, 5}, "scale variation collapse")
    _require(headings == set(_DIRS) and len(counts) >= 10 and len(masks) >= 100, "family variation collapse")
    p, q = generate_puzzle(19, 2, 1), generate_puzzle(19, 2, 1)
    _require(all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a), "determinism")
    for a, b in ((1, 1), (11, 1), (2, 2), (3, 4), (4, 0), (6, 5), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal count accepted")
    bad = {'train': p['train'], 'test': [{'input': p['test'][0]['input'], 'output': p['test'][0]['input'].copy()}]}
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("incorrect answer accepted")


if __name__ == '__main__':
    self_test()
    print('self_test passed')
