"""Demand-guided rail repair: exact-cover assembly constrained by deliveries.

Only standard library and NumPy are used.  No parent code/data is imported.
8 denotes repair material / singleton destination; 9 denotes a fixed pin.
"""
import numpy as np


HOLE, PIN = 8, 9
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "grid colors")
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _components(mask):
    """Four-connected components of an explicitly supplied binary mask."""
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        todo, cells = [first], []
        while todo:
            p = todo.pop()
            cells.append(p)
            for d in DIRS:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    todo.append(q)
        result.append(np.array(sorted(cells), dtype=int))
    return result


def _normalize(inp):
    candidates = []
    for k in range(4):
        g = np.rot90(inp, k)
        token = int(g[0, 0])
        if (token < 8 and np.all(g[:, 0] == token)
                and np.count_nonzero(g == token) == g.shape[0]):
            candidates.append((g.copy(), k, token))
    _require(len(candidates) == 1, "one unique full-edge token strip required")
    g, k, token = candidates[0]
    bg = int(g[0, 1])
    _require(bg < 8 and bg != token and np.all(g[:, 1] == bg), "clear approach column")
    return g, k, bg, token


def _read(inp):
    g, rotation, bg, token = _normalize(_grid(inp))
    markers = _components(g == HOLE)
    sinks = {tuple(c[0]) for c in markers if len(c) == 1}
    hole_groups = [c for c in markers if len(c) > 1]
    _require(sinks and hole_groups, "destinations and repair regions required")
    holes = {tuple(p) for c in hole_groups for p in c}
    mask = (g != bg) & (g != token)
    for p in sinks:
        mask[p] = False
    pieces, fixed = [], set()
    for cells in _components(mask):
        values = set(map(int, g[cells[:, 0], cells[:, 1]]))
        if PIN in values:
            fixed.update(map(tuple, cells.tolist()))
        else:
            _require(len(values) == 1 and HOLE not in values, "loose part must be monochrome")
            color = next(iter(values))
            local = cells - cells.min(axis=0)
            _require(len(cells) < int(np.prod(local.max(axis=0) + 1)), "loose part must be irregular")
            pieces.append(dict(cells=cells, local=local, color=color))
    _require(pieces and holes <= fixed, "every repair region must belong to a pinned fixture")
    _require(sum(len(p['cells']) for p in pieces) == len(holes), "spare-part area must equal missing material")
    base = g.copy()
    for piece in pieces:
        c = piece['cells']
        base[c[:, 0], c[:, 1]] = bg
    # Recover rectangular banks from the marked regions, independent of sampler
    # coordinates. Other thin rails can join a bank outside its column span.
    banks = []
    for cells in hole_groups:
        a, c = cells.min(axis=0)
        b, d = cells.max(axis=0) + 1
        while a > 0 and np.all((base[a-1, c:d] != bg) & (base[a-1, c:d] != token)):
            a -= 1
        while b < len(base) and np.all((base[b, c:d] != bg) & (base[b, c:d] != token)):
            b += 1
        _require(np.all(base[a:b, c:d] != bg), "repair footprint must complete a solid bank")
        banks.append((int(a), int(b), int(c), int(d)))
    return dict(grid=g, base=base, rotation=rotation, bg=bg, token=token,
                holes=holes, sinks=sinks, pieces=pieces, fixed=fixed, banks=banks)


def _paired(g, p, direction, bg, token):
    n = (-direction[1], direction[0])
    a, b = _add(p, n), _add(p, (-n[0], -n[1]))
    return (_inside(g, a) and _inside(g, b) and int(g[a]) == int(g[b])
            and int(g[a]) not in (bg, token, HOLE, PIN))


def _routes(g, bg, token, ignore_equality=False):
    """Directional rails, rather than connected background components.

    Marked destinations are free cells during tracing. Continue straight until
    blocked. A turn requires matching flanks at its first free cell.
    """
    free = lambda p: _inside(g, p) and int(g[p]) in (bg, HOLE)
    ordinary = lambda v: v not in (bg, token, HOLE, PIN)
    routes, mismatches, visited = [], [], set()
    for row in range(1, len(g)-1):
        entrance = None
        for col in range(1, g.shape[1]):
            if not free((row, col)):
                break
            a, b = int(g[row-1, col]), int(g[row+1, col])
            if ordinary(a) and ordinary(b):
                if a == b or ignore_equality:
                    entrance = (row, col)
                else:
                    mismatches.append((row, col))
                break
        if entrance is None:
            continue
        p, direction = (row, 1), (0, 1)
        path, states, turns = [p], set(), []
        while True:
            _require((p, direction) not in states, "cyclic route")
            states.add((p, direction))
            q = _add(p, direction)
            if not _inside(g, q):
                break
            if free(q):
                p = q
                path.append(p)
                continue
            choices = []
            for d in ((-direction[1], direction[0]), (direction[1], -direction[0])):
                q = _add(p, d)
                if free(q) and _paired(g, q, d, bg, token):
                    choices.append(d)
            _require(len(choices) <= 1, "branching turn")
            if not choices:
                break
            turns.append((p, direction, choices[0]))
            direction = choices[0]
        _require(len(path) == len(set(path)) and not (set(path) & visited), "routes must be disjoint")
        visited.update(path)
        routes.append(dict(start=(row, 0), end=p, path=path, turns=turns,
                           entrance=entrance, direction=direction,
                           open=p[0] in (0, len(g)-1) or p[1] == g.shape[1]-1))
    return routes, mismatches


def _deliver(repaired, scene):
    routes, mismatches = _routes(repaired, scene['bg'], scene['token'])
    _require({r['end'] for r in routes} == scene['sinks'], "deliveries must equal the marked destinations")
    out = repaired.copy()
    for route in routes:
        out[route['start']] = scene['bg']
        out[route['end']] = scene['token']
    return out, routes, mismatches


def _repairs(scene, require_delivery=True, limit=2):
    """Exhaust all translations and exact covers; quotient identical parts.

    The limit counts distinct colored outputs, never component permutations.
    Geometry-only enumeration is also used to establish a real B ablation.
    """
    holes = sorted(scene['holes'])
    index = {p: i for i, p in enumerate(holes)}
    pieces = scene['pieces']
    types = [(p['color'], tuple(map(tuple, p['local'].tolist()))) for p in pieces]
    placements, at_cell = [], [[] for _ in holes]
    for i, piece in enumerate(pieces):
        local = piece['local']
        for anchor in holes:
            offset = np.array(anchor) - local[0]
            cells = local + offset
            targets = list(map(tuple, cells.tolist()))
            if all(p in index for p in targets):
                bits = sum(1 << index[p] for p in targets)
                placement = (i, bits, cells)
                k = len(placements)
                placements.append(placement)
                for p in targets:
                    at_cell[index[p]].append(k)
    full, answers = (1 << len(holes))-1, {}

    def search(covered, used, chosen):
        if len(answers) >= limit:
            return
        if covered == full:
            _require(used == (1 << len(pieces))-1, "all pieces must be used")
            repaired = scene['base'].copy()
            for k in chosen:
                i, _, cells = placements[k]
                repaired[cells[:, 0], cells[:, 1]] = pieces[i]['color']
            try:
                if require_delivery:
                    out, routes, mismatches = _deliver(repaired, scene)
                else:
                    out, routes, mismatches = repaired, [], []
            except ValueError:
                return
            answers.setdefault(out.tobytes(), dict(output=out, repaired=repaired,
                placements=[(placements[k][0], placements[k][2]) for k in chosen],
                routes=routes, mismatches=mismatches))
            return
        eligible = []
        for i in range(len(pieces)):
            eligible.append(not (used >> i) & 1 and not any(
                not (used >> j) & 1 and types[j] == types[i] for j in range(i)))
        best = None
        for j in range(len(holes)):
            if (covered >> j) & 1:
                continue
            options = [k for k in at_cell[j] if eligible[placements[k][0]]
                       and not placements[k][1] & covered]
            if not options:
                return
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        for k in best:
            i, bits, _ = placements[k]
            search(covered | bits, used | (1 << i), chosen + [k])

    search(0, 0, [])
    return list(answers.values())


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer the unique joint repair and delivery using this input alone."""
    scene = _read(input_grid)
    answers = _repairs(scene)
    _require(len(answers) == 1, "expected one distinct joint repair/delivery")
    return np.rot90(answers[0]['output'], -scene['rotation']).copy()


def _polyline(points):
    path = [points[0]]
    for target in points[1:]:
        p = path[-1]
        d = (int(np.sign(target[0]-p[0])), int(np.sign(target[1]-p[1])))
        _require((d[0] == 0) != (d[1] == 0), "orthogonal nonempty segment")
        while p != target:
            p = _add(p, d)
            path.append(p)
    return path


def _dilate(mask):
    out = mask.copy()
    h, w = mask.shape
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            a, b, c, d = max(0, -dr), min(h, h-dr), max(0, -dc), min(w, w-dc)
            out[a+dr:b+dr, c+dc:d+dc] |= mask[a:b, c:d]
    return out


def _scatter(rng, base, pieces, bg, protected):
    h, w = base.shape
    for _ in range(18):
        g = base.copy()
        forbidden = _dilate(base != bg) | protected
        forbidden[:, :2] = True
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i][0]) + rng.random()*4)
        success = True
        for i in order:
            cells, color = pieces[i]
            local = cells-cells.min(axis=0)
            ph, pw = local.max(axis=0)+1
            valid = np.ones((h-ph+1, w-pw+1), dtype=bool)
            for r, c in local:
                valid &= ~forbidden[r:r+h-ph+1, c:c+w-pw+1]
            choices = np.argwhere(valid)
            if not len(choices):
                success = False
                break
            dest = local+choices[int(rng.integers(len(choices)))]
            g[dest[:, 0], dest[:, 1]] = color
            forbidden = _dilate(g != bg) | protected
            forbidden[:, :2] = True
        if success:
            return g
    raise ValueError("scatter space exhausted")


def _pin_fixtures(rng, g, bg, token, paths, sinks):
    mask = (g != bg) & (g != token)
    for p in sinks:
        mask[p] = False
    occupied_path = {p for path in paths for p in path}
    for cells in _components(mask):
        # Adding a pin outside the path preserves the route's geometry. A pin
        # is a blocker, but is never an ordinary matching guide identity.
        choices = set()
        for p in map(tuple, cells.tolist()):
            for d in DIRS:
                q = _add(p, d)
                if (_inside(g, q) and q[1] >= 2 and g[q] == bg
                        and q not in occupied_path):
                    choices.add(q)
        _require(choices, "fixture needs a safe pin location")
        choices = sorted(choices)
        g[choices[int(rng.integers(len(choices)))]] = PIN


def _make_pair(rng, gate_count, width_range, rotation, roles, demo):
    bg, token, palette = roles
    w = int(rng.integers(*width_range))
    part_width = int(rng.integers(3, 7))
    heights = [4]*gate_count if gate_count == 4 else list(map(int, rng.integers(5, 7, gate_count)))
    used_height = sum(h+3 for h in heights)
    _require(used_height <= 30, "gate stack height")
    h = int(rng.integers(max(used_height, 18), 31))
    gaps = [0]*(gate_count+1)
    for _ in range(h-used_height):
        gaps[int(rng.integers(gate_count+1))] += 1
    qmax = int(rng.integers(2, min(3, min(heights)-2)+1))
    q = rng.integers(1, qmax+1, part_width)
    q[0], q[-1] = 1, qmax
    if rng.random() < .5:
        q = q[::-1]
    colors = list(map(int, rng.permutation(palette)))
    caps, inner, fixed = colors[:gate_count], colors[gate_count], colors[gate_count+1]
    inactive = int(rng.integers(gate_count))
    active = [i for i in range(gate_count) if i != inactive]
    kinds = list(map(int, rng.integers(0, 3, gate_count)))
    if demo == 0:
        kinds = [0]*gate_count
    elif demo == 1:
        kinds[active[0]], kinds[active[1]] = 1, 2
    g = np.full((h, w), bg, dtype=np.int8)
    g[:, 0] = token
    pieces, holes, paths, sinks, intended = [], set(), [], set(), {}
    lower_colors = {}
    top = gaps[0]
    for i, bank_h in enumerate(heights):
        module_h = bank_h+3
        start = int(rng.integers(2, 5))
        bank_end = start+part_width-1
        r = 1
        rail_color = caps[i] if i != inactive else caps[active[int(rng.integers(len(active)))]]
        module = np.full((module_h, w), bg, dtype=np.int8)
        kind = kinds[i]
        closed = kind == 1 or bool(rng.integers(2))
        if demo == 0 and i in active:
            closed = False
        if demo == 1 and i == active[1]:
            closed = True
        turn = int(rng.integers(bank_end+3, w-4))
        offset = int(rng.integers(2, bank_h))
        if kind == 0:
            end = int(rng.integers(bank_end+4, w-1)) if closed else w-1
            points = [(r, 1), (r, end)]
        elif kind == 1:
            points = [(r, 1), (r, turn), (r+offset, turn)]
        else:
            end = int(rng.integers(turn+2, w-1)) if closed else w-1
            points = [(r, 1), (r, turn), (r+offset, turn), (r+offset, end)]
        path = _polyline(points)
        path_set = set(path)
        for p in path:
            for d in DIRS:
                z = _add(p, d)
                if _inside(module, z) and z[1] >= start and z not in path_set:
                    module[z] = rail_color
        # Complementary layers: common outer cap, shaped interior, fixed root.
        s = rng.integers(int(q.max())+1, bank_h, part_width)
        cap_cells, lower_cells, marked = [], [], []
        for c in range(part_width):
            module[r+1:r+1+bank_h, start+c] = fixed
            for y in range(int(s[c])):
                p = (r+1+y, start+c)
                module[p] = caps[i] if y < q[c] else inner
                (cap_cells if y < q[c] else lower_cells).append(p)
                marked.append(p)
        # Remove only straight evidence away from banks, approaches and turns.
        directions = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
        candidates = []
        for j in range(3, len(path)-3):
            p = path[j]
            if p[1] <= bank_end+1 or len(set(directions[j-2:j+2])) != 1:
                continue
            d = directions[j]
            n = (-d[1], d[0])
            flanks = [_add(p, n), _add(p, (-n[0], -n[1]))]
            if all(_inside(module, z) and module[z] == rail_color for z in flanks):
                candidates.append((j, flanks))
        if candidates:
            # Every first demo has a true interruption after its broken bank.
            first, _ = candidates[int(rng.integers(len(candidates)))]
            for j, flanks in candidates:
                if first <= j < first+int(rng.integers(1, 3)):
                    for z in flanks:
                        module[z] = bg
        # Unsupported final runs make the destination different from rail end.
        last_d = directions[-1]
        n = (-last_d[1], last_d[0])
        for j in range(len(path)-1, max(1, len(path)-3), -1):
            if directions[j-1] != last_d or path[j][1] <= bank_end+1:
                break
            for z in (_add(path[j], n), _add(path[j], (-n[0], -n[1]))):
                if _inside(module, z):
                    module[z] = bg
        flip = bool(rng.integers(2))
        def mapped(cells):
            a = np.array(cells, dtype=int)
            if flip:
                a[:, 0] = module_h-1-a[:, 0]
            a[:, 0] += top
            return a
        if flip:
            module = module[::-1]
        cp, lp, mk, pp = map(mapped, (cap_cells, lower_cells, marked, path))
        lower_shape = tuple(map(tuple, (lp-lp.min(axis=0))[np.lexsort((lp[:, 1], lp[:, 0]))].tolist()))
        if lower_shape not in lower_colors:
            options = [v for v in palette if v != fixed and (not lower_colors or v != list(lower_colors.values())[-1])]
            lower_colors[lower_shape] = inner if not lower_colors else int(rng.choice(options))
        lower_color = lower_colors[lower_shape]
        local_lower = lp-np.array([top, 0])
        module[local_lower[:, 0], local_lower[:, 1]] = lower_color
        g[top:top+module_h, 2:] = module[:, 2:]
        pieces.extend(((cp, caps[i]), (lp, lower_color)))
        holes.update(map(tuple, mk.tolist()))
        paths.append(list(map(tuple, pp.tolist())))
        if i != inactive:
            sink = tuple(pp[-1])
            sinks.add(sink)
            intended[(int(pp[0, 0]), 0)] = sink
        top += module_h+gaps[i+1]
    if demo == 1:
        _require(len(set(lower_colors.values())) >= 2, "demonstrate shape-dependent interior colors")
    # Put visible requests on the free destinations before marking repair holes.
    repaired = g.copy()
    for p in sinks:
        repaired[p] = HOLE
    _pin_fixtures(rng, repaired, bg, token, paths, sinks)
    actual, mismatches = _routes(repaired, bg, token)
    _require({r['start']: r['end'] for r in actual} == intended and mismatches,
             "sampled rails do not encode intended deliveries")
    for route in actual:
        _require(route['path'] in paths, "actual guide path differs from constructed path")
    base = repaired.copy()
    for p in holes:
        base[p] = HOLE
    protected = np.zeros_like(base, dtype=bool)
    for path in paths:
        for p in path:
            protected[p] = True
    inp = _scatter(rng, base, pieces, bg, protected)
    out = repaired.copy()
    for start, end in intended.items():
        out[start] = bg
        out[end] = token
    inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
    scene = _read(inp)
    answers = _repairs(scene)
    _require(len(answers) == 1, "joint assembly ambiguous")
    _require(np.array_equal(np.rot90(answers[0]['output'], -scene['rotation']), out),
             "construction and input-derived solution disagree")
    _require(len(_repairs(scene, False)) == 2, "geometry alone must be ambiguous")
    return dict(input=inp, output=out)


def _count(value, name, low, high):
    _require(not isinstance(value, (bool, np.bool_)) and isinstance(value, (int, np.integer)),
             name+" must be an integer")
    value = int(value)
    _require(low <= value <= high, name+" out of range")
    return value


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    n, m = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    _require(n > m, "num_train must exceed num_test")
    _require(not isinstance(seed, (bool, np.bool_)) and isinstance(seed, (int, np.integer)), "integer seed required")
    rng = np.random.default_rng(int(seed))
    palette = list(map(int, rng.permutation(8)))
    shared_roles = (palette[0], palette[1], palette[2:])
    stable = bool(rng.integers(2))
    base_rotation, shift = int(rng.integers(4)), int(rng.integers(3))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', n), ('test', m)):
        for i in range(count):
            if split == 'train':
                gates = 2 if i == 0 else 3 if i == 1 else int(rng.integers(2, 4))
                rotation = (base_rotation+i % 2) % 4 if shift == 1 else base_rotation
                widths = (20, 26)
                demo = i if i < 2 else -1
            else:
                gates = 4 if shift == 1 else int(rng.integers(2, 4))
                rotation = (base_rotation+int(rng.integers(1, 4))) % 4 if shift == 0 else base_rotation
                widths = (28, 31) if shift == 2 else (23, 31)
                demo = -1
            if stable:
                roles = shared_roles
            else:
                palette = list(map(int, rng.permutation(8)))
                roles = (palette[0], palette[1], palette[2:])
            for attempt in range(180):
                try:
                    pair = _make_pair(rng, gates, widths, rotation, roles, demo)
                    if _key(pair['input']) in seen:
                        continue
                    info = _pair_info(pair)
                    if split == 'train' and i == 0 and not info['gaps']:
                        continue
                    break
                except ValueError:
                    continue
            else:
                raise RuntimeError("bounded scene construction exhausted")
            seen.add(_key(pair['input']))
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_info(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
    inp, out = _grid(pair['input']), _grid(pair['output'])
    _require(inp.shape == out.shape and not np.array_equal(inp, out), "nontrivial shape-preserving transformation")
    s = _read(inp)
    answers = _repairs(s)
    _require(len(answers) == 1, "unique colored joint solution required")
    a = answers[0]
    canonical = np.rot90(out, s['rotation'])
    _require(np.array_equal(canonical, a['output']), "incorrect output")
    _require(len(_repairs(s, False)) == 2, "routing must disambiguate assembly")
    _require(not np.any(out == HOLE), "all missing material and requests must be resolved")
    _require(np.array_equal(inp == PIN, out == PIN), "pins preserved")
    for p in s['fixed'] - s['holes']:
        _require(canonical[p] == s['grid'][p], "fixed guide changed")
    for color in range(10):
        if color not in (s['bg'], HOLE):
            _require(np.count_nonzero(inp == color) == np.count_nonzero(out == color), "color/token conservation")
    for i, cells in a['placements']:
        piece = s['pieces'][i]
        _require(np.array_equal(cells-cells.min(axis=0), piece['local']), "part must translate exactly")
        _require(np.all(a['repaired'][cells[:, 0], cells[:, 1]] == piece['color']), "part color preserved")
    _require(len(a['placements']) == len(s['pieces']), "every part used once")
    interior_colors, interior_shapes = set(), set()
    for r0, r1, c0, c1 in s['banks']:
        _require(np.all(a['repaired'][r0:r1, c0:c1] != s['bg']), "completed solid bank")
        faces = [row for row in (r0, r1-1)
                 if all((row, col) in s['holes'] for col in range(c0, c1))]
        _require(len(faces) == 1, "one marked bank face required")
        for i, cells in a['placements']:
            if (np.all((r0 <= cells[:, 0]) & (cells[:, 0] < r1)
                       & (c0 <= cells[:, 1]) & (cells[:, 1] < c1))
                    and not np.any(cells[:, 0] == faces[0])):
                interior_colors.add(s['pieces'][i]['color'])
                interior_shapes.add(tuple(map(tuple, s['pieces'][i]['local'].tolist())))
    _require(a['mismatches'], "inactive mismatched entrance required")
    try:
        color_blind, _ = _routes(a['repaired'], s['bg'], s['token'], ignore_equality=True)
    except ValueError:
        color_blind = None
    _require(color_blind is None or {r['end'] for r in color_blind} != s['sinks'],
             "guide equality must affect the deliveries")
    no_shape = s['base'].copy()
    for p in s['holes']:
        no_shape[p] = s['bg']
    # Without reconstructing A's occupied shapes the demanded rail behavior is
    # absent. This is a failure witness, not a separate semantic certification.
    try:
        bad, _ = _routes(no_shape, s['bg'], s['token'])
    except ValueError:
        bad = None
    _require(bad is None or {r['end'] for r in bad} != s['sinks'],
             "repair concept must affect deliveries")
    route_gaps, tails = [], []
    for route in a['routes']:
        path = route['path']
        ds = [(b[0]-p[0], b[1]-p[1]) for p, b in zip(path, path[1:])]
        supported = [_paired(a['repaired'], p, d, s['bg'], s['token']) for p, d in zip(path[1:], ds)]
        for j in range(1, len(supported)-1):
            p, direction = path[j+1], ds[j]
            normal = (-direction[1], direction[0])
            flanks = (_add(p, normal), _add(p, (-normal[0], -normal[1])))
            if (not supported[j] and any(supported[:j]) and any(supported[j+1:])
                    and len(set(ds[max(0, j-1):j+2])) == 1
                    and all(_inside(a['repaired'], z) and a['repaired'][z] == s['bg'] for z in flanks)):
                route_gaps.append(j)
        last = max((j for j, yes in enumerate(supported) if yes), default=-1)
        tails.append(len(supported)-1-last)
    return dict(scene=s, answer=a, gates=len(s['banks']), shape=inp.shape,
                pieces=len(s['pieces']), gaps=route_gaps, tails=tails,
                interior_colors=interior_colors, interior_shapes=interior_shapes,
                bends=[len(r['turns']) for r in a['routes']],
                opens=[r['open'] for r in a['routes']])


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[k], list) for k in puzzle), "split lists")
    n, m = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= n <= 10 and 1 <= m <= 4 and n > m, "example counts")
    seen, train = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            info = _pair_info(pair)
            key = _key(np.asarray(pair['input']))
            _require(key not in seen, "duplicate input")
            seen.add(key)
            if split == 'train':
                train.append(info)
    _require({0, 1, 2} <= {v for f in train for v in f['bends']}, "training must demonstrate straight, elbow and offset")
    _require(any(f['gaps'] for f in train) and any(any(t > 0 for t in f['tails']) for f in train), "training must show inertia")
    _require({True, False} <= {v for f in train for v in f['opens']}, "training must show both terminal types")
    _require(len({f['gates'] for f in train}) > 1, "training bank counts vary")
    _require(any(len(f['interior_colors']) >= 2 and len(f['interior_shapes']) >= 2 for f in train),
             "training must distinguish differently colored interior shapes")


def _same(p, q):
    return all(len(p[s]) == len(q[s]) and all(np.array_equal(a[k], b[k])
        for a, b in zip(p[s], q[s]) for k in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    """Bounded, actual-grid variation and family-specific falsification checks."""
    features, structures, firsts = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        firsts.add(_key(puzzle['train'][0]['input']))
        for pair in puzzle['train']+puzzle['test']:
            info = _pair_info(pair)
            features.append(info)
            g = info['scene']['grid']
            roles = {}
            normalized = tuple(roles.setdefault(int(v), len(roles)) for v in g.flat)
            structures.add((g.shape, normalized))
            _require(np.array_equal(solve(pair['input'].tolist()), pair['output']), "input-only solve")
    _require(len(firsts) == 42 and len(structures) >= 200, "structural variation across 42 seeds")
    _require(len({f['shape'] for f in features}) >= 35, "canvas variation")
    _require({2, 3, 4} <= {f['gates'] for f in features}, "bank-count variation")
    _require({0, 1, 2, 3} == {f['scene']['rotation'] for f in features}, "entry-side variation")
    _require(len({tuple(f['bends']) for f in features}) >= 10, "route-geometry variation")
    _require(len({(b-a, d-c) for f in features for a, b, c, d in f['scene']['banks']}) >= 10, "bank geometry variation")
    _require(len({tuple(p['local'].flatten()) for f in features for p in f['scene']['pieces']}) >= 50, "complementary-mask variation")
    _require(len({(f['scene']['bg'], f['scene']['token']) for f in features}) >= 20, "palette-role variation")
    p = generate_puzzle(53, 2, 1)
    _require(_same(p, generate_puzzle(53, 2, 1)), "determinism")
    for n in range(2, 11):
        for m in range(1, min(4, n-1)+1):
            q = generate_puzzle(71+n*7+m, n, m)
            _require(len(q['train']) == n and len(q['test']) == m, "explicit counts")
    for n, m in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, n, m)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    broken = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    broken['test'][0]['output'] = broken['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(broken)
    except ValueError:
        pass
    else:
        raise AssertionError("incorrect output accepted")
