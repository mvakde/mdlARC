"""Collision collars: ray encounters synthesize keys for rigid movable pieces.

Only standard-library modules and NumPy are used.  The input-only solver parses
visible objects, forecasts spatial encounters, and matches complete marker masks.
It neither imports parent generators nor reads any files.
"""
from collections import deque
import json
import numpy as np

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))
PORTS = {(-1, -1): (-1, -1), (-1, 0): (-1, 0), (-1, 1): (-1, 0),
         (-1, 2): (-1, 1), (0, -1): (0, -1), (1, -1): (0, -1),
         (0, 2): (0, 1), (1, 2): (0, 1), (2, -1): (1, -1),
         (2, 0): (1, 0), (2, 1): (1, 0), (2, 2): (1, 1)}
KINDS = ('perpendicular', 'opposing_diagonals', 'staircase')


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), 'invalid grid dimensions')
    _require(np.issubdtype(a.dtype, np.integer), 'grid must have integer cells')
    _require(np.all((a >= 0) & (a <= 9)), 'invalid ARC colors')
    return a.astype(np.int8, copy=False)


def _components(cells, neighbors=N8):
    left = set(cells)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in neighbors:
                q = r + dr, c + dc
                if q in left:
                    left.remove(q)
                    group.add(q)
                    stack.append(q)
        result.append(group)
    return result


def _box(cells):
    return min(r for r, c in cells), min(c for r, c in cells), max(r for r, c in cells), max(c for r, c in cells)


def _key(cells):
    t, l, _, _ = _box(cells)
    return tuple(sorted((r - t, c - l) for r, c in cells))


def _shift(cells, delta):
    return {(r + delta[0], c + delta[1]) for r, c in cells}


def _inside(q, shape):
    return 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1]


def _buffer(cells):
    return {(r + dr, c + dc) for r, c in cells for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _has_square(cells):
    return any({(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} <= cells for r, c in cells)


def _parse(a):
    counts = np.bincount(a.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg] > a.size // 2, 'background must occupy more than half the input')
    groups = _components(map(tuple, np.argwhere(a != bg).tolist()))
    _require(len(groups) >= 3, 'need fixed launchers and loose couplers')
    palettes = [{int(a[q]) for q in group} for group in groups]
    _require(all(len(p) == 2 for p in palettes), 'every input object must have two colors')
    common = set.intersection(*palettes)
    _require(len(common) == 1, 'common structure/connector color must be visible')
    marker = common.pop()
    sources, tokens = [], []
    for cells, palette in zip(groups, palettes):
        color = (palette - {marker}).pop()
        body = {q for q in cells if a[q] == color}
        pins = cells - body
        cores = [(r, c) for r, c in sorted(body)
                 if {(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)} == body]
        if cores:
            r, c = cores[0]
            footprint = {(r + dr, c + dc) for dr in range(-1, 3) for dc in range(-1, 3)}
            _require(all(_inside(q, a.shape) for q in footprint), 'launcher ring is not clipped')
            _require(cells <= footprint, 'extra cells on a launcher')
            rays = []
            for offset, direction in PORTS.items():
                q = r + offset[0], c + offset[1]
                _require(int(a[q]) in (bg, marker), 'invalid launcher ring')
                if a[q] == bg:
                    rays.append((q, direction))
            _require(1 <= len(rays) <= 4 and len(pins) >= 8, 'invalid launcher ports')
            sources.append(dict(cells=cells, body=body, pins=pins, color=color, core=(r, c), rays=rays))
        else:
            _require(len(pins) == 6, 'a loose collar has exactly six marker cells')
            _require(len(_components(body, N4)) == 1 and not _has_square(body), 'loose payload must be connected and contain no solid 2x2')
            tokens.append(dict(cells=cells, body=body, pins=pins, color=color))
    _require(2 <= len(sources) <= 4 and 1 <= len(tokens) <= 3, 'unsupported object counts')
    _require(len(tokens) == len(sources) - 1, 'source/coupler counts do not form a tree')
    return bg, marker, sources, tokens


def _walk(start, velocity, shape, length=None):
    vr, vc = velocity
    scale = max(abs(vr), abs(vc))
    _require(scale > 0, 'zero heading')
    result = []
    for tick in range(length if length is not None else 3 * max(shape) + 1):
        q = (start[0] + (1 if vr >= 0 else -1) * (tick * abs(vr) // scale),
             start[1] + (1 if vc >= 0 else -1) * (tick * abs(vc) // scale))
        if not _inside(q, shape):
            break
        result.append(q)
    return result


def _kind(u, v):
    n, m = sum(map(abs, u)), sum(map(abs, v))
    dot = u[0] * v[0] + u[1] * v[1]
    if n == m == 1 and dot == 0:
        return 'perpendicular'
    if n == m == 2 and dot == 0:
        return 'opposing_diagonals'
    if {n, m} == {1, 2} and dot == 1:
        return 'staircase'
    raise ValueError('unsupported parallel, head-on, or obtuse encounter')


def _encounter(a, b):
    lookup = {q: i for i, q in enumerate(a)}
    hits = [(lookup[q] - 1, j - 1, 'cell') for j, q in enumerate(b) if q in lookup]
    mids = {(p[0] + q[0], p[1] + q[1]): i for i, (p, q) in enumerate(zip(a, a[1:]))
            if p[0] != q[0] and p[1] != q[1]}
    for j, (p, q) in enumerate(zip(b, b[1:])):
        if p[0] == q[0] or p[1] == q[1]:
            continue
        mid = p[0] + q[0], p[1] + q[1]
        if mid in mids:
            i = mids[mid]
            if {a[i], a[i + 1]} != {p, q}:
                hits.append((i, j, 'between'))
    if not hits:
        return None
    return min(hits, key=lambda x: (max(x[:2]), sum(x[:2]), x[2]))


def _trace(sources, shape):
    """Forecast intersections in space, independent of emission clocks.

    A dock absorbs its two strands after a three-cell redirected collar.  Thus
    there are no later bundle events or competing raster phases to infer.
    """
    rays = []
    for i, source in enumerate(sources):
        for start, direction in source['rays']:
            rays.append(dict(source=i, start=start, direction=direction,
                             path=_walk(start, direction, shape), color=source['color']))
    pending = set(range(len(rays)))
    events = []
    while pending:
        candidates, first = [], {}
        for i in sorted(pending):
            for j in sorted(pending):
                if j <= i:
                    continue
                hit = _encounter(rays[i]['path'], rays[j]['path'])
                if hit is not None:
                    _require(rays[i]['source'] != rays[j]['source'], 'ports from one source intersect')
                    ci, cj, mode = hit
                    _require(ci >= 1 and cj >= 1, 'encounter too near a port')
                    candidates.append((i, j, ci, cj, mode))
                    first[i] = min(first.get(i, 1000), ci)
                    first[j] = min(first.get(j, 1000), cj)
        if not candidates:
            break
        ready = [x for x in candidates if x[2] == first[x[0]] and x[3] == first[x[1]]]
        _require(bool(ready), 'cyclic or ambiguous spatial encounter order')
        i, j, ci, cj, mode = min(ready, key=lambda x: (max(x[2:4]), sum(x[2:4]), x[:2]))
        _require(sum(x[2] == ci for x in candidates if x[0] == i) +
                 sum(x[3] == ci for x in candidates if x[1] == i) == 1, 'tied encounter')
        _require(sum(x[2] == cj for x in candidates if x[0] == j) +
                 sum(x[3] == cj for x in candidates if x[1] == j) == 1, 'tied encounter')
        u, v = rays[i]['direction'], rays[j]['direction']
        kind = _kind(u, v)
        vr, vc = u[0] + v[0], u[1] + v[1]
        divisor = int(np.gcd(abs(vr), abs(vc)))
        velocity = vr // divisor, vc // divisor
        lasts = (rays[i]['path'][ci], rays[j]['path'][cj])
        collar_paths = [_walk(p, velocity, shape, 3) for p in lasts]
        _require(all(len(p) == 3 for p in collar_paths), 'collar clipped at boundary')
        socket = set(collar_paths[0]) | set(collar_paths[1])
        _require(len(socket) == 6, 'overlapping collar strands')
        events.append(dict(ray_ids=(i, j), cuts=(ci, cj), mode=mode, kind=kind,
                           velocity=velocity, socket=socket, lasts=lasts,
                           distances=(ci + (1 if mode == 'cell' else .5), cj + (1 if mode == 'cell' else .5)),
                           sources=(rays[i]['source'], rays[j]['source'])))
        pending -= {i, j}
    _require(len(events) == len(sources) - 1, 'encounters must connect the launchers as one tree')
    graph = {i: set() for i in range(len(sources))}
    for event in events:
        i, j = event['sources']
        graph[i].add(j)
        graph[j].add(i)
    reached, todo = {0}, [0]
    while todo:
        for j in graph[todo.pop()]:
            if j not in reached:
                reached.add(j)
                todo.append(j)
    _require(len(reached) == len(sources), 'disconnected source network')
    fixed = set().union(*(s['cells'] for s in sources))
    sockets = set()
    for event in events:
        _require(not sockets & event['socket'], 'collars overlap')
        sockets |= event['socket']
    _require(not fixed & sockets, 'collar overwrites a launcher')
    stops = {ray: cut for event in events for ray, cut in zip(event['ray_ids'], event['cuts'])}
    trails = {}
    for i, ray in enumerate(rays):
        path = ray['path'][:stops[i] + 1] if i in stops else ray['path']
        for q in path:
            _require(q not in fixed and q not in trails, 'ray overwrites launcher or another ray')
            trails[q] = ray['color']
    for event in events:
        own_lasts = set(event['lasts'])
        _require(event['socket'] & set(trails) == own_lasts, 'collar intersects an incoming trail')
    return events, trails, pending


def _infer(input_grid):
    a = _grid(input_grid)
    bg, marker, sources, tokens = _parse(a)
    events, trails, free = _trace(sources, a.shape)
    base = np.full(a.shape, bg, dtype=np.int8)
    occupied = set()
    for source in sources:
        for q in source['body']:
            base[q] = source['color']
        for q in source['pins']:
            base[q] = marker
        occupied |= source['cells']
    for q, color in trails.items():
        base[q] = color
        occupied.add(q)
    all_pins = set().union(*(e['socket'] for e in events))
    occupied |= all_pins
    for q in all_pins:
        base[q] = marker
    options = []
    for event in events:
        choices = []
        r, c, _, _ = _box(event['socket'])
        for i, token in enumerate(tokens):
            if _key(token['pins']) != _key(event['socket']):
                continue
            t, l, _, _ = _box(token['pins'])
            delta = r - t, c - l
            body = _shift(token['body'], delta)
            if all(_inside(q, a.shape) for q in body) and not body & occupied:
                choices.append((i, delta, body))
        _require(bool(choices), 'socket has no collision-free matching piece')
        options.append(choices)
    order = sorted(range(len(events)), key=lambda i: len(options[i]))
    answers = {}

    def visit(depth, placements, bodies, out):
        if depth == len(order):
            answers.setdefault(out.tobytes(), (out, dict(placements)))
            return
        for i, delta, body in options[order[depth]]:
            if i in placements or body & bodies:
                continue
            nxt = out.copy()
            for q in body:
                nxt[q] = tokens[i]['color']
            placements[i] = delta
            visit(depth + 1, placements, bodies | body, nxt)
            del placements[i]
            if len(answers) > 1:
                return

    visit(0, {}, set(), base)
    _require(len(answers) == 1, 'docking has no unique output under complete-piece constraints')
    out, placements = next(iter(answers.values()))
    info = dict(background=bg, marker=marker, sources=sources, tokens=tokens,
                events=events, trails=trails, free=free, placements=placements)
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the unique output from visible geometry and fixed family conventions."""
    return _infer(input_grid)[0]


def _source_grid(shape, specs):
    a = np.zeros(shape, dtype=np.int8)
    sources = []
    for spec in specs:
        r, c = spec['core']
        body = {(r + dr, c + dc) for dr in (0, 1) for dc in (0, 1)}
        ring = {(r + dr, c + dc) for dr, dc in PORTS}
        ports = {(r + dr, c + dc) for dr, dc in spec['ports']}
        pins = ring - ports
        cells = body | pins
        _require(all(_inside(q, shape) for q in body | ring), 'source outside canvas')
        _require(all(not _buffer(s['cells']) & cells for s in sources), 'source objects touch')
        for q in body:
            a[q] = spec['color']
        for q in pins:
            a[q] = 1
        sources.append(dict(cells=cells, body=body, pins=pins, color=spec['color'], core=(r, c),
                            rays=[((r + dr, c + dc), PORTS[dr, dc]) for dr, dc in spec['ports']]))
    return a, sources


def _pick(rng, sequence):
    return sequence[int(rng.integers(len(sequence)))]


def _add_source(rng, specs, shape, kind, long):
    for attempt in range(150):
        parent = int(rng.integers(len(specs)))
        source = specs[parent]
        used = {PORTS[o] for o in source['ports']}
        options = [o for o in PORTS if PORTS[o] not in used]
        if len(source['ports']) >= 3 or not options:
            continue
        offset = _pick(rng, options)
        u = PORTS[offset]
        possible = []
        for v in set(PORTS.values()):
            try:
                if _kind(u, v) == kind:
                    possible.append(v)
            except ValueError:
                pass
        if not possible:
            continue
        v = _pick(rng, sorted(possible))
        other_offset = _pick(rng, [o for o in PORTS if PORTS[o] == v])
        t, s = (int(x) for x in rng.integers(3, 12 if long else 9, 2))
        if kind == 'opposing_diagonals' and rng.random() < .5:
            t += .5
            s += .5
        p = source['core'][0] + offset[0], source['core'][1] + offset[1]
        q = p[0] + t * u[0] - s * v[0], p[1] + t * u[1] - s * v[1]
        if any(x != int(x) for x in q):
            continue
        core = int(q[0]) - other_offset[0], int(q[1]) - other_offset[1]
        candidate = [dict(x, ports=list(x['ports'])) for x in specs]
        candidate[parent]['ports'].append(offset)
        candidate.append(dict(core=core, ports=[other_offset], color=2 + len(specs)))
        try:
            _, sources = _source_grid(shape, candidate)
            events, _, free = _trace(sources, shape)
            _require(len(events) == len(candidate) - 1 and not free, 'unintended encounter')
            return candidate
        except ValueError:
            continue
    return None


def _add_free_port(rng, specs, shape):
    _, sources = _source_grid(shape, specs)
    old, _, free = _trace(sources, shape)
    old_keys = {_key(e['socket']) for e in old}
    choices = [(i, o) for i, s in enumerate(specs) for o in PORTS if o not in s['ports']
               and len(s['ports']) < 4]
    rng.shuffle(choices)
    for i, offset in choices:
        candidate = [dict(x, ports=list(x['ports'])) for x in specs]
        candidate[i]['ports'].append(offset)
        try:
            _, sources = _source_grid(shape, candidate)
            events, _, new_free = _trace(sources, shape)
            _require({_key(e['socket']) for e in events} == old_keys and len(new_free) == len(free) + 1, 'port changes encounters')
            return candidate
        except ValueError:
            pass
    return None


def _body(rng, socket, blocked, shape, budget):
    """Connect a thin payload to all marker components, then grow its silhouette."""
    t, l, b, r = _box(socket)
    allowed = {(a, c) for a in range(max(0, t - 3), min(shape[0], b + 4))
               for c in range(max(0, l - 3), min(shape[1], r + 4))} - blocked - socket
    groups = _components(socket)
    near = [{q for q in allowed if any((q[0] + dr, q[1] + dc) in g for dr, dc in N8)} for g in groups]
    if any(not g for g in near):
        return None
    starts = sorted(near[0])
    rng.shuffle(starts)
    for start in starts[:30]:
        body = {start}
        valid = True
        for target_set in near[1:]:
            if body & target_set:
                continue
            queue = deque(sorted(body))
            previous = {q: None for q in body}
            endpoint = None
            while queue:
                q = queue.popleft()
                if q in target_set:
                    endpoint = q
                    break
                for dr, dc in N4:
                    nxt = q[0] + dr, q[1] + dc
                    if nxt in allowed and nxt not in previous:
                        previous[nxt] = q
                        queue.append(nxt)
            if endpoint is None:
                valid = False
                break
            while endpoint is not None:
                body.add(endpoint)
                endpoint = previous[endpoint]
        if not valid or _has_square(body) or len(body) > budget:
            continue
        goal = int(rng.integers(max(3, len(body)), budget + 1))
        for step in range(budget * 3):
            if len(body) >= goal:
                break
            frontier = sorted({(a + dr, c + dc) for a, c in body for dr, dc in N4} & allowed - body)
            frontier = [q for q in frontier if not _has_square(body | {q})]
            if not frontier:
                break
            body.add(_pick(rng, frontier))
        if len(body) >= 3 and len(_components(body | socket)) == 1:
            return body
    return None


def _scatter(rng, shape, sources, tokens):
    fixed = set().union(*(s['cells'] for s in sources))
    for restart in range(20):
        occupied = _buffer(fixed)
        order = sorted(range(len(tokens)), key=lambda i: -len(tokens[i]['cells']))
        rng.shuffle(order)
        placements = {}
        for i in order:
            token = tokens[i]
            t, l, b, r = _box(token['cells'])
            bh, bw = b - t + 1, r - l + 1
            for attempt in range(350):
                delta = int(rng.integers(shape[0] - bh + 1)) - t, int(rng.integers(shape[1] - bw + 1)) - l
                cells = _shift(token['cells'], delta)
                if delta != (0, 0) and not cells & occupied:
                    placements[i] = delta
                    occupied |= _buffer(cells)
                    break
            else:
                break
        if len(placements) == len(tokens):
            a = np.zeros(shape, dtype=np.int8)
            for source in sources:
                for q in source['body']:
                    a[q] = source['color']
                for q in source['pins']:
                    a[q] = 1
            for i, token in enumerate(tokens):
                for q in _shift(token['body'], placements[i]):
                    a[q] = token['color']
                for q in _shift(token['pins'], placements[i]):
                    a[q] = 1
            return a
    return None


def _nearest_is_wrong(info):
    """A concrete competing rule for the distinct-key teaching example."""
    for event in info['events']:
        target = np.mean(sorted(event['socket']), axis=0)
        chosen = min(range(len(info['tokens'])), key=lambda i:
                     float(np.sum((np.mean(sorted(info['tokens'][i]['pins']), axis=0) - target) ** 2)))
        if _key(info['tokens'][chosen]['pins']) != _key(event['socket']):
            return True
    return False


def _make_pair(rng, edge_kinds, long=False, free_required=False, allow_free=True,
               profile=None, taught=None):
    n = len(edge_kinds) + 1
    for restart in range(180):
        lower = 26 if n == 4 else 23 if n == 3 else 18
        shape = tuple(int(x) for x in rng.integers(lower, 31 if long else 30, 2))
        h, w = shape
        specs = [dict(core=(int(rng.integers(2, h - 4)), int(rng.integers(2, w - 4))), ports=[], color=2)]
        for kind in edge_kinds:
            specs = _add_source(rng, specs, shape, kind, long)
            if specs is None:
                break
        if specs is None:
            continue
        if free_required or (allow_free and rng.random() < .30):
            extra = _add_free_port(rng, specs, shape)
            if extra is not None:
                specs = extra
            elif free_required:
                continue
        try:
            base, sources = _source_grid(shape, specs)
            events, trails, free = _trace(sources, shape)
            _require(set(e['kind'] for e in events) <= set(edge_kinds), 'new collision type')
            # Incremental construction must really expose the requested lessons.
            _require(set(edge_kinds) <= {e['kind'] for e in events}, 'requested collision type absent')
            key_count = len({_key(e['socket']) for e in events})
            if profile is not None:
                _require(n == 3 and key_count == (1 if profile == 'repeat' else 2), 'wrong teaching key profile')
            if not long:
                _require(all(e['distances'][0] != e['distances'][1] for e in events), 'teaching needs unequal approaches')
            if taught is not None:
                _require({(e['kind'], e['mode']) for e in events} <= taught, 'untaught intersection type')
            sockets = set().union(*(e['socket'] for e in events))
            fixed = set().union(*(s['cells'] for s in sources))
            blocked = fixed | set(trails) | sockets
            tokens = []
            for i, event in enumerate(events):
                body = _body(rng, event['socket'], blocked, shape, 14 if long else 11)
                if body is None:
                    raise ValueError('no readable coupler body')
                blocked |= body
                # Distinct coupler colors in the teaching multi-collar case;
                # test colors may repeat without changing object identities.
                color = 6 + i
                if long and i and rng.random() < .45:
                    color = tokens[0]['color']
                token = dict(body=body, pins=event['socket'], color=color, cells=body | event['socket'])
                tokens.append(token)
            a = _scatter(rng, shape, sources, tokens)
            if a is None:
                continue
            output, info = _infer(a)
            _require(all(d != (0, 0) for d in info['placements'].values()), 'stationary loose token')
            if profile == 'distinct':
                _require(_nearest_is_wrong(info), 'nearest-piece shortcut accidentally succeeds')
            # Permute all roles together, including black. Payload identity stays
            # attached to an object; orientation is already varied by geometry.
            palette = rng.permutation(10).astype(np.int8)
            return dict(input=palette[a], output=palette[output])
        except ValueError:
            continue
    raise RuntimeError('bounded collision-collar construction exhausted')


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or not low <= value <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    vocabulary = list(KINDS)
    rng.shuffle(vocabulary)
    k = int(rng.integers(1, min(3, nt) + 1))
    vocabulary = vocabulary[:k]
    curriculum = [[vocabulary[0], vocabulary[0]], [vocabulary[0], vocabulary[min(1, k - 1)]]]
    if k == 3:
        curriculum.append([vocabulary[2]])
    while len(curriculum) < nt:
        curriculum.append([_pick(rng, vocabulary) for _ in range(int(rng.integers(1, 3)))])
    puzzle = {'train': [], 'test': []}
    seen = set()
    taught = set()
    for split, count in (('train', nt), ('test', ne)):
        for index in range(count):
            edges = curriculum[index] if split == 'train' else [_pick(rng, vocabulary) for _ in range(3)]
            for attempt in range(20):
                profile = ('repeat' if index == 0 else 'distinct' if index == 1 else None) if split == 'train' else None
                pair = _make_pair(rng, edges, long=split == 'test', free_required=split == 'train' and index == 0,
                                  profile=profile, taught=taught if split == 'test' else None)
                key = pair['input'].shape, pair['input'].tobytes()
                if key not in seen:
                    break
            else:
                raise RuntimeError('distinct pair construction exhausted')
            puzzle[split].append(pair)
            seen.add(key)
            if split == 'train':
                taught |= {(e['kind'], e['mode']) for e in _infer(pair['input'])[1]['events']}
    rng.shuffle(puzzle['train'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    """Check the delivered grids, not generation-time labels or hidden metadata."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'invalid puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, taught, needed = set(), set(), set()
    multi, repeat, boundary, moved = False, False, False, 0
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'invalid pair keys')
            a, b = _grid(pair['input']), _grid(pair['output'])
            key = a.shape, a.tobytes()
            _require(key not in seen, 'duplicate input')
            seen.add(key)
            expected, info = _infer(a)
            _require(b.shape == a.shape and np.array_equal(b, expected), 'output disagrees with visible input')
            bg, marker = info['background'], info['marker']
            _require(np.count_nonzero(a == marker) == np.count_nonzero(b == marker), 'marker conservation failed')
            _require(not np.array_equal(a, b), 'identity pair')
            for source in info['sources']:
                _require(all(a[q] == b[q] for q in source['cells']), 'stationary source changed')
            for i, token in enumerate(info['tokens']):
                d = info['placements'][i]
                _require(d != (0, 0), 'loose coupler did not move')
                _require(all(b[q] == token['color'] for q in _shift(token['body'], d)), 'rigid payload not preserved')
                _require(all(b[q] == marker for q in _shift(token['pins'], d)), 'complete marker matching failed')
                moved += 1
            tags = {(e['kind'], e['mode']) for e in info['events']}
            if split == 'train':
                taught |= tags
                key_count = len({_key(e['socket']) for e in info['events']})
                multi |= key_count >= 2 and _nearest_is_wrong(info)
                repeat |= len(info['events']) >= 2 and key_count < len(info['events'])
                boundary |= bool(info['free'])
                _require(len(info['sources']) <= 3, 'training count regime')
                _require(all(e['distances'][0] != e['distances'][1] for e in info['events']), 'training lacks unequal approaches')
            else:
                needed |= tags
                _require(len(info['sources']) == 4, 'test must combine three docks')
    _require(multi and repeat and boundary and moved >= 4, 'training does not expose geometry, global docking, and absorption')
    _require(needed <= taught, 'test collision mechanism absent from training')
    return None


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(x[k], y[k]) for x, y in zip(a[s], b[s])
               for k in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed survey, semantic regressions, and count contract checks."""
    shapes, keys, bodies, roles, degrees, kinds, modes, velocities, distances = (set() for _ in range(9))
    signatures = set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        signatures.add(tuple((pair['input'].shape, pair['input'].tobytes()) for s in p for pair in p[s]))
        for split in p:
            for pair in p[split]:
                out, info = _infer(pair['input'])
                assert np.array_equal(out, pair['output'])
                shapes.add(out.shape)
                roles.add((info['background'], info['marker']))
                count = [0] * len(info['sources'])
                for event in info['events']:
                    keys.add(_key(event['socket']))
                    kinds.add(event['kind'])
                    modes.add(event['mode'])
                    velocities.add(event['velocity'])
                    distances.add(event['distances'])
                    for i in event['sources']:
                        count[i] += 1
                degrees.update(count)
                bodies.update(_key(t['body']) for t in info['tokens'])
    assert len(signatures) == 42
    assert len(shapes) >= 25 and len(keys) >= 12 and len(bodies) >= 100
    assert len(roles) >= 30 and kinds == set(KINDS)
    assert modes == {'cell', 'between'} and {1, 2, 3} <= degrees
    assert len(velocities) >= 12 and len(distances) >= 50
    assert _equal(generate_puzzle(17), generate_puzzle(17))
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            _counts(nt, ne)
    for counts in ((2, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(8123, *counts))
    for counts in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4., 1), (None, 1)):
        try:
            generate_puzzle(0, *counts)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    # Hand-derived ray geometries test spatial timing and full vector sums.
    def source(point, direction, color):
        return dict(cells=set(), rays=[(point, direction)], color=color)
    ev, _, _ = _trace([source((10, 4), (-1, 1), 2), source((12, 10), (-1, 0), 3)], (20, 20))
    assert ev[0]['velocity'] == (-2, 1)
    assert ev[0]['distances'] == (6, 8)
    assert ev[0]['socket'] == {(5, 9), (4, 9), (3, 10), (5, 10), (4, 10), (3, 11)}
    ev, _, _ = _trace([source((12, 3), (-1, 1), 2), source((10, 12), (-1, -1), 3)], (20, 20))
    assert ev[0]['mode'] == 'between' and ev[0]['distances'] == (5.5, 3.5)
    assert ev[0]['socket'] == {(7, 8), (6, 8), (5, 8), (7, 9), (6, 9), (5, 9)}
    # Equal marker masks with two differently colored interchangeable payloads
    # must be rejected: local signature matches are not a uniqueness proof.
    specs = [dict(core=(10, 10), ports=[(0, 2), (1, -1)], color=2),
             dict(core=(16, 17), ports=[(-1, 0)], color=3),
             dict(core=(4, 4), ports=[(2, 1)], color=4)]
    _, sources = _source_grid((25, 25), specs)
    events, _, _ = _trace(sources, (25, 25))
    assert _key(events[0]['socket']) == _key(events[1]['socket'])
    tokens = []
    for i, event in enumerate(events):
        r, c, _, _ = _box(event['socket'])
        body = {(r + 1, c + 2)}
        tokens.append(dict(body=body, pins=event['socket'], cells=body | event['socket'], color=6 + i))
    ambiguous = _scatter(np.random.default_rng(912), (25, 25), sources, tokens)
    try:
        solve(ambiguous)
    except ValueError as error:
        assert 'no unique output' in str(error)
    else:
        raise AssertionError('interchangeable repeated-key pieces accepted')
    p = generate_puzzle(0)
    bad = {s: [{k: np.array(v, copy=True) for k, v in pair.items()} for pair in p[s]] for s in p}
    _, info = _infer(bad['test'][0]['input'])
    q = min(info['events'][0]['socket'])
    bad['test'][0]['output'][q] = info['background']
    try:
        validate_generated_puzzle(bad)
    except ValueError:
        pass
    else:
        raise AssertionError('erased docking marker accepted')
    return {'seeds': 42, 'puzzles': len(signatures), 'shapes': len(shapes), 'socket_keys': len(keys),
            'payload_shapes': len(bodies), 'palettes': len(roles), 'headings': len(velocities),
            'distance_pairs': len(distances), 'collision_kinds': sorted(kinds), 'intersection_modes': sorted(modes)}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
