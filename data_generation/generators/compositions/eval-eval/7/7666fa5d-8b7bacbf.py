"""Paint bridges: diagonal interval geometry supplies a visible route graph.

Only standard-library modules and NumPy are used. The solver reads geometry
and paint cups from its one input; it has no access to the generator's plan.
"""
import heapq
import numpy as np

D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _components(cells, steps=D8):
    left = set(cells)
    result = []
    while left:
        start = min(left)
        left.remove(start)
        group, queue = {start}, [start]
        for r, c in queue:
            for dr, dc in steps:
                p = r + dr, c + dc
                if p in left:
                    left.remove(p)
                    group.add(p)
                    queue.append(p)
        result.append(group)
    return result


def _neighbors(cells, shape, steps=D8):
    h, w = shape
    cells = set(cells)
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps
            if 0 <= r + dr < h and 0 <= c + dc < w} - cells


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(int(v) for v in a.flat)


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((a >= 0) & (a <= 9))):
        raise ValueError('Expected an integer ARC grid with dimensions 1..30')
    return a.astype(np.int8, copy=True)


def _cup(rotation):
    # A five-neighbor U, open toward the right before rotation.
    source = (0, 0)
    wire = {(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0)}
    # Coordinates are (row, column).
    for _ in range(rotation):
        wire = {(-c, r) for r, c in wire}
    return source, wire


_CUPS = [_cup(k)[1] for k in range(4)]


def _read_input(value):
    """Recover typed objects and their actual attachment graph from input."""
    a = _grid(value)
    values, counts = np.unique(a, return_counts=True)
    bg = int(values[np.argmax(counts)])
    if int(counts.max()) <= a.size // 2:
        raise ValueError('The background must occupy more than half the grid')
    by_color = {int(c): _components(map(tuple, np.argwhere(a == c)))
                for c in values if c != bg}
    candidates = []
    for color, groups in by_color.items():
        if len(groups) < 2 or any(len(g) < 2 for g in groups):
            continue
        for slope in (-1, 1):
            if all(len({c - slope * r for r, c in g}) == 1
                   and max(r for r, c in g) - min(r for r, c in g) + 1 == len(g)
                   for g in groups):
                candidates.append((color, slope, groups))
    if len(candidates) != 1:
        raise ValueError('Expected one recognizable role of parallel diagonal rails')
    rail_color, slope, rail_cells = candidates[0]
    rails = []
    for cells in rail_cells:
        r, c = next(iter(cells))
        q = c - slope * r
        ts = [r + slope * c for r, c in cells]
        rails.append(dict(cells=cells, q=q, lo=min(ts), hi=max(ts)))
    rails.sort(key=lambda x: (x['q'], x['lo']))

    background_components = _components(map(tuple, np.argwhere(a == bg)), D4)
    cavities = []
    exterior = set()
    for cells in background_components:
        if any(r in (0, a.shape[0] - 1) or c in (0, a.shape[1] - 1)
               for r, c in cells):
            exterior |= cells
        else:
            shell = _neighbors(cells, a.shape, D4)
            colors = {int(a[p]) for p in shell}
            if len(colors) != 1 or bg in colors or rail_color in colors:
                raise ValueError('Every enclosed background region needs a uniform contour')
            cavities.append(dict(cells=cells, shell=shell, color=colors.pop()))
    if not cavities or len({x['color'] for x in cavities}) != 1:
        raise ValueError('Expected nonempty reservoirs of one contour role')
    contour_color = cavities[0]['color']
    shells = [x['shell'] for x in cavities]
    if (any(x & y for i, x in enumerate(shells) for y in shells[:i]) or
            set().union(*shells) != set(map(tuple, np.argwhere(a == contour_color)))):
        raise ValueError('Contour pixels must form separate minimal cavity shells')
    sources = []
    for color, groups in by_color.items():
        if color in (rail_color, contour_color):
            continue
        for group in groups:
            if len(group) != 1:
                continue
            p = next(iter(group))
            neighbors = {q for q in _neighbors({p}, a.shape) if a[q] != bg}
            offsets = {(r - p[0], c - p[1]) for r, c in neighbors}
            colors = {int(a[q]) for q in neighbors}
            if offsets in _CUPS and len(colors) == 1:
                wire_color = colors.pop()
                if wire_color not in (color, rail_color, contour_color, bg):
                    sources.append(dict(cell=p, paint=color, wire=wire_color))
    if not sources or len({s['wire'] for s in sources}) != 1:
        raise ValueError('Expected five-neighbor paint cups of one wire role')
    wire_color = sources[0]['wire']
    source_cells = {s['cell'] for s in sources}
    if set(int(c) for c in values) != ({bg, rail_color, contour_color, wire_color}
                                       | {s['paint'] for s in sources}):
        raise ValueError('Unexpected color role')
    for color in {s['paint'] for s in sources}:
        if set(map(tuple, np.argwhere(a == color))) != {s['cell'] for s in sources
                                                      if s['paint'] == color}:
            raise ValueError('Paint colors occur only in cups in the input')
    wires = by_color[wire_color]
    nodes = [s['cells'] for s in rails] + [x['shell'] for x in cavities] + wires
    nr, nc = len(rails), len(cavities)
    graph = [set() for _ in nodes]
    actual = [set() for _ in nodes]
    def link(g, i, j):
        g[i].add(j)
        g[j].add(i)
    # Only wires make visible attachments. Contour-to-contour and direct
    # rail-to-contour proximity never transmit paint.
    for i in range(nr + nc):
        near = _neighbors(nodes[i], a.shape)
        for j in range(nr + nc, len(nodes)):
            if near & nodes[j]:
                link(graph, i, j)
                link(actual, i, j)
    for i, rail in enumerate(rails):
        for j in range(i):
            other = rails[j]
            if (rail['q'] != other['q'] and
                    max(rail['lo'], other['lo']) <= min(rail['hi'], other['hi'])):
                link(graph, i, j)
    for source in sources:
        neighbors = _neighbors({source['cell']}, a.shape)
        attached = [j for j in range(nr + nc, len(nodes)) if nodes[j] & neighbors]
        if len(attached) != 1:
            raise ValueError('Each paint cup must belong to exactly one string')
        source['node'] = attached[0]
    info = dict(a=a, bg=bg, rail_color=rail_color, contour_color=contour_color,
                wire_color=wire_color, slope=slope, rails=rails, cavities=cavities,
                exterior=exterior, sources=sources, nodes=nodes, graph=graph, actual=actual)
    return info


def _supplies(info, graph=None):
    graph = info['graph'] if graph is None else graph
    supplied = {}
    for source in info['sources']:
        seen, queue = {source['node']}, [source['node']]
        for i in queue:
            for j in sorted(graph[i] - seen):
                seen.add(j)
                queue.append(j)
        for i in seen:
            if i in supplied and supplied[i] != source['paint']:
                raise ValueError('Different paints may not supply the same network')
            supplied[i] = source['paint']
    return supplied


def _predict(info, alternative=''):
    """Union of pairwise parallelograms; no generator parameters are used."""
    a = info['a']
    out = a.copy()
    graph = info['actual'] if alternative == 'no_virtual' else None
    if alternative == 'no_relay':
        graph = [set(x) for x in info['graph']]
        for i in range(len(info['rails']), len(info['rails']) + len(info['cavities'])):
            graph[i].clear()  # A reservoir receives paint but cannot relay it.
    nr = len(info['rails'])
    rows, cols = np.indices(a.shape)
    q = cols - info['slope'] * rows
    t = rows + info['slope'] * cols
    if alternative == 'pixel_contact':
        # Deliberately wrong: treat the displayed span as new foreground and
        # allow its pixel contacts to attach arbitrary strings or contours.
        graph = [set(x) for x in info['graph']]
        for i, x in enumerate(info['rails']):
            for y in info['rails'][:i]:
                mask = ((t >= max(x['lo'], y['lo'])) & (t <= min(x['hi'], y['hi']))
                        & (q > min(x['q'], y['q'])) & (q < max(x['q'], y['q'])))
                band = set(map(tuple, np.argwhere(mask)))
                near = band | _neighbors(band, a.shape)
                for j, node in enumerate(info['nodes']):
                    if node & near:
                        graph[i].add(j)
                        graph[j].add(i)
    supplied = _supplies(info, graph)
    exterior = np.zeros(a.shape, dtype=bool)
    for p in info['exterior']:
        exterior[p] = True
    rails = info['rails']
    pairs = [(i, j) for i in range(nr) for j in range(i)]
    if alternative == 'consecutive':
        # Static neighboring ranks within each geometric overlap component.
        g = [set() for _ in rails]
        for i, j in pairs:
            if max(rails[i]['lo'], rails[j]['lo']) <= min(rails[i]['hi'], rails[j]['hi']):
                g[i].add(j); g[j].add(i)
        groups, left = [], set(range(nr))
        while left:
            root = min(left); left.remove(root)
            seen, queue = {root}, [root]
            for i in queue:
                for j in sorted(g[i] & left):
                    left.remove(j); seen.add(j); queue.append(j)
            groups.append(sorted(seen, key=lambda i: rails[i]['q']))
        pairs = [(j, i) for group in groups for i, j in zip(group, group[1:])]
    if alternative != 'no_virtual':
        for i, j in pairs:
            x, y = rails[i], rails[j]
            if i not in supplied or j not in supplied:
                continue
            mask = (exterior & (t >= max(x['lo'], y['lo'])) &
                    (t <= min(x['hi'], y['hi'])) &
                    (q > min(x['q'], y['q'])) & (q < max(x['q'], y['q'])))
            if alternative == 'pixel_centers':
                mask &= ((t - x['q']) % 2 == 0) & ((t - y['q']) % 2 == 0)
            out[mask] = supplied[i]
    for i, cavity in enumerate(info['cavities'], nr):
        if i in supplied:
            for p in cavity['cells']:
                out[p] = supplied[i]
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve entirely from input-visible rails, contour ports and paint cups."""
    return _predict(_read_input(input_grid))


def _ri(rng, lo, hi):
    return int(rng.integers(lo, hi + 1))


def _stroke_plan(rng, counts, skip, compact):
    specs, groups = [], []
    for group_index, count in enumerate(counts):
        qs = [2 * _ri(rng, -2, 2)]
        for _ in range(count - 1):
            qs.append(qs[-1] + _ri(rng, 4, 5 if compact else 8))
        current = []
        style = int(rng.integers(3))
        for rank, transverse in enumerate(qs):
            length = _ri(rng, 2, 4 if compact else 7)
            if style == 0:
                start = transverse % 2 - 2 * _ri(rng, 0, length - 2)
            elif style == 1:
                start = transverse % 2
            else:
                start = 2 * rank + transverse % 2
            current.append([transverse, start, start + 2 * (length - 1)])
        if skip and group_index == 0:
            # Long outer rails, a short middle rail. Outer overlap persists
            # after the middle rail ends, with both endpoint phases possible.
            for rank in (0, count - 1):
                transverse = qs[rank]
                lo = -6 + transverse % 2
                current[rank] = [transverse, lo, lo + 12]
            rank = _ri(rng, 1, count - 2)
            current[rank] = [qs[rank], qs[rank] % 2, qs[rank] % 2 + 2]
        # Ensure every intended island is geometrically connected; no hidden
        # visual-distance clustering convention is used.
        if max(x[1] for x in current) > min(x[2] for x in current) and style != 2:
            return None
        adjacency = [set() for _ in current]
        for i, x in enumerate(current):
            for j, y in enumerate(current[:i]):
                if max(x[1], y[1]) <= min(x[2], y[2]):
                    adjacency[i].add(j); adjacency[j].add(i)
        seen, queue = {0}, [0]
        for i in queue:
            for j in adjacency[i] - seen:
                seen.add(j); queue.append(j)
        if len(seen) != count:
            return None
        shift = 0 if not specs else max(x[2] for x in specs) + _ri(rng, 3, 6) - min(x[1] for x in current)
        shift += shift % 2
        ids = []
        for transverse, lo, hi in current:
            ids.append(len(specs))
            specs.append((transverse, lo + shift, hi + shift))
        groups.append(ids)
    return specs, groups


def _contour(rng):
    """Connected row profiles, with a minimal orthogonal shell."""
    kind = int(rng.integers(4))
    h, w = _ri(rng, 1, 4), _ri(rng, 1, 4)
    cells = set()
    if kind == 1:
        radius = _ri(rng, 1, 2)
        cells = {(r, c) for r in range(-radius, radius + 1)
                 for c in range(-radius, radius + 1) if abs(r) + abs(c) <= radius}
    else:
        start = 0
        for r in range(h):
            if kind >= 2 and r:
                start += _ri(rng, -1, 1)
            width = max(2, w + _ri(rng, -1, 1)) if kind >= 2 else w
            cells.update((r, c) for c in range(start, start + width))
    if rng.random() < .5:
        cells = {(c, r) for r, c in cells}
    shell = {(r + dr, c + dc) for r, c in cells for dr, dc in D4} - cells
    mr, mc = min(r for r, c in shell), min(c for r, c in shell)
    return ({(r - mr, c - mc) for r, c in cells},
            {(r - mr, c - mc) for r, c in shell})


def _path(rng, starts, goals, allowed, shape):
    """Random weighted shortest path, giving bent and diagonal strings."""
    starts, goals = set(starts), set(goals)
    if not starts or not goals:
        return None
    costs = rng.uniform(1., 3., size=shape)
    previous = {p: None for p in starts}
    distance = {p: 0. for p in starts}
    heap = [(0., r, c) for r, c in sorted(starts)]
    heapq.heapify(heap)
    end = None
    while heap:
        cost, r, c = heapq.heappop(heap)
        p = r, c
        if cost != distance[p]:
            continue
        if p in goals:
            end = p
            break
        for dr, dc in D8:
            q = r + dr, c + dc
            if q not in allowed:
                continue
            new = cost + float(costs[q])
            if new < distance.get(q, float('inf')):
                distance[q] = new
                previous[q] = p
                heapq.heappush(heap, (new, q[0], q[1]))
    if end is None:
        return None
    route = set()
    while end is not None:
        route.add(end)
        end = previous[end]
    return route


def _scene(rng, mode, paints, roles, force_skip=False, teach_contact=False):
    """Build one combined circuit, then check the reconstructed graph."""
    bg, rail_color, wire_color, contour_color = roles
    if mode == 'single':
        counts = [_ri(rng, 2, 4)]
        owners = [paints[0]]
    elif mode == 'inactive':
        counts = [_ri(rng, 2, 3), _ri(rng, 2, 3)]
        owners = [paints[0], None]
    elif mode == 'multi':
        counts = [_ri(rng, 2, 3), _ri(rng, 2, 3)]
        owners = paints[:2]
    elif mode == 'relay':
        counts = [_ri(rng, 3, 4) if force_skip else _ri(rng, 2, 4), _ri(rng, 2, 3)]
        owners = [paints[0], paints[0]]
    else:  # Three geometric islands: relay plus an independent circuit.
        counts = [3 if force_skip else 2, 2, 2]
        owners = [paints[0], paints[0], paints[1]]
    compact = len(counts) == 3
    plan = _stroke_plan(rng, counts, force_skip, compact)
    if plan is None:
        return None
    specs, groups = plan
    rails = [{((t - q) // 2, (q + t) // 2) for t in range(lo, hi + 1, 2)}
             for q, lo, hi in specs]
    allrails = set().union(*rails)
    r0, c0 = min(r for r, c in allrails), min(c for r, c in allrails)
    rh = max(r for r, c in allrails) - r0 + 1
    cw = max(c for r, c in allrails) - c0 + 1
    low = 13 if mode == 'single' else 22
    if rh + 1 > 30 or cw + 1 > 30:
        return None
    h, w = _ri(rng, max(low, rh + 1), 30), _ri(rng, max(low, cw + 1), 30)
    rr, cc = _ri(rng, 0, h - rh), _ri(rng, 0, w - cw)
    rails = [{(r - r0 + rr, c - c0 + cc) for r, c in s} for s in rails]
    specs = []
    for cells in rails:
        p = next(iter(cells))
        ts = [r + c for r, c in cells]
        specs.append((p[1] - p[0], min(ts), max(ts)))
    allrails = set().union(*rails)
    shape, canvas = (h, w), {(r, c) for r in range(h) for c in range(w)}
    rows, cols = np.indices(shape)
    qcoords, tcoords = cols - rows, rows + cols
    bands = []
    for ids in groups:
        band = np.zeros(shape, dtype=bool)
        for t in range(h + w - 1):
            active = [specs[i][0] for i in ids if specs[i][1] <= t <= specs[i][2]]
            if len(active) >= 2:
                band |= ((tcoords == t) & (qcoords > min(active)) & (qcoords < max(active)))
        bands.append(band)
    band_cells = set(map(tuple, np.argwhere(np.logical_or.reduce(bands))))

    # Planned reservoirs and ports. The relay shell has two separate strings:
    # it carries supply onward from one geometric island to the next.
    loop_specs = []
    if mode in ('relay', 'complex'):
        loop_specs.append((owners[0], [groups[0][-1], groups[1][0]]))
        loop_specs.append((owners[1], [groups[1][-1]]))
        if mode == 'complex':
            loop_specs.append((owners[2], [groups[2][-1]]))
    else:
        for owner, ids in zip(owners, groups):
            loop_specs.append((owner, [ids[-1]]))
    if mode == 'inactive' or rng.random() < .55:
        loop_specs.append((None, []))
    if mode == 'single' and rng.random() < .45:
        loop_specs.append((owners[0], [groups[0][_ri(rng, 0, len(groups[0]) - 1)]]))
    # Random order prevents a productive-first placement convention.
    rng.shuffle(loop_specs)
    loops, occupied = [], set(allrails)
    for owner, ports in loop_specs:
        inner, shell = _contour(rng)
        lh, lw = max(r for r, c in shell) + 1, max(c for r, c in shell) + 1
        if lh > h or lw > w:
            return None
        placed = False
        for _ in range(60):
            y, x = _ri(rng, 0, h - lh), _ri(rng, 0, w - lw)
            ins = {(r + y, c + x) for r, c in inner}
            edge = {(r + y, c + x) for r, c in shell}
            cells = ins | edge
            if cells & (occupied | band_cells) or _neighbors(cells, shape) & occupied:
                continue
            loops.append(dict(inner=ins, shell=edge, owner=owner, ports=ports))
            occupied |= cells
            placed = True
            break
        if not placed:
            return None
    source_specs = [(owners[0], groups[0][0])]
    if mode in ('multi', 'complex'):
        source_specs.append((owners[-1], groups[-1][0]))
    rng.shuffle(source_specs)
    sources, allwire = [], set()
    for paint, port in source_specs:
        offsets = _CUPS[_ri(rng, 0, 3)]
        placed = False
        for _ in range(80):
            p = _ri(rng, 0, h - 1), _ri(rng, 0, w - 1)
            cup = {(p[0] + dr, p[1] + dc) for dr, dc in offsets}
            cells = cup | {p}
            if not cells <= canvas or cells & (occupied | band_cells):
                continue
            if _neighbors(cells, shape) & occupied:
                continue
            sources.append(dict(p=p, cup=cup, paint=paint, port=port))
            occupied |= cells
            allwire |= cup
            placed = True
            break
        if not placed:
            return None
    source_cells = {s['p'] for s in sources}
    objects = rails + [q['inner'] | q['shell'] for q in loops]
    object_union = set().union(*objects)
    jobs = []
    for source in sources:
        jobs.append(('source', source, source['port'], None))
    for index, loop in enumerate(loops):
        for port in loop['ports']:
            jobs.append(('loop', None, port, index))
    # Randomized order and free routing produce many geometric arrangements.
    rng.shuffle(jobs)
    for kind, source, rail_index, loop_index in jobs:
        first = rails[rail_index]
        second = source['cup'] if kind == 'source' else loops[loop_index]['shell']
        second_object = second if kind == 'source' else objects[len(rails) + loop_index]
        other_objects = object_union - first - second_object
        forbidden = object_union | _neighbors(other_objects, shape)
        forbidden |= source_cells | _neighbors(source_cells, shape)
        reserved_wire = allwire - second if kind == 'source' else allwire
        forbidden |= reserved_wire | _neighbors(reserved_wire, shape)
        if kind == 'source':
            # The open side of the cup remains open; the route can depart
            # from any cup pixel without adding a sixth source neighbor.
            own = source['cup']
            forbidden -= own
            starts = own
        else:
            starts = _neighbors(second, shape) - forbidden
        goals = _neighbors(first, shape) - forbidden
        allowed = canvas - forbidden
        allowed |= starts
        route = _path(rng, starts, goals, allowed, shape)
        if route is None:
            return None
        allwire |= route
    a = np.full(shape, bg, dtype=np.int8)
    for cells in rails:
        for p in cells:
            a[p] = rail_color
    for loop in loops:
        for p in loop['shell']:
            a[p] = contour_color
    for p in allwire:
        a[p] = wire_color
    for source in sources:
        a[source['p']] = source['paint']
    # Labels come from the intended island supply and reservoirs, with a
    # longitudinal sweep. solve independently reconstructs attachments and
    # unions pairwise parallelograms from the returned input.
    b = a.copy()
    for owner, band in zip(owners, bands):
        if owner is not None:
            b[band & (a == bg)] = owner
    for loop in loops:
        if loop['owner'] is not None:
            for p in loop['inner']:
                b[p] = loop['owner']
    rotation = _ri(rng, 0, 3)
    a, b = np.rot90(a, rotation), np.rot90(b, rotation)
    if rng.random() < .5:
        a, b = np.fliplr(a), np.fliplr(b)
    a, b = a.copy(), b.copy()
    try:
        info = _read_input(a)
        predicted = _predict(info)
        if not np.array_equal(predicted, b) or len(info['cavities']) != len(loops):
            return None
        f = _features(info, b)
    except ValueError:
        return None
    if not f['remote_cavities'] or not f['bridge_cells']:
        return None
    if mode == 'inactive' and not f['unpowered_bracket']:
        return None
    if force_skip and (not f['consecutive_witness'] or not f['phase_witness']):
        return None
    if teach_contact and not f['contact_witness']:
        return None
    return {'input': a, 'output': b}


def _features(info, out):
    supplies = _supplies(info)
    physical = _supplies(info, info['actual'])
    nr = len(info['rails'])
    remote = sum(nr + i in supplies and nr + i not in physical
                 for i in range(len(info['cavities'])))
    changed = info['a'] != out
    cavity_cells = set().union(*(c['cells'] for c in info['cavities']))
    changed_cells = set(map(tuple, np.argwhere(changed)))
    rows, cols = np.indices(info['a'].shape)
    transverse, longitudinal = cols - info['slope'] * rows, rows + info['slope'] * cols
    unpowered_pixels = np.zeros(info['a'].shape, dtype=bool)
    for i, x in enumerate(info['rails']):
        for j, y in enumerate(info['rails'][:i]):
            if i not in supplies and j not in supplies:
                unpowered_pixels |= ((info['a'] == info['bg']) &
                    (longitudinal >= max(x['lo'], y['lo'])) &
                    (longitudinal <= min(x['hi'], y['hi'])) &
                    (transverse > min(x['q'], y['q'])) &
                    (transverse < max(x['q'], y['q'])))
    unpowered = bool(np.any(unpowered_pixels))
    try:
        contact_witness = not np.array_equal(_predict(info, 'pixel_contact'), out)
    except ValueError:
        contact_witness = True  # The wrong rule mixes two independent paints.
    return dict(rails=nr, contours=len(info['cavities']), sources=len(info['sources']),
                slope=info['slope'], remote_cavities=remote,
                bridge_cells=len(changed_cells - cavity_cells),
                unpowered_bracket=unpowered,
                unpowered_bridge_cells=int(np.sum(unpowered_pixels)),
                detached=sum(not info['graph'][nr + i] for i in range(len(info['cavities']))),
                contour_relay=any(len(info['graph'][nr + i]) >= 2
                                  for i in range(len(info['cavities']))),
                relay_witness=not np.array_equal(_predict(info, 'no_relay'), out),
                contact_witness=contact_witness,
                consecutive_witness=not np.array_equal(_predict(info, 'consecutive'), out),
                phase_witness=not np.array_equal(_predict(info, 'pixel_centers'), out),
                paints=sorted({s['paint'] for s in info['sources']}),
                lengths=[len(s['cells']) for s in info['rails']],
                cavity_areas=[len(x['cells']) for x in info['cavities']],
                wire_pixels=sum(len(x) for x in info['nodes'][nr + len(info['cavities']):]),
                background=info['bg'], roles=(info['rail_color'], info['wire_color'], info['contour_color']))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
                or not low <= int(value) <= high):
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if (not isinstance(seed, (int, np.integer)) or isinstance(seed, (bool, np.bool_))
            or seed < 0):
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    palette = [int(c) for c in rng.permutation(10)]
    paints, structural = palette[:3], palette[3:]
    changing_roles = bool(rng.integers(2))
    fixed_roles = structural[:4]
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            if split == 'train' and index == 0:
                mode, chosen, skip = 'inactive', paints, False
            elif split == 'train' and index == 1:
                mode, chosen, skip = 'relay', [paints[1], paints[0]], True
            elif split == 'test':
                mode = str(rng.choice(['multi', 'relay', 'complex']))
                chosen = [paints[2], paints[int(rng.integers(2))]]
                skip = mode == 'relay' and rng.random() < .65
            else:
                mode = str(rng.choice(['single', 'inactive', 'multi', 'relay', 'complex']))
                chosen = [paints[i] for i in rng.permutation(2)]
                skip = mode == 'relay' and rng.random() < .4
            if mode == 'multi' and rng.random() < .2:
                # Separate sources may copy the same paint without becoming
                # one physical circuit or relying on a globally unique color.
                chosen = [chosen[0], chosen[0]]
            roles = [int(c) for c in rng.permutation(structural)[:4]] if changing_roles else fixed_roles
            for _ in range(200):
                pair = _scene(rng, mode, chosen, roles, skip,
                              teach_contact=(split == 'train' and index == 0))
                if pair is not None and _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError(f'Bounded placement exhausted for {split} {index}, mode {mode}')
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape and _key(a) not in seen
            seen.add(_key(a))
            info = _read_input(a)
            prediction = _predict(info)
            assert np.array_equal(prediction, b), 'Input-derived supply/bridge/cavity mismatch'
            assert np.any(a != b) and np.all(a[a != b] == info['bg'])
            assert np.array_equal(a[a != info['bg']], b[a != info['bg']])
            # Contours must be genuine closed reservoirs; each is atomic.
            for i, cavity in enumerate(info['cavities'], len(info['rails'])):
                cells = sorted(cavity['cells'])
                vals = {int(b[p]) for p in cells}
                assert len(vals) == 1
                for x in info['rails']:
                    for y in info['rails']:
                        assert not any(max(x['lo'], y['lo']) <= r + info['slope'] * c <= min(x['hi'], y['hi'])
                                       and min(x['q'], y['q']) < c - info['slope'] * r < max(x['q'], y['q'])
                                       for r, c in cavity['cells'] | cavity['shell']), 'Reservoir overlaps a virtual span'
            f = _features(info, b)
            assert f['remote_cavities'] >= 1 and f['bridge_cells'] >= 1
            features[split].append(f)
    train = features['train']
    assert len({c for f in train for c in f['paints']}) >= 2, 'Teach copying two paints'
    taught_paints = {c for f in train for c in f['paints']}
    assert all(set(f['paints']) - taught_paints for f in features['test']), 'Tests must copy a new visible paint'
    assert any(f['unpowered_bracket'] and f['detached'] for f in train), 'Teach unsupplied preservation'
    assert any(f['contact_witness'] for f in train), 'Teach virtual spans are endpoint links'
    assert any(f['relay_witness'] for f in train), 'Teach propagation through reservoir ports'
    for mechanism in ('consecutive_witness', 'phase_witness', 'relay_witness', 'contact_witness'):
        if any(f[mechanism] for f in features['test']):
            assert any(f[mechanism] for f in train), f'Missing {mechanism} demonstration'


def self_test():
    """Bounded 42-seed audit plus fixtures, count boundaries and bad-label tests."""
    # Hand-computed fixture: two length-three rails bracket exactly seven
    # exterior pixels. The second rail's reservoir adds four more pixels;
    # the equal-colored detached reservoir stays empty. No expected-output
    # construction helper is involved in these eleven asserted cells.
    fixture = np.zeros((12, 15), dtype=np.int8)
    for p in ((4, 6), (5, 5), (6, 4), (6, 8), (7, 7), (8, 6)):
        fixture[p] = 1
    for p in ((0, 0), (0, 1), (1, 0), (2, 0), (2, 1),
              (3, 2), (3, 3), (4, 4), (6, 9), (6, 10), (7, 10)):
        fixture[p] = 3
    for p in ((7, 11), (7, 12), (10, 11), (10, 12),
              (8, 10), (9, 10), (8, 13), (9, 13),
              (0, 11), (0, 12), (3, 11), (3, 12),
              (1, 10), (2, 10), (1, 13), (2, 13)):
        fixture[p] = 5
    fixture[1, 1] = 2
    expected = fixture.copy()
    for p in ((5, 6), (5, 7), (6, 5), (6, 6), (6, 7), (7, 5), (7, 6),
              (8, 11), (8, 12), (9, 11), (9, 12)):
        expected[p] = 2
    assert np.array_equal(solve(fixture), expected)
    assert _features(_read_input(fixture), expected)['remote_cavities'] == 1
    assert not np.array_equal(_predict(_read_input(fixture), 'pixel_centers'), expected)
    assert not np.array_equal(_predict(_read_input(fixture), 'no_virtual'), expected)
    permutation = np.array([8, 4, 0, 9, 6, 1, 2, 3, 7, 5], dtype=np.int8)
    assert np.array_equal(solve(permutation[fixture]), permutation[expected])
    signatures, shapes, structures, records = set(), set(), set(), []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(tuple(_key(p['input']) for s in ('train', 'test') for p in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                a = pair['input']
                shapes.add(a.shape)
                roles = {}
                structures.add((a.shape, tuple(roles.setdefault(int(c), len(roles)) for c in a.flat)))
                records.append(_features(_read_input(a), pair['output']))
    assert len(signatures) == 42 and len(structures) >= 180 and len(shapes) >= 30
    assert {r['slope'] for r in records} == {-1, 1}
    assert {1, 2} <= {r['sources'] for r in records}
    assert {2, 3, 4, 5, 6, 7} <= {r['rails'] for r in records}
    assert len({r['background'] for r in records}) >= 8
    assert len({r['roles'] for r in records}) >= 20
    assert max(n for r in records for n in r['cavity_areas']) >= 12
    assert min(n for r in records for n in r['cavity_areas']) == 1
    assert max(n for r in records for n in r['lengths']) >= 7
    for train, test in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(817, train, test))
    for train, test in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5)):
        try:
            generate_puzzle(0, train, test)
        except ValueError:
            continue
        raise AssertionError('Illegal counts accepted')
    p = generate_puzzle(0, 2, 1)
    repeated = generate_puzzle(0, 2, 1)
    assert all(np.array_equal(p[s][i][role], repeated[s][i][role])
               for s in p for i in range(len(p[s])) for role in ('input', 'output'))
    # Every isometry preserves the rule. This also checks input-only parsing
    # when paint cups, attachments and a relay are reflected together.
    for pair in p['train'] + p['test']:
        for rotation in range(4):
            for reflect in (False, True):
                a, b = np.rot90(pair['input'], rotation), np.rot90(pair['output'], rotation)
                if reflect:
                    a, b = np.fliplr(a), np.fliplr(b)
                assert np.array_equal(solve(a), b)
    bad = {s: [{role: a.copy() for role, a in pair.items()} for pair in p[s]] for s in p}
    pair = bad['test'][0]
    point = tuple(np.argwhere(pair['input'] != pair['output'])[0])
    pair['output'][point] = pair['input'][point]
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('A partial or missing fill was accepted')


if __name__ == '__main__':
    self_test()
    print('42-seed self-test passed')
