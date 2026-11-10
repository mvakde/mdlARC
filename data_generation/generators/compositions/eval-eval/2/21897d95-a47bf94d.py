"""Routed Weaving: simultaneous, component-local dye on complementary phases.

The constructor retains room IDs and actual paths for its answers. The input
reader instead recognizes local glyphs, restores rooms, and traces visible wires.
No parent module, seed, example, or construction metadata is used by solve.
"""
import heapq
import json

import numpy as np


D = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIAMOND = {(-1, 0), (0, -1), (0, 1), (1, 0)}
CHECKER = {(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)}


def _need(condition, message):
    if not condition:
        raise AssertionError(message)


def _add(p, d, k=1):
    return p[0] + d[0] * k, p[1] + d[1] * k


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _grid(value):
    a = np.asarray(value)
    _need(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), 'invalid dimensions')
    _need(np.issubdtype(a.dtype, np.integer), 'noninteger grid')
    _need(bool(np.all((0 <= a) & (a <= 9))), 'invalid colors')
    return a.astype(np.int8, copy=True)


def _components(clean, excluded):
    ids = np.full(clean.shape, -1, dtype=np.int16)
    rooms = []
    for p in np.ndindex(clean.shape):
        if ids[p] >= 0 or int(clean[p]) in excluded:
            continue
        color, i = int(clean[p]), len(rooms)
        cells, stack = set(), [p]
        ids[p] = i
        while stack:
            q = stack.pop()
            cells.add(q)
            for d in D:
                t = _add(q, d)
                if _inside(t, clean.shape) and ids[t] < 0 and clean[t] == color:
                    ids[t] = i
                    stack.append(t)
        rooms.append(dict(color=color, cells=cells))
    return ids, rooms


def _local_glyphs(g):
    """Find T launchers and phase sockets using their unmarked host pixels.

    The full 3x3 footprint is inside a room. Of the four points two steps
    from its center, exactly the outward one is a wire; the other three
    retain the host color. Thus socket orientation and wire role are visible.
    """
    found = []
    h, w = g.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = (r, c)
            patch = {(r + a, c + b) for a in range(-1, 2) for b in range(-1, 2)}
            if not all(_inside(_add(center, d, 2), g.shape) for d in D):
                continue
            for direction, d in enumerate(D):
                normal = _add(center, d, 2)
                support = [_add(center, dd, 2) for k, dd in enumerate(D) if k != direction]
                if len({int(g[p]) for p in support}) != 1:
                    continue
                host, wire = int(g[support[0]]), int(g[normal])
                if host == wire:
                    continue
                perp = (-d[1], d[0])
                t_cells = {center, _add(center, perp), _add(center, perp, -1), _add(center, d)}
                for kind, offsets in (('T', None), ('diamond', DIAMOND), ('checker', CHECKER)):
                    cells = t_cells if offsets is None else {(r + a, c + b) for a, b in offsets}
                    read_ink = cells - {center} if kind == 'T' else cells
                    ink_values = {int(g[p]) for p in read_ink}
                    if len(ink_values) != 1:
                        continue
                    ink = next(iter(ink_values))
                    if ink in (host, wire):
                        continue
                    if not all(g[p] == host for p in patch - cells):
                        continue
                    payload = int(g[center]) if kind == 'T' and g[center] != ink else host
                    found.append(dict(kind=kind, center=center, d=direction, port=normal,
                                      host=host, ink=ink, wire=wire, cells=cells,
                                      patch=patch, payload=payload,
                                      explicit=kind == 'T' and int(g[center]) != ink))
    return found


def _read(g):
    corners = [int(g[p]) for p in ((0, 0), (0, -1), (-1, 0), (-1, -1))]
    _need(len(set(corners)) == 1, 'background corners disagree')
    bg = corners[0]
    candidates, answers = _local_glyphs(g), []
    roles = sorted({(a['ink'], a['wire']) for a in candidates if a['kind'] == 'T'})
    for ink, wire in roles:
        if bg in (ink, wire):
            continue
        glyphs = [a for a in candidates if a['ink'] == ink and a['wire'] == wire]
        launchers = [a for a in glyphs if a['kind'] == 'T']
        receivers = [a for a in glyphs if a['kind'] != 'T']
        if not launchers or len(launchers) != len(receivers):
            continue
        try:
            writes, covered, ports = set(), set(), {}
            clean = g.copy()
            for a in glyphs:
                _need(not writes & a['cells'], 'overlapping glyphs')
                _need(a['port'] not in ports, 'shared cable port')
                writes.update(a['cells'])
                covered.update(p for p in a['cells'] if g[p] == ink)
                ports[a['port']] = a
                for p in a['cells']:
                    clean[p] = a['host']
            _need(covered == set(map(tuple, np.argwhere(g == ink))), 'unexplained glyph ink')
            crossings = set()
            for p in np.ndindex(g.shape):
                if int(g[p]) in (bg, ink, wire):
                    continue
                neighbors = [_add(p, d) for d in D]
                if all(_inside(q, g.shape) and g[q] == wire for q in neighbors):
                    crossings.add(p)
            cross_colors = {int(g[p]) for p in crossings}
            _need(len(cross_colors) <= 1, 'several crossover roles')
            cross = next(iter(cross_colors), None)
            visible = set(map(tuple, np.argwhere(g == wire))) | crossings
            _need(cross is None or set(map(tuple, np.argwhere(g == cross))) == crossings,
                  'crossover color also used as a fill')
            graph = {p: {k: _add(p, d) for k, d in enumerate(D) if _add(p, d) in visible}
                     for p in visible}
            for p, edges in graph.items():
                if p in ports:
                    _need(set(edges) == {ports[p]['d']}, 'port does not point outward into wire')
                elif p in crossings:
                    _need(len(edges) == 4, 'incomplete crossing')
                else:
                    _need(len(edges) == 2, 'wire branch or break')
            excluded = {bg, ink, wire} | cross_colors
            ids, rooms = _components(clean, excluded)
            _need(len(rooms) >= 2, 'too few territories')
            for a in glyphs:
                room = int(ids[a['center']])
                _need(room >= 0 and all(ids[p] == room for p in a['patch']), 'glyph spans territories')
                a['room'] = room
            routes, assigned, visited, destinations = [], {}, set(), set()
            for source in launchers:
                p, incoming = source['port'], (source['d'] + 2) % 4
                path, states = [], set()
                while True:
                    _need((p, incoming) not in states, 'wire loop')
                    states.add((p, incoming))
                    path.append(p)
                    visited.add(p)
                    if p in ports and p != source['port']:
                        target = ports[p]
                        _need(target['kind'] != 'T', 'wire joins two launchers')
                        _need(p not in destinations, 'receiver used twice')
                        destinations.add(p)
                        break
                    edges = graph[p]
                    if p == source['port']:
                        outgoing = source['d']
                    elif p in crossings:
                        _need(incoming in edges, 'wrong crossing entrance')
                        outgoing = (incoming + 2) % 4
                    else:
                        _need(incoming in edges, 'wrong wire entrance')
                        outgoing = next(k for k in edges if k != incoming)
                    p = edges[outgoing]
                    incoming = (outgoing + 2) % 4
                _need(source['room'] != target['room'], 'self-directed cable')
                _need(source['payload'] not in excluded, 'structural payload color')
                phase = (sum(target['center']) + (target['kind'] == 'diamond')) % 2
                key = (target['room'], phase)
                _need(key not in assigned or assigned[key] == source['payload'], 'conflicting phase arrivals')
                assigned[key] = source['payload']
                routes.append(dict(source=source, target=target, path=path, phase=phase,
                                   payload=source['payload']))
            _need(visited == visible and len(destinations) == len(receivers), 'unused routing fragments')
            answers.append(dict(grid=g, clean=clean, bg=bg, ink=ink, wire=wire,
                                cross=cross, crossings=crossings, visible=visible,
                                ids=ids, rooms=rooms, glyphs=glyphs, routes=routes, assigned=assigned))
        except (AssertionError, StopIteration, KeyError):
            continue
    _need(len(answers) == 1, 'input must have a unique glyph/route interpretation')
    return answers[0]


def _paint(info, assignments=None):
    out = info['clean'].copy()
    rr, cc = np.indices(out.shape)
    for (room, phase), color in (info['assigned'] if assignments is None else assignments).items():
        out[(info['ids'] == room) & ((rr + cc) % 2 == phase)] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Interpret only the supplied input; all conventions are geometrically fixed."""
    return _paint(_read(_grid(input_grid)))


def _key(a):
    a = np.asarray(a, dtype=np.int8)
    return a.shape, a.tobytes()


def _room_layout(rng, count, rich, minimum=7):
    # Unequal room rectangles occupy a shuffled subset of unequal slots. The
    # unused slot, corner cuts and four-side sockets defeat opposed-bank rules.
    nr, nc = ((2, 2) if count == 4 else ((2, 3) if count >= 5 else (1, count)))
    gap_r = int(rng.integers(3, 6))
    gap_c = int(rng.integers(3, 6)) if nc <= 2 else 3
    heights = np.full(nr, minimum, dtype=int)
    widths = np.full(nc, minimum, dtype=int)
    for values, gap in ((heights, gap_r), (widths, gap_c)):
        budget = 30 - (int(values.sum()) + gap * (len(values) - 1) + 2)
        for _ in range(int(rng.integers(min(budget, 7) + 1))):
            values[int(rng.integers(len(values)))] += 1
    h = int(heights.sum() + gap_r * (nr - 1) + 2)
    w = int(widths.sum() + gap_c * (nc - 1) + 2)
    labels = np.full((h, w), -1, dtype=np.int16)
    slots = list(range(nr * nc))
    # The first routing lesson uses diagonals of four islands. Later inputs
    # randomize semantic room IDs and may omit any one of six positions.
    if count != 4:
        rng.shuffle(slots)
    boxes = []
    for i, slot in enumerate(slots[:count]):
        row, col = divmod(slot, nc)
        r0 = 1 + int(heights[:row].sum()) + row * gap_r
        c0 = 1 + int(widths[:col].sum()) + col * gap_c
        r1, c1 = r0 + int(heights[row]), c0 + int(widths[col])
        labels[r0:r1, c0:c1] = i
        if rich or rng.random() < .4:
            for corner in rng.choice(4, int(rng.integers(1, 4)), replace=False):
                cut_h, cut_w = int(rng.integers(1, 3)), int(rng.integers(1, 3))
                rs = slice(r0, r0 + cut_h) if corner < 2 else slice(r1 - cut_h, r1)
                cs = slice(c0, c0 + cut_w) if corner % 2 == 0 else slice(c1 - cut_w, c1)
                labels[rs, cs] = -1
        boxes.append((r0, r1, c0, c1))
    return labels, boxes


def _port_choices(labels, room):
    # Check the fixed support stencils in NumPy once, rather than rebuilding
    # nine-cell patches at every room pixel. Keep the original row/column/
    # direction enumeration order so seeded construction remains identical.
    h, w = labels.shape
    padded = np.pad(labels, 3, constant_values=-2)
    def shifted(dr, dc):
        return padded[3 + dr:3 + dr + h, 3 + dc:3 + dc + w]
    complete = np.logical_and.reduce([shifted(dr, dc) == room
                                     for dr in range(-1, 2) for dc in range(-1, 2)])
    supports = [shifted(2 * dr, 2 * dc) == room for dr, dc in D]
    valid = []
    for k, (dr, dc) in enumerate(D):
        mask = complete & (shifted(2 * dr, 2 * dc) == -1) & (shifted(3 * dr, 3 * dc) == -1)
        for j in range(4):
            if j != k:
                mask &= supports[j]
        valid.append(mask)
    valid = np.stack(valid)
    result = []
    for position in np.argwhere(np.any(valid, axis=0)):
        r, c = map(int, position)
        center = (r, c)
        patch = {(r + a, c + b) for a in range(-1, 2) for b in range(-1, 2)}
        for k, d in enumerate(D):
            if valid[k, r, c]:
                support = {_add(center, dd, 2) for j, dd in enumerate(D) if k != j}
                result.append(dict(center=center, d=k, port=_add(center, d, 2),
                                   patch=patch, support=support))
    return result


def _routing_context(labels, used, blocked=None):
    """Geometry shared by candidate paths for one unchanged set of old wires."""
    neighbors = {}
    for p in used:
        for k, d in enumerate(D):
            q = _add(p, d, -1)
            neighbors.setdefault(q, []).append((k, p))
    # Dictionary/set traversal order must not change the old D-order lists.
    neighbors = {p: tuple(sorted(edges)) for p, edges in neighbors.items()}
    straight = {}
    for p in used:
        axes = {k for k, _ in neighbors.get(p, ())}
        straight[p] = min(axes) % 2 if axes in ({0, 2}, {1, 3}) else -1
    return neighbors, straight, (labels >= 0).tolist() if blocked is None else blocked


def _route(rng, labels, used, source, target, context=None):
    """Weighted pixel routing; old strands can only be crossed perpendicularly.

    States remember the entrance axis. A candidate next to an existing strand
    must enter its straight segment, and must leave it straight on the other
    side. This cannot manufacture a branch or hide a route permutation.
    """
    start, end = source['port'], target['port']
    if start in used or end in used:
        return None
    neighbors, straight, blocked = _routing_context(labels, used) if context is None else context
    if neighbors.get(start) or neighbors.get(end):
        return None
    noise = (rng.random(labels.shape) * .65).tolist()
    turn_cost = float(rng.uniform(.2, 1.6))
    initial = (start, source['d'])
    queue = [(0., 0, initial)]
    distances, previous = {initial: 0.}, {}
    serial = 1
    final = None
    while queue:
        cost, _, state = heapq.heappop(queue)
        if cost != distances[state]:
            continue
        p, forward = state
        if p == end and forward == (target['d'] + 2) % 4:
            final = state
            break
        directions = list(range(4))
        if p == start:
            directions = [source['d']]
        elif p in used:
            directions = [forward]
        else:
            nearby = neighbors.get(p, ())
            behind = _add(p, D[forward], -1)
            if nearby and nearby[0][1] != behind:
                directions = [nearby[0][0]]
        for k in directions:
            if k == (forward + 2) % 4:
                continue
            q = _add(p, D[k])
            if not _inside(q, labels.shape) or blocked[q[0]][q[1]] or q == start:
                continue
            if q == end and k != (target['d'] + 2) % 4:
                continue
            if q in used:
                axis = straight[q]
                if axis < 0 or k % 2 == axis:
                    continue
            else:
                near = neighbors.get(q, ())
                if near:
                    if len(near) != 1:
                        continue
                    j, old = near[0]
                    if old != p:
                        axis = straight[old]
                        if j != k or axis < 0 or k % 2 == axis:
                            continue
            next_state = (q, k)
            nc = cost + 1 + noise[q[0]][q[1]] + turn_cost * (k != forward)
            if nc < distances.get(next_state, float('inf')):
                distances[next_state] = nc
                previous[next_state] = state
                heapq.heappush(queue, (nc, serial, next_state))
                serial += 1
    if final is None:
        return None
    path = []
    while True:
        path.append(final[0])
        if final == initial:
            break
        final = previous[final]
    path.reverse()
    if len(set(path)) != len(path):
        return None
    together = used | set(path)
    crossing = used & set(path)
    # Reject nonconsecutive contact of the new path with itself.
    for i, p in enumerate(path):
        neighbors = {_add(p, d) for d in D} & together
        expected = ({path[i - 1]} if i else set()) | ({path[i + 1]} if i + 1 < len(path) else set())
        if p in crossing:
            if len(neighbors) != 4:
                return None
        elif neighbors != expected:
            return None
    return path


def _construct(rng, palette, lesson, count, rich, extra_edges):
    minimum = 9 if lesson == 'weave' or (count == 4 and extra_edges >= 3 and lesson is None) else 7
    labels, boxes = _room_layout(rng, count, rich, minimum=minimum)
    choices = [_port_choices(labels, i) for i in range(count)]
    if lesson == 'routing':
        edges = [(0, 3), (1, 2)]
    elif lesson == 'weave':
        edges = [(0, 1), (2, 1), (1, 3)]
    else:
        order = list(map(int, rng.permutation(count)))
        # A chain/cycle is geometrically routed, rather than position matched.
        n = min(extra_edges, count)
        edges = [(order[i], order[(i + 1) % count]) for i in range(n)]
        if n >= 3 and rng.random() < .5:
            edges[1] = (order[2], order[1])
    selected, paths, used = [], [], set()
    protected_patches, protected_support = set(), set()
    blocked = (labels >= 0).tolist()
    for a, b in edges:
        context = _routing_context(labels, used, blocked)
        candidates = [(s, t) for s in choices[a] for t in choices[b]
                      if not (s['patch'] | t['patch']) & (protected_patches | protected_support)
                      and not (s['support'] | t['support']) & protected_patches
                      and not s['patch'] & t['patch']]
        rng.shuffle(candidates)
        result = None
        for source, target in candidates[:48]:
            path = _route(rng, labels, used, source, target, context)
            if path is not None:
                result = (source, target, path)
                break
        if result is None:
            return None
        source, target, path = result
        selected.append((a, b, source, target))
        paths.append(path)
        used.update(path)
        protected_patches.update(source['patch'] | target['patch'])
        protected_support.update(source['support'] | target['support'])
    bg, wire, ink, cross = palette[:4]
    fills = list(map(int, rng.choice(palette[4:], count, replace=count > 6)))
    if lesson == 'weave':
        fills[3] = fills[0]  # donor A and recipient D share a fill but behave differently
    elif count >= 3 and rng.random() < .35:
        fills[-1] = fills[0]
    clean = np.full(labels.shape, bg, dtype=np.int8)
    for i, color in enumerate(fills):
        clean[labels == i] = color
    visits = {}
    for i, path in enumerate(paths):
        for p in path:
            visits.setdefault(p, set()).add(i)
            clean[p] = wire
    crossings = {p for p, owners in visits.items() if len(owners) == 2}
    for p in crossings:
        clean[p] = cross
    if lesson == 'routing' and not crossings:
        return None
    inp, out = clean.copy(), clean.copy()
    assigned = {}
    source_payloads = []
    for i, (a, b, source, target) in enumerate(selected):
        explicit = (lesson == 'routing' and i == 0) or (lesson is None and rng.random() < .35)
        options = [c for c in palette[4:] if c not in (fills[a], fills[b])]
        if lesson == 'routing':
            options = [c for c in palette[4:] if c not in fills] or options
        payload = int(rng.choice(options)) if explicit else fills[a]
        # Nonlesson collisions on one phase must agree. Choose the other phase
        # when available; the rule itself has no arrival priority.
        phase = int(rng.integers(2))
        if lesson == 'weave' and i == 1:
            phase = 1 - next(p for (room, p) in assigned if room == b)
        elif (b, phase) in assigned and assigned[b, phase] != payload:
            phase = 1 - phase
            if (b, phase) in assigned and assigned[b, phase] != payload:
                return None
        assigned[b, phase] = payload
        source_payloads.append(payload)
        center, d = source['center'], D[source['d']]
        perp = (-d[1], d[0])
        t_cells = {center, _add(center, perp), _add(center, perp, -1), _add(center, d)}
        for p in t_cells:
            inp[p] = ink
        if explicit:
            inp[center] = payload
        center = target['center']
        offsets = CHECKER if sum(center) % 2 == phase else DIAMOND
        for dr, dc in offsets:
            inp[center[0] + dr, center[1] + dc] = ink
    rr, cc = np.indices(out.shape)
    for (room, phase), payload in assigned.items():
        out[(labels == room) & ((rr + cc) % 2 == phase)] = payload
    if lesson == 'routing' and len(set(source_payloads)) != len(source_payloads):
        return None
    k = int(rng.integers(4))
    inp, out = np.rot90(inp, k).copy(), np.rot90(out, k).copy()
    if rng.random() < .5:
        inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
    try:
        info = _read(inp)
        if not np.array_equal(_paint(info), out):
            return None
        features = _features(info)
        if lesson == 'weave' and not (features['weave'] and features['simultaneous'] and features['locality']
                                     and set(features['stencils']) == {'diamond', 'checker'}
                                     and features['anchor_contrast']):
            return None
    except AssertionError:
        return None
    return dict(input=inp, output=out)


def _features(info):
    routes, rooms = info['routes'], info['rooms']
    out = _paint(info)
    weave = any((i, 0) in info['assigned'] and (i, 1) in info['assigned']
                and info['assigned'][i, 0] != info['assigned'][i, 1] for i in range(len(rooms)))
    simultaneous = False
    for route in routes:
        source = route['source']
        if not source['explicit'] and out[source['center']] != source['host']:
            simultaneous = True
    locality = any(a['color'] == b['color'] and
                   {int(out[p]) for p in a['cells']} != {int(out[p]) for p in b['cells']}
                   for i, a in enumerate(rooms) for b in rooms[:i])
    nonrect = 0
    far = 0
    for room in rooms:
        rr, cc = zip(*room['cells'])
        nonrect += len(room['cells']) != (max(rr) - min(rr) + 1) * (max(cc) - min(cc) + 1)
    for route in routes:
        r, c = route['target']['center']
        far += any(abs(p[0] - r) + abs(p[1] - c) >= 4 and
                   sum(p) % 2 == route['phase'] and out[p] == route['payload']
                   and info['clean'][p] != out[p] for p in rooms[route['target']['room']]['cells'])
    phases_by_kind = {kind: {r['phase'] for r in routes if r['target']['kind'] == kind
                            and r['payload'] != rooms[r['target']['room']]['color']}
                      for kind in ('diamond', 'checker')}
    return dict(shape=info['grid'].shape, rooms=len(rooms), routes=len(routes),
                crossings=len(info['crossings']), weave=weave, simultaneous=simultaneous,
                anchor_contrast=any(len(phases) == 2 for phases in phases_by_kind.values()),
                locality=locality, explicit=sum(r['source']['explicit'] for r in routes),
                singleton=sum(r['source']['explicit'] and r['payload'] not in {a['color'] for a in rooms}
                              for r in routes), nonrect=nonrect, far=far,
                directions=tuple(sorted({r['source']['d'] for r in routes})),
                stencils=tuple(sorted({r['target']['kind'] for r in routes})),
                roles=(info['bg'], info['wire'], info['ink'], info['cross']))


def _omission_witnesses(info):
    """Concrete full-array contrasts, not an assertion based on motif presence."""
    correct = _paint(info)
    host_only, updated, fixed_phase = dict(info['assigned']), dict(info['assigned']), {}
    flood = info['clean'].copy()
    color_wide = info['clean'].copy()
    rr, cc = np.indices(correct.shape)
    for route in info['routes']:
        source, target = route['source'], route['target']
        key = (target['room'], route['phase'])
        fixed_phase[target['room'], int(target['kind'] == 'diamond')] = route['payload']
        if source['explicit']:
            host_only[key] = source['host']
        else:
            updated[key] = int(correct[source['center']])
        flood[info['ids'] == target['room']] = route['payload']
        target_fill = info['rooms'][target['room']]['color']
        color_wide[(info['clean'] == target_fill) & ((rr + cc) % 2 == route['phase'])] = route['payload']
    switched = False
    for i, a in enumerate(info['routes']):
        for b in info['routes'][:i]:
            if (set(a['path']) & set(b['path']) & info['crossings'] and a['payload'] != b['payload']
                    and (a['target']['room'], a['phase']) != (b['target']['room'], b['phase'])):
                wrong = dict(info['assigned'])
                wrong[a['target']['room'], a['phase']] = b['payload']
                wrong[b['target']['room'], b['phase']] = a['payload']
                switched |= not np.array_equal(correct, _paint(info, wrong))
    return dict(ignore_explicit=not np.array_equal(correct, _paint(info, host_only)),
                read_updated_donor=not np.array_equal(correct, _paint(info, updated)),
                ignore_component=not np.array_equal(correct, color_wide),
                ignore_anchor=not np.array_equal(correct, _paint(info, fixed_phase)),
                ignore_phase=not np.array_equal(correct, flood), switch_crossed_strands=switched)


def _counts(rng, a, b):
    for value, name, lo, hi in ((a, 'num_train', 2, 10), (b, 'num_test', 1, 4)):
        if value is not None and (isinstance(value, (bool, np.bool_)) or
                                 not isinstance(value, (int, np.integer)) or not lo <= value <= hi):
            raise ValueError(name + ' must be a legal integer')
    if a is None:
        a = int(rng.integers(max(2, 1 + (b or 1)), 11))
    if b is None:
        b = int(rng.integers(1, min(4, a - 1) + 1))
    if a <= b:
        raise ValueError('num_train must exceed num_test')
    return int(a), int(b)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    nt, nq = _counts(rng, num_train, num_test)
    palette = list(map(int, rng.permutation(10)))
    vary_palette = bool(rng.integers(2))
    shift = int(rng.integers(3))
    puzzle, seen = dict(train=[], test=[]), set()
    for split, total in (('train', nt), ('test', nq)):
        for index in range(total):
            lesson = ('routing' if index == 0 else 'weave' if index == 1 else None) if split == 'train' else None
            count = 4 if lesson else int(rng.integers(2, 6))
            if split == 'test' and shift == 0:
                count = int(rng.integers(5, 7))
            elif split == 'test':
                count = int(rng.integers(4, 6))
            rich = lesson == 'weave' or split == 'test' or bool(rng.integers(2))
            edges = int(rng.integers(1, min(4, count) + 1))
            if split == 'test':
                edges = 3 if shift != 1 else 4
            local_palette = list(map(int, rng.permutation(10))) if vary_palette else palette
            for _ in range(100):
                pair = _construct(rng, local_palette, lesson, count, rich, edges)
                if pair is None or _key(pair['input']) in seen:
                    continue
                puzzle[split].append(pair)
                seen.add(_key(pair['input']))
                break
            else:
                raise RuntimeError(f'bounded scene construction exhausted: seed={seed}, {split}[{index}], '
                                   f'lesson={lesson}, rooms={count}, routes={edges}')
    rng.shuffle(puzzle['train'])
    rng.shuffle(puzzle['test'])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'wrong puzzle schema')
    _need(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    a, b = len(puzzle['train']), len(puzzle['test'])
    _need(2 <= a <= 10 and 1 <= b <= 4 and a > b, 'illegal example counts')
    seen, evidence, witnesses = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'wrong pair schema')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _need(inp.shape == out.shape and _key(inp) not in seen, 'duplicate input or changed shape')
            seen.add(_key(inp))
            info = _read(inp)
            _need(np.array_equal(_paint(info), out), 'incorrect routed weave')
            _need(not np.array_equal(inp, out), 'identity example')
            _need(all(inp[p] == out[p] for p in info['visible']), 'wire geometry changed')
            _need(np.array_equal(inp[inp == info['bg']], out[inp == info['bg']]), 'background changed')
            _need(not np.any(out == info['ink']), 'glyph ink survived')
            for room, cells in enumerate(info['rooms']):
                for phase in (0, 1):
                    selected = [p for p in cells['cells'] if sum(p) % 2 == phase]
                    expected = info['assigned'].get((room, phase), cells['color'])
                    _need(all(out[p] == expected for p in selected), 'wrong component or parity fill')
            if split == 'train':
                evidence.append(_features(info))
                witnesses.append(_omission_witnesses(info))
    _need(any(f['crossings'] and f['explicit'] for f in evidence), 'missing routing/payload demonstration')
    _need(any(f['weave'] and f['simultaneous'] and f['locality'] and f['anchor_contrast'] for f in evidence),
          'missing original-color complementary-weave demonstration')
    _need({s for f in evidence for s in f['stencils']} == {'diamond', 'checker'}, 'phase convention untaught')
    _need(any(f['far'] for f in evidence), 'no stencil extension beyond 3x3')
    for omission in ('ignore_explicit', 'read_updated_donor', 'ignore_component', 'ignore_anchor',
                     'ignore_phase', 'switch_crossed_strands'):
        _need(any(w[omission] for w in witnesses), 'training does not distinguish ' + omission)


def self_test():
    """Bounded semantic checks and measured structural diversity on 42 seeds."""
    features, signatures, structures = [], set(), set()
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                inp, out = pair['input'], pair['output']
                info = _read(inp)
                features.append(_features(info))
                signatures.add(_key(inp))
                # Region arrangement, local glyph placement and actual paths,
                # independent of literal palette names.
                names = {}
                normalized = np.array([names.setdefault(int(c), len(names)) for c in inp.ravel()], dtype=np.int8)
                structures.add((inp.shape, normalized.tobytes()))
                _need(np.array_equal(solve(inp.copy()), out), 'input-only solve mismatch')
        if seed < 2:
            other = generate_puzzle(seed)
            _need(all(_key(p[k]) == _key(q[k]) for s in puzzle
                      for p, q in zip(puzzle[s], other[s]) for k in ('input', 'output')), 'not deterministic')
    _need(len(signatures) == 210 and len(structures) >= 180, 'collapsed structure variation')
    _need(len({f['shape'] for f in features}) >= 25, 'collapsed shapes')
    _need({f['rooms'] for f in features} >= {2, 3, 4, 5, 6}, 'missing room counts')
    _need({f['routes'] for f in features} >= {1, 2, 3, 4}, 'missing route counts')
    _need({d for f in features for d in f['directions']} == set(range(4)), 'missing launch directions')
    for name in ('weave', 'simultaneous', 'locality', 'anchor_contrast', 'crossings', 'singleton', 'nonrect', 'far'):
        _need(sum(bool(f[name]) for f in features) >= 5, 'missing family feature: ' + name)
    _need(len({f['roles'][:3] for f in features}) >= 25, 'collapsed structural color roles')
    # Metamorphic check: all ten colors can exchange roles, with no color map
    # secretly learned from the seed or retained examples.
    pair = puzzles[0]['test'][0]
    lut = np.array([7, 3, 9, 0, 6, 1, 8, 5, 2, 4], dtype=np.int8)
    _need(np.array_equal(solve(lut[pair['input']]), lut[pair['output']]), 'palette dependence')
    # Whole-canvas turns/reflections preserve socket-anchored checkerboard phase.
    for k in range(4):
        _need(np.array_equal(solve(np.rot90(pair['input'], k)), np.rot90(pair['output'], k)), 'rotation dependence')
    _need(np.array_equal(solve(pair['input'][:, ::-1]), pair['output'][:, ::-1]), 'reflection dependence')
    # Boundary and partial requests here; verify.py separately exercises every
    # legal explicit count combination, avoiding a duplicate exhaustive sweep.
    for index, (a, b) in enumerate(((2, 1), (5, 4), (10, 4), (None, 4), (10, None), (None, None))):
        p = generate_puzzle(900 + index, a, b)
        _need(a is None or len(p['train']) == a, 'training count request ignored')
        _need(b is None or len(p['test']) == b, 'test count request ignored')
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError('illegal count accepted')
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in ('train', 'test')}
    bad['test'][0]['output'][0, 0] = (int(bad['test'][0]['output'][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('background corruption accepted')
    return dict(seeds=42, pairs=210, shapes=len({f['shape'] for f in features}),
                structures=len(structures), feature_pairs={k: sum(bool(f[k]) for f in features)
                for k in ('weave', 'simultaneous', 'locality', 'anchor_contrast', 'crossings', 'singleton', 'nonrect', 'far')})


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
