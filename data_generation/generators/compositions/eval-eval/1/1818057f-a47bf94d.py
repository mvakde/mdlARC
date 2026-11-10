"""Parity-gated cables: local plus detectors control distant terminal stencils.

Only standard-library modules and NumPy are used. Structural colors are fixed;
label and chamber-mark colors are supplied by the input. Generation constructs
paths and sensor contents, while solve rebuilds a directional graph from cells.
"""
from __future__ import annotations

from itertools import product
import numpy as np

BG, WIRE, CROSS, FRAME = 0, 5, 6, 9
INKS = (1, 2, 3, 4, 7, 8)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or not np.all((g >= 0) & (g <= 9))):
        raise ValueError("expected an integer ARC grid, dimensions 1..30, colors 0..9")
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _plus_centers(mask):
    result = np.zeros(mask.shape, dtype=bool)
    if min(mask.shape) >= 3:
        result[1:-1, 1:-1] = (mask[1:-1, 1:-1] & mask[:-2, 1:-1]
                                      & mask[2:, 1:-1] & mask[1:-1, :-2]
                                      & mask[1:-1, 2:])
    return result


def _count_by_coordinates(mask):
    # Construction/validation check uses direct neighborhoods, not dilation.
    return sum(all(mask[r + dr, c + dc] for dr, dc in ((0, 0),) + DIRS)
               for r in range(1, mask.shape[0] - 1)
               for c in range(1, mask.shape[1] - 1))


def _components(points):
    remaining = set(points)
    result = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        found, stack = {p}, [p]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                q = r + dr, c + dc
                if q in remaining:
                    remaining.remove(q)
                    found.add(q)
                    stack.append(q)
        result.append(found)
    return result


def _inside(p, rect):
    a, b, x, y = rect
    return a <= p[0] <= b and x <= p[1] <= y


def _rooms(g):
    rooms = []
    for cells in _components(map(tuple, np.argwhere(g == FRAME))):
        rows, cols = zip(*cells)
        a, b, x, y = min(rows), max(rows), min(cols), max(cols)
        if b - a < 4 or y - x < 4:
            raise ValueError("chamber interiors must be at least 3 by 3")
        boundary = {(r, c) for r in range(a, b + 1) for c in range(x, y + 1)
                    if r in (a, b) or c in (x, y)}
        if cells != boundary:
            raise ValueError("9 must form separated rectangular chamber borders")
        content = g[a + 1:b, x + 1:y]
        inks = set(map(int, np.unique(content))) - {BG}
        if not inks.issubset(INKS) or len(inks) > 1:
            raise ValueError("a chamber has background and at most one mark color")
        centers = _plus_centers(content != BG)
        rooms.append(dict(rect=(a, b, x, y), count=int(centers.sum()),
                          centers=tuple(map(tuple, np.argwhere(centers))),
                          mask=content != BG, ink=next(iter(inks), None)))
    return rooms


def _visible_graph(g):
    visible = set(map(tuple, np.argwhere((g == WIRE) | (g == CROSS))))
    graph = {p: {} for p in visible}
    for r, c in visible:
        for d, (dr, dc) in enumerate(DIRS):
            q = r + dr, c + dc
            if q in visible:
                # Third item is the sensor index; None means an ordinary edge.
                graph[r, c][d] = (q, (d + 2) % 4, None)
    return graph


def _ports(rect, graph):
    a, b, x, y = rect
    candidates = ([((a - 1, c), 2) for c in range(x, y + 1)]
                  + [((b + 1, c), 0) for c in range(x, y + 1)]
                  + [((r, x - 1), 1) for r in range(a, b + 1)]
                  + [((r, y + 1), 3) for r in range(a, b + 1)])
    # A parallel strand beside a border has degree two and is not a port.
    # Degree zero is a one-cell stub between a terminal and a chamber, or
    # between two chambers; the missing edge is still visibly directed inward.
    return [(p, d) for p, d in candidates if p in graph
            and len(graph[p]) in (0, 1, 3) and d not in graph[p]]


def _terminal(g, p, inward):
    dr, dc = DIRS[inward]
    r, c = p[0] - 2 * dr, p[1] - 2 * dc
    if not (1 <= r < g.shape[0] - 1 and 1 <= c < g.shape[1] - 1):
        raise ValueError("terminal footprint falls outside canvas")
    block = g[r - 1:r + 2, c - 1:c + 2]
    if np.all(block == BG):
        return dict(center=(r, c), color=None, state=None)
    inks = set(map(int, np.unique(block))) - {BG}
    if len(inks) != 1 or not inks.issubset(INKS):
        raise ValueError("terminal must have one nonstructural label color")
    color = next(iter(inks))
    for state, stencil in enumerate((DIAMOND, CHECKER)):
        if np.array_equal(block, np.where(stencil, color, BG)):
            return dict(center=(r, c), color=color, state=state)
    raise ValueError("terminal clue must be a diamond or checker")


def _parse(g):
    rooms = _rooms(g)
    graph = _visible_graph(g)
    if not graph:
        raise ValueError("no cables")
    # Classify all ports on the original visible graph, before any jumps are
    # added. A later jump must not turn an earlier side contact into a port.
    room_ports = [_ports(room["rect"], graph) for room in rooms]
    for index, room in enumerate(rooms):
        ports = room_ports[index]
        if len(ports) != 2:
            raise ValueError("every chamber must have exactly two true cable ports")
        (p, dp), (q, dq) = ports
        if dp in graph[p] or dq in graph[q]:
            raise ValueError("two chambers compete for the same directed port")
        graph[p][dp] = (q, dq, index)
        graph[q][dq] = (p, dp, index)
        room["ports"] = tuple(ports)
    endpoints = {}
    crossings = []
    for p, edges in graph.items():
        if g[p] == CROSS:
            if len(edges) != 4:
                raise ValueError("6 is a straight-through crossing, requiring four directions")
            crossings.append(p)
        elif len(edges) == 1:
            endpoints[p] = _terminal(g, p, next(iter(edges)))
        elif len(edges) != 2:
            raise ValueError("ordinary cables may neither branch nor break")
    if len(endpoints) < 2 or len(endpoints) % 2:
        raise ValueError("cables must have paired endpoints")
    # Every non-background pixel must have a structural or terminal role.
    allowed = np.zeros(g.shape, dtype=bool)
    for p in graph:
        allowed[p] = True
    for room in rooms:
        a, b, x, y = room["rect"]
        if allowed[a:b + 1, x:y + 1].any():
            raise ValueError("wire cells must be outside chambers")
        allowed[a:b + 1, x:y + 1] = True
    for terminal in endpoints.values():
        r, c = terminal["center"]
        if allowed[r - 1:r + 2, c - 1:c + 2].any():
            raise ValueError("terminal footprints must be disjoint from other objects")
        allowed[r - 1:r + 2, c - 1:c + 2] = True
    if np.any((g != BG) & ~allowed):
        raise ValueError("unattached label/mark cells")
    unused = set(endpoints)
    routes, used_edges, room_owners = [], set(), {}
    while unused:
        start = min(unused)
        p, incoming = start, None
        states, cells, gates = set(), [], []
        while True:
            if (p, incoming) in states:
                raise ValueError("closed cable cycle")
            states.add((p, incoming))
            cells.append(p)
            if p != start and p in endpoints:
                if p not in unused:
                    raise ValueError("endpoint belongs to multiple cables")
                break
            edges = graph[p]
            if incoming is None:
                outgoing = next(iter(edges))
            elif len(edges) == 4:
                outgoing = (incoming + 2) % 4
            else:
                others = [d for d in edges if d != incoming]
                if len(others) != 1:
                    raise ValueError("ambiguous cable continuation")
                outgoing = others[0]
            q, next_incoming, gate = edges[outgoing]
            edge = frozenset(((p, outgoing), (q, next_incoming)))
            if edge in used_edges:
                raise ValueError("a cable strand is used more than once")
            used_edges.add(edge)
            if gate is not None:
                if gate in room_owners:
                    raise ValueError("a chamber belongs to more than one cable")
                room_owners[gate] = len(routes)
                gates.append(gate)
            p, incoming = q, next_incoming
        unused.remove(start)
        unused.remove(p)
        total = sum(rooms[i]["count"] for i in gates)
        near, far = endpoints[start], endpoints[p]
        labels = {t["color"] for t in (near, far) if t["color"] is not None}
        if len(labels) != 1:
            raise ValueError("each cable requires agreeing labels at one or both ends")
        relation = 1 ^ (total % 2)
        if near["state"] is not None:
            states_out = near["state"], near["state"] ^ relation
        else:
            states_out = far["state"] ^ relation, far["state"]
        if any(t["state"] is not None and t["state"] != s
               for t, s in zip((near, far), states_out)):
            raise ValueError("endpoint stencils disagree with plus parity")
        routes.append(dict(ends=(start, p), terminals=(near, far), cells=tuple(cells),
                           gates=tuple(gates), count=total, color=labels.pop(),
                           states=states_out, relation=relation))
    all_edges = {frozenset(((p, d), (q, dq))) for p, edges in graph.items()
                 for d, (q, dq, _) in edges.items()}
    if used_edges != all_edges or len(room_owners) != len(rooms):
        raise ValueError("unattached cable loop or chamber")
    return dict(rooms=rooms, routes=routes, graph=graph, crossings=crossings)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Complete labels and parity-dependent stencils from this input alone."""
    g = _grid(input_grid)
    info = _parse(g)
    result = g.copy()
    for route in info["routes"]:
        for terminal, state in zip(route["terminals"], route["states"]):
            r, c = terminal["center"]
            result[r - 1:r + 2, c - 1:c + 2] = np.where(
                (DIAMOND, CHECKER)[state], route["color"], BG)
    return result


def _line(a, b):
    dr, dc = int(np.sign(b[0] - a[0])), int(np.sign(b[1] - a[1]))
    if dr and dc:
        raise AssertionError("orthogonal segments only")
    length = abs(b[0] - a[0]) + abs(b[1] - a[1])
    return [(a[0] + k * dr, a[1] + k * dc) for k in range(length + 1)]


def _elbow_paths(shape, rows, columns, sides):
    h, w = shape
    result = []
    for r, c, (horizontal, vertical) in zip(rows, columns, sides):
        a, b = (r, 3 if horizontal == 0 else w - 4), (3 if vertical == 0 else h - 4, c)
        result.append(_line(a, (r, c)) + _line((r, c), b)[1:])
    return result


def _draw_paths(shape, paths):
    g = np.zeros(shape, dtype=np.int8)
    owners = {}
    for i, path in enumerate(paths):
        for p in path:
            owners.setdefault(p, set()).add(i)
            g[p] = WIRE
    for p, ids in owners.items():
        if len(ids) == 2:
            g[p] = CROSS
        elif len(ids) > 2:
            raise AssertionError("too many strands at an intersection")
    return g


def _valid_path_field(shape, paths):
    """Reject touching strands and turning intersections before placing rooms."""
    if any(len(path) != len(set(path)) for path in paths):
        return False
    try:
        field = _draw_paths(shape, paths)
    except AssertionError:
        return False
    graph = _visible_graph(field)
    endpoints = {p for path in paths for p in (path[0], path[-1])}
    if len(endpoints) != 2 * len(paths):
        return False
    for p, edges in graph.items():
        if len(edges) != (1 if p in endpoints else 4 if field[p] == CROSS else 2):
            return False
    for path in paths:
        for index, p in enumerate(path):
            if field[p] == CROSS:
                if index == 0 or index == len(path) - 1:
                    return False
                a, b = path[index - 1], path[index + 1]
                if (a[0] + b[0], a[1] + b[1]) != (2 * p[0], 2 * p[1]):
                    return False
    return True


def _detours(rng, shape, paths, wanted):
    """Replace variable straight subsegments with safe orthogonal excursions.

    This adds independently varied bends and crossings, rather than encoding
    all connectivity as a repeated lane-swap glyph or a single L per cable.
    """
    paths = [list(path) for path in paths]
    centers = [p for ends in _end_centers(paths) for p in ends]
    added = 0
    for _ in range(90):
        if added == wanted:
            break
        owner = int(rng.integers(len(paths)))
        path = paths[owner]
        if len(path) < 10:
            continue
        first = int(rng.integers(2, len(path) - 6))
        last = min(len(path) - 3, first + int(rng.integers(3, 10)))
        a, b = path[first], path[last]
        if a[0] != b[0] and a[1] != b[1]:
            continue
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != last - first:
            continue
        offset = int(rng.integers(2, 6)) * int(rng.choice((-1, 1)))
        dr, dc = (offset, 0) if a[0] == b[0] else (0, offset)
        aa, bb = (a[0] + dr, a[1] + dc), (b[0] + dr, b[1] + dc)
        replacement = _line(a, aa) + _line(aa, bb)[1:] + _line(bb, b)[1:]
        if any(not (3 <= r <= shape[0] - 4 and 3 <= c <= shape[1] - 4)
               for r, c in replacement):
            continue
        if any(abs(r - tr) <= 1 and abs(c - tc) <= 1
               for r, c in replacement for tr, tc in centers):
            continue
        candidate = [list(p) for p in paths]
        candidate[owner] = path[:first] + replacement + path[last + 1:]
        if not _valid_path_field(shape, candidate):
            continue
        paths, added = candidate, added + 1
    return paths


def _end_centers(paths):
    result = []
    for path in paths:
        ends = []
        for p, q in ((path[0], path[1]), (path[-1], path[-2])):
            dr, dc = q[0] - p[0], q[1] - p[1]
            ends.append((p[0] - 2 * dr, p[1] - 2 * dc))
        result.append(ends)
    return result


def _stamp_pattern(rng, shape, style=None):
    h, w = shape
    mask = np.zeros(shape, dtype=bool)
    style = int(rng.integers(6)) if style is None else style
    if style == 0:
        kind = int(rng.integers(3))
        stamp = (np.array([[0, 0, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
                 CHECKER, np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=bool))[kind]
        stamp = np.rot90(stamp, int(rng.integers(4)))
        r, c = int(rng.integers(h - 2)), int(rng.integers(w - 2))
        mask[r:r + 3, c:c + 3] = stamp
    elif style == 2 and max(shape) >= 4:
        bh, bw = (3, 4) if w >= 4 else (4, 3)
        r, c = int(rng.integers(h - bh + 1)), int(rng.integers(w - bw + 1))
        mask[r:r + bh, c:c + bw] = True
    elif style == 3:
        mask = rng.random(shape) < rng.uniform(.45, .9)
    elif style == 4:
        mask[0, :] = mask[-1, :] = True
        mask[:, 0] = mask[:, -1] = True
        if rng.random() < .5:
            mask[h // 2, w // 2] = True
    else:
        for _ in range(1 if style == 1 else int(rng.integers(1, 5))):
            r, c = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
            for dr, dc in ((0, 0),) + DIRS:
                mask[r + dr, c + dc] = True
            if rng.random() < .6:
                mask[r, :] = True
            for dr, dc in product((-1, 1), repeat=2):
                if rng.random() < .3:
                    mask[r + dr, c + dc] = True
    return mask


def _paint_room(rng, g, rect, owner, pattern):
    a, b, x, y = rect
    g[a:b + 1, x:y + 1] = BG
    g[a, x:y + 1] = g[b, x:y + 1] = FRAME
    g[a:b + 1, x] = g[a:b + 1, y] = FRAME
    ink = int(rng.choice(INKS))
    g[a + 1:b, x + 1:y] = np.where(pattern, ink, BG)
    return dict(rect=rect, owner=owner, mask=pattern,
                count=int(_count_by_coordinates(pattern)), ink=ink)


def _add_random_rooms(rng, g, paths, wanted, rich=False):
    rooms = []
    centers = [p for ends in _end_centers(paths) for p in ends]
    sizes = ((5, 5), (5, 6), (6, 5), (5, 7), (7, 5), (6, 7), (7, 6), (7, 7), (5, 9), (9, 5))
    for _ in range(220):
        if len(rooms) >= wanted:
            break
        owner = int(rng.integers(len(paths)))
        p = paths[owner][int(rng.integers(1, len(paths[owner]) - 1))]
        hh, ww = sizes[int(rng.integers(len(sizes)))]
        if rich and not rooms and max(hh, ww) < 7:
            continue
        a = p[0] - hh // 2 + int(rng.integers(-1, 2))
        x = p[1] - ww // 2 + int(rng.integers(-1, 2))
        b, y = a + hh - 1, x + ww - 1
        rect = a, b, x, y
        if a < 1 or x < 1 or b >= g.shape[0] - 1 or y >= g.shape[1] - 1:
            continue
        if any(not (b + 1 < q["rect"][0] or q["rect"][1] + 1 < a
                    or y + 1 < q["rect"][2] or q["rect"][3] + 1 < x) for q in rooms):
            continue
        if any(a - 1 <= r <= b + 1 and x - 1 <= c <= y + 1 for r, c in centers):
            continue
        occupied = [i for i, path in enumerate(paths) if any(_inside(q, rect) for q in path)]
        if occupied != [owner]:
            continue
        path = paths[owner]
        hidden = [i for i, q in enumerate(path) if _inside(q, rect)]
        if not hidden or hidden[0] == 0 or hidden[-1] == len(path) - 1:
            continue
        if hidden != list(range(hidden[0], hidden[-1] + 1)):
            continue
        actual = {path[hidden[0] - 1], path[hidden[-1] + 1]}
        candidate = g.copy()
        candidate[a:b + 1, x:y + 1] = BG
        candidate_graph = _visible_graph(candidate)
        ports = _ports(rect, candidate_graph)
        if len(ports) != 2 or {q for q, _ in ports} != actual:
            continue
        # A new border may remove a neighbor of a strand beside an earlier
        # chamber. Require every earlier chamber's two-port reading to remain
        # correct on the final visible graph; never rely on insertion order.
        if any(len(_ports(room["rect"], candidate_graph)) != 2
               or {q for q, _ in _ports(room["rect"], candidate_graph)} != room["actual_ports"]
               for room in rooms):
            continue
        pattern = _stamp_pattern(rng, (hh - 2, ww - 2))
        if rich and not rooms:
            for _ in range(20):
                pattern = _stamp_pattern(rng, (hh - 2, ww - 2), 3)
                if _count_by_coordinates(pattern) >= 3:
                    break
            else:
                # A finite fallback retains variable crop, orientation, ink,
                # geometry and positions; three adjacent centers are enough.
                pattern = np.zeros((hh - 2, ww - 2), dtype=bool)
                bh, bw = (3, 5) if ww >= 7 else (5, 3)
                pattern[:bh, :bw] = True
        elif not rich:
            # Training already teaches 0/1/2 centers and both overlap types.
            # Test increases local motif multiplicity under the same rule.
            while _count_by_coordinates(pattern) > 2:
                centers_here = np.argwhere(_plus_centers(pattern))
                pattern[tuple(centers_here[int(rng.integers(len(centers_here)))])] = False
        room = _paint_room(rng, g, rect, owner, pattern)
        room["actual_ports"] = actual
        rooms.append(room)
    return rooms


def _axis(rng, size, count):
    # Random unequal gaps, not a regular bank. Minimum spacing avoids touching.
    gaps = [5] * (count - 1)
    spare = size - 11 - sum(gaps)
    for i in rng.permutation(count - 1):
        extra = int(rng.integers(min(3, spare) + 1))
        gaps[int(i)] += extra
        spare -= extra
    first = int(rng.integers(5, 6 + spare))
    return [first] + [first + int(v) for v in np.cumsum(gaps)]


def _scene(rng, count, lesson=None, rich=False):
    if lesson is not None:
        h, w = map(int, rng.integers(28, 31, 2))
        r0, c0 = int(rng.integers(10, 13)), int(rng.integers(14, 17))
        r1, c1 = h - int(rng.integers(5, 8)), w - int(rng.integers(5, 8))
        rows, cols, sides = [r0, r1], [c1, c0], [(0, 1), (1, 0)]
        if lesson == "selective":
            rows.append(17)
            cols.append(int(rng.integers(6, 9)))
            sides.append((0, 1))
        paths = _elbow_paths((h, w), rows, cols, sides)
        g = _draw_paths((h, w), paths)
        if lesson == "selective":
            # Eight marked cells, but exactly ONE embedded plus center.
            pattern = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [1, 1, 0, 0]], dtype=bool)
            # Two plus centers sharing an arm: union has nine cells (odd),
            # whereas center count is two (even). Count centers, not pixels.
            shared = np.array([[0, 1, 0, 1, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0]], dtype=bool)
            rooms = [_paint_room(rng, g, (r0 - 2, r0 + 2, 4, 9), 0, pattern),
                     _paint_room(rng, g, (4, 8, c0 - 3, c0 + 3), 1, shared),
                     _paint_room(rng, g, (19, 23, cols[2] - 2, cols[2] + 2), 2, CHECKER)]
        else:
            plus = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
            rooms = [_paint_room(rng, g, (r0 - 2, r0 + 2, 4, 9), 0, np.ones((3, 4), bool)),
                     _paint_room(rng, g, (4, 8, c0 - 2, c0 + 2), 1, plus)]
            rooms.append(_paint_room(rng, g, (r0 + 2, r0 + 6, c0 - 2, c0 + 2), 1, plus))
        return g, paths, rooms
    minimum = max(16, 11 + 5 * (count - 1))
    h, w = map(int, rng.integers(minimum, 31, 2))
    rows, cols = _axis(rng, h, count), _axis(rng, w, count)
    rng.shuffle(cols)
    sides = [tuple(map(int, rng.integers(2, size=2))) for _ in range(count)]
    paths = _elbow_paths((h, w), rows, cols, sides)
    paths = _detours(rng, (h, w), paths, int(rng.integers(5 if rich else 4)))
    g = _draw_paths((h, w), paths)
    wanted = int(rng.integers(1, min(7, 2 * count + 2)))
    if not rich and rng.random() < .15:
        wanted = 0
    rooms = _add_random_rooms(rng, g, paths, wanted, rich)
    return g, paths, rooms


def _make_pair(rng, count, lesson=None, rich=False):
    for _ in range(30):
        g, paths, rooms = _scene(rng, count, lesson, rich)
        if not rich or rooms:
            break
    else:
        raise RuntimeError("could not place a sensor chamber")
    inp, out = g.copy(), g.copy()
    centers = _end_centers(paths)
    if lesson:
        colors = list(map(int, rng.choice(INKS, len(paths), replace=False)))
    else:
        palette = list(map(int, rng.choice(INKS, int(rng.integers(1, min(6, count) + 1)), replace=False)))
        colors = list(map(int, rng.choice(palette, len(paths))))
    clue_mode = int(rng.integers(3))
    for i, ends in enumerate(centers):
        k = sum(room["count"] for room in rooms if room["owner"] == i)
        state = int(rng.integers(2))
        states = state, state ^ (1 ^ (k % 2))
        if lesson:
            visible = (i % 2 == 0, i % 2 == 1)
        elif clue_mode < 2:
            visible = (clue_mode == 0, clue_mode == 1)
        else:
            choice = int(rng.integers(3))
            visible = (choice != 1, choice != 0)
        # At least one missing terminal makes every pair nonidentity.
        if i == 0 and all(visible):
            visible = True, False
        for j, ((r, c), s) in enumerate(zip(ends, states)):
            block = np.where((DIAMOND, CHECKER)[s], colors[i], BG)
            out[r - 1:r + 2, c - 1:c + 2] = block
            if visible[j]:
                inp[r - 1:r + 2, c - 1:c + 2] = block
    turns = int(rng.integers(4))
    inp, out = np.rot90(inp, turns).copy(), np.rot90(out, turns).copy()
    if rng.random() < .5:
        inp, out = inp[:, ::-1].copy(), out[:, ::-1].copy()
    # Construction-derived answer must agree with input-only path semantics.
    assert np.array_equal(solve(inp), out)
    return {"input": inp, "output": out}


def _count_arg(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not low <= value <= high:
        raise ValueError(f"{name} must be in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """Generate one fixed-rule puzzle, with two shuffled conceptual witnesses."""
    nt = _count_arg(num_train, "num_train", 2, 10)
    nq = _count_arg(num_test, "num_test", 1, 4)
    if nt <= nq:
        raise ValueError("num_train must be greater than num_test")
    if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
            or seed < 0):
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    shift = int(rng.integers(2))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, total in (("train", nt), ("test", nq)):
        for index in range(total):
            lesson = ("selective" if index == 0 else "accumulate" if index == 1 else None) if split == "train" else None
            count = (2 if lesson else int(rng.integers(1, 4))) if split == "train" else (4 if shift == 0 else int(rng.integers(2, 4)))
            rich = split == "test"
            for _ in range(40):
                pair = _make_pair(rng, count, lesson, rich)
                key = _key(pair["input"])
                if key not in seen:
                    break
            else:
                raise RuntimeError("could not construct distinct inputs")
            seen.add(key)
            puzzle[split].append(pair)
    puzzle["train"] = [puzzle["train"][int(i)] for i in rng.permutation(nt)]
    validate_generated_puzzle(puzzle)
    return puzzle


def _evidence(info):
    evidence = set()
    for room in info["rooms"]:
        mask, centers = room["mask"], _plus_centers(room["mask"])
        count = room["count"]
        if count == 0 and mask.any():
            evidence.add("near_miss")
        if count == 1:
            union = centers.copy()
            union[1:] |= centers[:-1]
            union[:-1] |= centers[1:]
            union[:, 1:] |= centers[:, :-1]
            union[:, :-1] |= centers[:, 1:]
            if np.any(mask & ~union):
                evidence.add("embedded")
        if count == 2 and (np.any(centers[:-1] & centers[1:])
                           or np.any(centers[:, :-1] & centers[:, 1:])):
            evidence.add("overlap_two")
        if count == 2 and (np.any(centers[:-2] & centers[2:])
                           or np.any(centers[:, :-2] & centers[:, 2:])):
            evidence.add("shared_arm_two")
        if count % 2 != int(mask.sum()) % 2:
            evidence.add("pixel_count_wrong")
    for route in info["routes"]:
        if route["count"] % 2:
            evidence.add("odd")
        else:
            evidence.add("even")
        if len(route["gates"]) >= 2 and all(info["rooms"][i]["count"] == 1 for i in route["gates"]):
            evidence.add("distributed")
    for p in info["crossings"]:
        owners = [r for r in info["routes"] if p in r["cells"]]
        if len(owners) == 2 and owners[0]["color"] != owners[1]["color"]:
            evidence.add("distinct_crossing")
    return evidence


def _ablation_answers(inp, info):
    """Concrete shortcuts: ignore pluses; flood through cable crossings.

    The flood alternative is even granted the correct terminal phases, so a
    label error specifically establishes the need for individual strand tracing.
    Both alternatives read only the input and its parsed geometry.
    """
    no_plus, flood = inp.copy(), solve(inp)
    adjacency = {p: {q for q, _, _ in edges.values()}
                 for p, edges in info["graph"].items()}
    components, unseen = [], set(adjacency)
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        found, stack = {p}, [p]
        while stack:
            for q in adjacency[stack.pop()]:
                if q in unseen:
                    unseen.remove(q)
                    found.add(q)
                    stack.append(q)
        components.append(found)
    for route in info["routes"]:
        near, far = route["terminals"]
        if near["state"] is not None:
            phases = near["state"], near["state"] ^ 1
        else:
            phases = far["state"] ^ 1, far["state"]
        component = next(c for c in components if route["ends"][0] in c)
        visible_colors = [t["color"] for other in info["routes"]
                          for p, t in zip(other["ends"], other["terminals"])
                          if p in component and t["color"] is not None]
        flood_color = min(visible_colors)
        for terminal, wrong_phase, correct_phase in zip(route["terminals"], phases, route["states"]):
            if terminal["color"] is None:
                r, c = terminal["center"]
                no_plus[r - 1:r + 2, c - 1:c + 2] = np.where(
                    (DIAMOND, CHECKER)[wrong_phase], route["color"], BG)
                flood[r - 1:r + 2, c - 1:c + 2] = np.where(
                    (DIAMOND, CHECKER)[correct_phase], flood_color, BG)
    return no_plus, flood


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, evidence = set(), set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            assert inp.shape == out.shape
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            assert np.array_equal(solve(inp), out)
            footprints = np.zeros(inp.shape, bool)
            for route in info["routes"]:
                for terminal, state in zip(route["terminals"], route["states"]):
                    r, c = terminal["center"]
                    footprints[r - 1:r + 2, c - 1:c + 2] = True
                    assert np.array_equal(out[r - 1:r + 2, c - 1:c + 2],
                                          np.where((DIAMOND, CHECKER)[state], route["color"], BG))
                assert route["states"][0] ^ route["states"][1] == 1 ^ (route["count"] % 2)
            assert np.array_equal(inp[~footprints], out[~footprints])
            assert np.any(inp != out)
            for room in info["rooms"]:
                assert room["count"] == _count_by_coordinates(room["mask"])
            if split == "train":
                assert all(room["count"] <= 2 for room in info["rooms"])
                evidence |= _evidence(info)
                no_plus, flood = _ablation_answers(inp, info)
                if not np.array_equal(no_plus, out):
                    evidence.add("A_changes_answer")
                if not np.array_equal(flood, out):
                    evidence.add("B_changes_answer")
            else:
                assert any(room["count"] >= 3 for room in info["rooms"])
    assert {"near_miss", "embedded", "overlap_two", "shared_arm_two", "pixel_count_wrong",
            "distributed", "odd", "even",
            "distinct_crossing", "A_changes_answer", "B_changes_answer"} <= evidence


def _fingerprint(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def self_test() -> dict:
    """Bounded semantic regressions and observed structural coverage, 42 seeds."""
    shapes, signatures, geometries, route_counts, room_counts = set(), set(), set(), set(), set()
    motif_counts, colors, inks, layouts = set(), set(), set(), set()
    visible_bends = set()
    elbows = repeats = double_clues = no_room = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                inp = pair["input"]
                info = _parse(inp)
                shapes.add(inp.shape)
                route_counts.add(len(info["routes"]))
                room_counts.add(len(info["rooms"]))
                geometries.add((inp.shape, tuple(tuple(int(v in (WIRE, CROSS, FRAME)) for v in row) for row in inp)))
                motif_counts.update(r["count"] for r in info["routes"])
                colors.update(r["color"] for r in info["routes"])
                inks.update(r["ink"] for r in info["rooms"] if r["ink"] is not None)
                repeats += len({r["color"] for r in info["routes"]}) < len(info["routes"])
                double_clues += sum(all(t["color"] is not None for t in r["terminals"]) for r in info["routes"])
                for route in info["routes"]:
                    bends = sum(len(info["graph"][p]) == 2
                                and abs(next(iter(info["graph"][p])) - list(info["graph"][p])[1]) != 2
                                for p in route["cells"])
                    visible_bends.add(bends)
                no_room += not info["rooms"]
                for room in info["rooms"]:
                    (p, d), (q, e) = room["ports"]
                    elbows += d % 2 != e % 2
                    a, b, x, y = room["rect"]
                    layouts.add((b - a + 1, y - x + 1))
                assert np.array_equal(solve(inp.copy()), pair["output"])
    assert len(signatures) == 42 and len(shapes) >= 25 and len(geometries) >= 150
    assert {1, 2, 3, 4} <= route_counts
    assert {0, 1, 2, 3} <= room_counts and max(room_counts) >= 4
    assert {0, 1, 2, 3} <= motif_counts and max(motif_counts) >= 4
    assert colors == set(INKS) and inks == set(INKS)
    assert len(layouts) >= 10 and elbows and repeats and double_clues and no_room
    assert max(visible_bends) >= 5, "excursions must add real visible bends"
    assert _fingerprint(generate_puzzle(17)) == _fingerprint(generate_puzzle(17))
    for bits in range(512):
        mask = np.array([(bits >> i) & 1 for i in range(9)], bool).reshape(3, 3)
        assert int(_plus_centers(mask).sum()) == _count_by_coordinates(mask)
    block = np.ones((3, 4), bool)
    assert int(_plus_centers(block).sum()) == 2
    shared = np.array([[0, 1, 0, 1, 0], [1, 1, 1, 1, 1], [0, 1, 0, 1, 0]], bool)
    assert int(_plus_centers(shared).sum()) == 2 and shared.sum() == 9
    assert not _plus_centers(CHECKER).any()
    assert int(_plus_centers(np.ones((3, 5), bool)).sum()) == 3
    remaining, sequential = block.copy(), 0
    for r in range(1, 2):
        for c in range(1, 3):
            if all(remaining[r + dr, c + dc] for dr, dc in ((0, 0),) + DIRS):
                sequential += 1
                for dr, dc in ((0, 0),) + DIRS:
                    remaining[r + dr, c + dc] = False
    assert sequential == 1
    # The training witnesses make both inherited mechanisms answer-relevant.
    example = generate_puzzle(0, 2, 1)
    infos = [_parse(p["input"]) for p in example["train"]]
    assert any(r["count"] % 2 and any(t["color"] is None for t in r["terminals"])
               for info in infos for r in info["routes"])
    assert any("distinct_crossing" in _evidence(info) for info in infos)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(9100 + 10 * nt + nq, nt, nq)
            assert len(puzzle["train"]) == nt and len(puzzle["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal example counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in example[s]] for s in example}
    bad["test"][0]["output"][0, 0] = 1
    try:
        validate_generated_puzzle(bad)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("validator accepted nonterminal corruption")
    return dict(seeds=42, pairs=210, distinct_puzzles=len(signatures),
                grid_shapes=len(shapes), structural_geometries=len(geometries),
                route_counts=sorted(route_counts), chamber_counts=sorted(room_counts),
                plus_totals=sorted(motif_counts), chamber_shapes=len(layouts),
                elbow_chambers=int(elbows), repeated_label_pairs=int(repeats),
                already_labeled_routes=int(double_clues), chamber_free_pairs=int(no_room),
                visible_bend_counts=sorted(visible_bends))


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
