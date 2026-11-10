"""Guides choose the seam: joint pattern registration and boundary projection.

Only input pixels are used by solve. Generation builds complete geometric
fragments first, and checks the pixel reconstruction against that construction.
"""
from collections import Counter, deque
import json

import numpy as np


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= x <= 30 for x in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer cells required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "color range")
    return a


def _groups(points):
    todo, result = set(points), []
    while todo:
        start = min(todo)
        todo.remove(start)
        group, stack = {start}, [start]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    group.add(p)
                    stack.append(p)
        result.append(group)
    return result


def _bounds(cells):
    return (min(r for r, c in cells), min(c for r, c in cells),
            max(r for r, c in cells), max(c for r, c in cells))


def _normalize(cells):
    r0, c0, _, _ = _bounds(cells)
    return {(r-r0, c-c0): int(v) for (r, c), v in cells.items()}, (r0, c0)


def _token(cells):
    norm, origin = _normalize(cells)
    return tuple((r, c, v) for (r, c), v in sorted(norm.items())), origin


def _shift(cells, offset):
    return {(r+offset[0], c+offset[1]): v for (r, c), v in cells.items()}


def _union(parts):
    result = {}
    for part in parts:
        for p, v in part.items():
            if p in result and result[p] != v:
                return None
            result[p] = v
    return result


def _decode(input_grid):
    grid = _grid(input_grid)
    h, w = grid.shape
    colors, counts = np.unique(grid, return_counts=True)
    bg = int(colors[counts.argmax()])
    _require(int(counts.max()) * 2 > grid.size, "background must occupy a majority")
    components = _groups(map(tuple, np.argwhere(grid != bg)))
    anchor_cells, raw_parts = {}, []
    for component in components:
        p = next(iter(component))
        if len(component) == 1 and (p[0] in (0, h-1) or p[1] in (0, w-1)):
            anchor_cells[p] = int(grid[p])
        else:
            raw_parts.append({p: int(grid[p]) for p in component})
    _require(len(raw_parts) >= 3, "need a hub, a free fragment, and an end fragment")
    markers = set(anchor_cells.values())
    _require(len(markers) == 2, "exactly two guide identities")
    systems = {}
    for marker in sorted(markers):
        points = [p for p, v in anchor_cells.items() if v == marker]
        horizontal = len({r for r, c in points}) == 1 and all(c in (0, w-1) for r, c in points)
        vertical = len({c for r, c in points}) == 1 and all(r in (0, h-1) for r, c in points)
        _require(horizontal != vertical, "ambiguous boundary guide")
        axis = 0 if horizontal else 1
        guide = points[0][axis]
        _require(0 < guide < grid.shape[axis]-1, "corner guide")
        systems[marker] = (axis, guide)
    fg_counts = Counter(v for part in raw_parts for v in part.values() if v not in markers)
    _require(bool(fg_counts), "no scaffold")
    ordered = fg_counts.most_common()
    _require(len(ordered) >= 3 and ordered[0][1] > ordered[1][1], "scaffold color is not unique")
    body = ordered[0][0]
    parts, pins, registrations = [], {}, {}
    for i, raw in enumerate(raw_parts):
        cells, origin = _normalize(raw)
        handles = [(p, v) for p, v in cells.items() if v in markers]
        _require(len(handles) <= 1, "at most one handle per fragment")
        stamps = {}
        for island in _groups(p for p, v in cells.items() if v != body and v not in markers):
            motif = {p: cells[p] for p in island}
            _require(len(motif) >= 3 and len(set(motif.values())) >= 2, "unrecognizable stamp")
            token, local = _token(motif)
            stamps.setdefault(token, []).append(local)
        _require(bool(stamps), "every fragment needs a registration stamp")
        _require(body in cells.values(), "every fragment needs scaffold material")
        parts.append({'cells': cells, 'origin': origin, 'stamps': stamps})
        if handles:
            (r, c), marker = handles[0]
            _require(marker not in registrations, "handle identity used on two fragments")
            axis, guide = systems[marker]
            delta = guide - (origin[axis] + (r, c)[axis])
            position = list(origin)
            position[axis] += delta
            pins[i] = tuple(position)
            registrations[marker] = (i, (r, c), axis, delta)
    _require(set(registrations) == markers, "each guide needs exactly one handle")
    edges = {i: {} for i in range(len(parts))}
    for i, a in enumerate(parts):
        for j in range(i+1, len(parts)):
            b = parts[j]
            common = set(a['stamps']) & set(b['stamps'])
            if not common:
                continue
            domain = None
            for token in common:
                options = {(r-u, c-v) for r, c in a['stamps'][token] for u, v in b['stamps'][token]}
                domain = options if domain is None else domain & options
            _require(bool(domain), "incompatible shared stamps")
            edges[i][j] = domain
            edges[j][i] = {(-r, -c) for r, c in domain}
    reached, stack = {0}, [0]
    while stack:
        for j in edges[stack.pop()]:
            if j not in reached:
                reached.add(j)
                stack.append(j)
    _require(len(reached) == len(parts), "disconnected registration graph")
    owners = Counter(token for part in parts for token in part['stamps'])
    _require(all(n >= 2 for n in owners.values()), "unshared stamp identity")
    hubs = [i for i, part in enumerate(parts) if any(len(v) > 1 for v in part['stamps'].values())]
    _require(len(hubs) == 1 and hubs[0] in pins, "one pinned hub with duplicate sockets")
    return {'grid': grid, 'bg': bg, 'body': body, 'anchors': anchor_cells,
            'parts': parts, 'edges': edges, 'pins': pins,
            'registrations': registrations, 'hub': hubs[0]}


def _placements(scene, use_guides=True, limit=2):
    """Finite joint constraint search, in full coordinates before clipping."""
    parts, edges = scene['parts'], scene['edges']
    positions = dict(scene['pins']) if use_guides else {scene['hub']: (0, 0)}
    anchors = scene['anchors'] if use_guides else {}
    answers, visits = [], 0

    def consistent():
        for i, pos in positions.items():
            for j, deltas in edges[i].items():
                if j in positions:
                    delta = positions[j][0]-pos[0], positions[j][1]-pos[1]
                    if delta not in deltas:
                        return False
        placed = [_shift(parts[i]['cells'], pos) for i, pos in positions.items()]
        union = _union(placed)
        return union is not None and not (set(union) & set(anchors))

    def search():
        nonlocal visits
        visits += 1
        _require(visits <= 2000, "registration search limit")
        if len(answers) >= limit or not consistent():
            return
        if len(positions) == len(parts):
            answers.append(dict(positions))
            return
        choices = []
        for j in range(len(parts)):
            if j in positions:
                continue
            domain = None
            for i, deltas in edges[j].items():
                if i in positions:
                    proposed = {(positions[i][0]-r, positions[i][1]-c) for r, c in deltas}
                    domain = proposed if domain is None else domain & proposed
            if domain is not None:
                if not domain:
                    return
                choices.append((len(domain), j, domain))
        if not choices:
            return
        _, j, domain = min(choices, key=lambda x: (x[0], x[1]))
        for pos in sorted(domain):
            positions[j] = pos
            search()
            del positions[j]
    search()
    return answers


def _render(scene, positions):
    result = np.full(scene['grid'].shape, scene['bg'], dtype=np.int8)
    union = _union([_shift(part['cells'], positions[i]) for i, part in enumerate(scene['parts'])])
    _require(union is not None, "conflicting registered material")
    _require(not set(union) & set(scene['anchors']), "material collides with fixed anchor")
    for p, v in list(union.items()) + list(scene['anchors'].items()):
        if 0 <= p[0] < result.shape[0] and 0 <= p[1] < result.shape[1]:
            result[p] = v
    return result, union


def solve(input_grid: np.ndarray) -> np.ndarray:
    scene = _decode(input_grid)
    answers = _placements(scene)
    _require(len(answers) == 1, "guide and stamp constraints must have one solution")
    return _render(scene, answers[0])[0]


def _stamp(rng, inks):
    kind = int(rng.integers(7))
    length = int(rng.integers(3, 6))
    if kind == 0:
        points = [(0, c) for c in range(length)]
    elif kind == 1:
        points = [(r, 0) for r in range(length)]
    elif kind == 2:
        points = [(0, 0), (1, 0), (1, 1)]
    elif kind == 3:
        points = [(r, c) for r in range(2) for c in range(2)]
    elif kind == 4:
        points = [(0, 0), (0, 1), (0, 2), (1, 1)]
    elif kind == 5:
        points = [(0, 0), (0, 1), (1, 1), (1, 2)]
    else:
        points = [(r, c) for r in range(3) for c in range(3) if r in (0, 2) or c in (0, 2)]
    turns = int(rng.integers(4))
    for _ in range(turns):
        points = [(c, -r) for r, c in points]
    points = sorted(points)
    values = list(map(int, rng.choice(inks, len(points))))
    values[0], values[-1] = map(int, rng.choice(inks, 2, replace=False))
    return _normalize(dict(zip(points, values)))[0]


def _separated(a, b, gap=1):
    return all(abs(r-u)+abs(c-v) > gap for r, c in a for u, v in b)


def _route(start, end, blocked, rng):
    # Small, bounded BFS: routing is a construction device, not the task rule.
    bounds = _bounds(set(blocked) | {start, end})
    r0, c0, r1, c1 = bounds
    queue, previous = deque([start]), {start: None}
    dirs = [DIRS[int(i)] for i in rng.permutation(4)]
    while queue:
        p = queue.popleft()
        if p == end:
            path = []
            while p is not None:
                path.append(p)
                p = previous[p]
            return path
        for dr, dc in dirs:
            q = p[0]+dr, p[1]+dc
            if (r0-2 <= q[0] <= r1+2 and c0-2 <= q[1] <= c1+2
                    and q not in previous and (q not in blocked or q == end)):
                previous[q] = p
                queue.append(q)
    raise AssertionError("routing failed")


def _join(a, b, all_ink, body, rng):
    pairs = [(abs(r-u)+abs(c-v), (r, c), (u, v)) for r, c in a for u, v in b]
    distance = min(x[0] for x in pairs)
    options = [(p, q) for d, p, q in pairs if d == distance]
    p, q = options[int(rng.integers(len(options)))]
    result = dict(a)
    result.update(b)
    for p in _route(p, q, set(all_ink), rng):
        if p not in all_ink:
            result[p] = body
    return result


def _geometry(rng, nfree, socket, palette, branches=0):
    bg, body, ma, mb = map(int, palette[:4])
    inks = list(map(int, palette[4:4+int(rng.integers(2, 7))]))
    stamps, tokens = [], set()
    for _ in range(nfree+1):
        for attempt in range(50):
            stamp = _stamp(rng, inks)
            token = _token(stamp)[0]
            if token not in tokens:
                tokens.add(token)
                stamps.append(stamp)
                break
        else:
            raise AssertionError("stamp variety exhausted")
    dr, dc = DIRS[int(rng.integers(4))]
    distance = int(rng.integers(6, 10))
    jitter = int(rng.integers(-2, 3))
    decoy = dr*distance + dc*jitter, dc*distance + dr*jitter
    sockets = sorted([(0, 0), decoy])
    hub_ink = [_shift(stamps[0], p) for p in sockets]
    _require(_separated(*hub_ink, gap=2), "sockets too close")
    nodes, ink_parts = [sockets[socket]], list(hub_ink)
    for i in range(1, nfree+1):
        for _ in range(90):
            dr, dc = DIRS[int(rng.integers(4))]
            distance = int(rng.integers(6, 10))
            jitter = int(rng.integers(-2, 3))
            p = nodes[-1][0]+dr*distance+dc*jitter, nodes[-1][1]+dc*distance+dr*jitter
            ink = _shift(stamps[i], p)
            if not all(_separated(ink, other, gap=2) for other in ink_parts):
                continue
            union = _union(ink_parts+[ink])
            r0, c0, r1, c1 = _bounds(union)
            if r1-r0 > 22 or c1-c0 > 22:
                continue
            nodes.append(p)
            ink_parts.append(ink)
            break
        else:
            raise AssertionError("key walk did not fit")
    all_ink = _union(ink_parts)
    pieces = [_join(hub_ink[0], hub_ink[1], all_ink, body, rng)]
    for i in range(nfree):
        pieces.append(_join(_shift(stamps[i], nodes[i]), _shift(stamps[i+1], nodes[i+1]),
                            all_ink, body, rng))
    # A side fragment can register at an intermediate stamp and contribute an
    # asymmetric arm. Several pieces may thus share one stamp occurrence.
    # This changes the registration graph and output branching, not the rule.
    for branch in range(branches):
        key_index = int(rng.integers(1, nfree+1))
        leaf = _shift(stamps[key_index], nodes[key_index])
        a, b, c, d = _bounds(leaf)
        ends = [(a-5, b), (c+5, d), (a, b-5), (c, d+5)]
        rng.shuffle(ends)
        for end in ends:
            if end in all_ink:
                continue
            aa, bb, cc, dd = _bounds(set(all_ink) | {end})
            if cc-aa > 24 or dd-bb > 24:
                continue
            start = min(leaf, key=lambda p: abs(p[0]-end[0])+abs(p[1]-end[1]))
            for p in _route(start, end, set(all_ink), rng):
                if p not in all_ink:
                    leaf[p] = body
            prior = _union(pieces)
            if len(set(leaf)-set(prior)) >= 2:
                pieces.append(leaf)
                break
        else:
            raise AssertionError("no contributing side branch")
    terminal = _shift(stamps[-1], nodes[-1])
    # Give the end fragment an eccentric body and a visible registration handle.
    r0, c0, r1, c1 = _bounds(terminal)
    endpoints = [(r0-int(rng.integers(4, 8)), c0),
                 (r1+int(rng.integers(4, 8)), c1),
                 (r0, c0-int(rng.integers(4, 8))),
                 (r1, c1+int(rng.integers(4, 8)))]
    rng.shuffle(endpoints)
    for end in endpoints:
        if end in all_ink:
            continue
        full = set(all_ink) | {end}
        a, b, c, d = _bounds(full)
        if c-a > 24 or d-b > 24:
            continue
        start = min(terminal, key=lambda p: abs(p[0]-end[0])+abs(p[1]-end[1]))
        path = _route(start, end, set(all_ink), rng)
        for p in path:
            if p not in all_ink:
                terminal[p] = body
        break
    else:
        raise AssertionError("no terminal extension")
    pieces.append(terminal)
    # Independent local branches and occasional hollow scaffold loops.
    for part in pieces:
        for _ in range(int(rng.integers(0, 9))):
            candidates = sorted({(r+dr, c+dc) for (r, c), v in part.items() if v == body
                                 for dr, dc in DIRS} - set(part) - set(all_ink))
            if candidates:
                p = candidates[int(rng.integers(len(candidates)))]
                a, b, c, d = _bounds(set(part) | {p})
                if c-a <= 13 and d-b <= 13:
                    part[p] = body
        if rng.random() < .28:
            bases = [p for p, v in part.items() if v == body]
            if bases:
                r, c = bases[int(rng.integers(len(bases)))]
                hh, ww = int(rng.integers(2, 5)), int(rng.integers(2, 5))
                sr, sc = int(rng.choice([-1, 1])), int(rng.choice([-1, 1]))
                loop = {(r+sr*u, c+sc*v) for u in range(hh+1) for v in range(ww+1)
                        if u in (0, hh) or v in (0, ww)}
                if not loop & set(all_ink):
                    part.update({p: body for p in loop})
    for part, marker in ((pieces[0], ma), (pieces[-1], mb)):
        candidates = [p for p, v in part.items() if v == body
                      and all(abs(p[0]-r)+abs(p[1]-c) > 1 for r, c in all_ink)]
        _require(bool(candidates), "handle cannot be separated from ink")
        part[candidates[int(rng.integers(len(candidates)))]] = marker
    union = _union(pieces)
    _require(union is not None, "geometry conflicts")
    r0, c0, r1, c1 = _bounds(union)
    _require(r1-r0 <= 25 and c1-c0 <= 25, "geometry too large")
    counts = Counter(v for part in pieces for v in part.values() if v not in (ma, mb))
    _require(counts[body] > max(n for v, n in counts.items() if v != body), "scaffold not modal")
    # Both sockets must remain possible if boundary information is removed.
    delta = sockets[1-socket][0]-nodes[0][0], sockets[1-socket][1]-nodes[0][1]
    _require(_union([pieces[0]]+[_shift(p, delta) for p in pieces[1:]]) is not None,
             "wrong socket eliminated by geometry alone")
    return pieces, (bg, body, ma, mb)


def _pack(pieces, origins, axes, anchors, shape, bg, rng):
    norms = [_normalize(p)[0] for p in pieces]
    h, w = shape
    order = [0, len(pieces)-1] + sorted(range(1, len(pieces)-1), key=lambda i: -len(pieces[i]))
    for _ in range(18):
        grid = np.full(shape, bg, np.int8)
        blocked = np.zeros(shape, bool)
        for p, v in anchors.items():
            grid[p] = v
            r, c = p
            blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
        okay = True
        for i in order:
            part = norms[i]
            ph, pw = max(r for r, c in part)+1, max(c for r, c in part)+1
            if i in axes:
                axis = axes[i]
                if axis == 0:
                    options = [(r, origins[i][1]) for r in range(h-ph+1) if r != origins[i][0]]
                else:
                    options = [(origins[i][0], c) for c in range(w-pw+1) if c != origins[i][1]]
            else:
                options = [(r, c) for r in range(h-ph+1) for c in range(w-pw+1)]
            rng.shuffle(options)
            for rr, cc in options:
                if rr < 0 or cc < 0 or rr+ph > h or cc+pw > w:
                    continue
                if any(blocked[r+rr, c+cc] for r, c in part):
                    continue
                for (r, c), v in part.items():
                    r, c = r+rr, c+cc
                    grid[r, c] = v
                    blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
                break
            else:
                okay = False
                break
        if okay:
            return grid
    return None


def _frame(pieces, colors, axes_pair, clipped, rng):
    bg, body, ma, mb = colors
    union = _union(pieces)
    r0, c0, r1, c1 = _bounds(union)
    height, width = r1-r0+1, c1-c0+1
    marked = (0, len(pieces)-1)
    axes = dict(zip(marked, axes_pair))
    handles = {i: next(p for p, v in pieces[i].items() if v == marker)
               for i, marker in zip(marked, (ma, mb))}
    for _ in range(16):
        h = int(rng.integers(max(19, height+2), 31))
        w = int(rng.integers(max(19, width+2), 31))
        if h*w <= 2*(sum(map(len, pieces))+4):
            continue
        shifts = []
        for dr in range(-r0-2, h-r1+2):
            for dc in range(-c0-2, w-c1+2):
                outside = not (r0+dr >= 0 and r1+dr < h and c0+dc >= 0 and c1+dc < w)
                if outside != clipped:
                    continue
                legal = True
                for i in marked:
                    axis = axes[i]
                    mark = handles[i][axis]+(dr, dc)[axis]
                    a, b, c, d = _bounds(pieces[i])
                    along_lo = (a+dr, b+dc)[1-axis]
                    along_hi = (c+dr, d+dc)[1-axis]
                    if not (1 <= mark < (h, w)[axis]-1 and 0 <= along_lo <= along_hi < (h, w)[1-axis]):
                        legal = False
                if legal:
                    shifts.append((dr, dc))
        rng.shuffle(shifts)
        for offset in shifts[:18]:
            targets = [_shift(p, offset) for p in pieces]
            target_union = _union(targets)
            origins = [_normalize(p)[1] for p in targets]
            anchors = {}
            for i, marker in zip(marked, (ma, mb)):
                axis = axes[i]
                guide = handles[i][axis]+offset[axis]
                ends = [0, 1] if rng.random() < .3 else [int(rng.integers(2))]
                for end in ends:
                    p = (guide, end*(w-1)) if axis == 0 else (end*(h-1), guide)
                    anchors[p] = marker
            if len(set(anchors.values())) != 2 or set(anchors) & set(target_union):
                continue
            if any(len(g) > 1 for g in _groups(anchors)):
                continue
            inp = _pack(pieces, origins, axes, anchors, (h, w), bg, rng)
            if inp is None:
                continue
            out = np.full((h, w), bg, np.int8)
            for p, v in list(target_union.items())+list(anchors.items()):
                if 0 <= p[0] < h and 0 <= p[1] < w:
                    out[p] = v
            return {'input': inp, 'output': out}
    raise AssertionError("could not frame and scatter fragments")


def _analyze(pair):
    scene = _decode(pair['input'])
    answers = _placements(scene)
    _require(len(answers) == 1, "joint placement is not unique")
    positions = answers[0]
    expected, union = _render(scene, positions)
    _require(np.array_equal(expected, _grid(pair['output'])), "wrong registered union")
    _require(not np.array_equal(expected, scene['grid']), "identity example")
    relaxed = _placements(scene, use_guides=False)
    _require(len(relaxed) >= 2, "boundary guides do not resolve ambiguity")
    # Hold the hub at its actual target: the ambiguity must be the seam, not
    # just the unconstrained absolute position of the entire assembly. Both
    # alternatives must give different visible pictures on this very canvas.
    hub_offset = scene['pins'][scene['hub']]
    visible_alternatives = set()
    for relative in relaxed:
        candidate = {i: (p[0]+hub_offset[0], p[1]+hub_offset[1]) for i, p in relative.items()}
        candidate_union = _union([_shift(part['cells'], candidate[i])
                                  for i, part in enumerate(scene['parts'])])
        if not set(candidate_union) & set(scene['anchors']):
            visible_alternatives.add(_signature(_render(scene, candidate)[0]))
    _require(len(visible_alternatives) >= 2, "guide ablation has no visible seam ambiguity")
    # Ablate A: project marked components, leave every unmarked component where it was.
    naive = {i: scene['pins'].get(i, part['origin']) for i, part in enumerate(scene['parts'])}
    raw = np.full_like(expected, scene['bg'])
    for i, part in enumerate(scene['parts']):
        for p, v in _shift(part['cells'], naive[i]).items():
            if 0 <= p[0] < raw.shape[0] and 0 <= p[1] < raw.shape[1]:
                raw[p] = v
    for p, v in scene['anchors'].items():
        raw[p] = v
    _require(not np.array_equal(raw, expected), "pattern registration is unnecessary")
    hub = scene['hub']
    duplicates = [(t, sorted(v)) for t, v in scene['parts'][hub]['stamps'].items() if len(v) > 1]
    _require(len(duplicates) == 1 and len(duplicates[0][1]) == 2, "two-socket hub required")
    token, sockets = duplicates[0]
    matching = []
    for j, part in enumerate(scene['parts']):
        if j != hub and token in part['stamps']:
            for local in part['stamps'][token]:
                world = local[0]+positions[j][0], local[1]+positions[j][1]
                matching += [k for k, p in enumerate(sockets)
                             if (p[0]+positions[hub][0], p[1]+positions[hub][1]) == world]
    _require(len(matching) == 1, "one socket must be selected")
    h, w = expected.shape
    clipped = sum(not (0 <= r < h and 0 <= c < w) for r, c in union)
    overlap = sum(len(p['cells']) for p in scene['parts'])-len(union)
    _require(overlap >= 3, "no actual material merging")
    # Distance through the stamp graph measures propagation independently of
    # extra side fragments sharing an intermediate key.
    distances, queue = {hub: 0}, deque([hub])
    while queue:
        for j in scene['edges'][queue.popleft()]:
            if j not in distances:
                distances[j] = 1+min(distances[i] for i in scene['edges'][j] if i in distances)
                queue.append(j)
    other_pin = next(i for i in scene['pins'] if i != hub)
    leaves = sum(len(part['stamps']) == 1 for i, part in enumerate(scene['parts']) if i not in scene['pins'])
    return {'free': len(scene['parts'])-2, 'depth': distances[other_pin]-1, 'side_fragments': leaves,
            'socket': matching[0], 'clipped': clipped,
            'shape': tuple(expected.shape), 'background': scene['bg'], 'body': scene['body'],
            'axes': tuple(x[2] for x in scene['registrations'].values()),
            'stamps': tuple(sorted({t for p in scene['parts'] for t in p['stamps']})),
            'overlap': overlap, 'scene': scene}


def _signature(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _count(value, low, high, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name+" must be an integer")
    if not low <= int(value) <= high:
        raise ValueError(name+" is out of range")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt = _count(num_train, 2, 10, 'num_train')
    nq = _count(num_test, 1, 4, 'num_test')
    if nt <= nq:
        raise ValueError("num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base_palette = rng.permutation(10)
    stable = bool(rng.random() < .3)
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            nfree = 3 if split == 'test' else (1+index if index < 2 else int(rng.integers(1, 3)))
            socket = index if split == 'train' and index < 2 else int(rng.integers(2))
            clipped = (index == 1) if split == 'train' and index < 2 else bool(rng.random() < .4)
            axes = (0, 1) if split == 'train' and index == 0 else tuple(map(int, rng.integers(2, size=2)))
            branches = (int(rng.integers(3)) if split == 'test' else
                        0 if index == 0 else 1 if index == 1 else int(rng.integers(2)))
            for attempt in range(100):
                palette = base_palette if stable else rng.permutation(10)
                try:
                    pieces, colors = _geometry(rng, nfree, socket, palette, branches)
                    pair = _frame(pieces, colors, axes, clipped, rng)
                    info = _analyze(pair)
                    key = _signature(pair['input'])
                    if key in seen or info['socket'] != socket:
                        continue
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
                except AssertionError:
                    continue
            else:
                raise RuntimeError("bounded generation exhausted")
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "pair counts")
    seen, stats = set(), {'train': [], 'test': []}
    for split in puzzle:
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair keys")
            for value in pair.values():
                _grid(value)
            key = _signature(pair['input'])
            _require(key not in seen, "duplicate input")
            seen.add(key)
            stats[split].append(_analyze(pair))
    _require({s['socket'] for s in stats['train']} == {0, 1}, "both socket choices need demonstrations")
    _require(any(s['clipped'] for s in stats['train']), "clipping needs a demonstration")
    _require(any(not s['clipped'] for s in stats['train']), "complete assembly needs a demonstration")
    _require({a for s in stats['train'] for a in s['axes']} == {0, 1}, "both guide axes need demonstrations")
    _require(any(s['side_fragments'] for s in stats['train']), "shared-stamp side arm needs a demonstration")


def self_test():
    """Bounded diversity, ablation, metamorphic, corruption and count checks."""
    fingerprints, shapes, layouts, stamps, backgrounds, bodies, axes = set(), set(), set(), set(), set(), set(), set()
    clip_count, overlap_sizes, side_counts = 0, set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        fingerprint = tuple(_signature(p[k]) for s in ('train', 'test') for p in puzzle[s] for k in ('input', 'output'))
        fingerprints.add(fingerprint)
        local = []
        for split in puzzle:
            for pair in puzzle[split]:
                f = _analyze(pair)
                shapes.add(f['shape'])
                stamps.update(f['stamps'])
                backgrounds.add(f['background'])
                bodies.add(f['body'])
                axes.add(f['axes'])
                clip_count += f['clipped'] > 0
                overlap_sizes.add(f['overlap'])
                side_counts.add(f['side_fragments'])
                scene = f['scene']
                local.append(tuple(tuple(sorted(part['cells'].items())) for part in scene['parts']))
                _require(np.array_equal(solve(pair['input']), pair['output']), "solver disagreement")
            depths = [_analyze(p)['depth'] for p in puzzle[split]]
            _require((max(depths) <= 2) if split == 'train' else (min(depths) == 3), "propagation shift")
        layouts.add(tuple(local))
    _require(len(fingerprints) == len(layouts) == 42, "seed diversity")
    _require(len(shapes) >= 45 and len(stamps) >= 100, "geometry and stamp diversity")
    _require(backgrounds == bodies == set(range(10)), "palette role diversity")
    _require(len(axes) == 4 and clip_count >= 42 and len(overlap_sizes) >= 10, "joint mechanism diversity")
    _require({0, 1, 2} <= side_counts, "side-fragment count diversity")
    p = generate_puzzle(42, 2, 1)
    _require(all(np.array_equal(p[s][i][k], generate_puzzle(42, 2, 1)[s][i][k])
                 for s in p for i in range(len(p[s])) for k in ('input', 'output')), "determinism")
    # Global changes of orientation and palette preserve all input-visible roles.
    mapping = np.array([7, 4, 9, 0, 2, 8, 1, 6, 5, 3], np.int8)
    for turns in (1, 2, 3):
        for pair in p['train']+p['test']:
            inp = mapping[np.rot90(pair['input'], turns)]
            out = mapping[np.rot90(pair['output'], turns)]
            _require(np.array_equal(solve(inp), out), "orientation/palette metamorphism")
    bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    out = bad['test'][0]['output']
    out[0, 0] = (int(out[0, 0])+1) % 10
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    for a, b in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(100+a+b, a, b))
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")


if __name__ == '__main__':
    self_test()
    print(json.dumps({'self_test': 'passed', 'seeds': 42}, separators=(',', ':')))
