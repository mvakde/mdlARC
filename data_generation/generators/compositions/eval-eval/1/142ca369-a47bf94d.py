"""Wired mirrors: cable pairing transports a diagonal reflection to another face.

All interpretations come from input geometry. No parent module, stored puzzle,
seed, or construction path is consulted by solve().
"""
from collections import deque
from functools import lru_cache

import numpy as np


CARDINAL = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _add(p, d, steps=1):
    return p[0] + steps * d[0], p[1] + steps * d[1]


def _neg(d):
    return -d[0], -d[1]


def _tangent(normal):
    # Clockwise tangent in row/column coordinates.
    return normal[1], -normal[0]


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


@lru_cache(maxsize=4096)
def _box(center, radius=1):
    return frozenset((center[0] + a, center[1] + b)
            for a in range(-radius, radius + 1)
            for b in range(-radius, radius + 1))


def _mirror_fits(center, normal, shape):
    """Bounds for the radius-two clearance and the three-cell back lead."""
    return (2 <= center[0] < shape[0] - 2 and
            2 <= center[1] < shape[1] - 2 and
            _inside(_add(center, normal, 4), shape))


def _free_mirror_centers(shape, blocked):
    """Clearance maps for all four faces, computed once per placement stage."""
    h,w = shape
    occupied = np.ones((h+8,w+8), dtype=bool)
    occupied[4:4+h,4:4+w] = False
    for r,c in blocked:
        if 0 <= r < h and 0 <= c < w:
            occupied[r+4,c+4] = True
    clear = np.ones(shape, dtype=bool)
    for dr in range(-2,3):
        for dc in range(-2,3):
            clear &= ~occupied[4+dr:4+dr+h,4+dc:4+dc+w]
    result = []
    for dr,dc in CARDINAL:
        valid = clear.copy()
        for step in (3,4):
            valid &= ~occupied[4+step*dr:4+step*dr+h,4+step*dc:4+step*dc+w]
        result.append(valid)
    return result


def _l_cells(elbow, direction):
    return {elbow, (elbow[0] - direction[0], elbow[1]),
            (elbow[0], elbow[1] - direction[1])}


def _components(cells):
    unseen, groups = set(cells), []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        todo, group = [first], {first}
        while todo:
            p = todo.pop()
            for d in CARDINAL:
                q = _add(p, d)
                if q in unseen:
                    unseen.remove(q)
                    group.add(q)
                    todo.append(q)
        groups.append(group)
    return groups


def _rectangles(cells):
    result = []
    for group in _components(cells):
        rs, cs = zip(*group)
        r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
        assert r1 > r0 and c1 > c0, "mask must have two nontrivial dimensions"
        assert len(group) == (r1 - r0 + 1) * (c1 - c0 + 1)
        result.append((r0, r1, c0, c1))
    return result


def _mask_links(rect, visible, endpoints):
    r0, r1, c0, c1 = rect
    boundary = [((r0 - 1, c), (1, 0)) for c in range(c0, c1 + 1)]
    boundary += [((r1 + 1, c), (-1, 0)) for c in range(c0, c1 + 1)]
    boundary += [((r, c0 - 1), (0, 1)) for r in range(r0, r1 + 1)]
    boundary += [((r, c1 + 1), (0, -1)) for r in range(r0, r1 + 1)]
    ports = []
    for p, inward in boundary:
        degree = sum(_add(p, d) in visible for d in CARDINAL)
        if p in visible and (degree in (1, 3) or endpoints.get(p) == inward):
            ports.append((p, inward))
    assert len(ports) >= 2 and len(ports) % 2 == 0
    if len(ports) == 2:
        return [(ports[0], ports[1])]
    remaining, links = set(ports), []
    while remaining:
        a = min(remaining)
        remaining.remove(a)
        p, d = a
        choices = [b for b in remaining if b[1] == _neg(d)
                   and (b[0][1] == p[1] if d[0] else b[0][0] == p[0])]
        assert len(choices) == 1, "opaque region hides an unspecified pairing"
        b = choices[0]
        remaining.remove(b)
        links.append((a, b))
    return links


def _trace_wires(grid, mirrors, carrier, cross, masks):
    visible = set(map(tuple, np.argwhere(grid == carrier)))
    if cross is not None:
        visible |= set(map(tuple, np.argwhere(grid == cross)))
    endpoints = {m['back']: m['normal'] for m in mirrors}
    assert len(endpoints) == len(mirrors)
    graph = {p: {d: (_add(p, d), _neg(d)) for d in CARDINAL
                 if _add(p, d) in visible} for p in visible}
    mask_info = []
    for rect in masks:
        links = _mask_links(rect, visible, endpoints)
        for (p, d), (q, e) in links:
            assert d not in graph[p] and e not in graph[q]
            graph[p][d], graph[q][e] = (q, e), (p, d)
        mask_info.append((rect, links))
    crossings = []
    for p, edges in graph.items():
        if p in endpoints:
            assert len(edges) == 1 and endpoints[p] in edges, "invalid back lead"
        elif len(edges) == 4:
            assert cross is not None and grid[p] == cross, "unmarked wire junction"
            crossings.append(p)
        else:
            assert len(edges) == 2, "broken or branching wire"
            assert cross is None or grid[p] != cross, "cross marker lacks four arms"
    site_index = {m['back']: i for i, m in enumerate(mirrors)}
    partner, routes, used = {}, [], set()
    for i, m in enumerate(mirrors):
        if i in partner:
            continue
        p, incoming = m['back'], _neg(m['normal'])
        states, route = set(), []
        while True:
            assert (p, incoming) not in states, "closed cable cycle"
            states.add((p, incoming))
            route.append(p)
            used.add(p)
            if p in site_index and p != m['back']:
                j = site_index[p]
                assert j not in partner
                partner[i], partner[j] = j, i
                break
            edges = graph[p]
            if p == m['back']:
                outgoing = m['normal']
            elif len(edges) == 4:
                assert incoming in edges
                outgoing = _neg(incoming)
            else:
                assert incoming in edges
                outgoing = next(d for d in edges if d != incoming)
            p, incoming = edges[outgoing]
        routes.append((i, j, route))
    assert used == visible, "unattached cable fragments"
    return partner, routes, mask_info, crossings


def _parse(input_grid):
    g = np.asarray(input_grid)
    assert g.ndim == 2 and np.issubdtype(g.dtype, np.integer)
    assert all(1 <= x <= 30 for x in g.shape) and np.all((g >= 0) & (g <= 9))
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    assert counts[bg] > g.size / 2, "background must be a strict majority"
    # Compare the eight ring offsets at once, preserving row-major frame order.
    corners = g[:-2, :-2]
    is_frame = (corners != bg) & (corners != g[1:-1, 1:-1])
    for dr, dc in ((0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)):
        is_frame &= corners == g[dr:g.shape[0]-2+dr, dc:g.shape[1]-2+dc]
    frames = [((int(r)+1, int(c)+1), int(corners[r,c]))
              for r,c in np.argwhere(is_frame)]
    assert len(frames) >= 4 and len(frames) % 2 == 0
    assert len({v for _, v in frames}) == 1
    frame = frames[0][1]
    footprints = set()
    for center, _ in frames:
        assert not footprints & _box(center), "overlapping mirrors"
        footprints |= _box(center)
    assert {tuple(p) for p in np.argwhere(g == frame)} == (
        footprints - {p for p, _ in frames})
    # A unique shared color occupies exactly one back position per frame.
    candidate_roles = []
    for shade in set(map(int, np.unique(g))) - {bg, frame}:
        normals = [[d for d in CARDINAL if _inside(_add(p, d, 2), g.shape)
                    and g[_add(p, d, 2)] == shade] for p, _ in frames]
        if all(len(ds) == 1 for ds in normals):
            candidate_roles.append((shade, normals))
    assert len(candidate_roles) == 1, "ambiguous wire role"
    carrier, normals = candidate_roles[0]
    mirrors = []
    for (center, _), ds in zip(frames, normals):
        n = ds[0]
        contact, back = _add(center, n, -2), _add(center, n, 2)
        assert _inside(contact, g.shape) and g[contact] == bg
        label = int(g[center])
        assert label not in (carrier, frame)
        mirrors.append(dict(center=center, normal=n, contact=contact, back=back,
                            clue=None if label == bg else label))
    remaining = {tuple(p) for p in np.argwhere(g != bg)} - footprints
    remaining -= {p for p in remaining if g[p] == carrier}
    sources, auxiliary = [], {}
    for shade in sorted({int(g[p]) for p in remaining}):
        cells = {p for p in remaining if g[p] == shade}
        for group in _components(cells):
            rs, cs = zip(*group)
            if len(group) == 3 and max(rs) - min(rs) == max(cs) - min(cs) == 1:
                missing = next(iter({(r, c) for r in (min(rs), max(rs))
                                     for c in (min(cs), max(cs))} - group))
                elbow = (min(rs) + max(rs) - missing[0], min(cs) + max(cs) - missing[1])
                d = (elbow[0] - missing[0], elbow[1] - missing[1])
                sources.append(dict(cells=group, elbow=elbow, direction=d, color=shade))
            else:
                auxiliary.setdefault(shade, set()).update(group)
    assert 1 <= len(sources) <= 3
    cross, mask, rectangles = None, None, []
    for shade, cells in auxiliary.items():
        try:
            rects = _rectangles(cells)
        except AssertionError:
            assert cross is None and all(len(x) == 1 for x in _components(cells))
            cross = shade
        else:
            assert mask is None
            mask, rectangles = shade, rects
    label_colors = {s['color'] for s in sources} | {m['clue'] for m in mirrors if m['clue'] is not None}
    assert label_colors.isdisjoint({bg, frame, carrier, cross, mask})
    partners, routes, mask_info, crossings = _trace_wires(g, mirrors, carrier, cross, rectangles)
    for i, j, _ in routes:
        clues = {mirrors[k]['clue'] for k in (i, j)} - {None}
        assert len(clues) == 1, "a cable needs agreeing clues at one or both ends"
        label = next(iter(clues))
        mirrors[i]['label'] = mirrors[j]['label'] = label
    return dict(grid=g, background=bg, frame=frame, carrier=carrier, cross=cross,
                mask=mask, mirrors=mirrors, partners=partners, wires=routes,
                masks=mask_info, crossings=crossings, sources=sources)


def _optics(info, mode='wired'):
    g, mirrors = info['grid'], info['mirrors']
    output = g.copy()
    for m in mirrors:
        output[m['center']] = m['label']
    contacts = {}
    for i, m in enumerate(mirrors):
        assert m['contact'] not in contacts, "two mirrors share a front contact"
        contacts[m['contact']] = i
    painted, histories = {}, []

    def paint(p, shade):
        assert g[p] == info['background'], "light strikes a solid object"
        assert p not in painted or painted[p] == shade, "conflicting light overlap"
        painted[p] = shade
        output[p] = shade

    for source in info['sources']:
        p, d, shade = _add(source['elbow'], source['direction']), source['direction'], source['color']
        seen, history = set(), []
        while _inside(p, g.shape):
            state = p, d
            assert state not in seen, "optical loop"
            seen.add(state)
            assert len(seen) <= 4 * g.size
            index = contacts.get(p)
            if index is not None and _dot(d, mirrors[index]['normal']) == 1 and mode != 'ignore':
                entry = mirrors[index]
                shade = entry['label']
                paint(p, shade)
                target = index if mode == 'local' else info['partners'][index]
                exit_site = mirrors[target]
                tangent_sign = _dot(d, _tangent(entry['normal']))
                n, t = exit_site['normal'], _tangent(exit_site['normal'])
                d = (-n[0] + tangent_sign * t[0], -n[1] + tangent_sign * t[1])
                assert d in DIAGONAL
                p = exit_site['contact']
                paint(p, shade)
                history.append((index, target, p, d, shade))
            else:
                paint(p, shade)
            p = _add(p, d)
        histories.append(history)
    return output, dict(painted=painted, histories=histories)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Decode the scene and simulate the same fixed rule for every family member."""
    return _optics(_parse(input_grid))[0]


def _wire_graph(wires):
    return {p: {d for d in CARDINAL if _add(p, d) in wires} for p in wires}


@lru_cache(maxsize=128)
def _routing_geometry(shape):
    """Input-independent adjacency, shared by searches on the same canvas size."""
    h, w = shape
    return tuple((p-w if p >= w else -1,
                  p+1 if p % w < w-1 else -1,
                  p+w if p < (h-1)*w else -1,
                  p-1 if p % w else -1) for p in range(h*w))


def _route_between(a, b, blocked, existing, shape, rng):
    """Direction-aware BFS; cross an existing straight strand without joining it.

    Use integer states and precomputed local edge masks rather than constructing
    neighboring-coordinate sets at every BFS transition. The traversal order,
    admissible transitions, and shortest-path tie breaking stay the same.
    """
    neighbors = _routing_geometry(shape)
    size, w = len(neighbors), shape[1]
    occupied, unavailable = bytearray(size), bytearray(size)
    for r,c in existing:
        occupied[r*w+c] = 1
    for r,c in blocked:
        if 0 <= r < shape[0] and 0 <= c < w:
            unavailable[r*w+c] = 1
    # Each bit says whether the neighboring cell in that direction is occupied.
    padded = np.pad(np.frombuffer(occupied, dtype=np.uint8).reshape(shape), 1)
    bits = (padded[:-2,1:-1] | padded[1:-1,2:] << 1 |
            padded[2:,1:-1] << 2 | padded[1:-1,:-2] << 3).ravel()
    nearby = bits.tobytes()
    axes = (5,10,5,10)  # vertical / horizontal direction bit sets
    on_wire = np.frombuffer(occupied, dtype=np.uint8).astype(bool)
    available = ~np.frombuffer(unavailable, dtype=np.uint8).astype(bool)
    vertical = (available & ((on_wire & (bits == 10)) |
                (~on_wire & ((bits == 0) | (bits == 1) | (bits == 4))))).tobytes()
    horizontal = (available & ((on_wire & (bits == 5)) |
                  (~on_wire & ((bits == 0) | (bits == 2) | (bits == 8))))).tobytes()
    arrival = (vertical,horizontal,vertical,horizontal)
    order = list(range(4))
    rng.shuffle(order)
    first, target = a[0]*w+a[1], b[0]*w+b[1]
    start = size*4  # the one state with no incoming direction
    previous = [-1] * (start+1)
    previous[start] = -2
    queue, finish = deque([start]), None
    while queue:
        state = queue.popleft()
        p, last = (first,-1) if state == start else divmod(state,4)
        if p == target:
            finish = state
            break
        if occupied[p] or nearby[p]:
            if last < 0 or (not occupied[p] and nearby[p] & ~axes[last]):
                continue
            directions = (last,)
        else:
            directions = order
        for d in directions:
            if last >= 0 and d == (last+2) % 4:
                continue
            q = neighbors[p][d]
            if q < 0 or not arrival[d][q]:
                continue
            next_state = q*4+d
            if previous[next_state] == -1:
                previous[next_state] = state
                queue.append(next_state)
    if finish is None:
        return None
    path = []
    while finish != -2:
        p = first if finish == start else finish//4
        path.append(divmod(p,w))
        finish = previous[finish]
    return path[::-1]


def _planned_scene(rng, shape, chain, extra, lesson):
    """Plan open-space optical segments, then connect scattered mirror backs."""
    h, w = shape
    direction = DIAGONAL[int(rng.integers(4))]
    elbow = tuple(int(rng.integers(2, length // 3)) if step > 0
                  else int(rng.integers(length - length // 3, length - 2))
                  for length, step in zip(shape, direction))
    source = dict(elbow=elbow, direction=direction, cells=_l_cells(elbow, direction))
    solids, protected, mirrors, pairings = set(source['cells']), set(), [], []
    anchor, d = elbow, direction
    for k in range(chain):
        candidates = []
        forbidden = solids | protected
        free = _free_mirror_centers(shape, forbidden)
        for distance in range(3, max(h, w)):
            p = _add(anchor, d, distance)
            if not _inside(p, shape):
                break
            for n in ((d[0], 0), (0, d[1])):
                center = _add(p, n, 2)
                if _inside(center, shape) and free[CARDINAL.index(n)][center]:
                    flight = {_add(anchor, d, j) for j in range(1, distance + 1)}
                    if not flight & solids:
                        candidates.append((p, n, center, flight))
        if not candidates:
            return None
        contact, normal, center, flight = candidates[int(rng.integers(len(candidates)))]
        if lesson == 0 and k == 0:
            exit_normal = normal
        elif lesson == 1 and k == 0:
            exit_normal = _tangent(normal) if rng.integers(2) else _neg(_tangent(normal))
        else:
            exit_normal = CARDINAL[int(rng.integers(4))]
        tangent_sign = _dot(d, _tangent(normal))
        tangent = _tangent(exit_normal)
        out_d = (-exit_normal[0] + tangent_sign * tangent[0],
                 -exit_normal[1] + tangent_sign * tangent[1])
        entry_lead = {_add(center, normal, s) for s in (2,3,4)}
        solids |= _box(center) | entry_lead
        protected |= flight
        receivers = []
        forbidden = solids | protected
        free = _free_mirror_centers(shape, forbidden)
        exit_free = free[CARDINAL.index(exit_normal)]
        for _ in range(80):
            other = (int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2)))
            front = _add(other, exit_normal, -2)
            start = {front, _add(front, out_d), _add(front, out_d, 2)}
            lead = {_add(other, exit_normal, s) for s in (2, 3, 4)}
            if (exit_free[other] and _inside(_add(front, out_d, 2), shape)
                    and not start & solids):
                new_solids = _box(other) | lead
                occupied = solids | new_solids
                if k < chain - 1:
                    viable = False
                    for distance in range(3, max(h,w)):
                        next_front = _add(front, out_d, distance)
                        if not _inside(next_front, shape) or next_front in occupied:
                            break
                        for nn in ((out_d[0],0),(0,out_d[1])):
                            cc = _add(next_front,nn,2)
                            if (not _inside(cc,shape) or not free[CARDINAL.index(nn)][cc]):
                                continue
                            space = _box(cc,2) | {_add(cc,nn,s) for s in (3,4)}
                            if not space & new_solids:
                                viable = True
                        if viable:
                            break
                    if not viable:
                        continue
                else:
                    length = min(h-1-front[0] if out_d[0] > 0 else front[0],
                                 w-1-front[1] if out_d[1] > 0 else front[1])
                    tail_points = {_add(front,out_d,s) for s in range(1,length+1)}
                    if len(tail_points) < 2 or tail_points & occupied:
                        continue
                receivers.append((other, front))
        if not receivers:
            return None
        other, front = receivers[int(rng.integers(len(receivers)))]
        i = len(mirrors)
        mirrors.extend([dict(center=center, normal=normal, contact=contact),
                        dict(center=other, normal=exit_normal, contact=front)])
        pairings.append((i, i + 1))
        solids |= _box(other) | {_add(other, exit_normal, s) for s in (2,3,4)}
        protected.add(front)
        anchor, d = front, out_d
    tail = set()
    for steps in range(1, max(h, w) + 1):
        p = _add(anchor, d, steps)
        if not _inside(p, shape):
            break
        if p in solids:
            return None
        tail.add(p)
    if len(tail) < 2:
        return None
    protected |= tail
    for _ in range(extra):
        pair = []
        for _ in range(2):
            forbidden = solids | protected
            for attempt in range(100):
                center = (int(rng.integers(2, h - 2)), int(rng.integers(2, w - 2)))
                n = CARDINAL[int(rng.integers(4))]
                area = _box(center, 2)
                lead = {_add(center, n, s) for s in (2, 3, 4)}
                if (_mirror_fits(center, n, shape) and
                        not area & forbidden and not lead & protected):
                    break
            else:
                return None
            m = dict(center=center, normal=n, contact=_add(center, n, -2))
            pair.append(m)
            solids |= _box(center) | lead
            protected.add(m['contact'])
        i = len(mirrors)
        mirrors.extend(pair)
        pairings.append((i, i + 1))
    # Reserve every other side of the frame: only its back can touch a wire.
    blocked = protected | source['cells'] | { _add(p, d) for p in source['cells'] for d in CARDINAL }
    all_leads, tips = set(), []
    for m in mirrors:
        lead = {_add(m['center'], m['normal'], s) for s in (2, 3, 4)}
        own_frame_and_lead = _box(m['center']) | lead
        if lead & (protected | (solids - own_frame_and_lead) | all_leads):
            return None
        all_leads |= lead
        m['back'] = _add(m['center'], m['normal'], 2)
        tips.append(_add(m['center'], m['normal'], 4))
        blocked |= _box(m['center'], 2) - lead
    if any(sum(_add(p, d) in all_leads for d in CARDINAL) > 2 for p in all_leads):
        return None
    wires, wire_paths = set(), []
    ordering = list(rng.permutation(len(pairings)))
    for pi in ordering:
        i, j = pairings[int(pi)]
        own = {_add(mirrors[k]['center'], mirrors[k]['normal'], s)
               for k in (i, j) for s in (2, 3, 4)}
        unavailable = blocked | (all_leads - own)
        # Other reserved leads cannot even be touched before their turn.
        unavailable |= {_add(p, d) for p in all_leads - own for d in CARDINAL} - own
        path = _route_between(tips[i], tips[j], unavailable, wires, shape, rng)
        if path is None:
            return None
        candidate = wires | set(path) | own
        graph = _wire_graph(candidate)
        ends = {mirrors[k]['back'] for pidx in ordering[:len(wire_paths) + 1]
                for k in pairings[int(pidx)]}
        if any(len(edges) != (1 if p in ends else 4 if len(edges) == 4 else 2)
               for p, edges in graph.items()):
            return None
        wire_paths.append((i, j, path))
        wires = candidate
    return dict(source=source, mirrors=mirrors, pairings=pairings, wires=wires,
                protected=protected, chain=chain)


def _mask_types(parsed):
    return {'multi' if len(links)>1 else 'straight' if links[0][0][1]==_neg(links[0][1][1])
            else 'elbow' for _,links in parsed['masks']}


def _apply_masks(grid, info, mask_color, rng, wanted, allowed=None, keep_crossing=False):
    g = grid.copy()
    wires = sorted(info['wires'])
    for _ in range(70):
        if wanted == 0:
            break
        p = wires[int(rng.integers(len(wires)))]
        hh, ww = int(rng.integers(2, 5)), int(rng.integers(2, 6))
        r0, c0 = p[0] - int(rng.integers(hh)), p[1] - int(rng.integers(ww))
        cells = {(r0 + a, c0 + b) for a in range(hh) for b in range(ww)}
        margin = cells | {_add(q, d) for q in cells for d in CARDINAL}
        if (not all(_inside(q, g.shape) for q in cells) or
                cells & info['protected'] or
                any(g[q] == mask_color for q in margin if _inside(q, g.shape)) or
                any(g[q] not in (info['bg'], info['carrier'], info['cross']) for q in cells)):
            continue
        trial = g.copy()
        for q in cells:
            trial[q] = mask_color
        try:
            parsed = _parse(trial)
            actual = {frozenset((info['mirrors'][i]['center'], info['mirrors'][j]['center']))
                      for i, j in info['pairings']}
            inferred = {frozenset((parsed['mirrors'][i]['center'], parsed['mirrors'][j]['center']))
                        for i, j, _ in parsed['wires']}
            assert actual == inferred
            assert allowed is None or _mask_types(parsed) <= allowed
            assert not keep_crossing or parsed['crossings']
            _optics(parsed)
        except (AssertionError, KeyError, StopIteration):
            continue
        g, wanted = trial, wanted - 1
    return g


def _add_spare(plan, shape, rng):
    """Optionally fit another unvisited connection after the active chain fits.

    Its requested presence never makes an otherwise valid large circuit fail.
    """
    blocked = plan['protected'] | plan['source']['cells']
    blocked |= {_add(p,d) for p in plan['source']['cells'] for d in CARDINAL}
    for m in plan['mirrors']:
        blocked |= _box(m['center'],2)
        blocked |= {_add(m['center'],m['normal'],s) for s in (3,4)}
    for _ in range(100):
        fresh, leads = [], set()
        for side in range(2):
            center = tuple(int(rng.integers(2,n-2)) for n in shape)
            normal = CARDINAL[int(rng.integers(4))]
            lead = {_add(center,normal,s) for s in (2,3,4)}
            area = _box(center,2) | lead
            if (not all(_inside(p,shape) for p in area) or
                    area & (blocked | plan['wires'] | leads) or
                    any(_add(p,d) in plan['wires'] for p in area for d in CARDINAL)):
                break
            fresh.append(dict(center=center,normal=normal,contact=_add(center,normal,-2),
                              back=_add(center,normal,2)))
            leads |= lead
        if len(fresh) != 2 or _box(fresh[0]['center'],2) & _box(fresh[1]['center'],2):
            continue
        unavailable = blocked.copy()
        for m in fresh:
            unavailable |= _box(m['center'],2) - leads
        path = _route_between(_add(fresh[0]['center'],fresh[0]['normal'],4),
                              _add(fresh[1]['center'],fresh[1]['normal'],4),
                              unavailable,plan['wires'],shape,rng)
        if path is None:
            continue
        wires = plan['wires'] | set(path) | leads
        ends = {m['back'] for m in plan['mirrors'] + fresh}
        graph = _wire_graph(wires)
        if any(len(edges) != (1 if p in ends else 4 if len(edges)==4 else 2)
               for p,edges in graph.items()):
            continue
        i = len(plan['mirrors'])
        plan['mirrors'] += fresh
        plan['pairings'].append((i,i+1))
        plan['wires'] = wires
        plan['protected'] |= {m['contact'] for m in fresh}
        return


def _make_pair(rng, palette, lesson=None, test=False, mask_types=None):
    bg, carrier, frame, cross, mask = palette[:5]
    choices = palette[5:]
    chain = 1 if lesson == 0 else 2 if lesson == 1 else 3 if test else int(rng.integers(1, 3))
    base_total = max(2, chain)
    for attempt in range(1500):
        shape = (int(rng.integers(26 if test else 20, 31)),
                 int(rng.integers(26 if test else 20, 31)))
        plan = _planned_scene(rng, shape, chain, base_total - chain, lesson)
        if plan is None:
            continue
        if lesson is None and rng.random() < .35:
            _add_spare(plan,shape,rng)
        total = len(plan['pairings'])
        g = np.full(shape, bg, dtype=np.int8)
        for p, edges in _wire_graph(plan['wires']).items():
            g[p] = cross if len(edges) == 4 else carrier
        labels = list(map(int, rng.choice(choices, total, replace=True)))
        if total >= 2 and lesson is not None:
            labels[1] = next(c for c in choices if c != labels[0])
        for k, (i, j) in enumerate(plan['pairings']):
            mode = int(rng.integers(3))  # clue at entry, exit, or both
            if k == 0 and lesson is not None:
                mode = 1 if lesson == 0 else 0
            for site, index in enumerate((i, j)):
                m = plan['mirrors'][index]
                for q in _box(m['center']) - {m['center']}:
                    g[q] = frame
                if mode == 2 or site == mode:
                    g[m['center']] = labels[k]
        source_color = labels[0] if lesson == 0 else int(rng.choice(choices))
        if lesson == 1:
            source_color = next(c for c in choices if c != labels[0])
        for p in plan['source']['cells']:
            g[p] = source_color
        if np.count_nonzero(g == bg) <= g.size / 2:
            continue
        try:
            parsed = _parse(g)
            out, optical = _optics(parsed)
            assert len(optical['histories'][0]) == chain
            actual = {frozenset((plan['mirrors'][i]['center'], plan['mirrors'][j]['center']))
                      for i, j in plan['pairings']}
            inferred = {frozenset((parsed['mirrors'][i]['center'], parsed['mirrors'][j]['center']))
                        for i, j, _ in parsed['wires']}
            assert actual == inferred
            if lesson == 0:
                assert parsed['crossings'], "demonstrate separate strands"
        except (AssertionError, KeyError, StopIteration):
            continue
        plan.update(bg=bg, carrier=carrier, cross=cross)
        wanted = 1 if lesson == 1 else int(rng.integers(3))
        g = _apply_masks(g, plan, mask, rng, wanted, mask_types, lesson == 0)
        # A second L may traverse another open flight. It is optional, never a
        # required split-specific convention, and every scene remains one circuit.
        if lesson is None and rng.random() < .4:
            for _ in range(24):
                d = DIAGONAL[int(rng.integers(4))]
                elbow = (int(rng.integers(1, shape[0] - 1)), int(rng.integers(1, shape[1] - 1)))
                cells = _l_cells(elbow, d)
                margin = cells | {_add(p, e) for p in cells for e in CARDINAL}
                if not all(_inside(p, shape) and g[p] == bg for p in margin):
                    continue
                trial = g.copy()
                for p in cells:
                    trial[p] = int(choices[0])
                try:
                    pinfo = _parse(trial)
                    trial_out, oinfo = _optics(pinfo)
                    if (sum(len(h) for h in oinfo['histories']) < chain or
                            max(map(len, oinfo['histories'])) > (3 if test else 2)):
                        continue
                except (AssertionError, KeyError, StopIteration):
                    continue
                g = trial
                break
        return {'input': g, 'output': solve(g)}
    raise RuntimeError(f"bounded construction exhausted for lesson={lesson}, test={test}")


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    if not low <= int(value) <= high:
        raise ValueError(f'{name} must lie in {low}..{high}')
    return int(value)


def _key(grid):
    g = np.asarray(grid, dtype=np.uint8)
    return g.shape, g.tobytes()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _count(num_train, 'num_train', 2, 10), _count(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError('seed must be an integer')
    rng = np.random.default_rng(seed)
    palette = list(map(int, rng.permutation(10)))
    puzzle, seen, mask_evidence = {'train': [], 'test': []}, set(), set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            lesson = index if split == 'train' and index < 2 else None
            for _ in range(20):
                pair = _make_pair(rng, palette, lesson, split == 'test',
                                  mask_evidence if split == 'test' else None)
                if lesson == 1:
                    first_direction = _parse(puzzle['train'][0]['input'])['sources'][0]['direction']
                    if _parse(pair['input'])['sources'][0]['direction'] == first_direction:
                        pair = {k:np.rot90(v).copy() for k,v in pair.items()}
                if _key(pair['input']) not in seen:
                    break
            else:
                raise RuntimeError('bounded uniqueness attempts exhausted')
            seen.add(_key(pair['input']))
            if split == 'train':
                mask_evidence |= _mask_types(_parse(pair['input']))
            puzzle[split].append(pair)
    rng.shuffle(puzzle['train'])
    return puzzle


def inspect_pair(pair):
    info = _parse(pair['input'])
    _, optical = _optics(info)
    axes = []
    for history in optical['histories']:
        for a, b, _, _, _ in history:
            axes.append((_dot(info['mirrors'][a]['normal'], info['mirrors'][b]['normal'])))
    return dict(shape=tuple(info['grid'].shape), background=info['background'],
                roles=(info['background'], info['carrier'], info['frame'], info['cross'], info['mask']),
                cables=len(info['wires']), sources=len(info['sources']),
                directions=tuple(s['direction'] for s in info['sources']),
                normals=tuple(m['normal'] for m in info['mirrors']),
                crossings=len(info['crossings']), masks=len(info['masks']),
                mask_shapes=tuple((r1-r0+1, c1-c0+1) for (r0,r1,c0,c1), _ in info['masks']),
                jumps=tuple(len(h) for h in optical['histories']), face_relations=tuple(axes),
                clues=tuple(sum(info['mirrors'][k]['clue'] is not None for k in (i,j))
                            for i,j,_ in info['wires']),
                labels=tuple(info['mirrors'][i]['label'] for i,j,_ in info['wires']),
                mask_types=tuple(sorted(_mask_types(info))),
                ray_cells=len(optical['painted']))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, evidence, mask_evidence, directions = set(), set(), set(), set()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            inp, out = np.asarray(pair['input']), np.asarray(pair['output'])
            for g in (inp, out):
                assert g.ndim == 2 and np.issubdtype(g.dtype, np.integer)
                assert all(1 <= x <= 30 for x in g.shape) and np.all((g >= 0) & (g <= 9))
            assert inp.shape == out.shape and _key(inp) not in seen
            seen.add(_key(inp))
            info = _parse(inp)
            if split == 'train':
                mask_evidence |= _mask_types(info)
                directions |= {s['direction'] for s in info['sources']}
            else:
                assert _mask_types(info) <= mask_evidence, 'query mask grammar not demonstrated'
            expected, optical = _optics(info)
            assert np.array_equal(out, expected), 'incorrect wired reflection or label completion'
            allowed = set(optical['painted']) | {m['center'] for m in info['mirrors']}
            changed = set(map(tuple, np.argwhere(inp != out)))
            assert changed and changed <= allowed
            assert set(map(int, np.unique(out))) <= set(map(int, np.unique(inp)))
            assert any(optical['histories']), 'at least one connection must affect the optics'
            for i,j,_ in info['wires']:
                assert out[info['mirrors'][i]['center']] == out[info['mirrors'][j]['center']]
            if split == 'train':
                for s, history in zip(info['sources'], optical['histories']):
                    for a,b,_,_,shade in history:
                        ma, mb = info['mirrors'][a], info['mirrors'][b]
                        if ma['normal'] == mb['normal']:
                            evidence.add('ordinary_reflection')
                        if _dot(ma['normal'], mb['normal']) == 0:
                            evidence.add('rotated_face')
                        if ma['clue'] is None:
                            evidence.add('exit_clue')
                        if mb['clue'] is None:
                            evidence.add('entry_clue')
                        evidence.add('unchanged_color' if shade == s['color'] else 'changed_color')
                if info['crossings']:
                    evidence.add('crossing')
    required = {'ordinary_reflection', 'rotated_face', 'entry_clue', 'exit_clue',
                'unchanged_color', 'changed_color', 'crossing'}
    assert required <= evidence, f'missing demonstration evidence: {required - evidence}'
    assert len(directions) >= 2


def _equal(a, b):
    return all(len(a[s]) == len(b[s]) and all(np.array_equal(p[k], q[k])
               for p,q in zip(a[s],b[s]) for k in ('input','output')) for s in a)


def self_test():
    records, normalized, puzzle_keys = [], set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        puzzle_keys.add(tuple(_key(p['input']) for s in puzzle for p in puzzle[s]))
        train_jumps, test_jumps = [], []
        for split in ('train','test'):
            for pair in puzzle[split]:
                info = _parse(pair['input'])
                prediction = solve(np.asarray(pair['input']).copy())
                assert np.array_equal(prediction, pair['output'])
                f = inspect_pair(pair)
                records.append(f)
                (train_jumps if split == 'train' else test_jumps).append(max(f['jumps']))
                mapping, flat = {}, []
                for value in np.asarray(pair['input']).flat:
                    value = int(value)
                    flat.append(mapping.setdefault(value, len(mapping)))
                normalized.add((f['shape'], bytes(flat)))
                # Removing either ray propagation or cable-mediated transport
                # must fail on real demonstrations, including the minimum set.
                if split == 'train':
                    centers_only = np.asarray(pair['input']).copy()
                    for m in info['mirrors']:
                        centers_only[m['center']] = m['label']
                    assert not np.array_equal(centers_only, prediction)
                    try:
                        local = _optics(info, 'local')[0]
                    except AssertionError:
                        local = None
                    assert local is None or not np.array_equal(local, prediction)
        assert max(test_jumps) >= 3 and max(train_jumps) <= 2
    assert len(puzzle_keys) == 42 and len(normalized) >= 180
    assert len({r['shape'] for r in records}) >= 50
    assert {r['background'] for r in records} == set(range(10))
    assert {r['cables'] for r in records} >= {2,3,4}
    assert {r['sources'] for r in records} >= {1,2}
    assert {d for r in records for d in r['directions']} == set(DIAGONAL)
    assert {n for r in records for n in r['normals']} == set(CARDINAL)
    assert {a for r in records for a in r['face_relations']} >= {-1,0,1}
    assert {r['masks'] for r in records} >= {0,1,2}
    assert any(len(set(r['labels'])) < len(r['labels']) for r in records)
    assert any(2 in r['clues'] for r in records)
    assert _equal(generate_puzzle(0), generate_puzzle(0))
    # The host exhausts all legal counts; keep this local test bounded.
    for a,b in ((2,1),(10,1),(10,4)):
        p = generate_puzzle(42, a, b)
        assert len(p['train']) == a and len(p['test']) == b
        validate_generated_puzzle(p)
    for a,b in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4.5,1),(4,None)):
        try:
            generate_puzzle(0,a,b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    _direct_regression()


def _direct_regression():
    # A fully specified scene with a hand-enumerated answer. The active cable
    # joins right-backed frames; its two background contacts are (7,7),(13,7).
    fixture = np.zeros((20,22), dtype=np.int8)
    for center, clue in (((7,9),0),((13,9),2),((3,14),3),((5,18),0)):
        for p in _box(center)-{center}:
            fixture[p] = 8
        fixture[center] = clue
    wire = {(7,c) for c in range(11,16)} | {(r,15) for r in range(7,14)}
    wire |= {(13,c) for c in range(11,16)}
    wire |= {(0,c) for c in range(14,19)} | {(r,18) for r in range(0,4)} | {(1,14)}
    for p in wire:
        fixture[p] = 5
    for p in _l_cells((2,2),(1,1)):
        fixture[p] = 1
    expected = fixture.copy()
    expected[7,9], expected[5,18] = 2,3
    for p in ((3,3),(4,4),(5,5),(6,6)):
        expected[p] = 1
    for p in ((7,7),(13,7),(14,6),(15,5),(16,4),(17,3),(18,2),(19,1)):
        expected[p] = 2
    assert np.array_equal(solve(fixture), expected)
    # Unit geometry is checked against a hand-calculated local basis table.
    for normal in CARDINAL:
        for sign in (-1,1):
            tangent = _tangent(normal)
            incoming = _add(normal, tangent, sign)
            out = _add(_neg(normal), tangent, sign)
            assert _dot(incoming,normal) == 1 and _dot(out,normal) == -1
            assert _dot(incoming,tangent) == _dot(out,tangent) == sign
    puzzle = generate_puzzle(314159,2,1)
    bad = {s:[{k:np.asarray(v).copy() for k,v in p.items()} for p in puzzle[s]] for s in puzzle}
    info = _parse(bad['test'][0]['input'])
    m = info['mirrors'][0]
    p = next(iter(_box(m['center']) - {m['center']}))
    bad['test'][0]['output'][p] = info['background']
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError('modified frame accepted')


if __name__ == '__main__':
    self_test()
    print('self_test passed')
