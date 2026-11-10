"""Woven reservoirs: route complementary paint phases into pinned cavities.

Only NumPy and the standard library are used. Construction owns explicit paths
and cavity sets; the solver instead parses motifs, holes and a directional graph.
"""
import heapq
import numpy as np


D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))
D8 = D4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))
DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND


def _neighbors(cells, steps=D4):
    return {(r + dr, c + dc) for r, c in cells for dr, dc in steps} - set(cells)


def _components(cells, steps=D4):
    unseen = set(cells)
    result = []
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        group, todo = {root}, [root]
        for r, c in todo:
            for dr, dc in steps:
                p = (r + dr, c + dc)
                if p in unseen:
                    unseen.remove(p)
                    group.add(p)
                    todo.append(p)
        result.append(group)
    return result


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((a >= 0) & (a <= 9))
    return a.astype(np.int64, copy=True)


def _analyze(value):
    """Decode the observable grammar, without construction data or answers."""
    a = _grid(value)
    h, w = a.shape
    bg = int(a[0, 0])
    assert np.all(a[0] == bg) and np.all(a[-1] == bg)
    assert np.all(a[:, 0] == bg) and np.all(a[:, -1] == bg)
    canvas = {(r, c) for r in range(h) for c in range(w)}
    chambers = []
    for hole in _components((int(r), int(c)) for r, c in np.argwhere(a == bg)):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in hole):
            continue
        rim = _neighbors(hole) & canvas
        colors = {int(a[p]) for p in rim}
        if len(colors) != 1:
            continue
        parts = _components(rim, D8)
        pins = [s for s in parts if len(s) == 1]
        shells = [s for s in parts if len(s) >= 12]
        if len(parts) != 2 or len(pins) != 1 or len(shells) != 1:
            continue
        pin = next(iter(pins[0]))
        inside = hole | {pin}
        shell = shells[0]
        assert len(hole) >= 8 and _neighbors(inside) == shell
        assert all(p in inside for p in _neighbors({pin}, D8))
        chambers.append(dict(cells=hole, inside=inside, shell=shell,
                             pin=pin, color=next(iter(colors))))
    assert chambers, "no pinned chambers"
    assert len({q['color'] for q in chambers}) == 1
    frame = chambers[0]['color']
    footprints = set().union(*(q['inside'] | q['shell'] for q in chambers))
    assert sum(len(q['inside'] | q['shell']) for q in chambers) == len(footprints)
    for q in chambers:
        own = q['inside'] | q['shell']
        assert not (_neighbors(own, D8) & (footprints - own))
    # Find the two source silhouettes. Their 3x3 footprints are otherwise blank.
    sources = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            cells = {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}
            if cells & footprints:
                continue
            block = a[r - 1:r + 2, c - 1:c + 2]
            for phase, pattern in ((1, DIAMOND), (0, CHECKER)):
                label = int(block[0, 1] if phase else block[1, 1])
                if label in (bg, frame):
                    continue
                if not np.array_equal(block, np.where(pattern, label, bg)):
                    continue
                ports = []
                for d, (dr, dc) in enumerate(D4):
                    p = (r + 2 * dr, c + 2 * dc)
                    if p in canvas and a[p] != bg:
                        ports.append((p, d, int(a[p])))
                if len(ports) == 1 and ports[0][2] not in (frame, label):
                    sources.append(dict(center=(r, c), phase=phase, label=label,
                                        footprint=cells, port=ports[0][0],
                                        direction=ports[0][1], carrier=ports[0][2]))
    assert sources and len({s['carrier'] for s in sources}) == 1
    carrier = sources[0]['carrier']
    assert carrier not in (bg, frame)
    source_cells = set().union(*(s['footprint'] for s in sources))
    assert sum(len(s['footprint']) for s in sources) == len(source_cells)
    excluded = footprints | source_cells
    residual = {int(a[p]) for p in canvas - excluded} - {bg, carrier}
    assert len(residual) <= 1
    bridge = next(iter(residual)) if residual else None
    assert bridge not in (bg, frame, carrier)
    assert {s['label'] for s in sources}.isdisjoint({carrier, bridge})
    visible = {p for p in canvas - excluded if int(a[p]) in (carrier, bridge)}
    crossings = {p for p in visible if a[p] == bridge}
    roots = {s['port']: i for i, s in enumerate(sources)}
    assert len(roots) == len(sources) and set(roots) <= visible
    assert all((_neighbors(s['footprint']) & visible) == {s['port']} for s in sources)

    def node(p, axis):
        return (p[0], p[1], axis if p in crossings else -1)

    graph = {}
    endpoints = {}
    forks = []
    for p in sorted(visible):
        adjacent = [d for d, (dr, dc) in enumerate(D4)
                    if (p[0] + dr, p[1] + dc) in visible]
        if p in crossings:
            assert len(adjacent) == 4
            assert all(a[p[0] + dr, p[1] + dc] == carrier for dr, dc in D4)
        else:
            assert len(adjacent) in (1, 2, 3), "unmarked crossing or broken cable"
            if len(adjacent) == 3:
                forks.append(p)
        for d in adjacent:
            dr, dc = D4[d]
            q = (p[0] + dr, p[1] + dc)
            graph.setdefault(node(p, d % 2), set()).add(node(q, d % 2))
        touching = [i for i, chamber in enumerate(chambers)
                    if _neighbors({p}) & chamber['shell']]
        if p in roots:
            assert len(adjacent) == 1 and not touching
            s = sources[roots[p]]
            dr, dc = D4[s['direction']]
            assert (p[0] + dr, p[1] + dc) in visible
        elif len(adjacent) == 1:
            assert len(touching) == 1, "free cable end"
            i = touching[0]
            contact = _neighbors({p}) & chambers[i]['shell']
            assert len(contact) == 1
            q = next(iter(contact))
            assert (2 * q[0] - p[0], 2 * q[1] - p[1]) in chambers[i]['inside']
            endpoints[p] = i
        else:
            assert not touching, "side contact is not an inlet"
    supplies = [dict() for _ in chambers]
    unseen = set(graph)
    routes = []
    ownership = {}
    while unseen:
        root = min(unseen)
        unseen.remove(root)
        group, todo = {root}, [root]
        for v in todo:
            for u in graph[v] & unseen:
                unseen.remove(u)
                group.add(u)
                todo.append(u)
        source_ids = {roots[v[:2]] for v in group if v[:2] in roots}
        assert len(source_ids) == 1, "each cable tree has exactly one source"
        source_id = next(iter(source_ids))
        assert sum(len(graph[v]) for v in group) == 2 * (len(group) - 1), "cable cycle"
        target_ids = [endpoints[v[:2]] for v in group if v[:2] in endpoints]
        assert target_ids and len(set(target_ids)) == len(target_ids)
        source = sources[source_id]
        for i in target_ids:
            assert source['phase'] not in supplies[i], "two sources compete for one phase"
            supplies[i][source['phase']] = source['label']
        for v in group:
            ownership[v] = source_id
        routes.append(dict(source=source_id, nodes=group, targets=target_ids))
    assert len(routes) == len(sources)
    out = a.copy()
    for chamber, paint in zip(chambers, supplies):
        pr, pc = chamber['pin']
        for r, c in chamber['cells']:
            phase = (r - pr + c - pc) % 2
            if phase in paint:
                out[r, c] = paint[phase]
    info = dict(bg=bg, frame=frame, carrier=carrier, bridge=bridge,
                chambers=chambers, sources=sources, supplies=supplies,
                crossings=crossings, forks=forks, routes=routes,
                ownership=ownership, endpoints=endpoints)
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Fill only the routed stencil phases of each pinned chamber."""
    return _analyze(input_grid)[0]


def _nearest_mismatch(info):
    """Is a supplied chamber strictly closer to a different source?"""
    for route in info['routes']:
        source_id = route['source']
        for i in route['targets']:
            pin = info['chambers'][i]['pin']
            distances = [sum(abs(x - y) for x, y in zip(pin, s['center']))
                         for s in info['sources']]
            if min(distances) < distances[source_id]:
                return True
    return False


def _ablation_evidence(a, info):
    """Concrete counterexamples to plausible simpler filling rules."""
    expected = solve(a)
    uniform, global_fill, box_phase = a.copy(), a.copy(), a.copy()
    first = {}
    for source in info['sources']:
        first.setdefault(source['phase'], source['label'])
    for chamber, paint in zip(info['chambers'], info['supplies']):
        pr, pc = chamber['pin']
        mr = min(r for r, c in chamber['inside'])
        mc = min(c for r, c in chamber['inside'])
        for r, c in chamber['cells']:
            if paint:
                uniform[r, c] = paint[min(paint)]
            phase = (r + c - pr - pc) % 2
            if phase in first:
                global_fill[r, c] = first[phase]
            wrong_phase = (r + c - mr - mc) % 2
            if wrong_phase in paint:
                box_phase[r, c] = paint[wrong_phase]
    # Treating bridges as junctions transmits both source signals to all
    # reservoirs in a merged foreground component. The diagnostic crossing
    # lesson has one source of each phase, so this alternative is unambiguous.
    junction = a.copy()
    wire = {v[:2] for route in info['routes'] for v in route['nodes']}
    for group in _components(wire):
        paint = {}
        source_ids = [i for i, s in enumerate(info['sources']) if s['port'] in group]
        conflict = False
        for i in source_ids:
            s = info['sources'][i]
            if s['phase'] in paint and paint[s['phase']] != s['label']:
                conflict = True
            paint[s['phase']] = s['label']
        if conflict:
            continue
        targets = {info['endpoints'][p] for p in group if p in info['endpoints']}
        for i in targets:
            chamber = info['chambers'][i]
            for r, c in chamber['cells']:
                phase = (r + c - sum(chamber['pin'])) % 2
                if phase in paint:
                    junction[r, c] = paint[phase]
    return dict(uniform=not np.array_equal(uniform, expected),
                global_fill=not np.array_equal(global_fill, expected),
                box_phase=not np.array_equal(box_phase, expected),
                junction=not np.array_equal(junction, expected),
                nearest=_nearest_mismatch(info))


def _ri(rng, low, high):
    return int(rng.integers(low, high + 1))


def _shape(rng, large=False, simple=False):
    h = _ri(rng, 3, 6 if large else 4)
    w = _ri(rng, 6, 8) if large else _ri(rng, 3, 4)
    kind = 'box' if simple else str(rng.choice(['box', 'box', 'step', 'slant', 'bite']))
    cells = {(r, c) for r in range(h) for c in range(w)}
    if kind == 'step':
        cells = {(r, c + (r >= h // 2)) for r, c in cells}
    elif kind == 'slant':
        cells = {(r, c + r // 2) for r, c in cells}
    elif kind == 'bite' and w >= 4:
        cells -= {(r, w - 1) for r in range(max(1, h // 2))}
    if rng.random() < .5:
        cells = {(c, r) for r, c in cells}
    shell = _neighbors(cells)
    mr, mc = min(r for r, c in shell), min(c for r, c in shell)
    cells = {(r - mr, c - mc) for r, c in cells}
    shell = {(r - mr, c - mc) for r, c in shell}
    pins = [p for p in sorted(cells) if _neighbors({p}, D8) <= cells
            and len(_components(cells - {p})) == 1]
    if not pins:
        return _shape(rng, large, True)
    pin = pins[_ri(rng, 0, len(pins) - 1)]
    return cells, shell, pin


def _routing(rng, own, all_wire, crosses, blocked, goals, owner, source_roots, target_ends, shape):
    """Dijkstra in empty space, with straight-through jumps over old strands."""
    h, w = shape
    canvas = {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)}
    occupied = set(all_wire)
    free = canvas - blocked - occupied - _neighbors(occupied)
    goals = set(goals) & (canvas - blocked - occupied - _neighbors(occupied))
    if not goals:
        return None
    free |= goals
    fronts = {}
    for p in sorted(own - crosses - source_roots - target_ends):
        if len(_neighbors({p}) & occupied) >= 3:
            continue
        for q in sorted(_neighbors({p}) & canvas - blocked - occupied):
            if (_neighbors({q}) & occupied) == {p}:
                fronts[q] = p
    if not fronts:
        return None
    # Random positive terrain varies bend positions without permitting loops.
    weights = rng.integers(1, 5, size=shape)
    distances, previous, queue = {}, {}, []
    for q, anchor in sorted(fronts.items()):
        distances[q] = int(weights[q])
        previous[q] = (None, [q], anchor)
        heapq.heappush(queue, (distances[q], q))
    jumps = {}
    for p in sorted(occupied - crosses):
        if all_wire[p] == owner:
            continue
        neighbors = _neighbors({p}) & occupied
        if len(neighbors) != 2:
            continue
        for axis in range(2):
            dr, dc = D4[axis]
            along = {(p[0] + dr, p[1] + dc), (p[0] - dr, p[1] - dc)}
            # New strand uses this axis; the old one must be perpendicular.
            if neighbors != {(p[0] - dc, p[1] + dr), (p[0] + dc, p[1] - dr)}:
                continue
            u = (p[0] - 2 * dr, p[1] - 2 * dc)
            v = (p[0] + 2 * dr, p[1] + 2 * dc)
            if u not in free or v not in free or along & (blocked | occupied):
                continue
            if not along <= canvas or any((_neighbors({q}) & occupied) != {p} for q in along):
                continue
            if _neighbors({p}) & crosses:
                continue
            middle = [(p[0] - dr, p[1] - dc), p, (p[0] + dr, p[1] + dc)]
            jumps.setdefault(u, []).append((v, middle + [v]))
            jumps.setdefault(v, []).append((u, list(reversed(middle)) + [u]))
    end = None
    while queue:
        cost, p = heapq.heappop(queue)
        if cost != distances[p]:
            continue
        if p in goals:
            end = p
            break
        options = [(q, [q]) for q in sorted(_neighbors({p}) & free)]
        options += jumps.get(p, [])
        for q, segment in options:
            new_cost = cost + sum(int(weights[v]) for v in segment)
            if new_cost < distances.get(q, 10**9):
                distances[q] = new_cost
                previous[q] = (p, segment, None)
                heapq.heappush(queue, (new_cost, q))
    if end is None:
        return None
    tail, p = [], end
    while True:
        parent, segment, anchor = previous[p]
        tail.append(segment)
        if parent is None:
            break
        p = parent
    path = [anchor] + [q for segment in reversed(tail) for q in segment]
    if len(set(path)) != len(path):
        return None
    # Jump interiors can create a chord not represented in the search graph.
    # A path must be a simple chain before its one tree attachment is added.
    path_set = set(path)
    if any(len(_neighbors({q}) & path_set) != (1 if k in (0, len(path) - 1) else 2)
           for k, q in enumerate(path)):
        return None
    new_crosses = path_set & occupied - {anchor}
    return path, new_crosses, end


def _scene(rng, plan, nchamber, large=False, require_cross=False, compact=False, lesson=None):
    """Place objects freely, then route the explicitly assigned paint trees."""
    nsource = len(plan)
    h = _ri(rng, 16, 22) if compact else _ri(rng, 25, 30)
    w = _ri(rng, 16, 23) if compact else _ri(rng, 25, 30)
    shape = (h, w)
    canvas = {(r, c) for r in range(1, h - 1) for c in range(1, w - 1)}
    occupied_objects = set()
    regions = []
    for i in range(nchamber):
        cells, shell, pin = _shape(rng, large=large, simple=compact)
        if lesson == 0 and i == 0:
            # The literal 3x3 source tile is the smallest target lesson.
            cells = {(r, c) for r in range(1, 4) for c in range(1, 4)}
            shell, pin = _neighbors(cells), (2, 2)
        if lesson == 1 and i == 0:
            # Odd pin parity relative to the box disproves resetting the
            # pattern at its top-left corner (the 3x3 lesson has even parity).
            cells = {(r, c) for r in range(1, 5) for c in range(1, 5)}
            shell = _neighbors(cells)
            pin = (2, 3) if rng.random() < .5 else (3, 2)
        sh, sw = max(r for r, c in shell) + 1, max(c for r, c in shell) + 1
        placed = False
        for _ in range(80):
            rr, cc = _ri(rng, 1, h - sh - 1), _ri(rng, 1, w - sw - 1)
            inner = {(r + rr, c + cc) for r, c in cells}
            edge = {(r + rr, c + cc) for r, c in shell}
            both = inner | edge
            if (both | _neighbors(both, D8)) & occupied_objects:
                continue
            regions.append(dict(inside=inner, shell=edge, pin=(pin[0] + rr, pin[1] + cc)))
            occupied_objects |= both
            placed = True
            break
        if not placed:
            return None
    if lesson == 1 and sum(regions[0]['pin']) % 2 == sum(regions[1]['pin']) % 2:
        return None
    sources = []
    protected = set(occupied_objects)
    for i in range(nsource):
        placed = False
        for _ in range(100):
            r, c = _ri(rng, 3, h - 4), _ri(rng, 3, w - 4)
            d = _ri(rng, 0, 3)
            dr, dc = D4[d]
            footprint = {(r + y, c + x) for y in (-1, 0, 1) for x in (-1, 0, 1)}
            stem = {(r + k * dr, c + k * dc) for k in (2, 3)}
            all_cells = footprint | stem
            if not all_cells <= canvas:
                continue
            if (all_cells | _neighbors(all_cells, D8)) & protected:
                continue
            sources.append(dict(center=(r, c), footprint=footprint, stem=stem,
                                port=(r + 2 * dr, c + 2 * dc), phase=plan[i][1]))
            protected |= all_cells
            placed = True
            break
        if not placed:
            return None
    # Canonical construction roles: 0 background, 1 shell/pin, 2 wire, 3 bridge.
    a = np.zeros(shape, dtype=np.int64)
    for chamber in regions:
        for p in chamber['shell'] | {chamber['pin']}:
            a[p] = 1
    for i, source in enumerate(sources):
        r, c = source['center']
        pattern = DIAMOND if source['phase'] else CHECKER
        a[r - 1:r + 2, c - 1:c + 2] = np.where(pattern, 4 + i, 0)
    all_wire = {p: i for i, s in enumerate(sources) for p in s['stem']}
    crosses, target_ends = set(), set()
    roots = {s['port'] for s in sources}
    blocked_objects = occupied_objects | set().union(*(s['footprint'] for s in sources))
    blocked = (blocked_objects | _neighbors(blocked_objects)) & canvas
    for source_id in map(int, rng.permutation(nsource)):
        targets, phase = plan[source_id]
        for target_id in [targets[int(i)] for i in rng.permutation(len(targets))]:
            chamber = regions[target_id]
            # Choose among all true outward-facing boundary ports.
            goals = set()
            for p in chamber['shell']:
                for dr, dc in D4:
                    q = (p[0] + dr, p[1] + dc)
                    behind = (p[0] - dr, p[1] - dc)
                    if (behind in chamber['inside'] and q in canvas - blocked_objects
                            and (_neighbors({q}) & occupied_objects) == {p}):
                        goals.add(q)
            # Goals are the sole exception to chamber clearance, never to other
            # source footprints or to the one-cell border.
            own = {p for p, owner in all_wire.items() if owner == source_id}
            result = _routing(rng, own, all_wire, crosses, blocked - goals,
                              goals, source_id, roots, target_ends, shape)
            if result is None:
                return None
            path, new_crosses, end = result
            for p in path:
                if p not in all_wire:
                    all_wire[p] = source_id
            crosses |= new_crosses
            target_ends.add(end)
    if require_cross and not crosses:
        return None
    for p in all_wire:
        a[p] = 3 if p in crosses else 2
    b = a.copy()
    for source_id, (targets, phase) in enumerate(plan):
        for target in targets:
            pr, pc = regions[target]['pin']
            for r, c in regions[target]['inside'] - {regions[target]['pin']}:
                if (r - pr + c - pc) % 2 == phase:
                    b[r, c] = 4 + source_id
    # Relabel every role, not only paint. This is decoded through local grammar.
    palette = np.array(rng.permutation(10), dtype=np.int64)
    # Repeated source colors are legal except in the two diagnostic lessons.
    if not require_cross and nsource > 2 and rng.random() < .3:
        palette[5] = palette[4]
    a, b = palette[a], palette[b]
    rotation = _ri(rng, 0, 3)
    a, b = np.rot90(a, rotation).copy(), np.rot90(b, rotation).copy()
    if rng.random() < .5:
        a, b = np.fliplr(a).copy(), np.fliplr(b).copy()
    try:
        predicted, info = _analyze(a)
        if not np.array_equal(predicted, b):
            return None
        if len(info['sources']) != nsource or len(info['chambers']) != nchamber:
            return None
        if lesson == 0 and not _nearest_mismatch(info):
            return None
        if lesson == 1 and not _ablation_evidence(a, info)['box_phase']:
            return None
    except AssertionError:
        return None
    return dict(input=a, output=b)


def _plan(rng, lesson, split, shift):
    if lesson == 0:
        return [([0], 1), ([1], 0)], 2
    if lesson == 1:
        return [([0, 1], 1), ([0], 0)], 3
    if split == 'test' and shift == 'count':
        n = 4
    else:
        n = _ri(rng, 1 if split == 'train' else 2, 3)
    suppliers = [dict() for _ in range(n)]
    # A source tree may fan out to several chambers; assigning at most one
    # supplier to a given phase keeps every requested pixel deterministic.
    ns = _ri(rng, 2, 4) if n > 1 else _ri(rng, 1, 2)
    offset = _ri(rng, 0, 1)
    phases = [(i + offset) % 2 for i in range(ns)]
    rng.shuffle(phases)
    targets = [[] for _ in range(ns)]
    for s in range(ns):
        options = [i for i in range(n) if phases[s] not in suppliers[i]]
        if not options:
            continue
        i = options[_ri(rng, 0, len(options) - 1)]
        suppliers[i][phases[s]] = s
        targets[s].append(i)
    for s in range(ns):
        for i in range(n):
            if phases[s] not in suppliers[i] and rng.random() < .32:
                suppliers[i][phases[s]] = s
                targets[s].append(i)
    if shift == 'fanin':
        if split == 'train':
            dual = [i for i in range(n) if len(suppliers[i]) == 2]
            for i in dual[1:]:
                s = suppliers[i][0]
                targets[s].remove(i)
                del suppliers[i][0]
        else:
            # Two dual reservoirs recombine phases, already shown singly in
            # lesson 1. With two sources this is a pair of branching trees.
            for i in range(2):
                for phase in (0, 1):
                    if phase not in suppliers[i]:
                        s = phases.index(phase)
                        suppliers[i][phase] = s
                        targets[s].append(i)
    plan = [(sorted(t), p) for t, p in zip(targets, phases) if t]
    return plan, n


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, v, lo, hi in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or not lo <= v <= hi:
            raise ValueError(f'{name} must be an integer in {lo}..{hi}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(['count', 'shape', 'fanin']))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            lesson = index if split == 'train' and index < 2 else None
            plan, nchamber = _plan(rng, lesson, split, shift)
            for attempt in range(250):
                pair = _scene(rng, plan, nchamber, large=split == 'test' and shift == 'shape',
                              require_cross=lesson == 0,
                              compact=nchamber == 1 and attempt < 100, lesson=lesson)
                if pair is None:
                    continue
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError(f'placement failed for seed {seed}, {split} {index}, {plan}')
            seen.add(key)
            puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(pair):
    _, f = _analyze(pair['input'])
    normalized = []
    offsets = []
    for q in f['chambers']:
        mr = min(r for r, c in q['inside'])
        mc = min(c for r, c in q['inside'])
        cells = tuple(sorted((r - mr, c - mc) for r, c in q['inside']))
        normalized.append((max(r for r, c in cells) + 1, max(c for r, c in cells) + 1, cells))
        offsets.append((q['pin'][0] - mr, q['pin'][1] - mc))
    return dict(shape=tuple(np.asarray(pair['input']).shape), background=f['bg'],
                roles=(f['bg'], f['frame'], f['carrier'], f['bridge']),
                chambers=len(f['chambers']), sources=len(f['sources']),
                crossings=len(f['crossings']), forks=len(f['forks']),
                dual=sum(len(p) == 2 for p in f['supplies']),
                single=sum(len(p) == 1 for p in f['supplies']),
                empty=sum(not p for p in f['supplies']),
                fanout=max(len(t['targets']) for t in f['routes']),
                colors=len(np.unique(pair['input'])),
                repeated_labels=len({s['label'] for s in f['sources']}) < len(f['sources']),
                nonrectangular=sum(len(s[2]) < s[0] * s[1] for s in normalized),
                pin_offsets=tuple(offsets),
                route_lengths=tuple(len(t['nodes']) for t in f['routes']),
                areas=tuple(sorted(len(c['inside']) for c in f['chambers'])),
                phases=tuple(sorted(s['phase'] for s in f['sources'])),
                cavity_shapes=tuple(normalized))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training, evidence = set(), [], set()
    split_features = {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            a, b = _grid(pair['input']), _grid(pair['output'])
            assert a.shape == b.shape
            key = (a.shape, a.tobytes())
            assert key not in seen
            seen.add(key)
            predicted, f = _analyze(a)
            split_features[split].append(dict(chambers=len(f['chambers']),
                                              dual=sum(len(p) == 2 for p in f['supplies']),
                                              areas=[len(q['inside']) for q in f['chambers']]))
            assert np.array_equal(predicted, b), 'input-derived fill mismatch'
            changed = {tuple(p) for p in np.argwhere(a != b)}
            assert changed and changed <= set().union(*(q['cells'] for q in f['chambers']))
            assert all(a[p] == f['bg'] for p in changed)
            for q, supplies in zip(f['chambers'], f['supplies']):
                assert b[q['pin']] == a[q['pin']]
                for phase in (0, 1):
                    cells = {p for p in q['cells'] if (p[0] + p[1] - sum(q['pin'])) % 2 == phase}
                    assert cells
                    expected = supplies.get(phase, f['bg'])
                    assert all(b[p] == expected for p in cells)
            if split == 'train':
                training.append(f)
                evidence.update(k for k, v in _ablation_evidence(a, f).items() if v)
    # These lessons remain present even at the minimum legal count 2/1.
    assert any(f['crossings'] and len({s['label'] for s in f['sources']}) >= 2 for f in training)
    assert any(any(len(p) == 2 and len(set(p.values())) == 2 for p in f['supplies']) for f in training)
    assert any(any(len(p) == 1 for p in f['supplies']) for f in training)
    assert any(any(not p for p in f['supplies']) for f in training)
    assert any(any(len(t['targets']) >= 2 for t in f['routes']) for f in training)
    assert {s['phase'] for f in training for s in f['sources']} == {0, 1}
    assert {'uniform', 'global_fill', 'box_phase', 'junction', 'nearest'} <= evidence
    tr, te = split_features['train'], split_features['test']
    assert ((max(f['chambers'] for f in tr) < min(f['chambers'] for f in te))
            or (max(v for f in tr for v in f['areas']) < min(v for f in te for v in f['areas']))
            or (max(f['dual'] for f in tr) < min(f['dual'] for f in te))), 'missing structural shift'


def self_test():
    """Bounded semantic, count, determinism, and 42-seed diversity checks."""
    measurements, signatures, normalized = [], set(), set()
    puzzles = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        validate_generated_puzzle(puzzle)
        for pair in puzzle['train'] + puzzle['test']:
            measurements.append(_features(pair))
            a = pair['input']
            signatures.add((a.shape, a.tobytes()))
            lut = {}
            normalized.add((a.shape, tuple(lut.setdefault(int(c), len(lut)) for c in a.flat)))
    assert len(signatures) == 210 and len(normalized) >= 200
    assert len({m['shape'] for m in measurements}) >= 25
    assert {m['chambers'] for m in measurements} >= {1, 2, 3, 4}
    assert {m['sources'] for m in measurements} >= {1, 2, 3, 4}
    assert {m['background'] for m in measurements} == set(range(10))
    assert len({s for m in measurements for s in m['cavity_shapes']}) >= 20
    assert max(m['fanout'] for m in measurements) >= 3
    assert max(m['dual'] for m in measurements) >= 2
    assert max(m['crossings'] for m in measurements) >= 2
    assert len({m['roles'] for m in measurements}) >= 100
    assert any(m['repeated_labels'] for m in measurements)
    assert sum(m['nonrectangular'] > 0 for m in measurements) >= 30
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(4200 + 10 * nt + nq, nt, nq)
            assert len(p['train']) == nt and len(p['test']) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    repeated = generate_puzzle(0)
    assert all(np.array_equal(p[k], q[k]) for s in ('train', 'test')
               for p, q in zip(puzzles[0][s], repeated[s]) for k in ('input', 'output'))
    # The rule commutes with rotation/reflection and arbitrary palette bijection.
    rng = np.random.default_rng(12345)
    for pair in puzzles[0]['train'] + puzzles[0]['test']:
        a, b = pair['input'], pair['output']
        before = a.copy()
        assert np.array_equal(solve(a), b) and np.array_equal(a, before)
        for k in (1, 2, 3):
            assert np.array_equal(solve(np.rot90(a, k)), np.rot90(b, k))
        assert np.array_equal(solve(a[:, ::-1]), b[:, ::-1])
        palette = rng.permutation(10)
        assert np.array_equal(solve(palette[a]), palette[b])
    # Exchanging the two source silhouettes must exchange which lattice each
    # source supplies, even though its route and color stay fixed.
    pair = puzzles[0]['train'][0]
    a = pair['input'].copy()
    _, f = _analyze(a)
    swapped, expected = a.copy(), a.copy()
    for s in f['sources']:
        r, c = s['center']
        pattern = CHECKER if s['phase'] else DIAMOND
        block = np.where(pattern, s['label'], f['bg'])
        swapped[r - 1:r + 2, c - 1:c + 2] = block
        expected[r - 1:r + 2, c - 1:c + 2] = block
    for q, paint in zip(f['chambers'], f['supplies']):
        for r, c in q['cells']:
            phase = (r + c - sum(q['pin'])) % 2
            if 1 - phase in paint:
                expected[r, c] = paint[1 - phase]
    assert np.array_equal(solve(swapped), expected)
    assert not np.array_equal(solve(swapped), solve(a))
    damaged = {s: [{k: np.array(v, copy=True) for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    a, b = damaged['test'][0]['input'], damaged['test'][0]['output']
    p = tuple(np.argwhere(a != b)[0])
    b[p] = a[p]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError('partial phase fill accepted')
    return dict(seeds=42, pairs=len(measurements), distinct_structures=len(normalized),
                shapes=len({m['shape'] for m in measurements}),
                cavity_shapes=len({s for m in measurements for s in m['cavity_shapes']}))


if __name__ == '__main__':
    print(self_test())
