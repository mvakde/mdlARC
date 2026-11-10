"""One pulse, a visible color function, and clocked forks in paired rails.

Only NumPy and the Python standard library are used. Construction metadata is
discarded: solve and validation read geometry and relations from the input.
"""
from __future__ import annotations

import json
from collections import deque
import numpy as np


_D4 = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _add(p, d, n=1):
    return p[0] + n * d[0], p[1] + n * d[1]


def _inside(g, p):
    return 0 <= p[0] < g.shape[0] and 0 <= p[1] < g.shape[1]


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= s <= 30 for s in g.shape), "grid bounds")
    _require(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _require(np.all((0 <= g) & (g <= 9)), "ARC colors required")
    return g.astype(int, copy=False)


def _key(g):
    g = np.asarray(g, dtype=np.int8)
    return g.shape, g.tobytes()


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    parts = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        cells, todo = {first}, [first]
        while todo:
            p = todo.pop()
            for d in _D4:
                q = _add(p, d)
                if q in remaining:
                    remaining.remove(q)
                    cells.add(q)
                    todo.append(q)
        parts.append(cells)
    return parts


def _read(g):
    """Recover background, square relations, launch cross and clock dots."""
    colors, counts = np.unique(g, return_counts=True)
    _require(np.count_nonzero(counts == counts.max()) == 1, "unique background")
    bg = int(colors[np.argmax(counts)])
    parts = _components(g != bg)
    cards, mapping = [], {}
    for cells in parts:
        if len(cells) != 9:
            continue
        rs, cs = zip(*cells)
        if max(rs) - min(rs) != 2 or max(cs) - min(cs) != 2:
            continue
        center = min(rs) + 1, min(cs) + 1
        bodies = {int(g[p]) for p in cells if p != center}
        if len(bodies) != 1:
            continue
        body, value = bodies.pop(), int(g[center])
        _require(body not in mapping or mapping[body] == value, "conflicting relation tiles")
        mapping[body] = value
        cards.append(dict(cells=cells, center=center, body=body, value=value))
    _require(3 <= len(mapping) <= 6 and 3 <= len(cards) <= 8, "relation table size")
    _require(set(mapping.values()) <= set(mapping), "lookup table must be total on identities")
    sources = []
    identities = set(mapping)
    for cells in parts:
        if len(cells) != 5:
            continue
        for center in sorted(cells):
            arms = {_add(center, d) for d in _D4}
            if arms | {center} != cells:
                continue
            bodies = {int(g[p]) for p in arms}
            if len(bodies) == 1:
                marker = bodies.pop()
                if marker not in identities and int(g[center]) in identities:
                    sources.append((center, arms, marker))
    _require(len(sources) == 1, "one isolated launch cross required")
    source, arms, marker = sources[0]
    hubs = set(map(tuple, np.argwhere(g == marker).tolist())) - arms
    _require(hubs and source not in hubs, "clock dots required")
    singletons = {next(iter(cells)) for cells in parts if len(cells) == 1}
    _require(hubs <= singletons, "junction markers must be isolated single pixels")
    _require(set(map(int, colors)) <= identities | {marker, bg}, "unknown color role")
    card_cells = set().union(*(c['cells'] for c in cards))
    return dict(background=bg, marker=marker, source=source, arms=arms,
                hubs=hubs, cards=cards, card_cells=card_cells, mapping=mapping)


def _flanks(g, p, d):
    normal = -d[1], d[0]
    a, b = _add(p, normal), _add(p, normal, -1)
    if not (_inside(g, a) and _inside(g, b)):
        return None
    return int(g[a]), int(g[b])


def _ports(g, roles, center, launch=False):
    """A clock's ports are two cells away; a cross's port is three away.

    The one-cell clearance at clocks and two-cell clearance at launch crosses
    is part of the visible glyph convention, not a sampled hidden parameter.
    """
    distance = 3 if launch else 2
    result = []
    for d in _D4:
        corridor = [_add(center, d, i) for i in range(2 if launch else 1, distance + 1)]
        if not all(_inside(g, p) and g[p] == roles['background'] for p in corridor):
            continue
        pair = _flanks(g, corridor[-1], d)
        if pair is not None and all(c in roles['mapping'] for c in pair):
            result.append(dict(direction=d, colors=pair, cell=corridor[-1]))
    return result


def _trace(g, roles, *, lookup=True, turns=True, permissive=False, extra_bend_lookups=False):
    """Directional input-only reader; no flood filling of free space."""
    bg, source = roles['background'], roles['source']
    state = int(g[source])
    initial = [p for p in _ports(g, roles, source, True) if p['colors'] == (state, state)]
    _require(len(initial) == 1, "launch needs one matching port")
    direction = initial[0]['direction']
    p = _add(source, direction, 2)  # The pulse passes its cross's marker arm.
    trail, events, bends, seen = [(p, direction, state)], [], [], set()
    destination = None
    for _ in range(3600):
        key = p, direction, state
        _require(key not in seen, "pulse route cycles")
        seen.add(key)
        q = _add(p, direction)
        if not _inside(g, q):
            destination = p
            break
        if g[q] == bg:
            p = q
            trail.append((p, direction, state))
            continue
        if q in roles['hubs']:
            p = q
            before = state
            state = roles['mapping'][state] if lookup else state
            reverse = -direction[0], -direction[1]
            ports = [port for port in _ports(g, roles, p) if port['direction'] != reverse]
            choices = [port for port in ports if port['colors'] == (state, state)]
            events.append(dict(cell=p, before=before, after=state, ports=ports,
                               incoming=direction, choices=choices))
            if len(choices) != 1:
                if permissive:
                    break  # Counterfactual no-lookup model loses the pulse.
                raise AssertionError("clock needs exactly one matching outgoing pair")
            direction = choices[0]['direction']
            trail.append((p, direction, state))
            continue
        choices = []
        for d in ((-direction[1], direction[0]), (direction[1], -direction[0])):
            q = _add(p, d)
            if (_inside(g, q) and g[q] == bg
                    and _flanks(g, q, d) == (state, state)):
                choices.append(d)
        _require(len(choices) <= 1, "ambiguous ordinary bend")
        if choices and extra_bend_lookups:
            state = roles['mapping'][state]
            choices = [d for d in choices if _flanks(g, _add(p, d), d) == (state, state)]
        if not choices or not turns:
            destination = p
            break
        bends.append((p, direction, choices[0]))
        direction = choices[0]
    else:
        raise AssertionError("trace bound exhausted")
    if destination is not None:
        _require(g[destination] == bg, "terminal must be free")
    return dict(destination=destination, final=state, trail=trail, events=events, bends=bends)


def _answer(g, roles, trace):
    out = g.copy()
    out[roles['source']] = roles['background']
    if trace['destination'] is not None:
        out[trace['destination']] = trace['final']
    return out


def _lane(g, roles, start, direction, color, launch=False):
    """Read one lane up to a clock or free terminal, without a lookup."""
    p = _add(start, direction, 2) if launch else start
    path, bends, seen = [p], 0, set()
    for _ in range(1800):
        _require((p, direction) not in seen, "lane cycle")
        seen.add((p, direction))
        q = _add(p, direction)
        if not _inside(g, q):
            return dict(end=p, kind='terminal', arrival=direction, path=path, bends=bends)
        if q in roles['hubs']:
            return dict(end=q, kind='clock', arrival=direction, path=path+[q], bends=bends)
        if g[q] == roles['background']:
            p = q
            path.append(p)
            continue
        choices = [d for d in ((-direction[1], direction[0]), (direction[1], -direction[0]))
                   if _inside(g, _add(p, d)) and g[_add(p, d)] == roles['background']
                   and _flanks(g, _add(p, d), d) == (color, color)]
        _require(len(choices) <= 1, "lane branches without a clock")
        if not choices:
            _require(g[p] == roles['background'], "closed lane terminal must be free")
            return dict(end=p, kind='terminal', arrival=direction, path=path, bends=bends)
        direction = choices[0]
        bends += 1
    raise AssertionError("lane bound exhausted")


def _network(g, roles):
    """Audit every reachable matched lane, including dormant alternatives.

    This is an invariant reader within this generator, not an independent
    semantic solver. Deliberately mismatched gates are recorded as closed ports.
    """
    source = roles['source']
    state = int(g[source])
    ports = [p for p in _ports(g, roles, source, True) if p['colors'] == (state, state)]
    _require(len(ports) == 1, "unique launch lane")
    pending = deque([(source, ports[0], True)])
    nodes, occupied, terminals, lanes, children = {source}, {}, set(), [], {source: []}
    degrees, gated = [], 0
    while pending:
        start, port, launch = pending.popleft()
        color = port['colors'][0]
        lane = _lane(g, roles, start, port['direction'], color, launch)
        for p in lane['path']:
            if p in roles['hubs']:
                continue
            _require(p not in occupied, "matched lanes merge or cross")
            occupied[p] = start
        lanes.append(lane)
        end = lane['end']
        _require(end not in nodes and end not in terminals, "network cycle or conflicting terminals")
        children[start].append(end)
        children[end] = []
        if lane['kind'] == 'terminal':
            terminals.add(end)
            continue
        nodes.add(end)
        reverse = -lane['arrival'][0], -lane['arrival'][1]
        visible = _ports(g, roles, end)
        incoming = [p for p in visible if p['direction'] == reverse]
        _require(len(incoming) == 1 and incoming[0]['colors'] == (color, color), "clock's incoming pair")
        outgoing = [p for p in visible if p['direction'] != reverse]
        _require(2 <= len(outgoing) <= 3, "a clock must mark a fork")
        matched = [p for p in outgoing if p['colors'][0] == p['colors'][1]]
        labels = [p['colors'][0] for p in matched]
        _require(len(labels) == len(set(labels)), "duplicate outgoing identity")
        _require(labels.count(roles['mapping'][color]) == 1, "dormant fork also needs unique dispatch")
        degrees.append(len(outgoing))
        gated += len(outgoing) - len(matched)
        for p in matched:
            pending.append((end, p, False))
    _require(nodes - {source} == roles['hubs'], "unreachable clock dot")
    def topology(node):
        return tuple(sorted((topology(q) for q in children[node]), key=repr))
    return dict(lanes=lanes, terminals=terminals, degrees=degrees,
                gated=gated, topology=topology(source))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from the input alone. Relation tiles always retain their input values."""
    g = _grid(input_grid)
    roles = _read(g)
    return _answer(g, roles, _trace(g, roles))


def _coarse_neighbors(p, rows, cols):
    return [_add(p, d) for d in _D4
            if 0 <= _add(p, d)[0] < rows and 0 <= _add(p, d)[1] < cols]


def _skeleton(rng, size, depth, witness):
    """Sample a spatial tree with a chosen root-to-leaf clock depth.

    Unequal lattice spacings are only packing aids. Degree-two vertices become
    ordinary, possibly bent lanes; they never act as implicit clock ticks.
    """
    for _ in range(180):
        length = int(rng.integers(depth + 3, min(size * size - 2, depth + 8) + 1))
        source = int(rng.integers(size)), int(rng.integers(size))
        path, used = [source], {source}
        for _ in range(length - 1):
            options = [q for q in _coarse_neighbors(path[-1], size, size) if q not in used]
            if not options:
                break
            q = options[int(rng.integers(len(options)))]
            path.append(q)
            used.add(q)
        if len(path) != length:
            continue
        available = {i: [q for q in _coarse_neighbors(p, size, size) if q not in used]
                     for i, p in enumerate(path[1:-1], 1)}
        candidates = [i for i, options in available.items() if len(options) >= (2 if witness == 'switch' else 1)]
        if witness == 'switch':
            candidates = [i for i in candidates if
                          (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1]) !=
                          (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])]
        if len(candidates) < depth:
            continue
        indices = sorted(map(int, rng.choice(candidates, depth, replace=False)))
        edges = [(a, b) for a, b in zip(path, path[1:])]
        branch_leaves = []
        good = True
        for i in indices:
            room = sum(q not in used for q in _coarse_neighbors(path[i], size, size))
            branches = 2 if witness == 'switch' else (2 if room >= 2 and rng.random() < .25 else 1)
            for _ in range(branches):
                options = [q for q in _coarse_neighbors(path[i], size, size) if q not in used]
                if not options:
                    good = False
                    break
                q = options[int(rng.integers(len(options)))]
                edges.append((path[i], q))
                used.add(q)
                # A branch may itself bend or extend; it is not a fixed stub.
                for _ in range(int(rng.integers(0, 3))):
                    options = [v for v in _coarse_neighbors(q, size, size) if v not in used]
                    if not options:
                        break
                    v = options[int(rng.integers(len(options)))]
                    edges.append((q, v))
                    used.add(v)
                    q = v
                branch_leaves.append(q)
            if not good:
                break
        if not good:
            continue
        if witness != 'switch':
            # Independently expand dormant subtrees, including forks beyond
            # other dormant forks. Space can naturally limit the chosen count.
            desired = int(rng.choice([0, 1, 2, 3], p=[.5, .25, .18, .07]))
            for _ in range(desired):
                options_by_leaf = [(p, [q for q in _coarse_neighbors(p, size, size) if q not in used])
                                   for p in branch_leaves]
                options_by_leaf = [(p, qs) for p, qs in options_by_leaf if len(qs) >= 2]
                if not options_by_leaf:
                    break
                leaf, options = options_by_leaf[int(rng.integers(len(options_by_leaf)))]
                branch_leaves.remove(leaf)
                rng.shuffle(options)
                count = 3 if len(options) >= 3 and rng.random() < .3 else 2
                for q in options[:count]:
                    edges.append((leaf, q))
                    used.add(q)
                    branch_leaves.append(q)
        adjacency = {p: set() for p in used}
        for a, b in edges:
            adjacency[a].add(b)
            adjacency[b].add(a)
        return adjacency, path
    return None


def _axes(rng, size):
    for _ in range(50):
        gaps = list(map(int, rng.integers(5, 7, size=size - 1)))
        start, end = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        if start + sum(gaps) + end + 1 <= 30:
            points = [start]
            for gap in gaps:
                points.append(points[-1] + gap)
            return points, points[-1] + end + 1
    return list(range(2, 2 + 5 * size, 5)), 5 * (size - 1) + 5


def _polyline(points):
    path = [points[0]]
    for target in points[1:]:
        p = path[-1]
        d = int(np.sign(target[0] - p[0])), int(np.sign(target[1] - p[1]))
        _require((d[0] == 0) != (d[1] == 0), "orthogonal polyline required")
        while p != target:
            p = _add(p, d)
            path.append(p)
    return path


def _compress(adjacency, source, row_axis, col_axis):
    vertices = {p for p, adjacent in adjacency.items() if len(adjacent) != 2}
    _require(source in vertices, "root must be a leaf")
    edges, graph = [], {p: [] for p in vertices}
    seen = set()
    for node in sorted(vertices):
        for neighbor in sorted(adjacency[node]):
            if frozenset((node, neighbor)) in seen:
                continue
            points, prev, p = [node, neighbor], node, neighbor
            seen.add(frozenset((node, neighbor)))
            while p not in vertices:
                q = next(v for v in adjacency[p] if v != prev)
                seen.add(frozenset((p, q)))
                points.append(q)
                prev, p = p, q
            cells = _polyline([(row_axis[r], col_axis[c]) for r, c in points])
            index = len(edges)
            edges.append(dict(a=node, b=p, path=cells))
            graph[node].append((p, index))
            graph[p].append((node, index))
    return edges, graph


def _construct(rng, palette, depth, witness):
    bg, marker = palette[:2]
    k = 3 if witness == 'switch' else int(rng.integers(3, 7))
    identities = palette[2:2+k]
    # A three-color chain is present in both minimal training witnesses. The
    # remaining relations can converge, cycle or be identity relations.
    values = list(map(int, rng.choice(identities, size=k)))
    values[0], values[1] = identities[1], identities[2]
    values[2] = int(rng.choice(identities))
    mapping = dict(zip(identities, values))
    initial = identities[0]
    size = (int(rng.choice([3, 4])) if depth == 1 else
            int(rng.choice([4, 5])) if depth == 2 else 5)
    skeleton = _skeleton(rng, size, depth, witness)
    if skeleton is None:
        return None
    adjacency, main = skeleton
    ra, h = _axes(rng, size)
    ca, w = _axes(rng, size)
    source_node, target_node = main[0], main[-1]
    edges, graph = _compress(adjacency, source_node, ra, ca)
    source = ra[source_node[0]], ca[source_node[1]]
    hubs = {(ra[p[0]], ca[p[1]]) for p in graph if len(graph[p]) >= 3}
    # Root and orient the compressed tree; find the desired structural depth.
    parent, order = {source_node: None}, [source_node]
    for node in order:
        for child, index in sorted(graph[node]):
            if child not in parent:
                parent[child] = node
                order.append(child)
    main_edges, cursor = set(), target_node
    while parent[cursor] is not None:
        par = parent[cursor]
        main_edges.add(next(i for q, i in graph[cursor] if q == par))
        cursor = par
    incoming = {source_node: initial}
    first_hub, wrong_edge = None, None
    for node in order:
        children = [(q, i) for q, i in graph[node] if parent.get(q) == node]
        if not children:
            continue
        before = incoming[node]
        after = before if node == source_node else mapping[before]
        selected = next((i for q, i in children if i in main_edges), None)
        if selected is None:
            selected = children[int(rng.integers(len(children)))][1]
        other_colors = [c for c in identities if c != after]
        rng.shuffle(other_colors)
        if witness == 'switch' and len(children) == 3:
            other_colors.remove(before)
            other_colors.insert(0, before)
            first_hub = node
        for child, i in children:
            color = after if i == selected else other_colors.pop(0)
            edges[i]['color'] = color
            incoming[child] = color
            if edges[i]['a'] != node:
                edges[i]['a'], edges[i]['b'] = edges[i]['b'], edges[i]['a']
                edges[i]['path'].reverse()
            if witness == 'switch' and node == first_hub and i != selected and color != before:
                wrong_edge = i
    # Open exits extend the last segment to the border. Closed exits keep a cap.
    target_edge = next(i for q, i in graph[target_node])
    for node in order:
        if node == source_node or len(graph[node]) != 1:
            continue
        i = graph[node][0][1]
        want_open = (witness == 'switch' if node == target_node and witness in ('switch', 'repeat')
                     else rng.random() < .4)
        if want_open:
            path = edges[i]['path']
            d = path[-1][0] - path[-2][0], path[-1][1] - path[-2][1]
            p = path[-1]
            while 0 <= p[0] + d[0] < h and 0 <= p[1] + d[1] < w:
                p = _add(p, d)
                path.append(p)
    all_paths = [p for e in edges for p in e['path']]
    allowed_duplicates = set(hubs) | {source}
    counts = {}
    for p in all_paths:
        counts[p] = counts.get(p, 0) + 1
    if any(n > 1 and p not in allowed_duplicates for p, n in counts.items()):
        return None
    cells, walls = set(all_paths), {}
    for e in edges:
        for p in e['path']:
            for d in _D4:
                q = _add(p, d)
                if not (0 <= q[0] < h and 0 <= q[1] < w) or q in cells:
                    continue
                if max(abs(q[0]-source[0]), abs(q[1]-source[1])) <= 2:
                    continue
                if any(max(abs(q[0]-a), abs(q[1]-b)) <= 1 for a, b in hubs):
                    continue
                if q in walls and walls[q] != e['color']:
                    return None
                walls[q] = e['color']
    g = np.full((h, w), bg, dtype=int)
    for p, color in walls.items():
        g[p] = color
    for p in hubs:
        g[p] = marker
    for d in _D4:
        g[_add(source, d)] = marker
    g[source] = initial
    if wrong_edge is not None:
        e = edges[wrong_edge]
        p = e['path'][2]
        d = e['path'][1][0] - e['path'][0][0], e['path'][1][1] - e['path'][0][1]
        n = -d[1], d[0]
        g[_add(p, n)] = mapping[incoming[first_hub]]
        # The opposite flank still carries the unused edge's different color.
    # Interrupt an ordinary straight run, away from every port and bend.
    gap_candidates = []
    for i in sorted(main_edges):
        path = edges[i]['path']
        directions = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
        for j in range(3, len(path)-3):
            p = path[j]
            if len(set(directions[j-2:j+2])) != 1:
                continue
            if max(abs(p[0]-source[0]), abs(p[1]-source[1])) <= 4:
                continue
            if any(max(abs(p[0]-a), abs(p[1]-b)) <= 3 for a, b in hubs):
                continue
            d = directions[j]
            n = -d[1], d[0]
            a, b = _add(p, n), _add(p, n, -1)
            if _inside(g, a) and _inside(g, b) and g[a] == g[b] == edges[i]['color']:
                gap_candidates.append((a, b))
    if witness == 'repeat' and not gap_candidates:
        return None
    if gap_candidates and (witness == 'repeat' or rng.random() < .6):
        a, b = gap_candidates[int(rng.integers(len(gap_candidates)))]
        g[a] = g[b] = bg
    # The table is dispersed into currently blank spaces. Its spatial order is
    # irrelevant; duplicate bodies are permitted only with identical centers.
    relations = list(mapping.items())
    if witness != 'switch':
        relations += [relations[int(rng.integers(k))] for _ in range(int(rng.integers(0, 3)))]
    rng.shuffle(relations)
    for body, value in relations:
        candidates = []
        for r in range(1, h-1):
            for c in range(1, w-1):
                if np.all(g[max(0, r-2):min(h, r+3), max(0, c-2):min(w, c+3)] == bg):
                    candidates.append((r, c))
        if not candidates:
            return None
        r, c = candidates[int(rng.integers(len(candidates)))]
        g[r-1:r+2, c-1:c+2] = body
        g[r, c] = value
    destination = edges[target_edge]['path'][-1]
    final = initial
    for _ in range(depth):
        final = mapping[final]
    out = g.copy()
    out[source] = bg
    out[destination] = final
    try:
        roles = _read(g)
        trace = _trace(g, roles)
        _network(g, roles)
        if len(trace['events']) != depth or trace['destination'] != destination or trace['final'] != final:
            return None
        if witness == 'repeat':
            corners = {p for p, before, after in trace['bends']}
            if not corners or not any(mapping[c] != c for p, d, c in trace['trail'] if p in corners):
                return None
        if witness == 'switch':
            event = trace['events'][0]
            if not any(a != b and final in (a, b) for a, b in (p['colors'] for p in event['ports'])):
                return None
        if not np.array_equal(_answer(g, roles, trace), out):
            return None
    except AssertionError:
        return None
    return dict(input=g, output=out)


def _counts(num_train, num_test):
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    palette = tuple(map(int, rng.permutation(10)))
    mixed_roles = bool(rng.random() < .6)
    mixed_orientation = bool(rng.random() < .7)
    rotation = int(rng.integers(4))
    puzzle, seen = {'train': [], 'test': []}, set()
    modes = ['switch', 'repeat'] + ['ordinary'] * (int(num_train) - 2)
    rng.shuffle(modes)
    for split, count in (('train', int(num_train)), ('test', int(num_test))):
        for index in range(count):
            mode = modes[index] if split == 'train' else 'target'
            depth = (1 if mode == 'switch' else 2 if mode == 'repeat' else
                     int(rng.integers(1, 3)) if split == 'train' else int(rng.integers(3, 5)))
            roles = tuple(map(int, rng.permutation(10))) if mixed_roles else palette
            for _ in range(320):
                pair = _construct(rng, roles, depth, mode)
                if pair is None:
                    continue
                k = int(rng.integers(4)) if mixed_orientation else rotation
                reflected = bool(rng.integers(2))
                pair = {key: np.rot90(g, k).copy() for key, g in pair.items()}
                if reflected:
                    pair = {key: np.fliplr(g).copy() for key, g in pair.items()}
                signature = _key(pair['input'])
                if signature in seen:
                    continue
                seen.add(signature)
                puzzle[split].append(pair)
                break
            else:
                raise AssertionError("bounded tree packing exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(g):
    roles = _read(g)
    t = _trace(g, roles)
    network = _network(g, roles)
    # A gap must be inside a straight run with matching support before/after.
    support = [_flanks(g, p, d) == (c, c) for p, d, c in t['trail']]
    gaps = 0
    for j in range(1, len(support)-1):
        if not support[j] and support[j-1] and support[j+1]:
            a, b, c = t['trail'][j-1:j+2]
            if a[1] == b[1] == c[1] and b[0] not in roles['hubs']:
                gaps += 1
    mismatched = sum(a != b and e['after'] in (a, b) for e in t['events']
                     for a, b in (p['colors'] for p in e['ports']))
    valid_other = sum(a == b and a != e['after'] for e in t['events']
                      for a, b in (p['colors'] for p in e['ports']))
    corners = {p for p, before, after in t['bends']}
    nontrivial_bend = any(roles['mapping'][c] != c for p, d, c in t['trail'] if p in corners)
    dest = t['destination']
    return dict(shape=g.shape, clocks=len(t['events']), hubs=len(roles['hubs']),
                cards=len(roles['cards']), identities=len(roles['mapping']),
                bends=len(t['bends']), distance=len(t['trail']), gaps=gaps,
                nontrivial_bend=nontrivial_bend,
                mismatch=mismatched, competing=valid_other,
                changes=sum(e['before'] != e['after'] for e in t['events']),
                duplicates=len(roles['cards'])-len(roles['mapping']),
                solid=sum(c['body'] == c['value'] for c in roles['cards']),
                open=dest[0] in (0, g.shape[0]-1) or dest[1] in (0, g.shape[1]-1),
                background=roles['background'], marker=roles['marker'],
                initial=int(g[roles['source']]), final=t['final'],
                lanes=len(network['lanes']), leaves=len(network['terminals']),
                fork_degrees=tuple(sorted(network['degrees'])), topology=network['topology'],
                function=tuple(sorted(roles['mapping'].items())))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _counts(len(puzzle['train']), len(puzzle['test']))
    seen, features = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            g, out = _grid(pair['input']), _grid(pair['output'])
            _require(g.shape == out.shape, "dimensions must be preserved")
            signature = _key(g)
            _require(signature not in seen, "duplicate input")
            seen.add(signature)
            roles = _read(g)
            trace = _trace(g, roles)
            _network(g, roles)
            _require(trace['destination'] is not None, "pulse must reach a free terminal")
            _require(len({e['cell'] for e in trace['events']}) == len(trace['events']), "revisited clock")
            _require(np.array_equal(_answer(g, roles, trace), out), "rail dispatch or color composition mismatch")
            changed = set(map(tuple, np.argwhere(g != out).tolist()))
            _require(changed == {roles['source'], trace['destination']}, "only pulse source and terminal may change")
            _require(all(out[p] == g[p] for p in roles['card_cells'] | roles['arms'] | roles['hubs']),
                     "relation tiles and structural markers must survive")
            for e in trace['events']:
                _require(e['after'] == roles['mapping'][e['before']], "exactly one lookup per clock")
                _require(len(e['choices']) == 1, "unique selected paired port")
            features[split].append(_features(g))
    train, test = features['train'], features['test']
    _require(any(f['clocks'] == 1 and f['mismatch'] and f['competing'] and f['open'] for f in train),
             "matched/mismatched/other-color branch witness required")
    _require(any(f['clocks'] == 2 and f['nontrivial_bend'] and f['gaps'] and not f['open'] for f in train),
             "repeated lookup, ordinary bend, inertia and internal stop witness required")
    _require(any(f['changes'] >= 2 for f in train), "repeated nontrivial composition must be taught")
    _require(all(f['clocks'] in (1, 2) for f in train), "training depth regime")
    _require(all(f['clocks'] in (3, 4) for f in test), "test must extend demonstrated clock iteration")


def self_test():
    """Bounded 42-seed audit plus causal, semantic and count-contract checks."""
    puzzles, rows, signatures = [], [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        puzzles.append(puzzle)
        signatures.add(tuple(_key(pair['input']) for s in ('train', 'test') for pair in puzzle[s]))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                _require(np.array_equal(solve(pair['input']), pair['output']), "input-only solve")
                rows.append(_features(pair['input']))
        # Both mechanisms are essential in actual training examples.
        _require(any(not np.array_equal(
            _answer(pair['input'], _read(pair['input']),
                    _trace(pair['input'], _read(pair['input']), lookup=False, permissive=True)),
            pair['output']) for pair in puzzle['train']), "lookup ablation has no effect")
        _require(any(not np.array_equal(
            _answer(pair['input'], _read(pair['input']),
                    _trace(pair['input'], _read(pair['input']), turns=False)),
            pair['output']) for pair in puzzle['train']), "rail-turn ablation has no effect")
        _require(any(not np.array_equal(
            _answer(pair['input'], _read(pair['input']),
                    _trace(pair['input'], _read(pair['input']), extra_bend_lookups=True)),
            pair['output']) for pair in puzzle['train']), "unmarked-bend lookup shortcut survives")
    _require(len(signatures) == 42, "cross-seed variation")
    _require(len({r['shape'] for r in rows}) >= 40, "dimension diversity")
    _require({1, 2, 3, 4} <= {r['clocks'] for r in rows}, "clock depth diversity")
    _require({3, 4, 5, 6} <= {r['identities'] for r in rows}, "table size diversity")
    _require(len({r['distance'] for r in rows}) >= 20, "route-length diversity")
    _require(len({r['hubs'] for r in rows}) >= 4, "network topology diversity")
    _require(len({r['topology'] for r in rows}) >= 8, "branch topology diversity")
    _require(len({r['function'] for r in rows}) > 60, "visible map diversity")
    _require(len({(r['background'], r['marker']) for r in rows}) >= 30, "role diversity")
    _require(any(r['duplicates'] for r in rows) and any(r['solid'] for r in rows), "donor variety")
    _require(any(r['initial'] == r['final'] for r in rows), "cycles may restore the initial state")
    repeated = generate_puzzle(0)
    _require(all(_key(a[k]) == _key(b[k]) for s in puzzles[0]
                 for a, b in zip(puzzles[0][s], repeated[s]) for k in ('input', 'output')), "determinism")
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(9000 + 10*a + b, a, b)
            _require(len(p['train']) == a and len(p['test']) == b, "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # A donor-center mutation changes which branch is selected. The switch
    # witness has one physical tile per identity, so no conflicting duplicate.
    witness = next(pair for pair in puzzles[0]['train'] if _features(pair['input'])['mismatch'])
    g = witness['input'].copy()
    roles = _read(g)
    trace = _trace(g, roles)
    event = trace['events'][0]
    alternative = next(p['colors'][0] for p in event['ports']
                       if p['colors'][0] == p['colors'][1] != event['after'])
    donor = next(c for c in roles['cards'] if c['body'] == event['before'])
    g[donor['center']] = alternative
    _require(not np.array_equal(solve(g), witness['output']), "donor center must causally change dispatch")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad['test'][0]['output'] = bad['test'][0]['input'].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong terminal accepted")
    return dict(seeds=42, pairs=len(rows), distinct_puzzles=len(signatures),
                shapes=len({r['shape'] for r in rows}), route_lengths=len({r['distance'] for r in rows}),
                clock_depths=sorted({r['clocks'] for r in rows}),
                identity_counts=sorted({r['identities'] for r in rows}),
                topologies=len({r['topology'] for r in rows}),
                functions=len({r['function'] for r in rows}),
                semantic_verification="generator-owned invariants and ablations")


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
