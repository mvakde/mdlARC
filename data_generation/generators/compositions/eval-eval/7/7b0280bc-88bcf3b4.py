"""Thread a finite strand through the fewest affordable stations.

Station dots and guide lines use four-connectivity; slack material uses
eight-connectivity. All semantic roles are recoverable from the input.
No parent code, source grids, seed state, or training labels are used by solve.
"""
import numpy as np


D4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
D8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _neighbors(p, directions=D4):
    return [(p[0] + dr, p[1] + dc) for dr, dc in directions]


def _components(cells, directions=D4):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        todo, component = [start], {start}
        while todo:
            for q in _neighbors(todo.pop(), directions):
                if q in remaining:
                    remaining.remove(q)
                    component.add(q)
                    todo.append(q)
        result.append(component)
    return result


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((0 <= g) & (g <= 9))
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _inside(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _recover_graph(track, stations):
    """Recover edges and their ordered physical cells, including hidden bends."""
    nodes = sorted(stations)
    ids = {p: i for i, p in enumerate(nodes)}
    adjacency = [[] for _ in nodes]
    edges = []
    for component in _components(track):
        touching = sorted({q for p in component for q in _neighbors(p) if q in ids})
        assert len(touching) == 2, 'each guide must join exactly two station dots'
        a, b = touching
        permitted = component | {a, b}
        # Every guide is a simple chain. An elbow is not a station.
        assert all(sum(q in permitted for q in _neighbors(p)) == 2 for p in component)
        assert sum(q in component for q in _neighbors(a)) == 1
        assert sum(q in component for q in _neighbors(b)) == 1
        ordered, previous, current = [a], None, a
        while current != b:
            choices = [q for q in _neighbors(current) if q in permitted and q != previous]
            assert len(choices) == 1
            previous, current = current, choices[0]
            assert current not in ordered
            ordered.append(current)
        assert set(ordered) == permitted
        edge = len(edges)
        edges.append(ordered)
        adjacency[ids[a]].append((ids[b], edge))
        adjacency[ids[b]].append((ids[a], edge))
    assert len(nodes) >= 5 and all(1 <= len(a) <= 4 for a in adjacency)
    reached, todo = {0}, [0]
    while todo:
        for n, _ in adjacency[todo.pop()]:
            if n not in reached:
                reached.add(n)
                todo.append(n)
    assert len(reached) == len(nodes)
    assert len(edges) - len(nodes) + 1 <= 4
    return nodes, adjacency, edges


def _routes(nodes, adjacency, edges, source, target):
    """Enumerate simple graph routes; material cost is a separate quantity."""
    found = []
    todo = [(source, [source], [nodes[source]])]
    while todo:
        current, visited, pixels = todo.pop()
        if current == target:
            material = [p for p in pixels if p not in (nodes[source], nodes[target])]
            found.append(dict(nodes=visited, pixels=pixels, material=material,
                              hops=len(visited) - 2, cost=len(material)))
            continue
        for nxt, e in adjacency[current]:
            if nxt in visited:
                continue
            part = edges[e]
            if part[0] != nodes[current]:
                part = part[::-1]
            todo.append((nxt, visited + [nxt], pixels + part[1:]))
    assert found
    return sorted(found, key=lambda p: (p['hops'], p['cost'], p['nodes']))


def _select(routes, length):
    feasible = [r for r in routes if r['cost'] <= length]
    assert feasible, 'the strand cannot reach the destination on any route'
    best = min(r['hops'] for r in feasible)
    winners = [r for r in feasible if r['hops'] == best]
    assert len(winners) == 1, 'fewest-station feasible route is tied'
    return winners[0]


def _read_input(g):
    colors, counts = np.unique(g, return_counts=True)
    assert len(colors) == 5
    background = int(colors[np.argmax(counts)])
    assert np.count_nonzero(counts == counts.max()) == 1
    by_color = {int(c): set(map(tuple, np.argwhere(g == c).tolist()))
                for c in colors if c != background}
    pin_colors = [c for c, cells in by_color.items() if len(cells) == 2
                  and len(_components(cells)) == 2]
    assert len(pin_colors) == 1
    pin_color = pin_colors[0]
    ordinary_colors = [c for c, cells in by_color.items() if c != pin_color
                       and len(cells) >= 3 and all(len(p) == 1 for p in _components(cells))]
    assert len(ordinary_colors) == 1
    ordinary_color = ordinary_colors[0]
    candidates = []
    for track_color in by_color.keys() - {pin_color, ordinary_color}:
        pieces = _components(by_color[track_color])
        if len(pieces) < 4 or any(len(p) < 2 for p in pieces):
            continue
        wire_color = next(c for c in by_color if c not in (pin_color, ordinary_color, track_color))
        wire, track, pins = by_color[wire_color], by_color[track_color], by_color[pin_color]
        if len(wire) < 5 or len(_components(wire, D8)) != 1:
            continue
        # The anchor replaces one guide cell: it touches a pin and a guide.
        anchors = [p for p in wire if any(q in pins for q in _neighbors(p))
                   and any(q in track for q in _neighbors(p))]
        if len(anchors) != 1:
            continue
        anchor = anchors[0]
        if any(p != anchor and any(q in track for q in _neighbors(p)) for p in wire):
            continue
        roots = pins.intersection(_neighbors(anchor))
        if len(roots) != 1:
            continue
        root = next(iter(roots))
        target = next(p for p in pins if p != root)
        try:
            station_cells = pins | by_color[ordinary_color]
            assert all(len(c) == 1 for c in _components(station_cells))
            assert not any(q in station_cells - {root} for p in wire for q in _neighbors(p))
            nodes, adjacency, edges = _recover_graph(track | {anchor}, station_cells)
            source_id, target_id = nodes.index(root), nodes.index(target)
            assert len(adjacency[source_id]) == len(adjacency[target_id]) == 1
            routes = _routes(nodes, adjacency, edges, source_id, target_id)
            assert all(r['material'][0] == anchor for r in routes)
            chosen = _select(routes, len(wire))
            # A clear destination port: extend its last axial guide step.
            before = chosen['pixels'][-2]
            direction = (target[0] - before[0], target[1] - before[1])
            assert direction in D4
        except AssertionError:
            continue
        candidates.append(dict(background=background, track_color=track_color,
                               ordinary_color=ordinary_color, pin_color=pin_color,
                               wire_color=wire_color, wire=wire, track=track, pins=pins,
                               ordinary=by_color[ordinary_color], anchor=anchor, root=root,
                               target=target, direction=direction, nodes=nodes,
                               adjacency=adjacency, edges=edges, routes=routes, chosen=chosen))
    assert len(candidates) == 1, 'ambiguous or invalid input roles/network'
    return candidates[0]


def _strand_positions(d, route=None, length=None):
    route = d['chosen'] if route is None else route
    length = len(d['wire']) if length is None else length
    assert route['cost'] <= length
    target = d['target']
    before = route['pixels'][-2]
    dr, dc = target[0] - before[0], target[1] - before[1]
    tail = [(target[0] + k * dr, target[1] + k * dc)
            for k in range(1, length - route['cost'] + 1)]
    return route['material'] + tail


def _paint(g, d, route=None, length=None):
    out = g.copy()
    for p in d['wire']:
        out[p] = d['background']
    positions = _strand_positions(d, route, length)
    assert len(set(positions)) == len(positions)
    for p in positions:
        if _inside(p, g.shape):
            assert p not in d['pins']
            assert out[p] in (d['background'], d['track_color'], d['ordinary_color'])
            out[p] = d['wire_color']
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Predict from this input alone; the rule has no learned literal colors."""
    g = _grid(input_grid)
    return _paint(g, _read_input(g))


def _line(a, b):
    assert a[0] == b[0] or a[1] == b[1]
    dr = int(np.sign(b[0] - a[0]))
    dc = int(np.sign(b[1] - a[1]))
    n = abs(b[0] - a[0]) + abs(b[1] - a[1])
    return [(a[0] + k * dr, a[1] + k * dc) for k in range(n + 1)]


def _constructive_skeleton(rng, loops, lesson):
    """Procedural conflict construction used if random search is exhausted.

    Three independently spaced corridors have short/dense, medium/sparse,
    and long/dense routes. A sampled cross-link supplies a fourth competition
    when three cycles are requested. This is not a bank of complete grids.
    """
    root_c = int(rng.integers(5, 8))
    left = root_c + 4
    span = int(rng.integers(11, min(15, (24 if lesson else 25) - left) + 1))
    right = left + span
    top = int(rng.integers(1, 4))
    up = int(rng.integers(3, 6))
    down = up + int(rng.integers(2, 5))
    middle, bottom = top + up, top + up + down
    h = max(18, bottom + int(rng.integers(2, 5)))
    root, target = (middle, root_c), (middle, right + 3)
    lnode, rnode = (middle, left), (middle, right)
    track = set(_line(root, target))
    track.update(_line(lnode, (top, left)))
    track.update(_line((top, left), (top, right)))
    track.update(_line((top, right), rnode))
    nmiddle = int(rng.integers(2, 4 if span >= 12 else 3))
    dense_columns = [left + (k * span) // (nmiddle + 1) for k in range(1, nmiddle + 1)]
    dots = {lnode, rnode} | {(middle, c) for c in dense_columns}
    if rng.random() < .5:
        dots.add((top, left + span // 2))
    if loops >= 2:
        track.update(_line(lnode, (bottom, left)))
        track.update(_line((bottom, left), (bottom, right)))
        track.update(_line((bottom, right), rnode))
        cross_c = int(rng.choice(dense_columns))
        dots.update({(bottom, left), (bottom, right), (bottom, cross_c)})
        if loops == 3:
            track.update(_line((middle, cross_c), (bottom, cross_c)))
    all_dots = dots | {root, target}
    track -= all_dots
    nodes, adjacency, edges = _recover_graph(track, all_dots)
    routes = _routes(nodes, adjacency, edges, nodes.index(root), nodes.index(target))
    minimum = min(r['hops'] for r in routes)
    high = next(r for r in routes if r['hops'] == minimum)
    assert sum(r['hops'] == minimum for r in routes) == 1
    low_length = min(r['cost'] for r in routes)
    low = _select(routes, low_length)
    assert high['cost'] > low_length and low['hops'] > high['hops']
    if lesson:
        assert len(routes) == 3 and max(r['cost'] for r in routes) > high['cost']
    high_length = (max(r['cost'] for r in routes) if lesson else high['cost']) + int(rng.integers(3, 8))
    w = target[1] + 3 if lesson else min(30, target[1] + int(rng.integers(2, 4)))
    return dict(shape=(h, w), pins={root, target}, root=root, target=target,
                ordinary=dots, track=track, anchor=(middle, root_c + 1),
                low_length=low_length, high_length=high_length,
                routes=routes, high=high, low=low)


def _skeleton(rng, loops, lesson=False):
    """Random lattice subgraph, uneven spacing, and optional degree-two dots.

    This constructs a new guide network. It does not invoke either parent.
    Every returned skeleton has a real budget conflict, not just a color change.
    """
    for attempt in range(160):
        nr, nc = int(rng.integers(2, 5)), int(rng.integers(3, 5))
        vertices = [(r, c) for r in range(nr) for c in range(nc)]
        possible = [(a, b) for i, a in enumerate(vertices) for b in vertices[i + 1:]
                    if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
        # A randomized spanning tree and then a bounded number of extra edges.
        active = {vertices[int(rng.integers(len(vertices)))]}
        links = []
        while len(active) < len(vertices):
            frontier = [e for e in possible if (e[0] in active) != (e[1] in active)]
            e = frontier[int(rng.integers(len(frontier)))]
            active.update(e)
            links.append(e)
        extra = [e for e in possible if e not in links]
        rng.shuffle(extra)
        if len(extra) < loops:
            continue
        links += extra[:loops]
        slack_width = int(rng.integers(4, 7))
        root_c = slack_width + 1
        x = [root_c + 4]
        for _ in range(nc - 1):
            x.append(x[-1] + int(rng.integers(3, 7)))
        if x[-1] + 4 > 30:
            continue
        y = [int(rng.integers(1, 4))]
        for _ in range(nr - 1):
            y.append(y[-1] + int(rng.integers(3, 8)))
        height = max(16, y[-1] + int(rng.integers(2, 5)))
        if height > 30:
            continue
        anchors = {v: (y[v[0]], x[v[1]]) for v in vertices}
        degrees = {v: sum(v in e for e in links) for v in vertices}
        visible = {v for v in vertices if degrees[v] != 2 or rng.random() < .5}
        # Root/destination rows vary independently; all route alternatives
        # share their docking edges. Unused branches remain part of the scene.
        starts = [(r, 0) for r in range(nr)]
        ends = [(r, nc - 1) for r in range(nr)]
        choices = [(a, b) for a in starts for b in ends]
        rng.shuffle(choices)
        for start, end in choices:
            nodes_visible = visible | {start, end}
            if len(nodes_visible) < 3:
                continue
            root = (anchors[start][0], root_c)
            target = (anchors[end][0], x[-1] + 3)
            if lesson and target[1] + 3 > 30:
                continue
            node_cells = {anchors[v] for v in nodes_visible} | {root, target}
            track = set()
            for a, b in links:
                track.update(_line(anchors[a], anchors[b]))
            track.update(_line(root, anchors[start]))
            track.update(_line(anchors[end], target))
            track -= node_cells
            try:
                nodes, adjacency, edges = _recover_graph(track, node_cells)
                routes = _routes(nodes, adjacency, edges, nodes.index(root), nodes.index(target))
                if len(routes) < (2 if loops == 1 else 3):
                    continue
                if lesson and len(routes) != 3:
                    continue
                if loops == 3 and len(routes) < 4:
                    continue
                best_hops = min(r['hops'] for r in routes)
                high_choices = [r for r in routes if r['hops'] == best_hops]
                if len(high_choices) != 1:
                    continue
                high = high_choices[0]
                # The lesson also refutes "use the longest route that fits".
                # Its minimum-station route has both a shorter and a longer
                # competitor, and the high budget can afford all three.
                if lesson and not any(r['cost'] > high['cost'] for r in routes):
                    continue
                shorter = [r for r in routes if r['cost'] < high['cost']]
                low_options = []
                for budget in sorted({r['cost'] for r in shorter}):
                    feasible = [r for r in routes if r['cost'] <= budget]
                    mh = min(r['hops'] for r in feasible)
                    winners = [r for r in feasible if r['hops'] == mh]
                    if len(winners) == 1 and mh > high['hops']:
                        low_options.append((budget, winners[0]))
                if not low_options:
                    continue
                low_length, low = low_options[int(rng.integers(len(low_options)))]
                high_length = (max(r['cost'] for r in routes) if lesson else high['cost']) + int(rng.integers(3, 8))
                # The slack is packed into a left-side region of the same
                # canvas; no conceptual strand material is hidden at input.
                capacity = height * root_c - 6
                if high_length > capacity or low_length < 5:
                    continue
                width = target[1] + 3 if lesson else min(30, target[1] + int(rng.integers(2, 4)))
                anchor = (root[0], root[1] + 1)
                return dict(shape=(height, width), pins={root, target}, root=root,
                            target=target, ordinary=node_cells - {root, target},
                            track=track, anchor=anchor, low_length=low_length,
                            high_length=high_length, routes=routes, high=high, low=low)
            except AssertionError:
                continue
    return _constructive_skeleton(rng, loops, lesson)


def _slack(rng, skeleton, length, diagonal):
    root, anchor = skeleton['root'], skeleton['anchor']
    h, _ = skeleton['shape']
    # The bridge may touch the source pin, but only the anchor touches both a
    # pin and a guide. All remaining material lies behind the guide network.
    side = 1 if root[0] + 1 < h else -1
    bridge = (root[0] + side, root[1])
    cells = {anchor, bridge}
    forbidden = {root, (root[0], root[1] - 1)}
    allowed = {(r, c) for r in range(h) for c in range(root[1])} - forbidden
    allowed.add(bridge)
    if not diagonal:
        cells.add((root[0] + side, root[1] + 1))
    # Dense, sparse and directional growth change shape independently of count.
    style = int(rng.integers(3))
    directions = D8 if diagonal or style == 1 else D4
    while len(cells) < length:
        frontier = sorted({q for p in cells for q in _neighbors(p, directions)
                           if q in allowed and q not in cells})
        assert frontier
        scores = []
        for p in frontier:
            degree = sum(q in cells for q in _neighbors(p, D8))
            scores.append((degree + 1) ** 2 if style == 0 else
                          1 / (degree + 1) if style == 1 else 1 + p[1])
        weights = np.array(scores, dtype=float)
        cells.add(frontier[int(rng.choice(len(frontier), p=weights / weights.sum()))])
    assert len(cells) == length and len(_components(cells, D8)) == 1
    assert any(q in cells for p in cells for q in _neighbors(p))
    if diagonal:
        assert len(next(c for c in _components(cells) if anchor in c)) == 1
    return cells


def _make_pair(rng, skeleton, length, roles, transform, diagonal):
    background, track_color, ordinary_color, pin_color, wire_color = roles
    g = np.full(skeleton['shape'], background, dtype=np.int8)
    for role, color in (('track', track_color), ('ordinary', ordinary_color), ('pins', pin_color)):
        for p in skeleton[role]:
            g[p] = color
    wire = _slack(rng, skeleton, length, diagonal)
    for p in wire:
        assert g[p] == background or p == skeleton['anchor']
        g[p] = wire_color
    # Output construction is based on the sampled graph, then compared with
    # reconstruction of the actual returned pixels in validation.
    route = _select(skeleton['routes'], length)
    out = g.copy()
    for p in wire:
        out[p] = background
    for p in route['material']:
        out[p] = wire_color
    for k in range(1, length - route['cost'] + 1):
        p = (skeleton['target'][0], skeleton['target'][1] + k)
        if _inside(p, g.shape):
            assert out[p] == background
            out[p] = wire_color
    rotation, reflect = transform
    g, out = np.rot90(g, rotation), np.rot90(out, rotation)
    if reflect:
        g, out = g[:, ::-1], out[:, ::-1]
    return {'input': g.copy(), 'output': out.copy()}


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + ' must be an integer')
    if not lo <= value <= hi:
        raise ValueError(name + ' outside supported range')
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(int(seed))
    roles = tuple(map(int, rng.choice(10, 5, replace=False)))
    # A paired lesson isolates the effect of material length on the same graph.
    lesson = _skeleton(rng, loops=2, lesson=True)
    # The short lesson displays all of its surplus, so conservation and the
    # destination port are taught without relying exclusively on clipping.
    free_tail = lesson['shape'][1] - lesson['target'][1] - 1
    delta = min(free_tail, lesson['high']['cost'] - lesson['low_length'] - 1)
    assert delta >= 1
    lesson['low_length'] += 1
    transform = (int(rng.integers(4)), bool(rng.integers(2)))
    training = [_make_pair(rng, lesson, lesson['high_length'], roles, transform, True),
                _make_pair(rng, lesson, lesson['low_length'], roles, transform, False)]
    while len(training) < num_train:
        s = _skeleton(rng, loops=int(rng.integers(1, 4)))
        length = s['high_length'] if rng.random() < .5 else s['low_length']
        fresh_roles = tuple(map(int, rng.choice(10, 5, replace=False)))
        training.append(_make_pair(rng, s, length, fresh_roles,
                                  (int(rng.integers(4)), bool(rng.integers(2))), bool(rng.integers(2))))
    rng.shuffle(training)
    puzzle = {'train': training, 'test': []}
    seen = {_key(p['input']) for p in training}
    assert len(seen) == num_train
    for _ in range(num_test):
        for retry in range(12):
            # Four or more competing routes, versus the paired three-route lesson.
            s = _skeleton(rng, loops=3)
            length = s['high_length'] if rng.random() < .5 else s['low_length']
            if rng.random() < .6:
                # Recombine exact fits, fully visible tails and clipped tails.
                route = _select(s['routes'], length)
                candidate_length = route['cost'] + int(rng.integers(0, 5))
                try:
                    _select(s['routes'], candidate_length)
                    length = candidate_length
                except AssertionError:
                    pass
            fresh_roles = tuple(map(int, rng.choice(10, 5, replace=False)))
            pair = _make_pair(rng, s, length, fresh_roles,
                              (int(rng.integers(4)), bool(rng.integers(2))), bool(rng.integers(2)))
            key = _key(pair['input'])
            if key not in seen:
                seen.add(key)
                puzzle['test'].append(pair)
                break
        else:
            raise RuntimeError('distinct test construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(g, d):
    routes, chosen = d['routes'], d['chosen']
    length = len(d['wire'])
    global_hops = min(r['hops'] for r in routes)
    positions = _strand_positions(d)
    return dict(shape=g.shape, length=length, stations=len(d['nodes']),
                selected_stations=chosen['hops'], selected_cost=chosen['cost'],
                route_count=len(routes), loop_rank=len(d['edges']) - len(d['nodes']) + 1,
                lengths=tuple(sorted({r['cost'] for r in routes})),
                station_counts=tuple(sorted({r['hops'] for r in routes})),
                station_witness=chosen['cost'] > min(r['cost'] for r in routes),
                budget_witness=chosen['hops'] > global_hops,
                longest_witness=any(chosen['cost'] < r['cost'] <= length for r in routes),
                diagonal=len(next(c for c in _components(d['wire']) if d['anchor'] in c)) < length,
                clipped=sum(not _inside(p, g.shape) for p in positions),
                surplus=length - chosen['cost'],
                bends=sum(any(a[0] != b[0] and b[1] != c[1] or a[1] != b[1] and b[0] != c[0]
                              for a, b, c in zip(e, e[1:], e[2:])) for e in d['edges']),
                degrees=tuple(sorted(len(a) for a in d['adjacency'])))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle['train']) <= 10 and 1 <= len(puzzle['test']) <= 4
    assert len(puzzle['train']) > len(puzzle['test'])
    seen, training = set(), []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
            g, out = _grid(pair['input']), _grid(pair['output'])
            assert g.shape == out.shape and _key(g) not in seen
            seen.add(_key(g))
            d = _read_input(g)
            assert np.array_equal(_paint(g, d), out)
            assert not np.array_equal(g, out)
            assert out[d['anchor']] == g[d['anchor']] == d['wire_color']
            assert all(out[p] == d['pin_color'] for p in d['pins'])
            positions = _strand_positions(d)
            assert len(positions) == len(d['wire'])
            visible = {p for p in positions if _inside(p, g.shape)}
            assert np.count_nonzero(out == d['wire_color']) == len(visible)
            # Off-route guides and station dots survive, even at a visited junction.
            preserve = (d['track'] | d['ordinary'] | d['pins']) - visible
            assert all(out[p] == g[p] for p in preserve)
            tail = positions[d['chosen']['cost']:]
            assert all(not _inside(p, g.shape) or g[p] == d['background'] for p in tail)
            f = _features(g, d)
            if split == 'train':
                training.append(f)
            else:
                assert f['route_count'] >= 4
    # These are semantic counterexamples, not merely structural hash checks.
    assert any(f['station_witness'] for f in training), 'no station-vs-distance lesson'
    assert any(f['budget_witness'] for f in training), 'no finite-material lesson'
    assert any(f['longest_witness'] for f in training), 'no longest-affordable shortcut lesson'
    assert any(f['diagonal'] for f in training), 'no eight-connected count lesson'
    assert any(f['clipped'] for f in training), 'no boundary clipping lesson'
    assert any(f['clipped'] == 0 for f in training), 'no visible material conservation lesson'
    assert any(f['clipped'] == 0 and f['surplus'] > 0 for f in training), 'no fully visible surplus lesson'


def self_test():
    """Bounded 42-seed audit plus semantic ablation and corruption checks."""
    shapes, structures, palettes, stations, costs, loops, directions = (set() for _ in range(7))
    stats = []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        if seed < 3:
            q = generate_puzzle(seed)
            assert all(_key(a[k]) == _key(b[k]) for s in p for a, b in zip(p[s], q[s])
                       for k in ('input', 'output'))
        for split in p:
            for pair in p[split]:
                g, out = pair['input'], pair['output']
                d = _read_input(g)
                assert np.array_equal(solve(g.copy()), out)
                f = _features(g, d)
                stats.append(f)
                shapes.add(g.shape)
                stations.add(f['stations'])
                costs.add(f['selected_cost'])
                loops.add(f['loop_rank'])
                directions.add(d['direction'])
                roles = (d['background'], d['track_color'], d['ordinary_color'], d['pin_color'], d['wire_color'])
                palettes.add(roles)
                semantic = np.zeros_like(g)
                for i, color in enumerate(roles):
                    semantic[g == color] = i
                structures.add(_key(semantic))
                if f['station_witness']:
                    shortest = min(d['routes'], key=lambda r: r['cost'])
                    assert shortest['cost'] <= len(d['wire'])
                    assert not np.array_equal(_paint(g, d, shortest), out)
                if f['budget_witness']:
                    cheapest_hops = min(d['routes'], key=lambda r: r['hops'])
                    assert cheapest_hops['cost'] > len(d['wire'])
                    # Ignoring B would thread a different set of graph pixels.
                    assert set(cheapest_hops['material']) != set(d['chosen']['material'])
                    assert not np.array_equal(_paint(g, d, cheapest_hops, cheapest_hops['cost']), out)
                if f['longest_witness']:
                    longest = max((r for r in d['routes'] if r['cost'] <= len(d['wire'])), key=lambda r: r['cost'])
                    assert not np.array_equal(_paint(g, d, longest), out)
                if split == 'train' and not f['clipped'] and f['surplus'] == 1:
                    # The paired short lesson leaves room for a second tail
                    # cell. A bounding-box-area budget must change the route
                    # or add that cell (or create an invalid minimum tie).
                    spans = np.ptp(np.array(list(d['wire'])), axis=0) + 1
                    area = int(np.prod(spans))
                    assert area > len(d['wire'])
                    try:
                        rectangle_route = _select(d['routes'], area)
                    except AssertionError:
                        pass
                    else:
                        assert not np.array_equal(_paint(g, d, rectangle_route, area), out)
        if seed == 0:
            damaged = {s: [{'input': a['input'].copy(), 'output': a['output'].copy()} for a in p[s]] for s in p}
            d = _read_input(damaged['test'][0]['input'])
            pin = min(d['pins'])
            damaged['test'][0]['output'][pin] = d['background']
            try:
                validate_generated_puzzle(damaged)
            except AssertionError:
                pass
            else:
                raise AssertionError('damaged pin accepted')
    assert len(shapes) >= 35 and len(structures) >= 160 and len(palettes) >= 75
    assert len(stations) >= 6 and len(costs) >= 20 and loops >= {1, 2, 3}
    assert directions == set(D4)
    assert any(f['bends'] > 0 for f in stats)
    assert any(1 in f['degrees'] and 3 in f['degrees'] for f in stats)
    assert any(not f['diagonal'] for f in stats)
    # Unit budgets: exact fit is affordable, one cell short is not; ties reject.
    dummy = [dict(hops=1, cost=9), dict(hops=2, cost=6)]
    assert _select(dummy, 9) is dummy[0] and _select(dummy, 8) is dummy[1]
    for routes, length in ((dummy, 5), ([dict(hops=1, cost=8), dict(hops=1, cost=9)], 9)):
        try:
            _select(routes, length)
        except AssertionError:
            pass
        else:
            raise AssertionError('infeasible or tied path accepted')
    for a, b in ((2, 1), (10, 4), (5, 3)):
        generate_puzzle(123, a, b)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    return dict(seeds=42, pairs=len(stats), shapes=len(shapes), structures=len(structures),
                palettes=len(palettes), station_counts=sorted(stations), selected_costs=sorted(costs),
                loop_ranks=sorted(loops), station_witnesses=sum(f['station_witness'] for f in stats),
                budget_witnesses=sum(f['budget_witness'] for f in stats),
                longest_affordable_witnesses=sum(f['longest_witness'] for f in stats),
                clipped_pairs=sum(bool(f['clipped']) for f in stats))
