"""Route a finite, oriented piece inventory through a station network.

The selected route is the unique minimum-station route whose socket cells can
be tiled using every loose piece exactly once.  Selected ordinary stations
acquire the endpoint color; socket cells retain the colors of their pieces.
All inference uses the supplied input, including palette-role recovery.
"""
from collections import Counter
from functools import lru_cache
from itertools import combinations

import numpy as np


_N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_MAX_ROUTES = 256
_MAX_SEARCH = 50000


class _Invalid(ValueError):
    """A scene is outside the family's uniqueness or readability constraints."""


def _require(condition, message):
    if not condition:
        raise _Invalid(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "Invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "Cells must be integers")
    _require(np.all((0 <= a) & (a <= 9)), "Invalid colors")
    return a.astype(np.int8, copy=True)


def _key(a):
    return a.shape, np.ascontiguousarray(a, dtype=np.int8).tobytes()


def _components(cells):
    remaining = set(cells)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr, dc in _N4:
                p = r + dr, c + dc
                if p in remaining:
                    remaining.remove(p)
                    stack.append(p)
        result.append(tuple(sorted(component)))
    return result


def _cells(mask):
    return [tuple(map(int, p)) for p in np.argwhere(mask)]


def _normalize(cells):
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _square(cells):
    h = max(r for r, _ in cells) - min(r for r, _ in cells) + 1
    w = max(c for _, c in cells) - min(c for _, c in cells) + 1
    return h if h == w and len(cells) == h * w else 0


def _network(board, background, panel, ordinary, endpoint):
    station_mask = (board == ordinary) | (board == endpoint)
    stations = _components(_cells(station_mask))
    sizes = [_square(s) for s in stations]
    _require(5 <= len(stations) <= 16 and len(set(sizes)) == 1 and sizes[0] >= 2,
             "Stations must be separated equal solid squares")
    station_map = {p: i for i, block in enumerate(stations) for p in block}
    endpoints = []
    for i, block in enumerate(stations):
        colors = {int(board[p]) for p in block}
        _require(len(colors) == 1, "A station must be monochrome")
        if next(iter(colors)) == endpoint:
            endpoints.append(i)
    _require(len(endpoints) == 2, "Exactly two endpoints are required")
    masks = _components(_cells(board == background))
    adjacency = [[] for _ in stations]
    edges = []
    for e, mask in enumerate(masks):
        touching = {station_map[q] for r, c in mask for dr, dc in _N4
                    if (q := (r + dr, c + dc)) in station_map}
        _require(len(touching) == 2, "Each socket must join exactly two stations")
        a, b = sorted(touching)
        edges.append((a, b))
        adjacency[a].append((b, e))
        adjacency[b].append((a, e))
    _require(len(set(edges)) == len(edges), "Parallel sockets are excluded")
    _require(all(1 <= len(a) <= 4 for a in adjacency), "Invalid station degrees")
    reached, stack = {0}, [0]
    while stack:
        for b, _ in adjacency[stack.pop()]:
            if b not in reached:
                reached.add(b)
                stack.append(b)
    _require(len(reached) == len(stations), "The network must be connected")
    centers = [np.mean(block, axis=0) for block in stations]
    lengths = [float(np.abs(centers[a] - centers[b]).sum()) for a, b in edges]
    return dict(stations=stations, size=sizes[0], endpoints=endpoints,
                masks=masks, edges=edges, adjacency=adjacency, lengths=lengths,
                roles=(background, panel, ordinary, endpoint))


def _paths(data):
    start, target = data['endpoints']
    stack = [(start, (start,), ())]
    result = []
    while stack:
        node, nodes, edges = stack.pop()
        if node == target:
            result.append((nodes, edges))
            _require(len(result) <= _MAX_ROUTES, "Too many simple endpoint routes")
            continue
        for nxt, edge in data['adjacency'][node]:
            if nxt not in nodes:
                stack.append((nxt, nodes + (nxt,), edges + (edge,)))
    return sorted(result, key=lambda p: (len(p[0]), p))


def _holes(data, path):
    return frozenset(p for edge in path[1] for p in data['masks'][edge])


def _tile(holes, pieces, existence_only=False, node_limit=_MAX_SEARCH):
    """Exact inventory cover; report up to two distinct colored completions.

    Pieces are translated, with their supplied orientation unchanged. Identical
    shape/color copies are represented by a count. For a monochrome inventory,
    existence already proves that its colored completion is unique.
    """
    holes = tuple(sorted(holes))
    if sum(len(shape) for _, shape in pieces) != len(holes):
        return ()
    inventory = Counter((int(color), _normalize(shape)) for color, shape in pieces)
    types = sorted(inventory)
    index = {cell: i for i, cell in enumerate(holes)}
    options = [[] for _ in holes]
    for t, (color, shape) in enumerate(types):
        anchor = shape[0]
        for r, c in holes:
            origin = r - anchor[0], c - anchor[1]
            translated = [(origin[0] + dr, origin[1] + dc) for dr, dc in shape]
            if not all(p in index for p in translated):
                continue
            ids = [index[p] for p in translated]
            mask = sum(1 << i for i in ids)
            paint = sum((color + 1) << (4 * i) for i in ids)
            move = t, mask, paint
            for i in ids:
                options[i].append(move)
    first_only = existence_only or len({color for color, _ in types}) == 1
    visits = 0

    @lru_cache(maxsize=None)
    def search(remaining, counts):
        nonlocal visits
        visits += 1
        _require(node_limit is None or visits <= node_limit, "Exact-cover search bound exceeded")
        if not remaining:
            return (0,) if not any(counts) else ()
        best = None
        bits = remaining
        while bits:
            bit = bits & -bits
            i = bit.bit_length() - 1
            moves = [m for m in options[i] if counts[m[0]] and m[1] & remaining == m[1]]
            if not moves:
                return ()
            if best is None or len(moves) < len(best):
                best = moves
                if len(best) == 1:
                    break
            bits ^= bit
        answers = set()
        for t, mask, paint in best:
            rest = list(counts)
            rest[t] -= 1
            for suffix in search(remaining ^ mask, tuple(rest)):
                answers.add(paint | suffix)
                if first_only or len(answers) == 2:
                    return tuple(sorted(answers))
        return tuple(sorted(answers))

    codes = search((1 << len(holes)) - 1, tuple(inventory[t] for t in types))
    return tuple({cell: ((code >> (4 * i)) & 15) - 1 for i, cell in enumerate(holes)}
                 for code in codes)


def _select(data, pieces, collect=False, node_limit=_MAX_SEARCH):
    routes = _paths(data)
    area = sum(len(shape) for _, shape in pieces)
    viable = []
    chosen = None
    minimum = None
    for path in routes:
        if minimum is not None and len(path[0]) > minimum and not collect:
            break
        holes = _holes(data, path)
        if len(holes) != area:
            continue
        answers = _tile(holes, pieces, existence_only=minimum is not None and len(path[0]) > minimum,
                        node_limit=node_limit)
        if not answers:
            continue
        viable.append(path)
        if minimum is None:
            minimum = len(path[0])
            _require(len(answers) == 1, "The winning route has ambiguous colored assembly")
            chosen = path, answers[0]
        elif len(path[0]) == minimum:
            raise _Invalid("Minimum feasible station route is tied")
    _require(chosen is not None, "No route can consume the inventory")
    return chosen[0], chosen[1], routes, viable


def _read_scene(inp):
    background = int(inp[0, 0])
    _require(all(int(inp[p]) == background for p in ((0, -1), (-1, 0), (-1, -1))),
             "Canvas corners must identify the background")
    candidates = []
    for panel in map(int, np.unique(inp)):
        if panel == background:
            continue
        cells = np.argwhere(inp == panel)
        low, high = cells.min(axis=0), cells.max(axis=0) + 1
        r, c = map(int, low)
        h, w = map(int, high - low)
        if min(h, w) < 6 or r == 0 or c == 0 or r + h == inp.shape[0] or c + w == inp.shape[1]:
            continue
        board = inp[r:r + h, c:c + w]
        if not (np.all(board[0] == panel) and np.all(board[-1] == panel)
                and np.all(board[:, 0] == panel) and np.all(board[:, -1] == panel)):
            continue
        colors = set(map(int, np.unique(board))) - {background, panel}
        if len(colors) != 2:
            continue
        by_color = {color: _components(_cells(board == color)) for color in colors}
        endpoint_colors = [color for color, blocks in by_color.items()
                           if len(blocks) == 2 and all(_square(b) >= 2 for b in blocks)]
        if len(endpoint_colors) != 1:
            continue
        endpoint = endpoint_colors[0]
        ordinary = next(color for color in colors if color != endpoint)
        try:
            data = _network(board, background, panel, ordinary, endpoint)
            exterior = inp != background
            exterior[r:r + h, c:c + w] = False
            components = _components(_cells(exterior))
            _require(2 <= len(components) <= 40, "Invalid loose-piece count")
            pieces = []
            for component in components:
                palette = {int(inp[p]) for p in component}
                _require(len(palette) == 1 and palette.isdisjoint(data['roles']),
                         "Loose pieces must be monochrome and use a disjoint palette")
                pieces.append((next(iter(palette)), _normalize(component)))
            _require(4 <= sum(len(sh) for _, sh in pieces) <= 70, "Invalid inventory area")
            _paths(data)
            candidates.append((board.copy(), data, pieces, (r, c, h, w), components))
        except _Invalid:
            continue
    _require(len(candidates) == 1, "A unique framed network panel is required")
    return candidates[0]


def _paint(board, data, path, paint):
    result = board.copy()
    for node in path[0][1:-1]:
        for cell in data['stations'][node]:
            result[cell] = data['roles'][3]
    for cell, color in paint.items():
        result[cell] = color
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the rule's answer solely from this input; never mutate it."""
    inp = _grid(input_grid)
    board, data, pieces, _, _ = _read_scene(inp)
    # Proposal searches are bounded, but the solver's exact finite search must
    # not reject a valid family member merely for being expensive to assemble.
    path, paint, _, _ = _select(data, pieces, node_limit=None)
    return _paint(board, data, path, paint)


def _render(size, rows, cols, edges, visible, widths, endpoints):
    """Render a connected lattice subgraph, suppressing some degree-two nodes."""
    blocks = {v: {(rows[v[0]] - size // 2 + dr, cols[v[1]] - size // 2 + dc)
                  for dr in range(size) for dc in range(size)} for v in visible}
    wire = set()
    for (a, b), width in zip(edges, widths):
        ar, ac, br, bc = rows[a[0]], cols[a[1]], rows[b[0]], cols[b[1]]
        if ar == br:
            wire.update((r, c) for r in range(ar, ar + width)
                        for c in range(min(ac, bc), max(ac, bc) + 1))
        else:
            wire.update((r, c) for r in range(min(ar, br), max(ar, br) + 1)
                        for c in range(ac, ac + width))
    wire -= set().union(*blocks.values())
    foreground = wire | set().union(*blocks.values())
    h = max(r for r, _ in foreground) + 2
    w = max(c for _, c in foreground) + 2
    _require(max(h, w) <= 24, "Panel too large to scatter its inventory")
    board = np.ones((h, w), dtype=np.int8)
    for p in wire:
        board[p] = 0
    for v, block in blocks.items():
        for p in block:
            board[p] = 3 if v in endpoints else 2
    data = _network(board, 0, 1, 2, 3)
    # Verify topology against the actual rendered pixels, including suppression.
    raw_adj = {v: [] for v in set(v for e in edges for v in e)}
    for a, b in edges:
        raw_adj[a].append(b)
        raw_adj[b].append(a)
    order = sorted(visible, key=lambda v: (rows[v[0]], cols[v[1]]))
    ids = {v: i for i, v in enumerate(order)}
    expected = set()
    for a in visible:
        for b in raw_adj[a]:
            previous, current = a, b
            while current not in visible:
                choices = [v for v in raw_adj[current] if v != previous]
                _require(len(choices) == 1, "Only degree-two stations can be suppressed")
                previous, current = current, choices[0]
            _require(a != current, "Suppressed self-loop")
            expected.add(tuple(sorted((ids[a], ids[current]))))
    _require(set(data['edges']) == expected and 2 * len(expected) == sum(len(raw_adj[v]) for v in visible),
             "Rendered sockets changed the construction graph")
    return board, data


def _circuit(rng, kind):
    """A procedural teaching witness: long thin arc and short thick arc.

    With lower height b and horizontal span G, upper height 2b+G/2-9
    makes both arcs have 4b+2G-24 socket cells. One intermediate station
    on the upper arc beats three on the lower arc, despite greater distance.
    """
    options = [(b, l, r) for b in range(4, 7) for l in range(5, 9) for r in range(5, 9)
               if (l + r) % 2 == 0 and b < 2 * b + (l + r) // 2 - 9
               and 3 * b + (l + r) // 2 - 9 <= 18
               and (kind != 'B' or (l % 2 == 1 and r % 2 == 1))]
    b, left, right = options[int(rng.integers(len(options)))]
    top = 2 * b + (left + right) // 2 - 9
    rows, cols = [2, 2 + top, 2 + top + b], [2, 2 + left, 2 + left + right]
    edges = [((0, 0), (0, 1)), ((0, 1), (0, 2)), ((0, 0), (1, 0)),
             ((0, 2), (1, 2)), ((1, 0), (2, 0)), ((1, 2), (2, 2)),
             ((2, 0), (2, 1)), ((2, 1), (2, 2))]
    widths = [1] * 4 + [2] * 4
    visible = {(0, 1), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)}
    if kind == 'combined':
        # The central, zero-intermediate route lacks enough socket cells.
        edges.extend([((1, 0), (1, 1)), ((1, 1), (1, 2))])
        widths.extend([1, 1])
    elif rng.random() < .75:
        visible.add((1, 1))
        edges.append(((0, 1), (1, 1)) if kind == 'A' else ((1, 1), (2, 1)))
        widths.append(1)
    return _render(3, rows, cols, edges, visible, widths, {(1, 0), (1, 2)})


def _random_network(rng):
    size = int(rng.choice([2, 3, 3, 4]))
    nr, nc = map(int, rng.integers(2, 5, size=2))
    rows = list(map(int, np.r_[0, np.cumsum(rng.integers(size + 2, size + 6, nr - 1))] + size // 2 + 1))
    cols = list(map(int, np.r_[0, np.cumsum(rng.integers(size + 2, size + 6, nc - 1))] + size // 2 + 1))
    vertices = [(r, c) for r in range(nr) for c in range(nc)]
    possible = [(a, b) for a, b in combinations(vertices, 2)
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1]
    active = {vertices[int(rng.integers(len(vertices)))]}
    edges = []
    goal = int(rng.integers(max(5, len(vertices) - 3), len(vertices) + 1)) if len(vertices) >= 5 else 4
    while len(active) < goal:
        choices = [e for e in possible if (e[0] in active) != (e[1] in active)]
        edge = choices[int(rng.integers(len(choices)))]
        edges.append(edge)
        active.update(edge)
    chance = float(rng.uniform(.35, .9))
    edges += [e for e in possible if e not in edges and e[0] in active and e[1] in active and rng.random() < chance]
    degrees = Counter(v for e in edges for v in e)
    visible = {v for v in active if degrees[v] != 2 or rng.random() > .25}
    _require(len(visible) >= 5, "Not enough visible stations")
    widths = [int(rng.choice([1, 1, 2])) if size >= 3 else 1 for _ in edges]
    endpoints = set(rng.choice(len(visible), 2, replace=False).tolist())
    ordered = sorted(visible)
    return _render(size, rows, cols, edges, visible, widths, {ordered[i] for i in endpoints})


def _partition(rng, holes, domino=False, whole=False):
    if whole:
        return [_normalize(c) for c in _components(holes)]
    remaining, pieces = set(holes), []
    while remaining:
        start = min(remaining) if domino else sorted(remaining)[int(rng.integers(len(remaining)))]
        cells = {start}
        remaining.remove(start)
        target = 2 if domino else int(rng.integers(2, 8))
        while len(cells) < target:
            frontier = sorted({(r + dr, c + dc) for r, c in cells for dr, dc in _N4} & remaining)
            if not frontier:
                break
            p = frontier[int(rng.integers(len(frontier)))]
            cells.add(p)
            remaining.remove(p)
        pieces.append(_normalize(cells))
    return pieces


def _color_pieces(rng, shapes, palette, monochrome=False):
    signatures = sorted(set(shapes))
    count = 1 if monochrome else int(rng.integers(1, min(3, len(signatures)) + 1))
    colors = list(map(int, palette[4:4 + count]))
    assignment = {sh: colors[i % count] for i, sh in enumerate(signatures)}
    return [(assignment[sh], sh) for sh in shapes]


def _witnesses(data, pieces, path, routes, feasible):
    selected = len(path[0])
    area = sum(len(sh) for _, sh in pieces)
    cost = sum(data['lengths'][e] for e in path[1])
    a = any(len(p[0]) > selected for p in feasible)
    physical = any(len(p[0]) > selected and sum(data['lengths'][e] for e in p[1]) < cost for p in feasible)
    b = any(len(p[0]) < selected for p in routes)
    shape = any(len(p[0]) < selected and len(_holes(data, p)) == area and p not in feasible for p in routes)
    orientation = False
    dot, h2, v2 = ((0, 0),), ((0, 0), (0, 1)), ((0, 0), (1, 0))
    shapes = {sh for _, sh in pieces}
    # A visible sufficient witness: all supplied dominoes have one orientation,
    # and a rejected equal-area route would fit if some were allowed to turn.
    if shape and shapes <= {dot, h2, v2} and not {h2, v2} <= shapes and len({c for c, _ in pieces}) == 1:
        color = pieces[0][0]
        dots = sum(sh == dot for _, sh in pieces)
        dominoes = len(pieces) - dots
        for p in routes:
            if len(p[0]) >= selected or len(_holes(data, p)) != area or p in feasible:
                continue
            for vertical in range(dominoes + 1):
                rotated = ([(color, dot)] * dots + [(color, h2)] * (dominoes - vertical)
                           + [(color, v2)] * vertical)
                if _tile(_holes(data, p), rotated, existence_only=True):
                    orientation = True
                    break
            if orientation:
                break
    return dict(A=a, physical=physical, B=b, shape=shape, orientation=orientation)


def _scatter(rng, board, pieces, background):
    h, w = board.shape
    for attempt in range(20):
        ch = int(rng.integers(h + 4, 31)) if attempt < 12 else 30
        cw = int(rng.integers(w + 4, 31)) if attempt < 12 else 30
        r, c = int(rng.integers(1, ch - h)), int(rng.integers(1, cw - w))
        if attempt >= 12:
            r, c = (1 if rng.random() < .5 else ch - h - 1), (1 if rng.random() < .5 else cw - w - 1)
        inp = np.full((ch, cw), background, dtype=np.int8)
        inp[r:r + h, c:c + w] = board
        blocked = np.zeros(inp.shape, dtype=bool)
        blocked[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1] = True
        blocked[0, :] = blocked[-1, :] = True
        blocked[:, 0] = blocked[:, -1] = True
        order = sorted(range(len(pieces)), key=lambda i: (-len(pieces[i][1]), i))
        for i in order:
            color, shape = pieces[i]
            ph = max(dr for dr, _ in shape) + 1
            pw = max(dc for _, dc in shape) + 1
            candidates = [(rr, cc) for rr in range(1, ch - ph) for cc in range(1, cw - pw)
                          if all(not blocked[rr + dr, cc + dc] for dr, dc in shape)]
            if not candidates:
                break
            rr, cc = candidates[int(rng.integers(len(candidates)))]
            for dr, dc in shape:
                p = rr + dr, cc + dc
                inp[p] = color
                blocked[p] = True
                for ar, ac in _N4:
                    blocked[p[0] + ar, p[1] + ac] = True
        else:
            return inp
    raise _Invalid("Could not scatter the pieces with clear separation")


def _materialize(rng, board, pieces, palette):
    colored = np.asarray(palette[:4], dtype=np.int8)[board]
    inp = _scatter(rng, colored, pieces, int(palette[0]))
    # Whole-scene transformations also transform each loose piece's orientation.
    turns = int(rng.integers(4))
    inp = np.rot90(inp, turns).copy()
    if rng.random() < .5:
        inp = inp[:, ::-1].copy()
    return {'input': inp, 'output': solve(inp)}


def _teaching_pair(rng, kind, palette):
    for _ in range(80):
        try:
            board, data = _circuit(rng, kind)
            routes = _paths(data)
            target = routes[0] if kind != 'B' else routes[-1]
            if kind == 'combined':
                target = routes[1]
            holes = _holes(data, target)
            if kind == 'B':
                # Odd-span choices make the lower horizontal socket lengths
                # even. Horizontal dominoes tile all thick sockets. At most two
                # singletons allow the thin, bend-shaped alternative to tile
                # only when domino rotations are additionally permitted.
                thin = _holes(data, routes[0])
                dots = sum(len(component) % 2 for component in _components(thin))
                shapes = [((0, 0),)] * dots + [((0, 0), (0, 1))] * ((len(holes) - dots) // 2)
            else:
                shapes = _partition(rng, holes, domino=True)
            _require(2 <= len(shapes) <= 40, "Bad piece count")
            pieces = _color_pieces(rng, shapes, palette, monochrome=True)
            path, _, paths, viable = _select(data, pieces, collect=True)
            f = _witnesses(data, pieces, path, paths, viable)
            needed = ('A', 'physical') if kind == 'A' else ('B', 'shape', 'orientation') if kind == 'B' else ('A', 'B', 'physical')
            _require(all(f[k] for k in needed), "Teaching witness lost its distinction")
            return _materialize(rng, board, pieces, palette)
        except _Invalid:
            continue
    raise _Invalid("Bounded teaching construction exhausted")


def _general_pair(rng, palette, combined=False):
    for _ in range(90):
        try:
            board, data = _random_network(rng)
            routes = _paths(data)
            if combined:
                _require(len(routes) >= 3, "A combined query needs three alternatives")
                groups = {}
                for route in routes:
                    groups.setdefault(len(_holes(data, route)), []).append(route)
                targets = [ps[0] for area, ps in groups.items() if 4 <= area <= 50 and len(ps) >= 2
                           and len(ps[0][0]) >= 3 and len(ps[0][0]) < len(ps[1][0])
                           and len(routes[0][0]) < len(ps[0][0])]
            else:
                targets = [p for p in routes if 3 <= len(p[0]) <= 10 and 4 <= len(_holes(data, p)) <= 60]
            rng.shuffle(targets)
            for target in targets[:8]:
                shapes = _partition(rng, _holes(data, target), domino=combined and rng.random() < .6,
                                    whole=not combined and rng.random() < .25)
                if not 2 <= len(shapes) <= 35:
                    continue
                pieces = _color_pieces(rng, shapes, palette, monochrome=rng.random() < (.8 if combined else .55))
                try:
                    path, _, paths, viable = _select(data, pieces, collect=True)
                    f = _witnesses(data, pieces, path, paths, viable)
                    _require(len(path[0]) >= 3, "Positive station witness needed")
                    if combined:
                        _require(f['A'] and f['B'], "Query does not combine both distinctions")
                    return _materialize(rng, board, pieces, palette)
                except _Invalid:
                    continue
        except _Invalid:
            continue
    # A bounded, procedural construction preserves the same query invariants.
    return _teaching_pair(rng, 'combined' if combined else 'B', palette)


def _count(value, name, lo, hi):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= int(value) <= hi:
        raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _count(num_train, 'num_train', 2, 10)
    num_test = _count(num_test, 'num_test', 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    common_palette = rng.permutation(10)
    consistent = bool(rng.random() < .25)
    witness_order = rng.permutation(num_train)
    kinds = {int(witness_order[0]): 'A', int(witness_order[1]): 'B'}
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            palette = common_palette if consistent else rng.permutation(10)
            for _ in range(12):
                pair = (_teaching_pair(rng, kinds[i], palette) if split == 'train' and i in kinds
                        else _general_pair(rng, palette, combined=split == 'test'))
                if _key(pair['input']) not in seen:
                    seen.add(_key(pair['input']))
                    puzzle[split].append(pair)
                    break
            else:
                raise _Invalid("Distinct-example retries exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    inp, output = _grid(pair['input']), _grid(pair['output'])
    board, data, pieces, bbox, exterior = _read_scene(inp)
    path, paint, routes, feasible = _select(data, pieces, collect=True)
    expected = _paint(board, data, path, paint)
    _require(np.array_equal(output, expected), "Wrong route, assembly, highlight, or crop")
    chosen_holes = _holes(data, path)
    _require(set(paint) == set(chosen_holes), "Selected sockets were not filled exactly")
    _require(Counter(paint.values()) == Counter(color for color, shape in pieces for _ in shape),
             "Piece color area was not conserved")
    untouched = np.ones(board.shape, dtype=bool)
    for cell in chosen_holes:
        untouched[cell] = False
    for node in path[0][1:-1]:
        for cell in data['stations'][node]:
            untouched[cell] = False
    _require(np.array_equal(output[untouched], board[untouched]), "Off-route cells changed")
    for node in data['endpoints']:
        _require(all(output[p] == data['roles'][3] for p in data['stations'][node]), "Endpoint changed")
    unused_at_junction = any(len(data['adjacency'][node]) >= 3 and any(e not in path[1] for _, e in data['adjacency'][node])
                             for node in path[0][1:-1])
    signatures = {sh for _, sh in pieces}
    features = dict(_witnesses(data, pieces, path, routes, feasible),
                    board_shape=board.shape, canvas_shape=inp.shape, roles=data['roles'],
                    stations=len(data['stations']), station_size=data['size'],
                    edges=len(data['edges']), loops=len(data['edges']) - len(data['stations']) + 1,
                    selected=len(path[0]) - 2, candidate_routes=len(routes), feasible_routes=len(feasible),
                    pieces=len(pieces), area=len(chosen_holes), piece_shapes=signatures,
                    payload_colors=len({c for c, _ in pieces}), unused_junction=unused_at_junction,
                    bent_sockets=sum(len(_normalize(m)) != (max(r for r, _ in _normalize(m)) + 1)
                                     * (max(c for _, c in _normalize(m)) + 1) for m in data['masks']),
                    degrees=tuple(sorted(len(a) for a in data['adjacency'])),
                    layout=_key(board), bbox=bbox)
    return features


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "Invalid puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "Splits must be lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "Invalid example counts")
    seen, records = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "Invalid pair keys")
            inp = _grid(pair['input'])
            _require(_key(inp) not in seen, "Repeated input")
            seen.add(_key(inp))
            records[split].append(_analyze_pair(pair))
    _require(any(f['A'] and f['physical'] for f in records['train']), "Training lacks a station-count witness")
    _require(any(f['B'] and f['shape'] and f['orientation'] for f in records['train']),
             "Training lacks a shape- and orientation-sensitive inventory witness")
    _require(all(f['A'] and f['B'] and f['candidate_routes'] >= 3 for f in records['test']),
             "Tests must recombine feasibility and station minimization")
    return True


def _equal(p, q):
    return all(len(p[s]) == len(q[s]) and all(np.array_equal(a[k], b[k]) for a, b in zip(p[s], q[s])
                                           for k in ('input', 'output')) for s in ('train', 'test'))


def self_test():
    """Bounded semantic, API, corruption, equivariance and 42-seed checks."""
    shapes, layouts, signatures, counts, sizes, degrees, palettes = set(), set(), set(), set(), set(), set(), set()
    metrics = []
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        if seed < 3:
            _require(_equal(puzzle, generate_puzzle(seed)), "Nondeterministic generation")
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _analyze_pair(pair)
                metrics.append((split, f))
                shapes.add(f['board_shape'])
                counts.add(f['stations'])
                sizes.add(f['station_size'])
                degrees.update(f['degrees'])
                palettes.add(f['roles'])
                signatures.update(f['piece_shapes'])
                inp = _grid(pair['input'])
                board, data, _, _, _ = _read_scene(inp)
                semantic = np.zeros(board.shape, dtype=np.int8)
                for i, c in enumerate(data['roles']):
                    semantic[board == c] = i
                layouts.add(_key(semantic))
                before = inp.copy()
                _require(np.array_equal(solve(inp), pair['output']), "Input-only solution failed")
                _require(np.array_equal(inp, before), "Solver mutated the input")
    _require(len(layouts) >= 150 and len(shapes) >= 25 and len(signatures) >= 35, "Structural variation collapsed")
    _require(len(counts) >= 7 and {2, 3, 4} <= sizes and {1, 2, 3, 4} <= degrees, "Network variation collapsed")
    _require(len(palettes) >= 50, "Palette roles do not vary")
    _require(any(f['payload_colors'] >= 2 for _, f in metrics), "Missing colored assemblies")
    _require(any(f['unused_junction'] for _, f in metrics), "Missing preservation at selected junctions")
    _require(any(f['loops'] >= 3 and f['bent_sockets'] for _, f in metrics), "Missing combined network geometry")
    # Counts: exercise every legal explicit combination, and reject type errors.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            puzzle = generate_puzzle(400 + 10 * nt + ne, nt, ne)
            _require(len(puzzle['train']) == nt and len(puzzle['test']) == ne, "Counts ignored")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, False), (4.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal counts accepted")
    # Assembly ambiguity is about the final colors, not identities of copies.
    holes = {(0, 0), (0, 2)}
    dot = ((0, 0),)
    _require(len(_tile(holes, [(5, dot), (6, dot)])) == 2, "Color ambiguity missed")
    _require(len(_tile(holes, [(5, dot), (5, dot)])) == 1, "Identical copies treated as ambiguity")
    _require(not _tile({(0, 0), (1, 0)}, [(5, ((0, 0), (0, 1)))]), "Illegal piece rotation")
    diamond = dict(endpoints=[0, 3], adjacency=[[(1, 0), (2, 1)], [(0, 0), (3, 2)],
                                              [(0, 1), (3, 3)], [(1, 2), (2, 3)]],
                   masks=[((r, 0), (r, 1)) for r in range(4)])
    try:
        _select(diamond, [(5, ((0, 0), (0, 1)))] * 2)
    except _Invalid:
        pass
    else:
        raise AssertionError("Tied minimum feasible routes were accepted")
    first = generate_puzzle(1000, 2, 1)
    for pair in first['train'] + first['test']:
        for turns in (1, 2, 3):
            _require(np.array_equal(solve(np.rot90(pair['input'], turns)), np.rot90(pair['output'], turns)),
                     "Whole-scene rotation changed the rule")
        _require(np.array_equal(solve(pair['input'][:, ::-1]), pair['output'][:, ::-1]), "Reflection failed")
        recolor = np.array([8, 6, 9, 1, 4, 0, 2, 7, 5, 3], dtype=np.int8)
        _require(np.array_equal(solve(recolor[pair['input']]), recolor[pair['output']]), "Palette-role recovery failed")
    for kind in ('output', 'missing'):
        bad = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in first[s]] for s in first}
        pair = bad['test'][0]
        if kind == 'output':
            pair['output'][0, 0] = (int(pair['output'][0, 0]) + 1) % 10
        else:
            _, data, _, _, exterior = _read_scene(pair['input'])
            pair['input'][exterior[0][0]] = data['roles'][0]
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("Corruption was accepted")
    return dict(seeds=42, pairs=len(metrics), board_shapes=len(shapes),
                semantic_panel_layouts=len(layouts), oriented_piece_shapes=len(signatures),
                station_counts=sorted(counts), station_sizes=sorted(sizes), palette_assignments=len(palettes),
                multiple_payload_colors=sum(f['payload_colors'] > 1 for _, f in metrics),
                selected_unused_junctions=sum(f['unused_junction'] for _, f in metrics),
                combined_tests=sum(s == 'test' and f['A'] and f['B'] for s, f in metrics))


if __name__ == '__main__':
    print(self_test())
