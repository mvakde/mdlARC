"""A moving token stitches scattered circuit tiles into an anchored rectangle.

Only standard-library modules and NumPy are used. No source fixtures, generation
metadata, cached answers, or training examples are consulted by solve().
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name, low=None, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if low is not None and value < low or high is not None and value > high:
        raise ValueError(f"{name} outside its allowed range")
    return value


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "grid colors")
    return g.astype(np.int8, copy=True)


def _add(p, d):
    return p[0] + d[0], p[1] + d[1]


def _inside(shape, p):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1]


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        cells = [p]
        for p in cells:
            for d in DIRS:
                q = _add(p, d)
                if q in unseen:
                    unseen.remove(q)
                    cells.append(q)
        result.append(cells)
    return result


def _edge_cells(shape, side):
    h, w = shape
    if side == 0:
        return [(0, x) for x in range(1, w - 1)]
    if side == 1:
        return [(y, w - 1) for y in range(1, h - 1)]
    if side == 2:
        return [(h - 1, x) for x in range(1, w - 1)]
    return [(y, 0) for y in range(1, h - 1)]


def _flanks(p, direction):
    normal = (-direction[1], direction[0])
    return _add(p, normal), _add(p, (-normal[0], -normal[1]))


def _pair_color(m, p, direction, carrier, guides):
    a, b = _flanks(p, direction)
    if not (_inside(m.shape, a) and _inside(m.shape, b)):
        return None
    x, y = int(m[a]), int(m[b])
    return x if x == y and x in guides else None


def _extract(inp):
    """Roles are visible: most frequent background, 3-cell L, solid tile corners,
    and the unique remaining color with exactly two cells (the two tokens).
    """
    g = _grid(inp)
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(sum(counts == counts[bg]) == 1, "background frequency must be unique")
    markers = []
    for color in range(10):
        if color == bg or counts[color] != 3:
            continue
        cells = set(map(tuple, np.argwhere(g == color).tolist()))
        y0, x0 = min(y for y, x in cells), min(x for y, x in cells)
        square = {(y0, x0), (y0 + 1, x0), (y0, x0 + 1), (y0 + 1, x0 + 1)}
        if cells < square:
            point = (square - cells).pop()
            if int(g[point]) not in (bg, color):
                markers.append((color, cells, point, (point[0] == y0, point[1] == x0)))
    _require(len(markers) == 1, "one external L marker required")
    marker, marker_cells, point, bottom_right = markers[0]
    pieces = []
    anchor = None
    for cells in _components((g != bg) & (g != marker)):
        q = np.asarray(cells)
        r, c = map(int, q.min(0))
        h, w = map(int, q.max(0) - q.min(0) + 1)
        _require(len(cells) == h * w and h >= 5 and w >= 5, "solid rectangular tiles")
        if point in cells:
            anchor = len(pieces)
        pieces.append((g[r:r + h, c:c + w].copy(), (r, c)))
    _require(3 <= len(pieces) <= 6 and anchor is not None, "tile count or anchor")
    corners = {int(m[p]) for m, _ in pieces for p in
               ((0, 0), (0, m.shape[1] - 1), (m.shape[0] - 1, 0),
                (m.shape[0] - 1, m.shape[1] - 1))}
    _require(len(corners) == 1, "tile corners identify one carrier color")
    carrier = corners.pop()
    tokens = [c for c in range(10) if c not in (bg, marker, carrier) and counts[c] == 2]
    _require(len(tokens) == 1, "one two-cell token role required")
    token = tokens[0]
    guides = set(map(int, np.unique(g))) - {bg, marker, carrier, token}
    _require(2 <= len(guides) <= 5 and all(counts[c] >= 4 for c in guides), "guide palette")
    ar, ac = pieces[anchor][1]
    ah, aw = pieces[anchor][0].shape
    bottom, right = bottom_right
    _require(point == (ar + (ah - 1 if bottom else 0), ac + (aw - 1 if right else 0)),
             "marker must pin the corresponding tile corner")
    active, inactive = [], []
    for i, (m, old) in enumerate(pieces):
        for p in map(tuple, np.argwhere(m == token).tolist()):
            sides = [s for s in range(4) if p in _edge_cells(m.shape, s)]
            _require(len(sides) == 1, "tokens occupy noncorner boundary ports")
            side = sides[0]
            d = tuple(-x for x in DIRS[side])
            a, b = _flanks(p, d)
            _require(int(m[a]) in guides and int(m[b]) in guides, "token flanks must be rails")
            if _pair_color(m, p, d, carrier, guides) is None:
                inactive.append((i, p, d))
            else:
                active.append((i, p, d))
    _require(len(active) == len(inactive) == 1, "one matched and one mismatched token")
    _require(active[0][0] == anchor, "active token must be on the anchor tile")
    return dict(grid=g, bg=bg, marker=marker, marker_cells=marker_cells,
                point=point, bottom=bottom, right=right, pieces=pieces,
                anchor=anchor, carrier=carrier, token=token, guides=guides,
                active=active[0], inactive=inactive[0])


def _trace_tile(m, start, direction, carrier, guides, token):
    """Directional rail motion, confined to a tile until it reaches a cut.
    Continue straight; a blocker permits only a uniquely paired orthogonal turn.
    """
    p = start
    seen = set()
    path = [p]
    turns = 0
    while True:
        state = p, direction
        _require(state not in seen, "route cycle")
        seen.add(state)
        q = _add(p, direction)
        if not _inside(m.shape, q):
            color = _pair_color(m, p, direction, carrier, guides)
            _require(color is not None and int(m[p]) == carrier, "exit must be a matched port")
            return dict(end=p, direction=direction, color=color, path=path, turns=turns)
        if int(m[q]) == carrier:
            p = q
            path.append(p)
            continue
        choices = []
        for d in ((-direction[1], direction[0]), (direction[1], -direction[0])):
            q = _add(p, d)
            if (_inside(m.shape, q) and int(m[q]) == carrier
                    and _pair_color(m, q, d, carrier, guides) is not None):
                choices.append(d)
        _require(len(choices) <= 1, "ambiguous turn")
        if not choices:
            _require(p != start and int(m[p]) == carrier, "empty route")
            return dict(end=p, direction=direction, color=None, path=path, turns=turns)
        direction = choices[0]
        turns += 1


def _analyse(inp):
    a = _extract(inp)
    pieces, carrier, guides = a['pieces'], a['carrier'], a['guides']
    current, entry, direction = a['active']
    positions = {current: pieces[current][1]}
    route_order, routes = [], []
    while True:
        _require(current not in route_order, "tile revisited")
        route_order.append(current)
        m = pieces[current][0]
        route = _trace_tile(m, entry, direction, carrier, guides, a['token'])
        routes.append(route)
        if route['color'] is None:
            break
        direction = route['direction']
        side = (DIRS.index(direction) + 2) % 4
        candidates = []
        for i, (other, old) in enumerate(pieces):
            if i in positions:
                continue
            for p in _edge_cells(other.shape, side):
                if (int(other[p]) == carrier and
                        _pair_color(other, p, direction, carrier, guides) == route['color']):
                    candidates.append((i, p))
        _require(len(candidates) == 1, "one facing matched port on an unused tile required")
        other, port = candidates[0]
        r, c = positions[current]
        ey, ex = _add(route['end'], direction)
        positions[other] = (r + ey - port[0], c + ex - port[1])
        current, entry = other, port
    _require(len(positions) == len(pieces), "route must stitch every tile")
    assembled = np.full_like(a['grid'], a['bg'])
    occupied = np.zeros(assembled.shape, dtype=bool)
    for i, (m, old) in enumerate(pieces):
        r, c = positions[i]
        h, w = m.shape
        _require(r >= 0 and c >= 0 and r + h <= len(assembled) and c + w <= assembled.shape[1],
                 "assembly outside canvas")
        _require(not occupied[r:r + h, c:c + w].any(), "tile overlap")
        assembled[r:r + h, c:c + w] = m
        occupied[r:r + h, c:c + w] = True
    q = np.argwhere(occupied)
    y, x = map(int, q.min(0))
    z, w = map(int, q.max(0) + 1)
    _require(occupied[y:z, x:w].all(), "stitched tiles must close into one rectangle")
    _require(a['point'] == (z - 1 if a['bottom'] else y, w - 1 if a['right'] else x),
             "anchor corner must also be the rectangle corner")
    _require(all(not occupied[p] for p in a['marker_cells']), "marker lies outside assembly")
    start_tile, local_start, _ = a['active']
    start = _add(positions[start_tile], local_start)
    destination = _add(positions[current], routes[-1]['end'])
    out = assembled.copy()
    out[start] = carrier
    out[destination] = a['token']
    a.update(positions=positions, order=route_order, routes=routes, assembled=assembled,
             occupied=occupied, start=start, destination=destination, output=out,
             target_shape=(z - y, w - x), total_turns=sum(r['turns'] for r in routes))
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles, stitch tiles as the token travels, and return the full canvas."""
    return _analyse(input_grid)['output'].copy()


def _polyline(points):
    result = [tuple(points[0])]
    for target in points[1:]:
        p = result[-1]
        if p == target:
            continue
        _require((p[0] == target[0]) != (p[1] == target[1]), "orthogonal polyline")
        d = (int(np.sign(target[0] - p[0])), int(np.sign(target[1] - p[1])))
        while p != target:
            p = _add(p, d)
            result.append(p)
    _require(len(set(result)) == len(result), "simple route required")
    centers = set(result)
    for i, p in enumerate(result):
        degree = sum(_add(p, d) in centers for d in DIRS)
        _require(degree == (1 if i in (0, len(result) - 1) else 2), "self-touching route")
    return result


def _join_path(shape, first, last, rng):
    """Connect boundary ports, preserving an inward first and outward last step."""
    s, p, _ = first
    t, q, _ = last
    h, w = shape
    if (s - t) % 2 == 1:
        horizontal = p if s in (1, 3) else q
        vertical = p if s in (0, 2) else q
        return _polyline([p, (horizontal[0], vertical[1]), q])
    _require(s != t, "two ports cannot share a side in the sampler")
    if s in (1, 3):
        if p[0] == q[0]:
            return _polyline([p, q])
        bend = int(rng.integers(2, w - 2))
        return _polyline([p, (p[0], bend), (q[0], bend), q])
    if p[1] == q[1]:
        return _polyline([p, q])
    bend = int(rng.integers(2, h - 2))
    return _polyline([p, (bend, p[1]), (bend, q[1]), q])


def _port(shape, side, offset, color):
    h, w = shape
    p = (0, offset) if side == 0 else (offset, w - 1) if side == 1 else (
        (h - 1, offset) if side == 2 else (offset, 0))
    return side, p, color


def _contact(first, second):
    """Side and absolute shared segment for two abutting rectangles."""
    r, c, h, w = first
    y, x, z, v = second
    if c + w == x or x + v == c:
        lo, hi = max(r, y), min(r + h, y + z)
        if hi - lo >= 5:
            return (1 if c + w == x else 3), lo, hi
    if r + h == y or y + z == r:
        lo, hi = max(c, x), min(c + w, x + v)
        if hi - lo >= 5:
            return (2 if r + h == y else 0), lo, hi
    return None


def _hamiltonian(rng, rectangles):
    start = next(i for i, (r, c, h, w) in enumerate(rectangles) if r == c == 0)
    candidates = []
    def visit(path):
        if len(path) == len(rectangles):
            candidates.append(path)
            return
        for q in range(len(rectangles)):
            if q not in path and _contact(rectangles[path[-1]], rectangles[q]) is not None:
                visit(path + [q])
    visit([start])
    _require(bool(candidates), "partition needs a route through all tiles")
    return candidates[int(rng.integers(len(candidates)))]


def _walls(shape, path, carrier, core, first, last):
    m = np.full(shape, carrier, dtype=np.int8)
    centers = set(path)
    for p in path:
        for d in DIRS:
            q = _add(p, d)
            if _inside(shape, q) and q not in centers:
                m[q] = core
    for port in (first, last):
        if port is not None:
            side, p, color = port
            for q in _flanks(p, DIRS[side]):
                m[q] = color
    return m


def _marker_at(point):
    r, c = point
    return [(r - 1, c - 1), (r - 1, c), (r, c - 1)]


def _gap(a, b):
    r, c, h, w = a
    y, x, z, v = b
    return r + h < y or y + z < r or c + w < x or x + v < c


def _scatter(rng, matrices, anchor_position, marker_cells, canvas):
    """Bounded randomized packing; no file output or retry payload retention."""
    indices = sorted(range(1, len(matrices)), key=lambda i: -matrices[i].size)
    for attempt in range(24):
        positions = [anchor_position] + [None] * (len(matrices) - 1)
        rects = [(*anchor_position, *matrices[0].shape)]
        for i in indices:
            h, w = matrices[i].shape
            options = [(r, c) for r in range(canvas[0] - h + 1)
                       for c in range(canvas[1] - w + 1)]
            found = False
            for k in rng.permutation(len(options)):
                r, c = options[int(k)]
                if any(r - 1 <= y <= r + h and c - 1 <= x <= c + w for y, x in marker_cells):
                    continue
                if all(_gap((r, c, h, w), other) for other in rects):
                    positions[i] = (r, c)
                    rects.append((r, c, h, w))
                    found = True
                    break
            if not found:
                break
        else:
            return positions
    return None


def _make_pair(rng, layout, rotation):
    rows, cols = layout[:2]
    merge = len(layout) == 3
    count = rows * cols - int(merge)
    colors = list(map(int, rng.permutation(10)))
    bg, marker, carrier, token = colors[:4]
    guides = colors[4:4 + count - 1]
    heights = list(map(int, rng.integers(5, 9, size=rows)))
    widths = list(map(int, rng.integers(5, 7 if cols == 4 else 9, size=cols)))
    offsets_y = np.cumsum([0] + heights).tolist()
    offsets_x = np.cumsum([0] + widths).tolist()
    rectangles = [(offsets_y[r], offsets_x[c], heights[r], widths[c])
                  for r in range(rows) for c in range(cols)]
    if merge:
        col = int(rng.integers(cols))
        rectangles = [rect for i, rect in enumerate(rectangles) if i not in (col, cols + col)]
        rectangles.append((0, offsets_x[col], sum(heights), widths[col]))
    order = _hamiltonian(rng, rectangles)
    rects = [rectangles[i] for i in order]
    shapes = [(h, w) for r, c, h, w in rects]
    incoming, outgoing = [None] * count, [None] * count
    seam_specs = []
    for i, (p, q) in enumerate(zip(rects, rects[1:])):
        side, lo, hi = _contact(p, q)
        along = int(rng.choice(list(range(lo + 2, hi - 2, 2))))
        axis = 1 if side in (0, 2) else 0
        outgoing[i] = _port(shapes[i], side, along - p[axis], guides[i])
        incoming[i + 1] = _port(shapes[i + 1], (side + 2) % 4, along - q[axis], guides[i])
        seam_specs.append((side, guides[i]))
    # The first tile is the upper-left corner; whole-scene rotation later gives
    # all four anchor corners. Individual pieces are never rotated by solve().
    start_side = int(rng.choice([0, 3]))
    length = shapes[0][1] if start_side == 0 else shapes[0][0]
    start_offset = int(rng.choice(list(range(2, length - 2, 2))))
    incoming[0] = _port(shapes[0], start_side, start_offset, int(rng.choice(guides)))
    matrices, paths = [], []
    for i, shape in enumerate(shapes):
        first = incoming[i]
        if outgoing[i] is not None:
            try:
                path = _join_path(shape, first, outgoing[i], rng)
            except ValueError:
                return None
        else:
            side, p, _ = first
            d = tuple(-x for x in DIRS[side])
            depth = shape[0] if side in (0, 2) else shape[1]
            distance = int(rng.integers(2, depth - 2))
            end = (p[0] + distance * d[0], p[1] + distance * d[1])
            path = _polyline([p, end])
        m = _walls(shape, path, carrier, int(rng.choice(guides)), first, outgoing[i])
        matrices.append(m)
        paths.append(path)
    matrices[0][incoming[0][1]] = token
    # Add two false ports with unequal rails. One carries the second, inactive
    # token; the other shares one flank identity with an actual docking seam.
    # They are not decorative: an equality-free port reader sees extra options.
    decoys = []
    for i, m in enumerate(matrices):
        reserved = set(paths[i])
        for p in paths[i]:
            reserved.update(_add(p, d) for d in DIRS)
        for side in range(4):
            for p in _edge_cells(m.shape, side):
                flank = _flanks(p, DIRS[side])
                cells = (p, *flank)
                corners = {(0, 0), (0, m.shape[1] - 1),
                           (m.shape[0] - 1, 0), (m.shape[0] - 1, m.shape[1] - 1)}
                if not corners.intersection(cells) and all(
                        q not in reserved and int(m[q]) == carrier for q in cells):
                    decoys.append((i, side, p))
    rng.shuffle(decoys)
    # A false docking port must actually compete with a real join while its
    # tile is still unused, rather than merely looking suggestive in isolation.
    competing = [(j, side, p, seam_color) for j, side, p in decoys
                 for i, (out_side, seam_color) in enumerate(seam_specs)
                 if j > i and side == (out_side + 2) % 4]
    if not competing:
        return None
    j, side, p, seam_color = competing[int(rng.integers(len(competing)))]
    flanks = _flanks(p, DIRS[side])
    matrices[j][flanks[0]] = seam_color
    matrices[j][flanks[1]] = int(rng.choice([c for c in guides if c != seam_color]))
    installed = []
    for i, side, p in decoys:
        m = matrices[i]
        flank = _flanks(p, DIRS[side])
        if not all(int(m[q]) == carrier for q in (p, *flank)):
            continue
        a, b = map(int, rng.choice(guides, 2, replace=False))
        m[flank[0]], m[flank[1]] = a, b
        m[p] = token
        installed.append((i, side, p))
        break
    if not installed:
        return None
    h, w = sum(heights), sum(widths)
    # At most 384 payload cells, leaving ample background and scattering space.
    canvas = (int(rng.integers(max(22, h + 3), 31)),
              int(rng.integers(max(22, w + 3), 31)))
    origin = (int(rng.integers(1, canvas[0] - h + 1)),
              int(rng.integers(1, canvas[1] - w + 1)))
    final = [(origin[0] + r, origin[1] + c) for r, c, ph, pw in rects]
    markers = _marker_at(origin)
    positions = _scatter(rng, matrices, final[0], markers, canvas)
    if positions is None or all(p == q for p, q in zip(positions, final)):
        return None
    inp = np.full(canvas, bg, dtype=np.int8)
    out = inp.copy()
    for m, old, new in zip(matrices, positions, final):
        ph, pw = m.shape
        inp[old[0]:old[0] + ph, old[1]:old[1] + pw] = m
        out[new[0]:new[0] + ph, new[1]:new[1] + pw] = m
    for p in markers:
        inp[p] = marker
    out[_add(final[0], incoming[0][1])] = carrier
    out[_add(final[-1], paths[-1][-1])] = token
    inp, out = np.rot90(inp, rotation).copy(), np.rot90(out, rotation).copy()
    # This comparison checks construction against the input-only interpretation;
    # the output above is constructed from tile coordinates, not solve().
    try:
        a = _analyse(inp)
    except ValueError:
        return None
    if not np.array_equal(a['output'], out):
        return None
    return dict(input=inp, output=out)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    ne = _integer(num_test, 'num_test', 1, 4)
    _require(nt > ne, "num_train must exceed num_test")
    rng = np.random.default_rng(seed)
    base_rotation = int(rng.integers(4))
    # A three-tile strip and a turning four-tile example are both
    # present even at minimum counts. Tests combine familiar ports/turns over
    # five/six-tile rectangles, with more joins and potentially internal doglegs.
    plans = [(1, 3), (2, 2)]
    extras = [(1, 3), (1, 4), (2, 2), (2, 3), (2, 3, 'merge')]
    plans.extend(extras[int(rng.integers(len(extras)))] for _ in range(nt - 2))
    rng.shuffle(plans)
    seen = set()
    puzzle = {'train': [], 'test': []}
    test_plans = [((2, 3) if rng.integers(2) else (2, 3, 'merge')) for _ in range(ne)]
    for split, layouts in (('train', plans), ('test', test_plans)):
        for i, layout in enumerate(layouts):
            rotation = (base_rotation + i) % 4
            for attempt in range(80):
                pair = _make_pair(rng, layout, rotation)
                if pair is None:
                    continue
                key = (pair['input'].shape, pair['input'].tobytes())
                if key not in seen:
                    puzzle[split].append(pair)
                    seen.add(key)
                    break
            else:
                raise RuntimeError("bounded circuit construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(a):
    false_options = 0
    for k, (tile, route) in enumerate(zip(a['order'], a['routes'])):
        if route['color'] is None:
            continue
        side = (DIRS.index(route['direction']) + 2) % 4
        for i in a['order'][k + 1:]:
            m = a['pieces'][i][0]
            for p in _edge_cells(m.shape, side):
                x, y = _flanks(p, DIRS[side])
                if int(m[p]) == a['carrier'] and m[x] != m[y]:
                    false_options += int(route['color'] in (int(m[x]), int(m[y]))
                                         and int(m[x]) in a['guides'] and int(m[y]) in a['guides'])
    return dict(tiles=len(a['pieces']), shape=a['grid'].shape, target=a['target_shape'],
                turns=a['total_turns'], corner=(a['bottom'], a['right']),
                roles=(a['bg'], a['marker'], a['carrier'], a['token']),
                palette=len(a['guides']),
                false_docking_options=false_options,
                lengths=tuple(len(r['path']) for r in a['routes']),
                dimensions=tuple(m.shape for m, p in a['pieces']),
                moved_tiles=sum(a['positions'][i] != p for i, (m, p) in enumerate(a['pieces'])))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    nt, ne = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne, "pair counts")
    seen = set()
    training = []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair schema")
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(inp.shape == out.shape, "canvas dimensions must be preserved")
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            a = _analyse(inp)
            _require(np.array_equal(a['output'], out), "incorrect stitched transport")
            _require(not np.any(out == a['marker']), "marker must disappear")
            cin, cout = np.bincount(inp.ravel(), minlength=10), np.bincount(out.ravel(), minlength=10)
            _require(cout[a['bg']] == cin[a['bg']] + 3, "only the marker changes occupancy")
            _require(all(cin[c] == cout[c] for c in range(10) if c not in (a['marker'], a['bg'])),
                     "payload color conservation")
            _require(np.count_nonzero(a['assembled'] != out) == 2, "only active token transfers")
            _require(a['positions'][a['anchor']] == a['pieces'][a['anchor']][1], "anchor moved")
            i, p, _ = a['inactive']
            _require(out[_add(a['positions'][i], p)] == a['token'], "inactive token must stay on its tile")
            _require(_features(a)['moved_tiles'] > 0, "assembly must require translation")
            # A ablation: local routing without joining cannot reach another tile.
            _require(a['order'][0] != a['order'][-1], "route must span multiple tiles")
            # B ablation: assembling the pieces but retaining both source tokens
            # differs at exactly the active source and far terminal.
            _require(not np.array_equal(a['assembled'], out), "motion must affect the answer")
            # Both kinds of pseudo-port must be visible, beyond ordinary ports.
            mismatches = 0
            for m, old in a['pieces']:
                for side in range(4):
                    for p in _edge_cells(m.shape, side):
                        x, y = _flanks(p, DIRS[side])
                        if int(m[p]) in (a['carrier'], a['token']):
                            mismatches += int(int(m[x]) in a['guides'] and int(m[y]) in a['guides']
                                              and m[x] != m[y])
            _require(mismatches >= 2, "mismatched docking and token ports required")
            _require(_features(a)['false_docking_options'] >= 1, "rail equality must affect docking")
            if split == 'train':
                training.append(_features(a))
    _require({3, 4} <= {f['tiles'] for f in training}, "training must show multiple assembly sizes")
    _require(any(f['turns'] >= 2 for f in training), "training must show turning transport")


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            for role in ('input', 'output'):
                g = np.asarray(pair[role], dtype=np.int8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """Bounded 42-seed coverage, determinism, contracts, and mutation checks.

    No puzzles, traces or logs are saved; only compact aggregate data is returned.
    A caller may serialize that aggregate with compact JSON.
    """
    shapes, targets, roles, corners, layouts, fingerprints = set(), set(), set(), set(), set(), set()
    palettes, tiles, turns = Counter(), Counter(), Counter()
    pair_count = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        _require(_fingerprint(puzzle) == _fingerprint(generate_puzzle(seed)), "seed reproducibility")
        fingerprints.add(_fingerprint(puzzle))
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                before = pair['input'].copy()
                a = _analyse(before)
                _require(np.array_equal(solve(before), pair['output']), "input-only solve")
                _require(np.array_equal(before, pair['input']), "solver mutated input")
                f = _features(a)
                shapes.add(f['shape']); targets.add(f['target']); roles.add(f['roles'])
                corners.add(f['corner']); layouts.add((f['target'], f['dimensions'], f['lengths']))
                palettes[f['palette']] += 1; tiles[f['tiles']] += 1; turns[f['turns']] += 1
                pair_count += 1
    _require(len(fingerprints) == 42, "cross-seed distinction")
    _require(len(shapes) >= 30 and len(targets) >= 30 and len(layouts) >= 150, "geometric variation")
    _require(len(roles) >= 100 and len(corners) == 4, "role and orientation variation")
    _require({3, 4, 5, 6} <= set(tiles) and {2, 3, 4, 5} <= set(palettes), "count variation")
    _require(len(turns) >= 4, "turn variation")
    for nt in range(2, 11):
        for ne in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(734, nt, ne)
            _require(len(p['train']) == nt and len(p['test']) == ne, "explicit counts")
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4, False), (3.0, 1), (4, None), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p = generate_puzzle(0)
    for corruption in ('frozen', 'marker', 'moved', 'duplicate'):
        bad = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = bad['test'][0]
        a = _analyse(pair['input'])
        if corruption == 'frozen':
            pair['output'] = a['assembled']
        elif corruption == 'marker':
            pair['output'][next(iter(a['marker_cells']))] = a['marker']
        elif corruption == 'moved':
            pair['output'] = np.roll(pair['output'], 1, axis=0)
        else:
            bad['train'][1] = {k: v.copy() for k, v in bad['train'][0].items()}
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("corruption accepted: " + corruption)
    return dict(seeds=42, pairs=pair_count, distinct_puzzles=len(fingerprints),
                canvas_shapes=len(shapes), target_shapes=len(targets), layouts=len(layouts),
                role_assignments=len(roles), corners=len(corners),
                tiles=dict(sorted(tiles.items())), guide_counts=dict(sorted(palettes.items())),
                turns=dict(sorted(turns.items())), independent_solver=False, blind_solve=False)


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
