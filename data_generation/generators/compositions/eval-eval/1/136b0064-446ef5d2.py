"""Programmed jigsaw: decoded strokes are the connectors used in assembly.

Only standard-library/NumPy code. No parent imports, example lookup, or hidden
puzzle dictionary. Every prediction is reconstructed from the supplied input.
"""
from __future__ import annotations

import itertools
import json
import numpy as np

# Clockwise cardinal directions; triangles have a broad base and a single tip.
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))
GLYPHS = (
    np.array([[0, 1, 0], [1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 1], [0, 1, 1]], dtype=bool),
)
CORNERS = ("TL", "TR", "BR", "BL")


def _integer(value, name, low, high=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    value = int(value)
    if value < low or (high is not None and value > high):
        raise ValueError(name + " is out of range")
    return value


def _grid(value):
    g = np.asarray(value)
    assert g.ndim == 2 and all(1 <= n <= 30 for n in g.shape)
    assert np.issubdtype(g.dtype, np.integer)
    assert np.all((0 <= g) & (g <= 9))
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.tobytes()


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for q in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
        result.append(cells)
    return result


def _marker_cells(point, corner):
    r, c = point
    dy = -1 if corner[0] == "T" else 1
    dx = -1 if corner[1] == "L" else 1
    return ((r+dy, c), (r, c+dx), (r+dy, c+dx))


def _edge_point(shape, center, side):
    h, w = shape
    y, x = center
    return ((0, x), (y, w-1), (h-1, x), (y, 0))[side]


def _extract(g):
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    assert np.count_nonzero(counts == counts[bg]) == 1, "background must be unique modal color"
    marks = []
    for color in range(10):
        if color == bg or counts[color] != 3:
            continue
        q = np.argwhere(g == color)
        y, x = q.min(0)
        yy, xx = q.max(0)
        if (yy-y, xx-x) != (1, 1):
            continue
        square = {(int(y), int(x)), (int(y), int(xx)),
                  (int(yy), int(x)), (int(yy), int(xx))}
        missing = square - set(map(tuple, q.tolist()))
        point = missing.pop()
        if int(g[point]) in (bg, color):
            continue
        corner = {(1, 1): "TL", (1, 0): "TR", (0, 0): "BR", (0, 1): "BL"}[
            (point[0]-int(y), point[1]-int(x))]
        marks.append((color, point, corner))
    assert len(marks) == 1, "one external L marker required"
    marker, point, corner = marks[0]
    tiles, anchor = [], None
    for cells in _components((g != bg) & (g != marker)):
        q = np.array(cells)
        top, left = map(int, q.min(0))
        bottom, right = map(int, q.max(0)+1)
        block = g[top:bottom, left:right].copy()
        h, w = block.shape
        assert 5 <= h <= 7 and 5 <= w <= 7
        assert len(cells) == h*w, "tiles are filled rectangles"
        border = np.zeros(block.shape, bool)
        border[[0, -1], :] = True
        border[:, [0, -1]] = True
        carrier = int(np.bincount(block[border], minlength=10).argmax())
        port_positions = np.argwhere(border & (block != carrier))
        assert len(port_positions) == 1, "each tile has one boundary socket"
        entry = tuple(map(int, port_positions[0]))
        port = int(block[entry])
        er, ec = entry
        assert (er in (0, h-1)) != (ec in (0, w-1)), "socket cannot be a corner"
        incoming = 0 if er == 0 else 2 if er == h-1 else 3 if ec == 0 else 1
        ink_positions = np.argwhere((block != carrier) & ~border)
        assert len(ink_positions) == 7
        r, c = map(int, ink_positions.min(0))
        y, x = map(int, ink_positions.max(0)+1)
        assert (y-r, x-c) == (3, 3)
        ink = int(block[tuple(ink_positions[0])])
        assert np.all(block[tuple(ink_positions.T)] == ink)
        assert len({bg, marker, carrier, port, ink}) == 5
        mask = block[r:y, c:x] == ink
        glyph = [k for k, m in enumerate(GLYPHS) if np.array_equal(mask, m)]
        assert len(glyph) == 1, "unknown triangle command"
        direction = glyph[0]
        center = (r+1, c+1)
        assert 2 <= center[0] <= h-3 and 2 <= center[1] <= w-3
        assert entry == _edge_point((h, w), center, incoming)
        assert direction != incoming, "U-turns are excluded"
        expected = np.full(block.shape, carrier, np.int8)
        expected[entry] = port
        expected[r:y, c:x][mask] = ink
        assert np.array_equal(expected, block), "unexpected tile payload"
        if point in cells:
            anchor = len(tiles)
            assert point == (top+(h-1 if corner[0] == "B" else 0),
                             left+(w-1 if corner[1] == "R" else 0))
        tiles.append(dict(shape=(h, w), old=(top, left), block=block,
                          carrier=carrier, port=port, ink=ink, entry=entry,
                          incoming=incoming, direction=direction, center=center,
                          exit=_edge_point((h, w), center, direction)))
    assert 3 <= len(tiles) <= 8 and anchor is not None
    assert len({t['carrier'] for t in tiles}) == len({t['port'] for t in tiles}) == 1
    return dict(background=bg, marker=marker, point=point, corner=corner,
                tiles=tiles, anchor=anchor, canvas=g.shape)


def _stroke_cells(tile):
    """Execute a glyph locally; lengths are supplied by the visible rectangle."""
    a, b, z = tile['entry'], tile['center'], tile['exit']
    result = []
    for p, q in ((a, b), (b, z)):
        dy = int(np.sign(q[0]-p[0]))
        dx = int(np.sign(q[1]-p[1]))
        assert not (dy and dx)
        for k in range(abs(q[0]-p[0])+abs(q[1]-p[1])+1):
            cell = (p[0]+dy*k, p[1]+dx*k)
            if not result or result[-1] != cell:
                result.append(cell)
    assert len(result) == len(set(result))
    return result


def _render(info, positions):
    out = np.full(info['canvas'], info['background'], dtype=np.int8)
    for t, (r, c) in zip(info['tiles'], positions):
        h, w = t['shape']
        out[r:r+h, c:c+w] = t['carrier']
        for y, x in _stroke_cells(t):
            out[r+y, c+x] = t['ink']
    t = info['tiles'][info['anchor']]
    r, c = positions[info['anchor']]
    y, x = t['entry']
    out[r+y, c+x] = t['port']
    return out


def _overlap(a, sa, b, sb):
    return a[0] < b[0]+sb[0] and b[0] < a[0]+sa[0] and \
        a[1] < b[1]+sb[1] and b[1] < a[1]+sa[1]


def _assemble(info):
    """Enumerate socket-connected chains; reject different tied output grids.

    A glyph's exit and the next socket determine an exact translation, so there
    is no arbitrary offset search and no generation metadata is used.
    """
    tiles, anchor = info['tiles'], info['anchor']
    area = sum(h*w for h, w in (t['shape'] for t in tiles))
    positions = [None] * len(tiles)
    positions[anchor] = tiles[anchor]['old']
    answers = {}
    point, corner = info['point'], info['corner']

    def permitted(p, shape):
        r, c = p
        h, w = shape
        if r < 0 or c < 0 or r+h > info['canvas'][0] or c+w > info['canvas'][1]:
            return False
        return ((r >= point[0] if corner[0] == 'T' else r+h-1 <= point[0]) and
                (c >= point[1] if corner[1] == 'L' else c+w-1 <= point[1]))

    def visit(order, remaining):
        if len(answers) > 1:
            return
        i = order[-1]
        t, p = tiles[i], positions[i]
        if not remaining:
            top = min(q[0] for q in positions)
            left = min(q[1] for q in positions)
            bottom = max(q[0]+v['shape'][0] for v, q in zip(tiles, positions))
            right = max(q[1]+v['shape'][1] for v, q in zip(tiles, positions))
            if (bottom-top)*(right-left) != area:
                return
            # Both open ends must lie on the outside of the completed rectangle.
            for j, local, direction in ((anchor, tiles[anchor]['entry'], tiles[anchor]['incoming']),
                                        (i, t['exit'], t['direction'])):
                r, c = positions[j]
                dy, dx = DIRECTIONS[direction]
                y, x = r+local[0]+dy, c+local[1]+dx
                if top <= y < bottom and left <= x < right:
                    return
            out = _render(info, positions)
            answers.setdefault(_key(out), (out, tuple(positions), tuple(order)))
            return
        dy, dx = DIRECTIONS[t['direction']]
        target = (p[0]+t['exit'][0]+dy, p[1]+t['exit'][1]+dx)
        for j in remaining:
            other = tiles[j]
            if other['incoming'] != (t['direction']+2) % 4:
                continue
            q = (target[0]-other['entry'][0], target[1]-other['entry'][1])
            if not permitted(q, other['shape']):
                continue
            if any(_overlap(q, other['shape'], positions[k], tiles[k]['shape']) for k in order):
                continue
            positions[j] = q
            visit(order+[j], tuple(k for k in remaining if k != j))
            positions[j] = None

    visit([anchor], tuple(i for i in range(len(tiles)) if i != anchor))
    assert len(answers) == 1, "no unique complete programmed assembly"
    return next(iter(answers.values()))


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer entirely from this input and the fixed family rule."""
    return _assemble(_extract(_grid(input_grid)))[0].copy()


def _parts(rng, count, total):
    options = [p for p in itertools.product(range(5, 8), repeat=count) if sum(p) == total]
    return options[int(rng.integers(len(options)))]


def _layout(rng, rows, cols):
    heights = [int(v) for v in rng.integers(5, 8, size=rows)]
    width = int(rng.integers(5*cols, 7*cols+1))
    rects, y = [], 0
    for h in heights:
        x = 0
        # Independently cut each band, so contacts need not be full-edge matches.
        for w in _parts(rng, cols, width):
            rects.append((y, x, h, w))
            x += w
        y += h
    return rects, (y, width)


def _contact(a, b):
    r, c, h, w = a
    y, x, hh, ww = b
    if c+w == x and max(r, y) < min(r+h, y+hh):
        return 1
    if x+ww == c and max(r, y) < min(r+h, y+hh):
        return 3
    if r+h == y and max(c, x) < min(c+w, x+ww):
        return 2
    if y+hh == r and max(c, x) < min(c+w, x+ww):
        return 0
    return None


def _center_ranges(rects, order):
    # Horizontal route contacts equate center rows; vertical ones equate columns.
    ranges = []
    for axis in (0, 1):
        parent = list(range(len(rects)))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for i, j in zip(order, order[1:]):
            d = _contact(rects[i], rects[j])
            if (axis == 0 and d in (1, 3)) or (axis == 1 and d in (0, 2)):
                parent[root(j)] = root(i)
        groups = {}
        for i, rect in enumerate(rects):
            groups.setdefault(root(i), []).append(i)
        limits = []
        for ids in groups.values():
            low = max(rects[i][axis]+2 for i in ids)
            high = min(rects[i][axis]+rects[i][axis+2]-3 for i in ids)
            if low > high:
                return None
            limits.append((ids, low, high))
        ranges.append(limits)
    return ranges


def _route_plans(rects, shape, corner, require_all):
    h, w = shape
    point = (h-1 if corner[0] == 'B' else 0, w-1 if corner[1] == 'R' else 0)
    anchor = next(i for i, (r, c, hh, ww) in enumerate(rects)
                  if r <= point[0] < r+hh and c <= point[1] < c+ww)
    graph = {i: [j for j in range(len(rects)) if j != i and _contact(rects[i], rects[j]) is not None]
             for i in range(len(rects))}
    plans = []
    def outer(i):
        r, c, hh, ww = rects[i]
        return [d for d, ok in enumerate((r == 0, c+ww == w, r+hh == h, c == 0)) if ok]
    def visit(order):
        if len(order) < len(rects):
            for j in graph[order[-1]]:
                if j not in order:
                    visit(order+[j])
            return
        ranges = _center_ranges(rects, order)
        if ranges is None:
            return
        exits = [_contact(rects[i], rects[j]) for i, j in zip(order, order[1:])]
        incoming_last = (exits[-1]+2) % 4
        for last in outer(order[-1]):
            if last == incoming_last:
                continue
            ds = exits+[last]
            if require_all and set(ds) != set(range(4)):
                continue
            for start in outer(anchor):
                if start != ds[0]:
                    plans.append((tuple(order), tuple(ds), start, ranges))
    visit([anchor])
    return plans


def _scatter(rng, shapes, anchor, fixed, canvas):
    # First try free placement. A separated 3x3 shelf is a bounded fallback;
    # it is a sampling device, never a solution convention.
    for _ in range(12):
        positions = [None]*len(shapes)
        positions[anchor] = fixed
        placed = [anchor]
        remaining = sorted((i for i in range(len(shapes)) if i != anchor),
                           key=lambda i: -shapes[i][0]*shapes[i][1])
        for i in remaining:
            h, w = shapes[i]
            found = False
            for _ in range(90):
                q = (int(rng.integers(1, canvas[0]-h)), int(rng.integers(1, canvas[1]-w)))
                # Inflate by one cell to guarantee a blank separating gap.
                if all(not _overlap((q[0]-1, q[1]-1), (h+2, w+2), positions[j], shapes[j]) for j in placed):
                    positions[i] = q
                    placed.append(i)
                    found = True
                    break
            if not found:
                break
        if len(placed) == len(shapes):
            return positions
    if min(canvas) < 27:
        return None
    positions = [None]*len(shapes)
    positions[anchor] = fixed
    ar = 0 if fixed[0] < 9 else 2
    ac = 0 if fixed[1] < 9 else 2
    slots = [(r, c) for r in range(3) for c in range(3) if (r, c) != (ar, ac)]
    rng.shuffle(slots)
    for i, (rr, cc) in zip((i for i in range(len(shapes)) if i != anchor), slots):
        h, w = shapes[i]
        rb = (0, 9, canvas[0]-9)[rr]
        cb = (0, 9, canvas[1]-9)[cc]
        positions[i] = (rb+1+int(rng.integers(8-h)), cb+1+int(rng.integers(8-w)))
    return positions


def _make_pair(rng, roles, palette, corner, layout, require_all=False):
    bg, carrier, port, marker = roles
    rows, cols = layout
    for attempt in range(100):
        rects, shape = _layout(rng, rows, cols)
        plans = _route_plans(rects, shape, corner, require_all)
        if not plans:
            continue
        order, exits, start, ranges = plans[int(rng.integers(len(plans)))]
        centers = [[0, 0] for _ in rects]
        for axis, groups in enumerate(ranges):
            for ids, low, high in groups:
                value = int(rng.integers(low, high+1))
                for i in ids:
                    centers[i][axis] = value
        min_h, min_w = shape[0]+2, shape[1]+2
        canvas = (int(rng.integers(max(22, min_h), 31)),
                  int(rng.integers(max(22, min_w), 31)))
        # Target extent <=28 permits at least one blank cell on both sides.
        margin_y, margin_x = int(rng.integers(1, 3)), int(rng.integers(1, 3))
        oy = margin_y if corner[0] == 'T' else canvas[0]-shape[0]-margin_y
        ox = margin_x if corner[1] == 'L' else canvas[1]-shape[1]-margin_x
        anchor = order[0]
        tiles = [None]*len(rects)
        inks = [int(palette[int(rng.integers(len(palette)))]) for _ in rects]
        if require_all:
            # Two ink colors for four directions explicitly defeat color decoding.
            inks = [int(palette[k % 2]) for k in rng.permutation(len(rects))]
        for k, i in enumerate(order):
            r, c, h, w = rects[i]
            center = (centers[i][0]-r, centers[i][1]-c)
            incoming = start if k == 0 else (exits[k-1]+2) % 4
            entry = _edge_point((h, w), center, incoming)
            block = np.full((h, w), carrier, np.int8)
            block[entry] = port
            y, x = center
            block[y-1:y+2, x-1:x+2][GLYPHS[exits[k]]] = inks[i]
            tiles[i] = dict(shape=(h, w), block=block, entry=entry, incoming=incoming,
                            direction=exits[k], center=center, ink=inks[i], carrier=carrier,
                            port=port, exit=_edge_point((h, w), center, exits[k]))
        final = [(oy+r, ox+c) for r, c, _, _ in rects]
        point = (oy+(shape[0]-1 if corner[0] == 'B' else 0),
                 ox+(shape[1]-1 if corner[1] == 'R' else 0))
        positions = _scatter(rng, [t['shape'] for t in tiles], anchor, final[anchor], canvas)
        if positions is None:
            # Resample a larger canvas on the next bounded candidate attempt.
            continue
        inp = np.full(canvas, bg, np.int8)
        for t, (r, c) in zip(tiles, positions):
            h, w = t['shape']
            t['old'] = (r, c)
            inp[r:r+h, c:c+w] = t['block']
        for p in _marker_cells(point, corner):
            inp[p] = marker
        intended_info = dict(canvas=canvas, background=bg, tiles=tiles, anchor=anchor)
        expected = _render(intended_info, final)
        try:
            info = _extract(inp)
            predicted, recovered, route = _assemble(info)
        except AssertionError:
            continue
        if np.array_equal(predicted, expected):
            return {'input': inp, 'output': expected}
    raise RuntimeError('bounded construction exhausted; no ambiguous candidate was accepted')


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, 'seed', 0)
    nt = _integer(num_train, 'num_train', 2, 10)
    nq = _integer(num_test, 'num_test', 1, 4)
    if nt <= nq:
        raise ValueError('num_train must exceed num_test')
    rng = np.random.default_rng(seed)
    colors = [int(v) for v in rng.permutation(10)]
    roles = tuple(colors[:4])
    palette = colors[4:4+int(rng.integers(2, 5))]
    corner = str(rng.choice(CORNERS))
    opposite = CORNERS[(CORNERS.index(corner)+2) % 4]
    heldout = [c for c in CORNERS if c not in (corner, opposite)]
    puzzle = {'train': [], 'test': []}
    seen = set()
    for split, count in (('train', nt), ('test', nq)):
        for index in range(count):
            for _ in range(30):
                if split == 'train':
                    c = corner if index % 2 == 0 else opposite
                    layout = (2, 3) if index == 0 else (2, 2) if index == 1 else \
                        ((1, 3), (1, 4), (2, 2), (2, 3), (3, 2))[int(rng.integers(5))]
                else:
                    c = str(rng.choice(heldout))
                    layout = (2, 4) if rng.integers(2) else (4, 2)
                pair = _make_pair(rng, roles, palette, c, layout,
                                  require_all=(split == 'train' and index == 0))
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError('distinct-input construction exhausted')
    validate_generated_puzzle(puzzle)
    return puzzle


def _geometry_witness(info, positions):
    """A different rectangle made by reordering one unanchored row band.

    This is a constructive witness that the literal plain edges and anchor
    cannot select the programmed answer without the decoded socket geometry.
    """
    groups = {}
    for i, (t, p) in enumerate(zip(info['tiles'], positions)):
        groups.setdefault((p[0], t['shape'][0]), []).append(i)
    for ids in groups.values():
        if info['anchor'] in ids or len(ids) < 2:
            continue
        ids.sort(key=lambda i: positions[i][1])
        if any(positions[a][1]+info['tiles'][a]['shape'][1] != positions[b][1]
               for a, b in zip(ids, ids[1:])):
            continue
        perm = positions[:]
        x = positions[ids[0]][1]
        for i in reversed(ids):
            perm[i] = (positions[i][0], x)
            x += info['tiles'][i]['shape'][1]
        if not np.array_equal(_render(info, perm), _render(info, positions)):
            return perm
    return None


def _inspect_pair(pair):
    assert isinstance(pair, dict) and set(pair) == {'input', 'output'}
    inp, out = _grid(pair['input']), _grid(pair['output'])
    assert inp.shape == out.shape
    info = _extract(inp)
    predicted, positions, order = _assemble(info)
    assert np.array_equal(out, predicted), 'incorrect programmed assembly'
    tiles, anchor = info['tiles'], info['anchor']
    assert positions[anchor] == tiles[anchor]['old'], 'anchor moved'
    assert not np.any(out == info['marker']), 'external L retained'
    assert all(out[p] == info['background'] for p in _marker_cells(info['point'], info['corner']))
    occupied = np.zeros(inp.shape, bool)
    for t, p in zip(tiles, positions):
        r, c = p
        h, w = t['shape']
        assert not np.any(occupied[r:r+h, c:c+w])
        occupied[r:r+h, c:c+w] = True
    assert np.array_equal(out != info['background'], occupied), 'rectangle cell conservation'
    assert np.count_nonzero(out == tiles[0]['port']) == 1, 'only route seed survives'
    # Independently inspect the global route's connectivity and edge crossings.
    route_cells = []
    for i in order:
        t, p = tiles[i], positions[i]
        route_cells.extend((p[0]+r, p[1]+c) for r, c in _stroke_cells(t))
    assert len(route_cells) == len(set(route_cells)), 'route repeats a cell'
    assert all(abs(a[0]-b[0])+abs(a[1]-b[1]) == 1 for a, b in zip(route_cells, route_cells[1:]))
    assert sum(np.count_nonzero(out == c) for c in set(t['ink'] for t in tiles)) == len(route_cells)-1
    # Omitting assembly leaves all routes at the scattered positions.
    scattered = _render(info, [t['old'] for t in tiles])
    assert not np.array_equal(scattered, out), 'assembly has no effect'
    # Omitting execution retains the glyphs, rather than producing the route.
    literal = np.full(inp.shape, info['background'], np.int8)
    for t, (r, c) in zip(tiles, positions):
        h, w = t['shape']
        literal[r:r+h, c:c+w] = t['block']
    assert not np.array_equal(literal, out), 'glyph execution has no effect'
    info.update(positions=list(positions), order=order, route_length=len(route_cells),
                turns=sum((t['incoming']+2) % 4 != t['direction'] for t in tiles),
                geometric_witness=_geometry_witness(info, list(positions)) is not None)
    return info


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    nt, nq = len(puzzle['train']), len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq
    seen, infos = set(), {'train': [], 'test': []}
    for split in infos:
        for pair in puzzle[split]:
            key = _key(_grid(pair['input']))
            assert key not in seen, 'duplicate input'
            seen.add(key)
            infos[split].append(_inspect_pair(pair))
    all_infos = infos['train']+infos['test']
    assert len({(v['background'], v['marker'], v['tiles'][0]['carrier'], v['tiles'][0]['port'])
                for v in all_infos}) == 1
    first = infos['train'][0]
    assert len(first['tiles']) == 6 and {t['direction'] for t in first['tiles']} == set(range(4))
    assert first['geometric_witness'], 'missing ambiguity witness when commands are ignored'
    assert first['turns'] >= 2
    assert any(t['incoming'] == (t['direction']+2) % 4 for t in first['tiles']), \
        'straight-through operation must be demonstrated'
    by_color = {}
    for t in first['tiles']:
        by_color.setdefault(t['ink'], set()).add(t['direction'])
    assert any(len(ds) > 1 for ds in by_color.values()), 'color-only command shortcut'
    assert len(infos['train'][1]['tiles']) == 4
    assert {v['corner'] for v in infos['train']}.isdisjoint({v['corner'] for v in infos['test']})
    assert all(len(v['tiles']) == 8 for v in infos['test'])
    assert max(len(v['tiles']) for v in infos['train']) < 8


def self_test():
    """Bounded 42-seed semantic/diversity checks plus corruption regressions."""
    def fixture(rects, order, exits, centers, inks):
        # Hand-constructed semantic cases, never used by generate_puzzle.
        tiles = [None]*len(rects)
        inp = np.zeros((30, 30), np.int8)
        final = [(1+r, 1+c) for r, c, h, w in rects]
        for k, i in enumerate(order):
            r, c, h, w = rects[i]
            center = centers[i]
            incoming = 3 if k == 0 else (exits[k-1]+2) % 4
            entry = _edge_point((h, w), center, incoming)
            block = np.ones((h, w), np.int8)
            block[entry] = 2
            y, x = center
            block[y-1:y+2, x-1:x+2][GLYPHS[exits[k]]] = inks[i]
            old = (1+(i//3)*9, 1+(i % 3)*9)
            inp[old[0]:old[0]+h, old[1]:old[1]+w] = block
            tiles[i] = dict(shape=(h, w), old=old, entry=entry, center=center,
                            direction=exits[k], incoming=incoming, carrier=1,
                            port=2, ink=inks[i], block=block,
                            exit=_edge_point((h, w), center, exits[k]))
        for p in _marker_cells((1, 1), 'TL'):
            inp[p] = 3
        expected = _render(dict(canvas=inp.shape, background=0, tiles=tiles,
                                anchor=0), final)
        return {'input': inp, 'output': expected}

    # Five and seven tiles are semantic family members outside the current
    # equal-count-per-band sampler. Identical permutations may have one answer.
    five = fixture([(0, 5*i, 5, 5) for i in range(5)], tuple(range(5)),
                   (1,)*5, [(2, 2)]*5, [5, 4, 4, 4, 4])
    _inspect_pair(five)
    seven_rects = [(0, 0, 5, 6), (0, 6, 5, 7), (0, 13, 5, 7)] + \
        [(5, 5*i, 5, 5) for i in range(4)]
    seven = fixture(seven_rects, (0, 1, 2, 6, 5, 4, 3),
                    (1, 1, 2, 3, 3, 3, 3),
                    [(2, 2), (2, 2), (2, 4)]+[(2, 2)]*4,
                    [5, 6, 7, 4, 4, 4, 4])
    _inspect_pair(seven)
    # Same geometry, different colors on interchangeable straight tiles:
    # these imply distinct equally supported outputs and must be rejected.
    ambiguous = fixture([(5*r, 5*c, 5, 5) for r in range(2) for c in range(4)],
                        (0, 1, 2, 3, 7, 6, 5, 4),
                        (1, 1, 1, 2, 3, 3, 3, 3), [(2, 2)]*8,
                        [4, 5, 6, 4, 4, 4, 4, 4])
    try:
        solve(ambiguous['input'])
    except AssertionError:
        pass
    else:
        raise AssertionError('distinct tied outputs accepted')
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({tuple(_key(_grid(p[k])) for s in ('train', 'test')
                           for p in puzzle[s] for k in ('input', 'output')) for puzzle in puzzles}) == 42
    infos = [_inspect_pair(pair) for p in puzzles for s in ('train', 'test') for pair in p[s]]
    assert {v['corner'] for v in infos} == set(CORNERS)
    assert {len(v['tiles']) for v in infos} >= {3, 4, 6, 8}
    assert len({v['canvas'] for v in infos}) >= 20
    assert len({t['shape'] for v in infos for t in v['tiles']}) == 9
    assert len({v['route_length'] for v in infos}) >= 20
    assert len({v['turns'] for v in infos}) >= 4
    assert len({(v['background'], v['marker'], v['tiles'][0]['carrier'], v['tiles'][0]['port'])
                for v in infos}) >= 30
    assert any(any(t['incoming'] == (t['direction']+2) % 4 for t in v['tiles']) for v in infos)
    # Some contacts use only a subedge of a neighboring tile.
    subedges = 0
    for v in infos:
        for i, j in zip(v['order'], v['order'][1:]):
            a, b = v['tiles'][i], v['tiles'][j]
            r, c = v['positions'][i]
            y, x = v['positions'][j]
            horizontal = a['direction'] in (1, 3)
            overlap = min(r+a['shape'][0], y+b['shape'][0])-max(r, y) if horizontal else \
                min(c+a['shape'][1], x+b['shape'][1])-max(c, x)
            subedges += int(overlap < max(a['shape'][0 if horizontal else 1],
                                         b['shape'][0 if horizontal else 1]))
    assert subedges >= 10
    for nt in range(2, 11):
        for nq in range(1, min(4, nt-1)+1):
            validate_generated_puzzle(generate_puzzle(741+nt*5+nq, nt, nq))
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    p = generate_puzzle(12345, 3, 2)
    q = generate_puzzle(12345, 3, 2)
    assert all(np.array_equal(a[k], b[k]) for s in p for a, b in zip(p[s], q[s]) for k in a)
    # Segmentation order is not an execution or tie-break convention.
    info = _extract(p['test'][0]['input'])
    reference = _assemble(info)[0]
    n = len(info['tiles'])
    info['tiles'] = list(reversed(info['tiles']))
    info['anchor'] = n-1-info['anchor']
    assert np.array_equal(_assemble(info)[0], reference)
    for damage in ('route', 'marker', 'duplicate', 'glyph'):
        q = {s: [{k: g.copy() for k, g in pair.items()} for pair in p[s]] for s in p}
        pair = q['test'][0]
        v = _extract(pair['input'])
        t = v['tiles'][v['anchor']]
        r, c = t['old']
        if damage == 'route':
            i = v['anchor']
            y, x = t['center']
            pair['output'][r+y, c+x] = v['background']
        elif damage == 'marker':
            pair['output'][_marker_cells(v['point'], v['corner'])[0]] = v['marker']
        elif damage == 'duplicate':
            q['train'][1] = {k: g.copy() for k, g in q['train'][0].items()}
        else:
            y, x = t['center']
            pair['input'][r+y, c+x] = t['carrier']
        try:
            validate_generated_puzzle(q)
        except AssertionError:
            pass
        else:
            raise AssertionError('semantic corruption accepted: '+damage)
    return {'seeds': 42, 'pairs': len(infos), 'canvas_shapes': len({v['canvas'] for v in infos}),
            'route_lengths': len({v['route_length'] for v in infos}), 'subedge_contacts': subedges}


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
