"""Quilting on a topological lattice: enclosure, remeshing, finite exact cover.

Only NumPy and the standard library are used. No parent code or examples are
imported. All decoding, including source extraction, uses the visible input.
"""
from __future__ import annotations

import json
from collections import Counter
import numpy as np


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "colors outside 0..9")
    return g.astype(np.int8, copy=True)


def _key(g):
    return g.shape, g.astype(np.int8).tobytes()


def _components(mask):
    seen = np.zeros(mask.shape, bool)
    result = []
    h, w = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if seen[y, x]:
            continue
        cells, stack = [], [(int(y), int(x))]
        seen[y, x] = True
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        result.append(np.array(cells, dtype=int))
    return result


def _holes(mask):
    m = np.pad(mask, 1)
    outside = np.zeros(m.shape, bool)
    outside[0, 0] = True
    stack = [(0, 0)]
    h, w = m.shape
    while stack:
        r, c = stack.pop()
        for rr, cc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= rr < h and 0 <= cc < w and not m[rr, cc] and not outside[rr, cc]:
                outside[rr, cc] = True
                stack.append((rr, cc))
    return (~m & ~outside)[1:-1, 1:-1]


def _tracks(ids, records, axis):
    """Overlapping projections form tracks; complete whitespace separates tracks."""
    ordered = sorted(ids, key=lambda i: int(records[i]['lo'][axis]))
    bands, end = [], -2
    for i in ordered:
        lo, hi = int(records[i]['lo'][axis]), int(records[i]['hi'][axis])
        if not bands or lo > end:
            bands.append([])
        bands[-1].append(i)
        end = max(end, hi)
    mapping = {i: k for k, band in enumerate(bands) for i in band}
    # There must be at least one blank pixel between distinct tracks.
    for a, b in zip(bands, bands[1:]):
        _require(min(records[i]['lo'][axis] for i in b) -
                 max(records[i]['hi'][axis] for i in a) >= 2, "tracks need gutters")
    # Transitive, staggered overlaps are excluded: each track has a shared projection.
    for band in bands:
        _require(max(records[i]['lo'][axis] for i in band) <=
                 min(records[i]['hi'][axis] for i in band), "unanchored track")
    return mapping, len(bands)


def _decode(input_grid):
    grid = _grid(input_grid)
    bg = int(grid[0, 0])
    _require(np.all(grid[0] == bg) and np.all(grid[-1] == bg) and
             np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg), "uniform perimeter")
    records = []
    for cells in _components(grid != bg):
        lo, hi = cells.min(axis=0), cells.max(axis=0)
        mask = np.zeros(grid.shape, bool)
        mask[cells[:, 0], cells[:, 1]] = True
        hole = _holes(mask)
        colors = set(int(c) for c in grid[cells[:, 0], cells[:, 1]])
        if hole.any():
            _require(len(colors) == 1 and len(_components(hole)) == 1, "invalid contour")
        records.append(dict(cells=cells, lo=lo, hi=hi, mask=mask, hole=hole,
                            colors=colors, contour=bool(hole.any())))
    loops = [i for i, r in enumerate(records) if r['contour']]
    _require(bool(loops), "no contour")
    inks = {next(iter(records[i]['colors'])) for i in loops}
    _require(len(inks) == 1, "contours must share drawing color")
    ink = inks.pop()
    parents = []
    for i, rec in enumerate(records):
        containing = []
        for j in loops:
            if i == j:
                continue
            inside = records[j]['hole'][rec['cells'][:, 0], rec['cells'][:, 1]]
            _require(inside.all() or not inside.any(), "crossing objects")
            if inside.all():
                containing.append(j)
        parents.append(min(containing, key=lambda j: int(records[j]['hole'].sum()))
                       if containing else None)
    children = {i: [] for i in loops}
    roots, tiles = [], []
    labels = np.zeros(grid.shape, int)
    for i, rec in enumerate(records):
        labels[rec['mask']] = i+1
        if ink in rec['colors']:
            _require(rec['colors'] == {ink}, "payload cannot contain drawing color")
            if parents[i] is None:
                _require(rec['contour'], "exterior demand blob")
                roots.append(i)
            else:
                children[parents[i]].append(i)
            if not rec['contour']:
                _require(len(rec['cells']) >= .4 * np.prod(rec['hi']-rec['lo']+1),
                         "blob is not compact")
        else:
            _require(parents[i] is None, "source tile inside contour")
            y, x = rec['lo']; yy, xx = rec['hi']+1
            tile = grid[y:yy, x:xx]
            _require(tile.size == len(rec['cells']) and np.all(tile != bg), "nonrectangular source")
            tiles.append(tile.copy())
    for a, b in ((labels[:-1, :-1], labels[1:, 1:]), (labels[:-1, 1:], labels[1:, :-1])):
        _require(not np.any((a != b) & (a > 0) & (b > 0)), "diagonally touching objects")
    _require(len(roots) == 1 and bool(tiles), "one scene root and a supply are required")
    boards = []

    def node(i):
        ids = children[i]
        _require(bool(ids), "empty contour")
        kinds = {records[j]['contour'] for j in ids}
        _require(len(kinds) == 1, "a room cannot mix blobs and child contours")
        rows, nr = _tracks(ids, records, 0)
        cols, nc = _tracks(ids, records, 1)
        positions = [(rows[j], cols[j]) for j in ids]
        _require(len(set(positions)) == len(ids), "multiple objects in one logical cell")
        if kinds == {False}:
            mask = np.zeros((nr, nc), bool)
            for p in positions:
                mask[p] = True
            index = len(boards)
            boards.append(mask)
            return dict(board=index, shape=mask.shape, children=None, record=i)
        child_map = {p: node(j) for p, j in zip(positions, ids)}
        return dict(board=None, shape=(nr, nc), children=child_map, record=i)

    tree = node(roots[0])
    return dict(grid=grid, bg=bg, ink=ink, records=records, tree=tree,
                boards=boards, tiles=tiles, children=children, root=roots[0])


def _cover(boards, tiles, max_outcomes=2):
    """Joint exact cover of ALL rooms with one finite inventory.

    Return distinct rendered outcomes and one placement witness for each.
    Tile identities may be interchangeable; no source rotation is allowed.
    There is no seed, output label, or solver search budget.
    """
    groups, lookup = [], {}
    for tile in tiles:
        key = _key(tile)
        if key in lookup:
            groups[lookup[key]]['count'] += 1
        else:
            lookup[key] = len(groups)
            groups.append(dict(tile=tile, count=1))
    coords, indices = [], []
    for b, mask in enumerate(boards):
        ix = np.full(mask.shape, -1, int)
        for y, x in zip(*np.nonzero(mask)):
            ix[y, x] = len(coords)
            coords.append((b, int(y), int(x)))
        indices.append(ix)
    n = len(coords)
    by_cell = [[] for _ in range(n)]
    placements = []
    for gi, group in enumerate(groups):
        tile = group['tile']
        h, w = tile.shape
        for b, mask in enumerate(boards):
            for y in range(mask.shape[0]-h+1):
                for x in range(mask.shape[1]-w+1):
                    if not np.all(mask[y:y+h, x:x+w]):
                        continue
                    cells = tuple(int(v) for v in indices[b][y:y+h, x:x+w].ravel())
                    bits = sum(1 << c for c in cells)
                    pi = len(placements)
                    placements.append((gi, bits, cells, tuple(int(v) for v in tile.flat), b, y, x))
                    for c in cells:
                        by_cell[c].append(pi)
    remaining = [g['count'] for g in groups]
    rendering = [-1]*n
    full = (1 << n)-1
    outcomes, dead, path = {}, set(), []
    nodes = 0

    def search(covered):
        nonlocal nodes
        nodes += 1
        if covered == full:
            outcomes.setdefault(tuple(rendering), tuple(path))
            return True
        state = covered, tuple(remaining)
        if state in dead:
            return False
        if sum(remaining[i]*g['tile'].size for i, g in enumerate(groups)) < n-covered.bit_count():
            dead.add(state)
            return False
        best = None
        for c in range(n):
            if covered & (1 << c):
                continue
            options = [p for p in by_cell[c] if remaining[placements[p][0]] and
                       not (covered & placements[p][1])]
            if not options:
                dead.add(state)
                return False
            if best is None or len(options) < len(best):
                best = options
                if len(best) == 1:
                    break
        found = False
        for pi in best:
            gi, bits, cells, colors, b, y, x = placements[pi]
            remaining[gi] -= 1
            for c, color in zip(cells, colors):
                rendering[c] = color
            path.append((gi, b, y, x))
            found = search(covered | bits) or found
            path.pop()
            remaining[gi] += 1
            for c in cells:
                rendering[c] = -1
            if len(outcomes) >= max_outcomes:
                return True
        if not found:
            dead.add(state)
        return found

    search(0)
    return dict(outcomes=outcomes, groups=groups, coords=coords, nodes=nodes)


def _frame(content, bg, ink):
    out = np.pad(content, 2, constant_values=bg)
    out[0] = out[-1] = ink
    out[:, 0] = out[:, -1] = ink
    return out


def _layout(parts, shape, bg, gap):
    nr, nc = shape
    hs = [max(g.shape[0] for (r, c), g in parts.items() if r == y) for y in range(nr)]
    ws = [max(g.shape[1] for (r, c), g in parts.items() if c == x) for x in range(nc)]
    ys = np.cumsum([0]+[v+gap for v in hs[:-1]]).tolist()
    xs = np.cumsum([0]+[v+gap for v in ws[:-1]]).tolist()
    out = np.full((sum(hs)+gap*(nr-1), sum(ws)+gap*(nc-1)), bg, np.int8)
    for (r, c), g in parts.items():
        y, x = ys[r], xs[c]
        out[y:y+g.shape[0], x:x+g.shape[1]] = g
    return out


def _render_tree(tree, boards, bg, ink):
    if tree['children'] is None:
        return _frame(boards[tree['board']], bg, ink)
    parts = {p: _render_tree(child, boards, bg, ink) for p, child in tree['children'].items()}
    return _frame(_layout(parts, tree['shape'], bg, 1), bg, ink)


def _prediction(info):
    cover = _cover(info['boards'], info['tiles'])
    _require(len(cover['outcomes']) == 1, "no unique rendered global packing")
    colors = next(iter(cover['outcomes']))
    boards = [np.full(m.shape, info['bg'], np.int8) for m in info['boards']]
    for color, (b, y, x) in zip(colors, cover['coords']):
        boards[b][y, x] = color
    out = _render_tree(info['tree'], boards, info['bg'], info['ink'])
    _require(max(out.shape) <= 30, "normalized output too large")
    return out, cover


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer topology, logical tracks, inventory, and unique quilt from input only."""
    return _prediction(_decode(input_grid))[0]


def _pattern(shape, palette, rng, asymmetric=False):
    h, w = shape
    a, b = (int(v) for v in rng.choice(palette, 2, replace=False))
    out = np.full(shape, a, np.int8)
    mode = int(rng.integers(5))
    yy, xx = np.indices(shape)
    if mode == 1:
        out[(yy+xx) % 2 == 1] = b
    elif mode == 2:
        out[yy % 2 == 1] = b
    elif mode == 3:
        out[xx % 2 == 1] = b
    elif mode == 4:
        out = rng.choice(palette, size=shape).astype(np.int8)
    if asymmetric and out.size > 1:
        out.fill(a)
        out[0, 0] = b
    return out


def _random_board(rng, cap, count):
    h, w = cap
    mask = np.zeros((h, w), bool)
    rects = []
    for i in range(count):
        options = []
        for _ in range(65):
            th, tw = int(rng.integers(1, min(h, 3)+1)), int(rng.integers(1, min(w, 3)+1))
            y, x = int(rng.integers(h-th+1)), int(rng.integers(w-tw+1))
            if np.any(mask[y:y+th, x:x+tw]):
                continue
            expanded = mask[max(0, y-1):min(h, y+th+1), max(0, x-1):min(w, x+tw+1)]
            # Require an edge contact, not just a corner contact.
            edge = ((y > 0 and mask[y-1, x:x+tw].any()) or
                    (y+th < h and mask[y+th, x:x+tw].any()) or
                    (x > 0 and mask[y:y+th, x-1].any()) or
                    (x+tw < w and mask[y:y+th, x+tw].any()))
            if rects and (not expanded.any() or not edge):
                continue
            options.append((y, x, th, tw))
        if not options:
            return None
        y, x, th, tw = options[int(rng.integers(len(options)))]
        mask[y:y+th, x:x+tw] = True
        rects.append((y, x, th, tw))
    ys, xs = np.nonzero(mask)
    y0, x0 = int(ys.min()), int(xs.min())
    mask = mask[y0:int(ys.max())+1, x0:int(xs.max())+1]
    return mask, [(y-y0, x-x0, th, tw) for y, x, th, tw in rects]


def _tree_for_boards(n, rng, deep=False):
    leaves = [dict(board=i, shape=None, children=None) for i in range(n)]
    if n == 1:
        return leaves[0]
    rng.shuffle(leaves)
    if deep:
        wrapped = int(rng.integers(n))
        leaves[wrapped] = dict(board=None, shape=(1, 1), children={(0, 0): leaves[wrapped]})
    axis = int(rng.integers(2))
    if n == 2 or rng.random() < .6:
        shape = (n, 1) if axis == 0 else (1, n)
        positions = [(i, 0) if axis == 0 else (0, i) for i in range(n)]
    else:
        shape = (2, 2)
        positions = [(0, 0), (0, 1), (1, int(rng.integers(2)))]
    return dict(board=None, shape=shape, children=dict(zip(positions, leaves)))


def _core_construction(rng, bg, ink, palette):
    """A parametric resource competition, with either all-used or surplus supply."""
    h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
    horizontal = bool(rng.integers(2))
    copies = h if horizontal else w
    demand_copies = int(rng.integers(2, copies+1))
    strip_shape = (1, w) if horizontal else (h, 1)
    long_shape = (1, demand_copies*w) if horizontal else (demand_copies*h, 1)
    strip = _pattern(strip_shape, palette, rng, asymmetric=True)
    big = _pattern((h, w), palette, rng, asymmetric=True)
    stripes = np.tile(strip, (h, 1)) if horizontal else np.tile(strip, (1, w))
    if np.array_equal(big, stripes):
        big[-1, -1] = next(int(c) for c in palette if c != int(big[-1, -1]))
    masks = [np.ones((h, w), bool), np.ones(long_shape, bool)]
    fills = [big, np.tile(strip, (1, demand_copies)) if horizontal else np.tile(strip, (demand_copies, 1))]
    tiles = [big.copy()]+[strip.copy() for _ in range(copies)]
    return masks, fills, tiles, _tree_for_boards(2, rng)


def _random_construction(rng, rooms, palette, witness=False, deep=False):
    masks, rectlists, dims = [], [], set()
    for i in range(rooms):
        upper = 5 if rooms == 1 else 4
        cap = (int(rng.integers(2, upper)), int(rng.integers(2, upper)))
        count = int(rng.integers(2, 5)) if rooms == 1 else int(rng.integers(1, 4))
        board = _random_board(rng, cap, count)
        if board is None:
            return None
        m, rects = board
        masks.append(m); rectlists.append(rects)
        dims.update((h, w) for y, x, h, w in rects)
    if witness and (masks[0].all() or min(masks[0].shape) < 2):
        return None
    patterns = {d: _pattern(d, palette, rng) for d in sorted(dims)}
    if witness:
        candidates = [d for d in sorted(dims) if np.prod(d) > 1]
        if not candidates:
            return None
        d = candidates[int(rng.integers(len(candidates)))]
        patterns[d] = _pattern(d, palette, rng, asymmetric=True)
    fills, tiles = [], []
    for mask, rects in zip(masks, rectlists):
        fill = np.full(mask.shape, -1, np.int8)
        for y, x, h, w in rects:
            tile = patterns[(h, w)]
            fill[y:y+h, x:x+w] = tile
            tiles.append(tile.copy())
        fills.append(fill)
    return masks, fills, tiles, _tree_for_boards(rooms, rng, deep)


def _outline(h, w, pads, rng, compact):
    """Four safe, independently stepped corridors around the contents."""
    t, b, l, r = pads
    top = np.zeros(w, int); bottom = np.full(w, h-1, int)
    left = np.zeros(h, int); right = np.full(h, w-1, int)
    for profile, start, stop, cap, sign in ((top, l, w-r, t-2, 1),
                                           (bottom, l, w-r, b-2, -1),
                                           (left, t, h-b, l-2, 1),
                                           (right, t, h-b, r-2, -1)):
        if stop > start and cap and rng.random() < .8:
            profile[start:stop] += sign * rng.integers(0, cap+1, stop-start)
    out = np.zeros((h, w), bool)

    def path(points):
        for (y, x), (yy, xx) in zip(points, points[1:]):
            out[y, min(x, xx):max(x, xx)+1] = True
            out[min(y, yy):max(y, yy)+1, xx] = True
    path([(int(top[x]), x) for x in range(w)])
    path([(int(bottom[x]), x) for x in range(w)])
    path([(y, int(left[y])) for y in range(h)])
    path([(y, int(right[y])) for y in range(h)])
    return out


def _metric_scene(tree, masks, rng, compact=False, tight=False):
    def render(node):
        if node['children'] is None:
            mask = masks[node['board']]
            parts = {}
            for y, x in zip(*np.nonzero(mask)):
                bh, bw = (int(rng.integers(1, 2 if tight else 3 if compact else 4)) for _ in range(2))
                blob = np.ones((bh, bw), np.int8)
                if bh > 1 and bw > 1 and rng.random() < .5:
                    blob[-1, -1] = 0
                parts[(int(y), int(x))] = blob
            content = _layout(parts, mask.shape, 0, int(rng.integers(1, 2 if compact else 4)))
        else:
            parts = {p: render(child) for p, child in node['children'].items()}
            content = _layout(parts, node['shape'], 0, 1 if tight else int(rng.integers(1, 3 if compact else 5)))
        # One side has extra room even in compact mode, so genuine displaced
        # contours occur without changing either the topology or logical tracks.
        pads = [2 if tight else int(rng.integers(2, 4 if compact else 5)) for _ in range(4)]
        t, b, l, r = pads
        h, w = content.shape
        out = _outline(h+t+b, w+l+r, pads, rng, compact)
        out[t:t+h, l:l+w] |= content.astype(bool)
        return out.astype(np.int8)
    return render(tree).astype(bool)


def _scatter(tiles, height, width, bg, rng):
    if height < max(t.shape[0] for t in tiles) or width < max(t.shape[1] for t in tiles):
        return None
    order = sorted(range(len(tiles)), key=lambda i: -tiles[i].size)
    for _ in range(12):
        tray = np.full((height, width), bg, np.int8)
        occupied = np.zeros(tray.shape, bool)
        ok = True
        for i in order:
            h, w = tiles[i].shape
            legal = []
            for y in range(height-h+1):
                for x in range(width-w+1):
                    if not occupied[max(0, y-1):min(height, y+h+1),
                                    max(0, x-1):min(width, x+w+1)].any():
                        legal.append((y, x))
            if not legal:
                ok = False
                break
            y, x = legal[int(rng.integers(len(legal)))]
            tray[y:y+h, x:x+w] = tiles[i]
            occupied[y:y+h, x:x+w] = True
        if ok:
            return tray
    return None


def _join(scene, tiles, side, bg, ink, rng):
    h, w = scene.shape
    gap = int(rng.integers(1, 3))
    vertical_cut = side in ('left', 'right')
    available = 28-(w if vertical_cut else h)-gap
    minimum = max(t.shape[1 if vertical_cut else 0] for t in tiles)
    if available < minimum or max(h, w) > 28:
        return None
    for _ in range(15):
        cross = int(rng.integers(max(h if vertical_cut else w,
                                      max(t.shape[0 if vertical_cut else 1] for t in tiles)), 29))
        extent = int(rng.integers(minimum, available+1))
        th, tw = (cross, extent) if vertical_cut else (extent, cross)
        tray = _scatter(tiles, th, tw, bg, rng)
        if tray is None:
            continue
        if vertical_cut:
            canvas = np.full((cross, w+gap+tw), bg, np.int8)
            sy = int(rng.integers(cross-h+1))
            sx, tx = (0, w+gap) if side == 'left' else (tw+gap, 0)
            canvas[sy:sy+h, sx:sx+w] = np.where(scene, ink, bg)
            canvas[:, tx:tx+tw] = tray
        else:
            canvas = np.full((h+gap+th, cross), bg, np.int8)
            sx = int(rng.integers(cross-w+1))
            sy, ty = (0, h+gap) if side == 'top' else (th+gap, 0)
            canvas[sy:sy+h, sx:sx+w] = np.where(scene, ink, bg)
            canvas[ty:ty+th] = tray
        return np.pad(canvas, 1, constant_values=bg)
    return None


def _make_pair(rng, kind, side, roles, allow_surplus=True):
    bg, ink = roles
    palette = [c for c in range(10) if c not in roles]
    rng.shuffle(palette)
    palette = palette[:int(rng.integers(2, 7))]
    for attempt in range(360):
        if kind == 'resource':
            built = _core_construction(rng, bg, ink, palette)
        else:
            rooms = 1 if kind == 'shape' else int(rng.integers(1, 3)) if kind == 'ordinary' else int(rng.integers(2, 4))
            built = _random_construction(rng, rooms, palette, witness=kind == 'shape',
                                         deep=kind == 'test')
        if built is None:
            continue
        masks, fills, tiles, tree = built
        fills = [np.where(m, f, bg).astype(np.int8) for m, f in zip(masks, fills)]
        cover = _cover(masks, tiles)
        expected_colors = tuple(int(fills[b][y, x]) for b, y, x in cover['coords'])
        if len(cover['outcomes']) != 1 or expected_colors not in cover['outcomes']:
            continue
        if kind == 'shape' and len(next(iter(cover['outcomes'].values()))) < 2:
            continue
        # Optional surplus is tested against the complete global packing.
        if allow_surplus and kind != 'resource' and rng.random() < .4:
            for _ in range(3):
                d = (int(rng.integers(1, 5)), int(rng.integers(1, 5)))
                matching = next((t for t in tiles if t.shape == d), None)
                extra = matching.copy() if matching is not None else _pattern(d, palette, rng)
                trial = _cover(masks, tiles+[extra])
                if len(trial['outcomes']) == 1 and expected_colors in trial['outcomes']:
                    tiles.append(extra)
                    break
        expected = _render_tree(tree, fills, bg, ink)
        if max(expected.shape) > 30:
            continue
        scene = _metric_scene(tree, masks, rng, compact=attempt >= 12 or kind in ('resource', 'test'),
                              tight=kind == 'test' and len(masks) == 3)
        grid = _join(scene, tiles, side, bg, ink, rng)
        if grid is None:
            continue
        try:
            info = _decode(grid)
            actual, actual_cover = _prediction(info)
        except ValueError:
            continue
        if not np.array_equal(expected, actual):
            continue
        if not allow_surplus and len(tiles) > len(next(iter(actual_cover['outcomes'].values()))):
            continue
        if kind == 'resource' and not any(len(_cover([m], info['tiles'])['outcomes']) >= 2
                                          for m in info['boards']):
            continue
        return dict(input=grid, output=expected)
    raise RuntimeError('bounded construction exhausted for '+kind)


def _counts(num_train, num_test):
    for name, value, low, high in (('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(name+' must be an integer in its legal range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train, num_test = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    base = tuple(int(c) for c in rng.choice(10, 2, replace=False))
    palette_policy = int(rng.integers(3))
    sides = list(rng.permutation(['left', 'right', 'top', 'bottom']))
    opposite = dict(left='right', right='left', top='bottom', bottom='top')
    held_side = bool(rng.integers(2))
    puzzle, seen = dict(train=[], test=[]), set()
    surplus_taught = False
    for split, count in (('train', num_train), ('test', num_test)):
        for i in range(count):
            kind = ('shape' if i == 0 else 'resource' if i == 1 else 'ordinary') if split == 'train' else 'test'
            side = sides[i % 2] if split == 'train' else opposite[sides[0]] if held_side else sides[i % 2]
            if palette_policy == 0:
                roles = base
            elif palette_policy == 1:
                roles = base if (i+(split == 'test')) % 2 == 0 else base[::-1]
            else:
                roles = tuple(int(c) for c in rng.choice(10, 2, replace=False))
            for _ in range(20):
                pair = _make_pair(rng, kind, side, roles,
                                  allow_surplus=split == 'train' or surplus_taught)
                key = _key(pair['input'])
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    if split == 'train':
                        _, inventory_check = _prediction(_decode(pair['input']))
                        used = len(next(iter(inventory_check['outcomes'].values())))
                        surplus_taught |= len(_decode(pair['input'])['tiles']) > used
                    break
            else:
                raise RuntimeError('distinct input construction exhausted')
        rng.shuffle(puzzle[split])
    return puzzle


def _depth(tree):
    return 1 if tree['children'] is None else 1+max(_depth(c) for c in tree['children'].values())


def _features(info, cover):
    witness = next(iter(cover['outcomes'].values()))
    used = Counter(p[0] for p in witness)
    patterns = [cover['groups'][i]['tile'] for i in used]
    nonrect = sum(not m.all() for m in info['boards'])
    local_ambiguous = any(len(_cover([m], info['tiles'])['outcomes']) > 1 for m in info['boards'])
    nonrect_contours = 0
    for rec in info['records']:
        if rec['contour']:
            y, x = rec['lo']; yy, xx = rec['hi']+1
            m = rec['mask'][y:yy, x:xx]
            nonrect_contours += not (m[0].all() and m[-1].all() and m[:, 0].all() and m[:, -1].all())
    return dict(rooms=len(info['boards']), depth=_depth(info['tree']), nonrect=nonrect,
                contours=sum(r['contour'] for r in info['records']),
                distorted=nonrect_contours, demand=sum(int(m.sum()) for m in info['boards']),
                used=len(witness), supplied=len(info['tiles']),
                repeats=any(n > 1 for n in used.values()), local_ambiguous=local_ambiguous,
                asymmetric=any(t.size > 1 and not np.array_equal(t, t[::-1, ::-1]) for t in patterns),
                shapes=tuple(m.shape for m in info['boards']))


def validate_generated_puzzle(puzzle):
    """Validate visible topology, packing uniqueness, normalization and teaching."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'puzzle keys')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'pair lists')
    nt, ne = _counts(len(puzzle['train']), len(puzzle['test']))
    seen, train_features, test_features = set(), [], []
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, 'pair keys')
            inp, out = _grid(pair['input']), _grid(pair['output'])
            _require(_key(inp) not in seen, 'duplicate input')
            seen.add(_key(inp))
            info = _decode(inp)
            expected, cover = _prediction(info)
            _require(np.array_equal(expected, out), 'incorrect output')
            # Every output payload cell represents exactly one input terminal blob.
            demand = sum(int(m.sum()) for m in info['boards'])
            _require(int(((out != info['bg']) & (out != info['ink'])).sum()) == demand,
                     'demand conservation')
            _require(set(np.unique(out)) <= set(np.unique(inp)), 'invented output color')
            _require(np.all(out[0] == info['ink']) and np.all(out[-1] == info['ink']) and
                     np.all(out[:, 0] == info['ink']) and np.all(out[:, -1] == info['ink']), 'root crop')
            output_loops = [c for c in _components(out == info['ink'])
                            if len(c) > 1]
            _require(len(output_loops) == sum(r['contour'] for r in info['records']), 'contour conservation')
            for cells in output_loops:
                lo, hi = cells.min(axis=0), cells.max(axis=0)
                y, x = lo; yy, xx = hi
                _require(len(cells) == 2*(yy-y+xx-x), 'frame perimeter')
                _require(np.all(out[y+1, x+1:xx] == info['bg']) and
                         np.all(out[yy-1, x+1:xx] == info['bg']) and
                         np.all(out[y+1:yy, x+1] == info['bg']) and
                         np.all(out[y+1:yy, xx-1] == info['bg']), 'one-cell interior moat')
            f = _features(info, cover)
            (train_features if split == 'train' else test_features).append(f)
    _require(any(f['nonrect'] and f['used'] >= 2 and f['asymmetric'] for f in train_features),
             'missing silhouette and pattern witness')
    _require(any(f['rooms'] >= 2 and f['local_ambiguous'] and f['repeats'] for f in train_features),
             'missing shared-inventory witness')
    _require(any(f['depth'] >= 2 for f in train_features), 'missing nesting witness')
    _require(all(f['depth'] >= 3 for f in test_features), 'missing test recombination shift')
    _require(not any(f['supplied'] > f['used'] for f in test_features) or
             any(f['supplied'] > f['used'] for f in train_features), 'untaught surplus selection')


def _puzzle_key(puzzle):
    return tuple((_key(p['input']), _key(p['output'])) for s in ('train', 'test') for p in puzzle[s])


def self_test():
    """Bounded 42-seed variation and semantic counterexample checks."""
    puzzles, features, normalized, input_shapes, output_shapes, topologies = [], [], set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        for split in ('train', 'test'):
            for pair in p[split]:
                info = _decode(pair['input'])
                _, cover = _prediction(info)
                f = _features(info, cover)
                features.append(f)
                input_shapes.add(pair['input'].shape)
                output_shapes.add(pair['output'].shape)
                topologies.add((f['rooms'], f['contours'], f['depth']))
                roles = {info['bg']: 0, info['ink']: 1}
                normalized.add((pair['output'].shape, tuple(roles.setdefault(int(c), len(roles)) for c in pair['output'].flat)))
    assert len({_puzzle_key(p) for p in puzzles}) == 42
    assert len(normalized) >= 50 and len(input_shapes) >= 25 and len(output_shapes) >= 15
    assert len(topologies) >= 4
    assert {f['rooms'] for f in features} >= {1, 2, 3}
    assert {f['depth'] for f in features} >= {1, 2, 3}
    assert max(f['demand'] for f in features) >= 15
    assert sum(f['distorted'] > 0 for f in features) >= 40
    assert sum(f['local_ambiguous'] for f in features) >= 42
    assert any(f['supplied'] == f['used'] for f in features)
    assert any(f['supplied'] > f['used'] for f in features)
    assert len({f['shapes'] for f in features}) >= 30
    assert _puzzle_key(generate_puzzle(0)) == _puzzle_key(puzzles[0])
    # Semantic role permutation preserves the answer; source positions do not matter.
    permutation = np.random.default_rng(900).permutation(10)
    for p in puzzles[:6]:
        pair = p['test'][0]
        assert np.array_equal(solve(permutation[pair['input']]), permutation[pair['output']])
    # A topology-preserving change of metric size/spacing changes no answer.
    p = puzzles[0]
    pair = next(x for x in p['train'] if _decode(x['input'])['tree']['children'] is None)
    info = _decode(pair['input'])
    # Stretch a row and a column that contain only scene/background, never tiles.
    root = info['records'][info['root']]
    y0, x0 = root['lo']; y1, x1 = root['hi']
    altered = pair['input'].copy()
    if altered.shape[0] < 30:
        row = next((y for y in range(int(y0), int(y1)+1)
                    if set(int(c) for c in altered[y]) <= {info['bg'], info['ink']}), None)
        if row is not None:
            altered = np.insert(altered, row, altered[row], axis=0)
            assert np.array_equal(solve(altered), pair['output'])
    # Omitting A's remeshing changes dimensions/area: metric terminal components
    # are not the compressed demand cells. Omitting B leaves drawing-colored dots.
    blobs = [r for r in info['records'] if not r['contour'] and r['colors'] == {info['ink']}]
    assert sum(len(r['cells']) for r in blobs) > sum(int(m.sum()) for m in info['boards'])
    marker_only = _render_tree(info['tree'], [np.where(m, info['ink'], info['bg']) for m in info['boards']],
                               info['bg'], info['ink'])
    assert not np.array_equal(marker_only, pair['output'])
    # Keep scene/topology fixed and perturb the uniquely supplied, used big tile.
    # Its dimensions and legal placements stay fixed; its copied pattern changes.
    resource = next(x for x in p['train'] if
                    _features(_decode(x['input']), _prediction(_decode(x['input']))[1])['local_ambiguous'])
    resource_info = _decode(resource['input'])
    _, resource_cover = _prediction(resource_info)
    used_groups = {place[0] for place in next(iter(resource_cover['outcomes'].values()))}
    big_tile = next(g['tile'] for i, g in enumerate(resource_cover['groups'])
                    if i in used_groups and g['count'] == 1 and min(g['tile'].shape) >= 2)
    changed = resource['input'].copy()
    source = next(r for r in resource_info['records'] if resource_info['ink'] not in r['colors']
                  and np.array_equal(changed[r['lo'][0]:r['hi'][0]+1, r['lo'][1]:r['hi'][1]+1], big_tile))
    y, x = source['lo']
    changed[y, x] = next(c for c in range(10) if c not in
                         (resource_info['bg'], resource_info['ink'], int(changed[y, x])))
    changed_output = solve(changed)
    assert changed_output.shape == resource['output'].shape
    assert not np.array_equal(changed_output, resource['output'])
    # Shared inventory counterexample: full room can take a big tile or two
    # strips, but the second room consumes both strips. Unlimited reuse is wrong.
    big = np.array([[2, 3], [4, 5]], np.int8)
    strip = np.array([[6, 7]], np.int8)
    masks = [np.ones((2, 2), bool), np.ones((1, 4), bool)]
    tiles = [big, strip, strip]
    assert len(_cover(masks, tiles)['outcomes']) == 1
    assert len(_cover([masks[0]], tiles)['outcomes']) == 2
    assert len(_cover(masks, tiles+[strip, strip])['outcomes']) == 2
    assert not _cover(masks, [strip, strip])['outcomes']
    # Pixel-equivalent exchange of duplicates is accepted; rotation is rejected.
    assert len(_cover([np.ones((1, 4), bool)], [strip, strip])['outcomes']) == 1
    assert not _cover([np.ones((2, 1), bool)], [strip])['outcomes']
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            _counts(nt, ne)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError('illegal counts accepted')
    for nt, ne in ((2, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(2026, nt, ne))
    # Returned labels must satisfy the independently checked conservation/frame invariants.
    corrupted = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in ('train', 'test')}
    corrupted['test'][0]['output'][0, 0] = _decode(corrupted['test'][0]['input'])['bg']
    try:
        validate_generated_puzzle(corrupted)
    except ValueError:
        pass
    else:
        raise AssertionError('bad frame accepted')
    return dict(seeds=42, pairs=len(features), normalized_outputs=len(normalized),
                input_shapes=len(input_shapes), output_shapes=len(output_shapes),
                topologies=len(topologies), demand_range=[min(f['demand'] for f in features), max(f['demand'] for f in features)],
                distorted_pairs=sum(f['distorted'] > 0 for f in features),
                locally_ambiguous_pairs=sum(f['local_ambiguous'] for f in features),
                semantic_verification='generator-owned invariants and metamorphic checks')


if __name__ == '__main__':
    print(json.dumps(self_test(), separators=(',', ':')))
