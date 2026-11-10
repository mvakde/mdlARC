"""Query-locked mosaics: seams and a complete queried line jointly fix a tiling.

No parent code, parent pairs, generation metadata, or training outputs are used
by solve. Packing memoization is keyed solely by visible occupied masks.
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import itertools

import numpy as np


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not all(1 <= n <= 30 for n in a.shape)
            or not np.issubdtype(a.dtype, np.integer)
            or not np.all((0 <= a) & (a <= 9))):
        raise ValueError("grids must be integer 2D arrays, sides 1..30, colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask):
    remaining = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        queue = [first]
        for r, c in queue:
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in remaining:
                    remaining.remove(p)
                    queue.append(p)
        result.append(queue)
    return result


def _key(a):
    return a.shape, a.tobytes()


def _trim(a, mask):
    q = np.argwhere(mask)
    r, c = q.min(0)
    y, x = q.max(0)+1
    return a[r:y, c:x]


def _marker_cells(point, corner):
    r, c = point
    dy = -1 if corner[0] == "T" else 1
    dx = -1 if corner[1] == "L" else 1
    return ((r+dy, c), (r, c+dx), (r+dy, c+dx))


def _parse(grid):
    """Infer query frame, anchor, slot pitch, pieces, ports, and exact keys."""
    bg = int(grid[0, 0])
    assert np.all(grid[[0, -1], :] == bg) and np.all(grid[:, [0, -1]] == bg)
    candidates = []
    for color in range(10):
        if color == bg:
            continue
        comps = _components(grid == color)
        frames, markers = [], []
        for cells in comps:
            q = np.array(cells)
            r, c = map(int, q.min(0))
            y, x = map(int, q.max(0)+1)
            h, w = y-r, x-c
            if len(cells) == 3 and (h, w) == (2, 2):
                missing = ({(r,c), (r,x-1), (y-1,c), (y-1,x-1)} - set(cells)).pop()
                if int(grid[missing]) not in (bg, color):
                    corner = {(1,1): "TL", (1,0): "TR", (0,1): "BL", (0,0): "BR"}[
                        (missing[0]-r, missing[1]-c)]
                    markers.append((missing, corner, tuple(cells)))
            if min(h, w) not in (6, 7) or max(h, w) <= min(h, w):
                continue
            ring = {(rr, cc) for rr in range(r, y) for cc in range(c, x)
                    if rr in (r, y-1) or cc in (c, x-1)}
            if set(cells) == ring:
                s = min(h, w)-2
                length = max(h, w)-2
                if length % s == 0 and 2 <= length//s <= 4:
                    frames.append((r, c, h, w, s, length//s, h < w))
        if len(frames) == len(markers) == 1:
            frame, mark = frames[0], markers[0]
            if int(np.count_nonzero(grid == color)) == 2*(frame[2]+frame[3]-2)+3:
                candidates.append((color, frame, mark))
    assert len(candidates) == 1, "one query outline and one external L must share a color"
    paint, frame, (point, corner, marker_cells) = candidates[0]
    r, c, h, w, s, k, horizontal = frame
    inside = grid[r+1:r+h-1, c+1:c+w-1]
    carrier = int(inside[0, 0])
    assert carrier not in (bg, paint)
    query = []
    for i in range(k):
        block = inside[:, i*s:(i+1)*s] if horizontal else inside[i*s:(i+1)*s, :]
        assert block.shape == (s, s)
        assert np.all(block[[0, -1], :] == carrier) and np.all(block[:, [0, -1]] == carrier)
        query.append(block[1:-1, 1:-1].copy())
    payload = (grid != bg) & (grid != paint)
    payload[r:r+h, c:c+w] = False
    pieces, tiles, owner, anchor = [], [], [], None
    for cells in _components(payload):
        q = np.array(cells)
        pr, pc = map(int, q.min(0))
        py, px = map(int, q.max(0)+1)
        assert (py-pr) % s == (px-pc) % s == 0
        mask = np.zeros(((py-pr)//s, (px-pc)//s), bool)
        ids = []
        for yy, xx in np.ndindex(mask.shape):
            rs, cs = slice(pr+yy*s, pr+(yy+1)*s), slice(pc+xx*s, pc+(xx+1)*s)
            occupied = payload[rs, cs]
            assert occupied.all() or not occupied.any(), "cuts follow whole slot boundaries"
            if occupied.all():
                mask[yy, xx] = True
                ids.append(len(tiles))
                tiles.append(grid[rs, cs].copy())
                owner.append(len(pieces))
        assert len(_components(mask)) == 1
        if point in cells:
            anchor = len(pieces)
            expected_point = (py-1 if corner[0] == "B" else pr,
                              px-1 if corner[1] == "R" else pc)
            assert point == expected_point
        pieces.append((mask, (pr, pc), tuple(ids)))
    assert anchor is not None and 3 <= len(pieces) <= 7
    assert 6 <= len(tiles) <= 16
    glyph_keys, shape_keys, inks, edges = [], [], [], []
    for tile in tiles:
        assert np.all(tile[np.ix_([0,s-1], [0,s-1])] == carrier)
        glyph = tile[1:-1, 1:-1]
        colors = np.unique(glyph[glyph != carrier])
        assert len(colors) == 1 and int(colors[0]) not in (bg, paint)
        glyph_keys.append(_key(glyph))
        shape_keys.append(_key(_trim(glyph != carrier, glyph != carrier)))
        inks.append(int(colors[0]))
        edges.append((tuple(map(int, tile[0])), tuple(map(int, tile[:, -1])),
                      tuple(map(int, tile[-1])), tuple(map(int, tile[:, 0]))))
    assert len(set(glyph_keys)) == len(tiles)
    assert len(set((_key(_trim(t[1:-1,1:-1], t[1:-1,1:-1] != carrier))) for t in tiles)) == len(tiles)
    lookup = {key: i for i, key in enumerate(glyph_keys)}
    assert all(_key(q) in lookup for q in query)
    selected_order = tuple(lookup[_key(q)] for q in query)
    assert len(set(selected_order)) == k
    # Sibling pieces have at least one background cell between their pixels.
    labels = np.full(grid.shape, -1, int)
    for i, cells in enumerate(_components(payload)):
        for p in cells:
            labels[p] = i
    for dy, dx in ((0,1), (1,0), (1,1), (1,-1)):
        a = labels[max(0,dy):grid.shape[0]+min(0,dy), max(0,dx):grid.shape[1]+min(0,dx)]
        b = labels[max(0,-dy):grid.shape[0]+min(0,-dy), max(0,-dx):grid.shape[1]+min(0,-dx)]
        assert not np.any((a >= 0) & (b >= 0) & (a != b))
    return dict(grid=grid, bg=bg, paint=paint, carrier=carrier, frame=frame,
                point=point, corner=corner, marker_cells=marker_cells, s=s,
                pieces=pieces, tiles=tiles, owner=np.array(owner), edges=edges, anchor=anchor,
                selected=frozenset(selected_order), query_order=selected_order,
                shapes=shape_keys, inks=inks)


@lru_cache(maxsize=64)
def _packings(keys, anchor, corner):
    """Complete translation-only packing search at slot scale (no color clues).

    The first uncovered cell must receive the first occupied cell of one unused
    piece. This remains complete for stepped masks whose first row has a notch.
    No solver search truncation or generation-order tie break is used.
    """
    masks = [np.frombuffer(data, np.uint8).reshape(shape).astype(bool) for shape, data in keys]
    counts = [int(m.sum()) for m in masks]
    offsets = np.cumsum([0]+counts)
    area, results = sum(counts), []
    for h in range(2, 5):
        if area % h:
            continue
        w = area//h
        if not 2 <= w <= 4 or any(m.shape[0] > h or m.shape[1] > w for m in masks):
            continue
        configs = []
        for m in masks:
            cells = [tuple(map(int, p)) for p in np.argwhere(m)]
            configs.append((cells, cells[0][1], sum(1 << (r*w+c) for r,c in cells)))
        ah, aw = masks[anchor].shape
        ar, ac = (h-ah if corner[0] == "B" else 0), (w-aw if corner[1] == "R" else 0)
        positions = [None]*len(masks)
        positions[anchor] = (ar, ac)
        full = (1 << area)-1

        def visit(occupied, remaining):
            if not remaining:
                mapping = np.full((h, w), -1, np.int16)
                for i, (r,c) in enumerate(positions):
                    for j, (y,x) in enumerate(configs[i][0]):
                        mapping[r+y, c+x] = int(offsets[i])+j
                mapping.setflags(write=False)
                results.append((h, w, tuple(positions), mapping))
                return
            free = full ^ occupied
            r, c = divmod((free & -free).bit_length()-1, w)
            for i in remaining:
                ph, pw = masks[i].shape
                _, first_col, bits = configs[i]
                x = c-first_col
                if x < 0 or r+ph > h or x+pw > w:
                    continue
                placed = bits << (r*w+x)
                if not placed & occupied:
                    positions[i] = (r, x)
                    visit(occupied | placed, tuple(j for j in remaining if j != i))
        visit(configs[anchor][2] << (ar*w+ac), tuple(i for i in range(len(masks)) if i != anchor))
    return tuple(results)


def _lines(mapping):
    return ([(frozenset(map(int,row)), "row", i) for i,row in enumerate(mapping)]
            + [(frozenset(map(int,col)), "col", i) for i,col in enumerate(mapping.T)])


def _line(mapping, selected):
    for ids, axis, index in _lines(mapping):
        if ids == selected:
            return axis, index
    return None


def _seams_match(candidate, owner, edges):
    m = candidate[3]
    for r,c in np.ndindex(m.shape):
        a = int(m[r,c])
        for y,x,side,facing in ((r,c+1,1,3), (r+1,c,2,0)):
            if y < m.shape[0] and x < m.shape[1]:
                b = int(m[y,x])
                if owner[a] != owner[b] and edges[a][side] != edges[b][facing]:
                    return False
    return True


def _origin(point, corner, h, w, s):
    return (point[0]-(h*s-1 if corner[0] == "B" else 0),
            point[1]-(w*s-1 if corner[1] == "R" else 0))


def _fits(info, candidate):
    h, w = candidate[:2]
    r, c = _origin(info['point'], info['corner'], h, w, info['s'])
    y, x = r+h*info['s'], c+w*info['s']
    if not (1 <= r < y < info['grid'].shape[0] and 1 <= c < x < info['grid'].shape[1]):
        return False
    qr, qc, qh, qw = info['frame'][:4]
    return y <= qr or qr+qh <= r or x <= qc or qc+qw <= c


def _analyze(grid):
    info = _parse(grid)
    keys = tuple(_key(mask.astype(np.uint8)) for mask, _, _ in info['pieces'])
    anchor = info['anchor']
    geometry = [p for p in _packings(keys, anchor, info['corner']) if _fits(info, p)]
    edge_valid = [p for p in geometry if _seams_match(p, info['owner'], info['edges'])]
    joint = [p for p in edge_valid if _line(p[3], info['selected']) is not None]
    assert len(joint) == 1, "query and seams must identify exactly one assembly"
    info.update(anchor=anchor, geometry=geometry, edge_valid=edge_valid, winner=joint[0],
                axis=_line(joint[0][3], info['selected'])[0],
                line_index=_line(joint[0][3], info['selected'])[1],
                without_seams=sum(_line(p[3], info['selected']) is not None for p in geometry))
    return info


def _render_solution(info):
    output = np.full(info['grid'].shape, info['bg'], np.int8)
    h, w, _, mapping = info['winner']
    r, c = _origin(info['point'], info['corner'], h, w, info['s'])
    for y,x in np.ndindex(mapping.shape):
        tid = int(mapping[y,x])
        tile = info['tiles'][tid].copy()
        if tid in info['selected']:
            tile[tile == info['carrier']] = info['paint']
        output[r+y*info['s']:r+(y+1)*info['s'], c+x*info['s']:c+(x+1)*info['s']] = tile
    return output


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve one input, using only its visible geometry, colors and query."""
    try:
        return _render_solution(_analyze(_grid(input_grid)))
    except AssertionError as exc:
        raise ValueError("not an unambiguous query-locked mosaic: "+str(exc)) from exc


def _partition(rng, h, w, count, stepped):
    rectangles = [(0,0,h,w)]
    for _ in range(count-1):
        options = [(i,d) for i,(_,_,ph,pw) in enumerate(rectangles)
                   for d,v in enumerate((ph,pw)) if v >= 2]
        i, d = options[int(rng.integers(len(options)))]
        r,c,ph,pw = rectangles.pop(i)
        cut = int(rng.integers(1, (ph,pw)[d]))
        rectangles.extend([(r,c,cut,pw), (r+cut,c,ph-cut,pw)] if d == 0 else
                          [(r,c,ph,cut), (r,c+cut,ph,pw-cut)])
    labels = np.full((h,w), -1, int)
    for i,(r,c,ph,pw) in enumerate(rectangles):
        labels[r:r+ph, c:c+pw] = i
    if stepped:
        for _ in range(10):
            r,c = int(rng.integers(h)), int(rng.integers(w))
            donor = int(labels[r,c])
            choices = [int(labels[y,x]) for y,x in ((r-1,c),(r+1,c),(r,c-1),(r,c+1))
                       if 0 <= y < h and 0 <= x < w and labels[y,x] != donor]
            if not choices:
                continue
            recipient = int(rng.choice(choices))
            labels[r,c] = recipient
            if len(_components(labels == donor)) != 1 or len(_components(labels == recipient)) != 1:
                labels[r,c] = donor
    return labels


def _model(labels, corner):
    pieces, intended, owner, offset = [], np.full(labels.shape, -1, int), [], 0
    for i in range(int(labels.max())+1):
        cells = np.argwhere(labels == i)
        r,c = map(int, cells.min(0)); y,x = map(int, cells.max(0)+1)
        mask = labels[r:y,c:x] == i
        for j,(yy,xx) in enumerate(np.argwhere(mask)):
            intended[r+yy,c+xx] = offset+j
            owner.append(i)
        offset += int(mask.sum())
        pieces.append(mask)
    point = (labels.shape[0]-1 if corner[0] == 'B' else 0,
             labels.shape[1]-1 if corner[1] == 'R' else 0)
    anchor = int(labels[point])
    geometry = _packings(tuple(_key(m.astype(np.uint8)) for m in pieces), anchor, corner)
    return dict(pieces=pieces, owner=owner, anchor=anchor, intended=intended, geometry=geometry)


def _good_queries(model, edges, axis, edge, contrast):
    geometry = model['geometry']
    eligible = [p for p in geometry if _seams_match(p, model['owner'], edges)]
    if len(eligible) < 2:
        return None
    gc = Counter(ids for p in geometry for ids,_,_ in _lines(p[3]))
    ec = Counter(ids for p in eligible for ids,_,_ in _lines(p[3]))
    good = [(p,ids,a,i) for p in eligible for ids,a,i in _lines(p[3])
            if gc[ids] >= 2 and ec[ids] == 1]
    first = [v for v in good if v[2] == axis and
             (v[3] in (0, v[0][0 if axis == 'row' else 1]-1)) == edge]
    if not contrast:
        return [(v, None) for v in first] or None
    result = []
    for v in first:
        other_axis = 'col' if axis == 'row' else 'row'
        for z in good:
            if (z[2] == other_axis and z[3] in (0,z[0][0 if other_axis == 'row' else 1]-1)
                    and not np.array_equal(z[0][3], v[0][3])
                    and len(z[1]) == len(v[1])):
                result.append((v,z))
    return result or None


def _design(rng, shape, corner, axis, edge, stepped, contrast=False):
    for _ in range(70):
        h,w = shape
        labels = _partition(rng, h, w, int(rng.integers(4, min(7,h*w)+1)), stepped)
        model = _model(labels, corner)
        if stepped and not any(m.sum() < m.size for m in model['pieces']):
            continue
        if len(model['geometry']) < 3 or len(model['geometry']) > 5000:
            continue
        intended = model['intended']
        for _ in range(12):
            ports = rng.choice(3, (h*w,4), p=[.55,.30,.15])
            for r,c in np.ndindex(intended.shape):
                t = int(intended[r,c])
                if c+1 < w:
                    ports[int(intended[r,c+1]),3] = ports[t,1]
                if r+1 < h:
                    ports[int(intended[r+1,c]),0] = ports[t,2]
            edges = [tuple(map(int,row)) for row in ports]
            queries = _good_queries(model, edges, axis, edge, contrast)
            if queries:
                queries = [q for q in queries if q[0][0][:2] == shape]
            if queries:
                choice = queries[int(rng.integers(len(queries)))]
                model.update(ports=ports, choices=choice)
                return model
    return None


@lru_cache(maxsize=2)
def _shape_bank(g):
    bank = {}
    for code in range(1, 1 << (g*g)):
        m = np.array([(code >> i) & 1 for i in range(g*g)], bool).reshape(g,g)
        crop = _trim(m, m)
        centered = np.zeros((g,g), bool)
        r,c = (g-crop.shape[0])//2, (g-crop.shape[1])//2
        centered[r:r+crop.shape[0], c:c+crop.shape[1]] = crop
        bank[_key(crop)] = centered
    return tuple(bank.values())


def _palette(rng):
    colors = list(map(int, rng.permutation(10)))
    nink = int(rng.integers(2,5))
    return colors[0], colors[1], colors[2], colors[3:3+nink], colors[3+nink:]


def _tiles(rng, model, s, palette):
    bg, paint, carrier, inks, seam_inks = palette
    count = len(model['owner'])
    bank = _shape_bank(s-2)
    first = sorted(model['choices'][0][1])
    outside = [i for i in range(count) if i not in first]
    selected = int(rng.choice(first))
    rng.shuffle(outside)
    shape = int(rng.integers(len(bank)))
    other_shape = (shape+int(rng.integers(1,len(bank)))) % len(bank)
    assignments = {selected: (shape,0), outside[0]: (shape,1), outside[1]: (other_shape,0)}
    pool = [(a,b) for a in range(len(bank)) for b in range(len(inks)) if (a,b) not in assignments.values()]
    rng.shuffle(pool)
    for tid in range(count):
        if tid not in assignments:
            assignments[tid] = pool.pop()
    # Only three distinct signatures are needed, deliberately reused. They may
    # be asymmetric; matching compares their displayed ordered pixel sequences.
    possible = [v for v in itertools.product([carrier]+seam_inks, repeat=s-2)
                if any(c != carrier for c in v)]
    idx = rng.choice(len(possible), 2, replace=False)
    codes = [(carrier,)*(s-2), possible[int(idx[0])], possible[int(idx[1])]]
    tiles = []
    for tid in range(count):
        tile = np.full((s,s), carrier, np.int8)
        a,b = assignments[tid]
        tile[1:-1,1:-1][bank[a]] = inks[b]
        for d in range(4):
            code = codes[int(model['ports'][tid,d])]
            if d == 0:
                tile[0,1:-1] = code
            elif d == 1:
                tile[1:-1,-1] = code
            elif d == 2:
                tile[-1,1:-1] = code
            else:
                tile[1:-1,0] = code
        tiles.append(tile)
    return tiles


def _piece_pixels(model, tiles, s, bg):
    pieces, offset = [], 0
    for m in model['pieces']:
        a = np.full((m.shape[0]*s, m.shape[1]*s), bg, np.int8)
        for y,x in np.argwhere(m):
            a[y*s:(y+1)*s, x*s:(x+1)*s] = tiles[offset]
            offset += 1
        pieces.append(a)
    return pieces


def _scatter(rng, pieces, anchor, point, corner, shape, s, bg):
    height, width = shape
    ph,pw = pieces[anchor].shape
    fixed = (point[0]-(ph-1 if corner[0] == 'B' else 0),
             point[1]-(pw-1 if corner[1] == 'R' else 0))
    marker = _marker_cells(point, corner)
    start = s+4
    order = sorted((i for i in range(len(pieces)) if i != anchor), key=lambda i: -pieces[i].size)
    for _ in range(10):
        positions = {anchor: fixed}
        occupied = np.zeros(shape, bool)
        r,c = fixed
        occupied[r:r+ph,c:c+pw] = pieces[anchor] != bg
        for p in marker:
            occupied[p] = True
        for i in order:
            a = pieces[i]; hh,ww = a.shape
            options = [(r,c) for r in range(start, height-hh) for c in range(1,width-ww)]
            forbidden = occupied.copy()
            for dy,dx in itertools.product((-1,0,1), repeat=2):
                forbidden[max(0,dy):height+min(0,dy), max(0,dx):width+min(0,dx)] |= occupied[
                    max(0,-dy):height+min(0,-dy), max(0,-dx):width+min(0,-dx)]
            found = False
            for j in rng.permutation(len(options)):
                r,c = options[int(j)]
                if not np.any(forbidden[r:r+hh,c:c+ww] & (a != bg)):
                    positions[i] = (r,c)
                    occupied[r:r+hh,c:c+ww] |= a != bg
                    found = True
                    break
            if not found:
                break
        if len(positions) == len(pieces):
            return [positions[i] for i in range(len(pieces))]
    return None


def _draw_pair(base, model, tiles, choice, frame_origin, point, corner, s, palette, ordered=False):
    bg, paint, carrier = palette[:3]
    candidate, selected, _, _ = choice
    # Board-coordinate order occasionally appears; most queries are permuted.
    board_order = [int(t) for t in candidate[3].ravel() if int(t) in selected]
    order = board_order if ordered else board_order[1:]+board_order[:1]
    k = len(order)
    qr,qc = frame_origin
    inp = base.copy()
    inp[qr:qr+s+2, qc:qc+k*s+2] = paint
    inp[qr+1:qr+s+1, qc+1:qc+k*s+1] = carrier
    for i,tid in enumerate(order):
        inp[qr+2:qr+s, qc+2+i*s:qc+s+i*s] = tiles[tid][1:-1,1:-1]
    output = np.full(inp.shape, bg, np.int8)
    h,w,_,mapping = candidate
    r,c = _origin(point, corner, h, w, s)
    for y,x in np.ndindex(mapping.shape):
        tid = int(mapping[y,x]); block = tiles[tid].copy()
        if tid in selected:
            block[block == carrier] = paint
        output[r+y*s:r+(y+1)*s, c+x*s:c+(x+1)*s] = block
    return {'input': inp, 'output': output}


def _make_pairs(rng, shape, corner, axis, edge, stepped, s, palette, contrast=False):
    for _ in range(8):
        model = _design(rng, shape, corner, axis, edge, stepped, contrast)
        if model is None:
            break
        cap = 4 if s == 4 else 3
        canvas = (int(rng.integers(max(26,s+4+cap*s+2),31)), int(rng.integers(max(23,cap*s+4),31)))
        low_y, high_y = ((s+4,canvas[0]-cap*s-1) if corner[0] == 'T' else
                         (s+4+cap*s-1,canvas[0]-3))
        low_x, high_x = ((2,canvas[1]-cap*s-1) if corner[1] == 'L' else
                         (2+cap*s-1,canvas[1]-3))
        point = (int(rng.integers(low_y,high_y+1)), int(rng.integers(low_x,high_x+1)))
        tiles = _tiles(rng, model, s, palette)
        pieces = _piece_pixels(model, tiles, s, palette[0])
        positions = _scatter(rng, pieces, model['anchor'], point, corner, canvas, s, palette[0])
        if positions is None:
            continue
        base = np.full(canvas, palette[0], np.int8)
        for a,(r,c) in zip(pieces,positions):
            block = base[r:r+a.shape[0], c:c+a.shape[1]]
            block[a != palette[0]] = a[a != palette[0]]
        for p in _marker_cells(point, corner):
            base[p] = palette[1]
        k = len(model['choices'][0][1])
        qwidth = k*s+2
        qcol = int(rng.integers(1,canvas[1]-qwidth))
        # The external L must remain a separate same-colored component.
        if any(0 <= p[0] <= s+3 and qcol-1 <= p[1] <= qcol+qwidth for p in _marker_cells(point,corner)):
            continue
        result = [_draw_pair(base, model, tiles, v, (1,qcol), point, corner, s, palette,
                             ordered=(j > 0 and rng.random() < .35))
                  for j,v in enumerate(model['choices']) if v is not None]
        if all(_pair_valid(p, edge=edge, stepped=stepped) for p in result):
            return result
    return _fallback(rng, corner, axis, edge, stepped, s, palette, contrast)


# Guaranteed construction-only escape hatch: small mask/port grammars, not
# saved input/output examples. All glyphs, palettes and scene transforms are
# freshly constructed; solve never reads these constants.
_FALLBACK = (
    ([[0,0,1],[0,0,2],[3,3,4]],
     [[1,0,0,0],[0,0,0,0],[0,1,0,1],[0,1,0,1],[0,1,0,0],
      [0,2,0,1],[0,2,0,1],[0,0,0,2],[0,2,0,0]],
     [(9,2),(9,15),(17,15),(23,2),(23,15)]),
    ([[1,2,3],[4,5,0],[4,0,0]],
     [[1,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0],[1,0,2,0],
      [0,0,1,0],[0,0,0,0],[0,0,1,2],[2,0,0,0]],
     [(17,15),(9,2),(9,10),(9,18),(17,2),(17,9)]),
)


def _fallback(rng, corner, axis, edge, stepped, s, palette, contrast):
    labels, ports, positions = _FALLBACK[int(stepped)]
    model = _model(np.array(labels), 'TL')
    edges = [tuple(row) for row in ports]
    queries = _good_queries(model, edges, axis, edge, contrast)
    if queries is None:  # Both winning axes are taught; a fallback may use either.
        queries = _good_queries(model, edges, 'col' if axis == 'row' else 'row', edge, contrast)
    assert queries is not None
    model.update(ports=np.array(ports), choices=queries[int(rng.integers(len(queries)))])
    tiles = _tiles(rng,model,s,palette)
    pieces = _piece_pixels(model,tiles,s,palette[0])
    base = np.full((30,30),palette[0],np.int8)
    for a,(r,c) in zip(pieces,positions):
        block = base[r:r+a.shape[0],c:c+a.shape[1]]
        block[a != palette[0]] = a[a != palette[0]]
    for p in _marker_cells((9,2),'TL'):
        base[p] = palette[1]
    pairs = [_draw_pair(base,model,tiles,v,(1,8),(9,2),'TL',s,palette,ordered=j > 0)
             for j,v in enumerate(model['choices']) if v is not None]
    for pair in pairs:
        for name in pair:
            if corner[0] == 'B':
                pair[name] = pair[name][::-1].copy()
            if corner[1] == 'R':
                pair[name] = pair[name][:,::-1].copy()
        assert _pair_valid(pair,edge=edge,stepped=stepped)
    return pairs


def _pair_valid(pair, *, edge=None, stepped=None):
    try:
        info = _analyze(pair['input'])
        assert np.array_equal(_render_solution(info),pair['output'])
        assert len(info['edge_valid']) >= 2 and info['without_seams'] >= 2
        if edge is not None:
            length = info['winner'][0 if info['axis'] == 'row' else 1]
            assert (info['line_index'] in (0,length-1)) == edge
        if stepped:
            assert any(m.sum() < m.size for m,_,_ in info['pieces'])
        return True
    except AssertionError:
        return False


def _orient(grid, orientation):
    return np.rot90(np.fliplr(grid) if orientation >= 4 else grid, orientation % 4).copy()


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, lo, hi in ((num_train,'num_train',2,10), (num_test,'num_test',1,4)):
        if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or not lo <= value <= hi:
            raise ValueError(name+' is outside its legal integer range')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    s = int(rng.choice([4,5],p=[.75,.25]))
    stepped = bool(rng.random() < .65)
    orientation = int(rng.integers(8))
    fixed_palette = _palette(rng)
    varied_palette = bool(rng.integers(2))
    corner = str(rng.choice(['TL','TR','BL','BR']))
    opposite = {'TL':'BR','BR':'TL','TR':'BL','BL':'TR'}[corner]
    holdout = num_train >= 3 and rng.random() < .6
    train = _make_pairs(rng,(3,3),corner,'row',True,stepped,s,fixed_palette,contrast=True)
    for i in range(2,num_train):
        pal = _palette(rng) if varied_palette else fixed_palette
        if i == 2:
            shape, axis = (2,3), 'col'
        elif i == 3 and s == 4:
            shape, axis = (3,4), 'row'
        else:
            shape = [(2,3),(3,3),(2,4),(3,4),(4,3)][int(rng.integers(5))] if s == 4 else [(2,3),(3,3)][int(rng.integers(2))]
            axis = str(rng.choice(['row','col']))
        co = opposite if holdout and i == 2 else str(rng.choice([corner,opposite])) if holdout else corner
        train.extend(_make_pairs(rng,shape,co,axis,True,stepped,s,pal))
    test = []
    taught_lengths = {len(_parse(p['input'])['selected']) for p in train}
    broad_sizes_taught = s == 4 and num_train >= 4 and {2,4} <= taught_lengths
    for _ in range(num_test):
        pal = _palette(rng) if varied_palette else fixed_palette
        shape = [(3,3),(3,4),(4,3),(4,4)][int(rng.integers(4))] if broad_sizes_taught else (3,3)
        co = str(rng.choice([c for c in ('TL','TR','BL','BR') if c not in (corner,opposite)])) if holdout else corner
        test.extend(_make_pairs(rng,shape,co,str(rng.choice(['row','col'])),False,stepped,s,pal))
    rng.shuffle(train)
    rng.shuffle(test)
    puzzle = {split:[{name:_orient(grid,orientation) for name,grid in pair.items()} for pair in pairs]
              for split,pairs in (('train',train),('test',test))}
    # Fresh token assignments and varying source layouts make duplicates unlikely;
    # a deterministic retry handles the finite possibility without modifying keys.
    seen = set()
    for split in ('train','test'):
        for i,pair in enumerate(puzzle[split]):
            key = _key(pair['input'])
            if key in seen:
                pal = _palette(rng)
                replacement = _make_pairs(rng,(3,3),corner,'row',split == 'train',stepped,s,pal)[0]
                puzzle[split][i] = {k:_orient(g,orientation) for k,g in replacement.items()}
                key = _key(puzzle[split][i]['input'])
            assert key not in seen
            seen.add(key)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle,dict) and set(puzzle) == {'train','test'}
    assert all(isinstance(puzzle[s],list) for s in puzzle)
    nt,ne = len(puzzle['train']),len(puzzle['test'])
    assert 2 <= nt <= 10 and 1 <= ne <= 4 and nt > ne
    seen, infos = set(), {'train':[], 'test':[]}
    for split in ('train','test'):
        for pair in puzzle[split]:
            assert isinstance(pair,dict) and set(pair) == {'input','output'}
            inp,out = _grid(pair['input']),_grid(pair['output'])
            assert inp.shape == out.shape and not np.array_equal(inp,out)
            assert _key(inp) not in seen
            seen.add(_key(inp))
            info = _analyze(inp)
            assert np.array_equal(_render_solution(info),out)
            assert len(info['edge_valid']) >= 2, 'without B there must be multiple assemblies'
            assert info['without_seams'] >= 2, 'without A seams the queried line must leave ambiguity'
            h,w,positions,mapping = info['winner']; s = info['s']
            r,c = _origin(info['point'],info['corner'],h,w,s)
            assert np.all(out[r:r+h*s,c:c+w*s] != info['bg'])
            assert np.count_nonzero(out != info['bg']) == len(info['tiles'])*s*s
            assert all(out[p] == info['bg'] for p in info['marker_cells'])
            qr,qc,qh,qw = info['frame'][:4]
            assert np.all(out[qr:qr+qh,qc:qc+qw] == info['bg'])
            ar,ac = positions[info['anchor']]
            assert (r+ar*s,c+ac*s) == info['pieces'][info['anchor']][1]
            # Check translation/color conservation separately, pixel by pixel.
            painted = 0
            for yy,xx in np.ndindex(mapping.shape):
                tid = int(mapping[yy,xx]); tile = info['tiles'][tid]
                block = out[r+yy*s:r+(yy+1)*s,c+xx*s:c+(xx+1)*s]
                unpainted = tile != info['carrier']
                assert np.array_equal(tile[unpainted],block[unpainted])
                expected = info['paint'] if tid in info['selected'] else info['carrier']
                assert np.all(block[~unpainted] == expected)
                if tid in info['selected']:
                    painted += int((~unpainted).sum())
            assert np.count_nonzero(out == info['paint']) == painted
            infos[split].append(info)
    # Generator teaching/shift guarantees, reconstructed from input pixels.
    assert {i['axis'] for i in infos['train']} == {'row','col'}
    assert all(i['line_index'] in (0,i['winner'][0 if i['axis'] == 'row' else 1]-1) for i in infos['train'])
    assert all(i['line_index'] not in (0,i['winner'][0 if i['axis'] == 'row' else 1]-1) for i in infos['test'])
    if any(m.sum() < m.size for i in infos['test'] for m,_,_ in i['pieces']):
        assert any(m.sum() < m.size for i in infos['train'] for m,_,_ in i['pieces'])
    # At least one demonstration must force colored identity rather than shape.
    assert any(i['shapes'][a] == i['shapes'][b] and i['inks'][a] != i['inks'][b]
               for i in infos['train'] for a in i['selected'] for b in range(len(i['tiles'])) if b not in i['selected'])
    if any(max(i['winner'][:2]) == 4 for i in infos['test']):
        assert {2,4} <= {len(i['selected']) for i in infos['train']}, 'teach variable line lengths before larger tests'
    # Query dependence must be visible in two lessons sharing all source pieces.
    contrasts = False
    for a,b in itertools.combinations(infos['train'],2):
        if a['grid'].shape != b['grid'].shape or a['axis'] == b['axis']:
            continue
        aa,bb = a['grid'].copy(),b['grid'].copy()
        for grid,info in ((aa,a),(bb,b)):
            qr,qc,qh,qw = info['frame'][:4]
            grid[qr:qr+qh,qc:qc+qw] = info['bg']
        if not np.array_equal(aa,bb):
            continue
        ao,bo = _render_solution(a),_render_solution(b)
        ao[ao == a['paint']] = a['carrier']
        bo[bo == b['paint']] = b['carrier']
        if not np.array_equal(ao,bo):
            contrasts = True
            break
    assert contrasts, 'two changed queries must lock different pictures from the same source pieces'


def measure_pair(pair):
    i = _analyze(_grid(pair['input']))
    h,w,positions,mapping = i['winner']
    labels = np.array(i['owner'])[mapping]
    rename = {}
    normalized = tuple(rename.setdefault(int(v),len(rename)) for v in labels.ravel())
    return dict(shape=tuple(np.asarray(pair['input']).shape), board=(h,w), pitch=i['s'],
                pieces=len(i['pieces']), stepped=sum(int(m.sum() < m.size) for m,_,_ in i['pieces']),
                corner=i['corner'], axis=i['axis'], line=i['line_index'],
                edge_candidates=len(i['edge_valid']), geometry_candidates=len(i['geometry']),
                without_seams=i['without_seams'], query_count=len(i['selected']),
                partition=(h,w,normalized), palette=tuple(map(int,np.unique(i['grid']))),
                roles=(i['bg'],i['carrier'],i['paint']),
                glyph_shapes=len(set(i['shapes'])))


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ('train','test'):
        digest.update(bytes([len(puzzle[split])]))
        for pair in puzzle[split]:
            for name in ('input','output'):
                a = _grid(pair[name]); digest.update(bytes(a.shape)); digest.update(a.tobytes())
    return digest.hexdigest()


def self_test():
    records, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_fingerprint(puzzle))
        records.extend(measure_pair(p) for split in ('train','test') for p in puzzle[split])
    assert len(signatures) == 42
    assert {r['pitch'] for r in records} == {4,5}
    assert {r['axis'] for r in records} == {'row','col'}
    assert {r['corner'] for r in records} == {'TL','TR','BL','BR'}
    assert len({r['shape'] for r in records}) >= 12
    assert len({r['board'] for r in records}) >= 5
    assert len({r['partition'] for r in records}) >= 60
    assert len({r['pieces'] for r in records}) >= 3
    assert len({r['roles'] for r in records}) >= 25
    assert any(r['stepped'] for r in records) and any(not r['stepped'] for r in records)
    assert all(r['edge_candidates'] >= 2 and r['without_seams'] >= 2 for r in records)
    p = generate_puzzle(924,5,3)
    assert _fingerprint(p) == _fingerprint(generate_puzzle(924,5,3))
    # Whole-scene D4 symmetry preserves exact query copies and seam order.
    pair = p['test'][0]
    for o in range(8):
        assert np.array_equal(solve(_orient(pair['input'],o)),_orient(pair['output'],o))
    # Query permutation cannot change the selected tiling or output picture.
    info = _parse(pair['input'])
    changed = pair['input'].copy()
    r,c,h,w,s,k,horizontal = info['frame']
    interior = changed[r+1:r+h-1,c+1:c+w-1]
    chunks = [interior[:,j*s:(j+1)*s].copy() if horizontal else interior[j*s:(j+1)*s,:].copy() for j in range(k)]
    for j,chunk in enumerate(chunks[1:]+chunks[:1]):
        if horizontal:
            interior[:,j*s:(j+1)*s] = chunk
        else:
            interior[j*s:(j+1)*s,:] = chunk
    assert np.array_equal(solve(changed),pair['output'])
    # Exercise guaranteed construction grammars at both pitches and all corners.
    rng = np.random.default_rng(382)
    for s in (4,5):
        for stepped in (False,True):
            for corner in ('TL','TR','BL','BR'):
                pal = _palette(rng)
                lessons = _fallback(rng,corner,'row',True,stepped,s,pal,True)
                challenge = _fallback(rng,corner,'col',False,stepped,s,pal,False)
                validate_generated_puzzle({'train':lessons,'test':challenge})
    for train in range(2,11):
        for test in range(1,min(4,train-1)+1):
            validate_generated_puzzle(generate_puzzle(510+train*10+test,train,test))
    for train,test in ((1,1),(11,1),(4,0),(6,5),(2,2),(3,4),(True,1),(4,1.5),(None,1)):
        try:
            generate_puzzle(0,train,test)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    for corruption in ('glyph','anchor','marker','duplicate'):
        bad = {split:[{k:g.copy() for k,g in pair.items()} for pair in p[split]] for split in p}
        if corruption == 'duplicate':
            bad['train'][1] = {k:g.copy() for k,g in bad['train'][0].items()}
        else:
            q = bad['test'][0]; i = _analyze(q['input']); out = q['output']
            if corruption == 'glyph':
                rr,cc = np.argwhere((out != i['bg']) & (out != i['paint']) & (out != i['carrier']))[0]
                out[rr,cc] = i['carrier']
            elif corruption == 'anchor':
                q['output'] = np.roll(out,1,axis=0)
            else:
                out[i['marker_cells'][0]] = i['paint']
        try:
            validate_generated_puzzle(bad)
        except (AssertionError,ValueError):
            continue
        raise AssertionError('validator accepted '+corruption)
    return {'seeds':42,'pairs':len(records),'distinct_puzzles':len(signatures),
            'canvas_shapes':len({r['shape'] for r in records}),
            'partitions':len({r['partition'] for r in records}),
            'boards':sorted({r['board'] for r in records}),
            'concept_ablation_checked':True}


if __name__ == '__main__':
    import json
    print(json.dumps(self_test(),separators=(',',':')))
