"""Keyed chambers: translation jigsaws joined by one-step material identities.

Only the standard library and NumPy are used. The solver consumes just a grid;
neither source positions (except the anchor) nor generation choices are clues.
"""
from __future__ import annotations

import copy
import json
from collections import deque

import numpy as np

STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
CORNERS = ("TL", "TR", "BL", "BR")
SIDES = ("top", "right", "bottom", "left")


def _grid(value):
    a = np.asarray(value)
    if (a.ndim != 2 or not np.issubdtype(a.dtype, np.integer)
            or any(not 1 <= d <= 30 for d in a.shape)
            or np.any((a < 0) | (a > 9))):
        raise ValueError("expected a 1..30 integer grid with colors 0..9")
    return a.astype(np.int8, copy=True)


def _components(mask, diagonal=False):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        cells = [start]
        for r, c in cells:
            steps = STEPS + ((-1, -1), (-1, 1), (1, -1), (1, 1)) if diagonal else STEPS
            for dr, dc in steps:
                q = (r + dr, c + dc)
                if q in todo:
                    todo.remove(q)
                    cells.append(q)
        result.append(cells)
    return result


def _marker_cells(point, corner):
    r, c = point
    dy = -1 if corner[0] == "T" else 1
    dx = -1 if corner[1] == "L" else 1
    return ((r + dy, c), (r, c + dx), (r + dy, c + dx))


def _extract(g):
    counts = np.bincount(g.ravel(), minlength=10)
    if np.count_nonzero(counts == counts.max()) != 1:
        raise ValueError("background must be uniquely most frequent")
    bg = int(counts.argmax())
    h, w = g.shape
    components = _components(g != bg)
    border = [p for p in components if any(r in (0, h-1) or c in (0, w-1) for r, c in p)]
    if len(border) != 1:
        raise ValueError("exactly one component, the key, may touch the canvas edge")
    cells = border[0]
    q = np.array(cells)
    r, c = q.min(0)
    y, x = q.max(0) + 1
    kh, kw = int(y-r), int(x-c)
    if len(cells) != kh*kw:
        raise ValueError("key must be a dense strip")
    if kh == 2 and kw >= 3 and r == 0:
        side, sources, fills = "top", g[r, c:x], g[r+1, c:x]
    elif kh == 2 and kw >= 3 and y == h:
        side, sources, fills = "bottom", g[y-1, c:x], g[y-2, c:x]
    elif kw == 2 and kh >= 3 and c == 0:
        side, sources, fills = "left", g[r:y, c], g[r:y, c+1]
    elif kw == 2 and kh >= 3 and x == w:
        side, sources, fills = "right", g[r:y, x-1], g[r:y, x-2]
    else:
        raise ValueError("key must touch one edge along its long direction")
    # The short end does not touch another edge; the direction is unambiguous.
    if (kh == 2 and (c == 0 or x == w)) or (kw == 2 and (r == 0 or y == h)):
        raise ValueError("key endpoints must stay away from corners")
    sources, fills = list(map(int, sources)), list(map(int, fills))
    if len(set(sources)) != len(sources):
        raise ValueError("key sources must be distinct")
    mapping = dict(zip(sources, fills))
    key_mask = np.zeros(g.shape, bool)
    for p in cells:
        key_mask[p] = True
    markers = []
    for color in range(10):
        q = np.argwhere((g == color) & ~key_mask)
        if color == bg or color in sources + fills or len(q) != 3:
            continue
        a, b = q.min(0)
        z, v = q.max(0)
        if z-a != 1 or v-b != 1:
            continue
        missing = {(int(a), int(b)), (int(a), int(v)), (int(z), int(b)), (int(z), int(v))}
        missing -= set(map(tuple, q.tolist()))
        point = missing.pop()
        if g[point] in (bg, color) or key_mask[point]:
            continue
        corner = {(1, 1): "TL", (1, 0): "TR", (0, 1): "BL", (0, 0): "BR"}[(point[0]-a, point[1]-b)]
        markers.append((color, point, corner))
    if len(markers) != 1:
        raise ValueError("expected one distinct three-cell L marker")
    marker, point, corner = markers[0]
    payload = (g != bg) & (g != marker) & ~key_mask
    pieces, anchor = [], None
    for cells in _components(payload):
        q = np.array(cells)
        r, c = q.min(0)
        y, x = q.max(0) + 1
        if len(cells) != (y-r)*(x-c):
            raise ValueError("payload pieces must be full rectangles")
        if point in cells:
            anchor = len(pieces)
        pieces.append((g[r:y, c:x].copy(), (int(r), int(c))))
    if not 3 <= len(pieces) <= 7 or anchor is None:
        raise ValueError("expected 3..7 rectangular pieces and an anchor")
    palette = {int(v) for m, _ in pieces for v in m.flat}
    carrier_candidates = palette - set(sources + fills)
    if len(carrier_candidates) != 1:
        raise ValueError("substrate must be the sole tile color absent from both key lanes")
    carrier = carrier_candidates.pop()
    if carrier in (bg, marker):
        raise ValueError("role colors must be locally distinct")
    m, (ar, ac) = pieces[anchor]
    expected = (ar + (m.shape[0]-1 if corner[0] == "B" else 0),
                ac + (m.shape[1]-1 if corner[1] == "R" else 0))
    if point != expected:
        raise ValueError("marker must mark a tile corner")
    return dict(bg=bg, key_mask=key_mask, side=side, mapping=mapping,
                marker=marker, point=point, corner=corner, pieces=pieces,
                anchor=anchor, carrier=carrier)


def _assemble(g, info, raw=False):
    """Complete enumeration, pruning with visible outer rim and seam evidence."""
    pieces = info["pieces"]
    carrier, anchor, corner, point = (info[k] for k in ("carrier", "anchor", "corner", "point"))
    mapping = {} if raw else info["mapping"]
    lut = np.arange(10, dtype=np.int8)
    for s, f in mapping.items():
        lut[s] = f  # Simultaneous, single lookup; no recursive normalization.
    normalized = [lut[m] for m, _ in pieces]
    area = sum(m.size for m, _ in pieces)
    winners = {}
    nodes = 0
    for h in range(2, g.shape[0]-1):
        if area % h:
            continue
        w = area // h
        oy = point[0] - (h-1 if corner[0] == "B" else 0)
        ox = point[1] - (w-1 if corner[1] == "R" else 0)
        if not (w >= 2 and 1 <= oy and 1 <= ox and oy+h < g.shape[0] and ox+w < g.shape[1]):
            continue
        if info["key_mask"][max(0,oy-1):oy+h+1, max(0,ox-1):ox+w+1].any():
            continue
        if any(m.shape[0] > h or m.shape[1] > w for m, _ in pieces):
            continue
        work = np.full((h, w), -1, np.int8)
        native = np.full((h, w), -1, np.int8)
        owners = np.full((h, w), -1, np.int8)
        positions = [None]*len(pieces)
        full = (1 << area)-1
        bits = [sum(((1 << m.shape[1])-1) << (r*w) for r in range(m.shape[0])) for m, _ in pieces]

        def fits(i, r, c):
            m = normalized[i]
            ph, pw = m.shape
            if r < 0 or c < 0 or r+ph > h or c+pw > w:
                return False
            if r == 0 and np.any(m[0] != carrier):
                return False
            if c == 0 and np.any(m[:, 0] != carrier):
                return False
            if r+ph == h and np.any(m[-1] != carrier):
                return False
            if c+pw == w and np.any(m[:, -1] != carrier):
                return False
            for a, b in ((work[r-1, c:c+pw], m[0]) if r else ((), ()),
                         (work[r+ph, c:c+pw], m[-1]) if r+ph < h else ((), ()),
                         (work[r:r+ph, c-1], m[:, 0]) if c else ((), ()),
                         (work[r:r+ph, c+pw], m[:, -1]) if c+pw < w else ((), ())):
                if len(a):
                    used = a >= 0
                    if np.any(a[used] != b[used]):
                        return False
            return True

        def put(i, r, c, remove=False):
            ph, pw = pieces[i][0].shape
            work[r:r+ph, c:c+pw] = -1 if remove else normalized[i]
            native[r:r+ph, c:c+pw] = -1 if remove else pieces[i][0]
            owners[r:r+ph, c:c+pw] = -1 if remove else i
            positions[i] = None if remove else (r+oy, c+ox)

        ar, ac = pieces[anchor][1][0]-oy, pieces[anchor][1][1]-ox
        if not fits(anchor, ar, ac):
            continue
        put(anchor, ar, ac)
        initial = bits[anchor] << (ar*w+ac)

        def visit(occupied, remaining):
            nonlocal nodes
            nodes += 1
            if nodes > 250000:
                raise ValueError("assembly search exceeded its complete-search budget")
            if not remaining:
                key = (h, w, oy, ox, native.tobytes())
                winners[key] = (native.copy(), owners.copy(), (oy, ox), tuple(positions))
                if len(winners) > 1:
                    raise ValueError("distinct compatible assemblies")
                return
            free = full ^ occupied
            r, c = divmod((free & -free).bit_length()-1, w)
            for i in remaining:
                if not fits(i, r, c):
                    continue
                bit = bits[i] << (r*w+c)
                if bit & occupied:
                    continue
                put(i, r, c)
                visit(occupied | bit, tuple(j for j in remaining if j != i))
                put(i, r, c, remove=True)

        visit(initial, tuple(i for i in range(len(pieces)) if i != anchor))
    if not winners:
        raise ValueError("no compatible rectangular assembly")
    return next(iter(winners.values()))


def _cavities(board, carrier, mapping):
    regions = []
    h, w = board.shape
    for cells in _components(board == carrier):
        boundary = set()
        edge = False
        for r, c in cells:
            edge |= r in (0, h-1) or c in (0, w-1)
            for dr, dc in STEPS:
                y, x = r+dr, c+dc
                if 0 <= y < h and 0 <= x < w and board[y, x] != carrier:
                    boundary.add(int(board[y, x]))
        eligible = not edge and bool(boundary) and boundary <= mapping.keys()
        destinations = {mapping[c] for c in boundary} if eligible else set()
        fill = next(iter(destinations)) if len(destinations) == 1 else None
        regions.append(dict(cells=cells, boundary=boundary, edge=edge, fill=fill))
    return regions


def _render(g, info, assembly):
    board, _, (r, c), _ = assembly
    painted = board.copy()
    for region in _cavities(board, info["carrier"], info["mapping"]):
        if region["fill"] is not None:
            for p in region["cells"]:
                painted[p] = region["fill"]
    out = np.full(g.shape, info["bg"], np.int8)
    out[info["key_mask"]] = g[info["key_mask"]]
    out[r:r+board.shape[0], c:c+board.shape[1]] = painted
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    g = _grid(input_grid)
    info = _extract(g)
    return _render(g, info, _assemble(g, info))


def _dilate(mask):
    result = mask.copy()
    for dr, dc in STEPS:
        a = (slice(max(0, dr), mask.shape[0]+min(0, dr)),
             slice(max(0, dc), mask.shape[1]+min(0, dc)))
        b = (slice(max(0, -dr), mask.shape[0]+min(0, -dr)),
             slice(max(0, -dc), mask.shape[1]+min(0, -dc)))
        result[a] |= mask[b]
    return result


def _room(rng, kind, small=False):
    if kind == "diamond":
        radius = int(rng.integers(1, 3))
        rr, cc = np.indices((radius*2+1, radius*2+1))
        return (abs(rr-radius)+abs(cc-radius) == radius)
    h, w = (1, 1) if small else (int(rng.integers(1, 4)), int(rng.integers(2, 5)))
    hole = np.ones((h, w), bool)
    if kind == "profile":
        h, w = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        hole = np.ones((h, w), bool)
        hole[-1, -1] = False
        if w >= 3 and rng.random() < .5:
            hole[0, 0] = False
    hole = np.pad(hole, 1)
    wall = _dilate(hole) & ~hole
    # Cardinal dilation admits diagonal junctions while background stays 4-connected.
    if kind == "rectangle" or kind == "open":
        wall[0, :] = wall[-1, :] = True
        wall[:, 0] = wall[:, -1] = True
    if kind == "open":
        wall[0, 1] = False
    return wall


def _logical_picture(rng, lesson):
    main_kind = "profile" if lesson == 0 else "diamond" if lesson == 1 else str(rng.choice(("profile", "diamond", "rectangle")))
    for _ in range(50):
        h, w = int(rng.integers(6, 12)), int(rng.integers(7, 13))
        if h*w > 110:
            continue
        if lesson == 0:
            kinds = [(1, main_kind), (1, "open")]
            if rng.random() < .5:
                kinds.append((2, "rectangle"))
        elif lesson == 1:
            kinds = [(1, main_kind), (2, "rectangle"), (3, "rectangle")]
        else:
            kinds = [(1, main_kind), (2, "rectangle")]
            number = int(rng.choice((2, 3, 4), p=(.25, .6, .15)))
            if number >= 3:
                kinds.append((3, "rectangle"))
            if number == 4:
                kinds.append((2, "open"))
        logical = np.zeros((h, w), np.int8)
        blocked = np.zeros((h, w), bool)
        placements = []
        for index, (label, kind) in enumerate(kinds):
            wall = _room(rng, kind, small=index > 0)
            ph, pw = wall.shape
            options = [(r, c) for r in range(1, h-ph) for c in range(1, w-pw)
                       if not blocked[r:r+ph, c:c+pw].any()]
            if not options:
                break
            r, c = options[int(rng.integers(len(options)))]
            logical[r:r+ph, c:c+pw][wall] = label
            blocked[max(0, r-1):r+ph+1, max(0, c-1):c+pw+1] = True
            placements.append((r, c, wall, label, kind))
        else:
            return logical, placements
    return None


def _partition(rng, logical, placements, count):
    h, w = logical.shape
    # The first cut passes through the main hole. Subsequent recursive cuts vary
    # widths, heights and long-to-short contacts without changing material seams.
    r, c, wall, _, _ = placements[0]
    local_regions = _cavities(wall.astype(np.int8), 0, {1: 1})
    holes = [p["cells"] for p in local_regions if p["fill"] is not None]
    if not holes:
        return None
    cells = [(r+y, c+x) for y, x in max(holes, key=len)]
    choices = [(axis, cut) for axis, length in enumerate((h, w))
               for cut in range(2, length-1)
               if min(p[axis] for p in cells) < cut <= max(p[axis] for p in cells)]
    if not choices:
        return None
    axis, cut = choices[int(rng.integers(len(choices)))]
    rects = [(0, 0, cut, w), (cut, 0, h-cut, w)] if axis == 0 else [(0, 0, h, cut), (0, cut, h, w-cut)]
    for _ in range(count-2):
        options = [(i, a, k) for i, (_, _, ph, pw) in enumerate(rects)
                   for a, length in enumerate((ph, pw)) for k in range(2, length-1)]
        if not options:
            return None
        # Favor larger pieces, with all legal cut locations still available.
        weights = np.array([rects[i][2]*rects[i][3] for i, _, _ in options], float)
        i, a, k = options[int(rng.choice(len(options), p=weights/weights.sum()))]
        y, x, ph, pw = rects.pop(i)
        rects.extend([(y, x, k, pw), (y+k, x, ph-k, pw)] if a == 0
                     else [(y, x, ph, k), (y, x+k, ph, pw-k)])
    labels = np.full(logical.shape, -1, np.int8)
    for i, (y, x, ph, pw) in enumerate(rects):
        labels[y:y+ph, x:x+pw] = i
    return labels, rects, axis, cut


def _key(g, side, sources, fills, start):
    for i, (s, f) in enumerate(zip(sources, fills)):
        p = start+i
        if side == "top":
            g[0, p], g[1, p] = s, f
        elif side == "bottom":
            g[-1, p], g[-2, p] = s, f
        elif side == "left":
            g[p, 0], g[p, 1] = s, f
        else:
            g[p, -1], g[p, -2] = s, f


def _halo_clear(blocked, r, c, h, w):
    return not blocked[max(0, r-1):r+h+1, max(0, c-1):c+w+1].any()


def _make_pair(rng, side, corner, count, lesson):
    for attempt in range(220):
        data = _logical_picture(rng, lesson)
        if data is None:
            continue
        logical, placements = data
        partition = _partition(rng, logical, placements, count)
        if partition is None:
            continue
        labels, rects, axis, cut = partition
        colors = list(map(int, rng.permutation(10)))
        bg, marker, carrier = colors[:3]
        source_count = int(rng.integers(3, 6))
        sources = colors[3:3+source_count]
        outside = colors[3+source_count:]
        style = "chain" if lesson == 0 else "self" if lesson == 1 else str(rng.choice(("chain", "self", "separate")))
        f0 = sources[2] if style == "chain" else sources[0] if style == "self" else outside[1]
        f1 = outside[0]
        mapping = {s: int(rng.choice((f0, f1))) for s in sources}
        mapping[sources[0]] = mapping[sources[1]] = f0
        mapping[sources[2]] = f1
        mixed = lesson == 1 or (lesson == 2 and rng.random() < .4)
        types = logical.repeat(2, 0).repeat(2, 1)
        mixed_top = np.zeros(types.shape, bool)
        if mixed:
            mr, mc, mwall, _, _ = placements[1]
            mixed_top[2*mr:2*mr+2, 2*mc:2*(mc+mwall.shape[1])] = True
        board = np.full(types.shape, carrier, np.int8)
        physical_labels = np.full(types.shape, -1, np.int8)
        for i, (r, c, ph, pw) in enumerate(rects):
            native = sources[0] if (r if axis == 0 else c) < cut else sources[1]
            y0, x0 = (2*r-1 if r else 0), (2*c-1 if c else 0)
            y1 = 2*(r+ph)-(1 if r+ph < logical.shape[0] else 0)
            x1 = 2*(c+pw)-(1 if c+pw < logical.shape[1] else 0)
            patch = board[y0:y1, x0:x1]
            local_types = types[y0:y1, x0:x1]
            patch[local_types == 1] = native
            patch[local_types == 2] = sources[2]
            patch[(local_types == 2) & mixed_top[y0:y1, x0:x1]] = sources[0]
            patch[local_types == 3] = f1
            physical_labels[y0:y1, x0:x1] = i
        # Cut THROUGH duplicated logical cells. Facing seam cells therefore
        # share material identity, even when their native colors differ.
        labels = physical_labels
        if rng.random() < .5:
            board, labels = board.T.copy(), labels.T.copy()
        if rng.random() < .5:
            board, labels = board[::-1].copy(), labels[::-1].copy()
        if rng.random() < .5:
            board, labels = board[:, ::-1].copy(), labels[:, ::-1].copy()
        pieces = []
        for i in range(count):
            q = np.argwhere(labels == i)
            r, c = q.min(0)
            y, x = q.max(0)+1
            pieces.append((board[r:y, c:x].copy(), (int(r), int(c))))
        h, w = board.shape
        base_h = max(h+4, int(np.ceil(np.sqrt(board.size*2.2))))
        base_w = max(w+4, int(np.ceil(board.size*2.2/base_h)))
        canvas = (min(30, base_h+int(rng.integers(0, 4))), min(30, base_w+int(rng.integers(0, 4))))
        g = np.full(canvas, bg, np.int8)
        order = rng.permutation(source_count)
        ordered = [sources[int(i)] for i in order]
        start = int(rng.integers(1, canvas[1 if side in ("top", "bottom") else 0]-source_count))
        _key(g, side, ordered, [mapping[s] for s in ordered], start)
        key_mask = g != bg
        origins = [(r, c) for r in range(1, canvas[0]-h) for c in range(1, canvas[1]-w)
                   if _halo_clear(key_mask, r, c, h, w)
                   and (r >= 2 if corner[0] == "T" else r+h <= canvas[0]-2)
                   and (c >= 2 if corner[1] == "L" else c+w <= canvas[1]-2)]
        if not origins:
            continue
        oy, ox = origins[int(rng.integers(len(origins)))]
        local_point = (h-1 if corner[0] == "B" else 0, w-1 if corner[1] == "R" else 0)
        point = (oy+local_point[0], ox+local_point[1])
        anchor = int(labels[local_point])
        m, (r, c) = pieces[anchor]
        old_anchor = (oy+r, ox+c)
        blocked = key_mask.copy()
        for p in _marker_cells(point, corner):
            if key_mask[max(0, p[0]-1):p[0]+2, max(0, p[1]-1):p[1]+2].any():
                break
            blocked[p] = True
        else:
            blocked[old_anchor[0]:old_anchor[0]+m.shape[0], old_anchor[1]:old_anchor[1]+m.shape[1]] = True
            scattered = None
            others = sorted((i for i in range(count) if i != anchor), key=lambda i: -pieces[i][0].size)
            for _ in range(15):
                busy = blocked.copy()
                pos = [None]*count
                pos[anchor] = old_anchor
                for i in others:
                    ph, pw = pieces[i][0].shape
                    options = [(r, c) for r in range(1, canvas[0]-ph) for c in range(1, canvas[1]-pw)
                               if _halo_clear(busy, r, c, ph, pw)]
                    if not options:
                        break
                    r, c = options[int(rng.integers(len(options)))]
                    pos[i] = (r, c)
                    busy[r:r+ph, c:c+pw] = True
                else:
                    scattered = pos
                    break
            if scattered is None:
                continue
            for (m, _), (r, c) in zip(pieces, scattered):
                g[r:r+m.shape[0], c:c+m.shape[1]] = m
            for p in _marker_cells(point, corner):
                g[p] = marker
            if int(np.bincount(g.ravel(), minlength=10).argmax()) != bg:
                continue
            try:
                info = _extract(g)
                assembly = _assemble(g, info)
                if not np.array_equal(assembly[0], board) or assembly[2] != (oy, ox):
                    continue
                out = _render(g, info, assembly)
                features = _features(info, assembly)
                if not features["cross_cut_fill"] or not features["mapped_seam"] or not features["hybrid_fill"]:
                    continue
                if lesson == 0 and not (features["open_wall"] and features["chain_fill"] and features["irregular"]):
                    continue
                if lesson == 1 and not (features["unlisted_hole"] and features["conflicting_hole"]
                                        and features["self_fill"] and features["diagonal_closure"]):
                    continue
            except ValueError:
                continue
            return {"input": g, "output": out}
    raise RuntimeError("bounded keyed-chamber construction failed")


def _features(info, assembly):
    board, owners, _, _ = assembly
    carrier, mapping = info["carrier"], info["mapping"]
    regions = _cavities(board, carrier, mapping)
    active = [r for r in regions if r["fill"] is not None]
    inert = [r for r in regions if not r["edge"] and r["fill"] is None]
    seam = 0
    for a, b, oa, ob in ((board[:, :-1], board[:, 1:], owners[:, :-1], owners[:, 1:]),
                         (board[:-1], board[1:], owners[:-1], owners[1:])):
        seam += int(np.count_nonzero((oa != ob) & (a != b)))
    exterior = next(r for r in regions if r["edge"])
    # Group the actual contour ink across diagonal contacts and native colors.
    # Material fragments of a closed conflicting contour are not open contours.
    open_wall = False
    diagonal = False
    for comp in _components(board != carrier, diagonal=True):
        if any(int(board[p]) not in mapping for p in comp):
            continue
        rows, cols = zip(*comp)
        if len(comp) <= 4:
            continue
        local = np.zeros((max(rows)-min(rows)+3, max(cols)-min(cols)+3), np.int8)
        for y, x in comp:
            local[y-min(rows)+1, x-min(cols)+1] = 1
        if not any(not reg["edge"] for reg in _cavities(local, 0, {1: 1})):
            open_wall = True
    # Compare 8-connected carrier escape with the intended 4-connected escape.
    seen = np.zeros(board.shape, bool)
    queue = deque(p for p in exterior["cells"])
    for p in exterior["cells"]:
        seen[p] = True
    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                y, x = r+dr, c+dc
                if 0 <= y < board.shape[0] and 0 <= x < board.shape[1] and board[y, x] == carrier and not seen[y, x]:
                    seen[y, x] = True
                    queue.append((y, x))
    diagonal = any(any(seen[p] for p in reg["cells"]) for reg in active)
    irregular = any(len(reg["cells"]) != (max(r for r, _ in reg["cells"])-min(r for r, _ in reg["cells"])+1)
                    * (max(c for _, c in reg["cells"])-min(c for _, c in reg["cells"])+1) for reg in active)
    return dict(piece_count=len(info["pieces"]), shape=board.shape, side=info["side"], corner=info["corner"],
                key_length=len(mapping), background=info["bg"], mapped_seam=seam,
                cross_cut_fill=any(len({int(owners[p]) for p in r["cells"]}) > 1 for r in active),
                hybrid_fill=any(len(r["boundary"]) > 1 for r in active),
                chain_fill=any(r["fill"] in mapping and mapping[r["fill"]] != r["fill"] for r in active),
                self_fill=any(any(mapping[c] == c for c in r["boundary"]) for r in active),
                inert_hole=bool(inert), irregular=irregular, diagonal_closure=diagonal,
                conflicting_hole=any(r["boundary"] <= mapping.keys()
                                     and len({mapping[c] for c in r["boundary"]}) > 1 for r in inert),
                unlisted_hole=any(not r["boundary"] <= mapping.keys() for r in inert),
                open_wall=open_wall, filled_regions=len(active), filled_area=sum(len(r["cells"]) for r in active),
                native_regions=regions)


def _count(value, name, low, high):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _count(num_train, "num_train", 2, 10), _count(num_test, "num_test", 1, 4)
    if nt <= ne:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    sides = [str(s) for s in rng.permutation(SIDES)]
    corners = [str(c) for c in rng.permutation(CORNERS)]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            side = sides[i % 2] if split == "train" else sides[2+i % 2]
            corner = corners[i % 2] if split == "train" else corners[2+i % 2]
            n = int(rng.integers(3, 6)) if split == "train" else int(rng.integers(6, 8))
            lesson = i if split == "train" and i < 2 else 2
            for _ in range(8):
                pair = _make_pair(rng, side, corner, n, lesson)
                fingerprint = (pair["input"].shape, pair["input"].tobytes())
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input construction failed")
    # Order is not a semantic cue.
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, features = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert g.shape == out.shape
            key = (g.shape, g.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            info = _extract(g)
            assembly = _assemble(g, info)
            assert np.array_equal(_render(g, info, assembly), out), "incorrect output"
            board, owners, (oy, ox), positions = assembly
            assert positions[info["anchor"]] == info["pieces"][info["anchor"]][1]
            assert all(out[p] == info["bg"] for p in _marker_cells(info["point"], info["corner"]))
            assert not np.any(out == info["marker"])
            assert np.array_equal(g[info["key_mask"]], out[info["key_mask"]])
            crop = out[oy:oy+board.shape[0], ox:ox+board.shape[1]]
            assert np.array_equal(crop[board != info["carrier"]], board[board != info["carrier"]]), "native ink changed"
            assert np.all(crop[[0, -1], :] == info["carrier"]) and np.all(crop[:, [0, -1]] == info["carrier"])
            assert sum(m.size for m, _ in info["pieces"]) == board.size
            assert np.all(owners >= 0)
            for i, ((m, _), (r, c)) in enumerate(zip(info["pieces"], positions)):
                assert np.array_equal(board[r-oy:r-oy+m.shape[0], c-ox:c-ox+m.shape[1]], m)
            f = _features(info, assembly)
            assert f["cross_cut_fill"] and f["hybrid_fill"] and f["mapped_seam"] > 0
            assert not np.array_equal(g, out)
            features[split].append(f)
    train, test = features["train"], features["test"]
    for lesson in ("open_wall", "unlisted_hole", "conflicting_hole", "irregular", "diagonal_closure", "chain_fill", "self_fill"):
        assert any(f[lesson] for f in train), "missing training witness: " + lesson
    assert len({f["side"] for f in train}) >= 2
    assert len({f["corner"] for f in train}) >= 2
    assert min(f["piece_count"] for f in test) > max(f["piece_count"] for f in train)


def _fingerprint(puzzle):
    return tuple((s, tuple((np.asarray(p[k]).shape, np.asarray(p[k], dtype=np.int8).tobytes())
                          for p in puzzle[s] for k in ("input", "output"))) for s in ("train", "test"))


def self_test():
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    assert len({_fingerprint(p) for p in puzzles}) == 42
    shapes, targets, palettes, partitions = set(), set(), set(), set()
    counts, sides, corners, backgrounds = set(), set(), set(), set()
    for puzzle in puzzles:
        for split in ("train", "test"):
            for pair in puzzle[split]:
                g = pair["input"]
                info = _extract(g)
                assembly = _assemble(g, info)
                board, owners, _, _ = assembly
                f = _features(info, assembly)
                assert np.array_equal(solve(g.copy()), pair["output"])
                assert f["cross_cut_fill"] and f["hybrid_fill"] and f["mapped_seam"]
                # Omitting the dictionary in joining cannot yield the intended board.
                try:
                    alternative = _assemble(g, info, raw=True)
                except ValueError:
                    pass
                else:
                    assert not np.array_equal(_render(g, info, alternative), pair["output"])
                # Filling before translating leaves the cross-cut room open.
                prefilled = board.copy()
                oy, ox = assembly[2]
                for (tile, _), (r, c) in zip(info["pieces"], assembly[3]):
                    patch = tile.copy()
                    for reg in _cavities(tile, info["carrier"], info["mapping"]):
                        if reg["fill"] is not None:
                            for p in reg["cells"]:
                                patch[p] = reg["fill"]
                    prefilled[r-oy:r-oy+tile.shape[0], c-ox:c-ox+tile.shape[1]] = patch
                painted = pair["output"][oy:oy+board.shape[0], ox:ox+board.shape[1]]
                assert not np.array_equal(prefilled, painted)
                shapes.add(g.shape)
                targets.add(board.shape)
                palettes.add(tuple(sorted(info["mapping"].items())))
                normalized = {}
                layout = tuple(normalized.setdefault(int(v), len(normalized)) for v in owners.flat)
                partitions.add((board.shape, layout))
                counts.add(f["piece_count"])
                sides.add(f["side"])
                corners.add(f["corner"])
                backgrounds.add(f["background"])
    assert len(shapes) >= 15 and len(targets) >= 12 and len(partitions) >= 100
    assert len(palettes) >= 100 and counts == set(range(3, 8))
    assert sides == set(SIDES) and corners == set(CORNERS) and backgrounds == set(range(10))
    assert _fingerprint(puzzles[0]) == _fingerprint(generate_puzzle(0))
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(123 + nt + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    damaged = copy.deepcopy(puzzles[0])
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except (AssertionError, ValueError):
        pass
    else:
        raise AssertionError("output mutation accepted")
    return dict(seeds=42, pairs=210, canvas_shapes=len(shapes), board_shapes=len(targets),
                partitions=len(partitions), mappings=len(palettes), piece_counts=sorted(counts))


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
