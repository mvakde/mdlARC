"""Socket jigsaws: reciprocal two-cell docking and anchored rectangular closure.

Only standard Python and NumPy are used. Prediction enumerates input-derived
quarter-turn placements; it never consults a seed or a construction partition.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import numpy as np


class SearchLimit(ValueError):
    """An unfinished enumeration cannot establish a unique answer."""


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid")
    return g.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = [start]
        for r, c in queue:
            for p in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
                if p in unseen:
                    unseen.remove(p)
                    queue.append(p)
        result.append(queue)
    return result


def _ports(matrix, details=False):
    """Recognize one-deep, two-wide tongues and their recessed socket bars.

    Both shapes have occupied shoulders. Tests use a padded occupancy mask so
    they apply to every straight boundary segment, including bounding edges.
    """
    found = np.zeros(matrix.shape, dtype=bool)
    tongue_count = socket_count = 0
    segments = []
    for turns in range(4):
        m = np.rot90(matrix >= 0, turns)
        flags = np.zeros(m.shape, dtype=bool)
        a = np.pad(m, 1)
        h, w = m.shape
        for r in range(1, h):
            for c in range(1, w-2):
                # A downward protrusion: .##. with #### immediately above.
                if (a[r+1, c+1:c+3].all() and not a[r+1, c]
                        and not a[r+1, c+3] and a[r, c:c+4].all()
                        and not a[r+2, c+1:c+3].any()):
                    flags[r, c:c+2] = True
                    tongue_count += 1
                    segment = np.zeros(m.shape, dtype=bool)
                    segment[r, c:c+2] = True
                    coords = np.argwhere(np.rot90(segment, -turns))
                    segments.append(("tongue", coords, ((1, 0), (0, -1), (-1, 0), (0, 1))[turns]))
        for r in range(h-1):
            for c in range(1, w-2):
                # A downward-facing inset bar: #### above #..#.
                if (a[r+1, c:c+4].all() and a[r+2, c]
                        and a[r+2, c+3] and not a[r+2, c+1:c+3].any()):
                    flags[r, c:c+2] = True
                    socket_count += 1
                    segment = np.zeros(m.shape, dtype=bool)
                    segment[r, c:c+2] = True
                    coords = np.argwhere(np.rot90(segment, -turns))
                    segments.append(("socket", coords, ((1, 0), (0, -1), (-1, 0), (0, 1))[turns]))
        found |= np.rot90(flags, -turns)
    result = (found, tongue_count, socket_count)
    return result+(segments,) if details else result


def _extract(g):
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    if counts[bg] <= g.size // 2:
        raise ValueError("background must occupy a strict majority")
    markers = []
    for color in range(10):
        q = np.argwhere(g == color)
        if color == bg or len(q) != 3:
            continue
        lo, hi = q.min(0), q.max(0)
        if tuple(hi-lo) != (1, 1):
            continue
        square = {(int(r), int(c)) for r in (lo[0], hi[0]) for c in (lo[1], hi[1])}
        point = (square - set(map(tuple, q.tolist()))).pop()
        if g[point] in (bg, color):
            continue
        corner = (int(point[0] == lo[0]), int(point[1] == lo[1]))
        # True means the marked payload corner is the bottom/right corner.
        markers.append((color, point, corner, q))
    if len(markers) != 1:
        raise ValueError("one unique external three-cell L marker is required")
    marker, point, corner, marker_cells = markers[0]
    pieces, anchor = [], None
    for cells in _components((g != bg) & (g != marker)):
        q = np.array(cells)
        r, c = map(int, q.min(0))
        h, w = map(int, q.max(0)-q.min(0)+1)
        m = np.full((h, w), -1, dtype=np.int8)
        m[q[:, 0]-r, q[:, 1]-c] = g[q[:, 0], q[:, 1]]
        if point in cells:
            anchor = len(pieces)
            if point != (r+(h-1 if corner[0] else 0), c+(w-1 if corner[1] else 0)):
                raise ValueError("marker is not at a bounding corner")
        pieces.append((m, (r, c)))
    if anchor is None or len(pieces) < 2:
        raise ValueError("missing anchored fragment or loose fragments")
    return bg, marker, point, corner, pieces, anchor, marker_cells


def _variants(m):
    seen, result = set(), []
    for turns in range(4):
        v = np.rot90(m, turns).copy()
        key = (v.shape, v.tobytes())
        if key in seen:
            continue
        seen.add(key)
        q = np.argwhere(v >= 0)
        ports, nt, ns = _ports(v)
        result.append((v, q, v[v >= 0], ports[v >= 0], turns, nt, ns))
    return result


def _docking_ok(pieces, placements):
    """Check reciprocal ports, including accidental silhouette-defined ones."""
    tongues, sockets = [], {}
    socket_count = 0
    for i, (m, _) in enumerate(pieces):
        r, c, turns = placements[i]
        v = np.rot90(m, turns)
        for kind, coords, direction in _ports(v, details=True)[3]:
            colors = v[coords[:, 0], coords[:, 1]]
            if colors[0] != colors[1]:
                return False
            absolute = coords+np.array((r, c))
            if kind == "tongue":
                tongues.append((absolute, direction, int(colors[0])))
            else:
                socket_count += 1
                for y, x in absolute:
                    sockets[(int(y), int(x), direction)] = int(colors[0])
    if len(tongues) < 2 or socket_count != len(tongues):
        return False
    return all(sockets.get((int(y+dy), int(x+dx), (-dy, -dx))) == color
               for coords, (dy, dx), color in tongues for y, x in coords)


def _search(g, mode="full", rotate=True, max_solutions=None, node_limit=None):
    """Complete exact cover over every area-factor rectangle fitting the canvas.

    full: every opposing boundary pixel agrees. ports: compare only contacts
    incident to recognizable tongue tips/socket bars (a deliberate ablation).
    A two-result cutoff proves ambiguity only; unique full results exhaust search.
    """
    bg, marker, point, corner, pieces, anchor, marker_cells = _extract(g)
    area = sum(int((m >= 0).sum()) for m, _ in pieces)
    variants = [_variants(m) if rotate and i != anchor else [_variants(m)[0]]
                for i, (m, _) in enumerate(pieces)]
    solutions = {}
    nodes = 0
    for h in range(1, g.shape[0]+1):
        if area % h:
            continue
        w = area // h
        oy = point[0]-(h-1 if corner[0] else 0)
        ox = point[1]-(w-1 if corner[1] else 0)
        if w > g.shape[1] or min(oy, ox) < 0 or oy+h > g.shape[0] or ox+w > g.shape[1]:
            continue
        ar, ac = pieces[anchor][1][0]-oy, pieces[anchor][1][1]-ox
        am, aq, av, ap, _, _, _ = variants[anchor][0]
        if ar < 0 or ac < 0 or ar+am.shape[0] > h or ac+am.shape[1] > w:
            continue
        board = np.full((h, w), -1, dtype=np.int8)
        owner = np.full((h, w), -1, dtype=np.int8)
        portboard = np.zeros((h, w), dtype=bool)
        board[aq[:, 0]+ar, aq[:, 1]+ac] = av
        owner[aq[:, 0]+ar, aq[:, 1]+ac] = anchor
        portboard[aq[:, 0]+ar, aq[:, 1]+ac] = ap
        full = (1 << (h*w))-1
        abit = sum(1 << (int(y+ar)*w+int(x+ac)) for y, x in aq)
        configs = []
        for vv in variants:
            entries = []
            for m, q, values, ports, turns, nt, ns in vv:
                ph, pw = m.shape
                if ph > h or pw > w:
                    continue
                bits = sum(1 << (int(y)*w+int(x)) for y, x in q)
                # Boundary contacts, including notch walls, not bbox edges.
                edges = []
                for k, (y, x) in enumerate(q):
                    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        yy, xx = int(y+dy), int(x+dx)
                        if not (0 <= yy < ph and 0 <= xx < pw) or m[yy, xx] < 0:
                            edges.append((yy, xx, int(values[k]), bool(ports[k])))
                entries.append((m, q, values, ports, turns, bits, int(q[0, 1]), edges))
            configs.append(entries)
        placements = {anchor: (ar+oy, ac+ox, 0)}

        def visit(occupied, remaining):
            nonlocal nodes
            nodes += 1
            if node_limit is not None and nodes > node_limit:
                raise SearchLimit("placement search exceeded its node budget")
            if not remaining:
                if not _docking_ok(pieces, placements):
                    return False
                out = np.full(g.shape, bg, dtype=np.int8)
                out[oy:oy+h, ox:ox+w] = board
                key = out.tobytes()
                if key not in solutions:
                    solutions[key] = (out, dict(placements), (h, w), owner.copy())
                return max_solutions is not None and len(solutions) >= max_solutions
            free = full ^ occupied
            pos = (free & -free).bit_length()-1
            y, x = divmod(pos, w)
            for i in remaining:
                for m, q, values, ports, turns, bits, first_x, edges in configs[i]:
                    left = x-first_x
                    ph, pw = m.shape
                    if left < 0 or y+ph > h or left+pw > w:
                        continue
                    bit = bits << (y*w+left)
                    if occupied & bit:
                        continue
                    good = True
                    for ey, ex, color, is_port in edges:
                        yy, xx = ey+y, ex+left
                        if 0 <= yy < h and 0 <= xx < w and board[yy, xx] >= 0:
                            if (mode == "full" or is_port or portboard[yy, xx]) and board[yy, xx] != color:
                                good = False
                                break
                    if not good:
                        continue
                    rr, cc = q[:, 0]+y, q[:, 1]+left
                    board[rr, cc] = values
                    owner[rr, cc] = i
                    portboard[rr, cc] = ports
                    placements[i] = (y+oy, left+ox, turns)
                    if visit(occupied | bit, tuple(j for j in remaining if j != i)):
                        return True
                    board[rr, cc] = -1
                    owner[rr, cc] = -1
                    portboard[rr, cc] = False
                    placements.pop(i)
            return False

        if visit(abit, tuple(i for i in range(len(pieces)) if i != anchor)):
            break
    return list(solutions.values()), nodes


@lru_cache(maxsize=96)
def _prediction(shape, data):
    g = np.frombuffer(data, dtype=np.int8).reshape(shape)
    # Two different rendered answers prove ambiguity immediately. A single
    # answer is returned only after exhaustive search, with no solver cutoff.
    results, nodes = _search(g, max_solutions=2)
    if len(results) != 1:
        raise ValueError(f"expected one output, found {len(results)}")
    out, places, target, labels = results[0]
    return out, places, target, labels, nodes


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Predict solely from visible fragments, sockets, patterns and anchor."""
    g = _grid(input_grid)
    return _prediction(g.shape, g.tobytes())[0].copy()


def _partition(rng, kind):
    if kind == "cycle":
        hh = ww = int(rng.integers(5, 8))
        labels = np.empty((2*hh, 2*ww), dtype=int)
        labels[:hh, :ww] = 0
        labels[:hh, ww:] = 1
        labels[hh:, ww:] = 2
        labels[hh:, :ww] = 3
    elif kind == "strips":
        n = int(rng.integers(3, 6))
        hh = int(rng.integers(5, 9))
        widths = rng.integers(4, 7, size=n)
        labels = np.repeat(np.arange(n)[None, :], hh, axis=0)
        labels = np.repeat(labels, widths, axis=1)
    elif kind == "staggered":
        widths = rng.integers(5, 8, size=3)
        hh = int(rng.integers(10, 15))
        labels = np.empty((hh, int(sum(widths))), dtype=int)
        x = 0
        k = 0
        for width in widths:
            cut = int(rng.integers(4, hh-3))
            if rng.random() < .25 and k > 0:
                labels[:, x:x+width] = k
                k += 1
            else:
                labels[:cut, x:x+width] = k
                labels[cut:, x:x+width] = k+1
                k += 2
            x += int(width)
    else:
        h, w = int(rng.integers(10, 15)), int(rng.integers(12, 19))
        rects = [(0, 0, h, w)]
        n = int(rng.integers(3, 7))
        for _ in range(n-1):
            choices = [(i, axis) for i, (_, _, ph, pw) in enumerate(rects)
                       for axis, size in enumerate((ph, pw)) if size >= 8]
            if not choices:
                break
            i, axis = choices[int(rng.integers(len(choices)))]
            r, c, ph, pw = rects.pop(i)
            cut = int(rng.integers(4, (ph if axis == 0 else pw)-3))
            rects.extend([(r, c, cut, pw), (r+cut, c, ph-cut, pw)] if axis == 0
                         else [(r, c, ph, cut), (r, c+cut, ph, pw-cut)])
        labels = np.empty((h, w), dtype=int)
        for i, (r, c, ph, pw) in enumerate(rects):
            labels[r:r+ph, c:c+pw] = i
    return labels


def _cut_sockets(rng, labels, cycle):
    """Transfer two adjacent seam cells to create a real tab/recess pair."""
    base = labels.copy()
    h, w = labels.shape
    seams = {}
    for r in range(h):
        for c in range(w):
            for dy, dx in ((1, 0), (0, 1)):
                yy, xx = r+dy, c+dx
                if yy < h and xx < w and base[r, c] != base[yy, xx]:
                    key = (int(base[r, c]), int(base[yy, xx]), dy, dx)
                    seams.setdefault(key, []).append((r, c))
    ports = []
    for (a, b, dy, dx), cells in sorted(seams.items()):
        cells = sorted(cells)
        if len(cells) < 4:
            continue
        start = int(rng.integers(1, len(cells)-2))
        segment = cells[start:start+2]
        give_to_first = bool(rng.integers(2))
        if cycle:
            # Cyclic congruent keyed corners: geometry alone leaves alternatives.
            give_to_first = (a, b) in ((0, 1), (1, 2))
            start = (len(cells)-2)//2
            segment = cells[start:start+2]
        target = a if give_to_first else b
        moved = [(r+dy, c+dx) if give_to_first else (r, c) for r, c in segment]
        behind = [(r+2*dy, c+2*dx) if give_to_first else (r-dy, c-dx) for r, c in segment]
        if any(not (0 <= r < h and 0 <= c < w) for r, c in behind):
            continue
        for r, c in moved:
            labels[r, c] = target
        ports.append(moved+behind)
    if rng.integers(2):
        # Transform entire completed partition; fragments later turn separately.
        labels = labels.T.copy()
        ports = [[(c, r) for r, c in p] for p in ports]
    return labels, ports


def _picture(rng, labels, ports, palette, repeated_ports=False):
    h, w = labels.shape
    carrier, accent, *port_colors = palette
    if repeated_ports:
        port_colors = port_colors[:1]
    pic = np.full((h, w), carrier, dtype=np.int8)
    style = int(rng.integers(3))
    if style == 0:
        for r in range(int(rng.integers(3)), h, int(rng.integers(3, 6))):
            pic[r:r+2] = accent
    elif style == 1:
        for _ in range(int(rng.integers(2, 5))):
            r, c = int(rng.integers(h-2)), int(rng.integers(w-2))
            ph, pw = int(rng.integers(2, h-r+1)), int(rng.integers(2, w-c+1))
            pic[r:r+ph, c:c+pw] = accent
    else:
        for r in range(1, h-1, 4):
            for c in range(1, w-1, 4):
                pic[r:r+2, c:c+2] = accent
    parent = list(range(h*w))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def unite(a, b):
        parent[root(a)] = root(b)

    for r in range(h):
        for c in range(w):
            for yy, xx in ((r+1, c), (r, c+1)):
                if yy < h and xx < w and labels[r, c] != labels[yy, xx]:
                    unite(r*w+c, yy*w+xx)
    for p in ports:
        for a in p[1:]:
            unite(p[0][0]*w+p[0][1], a[0]*w+a[1])
    groups = {}
    for i in range(h*w):
        groups.setdefault(root(i), []).append(i)
    flat = pic.ravel()
    for cells in groups.values():
        if len(cells) > 1:
            # Repeated, often quiet runs, rather than unique per-edge codes.
            flat[cells] = int(rng.choice((carrier, accent)))
    rng.shuffle(ports)
    for i, p in enumerate(ports):
        flat[groups[root(p[0][0]*w+p[0][1])]] = port_colors[i % len(port_colors)]
    return pic


def _marker(point, corner):
    r, c = point
    dy, dx = (1 if corner[0] else -1), (1 if corner[1] else -1)
    return [(r+dy, c), (r, c+dx), (r+dy, c+dx)]


def _build(rng, kind, corner, witness=False):
    for attempt in range(90):
        labels, port_groups = _cut_sockets(rng, _partition(rng, kind), kind == "cycle")
        palette = list(map(int, rng.permutation(10)))
        bg, marker = palette[:2]
        pic = _picture(rng, labels, port_groups, palette[2:6], witness)
        h, w = labels.shape
        if h > 28 or w > 28:
            continue
        pieces = []
        for i in range(int(labels.max())+1):
            q = np.argwhere(labels == i)
            r, c = q.min(0)
            y, x = q.max(0)+1
            m = np.where(labels[r:y, c:x] == i, pic[r:y, c:x], -1).astype(np.int8)
            pieces.append((m, (int(r), int(c))))
        if len(port_groups) < 2 or any(len(_components(m >= 0)) != 1 for m, _ in pieces):
            continue
        local = (h-1 if corner[0] else 0, w-1 if corner[1] else 0)
        anchor = int(labels[local])
        shape = (min(30, max(h+4, int(rng.integers(23, 31)))),
                 min(30, max(w+4, int(rng.integers(23, 31)))))
        if max(shape) > 30:
            continue
        origin = (int(rng.integers(1, shape[0]-h)), int(rng.integers(1, shape[1]-w)))
        point = (origin[0]+local[0], origin[1]+local[1])
        marker_cells = _marker(point, corner)
        grid = np.full(shape, bg, dtype=np.int8)
        blocked = np.zeros(shape, dtype=bool)
        for p in marker_cells:
            grid[p] = marker
            r, c = p
            blocked[max(0, r-1):r+2, max(0, c-1):c+2] = True
        rotations = [int(rng.integers(4)) for _ in pieces]
        rotations[anchor] = 0
        other = [i for i in range(len(pieces)) if i != anchor]
        if witness:
            turns = list(map(int, rng.permutation((1, 2, 3))))
            for i, turn in zip(other, turns):
                rotations[i] = turn
        else:
            rotations[other[0]] = int(rng.integers(1, 4))
        order = [anchor]+sorted(other, key=lambda i: -pieces[i][0].size)
        good = True
        for i in order:
            m = np.rot90(pieces[i][0], rotations[i])
            ph, pw = m.shape
            q = np.argwhere(m >= 0)
            if i == anchor:
                r, c = origin[0]+pieces[i][1][0], origin[1]+pieces[i][1][1]
            else:
                tops, lefts = np.indices((shape[0]-ph+1, shape[1]-pw+1))
                rr, cc = tops.ravel()[:, None]+q[:, 0], lefts.ravel()[:, None]+q[:, 1]
                available = np.flatnonzero(~np.any(blocked[rr, cc], axis=1))
                if len(available) == 0:
                    good = False
                    break
                k = int(rng.choice(available))
                r, c = int(tops.ravel()[k]), int(lefts.ravel()[k])
            rr, cc = q[:, 0]+r, q[:, 1]+c
            grid[rr, cc] = m[m >= 0]
            # Bounding-box separation keeps multicolor objects readable.
            blocked[max(0, r-1):r+ph+1, max(0, c-1):c+pw+1] = True
        if not good or np.count_nonzero(grid == bg) <= grid.size//2:
            continue
        try:
            expected = np.full(shape, bg, dtype=np.int8)
            expected[origin[0]:origin[0]+h, origin[1]:origin[1]+w] = pic
            predicted = solve(grid)
            if not np.array_equal(predicted, expected):
                continue
            no_turns, _ = _search(grid, rotate=False, max_solutions=1)
            if no_turns:
                continue
            if witness:
                loose, _ = _search(grid, mode="ports", max_solutions=2)
                if len(loose) < 2:
                    continue
                recovered = _prediction(grid.shape, grid.tobytes())[1]
                if {place[2] for place in recovered.values()} != set(range(4)):
                    continue
            return {"input": grid, "output": expected}
        except (ValueError, SearchLimit):
            continue
    raise RuntimeError(f"could not build an unambiguous {kind} socket jigsaw")


def _counts(num_train, num_test):
    for value, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError("counts must be integers: train 2..10, test 1..4")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    corners = list(map(tuple, ((0, 0), (1, 1), (0, 1), (1, 0))))
    rng.shuffle(corners)
    plan = [("cycle", corners[0], True), ("strips", corners[1], False)]
    modes = ("cycle", "strips", "recursive", "staggered")
    for i in range(2, num_train):
        plan.append((str(rng.choice(modes)), corners[i % 2], False))
    rng.shuffle(plan)
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, specs in (("train", plan), ("test", [(str(rng.choice(("recursive", "staggered"))), corners[2+i % 2], False)
                                                   for i in range(num_test)])):
        for kind, corner, witness in specs:
            for _ in range(8):
                pair = _build(rng, kind, corner, witness)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("could not obtain distinct inputs")
    return puzzle


def _features(pair, ablate=False):
    g, out = _grid(pair["input"]), _grid(pair["output"])
    bg, marker, point, corner, pieces, anchor, marker_cells = _extract(g)
    predicted, places, target, labels, nodes = _prediction(g.shape, g.tobytes())
    assert np.array_equal(out, predicted), "wrong assembly"
    assert out.shape == g.shape and not np.array_equal(g, out)
    assert places[anchor] == (*pieces[anchor][1], 0), "marked fragment moved or turned"
    assert all(out[tuple(p)] == bg for p in marker_cells) and not np.any(out == marker)
    cin, cout = np.bincount(g.ravel(), minlength=10), np.bincount(out.ravel(), minlength=10)
    assert cin[marker] == 3 and cout[bg] == cin[bg]+3
    assert all(cin[c] == cout[c] for c in range(10) if c not in (bg, marker))
    turnset = set()
    masks = []
    tongues = sockets = 0
    tongue_segments, socket_cells = [], {}
    for i, (m, old) in enumerate(pieces):
        r, c, turns = places[i]
        v = np.rot90(m, turns)
        section = out[r:r+v.shape[0], c:c+v.shape[1]]
        assert np.array_equal(section[v >= 0], v[v >= 0]), "piece contents changed"
        turnset.add(turns)
        _, nt, ns, segments = _ports(v, details=True)
        tongues += nt
        sockets += ns
        for kind, coords, direction in segments:
            absolute = coords+np.array((r, c))
            colors = v[coords[:, 0], coords[:, 1]]
            assert colors[0] == colors[1], "a port bar is not monochrome"
            if kind == "tongue":
                tongue_segments.append((absolute, direction, int(colors[0])))
            else:
                for y, x in absolute:
                    socket_cells[(int(y), int(x), direction)] = int(colors[0])
        masks.append(min((np.rot90(m >= 0, t).shape, np.rot90(m >= 0, t).tobytes()) for t in range(4)))
    assert tongues >= 2 and sockets == tongues
    for coords, (dy, dx), color in tongue_segments:
        for y, x in coords:
            assert socket_cells.get((int(y+dy), int(x+dx), (-dy, -dx))) == color, "tongue did not dock to a matching socket"
    # Explicit contacts in the solved mask include notch sidewalls and subedges.
    oy = point[0]-(target[0]-1 if corner[0] else 0)
    ox = point[1]-(target[1]-1 if corner[1] else 0)
    picture = out[oy:oy+target[0], ox:ox+target[1]]
    right = labels[:, :-1] != labels[:, 1:]
    down = labels[:-1, :] != labels[1:, :]
    assert np.all(picture[:, :-1][right] == picture[:, 1:][right])
    assert np.all(picture[:-1, :][down] == picture[1:, :][down])
    relaxed_count = None
    if ablate:
        no_turns, _ = _search(g, rotate=False, max_solutions=1)
        assert not no_turns, "rotation concept is unnecessary"
        relaxed, _ = _search(g, mode="ports", max_solutions=2)
        relaxed_count = len(relaxed)
    return dict(corner=corner, count=len(pieces), target=target, shape=g.shape,
                tongues=tongues, sockets=sockets, turns=tuple(sorted(turnset)),
                masks=tuple(masks), relaxed_count=relaxed_count, nodes=nodes,
                palette=tuple(map(int, np.unique(out))), background=bg)


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g = _grid(pair["input"])
            key = (g.shape, g.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            _features(pair)
    train = [_features(p, ablate=True) for p in puzzle["train"]]
    assert any(f["relaxed_count"] == 2 for f in train), "training lacks a full-pattern witness"
    assert set().union(*(set(f["turns"]) for f in train)) == set(range(4)), "training lacks rotation witnesses"
    assert len({f["corner"] for f in train}) >= 2


def _fingerprint(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for name in ("input", "output"):
                g = np.asarray(pair[name], dtype=np.int8)
                digest.update(bytes(g.shape))
                digest.update(g.tobytes())
    return digest.hexdigest()


def self_test():
    """42 streamed seeds, bounded search, no disk writes or retained grids."""
    fingerprints, shapes, targets, masks, corners, counts = set(), set(), set(), set(), set(), set()
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        fingerprints.add(_fingerprint(p))
        for split in ("train", "test"):
            for pair in p[split]:
                f = _features(pair)
                shapes.add(f["shape"])
                targets.add(f["target"])
                masks.update(f["masks"])
                corners.add(f["corner"])
                counts.add(f["count"])
    assert len(fingerprints) == 42
    assert len(shapes) >= 20 and len(targets) >= 30 and len(masks) >= 60
    assert len(corners) == 4 and len(counts) >= 3
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(generate_puzzle(0))
    for a in range(2, 11):
        for b in range(1, min(4, a-1)+1):
            p = generate_puzzle(1000+a*10+b, a, b)
            assert len(p["train"]) == a and len(p["test"]) == b
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = generate_puzzle(0, 2, 1)
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "canvas_shapes": len(shapes),
            "target_shapes": len(targets), "rotation_normalized_masks": len(masks), "piece_counts": sorted(counts)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
