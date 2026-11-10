"""Assemble a signed rectangular field, then complete its checkpoint routes.

Only Python/NumPy are required. No parent generator, files, training pairs, seed,
or construction metadata are consulted by solve(). Searches fail closed on ties
and on exhausted budgets. No execution traces or generated datasets are saved.
"""
from collections import deque
from functools import lru_cache
from heapq import heappop, heappush
import hashlib
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
CORNERS = ("TL", "BR", "TR", "BL")


class Ambiguous(ValueError):
    """Missing, tied, or incompletely searched interpretation."""


def _grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid")
    return g.astype(np.int8, copy=True)


def _components(mask):
    cells = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while cells:
        start = min(cells)
        cells.remove(start)
        todo = [start]
        for r, c in todo:
            for dr, dc in DIRS:
                p = r + dr, c + dc
                if p in cells:
                    cells.remove(p)
                    todo.append(p)
        result.append(tuple(todo))
    return result


def _ordered(cells):
    cells = set(cells)
    if len(cells) == 1:
        return tuple(cells)
    adjacent = {p: [q for dr, dc in DIRS
                    if (q := (p[0]+dr, p[1]+dc)) in cells] for p in cells}
    ends = sorted(p for p in cells if len(adjacent[p]) == 1)
    if len(ends) != 2 or any(len(a) not in (1, 2) for a in adjacent.values()):
        raise Ambiguous("checkpoint component is not a simple path")
    sequence = [ends[0]]
    previous = None
    while sequence[-1] != ends[1]:
        nxt = [p for p in adjacent[sequence[-1]] if p != previous]
        previous = sequence[-1]
        sequence.append(nxt[0])
    if len(sequence) != len(cells):
        raise Ambiguous("disconnected path")
    return tuple(sequence)


def _bends(path):
    directions = [(b[0]-a[0], b[1]-a[1]) for a, b in zip(path, path[1:])]
    return sum(a != b for a, b in zip(directions, directions[1:]))


def _marker(point, corner):
    r, c = point
    dr = -1 if corner[0] == "T" else 1
    dc = -1 if corner[1] == "L" else 1
    return ((r+dr, c), (r, c+dc), (r+dr, c+dc))


def _extract(g):
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    if np.count_nonzero(counts == counts[bg]) != 1:
        raise Ambiguous("exterior must be the unique modal color")
    markers = []
    for color in np.flatnonzero(counts == 3):
        if color == bg:
            continue
        coords = set(map(tuple, np.argwhere(g == color).tolist()))
        r, c = np.min(list(coords), axis=0)
        square = {(int(r), int(c)), (int(r+1), int(c)),
                  (int(r), int(c+1)), (int(r+1), int(c+1))}
        if len(coords & square) != 3:
            continue
        p = (square - coords).pop()
        if g[p] in (bg, color):
            continue
        corner = ("T" if p[0] == r+1 else "B") + ("L" if p[1] == c+1 else "R")
        markers.append((int(color), p, corner))
    if len(markers) != 1:
        raise Ambiguous("expected one external three-cell L")
    marker, point, corner = markers[0]
    pieces, anchor = [], None
    for cells in _components((g != bg) & (g != marker)):
        r, c = np.min(cells, axis=0)
        y, x = np.max(cells, axis=0)+1
        if len(cells) != (y-r)*(x-c):
            raise Ambiguous("payloads must be filled rectangles")
        if point in cells:
            anchor = len(pieces)
        pieces.append((g[r:y, c:x].copy(), (int(r), int(c))))
    if anchor is None or not 3 <= len(pieces) <= 7:
        raise Ambiguous("invalid payload count or anchor")
    m, (r, c) = pieces[anchor]
    if point != (r+(m.shape[0]-1 if corner[0] == "B" else 0),
                 c+(m.shape[1]-1 if corner[1] == "R" else 0)):
        raise Ambiguous("L must mark the corresponding payload corner")
    wall = int(g[point])
    payload = np.concatenate([m.ravel() for m, _ in pieces])
    pc = np.bincount(payload, minlength=10)
    pc[[bg, marker, wall]] = 0
    free = int(pc.argmax())
    if pc[free] == 0 or np.count_nonzero(pc == pc[free]) != 1:
        raise Ambiguous("free carrier must be modal among non-wall payload colors")
    return bg, marker, wall, free, point, corner, pieces, anchor


def _assembly(g, check_edges=True, stop_after=2):
    """Complete rectangle exact cover, including all feasible area factors.

    Filled rectangles covering the first uncovered cell must start there. Edge
    compatibility is checked against every previously placed neighbor, including
    partial long/short contacts. Source order and scattered distances do not rank
    alternatives. The wall frame is another visible boundary constraint.
    """
    bg, marker, wall, free, point, corner, pieces, anchor = _extract(g)
    area = sum(m.size for m, _ in pieces)
    solutions = []
    nodes = 0
    for h in range(3, g.shape[0]+1):
        if area % h:
            continue
        w = area // h
        oy = point[0] - (h-1 if corner[0] == "B" else 0)
        ox = point[1] - (w-1 if corner[1] == "R" else 0)
        if w < 3 or oy < 0 or ox < 0 or oy+h > g.shape[0] or ox+w > g.shape[1]:
            continue
        board = np.full((h, w), -1, dtype=np.int8)
        labels = np.full((h, w), -1, dtype=np.int8)
        positions = [None] * len(pieces)

        def fits(i, r, c):
            m = pieces[i][0]
            ph, pw = m.shape
            if r < 0 or c < 0 or r+ph > h or c+pw > w:
                return False
            if np.any(labels[r:r+ph, c:c+pw] >= 0):
                return False
            if ((r == 0 and np.any(m[0] != wall))
                    or (c == 0 and np.any(m[:, 0] != wall))
                    or (r+ph == h and np.any(m[-1] != wall))
                    or (c+pw == w and np.any(m[:, -1] != wall))):
                return False
            if check_edges:
                contacts = []
                if r:
                    contacts.append((board[r-1, c:c+pw], m[0]))
                if r+ph < h:
                    contacts.append((board[r+ph, c:c+pw], m[-1]))
                if c:
                    contacts.append((board[r:r+ph, c-1], m[:, 0]))
                if c+pw < w:
                    contacts.append((board[r:r+ph, c+pw], m[:, -1]))
                if any(np.any((a >= 0) & (a != b)) for a, b in contacts):
                    return False
            return True

        def place(i, r, c):
            m = pieces[i][0]
            ph, pw = m.shape
            board[r:r+ph, c:c+pw] = m
            labels[r:r+ph, c:c+pw] = i
            positions[i] = (r, c)

        def visit(remaining):
            nonlocal nodes
            nodes += 1
            if nodes > 40000:
                raise Ambiguous("exact-cover search budget exceeded")
            if not remaining:
                solutions.append((board.copy(), labels.copy(), (oy, ox), tuple(positions)))
                return len(solutions) >= stop_after
            missing = np.argwhere(labels < 0)
            r, c = map(int, missing[0])
            for i in remaining:
                if fits(i, r, c):
                    place(i, r, c)
                    if visit(tuple(j for j in remaining if j != i)):
                        return True
                    ph, pw = pieces[i][0].shape
                    board[r:r+ph, c:c+pw] = -1
                    labels[r:r+ph, c:c+pw] = -1
                    positions[i] = None
            return False

        ar, ac = pieces[anchor][1][0]-oy, pieces[anchor][1][1]-ox
        if fits(anchor, ar, ac):
            place(anchor, ar, ac)
            if visit(tuple(i for i in range(len(pieces)) if i != anchor)):
                break
    return solutions, (bg, marker, wall, free, point, corner, pieces, anchor)


@lru_cache(maxsize=256)
def _route_cached(shape, data, color, free, prefer_bends=True):
    """Exact A* search over induced paths through atomic visible fragments.

    Cost is (new free cells, turns), equivalent to (total cells, turns) for this
    fixed input. A 0/1-distance MST is an admissible lower bound. Hidden turns,
    arbitrary fragment order, and reversal equivalence are all handled. There is
    no checkpoint-count-to-length formula and no generation-provided bound.
    """
    g = np.frombuffer(data, dtype=np.int8).reshape(shape)
    h, w = shape
    values = g.ravel()
    fragments = [tuple(r*w+c for r, c in _ordered(cells))
                 for cells in _components(g == color)]
    if not fragments:
        raise Ambiguous("missing checkpoint color")
    if len(fragments) == 1:
        return tuple(divmod(p, w) for p in fragments[0])
    n = len(fragments)
    neighbors = [tuple(rr*w+cc for dr, dc in DIRS
                       if 0 <= (rr := r+dr) < h and 0 <= (cc := c+dc) < w)
                 for r in range(h) for c in range(w)]
    endpoints = {p: i for i, f in enumerate(fragments) for p in (f[0], f[-1])}
    distances = []
    for fragment in fragments:
        dist = [10000] * (h*w)
        todo = deque(fragment)
        for p in fragment:
            dist[p] = 0
        while todo:
            p = todo.popleft()
            for q in neighbors[p]:
                if values[q] not in (free, color):
                    continue
                cost = int(values[q] == free)
                if dist[q] > dist[p]+cost:
                    dist[q] = dist[p]+cost
                    (todo.append if cost else todo.appendleft)(q)
        distances.append(dist)
    weights = [[min(distances[i][p] for p in fragments[j]) for j in range(n)]
               for i in range(n)]

    @lru_cache(maxsize=None)
    def mst(mask):
        remaining = [i for i in range(n) if mask & (1 << i)]
        if not remaining:
            return 0
        included = [remaining.pop()]
        total = 0
        while remaining:
            cost, j = min((weights[i][j], j) for i in included for j in remaining)
            total += cost
            included.append(j)
            remaining.remove(j)
        return total

    all_mask = (1 << n)-1

    def lower(current, mask):
        remaining = all_mask ^ mask
        if not remaining:
            return 0
        near = min(distances[i][current] for i in range(n) if remaining & (1 << i))
        return max(0, near-int(values[current] == free)) + mst(remaining)

    def append(path, visited, cells, added, bends, direction):
        for q in cells:
            if visited & (1 << q):
                return None
            if path:
                if sum(bool(visited & (1 << p)) for p in neighbors[q]) != 1:
                    return None
                p = path[-1]
                if p not in neighbors[q]:
                    return None
                d = (q//w-p//w, q%w-p%w)
                bends += int(direction is not None and direction != d)
                direction = d
            visited |= 1 << q
            added += int(values[q] == free)
            path += (q,)
        return path, visited, added, bends, direction

    queue = []
    serial = 0
    for i, fragment in enumerate(fragments):
        orientations = (fragment,) if len(fragment) == 1 else (fragment, fragment[::-1])
        for oriented in orientations:
            path, bits, added, bends, d = append((), 0, oriented, 0, 0, None)
            mask = 1 << i
            heappush(queue, (lower(path[-1], mask), bends if prefer_bends else 0, added, serial,
                             bends, mask, path, bits, d))
            serial += 1
    best = None
    winners = set()
    expanded = 0
    while queue:
        estimate, ranked_bends, added, _, bends, mask, path, bits, direction = heappop(queue)
        if best is not None and (estimate, ranked_bends) > best:
            break
        if mask == all_mask:
            key = min(path, path[::-1])
            if best is None:
                best = added, ranked_bends
            winners.add(key)
            if len(winners) > 1:
                raise Ambiguous("tied optimal checkpoint paths")
            continue
        expanded += 1
        if expanded > 40000 or len(queue) > 60000:
            raise Ambiguous("route uniqueness search budget exceeded")
        for q in neighbors[path[-1]]:
            if bits & (1 << q):
                continue
            if values[q] == free:
                cells, new_mask = (q,), mask
            elif values[q] == color and q in endpoints:
                owner = endpoints[q]
                if mask & (1 << owner):
                    continue
                f = fragments[owner]
                cells = f if q == f[0] else f[::-1]
                new_mask = mask | (1 << owner)
            else:
                continue
            result = append(path, bits, cells, added, bends, direction)
            if result is None:
                continue
            pp, bb, aa, tt, dd = result
            bound = aa + lower(pp[-1], new_mask)
            rank = tt if prefer_bends else 0
            if best is None or (bound, rank) <= best:
                heappush(queue, (bound, rank, aa, serial, tt, new_mask, pp, bb, dd))
                serial += 1
    if len(winners) != 1:
        raise Ambiguous("no checkpoint path")
    return tuple(divmod(p, w) for p in winners.pop())


def _route(g, color, free, prefer_bends=True):
    return _route_cached(g.shape, g.astype(np.int8).tobytes(), int(color), int(free), prefer_bends)


def _length_tie(g, color, free):
    try:
        _route(g, color, free, prefer_bends=False)
    except Ambiguous as error:
        return str(error) == "tied optimal checkpoint paths"
    return False


def _complete(board, wall, free):
    out = board.copy()
    occupied = set()
    for color in sorted(set(map(int, np.unique(board))) - {wall, free}):
        path = _route(board, color, free)
        if occupied & set(path):
            raise Ambiguous("individually optimal routes conflict")
        occupied.update(path)
        for p in path:
            if board[p] not in (free, color):
                raise Ambiguous("route overwrites a foreign cell")
            out[p] = color
    return out


@lru_cache(maxsize=128)
def _infer(shape, data):
    g = np.frombuffer(data, dtype=np.int8).reshape(shape)
    solutions, roles = _assembly(g)
    if len(solutions) != 1:
        raise Ambiguous("assembly is missing or nonunique")
    board, labels, (r, c), positions = solutions[0]
    bg, marker, wall, free, point, corner, pieces, anchor = roles
    completed = _complete(board, wall, free)
    result = np.full(g.shape, bg, dtype=np.int8)
    result[r:r+board.shape[0], c:c+board.shape[1]] = completed
    return result, solutions[0], roles


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Solve from this input alone; cached values are pure input functions."""
    g = _grid(input_grid)
    return _infer(g.shape, g.tobytes())[0].copy()


def _partition(rng, h, w, count, mode):
    if mode == "strips":
        widths = [3] * count
        for _ in range(w-3*count):
            widths[int(rng.integers(count))] += 1
        rects, c = [], 0
        for pw in widths:
            rects.append((0, c, h, pw))
            c += pw
    elif mode == "bands":
        cut = int(rng.integers(3, w-2))
        counts = [count//2, count-count//2]
        if max(counts)*3 > h:
            return None
        rects = []
        for (c, pw), n in zip(((0, cut), (cut, w-cut)), counts):
            heights = [3]*n
            for _ in range(h-3*n):
                heights[int(rng.integers(n))] += 1
            r = 0
            for ph in heights:
                rects.append((r, c, ph, pw))
                r += ph
    else:
        rects = [(0, 0, h, w)]
        for _ in range(count-1):
            options = [(i, axis) for i, (_, _, ph, pw) in enumerate(rects)
                       for axis, length in enumerate((ph, pw)) if length >= 6]
            if not options:
                return None
            i, axis = options[int(rng.integers(len(options)))]
            r, c, ph, pw = rects.pop(i)
            cut = int(rng.integers(3, (ph if axis == 0 else pw)-2))
            if axis == 0:
                rects.extend(((r, c, cut, pw), (r+cut, c, ph-cut, pw)))
            else:
                rects.extend(((r, c, ph, cut), (r, c+cut, ph, pw-cut)))
    labels = np.full((h, w), -1, dtype=np.int8)
    for i, (r, c, ph, pw) in enumerate(rects):
        labels[r:r+ph, c:c+pw] = i
    return labels


def _field(rng, labels, free, wall):
    h, w = labels.shape
    parents = list(range(h*w))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seam = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            for rr, cc in ((r+1, c), (r, c+1)):
                if rr < h and cc < w and labels[r, c] != labels[rr, cc]:
                    parents[root(rr*w+cc)] = root(r*w+c)
                    seam[r, c] = seam[rr, cc] = True
    groups = {}
    for i in range(h*w):
        groups.setdefault(root(i), []).append(divmod(i, w))
    board = np.full((h, w), free, dtype=np.int8)
    for cells in groups.values():
        boundary = any(r in (0, h-1) or c in (0, w-1) for r, c in cells)
        density = .31 if any(seam[p] for p in cells) else float(rng.uniform(.04, .14))
        if boundary or rng.random() < density:
            for p in cells:
                board[p] = wall
    return board, seam


def _terminal_path(board, start, end, free):
    """Small ordinary shortest/turn search to propose, never certify, a route."""
    h, w = board.shape
    queue = [(0, 0, start, -1, (start,))]
    costs = {(start, -1): (0, 0)}
    while queue:
        steps, turns, p, direction, path = heappop(queue)
        if costs.get((p, direction)) != (steps, turns):
            continue
        if p == end:
            return path
        for d, (dr, dc) in enumerate(DIRS):
            q = p[0]+dr, p[1]+dc
            if not (0 <= q[0] < h and 0 <= q[1] < w) or board[q] != free:
                continue
            cost = steps+1, turns+int(direction != -1 and d != direction)
            if cost < costs.get((q, d), (10000, 10000)):
                costs[q, d] = cost
                heappush(queue, (*cost, q, d, path+(q,)))
    return None


def _shortest_multiplicity(board, start, end, free, color):
    """Count up to two shortest walks, without the bend preference."""
    dist, counts = {start: 0}, {start: 1}
    todo = deque([start])
    while todo:
        p = todo.popleft()
        for dr, dc in DIRS:
            q = p[0]+dr, p[1]+dc
            if not (0 <= q[0] < board.shape[0] and 0 <= q[1] < board.shape[1]):
                continue
            if board[q] not in (free, color):
                continue
            if q not in dist:
                dist[q], counts[q] = dist[p]+1, counts[p]
                todo.append(q)
            elif dist[q] == dist[p]+1:
                counts[q] = min(2, counts[q]+counts[p])
    return counts.get(end, 0)


def _add_routes(rng, board, labels, seam, free, colors, detour, bend_evidence):
    clues = board.copy()
    completed = board.copy()
    paths = []
    for index, color in enumerate(colors):
        eligible = [tuple(map(int, p)) for p in np.argwhere((completed == free) & ~seam)]
        if len(eligible) < 2:
            return None
        accepted = False
        for _ in range(90):
            a, b = [eligible[int(k)] for k in rng.choice(len(eligible), 2, replace=False)]
            if labels[a] == labels[b] or abs(a[0]-b[0])+abs(a[1]-b[1]) < 4:
                continue
            path = _terminal_path(completed, a, b, free)
            if path is None or len(path) > 27:
                continue
            excess = len(path)-1-abs(a[0]-b[0])-abs(a[1]-b[1])
            if index == 0 and detour and excess < 2:
                continue
            if index == 0 and bend_evidence and _shortest_multiplicity(completed, a, b, free, color) < 2:
                continue
            # All clues stay off cut contacts: the signatures are immutable,
            # while completed routes may cross their free windows.
            mark = {a, b}
            if not bend_evidence:
                for oriented in (path, path[::-1]):
                    for p in oriented[1:int(rng.integers(1, 4))]:
                        if seam[p]:
                            break
                        mark.add(p)
                choices = [p for p in path[2:-2] if not seam[p] and p not in mark]
                rng.shuffle(choices)
                for p in choices[:int(rng.integers(0, 3))]:
                    mark.add(p)
            candidate = clues.copy()
            for p in mark:
                candidate[p] = color
            if len(_components(candidate == color)) < 2:
                continue
            try:
                inferred = _route(candidate, color, free)
            except Ambiguous:
                continue
            if set(inferred) != set(path):
                continue
            if index == 0 and detour:
                turns = [path[k] for k in range(1, len(path)-1)
                         if (path[k][0]-path[k-1][0], path[k][1]-path[k-1][1])
                         != (path[k+1][0]-path[k][0], path[k+1][1]-path[k][1])]
                if not any(p not in mark for p in turns):
                    continue
            clues = candidate
            for p in path:
                completed[p] = color
            paths.append(path)
            accepted = True
            break
        if not accepted:
            return None
    # Later checkpoints must not invalidate earlier optima. Optimize all colors
    # against exactly the same assembled clue field and check disjointness.
    try:
        prediction = _complete(clues, int(board[0, 0]), free)
    except Ambiguous:
        return None
    if not np.array_equal(prediction, completed):
        return None
    if bend_evidence and (len({len(path) for path in paths}) < 2
                          or not any(_length_tie(clues, color, free) for color in colors)):
        return None
    return clues, completed


def _scatter(rng, board, labels, corner, roles):
    bg, free, wall, marker = roles
    h, w = board.shape
    pieces = []
    for i in range(int(labels.max())+1):
        cells = np.argwhere(labels == i)
        r, c = cells.min(0)
        y, x = cells.max(0)+1
        pieces.append((board[r:y, c:x].copy(), (int(r), int(c))))
    # Canvas area is >2x payload, guaranteeing an obvious exterior.
    ch = min(30, max(h+5, int(np.ceil(np.sqrt(h*w*2.8)))+int(rng.integers(1, 5))))
    cw = min(30, max(w+5, int(np.ceil(h*w*2.8/ch))+int(rng.integers(2, 7))))
    oy = int(rng.integers(1, ch-h))
    ox = int(rng.integers(1, cw-w))
    point = oy+(h-1 if corner[0] == "B" else 0), ox+(w-1 if corner[1] == "R" else 0)
    anchor = int(labels[h-1 if corner[0] == "B" else 0, w-1 if corner[1] == "R" else 0])
    marker_cells = _marker(point, corner)
    positions = {anchor: (oy+pieces[anchor][1][0], ox+pieces[anchor][1][1])}

    def separated(first, second):
        r, c, ph, pw = first
        y, x, qh, qw = second
        return r+ph < y or y+qh < r or c+pw < x or x+qw < c

    remaining = sorted((i for i in range(len(pieces)) if i != anchor), key=lambda i: -pieces[i][0].size)
    for _ in range(16):
        rectangles = [(*positions[anchor], *pieces[anchor][0].shape)]
        current = dict(positions)
        for i in remaining:
            ph, pw = pieces[i][0].shape
            options = [(r, c) for r in range(ch-ph+1) for c in range(cw-pw+1)]
            found = False
            for k in rng.permutation(len(options)):
                r, c = options[int(k)]
                if any(r-1 <= y <= r+ph and c-1 <= x <= c+pw for y, x in marker_cells):
                    continue
                if all(separated((r, c, ph, pw), rect) for rect in rectangles):
                    current[i] = (r, c)
                    rectangles.append((r, c, ph, pw))
                    found = True
                    break
            if not found:
                break
        else:
            g = np.full((ch, cw), bg, dtype=np.int8)
            for i, (m, _) in enumerate(pieces):
                r, c = current[i]
                g[r:r+m.shape[0], c:c+m.shape[1]] = m
            for p in marker_cells:
                g[p] = marker
            return g, (oy, ox)
    return None


def _pair(rng, corner, roles, colors, kind, test=False):
    bg, free, wall, marker = roles
    for attempt in range(140):
        if kind == "detour":
            h, w, count, mode = int(rng.integers(8, 12)), int(rng.integers(12, 17)), 4, "strips"
        else:
            h = int(rng.integers(9, 14 if test else 13))
            w = int(rng.integers(11, 16 if test else 15))
            count = int(rng.integers(5, 7)) if test else int(rng.integers(3, 6))
            mode = "bands" if kind == "bends" else str(rng.choice(("bands", "recursive")))
        labels = _partition(rng, h, w, count, mode)
        if labels is None:
            continue
        board, seam = _field(rng, labels, free, wall)
        routes = _add_routes(rng, board, labels, seam, free, colors,
                             kind == "detour", kind == "bends")
        if routes is None:
            continue
        clues, completed = routes
        # Optional inert color: one supplied component, copied without extension.
        if len(colors) < 3 and (kind == "bends" or rng.random() < .6):
            inert = int(rng.choice([c for c in range(10) if c not in roles and c not in colors]))
            options = np.argwhere((completed == free) & ~seam)
            if kind == "bends" and not len(options):
                continue
            if len(options):
                p = tuple(map(int, options[int(rng.integers(len(options)))]))
                clues[p] = completed[p] = inert
        if kind == "bends" and not any(_length_tie(clues, color, free) for color in colors):
            continue
        if rng.integers(2):
            clues, completed, labels = clues.T.copy(), completed.T.copy(), labels.T.copy()
        # Reflect the constructed field before cutting; pieces themselves are
        # never rotated or reflected by the assembly operation.
        if rng.integers(2):
            clues, completed, labels = clues[::-1].copy(), completed[::-1].copy(), labels[::-1].copy()
        if rng.integers(2):
            clues, completed, labels = clues[:, ::-1].copy(), completed[:, ::-1].copy(), labels[:, ::-1].copy()
        scattered = _scatter(rng, clues, labels, corner, roles)
        if scattered is None:
            continue
        g, (r, c) = scattered
        expected = np.full_like(g, bg)
        expected[r:r+clues.shape[0], c:c+clues.shape[1]] = completed
        try:
            predicted, assembly, _ = _infer(g.shape, g.tobytes())
            if not np.array_equal(predicted, expected):
                continue
            if kind == "detour" and len(_assembly(g, check_edges=False)[0]) < 2:
                continue
        except Ambiguous:
            continue
        return {"input": g, "output": expected}
    raise RuntimeError("bounded construction could not find a unique composed example")


def _integer(value, name, low, high):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or not low <= int(value) <= high):
        raise ValueError(f"{name} must be an integer in {low}..{high}")
    return int(value)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    num_train = _integer(num_train, "num_train", 2, 10)
    num_test = _integer(num_test, "num_test", 1, 4)
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    palette = [int(c) for c in rng.permutation(10)]
    roles, ordinary = tuple(palette[:4]), palette[4:]
    first = int(rng.integers(2))*2
    train_corners = CORNERS[first:first+2]
    test_corners = CORNERS[2-first:4-first]
    puzzle = {"train": [], "test": []}
    seen = set()
    for split, count in (("train", num_train), ("test", num_test)):
        for i in range(count):
            kind = ("detour" if i == 0 else "bends" if i == 1 else "mixed") if split == "train" else "mixed"
            n = 1 if kind == "detour" else 2 if kind == "bends" else int(rng.integers(1, 4))
            active = [int(c) for c in rng.choice(ordinary, n, replace=False)]
            corners = train_corners if split == "train" else test_corners
            for _ in range(8):
                pair = _pair(rng, corners[i % 2], roles, active, kind, split == "test")
                key = pair["input"].shape, pair["input"].tobytes()
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("failed to construct distinct inputs")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    g, out = _grid(pair["input"]), _grid(pair["output"])
    predicted, assembled, roles = _infer(g.shape, g.tobytes())
    assert out.shape == g.shape and np.array_equal(predicted, out)
    board, labels, (r, c), positions = assembled
    bg, marker, wall, free, point, corner, pieces, anchor = roles
    finished = out[r:r+board.shape[0], c:c+board.shape[1]]
    # Assembly preserves every payload pixel, before route completion changes
    # only free carrier. This also checks coverage independently of the drawing.
    assert sum(m.size for m, _ in pieces) == board.size
    assert set(map(int, np.unique(labels))) == set(range(len(pieces)))
    for i, ((m, _), (y, x)) in enumerate(zip(pieces, positions)):
        ph, pw = m.shape
        assert np.all(labels[y:y+ph, x:x+pw] == i)
        assert np.array_equal(board[y:y+ph, x:x+pw], m)
    assert np.array_equal(np.bincount(board.ravel(), minlength=10),
                          np.bincount(np.concatenate([m.ravel() for m, _ in pieces]), minlength=10))
    assert positions[anchor] == (pieces[anchor][1][0]-r, pieces[anchor][1][1]-c)
    assert not np.any(out == marker)
    assert all(out[p] == bg for p in _marker(point, corner))
    immutable = board != free
    assert np.array_equal(board[immutable], finished[immutable])
    assert np.all((board == finished) | (board == free))
    assert np.all(finished[[0, -1], :] == wall) and np.all(finished[:, [0, -1]] == wall)
    assert not np.array_equal(board, finished), "omitting B must change every answer"
    assert any(positions[i] != (old[0]-r, old[1]-c)
               for i, (_, old) in enumerate(pieces)), "omitting A must change the field"
    routes, inert = [], []
    for color in sorted(set(map(int, np.unique(board))) - {wall, free}):
        parts = _components(board == color)
        if len(parts) == 1:
            assert np.array_equal(board == color, finished == color)
            inert.append(color)
            continue
        path = _ordered(map(tuple, np.argwhere(finished == color).tolist()))
        assert set(map(tuple, np.argwhere(board == color).tolist())) <= set(path)
        assert len({int(labels[p]) for p in path}) >= 2
        assert len({int(labels[p]) for part in parts for p in part}) >= 2
        hidden = 0
        for k in range(1, len(path)-1):
            a, p, b = path[k-1:k+2]
            if (p[0]-a[0], p[1]-a[1]) != (b[0]-p[0], b[1]-p[1]):
                hidden += int(board[p] == free)
        routes.append({"color": color, "length": len(path), "bends": _bends(path),
                       "fragments": len(parts), "checkpoints": int(np.count_nonzero(board == color)),
                       "hidden": hidden,
                       "length_tie": _length_tie(board, color, free),
                       "excess": len(path)-1-abs(path[0][0]-path[-1][0])-abs(path[0][1]-path[-1][1])})
    assert routes
    return {"roles": (bg, free, wall, marker), "corner": corner,
            "shape": board.shape, "canvas": g.shape, "pieces": len(pieces),
            "routes": routes, "inert": inert}


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in ("train", "test"))
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g = _grid(pair["input"])
            key = g.shape, g.tobytes()
            assert key not in seen
            seen.add(key)
            infos[split].append(_analyze_pair(pair))
    all_infos = infos["train"]+infos["test"]
    assert len({x["roles"] for x in all_infos}) == 1
    assert len({x["corner"] for x in infos["train"]}) == 2
    assert not ({x["corner"] for x in infos["train"]} & {x["corner"] for x in infos["test"]})
    assert any(p["excess"] > 0 and p["hidden"] for x in infos["train"] for p in x["routes"])
    assert any(len(x["routes"]) >= 2 for x in infos["train"])
    # At least one ambiguous geometric packing is resolved by edge signatures.
    assert len(_assembly(_grid(puzzle["train"][0]["input"]), check_edges=False)[0]) >= 2
    # Bend preference is evidenced, not just named in the rule. Two equally
    # sparse colors with different path lengths rule out counting checkpoints.
    assert any(route["length_tie"] for x in infos["train"] for route in x["routes"])
    assert any(a["checkpoints"] == b["checkpoints"] and a["length"] != b["length"]
               for x in infos["train"] for a in x["routes"] for b in x["routes"])
    assert any(x["inert"] for x in infos["train"])
    assert all(x["pieces"] >= 5 for x in infos["test"])
    return None


def self_test():
    """42 streaming seeds; bounded summaries only, no disk output."""
    fingerprints, shapes, canvases, palettes, counts, route_counts = set(), set(), set(), set(), set(), set()
    bends, lengths, fragments, checkpoints = set(), set(), set(), set()
    hidden = detours = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ("train", "test"):
            for pair in p[split]:
                info = _analyze_pair(pair)
                shapes.add(info["shape"])
                canvases.add(info["canvas"])
                palettes.add(info["roles"])
                counts.add(info["pieces"])
                route_counts.add(len(info["routes"]))
                for route in info["routes"]:
                    bends.add(route["bends"])
                    lengths.add(route["length"])
                    fragments.add(route["fragments"])
                    checkpoints.add(route["checkpoints"])
                    hidden += route["hidden"]
                    detours += route["excess"] > 0
                before = pair["input"].copy()
                assert np.array_equal(solve(before), pair["output"])
                assert np.array_equal(before, pair["input"])
                digest.update(str(before.shape).encode())
                digest.update(before.tobytes())
        fingerprints.add(digest.digest())
    assert len(fingerprints) == 42 and len(palettes) >= 30
    assert len(shapes) >= 25 and len(canvases) >= 25
    assert counts >= {3, 4, 5, 6} and route_counts == {1, 2, 3}
    assert len(bends) >= 4 and len(lengths) >= 10 and len(checkpoints) >= 4
    assert fragments >= {2, 3} and hidden >= 42 and detours >= 42
    a, b = generate_puzzle(123, 2, 1), generate_puzzle(123, 2, 1)
    assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    # Clear all performance caches before a prediction, then exercise palette
    # renaming and all global orientations. No generation metadata is required.
    _infer.cache_clear()
    _route_cached.cache_clear()
    renaming = np.array([7, 4, 9, 1, 5, 8, 0, 3, 6, 2], dtype=np.int8)
    pair = a["test"][0]
    for transpose in (False, True):
        for flip_r, flip_c in ((False, False), (True, False), (False, True), (True, True)):
            before, expected = renaming[pair["input"]], renaming[pair["output"]]
            if transpose:
                before, expected = before.T, expected.T
            before = before[::-1 if flip_r else 1, ::-1 if flip_c else 1]
            expected = expected[::-1 if flip_r else 1, ::-1 if flip_c else 1]
            assert np.array_equal(solve(before), expected)
    for nt, ne in ((1, 1), (11, 1), (4, 0), (5, 5), (2, 2), (3, 4), (True, 1), (4, 1.2), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in a[s]] for s in a}
    damaged["test"][0]["output"][:] = damaged["test"][0]["input"]
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("assembly-only/identity corruption accepted")
    # Explicit tied route: two equally short one-bend Ls must not be chosen by
    # traversal order. An unmarked corner must be supported by asymmetry.
    g = np.ones((5, 5), dtype=np.int8)
    g[[0, -1], :] = 8
    g[:, [0, -1]] = 8
    g[1, 1] = g[3, 3] = 2
    try:
        _route(g, 2, 1)
    except Ambiguous:
        pass
    else:
        raise AssertionError("tied route accepted")
    # Length has priority over turns: two corridors offer a 13-cell, six-bend
    # connection and a 17-cell, two-bend connection. The shorter one must win.
    g = np.full((10, 11), 8, dtype=np.int8)
    keys = ((4, 1), (3, 1), (3, 4), (2, 4), (2, 6), (3, 6), (3, 9), (4, 9))
    short = [keys[0]]
    for end in keys[1:]:
        while short[-1] != end:
            r, c = short[-1]
            short.append((r+int(np.sign(end[0]-r)), c+int(np.sign(end[1]-c))))
    for p in short:
        g[p] = 1
    g[4:9, 1] = g[4:9, 9] = g[8, 1:10] = 1
    g[4, 1] = g[4, 9] = 2
    path = _route(g, 2, 1)
    assert set(path) == set(short) and len(path) == 13 and _bends(path) == 6
    # Equal-length choices need the bend preference. The blocked upper-right
    # corner removes one L; the lower L is the unique one-bend optimum.
    g = np.full((5, 7), 8, dtype=np.int8)
    g[1:4, 1:6] = 1
    g[1, 5] = 8
    g[1, 1] = g[3, 5] = 2
    assert _length_tie(g, 2, 1)
    path = _route(g, 2, 1)
    assert len(path) == 7 and _bends(path) == 1 and (3, 1) in path
    g[2, 3] = 3
    assert _route(g, 3, 1) == ((2, 3),), "single components never extend"
    # Individually unique routes that cross are invalid, irrespective of color
    # traversal order; they are never silently routed sequentially.
    g = np.full((5, 7), 8, dtype=np.int8)
    g[1:4, 1:6] = 1
    g[2, 1] = g[2, 5] = 2
    g[1, 3] = g[3, 3] = 3
    try:
        _complete(g, 8, 1)
    except Ambiguous:
        pass
    else:
        raise AssertionError("conflicting optimal routes accepted")
    # Identical weak signatures leave multiple exact covers. Neither source
    # order nor anchor proximity may select one arbitrarily.
    g = np.zeros((17, 17), dtype=np.int8)
    for r, c in ((2, 2), (2, 7), (7, 2), (7, 7)):
        g[r:r+3, c:c+3] = 8
        g[r+1, c+1] = 1
    for p in _marker((2, 2), "TL"):
        g[p] = 9
    assert len(_assembly(g)[0]) == 2
    try:
        solve(g)
    except Ambiguous:
        pass
    else:
        raise AssertionError("ambiguous assembly accepted")
    return {"seeds": 42, "puzzles": len(fingerprints), "assembly_shapes": len(shapes),
            "canvas_shapes": len(canvases), "palettes": len(palettes),
            "piece_counts": sorted(counts), "active_counts": sorted(route_counts),
            "bends": sorted(bends), "path_lengths": sorted(lengths),
            "checkpoint_counts": sorted(checkpoints), "fragment_counts": sorted(fragments),
            "hidden_turns": hidden, "detour_routes": detours}
