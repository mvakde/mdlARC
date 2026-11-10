"""Stitch scattered conduit modules virtually; collect on their source modules.

Only standard-library code and NumPy are used. No parent modules or saved
examples are imported. The unique packing is found before supply is placed.
"""
from collections import deque
import json

import numpy as np


STEPS = ((-1, 0), (1, 0), (0, -1), (0, 1))
CORNERS = ("TL", "TR", "BL", "BR")


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    if not (_integer(num_train) and _integer(num_test)
            and 2 <= num_train <= 10 and 1 <= num_test <= 4
            and num_train > num_test):
        raise ValueError("Require integer counts 2..10 train, 1..4 test, train > test")
    return int(num_train), int(num_test)


def _grid(value):
    g = np.asarray(value)
    if not (g.ndim == 2 and all(1 <= x <= 30 for x in g.shape)
            and np.issubdtype(g.dtype, np.integer)
            and np.all((0 <= g) & (g <= 9))):
        raise ValueError("Grid must be an integer 2D array, sides 1..30, colors 0..9")
    return g.astype(np.int16, copy=True)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result = []
    for y, x in np.argwhere(mask):
        y, x = int(y), int(x)
        if seen[y, x]:
            continue
        seen[y, x] = True
        cells, queue = [], [(y, x)]
        for r, c in queue:
            cells.append((r, c))
            for dr, dc in STEPS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(cells)
    return result


def _marker_cells(point, corner):
    r, c = point
    dr = -1 if corner[0] == "T" else 1
    dc = -1 if corner[1] == "L" else 1
    return ((r + dr, c), (r, c + dc), (r + dr, c + dc))


def _counter_mask(shape, pos):
    mask = np.zeros(shape, bool)
    r, c = pos
    mask[r:r + 3, c:c + 3] = True
    return mask


def _decode(g):
    """Read every role and source module from this input alone."""
    bg = int(g[0, 0])
    if not all(np.all(edge == bg) for edge in (g[0], g[-1], g[:, 0], g[:, -1])):
        raise ValueError("Need a uniform exterior background frame")
    marker_candidates = []
    for color in np.unique(g):
        color = int(color)
        q = np.argwhere(g == color)
        if color == bg or len(q) != 3:
            continue
        lo, hi = q.min(0), q.max(0)
        if tuple(hi - lo) != (1, 1):
            continue
        square = {(int(r), int(c)) for r in range(lo[0], hi[0] + 1)
                  for c in range(lo[1], hi[1] + 1)}
        missing = square - set(map(tuple, q.tolist()))
        point = missing.pop()
        if g[point] in (bg, color):
            continue
        corner = {(1, 1): "TL", (1, 0): "TR", (0, 1): "BL", (0, 0): "BR"}[
            (point[0] - int(lo[0]), point[1] - int(lo[1]))]
        marker_candidates.append((color, point, corner))
    if len(marker_candidates) != 1:
        raise ValueError("Need one external three-cell L")
    marker, point, corner = marker_candidates[0]
    wall = int(g[point])
    pieces, starts, anchor = [], [], None
    for cells in _components((g != bg) & (g != marker)):
        q = np.array(cells)
        r, c = map(int, q.min(0))
        last_r, last_c = map(int, q.max(0))
        ph, pw = last_r - r + 1, last_c - c + 1
        if len(cells) != ph * pw or min(ph, pw) < 3:
            raise ValueError("Modules must be filled rectangles with sides at least three")
        if point in cells:
            anchor = len(pieces)
            expected = (r + (ph - 1 if corner[0] == "B" else 0),
                        c + (pw - 1 if corner[1] == "R" else 0))
            if point != expected:
                raise ValueError("L must mark the corresponding module corner")
        pieces.append(g[r:last_r + 1, c:last_c + 1].copy())
        starts.append((r, c))
    if anchor is None or not 3 <= len(pieces) <= 7:
        raise ValueError("Need three through seven modules and an anchor")
    for i, p in enumerate(pieces):
        for j in range(i):
            if not _apart((*starts[i], *p.shape), (*starts[j], *pieces[j].shape)):
                raise ValueError("Modules must have a background gap")
    counters = []
    for color in np.unique(g):
        color = int(color)
        if color in (bg, marker, wall):
            continue
        q = np.argwhere(g == color)
        if len(q) != 8 or tuple(q.max(0) - q.min(0)) != (2, 2):
            continue
        r, c = map(int, q.min(0))
        patch = g[r:r + 3, c:c + 3]
        mask = np.ones((3, 3), bool)
        mask[1, 1] = False
        key = int(patch[1, 1])
        if np.all(patch[mask] == color) and key not in (bg, marker, wall, color):
            counters.append(((r, c), color, key))
    if len(counters) != 1:
        raise ValueError("Need one marked 3x3 counter")
    counter, slot, material = counters[0]
    other = set(map(int, np.unique(g))) - {bg, marker, wall, slot, material}
    if len(other) != 1:
        raise ValueError("Need six distinct locally identifiable role colors")
    floor = other.pop()
    counter_piece = None
    for i, (p, (r, c)) in enumerate(zip(pieces, starts)):
        edges = np.concatenate((p[0], p[-1], p[:, 0], p[:, -1]))
        if not np.all((edges == floor) | (edges == wall)):
            raise ValueError("Only wall and floor may occur on module boundaries")
        cr, cc = counter
        if r < cr and c < cc and cr + 3 < r + p.shape[0] and cc + 3 < c + p.shape[1]:
            counter_piece = i
    if counter_piece is None:
        raise ValueError("Counter must lie strictly inside one module")
    return dict(bg=bg, marker=marker, point=point, corner=corner, wall=wall,
                floor=floor, slot=slot, material=material, pieces=pieces, starts=starts,
                anchor=anchor, counter=counter, counter_piece=counter_piece)


def _stitch(pieces, anchor, point, corner, canvas, wall):
    """Complete rectangle search. Contact alternatives are never all imposed.

    Rectangular modules covering the first unoccupied row-major cell must start
    at that cell. Each placement therefore branches only on the unused module.
    Stop early only upon proving ambiguity; a successful search is complete.
    """
    area = sum(p.size for p in pieces)
    answers = []
    for h in range(max(p.shape[0] for p in pieces), canvas[0] + 1):
        if area % h:
            continue
        w = area // h
        if w > canvas[1] or w < max(p.shape[1] for p in pieces):
            continue
        oy = point[0] - (h - 1 if corner[0] == "B" else 0)
        ox = point[1] - (w - 1 if corner[1] == "R" else 0)
        if min(oy, ox) < 0 or oy + h > canvas[0] or ox + w > canvas[1]:
            continue
        board = np.full((h, w), -1, np.int16)
        positions = [None] * len(pieces)

        def fits(i, r, c):
            p = pieces[i]
            ph, pw = p.shape
            if r < 0 or c < 0 or r + ph > h or c + pw > w:
                return False
            if np.any(board[r:r + ph, c:c + pw] >= 0):
                return False
            if ((r == 0 and np.any(p[0] != wall))
                    or (r + ph == h and np.any(p[-1] != wall))
                    or (c == 0 and np.any(p[:, 0] != wall))
                    or (c + pw == w and np.any(p[:, -1] != wall))):
                return False
            comparisons = []
            if r > 0:
                comparisons.append((board[r - 1, c:c + pw], p[0]))
            if r + ph < h:
                comparisons.append((board[r + ph, c:c + pw], p[-1]))
            if c > 0:
                comparisons.append((board[r:r + ph, c - 1], p[:, 0]))
            if c + pw < w:
                comparisons.append((board[r:r + ph, c + pw], p[:, -1]))
            return all(not np.any((a >= 0) & (a != b)) for a, b in comparisons)

        def place(i, r, c):
            ph, pw = pieces[i].shape
            board[r:r + ph, c:c + pw] = pieces[i]
            positions[i] = (r, c)

        def visit(remaining):
            if not remaining:
                answers.append((board.copy(), tuple(positions), (oy, ox)))
                if len(answers) > 1:
                    raise ValueError("Ambiguous anchored conduit reconstruction")
                return
            first = int(np.flatnonzero(board < 0)[0])
            r, c = divmod(first, w)
            for i in remaining:
                if fits(i, r, c):
                    place(i, r, c)
                    visit(tuple(j for j in remaining if j != i))
                    ph, pw = pieces[i].shape
                    board[r:r + ph, c:c + pw] = -1
                    positions[i] = None

        ah, aw = pieces[anchor].shape
        ar = h - ah if corner[0] == "B" else 0
        ac = w - aw if corner[1] == "R" else 0
        if fits(anchor, ar, ac):
            place(anchor, ar, ac)
            visit(tuple(i for i in range(len(pieces)) if i != anchor))
    if len(answers) != 1:
        raise ValueError("No anchored conduit reconstruction")
    return answers[0]


def _distances(board, counter, wall):
    distance = np.full(board.shape, -1, np.int16)
    cr, cc = counter
    queue = deque((cr + r, cc + c) for r in range(3) for c in range(3))
    distance[cr:cr + 3, cc:cc + 3] = 0
    while queue:
        r, c = queue.popleft()
        for dr, dc in STEPS:
            nr, nc = r + dr, c + dc
            if (0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]
                    and board[nr, nc] != wall and distance[nr, nc] < 0):
                distance[nr, nc] = distance[r, c] + 1
                queue.append((nr, nc))
    return distance


def _collect(board, counter, wall, material):
    distance = _distances(board, counter, wall)
    supply = (board == material) & ~_counter_mask(board.shape, counter)
    reachable = supply & (distance >= 0)
    ordered = np.sort(distance[reachable])
    if len(ordered) > 9 and ordered[8] == ordered[9]:
        raise ValueError("Distance tie crosses capacity")
    threshold = int(ordered[min(8, len(ordered) - 1)]) if len(ordered) else -1
    return reachable & (distance <= threshold), distance, supply


def _analyze(input_grid):
    g = _grid(input_grid)
    info = _decode(g)
    board, positions, origin = _stitch(info["pieces"], info["anchor"], info["point"],
                                        info["corner"], g.shape, info["wall"])
    cp = info["counter_piece"]
    local = (info["counter"][0] - info["starts"][cp][0],
             info["counter"][1] - info["starts"][cp][1])
    counter = (positions[cp][0] + local[0], positions[cp][1] + local[1])
    labels = np.full(board.shape, -1, np.int16)
    for i, (p, (r, c)) in enumerate(zip(info["pieces"], positions)):
        labels[r:r + p.shape[0], c:c + p.shape[1]] = i
    selected, distance, supply = _collect(board, counter, info["wall"], info["material"])
    info.update(grid=g, board=board, positions=positions, origin=origin,
                board_counter=counter, labels=labels, selected=selected,
                distance=distance, supply=supply)
    return info


def _render_collection(info):
    result = info["grid"].copy()
    for r, c in np.argwhere(info["selected"]):
        r, c = int(r), int(c)
        i = int(info["labels"][r, c])
        pr, pc = info["positions"][i]
        sr, sc = info["starts"][i]
        result[sr + r - pr, sc + c - pc] = info["floor"]
    cr, cc = info["counter"]
    for k in range(int(info["selected"].sum())):
        result[cr + k // 3, cc + k % 3] = info["material"]
    for cell in _marker_cells(info["point"], info["corner"]):
        result[cell] = info["bg"]
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Input-only: reconstruct connectivity and pull collection back to sources."""
    return _render_collection(_analyze(input_grid))


def _apart(a, b):
    r, c, h, w = a
    y, x, ph, pw = b
    return r + h < y or y + ph < r or c + w < x or x + pw < c


def _partition(rng, h, w, count):
    rects = [(0, 0, h, w)]
    for _ in range(count - 1):
        choices = []
        for i, (r, c, ph, pw) in enumerate(rects):
            for axis, length in enumerate((ph, pw)):
                for cut in range(3, length - 2):
                    if axis == 0:
                        new = [(r, c, cut, pw), (r + cut, c, ph - cut, pw)]
                    else:
                        new = [(r, c, ph, cut), (r, c + cut, ph, pw - cut)]
                    prospective = rects[:i] + rects[i + 1:] + new
                    if any(a >= 5 and b >= 5 for _, _, a, b in prospective):
                        choices.append((i, new, ph * pw))
        if not choices:
            return None
        weights = np.array([x[2] for x in choices], dtype=float)
        i, new, _ = choices[int(rng.choice(len(choices), p=weights / weights.sum()))]
        rects = rects[:i] + rects[i + 1:] + new
    labels = np.full((h, w), -1, np.int16)
    for i, (r, c, ph, pw) in enumerate(rects):
        labels[r:r + ph, c:c + pw] = i
    return rects, labels


def _conduits(rng, labels, counter):
    h, w = labels.shape
    walls = np.ones((h, w), bool)
    nodes = [(r, c) for r in range(1, h - 1, 2) for c in range(1, w - 1, 2)]
    start = nodes[int(rng.integers(len(nodes)))]
    seen, stack, links = {start}, [start], []
    walls[start] = False
    while stack:
        r, c = stack[-1]
        options = [(r + 2 * dr, c + 2 * dc) for dr, dc in STEPS
                   if 1 <= r + 2 * dr < h - 1 and 1 <= c + 2 * dc < w - 1
                   and (r + 2 * dr, c + 2 * dc) not in seen]
        if not options:
            stack.pop()
            continue
        nr, nc = options[int(rng.integers(len(options)))]
        mid = ((r + nr) // 2, (c + nc) // 2)
        walls[mid] = walls[nr, nc] = False
        links.append(mid)
        seen.add((nr, nc))
        stack.append((nr, nc))
    for k in rng.permutation(len(links))[:max(1, int(len(links) * rng.uniform(.06, .22)))]:
        walls[links[int(k)]] = True
    for _ in range(int(rng.integers(0, 4))):
        r, c = int(rng.integers(1, h - 2)), int(rng.integers(1, w - 2))
        ph, pw = int(rng.integers(2, 5)), int(rng.integers(2, 5))
        walls[r:min(h - 1, r + ph), c:min(w - 1, c + pw)] = False
    # A cut's two neighboring cells represent one shared conduit state.
    parent = list(range(h * w))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for r in range(h):
        for c in range(w):
            for nr, nc in ((r + 1, c), (r, c + 1)):
                if nr < h and nc < w and labels[r, c] != labels[nr, nc]:
                    parent[root(nr * w + nc)] = root(r * w + c)
    groups = {}
    for i in range(h * w):
        groups.setdefault(root(i), []).append(i)
    for cells in groups.values():
        if len(cells) < 2:
            continue
        rim = any(i // w in (0, h - 1) or i % w in (0, w - 1) for i in cells)
        open_state = not np.all(walls.flat[cells]) and rng.random() < .8
        walls.flat[cells] = rim or not open_state
    cr, cc = counter
    walls[cr:cr + 3, cc:cc + 3] = False
    return walls


def _sample_cells(rng, candidates, count, forced=()):
    chosen = list(dict.fromkeys(map(int, forced)))
    available = sorted(set(map(int, candidates)) - set(chosen))
    if count < len(chosen) or count - len(chosen) > len(available):
        return None
    if count > len(chosen):
        chosen.extend(map(int, rng.choice(available, count - len(chosen), replace=False)))
    return chosen


def _supply(rng, board, walls, labels, rects, counter, cp, request, roles):
    bg, wall, floor, marker, slot, material = roles
    distance = _distances(board, counter, wall)
    eligible = np.zeros(board.shape, bool)
    for r, c, ph, pw in rects:
        eligible[r + 1:r + ph - 1, c + 1:c + pw - 1] = True
    eligible &= ~walls & ~_counter_mask(board.shape, counter)
    active = eligible & (distance >= 0)
    inactive = np.flatnonzero(eligible & (distance < 0))
    remote = labels != cp
    mode, load = request
    if mode == "random":
        prob = rng.random()
        load = 0 if prob < .06 else int(rng.integers(10, 19)) if prob < .31 else int(rng.integers(1, 10))
    if int(active.sum()) < load or (load > 0 and not np.any(active & remote)):
        return None
    if mode in ("partial", "excess") and not len(inactive):
        return None
    if load == 0:
        if not len(inactive):
            return None
        chosen = []
    elif load <= 9:
        forced = int(rng.choice(np.flatnonzero(active & remote)))
        chosen = _sample_cells(rng, np.flatnonzero(active), load, [forced])
    else:
        rr, cc = np.indices(board.shape)
        cr, cx = counter
        manhattan = np.maximum(cr - rr, 0) + np.maximum(rr - cr - 2, 0)
        manhattan += np.maximum(cx - cc, 0) + np.maximum(cc - cx - 2, 0)
        chosen = None
        for a in rng.permutation(np.flatnonzero(active)):
            a = int(a)
            near = active & (distance <= distance.flat[a])
            far = active & (distance > distance.flat[a])
            if near.sum() < 9 or far.sum() < load - 9 or not np.any(near & remote):
                continue
            possible_b = far & (manhattan < manhattan.flat[a]) if mode == "excess" else far
            if not possible_b.any():
                continue
            b = int(rng.choice(np.flatnonzero(possible_b)))
            remote_cell = a if remote.flat[a] else int(rng.choice(np.flatnonzero(near & remote)))
            selected = _sample_cells(rng, np.flatnonzero(near), 9, [a, remote_cell])
            retained = _sample_cells(rng, np.flatnonzero(far), load - 9, [b])
            chosen = selected + retained
            break
        if chosen is None:
            return None
    minimum = 1 if mode in ("partial", "excess") or load == 0 else 0
    maximum = min(len(inactive), int(rng.integers(2, 9)))
    if maximum < minimum:
        return None
    distraction_count = int(rng.integers(minimum, maximum + 1))
    distraction = _sample_cells(rng, inactive, distraction_count)
    result = board.copy()
    result.flat[chosen + distraction] = material
    return result


def _scatter(rng, pieces, anchor, anchor_start, marker_cells, canvas):
    order = sorted((i for i in range(len(pieces)) if i != anchor), key=lambda i: -pieces[i].size)
    for _ in range(12):
        starts = [None] * len(pieces)
        starts[anchor] = anchor_start
        placed = [(*anchor_start, *pieces[anchor].shape)]
        for i in order:
            ph, pw = pieces[i].shape
            candidates = [(r, c) for r in range(1, canvas[0] - ph)
                          for c in range(1, canvas[1] - pw)]
            for k in rng.permutation(len(candidates)):
                r, c = candidates[int(k)]
                if any(r - 1 <= mr <= r + ph and c - 1 <= mc <= c + pw for mr, mc in marker_cells):
                    continue
                if all(_apart((r, c, ph, pw), other) for other in placed):
                    starts[i] = (r, c)
                    placed.append((r, c, ph, pw))
                    break
            if starts[i] is None:
                break
        if all(p is not None for p in starts):
            return starts
    return None


def _make_pair(rng, roles, request, test=False):
    bg, wall, floor, marker, slot, material = roles
    for _ in range(240):
        h = int(rng.integers(12, 18)) if test else int(rng.integers(10, 16))
        w = int(rng.integers(12, 19)) if test else int(rng.integers(10, 17))
        count = int(rng.integers(6, 8)) if test else int(rng.integers(3, 6))
        partition = _partition(rng, h, w, count)
        if partition is None:
            continue
        rects, labels = partition
        eligible = [i for i, (_, _, ph, pw) in enumerate(rects) if ph >= 5 and pw >= 5]
        cp = int(rng.choice(eligible))
        r, c, ph, pw = rects[cp]
        counter = (r + int(rng.integers(1, ph - 3)), c + int(rng.integers(1, pw - 3)))
        walls = _conduits(rng, labels, counter)
        board = np.where(walls, wall, floor).astype(np.int16)
        cr, cc = counter
        board[cr:cr + 3, cc:cc + 3] = slot
        board[cr + 1, cc + 1] = material
        base = int(np.ceil(np.sqrt(h * w * 3)))
        canvas = (min(30, max(h + 4, base + int(rng.integers(0, 5)))),
                  min(30, max(w + 4, base + int(rng.integers(0, 5)))))
        corner = str(rng.choice(CORNERS))
        oy = int(rng.integers(2 if corner[0] == "T" else 1,
                              canvas[0] - h - (1 if corner[0] == "B" else 0)))
        ox = int(rng.integers(2 if corner[1] == "L" else 1,
                              canvas[1] - w - (1 if corner[1] == "R" else 0)))
        point = (oy + (h - 1 if corner[0] == "B" else 0),
                 ox + (w - 1 if corner[1] == "R" else 0))
        local_point = (point[0] - oy, point[1] - ox)
        anchor = int(labels[local_point])
        pieces = [board[r:r + ph, c:c + pw].copy() for r, c, ph, pw in rects]
        try:
            restored, positions, origin = _stitch(pieces, anchor, point, corner, canvas, wall)
        except ValueError:
            continue
        if origin != (oy, ox) or not np.array_equal(restored, board):
            continue
        supplied = _supply(rng, board, walls, labels, rects, counter, cp, request, roles)
        if supplied is None:
            continue
        pieces = [supplied[r:r + ph, c:c + pw].copy() for r, c, ph, pw in rects]
        anchor_start = (oy + rects[anchor][0], ox + rects[anchor][1])
        markers = _marker_cells(point, corner)
        starts = _scatter(rng, pieces, anchor, anchor_start, markers, canvas)
        if starts is None:
            continue
        g = np.full(canvas, bg, np.int16)
        for p, (r, c) in zip(pieces, starts):
            g[r:r + p.shape[0], c:c + p.shape[1]] = p
        for cell in markers:
            g[cell] = marker
        # Assert local role cues, not hidden color dictionaries, remain unique.
        try:
            info = _decode(g)
        except ValueError:
            continue
        if (info["wall"], info["floor"], info["material"]) != (wall, floor, material):
            continue
        # Label from the constructed coordinate correspondence. solve separately
        # reconstructs the placement from the returned input during validation.
        selected, distance, supply = _collect(supplied, counter, wall, material)
        construction_info = dict(grid=g, selected=selected, labels=labels,
                                 positions=tuple((r, c) for r, c, _, _ in rects),
                                 starts=starts, floor=floor, material=material,
                                 counter=(starts[cp][0] + cr - rects[cp][0],
                                          starts[cp][1] + cc - rects[cp][1]),
                                 point=point, corner=corner, bg=bg)
        return {"input": g, "output": _render_collection(construction_info)}
    raise RuntimeError("Bounded sampling could not produce the requested conduit evidence")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if not _integer(seed) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    roles = tuple(map(int, rng.choice(10, 6, replace=False)))
    requests = [("partial", int(rng.choice((2, 4)))), ("excess", int(rng.integers(10, 19)))]
    requests += [("random", None)] * (nt - 2)
    rng.shuffle(requests)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, plan in (("train", requests), ("test", [("random", None)] * ne)):
        for request in plan:
            for _ in range(16):
                pair = _make_pair(rng, roles, request, test=(split == "test"))
                sig = (pair["input"].shape, pair["input"].tobytes())
                if sig not in seen:
                    seen.add(sig)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("Duplicate inputs after bounded sampling")
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(info):
    selected, distance, supply = info["selected"], info["distance"], info["supply"]
    retained = supply & (distance >= 0) & ~selected
    rr, cc = np.indices(info["board"].shape)
    r, c = info["board_counter"]
    manhattan = (np.maximum(r - rr, 0) + np.maximum(rr - r - 2, 0)
                 + np.maximum(c - cc, 0) + np.maximum(cc - c - 2, 0))
    normalization = {}
    partition = tuple(normalization.setdefault(int(x), len(normalization)) for x in info["labels"].ravel())
    return dict(canvas=info["grid"].shape, board=info["board"].shape,
                modules=len(info["pieces"]), corner=info["corner"],
                load=int(selected.sum()), reachable=int((supply & (distance >= 0)).sum()),
                unreachable=int((supply & (distance < 0)).sum()), excess=int(retained.sum()),
                remote=int((selected & (info["labels"] != info["counter_piece"])).sum()),
                reversal=bool(selected.any() and retained.any()
                              and manhattan[selected].max() > manhattan[retained].min()),
                gap=int(distance[retained].min() - distance[selected].max()) if retained.any() else 0,
                max_distance=int(distance.max()), partition=(info["board"].shape, partition),
                roles=tuple(info[k] for k in ("bg", "wall", "floor", "marker", "slot", "material")))


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, palettes = set(), set()
    features = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g, out = _grid(pair["input"]), _grid(pair["output"])
            assert g.shape == out.shape
            signature = (g.shape, g.tobytes())
            assert signature not in seen, "Duplicate input"
            seen.add(signature)
            info = _analyze(g)
            feature = _features(info)
            features[split].append(feature)
            palettes.add(feature["roles"])
            assert np.array_equal(out, _render_collection(info)), "Incorrect collection label"
            assert not np.any(out == info["marker"]), "L marker was not erased"
            ar, ac = info["positions"][info["anchor"]]
            assert (ar + info["origin"][0], ac + info["origin"][1]) == info["starts"][info["anchor"]]
            assert feature["load"] == min(9, feature["reachable"])
            assert not feature["excess"] or feature["gap"] > 0
            # These conservation checks audit source coordinates, not the label
            # renderer's hidden construction metadata.
            assert np.all(out[g == info["wall"]] == info["wall"])
            assert np.all(out[g == info["floor"]] == info["floor"])
            erased = (g == info["material"]) & (out == info["floor"])
            assert int(erased.sum()) == feature["load"]
            added = (g == info["slot"]) & (out == info["material"])
            assert int(added.sum()) == feature["load"] - int(feature["load"] >= 5)
            for p, (r, c) in zip(info["pieces"], info["starts"]):
                assert np.all(out[r:r + p.shape[0], c:c + p.shape[1]] != info["bg"])
            background = g == info["bg"]
            assert np.all(out[background] == info["bg"])
    assert len(palettes) == 1, "All examples must share color roles"
    train = features["train"]
    assert any(f["load"] in (2, 4) and f["remote"] and f["unreachable"] for f in train), "Missing partial cross-seam demonstration"
    assert any(f["excess"] and f["unreachable"] and f["remote"] and f["reversal"] for f in train), "Missing path/capacity demonstration"
    assert all(3 <= f["modules"] <= 5 for f in train)
    assert all(6 <= f["modules"] <= 7 for f in features["test"])


def _fingerprint(puzzle):
    return tuple((s, tuple((p["input"].shape, p["input"].tobytes(), p["output"].tobytes())
                           for p in puzzle[s])) for s in ("train", "test"))


def self_test():
    """Bounded 42-seed structural, semantic, rejection and transport audit."""
    puzzles, all_features = [], []
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.append(p)
        fs = [_features(_analyze(pair["input"])) for s in p for pair in p[s]]
        all_features.extend(fs)
        assert len({f["partition"] for f in fs}) >= 3
        assert len({f["load"] for f in fs}) >= 2
    assert len({_fingerprint(p) for p in puzzles}) == 42
    assert _fingerprint(generate_puzzle(0)) == _fingerprint(puzzles[0])
    for field, minimum in (("canvas", 20), ("board", 20), ("roles", 20),
                           ("partition", 80), ("max_distance", 15)):
        assert len({f[field] for f in all_features}) >= minimum, field
    assert {f["modules"] for f in all_features} == set(range(3, 8))
    assert {f["corner"] for f in all_features} == set(CORNERS)
    assert {f["load"] for f in all_features} == set(range(10))
    assert sum(f["reversal"] for f in all_features) >= 42
    assert sum(f["remote"] > 0 for f in all_features) >= 160
    # Targeted B semantics: a 14-cell connected supply line is split at nine;
    # payload is traversable, and an isolated cell is not collected.
    board = np.full((7, 21), 1, np.int16)
    board[1:4, 1:4] = 4
    board[2, 2] = 5
    board[2, 4:18] = 5
    board[5, 19] = 5
    selected, distance, supply = _collect(board, (1, 1), 1, 5)
    expected = np.zeros(board.shape, bool)
    expected[2, 4:13] = True
    assert np.array_equal(selected, expected)
    assert distance[2, 17] == 14 and distance[5, 19] == -1
    # Twelve distance-one candidates cannot use a hidden queue-order tie break.
    tied = np.ones((7, 7), np.int16)
    tied[1:-1, 1:-1] = 2
    tied[2:5, 2:5] = 4
    tied[3, 3] = 5
    d = _distances(tied, (2, 2), 1)
    tied[d == 1] = 5
    try:
        _collect(tied, (2, 2), 1, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("Ambiguous distance tie accepted")
    # A plain set of modules does not determine a unique board.
    try:
        _stitch([np.ones((3, 3), np.int16) for _ in range(4)], 0, (3, 3), "TL", (20, 20), 1)
    except ValueError as exc:
        assert "Ambiguous" in str(exc)
    else:
        raise AssertionError("Ambiguous reconstruction accepted")
    # A new scatter has the same module-local answer. Physical positions carry
    # no clue, and source-coordinate pullback must follow the new positions.
    pair = puzzles[0]["train"][0]
    info = _analyze(pair["input"])
    starts = _scatter(np.random.default_rng(9181), info["pieces"], info["anchor"],
                      info["starts"][info["anchor"]],
                      _marker_cells(info["point"], info["corner"]), info["grid"].shape)
    assert starts is not None and starts != info["starts"]
    scattered = np.full(info["grid"].shape, info["bg"], np.int16)
    for p, (r, c) in zip(info["pieces"], starts):
        scattered[r:r + p.shape[0], c:c + p.shape[1]] = p
    for cell in _marker_cells(info["point"], info["corner"]):
        scattered[cell] = info["marker"]
    changed = solve(scattered)
    for p, (r, c), (old_r, old_c) in zip(info["pieces"], starts, info["starts"]):
        assert np.array_equal(changed[r:r + p.shape[0], c:c + p.shape[1]],
                              pair["output"][old_r:old_r + p.shape[0], old_c:old_c + p.shape[1]])
    # Reject corruption rather than only testing generation against itself.
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in puzzles[0][s]] for s in puzzles[0]}
    damaged["test"][0]["output"] = damaged["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(damaged)
    except AssertionError:
        pass
    else:
        raise AssertionError("Corrupted output accepted")
    for nt, ne in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(7100 + nt + ne, nt, ne))
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("Illegal count accepted")
    return {"seeds": 42, "pairs": len(all_features),
            "canvas_shapes": len({f["canvas"] for f in all_features}),
            "board_shapes": len({f["board"] for f in all_features}),
            "partitions": len({f["partition"] for f in all_features}),
            "max_distances": len({f["max_distance"] for f in all_features}),
            "loads": sorted({f["load"] for f in all_features}),
            "path_reversals": sum(f["reversal"] for f in all_features),
            "remote_collection_pairs": sum(f["remote"] > 0 for f in all_features)}


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
