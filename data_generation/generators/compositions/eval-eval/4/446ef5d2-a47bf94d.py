"""Shattered switchboards: jointly infer panel placement and cable labels.

Only Python's standard library and NumPy are used. Generation draws paths and
cuts a board; prediction enumerates translations and traces the resulting ink.
No generation metadata, seed, examples, or query output is available to solve.
"""
from __future__ import annotations

import hashlib
import numpy as np


DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
DIAMOND = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
CHECKER = ~DIAMOND


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((0 <= a) & (a <= 9)), "grid colors")
    return a.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        cells = [p]
        for r, c in cells:
            for dr, dc in DIRS:
                q = (r + dr, c + dc)
                if q in unseen:
                    unseen.remove(q)
                    cells.append(q)
        result.append(cells)
    return result


def _marker_cells(point, corner):
    r, c = point
    dy = -1 if corner[0] == "T" else 1
    dx = -1 if corner[1] == "L" else 1
    return ((r + dy, c), (r, c + dx), (r + dy, c + dx))


def _extract(grid):
    bg = int(grid[0, 0])
    _require(np.all(grid[0] == bg) and np.all(grid[-1] == bg)
             and np.all(grid[:, 0] == bg) and np.all(grid[:, -1] == bg), "blank exterior border")
    markers = []
    for color in np.unique(grid):
        if color == bg:
            continue
        q = np.argwhere(grid == color)
        if len(q) != 3 or tuple(np.ptp(q, axis=0)) != (1, 1):
            continue
        r, c = map(int, q.min(0))
        missing = ({(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)}
                   - set(map(tuple, q.tolist()))).pop()
        if grid[missing] in (bg, color):
            continue
        corner = {(1, 1): "TL", (1, 0): "TR", (0, 1): "BL", (0, 0): "BR"}[
            (missing[0] - r, missing[1] - c)]
        markers.append((int(color), missing, corner))
    _require(len(markers) == 1, "one unambiguous external L is required")
    marker, point, corner = markers[0]
    pieces, anchor = [], None
    for cells in _components((grid != bg) & (grid != marker)):
        q = np.array(cells)
        r, c = map(int, q.min(0))
        h, w = map(int, q.max(0) - q.min(0) + 1)
        _require(len(cells) == h * w and min(h, w) >= 2, "panels are filled rectangles")
        if point in cells:
            anchor = len(pieces)
            _require(point == (r + (h - 1 if corner[0] == "B" else 0),
                               c + (w - 1 if corner[1] == "R" else 0)), "L must mark a panel corner")
        pieces.append((grid[r:r + h, c:c + w].copy(), (r, c)))
    _require(3 <= len(pieces) <= 6 and anchor is not None, "three to six panels and an anchor")
    counts = np.bincount(np.concatenate([p.ravel() for p, _ in pieces]), minlength=10)
    substrate = int(counts.argmax())
    _require(counts[substrate] > max(counts[c] for c in range(10) if c != substrate), "unique substrate majority")
    _require(grid[point] == substrate, "marked corner is bare substrate")
    return dict(bg=bg, marker=marker, point=point, corner=corner, pieces=pieces,
                anchor=anchor, substrate=substrate, canvas=grid.shape)


def _tilings(info):
    """Complete search: the next rectangle starts at the first uncovered cell.

    At most six panels give at most sum(P(5,k))=326 visits per feasible area
    factor once the anchor is fixed. No truncation, optimization score, or tie
    breaking by input order is used.
    """
    pieces, anchor = info["pieces"], info["anchor"]
    area = sum(p.size for p, _ in pieces)
    pr, pc = info["point"]
    corner = info["corner"]
    for h in range(7, info["canvas"][0] - 1):
        if area % h:
            continue
        w = area // h
        if not 7 <= w <= info["canvas"][1] - 2:
            continue
        oy = pr - (h - 1 if corner[0] == "B" else 0)
        ox = pc - (w - 1 if corner[1] == "R" else 0)
        if oy < 1 or ox < 1 or oy + h >= info["canvas"][0] or ox + w >= info["canvas"][1]:
            continue
        if any(p.shape[0] > h or p.shape[1] > w for p, _ in pieces):
            continue
        configs = []
        for p, _ in pieces:
            ph, pw = p.shape
            bits = sum(((1 << pw) - 1) << (r * w) for r in range(ph))
            configs.append((ph, pw, bits))
        ar, ac = pieces[anchor][1][0] - oy, pieces[anchor][1][1] - ox
        ah, aw, abits = configs[anchor]
        if ar < 0 or ac < 0 or ar + ah > h or ac + aw > w:
            continue
        positions = [None] * len(pieces)
        positions[anchor] = (ar, ac)
        full = (1 << area) - 1

        def visit(occupied, remaining):
            if not remaining:
                board = np.empty((h, w), dtype=np.int8)
                for (p, _), (r, c) in zip(pieces, positions):
                    board[r:r + p.shape[0], c:c + p.shape[1]] = p
                yield board, (oy, ox), tuple(positions)
                return
            free = full ^ occupied
            r, c = divmod((free & -free).bit_length() - 1, w)
            for i in remaining:
                ph, pw, bit = configs[i]
                if r + ph > h or c + pw > w:
                    continue
                placed = bit << (r * w + c)
                if placed & occupied:
                    continue
                positions[i] = (r, c)
                yield from visit(occupied | placed, tuple(j for j in remaining if j != i))

        yield from visit(abits << (ar * w + ac), tuple(i for i in range(len(pieces)) if i != anchor))


def _block(block, substrate):
    colors = set(map(int, np.unique(block))) - {substrate}
    if not colors:
        return "blank", None
    _require(len(colors) == 1, "multicolor terminal")
    color = colors.pop()
    for name, stencil in (("solid", np.ones((3, 3), dtype=bool)),
                          ("diamond", DIAMOND), ("checker", CHECKER)):
        if np.array_equal(block, np.where(stencil, color, substrate)):
            return name, color
    raise ValueError("malformed terminal")


def _trace(board, conductor, bridge, centers):
    h, w = board.shape
    visible = set(map(tuple, np.argwhere((board == conductor) | (board == bridge)).tolist()))
    graph = {}
    endpoints = {(3, c) for c in centers} | {(h - 4, c) for c in centers}
    crossings, marked = set(), set()
    for r, c in visible:
        edges = {d: (r + dr, c + dc) for d, (dr, dc) in enumerate(DIRS)
                 if (r + dr, c + dc) in visible}
        graph[r, c] = edges
        if (r, c) in endpoints:
            _require(set(edges) == ({2} if r == 3 else {0}), "bad bank lead")
        elif len(edges) == 4:
            _require(board[r, c] == conductor and bridge is not None, "unmarked junction")
            pairs = [(a, b) for a, b in ((0, 2), (1, 3))
                     if board[edges[a]] == bridge and board[edges[b]] == bridge]
            _require(bool(pairs), "crossing needs opposite bridge marks")
            for a, b in pairs:
                marked.update((edges[a], edges[b]))
            crossings.add((r, c))
        else:
            _require(len(edges) == 2, "wire fragments do not reconnect")
    if bridge is not None:
        _require(marked == set(map(tuple, np.argwhere(board == bridge).tolist())), "orphan bridge ink")
    routes, permutation, seen = [], [], set()
    for c in centers:
        p, incoming, route, states = (3, c), 0, [], set()
        while True:
            _require((p, incoming) not in states, "closed wire loop")
            states.add((p, incoming))
            route.append(p)
            seen.add(p)
            if p[0] == h - 4 and p in endpoints:
                permutation.append(centers.index(p[1]))
                break
            if len(route) == 1:
                outgoing = 2
            elif p in crossings:
                outgoing = (incoming + 2) % 4
            else:
                _require(incoming in graph[p], "invalid path entrance")
                outgoing = next(d for d in graph[p] if d != incoming)
            p = graph[p][outgoing]
            incoming = (outgoing + 2) % 4
            _require(p[0] != 3, "cable returns to source bank")
        routes.append(route)
    _require(sorted(permutation) == list(range(len(centers))) and seen == visible, "nonbijective or unused cable")
    return permutation, routes, crossings


def _interpret(board, substrate, *, check_labels=True):
    """Infer direction and ink roles from cells, then check global circuit rules."""
    answers = []
    for rotation in range(4):
        b = np.rot90(board, rotation)
        h, w = b.shape
        if h < 9 or w < 9:
            continue
        for conductor in set(map(int, np.unique(b[3]))) - {substrate}:
            centers = list(map(int, np.flatnonzero(b[3] == conductor)))
            if not 2 <= len(centers) <= 4 or centers[0] < 2 or centers[-1] > w - 3:
                continue
            if any(y - x < 4 for x, y in zip(centers, centers[1:])):
                continue
            if centers != list(np.flatnonzero(b[-4] == conductor)):
                continue
            try:
                footprints = np.zeros(b.shape, dtype=bool)
                banks = []
                for r in (0, h - 3):
                    banks.append([_block(b[r:r + 3, c - 1:c + 2], substrate) for c in centers])
                    for c in centers:
                        footprints[r:r + 3, c - 1:c + 2] = True
                outside = ~footprints
                outside[3:h - 3] = False
                _require(np.all(b[outside] == substrate), "ink outside terminal sites")
                _require(all(k in ("blank", "solid", "diamond") for k, _ in banks[0])
                         and all(k in ("blank", "checker") for k, _ in banks[1]), "bank stencils")
                labels = {v for bank in banks for _, v in bank if v is not None}
                field_colors = set(map(int, np.unique(b[3:h - 3]))) - {substrate, conductor}
                _require(len(field_colors) <= 1, "too many structural inks")
                bridge = next(iter(field_colors)) if field_colors else None
                _require(labels.isdisjoint({conductor, bridge}), "label and routing roles overlap")
                perm, routes, crossings = _trace(b, conductor, bridge, centers)
                output = b.copy()
                conflicts = 0
                for i, j in enumerate(perm):
                    a, z = banks[0][i], banks[1][j]
                    clues = {v for _, v in (a, z) if v is not None}
                    _require(bool(clues), "a cable has no color clue")
                    conflicts += int(len(clues) != 1)
                    if check_labels:
                        _require(len(clues) == 1, "opposite labels disagree")
                    color = a[1] if a[1] is not None else z[1]
                    for r, c, stencil in ((0, centers[i], DIAMOND), (h - 3, centers[j], CHECKER)):
                        output[r:r + 3, c - 1:c + 2] = np.where(stencil, color, substrate)
                answers.append(dict(board=b, completed=np.rot90(output, -rotation).copy(),
                                    rotation=rotation, conductor=conductor, bridge=bridge,
                                    centers=centers, banks=banks, permutation=tuple(perm),
                                    routes=routes, crossings=crossings, footprints=footprints,
                                    conflicts=conflicts))
            except (ValueError, StopIteration, KeyError):
                continue
    _require(len(answers) == 1, "board lacks one valid circuit interpretation")
    return answers[0]


def _reconstruct(grid, *, audit=False):
    info = _extract(grid)
    winner, key = None, None
    geometries, circuits, colorless, rejected_labels = 0, 0, 0, 0
    for board, origin, positions in _tilings(info):
        geometries += 1
        try:
            parsed = _interpret(board, info["substrate"])
        except ValueError:
            if audit:
                try:
                    weak = _interpret(board, info["substrate"], check_labels=False)
                    colorless += 1
                    rejected_labels += int(weak["conflicts"] > 0)
                except ValueError:
                    pass
            continue
        circuits += 1
        colorless += 1
        # Distinct piece permutations producing the same visible board are benign.
        candidate_key = (board.shape, origin, board.tobytes(), parsed["completed"].tobytes())
        _require(key is None or candidate_key == key, "ambiguous circuit jigsaw")
        key = candidate_key
        winner = (board, origin, positions, parsed)
    _require(winner is not None, "no valid circuit assembly")
    board, (oy, ox), positions, parsed = winner
    out = np.full(grid.shape, info["bg"], dtype=np.int8)
    out[oy:oy + board.shape[0], ox:ox + board.shape[1]] = parsed["completed"]
    info.update(board=board, origin=(oy, ox), positions=positions, parsed=parsed,
                geometric_tilings=geometries, valid_tilings=circuits,
                colorless_tilings=colorless, rejected_label_tilings=rejected_labels)
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Return the unique repaired switchboard using only this input."""
    return _reconstruct(_grid(input_grid))[0]


def _moves(target, rng):
    """Construct paths with a vacant exterior track; no drawing template lookup."""
    n = len(target)
    state, moves = list(range(n)) + [None], []
    for j in map(int, rng.permutation(n)):
        if state[j] == target[j]:
            continue
        saved = state[j]
        moves.append((j, n))
        state[n], state[j] = state[j], None
        vacant = j
        while target[vacant] != saved:
            old = state.index(target[vacant])
            moves.append((old, vacant))
            state[vacant], state[old] = state[old], None
            vacant = old
        moves.append((n, vacant))
        state[vacant], state[n] = state[n], None
    return moves


def _draw(rng, n, checksum=False):
    pitch = int(rng.integers(4, 6)) if n <= 3 else 4
    centers = [2 + i * pitch for i in range(n)]
    columns = centers + [centers[-1] + int(rng.integers(3, 5))]
    w = columns[-1] + int(rng.integers(2, 4))
    if checksum:
        # Noncommuting switches have the same boundary port profile.
        moves = [(0, n), (1, 0), (n, 1), (1, n), (2, 1), (n, 2)]
        if rng.integers(2):
            moves = moves[3:] + moves[:3]
        rows = [5, 7, 9, 12, 14, 16]
        end = 18
    else:
        target = list(map(int, rng.permutation(n)))
        if rng.random() < .15:
            target = list(range(n))
        moves = _moves(target, rng)
        if len(moves) <= 4 and rng.random() < .4:
            lane = int(rng.integers(n))
            moves = [(lane, n), (n, lane)] + moves
        row = 3 + int(rng.integers(2, 4))
        rows = []
        for _ in moves:
            rows.append(row)
            row += int(rng.integers(2, 4))
        end = (rows[-1] if rows else 5) + int(rng.integers(2, 4))
    h = end + 4
    _require(h <= 26 and w <= 26, "routing exceeds board budget")
    paths = [[(3, c)] for c in centers]
    state, previous = list(range(n)) + [None], 3
    for row, (old, new) in zip(rows, moves):
        for lane, route in enumerate(state):
            if route is not None:
                paths[route].extend((r, columns[lane]) for r in range(previous + 1, row + 1))
        route = state[old]
        step = 1 if columns[new] > columns[old] else -1
        paths[route].extend((row, c) for c in range(columns[old] + step, columns[new] + step, step))
        state[new], state[old] = state[old], None
        previous = row
    for lane, route in enumerate(state):
        if route is not None:
            paths[route].extend((r, columns[lane]) for r in range(previous + 1, end + 1))
    # Abstract values: exterior 0, substrate 1, conductor 2, bridge 3, L 4,
    # terminal colors 5..9. A fresh permutation later gives local color roles.
    board = np.ones((h, w), dtype=np.int8)
    visits = {}
    for i, path in enumerate(paths):
        for p in path:
            board[p] = 2
            visits.setdefault(p, set()).add(i)
    for (r, c), owners in visits.items():
        if len(owners) == 2:
            for d in ((0, 2) if rng.integers(2) else (1, 3)):
                dr, dc = DIRS[d]
                board[r + dr, c + dc] = 3
    return board, centers, paths


def _terminals(rng, board, centers, paths, lesson):
    n = len(paths)
    labels = list(map(int, rng.permutation(np.arange(5, 10))[:n]))
    if lesson == "free" and n > 2 and rng.random() < .35:
        labels[-1] = labels[0]
    options = [("solid", "blank"), ("diamond", "blank"), ("blank", "checker"),
               ("solid", "checker"), ("diamond", "checker")]
    if lesson == "checksum":
        modes = [("solid" if i == 0 else "diamond", "checker" if i == 0 else "blank") for i in range(n)]
    elif lesson == "reverse":
        modes = [options[2], options[1], options[3]] + [options[int(rng.integers(5))] for _ in range(n - 3)]
        rng.shuffle(modes)
    else:
        modes = [options[int(rng.integers(5))] for _ in range(n)]
        if all(a == "blank" for a, _ in modes):
            modes[0] = options[1]
    completed = board.copy()
    for i, path in enumerate(paths):
        j = centers.index(path[-1][1])
        for r, c, pattern, mode in ((0, centers[i], DIAMOND, modes[i][0]),
                                    (board.shape[0] - 3, centers[j], CHECKER, modes[i][1])):
            completed[r:r + 3, c - 1:c + 2] = np.where(pattern, labels[i], 1)
            mask = np.ones((3, 3), dtype=bool) if mode == "solid" else pattern
            board[r:r + 3, c - 1:c + 2] = (1 if mode == "blank" else np.where(mask, labels[i], 1))
    return board, completed


def _partition(rng, h, w, count, checksum=False, mixed=False):
    if checksum:
        rects = [(0, 0, 4, w), (4, 0, 7, w), (11, 0, 7, w), (18, 0, 4, w)]
    else:
        rects = [(0, 0, h, w)]
    first_axis = None
    while len(rects) < count:
        choices = [(i, axis) for i, (_, _, ph, pw) in enumerate(rects)
                   for axis, length in enumerate((ph, pw)) if length >= 5]
        if mixed and len(rects) == 2:
            choices = [(i, axis) for i, axis in choices if axis != first_axis]
        _require(bool(choices), "cannot subdivide panels")
        i, axis = choices[int(rng.integers(len(choices)))]
        if len(rects) == 1:
            first_axis = axis
        r, c, ph, pw = rects.pop(i)
        cut = int(rng.integers(2, (ph if axis == 0 else pw) - 1))
        if axis == 0:
            rects.extend(((r, c, cut, pw), (r + cut, c, ph - cut, pw)))
        else:
            rects.extend(((r, c, ph, cut), (r, c + cut, ph, pw - cut)))
    return rects


def _separated(a, b):
    r, c, h, w = a
    y, x, ph, pw = b
    return r + h < y or y + ph < r or c + w < x or x + pw < c


def _scatter(rng, pieces, final, anchor, marker_cells, canvas):
    order = sorted((i for i in range(len(pieces)) if i != anchor), key=lambda i: -pieces[i].size)
    for _ in range(9):
        positions = {anchor: final[anchor]}
        placed = [(*final[anchor], *pieces[anchor].shape)]
        for i in order:
            h, w = pieces[i].shape
            candidates = [(r, c) for r in range(1, canvas[0] - h) for c in range(1, canvas[1] - w)]
            found = False
            for k in rng.permutation(len(candidates)):
                r, c = candidates[int(k)]
                if any(r - 1 <= y <= r + h and c - 1 <= x <= c + w for y, x in marker_cells):
                    continue
                if all(_separated((r, c, h, w), a) for a in placed):
                    positions[i] = (r, c)
                    placed.append((r, c, h, w))
                    found = True
                    break
            if not found:
                break
        if len(positions) == len(pieces):
            return [positions[i] for i in range(len(pieces))]
    return None


def _make_pair(rng, lesson, count, rotation, corner):
    for attempt in range(90):
        checksum = lesson == "checksum" or (lesson == "free" and attempt >= 65)
        n = 3 if lesson in ("checksum", "reverse") or checksum else int(rng.integers(2, 5))
        try:
            board, centers, paths = _draw(rng, n, checksum)
        except ValueError:
            continue
        clue_lesson = "checksum" if checksum and lesson == "free" else lesson
        board, completed = _terminals(rng, board, centers, paths, clue_lesson)
        canonical_rects = _partition(rng, *board.shape, count, checksum, mixed=lesson == "reverse" and not checksum)
        partition = np.empty(board.shape, dtype=int)
        for i, (r, c, h, w) in enumerate(canonical_rects):
            partition[r:r + h, c:c + w] = i
        board, completed, partition = (np.rot90(a, rotation).copy() for a in (board, completed, partition))
        h, w = board.shape
        pieces, rects = [], []
        for i in range(len(canonical_rects)):
            q = np.argwhere(partition == i)
            r, c = map(int, q.min(0))
            ph, pw = map(int, q.max(0) - q.min(0) + 1)
            pieces.append(board[r:r + ph, c:c + pw].copy())
            rects.append((r, c, ph, pw))
        canvas = (30, 30)
        # Less padding when the actual pieces and anchor permit it.
        if not checksum and rng.random() < .7:
            base = int(np.ceil(np.sqrt(board.size * 1.9)))
            canvas = (min(30, max(h + 4, base) + int(rng.integers(5))),
                      min(30, max(w + 4, base) + int(rng.integers(5))))
        origin = (int(rng.integers(2, canvas[0] - h - 1)), int(rng.integers(2, canvas[1] - w - 1)))
        point = (origin[0] + (h - 1 if corner[0] == "B" else 0),
                 origin[1] + (w - 1 if corner[1] == "R" else 0))
        local = (point[0] - origin[0], point[1] - origin[1])
        anchor = int(partition[local])
        markers = _marker_cells(point, corner)
        final = [(origin[0] + r, origin[1] + c) for r, c, _, _ in rects]
        positions = _scatter(rng, pieces, final, anchor, markers, canvas)
        if positions is None:
            continue
        palette = np.asarray(rng.permutation(10), dtype=np.int8)
        inp = np.zeros(canvas, dtype=np.int8)
        out = inp.copy()
        for p, (r, c) in zip(pieces, positions):
            inp[r:r + p.shape[0], c:c + p.shape[1]] = p
        for p in markers:
            inp[p] = 4
        out[origin[0]:origin[0] + h, origin[1]:origin[1] + w] = completed
        inp, out = palette[inp], palette[out]
        try:
            prediction, info = _reconstruct(inp, audit=lesson == "checksum")
            _require(np.array_equal(prediction, out), "construction and tracing disagree")
            _require(any(old != (info["origin"][0] + r, info["origin"][1] + c)
                         for (_, old), (r, c) in zip(info["pieces"], info["positions"])), "no assembly movement")
            if lesson == "checksum":
                _require(info["rejected_label_tilings"] > 0, "missing global assembly lesson")
        except ValueError:
            continue
        return {"input": inp, "output": out}
    raise RuntimeError("bounded construction attempts exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    """A deterministic puzzle, with two guaranteed teaching examples."""
    for name, value, lo, hi in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not lo <= value <= hi:
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    rotation = int(rng.integers(4))
    corners = ("TL", "TR", "BL", "BR")
    for split, total in (("train", num_train), ("test", num_test)):
        for i in range(total):
            lesson = ("checksum" if i == 0 else "reverse" if i == 1 else "free") if split == "train" else "free"
            count = 4 if lesson == "checksum" else int(rng.integers(3, 5)) if split == "train" else int(rng.integers(5, 7))
            direction = rotation if i < 2 and split == "train" else int(rng.integers(4))
            corner = str(rng.choice(corners))
            for _ in range(10):
                pair = _make_pair(rng, lesson, count, direction, corner)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    break
            else:
                raise RuntimeError("distinct input budget exhausted")
            seen.add(key)
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    return puzzle


def _inspect_pair(pair, audit=False):
    inp, out = _grid(pair["input"]), _grid(pair["output"])
    _require(inp.shape == out.shape, "canvas changed")
    predicted, info = _reconstruct(inp, audit=audit)
    _require(np.array_equal(predicted, out), "incorrect output")
    parsed, board = info["parsed"], info["board"]
    canonical_out = np.rot90(parsed["completed"], parsed["rotation"])
    _require(np.array_equal(parsed["board"][~parsed["footprints"]], canonical_out[~parsed["footprints"]]),
             "nonterminal panel ink changed")
    anchor = info["anchor"]
    oy, ox = info["origin"]
    r, c = info["positions"][anchor]
    _require((oy + r, ox + c) == info["pieces"][anchor][1], "anchor translated")
    _require(any(old != (oy + r, ox + c) for (_, old), (r, c)
                 in zip(info["pieces"], info["positions"])), "panels already assembled")
    occupancy = np.zeros(board.shape, dtype=bool)
    for (panel, _), (r, c) in zip(info["pieces"], info["positions"]):
        ph, pw = panel.shape
        _require(not occupancy[r:r + ph, c:c + pw].any(), "panel overlap")
        occupancy[r:r + ph, c:c + pw] = True
        _require(np.array_equal(board[r:r + ph, c:c + pw], panel), "panel content not conserved")
    _require(occupancy.all(), "unfilled assembly")
    _require(not np.any(out == info["marker"]), "marker retained")
    _require(all(out[p] == info["bg"] for p in _marker_cells(info["point"], info["corner"])), "marker footprint")
    outside = np.ones(inp.shape, dtype=bool)
    outside[oy:oy + board.shape[0], ox:ox + board.shape[1]] = False
    _require(np.all(out[outside] == info["bg"]), "vacated panels retained")
    _require(not np.array_equal(inp, out), "identity pair")
    return info


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "pair lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "counts")
    seen, evidence, split_counts = set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair keys")
            info = _inspect_pair(pair, audit=split == "train")
            key = (np.asarray(pair["input"]).shape, np.asarray(pair["input"], dtype=np.int8).tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            split_counts[split].append(len(info["pieces"]))
            if split == "train":
                parsed = info["parsed"]
                if any(p.shape[0] < info["board"].shape[0] and p.shape[1] < info["board"].shape[1]
                       for p, _ in info["pieces"]):
                    evidence.add("mixed_cuts")
                if info["rejected_label_tilings"]:
                    evidence.add("global")
                for i, j in enumerate(parsed["permutation"]):
                    a, b = parsed["banks"][0][i], parsed["banks"][1][j]
                    if a[0] == "blank":
                        evidence.add("reverse")
                    if b[0] == "blank":
                        evidence.add("forward")
                    evidence.update((a[0], b[0]))
                if parsed["crossings"]:
                    evidence.add("crossing")
    _require({"global", "reverse", "forward", "solid", "diamond", "checker", "crossing", "mixed_cuts"} <= evidence,
             "training lacks a required lesson")
    _require(min(split_counts["test"]) > max(split_counts["train"]), "missing finer-fragmentation shift")


def self_test():
    """Bounded in-memory coverage; no files, traces, or retained run payloads."""
    shapes, structures, partitions, palettes, counts, directions, corners = (set() for _ in range(7))
    checksum_lessons, reverse_lessons, repeated, crossing_free = 0, 0, 0, 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _inspect_pair(pair, audit=split == "train")
                parsed = info["parsed"]
                shapes.add(pair["input"].shape)
                counts.add(len(info["pieces"]))
                directions.add(parsed["rotation"])
                corners.add(info["corner"])
                palettes.add((info["bg"], info["substrate"], info["marker"], parsed["conductor"]))
                partitions.add((info["board"].shape, tuple(sorted((r, c, *p.shape)
                               for (p, _), (r, c) in zip(info["pieces"], info["positions"])))))
                rename = {}
                normalized = bytes(rename.setdefault(int(v), len(rename)) for v in info["board"].ravel())
                structures.add((info["board"].shape, hashlib.sha256(normalized).digest()))
                checksum_lessons += int(info["rejected_label_tilings"] > 0)
                clues = []
                for i, j in enumerate(parsed["permutation"]):
                    a, b = parsed["banks"][0][i], parsed["banks"][1][j]
                    reverse_lessons += int(a[0] == "blank")
                    clues.append(a[1] if a[1] is not None else b[1])
                repeated += int(len(set(clues)) < len(clues))
                crossing_free += int(not parsed["crossings"])
    assert checksum_lessons >= 42 and reverse_lessons >= 42
    assert counts == {3, 4, 5, 6} and directions == set(range(4)) and corners == {"TL", "TR", "BL", "BR"}
    assert len(structures) >= 150 and len(partitions) >= 100 and len(palettes) >= 100
    assert repeated > 0 and crossing_free > 0 and len(shapes) >= 8
    a, b = generate_puzzle(731, 2, 1), generate_puzzle(731, 2, 1)
    assert all(np.array_equal(p[k], q[k]) for s in a for p, q in zip(a[s], b[s]) for k in p)
    original = a["test"][0]["input"].copy()
    for rotation in range(4):
        for reflected in (False, True):
            query, expected = (a["test"][0][k] for k in ("input", "output"))
            if reflected:
                query, expected = query[:, ::-1], expected[:, ::-1]
            query, expected = np.rot90(query, rotation), np.rot90(expected, rotation)
            assert np.array_equal(solve(query), expected)
    assert np.array_equal(original, a["test"][0]["input"]), "solve mutated its input"
    # Remove only the receiving checksum. Both switch orders now satisfy every
    # local port and circuit constraint; arbitrary choice must be rejected.
    checksum_pair = next(p for p in a["train"] if _inspect_pair(p, True)["rejected_label_tilings"])
    info = _inspect_pair(checksum_pair, True)
    parsed = info["parsed"]
    changed = parsed["board"].copy()
    for j, (kind, _) in enumerate(parsed["banks"][1]):
        if kind == "checker":
            c = parsed["centers"][j]
            changed[-3:, c - 1:c + 2] = info["substrate"]
    changed = np.rot90(changed, -parsed["rotation"])
    ambiguous = checksum_pair["input"].copy()
    for (panel, (oldr, oldc)), (r, c) in zip(info["pieces"], info["positions"]):
        ph, pw = panel.shape
        ambiguous[oldr:oldr + ph, oldc:oldc + pw] = changed[r:r + ph, c:c + pw]
    try:
        solve(ambiguous)
    except ValueError as exc:
        assert "ambiguous" in str(exc)
    else:
        raise AssertionError("arbitrary switch order accepted without checksum")
    # Reversing the enumeration order cannot select a different assembly.
    info["pieces"] = list(reversed(info["pieces"]))
    info["anchor"] = len(info["pieces"]) - 1 - info["anchor"]
    reconstructed = []
    for board, origin, _ in _tilings(info):
        try:
            parsed = _interpret(board, info["substrate"])
        except ValueError:
            continue
        out = np.full(checksum_pair["input"].shape, info["bg"], dtype=np.int8)
        r, c = origin
        out[r:r + board.shape[0], c:c + board.shape[1]] = parsed["completed"]
        reconstructed.append(out)
    assert reconstructed and all(np.array_equal(out, checksum_pair["output"]) for out in reconstructed)
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(1000 + 10 * nt + nq, nt, nq)
            assert len(p["train"]) == nt and len(p["test"]) == nq
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: np.asarray(v).copy() for k, v in p.items()} for p in a[s]] for s in a}
    damaged["test"][0]["output"][0, 0] = (int(damaged["test"][0]["output"][0, 0]) + 1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupted exterior accepted")
