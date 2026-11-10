"""An anchored rectangular jigsaw reveals an aligned strict-union equation.

Only standard-library modules and NumPy are used. No parent code is imported.
The assembly search never consults the Boolean relation.
"""
from __future__ import annotations

from itertools import combinations
import hashlib
import json
import numpy as np

FOREGROUND = 2
OPERAND = 8
RESULT = 3
_PALETTE = (0, 1, 4, 5, 6, 7, 9)
_CORNERS = ("TL", "TR", "BL", "BR")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer) and np.all((a >= 0) & (a <= 9)), "invalid colors")
    return a.astype(np.int8, copy=True)


def _counts(num_train, num_test):
    for v, lo, hi in ((num_train, 2, 10), (num_test, 1, 4)):
        _require(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
                 and lo <= v <= hi, "illegal example count")
    _require(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    groups = []
    while unseen:
        p = min(unseen)
        unseen.remove(p)
        group = [p]
        for r, c in group:
            for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if q in unseen:
                    unseen.remove(q)
                    group.append(q)
        groups.append(group)
    return groups


def _marker_cells(point, corner):
    r, c = point
    dy = -1 if corner[0] == "T" else 1
    dx = -1 if corner[1] == "L" else 1
    return ((r + dy, c), (r, c + dx), (r + dy, c + dx))


def _extract(g):
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(np.count_nonzero(counts == counts[bg]) == 1, "background is not uniquely most frequent")
    _require(not np.any(np.isin(g, (OPERAND, RESULT))), "input contains output-only colors")
    candidates = []
    for color in range(10):
        if color == bg or counts[color] != 3:
            continue
        q = np.argwhere(g == color)
        r, c = q.min(0)
        y, x = q.max(0)
        if (y - r, x - c) != (1, 1):
            continue
        missing = {(int(r), int(c)), (int(r), int(x)), (int(y), int(c)), (int(y), int(x))}
        missing -= set(map(tuple, q.tolist()))
        point = missing.pop()
        if int(g[point]) in (bg, color):
            continue
        corner = {(1, 1): "TL", (1, 0): "TR", (0, 1): "BL", (0, 0): "BR"}[
            (point[0] - r, point[1] - c)]
        candidates.append((color, point, corner))
    _require(len(candidates) == 1, "expected one external L marker")
    marker, point, corner = candidates[0]
    pieces, anchor = [], None
    for cells in _components((g != bg) & (g != marker)):
        q = np.array(cells)
        r, c = map(int, q.min(0))
        y, x = map(int, q.max(0) + 1)
        _require(len(cells) == (y - r) * (x - c), "payload pieces must be solid rectangles")
        pieces.append((g[r:y, c:x].copy(), (r, c)))
        if point in cells:
            anchor = len(pieces) - 1
    _require(anchor is not None and 3 <= len(pieces) <= 8, "invalid payload partition")
    m, (r, c) = pieces[anchor]
    _require(point == (r + (m.shape[0] - 1 if corner[0] == "B" else 0),
                       c + (m.shape[1] - 1 if corner[1] == "R" else 0)), "marker does not pin a corner")
    return bg, marker, point, corner, pieces, anchor


def _lattice(board):
    """Recognize the backing and equal cells, without examining any unions."""
    h, w = board.shape
    if h % 2 or w % 2:
        return None
    logical = board[::2, ::2]
    if not np.array_equal(board, logical.repeat(2, 0).repeat(2, 1)):
        return None
    line = int(logical[0, 0])
    if line == FOREGROUND or not (np.all(logical[0] == line) and np.all(logical[-1] == line)
                                 and np.all(logical[:, 0] == line) and np.all(logical[:, -1] == line)):
        return None
    ys = np.flatnonzero(np.all(logical == line, axis=1))
    xs = np.flatnonzero(np.all(logical == line, axis=0))
    if len(ys) < 2 or len(xs) < 2:
        return None
    dy, dx = np.diff(ys), np.diff(xs)
    if not (np.all(dy == dy[0]) and np.all(dx == dx[0]) and min(dy[0], dx[0]) >= 3):
        return None
    rows, columns = len(ys) - 1, len(xs) - 1
    if not 4 <= rows * columns <= 9:
        return None
    mh, mw = int(dy[0] - 1), int(dx[0] - 1)
    blocks, masks, slots = [], [], []
    for r in ys[:-1]:
        for c in xs[:-1]:
            block = logical[r + 1:r + 1 + mh, c + 1:c + 1 + mw]
            if np.any(block == line) or not np.any(block == FOREGROUND):
                return None
            blocks.append(block)
            masks.append(block == FOREGROUND)
            slots.append((slice(2 * (r + 1), 2 * (r + 1 + mh)),
                          slice(2 * (c + 1), 2 * (c + 1 + mw))))
    colors = set(map(int, np.unique(np.concatenate([b.ravel() for b in blocks]))))
    if len(colors) != 2 or FOREGROUND not in colors:
        return None
    return dict(rows=rows, columns=columns, mh=mh, mw=mw, line=line,
                carrier=next(iter(colors - {FOREGROUND})), masks=masks, slots=slots)


def _assemble(g, extracted=None):
    """Exhaust all edge-compatible covers; reject distinct assembled pictures.

    The first empty square is the top-left of its covering rectangle. Anchoring
    and total area bound all possible board dimensions. No candidate list or
    search trace is saved, and no mask-union relation influences this stage.
    """
    bg, marker, point, corner, pieces, anchor = extracted or _extract(g)
    area = sum(m.size for m, _ in pieces)
    n = len(pieces)
    winner, winning_key, nodes, geometric = None, None, 0, 0
    for h in range(2, g.shape[0] + 1, 2):
        if area % h:
            continue
        w = area // h
        if w % 2 or w > g.shape[1]:
            continue
        oy = point[0] - (h - 1 if corner[0] == "B" else 0)
        ox = point[1] - (w - 1 if corner[1] == "R" else 0)
        if oy < 0 or ox < 0 or oy + h > g.shape[0] or ox + w > g.shape[1]:
            continue
        if any(m.shape[0] > h or m.shape[1] > w for m, _ in pieces):
            continue
        board = np.full((h, w), -1, dtype=np.int8)
        am, ap = pieces[anchor]
        ar, ac = ap[0] - oy, ap[1] - ox
        ah, aw = am.shape
        if ar < 0 or ac < 0 or ar + ah > h or ac + aw > w:
            continue
        board[ar:ar + ah, ac:ac + aw] = am
        positions = [None] * n
        positions[anchor] = (ar, ac)
        bits = [sum(((1 << m.shape[1]) - 1) << (r * w) for r in range(m.shape[0]))
                for m, _ in pieces]
        full = (1 << area) - 1

        def visit(occupied, remaining):
            nonlocal winner, winning_key, nodes, geometric
            nodes += 1
            _require(nodes <= 45000, "assembly search budget exhausted; uniqueness not established")
            if not remaining:
                geometric += 1
                info = _lattice(board)
                if info is None:
                    return
                key = (oy, ox, board.shape, board.tobytes())
                _require(winning_key is None or key == winning_key, "ambiguous assembled rectangle")
                winning_key = key
                if winner is None:
                    absolute = tuple((r + oy, c + ox) for r, c in positions)
                    winner = (board.copy(), (oy, ox), absolute, info)
                return
            free = full ^ occupied
            k = (free & -free).bit_length() - 1
            r, c = divmod(k, w)
            tried = set()
            for i in remaining:
                m = pieces[i][0]
                ph, pw = m.shape
                signature = (m.shape, m.tobytes())
                if signature in tried:
                    continue
                tried.add(signature)
                if r + ph > h or c + pw > w:
                    continue
                placed = bits[i] << k
                if occupied & placed:
                    continue
                compatible = True
                for side, values in (
                    (board[r - 1, c:c + pw] if r else np.array([], dtype=int), m[0]),
                    (board[r + ph, c:c + pw] if r + ph < h else np.array([], dtype=int), m[-1]),
                    (board[r:r + ph, c - 1] if c else np.array([], dtype=int), m[:, 0]),
                    (board[r:r + ph, c + pw] if c + pw < w else np.array([], dtype=int), m[:, -1]),
                ):
                    if side.size and np.any((side >= 0) & (side != values)):
                        compatible = False
                        break
                if not compatible:
                    continue
                board[r:r + ph, c:c + pw] = m
                positions[i] = (r, c)
                visit(occupied | placed, tuple(j for j in remaining if j != i))
                board[r:r + ph, c:c + pw] = -1
        visit(bits[anchor] << (ar * w + ac), tuple(i for i in range(n) if i != anchor))
    _require(winner is not None, "no edge-compatible lattice assembly")
    board, origin, positions, info = winner
    return board, origin, positions, info, nodes, geometric


def _relations(masks):
    triples = []
    for i, j in combinations(range(len(masks)), 2):
        a, b = masks[i], masks[j]
        if not (np.any(a & ~b) and np.any(b & ~a)):
            continue
        union = a | b
        for k, mask in enumerate(masks):
            if k not in (i, j) and np.array_equal(union, mask):
                triples.append((i, j, k))
    return triples


def _relation(masks):
    triples = _relations(masks)
    _require(bool(triples), "no strict union equation")
    signatures = {(tuple(sorted((masks[i].tobytes(), masks[j].tobytes()))),
                   masks[k].tobytes()) for i, j, k in triples}
    _require(len(signatures) == 1 and len({k for _, _, k in triples}) == 1,
             "competing union equations or duplicate results")
    return min(triples)


def _paint(board, info):
    relation = _relation(info["masks"])
    output = board.copy()
    for slot, color in ((relation[0], OPERAND), (relation[1], OPERAND), (relation[2], RESULT)):
        block = output[info["slots"][slot]]
        block[block == FOREGROUND] = color
    return output, relation


def _infer(g):
    extracted = _extract(g)
    board, origin, positions, info, nodes, geometric = _assemble(g, extracted)
    painted, relation = _paint(board, info)
    output = np.full_like(g, extracted[0])
    r, c = origin
    output[r:r + board.shape[0], c:c + board.shape[1]] = painted
    return output, (extracted, board, origin, positions, info, relation, nodes, geometric)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the complete answer from this input alone; no generation state."""
    return _infer(_grid(input_grid))[0]


def _orientation(i, j, columns):
    a, b = divmod(i, columns), divmod(j, columns)
    return "H" if a[0] == b[0] else "V" if a[1] == b[1] else "D"


def _make_masks(rng, config, orientation, duplicate, large_decoy):
    rows, cols, h, w = config
    n = rows * cols
    choices = [(i, j) for i, j in combinations(range(n), 2) if _orientation(i, j, cols) == orientation]
    for _ in range(1800):
        i, j = choices[int(rng.integers(len(choices)))]
        k = int(rng.choice([x for x in range(n) if x not in (i, j)]))
        # Shared bases include bars and open masks. Three
        # sampled pixels guarantee overlap and an exclusive contribution each.
        base = rng.random((h, w)) < float(rng.uniform(.12, .48))
        if rng.random() < .4:
            base[int(rng.integers(h)), :] = True
        a, b = base.copy(), base.copy()
        a |= rng.random((h, w)) < float(rng.uniform(.08, .35))
        b |= rng.random((h, w)) < float(rng.uniform(.08, .35))
        u, v, shared = map(int, rng.choice(h * w, 3, replace=False))
        a.flat[u], b.flat[u] = True, False
        a.flat[v], b.flat[v] = False, True
        a.flat[shared] = b.flat[shared] = True
        masks = [None] * n
        masks[i], masks[j], masks[k] = a, b, a | b
        free = [x for x in range(n) if masks[x] is None]
        if duplicate:
            later = [x for x in free if x > i]
            if not later:
                continue
            slot = int(rng.choice(later))
            masks[slot] = a.copy()
            free.remove(slot)
        if large_decoy:
            if not free or np.all(a | b):
                continue
            masks[free.pop()] = np.ones((h, w), dtype=bool)
        for slot in free:
            candidate = (a | b).copy() if rng.random() < .5 else base.copy()
            flips = rng.choice(h * w, int(rng.integers(1, min(4, h * w) + 1)), replace=False)
            candidate.flat[flips] = ~candidate.flat[flips]
            masks[slot] = candidate
        if any(not np.any(m) for m in masks):
            continue
        if not duplicate and len({m.tobytes() for m in masks}) != n:
            continue
        try:
            found = _relation(masks)
        except ValueError:
            continue
        if found == (i, j, k):
            return masks
    raise ValueError("bounded mask sampling exhausted")


def _board(config, masks, line, carrier):
    rows, cols, h, w = config
    logical = np.full((rows * (h + 1) + 1, cols * (w + 1) + 1), line, dtype=np.int8)
    for slot, mask in enumerate(masks):
        r, c = divmod(slot, cols)
        r, c = 1 + r * (h + 1), 1 + c * (w + 1)
        logical[r:r + h, c:c + w] = np.where(mask, FOREGROUND, carrier)
    return logical.repeat(2, 0).repeat(2, 1)


def _partition(rng, shape, count):
    rectangles = [(0, 0, *shape)]
    for _ in range(count - 1):
        choices = []
        for i, (r, c, h, w) in enumerate(rectangles):
            for axis, start, length in ((0, r, h), (1, c, w)):
                for cut in range(start + 3, start + length - 2):
                    if cut % 2:
                        choices.append((i, axis, cut))
        _require(bool(choices), "cannot partition into requested pieces")
        weights = np.array([rectangles[i][2] * rectangles[i][3] for i, _, _ in choices], dtype=float)
        i, axis, cut = choices[int(rng.choice(len(choices), p=weights / weights.sum()))]
        r, c, h, w = rectangles.pop(i)
        if axis == 0:
            rectangles.extend(((r, c, cut - r, w), (cut, c, r + h - cut, w)))
        else:
            rectangles.extend(((r, c, h, cut - c), (r, cut, h, c + w - cut)))
    return rectangles


def _gap(a, b):
    r, c, h, w = a
    y, x, ph, pw = b
    return r + h < y or y + ph < r or c + w < x or x + pw < c


def _scatter(rng, board, rectangles, corner, colors):
    bg, marker = colors
    h, w = board.shape
    canvas = (int(rng.integers(max(26, h + 2), 31)), int(rng.integers(max(26, w + 2), 31)))
    oy = int(rng.integers(1, canvas[0] - h))
    ox = int(rng.integers(1, canvas[1] - w))
    point = (oy + (h - 1 if corner[0] == "B" else 0), ox + (w - 1 if corner[1] == "R" else 0))
    local = (point[0] - oy, point[1] - ox)
    anchor = next(i for i, (r, c, ph, pw) in enumerate(rectangles)
                  if r <= local[0] < r + ph and c <= local[1] < c + pw)
    markers = _marker_cells(point, corner)
    for _ in range(12):
        positions = {anchor: (oy + rectangles[anchor][0], ox + rectangles[anchor][1])}
        placed = [(*positions[anchor], *rectangles[anchor][2:])]
        remaining = sorted((i for i in range(len(rectangles)) if i != anchor),
                           key=lambda i: -rectangles[i][2] * rectangles[i][3])
        for i in remaining:
            _, _, ph, pw = rectangles[i]
            options = [(r, c) for r in range(canvas[0] - ph + 1) for c in range(canvas[1] - pw + 1)]
            for index in rng.permutation(len(options)):
                r, c = options[int(index)]
                if any(r - 1 <= y <= r + ph and c - 1 <= x <= c + pw for y, x in markers):
                    continue
                if all(_gap((r, c, ph, pw), other) for other in placed):
                    positions[i] = (r, c)
                    placed.append((r, c, ph, pw))
                    break
            else:
                break
        if len(positions) == len(rectangles):
            grid = np.full(canvas, bg, dtype=np.int8)
            for i, (r, c, ph, pw) in enumerate(rectangles):
                y, x = positions[i]
                grid[y:y + ph, x:x + pw] = board[r:r + ph, c:c + pw]
            for p in markers:
                grid[p] = marker
            return grid, (oy, ox)
    raise ValueError("bounded scattering exhausted")


def _fractures(board, rectangles, info, relation):
    labels = np.empty(board.shape, dtype=int)
    for i, (r, c, h, w) in enumerate(rectangles):
        labels[r:r + h, c:c + w] = i
    return sum(len(np.unique(labels[info["slots"][s]][board[info["slots"][s]] == FOREGROUND])) > 1
               for s in relation)


def _make_pair(rng, config, count, corner, orientation, duplicate=False, large_decoy=False):
    masks = _make_masks(rng, config, orientation, duplicate, large_decoy)
    bg, marker, line, carrier = map(int, rng.choice(_PALETTE, 4, replace=False))
    board = _board(config, masks, line, carrier)
    info = _lattice(board)
    painted, relation = _paint(board, info)
    for _ in range(220):
        try:
            rectangles = _partition(rng, board.shape, count)
            if _fractures(board, rectangles, info, relation) < 1:
                continue
            grid, origin = _scatter(rng, board, rectangles, corner, (bg, marker))
            recovered, recovered_origin, _, _, _, _ = _assemble(grid)
            if recovered_origin != origin or not np.array_equal(recovered, board):
                continue
            output = np.full_like(grid, bg)
            r, c = origin
            output[r:r + board.shape[0], c:c + board.shape[1]] = painted
            return {"input": grid, "output": output}
        except ValueError:
            continue
    raise ValueError("bounded unique assembly sampling exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    _require(isinstance(seed, (int, np.integer)) and not isinstance(seed, (bool, np.bool_))
             and seed >= 0, "seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    small = ((2, 2, 2, 3), (2, 2, 3, 2), (2, 2, 3, 3))[int(rng.integers(3))]
    broad = ((3, 2, 2, 2), (2, 3, 2, 2), (3, 2, 2, 3), (2, 3, 3, 2))[int(rng.integers(4))]
    corners = ("TL", "BR") if rng.integers(2) else ("TR", "BL")
    test_corners = tuple(c for c in _CORNERS if c not in corners)
    tie = bool(rng.random() < .4)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ne)):
        for index in range(count):
            config = small if split == "train" and index % 2 == 0 else broad
            pieces = (3 if index == 0 else 4 if index == 1 else int(rng.integers(3, 6))) if split == "train" else int(rng.integers(6, 8))
            corner = corners[index % 2] if split == "train" else test_corners[index % 2]
            orientation = ("D", "H", "V")[index % 3] if split == "train" else ("H", "V", "D")[int(rng.integers(3))]
            duplicate = tie and config == broad and (index == 1 or split == "test")
            for _ in range(12):
                pair = _make_pair(rng, config, pieces, corner, orientation, duplicate,
                                  large_decoy=split == "train" and index == 0)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                raise ValueError("bounded distinct-input sampling exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _analyze_pair(pair):
    g, output = _grid(pair["input"]), _grid(pair["output"])
    _require(g.shape == output.shape, "canvas shape changed")
    predicted, details = _infer(g)
    _require(np.array_equal(output, predicted), "output differs from input-derived answer")
    extracted, board, origin, positions, info, relation, nodes, geometric = details
    bg, marker, point, corner, pieces, anchor = extracted
    _require(positions[anchor] == pieces[anchor][1], "anchor moved")
    _require(not np.any(output == marker), "marker retained")
    _require(all(output[p] == bg for p in _marker_cells(point, corner)), "marker not erased")
    labels = np.full(board.shape, -1, dtype=int)
    for i, ((m, _), (r, c)) in enumerate(zip(pieces, positions)):
        r, c = r - origin[0], c - origin[1]
        h, w = m.shape
        _require(np.all(labels[r:r + h, c:c + w] == -1), "assembly overlaps")
        labels[r:r + h, c:c + w] = i
        _require(np.array_equal(board[r:r + h, c:c + w], m), "piece changed during assembly")
    _require(np.all(labels >= 0), "assembly has gaps")
    plain = np.full_like(output, bg)
    r, c = origin
    plain[r:r + board.shape[0], c:c + board.shape[1]] = board
    _require(not np.array_equal(plain, output), "omitting parent B did not change answer")
    # A deliberately generous no-A baseline is even given the correct labels:
    # copy each solved, painted fragment back to its scattered input position.
    # It still cannot produce the assembled answer.
    inplace = np.full_like(g, bg)
    painted = output[r:r + board.shape[0], c:c + board.shape[1]]
    for (m, old), final in zip(pieces, positions):
        y, x = final[0] - r, final[1] - c
        ph, pw = m.shape
        inplace[old[0]:old[0] + ph, old[1]:old[1] + pw] = painted[y:y + ph, x:x + pw]
    _require(not np.array_equal(inplace, output), "omitting parent A did not change answer")
    fragments = sum(len(np.unique(labels[info["slots"][s]][board[info["slots"][s]] == FOREGROUND])) > 1
                    for s in relation)
    _require(fragments >= 1, "no selected foreground is fragmented across pieces")
    a, b = (info["masks"][i] for i in relation[:2])
    _require(np.any(a & ~b) and np.any(b & ~a) and np.any(a & b), "operand contribution/overlap missing")
    cin, cout = np.bincount(g.ravel(), minlength=10), np.bincount(output.ravel(), minlength=10)
    _require(cout[bg] == cin[bg] + 3 and cin[marker] == 3, "occupied area not conserved")
    _require(cout[info["line"]] == cin[info["line"]] and cout[info["carrier"]] == cin[info["carrier"]], "backing/lattice changed")
    _require(cout[FOREGROUND] + cout[OPERAND] + cout[RESULT] == cin[FOREGROUND], "foreground support changed")
    return dict(corner=corner, piece_count=len(pieces), shape=board.shape,
                canvas=g.shape, config=(info["rows"], info["columns"], info["mh"], info["mw"]),
                relation=relation, orientation=_orientation(*relation[:2], info["columns"]),
                raw_relations=len(_relations(info["masks"])), fragments=fragments,
                palette=(bg, marker, info["line"], info["carrier"]), nodes=nodes,
                max_decoy=max(int(m.sum()) for i, m in enumerate(info["masks"]) if i not in relation),
                result_area=int(info["masks"][relation[2]].sum()))


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "wrong puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, infos = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "wrong pair keys")
            g = _grid(pair["input"])
            key = (g.shape, g.tobytes())
            _require(key not in seen, "duplicate input")
            seen.add(key)
            infos[split].append(_analyze_pair(pair))
    train, test = infos["train"], infos["test"]
    _require(len({i["orientation"] for i in train}) >= 2, "demonstrations need varied operand placement")
    _require(any(i["orientation"] == "D" for i in train), "missing nonadjacent operand demonstration")
    _require(any(i["max_decoy"] > i["result_area"] for i in train), "largest-mask shortcut not refuted")
    if any(i["raw_relations"] > 1 for i in test):
        _require(any(i["raw_relations"] > 1 for i in train), "duplicate tie is unseen in training")
    _require({i["config"] for i in test} <= {i["config"] for i in train}, "test lattice geometry not demonstrated")
    return None


def _fingerprint(puzzle):
    h = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                g = np.asarray(pair[role], dtype=np.uint8)
                h.update(bytes(g.shape))
                h.update(g.tobytes())
    return h.hexdigest()


def self_test():
    """Stream 42 seeds; retain aggregate measurements, never all grid payloads."""
    signatures, configs, shapes, palettes, orientations, corners, piece_counts = (set() for _ in range(7))
    ties = fragments = max_nodes = 0
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_fingerprint(puzzle))
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _analyze_pair(pair)
                configs.add(info["config"])
                shapes.add(info["canvas"])
                palettes.add(info["palette"])
                orientations.add(info["orientation"])
                corners.add(info["corner"])
                piece_counts.add(info["piece_count"])
                ties += info["raw_relations"] > 1
                fragments += info["fragments"]
                max_nodes = max(max_nodes, info["nodes"])
    assert len(signatures) == 42 and len(configs) >= 6 and len(shapes) >= 8
    assert len(palettes) >= 40 and orientations == {"H", "V", "D"}
    assert corners == set(_CORNERS) and piece_counts >= {3, 4, 5, 6, 7}
    assert ties >= 8 and fragments >= 210
    p = generate_puzzle(123456, 5, 3)
    assert _fingerprint(p) == _fingerprint(generate_puzzle(123456, 5, 3))
    g = p["test"][0]["input"].copy()
    before = g.copy()
    assert np.array_equal(solve(g), p["test"][0]["output"]) and np.array_equal(g, before)
    # Reversing segmentation order cannot change assembly or the final answer.
    extracted = list(_extract(g))
    extracted[4] = list(reversed(extracted[4]))
    extracted[5] = len(extracted[4]) - 1 - extracted[5]
    reversed_board, reversed_origin, _, _, _, _ = _assemble(g, extracted)
    board, origin, _, _, _, _ = _assemble(g)
    assert reversed_origin == origin and np.array_equal(reversed_board, board)
    # Direct relation tests distinguish strictness, equivalent operand copies,
    # and genuinely ambiguous results without relying on sampled masks.
    a = np.array([[1, 1, 0], [0, 0, 0]], dtype=bool)
    b = np.array([[1, 0, 1], [0, 0, 0]], dtype=bool)
    assert _relation([a, b, a | b, a.copy()]) == (0, 1, 2)
    for masks in ([a, a.copy(), a.copy()], [a, b, a | b, (a | b).copy()]):
        try:
            _relation(masks)
        except ValueError:
            pass
        else:
            raise AssertionError("redundant or ambiguous union accepted")
    for nt, ne in ((2, 1), (10, 4)):
        q = generate_puzzle(733 + nt, nt, ne)
        assert len(q["train"]) == nt and len(q["test"]) == ne
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.0), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    damaged = {s: [{k: v.copy() for k, v in pair.items()} for pair in p[s]] for s in p}
    damaged["test"][0]["output"][0, 0] = OPERAND
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError("corrupt output accepted")
    return dict(seeds=42, pairs=210, configurations=len(configs), canvas_shapes=len(shapes),
                palettes=len(palettes), duplicate_equations=ties, selected_fragments=fragments,
                piece_counts=sorted(piece_counts), max_search_nodes=max_nodes)


if __name__ == "__main__":
    print(json.dumps(self_test(), separators=(",", ":")))
