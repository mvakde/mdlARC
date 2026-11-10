"""Seam-sealed stencils: a translation jigsaw changes ink connectivity.

Only standard library and NumPy. No parent code/data is imported. Nothing is
written by generation, validation, or self_test. Search rejects ambiguity.
"""
from collections import deque
import hashlib
import numpy as np

N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer cells required")
    _require(bool(((g >= 0) & (g <= 9)).all()), "ARC colors required")
    return g.astype(np.int8, copy=True)


def _components(mask, neighbors=N4):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        cells = [first]
        for r, c in cells:
            for dr, dc in neighbors:
                p = (r + dr, c + dc)
                if p in unseen:
                    unseen.remove(p)
                    cells.append(p)
        result.append(cells)
    return result


def _holes(body, neighbors=N4):
    """Exterior includes the space outside the paper, via padding."""
    padded = np.pad(body, 1)
    reached = np.zeros_like(padded)
    reached[0, 0] = True
    queue = deque([(0, 0)])
    h, w = padded.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbors:
            y, x = r + dr, c + dc
            if 0 <= y < h and 0 <= x < w and not padded[y, x] and not reached[y, x]:
                reached[y, x] = True
                queue.append((y, x))
    return (~padded & ~reached)[1:-1, 1:-1]


def _dilate(body):
    h, w = body.shape
    p = np.pad(body, 1)
    out = np.zeros_like(body)
    for dr, dc in N8:
        out |= p[1 + dr:1 + dr + h, 1 + dc:1 + dc + w]
    return out


def _zones(body):
    """Classify assembled ink, never paper tiles or outlined pixels."""
    labels = np.full(body.shape, -1, dtype=int)
    hollow = np.zeros_like(body)
    holes = _holes(body)
    objects = _components(body)
    for i, cells in enumerate(objects):
        rr, cc = np.array(cells).T
        labels[rr, cc] = i
    for cavity in _components(holes):
        owners = set()
        for r, c in cavity:
            for dr, dc in N4:
                y, x = r + dr, c + dc
                if 0 <= y < body.shape[0] and 0 <= x < body.shape[1] and labels[y, x] >= 0:
                    owners.add(int(labels[y, x]))
        _require(len(owners) == 1, "nested/collective cavity")
        hollow |= labels == next(iter(owners))
    outline = _dilate(body) & ~body & ~holes
    return dict(solid=body & ~hollow, hollow=hollow, cavity=holes,
                outline=outline, far=~body & ~holes & ~outline, labels=labels)


def _parse(g):
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(counts.argmax())
    _require(counts[bg] > g.size // 2, "background must exceed half the input")
    singletons = [i for i in range(10) if counts[i] == 1]
    _require(len(singletons) == 3, "three unique legend cells required")
    cells = np.argwhere(np.isin(g, singletons))
    r, c = cells.min(0)
    y, x = cells.max(0)
    _require(y - r == x - c == 1, "legend must be an L")
    missing = ({(int(r), int(c)), (int(r), int(x)), (int(y), int(c)), (int(y), int(x))}
               - set(map(tuple, cells.tolist())))
    _require(len(missing) == 1, "legend geometry")
    point = missing.pop()
    dr = -1 if point[0] == y else 1
    dc = -1 if point[1] == x else 1
    paper = int(g[point])
    _require(paper not in singletons and paper != bg, "marker must touch paper corner")
    roles = dict(background=bg, paper=paper,
                 outline=int(g[point[0], point[1] + dc]),
                 cavity=int(g[point[0] + dr, point[1]]),
                 hollow=int(g[point[0] + dr, point[1] + dc]))
    ink = [i for i in range(10) if counts[i] and i not in roles.values()]
    _require(len(ink) == 1, "one ink color required")
    roles['ink'] = ink[0]
    mask = (g == paper) | (g == ink[0])
    pieces, anchor = [], None
    for cells in _components(mask):
        rr, cc = np.array(cells).T
        r, c = int(rr.min()), int(cc.min())
        h, w = int(rr.max() - r + 1), int(cc.max() - c + 1)
        _require(len(cells) == h * w, "paper tiles must be full rectangles")
        if point in cells:
            anchor = len(pieces)
            _require(point == (r if dr == -1 else r + h - 1,
                               c if dc == -1 else c + w - 1), "anchor is not a tile corner")
        pieces.append((g[r:r + h, c:c + w].copy(), (r, c)))
    _require(anchor is not None and 2 <= len(pieces) <= 6, "tile count")
    # One blank row/column separates bounding boxes, including diagonals.
    for i, (a, (r, c)) in enumerate(pieces):
        for b, (s, t) in pieces[i + 1:]:
            _require(r + a.shape[0] < s or s + b.shape[0] < r or
                     c + a.shape[1] < t or t + b.shape[1] < c, "tiles touch")
    return roles, point, (dr, dc), pieces, anchor


def _assemble(shape, point, direction, pieces, anchor, matching=True):
    """Complete rectangle-tiling enumeration; identical rendered ties are harmless.

    A rectangle covering the first free cell must start there. The already fixed
    corner tile is inserted first, so this ordering is complete for any corner.
    With <=6 full rectangles there are at most sum(P(5,k))=326 nodes per
    area-factor rectangle: placement of the next tile has no free offset.
    """
    area = sum(m.size for m, _ in pieces)
    solutions = {}
    n = len(pieces)
    dr, dc = direction
    for h in range(1, shape[0] + 1):
        if area % h:
            continue
        w = area // h
        oy = point[0] - (h - 1 if dr == 1 else 0)
        ox = point[1] - (w - 1 if dc == 1 else 0)
        if w > shape[1] or oy < 0 or ox < 0 or oy + h > shape[0] or ox + w > shape[1]:
            continue
        if any(m.shape[0] > h or m.shape[1] > w for m, _ in pieces):
            continue
        board = np.full((h, w), -1, dtype=np.int8)
        labels = np.full((h, w), -1, dtype=np.int8)
        positions = [None] * n
        m, (ar, ac) = pieces[anchor]
        ar, ac = ar - oy, ac - ox
        if ar < 0 or ac < 0 or ar + m.shape[0] > h or ac + m.shape[1] > w:
            continue
        board[ar:ar + m.shape[0], ac:ac + m.shape[1]] = m
        labels[ar:ar + m.shape[0], ac:ac + m.shape[1]] = anchor
        positions[anchor] = (ar + oy, ac + ox)

        def visit(remaining):
            if not remaining:
                key = (board.shape, oy, ox, board.tobytes())
                if key not in solutions:
                    solutions[key] = (board.copy(), (oy, ox), tuple(positions), labels.copy())
                return len(solutions) >= 2
            r, c = np.argwhere(board < 0)[0]
            r, c = int(r), int(c)
            for i in remaining:
                tile = pieces[i][0]
                ph, pw = tile.shape
                if r + ph > h or c + pw > w or np.any(board[r:r + ph, c:c + pw] >= 0):
                    continue
                if matching:
                    edges = []
                    if r:
                        edges.append((board[r - 1, c:c + pw], tile[0]))
                    if r + ph < h:
                        edges.append((board[r + ph, c:c + pw], tile[-1]))
                    if c:
                        edges.append((board[r:r + ph, c - 1], tile[:, 0]))
                    if c + pw < w:
                        edges.append((board[r:r + ph, c + pw], tile[:, -1]))
                    if any(np.any((a >= 0) & (a != b)) for a, b in edges):
                        continue
                board[r:r + ph, c:c + pw] = tile
                labels[r:r + ph, c:c + pw] = i
                positions[i] = (r + oy, c + ox)
                stop = visit(tuple(j for j in remaining if j != i))
                board[r:r + ph, c:c + pw] = -1
                labels[r:r + ph, c:c + pw] = -1
                if stop:
                    return True
            return False

        visit(tuple(i for i in range(n) if i != anchor))
        if len(solutions) >= 2:
            break
    _require(len(solutions) == 1, "no unique boundary-matched rectangular assembly")
    return next(iter(solutions.values()))


def _render(board, origin, shape, roles):
    z = _zones(board == roles['ink'])
    result = np.full(shape, roles['background'], dtype=np.int8)
    r, c = origin
    patch = np.full(board.shape, roles['paper'], dtype=np.int8)
    for name in ('solid', 'hollow', 'cavity', 'outline'):
        patch[z[name]] = roles['ink' if name == 'solid' else name]
    result[r:r + board.shape[0], c:c + board.shape[1]] = patch
    return result


def solve(input_grid: np.ndarray) -> np.ndarray:
    g = _grid(input_grid)
    roles, point, direction, pieces, anchor = _parse(g)
    board, origin, _, _ = _assemble(g.shape, point, direction, pieces, anchor)
    return _render(board, origin, g.shape, roles)


def _crop(mask):
    q = np.argwhere(mask)
    r, c = q.min(0)
    y, x = q.max(0) + 1
    return mask[r:y, c:x].copy()


def _chamber(rng, opened=False, multiple=False, irregular=False):
    h = int(rng.integers(4 if irregular else 3, 6))
    w = int(rng.integers(6 if multiple and irregular else 5 if multiple else 4 if irregular else 3,
                         8 if multiple else 6))
    mask = np.ones((h, w), bool)
    mask[1:-1, 1:-1] = False
    if multiple:
        mask[:, int(rng.integers(2, w - 2))] = True
    if irregular and h >= 4 and w >= 4:
        pockets = [np.array(q) for q in _components(_holes(mask))]
        pocket = max(pockets, key=lambda q: int(np.prod(q.max(0) - q.min(0) + 1)))
        r, c = pocket.min(0)
        mask[r, c] = True
    if opened:
        # A straight door to exterior. Its bay is not a hole, even if the
        # eventual outline plugs the door visually.
        mask[0, int(rng.integers(1, w - 1))] = False
    # Optional asymmetric extensions carry classification away from the pocket.
    # Compare topology before/after so a lobe can never secretly seal the door.
    if rng.random() < .55:
        old_holes = _holes(mask)
        extended = np.pad(mask, ((0, 2), (0, 0)))
        col = int(rng.integers(w))
        extended[h:, col:min(w, col + int(rng.integers(1, 3)))] = True
        if (len(_components(extended)) == 1 and
                np.array_equal(_holes(extended), np.pad(old_holes, ((0, 2), (0, 0)))) and
                len(_components(extended, N8)) == 1):
            mask = _crop(extended)
    k = int(rng.integers(4))
    return np.rot90(mask, k).copy()


def _scene(rng, kind, multiple, irregular):
    """Pack ink objects before any paper cuts; no compulsory object slots."""
    if kind == 'contrast':
        closed = _chamber(rng, irregular=irregular)
        # Same silhouette and cavity, opened by one orthogonal passage.
        for k in range(4):
            opened = np.rot90(closed, k).copy()
            holes = _holes(opened)
            r, c = np.argwhere(holes)[0]
            opened[:r + 1, c] = False
            if len(_components(opened)) == 1 and not _holes(opened).any():
                opened = np.rot90(opened, -k).copy()
                break
        masks = [closed, opened]
    elif kind == 'rich':
        masks = [_chamber(rng, multiple=multiple, irregular=irregular)]
        if rng.random() < .8:
            masks.append(np.ones((1, int(rng.integers(1, 4))), bool))
    else:
        masks = []
        for _ in range(int(rng.integers(1, 4))):
            if rng.random() < .65:
                masks.append(_chamber(rng, opened=rng.random() < .4,
                                      multiple=multiple and rng.random() < .3,
                                      irregular=irregular))
            else:
                h, w = int(rng.integers(1, 4)), int(rng.integers(1, 5))
                m = np.ones((h, w), bool)
                if min(h, w) > 1 and rng.random() < .6:
                    m[0, 0] = False
                masks.append(m)
    # Shelf orientation and ordering are random; expansion below makes wall
    # thickness and cavity shape anisotropic rather than a fixed zoom factor.
    rng.shuffle(masks)
    horizontal = bool(rng.integers(2))
    if not horizontal:
        masks = [m.T for m in masks]
    h = max(m.shape[0] for m in masks) + 2
    w = sum(m.shape[1] for m in masks) + len(masks) + 1
    if h > 11 or w > 14:
        return None
    coarse = np.zeros((h, w), bool)
    x = 1
    for m in masks:
        y = int(rng.integers(1, h - m.shape[0]))
        coarse[y:y + m.shape[0], x:x + m.shape[1]] = m
        x += m.shape[1] + 1
    if not horizontal:
        coarse = coarse.T
    # Include quiet extra paper independently of ink count.
    coarse = np.pad(coarse, ((0, int(rng.integers(2))), (0, int(rng.integers(2)))))
    rows = rng.choice((1, 2, 3), size=coarse.shape[0], p=(.15, .7, .15))
    cols = rng.choice((1, 2, 3), size=coarse.shape[1], p=(.15, .7, .15))
    body = np.repeat(np.repeat(coarse, rows, axis=0), cols, axis=1)
    if min(body.shape) < 6 or max(body.shape) > 23 or body.size > 320:
        return None
    cuts_y = tuple(int(v + j) for v, length in zip(np.cumsum(np.r_[0, rows[:-1]]), rows)
                   for j in range(1, int(length)))
    cuts_x = tuple(int(v + j) for v, length in zip(np.cumsum(np.r_[0, cols[:-1]]), cols)
                   for j in range(1, int(length)))
    return body, cuts_y, cuts_x


def _partition(rng, shape, count, cuts_y, cuts_x):
    rects = [(0, 0, *shape)]
    for _ in range(count - 1):
        choices = []
        for i, (r, c, h, w) in enumerate(rects):
            choices += [(i, 0, y) for y in cuts_y if r + 2 <= y <= r + h - 2]
            choices += [(i, 1, x) for x in cuts_x if c + 2 <= x <= c + w - 2]
        if not choices:
            return None
        # Area weighting favors varied subdivisions instead of a long chain
        # of tiny slices. No partition family is tied to the marker corner.
        weights = np.array([rects[i][2] * rects[i][3] for i, _, _ in choices], float)
        i, axis, cut = choices[int(rng.choice(len(choices), p=weights / weights.sum()))]
        r, c, h, w = rects.pop(i)
        if axis == 0:
            rects.extend(((r, c, cut - r, w), (cut, c, r + h - cut, w)))
        else:
            rects.extend(((r, c, h, cut - c), (r, cut, h, c + w - cut)))
    return rects


def _scatter(rng, matrices, anchor, anchor_pos, marker, canvas):
    order = sorted((i for i in range(len(matrices)) if i != anchor),
                   key=lambda i: -matrices[i].size)
    for _ in range(12):
        positions = [None] * len(matrices)
        positions[anchor] = anchor_pos
        blocked = np.zeros(canvas, bool)

        def block(r, c, h, w):
            blocked[max(0, r - 1):min(canvas[0], r + h + 1),
                    max(0, c - 1):min(canvas[1], c + w + 1)] = True

        block(*anchor_pos, *matrices[anchor].shape)
        for r, c in marker:
            block(r, c, 1, 1)
        for i in order:
            h, w = matrices[i].shape
            p = np.pad(blocked.astype(np.int16), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
            sums = p[h:, w:] - p[:-h, w:] - p[h:, :-w] + p[:-h, :-w]
            candidates = np.argwhere(sums == 0)
            if not len(candidates):
                break
            r, c = map(int, candidates[int(rng.integers(len(candidates)))])
            positions[i] = (r, c)
            block(r, c, h, w)
        else:
            return positions
    return None


def _inspect(g):
    roles, point, direction, pieces, anchor = _parse(g)
    board, origin, positions, labels = _assemble(g.shape, point, direction, pieces, anchor)
    try:
        _assemble(g.shape, point, direction, pieces, anchor, matching=False)
        edge_needed = False
    except ValueError:
        # A matching assembly already exists, so this failure means multiple
        # distinct pictures fit the shapes if boundary evidence is omitted.
        edge_needed = True
    body = board == roles['ink']
    z = _zones(body)
    # Counterfactual: classify fragments before joining the paper.
    local_hollow = np.zeros_like(body)
    local_cavity = np.zeros_like(body)
    for (tile, _), (r, c) in zip(pieces, positions):
        r, c = r - origin[0], c - origin[1]
        a = _zones(tile == roles['ink'])
        local_hollow[r:r + tile.shape[0], c:c + tile.shape[1]] = a['hollow']
        local_cavity[r:r + tile.shape[0], c:c + tile.shape[1]] = a['cavity']
    object_count = len(_components(body))
    hollow_count = len(_components(z['hollow']))
    seam_hollow = int(np.count_nonzero(z['hollow'] & ~local_hollow))
    seam_cavity = int(np.count_nonzero(z['cavity'] & ~local_cavity))
    # Per-object evidence for whole-component recoloring across a seam.
    spanning = 0
    open_bays = 0
    multiple = 0
    irregular = 0
    for cells in _components(body):
        rr, cc = np.array(cells).T
        m = np.zeros_like(body)
        m[rr, cc] = True
        holes = _holes(m)
        spanning += int(holes.any() and len(np.unique(labels[rr, cc])) > 1)
        multiple += int(len(_components(holes)) > 1)
        for cavity in _components(holes):
            q = np.array(cavity)
            irregular += int(len(cavity) < int(np.prod(q.max(0) - q.min(0) + 1)))
        if not holes.any():
            crop = m[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
            for a in (crop, crop.T):
                if any(len(v := np.flatnonzero(row)) > 1 and np.any(np.diff(v) > 1) for row in a):
                    open_bays += 1
                    break
    return dict(roles=roles, point=point, direction=direction, pieces=pieces, anchor=anchor,
                board=board, origin=origin, positions=positions, labels=labels, zones=z,
                tiles=len(pieces), objects=object_count, hollows=hollow_count,
                edge_needed=edge_needed,
                seam_hollow=seam_hollow, seam_cavity=seam_cavity, spanning=spanning,
                open_bays=open_bays, multiple=multiple, irregular=irregular)


def _make_pair(rng, colors, count, kind, multiple, irregular, corner, seen):
    bg, paper, ink, outline, hollow, cavity = colors
    roles = dict(background=bg, paper=paper, ink=ink, outline=outline, hollow=hollow, cavity=cavity)
    for attempt in range(240):
        scene = _scene(rng, kind, multiple, irregular)
        if scene is None:
            continue
        body, cuts_y, cuts_x = scene
        h, w = body.shape
        rects = _partition(rng, body.shape, count, cuts_y, cuts_x)
        if rects is None:
            continue
        board = np.where(body, ink, paper).astype(np.int8)
        matrices = [board[r:r + ph, c:c + pw].copy() for r, c, ph, pw in rects]
        local_point = (h - 1 if corner[0] else 0, w - 1 if corner[1] else 0)
        anchor = next(i for i, (r, c, ph, pw) in enumerate(rects)
                      if r <= local_point[0] < r + ph and c <= local_point[1] < c + pw)
        ch = int(rng.integers(max(h + 2, 18, int(np.ceil(h * w * 2.2 / 30))), 31))
        cw_min = max(w + 2, 18, int(np.ceil(h * w * 2.2 / ch)))
        if cw_min > 30:
            continue
        cw = int(rng.integers(cw_min, 31))
        origin = (int(rng.integers(1, ch - h)), int(rng.integers(1, cw - w)))
        point = (origin[0] + local_point[0], origin[1] + local_point[1])
        dr, dc = (1 if corner[0] else -1), (1 if corner[1] else -1)
        marker = [(point[0], point[1] + dc), (point[0] + dr, point[1]),
                  (point[0] + dr, point[1] + dc)]
        anchor_pos = (origin[0] + rects[anchor][0], origin[1] + rects[anchor][1])
        positions = _scatter(rng, matrices, anchor, anchor_pos, marker, (ch, cw))
        if positions is None:
            continue
        g = np.full((ch, cw), bg, dtype=np.int8)
        for m, (r, c) in zip(matrices, positions):
            g[r:r + m.shape[0], c:c + m.shape[1]] = m
        for p, col in zip(marker, (outline, cavity, hollow)):
            g[p] = col
        if _signature(g) in seen:
            continue
        try:
            f = _inspect(g)
        except ValueError:
            continue
        if f['origin'] != origin or not np.array_equal(f['board'], board):
            continue
        # Every returned example contains a new closure caused by joining.
        if not f['seam_hollow'] or not f['seam_cavity']:
            continue
        if kind == 'contrast' and (not f['open_bays'] or not f['edge_needed']):
            continue
        if kind == 'rich' and ((multiple and not f['multiple']) or (irregular and not f['irregular'])):
            continue
        if not np.array_equal(_holes(body), _holes(body, N8)):
            continue
        if len(_components(body)) != len(_components(body, N8)):
            continue
        return {'input': g, 'output': _render(board, origin, g.shape, roles)}
    raise RuntimeError("bounded stencil construction exhausted")


def _signature(g):
    return (g.shape, g.astype(np.int8).tobytes())


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for value, name, low, high in ((num_train, 'num_train', 2, 10), (num_test, 'num_test', 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(name + " outside legal integer range")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    base = tuple(map(int, rng.choice(10, size=6, replace=False)))
    variable_palette = bool(rng.random() < .65)
    multiple = bool(rng.integers(2))
    irregular = bool(rng.integers(2))
    # The shift is more cuts and more joins, not a new topological convention.
    corners = [tuple(map(int, rng.integers(2, size=2))) for _ in range(num_train + num_test)]
    corners[1] = (1 - corners[0][0], 1 - corners[0][1])
    p, seen = {'train': [], 'test': []}, set()
    for split, n in (('train', num_train), ('test', num_test)):
        for i in range(n):
            colors = tuple(map(int, rng.choice(10, size=6, replace=False))) if variable_palette else base
            kind = 'contrast' if split == 'train' and i == 0 else 'rich' if split == 'train' and i == 1 else 'ordinary'
            count = int(rng.integers(3, 5)) if split == 'train' else int(rng.integers(5, 7))
            corner = corners[i if split == 'train' else num_train + i]
            pair = _make_pair(rng, colors, count, kind, multiple, irregular, corner, seen)
            seen.add(_signature(pair['input']))
            p[split].append(pair)
    validate_generated_puzzle(p)
    return p


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, "puzzle keys")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    a, b = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= a <= 10 and 1 <= b <= 4 and a > b, "pair counts")
    seen, infos = set(), {'train': [], 'test': []}
    for split in ('train', 'test'):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {'input', 'output'}, "pair keys")
            g, out = _grid(pair['input']), _grid(pair['output'])
            _require(g.shape == out.shape, "canvas changed")
            sig = _signature(g)
            _require(sig not in seen, "duplicate input")
            seen.add(sig)
            f = _inspect(g)
            roles, z, board, origin = f['roles'], f['zones'], f['board'], f['origin']
            _require(f['positions'][f['anchor']] == f['pieces'][f['anchor']][1], "anchor moved")
            _require(f['seam_hollow'] > 0 and f['seam_cavity'] > 0 and f['spanning'] > 0,
                     "assembly does not create a cavity")
            _require(np.array_equal(_holes(board == roles['ink']), _holes(board == roles['ink'], N8)), "diagonal leak")
            _require(len(_components(board == roles['ink'])) == len(_components(board == roles['ink'], N8)), "diagonal contact")
            # Independently inspect the five output zones and surrounding canvas.
            r, c = origin
            h, w = board.shape
            exterior = np.ones(g.shape, bool)
            exterior[r:r + h, c:c + w] = False
            _require(np.all(out[exterior] == roles['background']), "scatter/legend residue")
            patch = out[r:r + h, c:c + w]
            for name in ('solid', 'hollow', 'cavity', 'outline', 'far'):
                color = roles['ink' if name == 'solid' else 'paper' if name == 'far' else name]
                _require(np.all(patch[z[name]] == color), "incorrect zone: " + name)
            # Translation conservation is checked before topological repainting.
            rebuilt = np.full_like(board, -1)
            for (tile, _), (y, x) in zip(f['pieces'], f['positions']):
                y, x = y - r, x - c
                region = rebuilt[y:y + tile.shape[0], x:x + tile.shape[1]]
                _require(np.all(region == -1), "tile overlap")
                region[:] = tile
            _require(np.array_equal(rebuilt, board), "tile conservation")
            infos[split].append(f)
    _require(any(f['open_bays'] and f['zones']['solid'].any() for f in infos['train']), "open-bay demonstration missing")
    _require(any(f['edge_needed'] for f in infos['train']), "boundary evidence is unnecessary in training")
    _require(any(np.any(f['zones']['hollow'] & ~_dilate(f['zones']['cavity'])) for f in infos['train']),
             "whole-body recoloring not demonstrated")
    _require(len({f['direction'] for f in infos['train']}) >= 2, "corner demonstration missing")
    for key in ('multiple', 'irregular'):
        _require(not any(f[key] for f in infos['test']) or any(f[key] for f in infos['train']), "unshown cavity extension")


def self_test():
    """Stream 42 seeds, store only hashes/counters. No files or execution traces."""
    hashes, shapes, boards, layouts, object_shapes, palettes = set(), set(), set(), set(), set(), set()
    tile_counts, object_counts, directions = set(), set(), set()
    totals = dict(pairs=0, spanning=0, open_bays=0, multiple=0, irregular=0, edge_needed=0, seam_cavity_cells=0)
    sample = None
    for seed in range(42):
        p = generate_puzzle(seed)
        digest = hashlib.sha256()
        for split in ('train', 'test'):
            for pair in p[split]:
                g = pair['input']
                f = _inspect(g)
                body = f['board'] == f['roles']['ink']
                digest.update(g.tobytes())
                digest.update(pair['output'].tobytes())
                shapes.add(g.shape)
                boards.add(body.shape)
                layouts.add((body.shape, f['labels'].tobytes()))
                palettes.add(tuple(f['roles'].values()))
                tile_counts.add(f['tiles'])
                object_counts.add(f['objects'])
                directions.add(f['direction'])
                for cells in _components(body):
                    m = np.zeros_like(body)
                    rr, cc = np.array(cells).T
                    m[rr, cc] = True
                    m = _crop(m)
                    object_shapes.add((m.shape, m.tobytes()))
                for key in ('spanning', 'open_bays', 'multiple', 'irregular', 'edge_needed'):
                    totals[key] += f[key] > 0
                totals['pairs'] += 1
                totals['seam_cavity_cells'] += f['seam_cavity']
                _require(np.array_equal(solve(g.copy()), pair['output']), "input-only prediction")
        hashes.add(digest.hexdigest())
        if seed == 0:
            sample = p
    _require(len(hashes) == 42 and len(shapes) >= 35 and len(boards) >= 25, "size diversity")
    _require(len(layouts) >= 100 and len(object_shapes) >= 80 and len(palettes) >= 35, "structural diversity")
    _require(tile_counts == {3, 4, 5, 6} and len(object_counts) >= 2 and len(directions) == 4, "counts/corners")
    _require(totals['open_bays'] >= 42 and totals['multiple'] >= 10 and totals['irregular'] >= 10, "topology diversity")
    again = generate_puzzle(0)
    _require(all(np.array_equal(a[k], b[k]) for s in sample for a, b in zip(sample[s], again[s])
                 for k in ('input', 'output')), "determinism")
    # Meaningful corruptions cover body, fill, outline, unpainted paper, and
    # outer background. A retained marker or shifted board is also rejected.
    for role in ('solid', 'hollow', 'cavity', 'outline', 'far', 'outside'):
        damaged = {s: [{k: np.asarray(v).copy() for k, v in q.items()} for q in sample[s]] for s in sample}
        changed = False
        for pair in damaged['train']:
            f = _inspect(pair['input'])
            if role == 'outside':
                y, x = 0, 0
            else:
                q = np.argwhere(f['zones'][role])
                if not len(q):
                    continue
                y, x = q[0] + np.array(f['origin'])
            pair['output'][y, x] = (int(pair['output'][y, x]) + 1) % 10
            changed = True
            break
        _require(changed, "corruption zone absent")
        try:
            validate_generated_puzzle(damaged)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted corruption: " + role)
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4, 1.5), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    return dict(seeds=42, **totals, canvas_shapes=len(shapes), board_shapes=len(boards),
                partitions=len(layouts), object_masks=len(object_shapes), palettes=len(palettes))
