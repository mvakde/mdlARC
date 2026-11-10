"""Reconstruct a fragmented shutter, dock the fused mask, and project its holes.

Only standard-library modules and NumPy are used. Solving has no random state,
training examples, stored puzzles, parent imports, or generation metadata.
"""

from itertools import product
import numpy as np


_DIRS = {"T": (-1, 0), "B": (1, 0), "L": (0, -1), "R": (0, 1)}
_OPPOSITE = {"T": "B", "B": "T", "L": "R", "R": "L"}
_CONTRASTS = ("positive", "closure", "recessed", "open", "partial", "motion")
_A_CONTRASTS = ("ordered_keys", "both_seam_axes")


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _grid(value):
    g = np.asarray(value)
    _require(g.ndim == 2 and all(1 <= d <= 30 for d in g.shape), "grid dimensions")
    _require(np.issubdtype(g.dtype, np.integer), "integer grid required")
    _require(bool(np.all((0 <= g) & (g <= 9))), "ARC colors required")
    return g.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        stack = [first]
        for r, c in stack:
            for p in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if p in unseen:
                    unseen.remove(p)
                    stack.append(p)
        result.append(stack)
    return result


def _holes(mask):
    h, w = mask.shape
    return [q for q in _components(~mask)
            if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in q)]


def _profile(mask):
    holes = _holes(mask)
    extreme = int(np.nonzero(mask)[1].max())
    enclosed = {p for q in holes for p in q}
    rows = {r for r, c in enclosed if c == extreme - 1 and mask[r, extreme]}
    naive = {r for r in range(mask.shape[0])
             if extreme > 0 and mask[r, extreme] and not mask[r, extreme - 1]}
    all_hole_rows = {r for r, c in enclosed}
    partial = [q for q in holes
               if 0 < len({r for r, c in q if c == extreme - 1}) < len({r for r, c in q})]
    whole_hole_rows = {r for q in holes if any(c == extreme - 1 for r, c in q)
                       for r, c in q}
    row_extremes = {r: int(np.nonzero(mask[r])[0].max())
                    for r in range(mask.shape[0]) if mask[r].any()}
    veto_rows = {r for r, e in row_extremes.items()
                 if e < extreme and (r, e - 1) in enclosed}
    return dict(holes=holes, rows=rows, naive=naive,
                decoy_rows=all_hole_rows - rows, partial=partial, extreme=extreme,
                whole_hole_rows=whole_hole_rows, veto_rows=veto_rows)


def _scene(g):
    """Rotate to objects left of the unique exclusive rail, blank band right."""
    found = []
    for turn in range(4):
        a = np.rot90(g, turn)
        h, w = a.shape
        for c in range(1, w - 1):
            rail = int(a[0, c])
            if not np.all(a[:, c] == rail) or np.count_nonzero(a == rail) != h:
                continue
            bg = int(a[0, -1])
            if bg == rail or not np.all(a[:, c + 1:] == bg):
                continue
            if np.any(a[:, :c] != bg):
                found.append((a.copy(), turn, bg, rail, c))
    _require(len(found) == 1, "unique rail and blank projection side required")
    return found[0]


def _runs(cells):
    """Consecutive (along-coordinate, value) runs, retaining their first index."""
    runs, current = [], []
    for p in sorted(cells):
        if current and p[0] != current[-1][0] + 1:
            runs.append(current)
            current = []
        current.append(p)
    if current:
        runs.append(current)
    return runs


def _extract(g):
    a, turn, bg, rail, col = _scene(g)
    marker_options = []
    for color in np.unique(a[:, :col]):
        color = int(color)
        q = np.argwhere(a == color)
        if color == bg or len(q) != 3:
            continue
        r0, c0 = q.min(0)
        r1, c1 = q.max(0)
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        points = set(map(tuple, q.tolist()))
        missing = {(int(r0), int(c0)), (int(r0), int(c1)),
                   (int(r1), int(c0)), (int(r1), int(c1))} - points
        point = missing.pop()
        if int(a[point]) not in (bg, rail, color):
            marker_options.append((color, points, point))
    _require(len(marker_options) == 1, "one exclusive external L required")
    ink, marker_cells, point = marker_options[0]
    payload = (a[:, :col] != bg) & (a[:, :col] != ink)
    values, counts = np.unique(a[:, :col][payload], return_counts=True)
    _require(len(values) >= 2 and np.count_nonzero(counts == counts.max()) == 1,
             "unique majority carrier required")
    base = int(values[counts.argmax()])
    pieces, anchor = [], None
    for cells in _components(payload):
        q = np.array(cells)
        r0, c0 = map(int, q.min(0))
        r1, c1 = map(int, q.max(0) + 1)
        m = np.full((r1 - r0, c1 - c0), -1, dtype=np.int8)
        for r, c in cells:
            m[r - r0, c - c0] = a[r, c]
        if point in cells:
            anchor = len(pieces)
            _require(point in ((r0, c0), (r0, c1 - 1),
                               (r1 - 1, c0), (r1 - 1, c1 - 1)),
                     "L must identify an occupied bounding-box corner")
        pieces.append(dict(matrix=m, old=(r0, c0)))
    _require(2 <= len(pieces) <= 6 and anchor is not None, "2..6 keyed fragments required")
    ports = {}
    for i, piece in enumerate(pieces):
        m = piece["matrix"]
        mask = m >= 0
        h, w = m.shape
        lines = {}
        for r, c in np.argwhere(mask & (m != base)):
            r, c = int(r), int(c)
            exposed = [side for side, (dy, dx) in _DIRS.items()
                       if not (0 <= r + dy < h and 0 <= c + dx < w
                               and mask[r + dy, c + dx])]
            _require(len(exposed) == 1, "seam ink must lie on one straight exposed face")
            side = exposed[0]
            fixed, along = (r, c) if side in ("T", "B") else (c, r)
            lines.setdefault((side, fixed), []).append((along, int(m[r, c])))
        for (side, fixed), line in lines.items():
            for run in _runs(line):
                _require(2 <= len(run) <= 4, "seam signatures have length 2..4")
                sequence = tuple(value for _, value in run)
                start = (fixed, run[0][0]) if side in ("T", "B") else (run[0][0], fixed)
                axis = "v" if side in ("T", "B") else "h"
                ports.setdefault((axis, sequence), []).append((i, side, start))
    links = [[] for _ in pieces]
    for code, ends in ports.items():
        _require(len(ends) == 2, "each seam signature must have exactly two faces")
        (i, side, p), (j, facing, q) = ends
        _require(i != j and facing == _OPPOSITE[side], "ports must face oppositely")
        dy, dx = _DIRS[side]
        delta = (p[0] + dy - q[0], p[1] + dx - q[1])
        links[i].append((j, delta))
        links[j].append((i, (-delta[0], -delta[1])))
    positions = [None] * len(pieces)
    positions[anchor] = pieces[anchor]["old"]
    queue = [anchor]
    for i in queue:
        r, c = positions[i]
        for j, (dy, dx) in links[i]:
            proposed = (r + dy, c + dx)
            if positions[j] is None:
                positions[j] = proposed
                queue.append(j)
            else:
                _require(positions[j] == proposed, "inconsistent port cycle")
    _require(all(p is not None for p in positions), "port graph must connect every fragment")
    rows, columns = [], []
    for piece, (r, c) in zip(pieces, positions):
        yy, xx = np.nonzero(piece["matrix"] >= 0)
        rows.extend((yy + r).tolist())
        columns.extend((xx + c).tolist())
    top, left = min(rows), min(columns)
    bottom, right = max(rows), max(columns)
    picture = np.full((bottom - top + 1, right - left + 1), -1, dtype=np.int8)
    labels = np.full(picture.shape, -1, dtype=np.int8)
    for i, (piece, (r, c)) in enumerate(zip(pieces, positions)):
        m = piece["matrix"]
        mask = m >= 0
        sub = picture[r - top:r - top + m.shape[0], c - left:c - left + m.shape[1]]
        lab = labels[r - top:r - top + m.shape[0], c - left:c - left + m.shape[1]]
        _require(not np.any(sub[mask] >= 0), "fragments overlap after joining")
        sub[mask] = m[mask]
        lab[mask] = i
    mask = picture >= 0
    _require(len(_components(mask)) == 1, "assembled shutter must be connected")
    shift = col - 1 - right
    _require(shift >= 0, "shutter must approach its rail without crossing it")
    _require(0 <= top <= bottom < a.shape[0] and left + shift >= 0,
             "docked shutter must fit the work area")
    return dict(canonical=a, turn=turn, bg=bg, rail=rail, col=col, ink=ink,
                marker_cells=marker_cells, point=point, base=base, pieces=pieces,
                anchor=anchor, positions=positions, top=top, left=left,
                shift=shift, picture=picture, labels=labels, ports=ports,
                profile=_profile(mask))


def _render(info):
    out = info["canonical"].copy()
    out[:, :info["col"]] = info["bg"]
    picture = info["picture"]
    h, w = picture.shape
    r, c = info["top"], info["left"] + info["shift"]
    target = out[r:r + h, c:c + w]
    mask = picture >= 0
    target[mask] = picture[mask]
    for row in info["profile"]["rows"]:
        out[r + row, info["col"] + 1:] = info["ink"]
    return np.rot90(out, -info["turn"]).copy()


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer roles, join keyed faces, dock by global extent, flood-fill, project."""
    return _render(_extract(_grid(input_grid)))


def _port_candidates(labels):
    """Possible informative seams; other contacts remain unconstrained in color."""
    h, w = labels.shape
    lines = {}
    for side, (dy, dx) in (("R", (0, 1)), ("B", (1, 0))):
        for r in range(h - dy):
            for c in range(w - dx):
                i, j = int(labels[r, c]), int(labels[r + dy, c + dx])
                if i < 0 or j < 0 or i == j:
                    continue
                good = True
                for y, x, k, excluded in ((r, c, i, (dy, dx)),
                                          (r + dy, c + dx, j, (-dy, -dx))):
                    for sy, sx in _DIRS.values():
                        if (sy, sx) == excluded:
                            continue
                        if not (0 <= y + sy < h and 0 <= x + sx < w
                                and labels[y + sy, x + sx] == k):
                            good = False
                if good:
                    fixed, along = (c, r) if side == "R" else (r, c)
                    lines.setdefault((i, j, side, fixed), []).append((along, 0))
    candidates = []
    for (i, j, side, fixed), line in lines.items():
        for run in _runs(line):
            for length in range(2, min(4, len(run)) + 1):
                for offset in range(len(run) - length + 1):
                    along = run[offset][0]
                    cells = [(along + k, fixed) if side == "R" else (fixed, along + k)
                             for k in range(length)]
                    dy, dx = _DIRS[side]
                    other = [(r + dy, c + dx) for r, c in cells]
                    candidates.append((i, j, side, cells, other))
    return candidates


def _connected_port_graph(labels):
    n = int(labels.max()) + 1
    seen = {0}
    edges = {(i, j) for i, j, _, _, _ in _port_candidates(labels)}
    for _ in range(n):
        for i, j in edges:
            if i in seen or j in seen:
                seen.update((i, j))
    return len(seen) == n


def _make_mask(rng, kind, is_test):
    h = int(rng.integers(13, 17) if is_test else rng.integers(10, 15))
    w = int(rng.integers(12, 17) if is_test else rng.integers(10, 15))
    right_width = int(rng.integers(4, 6))
    cut = w - right_width
    spur = int(rng.integers(2, 4)) if kind in ("veto", "mixed") else 0
    mask = np.zeros((h, w + spur), dtype=bool)
    mask[:, :w] = True
    # Open indentation behind an intact leading wall, visibly reaching outside.
    open_height = int(rng.integers(2, 4))
    mask[:open_height, w - 3:w - 1] = False
    # A stepped pocket crosses the mandatory vertical seam. Some rows touch the
    # one-cell leading wall and others stop one column earlier.
    start, stop = open_height + 1, h - 3
    if spur:
        # Leave at least one formerly leading pocket row above the lobe. Its
        # local wall stays thin while the joined global extreme moves farther.
        lobe_height = int(rng.integers(4, min(6, h - start - 1) + 1))
        mask[h - lobe_height:, w:] = True
    back = max(1, cut - int(rng.integers(1, 3)))
    for r in range(start, stop):
        end = w - 1 if (r - start) % 2 == 0 else w - 2
        row_back = min(cut - 1, back + int(rng.integers(2)))
        mask[r, row_back:end] = False
    # A solid bridge can divide the seam-crossing pocket into two apertures.
    # Both sides retain contact and noncontact rows, so no new rule is needed.
    if stop - start >= 5 and rng.random() < .55:
        bridge = int(rng.integers(start + 2, stop - 2))
        mask[bridge, back:w - 1] = True
    # Quiet, wholly recessed enclosed pockets can survive within one fragment.
    if cut >= 5:
        row = int(rng.integers(1, h - 3))
        mask[row:row + int(rng.integers(1, 3)), 1:cut - 1] = False
    # Independently add separated closed pockets in quiet material. A one-cell
    # foreground buffer prevents accidental hole merging or opening.
    for _ in range(int(rng.integers(4))):
        for _ in range(12):
            ph, pw = int(rng.integers(1, 3)), int(rng.integers(1, min(3, cut - 3) + 1))
            r = int(rng.integers(1, h - ph))
            c = int(rng.integers(1, cut - pw))
            if not mask[r - 1:r + ph + 1, c - 1:c + pw + 1].all():
                continue
            mask[r:r + ph, c:c + pw] = False
            if ph == pw == 2 and rng.random() < .5:
                mask[r, c] = True
            break
    if spur and kind == "mixed":
        # A new leading aperture in the joined lobe coexists with the now
        # recessed seam-created pocket in the old body.
        mask[h - 3, w + spur - 2] = False
    if is_test or rng.random() < .65:
        # Exterior notch on the back, independent of cavity positions and keys.
        r = int(rng.integers(2, h - 2))
        trial = mask.copy()
        trial[r:r + int(rng.integers(1, 3)), :int(rng.integers(1, 3))] = False
        if len(_components(trial)) == 1:
            mask = trial
    if len(_components(mask)) != 1:
        return None
    labels = np.where(mask, np.where(np.indices(mask.shape)[1] < cut, 0, 1), -1).astype(int)
    if any(len(_components(labels == i)) != 1 for i in range(2)):
        return None
    return labels


def _partition(rng, labels, n, stepped):
    labels = labels.copy()
    while int(labels.max()) + 1 < n:
        options = []
        for i in range(int(labels.max()) + 1):
            q = np.argwhere(labels == i)
            for axis in range(2):
                for cut in range(int(q[:, axis].min()) + 2, int(q[:, axis].max())):
                    subset = (labels == i) & (np.indices(labels.shape)[axis] >= cut)
                    other = (labels == i) & ~subset
                    if min(int(subset.sum()), int(other.sum())) < 10:
                        continue
                    if len(_components(subset)) != 1 or len(_components(other)) != 1:
                        continue
                    trial = labels.copy()
                    trial[subset] = int(labels.max()) + 1
                    if _connected_port_graph(trial):
                        options.append(trial)
        if not options:
            return None
        labels = options[int(rng.integers(len(options)))]
    if stepped:
        # Complementary boundary tabs, with real occupied-mask collision logic.
        for _ in range(int(rng.integers(1, 5))):
            possibilities = []
            for r, c in np.argwhere(labels >= 0):
                r, c = int(r), int(c)
                i = int(labels[r, c])
                for dy, dx in _DIRS.values():
                    y, x = r + dy, c + dx
                    if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
                        j = int(labels[y, x])
                        if j >= 0 and j != i:
                            possibilities.append((r, c, i, j))
            rng.shuffle(possibilities)
            for r, c, i, j in possibilities[:40]:
                trial = labels.copy()
                trial[r, c] = j
                if min(np.count_nonzero(trial == i), np.count_nonzero(trial == j)) < 10:
                    continue
                if any(len(_components(trial == k)) != 1 for k in (i, j)):
                    continue
                if _connected_port_graph(trial):
                    labels = trial
                    break
    if rng.random() < .5:
        labels = labels[::-1].copy()
    return labels


def _add_keys(rng, labels, base, accents):
    n = int(labels.max()) + 1
    candidates = _port_candidates(labels)
    for _ in range(24):
        parent = list(range(n))
        selected, occupied = [], {}

        def root(i):
            while parent[i] != i:
                i = parent[i]
            return i

        # Favor short ports where several keys compete for a small fragment,
        # while still allowing longer signatures on roomy faces.
        ranked = sorted(range(len(candidates)),
                        key=lambda k: len(candidates[k][3]) + float(rng.uniform(0, 3)))
        for k in ranked:
            edge = candidates[int(k)]
            i, j, side, cells, other = edge
            if root(i) == root(j):
                continue
            # Only overlap and concatenation along the SAME exposed face are
            # forbidden. Nearby ports on different faces are unambiguous.
            if any(p in occupied for p in cells + other):
                continue
            along = (1, 0) if side == "R" else (0, 1)
            if any(occupied.get((r + sign * along[0], c + sign * along[1])) == face
                   for points, face in ((cells, side), (other, _OPPOSITE[side]))
                   for r, c in points for sign in (-1, 1)):
                continue
            parent[root(i)] = root(j)
            selected.append(edge)
            occupied.update((p, side) for p in cells)
            occupied.update((p, _OPPOSITE[side]) for p in other)
        if len(selected) != n - 1:
            continue
        pic = np.where(labels >= 0, base, -1).astype(np.int8)
        used = set()
        rng.shuffle(selected)
        for i, j, side, cells, other in selected:
            codes = [tuple(p) for p in product(accents, repeat=len(cells)) if tuple(p) not in used]
            if not codes:
                break
            code = codes[int(rng.integers(len(codes)))]
            used.add(code)
            for p, q, color in zip(cells, other, code):
                pic[p] = pic[q] = color
        else:
            return pic
    return None


def _pieces(labels, picture):
    pieces = []
    for i in range(int(labels.max()) + 1):
        q = np.argwhere(labels == i)
        r, c = map(int, q.min(0))
        y, x = map(int, q.max(0) + 1)
        m = np.where(labels[r:y, c:x] == i, picture[r:y, c:x], -1)
        pieces.append(dict(matrix=m.astype(np.int8), local=(r, c)))
    return pieces


def _marker(point, bottom):
    r, c = point
    dy = 1 if bottom else -1
    return [(r, c - 1), (r + dy, c - 1), (r + dy, c)]


def _scatter(rng, pieces, anchor, anchor_position, marker_cells, canvas, col):
    """Actual-mask spacing permits notches and overlapping empty bounding boxes."""
    h, _ = canvas
    for _ in range(28):
        blocked = np.zeros((h, col), dtype=bool)
        for r, c in marker_cells:
            blocked[max(0, r - 1):min(h, r + 2), max(0, c - 1):min(col, c + 2)] = True
        positions = [None] * len(pieces)
        order = [anchor] + sorted((i for i in range(len(pieces)) if i != anchor),
                                  key=lambda i: -np.count_nonzero(pieces[i]["matrix"] >= 0))
        failed = False
        for i in order:
            mask = pieces[i]["matrix"] >= 0
            ph, pw = mask.shape
            coords = np.argwhere(mask)
            if i == anchor:
                options = [anchor_position]
            else:
                options = [(r, c) for r in range(h - ph + 1) for c in range(col - pw + 1)]
                rng.shuffle(options)
            for r, c in options:
                if i != anchor and np.any(blocked[r:r + ph, c:c + pw][mask]):
                    continue
                positions[i] = (r, c)
                for y, x in coords:
                    y, x = int(y) + r, int(x) + c
                    blocked[max(0, y - 1):min(h, y + 2), max(0, x - 1):min(col, x + 2)] = True
                break
            else:
                failed = True
                break
        if not failed:
            return positions
    return None


def _training_evidence(info):
    mask = info["picture"] >= 0
    profile = info["profile"]
    individual_enclosed = set()
    for i in range(len(info["pieces"])):
        m = info["labels"] == i
        yy, xx = np.nonzero(m)
        r, c = int(yy.min()), int(xx.min())
        local = m[r:int(yy.max()) + 1, c:int(xx.max()) + 1]
        individual_enclosed.update((r + y, c + x) for q in _holes(local) for y, x in q)
    closed_by_join = {r for r in profile["rows"]
                      if (r, profile["extreme"] - 1) not in individual_enclosed}
    codes = list(info["ports"])
    ordered = (any(len(set(seq)) > 1 for axis, seq in codes)
               and any(set(a[1]) & set(b[1]) for i, a in enumerate(codes) for b in codes[i + 1:]))
    return dict(positive=bool(profile["rows"]), closure=bool(closed_by_join),
                recessed=bool(profile["decoy_rows"]),
                open=bool(profile["naive"] - profile["rows"]),
                partial=bool(profile["whole_hole_rows"] - profile["rows"]),
                motion=info["shift"] > 0, protrusion=bool(profile["veto_rows"]),
                ordered_keys=ordered, both_seam_axes={axis for axis, seq in codes} == {"h", "v"})


def _generate_pair(rng, kind, n, is_test, roles, turn, need_contrast=False):
    bg, rail, ink, base, *accents = roles
    for _ in range(100):
        labels = _make_mask(rng, kind, is_test)
        if labels is None:
            continue
        labels = _partition(rng, labels, n, stepped=is_test or rng.random() < .45)
        if labels is None:
            continue
        picture = _add_keys(rng, labels, base, accents)
        if picture is None:
            continue
        pieces = _pieces(labels, picture)
        bh, bw = labels.shape
        height = int(rng.integers(max(20, bh + 8), 31))
        width = int(rng.integers(max(23, bw + 5), 31))
        col = int(rng.integers(bw + 3, width - 1))
        final_left = col - bw
        final_top = int(rng.integers(1, height - bh))
        bottom = bool(rng.integers(2))
        corner = (bh - 1 if bottom else 0, 0)
        anchor = int(labels[corner])
        if anchor < 0:
            continue
        piece = pieces[anchor]
        pr, pc = piece["local"]
        # In generated scenes the marked corner is also a back corner of the
        # whole shutter. The solver only needs the fragment's transverse anchor.
        ph, pw = piece["matrix"].shape
        if corner != (pr + (ph - 1 if bottom else 0), pc):
            continue
        shift = int(rng.integers(1, final_left)) if need_contrast else int(rng.integers(final_left))
        anchor_position = (final_top + pr, final_left + pc - shift)
        point = (final_top + corner[0], anchor_position[1])
        markers = _marker(point, bottom)
        positions = _scatter(rng, pieces, anchor, anchor_position, markers, (height, width), col)
        if positions is None:
            continue
        inp = np.full((height, width), bg, dtype=np.int8)
        inp[:, col] = rail
        for piece, (r, c) in zip(pieces, positions):
            m = piece["matrix"]
            sub = inp[r:r + m.shape[0], c:c + m.shape[1]]
            sub[m >= 0] = m[m >= 0]
        for p in markers:
            inp[p] = ink
        # Labels are rendered from the constructed complete picture, separately
        # from the input graph traversal used by solve.
        out = np.full_like(inp, bg)
        out[:, col] = rail
        sub = out[final_top:final_top + bh, final_left:final_left + bw]
        sub[picture >= 0] = picture[picture >= 0]
        for r in _profile(picture >= 0)["rows"]:
            out[final_top + r, col + 1:] = ink
        inp, out = np.rot90(inp, turn).copy(), np.rot90(out, turn).copy()
        try:
            info = _extract(inp)
            evidence = _training_evidence(info)
            if need_contrast and not all(evidence[k] for k in _CONTRASTS + _A_CONTRASTS):
                continue
            if kind in ("veto", "mixed") and not evidence["protrusion"]:
                continue
            if not np.array_equal(_render(info), out):
                continue
        except AssertionError:
            continue
        return {"input": inp, "output": out}
    return _fallback_pair(rng, kind, n, roles, turn)


def _fallback_pair(rng, kind, n, roles, turn, placement=None):
    """A finite constructive layout under the identical rule, without retries.

    Roomy left panels join a C-shaped leading panel across a stepped pocket.
    Additional cuts give every count 2..6, with explicit nonconflicting ports.
    This supplies a total bounded construction if random partitions do not fit.
    """
    bg, rail, ink, base, *accents = roles
    spur = 2 if kind in ("veto", "mixed") else 0
    mask = np.zeros((16, 15 + spur), dtype=bool)
    mask[:, :15] = True
    mask[:2, 12:14] = False
    for r in range(4, 12):
        mask[r, 9:14 if r % 2 == 0 else 13] = False
    mask[3:5, 1:4] = False
    if spur:
        mask[10:, 15:] = True
        if kind == "mixed":
            mask[13, 15] = False
    yy, xx = np.indices(mask.shape)
    labels = np.where(mask, np.where(xx < 10, 0, 1), -1).astype(int)
    # Each extra cut preserves connectivity, topology, and a guaranteed port.
    seams = [(0, 1, [(1, 9), (2, 9)], [(1, 10), (2, 10)])]
    if n >= 3:
        labels[(labels == 0) & (yy >= 8)] = 2
        seams.append((0, 2, [(7, 1), (7, 2)], [(8, 1), (8, 2)]))
    if n >= 4:
        labels[(labels == 0) & (xx >= 5)] = 3
        seams[0] = (3, 1, seams[0][2], seams[0][3])
        seams.append((0, 3, [(1, 4), (2, 4)], [(1, 5), (2, 5)]))
    if n >= 5:
        # label 2 is the lower half: split it vertically, preserving its left
        # connection to label 0 and joining the new right half to label 3.
        labels[(labels == 2) & (xx >= 5)] = 4
        seams.append((3, 4, [(7, 6), (7, 7)], [(8, 6), (8, 7)]))
    if n >= 6:
        labels[(labels == 1) & (yy >= 14)] = 5
        seams.append((1, 5, [(13, 11), (13, 12)], [(14, 11), (14, 12)]))
    pic = np.where(mask, base, -1).astype(np.int8)
    codes = list(product(accents, repeat=2))
    rng.shuffle(codes)
    _require(len(codes) >= len(seams), "insufficient fallback signatures")
    if n >= 3:
        first, second = (accents[0], accents[1]), (accents[0], accents[0])
        codes = [first, second] + [code for code in codes if code not in (first, second)]
    for (_, _, cells, other), code in zip(seams, codes):
        for p, q, color in zip(cells, other, code):
            pic[p] = pic[q] = color
    pieces = _pieces(labels, pic)
    if placement is None:
        bottom = bool(rng.integers(2))
        final_top = int(rng.integers(1, 8))
        anchor_col = int(rng.integers(1, 5))
    else:
        bottom, final_top, anchor_col = placement
    anchor = int(labels[15 if bottom else 0, 0])
    positions = []
    for i, piece in enumerate(pieces):
        local_r, local_c = piece["local"]
        if i == anchor:
            positions.append((final_top + local_r, anchor_col))
        elif local_c == 0:
            positions.append((20 if local_r else 0, 2))
        elif local_c == 5:
            positions.append((20 if local_r else 0, 10))
        elif i == 5:
            positions.append((27, 18))
        else:
            positions.append((int(rng.integers(0, 11 if n == 6 else 15)), 18))
    inp = np.full((30, 30), bg, dtype=np.int8)
    inp[:, 28] = rail
    occupied = np.zeros_like(inp, dtype=bool)
    for piece, (r, c) in zip(pieces, positions):
        m = piece["matrix"]
        sub = inp[r:r + m.shape[0], c:c + m.shape[1]]
        used = occupied[r:r + m.shape[0], c:c + m.shape[1]]
        _require(not np.any(used[m >= 0]), "fallback scatter overlap")
        sub[m >= 0] = m[m >= 0]
        used[m >= 0] = True
    point = (final_top + (15 if bottom else 0), anchor_col)
    for p in _marker(point, bottom):
        _require(inp[p] == bg, "fallback L overlap")
        inp[p] = ink
    out = np.full_like(inp, bg)
    out[:, 28] = rail
    sub = out[final_top:final_top + 16, 28 - pic.shape[1]:28]
    sub[mask] = pic[mask]
    for r in _profile(mask)["rows"]:
        out[final_top + r, 29] = ink
    inp, out = np.rot90(inp, turn).copy(), np.rot90(out, turn).copy()
    _require(np.array_equal(solve(inp), out), "fallback recovery")
    if kind == "closure":
        ev = _training_evidence(_extract(inp))
        _require(all(ev[k] for k in _CONTRASTS), "fallback contrast coverage")
        if n >= 3:
            _require(all(ev[k] for k in _A_CONTRASTS), "fallback seam evidence")
    else:
        _require(_training_evidence(_extract(inp))["protrusion"], "fallback lobe contrast")
    return {"input": inp, "output": out}


def _fresh_fallback(rng, kind, n, roles, turn, seen):
    # At most thirteen inputs precede a pair. These fourteen candidates have
    # distinct exclusive-L coordinates, so at least one is necessarily fresh.
    placements = list(product((False, True), range(1, 8), range(1, 5)))[:14]
    for placement in placements:
        pair = _fallback_pair(rng, kind, n, roles, turn, placement)
        if (pair["input"].shape, pair["input"].tobytes()) not in seen:
            return pair
    raise AssertionError("fourteen distinct candidates cannot all be blocked by thirteen inputs")


def _counts(num_train, num_test):
    for value, name, lo, hi in ((num_train, "num_train", 2, 10), (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                or not lo <= int(value) <= hi):
            raise ValueError(f"{name} must be an integer in {lo}..{hi}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed) % (1 << 64))
    common_roles = tuple(map(int, rng.permutation(10)[:7]))
    varying_roles = bool(rng.integers(2))
    mixed_direction = bool(rng.integers(2))
    main_turn = int(rng.integers(4))
    puzzle, seen = {"train": [], "test": []}, set()
    plans = [("closure", 3, True), (str(rng.choice(("veto", "mixed"))), 2, False)]
    plans += [(str(rng.choice(("closure", "veto", "mixed"))), int(rng.integers(2, 4)), False)
              for _ in range(nt - 2)]
    rng.shuffle(plans)
    for split, count in (("train", nt), ("test", ne)):
        for i in range(count):
            kind, n, contrast = (plans[i] if split == "train" else
                                (str(rng.choice(("closure", "veto", "mixed"), p=(.45, .15, .40))),
                                 int(rng.integers(4, 7)), False))
            roles = tuple(map(int, rng.permutation(10)[:7])) if varying_roles else common_roles
            # Use two or three seam colors independently of the geometry, except
            # six-piece scenes need the larger finite pool of two-cell codes.
            if n < 6 and rng.random() < .5:
                roles = roles[:6]
            turn = int(rng.integers(4)) if mixed_direction else main_turn
            for _ in range(30):
                pair = _generate_pair(rng, kind, n, split == "test", roles, turn, contrast)
                key = (pair["input"].shape, pair["input"].tobytes())
                if key not in seen:
                    seen.add(key)
                    puzzle[split].append(pair)
                    break
            else:
                pair = _fresh_fallback(rng, kind, n, roles, turn, seen)
                seen.add((pair["input"].shape, pair["input"].tobytes()))
                puzzle[split].append(pair)
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "split lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, evidence, train_counts, test_counts = set(), [], [], []
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas must be preserved")
            key = (inp.shape, inp.tobytes())
            _require(key not in seen, "distinct inputs required")
            seen.add(key)
            info = _extract(inp)
            _require(np.array_equal(_render(info), out), "output violates input-derived rule")
            canonical_out = np.rot90(out, info["turn"])
            _require(np.all(canonical_out[:, info["col"]] == info["rail"]), "rail changed")
            _require(not np.any(canonical_out[:, :info["col"]] == info["ink"]), "L remains in work area")
            before = np.bincount(inp.ravel(), minlength=10)
            after = np.bincount(out.ravel(), minlength=10)
            for color in range(10):
                if color not in (info["bg"], info["ink"]):
                    _require(before[color] == after[color], "payload or rail color changed")
            expected_ink = len(info["profile"]["rows"]) * (canonical_out.shape[1] - info["col"] - 1)
            _require(after[info["ink"]] == expected_ink, "ray ink accounting")
            _require(before[info["ink"]] == 3, "exclusive L ink")
            _require(np.array_equal(solve(inp), out), "public solver disagreement")
            (train_counts if split == "train" else test_counts).append(len(info["pieces"]))
            if split == "train":
                evidence.append(_training_evidence(info))
    for name in _CONTRASTS + _A_CONTRASTS + ("protrusion",):
        _require(any(e[name] for e in evidence), "training lacks " + name + " contrast")
    _require(min(test_counts) > max(train_counts), "test must combine more fragments than training")


def _fingerprint(puzzle):
    return tuple((split, tuple((p[k].shape, p[k].tobytes())
                              for p in puzzle[split] for k in ("input", "output")))
                 for split in ("train", "test"))


def self_test():
    """Bounded 42-seed structural, semantic, equivariance and corruption audit."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _require(len({_fingerprint(p) for p in puzzles}) == 42, "seed diversity")
    shapes, geometries, palettes, directions, counts, shifts = set(), set(), set(), set(), set(), set()
    partial = silent = mixed = stepped = 0
    hole_counts, active_counts = set(), set()
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _extract(pair["input"])
                mask = info["picture"] >= 0
                shapes.add(pair["input"].shape)
                geometries.add((mask.shape, mask.tobytes(), info["labels"].tobytes()))
                palettes.add((info["bg"], info["rail"], info["ink"], info["base"]))
                directions.add(info["turn"])
                counts.add(len(info["pieces"]))
                shifts.add(info["shift"])
                partial += bool(info["profile"]["partial"])
                silent += not bool(info["profile"]["rows"])
                mixed += bool(info["profile"]["rows"] and info["profile"]["decoy_rows"])
                stepped += any(np.count_nonzero(p["matrix"] >= 0) < p["matrix"].size
                               for p in info["pieces"])
                hole_counts.add(len(info["profile"]["holes"]))
                active_counts.add(sum(any(c == info["profile"]["extreme"] - 1 for r, c in q)
                                      for q in info["profile"]["holes"]))
    _require(len(shapes) >= 25 and len(geometries) >= 100, "structural variation collapsed")
    _require(len(palettes) >= 30 and directions == set(range(4)), "role/direction variation")
    _require(counts == set(range(2, 7)) and len(shifts) >= 4, "count/motion variation")
    _require(partial >= 42 and silent >= 5 and mixed >= 42 and stepped >= 100,
             "family-specific variation collapsed")
    _require(max(hole_counts) >= 4 and max(active_counts) >= 2, "aperture counts collapsed")
    _require(_fingerprint(generate_puzzle(907, 7, 3)) == _fingerprint(generate_puzzle(907, 7, 3)),
             "determinism")
    for n in range(2, 7):
        roles = (0, 1, 2, 3, 4, 5, 6) if n == 6 else (0, 1, 2, 3, 4, 5)
        for kind in ("closure", "veto", "mixed"):
            for turn in range(4):
                q = _fallback_pair(np.random.default_rng(170 + n), kind, n, roles, turn)
                _require(np.array_equal(solve(q["input"]), q["output"]), "constructive fallback")
    # Force the duplicate path by blocking its first thirteen candidates.
    rng = np.random.default_rng(888)
    state = rng.bit_generator.state
    blocked = set()
    placements = list(product((False, True), range(1, 8), range(1, 5)))[:14]
    for placement in placements[:13]:
        q = _fallback_pair(rng, "closure", 3, (0, 1, 2, 3, 4, 5), 0, placement)
        blocked.add((q["input"].shape, q["input"].tobytes()))
    _require(len(blocked) == 13, "fallback coordinate orbit collapsed")
    rng.bit_generator.state = state
    fresh = _fresh_fallback(rng, "closure", 3, (0, 1, 2, 3, 4, 5), 0, blocked)
    _require((fresh["input"].shape, fresh["input"].tobytes()) not in blocked, "freshness guarantee")
    _require(_extract(fresh["input"])["point"] == (4, 2), "thirteenth blocked candidate was not exercised")
    # A fifth key closes a consistent cycle in the five-fragment fallback.
    # Add it to the input via recovered local coordinates, not source positions.
    cycle = _fallback_pair(np.random.default_rng(12), "closure", 5, (0, 1, 2, 3, 4, 5), 0)
    info = _extract(cycle["input"])
    for cells in ([(9, 4), (10, 4), (11, 4), (12, 4)],
                  [(9, 5), (10, 5), (11, 5), (12, 5)]):
        for (r, c), color in zip(cells, (4, 5, 4, 5)):
            i = int(info["labels"][r, c])
            ar, ac = info["positions"][i]
            old_r, old_c = info["pieces"][i]["old"]
            cycle["input"][old_r + info["top"] + r - ar,
                           old_c + info["left"] + c - ac] = color
            cycle["output"][info["top"] + r, info["left"] + info["shift"] + c] = color
    _require(len(_extract(cycle["input"])["ports"]) == 5, "redundant cycle not encoded")
    _require(np.array_equal(solve(cycle["input"]), cycle["output"]), "consistent-cycle support")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(713 + 10 * a + b, a, b)
            _require((len(p["train"]), len(p["test"])) == (a, b), "explicit counts ignored")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                 (True, 1), (4, False), (3.5, 1), (4, 1.5), (None, 1), (4, None)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            continue
        raise AssertionError("illegal counts accepted")
    pair = puzzles[0]["train"][0]
    for turn in range(4):
        _require(np.array_equal(solve(np.rot90(pair["input"], turn)),
                                np.rot90(pair["output"], turn)), "rotation equivariance")
    # Recolor all roles, including swapping background with the ink role.
    lookup = np.arange(9, -1, -1, dtype=np.int8)
    _require(np.array_equal(solve(lookup[pair["input"]]), lookup[pair["output"]]), "color equivariance")
    for damage in ("ray", "payload", "rail", "duplicate"):
        p = {s: [{k: g.copy() for k, g in q.items()} for q in puzzles[0][s]]
             for s in ("train", "test")}
        if damage == "duplicate":
            p["train"][1] = {k: g.copy() for k, g in p["train"][0].items()}
        else:
            q = p["test"][0]
            info = _extract(q["input"])
            a = np.rot90(q["output"], info["turn"])
            if damage == "rail":
                a[0, info["col"]] = info["bg"]
            elif damage == "ray":
                a[0, info["col"] + 1] = info["ink"]
            else:
                r, c = np.argwhere((a != info["bg"]) & (a != info["rail"]) & (a != info["ink"]))[0]
                a[r, c] = info["bg"]
        try:
            validate_generated_puzzle(p)
        except (AssertionError, ValueError):
            continue
        raise AssertionError("validator accepted corruption: " + damage)


if __name__ == "__main__":
    self_test()
    print("self_test passed (42 seeds, all legal counts)")
