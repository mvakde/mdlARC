"""Anchored reconstruction and repair of a torn concentric rectangular mural.

Only surviving paint and material geometry enter inference. Gray (5) is lost
paint, never background. The palette at each frame depth is inferred jointly
with an exhaustive translation-only tiling. No parent code is imported.
"""
from __future__ import annotations

import copy
import numbers
import numpy as np

GRAY = 5
CORNERS = ("TL", "TR", "BL", "BR")


def _arc_grid(value):
    g = np.asarray(value)
    if (g.ndim != 2 or not all(1 <= d <= 30 for d in g.shape)
            or not np.issubdtype(g.dtype, np.integer)
            or np.any((g < 0) | (g > 9))):
        raise ValueError("expected an integer ARC grid with sides 1..30")
    return g.astype(np.int8, copy=True)


def _components(mask):
    unseen = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while unseen:
        first = min(unseen)
        unseen.remove(first)
        queue = [first]
        for y, x in queue:
            for p in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if p in unseen:
                    unseen.remove(p)
                    queue.append(p)
        result.append(queue)
    return result


def _markers(point, corner):
    y, x = point
    dy = -1 if corner[0] == "T" else 1
    dx = -1 if corner[1] == "L" else 1
    return ((y+dy, x), (y, x+dx), (y+dy, x+dx))


def _extract(g):
    counts = np.bincount(g.ravel(), minlength=10)
    bg = int(np.argmax(counts))
    if bg == GRAY or np.count_nonzero(counts == counts[bg]) != 1:
        raise ValueError("background must be the unique most frequent color")
    markers = []
    for color in range(10):
        if color in (bg, GRAY) or counts[color] != 3:
            continue
        q = np.argwhere(g == color)
        lo, hi = q.min(0), q.max(0)
        if not np.array_equal(hi-lo, (1, 1)):
            continue
        square = {(int(y), int(x)) for y in range(lo[0], hi[0]+1)
                  for x in range(lo[1], hi[1]+1)}
        point = (square - set(map(tuple, q.tolist()))).pop()
        if g[point] in (bg, color):
            continue
        corner = {(1, 1): "TL", (1, 0): "TR", (0, 1): "BL",
                  (0, 0): "BR"}[(point[0]-lo[0], point[1]-lo[1])]
        markers.append((color, point, corner))
    if len(markers) != 1:
        raise ValueError("expected one external L marker")
    marker, point, corner = markers[0]
    pieces, anchor = [], None
    for cells in _components((g != bg) & (g != marker)):
        q = np.asarray(cells)
        top, left = q.min(0)
        bottom, right = q.max(0)+1
        m = np.full((bottom-top, right-left), -1, dtype=np.int8)
        for y, x in cells:
            m[y-top, x-left] = g[y, x]
        if point in cells:
            anchor = len(pieces)
        pieces.append((m, (int(top), int(left))))
    if not 3 <= len(pieces) <= 8 or anchor is None:
        raise ValueError("expected 3..8 fragments including the anchor")
    m, (y, x) = pieces[anchor]
    expected = (y + (m.shape[0]-1 if corner[0] == "B" else 0),
                x + (m.shape[1]-1 if corner[1] == "R" else 0))
    if point != expected:
        raise ValueError("marker must touch a bounding corner of its fragment")
    return bg, marker, point, corner, pieces, anchor


def _depths(h, w):
    row = np.minimum(np.arange(h), np.arange(h)[::-1])
    col = np.minimum(np.arange(w), np.arange(w)[::-1])
    return np.minimum(row[:, None], col[None, :])


def _search(pieces, anchor, point, corner, canvas, *, layers=True,
            paint_reference=None):
    """Complete search over area factors and unrotated occupied-mask tilings.

    With layers=False, stop at a geometric tiling with different dimensions,
    anchoring bounds, or surviving paint from the reference mural. Merely
    permuting gray losses is not evidence that color constrains placement.
    With layers=True, different piece assignments are permitted only when they
    give exactly the same restored output. An unobserved depth is ambiguous.
    """
    area = sum(int(np.count_nonzero(m >= 0)) for m, _ in pieces)
    data = []
    for m, _ in pieces:
        coords = [(int(y), int(x)) for y, x in np.argwhere(m >= 0)]
        paint = [(y, x, int(m[y, x])) for y, x in coords if m[y, x] != GRAY]
        data.append((m.shape, coords, paint))
    winner = None
    winner_key = None
    nodes = 0
    valid_assignments = 0
    for h in range(6, min(18, canvas[0])+1):
        if area % h:
            continue
        w = area // h
        if not 6 <= w <= min(18, canvas[1]):
            continue
        oy = point[0] - (h-1 if corner[0] == "B" else 0)
        ox = point[1] - (w-1 if corner[1] == "R" else 0)
        if oy < 0 or ox < 0 or oy+h > canvas[0] or ox+w > canvas[1]:
            continue
        if any(ph > h or pw > w for (ph, pw), _, _ in data):
            continue
        dep = _depths(h, w)
        k = (min(h, w)+1)//2
        full = (1 << area)-1
        bits = [sum(1 << (y*w+x) for y, x in cells) for _, cells, _ in data]
        placements = {}

        def at(i, r, c):
            key = (i, r, c)
            if key in placements:
                return placements[key]
            (ph, pw), _, paint = data[i]
            if r < 0 or c < 0 or r+ph > h or c+pw > w:
                placements[key] = None
                return None
            clues = {}
            if layers:
                for y, x, color in paint:
                    d = int(dep[r+y, c+x])
                    if d in clues and clues[d] != color:
                        placements[key] = None
                        return None
                    clues[d] = color
            placed = bits[i] << (r*w+c)
            placements[key] = (placed, tuple(clues.items()))
            return placements[key]

        r = pieces[anchor][1][0]-oy
        c = pieces[anchor][1][1]-ox
        start = at(anchor, r, c)
        if start is None:
            continue
        code = [-1]*k
        for d, color in start[1]:
            code[d] = color
        positions = [None]*len(pieces)
        positions[anchor] = (r, c)

        def visit(occupied, remaining, known):
            nonlocal winner, winner_key, nodes, valid_assignments
            nodes += 1
            if not remaining:
                valid_assignments += 1
                if layers and -1 in known:
                    raise ValueError("a feasible assembly has an unwitnessed layer")
                raw = np.full((h, w), -1, dtype=np.int8)
                labels = np.full((h, w), -1, dtype=np.int8)
                absolute = []
                for i, (y, x) in enumerate(positions):
                    m = pieces[i][0]
                    sub = raw[y:y+m.shape[0], x:x+m.shape[1]]
                    sub[m >= 0] = m[m >= 0]
                    sub = labels[y:y+m.shape[0], x:x+m.shape[1]]
                    sub[m >= 0] = i
                    absolute.append((y+oy, x+ox))
                picture = np.asarray(known, dtype=np.int8)[dep] if layers else raw
                key = ((h, w), (oy, ox), picture.tobytes())
                candidate = dict(shape=(h, w), origin=(oy, ox), code=tuple(known),
                                 positions=tuple(absolute), raw=raw, labels=labels,
                                 picture=picture, nodes=0, assignments=0)
                if not layers:
                    ref_shape, ref_origin, ref_picture = paint_reference
                    if ((h, w) != ref_shape or (oy, ox) != ref_origin
                            or np.any((raw != GRAY) & (raw != ref_picture))):
                        winner = candidate
                        return True
                elif winner_key is None:
                    winner_key, winner = key, candidate
                elif winner_key != key:
                    raise ValueError("distinct restored murals satisfy the input")
                return False
            free = full ^ occupied
            first = (free & -free).bit_length()-1
            y, x = divmod(first, w)
            for i in remaining:
                # Earlier cells are occupied, so the lexicographically first
                # occupied cell of a new fragment must be the first free cell.
                c = x-data[i][1][0][1]
                placement = at(i, y, c)
                if placement is None or occupied & placement[0]:
                    continue
                next_code = list(known)
                if any(next_code[d] not in (-1, color) for d, color in placement[1]):
                    continue
                for d, color in placement[1]:
                    next_code[d] = color
                positions[i] = (y, c)
                if visit(occupied | placement[0], tuple(j for j in remaining if j != i), next_code):
                    return True
            return False

        if visit(start[0], tuple(i for i in range(len(pieces)) if i != anchor), code):
            break
    if winner is None and layers:
        raise ValueError("no consistent anchored mural")
    if winner is not None:
        winner["nodes"] = nodes
        winner["assignments"] = valid_assignments
    return winner


def _infer(input_grid):
    g = _arc_grid(input_grid)
    bg, marker, point, corner, pieces, anchor = _extract(g)
    info = _search(pieces, anchor, point, corner, g.shape)
    out = np.full_like(g, bg)
    y, x = info["origin"]
    h, w = info["shape"]
    out[y:y+h, x:x+w] = info["picture"]
    info.update(background=bg, marker=marker, point=point, corner=corner,
                pieces=pieces, anchor=anchor)
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Infer from this input alone; do not retain inputs or generation state."""
    return _infer(input_grid)[0]


def _split_lengths(rng, length, n, minimum=2):
    cuts = sorted(int(x) for x in rng.choice(length-n*minimum+n-1, n-1, replace=False))
    lengths = np.diff([-1]+cuts+[length-n*minimum+n-1])+minimum-1
    return [int(x) for x in lengths]


def _partition(rng, h, w, n, mode, steps):
    if mode == "quarters":
        labels = np.zeros((h, w), dtype=np.int8)
        labels[:h//2, w//2:] = 1
        labels[h//2:, :w//2] = 2
        labels[h//2:, w//2:] = 3
    elif mode == "bands":
        # Narrow outer bands omit inner-depth evidence at every corner.
        left, right = int(rng.integers(2, 4)), int(rng.integers(2, 4))
        labels = np.zeros((h, w), dtype=np.int8)
        labels[:, w-right:] = n-1
        y = 0
        for i, length in enumerate(_split_lengths(rng, h, n-2)):
            labels[y:y+length, left:w-right] = i+1
            y += length
    else:
        rects = [(0, 0, h, w)]
        for _ in range(n-1):
            choices = [(i, axis) for i, (_, _, ph, pw) in enumerate(rects)
                       for axis, length in enumerate((ph, pw)) if length >= 4]
            if not choices:
                return None
            weights = np.asarray([rects[i][2]*rects[i][3] for i, _ in choices], dtype=float)
            i, axis = choices[int(rng.choice(len(choices), p=weights/weights.sum()))]
            y, x, ph, pw = rects.pop(i)
            cut = int(rng.integers(2, (ph if axis == 0 else pw)-1))
            if axis == 0:
                rects.extend(((y, x, cut, pw), (y+cut, x, ph-cut, pw)))
            else:
                rects.extend(((y, x, ph, cut), (y, x+cut, ph, pw-cut)))
        labels = np.full((h, w), -1, dtype=np.int8)
        for i, (y, x, ph, pw) in enumerate(rects):
            labels[y:y+ph, x:x+pw] = i
    if steps:
        options = []
        # Transfer cells only inside the outer boundary, preserving all corners.
        for y in range(1, h-1):
            for x in range(1, w-1):
                for yy, xx in ((y+1, x), (y, x+1)):
                    if labels[y, x] != labels[yy, xx]:
                        options.append((y, x, yy, xx))
        rng.shuffle(options)
        changed = 0
        for y, x, yy, xx in options:
            before = labels.copy()
            donor, receiver = int(labels[y, x]), int(labels[yy, xx])
            if donor == receiver:
                continue
            labels[y, x] = receiver
            if (len(_components(labels == donor)) != 1
                    or len(_components(labels == receiver)) != 1
                    or np.count_nonzero(labels == donor) < 4):
                labels = before
                continue
            changed += 1
            if changed >= int(rng.integers(1, 4)):
                break
        irregular = any(len(q) < int(np.prod(np.ptp(q, axis=0)+1))
                        for i in range(n) if len(q := np.argwhere(labels == i)))
        if not irregular:
            return None
    if mode != "quarters":
        if rng.integers(2):
            labels = labels.T.copy()
        if rng.integers(2):
            labels = labels[::-1].copy()
        if rng.integers(2):
            labels = labels[:, ::-1].copy()
    return labels


def _layer_code(rng, k, available, style):
    palette = [int(x) for x in rng.permutation(available)]
    if style == "distinct":
        return np.asarray(palette[:k], dtype=np.int8)
    if style == "repeat":
        code = [palette[0], palette[0], palette[1], palette[2], palette[0]]
        code += [palette[int(rng.integers(min(5, k)))]]*(k-5)
        return np.asarray(code, dtype=np.int8)
    style = int(rng.integers(4))
    if style == 0:
        return np.asarray([palette[i % len(palette)] for i in range(k)], dtype=np.int8)
    size = int(rng.integers(2, min(k, len(palette))+1))
    code = [palette[int(rng.integers(size))] for _ in range(k)]
    code[0], code[-1] = palette[0], palette[1]
    if style == 1 and k >= 4:
        code[1] = code[0]
        code[2] = palette[1]
        code[-1] = code[0]
    elif style == 2:
        cut = int(rng.integers(1, k))
        code = [palette[0]]*cut+[palette[1]]*(k-cut)
    return np.asarray(code, dtype=np.int8)


def _damage(rng, picture, labels, fraction):
    h, w = picture.shape
    masked = picture.copy()
    target = max(4, int(h*w*fraction))
    # Paint losses form small patches and strokes, some crossing cut seams.
    for _ in range(h*w):
        if np.count_nonzero(masked == GRAY) >= target:
            break
        y, x = int(rng.integers(h)), int(rng.integers(w))
        ph, pw = int(rng.integers(1, 4)), int(rng.integers(1, 4))
        masked[y:y+ph, x:x+pw] = GRAY
    dep = _depths(h, w)
    # Every true depth and every piece retains evidence; no unknown color is
    # supplied through generation metadata to the solver.
    for d in range(int(dep.max())+1):
        q = np.argwhere(dep == d)
        keep = q[rng.choice(len(q), min(2, len(q)), replace=False)]
        masked[keep[:, 0], keep[:, 1]] = picture[keep[:, 0], keep[:, 1]]
    for i in range(int(labels.max())+1):
        q = np.argwhere(labels == i)
        keep = q[rng.choice(len(q), min(3, len(q)), replace=False)]
        masked[keep[:, 0], keep[:, 1]] = picture[keep[:, 0], keep[:, 1]]
    return masked


def _pieces(labels, masked):
    result = []
    for i in range(int(labels.max())+1):
        q = np.argwhere(labels == i)
        y, x = q.min(0)
        bottom, right = q.max(0)+1
        m = np.where(labels[y:bottom, x:right] == i, masked[y:bottom, x:right], -1)
        result.append((m.astype(np.int8), (int(y), int(x))))
    return result


def _scatter(rng, pieces, anchor, final, marker_cells, canvas):
    remaining = sorted((i for i in range(len(pieces)) if i != anchor),
                       key=lambda i: -pieces[i][0].size)
    for _ in range(20):
        positions = {anchor: final[anchor]}
        rects = [(*final[anchor], *pieces[anchor][0].shape)]
        for i in remaining:
            h, w = pieces[i][0].shape
            choices = [(r, c) for r in range(canvas[0]-h+1)
                       for c in range(canvas[1]-w+1)]
            found = False
            for j in rng.permutation(len(choices)):
                r, c = choices[int(j)]
                if any(r-1 <= y <= r+h and c-1 <= x <= c+w for y, x in marker_cells):
                    continue
                if all(r+h < y or y+ph < r or c+w < x or x+pw < c
                       for y, x, ph, pw in rects):
                    positions[i] = (r, c)
                    rects.append((r, c, h, w))
                    found = True
                    break
            if not found:
                break
        if len(positions) == len(pieces):
            return [positions[i] for i in range(len(pieces))]
    return None


def _geometry_alternative(info, canvas):
    reference = (info["shape"], info["origin"], info["picture"])
    return _search(info["pieces"], info["anchor"], info["point"], info["corner"],
                   canvas, layers=False, paint_reference=reference) is not None


def _make_pair(rng, role, corner, seen, train_shapes):
    for attempt in range(150):
        if role == "teaching_geometry":
            h, w = int(rng.choice((8, 10))), int(rng.choice((8, 10, 12)))
            n, mode, steps, style = 4, "quarters", False, "distinct"
            fraction = float(rng.uniform(.08, .18))
        elif role == "teaching_depth":
            h, w = int(rng.integers(10, 13)), int(rng.integers(10, 16))
            n, mode, steps, style = 5, "bands", True, "repeat"
            fraction = float(rng.uniform(.12, .24))
        else:
            test = role == "test"
            if test and train_shapes and rng.random() < .35:
                h, w = train_shapes[int(rng.integers(len(train_shapes)))]
            else:
                if rng.random() < .30:
                    h, w = int(rng.integers(6, 10)), int(rng.integers(13, 19))
                else:
                    h = int(rng.integers(7 if not test else 9, 15 if not test else 18))
                    w = int(rng.integers(7 if not test else 9, 17 if not test else 19))
            n = int(rng.integers(6, 9) if test else rng.integers(3, 6))
            mode = "bands" if rng.random() < .35 and h >= 2*(n-2) and w >= 8 else "recursive"
            steps, style = bool(rng.random() < .7), "free"
            fraction = float(rng.uniform(.14, .34) if test else rng.uniform(.08, .25))
        labels = _partition(rng, h, w, n, mode, steps)
        if labels is None:
            continue
        h, w = labels.shape
        if role == "teaching_depth" and (h, w) in train_shapes:
            continue
        colors = [int(x) for x in rng.permutation([c for c in range(10) if c != GRAY])]
        bg, marker = colors[:2]
        code = _layer_code(rng, (min(h, w)+1)//2, colors[2:], style)
        picture = code[_depths(h, w)]
        masked = _damage(rng, picture, labels, fraction)
        if np.count_nonzero(masked == GRAY) < 4:
            continue
        if role.startswith("teaching") and len(set(int(v) for v in picture[masked == GRAY])) < 2:
            continue
        pieces = _pieces(labels, masked)
        local = (h-1 if corner[0] == "B" else 0, w-1 if corner[1] == "R" else 0)
        anchor = int(labels[local])
        base = max(18, int(np.ceil(np.sqrt(2.45*h*w))))
        canvas = (min(30, max(h+3, base)+int(rng.integers(1, 5))),
                  min(30, max(w+3, base)+int(rng.integers(1, 5))))
        if canvas[0]*canvas[1] <= 2*h*w+6:
            continue
        oy = int(rng.integers(1, canvas[0]-h))
        ox = int(rng.integers(1, canvas[1]-w))
        point = (oy+local[0], ox+local[1])
        marker_cells = _markers(point, corner)
        final = [(oy+y, ox+x) for _, (y, x) in pieces]
        positions = _scatter(rng, pieces, anchor, final, marker_cells, canvas)
        if positions is None:
            continue
        g = np.full(canvas, bg, dtype=np.int8)
        for (m, _), (y, x) in zip(pieces, positions):
            sub = g[y:y+m.shape[0], x:x+m.shape[1]]
            sub[m >= 0] = m[m >= 0]
        for p in marker_cells:
            g[p] = marker
        key = (g.shape, g.tobytes())
        if key in seen:
            continue
        try:
            out, info = _infer(g)
        except ValueError:
            continue
        expected = np.full(canvas, bg, dtype=np.int8)
        expected[oy:oy+h, ox:ox+w] = picture
        if not np.array_equal(out, expected):
            continue
        if role == "teaching_geometry" and not _geometry_alternative(info, g.shape):
            continue
        if role == "teaching_depth":
            am, (ay, ax) = info["pieces"][info["anchor"]]
            fy, fx = info["positions"][info["anchor"]]
            dy, dx = fy-oy, fx-ox
            depths = _depths(h, w)[dy:dy+am.shape[0], dx:dx+am.shape[1]]
            witnessed = set(int(d) for d in depths[(am >= 0) & (am != GRAY)])
            if len(witnessed) == len(code):
                continue
            if not set(int(v) for v in code)-set(int(v) for v in am[(am >= 0) & (am != GRAY)]):
                continue
        seen.add(key)
        return {"input": g, "output": out}
    raise RuntimeError(f"could not construct a unique {role} mural after 150 attempts")


def _counts(num_train, num_test):
    for value, name, low, high in ((num_train, "num_train", 2, 10),
                                   (num_test, "num_test", 1, 4)):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral)
                or not low <= int(value) <= high):
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must be greater than num_test")
    return int(num_train), int(num_test)


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, ne = _counts(num_train, num_test)
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, numbers.Integral):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    first = str(rng.choice(CORNERS))
    opposite = {"TL": "BR", "TR": "BL", "BL": "TR", "BR": "TL"}[first]
    order = [first, opposite]+[str(x) for x in rng.permutation(
        [c for c in CORNERS if c not in (first, opposite)])]
    seen, train_shapes = set(), []
    puzzle = {"train": [], "test": []}
    for i in range(nt):
        role = ("teaching_geometry", "teaching_depth")[i] if i < 2 else "train"
        pair = _make_pair(rng, role, order[i % 2], seen, train_shapes)
        puzzle["train"].append(pair)
        train_shapes.append(_infer(pair["input"])[1]["shape"])
    rng.shuffle(puzzle["train"])
    # More pieces is always a test shift; corner holdout is an optional second
    # axis. Both marker directions have already been demonstrated geometrically.
    test_corners = order[2:] if rng.random() < .5 else order
    for i in range(ne):
        puzzle["test"].append(_make_pair(rng, "test", test_corners[i % len(test_corners)],
                                          seen, train_shapes))
    validate_generated_puzzle(puzzle)
    return puzzle


def _pair_info(pair):
    g, out = _arc_grid(pair["input"]), _arc_grid(pair["output"])
    predicted, info = _infer(g)
    assert np.array_equal(out, predicted), "output differs from input-derived reconstruction"
    assert g.shape == out.shape
    bg, marker = info["background"], info["marker"]
    h, w = info["shape"]
    oy, ox = info["origin"]
    crop = out[oy:oy+h, ox:ox+w]
    # Validate layer uniformity by inward border peeling, separately from the
    # min-distance renderer used by the solver.
    for d, color in enumerate(info["code"]):
        inner = crop[d:h-d, d:w-d]
        assert np.all(inner[0] == color) and np.all(inner[-1] == color)
        assert np.all(inner[:, 0] == color) and np.all(inner[:, -1] == color)
    assert marker not in out and GRAY not in out
    assert all(out[p] == bg for p in _markers(info["point"], info["corner"]))
    assert np.count_nonzero(g == GRAY) >= 4
    assert len(set(info["code"])) >= 2
    assert np.count_nonzero(out != bg) == sum(np.count_nonzero(m >= 0) for m, _ in info["pieces"])
    assert np.count_nonzero(out == bg) == np.count_nonzero(g == bg)+3
    repair = info["raw"] == GRAY
    repaired_colors = set(int(v) for v in crop[repair])
    for color in range(10):
        if color not in (bg, marker, GRAY):
            assert np.count_nonzero(out == color) == (np.count_nonzero(g == color)
                   + np.count_nonzero(crop[repair] == color)), "paint histogram not conserved"
    assert info["positions"][info["anchor"]] == info["pieces"][info["anchor"]][1]
    coverage = np.zeros_like(out, dtype=bool)
    moved, stepped, anchor_depths = 0, 0, set()
    dep = _depths(h, w)
    for i, ((m, old), (y, x)) in enumerate(zip(info["pieces"], info["positions"])):
        mask = m >= 0
        paint = mask & (m != GRAY)
        region = out[y:y+m.shape[0], x:x+m.shape[1]]
        assert np.array_equal(region[paint], m[paint]), "surviving paint changed"
        assert not coverage[y:y+m.shape[0], x:x+m.shape[1]][mask].any()
        coverage[y:y+m.shape[0], x:x+m.shape[1]][mask] = True
        assert len(_components(mask)) == 1 and np.count_nonzero(paint) >= 1
        moved += old != (y, x)
        stepped += np.count_nonzero(mask) != m.size
        if i == info["anchor"]:
            d = dep[y-oy:y-oy+m.shape[0], x-ox:x-ox+m.shape[1]]
            anchor_depths = set(int(v) for v in d[paint])
    assert np.array_equal(coverage, out != bg), "fragments do not tile the mural"
    assert moved >= 1, "example does not demonstrate translation"
    info.update(moved=moved, stepped=stepped, anchor_depths=anchor_depths,
                repaired_colors=repaired_colors)
    return info


def validate_generated_puzzle(puzzle) -> None:
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert isinstance(puzzle["train"], list) and isinstance(puzzle["test"], list)
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen = set()
    infos = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            g = _arc_grid(pair["input"])
            key = (g.shape, g.tobytes())
            assert key not in seen, "duplicate input"
            seen.add(key)
            infos[split].append(_pair_info(pair))
    train = infos["train"]
    assert any(info["stepped"] for info in train), "training omits complementary notches"
    assert any(len(info["anchor_depths"]) < len(info["code"]) for info in train)
    assert any(set(info["code"])-set(int(v) for v in info["pieces"][info["anchor"]][0].flat
                                    if v not in (-1, GRAY)) for info in train)
    assert any(len(set(info["code"])) == len(info["code"]) for info in train)
    assert any(any(a == b for a, b in zip(info["code"], info["code"][1:]))
               and len(set(info["code"])) < len(info["code"])-1 for info in train)
    runs = [tuple(c for i, c in enumerate(info["code"])
                  if i == 0 or c != info["code"][i-1]) for info in train]
    assert any(len(seq) > len(set(seq)) for seq in runs), "training omits a color return"
    assert len({info["corner"] for info in train}) >= 2
    assert len({info["shape"] for info in train}) >= 2
    assert sum(len(info["repaired_colors"]) >= 2 for info in train) >= 2
    assert any(_geometry_alternative(info, np.asarray(pair["input"]).shape)
               for info, pair in zip(train, puzzle["train"]))
    assert all(len(info["pieces"]) > max(len(t["pieces"]) for t in train)
               for info in infos["test"])


def _signature(puzzle):
    return tuple((split, tuple((np.asarray(p["input"]).shape, np.asarray(p["input"]).tobytes(),
                               np.asarray(p["output"]).tobytes()) for p in puzzle[split]))
                 for split in ("train", "test"))


def self_test() -> dict:
    """Bounded authored checks over 42 seeds; no claim of independent judging."""
    shapes, canvases, partitions, codes, corners, counts, palettes = [set() for _ in range(7)]
    signatures = set()
    total, stepped, repeated, odd, square, gray_total = 0, 0, 0, 0, 0, 0
    sample = None
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        signatures.add(_signature(puzzle))
        if seed == 0:
            sample = puzzle
        for split in ("train", "test"):
            for pair in puzzle[split]:
                info = _pair_info(pair)
                total += 1
                shapes.add(info["shape"])
                canvases.add(pair["input"].shape)
                # Normalize recovered fragment labels, excluding literal colors.
                rename = {}
                normalized = tuple(rename.setdefault(int(v), len(rename)) for v in info["labels"].flat)
                partitions.add((info["shape"], normalized))
                rename = {}
                codes.add(tuple(rename.setdefault(c, len(rename)) for c in info["code"]))
                corners.add(info["corner"])
                counts.add(len(info["pieces"]))
                palettes.add(len(set(info["code"])))
                stepped += info["stepped"] > 0
                repeated += len(set(info["code"])) < len(info["code"])
                odd += min(info["shape"]) % 2 == 1
                square += info["shape"][0] == info["shape"][1]
                gray_total += int(np.count_nonzero(pair["input"] == GRAY))
                original = pair["input"].copy()
                assert np.array_equal(solve(original), pair["output"])
                assert np.array_equal(original, pair["input"]), "solve mutated input"
    assert len(signatures) == 42
    assert len(shapes) >= 35 and len(canvases) >= 20
    assert len(partitions) >= 120 and len(codes) >= 20
    assert corners == set(CORNERS) and counts == set(range(3, 9))
    assert {2, 3, 4, 5} <= palettes
    assert stepped >= 80 and repeated >= 70 and odd >= 25 and square >= 10
    assert _signature(generate_puzzle(81273, 7, 3)) == _signature(generate_puzzle(81273, 7, 3))
    # Exercise all explicit count combinations on a different seed from the
    # host's count sweep; validation includes the two-example evidence boundary.
    for nt in range(2, 11):
        for ne in range(1, min(4, nt-1)+1):
            p = generate_puzzle(1000+nt*10+ne, nt, ne)
            assert len(p["train"]) == nt and len(p["test"]) == ne
    for nt, ne in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4),
                   (True, 1), (4.0, 1), (4, False), (None, 1)):
        try:
            generate_puzzle(0, nt, ne)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    # Output corruption and duplicate inputs must not pass semantic validation.
    for kind in ("paint", "marker", "anchor", "duplicate"):
        bad = copy.deepcopy(sample)
        pair = bad["test"][0]
        info = _pair_info(pair)
        if kind == "paint":
            y, x = info["point"]
            pair["output"][y, x] = GRAY
        elif kind == "marker":
            pair["output"][_markers(info["point"], info["corner"])[0]] = info["marker"]
        elif kind == "anchor":
            pair["output"] = np.roll(pair["output"], 1, axis=0)
        else:
            bad["train"][1] = copy.deepcopy(bad["train"][0])
        try:
            validate_generated_puzzle(bad)
        except (AssertionError, ValueError):
            pass
        else:
            raise AssertionError("validator accepted corruption: "+kind)
    # Reordering extracted objects must not change the chosen restored mural.
    info = _pair_info(sample["test"][0])
    order = list(reversed(range(len(info["pieces"]))))
    other = _search([info["pieces"][i] for i in order], order.index(info["anchor"]),
                    info["point"], info["corner"], sample["test"][0]["input"].shape)
    assert np.array_equal(other["picture"], info["picture"])
    # Sparse 4x4 tiles permit two completely witnessed frame codes, (1,2,3,4)
    # and (1,3,2,4). A solver must reject rather than choose traversal order.
    tiles = [np.full((4, 4), GRAY, dtype=np.int8) for _ in range(4)]
    tiles[0][0, 0], tiles[0][3, 3] = 1, 4
    tiles[1][1, 2], tiles[2][2, 1], tiles[3][0, 0] = 2, 3, 1
    try:
        _search([(m, (1, 1)) for m in tiles], 0, (1, 1), "TL", (12, 12))
    except ValueError as exc:
        assert "distinct restored murals" in str(exc)
    else:
        raise AssertionError("ambiguous frame codes were accepted")
    tiles = [np.full((3, 3), GRAY, dtype=np.int8) for _ in range(4)]
    tiles[0][0, 0], tiles[1][2, 0] = 1, 2
    try:
        _search([(m, (1, 1)) for m in tiles], 0, (1, 1), "TL", (10, 10))
    except ValueError as exc:
        assert "unwitnessed layer" in str(exc)
    else:
        raise AssertionError("unobserved paint color was invented")
    return dict(seeds=42, pairs=int(total), mural_shapes=len(shapes), canvas_shapes=len(canvases),
                normalized_partitions=len(partitions), normalized_codes=len(codes),
                stepped_pairs=int(stepped), repeated_depth_pairs=int(repeated), odd_center_pairs=int(odd),
                square_pairs=int(square), repaired_cells=int(gray_total),
                independent_solver=False)
