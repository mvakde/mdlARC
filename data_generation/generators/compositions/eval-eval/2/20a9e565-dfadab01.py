"""Topology under the eraser: arithmetic carriers gate symbolic print codes.

Only standard-library modules and NumPy are used. Labels are constructed from
explicit latent objects; solve independently fits geometry from visible strokes.
The fixed stamp vocabulary is inherited from B, but membership uses completed,
temporarily invisible carriers. See task_spec.md for the family domain.
"""
from __future__ import annotations

import copy
import numpy as np


BOOK = {
    2: (4, ("1111", "1001", "1001", "1111")),
    3: (1, ("0110", "1001", "1001", "0110")),
    5: (6, ("1100", "1100", "0011", "0011")),
    8: (7, ("1001", "0110", "0110", "1001")),
}
CODES = frozenset(BOOK)


def _grid(value):
    a = np.asarray(value)
    assert a.ndim == 2 and all(1 <= n <= 30 for n in a.shape)
    assert np.issubdtype(a.dtype, np.integer)
    assert np.all((0 <= a) & (a <= 9))
    return a.astype(np.int8, copy=False)


def _key(a):
    a = np.asarray(a)
    return a.shape, tuple(map(int, a.flat))


def _corners(bb):
    r0, r1, c0, c1 = bb
    return {(r0, c0), (r0, c0 + 1), (r0 + 1, c0),
            (r1, c1), (r1, c1 - 1), (r1 - 1, c1)}


def _annotations(a, bg):
    found = []
    for raw in np.unique(a):
        color = int(raw)
        if color == bg or color in CODES:
            continue
        p = np.argwhere(a == color)
        if len(p) != 6:
            continue
        r0, c0 = map(int, p.min(axis=0))
        r1, c1 = map(int, p.max(axis=0))
        if min(r1 - r0, c1 - c0) < 1:
            continue
        bb = (r0, r1, c0, c1)
        if set(map(tuple, p)) != _corners(bb):
            continue
        tl = [(r0, c0), (r0, c0 + 1), (r0 + 1, c0)]
        br = [(r1, c1), (r1, c1 - 1), (r1 - 1, c1)]
        if any(max(abs(r - s), abs(c - t)) <= 1
               for r, c in tl for s, t in br):
            continue
        found.append((color, bb))
    assert len(found) == 1, "expected two unambiguous disconnected corner Ls"
    return found[0]


def _mask(style, h, w, spacing):
    """Input-side predicates, with tight bounds and top-anchored tooth phase."""
    r, c = np.indices((h, w))
    if style == "frame":
        return (r == 0) | (r == h - 1) | (c == 0) | (c == w - 1)
    teeth = r % spacing == 0
    if style == "left_comb":
        return (c == 0) | teeth
    if style == "right_comb":
        return (c == w - 1) | teeth
    if style == "ladder":
        return (c == 0) | (c == w - 1) | teeth
    raise AssertionError("unknown carrier")


def _stroke_object(style, h, w, spacing):
    """Generation-side drawing by strokes, separate from the fit predicates."""
    a = np.zeros((h, w), dtype=bool)
    if style == "frame":
        a[0, :] = a[-1, :] = True
        a[:, 0] = a[:, -1] = True
    else:
        if style in ("left_comb", "ladder"):
            a[:, 0] = True
        if style in ("right_comb", "ladder"):
            a[:, -1] = True
        for r in range(0, h, spacing):
            a[r, :] = True
    return a


def _linear(v):
    d = v[1] - v[0]
    assert all(x == v[0] + i * d for i, x in enumerate(v)), "nonarithmetic evidence"
    return int(v[0]), int(d)


def _geom(m, i):
    h = m["h"][0] + i * m["h"][1]
    w = m["w"][0] + i * m["w"][1]
    y = m["y"][0] + i * m["y"][1]
    x = (m["x"] + i * (m["w"][0] + m["gap"][0])
         + i * (i - 1) // 2 * (m["w"][1] + m["gap"][1]))
    return y, x, h, w


def _fit(a, scaffold, bb):
    """Fit consecutive, fully visible objects, never reading the answer."""
    candidates = []
    for transposed in (False, True):
        z = a.T if transposed else a
        q = (bb[2], bb[3], bb[0], bb[1]) if transposed else bb
        cols = np.flatnonzero(np.any(z == scaffold, axis=0))
        groups = np.split(cols, np.flatnonzero(np.diff(cols) > 1) + 1)
        if not 3 <= len(groups) <= 6 or any(len(g) == 0 for g in groups):
            continue
        if not (cols[-1] < q[2] or cols[0] > q[3]):
            continue
        starts, tops, patches = [], [], []
        for g in groups:
            rows = np.flatnonzero(np.any(z[:, g] == scaffold, axis=1))
            starts.append(int(g[0]))
            tops.append(int(rows[0]))
            patches.append(z[rows[0]:rows[-1] + 1, g[0]:g[-1] + 1] == scaffold)
        hs, ws = [p.shape[0] for p in patches], [p.shape[1] for p in patches]
        if min(hs) < 8 or min(ws) < 3:
            continue
        gaps = [starts[i + 1] - starts[i] - ws[i] for i in range(len(starts) - 1)]
        try:
            hh, ww, yy = _linear(hs), _linear(ws), _linear(tops)
            gg = _linear(gaps)
        except AssertionError:
            continue
        if min(gaps) < 1:
            continue
        for style in ("frame", "left_comb", "right_comb", "ladder"):
            for spacing in ((0,) if style == "frame" else (2, 3, 4, 5)):
                if not all(np.array_equal(p, _mask(style, *p.shape, spacing)) for p in patches):
                    continue
                m = dict(style=style, spacing=spacing, h=hh, w=ww, y=yy,
                         gap=gg, x=starts[0], n=len(groups), transposed=transposed)
                candidates.append(m)
    assert candidates, "visible strokes establish no supported carrier progression"
    return candidates


def _completed(a, scaffold, bb, m):
    """Complete whole relevant carriers before measuring component origins.

    Traversal cannot jump across a nonpositive size or gap. Relevant objects
    must fit the input canvas, even if their origin lies outside the output crop.
    """
    shape = a.T.shape if m["transposed"] else a.shape
    q = (bb[2], bb[3], bb[0], bb[1]) if m["transposed"] else bb
    field = np.zeros(shape, dtype=bool)
    requested = []
    forward = q[2] > m["x"]
    indices = list(range(m["n"]))
    extension = range(m["n"], 40) if forward else range(-1, -40, -1)
    for i in extension:
        y, x, h, w = _geom(m, i)
        if (forward and x > q[3]) or (not forward and x + w <= q[2]):
            break
        gi = i - 1 if forward else i
        gap = m["gap"][0] + gi * m["gap"][1]
        assert h >= 1 and w >= 3 and gap >= 1, "undefined intervening carrier"
        indices.append(i)
    for i in indices:
        y, x, h, w = _geom(m, i)
        intersects = x <= q[3] and x + w > q[2] and y <= q[1] and y + h > q[0]
        if i not in range(m["n"]) and not intersects:
            continue
        assert 0 <= y and y + h <= shape[0] and 0 <= x and x + w <= shape[1], "carrier outside input canvas"
        field[y:y + h, x:x + w] |= _mask(m["style"], h, w, m["spacing"])
        if intersects:
            requested.append((i, (y, x, h, w)))
    assert requested, "target must intersect at least one extrapolated carrier"
    return (field.T.copy() if m["transposed"] else field), requested


def _components(fg, diagonal=True):
    remaining = set(map(tuple, np.argwhere(fg)))
    groups = []
    while remaining:
        p = min(remaining)
        remaining.remove(p)
        stack, cells = [p], []
        while stack:
            r, c = stack.pop()
            cells.append((int(r), int(c)))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if not diagonal and dr and dc:
                        continue
                    n = (r + dr, c + dc)
                    if n in remaining:
                        remaining.remove(n)
                        stack.append(n)
        groups.append(cells)
    return groups


def _members(a, field, bb=None, diagonal=True):
    fg = field | np.isin(a, list(CODES))
    if bb is not None:  # Deliberately wrong crop-first ablation used in checks.
        keep = np.zeros(a.shape, dtype=bool)
        r0, r1, c0, c1 = bb
        keep[r0:r1 + 1, c0:c1 + 1] = True
        fg &= keep
    live, dead, multi, cavities = [], [], 0, []
    for cells in _components(fg, diagonal=diagonal):
        points = [(r, c, int(a[r, c])) for r, c in cells if int(a[r, c]) in CODES]
        origin = min(r for r, _ in cells), min(c for _, c in cells)
        if len(points) == 1 and points[0][:2] == origin:
            live.append(points[0])
            if len(cells) == 1:
                cavities.append(points[0])
        else:
            dead.extend(points)
            multi += int(len(points) > 1)
    return sorted(live), sorted(dead), multi, cavities


def _print(bb, bg, records):
    r0, r1, c0, c1 = bb
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=np.int8)
    for r, c, code in records:
        ink, rows = BOOK[code]
        for dr, row in enumerate(rows):
            for dc, value in enumerate(row):
                if value == "1" and r0 <= r + dr <= r1 and c0 <= c + dc <= c1:
                    out[r + dr - r0, c + dc - c0] = ink
    return out


def _analyze(input_grid):
    a = _grid(input_grid)
    vals, counts = np.unique(a, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    assert np.count_nonzero(a == bg) > a.size // 2 and bg not in CODES
    marker, bb = _annotations(a, bg)
    colors = set(map(int, vals)) - {bg, marker} - CODES
    assert len(colors) == 1, "exactly one structural ink is required"
    scaffold = colors.pop()
    r0, r1, c0, c1 = bb
    assert not np.any(a[r0:r1 + 1, c0:c1 + 1] == scaffold)
    points = [(int(r), int(c), int(a[r, c])) for r, c in np.argwhere(np.isin(a, list(CODES)))]
    assert points and all(r0 <= r <= r1 and c0 <= c <= c1 for r, c, _ in points)
    assert {bg, marker, scaffold}.isdisjoint({BOOK[c][0] for _, _, c in points})
    predictions = []
    for m in _fit(a, scaffold, bb):
        try:
            field, requested = _completed(a, scaffold, bb, m)
        except AssertionError:
            continue
        live, dead, multi, isolated = _members(a, field)
        predictions.append((_print(bb, bg, live), m, field, requested, live, dead, multi, isolated))
    assert predictions, "no well-defined completion reaches the crop"
    assert all(np.array_equal(p[0], predictions[0][0]) for p in predictions), "ambiguous completion"
    out, m, field, requested, live, dead, multi, isolated = predictions[0]
    assert live and dead, "examples must reveal selective activation"
    occupied = set()
    for r, c, _ in live:
        footprint = {(rr, cc) for rr in range(r, min(r + 4, r1 + 1))
                     for cc in range(c, min(c + 4, c1 + 1))}
        assert not occupied.intersection(footprint), "live stamp windows overlap"
        occupied.update(footprint)
    raw = _print(bb, bg, points)
    assert not np.array_equal(raw, out), "restoring connectivity must affect the answer"
    info = dict(background=bg, marker=marker, scaffold=scaffold, bounds=bb,
                model=m, field=field, requested=requested, live=live, dead=dead,
                multi=multi, isolated=isolated, points=points,
                crop_first_differs=not np.array_equal(_print(bb, bg, _members(a, field, bb)[0]), out))
    return out, info


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive geometry, topology, substitutions and crop entirely from input."""
    return _analyze(input_grid)[0].copy()


def _full_intersects(p, bb):
    r, c, _ = p
    r0, r1, c0, c1 = bb
    return r0 <= r and r + 3 <= r1 and c0 <= c and c + 3 <= c1


def _make_pair(rng, roles, index, test):
    bg, scaffold, marker, codes = roles
    # The first two demonstrations teach whole-object membership, multiple
    # codes, cavities and completion outside the crop, even for a 2/1 puzzle.
    first = not test and index == 0
    second = not test and index == 1
    axis = int(rng.integers(2))
    if first or second:
        axis = roles.axis ^ int(second)
    targets = 2 if first else (1 if second or test else int(rng.integers(1, 3)))
    skip = int(rng.integers(1, 3)) if test else (0 if index < 2 else int(rng.integers(2)))
    visible = 3 if first or second else int(rng.integers(3, 5))
    total = visible + skip + targets
    backward = bool(rng.integers(2))
    style = "frame" if first else (str(rng.choice(["frame", "ladder"])) if second
                                  else str(rng.choice(["frame", "left_comb", "right_comb", "ladder"])))
    spacing = 0 if style == "frame" else int(rng.integers(2, 6))
    if second and style == "ladder":
        spacing = int(rng.integers(4, 6))
    width = int(rng.integers(3, 5)) if first else (5 if second else int(rng.integers(3, 7)))
    dw = int(rng.choice([0, 0, 0, 1, -1])) if not second else 0
    if dw < 0:
        width += total - 1
    dh = int(rng.choice([-2, -1, 1, 2])) if first else int(rng.integers(-2, 3))
    if first:
        # The requested carriers grow in the continuation direction. A probe
        # on the farther carrier's new bottom stroke defeats height freezing.
        dh = abs(dh) * (-1 if backward else 1)
    if second:
        dh = int(rng.choice([-1, 1]))
    height = int(rng.integers(8, 13)) + max(0, -(total - 1) * dh)
    drift = int(rng.integers(-2, 3))
    gap = 2 if second else int(rng.integers(1, 4))
    if first:
        gap = max(2, gap)
    dg = 0 if first or second else int(rng.choice([0, 0, 0, 1, -1]))
    if dg < 0:
        gap += total - 2
    objects, x = [], 0
    for i in range(total):
        h, w, y = height + i * dh, width + i * dw, i * drift
        if h < 8 or w < 3:
            return None
        z = _stroke_object(style, h, w, spacing)
        objects.append((y, x, z))
        x += w + gap + i * dg
    seen_indices = list(range(total - visible, total)) if backward else list(range(visible))
    target_indices = list(range(targets)) if backward else list(range(visible + skip, total))
    ty0 = min(objects[i][0] for i in target_indices)
    ty1 = max(objects[i][0] + len(objects[i][2]) - 1 for i in target_indices)
    tx0 = min(objects[i][1] for i in target_indices)
    tx1 = max(objects[i][1] + objects[i][2].shape[1] - 1 for i in target_indices)
    top_cut = 1 if second else (int(rng.integers(0, 4)) if test else int(rng.choice([-2, -2, 0, 1])))
    if first:
        top_cut = -2
    left_pad = 2 if second else int(rng.integers(min(gap, 2) + 1))
    right_pad = min(gap, 1 if second else int(rng.integers(3)))
    if first:
        if backward:
            left_pad = max(1, left_pad)
        else:
            right_pad = max(1, right_pad)
    c0, c1 = tx0 - left_pad, tx1 + right_pad
    if test and rng.random() < .5:
        c0 = tx0 + int(rng.integers(0, 2))
    r0 = ty0 + top_cut
    free_offset = 3 if first else int(rng.integers(2, 4))
    free_visible = (int(rng.integers(2, 4)) if second else
                    int(rng.integers(2, 5)) if test else 4)
    r1 = ty1 + free_offset + free_visible - 1
    if c1 - c0 < 3:
        return None
    ymin = min([r0] + [y for y, _, _ in objects])
    ymax = max([r1] + [y + len(z) - 1 for y, _, z in objects])
    xmin = min(0, c0)
    xmax = max(x - (gap + (total - 1) * dg) - 1, c1)
    H, W = ymax - ymin + 1, xmax - xmin + 1
    if max(H, W) > 30:
        return None
    top = int(rng.integers(min(2, 30 - H) + 1))
    bottom = int(rng.integers(min(2, 30 - H - top) + 1))
    left = int(rng.integers(min(2, 30 - W) + 1))
    right = int(rng.integers(min(2, 30 - W - left) + 1))
    dr, dc = top - ymin, left - xmin
    a = np.full((H + top + bottom, W + left + right), bg, dtype=np.int8)
    field = np.zeros(a.shape, dtype=bool)
    moved = []
    for i, (y, x, z) in enumerate(objects):
        y, x = y + dr, x + dc
        field[y:y + len(z), x:x + z.shape[1]] |= z
        if i in seen_indices:
            a[y:y + len(z), x:x + z.shape[1]][z] = scaffold
        moved.append((y, x, z))
    bb = (r0 + dr, r1 + dr, c0 + dc, c1 + dc)
    if axis:
        a, field = a.T.copy(), field.T.copy()
        moved = [(x, y, z.T.copy()) for y, x, z in moved]
        bb = (bb[2], bb[3], bb[0], bb[1])
    forbidden = _corners(bb)
    r0, r1, c0, c1 = bb
    points = []

    def add(r, c, code):
        p = (int(r), int(c))
        if p in forbidden or not (r0 <= r <= r1 and c0 <= c <= c1):
            return False
        if any(max(abs(r - rr), abs(c - cc)) <= 1 for rr, cc, _ in points):
            return False
        a[p] = int(code)
        points.append((p[0], p[1], int(code)))
        return True

    # A sole code at a completed carrier's origin is a valid instruction.
    live_index = target_indices[-1] if first and backward else target_indices[0]
    if targets == 2:
        y, x, _ = moved[live_index]
        if not add(y, x, codes[0]):
            return None
    # One other carrier carries an off-origin code. First demonstration also
    # puts a code at its origin: the two connected codes invalidate each other.
    rejected_index = target_indices[0] if first and backward else target_indices[-1]
    y, x, z = moved[rejected_index]
    if first:
        if not add(y, x, codes[1]):
            return None
    locations = [(y + int(r), x + int(c)) for r, c in np.argwhere(z)
                 if (int(r), int(c)) != (0, 0)]
    rng.shuffle(locations)
    if first:
        # A diagonal-only attachment beyond the far carrier's new bottom
        # corner distinguishes eight-connectivity from four-connectivity.
        cy, cx, cz = objects[rejected_index]
        probe_y = cy + dr + len(cz)
        probe_x = cx + dc + cz.shape[1]
        locations = [(probe_x, probe_y)] if axis else [(probe_y, probe_x)]
    if second:
        # First surviving structural coordinate after the transverse top cut.
        locations = [(y, x + 1)] if axis else [(y + 1, x)]
    added = False
    for rr, cc in locations:
        if add(rr, cc, int(rng.choice(codes))):
            added = True
            break
    if not added:
        return None
    if not first and not second and rng.random() < .45:
        for rr, cc in locations:
            if add(rr, cc, int(rng.choice(codes))):
                break
    # Isolated codes in true cavities are active even inside a carrier's box.
    cavity_required = second
    cavity_added = False
    if second or rng.random() < .55:
        hole_candidates = []
        for i in target_indices:
            y, x, z = moved[i]
            for rr in range(y + 2, y + len(z) - 2):
                for cc in range(x + 2, x + z.shape[1] - 2):
                    if field[rr - 1:rr + 2, cc - 1:cc + 2].any():
                        continue
                    hole_candidates.append((rr, cc))
        rng.shuffle(hole_candidates)
        for rr, cc in hole_candidates:
            if not _full_intersects((rr, cc, codes[1]), bb):
                continue
            # Avoid all live-origin stamp rectangles, including their blanks.
            if any(abs(rr - r) < 4 and abs(cc - c) < 4 for r, c, _ in points
                   if (r, c) == moved[live_index][:2] and targets == 2):
                continue
            if add(rr, cc, codes[1]):
                cavity_added = True
                break
    if cavity_required and not cavity_added:
        return None
    # Another isolated code is beyond the carriers, establishing a second
    # dictionary entry with a full 4x4 occurrence in each first demonstration.
    rr = ty1 + dr + free_offset
    cc = c0 if not axis else (tx0 + dc if second else r0)
    if axis:
        rr, cc = cc, rr
    else:
        cc = tx0 + dc if second else int(rng.integers(c0, max(c0 + 1, c1 - 2)))
    if first:
        cc = c0 if not axis else cc
    if not add(rr, cc, codes[1] if first else codes[0]):
        return None
    for p in forbidden:
        a[p] = marker
    live, dead, _, _ = _members(a, field)
    if not live or not dead:
        return None
    if second and (rr, cc, codes[0]) not in live:
        return None
    # Generation's ground truth uses explicit strokes and its selected tokens.
    out = _print(bb, bg, live)
    try:
        pred, info = _analyze(a)
    except AssertionError:
        return None
    if not np.array_equal(pred, out):
        return None
    if first and not all(any(c == code and _full_intersects((r, ccol, c), bb)
                             for r, ccol, c in live) for code in codes):
        return None
    if second and not info["crop_first_differs"]:
        return None
    return {"input": a, "output": out}


class _Roles(tuple):
    """Generation-only conventions; never passed to solve or validation."""
    def __new__(cls, values, axis):
        obj = tuple.__new__(cls, values)
        obj.axis = axis
        return obj


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in (("num_train", num_train, 2, 10), ("num_test", num_test, 1, 4)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in {low}..{high}")
    if num_train <= num_test:
        raise ValueError("num_train must exceed num_test")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    rng = np.random.default_rng(seed)
    codes = tuple(map(int, rng.choice(sorted(CODES), 2, replace=False)))
    available = [c for c in range(10) if c not in CODES and c not in {BOOK[k][0] for k in codes}]
    bg, scaffold, marker = map(int, rng.choice(available, 3, replace=False))
    roles = _Roles((bg, scaffold, marker, codes), int(rng.integers(2)))
    puzzle, used = {"train": [], "test": []}, set()
    for split, count in (("train", int(num_train)), ("test", int(num_test))):
        for index in range(count):
            for _ in range(600):
                pair = _make_pair(rng, roles, index, split == "test")
                if pair is not None and _key(pair["input"]) not in used:
                    used.add(_key(pair["input"]))
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("bounded geometry construction exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def validate_generated_puzzle(puzzle):
    assert isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}
    assert all(isinstance(puzzle[s], list) for s in puzzle)
    assert 2 <= len(puzzle["train"]) <= 10 and 1 <= len(puzzle["test"]) <= 4
    assert len(puzzle["train"]) > len(puzzle["test"])
    seen, conventions, taught, infos = set(), set(), set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            assert isinstance(pair, dict) and set(pair) == {"input", "output"}
            a, b = _grid(pair["input"]), _grid(pair["output"])
            assert _key(a) not in seen
            seen.add(_key(a))
            expected, info = _analyze(a)
            assert np.array_equal(expected, b), "label differs from input-derived prediction"
            infos[split].append(info)
            conventions.add((info["background"], info["scaffold"], info["marker"]))
            r0, r1, c0, c1 = info["bounds"]
            assert b.shape == (r1 - r0 + 1, c1 - c0 + 1)
            assert set(map(int, np.unique(b))) <= {info["background"]} | {BOOK[c][0] for _, _, c in info["live"]}
            # Direct point copying and structural completion are distinct wrong
            # outputs; the temporary material never becomes output ink.
            point_only = np.full(b.shape, info["background"], dtype=np.int8)
            for r, c, code in info["live"]:
                point_only[r - r0, c - c0] = code
            structural = np.full(b.shape, info["background"], dtype=np.int8)
            structural[info["field"][r0:r1 + 1, c0:c1 + 1]] = info["scaffold"]
            assert not np.array_equal(b, point_only) and not np.array_equal(b, structural)
            if split == "train":
                taught.update(code for r, c, code in info["live"] if _full_intersects((r, c, code), info["bounds"]))
    assert len(conventions) == 1
    used_codes = {code for s in infos.values() for info in s for _, _, code in info["points"]}
    assert len(used_codes) == 2 and used_codes <= taught
    assert any(i["multi"] for i in infos["train"]), "multiple connected codes must be demonstrated"
    assert any(i["crop_first_differs"] for i in infos["train"]), "whole-object membership must be demonstrated"
    assert any(i["model"]["h"][1] or i["model"]["w"][1] for i in infos["train"])
    assert any(_cavity_records(i) for i in infos["train"]), "cavity activation must be demonstrated"
    assert any(not _full_intersects(t, i["bounds"]) for i in infos["train"] for t in i["live"]), "clipped live stamps must be demonstrated"
    assert any(not np.array_equal(_frozen_height_prediction(_grid(pair["input"]), i), pair["output"])
               for pair, i in zip(puzzle["train"], infos["train"])), "constant-height shortcut must fail in training"
    assert any(not np.array_equal(_print(i["bounds"], i["background"],
                                       _members(_grid(pair["input"]), i["field"], diagonal=False)[0]), pair["output"])
               for pair, i in zip(puzzle["train"], infos["train"])), "four-connectivity shortcut must fail in training"
    assert any(not np.array_equal(_origin_only_prediction(_grid(pair["input"]), i), pair["output"])
               for pair, i in zip(puzzle["train"], infos["train"])), "ignoring the second code must fail in training"
    # Requested object indices come from input fits, not generation flags.
    def distance(i):
        m = i["model"]
        indices = [j for j, _ in i["requested"]]
        return min(indices) - m["n"] if min(indices) >= m["n"] else -max(indices) - 1
    assert all(distance(i) >= 1 for i in infos["test"]), "test must skip at least one unshown carrier"
    assert any(distance(i) == 0 for i in infos["train"])


def _cavity_records(info):
    """Isolated live codes strictly inside a fitted carrier's bounding box."""
    out = []
    m = info["model"]
    for p in info["isolated"]:
        r, c, _ = p
        if m["transposed"]:
            r, c = c, r
        if any(y < r < y + h - 1 and x < c < x + w - 1
               for _, (y, x, h, w) in info["requested"]):
            out.append(p)
    return out


def _frozen_height_prediction(a, info):
    """Wrong alternative: use the nearest observed height for every target."""
    m, bb = info["model"], info["bounds"]
    forward = min(i for i, _ in info["requested"]) >= m["n"]
    height = m["h"][0] + (m["n"] - 1) * m["h"][1] if forward else m["h"][0]
    canonical = (a == info["scaffold"]).T.copy() if m["transposed"] else (a == info["scaffold"]).copy()
    for _, (y, x, _, w) in info["requested"]:
        # A wrong taller template may reach the input boundary; intersection
        # is used here so the ablation is still a fully defined alternative.
        z = _mask(m["style"], height, w, m["spacing"])
        lo, hi = max(0, y), min(canonical.shape[0], y + height)
        canonical[lo:hi, x:x + w] |= z[lo - y:hi - y]
    field = canonical.T if m["transposed"] else canonical
    return _print(bb, info["background"], _members(a, field)[0])


def _origin_only_prediction(a, info):
    """Wrong alternative: let an origin code print despite another code."""
    records = []
    for cells in _components(info["field"] | np.isin(a, list(CODES))):
        r, c = min(p[0] for p in cells), min(p[1] for p in cells)
        if int(a[r, c]) in CODES:
            records.append((r, c, int(a[r, c])))
    return _print(info["bounds"], info["background"], records)


def self_test():
    """Bounded checks with input-derived diversity and mechanism ablations."""
    # Literal topology fixtures have manually specified outcomes. In particular,
    # being inside a box differs from being connected to its structural strokes.
    field = np.zeros((12, 12), dtype=bool)
    field[3:10, 3] = field[3:10, 9] = True
    field[3, 3:10] = field[9, 3:10] = True
    a = np.zeros((12, 12), dtype=np.int8)
    a[3, 3], a[6, 6] = 2, 8
    assert _members(a, field)[0] == [(3, 3, 2), (6, 6, 8)]
    a[:] = 0
    a[3, 5] = 2
    assert _members(a, field)[0] == [] and _members(a, field)[1] == [(3, 5, 2)]
    a[3, 3] = 8
    assert _members(a, field)[0] == [] and _members(a, field)[2] == 1
    a[:] = 0
    a[10, 10] = 2
    assert _members(a, field)[1] == [(10, 10, 2)]  # diagonal contact counts
    a[:] = 0
    a[4, 3] = 2
    assert _members(a, field)[0] == []
    assert _members(a, field, (4, 11, 2, 11))[0] == [(4, 3, 2)]
    assert np.array_equal(_print((0, 1, 0, 1), 0, [(0, 0, 2)]), np.array([[4, 4], [4, 0]]))
    puzzles, shapes, styles, axes, growth, gaps, sizes, lives = set(), set(), set(), set(), set(), set(), set(), set()
    dictionaries, role_sets, directions, distances = set(), set(), set(), set()
    cavity_total = partial_total = multi_total = clipped_total = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        validate_generated_puzzle(p)
        puzzles.add(tuple(_key(pair["input"]) for s in ("train", "test") for pair in p[s]))
        puzzle_codes = set()
        for split in ("train", "test"):
            for pair in p[split]:
                pred, info = _analyze(pair["input"])
                assert np.array_equal(pred, pair["output"])
                m = info["model"]
                shapes.add((pair["input"].shape, pair["output"].shape))
                styles.add(m["style"])
                axes.add(m["transposed"])
                growth.add((m["h"][1], m["w"][1]))
                gaps.add(m["gap"])
                sizes.update((g[2], g[3]) for _, g in info["requested"])
                lives.add(len(info["live"]))
                puzzle_codes.update(c for _, _, c in info["points"])
                role_sets.add((info["background"], info["scaffold"], info["marker"]))
                indices = [j for j, _ in info["requested"]]
                forward = min(indices) >= m["n"]
                directions.add((m["transposed"], forward))
                distances.add(min(indices) - m["n"] if forward else -max(indices) - 1)
                cavity_total += bool(_cavity_records(info))
                partial_total += info["crop_first_differs"]
                multi_total += bool(info["multi"])
                clipped_total += any(not _full_intersects(t, info["bounds"]) for t in info["live"])
        dictionaries.add(tuple(sorted(puzzle_codes)))
    assert len(puzzles) == 42 and len(shapes) >= 60
    assert styles == {"frame", "left_comb", "right_comb", "ladder"} and len(axes) == 2
    assert len(growth) >= 8 and any(dw for _, dw in growth)
    assert len(gaps) >= 5 and len(sizes) >= 20 and len(lives) >= 3
    assert len(dictionaries) == 6 and len(role_sets) >= 20 and len(directions) == 4
    assert {0, 1, 2} <= distances
    assert min(cavity_total, partial_total, multi_total, clipped_total) >= 20
    # Count boundaries, determinism, and corrupted-label rejection.
    for nt, nq in ((2, 1), (10, 4), (6, 3)):
        a, b = generate_puzzle(918870884, nt, nq), generate_puzzle(918870884, nt, nq)
        assert all(np.array_equal(x[k], y[k]) for s in a for x, y in zip(a[s], b[s]) for k in x)
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3, 1.2), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal count accepted")
    p = copy.deepcopy(generate_puzzle(0, 2, 1))
    out, info = _analyze(p["test"][0]["input"])
    r, c, code = info["live"][0]
    r0, r1, c0, c1 = info["bounds"]
    p["test"][0]["output"][r - r0:min(r - r0 + 4, out.shape[0]), c - c0:min(c - c0 + 4, out.shape[1])] = info["background"]
    try:
        validate_generated_puzzle(p)
    except AssertionError:
        pass
    else:
        raise AssertionError("deleted stamp accepted")


if __name__ == "__main__":
    self_test()
    print("self_test passed across 42 seeds")
