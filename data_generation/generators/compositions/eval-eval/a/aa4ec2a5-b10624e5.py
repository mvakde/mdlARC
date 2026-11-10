"""Sealed-window printing: square registration inside topological stencils.

Only NumPy and the standard library are used. All solving roles are recovered
from input pixels; generation preferences never enter solve().
"""
from collections import deque
from math import gcd

import numpy as np


N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
N8 = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c)


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(name + " must be an integer")
    return int(value)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= d <= 30 for d in a.shape), "invalid grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "grid must have integer cells")
    _require(np.all((a >= 0) & (a <= 9)), "invalid color")
    return a.astype(np.int8, copy=True)


def _regions(mask, neighbors=N4):
    """Return deterministic components and a label raster (exterior = -1)."""
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    components = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if labels[r, c] >= 0:
            continue
        labels[r, c] = len(components)
        cells, stack = [], [(r, c)]
        while stack:
            y, x = stack.pop()
            cells.append((y, x))
            for dy, dx in neighbors:
                yy, xx = y + dy, x + dx
                if 0 <= yy < h and 0 <= xx < w and mask[yy, xx] and labels[yy, xx] < 0:
                    labels[yy, xx] = len(components)
                    stack.append((yy, xx))
        components.append(cells)
    return components, labels


def _enclosed(walls, neighbors=N4):
    """Marks and motif colors are occupants, so only wall pixels obstruct flow."""
    blocked = np.pad(walls, 1, constant_values=False)
    reached = np.zeros(blocked.shape, dtype=bool)
    reached[0, 0] = True
    queue = deque([(0, 0)])
    h, w = blocked.shape
    while queue:
        r, c = queue.popleft()
        for dr, dc in neighbors:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and not blocked[rr, cc] and not reached[rr, cc]:
                reached[rr, cc] = True
                queue.append((rr, cc))
    return (~blocked & ~reached)[1:-1, 1:-1]


def _box(cells):
    rr, cc = np.array(cells, dtype=int).T
    return int(rr.min()), int(rr.max()) + 1, int(cc.min()), int(cc.max()) + 1


def _square(cells):
    r0, r1, c0, c1 = _box(cells)
    if r1 - r0 == c1 - c0 and len(cells) == (r1 - r0) ** 2:
        return r0, c0, r1 - r0
    return None


def _key(grid):
    a = np.asarray(grid, dtype=np.int8)
    return a.shape, a.tobytes()


def _decode(value):
    grid = _grid(value)
    bg = int(grid[0, 0])
    _require(all(np.all(edge == bg) for edge in (grid[0], grid[-1], grid[:, 0], grid[:, -1])),
             "expected an exterior background border")
    components, scene_labels = _regions(grid != bg)
    candidates = []
    for color in sorted(set(map(int, np.unique(grid))) - {bg}):
        parts, _ = _regions(grid == color)
        squares = [_square(cells) for cells in parts]
        if len(parts) < 2 or any(s is None for s in squares):
            continue
        if not any(len(components[int(scene_labels[cells[0]])]) == len(cells) for cells in parts):
            continue
        mixed = {int(scene_labels[cells[0]]) for cells in parts
                 if len(set(int(grid[p]) for p in components[int(scene_labels[cells[0]])])) > 1}
        if len(mixed) == 1:
            candidates.append((color, parts, squares, mixed.pop()))
    _require(len(candidates) == 1, "registration/reference identity is ambiguous")
    marker, parts, squares, reference_id = candidates[0]
    source_cells = components[reference_id]
    source_mask = scene_labels == reference_id
    reference_parts = [i for i, cells in enumerate(parts) if source_mask[cells[0]]]
    _require(len(reference_parts) == 1, "reference has more than one square")
    reference_index = reference_parts[0]
    ar, ac, side = squares[reference_index]
    outside_colors = set(map(int, grid[(grid != bg) & ~source_mask])) - {marker}
    _require(len(outside_colors) == 1, "outside foreground must have one wall color")
    wall = next(iter(outside_colors))
    source_colors = set(int(grid[p]) for p in source_cells)
    _require(wall not in source_colors and source_colors - {marker}, "invalid source palette")
    wall_mask = grid == wall
    wall_parts, wall_labels = _regions(wall_mask)
    _require(len(wall_parts) == len(_regions(wall_mask, N8)[0]), "diagonal wall contact")
    holes = _enclosed(wall_mask)
    _require(np.array_equal(holes, _enclosed(wall_mask, N8)), "diagonal-only cavity leak")
    cavity_parts, cavity_labels = _regions(holes)
    _require(len(cavity_parts) == len(_regions(holes, N8)[0]), "diagonally joined cavities")
    for cells in cavity_parts:
        surrounding = {int(wall_labels[r + dr, c + dc]) for r, c in cells for dr, dc in N4
                       if 0 <= r + dr < grid.shape[0] and 0 <= c + dc < grid.shape[1]
                       and wall_mask[r + dr, c + dc]}
        _require(len(surrounding) == 1, "collective/nested cavity ownership")
    _require(not holes[source_mask].any(), "reference must be exterior")
    motif = {(r - ar, c - ac): int(grid[r, c]) for r, c in source_cells}
    anchors, occupied_cavities = [], set()
    for i, (cells, square) in enumerate(zip(parts, squares)):
        if i == reference_index:
            continue
        r, c, d = square
        _require(len(components[int(scene_labels[cells[0]])]) == d * d,
                 "destination square must be isolated")
        labels = {int(cavity_labels[p]) for p in cells}
        _require(len(labels) == 1, "square straddles cavity/exterior")
        cavity = labels.pop()
        if cavity >= 0:
            _require(cavity not in occupied_cavities, "two squares in one cavity")
            occupied_cavities.add(cavity)
        fy, fx = 2 * r + d < 2 * ar + side, 2 * c + d < 2 * ac + side
        anchors.append({"r": r, "c": c, "side": d, "cavity": cavity, "flips": (fy, fx)})
    _require(any(a["cavity"] >= 0 for a in anchors), "no active window")
    return {"grid": grid, "background": bg, "wall": wall, "marker": marker,
            "source_mask": source_mask, "source": (ar, ac, side), "motif": motif,
            "wall_mask": wall_mask, "cavity_labels": cavity_labels,
            "cavities": cavity_parts, "anchors": anchors}


def _exact_pattern(motif, source_side, destination_side, flips, bg):
    """Reduce uniform denominator-sized cells, then expand numerator-sized cells.

    This tests actual color boundaries, not an assumed generator lattice.
    Coordinates remain relative to the square; reflection reverses its interior.
    """
    divisor = gcd(source_side, destination_side)
    numerator, denominator = destination_side // divisor, source_side // divisor
    r0, r1, c0, c1 = _box(list(motif))
    r0, c0 = r0 // denominator * denominator, c0 // denominator * denominator
    r1 = -(-r1 // denominator) * denominator
    c1 = -(-c1 // denominator) * denominator
    raw = np.full((r1 - r0, c1 - c0), bg, dtype=np.int8)
    for (r, c), color in motif.items():
        raw[r - r0, c - c0] = color
    h, w = raw.shape[0] // denominator, raw.shape[1] // denominator
    blocks = raw.reshape(h, denominator, w, denominator)
    low, high = blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))
    _require(np.array_equal(low, high), "ambiguous fractional scaling")
    raster = np.repeat(np.repeat(low, numerator, axis=0), numerator, axis=1)
    base_r, base_c = r0 // denominator * numerator, c0 // denominator * numerator
    result = {}
    for rr, cc in np.argwhere(raster != bg):
        r, c = int(rr) + base_r, int(cc) + base_c
        if flips[0]:
            r = destination_side - 1 - r
        if flips[1]:
            c = destination_side - 1 - c
        result[r, c] = int(raster[rr, cc])
    return result


def _render(data, *, omit_axis=None, free=False, rectangular=False, fill=False, unit=False,
            absolute=False):
    inp = data["grid"]
    out = inp.copy()
    _, _, s = data["source"]
    bg, marker = data["background"], data["marker"]
    for a in data["anchors"]:
        cavity = a["cavity"]
        if cavity < 0 and not free:
            continue
        mask = data["cavity_labels"] == cavity if cavity >= 0 else inp == bg
        if rectangular and cavity >= 0:
            r0, r1, c0, c1 = _box(data["cavities"][cavity])
            mask = np.zeros(inp.shape, bool)
            mask[r0:r1, c0:c1] = True
        if free:
            mask = (inp == bg) | (inp == marker)
            mask &= ~data["source_mask"]
        if fill:
            detail = next(color for _, color in sorted(data["motif"].items()) if color != marker)
            out[mask & (inp == bg)] = detail
            continue
        flips = list(a["flips"])
        if absolute:
            flips = [2 * a["r"] + a["side"] < inp.shape[0],
                     2 * a["c"] + a["side"] < inp.shape[1]]
        if omit_axis in (0, 1):
            flips[omit_axis] = False
        if omit_axis == 2:
            flips = [False, False]
        pattern = _exact_pattern(data["motif"], s, s if unit else a["side"], flips, bg)
        for (r, c), color in pattern.items():
            rr, cc = a["r"] + r, a["c"] + c
            if 0 <= rr < inp.shape[0] and 0 <= cc < inp.shape[1] and mask[rr, cc]:
                if inp[rr, cc] == bg or (rectangular and inp[rr, cc] == data["wall"]):
                    out[rr, cc] = color
    return out


def solve(input_grid: np.ndarray) -> np.ndarray:
    return _render(_decode(input_grid))


def _asymmetric(motif, side):
    return all({(side - 1 - r if fy else r, side - 1 - c if fx else c): color
                for (r, c), color in motif.items()} != motif
               for fy, fx in ((True, False), (False, True), (True, True)))


def _motif(rng, marker, details, teaching=False, fine=False):
    a = 2 if fine else (1 if teaching else int(rng.choice([1, 1, 1, 2])))
    cells = {(r, c): marker for r in range(a) for c in range(a)}
    # Unequal arms, then optional branches; not a catalog of fixed templates.
    arm_y = int(rng.integers(1, 3 if teaching else 4))
    arm_x = int(rng.integers(1, 3 if teaching else 4))
    for r in range(-arm_y, 0):
        cells[r, 0] = details[0]
    for c in range(a, a + arm_x):
        cells[a - 1, c] = details[-1]
    if not teaching:
        for step in range(int(rng.integers(1, 6))):
            r, c = sorted(cells)[int(rng.integers(len(cells)))]
            dr, dc = N4[int(rng.integers(4))]
            q = r + dr, c + dc
            rs, cs = zip(*(set(cells) | {q}))
            if max(rs) - min(rs) < 5 and max(cs) - min(cs) < 5 and q not in cells:
                cells[q] = details[step % len(details)]
    for _ in range(int(rng.integers(4))):
        cells = {(c, a - 1 - r): color for (r, c), color in cells.items()}
    if rng.random() < .5:
        cells = {(r, a - 1 - c): color for (r, c), color in cells.items()}
    return cells, a


def _logical_pattern(motif, a, unit, flips):
    # Construction uses direct rectangle expansion of the logical source.
    pixels = {}
    for (r, c), color in motif.items():
        y, x = a - 1 - r if flips[0] else r, a - 1 - c if flips[1] else c
        for rr in range(y * unit, (y + 1) * unit):
            for cc in range(x * unit, (x + 1) * unit):
                pixels[rr, cc] = color
    return pixels


def _window(rng, motif, a, unit, flips, *, full=False, opened=False,
            irregular=False, secondary=False, simple=False):
    side = a * unit
    pattern = _logical_pattern(motif, a, unit, flips)
    pr0, pr1, pc0, pc1 = _box(list(pattern))
    if full:
        ch, cw = pr1 - pr0 + 2, pc1 - pc0 + 2
        sr, sc = 1 - pr0, 1 - pc0
    else:
        ch = int(rng.integers(side + 2, max(side + 3, 8)))
        cw = int(rng.integers(side + 2, max(side + 3, 8)))
        sr, sc = int(rng.integers(1, ch - side)), int(rng.integers(1, cw - side))
    if max(ch, cw) > 12:
        return None
    cavity = np.ones((ch, cw), bool)
    protected = np.zeros_like(cavity)
    protected[max(0, sr - 1):sr + side + 1, max(0, sc - 1):sc + side + 1] = True
    if irregular:
        for _ in range(10):
            cut = np.zeros_like(cavity)
            edge = int(rng.integers(4))
            if edge < 2:
                left = int(rng.integers(cw))
                right = int(rng.integers(left + 1, cw + 1))
                depth = int(rng.integers(1, max(2, ch // 2)))
                cut[:depth if edge == 0 else ch, left:right] = True
                if edge == 1:
                    cut[:ch - depth] = False
            else:
                top = int(rng.integers(ch))
                bottom = int(rng.integers(top + 1, ch + 1))
                depth = int(rng.integers(1, max(2, cw // 2)))
                cut[top:bottom, :depth if edge == 2 else cw] = True
                if edge == 3:
                    cut[:, :cw - depth] = False
            candidate = cavity & ~(cut & ~protected)
            if len(_regions(candidate)[0]) == 1 and np.array_equal(_enclosed(~candidate),
                                                                 _enclosed(~candidate, N8)):
                cavity = candidate
    # Require printing evidence and, for a partial aperture, true lost detail.
    detail = [(r, c) for (r, c), color in pattern.items() if not (0 <= r < side and 0 <= c < side)]
    visible = sum(0 <= sr + r < ch and 0 <= sc + c < cw and cavity[sr + r, sc + c]
                  for r, c in detail)
    if not visible or (not full and not opened and visible == len(detail)):
        return None
    thick = 1 if simple else int(rng.choice([1, 1, 2]))
    extra = 4 if secondary else 0
    walls = np.ones((ch + 2 * thick, cw + 2 * thick + extra), bool)
    aperture = np.zeros_like(walls)
    aperture[thick:thick + ch, thick:thick + cw] = cavity
    walls[aperture] = False
    if secondary:
        walls[thick:thick + 2, -3:-1] = False
    # Outer lobes vary the body's silhouette independently of its aperture.
    pad = 0 if simple else int(rng.integers(0, 2))
    if pad:
        old_h, old_w = walls.shape
        walls, aperture = np.pad(walls, pad), np.pad(aperture, pad)
        for _ in range(int(rng.integers(1, 4))):
            col = int(rng.integers(pad, pad + old_w))
            walls[:pad + 1, col:col + 1] = True
        thick += pad
    sr, sc = sr + thick, sc + thick
    if opened:
        # Cut an orthogonal path from a top-edge cavity cell to exterior.
        columns = np.flatnonzero(aperture.any(axis=0))
        col = int(rng.choice(columns))
        row = int(np.flatnonzero(aperture[:, col])[0])
        walls[:row + 1, col] = False
    holes = _enclosed(walls)
    if len(_regions(walls)[0]) != 1 or len(_regions(walls, N8)[0]) != 1:
        return None
    if not np.array_equal(holes, _enclosed(walls, N8)):
        return None
    active = bool(holes[sr, sc])
    if active == opened:
        return None
    if active:
        _, labels = _regions(holes)
        aperture = labels == labels[sr, sc]
    if not aperture[sr:sr + side, sc:sc + side].all():
        return None
    if not full:
        # Rotate the vessel independently of the prototype, so openings and
        # clipping peninsulas occur on every side rather than always on top.
        for _ in range(int(rng.integers(4))):
            sr, sc = walls.shape[1] - sc - side, sr
            walls, aperture = np.rot90(walls), np.rot90(aperture)
        if rng.random() < .5:
            sc = walls.shape[1] - sc - side
            walls, aperture = np.fliplr(walls), np.fliplr(aperture)
        visible = sum(0 <= sr + r < walls.shape[0] and 0 <= sc + c < walls.shape[1]
                      and aperture[sr + r, sc + c] for r, c in detail)
        if not visible or (not opened and visible == len(detail)):
            return None
    return {"walls": walls, "aperture": aperture, "seed": (sr, sc, side),
            "unit": unit, "active": active, "flips": flips}


def _placements(occupied, shape, border=1):
    mh, mw = shape
    h, w = occupied.shape
    if mh + 2 * border > h or mw + 2 * border > w:
        return np.empty((0, 2), int)
    p = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    valid = p[mh:, mw:] - p[:-mh, mw:] - p[mh:, :-mw] + p[:-mh, :-mw] == 0
    valid[:border] = valid[-border:] = False
    valid[:, :border] = valid[:, -border:] = False
    return np.argwhere(valid)


def _reserve(occupied, origin, shape):
    r, c = origin
    h, w = shape
    occupied[max(0, r - 1):r + h + 1, max(0, c - 1):c + w + 1] = True


def _assemble(rng, motif, a, source_unit, windows, palette, shape, *, fixed=False):
    bg, wall, marker = palette[:3]
    source = _logical_pattern(motif, a, source_unit, (False, False))
    r0, r1, c0, c1 = _box(list(source))
    source_shape = r1 - r0, c1 - c0
    for _ in range(1 if fixed else 45):
        occupied = np.zeros(shape, bool)
        choices = _placements(occupied, source_shape)
        if not len(choices):
            return None
        if fixed:
            registration_origin = 14 if a * source_unit >= 4 else 13
            origin = (registration_origin + r0, registration_origin + c0)
        else:
            # Prefer an interior reference for freely placed teaching directions.
            center = choices + np.array(source_shape) / 2
            near = choices[((center > np.array(shape) * .25) &
                            (center < np.array(shape) * .75)).all(axis=1)]
            pool = near if len(near) else choices
            origin = tuple(map(int, pool[int(rng.integers(len(pool)))]))
        ar, ac = origin[0] - r0, origin[1] - c0
        _reserve(occupied, origin, source_shape)
        positions = {}
        order = sorted(range(len(windows)), key=lambda i: -windows[i]["walls"].size)
        for i in order:
            window = windows[i]
            wh, ww = window["walls"].shape
            sr, sc, d = window["seed"]
            choices = _placements(occupied, (wh, ww))
            if not len(choices):
                break
            fy, fx = window["flips"]
            signs = ((2 * (choices[:, 0] + sr) + d < 2 * ar + a * source_unit) == fy) & \
                    ((2 * (choices[:, 1] + sc) + d < 2 * ac + a * source_unit) == fx)
            choices = choices[signs]
            if not len(choices):
                break
            if fixed and i < 2:
                desired = (1 if fy else 19), (1 if fx else 19)
                if i == 1 and ww > 10:
                    desired = (1, 10)
                matching = choices[(choices == desired).all(axis=1)]
                if not len(matching):
                    break
                pos = tuple(map(int, matching[0]))
            else:
                pos = tuple(map(int, choices[int(rng.integers(len(choices)))]))
            positions[i] = pos
            _reserve(occupied, pos, (wh, ww))
        else:
            inp = np.full(shape, bg, np.int8)
            for (r, c), color in source.items():
                inp[ar + r, ac + c] = color
            for i, window in enumerate(windows):
                r, c = positions[i]
                h, w = window["walls"].shape
                inp[r:r + h, c:c + w][window["walls"]] = wall
                sr, sc, d = window["seed"]
                inp[r + sr:r + sr + d, c + sc:c + sc + d] = marker
            out = inp.copy()
            for i, window in enumerate(windows):
                if not window["active"]:
                    continue
                r, c = positions[i]
                sr, sc, _ = window["seed"]
                pattern = _logical_pattern(motif, a, window["unit"], window["flips"])
                for (y, x), color in pattern.items():
                    yy, xx = sr + y, sc + x
                    if (0 <= yy < window["walls"].shape[0] and 0 <= xx < window["walls"].shape[1]
                            and window["aperture"][yy, xx]):
                        out[r + yy, c + xx] = color
            return {"input": inp, "output": out}
    return None


def _build_pair(rng, teaching, shift, test=False):
    palette = list(map(int, rng.permutation(10)))
    bg, wall, marker = palette[:3]
    details = palette[3:3 + int(rng.integers(1, 5))]
    motif, a = _motif(rng, marker, details, teaching=teaching is not None, fine=teaching == 1)
    if not _asymmetric(motif, a):
        return None
    source_unit = (2 if teaching == 0 else 1) if teaching is not None else int(rng.choice([1, 1, 2, 3]))
    if teaching is not None:
        flips = [(False, False), (True, True)] if teaching == 0 else [(False, True), (True, False)]
        units = [1, 2]
        descriptors = [(flips[0], units[0], True, False, False, False),
                       (flips[1], units[1], False, False, teaching == 1, teaching == 1),
                       ((bool(rng.integers(2)), bool(rng.integers(2))), 1, False, True, False, False)]
    else:
        n = 4 if test and shift == "count" else (3 if test else int(rng.integers(1, 4)))
        descriptors = []
        for i in range(n):
            opened = (test and i == n - 1) or (not test and i > 0 and rng.random() < .3)
            unit = 3 if test and shift == "scale" and i == 0 else int(rng.choice([1, 1, 2]))
            full = not opened and rng.random() < .3 and not (test and shift == "aperture")
            irregular = not full and (rng.random() < (.9 if test else .55))
            descriptors.append(((bool(rng.integers(2)), bool(rng.integers(2))), unit, full,
                                opened, irregular, not opened and rng.random() < .25))
    windows = []
    for flips, unit, full, opened, irregular, secondary in descriptors:
        window = _window(rng, motif, a, unit, flips, full=full, opened=opened,
                         irregular=irregular, secondary=secondary)
        if window is None:
            return None
        windows.append(window)
    low = 24 if teaching is not None or test else 16 + 2 * len(windows)
    shape = int(rng.integers(low, 31)), int(rng.integers(low, 31))
    return _assemble(rng, motif, a, source_unit, windows, palette, shape)


def _fallback(rng, teaching, seen, *, test=False, shift=None):
    """Finite small geometry and 720 visible four-role palettes."""
    palette = list(map(int, rng.permutation(10)))
    marker, color = palette[2], palette[3]
    a = 2 if teaching == 1 or (test and shift == "aperture") else 1
    motif = {(r, c): marker for r in range(a) for c in range(a)}
    motif[-1, 0], motif[a - 1, a] = color, palette[4]
    flips = [(False, False), (True, True)] if teaching != 1 else [(False, True), (True, False)]
    source_unit = 1 if teaching == 1 else 2
    windows = []
    for i in range(4 if test and shift == "count" else 3):
        f = flips[i] if i < 2 else ((True, True) if i == 2 else (False, True))
        unit = (3 if test and shift == "scale" else 2) if i == 1 else 1
        side = a * unit
        if i in (0, 3):
            pattern = _logical_pattern(motif, a, unit, f)
            r0, r1, c0, c1 = _box(list(pattern))
            aperture = np.pad(np.ones((r1 - r0 + 2, c1 - c0 + 2), bool), 1)
            sr, sc = 2 - r0, 2 - c0
            walls = ~aperture
        elif i == 1 and a == 2:
            # An indented aperture clips one half of a sub-square detail;
            # a separate small cavity is deliberately unmarked.
            walls = np.ones((10, 13), bool)
            aperture = np.zeros_like(walls)
            aperture[1:9, 1:9] = True
            if f[1]:
                aperture[7:9, 3:6] = False
            else:
                aperture[1:4, 7:9] = False
            walls[aperture] = False
            walls[1:3, 10:12] = False
            sr, sc = 2, 2
        else:
            n = side + 4
            walls = np.ones((n, n), bool)
            aperture = np.zeros_like(walls)
            aperture[1:-1, 1:-1] = True
            walls[aperture] = False
            sr, sc = 2, 2
            if i == 2:
                walls[0, 2] = False
        windows.append({"walls": walls, "aperture": aperture, "seed": (sr, sc, side),
                        "unit": unit, "active": i != 2, "flips": f})
    pair = _assemble(rng, motif, a, source_unit, windows, palette, (30, 30), fixed=True)
    _require(pair is not None, "fallback placement failed")
    for delta in range(10):
        remap = (np.arange(10) + delta) % 10
        candidate = {k: remap[v].astype(np.int8) for k, v in pair.items()}
        if _key(candidate["input"]) not in seen:
            return candidate
    # Exhaust 10*9*8 bijections for three visible role colors, completing the
    # bijection in sorted order. This exceeds the at-most-13 previous inputs.
    old_roles = palette[:3]
    old_rest = [c for c in range(10) if c not in old_roles]
    for b in range(10):
        for w in range(10):
            for m in range(10):
                if len({b, w, m}) != 3:
                    continue
                new_rest = [c for c in range(10) if c not in (b, w, m)]
                remap = np.empty(10, int)
                remap[old_roles + old_rest] = [b, w, m] + new_rest
                candidate = {k: remap[v].astype(np.int8) for k, v in pair.items()}
                if _key(candidate["input"]) not in seen:
                    return candidate
    raise AssertionError("finite uniqueness space exhausted")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    seed = _integer(seed, "seed")
    nt, nq = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if not (2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq):
        raise ValueError("require 2..10 train, 1..4 test, and train > test")
    rng = np.random.default_rng(seed)
    shift = str(rng.choice(["count", "scale", "aperture"]))
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", nq)):
        for index in range(count):
            teaching = index if split == "train" and index < 2 else None
            for _ in range(160):
                pair = _build_pair(rng, teaching, shift, test=split == "test")
                if pair is None or _key(pair["input"]) in seen:
                    continue
                data = _decode(pair["input"])
                if not np.array_equal(_render(data), pair["output"]):
                    raise AssertionError("constructor and pixel reconstruction disagree")
                if teaching == 1 and np.array_equal(_render(data, rectangular=True), pair["output"]):
                    continue
                if teaching == 1 and np.array_equal(_render(data, absolute=True), pair["output"]):
                    continue
                if np.array_equal(_render(data, free=True), pair["output"]):
                    continue
                break
            else:
                pair = _fallback(rng, teaching, seen, test=split == "test", shift=shift)
            seen.add(_key(pair["input"]))
            puzzle[split].append(pair)
    rng.shuffle(puzzle["train"])
    validate_generated_puzzle(puzzle)
    return puzzle


def _features(data):
    s = data["source"][2]
    bg = data["background"]
    active, losses, full, ratios, flip_types = [], [], 0, set(), set()
    open_printable = 0
    for a in data["anchors"]:
        if a["cavity"] < 0:
            p = _exact_pattern(data["motif"], s, a["side"], a["flips"], bg)
            open_printable += any(color != data["marker"] and
                                  0 <= a["r"] + r < data["grid"].shape[0] and
                                  0 <= a["c"] + c < data["grid"].shape[1] and
                                  data["grid"][a["r"] + r, a["c"] + c] == bg
                                  for (r, c), color in p.items())
            continue
        p = _exact_pattern(data["motif"], s, a["side"], a["flips"], bg)
        detail = [(r, c) for (r, c), color in p.items() if color != data["marker"]]
        kept = sum(0 <= a["r"] + r < data["grid"].shape[0] and
                   0 <= a["c"] + c < data["grid"].shape[1] and
                   data["cavity_labels"][a["r"] + r, a["c"] + c] == a["cavity"]
                   for r, c in detail)
        _require(kept > 0, "active generated window has no visible detail")
        active.append(a)
        losses.append(len(detail) - kept)
        full += kept == len(detail)
        g = gcd(s, a["side"])
        ratios.add((a["side"] // g, s // g))
        flip_types.add(a["flips"])
    seeded = {a["cavity"] for a in active}
    boundaries = [offset for (r, c), color in data["motif"].items()
                  for dr, dc, offset in ((-1, 0, r), (1, 0, r + 1), (0, -1, c), (0, 1, c + 1))
                  if data["motif"].get((r + dr, c + dc), bg) != color]
    return {"shape": data["grid"].shape, "active": len(active),
            "inactive": sum(a["cavity"] < 0 for a in data["anchors"]),
            "open_printable": int(open_printable), "fine": any(v % s for v in boundaries),
            "unmarked": len(data["cavities"]) - len(seeded), "full": full,
            "clipped": sum(v > 0 for v in losses), "losses": losses,
            "ratios": ratios, "flips": flip_types, "source_side": s,
            "wall_count": len(_regions(data["wall_mask"])[0]),
            "irregular": sum(len(cells) < (lambda b: (b[1] - b[0]) * (b[3] - b[2]))(_box(cells))
                             for cells in data["cavities"]),
            "palette": (bg, data["wall"], data["marker"]),
            "details": len(set(data["motif"].values()) - {data["marker"]})}


def validate_generated_puzzle(puzzle):
    _require(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "bad puzzle schema")
    _require(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    nt, nq = len(puzzle["train"]), len(puzzle["test"])
    _require(2 <= nt <= 10 and 1 <= nq <= 4 and nt > nq, "bad counts")
    seen, training, feats = set(), [], {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _require(isinstance(pair, dict) and set(pair) == {"input", "output"}, "bad pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _require(inp.shape == out.shape, "canvas dimensions changed")
            _require(_key(inp) not in seen, "duplicate input")
            seen.add(_key(inp))
            data = _decode(inp)
            expected = _render(data)
            _require(np.array_equal(expected, out), "incorrect stencil printing")
            _require(not np.array_equal(inp, out), "identity example")
            unchanged = data["source_mask"] | data["wall_mask"] | (data["cavity_labels"] < 0)
            _require(np.array_equal(inp[unchanged], out[unchanged]), "source/wall/exterior changed")
            marked = {a["cavity"] for a in data["anchors"] if a["cavity"] >= 0}
            for i in range(len(data["cavities"])):
                if i not in marked:
                    mask = data["cavity_labels"] == i
                    _require(np.array_equal(inp[mask], out[mask]), "unmarked cavity changed")
            feats[split].append(_features(data))
            if split == "train":
                training.append((data, out))
    _require(all(any(not np.array_equal(_render(d, **variant), o) for d, o in training)
                 for variant in ({"free": True}, {"fill": True}, {"rectangular": True},
                                 {"omit_axis": 0}, {"omit_axis": 1}, {"omit_axis": 2}, {"unit": True})),
             "training fails to distinguish a parent/axis/scale/stencil omission")
    _require(any(not np.array_equal(_render(d, absolute=True), o) for d, o in training),
             "training permits absolute canvas-quadrant reflection")
    tf = feats["train"]
    _require(set().union(*(f["flips"] for f in tf)) == {(False, False), (False, True), (True, False), (True, True)},
             "training lacks a relative reflection direction")
    _require(any(f["full"] for f in tf) and any(f["clipped"] for f in tf), "need full and clipped demonstrations")
    _require(any(f["inactive"] for f in tf) and any(f["unmarked"] for f in tf), "need open and unmarked counterexamples")
    _require(any(f["open_printable"] for f in tf), "open squares offer no topology counterexample")
    _require(any(f["fine"] for f in tf), "training must expose sub-square detail")
    ratios = set().union(*(f["ratios"] for f in tf))
    _require(any(n < d for n, d in ratios) and any(n > d for n, d in ratios), "need reduction and enlargement")


def _puzzle_key(p):
    return tuple(tuple((_key(pair["input"]), _key(pair["output"])) for pair in p[s]) for s in ("train", "test"))


def _regressions():
    # Opening a door changes topology before inspecting its registration mark.
    closed = np.ones((7, 7), bool)
    closed[1:6, 1:6] = False
    opened = closed.copy()
    opened[0, 3] = False
    _require(_enclosed(closed)[3, 3] and not _enclosed(opened).any(), "sealed/open regression")
    # Thin detail smaller than a 2x2 square: exact doubling, ambiguous 3/2.
    motif = {(r, c): 2 for r in range(2) for c in range(2)}
    motif[-1, 0], motif[0, 2] = 3, 4
    p = _exact_pattern(motif, 2, 4, (True, True), 0)
    _require(p[4, 2] == 3 and p[2, -2] == 4, "sub-square reflection regression")
    try:
        _exact_pattern(motif, 2, 3, (False, False), 0)
    except AssertionError:
        pass
    else:
        raise AssertionError("ambiguous scale accepted")
    coarse = _logical_pattern({(0, 0): 2, (0, 1): 3}, 1, 2, (False, False))
    _require(_exact_pattern(coarse, 2, 3, (False, False), 0)[0, 3] == 3, "exact rational scale rejected")
    # A complete input-only fixture: tied columns cause no horizontal flip;
    # two marked cavities in one connected wall are processed independently.
    grid = np.zeros((27, 27), np.int8)
    grid[1:8, 7:15] = 5
    grid[2:7, 8:14] = 0
    grid[3:5, 10:12] = 2
    grid[16:25, 3:24] = 5
    grid[17:24, 4:11] = 0
    grid[17:24, 13:22] = 0
    grid[20:22, 7:9] = 2
    grid[20:22, 16:18] = 2
    grid[10:12, 10:12] = 2
    grid[9, 10], grid[11, 12] = 3, 4
    expected = grid.copy()
    for r, c, color in ((5, 10, 3), (3, 12, 4), (19, 8, 3), (21, 6, 4),
                        (19, 16, 3), (21, 18, 4)):
        expected[r, c] = color
    _require(np.array_equal(solve(grid), expected), "ties/multiple marked cavities regression")
    grid[1, 11] = 0
    expected[1, 11], expected[5, 10], expected[3, 12] = 0, 0, 0
    _require(np.array_equal(solve(grid), expected), "one-cell opening must deactivate the print")


def self_test():
    _regressions()
    features, signatures, shapes, geometries, palettes, ratios = [], set(), set(), set(), set(), set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        if seed < 3:
            _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), "nondeterminism")
        for split in ("train", "test"):
            for pair in puzzle[split]:
                data = _decode(pair["input"])
                f = _features(data)
                features.append(f)
                shapes.add(f["shape"])
                palettes.add(f["palette"])
                ratios.update(f["ratios"])
                normalized = {}
                signature = tuple((r, c, normalized.setdefault(color, len(normalized)))
                                  for (r, c), color in sorted(data["motif"].items()))
                geometries.add((data["source"][2], signature))
                _require(np.array_equal(solve(pair["input"].tolist()), pair["output"]), "list input solve disagreement")
    _require(len(signatures) == 42, "cross-seed puzzle repetition")
    _require(len(shapes) >= 25 and len(geometries) >= 70 and len(palettes) >= 100, "missing structural/palette variation")
    _require(len(ratios) >= 5 and any(n > d for n, d in ratios) and any(n < d for n, d in ratios), "scale diversity")
    _require({f["active"] for f in features} >= {1, 2, 3}, "window count diversity")
    _require(any(f["irregular"] for f in features) and any(f["unmarked"] for f in features), "stencil diversity")
    _require(sum(f["fine"] for f in features) >= 42, "fine geometry coverage")
    for nt in range(2, 11):
        for nq in range(1, min(4, nt - 1) + 1):
            p = generate_puzzle(917 + 10 * nt + nq, nt, nq)
            _require(len(p["train"]) == nt and len(p["test"]) == nq, "explicit counts not honored")
    for nt, nq in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (3.0, 1), (None, 1)):
        try:
            generate_puzzle(0, nt, nq)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    p = generate_puzzle(0, 2, 1)
    for zone in ("wall", "reference", "printed", "exterior"):
        q = {s: [{k: np.asarray(v).copy() for k, v in pair.items()} for pair in p[s]] for s in p}
        pair = q["train"][0]
        d = _decode(pair["input"])
        mask = {"wall": d["wall_mask"], "reference": d["source_mask"],
                "printed": pair["input"] != pair["output"], "exterior": d["cavity_labels"] < 0}[zone]
        r, c = np.argwhere(mask)[0]
        pair["output"][r, c] = (int(pair["output"][r, c]) + 1) % 10
        try:
            validate_generated_puzzle(q)
        except AssertionError:
            pass
        else:
            raise AssertionError("validator accepted " + zone + " corruption")
    fallback = {"train": [_fallback(np.random.default_rng(11), i, set()) for i in (0, 1)],
                "test": [_fallback(np.random.default_rng(12), None, set())]}
    validate_generated_puzzle(fallback)
    for shift in ("count", "scale", "aperture"):
        pair = _fallback(np.random.default_rng(19), None, set(), test=True, shift=shift)
        _require(np.array_equal(solve(pair["input"]), pair["output"]), "fallback shift rendering")
        f = _features(_decode(pair["input"]))
        _require(f["wall_count"] == 4 if shift == "count" else
                 (3, 2) in f["ratios"] if shift == "scale" else f["irregular"] > 0,
                 "fallback omitted its test recombination")
    return {"seeds": 42, "pairs": len(features), "shapes": len(shapes),
            "motifs": len(geometries), "role_palettes": len(palettes), "ratios": sorted(ratios)}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
