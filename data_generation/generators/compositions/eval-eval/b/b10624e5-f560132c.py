"""Stenciled jigsaw: square packing constrained by scaled reflected ink.

All solving information is visible in a single input. Construction never imports
parent generators or provides an owner grid, seed, or palette to solve().
"""
from itertools import combinations
from math import isqrt

import numpy as np


NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _grid(value):
    a = np.asarray(value)
    _require(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _require(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _require(bool(np.all((a >= 0) & (a <= 9))), "color range")
    return a.astype(np.int16, copy=True)


def _components(mask):
    todo = set(map(tuple, np.argwhere(mask).tolist()))
    result = []
    while todo:
        start = min(todo)
        todo.remove(start)
        stack, cells = [start], {start}
        while stack:
            r, c = stack.pop()
            for dr, dc in NEIGHBORS:
                p = r + dr, c + dc
                if p in todo:
                    todo.remove(p)
                    cells.add(p)
                    stack.append(p)
        result.append(cells)
    return result


def _square(points):
    if not points:
        return None
    rows, cols = zip(*points)
    r, c = min(rows), min(cols)
    side = max(rows) - r + 1
    if max(cols) - c + 1 == side and len(points) == side * side:
        return r, c, side
    return None


def _border(a):
    return np.concatenate((a[0], a[-1], a[1:-1, 0], a[1:-1, -1]))


def _parse(value):
    grid = _grid(value)
    border = _border(grid)
    bg = int(np.bincount(border, minlength=10).argmax())
    _require(np.count_nonzero(border == bg) > len(border) // 2, "readable background")
    comps = _components(grid != bg)
    _require(len(comps) == 4, "four connected tiles required")
    for a, b in combinations(comps, 2):
        _require(all((r + dr, c + dc) not in b for r, c in a
                     for dr in (-1, 0, 1) for dc in (-1, 0, 1)), "tile separation")
    singleton = {c for c in range(10) if np.count_nonzero(grid == c) == 1}
    keys = []
    for r in range(grid.shape[0] - 1):
        for c in range(grid.shape[1] - 1):
            patch = grid[r:r+2, c:c+2]
            values = set(map(int, patch.flat))
            if len(values) == 4 and values <= singleton and bg not in values:
                keys.append((r, c, patch.copy()))
    _require(len(keys) == 1, "unique four-singleton legend required")
    kr, kc, key = keys[0]
    reference = next(i for i, p in enumerate(comps) if (kr, kc) in p)
    comps = [comps[reference]] + [p for i, p in enumerate(comps) if i != reference]
    common = set.intersection(*[{int(grid[p]) for p in comp} for comp in comps])
    _require(len(common) == 2, "one shared material color and one shared registration color")
    anchors = [color for color in common if all(
        _square({p for p in comp if grid[p] == color}) is not None for comp in comps)]
    _require(len(anchors) == 1, "unambiguous square registration marks")
    anchor = anchors[0]
    carrier = next(iter(common - {anchor}))
    key_colors = set(map(int, key.flat))
    _require(not key_colors & {bg, carrier, anchor}, "legend role collision")
    details = {int(grid[p]) for p in comps[0]} - {carrier, anchor} - key_colors
    _require(1 <= len(details) <= 3, "one to three stencil colors required")
    pieces = []
    for i, comp in enumerate(comps):
        rows, cols = zip(*comp)
        r0, c0 = min(rows), min(cols)
        a = np.full((max(rows)-r0+1, max(cols)-c0+1), -1, dtype=np.int16)
        for r, c in comp:
            a[r-r0, c-c0] = grid[r, c]
        allowed = {carrier, anchor} | details | (key_colors if i == 0 else set())
        _require(set(map(int, a[a >= 0])) <= allowed, "unrelated tile marks")
        pieces.append(a)
    a = pieces[0]
    ar, ac, side = _square(set(map(tuple, np.argwhere(a == anchor).tolist())))
    motif = {(int(r)-ar, int(c)-ac): int(a[r, c])
             for r, c in np.argwhere(np.isin(a, [anchor, *details]))}
    ink_mask = np.isin(a, [anchor, *details])
    _require(len(_components(ink_mask)) == 1, "connected reference stencil")
    total = sum(len(p) for p in comps)
    n = isqrt(total)
    _require(n * n == total and n <= 30, "tile area must make a square")
    _require(a[0, 0] >= 0, "reference must occupy top-left corner")
    return dict(grid=grid, bg=bg, carrier=carrier, anchor=anchor, details=details,
                key=key, pieces=pieces, components=comps, motif=motif,
                reference_side=side, side=n)


def _scaled(motif, source_side, dest_side, corner):
    """Exact rational rasterization; -1 represents transparent stencil cells."""
    rs, cs = zip(*motif)
    r0 = min(rs) // source_side * source_side
    c0 = min(cs) // source_side * source_side
    r1 = ((max(rs) + source_side) // source_side) * source_side
    c1 = ((max(cs) + source_side) // source_side) * source_side
    a = np.full((r1-r0, c1-c0), -1, dtype=np.int16)
    for (r, c), color in motif.items():
        a[r-r0, c-c0] = color
    expanded = np.repeat(np.repeat(a, dest_side, axis=0), dest_side, axis=1)
    h, w = expanded.shape[0] // source_side, expanded.shape[1] // source_side
    blocks = expanded.reshape(h, source_side, w, source_side)
    lo, hi = blocks.min(axis=(1, 3)), blocks.max(axis=(1, 3))
    _require(np.array_equal(lo, hi), "nonintegral stencil color boundary")
    offset = [-r0 * dest_side // source_side, -c0 * dest_side // source_side]
    if corner // 2:
        lo = lo[::-1]
        offset[0] = h - offset[0] - dest_side
    if corner % 2:
        lo = lo[:, ::-1]
        offset[1] = w - offset[1] - dest_side
    return lo, tuple(offset)


def _ink_on_piece(data, a, corner, *, reflect=True, scale=True, box=False):
    ar, ac, dest = _square(set(map(tuple, np.argwhere(a == data['anchor']).tolist())))
    raster, offset = _scaled(data['motif'], data['reference_side'],
                             dest if scale else data['reference_side'],
                             corner if reflect else 0)
    # The no-scale counterfactual aligns the top-left of the unchanged anchor.
    r0, c0 = ar - offset[0], ac - offset[1]
    ink = {}
    for r, c in np.argwhere(raster >= 0):
        y, x = int(r) + r0, int(c) + c0
        if 0 <= y < a.shape[0] and 0 <= x < a.shape[1] and (box or a[y, x] >= 0):
            ink[y, x] = int(raster[r, c])
    return ink, int(np.count_nonzero(raster >= 0))


def _candidates(data, piece, corner, marks=True):
    n = data['side']
    result, seen = [], set()
    for k in ([0] if piece == 0 else range(4)):
        a = np.rot90(data['pieces'][piece], -k)
        h, w = a.shape
        if h > n or w > n:
            continue
        r0, c0 = (0 if corner < 2 else n-h), (0 if corner % 2 == 0 else n-w)
        cells = [(int(r)+r0, int(c)+c0) for r, c in np.argwhere(a >= 0)]
        corners = {(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)}
        target = ((0, 0), (0, n-1), (n-1, 0), (n-1, n-1))[corner]
        if set(cells) & corners != {target}:
            continue
        try:
            ink, total = _ink_on_piece(data, a, corner)
        except ValueError:
            continue
        if marks and any(ink.get((int(r), int(c))) != int(a[r, c])
                         for r, c in np.argwhere(np.isin(a, list(data['details'])))):
            continue
        painted = np.full(a.shape, -1, dtype=np.int16)
        painted[a >= 0] = int(data['key'].flat[corner])
        for p, color in ink.items():
            painted[p] = color
        mask = sum(1 << (r*n+c) for r, c in cells)
        sig = (mask, painted.tobytes())
        if sig not in seen:
            seen.add(sig)
            result.append(dict(piece=piece, corner=corner, rotation=k, a=a,
                               offset=(r0, c0), mask=mask, ink=ink,
                               total=total, painted=painted))
    return result


def _assemblies(data, marks=True, limit=2):
    table = {(i, q): _candidates(data, i, q, marks)
             for i in range(4) for q in ([0] if i == 0 else [1, 2, 3])}
    results, seen = [], set()
    full = (1 << (data['side'] ** 2)) - 1

    def visit(q, remaining, occupied, chosen):
        if len(results) >= limit:
            return
        if q == 4:
            if occupied != full:
                return
            out = np.full((data['side'], data['side']), -1, dtype=np.int16)
            for part in chosen:
                r, c = part['offset']
                a = part['painted']
                target = out[r:r+a.shape[0], c:c+a.shape[1]]
                target[a >= 0] = a[a >= 0]
            sig = out.tobytes()
            if sig not in seen:
                seen.add(sig)
                results.append(dict(output=out.astype(np.int8), placements=tuple(chosen)))
            return
        for i in ([0] if q == 0 else remaining):
            for part in table[i, q]:
                if part['mask'] & occupied:
                    continue
                visit(q+1, tuple(j for j in remaining if j != i),
                      occupied | part['mask'], chosen+[part])
                if len(results) >= limit:
                    return

    visit(0, (1, 2, 3), 0, [])
    return results


def solve(input_grid: np.ndarray) -> np.ndarray:
    data = _parse(input_grid)
    answers = _assemblies(data)
    _require(len(answers) == 1, "tiles and surviving ink must determine one decorated square")
    return answers[0]['output'].copy()


def _cut(rng, n, style):
    y = int(rng.integers(3, n-2))
    xt, xb = int(rng.integers(3, n-1)), int(rng.integers(2, n-1))
    a = np.empty((n, n), dtype=np.int8)
    for r in range(n):
        if r < y:
            a[r, :xt], a[r, xt:] = 0, 1
            if style and r >= 1:
                xt = int(np.clip(xt + rng.integers(-1, 2), 2, n-2))
        else:
            a[r, :xb], a[r, xb:] = 2, 3
            if style:
                xb = int(np.clip(xb + rng.integers(-1, 2), 1, n-1))
    if style >= 2:
        corners = {(0, 0), (0, n-1), (n-1, 0), (n-1, n-1)}
        for _ in range(n * (style-1)):
            r, c = int(rng.integers(n)), int(rng.integers(n))
            if (r, c) in corners:
                continue
            old = int(a[r, c])
            adjacent = [int(a[r+dr, c+dc]) for dr, dc in NEIGHBORS
                        if 0 <= r+dr < n and 0 <= c+dc < n and a[r+dr, c+dc] != old]
            if not adjacent:
                continue
            new = int(rng.choice(adjacent))
            a[r, c] = new
            if any(len(_components(a == i)) != 1 for i in (old, new)):
                a[r, c] = old
    if min(np.count_nonzero(a == q) for q in range(4)) < 5:
        return None
    return a


def _motif(rng, anchor, details, unit, fine=False):
    if fine and unit == 2:
        # Sub-anchor details. Every boundary remains exact; incompatible
        # fractional destination scales are rejected rather than rounded.
        cells = {(r, c): anchor for r in range(unit) for c in range(unit)}
        for i in range(int(rng.integers(4, 13))):
            frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in NEIGHBORS
                               if (r+dr, c+dc) not in cells
                               and -3 <= r+dr <= 4 and -3 <= c+dc <= 4})
            p = frontier[int(rng.integers(len(frontier)))]
            cells[p] = details[i % len(details)]
        return cells
    cells = {(0, 0): anchor}
    target = int(rng.integers(3, 9))
    direction = NEIGHBORS[int(rng.integers(4))]
    for i in range(target-1):
        frontier = sorted({(r+dr, c+dc) for r, c in cells for dr, dc in NEIGHBORS
                           if (r+dr, c+dc) not in cells and abs(r+dr) <= 3 and abs(c+dc) <= 3})
        if rng.random() < .45:
            p = max(frontier, key=lambda p: (p[0]*direction[0]+p[1]*direction[1], -abs(p[0])-abs(p[1])))
        else:
            p = frontier[int(rng.integers(len(frontier)))]
        cells[p] = details[i % len(details)]
    return {(r*unit+dr, c*unit+dc): color for (r, c), color in cells.items()
            for dr in range(unit) for dc in range(unit)}


def _locations(mask, s):
    return [(r, c) for r in range(mask.shape[0]-s+1) for c in range(mask.shape[1]-s+1)
            if mask[r:r+s, c:c+s].all()]


def _scatter(rng, pieces):
    # Random shelves, rows/columns when feasible, and unconstrained placement.
    offsets = None
    if rng.random() < .6:
        for _ in range(20):
            transpose = bool(rng.integers(2))
            dims = [a.shape[::-1] if transpose else a.shape for a in pieces]
            order = [int(i) for i in rng.permutation(4)]
            groups, group = [], []
            prob = float(rng.choice([0, .3, .7, 1]))
            for j, i in enumerate(order):
                group.append(i)
                if j == 3 or rng.random() < prob:
                    groups.append(group)
                    group = []
            gap = int(rng.integers(1, 4))
            widths = [sum(dims[i][1] for i in g)+gap*(len(g)-1) for g in groups]
            heights = [max(dims[i][0] for i in g) for g in groups]
            h, w = sum(heights)+gap*(len(groups)-1), max(widths)
            if max(h, w) > 30:
                continue
            offsets, y = [None]*4, 0
            for g, hh, ww in zip(groups, heights, widths):
                x = int(rng.integers(w-ww+1))
                for i in g:
                    yy = y + int(rng.integers(hh-dims[i][0]+1))
                    offsets[i] = (x, yy) if transpose else (yy, x)
                    x += dims[i][1]+gap
                y += hh+gap
            break
    for attempt in range(0 if offsets is not None else 25):
        occupied = np.zeros((30, 30), dtype=bool)
        offsets = [None]*4
        for i in rng.permutation(4):
            a = pieces[int(i)]
            h, w = a.shape
            pts = np.argwhere(a >= 0)
            for _ in range(60):
                r, c = int(rng.integers(31-h)), int(rng.integers(31-w))
                rr, cc = pts[:, 0]+r, pts[:, 1]+c
                if occupied[rr, cc].any():
                    continue
                offsets[int(i)] = (r, c)
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        occupied[np.clip(rr+dr, 0, 29), np.clip(cc+dc, 0, 29)] = True
                break
            else:
                break
        if all(p is not None for p in offsets):
            break
    if offsets is None or any(p is None for p in offsets):
        # Two shelves always fit for supported output sides <=13.
        order = [int(i) for i in rng.permutation(4)]
        offsets = [None]*4
        y = 0
        for group in (order[:2], order[2:]):
            x = 0
            for i in group:
                offsets[i] = (y, x)
                x += pieces[i].shape[1] + 1
            y += max(pieces[i].shape[0] for i in group) + 1
    rmin = min(r for r, c in offsets)
    cmin = min(c for r, c in offsets)
    offsets = [(r-rmin, c-cmin) for r, c in offsets]
    height = max(r+pieces[i].shape[0] for i, (r, c) in enumerate(offsets))
    width = max(c+pieces[i].shape[1] for i, (r, c) in enumerate(offsets))
    pads = []
    for dim in (height, width):
        room = 30-dim
        before = int(rng.integers(min(3, room)+1))
        after = int(rng.integers(min(3, room-before)+1))
        pads.append((before, after))
    offsets = [(r+pads[0][0], c+pads[1][0]) for r, c in offsets]
    return offsets, (height+sum(pads[0]), width+sum(pads[1]))


def _build(rng, n, style, mode, rotations=None, test=False):
    owner = _cut(rng, n, style)
    if owner is None:
        return None
    colors = [int(c) for c in rng.permutation(10)]
    bg, carrier, anchor = colors[:3]
    key = np.array(colors[3:7], dtype=np.int8).reshape(2, 2)
    details = colors[7:7+int(rng.integers(1, 4))]
    unit = int(rng.choice([1, 1, 1, 2]))
    motif = _motif(rng, anchor, details, unit,
                   fine=mode != 'bare' and rng.random() < .4)
    source_positions = []
    for r in range(n):
        for c in range(n):
            ink = {(y+r, x+c): col for (y, x), col in motif.items()}
            if all(0 <= y < n and 0 <= x < n and owner[y, x] == 0 for y, x in ink):
                eligible = owner == 0
                eligible = eligible.copy()
                for p in ink:
                    eligible[p] = False
                keys = _locations(eligible, 2)
                if keys:
                    source_positions.append((r, c, ink, keys))
    if not source_positions:
        return None
    r, c, source_ink, key_positions = source_positions[int(rng.integers(len(source_positions)))]
    kr, kc = key_positions[int(rng.integers(len(key_positions)))]
    full = []
    for q in range(4):
        a = np.full((n, n), -1, dtype=np.int16)
        a[owner == q] = carrier
        full.append(a)
    for p, col in source_ink.items():
        full[0][p] = col
    full[0][kr:kr+2, kc:kc+2] = key
    data = dict(motif=motif, reference_side=unit, anchor=anchor, carrier=carrier,
                details=set(details), key=key, side=n)
    ink_by_corner = [source_ink]
    for q in range(1, 4):
        options = []
        for s in range(1, 4):
            try:
                raster, off = _scaled(motif, unit, s, q)
            except ValueError:
                continue
            for y, x in _locations(owner == q, s):
                ink = {(int(rr)+y-off[0], int(cc)+x-off[1]): int(raster[rr, cc])
                       for rr, cc in np.argwhere(raster >= 0)
                       if 0 <= int(rr)+y-off[0] < n and 0 <= int(cc)+x-off[1] < n
                       and owner[int(rr)+y-off[0], int(cc)+x-off[1]] == q}
                if sum(col != anchor for col in ink.values()) >= 1:
                    options.append((y, x, s, ink))
        if not options:
            return None
        y, x, s, ink = options[int(rng.integers(len(options)))]
        full[q][y:y+s, x:x+s] = anchor
        ink_by_corner.append(ink)
    marked = [int(q) for q in rng.choice([1, 2, 3], 2, replace=False)] if mode != 'bare' else []
    for q in marked:
        for p, col in ink_by_corner[q].items():
            if col != anchor:
                full[q][p] = col
    output = key.flat[owner].astype(np.int8)
    for ink in ink_by_corner:
        for p, col in ink.items():
            output[p] = col
    if rotations is None:
        rotations = [0] + [int(rng.integers(4)) for _ in range(3)]
    pieces = []
    for q, a in enumerate(full):
        pts = np.argwhere(a >= 0)
        lo, hi = pts.min(0), pts.max(0)+1
        pieces.append(np.rot90(a[lo[0]:hi[0], lo[1]:hi[1]], -rotations[q]).copy())
    offsets, shape = _scatter(rng, pieces)
    inp = np.full(shape, bg, dtype=np.int8)
    for i, a in enumerate(pieces):
        r, c = offsets[i]
        target = inp[r:r+a.shape[0], c:c+a.shape[1]]
        target[a >= 0] = a[a >= 0]
    try:
        parsed = _parse(inp)
        answers = _assemblies(parsed)
        if len(answers) != 1 or not np.array_equal(answers[0]['output'], output):
            return None
        ambiguous = len(_assemblies(parsed, marks=False)) > 1
        if mode == 'ambiguous' and not ambiguous:
            return None
        if mode == 'bare' and ambiguous:
            return None
        if any(not np.any(output[owner == q] == key.flat[q]) for q in range(4)):
            return None
        # Remove redundant witnesses, while retaining positive evidence and a
        # unique answer. Blanks are never used as negative evidence by solve.
        points = [(int(r), int(c)) for r, c in np.argwhere(np.isin(inp, details))
                  if (int(r), int(c)) not in {
                      (int(y)+offsets[0][0], int(x)+offsets[0][1])
                      for y, x in np.argwhere(np.isin(pieces[0], details))}]
        original_colors = {p: int(inp[p]) for p in points}
        rng.shuffle(points)
        for p in points:
            color = int(inp[p])
            inp[p] = carrier
            candidate = _parse(inp)
            if len(_assemblies(candidate)) != 1:
                inp[p] = color
        # Some examples retain more evidence than the minimal distinguishing
        # set. This varies visible fragment counts and permits nonessential
        # fragments in otherwise shape-determined assemblies.
        if mode == 'free' and points and rng.random() < .5:
            count = int(rng.integers(1, min(12, len(points))+1))
            for p in points[:count]:
                inp[p] = original_colors[p]
        final = _parse(inp)
        answer = _assemblies(final)[0]
        features = _features(final, answer)
        if mode == 'bare' and not (all(features['reflection_witnesses'])
                                  and features['reflection_options'] == ((1,), (2,), (3,))
                                  and features['scale_witness']
                                  and features['notch_material_loss']
                                  and set(features['rotations']) == {1, 2, 3}):
            return None
        if test and not features['seam_loss']:
            return None
        return dict(input=inp, output=output)
    except (ValueError, StopIteration):
        return None


def _key(a):
    return a.shape, a.tobytes()


def _features(data, answer):
    placements = answer['placements']
    reflection = []
    reflection_options = []
    scaled = False
    local_transfer_diff = 0
    seam_loss = 0
    notch_loss = 0
    notch_material_loss = 0
    notch_color_changes = 0
    clipped = []
    for part in placements:
        q, a = part['corner'], part['a']
        box_ink, _ = _ink_on_piece(data, a, q, box=True)
        missing = set(box_ink) - set(part['ink'])
        notch_loss += len(missing)
        r0, c0 = part['offset']
        notch_material_loss += sum(int(answer['output'][r0+r, c0+c]) in data['key']
                                   for r, c in missing)
        notch_color_changes += sum(answer['output'][r0+r, c0+c] != box_ink[r, c]
                                   for r, c in missing)
        ink = {p: c for p, c in part['ink'].items() if c != data['anchor']}
        raw, _ = _ink_on_piece(data, a, q, reflect=False)
        raw = {p: c for p, c in raw.items() if c != data['anchor']}
        if q:
            reflection.append(ink != raw)
            options = []
            for alternative in range(4):
                other, _ = _ink_on_piece(data, a, alternative)
                other = {p: c for p, c in other.items() if c != data['anchor']}
                if ink == other:
                    options.append(alternative)
            reflection_options.append(tuple(options))
            small, _ = _ink_on_piece(data, a, q, scale=False)
            small = {p: c for p, c in small.items() if c != data['anchor']}
            scaled |= ink != small
            observed = data['pieces'][part['piece']]
            local, _ = _ink_on_piece(data, observed, q)
            layer = np.full(observed.shape, -1, dtype=np.int16)
            for p, color in local.items():
                layer[p] = color
            layer = np.rot90(layer, -part['rotation'])
            moved = {(int(r), int(c)): int(layer[r, c])
                     for r, c in np.argwhere((layer >= 0) & (layer != data['anchor']))}
            local_transfer_diff += moved != ink
        ar, ac, dest = _square(set(map(tuple, np.argwhere(a == data['anchor']).tolist())))
        raster, off = _scaled(data['motif'], data['reference_side'], dest, q)
        dr, dc = part['offset']
        for r, c in np.argwhere(raster >= 0):
            y, x = int(r)+ar-off[0], int(c)+ac-off[1]
            if 0 <= y+dr < data['side'] and 0 <= x+dc < data['side']:
                if not (0 <= y < a.shape[0] and 0 <= x < a.shape[1] and a[y, x] >= 0):
                    seam_loss += 1
        clipped.append(part['total']-len(part['ink']))
    anchors = [_square(set(map(tuple, np.argwhere(a == data['anchor']).tolist())))[2]
               for a in data['pieces']]
    fills = [np.count_nonzero(a >= 0)/a.size for a in data['pieces']]
    areas = [int(np.count_nonzero(a >= 0)) for a in data['pieces']]
    return dict(side=data['side'], canvas=data['grid'].shape, reference_side=data['reference_side'],
                anchor_sides=tuple(anchors), reflection_witnesses=tuple(reflection),
                reflection_options=tuple(reflection_options),
                scale_witness=bool(scaled), seam_loss=seam_loss,
                notch_loss=notch_loss, clipped=tuple(clipped),
                notch_material_loss=notch_material_loss,
                notch_color_changes=int(notch_color_changes),
                local_transfer_diff=local_transfer_diff,
                rotations=tuple(p['rotation'] for p in placements[1:]),
                detail_count=len(data['details']), area_ratio=max(areas)/min(areas),
                min_fill=min(fills), colors=(data['bg'], data['carrier'], data['anchor']),
                key=tuple(map(int, data['key'].flat)),
                fragments=tuple(int(np.count_nonzero(np.isin(a, list(data['details'])))) for a in data['pieces'][1:]))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    for name, value, low, high in [('num_train', num_train, 2, 10), ('num_test', num_test, 1, 4)]:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or not low <= int(value) <= high:
            raise ValueError(f'{name} must be an integer in {low}..{high}')
    if num_train <= num_test:
        raise ValueError('num_train must exceed num_test')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    regime = int(rng.integers(3))
    puzzle, seen = {'train': [], 'test': []}, set()
    for split, count in [('train', num_train), ('test', num_test)]:
        for i in range(count):
            mode = 'bare' if split == 'train' and i == 0 else 'ambiguous' if split == 'train' and i == 1 else 'free'
            for attempt in range(500):
                if regime == 0:
                    n = int(rng.integers(8, 11) if split == 'train' else rng.integers(11, 14))
                elif regime == 1:
                    n = int(rng.integers(11, 14) if split == 'train' else rng.integers(8, 11))
                else:
                    n = int(rng.integers(8, 14))
                # Tests combine the demonstrated ink constraints and clipping
                # with new irregular seams; output size may grow or shrink.
                style = int(rng.integers(1, 4)) if mode == 'bare' or split == 'test' else int(rng.integers(4))
                pair = _build(rng, n, style, mode,
                              [0, 1, 2, 3] if mode == 'bare' else None,
                              test=split == 'test')
                if pair is not None and _key(pair['input']) not in seen:
                    if split == 'train' and i == 1 and np.array_equal(
                            _parse(pair['input'])['key'], _parse(puzzle['train'][0]['input'])['key']):
                        continue
                    puzzle[split].append(pair)
                    seen.add(_key(pair['input']))
                    break
            else:
                raise RuntimeError(f'bounded construction failed: {seed}/{split}/{i}/{mode}')
    return puzzle


def _analyze_pair(pair):
    _require(isinstance(pair, dict) and set(pair) == {'input', 'output'},
             'pair must contain exactly input and output')
    data = _parse(pair['input'])
    out = _grid(pair['output'])
    answers = _assemblies(data)
    _require(len(answers) == 1, 'one decorated square required')
    answer = answers[0]
    _require(np.array_equal(out, answer['output']), 'incorrect decorated assembly')
    features = _features(data, answer)
    features['ambiguous_without_fragments'] = len(_assemblies(data, marks=False)) > 1
    features['bare'] = not any(features['fragments'])
    ratios = []
    for p in answer['placements'][1:]:
        side = _square(set(map(tuple, np.argwhere(p['a'] == data['anchor']).tolist())))[2]
        divisor = int(np.gcd(side, data['reference_side']))
        ratios.append((side // divisor, data['reference_side'] // divisor))
    features['ratios'] = tuple(ratios)
    features['fine_geometry'] = any(
        offset % data['reference_side']
        for (r, c), color in data['motif'].items()
        for dr, dc, offset in ((-1, 0, r), (1, 0, r+1), (0, -1, c), (0, 1, c+1))
        if data['motif'].get((r+dr, c+dc), -1) != color)
    features['cut_signature'] = tuple(p['mask'] for p in answer['placements'])
    color_roles = {data['anchor']: 0}
    features['motif_signature'] = tuple((r, c, color_roles.setdefault(color, len(color_roles)))
                                        for (r, c), color in sorted(data['motif'].items()))
    bounds = []
    for a in data['components']:
        rows, cols = zip(*a)
        bounds.append((min(rows), max(rows)+1, min(cols), max(cols)+1))
    features['row'] = (all(x[3] <= y[2] or y[3] <= x[2] for x, y in combinations(bounds, 2))
                       and max(x[0] for x in bounds) < min(x[1] for x in bounds))
    features['column'] = (all(x[1] <= y[0] or y[1] <= x[0] for x, y in combinations(bounds, 2))
                          and max(x[2] for x in bounds) < min(x[3] for x in bounds))
    for part in answer['placements']:
        material = (part['a'] >= 0).copy()
        for p in part['ink']:
            material[p] = False
        _require(material.any(), 'every legend color needs visible material')
        _require(all(part['painted'][p] == data['key'].flat[part['corner']]
                     for p in map(tuple, np.argwhere(material).tolist())), 'material legend mismatch')
        _require(any(color in data['details'] for color in part['ink'].values()),
                 'every tile needs visible stencil detail')
    return features


def validate_generated_puzzle(puzzle) -> None:
    """Check returned pixels and teaching evidence; no construction metadata."""
    _require(isinstance(puzzle, dict) and set(puzzle) == {'train', 'test'}, 'train/test schema')
    _require(all(isinstance(puzzle[s], list) for s in puzzle), 'splits must be lists')
    tr, te = len(puzzle['train']), len(puzzle['test'])
    _require(2 <= tr <= 10 and 1 <= te <= 4 and tr > te, 'illegal pair counts')
    features, seen = {'train': [], 'test': []}, set()
    for split in puzzle:
        for pair in puzzle[split]:
            f = _analyze_pair(pair)
            signature = _key(_grid(pair['input']))
            _require(signature not in seen, 'duplicate input')
            seen.add(signature)
            features[split].append(f)
    witnesses = [f for f in features['train'] if f['bare'] and f['scale_witness']
                 and f['reflection_options'] == ((1,), (2,), (3,)) and f['notch_material_loss']
                 and set(f['rotations']) == {1, 2, 3} and f['local_transfer_diff']]
    _require(witnesses, 'training must expose scale, reflections, rotation frames and notch clipping')
    _require(any(f['ambiguous_without_fragments'] for f in features['train']),
             'training must show fragments resolving different answers')
    _require(len({f['key'] for f in features['train']}) >= 2, 'training must vary the visible legend')
    _require(all(f['seam_loss'] for f in features['test']), 'tests must recombine seam clipping')


def _puzzle_key(puzzle):
    return tuple(tuple((_key(np.asarray(p['input'])), _key(np.asarray(p['output'])))
                       for p in puzzle[s]) for s in ('train', 'test'))


def self_test():
    """Bounded 42-seed checks, observable diversity, and negative regressions.

    Returns descriptive measurements. These checks use this module's solver;
    they are not independent semantic approval.
    """
    rows, signatures = [], set()
    for seed in range(42):
        puzzle = generate_puzzle(seed)
        validate_generated_puzzle(puzzle)
        signatures.add(_puzzle_key(puzzle))
        if seed < 3:
            _require(_puzzle_key(puzzle) == _puzzle_key(generate_puzzle(seed)), 'determinism')
        local = []
        for split in ('train', 'test'):
            for pair in puzzle[split]:
                f = _analyze_pair(pair)
                f['split'] = split
                rows.append(f)
                local.append(f)
                _require(np.array_equal(solve(pair['input']), pair['output']), 'solve disagreement')
        _require(len({f['cut_signature'] for f in local}) >= 2, 'cuts collapsed within puzzle')
        _require(len({f['motif_signature'] for f in local}) >= 2, 'stencils collapsed within puzzle')
    _require(len(signatures) == 42, 'whole-puzzle variation')
    _require(len({f['canvas'] for f in rows}) >= 35, 'canvas variation')
    _require(len({f['cut_signature'] for f in rows}) >= 100, 'cut variation')
    _require(len({f['motif_signature'] for f in rows}) >= 60, 'stencil variation')
    _require({f['reference_side'] for f in rows} == {1, 2}, 'source-scale variation')
    _require({s for f in rows for s in f['anchor_sides']} == {1, 2, 3}, 'mark-size variation')
    _require({f['detail_count'] for f in rows} == {1, 2, 3}, 'ink-color-count variation')
    _require(len({f['colors'][0] for f in rows}) == 10, 'background palette variation')
    _require(any(f['fine_geometry'] for f in rows), 'sub-anchor geometry missing')
    _require(any(f['area_ratio'] >= 4 for f in rows), 'unequal pieces missing')
    _require(any(f['min_fill'] <= .6 for f in rows), 'concave pieces missing')
    _require(any(f['row'] for f in rows) and any(f['column'] for f in rows), 'linear scatter missing')
    _require(any(not f['bare'] and not f['ambiguous_without_fragments'] for f in rows),
             'optional fragment cases missing')
    for tr, te in ((2, 1), (10, 1), (10, 4)):
        validate_generated_puzzle(generate_puzzle(73, tr, te))
    for tr, te in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (2.0, 1), (None, 1)):
        try:
            generate_puzzle(0, tr, te)
        except ValueError:
            continue
        raise AssertionError('illegal counts accepted')
    # Verify rational rendering itself: a fine pixel cannot be rounded into
    # a 3/2 copy, while a uniform block has an exact 3/2 expansion.
    coarse = {(r, c): 1 for r in range(2) for c in range(2)}
    coarse.update({(r, c): 2 for r in range(2) for c in range(2, 4)})
    _scaled(coarse, 2, 3, 1)
    fine = dict(coarse)
    fine[-1, 0] = 3
    _scaled(fine, 2, 2, 2)
    try:
        _scaled(fine, 2, 3, 1)
    except ValueError:
        pass
    else:
        raise AssertionError('ambiguous fractional boundary accepted')
    damaged = generate_puzzle(91, 2, 1)
    damaged['test'][0]['output'][0, 0] = (int(damaged['test'][0]['output'][0, 0])+1) % 10
    try:
        validate_generated_puzzle(damaged)
    except ValueError:
        pass
    else:
        raise AssertionError('corrupted output accepted')
    return {'seeds': 42, 'pairs': len(rows), 'distinct_puzzles': len(signatures),
            'canvas_shapes': len({f['canvas'] for f in rows}),
            'cut_structures': len({f['cut_signature'] for f in rows}),
            'stencil_structures': len({f['motif_signature'] for f in rows}),
            'fine_geometry_pairs': sum(f['fine_geometry'] for f in rows),
            'fragment_dependent_pairs': sum(f['ambiguous_without_fragments'] for f in rows),
            'notch_clipped_pairs': sum(f['notch_loss'] > 0 for f in rows),
            'row_layouts': sum(f['row'] for f in rows),
            'column_layouts': sum(f['column'] for f in rows)}
