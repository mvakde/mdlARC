"""Registered window census: exact crop addresses control component tallies.

Only Python's standard library and NumPy are used. No parent imports, stored
examples, learned bitmap, seed-dependent solver state, or filesystem access.
"""
from __future__ import annotations

import numpy as np


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _counts(num_train, num_test):
    a, b = _integer(num_train, "num_train"), _integer(num_test, "num_test")
    if not (2 <= a <= 10 and 1 <= b <= 4 and a > b):
        raise ValueError("require 2..10 training, 1..4 test, and num_train > num_test")
    return a, b


def _grid(value):
    a = np.asarray(value)
    _check(a.ndim == 2 and all(1 <= n <= 30 for n in a.shape), "grid dimensions")
    _check(np.issubdtype(a.dtype, np.integer), "integer grid required")
    _check(np.all((0 <= a) & (a <= 9)), "colors outside 0..9")
    return a.astype(np.int8, copy=True)


def _key(a):
    return tuple(a.shape), a.astype(np.int8, copy=False).tobytes()


def _d4(a, t):
    return np.rot90(np.fliplr(a) if t >= 4 else a, t % 4).copy()


def _direction(vector, t):
    r, c = vector
    if t >= 4:
        c = -c
    for _ in range(t % 4):
        r, c = -c, r
    return r, c


def _components(a, excluded):
    visited = np.zeros(a.shape, dtype=bool)
    parts = []
    for r in range(a.shape[0]):
        for c in range(a.shape[1]):
            color = int(a[r, c])
            if visited[r, c] or color in excluded:
                continue
            stack, cells = [(r, c)], []
            visited[r, c] = True
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (0 <= nr < a.shape[0] and 0 <= nc < a.shape[1]
                            and not visited[nr, nc] and int(a[nr, nc]) == color):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            rs, cs = zip(*cells)
            parts.append(dict(color=color, cells=tuple(cells),
                              box=(min(rs), max(rs), min(cs), max(cs))))
    return parts


def _perimeter(box):
    r0, r1, c0, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)
            if r in (r0, r1) or c in (c0, c1)}


def _inside(cell, box):
    r, c = cell
    r0, r1, c0, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _mask(shape, box):
    a = np.zeros(shape, dtype=bool)
    r0, r1, c0, c1 = box
    a[r0:r1 + 1, c0:c1 + 1] = True
    return a


def _atlas(parts, a, bg):
    """Find the one rectangular outline with a two-cell tab and colored tip."""
    found = []
    for part in parts:
        R0, R1, C0, C1 = part["box"]
        for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
            box = (R0 + (2 if dr == -1 else 0), R1 - (2 if dr == 1 else 0),
                   C0 + (2 if dc == -1 else 0), C1 - (2 if dc == 1 else 0))
            r0, r1, c0, c1 = box
            h, w = r1 - r0 + 1, c1 - c0 + 1
            if min(h, w) < 13 or h % 2 == 0 or w % 2 == 0:
                continue
            root = ((r0 + r1) // 2 if not dr else (r1 if dr == 1 else r0),
                    (c0 + c1) // 2 if not dc else (c1 if dc == 1 else c0))
            tab = {(root[0] + i * dr, root[1] + i * dc) for i in (1, 2)}
            tip = root[0] + 3 * dr, root[1] + 3 * dc
            if not (0 <= tip[0] < a.shape[0] and 0 <= tip[1] < a.shape[1]):
                continue
            if (set(part["cells"]) == _perimeter(box) | tab
                    and int(a[tip]) not in (bg, part["color"])):
                found.append(dict(box=box, direction=(dr, dc), tip=tip,
                                  ink=int(a[tip]), border=part["color"],
                                  cells=part["cells"]))
    # A coincidental tab-like landmark structure enclosed by the real atlas
    # is part of its drawing, not a second top-level reference frame.
    found = [f for f in found if not any(f is not g and all(
        _inside(p, g["box"]) for p in (*f["cells"], f["tip"])) for g in found)]
    _check(len(found) == 1, "expected one tabbed atlas")
    return found[0]


def _locations(full, patch):
    h, w = patch.shape
    if h > full.shape[0] or w > full.shape[1]:
        return []
    windows = np.lib.stride_tricks.sliding_window_view(full, (h, w))
    rr, cc = np.nonzero(np.all(windows == patch, axis=(-2, -1)))
    return list(zip(rr.tolist(), cc.tolist()))


def _matches(reference, patch):
    matches = []
    for t in range(8):
        full = _d4(reference, t)
        for swap in (0, 1):
            for r, c in _locations(full, patch if swap == 0 else 1 - patch):
                matches.append((t, swap, r, c))
    return matches


def _original_box(shape, t, r, c, h, w):
    labels = _d4(np.arange(shape[0] * shape[1]).reshape(shape), t)
    ids = labels[r:r + h, c:c + w]
    rows, cols = ids // shape[1], ids % shape[1]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _tally(a, box, direction, count, color):
    r0, r1, c0, c1 = box
    dr, dc = direction
    r = r1 if dr == 1 else r0 if dr == -1 else (r0 + r1) // 2
    c = c1 if dc == 1 else c0 if dc == -1 else (c0 + c1) // 2
    for i in range(1, count + 1):
        a[r - 2 * i * dr, c - 2 * i * dc] = color


def _analyze(input_grid):
    a = _grid(input_grid)
    colors, frequencies = np.unique(a, return_counts=True)
    _check(np.count_nonzero(frequencies == frequencies.max()) == 1, "background tie")
    bg = int(colors[np.argmax(frequencies)])
    parts = _components(a, {bg})
    atlas = _atlas(parts, a, bg)
    r0, r1, c0, c1 = atlas["box"]
    inner = (r0 + 1, r1 - 1, c0 + 1, c1 - 1)
    field = a[r0 + 1:r1, c0 + 1:c1]
    reference = (field == atlas["ink"]).astype(np.int8)
    _check(len({_key(_d4(reference, t) ^ s) for t in range(8) for s in (0, 1)}) == 16,
           "atlas must have sixteen different geometric/role candidates")
    frames = []
    for part in parts:
        box = part["box"]
        h, w = box[1] - box[0] + 1, box[3] - box[2] + 1
        if (min(h, w) in (5, 7) and max(h, w) in (9, 11)
                and set(part["cells"]) == _perimeter(box)
                and not any(_inside(p, atlas["box"]) for p in part["cells"])):
            frames.append(part)
    # A landmark loop inside a key is not another query frame.
    frames = [f for f in frames if not any(f is not g and all(
        _inside(p, g["box"]) for p in f["cells"]) for g in frames)]
    _check(1 <= len(frames) <= 3, "one through three query frames required")
    cat_colors = {f["color"] for f in frames}
    _check(not cat_colors & {bg, atlas["ink"], atlas["border"]}, "category/atlas role collision")
    objects = _components(field, {bg, atlas["ink"]})
    _check(all(1 <= len(o["cells"]) <= 8 for o in objects), "census object area")
    _check(all(o["color"] != atlas["border"] for o in objects), "atlas border color used by an object")
    # Even diagonally, different objects have a one-cell background separation.
    labels = np.full(field.shape, -1, dtype=int)
    for i, o in enumerate(objects):
        for p in o["cells"]:
            labels[p] = i
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        x = labels[max(dr, 0):labels.shape[0] + min(dr, 0),
                   max(dc, 0):labels.shape[1] + min(dc, 0)]
        y = labels[max(-dr, 0):labels.shape[0] + min(-dr, 0),
                   max(-dc, 0):labels.shape[1] + min(-dc, 0)]
        _check(not np.any((x >= 0) & (y >= 0) & (x != y)), "census objects touch")
    queries = []
    for f in frames:
        R0, R1, C0, C1 = f["box"]
        key = a[R0 + 1:R1, C0 + 1:C1]
        palette = np.unique(key)
        _check(len(palette) == 2 and not set(palette.tolist()) & (cat_colors | {atlas["border"]}),
               "keys require two non-category, non-border colors")
        patch = (key == palette[1]).astype(np.int8)
        matches = _matches(reference, patch)
        _check(len(matches) == 1, "key has ambiguous or absent registration")
        t, swap, r, c = matches[0]
        box = _original_box(reference.shape, t, r, c, *patch.shape)
        direction = _direction(atlas["direction"], t)
        _check((direction[0] != 0) == (key.shape[0] > key.shape[1]),
               "transported tab must face an end of the long axis")
        selected = [i for i, o in enumerate(objects) if o["color"] == f["color"]
                    and all(_inside(p, box) for p in o["cells"])]
        count = len(selected)
        capacity = (max(key.shape) - 1) // 2
        _check(count <= capacity, "tally overflows frame")
        queries.append(dict(frame=f, window=box, direction=direction, count=count,
                            area=sum(len(objects[i]["cells"]) for i in selected),
                            selected=tuple(selected), capacity=capacity, transform=t,
                            swap=swap, key_shape=tuple(key.shape)))
    # All census pixels are confined to the atlas; all other content belongs
    # to its outline/tab/tip/landmarks or to one of the disjoint query cards.
    covered = np.zeros(a.shape, dtype=bool)
    covered[r0:r1 + 1, c0:c1 + 1] = True
    for p in atlas["cells"]:
        covered[p] = True
    covered[atlas["tip"]] = True
    for f in frames:
        R0, R1, C0, C1 = f["box"]
        _check(not np.any(covered[R0:R1 + 1, C0:C1 + 1]), "overlapping atlas/cards")
        covered[R0:R1 + 1, C0:C1 + 1] = True
    _check(np.all(a[~covered] == bg), "unclassified cells outside the atlas/cards")
    return dict(grid=a, bg=bg, atlas=atlas, inner=inner, reference=reference,
                objects=objects, queries=queries)


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Recover the complete output from this input alone."""
    d = _analyze(input_grid)
    result = np.full_like(d["grid"], d["bg"])
    for p in d["atlas"]["cells"]:
        result[p] = d["atlas"]["border"]
    result[d["atlas"]["tip"]] = d["atlas"]["ink"]
    r0, r1, c0, c1 = d["inner"]
    result[r0:r1 + 1, c0:c1 + 1] = np.where(d["reference"], d["atlas"]["ink"], d["bg"])
    for q in d["queries"]:
        f = q["frame"]
        for p in f["cells"]:
            result[p] = f["color"]
        _tally(result, f["box"], q["direction"], q["count"], f["color"])
    return result


def _window_bank(reference, shape):
    """Count every exact location under all eight isometries and both role maps."""
    counts = {}
    h, w = shape
    for t in range(8):
        a = _d4(reference, t)
        for r in range(a.shape[0] - h + 1):
            for c in range(a.shape[1] - w + 1):
                patch = a[r:r + h, c:c + w]
                for s in (0, 1):
                    k = (patch ^ s).tobytes()
                    counts[k] = counts.get(k, 0) + 1
    bank = []
    for r in range(reference.shape[0] - h + 1):
        for c in range(reference.shape[1] - w + 1):
            patch = reference[r:r + h, c:c + w]
            if 0 < patch.sum() < patch.size and counts[patch.tobytes()] == 1:
                bank.append((r, r + h - 1, c, c + w - 1))
    return bank


def _reference(rng):
    H, W = int(rng.choice((13, 15))), int(rng.choice((15, 17, 19)))
    for _ in range(80):
        a = np.zeros((H, W), dtype=np.int8)
        # Short/long runs, unequal bends, small loops, stepped paths and combs.
        for j in range(int(rng.integers(5, 9))):
            h, w = int(rng.integers(3, 7)), int(rng.integers(3, 8))
            tile = np.zeros((h, w), dtype=np.int8)
            kind = int(rng.integers(5))
            if kind == 0:
                tile[0, :] = 1
            elif kind == 1:
                tile[:, 0] = 1
                tile[-1, :] = 1
            elif kind == 2:
                tile[0, :] = tile[-1, :] = 1
                tile[:, 0] = tile[:, -1] = 1
            elif kind == 3:
                for r in range(h):
                    c = min(w - 1, r * w // h)
                    tile[r, c] = 1
                    if r:
                        tile[r - 1, min(w - 1, (r - 1) * w // h):c + 1] = 1
            else:
                tile[:, 0] = 1
                for r in range(0, h, 2):
                    tile[r, :int(rng.integers(2, w + 1))] = 1
            tile = _d4(tile, int(rng.integers(8)))
            h, w = tile.shape
            r, c = int(rng.integers(H - h + 1)), int(rng.integers(W - w + 1))
            a[r:r + h, c:c + w] |= tile
        if not .16 <= a.mean() <= .38:
            continue
        banks = {shape: _window_bank(a, shape) for shape in ((5, 7), (5, 9), (3, 7), (3, 9))}
        if min(len(banks[(5, 7)]), len(banks[(3, 7)])) >= 16:
            return a, banks
    raise RuntimeError("could not build a distinctive structured atlas")


def _shape(rng, area):
    cells = {(0, 0)}
    if area <= 6 and rng.random() < .25:
        cells = {(0, i) for i in range(area)}
    else:
        while len(cells) < area:
            frontier = sorted({(r + dr, c + dc) for r, c in cells
                               for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))} - cells)
            # Compact and branched masks, sampled independently of any count.
            candidates = [p for p in frontier if max([r for r, c in cells] + [p[0]])
                          - min([r for r, c in cells] + [p[0]]) < 4
                          and max([c for r, c in cells] + [p[1]])
                          - min([c for r, c in cells] + [p[1]]) < 4]
            cells.add(candidates[int(rng.integers(len(candidates)))])
    for _ in range(int(rng.integers(4))):
        cells = {(c, -r) for r, c in cells}
    if rng.random() < .5:
        cells = {(r, -c) for r, c in cells}
    r0, c0 = min(r for r, c in cells), min(c for r, c in cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _commit(occupied, objects, color, cells):
    objects.append((color, tuple(cells)))
    for r, c in cells:
        occupied[max(0, r - 1):r + 2, max(0, c - 1):c + 2] = True


def _put(rng, reference, occupied, objects, color, area, allowed):
    free = (reference == 0) & ~occupied & allowed
    H, W = free.shape
    for _ in range(50):
        shape = _shape(rng, area)
        h, w = max(r for r, c in shape) + 1, max(c for r, c in shape) + 1
        if h > H or w > W:
            continue
        valid = np.ones((H - h + 1, W - w + 1), dtype=bool)
        for r, c in shape:
            valid &= free[r:r + H - h + 1, c:c + W - w + 1]
        rr, cc = np.nonzero(valid)
        if len(rr):
            i = int(rng.integers(len(rr)))
            cells = tuple((int(rr[i]) + r, int(cc[i]) + c) for r, c in shape)
            _commit(occupied, objects, color, cells)
            return True
    return False


def _crossing(rng, reference, occupied, objects, color, box, other):
    candidates = []
    H, W = reference.shape
    for r in range(H):
        for c in range(W):
            for dr, dc in ((0, 1), (1, 0)):
                cells = tuple((r + i * dr, c + i * dc) for i in range(3))
                if not all(0 <= rr < H and 0 <= cc < W for rr, cc in cells):
                    continue
                if (sum(_inside(p, box) for p in cells) == 2
                        and not all(_inside(p, other) for p in cells)
                        and all(reference[p] == 0 and not occupied[p] for p in cells)):
                    candidates.append(cells)
    if not candidates:
        return False
    _commit(occupied, objects, color, candidates[int(rng.integers(len(candidates)))])
    return True


def _teaching(rng, reference, banks, stage, palette):
    bank = banks[(5, 7)] + banks[(5, 9)]
    color, noise = palette[5], palette[8]
    for _ in range(160):
        ids = rng.choice(len(bank), 2, replace=False)
        boxes = [bank[int(i)] for i in ids]
        masks = [_mask(reference.shape, b) for b in boxes]
        intersect = masks[0] & masks[1]
        if stage == 0 and intersect.any():
            continue
        if stage == 1 and (intersect.sum() < 6 or (masks[0] & ~masks[1]).sum() < 10):
            continue
        occupied = np.zeros(reference.shape, dtype=bool)
        objects = []
        specs = [(4, masks[0]), (2, masks[1]), (2, masks[1])] if stage == 0 else [
            (2, intersect), (4, masks[0] & ~masks[1])]
        if not all(_put(rng, reference, occupied, objects, color, area, allowed)
                   for area, allowed in specs):
            continue
        if stage == 0:
            if not _crossing(rng, reference, occupied, objects, color, boxes[0], boxes[1]):
                continue
            if not _put(rng, reference, occupied, objects, noise, 1, masks[0]):
                continue
        else:
            _put(rng, reference, occupied, objects, noise, int(rng.integers(1, 6)),
                 np.ones(reference.shape, dtype=bool))
        ts = (0, 2) if stage == 0 else (1, 4)
        queries = [(box, color, t, (j + stage) % 2) for j, (box, t) in enumerate(zip(boxes, ts))]
        return objects, queries
    raise RuntimeError("unable to pack the teaching windows")


def _ordinary(rng, reference, banks, is_test, palette):
    n = int(rng.integers(1, 4))
    short = 3 if is_test else 5
    # Three cards can always fit using seven-cell long sides; smaller query
    # sets also use nine-cell strips and their capacity-four tally lines.
    bank = banks[(short, 7)] + ([] if n == 3 else banks[(short, 9)])
    for _ in range(100):
        boxes = [bank[int(i)] for i in rng.choice(len(bank), n, replace=False)]
        masks = [_mask(reference.shape, b) for b in boxes]
        colors = [int(rng.choice(palette[5:8])) for _ in boxes]
        mode = int(rng.integers(4))
        occupied = np.zeros(reference.shape, dtype=bool)
        objects = []
        if mode == 0 and n >= 2:
            colors[1] = colors[0]
            overlap = masks[0] & masks[1]
            if overlap.sum() < 3 or not _put(rng, reference, occupied, objects, colors[0],
                                            int(rng.integers(1, 4)), overlap):
                continue
        elif mode == 1:
            # Intermediate and full counts use independently drawn object sizes.
            cap = (boxes[0][3] - boxes[0][2]) // 2
            target = int(rng.choice((2, cap)))
            if not all(_put(rng, reference, occupied, objects, colors[0],
                            int(rng.integers(1, 3)), masks[0]) for _ in range(target)):
                continue
        elif mode == 3:
            if not _put(rng, reference, occupied, objects, colors[0],
                        int(rng.integers(1, 6)), masks[0]):
                continue
        for _ in range(int(rng.integers(2, 10))):
            color = int(rng.choice(palette[5:9]))
            _put(rng, reference, occupied, objects, color, int(rng.integers(1, 9)),
                 np.ones(reference.shape, dtype=bool))
        counts = [sum(c == color and all(_inside(p, b) for p in cells) for c, cells in objects)
                  for b, color in zip(boxes, colors)]
        if any(count > (b[3] - b[2]) // 2 for count, b in zip(counts, boxes)):
            continue
        if is_test and mode != 2 and not any(counts):
            continue
        queries = [(b, color, int(rng.integers(8)), int(rng.integers(2)))
                   for b, color in zip(boxes, colors)]
        return objects, queries
    raise RuntimeError("unable to pack an ordinary scene")


def _layout(rng, atlas_shape, card_shapes):
    mh, mw = atlas_shape[0] + 2, atlas_shape[1] + 5  # includes tab and tip
    if rng.random() < .5:
        # Scatter cards around an offset atlas, with free pixel placements.
        for _ in range(16):
            H, W = 30, int(rng.integers(max(mw, 26), 31))
            occupied = np.zeros((H, W), dtype=bool)
            master = int(rng.integers(H - mh + 1)), int(rng.integers(W - mw + 1))
            def reserve(r, c, h, w):
                occupied[max(0, r - 1):min(H, r + h + 1),
                         max(0, c - 1):min(W, c + w + 1)] = True
            reserve(*master, mh, mw)
            positions = [None] * len(card_shapes)
            for i in sorted(range(len(card_shapes)), key=lambda j: -np.prod(card_shapes[j])):
                h, w = card_shapes[i]
                sums = np.pad(occupied.astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
                used = sums[h:, w:] - sums[:-h, w:] - sums[h:, :-w] + sums[:-h, :-w]
                rr, cc = np.nonzero(used == 0)
                if not len(rr):
                    break
                j = int(rng.integers(len(rr)))
                positions[i] = int(rr[j]), int(cc[j])
                reserve(*positions[i], h, w)
            if all(p is not None for p in positions):
                return (H, W), master, positions
    # A bounded, roomy row arrangement retains independently transformed cards.
    total = sum(w for h, w in card_shapes) + len(card_shapes) - 1
    W = max(mw, total) + int(rng.integers(0, min(3, 30 - max(mw, total)) + 1))
    top = int(rng.integers(0, 2))
    H = top + mh + 1 + max(h for h, w in card_shapes)
    if H > 30:
        top, H = 0, mh + 1 + max(h for h, w in card_shapes)
    H += int(rng.integers(0, min(2, 30 - H) + 1))
    master = top, int(rng.integers(W - mw + 1))
    c = int(rng.integers(W - total + 1))
    positions = []
    for h, w in card_shapes:
        r = mh + top + 1 + int(rng.integers(H - (mh + top + 1) - h + 1))
        positions.append((r, c))
        c += w + 1
    return (H, W), master, positions


def _draw_border(a, box, color):
    r0, r1, c0, c1 = box
    a[r0, c0:c1 + 1] = a[r1, c0:c1 + 1] = color
    a[r0:r1 + 1, c0] = a[r0:r1 + 1, c1] = color


def _render(rng, reference, objects, queries, palette):
    bg, ink, border, q0, q1 = palette[:5]
    patches = [_d4(reference[b[0]:b[1] + 1, b[2]:b[3] + 1], t)
               for b, color, t, swap in queries]
    shapes = [(p.shape[0] + 2, p.shape[1] + 2) for p in patches]
    shape, (r, c), positions = _layout(rng, reference.shape, shapes)
    inp = np.full(shape, bg, dtype=np.int8)
    out = inp.copy()
    H, W = reference.shape
    master_box = r, r + H + 1, c, c + W + 1
    for a in (inp, out):
        _draw_border(a, master_box, border)
        a[r + 1:r + H + 1, c + 1:c + W + 1] = np.where(reference, ink, bg)
        a[r + (H + 1) // 2, c + W + 2:c + W + 4] = border
        a[r + (H + 1) // 2, c + W + 4] = ink
    for color, cells in objects:
        for rr, cc in cells:
            inp[r + 1 + rr, c + 1 + cc] = color
    for query, patch, (rr, cc), (h, w) in zip(queries, patches, positions, shapes):
        box, color, t, swap = query
        frame_box = rr, rr + h - 1, cc, cc + w - 1
        _draw_border(inp, frame_box, color)
        _draw_border(out, frame_box, color)
        inp[rr + 1:rr + h - 1, cc + 1:cc + w - 1] = np.where(patch ^ swap, q1, q0)
        count = sum(col == color and all(_inside(p, box) for p in cells) for col, cells in objects)
        _tally(out, frame_box, _direction((0, 1), t), count, color)
    g = int(rng.integers(8))
    return dict(input=_d4(inp, g), output=_d4(out, g))


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    nt, nq = _counts(num_train, num_test)
    seed = _integer(seed, "seed")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    rng = np.random.default_rng(seed)
    # Bounded resampling can replace an unsuitable landmark drawing, never
    # alter the solver's convention or relax uniqueness/teaching constraints.
    for _ in range(12):
        reference, banks = _reference(rng)
        palette = [int(c) for c in rng.permutation(10)[:9]]
        puzzle, seen = {"train": [], "test": []}, set()
        try:
            for i in range(nt + nq):
                local_palette = palette[:]
                if i == 1:
                    local_palette[0], local_palette[1] = palette[1], palette[0]
                elif i >= 2:
                    local_palette = [int(c) for c in rng.permutation(10)[:9]]
                for _ in range(12):
                    if i < 2:
                        objects, queries = _teaching(rng, reference, banks, i, local_palette)
                    else:
                        objects, queries = _ordinary(rng, reference, banks, i >= nt, local_palette)
                    pair = _render(rng, reference, objects, queries, local_palette)
                    key = _key(pair["input"])
                    if key not in seen:
                        break
                else:
                    raise RuntimeError("distinct scene attempts exhausted")
                seen.add(key)
                puzzle["train" if i < nt else "test"].append(pair)
            validate_generated_puzzle(puzzle)
            return puzzle
        except RuntimeError:
            continue
    raise RuntimeError("bounded construction attempts exhausted")


def _evidence(details):
    rows = [(q["frame"]["color"], q["count"], q["area"])
            for d in details for q in d["queries"]]
    collision = any(c == C and a == A and n != N for c, n, a in rows for C, N, A in rows)
    independence = any(n == N and a != A for c, n, a in rows for C, N, A in rows)
    localized = crossing = shared = centroid_trap = color_trap = False
    directions = set()
    for d in details:
        for q in d["queries"]:
            matching = [o for o in d["objects"] if o["color"] == q["frame"]["color"]]
            localized |= len(matching) != q["count"]
            crossing |= any(any(_inside(p, q["window"]) for p in o["cells"])
                            and not all(_inside(p, q["window"]) for p in o["cells"])
                            for o in matching)
            centroid_trap |= any(_inside(tuple(np.mean(o["cells"], axis=0)), q["window"])
                                 and not all(_inside(p, q["window"]) for p in o["cells"])
                                 for o in matching)
            color_trap |= any(o["color"] != q["frame"]["color"] and all(
                _inside(p, q["window"]) for p in o["cells"]) for o in d["objects"])
            if 0 < q["count"] < q["capacity"]:
                directions.add(q["direction"])
        shared |= any(set(a["selected"]) & set(b["selected"])
                      for i, a in enumerate(d["queries"]) for b in d["queries"][i + 1:])
    return collision, independence, localized, crossing, shared, directions, centroid_trap, color_trap


def validate_generated_puzzle(puzzle):
    _check(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "puzzle schema")
    _check(all(isinstance(puzzle[s], list) for s in puzzle), "split schema")
    nt, nq = _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, details = set(), {"train": [], "test": []}
    classes = set()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _check(isinstance(pair, dict) and set(pair) == {"input", "output"}, "pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            _check(inp.shape == out.shape and _key(inp) not in seen, "shape change or duplicate input")
            seen.add(_key(inp))
            d = _analyze(inp)
            _check(np.array_equal(solve(inp), out), "wrong census/transport/erasure output")
            _check(not np.array_equal(inp, out), "unchanged example")
            classes.add(min(_key(_d4(d["reference"], t)) for t in range(8)))
            details[split].append(d)
    _check(len(classes) == 1, "atlas landmark reference changes within puzzle")
    collision, independence, localized, crossing, shared, directions, centroid_trap, color_trap = _evidence(details["train"])
    _check(collision and independence, "area/count teaching evidence missing")
    _check(localized and crossing, "registration and whole-containment evidence missing")
    _check(shared, "overlap/reuse teaching evidence missing")
    _check(centroid_trap and color_trap, "centroid/ignore-color shortcuts remain possible")
    _check(len(directions) >= 2, "transported tally-end evidence missing")
    _check({1, 2} <= {q["count"] for d in details["train"] for q in d["queries"]}, "unary progression")
    _check(all(min(q["key_shape"]) == 5 for d in details["train"] for q in d["queries"]),
           "training crop short side")
    _check(all(all(min(q["key_shape"]) == 3 for q in d["queries"])
               for d in details["test"]), "narrower-key test shift missing")
    return True


def _signature(puzzle):
    return tuple((_key(p["input"]), _key(p["output"])) for s in ("train", "test") for p in puzzle[s])


def self_test():
    """Bounded 42-seed semantic/variation checks, plus count and corruption probes."""
    puzzles = [generate_puzzle(seed) for seed in range(42)]
    _check(len({_signature(p) for p in puzzles}) == 42, "seed repetition")
    references, shapes, windows, masks, counts, directions, bgs, palettes = (set() for _ in range(8))
    query_numbers = {"train": set(), "test": set()}
    zero = full = unequal = crossing = 0
    for puzzle in puzzles:
        validate_generated_puzzle(puzzle)
        for split in ("train", "test"):
            for pair in puzzle[split]:
                d = _analyze(pair["input"])
                query_numbers[split].add(len(d["queries"]))
                references.add(min(_key(_d4(d["reference"], t)) for t in range(8)))
                shapes.add(tuple(pair["input"].shape))
                bgs.add(d["bg"])
                palettes.add(tuple(sorted({q["frame"]["color"] for q in d["queries"]})))
                unequal += len({q["key_shape"] for q in d["queries"]}) > 1
                for q in d["queries"]:
                    windows.add((tuple(d["reference"].shape), q["window"]))
                    directions.add(q["direction"])
                    counts.add(q["count"])
                    zero += q["count"] == 0
                    full += q["count"] == q["capacity"]
                for o in d["objects"]:
                    r0, r1, c0, c1 = o["box"]
                    masks.add(tuple(sorted((r - r0, c - c0) for r, c in o["cells"])))
                crossing += _evidence([d])[3]
    _check(len(references) == 42, "atlas content repetition after geometric normalization")
    _check(len(shapes) >= 30 and len(windows) >= 100, "geometry/localization diversity")
    _check(len(masks) >= 40 and max(len(m) for m in masks) == 8, "object geometry diversity")
    _check({0, 1, 2, 3} <= counts and zero >= 20 and full >= 10, "count diversity")
    _check(directions == {(0, 1), (0, -1), (1, 0), (-1, 0)}, "tally directions")
    _check(bgs == set(range(10)) and len(palettes) >= 20, "local palette diversity")
    _check(unequal >= 30 and crossing >= 42, "combined variation")
    _check(all(ns == {1, 2, 3} for ns in query_numbers.values()), "query-count diversity")
    _check(_signature(puzzles[0]) == _signature(generate_puzzle(0)), "determinism")
    # A geometric counterfactual: rotate a valid key in place by 180 degrees.
    # Its counting window is unchanged, but the transported tally end reverses.
    original = puzzles[0]["train"][0]["input"]
    before = _analyze(original)
    f = before["queries"][0]["frame"]
    r0, r1, c0, c1 = f["box"]
    rotated = original.copy()
    rotated[r0 + 1:r1, c0 + 1:c1] = np.rot90(rotated[r0 + 1:r1, c0 + 1:c1], 2)
    after = _analyze(rotated)
    old_q = before["queries"][0]
    new_q = next(q for q in after["queries"] if q["frame"]["box"] == f["box"])
    _check(new_q["window"] == old_q["window"] and new_q["count"] == old_q["count"], "rotation changed census domain")
    _check(new_q["direction"] == tuple(-x for x in old_q["direction"]), "crop rotation did not transport direction")
    _check(not np.array_equal(solve(original), solve(rotated)), "omitting crop orientation leaves answer unchanged")
    # B's size invariance: grow one object without changing any window's
    # whole-containment membership, touching another object, or covering ink.
    grown = None
    ar0, ar1, ac0, ac1 = before["inner"]
    for o in before["objects"]:
        if len(o["cells"]) >= 8:
            continue
        for r, c in o["cells"]:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                cell = r + dr, c + dc
                if not (0 <= cell[0] < before["reference"].shape[0] and 0 <= cell[1] < before["reference"].shape[1]):
                    continue
                pos = ar0 + cell[0], ac0 + cell[1]
                if original[pos] != before["bg"]:
                    continue
                if any(all(_inside(p, q["window"]) for p in o["cells"])
                       and not _inside(cell, q["window"]) for q in before["queries"]):
                    continue
                if any(any(max(abs(cell[0] - p[0]), abs(cell[1] - p[1])) <= 1 for p in other["cells"])
                       for other in before["objects"] if other is not o):
                    continue
                grown = original.copy()
                grown[pos] = o["color"]
                break
            if grown is not None:
                break
        if grown is not None:
            break
    _check(grown is not None and np.array_equal(solve(original), solve(grown)), "object area affected tally")
    for a in range(2, 11):
        for b in range(1, min(4, a - 1) + 1):
            p = generate_puzzle(71, a, b)
            _check(len(p["train"]) == a and len(p["test"]) == b, "explicit counts")
    for a, b in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.5, 1), (None, 1)):
        try:
            generate_puzzle(0, a, b)
        except ValueError:
            pass
        else:
            raise AssertionError("illegal counts accepted")
    bad = {s: [{k: v.copy() for k, v in p.items()} for p in puzzles[0][s]] for s in puzzles[0]}
    bad["test"][0]["output"] = bad["test"][0]["input"].copy()
    try:
        validate_generated_puzzle(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("unerased field/crop accepted")
    return dict(seeds=42, atlas_classes=len(references), canvas_shapes=len(shapes),
                registered_windows=len(windows), object_masks=len(masks), counts=sorted(counts),
                zero_queries=zero, full_queries=full, tally_directions=len(directions),
                query_numbers={s: sorted(ns) for s, ns in query_numbers.items()})


if __name__ == "__main__":
    print(self_test())
