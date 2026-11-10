"""Seam switches: shape assembly controls a scaffold-registered involution.

Only standard-library modules and NumPy are used. No parent data, labels, random
state, or generation metadata are consulted by solve(). No files are written.
"""
from __future__ import annotations

import hashlib
import itertools
import math

import numpy as np


_STEPS = ((-1, 0), (0, -1), (0, 1), (1, 0))


def _need(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


def _counts(num_train, num_test):
    _need(_integer(num_train) and 2 <= num_train <= 10, "num_train must be an integer in 2..10")
    _need(_integer(num_test) and 1 <= num_test <= 4, "num_test must be an integer in 1..4")
    _need(num_train > num_test, "num_train must exceed num_test")
    return int(num_train), int(num_test)


def _grid(value):
    g = np.asarray(value)
    _need(g.ndim == 2 and all(1 <= n <= 30 for n in g.shape), "invalid grid dimensions")
    _need(np.issubdtype(g.dtype, np.integer), "grid must be integral")
    _need(bool(np.all((g >= 0) & (g <= 9))), "invalid grid color")
    return g.astype(np.int8, copy=True)


def _groups(mask, diagonal=False):
    unseen = set(zip(*np.nonzero(mask)))
    steps = tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1) if y or x) if diagonal else _STEPS
    result = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack, cells = [start], [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in steps:
                q = y + dy, x + dx
                if q in unseen:
                    unseen.remove(q)
                    stack.append(q)
                    cells.append(q)
        result.append(np.array(cells, dtype=int))
    return result


def _box(points):
    a = np.asarray(list(points), dtype=int)
    lo, hi = a.min(axis=0), a.max(axis=0) + 1
    return int(lo[0]), int(lo[1]), int(hi[0] - lo[0]), int(hi[1] - lo[1])


def _bits(points, side):
    result = 0
    for y, x in points:
        result |= 1 << (int(y) * side + int(x))
    return result


def _shape(points):
    r, c, h, w = _box(points)
    mask = np.zeros((h, w), dtype=bool)
    a = np.asarray(points)
    mask[a[:, 0] - r, a[:, 1] - c] = True
    return mask


def _packing(shapes, side, limit=2):
    """All corner-constrained placements; rotations retain texture identity.

    Unlike a footprint-only deduplication, symmetric rotations are distinct:
    they might carry different fragments. Accepted scenes have a single result.
    """
    if len(shapes) != 4 or sum(int(s.sum()) for s in shapes) != side * side:
        return []
    corners = ((0, 0), (0, side - 1), (side - 1, 0), (side - 1, side - 1))
    table = {}
    for p, shape in enumerate(shapes):
        for corner in range(4):
            candidates = []
            for k in (range(1) if p == 0 else range(4)):
                if p == 0 and corner != 0:
                    continue
                rotated = np.rot90(shape, k)
                h, w = rotated.shape
                if h > side or w > side:
                    continue
                y = 0 if corner < 2 else side - h
                x = 0 if corner % 2 == 0 else side - w
                pts = np.argwhere(rotated) + (y, x)
                occupied = set(map(tuple, pts))
                if [q in occupied for q in corners] != [j == corner for j in range(4)]:
                    continue
                candidates.append((_bits(pts, side), p, k, y, x))
            table[p, corner] = candidates
    answers = []
    full = (1 << (side * side)) - 1

    def visit(corner, remaining, used, placements):
        if len(answers) >= limit:
            return
        if corner == 4:
            if used == full:
                answers.append(tuple(placements))
            return
        for p in remaining:
            for candidate in table[p, corner]:
                mask = candidate[0]
                if not used & mask:
                    visit(corner + 1, tuple(q for q in remaining if q != p), used | mask,
                          placements + [candidate])
                    if len(answers) >= limit:
                        return

    for candidate in table[0, 0]:
        visit(1, (1, 2, 3), candidate[0], [candidate])
    return answers


def _border(g):
    return np.concatenate((g[0], g[-1], g[1:-1, 0], g[1:-1, -1]))


def _assemble(g):
    border = _border(g)
    bg = int(np.bincount(border, minlength=10).argmax())
    _need(np.count_nonzero(border == bg) * 2 > len(border), "ambiguous border background")
    pieces = _groups(g != bg)
    _need(len(pieces) == 4, "expected four plates")
    for a, b in itertools.combinations(pieces, 2):
        _need(int(np.max(np.abs(a[:, None] - b[None, :]), axis=2).min()) >= 2,
              "plates must have a background gap")
    frequencies = np.bincount(g.ravel(), minlength=10)
    singleton = np.flatnonzero(frequencies == 1)
    _need(len(singleton) == 4 and bg not in singleton, "expected four singleton key colors")
    key_points = np.argwhere(np.isin(g, singleton))
    r, c, h, w = _box(key_points)
    _need((h, w) == (2, 2), "key must be a 2 by 2 block")
    key = g[r:r + 2, c:c + 2].copy()
    anchor_indices = [i for i, pts in enumerate(pieces)
                      if set(map(tuple, key_points)) <= set(map(tuple, pts))]
    _need(len(anchor_indices) == 1, "key must belong to one plate")
    anchor = anchor_indices[0]
    pieces = [pieces[anchor]] + [p for i, p in enumerate(pieces) if i != anchor]
    remaining_colors = [color for color in range(10) if color != bg and color not in singleton]
    carrier = max(remaining_colors, key=lambda color: int(frequencies[color]))
    _need(all(frequencies[carrier] > frequencies[color] for color in remaining_colors if color != carrier),
          "temporary carrier must be the most frequent foreground color")
    shapes, patches = [], []
    for pts in pieces:
        y, x, ph, pw = _box(pts)
        mask = _shape(pts)
        patch = g[y:y + ph, x:x + pw].copy()
        patch[np.isin(patch, singleton)] = carrier
        shapes.append(mask)
        patches.append(patch)
    side = math.isqrt(sum(map(len, pieces)))
    _need(side * side == sum(map(len, pieces)), "total area must be square")
    solutions = _packing(shapes, side)
    _need(len(solutions) == 1, "packing including fragment orientations must be unique")
    assembled = np.full((side, side), carrier, dtype=np.int8)
    owner = np.full((side, side), -1, dtype=np.int8)
    for corner, (_, p, k, y, x) in enumerate(solutions[0]):
        mask = np.rot90(shapes[p], k)
        patch = np.rot90(patches[p], k)
        rr, cc = np.nonzero(mask)
        assembled[rr + y, cc + x] = patch[rr, cc]
        owner[rr + y, cc + x] = corner
    _need(bool(np.all(owner >= 0)), "assembly has a gap")
    return {"grid": assembled, "owner": owner, "key": key, "carrier": carrier,
            "background": bg, "solution": solutions[0], "shapes": shapes,
            "input_piece_order": tuple(int(item[1]) for item in solutions[0])}


def _primitive(crop):
    lengths = []
    for view in (crop, crop.T):
        for row in view:
            cuts = np.r_[0, np.flatnonzero(row[1:] != row[:-1]) + 1, len(row)]
            lengths.extend(int(v) for v in np.diff(cuts))
    scale = int(np.gcd.reduce(lengths))
    small = crop[::scale, ::scale]
    _need(np.array_equal(crop, np.repeat(np.repeat(small, scale, 0), scale, 1)), "nonuniform scale")
    return scale, small


def _signature(mapping):
    return tuple(sorted((y, x, int(color)) for (y, x), color in mapping.items()))


def _read_motifs(g, carrier):
    objects, signatures = [], set()
    for pts in _groups(g != carrier, diagonal=True):
        y, x, h, w = _box(pts)
        crop = g[y:y + h, x:x + w]
        _need(np.count_nonzero(crop != carrier) == len(pts), "interleaved motifs")
        scale, small = _primitive(crop)
        mapping = {(int(r), int(c)): int(small[r, c]) for r, c in np.argwhere(small != carrier)}
        sig = _signature(mapping)
        signatures.add(sig)
        objects.append({"box": (y, x), "scale": scale, "signature": sig})
    _need(len(signatures) == 2, "expected exactly two primitive ornament states")
    sigs = sorted(signatures)
    _need({o["signature"] for o in objects if o["scale"] == 1} == signatures,
          "both states need unit references")
    raw = [{(y, x): c for y, x, c in sig} for sig in sigs]
    candidates = {}
    for color in set(raw[0].values()) & set(raw[1].values()):
        a = {p for p, c in raw[0].items() if c == color}
        b = {p for p, c in raw[1].items() if c == color}
        ay, ax, _, _ = _box(a)
        by, bx, _, _ = _box(b)
        shift = ay - by, ax - bx
        if a == {(y + shift[0], x + shift[1]) for y, x in b}:
            moved = {(y + shift[0], x + shift[1]): c for (y, x), c in raw[1].items()}
            same = {p for p, c in raw[0].items() if moved.get(p) == c}
            candidates[shift] = (len(same), moved, same)
    _need(bool(candidates), "no invariant scaffold registration")
    best = max(v[0] for v in candidates.values())
    winners = [(shift, v) for shift, v in candidates.items() if v[0] == best]
    _need(len(winners) == 1 and best >= 5, "ambiguous or undersized invariant scaffold")
    shift, (_, moved, same) = winners[0]
    oy, ox, _, _ = _box(same)
    templates = [{(y - oy, x - ox): c for (y, x), c in m.items()} for m in (raw[0], moved)]
    common = {(y - oy, x - ox) for y, x in same}
    union = set(templates[0]) | set(templates[1])
    _need(set(templates[0]) != set(templates[1]), "states must relocate a decoration")
    _need(any(templates[0][p] != templates[1][p] for p in set(templates[0]) & set(templates[1])),
          "states must recolor their interior")
    rectangles = []
    uy, ux, uh, uw = _box(union)
    for obj in objects:
        state = sigs.index(obj["signature"])
        dy, dx = shift if state else (0, 0)
        s = obj["scale"]
        anchor = obj["box"][0] + (oy - dy) * s, obj["box"][1] + (ox - dx) * s
        rect = (anchor[0] + uy * s, anchor[1] + ux * s, uh * s, uw * s)
        y, x, h, w = rect
        _need(0 <= y and 0 <= x and y + h <= g.shape[0] and x + w <= g.shape[1],
              "replacement clips the square")
        _need(all(_separated(rect, other) for other in rectangles), "replacement rectangles collide")
        rectangles.append(rect)
        obj.update(state=state, anchor=anchor)
    return templates, common, objects


def _separated(a, b):
    y, x, h, w = a
    r, c, ph, pw = b
    return y + h < r or r + ph < y or x + w < c or c + pw < x


def _pixels(points, anchor, scale):
    return [(anchor[0] + y * scale + dy, anchor[1] + x * scale + dx)
            for y, x in points for dy in range(scale) for dx in range(scale)]


def _owners(owner, points, anchor, scale):
    return {int(owner[y, x]) for y, x in _pixels(points, anchor, scale)}


def _paint(g, template, anchor, scale):
    for (y, x), color in template.items():
        yy, xx = anchor[0] + y * scale, anchor[1] + x * scale
        g[yy:yy + scale, xx:xx + scale] = color


def _analyze(input_grid):
    g = _grid(input_grid)
    a = _assemble(g)
    templates, common, objects = _read_motifs(a["grid"], a["carrier"])
    substrate = a["key"].ravel()[a["owner"]]
    out = substrate.copy()
    for obj in objects:
        n = len(_owners(a["owner"], common, obj["anchor"], obj["scale"]))
        occupied_owners = _owners(a["owner"], templates[obj["state"]], obj["anchor"], obj["scale"])
        obj.update(owners=n, switch=n > 1, decoration_only=n == 1 and len(occupied_owners) > 1)
        _paint(out, templates[obj["state"] ^ (n > 1)], obj["anchor"], obj["scale"])
    a.update(templates=templates, common=common, objects=objects, substrate=substrate, output=out)
    return a


def solve(input_grid: np.ndarray) -> np.ndarray:
    """Derive the answer from visible geometry, key, and motif pixels only."""
    return _analyze(input_grid)["output"].copy()


def _partition(rng, side):
    cut = int(rng.integers(4, side - 3))
    top, bottom = int(rng.integers(4, side - 3)), int(rng.integers(4, side - 3))
    owner = np.empty((side, side), dtype=np.int8)
    for y in range(side):
        if y < cut:
            if rng.random() < .6:
                top = int(np.clip(top + rng.integers(-1, 2), 3, side - 3))
            owner[y, :top], owner[y, top:] = 0, 1
        else:
            if rng.random() < .6:
                bottom = int(np.clip(bottom + rng.integers(-1, 2), 3, side - 3))
            owner[y, :bottom], owner[y, bottom:] = 2, 3
    for _ in range(int(rng.integers(side, 3 * side))):
        y, x = int(rng.integers(1, side - 1)), int(rng.integers(1, side - 1))
        dy, dx = _STEPS[int(rng.integers(4))]
        old, new = int(owner[y, x]), int(owner[y + dy, x + dx])
        if old == new:
            continue
        owner[y, x] = new
        if any(len(_groups(owner == label)) != 1 for label in (old, new)):
            owner[y, x] = old
    shapes = [_shape(np.argwhere(owner == p)) for p in range(4)]
    if any(int(s.sum()) < 10 for s in shapes):
        return None
    solutions = _packing(shapes, side)
    if len(solutions) != 1:
        return None
    expected = tuple(_bits(np.argwhere(owner == p), side) for p in range(4))
    if tuple(item[0] for item in solutions[0]) != expected:
        return None
    return owner


def _design(rng, compact):
    h, w = (3, 3) if compact else (int(rng.integers(3, 5)), int(rng.integers(3, 5)))
    core = {(y, x) for y in range(h) for x in range(w)
            if y in (0, h - 1) or x in (0, w - 1)}
    inside = {(y, x) for y in range(1, h - 1) for x in range(1, w - 1)}
    protect = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1), (0, 1), (1, w - 1)}
    removable = sorted(core - protect)
    rng.shuffle(removable)
    for p in removable[:int(rng.integers(0, min(2, len(removable)) + 1))]:
        trial = core - {p}
        mask = np.zeros((h, w), dtype=bool)
        for q in trial | inside:
            mask[q] = True
        if len(_groups(mask, diagonal=True)) == 1:
            core = trial
    stable = bool(rng.integers(2))
    common = {p: 2 for p in core}
    if stable:
        common[sorted(core)[int(rng.integers(len(core)))]] = 5
    a_patch = {(-1, 1)}
    b_patch = {(1, w)}
    if rng.random() < .5:
        a_patch.add((-1, 0 if rng.random() < .5 else 2))
    if rng.random() < .5:
        b_patch.add((0 if rng.random() < .5 else 2, w))
    templates = []
    for state, patch in enumerate((a_patch, b_patch)):
        t = dict(common)
        t.update({p: 3 + state for p in inside | patch})
        templates.append(t)
    # Rotate/reflection of the complete vocabulary, never independent motifs.
    k, reflect = int(rng.integers(4)), bool(rng.integers(2))

    def point(p):
        y, x = p
        for _ in range(k):
            y, x = -x, y
        return y, -x if reflect else x

    transformed = [{point(p): c for p, c in t.items()} for t in templates]
    union = set(transformed[0]) | set(transformed[1])
    y, x, _, _ = _box(union)
    transformed = [{(r - y, c - x): color for (r, c), color in t.items()} for t in transformed]
    common = {p for p, c in transformed[0].items() if transformed[1].get(p) == c}
    return transformed, common, stable


def _place(rng, owner, templates, common, requests):
    """Bounded exact rectangle placement, categories derived from geometry."""
    side = len(owner)
    union = set(templates[0]) | set(templates[1])
    _, _, uh, uw = _box(union)
    domains = []
    for scale, state, category in requests:
        h, w = uh * scale, uw * scale
        candidates = []
        for y in range(side - h + 1):
            for x in range(side - w + 1):
                owners = _owners(owner, common, (y, x), scale)
                n = len(owners)
                accept = category == "any"
                if category == "whole":
                    accept = np.all(owner[y:y + h, x:x + w] == owner[y, x])
                elif category == "two":
                    accept = n == 2
                elif category == "three":
                    accept = n == 3
                elif category == "multi":
                    accept = n >= 3
                elif category == "seam":
                    accept = n >= 2
                elif category == "decoration":
                    accept = n == 1 and len(_owners(owner, templates[state], (y, x), scale)) > 1
                if accept:
                    rect = _bits(((r, c) for r in range(y, y + h) for c in range(x, x + w)), side)
                    halo = _bits(((r, c) for r in range(max(0, y - 1), min(side, y + h + 1))
                                  for c in range(max(0, x - 1), min(side, x + w + 1))), side)
                    candidates.append((y, x, rect, halo))
        if not candidates:
            return None
        rng.shuffle(candidates)
        domains.append(candidates)
    order = sorted(range(len(requests)), key=lambda i: (len(domains[i]), -requests[i][0]))
    result, nodes = [None] * len(requests), 0

    def visit(depth, used):
        nonlocal nodes
        nodes += 1
        if nodes > 3500:
            return False
        if depth == len(order):
            return True
        i = order[depth]
        for y, x, rect, halo in domains[i]:
            if not used & halo:
                result[i] = (y, x, requests[i][0], requests[i][1])
                if visit(depth + 1, used | rect):
                    return True
        return False

    return result if visit(0, 0) else None


def _scatter(rng, patches, masks, bg):
    # Enumerate shelf groupings along both axes; a two-by-two fallback always fits.
    options = []
    for transpose in (False, True):
        dims = [p.shape[::-1] if transpose else p.shape for p in patches]
        for perm in itertools.permutations(range(4)):
            for cuts in range(8):
                rows, row = [], []
                for j, p in enumerate(perm):
                    row.append(p)
                    if j == 3 or (cuts >> j) & 1:
                        rows.append(row)
                        row = []
                widths = [sum(dims[i][1] for i in row) + len(row) - 1 for row in rows]
                heights = [max(dims[i][0] for i in row) for row in rows]
                h, w = sum(heights) + len(rows) - 1, max(widths)
                if max(h, w) <= 29:
                    options.append((transpose, dims, rows, widths, heights, h, w))
    _need(bool(options), "scatter cannot fit")
    transpose, dims, rows, widths, heights, h, w = options[int(rng.integers(len(options)))]
    offsets, y = [None] * 4, 0
    for row, width, height in zip(rows, widths, heights):
        x = int(rng.integers(w - width + 1))
        for p in row:
            r = y + int(rng.integers(height - dims[p][0] + 1))
            offsets[p] = (x, r) if transpose else (r, x)
            x += dims[p][1] + 1
        y += height + 1
    if transpose:
        h, w = w, h
    # Modest independent padding, including fully inset and edge-contact scenes.
    pads = []
    for extent in (h, w):
        extra = int(rng.integers(1, min(6, 30 - extent) + 1))
        before = int(rng.integers(extra + 1))
        pads.append((before, extra - before))
    offsets = [(y + pads[0][0], x + pads[1][0]) for y, x in offsets]
    canvas = np.full((h + sum(pads[0]), w + sum(pads[1])), bg, dtype=np.int8)

    def paint(canvas, offsets):
        for patch, mask, (y, x) in zip(patches, masks, offsets):
            rr, cc = np.nonzero(mask)
            canvas[rr + y, cc + x] = patch[rr, cc]
        return canvas

    canvas = paint(canvas, offsets)
    if np.count_nonzero(_border(canvas) == bg) * 2 > len(_border(canvas)):
        return canvas
    # Top and left blank borders ensure a strict majority, even at extent 29.
    mins = np.min(offsets, axis=0)
    offsets = [(y - int(mins[0]) + 1, x - int(mins[1]) + 1) for y, x in offsets]
    return paint(np.full((30, 30), bg, dtype=np.int8), offsets)


def _make_pair(rng, stage, index):
    # The first two demonstrations are capabilities, not hidden rule parameters.
    compact = stage != "later" or rng.random() < .55
    templates, common, stable = _design(rng, compact)
    if stage == "first":
        side = 14
        requests = [(1, 0, "whole"), (1, 1, "whole"), (1, 0, "three"),
                    (1, 1, "two"), (1, int(rng.integers(2)), "decoration")]
    elif stage == "second":
        side = 14
        state = int(rng.integers(2))
        requests = [(1, 0, "whole"), (1, 1, "whole"), (2, state, "two"), (1, 1 - state, "seam")]
    elif stage == "test":
        side = 14
        state = int(rng.integers(2))
        requests = [(1, 0, "whole"), (1, 1, "whole"), (2, state, "multi"),
                    (1, 1 - state, "seam"), (1, int(rng.integers(2)), "any")]
    else:
        side = int(rng.integers(12, 15))
        requests = [(1, 0, "whole"), (1, 1, "whole"), (1, 0, "seam"), (1, 1, "seam")]
        if compact and side == 14:
            for _ in range(int(rng.integers(3))):
                requests.append((1, int(rng.integers(2)), "any"))
    for _ in range(150):
        owner = _partition(rng, side)
        if owner is None:
            continue
        placements = _place(rng, owner, templates, common, requests)
        if placements is None:
            continue
        reserved = np.zeros((side, side), dtype=bool)
        for y, x, scale, state in placements:
            pts = _pixels(set(templates[0]) | set(templates[1]), (y, x), scale)
            for p in pts:
                reserved[p] = True
        keys = [(y, x) for y in range(side - 1) for x in range(side - 1)
                if np.all(owner[y:y + 2, x:x + 2] == 0) and not reserved[y:y + 2, x:x + 2].any()]
        if not keys:
            continue
        key_pos = keys[int(rng.integers(len(keys)))]
        palette = rng.permutation(10)
        bg, carrier = int(palette[0]), int(palette[1])
        colored = [{p: int(palette[c]) for p, c in t.items()} for t in templates]
        key = palette[6:10].reshape(2, 2).astype(np.int8)
        assembled = np.full((side, side), carrier, dtype=np.int8)
        output = key.ravel()[owner].copy()
        for y, x, scale, state in placements:
            switch = len(_owners(owner, common, (y, x), scale)) > 1
            _paint(assembled, colored[state], (y, x), scale)
            _paint(output, colored[state ^ switch], (y, x), scale)
        y, x = key_pos
        assembled[y:y + 2, x:x + 2] = key
        foreground_counts = np.bincount(assembled.ravel(), minlength=10)
        if any(foreground_counts[carrier] <= foreground_counts[c] for c in range(10) if c != carrier):
            continue
        patches, masks = [], []
        turns = [0] + ([int(k) for k in rng.permutation([1, 2, 3])] if stage == "first"
                       else [int(rng.integers(4)) for _ in range(3)])
        for p in range(4):
            r, c, h, w = _box(np.argwhere(owner == p))
            mask = owner[r:r + h, c:c + w] == p
            patch = np.full((h, w), bg, dtype=np.int8)
            source = assembled[r:r + h, c:c + w]
            patch[mask] = source[mask]
            k = turns[p]
            patches.append(np.rot90(patch, k).copy())
            masks.append(np.rot90(mask, k).copy())
        return {"input": _scatter(rng, patches, masks, bg), "output": output}
    raise RuntimeError(f"bounded scene construction exhausted at {stage}/{index}")


def generate_puzzle(seed: int, num_train: int = 4, num_test: int = 1) -> dict:
    _need(_integer(seed) and seed >= 0, "seed must be a nonnegative integer")
    nt, ns = _counts(num_train, num_test)
    rng = np.random.default_rng(seed)
    puzzle, seen = {"train": [], "test": []}, set()
    for split, count in (("train", nt), ("test", ns)):
        for i in range(count):
            stage = "test" if split == "test" else "first" if i == 0 else "second" if i == 1 else "later"
            for _ in range(8):
                pair = _make_pair(rng, stage, i)
                signature = pair["input"].shape, pair["input"].tobytes()
                if signature not in seen:
                    seen.add(signature)
                    puzzle[split].append(pair)
                    break
            else:
                raise RuntimeError("distinct input attempts exhausted")
    validate_generated_puzzle(puzzle)
    return puzzle


def _counterfactual(a, mode):
    out = (np.full_like(a["grid"], a["carrier"]) if mode == "omit_key" else a["substrate"].copy())
    for obj in a["objects"]:
        switch = obj["switch"]
        if mode == "omit_exchange":
            switch = False
        elif mode == "all":
            switch = True
        elif mode == "any_pixel":
            switch = switch or obj["decoration_only"]
        elif mode == "per_extra_plate":
            switch = (obj["owners"] - 1) % 2 == 1
        _paint(out, a["templates"][obj["state"] ^ switch], obj["anchor"], obj["scale"])
    return out


def validate_generated_puzzle(puzzle):
    _need(isinstance(puzzle, dict) and set(puzzle) == {"train", "test"}, "invalid puzzle schema")
    _need(all(isinstance(puzzle[s], list) for s in puzzle), "splits must be lists")
    _counts(len(puzzle["train"]), len(puzzle["test"]))
    seen, records = set(), {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in puzzle[split]:
            _need(isinstance(pair, dict) and set(pair) == {"input", "output"}, "invalid pair schema")
            inp, out = _grid(pair["input"]), _grid(pair["output"])
            signature = inp.shape, inp.tobytes()
            _need(signature not in seen, "duplicate inputs")
            seen.add(signature)
            a = _analyze(inp)
            _need(np.array_equal(a["output"], out), "label disagrees with input-only seam switching")
            objects = a["objects"]
            _need(4 <= len(objects) <= 6, "expected four to six ornaments")
            _need({o["state"] for o in objects if o["switch"]} == {0, 1}, "both switch directions required")
            _need({o["state"] for o in objects if not o["switch"]} == {0, 1}, "both unchanged states required")
            _need({o["scale"] for o in objects} <= {1, 2}, "unsupported scale")
            for mode in ("omit_exchange", "omit_key", "all"):
                _need(not np.array_equal(_counterfactual(a, mode), out), f"ineffective contribution: {mode}")
            records[split].append(a)
    first, second = records["train"][:2]
    _need({int(item[2]) for item in first["solution"][1:]} == {1, 2, 3},
          "first demonstration must expose all nonzero quarter turns")
    _need(all(o["scale"] == 1 for o in first["objects"]), "first demonstration must be unit scale")
    _need(any(o["owners"] == 3 for o in first["objects"]), "missing unit three-plate demonstration")
    _need(any(o["decoration_only"] for o in first["objects"]), "missing attachment-only crossing witness")
    _need(not np.array_equal(_counterfactual(first, "any_pixel"), first["output"]), "shortcut witness ineffective")
    _need(not np.array_equal(_counterfactual(first, "per_extra_plate"), first["output"]), "parity shortcut witness ineffective")
    _need(any(o["scale"] == 2 and o["owners"] == 2 for o in second["objects"]), "missing enlarged two-plate demonstration")
    for a in records["test"]:
        _need(any(o["scale"] == 2 and o["owners"] >= 3 for o in a["objects"]), "missing multi-plate test shift")
    return None


def _digest(puzzle):
    digest = hashlib.sha256()
    for split in ("train", "test"):
        for pair in puzzle[split]:
            for role in ("input", "output"):
                grid = np.asarray(pair[role], dtype=np.int8)
                digest.update(bytes(grid.shape))
                digest.update(grid.tobytes())
    return digest.hexdigest()


def self_test():
    """Stream 42 seeds, retain small feature sets only; no files or traces."""
    shapes, cuts, designs, palettes, counts, rotations, crossing = set(), set(), set(), set(), set(), set(), set()
    fingerprints = set()
    within_cuts = within_designs = 0
    for seed in range(42):
        p = generate_puzzle(seed)
        _need(_digest(p) == _digest(generate_puzzle(seed)), "nondeterministic generation")
        fingerprints.add(_digest(p))
        local_cuts, local_designs = set(), set()
        for split in ("train", "test"):
            for pair in p[split]:
                before = pair["input"].copy()
                a = _analyze(pair["input"])
                _need(np.array_equal(solve(pair["input"]), pair["output"]), "solver mismatch")
                _need(np.array_equal(before, pair["input"]), "solver mutated input")
                shapes.add(pair["input"].shape)
                cut = a["owner"].shape, a["owner"].tobytes()
                cuts.add(cut)
                local_cuts.add(cut)
                vocabulary = tuple(sorted(tuple(sorted(t)) for t in a["templates"]))
                designs.add(vocabulary)
                local_designs.add(vocabulary)
                palettes.add(tuple(int(c) for c in np.unique(pair["input"])))
                counts.add(len(a["objects"]))
                rotations.update(int(item[2]) for item in a["solution"])
                crossing.update(o["owners"] for o in a["objects"])
        within_cuts += len(local_cuts) > 1
        within_designs += len(local_designs) > 1
    _need(len(fingerprints) == 42, "seeds collapsed")
    _need(len(shapes) >= 20 and len(cuts) >= 150 and len(designs) >= 15, "insufficient structural variation")
    _need(len(palettes) >= 8 and counts >= {4, 5} and rotations == {0, 1, 2, 3}, "missing palette/count/motion variation")
    _need(crossing >= {1, 2, 3} and within_cuts == 42 and within_designs >= 30, "missing ownership/design variation")
    for nt, ns in ((1, 1), (11, 1), (4, 0), (6, 5), (2, 2), (3, 4), (True, 1), (4.0, 1), (4, None)):
        try:
            generate_puzzle(0, nt, ns)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid counts accepted")
    # Corrupt output cells, including a motif pixel and a substrate pixel.
    p = generate_puzzle(0, 2, 1)
    analysis = _analyze(p["test"][0]["input"])
    motif = analysis["objects"][0]
    common_point = next(iter(analysis["common"]))
    motif_point = _pixels([common_point], motif["anchor"], motif["scale"])[0]
    substrate_point = tuple(np.argwhere(analysis["output"] == analysis["substrate"])[0])
    for point in (motif_point, substrate_point):
        bad = {s: [{r: g.copy() for r, g in pair.items()} for pair in p[s]] for s in p}
        out = bad["test"][0]["output"]
        out[point] = (int(out[point]) + 1) % 10
        try:
            validate_generated_puzzle(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("corrupt label accepted")
    return {"seeds": 42, "distinct_puzzles": len(fingerprints), "input_shapes": len(shapes),
            "cuts": len(cuts), "support_vocabularies": len(designs), "palettes": len(palettes),
            "motif_counts": sorted(counts), "assembly_rotations": sorted(rotations),
            "scaffold_owner_counts": sorted(crossing), "within_puzzle_cut_variation": within_cuts,
            "within_puzzle_support_variation": within_designs}


if __name__ == "__main__":
    import json
    print(json.dumps(self_test(), separators=(",", ":")))
